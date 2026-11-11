"""Why a stock moved: the market, its industry, a linked driver, or its own news (docs/SCAN.md, "Why it moved").

Prices move on more than their own charts: on the market, on their industry, on what drives their industry (oil for
oil producers and airlines, interest rates for banks, homebuilders and tech) and on their own news. A check between
scans (``brain/scanwatch.py``) splits each flagged stock's move since the last close into those parts, so a reader
can see which one it was:

- **the market:** SPY's move since the close;
- **its industry:** the median move of the other checked stocks with its SEC industry code (four digits; else
  three, else two, with at least MIN_PEERS of them);
- **its own part:** its move beyond that median (beyond the market's when it has no industry);
- **drivers:** oil, the 10-year Treasury yield, the dollar, gold and bitcoin since their last close, and for the
  industries the chain engine links to one of them, the link as recorded (``knowledge/links.json``; L1 in
  ``research/RESULTS.md``);
- **its own news:** SEC filings and Yahoo Finance headlines since the close;
- **the day's scheduled macro releases** (``knowledge/macro_calendar.json``).

An industry whose median moved SECTOR_MOVE or more is flagged as moving together: its stocks' ranges are likely too
narrow until the next scan, as with a market-wide move.

It explains; it does not forecast. In this repo's tests the chain engine's links moved stocks in the same month and
said nothing about the next (L1), and results-day news moved stocks on the day with no drift after (E1, E2).
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np

from brain import config

MIN_PEERS = 8               # other checked stocks an industry needs before its median counts
SECTOR_MOVE = 0.03          # an industry's median move since the close that counts as moving together
NEWS_SHARE = 0.5            # a stock with news whose own part is at least this share of its move: its own news
INDUSTRY_MOVE = 0.02        # an industry median this large can carry a stock's move
MARKET_MOVE = 0.01          # a market move this large can carry a stock's move
HEADLINES = 2               # headlines kept for each flagged stock
DRIVERS = (("CL=F", "oil", "pct"), ("^TNX", "10-year yield", "points"), ("DX-Y.NYB", "dollar", "pct"),
           ("GC=F", "gold", "pct"), ("BTC-USD", "bitcoin", "pct"))
# industries the chain engine's links reach: SEC codes (four digits, or a three-digit prefix), the driver, the test
LINKS = (("oil producers", ("1311", "1381", "1382", "1389", "2911"), "CL=F", "L1-oil-producers"),
         ("airlines", ("4512", "4513", "4522"), "CL=F", "L1-oil-airlines"),
         ("banks", ("6021", "6022", "6029", "6035", "6036"), "^TNX", "L1-rates-banks"),
         ("homebuilders", ("1520", "1521", "1531"), "^TNX", "L1-rates-homebuilders"),
         ("tech", ("357", "367", "737"), "^TNX", "L1-rates-tech"))
MACRO = (("fomc_decision", "the Fed's rate decision"), ("cpi_release", "the CPI report"),
         ("jobs_release", "the jobs report"))
NY = ZoneInfo("America/New_York")


# ---------- industries ----------
def industry_map(view, tickers) -> dict[str, dict]:
    """Each ticker's SEC industry code and its description, as the bank holds them."""
    out = {}
    for t in tickers:
        f = view.latest(t, "identity.sic")
        code = str(f.value).strip() if f is not None and f.value not in (None, "") else ""
        if len(code) == 4 and code.isdigit():
            g = view.latest(t, "identity.industry")
            out[t] = {"sic": code, "name": str(g.value) if g is not None and g.value else None}
    return out


def industry_context(moves: dict[str, float], ind: dict[str, dict], min_peers: int = MIN_PEERS) -> dict[str, dict]:
    """Each stock's industry move: the median of the other checked stocks with its code, at the finest level (4, 3
    or 2 digits) that has at least ``min_peers`` of them."""
    groups: dict[tuple[int, str], list[str]] = {}
    for t, i in ind.items():
        if t in moves:
            for lv in (4, 3, 2):
                groups.setdefault((lv, i["sic"][:lv]), []).append(t)
    out = {}
    for t, i in ind.items():
        if t not in moves:
            continue
        for lv in (4, 3, 2):
            peers = [p for p in groups[(lv, i["sic"][:lv])] if p != t]
            if len(peers) >= min_peers:
                names = Counter(ind[p]["name"] for p in peers if ind[p]["name"])
                name = i["name"] if lv == 4 and i["name"] else (names.most_common(1)[0][0] if names else None)
                out[t] = {"level": lv, "code": i["sic"][:lv], "name": name, "n": len(peers),
                          "median": float(np.median([moves[p] for p in peers]))}
                break
    return out


def sector_moves(moves: dict[str, float], ind: dict[str, dict], min_peers: int = MIN_PEERS,
                 threshold: float = SECTOR_MOVE) -> list[dict]:
    """Industries (four-digit codes with at least ``min_peers`` checked stocks) whose median moved ``threshold`` or
    more either way since the close, the largest first."""
    groups: dict[str, list[str]] = {}
    for t, i in ind.items():
        if t in moves:
            groups.setdefault(i["sic"], []).append(t)
    out = []
    for code, ts in groups.items():
        if len(ts) >= min_peers:
            med = float(np.median([moves[t] for t in ts]))
            if abs(med) >= threshold:
                names = Counter(ind[t]["name"] for t in ts if ind[t]["name"])
                out.append({"code": code, "name": names.most_common(1)[0][0] if names else f"SEC industry {code}",
                            "n": len(ts), "median": med})
    return sorted(out, key=lambda x: -abs(x["median"]))


# ---------- drivers and the chain engine's links ----------
def drivers(asof: str, quote=None, close_on=None) -> list[dict]:
    """Each driver's move since its close on ``asof``: a percentage, or percentage points for the 10-year yield."""
    from brain import scanwatch
    quote, close_on = quote or scanwatch.quote, close_on or scanwatch.market_close_on
    def read(sym: str):
        try:
            return quote(sym), close_on(sym, asof)
        except Exception:  # noqa: BLE001 - one driver's outage must not stop the check
            return None, None

    out = []
    for sym, name, unit in DRIVERS:
        q, c = read(sym)
        if q and c:
            move = q["price"] - c if unit == "points" else q["price"] / c - 1
            out.append({"symbol": sym, "name": name, "unit": unit, "move": float(move)})
    return out


def link_for(sic: str | None):
    """The chain engine's link that reaches an industry code, if any."""
    if not sic:
        return None
    return next((lk for lk in LINKS if any(sic == c or (len(c) == 3 and sic.startswith(c)) for c in lk[1])), None)


def link_records() -> dict[str, dict]:
    """The L1 link tests as the ledger records them: the same month's effect per typical move of the driver, and
    the verdicts of the same-month and next-month tests."""
    try:
        from brain.research import ledger, run
        recs = ledger.read(config.LEDGER)
        verdicts = {r["id"]: r for r in run.verdicts() if r.get("kind") == "link"}
    except Exception:  # noqa: BLE001 - no ledger: no links quoted
        return {}
    specs = {r["id"]: r.get("spec") or {} for r in recs if r.get("op") == "declare" and r.get("kind") == "link"}
    out = {}
    for base in {lk[3] for lk in LINKS}:
        same, nxt = verdicts.get(f"{base}-same"), verdicts.get(f"{base}-next")
        if not same:
            continue
        measure = (specs.get(f"{base}-same") or {}).get("cause_measure") or {}
        out[base] = {"per_move": float(same["metrics"]["test"]["per_move"]), "move_label": measure.get("move_label"),
                     "same": same["verdict"].split(":")[0], "next": nxt["verdict"].split(":")[0] if nxt else None}
    return out


def link_note(info: dict | None, drv: list[dict], records: dict[str, dict]) -> str | None:
    """A flagged stock's chain link in words: the driver's move today and the link as recorded."""
    lk = link_for(info["sic"] if info else None)
    if not lk:
        return None
    name, _, sym, base = lk
    d = next((x for x in drv if x["symbol"] == sym), None)
    rec = records.get(base)
    today = (f"{d['name']} {d['move']:+.2f} pt" if d and d["unit"] == "points" else
             f"{d['name']} {d['move'] * 100:+.1f}%" if d else None)
    parts = [f"chain: {name}"]
    if today:
        parts.append(f"{today} since the close")
    if rec:
        parts.append(f"recorded link {rec['per_move'] * 100:+.2f}% per {rec['move_label']} in the same month "
                     f"({rec['same']}); the next month {rec['next'] or 'untested'}")
    return "; ".join(parts)


# ---------- the day and the news ----------
def macro_today(day: str, path=None) -> list[str]:
    """Scheduled macro releases on ``day`` (New York)."""
    try:
        cal = json.loads((path or config.ROOT / "knowledge" / "macro_calendar.json").read_text())
    except (OSError, ValueError):
        return []
    return [label for key, label in MACRO if day in set(cal.get(key) or [])]


def close_utc(day: str) -> str:
    """The 16:00 New York close of ``day``, in UTC."""
    t = datetime.fromisoformat(f"{day}T16:00:00").replace(tzinfo=NY).astimezone(UTC)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def headlines_since(t: str, since: str, ingest=None) -> list[str]:
    """Yahoo Finance headlines for ``t`` published after ``since`` (UTC), newest first."""
    if ingest is None:
        from brain.sources import news
        ingest = news.ingest
    try:
        facts = ingest(t)
    except Exception:  # noqa: BLE001 - one feed's outage must not stop the check
        return []
    new = sorted((f for f in facts if f.known_at > since), key=lambda f: f.known_at, reverse=True)
    return [f"{f.value} ({(f.source or {}).get('name', '?')}, {f.known_at[:16]}Z)" for f in new[:HEADLINES]]


def attribute(move: float, market: float, industry: float | None, news: bool) -> str:
    """Which part carried a move, in words: its own news, its industry, the market, or its own with no news found."""
    own = move - (industry if industry is not None else market)
    if news and abs(own) >= NEWS_SHARE * abs(move):
        return "its own news"
    if industry is not None and abs(industry) >= INDUSTRY_MOVE and abs(own) < abs(industry):
        return "its industry"
    if abs(market) >= MARKET_MOVE and abs(move - market) < abs(market):
        return "the market"
    return "its own news" if news else "its own move, no news found"
