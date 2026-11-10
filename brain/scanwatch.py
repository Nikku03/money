"""Between scans: what has changed since the last recorded scan's close (docs/SCAN.md, "Between scans").

The scan reads every range from a close, and W1a-W1c tested it from closes. The market keeps moving in between:
news after the close, trading before the open, the session itself. ``python -m brain scan watch`` holds the latest
recorded scan against what has happened since:

- **each stock's latest trade** (Yahoo's five-minute chart, trading before the open and after the close included)
  against the close the scan read, and against the range the scan recorded for the five sessions;
- **the market:** SPY, QQQ and IWM, S&P 500 futures (``ES=F``, which trade through the night) and the VIX. A
  market-wide move is flagged: the replay's worst weeks were market-wide, when every range broke at once;
- **filings since that close** (SEC's per-company list), for each stock outside its range and each of the scan's 25
  likeliest 20%+ days: a results report, a share sale, other 8-Ks.

It forecasts nothing new. A flag says where a range is already used up or where news has arrived; the next close's
scan re-reads every range from the new facts. Paper only: nothing here trades.
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote as urlquote

from brain import scan, shocks
from brain.bank import Bank, canon
from brain.net import fetch
from brain.timeutil import now_iso

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=2d&interval=5m&includePrePost=true"
MARKET = (("SPY", "S&P 500 (SPY)"), ("QQQ", "Nasdaq 100 (QQQ)"), ("IWM", "Russell 2000 (IWM)"),
          ("ES=F", "S&P 500 futures"), ("^VIX", "VIX"))
MARKET_MOVE = 0.015         # SPY or the futures this far from the scan's close: a market-wide move
VIX_JUMP = 0.15             # the VIX up this much: fear rising
TTL = 120                   # seconds a fetched chart is reused
TOP_SHOCKS = 25             # the scan's likeliest 20%+ days, whose filings are always read
BAD_MOVE = 1.5              # a log move beyond this since the close is a split or a bad print, not a move
FILING_FORMS = ("8-K", "8-K/A", *shocks.OFFER_FORMS)


# ---------- reading ----------
def latest_scan() -> tuple[Path, dict] | None:
    files = sorted(scan.scans_dir().glob("????-??-??.json"))
    return (files[-1], json.loads(files[-1].read_text())) if files else None


def session_of(ts: float, periods: dict) -> str:
    """Which part of the trading day a bar's time falls in, by the chart's current trading periods."""
    pre, reg, post = (periods.get(k) or {} for k in ("pre", "regular", "post"))
    if reg and reg["start"] <= ts < reg["end"]:
        return "in session"
    if pre and pre["start"] <= ts < pre["end"]:
        return "before the open"
    if (post and post["start"] <= ts < post["end"]) or (reg and ts >= reg["end"]):
        return "after the close"
    return "overnight"


def parse_quote(body: bytes) -> dict | None:
    """The latest trade in a five-minute chart: its price, its time and which part of the day it falls in."""
    res = json.loads(body)["chart"]["result"][0]
    ts = res.get("timestamp") or []
    closes = ((res.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    last = next(((t, c) for t, c in zip(reversed(ts), reversed(closes)) if c is not None and c > 0), None)
    if last is None:
        return None
    periods = (res.get("meta") or {}).get("currentTradingPeriod") or {}
    return {"price": float(last[1]), "at": datetime.fromtimestamp(last[0], UTC).strftime("%Y-%m-%dT%H:%MZ"),
            "session": session_of(last[0], periods)}


def quote(symbol: str, ttl: float = TTL) -> dict | None:
    try:
        return parse_quote(fetch(CHART.format(sym=urlquote(symbol)), ttl=ttl).body)
    except Exception:  # noqa: BLE001 - one symbol's outage must not stop the sweep
        return None


def close_on(view, t: str, day: str) -> float | None:
    """The stock's close on ``day`` as the bank holds it (split-adjusted, like the live price)."""
    df = shocks._series(view, t)
    if df is None:
        return None
    d = df[df["date"].astype(str) == day]
    return float(d["close"].iloc[-1]) if len(d) else None


def market_close_on(symbol: str, day: str) -> float | None:
    """A market symbol's close on ``day`` from its daily chart (they are not all in the bank)."""
    from brain.sources import prices
    try:
        df = prices.fetch_daily(symbol, start=(datetime.fromisoformat(day) - timedelta(days=10)).date().isoformat(),
                                ttl=3600)[0]
    except Exception:  # noqa: BLE001
        return None
    d = df[df["date"].astype(str) <= day]
    return float(d["close"].iloc[-1]) if len(d) else None


def since_close(filings: list, asof: str) -> list:
    """Filings first public after the close of ``asof`` (New York time)."""
    from brain import earnradar
    out = []
    for f in filings:
        day = earnradar.ny_date(f.known_at).isoformat()
        if day > asof or (day == asof and earnradar.timing(f.known_at) == "after close"):
            out.append(f)
    return out


def describe(f) -> str:
    form = str(f.value).upper()
    if form.startswith("8-K"):
        items = [x.strip() for x in str((f.context or {}).get("items") or "").split(",") if x.strip()]
        if "2.02" in items:
            return "results report (8-K 2.02)"
        return "8-K " + ", ".join(i for i in items if i not in ("9.01",)) if items else "8-K"
    return f"share-sale filing ({form})"


def filings_since(t: str, asof: str) -> list[str]:
    from brain.sources import sec
    try:
        facts, _ = sec.filings_for(t, all_pages=False, ttl=600)
    except Exception:  # noqa: BLE001
        return []
    new = [f for f in facts if f.attr == "filing" and str(f.value).upper().startswith(FILING_FORMS)]
    seen, out = set(), []
    for f in sorted(since_close(new, asof), key=lambda f: f.known_at):
        key = (str(f.value), (f.context or {}).get("accession") or f.known_at)
        if key not in seen:
            seen.add(key)
            out.append(f"{describe(f)} at {f.known_at[:16]}Z")
    return out


# ---------- judging ----------
def judge(move: float, rng: list[float]) -> dict:
    """A move since the scan's close (log) against the scan's recorded five-session range (log returns)."""
    lo90, lo68, hi68, hi90 = rng
    edge = hi68 if move >= 0 else -lo68
    return {"outside68": bool(move < lo68 or move > hi68), "outside90": bool(move < lo90 or move > hi90),
            "used": float(abs(move) / edge) if edge > 0 else float("nan")}


def watch(bank: Bank, tickers: list[str] | None = None, top: int | None = None, log=None) -> dict:
    """The latest recorded scan against everything since its close."""
    got = latest_scan()
    if not got:
        raise SystemExit("no recorded scan yet: python -m brain scan run --record")
    path, rec = got
    asof, rows = rec["asof"], rec["rows"]
    names = [t.upper() for t in tickers] if tickers else list(rows)
    if top and not tickers:
        half = {t: math.expm1(r[2]) - math.expm1(r[1]) for t, r in rows.items()}
        shock = {t: r[4] or 0.0 for t, r in rows.items()}
        names = sorted(set(sorted(rows, key=lambda t: -half[t])[:top]) | set(sorted(rows, key=lambda t: -shock[t])[:top]))
    market = []
    for sym, name in MARKET:
        q, c = quote(sym), market_close_on(sym, asof)
        if q and c:
            market.append({"symbol": sym, "name": name, "move": q["price"] / c - 1, "at": q["at"], "session": q["session"]})
    view = bank.view()
    likeliest = set(sorted(rows, key=lambda t: -(rows[t][4] or 0.0))[:TOP_SHOCKS])
    checked, flags = 0, []
    for k, t in enumerate(names):
        if t not in rows:
            continue
        q, c = quote(t), close_on(view, t, asof)
        if not (q and c):
            continue
        move = math.log(q["price"] / c)
        if abs(move) > BAD_MOVE:
            continue
        checked += 1
        j = judge(move, rows[t][:4])
        if j["outside68"] or t in likeliest:
            flags.append({"t": t, "move": math.expm1(move), "at": q["at"], "session": q["session"], **j,
                          "range68": [math.expm1(rows[t][1]), math.expm1(rows[t][2])],
                          "range90": [math.expm1(rows[t][0]), math.expm1(rows[t][3])],
                          "shock": rows[t][4], "likeliest": t in likeliest})
        if log and (k + 1) % 200 == 0:
            log(f"  {k + 1}/{len(names)} checked")
    for f in flags:
        if f["outside68"] or f["likeliest"]:
            f["filings"] = filings_since(f["t"], asof)
    flags.sort(key=lambda f: (-f["outside90"], -f["outside68"], -(f["used"] if f["used"] == f["used"] else 0)))
    spy = next((m for m in market if m["symbol"] in ("ES=F", "SPY") and abs(m["move"]) >= MARKET_MOVE), None)
    vix = next((m for m in market if m["symbol"] == "^VIX"), None)
    return {"asof": asof, "scan": path.name, "at": now_iso(), "market": market, "checked": checked,
            "listed": len(rows), "outside68": sum(f["outside68"] for f in flags),
            "outside90": sum(f["outside90"] for f in flags),
            "market_wide": bool(spy), "fear_up": bool(vix and vix["move"] >= VIX_JUMP), "flags": flags}


def updates_dir(asof: str) -> Path:
    return scan.scans_dir() / "updates" / asof


def record(res: dict) -> Path:
    p = updates_dir(res["asof"]) / f"{res['at'][:16].replace(':', '')}Z.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(canon(res) + "\n")
    return p


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:+.{d}f}%"


def show(res: dict, top: int = 25) -> None:
    print(f"Between scans: what has changed since the close of {res['asof']} (the recorded scan of {res['listed']} "
          f"stocks), read at {res['at'][:16]}Z.")
    if res["market"]:
        print("  Market: " + ", ".join(f"{m['name']} {_pct(m['move'])}" for m in res["market"])
              + f" ({res['market'][0]['session']})")
    if res["market_wide"]:
        print("  A market-wide move: every range is likely too narrow until the next scan. In the replay the ranges' "
              "worst weeks were market-wide.")
    if res["fear_up"]:
        print("  Fear is rising (the VIX is up 15% or more): the next scan's ranges will widen.")
    n = max(res["checked"], 1)
    print(f"  {res['checked']} stocks checked: {res['outside90']} already outside their 9-in-10 range "
          f"({res['outside90'] / n * 100:.1f}%), {res['outside68']} outside their 2-in-3 range "
          f"({res['outside68'] / n * 100:.1f}%). Over a full five sessions about 10% and 32% end outside.")
    shown = [f for f in res["flags"] if f["outside68"]][:top]
    for f in shown:
        tag = "OUTSIDE 9-in-10" if f["outside90"] else "outside 2-in-3"
        news = "; ".join(f.get("filings") or []) or "no filing since the close"
        print(f"  {f['t']:6} {_pct(f['move']):>8} since the close ({f['session']})  2 in 3: {_pct(f['range68'][0], 0)}.."
              f"{_pct(f['range68'][1], 0)}  9 in 10: {_pct(f['range90'][0], 0)}..{_pct(f['range90'][1], 0)}  {tag}  "
              f"news: {news}")
    likely = [f for f in res["flags"] if f["likeliest"] and not f["outside68"] and f.get("filings")]
    if likely:
        print("  Among the scan's likeliest 20%+ days, new filings: " + "; ".join(
            f"{f['t']} {_pct(f['move'])} ({', '.join(f['filings'])})" for f in likely))
    print("A flag says where to look, not which way to trade. The next close's scan re-reads every range.")


MIN_EVERY = 15              # minutes: the shortest gap between the starts of two checks in a loop


def cmd(a, sleep=None, cycles: int | None = None) -> int:
    """One check, or with ``--every`` a check every so many minutes until Ctrl+C (``cycles`` caps it in tests)."""
    import time
    sleep = sleep or time.sleep
    every = max(MIN_EVERY, int(a.every)) * 60 if a.every else None
    n = 0
    try:
        while True:
            start = time.monotonic()
            res = watch(Bank(), a.tickers or None, top=a.only_top, log=None if every else lambda m: print(m, flush=True))
            show(res)
            if a.record:
                print(f"recorded {record(res)}", flush=True)
            n += 1
            if not every or (cycles and n >= cycles):
                return 0
            sleep(max(0.0, every - (time.monotonic() - start)))
    except KeyboardInterrupt:
        return 0
