"""The fifth forward test (F5): P12 (P11's lessons plus the insider layer) on paper from the week of
2026-11-09, beside F1-F4 (docs/PAPER.md), with every live-only source recorded beside each pick.

Same list (universe/us_small.json), same ledger rules (research/paper/forward5/ledger.jsonl,
append-only, each week written on its first evening from data before that morning's open), the same
following of exits and the same memory bank as F1-F4.

**Its insider layer** comes from research/paper/layers/insiders.jsonl: the open-market trades of the
list's companies since 2024, from SEC's Form 4/5 data sets, with no insider names or IDs. Each run adds
the Form 4s filed since the file's last date (``update_insiders``); commit that file with the ledger.

**Recorded beside each pick** (``context``), with no weight until a forward record tests them, since
none has free history:
- **options:** the at-the-money straddle's priced move to the first expiry after the entry (Yahoo
  option chains);
- **headlines:** the latest three published before that morning's open (Yahoo RSS);
- **guidance:** the event tags of an earnings release filed in the 30 days before the open (SEC 8-K
  exhibit 99, read by the repo's fixed rules);
- **short selling:** the share of FINRA-reported volume sold short over the last 20 sessions. It
  flipped sign between the review's two periods, so it carries no weight.

Alternative data (company aircraft, port traffic, traffic cameras) maps to none of these companies,
and is not recorded.

Paper only. Nothing here places an order.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd

from brain import config
from brain.paper import forward as f1
from brain.paper import memory as mem
from brain.paper import moonshot as m1
from brain.paper import moonshot6 as m6
from brain.timeutil import now_iso

FORWARD_START = f1.FORWARD_START
DIR = config.ROOT / "research" / "paper" / "forward5"
LEDGER = DIR / "ledger.jsonl"
CACHE = config.ROOT / "research" / "paper" / "layers" / "insiders.jsonl"
SOURCE = "F5 forward (P12, every layer wired in)"
FINRA = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"


def events() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()] if LEDGER.exists() else []


def _append(e: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"recorded_at": now_iso(), **e}, default=float, separators=(",", ":")) + "\n")


# --- the insider layer ------------------------------------------------------------------------------------
def load_insiders() -> dict:
    rows = {}
    if CACHE.exists():
        for x in CACHE.read_text().splitlines():
            if x.strip():
                t, filed, day, code, shares, price = json.loads(x)
                rows.setdefault(t, []).append((filed, day, code, shares, price, "", ""))
    return rows


def update_insiders(stocks: list[str], log=print) -> int:
    """Add the Form 4s filed since the cache's last filing date (their open-market trades only)."""
    from brain.net import fetch
    from brain.sources import insiders as ins
    from brain.sources import sec
    rows = load_insiders()
    last = max((r[0] for v in rows.values() for r in v), default="2024-01-01")
    have = {(t, *r[:5]) for t, v in rows.items() for r in v}
    new = []
    for t in stocks:
        try:
            fil, _ = sec.filings_for(t, all_pages=False, ttl=86400)
        except Exception as e:
            log(f"  {t}: no filings ({type(e).__name__})")
            continue
        for f in fil:
            url = ins.xml_url(f) if f.value == "4" and f.known_at[:10] > last else None
            if not url:
                continue
            try:
                r = fetch(url, user_agent=config.sec_user_agent(), ttl=365 * 86400)
                for g in ins.parse_form4(r.body, t, f.context.get("accn"), f.known_at, url):
                    row = (t, g.known_at[:10], g.as_of, g.context["code"], float(g.value), g.context.get("price"))
                    if row not in have:
                        have.add(row)
                        new.append(row)
            except Exception as e:
                log(f"  {t}: a Form 4 unreadable ({type(e).__name__})")
    if new:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a") as fh:
            for row in sorted(new, key=lambda r: (r[0], r[1], r[2] or "")):
                fh.write(json.dumps(list(row), separators=(",", ":")) + "\n")
    return len(new)


# --- what it records beside each pick ---------------------------------------------------------------------
def short_share(dates: list[str], tickers: list[str], n: int = 20) -> dict:
    import urllib.request
    tot = {t: [0.0, 0.0] for t in tickers}
    for d in dates[-n:]:
        try:
            req = urllib.request.Request(FINRA.format(ymd=d.replace("-", "")), headers={"User-Agent": "money-brain research"})
            txt = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace").splitlines()
        except Exception:
            continue
        head = txt[0].split("|") if txt else []
        if not {"Symbol", "ShortVolume", "TotalVolume"} <= set(head):
            continue
        iy, i_s, iv = head.index("Symbol"), head.index("ShortVolume"), head.index("TotalVolume")
        for line in txt[1:]:
            p = line.split("|")
            t = p[iy].replace(".", "-") if len(p) > iv else None
            if t in tot:
                tot[t][0] += float(p[i_s] or 0)
                tot[t][1] += float(p[iv] or 0)
    return {t: (s / v if v else None) for t, (s, v) in tot.items()}


def context(ticker: str, week: str, opened_at: str) -> dict:
    """Options' priced move, headlines and guidance tags for one pick, as known before ``opened_at``."""
    out = {"options_priced_move_pct": None, "options_expiry": None, "headlines": [], "guidance": []}
    try:
        from brain.sources import options
        ch = options.fetch_chain(ticker)
        exp = options.first_expiry(ch, week)
        s = options.straddle(ch, exp) if exp else None
        if s:
            out["options_priced_move_pct"], out["options_expiry"] = s["implied_pct"], s["expiry"]
    except Exception:
        pass
    try:
        from brain.sources import news
        hs = sorted((h for h in news.ingest(ticker) if h.known_at < opened_at), key=lambda h: h.known_at)[-3:]
        out["headlines"] = [{"at": h.known_at, "title": h.value, "url": h.source.get("url")} for h in hs]
    except Exception:
        pass
    try:
        from brain import events as ev
        from brain.sources import releases, sec
        fil, _ = sec.filings_for(ticker, all_pages=False, ttl=86400)
        since = (datetime.fromisoformat(week) - timedelta(days=30)).strftime("%Y-%m-%d")
        recent = [f for f in fil if ev.is_earnings(f) and since <= f.known_at[:10] and f.known_at < opened_at]
        for f in sorted(recent, key=lambda f: f.known_at)[-1:]:
            r = releases.read_release(f)
            out["guidance"] = r.context.get("events") or []
            out["release"] = {"at": f.known_at, "title": r.value, "url": r.source.get("url")}
    except Exception:
        pass
    return out


# --- the step ----------------------------------------------------------------------------------------------
def step(log=print) -> dict:
    stocks = json.loads(f1.UNIVERSE.read_text())["tickers"]
    raw, sec, fx = f1.inputs(stocks, log, price_start="2010-06-01")
    added = update_insiders(stocks, log)
    P = m1.panels(raw)
    dates = [str(x) for x in P["close"].index]
    wk = pd.Series(pd.to_datetime(dates)).dt.to_period("W").values
    week_firsts = [dates[i] for i in range(1, len(dates)) if wk[i] != wk[i - 1] and dates[i] >= FORWARD_START]
    ev = events()
    written = {e["week"] for e in ev if e["event"] in ("open", "flat")}
    todo = [w for w in week_firsts if w not in written]
    opened, new_weeks, mb = [], [], mem.bank()
    if todo:
        run = m6.simulate(raw, sec, fx, stocks, start=FORWARD_START, layers={"insiders": load_insiders(), "short": {}},
                          live=True, journal=False)
        shorts = None
        for pw in run["pending_weeks"]:
            w = pw["date"]
            if w not in todo:
                continue
            new_weeks.append(w)
            picks = [p for p in run["pending"] if p["date"] == w]
            gate = {"hunt": pw["hunt"], "set_aside": pw["set_aside"]}
            if not picks:
                _append({"event": "flat", "week": w, **gate, "data_through": run["last_date"]})
            if picks and shorts is None:
                shorts = short_share([d for d in dates if d < w], [p["ticker"] for p in run["pending"]])
            for p in picks:
                opened_at = f"{w}T13:30:00Z"                   # before the open, in either season
                p["memory"] = mem.card(mem.recall(mb, p["ticker"], opened_at))
                p["context"] = {**context(p["ticker"], w, opened_at), "short_share_20d": shorts.get(p["ticker"])}
                _append({"event": "open", "week": w, **p, **gate, "data_through": run["last_date"]})
                opened.append(p)
    closed = []
    done = {(e["week"], e["ticker"]) for e in ev if e["event"] == "close"}
    for e in events():
        if e["event"] != "open" or (e["week"], e["ticker"]) in done or e["week"] not in dates:
            continue
        net, when = f1.follow(P, e["ticker"], dates.index(e["week"]), e["exit"], e["sig"])
        if net is not None:
            _append({"event": "close", "week": e["week"], "ticker": e["ticker"], "exit": e["exit"], "net": net, "closed": when})
            closed.append((e["ticker"], e["week"], net))
            mem.remember(mb, [mem.trade_fact({"date": e["week"], "closed": when, "ticker": e["ticker"], "exit": e["exit"],
                                              "net": net, "lesson": f"{e['ticker']} {net:+.1%} with {e['exit']} (F5, week of {e['week']})"},
                                             SOURCE)])
    return {"data_through": dates[-1], "new_weeks": new_weeks, "opened": opened, "closed": closed, "insider_trades_added": added}


def status() -> dict:
    saved = f1.LEDGER
    try:
        f1.LEDGER = LEDGER
        return f1.status()
    finally:
        f1.LEDGER = saved
