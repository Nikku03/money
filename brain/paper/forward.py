"""The forward test (F1): P5's machine, frozen, trading on paper from 2026-11-09, on a list chosen
before that date (docs/PAPER.md). Hindsight cannot reach it: each week's picks are written to the
ledger on that week's first evening, from data before that morning's open, and their outcomes
arrive later.

    step    after a close: read prices, SEC filings and financial reports; run the machine with
            ``live=True``; write every week from FORWARD_START not yet in the ledger (its picks, with
            their dossiers and reasons, or that it stayed out); then follow every open trade with
            its own exit rule through the sessions since, and write those that have closed.
    status  the record so far; under 30 closed weeks it says nothing, good or bad.

The ledger (research/paper/forward/ledger.jsonl) is append-only: a week once written is never
rewritten, even if later data would change its picks. Commit it after every step.

Paper only. Nothing here places an order.
"""
from __future__ import annotations

import json

import numpy as np

from brain import config
from brain.paper import dossier as ds
from brain.paper import moonshot as m1
from brain.paper import moonshot2 as m2
from brain.timeutil import now_iso

FORWARD_START = "2026-11-09"
DIR = config.ROOT / "research" / "paper" / "forward"
LEDGER = DIR / "ledger.jsonl"
UNIVERSE = config.ROOT / "universe" / "us_small.json"      # built 2026-10-12, before the forward start
MACRO = ("SPY", "USO", "TLT", "UUP", "^VIX")


def events() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()] if LEDGER.exists() else []


def _append(e: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"recorded_at": now_iso(), **e}, default=float, separators=(",", ":")) + "\n")


def inputs(stocks: list[str], log=print) -> tuple[dict, dict, dict]:
    """Prices (Yahoo), filings and SIC codes (SEC submissions), financial reports (SEC XBRL)."""
    from brain.sources import prices, sec
    raw, filings, sic, facts, names = {}, {}, {}, {}, {}
    for t in list(MACRO) + stocks:
        try:
            raw[t] = prices.fetch_daily(t, start="2019-06-01")[0]
        except Exception as e:
            log(f"  {t}: no prices ({type(e).__name__})")
    for t in stocks:
        try:
            f, ident = sec.filings_for(t, all_pages=True, ttl=86400)
            filings[t] = [(x.known_at, x.value, str(x.context.get("items", ""))) for x in f]
            sic[t] = next((i.value for i in ident if i.attr == "identity.sic"), None)
            data, fetched = sec._get_json(sec.FACTS_URL.format(cik=sec.cik_for(t)), ttl=7 * 86400)
            names[t] = data.get("entityName")
            facts[t] = [(x.attr, float(x.value), x.context.get("start"), x.as_of, x.known_at, x.context.get("form"),
                         x.context.get("duration_days")) for x in sec.parse_companyfacts(t, data, fetched)
                        if isinstance(x.value, (int, float))]
        except Exception as e:
            log(f"  {t}: no SEC data ({type(e).__name__})")
    return raw, {"filings": filings, "sic": sic}, {"facts": facts, "names": names}


def follow(P: dict, t: str, d: int, rule: str, sig: float) -> tuple[float | None, str | None]:
    """A trade's net and exit date under its own rule, if it has closed in the sessions so far."""
    prm = m2.rule_params(rule, np.array([sig]))
    need = d + prm["hold"]
    arr = []
    for k in ("open", "high", "low", "close"):
        a = P[k][[t]].values.astype(float)
        if len(a) < need:
            a = np.vstack([a, np.full((need - len(a), 1), np.nan)])
        arr.append(a)
    net, ex = m2.run_exit(*arr, d, prm)
    if not np.isfinite(net[0]):
        return None, None
    return float(net[0]), str(P["close"].index[int(ex[0])])


def step(log=print) -> dict:
    stocks = json.loads(UNIVERSE.read_text())["tickers"]
    raw, sec, fx = inputs(stocks, log)
    P = m1.panels(raw)
    dates = list(P["close"].index)
    dos = ds.Dossier(fx["facts"], fx["names"], P, m1.filing_events(sec, P["close"].index.values), sec["sic"])
    run = m2.simulate(raw, sec, stocks, dossier=dos, live=True, journal=False)
    ev = events()
    written = {e["week"] for e in ev if e["event"] in ("open", "flat")}
    new_weeks = [w for w in run["pending_weeks"] if w >= FORWARD_START and w not in written]
    opened = []
    for w in new_weeks:
        picks = [p for p in run["pending"] if p["date"] == w]
        if not picks:
            _append({"event": "flat", "week": w, "data_through": run["last_date"]})
        for p in picks:
            _append({"event": "open", "week": w, **p, "data_through": run["last_date"]})
            opened.append(p)
    closed = []
    done = {(e["week"], e["ticker"]) for e in ev if e["event"] == "close"}
    for e in events():
        if e["event"] != "open" or (e["week"], e["ticker"]) in done or e["week"] not in dates:
            continue
        net, when = follow(P, e["ticker"], dates.index(e["week"]), e["exit"], e["sig"])
        if net is not None:
            _append({"event": "close", "week": e["week"], "ticker": e["ticker"], "exit": e["exit"], "net": net, "closed": when})
            closed.append((e["ticker"], e["week"], net))
    return {"data_through": run["last_date"], "new_weeks": new_weeks, "opened": opened, "closed": closed}


def status() -> dict:
    ev = events()
    opens = [e for e in ev if e["event"] == "open"]
    closes = [e for e in ev if e["event"] == "close"]
    weeks = sorted({e["week"] for e in ev if e["event"] in ("open", "flat")})
    by_week = {}
    for c in closes:
        by_week.setdefault(c["week"], []).append(c["net"])
    done_weeks = [w for w in weeks if all((w, o["ticker"]) in {(c["week"], c["ticker"]) for c in closes}
                                          for o in opens if o["week"] == w)]
    weekly = [float(np.mean(by_week[w])) if w in by_week else 0.0 for w in done_weeks]
    return {"weeks": len(weeks), "closed_weeks": len(done_weeks), "trades": len(opens), "closed_trades": len(closes),
            "weekly": m1.tstat(weekly) if weekly else None,
            "open": [(e["week"], e["ticker"], e["exit"]) for e in opens
                     if (e["week"], e["ticker"]) not in {(c["week"], c["ticker"]) for c in closes}]}
