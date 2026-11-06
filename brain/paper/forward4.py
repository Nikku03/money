"""The fourth forward test (F4): the P10 machine (it learns how the market works first, then trades)
on paper from the week of 2026-11-09, beside F1-F3 (docs/PAPER.md). Same list as F1
(universe/us_small.json, built before the start), same ledger rules (research/paper/forward4/
ledger.jsonl, append-only, each week written on its first evening from data before that morning's
open), the same following of exits, and the same memory bank, which also keeps each week's market
lesson (memory.market) from the open after the week.

P10 studies everything back to 2011 before deciding, so the machine runs only on evenings when a new
week has started that the ledger does not hold yet. Other evenings, it only follows open trades.

Paper only. Nothing here places an order.
"""
from __future__ import annotations

import json

import pandas as pd

from brain import config
from brain.paper import forward as f1
from brain.paper import market as mk
from brain.paper import memory as mem
from brain.paper import moonshot as m1
from brain.paper import moonshot4 as m4
from brain.timeutil import now_iso

FORWARD_START = f1.FORWARD_START
DIR = config.ROOT / "research" / "paper" / "forward4"
LEDGER = DIR / "ledger.jsonl"
SOURCE = "F4 forward (P10, the market model)"


def events() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()] if LEDGER.exists() else []


def _append(e: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"recorded_at": now_iso(), **e}, default=float, separators=(",", ":")) + "\n")


def inputs(stocks: list[str], log=print):
    from brain.sources import prices
    raw, sec, fx = f1.inputs(stocks, log, price_start="2010-06-01")
    for a in mk.NEEDED:
        if a not in raw:
            raw[a] = prices.fetch_daily(a, start="2010-06-01")[0]
    return raw, sec, fx


def step(log=print) -> dict:
    stocks = json.loads(f1.UNIVERSE.read_text())["tickers"]
    raw, sec, fx = inputs(stocks, log)
    P = m1.panels(raw)
    dates = [str(x) for x in P["close"].index]
    wk = pd.Series(pd.to_datetime(dates)).dt.to_period("W").values
    week_firsts = [dates[i] for i in range(1, len(dates)) if wk[i] != wk[i - 1] and dates[i] >= FORWARD_START]
    ev = events()
    written = {e["week"] for e in ev if e["event"] in ("open", "flat")}
    todo = [w for w in week_firsts if w not in written]
    opened, new_weeks, mb = [], [], mem.bank()
    if todo:
        run = m4.simulate(raw, sec, fx, stocks, start=FORWARD_START, live=True, journal=False)
        known = {f.as_of for f in mb.view(now_iso()).facts(mem.MARKET, "memory.market")}
        mem.remember(mb, [mem.market_fact(les, SOURCE) for les in run["market"]
                          if les["known"] >= FORWARD_START and les["week"] not in known])
        for pw in run["pending_weeks"]:
            w = pw["date"]
            if w not in todo:
                continue
            new_weeks.append(w)
            picks = [p for p in run["pending"] if p["date"] == w]
            gate = {k: pw[k] for k in ("gate", "gate_mean", "gate_t")}
            if not picks:
                _append({"event": "flat", "week": w, **gate, "market": pw.get("market"), "data_through": run["last_date"]})
            for p in picks:
                p["memory"] = mem.card(mem.recall(mb, p["ticker"], f"{w}T13:00:00Z"))
                _append({"event": "open", "week": w, **p, **gate, "market": pw.get("market"), "data_through": run["last_date"]})
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
                                              "net": net, "predicted": e.get("predicted"), "reasons": e.get("reasons"),
                                              "lesson": f"{e['ticker']} {net:+.1%} with {e['exit']} (F4, week of {e['week']})"},
                                             SOURCE)])
    return {"data_through": dates[-1], "new_weeks": new_weeks, "opened": opened, "closed": closed}


def status() -> dict:
    saved = f1.LEDGER
    try:
        f1.LEDGER = LEDGER
        return f1.status()
    finally:
        f1.LEDGER = saved
