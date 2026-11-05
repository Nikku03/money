"""Walk-forward replay: the trader learns through the past week by week, as if live.

Each decision day (every 5th session) it scores every stock from features.at, takes a paper
long-short book (long the TOP highest-scored, short the TOP lowest) if the predicted spread beats
the round-trip cost, and holds it 5 sessions. When the book closes, the outcomes of every stock it
scored (traded or not: it watched them all) update the learner, before the next decision. It never
sees an outcome before its trade has closed.

    cost       0.1% each way on each leg: 0.4% per week of long-short book traded
    P&L        weekly return of $1 long and $1 short, after costs; 0 in weeks it stays flat
    controls   shuffled: the same learner taught with each week's outcomes shuffled across stocks
               (any "skill" it then shows is chance); momentum: the textbook 12-1 momentum book,
               no learning; SPY and the equal-weighted universe, held (context only)

The pre-declared test (P1, docs/PAPER.md) is graded by ``grade``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from brain.paper import data, features
from brain.paper.learner import Learner

TOP = 5
COST = 0.001                 # per trade, each way
ROUND_TRIP = 4 * COST        # long leg in and out + short leg in and out
HORIZON = 5
WARMUP = 52                  # learner updates before the test period starts (about a year)


def run(px: pd.DataFrame | None = None, shuffle: bool = False, seed: int = 0, mode: str = "learn") -> dict:
    """``mode`` "learn" (the trader) or "momentum" (fixed 12-1 momentum book, no learning)."""
    stocks = data.universe()
    if px is None:
        px = data.closes(stocks + [m for m in data.MACRO if m != "SPY"])
    stocks = [s for s in stocks if s in px and px[s].notna().iloc[features.YEAR:].all()]
    rng = np.random.default_rng(seed)
    L = Learner(list(features.PRICE_FEATURES))
    rows = list(range(features.YEAR + 21, len(px) - HORIZON, HORIZON))
    weeks = []
    pending = None
    for i in rows:
        if pending is not None:                                  # the last book closed at this close: learn
            F0, y0 = pending
            if shuffle:
                y0 = rng.permutation(y0)
            L.update(F0, y0)
        F = features.at(px, i, stocks)
        y = features.forward(px, i, stocks, HORIZON).values
        pred = L.predict(F.values) if mode == "learn" else F["mom_12_1"].values
        order = np.argsort(pred)
        longs, shorts = order[-TOP:], order[:TOP]
        spread_hat = pred[longs].mean() - pred[shorts].mean()
        trade = (mode == "momentum") or (L.updates >= 1 and spread_hat > ROUND_TRIP)
        gross = y[longs].mean() - y[shorts].mean() if trade else 0.0
        weeks.append({"date": str(px.index[i]), "trade": bool(trade), "gross": float(gross),
                      "net": float(gross - ROUND_TRIP) if trade else 0.0, "spread_hat": float(spread_hat),
                      "updates": L.updates, "longs": [stocks[j] for j in longs] if trade else [],
                      "shorts": [stocks[j] for j in shorts] if trade else [],
                      "w": L.w.round(6).tolist()})
        pending = (F.values, y)
    spy = px["SPY"].iloc[rows].pct_change().shift(-1).fillna(0).values
    ew = (px[stocks].iloc[rows].pct_change().shift(-1).fillna(0).mean(axis=1).values)
    return {"mode": mode, "shuffled": shuffle, "stocks": stocks, "features": list(features.PRICE_FEATURES),
            "weeks": weeks, "spy_weekly": spy.tolist(), "ew_weekly": ew.tolist(), "final_w": L.w.tolist()}


def stats(net: list[float]) -> dict:
    x = np.asarray(net, float)
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return {"weeks": n, "mean": float(x.mean()) if n else 0.0, "t": 0.0, "p": 1.0, "ann": 0.0, "sharpe": 0.0,
                "total": 0.0, "max_dd": 0.0}
    t = x.mean() / (x.std(ddof=1) / math.sqrt(n))
    p = 0.5 * math.erfc(t / math.sqrt(2))                       # one-sided, normal approximation (n is large)
    curve = np.cumprod(1 + x)
    dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    return {"weeks": n, "mean": float(x.mean()), "t": float(t), "p": float(p), "ann": float((1 + x.mean()) ** 52 - 1),
            "sharpe": float(x.mean() / x.std(ddof=1) * math.sqrt(52)), "total": float(curve[-1] - 1), "max_dd": dd}


def test_weeks(r: dict) -> list[dict]:
    """The test period: weeks after the learner's first WARMUP updates."""
    return [w for w in r["weeks"] if w["updates"] >= WARMUP]


def grade(trader: dict, shuffled: dict) -> dict:
    """P1, as declared in docs/PAPER.md before the first run."""
    tw, sw = test_weeks(trader), test_weeks(shuffled)
    net = [w["net"] for w in tw]
    half = len(net) // 2
    s, s1, s2 = stats(net), stats(net[:half]), stats(net[half:])
    sh = stats([w["net"] for w in sw])
    checks = {"mean > 0 with one-sided p < 0.05": s["mean"] > 0 and s["p"] < 0.05,
              "beats the shuffled control": s["mean"] > sh["mean"],
              "mean > 0 in both halves": s1["mean"] > 0 and s2["mean"] > 0}
    return {"test": "P1", "from": tw[0]["date"] if tw else None, "to": tw[-1]["date"] if tw else None,
            "trader": s, "first_half": s1, "second_half": s2, "shuffled": sh,
            "traded_weeks": sum(w["trade"] for w in tw), "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL"}
