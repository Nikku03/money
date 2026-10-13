"""Backtest and matched null.

Portfolio: each rebalance day, hold some listed tickers by signal, equal weight.
Benchmark: all listed tickers, equal weight (same universe, same timing), so a
universe-wide drift, survivorship included, cancels.

Which tickers (``select``, set per hypothesis):
- ``top_frac`` (default): the top ``frac`` of tickers that have a signal value. The null
  shuffles the signal across those same tickers.
- ``top_frac_vs_all``: the same holdings; the null draws as many names from every listed
  ticker, so "has a value at all" (say, an event this month) is part of what is tested.
- ``positive``: every listed ticker whose signal is above 0 (at least 3); the null draws as
  many names from every listed ticker.
In every mode a month with fewer than 3 eligible names holds nothing.

The p-value is the share of null draws whose mean gross excess return is at least the
observed one; the null holds the same number of names in the same months. Costs:
``cost_bps`` per unit of one-way turnover (drift ignored).
"""
from __future__ import annotations

import numpy as np

from brain.research.stats import binom_two_sided

SELECT = ("top_frac", "top_frac_vs_all", "positive")


def choose(s: np.ndarray, listed: np.ndarray, frac: float, select: str = "top_frac"):
    """Holdings on one rebalance day and the pool the null draws from; (None, None) when fewer
    than 3 names are eligible."""
    if select not in SELECT:
        raise ValueError(f"unknown select {select!r}; use one of {SELECT}")
    valid = listed & np.isfinite(s)
    if select == "positive":
        with np.errstate(invalid="ignore"):
            valid = valid & (s > 0)
    n = int(valid.sum())
    if n < 3:
        return None, None
    idx = np.flatnonzero(valid)
    if select == "positive":
        hold = idx
    else:
        k = max(1, int(round(frac * n)))
        hold = idx[np.argsort(-s[idx], kind="stable")[:k]]
    return hold, (valid if select == "top_frac" else listed)


def portfolio(S: np.ndarray, R: np.ndarray, listed: np.ndarray, frac: float, cost_bps: float,
              select: str = "top_frac"):
    M = R.shape[0]
    N = R.shape[1]
    W = np.zeros((M, N))
    gross = np.full(M, np.nan)
    bench = np.full(M, np.nan)
    turn = np.zeros(M)
    missing = 0
    prev = np.zeros(N)
    for m in range(M):
        r = np.nan_to_num(R[m], nan=0.0)
        if listed[m].any():
            bench[m] = r[listed[m]].mean()
        hold, _ = choose(S[m], listed[m], frac, select)
        if hold is None:
            prev = np.zeros(N)
            continue
        k = len(hold)
        W[m, hold] = 1.0 / k
        missing += int((~np.isfinite(R[m, hold])).sum())
        gross[m] = r[hold].mean()
        turn[m] = np.abs(W[m] - prev).sum()
        prev = W[m]
    excess = gross - bench
    net = excess - turn * cost_bps / 1e4
    return {"W": W, "gross": gross, "bench": bench, "excess": excess, "net": net, "turnover": turn,
            "missing_returns": missing}


def perm_null(S, R, listed, frac, months, n_perm: int, seed: int, select: str = "top_frac") -> np.ndarray:
    """Mean gross excess return of n_perm null portfolios over ``months``: each month, as many names
    as the portfolio holds, drawn at random from the mode's pool."""
    rng = np.random.default_rng(seed)
    total = np.zeros(n_perm)
    used = 0
    for m in months:
        hold, pool = choose(S[m], listed[m], frac, select)
        if hold is None:
            continue
        k = len(hold)
        n = int(pool.sum())
        r = np.nan_to_num(R[m], nan=0.0)
        rv = r[pool]
        bench = r[listed[m]].mean()
        keys = rng.random((n_perm, n))
        pick = np.argpartition(keys, k - 1, axis=1)[:, :k]
        total += rv[pick].mean(axis=1) - bench
        used += 1
    return total / max(used, 1)


def summarize(pf: dict, months: np.ndarray) -> dict:
    ex = pf["excess"][months]
    net = pf["net"][months]
    ok = np.isfinite(net)
    ex, net = ex[ok], net[ok]
    n = len(net)
    if n == 0:
        return {"n_months": 0}
    wins = int((net > 0).sum())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    curve = np.cumprod(1 + net)
    dd = float((curve / np.maximum.accumulate(curve) - 1).min())
    return {
        "n_months": n,
        "mean_gross_excess": float(ex.mean()),
        "mean_net_excess": float(net.mean()),
        "ann_net_excess": float((1 + net.mean()) ** 12 - 1),
        "info_ratio": float(net.mean() / sd * np.sqrt(12)) if sd and sd > 0 else float("nan"),
        "months_beating_benchmark": wins,
        "sign_test_p": binom_two_sided(wins, n),
        "max_drawdown_vs_benchmark": dd,
        "avg_turnover": float(pf["turnover"][months][ok].mean()),
        "benchmark_ann_return": float((1 + np.nanmean(pf["bench"][months][ok])) ** 12 - 1),
    }


def evaluate(S, R, listed, months, *, frac, cost_bps, n_perm, seed=0, select="top_frac") -> dict:
    pf = portfolio(S, R, listed, frac, cost_bps, select)
    s = summarize(pf, months)
    if s["n_months"] == 0:
        return {**s, "perm_p": float("nan")}
    null = perm_null(S, R, listed, frac, months, n_perm, seed, select)
    obs = s["mean_gross_excess"]
    s["perm_p"] = float((1 + (null >= obs - 1e-15).sum()) / (1 + n_perm))
    s["null_mean"] = float(null.mean())
    s["null_sd"] = float(null.std())
    s["missing_returns"] = pf["missing_returns"]
    return s
