"""Signals: one number per ticker per rebalance day, higher = buy.

Each signal sees only a MonthCtx: prices sliced to the rebalance day and a bank
View at that day. Nothing here can index past ``ctx.i``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain import metrics
from brain.bank import Bank, View
from brain.research.panel import Panel


@dataclass
class MonthCtx:
    m: int
    i: int                 # calendar index of the rebalance day
    date: str
    adj: np.ndarray        # (i+1, N): prices up to and including the rebalance day
    view: View
    panel: Panel


def contexts(bank: Bank, panel: Panel):
    for m, i in enumerate(panel.rebal):
        d = str(panel.dates[i])
        yield MonthCtx(m, int(i), d, panel.adj[: i + 1], bank.view(d), panel)


def _ret(adj: np.ndarray, back_from: int, back_to: int) -> np.ndarray:
    """Return from back_to days ago to back_from days ago (row -1 is today)."""
    if adj.shape[0] <= back_to:
        return np.full(adj.shape[1], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return adj[-1 - back_from] / adj[-1 - back_to] - 1.0


def mom12_1(ctx: MonthCtx) -> np.ndarray:
    """12-month return skipping the latest month (Jegadeesh & Titman 1993)."""
    return _ret(ctx.adj, 21, 252)


def rev1(ctx: MonthCtx) -> np.ndarray:
    """Minus last month's return: short-term reversal (Jegadeesh 1990)."""
    return -_ret(ctx.adj, 0, 21)


def _per_ticker(ctx: MonthCtx, fn) -> np.ndarray:
    out = np.full(len(ctx.panel.tickers), np.nan)
    for j, t in enumerate(ctx.panel.tickers):
        r = fn(ctx.view, t)
        if r is not None:
            out[j] = r.value
    return out


def earnings_yield(ctx: MonthCtx) -> np.ndarray:
    """Last fiscal-year net income / market value (Basu 1977). From SEC filings, point-in-time."""
    book = ctx.panel.book
    return _per_ticker(ctx, lambda v, t: metrics.earnings_yield(v, t, book))


def low_asset_growth(ctx: MonthCtx) -> np.ndarray:
    """Minus fiscal-year asset growth (Cooper, Gulen & Schill 2008)."""
    return -_per_ticker(ctx, metrics.asset_growth)


HAND = {"mom12_1": mom12_1, "rev1": rev1, "earnings_yield": earnings_yield, "low_asset_growth": low_asset_growth}


def compute(bank: Bank, panel: Panel, names) -> dict[str, np.ndarray]:
    """(M+1, N) matrix per signal; row m uses information up to rebalance day m."""
    out = {n: np.full((len(panel.rebal), len(panel.tickers)), np.nan) for n in names}
    for ctx in contexts(bank, panel):
        for n in names:
            out[n][ctx.m] = HAND[n](ctx)
    return out


def rank_cs(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cross-sectional rank scaled to [-0.5, 0.5]; invalid entries -> NaN."""
    out = np.full(x.shape, np.nan)
    ok = valid & np.isfinite(x)
    n = ok.sum()
    if n >= 2:
        r = np.argsort(np.argsort(x[ok], kind="stable"), kind="stable")
        out[ok] = r / (n - 1) - 0.5
    return out


def ridge_learner(features: dict[str, np.ndarray], panel: Panel, lam: float = 1.0, min_months: int = 24):
    """Generic learner: ridge on cross-sectional feature ranks, refit every month on
    months whose outcome was already realised on the rebalance day (expanding window).
    Returns the score matrix and the training months used per month (for audit)."""
    names = sorted(features)
    Mp1, N = features[names[0]].shape
    M = len(panel.entry)
    X = np.zeros((Mp1, N, len(names)))
    for m in range(Mp1):
        for k, n in enumerate(names):
            X[m, :, k] = np.nan_to_num(rank_cs(features[n][m], panel.listed[m]), nan=0.0)
    S = np.full((Mp1, N), np.nan)
    used = {}
    for m in range(Mp1):
        train = [k for k in range(min(m, M)) if panel.exit[k] <= panel.rebal[m]]
        if len(train) < min_months:
            continue
        xs, ys = [], []
        for k in train:
            y = rank_cs(panel.R[k], panel.listed[k])
            ok = np.isfinite(y)
            xs.append(X[k][ok])
            ys.append(y[ok])
        A, b = np.vstack(xs), np.concatenate(ys)
        beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
        S[m] = X[m] @ beta
        S[m, ~panel.listed[m]] = np.nan
        used[m] = train
    return S, used


def compute_last(bank: Bank, panel: Panel, names) -> dict[str, np.ndarray]:
    """Signals on the latest rebalance day only (for briefs)."""
    m = len(panel.rebal) - 1
    i = int(panel.rebal[m])
    d = str(panel.dates[i])
    ctx = MonthCtx(m, i, d, panel.adj[: i + 1], bank.view(d), panel)
    return {n: HAND[n](ctx) for n in names}
