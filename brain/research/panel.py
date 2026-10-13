"""Monthly panel: trading calendar, rebalance days, forward returns.

Timing (conservative): a signal uses data up to the close of the last trading day
of month m (rebalance day). The trade fills at the next day's close (entry) and is
held to the close of the day after the next rebalance (exit). R[m] is that
entry-to-exit return from dividend-adjusted closes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from brain.bank import Bank
from brain.metrics import PriceBook


@dataclass
class Panel:
    tickers: list[str]
    dates: np.ndarray        # (T,) trading calendar
    adj: np.ndarray          # (T, N) dividend-adjusted close
    rebal: np.ndarray        # (M+1,) calendar index of each rebalance day; the last has no outcome yet
    entry: np.ndarray        # (M,)
    exit: np.ndarray         # (M,)
    R: np.ndarray            # (M, N) forward return, NaN when unknown
    listed: np.ndarray       # (M+1, N) has a price on the rebalance day
    book: PriceBook

    @property
    def months(self) -> np.ndarray:
        return self.dates[self.rebal[:-1]]

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(",".join(self.tickers).encode())
        h.update(self.dates.astype("U10").tobytes())
        h.update(np.nan_to_num(self.adj, nan=-1).round(6).tobytes())
        return h.hexdigest()[:16]


def build(bank: Bank, tickers: list[str], ref: str = "SPY", start: str = "2009-01-01", asof: str | None = None) -> Panel:
    """``asof``: use the price snapshots known at that time (default: the latest), so a past run
    can be rebuilt exactly from a bank that has taken in newer prices since."""
    book = PriceBook(bank, [*tickers, ref], asof)
    missing = [t for t in [*tickers, ref] if not book.has(t)]
    if missing:
        raise SystemExit(f"no price snapshot for {missing}; run: python -m brain ingest {' '.join(missing)}")
    cal = book.dates[ref]
    cal = cal[cal >= start]
    T, N = len(cal), len(tickers)
    adj = np.full((T, N), np.nan)
    for j, t in enumerate(tickers):
        s = pd.Series(book.adj[t], index=book.dates[t])
        s = s[~s.index.duplicated(keep="last")]
        adj[:, j] = s.reindex(cal).ffill(limit=3).to_numpy()
    month = np.array([d[:7] for d in cal])
    rebal = np.flatnonzero(month[:-1] != month[1:])  # last trading day of each completed month
    entry = rebal[:-1] + 1
    exit_ = rebal[1:] + 1
    ok = exit_ < T
    entry, exit_ = entry[ok], exit_[ok]
    rebal = rebal[: len(entry) + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        R = adj[exit_] / adj[entry] - 1.0
    listed = np.isfinite(adj[rebal])
    return Panel(list(tickers), cal, adj, rebal, entry, exit_, R, listed, book)
