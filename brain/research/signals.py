"""Signals: one number per ticker per rebalance day, higher = buy.

Each signal sees only a MonthCtx: prices sliced to the rebalance day and a bank
View at that day. Nothing here can index past ``ctx.i``.

Two kinds. ``HAND`` signals score every ticker at once, month by month (H1-H5).
``PER_TICKER`` signals score one ticker on one rebalance day; they read filings and
insider trades, which for a broad universe do not fit in memory together, so they
run ticker by ticker, each through its own Bank on the same journal.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from brain import events, metrics
from brain.bank import Bank, View
from brain.research.panel import Panel, build
from brain.sources import insiders
from brain.timeutil import days_between, norm_known_at


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


# ---------- per-ticker signals (H3b-H3c, H6-H8b) ----------
INSIDER_DAYS = 90
EVENT_DAYS = 31


def earnings_yield_ttm(ctx: MonthCtx, t: str) -> float | None:
    """Trailing four quarters' net income / market value, broader concepts (metrics.earnings_yield_ttm)."""
    m = metrics.earnings_yield_ttm(ctx.view, t, ctx.panel.book)
    return m.value if m else None


def sue(ctx: MonthCtx, t: str) -> float | None:
    """Standardized unexpected earnings of the latest quarter known on the rebalance day (metrics.sue)."""
    m = metrics.sue(ctx.view, t)
    return m.value if m else None


def insider_buying(ctx: MonthCtx, t: str) -> float:
    """USD of open-market purchases (code P) by officers and directors, in Form 4/5 filings known in
    the 90 days to the rebalance day. 0 when there were none (every listed stock has a value)."""
    since = norm_known_at((date.fromisoformat(ctx.date) - timedelta(days=INSIDER_DAYS)).isoformat())
    buys = insiders.trades(ctx.view, t, lambda f: f.known_at > since and f.context.get("code") == "P")
    return float(sum(f.context.get("value_usd") or 0.0 for f in buys if insiders.is_officer_or_director(f)))


def earnings_reaction(ctx: MonthCtx, t: str) -> float | None:
    """car_3d (excess return over SPY, days -1..+1) of the latest earnings release known in the 31 days
    to the rebalance day whose three days have closed by then. A re-file within a day and an 8-K/A
    within 20 days are the same release (events.distinct). None when there is no such release."""
    evs = [e for e in events.distinct(events.earnings_events(ctx.view, t))
           if days_between(e.known_at, ctx.date) <= EVENT_DAYS]
    for e in reversed(evs):
        r = events.reaction(ctx.panel.book, ctx.view, t, e)
        if r is not None and r["car_3d"] is not None:
            return r["car_3d"]
    return None


PER_TICKER = {"earnings_yield_ttm": earnings_yield_ttm, "sue": sue, "insider_buying": insider_buying,
              "earnings_reaction": earnings_reaction}


def _ticker_major(bank: Bank, panel: Panel, names) -> dict[str, np.ndarray]:
    out = {n: np.full((len(panel.rebal), len(panel.tickers)), np.nan) for n in names}
    for j, t in enumerate(panel.tickers):
        b = Bank(bank.root, use_index=bank.use_index)  # this ticker's facts only; freed after it
        for m, i in enumerate(panel.rebal):
            if not panel.listed[m, j]:
                continue  # never held or ranked: the harness ignores unlisted names
            d = str(panel.dates[i])
            ctx = MonthCtx(m, int(i), d, panel.adj[: i + 1], b.view(d), panel)
            for n in names:
                v = PER_TICKER[n](ctx, t)
                if v is not None:
                    out[n][m, j] = v
    return out


def compute(bank: Bank, panel: Panel, names) -> dict[str, np.ndarray]:
    """(M+1, N) matrix per signal; row m uses information up to rebalance day m."""
    hand = [n for n in names if n in HAND]
    out = {n: np.full((len(panel.rebal), len(panel.tickers)), np.nan) for n in hand}
    if hand:
        for ctx in contexts(bank, panel):
            for n in hand:
                out[n][ctx.m] = HAND[n](ctx)
    rest = [n for n in names if n not in HAND]
    if rest:
        out.update(_ticker_major(bank, panel, rest))
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
    """Signals on the latest rebalance day only (for briefs). Per-ticker signals read through the
    ``bank`` given; for a broad universe use :func:`cross_section`, which reads one ticker at a time."""
    m = len(panel.rebal) - 1
    i = int(panel.rebal[m])
    d = str(panel.dates[i])
    ctx = MonthCtx(m, i, d, panel.adj[: i + 1], bank.view(d), panel)
    out = {}
    for n in names:
        if n in HAND:
            out[n] = HAND[n](ctx)
        else:
            vals = [PER_TICKER[n](ctx, t) for t in panel.tickers]
            out[n] = np.array([np.nan if v is None else v for v in vals], dtype=float)
    return out


# ---------- one rebalance day, many tickers, one at a time (briefs, the terminal's EV) ----------
CROSS_SECTION_VERSION = 1  # bump when a per-ticker signal's code changes: cached values are then recomputed
CROSS_SECTION_CHUNK = 10   # tickers per price read


def _shard_state(bank: Bank, t: str) -> list[int]:
    try:
        st = os.stat(bank.shard(t))
    except FileNotFoundError:
        return [0, 0]
    return [st.st_ino, st.st_size]


def _unchanged_for(bank: Bank, t: str, state, cutoff: str) -> bool:
    """True when ``t``'s journal shard has only grown since ``state`` and every record appended since adds
    a fact known after ``cutoff``: a view at the rebalance day sees exactly what it saw then. Daily prices
    and new filings are known after a past month-end; a backfill (older filings, an insider data set) is
    not, and neither is a downgrade, so those recompute."""
    now = _shard_state(bank, t)
    if now == list(state):
        return True
    if now[0] != state[0] or now[1] < state[1]:
        return False
    with open(bank.shard(t), "rb") as fh:
        fh.seek(state[1])
        tail = fh.read(now[1] - state[1])
    for line in tail.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec.get("op") != "add" or not rec["fact"]["known_at"] > cutoff:
                return False
        except (ValueError, KeyError, TypeError, AttributeError):
            return False
    return True


def _finite(v) -> float | None:
    return float(v) if v is not None and math.isfinite(v) else None


def _save_cache(path: Path, key: dict, entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps({"key": key, "tickers": entries}, separators=(",", ":")))
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)  # a cache: the values can always be computed again


def cross_section(bank: Bank, tickers: list[str], names, cache_path=None, budget_s: float | None = None,
                  first=(), asof: str | None = None) -> dict:
    """Per-ticker signals ``names`` for every ticker on the latest rebalance day, as the backtest computes
    them there: each ticker through its own Bank on the journal (freed after it), only listed tickers
    valued, prices from a panel on the same calendar.

    ``cache_path`` keeps each ticker's values with the state of its journal shard, so a later call on the
    same rebalance day recomputes only tickers that gained something that day's view can see.
    ``budget_s`` bounds the time spent computing; the tickers in ``first`` are always computed. ``asof`` is
    the time the prices are read at (default: the latest), as ``panel.build`` takes it: a decide run reads
    at its T_RUN, so a close ingested this evening, stamped as known at the end of its day, marks the
    month-end it closes. Returns
    ``{"rebalance_date", "values": {name: (N,) array, NaN = no value}, "listed": (N,) bool, "computed",
    "total", "complete"}``; with ``complete`` False the values are a partial cross-section."""
    names = [n for n in dict.fromkeys(names) if n in PER_TICKER]
    base = build(bank, [], asof=asof)  # SPY's calendar: every panel built from this bank has these rebalance days
    m = len(base.rebal) - 1
    if m < 0:
        raise SystemExit("not enough price history for a month-end ranking")
    d = str(base.dates[base.rebal[m]])
    cutoff = norm_known_at(d)
    key = {"version": CROSS_SECTION_VERSION, "rebalance_date": d, "names": sorted(names)}
    path = Path(cache_path) if cache_path else None
    entries: dict = {}
    if path is not None:
        try:
            c = json.loads(path.read_text())
            entries = c["tickers"] if c.get("key") == key and isinstance(c.get("tickers"), dict) else {}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            entries = {}
    want = list(dict.fromkeys([*(t.upper() for t in first), *(t.upper() for t in tickers)]))
    must = {t.upper() for t in first}
    stale = [t for t in want if not isinstance(entries.get(t), dict)
             or not _unchanged_for(bank, t, entries[t].get("state") or [0, 0], cutoff)]
    pending = set(stale)
    start = time.monotonic()
    now = bank.view(asof)
    for k in range(0, len(stale), CROSS_SECTION_CHUNK):
        chunk = stale[k:k + CROSS_SECTION_CHUNK]
        if budget_s is not None and time.monotonic() - start > budget_s:
            chunk = [t for t in chunk if t in must]
            if not chunk:
                continue
        states = {t: _shard_state(bank, t) for t in chunk}  # before reading: a write meanwhile is seen next time
        priced = [t for t in chunk if now.latest(t, "series.price_daily") is not None]
        for t in chunk:
            entries[t] = {"state": states[t], "listed": False, "values": dict.fromkeys(names)}
        if priced:
            panel = build(bank, priced, asof=asof)
            i = int(panel.rebal[m])
            assert str(panel.dates[i]) == d, "one calendar for every panel on this bank"
            for j, t in enumerate(priced):
                if not panel.listed[m, j]:
                    continue  # never held or ranked: the harness ignores unlisted names
                b = Bank(bank.root, use_index=bank.use_index)
                ctx = MonthCtx(m, i, d, panel.adj[: i + 1], b.view(d), panel)
                entries[t] = {"state": states[t], "listed": True,
                              "values": {n: _finite(PER_TICKER[n](ctx, t)) for n in names}}
        pending.difference_update(chunk)
        if path is not None:
            _save_cache(path, key, entries)
    nan = float("nan")
    tk = [t.upper() for t in tickers]

    def val(t, n):
        v = (entries.get(t) or {}).get("values", {}).get(n) if t not in pending else None
        return nan if v is None else float(v)
    return {"rebalance_date": d, "values": {n: np.array([val(t, n) for t in tk], dtype=float) for n in names},
            "listed": np.array([t not in pending and bool((entries.get(t) or {}).get("listed")) for t in tk]),
            "computed": sum(t not in pending for t in tk), "total": len(tk), "complete": not (pending & set(tk))}
