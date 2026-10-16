"""Point-in-time features and the one label rule for the forecaster (docs/FORECAST.md, sections 2 and 3).

Timing. A forecast is made after the close of a SPY session d.
- Prices come from one ``PriceBook`` per run, built at the run's pinned ``prices_asof``, plus the same
  snapshots' daily CSVs for high, low and volume. Every array is read only up to the row of d. (The bank's
  price snapshots were all fetched recently, so a book at T(d) would hold no prices for a past d.)
- Filings, insider trades and earnings releases are read through ``bank.view(T(d))``, T(d) = d 23:59:59Z.
- Fundamentals are computed at month-end views and carried forward: a forecast on d takes the values of
  the last month-end strictly before d's month, so it needs no bar after d to know which month-end that is.
- The label is ``claims.outcome`` from the first SPY session strictly after d, so anything public on d,
  even an evening filing stamped T(d), is known before its window starts.

Per date, P, B and F features are ranked across the stocks listed that day into [-0.5, 0.5] (ties share
their mean rank), NaN becomes 0 and the ``has_`` indicators (left as 0/1) carry the gap. R features are
z-scored by their own values on earlier grid dates only.

Concepts with no ``metrics`` helper, read with ``metrics.annual`` or ``metrics.ratio``:
gross profit ``us-gaap:GrossProfit``; assets ``us-gaap:Assets``; liabilities ``us-gaap:Liabilities``;
operating cash flow ``us-gaap:NetCashProvidedByUsedInOperatingActivities``; net income ``metrics.NET_INCOME``.
Many filers tag neither gross profit nor total liabilities, so each has one fallback, used only when the
first reading is missing or stale: gross profit = revenue (``metrics``' REVENUE) minus cost of revenue
(``us-gaap:CostOfRevenue``, else ``us-gaap:CostOfGoodsAndServicesSold``) in the latest fiscal year all three
report; liabilities over assets = 1 - ``us-gaap:StockholdersEquity`` over assets.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from brain import bank as bank_module
from brain import claims, config, events, metrics, series, timeutil
from brain.bank import Bank, View, canon
from brain.metrics import PriceBook
from brain.research import signals
from brain.sources import insiders, sec
from brain.timeutil import days_between, norm_known_at, now_iso

FEATURES_VERSION = 1

BENCH = "SPY"
VIX = "^VIX"
PRICE_START = "2007-01-01"   # the calendar's first session: four years before training starts, for ltr_3y
FILL_SESSIONS = 3            # a missing bar takes the last close for up to 3 sessions, as research.panel does
PRICE_CHUNK = 64             # tickers per PriceBook read (well under series' 512-snapshot cache)
LISTED_MIN_BARS = 252
STALE_CLOSE_DAYS = claims.STALE_CLOSE_DAYS  # 4: a stock is listed at d when its last close is this recent
MIN_SHARE = 0.8              # a window statistic needs at least this share of its values
RSI_PERIOD = 14
RSI_WINDOW = 250             # Wilder's smoothing runs over the last 250 changes; the seed's weight is then < 1e-7
INSIDER_DAYS = signals.INSIDER_DAYS   # 90
EARNINGS_SESSIONS = 63
Z_MIN_DATES = 52

GROSS_PROFIT = ["us-gaap:GrossProfit"]
ASSETS = ["us-gaap:Assets"]
LIABILITIES = ["us-gaap:Liabilities"]
OPERATING_CASH_FLOW = ["us-gaap:NetCashProvidedByUsedInOperatingActivities"]
NET_INCOME = metrics.NET_INCOME
REVENUE = list(metrics.REVENUE)
COST_OF_REVENUE = ["us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold"]   # gross profit's fallback
EQUITY = ["us-gaap:StockholdersEquity"]                                             # leverage's fallback

PRICE_FEATURES = [
    ("r_5", "P", "return over the last 5 sessions"),
    ("r_21", "P", "return over the last 21 sessions"),
    ("r_63", "P", "return over the last 63 sessions"),
    ("r_126", "P", "return over the last 126 sessions"),
    ("mom_12_1", "P", "return from 252 to 21 sessions ago"),
    ("ma_50", "P", "adjusted close over its 50-session mean, minus 1"),
    ("ma_200", "P", "adjusted close over its 200-session mean, minus 1"),
    ("rsi_14", "P", "Wilder's 14-session relative strength index (0-100)"),
    ("vol_21", "P", "standard deviation of daily log returns over 21 sessions"),
    ("vol_63", "P", "standard deviation of daily log returns over 63 sessions"),
    ("vol_ratio", "P", "vol_21 / vol_63"),
    ("beta_252", "P", "slope of the stock's daily returns on SPY's over 252 sessions"),
    ("idio_vol_63", "P", "standard deviation over 63 sessions of daily returns less beta_252 times SPY's"),
    ("skew_63", "P", "skewness of daily log returns over 63 sessions"),
    ("range_21", "P", "mean of (high - low) / close over 21 sessions"),
    ("dist_52w_high", "B", "close over its 252-session high, minus 1 (anchoring)"),
    ("dist_52w_low", "B", "close over its 252-session low, minus 1 (anchoring)"),
    ("max_ret_21", "B", "largest daily return in 21 sessions (lottery preference)"),
    ("min_ret_21", "B", "smallest daily return in 21 sessions"),
    ("rev_5", "B", "minus r_5 (short-term overreaction)"),
    ("rev_21", "B", "minus r_21 (short-term overreaction)"),
    ("ltr_3y", "B", "minus the return from 756 to 252 sessions ago (long-term overreaction)"),
    ("abn_volume", "B", "log of the 5-session mean volume over the 63-session mean (attention)"),
    ("turnover_trend", "B", "log of the 21-session mean dollar volume over the 252-session mean"),
]
EVENT_FEATURES = [
    ("insider_buy_90d", "B", ("log1p of the USD of open-market purchases (code P) by officers and directors in "
                              "Form 4/5 filings known in the 90 days to d (signals.insider_buying's rule)")),
    ("earnings_car_3d", "B", ("3-day excess return over SPY (events.reaction) around the latest earnings release "
                              "whose window closed by d, within 63 sessions; 0 when there is none")),
    ("has_earnings_recent", "B", "1 when earnings_car_3d comes from a release, else 0"),
]
FUND_FEATURES = [
    ("earnings_yield_ttm", "F", "trailing four quarters' net income over market value (metrics.earnings_yield_ttm)"),
    ("log_mktcap", "F", "log of market value (metrics.market_cap_any)"),
    ("sales_growth", "F", "revenue growth, fiscal year over year (metrics.revenue_growth)"),
    ("gross_profitability", "F", "gross profit over assets, latest fiscal year (metrics.ratio)"),
    ("asset_growth", "F", "total assets growth, fiscal year over year (metrics.asset_growth)"),
    ("accruals", "F", "net income minus operating cash flow, over assets, latest fiscal year"),
    ("leverage", "F", "liabilities over assets, latest fiscal year"),
    ("sue", "F", "standardized unexpected earnings of the latest quarter (metrics.sue)"),
]
HAS_FEATURES = [(f"has_{n}", "F", f"1 when the bank has {n} at the month-end, else 0") for n, _, _ in FUND_FEATURES]
REGIME_FEATURES = [
    ("vix", "R", "^VIX close on d"),
    ("vix_chg_21", "R", "^VIX close on d minus its close 21 sessions earlier"),
    ("spy_r_21", "R", "SPY's return over the last 21 sessions"),
    ("spy_r_252", "R", "SPY's return over the last 252 sessions"),
    ("spy_above_200", "R", "1 when SPY closes above its 200-session mean, else 0"),
    ("mkt_vol_21", "R", "standard deviation of SPY's daily log returns over 21 sessions"),
]
FEATURES: list[tuple[str, str, str]] = PRICE_FEATURES + EVENT_FEATURES + FUND_FEATURES + HAS_FEATURES + REGIME_FEATURES
NAMES = [n for n, _, _ in FEATURES]
GROUPS = [g for _, g, _ in FEATURES]
INDICATORS = {"has_earnings_recent", *(n for n, _, _ in HAS_FEATURES)}
FUND_NAMES = [n for n, _, _ in FUND_FEATURES]
EVENT_NAMES = [n for n, _, _ in EVENT_FEATURES]
REGIME_NAMES = [n for n, _, _ in REGIME_FEATURES]

# Modules whose code computes a feature or the label; their shas enter the cache key with features.py's. The
# bank's View decides what is visible at T(d) for every feature, label and price read; timeutil normalises the
# times it compares; sources/sec holds the revenue concepts metrics reads.
_SOURCE = Path(__file__)
_DEPS = (metrics, events, claims, insiders, signals, series, bank_module, timeutil, sec)


def T(d: str) -> str:
    """The time a forecast on session d reads the bank at: the end of d's UTC day."""
    return str(d)[:10] + "T23:59:59Z"


# ---------- prices ----------
@dataclass
class Prices:
    """Every universe stock on SPY's calendar from PRICE_START, from one book at ``asof``."""
    tickers: list[str]
    dates: np.ndarray        # (T,) SPY's sessions, 'YYYY-MM-DD'
    adj: np.ndarray          # (T, N) split- and dividend-adjusted close, missing bars filled for up to 3 sessions
    close: np.ndarray        # (T, N) split-adjusted close, filled the same way
    high: np.ndarray         # (T, N) split-adjusted, NaN where the stock has no bar that session
    low: np.ndarray          # (T, N)
    volume: np.ndarray       # (T, N)
    bars: np.ndarray         # (T, N) the stock's own bars dated <= that session, counted from its first bar
    age: np.ndarray          # (T, N) days from the stock's last bar dated <= that session to the session
    spy: np.ndarray          # (T,) SPY's adjusted close
    vix: np.ndarray          # (T,) ^VIX close from rows dated <= the session (filled up to 3 sessions); NaN before 2015
    book: PriceBook
    asof: str
    retrieved_by: str | None = None   # pinned: only snapshots fetched by then (a declared test's prices_asof)
    _pos: dict = field(default_factory=dict, repr=False)

    def index(self, d: str) -> int:
        """The calendar row of session d; KeyError when d is not a SPY session in the book."""
        if not self._pos:
            self._pos.update({str(x): k for k, x in enumerate(self.dates)})
        return self._pos[str(d)[:10]]

    def listed(self, i: int) -> np.ndarray:
        """(N,) a close dated within 4 days of session i and at least 252 bars up to it."""
        return (self.bars[i] >= LISTED_MIN_BARS) & (self.age[i] <= STALE_CLOSE_DAYS)

    def snapshots(self) -> dict[str, str]:
        """The id of every price snapshot read (stocks, SPY, ^VIX)."""
        return {t: f.id for t, f in sorted(self.book.snapshot.items())}


def _universe(tickers) -> list[str]:
    out = []
    for t in tickers:
        t = str(t).upper()
        if t not in (BENCH, VIX) and t not in out:
            out.append(t)
    return out


def prices(bank: Bank, tickers, prices_asof: str | None, start: str = PRICE_START, *,
           retrieved_by: str | None = None) -> Prices:
    """One book at ``prices_asof`` (its latest snapshot per ticker) and the same snapshots' CSVs, on SPY's
    calendar from ``start``. SPY and ^VIX are context, never universe stocks: they are left out of
    ``tickers`` here. ``retrieved_by``: read only the snapshots fetched by then (``PriceBook``), so a declared
    test reads the book as it stood at its pinned ``prices_asof``. Raises SystemExit when SPY has no snapshot
    at ``prices_asof``."""
    tk = _universe(tickers)
    asof = norm_known_at(prices_asof) if prices_asof else now_iso()
    rb = norm_known_at(retrieved_by) if retrieved_by else None
    book = PriceBook(bank, [BENCH, VIX], asof, retrieved_by=rb)
    if not book.has(BENCH):
        raise SystemExit(f"no {BENCH} price snapshot known at {asof}; run: python -m brain ingest {BENCH}")
    cal_all = book.dates[BENCH]
    keep = cal_all >= start
    cal = cal_all[keep]
    T_, N = len(cal), len(tk)
    where = pd.Index(cal)
    cal_day = cal.astype("datetime64[D]")
    adj, close, high, low, vol = (np.full((T_, N), np.nan) for _ in range(5))
    bars = np.zeros((T_, N), dtype=np.int32)
    age = np.full((T_, N), 1 << 20, dtype=np.int32)
    for j, t in enumerate(tk):
        if j % PRICE_CHUNK == 0:
            # The book is read in chunks at the same asof, so each chunk's CSVs are still in series' cache
            # when their high, low and volume are read just after.
            part = PriceBook(bank, tk[j:j + PRICE_CHUNK], asof, retrieved_by=rb)
            for name, d in vars(part).items():
                if isinstance(d, dict):
                    getattr(book, name).update(d)
        if not book.has(t):
            continue
        own = book.dates[t]
        df = series.load(book.snapshot[t].value)
        pos = where.get_indexer(own)
        m = pos >= 0
        for arr, src in ((adj, book.adj[t]), (close, book.close[t]), (high, df["high"]), (low, df["low"]),
                         (vol, df["volume"])):
            arr[pos[m], j] = np.asarray(src, dtype=float)[m]
        k = np.searchsorted(own, cal, side="right")
        bars[:, j] = k
        has = k > 0
        last = own.astype("datetime64[D]")[np.maximum(k - 1, 0)]
        age[has, j] = (cal_day[has] - last[has]).astype(np.int64)
    adj = pd.DataFrame(adj).ffill(limit=FILL_SESSIONS).to_numpy()
    close = pd.DataFrame(close).ffill(limit=FILL_SESSIONS).to_numpy()
    vix = np.full(T_, np.nan)
    if book.has(VIX):
        vdf = series.load(book.snapshot[VIX].value)
        vpos = where.get_indexer(book.dates[VIX])
        vm = vpos >= 0
        vix[vpos[vm]] = np.asarray(vdf["close"], dtype=float)[vm]
        vix = pd.Series(vix).ffill(limit=FILL_SESSIONS).to_numpy()
    spy = np.asarray(book.adj[BENCH], dtype=float)[keep]
    return Prices(tk, cal, adj, close, high, low, vol, bars, age, spy, vix, book, asof, retrieved_by=rb)


# ---------- the label ----------
def outcome(book: PriceBook, ticker: str, d: str, h: int) -> dict | None:
    """``claims.outcome(book, ticker, T(d), h, "SPY")`` verbatim: entry = the first SPY session strictly
    after d, exit = h SPY sessions after the entry; {"entry", "exit", "excess"}; None when a close is
    missing or more than 4 days stale, or the exit is not in the book yet."""
    return claims.outcome(book, ticker, T(d), h, BENCH)


# ---------- window statistics over (rows, N), oldest row first ----------
def _need(k: int) -> int:
    return math.ceil(MIN_SHARE * k)


def _nan(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def _mean(w: np.ndarray | None, k: int, n: int) -> np.ndarray:
    if w is None or w.shape[0] < k:
        return _nan(n)
    w = w[-k:]
    ok = np.isfinite(w)
    c = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(ok, w, 0.0).sum(0) / c
    return np.where(c >= _need(k), m, np.nan)


def _moments(w: np.ndarray | None, k: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(sample standard deviation, skewness) over the last k rows."""
    if w is None or w.shape[0] < k:
        return _nan(n), _nan(n)
    w = w[-k:]
    ok = np.isfinite(w)
    c = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(ok, w, 0.0).sum(0) / c
        dv = np.where(ok, w - m, 0.0)
        s2 = (dv * dv).sum(0)
        sd = np.sqrt(s2 / (c - 1))
        m2, m3 = s2 / c, (dv ** 3).sum(0) / c
        sk = np.where(m2 > 0, m3 / m2 ** 1.5, np.nan)
    good = c >= _need(k)
    return np.where(good, sd, np.nan), np.where(good, sk, np.nan)


def _extreme(w: np.ndarray | None, k: int, n: int, fn) -> np.ndarray:
    if w is None or w.shape[0] < k:
        return _nan(n)
    w = w[-k:]
    ok = np.isfinite(w)
    fill = -np.inf if fn is np.max else np.inf
    v = fn(np.where(ok, w, fill), axis=0)
    return np.where(ok.sum(0) >= _need(k), v, np.nan)


def _beta(rs: np.ndarray, rm: np.ndarray, k: int, n: int) -> np.ndarray:
    if rs.shape[0] < k:
        return _nan(n)
    y, x = rs[-k:], np.broadcast_to(rm[-k:, None], rs[-k:].shape)
    ok = np.isfinite(y) & np.isfinite(x)
    c = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.where(ok, x, 0.0).sum(0) / c
        my = np.where(ok, y, 0.0).sum(0) / c
        dx, dy = np.where(ok, x - mx, 0.0), np.where(ok, y - my, 0.0)
        var = (dx * dx).sum(0)
        b = (dx * dy).sum(0) / var
    return np.where((c >= _need(k)) & (var > 0), b, np.nan)


def _rsi(a: np.ndarray, n: int) -> np.ndarray:
    """Wilder's RSI over the last RSI_WINDOW changes of ``a`` (rows oldest first): the first 14 changes
    seed the average gain and loss, each later change moves them by 1/14. A missing change counts as no
    move; NaN with fewer than 15 known changes."""
    ch = np.diff(a[-(RSI_WINDOW + 1):], axis=0)
    if ch.shape[0] < RSI_PERIOD:
        return _nan(n)
    ok = np.isfinite(ch)
    ch = np.where(ok, ch, 0.0)
    gain, loss = np.maximum(ch, 0.0), np.maximum(-ch, 0.0)
    ag, al = gain[:RSI_PERIOD].mean(0), loss[:RSI_PERIOD].mean(0)
    for g, lo in zip(gain[RSI_PERIOD:], loss[RSI_PERIOD:]):
        ag = (ag * (RSI_PERIOD - 1) + g) / RSI_PERIOD
        al = (al * (RSI_PERIOD - 1) + lo) / RSI_PERIOD
    with np.errstate(invalid="ignore", divide="ignore"):
        rsi = np.where(al > 0, 100.0 - 100.0 / (1.0 + ag / al), np.where(ag > 0, 100.0, 50.0))
    return np.where((ok.sum(0) >= RSI_PERIOD + 1) & np.isfinite(a[-1]), rsi, np.nan)


def _log_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((a > 0) & (b > 0), np.log(a / b), np.nan)


# ---------- features on one date ----------
def stock_features(px: Prices, i: int) -> dict[str, np.ndarray]:
    """P features and the price-based B features at calendar row i (= d), from rows <= i only. (N,) each,
    NaN when undefined (too little history, or a missing close at an end point)."""
    n = len(px.tickers)
    A, C = px.adj[: i + 1], px.close[: i + 1]
    H, L, V = px.high[: i + 1], px.low[: i + 1], px.volume[: i + 1]
    spy = px.spy[: i + 1]
    a = A[-1]

    def back(k: int) -> np.ndarray:
        return A[-1 - k] if A.shape[0] > k else _nan(n)

    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in (5, 21, 63, 126):
            out[f"r_{k}"] = a / back(k) - 1.0
        out["mom_12_1"] = back(21) / back(252) - 1.0
        out["ma_50"] = a / _mean(A, 50, n) - 1.0
        out["ma_200"] = a / _mean(A, 200, n) - 1.0
        out["rsi_14"] = _rsi(A, n)
        w = A[-253:]
        sr = w[1:] / w[:-1] - 1.0                      # daily simple returns, the last 252
        lr = np.log(w[1:] / w[:-1])                    # daily log returns
        sw = spy[-253:]
        rm = sw[1:] / sw[:-1] - 1.0
        out["vol_21"] = _moments(lr, 21, n)[0]
        sd63, sk63 = _moments(lr, 63, n)
        out["vol_63"] = sd63
        out["vol_ratio"] = np.where(sd63 > 0, out["vol_21"] / sd63, np.nan)
        beta = _beta(sr, rm, 252, n)
        out["beta_252"] = beta
        if sr.shape[0] >= 63:
            resid = sr[-63:] - beta * rm[-63:, None]
            out["idio_vol_63"] = _moments(resid, 63, n)[0]
        else:
            out["idio_vol_63"] = _nan(n)
        out["skew_63"] = sk63
        out["range_21"] = _mean((H - L) / C, 21, n)
        c = C[-1]
        out["dist_52w_high"] = c / _extreme(C, 252, n, np.max) - 1.0
        out["dist_52w_low"] = c / _extreme(C, 252, n, np.min) - 1.0
        out["max_ret_21"] = _extreme(sr, 21, n, np.max)
        out["min_ret_21"] = _extreme(sr, 21, n, np.min)
        out["rev_5"] = -out["r_5"]
        out["rev_21"] = -out["r_21"]
        out["ltr_3y"] = -(back(252) / back(756) - 1.0)
        out["abn_volume"] = _log_ratio(_mean(V, 5, n), _mean(V, 63, n))
        dv = C * V
        out["turnover_trend"] = _log_ratio(_mean(dv, 21, n), _mean(dv, 252, n))
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in out.items()}


def regime_features(px: Prices, i: int) -> dict[str, float]:
    """R features at calendar row i, the same for every stock: from SPY's and ^VIX's rows <= i."""
    s, v = px.spy[: i + 1], px.vix[: i + 1]

    def at(x, k):
        return float(x[-1 - k]) if x.shape[0] > k else math.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(s[-22:][1:] / s[-22:][:-1]) if s.shape[0] >= 22 else np.array([])
        ma = float(s[-200:].mean()) if s.shape[0] >= 200 else math.nan
        out = {
            "vix": at(v, 0),
            "vix_chg_21": at(v, 0) - at(v, 21),
            "spy_r_21": at(s, 0) / at(s, 21) - 1.0,
            "spy_r_252": at(s, 0) / at(s, 252) - 1.0,
            "spy_above_200": (1.0 if at(s, 0) > ma else 0.0) if math.isfinite(ma) else math.nan,
            "mkt_vol_21": float(np.std(lr, ddof=1)) if lr.shape[0] == 21 else math.nan,
        }
    return {k: (x if math.isfinite(x) else math.nan) for k, x in out.items()}


# ---------- what one ticker's journal says: insider buys, earnings reactions, fundamentals ----------
def _insider_rows(view: View, t: str) -> tuple[list, list[str]]:
    """Code-P trades known at the view, oldest filing first, as ``insiders.trades`` orders them."""
    buys = sorted((f for f in view.facts(t, insiders.ATTR) if f.context.get("code") == "P"),
                  key=lambda f: (f.known_at, f.id))
    return buys, [f.known_at for f in buys]


def _insider_usd(buys: list, keys: list[str], d: str) -> float:
    """``signals.insider_buying`` on d: the code-P trades known in (d - 90 days, T(d)], one copy per
    transaction (``insiders.trades``' key, the earlier-known copy kept), by officers and directors, in USD."""
    since = norm_known_at((date.fromisoformat(d[:10]) - timedelta(days=INSIDER_DAYS)).isoformat())
    lo, hi = bisect.bisect_right(keys, since), bisect.bisect_right(keys, T(d))
    seen, usd = set(), 0.0
    for f in buys[lo:hi]:
        c = f.context
        k = (c.get("accn"), c.get("owner_cik"), f.as_of, c.get("code"), f.value, c.get("price"), c.get("ownership"),
             c.get("lot"))
        if k in seen:
            continue
        seen.add(k)
        if insiders.is_officer_or_director(f):
            usd += c.get("value_usd") or 0.0
    return float(usd)


def _earnings_car(bank: Bank, book: PriceBook, t: str, evs: list, keys: list[str], d: str,
                  spy_dates: np.ndarray) -> tuple[float, float]:
    """(car_3d, 1) of the latest distinct release known at T(d) whose 3-day window closed by d, when its
    day is within 63 SPY sessions of d; else (0, 0). Releases known at T(d) are a prefix of those known
    later (oldest first), and ``events.distinct`` keeps or drops each by the ones before it only."""
    n = bisect.bisect_right(keys, T(d))
    if n == 0:
        return 0.0, 0.0
    view = bank.view(T(d))
    i_d = int(np.searchsorted(spy_dates, d[:10], side="right")) - 1
    for e in reversed(evs[:n]):
        r = events.reaction(book, view, t, e)
        if r is None or r["car_3d"] is None:
            continue
        if i_d - int(np.searchsorted(spy_dates, r["date"], side="left")) <= EARNINGS_SESSIONS:
            return float(r["car_3d"]), 1.0
        break
    return 0.0, 0.0


def _latest_year(view: View, t: str, parts: dict) -> dict | None:
    """The facts of the latest fiscal year that every one of ``parts`` ({name: (concepts, flow)}) reports,
    when that year is current and its assets (``"assets"``) are positive; else None. As ``metrics.ratio``."""
    by = {k: {f.as_of: f for f in metrics.annual(view, t, c, flow=fl)} for k, (c, fl) in parts.items()}
    years = sorted(set.intersection(*(set(s) for s in by.values())))
    if not years:
        return None
    y = years[-1]
    a = by["assets"][y].value
    if days_between(y, view.t) > metrics.STALE_DAYS or not a or a <= 0:
        return None
    return {k: s[y] for k, s in by.items()}


def _val(m) -> float:
    return float(m.value) if m is not None and m.value is not None and math.isfinite(m.value) else math.nan


def fundamentals_with_inputs(view: View, t: str, book: PriceBook) -> tuple[dict[str, float], list]:
    """The F features of one ticker at one view (a month-end's), NaN where the bank lacks them, and the
    facts each defined value was computed from (price snapshots included), for a forecast to cite."""
    out: dict[str, float] = {}
    used: dict[str, list] = {}

    def take(name: str, m) -> None:
        out[name] = _val(m)
        if math.isfinite(out[name]):
            used[name] = list(m.inputs)

    take("earnings_yield_ttm", metrics.earnings_yield_ttm(view, t, book))
    mc = metrics.market_cap_any(view, t, book)
    take("log_mktcap", mc)
    out["log_mktcap"] = math.log(out["log_mktcap"]) if out["log_mktcap"] > 0 else math.nan
    take("sales_growth", metrics.revenue_growth(view, t))
    gp = metrics.ratio(view, t, GROSS_PROFIT, ASSETS, flow_num=True, flow_den=False)
    if gp is None:  # fallback: revenue minus cost of revenue, over assets
        y = _latest_year(view, t, {"rev": (REVENUE, True), "cost": (COST_OF_REVENUE, True), "assets": (ASSETS, False)})
        if y is not None:
            gp = metrics.M((float(y["rev"].value) - float(y["cost"].value)) / float(y["assets"].value),
                           y["assets"].as_of, [y["rev"], y["cost"], y["assets"]])
    take("gross_profitability", gp)
    take("asset_growth", metrics.asset_growth(view, t))
    y = _latest_year(view, t, {"ni": (NET_INCOME, True), "ocf": (OPERATING_CASH_FLOW, True), "assets": (ASSETS, False)})
    take("accruals", None if y is None else metrics.M(
        (float(y["ni"].value) - float(y["ocf"].value)) / float(y["assets"].value), y["assets"].as_of,
        [y["ni"], y["ocf"], y["assets"]]))
    lev = metrics.ratio(view, t, LIABILITIES, ASSETS, flow_num=False, flow_den=False)
    if lev is None:  # fallback: everything that is not shareholders' equity, over assets
        y = _latest_year(view, t, {"equity": (EQUITY, False), "assets": (ASSETS, False)})
        if y is not None:
            lev = metrics.M(1.0 - float(y["equity"].value) / float(y["assets"].value), y["assets"].as_of,
                            [y["equity"], y["assets"]])
    take("leverage", lev)
    take("sue", metrics.sue(view, t))
    vals = {k: (out[k] if math.isfinite(out[k]) else math.nan) for k in FUND_NAMES}
    facts = list({f.id: f for k in FUND_NAMES if math.isfinite(vals[k]) for f in used.get(k, [])}.values())
    return vals, facts


def fundamentals_at(view: View, t: str, book: PriceBook) -> dict[str, float]:
    """The F features of one ticker at one view (a month-end's), NaN where the bank lacks them."""
    return fundamentals_with_inputs(view, t, book)[0]


def _ticker_rows(root, use_index: bool, t: str, book: PriceBook, dates: list[str], month_ends: list[str]):
    """One ticker through its own Bank (freed after it): (fundamentals (M, 8) at each month-end view,
    events (D, 3) at each date's view). ``dates`` and ``month_ends`` ascending."""
    b = Bank(root, use_index=use_index)
    fund = np.full((len(month_ends), len(FUND_NAMES)), np.nan)
    for m, me in enumerate(month_ends):
        r = fundamentals_at(b.view(T(me)), t, book)
        fund[m] = [r[k] for k in FUND_NAMES]
    ev = np.zeros((len(dates), len(EVENT_NAMES)))
    if dates:
        last = b.view(T(dates[-1]))
        buys, bkeys = _insider_rows(last, t)
        evs = events.distinct(events.earnings_events(last, t))
        ekeys = [e.known_at for e in evs]
        spy_dates = book.dates[BENCH]
        for g, d in enumerate(dates):
            car, has = _earnings_car(b, book, t, evs, ekeys, d, spy_dates)
            ev[g] = [math.log1p(_insider_usd(buys, bkeys, d)), car, has]
    return fund, ev


def event_features(bank: Bank, tickers, d: str, book: PriceBook) -> dict[str, np.ndarray]:
    """insider_buy_90d, earnings_car_3d and has_earnings_recent at view(T(d)), (N,) each."""
    tk = [str(t).upper() for t in tickers]
    out = np.zeros((len(tk), len(EVENT_NAMES)))
    for j, t in enumerate(tk):
        out[j] = _ticker_rows(bank.root, bank.use_index, t, book, [str(d)[:10]], [])[1][0]
    return {k: out[:, c].copy() for c, k in enumerate(EVENT_NAMES)}


def fundamental_features(bank: Bank, tickers, month_ends, book: PriceBook) -> dict[str, np.ndarray]:
    """The F features at view(T(month_end)), with market values from ``book`` at that date: (M, N) each,
    NaN where the bank lacks one, plus each one's ``has_`` indicator."""
    tk = [str(t).upper() for t in tickers]
    mes = [str(m)[:10] for m in month_ends]
    out = np.full((len(mes), len(tk), len(FUND_NAMES)), np.nan)
    for j, t in enumerate(tk):
        out[:, j, :] = _ticker_rows(bank.root, bank.use_index, t, book, [], mes)[0]
    res = {k: out[:, :, c].copy() for c, k in enumerate(FUND_NAMES)}
    res.update({f"has_{k}": np.isfinite(res[k]).astype(float) for k in FUND_NAMES})
    return res


def month_end_index(dates: np.ndarray, i: int) -> int:
    """The row of the month-end whose fundamentals a forecast on row i uses: the last session before the
    first day of i's month (-1 when there is none). Values computed there carry forward to every later
    date through the next month-end, and never back."""
    return int(np.searchsorted(dates, str(dates[i])[:7] + "-01", side="left")) - 1


# ---------- transforms ----------
def rank_unit(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cross-sectional rank of the valid, finite entries scaled to [-0.5, 0.5]; ties share their mean rank;
    every other entry is 0."""
    out = np.zeros(x.shape)
    ok = valid & np.isfinite(x)
    n = int(ok.sum())
    if n < 2:
        return out
    v = x[ok]
    order = np.argsort(v, kind="mergesort")
    s = v[order]
    first = np.concatenate(([True], s[1:] != s[:-1]))
    grp = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    ends = np.concatenate((starts[1:], [n])) - 1
    r = np.empty(n)
    r[order] = ((starts + ends) / 2.0)[grp]
    out[ok] = r / (n - 1) - 0.5
    return out


def zscore_earlier(v: np.ndarray, min_dates: int = Z_MIN_DATES) -> np.ndarray:
    """Each date's value z-scored by the mean and standard deviation of the finite values on earlier dates
    only (``v`` in date order); 0 with fewer than ``min_dates`` of them, a NaN value or no spread."""
    out = np.zeros(len(v))
    for g in range(len(v)):
        prev = v[:g][np.isfinite(v[:g])]
        if len(prev) < min_dates or not np.isfinite(v[g]):
            continue
        sd = float(prev.std(ddof=1))
        if sd > 0:
            out[g] = (v[g] - float(prev.mean())) / sd
    return out


def transform(raw: np.ndarray, listed: np.ndarray) -> np.ndarray:
    """One date's (N, F) raw values to model inputs: P, B and F ranked across the listed stocks, NaN to 0,
    indicators as they are. R columns are left to ``zscore_earlier`` (filled with 0 here)."""
    X = np.zeros(raw.shape)
    for c, (name, group, _) in enumerate(FEATURES):
        if group == "R":
            continue
        if name in INDICATORS:
            X[:, c] = np.where(listed, np.nan_to_num(raw[:, c], nan=0.0), 0.0)
        else:
            X[:, c] = rank_unit(raw[:, c], listed)
    return X


# ---------- the feature set ----------
@dataclass
class FeatureSet:
    """One row per (grid date, listed stock) whose outcome is known, in date then ticker order."""
    X: np.ndarray            # (S, F) model inputs
    raw: np.ndarray          # (S, F) float32 values before the transforms (NaN where undefined)
    y: np.ndarray            # (S,) int8: 1 when excess > 0 (an exactly zero excess is dropped)
    excess: np.ndarray       # (S,) excess return over SPY, entry to exit (outcome)
    r_total: np.ndarray      # (S,) the stock's own return over the same window and closes
    close: np.ndarray        # (S,) the split-adjusted close on d
    entry: np.ndarray        # (S,) 'YYYY-MM-DD'
    exit: np.ndarray         # (S,) the stored exit: every cutoff compares with it
    date: np.ndarray         # (S,) the forecast date d
    ticker: np.ndarray       # (S,)
    names: list[str]
    groups: list[str]
    dropped: dict            # {"unlisted", "no_outcome", "zero_excess", "pending"}: (date, stock) pairs left out
    pending: dict            # listed rows whose window has not closed in the book: {"X", "raw", "date", "ticker", "close"}
    grid: list[str]
    h: int
    prices_asof: str
    key: dict                # the cache key (goes into each result)
    tickers: list[str] = field(default_factory=list)
    seconds: float = 0.0     # time to build (0 when read from the cache)

    def realised(self, cutoff: str) -> np.ndarray:
        """(S,) samples whose stored exit is on or before the cutoff date."""
        return self.exit <= str(cutoff)[:10]

    def col(self, name: str) -> int:
        return self.names.index(name)


def _sha_bytes(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(hashlib.sha256(p).digest())
    return h.hexdigest()[:16]


def code_sha() -> str:
    """sha256[:16] of features.py."""
    return hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]


def deps_sha() -> str:
    """sha of the modules features.py calls to compute a feature or the label."""
    return _sha_bytes(*(Path(m.__file__).read_bytes() for m in _DEPS))


def snapshot_ids(bank: Bank, tickers, prices_asof: str, retrieved_by: str | None = None) -> list[list]:
    """[ticker, id of the price snapshot a book at ``prices_asof`` reads, or None] for each universe stock,
    then SPY and ^VIX; ``retrieved_by`` as in ``prices``."""
    v = bank.view(norm_known_at(prices_asof))
    keep = metrics.fetched_by(retrieved_by) if retrieved_by else None
    out = []
    for t in [*_universe(tickers), BENCH, VIX]:
        f = v.latest(t, "series.price_daily", keep)
        out.append([t, f.id if f is not None else None])
    return out


def snapshots_sha(bank: Bank, tickers, prices_asof: str, retrieved_by: str | None = None) -> str:
    """The sha of ``snapshot_ids``: the cache key's ``snapshots_sha``, and what a declared test pins."""
    return _sha_bytes(canon(snapshot_ids(bank, tickers, prices_asof, retrieved_by)).encode())


def cache_key(bank: Bank, tickers, grid, prices_asof: str, h: int, retrieved_by: str | None = None) -> dict:
    """What a cached feature set depends on: FEATURES_VERSION, the code, the tickers, the grid, h,
    prices_asof, the price snapshots read (``retrieved_by`` as in ``prices``) and every ticker's journal shard
    (inode and size, ``signals._shard_state``). Long parts are kept as shas; ``sha`` names the whole."""
    tk = _universe(tickers)
    gr = sorted({str(d)[:10] for d in grid})
    asof = norm_known_at(prices_asof)
    every = [*tk, BENCH, VIX]
    shards = [[t, *signals._shard_state(bank, t)] for t in every]  # before reading anything
    snaps = snapshot_ids(bank, tk, asof, retrieved_by)
    key = {"features_version": FEATURES_VERSION, "features_sha": code_sha(), "deps_sha": deps_sha(),
           "names_sha": _sha_bytes(canon(FEATURES).encode()),
           "universe_sha": _sha_bytes(",".join(tk).encode()), "n_tickers": len(tk),
           "grid_sha": _sha_bytes(",".join(gr).encode()), "n_grid": len(gr),
           "grid_first": gr[0] if gr else None, "grid_last": gr[-1] if gr else None,
           "h": int(h), "prices_asof": asof, **({"retrieved_by": norm_known_at(retrieved_by)} if retrieved_by else {}),
           "snapshots_sha": _sha_bytes(canon(snaps).encode()), "shards_sha": _sha_bytes(canon(shards).encode())}
    key["sha"] = hashlib.sha256(canon(key).encode()).hexdigest()[:24]
    return key


def cache_dir() -> Path:
    return config.home() / "cache" / "forecast"


_ARRAYS = ("X", "raw", "y", "excess", "r_total", "close", "entry", "exit", "date", "ticker")
_PENDING = ("X", "raw", "date", "ticker", "close")


def _save(fs: FeatureSet) -> None:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"features-{fs.key['sha']}.npz"
    tmp = d / f"features-{fs.key['sha']}.{os.getpid()}.tmp.npz"
    meta = {"key": fs.key, "names": fs.names, "groups": fs.groups, "dropped": fs.dropped, "grid": fs.grid,
            "h": fs.h, "prices_asof": fs.prices_asof, "tickers": fs.tickers}
    arrays = {k: getattr(fs, k) for k in _ARRAYS}
    arrays.update({f"pending_{k}": fs.pending[k] for k in _PENDING})
    try:
        np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, p)
    except OSError:
        tmp.unlink(missing_ok=True)  # a cache: the features can always be computed again


def _load(key: dict) -> FeatureSet | None:
    p = cache_dir() / f"features-{key['sha']}.npz"
    try:
        with np.load(p, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        meta = json.loads(str(a.pop("meta")))
    except (OSError, ValueError, KeyError):
        return None
    if meta.get("key") != key or meta.get("names") != NAMES:
        return None
    return FeatureSet(**{k: a[k] for k in _ARRAYS}, names=meta["names"], groups=meta["groups"],
                      dropped=meta["dropped"], pending={k: a[f"pending_{k}"] for k in _PENDING}, grid=meta["grid"],
                      h=meta["h"], prices_asof=meta["prices_asof"], key=meta["key"], tickers=meta["tickers"])


_JOB: dict = {}


def _job(task):
    j, t, dates, mes = task
    return j, _ticker_rows(_JOB["root"], _JOB["use_index"], t, _JOB["book"], dates, mes)


def _workers(n_tasks: int, workers: int | None) -> int:
    if workers is None:
        workers = int(os.environ.get("BRAIN_FORECAST_WORKERS") or min(4, os.cpu_count() or 1))
    return max(1, min(int(workers), n_tasks))


def _each_ticker(bank: Bank, book: PriceBook, tasks: list, workers: int, log) -> list:
    """Run ``_ticker_rows`` for every task, in ``workers`` forked processes when more than one."""
    out = [None] * len(tasks)
    start, done = time.monotonic(), 0
    _JOB.update(root=bank.root, use_index=bank.use_index, book=book)
    try:
        if workers > 1:
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(workers) as pool:
                for j, res in pool.imap_unordered(_job, [(k, *t) for k, t in enumerate(tasks)]):
                    out[j] = res
                    done += 1
                    if log and done % 50 == 0:
                        log(f"  features: {done}/{len(tasks)} tickers read ({time.monotonic() - start:.0f}s)")
        else:
            for k, t in enumerate(tasks):
                out[k] = _job((k, *t))[1]
                if log and (k + 1) % 50 == 0:
                    log(f"  features: {k + 1}/{len(tasks)} tickers read ({time.monotonic() - start:.0f}s)")
    finally:
        _JOB.clear()
    return out


def feature_set(bank: Bank, tickers, grid, prices_asof: str, h: int, *, cache: bool = True,
                workers: int | None = None, log=None, px: Prices | None = None,
                retrieved_by: str | None = None) -> FeatureSet:
    """Features and outcomes on every grid date (SPY sessions; pass the whole grid, since R z-scores use the
    earlier grid dates). Rows are listed stocks with an outcome; listed rows whose window has not closed in
    the book go to ``pending`` (for a live forecast); the rest are counted in ``dropped``. Reads and writes
    the cache under ``<home>/cache/forecast/`` unless ``cache`` is False. ``workers`` (default
    $BRAIN_FORECAST_WORKERS, else up to 4) read tickers' journals in parallel. ``px``: the run's
    ``prices(bank, tickers, prices_asof)`` when the caller has built it already (it is not built twice; its
    ``retrieved_by`` pin is then the run's). ``retrieved_by``: as in ``prices``, when ``px`` is not given."""
    t0 = time.monotonic()
    tk = _universe(tickers)
    asof = norm_known_at(prices_asof) if prices_asof else now_iso()
    rb = px.retrieved_by if px is not None else (norm_known_at(retrieved_by) if retrieved_by else None)
    gr = sorted({str(d)[:10] for d in grid})
    key = cache_key(bank, tk, gr, asof, h, retrieved_by=rb)
    if cache:
        hit = _load(key)
        if hit is not None:
            return hit
    if px is None:
        px = prices(bank, tk, asof, retrieved_by=rb)
    elif px.asof != asof or px.tickers != tk:
        raise ValueError("px was built for other tickers or another prices_asof")
    N, F = len(tk), len(FEATURES)
    try:
        rows = [px.index(d) for d in gr]
    except KeyError as e:
        raise ValueError(f"grid date {e.args[0]} is not a SPY session in the book at {asof}") from None
    listed = np.array([px.listed(i) for i in rows]).reshape(len(rows), N)

    # Bank reads, ticker by ticker: events on the dates each is listed, fundamentals at their month-ends.
    me_rows = [month_end_index(px.dates, i) for i in rows]
    tasks, plan = [], []
    for j, t in enumerate(tk):
        gs = [g for g in range(len(gr)) if listed[g, j]]
        if not gs:
            continue
        mes = sorted({me_rows[g] for g in gs if me_rows[g] >= 0})
        tasks.append((t, [gr[g] for g in gs], [str(px.dates[m]) for m in mes]))
        plan.append((j, gs, mes))
    results = _each_ticker(bank, px.book, tasks, _workers(len(tasks), workers), log)

    raw = np.full((len(gr), N, F), np.nan)
    c_ev = [NAMES.index(k) for k in EVENT_NAMES]
    c_f = [NAMES.index(k) for k in FUND_NAMES]
    c_has = [NAMES.index(f"has_{k}") for k in FUND_NAMES]
    for (j, gs, mes), (fund, ev) in zip(plan, results):
        at = {m: k for k, m in enumerate(mes)}
        for k, g in enumerate(gs):
            raw[g, j, c_ev] = ev[k]
            fv = fund[at[me_rows[g]]] if me_rows[g] in at else np.full(len(FUND_NAMES), np.nan)
            raw[g, j, c_f] = fv
            raw[g, j, c_has] = np.isfinite(fv).astype(float)
    c_px = {n: NAMES.index(n) for n, _, _ in PRICE_FEATURES}
    c_r = [NAMES.index(k) for k in REGIME_NAMES]
    reg = np.full((len(gr), len(REGIME_NAMES)), np.nan)
    for g, i in enumerate(rows):
        for n, v in stock_features(px, i).items():
            raw[g, :, c_px[n]] = v
        r = regime_features(px, i)
        reg[g] = [r[k] for k in REGIME_NAMES]
        raw[g][:, c_r] = reg[g]
    raw[~listed] = np.nan

    X = np.zeros(raw.shape)
    for g in range(len(gr)):
        X[g] = transform(raw[g], listed[g])
    for k, c in enumerate(c_r):
        X[:, :, c] = np.where(listed, zscore_earlier(reg[:, k])[:, None], 0.0)

    # Outcomes: claims.outcome from the first SPY session after d.
    book = px.book
    bd, ba = book.dates[BENCH], book.adj[BENCH]
    keep, pend = [], []
    dropped = {"unlisted": int((~listed).sum()), "no_outcome": 0, "zero_excess": 0, "pending": 0}
    lab = {}
    for g, d in enumerate(gr):
        j0 = int(np.searchsorted(bd, d, side="right"))
        open_window = j0 + int(h) >= len(bd)
        for j in map(int, np.flatnonzero(listed[g])):
            if open_window:
                pend.append((g, j))
                continue
            o = outcome(book, tk[j], d, h)
            if o is None:
                dropped["no_outcome"] += 1
            elif o["excess"] == 0:
                dropped["zero_excess"] += 1
            else:
                k0, k1 = int(np.searchsorted(bd, o["entry"])), int(np.searchsorted(bd, o["exit"]))
                lab[(g, j)] = (o, float(o["excess"]) + (float(ba[k1]) / float(ba[k0]) - 1.0))
                keep.append((g, j))
    dropped["pending"] = len(pend)
    gi = np.array([g for g, _ in keep], dtype=int)
    ji = np.array([j for _, j in keep], dtype=int)
    exc = np.array([lab[k][0]["excess"] for k in keep], dtype=float)
    gp = np.array([g for g, _ in pend], dtype=int)
    jp = np.array([j for _, j in pend], dtype=int)
    grid_arr = np.array(gr, dtype="U10")
    tick_arr = np.array(tk, dtype=f"U{max([len(t) for t in tk] or [1])}")
    fs = FeatureSet(
        X=X[gi, ji].reshape(len(keep), F), raw=raw[gi, ji].reshape(len(keep), F).astype(np.float32),
        y=(exc > 0).astype(np.int8), excess=exc, r_total=np.array([lab[k][1] for k in keep], dtype=float),
        close=px.close[np.array(rows, dtype=int)[gi], ji] if keep else np.zeros(0),
        entry=np.array([lab[k][0]["entry"] for k in keep], dtype="U10"),
        exit=np.array([lab[k][0]["exit"] for k in keep], dtype="U10"),
        date=grid_arr[gi], ticker=tick_arr[ji], names=list(NAMES), groups=list(GROUPS), dropped=dropped,
        pending={"X": X[gp, jp].reshape(len(pend), F), "raw": raw[gp, jp].reshape(len(pend), F).astype(np.float32),
                 "date": grid_arr[gp], "ticker": tick_arr[jp],
                 "close": px.close[np.array(rows, dtype=int)[gp], jp] if pend else np.zeros(0)},
        grid=gr, h=int(h), prices_asof=asof, key=key, tickers=tk, seconds=time.monotonic() - t0)
    if cache:
        _save(fs)
    return fs

