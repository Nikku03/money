"""The scan as a trader: two years replayed session by session (docs/SCAN.md, "The trader replay").

The weekly replay (brain/scanreplay.py) read the scan at one close a week. This replay works the way a trader would,
with the same frozen W1 model and only what was known at each moment:

- **after every close** it reads the scan (each stock's range for the next five sessions);
- **at the open** it trades to its new book, after checking the market: if SPY opens 1.5% or more below the last
  close, or the VIX 15% or more above it, the book is cut to half for the day;
- **at midday** (12:30 New York) it checks the market again, and a fall or a fear jump not seen at the open cuts
  the book to half then;
- **all day** a standing stop sits at each held stock's 9-in-10 floor (the scan's 5% quantile from the last close);
  a stock stopped out is not bought again for five sessions;
- **at the close** everything is marked, and cash earns the 13-week Treasury bill rate.

Three books, each run as fixed (the old weekly reading) and as the trader, with the steps in between so each
change's effect shows on its own (POLICIES):

- **movers:** the scan's 25 widest ranges, the list the weekly replay bought (a misuse: a range is a size, not a
  direction);
- **all:** every tradable stock;
- **spy:** the S&P 500 (SPY), whose risk the scan reads as the typical range across every stock.

The rules are fixed here before the replay runs, and it is run once. It is descriptive, not a test: the scan's
declared tests are W1a-W1c and S1a. A replay is one path of one market, so a book that wins it may have been
lucky. Survivorship: the lists were chosen in 2026, so stocks delisted or bought out in between are missing, which
flatters small stocks. Paper only: nothing here trades.
"""
from __future__ import annotations

import gc
import gzip
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote as urlquote
from zoneinfo import ZoneInfo

import numpy as np

from brain import config, scan, shocks
from brain.bank import Bank, canon
from brain.net import fetch

# ---------- the rules, fixed before the replay runs ----------
TOP = 25                    # the movers book: the scan's 25 widest ranges
KEEP = 50                   # a mover already held stays while it is among the 50 widest (the trader's books)
FALL = 0.015                # the market check: SPY this far below the last close ...
FEAR = 0.15                 # ... or the VIX this far above it
CUT = 0.5                   # cuts the book to half until the next open
COOL = 5                    # sessions a stock stopped out is not bought again
STOP_SLIP = 0.005           # a stop fills this far below its level (and never below the day's low)
REF_DAYS = 250              # the book's usual risk: the median over the last 250 sessions' scans
STALE = 5                   # a stock with no trade for this many sessions is sold at its last price
COSTS = ((100e6, 0.0005), (20e6, 0.001), (5e6, 0.002), (0.0, 0.004))   # a side, by dollar volume a day
SPY_COST = 0.0001
ONE_BAND = 0.10             # one trade a day: SPY is traded to its target only when this far off it (or on a check)
CHART_1H = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=2y&interval=1h"
MIDDAY = (12, 30)           # New York time of the midday check: the last price of the bar that ended by then
FIRST = "2024-11-11"        # the first session with Yahoo's hourly prices (they reach back two years)
NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Policy:
    key: str
    book: str               # "movers", "all" or "spy"
    label: str
    daily: bool = False     # read the scan after every close (else after each week's last close)
    risk: bool = False      # size each stock by its range and scale the book to its usual risk
    stops: bool = False     # standing stops at each stock's 9-in-10 floor
    market: bool = False    # the open and midday market checks
    one: bool = False       # one order a day at most (simulate_one)


def _steps(book: str, p: str) -> tuple[Policy, ...]:
    return (Policy(f"{p}0", book, "fixed: read once a week"),
            Policy(f"{p}1", book, "+ read after every close", daily=True),
            Policy(f"{p}2", book, "+ sized by risk", daily=True, risk=True),
            Policy(f"{p}3", book, "+ standing stops", daily=True, risk=True, stops=True),
            Policy(f"{p}4", book, "+ market checks: the trader", daily=True, risk=True, stops=True, market=True))


POLICIES = (*_steps("movers", "M"), *_steps("all", "A"),
            Policy("S0", "spy", "fixed: bought and held"),
            Policy("S1", "spy", "+ sized by risk, read after every close", daily=True, risk=True),
            Policy("S2", "spy", "+ market checks: the trader", daily=True, risk=True, market=True))
ONE_POLICIES = (Policy("M5", "movers", "one trade a day at most", daily=True, stops=True, market=True, one=True),
                Policy("S3", "spy", "one trade a day at most", daily=True, risk=True, market=True, one=True))
ONE_COMPARE = ("M0", "M4", "S0", "S2")      # run beside them: the fixed books and the trader with no limit
PAIRS = (("movers", "M0", "M4"), ("all", "A0", "A4"), ("spy", "S0", "S2"))
BOOKS = {"movers": f"the scan's {TOP} widest ranges", "all": "every tradable stock", "spy": "the S&P 500 (SPY)"}


# ---------- midday prices ----------
def midday_path() -> Path:
    return config.ROOT / "var" / "cache" / "scantrader" / "midday.json.gz"


def parse_midday(body: bytes) -> dict[str, float]:
    """Each day's price at the midday check, from an hourly chart: the close of the last hourly bar that ended at
    or before 12:30 New York time (the 11:30 bar for a stock; the 11:00 bar for the VIX, whose bars run on the
    hour)."""
    res = json.loads(body)["chart"]["result"][0]
    ts = res.get("timestamp") or []
    closes = (((res.get("indicators") or {}).get("quote") or [{}])[0].get("close")) or []
    out: dict[str, float] = {}
    for t, c in zip(ts, closes):
        if c is None or not c > 0:
            continue
        start = datetime.fromtimestamp(t, UTC).astimezone(NY)
        end = start + timedelta(hours=1)
        if end.date() == start.date() and (end.hour, end.minute) <= MIDDAY:
            out[start.date().isoformat()] = float(c)          # later bars overwrite: the last one by 12:30
    return out


def load_midday() -> dict[str, dict[str, float]]:
    p = midday_path()
    return json.loads(gzip.decompress(p.read_bytes())) if p.exists() else {}


def fetch_midday(symbols: list[str], log=print, every: int = 100) -> dict[str, dict[str, float]]:
    """Midday prices for ``symbols`` (two years of Yahoo's hourly chart each), kept in var/ (Yahoo's data is never
    committed). Resumes: symbols already kept are not fetched again."""
    have = load_midday()
    todo = [s for s in symbols if s not in have]
    for k, s in enumerate(todo):
        try:
            have[s] = parse_midday(fetch(CHART_1H.format(sym=urlquote(s)), ttl=7 * 86400).body)
        except Exception:  # noqa: BLE001 - one symbol's outage must not stop the sweep
            have[s] = {}
        if (k + 1) % every == 0 or k + 1 == len(todo):
            _save_midday(have)
            if log:
                log(f"  midday prices: {k + 1}/{len(todo)}")
    return have


def _save_midday(have: dict) -> None:
    p = midday_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(have, separators=(",", ":")).encode()))
    tmp.replace(p)


# ---------- the data: every session, every stock ----------
@dataclass
class Data:
    """Sessions (rows) by stocks (columns), with SPY as the last column. Prices are total-return prices (the
    day's dividend adjustment applied to its open, low, midday price and close), NaN where there was no trade."""
    dates: np.ndarray       # (n,) "YYYY-MM-DD"
    names: list[str]        # (m,) the last is SPY
    open: np.ndarray
    low: np.ndarray
    mid: np.ndarray         # the midday price (NaN when the hourly chart has none)
    close: np.ndarray
    has: np.ndarray         # a bar that session
    dv: np.ndarray          # mean dollar volume over the last 20 sessions, at that close
    q: np.ndarray           # (n, m, 5) the scan's quantiles read at that close (log returns); NaN where not tradable
    l5: np.ndarray          # the five-session log return from that close (NaN while the window is open)
    vix_open: np.ndarray
    vix_mid: np.ndarray
    vix_close: np.ndarray
    rf: np.ndarray          # cash's return overnight into each session
    week_end: np.ndarray    # each completed week's last session

    @property
    def spy(self) -> int:
        return len(self.names) - 1


def _frame_cols(df, cal: np.ndarray) -> dict[str, np.ndarray]:
    """A stock's own bars on the calendar, not carried over missing days."""
    import pandas as pd
    out = {}
    for c in ("open", "high", "low", "close", "adjclose", "volume"):
        s = pd.Series(df[c].to_numpy(float), index=df["date"].to_numpy(dtype=str))
        out[c] = s[~s.index.duplicated(keep="last")].reindex(cal).to_numpy()
    return out


def _prices(cols: dict, mids: dict[str, float], cal: np.ndarray) -> dict[str, np.ndarray]:
    """Total-return open, low, midday and close, the bar flag and the dollar volume, from a stock's own bars."""
    import pandas as pd
    raw_c = cols["close"]
    has = np.isfinite(raw_c) & (raw_c > 0)
    with np.errstate(all="ignore"):
        fac = np.where(has, cols["adjclose"] / raw_c, np.nan)
        mid = np.array([mids.get(str(d), np.nan) for d in cal], float)
        mid = np.where((mid >= cols["low"] * 0.98) & (mid <= cols["high"] * 1.02), mid, np.nan)   # else a bad print
        traded = has & (np.nan_to_num(cols["volume"]) > 0)
        dv = pd.Series(np.where(traded, raw_c * np.nan_to_num(cols["volume"]), 0.0)).rolling(20, min_periods=1).mean()
    return {"open": cols["open"] * fac, "low": cols["low"] * fac, "mid": mid * fac, "close": cols["adjclose"],
            "has": has, "dv": dv.to_numpy()}


def _rf(cal: np.ndarray) -> np.ndarray:
    """Cash's return overnight into each session: the 13-week bill rate (^IRX, percent a year) at the last close."""
    from brain.sources import prices
    try:
        df = prices.fetch_daily("^IRX", start=str(cal[0]), ttl=86400)[0]
    except Exception:  # noqa: BLE001
        return np.zeros(len(cal))
    rate = shocks._aligned(df, "close", cal)
    rate = np.where(np.isfinite(rate), rate, np.nan)
    import pandas as pd
    rate = pd.Series(rate).ffill().bfill().to_numpy()
    return np.r_[0.0, rate[:-1] / 100 / 252]


def load(bank: Bank, tickers: list[str], first: str, last: str, warm: int = REF_DAYS + 10, log=print) -> Data:
    """Every session from ``warm`` sessions before ``first`` to ``last``: each stock's scan and prices, SPY's and the
    VIX's, with what was known at each close only (scan.ticker_frame reads the bank as of then)."""
    model = scan.load()
    if model is None:
        raise SystemExit("no fitted model: run python -m brain scan fit")
    w1 = model["w1"]
    mids = load_midday()
    v = bank.view()
    mk = shocks.market(v)
    if mk is None:
        raise SystemExit(f"no {shocks.BENCH} or {shocks.VIX} prices in the bank")
    cal_all = mk[0]
    i1 = int(np.searchsorted(cal_all, last, side="right")) - 1
    i0 = max(int(np.searchsorted(cal_all, first)) - warm, 0)
    cal = cal_all[i0:i1 + 1]
    n, m = len(cal), len(tickers) + 1
    blank = lambda: np.full((n, m), np.nan)
    D = Data(dates=cal.astype(str), names=[*tickers, shocks.BENCH], open=blank(), low=blank(), mid=blank(),
             close=blank(), has=np.zeros((n, m), bool), dv=blank(), q=np.full((n, m, len(scan.QS)), np.nan,
             np.float32), l5=blank(), vix_open=np.full(n, np.nan), vix_mid=np.full(n, np.nan),
             vix_close=np.full(n, np.nan), rf=_rf(cal), week_end=np.zeros(n, bool))
    wk = shocks.week_ends(cal_all)
    D.week_end[wk[(wk >= i0) & (wk <= i1)] - i0] = True
    p = _prices(_frame_cols(shocks._series(v, shocks.BENCH), cal), mids.get(shocks.BENCH, {}), cal)
    for k in ("open", "low", "mid", "close", "has", "dv"):
        getattr(D, k)[:, m - 1] = p[k]
    raw = _frame_cols(shocks._series(v, shocks.VIX), cal)
    D.vix_open, D.vix_close = raw["open"], raw["close"]
    D.vix_mid = np.array([mids.get(shocks.VIX, {}).get(str(d), np.nan) for d in cal], float)
    del v
    for c in range(0, len(tickers), scan.CHUNK):
        b = Bank(bank.root)
        v = b.view()
        mk = shocks.market(v)
        for j, t in enumerate(tickers[c:c + scan.CHUNK], start=c):
            got = scan.ticker_frame(v, t, mk)
            if not got:
                continue
            f = got[1]
            X = np.column_stack([f[k][i0:i1 + 1] for k in scan.FEATS])
            ok = np.isfinite(X).all(1) & f["ok"][i0:i1 + 1].astype(bool)
            if ok.any():
                D.q[ok, j, :] = scan.quantiles(w1, X[ok]).astype(np.float32)
            D.l5[:, j] = f["l5"][i0:i1 + 1]
            p = _prices(_frame_cols(shocks._series(v, t), cal), mids.get(t, {}), cal)
            for k in ("open", "low", "mid", "close", "has", "dv"):
                getattr(D, k)[:, j] = p[k]
        if log:
            log(f"  {min(c + scan.CHUNK, len(tickers))}/{len(tickers)} stocks loaded")
        del b, v
        gc.collect()
    return D


# ---------- the trader ----------
def _hm(x: np.ndarray) -> float:
    x = x[np.isfinite(x) & (x > 0)]
    return float(len(x) / np.sum(1 / x)) if len(x) else float("nan")


def readings(D: Data) -> dict[str, np.ndarray]:
    """What each close's scan says for sizing: each stock's half range (price terms) and log half-width, and each
    book's risk (the harmonic mean of its stocks' half-widths: what a book sized by risk carries per stock) with
    its usual level, the median over the previous REF_DAYS scans."""
    q = D.q.astype(float)
    half = (np.exp(q[:, :, 3]) - np.exp(q[:, :, 1])) / 2
    sig = (q[:, :, 3] - q[:, :, 1]) / 2
    n = len(D.dates)
    risk = {"movers": np.full(n, np.nan), "all": np.full(n, np.nan)}
    for d in range(n):
        ok = np.flatnonzero(np.isfinite(half[d]))
        if len(ok):
            top = ok[np.argsort(-half[d, ok], kind="mergesort")[:TOP]]
            risk["movers"][d], risk["all"][d] = _hm(sig[d, top]), _hm(sig[d, ok])
    usual = {}
    for k, r in risk.items():
        u = np.full(n, np.nan)
        for d in range(n):
            past = r[max(d - REF_DAYS, 0):d]
            past = past[np.isfinite(past)]
            if len(past) >= REF_DAYS // 2:
                u[d] = float(np.median(past))
        usual[k] = u
    risk["spy"], usual["spy"] = risk["all"], usual["all"]
    return {"half": half, "sig": sig, "risk": risk, "usual": usual}


def cost_side(dv: np.ndarray, spy: int) -> np.ndarray:
    c = np.full(len(dv), COSTS[-1][1])
    for floor, cost in reversed(COSTS):
        c = np.where(np.nan_to_num(dv) >= floor, cost, c)
    c[spy] = SPY_COST
    return c


def target(D: Data, R: dict, p: Policy, s: int, d: int, held: np.ndarray, cool: np.ndarray,
           block: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """The book the scan read at close ``s`` asks for at session ``d``'s open: weights summing to one, and how
    much of the money to put in (gross). ``block``: stocks not to buy today (already through their floor)."""
    m = len(D.names)
    w = np.zeros(m)
    if p.book == "spy":
        w[D.spy] = 1.0
        idx = np.array([D.spy])
    else:
        half = R["half"][s]
        ok = np.flatnonzero(np.isfinite(half) & (cool < d) & (True if block is None else ~(block & ~held)))
        ok = ok[ok != D.spy]
        if p.book == "movers":
            order = ok[np.argsort(-half[ok], kind="mergesort")]
            if p.daily:
                stay = [i for i in order[:KEEP] if held[i]]
                fresh = [i for i in order if not held[i]][:max(TOP - len(stay), 0)]
                idx = np.array(sorted(set(stay) | set(fresh)), int)
            else:
                idx = order[:TOP]
        else:
            idx = ok
        if not len(idx):
            return w, 0.0
        w[idx] = 1.0 / R["sig"][s, idx] if p.risk else 1.0
        w /= w.sum()
    gross = 1.0
    if p.risk:
        now = R["risk"][p.book][s] if p.book == "spy" else _hm(R["sig"][s, idx])
        usual = R["usual"][p.book][s]
        gross = min(1.0, usual / now) if np.isfinite(usual) and np.isfinite(now) and now > 0 else 1.0
    return w, gross


def simulate(D: Data, R: dict, p: Policy, start: int, end: int, log_trades: bool = False) -> dict:
    """One book through sessions ``start``..``end``, from $1 of cash at the close before ``start``."""
    m = len(D.names)
    val, lp, entry = np.zeros(m), np.full(m, np.nan), np.full(m, np.nan)
    stop, cool, last_bar = np.full(m, np.nan), np.full(m, -1), np.full(m, -1)
    acct = {"cash": 1.0, "costs": 0.0, "turnover": 0.0}
    nav, gross_log, flags, trades = [], [], [], []
    counts = {"stops": 0, "cuts_open": 0, "cuts_midday": 0, "buys": 0, "sells": 0, "stale": 0, "orders": 0}
    per_day: list[int] = []

    def mark(px: np.ndarray) -> None:
        who = (val != 0) & np.isfinite(px) & np.isfinite(lp)
        val[who] *= px[who] / lp[who]
        lp[who] = px[who]

    def move(idx: np.ndarray, new: np.ndarray, px: np.ndarray, c: np.ndarray, extra: float = 0.0) -> None:
        """Trade the stocks ``idx`` to the values ``new`` at the prices ``px`` (val already marked to px)."""
        if not len(idx):
            return
        dlt = new - val[idx]
        counts["orders"] += int(np.sum(np.abs(dlt) > 1e-12))
        cost = np.abs(dlt) * (c[idx] + extra)
        acct["cash"] -= float(dlt.sum() + cost.sum())
        acct["costs"] += float(cost.sum())
        acct["turnover"] += float(np.abs(dlt).sum())
        opened = (val[idx] == 0) & (new > 0)
        entry[idx[opened]] = px[opened]
        val[idx] = new
        lp[idx] = px

    def note(d: int, when: str, i: int | None, what: str, why: str, ret: float | None = None) -> None:
        if log_trades:
            trades.append({"date": str(D.dates[d]), "when": when, "stock": D.names[i] if i is not None else "",
                           "action": what, "why": why, **({"return": round(float(ret), 4)} if ret is not None else {})})

    def sell(d: int, i: int, px: float, c: np.ndarray, when: str, why: str, extra: float = 0.0) -> None:
        if np.isfinite(lp[i]) and lp[i] > 0:
            val[i] *= px / lp[i]
        ret = px / entry[i] - 1 if np.isfinite(entry[i]) and entry[i] > 0 else None
        move(np.array([i]), np.array([0.0]), np.array([px]), c, extra)
        counts["sells"] += 1
        note(d, when, i, "sell", why, ret)

    def stopped(d: int, i: int, px: float, c: np.ndarray, when: str, s: int) -> None:
        sell(d, i, px, c, when, f"fell through its 9-in-10 floor ({stop[i] / D.close[s, i] - 1:+.1%} from the "
             "last close)")
        cool[i] = d + COOL
        counts["stops"] += 1

    def check(spy_px: float, vix_px: float, s: int) -> str | None:
        a, b = spy_px / D.close[s, D.spy] - 1, vix_px / D.vix_close[s] - 1
        return f"SPY {a * 100:+.1f}%, the VIX {b * 100:+.0f}%" if (a <= -FALL or b >= FEAR) else None

    for d in range(start, end + 1):
        per_day.append(counts["orders"])
        s = d - 1
        c = cost_side(D.dv[s], D.spy)
        acct["cash"] *= 1 + D.rf[d]
        last_bar[D.has[d]] = d
        if p.stops:                                   # the floors the last close's scan set
            fresh = D.close[s] * np.exp(D.q[s, :, 0].astype(float))
            stop = np.where(np.isfinite(fresh), fresh, stop)
        for i in np.flatnonzero((val > 0) & (d - last_bar >= STALE)):
            sell(d, i, lp[i], c, "open", f"no trade for {STALE} sessions: sold at its last price")
            counts["stale"] += 1
        # the open
        po = D.open[d]
        mark(po)
        flag = check(D.open[d, D.spy], D.vix_open[d], s) if p.market else None
        if flag:
            counts["cuts_open"] += 1
            flag = f"{flag} at the open"
        if p.stops:
            for i in np.flatnonzero((val > 0) & np.isfinite(stop) & D.has[d] & (po <= stop)):
                stopped(d, i, po[i], c, "open", s)
        if p.book == "spy" and not p.daily:
            rebalance = d == start
        else:
            rebalance = d == start or p.daily or bool(D.week_end[s])
        if rebalance:
            block = (np.isfinite(stop) & np.isfinite(po) & (po <= stop)) if p.stops else None
            w, gross = target(D, R, p, s, d, val > 0, cool, block)
            if flag:
                gross *= CUT
            tv = (acct["cash"] + val.sum()) * gross * w
            can = D.has[d] & np.isfinite(po)
            tv = np.where(can, tv, val)                # what cannot trade today stays as it is
            for i in np.flatnonzero(can & (tv == 0) & (val > 0)):
                why = ("not tradable at the close" if not np.isfinite(R["half"][s, i])
                       else f"no longer among the {KEEP if p.daily else TOP} widest ranges")
                sell(d, i, po[i], c, "open", why)
            new = np.flatnonzero(can & (tv > 0) & (val == 0))
            for i in new:
                counts["buys"] += 1
                if p.book != "all":
                    lo, hi = np.expm1(D.q[s, i, 1]), np.expm1(D.q[s, i, 3])
                    note(d, "open", i, "buy", "the S&P 500" if i == D.spy else
                         f"among the {TOP} widest ranges at {D.dates[s]}'s close (2 in 3: {lo:+.0%}..{hi:+.0%})")
            idx = np.flatnonzero(can & ((tv > 0) | (val > 0)))
            move(idx, tv[idx], po[idx], c)
            if flag:
                note(d, "open", None, "cut", f"{flag}: the book cut to half")
        elif flag:
            idx = np.flatnonzero((val > 0) & D.has[d] & np.isfinite(po))
            move(idx, val[idx] * CUT, po[idx], c)
            note(d, "open", None, "cut", f"{flag}: the book cut to half")
        # midday: stops already passed, then the market check
        pm = D.mid[d]
        if p.stops:
            for i in np.flatnonzero((val > 0) & np.isfinite(stop) & np.isfinite(pm) & (pm <= stop)):
                lo = D.low[d, i]
                stopped(d, i, max(stop[i] * (1 - STOP_SLIP), lo) if np.isfinite(lo) else stop[i] * (1 - STOP_SLIP),
                        c, "morning", s)
        late = np.zeros(m, bool)
        if p.market and not flag:
            flag = check(D.mid[d, D.spy], D.vix_mid[d], s)
            if flag:
                counts["cuts_midday"] += 1
                flag = f"{flag} at midday"
                held = val > 0
                mark(pm)
                idx = np.flatnonzero(held & np.isfinite(pm))
                move(idx, val[idx] * CUT, pm[idx], c)
                late = held & ~np.isfinite(pm)        # no midday price: cut at the close
                note(d, "midday", None, "cut", f"{flag}: the book cut to half")
        # the afternoon: standing stops not hit by midday
        if p.stops:
            lo = D.low[d]
            for i in np.flatnonzero((val > 0) & np.isfinite(stop) & D.has[d] & np.isfinite(lo) & (lo <= stop)):
                stopped(d, i, max(stop[i] * (1 - STOP_SLIP), lo[i]), c, "afternoon", s)
        # the close
        pc = D.close[d]
        mark(pc)
        idx = np.flatnonzero(late & (val > 0) & np.isfinite(pc))
        move(idx, val[idx] * CUT, pc[idx], c)
        total = acct["cash"] + val.sum()
        nav.append(total)
        gross_log.append(float(val.sum() / total) if total > 0 else 0.0)
        flags.append(flag or "")
    per_day = np.diff(np.r_[per_day, counts["orders"]])
    counts["most_orders_a_day"] = int(per_day.max()) if len(per_day) else 0
    return {"key": p.key, "nav": np.array(nav), "gross": np.array(gross_log), "flags": flags, "costs": acct["costs"],
            "turnover": acct["turnover"], "counts": counts, "trades": trades}


def simulate_one(D: Data, R: dict, p: Policy, start: int, end: int, log_trades: bool = False) -> dict:
    """One book placing one order a day at most. Each session the trader picks its single most important order:

    - **movers:** a holding that opened through its floor is sold; else a holding no longer among the KEEP widest
      ranges (or not tradable) is sold; else, with a slot free and no market check at the open, the widest range
      not held is bought with a TOP-th of the account. With no order by then, a standing stop may sell the first
      holding to fall through its floor during the session. Bought stocks are never resized.
    - **spy:** a market check at the open cuts the holding to half; else the holding is traded to the scan's
      target (sizing by risk) when it is ONE_BAND or more away from it; with no order by then, a market check at
      midday cuts it to half.
    """
    m = len(D.names)
    val, lp, entry = np.zeros(m), np.full(m, np.nan), np.full(m, np.nan)
    stop, cool, last_bar = np.full(m, np.nan), np.full(m, -1), np.full(m, -1)
    acct = {"cash": 1.0, "costs": 0.0, "turnover": 0.0}
    nav, gross_log, flags, trades, per_day = [], [], [], [], []
    counts = {"stops": 0, "cuts_open": 0, "cuts_midday": 0, "buys": 0, "sells": 0, "stale": 0, "orders": 0}

    def mark(px: np.ndarray) -> None:
        who = (val != 0) & np.isfinite(px) & np.isfinite(lp)
        val[who] *= px[who] / lp[who]
        lp[who] = px[who]

    def order(d: int, when: str, i: int, new: float, px: float, c: np.ndarray, why: str) -> None:
        if np.isfinite(lp[i]) and lp[i] > 0 and val[i] != 0:
            val[i] *= px / lp[i]
        dlt = new - val[i]
        cost = abs(dlt) * c[i]
        acct["cash"] -= dlt + cost
        acct["costs"] += cost
        acct["turnover"] += abs(dlt)
        ret = px / entry[i] - 1 if new == 0 and np.isfinite(entry[i]) and entry[i] > 0 else None
        if val[i] == 0 and new > 0:
            entry[i] = px
            counts["buys"] += 1
        elif new == 0:
            counts["sells"] += 1
        val[i], lp[i] = new, px
        counts["orders"] += 1
        if log_trades:
            what = "buy" if dlt > 0 else "sell"
            trades.append({"date": str(D.dates[d]), "when": when, "stock": D.names[i], "action": what, "why": why,
                           **({"return": round(float(ret), 4)} if ret is not None else {})})

    def check(spy_px: float, vix_px: float, s: int) -> str | None:
        a, b = spy_px / D.close[s, D.spy] - 1, vix_px / D.vix_close[s] - 1
        return f"SPY {a * 100:+.1f}%, the VIX {b * 100:+.0f}%" if (a <= -FALL or b >= FEAR) else None

    for d in range(start, end + 1):
        s = d - 1
        c = cost_side(D.dv[s], D.spy)
        acct["cash"] *= 1 + D.rf[d]
        last_bar[D.has[d]] = d
        if p.stops:
            fresh = D.close[s] * np.exp(D.q[s, :, 0].astype(float))
            stop = np.where(np.isfinite(fresh), fresh, stop)
        po, pm, lo = D.open[d], D.mid[d], D.low[d]
        mark(po)
        done, before = False, counts["orders"]
        flag = check(D.open[d, D.spy], D.vix_open[d], s) if p.market else None
        if flag:
            counts["cuts_open"] += 1
            flag = f"{flag} at the open"
        held = val > 0
        gone = np.flatnonzero(held & (d - last_bar >= STALE))
        if len(gone):                                      # a stock that stopped trading: the day's order
            order(d, "open", int(gone[0]), 0.0, lp[gone[0]], c, f"no trade for {STALE} sessions: sold at its last price")
            counts["stale"] += 1
            done = True
        if p.book == "spy":
            i = D.spy
            tot = acct["cash"] + val.sum()
            cur = val[i] / tot if tot > 0 else 0.0
            if not done and flag and cur > 0 and np.isfinite(po[i]):
                order(d, "open", i, val[i] * CUT, po[i], c, f"{flag}: the holding cut to half")
                done = True
            elif not done and np.isfinite(po[i]):
                gross = target(D, R, p, s, d, held, cool)[1]
                if abs(gross - cur) >= ONE_BAND:
                    order(d, "open", i, tot * gross, po[i], c,
                          f"to the scan's target: {gross:.0%} of the account (from {cur:.0%})")
                    done = True
            if not done and p.market and val[i] > 0 and np.isfinite(pm[i]):
                fl = check(pm[i], D.vix_mid[d], s)
                if fl:
                    counts["cuts_midday"] += 1
                    order(d, "midday", i, val[i] * pm[i] / lp[i] * CUT, pm[i], c, f"{fl} at midday: the holding cut to half")
                    flag, done = f"{fl} at midday", True
        else:
            can = D.has[d] & np.isfinite(po)
            if not done and p.stops:
                gap = np.flatnonzero(held & can & np.isfinite(stop) & (po <= stop))
                if len(gap):
                    i = int(gap[np.argmin(po[gap] / stop[gap])])
                    order(d, "open", i, 0.0, po[i], c, f"opened through its 9-in-10 floor ({stop[i] / D.close[s, i] - 1:+.1%})")
                    cool[i] = d + COOL
                    counts["stops"] += 1
                    done = True
            half = R["half"][s]
            ok = np.flatnonzero(np.isfinite(half))
            ok = ok[ok != D.spy]
            ranked = ok[np.argsort(-half[ok], kind="mergesort")]
            rank = np.full(m, np.inf)
            rank[ranked] = np.arange(len(ranked))
            if not done:
                out = np.flatnonzero(held & can & (rank >= KEEP))
                if len(out):
                    i = int(out[np.argmax(rank[out])])
                    order(d, "open", i, 0.0, po[i], c, "not tradable at the close" if not np.isfinite(rank[i])
                          else f"no longer among the {KEEP} widest ranges")
                    done = True
            if not done and not flag and np.sum(val > 0) < TOP:
                tot = acct["cash"] + val.sum()
                size = min(acct["cash"], tot / TOP)
                if size >= 0.5 * tot / TOP:
                    for i in ranked:
                        if val[i] > 0 or cool[i] >= d or not can[i] or (p.stops and np.isfinite(stop[i]) and po[i] <= stop[i]):
                            continue
                        lo68, hi68 = np.expm1(D.q[s, i, 1]), np.expm1(D.q[s, i, 3])
                        order(d, "open", int(i), size, po[i], c, f"the widest range not held at {D.dates[s]}'s close "
                              f"(2 in 3: {lo68:+.0%}..{hi68:+.0%})")
                        done = True
                        break
            if not done and p.stops:                       # a standing stop: the first holding through its floor
                hit = np.flatnonzero((val > 0) & D.has[d] & np.isfinite(stop) & np.isfinite(lo) & (lo <= stop))
                if len(hit):
                    morning = hit[np.isfinite(pm[hit]) & (pm[hit] <= stop[hit])]
                    i = int(morning[np.argmin(pm[morning] / stop[morning])] if len(morning)
                            else hit[np.argmin(lo[hit] / stop[hit])])
                    when = "morning" if len(morning) else "afternoon"
                    order(d, when, i, 0.0, max(stop[i] * (1 - STOP_SLIP), lo[i]), c,
                          f"fell through its 9-in-10 floor ({stop[i] / D.close[s, i] - 1:+.1%})")
                    cool[i] = d + COOL
                    counts["stops"] += 1
        mark(D.close[d])
        total = acct["cash"] + val.sum()
        nav.append(total)
        gross_log.append(float(val.sum() / total) if total > 0 else 0.0)
        flags.append(flag or "")
        per_day.append(counts["orders"] - before)
    counts["most_orders_a_day"] = int(max(per_day)) if per_day else 0
    return {"key": p.key, "nav": np.array(nav), "gross": np.array(gross_log), "flags": flags, "costs": acct["costs"],
            "turnover": acct["turnover"], "counts": counts, "trades": trades}


# ---------- reading the results ----------
def stats(nav: np.ndarray, spy: np.ndarray, rf: np.ndarray, gross: np.ndarray | None = None) -> dict:
    """$1's path: where it ended, a year's growth, swings, worst fall, Sharpe, beta to SPY, and the end without
    its five best sessions."""
    r = np.diff(np.r_[1.0, nav]) / np.r_[1.0, nav[:-1]]
    rs = np.diff(np.r_[1.0, spy]) / np.r_[1.0, spy[:-1]]
    yrs = len(r) / 252
    peak = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    ex = r - rf[:len(r)]
    beta = float(np.cov(r, rs)[0, 1] / np.var(rs, ddof=1)) if len(r) > 2 else float("nan")
    best5 = np.sort(r)[-5:] if len(r) >= 5 else r
    return {"end": float(nav[-1]), "cagr": float(nav[-1] ** (1 / yrs) - 1) if yrs > 0 and nav[-1] > 0 else float("nan"),
            "vol": float(r.std(ddof=1) * math.sqrt(252)), "worst_fall": float((nav / peak - 1).min()),
            "sharpe": float(ex.mean() / ex.std(ddof=1) * math.sqrt(252)) if ex.std(ddof=1) > 0 else float("nan"),
            "beta": beta, "best_day": float(r.max()), "worst_day": float(r.min()),
            "end_without_best5": float(nav[-1] / np.prod(1 + best5)),
            "exposure": float(np.mean(gross)) if gross is not None else 1.0}


def accuracy(D: Data, start: int) -> dict:
    """The ranges read after every close against the week's fixed reading on the same five sessions: how often each
    held, and after the days the market moved most (when the fixed reading is most out of date)."""
    q = D.q.astype(float)
    n = len(D.dates)
    we = np.flatnonzero(D.week_end)
    out = {"all": [], "midweek": [], "after_big_day": []}
    spy = D.close[:, D.spy]
    big = np.r_[False, np.abs(spy[1:] / spy[:-1] - 1) >= 0.02]
    vixj = np.r_[False, D.vix_close[1:] / D.vix_close[:-1] - 1 >= FEAR]
    for d in range(start, n):
        k = np.searchsorted(we, d, side="right") - 1
        if k < 0:
            continue
        w = we[k]
        y = D.l5[d]
        ok = np.isfinite(y) & np.isfinite(q[d, :, 0]) & np.isfinite(q[w, :, 0])
        if not ok.any():
            continue
        row = []
        for qq in (q[d][ok], q[w][ok]):
            yy = y[ok]
            row.append((np.sum((yy >= qq[:, 1]) & (yy <= qq[:, 3])), np.sum((yy >= qq[:, 0]) & (yy <= qq[:, 4])),
                        float(np.sum(_pinball(qq, yy)))))
        rec = (int(ok.sum()), row)
        out["all"].append(rec)
        if w != d:
            out["midweek"].append(rec)
            if big[d] or vixj[d] or any(big[w + 1:d + 1]) or any(vixj[w + 1:d + 1]):
                out["after_big_day"].append(rec)
    res = {}
    for k, recs in out.items():
        n_ = sum(r[0] for r in recs)
        if not n_:
            continue
        f68, f90, fl = (sum(r[1][0][j] for r in recs) for j in range(3))
        s68, s90, sl = (sum(r[1][1][j] for r in recs) for j in range(3))
        res[k] = {"readings": n_, "sessions": len(recs), "fresh68": f68 / n_, "fresh90": f90 / n_,
                  "fixed68": s68 / n_, "fixed90": s90 / n_, "fresh_loss": fl / n_, "fixed_loss": sl / n_}
    return res


def _pinball(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    dlt = y[:, None] - q
    a = np.array(scan.QS)[None, :]
    return np.maximum(a * dlt, (a - 1) * dlt).mean(1)


def checks(D: Data, start: int, end: int) -> dict:
    """What the market checks and the opening gaps said, against the five sessions that followed: on days the check
    fired, how many stocks broke the range read at the last close, against the other days."""
    q = D.q.astype(float)
    spy, rows = D.close[:, D.spy], {"open": [], "midday": [], "none": []}
    gap = {"outside68_at_open": [0, 0], "base": [0, 0]}
    for d in range(start, end + 1):
        s = d - 1
        o = D.open[d, D.spy] / spy[s] - 1 <= -FALL or D.vix_open[d] / D.vix_close[s] - 1 >= FEAR
        md = D.mid[d, D.spy] / spy[s] - 1 <= -FALL or D.vix_mid[d] / D.vix_close[s] - 1 >= FEAR
        kind = "open" if o else "midday" if md else "none"
        y = D.l5[s]
        ok = np.isfinite(y) & np.isfinite(q[s, :, 0])
        out90 = float(np.mean((y[ok] < q[s, ok, 0]) | (y[ok] > q[s, ok, 4]))) if ok.any() else float("nan")
        rows[kind].append((str(D.dates[d]), out90, int(ok.sum())))
        if ok.any():
            with np.errstate(all="ignore"):
                g = np.log(D.open[d] / D.close[s])
            o68 = ok & np.isfinite(g) & ((g < q[s, :, 1]) | (g > q[s, :, 3]))
            br = (y < q[s, :, 0]) | (y > q[s, :, 4])
            gap["outside68_at_open"][0] += int(np.sum(o68 & br))
            gap["outside68_at_open"][1] += int(np.sum(o68))
            gap["base"][0] += int(np.sum(ok & br))
            gap["base"][1] += int(np.sum(ok))
    res = {k: {"days": len(v), "broke90": float(np.nanmean([x[1] for x in v])) if any(x[2] for x in v) else float("nan"),
               "judged": sum(1 for x in v if x[2]), "dates": [x[0] for x in v] if k != "none" else []}
           for k, v in rows.items()}
    res["gaps"] = {k: {"broke90": v[0] / v[1] if v[1] else float("nan"), "readings": v[1]} for k, v in gap.items()}
    return res


def run(bank: Bank, first: str = FIRST, last: str = "2026-12-31", tickers=None, log=print, data: Data | None = None,
        policies: tuple[Policy, ...] = POLICIES, tag: str = "") -> dict:
    """The replay of ``policies`` (each needs S0 beside it, the yardstick for beta). ``tag`` names the saved file."""
    tickers = tickers or scan.scan_tickers()
    missing = [t for t in [shocks.BENCH, shocks.VIX, *tickers] if t not in load_midday()]
    if missing and data is None:
        log(f"fetching midday prices for {len(missing)} symbols (Yahoo's hourly chart, two years each)")
        fetch_midday(missing, log=log)
    D = data if data is not None else load(bank, tickers, first, last, log=log)
    start = int(np.searchsorted(D.dates, first))
    end = len(D.dates) - 1
    R = readings(D)
    spy_nav = None
    out = {}
    for p in policies:
        sim = simulate_one if p.one else simulate
        r = sim(D, R, p, start, end, log_trades=p.one or p.key in ("M0", "M4", "A4", "S2"))
        out[p.key] = r
        if p.key == "S0":
            spy_nav = r["nav"]
    rf = D.rf[start:end + 1]
    res = {"first": str(D.dates[start]), "last": str(D.dates[end]), "sessions": end - start + 1,
           "stocks": len(D.names) - 1, "w1_sha": scan.model_sha(), "rules": rules(policies), "tag": tag,
           "policies": [], "accuracy": accuracy(D, start), "checks": checks(D, start, end), "trades": {}}
    for p in policies:
        r = out[p.key]
        st = stats(r["nav"], spy_nav, rf, r["gross"])
        res["policies"].append({"key": p.key, "book": p.book, "label": p.label, **st, "costs": r["costs"],
                                "turnover": r["turnover"], **r["counts"],
                                "nav": [round(float(x), 5) for x in r["nav"]],
                                "gross": [round(float(x), 3) for x in r["gross"]],
                                "cut_days": [str(D.dates[start + k]) for k, f in enumerate(r["flags"]) if f]})
        if r["trades"]:
            res["trades"][p.key] = r["trades"]
    res["dates"] = [str(x) for x in D.dates[start:end + 1]]
    return res


def rules(policies: tuple[Policy, ...] = POLICIES) -> dict:
    return {"top": TOP, "keep": KEEP, "fall": FALL, "fear": FEAR, "cut": CUT, "cool": COOL, "stop_slip": STOP_SLIP,
            "ref_days": REF_DAYS, "stale": STALE, "costs": [list(c) for c in COSTS], "spy_cost": SPY_COST,
            "one_band": ONE_BAND, "midday": f"{MIDDAY[0]}:{MIDDAY[1]:02d} New York",
            "policies": [p.__dict__ for p in policies]}


def one_policies() -> tuple[Policy, ...]:
    """One trade a day at most (M5, S3), beside the fixed books and the trader with no limit."""
    return (*[p for p in POLICIES if p.key in ONE_COMPARE], *ONE_POLICIES)


def save(res: dict) -> Path:
    tag = f"-{res['tag']}" if res.get("tag") else ""
    p = config.ROOT / "research" / "scans" / "replays" / f"trader{tag}-{res['first']}_{res['last']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(canon(res) + "\n")
    return p


def _pct(x: float, d: int = 1, signed: bool = False) -> str:
    return "n/a" if math.isnan(x) else (f"{x * 100:+.{d}f}%" if signed else f"{x * 100:.{d}f}%")


def show(res: dict) -> None:
    print(f"The scan as a trader, {res['first']} to {res['last']}: {res['sessions']} sessions, {res['stocks']} stocks. "
          "Read after every close, checked at the open and at midday, standing stops; paper only, after costs.")
    print(f"  {'':4} {'book and rules':48} {'$1 ->':>6} {'a year':>7} {'worst fall':>10} {'Sharpe':>6} {'beta':>5} "
          f"{'held':>5} {'costs':>6} {'w/o best 5 days':>15} {'orders':>7} {'most a day':>10}")
    for r in res["policies"]:
        print(f"  {r['key']:4} {BOOKS[r['book']][:22] + ': ' + r['label']:48.48} {r['end']:6.2f} {_pct(r['cagr'], 1, True):>7} "
              f"{_pct(r['worst_fall'], 0):>10} {r['sharpe']:6.2f} {r['beta']:5.2f} {_pct(r['exposure'], 0):>5} "
              f"{_pct(r['costs'], 1):>6} {r['end_without_best5']:15.2f} {r.get('orders', 0):7d} "
              f"{r.get('most_orders_a_day', 0):10d}")
    a = res["accuracy"]
    if "midweek" in a:
        x = a["midweek"]
        print(f"  Ranges read after every close vs the week's fixed reading, Monday to Thursday ({x['readings']} "
              f"readings): 2 in 3 held {_pct(x['fresh68'])} vs {_pct(x['fixed68'])}; 9 in 10 {_pct(x['fresh90'])} vs "
              f"{_pct(x['fixed90'])}.")
    if "after_big_day" in a:
        x = a["after_big_day"]
        print(f"  After a 2% market day or a 15% VIX jump since the fixed reading ({x['readings']} readings): 2 in 3 "
              f"held {_pct(x['fresh68'])} vs {_pct(x['fixed68'])}; 9 in 10 {_pct(x['fresh90'])} vs {_pct(x['fixed90'])}.")
    ch = res["checks"]
    print(f"  Market checks: {ch['open']['days']} sessions flagged at the open and {ch['midday']['days']} more at midday. "
          f"Stocks outside their 9-in-10 range after five sessions: {_pct(ch['open']['broke90'])} on open-flag days, "
          f"{_pct(ch['midday']['broke90'])} on midday-flag days, {_pct(ch['none']['broke90'])} on the others.")
    g = ch["gaps"]
    print(f"  A stock already outside its 2-in-3 range at the open ended the five sessions outside its 9-in-10 range "
          f"{_pct(g['outside68_at_open']['broke90'])} of the time ({g['outside68_at_open']['readings']} readings), "
          f"against {_pct(g['base']['broke90'])} for every stock.")
    print("A range is a size, not a direction. One path of one market, after the fact: descriptive, not a test. "
          "Survivorship: the lists were chosen in 2026.")


def cmd(a) -> int:
    one = getattr(a, "one_trade", False)
    res = run(Bank(), max(a.first, FIRST), a.last, policies=one_policies() if one else POLICIES, tag="one" if one else "")
    show(res)
    if a.save:
        print(f"saved {save(res)}")
    return 0
