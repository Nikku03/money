"""Live forecasts: F1 on the latest session, the printed card, the ``forecast`` fact and live scoring
(docs/FORECAST.md, section 10).

What ``brain forecast AAPL`` does:

1. The forecast date d is SPY's latest session in the bank. Everything is read as the bank stood at
   T(d) = d 23:59:59Z: prices from one book at T(d); filings, insider trades and earnings through
   ``bank.view(T(d))``.
2. The model (``model``) is F1 trained by F1a's rule: the grid of every h-th session counted back from d
   (``evaluate.grid``: the windows end at d and never overlap), every realised sample since 2011-01-01
   (stored exit on or before d), ``max_rows``, the same calibration and range fit, cutoff d. It is kept as
   plain JSON under ``<home>/models/forecast/<key>.json``, the key naming the code, the spec, the feature
   list, h and the universe, and it is retrained once the newest realised exit is more than 5 sessions past
   the cached one.
3. The stock's features at d are ranked across every universe stock listed at d (``cross_section``), and the
   forecast is section 6's dict: the 68% and 90% bands of its close at the exit, the chance it beats SPY,
   the drivers.
4. It is recorded as a ``forecast`` fact (tier derived, known at T(d)), citing the stock's, SPY's and ^VIX's
   price snapshots and the fundamental facts read. ``--llm`` asks the local model too and records its
   answer as a second fact (``context.forecaster`` "F1-llm"). A failed answer is recorded as F1's, flagged,
   as the declared test F1d counts it, so failing never flatters the model's record.

``brain forecast score`` scores the recorded forecasts once their exit has passed: band coverage, the Brier
score against climatology with its within-date part, placement and IC, per date. Only non-overlapping
windows count (forecasts on one date share a window), and a forecast recorded after its window opened is
late and not scored.

The development data rule (docs/FORECAST.md, section 1) is decided by the bank's own prices, not by where the
bank lives: when SPY's prices run past 2018-12-31, a live forecast would train on and forecast the held-out
years, so it is refused until the committed ledger (HEAD's ``research/ledger.jsonl``) holds a recorded result
of every declared F1 test, and ``--llm`` also until F1d has one or is withdrawn. A copy of the real bank under
another MONEY_HOME is refused the same. A synthetic test bank with recent dates opts out with
BRAIN_FORECAST_SYNTHETIC=1, which never covers the real home or a bank whose SPY prices came from the price
source (``held_out_guard``).

Paper only: nothing here trades, and none of it is financial advice.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from brain import config
from brain.bank import Bank, Fact, canon
from brain.timeutil import days_between, now_iso, today

HORIZONS = (5, 21)
RETRAIN_AFTER = 5          # sessions the newest realised exit may move before the model is refitted
PRICE_MAX_AGE_DAYS = 7     # no forecast when SPY's latest close is older (calendar days), as decide's data gate
CLOSE_UTC = "T20:00:00Z"   # a US session's close, taken early (EDT): a forecast recorded later on its entry day is late
ATTR = "forecast"
SOURCE = "brain.forecast"
BENCH = "SPY"
VIX = "^VIX"
UNIVERSE = "universe/us_broad.json"
MODEL_VERSION = 1
F1, F1_LLM = "F1", "F1-llm"
GUARD_TESTS = ("F1a-w5-recent", "F1b-w5-test", "F1c-m21-test")   # the declared tests F1 is judged by
REAL_HOME = config.ROOT / "var"
COMMITTED_LEDGER = "research/ledger.jsonl"   # read as committed at HEAD: uncommitted lines do not count
LAST_TRAIN = "2018-12-31"                    # the TRAIN years end here; later prices are the held-out years
SYNTHETIC_ENV = "BRAIN_FORECAST_SYNTHETIC"   # "1": a synthetic test bank with recent dates (never real prices)
RECENT_TEST = {5: "F1a-w5-recent"}                                  # descriptive retests, per horizon
GATED_TEST = {5: "F1b-w5-test", 21: "F1c-m21-test"}                 # gated tests, per horizon
LLM_TEST = "F1d-llm-w5-recent"
RAW_KEEP = 2000            # characters of the model's raw reply kept in the fact
PAPER = "forecast F1 · paper only · not financial advice"
_TICKER = re.compile(r"[A-Z0-9^][A-Z0-9.^=\-]{0,19}")
WORDS = ("score",)


class ForecastError(Exception):
    """Something the user can fix or must know: no prices, not listed, the development data rule.
    ``hint`` is the command that fixes it, when there is one."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message, self.hint = message, hint

    def __str__(self) -> str:
        return self.message


# ---------- the engine, imported on use (tests stand in for it) ----------
def _features():
    from brain.forecast import features
    return features


def _learn():
    from brain.forecast import learn
    return learn


def _evaluate():
    from brain.forecast import evaluate
    return evaluate


def _think():
    from brain.forecast import think
    return think


# ---------- small helpers ----------
def T(d: str) -> str:
    """The end of d's UTC day: when a forecast made after the close of d reads the bank."""
    return str(d)[:10] + "T23:59:59Z"


def run_time() -> str:
    """The end of the current UTC day, as the terminal reads the bank."""
    return today() + "T23:59:59Z"


def plain(x):
    """Plain JSON values: numpy scalars and arrays become Python ones, a non-finite float None, a tuple a list."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return str(x)


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float, np.number)) and math.isfinite(float(x)) else None


def _sha(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(hashlib.sha256(p).digest())
    return h.hexdigest()[:16]


def _same(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


def price_text(v) -> str:
    """A price as the card prints it: whole units from 100, cents below, more digits under 1."""
    v = _num(v)
    if v is None:
        return "–"
    a = abs(v)
    return f"{v:,.0f}" if a >= 100 else f"{v:,.2f}" if a >= 1 else f"{v:.4f}"


def pct_text(v, digits: int = 1) -> str:
    v = _num(v)
    return "–" if v is None else f"{v:+.{digits}f}%"


# ---------- the universe, the calendar and the development data rule ----------
def universe_tickers() -> list[str]:
    """``universe/us_broad.json`` minus SPY, upper case, in file order."""
    uni = json.loads((config.ROOT / UNIVERSE).read_text())
    return [t for t in dict.fromkeys(str(x).upper() for x in uni["tickers"]) if t not in (BENCH, VIX)]


def calendar(bank: Bank, t: str | None = None) -> np.ndarray:
    """SPY's sessions in its latest price snapshot known at ``t`` (default: the end of today)."""
    from brain.metrics import PriceBook
    book = PriceBook(bank, [BENCH], t or run_time())
    if not book.has(BENCH):
        raise ForecastError("no SPY prices in the bank: a forecast is measured against SPY", "INGEST SPY")
    return np.asarray(book.dates[BENCH]).astype(str)


def committed_ledger() -> list[dict]:
    """The research ledger's records as committed at HEAD (``git show HEAD:research/ledger.jsonl``): lines
    appended since the last commit do not count. [] when git cannot say."""
    try:
        r = subprocess.run(["git", "show", f"HEAD:{COMMITTED_LEDGER}"], cwd=config.ROOT, capture_output=True,
                           text=True, check=False)
    except OSError:
        return []
    out = []
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        try:
            rec = json.loads(line) if line.strip() else None
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _synthetic(spy: Fact) -> bool:
    """The opt-in for a synthetic test bank with recent dates: BRAIN_FORECAST_SYNTHETIC=1, never for the real
    home and never when SPY's snapshot came from the price source (a copy of the real bank)."""
    from brain.sources import prices
    home = Path(os.environ.get("MONEY_HOME") or REAL_HOME)
    return (os.environ.get(SYNTHETIC_ENV) == "1" and not _same(home, REAL_HOME)
            and (spy.source or {}).get("name") != prices.SOURCE_NAME)


def held_out_guard(bank: Bank | None = None, llm: bool = False) -> str | None:
    """Why a live forecast on ``bank`` (default: the home's) is refused, or None.

    Decided by the data: when SPY's prices in the bank run past 2018-12-31, a live model would train on and
    forecast the held-out years, and until the declared runs nothing may compute a forecast or a feature-based
    number for those dates (docs/FORECAST.md, section 1). So the committed ledger (``committed_ledger``) must
    first hold a recorded result of F1a, F1b and F1c (trial results do not count), and with ``llm`` also F1d's
    result or its withdrawal: the model's answers on these dates could otherwise shape its primer or prompt
    before F1d is pinned. A bank whose prices end in the TRAIN years is never refused; nor is a synthetic test
    bank that opts out (``_synthetic``)."""
    from brain.metrics import PriceBook
    bank = bank if bank is not None else Bank()
    book = PriceBook(bank, [BENCH], run_time())
    if not book.has(BENCH) or not len(book.dates[BENCH]):
        return None   # nothing can be forecast; the caller says why
    last = str(book.dates[BENCH][-1])
    if last <= LAST_TRAIN or _synthetic(book.snapshot[BENCH]):
        return None
    done, gone = set(), set()
    for r in committed_ledger():
        if r.get("op") == "result" and r.get("kind") == "forecast" and not r.get("trial"):
            done.add(r.get("id"))
        elif r.get("op") == "withdraw":
            gone.add(r.get("id"))
    missing = [i for i in GUARD_TESTS if i not in done]
    if missing:
        return (f"live forecasts wait for the declared tests: the bank's SPY prices run to {last}, past "
                f"{LAST_TRAIN}, and {', '.join(missing)} {'has' if len(missing) == 1 else 'have'} no recorded result "
                f"in the committed {COMMITTED_LEDGER} yet, so a live model would train on and forecast the held-out "
                "years first (docs/FORECAST.md, section 1: the development data rule). Run `brain forecast "
                "declare`, commit, then `brain forecast test` for each and commit the results.")
    if llm and LLM_TEST not in done and LLM_TEST not in gone:
        return (f"--llm waits for {LLM_TEST}: it has no recorded result in the committed {COMMITTED_LEDGER} and "
                f"is not withdrawn, and the model's answers on dates after {LAST_TRAIN} could shape its primer or "
                "prompt before F1d is pinned (docs/FORECAST.md, section 1). Declare and run F1d, or drop it "
                f"(`brain research withdraw {LLM_TEST} --reason ...`), or forecast without --llm.")
    return None


def check_ticker(t: str) -> str:
    """The ticker in upper case, or ForecastError: a forecast is about a stock against SPY, so neither SPY
    nor a market symbol (an index, currency, commodity or coin) can have one (``decide.scorable``)."""
    T_ = str(t or "").strip().upper()
    if not _TICKER.fullmatch(T_):
        raise ForecastError(f"not a ticker: {str(t)[:40]!r}")
    from brain import decide
    ok, why = decide.scorable(T_)
    if not ok:
        raise ForecastError(f"no forecast for {T_}: {why}", "FCST")
    return T_


# ---------- the model ----------
@dataclass
class Live:
    """F1 as the live forecast uses it: the model, its range quantiles and where they came from."""
    model: object               # learn.Model
    range: dict                 # learn.fit_range's result
    key: dict                   # what the cache file is named by
    meta: dict                  # trained_through, newest_exit, the grid, prices_asof, rows, samples, when, seconds
    path: Path
    retrained: bool
    fs: object = field(default=None, repr=False)   # the feature set trained on (this call only): d's rows pending
    px: object = field(default=None, repr=False)


def model_dir() -> Path:
    return config.home() / "models" / "forecast"


def code_sha() -> str:
    """sha of every Python file under brain/forecast/ (this one included): a change to any of them is a new
    model."""
    files = sorted(p for p in Path(__file__).resolve().parent.glob("*.py"))
    return _sha(*(p.name.encode() + b"\0" + p.read_bytes() for p in files))


def model_key(tickers, h: int) -> dict:
    """What a cached live model depends on: the code (brain/forecast/ and the modules features reads), the
    learner's and the range's spec, the feature list, h and the universe. Not the date: the cached model is
    reused until it is stale (``_fresh``)."""
    F, L = _features(), _learn()
    tk = [str(t).upper() for t in tickers]
    key = {"kind": "F1-live", "version": MODEL_VERSION, "h": int(h), "code_sha": code_sha(),
           "deps_sha": str(getattr(F, "deps_sha", lambda: "")()),
           "features_version": getattr(F, "FEATURES_VERSION", None),
           "names_sha": _sha(canon(list(getattr(F, "NAMES", []))).encode()),
           "LEARN": plain(L.LEARN), "RANGE": plain(L.RANGE),
           "universe_sha": _sha(",".join(tk).encode()), "n_tickers": len(tk)}
    key["sha"] = hashlib.sha256(canon(key).encode()).hexdigest()[:24]
    return key


def _load_model(path: Path, key: dict) -> Live | None:
    try:
        d = json.loads(path.read_text())
        if d.get("key") != key:
            return None
        m = _learn().from_json(d["model"])
        return Live(model=m, range=d["range"], key=key, meta=d["meta"], path=path, retrained=False)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_model(live: Live) -> None:
    live.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = live.path.with_name(f"{live.path.stem}.{os.getpid()}.tmp")
    body = {"key": live.key, "meta": live.meta, "model": _learn().to_json(live.model), "range": plain(live.range)}
    try:
        tmp.write_text(json.dumps(plain(body)))
        os.replace(tmp, live.path)
    except OSError:
        tmp.unlink(missing_ok=True)  # a cache: the model can always be trained again


def _fresh(live: Live, cal: np.ndarray, d: str, newest: str) -> bool:
    """A cached model serves d when it saw nothing after d and the newest realised exit has moved at most
    RETRAIN_AFTER sessions since it was trained."""
    m = live.meta
    if str(m.get("trained_through", "9999")) > d or str(m.get("newest_exit", "9999")) > newest:
        return False
    moved = int(np.searchsorted(cal, newest, side="right")) - int(np.searchsorted(cal, m["newest_exit"], side="right"))
    return moved <= RETRAIN_AFTER


def model(bank: Bank, h: int = 5, *, tickers=None, cal=None, d: str | None = None, refit: bool = False,
          log=None) -> Live:
    """F1 for live forecasts on d (default: SPY's latest session), trained by F1a's rule: every realised
    sample of the grid counted back from d since 2011-01-01, cutoff d, the same draw, calibration and range
    fit. Read from ``<home>/models/forecast/<key>.json`` while fresh; otherwise (or with ``refit``) trained
    and written there. Refused on a bank with prices past 2018 until the declared tests have run
    (``held_out_guard``)."""
    why = held_out_guard(bank)
    if why:
        raise ForecastError(why)
    F, L, E = _features(), _learn(), _evaluate()
    tk = list(tickers) if tickers is not None else universe_tickers()
    cal = calendar(bank) if cal is None else np.asarray(cal).astype(str)
    d = str(d or cal[-1])[:10]
    i = int(np.searchsorted(cal, d))
    if i >= len(cal) or cal[i] != d:
        raise ForecastError(f"{d} is not a SPY session in the bank")
    cal = cal[:i + 1]   # nothing after d
    key = model_key(tk, h)
    path = model_dir() / f"{key['sha']}.json"
    grid = list(E.grid(cal, d, h))
    if not grid:
        raise ForecastError(f"not enough SPY history before {d} for a training grid", "INGEST SPY")
    newest = E.exit_of(cal, grid[-1], h) or d
    if not refit:
        hit = _load_model(path, key)
        if hit is not None and _fresh(hit, cal, d, newest):
            return hit
    t0 = time.monotonic()
    asof = T(d)
    if log:
        log(f"training F1 (h={h}) on {len(tk)} stocks, {len(grid)} grid dates {grid[0]}..{grid[-1]}, cutoff {d}")
    px = F.prices(bank, tk, asof)
    t1 = time.monotonic()
    fs = F.feature_set(bank, tk, grid + [d], asof, h, cache=False, log=log, px=px)
    t2 = time.monotonic()
    try:
        m = L.fit(fs, d)
        rng = L.fit_range(fs, d, px)
    except ValueError as e:  # no realised samples, or all one outcome: the universe's prices are not in the bank
        raise ForecastError(f"F1 cannot be trained on the bank as it stands ({e}): it learns from the universe's "
                            f"price history ({UNIVERSE})", "INGEST UNIVERSE") from None
    t3 = time.monotonic()
    samples = int(np.sum(np.asarray(fs.exit).astype(str) <= d)) if len(fs.exit) else 0
    meta = {"trained_through": d, "newest_exit": str(newest), "grid_first": grid[0], "grid_last": grid[-1],
            "n_grid": len(grid), "prices_asof": asof, "samples": samples, "rows": int(m.rows),
            "stocks": int(np.isfinite(px.adj).any(axis=0).sum()),   # universe stocks with prices in the bank
            "dropped": plain(dict(getattr(fs, "dropped", {}) or {})), "trained_at": now_iso(),
            "seconds": {"prices": round(t1 - t0, 1), "features": round(t2 - t1, 1), "fit": round(t3 - t2, 1)}}
    live = Live(model=m, range=plain(rng), key=key, meta=meta, path=path, retrained=True, fs=fs, px=px)
    _save_model(live)
    if log:
        log(f"  trained in {t3 - t0:.0f}s (prices {t1 - t0:.0f}s, features {t2 - t1:.0f}s, fit {t3 - t2:.0f}s): "
            f"{m.rows} rows of {samples} realised samples from {meta['stocks']} of {len(tk)} stocks -> {path}")
    return live


# ---------- the cross-section at d ----------
def cross_section(bank: Bank, live: Live, tickers, d: str, h: int, cal) -> dict:
    """Every stock listed at d with its model inputs, ranked across them: {"X", "ticker", "close", "px"}.

    When the model was trained in this call its feature set already holds d's rows (pending: their window
    is open). Otherwise one date is computed: P, B and F features ranked across the stocks listed at d, and R
    z-scored by their values on the same grid's earlier dates, as training saw them."""
    F, E = _features(), _evaluate()
    tk = list(dict.fromkeys(str(t).upper() for t in tickers))
    if live.fs is not None and live.px is not None and list(live.px.tickers) == tk:
        px, P = live.px, live.fs.pending
        X = np.asarray(P["X"], dtype=float)
    else:
        px = F.prices(bank, tk, T(d))
        one = F.feature_set(bank, tk, [d], T(d), h, cache=False, px=px)
        P = one.pending
        X = np.array(P["X"], dtype=float)
        if len(X):
            grid = list(E.grid(np.asarray(cal).astype(str)[:int(np.searchsorted(cal, d, side="right"))], d, h))
            reg = [F.regime_features(px, px.index(g)) for g in grid + [d]]
            for name in F.REGIME_NAMES:
                z = F.zscore_earlier(np.array([r[name] for r in reg], dtype=float))[-1]
                X[:, F.NAMES.index(name)] = z
    dates = np.asarray(P["date"]).astype(str)
    keep = dates == d
    return {"X": X[keep], "ticker": [str(t).upper() for t in np.asarray(P["ticker"]).astype(str)[keep]],
            "close": np.asarray(P["close"], dtype=float)[keep], "px": px}


def _not_listed(px, t: str, d: str) -> str:
    """Why t has no row at d: the listed rule (a close within 4 days of d and 252 bars up to it)."""
    book = px.book
    if not book.has(t):
        return f"no prices for {t} in the bank"
    own = np.asarray(book.dates[t]).astype(str)
    k = int(np.searchsorted(own, d, side="right"))
    if k == 0:
        return f"{t} has no close on or before {d}"
    if days_between(own[k - 1], d) > 4:
        return f"{t}'s latest close on or before {d} is {own[k - 1]}, more than 4 days old"
    if k < 252:
        return f"{t} has {k} sessions of prices up to {d}; a forecast needs 252"
    return f"{t} is not listed at {d}"


def _naive_sd(px, j: int, i: int, n: int = 252) -> float | None:
    """The standard deviation of the stock's daily log returns over the 252 sessions to d: the naive normal
    band's volatility, which live scoring compares the range with."""
    a = np.asarray(px.adj, dtype=float)[max(0, i - n):i + 1, j]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.diff(np.log(np.where(a > 0, a, np.nan)))
    r = r[np.isfinite(r)]
    return float(np.std(r, ddof=1)) if len(r) >= 21 else None


def expected_sessions(d: str, n: int) -> list[str]:
    """The n sessions after d as weekdays: the bank holds no sessions after d, and holidays are not known
    ahead. Scoring reads the real calendar."""
    base = np.datetime64(str(d)[:10], "D")
    return [str(np.busday_offset(base, k, roll="forward")) for k in range(1, int(n) + 2)]


# ---------- one forecast ----------
def _fundamental_inputs(bank: Bank, px, t: str, d: str) -> list[Fact]:
    """The fundamental facts behind t's F features at d: what ``features.fundamentals_with_inputs`` read at
    the month-end view the features use (price snapshots aside; they are cited on their own)."""
    from brain.forecast import features as reader  # the real reader, even when a test stands in for the engine
    try:
        i = px.index(d)
        m = _features().month_end_index(np.asarray(px.dates), i)
        if m < 0:
            return []
        _, facts = reader.fundamentals_with_inputs(bank.view(T(str(px.dates[m]))), t, px.book)
    except Exception:  # noqa: BLE001 - the forecast stands without them; its price inputs are always cited
        return []
    return [f for f in facts if f.attr != "series.price_daily"]


def forecast_one(bank: Bank, live: Live, cross: dict, t: str, d: str, h: int) -> tuple[dict, dict]:
    """(section 6's forecast dict, what else the fact records) for t on d."""
    L = _learn()
    px = cross["px"]
    if t not in cross["ticker"]:
        raise ForecastError(f"no forecast for {t} on {d}: {_not_listed(px, t, d)}", f"INGEST {t}")
    r = cross["ticker"].index(t)
    j = list(px.tickers).index(t)
    i = px.index(d)
    sig = _num(L.sigma(px, i)[j])
    if sig is None or sig <= 0:
        raise ForecastError(f"no forecast for {t} on {d}: too few recent daily returns for its volatility", f"INGEST {t}")
    close = float(cross["close"][r])
    own = np.asarray(px.book.dates[t]).astype(str)
    bar = str(own[int(np.searchsorted(own, d, side="right")) - 1])
    sess = expected_sessions(d, h)
    fc = L.forecast(live.model, live.range, ticker=t, d=d, h=h, close=close, sigma_d=sig, x_row=cross["X"][r],
                    entry=sess[0], exit=sess[h])
    fc = plain(fc)
    extra = {"close_date": bar, "sigma": sig, "naive_sd": _naive_sd(px, j, i), "base_rate": float(live.model.base_rate),
             "range_q": plain((live.range or {}).get("q") or {}), "listed": len(cross["ticker"]),
             "in_universe": t in set(live.px.tickers if live.px is not None else cross["ticker"]),
             "trained": {k: live.meta.get(k) for k in ("trained_through", "newest_exit", "grid_first", "grid_last",
                                                         "n_grid", "samples", "rows")},
             "model_key": live.key.get("sha"), "prices_asof": T(d),
             "dates": "entry and exit are the expected sessions (weekdays); scoring reads the real calendar"}
    return fc, extra


def _snapshot_refs(px, t: str) -> list[Fact]:
    return [px.book.snapshot[s] for s in (t, BENCH, VIX) if s in px.book.snapshot]


def band_text(fc: dict) -> str:
    b = fc.get("band") or {}
    return f"{price_text(b.get('q16'))}-{price_text(b.get('q84'))}"


def fact_for(fc: dict, extra: dict, inputs: list[Fact], forecaster: str = F1, llm: dict | None = None,
             cites: list[str] | None = None) -> Fact:
    """The forecast as a derived fact: entity the ticker, value ``p_beat 0.51; 68% 331-353``, known at T(d) (or
    when its latest input became known, if later), the forecast dict as context, citing its inputs."""
    p = float(fc["p_beat"])
    made = now_iso()  # each run is its own fact, so a job's forecast is never mistaken for an earlier identical one
    ctx = plain({**fc, **extra, "forecaster": forecaster, "made_at": made, **({"llm": llm} if llm is not None else {})})
    refs = [f.ref for f in inputs] + list(cites or [])
    known = max([T(fc["date"])] + [f.known_at for f in inputs])
    return Fact(entity=fc["ticker"], attr=ATTR, value=f"p_beat {p:.2f}; 68% {band_text(fc)}", as_of=fc["date"],
                known_at=known, tier="derived", source={"name": SOURCE, "forecaster": forecaster}, context=ctx,
                inputs=tuple(dict.fromkeys(refs)), retrieved_at=made)


def llm_answer(bank: Bank, px, t: str, d: str, fc: dict, client=None) -> dict:
    """The local model's probability for this forecast (``think.llm_forecast``): {"ok", "p_beat", "reason",
    "model", "error", "raw"}. A failed answer takes F1's p_beat and says why; this never raises."""
    t0 = time.monotonic()
    try:
        res = _think().llm_forecast(bank, px.book, t, d, fc, client=client)
    except Exception as e:  # noqa: BLE001 - a missing or broken thinking step is a failed answer
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    res = res if isinstance(res, dict) else {"ok": False, "error": "the model step returned nothing usable"}
    p = _num(res.get("p_beat"))
    ok = bool(res.get("ok")) and p is not None and 0.0 <= p <= 1.0
    return {"ok": ok, "p_beat": p if ok else float(fc["p_beat"]), "f1_p_beat": float(fc["p_beat"]),
            "reason": str(res.get("reason") or "")[:1000] if ok else "",
            "model": res.get("model"), "error": None if ok else str(res.get("error") or "no usable answer")[:500],
            "raw": str(res.get("raw") or "")[:RAW_KEEP], "seconds": round(_num(res.get("seconds")) or
                                                                         (time.monotonic() - t0), 1)}


def run(bank: Bank, tickers, *, h: int = 5, llm: bool = False, client=None, universe=None, refit: bool = False,
        log=None, record: bool = True) -> list[dict]:
    """Forecast each ticker on SPY's latest session and record each forecast as a fact (both facts with
    ``llm``). One result per ticker: {"ok", "ticker", "forecast", "ref", "llm", "record", "live", "live_llm",
    "notes"} or {"ok": False, "ticker", "error", "hint"}. With ``llm`` the record also quotes F1d and
    ``live_llm`` is the model's own live score, so the model's chance never stands without its record. Raises
    ForecastError when nothing can be forecast at all (the development data rule, no SPY prices, stale
    prices)."""
    why = held_out_guard(bank, llm=llm)
    if why:
        raise ForecastError(why)
    if int(h) not in HORIZONS:
        raise ForecastError(f"the horizon is 5 or 21 sessions, not {h}")
    h = int(h)
    want, out = [], {}
    for t in tickers:
        try:
            want.append(check_ticker(t))
        except ForecastError as e:
            k = str(t).strip().upper()
            out[k] = {"ok": False, "ticker": k, "error": e.message, "hint": e.hint}
    want = list(dict.fromkeys(want))
    if want:
        cal = calendar(bank)
        d = str(cal[-1])
        age = days_between(d, today())
        if age > PRICE_MAX_AGE_DAYS:
            raise ForecastError(f"SPY's latest close is {d}, {age} days ago: a forecast is made after the latest "
                                "close, so collect it first", "INGEST SPY")
        uni = list(universe) if universe is not None else universe_tickers()
        live = model(bank, h, tickers=uni, cal=cal, refit=refit, log=log)
        extra_t = [t for t in want if t not in set(uni)]
        cross = cross_section(bank, live, uni + extra_t, d, h, cal)
        rec, lv = record_lines(h, llm=llm), live_summary(bank, h)
        lv_llm = live_summary(bank, h, F1_LLM) if llm else None   # the model's own record, beside its chance
        for t in want:
            try:
                fc, extra = forecast_one(bank, live, cross, t, d, h)
            except ForecastError as e:
                out[t] = {"ok": False, "ticker": t, "error": e.message, "hint": e.hint}
                continue
            inputs = _snapshot_refs(cross["px"], t) + _fundamental_inputs(bank, cross["px"], t, d)
            fact = fact_for(fc, extra, inputs)
            notes = [] if extra["in_universe"] else [f"{t} is not in {UNIVERSE}: the model never trained on it, "
                                                     "and its features are ranked against the universe's"]
            stocks, n = live.meta.get("stocks"), live.key.get("n_tickers")
            if stocks is not None and n and stocks < n / 2:
                notes.append(f"the model learned from only {stocks} of the universe's {n} stocks (a small bank): it is "
                             "not the tested F1, so its numbers only show the pipeline working")
            res = {"ok": True, "ticker": t, "forecast": fc, "extra": extra, "ref": fact.ref, "known_at": fact.known_at,
                   "llm": None, "llm_ref": None, "record": rec, "live": lv, "live_llm": lv_llm, "notes": notes,
                   "model": {**live.model.card(), "retrained": live.retrained, "path": str(live.path)}}
            if record:
                bank.add(fact)
            if llm:
                ans = llm_answer(bank, cross["px"], t, d, fc, client=client)
                fc2 = {**fc, "p_beat": ans["p_beat"]}
                f2 = fact_for(fc2, {**extra, "llm_failed": not ans["ok"]}, inputs, F1_LLM,
                              llm={k: ans[k] for k in ("ok", "f1_p_beat", "reason", "model", "error", "raw")},
                              cites=[fact.ref])
                if record:
                    bank.add(f2)
                res.update(llm=ans, llm_ref=f2.ref)
            out[t] = res
    return [out[t] for t in dict.fromkeys(str(x).strip().upper() for x in tickers) if t in out]


# ---------- the record: the declared tests and live scoring ----------
def _verdict_rows() -> dict[str, dict]:
    from brain.research import run as research_run
    rows = research_run.forecast_verdicts(config.LEDGER) if hasattr(research_run, "forecast_verdicts") else [
        r for r in research_run.verdicts() if r.get("kind") == "forecast"]
    return {r["id"]: r for r in rows}


def _word(label) -> str:
    m = re.search(r"descriptive\)?:\s*(?:LLM vs F1\s*)?(\w+)", str(label or ""))
    return m.group(1) if m else (str(label) if label else "no reading")


def _years(r: dict) -> str:
    pl = r.get("plan") or {}
    a, b = str(pl.get("first") or "?")[:4], str(pl.get("last_exit") or pl.get("last") or "?")[:4]
    return a if a == b else f"{a}-{b}"


def _test_line(r: dict | None, test_id: str, what: str) -> str:
    """One declared test as the card quotes it, from the ledger as it stands."""
    short = test_id.split("-")[0]
    if r is None:
        return f"{what} ({short}): not run yet"
    trial = " (trial, TRAIN years)" if r.get("trial") else ""
    v = str(r.get("verdict") or "")
    if v.startswith("WITHDRAWN"):
        return f"{what} ({short}){trial}: withdrawn ({v.split(':', 1)[-1].strip()})"
    m = r.get("metrics") or {}
    if short == "F1d":
        c = m.get("comparison") or {}
        return (f"{what} ({short}){trial}: Brier F1 minus model {_fmt(c.get('mean'), '+.4f')} per date (90% interval "
                f"{_fmt(c.get('lo90'), '+.4f')}..{_fmt(c.get('hi90'), '+.4f')}), {_word(c.get('label') or m.get('label'))}; "
                f"{m.get('llm_failed', 0)} of {m.get('llm_asked', 0)} answers failed")
    sn = m.get("ic_size_neutral") or {}
    if v == "REPORT":
        return (f"{what} ({short}){trial}: size-neutral IC {_fmt(sn.get('mean'), '+.2f')} (90% interval "
                f"{_fmt(sn.get('lo90'), '+.2f')}..{_fmt(sn.get('hi90'), '+.2f')}), {_word(m.get('label'))}")
    return (f"{_years(r)} ({short}){trial}: size-neutral IC {_fmt(sn.get('mean'), '+.3f')}, p "
            f"{_fmt(r.get('p'), '.3f')}, {v or 'no verdict'}")


def _fmt(x, f: str) -> str:
    x = _num(x)
    return "–" if x is None else format(x, f)


def _range_line(r: dict | None) -> str | None:
    if r is None or str(r.get("verdict") or "").startswith("WITHDRAWN"):
        return None
    m = r.get("metrics") or {}
    m = m.get("llm", m) if "comparison" in m else m
    c = (m.get("range") or {}).get("cover68") or {}
    if _num(c.get("mean")) is None:
        return None
    iv = f" (interval {_fmt(c.get('lo90'), '.0%')}-{_fmt(c.get('hi90'), '.0%')})" if _num(c.get("lo90")) is not None else ""
    return f"Range ({r['id'].split('-')[0]}): 68% band held {_fmt(c.get('mean'), '.0%')} of the time{iv}"


def record_lines(h: int, llm: bool = False) -> dict:
    """What the declared tests say about forecasts at horizon h, quoted from the ledger as it stands:
    {"lines": [str], "tests": {id: verdict or None}, "error"}. Until a test has a result its line says so.
    ``llm`` adds the model's own test (F1d, 5 sessions only), for a card that shows the model's chance."""
    try:
        rows = _verdict_rows()
    except Exception as e:  # noqa: BLE001 - the card still prints; the record says it could not be read
        return {"lines": [f"the research ledger could not be read ({type(e).__name__}: {e})"], "tests": {}, "error": str(e)}
    lines, tests = [], {}
    recent, gated = RECENT_TEST.get(h), GATED_TEST.get(h)
    if recent:
        lines.append(_test_line(rows.get(recent), recent, "last 2 months"))
        tests[recent] = (rows.get(recent) or {}).get("verdict")
    if gated:
        lines.append(_test_line(rows.get(gated), gated, "2019-2026"))
        tests[gated] = (rows.get(gated) or {}).get("verdict")
    f1_tested = any(tests.values())
    if llm and h == 5:
        lines.append(_test_line(rows.get(LLM_TEST), LLM_TEST, "the model on F1a's weeks"))
        tests[LLM_TEST] = (rows.get(LLM_TEST) or {}).get("verdict")
    elif llm:
        lines.append(f"the model's chance at {h} sessions has no declared test")
    rl = _range_line(rows.get(recent)) if recent else None
    rl = rl or (_range_line(rows.get(gated)) if gated else None)
    if rl:
        lines.append(rl)
    if not f1_tested:
        lines.append("no declared test of F1 at this horizon has a result yet: this forecast is untested")
    return {"lines": lines, "tests": tests, "error": None}


def forecast_facts(bank: Bank, t: str | None = None) -> list[Fact]:
    """Every recorded forecast fact known by ``t`` (default: the end of today), across the bank."""
    t = t or run_time()
    out = []
    for e in bank.entities():
        b = Bank(bank.root, use_index=bank.use_index)  # one entity's facts at a time
        if ATTR not in b.attrs(e):
            continue
        out += [f for f in b.view(t).facts(e, ATTR) if (f.retrieved_at or f.known_at) <= t]
    return out


def history(bank: Bank, ticker: str) -> list[Fact]:
    """Every recorded forecast on the stock, newest first (as the journal took them in)."""
    return list(reversed(bank.facts(ticker.upper(), ATTR)))


def latest(bank: Bank, ticker: str, t: str | None = None) -> dict[tuple[int, str], Fact]:
    """The newest forecast per (horizon, forecaster) on the stock known by ``t``."""
    t = t or run_time()
    out: dict[tuple[int, str], Fact] = {}
    for f in bank.view(t).facts(ticker.upper(), ATTR):
        if (f.retrieved_at or f.known_at) > t:
            continue
        k = (int(f.context.get("horizon") or 0), str(f.context.get("forecaster") or F1))
        g = out.get(k)
        if g is None or (f.as_of, f.retrieved_at or "", f.id) >= (g.as_of, g.retrieved_at or "", g.id):
            out[k] = f
    return out


def _realised(book, t: str, d: str, exit_: str) -> tuple[float | None, float | None]:
    """(the stock's own total return from the close of d to the close at the exit, that close)."""
    sd, sa, sc = (np.asarray(book.dates[t]).astype(str), np.asarray(book.adj[t], dtype=float),
                  np.asarray(book.close[t], dtype=float))
    i0 = int(np.searchsorted(sd, d, side="right")) - 1
    i1 = int(np.searchsorted(sd, exit_, side="right")) - 1
    if i0 < 0 or i1 < 0 or not (sa[i0] > 0 and np.isfinite(sa[i1])):
        return None, None
    return float(sa[i1] / sa[i0] - 1), float(sc[i1])


def score(bank: Bank, t: str | None = None) -> dict:
    """Live forecasts scored once their exit has passed, per (forecaster, horizon):

    {"groups": {"F1 h=5": {"forecaster", "horizon", "forecasts", "scored", "dates", "overlapping", "late",
                           "pending", "unpriced", "metrics": evaluate.score(...) or None}},
     "rows": [{"ref", "ticker", "forecaster", "horizon", "date", "p_beat", "q16", "q84", "q05", "q95", "status",
               "entry", "exit", "excess", "close_exit", "in68", "in90", "beat"}]  (newest first)}
               (a date's rows by stock, horizon, F1 before F1-llm)

    One forecast counts per (stock, forecaster, horizon, date): the latest recorded. Forecasts on one date
    share a window; a later date whose window opens inside a counted one is ``overlapping`` and not scored
    (``claims.bets``' rule), and so is a forecast recorded after its window opened (``late``). The metrics are
    the declared tests' (``evaluate.score``) without size: band coverage, Brier skill and its within-date part,
    placement, and IC on dates with 3 or more stocks, each per date and then over dates."""
    from brain import claims
    from brain.metrics import PriceBook
    t = t or run_time()
    facts: dict[tuple, Fact] = {}
    for f in forecast_facts(bank, t):
        c = f.context
        k = (f.entity, str(c.get("forecaster") or F1), int(c.get("horizon") or 0), f.as_of)
        g = facts.get(k)
        if g is None or (f.retrieved_at or "", f.id) > (g.retrieved_at or "", g.id):
            facts[k] = f
    book = PriceBook(bank, sorted({k[0] for k in facts} | {BENCH}), asof=t)
    groups: dict[str, dict] = {}
    rows, calls = [], {}
    for (ent, who, h, d), f in sorted(facts.items(), key=lambda kv: (kv[0][3], kv[0][0], kv[0][1], kv[0][2])):
        c = f.context
        b = c.get("band") or {}
        name = f"{who} h={h}"
        g = groups.setdefault(name, {"forecaster": who, "horizon": h, "forecasts": 0, "scored": 0, "dates": 0,
                                     "overlapping": 0, "late": 0, "pending": 0, "unpriced": 0, "llm_failed": 0,
                                     "metrics": None})
        g["forecasts"] += 1
        g["llm_failed"] += bool(c.get("llm_failed"))
        row = {"ref": f.ref, "ticker": ent, "forecaster": who, "horizon": h, "date": d, "p_beat": _num(c.get("p_beat")),
               **{k: _num(b.get(k)) for k in ("q05", "q16", "q50", "q84", "q95")}, "status": "pending", "entry": None,
               "exit": None, "excess": None, "close_exit": None, "in68": None, "in90": None, "beat": None,
               "llm_failed": bool(c.get("llm_failed"))}
        status, o = claims.scored_at(book, ent, T(d), h, t, BENCH)
        row["status"] = status
        if o is not None:
            r_tot, cx = _realised(book, ent, d, o["exit"])
            row.update(entry=o["entry"], exit=o["exit"], excess=o["excess"], close_exit=cx,
                       beat=None if o["excess"] == 0 else bool(o["excess"] > 0))
            if cx is not None and row["q16"] is not None:
                row["in68"] = bool(row["q16"] <= cx <= row["q84"])
                row["in90"] = bool(row["q05"] <= cx <= row["q95"])
            if (f.retrieved_at or f.known_at) > o["entry"] + CLOSE_UTC:
                row["status"] = "late"
            else:
                calls.setdefault(name, []).append({"ref": f.ref, "ticker": ent, "direction": 1, "entry": o["entry"],
                                                   "exit": o["exit"], "excess": o["excess"], "_f": f, "_r": r_tot,
                                                   "_row": row})
        rows.append(row)
    for name, cs in calls.items():
        kept_bets, _ = claims.bets([{k: v for k, v in c.items() if not k.startswith("_")} for c in cs])
        kept = {r for b in kept_bets for r in b["refs"]}
        for c in cs:
            c["_row"]["status"] = "scored" if c["ref"] in kept else "overlapping"
        groups[name]["metrics"] = _metrics([c for c in cs if c["ref"] in kept], groups[name]["horizon"])
    for r in rows:
        g = groups[f"{r['forecaster']} h={r['horizon']}"]
        g[r["status"]] += 1
    for name, g in groups.items():
        g["dates"] = len({r["date"] for r in rows if r["status"] == "scored" and f"{r['forecaster']} h={r['horizon']}" == name})
    rows.sort(key=lambda r: (r["ticker"], r["horizon"], r["forecaster"]))
    rows.sort(key=lambda r: r["date"], reverse=True)  # newest first; within a date by stock, horizon, F1 first
    return plain({"groups": dict(sorted(groups.items())), "rows": rows, "at": t})


def _metrics(cs: list[dict], h: int) -> dict | None:
    """``evaluate.score`` on the scored live forecasts (size and momentum unknown: their baselines are left out)."""
    cs = [c for c in cs if c["excess"] != 0]
    if not cs:
        return None
    E = _evaluate()
    n = len(cs)
    ctx = [c["_f"].context for c in cs]
    qk = list(getattr(E, "QKEYS", ["q05", "q16", "q50", "q84", "q95"]))

    def num(x, missing=np.nan):  # 0.0 is a value (a quantile of 0, a base rate of 0), not a gap
        v = _num(x)
        return missing if v is None else v
    zq = np.array([[num((x.get("range_q") or {}).get(k)) for k in qk] for x in ctx], dtype=float)
    r = np.array([np.nan if c["_r"] is None else c["_r"] for c in cs], dtype=float)
    with np.errstate(invalid="ignore"):
        u = np.log1p(r)
    S = {"date": np.array([c["_f"].as_of for c in cs], dtype="U10"),
         "p": np.array([float(x["p_beat"]) for x in ctx]),
         "y": np.array([1.0 if c["excess"] > 0 else 0.0 for c in cs]),
         "excess": np.array([float(c["excess"]) for c in cs]),
         "size": np.full(n, np.nan), "mom": np.full(n, np.nan),
         "clim": np.array([num(x.get("base_rate"), 0.5) for x in ctx]),
         "u": u, "sig": np.array([num(x.get("sigma")) for x in ctx]),
         "naive": np.array([num(x.get("naive_sd")) for x in ctx]), "zq": zq}
    m = E.score(S, h, gated=False, cal_groups=5)
    return plain({k: m.get(k) for k in ("n_dates", "n_samples", "ic", "brier", "brier_gain", "within_gain",
                                        "placement", "range", "calibration", "spread")})


def live_summary(bank: Bank, h: int, forecaster: str = F1) -> dict:
    """The live record of one forecaster at horizon h: {"scored", "pending", ..., "line"}. The model's line
    (F1-llm) says whose record it is: "The model's live forecasts (F1-llm) scored: ..."."""
    who = "Live forecasts" if forecaster == F1 else f"The model's live forecasts ({forecaster})"
    try:
        g = score(bank)["groups"].get(f"{forecaster} h={h}")
    except Exception as e:  # noqa: BLE001 - the card prints without it and says so
        return {"scored": None, "line": f"{who}: live scoring unavailable ({type(e).__name__}: {e})"}
    if not g:
        return {"scored": 0, "line": f"{who} scored: 0."}
    return {**{k: g[k] for k in ("forecasts", "scored", "dates", "pending", "overlapping", "late", "unpriced")},
            "line": _live_line(g, who)}


def _live_line(g: dict, who: str = "Live forecasts") -> str:
    m = g.get("metrics") or {}
    head = f"{who} scored: {g['scored']}" + (f" on {g['dates']} date{'s' if g['dates'] != 1 else ''}"
                                              if g["scored"] else "")
    if not m:
        return head + f" ({g['pending']} pending)." if g.get("pending") else head + "."
    c68 = ((m.get("range") or {}).get("cover68") or {})
    bits = [f"68% band held {_fmt(c68.get('mean'), '.0%')}"]
    sk = (m.get("brier") or {}).get("skill")
    if _num(sk) is not None:
        bits.append(f"Brier skill {_fmt(sk, '+.1%')} against the base rate")
    return head + ": " + ", ".join(bits) + "."


# ---------- the printed card ----------
def card(res: dict) -> str:
    """One forecast result (``run``'s) as the CLI prints it."""
    if not res.get("ok"):
        return f"{res.get('ticker')}  no forecast: {res.get('error')}" + (f" ({res['hint']})" if res.get("hint") else "")
    fc = res["forecast"]
    ex = res.get("extra") or {}
    b = fc.get("band") or {}
    m68, m90 = fc.get("move68_pct") or [None, None], fc.get("move90_pct") or [None, None]
    head = f"{fc['ticker']}  after {fc['date']}, next {fc['horizon']} sessions"
    lines = [f"{head:<55}{PAPER}",
             f"Close {price_text(fc['close'])} on {ex.get('close_date') or fc['date']}",
             f"Likely range   68%: {price_text(b.get('q16'))} - {price_text(b.get('q84'))}  "
             f"({pct_text(m68[0])} .. {pct_text(m68[1])})     90%: {price_text(b.get('q05'))} - "
             f"{price_text(b.get('q95'))}  ({pct_text(m90[0])} .. {pct_text(m90[1])})",
             f"Chance it beats SPY: {float(fc['p_beat']) * 100:.0f}%   (a coin is 50%)"]
    ans = res.get("llm")
    if ans:
        if ans.get("ok"):
            lines.append(f"The model's chance (F1-llm{', ' + str(ans['model']) if ans.get('model') else ''}): "
                         f"{ans['p_beat'] * 100:.0f}%. {ans.get('reason') or ''}".rstrip())
        else:
            lines.append(f"The model (F1-llm) gave no usable answer ({ans.get('error')}); recorded as F1's "
                         f"{ans['p_beat'] * 100:.0f}%, flagged, as the declared test F1d counts it.")
    dr = fc.get("drivers") or []
    lines.append("Drivers: " + (", ".join(f"{x['name']} {x['push']:+.2f}" for x in dr) + (", ..." if len(dr) >= 5 else "")
                                if dr else "none (every input sits at the middle of its rank)"))
    rec = (res.get("record") or {}).get("lines") or []
    live = (res.get("live") or {}).get("line") or "Live forecasts scored: 0."
    body = [ln + "." if not ln.endswith(".") else ln for ln in rec] + [live]
    if ans:   # the model's chance is shown: so is its own live record
        body.append((res.get("live_llm") or {}).get("line") or f"The model's live forecasts ({F1_LLM}) scored: 0.")
    lines.append("Record: " + body[0])
    lines += ["        " + ln for ln in body[1:]]
    mod = res.get("model") or {}
    lines.append(f"Model {mod.get('id', 'F1')} {mod.get('sha', '?')}, trained through {mod.get('trained_through', '?')} "
                 f"on {mod.get('rows', '?')} rows{' (trained now)' if mod.get('retrained') else ''}. "
                 f"Recorded [{str(res.get('ref', '/')).partition('/')[2][:8]}]"
                 + (f" and [{str(res.get('llm_ref')).partition('/')[2][:8]}]" if res.get("llm_ref") else "") + ".")
    lines += [f"Note: {n}" for n in res.get("notes") or []]
    lines.append("The range is the model's; the chance has no tested edge unless the record above says PASS.")
    return "\n".join(lines)


def scorecard(sc: dict) -> str:
    """``score``'s result as ``brain forecast score`` prints it."""
    groups = sc.get("groups") or {}
    if not groups:
        return "No forecasts recorded yet: brain forecast TICKER records one (paper only)."
    out = ["Live forecasts, scored against SPY once their exit has passed (paper only, not financial advice)."]
    for name, g in groups.items():
        out.append(f"\n{name}: {g['forecasts']} forecast{'s' if g['forecasts'] != 1 else ''}; {g['scored']} scored on "
                   f"{g['dates']} date{'s' if g['dates'] != 1 else ''}, {g['overlapping']} overlapping, {g['late']} late, "
                   f"{g['pending']} pending, {g['unpriced']} unpriced"
                   + (f"; {g['llm_failed']} model answers failed (counted as F1's)" if g.get("llm_failed") else ""))
        m = g.get("metrics")
        if not m:
            out.append("  nothing scored yet")
            continue
        rg = m.get("range") or {}
        c68, c90 = rg.get("cover68") or {}, rg.get("cover90") or {}
        out.append(f"  range: 68% band held {_fmt(c68.get('mean'), '.0%')} per date (90% interval "
                   f"{_fmt(c68.get('lo90'), '.0%')} to {_fmt(c68.get('hi90'), '.0%')}), 90% band "
                   f"{_fmt(c90.get('mean'), '.0%')} ({_fmt(c90.get('lo90'), '.0%')} to {_fmt(c90.get('hi90'), '.0%')})"
                   + (f"; pinball loss {_fmt(rg.get('pinball_skill'), '+.0%')} against a naive normal band"
                      if _num(rg.get("pinball_skill")) is not None else ""))
        br = m.get("brier") or {}
        out.append(f"  chance: Brier skill {_fmt(br.get('skill'), '+.1%')} against the training base rate "
                   f"(within-date {_fmt(br.get('within_skill'), '+.1%')}); placement "
                   f"{_fmt((m.get('placement') or {}).get('mean'), '.0%')}; IC {_fmt((m.get('ic') or {}).get('mean'), '+.2f')}"
                   f" over {(m.get('ic') or {}).get('k', 0)} date(s) with 3 or more stocks")
    out.append("\nOne forecast per stock, forecaster, horizon and date (the latest recorded). Overlapping: its window "
               "opened inside an earlier counted one. Late: recorded after its window opened. Every interval is "
               "date-level, since stocks move together.")
    return "\n".join(out)


# ---------- the command line ----------
def _usage(msg: str) -> int:
    print(f"brain forecast: {msg}", file=sys.stderr)
    return 1


def _cmd(a, tests=None, tests_error: str | None = None) -> int:
    words = list(getattr(a, "words", None) or [])
    if not words:
        return _usage("name a ticker (brain forecast AAPL), or score, declare, test ID...")
    head = words[0]
    if head in ("declare", "test"):
        if a.llm or a.json or a.horizon is not None or a.refit:
            return _usage(f"`{head}` takes no --llm, --json, --horizon or --refit (the model's cutoff is --llm-cutoff)")
        if tests is None:
            return _usage(f"the declared forecast tests are not available ({tests_error or 'not loaded'})")
        return tests(a)
    if getattr(a, "trial", False) or getattr(a, "llm_cutoff", None):
        return _usage("--trial and --llm-cutoff go with declare and test")
    if head in WORDS:
        if len(words) > 1 or a.llm or a.horizon is not None or a.refit:
            return _usage("score takes no arguments")
        sc = score(Bank())
        print(json.dumps(sc, indent=1) if a.json else scorecard(sc))
        return 0
    log = (lambda m: print(m, file=sys.stderr, flush=True))
    try:
        results = run(Bank(), words, h=a.horizon or 5, llm=a.llm, refit=a.refit, log=log)
    except ForecastError as e:
        return _usage(e.message + (f" ({e.hint})" if e.hint else ""))
    if a.json:
        print(json.dumps(plain([{k: v for k, v in r.items()} for r in results]), indent=1))
    else:
        for r in results:
            print(card(r) + "\n", flush=True)
            if not r["ok"]:
                print(f"brain forecast: {r['error']}", file=sys.stderr)
    return 0 if results and all(r["ok"] for r in results) else 1


def add_cli(sub) -> None:
    """Register ``brain forecast TICKER [--horizon 5|21] [--llm]``, ``brain forecast score``, and the declared
    tests' ``brain forecast declare|test`` (``brain.research.forecast_tests``, imported here so a broken test
    module never takes the card down)."""
    p = sub.add_parser("forecast", help="weather-style forecast (paper): the likely price range and the chance to beat "
                                        "SPY; also: forecast score | declare | test ID...",
                       description="Forecast a stock over the next 5 or 21 sessions after SPY's latest close "
                                   "(docs/FORECAST.md): its 68% and 90% price bands and its chance to beat SPY, "
                                   "recorded as a fact and scored later. Paper only: nothing here trades.")
    p.add_argument("words", nargs="*", metavar="TICKER | score | declare | test ID...",
                   help="tickers to forecast (upper case), or one of the lowercase words score, declare, test")
    p.add_argument("--horizon", type=int, choices=HORIZONS, default=None, help="sessions ahead: 5 (default) or 21")
    p.add_argument("--llm", action="store_true", help="also ask the local model (recorded as F1-llm)")
    p.add_argument("--refit", action="store_true", help="retrain the live model now, even if the cached one is fresh")
    p.add_argument("--json", action="store_true", help="print the forecasts (or the score) as JSON")
    tests, why = None, None
    try:
        from brain.research import forecast_tests
        tests = forecast_tests.add_cli(p)
    except Exception as e:  # noqa: BLE001 - the card works without the declared tests' commands
        why = f"{type(e).__name__}: {e}"
    p.set_defaults(fn=functools.partial(_cmd, tests=tests, tests_error=why))


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser(prog="brain")
    add_cli(ap.add_subparsers(dest="cmd", required=True))
    ns = ap.parse_args()
    raise SystemExit(ns.fn(ns))
