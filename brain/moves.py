"""Earnings move size (M1): how far a stock is likely to move over its earnings release, whichever way.

Direction has failed every test in this repo; size is different, because how much a stock moves persists. The
outcome is the unsigned ``car_3d`` of ``brain.events.reaction``: the stock's excess return over SPY from the close
two sessions before the release's first session to the close one session after it (days -1..+1).

Known before a release, at the close two sessions before its first one (``end``):

- ``med8``: the median move of its last eight releases (at least four), counting only those whose three days had
  closed by then;
- ``last``: the most recent of those moves;
- ``day_vol``: the standard deviation of its daily excess returns over the 160 sessions to ``end``, leaving out
  the sessions around earlier releases (-2..+3);
- ``vix``: the VIX close on or before ``end``.

The model is least squares on logs, ``log(move + EPS) = b0 + b1 log(med8 + EPS) + b2 log(last + EPS)
+ b3 log(day_vol * sqrt 3) + b4 log(vix) + e``, with the spread of ``e`` over the training years giving the range:
the p-quantile is ``exp(fit + quantile_p(e)) - EPS``, and the chance of moving more than x is the share of training
residuals above ``log(x + EPS) - fit``. The history-only baseline is the same fit on ``med8`` alone.

The fitted models live in ``knowledge/moves_m1.json``; their training years stop at TRAIN_END, and the years after
are the declared test's (``brain/research/movetest.py``, docs/MOVES.md). A size is not a direction, and nothing here
trades.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from brain import config, events
from brain.bank import Bank, canon

EPS = 0.005
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
GRID = tuple(round(0.005 * k, 3) for k in range(1, 200))   # residual quantiles kept for chances
FEATS = ("med8", "last", "day_vol", "vix")
BASELINE = ("med8",)
LOOKBACK = 160      # sessions of ordinary days behind day_vol
MIN_DAYS = 60       # ordinary days needed for day_vol
MIN_PREV = 4        # earlier moves needed for med8
TRAIN_START, TRAIN_END = "2011-01-01", "2023-12-31"
VIX = "^VIX"
BENCH = "SPY"
DEFAULT_MODEL = config.ROOT / "knowledge" / "moves_m1.json"
UNIVERSES = ("us_broad", "us_small", "us_micro")
CHUNK = 200


# ---------- features, from arrays (pure) ----------
def bench_on(dates: np.ndarray, bdates: np.ndarray, badj: np.ndarray) -> np.ndarray:
    """The benchmark's close on or before each of ``dates`` (NaN before its first)."""
    j = np.searchsorted(bdates, dates, side="right") - 1
    return np.where(j >= 0, badj[np.clip(j, 0, None)], np.nan)


def snapshot(dates, adj, bclose, vdates, vclose, prev: list[tuple[int, float]], end: int) -> dict | None:
    """The four features known at the close of session ``end``. ``prev`` holds (first-session index, move) of
    earlier releases; only those whose window (to index + 1) had closed by ``end`` count. ``bclose`` is the
    benchmark's close aligned to ``dates``. None when there are fewer than MIN_PREV earlier moves, too few
    ordinary days, or no VIX close."""
    done = [(i, m) for i, m in prev if i + 1 <= end]
    if len(done) < MIN_PREV or end < 1 or end >= len(dates):
        return None
    moves = [m for _, m in done]
    lo = max(1, end - LOOKBACK + 1)
    k = np.arange(lo, end + 1)
    keep = np.ones(len(k), bool)
    for i, _ in prev:
        keep &= ~((k >= i - 2) & (k <= i + 3))
    k = k[keep]
    with np.errstate(all="ignore"):
        r = (adj[k] / adj[k - 1] - 1) - (bclose[k] / bclose[k - 1] - 1)
    r = r[np.isfinite(r)]
    if len(r) < MIN_DAYS:
        return None
    j = int(np.searchsorted(vdates, dates[end], side="right")) - 1
    if j < 0 or not np.isfinite(vclose[j]) or vclose[j] <= 0:
        return None
    return {"med8": float(np.median(moves[-8:])), "last": float(moves[-1]), "day_vol": float(np.std(r)),
            "vix": float(vclose[j]), "n_prev": len(done)}


def release_rows(dates, adj, bdates, badj, vdates, vclose, firsts: list[int], known: list[str]) -> list[dict]:
    """One row per release (first-session index ``firsts[k]``, filed at ``known[k]``): its date, its move (None
    until day +1 has closed) and the features known at the close two sessions before its first session."""
    bclose = bench_on(dates, bdates, badj)
    rows, prev = [], []
    for i, ka in zip(firsts, known):
        if i - 2 < 0:
            continue
        move = None
        if i + 1 < len(dates):
            b0, b1 = bclose[i - 2], bclose[i + 1]
            if adj[i - 2] > 0 and np.isfinite(b0) and np.isfinite(b1) and b0 > 0:
                move = abs(float((adj[i + 1] / adj[i - 2] - 1) - (b1 / b0 - 1)))
        f = snapshot(dates, adj, bclose, vdates, vclose, prev, i - 2)
        rows.append({"date": str(dates[i]) if i < len(dates) else None, "known_at": ka, "move": move,
                     **(f or {})})
        if move is not None:
            prev.append((i, move))
    return rows


# ---------- from the bank ----------
def ticker_rows(view, book, ticker: str) -> list[dict]:
    """Every distinct earnings release of ``ticker`` known at ``view``, with its move and features."""
    t = ticker.upper()
    if not (book.has(t) and book.has(BENCH) and book.has(VIX)):
        return []
    dates, adj = book.adj_upto(view, t)
    bdates, badj = book.adj_upto(view, BENCH)
    vdates = book.dates[VIX]
    vclose = book.close[VIX]
    j = int(np.searchsorted(vdates, view.t, side="right"))
    evs = events.distinct(events.earnings_events(view, t))
    firsts = [events.event_index(dates, e.known_at) for e in evs]
    rows = release_rows(dates, adj, bdates, badj, vdates[:j], vclose[:j], firsts, [e.known_at for e in evs])
    for r in rows:
        r["t"] = t
    return rows


def universe_tickers(names=UNIVERSES) -> list[str]:
    out: list[str] = []
    for n in names:
        for t in json.loads((config.ROOT / "universe" / f"{n}.json").read_text())["tickers"]:
            if t not in out:
                out.append(t)
    return out


def dataset(bank: Bank, tickers: list[str], first: str, last: str, log=print) -> list[dict]:
    """Rows of releases whose first session falls in [first, last] and whose move and features are all known. A
    fresh bank per CHUNK tickers keeps memory flat."""
    from brain.metrics import PriceBook
    out = []
    for c in range(0, len(tickers), CHUNK):
        part = tickers[c:c + CHUNK]
        b = Bank(bank.root)
        v = b.view()
        book = PriceBook(b, [*part, BENCH, VIX])
        for t in part:
            out += [r for r in ticker_rows(v, book, t)
                    if r.get("move") is not None and "med8" in r and r["date"] and first <= r["date"] <= last]
        if log:
            log(f"{min(c + CHUNK, len(tickers))}/{len(tickers)} tickers, {len(out)} releases")
        del b, v, book
        gc.collect()
    return out


# ---------- the model ----------
def design(rows: list[dict], feats=FEATS) -> np.ndarray:
    cols = [np.ones(len(rows))]
    for f in feats:
        x = np.array([r[f] for r in rows], float)
        cols.append(np.log(x * np.sqrt(3)) if f == "day_vol" else np.log(x) if f == "vix" else np.log(x + EPS))
    return np.column_stack(cols)


def fit(rows: list[dict], feats=FEATS) -> dict:
    X = design(rows, feats)
    y = np.log(np.array([r["move"] for r in rows], float) + EPS)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    return {"feats": list(feats), "beta": [float(b) for b in beta], "eps": EPS, "n": len(rows),
            "resid": [float(x) for x in np.quantile(e, GRID)], "resid_q": {str(q): float(np.quantile(e, q)) for q in QS}}


def center(model: dict, row: dict) -> float:
    return float((design([row], model["feats"]) @ np.array(model["beta"]))[0])


def quantiles(model: dict, row: dict) -> dict[float, float]:
    """The move's quantiles (fractions, e.g. 0.06 for 6%) for the QS levels."""
    mu = center(model, row)
    return {q: max(0.0, float(np.exp(mu + model["resid_q"][str(q)]) - model["eps"])) for q in QS}


def expected(model: dict, row: dict) -> float:
    """The mean move (a fraction), over the training residuals' spread: what a straddle's buyer can expect to get
    back, in move terms."""
    mu = center(model, row)
    return float(np.mean(np.maximum(np.exp(mu + np.array(model["resid"])) - model["eps"], 0.0)))


def chance_above(model: dict, row: dict, x: float) -> float:
    """The chance the move is more than ``x`` (a fraction), from the training residuals' spread."""
    th = np.log(x + model["eps"]) - center(model, row)
    return float(np.mean(np.array(model["resid"]) > th))


def pinball(qs: dict[float, float], y: float) -> float:
    """Mean pinball loss over the QS levels: lower is better; 0 only for a perfect range."""
    return float(np.mean([max(q * (y - v), (q - 1) * (y - v)) for q, v in qs.items()]))


def model_path() -> Path:
    """The fitted models' file: ``knowledge/moves_m1.json``, or BRAIN_MOVES_MODEL (tests, trials)."""
    return Path(os.environ.get("BRAIN_MOVES_MODEL") or DEFAULT_MODEL)


def load(path: Path | None = None) -> dict | None:
    p = Path(path or model_path())
    return json.loads(p.read_text()) if p.exists() else None


def model_sha(path: Path | None = None) -> str:
    return hashlib.sha256(Path(path or model_path()).read_bytes()).hexdigest()[:16]


def fit_and_save(bank: Bank, path: Path | None = None, log=print, tickers=None) -> dict:
    """Fit M1 and the baseline on the universe's releases from TRAIN_START to TRAIN_END, and save both. Nothing
    after TRAIN_END is read into the fit: those years belong to the declared test."""
    rows = dataset(bank, tickers or universe_tickers(), TRAIN_START, TRAIN_END, log=log)
    if not rows:
        raise SystemExit("no releases to fit on")
    out = {"name": "M1", "train": [TRAIN_START, TRAIN_END], "universes": list(UNIVERSES),
           "m1": fit(rows, FEATS), "baseline": fit(rows, BASELINE)}
    p = Path(path or model_path())
    p.write_text(canon(out) + "\n")
    if log:
        log(f"fitted on {len(rows)} releases ({TRAIN_START}..{TRAIN_END}); saved {p}")
    return out


# ---------- live ----------
def live(bank: Bank, ticker: str, book=None) -> dict | None:
    """The features for a ticker's next release, known at the bank's latest close (that close stands in for the
    close two sessions before the release; a report a week away is read a few sessions early)."""
    from brain.metrics import PriceBook
    t = ticker.upper()
    view = bank.view()
    book = book or PriceBook(bank, [t, BENCH, VIX])
    if not (book.has(t) and book.has(BENCH) and book.has(VIX)):
        return None
    rows = ticker_rows(view, book, t)
    prev = []
    dates, adj = book.adj_upto(view, t)
    for r in rows:
        if r.get("move") is not None and r["date"]:
            prev.append((int(np.searchsorted(dates, r["date"])), r["move"]))
    bdates, badj = book.adj_upto(view, BENCH)
    f = snapshot(dates, adj, bench_on(dates, bdates, badj), book.dates[VIX], book.close[VIX], prev, len(dates) - 1)
    if f:
        f["asof"] = str(dates[-1])
    return f


def card(model: dict, f: dict, implied: float | None = None) -> dict:
    m = model["m1"]
    out = {"quantiles": quantiles(m, f), "features": f}
    if implied:
        out["implied"] = implied
        out["chance_above_implied"] = chance_above(m, f, implied)
    return out


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _cmd(a) -> int:
    if a.action == "fit":
        fit_and_save(Bank())
        return 0
    model = load()
    if model is None:
        raise SystemExit("no fitted model: run python -m brain moves fit")
    bank = Bank()
    for t in a.tickers:
        f = live(bank, t)
        if f is None:
            print(f"{t.upper()}: not enough history (needs prices, four earlier earnings reactions and VIX)")
            continue
        c = card(model, f, a.implied / 100 if a.implied else None)
        q = c["quantiles"]
        line = (f"{t.upper()}: next report's move, either way (M1, as of {f['asof']}): middle {_pct(q[0.5])}, "
                f"half the time {_pct(q[0.25])}-{_pct(q[0.75])}, 8 in 10 {_pct(q[0.1])}-{_pct(q[0.9])}. "
                f"Its last eight: median {_pct(f['med8'])}, last {_pct(f['last'])}.")
        if a.implied:
            line += f" Chance it moves more than the {a.implied:.1f}% priced: {c['chance_above_implied'] * 100:.0f}%."
        print(line)
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("moves", help="earnings move size: fit M1, or its range for tickers' next reports (docs/MOVES.md)")
    p.add_argument("action", choices=["fit", "predict"])
    p.add_argument("tickers", nargs="*")
    p.add_argument("--implied", type=float, help="the options' priced move in percent, for the chance of beating it")
    p.set_defaults(fn=_cmd)
