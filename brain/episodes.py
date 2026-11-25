"""Earnings episodes: every past results report with what was expected before it, how the stock reacted, what
followed, and why, in a fixed form (docs/EARNINGS.md).

Asked for on 2026-11-25: "to predict the future of something, you need to know the past and present, not just what
but how and why too." The repo's failed direction tests all asked "this happened, so what next?" A stock moves on
the gap between what happens and what was expected, so each episode keeps the expectation as the price held it.
There is no point-in-time consensus history in the bank, and no history of option prices, so the expectation is
measured from prices, the market's own expectation: how far the stock ran into the report, how it usually reacts,
and the market's mood.

An episode is one earnings release (an 8-K item 2.02, ``brain/events.py``). Its report session ``i`` is the first
close at or after the release became public; ``p = i - 1`` is the last close before it, the moment a trade "before
the report" is made. Everything in the setup is known at the close of ``p``:

- ``run5``, ``run21``, ``run63``: the stock's excess return over SPY in the 5, 21 and 63 sessions to ``p``, in its
  own usual moves for that many days (``day_vol * sqrt(n)``);
- ``hi52``: how far below its 52-week high it closed at ``p`` (0 at the high, -0.4 for 40% below);
- ``med8``, ``last``: the size of its last eight report reactions (median) and of the last one, as M1 has them;
- ``mean8``: the mean signed reaction of its last eight reports, in its usual report moves (``med8``); ``up8``:
  the share of them that rose;
- ``day_vol``: the standard deviation of its daily excess returns over 160 ordinary sessions;
- ``vix`` and ``spy21``: the VIX close and SPY's return over the 21 sessions to ``p``.

What happened:

- ``r1``: the excess return from the close of ``p`` to the close of ``i``, the report session (a trade made
  before the report and closed the next session);
- ``car3``: days -1..+1 (from the close two sessions before ``i`` to the close after it), as M1;
- ``post5``, ``post21``: the excess return from the close of ``i`` over the next 5 and 21 sessions, the drift
  after the reaction. A window that has not closed is None.

``why`` names the episode in a fixed form, from its setup and its reaction measured in its own usual report moves:
the setup is ``ran up`` (``run21`` at least +1), ``sold off`` (at most -1) or ``flat``; the reaction is ``rose`` or
``fell`` (beyond a quarter of its usual move) or ``flat``. ``ran up`` and ``fell`` is ``priced in``: the good news
was already in the price; ``sold off`` and ``rose`` is ``relief``; ``ran up`` and ``rose`` is ``confirmed``;
``sold off`` and ``fell`` is ``capitulation``; a flat setup gives ``surprise up`` or ``surprise down``.

Point-in-time: prices are read as the bank held them; an episode's analogs come only from episodes whose own
windows had closed before its ``p``. EDGAR's acceptance times run a few hours late (``brain/events.py``): an
after-close release lands on the right session, and a release made during the session can land a session late,
which shifts its reaction into ``p``; earnings releases inside trading hours are rare.
"""
from __future__ import annotations

import gc
import gzip
import json
import math
from pathlib import Path

import numpy as np

from brain import config, events
from brain.bank import Bank

BENCH, VIX = "SPY", "^VIX"
UNIVERSES = ("us_broad", "us_small", "us_micro", "us_watch")
LOOKBACK, MIN_DAYS, MIN_PREV = 160, 60, 4
RUN = (5, 21, 63)
CHUNK = 200
FEATS = ("run5", "run21", "run63", "hi52", "mean8", "up8", "med8", "vix", "spy21")
RAN = 1.0            # run21 in usual 21-session moves for "ran up" / "sold off"
MOVED = 0.25         # a reaction beyond this share of its usual report move "rose" / "fell"
TRAIN_START, TRAIN_END = "2011-01-01", "2023-12-31"


def path() -> Path:
    return config.ROOT / "var" / "episodes" / "episodes.json.gz"


# ---------- pure ----------
def _excess(adj, bclose, a: int, b: int) -> float | None:
    if a < 0 or b >= len(adj) or not (adj[a] > 0) or not (bclose[a] > 0) or not np.isfinite(bclose[b]):
        return None
    x = (adj[b] / adj[a] - 1) - (bclose[b] / bclose[a] - 1)
    return float(x) if np.isfinite(x) else None


def day_vol(adj, bclose, end: int, report_idx: list[int]) -> float | None:
    """Standard deviation of daily excess returns over the LOOKBACK sessions to ``end``, leaving out the sessions
    around earlier reports (-2..+3)."""
    lo = max(1, end - LOOKBACK + 1)
    k = np.arange(lo, end + 1)
    keep = np.ones(len(k), bool)
    for i in report_idx:
        keep &= ~((k >= i - 2) & (k <= i + 3))
    k = k[keep]
    with np.errstate(all="ignore"):
        r = (adj[k] / adj[k - 1] - 1) - (bclose[k] / bclose[k - 1] - 1)
    r = r[np.isfinite(r)]
    return float(np.std(r)) if len(r) >= MIN_DAYS else None


def setup(adj, bclose, vclose, p: int, prev: list[tuple[int, float]]) -> dict | None:
    """The setup known at the close of ``p``. ``prev`` holds (report index, signed car3) of earlier reports; only
    those whose window (to index + 1) had closed by ``p`` count. ``bclose`` and ``vclose`` are aligned to the
    stock's dates. None without enough history."""
    done = [(i, m) for i, m in prev if i + 1 <= p]
    if len(done) < MIN_PREV or p < max(RUN) or p >= len(adj):
        return None
    dv = day_vol(adj, bclose, p, [i for i, _ in prev])
    if not dv or dv <= 0 or not np.isfinite(vclose[p]) or vclose[p] <= 0:
        return None
    out = {"day_vol": dv, "vix": float(vclose[p])}
    for n in RUN:
        x = _excess(adj, bclose, p - n, p)
        if x is None:
            return None
        out[f"run{n}"] = x / (dv * math.sqrt(n))
    lo = max(0, p - 251)
    hi = np.nanmax(adj[lo:p + 1])
    out["hi52"] = float(adj[p] / hi - 1) if hi > 0 else 0.0
    moves = [m for _, m in done]
    size = [abs(m) for m in moves]
    med8 = float(np.median(size[-8:]))
    out["med8"] = med8
    out["last"] = size[-1]
    out["mean8"] = float(np.mean(moves[-8:]) / med8) if med8 > 0 else 0.0
    out["up8"] = float(np.mean([m > 0 for m in moves[-8:]]))
    out["n_prev"] = len(done)
    out["spy21"] = float(bclose[p] / bclose[p - 21] - 1) if bclose[p - 21] > 0 else 0.0
    return out


def why(run21: float, r1: float, med8: float) -> str:
    """The episode's name in a fixed form (module docstring)."""
    s = "ran up" if run21 >= RAN else "sold off" if run21 <= -RAN else "flat"
    z = r1 / med8 if med8 > 0 else 0.0
    m = "rose" if z > MOVED else "fell" if z < -MOVED else "flat"
    if m == "flat":
        return "no reaction"
    return {("ran up", "fell"): "priced in", ("sold off", "rose"): "relief", ("ran up", "rose"): "confirmed",
            ("sold off", "fell"): "capitulation", ("flat", "rose"): "surprise up",
            ("flat", "fell"): "surprise down"}[(s, m)]


def episode_rows(dates, adj, bclose, vclose, firsts: list[int], known: list[str]) -> list[dict]:
    """One row per release (report session ``firsts[k]``, public at ``known[k]``)."""
    rows, prev = [], []
    for i, ka in zip(firsts, known):
        if i - 1 < 1 or i >= len(dates):
            continue
        p = i - 1
        car3 = _excess(adj, bclose, i - 2, i + 1) if i - 2 >= 0 else None
        r = {"date": str(dates[i]), "pre": str(dates[p]), "known_at": ka,
             "r1": _excess(adj, bclose, p, i), "car3": car3,
             "post5": _excess(adj, bclose, i, i + 5), "post21": _excess(adj, bclose, i, i + 21)}
        s = setup(adj, bclose, vclose, p, prev)
        if s:
            r.update(s)
            if r["r1"] is not None:
                r["why"] = why(s["run21"], r["r1"], s["med8"])
        rows.append(r)
        if car3 is not None:
            prev.append((i, car3))
    return rows


def aligned(dates: np.ndarray, odates: np.ndarray, oclose: np.ndarray) -> np.ndarray:
    """Another series' close on or before each of ``dates`` (NaN before its first)."""
    j = np.searchsorted(odates, dates, side="right") - 1
    return np.where(j >= 0, oclose[np.clip(j, 0, None)], np.nan)


# ---------- from the bank ----------
def ticker_episodes(view, book, ticker: str) -> list[dict]:
    t = ticker.upper()
    if not (book.has(t) and book.has(BENCH) and book.has(VIX)):
        return []
    dates, adj = book.adj_upto(view, t)
    bdates, badj = book.adj_upto(view, BENCH)
    vdates, vclose = book.adj_upto(view, VIX)
    bclose = aligned(dates, bdates, badj)
    vix = aligned(dates, vdates, book.close[VIX][:len(vdates)])
    evs = events.distinct(events.earnings_events(view, t))
    firsts = [events.event_index(dates, e.known_at) for e in evs]
    rows = episode_rows(dates, adj, bclose, vix, firsts, [e.known_at for e in evs])
    for r in rows:
        r["t"] = t
    return rows


def universe_tickers(names=UNIVERSES) -> dict[str, str]:
    """Ticker -> universe band, first band wins."""
    out: dict[str, str] = {}
    for n in names:
        p = config.ROOT / "universe" / f"{n}.json"
        if p.exists():
            for t in json.loads(p.read_text())["tickers"]:
                out.setdefault(t.upper(), n.replace("us_", ""))
    return out


def build(bank: Bank, tickers: dict[str, str] | None = None, log=print) -> list[dict]:
    """Every episode of ``tickers`` in the bank, with the band each came from."""
    from brain.metrics import PriceBook
    tickers = tickers or universe_tickers()
    names = list(tickers)
    out: list[dict] = []
    for c in range(0, len(names), CHUNK):
        part = names[c:c + CHUNK]
        b = Bank(bank.root)
        v = b.view()
        book = PriceBook(b, [*part, BENCH, VIX])
        for t in part:
            for r in ticker_episodes(v, book, t):
                r["band"] = tickers[t]
                out.append(r)
        if log:
            log(f"{min(c + CHUNK, len(names))}/{len(names)} tickers, {len(out)} episodes")
        del b, v, book
        gc.collect()
    return out


def save(rows: list[dict], p: Path | None = None) -> Path:
    p = p or path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return p


def load(p: Path | None = None) -> list[dict]:
    p = p or path()
    if not p.exists():
        return []
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def complete(rows: list[dict], key: str = "r1") -> list[dict]:
    """Episodes with a full setup and ``key`` known."""
    return [r for r in rows if r.get(key) is not None and all(r.get(f) is not None for f in FEATS)]


# ---------- analogs ----------
def matrix(rows: list[dict], feats=FEATS) -> np.ndarray:
    return np.array([[float(r[f]) for f in feats] for r in rows], dtype=float)


def scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust centre and spread per column (median, 1.4826 x MAD), so a few extreme setups do not set the scale."""
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0) * 1.4826
    return med, np.where(mad > 0, mad, 1.0)


def nearest(pool: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` rows of ``pool`` (already scaled) closest to ``x`` (scaled), nearest first."""
    d = np.sqrt(((pool - x) ** 2).sum(axis=1))
    k = min(k, len(d))
    idx = np.argpartition(d, k - 1)[:k]
    return idx[np.argsort(d[idx])]
