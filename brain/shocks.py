"""Shock radar (S1): the chance that a stock moves 20% or more against SPY, either way, in one session of the next five.

Surprises cannot be timed or signed from public data, or the price would already show them. They are not spread
evenly, though: they land on small, cheap, jumpy stocks, on drug companies waiting for an answer, on stocks that
moved hard lately and on stocks about to report. S1 ranks every stock by that chance each week.

A row is one stock at one week's last session (``end``). Everything in it was known at that close:

- ``lv20``, ``lv60``, ``lv250``: the log standard deviation of its daily returns over SPY's, 20/60/250 sessions;
- ``lmax250``: the log of its largest one-day move against SPY over 250 sessions; ``n20``, ``n10``: how many of
  those sessions moved 20%+ and 10%+;
- ``a1``, ``a5``: the latest session's move and the largest of the last five; ``big1``: the latest was 10%+;
- ``ret5``, ``ret20``, ``ret60``: the log return over SPY over 5/20/60 sessions;
- ``logp``: the log of the (unadjusted) close, ``pen``: under $5; ``ldv``: the log of the 20-session average
  dollar volume; ``volsurge``: the log of the 5-session over the 60-session average volume;
- ``hi52``, ``lo52``: the distance from the 52-week high and low;
- ``news5``: 8-Ks other than results in the last five sessions; ``offer30``: a prospectus or shelf filing (424B,
  S-1, S-3, F-1, F-3) in the last 30, a sign a share sale is near;
- ``earn_win``: last year's report for the same quarter, moved on 52 weeks, falls in the next eight days (only
  past reports are used, so the expected date is itself point in time); ``earn_known`` says whether a year of
  reports exists;
- ``bio``: the SEC industry code is drugs, diagnostics, biologics or biology research (2834, 2835, 2836, 8731);
  ``bio_known`` says whether the code is in the bank (codes are today's: a company that changed industry is
  read at its current code);
- ``vix``: the log VIX close.

The outcome ``y`` is 1 when any of the next five sessions moved 20%+ against SPY; ``up`` and ``down`` split it.

The model is a logistic regression with a small ridge on the features, standardized and clipped at the training
years' 0.1 and 99.9 percentiles, fitted on 2011-2023. The baseline is the same fit on the volatility history alone
(``lv20``, ``lv60``, ``lv250``, ``lmax250``). Both live in ``knowledge/shocks_s1.json``; the years after
TRAIN_END are the declared test's (``brain/research/shocktest.py``, docs/SHOCKS.md).

A chance of a shock is a size, not a direction: nothing here says which way, and nothing here trades.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import numpy as np

from brain import config, events, series
from brain.bank import Bank, canon

H = 5                   # sessions ahead
SHOCK = 0.20            # a shock: one session's return beats or trails SPY's by this much
TRAIN_START, TRAIN_END = "2011-01-01", "2023-12-31"
BENCH, VIX = "SPY", "^VIX"
UNIVERSES = ("us_broad", "us_small", "us_micro")
BIO_SIC = ("2834", "2835", "2836", "8731")
OFFER_FORMS = ("424B", "S-1", "S-3", "F-1", "F-3")
FEATS = ("lv20", "lv60", "lv250", "lmax250", "n20", "n10", "a1", "a5", "big1", "ret5", "ret20", "ret60", "logp",
         "logp2", "pen", "ldv", "ldv2", "lmax2", "volsurge", "hi52", "lo52", "news5", "offer30", "earn_win",
         "earn_known", "bio", "bio_known", "vix")
BASELINE = ("lv20", "lv60", "lv250", "lmax250")
LAM = 1.0
CLIP = (0.1, 99.9)
MIN_RET = 150           # returns needed behind the 250-session features
BAD_UP, BAD_DOWN, MIN_PRICE = 5.0, -0.95, 0.05   # a day beyond these is a data error, not a move
DEFAULT_MODEL = config.ROOT / "knowledge" / "shocks_s1.json"
CHUNK = 200


# ---------- features, from arrays (pure) ----------
def _roll(x: np.ndarray, k: int, fn: str, minp: int | None = None) -> np.ndarray:
    """Trailing window statistic ending at each index (inclusive), NaN until ``minp`` finite values."""
    import pandas as pd
    return getattr(pd.Series(x).rolling(k, min_periods=minp or k), fn)().to_numpy()


def _fwd_any(mask: np.ndarray, n: int) -> np.ndarray:
    """1.0 if any of the next ``n`` entries is True, NaN where the window runs past the end."""
    c = np.r_[0, np.cumsum(mask.astype(float))]
    out = np.full(len(mask), np.nan)
    i = np.arange(max(len(mask) - n, 0))
    out[i] = ((c[i + n + 1] - c[i + 1]) > 0).astype(float)
    return out


def week_ends(dates: np.ndarray) -> np.ndarray:
    """Indices of each completed week's last session (ISO weeks); the current, unfinished week is left out."""
    d = [date.fromisoformat(str(x)) for x in dates]
    wk = np.array([x.isocalendar()[0] * 100 + x.isocalendar()[1] for x in d])
    return np.flatnonzero(np.r_[wk[:-1] != wk[1:], False])


def earn_window(dates: np.ndarray, reports: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """For each session: 1.0 when a report made a year ago, moved on 52 weeks, lands 1-8 days ahead; and whether
    the estimate can be made at all (0 until the first report's estimate is within reach)."""
    win, known = np.zeros(len(dates)), np.zeros(len(dates))
    if not reports:
        return win, known
    rep = np.array(sorted(np.datetime64(r) for r in reports))
    est = rep + np.timedelta64(364, "D")
    d = np.array([np.datetime64(str(x)) for x in dates])
    lo = np.searchsorted(est, d + np.timedelta64(1, "D"))
    hi = np.searchsorted(est, d + np.timedelta64(9, "D"))
    known = (d + np.timedelta64(8, "D") >= est[0]).astype(float)
    win = np.where(known > 0, (hi > lo).astype(float), 0.0)
    return win, known


def frame(dates, close, adj, vol, bclose, vclose, reports: list[str], news_idx: list[int], offer_idx: list[int],
          bio: float | None) -> dict[str, np.ndarray]:
    """Every feature and outcome for every session of one stock. Arrays are aligned to ``dates``; ``bclose`` and
    ``vclose`` are SPY's adjusted close and the VIX close on those dates. ``news_idx`` and ``offer_idx`` are the
    sessions at whose close an 8-K (other than results) or an offering filing was first known."""
    n = len(dates)
    with np.errstate(all="ignore"):
        r = np.r_[np.nan, adj[1:] / adj[:-1] - 1]
        rb = np.r_[np.nan, bclose[1:] / bclose[:-1] - 1]
    bad = (r > BAD_UP) | (r < BAD_DOWN) | (close < MIN_PRICE)
    r = np.where(bad, np.nan, r)
    e = r - rb
    a = np.abs(e)
    news, offer = np.zeros(n), np.zeros(n)
    for k in news_idx:
        if 0 <= k < n:
            news[k] += 1
    for k in offer_idx:
        if 0 <= k < n:
            offer[k] = 1
    win, eknown = earn_window(dates, reports)
    lr = np.log1p(np.nan_to_num(e, nan=0.0))
    with np.errstate(all="ignore"):
        f = {
            "lv20": np.log(_roll(e, 20, "std", 15)), "lv60": np.log(_roll(e, 60, "std", 45)),
            "lv250": np.log(_roll(e, 250, "std", MIN_RET)),
            "lmax250": np.log(_roll(a, 250, "max", MIN_RET) + 1e-4),
            "n20": _roll((a >= 0.20).astype(float), 250, "sum", MIN_RET),
            "n10": _roll((a >= 0.10).astype(float), 250, "sum", MIN_RET),
            "a1": a, "a5": _roll(a, 5, "max", 4),
            "ret5": _roll(lr, 5, "sum"), "ret20": _roll(lr, 20, "sum"), "ret60": _roll(lr, 60, "sum"),
            "logp": np.log(close), "ldv": np.log(_roll(close * vol, 20, "mean", 15) + 1),
            "volsurge": np.log((_roll(vol, 5, "mean", 4) + 1) / (_roll(vol, 60, "mean", 45) + 1)),
            "hi52": adj / _roll(adj, 252, "max", 200) - 1, "lo52": adj / _roll(adj, 252, "min", 200) - 1,
            "news5": _roll(news, 5, "sum"), "offer30": (_roll(offer, 30, "sum") > 0).astype(float),
            "earn_win": win, "earn_known": eknown,
            "bio": np.full(n, 0.0 if bio is None else float(bio)), "bio_known": np.full(n, 0.0 if bio is None else 1.0),
            "vix": np.log(vclose),
        }
    f["big1"] = (np.nan_to_num(a) >= 0.10).astype(float)
    f["logp2"], f["ldv2"], f["lmax2"] = f["logp"] ** 2, f["ldv"] ** 2, f["lmax250"] ** 2
    f["pen"] = (close < 5).astype(float)
    f["y"] = _fwd_any(np.nan_to_num(a) >= SHOCK, H)
    f["up"] = _fwd_any(np.nan_to_num(e) >= SHOCK, H)
    f["down"] = _fwd_any(np.nan_to_num(e) <= -SHOCK, H)
    # a window with a missing return is not an outcome
    miss = _fwd_any(~np.isfinite(e), H)
    for k in ("y", "up", "down"):
        f[k] = np.where(miss > 0, np.nan, f[k])
    return f


def rows_at(f: dict[str, np.ndarray], dates: np.ndarray, idx, ticker: str, outcome: bool = True) -> list[dict]:
    """Rows at the sessions ``idx`` whose features are all finite (and, with ``outcome``, whose window closed)."""
    out = []
    for i in idx:
        row = {k: float(f[k][i]) for k in FEATS}
        if not all(np.isfinite(v) for v in row.values()):
            continue
        if outcome:
            if not np.isfinite(f["y"][i]):
                continue
            row.update(y=float(f["y"][i]), up=float(f["up"][i]), down=float(f["down"][i]))
        out.append({"t": ticker, "date": str(dates[i]), **row})
    return out


# ---------- from the bank ----------
def _series(view, t: str):
    fct = view.latest(t, "series.price_daily")
    return None if fct is None else series.load(fct.value)


def _aligned(df, col: str, cal: np.ndarray) -> np.ndarray:
    import pandas as pd
    s = pd.Series(df[col].to_numpy(float), index=df["date"].to_numpy(dtype=str))
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(cal).ffill(limit=3).to_numpy()


def market(view) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """SPY's calendar from 2010 with its adjusted close and the VIX close on each session, as known at ``view``."""
    spy, vix = _series(view, BENCH), _series(view, VIX)
    if spy is None or vix is None:
        return None
    cal = spy["date"].to_numpy(dtype=str)
    cal = cal[(cal >= "2010-01-04") & (cal <= view.t[:10])]
    return cal, _aligned(spy, "adjclose", cal), _aligned(vix, "close", cal)


def filings(view, t: str, cal: np.ndarray) -> tuple[list[str], list[int], list[int]]:
    """Results dates, other-8-K sessions and offering sessions for ``t``, as known at ``view``."""
    reports, news, offer = set(), [], []
    for f in view.facts(t, "filing"):
        form = str(f.value).upper()
        k = events.event_index(cal, f.known_at)
        if k >= len(cal):
            continue
        if form in ("8-K", "8-K/A"):
            items = {x.strip() for x in str(f.context.get("items") or "").split(",")}
            if "2.02" in items:
                reports.add(str(cal[k]))
            elif items - {"9.01", "7.01"}:
                news.append(k)
        elif form.startswith(OFFER_FORMS):
            offer.append(k)
    return sorted(reports), news, offer


def is_bio(view, t: str) -> float | None:
    f = view.latest(t, "identity.sic")
    return None if f is None or f.value in (None, "") else float(str(f.value) in BIO_SIC)


def ticker_frame(view, t: str, mk) -> tuple[np.ndarray, dict] | None:
    cal, bclose, vclose = mk
    df = _series(view, t)
    if df is None or len(df) < MIN_RET + 60:
        return None
    close, adj, vol = (_aligned(df, c, cal) for c in ("close", "adjclose", "volume"))
    reports, news, offer = filings(view, t, cal)
    return cal, frame(cal, close, adj, vol, bclose, vclose, reports, news, offer, is_bio(view, t))


def universe_tickers(names=UNIVERSES) -> list[str]:
    out: list[str] = []
    for n in names:
        for t in json.loads((config.ROOT / "universe" / f"{n}.json").read_text())["tickers"]:
            if t not in out:
                out.append(t)
    return out


def dataset(bank: Bank, tickers: list[str], first: str, last: str, log=print) -> list[dict]:
    """Weekly rows whose week ends in [first, last] and whose five-session outcome is known. A fresh bank per CHUNK
    tickers keeps memory flat."""
    out = []
    for c in range(0, len(tickers), CHUNK):
        part = tickers[c:c + CHUNK]
        b = Bank(bank.root)
        v = b.view()
        mk = market(v)
        if mk is None:
            raise SystemExit(f"no {BENCH} or {VIX} prices in the bank: python -m brain ingest {BENCH} {VIX}")
        ends = week_ends(mk[0])
        ends = ends[(mk[0][ends] >= first) & (mk[0][ends] <= last)]
        for t in part:
            got = ticker_frame(v, t, mk)
            if got:
                out += rows_at(got[1], got[0], ends, t)
        if log:
            log(f"{min(c + CHUNK, len(tickers))}/{len(tickers)} tickers, {len(out)} stock-weeks")
        del b, v
        gc.collect()
    return out


# ---------- the model ----------
def _matrix(rows: list[dict], feats) -> np.ndarray:
    return np.array([[r[f] for f in feats] for r in rows], float)


def _logit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge logistic regression by Newton steps; the intercept (column 0) is not shrunk."""
    w = np.zeros(X.shape[1])
    m = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    w[0] = np.log(m / (1 - m))
    pen = lam * np.eye(X.shape[1])
    pen[0, 0] = 0.0
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (y - p) - pen @ w
        step = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + pen, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def fit(rows: list[dict], feats=FEATS, lam: float = LAM) -> dict:
    Z = _matrix(rows, feats)
    lo, hi = np.percentile(Z, CLIP[0], axis=0), np.percentile(Z, CLIP[1], axis=0)
    Z = np.clip(Z, lo, hi)
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    X = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
    y = np.array([r["y"] for r in rows], float)
    w = _logit(X, y, lam)
    return {"feats": list(feats), "lo": lo.tolist(), "hi": hi.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
            "w": w.tolist(), "lam": lam, "n": len(rows), "rate": float(y.mean())}


def chances(part: dict, rows: list[dict]) -> np.ndarray:
    """The fitted chance of a shock for each row."""
    if not rows:
        return np.zeros(0)
    Z = np.clip(_matrix(rows, part["feats"]), part["lo"], part["hi"])
    X = np.column_stack([np.ones(len(Z)), (Z - np.array(part["mu"])) / np.array(part["sd"])])
    return 1 / (1 + np.exp(-np.clip(X @ np.array(part["w"]), -30, 30)))


def model_path() -> Path:
    """The fitted models' file: ``knowledge/shocks_s1.json``, or BRAIN_SHOCKS_MODEL (tests, trials)."""
    return Path(os.environ.get("BRAIN_SHOCKS_MODEL") or DEFAULT_MODEL)


def load(path: Path | None = None) -> dict | None:
    p = Path(path or model_path())
    return json.loads(p.read_text()) if p.exists() else None


def model_sha(path: Path | None = None) -> str:
    return hashlib.sha256(Path(path or model_path()).read_bytes()).hexdigest()[:16]


def fit_and_save(bank: Bank, path: Path | None = None, log=print, tickers=None) -> dict:
    """Fit S1 and the baseline on the universe's stock-weeks from TRAIN_START to TRAIN_END and save both. Nothing
    after TRAIN_END is read into the fit: those years belong to the declared test."""
    rows = dataset(bank, tickers or universe_tickers(), TRAIN_START, TRAIN_END, log=log)
    if not rows:
        raise SystemExit("no stock-weeks to fit on")
    out = {"name": "S1", "train": [TRAIN_START, TRAIN_END], "universes": list(UNIVERSES), "h": H, "shock": SHOCK,
           "s1": fit(rows, FEATS), "baseline": fit(rows, BASELINE)}
    p = Path(path or model_path())
    p.write_text(canon(out) + "\n")
    if log:
        log(f"fitted on {len(rows)} stock-weeks ({TRAIN_START}..{TRAIN_END}), shock rate "
            f"{out['s1']['rate'] * 100:.2f}% a week; saved {p}")
    return out


# ---------- live ----------
def radar(bank: Bank, tickers: list[str], log=None) -> list[dict]:
    """Each ticker's chance of a 20%+ session against SPY in the next five, from the bank's latest close, highest
    first. A row whose features are not all known is left out."""
    model = load()
    if model is None:
        raise SystemExit("no fitted model: run python -m brain shocks fit")
    out = []
    for c in range(0, len(tickers), CHUNK):
        part = tickers[c:c + CHUNK]
        b = Bank(bank.root)
        v = b.view()
        mk = market(v)
        if mk is None:
            raise SystemExit(f"no {BENCH} or {VIX} prices in the bank: python -m brain ingest {BENCH} {VIX}")
        for t in part:
            got = ticker_frame(v, t.upper(), mk)
            if not got:
                continue
            cal, f = got
            i = len(cal) - 1
            while i > 0 and not np.isfinite(f["a1"][i]):
                i -= 1
            rows = rows_at(f, cal, [i], t.upper(), outcome=False)
            if rows:
                r = rows[0]
                r["chance"] = float(chances(model["s1"], [r])[0])
                r["baseline"] = float(chances(model["baseline"], [r])[0])
                out.append(r)
        if log:
            log(f"{min(c + CHUNK, len(tickers))}/{len(tickers)} tickers")
        del b, v
        gc.collect()
    return sorted(out, key=lambda r: -r["chance"])


def _cmd(a) -> int:
    bank = Bank()
    if a.action == "fit":
        fit_and_save(bank)
        return 0
    tickers = [t.upper() for t in a.tickers] or universe_tickers()
    rows = radar(bank, tickers)
    model = load()
    print(f"Shock radar (S1, fitted {model['train'][0]}..{model['train'][1]}): the chance of a 20%+ session against "
          f"SPY, either way, in the next {H} sessions. The average stock-week: {model['s1']['rate'] * 100:.1f}%. "
          "A size, not a direction.")
    for r in rows[:a.top]:
        print(f"  {r['t']:6} {r['chance'] * 100:5.1f}%  (volatility alone {r['baseline'] * 100:4.1f}%)  close "
              f"{np.exp(r['logp']):.2f} on {r['date']}" + ("  drug/biotech" if r["bio"] else "")
              + ("  offering filed" if r["offer30"] else "") + ("  report due" if r["earn_win"] else ""))
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("shocks", help="the chance of a 20%+ one-day move in the next five sessions (docs/SHOCKS.md)")
    p.add_argument("action", choices=["fit", "radar"])
    p.add_argument("tickers", nargs="*")
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(fn=_cmd)
