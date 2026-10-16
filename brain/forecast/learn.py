"""F1, the learner: logistic and gradient boosting, blended and calibrated, plus the range model
(docs/FORECAST.md, sections 4 and 5).

What a fit does, at cutoff C:

1. It keeps the samples whose stored exit is on or before C (and dated on or after ``since``). No other
   cutoff is used anywhere here: never ``d + h`` arithmetic.
2. When more than ``max_rows`` remain, it draws that many, the same share from every date, seeded 7.
3. It splits off the samples with the latest 20% of exits. A logistic and a boosted model are fitted on
   the rest, their mean probability is taken on that slice, and Platt scaling (a logistic on the blend's
   log-odds) is fitted there.
4. It refits both models on every kept sample. A forecast is the Platt-scaled mean of their two
   probabilities.

The hyperparameters are the ones in ``LEARN`` below, frozen into every declared spec, and never tuned on a
test window. The same samples and spec give the same model. A model is plain JSON (``to_json``), never a pickle.

The range is separate: a stock's move over the window, over its EWMA volatility times sqrt(h), has
quantiles learned from the realised samples, and a band is the close times exp(quantile * sigma * sqrt(h)).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from brain.forecast import gbm

LEARN = {
    "logistic": {"l2": 1.0, "iters": 200},
    "gbm": {"trees": 120, "depth": 3, "lr": 0.05, "bins": 64, "min_leaf": 200, "subsample": 0.8, "colsample": 0.8,
            "seed": 7},
    "blend": "mean of the two probabilities",
    "calibration": "Platt scaling on the training samples with the latest 20% of exits; models refit on the rest first",
    "max_rows": 150000,          # training rows, by a date-stratified draw seeded 7 when more
    "since": "2011-01-01",
}
RANGE = {"ewma_lambda": 0.94, "lookback": 252, "quantiles": [0.05, 0.16, 0.5, 0.84, 0.95]}

CALIBRATION_SHARE = 0.20   # the latest exits that Platt scaling is fitted on (LEARN["calibration"])
DRAW_SEED = 7              # the date-stratified draw (LEARN["max_rows"])
MIN_SIGMA_RETURNS = 21     # daily returns sigma needs before it is defined
P_CLIP = 1e-6              # forecasts stay inside (0, 1)


# ---------- the model ----------
@dataclass
class Model:
    names: list[str]
    groups: list[str]
    coef: np.ndarray              # logistic coefficients, one per feature
    intercept: float
    gbm: dict                     # gbm.fit's plain dict
    platt: tuple[float, float]    # (a, b): P = sigmoid(a * logit(blend) + b)
    base_rate: float              # share of training samples with y = 1: the climatology forecast
    trained_through: str          # the cutoff C
    rows: int                     # training rows after the draw
    n_calibration: int            # of which in the calibration slice
    last_exit: str                # the latest stored exit among the training rows
    spec: dict = field(default_factory=dict)

    @property
    def sha(self) -> str:
        return hashlib.sha256(json.dumps(_plain_json(self), sort_keys=True).encode()).hexdigest()[:16]

    def card(self) -> dict:
        """The ``model`` block of a forecast (docs/FORECAST.md, section 6)."""
        return {"id": "F1", "sha": self.sha, "trained_through": self.trained_through, "rows": self.rows}


def _get(fs, name):
    v = getattr(fs, name, None)
    if v is None and isinstance(fs, dict):
        v = fs.get(name)
    return v


def _training_rows(fs, cutoff: str, since: str) -> np.ndarray:
    """Indices of samples whose stored exit is on or before ``cutoff``, dated on or after ``since``."""
    exit_ = np.asarray(_get(fs, "exit")).astype(str)
    date = np.asarray(_get(fs, "date")).astype(str)
    return np.flatnonzero((exit_ <= cutoff[:10]) & (date >= since[:10]))


def stratified_draw(dates: np.ndarray, n_max: int, seed: int = DRAW_SEED) -> np.ndarray:
    """Positions of at most ``n_max`` rows, the same share of every date (largest remainders get the
    leftover rows, earlier dates first on ties), each date's rows picked by a seeded random key. All rows
    (in order) when there are not more than ``n_max``."""
    n = len(dates)
    if n <= n_max:
        return np.arange(n)
    u, inv, cnt = np.unique(dates, return_inverse=True, return_counts=True)
    quota = cnt * (n_max / n)
    take = np.floor(quota).astype(int)
    left = n_max - int(take.sum())
    order = np.lexsort((np.arange(len(u)), -(quota - take)))
    take[order[:left]] += 1
    keys = np.random.default_rng(seed).random(n)
    srt = np.lexsort((keys, inv))                       # by date, then key
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pos = np.empty(n, dtype=int)
    pos[srt] = np.arange(n) - start[inv[srt]]           # each row's place within its date
    return np.sort(np.flatnonzero(pos < take[inv]))


def calibration_split(exits: np.ndarray, share: float = CALIBRATION_SHARE) -> np.ndarray:
    """Boolean mask of the rows with the latest exits: the fewest latest exit dates holding at least
    ``share`` of the rows. All False when there are fewer than two exit dates."""
    ex = np.asarray(exits).astype(str)
    ue, cnt = np.unique(ex, return_counts=True)
    if len(ue) < 2:
        return np.zeros(len(ex), dtype=bool)
    tail = np.cumsum(cnt[::-1])
    k = int(np.searchsorted(tail, share * len(ex))) + 1   # latest k exit dates
    k = min(k, len(ue) - 1)                               # the rest keeps at least one exit date
    return ex >= ue[::-1][k - 1]


def _logistic(X: np.ndarray, y: np.ndarray, l2: float, iters: int) -> tuple[np.ndarray, float]:
    """Penalised logistic regression by Newton's method, step halved when the loss would rise:
    minimises sum(log loss) + l2/2 * |w|^2; the intercept is not penalised."""
    n, F = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    pen = np.full(F + 1, float(l2))
    pen[-1] = 0.0
    w = np.zeros(F + 1)

    def loss(w):
        z = A @ w
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(pen * w * w))

    cur = loss(w)
    for _ in range(int(iters)):
        p = gbm.sigmoid(A @ w)
        grad = A.T @ (p - y) + pen * w
        hess = (A * (p * (1 - p))[:, None]).T @ A + np.diag(pen) + 1e-9 * np.eye(F + 1)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while True:
            nw = w - t * step
            new = loss(nw)
            if new <= cur + 1e-12 * max(1.0, abs(cur)) or t < 1e-8:
                break
            t /= 2
        moved = float(np.max(np.abs(nw - w)))
        w, gain, cur = nw, cur - new, new
        if moved < 1e-10 or gain < 1e-12 * max(1.0, abs(cur)):
            break
    return w[:-1].copy(), float(w[-1])


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _blend(coef, intercept, gbm_model, X) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return 0.5 * (gbm.sigmoid(X @ coef + intercept) + gbm.predict(gbm_model, X))


def _fit_pair(X, y, spec):
    coef, b = _logistic(X, y, spec["logistic"]["l2"], spec["logistic"]["iters"])
    g = gbm.fit(X, y, **spec["gbm"])
    return coef, b, g


def fit(fs, cutoff: str, spec=None) -> Model:
    """F1 trained on the samples of ``fs`` whose stored exit is on or before ``cutoff``, under ``spec``
    (default ``LEARN``). Deterministic."""
    spec = spec or LEARN
    cutoff = str(cutoff)[:10]
    rows = _training_rows(fs, cutoff, spec["since"])
    if len(rows) == 0:
        raise ValueError(f"no training samples with an exit on or before {cutoff}")
    date = np.asarray(_get(fs, "date")).astype(str)[rows]
    rows = rows[stratified_draw(date, int(spec["max_rows"]))]
    X = np.nan_to_num(np.asarray(_get(fs, "X"), dtype=float)[rows], nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(_get(fs, "y"), dtype=float)[rows]
    exits = np.asarray(_get(fs, "exit")).astype(str)[rows]
    if len(np.unique(y)) < 2:
        raise ValueError("training samples all have the same outcome")
    cal = calibration_split(exits)
    a, b = 1.0, 0.0
    if cal.any() and len(np.unique(y[~cal])) == 2 and len(np.unique(y[cal])) == 2:
        c0, b0, g0 = _fit_pair(X[~cal], y[~cal], spec)
        s = _logit(_blend(c0, b0, g0, X[cal]))
        (a,), b = _logistic(s[:, None], y[cal], 0.0, 100)
    coef, intercept, g = _fit_pair(X, y, spec)
    return Model(names=list(_get(fs, "names")), groups=list(_get(fs, "groups")), coef=coef, intercept=intercept,
                 gbm=g, platt=(float(a), float(b)), base_rate=float(y.mean()), trained_through=cutoff,
                 rows=len(rows), n_calibration=int(cal.sum()), last_exit=str(np.sort(exits)[-1]),
                 spec=json.loads(json.dumps(spec)))


def predict(model: Model, X) -> np.ndarray:
    """P(y = 1) for each row of ``X`` (features in ``model.names`` order), inside (0, 1)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    a, b = model.platt
    p = gbm.sigmoid(a * _logit(_blend(model.coef, model.intercept, model.gbm, X)) + b)
    return np.clip(p, P_CLIP, 1 - P_CLIP)


def drivers(model: Model, x_row, names, k: int = 5) -> list[dict]:
    """The ``k`` features that push this forecast most, by logistic coefficient times the feature's value
    (log-odds units; + toward beating SPY). A feature at 0 (missing, or at the middle rank) pushes nothing
    and is left out."""
    coef = dict(zip(model.names, np.asarray(model.coef, dtype=float)))
    group = dict(zip(model.names, model.groups))
    x = np.nan_to_num(np.asarray(x_row, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    push = [(n, float(coef.get(n, 0.0) * v)) for n, v in zip(names, x)]
    push = sorted((t for t in push if t[1] != 0.0), key=lambda t: (-abs(t[1]), t[0]))
    return [{"name": n, "group": group.get(n, ""), "push": v} for n, v in push[:k]]


def _plain_json(model: Model) -> dict:
    return {"kind": "F1", "names": list(model.names), "groups": list(model.groups),
            "coef": [float(c) for c in model.coef], "intercept": float(model.intercept), "gbm": model.gbm,
            "platt": [float(model.platt[0]), float(model.platt[1])], "base_rate": float(model.base_rate),
            "trained_through": model.trained_through, "rows": int(model.rows),
            "n_calibration": int(model.n_calibration), "last_exit": model.last_exit, "spec": model.spec}


def to_json(model: Model) -> dict:
    """Plain JSON (lists, numbers, strings). ``from_json(to_json(m))`` predicts exactly as ``m``."""
    return {**_plain_json(model), "sha": model.sha}


def from_json(d: dict) -> Model:
    if d.get("kind") != "F1":
        raise ValueError("not an F1 model")
    m = Model(names=list(d["names"]), groups=list(d["groups"]), coef=np.asarray(d["coef"], dtype=float),
              intercept=float(d["intercept"]), gbm=d["gbm"], platt=(float(d["platt"][0]), float(d["platt"][1])),
              base_rate=float(d["base_rate"]), trained_through=str(d["trained_through"]), rows=int(d["rows"]),
              n_calibration=int(d.get("n_calibration", 0)), last_exit=str(d.get("last_exit", "")),
              spec=d.get("spec") or {})
    if d.get("sha") and d["sha"] != m.sha:
        raise ValueError("model JSON does not match its sha")
    return m


# ---------- the range (section 5) ----------
def qkey(q: float) -> str:
    """0.05 -> 'q05', 0.5 -> 'q50'."""
    return f"q{round(q * 100):02d}"


def _adj(px) -> np.ndarray:
    return np.asarray(_get(px, "adj"), dtype=float)


def sigma(px, i: int) -> np.ndarray:
    """(N,) EWMA volatility of daily log returns at calendar index ``i`` (= d), from bars dated <= d only:
    the last ``lookback`` returns, the latest weighted 1 and each earlier one ``ewma_lambda`` times the next,
    over the weights of the returns that exist. NaN with fewer than 21 returns."""
    adj = _adj(px)
    lo = max(0, int(i) + 1 - (RANGE["lookback"] + 1))
    a = adj[lo:int(i) + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.diff(np.log(np.where(a > 0, a, np.nan)), axis=0)
    if len(r) == 0:
        return np.full(adj.shape[1], np.nan)
    w = RANGE["ewma_lambda"] ** np.arange(len(r))[::-1]
    ok = np.isfinite(r)
    num = (w[:, None] * np.where(ok, r * r, 0.0)).sum(axis=0)
    den = (w[:, None] * ok).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(num / den)
    out[ok.sum(axis=0) < MIN_SIGMA_RETURNS] = np.nan
    return out


def range_inputs(fs, px) -> tuple[np.ndarray, np.ndarray]:
    """Per sample of ``fs``: the stock's own total return from the close of d to the close of its stored
    exit (adjusted closes from ``px``; a missing bar takes the last close within 4 sessions), and sigma at
    d. The band is measured from the close of d (section 6), so its quantiles are learned on the same move:
    from d, not from the entry session (``fs.r_total``), which is one session shorter."""
    dates = np.asarray(_get(px, "dates")).astype(str)
    adj = _ffill(_adj(px), 4)
    col = {str(t).upper(): j for j, t in enumerate(_get(px, "tickers"))}
    d = np.asarray(_get(fs, "date")).astype(str)
    ex = np.asarray(_get(fs, "exit")).astype(str)
    tk = np.asarray(_get(fs, "ticker")).astype(str)
    j = np.array([col.get(t.upper(), -1) for t in tk], dtype=int)
    i_d = np.searchsorted(dates, d, side="right") - 1
    i_x = np.searchsorted(dates, ex, side="right") - 1
    ok = (j >= 0) & (i_d >= 0) & (i_x >= 0)
    r = np.full(len(d), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[ok] = adj[i_x[ok], j[ok]] / adj[i_d[ok], j[ok]] - 1.0
    sig = np.full(len(d), np.nan)
    for i in np.unique(i_d[ok]):
        rows = np.flatnonzero(ok & (i_d == i))
        sig[rows] = sigma(px, int(i))[j[rows]]
    return r, sig


def _ffill(a: np.ndarray, limit: int) -> np.ndarray:
    """Each column's last finite value carried forward up to ``limit`` rows."""
    return pd.DataFrame(a).ffill(limit=limit).to_numpy(dtype=float)


def fit_range(fs, cutoff: str, px=None, *, r_total=None, sig=None, h: int | None = None) -> dict:
    """Quantiles of z = log(1 + r_total) / (sigma_d * sqrt(h)) over the samples whose stored exit is on or
    before ``cutoff``. ``r_total`` (from the close of d to the close of the exit) and ``sig`` (sigma at d)
    come from ``range_inputs(fs, px)`` unless given; ``h`` is ``fs.h`` unless given."""
    cutoff = str(cutoff)[:10]
    h = int(h if h is not None else _get(fs, "h"))
    if r_total is None or sig is None:
        if px is None:
            raise ValueError("fit_range needs px, or r_total and sig")
        r_total, sig = range_inputs(fs, px)
    r = np.asarray(r_total, dtype=float)
    s = np.asarray(sig, dtype=float)
    rows = _training_rows(fs, cutoff, LEARN["since"])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.log1p(r[rows]) / (s[rows] * np.sqrt(h))
    z = z[np.isfinite(z)]
    if len(z) == 0:
        raise ValueError(f"no realised samples for the range with an exit on or before {cutoff}")
    qs = RANGE["quantiles"]
    vals = np.quantile(z, qs)
    return {"q": {qkey(q): float(v) for q, v in zip(qs, vals)}, "h": h, "n": len(z), "trained_through": cutoff}


def band(close_d, sigma_d, h: int, q) -> dict:
    """Prices close_d * exp(q * sigma_d * sqrt(h)) for each quantile key in ``q`` (a ``fit_range`` result or
    its ``q`` dict). Arrays broadcast."""
    qd = q["q"] if isinstance(q, dict) and "q" in q else q
    c = np.asarray(close_d, dtype=float)
    s = np.asarray(sigma_d, dtype=float) * np.sqrt(int(h))
    out = {}
    for k, z in qd.items():
        v = c * np.exp(float(z) * s)
        out[k] = float(v) if np.ndim(v) == 0 else v
    return out


def forecast(model: Model, rng: dict, *, ticker: str, d: str, h: int, close: float, sigma_d: float, x_row,
             entry: str | None = None, exit: str | None = None, k: int = 5) -> dict:
    """One forecast as docs/FORECAST.md section 6 writes it."""
    p = float(predict(model, np.asarray(x_row, dtype=float)[None, :])[0])
    b = band(close, sigma_d, h, rng)

    def pct(lo, hi):
        return [round((b[lo] / close - 1) * 100, 2), round((b[hi] / close - 1) * 100, 2)] \
            if np.isfinite(b[lo]) and np.isfinite(b[hi]) and close else [None, None]

    return {"ticker": ticker, "date": d, "horizon": int(h), "entry": entry, "exit": exit, "close": float(close),
            "p_beat": p, "band": {kk: (float(v) if np.isfinite(v) else None) for kk, v in b.items()},
            "move68_pct": pct("q16", "q84"), "move90_pct": pct("q05", "q95"),
            "drivers": drivers(model, x_row, model.names, k), "model": model.card()}
