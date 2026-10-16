"""Gradient boosting for log loss, in numpy alone (docs/FORECAST.md, section 4).

How a model is grown:

- Each feature is cut into at most ``bins`` bins at quantiles of the training rows.
- Each tree sees its own seeded draw of rows (``subsample``) and of columns (``colsample``).
- A tree grows level by level to ``depth``. At a node, the sums of gradient, hessian and rows per bin come
  from ``np.bincount``. Only the child with fewer rows is counted; its sibling is the parent minus it.
- A split must leave ``min_leaf`` rows on each side and lower the loss (second order, no penalty).
- A leaf adds ``lr`` times its Newton step, ``-sum(g) / sum(h)``, to the log-odds.

A tree is kept as plain lists, its thresholds as raw feature values (go left when ``x < threshold``), so a
model read back from JSON predicts exactly as the one in memory. A missing value (NaN) counts as 0, as the
feature transforms make it.
"""
from __future__ import annotations

import numpy as np

MIN_HESSIAN = 1e-6   # a side of a split needs at least this much curvature (a guard; 200 rows have far more)


def sigmoid(z):
    """1 / (1 + e^-z), without overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def _clean(X) -> np.ndarray:
    return np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def bin_edges(X: np.ndarray, bins: int) -> list[np.ndarray]:
    """Per column, the distinct inner quantiles of ``X`` at 1/bins, 2/bins, ...: at most ``bins - 1`` edges."""
    qs = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    Q = np.quantile(X, qs, axis=0) if len(X) else np.zeros((len(qs), X.shape[1]))
    return [np.unique(Q[:, f]) for f in range(X.shape[1])]


def binned(X: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """(F, n) bin numbers: the count of a column's edges at or below each value (0 .. len(edges))."""
    out = np.empty((X.shape[1], X.shape[0]), dtype=np.uint8)
    for f, e in enumerate(edges):
        out[f] = np.searchsorted(e, X[:, f], side="right")
    return out


class _Hist:
    """Per-bin sums for one node over the columns a tree uses: gradient, hessian, rows; each (k, B)."""
    __slots__ = ("C", "G", "H")

    def __init__(self, G, H, C):
        self.G, self.H, self.C = G, H, C

    def minus(self, other: _Hist) -> _Hist:
        return _Hist(self.G - other.G, self.H - other.H, self.C - other.C)


def _hist(XI: np.ndarray, cols: np.ndarray, idx: np.ndarray, g: np.ndarray, h: np.ndarray, B: int) -> _Hist:
    """Histograms of the rows ``idx`` for the columns ``cols`` of ``XI`` ((F, n) bins): one bincount per
    column and sum (faster in numpy than one bincount over every column at once)."""
    k = len(cols)
    G, H = np.zeros((k, B)), np.zeros((k, B))
    C = np.zeros((k, B), dtype=np.int64)
    gi, hi = g[idx], h[idx]
    for j, f in enumerate(cols):
        col = XI[f][idx]
        G[j] = np.bincount(col, weights=gi, minlength=B)
        H[j] = np.bincount(col, weights=hi, minlength=B)
        C[j] = np.bincount(col, minlength=B)
    return _Hist(G, H, C)


def _best_split(hs: _Hist, Gn: float, Hn: float, n: int, min_leaf: int, n_edges: np.ndarray):
    """(gain, column position, bin) of the best split "bin <= b goes left", or None."""
    GL = np.cumsum(hs.G, axis=1)[:, :-1]
    HL = np.cumsum(hs.H, axis=1)[:, :-1]
    CL = np.cumsum(hs.C, axis=1)[:, :-1]
    GR, HR, CR = Gn - GL, Hn - HL, n - CL
    b = np.arange(GL.shape[1])[None, :]
    ok = (CL >= min_leaf) & (CR >= min_leaf) & (HL > MIN_HESSIAN) & (HR > MIN_HESSIAN) & (b < n_edges[:, None])
    if not ok.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(ok, GL * GL / HL + GR * GR / HR - Gn * Gn / Hn, -np.inf)
    j = int(np.argmax(gain))  # the first of equal gains: deterministic
    best = float(gain.flat[j])
    if not best > 1e-12:
        return None
    return best, j // gain.shape[1], j % gain.shape[1]


def _grow(XI, cols, edges, g, h, rows, B, depth, min_leaf, lr, raw) -> dict:
    """One tree on the rows ``rows`` and the columns ``cols`` of the bins ``XI`` ((F, n)). Each of ``rows``
    gets its leaf's value added to ``raw`` here; the caller adds it for the other rows."""
    feat, thr, tbin, left, right, value = [], [], [], [], [], []
    n_edges = np.array([len(edges[c]) for c in cols])

    def node() -> int:
        for lst, v in ((feat, -1), (thr, 0.0), (tbin, 0), (left, -1), (right, -1), (value, 0.0)):
            lst.append(v)
        return len(feat) - 1

    level = [(node(), rows, _hist(XI, cols, rows, g, h, B) if depth > 0 else None)]
    for d in range(depth + 1):
        nxt = []
        for nid, idx, hs in level:
            Gn, Hn = float(g[idx].sum()), float(h[idx].sum())
            split = None if d == depth or len(idx) < 2 * min_leaf else _best_split(hs, Gn, Hn, len(idx), min_leaf, n_edges)
            if split is None:
                value[nid] = float(-lr * Gn / max(Hn, MIN_HESSIAN))
                raw[idx] += value[nid]
                continue
            _, c, b = split
            go = XI[cols[c]][idx] <= b
            li, ri = idx[go], idx[~go]
            small_left = len(li) <= len(ri)
            if d + 1 < depth:  # the children split again: count the smaller, the other is parent minus it
                hs_small = _hist(XI, cols, li if small_left else ri, g, h, B)
                hs_big = hs.minus(hs_small)
            else:              # the children are leaves: no histogram needed
                hs_small = hs_big = None
            L, R = node(), node()
            feat[nid], tbin[nid] = int(cols[c]), int(b)
            thr[nid] = float(edges[cols[c]][b])  # bin <= b  <=>  x < edges[b]
            left[nid], right[nid] = L, R
            nxt += [(L, li, hs_small if small_left else hs_big), (R, ri, hs_big if small_left else hs_small)]
        level = nxt
        if not level:
            break
    return {"feature": feat, "threshold": thr, "bin": tbin, "left": left, "right": right, "value": value}


def _apply(tree: dict, depth: int, rows_x, inner_go) -> np.ndarray:
    """Leaf values for every row: walk ``depth`` levels; ``inner_go(f, node)`` says which rows go left."""
    feat = np.asarray(tree["feature"], dtype=np.intp)
    left = np.asarray(tree["left"], dtype=np.intp)
    right = np.asarray(tree["right"], dtype=np.intp)
    value = np.asarray(tree["value"], dtype=float)
    node = np.zeros(rows_x, dtype=np.intp)
    for _ in range(depth):
        f = feat[node]
        inner = f >= 0
        if not inner.any():
            break
        go = inner_go(np.where(inner, f, 0), node)
        node = np.where(inner, np.where(go, left[node], right[node]), node)
    return value[node]


def fit(X, y, *, trees: int, depth: int, lr: float, bins: int, min_leaf: int, subsample: float,
        colsample: float, seed: int) -> dict:
    """A boosted model of P(y = 1) as a plain dict (JSON as it stands). Same inputs, same model."""
    X = _clean(X)
    y = np.asarray(y, dtype=float)
    n, F = X.shape
    if n == 0:
        raise ValueError("no training rows")
    rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    base = float(np.log(rate / (1 - rate)))
    edges = bin_edges(X, bins)
    XI = binned(X, edges).astype(np.intp)  # bincount reads intp without a copy
    B = max(2, max((len(e) for e in edges), default=0) + 1)
    rng = np.random.default_rng(seed)
    n_rows = max(1, round(subsample * n))
    n_cols = max(1, round(colsample * F))
    raw = np.full(n, base)
    out = []
    every = np.arange(n)
    for _ in range(trees):
        rows = np.sort(rng.choice(n, size=n_rows, replace=False)) if n_rows < n else every
        cols = np.sort(rng.choice(F, size=n_cols, replace=False)) if n_cols < F else np.arange(F)
        p = sigmoid(raw)
        g, h = p - y, p * (1 - p)
        tree = _grow(XI, cols, edges, g, h, rows, B, depth, min_leaf, lr, raw)
        if len(rows) < n:  # rows left out of this tree's draw still take its step
            rest = np.setdiff1d(every, rows, assume_unique=True)
            tb = np.asarray(tree["bin"])
            raw[rest] += _apply(tree, depth, len(rest), lambda f, node, rest=rest, tb=tb: XI[f, rest] <= tb[node])
        del tree["bin"]  # training only: a stored tree splits on raw values
        out.append(tree)
    return {"base": base, "depth": int(depth), "n_features": int(F), "trees": out}


def raw_score(model: dict, X) -> np.ndarray:
    """Log-odds of P(y = 1) for each row of ``X``."""
    X = _clean(X)
    n = len(X)
    rows = np.arange(n)
    out = np.full(n, float(model["base"]))
    for tree in model["trees"]:
        thr = np.asarray(tree["threshold"], dtype=float)
        out += _apply(tree, int(model["depth"]), n, lambda f, node, thr=thr: X[rows, f] < thr[node])
    return out


def predict(model: dict, X) -> np.ndarray:
    return sigmoid(raw_score(model, X))
