"""The discovery engine: which few kinds of term explain the measurements?

It knows no physics. It is given a target (say, a measured acceleration) and a library of candidate
terms built from what was measured (positions, velocities, field readings), most of them wrong, and
it looks for the smallest set of terms whose linear combination reproduces the target. This is
sparse regression, after SINDy (Brunton, Proctor & Kutz, PNAS 2016), with three changes that the
problems here needed.

    families   terms come in families: one shape at one power of r, in each orientation (e_x/r^3,
               e_y/r^3, e_z/r^3 are one family). The engine chooses or rejects whole families, never
               one axis of a shape. Space has no preferred axes, so a law cannot hinge on how the
               coordinates were pointed. This is a symmetry, not a hint: every family is offered on
               the same footing, wrong ones included.
    grow       forward selection (group orthogonal matching pursuit): start with nothing; each step adds
               the family that best explains what is still unexplained, and refits everything chosen.
               Columns are scaled to unit norm, so families compete on shape, not units. (Thresholding a
               full least-squares fit, the usual SINDy step, fails here: over 1-2.5 Earth radii
               r^-1 .. r^-5 look alike, and the full fit spreads weight across them with huge
               cancelling coefficients.) Growing stops ``PATIENCE`` steps after the last improvement
               on a validation split.
    choose     by the Bayesian information criterion (n log(RSS/n) + k log n): the point on the growth
               path with the lowest BIC; then families are removed while BIC improves; then swaps,
               each chosen family tried against each unchosen one, kept if BIC drops by more than
               ``SWAP_GAIN`` log n, then pruning again. Over a narrow range of r a blend of three wrong
               powers can stand in for one right one, and growth alone cannot leave it. (A rule of
               "within 5% of the best validation error" was tried first: with 1% noise it dropped a
               real quadrupole family worth 4% of the error.)
    stable     the choice is rerun on 30 bootstrap resamples; a family counts only if it is chosen in
               at least 90% of them.
    test       the chosen law is scored on a held-out test set the search never saw, usually from
               another region (extrapolation), where a wrong power law fails. A law is FOUND only if
               its families are stable and it extrapolates: test error under ``ACCEPT``, or no more than
               ``EXTRAP`` times its validation error (measurement noise sets a floor no law can beat),
               and under ``LIMIT`` regardless.

The engine can only find what its library can express. A law outside the library is NOT FOUND, never
approximated into a false discovery (the ablations in laws.py test this).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACCEPT = 0.05           # a law is found if its test error is under this...
EXTRAP = 1.25           # ...or no worse than this many times its validation error (noise sets a floor no
LIMIT = 0.3             #    law can beat), and in any case under LIMIT
BOOT = 30
STABLE = 0.9
SWAP_GAIN = 10.0        # a swap must lower BIC by this many log(n): noise alone must not steer it
PATIENCE = 4            # stop growing after this many steps without a better validation error


def rel_err(X, y, xi) -> float:
    return float(np.linalg.norm(y - X @ xi) / np.linalg.norm(y))


def _fit(X, y, cols) -> np.ndarray:
    xi = np.zeros(X.shape[1])
    if cols:
        xi[cols] = np.linalg.lstsq(X[:, cols], y, rcond=None)[0]
    return xi


def _bic(X, y, cols) -> float:
    """Bayesian information criterion of the least-squares fit on these columns: n log(RSS/n) + k log n."""
    n = len(y)
    resid = y - X @ _fit(X, y, cols)
    return float(n * np.log(resid @ resid / n + 1e-300) + len(cols) * np.log(n))


def _choose(X, y, fam: list[list[int]]) -> list[int]:
    """Grow, prune, swap, by BIC; returns the chosen family indices."""
    cols = lambda fs: [c for f in fs for c in fam[f]]
    score = lambda fs: _bic(X, y, cols(fs))
    active, path = [], []
    resid = y
    for _ in range(len(fam)):
        pull = [np.linalg.norm(X[:, fam[f]].T @ resid) / np.sqrt(len(fam[f])) if f not in active else -1.0
                for f in range(len(fam))]
        active = active + [int(np.argmax(pull))]
        resid = y - X @ _fit(X, y, cols(active))
        path.append((score(active), list(active)))
        if np.linalg.norm(resid) < 1e-12 * np.linalg.norm(y) or len(path) - 1 - int(np.argmin([p[0] for p in path])) >= PATIENCE:
            break
    best, active = min(path, key=lambda p: p[0])

    def prune(active, best):
        while len(active) > 1:
            b, f = min((score([a for a in active if a != f]), f) for f in active)
            if b > best:
                break
            active, best = [a for a in active if a != f], b
        return active, best

    active, best = prune(active, best)
    margin = SWAP_GAIN * np.log(len(y))
    for _ in range(10):                                   # swap: a blend of wrong families can stand in for
        trials = [(score([h if a == f else a for a in active]), [h if a == f else a for a in active])
                  for f in active for h in range(len(fam)) if h not in active]
        if not trials:
            break
        trial = min(trials, key=lambda t: t[0])
        if trial[0] > best - margin:                      # a right one, and greedy growth alone cannot leave it
            break
        active, best = prune(trial[1], trial[0])
    return active


@dataclass
class Law:
    target: str
    terms: dict                 # term name -> coefficient, in the library's own units
    families: list              # names of the chosen families
    val_err: float
    test_err: float
    stability: dict             # family name -> share of bootstrap runs that chose it
    found: bool
    note: str = ""

    def equation(self, digits: int = 4, min_share: float = 0.0) -> str:
        if not self.terms:
            return f"{self.target} = (nothing)"
        big = max(abs(c) for c in self.terms.values())
        parts = []
        for k, c in sorted(self.terms.items(), key=lambda kv: -abs(kv[1])):
            if abs(c) < min_share * big:
                continue
            parts.append(f"{'-' if c < 0 else '+'} {abs(c):.{digits}g} {k}")
        s = " ".join(parts)
        return f"{self.target} = {s[2:] if s.startswith('+') else s}"


def discover(target: str, names: list[str], X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
             seed: int = 0, groups: np.ndarray | None = None, families: list[str] | None = None,
             val_groups: np.ndarray | None = None) -> Law:
    """Search the library (columns of X, named ``names``; ``families`` gives each column's family,
    default one family per column) for a sparse law y = X xi.

    Rows of X may be stacked vector components; ``groups`` (one label per row) keeps the rows of one
    sample together when splitting and resampling. ``val_groups`` names the groups to validate on
    (default: a random quarter); choosing them at the edge of the data validates by extrapolation."""
    rng = np.random.default_rng(seed)
    families = list(names) if families is None else families
    fam_names = list(dict.fromkeys(families))
    fam = [[i for i, f in enumerate(families) if f == n] for n in fam_names]
    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0] = 1.0
    Xs, Xts = X / scale, X_test / scale
    g = np.arange(len(y)) if groups is None else groups
    ids = np.unique(g)
    rng.shuffle(ids)
    val_ids = ids[: len(ids) // 4] if val_groups is None else np.asarray(val_groups)
    v = np.isin(g, val_ids)
    chosen = _choose(Xs[~v], y[~v], fam)
    rows_of = {k: np.flatnonzero(g == k) for k in ids} if groups is not None else None
    count = np.zeros(len(fam))
    runs = 0
    for _ in range(BOOT):
        pick = rng.choice(ids, size=len(ids))
        rows = np.concatenate([rows_of[p] for p in pick]) if rows_of is not None else pick
        vb = np.isin(g[rows], val_ids)
        if vb.all() or (~vb).all():
            continue
        runs += 1
        for f in _choose(Xs[rows][~vb], y[rows][~vb], fam):
            count[f] += 1
    share = count / max(1, runs)
    keep = [f for f in chosen if share[f] >= STABLE]
    cols = [c for f in keep for c in fam[f]]
    final = _fit(Xs, y, cols)
    val = rel_err(Xs[v], y[v], final) if cols else 1.0
    test = rel_err(Xts, y_test, final) if cols else 1.0
    unstable = [fam_names[f] for f in chosen if share[f] < STABLE]
    note = f"left out as unstable: {', '.join(unstable)}" if unstable else ""
    return Law(target=target, terms={names[i]: float(final[i] / scale[i]) for i in cols},
               families=[fam_names[f] for f in keep], val_err=val, test_err=test,
               stability={fam_names[f]: float(share[f]) for f in range(len(fam)) if share[f] > 0},
               found=bool(cols and test < LIMIT and (test < ACCEPT or test <= EXTRAP * val)), note=note)
