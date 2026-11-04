"""Head to head: this repo's engine against the field's standard tools, on the same tasks.

Every system gets the same training rows, the same candidate library and the same held-out test rows
(laws.field_data, laws.motion_data), and is graded against the terms the world is made of:

    ours        engine.discover: families, matching pursuit, BIC, swaps, bootstrap stability
    STLSQ       PySINDy's sequentially thresholded least squares (the original SINDy optimizer)
    SR3         PySINDy's sparse relaxed regularized regression, L0 penalty
    SSR         PySINDy's stepwise sparse regression (backward elimination)
    FROLS       PySINDy's forward-regression orthogonal least squares (greedy forward selection)
    E-SINDy     PySINDy's ensemble: STLSQ bagged over 20 resamples; a term is kept if chosen in 90%
    LassoCV     scikit-learn's cross-validated L1 regression, the classical baseline

None of the PySINDy optimizers chooses its own sparsity. Each gets a sweep of its knob (threshold, or
its own path of models) and the model with the lowest BIC on the training rows is taken, the same
criterion this engine uses, then refitted by least squares on its terms. Columns are scaled to unit
norm for all, as here. PySR, which searches formulas instead of a library, is run separately
(compare_pysr), when it is installed.

Grades: exact (the chosen terms are the true ones, no more, no fewer), true and false terms, the
held-out (extrapolated) error, and time.
"""
from __future__ import annotations

import time

import numpy as np

from brain.discover import engine, laws


def _bic(X, y, cols):
    n = len(y)
    xi = np.zeros(X.shape[1])
    if len(cols):
        xi[cols] = np.linalg.lstsq(X[:, cols], y, rcond=None)[0]
    r = y - X @ xi
    return n * np.log(r @ r / n + 1e-300) + len(cols) * np.log(n), xi


def _best(X, y, candidates):
    """Among candidate supports (index lists), the lowest-BIC one, refitted."""
    seen, best = set(), None
    for c in candidates:
        c = tuple(sorted(int(i) for i in c))
        if c in seen or not c:
            continue
        seen.add(c)
        b, xi = _bic(X, y, list(c))
        if best is None or b < best[0]:
            best = (b, xi)
    return best[1] if best else np.zeros(X.shape[1])


def _support(coef, tol=1e-10):
    coef = np.ravel(coef)
    return list(np.flatnonzero(np.abs(coef) > tol * max(1e-300, np.abs(coef).max())))


def fit_system(name: str, X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Coefficients (on unit-norm columns) chosen by system ``name``."""
    import pysindy as ps
    from sklearn.linear_model import LassoCV
    thresholds = np.logspace(-4, 0, 30)
    if name == "STLSQ":
        return _best(X, y, [_support(ps.STLSQ(threshold=t, alpha=1e-8).fit(X, y).coef_) for t in thresholds])
    if name == "SR3":
        cands = []
        for t in thresholds:
            o = ps.SR3(reg_weight_lam=t ** 2 / 2, regularizer="L0", max_iter=200).fit(X, y)
            cands.append(_support(o.coef_, 1e-6))
        return _best(X, y, cands)
    if name == "SSR":
        o = ps.SSR(alpha=1e-8, max_iter=X.shape[1] - 1).fit(X, y)
        return _best(X, y, [_support(h) for h in o.history_])
    if name == "FROLS":
        o = ps.FROLS(alpha=1e-8, max_iter=min(X.shape[1], 40)).fit(X, y)
        return _best(X, y, [_support(h) for h in o.history_])
    if name == "E-SINDy":
        best = None
        for t in np.logspace(-4, 0, 8):                 # 8 thresholds: each one bags 20 fits
            np.random.seed(seed)
            o = ps.EnsembleOptimizer(opt=ps.STLSQ(threshold=t, alpha=1e-8), bagging=True, n_models=20).fit(X, y)
            share = (np.abs(np.array(o.coef_list)[:, 0, :]) > 0).mean(axis=0)
            cols = list(np.flatnonzero(share >= 0.9))
            if cols:
                b, xi = _bic(X, y, cols)
                if best is None or b < best[0]:
                    best = (b, xi)
        return best[1] if best else np.zeros(X.shape[1])
    if name == "LassoCV":
        o = LassoCV(cv=5, fit_intercept=False, alphas=60, max_iter=20000, random_state=seed).fit(X, y)
        return _best(X, y, [_support(o.coef_, 1e-8)])
    raise ValueError(name)


SYSTEMS = ("ours", "STLSQ", "SR3", "SSR", "FROLS", "E-SINDy", "LassoCV")


def run(task: dict, truth: set, systems=SYSTEMS, seed: int = 0) -> list[dict]:
    """Run each system on one task (a laws.*_data dict) and grade it against ``truth`` (term names)."""
    X, y, Xt, yt, names = task["X"], task["y"], task["X_test"], task["y_test"], task["names"]
    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0] = 1.0
    Xs, Xts = X / scale, Xt / scale
    out = []
    for s in systems:
        t0 = time.perf_counter()
        if s == "ours":
            law = engine.discover("y", names, X, y, Xt, yt, seed=seed, groups=task["groups"], families=task["families"])
            chosen = set(law.terms)
            test = law.test_err
            verdict = "FOUND" if law.found else "NOT FOUND"
            coefs = law.terms
        else:
            xi = fit_system(s, Xs, y, seed)
            chosen = {names[i] for i in np.flatnonzero(xi)}
            test = engine.rel_err(Xts, yt, xi) if chosen else 1.0
            verdict = ""                                       # these return a model, never "not found"
            coefs = {names[i]: float(xi[i] / scale[i]) for i in np.flatnonzero(xi)}
        out.append({"system": s, "exact": chosen == truth, "true": len(chosen & truth), "of": len(truth),
                    "false": len(chosen - truth), "terms": len(chosen), "test_err": float(test),
                    "seconds": time.perf_counter() - t0, "verdict": verdict, "coefs": coefs})
    return out


TASKS = {
    "field, no noise": (lambda s: laws.field_data(noise=0.0, seed=s), lambda: laws.FIELD_TRUTH),
    "field, 1% noise": (lambda s: laws.field_data(noise=0.01, seed=s), lambda: laws.FIELD_TRUTH),
    "field, 5% noise": (lambda s: laws.field_data(noise=0.05, seed=s), lambda: laws.FIELD_TRUTH),
    "gravity, satellites": (lambda s: laws.motion_data("satellite", seed=s), lambda: {"u/r^2"}),
    "protons, unaided": (lambda s: laws.motion_data("proton", seed=s), lambda: {"v x B", "u/r^2"}),
    "protons, gravity carried over": (lambda s: laws.motion_data("proton", seed=s, known={"u/r^2": -1.0}), lambda: {"v x B"}),
    "control: field shuffled": (lambda s: laws.field_data(noise=0.01, seed=s, shuffle=True), lambda: set()),
    "control: field without r^-3": (lambda s: laws.field_data(noise=0.01, seed=s, powers=(1, 2, 4, 5)), lambda: set()),
}


def _one(args):
    task, seed = args
    make, truth = TASKS[task]
    return task, seed, run(make(seed), truth(), seed=seed)


def benchmark(seeds=(0, 1, 2), tasks=None, workers: int = 4) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(t, s) for t in (tasks or TASKS) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_one, jobs))
    rows = []
    for task, seed, out in res:
        for r in out:
            rows.append({"task": task, "seed": seed, **{k: v for k, v in r.items() if k != "coefs"}})
    return rows


# --- PySR: symbolic regression over formulas, no library ------------------------------------------------
def compare_pysr(seconds: int = 120, seed: int = 0) -> list[dict]:
    """PySR (Cranmer 2023) searches expression trees built from + - * / and powers over raw measured
    variables, with no library. It works one scalar target at a time. Given the same measurements as
    above, can it write the gravity law, the Lorentz force, and one component of the field?"""
    from pysr import PySRRegressor

    def search(X, y, names, ops, Xt, yt):
        m = PySRRegressor(niterations=10_000, timeout_in_seconds=seconds, binary_operators=["+", "-", "*", "/"],
                          unary_operators=ops, maxsize=40, populations=24, random_state=seed, deterministic=True,
                          parallelism="serial", verbosity=0, progress=False, model_selection="best")
        t0 = time.perf_counter()
        m.fit(X, y, variable_names=names)
        pred = m.predict(Xt)
        return {"equation": str(m.sympy()), "test_err": float(np.linalg.norm(yt - pred) / np.linalg.norm(yt)),
                "seconds": time.perf_counter() - t0}

    out = []
    sat = laws.motion_data("satellite", seed=seed)
    tr = np.vstack([np.hstack([o["r"], o["a"]]) for o in sat["train"]])[::20]
    te = np.vstack([np.hstack([o["r"], o["a"]]) for o in sat["test"]])[::20]
    out.append({"task": "gravity, satellites: a_x(x, y, z)", "truth": "-x / (x^2+y^2+z^2)^(3/2)",
                **search(tr[:, :3], tr[:, 3], ["x", "y", "z"], ["sqrt"], te[:, :3], te[:, 3])})
    pro = laws.motion_data("proton", seed=seed, known={"u/r^2": -1.0})
    cols = lambda obs: np.vstack([np.hstack([o["v"], o["B"], o["a"] + o["r"] / np.linalg.norm(o["r"], axis=1, keepdims=True) ** 3])
                                  for o in obs])[::20]
    tr, te = cols(pro["train"]), cols(pro["test"])
    out.append({"task": "protons, gravity carried over: a_x(v, B)", "truth": "50 (vy Bz - vz By)",
                **search(tr[:, :6], tr[:, 6], ["vx", "vy", "vz", "Bx", "By", "Bz"], [], te[:, :6], te[:, 6])})
    f = laws.field_data(noise=0.0, seed=seed)
    out.append({"task": "field: Bz(x, y, z), no noise", "truth": "dipole + quadrupole, z-component",
                **search(f["r"], f["B"][:, 2], ["x", "y", "z"], ["sqrt"], f["r_test"], f["B_test"][:, 2])})
    return out


def table(rows: list[dict]) -> str:
    """Per task and system: exact recoveries out of seeds, mean true/false terms, median held-out error,
    mean seconds."""
    lines = []
    for task in dict.fromkeys(r["task"] for r in rows):
        rs = [r for r in rows if r["task"] == task]
        lines.append(f"\n{task}  (true terms: {rs[0]['of'] or 'none: any law is false'})")
        lines.append(f"  {'system':9s} {'exact':>6s} {'true':>6s} {'false':>6s} {'held-out err':>13s} {'seconds':>8s}  verdicts")
        for s in dict.fromkeys(r["system"] for r in rs):
            g = [r for r in rs if r["system"] == s]
            verdicts = ", ".join(r["verdict"] for r in g if r["verdict"])
            lines.append(f"  {s:9s} {sum(r['exact'] for r in g):3d}/{len(g):<2d} {np.mean([r['true'] for r in g]):6.1f}"
                         f" {np.mean([r['false'] for r in g]):6.1f} {np.median([r['test_err'] for r in g]):12.2%}"
                         f" {np.mean([r['seconds'] for r in g]):8.1f}  {verdicts}")
    return "\n".join(lines)
