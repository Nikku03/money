"""The experiments: what the engine is shown, what it may use, and what it has to find.

Each builds a library of candidate terms from measurements only (positions, magnetometer readings,
positions over time) and hands it to ``engine.discover``. None of them reads the world's constants;
they are used only afterwards, to grade what was found.

    field       magnetometer survey at 1-2.5 Earth radii; test at 2.5-6 (extrapolation).
                85 candidate vector terms: for each power r^-1 .. r^-5, a uniform field along each
                axis, first-order angular shapes, second-order shapes, and a radial (monopole) term.
    motion      positions of charged particles and of neutral satellites, sampled over time; velocity
                and acceleration by finite differences, the field from an onboard magnetometer.
                16 candidate force terms. Test on other particles, launched elsewhere.
    maxwell     field derivatives from a small magnetometer array (spacing h); which linear relations
                among the 9 derivatives hold everywhere? (implicit: each derivative regressed on the
                other eight in turn)
    invariants  along each particle's path, which quantities, or combinations, stay constant?

Controls: the same searches on shuffled data, and with the true terms removed from the library.
"""
from __future__ import annotations

import numpy as np

from brain.discover import earth
from brain.discover.engine import Law, discover

AX = "xyz"


# --- the field --------------------------------------------------------------------------------------
def field_library(r: np.ndarray, powers=(1, 2, 3, 4, 5)) -> tuple[list[str], list[np.ndarray], list[str]]:
    """Candidate field terms at each power r^-n, in five families (each family = one shape, in every
    orientation): uniform e_i; first-order u_i u; radial u; second-order symmetric (e_i u_j + e_j u_i)
    with its traceless diagonal; second-order u_i u_j u with its traceless diagonal."""
    d = np.linalg.norm(r, axis=1, keepdims=True)
    u = r / d
    E = np.eye(3)
    names, cols, fams = [], [], []

    def add(name, vec, family):
        names.append(name)
        cols.append(vec)
        fams.append(family)

    for n in powers:
        p = d ** -n
        for i in range(3):
            add(f"e{AX[i]}/r^{n}", np.broadcast_to(E[i], r.shape) * p, f"uniform e/r^{n}")
            add(f"u{AX[i]} u/r^{n}", u[:, [i]] * u * p, f"first-order (e.u) u/r^{n}")
        add(f"u/r^{n}", u * p, f"radial u/r^{n}")
        for i, j in ((0, 1), (0, 2), (1, 2)):
            add(f"(e{AX[i]} u{AX[j]} + e{AX[j]} u{AX[i]})/r^{n}", (E[i] * u[:, [j]] + E[j] * u[:, [i]]) * p,
                f"second-order sym(e u)/r^{n}")
            add(f"u{AX[i]} u{AX[j]} u/r^{n}", u[:, [i]] * u[:, [j]] * u * p, f"second-order (u.u) u/r^{n}")
        for i in (0, 1):
            add(f"(e{AX[i]} u{AX[i]} - ez uz)/r^{n}", (E[i] * u[:, [i]] - E[2] * u[:, [2]]) * p, f"second-order sym(e u)/r^{n}")
            add(f"(u{AX[i]}^2 - uz^2) u/r^{n}", (u[:, [i]] ** 2 - u[:, [2]] ** 2) * u * p, f"second-order (u.u) u/r^{n}")
    return names, cols, fams


def _stack(cols, w=None):
    X = np.stack([c.reshape(-1) for c in cols], axis=1)
    if w is not None:
        X = X * np.repeat(w, 3)[:, None]
    return X


def field_data(n: int = 600, noise: float = 0.01, seed: int = 0, powers=(1, 2, 3, 4, 5), shuffle: bool = False) -> dict:
    """The field experiment's inputs: a survey at 1-2.5 Earth radii (train) and 2.5-6 (test), the raw
    readings, and the library rows, each reading weighted by 1/|B| so errors are relative."""
    rng = np.random.default_rng(seed)
    r, B = earth.survey(n, 1.0, 2.5, rng, noise)
    rt, Bt = earth.survey(n // 2, 2.5, 6.0, rng, noise)
    if shuffle:
        B = B[rng.permutation(len(B))]
    w, wt = 1 / np.linalg.norm(B, axis=1), 1 / np.linalg.norm(Bt, axis=1)   # relative error per reading
    names, cols, fams = field_library(r, powers)
    _, tcols, _ = field_library(rt, powers)
    return {"names": names, "families": fams, "X": _stack(cols, w), "y": (B * w[:, None]).reshape(-1),
            "X_test": _stack(tcols, wt), "y_test": (Bt * wt[:, None]).reshape(-1), "groups": np.repeat(np.arange(n), 3),
            "r": r, "B": B, "r_test": rt, "B_test": Bt}


def field_law(n: int = 600, noise: float = 0.01, seed: int = 0, powers=(1, 2, 3, 4, 5), shuffle: bool = False) -> dict:
    d = field_data(n, noise, seed, powers, shuffle)
    law = discover("B", d["names"], d["X"], d["y"], d["X_test"], d["y_test"], seed=seed, groups=d["groups"],
                   families=d["families"])
    return {"law": law, **grade_field(law)}


# The terms the world is made of, written in each library's names: what a perfect engine would choose.
FIELD_TRUTH = ({f"e{a}/r^3" for a in AX} | {f"u{a} u/r^3" for a in AX}
               | {f"(e{AX[i]} u{AX[j]} + e{AX[j]} u{AX[i]})/r^4" for i, j in ((0, 1), (0, 2), (1, 2))}
               | {f"u{AX[i]} u{AX[j]} u/r^4" for i, j in ((0, 1), (0, 2), (1, 2))}
               | {f"(e{a} u{a} - ez uz)/r^4" for a in "xy"} | {f"({a}^2 - uz^2) u/r^4".replace(a, f"u{a}") for a in "xy"})


def grade_field(law: Law) -> dict:
    """Read the dipole moment and quadrupole back out of the found coefficients, to compare with the
    world's (B_dip = 3 (m.u) u/r^3 - m/r^3; B_quad = 5 (u.Qu) u/r^4 - 2 Q u/r^4)."""
    t = law.terms
    m = np.array([(t.get(f"u{a} u/r^3", 0) / 3 - t.get(f"e{a}/r^3", 0)) / 2 for a in AX])   # two estimates, averaged
    Q = np.zeros((3, 3))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        Q[i, j] = Q[j, i] = (t.get(f"u{AX[i]} u{AX[j]} u/r^4", 0) / 10 - t.get(f"(e{AX[i]} u{AX[j]} + e{AX[j]} u{AX[i]})/r^4", 0) / 2) / 2
    for i in (0, 1):
        Q[i, i] = (t.get(f"(u{AX[i]}^2 - uz^2) u/r^4", 0) / 5 - t.get(f"(e{AX[i]} u{AX[i]} - ez uz)/r^4", 0) / 2) / 2
    Q[2, 2] = -Q[0, 0] - Q[1, 1]
    tilt = np.degrees(np.arccos(abs(m[2]) / np.linalg.norm(m))) if np.linalg.norm(m) else float("nan")
    powers = sorted({int(k.split("r^")[-1]) for k in t})
    return {"m": m.tolist(), "m_true": earth.M.tolist(), "m_error": float(np.linalg.norm(m - earth.M)),
            "tilt_deg": float(tilt), "Q_error": float(np.linalg.norm(Q - earth.Q) / np.linalg.norm(earth.Q)),
            "powers": powers, "monopole": t.get("u/r^2", 0.0)}


# --- motion -----------------------------------------------------------------------------------------
def observe(qm: float, starts: list, t_end: float, sample: float, noise: float, rng) -> list[dict]:
    """Track each body: positions (with noise) every ``sample``; velocity and acceleration by fourth-order
    central differences; the field from an onboard magnetometer (same relative noise)."""
    out = []
    for r0, v0 in starts:
        _, p = earth.trajectory(r0, v0, qm, t_end, sample)
        p = p + noise * rng.normal(size=p.shape)
        h = sample
        v = (-p[4:] + 8 * p[3:-1] - 8 * p[1:-3] + p[:-4]) / (12 * h)
        a = (-p[4:] + 16 * p[3:-1] - 30 * p[2:-2] + 16 * p[1:-3] - p[:-4]) / (12 * h * h)
        r = p[2:-2]
        B = earth.field(r)
        B = B * (1 + noise * rng.normal(size=B.shape))
        out.append({"r": r, "v": v, "a": a, "B": B})
    return out


MOTION_TERMS = ("v x B", "B", "v", "u/r^2", "u/r^3", "u/r", "u", "r x v", "(v.B) B", "|v| v",
                "|v|^2 u", "ex", "ey", "ez", "v/r^3", "B x r", "(v.u) u/r^2")


def motion_library(o: dict, drop=()) -> tuple[list[str], list[np.ndarray]]:
    r, v, B = o["r"], o["v"], o["B"]
    d = np.linalg.norm(r, axis=1, keepdims=True)
    u = r / d
    terms = {"v x B": np.cross(v, B), "B": B, "v": v, "u/r^2": u / d**2, "u/r^3": u / d**3, "u/r": u / d, "u": u,
             "r x v": np.cross(r, v), "(v.B) B": (v * B).sum(1, keepdims=True) * B,
             "|v| v": np.linalg.norm(v, axis=1, keepdims=True) * v, "|v|^2 u": (v * v).sum(1, keepdims=True) * u,
             "ex": np.broadcast_to([1.0, 0, 0], r.shape), "ey": np.broadcast_to([0.0, 1, 0], r.shape),
             "ez": np.broadcast_to([0.0, 0, 1], r.shape),
             "v/r^3": v / d**3, "B x r": np.cross(B, r), "(v.u) u/r^2": (v * u).sum(1, keepdims=True) * u / d**2}
    names = [k for k in MOTION_TERMS if k not in drop]
    return names, [terms[k] for k in names], ["constant e" if k in ("ex", "ey", "ez") else k for k in names]


def _bodies(kind: str, rng, n: int, L=(1.8, 3.0)):
    starts = []
    for _ in range(n):
        Lr = rng.uniform(*L)
        phi = rng.uniform(0, 2 * np.pi)
        r0 = Lr * np.array([np.cos(phi), np.sin(phi), rng.uniform(-0.1, 0.1)])
        t = np.array([-np.sin(phi), np.cos(phi), 0.0])
        if kind == "proton":
            v0 = rng.uniform(0.15, 0.35) * (t * rng.uniform(-1, 1) + np.array([0, 0, rng.choice([-1, 1])]))
            v0 = v0 + t * np.sqrt(earth.GM / Lr)       # orbital speed, so gravity alone would not drop it
        else:
            v0 = t * np.sqrt(earth.GM / Lr) * rng.uniform(0.85, 1.15) + rng.normal(scale=0.05, size=3)
        starts.append((r0, v0))
    return starts


def motion_law(kind: str = "proton", noise: float = 1e-7, seed: int = 0, drop=(), shuffle: bool = False,
               train_L=(1.8, 3.0), test_L=(3.0, 4.0), known: dict | None = None) -> dict:
    """``kind`` "proton" (q/m = 50) or "satellite" (neutral). Train on 6 bodies launched at ``train_L``
    Earth radii, test on 3 launched at ``test_L``. ``known`` carries laws found earlier (term ->
    coefficient): their part of the acceleration is taken off first, and the engine looks for the rest."""
    d = motion_data(kind, noise, seed, drop, shuffle, train_L, test_L, known)
    law = discover("a", d["names"], d["X"], d["y"], d["X_test"], d["y_test"], seed=seed, groups=d["groups"],
                   families=d["families"])
    return {"law": law, "q/m": law.terms.get("v x B"), "GM": -law.terms.get("u/r^2", 0.0) if "u/r^2" in law.terms else None}


def motion_data(kind: str = "proton", noise: float = 1e-7, seed: int = 0, drop=(), shuffle: bool = False,
                train_L=(1.8, 3.0), test_L=(3.0, 4.0), known: dict | None = None) -> dict:
    """The motion experiment's inputs: library rows and accelerations for training and test bodies,
    plus the raw tracks (positions, velocities, field, accelerations)."""
    rng = np.random.default_rng(seed)
    qm = earth.QM if kind == "proton" else 0.0
    t_end, sample = (6.0, 0.005) if kind == "proton" else (40.0, 0.02)
    train = observe(qm, _bodies(kind, rng, 6, L=train_L), t_end, sample, noise, rng)
    test = observe(qm, _bodies(kind, rng, 3, L=test_L), t_end, sample, noise, rng)

    def build(obs):
        names, X, y, g = None, [], [], []
        for k, o in enumerate(obs):
            names, cols, fams = motion_library(o, drop)
            X.append(_stack(cols))
            a = o["a"].copy()
            if known:
                full, allcols, _ = motion_library(o)
                for term, c in known.items():
                    a -= c * allcols[full.index(term)]
            y.append(a.reshape(-1))
            g.append(np.repeat(np.arange(len(o["a"])) // 50 + 10_000 * k, 3))   # blocks of 50 samples
        return names, fams, np.vstack(X), np.concatenate(y), np.concatenate(g)

    names, fams, X, y, g = build(train)
    _, _, Xt, yt, _ = build(test)
    if shuffle:
        y = y.reshape(-1, 3)[rng.permutation(len(y) // 3)].reshape(-1)
    return {"names": names, "families": fams, "X": X, "y": y, "X_test": Xt, "y_test": yt, "groups": g,
            "train": train, "test": test}


# --- Maxwell ----------------------------------------------------------------------------------------
DERIVS = [f"dB{AX[i]}/d{AX[j]}" for i in range(3) for j in range(3)]


def maxwell(n: int = 500, h: float = 0.01, noise: float = 1e-6, seed: int = 0) -> dict:
    """Each sample point carries a 7-magnetometer array (centre and +-h on each axis); derivatives are
    central differences of the noisy readings. Each derivative is regressed on the other eight."""
    rng = np.random.default_rng(seed)

    def jac(m, rmin, rmax):
        r, _ = earth.survey(m, rmin, rmax, rng)
        J = np.zeros((m, 3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            Bp, Bm = earth.field(r + e), earth.field(r - e)
            scale = np.linalg.norm(earth.field(r), axis=1, keepdims=True)
            Bp = Bp + noise * scale * rng.normal(size=Bp.shape)
            Bm = Bm + noise * scale * rng.normal(size=Bm.shape)
            J[:, :, j] = (Bp - Bm) / (2 * h)
        w = 1 / np.linalg.norm(J.reshape(m, 9), axis=1, keepdims=True)    # relative, so near and far count alike
        return J.reshape(m, 9) * w

    D, Dt = jac(n, 1.0, 2.5), jac(n // 2, 2.5, 6.0)
    found = {}
    for k in range(9):
        others = [i for i in range(9) if i != k]
        law = discover(DERIVS[k], [DERIVS[i] for i in others], D[:, others], D[:, k], Dt[:, others], Dt[:, k], seed=seed + k)
        if not law.found:
            continue
        coef = np.zeros(9)
        coef[k] = 1.0
        for name, c in law.terms.items():
            coef[DERIVS.index(name)] = -c
        coef = coef / coef[np.argmax(np.abs(coef))]
        key = tuple(np.round(coef, 2))
        found.setdefault(key, (law, coef))
    eqs = []
    for law, coef in found.values():
        lhs = " ".join(f"{'-' if c < 0 else '+'} {'' if abs(abs(c) - 1) < 0.02 else f'{abs(c):.3g} '}{DERIVS[i]}"
                       for i, c in enumerate(coef) if abs(c) > 1e-9)
        eqs.append({"equation": (lhs[2:] if lhs.startswith("+") else lhs) + " = 0", "test_err": law.test_err,
                    "coef": coef.tolist()})
    return {"equations": eqs}


# --- invariants -------------------------------------------------------------------------------------
SCALARS = ("v_par^2", "v_perp^2", "1/r", "r", "|B|", "v_perp^2/|B|", "r^2", "v_par^2/|B|")


def scalars(o: dict) -> dict:
    r, v, B = o["r"], o["v"], o["B"]
    b = np.linalg.norm(B, axis=1)
    vpar2 = ((v * B).sum(1) / b) ** 2
    vperp2 = (v * v).sum(1) - vpar2
    d = np.linalg.norm(r, axis=1)
    return {"v_par^2": vpar2, "v_perp^2": vperp2, "1/r": 1 / d, "r": d, "|B|": b, "v_perp^2/|B|": vperp2 / b,
            "r^2": d * d, "v_par^2/|B|": vpar2 / b}


def invariants(seed: int = 0, noise: float = 1e-7, qm: float = earth.QM) -> dict:
    """Which quantities hold still along each proton's path? Alone: a quantity whose spread along each
    path is under 2% of its size. In combination: a quantity whose ups and downs (each path's mean
    removed) are explained by a sparse combination of the others, on held-out particles."""
    rng = np.random.default_rng(seed)
    sample = 0.005 * min(1.0, earth.QM / qm)                 # faster gyration needs finer sampling
    train = observe(qm, _bodies("proton", rng, 6), 6.0, sample, noise, rng)
    test = observe(qm, _bodies("proton", rng, 3, L=(3.0, 4.0)), 6.0, sample, noise, rng)
    S, St = [scalars(o) for o in train], [scalars(o) for o in test]
    alone = {}
    for k in SCALARS:
        spread = np.mean([np.std(s[k]) / np.mean(np.abs(s[k])) for s in S + St])
        alone[k] = float(spread)
    centred = lambda ss, k: np.concatenate([s[k] - s[k].mean() for s in ss])
    combos, seen = [], set()
    for k in SCALARS:
        if alone[k] < 0.02:
            continue
        others = [x for x in SCALARS if x != k and alone[x] >= 0.02]
        X = np.stack([centred(S, x) for x in others], 1)
        Xt = np.stack([centred(St, x) for x in others], 1)
        groups = np.concatenate([np.arange(len(s[k])) // 50 + 10_000 * i for i, s in enumerate(S)])
        law = discover(f"change in {k}", [f"change in {x}" for x in others], X, centred(S, k), Xt, centred(St, k),
                       seed=seed, groups=groups)
        if law.found:
            coef = {k: 1.0, **{t.replace("change in ", ""): -c for t, c in law.terms.items()}}
            key = frozenset((t, round(c / max(coef.values(), key=abs), 2)) for t, c in coef.items())
            if key not in seen:                                    # one relation, found from each of its terms
                seen.add(key)
                combos.append(law)
    return {"alone": alone, "combos": combos, "qm": qm}
