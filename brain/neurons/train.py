"""Training networks of each kind of cell on a task nobody wired: telling patterns apart by timing alone.

He et al. (2024, Nat Comput Sci 4:584, doi:10.1038/s43588-024-00674-9) train networks of rich and of
simple cells and compare how many cells each needs. Our circuits were wired by hand for one game each.
This adds training, with the same trainer for every kind of cell, and keeps what our circuits are good
at: conductance synapses and a delay on every connection, both of which the trainer sets.

**The task.** Six input fibres. Each of four classes is a fixed pattern in which every fibre fires once,
at its own time between 5 and 50 ms. A trial is one class's pattern with every spike moved by about
1.5 ms at random, plus, on each fibre with chance 0.2, one extra spike at a random time. Every class
has the same number of spikes on every fibre, so counting spikes says nothing (chance is 25%): only
timing tells the classes apart.

**The network.** The fibres feed n hidden cells of one kind (``circuit.CELLS``). Every fibre-to-cell
synapse has a weight (its sign makes it excitatory or inhibitory) and a delay of 0.25-20 ms. The
answer is read from how many times each hidden cell fired: a linear readout fitted by ridge
regression, so the hidden cells must turn timing into counts.

**The training.** The cross-entropy method on the weights and delays: 32 networks per generation, the
best 6 kept, 30 generations, scored on 64 validation trials with the readout fitted on 64 training
trials. Then the best network is tested once on 256 fresh trials, its readout refitted on all 128.
The untrained start (random weights and delays, readout fitted) is tested on the same trials.

    python -m brain neurons train [--sizes 1,2,4,8,16] [--seeds 3] [--json FILE]
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from brain.neurons import circuit

DT = 0.25
FIBRES, CLASSES = 6, 4
WINDOW_MS, TRIAL_MS = 50.0, 80.0          # pattern spikes fall in [5, 50] ms; the trial lasts 80 ms
MAX_DELAY = 20.0
JITTER_MS, EXTRA_P = 1.5, 0.2
W_MAX = 0.6
POP, ELITE, GENS = 32, 6, 30
N_TRAIN, N_VAL, N_TEST = 64, 64, 256
SIZES = (1, 2, 4, 8, 16)
TASK_SEED = 7


def templates(seed: int = TASK_SEED) -> np.ndarray:
    """(classes, fibres) spike times in ms: each class's pattern."""
    return np.random.default_rng(seed).uniform(5.0, WINDOW_MS, (CLASSES, FIBRES))


def trials(n: int, seed: int, tpl: np.ndarray | None = None) -> tuple[list, np.ndarray]:
    """``n`` trials, classes balanced: each a list of (fibre, time ms), and the class labels."""
    tpl = templates() if tpl is None else tpl
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % CLASSES
    rng.shuffle(labels)
    out = []
    for c in labels:
        sp = [(f, float(np.clip(tpl[c, f] + rng.normal(0, JITTER_MS), 0.0, TRIAL_MS - 1)))
              for f in range(FIBRES)]
        sp += [(f, float(rng.uniform(0, WINDOW_MS))) for f in range(FIBRES) if rng.random() < EXTRA_P]
        out.append(sp)
    return out, labels


def counts(cells: str, W: np.ndarray, D: np.ndarray, tr: list) -> np.ndarray:
    """Hidden spike counts for every network and trial. ``W`` and ``D`` are (networks, fibres, n):
    weights (signed) and delays (ms). All networks and trials run at once as one big circuit of
    disconnected copies, on ``circuit``'s own cell code. Returns (networks, trials, n)."""
    P, _, n = W.shape
    T = len(tr)
    N = P * T * n
    c = circuit.Circuit()
    for i in range(N):
        c.add("H", "RS")                       # kind matters only for Izhikevich: a regular spiking cell
    sim = c.compile(0, cells)
    steps = int((TRIAL_MS + MAX_DELAY) / DT)
    GE = np.zeros((steps, N), np.float32)
    GI = np.zeros((steps, N), np.float32)
    ids = np.arange(N).reshape(P, T, n)
    for t, spikes in enumerate(tr):
        for f, ms in spikes:
            k = np.rint((ms + D[:, f, :]) / DT).astype(int)          # (P, n) arrival step
            w = W[:, f, :]
            tgt = ids[:, t, :]
            exc = w > 0
            np.add.at(GE, (k[exc], tgt[exc]), w[exc])
            np.add.at(GI, (k[~exc], tgt[~exc]), -w[~exc])
    out = np.zeros(N)
    for k in range(steps):
        fired = sim.step(None, GE[k], GI[k])
        out[fired] += 1
    return out.reshape(P, T, n)


def readout(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Ridge regression to one-hot classes, for each network at once. ``X`` is (networks, trials, n).
    Returns predicted classes (networks, test trials)."""
    P, T, n = Xtr.shape
    A = np.concatenate([Xtr, np.ones((P, T, 1))], 2)
    B = np.concatenate([Xte, np.ones((P, Xte.shape[1], 1))], 2)
    Y = np.eye(CLASSES)[ytr]
    G = A.transpose(0, 2, 1) @ A + lam * T * np.eye(n + 1)
    Wo = np.linalg.solve(G, A.transpose(0, 2, 1) @ Y[None])
    return (B @ Wo).argmax(2)


FIXED_DELAY = 1.0


def decode(U: np.ndarray, n: int, delays: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Unit-cube candidates to weights and delays: (networks, 2 * fibres * n) when delays are trained,
    (networks, fibres * n) when every delay is fixed at ``FIXED_DELAY``."""
    U = np.clip(U, 0, 1).reshape(len(U), 2 if delays else 1, FIBRES, n)
    W = (2 * U[:, 0] - 1) * W_MAX
    D = 0.25 + U[:, 1] * (MAX_DELAY - 0.25) if delays else np.full(W.shape, FIXED_DELAY)
    return W, D


def train(cells: str, n: int, seed: int = 0, pop: int = POP, elite: int = ELITE, gens: int = GENS,
          delays: bool = True) -> dict:
    """Train one network size on one cell. With ``delays`` False every delay is fixed at 1 ms and only
    the weights are trained: the timing must then come from inside the cells."""
    rng = np.random.default_rng(seed)
    tpl = templates()
    tr_s, tr_y = trials(N_TRAIN, 100 + seed, tpl)
    va_s, va_y = trials(N_VAL, 200 + seed, tpl)
    te_s, te_y = trials(N_TEST, 300 + seed, tpl)
    both = tr_s + va_s
    dim = (2 if delays else 1) * FIBRES * n
    mu, sd = np.full(dim, 0.5), np.full(dim, 0.3)
    best_u, best_f, curve = None, -1.0, []
    t0 = time.process_time()
    start = None
    for gen in range(gens):
        U = np.clip(mu + sd * rng.standard_normal((pop, dim)), 0, 1)
        if gen == 0:
            start = U[: min(8, pop)].copy()           # untrained networks, for the comparison
        if best_u is not None:
            U[0] = best_u
        W, D = decode(U, n, delays)
        C = counts(cells, W, D, both)
        pred = readout(C[:, :N_TRAIN], tr_y, C[:, N_TRAIN:])
        acc = (pred == va_y[None]).mean(1)
        order = np.argsort(-acc)
        if acc[order[0]] > best_f:
            best_f, best_u = float(acc[order[0]]), U[order[0]].copy()
        E = U[order[:elite]]
        mu = 0.7 * E.mean(0) + 0.3 * mu
        sd = np.maximum(0.7 * E.std(0) + 0.3 * sd, 0.03)
        curve.append(best_f)
    train_cpu = time.process_time() - t0

    def test(Us: np.ndarray) -> np.ndarray:
        W, D = decode(Us, n, delays)
        C = counts(cells, W, D, both + te_s)
        y_fit = np.concatenate([tr_y, va_y])
        pred = readout(C[:, : len(both)], y_fit, C[:, len(both):])
        return (pred == te_y[None]).mean(1)

    t1 = time.process_time()
    acc_best = float(test(best_u[None])[0])
    test_cpu = time.process_time() - t1
    acc_start = test(start)
    return {"cells": cells, "n": n, "seed": seed, "delays": delays, "test": acc_best, "val": best_f, "curve": curve,
            "untrained_mean": float(acc_start.mean()), "untrained_best": float(acc_start.max()),
            "train_cpu_s": train_cpu, "cpu_per_net_trial_ms": 1000 * test_cpu / (len(both) + len(te_s))}


def _one(job):
    return train(*job)


def run(sizes=SIZES, seeds: int = 3, cells=circuit.CELLS, workers: int = 4, delays: bool = True) -> list[dict]:
    jobs = [(c, n, s, POP, ELITE, GENS, delays) for c in cells for n in sizes for s in range(seeds)]
    jobs.sort(key=lambda j: -(j[1] * (6 if j[0] == "hh" else 1)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, jobs))


def table(rows: list[dict]) -> list[str]:
    sizes = sorted({r["n"] for r in rows})
    trained = "weights and delays trained" if rows and rows[0].get("delays", True) else "weights trained, every delay 1 ms"
    out = [f"test accuracy on 256 fresh trials (chance 25%), mean over seeds; untrained in brackets; {trained}",
           f"{'cells':16s} " + " ".join(f"{'n=' + str(n):>14s}" for n in sizes) + f" {'cpu ms/net/trial':>17s}"]
    name = {"lif": "LIF (1 var)", "izh": "Izhikevich (2)", "hh": "HH (4 var)"}
    for c in circuit.CELLS:
        rs = [r for r in rows if r["cells"] == c]
        if not rs:
            continue
        cols = []
        for n in sizes:
            x = [r for r in rs if r["n"] == n]
            cols.append(f"{np.mean([r['test'] for r in x]):5.0%} [{np.mean([r['untrained_mean'] for r in x]):4.0%}]" if x else "-")
        cpu = np.mean([r["cpu_per_net_trial_ms"] / r["n"] for r in rs])
        out.append(f"{name[c]:16s} " + " ".join(f"{x:>14s}" for x in cols) + f" {cpu:12.3f} /cell")
    return out
