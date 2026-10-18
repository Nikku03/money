"""What the engine measures inside a network: looking (weights, Fourier spectra) and intervening (projecting,
ablating, patching), each beside a null that keeps everything but the structure in question.

Every function is deterministic given its ``seed``. Nulls draw ``n_null`` times; a permutation p is
``(1 + draws at least as extreme) / (1 + n_null)``.
"""
from __future__ import annotations

import numpy as np

from brain.neuro.subjects import ChainSubject, ModAdd, questions


# ---------- modular addition: looking ----------
def spectrum(rows: np.ndarray) -> dict:
    """Per row (one neuron's weights over the p inputs): the frequency with the most non-constant power, its
    share of that power, its phase and its amplitude (the cosine's height)."""
    p = rows.shape[1]
    F = np.fft.rfft(rows, axis=1)[:, 1:(p - 1) // 2 + 1]
    pw = np.abs(F) ** 2
    top = pw.argmax(1)
    k = np.arange(len(rows))
    return {"freq": top + 1, "share": pw[k, top] / np.maximum(pw.sum(1), 1e-30),
            "phase": np.angle(F[k, top]), "amp": 2 * np.abs(F[k, top]) / p}


def neurons(net: ModAdd) -> dict:
    return {"u": spectrum(net.U), "v": spectrum(net.V), "w": spectrum(net.W.T)}


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def law_strength(sp: dict, law: str, mask=None) -> float:
    """Mean resultant length of the output phase minus what ``law`` predicts from the input phases (1 = the law
    holds exactly for every neuron, about 1/sqrt(n) when the phases are unrelated), over neurons in ``mask``."""
    pu, pv, pw = sp["u"]["phase"], sp["v"]["phase"], sp["w"]["phase"]
    pred = pu + pv if law == "sum" else pu - pv
    r = np.exp(1j * (pw - pred))
    if mask is not None:
        r = r[mask]
    return float(np.abs(r.mean())) if len(r) else 0.0


def sparse_stat(net: ModAdd) -> float:
    """The smaller of the median top-frequency shares of U and V."""
    sp = neurons(net)
    return float(min(np.median(sp["u"]["share"]), np.median(sp["v"]["share"])))


def same_freq(sp: dict) -> np.ndarray:
    return (sp["u"]["freq"] == sp["v"]["freq"]) & (sp["u"]["freq"] == sp["w"]["freq"])


def null_sparse(net: ModAdd, n_null: int, seed: int) -> list[float]:
    """Each neuron's weights shuffled over the inputs, independently: magnitudes kept, order destroyed."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_null):
        U = np.take_along_axis(net.U, np.argsort(rng.random(net.U.shape), 1), 1)
        V = np.take_along_axis(net.V, np.argsort(rng.random(net.V.shape), 1), 1)
        out.append(sparse_stat(ModAdd(U, V, net.W, "null", {})))
    return out


def null_same_freq(sp: dict, n_null: int, seed: int) -> list[float]:
    """Output weights paired with another neuron's input weights."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_null):
        j = rng.permutation(len(sp["w"]["freq"]))
        out.append(float(((sp["u"]["freq"] == sp["v"]["freq"]) & (sp["u"]["freq"] == sp["w"]["freq"][j])).mean()))
    return out


def null_law(sp: dict, mask, n_null: int, seed: int) -> list[float]:
    """Output phases shuffled among the neurons that share a frequency."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask)
    out = []
    for _ in range(n_null):
        pw = sp["w"]["phase"].copy()
        pw[idx] = pw[rng.permutation(idx)]
        out.append(law_strength({**sp, "w": {**sp["w"], "phase": pw}}, "sum", mask))
    return out


def perm_p(obs: float, null: list[float]) -> float:
    return (1 + sum(x >= obs for x in null)) / (1 + len(null))


# ---------- modular addition: intervening ----------
def project(net: ModAdd, freqs) -> ModAdd:
    """Every neuron's U, V and W reduced to their one Fourier component at ``freqs[k]`` (constant and every
    other frequency removed)."""
    p = net.p
    x = np.arange(p)

    def keep(rows, f):
        F = np.fft.rfft(rows, axis=1)
        c = F[np.arange(len(rows)), f]
        return (2 / p) * np.real(c[:, None] * np.exp(2j * np.pi * np.outer(f, x) / p))
    return ModAdd(keep(net.U, freqs), keep(net.V, freqs), keep(net.W.T, freqs).T, "projected", net.spec, net.seed,
                  net.split)


def null_project(net: ModAdd, freqs, n_null: int, seed: int) -> list[float]:
    """Each neuron reduced to a frequency other than its own, drawn at random."""
    F = (net.p - 1) // 2
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_null):
        other = (np.asarray(freqs) - 1 + rng.integers(1, F, len(freqs))) % F + 1
        out.append(project(net, other).accuracy())
    return out


# ---------- the chain engine ----------
def reference_states(s: ChainSubject, seed: int = 4, n: int = 2000, k: int = 16) -> dict:
    """Mean memory and mean pointer code over fresh questions: what an ablation puts in their place."""
    X, _ = questions(np.random.default_rng(seed), n, k)
    hs = []
    s.run(X, hook_h=lambda t, h: hs.append(h.mean(0)) or None)
    return {"h": np.mean(hs[1:], 0) if len(hs) > 1 else hs[0], "e": s.Emb.mean(0)}


def ablate(s: ChainSubject, what: str, k: int = 16, n: int = 2000, seed: int = 5, ref=None) -> float:
    """Accuracy after k steps with the memory (``h``) or the pointer (``e``) going into every step replaced by
    its mean. The start item is in the input at step 0, so ablating the pointer still lets one step through."""
    ref = ref or reference_states(s)
    X, Y = questions(np.random.default_rng(seed), n, k)
    if what == "h":
        got = s.run(X, hook_h=lambda t, h: np.broadcast_to(ref["h"], h.shape))
    elif what == "e":
        got = s.run(X, hook_e=lambda t, e, items: np.broadcast_to(ref["e"], e.shape))
    else:
        got = s.run(X)
    return float((got[:, -1] == Y[:, -1]).mean())


def patch(s: ChainSubject, at: int = 3, k: int = 6, n: int = 2000, seed: int = 6) -> dict:
    """At step ``at``, the pointer is replaced by the code of a different item j. Does the network then write
    table[j] (``follows``), and carry on along the chain from there to step k (``carries``)? ``stays``: it
    wrote what it would have written anyway."""
    rng = np.random.default_rng(seed)
    X, Y = questions(rng, n, k)
    M = s.M
    succ = X[:, 0, :M * M].reshape(n, M, M).argmax(2)
    j = {}

    def hook(t, e, items):
        if t != at:
            return None
        j["j"] = (items[-1] + rng.integers(1, M, n)) % M
        return s.Emb[j["j"]]
    got = s.run(X, hook_e=hook)
    rows = np.arange(n)
    cur = succ[rows, j["j"]]
    follows = got[:, at] == cur
    carries = follows.copy()
    for t in range(at + 1, k):
        cur = succ[rows, cur]
        carries &= got[:, t] == cur
    return {"follows": float(follows.mean()), "carries": float(carries.mean()),
            "stays": float((got[:, at] == Y[:, at]).mean()), "n": n}


def unit_effects(s: ChainSubject, k: int = 16, n: int = 1000, seed: int = 7) -> list[float]:
    """Accuracy after k steps with one unit of the step's output held at its mean, unit by unit. Descriptive:
    where in the network the answer is read from."""
    ref = reference_states(s, n=n)

    def run_without(u):
        step = s._step

        def hooked(x, e, h):
            hn, _ = step(x, e, h)
            hn = hn.copy()
            hn[:, u] = ref["h"][u]
            return hn, _readout(s, hn)
        return ChainSubject(s.kind, s.H, s.P, s.Emb, hooked, s.spec, s.seed, s.weights)
    X, Y = questions(np.random.default_rng(seed), n, k)
    return [float((run_without(u).run(X)[:, -1] == Y[:, -1]).mean()) for u in range(s.H)]


def _readout(s: ChainSubject, h):
    Wo, bo = s.weights[-3], s.weights[-2]   # SnapNet.ps ends [Wo, bo, Emb]
    return h @ Wo + bo
