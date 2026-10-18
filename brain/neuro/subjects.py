"""The networks the engine is handed, and the controls it is checked on.

Two subjects, each with controls whose answer is known:

``modadd``: a one-hidden-layer network trained on (a + b) mod p. Input one-hot a and b, hidden
``h = (U[:, a] + V[:, b]) ** 2``, logits ``W @ h``. Trained on half the pairs with Adam and weight decay, it
generalises to the other half. The interpretability literature says how such networks do it: each hidden neuron
is a cosine of one frequency in a, the same in b, and its output weights are that frequency again with phase the
sum of the two input phases (Nanda et al. 2023, arXiv:2301.05217; Gromov 2023, arXiv:2301.02679). That gives a
known answer to grade the engine against.
- ``planted``: a network written from that construction by hand, never trained. The instrument must find it.
- ``memorizer``: the same network trained the same way with almost no weight decay. It fits the training pairs
  and fails the held-out ones. A claim about an algorithm must fail on it.

``chain``: the chain engine's own network (``brain.chain.train``: the bump-node tree with the snap). How it
follows a chain is not written down anywhere. Each step reads ``[table, pointer e, memory h]`` and writes the
next memory and pointer.
- ``planted_lookup``: a hand-written step that reads the pointer and looks the next item up in the table.
- ``planted_memory``: a hand-written step that keeps the current item in its memory and ignores the pointer.
- ``nomemory``: the chain network rebuilt without its continuous memory (the tree sees only the table and the
  pointer), trained the same way. It exists to test a prediction, not as a control.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

MODADD = {"p": 23, "hidden": 96, "train_frac": 0.5, "steps": 4000, "lr": 3e-3, "betas": [0.9, 0.98],
          "weight_decay": 1.0, "loss": "softmax cross-entropy, full batch", "init": "normal, sd 1/sqrt(fan-in)"}
MEMORIZER = {**MODADD, "weight_decay": 1e-3}
CHAIN = {"steps": 3000, "trainer": "brain.chain.train (teacher forcing, step supervision, state noise 0.2)",
         "items": 6, "format": "every"}


def sha(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


# ---------- modular addition ----------
class ModAdd:
    """``U`` (hidden, p) and ``V`` (hidden, p) read a and b, ``W`` (p, hidden) writes the logits."""

    def __init__(self, U, V, W, kind: str, spec: dict, seed=None, split=None):
        self.U, self.V, self.W = U, V, W
        self.kind, self.spec, self.seed, self.split = kind, spec, seed, split
        self.p = U.shape[1]

    @property
    def weights(self):
        return [self.U, self.V, self.W]

    def logits(self, a, b):
        return self.W @ (self.U[:, a] + self.V[:, b]) ** 2

    def accuracy(self, idx=None) -> float:
        """Share right over pair indices ``idx`` (a * p + b); every pair when None."""
        p = self.p
        idx = np.arange(p * p) if idx is None else np.asarray(idx)
        a, b = idx // p, idx % p
        return float((self.logits(a, b).argmax(0) == (a + b) % p).mean())

    def heldout(self) -> float:
        return self.accuracy(self.split["test"]) if self.split else self.accuracy()


def train_modadd(seed: int, spec: dict = MODADD, kind: str = "trained") -> ModAdd:
    p, N = spec["p"], spec["hidden"]
    rng = np.random.default_rng(seed)
    A, B = np.divmod(np.arange(p * p), p)
    C = (A + B) % p
    perm = rng.permutation(p * p)
    ntr = round(spec["train_frac"] * p * p)
    tr = perm[:ntr]
    U = rng.normal(0, 1 / math.sqrt(p), (N, p))
    V = rng.normal(0, 1 / math.sqrt(p), (N, p))
    W = rng.normal(0, 1 / math.sqrt(N), (p, N))
    ps = [U, V, W]
    m1 = [np.zeros_like(x) for x in ps]
    m2 = [np.zeros_like(x) for x in ps]
    b1, b2 = spec["betas"]
    lr, wd = spec["lr"], spec["weight_decay"]
    a, b, c = A[tr], B[tr], C[tr]
    for t in range(1, spec["steps"] + 1):
        pre = U[:, a] + V[:, b]
        h = pre ** 2
        lo = W @ h
        lo -= lo.max(0)
        P = np.exp(lo)
        P /= P.sum(0)
        P[c, np.arange(ntr)] -= 1
        d = P / ntr
        gW = d @ h.T
        dpre = 2 * pre * (W.T @ d)
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        np.add.at(gU.T, a, dpre.T)
        np.add.at(gV.T, b, dpre.T)
        for i, (x, g) in enumerate(zip(ps, (gU, gV, gW))):
            m1[i] = b1 * m1[i] + (1 - b1) * g
            m2[i] = b2 * m2[i] + (1 - b2) * g * g
            x -= lr * (m1[i] / (1 - b1 ** t) / (np.sqrt(m2[i] / (1 - b2 ** t)) + 1e-8) + wd * x)
    return ModAdd(U, V, W, kind, spec, seed, {"train": np.sort(tr), "test": np.sort(perm[ntr:])})


def cosine_net(p: int, freqs, amp, phase_u, phase_v, law: str = "sum", kind: str = "planted", spec=None) -> ModAdd:
    """A network whose neuron k is a cosine of frequency ``freqs[k]``: U and V with the given phases and
    amplitudes ``amp[k] = (au, av, aw)``, W with the phase the ``law`` gives (``sum``: phase_u + phase_v,
    ``diff``: phase_u - phase_v)."""
    f = np.asarray(freqs, float)[:, None]
    x = np.arange(p)[None, :]
    amp = np.asarray(amp, float)
    pu, pv = np.asarray(phase_u, float)[:, None], np.asarray(phase_v, float)[:, None]
    pw = pu + pv if law == "sum" else pu - pv
    th = 2 * np.pi * f * x / p
    U = amp[:, :1] * np.cos(th + pu)
    V = amp[:, 1:2] * np.cos(th + pv)
    W = (amp[:, 2:3] * np.cos(th + pw)).T
    return ModAdd(U, V, W, kind, spec or {"p": p, "hidden": len(freqs), "law": law})


def planted_modadd(seed: int, p: int = MODADD["p"], hidden: int = MODADD["hidden"]) -> ModAdd:
    """The literature's construction, by hand: neurons spread evenly over every frequency, random phases."""
    rng = np.random.default_rng(seed)
    F = (p - 1) // 2
    freqs = np.arange(hidden) % F + 1
    return cosine_net(p, freqs, np.ones((hidden, 3)), rng.uniform(0, 2 * np.pi, hidden),
                      rng.uniform(0, 2 * np.pi, hidden), "sum", "planted",
                      {"p": p, "hidden": hidden, "construction": "cosine per neuron, phase law sum, by hand"})


# ---------- the chain engine ----------
class ChainSubject:
    """One step of a chain network as ``step(x, e, h) -> (h_next, logits)``, with the pointer codes ``Emb``.
    The loop around it (``run``) is the snap: the argmax item is written down and its code fed to the next step.
    ``hooks`` replaces the pointer or the memory going into a step, which is how the probes intervene."""
    M = 6

    def __init__(self, kind: str, H: int, P: int, Emb, step, spec: dict, seed=None, weights=()):
        self.kind, self.H, self.P, self.Emb = kind, H, P, Emb
        self._step, self.spec, self.seed, self.weights = step, spec, seed, list(weights)

    def run(self, X, hook_e=None, hook_h=None):
        """Items written at every step, (n, T). ``hook_e(t, e, items_so_far)`` and ``hook_h(t, h)`` may
        return a replacement for the pointer or memory read at step t."""
        n, T, _ = X.shape
        h = np.zeros((n, self.H))
        e = np.zeros((n, self.P))
        items = []
        for t in range(T):
            e_in = hook_e(t, e, items) if hook_e else None
            h_in = hook_h(t, h) if hook_h else None
            h, lg = self._step(X[:, t], e if e_in is None else e_in, h if h_in is None else h_in)
            it = lg.argmax(1)
            items.append(it)
            e = self.Emb[it]
        return np.stack(items, 1)

    def accuracy(self, k: int, n: int = 1000, seed: int = 99) -> float:
        X, Y = questions(np.random.default_rng(seed + k), n, k)
        return float((self.run(X)[:, -1] == Y[:, -1]).mean())


def questions(rng, n, k):
    from brain import chain
    return chain._functions(rng, n, k)


def _snap_step(net, memory: bool = True):
    def step(x, e, h):
        z = np.concatenate([x, e, h if memory else np.zeros_like(h)], 1)
        pre, _ = net.f_fwd(z)
        hn = np.tanh(pre)
        return hn, hn @ net.Wo + net.bo
    return step


def train_chain(seed: int, memory: bool = True, steps: int = CHAIN["steps"]) -> ChainSubject:
    """``brain.chain.train`` itself for the engine's network. Without memory, the same loop with the previous
    state zeroed going in (its gradient is then zero too, which the engine's ``broken`` flag gives)."""
    from brain import chain
    if memory:
        net = chain.train(steps, seed=seed, log=None)
    else:
        cs = chain._engine()
        ce = cs.ce

        class NoMemory(cs.SnapNet):
            broken = True

            def run(self, X, Y=None, start=0, noise=0.0, rng=None):
                n, T, _ = X.shape
                e = np.zeros((n, cs.P))
                self.cache, self.hs, self.items = [], [], []
                for t in range(T):
                    z = np.concatenate([X[:, t], e, np.zeros((n, self.H))], 1)
                    pre, c = self.f_fwd(z)
                    a = np.tanh(pre)
                    h = a + noise * rng.normal(size=a.shape) if noise > 0 else a
                    self.cache.append((z, c, a))
                    self.hs.append(h)
                    item = Y[:, t] if Y is not None else (h @ self.Wo + self.bo).argmax(1)
                    e = self.Emb[item]
                    self.items.append(item)
                return h @ self.Wo + self.bo

        net = NoMemory("combined", np.random.default_rng(seed), ce.dx_of(chain.M, "every"), chain.M)
        rng, st = np.random.default_rng(700 + seed), {}
        for _ in range(steps):
            X, Y = chain._functions(rng, ce.BS, int(rng.choice(ce.TRAIN_K)))
            net.run(X, Y, 0, ce.NOISE, rng)
            dl, _ = ce.step_losses(net, Y, True)
            ce.adam_clip(net.ps, net.backward(dl), st, ce.LR)
    kind = "trained" if memory else "nomemory"
    return ChainSubject(kind, net.H, net.Emb.shape[1], net.Emb, _snap_step(net, memory),
                        {**CHAIN, "memory": memory, "steps": steps}, seed, net.ps)


def _table(x, M):
    return x[:, :M * M].reshape(-1, M, M).argmax(2), x[:, M * M:M * M + M]


def planted_lookup(H: int = 24, P: int = 16) -> ChainSubject:
    """Reads the pointer (a one-hot code; the start item at step 0) and looks the next item up in the table.
    Its memory is noise it never reads."""
    M = ChainSubject.M
    Emb = np.eye(M, P)

    def step(x, e, h):
        succ, start = _table(x, M)
        cur = np.where(start.sum(1) > 0, start.argmax(1), e[:, :M].argmax(1))
        nxt = succ[np.arange(len(x)), cur]
        return np.tanh(np.random.default_rng(len(x)).normal(size=(len(x), H))), 10.0 * np.eye(M)[nxt]
    return ChainSubject("planted_lookup", H, P, Emb, step, {"construction": "pointer -> table row, by hand"})


def planted_memory(H: int = 24, P: int = 16) -> ChainSubject:
    """Keeps the current item one-hot in its memory and ignores the pointer."""
    M = ChainSubject.M
    Emb = np.eye(M, P)

    def step(x, e, h):
        succ, start = _table(x, M)
        cur = np.where(start.sum(1) > 0, start.argmax(1), h[:, :M].argmax(1))
        nxt = succ[np.arange(len(x)), cur]
        hn = np.zeros((len(x), H))
        hn[np.arange(len(x)), nxt] = 1.0
        return hn, 10.0 * np.eye(M)[nxt]
    return ChainSubject("planted_memory", H, P, Emb, step, {"construction": "memory -> table row, by hand"})
