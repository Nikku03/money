"""Make the combined network EXACT on long chains, then make the chains harder.

FROM chainreason.py. The bump-tree memory carried the hop it learnt (on 1-3 hops) to 8 hops at 0.68
-- far past both matched controls -- but it FADES (0.91 at 4, 0.68 at 8): the state after a hop is
only approximately the next pointer, and the error compounds.

TWO FIXES, chosen for that diagnosis before any run:
  S  STEP SUPERVISION: during training the readout must name the intermediate item after EVERY hop,
     not only the last, so each step is trained to be one exact hop.
  N  STATE NOISE: Gaussian noise (sd 0.2) added to the state during training only, so the next step
     must clean up a slightly wrong state -- pushing the states toward attractors.
     (memmatch.py: noisy memory is what produced commitment.)
Speed: the leaf contraction is a matmul (netprobe.py: same arithmetic, 6-8x faster than einsum).

=================================================================================================
PART 1 -- EXACTNESS. 6 items, facts every step, train on 1-3 hops, fixed 3,000 steps, seeds 10-19.
=================================================================================================
   C0 combined (as chainreason)      C1 combined + S      C2 combined + S + N
   T2 tanh memory + S + N            B2 bump memory + S + N         (matched parameters)
   Test depths 1, 2, 3, 4, 6, 8, 12, 16, 24, 32.  LONG SCORE = mean accuracy over depths 4-32.
X1 EXACT if accuracy at 16 hops >= 0.99 on >= 9/10 seeds. Reported for every arm; C2 is primary.
X2 DO THE FIXES HELP?  C2 vs C0 on long score, >= 9/10 seeds.  X3 parts: C1 vs C0, C2 vs C1.
X4 DOES THE ARCHITECTURE STILL MATTER WITH THE FIXES?  C2 vs T2 and C2 vs B2, >= 9/10.

=================================================================================================
PART 2 -- HARDER VERSIONS, every model with S + N, seeds 20-29, fixed 3,000 steps.
=================================================================================================
   H1 TWELVE ITEMS, facts every step. Train 1-3 hops, test 1, 2, 3, 4, 6, 8, 12, 16.
   H2 FACTS SHOWN ONCE: the 6 facts arrive one per step in random order (source, target), then the
      start item, then blank steps; one hop per step; answer after k hops. The state must hold the
      whole table AND the pointer. Train 1-3, test 1, 2, 3, 4, 6, 8.
      Disclosed: at matched parameters the tree's state is 24 units, the RNNs' ~120 -- the RNNs
      have five times the memory, so H2 is stacked against the combined network.
Per version: LEARNT if in-distribution accuracy (1-3 hops) >= 0.90; a model with < 9/10 learnt
seeds gets no verdict. Combined vs each control on long score, >= 9/10 rule. EXACT reported as X1
(at 16 hops for H1, at 8 hops for H2).

GATES. Q0 BPTT gradient checks (per-step loss, both input formats) with a broken copy (gradient to
the previous state dropped) that MUST fail. Q1 harness: an exact solver scores 1.000 at every depth
of every format through the same evaluation code; a constant guesser scores within 0.03 of 1/items.
WHAT THIS IS NOT. Synthetic chains; one noise level, one budget, one tree shape, not swept.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import math
import multiprocessing as mp
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_chainexact.txt")
ART = R("outputs", "chainexact.json")
RULE = "=" * 97

U, B, LV = 24, 4, 2
LR, BS, CLIP, STEPS, NOISE = 0.003, 128, 1.0, 3000, 0.2
TRAIN_K = (1, 2, 3)
NEVAL = 1000
ARMS1 = {"C0": ("combined", False, 0.0), "C1": ("combined", True, 0.0), "C2": ("combined", True, NOISE),
         "T2": ("tanhrnn", True, NOISE), "B2": ("bumprnn", True, NOISE)}
VERSIONS = {"P1": dict(m=6, fmt="every", test=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32), exact_k=16, seeds=tuple(range(10, 20))),
            "H1": dict(m=12, fmt="every", test=(1, 2, 3, 4, 6, 8, 12, 16), exact_k=16, seeds=tuple(range(20, 30))),
            "H2": dict(m=6, fmt="once", test=(1, 2, 3, 4, 6, 8), exact_k=8, seeds=tuple(range(20, 30)))}


def dx_of(m, fmt):
    return m * m + m if fmt == "every" else 3 * m


def batch(rng, n, k, m, fmt):
    """X (n, T, dx); Y (n, T) target item at each step or -1; the answer is Y[:, -1]."""
    perm = np.argsort(rng.random((n, m)), axis=1)
    s = rng.integers(0, m, n)
    ys, cur = [], s.copy()
    for _ in range(k):
        cur = perm[np.arange(n), cur]; ys.append(cur.copy())
    ys = np.stack(ys, 1)
    if fmt == "every":
        T_ = np.zeros((n, m, m)); T_[np.arange(n)[:, None], np.arange(m)[None, :], perm] = 1.0
        X = np.zeros((n, k, m * m + m)); X[:, :, :m * m] = T_.reshape(n, -1)[:, None, :]
        X[np.arange(n), 0, m * m + s] = 1.0
        return X, ys
    order = np.argsort(rng.random((n, m)), axis=1)
    X = np.zeros((n, m + k, 3 * m))
    for j in range(m):
        src = order[:, j]
        X[np.arange(n), j, src] = 1.0
        X[np.arange(n), j, m + perm[np.arange(n), src]] = 1.0
    X[np.arange(n), m, 2 * m + s] = 1.0
    Y = -np.ones((n, m + k), int); Y[:, m:] = ys
    return X, Y


def bump_f(v, a, mu, s):
    Dn = s * s + 0.01
    q = (v - mu) ** 2 / Dn
    e = np.exp(-q)
    return v + a * e, (v, e, q, Dn)


def bump_b(dout, c, a, mu, s):
    v, e, q, Dn = c
    k = a * e * 2 * (v - mu) / Dn
    return dout * (1 - k), dout * e, dout * k, dout * a * e * q * 2 * s / Dn


class Net:
    broken = False

    def __init__(self, kind, rng, dx, m, H=None, u=U):
        self.kind, self.dx, self.m = kind, dx, m
        self.H = u if kind == "combined" else H
        D = dx + self.H
        if kind == "combined":
            self.K = B ** LV
            self.Wl = rng.normal(0, 1 / math.sqrt(D), (u, self.K, D))
            self.W, self.a, self.mu, self.s = [], [], [], []
            Mn = self.K
            for _ in range(LV):
                Mn //= B
                self.W.append(rng.normal(0, 1 / math.sqrt(B), (u, Mn, B)))
                self.a.append(np.full((u, Mn), 0.5)); self.mu.append(np.zeros((u, Mn))); self.s.append(np.ones((u, Mn)))
            self.F = [self.Wl] + self.W + self.a + self.mu + self.s
        else:
            self.Wr = rng.normal(0, 1 / math.sqrt(D), (self.H, D)); self.br = np.zeros(self.H)
            self.F = [self.Wr, self.br]
            if kind == "bumprnn":
                self.q = [np.full(self.H, 0.5), np.zeros(self.H), np.ones(self.H)]
                self.F += self.q
        self.Wo = rng.normal(0, 1 / math.sqrt(self.H), (self.H, m)); self.bo = np.zeros(m)
        self.ps = self.F + [self.Wo, self.bo]

    def f_fwd(self, z):
        n = z.shape[0]
        if self.kind == "combined":
            u, K, D = self.Wl.shape
            h = (z @ self.Wl.reshape(u * K, D).T).reshape(n, u, K)
            cache = [None]
            for l in range(LV):
                chn = h.reshape(n, u, -1, B)
                v = (chn * self.W[l][None]).sum(-1)
                h, c = bump_f(v, self.a[l][None], self.mu[l][None], self.s[l][None])
                cache.append((chn, c))
            return h[:, :, 0], cache
        v = z @ self.Wr.T + self.br
        if self.kind == "tanhrnn":
            return v, None
        return bump_f(v, self.q[0], self.q[1], self.q[2])

    def f_bwd(self, dout, z, cache):
        n = z.shape[0]
        if self.kind == "combined":
            u, K, D = self.Wl.shape
            dh = dout[:, :, None]
            gW, ga, gm, gs = [None] * LV, [None] * LV, [None] * LV, [None] * LV
            for l in range(LV - 1, -1, -1):
                chn, c = cache[l + 1]
                dv, da, dm, ds = bump_b(dh, c, self.a[l][None], self.mu[l][None], self.s[l][None])
                ga[l], gm[l], gs[l] = da.sum(0), dm.sum(0), ds.sum(0)
                gW[l] = (dv[:, :, :, None] * chn).sum(0)
                dh = (dv[:, :, :, None] * self.W[l][None]).reshape(n, u, -1)
            dflat = dh.reshape(n, u * K)
            gWl = (dflat.T @ z).reshape(u, K, D)
            dz = dflat @ self.Wl.reshape(u * K, D)
            return dz, [gWl] + gW + ga + gm + gs
        if self.kind == "tanhrnn":
            dv, extra = dout, []
        else:
            dv, da, dm, ds = bump_b(dout, cache, self.q[0], self.q[1], self.q[2])
            extra = [da.sum(0), dm.sum(0), ds.sum(0)]
        return dv @ self.Wr, [dv.T @ z, dv.sum(0)] + extra

    def run(self, X, noise=0.0, rng=None):
        n, T, _ = X.shape
        h = np.zeros((n, self.H))
        self.cache, self.hs = [], []
        for t in range(T):
            z = np.concatenate([X[:, t], h], 1)
            pre, c = self.f_fwd(z)
            a = np.tanh(pre)
            h = a + noise * rng.normal(size=a.shape) if noise > 0 else a
            self.cache.append((z, c, a)); self.hs.append(h)
        return h @ self.Wo + self.bo

    def logits_at(self, t):
        return self.hs[t] @ self.Wo + self.bo

    def backward(self, dlogs):
        gWo = np.zeros_like(self.Wo); gbo = np.zeros_like(self.bo)
        gF = [np.zeros_like(p) for p in self.F]
        dh = np.zeros_like(self.hs[0])
        for t in range(len(self.cache) - 1, -1, -1):
            if dlogs[t] is not None:
                gWo += self.hs[t].T @ dlogs[t]; gbo += dlogs[t].sum(0); dh = dh + dlogs[t] @ self.Wo.T
            z, c, a = self.cache[t]
            dz, g = self.f_bwd(dh * (1 - a * a), z, c)
            for i, gi in enumerate(g):
                gF[i] += gi
            dh = dz[:, self.dx:] * (0.0 if self.broken else 1.0)
        return gF + [gWo, gbo]


class BrokenNet(Net):
    broken = True


def nparams(net):
    return sum(p.size for p in net.ps)


def make(kind, rng, m, fmt, width):
    dx = dx_of(m, fmt)
    return Net(kind, rng, dx, m) if kind == "combined" else Net(kind, rng, dx, m, H=width[kind])


def widths(m, fmt):
    dx = dx_of(m, fmt)
    tgt = nparams(Net("combined", np.random.default_rng(0), dx, m))
    return {kd: min(range(8, 600), key=lambda H: abs(nparams(Net(kd, np.random.default_rng(0), dx, m, H=H)) - tgt))
            for kd in ("tanhrnn", "bumprnn")}


def step_losses(net, Y, stepsup):
    """Per-step softmax gradients (None where unsupervised) and the mean loss."""
    T = Y.shape[1]
    sup = [t for t in range(T) if (Y[:, t] >= 0).all()]
    if not stepsup:
        sup = [T - 1]
    dlogs, loss = [None] * T, 0.0
    for t in sup:
        lg = net.logits_at(t)
        z = lg - lg.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        y = Y[:, t]
        loss += float(-np.log(p[np.arange(len(y)), y] + 1e-300).mean()) / len(sup)
        p[np.arange(len(y)), y] -= 1
        dlogs[t] = p / (len(y) * len(sup))
    return dlogs, loss


def gradcheck(kind, fmt, broken=False, seed=5):
    rng = np.random.default_rng(seed)
    m = 4
    cls = BrokenNet if broken else Net
    dx = dx_of(m, fmt)
    net = cls(kind, rng, dx, m, H=6, u=3)
    for p in (net.a + net.mu + net.s if kind == "combined" else net.q if kind == "bumprnn" else []):
        p[...] = rng.normal(0.4, 0.3, p.shape)
    X, Y = batch(rng, 5, 3, m, fmt)
    net.run(X); dl, _ = step_losses(net, Y, True)
    gs = net.backward(dl)

    def L():
        net.run(X); return step_losses(net, Y, True)[1]

    ratio = 0.0
    for p, g in zip(net.ps, gs):
        for _ in range(4):
            idx = tuple(rng.integers(0, s_) for s_ in p.shape)
            o = p[idx]
            p[idx] = o + 1e-6; lp = L()
            p[idx] = o - 1e-6; lm = L()
            p[idx] = o
            num = (lp - lm) / 2e-6
            ratio = max(ratio, abs(num - g[idx]) / (1e-5 * max(abs(num), abs(g[idx])) + 1e-9))
    return ratio


def adam_clip(ps, gs, st, lr):
    nrm = math.sqrt(sum(float((g * g).sum()) for g in gs))
    if nrm > CLIP:
        gs = [g * (CLIP / nrm) for g in gs]
    for i, (p, g) in enumerate(zip(ps, gs)):
        if i not in st:
            st[i] = [np.zeros_like(p), np.zeros_like(p), 0]
        m_, v_, t = st[i]
        t += 1
        m_[...] = 0.9 * m_ + 0.1 * g
        v_[...] = 0.999 * v_ + 0.001 * g * g
        st[i][2] = t
        p -= lr * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)


def evaluate(predict, k, seed, m, fmt, n=NEVAL):
    X, Y = batch(np.random.default_rng(10_000 + 97 * seed + k + 1000 * m), n, k, m, fmt)
    return float((predict(X) == Y[:, -1]).mean())


def net_predict(net):
    def f(X):
        lg = net.run(X)
        return np.where(np.all(np.isfinite(lg), 1), lg.argmax(1), -1)
    return f


def job(args):
    ver, arm, seed = args
    V = VERSIONS[ver]
    kind, stepsup, noise = ARMS1[arm] if ver == "P1" else ({"C": "combined", "T": "tanhrnn", "B": "bumprnn"}[arm[0]], True, NOISE)
    W = widths(V["m"], V["fmt"])
    net = make(kind, np.random.default_rng(seed), V["m"], V["fmt"], W)
    rng = np.random.default_rng(700 + seed)
    st, t0 = {}, time.time()
    for _ in range(STEPS):
        k = int(rng.choice(TRAIN_K))
        X, Y = batch(rng, BS, k, V["m"], V["fmt"])
        lg = net.run(X, noise, rng)
        if not np.all(np.isfinite(lg)):
            break
        dl, _ = step_losses(net, Y, stepsup)
        adam_clip(net.ps, net.backward(dl), st, LR)
    acc = {k_: evaluate(net_predict(net), k_, seed, V["m"], V["fmt"]) for k_ in V["test"]}
    return dict(version=ver, arm=arm, seed=seed, acc=acc, params=nparams(net),
                ind=float(np.mean([acc[k_] for k_ in TRAIN_K])),
                long=float(np.mean([acc[k_] for k_ in V["test"] if k_ > 3])),
                seconds=round(time.time() - t0, 1))


def cmp(units, a, b, seeds):
    d = [units[(a, s)] - units[(b, s)] for s in seeds if (a, s) in units and (b, s) in units]
    w = sum(x > 0 for x in d); l = sum(x < 0 for x in d); n = len(d)
    p = sum(math.comb(n, i) for i in range(w, n + 1)) / 2 ** n if n else 1.0
    v = "BETTER" if w >= 9 else "WORSE" if l >= 9 else "NO DIFFERENCE"
    return w, l, n, p, float(np.mean(d)) if d else float("nan"), v


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    P_(RULE); P_("EXACT LONG CHAINS, THEN HARDER CHAINS"); P_(RULE)
    ok = True
    for kd in ("combined", "bumprnn", "tanhrnn"):
        for fmt in ("every", "once"):
            g, bg = gradcheck(kd, fmt), gradcheck(kd, fmt, broken=True)
            ok &= g <= 1.0 and bg > 1.0
            P_(f"  Q0 {kd:<9} {fmt:<5} gradient/tol {g:.3f}   broken {bg:.2e} ({'caught' if bg > 1 else 'NOT CAUGHT'})")
    h_ok = True
    for ver, V in VERSIONS.items():
        def exact(X, V=V):
            m = V["m"]; n = X.shape[0]
            if V["fmt"] == "every":
                perm = X[:, 0, :m * m].reshape(n, m, m).argmax(2); cur = X[:, 0, m * m:].argmax(1)
                k = X.shape[1]
            else:
                perm = np.zeros((n, m), int)
                for j in range(m):
                    perm[np.arange(n), X[:, j, :m].argmax(1)] = X[:, j, m:2 * m].argmax(1)
                cur = X[:, m, 2 * m:].argmax(1); k = X.shape[1] - m
            for _ in range(k):
                cur = perm[np.arange(n), cur]
            return cur
        ex = min(evaluate(exact, k_, 0, V["m"], V["fmt"], 300) for k_ in V["test"])
        co = float(np.mean([evaluate(lambda X: np.zeros(X.shape[0], int), k_, 0, V["m"], V["fmt"]) for k_ in V["test"]]))
        P_(f"  Q1 {ver}: exact solver min {ex:.3f}, constant guesser {co:.3f} (1/items {1 / V['m']:.3f})")
        h_ok &= ex == 1.0 and abs(co - 1 / V["m"]) <= 0.03
    if not (ok and h_ok):
        P_("  BLOCKED."); open(OUT, "w").write("\n".join(out) + "\n"); return
    for ver, V in VERSIONS.items():
        W = widths(V["m"], V["fmt"])
        P_(f"  {ver} parameters: combined {nparams(make('combined', np.random.default_rng(0), V['m'], V['fmt'], W)):,}, "
           + ", ".join(f"{kd} {nparams(make(kd, np.random.default_rng(0), V['m'], V['fmt'], W)):,} (width {W[kd]})" for kd in W))
    jobs = ([("P1", a, s) for a in ARMS1 for s in VERSIONS["P1"]["seeds"]]
            + [(v, a, s) for v in ("H1", "H2") for a in ("C2", "T2", "B2") for s in VERSIONS[v]["seeds"]])
    jobs.sort(key=lambda j: (j[1][0] != "C", j[0] != "H1"))
    pool = mp.get_context("fork").Pool(4)
    runs = list(pool.imap_unordered(job, jobs, chunksize=1))
    pool.close()
    res = {}
    for ver, V in VERSIONS.items():
        arms = list(ARMS1) if ver == "P1" else ["C2", "T2", "B2"]
        name = {"C0": "combined", "C1": "combined +S", "C2": "combined +S+N", "T2": "tanh mem +S+N", "B2": "bump mem +S+N"}
        P_("\n" + RULE)
        P_({"P1": "PART 1 -- EXACTNESS (6 items, facts every step)", "H1": "H1 -- TWELVE ITEMS",
            "H2": "H2 -- FACTS SHOWN ONCE (6 items)"}[ver] + "; trained on 1-3 hops")
        P_(RULE)
        P_("    " + f"{'':<16}" + "".join(f"{'k=' + str(k_):>7}" for k_ in V["test"]) + "   learnt  exact@" + str(V["exact_k"]))
        units = {}
        for a in arms:
            rs = [r for r in runs if r["version"] == ver and r["arm"] == a]
            lr_ = [r for r in rs if r["ind"] >= 0.90]
            ex = sum(r["acc"][V["exact_k"]] >= 0.99 for r in rs)
            for r in lr_:
                units[(a, r["seed"])] = r["long"]
            P_(f"    {name[a]:<16}" + "".join(f"{np.mean([r['acc'][k_] for r in rs]):>7.3f}" for k_ in V["test"])
               + f"   {len(lr_):>2}/10   {ex:>2}/10" + ("  EXACT" if ex >= 9 else ""))
            res[(ver, a, "exact")] = ex
            res[(ver, a, "learnt")] = len(lr_)
        pairs = ([("C2", "C0", "X2 do the fixes help?"), ("C1", "C0", "X3 step supervision alone"),
                  ("C2", "C1", "X3 adding state noise"), ("C2", "T2", "X4 architecture vs tanh memory (both fixed)"),
                  ("C2", "B2", "X4 architecture vs bump memory (both fixed)")] if ver == "P1" else
                 [("C2", "T2", "combined vs tanh memory"), ("C2", "B2", "combined vs bump memory")])
        P_("\n  long score = mean accuracy at depths > 3, per seed; >= 9/10 rule")
        for a_, b_, nm in pairs:
            if res[(ver, a_, "learnt")] < 9 or res[(ver, b_, "learnt")] < 9:
                P_(f"    {nm}: NO VERDICT (learnt {res[(ver, a_, 'learnt')]}/10 vs {res[(ver, b_, 'learnt')]}/10)"); continue
            w, l, n, p, md, v = cmp(units, a_, b_, V["seeds"])
            res[(ver, a_, b_)] = dict(wins=w, losses=l, n=n, p=p, mean=md, verdict=v)
            P_(f"    {nm:<46} {name[a_]} higher on {w}/{n}, mean {md:+.3f}, p {p:.4f} -> {name[a_].upper()} {v}")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"results": {"|".join(map(str, k)): v for k, v in res.items()},
               "runs": [{**r, "acc": {str(k_): v for k_, v in r["acc"].items()}} for r in runs]}, open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/chainexact.json   runtime {time.time() - t0:.0f}s")
    P_("  NOT: synthetic chains; one noise level, one budget, one tree shape, not swept.")
    open(OUT, "w").write("\n".join(out) + "\n")


def diagnose():
    """POST-RUN (added after seeing the plateau near 0.90): retrain C2 on seeds 10 and 11 exactly as
    in the run and split accuracy at 16 hops by the length of the cycle containing the start item."""
    out = ["", RULE, "POST-RUN DIAGNOSTIC: WHERE DOES THE FIXED COMBINED NETWORK FAIL? (C2, seeds 10-11, 16 hops)", RULE]
    for seed in (10, 11):
        r_net = {}
        V = VERSIONS["P1"]; W = widths(V["m"], V["fmt"])
        net = make("combined", np.random.default_rng(seed), V["m"], V["fmt"], W)
        rng = np.random.default_rng(700 + seed); st = {}
        for _ in range(STEPS):
            k = int(rng.choice(TRAIN_K)); X, Y = batch(rng, BS, k, V["m"], V["fmt"])
            net.run(X, NOISE, rng); dl, _ = step_losses(net, Y, True); adam_clip(net.ps, net.backward(dl), st, LR)
        X, Y = batch(np.random.default_rng(10_000 + 97 * seed + 16 + 6000), 4000, 16, 6, "every")
        perm = X[:, 0, :36].reshape(-1, 6, 6).argmax(2); s0 = X[:, 0, 36:].argmax(1)
        cyc = np.zeros(len(s0), int)
        for i in range(len(s0)):
            c, L = perm[i, s0[i]], 1
            while c != s0[i]:
                c, L = perm[i, c], L + 1
            cyc[i] = L
        ok = net_predict(net)(X) == Y[:, -1]
        out.append(f"  seed {seed}: overall {ok.mean():.3f}   by cycle length of the start item: " + "   ".join(
            f"{L}: {ok[cyc == L].mean():.3f} (n {int((cyc == L).sum())})" for L in range(1, 7) if (cyc == L).any()))
        acc1 = []
        for kk in (1, 2, 3, 4, 8, 16):
            Xk, Yk = batch(np.random.default_rng(55 + kk), 2000, kk, 6, "every")
            pk = Xk[:, 0, :36].reshape(-1, 6, 6).argmax(2); sk = Xk[:, 0, 36:].argmax(1)
            fixed = pk[np.arange(len(sk)), sk] == sk
            okk = net_predict(net)(Xk) == Yk[:, -1]
            acc1.append(f"k={kk}: fixed-point starts {okk[fixed].mean():.3f}, others {okk[~fixed].mean():.3f}")
        out.append("      " + "  |  ".join(acc1))
    print("\n".join(out))
    open(OUT, "a").write("\n".join(out) + "\n")


if __name__ == "__main__":
    import sys
    if "--diagnose" in sys.argv:
        diagnose()
    else:
        main()
