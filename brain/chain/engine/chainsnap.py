"""A snapping step: write down each intermediate answer, so chain errors cannot accumulate.

FROM chainexact.py. With step supervision and state noise the combined network reached 0.90 at 16
hops but not the 0.99 bar; the diagnostic showed a small residual error on EVERY hop (~0.3%) that
accumulates -- no structural blind spot. A continuous state cannot be exactly the next item.

THE SNAP. The state carried to the next hop is split in two:
    h_t  continuous memory (context; in the facts-once task it must hold the table)
    e_t  the WRITTEN-DOWN pointer: e_t = Emb[item_t], a learnt code (16 numbers) for one of the items
    step:  h_t = tanh(F([x_t, e_{t-1}, h_{t-1}]))   +  training noise (sd 0.2),
           item_t = argmax(Wo h_t + bo)              read out after every hop
Training: step supervision with TEACHER FORCING (e_t = Emb[true item after hop t]); test: e_t =
Emb[the network's own argmax] -- a hard, discrete snap. If one hop is computed exactly, any number of
hops is exact. In the facts-once task the snap starts at the first reasoning step; reading steps
carry e = 0. Parameters re-matched across the three update functions (Emb included).

=================================================================================================
GATES, PREDECLARED
=================================================================================================
Q0 BPTT gradient (teacher-forced, per-step loss, both formats, Emb included) within tolerance; a
   broken copy (gradient to the previous continuous state dropped) MUST fail.
Q1 Harness: chainexact's exact-solver / constant-guesser check on the same evaluation code.
Seeds and test trials IDENTICAL to chainexact.py (P1 seeds 10-19; H1, H2 seeds 20-29), so the
no-snap reference is chainexact's C2 / T2 / B2 run on the same seeds and the same test questions.
Fixed 3,000 steps; trained on 1-3 hops.
   P1  6 items, facts every step    test 1-4, 6, 8, 12, 16, 24, 32, 64
   H1  12 items, facts every step   test 1-4, 6, 8, 12, 16
   H2  6 items, facts shown once    test 1-4, 6, 8
Z1 EXACT if accuracy >= 0.99 at the version's deep test (P1: 16 AND 64; H1: 16; H2: 8) on >= 9/10.
Z2 DOES SNAPPING HELP? snap vs no-snap (chainexact) on the long score (mean over depths > 3 shared
   with chainexact), per seed, >= 9/10.
   DISCLOSED: the 16-number pointer code feeds every unit, so snapped networks are larger than
   chainexact's (P1 32.6k vs 26.3k, H1 76.6k vs 70.3k, H2 23.4k vs 17.1k); Z2 carries that confound.
Z3 DOES THE TREE STILL MATTER WITH SNAPPING? combined vs tanh / bump memory (both snapped), >= 9/10,
   only where both have >= 9/10 seeds learnt (in-distribution >= 0.90).
Z4 WHAT THIS IS AND IS NOT. The snap is a discrete scratchpad: it is designed in, not learnt. If every
   model becomes exact with it, exactness belongs to the snap, and that will be said.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util
import json
import math
import multiprocessing as mp
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_chainsnap.txt")
ART = R("outputs", "chainsnap.json")
RULE = "=" * 97
_s = importlib.util.spec_from_file_location("chainexact", os.path.join(HERE, "chainexact.py"))
ce = importlib.util.module_from_spec(_s); _s.loader.exec_module(ce)

P = 16
VERS = {"P1": dict(m=6, fmt="every", test=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64), exact=(16, 64), seeds=tuple(range(10, 20))),
        "H1": dict(m=12, fmt="every", test=(1, 2, 3, 4, 6, 8, 12, 16), exact=(16,), seeds=tuple(range(20, 30))),
        "H2": dict(m=6, fmt="once", test=(1, 2, 3, 4, 6, 8), exact=(8,), seeds=tuple(range(20, 30)))}
KINDS = ("combined", "tanhrnn", "bumprnn")
REF = {"combined": "C2", "tanhrnn": "T2", "bumprnn": "B2"}


class SnapNet(ce.Net):
    def __init__(self, kind, rng, dx, m, H=None, u=ce.U):
        super().__init__(kind, rng, dx + P, m, H=H, u=u)
        self.dx0 = dx
        self.Emb = rng.normal(0, 1 / math.sqrt(P), (m, P))
        self.ps = self.F + [self.Wo, self.bo, self.Emb]

    def run(self, X, Y=None, start=0, noise=0.0, rng=None):
        n, T, _ = X.shape
        h = np.zeros((n, self.H)); e = np.zeros((n, P))
        self.cache, self.hs, self.items = [], [], []
        for t in range(T):
            z = np.concatenate([X[:, t], e, h], 1)
            pre, c = self.f_fwd(z)
            a = np.tanh(pre)
            h = a + noise * rng.normal(size=a.shape) if noise > 0 else a
            self.cache.append((z, c, a)); self.hs.append(h)
            if t >= start:
                item = Y[:, t] if Y is not None else (h @ self.Wo + self.bo).argmax(1)
                e = self.Emb[item]; self.items.append(item)
            else:
                e = np.zeros((n, P)); self.items.append(None)
        return h @ self.Wo + self.bo

    def backward(self, dlogs):
        gWo = np.zeros_like(self.Wo); gbo = np.zeros_like(self.bo); gE = np.zeros_like(self.Emb)
        gF = [np.zeros_like(p) for p in self.F]
        n = self.hs[0].shape[0]
        dh = np.zeros((n, self.H)); de = np.zeros((n, P))
        for t in range(len(self.cache) - 1, -1, -1):
            if self.items[t] is not None:
                np.add.at(gE, self.items[t], de)
            if dlogs[t] is not None:
                gWo += self.hs[t].T @ dlogs[t]; gbo += dlogs[t].sum(0); dh = dh + dlogs[t] @ self.Wo.T
            z, c, a = self.cache[t]
            dz, g = self.f_bwd(dh * (1 - a * a), z, c)
            for i, gi in enumerate(g):
                gF[i] += gi
            de = dz[:, self.dx0:self.dx]
            dh = dz[:, self.dx:] * (0.0 if self.broken else 1.0)
        return gF + [gWo, gbo, gE]


class BrokenSnap(SnapNet):
    broken = True


def start_of(m, fmt):
    return 0 if fmt == "every" else m


def widths(m, fmt):
    dx = ce.dx_of(m, fmt)
    tgt = ce.nparams(SnapNet("combined", np.random.default_rng(0), dx, m))
    return {kd: min(range(8, 600), key=lambda H: abs(ce.nparams(SnapNet(kd, np.random.default_rng(0), dx, m, H=H)) - tgt))
            for kd in ("tanhrnn", "bumprnn")}


def make(kind, rng, m, fmt, W):
    dx = ce.dx_of(m, fmt)
    return SnapNet(kind, rng, dx, m) if kind == "combined" else SnapNet(kind, rng, dx, m, H=W[kind])


def gradcheck(kind, fmt, broken=False, seed=5):
    rng = np.random.default_rng(seed)
    m = 4; dx = ce.dx_of(m, fmt)
    net = (BrokenSnap if broken else SnapNet)(kind, rng, dx, m, H=6, u=3)
    for p in (net.a + net.mu + net.s if kind == "combined" else net.q if kind == "bumprnn" else []):
        p[...] = rng.normal(0.4, 0.3, p.shape)
    X, Y = ce.batch(rng, 5, 3, m, fmt)
    st = start_of(m, fmt)

    def L():
        net.run(X, Y, st); return ce.step_losses(net, Y, True)[1]

    net.run(X, Y, st); dl, _ = ce.step_losses(net, Y, True); gs = net.backward(dl)
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


def job(args):
    ver, kind, seed = args
    V = VERS[ver]; m, fmt = V["m"], V["fmt"]; st0 = start_of(m, fmt)
    net = make(kind, np.random.default_rng(seed), m, fmt, widths(m, fmt))
    rng = np.random.default_rng(700 + seed)
    st, t0 = {}, time.time()
    for _ in range(ce.STEPS):
        k = int(rng.choice(ce.TRAIN_K))
        X, Y = ce.batch(rng, ce.BS, k, m, fmt)
        lg = net.run(X, Y, st0, ce.NOISE, rng)
        if not np.all(np.isfinite(lg)):
            break
        dl, _ = ce.step_losses(net, Y, True)
        ce.adam_clip(net.ps, net.backward(dl), st, ce.LR)

    def pred(X):
        lg = net.run(X, None, st0)
        return np.where(np.all(np.isfinite(lg), 1), lg.argmax(1), -1)

    acc = {k_: ce.evaluate(pred, k_, seed, m, fmt) for k_ in V["test"]}
    return dict(version=ver, kind=kind, seed=seed, acc=acc, params=ce.nparams(net),
                ind=float(np.mean([acc[k_] for k_ in ce.TRAIN_K])), seconds=round(time.time() - t0, 1))


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    P_(RULE); P_("THE SNAPPING STEP: WRITE DOWN EACH INTERMEDIATE ANSWER"); P_(RULE)
    ok = True
    for kd in KINDS:
        for fmt in ("every", "once"):
            g, bg = gradcheck(kd, fmt), gradcheck(kd, fmt, broken=True)
            ok &= g <= 1.0 and bg > 1.0
            P_(f"  Q0 {kd:<9} {fmt:<5} gradient/tol {g:.3f}   broken {bg:.2e} ({'caught' if bg > 1 else 'NOT CAUGHT'})")
    if not ok:
        P_("  BLOCKED."); open(OUT, "w").write("\n".join(out) + "\n"); return
    ref = json.load(open(R("outputs", "chainexact.json")))["runs"]
    for ver, V in VERS.items():
        W = widths(V["m"], V["fmt"])
        P_(f"  {ver} parameters: combined {ce.nparams(make('combined', np.random.default_rng(0), V['m'], V['fmt'], W)):,}, "
           + ", ".join(f"{kd} {ce.nparams(make(kd, np.random.default_rng(0), V['m'], V['fmt'], W)):,}" for kd in W))
    jobs = [(v, kd, s) for v in VERS for kd in KINDS for s in VERS[v]["seeds"]]
    jobs.sort(key=lambda j: (j[1] != "combined", j[0] != "H2"))
    pool = mp.get_context("fork").Pool(4)
    runs = list(pool.imap_unordered(job, jobs, chunksize=1))
    pool.close()
    res = {}
    for ver, V in VERS.items():
        P_("\n" + RULE)
        P_({"P1": "P1 -- 6 ITEMS, FACTS EVERY STEP", "H1": "H1 -- 12 ITEMS", "H2": "H2 -- FACTS SHOWN ONCE"}[ver]
           + "; trained on 1-3 hops; SNAPPED (no-snap reference = chainexact, same seeds and questions)")
        P_(RULE)
        P_("    " + f"{'':<20}" + "".join(f"{'k=' + str(k_):>7}" for k_ in V["test"]) + "   learnt  exact")
        shared = [k_ for k_ in V["test"] if k_ > 3 and k_ != 64]
        snapu, refu, learnt = {}, {}, {}
        for kd in KINDS:
            rs = sorted([r for r in runs if r["version"] == ver and r["kind"] == kd], key=lambda r: r["seed"])
            ex = sum(all(r["acc"][k_] >= 0.99 for k_ in V["exact"]) for r in rs)
            learnt[kd] = sum(r["ind"] >= 0.90 for r in rs)
            P_(f"    {kd + ' SNAP':<20}" + "".join(f"{np.mean([r['acc'][k_] for r in rs]):>7.3f}" for k_ in V["test"])
               + f"   {learnt[kd]:>2}/10   {ex:>2}/10" + ("  EXACT" if ex >= 9 else ""))
            rr = [x for x in ref if x["version"] == ver and x["arm"] == REF[kd]]
            P_(f"    {kd + ' no snap':<20}" + "".join(
                (f"{np.mean([x['acc'][str(k_)] for x in rr]):>7.3f}" if str(k_) in rr[0]["acc"] else f"{'-':>7}") for k_ in V["test"]))
            res[(ver, kd, "exact")] = ex; res[(ver, kd, "learnt")] = learnt[kd]
            for r in rs:
                snapu[(kd, r["seed"])] = float(np.mean([r["acc"][k_] for k_ in shared]))
            for x in rr:
                refu[(kd, x["seed"])] = float(np.mean([x["acc"][str(k_)] for k_ in shared]))
        P_("\n  Z2 does snapping help? (long score over depths " + ",".join(map(str, shared)) + ", same seeds)")
        for kd in KINDS:
            d = [snapu[(kd, s)] - refu[(kd, s)] for s in V["seeds"]]
            w = sum(x > 0 for x in d); l = sum(x < 0 for x in d)
            v = "HELPS" if w >= 9 else "HURTS" if l >= 9 else "NO CLEAR EFFECT"
            res[(ver, kd, "z2")] = dict(wins=w, losses=l, mean=float(np.mean(d)), verdict=v)
            P_(f"    {kd:<9} snap higher on {w}/10, mean {np.mean(d):+.3f} -> {v}")
        P_("  Z3 does the tree still matter with snapping?")
        for b_ in ("tanhrnn", "bumprnn"):
            if learnt["combined"] < 9 or learnt[b_] < 9:
                P_(f"    combined vs {b_}: NO VERDICT (learnt {learnt['combined']}/10 vs {learnt[b_]}/10)"); continue
            d = [snapu[("combined", s)] - snapu[(b_, s)] for s in V["seeds"]]
            w = sum(x > 0 for x in d); l = sum(x < 0 for x in d)
            v = "TREE BETTER" if w >= 9 else "TREE WORSE" if l >= 9 else "NO DIFFERENCE"
            res[(ver, "z3", b_)] = dict(wins=w, losses=l, mean=float(np.mean(d)), verdict=v)
            P_(f"    combined vs {b_}: combined higher on {w}/10, lower on {l}/10, mean {np.mean(d):+.3f} -> {v}")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"results": {"|".join(map(str, k)): v for k, v in res.items()},
               "runs": [{**r, "acc": {str(k_): v for k_, v in r["acc"].items()}} for r in runs]}, open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/chainsnap.json   runtime {time.time() - t0:.0f}s")
    P_("  Z4: the snap is a designed discrete scratchpad; if all models become exact, exactness belongs to it.")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
