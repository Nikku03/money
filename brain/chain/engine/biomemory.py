"""A memory for the chain network, built from Blue Brain synapse data and optogenetics recordings.

THE WEAKNESS IT TARGETS (chainsnap.py, transvs.py). The combined network is exact on long chains
when the facts are re-shown every step, but it cannot hold facts shown ONCE (all models 0/10) and
loses to a transformer on a 12-item table -- the transformer keeps every fact in view; ours has to
squeeze them into a 24-unit state.

THE BIOLOGY. The synaptic theory of working memory (Mongillo, Barak & Tsodyks, Science 2008) holds
items for ~1 s in short-term synaptic FACILITATION, not in spiking. So the memory here is synaptic:
each fact (source -> target) strengthens a fast synaptic trace, which fades on its own clock:
    A_t = lambda * A_{t-1} + U * sum_facts v(target) k(source)^T          (P = 16 code units)
    read with the current pointer:  r_t = A_t k(pointer)
and the network's ongoing activity leaks on a measured clock:
    h_t = (1 - alpha) h_{t-1} + alpha tanh(F([x_t, k(pointer), r_t, h_{t-1}]))
F is the combined bump-node tree; the pointer is snapped every hop (chainsnap.py). Codes k(.) and
v(.) are learnt (sigmoid, so non-negative like firing rates). One step = 0.1 s (memcommit.py).
Disclosed simplification: the Tsodyks-Markram SATURATION (u -> u + U(1 - u)) is not modelled -- the
trace is linear Hebbian with the measured time constant and release probability. Pair-specific
(Hebbian) writing is an ASSUMPTION; the Blue Brain data give the time course and size, not the rule.

=================================================================================================
PART A -- THE DATA -> THE NUMBERS (gates fixed now)
=================================================================================================
A1 BLUE BRAIN (NMC portal, pathways_physiology_factsheets_simplified.json; Markram et al. 2015).
   Pathways labelled 'Excitatory, facilitating'. U = median u_mean; tau_F = median f_mean (ms).
   Synapse-count-weighted medians (anatomy factsheets) reported beside them. Gate: >= 5 pathways.
   ->  lambda = exp(-0.1 s / tau_F),   write gain = U.
A2 OPTOGENETICS (DANDI 000060, ALM units 'good'/'ok', mean rate >= 1 Hz in the window).
   Intrinsic timescale (Murray et al. 2014): spike counts in 17 bins of 0.1 s from 4.6 to 2.9 s
   before the go cue (before any sample or pre-sample pulse) on trials with no pulse in that window,
   no early lick, outcome hit/miss. Across-trial correlation of bin i with bin j, averaged by lag
   and over units; fit R(lag) = A (exp(-lag/tau) + B) on lags 0.1-1.6 s. Bootstrap over units (200).
   Gate: fit R^2 >= 0.5 and 0.02 s <= tau <= 10 s.   ->  alpha = 1 - exp(-0.1 s / tau_ALM).

=================================================================================================
PART B -- THE TEST (seeds and test questions identical to chainsnap.py; 3,000 steps; 1-3 hops)
=================================================================================================
   BIO      lambda, U from Blue Brain, alpha from ALM -- fixed
   GENERIC  the non-biological twin: lambda and U LEARNT (init 0.9, 0.5), no leak (alpha = 1)
   versions P1 (6 items, facts every step), H1 (12 items), H2 (6 items, facts shown ONCE)
Q0 BPTT gradient check for both, both formats, memory parameters included; a broken copy (the
   trace's carry-over gradient lambda * dA dropped) MUST fail.
B1 PRIMARY -- H2: does the biological memory hold facts shown once? LEARNT if accuracy on 1-3 hops
   >= 0.90; EXACT if >= 0.99 at 8 hops; both need >= 9/10 seeds. Versus the network without memory
   (chainsnap combined) on long score (depths > 3), >= 9/10 rule.
B2 BIOLOGY vs GENERIC, every version, long score, >= 9/10 -> BIO BETTER / GENERIC BETTER / NO DIFFERENCE.
B3 P1 and H1: learnt / exact / vs no memory, as B1 (exact at 16 and 64 for P1, at 16 for H1).
B4 The transformer + CoT numbers (transvs.py, printed means only -- its per-seed file was lost) are
   shown for reference, not tested.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import importlib.util
import json
import math
import multiprocessing as mp
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_biomemory.txt")
ART = R("outputs", "biomemory.json")
BBP = os.path.join(HERE, "_cache", "bbp")
OPTO = os.path.join(HERE, "_cache", "opto_000060")
RULE = "=" * 97
_s = importlib.util.spec_from_file_location("chainexact", os.path.join(HERE, "chainexact.py"))
ce = importlib.util.module_from_spec(_s); _s.loader.exec_module(ce)

DT, P = 0.1, 16
VERS = {"P1": dict(m=6, fmt="every", test=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64), exact=(16, 64), seeds=tuple(range(10, 20))),
        "H1": dict(m=12, fmt="every", test=(1, 2, 3, 4, 6, 8, 12, 16), exact=(16,), seeds=tuple(range(20, 30))),
        "H2": dict(m=6, fmt="once", test=(1, 2, 3, 4, 6, 8), exact=(8,), seeds=tuple(range(20, 30)))}
TRANSFORMER_COT = {"P1": {4: 1.000, 8: 0.993, 16: 0.921, 32: 0.752, 64: 0.554},
                   "H1": {4: 1.000, 8: 1.000, 16: 1.000}, "H2": {4: 1.000, 8: 0.999}}
BIO = {}


# ---------------------------------------------------------------- PART A ----------------------
def bbp_numbers():
    phys = json.load(open(os.path.join(BBP, "pathways_physiology_factsheets_simplified.json")))
    anat = json.load(open(os.path.join(BBP, "pathways_anatomy_factsheets_simplified.json")))
    out = {}
    for cls in sorted({v.get("synapse_type") for v in phys.values()}):
        ks = [k for k, v in phys.items() if v.get("synapse_type") == cls]
        u = np.array([phys[k]["u_mean"] for k in ks]); f = np.array([phys[k]["f_mean"] for k in ks])
        d = np.array([phys[k]["d_mean"] for k in ks])
        w = np.array([anat.get(k, {}).get("total_synapse_count", 0) for k in ks], float)

        def wmed(x):
            if w.sum() <= 0:
                return float("nan")
            o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
            return float(x[o][np.searchsorted(c, 0.5)])
        out[cls] = dict(n=len(ks), U=float(np.median(u)), tau_F_ms=float(np.median(f)), tau_D_ms=float(np.median(d)),
                        U_w=wmed(u), tau_F_w=wmed(f), tau_D_w=wmed(d))
    return out


def alm_timescale(rng):
    import h5py
    from scipy.optimize import curve_fit
    dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a[()]]
    man = json.load(open(os.path.join(OPTO, "manifest.json")))
    edges = np.round(np.arange(-4.6, -2.9 + 1e-9, 0.1), 2)
    nb = len(edges) - 1
    unit_ac, n_units = [], 0
    for m in man:
        f = h5py.File(os.path.join(OPTO, m["path"].replace("/", "__")), "r")
        if json.loads(f[f["units"]["electrode_group"][0]].attrs["location"]).get("brain_area") != "ALM":
            f.close(); continue
        t = f["intervals/trials"]
        st, sp = t["start_time"][()], t["stop_time"][()]
        on, el, oc = dec(t["photostim_onset"]), dec(t["early_lick"]), dec(t["outcome"])
        le = f["acquisition/LabeledEvents"]
        labels = [x.decode() if isinstance(x, bytes) else str(x) for x in le["data"].attrs["labels"]]
        go_all = np.sort(le["timestamps"][()][le["data"][()] == labels.index("go_start_times")])
        gos = []
        for i in range(len(st)):
            g = go_all[(go_all >= st[i]) & (go_all <= sp[i])]
            if not len(g) or el[i] != "no early" or oc[i] not in ("hit", "miss"):
                continue
            ons = [float(x) for x in on[i].split(",") if x.strip() not in ("N/A", "")]
            if any(2.9 < o_ < 4.6 + 0.4 for o_ in ons):          # a pulse inside (or bleeding into) the window
                continue
            gos.append(g[0])
        gos = np.array(gos)
        if len(gos) < 20:
            f.close(); continue
        u = f["units"]; idx = u["spike_times_index"][()]; spk = u["spike_times"][()]
        starts = np.concatenate([[0], idx[:-1]]); qual = dec(u["quality"])
        for k in range(len(idx)):
            if qual[k] not in ("good", "ok"):
                continue
            s = spk[starts[k]:idx[k]]
            cnt = np.stack([np.searchsorted(s, gos + edges[b + 1]) - np.searchsorted(s, gos + edges[b]) for b in range(nb)], 1)
            if cnt.mean() / DT < 1.0:
                continue
            ac = collections.defaultdict(list)
            for i in range(nb):
                for j in range(i + 1, nb):
                    a_, b_ = cnt[:, i], cnt[:, j]
                    if a_.std() > 0 and b_.std() > 0:
                        ac[j - i].append(np.corrcoef(a_, b_)[0, 1])
            if len(ac) == nb - 1:
                unit_ac.append([np.mean(ac[l]) for l in range(1, nb)])
                n_units += 1
        f.close()
    U_ = np.array(unit_ac); lags = DT * np.arange(1, nb)
    fun = lambda x, A, tau, B: A * (np.exp(-x / tau) + B)

    def fit(curve):
        p, _ = curve_fit(fun, lags, curve, p0=(0.2, 0.3, 0.1), bounds=([0, 0.005, -1], [2, 50, 5]), maxfev=20000)
        pred = fun(lags, *p)
        r2 = 1 - ((curve - pred) ** 2).sum() / ((curve - curve.mean()) ** 2).sum()
        return p, r2

    pop = U_.mean(0)
    p, r2 = fit(pop)
    boots = []
    for _ in range(200):
        try:
            boots.append(fit(U_[rng.integers(0, len(U_), len(U_))].mean(0))[0][1])
        except Exception:
            pass
    return dict(units=n_units, tau=float(p[1]), A=float(p[0]), B=float(p[2]), r2=float(r2),
                ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], curve=pop.tolist(), lags=lags.tolist())


# ---------------------------------------------------------------- PART B ----------------------
def sig(x):
    return 1 / (1 + np.exp(-x))


class MemNet(ce.Net):
    broken = False

    def __init__(self, kind, rng, dx, m, mode, u=ce.U):
        super().__init__(kind, rng, dx + 2 * P, m, u=u)
        self.dx0, self.m, self.mode = dx, m, mode
        self.Ek = rng.normal(0, 1.0, (m, P)); self.Ev = rng.normal(0, 1.0, (m, P))
        self.lam_l = np.array([math.log(0.9 / 0.1)]); self.U_l = np.array([math.log(math.exp(0.5) - 1)])
        extra = [self.Ek, self.Ev] + ([self.lam_l, self.U_l] if mode == "generic" else [])
        self.ps = self.F + [self.Wo, self.bo] + extra

    def lamU(self):
        if self.mode == "bio":
            return BIO["lambda"], BIO["U"]
        return float(sig(self.lam_l[0])), float(np.log1p(np.exp(self.U_l[0])))

    def alpha(self):
        return BIO["alpha"] if self.mode == "bio" else 1.0

    def facts(self, X, t, fmt):
        """(src index array, tgt index array) of the facts presented at step t, or None."""
        m = self.m
        if fmt == "every":
            perm = X[:, t, :m * m].reshape(-1, m, m)
            return np.tile(np.arange(m), (X.shape[0], 1)), perm.argmax(2)
        if t < m:
            return X[:, t, :m].argmax(1)[:, None], X[:, t, m:2 * m].argmax(1)[:, None]
        return None

    def run(self, X, Y=None, fmt="every", noise=0.0, rng=None):
        n, T, _ = X.shape
        m = self.m; lam, U = self.lamU(); al = self.alpha()
        h = np.zeros((n, self.H)); A = np.zeros((n, P, P))
        start = 0 if fmt == "every" else m
        sK, sV = sig(self.Ek), sig(self.Ev)
        ptr = None
        self.cache, self.hs, self.items, self.As, self.Ws, self.F_ = [], [], [], [], [], []
        for t in range(T):
            fa = self.facts(X, t, fmt)
            W = np.zeros((n, P, P))
            if fa is not None:
                src, tgt = fa
                W = np.einsum('nfi,nfj->nij', sV[tgt], sK[src])
            Ap = A
            A = lam * A + U * W
            if t == start:
                ptr = X[:, t, (m * m if fmt == "every" else 2 * m):][:, :m].argmax(1)
            q = sK[ptr] if ptr is not None else np.zeros((n, P))
            r = np.einsum('nij,nj->ni', A, q)
            z = np.concatenate([X[:, t], q, r, h], 1)
            pre, c = self.f_fwd(z)
            a = np.tanh(pre)
            hp = h
            h = (1 - al) * h + al * a
            if noise > 0:
                h = h + noise * rng.normal(size=h.shape)
            self.cache.append((z, c, a, hp)); self.hs.append(h); self.As.append((Ap, A)); self.Ws.append((W, fa))
            self.F_.append(ptr.copy() if ptr is not None else None)
            if t >= start:
                ptr = Y[:, t] if Y is not None else (h @ self.Wo + self.bo).argmax(1)
            self.items.append(None)
        return h @ self.Wo + self.bo

    def backward(self, dlogs):
        lam, U = self.lamU(); al = self.alpha()
        sK, sV = sig(self.Ek), sig(self.Ev)
        gWo = np.zeros_like(self.Wo); gbo = np.zeros_like(self.bo)
        gF = [np.zeros_like(p) for p in self.F]
        gEk = np.zeros_like(self.Ek); gEv = np.zeros_like(self.Ev)
        glam = gU = 0.0
        n = self.hs[0].shape[0]
        dh = np.zeros((n, self.H)); dA = np.zeros((n, P, P))
        dsK = np.zeros((n, self.m, P)); dsV = np.zeros((n, self.m, P))
        for t in range(len(self.cache) - 1, -1, -1):
            if dlogs[t] is not None:
                gWo += self.hs[t].T @ dlogs[t]; gbo += dlogs[t].sum(0); dh = dh + dlogs[t] @ self.Wo.T
            z, c, a, hp = self.cache[t]
            dz, g = self.f_bwd(al * dh * (1 - a * a), z, c)
            for i, gi in enumerate(g):
                gF[i] += gi
            d0 = self.dx0
            dq = dz[:, d0:d0 + P]; dr = dz[:, d0 + P:d0 + 2 * P]
            dhp = dz[:, d0 + 2 * P:] * (0.0 if self.broken else 1.0) + (1 - al) * dh
            Ap, A = self.As[t]
            ptr = self.F_[t]
            if ptr is not None:
                q = sK[ptr]
                dA = dA + np.einsum('ni,nj->nij', dr, q)
                dq = dq + np.einsum('nij,ni->nj', A, dr)
                np.add.at(dsK, (np.arange(n), ptr), dq)
            W, fa = self.Ws[t]
            if fa is not None:
                src, tgt = fa
                dW = U * dA
                glam_t = None
                np.add.at(dsV, (np.arange(n)[:, None], tgt), np.einsum('nij,nfj->nfi', dW, sK[src]))
                np.add.at(dsK, (np.arange(n)[:, None], src), np.einsum('nij,nfi->nfj', dW, sV[tgt]))
                gU += float((dA * W).sum())
            glam += float((dA * Ap).sum())
            dA = dA * (0.0 if self.broken else lam)
            dh = dhp
        gEk = (dsK.sum(0)) * sK * (1 - sK)
        gEv = (dsV.sum(0)) * sV * (1 - sV)
        out = gF + [gWo, gbo, gEk, gEv]
        if self.mode == "generic":
            l_ = sig(self.lam_l[0]); out += [np.array([glam * l_ * (1 - l_)]), np.array([gU * sig(self.U_l[0])])]
        return out


class BrokenMem(MemNet):
    broken = True


def gradcheck(mode, fmt, broken=False, seed=5):
    rng = np.random.default_rng(seed)
    m = 4; dx = ce.dx_of(m, fmt)
    net = (BrokenMem if broken else MemNet)("combined", rng, dx, m, mode, u=3)
    for p in net.a + net.mu + net.s:
        p[...] = rng.normal(0.4, 0.3, p.shape)
    X, Y = ce.batch(rng, 5, 3, m, fmt)

    def L():
        net.run(X, Y, fmt); return ce.step_losses(net, Y, True)[1]

    net.run(X, Y, fmt); dl, _ = ce.step_losses(net, Y, True); gs = net.backward(dl)
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
    ver, mode, seed = args
    V = VERS[ver]; m, fmt = V["m"], V["fmt"]
    net = MemNet("combined", np.random.default_rng(seed), ce.dx_of(m, fmt), m, mode)
    rng = np.random.default_rng(700 + seed)
    st, t0 = {}, time.time()
    for _ in range(ce.STEPS):
        k = int(rng.choice(ce.TRAIN_K))
        X, Y = ce.batch(rng, ce.BS, k, m, fmt)
        lg = net.run(X, Y, fmt, ce.NOISE, rng)
        if not np.all(np.isfinite(lg)):
            break
        dl, _ = ce.step_losses(net, Y, True)
        ce.adam_clip(net.ps, net.backward(dl), st, ce.LR)

    def pred(X):
        lg = net.run(X, None, fmt)
        return np.where(np.all(np.isfinite(lg), 1), lg.argmax(1), -1)

    acc = {k_: ce.evaluate(pred, k_, seed, m, fmt) for k_ in V["test"]}
    lam, U = net.lamU()
    return dict(version=ver, mode=mode, seed=seed, acc=acc, params=ce.nparams(net), lam=lam, U=U,
                ind=float(np.mean([acc[k_] for k_ in ce.TRAIN_K])), seconds=round(time.time() - t0, 1))


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    P_(RULE); P_("PART A -- BLUE BRAIN SYNAPSES AND OPTOGENETIC RECORDINGS -> THE MEMORY'S NUMBERS"); P_(RULE)
    bb = bbp_numbers()
    for cls, v in bb.items():
        P_(f"  {cls:<26} pathways {v['n']:>5}   U {v['U']:.3f} (w {v['U_w']:.3f})   tau_F {v['tau_F_ms']:>6.0f} ms "
           f"(w {v['tau_F_w']:.0f})   tau_D {v['tau_D_ms']:>6.0f} ms (w {v['tau_D_w']:.0f})")
    fac = bb.get("Excitatory, facilitating")
    a1 = fac is not None and fac["n"] >= 5
    P_(f"  A1 excitatory facilitating pathways >= 5: {'PASS' if a1 else 'FAIL'}")
    al = alm_timescale(np.random.default_rng(0))
    a2 = al["r2"] >= 0.5 and 0.02 <= al["tau"] <= 10
    P_(f"  A2 ALM intrinsic timescale: tau {al['tau']:.3f} s  95% CI [{al['ci'][0]:.3f}, {al['ci'][1]:.3f}]  R^2 {al['r2']:.3f}  "
       f"units {al['units']}  -> {'PASS' if a2 else 'FAIL'}")
    P_("     population autocorrelation by lag: " + "  ".join(f"{l:.1f}:{c:.3f}" for l, c in zip(al["lags"][:8], al["curve"][:8])))
    if not (a1 and a2):
        open(OUT, "w").write("\n".join(out) + "\n"); return
    BIO["lambda"] = math.exp(-DT / (fac["tau_F_ms"] / 1000)); BIO["U"] = fac["U"]
    BIO["alpha"] = 1 - math.exp(-DT / al["tau"])
    P_(f"  TRANSLATED: lambda = exp(-0.1/{fac['tau_F_ms'] / 1000:.3f}) = {BIO['lambda']:.4f} per step;  write gain U = {BIO['U']:.3f};  "
       f"alpha = 1 - exp(-0.1/{al['tau']:.3f}) = {BIO['alpha']:.4f}")

    P_("\n" + RULE); P_("PART B -- THE MEMORY IN THE CHAIN NETWORK"); P_(RULE)
    ok = True
    for mode in ("bio", "generic"):
        for fmt in ("every", "once"):
            g, bg = gradcheck(mode, fmt), gradcheck(mode, fmt, broken=True)
            ok &= g <= 1.0 and bg > 1.0
            P_(f"  Q0 {mode:<8} {fmt:<5} gradient/tol {g:.3f}   broken {bg:.2e} ({'caught' if bg > 1 else 'NOT CAUGHT'})")
    if not ok:
        P_("  BLOCKED."); open(OUT, "w").write("\n".join(out) + "\n"); return
    jobs = [(v, md, s) for v in VERS for md in ("bio", "generic") for s in VERS[v]["seeds"]]
    jobs.sort(key=lambda j: (j[0] != "H1", j[0] != "H2"))
    pool = mp.get_context("fork").Pool(4)
    runs = list(pool.imap_unordered(job, jobs, chunksize=1))
    pool.close()
    sn = json.load(open(R("outputs", "chainsnap.json")))["runs"]
    res = {}
    for ver, V in VERS.items():
        P_("\n" + RULE); P_({"P1": "P1 -- 6 items, facts every step", "H1": "H1 -- 12 items", "H2": "H2 -- 6 items, FACTS SHOWN ONCE"}[ver]
                            + "; trained on 1-3 hops; same seeds and questions as chainsnap"); P_(RULE)
        P_("    " + f"{'':<26}" + "".join(f"{'k=' + str(k_):>7}" for k_ in V["test"]) + "  learnt exact")
        shared = [k_ for k_ in V["test"] if k_ > 3 and k_ != 64]
        U_ = {}
        for md, nm in (("bio", "BIO memory"), ("generic", "GENERIC memory (learnt)")):
            rs = sorted([r for r in runs if r["version"] == ver and r["mode"] == md], key=lambda r: r["seed"])
            lt = sum(r["ind"] >= 0.9 for r in rs); ex = sum(all(r["acc"][e] >= 0.99 for e in V["exact"]) for r in rs)
            P_(f"    {nm:<26}" + "".join(f"{np.mean([r['acc'][k_] for r in rs]):>7.3f}" for k_ in V["test"])
               + f"  {lt:>2}/10 {ex:>2}/10" + ("  EXACT" if ex >= 9 else ""))
            if md == "generic":
                P_(f"      learnt lambda {np.median([r['lam'] for r in rs]):.3f} (bio {BIO['lambda']:.3f}), U {np.median([r['U'] for r in rs]):.3f} (bio {BIO['U']:.3f})")
            U_[md] = {r["seed"]: float(np.mean([r["acc"][k_] for k_ in shared])) for r in rs}
            res[(ver, md)] = dict(learnt=lt, exact=ex)
        nomem = {x["seed"]: x for x in sn if x["version"] == ver and x["kind"] == "combined"}
        P_(f"    {'no memory (chainsnap)':<26}" + "".join(f"{np.mean([nomem[s]['acc'][str(k_)] for s in V['seeds']]):>7.3f}" for k_ in V["test"]))
        P_(f"    {'transformer + CoT (means)':<26}" + "".join(f"{TRANSFORMER_COT[ver].get(k_, float('nan')):>7.3f}" if k_ in TRANSFORMER_COT[ver] else f"{'':>7}" for k_ in V["test"]))
        U_["nomem"] = {s: float(np.mean([nomem[s]["acc"][str(k_)] for k_ in shared])) for s in V["seeds"]}
        for a_, b_, nm in (("bio", "nomem", "bio memory vs no memory"), ("bio", "generic", "BIO vs GENERIC")):
            d = [U_[a_][s] - U_[b_][s] for s in V["seeds"]]
            w = sum(x > 0 for x in d); l = sum(x < 0 for x in d)
            v = ("BIO BETTER" if w >= 9 else ("NO-MEMORY BETTER" if b_ == "nomem" else "GENERIC BETTER") if l >= 9 else "NO DIFFERENCE")
            res[(ver, a_ + "_vs_" + b_)] = dict(wins=w, losses=l, mean=float(np.mean(d)), verdict=v)
            P_(f"  {nm:<26} bio higher on {w}/10, lower on {l}/10, mean {np.mean(d):+.3f} -> {v}")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"bbp": bb, "alm": {k: v for k, v in al.items()}, "bio": BIO,
               "results": {"|".join(k): v for k, v in res.items()},
               "runs": [{**r, "acc": {str(k_): float(v) for k_, v in r["acc"].items()}} for r in runs]},
              open(ART, "w"), indent=1, default=float)
    P_(f"\n  artifact: outputs/biomemory.json   runtime {time.time() - t0:.0f}s")
    P_("  Licences: Blue Brain NMC portal data (EPFL, all rights reserved; no open licence found) and DANDI 000060")
    P_("  (draft, none declared) -- aggregates only; raw files stay in the gitignored cache.")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
