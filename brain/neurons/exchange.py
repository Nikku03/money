"""The exchange rate: how many simple cells does one Hodgkin-Huxley cell's behaviour cost?

He et al. (2024, Nat Comput Sci 4:584, doi:10.1038/s43588-024-00674-9) claim that a network of simple
leaky integrate-and-fire (LIF) cells has to grow to reproduce what one Hodgkin-Huxley (HH) cell does.
This measures that on our own simulator, with our own construction (not theirs):

1. **The target.** One HH cell (``circuit``'s integrator) driven by a fluctuating current: noise with a
   3 ms memory around a mean that jumps every 250 ms. Its spike train is what the others must copy.
2. **The candidates.** A small LIF circuit of k cells (k = 1, 2, 4, 8): one output cell and k - 1 helpers.
   Every cell hears the input with its own gain and sign. Each helper feeds the output through its own
   synapse, and each output spike kicks every helper, so helpers can act as memory, adaptation or a
   rebound detector. And one Izhikevich cell, the cell our circuits use. Every parameter is fitted.
3. **The fit.** The cross-entropy method: sample 128 parameter sets, keep the best 16, refit, 80 times;
   the best of 3 such fits. Every model may also delay its output by up to 3 ms (an axon). Fitted on
   4 s of input, scored on 4 s of fresh input it never saw.
4. **The score.** The coincidence factor Gamma (Kistler et al. 1997; Jolivet et al. 2008): the share
   of the target's spikes matched within 2 ms, beyond what chance matching at the same rate gives. 1 is
   a perfect copy and 0 is chance.
5. **The properties.** Two things HH does by its physics: it stays silent with no input, and it fires
   when released from inhibition (rebound). Each fitted model is checked for both. (``onset_rate`` also
   records where steady firing starts and how fast; with input steps of 0.25 it cannot separate a steep
   type I onset from a type II jump, so it is recorded, not judged.)

    python -m brain neurons exchange [--seeds 3] [--json FILE]
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from brain.neurons import circuit

DT = 0.25                 # ms, as in the circuits
DELTA = 2.0               # ms: the coincidence window
TRAIN_S, TEST_S = 4.0, 4.0
POP, ELITE, ITERS, RESTARTS = 128, 16, 80, 3
SIZES = (1, 2, 4, 8)


# --- the input and the target ----------------------------------------------------------------------
def drive(seconds: float, seed: int) -> np.ndarray:
    """Input current (uA/cm2) per 0.25 ms step: an Ornstein-Uhlenbeck noise (sd 3, 3 ms memory) around a
    mean drawn from [-4, 14] every 250 ms, so the cell sees silence, inhibition, onset and steady firing."""
    rng = np.random.default_rng(seed)
    T = int(seconds * 1000 / DT)
    block = int(250 / DT)
    mean = np.repeat(rng.uniform(-4.0, 14.0, T // block + 1), block)[:T]
    a = np.exp(-DT / 3.0)
    z = rng.standard_normal(T) * 3.0 * np.sqrt(1 - a * a)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = a * x[t - 1] + z[t]
    return mean + x


def hh_spikes(I: np.ndarray) -> np.ndarray:
    """The target: one Hodgkin-Huxley cell of ``circuit`` driven by ``I`` (uA/cm2). Spike flags per step."""
    c = circuit.Circuit()
    c.add("HH")
    s = c.compile(0, "hh")
    out = np.zeros(len(I), bool)
    b = np.zeros(1)
    for t, x in enumerate(I):
        b[0] = x / s.gcur                     # the circuit scales electrode current by gcur; undo it
        out[t] = len(s.step(b)) > 0
    return out


# --- the candidates, simulated for a whole population at once ------------------------------------------
@dataclass
class Space:
    """Parameter names with (low, high, log scale); a candidate is a point in the unit cube."""
    names: list
    lo: np.ndarray
    hi: np.ndarray
    log: np.ndarray

    def decode(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 0.0, 1.0)
        lin = self.lo + u * (self.hi - self.lo)
        lg = np.exp(np.log(np.maximum(self.lo, 1e-9)) + u * (np.log(np.maximum(self.hi, 1e-9)) - np.log(np.maximum(self.lo, 1e-9))))
        return np.where(self.log, lg, lin)


def lif_space(k: int) -> Space:
    rows = [("g0", 0.05, 5.0, True), ("tau0", 0.5, 20.0, True), ("theta0", 1.0, 30.0, False),
            ("reset0", -20.0, 0.0, False), ("ref0", 0.25, 8.0, True), ("bias0", -10.0, 10.0, False)]
    for i in range(1, k):
        rows += [(f"g{i}", -5.0, 5.0, False), (f"tau{i}", 0.5, 50.0, True), (f"theta{i}", 1.0, 30.0, False),
                 (f"bias{i}", -10.0, 10.0, False), (f"w{i}", -30.0, 30.0, False), (f"taus{i}", 0.5, 30.0, True),
                 (f"f{i}", -30.0, 30.0, False)]
    n, lo, hi, lg = zip(*rows)
    return Space(list(n), np.array(lo), np.array(hi), np.array(lg))


IZH_SPACE = Space(["a", "b", "c", "d", "g", "bias"], np.array([0.005, -0.1, -70.0, 0.0, 0.05, -10.0]),
                  np.array([0.2, 0.3, -45.0, 10.0, 10.0, 10.0]), np.array([True, False, False, False, True, False]))


def sim_lif(P: np.ndarray, k: int, I: np.ndarray) -> np.ndarray:
    """Simulate a population of k-cell LIF circuits (parameters ``P``, one row each) on input ``I``.
    Cell 0 is the output. Returns its spike flags, shape (population, steps)."""
    P = np.asarray(P, float)
    n = len(P)
    g0, tau0, th0, rs0, ref0, b0 = (P[:, j] for j in range(6))
    H = k - 1
    if H:
        hp = P[:, 6:].reshape(n, H, 7)
        g, tau, th, b, w, taus, f = (hp[:, :, j] for j in range(7))
        eh, es = np.exp(-DT / tau), np.exp(-DT / taus)
        vh = np.zeros((n, H))
        s = np.zeros((n, H))
        rh = np.zeros((n, H))
    e0 = np.exp(-DT / tau0)
    v0 = np.zeros(n)
    r0 = np.zeros(n)
    out = np.zeros((n, len(I)), bool)
    for t, x in enumerate(I):
        syn = (w * s).sum(1) if H else 0.0
        v0 = v0 * e0 + (1 - e0) * (g0 * x + b0 + syn)
        v0 = np.where(r0 > 0, rs0, v0)
        r0 = np.maximum(0.0, r0 - DT)
        sp = v0 >= th0
        v0 = np.where(sp, rs0, v0)
        r0 = np.where(sp, ref0, r0)
        out[:, t] = sp
        if H:
            vh = vh * eh + (1 - eh) * (g * x + b)
            vh = np.where(rh > 0, 0.0, vh)
            rh = np.maximum(0.0, rh - DT)
            vh = vh + f * sp[:, None]               # each output spike kicks every helper
            hs = vh >= th
            vh = np.where(hs, 0.0, vh)
            rh = np.where(hs, 1.0, rh)
            s = s * es + hs
    return out


def sim_izh(P: np.ndarray, I: np.ndarray) -> np.ndarray:
    a, b, c, d, g, bias = (P[:, j] for j in range(6))
    v = np.full(len(P), -65.0)
    u = b * v
    out = np.zeros((len(P), len(I)), bool)
    for t, x in enumerate(I):
        cur = g * x + bias
        for _ in range(2):
            v = v + 0.5 * DT * (0.04 * v * v + 5 * v + 140 - u + cur)
        v = np.minimum(v, 100.0)                  # a wild candidate must not overflow
        u = u + DT * a * (b * v - u)
        sp = v >= 30.0
        v = np.where(sp, c, v)
        u = np.where(sp, u + d, u)
        out[:, t] = sp
    return out


def simulate(model: str, P: np.ndarray, I: np.ndarray) -> np.ndarray:
    """Spike flags of each candidate. The last parameter of every model is an output delay (0-3 ms, an
    axon), the same for all, so no model loses for a constant offset between its spikes and HH's."""
    core, lag = P[:, :-1], np.rint(P[:, -1] / DT).astype(int)
    out = sim_izh(core, I) if model == "izh" else sim_lif(core, int(model.split("x")[1]), I)
    for i, L in enumerate(lag):
        if L > 0:
            out[i, L:] = out[i, :-L].copy()
            out[i, :L] = False
    return out


def space(model: str) -> Space:
    sp = IZH_SPACE if model == "izh" else lif_space(int(model.split("x")[1]))
    return Space(sp.names + ["lag"], np.append(sp.lo, 0.0), np.append(sp.hi, 3.0), np.append(sp.log, False))


def state_variables(model: str) -> int:
    """Numbers each candidate carries through time: voltages, recovery, synaptic traces."""
    if model == "izh":
        return 2
    k = int(model.split("x")[1])
    return k + (k - 1)


# --- the score -----------------------------------------------------------------------------------------
def gamma(target: np.ndarray, model: np.ndarray, delta: float = DELTA) -> np.ndarray:
    """Coincidence factor of each row of ``model`` against ``target`` (spike flags per step).

    Coincidences are counted one to one, as the smaller of: target spikes with a model spike within
    ``delta``, and model spikes with a target spike within ``delta``. A model firing so fast that chance
    alone would match most spikes (2 x rate x delta >= 0.5) scores -1: it is not a copy of anything."""
    model = np.atleast_2d(model)
    T = target.size
    w = int(round(delta / DT))
    ti = np.flatnonzero(target)
    nd = len(ti)
    nm = model.sum(1)
    if nd == 0:
        return np.zeros(len(model))
    cs = np.concatenate([np.zeros((len(model), 1)), np.cumsum(model, 1)], 1)
    lo, hi = np.maximum(0, ti - w), np.minimum(T, ti + w + 1)
    hit_d = ((cs[:, hi] - cs[:, lo]) > 0).sum(1)               # target spikes matched
    ct = np.concatenate([[0], np.cumsum(target)])
    t = np.arange(T)
    near = (ct[np.minimum(T, t + w + 1)] - ct[np.maximum(0, t - w)]) > 0
    hit_m = model.astype(float) @ near                           # model spikes matched
    nc = np.minimum(hit_d, hit_m)
    nu = nm / (T * DT)                                           # model spikes per ms
    chance = 2 * nu * delta
    expected = chance * nd
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (nc - expected) / (0.5 * (nd + nm)) / (1 - chance)
    return np.where(nm == 0, 0.0, np.where(chance >= 0.5, -1.0, g))


# --- the fit ---------------------------------------------------------------------------------------
def fit(model: str, I: np.ndarray, target: np.ndarray, seed: int = 0, pop: int = POP, elite: int = ELITE,
        iters: int = ITERS, restarts: int = RESTARTS) -> tuple[np.ndarray, list]:
    """The best of ``restarts`` cross-entropy fits (by training Gamma). Returns its parameters and its
    best Gamma per iteration."""
    runs = [_cem(model, I, target, 100 * seed + r, pop, elite, iters) for r in range(restarts)]
    return max(runs, key=lambda x: x[1][-1])


def _cem(model, I, target, seed, pop, elite, iters):
    sp = space(model)
    rng = np.random.default_rng(seed)
    mu, sd = np.full(len(sp.names), 0.5), np.full(len(sp.names), 0.3)
    best, best_g, curve = None, -np.inf, []
    for _ in range(iters):
        U = np.clip(mu + sd * rng.standard_normal((pop, len(mu))), 0, 1)
        if best is not None:
            U[0] = best                              # keep the best so far in every population
        g = gamma(target, simulate(model, sp.decode(U), I))
        g = np.nan_to_num(g, nan=-1.0)
        order = np.argsort(-g)
        if g[order[0]] > best_g:
            best_g, best = float(g[order[0]]), U[order[0]].copy()
        E = U[order[:elite]]
        mu = 0.7 * E.mean(0) + 0.3 * mu
        sd = np.maximum(0.7 * E.std(0) + 0.3 * sd, 0.02)
        curve.append(best_g)
    return sp.decode(best), curve


# --- the properties --------------------------------------------------------------------------------
def onset_rate(model: str, P: np.ndarray | None) -> tuple[float, float]:
    """The lowest steady firing rate (spikes/s) the model shows as a constant input rises from 0 to 20
    uA/cm2 in steps of 0.25, and the input where firing starts. HH starts near 50 spikes/s (type II);
    a plain LIF starts near zero (type I)."""
    levels = np.arange(0.0, 20.01, 0.25)
    rates = []
    for x in levels:
        I = np.full(int(1200 / DT), x)
        s = hh_spikes(I) if model == "hh" else simulate(model, P[None, :], I)[0]
        rates.append(s[int(200 / DT):].sum() / 1.0)
    rates = np.array(rates)
    on = np.flatnonzero(rates >= 2)                # at least 2 spikes in the second
    if not len(on):
        return float("nan"), float("nan")
    return float(rates[on[0]]), float(levels[on[0]])


def rebound(model: str, P: np.ndarray | None) -> bool:
    """Does the model fire within 30 ms of release from 100 ms of inhibition (-10 uA/cm2), with no
    input before or after?"""
    I = np.concatenate([np.zeros(int(100 / DT)), np.full(int(100 / DT), -10.0), np.zeros(int(100 / DT))])
    s = hh_spikes(I) if model == "hh" else simulate(model, P[None, :], I)[0]
    quiet = not s[: int(100 / DT)].any()
    return bool(quiet and s[int(200 / DT): int(230 / DT)].any())


def cpu_per_second(model: str, P: np.ndarray | None) -> float:
    """CPU seconds to simulate one cell circuit for one second of neural time (one at a time)."""
    I = drive(1.0, 99)
    t0 = time.process_time()
    hh_spikes(I) if model == "hh" else simulate(model, P[None, :], I)
    return time.process_time() - t0


# --- the experiment ------------------------------------------------------------------------------------
def models() -> list[str]:
    return [f"lifx{k}" for k in SIZES] + ["izh"]


def _one(job: tuple) -> dict:
    model, seed = job
    Itr, Ite = drive(TRAIN_S, 1000 + seed), drive(TEST_S, 2000 + seed)
    ttr, tte = hh_spikes(Itr), hh_spikes(Ite)
    P, curve = fit(model, Itr, ttr, seed=seed)
    g_te = float(gamma(tte, simulate(model, P[None, :], Ite))[0])
    rate, at = onset_rate(model, P)
    return {"model": model, "seed": seed, "params": dict(zip(space(model).names, map(float, P))),
            "gamma_train": curve[-1], "gamma_test": g_te, "curve": curve,
            "target_spikes_test": int(tte.sum()), "model_spikes_test": int(simulate(model, P[None, :], Ite)[0].sum()),
            "onset_rate": rate, "onset_input": at, "rebound": rebound(model, P),
            "state_variables": state_variables(model), "cpu_per_s": cpu_per_second(model, P)}


def target_profile() -> dict:
    rate, at = onset_rate("hh", None)
    return {"model": "hh", "onset_rate": rate, "onset_input": at, "rebound": rebound("hh", None),
            "state_variables": 4, "cpu_per_s": cpu_per_second("hh", None)}


def run(seeds: int = 3, workers: int = 4) -> dict:
    jobs = [(m, s) for s in range(seeds) for m in models()]
    jobs.sort(key=lambda j: -state_variables(j[0]))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_one, jobs))
    return {"target": target_profile(), "fits": rows}


def table(res: dict) -> list[str]:
    t = res["target"]
    silent = lambda r: not (r["onset_input"] == 0.0)          # no steady firing at zero input
    out = [f"target: one Hodgkin-Huxley cell, 4 state variables; silent at rest: yes; rebound after inhibition:"
           f" {'yes' if t['rebound'] else 'no'}; cpu {t['cpu_per_s']:.3f} s per neural second",
           f"{'model':8s} {'cells':>5s} {'state vars':>10s}   {'Gamma on unseen input':>22s}   {'spikes vs HH':>12s}"
           f" {'silent at rest':>15s} {'rebound':>8s} {'cpu/s':>7s}"]
    for m in models():
        rs = [r for r in res["fits"] if r["model"] == m]
        if not rs:
            continue
        gs = [r["gamma_test"] for r in rs]
        cells = 1 if m == "izh" else int(m.split("x")[1])
        label = "Izh x1" if m == "izh" else f"LIF x{cells}"
        ratio = np.mean([r["model_spikes_test"] / r["target_spikes_test"] for r in rs])
        out.append(f"{label:8s} {cells:5d} {rs[0]['state_variables']:10d}   {np.mean(gs):5.2f} ("
                   + ", ".join(f"{g:.2f}" for g in gs) + ")"
                   + f"   {ratio:11.1f}x {sum(silent(r) for r in rs):>12d}/{len(rs)}"
                   + f" {sum(r['rebound'] for r in rs):>6d}/{len(rs)} {np.mean([r['cpu_per_s'] for r in rs]):7.3f}")
    return out
