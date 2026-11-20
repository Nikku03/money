"""A spiking circuit: Izhikevich neurons, conductance synapses, axonal delays.

Each neuron carries two state variables through time: its membrane voltage ``v`` (mV) and a recovery
variable ``u``. Between spikes they follow

    dv/dt = 0.04 v^2 + 5 v + 140 - u + I        du/dt = a (b v - u)

and when ``v`` reaches +30 mV the neuron fires: ``v`` resets to ``c`` and ``u`` jumps by ``d``.
The input current ``I`` is a bias plus what the synapses deliver: each synapse opens a conductance in
its target, excitatory (reversal 0 mV, decay 5 ms) or inhibitory (reversal -80 mV, decay 10 ms), so the
same spike pushes a resting cell harder than one already near threshold. A spike reaches each target
only after that synapse's own delay, which is what makes the timing of spikes, not just their count,
carry the computation.

Every connection is explicit: ``inputs(n)`` lists who feeds neuron ``n``, with weight and delay, and
``outputs(n)`` who it feeds.

The same circuit can run on three kinds of cell (``compile(cells=...)``), the ladder of internal
complexity in He et al. (2024, Nat Comput Sci 4:584, doi:10.1038/s43588-024-00674-9):

    "lif"   leaky integrate-and-fire: 1 state variable. Voltage leaks to rest; at a fixed threshold the
            cell fires, resets and rests for a refractory period.
    "izh"   Izhikevich (the default, as every recorded result used): 2 state variables, voltage and
            recovery. It reproduces how real cells fire, not why.
    "hh"    Hodgkin-Huxley (1952): 4 state variables, voltage and the sodium (m, h) and potassium (n)
            channel gates, from the squid giant axon's measured physics. Nothing is reset by hand: the
            spike, the threshold and the refractory period come out of the channels.

Synapses, delays, electrode pulses and noise are shared. Each cell kind has two calibrated gains
(``GAINS``), so the same wiring means the same thing: a synaptic gain, set so that the weight of one
synaptic event that just fires a resting cell is the same as for the Izhikevich FE cell, and a current
gain, set so that a 1 ms electrode pulse that just fires it is the same as for the Izhikevich RS cell.
``calibrate()`` recomputes them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Izhikevich (2003) parameter sets: (a, b, c, d).
KINDS = {
    "RS": (0.02, 0.2, -65.0, 8.0),    # regular spiking: excitatory pyramidal cell
    "FS": (0.10, 0.2, -65.0, 2.0),    # fast spiking: inhibitory interneuron
    "CH": (0.02, 0.2, -50.0, 2.0),    # chattering: bursts
    "FE": (0.10, 0.2, -65.0, 2.0),    # fast-recovering excitatory: FS dynamics, excitatory sign
}
E_EXC, E_INH = 0.0, -80.0     # reversal potentials (mV)
TAU_EXC, TAU_INH = 5.0, 10.0  # conductance decay (ms)
V_PEAK = 30.0

CELLS = ("izh", "hh", "lif")
STATE_VARIABLES = {"lif": 1, "izh": 2, "hh": 4}

# Hodgkin & Huxley (1952), squid giant axon at 6.3 C, voltage shifted so rest is -65 mV.
HH_C, HH_GNA, HH_GK, HH_GL = 1.0, 120.0, 36.0, 0.3          # uF/cm2, mS/cm2
HH_ENA, HH_EK, HH_EL = 50.0, -77.0, -54.387                  # mV
HH_SUBSTEPS = 5                                              # 0.05 ms steps inside each 0.25 ms step
HH_SPIKE = 0.0                                               # a spike is an upward crossing of 0 mV

# Leaky integrate-and-fire. Its membrane time constant matches the Izhikevich cells' near rest
# (about 1.25 ms: the slope of 0.04 v^2 + 5 v + 140 - b v at the resting -70 mV), so the two differ in
# what the cell does at threshold, not in how fast it forgets.
LIF_TAU, LIF_REST, LIF_THRESH, LIF_RESET, LIF_REF = 1.25, -70.0, -50.0, -65.0, 1.0   # ms, mV, mV, mV, ms

# (synaptic gain, current gain) per cell kind, from calibrate(). Izhikevich is the reference.
GAINS = {"izh": (1.0, 1.0), "hh": (0.4705, 0.3642), "lif": (4.957, 1.910)}


@dataclass
class Neuron:
    id: int
    name: str
    kind: str = "RS"
    group: str = ""
    sign: int = +1            # +1 excitatory, -1 inhibitory (Dale's law: one sign per neuron)


@dataclass
class Synapse:
    pre: int
    post: int
    weight: float             # conductance added to the target per spike (dimensionless, >= 0)
    delay: float              # ms from the presynaptic spike to its arrival


@dataclass
class Circuit:
    dt: float = 0.25          # ms per step
    neurons: list[Neuron] = field(default_factory=list)
    synapses: list[Synapse] = field(default_factory=list)

    # --- building --------------------------------------------------------------------------------
    def add(self, name: str, kind: str = "RS", group: str = "", sign: int = +1) -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown neuron kind {kind!r}")
        n = Neuron(len(self.neurons), name, kind, group, sign)
        self.neurons.append(n)
        return n.id

    def connect(self, pre: int, post: int, weight: float, delay: float = 1.0) -> None:
        if pre == post:
            raise ValueError("no autapses: a neuron does not feed itself")
        if weight < 0:
            raise ValueError("weights are non-negative; the sign is the presynaptic neuron's (Dale's law)")
        if delay < self.dt:
            raise ValueError(f"delay {delay} ms is shorter than one step ({self.dt} ms)")
        self.synapses.append(Synapse(pre, post, float(weight), float(delay)))

    def group(self, name: str) -> list[int]:
        return [n.id for n in self.neurons if n.group == name]

    def by_name(self, name: str) -> int:
        for n in self.neurons:
            if n.name == name:
                return n.id
        raise KeyError(name)

    # --- the wiring, as a reader asks about it ---------------------------------------------------
    def inputs(self, n: int) -> list[Synapse]:
        """Who gives neuron ``n`` its input: every synapse ending at ``n``."""
        return [s for s in self.synapses if s.post == n]

    def outputs(self, n: int) -> list[Synapse]:
        """Whom neuron ``n`` feeds: every synapse starting at ``n``."""
        return [s for s in self.synapses if s.pre == n]

    def adjacency(self) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {n.id: set() for n in self.neurons}
        for s in self.synapses:
            out[s.pre].add(s.post)
        return out

    # --- running ---------------------------------------------------------------------------------
    def compile(self, seed: int = 0, cells: str = "izh") -> "Sim":
        return Sim(self, seed, cells)


def _hh_rates(v):
    """The HH opening (alpha) and closing (beta) rates per ms of the m, h and n gates at voltage v."""
    x = v + 40.0
    am = np.where(np.abs(x) < 1e-6, 1.0, 0.1 * x / -np.expm1(-x / 10.0))
    bm = 4.0 * np.exp(-(v + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(v + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
    y = v + 55.0
    an = np.where(np.abs(y) < 1e-6, 0.1, 0.01 * y / -np.expm1(-y / 10.0))
    bn = 0.125 * np.exp(-(v + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


class Sim:
    """The running state of a circuit. ``step(bias)`` advances one ``dt``; ``run`` advances many."""

    def __init__(self, c: Circuit, seed: int = 0, cells: str = "izh"):
        if cells not in CELLS:
            raise ValueError(f"unknown cells {cells!r}; one of {CELLS}")
        self.c, self.dt, self.cells = c, c.dt, cells
        self.gsyn, self.gcur = GAINS[cells]
        N = len(c.neurons)
        p = np.array([KINDS[n.kind] for n in c.neurons]).reshape(N, 4)
        self.a, self.b, self.cr, self.d = p.T
        self.sign = np.array([n.sign for n in c.neurons])
        self.v = np.full(N, -65.0)
        self.u = self.b * self.v
        self.ge = np.zeros(N)
        self.gi = np.zeros(N)
        self.noise = 0.0                       # sd of the input current noise, per sqrt(ms)
        self.alive = np.ones(N, bool)          # a lesioned neuron never fires
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.k = 0
        steps = [max(1, int(round(s.delay / self.dt))) for s in c.synapses]
        self.D = (max(steps) if steps else 1) + 1
        self.bufe = np.zeros((self.D, N))      # conductance waiting to arrive, by step and target
        self.bufi = np.zeros((self.D, N))
        # Per presynaptic neuron: its targets, weights and delays in steps, as arrays.
        self.out = []
        for i in range(N):
            ss = [(s.post, s.weight, st) for s, st in zip(c.synapses, steps) if s.pre == i]
            self.out.append((np.array([x[0] for x in ss]), np.array([x[1] for x in ss], float),
                             np.array([x[2] for x in ss])) if ss else None)
        self.de, self.di = np.exp(-self.dt / TAU_EXC), np.exp(-self.dt / TAU_INH)
        if cells == "hh":
            self.v = np.full(N, -65.0)
            am, bm, ah, bh, an, bn = _hh_rates(self.v)
            self.m, self.h, self.n = am / (am + bm), ah / (ah + bh), an / (an + bn)
            self._rest = (self.m.copy(), self.h.copy(), self.n.copy())
        elif cells == "lif":
            self.v = np.full(N, LIF_REST)
            self.ref = np.zeros(N)                 # refractory time left (ms)

    def set_weights(self, pre: int, weights: np.ndarray) -> None:
        """Replace neuron ``pre``'s outgoing weights, in the order its synapses were connected: the
        one way a running circuit changes (plasticity). Its targets and delays stay as wired."""
        o = self.out[pre]
        if o is None or len(o[1]) != len(weights):
            raise ValueError(f"neuron {pre} has {0 if o is None else len(o[1])} synapses, not {len(weights)}")
        if np.any(np.asarray(weights) < 0):
            raise ValueError("weights are non-negative")
        o[1][:] = weights

    def step(self, bias: np.ndarray | None = None, ge_in: np.ndarray | None = None,
             gi_in: np.ndarray | None = None) -> np.ndarray:
        """Advance one dt with the given bias current per neuron; return the ids that fired.
        ``ge_in`` and ``gi_in`` are conductance arriving now from outside the circuit (synapses from
        input fibres), excitatory and inhibitory, per neuron."""
        slot = self.k % self.D
        self.ge = self.ge * self.de + self.bufe[slot]
        self.gi = self.gi * self.di + self.bufi[slot]
        if ge_in is not None:
            self.ge = self.ge + ge_in
        if gi_in is not None:
            self.gi = self.gi + gi_in
        self.bufe[slot] = 0.0
        self.bufi[slot] = 0.0
        if self.cells == "izh":
            fired = self._step_izh(bias)
        else:
            ext = np.zeros(len(self.v)) if bias is None else np.asarray(bias, float)
            if self.noise:
                ext = ext + self.noise * self.rng.standard_normal(self.v.shape) / np.sqrt(self.dt)
            ext = self.gcur * ext
            ge, gi = self.gsyn * self.ge, self.gsyn * self.gi
            fired = self._step_hh(ge, gi, ext) if self.cells == "hh" else self._step_lif(ge, gi, ext)
        for i in fired:
            o = self.out[i]
            if o is None:
                continue
            post, w, st = o
            slots = (self.k + st) % self.D
            np.add.at(self.bufe if self.sign[i] > 0 else self.bufi, (slots, post.astype(int)), w)
        self.k += 1
        self.t = self.k * self.dt
        return fired

    def _step_izh(self, bias):
        v = self.v
        I = self.ge * (E_EXC - v) + self.gi * (E_INH - v)
        if bias is not None:
            I = I + bias
        if self.noise:
            I = I + self.noise * self.rng.standard_normal(v.shape) / np.sqrt(self.dt)
        # Two half steps on v for stability (as in Izhikevich's own code), one on u.
        for _ in range(2):
            v = v + 0.5 * self.dt * (0.04 * v * v + 5 * v + 140 - self.u + I)
        self.u = self.u + self.dt * self.a * (self.b * v - self.u)
        v = np.where(self.alive, v, -65.0)
        fired = np.flatnonzero(v >= V_PEAK)
        self.v_peak = v.copy()                 # with the spike's peak shown, for voltage traces
        self.v_peak[fired] = V_PEAK
        v[fired] = self.cr[fired]
        self.u[fired] += self.d[fired]
        self.v = v
        return fired

    def _step_hh(self, ge, gi, ext):
        """Exponential Euler on every variable (stable at 0.05 ms): each gate relaxes toward its
        steady state, and the voltage toward the conductance-weighted mean of the reversal potentials."""
        h_ = self.dt / HH_SUBSTEPS
        v, m, h, n = self.v, self.m, self.h, self.n
        crossed = np.zeros(len(v), bool)
        vmax = v.copy()
        for _ in range(HH_SUBSTEPS):
            am, bm, ah, bh, an, bn = _hh_rates(v)
            for x, a, b in ((m, am, bm), (h, ah, bh), (n, an, bn)):
                tot = a + b
                inf = a / tot
                x[:] = inf + (x - inf) * np.exp(-h_ * tot)
            gna, gk = HH_GNA * m ** 3 * h, HH_GK * n ** 4
            G = gna + gk + HH_GL + ge + gi
            E = (gna * HH_ENA + gk * HH_EK + HH_GL * HH_EL + ge * E_EXC + gi * E_INH + ext) / G
            nv = E + (v - E) * np.exp(-h_ * G / HH_C)
            crossed |= (v < HH_SPIKE) & (nv >= HH_SPIKE)
            v = nv
            vmax = np.maximum(vmax, v)
        dead = ~self.alive
        if dead.any():
            v[dead] = -65.0
            m[dead], h[dead], n[dead] = (r[dead] for r in self._rest)
        self.v = v
        self.v_peak = np.where(self.alive, vmax, -65.0)
        return np.flatnonzero(crossed & self.alive)

    def _step_lif(self, ge, gi, ext):
        """Exact for inputs held over the step: the voltage relaxes toward the conductance-weighted
        mean of rest and the reversal potentials, plus the injected current."""
        v = self.v
        G = 1.0 + ge + gi
        E = (LIF_REST + ge * E_EXC + gi * E_INH + ext) / G
        v = E + (v - E) * np.exp(-self.dt * G / LIF_TAU)
        resting = self.ref > 0
        v[resting] = LIF_RESET
        self.ref = np.maximum(0.0, self.ref - self.dt)
        v = np.where(self.alive, v, LIF_REST)
        fired = np.flatnonzero(v >= LIF_THRESH)
        self.v_peak = v.copy()
        self.v_peak[fired] = V_PEAK
        v[fired] = LIF_RESET
        self.ref[fired] = LIF_REF
        self.v = v
        return fired


# --- calibration: the same wiring means the same thing on every cell kind -----------------------------
def _one_cell(kind: str, cells: str) -> "Sim":
    c = Circuit()
    c.add("SRC", "RS")
    c.add("X", kind)
    c.connect(0, 1, 0.0, 1.0)
    return c.compile(0, cells)


def fires_on_event(weight: float, cells: str, kind: str = "FE", ms: float = 30.0) -> bool:
    """Does one excitatory synaptic event of ``weight`` fire a resting cell?"""
    s = _one_cell(kind, cells)
    for _ in range(int(20 / s.dt)):            # settle
        s.step()
    s.ge[1] += weight
    for _ in range(int(ms / s.dt)):
        if 1 in s.step():
            return True
    return False


def fires_on_pulse(amp: float, cells: str, kind: str = "RS", pulse_ms: float = 1.0, ms: float = 30.0) -> bool:
    """Does a ``pulse_ms`` electrode pulse of ``amp`` fire a resting cell?"""
    s = _one_cell(kind, cells)
    for _ in range(int(20 / s.dt)):
        s.step()
    b = np.array([0.0, amp])
    for k in range(int(ms / s.dt)):
        if 1 in s.step(b if k < int(pulse_ms / s.dt) else None):
            return True
    return False


def threshold(test, lo: float, hi: float, tol: float = 1e-4) -> float:
    """The smallest x in [lo, hi] with ``test(x)`` true, by bisection (test must be monotone)."""
    if not test(hi):
        return float("nan")
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if test(mid) else (mid, hi)
    return hi


def calibrate() -> dict:
    """Recompute ``GAINS``: for each cell kind, the synaptic gain that gives it the Izhikevich FE cell's
    single-event threshold, and the current gain that gives it the RS cell's 1 ms pulse threshold."""
    saved = dict(GAINS)
    try:
        GAINS["izh"] = (1.0, 1.0)
        w_ref = threshold(lambda w: fires_on_event(w, "izh"), 0.0, 2.0)
        a_ref = threshold(lambda a: fires_on_pulse(a, "izh"), 0.0, 400.0)
        out = {"izh": (1.0, 1.0), "reference": {"event": w_ref, "pulse": a_ref}}
        for cells in ("hh", "lif"):
            GAINS[cells] = (1.0, 1.0)
            w = threshold(lambda w: fires_on_event(w, cells), 0.0, 50.0)
            a = threshold(lambda a: fires_on_pulse(a, cells), 0.0, 4000.0)
            out[cells] = (w / w_ref, a / a_ref)
        return out
    finally:
        GAINS.clear()
        GAINS.update(saved)
