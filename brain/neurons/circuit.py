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
    def compile(self, seed: int = 0) -> "Sim":
        return Sim(self, seed)


class Sim:
    """The running state of a circuit. ``step(bias)`` advances one ``dt``; ``run`` advances many."""

    def __init__(self, c: Circuit, seed: int = 0):
        self.c, self.dt = c, c.dt
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

    def set_weights(self, pre: int, weights: np.ndarray) -> None:
        """Replace neuron ``pre``'s outgoing weights, in the order its synapses were connected: the
        one way a running circuit changes (plasticity). Its targets and delays stay as wired."""
        o = self.out[pre]
        if o is None or len(o[1]) != len(weights):
            raise ValueError(f"neuron {pre} has {0 if o is None else len(o[1])} synapses, not {len(weights)}")
        if np.any(np.asarray(weights) < 0):
            raise ValueError("weights are non-negative")
        o[1][:] = weights

    def step(self, bias: np.ndarray | None = None) -> np.ndarray:
        """Advance one dt with the given bias current per neuron; return the ids that fired."""
        slot = self.k % self.D
        self.ge = self.ge * self.de + self.bufe[slot]
        self.gi = self.gi * self.di + self.bufi[slot]
        self.bufe[slot] = 0.0
        self.bufi[slot] = 0.0
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
