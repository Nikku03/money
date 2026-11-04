"""A hand-wired spiking circuit that plays Pong.

The idea comes from DishBrain (Kagan et al., Neuron 2022), where cultured neurons on an electrode array
learned to play Pong from stimulation alone. Here nothing learns: every weight, delay and connection is
assigned by a rule written below, so the circuit plays from its first frame, and what it does can be
read off its wiring.

    eyes        8 BALL cells and 8 PADDLE cells, each tuned to one band of the screen's height.
                Every frame (20 ms) an electrode pulse drives each cell in proportion to how close the
                ball (or the paddle) is to its band, so the nearest cells fire, all at the frame's start.
    compare     64 coincidence cells, one per (ball band i, paddle band j). Each gets 0.08 from BALL_i and
                0.08 from PADDLE_j; one input alone stays under threshold (about 0.135), both together fire
                it. Spike timing is the comparison: only cells whose two inputs arrive together fire.
    decide      C_ij with i > j means "ball above paddle" and feeds the UP clique's source; i < j feeds
                DOWN. C_ii (aligned) feeds the HOLD interneuron, which inhibits both cliques.
    amplify     UP and DOWN are each a directed 11-simplex: 12 neurons, neuron j feeding every k > j
                (66 connections). Neuron k's weights split 0.3 evenly over its k inputs, and each delay
                is (k - j - 1) * 2.5 ms + 1 ms, so every predecessor's spike lands on k at the same moment.
                The volley is strong enough to survive lost members, and noise that is not one
                coherent volley dies out. The sink (neuron 11) is the motor output.
    choose      each clique's source drives an inhibitory cell that silences the other clique's source, so
                the two commands race and the first to start wins (winner-take-all); a volley already
                in flight is left to finish.
    move        each UP sink spike moves the paddle up one step; each DOWN sink spike, down one step.

Signals take about 2 frames from eye to paddle (the clique's volley takes ~30 ms), so the circuit
reacts to where the ball was, as any real nervous system does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.neurons.circuit import Circuit

BANDS = 8                  # place cells per eye
FRAME_MS = 20.0            # neural time per game frame
PULSE_MS = 1.0             # electrode pulse length at the frame's start
PULSE_AMP = 60.0           # pulse current to a cell centred on the stimulus
TUNING = 0.45              # tuning width, in bands
EYE_W = 0.08               # each eye's weight onto a coincidence cell (one alone < threshold < two)
DECIDE_W = 0.2             # coincidence cell -> clique source
HOLD_W = 0.2               # coincidence (aligned) -> HOLD
HOLD_INH = 0.5             # HOLD -> both cliques' sources
CLIQUE = 12                # neurons per directed simplex: dimension 11
CLIQUE_W = 0.3             # total weight onto each clique neuron, split over its inputs
STAGE_MS = 2.5             # time between successive clique neurons' spikes
WTA_W = 0.3                # source -> its inhibitory cell
WTA_INH = 1.0              # inhibitory cell -> the other clique's source
PADDLE_H = 0.25            # paddle height, as a fraction of the screen
STEP = 1.0 / 32            # paddle movement per motor spike
MAX_STEPS = 1              # the paddle's top speed, in steps per frame (a muscle saturates)


def build() -> Circuit:
    c = Circuit()
    ball = [c.add(f"BALL{i}", group="ball") for i in range(BANDS)]
    pad = [c.add(f"PADDLE{j}", group="paddle") for j in range(BANDS)]
    add_motor(c, ball, pad)
    return c


def add_motor(c: Circuit, target: list[int], pad: list[int], pad_delay: float = 1.0) -> None:
    """Compare where the paddle should be (``target``, one cell per band) with where it is (``pad``),
    and drive the paddle through the UP and DOWN cliques. ``pad_delay`` is the paddle cells' delay onto
    the coincidence cells: set it so their spikes arrive with the target cells' spikes."""
    hold = c.add("HOLD", "FS", group="hold", sign=-1)
    up = [c.add(f"UP{k}", "FE", group="up") for k in range(CLIQUE)]
    dn = [c.add(f"DOWN{k}", "FE", group="down") for k in range(CLIQUE)]
    for q in (up, dn):
        for k in range(1, CLIQUE):
            for j in range(k):
                c.connect(q[j], q[k], CLIQUE_W / k, (k - j - 1) * STAGE_MS + 1.0)
    inh_up = c.add("INH_UP", "FS", group="wta", sign=-1)      # fired by UP, silences DOWN
    inh_dn = c.add("INH_DOWN", "FS", group="wta", sign=-1)
    c.connect(up[0], inh_up, WTA_W, 0.5)
    c.connect(dn[0], inh_dn, WTA_W, 0.5)
    c.connect(inh_up, dn[0], WTA_INH, 0.5)
    c.connect(inh_dn, up[0], WTA_INH, 0.5)
    c.connect(hold, up[0], HOLD_INH, 0.5)
    c.connect(hold, dn[0], HOLD_INH, 0.5)
    for i in range(len(target)):
        for j in range(len(pad)):
            x = c.add(f"C{i}{j}", "FE", group="compare")
            c.connect(target[i], x, EYE_W, 1.0)
            c.connect(pad[j], x, EYE_W, pad_delay)
            c.connect(x, up[0] if i > j else dn[0] if i < j else hold, HOLD_W if i == j else DECIDE_W, 1.0)


def place_drive(y: float, bands: int = BANDS) -> np.ndarray:
    """Electrode current per band for a stimulus at height y in [0, 1] (0 = bottom)."""
    centres = (np.arange(bands) + 0.5) / bands
    d = (y - centres) * bands
    return PULSE_AMP * np.exp(-d * d / (2 * TUNING * TUNING))


@dataclass
class Game:
    """Pong on a unit square: the paddle on the right edge, walls elsewhere."""
    seed: int = 0
    speed: float = 0.02                # ball speed per frame
    bx: float = 0.5
    by: float = 0.5
    vx: float = 0.0
    vy: float = 0.0
    py: float = 0.5
    hits: int = 0
    misses: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.serve()

    def serve(self):
        self.bx, self.by = 0.1, float(self.rng.uniform(0.1, 0.9))
        ang = self.rng.uniform(-0.9, 0.9)          # radians from horizontal
        self.vx, self.vy = self.speed * np.cos(ang), self.speed * np.sin(ang)

    def move_paddle(self, dy: float):
        self.py = float(np.clip(self.py + dy, PADDLE_H / 2, 1 - PADDLE_H / 2))

    def tick(self) -> str | None:
        self.bx += self.vx
        self.by += self.vy
        if self.by < 0 or self.by > 1:
            self.by = -self.by if self.by < 0 else 2 - self.by
            self.vy = -self.vy
        if self.bx < 0:
            self.bx, self.vx = -self.bx, -self.vx
        if self.bx >= 1:
            if abs(self.by - self.py) <= PADDLE_H / 2:
                self.hits += 1
                self.bx, self.vx = 2 - self.bx, -self.vx
                self.vy += float(self.rng.uniform(-0.004, 0.004))   # a little spin, so rallies differ
                return "hit"
            self.misses += 1
            self.serve()
            return "miss"
        return None


def play(frames: int = 1500, seed: int = 0, noise: float = 0.0, lesion: float = 0.0,
         kill: tuple[str, ...] = (), record: bool = False, controller: str = "neurons") -> dict:
    """Play ``frames`` frames. ``lesion`` is the fraction of clique neurons removed at random (never a
    source or a sink); ``kill`` names neurons to remove; ``controller`` "neurons" plays with the
    circuit, "still" never moves, "random" moves at random.
    Returns the score and, with ``record``, a frame-by-frame replay with every spike and some voltages."""
    c = build()
    sim = c.compile(seed)
    sim.noise = noise
    rng = np.random.default_rng(seed + 1)
    up, dn = c.group("up"), c.group("down")
    middle = up[1:-1] + dn[1:-1]
    if lesion:
        for n in rng.choice(middle, int(round(lesion * len(middle))), replace=False):
            sim.alive[n] = False
    for name in kill:
        sim.alive[c.by_name(name)] = False
    ball, pad = np.array(c.group("ball")), np.array(c.group("paddle"))
    up_sink, dn_sink = up[-1], dn[-1]
    fixed = [up[0], up[5], up_sink, dn[0], dn_sink]       # traced every frame; the first two
    # traces follow the ball: the eye cell nearest it, and the coincidence cell for (ball, paddle) bands
    g = Game(seed=seed)
    steps = int(FRAME_MS / sim.dt)
    pulse = int(PULSE_MS / sim.dt)
    N = len(c.neurons)
    replay = []
    total_spikes = 0
    for f in range(frames):
        drive = np.zeros(N)
        drive[ball] = place_drive(g.by)
        drive[pad] = place_drive(g.py)
        ups = dns = 0
        spikes, volts = [], []
        bi, pj = min(BANDS - 1, int(g.by * BANDS)), min(BANDS - 1, int(g.py * BANDS))
        traced = [ball[bi], c.by_name(f"C{bi}{pj}")] + fixed
        for k in range(steps):
            fired = sim.step(drive if k < pulse else None)
            total_spikes += len(fired)
            ups += int(up_sink in fired)
            dns += int(dn_sink in fired)
            if record:
                spikes += [(round(sim.t, 2), int(n)) for n in fired]
                if k % 2 == 0:
                    volts.append([round(float(sim.v_peak[n]), 1) for n in traced])
        if controller == "neurons":
            g.move_paddle(STEP * max(-MAX_STEPS, min(MAX_STEPS, ups - dns)))
        elif controller == "random":
            g.move_paddle(STEP * int(rng.integers(-1, 2)))
        event = g.tick()
        if record:
            replay.append({"bx": round(g.bx, 4), "by": round(g.by, 4), "py": round(g.py, 4),
                           "up": ups, "down": dns, "event": event, "spikes": spikes, "v": volts,
                           "traced": [c.neurons[n].name for n in traced[:2]]})
    out = {"frames": frames, "hits": g.hits, "misses": g.misses,
           "hit_rate": g.hits / max(1, g.hits + g.misses), "spikes": total_spikes,
           "neurons": N, "synapses": len(c.synapses)}
    if record:
        out["replay"] = replay
        out["traced"] = ["eye cell nearest the ball", "its coincidence cell"] + [c.neurons[n].name for n in fixed]
        out["names"] = [n.name for n in c.neurons]
        out["groups"] = [n.group for n in c.neurons]
        out["dead"] = [int(i) for i in np.flatnonzero(~sim.alive)]
        out["synapse_list"] = [[s.pre, s.post, s.weight, s.delay] for s in c.synapses]
        out["signs"] = [n.sign for n in c.neurons]
    return out
