"""Catch: a harder game, where the circuit has to predict where the ball will land.

Each ball is served from the left edge at a random height and angle, fast, and bounces off the top and
bottom walls on its way to the paddle on the right. The paddle's top speed is half the ball's fastest
vertical speed, so a paddle that chases the ball's current height arrives late. To catch it, the circuit
has to work out where the ball will cross the right edge, bounces included, and go there early.

Nothing is trained. The prediction is wired by rule, in three stages, each a coincidence of spikes:

    eyes        16 HEIGHT cells (Y) and 8 DISTANCE cells (X), and 8 PADDLE cells. At each frame's start
                a fixed electrode pulse goes to the one or two cells whose band holds the object, so
                every eye spike has the same latency (``electrode``).
    motion      R(i, s) for height band i and shift s = -5..5 (146 cells): it hears Y_i directly and Y_(i-s) through a
                delay line of 6 frames (120 ms). Both arrive together only if the ball was at band i-s
                six frames ago and is at band i now: it is moving s/6 bands per frame. This is a
                Reichardt motion detector; the delay does the work of memory.
    predict     P(x, i, s) hears R(i, s) and X_x, the latter delayed by R's response time so the two
                arrive together. It fires for one (distance, height, velocity) and sends one synapse to
                the landing band the rule computes for them:
                    frames left  T = (1 - x) / vx
                    landing      y + vy * T, folded back into [0, 1] at each wall (a mirror ring)
                So the 1,168 P cells (8 distances x the 146 R cells whose shift stays on screen) are a
                lookup table drawn as wiring: the physics is in which
                landing cell each one feeds, not in any weight.
    land        8 LANDING cells (L). They take the place of the Pong ball cells: the same
                compare -> 11-simplex cliques -> paddle motor (pong.add_motor), with the paddle's
                spikes delayed to arrive with the landing cells' spikes.

The prediction cannot exist for the first six frames of a serve (the delay line still holds the old
ball), and it is wrong for six frames after a bounce (the motion it measures spans the bounce). It
sharpens as the ball approaches, since an error in velocity matters less the fewer frames are left.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.neurons import pong
from brain.neurons.circuit import Circuit

Y_BANDS = 16              # height cells
X_BANDS = 8               # distance cells
L_BANDS = pong.BANDS      # landing cells (8), fed to the Pong motor
K = 6                     # motion delay line, in frames
SHIFTS = range(-5, 6)     # band shifts over K frames that R cells detect
VX = 1.0 / 30             # horizontal speed per frame: 30 frames from serve to paddle
VY_MAX = 0.05             # fastest vertical speed per frame
PADDLE_H = 0.2
STEP = 1.0 / 40           # paddle step per motor spike
MAX_STEPS = 1             # at most one step per frame: top speed 0.025 per frame, half of VY_MAX
R_LATENCY = 7.0           # ms from an R cell's inputs to its spike, plus the eye's own lag (measured)
P_LATENCY = 4.5           # same for a P cell (measured)
L_LATENCY = 4.0           # P (one input, 0.2) -> L cell (measured)
P_TO_L_W = 0.2
FRAME_MS = pong.FRAME_MS
WIDTH = 0.6               # electrode reach, in bands: one cell, two within 0.1 band of an edge


def fold(y: float) -> float:
    """Reflect a height that ran past a wall back into [0, 1], as many times as it takes."""
    y = y % 2.0
    return 2.0 - y if y > 1.0 else y


def landing(x: float, y: float, vy: float, walls: bool = True) -> float:
    """Where a ball at (x, y) moving vy per frame (and VX) crosses the right edge. ``walls=False``
    ignores the bounces: a straight line, stopped at the edge of the screen."""
    y = y + vy * (1.0 - x) / VX
    return fold(y) if walls else min(1.0, max(0.0, y))


def electrode(y: float, n: int) -> np.ndarray:
    """Binary place code: a fixed pulse to every band whose centre is within WIDTH of y (one cell, or
    two right at a band edge). A graded pulse would make a cell's latency depend on where the ball sits in its band,
    and the motion detectors, which compare a spike now with one from 120 ms ago, need both spikes to
    keep the same latency."""
    d = np.abs(y * n - (np.arange(n) + 0.5))
    return np.where(d < WIDTH, pong.PULSE_AMP, 0.0)


def band(y: float, n: int) -> int:
    return min(n - 1, max(0, int(y * n)))


def build(walls: bool = True, plastic: bool = False) -> Circuit:
    """``walls=False`` wires the same circuit with each P cell's landing drawn as if there were no
    walls: the ablation that shows what the folding buys. ``plastic=True`` connects every P cell to
    every landing cell, the rule's choice at full weight and the others at zero, so that learning
    (brain/neurons/learn.py) can move the weight; with zero weights it plays exactly as the rule does."""
    c = Circuit()
    Y = [c.add(f"Y{i}", group="height") for i in range(Y_BANDS)]
    X = [c.add(f"X{x}", group="distance") for x in range(X_BANDS)]
    pad = [c.add(f"PADDLE{j}", group="paddle") for j in range(L_BANDS)]
    L = [c.add(f"L{k}", "FE", group="landing") for k in range(L_BANDS)]
    for i in range(Y_BANDS):
        for s in SHIFTS:
            if not 0 <= i - s < Y_BANDS:
                continue
            r = c.add(f"R{i}_{s:+d}", "FE", group="motion")
            c.connect(Y[i], r, pong.EYE_W, 1.0)
            c.connect(Y[i - s], r, pong.EYE_W, K * FRAME_MS + 1.0)      # the delay line
            yc, vy = (i + 0.5) / Y_BANDS, s / (K * Y_BANDS)
            for x in range(X_BANDS):
                p = c.add(f"P{x}_{i}_{s:+d}", "FE", group="predict")
                c.connect(r, p, pong.EYE_W, 1.0)
                c.connect(X[x], p, pong.EYE_W, 1.0 + R_LATENCY)
                k = band(landing((x + 0.5) / X_BANDS, yc, vy, walls), L_BANDS)
                for j in (range(L_BANDS) if plastic else [k]):
                    c.connect(p, L[j], P_TO_L_W if j == k else 0.0, 1.0)
    # Landing spikes come about 1 + R + 1 + P + 1 + L ms after the frame's pulse; the paddle's are
    # delayed by as much, so a landing cell and a paddle cell arrive at the comparator together.
    pong.add_motor(c, L, pad, pad_delay=3.0 + R_LATENCY + P_LATENCY + L_LATENCY)
    return c


@dataclass
class Game:
    seed: int = 0
    bx: float = 0.0
    by: float = 0.5
    vy: float = 0.0
    py: float = 0.5
    last_y: float = 0.5
    gravity: float = 0.0               # pull on vy per frame (0: the straight-line world the rule assumes)
    hits: int = 0
    misses: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.serve()

    def serve(self):
        self.bx, self.by = 0.0, float(self.rng.uniform(0.05, 0.95))
        self.vy = float(self.rng.uniform(-VY_MAX, VY_MAX))

    def landing(self) -> float:
        if not self.gravity:
            return landing(self.bx, self.by, self.vy)
        x, y, v = self.bx, self.by, self.vy             # with gravity: step a copy of the ball forward
        while True:
            x, y, v = x + VX, y + v, v - self.gravity
            if y < 0 or y > 1:
                y, v = fold(y), -v
            if x >= 1.0 - 1e-9:
                return y

    def move_paddle(self, dy: float):
        self.py = float(np.clip(self.py + dy, PADDLE_H / 2, 1 - PADDLE_H / 2))

    def tick(self) -> str | None:
        self.bx += VX
        self.by += self.vy
        self.vy -= self.gravity
        if self.by < 0 or self.by > 1:
            self.by = fold(self.by)
            self.vy = -self.vy
        if self.bx >= 1.0 - 1e-9:
            self.last_y = self.by                       # where it crossed: the landing the circuit feels
            hit = abs(self.by - self.py) <= PADDLE_H / 2
            self.hits += hit
            self.misses += not hit
            self.serve()
            return "hit" if hit else "miss"
        return None


def play(frames: int = 1200, seed: int = 0, controller: str = "neurons", noise: float = 0.0,
         kill: tuple[str, ...] = (), record: bool = False) -> dict:
    """Play ``frames`` frames (one serve takes 30). Controllers:

    neurons   the predicting circuit above
    nowalls   the same circuit, wired as if the ball never bounced
    reactive  the Pong circuit, which steers toward the ball's current height
    oracle    no neurons: moves at the paddle's top speed toward the true landing point (a ceiling)
    still     never moves
    random    a random step each frame
    """
    g = Game(seed=seed)
    rng = np.random.default_rng(seed + 1)
    c = build() if controller == "neurons" else build(walls=False) if controller == "nowalls" else pong.build() if controller == "reactive" else None
    sim = c.compile(seed) if c else None
    if sim:
        sim.noise = noise
        for name in kill:
            sim.alive[c.by_name(name)] = False
        N = len(c.neurons)
        up_sink, dn_sink = c.group("up")[-1], c.group("down")[-1]
        steps, pulse = int(FRAME_MS / sim.dt), int(pong.PULSE_MS / sim.dt)
        if controller in ("neurons", "nowalls"):
            Y, X, P = (np.array(c.group(k)) for k in ("height", "distance", "paddle"))
            L = np.array(c.group("landing"))
        else:
            Y, P = np.array(c.group("ball")), np.array(c.group("paddle"))
            L = np.array([], int)
    replay, errors = [], []
    for f in range(frames):
        ups = dns = 0
        spikes, lfired = [], []
        if sim:
            drive = np.zeros(N)
            if controller in ("neurons", "nowalls"):
                drive[Y] = electrode(g.by, Y_BANDS)
                drive[X] = electrode(g.bx, X_BANDS)
                drive[P] = electrode(g.py, L_BANDS)
            else:                                   # the Pong circuit, with the eyes it was tuned for
                drive[Y] = pong.place_drive(g.by)
                drive[P] = pong.place_drive(g.py)
            for k in range(steps):
                fired = sim.step(drive if k < pulse else None)
                ups += int(up_sink in fired)
                dns += int(dn_sink in fired)
                if len(L):
                    lfired += [int(n) for n in fired if n in L]
                if record:
                    spikes += [(round(sim.t, 2), int(n)) for n in fired]
            g.move_paddle(STEP * max(-MAX_STEPS, min(MAX_STEPS, ups - dns)))
        elif controller == "oracle":
            g.move_paddle(float(np.clip(g.landing() - g.py, -STEP * MAX_STEPS, STEP * MAX_STEPS)))
        elif controller == "random":
            g.move_paddle(STEP * int(rng.integers(-1, 2)))
        true_land = g.landing()
        guess = None
        if lfired:
            ks = [int(np.flatnonzero(L == n)[0]) for n in lfired]
            guess = (float(np.mean(ks)) + 0.5) / L_BANDS
            errors.append((g.bx, abs(guess - true_land)))
        state = {"bx": round(g.bx, 4), "by": round(g.by, 4), "vy": round(g.vy, 5), "py": round(g.py, 4),
                 "land": round(true_land, 4),
                 "guess": None if guess is None else round(guess, 4), "up": ups, "down": dns}
        event = g.tick()
        if record:
            state.update(event=event, spikes=spikes)
            replay.append(state)
    out = {"frames": frames, "hits": g.hits, "misses": g.misses, "hit_rate": g.hits / max(1, g.hits + g.misses)}
    if errors:
        e = np.array(errors)
        out["guess_error"] = {name: float(np.median(e[(e[:, 0] >= lo) & (e[:, 0] < hi), 1]))
                              for name, lo, hi in (("far", 0.2, 0.5), ("mid", 0.5, 0.8), ("near", 0.8, 1.0))
                              if ((e[:, 0] >= lo) & (e[:, 0] < hi)).any()}
    if c:
        out.update(neurons=len(c.neurons), synapses=len(c.synapses))
    if record:
        out["replay"] = replay
        out["names"] = [n.name for n in c.neurons] if c else []
        out["groups"] = [n.group for n in c.neurons] if c else []
        out["signs"] = [n.sign for n in c.neurons] if c else []
    return out
