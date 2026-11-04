"""Runner: a three-lane endless runner, played by a hand-wired spiking circuit.

An original game in the endless-runner genre. The runner holds one of three lanes while rows of
obstacles come at it, faster and faster:

    wall     fills a lane: switch lanes before it arrives
    hurdle   low: jump over it
    beam     high: duck under it

Every row leaves at least one lane without a wall. A lane switch has a 4-frame cooldown; a jump or a
duck lasts 8 frames. Speed starts at 1 unit per frame and rises 0.0015 per frame to 2.4; after a crash
the run restarts at the start speed.

Nothing is trained. The circuit, by rule:

    eyes      15 cells, pulsed at each frame's start with a fixed electrode pulse:
              LANE_l     the runner is in lane l
              WALL_l     a wall in lane l will arrive in 3 to 12 frames
              CLEAR_l    no wall in lane l within 13 frames (an OFF cell: spikes cannot signal an
                         absence, so the eye reports it, as a retina's OFF channel does)
              HURDLE_l   a hurdle in lane l arrives in 2 to 5 frames
              BEAM_l     a beam in lane l arrives in 2 to 5 frames
              The obstacle cells are tuned to time-to-contact, not distance, as the locust's
              looming detector (LGMD) is: at any speed they fire the same number of frames ahead.
    decide    12 coincidence cells:
              GO_R(l)    LANE_l + WALL_l + CLEAR_(l+1)      (and GO_R2: LANE_0 + WALL_0 + CLEAR_2)
              GO_L(l)    LANE_l + WALL_l + CLEAR_(l-1)      (and GO_L2: LANE_2 + WALL_2 + CLEAR_0)
              JUMP(l)    LANE_l + HURDLE_l
              DUCK(l)    LANE_l + BEAM_l
              Three-input cells get 0.055 per input (two are 0.11, under the ~0.135 threshold; three
              fire). Two-input cells get 0.08 each, as in Pong.
    act       four directed 11-simplex cliques, LEFT, RIGHT, JUMP, DUCK (pong.add_clique). Each sink
              spike is a command. LEFT and RIGHT race: each source fires an inhibitory cell that
              silences the other source, and the GO_L cells' axons are 1.5 ms longer, so when both
              sides are clear the circuit goes right.

Eye to command takes about 2 frames (the cliques' volley is ~30 ms), so the windows above are set
2 frames earlier than the moment the action is needed. The wall window reaches 12 frames because an
escape two lanes away takes two switches 4 frames apart, and the lane it ends in may hold a hurdle
or beam that still has to be seen and jumped (with a 9-frame window, every crash was this).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.neurons import pong
from brain.neurons.circuit import Circuit

LANES = 3
FRAME_MS = pong.FRAME_MS
SPAWN = 40.0              # distance at which a row appears
SPEED0, ACCEL, SPEED_MAX = 1.0, 0.0015, 2.4
GAP = (16, 26)            # frames between rows, at the current speed
SWITCH_COOLDOWN = 4
AIR, DUCK_TIME = 8, 8
WALL_TTC = (3.0, 12.0)    # frames ahead a WALL cell fires: early enough for a two-lane escape
CLEAR_TTC = 13.0          # CLEAR fires when no wall arrives sooner than this
LOW_TTC = (2.0, 5.0)      # frames ahead HURDLE and BEAM cells fire
TRI_W = 0.055             # each input of a three-input coincidence cell
LEFT_LAG = 1.5            # extra ms on the GO_L axons: the tie-break
ACTIONS = ("left", "right", "jump", "duck")


def build() -> Circuit:
    c = Circuit()
    eye = {}
    for kind in ("LANE", "WALL", "CLEAR", "HURDLE", "BEAM"):
        for l in range(LANES):
            eye[kind, l] = c.add(f"{kind}{l}", group="eye")
    src = {a: pong.add_clique(c, a.upper(), a) for a in ACTIONS}
    inh_r = c.add("INH_RIGHT", "FS", group="wta", sign=-1)
    inh_l = c.add("INH_LEFT", "FS", group="wta", sign=-1)
    c.connect(src["right"][0], inh_r, pong.WTA_W, 0.5)
    c.connect(src["left"][0], inh_l, pong.WTA_W, 0.5)
    c.connect(inh_r, src["left"][0], pong.WTA_INH, 0.5)
    c.connect(inh_l, src["right"][0], pong.WTA_INH, 0.5)

    def cell(name, inputs, w, target, delay=1.0):
        x = c.add(name, "FE", group="decide")
        for i in inputs:
            c.connect(eye[i], x, w, 1.0)
        c.connect(x, src[target][0], pong.DECIDE_W, delay)

    for l in range(LANES):
        for d, a, lag in ((+1, "right", 0.0), (-1, "left", LEFT_LAG)):
            for step in (1, 2):
                t = l + d * step
                if not 0 <= t < LANES or (step == 2 and l != (0 if d > 0 else LANES - 1)):
                    continue
                cell(f"GO_{a[0].upper()}{'2' if step == 2 else ''}{l}",
                     [("LANE", l), ("WALL", l), ("CLEAR", t)], TRI_W, a, 1.0 + lag)
        cell(f"JUMP{l}", [("LANE", l), ("HURDLE", l)], pong.EYE_W, "jump")
        cell(f"DUCK{l}", [("LANE", l), ("BEAM", l)], pong.EYE_W, "duck")
    return c


@dataclass
class Game:
    seed: int = 0
    lane: int = 1
    speed: float = SPEED0
    air: int = 0
    duck: int = 0
    cool: int = 0
    frame: int = 0
    run_frames: int = 0
    rows: int = 0                       # rows passed, this run
    runs: list = field(default_factory=list)   # (frames survived, rows passed, top speed, cause) per crash
    obstacles: list = field(default_factory=list)   # [lane, kind, distance]
    next_row: float = 10.0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def ttc(self, d: float) -> float:
        return d / self.speed

    def act(self, a: str) -> bool:
        if a in ("left", "right") and self.cool == 0:
            t = self.lane + (1 if a == "right" else -1)
            if 0 <= t < LANES:
                self.lane, self.cool = t, SWITCH_COOLDOWN
                return True
        if a == "jump" and not self.air:
            self.air, self.duck = AIR, 0
            return True
        if a == "duck" and not self.air and not self.duck:
            self.duck = DUCK_TIME
            return True
        return False

    def spawn(self):
        kinds = self.rng.choice(["none", "wall", "hurdle", "beam"], size=LANES, p=[0.35, 0.35, 0.15, 0.15])
        if all(k == "wall" for k in kinds):
            kinds[self.rng.integers(LANES)] = "none"
        if all(k == "none" for k in kinds):
            kinds[self.rng.integers(LANES)] = "wall"
        for l, k in enumerate(kinds):
            if k != "none":
                self.obstacles.append([l, str(k), SPAWN])

    def tick(self) -> str | None:
        """Advance one frame; return the crash cause, if any."""
        self.frame += 1
        self.run_frames += 1
        self.speed = min(SPEED_MAX, self.speed + ACCEL)
        self.air, self.duck, self.cool = max(0, self.air - 1), max(0, self.duck - 1), max(0, self.cool - 1)
        crash = None
        passed = set()
        for o in self.obstacles:
            before = o[2]
            o[2] -= self.speed
            if before > 0 >= o[2]:
                passed.add(id(o))
                if o[0] == self.lane and crash is None:
                    if o[1] == "wall" or (o[1] == "hurdle" and not self.air) or (o[1] == "beam" and not self.duck):
                        crash = o[1]
        rows_now = {round(o[2] + self.speed, 6) for o in self.obstacles if id(o) in passed}
        self.rows += len(rows_now)
        self.obstacles = [o for o in self.obstacles if o[2] > 0]
        self.next_row -= 1
        if self.next_row <= 0:
            self.spawn()
            self.next_row = float(self.rng.integers(*GAP))
        if crash:
            self.runs.append((self.run_frames, self.rows, round(self.speed, 3), crash))
            self.obstacles, self.speed, self.run_frames, self.rows = [], SPEED0, 0, 0
            self.air = self.duck = self.cool = 0
            self.lane, self.next_row = 1, 10.0
        return crash

    # --- what the eyes see -----------------------------------------------------------------------
    def view(self, by: str = "time") -> dict:
        """Which eye cells fire this frame. ``by="distance"`` tunes the obstacle cells to distance
        instead, matched to the start speed: the ablation that shows why time-to-contact matters."""
        f = (lambda d: d / self.speed) if by == "time" else (lambda d: d / SPEED0)
        on = {("LANE", self.lane)}
        for l in range(LANES):
            walls = [f(o[2]) for o in self.obstacles if o[0] == l and o[1] == "wall"]
            if any(WALL_TTC[0] <= t < WALL_TTC[1] for t in walls):
                on.add(("WALL", l))
            if not any(t < CLEAR_TTC for t in walls):
                on.add(("CLEAR", l))
            for kind, cellname in (("hurdle", "HURDLE"), ("beam", "BEAM")):
                if any(o[0] == l and o[1] == kind and LOW_TTC[0] <= f(o[2]) < LOW_TTC[1] for o in self.obstacles):
                    on.add((cellname, l))
        return on


def oracle(g: Game) -> list[str]:
    """No neurons, no delay: acts on the true state with the same moves the game allows."""
    out = []
    mine = [o for o in g.obstacles if o[0] == g.lane]
    if any(o[1] == "wall" and g.ttc(o[2]) < 7 for o in mine):
        clear = [l for l in range(LANES) if not any(o[0] == l and o[1] == "wall" and g.ttc(o[2]) < 9 for o in g.obstacles)]
        if clear:
            t = min(clear, key=lambda l: (abs(l - g.lane), -l))
            out.append("right" if t > g.lane else "left")
    if any(o[1] == "hurdle" and g.ttc(o[2]) < 2.5 for o in mine):
        out.append("jump")
    if any(o[1] == "beam" and g.ttc(o[2]) < 2.5 for o in mine):
        out.append("duck")
    return out


def play(frames: int = 5000, seed: int = 0, controller: str = "neurons", noise: float = 0.0,
         record: bool = False) -> dict:
    """Controllers: neurons (time-to-contact eyes), distance (the same circuit, distance-tuned eyes),
    oracle, still, random."""
    g = Game(seed=seed)
    rng = np.random.default_rng(seed + 1)
    c = build() if controller in ("neurons", "distance") else None
    if c:
        sim = c.compile(seed)
        sim.noise = noise
        N = len(c.neurons)
        cell = {n.name: n.id for n in c.neurons}
        sinks = {a: c.group(a)[-1] for a in ACTIONS}
        steps, pulse = int(FRAME_MS / sim.dt), int(pong.PULSE_MS / sim.dt)
    replay = []
    for _ in range(frames):
        spikes, cmds = [], []
        if c:
            drive = np.zeros(N)
            for kind, l in g.view("time" if controller == "neurons" else "distance"):
                drive[cell[f"{kind}{l}"]] = pong.PULSE_AMP
            for k in range(steps):
                fired = sim.step(drive if k < pulse else None)
                cmds += [a for a, s in sinks.items() if s in fired]
                if record:
                    spikes += [(round(sim.t, 2), int(n)) for n in fired]
        elif controller == "oracle":
            cmds = oracle(g)
        elif controller == "random" and rng.random() < 0.15:
            cmds = [str(rng.choice(ACTIONS))]
        done = [a for a in dict.fromkeys(cmds) if g.act(a)]
        state = None
        if record:
            state = {"lane": g.lane, "air": g.air, "duck": g.duck, "speed": round(g.speed, 3),
                     "obs": [[o[0], o[1][0], round(o[2], 2)] for o in g.obstacles], "acts": done}
        crash = g.tick()
        if record:
            state.update(crash=crash, spikes=spikes)
            replay.append(state)
    lives = g.runs + [(g.run_frames, g.rows, round(g.speed, 3), None)]
    out = {"frames": frames, "crashes": len(g.runs),
           "rows_per_crash": sum(r[1] for r in lives) / max(1, len(g.runs)),
           "best_run_rows": max(r[1] for r in lives), "top_speed": max(r[2] for r in lives),
           "causes": {k: sum(1 for r in g.runs if r[3] == k) for k in ("wall", "hurdle", "beam")},
           "rows": sum(r[1] for r in lives)}
    if c:
        out.update(neurons=len(c.neurons), synapses=len(c.synapses))
    if record:
        out.update(replay=replay, names=[n.name for n in c.neurons] if c else [],
                   groups=[n.group for n in c.neurons] if c else [], signs=[n.sign for n in c.neurons] if c else [])
    return out
