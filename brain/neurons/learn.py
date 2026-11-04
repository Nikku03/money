"""A school of Catch circuits that improve by themselves and from each other.

The world changes under them: the ball now falls (gravity), so the straight-line rule each circuit was
wired with predicts the wrong landing. What can change is one layer of synapses: each prediction cell P
connects to all 8 landing cells L. A P cell's eight weights hold what it expects: where the ball lands
when P fires. The strongest is always 0.2 (enough to fire L alone), the others in proportion; as built,
the rule's landing has everything and the others nothing.

Learning from itself
    When a ball crosses the paddle's line, the circuit feels where it crossed (its landing cell is
    told, the way a climbing fibre tells the cerebellum about an error). Every P cell that fired during
    that serve, from the seventh frame on (before then the motion delay line still holds the last
    ball), moves its expectation toward that landing:  p <- p + rate * (onehot(landing) - p).
    The rate is ``SELF_RATE``.

Learning from others
    Several circuits play side by side, each its own serves. After each round they share in two ways:

    experience  each shows the others its serves: which P cells fired and where the ball landed. The
                circuits are wired alike, so a peer's serve is one this circuit could have seen, and it
                learns from it as from its own, at ``SOCIAL_RATE`` times its trust in that peer.
    knowledge   each moves its expectations toward each peer's, at ``COPY_RATE`` times its trust.

Trust is earned, not given
    One circuit's landing sense is broken: it reports every landing mirrored. It learns the wrong
    lesson itself and passes it on. Trust in a peer is tested twice, and is the
    lower of the two: how well the peer's expectations predict *my own* recent serves (landing band
    within one), and how often the landings it reports agree with what my expectations predict for
    those serves. Each is scaled from a floor (0) to perfect (1). A peer whose knowledge fails on my
    experience, or whose claims contradict what I know, gets no weight, however it reads. This is the rule the market bank applies to its sources (CLAUDE.md, decide score).

Modes, the same serves for each:
    fixed    nobody learns: the hand-wired rule in a world it was not built for
    alone    each learns only from itself
    naive    alone + experience and knowledge from every peer at full trust, the broken one included
    trusted  as naive, with trust earned as above
    honest   as naive, but no circuit is broken: the ceiling for learning together
"""
from __future__ import annotations

import numpy as np

from brain.neurons import catch, pong

GRAVITY = 0.0015          # vy lost per frame; the rule-wired circuit catches about 55% here, the oracle 98%
SELF_RATE = 0.3
SOCIAL_RATE = 0.3
COPY_RATE = 0.2
TRUST_FLOOR = 0.4         # a peer predicting my serves (within one band) this well or worse gets no weight
TRUST_WINDOW = 40         # my most recent serves, used to test peers
ELIGIBLE_FROM = catch.K + 1
MODES = ("fixed", "alone", "naive", "trusted", "honest")


class Agent:
    def __init__(self, name: str, seed: int, broken: bool = False, gravity: float = GRAVITY):
        self.name, self.broken = name, broken
        self.c = catch.build(plastic=True)
        self.sim = self.c.compile(seed)
        self.game = catch.Game(seed=seed, gravity=gravity)
        self.P = np.array(self.c.group("predict"))
        self.L = np.array(self.c.group("landing"))
        self.Y, self.X = np.array(self.c.group("height")), np.array(self.c.group("distance"))
        self.pad = np.array(self.c.group("paddle"))
        self.row = np.full(len(self.c.neurons), -1)
        self.row[self.P] = np.arange(len(self.P))
        self.lrow = {int(n): k for k, n in enumerate(self.L)}
        # Expectations from the wiring as built: the rule's landing at 1, the rest at 0.
        self.p = np.zeros((len(self.P), len(self.L)))
        for r, n in enumerate(self.P):
            post, w, _ = self.sim.out[n]
            self.p[r, [self.lrow[int(x)] for x in post]] = w / catch.P_TO_L_W
        self.records: list[tuple[np.ndarray, int]] = []     # my own serves: (P rows that fired, landing band)
        self.up, self.dn = self.c.group("up")[-1], self.c.group("down")[-1]
        self.eligible_until = 1.0                              # record P cells only while the ball is short of this x

    def apply(self) -> None:
        """Write the expectations into the synapses: the strongest at full weight, the rest in proportion."""
        top = self.p.max(axis=1, keepdims=True)
        w = catch.P_TO_L_W * self.p / np.where(top > 0, top, 1.0)
        for r, n in enumerate(self.P):
            self.sim.set_weights(int(n), w[r])                 # P -> L0..L7, in wiring order

    def play(self, frames: int) -> tuple[int, int, list]:
        """Play ``frames`` frames; return hits, misses and this round's serve records (as reported)."""
        g, sim, N = self.game, self.sim, len(self.c.neurons)
        steps, pulse = int(catch.FRAME_MS / sim.dt), int(pong.PULSE_MS / sim.dt)
        h0, m0 = g.hits, g.misses
        fired_rows: set[int] = set()
        serve_frame = 0
        out = []
        for _ in range(frames):
            drive = np.zeros(N)
            drive[self.Y] = catch.electrode(g.by, catch.Y_BANDS)
            drive[self.X] = catch.electrode(g.bx, catch.X_BANDS)
            drive[self.pad] = catch.electrode(g.py, catch.L_BANDS)
            ups = dns = 0
            for k in range(steps):
                fired = sim.step(drive if k < pulse else None)
                ups += int(self.up in fired)
                dns += int(self.dn in fired)
                if serve_frame >= ELIGIBLE_FROM and g.bx < self.eligible_until:
                    fired_rows.update(int(r) for r in self.row[fired] if r >= 0)
            g.move_paddle(catch.STEP * max(-catch.MAX_STEPS, min(catch.MAX_STEPS, ups - dns)))
            serve_frame += 1
            if g.tick():
                k = catch.band(g.last_y, catch.L_BANDS)
                if self.broken:
                    k = catch.L_BANDS - 1 - k                  # its landing sense is mirrored
                if fired_rows:
                    out.append((np.array(sorted(fired_rows)), k))
                fired_rows, serve_frame = set(), 0
        self.records = (self.records + out)[-TRUST_WINDOW:]
        return g.hits - h0, g.misses - m0, out

    def learn(self, records: list, rate: float) -> None:
        for rows, k in records:
            target = np.zeros(len(self.L))
            target[k] = 1.0
            self.p[rows] += rate * (target - self.p[rows])

    def score(self, p: np.ndarray) -> float:
        """How well expectations ``p`` predict my own recent serves: landing band within one."""
        if not self.records:
            return 0.0
        ok = [abs(int(np.argmax(p[rows].sum(axis=0))) - k) <= 1 for rows, k in self.records]
        return float(np.mean(ok))

    def credibility(self, reports: list) -> float:
        """How often a peer's reported landings agree (within one band) with what I would predict
        for the same serves: its claims, checked against my knowledge."""
        if not reports:
            return 0.0
        return float(np.mean([abs(int(np.argmax(self.p[rows].sum(axis=0))) - k) <= 1 for rows, k in reports]))

    def trust(self, peer: "Agent", reports: list) -> float:
        """Both tests must pass: the peer's knowledge predicts my serves, and its reports agree with
        my knowledge. Each is scaled from the floor (0) to perfect (1); trust is the lower."""
        scale = lambda x: float(np.clip((x - TRUST_FLOOR) / (1 - TRUST_FLOOR), 0.0, 1.0))
        return min(scale(self.score(peer.p)), scale(self.credibility(reports)))


_EXAMS: dict = {}


def exam_set(serves: int = 200, seed: int = 5000, gravity: float = GRAVITY, until: float = 1.0) -> list:
    """Held-out serves as (P rows that fired, true landing band). P cells hear only the ball, never the
    paddle or the P -> L weights, so one recording scores any agent's expectations. ``until`` keeps
    only the P cells that fired while the ball was short of that x: 0.5 asks for the early guess, the
    one a slow paddle needs."""
    key = (serves, seed, gravity, until)
    if key not in _EXAMS:
        a = Agent("examiner", seed=seed, gravity=gravity)
        a.eligible_until = until
        recs: list = []
        while len(recs) < serves:
            recs += a.play(30 * (serves - len(recs)))[2]
        _EXAMS[key] = recs[:serves]
    return _EXAMS[key]


def knowledge(p: np.ndarray, records: list, within: int = 1) -> float:
    """Share of serves whose landing band ``p`` predicts within ``within`` bands."""
    return float(np.mean([abs(int(np.argmax(p[rows].sum(axis=0))) - k) <= within for rows, k in records]))


def grades(p: np.ndarray) -> dict:
    """Two exams on the same 200 held-out serves: the whole serve, band within one (lenient), and the
    far half only, exact band (what a paddle at half the ball's speed needs)."""
    return {"lenient": knowledge(p, exam_set()), "early": knowledge(p, exam_set(until=0.5), within=0)}


def school(mode: str, rounds: int = 12, frames: int = 300, agents: int = 4, seed: int = 0,
           exam_frames: int = 600) -> dict:
    """Run one mode. Agent 3 is broken except in ``honest``. Returns, per round, each agent's catch
    rate, its trust in each peer and its two exam grades (``grades``; round -1 is before any
    learning), and a final exam: each agent plays fresh serves
    with learning off."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    team = [Agent(f"A{i}", seed=seed + i, broken=(i == agents - 1 and mode != "honest")) for i in range(agents)]
    heard: list[list] = [[] for _ in team]                     # each agent's recent reports, as shared
    history = [{"round": -1, "catch": None, "trust": None, "grades": [grades(a.p) for a in team]}]
    for r in range(rounds):
        played = [a.play(frames) for a in team]
        for j, (_, _, recs) in enumerate(played):
            heard[j] = (heard[j] + recs)[-TRUST_WINDOW:]
        row = {"round": r, "catch": [h / max(1, h + m) for h, m, _ in played], "trust": None}
        if mode != "fixed":
            for a, (_, _, recs) in zip(team, played):
                a.learn(recs, SELF_RATE)
        if mode in ("naive", "trusted", "honest"):
            trust = [[1.0 if (mode != "trusted" or i == j) else a.trust(b, heard[j]) for j, b in enumerate(team)]
                     for i, a in enumerate(team)]
            before = [a.p.copy() for a in team]
            for i, a in enumerate(team):
                for j, (_, _, recs) in enumerate(played):
                    if i != j and trust[i][j] > 0:
                        a.learn(recs, SOCIAL_RATE * trust[i][j])
                pull = sum(trust[i][j] * (before[j] - a.p) for j in range(len(team)) if j != i)
                a.p += COPY_RATE * pull / (len(team) - 1)
            row["trust"] = trust
        for a in team:
            a.apply()
        row["grades"] = [grades(a.p) for a in team]
        history.append(row)
    exam = []
    for i, a in enumerate(team):                             # fresh serves, nobody learning
        a.game = catch.Game(seed=1000 + seed + i, gravity=a.game.gravity, py=a.game.py)
        h, m, _ = a.play(exam_frames)
        exam.append(h / max(1, h + m))
    return {"mode": mode, "seed": seed, "agents": [a.name + (" (broken)" if a.broken else "") for a in team],
            "rounds": history, "exam": exam, "frames": frames, "exam_frames": exam_frames}
