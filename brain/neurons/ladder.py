"""The same hand-wired circuits on each rung of the ladder of internal complexity.

He et al. (2024, Nat Comput Sci 4:584, doi:10.1038/s43588-024-00674-9) argue that richer cells
("internal complexity") can stand in for more simple ones ("external complexity"). Our circuits put
their complexity in the wiring: delays, coincidence cells, directed cliques. This runs every game, wired
exactly as recorded in docs/NEURONS.md, on each kind of cell (``circuit.CELLS``), clean and under
membrane noise, on the held-out seeds the docs report, and times each run.

    python -m brain neurons ladder [--workers 4] [--json FILE]
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

from brain.neurons import catch, circuit, pong, runner

# game -> (frames, seeds): the recorded settings in docs/NEURONS.md
GAMES = {"pong": (800, (0, 1, 2)), "catch": (900, (10, 11, 12, 13, 14)), "run": (5000, (20, 21, 22, 23))}
CONDITIONS = {
    "pong": ({"noise": 0.0}, {"noise": 2.0}, {"noise": 4.0}, {"lesion": 0.3}),
    "catch": ({"noise": 0.0}, {"noise": 2.0}, {"noise": 4.0}),
    "run": ({"noise": 0.0}, {"noise": 2.0}, {"noise": 4.0}),
}


def _one(job: tuple) -> dict:
    game, cells, cond, seed = job
    frames = GAMES[game][0]
    t0 = time.process_time()
    if game == "pong":
        r = pong.play(frames, seed=seed, cells=cells, **cond)
        out = {"good": r["hits"], "bad": r["misses"]}
    elif game == "catch":
        r = catch.play(frames, seed=seed, cells=cells, **cond)
        out = {"good": r["hits"], "bad": r["misses"]}
    else:
        r = runner.play(frames, seed=seed, cells=cells, **cond)
        out = {"good": r["rows"], "bad": r["crashes"]}
    out.update(game=game, cells=cells, cond=cond, seed=seed, cpu_s=time.process_time() - t0)
    return out


def jobs(games=tuple(GAMES), cells=circuit.CELLS) -> list[tuple]:
    return [(g, c, cond, s) for g in games for c in cells for cond in CONDITIONS[g] for s in GAMES[g][1]]


def run(workers: int = 4, games=tuple(GAMES), cells=circuit.CELLS) -> list[dict]:
    js = jobs(games, cells)
    # the slowest (Hodgkin-Huxley runner) first, so the pool finishes together
    js.sort(key=lambda j: -(GAMES[j[0]][0] * (9 if j[1] == "hh" else 1)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, js))


def table(rows: list[dict]) -> list[str]:
    """One line per game, condition and cell kind: the score as the docs report it, and CPU time."""
    out = []
    label = {"pong": ("hits", "rallies"), "catch": ("caught", "serves"), "run": ("crashes", "rows")}
    keys = sorted({(r["game"], tuple(sorted(r["cond"].items()))) for r in rows},
                  key=lambda k: (list(GAMES).index(k[0]), k[1]))
    for game, cond in keys:
        cs = ", ".join(f"{k} {v:g}" for k, v in cond)
        for cells in circuit.CELLS:
            rs = [r for r in rows if r["game"] == game and tuple(sorted(r["cond"].items())) == cond and r["cells"] == cells]
            if not rs:
                continue
            good, bad, cpu = sum(r["good"] for r in rs), sum(r["bad"] for r in rs), sum(r["cpu_s"] for r in rs)
            if game == "run":
                score = f"{bad} crashes in {good + bad} rows"
            else:
                score = f"{good}/{good + bad} {label[game][0]} ({good / max(1, good + bad):.0%})"
            out.append(f"{game:5s} {cs:10s} {cells:3s} ({circuit.STATE_VARIABLES[cells]} var)  {score:28s} cpu {cpu:7.1f} s")
    return out
