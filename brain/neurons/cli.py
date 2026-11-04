"""``python -m brain neurons play|wiring|topology``: the hand-wired spiking circuit (docs/NEURONS.md)."""
from __future__ import annotations

import json
from pathlib import Path


def _play(a):
    from brain.neurons import pong
    kill = tuple(x.strip() for x in (a.kill or "").split(",") if x.strip())
    rows = []
    for ctl in (["still", "random", "neurons"] if a.compare else [a.controller]):
        H = M = 0
        for s in range(a.seed, a.seed + a.games):
            r = pong.play(a.frames, seed=s, noise=a.noise, lesion=a.lesion, kill=kill, controller=ctl,
                          record=bool(a.replay or a.html) and ctl == "neurons" and s == a.seed)
            H, M = H + r["hits"], M + r["misses"]
            if "replay" in r:
                data = json.dumps(r, separators=(",", ":"))
                if a.replay:
                    Path(a.replay).write_text(data)
                    print(f"replay of game {s} written to {a.replay}")
                if a.html:
                    page = (Path(__file__).with_name("viewer.html")).read_text()
                    Path(a.html).write_text(page.replace("__REPLAY__", data.replace("</", "<\\/")))
                    print(f"viewer of game {s} written to {a.html}")
        rows.append((ctl, H, M))
        print(f"{ctl:8s} hits {H:4d}  misses {M:4d}  hit rate {H / max(1, H + M):.0%}"
              f"   ({a.games} game(s) x {a.frames} frames, noise {a.noise}, lesion {a.lesion:.0%}"
              f"{', killed ' + ','.join(kill) if kill else ''})")
    print(f"circuit: {r['neurons']} neurons, {r['synapses']} synapses; no weight was learned")


def _catch(a):
    from brain.neurons import catch
    kill = tuple(x.strip() for x in (a.kill or "").split(",") if x.strip())
    ctls = ["still", "random", "reactive", "nowalls", "neurons", "oracle"] if a.compare else [a.controller]
    for ctl in ctls:
        H = M = 0
        errs = []
        for s in range(a.seed, a.seed + a.games):
            first = s == a.seed and ctl == "neurons" and a.html
            r = catch.play(a.frames, seed=s, controller=ctl, noise=a.noise, kill=kill, record=bool(first))
            H, M = H + r["hits"], M + r["misses"]
            if r.get("guess_error"):
                errs.append(r["guess_error"])
            if first:
                page = Path(__file__).with_name("catch_viewer.html").read_text()
                data = json.dumps(r, separators=(",", ":")).replace("</", "<\\/")
                Path(a.html).write_text(page.replace("__REPLAY__", data))
                print(f"viewer of game {s} written to {a.html}")
        err = ""
        if errs:
            keys = [k for k in ("far", "mid", "near") if all(k in e for e in errs)]
            err = "   landing guess off by " + ", ".join(f"{k} {sum(e[k] for e in errs) / len(errs):.3f}" for k in keys)
        print(f"{ctl:9s} caught {H:4d}  missed {M:4d}  catch rate {H / max(1, H + M):.0%}{err}")
    print(f"({a.games} game(s) x {a.frames} frames from seed {a.seed}, noise {a.noise}"
          f"{', killed ' + ','.join(kill) if kill else ''}; paddle half-height {catch.PADDLE_H / 2})")


def _wiring(a):
    from brain.neurons import catch, pong
    c = catch.build() if a.game == "catch" else pong.build()
    names = [n.name for n in c.neurons]
    ids = [c.by_name(a.neuron)] if a.neuron else range(len(names))
    for i in ids:
        n = c.neurons[i]
        print(f"{n.name} ({n.kind}, {'excitatory' if n.sign > 0 else 'inhibitory'}, group {n.group})")
        ins = c.inputs(i)
        outs = c.outputs(i)
        print("  inputs from:  " + (", ".join(f"{names[s.pre]} w{s.weight:.3f} @{s.delay:g}ms" for s in ins) or "electrode only"))
        print("  outputs to:   " + (", ".join(f"{names[s.post]} w{s.weight:.3f} @{s.delay:g}ms" for s in outs) or "the paddle (motor)"))


def _topology(a):
    from brain.neurons import catch, pong
    from brain.neurons.topology import simplices
    c = catch.build() if a.game == "catch" else pong.build()
    counts = simplices(c.adjacency())
    print("directed simplices by dimension (k neurons+1, all-to-all forward):")
    for k, n in counts.items():
        print(f"  dim {k:2d}: {n:6d}" + ("   <- neurons" if k == 0 else "   <- connections" if k == 1 else ""))
    print(f"highest dimension: {max(counts)}")


def add_cli(sub) -> None:
    p = sub.add_parser("neurons", help="a hand-wired spiking circuit that plays Pong (docs/NEURONS.md)")
    s = p.add_subparsers(dest="neurons_cmd", required=True)
    q = s.add_parser("play", help="play Pong with the circuit and print the score")
    q.add_argument("--frames", type=int, default=1500, help="frames per game (20 ms of neural time each)")
    q.add_argument("--games", type=int, default=1)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--noise", type=float, default=0.0, help="membrane current noise (spontaneous firing starts near 4)")
    q.add_argument("--lesion", type=float, default=0.0, help="fraction of clique middle neurons removed at random")
    q.add_argument("--kill", help="comma-separated neuron names to remove, e.g. UP1,DOWN1")
    q.add_argument("--controller", choices=["neurons", "still", "random"], default="neurons")
    q.add_argument("--compare", action="store_true", help="also play with a still and a random paddle")
    q.add_argument("--replay", help="write a frame-by-frame replay (spikes, voltages) as JSON here")
    q.add_argument("--html", help="write a page that replays the first game: court, circuit, raster, voltages")
    q.set_defaults(fn=_play)
    q = s.add_parser("catch", help="Catch: a fast, bouncing ball the circuit must predict to reach")
    q.add_argument("--frames", type=int, default=900, help="frames per game (30 per serve)")
    q.add_argument("--games", type=int, default=1)
    q.add_argument("--seed", type=int, default=10)
    q.add_argument("--noise", type=float, default=0.0)
    q.add_argument("--kill", help="comma-separated neuron names to remove")
    q.add_argument("--controller", choices=["neurons", "nowalls", "reactive", "oracle", "still", "random"], default="neurons")
    q.add_argument("--compare", action="store_true", help="play every controller, same serves")
    q.add_argument("--html", help="write a page that replays the first game")
    q.set_defaults(fn=_catch)
    q = s.add_parser("wiring", help="who feeds each neuron and whom it feeds, with weights and delays")
    q.add_argument("neuron", nargs="?", help="one neuron's name, e.g. UP11, C62, or (catch) P3_9_+2")
    q.add_argument("--game", choices=["pong", "catch"], default="pong")
    q.set_defaults(fn=_wiring)
    q = s.add_parser("topology", help="count the circuit's directed simplices by dimension")
    q.add_argument("--game", choices=["pong", "catch"], default="pong")
    q.set_defaults(fn=_topology)
