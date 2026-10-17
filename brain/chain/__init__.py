"""The chain engine, beside the evidence engine: follow cause -> effect links one step at a time.

``engine/`` holds three files copied unchanged from Nikku03/cell (see engine/PROVENANCE.md): the bump-node
dendritic tree (``chainexact``), the loop that writes down one item per step (``chainsnap``, the snap), and the
fading synaptic memory (``biomemory``, unused here). The engine was built for one task: given a lookup table
("A leads to B") shown every step, where do you end up after k steps? It was exact to 64 steps on 6 items.

Here the table comes from ``knowledge/links.json``. Every path the links allow from a starting state is
written as such a table (each node on the path points to the next; every other node points to itself, so a
chain stops), and the engine follows it. A plain lookup gives the same chain exactly, so each answer is shown
beside it and the two are compared. The links themselves are textbook knowledge and untested: what a chain
says about a stock counts only once the evidence engine has measured those links on the bank's history.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from brain import config

M = 6                      # items per table: the size the engine was exact on (chainsnap P1)
LINKS = config.ROOT / "knowledge" / "links.json"
HERE = Path(__file__).resolve().parent


def _engine():
    spec = importlib.util.spec_from_file_location("chainsnap", HERE / "engine" / "chainsnap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_links(path=None) -> list[dict]:
    return json.loads(Path(path or LINKS).read_text())["links"]


def paths(links, start: str, hops: int) -> list[list[dict]]:
    """Every chain of up to ``hops`` links from ``start`` (no node twice), strongest first link first."""
    out_of = {}
    for ln in sorted(links, key=lambda x: -x.get("weight", 1.0)):
        out_of.setdefault(ln["cause"], []).append(ln)
    done = []

    def walk(node, chain):
        nxt = [ln for ln in out_of.get(node, []) if ln["effect"] not in {c["cause"] for c in chain} | {start}]
        if len(chain) == hops or not nxt:
            if chain:
                done.append(chain)
            return
        for ln in nxt:
            walk(ln["effect"], chain + [ln])
    walk(start, [])
    return done


def table(chain: list[dict], fill: list[str]) -> tuple[list[str], np.ndarray]:
    """The engine's input: M node names and a successor array. The chain's nodes come first, each pointing
    to the next; the last and every filler node point to themselves."""
    nodes = [chain[0]["cause"]] + [ln["effect"] for ln in chain]
    if len(nodes) > M:
        raise ValueError(f"a chain of {len(nodes)} nodes does not fit the engine's {M}-item table")
    nodes += [f for f in fill if f not in nodes][:M - len(nodes)]
    nodes += [f"_idle{i}" for i in range(M - len(nodes))]
    succ = np.arange(M)
    succ[:len(chain)] = np.arange(1, len(chain) + 1)
    return nodes, succ


def encode(succ: np.ndarray, start: int, k: int) -> np.ndarray:
    """One question in the engine's 'every' format: the table one-hot at every step, the start at step 0."""
    X = np.zeros((1, k, M * M + M))
    T = np.zeros((M, M))
    T[np.arange(M), succ] = 1.0
    X[0, :, :M * M] = T.reshape(-1)
    X[0, 0, M * M + start] = 1.0
    return X


def _functions(rng, n, k):
    """Training questions over random tables. Tables are any mapping, not only permutations, since cause
    and effect chains merge and stop; the encoding is chainexact.batch's 'every' format."""
    succ = rng.integers(0, M, (n, M))
    s = rng.integers(0, M, n)
    ys, cur = [], s.copy()
    for _ in range(k):
        cur = succ[np.arange(n), cur]
        ys.append(cur.copy())
    T = np.zeros((n, M, M))
    T[np.arange(n)[:, None], np.arange(M)[None, :], succ] = 1.0
    X = np.zeros((n, k, M * M + M))
    X[:, :, :M * M] = T.reshape(n, -1)[:, None, :]
    X[np.arange(n), 0, M * M + s] = 1.0
    return X, np.stack(ys, 1)


def train(steps: int = 3000, seed: int = 10, log=print):
    """The combined bump-node tree with the snap, trained as chainsnap trains it (teacher forcing, state
    noise, Adam with clipping) on 1-3 step questions over random tables."""
    cs = _engine()
    ce = cs.ce
    net = cs.SnapNet("combined", np.random.default_rng(seed), ce.dx_of(M, "every"), M)
    rng, st = np.random.default_rng(700 + seed), {}
    for i in range(steps):
        X, Y = _functions(rng, ce.BS, int(rng.choice(ce.TRAIN_K)))
        net.run(X, Y, 0, ce.NOISE, rng)
        dl, loss = ce.step_losses(net, Y, True)
        ce.adam_clip(net.ps, net.backward(dl), st, ce.LR)
        if log and (i + 1) % 500 == 0:
            log(f"  step {i + 1}/{steps}: loss {loss:.4f}")
    return net


def accuracy(net, k: int, n: int = 1000, seed: int = 99) -> float:
    """Share of fresh random tables on which the engine lands on the right node after k steps."""
    X, Y = _functions(np.random.default_rng(seed + k), n, k)
    return float((net.run(X, None, 0).argmax(1) == Y[:, -1]).mean())


def model_path() -> Path:
    return config.home() / "models" / "chain" / f"snap_m{M}.npz"


def save(net, path=None) -> Path:
    p = Path(path or model_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, *net.ps)
    return p


def load(path=None, seed: int = 10):
    cs = _engine()
    net = cs.SnapNet("combined", np.random.default_rng(seed), cs.ce.dx_of(M, "every"), M)
    with np.load(Path(path or model_path())) as z:
        for p, k in zip(net.ps, sorted(z.files, key=lambda f: int(f.split("_")[1]))):
            p[...] = z[k]
    return net


def follow(net, links, start: str, hops: int = 4) -> list[dict]:
    """Each chain from ``start``: the engine's nodes step by step, the exact lookup, whether they agree, and
    the direction the chain implies for its last state (the product of the link signs)."""
    out = []
    names = sorted({ln["cause"] for ln in links} | {ln["effect"] for ln in links})
    for chain in paths(links, start, min(hops, M - 1)):
        nodes, succ = table(chain, names)
        k = len(chain)
        net.run(encode(succ, 0, k), None, 0)
        got = [nodes[int(i[0])] for i in net.items]
        want = [ln["effect"] for ln in chain]
        sign = int(np.prod([ln.get("sign", 1) for ln in chain]))
        out.append({"engine": got, "exact": want, "agree": got == want, "sign": sign,
                    "status": "untested" if any(ln.get("status") != "tested" for ln in chain) else "tested"})
    return out


# ---------- CLI ----------
def _cmd(a) -> int:
    links = load_links()
    if a.action == "links":
        for ln in links:
            print(f"{ln['cause']:<26} -> {ln['effect']:<26} {'+' if ln.get('sign', 1) > 0 else '-'}  {ln['status']}  ({ln['source']})")
        return 0
    if a.action == "train":
        net = train(a.steps)
        accs = {k: accuracy(net, k) for k in (1, 2, 3, 4, 6, 8, 16)}
        print("accuracy on fresh tables by steps: " + ", ".join(f"{k}: {v:.3f}" for k, v in accs.items()))
        print(f"saved {save(net)}")
        return 0
    if not model_path().exists():
        print("no trained engine yet: python -m brain chain train")
        return 1
    net = load()
    rows = follow(net, links, a.start, a.hops)
    if not rows:
        print(f"no links lead out of {a.start!r} (python -m brain chain links)")
        return 1
    for r in rows:
        path = " -> ".join([a.start] + r["engine"])
        print(f"{'agrees' if r['agree'] else 'DIFFERS from lookup'}: {path}   ({r['status']})")
        if not r["agree"]:
            print(f"   lookup: {' -> '.join([a.start] + r['exact'])}")
    print("The links are textbook knowledge; a chain says something about a stock only once its links are tested.")
    return 0


def add_cli(sub) -> None:
    s = sub.add_parser("chain", help="follow cause -> effect links with the chain engine (snap network from "
                                     "Nikku03/cell): train | links | follow START")
    s.add_argument("action", choices=["train", "links", "follow"])
    s.add_argument("start", nargs="?", default="oil_price_up")
    s.add_argument("--hops", type=int, default=4)
    s.add_argument("--steps", type=int, default=3000)
    s.set_defaults(fn=_cmd)
