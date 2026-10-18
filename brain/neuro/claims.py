"""Claims about what is inside a network, declared before they run (docs/NEURO.md).

The research ledger's rules, applied to networks:
- **Look on dev networks, test on fresh ones.** The probes and every threshold below were set by looking at one
  network per subject (``DEV``: modular addition seed 0, the chain engine seed 10). Each claim is then tested on
  ten networks trained from seeds nothing looked at (``SEEDS``).
- **Declared and committed first.** ``declare`` writes each claim's full spec, with the sha of the code that will
  run it, to ``research/neuro_ledger.jsonl``; ``test`` refuses a claim whose declaration is not in HEAD's ledger or
  whose code has changed since. A declared claim is never edited: a change is a new ``VERSION``, so new ids.
- **Controls.** Every claim has a positive control (a hand-built network with the mechanism; the instrument must
  find it) and a negative one (a network without it; the instrument must not). A claim whose controls fail gets
  no verdict.
- **The verdict.** Per network, the claim's measure must clear its threshold and, where it has a null, beat the
  null at p < 0.05. A claim PASSES when at least 9 of its 10 networks learnt the task (else NO VERDICT), at least
  90% of those pass, the controls pass, and the Benjamini-Hochberg q over every claim in this ledger, withdrawn
  ones included, is below 0.10. Its p is the chance of that many passing networks if each passed with
  probability one half.

These claims are about networks, not markets, and they keep their own false-discovery family: putting near-zero
p-values into the market family would loosen every market hypothesis's q.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
from pathlib import Path

import numpy as np

from brain import config
from brain.bank import canon
from brain.neuro import memory, probe
from brain.neuro import subjects as S
from brain.research import ledger
from brain.research.stats import bh_adjust, binom_one_sided_greater
from brain.timeutil import now_iso

VERSION = "N1"
DEV = {"modadd": [0], "chain": [10]}
SEEDS = {"modadd": list(range(1, 11)), "chain": list(range(30, 40))}
N_NULL = 200
TRANSFER = {"p": 29, "hidden": 96}
Q = 0.10
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
CODE = ["brain/neuro/subjects.py", "brain/neuro/probe.py", "brain/neuro/memory.py", "brain/neuro/claims.py",
        "brain/chain/__init__.py", "brain/chain/engine/chainexact.py", "brain/chain/engine/chainsnap.py"]

CLAIMS = [
    {"id": "one-frequency", "subject": "modadd",
     "claim": "Each hidden neuron of a modular-addition network that generalises reads a and b through one "
              "frequency: the median share of a neuron's non-constant Fourier power in its top frequency is at "
              "least 0.8 in both U and V.",
     "measure": "min(median top-frequency share of U rows, of V rows)", "threshold": 0.8,
     "null": "each neuron's weights shuffled over the inputs (200 draws)",
     "positive": "planted", "negative": ["memorizer", "untrained"]},
    {"id": "same-frequency", "subject": "modadd",
     "claim": "A neuron's output weights use the same frequency as its inputs: at least 90% of neurons have one "
              "top frequency in U, V and W.",
     "measure": "share of neurons whose top frequency agrees in U, V and W", "threshold": 0.9,
     "null": "W's neurons paired with other neurons' U and V (200 draws)",
     "positive": "planted", "negative": ["memorizer", "untrained"]},
    {"id": "phase-sum", "subject": "modadd",
     "claim": "A neuron's output phase is the sum of its two input phases (the identity that turns cos(a)cos(b) "
              "into cos(a+b)): the mean resultant length of phase_W - phase_U - phase_V is at least 0.9.",
     "measure": "mean resultant length over neurons with one frequency in U, V and W", "threshold": 0.9,
     "null": "output phases shuffled among those neurons (200 draws)",
     "positive": "planted", "negative": ["untrained"]},
    {"id": "faithful", "subject": "modadd",
     "claim": "That description is the computation: with every neuron cut down to its one top-frequency cosine "
              "in U, V and W, the network gives its own original answer on at least 95% of all p*p pairs.",
     "measure": "agreement of the projected network with the original on every pair", "threshold": 0.95,
     "null": "each neuron cut down to a different, random frequency (200 draws)",
     "positive": "planted", "negative": ["memorizer"]},
    {"id": "hidden-circuit", "subject": "memorizer",
     "claim": "A network that memorised (train accuracy >= 0.99, held-out < 0.5) already holds the general "
              "algorithm: cut down to each neuron's top-frequency cosine it is right on at least 95% of all pairs.",
     "measure": "accuracy of the projected memorizer on every pair", "threshold": 0.95,
     "null": "each neuron cut down to a different, random frequency (200 draws)",
     "positive": "planted", "negative": ["untrained"]},
    {"id": "rebuild", "subject": "modadd",
     "claim": "A network written from the bank's facts alone (each neuron's recorded frequency, input phases and "
              "amplitudes; output weights from the recorded phase law, never copied) is right on at least 99% "
              "of all pairs.",
     "measure": "accuracy of the rebuilt network on every pair", "threshold": 0.99,
     "null": None, "positive": "planted", "negative": ["untrained", "wrong-law"]},
    {"id": "transfer", "subject": "modadd",
     "claim": "The recorded rule alone (one frequency per neuron, the phase law), with no per-neuron facts, "
              "builds a network for modulus 29, which nothing was trained on, that is right on at least 99% "
              "of its 841 pairs.",
     "measure": "accuracy of the transferred network on every mod-29 pair", "threshold": 0.99,
     "null": None, "positive": "planted", "negative": ["untrained"]},
    {"id": "chain-pointer", "subject": "chain",
     "claim": "The chain engine's network carries a chain in the written-down pointer, not in its continuous "
              "memory: with the memory replaced by its mean at every step its 16-step accuracy stays >= 0.99, "
              "and with the pointer replaced by its mean it falls to <= 0.5.",
     "measure": "16-step accuracy, memory ablated (>= 0.99) and pointer ablated (<= 0.5)", "threshold": 0.99,
     "null": None, "positive": "planted_lookup", "negative": ["planted_memory"]},
    {"id": "chain-lookup", "subject": "chain",
     "claim": "Each step is a lookup of the pointer's row: with the pointer switched to item j's code at step 3 "
              "of 6, the network writes table[j] and carries on along the chain from j on >= 95% of questions.",
     "measure": "share of patched questions that write table[j] and then follow the chain from it",
     "threshold": 0.95, "null": "binomial against 1/6, chance for a step that ignores the patch",
     "positive": "planted_lookup", "negative": ["planted_memory"]},
    {"id": "chain-rebuild", "subject": "nomemory",
     "claim": "What the probes found predicts a simpler network: the chain engine's network rebuilt without its "
              "continuous memory (the tree sees only the table and the pointer), trained the same way on 1-3 "
              "step questions, is exact: accuracy >= 0.99 at 16 and at 64 steps.",
     "measure": "min(accuracy at 16 steps, at 64 steps)", "threshold": 0.99,
     "null": None, "positive": "trained", "negative": ["nomemory-pointer-ablated"]},
]
GATE = {"min_learnt": 9, "pass_share": 0.9, "null_p_lt": 0.05, "bh_q": Q}
LEARNT = {"modadd": "held-out accuracy >= 0.99", "memorizer": "train accuracy >= 0.99 and held-out < 0.5",
          "chain": "16-step accuracy >= 0.99", "nomemory": "1-3 step accuracy >= 0.90 (chainsnap's rule)"}


def claim_id(c: dict) -> str:
    return f"{VERSION}-{c['id']}"


def paths(trial: bool = False) -> tuple[Path, Path]:
    """The committed ledger and results, or scratch copies for a trial (dev seeds, never the record)."""
    if trial:
        d = config.home() / "neuro" / "trial"
        d.mkdir(parents=True, exist_ok=True)
        return d / "neuro_ledger.jsonl", d / "NEURO.md"
    return (Path(os.environ.get("BRAIN_NEURO_LEDGER") or config.ROOT / "research" / "neuro_ledger.jsonl"),
            Path(os.environ.get("BRAIN_NEURO_RESULTS") or config.ROOT / "research" / "NEURO.md"))


def code_sha() -> str:
    h = hashlib.sha256()
    for f in CODE:
        h.update(f.encode() + b"\0" + (config.ROOT / f).read_bytes())
    return h.hexdigest()[:16]


def spec(c: dict) -> dict:
    return {**c, "version": VERSION, "seeds": SEEDS, "dev": DEV, "n_null": N_NULL, "gate": GATE,
            "learnt": LEARNT, "subjects": {"modadd": S.MODADD, "memorizer": S.MEMORIZER, "chain": S.CHAIN},
            "transfer": TRANSFER, "code_sha": code_sha()}


def spec_sha(s: dict) -> str:
    return hashlib.sha256(canon(s).encode()).hexdigest()[:16]


class Refused(SystemExit):
    pass


def _git(*args):
    return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=False)


def committed() -> set[tuple[str, str]]:
    r = _git("show", "HEAD:research/neuro_ledger.jsonl")
    out = set()
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("op") == "declare":
            out.add((rec["id"], rec["spec_sha"]))
    return out


def declare(trial: bool = False, log=print) -> list[dict]:
    led, _ = paths(trial)
    if not trial:
        dirty = _git("status", "--porcelain", "--", *CODE).stdout.strip()
        if dirty:
            raise Refused("commit brain/neuro and brain/chain first; a declaration pins the committed code:\n" + dirty)
    have = {r["id"]: r for r in ledger.read(led) if r["op"] == "declare"}
    new = []
    for c in CLAIMS:
        s = spec(c)
        i = claim_id(c)
        if i in have:
            if have[i]["spec_sha"] != spec_sha(s):
                raise Refused(f"{i} was declared with a different spec (or the code changed since); declared "
                              f"claims are frozen: raise VERSION in brain/neuro/claims.py for new ids")
            continue
        rec = {"op": "declare", "kind": "neuro", "id": i, "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
               **ledger.git_state()}
        ledger.append(rec, led)
        new.append(rec)
        if log:
            log(f"declared {i} ({rec['spec_sha']})")
    return new


# ---------- networks ----------
def _job(args):
    """Train one network in a worker; return what pickles (weights and splits, not closures)."""
    kind, seed = args
    if kind in ("trained", "memorizer", "untrained"):
        spec_ = {"trained": S.MODADD, "memorizer": S.MEMORIZER, "untrained": {**S.MODADD, "steps": 0}}[kind]
        n = S.train_modadd(seed, spec_, kind)
        return kind, seed, ("modadd", [n.U, n.V, n.W], n.split, spec_)
    s = S.train_chain(seed, memory=(kind == "chain"))
    return kind, seed, ("chain", s.weights, None, s.spec)


def _revive(kind, seed, packed):
    task, ws, split, spec_ = packed
    if task == "modadd":
        return S.ModAdd(*ws, kind, spec_, seed, split)
    from brain import chain
    cs = chain._engine()
    memory_ = kind == "chain"
    net = cs.SnapNet("combined", np.random.default_rng(seed), cs.ce.dx_of(chain.M, "every"), chain.M)
    for p, w in zip(net.ps, ws):
        p[...] = w
    return S.ChainSubject("trained" if memory_ else "nomemory", net.H, net.Emb.shape[1], net.Emb,
                          S._snap_step(net, memory_), spec_, seed, net.ps)


def networks(seeds: dict, log=print, procs: int = 4) -> dict:
    jobs = [(k, s) for k in ("trained", "memorizer", "untrained") for s in seeds["modadd"]]
    jobs += [(k, s) for k in ("chain", "nomemory") for s in seeds["chain"]]
    if log:
        log(f"training {len(jobs)} networks on {procs} processes")
    # One BLAS thread per worker: the engine trains small matrices, and four processes each running a thread
    # pool on four cores are many times slower than one thread each. Spawned workers read this at numpy import.
    saved = {v: os.environ.get(v) for v in THREAD_VARS}
    os.environ.update({v: "1" for v in THREAD_VARS})
    try:
        with mp.get_context("spawn").Pool(procs) as pool:
            out = pool.map(_job, jobs, chunksize=1)
    finally:
        for v, x in saved.items():
            if x is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = x
    nets: dict = {}
    for kind, seed, packed in out:
        nets.setdefault(kind, {})[seed] = _revive(kind, seed, packed)
    return nets


# ---------- per-network measures ----------
def _learnt(kind, net) -> bool:
    if kind == "trained":
        return net.heldout() >= 0.99
    if kind == "memorizer":
        return net.accuracy(net.split["train"]) >= 0.99 and net.heldout() < 0.5
    if kind == "chain":
        return net.accuracy(16) >= 0.99
    if kind == "nomemory":
        return min(net.accuracy(k) for k in (1, 2, 3)) >= 0.90
    return True


def measure(cid: str, net, bank, seed: int) -> dict:
    """One claim on one network: its statistic, its null p (if it has one), and whether it passes."""
    c = next(x for x in CLAIMS if x["id"] == cid)
    th = c["threshold"]
    if cid == "one-frequency":
        st = probe.sparse_stat(net)
        p = probe.perm_p(st, probe.null_sparse(net, N_NULL, seed))
        return {"stat": st, "null_p": p, "pass": st >= th and p < GATE["null_p_lt"]}
    if cid == "same-frequency":
        sp = probe.neurons(net)
        st = float(probe.same_freq(sp).mean())
        p = probe.perm_p(st, probe.null_same_freq(sp, N_NULL, seed))
        return {"stat": st, "null_p": p, "pass": st >= th and p < GATE["null_p_lt"]}
    if cid == "phase-sum":
        sp = probe.neurons(net)
        mask = probe.same_freq(sp)
        st = probe.law_strength(sp, "sum", mask)
        p = probe.perm_p(st, probe.null_law(sp, mask, N_NULL, seed)) if mask.sum() > 1 else 1.0
        return {"stat": st, "null_p": p, "diff_law": probe.law_strength(sp, "diff", mask), "neurons": int(mask.sum()),
                "pass": st >= th and p < GATE["null_p_lt"]}
    if cid in ("faithful", "hidden-circuit"):
        fq = probe.neurons(net)["u"]["freq"]
        pj = probe.project(net, fq)
        p_ = net.p
        a, b = np.divmod(np.arange(p_ * p_), p_)
        if cid == "faithful":
            orig = net.logits(a, b).argmax(0)
            st = float((pj.logits(a, b).argmax(0) == orig).mean())
            null = []
            rng = np.random.default_rng(seed)
            F = (p_ - 1) // 2
            for _ in range(N_NULL):
                other = (fq - 1 + rng.integers(1, F, len(fq))) % F + 1
                null.append(float((probe.project(net, other).logits(a, b).argmax(0) == orig).mean()))
        else:
            st = pj.accuracy()
            null = probe.null_project(net, fq, N_NULL, seed)
        p = probe.perm_p(st, null)
        return {"stat": st, "null_p": p, "null_max": float(max(null)), "original_acc": net.accuracy(),
                "pass": st >= th and p < GATE["null_p_lt"]}
    if cid in ("rebuild", "transfer"):
        ent = memory.record_modadd(bank, net)
        try:
            built = (memory.rebuild(bank, ent) if cid == "rebuild"
                     else memory.transfer(bank, ent, TRANSFER["p"], TRANSFER["hidden"], seed))
        except memory.NoMechanism as e:
            return {"stat": 0.0, "refused": str(e), "entity": ent, "pass": False}
        st = built.accuracy()
        out = {"stat": st, "entity": ent, "cites": built.spec["cites"], "pass": st >= th}
        if cid == "rebuild":
            out["numbers_used"] = int(len(net.U) * 6)
            out["weights_replaced"] = int(sum(w.size for w in net.weights))
        return out
    if cid == "chain-pointer":
        ent = memory.record_chain(bank, net, units=False)
        ab = {k: memory.latest(bank, ent, f"ablate/{k}")[f"ablate/{k}"].value["acc_16"] for k in ("none", "memory", "pointer")}
        return {"stat": ab["memory"], "pointer_ablated": ab["pointer"], "unablated": ab["none"], "entity": ent,
                "pass": ab["memory"] >= th and ab["pointer"] <= 0.5}
    if cid == "chain-lookup":
        ent = memory.record_chain(bank, net, units=False)
        pt = memory.latest(bank, ent, "patch/pointer")["patch/pointer"].value
        k = round(pt["carries"] * pt["n"])
        p = binom_one_sided_greater(k, pt["n"], 1 / S.ChainSubject.M)
        return {"stat": pt["carries"], "follows": pt["follows"], "stays": pt["stays"], "null_p": p, "entity": ent,
                "pass": pt["carries"] >= th and p < GATE["null_p_lt"]}
    if cid == "chain-rebuild":
        a16, a64 = net.accuracy(16), net.accuracy(64)
        return {"stat": min(a16, a64), "acc_16": a16, "acc_64": a64, "pass": min(a16, a64) >= th}
    raise KeyError(cid)


def _control(cid: str, name: str, nets: dict, bank, seeds) -> dict:
    """Run the claim on a control network for each seed; ``passes`` is how many cleared the claim."""
    rows = []
    for s in seeds:
        if name == "planted":
            net = S.planted_modadd(s)
        elif name in ("planted_lookup", "planted_memory"):
            net = getattr(S, name)()
        elif name == "trained":
            net = nets["chain"][s]
        elif name == "nomemory-pointer-ablated":
            n = nets["nomemory"][s]
            acc = probe.ablate(n, "e")
            rows.append({"seed": s, "stat": acc, "pass": acc >= 0.99})
            continue
        elif name == "wrong-law":
            ent = memory.record_modadd(bank, nets["trained"][s])
            try:
                acc = memory.rebuild(bank, ent, law="diff").accuracy()
            except memory.NoMechanism:
                acc = 0.0
            rows.append({"seed": s, "stat": acc, "pass": acc >= 0.99})
            continue
        else:
            net = nets[name][s]
        rows.append({"seed": s, **{k: v for k, v in measure(cid, net, bank, 1000 + s).items() if k != "cites"}})
        if name in ("planted_lookup", "planted_memory"):
            break   # hand-written and deterministic: one is every seed
    return {"name": name, "passes": sum(r["pass"] for r in rows), "n": len(rows), "rows": rows}


def evaluate(c: dict, nets: dict, bank, seeds: dict, log=print) -> dict:
    kind = {"modadd": "trained", "memorizer": "memorizer", "chain": "chain", "nomemory": "nomemory"}[c["subject"]]
    ss = seeds["chain" if kind in ("chain", "nomemory") else "modadd"]
    rows = []
    for s in ss:
        net = nets[kind][s]
        learnt = _learnt(kind, net)
        r = {"seed": s, "learnt": learnt}
        if learnt:
            r.update({k: v for k, v in measure(c["id"], net, bank, s).items()})
        rows.append(r)
    learnt = [r for r in rows if r["learnt"]]
    passes = sum(r["pass"] for r in learnt)
    pos = _control(c["id"], c["positive"], nets, bank, ss)
    negs = [_control(c["id"], n, nets, bank, ss) for n in c["negative"]]
    ctl_ok = (pos["passes"] >= math.ceil(GATE["pass_share"] * pos["n"])
              and all(n["n"] - n["passes"] >= math.ceil(GATE["pass_share"] * n["n"]) for n in negs))
    gate = len(learnt) >= GATE["min_learnt"] and passes >= math.ceil(GATE["pass_share"] * len(learnt))
    stats = [r["stat"] for r in learnt]
    out = {"n": len(rows), "learnt": len(learnt), "passes": passes,
           "p": binom_one_sided_greater(passes, len(learnt), 0.5) if learnt else 1.0,
           "median_stat": float(np.median(stats)) if stats else None,
           "min_stat": float(min(stats)) if stats else None, "rows": rows,
           "controls": {"positive": pos, "negative": negs, "pass": ctl_ok}, "gate": {"pass": gate}}
    if log:
        log(f"  {claim_id(c)}: {passes}/{len(learnt)} learnt networks pass (of {len(rows)}); median "
            f"{out['median_stat'] if stats else float('nan'):.3f}; controls {'ok' if ctl_ok else 'FAILED'}")
    return out


def verdicts(path=None) -> list[dict]:
    led = path or paths()[0]
    res = ledger.latest_results(led)
    gone = ledger.withdrawn(led)
    q_of = dict(zip(res, bh_adjust([r["p"] for r in res.values()]))) if res else {}
    rows = []
    for k, r in res.items():
        q = q_of[k]
        if k in gone:
            v = f"WITHDRAWN: {gone[k]}"
        elif r["learnt"] < GATE["min_learnt"]:
            v = f"NO VERDICT ({r['learnt']}/{r['n']} learnt)"
        elif not r["controls"]["pass"]:
            v = "NO VERDICT (controls failed)"
        else:
            v = "PASS" if r["gate"]["pass"] and q < Q else "FAIL"
        rows.append({**r, "bh_q": q, "verdict": v})
    return rows


def test(ids=None, trial: bool = False, log=print) -> list[dict]:
    led, res_md = paths(trial)
    recs = ledger.read(led)
    decls = {r["id"]: r for r in recs if r["op"] == "declare"}
    done = {r["id"] for r in recs if r["op"] == "result"}
    want = [i for i in (ids or decls) if ids or i not in done]
    if not want:
        if log:
            log("nothing to test: every declared claim has a result")
        return []
    pinned = committed()
    for i in want:
        if i not in decls:
            raise Refused(f"{i} is not declared: python -m brain neuro declare")
        if not trial and (i, decls[i]["spec_sha"]) not in pinned:
            raise Refused(f"{i}: its declaration is not in HEAD's research/neuro_ledger.jsonl; commit it first")
        if decls[i]["spec"]["code_sha"] != code_sha():
            raise Refused(f"{i}: the code changed since it was declared; declared claims are frozen")
    seeds = DEV if trial else SEEDS
    nets = networks(seeds, log)
    bank = memory.neuro_bank()
    out = []
    for i in want:
        c = next(x for x in CLAIMS if claim_id(x) == i)
        r = evaluate(c, nets, bank, seeds, log)
        rec = {"op": "result", "kind": "neuro", "id": i, "spec_sha": decls[i]["spec_sha"], **r, "at": now_iso(),
               "trial": trial, **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
    for s in seeds["chain"]:   # descriptive: every probe on each chain network, so its card is complete
        memory.record_chain(bank, nets["chain"][s])
    res_md.write_text(report(led))
    return out


def report(path=None) -> str:
    rows = verdicts(path)
    by = {r["id"]: r for r in rows}
    L = ["# What the engine found inside networks", "",
         ("Claims about two networks' insides, declared in `research/neuro_ledger.jsonl` before they ran and tested "
          "on networks trained from seeds nothing had looked at (`docs/NEURO.md`). Generated by "
          "`python -m brain neuro test`; do not edit by hand."), "",
         "| Claim | Networks passing | Median measure | p | BH q | Controls (+ / −) | Verdict |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in CLAIMS:
        r = by.get(claim_id(c))
        if r is None:
            L.append(f"| {claim_id(c)} | not run | | | | | |")
            continue
        ct = r["controls"]
        neg = ", ".join(f"{n['name']} {n['passes']}/{n['n']}" for n in ct["negative"])
        med = "–" if r["median_stat"] is None else f"{r['median_stat']:.3f}"
        L.append(f"| {claim_id(c)} | {r['passes']}/{r['learnt']} (of {r['n']}) | {med} | {r['p']:.2g} | "
                 f"{r['bh_q']:.2g} | {ct['positive']['name']} {ct['positive']['passes']}/{ct['positive']['n']}; "
                 f"{neg} | {r['verdict']} |")
    L += ["", ("Controls: the positive control must pass on at least 90% of its runs, each negative control must "
               "fail on at least 90% of its runs."), "", "## The claims", ""]
    for c in CLAIMS:
        L.append(f"- **{claim_id(c)}** ({c['subject']}): {c['claim']} Null: {c['null'] or 'none (a threshold)'}.")
    L += ["", "## Per network", ""]
    for c in CLAIMS:
        r = by.get(claim_id(c))
        if r is None:
            continue
        cells = []
        for x in r["rows"]:
            if not x["learnt"]:
                cells.append(f"seed {x['seed']}: not learnt")
                continue
            extra = f", null p {x['null_p']:.3f}" if "null_p" in x else ""
            extra += f", refused: {x['refused']}" if "refused" in x else ""
            cells.append(f"seed {x['seed']}: {x['stat']:.3f}{extra}{'' if x['pass'] else ' (fails)'}")
        L.append(f"- {claim_id(c)}: " + "; ".join(cells))
    return "\n".join(L) + "\n"
