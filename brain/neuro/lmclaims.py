"""Claims about how billion-parameter language models hold a number while they count, declared before they run.

The rules of ``claims`` (the N1 claims), adapted to one large model at a time:
- **Look on a small model, test on the big ones.** The probes, the thresholds and the layer rule were set by
  looking only at Pythia-410M (``DEV``). The claims are tested on Pythia-1B and Pythia-1.4B (``MODELS``), which
  nothing looked at inside; only that both count (a published capability) was assumed.
- **Pinned.** ``declare`` writes each claim with the Hugging Face commit of the model and of its untrained
  ``step0`` checkpoint, and the sha of this code; ``test`` refuses anything else, and refuses a declaration that
  is not in HEAD's ledger.
- **One layer per model, chosen before any claim is measured:** the layer where pasting another prompt's real
  state at the number works best, on a pairing (k -> k + 61) that no claim uses. Every claim reads that layer.
- **Per claim:** the measure clears its threshold and beats its null at p < 0.05; the controls behave; the model
  counts (accuracy >= 0.95, else NO VERDICT); and the BH q over every claim in research/neuro_ledger.jsonl (N1 and
  N2 together, withdrawn ones included) is below 0.10. Its p is the largest of its null p-values.
- ``helix-write`` is expected to FAIL: on the dev model the helix shift moved the model off its answer but almost
  never onto m + 1. It is declared so that the negative is on the record.
"""
from __future__ import annotations

import gc
import hashlib

import numpy as np

from brain import config
from brain.bank import canon
from brain.neuro import claims as n1
from brain.neuro import lm as L
from brain.research import ledger
from brain.research.stats import bh_adjust, binom_one_sided_greater
from brain.timeutil import now_iso

VERSION = "N2"
DEV = ["EleutherAI/pythia-410m"]
MODELS = ["EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b"]
N_SHUFFLE, N_RANDOM, N_RANDOM_FIT = 200, 50, 200   # N_RANDOM_FIT: the fit and read nulls, which need no forward pass
Q = 0.10
CODE = ["brain/neuro/lm.py", "brain/neuro/lmclaims.py"]
LEARNT = {"counting_acc_min": 0.95}

CLAIMS = [
    {"id": "helix-fit", "threshold": 0.2,
     "claim": "At the chosen layer, the state at the number token is a helix in the number: the basis [n, waves of "
              "period 2, 5, 10, 100] predicts the states of numbers held out of the fit with R^2 >= 0.2.",
     "null": f"the same fit with the numbers shuffled ({N_SHUFFLE} draws) and with 4 other periods ({N_RANDOM_FIT} sets)",
     "positive": "planted helix states", "negative": "the untrained step0 checkpoint, same layer"},
    {"id": "helix-read", "threshold": 0.9,
     "claim": "The number is in that helix exactly: projected onto the helix directions of a fit that never saw it, "
              "a number's state decodes to that number for >= 90% of numbers.",
     "null": f"the same with 4 other periods ({N_RANDOM_FIT} sets)",
     "positive": "planted helix states", "negative": "the untrained step0 checkpoint, same layer"},
    {"id": "locate", "threshold": 0.9,
     "claim": "The model reads the number from that token at that layer: with the state there replaced by the state "
              "another prompt had at its number m, it continues with m + 1 on >= 90% of prompts.",
     "null": "binomial against 1/100",
     "positive": None, "negative": "the same patch at the last layer (the number has moved on)"},
    {"id": "helix-steer", "threshold": 0.5,
     "claim": "The model uses the helix: moving only the helix part of a prompt's state from its number n to m (from "
              "a fit that never saw m) recovers >= half of the logit gap toward m + 1 that the real state of m opens "
              "(median over prompts), more than moving along 4 other periods.",
     "null": f"the same shift along 4 other periods ({N_RANDOM} sets)",
     "positive": None, "negative": "the same shift along a helix fitted to shuffled numbers"},
    {"id": "helix-write", "threshold": 0.5,
     "claim": "The helix is the whole number: that shift alone makes the model write exactly m + 1 on >= 50% of "
              "prompts. (Expected to fail; see the module docstring.)",
     "null": f"the same shift along 4 other periods ({N_RANDOM} sets)",
     "positive": None, "negative": "the same shift along a helix fitted to shuffled numbers"},
]


def short(model: str) -> str:
    return model.split("/")[-1].replace("pythia-", "pythia")


def claim_id(c: dict, model: str) -> str:
    return f"{VERSION}-{c['id']}-{short(model)}"


def code_sha() -> str:
    h = hashlib.sha256()
    for f in CODE:
        h.update(f.encode() + b"\0" + (config.ROOT / f).read_bytes())
    return h.hexdigest()[:16]


def spec(c: dict, model: str, shas: dict) -> dict:
    return {**c, "version": VERSION, "model": model, "sha": shas["main"], "untrained_sha": shas["step0"],
            "dev": DEV, "task": "'{c-2}, {c-1}, {c},' -> ' {c+1}', c = 2..101", "periods": list(L.PERIODS),
            "pool": list(L.POOL), "pair": L.PAIR, "pair_layer": L.PAIR_LAYER, "learnt": LEARNT,
            "n_shuffle": N_SHUFFLE, "n_random": N_RANDOM, "n_random_fit": N_RANDOM_FIT, "bh_q": Q, "code_sha": code_sha()}


def spec_sha(s: dict) -> str:
    return hashlib.sha256(canon(s).encode()).hexdigest()[:16]


def declare(trial: bool = False, log=print) -> list[dict]:
    led, _ = n1.paths(trial)
    if not trial:
        dirty = n1._git("status", "--porcelain", "--", *CODE).stdout.strip()
        if dirty:
            raise n1.Refused("commit brain/neuro/lm.py and lmclaims.py first; a declaration pins them:\n" + dirty)
    have = {r["id"]: r for r in ledger.read(led) if r["op"] == "declare"}
    new = []
    for model in DEV if trial else MODELS:
        shas = {"main": L.pinned_sha(model), "step0": L.pinned_sha(model, "step0")}
        for c in CLAIMS:
            s, i = spec(c, model, shas), claim_id(c, model)
            if i in have:
                if have[i]["spec_sha"] != spec_sha(s):
                    raise n1.Refused(f"{i} was declared with a different spec, model commit or code; declared claims "
                                     f"are frozen: raise VERSION in brain/neuro/lmclaims.py")
                continue
            rec = {"op": "declare", "kind": "neuro-lm", "id": i, "spec_sha": spec_sha(s), "spec": s,
                   "at": now_iso(), **ledger.git_state()}
            ledger.append(rec, led)
            new.append(rec)
            if log:
                log(f"declared {i} ({rec['spec_sha']}) on {model} @ {shas['main'][:10]}")
    return new


def _p_ge(obs: float, null) -> float:
    return (1 + sum(x >= obs for x in null)) / (1 + len(null))


def measure_model(model: str, sha: str, untrained_sha: str, log=print) -> dict:
    """Every claim's numbers for one model, its untrained checkpoint and the planted control."""
    lm = L.LM(model, "main", sha)
    if log:
        log(f"{model}: {lm.params / 1e9:.2f}B parameters, {lm.n_layers} layers, counting accuracy {lm.accuracy():.3f}")
    layer, by_layer = L.choose_layer(lm)
    H = lm.H[layer]
    rnd = L.random_periods(N_RANDOM, seed=11)
    rnd_fit = L.random_periods(N_RANDOM_FIT, seed=14)
    rng = np.random.default_rng(12)
    out = {"model": model, "params": lm.params, "layers": lm.n_layers, "counting_acc": lm.accuracy(),
           "layer": layer, "patch_by_layer": by_layer}
    r2 = L.helix_fit(H)
    out["helix-fit"] = {"stat": r2, "p_shuffle": _p_ge(r2, [L.helix_fit(H, labels=rng.permutation(100))
                                                        for _ in range(N_SHUFFLE)]),
                        "p_random": _p_ge(r2, [L.helix_fit(H, T) for T in rnd_fit])}
    rd = L.helix_read(H)
    out["helix-read"] = {"stat": rd, "p_random": _p_ge(rd, [L.helix_read(H, T) for T in rnd_fit])}
    loc = L.patch(lm, layer)
    out["locate"] = {"stat": loc, "p_binom": binom_one_sided_greater(round(loc * 100), 100, 0.01),
                     "last_layer": L.patch(lm, lm.n_layers - 1)}
    st = L.steer(lm, layer)
    null = [L.steer(lm, layer, T) for T in rnd]
    shuf = L.steer(lm, layer, labels=rng.permutation(100))
    out["helix-steer"] = {"stat": st["restored"], "stays": st["stays"],
                          "p_random": _p_ge(st["restored"], [x["restored"] for x in null]),
                          "shuffled": shuf["restored"], "random_median": float(np.median([x["restored"] for x in null]))}
    out["helix-write"] = {"stat": st["writes"], "p_random": _p_ge(st["writes"], [x["writes"] for x in null]),
                          "shuffled": shuf["writes"], "random_median": float(np.median([x["writes"] for x in null]))}
    out["amplitudes"] = L.amplitudes(H)
    b = L.bank()
    L.record(b, lm, {"layer/chosen": {"layer": layer, "patch": by_layer[layer - 1]},
                     "helix/fit": {"r2": r2}, "helix/read": {"acc": rd}, "helix/steer": st,
                     "helix/amplitudes": out["amplitudes"]})
    d = H.shape[1]
    del lm, H
    gc.collect()
    out["planted"] = planted_control(d)
    u = L.LM(model, "step0", untrained_sha)
    Hu = u.H[layer]
    out["untrained"] = {"counting_acc": u.accuracy(), "helix-fit": L.helix_fit(Hu), "helix-read": L.helix_read(Hu)}
    L.record(b, u, {"helix/fit": {"r2": out["untrained"]["helix-fit"]}, "helix/read": {"acc": out["untrained"]["helix-read"]}})
    del u, Hu
    gc.collect()
    if log:
        log(f"  layer {layer}; helix R^2 {r2:.3f}, read {rd:.2f}, locate {loc:.2f}, steer {st['restored']:.2f}, "
            f"write {st['writes']:.2f}; untrained R^2 {out['untrained']['helix-fit']:.3f}")
    return out


def planted_control(d: int) -> dict:
    """The fit and read claims on planted helix states of the model's width, with their own nulls."""
    P = L.planted_states(d, seed=13)
    r2, rd = L.helix_fit(P), L.helix_read(P)
    rnd = L.random_periods(N_RANDOM_FIT, seed=14)
    return {"helix-fit": r2, "helix-read": rd, "helix-fit_p": _p_ge(r2, [L.helix_fit(P, T) for T in rnd])}


def judge(c: dict, m: dict) -> dict:
    """One claim's result from a model's measurements."""
    cid, th = c["id"], c["threshold"]
    x = m[cid]
    learnt = m["counting_acc"] >= LEARNT["counting_acc_min"]
    if cid == "helix-fit":
        p = max(x["p_shuffle"], x["p_random"])
        ctl = {"positive": m["planted"]["helix-fit"] >= th and m["planted"]["helix-fit_p"] < 0.05,
               "negative": not (m["untrained"]["helix-fit"] >= th)}
    elif cid == "helix-read":
        p = x["p_random"]
        ctl = {"positive": m["planted"]["helix-read"] >= th, "negative": not (m["untrained"]["helix-read"] >= th)}
    elif cid == "locate":
        p = x["p_binom"]
        ctl = {"positive": True, "negative": x["last_layer"] < th}
    else:
        p = x["p_random"]
        ctl = {"positive": True, "negative": x["shuffled"] < th}
    gate = x["stat"] >= th and p < 0.05
    return {"n": 1, "learnt": int(learnt), "passes": int(gate), "p": p, "median_stat": x["stat"],
            "min_stat": x["stat"], "rows": [{"model": m["model"], "layer": m["layer"], **x}],
            "controls": {**ctl, "pass": all(ctl.values())}, "gate": {"pass": gate}, "model": m["model"],
            "layer": m["layer"]}


def verdicts(path) -> dict[str, dict]:
    """N2 rows with their verdicts; the BH q runs over every result in the neuro ledger, N1 included."""
    res = ledger.latest_results(path)
    gone = ledger.withdrawn(path)
    q_of = dict(zip(res, bh_adjust([r["p"] for r in res.values()]))) if res else {}
    out = {}
    for k, r in res.items():
        if r.get("kind") != "neuro-lm":
            continue
        q = q_of[k]
        if k in gone:
            v = f"WITHDRAWN: {gone[k]}"
        elif not r["learnt"]:
            v = "NO VERDICT (the model does not count)"
        elif not r["controls"]["pass"]:
            v = "NO VERDICT (controls failed)"
        else:
            v = "PASS" if r["gate"]["pass"] and q < Q else "FAIL"
        out[k] = {**r, "bh_q": q, "verdict": v}
    return out


def test(trial: bool = False, log=print) -> list[dict]:
    led, res_md = n1.paths(trial)
    recs = ledger.read(led)
    decls = {r["id"]: r for r in recs if r["op"] == "declare" and r.get("kind") == "neuro-lm"}
    done = {r["id"] for r in recs if r["op"] == "result"}
    want = [i for i in decls if i not in done]
    if not want:
        if log:
            log("nothing to test: every declared N2 claim has a result")
        return []
    pinned = n1.committed()
    for i in want:
        if not trial and (i, decls[i]["spec_sha"]) not in pinned:
            raise n1.Refused(f"{i}: its declaration is not in HEAD's research/neuro_ledger.jsonl; commit it first")
        if decls[i]["spec"]["code_sha"] != code_sha():
            raise n1.Refused(f"{i}: the code changed since it was declared; declared claims are frozen")
    out = []
    for model in dict.fromkeys(decls[i]["spec"]["model"] for i in want):
        s = next(decls[i]["spec"] for i in want if decls[i]["spec"]["model"] == model)
        m = measure_model(model, s["sha"], s["untrained_sha"], log)
        for c in CLAIMS:
            i = claim_id(c, model)
            if i not in want:
                continue
            rec = {"op": "result", "kind": "neuro-lm", "id": i, "spec_sha": decls[i]["spec_sha"], **judge(c, m),
                   "measurements": {k: v for k, v in m.items() if k in ("layer", "patch_by_layer", "amplitudes",
                                                                         "planted", "untrained", "counting_acc")},
                   "at": now_iso(), "trial": trial, **ledger.git_state()}
            ledger.append(rec, led)
            out.append(rec)
    res_md.write_text(n1.report(led) + report(led))
    return out


def report(path) -> str:
    vs = verdicts(path)
    if not vs:
        return ""
    L_ = ["", "# How billion-parameter models hold a number while they count", "",
          ("Claims declared before they ran on Pythia-1B and Pythia-1.4B, after looking only at Pythia-410M "
           "(`brain/neuro/lmclaims.py`, `docs/NEURO.md`). The BH q is shared with the claims above."), "",
          "| Claim | Model | Layer | Measure | Threshold | p | BH q | Controls | Verdict |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in vs.items():
        c = next(x for x in CLAIMS if k.startswith(f"{VERSION}-{x['id']}-"))
        ctl = "ok" if r["controls"]["pass"] else ", ".join(n for n in ("positive", "negative") if not r["controls"][n]) + " failed"
        L_.append(f"| {k} | {r['model'].split('/')[-1]} | {r['layer']} | {r['median_stat']:.3f} | {c['threshold']} | "
                  f"{r['p']:.2g} | {r['bh_q']:.2g} | {ctl} | {r['verdict']} |")
    L_ += ["", "## The claims", ""]
    L_ += [f"- **{c['id']}**: {c['claim']} Null: {c['null']}. Controls: positive {c['positive'] or 'none'}; "
           f"negative {c['negative']}." for c in CLAIMS]
    L_ += ["", "## Measurements", ""]
    for model in dict.fromkeys(r["model"] for r in vs.values()):
        r = next(x for x in vs.values() if x["model"] == model)
        ms = r["measurements"]
        steer = next(x for x in vs.values() if x["model"] == model and x["id"].startswith(f"{VERSION}-helix-steer"))
        write = next(x for x in vs.values() if x["model"] == model and x["id"].startswith(f"{VERSION}-helix-write"))
        sr, wr = steer["rows"][0], write["rows"][0]
        L_.append(f"- {model.split('/')[-1]}: counting accuracy {ms['counting_acc']:.3f}; layer {ms['layer']} "
                  f"(real-state patch by layer: {', '.join(f'{a:.2f}' for a in ms['patch_by_layer'])}). "
                  f"Helix shift: recovers {sr['stat']:.2f} of the gap (random periods median {sr['random_median']:.2f}, "
                  f"shuffled helix {sr['shuffled']:.2f}), keeps the original answer on {sr['stays']:.2f}, writes m+1 on "
                  f"{wr['stat']:.2f} (random periods {wr['random_median']:.2f}). Untrained checkpoint: counting "
                  f"{ms['untrained']['counting_acc']:.3f}, helix R^2 {ms['untrained']['helix-fit']:.3f}, read "
                  f"{ms['untrained']['helix-read']:.2f}. Planted: R^2 {ms['planted']['helix-fit']:.3f}, read "
                  f"{ms['planted']['helix-read']:.2f}. Fitted size per part: "
                  + ", ".join(f"{k} {v:.1f}" for k, v in ms["amplitudes"].items()) + ".")
    return "\n".join(L_) + "\n"
