"""Claims about how a ~30-billion-parameter model handles language, declared before they run (docs/NEURO.md).

Same rules as the N1 and N2 claims (``claims``, ``lmclaims``):
- **Look small, test big.** The probes, thresholds and layer rules were set by looking only at Qwen3-0.6B
  (``DEV``), on CPU. The claims are tested on a model of the same family that nothing looked at inside:
  Qwen3-30B-A3B (Ollama ``qwen3:30b``, bf16, needs an 80 GB GPU) or Qwen3-32B (Ollama ``qwen3:32b``, 4-bit NF4,
  fits a 40 GB GPU). Each is its own subject with its own ids; run whichever the GPU allows.
- **Three disjoint sets of passages:** ``SEL`` chooses each claim's layer (and the steering strength), ``TRAIN``
  gives the language means, ``TEST`` gives the verdict. Same sentences in every language.
- **Pinned:** the model's and the dataset's Hugging Face commits, the quantization and the code's sha are in each
  declaration; ``test`` refuses anything else, and a declaration not in HEAD's ledger.
- **Per claim:** the model must name the language of the TEST sentences (accuracy >= 0.9, else NO VERDICT); the
  measure clears its threshold and beats its null at p < 0.05; the negative control fails; and the BH q over
  every result in research/neuro_ledger.jsonl (N1, N2 and N3) is below 0.10.
"""
from __future__ import annotations

import gc
import hashlib

import numpy as np

from brain import config
from brain.bank import Bank, Fact
from brain.neuro import claims as n1
from brain.neuro import lang as G
from brain.neuro import lmclaims
from brain.research import ledger
from brain.timeutil import now_iso

VERSION = "N3"
DEV = ["Qwen/Qwen3-0.6B"]
MODELS = {"qwen3:30b": {"hf": "Qwen/Qwen3-30B-A3B", "quant": None, "gpu_gb": 75},
          "qwen3:32b": {"hf": "Qwen/Qwen3-32B", "quant": "nf4", "gpu_gb": 30}}
SEL, TRAIN, TEST = (0, 40), (40, 140), (140, 240)
TRIAL_SPLITS = ((0, 6), (6, 26), (26, 36))   # --trial only: small enough to exercise the code on a CPU
N_SHUFFLE, N_RANDOM, N_RANDOM_SPEAK = 200, 10, 5
SPEAK_FRACTIONS = (0.25, 0.375, 0.5, 0.625, 0.75)   # layers tried for speak, as a share of the depth
Q = 0.10
CODE = ["brain/neuro/lang.py", "brain/neuro/langclaims.py"]
LEARNT = {"naming_acc_min": 0.9}

CLAIMS = [
    {"id": "lang-read", "threshold": 0.95,
     "claim": "The model gathers the language into the last token: at the chosen layer, the state at the final ':' "
              "(the same token in every prompt) names the sentence's language by nearest language mean for at least "
              "the threshold share of TEST sentences, which share no content with the TRAIN ones.",
     "layer": "the earliest layer with the best SEL accuracy",
     "null": f"means from shuffled language labels ({N_SHUFFLE} draws)",
     "negative": "layer 0, the ':' token's own embedding"},
    {"id": "lang-swap", "threshold": 0.5,
     "claim": "That language part is what the model reads when it names the language: replacing it (state - mean of "
              "its language + mean of another) makes the model name the other language for at least the threshold "
              "share of TEST sentences.",
     "layer": "the layer with the best SEL swap rate (earliest on a tie)",
     "null": f"the same-size step in a random direction ({N_RANDOM} draws); p is one-sided binomial against their "
             f"pooled rate",
     "negative": "the swap with means from shuffled language labels"},
    {"id": "lang-shared", "threshold": 0.75,
     "claim": "Meaning and language are separate: with its language mean removed, a sentence's state (averaged over "
              "its own tokens) is nearest to its own translation, among the TEST sentences of each other language, "
              "for at least the threshold share (chance is 1 in 100).",
     "layer": "the layer with the best SEL retrieval",
     "null": f"translations shuffled ({N_SHUFFLE} draws)",
     "negative": "layer 0, the tokens' own embeddings"},
    {"id": "lang-speak", "threshold": 0.5,
     "claim": "Language can be written in: adding the difference between another language's mean and English's to "
              "every position while the model continues an English sentence makes the continuation that language "
              "for at least the threshold share of TEST sentences (French, German, Spanish, Russian, Chinese).",
     "layer": f"the (layer, strength) with the best SEL rate, layers at {SPEAK_FRACTIONS} of the depth, strengths "
              f"{G.ALPHAS}",
     "null": f"the same-size step in a random direction ({N_RANDOM_SPEAK} draws); p is one-sided binomial against "
             f"their pooled rate",
     "negative": "the step with means from shuffled language labels"},
]


def claim_id(c: dict, key: str) -> str:
    return f"{VERSION}-{c['id']}-{key.replace(':', '')}"


def code_sha() -> str:
    h = hashlib.sha256()
    for f in CODE:
        h.update(f.encode() + b"\0" + (config.ROOT / f).read_bytes())
    return h.hexdigest()[:16]


def spec(c: dict, key: str, sha: str) -> dict:
    m = MODELS.get(key, {"hf": key, "quant": None})
    return {**c, "version": VERSION, "ollama": key, "model": m["hf"], "quant": m["quant"], "sha": sha,
            "dataset": G.DATASET, "dataset_sha": G.DATASET_SHA, "langs": [x for x, _ in G.LANGS],
            "speak_targets": G.SPEAK, "splits": {"sel": SEL, "train": TRAIN, "test": TEST}, "dev": DEV,
            "prompt": G.PREFIX + "{sentence}" + G.SUFFIX, "alphas": list(G.ALPHAS), "new_tokens": G.NEW_TOKENS,
            "n_shuffle": N_SHUFFLE, "n_random": N_RANDOM, "n_random_speak": N_RANDOM_SPEAK, "learnt": LEARNT,
            "bh_q": Q, "code_sha": code_sha()}


def declare(trial: bool = False, log=print) -> list[dict]:
    led, _ = n1.paths(trial)
    if not trial:
        dirty = n1._git("status", "--porcelain", "--", *CODE).stdout.strip()
        if dirty:
            raise n1.Refused("commit brain/neuro/lang.py and langclaims.py first; a declaration pins them:\n" + dirty)
    have = {r["id"]: r for r in ledger.read(led) if r["op"] == "declare"}
    keys = DEV if trial else list(MODELS)
    new = []
    for key in keys:
        hf = MODELS[key]["hf"] if key in MODELS else key
        sha = lmclaims.L.pinned_sha(hf)
        for c in CLAIMS:
            s, i = spec(c, key, sha), claim_id(c, key)
            if i in have:
                if have[i]["spec_sha"] != lmclaims.spec_sha(s):
                    raise n1.Refused(f"{i} was declared with a different spec, model commit or code; declared claims "
                                     f"are frozen: raise VERSION in brain/neuro/langclaims.py")
                continue
            rec = {"op": "declare", "kind": "neuro-lang", "id": i, "spec_sha": lmclaims.spec_sha(s), "spec": s,
                   "at": now_iso(), **ledger.git_state()}
            ledger.append(rec, led)
            new.append(rec)
            if log:
                log(f"declared {i} ({rec['spec_sha']}) on {hf} @ {sha[:10]}")
    return new


# ---------- measuring ----------
def _split(P, rng_):
    a, b = rng_
    K = len(G.LANGS)
    sents = [P[c][i] for c, _ in G.LANGS for i in range(a, b)]
    return sents, np.repeat(np.arange(K), b - a)


def _p_ge(obs, null):
    return (1 + sum(x >= obs for x in null)) / (1 + len(null))


def p_against_pooled(rate: float, n: int, null_rates: list[float]) -> float:
    """One-sided binomial p of ``rate`` over ``n`` trials against the random draws' pooled rate (each draw also
    over ``n``), with one extra hit added to the null so a null that never hits is not taken as a rate of zero."""
    from brain.research.stats import binom_one_sided_greater
    p0 = (sum(round(r * n) for r in null_rates) + 1) / (len(null_rates) * n + 1)
    return binom_one_sided_greater(round(rate * n), n, p0)


def measure_model(key: str, sha: str, log=print, batch: int = 16, splits=None) -> dict:
    """Every claim's numbers for one model (``splits``: SEL, TRAIN, TEST; the declared ones unless a trial)."""
    SEL, TRAIN, TEST = splits or (globals()["SEL"], globals()["TRAIN"], globals()["TEST"])
    hf = MODELS[key]["hf"] if key in MODELS else key
    quant = MODELS[key]["quant"] if key in MODELS else None
    m = G.Model(hf, sha, quant)
    K = len(G.LANGS)
    P = G.passages(TEST[1])
    (ss, ys), (st, yt), (se, ye) = (_split(P, r) for r in (SEL, TRAIN, TEST))
    n_sel, n_test = SEL[1] - SEL[0], TEST[1] - TEST[0]
    if log:
        log(f"{hf}: {m.params / 1e9:.1f}B parameters, {m.n_layers} layers; states for {len(ss) + len(st) + len(se)} "
            f"sentences")
    Ls, Ps, _ = m.states(ss, batch)
    Lt, Pt, _ = m.states(st, batch)
    Le, Pe, Te = m.states(se, batch)
    naming = float((Te == m.name_ids[ye]).mean())
    out = {"model": hf, "ollama": key, "params": m.params, "layers": m.n_layers, "naming_acc": naming,
           "naming_by_lang": [float((Te == m.name_ids[ye])[ye == k].mean()) for k in range(K)]}
    rng = np.random.default_rng(21)
    shuffled_t = rng.permutation(yt)

    # read: the language at the ':' token
    sel_read = [G.read(Lt[:, i], yt, Ls[:, i], ys, K) for i in range(m.n_layers + 1)]
    lr = int(np.argmax(sel_read))
    rd = G.read(Lt[:, lr], yt, Le[:, lr], ye, K)
    out["lang-read"] = {"layer": lr, "stat": rd, "sel_by_layer": sel_read,
                        "p": _p_ge(rd, [G.read(Lt[:, lr], rng.permutation(yt), Le[:, lr], ye, K)
                                        for _ in range(N_SHUFFLE)]),
                        "negative": G.read(Lt[:, 0], yt, Le[:, 0], ye, K)}

    # swap: replace the language part at the ':' token
    def swap_rate(L_, y_, sents, layer, mu, extra=None):
        tgt = (y_ + 5) % K
        new = L_[:, layer] - mu[y_] + mu[tgt] if extra is None else L_[:, layer] + extra
        return float((m.top_with_last(sents, layer, new, batch) == m.name_ids[tgt]).mean())
    sel_swap = [swap_rate(Ls, ys, ss, i, G.means(Lt[:, i], yt, K)) for i in range(1, m.n_layers)]
    lw = 1 + int(np.argmax(sel_swap))
    mu = G.means(Lt[:, lw], yt, K)
    sw = swap_rate(Le, ye, se, lw, mu)
    delta = mu[(ye + 5) % K] - mu[ye]
    null = [swap_rate(Le, ye, se, lw, mu, G.random_like(delta, 100 + j)) for j in range(N_RANDOM)]
    out["lang-swap"] = {"layer": lw, "stat": sw, "sel_by_layer": sel_swap, "p": p_against_pooled(sw, len(se), null),
                        "random_median": float(np.median(null)),
                        "negative": swap_rate(Le, ye, se, lw, G.means(Lt[:, lw], shuffled_t, K))}

    # shared: meaning without language
    sel_shared = [G.shared(Ps[:, i], G.means(Pt[:, i], yt, K), n_sel, K) for i in range(m.n_layers + 1)]
    ls = int(np.argmax(sel_shared))
    muP = G.means(Pt[:, ls], yt, K)
    sh = G.shared(Pe[:, ls], muP, n_test, K)
    nulls = []
    for _ in range(N_SHUFFLE):
        X = Pe[:, ls].reshape(K, n_test, -1)
        X = np.stack([x[rng.permutation(n_test)] if k else x for k, x in enumerate(X)]).reshape(K * n_test, -1)
        nulls.append(G.shared(X, muP, n_test, K))
    out["lang-shared"] = {"layer": ls, "stat": sh, "sel_by_layer": sel_shared, "p": _p_ge(sh, nulls),
                          "without_removal": G.shared(Pe[:, ls], muP, n_test, K, remove=False),
                          "negative": G.shared(Pe[:, 0], G.means(Pt[:, 0], yt, K), n_test, K)}

    # speak: steer an English continuation
    codes = [c for c, _ in G.LANGS]

    def speak_rate(eng, layer, a, muP_, rand_seed=None):
        hits = []
        for tc in G.SPEAK:
            d = np.broadcast_to(a * (muP_[codes.index(tc)] - muP_[0]), (len(eng), muP_.shape[1])).copy()
            if rand_seed is not None:
                d = G.random_like(d, rand_seed)
            hits += [G.detect(x) == tc for x in m.continue_with(eng, layer, d, batch)]
        return float(np.mean(hits))
    eng_sel = P["eng_Latn"][SEL[0]:SEL[1]]
    eng_test = P["eng_Latn"][TEST[0]:TEST[1]]
    grid = []
    for f in SPEAK_FRACTIONS:
        layer = max(1, round(f * m.n_layers))
        muP_ = G.means(Pt[:, layer], yt, K)
        for a in G.ALPHAS:
            grid.append({"layer": layer, "alpha": a, "sel": speak_rate(eng_sel, layer, a, muP_)})
            if log:
                log(f"  speak SEL layer {layer} x{a}: {grid[-1]['sel']:.2f}")
    best = max(grid, key=lambda g: (g["sel"], -g["layer"], -g["alpha"]))
    layer, a = best["layer"], best["alpha"]
    muP_ = G.means(Pt[:, layer], yt, K)
    sp = speak_rate(eng_test, layer, a, muP_)
    null = [speak_rate(eng_test, layer, a, muP_, 200 + j) for j in range(N_RANDOM_SPEAK)]
    base = [G.detect(x) for x in m.continue_with(eng_test, None, None, batch)]
    out["lang-speak"] = {"layer": layer, "alpha": a, "stat": sp, "sel_grid": grid,
                         "p": p_against_pooled(sp, len(eng_test) * len(G.SPEAK), null),
                         "random_median": float(np.median(null)), "unsteered_english": float(np.mean([b == "eng_Latn" for b in base])),
                         "negative": speak_rate(eng_test, layer, a, G.means(Pt[:, layer], shuffled_t, K))}
    out["entity"] = record(m, out)
    m.model = None   # free the GPU before the caller loads anything else
    gc.collect()
    if log:
        log("  " + "; ".join(f"{c['id']} {out[c['id']]['stat']:.2f} at layer {out[c['id']]['layer']}" for c in CLAIMS))
    return out


def record(m, out: dict) -> str:
    b = Bank(config.home() / "neuro" / "bank")
    ent = f"LM-{m.name.split('/')[-1]}{'-' + m.quant if m.quant else ''}".upper()
    t = now_iso()
    subj = Fact(ent, "subject", {"task": "language", "model": m.name, "sha": m.sha, "quant": m.quant,
                                 "params": m.params, "layers": m.n_layers, "naming_acc": round(out["naming_acc"], 4)},
                t[:10], t, "reported", {"name": "huggingface", "url": f"https://huggingface.co/{m.name}/tree/{m.sha}"})
    facts = [subj] + [Fact(ent, f"lang/{c['id'].split('-')[1]}",
                           {k: v for k, v in out[c["id"]].items() if not isinstance(v, list)},
                           t[:10], t, "derived", {"name": "brain.neuro.lang"}, inputs=(subj.ref,)) for c in CLAIMS]
    b.add_many(facts)
    return ent


def judge(c: dict, m: dict) -> dict:
    x = m[c["id"]]
    th = c["threshold"]
    learnt = m["naming_acc"] >= LEARNT["naming_acc_min"]
    neg_ok = not (x["negative"] >= th)
    gate = x["stat"] >= th and x["p"] < 0.05
    return {"n": 1, "learnt": int(learnt), "passes": int(gate), "p": x["p"], "median_stat": x["stat"], "threshold": th,
            "min_stat": x["stat"], "rows": [{"model": m["model"], **x}],
            "controls": {"positive": True, "negative": neg_ok, "pass": neg_ok}, "gate": {"pass": gate},
            "model": m["model"], "layer": x["layer"]}


def verdicts(path) -> dict[str, dict]:
    res = ledger.latest_results(path)
    gone = ledger.withdrawn(path)
    q_of = dict(zip(res, lmclaims.bh_adjust([r["p"] for r in res.values()]))) if res else {}
    out = {}
    for k, r in res.items():
        if r.get("kind") != "neuro-lang":
            continue
        q = q_of[k]
        if k in gone:
            v = f"WITHDRAWN: {gone[k]}"
        elif not r["learnt"]:
            v = "NO VERDICT (the model does not name the languages)"
        elif not r["controls"]["pass"]:
            v = "NO VERDICT (controls failed)"
        else:
            v = "PASS" if r["gate"]["pass"] and q < Q else "FAIL"
        out[k] = {**r, "bh_q": q, "verdict": v}
    return out


def test(key: str, trial: bool = False, log=print) -> list[dict]:
    led, res_md = n1.paths(trial)
    recs = ledger.read(led)
    want = {r["id"]: r for r in recs if r["op"] == "declare" and r.get("kind") == "neuro-lang"
            and r["spec"]["ollama"] == key}
    done = {r["id"] for r in recs if r["op"] == "result"}
    want = {i: r for i, r in want.items() if i not in done}
    if not want:
        raise n1.Refused(f"nothing to test for {key}: no declared N3 claim without a result "
                         f"(declared: {sorted({r['spec']['ollama'] for r in recs if r.get('kind') == 'neuro-lang'})})")
    pinned = n1.committed()
    for i, d in want.items():
        if not trial and (i, d["spec_sha"]) not in pinned:
            raise n1.Refused(f"{i}: its declaration is not in HEAD's research/neuro_ledger.jsonl; commit it first")
        if d["spec"]["code_sha"] != code_sha():
            raise n1.Refused(f"{i}: the code changed since it was declared; declared claims are frozen")
    sha = next(iter(want.values()))["spec"]["sha"]
    m = measure_model(key, sha, log, splits=TRIAL_SPLITS if trial else None)
    out = []
    for c in CLAIMS:
        i = claim_id(c, key)
        if i not in want:
            continue
        rec = {"op": "result", "kind": "neuro-lang", "id": i, "spec_sha": want[i]["spec_sha"],
               **judge({**c, "threshold": want[i]["spec"]["threshold"]}, m),
               "measurements": {k: m[k] for k in ("naming_acc", "naming_by_lang", "params", "layers", "entity")},
               "at": now_iso(), "trial": trial, **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
    res_md.write_text(n1.report(led) + lmclaims.report(led) + report(led))
    return out


def report(path) -> str:
    vs = verdicts(path)
    if not vs:
        return ""
    L_ = ["", "# How a ~30B model handles language", "",
          ("Claims declared before they ran, after looking only at Qwen3-0.6B (`brain/neuro/langclaims.py`, "
           "`docs/NEURO.md`). The BH q is shared with every claim above."), "",
          "| Claim | Model | Layer | Measure | Threshold | p | BH q | Negative control | Verdict |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in vs.items():
        x = r["rows"][0]
        L_.append(f"| {k} | {r['model'].split('/')[-1]} | {r['layer']} | {r['median_stat']:.3f} | "
                  f"{r.get('threshold', '')} | {r['p']:.2g} | {r['bh_q']:.2g} | {x['negative']:.3f} "
                  f"({'fails, as it must' if r['controls']['negative'] else 'PASSES: control broken'}) | {r['verdict']} |")
    L_ += ["", "## The claims", ""] + [f"- **{c['id']}**: {c['claim']} Layer: {c['layer']}. Null: {c['null']}. "
                                       f"Negative control: {c['negative']}." for c in CLAIMS]
    L_ += ["", "## Measurements", ""]
    for model in dict.fromkeys(r["model"] for r in vs.values()):
        rs = {r["id"].split("-")[2]: r for r in vs.values() if r["model"] == model}
        ms = next(iter(rs.values()))["measurements"]
        sp = rs.get("speak", {}).get("rows", [{}])[0]
        sw = rs.get("swap", {}).get("rows", [{}])[0]
        sh = rs.get("shared", {}).get("rows", [{}])[0]
        L_.append(f"- {model}: {ms['params'] / 1e9:.1f}B parameters, {ms['layers']} layers; names the language of "
                  f"{ms['naming_acc']:.3f} of TEST sentences. Swap: random steps of the same size make it name the "
                  f"target {sw.get('random_median', float('nan')):.2f}. Shared meaning without removing the language "
                  f"mean: {sh.get('without_removal', float('nan')):.2f}. Speak: strength x{sp.get('alpha')}, random "
                  f"directions {sp.get('random_median', float('nan')):.2f}, unsteered continuations English "
                  f"{sp.get('unsteered_english', float('nan')):.2f}.")
    return "\n".join(L_) + "\n"
