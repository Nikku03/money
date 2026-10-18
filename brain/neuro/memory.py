"""What the engine learns about a network goes into the memory bank, and a new network is built from the bank alone.

Each network is an entity, ``NN-<weights sha>``. Its ``subject`` fact says what it is and cites the code that
made it. Everything measured inside it is a derived fact that cites the subject fact:

- modular addition: ``neuron/017`` (each neuron's frequency, share, phase and amplitude in U, V and W),
  ``law/phase`` (how strongly each candidate phase law holds), ``spectrum/summary``;
- the chain engine: ``ablate/none``, ``ablate/memory``, ``ablate/pointer``, ``patch/pointer``, ``unit/07``.

``rebuild`` writes a network from one entity's facts: each neuron a cosine of its recorded frequency, its input
phases and amplitudes as recorded, and its output weights from the recorded phase law, never copied. ``transfer``
uses only the rule (one frequency per neuron, the law) to write a network for a modulus nothing was trained on.
Neither reads a weight. The bank lives in ``<home>/neuro/bank`` beside the market bank, not in it.
"""
from __future__ import annotations

import hashlib

import numpy as np

from brain import config
from brain.bank import Bank, Fact, canon
from brain.neuro import probe
from brain.neuro.subjects import ChainSubject, ModAdd, cosine_net, sha
from brain.research import ledger
from brain.timeutil import now_iso

SHARE_MIN = 0.8   # a neuron description counts as "one frequency" at this median share
LAW_MIN = 0.9     # a phase law counts as recorded at this strength
REPO = "https://github.com/Nikku03/money"


def neuro_bank() -> Bank:
    return Bank(config.home() / "neuro" / "bank")


def entity_of(net) -> str:
    """A trained network is named by its weights; a hand-written step has none, so by what it is."""
    if net.weights:
        return f"NN-{sha(net.weights)[:12]}"
    return "NN-" + hashlib.sha256(canon([net.kind, net.spec]).encode()).hexdigest()[:12]


def _subject_fact(net, task: str, extra: dict) -> Fact:
    commit = ledger.git_state().get("commit")
    url = f"{REPO}/blob/{commit}/brain/neuro/subjects.py" if commit else f"{REPO}/blob/main/brain/neuro/subjects.py"
    t = now_iso()
    return Fact(entity_of(net), "subject", {"task": task, "kind": net.kind, "seed": net.seed, "spec": net.spec,
                                            "weights_sha": sha(net.weights), **extra},
                t[:10], t, "reported", {"name": "brain.neuro.subjects", "url": url, "commit": commit})


def _derived(ent, attr, value, subj: Fact, probe_name: str) -> Fact:
    t = now_iso()
    return Fact(ent, attr, value, t[:10], t, "derived", {"name": f"brain.neuro.probe:{probe_name}"},
                inputs=(subj.ref,))


def _r(x, n=4):
    return round(float(x), n)


def record_modadd(bank: Bank, net: ModAdd) -> str:
    """Measure a modular-addition network and write what was found. Returns its entity."""
    ent = entity_of(net)
    subj = _subject_fact(net, "modadd", {"p": net.p, "heldout_acc": _r(net.heldout()), "all_acc": _r(net.accuracy())})
    sp = probe.neurons(net)
    facts = [subj]
    for k in range(len(net.U)):
        facts.append(_derived(ent, f"neuron/{k:03d}", {
            part: {"freq": int(sp[part]["freq"][k]), "share": _r(sp[part]["share"][k]),
                   "phase": _r(sp[part]["phase"][k]), "amp": _r(sp[part]["amp"][k])} for part in ("u", "v", "w")},
            subj, "spectrum"))
    same = probe.same_freq(sp)
    facts.append(_derived(ent, "law/phase", {"sum": _r(probe.law_strength(sp, "sum", same)),
                                             "diff": _r(probe.law_strength(sp, "diff", same)),
                                             "neurons": int(same.sum())}, subj, "law_strength"))
    facts.append(_derived(ent, "spectrum/summary", {
        "median_share_u": _r(np.median(sp["u"]["share"])), "median_share_v": _r(np.median(sp["v"]["share"])),
        "same_freq": _r(same.mean()),
        "neurons_per_freq": np.bincount(sp["u"]["freq"], minlength=(net.p - 1) // 2 + 1)[1:].tolist()},
        subj, "spectrum"))
    bank.add_many(facts)
    return ent


def record_chain(bank: Bank, s: ChainSubject, units: bool = True) -> str:
    ent = entity_of(s)
    subj = _subject_fact(s, "chain", {"acc_16": _r(s.accuracy(16)), "acc_64": _r(s.accuracy(64))})
    ref = probe.reference_states(s)
    facts = [subj] + [_derived(ent, f"ablate/{name}", {"acc_16": _r(probe.ablate(s, what, ref=ref))}, subj, "ablate")
                      for name, what in (("none", None), ("memory", "h"), ("pointer", "e"))]
    facts.append(_derived(ent, "patch/pointer", {k: _r(v) if isinstance(v, float) else v
                                                 for k, v in probe.patch(s).items()}, subj, "patch"))
    if units and s.weights:
        for u, acc in enumerate(probe.unit_effects(s)):
            facts.append(_derived(ent, f"unit/{u:02d}", {"acc_16_without": _r(acc)}, subj, "unit_effects"))
    bank.add_many(facts)
    return ent


def latest(bank: Bank, ent: str, prefix: str) -> dict[str, Fact]:
    """Newest fact per attribute starting with ``prefix``."""
    out: dict[str, Fact] = {}
    for f in bank.facts(ent):
        if f.attr.startswith(prefix) and (f.attr not in out or f.known_at >= out[f.attr].known_at):
            out[f.attr] = f
    return out


class NoMechanism(Exception):
    """The bank holds no description good enough to build from."""


def description(bank: Bank, ent: str) -> dict:
    """The rule the bank recorded for one network: whether its neurons are single cosines, which phase law
    holds, and the facts that say so (cited by id)."""
    summ = latest(bank, ent, "spectrum/summary").get("spectrum/summary")
    law = latest(bank, ent, "law/phase").get("law/phase")
    if summ is None or law is None:
        raise NoMechanism(f"{ent}: no spectrum or phase-law facts")
    share = min(summ.value["median_share_u"], summ.value["median_share_v"])
    best = max(("sum", "diff"), key=lambda k: law.value[k])
    why = []
    if share < SHARE_MIN:
        why.append(f"neurons are not single cosines (median share {share:.2f} < {SHARE_MIN})")
    if law.value[best] < LAW_MIN:
        why.append(f"no phase law holds (best {best} {law.value[best]:.2f} < {LAW_MIN})")
    if why:
        raise NoMechanism(f"{ent}: " + "; ".join(why))
    return {"law": best, "law_strength": law.value[best], "share": share, "cites": [summ.ref, law.ref]}


def rebuild(bank: Bank, ent: str, law: str | None = None) -> ModAdd:
    """A network written from ``ent``'s facts alone. ``law`` overrides the recorded law (a control)."""
    d = description(bank, ent)
    subj = latest(bank, ent, "subject")["subject"].value
    rows = [f.value for _, f in sorted(latest(bank, ent, "neuron/").items())]
    return cosine_net(subj["p"], [r["u"]["freq"] for r in rows],
                      [(r["u"]["amp"], r["v"]["amp"], r["w"]["amp"]) for r in rows],
                      [r["u"]["phase"] for r in rows], [r["v"]["phase"] for r in rows],
                      law or d["law"], "rebuilt", {"from": ent, "law": law or d["law"], "cites": d["cites"]})


def transfer(bank: Bank, ent: str, p: int, hidden: int, seed: int) -> ModAdd:
    """The recorded rule applied to modulus ``p``: neurons spread over every frequency, random phases, unit
    amplitudes, output phase by the recorded law."""
    d = description(bank, ent)
    rng = np.random.default_rng(seed)
    F = (p - 1) // 2
    return cosine_net(p, np.arange(hidden) % F + 1, np.ones((hidden, 3)), rng.uniform(0, 2 * np.pi, hidden),
                      rng.uniform(0, 2 * np.pi, hidden), d["law"], "transferred",
                      {"from": ent, "p": p, "hidden": hidden, "law": d["law"], "cites": d["cites"]})


def card(bank: Bank, ent: str) -> str:
    """A one-screen account of what the bank holds about a network, each line citing its fact."""
    subj = latest(bank, ent, "subject").get("subject")
    if subj is None:
        return f"{ent}: not in the neuro bank"
    v = subj.value
    lines = [f"{ent}  {v['task']} {v['kind']} seed {v['seed']}  [{subj.id}]"]
    if v["task"] == "modadd":
        s = latest(bank, ent, "spectrum/summary")["spectrum/summary"]
        law = latest(bank, ent, "law/phase")["law/phase"]
        lines += [f"  held-out accuracy {v['heldout_acc']:.3f}, all pairs {v['all_acc']:.3f}",
                  (f"  median top-frequency share: U {s.value['median_share_u']:.2f}, V {s.value['median_share_v']:.2f}; "
                   f"same frequency in U, V, W: {s.value['same_freq']:.0%}  [{s.id}]"),
                  f"  neurons per frequency 1..{len(s.value['neurons_per_freq'])}: {s.value['neurons_per_freq']}",
                  (f"  phase law: out = in_a + in_b holds at {law.value['sum']:.2f}; in_a - in_b at "
                   f"{law.value['diff']:.2f}  [{law.id}]")]
        try:
            d = description(bank, ent)
            lines.append(f"  mechanism: single-cosine neurons, output phase by the '{d['law']}' law (rebuildable)")
        except NoMechanism as e:
            lines.append(f"  mechanism: none recorded ({str(e).split(': ', 1)[1]})")
    else:
        ab = latest(bank, ent, "ablate/")
        pt = latest(bank, ent, "patch/pointer").get("patch/pointer")
        lines.append(f"  16-step accuracy {v['acc_16']:.3f}, 64-step {v['acc_64']:.3f}")
        for name in ("none", "memory", "pointer"):
            f = ab.get(f"ablate/{name}")
            if f:
                lines.append(f"  {name + ' ablated' if name != 'none' else 'nothing ablated':<17} 16-step accuracy "
                             f"{f.value['acc_16']:.3f}  [{f.id}]")
        if pt:
            lines.append(f"  pointer patched to item j: writes table[j] {pt.value['follows']:.3f}, carries on from j "
                         f"{pt.value['carries']:.3f}, ignores the patch {pt.value['stays']:.3f}  [{pt.id}]")
        un = latest(bank, ent, "unit/")
        if un:
            accs = [f.value["acc_16_without"] for _, f in sorted(un.items())]
            hurt = [i for i, a in enumerate(accs) if a < 0.9]
            lines.append(f"  units whose loss drops 16-step accuracy below 0.9: {hurt or 'none'} of {len(accs)}")
    return "\n".join(lines)
