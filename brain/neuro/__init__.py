"""Hand the engine a neural network and see what it can find out about the inside (docs/NEURO.md).

The memory bank and the evidence engine, pointed at a network instead of a market. ``probe`` looks inside
(weights, spectra) and intervenes (projects, ablates, patches); what it finds goes into a bank of its own as
cited facts (``memory``); claims about the mechanism are declared before they run and tested on networks nothing
looked at, beside planted and negative controls (``claims``); and ``memory.rebuild`` / ``memory.transfer`` write
new networks from the bank's facts alone, to see whether what was learnt is enough to make the thing again.

Subjects (``subjects``): a modular-addition network, whose mechanism the interpretability literature has worked
out (a known answer to grade against), and the chain engine's own bump-tree network, whose mechanism nobody had
written down.
"""
from __future__ import annotations


def _cmd(a) -> int:
    from brain.neuro import claims, memory
    from brain.neuro import subjects as S
    if a.action == "look":
        bank = memory.neuro_bank()
        nets = [S.train_modadd(s) for s in claims.DEV["modadd"]]
        nets += [S.train_modadd(s, S.MEMORIZER, "memorizer") for s in claims.DEV["modadd"]]
        nets.append(S.planted_modadd(0))
        for n in nets:
            print(memory.card(bank, memory.record_modadd(bank, n)), "\n")
        for s in [S.train_chain(x) for x in claims.DEV["chain"]] + [S.planted_lookup(), S.planted_memory()]:
            print(memory.card(bank, memory.record_chain(bank, s)), "\n")
        print("dev networks only: these are looks, not results (python -m brain neuro declare, then test)")
        return 0
    if a.action == "declare":
        new = claims.declare(trial=a.trial)
        print(f"{len(new)} declared" + ("" if a.trial or not new else "; commit research/neuro_ledger.jsonl, then "
                                        "python -m brain neuro test"))
        return 0
    if a.action == "test":
        claims.test(a.words or None, trial=a.trial)
        print(claims.report(claims.paths(a.trial)[0]))
        return 0
    if a.action == "card":
        bank = memory.neuro_bank()
        ents = a.words or [e for e in bank.entities() if e.startswith("NN-")]
        for e in ents:
            print(memory.card(bank, e), "\n")
        return 0
    return 2


def add_cli(sub) -> None:
    s = sub.add_parser("neuro", help="look inside a neural network: probe it, record what is found, test claims "
                                     "about its mechanism, rebuild it from the facts: look | declare | test | card")
    s.add_argument("action", choices=["look", "declare", "test", "card"])
    s.add_argument("words", nargs="*", help="claim ids for test, entities (NN-...) for card")
    s.add_argument("--trial", action="store_true", help="dev seeds and a scratch ledger: exercise the code, "
                                                        "never the record")
    s.set_defaults(fn=_cmd)
