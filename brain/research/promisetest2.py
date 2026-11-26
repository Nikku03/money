"""The promise tests on the other universes (G2a-G2d): the questions G1 held back, and G1a again where no outcome
has been seen (brain/promises.py, docs/PROMISES.md).

    python -m brain promisetest2 declare      # after committing the code
    python -m brain promisetest2 test         # once; then commit the ledger and research/RESULTS.md

G1 ran on the research universe (``promisetest``). Its two held-back questions had too few cases there. These tests
run on the small, micro and newer-listing universes together, whose releases were re-read with their
forward-looking passages on 2026-11-25/26 and whose promises of 2024-2026 no one has set against an outcome. The
research universe is left out: G1b printed its week after a release.

Explored on promises made 2011-2023 only (``research/promises/explore_bands_train.py`` and its log; 3,993
quarterly promises of 405 companies of the three universes):

G2a (delivery)  A company's habit carries to its next promise: the rank correlation of the habit (the median
                surprise of its last four deliveries) with the surprise on the new promise. TRAIN +0.351. G1a
                again, on other companies.
G2b (delivery)  Beyond its habit, a promise that asks for more is beaten by more: the stretch (the growth the
                promise asks for less the growth just done) against the surprise less the habit. TRAIN +0.150.
                Asked for as "does a promise that asks for more fall short": on the old years it does not.
G2c (drift)     After a release, the stock follows what was delivered against the habit: what the company
                delivered against its last promise, less the habit that promise was made under, against the
                stock's move over the five sessions after the report session. TRAIN +0.074.
G2d (drift)     After a release, the stock follows the change in the year's promise: the change against the
                previous promise for the same fiscal year, against the same five sessions. TRAIN +0.059.

Each statistic is the season's rank correlation (a calendar quarter with at least MIN_N rows), averaged over the
seasons, with a one-sided sign-flip p over the seasons, a placebo and an oracle (``promisetest.ic_test``). Gates: a
positive statistic with p < 0.05, the controls, and Benjamini-Hochberg q < 0.10 across every gated test in the
ledger. G2a and G2b are about what a company delivers. G2c and G2d are about its stock after the report session's
close: a pass would be the first tested direction in this repo, a fail means the release is priced that session.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from brain import config, episodes, promises
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research import promisetest as pt
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

KIND = "promise"
IDS = ("G2a", "G2b", "G2c", "G2d")
GATE_TYPE = {"G2a": "delivery", "G2b": "delivery", "G2c": "drift", "G2d": "drift"}
QUESTION = {"G2a": "does the habit of beating its own revenue promise carry to the next promise, on the small, micro "
                   "and newer-listing universes",
            "G2b": "beyond its habit, is a promise that asks for more growth beaten by more",
            "G2c": "after a release, do the next five sessions follow what was delivered against the habit",
            "G2d": "after a release, do the next five sessions follow the change in the year's promise"}
VERSION = 1
UNIVERSES = ("us_small", "us_micro", "us_watch")
LEFT_OUT = "us_broad"
TRAIN, TEST, MIN_N = pt.TRAIN, pt.TEST, pt.MIN_N
P_MAX = pt.P_MAX
SEED = 41
CLEAN = ("brain", "knowledge", "universe")
HERE = Path(__file__).resolve()


def code_files() -> list[Path]:
    return [HERE, *pt.code_files()]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _tickers(name: str) -> set[str]:
    return {t.upper() for t in json.loads((config.ROOT / "universe" / f"{name}.json").read_text())["tickers"]}


def universe() -> list[str]:
    """The three universes' companies, without any of the research universe."""
    out: set[str] = set()
    for n in UNIVERSES:
        out |= _tickers(n)
    return sorted(out - _tickers(LEFT_OUT))


def universe_sha() -> str:
    h = hashlib.sha256()
    for n in (*UNIVERSES, LEFT_OUT):
        h.update((config.ROOT / "universe" / f"{n}.json").read_bytes())
    return h.hexdigest()[:16]


def spec(tid: str) -> dict:
    return {"id": tid, "version": VERSION, "train": list(TRAIN), "test": list(TEST), "universes": list(UNIVERSES),
            "left_out": LEFT_OUT, "universe_sha": universe_sha(), "min_n": MIN_N, "here_days": pt.HERE_DAYS,
            "habit": promises.HABIT, "reader": promises.READER_MIN, "code_sha": code_sha(), "n_flip": pt.N_FLIP,
            "n_boot": pt.N_BOOT, "n_placebo": pt.N_PLACEBO, "seed": SEED, "gate_type": GATE_TYPE[tid],
            "gate": {"stat_gt": 0.0, "p_lt": P_MAX, "bh_q_lt": 0.10, "placebo_fpr_max": pt.PLACEBO_FPR_MAX,
                     "oracle_p_max": pt.ORACLE_P_MAX}}


def spec_sha(s: dict) -> str:
    return hashlib.sha256(canon(s).encode()).hexdigest()[:16]


def lineage_problem(commit: str | None) -> str | None:
    if not commit:
        return "the declaration has no commit"
    if pt._git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        return f"HEAD is not the declare commit {commit[:12]} or a descendant of it"
    root = config.ROOT.resolve()
    files = [p.relative_to(root).as_posix() for p in code_files()]
    changed = [x for x in pt._git("diff", "--name-only", commit, "HEAD", "--", *files).stdout.splitlines() if x.strip()]
    return f"code changed since the declare commit {commit[:12]}: {', '.join(changed)}" if changed else None


def committed_declaration(tid: str, sha: str) -> bool:
    r = pt._git("show", "HEAD:research/ledger.jsonl")
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("op") == "declare" and rec.get("kind") == KIND \
                and rec.get("id") == tid and rec.get("spec_sha") == sha:
            return True
    return False


def _declared(recs: list[dict], tid: str) -> dict | None:
    return next((r for r in recs if r["op"] == "declare" and r.get("kind") == KIND and r["id"] == tid), None)


def declare(*, trial: bool = False, log=print) -> list[dict]:
    trial = is_trial(trial)
    led, _ = paths(trial)
    if not trial and (dirty := pt.tree_dirty()):
        raise Refused("commit brain/, knowledge/ and universe/ first; a declaration pins the committed code: "
                      + "; ".join(dirty))
    out = []
    for tid in IDS:
        s = spec(tid)
        have = _declared(ledger.read(led), tid)
        if have:
            if have["spec_sha"] != spec_sha(s):
                raise Refused(f"{tid} was declared with a different spec; a change is a new id")
            if log:
                log(f"{tid} is already declared ({have['spec_sha']})")
            continue
        rec = {"op": "declare", "kind": KIND, "id": tid, "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
               **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            log(f"declared {tid} ({rec['spec_sha']}): {QUESTION[tid]}; commit the ledger before running")
    return out


# ---------- the statistics (pure, given the two libraries) ----------
def evaluate(lib: list[dict], eps: list[dict], tickers=None, rng=None, span: tuple[str, str] = TEST) -> dict[str, dict]:
    """Every test's result from the promise library and the episode library (pure, given the rows)."""
    rng = rng or np.random.default_rng(SEED)
    Q, Y = pt.table(lib, eps, tickers)
    q = [r for r in Q if span[0] <= r["made"] <= span[1]]
    y = [r for r in Y if span[0] <= r["made"] <= span[1]]
    gq, gy = np.array([r["season"] for r in q]), np.array([r["season"] for r in y])
    habit, surprise = pt._col(q, "habit"), pt._col(q, "surprise")
    out = {}
    out["G2a"] = {**pt.ic_test(gq, habit, surprise, rng), "habit": pt.habit_report(q),
                  "companies": len({r["t"] for r in q})}
    out["G2b"] = pt.ic_test(gq, pt._col(q, "stretch"), surprise - habit, rng)
    out["G2c"] = pt.ic_test(gq, pt._col(q, "got") - pt._col(q, "got_habit"), pt._col(q, "post5"), rng)
    out["G2d"] = {**pt.ic_test(gy, pt._col(y, "change"), pt._col(y, "post5"), rng), "after": after_change(y)}
    return out


def after_change(y: list[dict]) -> dict:
    """The five sessions after the report session, by what the release did to the year's promise."""
    out = {}
    for g in ("raised", "held", "cut"):
        v = np.array([r["post5"] for r in y if r.get("changed") == g and r.get("post5") is not None], dtype=float)
        if len(v):
            out[g] = {"n": int(len(v)), "post5": float(v.mean()), "rose": float(np.mean(v > 0))}
    return out


def words(tid: str, m: dict) -> str:
    c = m["controls"]
    tail = (f" Controls: shuffled predictions reached p < 0.05 in {c['placebo_fpr'] * 100:.1f}% of draws; the oracle p "
            f"{c['oracle_p']:.4f}.")
    head = f"{m['rows']} rows in {m['seasons']} seasons"
    pos = f"(positive in {m['seasons_positive']} of {m['seasons']} seasons)"
    if tid == "G2a":
        base = (f"{m['rows']} quarterly promises of {m.get('companies', 0)} companies in {m['seasons']} seasons; rank "
                f"correlation of the habit with the surprise on the new promise {m['stat']:+.4f} {pos}. The top fifth "
                f"by habit delivered a median {m['top_fifth_median'] * 100:+.2f}% against its midpoint, the bottom "
                f"fifth {m['bottom_fifth_median'] * 100:+.2f}%.")
        if (h := m.get("habit")) and h.get("rows"):
            base += (f" Reported, not gated: the delivery was a median {h['miss_midpoint'] * 100:.2f}% from the "
                     f"midpoint and {h['miss_habit'] * 100:.2f}% from the midpoint moved by the habit; it fell inside "
                     f"the 2-in-3 range {h['in68'] * 100:.1f}% of the time and inside the 9-in-10 range "
                     f"{h['in90'] * 100:.1f}%; {h['missed_range'] * 100:.1f}% missed the company's own range.")
        return base + tail
    if tid == "G2b":
        return (f"{head}; rank correlation of the stretch with the surprise less the habit {m['stat']:+.4f} {pos}. The "
                f"fifth of promises asking for most delivered a median {m['top_fifth_median'] * 100:+.2f}% beyond its "
                f"habit, the fifth asking for least {m['bottom_fifth_median'] * 100:+.2f}%.{tail}")
    if tid == "G2c":
        return (f"{head}; rank correlation of the delivery against the habit with the stock's next five sessions "
                f"{m['stat']:+.4f} {pos}. After the top fifth the stock moved a mean {m['top_fifth'] * 100:+.2f}% "
                f"against the market, after the bottom fifth {m['bottom_fifth'] * 100:+.2f}%.{tail}")
    a = m.get("after") or {}
    said = "; ".join(f"{k} ({v['n']}): {v['post5'] * 100:+.2f}%, {v['rose'] * 100:.0f}% rose" for k, v in a.items())
    return (f"{head}; rank correlation of the change in the year's promise with the stock's next five sessions "
            f"{m['stat']:+.4f} {pos}. The five sessions after the report session, against the market: {said}.{tail}")


def test(bank: Bank, *, trial: bool = False, log=print, lib: list[dict] | None = None,
         eps: list[dict] | None = None) -> list[dict]:
    from brain.research import run
    trial = is_trial(trial)
    led, res = paths(trial)
    recs = ledger.read(led)
    decls = {}
    for tid in IDS:
        decl = _declared(recs, tid)
        if decl is None:
            raise Refused(f"{tid} is not declared")
        if spec_sha(decl["spec"]) != decl["spec_sha"]:
            raise Refused(f"{tid}: its declared spec does not match its spec_sha")
        if not trial:
            if dirty := pt.tree_dirty():
                raise Refused("the working tree has changes under brain/, knowledge/ or universe/: " + "; ".join(dirty))
            if why := lineage_problem(decl.get("commit")):
                raise Refused(f"{tid}: {why}")
            if decl["spec"]["code_sha"] != code_sha():
                raise Refused(f"{tid}: the code is not the declared code")
            if not committed_declaration(tid, decl["spec_sha"]):
                raise Refused(f"{tid}: commit the ledger with its declaration first")
        decls[tid] = decl
    todo = [t for t in IDS if not any(r["op"] == "result" and r.get("kind") == KIND and r["id"] == t for r in recs)]
    if not todo:
        if log:
            log("every G2 test already has its result")
        return []
    names = universe()
    if lib is None:
        if log:
            log(f"building the promise library from the bank ({len(names)} companies of the three universes)")
        lib = promises.build(bank, names, log=log)
    if eps is None:
        if log:
            log("building the episode library from the bank (the report sessions' moves)")
        eps = episodes.build(bank, log=log)
        episodes.save(eps)
    result = evaluate(lib, eps, names)
    out = []
    for tid in todo:
        m = result[tid]
        gate = {"stat_positive": m["stat"] > 0, "p_ok": m["p"] < P_MAX, "controls": m["controls"]["pass"]}
        gate["pass"] = bool(all(gate.values()))
        rec = {"op": "result", "kind": KIND, "id": tid, "spec_sha": decls[tid]["spec_sha"],
               "gate_type": GATE_TYPE[tid], "metrics": {**{k: v for k, v in m.items() if k != "controls"},
                                                        "question": QUESTION[tid], "words": words(tid, m)},
               "controls": m["controls"], "gate": gate, "p": m["p"], "at": now_iso(), **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            log(f"{tid}: {rec['metrics']['words']}")
    run.report(led, res)
    if log:
        log(f"wrote {res}" + ("" if trial else "; commit the ledger and RESULTS.md, whatever they say"))
    return out


def _cmd(a) -> int:
    if a.action == "declare":
        declare(trial=a.trial)
    else:
        eps = episodes.load() if (a.cached and a.trial) else None     # a real run builds the libraries afresh
        test(Bank(), trial=a.trial, eps=eps or None)
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("promisetest2", help="declare or run G2a-G2d: the promise tests on the other universes")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.add_argument("--cached", action="store_true", help="use the episode library already built (var/episodes)")
    p.set_defaults(fn=_cmd)
