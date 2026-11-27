"""The promise that asks for more, against the report day that delivers it (G3a; brain/promises.py,
docs/PROMISES.md).

    python -m brain promisetest3 declare      # after committing the code
    python -m brain promisetest3 test         # once; then commit the ledger and research/RESULTS.md

A quarterly revenue promise is made in one earnings release and delivered in the next, about three months later.
Its ``stretch`` is the growth it asks for less the growth the company has just reported, known the day the promise
is made. G1b said why a stock moves on the day it reports (the delivery against its habit, the change in the
year's promise); nothing known beforehand had said which way. Looking for what warns that a company will break
its habit, on report sessions of 2011-2023 only (``research/promises/explore_break_train.py`` and its log; 4,947
deliveries with a stretch, twelve things known beforehand looked at):

G3a (drift)  A promise that asks for more is followed by a better report session: the stretch against the
             stock's move against the market on the session that delivers the promised quarter. TRAIN +0.061
             (t +4.2, positive in 33 of 47 seasons; 2011-2018 +0.070, 2019-2023 +0.050; the third asking for least
             -0.19% and 48% rose, the third asking for most +0.82% and 52% rose). It was the only one of the twelve
             that said anything about the session; the stock's own run into the report said how the delivery would
             come against the habit (+0.15) and nothing about the session: the price knew.

A row is the last promise made for a quarter, made before that quarter ended, with a stretch; its session is the
first report session of the company from 5 to 100 days after the quarter's end, in TEST. The four universes are
taken together. No outcome of 2024-2026 was set against a stretch before this was declared: G2b held the stretch
against the delivery less the habit on three of the universes, never against a session.

The statistic is the season's rank correlation (the calendar quarter of the session, at least MIN_N rows), averaged
over the seasons, with a one-sided sign-flip p over the seasons, a placebo and an oracle (``promisetest.ic_test``).
Gates: a positive statistic with p < 0.05, the controls, and Benjamini-Hochberg q < 0.10 across every gated test
in the ledger. A pass would be the first tested direction in this repo, and a small one: a tilt of a report day's
odds known three months ahead, not a trade by itself.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from brain import config, episodes, promises
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research import promisetest as pt
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

KIND = "promise"
IDS = ("G3a",)
GATE_TYPE = {"G3a": "drift"}
QUESTION = {"G3a": "is a promise that asks for more growth followed by a better report session, three months on"}
VERSION = 1
UNIVERSES = ("us_broad", "us_small", "us_micro", "us_watch")
TRAIN, TEST, MIN_N = pt.TRAIN, pt.TEST, pt.MIN_N
P_MAX = pt.P_MAX
AFTER_END = (5, 100)      # the delivering report session comes this many days after the quarter's end
SEED = 43
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
    out: set[str] = set()
    for n in UNIVERSES:
        out |= _tickers(n)
    return sorted(out)


def universe_sha() -> str:
    h = hashlib.sha256()
    for n in UNIVERSES:
        h.update((config.ROOT / "universe" / f"{n}.json").read_bytes())
    return h.hexdigest()[:16]


def spec(tid: str) -> dict:
    return {"id": tid, "version": VERSION, "train": list(TRAIN), "test": list(TEST), "universes": list(UNIVERSES),
            "universe_sha": universe_sha(), "min_n": MIN_N, "after_end": list(AFTER_END),
            "reader": promises.READER_MIN, "code_sha": code_sha(), "n_flip": pt.N_FLIP, "n_boot": pt.N_BOOT,
            "n_placebo": pt.N_PLACEBO, "seed": SEED, "gate_type": GATE_TYPE[tid],
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


# ---------- the rows and the statistic (pure, given the two libraries) ----------
def deliveries(lib: list[dict], eps: list[dict], tickers=None) -> list[dict]:
    """One row a promised quarter: the last promise made for it (made before the quarter ended, not a mismatch) and
    the report session that delivered it, the company's first from AFTER_END[0] to AFTER_END[1] days after the
    quarter's end. ``stretch`` is None when the promise has none; ``r1`` is that session's move against the market."""
    keep = set(tickers) if tickers is not None else None
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in lib:
        if (keep is None or r["t"] in keep) and r["horizon"] == "quarter" and r.get("period_end") \
                and not r.get("mismatch") and r["made"] < r["period_end"]:
            by_t[r["t"]].append(r)
    ep_t: dict[str, list[dict]] = defaultdict(list)
    for e in sorted(eps, key=lambda e: e["date"]):
        if keep is None or e["t"] in keep:
            ep_t[e["t"]].append(e)
    out = []
    for t in sorted(by_t):
        last: dict[str, dict] = {}
        for r in sorted(by_t[t], key=lambda r: r["known_at"]):
            last[r["period_end"]] = r
        for pe in sorted(last):
            r, d0 = last[pe], date.fromisoformat(pe)
            ep = next((e for e in ep_t.get(t, [])
                       if AFTER_END[0] <= (date.fromisoformat(e["date"]) - d0).days <= AFTER_END[1]), None)
            if ep is None or ep.get("r1") is None:
                continue
            out.append({"t": t, "made": r["made"], "period_end": pe, "date": ep["date"],
                        "season": pt.season(ep["date"]), "stretch": r.get("stretch"), "r1": ep["r1"]})
    return out


def thirds(rows: list[dict]) -> dict:
    """The session by the third of the stretch inside each season (seasons with fewer than 9 rows left out)."""
    by_s: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["stretch"] is not None:
            by_s[r["season"]].append(r)
    got: dict[str, list[float]] = {"least": [], "middle": [], "most": []}
    for rs in by_s.values():
        if len(rs) < 9:
            continue
        s = np.array([r["stretch"] for r in rs])
        lo, hi = np.quantile(s, [1 / 3, 2 / 3])
        for r in rs:
            got["least" if r["stretch"] <= lo else "most" if r["stretch"] >= hi else "middle"].append(r["r1"])
    return {k: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
                "rose": float(np.mean(np.array(v) > 0))} for k, v in got.items() if v}


def evaluate(lib: list[dict], eps: list[dict], tickers=None, rng=None, span: tuple[str, str] = TEST) -> dict[str, dict]:
    rng = rng or np.random.default_rng(SEED)
    rows = [r for r in deliveries(lib, eps, tickers) if span[0] <= r["date"] <= span[1]]
    g = np.array([r["season"] for r in rows])
    m = pt.ic_test(g, pt._col(rows, "stretch"), pt._col(rows, "r1"), rng)
    with_s = [r for r in rows if r["stretch"] is not None]
    return {"G3a": {**m, "companies": len({r["t"] for r in with_s}), "thirds": thirds(rows)}}


def words(tid: str, m: dict) -> str:
    c = m["controls"]
    th = m.get("thirds") or {}
    said = "; ".join(f"the third asking for {k} ({v['n']}): {v['mean'] * 100:+.2f}%, {v['rose'] * 100:.0f}% rose"
                     for k, v in th.items())
    return (f"{m['rows']} promised quarters of {m.get('companies', 0)} companies in {m['seasons']} seasons; rank "
            f"correlation of the stretch with the move of the report session that delivered the quarter "
            f"{m['stat']:+.4f} (positive in {m['seasons_positive']} of {m['seasons']} seasons). That session, against "
            f"the market: {said}. Controls: shuffled predictions reached p < 0.05 in {c['placebo_fpr'] * 100:.1f}% of "
            f"draws; the oracle p {c['oracle_p']:.4f}.")


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
            log("every G3 test already has its result")
        return []
    names = universe()
    if lib is None:
        if log:
            log(f"building the promise library from the bank ({len(names)} companies of the four universes)")
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
    p = sub.add_parser("promisetest3", help="declare or run G3a: the promise that asks for more, against the report "
                                            "session that delivers it")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.add_argument("--cached", action="store_true", help="use the episode library already built (var/episodes)")
    p.set_defaults(fn=_cmd)
