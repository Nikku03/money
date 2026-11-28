"""The second guess of a delivery (G4a, G4b; brain/promises.py, docs/PROMISES.md).

    python -m brain promisetest4 declare      # after committing the code
    python -m brain promisetest4 test         # once; then commit the ledger and research/RESULTS.md

Asked on 2026-11-28 ("how can we make it better?"). G1a and G2a showed that a company's habit carries to its next
promise. Looking at promises made 2011-2023 only (``research/promises/explore_guess_train.py`` and its log; 3,872
promises of 279 companies of the four universes, fitted on 2011-2018 and checked on 2019-2023):

- A large habit overshoots. With a habit over 8% the delivery came in under the habit's guess 85% of the time
  (median habit +9.8%, median delivery +5.1%). A negative habit undershoots.
- Pulled in to ``PULL[0] + PULL[1] x habit`` (+1.0% + 0.50 x habit) the guess missed by a mean 3.09% on 2019-2023
  against 3.33% for the plain habit (median 1.83% against 1.98%), and by less in every year from 2012 to 2023.
- The fixed ranges held 90% for steady companies and 45% for erratic ones where they promise 2 in 3. Ranges that
  follow the company's own spread (``BAND2``) held 72.3% and 89.7% on 2019-2023, and 63% to 82% across companies
  of every spread.

G4a (delivery)  The pulled-in habit's guess is closer than the plain habit's: per season (the calendar quarter a
                promise is made in, at least MIN_N promises), the mean over its promises of the plain habit's miss
                less the pulled-in guess's miss, averaged over the seasons, with a one-sided sign-flip p over the
                seasons. Controls: with the two guesses swapped at random per promise the test must stay quiet;
                with the truth as the guess it must pass.
G4b (report)    How often the delivery fell inside the second guess's 2-in-3 and 9-in-10 ranges, in all and by the
                company's own spread, beside the fixed ranges around the plain habit. Reported, not gated.

A row is the last promise made for a quarter (graded, not a mismatch) with a habit, made in TEST, of the four
universes together. The outcomes of 2024-2026 have been set against the plain habit (G1a, G2a) and never against a
pulled-in one or these ranges. Gates for G4a: a positive statistic with p < 0.05, the controls, and
Benjamini-Hochberg q < 0.10 across every gated test in the ledger. This is about what a company delivers, not
about its stock.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from brain import config, promises
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research import promisetest as pt
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

KIND = "promise"
IDS = ("G4a", "G4b")
GATE_TYPE = {"G4a": "delivery", "G4b": "report"}
QUESTION = {"G4a": "is a habit pulled in (+1.0% + 0.50 x habit) a closer guess of the delivery than the plain habit",
            "G4b": "how often the delivery fell inside ranges that follow the company's own spread"}
VERSION = 1
UNIVERSES = ("us_broad", "us_small", "us_micro", "us_watch")
TRAIN, TEST, MIN_N = pt.TRAIN, pt.TEST, pt.MIN_N
P_MAX = pt.P_MAX
SPREADS = ((0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 9.0))
BIG = 0.08                # "a large habit" in the reported lines
SEED = 47
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
    gated = GATE_TYPE[tid] != "report"
    return {"id": tid, "version": VERSION, "train": list(TRAIN), "test": list(TEST), "universes": list(UNIVERSES),
            "universe_sha": universe_sha(), "min_n": MIN_N, "habit": promises.HABIT, "pull": list(promises.PULL),
            "band2": {k: list(v) for k, v in promises.BAND2.items()}, "band68": list(promises.BAND68),
            "band90": list(promises.BAND90), "spreads": [list(x) for x in SPREADS], "reader": promises.READER_MIN,
            "code_sha": code_sha(), "n_flip": pt.N_FLIP, "n_boot": pt.N_BOOT, "n_placebo": pt.N_PLACEBO, "seed": SEED,
            "gate_type": GATE_TYPE[tid],
            "gate": ({"stat_gt": 0.0, "p_lt": P_MAX, "bh_q_lt": 0.10, "placebo_fpr_max": pt.PLACEBO_FPR_MAX,
                      "oracle_p_max": pt.ORACLE_P_MAX} if gated else None)}


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


# ---------- the rows and the statistics (pure, given the promise library) ----------
def deliveries(lib: list[dict], tickers=None, span: tuple[str, str] = TEST) -> list[dict]:
    """One row a promised quarter with a known delivery and a habit: the last promise made for it (graded, not a
    mismatch), made inside ``span``, with the habit and the company's own spread when it was made, the plain
    habit's guess and the second guess with its ranges (fractions of the midpoint)."""
    keep = set(tickers) if tickers is not None else None
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in lib:
        if keep is None or r["t"] in keep:
            by_t[r["t"]].append(r)
    out = []
    for t in sorted(by_t):
        rs = sorted(by_t[t], key=lambda r: r["known_at"])
        last: dict[str, dict] = {}
        for r in rs:
            if r["horizon"] == "quarter" and r.get("grade") and r.get("period_end") and not r.get("mismatch"):
                last[r["period_end"]] = r
        for pe in sorted(last):
            r = last[pe]
            if not (span[0] <= r["made"] <= span[1]):
                continue
            h = promises.habit(rs, r["made"])
            if h is None:
                continue
            sd = promises.spread(rs, r["made"])
            out.append({"t": t, "made": r["made"], "season": pt.season(r["made"]), "s": r["surprise"], "habit": h,
                        "spread": sd, "g": promises.guess2(h, sd)})
    return out


def paired_test(group: np.ndarray, d: np.ndarray, rng, n_placebo: int | None = None) -> dict:
    """Per season (at least MIN_N rows) the mean of the paired differences ``d``; the statistic is the mean over
    seasons, with a one-sided sign-flip p over the seasons and a bootstrap interval."""
    seasons, means = [], []
    for g in sorted(set(group.tolist())):
        k = group == g
        if k.sum() >= MIN_N:
            seasons.append(g)
            means.append(float(d[k].mean()))
    m = np.array(means)
    p = pt.flip_p(m, 1, pt.N_FLIP, rng) if len(m) else 1.0
    out = {"seasons": len(m), "first": seasons[0] if seasons else None, "last": seasons[-1] if seasons else None,
           "stat": float(m.mean()) if len(m) else float("nan"),
           "interval90": pt.boot_interval(m, pt.N_BOOT, rng) if len(m) else [float("nan")] * 2,
           "seasons_positive": int((m > 0).sum()), "p": p}
    if n_placebo is not None:
        hits = 0
        for _ in range(n_placebo):
            flip = rng.choice((-1.0, 1.0), size=len(d))
            mp = np.array([float((d * flip)[group == g].mean()) for g in seasons])
            hits += bool(len(mp)) and pt.flip_p(mp, 1, 2000, rng) < P_MAX
        out["placebo_fpr"] = hits / n_placebo if n_placebo else 0.0
    return out


def coverage(rows: list[dict]) -> dict:
    """How often the delivery fell inside the second guess's ranges and inside the fixed ranges around the plain
    habit, in all and by the company's own spread."""
    def held(rs):
        if not rs:
            return None
        s = np.array([r["s"] for r in rs])
        h = np.array([r["habit"] for r in rs])
        g = {k: np.array([r["g"][k] for r in rs]) for k in ("lo68", "hi68", "lo90", "hi90")}
        return {"n": len(rs), "in68": float(np.mean((s >= g["lo68"]) & (s <= g["hi68"]))),
                "in90": float(np.mean((s >= g["lo90"]) & (s <= g["hi90"]))),
                "fixed68": float(np.mean((s - h >= promises.BAND68[0]) & (s - h <= promises.BAND68[1]))),
                "fixed90": float(np.mean((s - h >= promises.BAND90[0]) & (s - h <= promises.BAND90[1]))),
                "width68": float(np.median(g["hi68"] - g["lo68"]))}
    out = {"all": held(rows), "by_spread": []}
    for lo, hi in SPREADS:
        got = held([r for r in rows if lo < r["spread"] <= hi or (lo == 0.0 and r["spread"] == 0.0)])
        if got:
            out["by_spread"].append({"lo": lo, "hi": hi, **got})
    return out


def evaluate(lib: list[dict], tickers=None, rng=None, span: tuple[str, str] = TEST) -> dict[str, dict]:
    rng = rng or np.random.default_rng(SEED)
    rows = deliveries(lib, tickers, span)
    group = np.array([r["season"] for r in rows])
    s = np.array([r["s"] for r in rows], dtype=float)
    h = np.array([r["habit"] for r in rows], dtype=float)
    g = np.array([r["g"]["usual"] for r in rows], dtype=float)
    plain, second = np.abs(s - h), np.abs(s - g)
    a = paired_test(group, plain - second, rng, pt.N_PLACEBO) if len(rows) else {
        "seasons": 0, "stat": float("nan"), "interval90": [float("nan")] * 2, "seasons_positive": 0, "p": 1.0,
        "placebo_fpr": 0.0, "first": None, "last": None}
    oracle = paired_test(group, plain, rng)["p"] if len(rows) else 1.0     # the truth as the guess: its miss is 0
    fpr = a.pop("placebo_fpr")
    big = h > BIG
    med = lambda x: float(np.median(x)) if len(x) else float("nan")       # noqa: E731
    a.update(rows=len(rows), companies=len({r["t"] for r in rows}), miss_mid=med(np.abs(s)), miss_plain=med(plain),
             miss_second=med(second), mean_plain=float(plain.mean()) if len(rows) else float("nan"),
             mean_second=float(second.mean()) if len(rows) else float("nan"),
             closer=float(np.mean(second < plain)) if len(rows) else float("nan"),
             big_n=int(big.sum()), big_plain=med(plain[big]), big_second=med(second[big]),
             controls={"placebo_fpr": fpr, "oracle_p": oracle,
                       "pass": bool(fpr <= pt.PLACEBO_FPR_MAX and oracle < pt.ORACLE_P_MAX)})
    return {"G4a": a, "G4b": {"rows": len(rows), **coverage(rows)}}


def words(tid: str, m: dict) -> str:
    if tid == "G4a":
        c = m["controls"]
        return (f"{m['rows']} promised quarters of {m['companies']} companies in {m['seasons']} seasons. The pulled-in "
                f"guess missed by a mean {m['mean_second'] * 100:.2f}% against {m['mean_plain'] * 100:.2f}% for the plain "
                f"habit: {m['stat'] * 100:+.3f} points a season in its favour (positive in {m['seasons_positive']} of "
                f"{m['seasons']} seasons). Median miss {m['miss_second'] * 100:.2f}% against {m['miss_plain'] * 100:.2f}% "
                f"(the midpoint alone {m['miss_mid'] * 100:.2f}%); closer than the plain habit in "
                f"{m['closer'] * 100:.0f}% of promises. With a habit over {BIG * 100:.0f}% ({m['big_n']} promises): "
                f"{m['big_second'] * 100:.2f}% against {m['big_plain'] * 100:.2f}%. Controls: the two guesses swapped at "
                f"random reached p < 0.05 in {c['placebo_fpr'] * 100:.1f}% of draws; the oracle p {c['oracle_p']:.4f}.")
    a = m.get("all") or {}
    if not a:
        return "no rows"
    def name(b):
        return f"over {b['lo'] * 100:.0f}%" if b["hi"] > 1 else f"{b['lo'] * 100:.0f}% to {b['hi'] * 100:.0f}%"
    parts = "; ".join(f"own spread {name(b)} ({b['n']}): {b['in68'] * 100:.0f}% and {b['in90'] * 100:.0f}% (fixed "
                      f"ranges {b['fixed68'] * 100:.0f}% and {b['fixed90'] * 100:.0f}%)" for b in m.get("by_spread", []))
    return (f"{a['n']} promised quarters. The second guess's 2-in-3 range held {a['in68'] * 100:.1f}% and its 9-in-10 "
            f"range {a['in90'] * 100:.1f}% (the fixed ranges around the plain habit: {a['fixed68'] * 100:.1f}% and "
            f"{a['fixed90'] * 100:.1f}%). By the company's own spread: {parts}.")


def test(bank: Bank, *, trial: bool = False, log=print, lib: list[dict] | None = None) -> list[dict]:
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
            log("every G4 test already has its result")
        return []
    names = universe()
    if lib is None:
        if log:
            log(f"building the promise library from the bank ({len(names)} companies of the four universes)")
        lib = promises.build(bank, names, log=log)
    result = evaluate(lib, names)
    out = []
    for tid in todo:
        m = result[tid]
        if GATE_TYPE[tid] == "report":
            rec = {"op": "result", "kind": KIND, "id": tid, "spec_sha": decls[tid]["spec_sha"], "gate_type": "report",
                   "metrics": {**m, "question": QUESTION[tid], "stat": (m.get("all") or {}).get("in68", float("nan")),
                               "words": words(tid, m)},
                   "p": None, "at": now_iso(), **ledger.git_state()}
        else:
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
        test(Bank(), trial=a.trial)
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("promisetest4", help="declare or run G4a, G4b: the second guess of a delivery (a pulled-in "
                                            "habit, ranges by the company's own spread)")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.set_defaults(fn=_cmd)
