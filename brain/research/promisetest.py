"""Declared tests of the promise record (brain/promises.py, docs/PROMISES.md): does a company deliver what it says,
and what follows a change in what it says.

A promise is revenue guidance read from an earnings release; its outcome is the revenue first reported for the
promised period; ``surprise`` is the outcome over the promise's midpoint, less one. The tests use the research
universe (``universe/us_broad.json``), the companies whose releases were re-read with their forward-looking
passages. TRAIN is promises made 2011-01-01..2023-12-31; TEST is 2024-01-01..2026-09-30. Per reporting season (the
calendar quarter a promise was made in, at least MIN_N of them) the statistic is the rank correlation of a
prediction with the outcome; the seasons' values are averaged (a busy season counts once), with a one-sided
sign-flip p over the seasons.

The directions below were fixed on TRAIN only (``research/promises/explore_train.py``, its log beside it), before
any TEST promise was held against anything:

- **G1a, the habit.** The prediction is the company's usual surprise when the promise was made: the median surprise
  of its last eight quarterly promises whose outcomes were known by then (at least four). The outcome is the
  surprise on the new promise. Does a company that usually beats its own number keep beating it by as much?
- **G1b, the stretch.** The prediction is ``SIGN_STRETCH * stretch``: the growth over the year-ago quarter that the
  promise asks for, less the growth the company last reported. The outcome is the surprise. Does a promise that
  asks for more than the company has been doing fall short more often?
- **G1c, after a change of the year's promise.** The prediction is the change in the company's promise for its
  fiscal year against its previous promise for the same year (the midpoint's move). The outcome is ``post5``: the
  move against the market over the five sessions after the report session that carried it. Does the price keep
  going after a raise or a cut?
- **G1d** (reported, not gated): TEST releases by what they said (delivered above, inside or below the last promise;
  more or less against its midpoint than the company's habit; the year's promise raised, held or cut), with the
  report session's move and the week after. The same day: an explanation of a move, not a forecast.

G1a also reports, not gated: how close the delivery came to the promise's midpoint, and to the midpoint moved by
the habit; and how often the delivery fell inside the ranges ``promises.BAND68`` and ``promises.BAND90`` around
that (fixed on TRAIN).

Not declared, because TRAIN showed nothing: whether the record says how the stock reacts on the day the promised
quarter is reported (docs/PROMISES.md).

Every gated test also needs its controls: a placebo that shuffles the prediction among each season's promises
reaches p < 0.05 in at most 10% of draws, and an oracle that knows the outcome reaches p < 0.01. Then
Benjamini-Hochberg q < 0.10 across every gated test in the ledger (``research/run.py``). Results are written
whatever they say.

The files that decide the numbers are pinned (``code_files``). ``research/run.py`` only gathers the ledger into
RESULTS.md and is not pinned: pinning it made every later kind of test break the earlier ones' pins.
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
from brain.research.earntest import ranks, shuffled
from brain.research.eventtest import boot_interval, flip_p
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

KIND = "promise"
IDS = ("G1a", "G1b", "G1c", "G1d")
GATE_TYPE = {"G1a": "delivery", "G1b": "delivery", "G1c": "drift", "G1d": "report"}
QUESTION = {"G1a": "does a company's habit of beating its own revenue promise carry to the next promise",
            "G1b": "does a promise that asks for more than the company has been doing fall short",
            "G1c": "does the price keep going the week after the year's promise is raised or cut",
            "G1d": "the report session's move by what the release said"}
VERSION = 1
UNIVERSE = "us_broad"
TRAIN = ("2011-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-09-30")
SIGN_STRETCH = -1    # fixed on TRAIN (research/promises/explore_train.log)
SIGN_CHANGE = 1      # fixed on TRAIN
MIN_N = 20           # promises in a season
HERE_DAYS = 4        # the report session comes within this many days of the release
P_MAX = 0.05
PLACEBO_FPR_MAX, ORACLE_P_MAX = 0.10, 0.01
N_FLIP, N_BOOT, N_PLACEBO, SEED = 10000, 2000, 200, 31
CLEAN = ("brain", "knowledge", "universe")
HERE = Path(__file__).resolve()


def code_files() -> list[Path]:
    from brain import bank, events, metrics, series, timeutil
    from brain.research import earntest, eventtest, stats
    from brain.sources import releases, sec
    return [HERE, *(Path(m.__file__).resolve()
                    for m in (promises, episodes, releases, sec, events, metrics, bank, series, timeutil, earntest,
                              eventtest, stats))]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def universe() -> list[str]:
    return sorted({t.upper() for t in json.loads((config.ROOT / "universe" / f"{UNIVERSE}.json")
                                                 .read_text())["tickers"]})


def universe_sha() -> str:
    return hashlib.sha256((config.ROOT / "universe" / f"{UNIVERSE}.json").read_bytes()).hexdigest()[:16]


def spec(tid: str) -> dict:
    return {"id": tid, "version": VERSION, "train": list(TRAIN), "test": list(TEST), "universe": UNIVERSE,
            "universe_sha": universe_sha(), "sign_stretch": SIGN_STRETCH, "sign_change": SIGN_CHANGE,
            "min_n": MIN_N, "here_days": HERE_DAYS, "record": promises.RECORD, "min_record": promises.MIN_RECORD,
            "band68": list(promises.BAND68), "band90": list(promises.BAND90), "reader": promises.READER_MIN,
            "code_sha": code_sha(), "n_flip": N_FLIP, "n_boot": N_BOOT, "n_placebo": N_PLACEBO, "seed": SEED,
            "gate_type": GATE_TYPE[tid],
            "gate": ({"stat_gt": 0.0, "p_lt": P_MAX, "bh_q_lt": 0.10, "placebo_fpr_max": PLACEBO_FPR_MAX,
                      "oracle_p_max": ORACLE_P_MAX} if GATE_TYPE[tid] != "report" else None)}


def spec_sha(s: dict) -> str:
    return hashlib.sha256(canon(s).encode()).hexdigest()[:16]


def _git(*args):
    import subprocess
    return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=False)


def tree_dirty() -> list[str]:
    r = _git("status", "--porcelain", "--", *CLEAN)
    return [ln for ln in r.stdout.splitlines() if ln.strip()] if r.returncode == 0 else [f"git: {r.stderr.strip()}"]


def lineage_problem(commit: str | None) -> str | None:
    if not commit:
        return "the declaration has no commit"
    if _git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        return f"HEAD is not the declare commit {commit[:12]} or a descendant of it"
    root = config.ROOT.resolve()
    files = [p.relative_to(root).as_posix() for p in code_files()]
    changed = [x for x in _git("diff", "--name-only", commit, "HEAD", "--", *files).stdout.splitlines() if x.strip()]
    return f"code changed since the declare commit {commit[:12]}: {', '.join(changed)}" if changed else None


def committed_declaration(tid: str, sha: str) -> bool:
    r = _git("show", "HEAD:research/ledger.jsonl")
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
    if not trial and (dirty := tree_dirty()):
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


# ---------- the table (pure, given the two libraries) ----------
def season(d: str) -> str:
    """The reporting season of a date: its calendar quarter, "2025Q3"."""
    return f"{d[:4]}Q{(int(d[5:7]) - 1) // 3 + 1}"


def _first_after(eps: list[dict], d: str, within: int) -> dict | None:
    for e in eps:
        if e["date"] >= d:
            return e if (date.fromisoformat(e["date"]) - date.fromisoformat(d)).days <= within else None
    return None


def table(lib: list[dict], eps: list[dict], tickers=None) -> tuple[list[dict], list[dict]]:
    """(quarterly promises, changes of a year's promise), each row with what was known when it was made and what
    followed. ``lib`` is the promise library, ``eps`` the episode library; ``tickers`` limits both.

    A quarterly row: ``usual`` (the record's usual surprise when the promise was made, None under MIN_RECORD),
    ``stretch``, ``surprise`` (None until graded), ``got`` (what the same release delivered against the promise
    before it), and the report session's ``r1`` and ``post5`` when the episode library has it.
    A yearly row: ``change`` against the previous promise for the same fiscal year, ``r1`` and ``post5``."""
    keep = set(tickers) if tickers is not None else None
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in lib:
        if keep is None or r["t"] in keep:
            by_t[r["t"]].append(r)
    ep_t: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        if keep is None or e["t"] in keep:
            ep_t[e["t"]].append(e)
    for v in ep_t.values():
        v.sort(key=lambda e: e["date"])
    Q, Y = [], []
    for t in sorted(by_t):
        rs = sorted(by_t[t], key=lambda r: r["known_at"])
        prev = None
        for r in rs:
            here = _first_after(ep_t.get(t, []), r["made"], HERE_DAYS)
            if r["horizon"] == "year":
                if r.get("change") is not None and here is not None:
                    Y.append({"t": t, "made": r["made"], "season": season(r["made"]), "change": r["change"],
                              "changed": r["changed"], "r1": here["r1"], "post5": here.get("post5")})
                continue
            rec = promises.record(rs, r["known_at"])
            got = None
            if (prev is not None and prev.get("surprise") is not None and prev.get("period_end")
                    and prev["period_end"] < r["made"]
                    and (date.fromisoformat(r["made"]) - date.fromisoformat(prev["period_end"])).days <= 100):
                got = prev
            Q.append({"t": t, "made": r["made"], "season": season(r["made"]),
                      "usual": rec["surprise"] if rec else None, "stretch": r.get("stretch"),
                      "surprise": r.get("surprise"), "grade": r.get("grade"),
                      "got": got["surprise"] if got else None, "got_grade": got["grade"] if got else None,
                      "r1": here["r1"] if here else None, "post5": here.get("post5") if here else None})
            prev = r
    return Q, Y


def _col(rows: list[dict], k: str, sign: float = 1.0) -> np.ndarray:
    return np.array([sign * r[k] if r.get(k) is not None else np.nan for r in rows], dtype=float)


# ---------- the statistics (pure) ----------
def season_ic(group: np.ndarray, pred: np.ndarray, y: np.ndarray, min_n: int = MIN_N) -> tuple[list[str], np.ndarray]:
    """Per season with at least ``min_n`` usable rows: the rank correlation of ``pred`` with ``y``."""
    gs, out = [], []
    for g in sorted(set(group.tolist())):
        k = (group == g) & np.isfinite(pred) & np.isfinite(y)
        if k.sum() < min_n:
            continue
        a, b = ranks(pred[k]), ranks(y[k])
        if a.std() == 0 or b.std() == 0:
            continue
        gs.append(g)
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return gs, np.array(out)


def ic_test(group: np.ndarray, pred: np.ndarray, y: np.ndarray, rng, n_placebo: int = N_PLACEBO) -> dict:
    ok = np.isfinite(pred) & np.isfinite(y)
    group, pred, y = group[ok], pred[ok], y[ok]
    gs, d = season_ic(group, pred, y)
    p = flip_p(d, 1, N_FLIP, rng) if len(d) else 1.0
    hits = 0
    for _ in range(n_placebo):
        _, dp = season_ic(group, shuffled(group, pred, rng), y)
        hits += bool(len(dp)) and flip_p(dp, 1, 2000, rng) < P_MAX
    _, do = season_ic(group, y, y)
    oracle_p = flip_p(do, 1, 2000, rng) if len(do) else 1.0
    fpr = hits / n_placebo if n_placebo else 0.0
    q = np.quantile(pred, [0.2, 0.8]) if len(pred) else [np.nan, np.nan]
    top, bot = y[pred >= q[1]], y[pred <= q[0]]
    return {"seasons": len(d), "rows": int(len(y)), "first": gs[0] if gs else None, "last": gs[-1] if gs else None,
            "stat": float(d.mean()) if len(d) else float("nan"),
            "interval90": boot_interval(d, N_BOOT, rng) if len(d) else [float("nan")] * 2,
            "seasons_positive": int((d > 0).sum()), "p": p,
            "top_fifth": float(np.mean(top)) if len(top) else float("nan"),
            "bottom_fifth": float(np.mean(bot)) if len(bot) else float("nan"),
            "top_fifth_median": float(np.median(top)) if len(top) else float("nan"),
            "bottom_fifth_median": float(np.median(bot)) if len(bot) else float("nan"),
            "controls": {"placebo_fpr": fpr, "oracle_p": oracle_p,
                         "pass": bool(fpr <= PLACEBO_FPR_MAX and oracle_p < ORACLE_P_MAX)}}


def habit_report(rows: list[dict]) -> dict:
    """How close the delivery came to the midpoint and to the midpoint moved by the habit, and how often it fell in
    the fixed ranges around that (rows with a record and a graded outcome)."""
    u, s = _col(rows, "usual"), _col(rows, "surprise")
    k = np.isfinite(u) & np.isfinite(s)
    if not k.any():
        return {"rows": 0}
    e = s[k] - u[k]
    return {"rows": int(k.sum()), "miss_midpoint": float(np.median(np.abs(s[k]))),
            "miss_habit": float(np.median(np.abs(e))),
            "in68": float(np.mean((e >= promises.BAND68[0]) & (e <= promises.BAND68[1]))),
            "in90": float(np.mean((e >= promises.BAND90[0]) & (e <= promises.BAND90[1]))),
            "missed_range": float(np.mean([r["grade"] == "missed" for r, ok in zip(rows, k) if ok]))}


def by_what(Q: list[dict], Y: list[dict]) -> dict:
    """The report session's move, and the week after, by what the release said."""
    def cell(rows):
        a = np.array([r["r1"] for r in rows], dtype=float)
        b = np.array([r["post5"] if r.get("post5") is not None else np.nan for r in rows], dtype=float)
        return {"n": len(rows), "r1": float(a.mean()), "r1_median": float(np.median(a)),
                "rose": float(np.mean(a > 0)), "post5": float(np.nanmean(b)) if np.isfinite(b).any() else None}
    out = {"delivered": {}, "habit": {}, "year": {}}
    for g in ("beat", "met", "missed"):
        rows = [r for r in Q if r.get("got_grade") == g and r.get("r1") is not None]
        if rows:
            out["delivered"][g] = cell(rows)
    known = [r for r in Q if r.get("got") is not None and r.get("usual") is not None and r.get("r1") is not None]
    for g, rows in (("above", [r for r in known if r["got"] > r["usual"]]),
                    ("below", [r for r in known if r["got"] <= r["usual"]])):
        if rows:
            out["habit"][g] = cell(rows)
    for g in ("raised", "held", "cut"):
        rows = [r for r in Y if r.get("changed") == g and r.get("r1") is not None]
        if rows:
            out["year"][g] = cell(rows)
    return out


def evaluate(lib: list[dict], eps: list[dict], tickers=None, rng=None) -> dict[str, dict]:
    """Every test's result from the two libraries (pure, given the rows)."""
    rng = rng or np.random.default_rng(SEED)
    Q, Y = table(lib, eps, tickers)
    q = [r for r in Q if TEST[0] <= r["made"] <= TEST[1]]
    y = [r for r in Y if TEST[0] <= r["made"] <= TEST[1]]
    gq = np.array([r["season"] for r in q])
    out = {}
    out["G1a"] = {**ic_test(gq, _col(q, "usual"), _col(q, "surprise"), rng), "habit": habit_report(q)}
    out["G1b"] = ic_test(gq, _col(q, "stretch", SIGN_STRETCH), _col(q, "surprise"), rng)
    gy = np.array([r["season"] for r in y])
    out["G1c"] = ic_test(gy, _col(y, "change", SIGN_CHANGE), _col(y, "post5"), rng)
    out["G1d"] = {"by_what": by_what(q, y), "rows": len(q), "year_rows": len(y),
                  "companies": len({r["t"] for r in q})}
    return out


def words(tid: str, m: dict) -> str:
    if tid == "G1d":
        said = {"beat": "delivered above its last promise", "met": "delivered inside it", "missed": "delivered below it",
                "above": "delivered more against its midpoint than its habit", "below": "delivered less than its habit",
                "raised": "raised the year's promise", "held": "held it", "cut": "cut it"}
        parts = []
        for grp in ("delivered", "habit", "year"):
            for k, v in m["by_what"][grp].items():
                wk = f", the week after {v['post5'] * 100:+.2f}%" if v.get("post5") is not None else ""
                parts.append(f"{said[k]}: {v['n']} releases, the report session {v['r1'] * 100:+.2f}% against the "
                             f"market (median {v['r1_median'] * 100:+.2f}%, {v['rose'] * 100:.0f}% rose){wk}")
        return (f"{m['rows']} quarterly promises of {m['companies']} companies and {m['year_rows']} changes of a "
                f"year's promise on TEST. " + "; ".join(parts) + ".")
    c = m["controls"]
    if tid == "G1c":
        return (f"{m['rows']} changes of a year's promise in {m['seasons']} seasons; rank correlation of the change "
                f"with the next five sessions {m['stat']:+.4f} (positive in {m['seasons_positive']} of "
                f"{m['seasons']} seasons). The fifth raised most averaged {m['top_fifth'] * 100:+.2f}% against the "
                f"market over the week, the fifth cut most {m['bottom_fifth'] * 100:+.2f}%. Controls: shuffled "
                f"predictions reached p < 0.05 in {c['placebo_fpr'] * 100:.1f}% of draws; the oracle p "
                f"{c['oracle_p']:.4f}.")
    what = "its usual surprise" if tid == "G1a" else "the stretch (less is better)"
    base = (f"{m['rows']} quarterly promises in {m['seasons']} seasons; rank correlation of {what} with the surprise "
            f"on the new promise {m['stat']:+.4f} (positive in {m['seasons_positive']} of {m['seasons']} seasons). "
            f"The top fifth by the prediction delivered a median {m['top_fifth_median'] * 100:+.2f}% against its "
            f"midpoint, the bottom fifth {m['bottom_fifth_median'] * 100:+.2f}%. Controls: shuffled predictions "
            f"reached p < 0.05 in {c['placebo_fpr'] * 100:.1f}% of draws; the oracle p {c['oracle_p']:.4f}.")
    if tid == "G1a" and (h := m.get("habit")) and h.get("rows"):
        base += (f" Reported, not gated: of {h['rows']} promises with a record, the delivery was a median "
                 f"{h['miss_midpoint'] * 100:.2f}% from the midpoint and {h['miss_habit'] * 100:.2f}% from the "
                 f"midpoint moved by the habit; it fell inside the 2-in-3 range {h['in68'] * 100:.1f}% of the time "
                 f"and inside the 9-in-10 range {h['in90'] * 100:.1f}%; {h['missed_range'] * 100:.1f}% missed the "
                 f"company's own range.")
    return base


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
            if dirty := tree_dirty():
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
            log("every promise test already has its result")
        return []
    names = universe()
    if lib is None:
        if log:
            log(f"building the promise library from the bank ({len(names)} companies of the research universe)")
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
        if tid == "G1d":
            rec = {"op": "result", "kind": KIND, "id": tid, "spec_sha": decls[tid]["spec_sha"],
                   "gate_type": "report", "metrics": {**m, "question": QUESTION[tid], "stat": float("nan"),
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
        eps = episodes.load() if (a.cached and a.trial) else None     # a real run builds the libraries afresh
        test(Bank(), trial=a.trial, eps=eps or None)
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("promisetest", help="declare or run G1a-G1d: does a company deliver what it says")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.add_argument("--cached", action="store_true", help="use the episode library already built (var/episodes)")
    p.set_defaults(fn=_cmd)
