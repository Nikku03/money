"""Declared tests of what was expected before a results report and what followed (brain/episodes.py,
docs/EARNINGS.md).

Every episode is one results report of the research, small, micro and newer-listing universes. The setup is known
at the close before the report (``p``); ``r1`` is the excess return over SPY from that close to the close of the
report session, the trade "made before the report, closed the next session"; ``post5`` is the excess return over
the five sessions after the report session. TRAIN is 2011-01-01..2023-12-31; TEST is 2024-01-01..2026-09-30.
Per calendar month (at least MIN_N episodes) the statistic is the rank correlation of a prediction with the outcome;
the monthly values are averaged (a busy month counts once) with a one-sided sign-flip p over the months.

The directions below were fixed on TRAIN years only (``research/earnings/explore_train.py``, its log beside it),
before any TEST year was read:

- **X1a, priced in.** The prediction is ``SIGN_RUN * run21``: how far the stock ran into the report against the
  market over 21 sessions, in its own usual moves. On TRAIN the further it ran, the worse it reacted (monthly rank
  correlation -0.038, t -6.4; and -0.034 on 2019-2023 inside TRAIN), so ``SIGN_RUN`` is -1.
- **X1b, habit.** The prediction is ``SIGN_HABIT * up8``: the share of its last eight reports on which it rose
  (TRAIN +0.021, t +3.4).
- **X1c, analogs.** The prediction is the mean ``r1`` of the K nearest past episodes in the setup space
  (``episodes.FEATS``, scaled by TRAIN's median and MAD), drawn only from episodes in months before the episode's
  own: the past setups most like this one, and how they reacted. K = 100 was the best of 100, 200 and 500 when
  2019-2023 was judged from 2011-2018 (all inside TRAIN).
- **X1d, after the reaction.** The prediction is ``SIGN_DRIFT * z1``, the report session's reaction in the stock's
  usual report moves, against ``post5``: does the reaction keep going over the next week. TRAIN leaned to keep
  going, weakly (+0.012, t +1.5), so ``SIGN_DRIFT`` is +1.
- **X1e** (reported, not gated): TEST episodes by their ``why`` (priced in, relief, confirmed, capitulation, surprise
  up or down, no reaction), with their mean reaction and what followed.
- **X1f, the expectation score.** The prediction is ``z(-run21) + z(-run5) + z(up8)``, each scaled by its TRAIN
  median and MAD, equal weights, nothing fitted: how much the price already expects (the run into the report, over
  a month and over a week) against how the stock usually does at its reports. TRAIN +0.050 (t +6.9) on 2011-2018
  and +0.052 (t +5.2) on 2019-2023.

Every gated test also needs its controls: a placebo that shuffles the prediction among each month's episodes reaches
p < 0.05 in at most 10% of draws, and an oracle that knows the outcome reaches p < 0.01. Then Benjamini-Hochberg
q < 0.10 across every gated test in the ledger (``research/run.py``). Results are written whatever they say. Trading
costs are not in the outcome; a one-session trade around a report pays its spread twice.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from brain import config, episodes
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research.eventtest import boot_interval, flip_p
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

IDS = ("X1a", "X1b", "X1c", "X1d", "X1e", "X1f")
GATE_TYPE = {"X1a": "direction", "X1b": "direction", "X1c": "analog", "X1d": "drift", "X1e": "report",
             "X1f": "direction"}
QUESTION = {"X1a": "does the run into a report predict the reaction (priced in)",
            "X1b": "does a stock's habit at its reports predict the reaction",
            "X1c": "do the most similar past setups predict the reaction (analogs)",
            "X1d": "does the reaction keep going or come back over the next week",
            "X1e": "what each kind of episode did (priced in, relief, ...)",
            "X1f": "does the expectation score (run-up against habit) predict the reaction"}
VERSION = 1
TRAIN = ("2011-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-09-30")
SIGN_RUN = -1        # fixed on TRAIN (research/earnings/explore_train.log)
SIGN_HABIT = 1       # fixed on TRAIN
SIGN_DRIFT = 1       # fixed on TRAIN
K = 100              # fixed on TRAIN (research/earnings/explore_analogs_train.log)
SCORE = (("run21", -1), ("run5", -1), ("up8", 1))
MIN_N = 30
P_MAX = 0.05
PLACEBO_FPR_MAX, ORACLE_P_MAX = 0.10, 0.01
N_FLIP, N_BOOT, N_PLACEBO, SEED = 10000, 2000, 200, 29
CLEAN = ("brain", "knowledge", "universe")
HERE = Path(__file__).resolve()


def code_files() -> list[Path]:
    from brain import bank, events, metrics, series, timeutil
    from brain.research import eventtest, run, stats
    return [HERE, *(Path(m.__file__).resolve()
                    for m in (episodes, events, metrics, bank, series, timeutil, eventtest, run, stats))]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def universe_shas() -> dict[str, str]:
    return {n: hashlib.sha256((config.ROOT / "universe" / f"{n}.json").read_bytes()).hexdigest()[:16]
            for n in episodes.UNIVERSES}


def spec(tid: str) -> dict:
    return {"id": tid, "version": VERSION, "train": list(TRAIN), "test": list(TEST), "feats": list(episodes.FEATS),
            "sign_run": SIGN_RUN, "sign_habit": SIGN_HABIT, "sign_drift": SIGN_DRIFT, "k": K, "min_n": MIN_N,
            "score": [list(x) for x in SCORE],
            "ran": episodes.RAN, "moved": episodes.MOVED, "universes": universe_shas(), "code_sha": code_sha(),
            "n_flip": N_FLIP, "n_boot": N_BOOT, "n_placebo": N_PLACEBO, "seed": SEED,
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
        if isinstance(rec, dict) and rec.get("op") == "declare" and rec.get("kind") == "earn" \
                and rec.get("id") == tid and rec.get("spec_sha") == sha:
            return True
    return False


def _declared(recs: list[dict], tid: str) -> dict | None:
    return next((r for r in recs if r["op"] == "declare" and r.get("kind") == "earn" and r["id"] == tid), None)


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
        rec = {"op": "declare", "kind": "earn", "id": tid, "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
               **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            log(f"declared {tid} ({rec['spec_sha']}): {QUESTION[tid]}; commit the ledger before running")
    return out


# ---------- the statistics (pure) ----------
def ranks(x: np.ndarray) -> np.ndarray:
    """Ranks scaled to 0..1, ties given their average rank (``up8`` takes nine values only)."""
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.bincount(inv, weights=r)
        r = sums[inv] / counts[inv]
    return r / max(len(x) - 1, 1)


def monthly_ic(month: np.ndarray, pred: np.ndarray, y: np.ndarray, min_n: int = MIN_N) -> tuple[list[str], np.ndarray]:
    """Per month with at least ``min_n`` usable episodes: the rank correlation of ``pred`` with ``y``."""
    ms, out = [], []
    for m in sorted(set(month.tolist())):
        k = (month == m) & np.isfinite(pred) & np.isfinite(y)
        if k.sum() < min_n:
            continue
        a, b = ranks(pred[k]), ranks(y[k])
        if a.std() == 0 or b.std() == 0:
            continue
        ms.append(m)
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return ms, np.array(out)


def shuffled(month: np.ndarray, x: np.ndarray, rng) -> np.ndarray:
    out = x.copy()
    for m in set(month.tolist()):
        k = np.flatnonzero(month == m)
        out[k] = x[rng.permutation(k)]
    return out


def ic_test(month: np.ndarray, pred: np.ndarray, y: np.ndarray, rng, n_placebo: int = N_PLACEBO) -> dict:
    ms, d = monthly_ic(month, pred, y)
    p = flip_p(d, 1, N_FLIP, rng) if len(d) else 1.0
    hits = 0
    for _ in range(n_placebo):
        _, dp = monthly_ic(month, shuffled(month, pred, rng), y)
        hits += bool(len(dp)) and flip_p(dp, 1, 2000, rng) < P_MAX
    _, do = monthly_ic(month, y, y)
    oracle_p = flip_p(do, 1, 2000, rng) if len(do) else 1.0
    fpr = hits / n_placebo if n_placebo else 0.0
    k = np.isfinite(pred) & np.isfinite(y)
    q = np.quantile(pred[k], [0.2, 0.8]) if k.sum() else [np.nan, np.nan]
    top, bot = y[k & (pred >= q[1])], y[k & (pred <= q[0])]
    return {"months": len(d), "rows": int(k.sum()), "first": ms[0] if ms else None, "last": ms[-1] if ms else None,
            "stat": float(d.mean()) if len(d) else float("nan"),
            "interval90": boot_interval(d, N_BOOT, rng) if len(d) else [float("nan")] * 2,
            "months_positive": int((d > 0).sum()), "p": p,
            "top_fifth": float(np.mean(top)) if len(top) else float("nan"),
            "bottom_fifth": float(np.mean(bot)) if len(bot) else float("nan"),
            "top_fifth_rose": float(np.mean(top > 0)) if len(top) else float("nan"),
            "bottom_fifth_rose": float(np.mean(bot > 0)) if len(bot) else float("nan"),
            "controls": {"placebo_fpr": fpr, "oracle_p": oracle_p,
                         "pass": bool(fpr <= PLACEBO_FPR_MAX and oracle_p < ORACLE_P_MAX)}}


def analog_predictions(train: list[dict], every: list[dict], test: list[dict], k: int = K) -> np.ndarray:
    """For each TEST episode, the mean r1 of its ``k`` nearest episodes among those in earlier months."""
    X_train = episodes.matrix(train)
    med, mad = episodes.scale(X_train)
    pool_rows = sorted(every, key=lambda r: r["date"])
    P = (episodes.matrix(pool_rows) - med) / mad
    P = np.clip(P, -6, 6)
    py = np.array([r["r1"] for r in pool_rows])
    pmonth = np.array([r["date"][:7] for r in pool_rows])
    out = np.full(len(test), np.nan)
    tm = np.array([r["date"][:7] for r in test])
    T = np.clip((episodes.matrix(test) - med) / mad, -6, 6)
    for m in sorted(set(tm.tolist())):
        idx = np.flatnonzero(tm == m)
        pool = np.flatnonzero(pmonth < m)
        if len(pool) < k:
            continue
        Pm, ym = P[pool], py[pool]
        for i in idx:
            nn = episodes.nearest(Pm, T[i], k)
            out[i] = float(np.mean(ym[nn]))
    return out


def score_scale(train: list[dict]) -> dict[str, tuple[float, float]]:
    """Each score part's TRAIN median and spread (1.4826 x MAD)."""
    out = {}
    for f, _ in SCORE:
        x = np.array([float(r[f]) for r in train])
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)) * 1.4826) or 1.0
        out[f] = (med, mad)
    return out


def expectation_score(row: dict, sc: dict[str, tuple[float, float]]) -> float:
    """z(-run21) + z(-run5) + z(up8), each clipped at 6 spreads."""
    return float(sum(sign * np.clip((float(row[f]) - sc[f][0]) / sc[f][1], -6, 6) for f, sign in SCORE))


def by_why(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        if r.get("why") is None:
            continue
        out.setdefault(r["why"], []).append(r)
    res = {}
    for k, v in out.items():
        a = np.array([[x["r1"], x["post5"] if x.get("post5") is not None else np.nan,
                       x["post21"] if x.get("post21") is not None else np.nan] for x in v], dtype=float)
        same = np.sign(a[:, 0]) == np.sign(a[:, 1])
        ok = np.isfinite(a[:, 1])
        res[k] = {"n": len(v), "r1": float(np.nanmean(a[:, 0])), "post5": float(np.nanmean(a[:, 1])),
                  "post21": float(np.nanmean(a[:, 2])), "kept_going": float(np.mean(same[ok])) if ok.any() else None}
    return res


def evaluate(rows: list[dict], rng=None) -> dict[str, dict]:
    """Every test's result from the episode library ``rows`` (pure, given the rows)."""
    rng = rng or np.random.default_rng(SEED)
    full = episodes.complete(rows, "r1")
    train = [r for r in full if TRAIN[0] <= r["date"] <= TRAIN[1]]
    test = [r for r in full if TEST[0] <= r["date"] <= TEST[1]]
    month = np.array([r["date"][:7] for r in test])
    y1 = np.array([r["r1"] for r in test])
    out = {}
    run21 = np.array([r["run21"] for r in test])
    out["X1a"] = ic_test(month, SIGN_RUN * run21, y1, rng)
    up8 = np.array([r["up8"] for r in test])
    out["X1b"] = ic_test(month, SIGN_HABIT * up8, y1, rng)
    every = [r for r in full if r["date"] <= TEST[1]]
    pred = analog_predictions(train, every, test, k=K)
    out["X1c"] = ic_test(month, pred, y1, rng)
    post = [r for r in test if r.get("post5") is not None]
    pm = np.array([r["date"][:7] for r in post])
    z1 = np.array([r["r1"] / r["med8"] if r["med8"] > 0 else 0.0 for r in post])
    y5 = np.array([r["post5"] for r in post])
    out["X1d"] = ic_test(pm, SIGN_DRIFT * z1, y5, rng)
    out["X1e"] = {"by_why": by_why(test), "train_by_why": by_why(train), "rows": len(test)}
    sc = score_scale(train)
    out["X1f"] = ic_test(month, np.array([expectation_score(r, sc) for r in test]), y1, rng)
    return out


def words(tid: str, m: dict) -> str:
    if tid == "X1e":
        parts = [f"{k}: {v['n']} episodes, reaction {v['r1'] * 100:+.2f}%, next week {v['post5'] * 100:+.2f}%, "
                 f"kept going {v['kept_going'] * 100:.0f}%" for k, v in sorted(m["by_why"].items(),
                                                                              key=lambda kv: -kv[1]["n"])]
        return f"{m['rows']} TEST episodes by kind. " + "; ".join(parts) + "."
    c = m["controls"]
    key = "the next week's drift" if tid == "X1d" else "the reaction"
    return (f"{m['rows']} TEST episodes in {m['months']} months; monthly rank correlation with {key} "
            f"{m['stat']:+.4f} (positive in {m['months_positive']} of {m['months']} months). The top fifth by the "
            f"prediction averaged {m['top_fifth'] * 100:+.2f}% ({m['top_fifth_rose'] * 100:.0f}% rose), the bottom "
            f"fifth {m['bottom_fifth'] * 100:+.2f}% ({m['bottom_fifth_rose'] * 100:.0f}% rose). Controls: shuffled "
            f"predictions reached p < 0.05 in {c['placebo_fpr'] * 100:.1f}% of draws; the oracle p {c['oracle_p']:.4f}.")


def test(bank: Bank, *, trial: bool = False, log=print, rows: list[dict] | None = None) -> list[dict]:
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
    todo = [t for t in IDS if not any(r["op"] == "result" and r.get("kind") == "earn" and r["id"] == t for r in recs)]
    if not todo:
        if log:
            log("every earnings test already has its result")
        return []
    if rows is None:
        if log:
            log("building the episode library from the bank (every report of the four universes)")
        rows = episodes.build(bank, log=log)
        episodes.save(rows)
    result = evaluate(rows)
    out = []
    for tid in todo:
        m = result[tid]
        if tid == "X1e":
            rec = {"op": "result", "kind": "earn", "id": tid, "spec_sha": decls[tid]["spec_sha"],
                   "gate_type": "report", "metrics": {**m, "question": QUESTION[tid], "stat": float("nan"),
                                                       "words": words(tid, m)},
                   "p": None, "at": now_iso(), **ledger.git_state()}
        else:
            gate = {"stat_positive": m["stat"] > 0, "p_ok": m["p"] < P_MAX, "controls": m["controls"]["pass"]}
            gate["pass"] = bool(all(gate.values()))
            rec = {"op": "result", "kind": "earn", "id": tid, "spec_sha": decls[tid]["spec_sha"],
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
        rows = episodes.load() if (a.cached and a.trial) else None   # a real run builds the library afresh
        test(Bank(), trial=a.trial, rows=rows or None)
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("earntest", help="declare or run X1a-X1e: what was expected before a report and what followed")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.add_argument("--cached", action="store_true", help="use the episode library already built (var/episodes)")
    p.set_defaults(fn=_cmd)
