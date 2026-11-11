"""Declared test of the earnings-aware shock radar S3 (brain/catalysts.py, docs/SHOCKS.md "Catalysts").

**S3a** asks whether S3's chance of a 20%+ session against SPY in the next five beats S1's in the weeks a stock's
results report is expected: S1's ``earn_win`` (last year's report moved on 52 weeks) or S3's ``earnq`` (a quarter
after the last report), 1-8 days ahead. These are the weeks whose catalyst is known in advance. S1
(``knowledge/shocks_s1.json``) and S3 (``knowledge/shocks_s3.json``) were fitted on the research universe's
stock-weeks of 2011-2023 and are pinned here by their shas. TEST is every research-universe stock-week ending in
2024-01-01..2026-09-30 whose five sessions have closed and whose report is expected. Per stock-week the score is the
log loss of the chance; per month, S1's mean loss minus S3's (calendar time: a busy month counts once). The
statistic is the mean of those monthly differences, with a one-sided sign-flip p. S3a passes when the difference is
positive with p < 0.05; S3's mean chance over TEST is within CAL of the actual rate, and so is the rate among each
week's top twentieth of TEST by S3; the controls pass (S3's chances shuffled among each week's TEST stocks reach
p < 0.05 in at most 10% of draws; an oracle that knows the outcome reaches p < 0.01); and Benjamini-Hochberg
q < 0.10 across every gated test in the ledger.

Reported beside it, not gated: the same comparison on every stock-week of the period, and the share of each
model's week's top 5% among the shocks of report weeks.

Results are written whatever they say.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from brain import catalysts, config, shocks
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research.eventtest import boot_interval, flip_p
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.research.shocktest import auc, logloss, monthly, week_rank
from brain.timeutil import now_iso

IDS = ("S3a",)
VERSION = 1
TEST = ("2024-01-01", "2026-09-30")
P_MAX = 0.05
CAL = (0.67, 1.5)          # actual over predicted, overall and in the top twentieth
TOP = 0.05
PLACEBO_FPR_MAX = 0.10
ORACLE_P_MAX = 0.01
N_FLIP, N_BOOT, N_PLACEBO, SEED = 10000, 2000, 200, 13
CLEAN = ("brain", "knowledge", "universe")
HERE = Path(__file__).resolve()
MODELS = ("knowledge/shocks_s1.json", "knowledge/shocks_s3.json")


def code_files() -> list[Path]:
    """What the numbers run through: this file, both models' code, the bank's view, the series store, the
    release-window rule, the time rules, the sign-flip helpers, the shock test's scores, and the verdicts."""
    from brain import bank, events, series, timeutil
    from brain.research import eventtest, run, shocktest, stats
    return [HERE, *(Path(m.__file__).resolve()
                    for m in (catalysts, shocks, events, bank, series, timeutil, eventtest, shocktest, run, stats))]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def universe_shas() -> dict[str, str]:
    return {n: hashlib.sha256((config.ROOT / "universe" / f"{n}.json").read_bytes()).hexdigest()[:16]
            for n in shocks.UNIVERSES}


def spec(tid: str) -> dict:
    s1, s3 = shocks.load(), catalysts.load()
    if s1 is None or s3 is None:
        raise Refused("fit both models first (python -m brain shocks fit; python -m brain catalysts fit) and commit them")
    return {"id": tid, "version": VERSION, "test": list(TEST), "shock": shocks.SHOCK, "h": shocks.H,
            "universes": universe_shas(), "code_sha": code_sha(), "s1_sha": shocks.model_sha(),
            "s3_sha": catalysts.model_sha(), "train": s3["train"], "added": list(catalysts.ADDED),
            "due": {"habit_n": catalysts.HABIT_N, "habit_min": catalysts.HABIT_MIN, "quarter_days": catalysts.QUARTER,
                    "ahead_days": catalysts.AHEAD},
            "n_flip": N_FLIP, "n_boot": N_BOOT, "n_placebo": N_PLACEBO, "seed": SEED,
            "controls": {"placebo_fpr_max": PLACEBO_FPR_MAX, "oracle_p_max": ORACLE_P_MAX},
            "gate": {"diff_gt": 0.0, "p_lt": P_MAX, "cal": list(CAL), "top": TOP, "bh_q_lt": 0.10}}


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
    files = [p.relative_to(root).as_posix() for p in code_files()] + list(MODELS)
    changed = [x for x in _git("diff", "--name-only", commit, "HEAD", "--", *files).stdout.splitlines() if x.strip()]
    return f"code or a model changed since the declare commit {commit[:12]}: {', '.join(changed)}" if changed else None


def committed_declaration(tid: str, sha: str) -> bool:
    r = _git("show", "HEAD:research/ledger.jsonl")
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("op") == "declare" and rec.get("kind") == "shock" \
                and rec.get("id") == tid and rec.get("spec_sha") == sha:
            return True
    return False


def _declared(recs: list[dict], tid: str) -> dict | None:
    return next((r for r in recs if r["op"] == "declare" and r.get("kind") == "shock" and r["id"] == tid), None)


def declare(*, trial: bool = False, log=print) -> list[dict]:
    """Declare S3a (once; a changed spec is refused: a change is a new id)."""
    trial = is_trial(trial)
    led, _ = paths(trial)
    if not trial and (dirty := tree_dirty()):
        raise Refused("commit brain/, knowledge/ and universe/ first; a declaration pins the committed code and "
                      "models: " + "; ".join(dirty))
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
        rec = {"op": "declare", "kind": "shock", "id": tid, "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
               **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            log(f"declared {tid} ({rec['spec_sha']}): TEST {TEST[0]}..{TEST[1]}; commit the ledger before running it")
    return out


# ---------- S3a ----------
def _capture(y: np.ndarray, rank: np.ndarray, k: float) -> float:
    return float(y[rank <= k].sum() / y.sum()) if y.sum() else float("nan")


def controls(week: np.ndarray, month: np.ndarray, y: np.ndarray, p3: np.ndarray, loss1: np.ndarray, rng) -> dict:
    """Placebo: S3's chances shuffled among each week's TEST stocks. Oracle: a chance that knows the outcome."""
    groups = [np.flatnonzero(week == w) for w in np.unique(week)]
    hits = 0
    for _ in range(N_PLACEBO):
        perm = np.arange(len(y))
        for idx in groups:
            perm[idx] = rng.permutation(idx)
        _, d = monthly(month, loss1 - logloss(p3[perm], y))
        hits += flip_p(d, 1, 999, rng) < P_MAX
    _, d = monthly(month, loss1 - logloss(np.where(y > 0, 0.9, 0.001), y))
    oracle_p = flip_p(d, 1, N_FLIP, rng)
    fpr = hits / N_PLACEBO
    return {"n_placebo": N_PLACEBO, "placebo_fpr": fpr, "oracle_p": oracle_p,
            "pass": bool(fpr <= PLACEBO_FPR_MAX and oracle_p <= ORACLE_P_MAX)}


def evaluate(rows: list[dict], s1: dict, s3: dict) -> dict:
    """S3 against S1 on the report weeks among ``rows`` (gated), and on every row (reported)."""
    rng = np.random.default_rng(SEED)
    due = [r for r in rows if r["due"] > 0]
    y = np.array([r["y"] for r in due], float)
    week = np.array([r["date"] for r in due])
    month = np.array([d[:7] for d in week])
    p1, p3 = shocks.chances(s1["s1"], due), catalysts.chances(s3, due)
    l1, l3 = logloss(p1, y), logloss(p3, y)
    months, d = monthly(month, l1 - l3)
    p = flip_p(d, 1, N_FLIP, rng)
    r3, r1 = week_rank(week, p3), week_rank(week, p1)
    top = r3 <= TOP
    ratio = float(y.mean() / p3.mean()) if len(y) else float("nan")
    top_ratio = float(y[top].mean() / p3[top].mean()) if top.any() else float("nan")
    # every stock-week, reported only
    ya = np.array([r["y"] for r in rows], float)
    wa = np.array([r["date"] for r in rows])
    ma = np.array([x[:7] for x in wa])
    pa1, pa3 = shocks.chances(s1["s1"], rows), catalysts.chances(s3, rows)
    la1, la3 = logloss(pa1, ya), logloss(pa3, ya)
    ams, ad = monthly(ma, la1 - la3)
    ra3, ra1 = week_rank(wa, pa3), week_rank(wa, pa1)
    duea = np.array([r["due"] > 0 for r in rows])
    shocked = y > 0
    metrics = {
        "stock_weeks": len(due), "stocks": len({r["t"] for r in due}), "months": len(months),
        "first": months[0] if months else None, "last": months[-1] if months else None,
        "loss_s3": float(l3.mean()), "loss_s1": float(l1.mean()), "diff": float(d.mean()) if len(d) else float("nan"),
        "interval90": boot_interval(d, N_BOOT, rng), "months_better": int((d > 0).sum()), "p": p,
        "rate": float(y.mean()), "mean_s3": float(p3.mean()), "mean_s1": float(p1.mean()), "cal_ratio": ratio,
        "top_rate": float(y[top].mean()) if top.any() else float("nan"),
        "top_chance": float(p3[top].mean()) if top.any() else float("nan"), "top_ratio": top_ratio,
        "auc_s3": auc(p3, y), "auc_s1": auc(p1, y),
        "capture_s3": {f"top{int(k * 100)}": _capture(y, r3, k) for k in (0.05, 0.10)},
        "capture_s1": {f"top{int(k * 100)}": _capture(y, r1, k) for k in (0.05, 0.10)},
        "up_share": float(np.array([r["up"] for r in due])[shocked].mean()) if shocked.any() else float("nan"),
        "all_weeks": {"stock_weeks": len(rows), "rate": float(ya.mean()), "due_share": float(duea.mean()),
                      "due_shock_share": float(ya[duea].sum() / ya.sum()) if ya.sum() else float("nan"),
                      "loss_s3": float(la3.mean()), "loss_s1": float(la1.mean()),
                      "diff": float(ad.mean()) if len(ad) else float("nan"), "months_better": int((ad > 0).sum()),
                      "months": len(ams), "p": flip_p(ad, 1, N_FLIP, rng),
                      "auc_s3": auc(pa3, ya), "auc_s1": auc(pa1, ya),
                      "top5_s3": _capture(ya, ra3, 0.05), "top5_s1": _capture(ya, ra1, 0.05),
                      "due_top5_s3": _capture(ya[duea], ra3[duea], 0.05), "due_top5_s1": _capture(ya[duea], ra1[duea], 0.05)},
    }
    gate = {"diff_positive": metrics["diff"] > 0, "p_ok": p < P_MAX,
            "cal_ok": CAL[0] <= ratio <= CAL[1], "top_cal_ok": CAL[0] <= top_ratio <= CAL[1]}
    gate["pass"] = bool(all(gate.values()))
    return {"metrics": metrics, "gate": gate, "controls": controls(week, month, y, p3, l1, rng), "p": p}


def test(bank: Bank, *, trial: bool = False, log=print, tickers=None) -> list[dict]:
    """Run S3a once, after its declaration is committed; the result is written whatever it says."""
    from brain.research import run
    trial = is_trial(trial)
    led, res = paths(trial)
    recs = ledger.read(led)
    todo = []
    for tid in IDS:
        decl = _declared(recs, tid)
        if decl is None:
            raise Refused(f"{tid} is not declared")
        if any(r["op"] == "result" and r.get("kind") == "shock" and r["id"] == tid for r in recs):
            continue
        s = decl["spec"]
        if spec_sha(s) != decl["spec_sha"]:
            raise Refused(f"{tid}: its declared spec does not match its spec_sha")
        if s["s1_sha"] != shocks.model_sha() or s["s3_sha"] != catalysts.model_sha():
            raise Refused(f"{tid}: a model is not the declared one")
        if not trial:
            if dirty := tree_dirty():
                raise Refused("the working tree has changes under brain/, knowledge/ or universe/: " + "; ".join(dirty))
            if why := lineage_problem(decl.get("commit")):
                raise Refused(f"{tid}: {why}")
            if s["code_sha"] != code_sha():
                raise Refused(f"{tid}: the code is not the declared code")
            if s["universes"] != universe_shas():
                raise Refused(f"{tid}: a universe file changed since the declaration; declare a new id")
            if not committed_declaration(tid, decl["spec_sha"]):
                raise Refused(f"{tid}: commit the ledger with its declaration before running it")
        todo.append(decl)
    if not todo:
        raise Refused("S3a already has a result; a rerun needs research withdraw and a new id")
    rows = catalysts.dataset(bank, tickers or shocks.universe_tickers(), TEST[0], TEST[1], log=log)
    out = []
    for decl in todo:
        r = evaluate(rows, shocks.load(), catalysts.load())
        rec = {"op": "result", "kind": "shock", "id": decl["id"], "spec_sha": decl["spec_sha"],
               "gate_type": "logloss_report", **r, "at": now_iso(), **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            m = r["metrics"]
            log(f"{decl['id']}: {m['stock_weeks']} report-week stock-weeks in {m['months']} months; log loss S3 "
                f"{m['loss_s3']:.5f} vs S1 {m['loss_s1']:.5f} (monthly difference {m['diff']:+.5f}, p {m['p']:.4f}); "
                f"chance {m['mean_s3'] * 100:.2f}% vs actual {m['rate'] * 100:.2f}%. "
                f"Gate {'met' if r['gate']['pass'] else 'not met'}")
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
    p = sub.add_parser("catalysttest", help="declare or run S3a, the earnings-aware shock radar's test (docs/SHOCKS.md)")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.set_defaults(fn=_cmd)
