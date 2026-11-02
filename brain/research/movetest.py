"""Declared test of the earnings move-size model M1 (docs/MOVES.md).

M1a asks whether M1's range for a release's move (``brain/moves.py``) beats the history-only baseline on releases
neither has seen. Both were fitted on the research universe's releases of 2011-2023 and saved in
``knowledge/moves_m1.json``, pinned here by its sha. TEST is every research-universe release whose first session
falls in 2024-01-01..2026-09-30, read once, by the recorded run.

Per release the score is the mean pinball loss over the quantiles 0.1, 0.25, 0.5, 0.75 and 0.9 (lower is better);
per month, the baseline's mean loss minus M1's (calendar time: a busy month counts once). The statistic is the mean
of those monthly differences, with a one-sided sign-flip p. M1a passes when the difference is positive, p < 0.05,
M1's 80% range (0.1 to 0.9) holds between 75% and 85% of TEST releases, the controls pass, and Benjamini-Hochberg
q < 0.10 across every gated test in the ledger (``research/run.py``). Controls: M1's ranges shuffled among the
month's releases must reach p < 0.05 in at most 10% of draws; the actual move as the middle must reach p < 0.01.
The result is written whatever it says.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from brain import config, moves
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research.eventtest import boot_interval, flip_p
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

ID = "M1a"
VERSION = 1
TEST = ("2024-01-01", "2026-09-30")
COVER80 = (0.75, 0.85)
P_MAX = 0.05
PLACEBO_FPR_MAX = 0.10
ORACLE_P_MAX = 0.01
N_FLIP, N_BOOT, N_PLACEBO, SEED = 10000, 2000, 200, 7
CLEAN = ("brain", "knowledge", "universe")
HERE = Path(__file__).resolve()


def code_files() -> list[Path]:
    """What the numbers run through: this file, the model, the release windows, the bank's view, the price book,
    the series store, the time rules, the sign-flip helpers, and the verdicts with their correction."""
    from brain import bank, events, metrics, series, timeutil
    from brain.research import eventtest, run, stats
    return [HERE, *(Path(m.__file__).resolve()
                    for m in (moves, events, bank, metrics, series, timeutil, eventtest, run, stats))]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def universe_shas() -> dict[str, str]:
    return {n: hashlib.sha256((config.ROOT / "universe" / f"{n}.json").read_bytes()).hexdigest()[:16]
            for n in moves.UNIVERSES}


def spec() -> dict:
    model = moves.load()
    if model is None:
        raise Refused("no fitted model: run python -m brain moves fit, and commit knowledge/moves_m1.json")
    return {"id": ID, "version": VERSION, "model": "knowledge/moves_m1.json", "model_sha": moves.model_sha(),
            "train": model["train"], "test": list(TEST), "qs": list(moves.QS), "feats": list(moves.FEATS),
            "baseline": list(moves.BASELINE), "universes": universe_shas(), "code_sha": code_sha(),
            "gate": {"diff_gt": 0.0, "p_lt": P_MAX, "cover80": list(COVER80), "bh_q_lt": 0.10},
            "controls": {"placebo_fpr_max": PLACEBO_FPR_MAX, "oracle_p_max": ORACLE_P_MAX},
            "n_flip": N_FLIP, "n_boot": N_BOOT, "n_placebo": N_PLACEBO, "seed": SEED}


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
    files = [p.relative_to(root).as_posix() for p in code_files()] + ["knowledge/moves_m1.json"]
    changed = [x for x in _git("diff", "--name-only", commit, "HEAD", "--", *files).stdout.splitlines() if x.strip()]
    return f"code or model changed since the declare commit {commit[:12]}: {', '.join(changed)}" if changed else None


def committed_declaration(sha: str) -> bool:
    r = _git("show", "HEAD:research/ledger.jsonl")
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("op") == "declare" and rec.get("kind") == "move" \
                and rec.get("id") == ID and rec.get("spec_sha") == sha:
            return True
    return False


def _records(led) -> list[dict]:
    return ledger.read(led)


def declare(*, trial: bool = False, log=print) -> dict | None:
    trial = is_trial(trial)
    led, _ = paths(trial)
    if not trial and (dirty := tree_dirty()):
        raise Refused("commit brain/, knowledge/ and universe/ first; a declaration pins the committed code and "
                      "model: " + "; ".join(dirty))
    s = spec()
    have = [r for r in _records(led) if r["op"] == "declare" and r.get("kind") == "move" and r["id"] == ID]
    if have:
        if have[0]["spec_sha"] != spec_sha(s):
            raise Refused(f"{ID} was declared with a different spec; a change is a new id")
        if log:
            log(f"{ID} is already declared ({have[0]['spec_sha']})")
        return None
    rec = {"op": "declare", "kind": "move", "id": ID, "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
           **ledger.git_state()}
    ledger.append(rec, led)
    if log:
        log(f"declared {ID} ({rec['spec_sha']}): TEST {TEST[0]}..{TEST[1]}; commit the ledger before running it")
    return rec


def score(rows: list[dict], model: dict) -> dict:
    """Per-release losses and coverage for M1 and the baseline, and the months."""
    m1, base = model["m1"], model["baseline"]
    out = {"month": [], "loss_m1": [], "loss_base": [], "in80": [], "in50": [], "base_in80": [], "y": [],
           "q_m1": [], "above_m1": [], "above_base": [], "above": []}
    for r in rows:
        q, qb, y = moves.quantiles(m1, r), moves.quantiles(base, r), r["move"]
        out["month"].append(r["date"][:7])
        out["loss_m1"].append(moves.pinball(q, y))
        out["loss_base"].append(moves.pinball(qb, y))
        out["in80"].append(q[0.1] <= y <= q[0.9])
        out["in50"].append(q[0.25] <= y <= q[0.75])
        out["base_in80"].append(qb[0.1] <= y <= qb[0.9])
        out["y"].append(y)
        out["q_m1"].append([q[k] for k in moves.QS])
        out["above_m1"].append(moves.chance_above(m1, r, r["med8"]))
        out["above_base"].append(moves.chance_above(base, r, r["med8"]))
        out["above"].append(y > r["med8"])
    return {k: np.array(v) for k, v in out.items()}


def monthly(month: np.ndarray, d: np.ndarray) -> tuple[list[str], np.ndarray]:
    ms = sorted(set(month.tolist()))
    return ms, np.array([d[month == m].mean() for m in ms])


def _pin(qs: np.ndarray, y: np.ndarray) -> np.ndarray:
    qv = np.array(moves.QS)
    diff = y[:, None] - qs
    return np.mean(np.maximum(qv * diff, (qv - 1) * diff), axis=1)


def controls(sc: dict, rng) -> dict:
    """Placebo: M1's ranges shuffled among the month's releases. Oracle: the actual move as the middle."""
    month, y, qs = sc["month"], sc["y"], sc["q_m1"]
    hits = 0
    for _ in range(N_PLACEBO):
        perm = np.arange(len(y))
        for m in set(month.tolist()):
            idx = np.flatnonzero(month == m)
            perm[idx] = rng.permutation(idx)
        _, d = monthly(month, sc["loss_base"] - _pin(qs[perm], y))
        hits += flip_p(d, 1, 999, rng) < P_MAX
    scale = np.where(qs[:, 2:3] > 0, y[:, None] / np.where(qs[:, 2:3] > 0, qs[:, 2:3], 1.0), 1.0)
    oracle = qs * scale
    _, d = monthly(month, sc["loss_base"] - _pin(oracle, y))
    oracle_p = flip_p(d, 1, N_FLIP, rng)
    fpr = hits / N_PLACEBO
    return {"n_placebo": N_PLACEBO, "placebo_fpr": fpr, "oracle_p": oracle_p,
            "pass": bool(fpr <= PLACEBO_FPR_MAX and oracle_p <= ORACLE_P_MAX)}


def evaluate(rows: list[dict], model: dict) -> dict:
    rng = np.random.default_rng(SEED)
    sc = score(rows, model)
    months, d = monthly(sc["month"], sc["loss_base"] - sc["loss_m1"])
    p = flip_p(d, 1, N_FLIP, rng)
    cover80 = float(sc["in80"].mean()) if len(rows) else float("nan")
    brier = lambda pr: float(np.mean((pr - sc["above"]) ** 2)) if len(rows) else float("nan")
    metrics = {"releases": len(rows), "months": len(months), "first": months[0] if months else None,
               "last": months[-1] if months else None,
               "loss_m1": float(sc["loss_m1"].mean()), "loss_base": float(sc["loss_base"].mean()),
               "diff": float(d.mean()) if len(d) else float("nan"), "interval90": boot_interval(d, N_BOOT, rng),
               "months_better": int((d > 0).sum()), "p": p, "cover80": cover80, "cover50": float(sc["in50"].mean()),
               "base_cover80": float(sc["base_in80"].mean()), "median_move": float(np.median(sc["y"])),
               "above_usual_rate": float(sc["above"].mean()), "brier_m1": brier(sc["above_m1"]),
               "brier_base": brier(sc["above_base"]), "brier_rate": brier(np.full(len(rows), sc["above"].mean()))}
    gate = {"diff_positive": metrics["diff"] > 0, "p_ok": p < P_MAX,
            "cover_ok": COVER80[0] <= cover80 <= COVER80[1]}
    gate["pass"] = bool(all(gate.values()))
    return {"metrics": metrics, "gate": gate, "controls": controls(sc, rng), "p": p}


def test(bank: Bank, *, trial: bool = False, log=print, tickers=None) -> dict | None:
    from brain.research import run
    trial = is_trial(trial)
    led, res = paths(trial)
    recs = _records(led)
    decl = next((r for r in recs if r["op"] == "declare" and r.get("kind") == "move" and r["id"] == ID), None)
    if decl is None:
        raise Refused(f"{ID} is not declared")
    if any(r["op"] == "result" and r.get("kind") == "move" and r["id"] == ID for r in recs):
        raise Refused(f"{ID} already has a result; a rerun needs research withdraw and a new id")
    s = decl["spec"]
    if spec_sha(s) != decl["spec_sha"]:
        raise Refused(f"{ID}: its declared spec does not match its spec_sha")
    if s["model_sha"] != moves.model_sha():
        raise Refused(f"{ID}: knowledge/moves_m1.json is not the declared model")
    if not trial:
        if dirty := tree_dirty():
            raise Refused("the working tree has changes under brain/, knowledge/ or universe/: " + "; ".join(dirty))
        if why := lineage_problem(decl.get("commit")):
            raise Refused(f"{ID}: {why}")
        if s["code_sha"] != code_sha():
            raise Refused(f"{ID}: the code is not the declared code")
        if s["universes"] != universe_shas():
            raise Refused(f"{ID}: a universe file changed since the declaration; declare a new id")
        if not committed_declaration(decl["spec_sha"]):
            raise Refused(f"{ID}: commit the ledger with its declaration before running it")
    model = moves.load()
    rows = moves.dataset(bank, tickers or moves.universe_tickers(), s["test"][0], s["test"][1], log=log)
    r = evaluate(rows, model)
    rec = {"op": "result", "kind": "move", "id": ID, "spec_sha": decl["spec_sha"], "gate_type": "pinball",
           **r, "at": now_iso(), **ledger.git_state()}
    ledger.append(rec, led)
    run.report(led, res)
    if log:
        m = r["metrics"]
        log(f"{ID}: {m['releases']} releases in {m['months']} months; pinball loss M1 {m['loss_m1'] * 100:.3f} vs "
            f"baseline {m['loss_base'] * 100:.3f} points (monthly difference {m['diff'] * 100:+.3f}, p {m['p']:.4f}); "
            f"80% range held {m['cover80'] * 100:.1f}%. Gate {'met' if r['gate']['pass'] else 'not met'}; wrote {res}"
            + ("" if trial else "; commit the ledger and RESULTS.md, whatever they say"))
    return rec


def _cmd(a) -> int:
    if a.action == "declare":
        declare(trial=a.trial)
    else:
        test(Bank(), trial=a.trial)
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("movetest", help="declare or run M1a, the earnings move-size test (docs/MOVES.md)")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.set_defaults(fn=_cmd)
