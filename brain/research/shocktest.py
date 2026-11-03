"""Declared test of the shock radar S1 (docs/SHOCKS.md).

S1a asks whether S1's chance of a 20%+ session against SPY in the next five (``brain/shocks.py``) beats the
volatility-history baseline on stock-weeks neither has seen. Both were fitted on the research universe's stock-weeks of
2011-2023 and saved in ``knowledge/shocks_s1.json``, pinned here by its sha. TEST is every research-universe stock-week
ending in 2024-01-01..2026-09-30 whose five sessions have closed, read once, by the recorded run.

Per stock-week the score is the log loss of the chance (lower is better); per month, the baseline's mean loss minus
S1's (calendar time: a busy month counts once). The statistic is the mean of those monthly differences, with a
one-sided sign-flip p. S1a passes when the difference is positive with p < 0.05; S1's mean chance over TEST is within
CAL of the actual rate; the actual rate among each week's top twentieth by S1 is within CAL of S1's mean chance there;
the controls pass; and Benjamini-Hochberg q < 0.10 across every gated test in the ledger (``research/run.py``).
Controls: S1's chances shuffled among each week's stocks must reach p < 0.05 in at most 10% of draws; an oracle that
knows the outcome must reach p < 0.01. The result is written whatever it says.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from brain import config, shocks
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research.eventtest import boot_interval, flip_p
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

ID = "S1a"
VERSION = 1
TEST = ("2024-01-01", "2026-09-30")
P_MAX = 0.05
CAL = (0.67, 1.5)          # actual over predicted, overall and in the top twentieth
TOP = 0.05
PLACEBO_FPR_MAX = 0.10
ORACLE_P_MAX = 0.01
N_FLIP, N_BOOT, N_PLACEBO, SEED = 10000, 2000, 200, 11
CLEAN = ("brain", "knowledge", "universe")
HERE = Path(__file__).resolve()
TINY = 1e-6


def code_files() -> list[Path]:
    """What the numbers run through: this file, the model, the bank's view, the series store, the release-window
    rule, the time rules, the sign-flip helpers, and the verdicts with their correction."""
    from brain import bank, events, series, timeutil
    from brain.research import eventtest, run, stats
    return [HERE, *(Path(m.__file__).resolve()
                    for m in (shocks, events, bank, series, timeutil, eventtest, run, stats))]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def universe_shas() -> dict[str, str]:
    return {n: hashlib.sha256((config.ROOT / "universe" / f"{n}.json").read_bytes()).hexdigest()[:16]
            for n in shocks.UNIVERSES}


def spec() -> dict:
    model = shocks.load()
    if model is None:
        raise Refused("no fitted model: run python -m brain shocks fit, and commit knowledge/shocks_s1.json")
    return {"id": ID, "version": VERSION, "model": "knowledge/shocks_s1.json", "model_sha": shocks.model_sha(),
            "train": model["train"], "test": list(TEST), "h": shocks.H, "shock": shocks.SHOCK,
            "feats": list(shocks.FEATS), "baseline": list(shocks.BASELINE), "universes": universe_shas(),
            "code_sha": code_sha(),
            "gate": {"diff_gt": 0.0, "p_lt": P_MAX, "cal": list(CAL), "top": TOP, "bh_q_lt": 0.10},
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
    files = [p.relative_to(root).as_posix() for p in code_files()] + ["knowledge/shocks_s1.json"]
    changed = [x for x in _git("diff", "--name-only", commit, "HEAD", "--", *files).stdout.splitlines() if x.strip()]
    return f"code or model changed since the declare commit {commit[:12]}: {', '.join(changed)}" if changed else None


def committed_declaration(sha: str) -> bool:
    r = _git("show", "HEAD:research/ledger.jsonl")
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("op") == "declare" and rec.get("kind") == "shock" \
                and rec.get("id") == ID and rec.get("spec_sha") == sha:
            return True
    return False


def declare(*, trial: bool = False, log=print) -> dict | None:
    trial = is_trial(trial)
    led, _ = paths(trial)
    if not trial and (dirty := tree_dirty()):
        raise Refused("commit brain/, knowledge/ and universe/ first; a declaration pins the committed code and "
                      "model: " + "; ".join(dirty))
    s = spec()
    have = [r for r in ledger.read(led) if r["op"] == "declare" and r.get("kind") == "shock" and r["id"] == ID]
    if have:
        if have[0]["spec_sha"] != spec_sha(s):
            raise Refused(f"{ID} was declared with a different spec; a change is a new id")
        if log:
            log(f"{ID} is already declared ({have[0]['spec_sha']})")
        return None
    rec = {"op": "declare", "kind": "shock", "id": ID, "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
           **ledger.git_state()}
    ledger.append(rec, led)
    if log:
        log(f"declared {ID} ({rec['spec_sha']}): TEST {TEST[0]}..{TEST[1]}; commit the ledger before running it")
    return rec


# ---------- scoring ----------
def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, TINY, 1 - TINY)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc(s: np.ndarray, y: np.ndarray) -> float:
    n1 = float(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def week_rank(week: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Each row's rank within its week, as a fraction (the highest chance is near 0)."""
    out = np.empty(len(p))
    for w in np.unique(week):
        idx = np.flatnonzero(week == w)
        o = np.argsort(-p[idx], kind="mergesort")
        rk = np.empty(len(idx))
        rk[o] = (np.arange(len(idx)) + 1) / len(idx)
        out[idx] = rk
    return out


def score(rows: list[dict], model: dict) -> dict:
    y = np.array([r["y"] for r in rows], float)
    p1, pb = shocks.chances(model["s1"], rows), shocks.chances(model["baseline"], rows)
    week = np.array([r["date"] for r in rows])
    return {"y": y, "up": np.array([r["up"] for r in rows], float), "p1": p1, "pb": pb, "week": week,
            "month": np.array([d[:7] for d in week]), "loss_s1": logloss(p1, y), "loss_base": logloss(pb, y),
            "rank": week_rank(week, p1), "rank_base": week_rank(week, pb)}


def monthly(month: np.ndarray, d: np.ndarray) -> tuple[list[str], np.ndarray]:
    ms = sorted(set(month.tolist()))
    return ms, np.array([d[month == m].mean() for m in ms])


def capture(sc: dict, k: float, rank_key: str = "rank") -> tuple[float, float]:
    """The share of all shocks among each week's top ``k`` by chance, and the shock rate there."""
    top = sc[rank_key] <= k
    y = sc["y"]
    return (float(y[top].sum() / y.sum()) if y.sum() else float("nan"),
            float(y[top].mean()) if top.any() else float("nan"))


def controls(sc: dict, rng) -> dict:
    """Placebo: S1's chances shuffled among each week's stocks. Oracle: a chance that knows the outcome."""
    week, y, p1 = sc["week"], sc["y"], sc["p1"]
    groups = [np.flatnonzero(week == w) for w in np.unique(week)]
    hits = 0
    for _ in range(N_PLACEBO):
        perm = np.arange(len(y))
        for idx in groups:
            perm[idx] = rng.permutation(idx)
        _, d = monthly(sc["month"], sc["loss_base"] - logloss(p1[perm], y))
        hits += flip_p(d, 1, 999, rng) < P_MAX
    oracle = np.where(y > 0, 0.9, 0.001)
    _, d = monthly(sc["month"], sc["loss_base"] - logloss(oracle, y))
    oracle_p = flip_p(d, 1, N_FLIP, rng)
    fpr = hits / N_PLACEBO
    return {"n_placebo": N_PLACEBO, "placebo_fpr": fpr, "oracle_p": oracle_p,
            "pass": bool(fpr <= PLACEBO_FPR_MAX and oracle_p <= ORACLE_P_MAX)}


def evaluate(rows: list[dict], model: dict) -> dict:
    rng = np.random.default_rng(SEED)
    sc = score(rows, model)
    months, d = monthly(sc["month"], sc["loss_base"] - sc["loss_s1"])
    p = flip_p(d, 1, N_FLIP, rng)
    y = sc["y"]
    top = sc["rank"] <= TOP
    ratio = float(y.mean() / sc["p1"].mean()) if len(y) else float("nan")
    top_ratio = float(y[top].mean() / sc["p1"][top].mean()) if top.any() else float("nan")
    cap = {f"top{int(k * 100)}": capture(sc, k) for k in (0.01, 0.05, 0.10)}
    cap_base = {f"top{int(k * 100)}": capture(sc, k, "rank_base") for k in (0.01, 0.05, 0.10)}
    shocked = y > 0
    metrics = {"stock_weeks": len(rows), "stocks": len({r["t"] for r in rows}), "months": len(months),
               "first": months[0] if months else None, "last": months[-1] if months else None,
               "loss_s1": float(sc["loss_s1"].mean()), "loss_base": float(sc["loss_base"].mean()),
               "diff": float(d.mean()) if len(d) else float("nan"), "interval90": boot_interval(d, N_BOOT, rng),
               "months_better": int((d > 0).sum()), "p": p, "rate": float(y.mean()),
               "mean_s1": float(sc["p1"].mean()), "mean_base": float(sc["pb"].mean()), "cal_ratio": ratio,
               "top_rate": float(y[top].mean()) if top.any() else float("nan"),
               "top_chance": float(sc["p1"][top].mean()) if top.any() else float("nan"), "top_ratio": top_ratio,
               "auc_s1": auc(sc["p1"], y), "auc_base": auc(sc["pb"], y), "capture": cap, "capture_base": cap_base,
               "up_share": float(sc["up"][shocked].mean()) if shocked.any() else float("nan")}
    gate = {"diff_positive": metrics["diff"] > 0, "p_ok": p < P_MAX,
            "cal_ok": CAL[0] <= ratio <= CAL[1], "top_cal_ok": CAL[0] <= top_ratio <= CAL[1]}
    gate["pass"] = bool(all(gate.values()))
    return {"metrics": metrics, "gate": gate, "controls": controls(sc, rng), "p": p}


def test(bank: Bank, *, trial: bool = False, log=print, tickers=None) -> dict | None:
    from brain.research import run
    trial = is_trial(trial)
    led, res = paths(trial)
    recs = ledger.read(led)
    decl = next((r for r in recs if r["op"] == "declare" and r.get("kind") == "shock" and r["id"] == ID), None)
    if decl is None:
        raise Refused(f"{ID} is not declared")
    if any(r["op"] == "result" and r.get("kind") == "shock" and r["id"] == ID for r in recs):
        raise Refused(f"{ID} already has a result; a rerun needs research withdraw and a new id")
    s = decl["spec"]
    if spec_sha(s) != decl["spec_sha"]:
        raise Refused(f"{ID}: its declared spec does not match its spec_sha")
    if s["model_sha"] != shocks.model_sha():
        raise Refused(f"{ID}: knowledge/shocks_s1.json is not the declared model")
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
    model = shocks.load()
    rows = shocks.dataset(bank, tickers or shocks.universe_tickers(), s["test"][0], s["test"][1], log=log)
    r = evaluate(rows, model)
    rec = {"op": "result", "kind": "shock", "id": ID, "spec_sha": decl["spec_sha"], "gate_type": "logloss",
           **r, "at": now_iso(), **ledger.git_state()}
    ledger.append(rec, led)
    run.report(led, res)
    if log:
        m = r["metrics"]
        log(f"{ID}: {m['stock_weeks']} stock-weeks in {m['months']} months; log loss S1 {m['loss_s1']:.5f} vs "
            f"baseline {m['loss_base']:.5f} (monthly difference {m['diff']:+.5f}, p {m['p']:.4f}); chance "
            f"{m['mean_s1'] * 100:.2f}% vs actual {m['rate'] * 100:.2f}%; top 5% caught "
            f"{m['capture']['top5'][0] * 100:.0f}% of shocks. Gate {'met' if r['gate']['pass'] else 'not met'}; "
            f"wrote {res}" + ("" if trial else "; commit the ledger and RESULTS.md, whatever they say"))
    return rec


def _cmd(a) -> int:
    if a.action == "declare":
        declare(trial=a.trial)
    else:
        test(Bank(), trial=a.trial)
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("shocktest", help="declare or run S1a, the shock radar's test (docs/SHOCKS.md)")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.set_defaults(fn=_cmd)
