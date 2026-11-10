"""Declared tests of the scan's weekly range, W1 (docs/SCAN.md).

**W1a** asks whether W1's range for a stock's next five sessions (``brain/scan.py``) is honest and beats the range
its 60-session volatility alone gives, on stock-weeks neither has seen. Both were fitted on the research universe's
tradable stock-weeks of 2011-2023 and saved in ``knowledge/scan_w1.json``, pinned here by its sha. TEST is every
tradable research-universe stock-week ending in 2024-01-01..2026-09-30 whose five sessions have closed.

**W1b** asks the same, with the same model, gate and TEST years, of the scan's newer listings
(``universe/us_watch.json``: companies of SEC ranks 1-3000 that listed after 2010, so no research list holds them).
The model was fitted on none of them.

Per stock-week the score is the mean pinball loss of the five-session log return over the quantiles 0.05, 0.16,
0.5, 0.84 and 0.95 (lower is better: it rewards a range that is both right and tight); per month, the baseline's
mean loss minus W1's (calendar time: a busy month counts once). The statistic is the mean of those monthly
differences, with a one-sided sign-flip p. W1a passes when:

- the difference is positive with p < 0.05;
- W1's 68% range (0.16 to 0.84) holds between 63% and 73% of TEST stock-weeks, and its 90% range (0.05 to 0.95)
  between 86% and 94%;
- the controls pass: W1's ranges shuffled among each week's stocks reach p < 0.05 in at most 10% of draws, and
  a range that knows the outcome reaches p < 0.01;
- Benjamini-Hochberg q < 0.10 across every gated test in the ledger (``research/run.py``).

The result is written whatever it says. Nothing here tests a direction: the middle of the range is reported, not
gated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from brain import config, scan, shocks
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research.eventtest import boot_interval, flip_p
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

IDS = {"W1a": shocks.UNIVERSES, "W1b": ("us_watch",)}     # each test and the lists it reads
KIND = "range"
VERSION = 1
TEST = ("2024-01-01", "2026-09-30")
COVER68 = (0.63, 0.73)
COVER90 = (0.86, 0.94)
P_MAX = 0.05
PLACEBO_FPR_MAX = 0.10
ORACLE_P_MAX = 0.01
N_FLIP, N_BOOT, N_PLACEBO, SEED = 10000, 2000, 200, 13
CLEAN = ("brain", "knowledge", "universe")
MODEL_FILE = "knowledge/scan_w1.json"
HERE = Path(__file__).resolve()


def code_files() -> list[Path]:
    """What the numbers run through: this file, the range model, S1's features, the release-window rule, the bank's
    view, the series store, the time rules, the sign-flip helpers, and the verdicts with their correction."""
    from brain import bank, events, series, timeutil
    from brain.research import eventtest, run, stats
    return [HERE, *(Path(m.__file__).resolve()
                    for m in (scan, shocks, events, bank, series, timeutil, eventtest, run, stats))]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def universe_shas(names=shocks.UNIVERSES) -> dict[str, str]:
    return {n: hashlib.sha256((config.ROOT / "universe" / f"{n}.json").read_bytes()).hexdigest()[:16] for n in names}


def tickers_of(tid: str) -> list[str]:
    out: list[str] = []
    for n in IDS[tid]:
        for t in json.loads((config.ROOT / "universe" / f"{n}.json").read_text())["tickers"]:
            if t not in out:
                out.append(t)
    return out


def spec(tid: str = "W1a") -> dict:
    model = scan.load()
    if model is None:
        raise Refused(f"no fitted model: run python -m brain scan fit, and commit {MODEL_FILE}")
    return {"id": tid, "version": VERSION, "test": list(TEST), "universes": universe_shas(IDS[tid]),
            "code_sha": code_sha(),
            "model": MODEL_FILE, "model_sha": scan.model_sha(), "train": model["train"], "h": scan.H,
            "feats": list(scan.FEATS), "baseline": list(scan.BASELINE), "split": scan.SPLIT, "qs": list(scan.QS),
            "tradable": {"days": scan.TRADE_DAYS, "min_dollar_volume": scan.MIN_DV, "error_days": scan.ERROR_DAYS},
            "n_flip": N_FLIP, "n_boot": N_BOOT, "n_placebo": N_PLACEBO, "seed": SEED,
            "gate": {"diff_gt": 0.0, "p_lt": P_MAX, "cover68": list(COVER68), "cover90": list(COVER90),
                     "bh_q_lt": 0.10},
            "controls": {"placebo_fpr_max": PLACEBO_FPR_MAX, "oracle_p_max": ORACLE_P_MAX}}


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
    files = [p.relative_to(root).as_posix() for p in code_files()] + [MODEL_FILE]
    changed = [x for x in _git("diff", "--name-only", commit, "HEAD", "--", *files).stdout.splitlines() if x.strip()]
    return f"code or model changed since the declare commit {commit[:12]}: {', '.join(changed)}" if changed else None


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


def declare(*, trial: bool = False, log=print, ids=None) -> list[dict]:
    """Declare each test of ``ids`` (default all) not declared yet. A declared one is left as it is: its spec, as
    recorded, is what its run is held to, and a change is a new id."""
    trial = is_trial(trial)
    led, _ = paths(trial)
    if not trial and (dirty := tree_dirty()):
        raise Refused("commit brain/, knowledge/ and universe/ first; a declaration pins the committed code and "
                      "model: " + "; ".join(dirty))
    out = []
    for tid in ids or IDS:
        have = _declared(ledger.read(led), tid)
        if have:
            if log:
                log(f"{tid} is already declared ({have['spec_sha']})")
            continue
        s = spec(tid)
        rec = {"op": "declare", "kind": KIND, "id": tid, "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
               **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            log(f"declared {tid} ({rec['spec_sha']}): TEST {TEST[0]}..{TEST[1]} on {', '.join(IDS[tid])}; commit the "
                "ledger before running it")
    return out


# ---------- scoring ----------
def pinball(q: np.ndarray, y: np.ndarray, qs=scan.QS) -> np.ndarray:
    """Each row's mean pinball loss over the quantiles ``qs`` (``q`` has one column per quantile)."""
    d = y[:, None] - q
    a = np.array(qs)[None, :]
    return np.maximum(a * d, (a - 1) * d).mean(1)


def monthly(month: np.ndarray, d: np.ndarray) -> tuple[list[str], np.ndarray]:
    ms = sorted(set(month.tolist()))
    return ms, np.array([d[month == m].mean() for m in ms])


def _rank_corr(week: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """The mean over weeks of the rank correlation between ``a`` and ``b`` across that week's stocks."""
    out = []
    for w in np.unique(week):
        i = np.flatnonzero(week == w)
        if len(i) < 10:
            continue
        ra, rb = np.argsort(np.argsort(a[i])), np.argsort(np.argsort(b[i]))
        out.append(float(np.corrcoef(ra, rb)[0, 1]))
    return float(np.mean(out)) if out else float("nan")


def score(d: dict, model: dict) -> dict:
    y = np.asarray(d["l5"], float)
    q1, qb = scan.quantiles(model["w1"], d["X"]), scan.quantiles(model["baseline"], d["X"])
    s1, b1 = scan.scale(model["w1"], d["X"])
    sb, _ = scan.scale(model["baseline"], d["X"])
    week = np.array(d["date"])
    return {"y": y, "q1": q1, "qb": qb, "s1": s1, "sb": sb, "bucket": b1, "week": week,
            "month": np.array([x[:7] for x in d["date"]]), "loss_w1": pinball(q1, y), "loss_base": pinball(qb, y),
            "in68": (y >= q1[:, 1]) & (y <= q1[:, 3]), "in90": (y >= q1[:, 0]) & (y <= q1[:, 4]),
            "base68": (y >= qb[:, 1]) & (y <= qb[:, 3]), "base90": (y >= qb[:, 0]) & (y <= qb[:, 4]),
            "p_up": np.array(model["w1"]["up"])[b1]}


def controls(sc: dict, rng) -> dict:
    """Placebo: W1's ranges shuffled among each week's stocks. Oracle: a range that sits on the outcome."""
    week, y, q1 = sc["week"], sc["y"], sc["q1"]
    groups = [np.flatnonzero(week == w) for w in np.unique(week)]
    hits = 0
    for _ in range(N_PLACEBO):
        perm = np.arange(len(y))
        for idx in groups:
            perm[idx] = rng.permutation(idx)
        _, d = monthly(sc["month"], sc["loss_base"] - pinball(q1[perm], y))
        hits += flip_p(d, 1, 999, rng) < P_MAX
    _, d = monthly(sc["month"], sc["loss_base"] - pinball(np.repeat(y[:, None], q1.shape[1], axis=1), y))
    oracle_p = flip_p(d, 1, N_FLIP, rng)
    fpr = hits / N_PLACEBO
    return {"n_placebo": N_PLACEBO, "placebo_fpr": fpr, "oracle_p": oracle_p,
            "pass": bool(fpr <= PLACEBO_FPR_MAX and oracle_p <= ORACLE_P_MAX)}


def _bands() -> dict[str, str]:
    """Each ticker's list; for the newer listings, their SEC rank band."""
    out: dict[str, str] = {}
    for n in (*shocks.UNIVERSES, "us_watch"):
        p = config.ROOT / "universe" / f"{n}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for t in d["tickers"]:
            out.setdefault(t, f"{n} {d['bands'][t]}" if n == "us_watch" and "bands" in d else n)
    return out


def _part(sc: dict, m: np.ndarray) -> dict:
    return {"n": int(m.sum()), "cover68": float(sc["in68"][m].mean()) if m.any() else float("nan"),
            "cover90": float(sc["in90"][m].mean()) if m.any() else float("nan")}


def evaluate(d: dict, model: dict, bands: dict[str, str] | None = None) -> dict:
    rng = np.random.default_rng(SEED)
    sc = score(d, model)
    y, q1 = sc["y"], sc["q1"]
    months, diff = monthly(sc["month"], sc["loss_base"] - sc["loss_w1"])
    p = flip_p(diff, 1, N_FLIP, rng)
    n = len(y)

    def f(x) -> float:
        return float(np.mean(x)) if n else float("nan")

    bands = _bands() if bands is None else bands
    band = np.array([bands.get(t, "?") for t in d["t"]])
    earn = np.asarray(d["X"])[:, scan.FEATS.index("earn_win")] > 0
    metrics = {"stock_weeks": n, "stocks": len(set(d["t"])), "months": len(months),
               "first": months[0] if months else None, "last": months[-1] if months else None,
               "loss_w1": f(sc["loss_w1"]), "loss_base": f(sc["loss_base"]),
               "diff": float(diff.mean()) if len(diff) else float("nan"), "interval90": boot_interval(diff, N_BOOT, rng),
               "months_better": int((diff > 0).sum()), "p": p,
               "cover68": f(sc["in68"]), "cover90": f(sc["in90"]),
               "base_cover68": f(sc["base68"]), "base_cover90": f(sc["base90"]),
               "below05": f(y < q1[:, 0]), "above95": f(y > q1[:, 4]),
               "below16": f(y < q1[:, 1]), "above84": f(y > q1[:, 3]),
               "median_abs": float(np.median(np.abs(np.expm1(y)))) if n else float("nan"),
               "median_half68": float(np.median((np.expm1(q1[:, 3]) - np.expm1(q1[:, 1])) / 2)) if n else float("nan"),
               "rank_w1": _rank_corr(sc["week"], sc["s1"], np.abs(y)),
               "rank_base": _rank_corr(sc["week"], sc["sb"], np.abs(y)),
               "up_rate": f(y > 0), "up_said": f(sc["p_up"]), "above_middle": f(y > q1[:, 2]),
               "by_band": {b: _part(sc, band == b) for b in sorted(set(band.tolist()))},
               "report_due": _part(sc, earn), "no_report": _part(sc, ~earn),
               "by_tenth": [_part(sc, sc["bucket"] % scan.BUCKETS == k) for k in range(scan.BUCKETS)]}
    gate = {"diff_positive": metrics["diff"] > 0, "p_ok": p < P_MAX,
            "cover68_ok": COVER68[0] <= metrics["cover68"] <= COVER68[1],
            "cover90_ok": COVER90[0] <= metrics["cover90"] <= COVER90[1]}
    gate["pass"] = bool(all(gate.values()))
    return {"metrics": metrics, "gate": gate, "controls": controls(sc, rng), "p": p}


# ---------- running ----------
def test(bank: Bank, *, trial: bool = False, log=print, tickers=None, ids=None) -> list[dict]:
    """Run each declared test of ``ids`` (default all) that has no result yet, once. ``tickers`` replaces every
    test's lists (trials)."""
    from brain.research import run
    trial = is_trial(trial)
    led, res = paths(trial)
    recs = ledger.read(led)
    todo = []
    for tid in ids or IDS:
        decl = _declared(recs, tid)
        if decl is None:
            if ids:
                raise Refused(f"{tid} is not declared")
            continue
        if any(r["op"] == "result" and r.get("kind") == KIND and r["id"] == tid for r in recs):
            continue
        s = decl["spec"]
        if spec_sha(s) != decl["spec_sha"]:
            raise Refused(f"{tid}: its declared spec does not match its spec_sha")
        if s["model_sha"] != scan.model_sha():
            raise Refused(f"{tid}: {MODEL_FILE} is not the declared model")
        if not trial:
            if dirty := tree_dirty():
                raise Refused("the working tree has changes under brain/, knowledge/ or universe/: " + "; ".join(dirty))
            if why := lineage_problem(decl.get("commit")):
                raise Refused(f"{tid}: {why}")
            if s["code_sha"] != code_sha():
                raise Refused(f"{tid}: the code is not the declared code")
            if s["universes"] != universe_shas(IDS[tid]):
                raise Refused(f"{tid}: a universe file changed since the declaration; declare a new id")
            if not committed_declaration(tid, decl["spec_sha"]):
                raise Refused(f"{tid}: commit the ledger with its declaration before running it")
        todo.append(decl)
    if not todo:
        raise Refused("every declared weekly-range test already has a result; a rerun needs research withdraw and a "
                      "new id")
    out = []
    for decl in todo:
        tid = decl["id"]
        d = scan.dataset(bank, tickers or tickers_of(tid), TEST[0], TEST[1], log=log)
        r = evaluate(d, scan.load())
        rec = {"op": "result", "kind": KIND, "id": tid, "spec_sha": decl["spec_sha"], "gate_type": "pinball", **r,
               "at": now_iso(), **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            m = r["metrics"]
            log(f"{tid}: {m['stock_weeks']} stock-weeks in {m['months']} months; pinball loss W1 "
                f"{m['loss_w1'] * 100:.3f} vs baseline {m['loss_base'] * 100:.3f} (monthly difference "
                f"{m['diff'] * 100:+.3f}, p {m['p']:.4f}); 68% range held {m['cover68'] * 100:.1f}%, 90% range "
                f"{m['cover90'] * 100:.1f}%. Gate {'met' if r['gate']['pass'] else 'not met'}")
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
    p = sub.add_parser("rangetest", help="declare or run W1a and W1b, the tests of the scan's weekly range (docs/SCAN.md)")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.set_defaults(fn=_cmd)
