"""Declared forward tests of stocks that moved against their wave (brain/againstwave.py, docs/RIPPLES.md).

Declared on 2026-11-22, before any record. The record starts with the session of ``START`` and runs forward; each
record must be written before the close of the session after its date (``againstwave.record`` refuses later ones).
The outcome is each stock's own move (market removed, by its beta) over the five sessions after its date, in usual
days of five sessions. Readings are weekly: a calendar week's value is the mean over that week's recorded stocks
(one week is one reading: its stocks move together). A test is judged once, when it has ``MIN_WEEKS`` weekly
readings; until then ``test`` refuses.

- **R2a, bounce back:** stocks that moved against their wave **without** company news that session. Value per stock:
  ``-sign(own move) * outcome`` (positive when it bounced back).
- **R2b, keep going:** those **with** company news (an 8-K or results at the SEC that session). Value per stock:
  ``sign(own move) * outcome`` (positive when it kept going).
- **R2c, the wave matters:** every stock that moved against its wave, against its matched controls (same day, same
  direction, within half a usual day in size, same news status, no wave against them). Value per stock: its bounce
  back less its controls' mean bounce back.

Each passes when the mean of its weekly values is positive with a one-sided sign-flip p < 0.05 over the weeks, and
Benjamini-Hochberg q < 0.10 across every gated test in the ledger (``research/run.py``). There is no placebo: R2c is
the comparison that a placebo would give. Results are written whatever they say. Trading costs are not in the
outcome; the result reports the move in percent beside them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from brain import againstwave, config, connect, ripple, scan, shocks
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research.eventtest import boot_interval, flip_p
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

IDS = ("R2a", "R2b", "R2c")
GATE_TYPE = {"R2a": "bounce", "R2b": "continue", "R2c": "wave_matters"}
VERSION = 1
START = "2026-11-20"         # the first session recorded (its five sessions start Monday 2026-11-23)
MIN_WEEKS = 30
P_MAX = 0.05
N_FLIP, N_BOOT, SEED = 10000, 2000, 17
CLEAN = ("brain", "knowledge", "universe")
HERE = Path(__file__).resolve()
UNIVERSES = ("us_broad", "us_small", "us_micro", "us_watch")


def code_files() -> list[Path]:
    from brain import bank, series, timeutil
    from brain.research import eventtest, run, stats
    return [HERE, *(Path(m.__file__).resolve()
                    for m in (againstwave, ripple, connect, shocks, scan, bank, series, timeutil, eventtest, run, stats))]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def universe_shas() -> dict[str, str]:
    return {n: hashlib.sha256((config.ROOT / "universe" / f"{n}.json").read_bytes()).hexdigest()[:16] for n in UNIVERSES}


def spec(tid: str) -> dict:
    return {"id": tid, "version": VERSION, "start": START, "min_weeks": MIN_WEEKS, "h": againstwave.H,
            "absorb": ripple.ABSORB, "against": ripple.AGAINST, "match": againstwave.MATCH,
            "n_controls": againstwave.N_CONTROLS, "stone_z": ripple.STONE_Z, "edge": ripple.EDGE, "kappa": ripple.KAPPA,
            "universes": universe_shas(), "code_sha": code_sha(), "n_flip": N_FLIP, "n_boot": N_BOOT, "seed": SEED,
            "gate_type": GATE_TYPE[tid], "gate": {"mean_gt": 0.0, "p_lt": P_MAX, "bh_q_lt": 0.10}}


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
        if isinstance(rec, dict) and rec.get("op") == "declare" and rec.get("kind") == "ripple" \
                and rec.get("id") == tid and rec.get("spec_sha") == sha:
            return True
    return False


def _declared(recs: list[dict], tid: str) -> dict | None:
    return next((r for r in recs if r["op"] == "declare" and r.get("kind") == "ripple" and r["id"] == tid), None)


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
        rec = {"op": "declare", "kind": "ripple", "id": tid, "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
               **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            log(f"declared {tid} ({rec['spec_sha']}): forward from {START}, judged at {MIN_WEEKS} weekly readings; "
                "commit the ledger before the first record")
    return out


def eligible(rows: list[dict]) -> list[dict]:
    """Outcomes from records dated from START and written before the close of the session after their date
    (taken as 20:00 UTC that day, the earlier of the summer and winter closes)."""
    return [r for r in rows if r["date"] >= START and r["recorded_at"] < f"{r['next_session']}T20:00:00Z"]


def values(rows: list[dict], tid: str) -> tuple[np.ndarray, np.ndarray]:
    """Each eligible stock's value for ``tid`` and its week."""
    v, w = [], []
    for r in rows:
        if not np.isfinite(r["y"]):
            continue
        if tid == "R2a" and not r["news"]:
            x = -r["sign"] * r["y"]
        elif tid == "R2b" and r["news"]:
            x = r["sign"] * r["y"]
        elif tid == "R2c" and np.isfinite(r["y_controls"]):
            x = (-r["sign"] * r["y"]) - (-r["sign"] * r["y_controls"])
        else:
            continue
        v.append(x)
        w.append(str(np.datetime64(r["date"], "W")))
    return np.array(v), np.array(w)


def weekly(v: np.ndarray, w: np.ndarray) -> tuple[list[str], np.ndarray]:
    ws = sorted(set(w.tolist()))
    return ws, np.array([v[w == x].mean() for x in ws])


def evaluate(rows: list[dict], tid: str) -> dict:
    rng = np.random.default_rng(SEED)
    v, w = values(rows, tid)
    weeks, d = weekly(v, w)
    p = flip_p(d, 1, N_FLIP, rng) if len(d) else 1.0
    metrics = {"stocks": int(len(v)), "weeks": len(weeks), "first": weeks[0] if weeks else None,
               "last": weeks[-1] if weeks else None, "mean": float(d.mean()) if len(d) else float("nan"),
               "interval90": boot_interval(d, N_BOOT, rng) if len(d) else None,
               "weeks_positive": int((d > 0).sum()), "p": p, "stock_mean": float(v.mean()) if len(v) else float("nan")}
    gate = {"mean_positive": metrics["mean"] > 0, "p_ok": p < P_MAX, "weeks_ok": len(weeks) >= MIN_WEEKS}
    gate["pass"] = bool(all(gate.values()))
    return {"metrics": metrics, "gate": gate, "p": p,
            "controls": {"pass": True, "note": "no placebo: R2c is the matched comparison"}}


def test(bank: Bank, *, trial: bool = False, log=print, path=None) -> list[dict]:
    from brain.research import run
    trial = is_trial(trial)
    led, res = paths(trial)
    recs = ledger.read(led)
    rows = eligible(againstwave.outcomes(bank, againstwave.read(path)))
    out = []
    for tid in IDS:
        decl = _declared(recs, tid)
        if decl is None:
            raise Refused(f"{tid} is not declared")
        if any(r["op"] == "result" and r.get("kind") == "ripple" and r["id"] == tid for r in recs):
            continue
        s = decl["spec"]
        if spec_sha(s) != decl["spec_sha"]:
            raise Refused(f"{tid}: its declared spec does not match its spec_sha")
        if not trial:
            if dirty := tree_dirty():
                raise Refused("the working tree has changes under brain/, knowledge/ or universe/: " + "; ".join(dirty))
            if why := lineage_problem(decl.get("commit")):
                raise Refused(f"{tid}: {why}")
            if s["code_sha"] != code_sha():
                raise Refused(f"{tid}: the code is not the declared code")
            if not committed_declaration(tid, decl["spec_sha"]):
                raise Refused(f"{tid}: commit the ledger with its declaration first")
        _, w = values(rows, tid)
        n_weeks = len(set(w.tolist()))
        if n_weeks < MIN_WEEKS:
            raise Refused(f"{tid}: {n_weeks} weekly readings so far; it is judged once, at {MIN_WEEKS}")
        r = evaluate(rows, tid)
        rec = {"op": "result", "kind": "ripple", "id": tid, "spec_sha": decl["spec_sha"], "gate_type": GATE_TYPE[tid],
               **r, "at": now_iso(), **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            m = r["metrics"]
            log(f"{tid}: {m['stocks']} stocks in {m['weeks']} weeks; weekly mean {m['mean']:+.4f} (p {m['p']:.4f}). "
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
    p = sub.add_parser("againsttest", help="declare or judge R2a-R2c: stocks that moved against their wave (forward)")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.set_defaults(fn=_cmd)
