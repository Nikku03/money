"""Declared tests of the ripple picture (brain/ripple.py, docs/RIPPLES.md).

On the research universe's daily closes, every session from 2011 on is a day of stones (own moves of 3+ usual days,
market removed) and the waves they send through the pond of the year before. For each stock the waves reach that is not
itself a stone, the tests ask what follows over the next five sessions (``Y``: its own move, summed over the five
sessions, in usual days of five sessions). TRAIN is 2011-01-01..2023-12-31; TEST is 2024-01-01..2026-09-30. Per month
the statistic is averaged (calendar time: a busy month counts once), with a one-sided sign-flip p over the months.

**R1a** (reported, not gated) describes the waves on the day: how well the wave's height matched the stock's own move
that day, how high the stocks of each ring moved, and whether waves that cancelled moved stocks less.

**R1b, how high the ripples travel (size).** Does the wave a stock took today tell how far it moves over the next five
sessions, beyond what its own recent moves and the market's say? Two least-squares models of ``|Y|`` are fitted on
TRAIN: the baseline on the stock's own recent moves (5 and 60 sessions), its own move today and the market's recent
size; the ripple model adds the wave's height and energy. Per stock-day the loss is the squared error; per month, the
baseline's mean loss minus the ripple model's. Passes when that difference is positive with p < 0.05.

**R1c, which way the ripples travel (direction).** Does the wave's sign tell which way the stock moves next, as a wave
still arriving would? Per session, the rank correlation of the wave's height with ``Y`` among the stocks whose wave
energy is at least ``W_MIN`` (half a usual day). Passes when its monthly mean is positive with p < 0.05.

**R1d, interference.** Among stocks that took waves of energy ``W_HIGH`` (one usual day) or more, do waves that added up (interference
near 1) leave a bigger move over the next five sessions than waves that cancelled, beyond the baseline? Per session,
the rank correlation of the interference with ``|Y|`` less the baseline's prediction. Passes when its monthly mean is
positive with p < 0.05.

Every gated test also needs its controls (a placebo that shuffles the waves among each session's stocks reaches
p < 0.05 in at most 10% of draws; an oracle that knows the outcome reaches p < 0.01) and Benjamini-Hochberg q < 0.10
across every gated test in the ledger (``research/run.py``). Results are written whatever they say.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from brain import config, ripple, shocks
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research.eventtest import boot_interval, flip_p
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

IDS = ("R1a", "R1b", "R1c", "R1d")
GATE_TYPE = {"R1a": "report", "R1b": "size", "R1c": "direction", "R1d": "interference"}
VERSION = 1
TRAIN = ("2011-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-09-30")
P_MAX = 0.05
W_MIN = 0.5            # R1c: wave energy of at least half a usual day (ripple.KAPPA units)
W_HIGH = 1.0           # R1d: wave energy of at least one usual day
PLACEBO_FPR_MAX = 0.10
ORACLE_P_MAX = 0.01
N_FLIP, N_BOOT, N_PLACEBO, SEED = 10000, 2000, 200, 13
CLEAN = ("brain", "knowledge", "universe")
HERE = Path(__file__).resolve()
FIELDS = ("H", "W", "I", "ring", "z", "Y", "own5", "own60", "mkt5")


def code_files() -> list[Path]:
    from brain import bank, connect, series, timeutil
    from brain.research import eventtest, run, stats
    return [HERE, *(Path(m.__file__).resolve()
                    for m in (ripple, connect, shocks, bank, series, timeutil, eventtest, run, stats))]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def universe_shas() -> dict[str, str]:
    return {n: hashlib.sha256((config.ROOT / "universe" / f"{n}.json").read_bytes()).hexdigest()[:16]
            for n in shocks.UNIVERSES}


def spec(tid: str) -> dict:
    s = {"id": tid, "version": VERSION, "train": list(TRAIN), "test": list(TEST), "universes": universe_shas(),
         "code_sha": code_sha(), "stone_z": ripple.STONE_Z, "edge": ripple.EDGE, "ring": ripple.RING,
         "lookback": ripple.LOOKBACK, "rebuild": ripple.REBUILD, "h": ripple.H, "n_flip": N_FLIP, "n_boot": N_BOOT,
         "n_placebo": N_PLACEBO, "seed": SEED, "gate_type": GATE_TYPE[tid],
         "controls": {"placebo_fpr_max": PLACEBO_FPR_MAX, "oracle_p_max": ORACLE_P_MAX}}
    if tid == "R1b":
        s["gate"] = {"diff_gt": 0.0, "p_lt": P_MAX, "bh_q_lt": 0.10}
    elif tid == "R1c":
        s["gate"] = {"ic_gt": 0.0, "p_lt": P_MAX, "bh_q_lt": 0.10, "w_min": W_MIN}
    elif tid == "R1d":
        s["gate"] = {"ic_gt": 0.0, "p_lt": P_MAX, "bh_q_lt": 0.10, "w_high": W_HIGH}
    return s


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
    """Declare R1a-R1d (each once; a changed spec is refused: a change is a new id)."""
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
            log(f"declared {tid} ({rec['spec_sha']}): TRAIN {TRAIN[0]}..{TRAIN[1]}, TEST {TEST[0]}..{TEST[1]}; "
                "commit the ledger before running it")
    return out


# ---------- the rows ----------
def collect(px: dict, first: str, last: str, log=None) -> dict:
    """Per stock-day the waves reached (not a stone, energy above zero, outcome known): the fields in FIELDS plus the
    session's index and month. Also per session: the number of stones, and on sessions without stones the mean size
    of every stock's own move (R1a's calm baseline)."""
    cols: dict[str, list] = {f: [] for f in FIELDS}
    day, month = [], []
    per_day = []
    k = 0
    for d in ripple.days(px, first, last, log=log):
        z = d["z"]
        ok = (~d["stone"]) & (d["W"] > 0) & np.isfinite(d["Y"]) & np.isfinite(z) & np.isfinite(d["own5"]) \
            & np.isfinite(d["own60"])
        dry = (~d["stone"]) & (d["W"] == 0) & np.isfinite(z)
        per_day.append({"date": d["date"], "stones": int(len(d["S"])), "hit": int(ok.sum()),
                        "dry_absz": float(np.abs(z[dry]).mean()) if dry.any() else None})
        if ok.any() and np.isfinite(d["mkt5"]):
            for f in FIELDS:
                v = d[f]
                cols[f].append(np.asarray(v[ok] if np.ndim(v) else np.full(int(ok.sum()), v), np.float32))
            day.append(np.full(int(ok.sum()), k, np.int32))
            month.append(np.full(int(ok.sum()), d["date"][:7]))
        k += 1
    out = {f: (np.concatenate(cols[f]) if cols[f] else np.zeros(0, np.float32)) for f in FIELDS}
    out["day"] = np.concatenate(day) if day else np.zeros(0, np.int32)
    out["month"] = np.concatenate(month) if month else np.zeros(0, "U7")
    out["per_day"] = per_day
    return out


def features(rows: dict, ripple_terms: bool) -> np.ndarray:
    cols = [np.ones(len(rows["Y"])), np.log(rows["own5"] + 0.1), np.log(rows["own60"] + 0.1), np.abs(rows["z"]),
            rows["mkt5"]]
    if ripple_terms:
        cols += [np.log1p(np.abs(rows["H"])), np.log1p(rows["W"])]
    return np.column_stack(cols).astype(float)


def fit(rows: dict, ripple_terms: bool) -> np.ndarray:
    X, y = features(rows, ripple_terms), np.abs(rows["Y"]).astype(float)
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ---------- statistics ----------
def monthly(month: np.ndarray, v: np.ndarray) -> tuple[list[str], np.ndarray]:
    ms = sorted(set(month.tolist()))
    return ms, np.array([v[month == m].mean() for m in ms])


def ranks(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r


def daily_ic(day: np.ndarray, a: np.ndarray, b: np.ndarray, min_n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Per session, the rank correlation of ``a`` with ``b``; the sessions and their values."""
    ds, ics = [], []
    order = np.argsort(day, kind="mergesort")
    d_sorted = day[order]
    cuts = np.flatnonzero(np.diff(d_sorted)) + 1
    for idx in np.split(order, cuts):
        if len(idx) < min_n:
            continue
        ra, rb = ranks(a[idx]), ranks(b[idx])
        sa, sb = ra.std(), rb.std()
        if sa == 0 or sb == 0:
            continue
        ds.append(day[idx[0]])
        ics.append(float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb)))
    return np.array(ds), np.array(ics)


def day_months(rows: dict) -> dict[int, str]:
    return dict(zip(rows["day"].tolist(), rows["month"].tolist()))


def ic_stat(rows: dict, mask: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[list[str], np.ndarray, int]:
    dm = day_months(rows)
    ds, ics = daily_ic(rows["day"][mask], a[mask], b[mask])
    if not len(ds):
        return [], np.zeros(0), 0
    months, v = monthly(np.array([dm[int(d)] for d in ds]), ics)
    return months, v, len(ds)


def shuffled_within_day(day: np.ndarray, x: np.ndarray, rng) -> np.ndarray:
    out = x.copy()
    order = np.argsort(day, kind="mergesort")
    cuts = np.flatnonzero(np.diff(day[order])) + 1
    for idx in np.split(order, cuts):
        out[idx] = x[rng.permutation(idx)]
    return out


def evaluate_size(train: dict, test: dict) -> dict:
    rng = np.random.default_rng(SEED)
    b0, b1 = fit(train, False), fit(train, True)
    y = np.abs(test["Y"]).astype(float)
    p0, p1 = features(test, False) @ b0, features(test, True) @ b1
    l0, l1 = (y - p0) ** 2, (y - p1) ** 2
    months, d = monthly(test["month"], l0 - l1)
    p = flip_p(d, 1, N_FLIP, rng)
    hits = 0
    for _ in range(N_PLACEBO):
        sh = dict(test)
        sh["H"] = shuffled_within_day(test["day"], test["H"], rng)
        sh["W"] = shuffled_within_day(test["day"], test["W"], rng)
        _, dd = monthly(test["month"], l0 - (y - features(sh, True) @ b1) ** 2)
        hits += flip_p(dd, 1, 999, rng) < P_MAX
    _, dor = monthly(test["month"], l0 - 0.0 * y)
    oracle_p = flip_p(dor, 1, N_FLIP, rng)
    fpr = hits / N_PLACEBO
    ring = test["ring"]
    metrics = {"rows": int(len(y)), "months": len(months), "first": months[0] if months else None,
               "last": months[-1] if months else None, "loss_base": float(l0.mean()), "loss_ripple": float(l1.mean()),
               "diff": float(d.mean()) if len(d) else float("nan"), "interval90": boot_interval(d, N_BOOT, rng),
               "months_better": int((d > 0).sum()), "p": p, "coef_ripple": [float(x) for x in b1],
               "coef_base": [float(x) for x in b0], "mean_absY": float(y.mean()),
               "absY_by_withstand": {k: float(y[m_].mean()) if m_.any() else float("nan") for k, m_ in
                                     (("absorbed", np.abs(test["H"]) < ripple.ABSORB),
                                      ("felt", (np.abs(test["H"]) >= ripple.ABSORB) & (np.abs(test["H"]) < ripple.HIT)),
                                      ("hit", np.abs(test["H"]) >= ripple.HIT))},
               "absY_by_ring": {str(r_): float(y[ring == r_].mean()) if (ring == r_).any() else float("nan")
                                for r_ in (1, 2, 3)}}
    gate = {"diff_positive": metrics["diff"] > 0, "p_ok": p < P_MAX}
    gate["pass"] = bool(all(gate.values()))
    return {"metrics": metrics, "gate": gate, "p": p,
            "controls": {"n_placebo": N_PLACEBO, "placebo_fpr": fpr, "oracle_p": oracle_p,
                         "pass": bool(fpr <= PLACEBO_FPR_MAX and oracle_p <= ORACLE_P_MAX)}}


def _ic_test(rows: dict, mask: np.ndarray, a: np.ndarray, b: np.ndarray, rng) -> dict:
    months, v, n_days = ic_stat(rows, mask, a, b)
    p = flip_p(v, 1, N_FLIP, rng) if len(v) else 1.0
    hits = 0
    for _ in range(N_PLACEBO):
        sa = shuffled_within_day(rows["day"], a, rng)
        _, vv, _ = ic_stat(rows, mask, sa, b)
        hits += (flip_p(vv, 1, 999, rng) if len(vv) else 1.0) < P_MAX
    _, vo, _ = ic_stat(rows, mask, b, b)
    oracle_p = flip_p(vo, 1, N_FLIP, rng) if len(vo) else 1.0
    fpr = hits / N_PLACEBO
    return {"months": months, "v": v, "n_days": n_days, "p": p,
            "controls": {"n_placebo": N_PLACEBO, "placebo_fpr": fpr, "oracle_p": oracle_p,
                         "pass": bool(fpr <= PLACEBO_FPR_MAX and oracle_p <= ORACLE_P_MAX)}}


def evaluate_direction(test: dict, s_of_row: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(SEED)
    mask = test["W"] >= W_MIN
    H_, Y = test["H"].astype(float), test["Y"].astype(float)
    r = _ic_test(test, mask, H_, Y, rng)
    d = r["v"]
    # reported beside it: the top fifth by wave height against the bottom fifth, in usual days of five sessions
    spread = []
    dm = day_months(test)
    order = np.argsort(test["day"], kind="mergesort")
    cuts = np.flatnonzero(np.diff(test["day"][order])) + 1
    for idx in np.split(order, cuts):
        idx = idx[mask[idx]]
        if len(idx) < 20:
            continue
        q = np.quantile(H_[idx], [0.2, 0.8])
        top, bot = idx[H_[idx] >= q[1]], idx[H_[idx] <= q[0]]
        if len(top) and len(bot):
            spread.append((dm[int(test["day"][idx[0]])], float(Y[top].mean() - Y[bot].mean())))
    sm = {}
    for mth, v in spread:
        sm.setdefault(mth, []).append(v)
    sp = np.array([np.mean(v) for v in sm.values()]) if sm else np.zeros(0)
    gap_months, gap_v, _ = ic_stat(test, mask, H_ - test["z"].astype(float), Y)
    rev_months, rev_v, _ = ic_stat(test, mask, -test["z"].astype(float), Y)
    metrics = {"rows": int(mask.sum()), "sessions": r["n_days"], "months": len(r["months"]),
               "first": r["months"][0] if r["months"] else None, "last": r["months"][-1] if r["months"] else None,
               "ic": float(d.mean()) if len(d) else float("nan"), "interval90": boot_interval(d, N_BOOT, rng),
               "months_positive": int((d > 0).sum()), "p": r["p"],
               "spread_top_bottom": float(sp.mean()) if len(sp) else float("nan"),
               "ic_gap_reported": float(gap_v.mean()) if len(gap_v) else float("nan"),
               "ic_own_reversal_reported": float(rev_v.mean()) if len(rev_v) else float("nan")}
    gate = {"ic_positive": metrics["ic"] > 0, "p_ok": r["p"] < P_MAX}
    gate["pass"] = bool(all(gate.values()))
    return {"metrics": metrics, "gate": gate, "p": r["p"], "controls": r["controls"]}


def evaluate_interference(train: dict, test: dict) -> dict:
    rng = np.random.default_rng(SEED)
    b0 = fit(train, False)
    resid = np.abs(test["Y"]).astype(float) - features(test, False) @ b0
    mask = (test["W"] >= W_HIGH) & np.isfinite(test["I"])
    I_ = np.nan_to_num(test["I"].astype(float))
    r = _ic_test(test, mask, I_, resid, rng)
    d = r["v"]
    hi = mask & (I_ >= 0.5)
    lo = mask & (I_ < 0.5)
    y = np.abs(test["Y"]).astype(float)
    metrics = {"rows": int(mask.sum()), "sessions": r["n_days"], "months": len(r["months"]),
               "first": r["months"][0] if r["months"] else None, "last": r["months"][-1] if r["months"] else None,
               "ic": float(d.mean()) if len(d) else float("nan"), "interval90": boot_interval(d, N_BOOT, rng),
               "months_positive": int((d > 0).sum()), "p": r["p"],
               "absY_constructive": float(y[hi].mean()) if hi.any() else float("nan"),
               "absY_destructive": float(y[lo].mean()) if lo.any() else float("nan"),
               "share_destructive": float(lo.sum() / mask.sum()) if mask.any() else float("nan")}
    gate = {"ic_positive": metrics["ic"] > 0, "p_ok": r["p"] < P_MAX}
    gate["pass"] = bool(all(gate.values()))
    return {"metrics": metrics, "gate": gate, "p": r["p"], "controls": r["controls"]}


def evaluate_report(test: dict) -> dict:
    """R1a: the waves on the day, described."""
    rng = np.random.default_rng(SEED)
    mask = test["W"] >= W_MIN
    z = test["z"].astype(float)
    months, v, n_days = ic_stat(test, mask, test["H"].astype(float), z)
    hi = (test["W"] >= W_HIGH) & np.isfinite(test["I"])
    _, vi, _ = ic_stat(test, hi, np.nan_to_num(test["I"].astype(float)), np.abs(z))
    per = test["per_day"]
    calm = [d["dry_absz"] for d in per if d["dry_absz"] is not None]
    ring = test["ring"]
    H_ = np.abs(test["H"])
    metrics = {"sessions": len(per), "sessions_with_dry_land": len(calm),
               "stones_per_session": float(np.mean([d["stones"] for d in per])) if per else float("nan"),
               "hit_per_session": float(np.mean([d["hit"] for d in per])) if per else float("nan"),
               "same_day_ic": float(v.mean()) if len(v) else float("nan"),
               "same_day_ic_interval90": boot_interval(v, N_BOOT, rng) if len(v) else None,
               "absz_by_ring": {str(r_): float(np.abs(z[ring == r_]).mean()) if (ring == r_).any() else float("nan")
                                for r_ in (1, 2, 3)},
               "wave_by_ring": {str(r_): float(H_[ring == r_].mean()) if (ring == r_).any() else float("nan")
                                for r_ in (1, 2, 3)},
               "absz_dry_land": float(np.mean(calm)) if calm else float("nan"),
               "share_by_withstand": {"absorbed": float((H_ < ripple.ABSORB).mean()),
                                      "felt": float(((H_ >= ripple.ABSORB) & (H_ < ripple.HIT)).mean()),
                                      "hit": float((H_ >= ripple.HIT).mean())},
               "interference_same_day_ic": float(vi.mean()) if len(vi) else float("nan"),
               "months": len(months), "first": months[0] if months else None, "last": months[-1] if months else None}
    return {"metrics": metrics, "gate": {"pass": False}, "p": float("nan"), "controls": {"pass": True}}


# ---------- running ----------
def test(bank: Bank, *, trial: bool = False, log=print, tickers=None) -> list[dict]:
    """Run every declared ripple test that has no result yet, once."""
    from brain.research import run
    trial = is_trial(trial)
    led, res = paths(trial)
    recs = ledger.read(led)
    todo = []
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
            if s["universes"] != universe_shas():
                raise Refused(f"{tid}: a universe file changed since the declaration; declare a new id")
            if not committed_declaration(tid, decl["spec_sha"]):
                raise Refused(f"{tid}: commit the ledger with its declaration before running it")
        todo.append(decl)
    if not todo:
        raise Refused("R1a-R1d already have results; a rerun needs research withdraw and a new id")
    tickers = tickers or shocks.universe_tickers()
    px = ripple.prices(bank, tickers, log=log)
    if log:
        log("collecting TRAIN waves")
    train = collect(px, *TRAIN, log=None)
    if log:
        log(f"TRAIN: {len(train['Y'])} stock-days; collecting TEST waves")
    tst = collect(px, *TEST, log=None)
    if log:
        log(f"TEST: {len(tst['Y'])} stock-days")
    out = []
    for decl in todo:
        tid = decl["id"]
        if tid == "R1a":
            r = evaluate_report(tst)
        elif tid == "R1b":
            r = evaluate_size(train, tst)
        elif tid == "R1c":
            r = evaluate_direction(tst)
        else:
            r = evaluate_interference(train, tst)
        rec = {"op": "result", "kind": "ripple", "id": tid, "spec_sha": decl["spec_sha"], "gate_type": GATE_TYPE[tid],
               **r, "at": now_iso(), **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            m = r["metrics"]
            if tid == "R1a":
                log(f"R1a: same-day rank correlation of wave and own move {m['same_day_ic']:+.3f}; own move by ring "
                    f"{m['absz_by_ring']}, on dry land {m['absz_dry_land']:.3f}")
            elif tid == "R1b":
                log(f"R1b: loss baseline {m['loss_base']:.4f} vs ripple {m['loss_ripple']:.4f} (monthly difference "
                    f"{m['diff']:+.5f}, p {m['p']:.4f}). Gate {'met' if r['gate']['pass'] else 'not met'}")
            else:
                log(f"{tid}: monthly mean rank correlation {m['ic']:+.4f} (p {m['p']:.4f}). "
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
    p = sub.add_parser("rippletest", help="declare or run R1a-R1d, the ripple tests (docs/RIPPLES.md)")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.set_defaults(fn=_cmd)
