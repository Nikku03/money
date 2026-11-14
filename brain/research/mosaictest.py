"""Declared tests of the mosaic (brain/mosaic.py, docs/MOSAIC.md).

A picture is a hidden event that a later filing reveals: SALE, a sale of new shares; BOOKS, late or unreliable
accounts; CUTS, a restructuring or a write-off. Each has a model of the public pieces and a baseline that knows only the company's habit, both fitted on
2011-2023 and saved in ``knowledge/mosaic_p1.json``, pinned here by its sha.

TEST is the stock-weeks ending in 2024-01-01..2026-09-30 whose H sessions have closed, read at *periods*: the first
week's end in TEST, then each next week's end at least H sessions after the last one taken, so that no two periods'
outcomes overlap.

**P1a** (SALE), **P2a** (BOOKS) and **P3a** (CUTS) ask whether the pieces see the event coming better than the habit alone. Per
stock the score is the log loss of the chance; per period, the baseline's mean loss minus the model's. The statistic
is the mean of those period differences, with a one-sided sign-flip p. The test passes when the difference is
positive with p < 0.05; the model's mean chance over TEST is within CAL of the actual rate, and so is the rate
among each period's top twentieth by the model; and the controls pass (the model's chances shuffled among each
period's stocks reach p < 0.05 in at most 10% of draws; an oracle that knows the outcome reaches p < 0.01).
Reported beside it, not gated: the same ranking among the stocks with no such event in three years, where the habit
is silent and only the other pieces can see anything.

**P1b** (SALE), **P2b** (BOOKS) and **P3b** (CUTS) ask whether the stocks the picture flags then lag. Each period's top twentieth by
the model's chance is flagged. A flagged stock's outcome is its return over SPY's across the next H sessions, set
against the period's unflagged stocks in its 60-session volatility decile: the share of them it beat, less one half.
Per period these are averaged; the statistic is the mean of the period values with a one-sided sign-flip p. The
test passes when it is negative with p < 0.05 and its controls pass (a random peer standing in for each flagged
stock reaches p < 0.05 in at most 10% of draws; an oracle that puts each at its peers' 10th percentile reaches
p < 0.01). The mean return against those peers, which a trade would earn before costs, is reported beside it.

Every test also needs Benjamini-Hochberg q < 0.10 across every gated test in the ledger (``research/run.py``).
Results are written whatever they say.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from brain import config, mosaic, shocks
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research.eventtest import boot_interval, flip_p
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.research.shocktest import auc, logloss
from brain.timeutil import now_iso

IDS = ("P1a", "P1b", "P2a", "P2b", "P3a", "P3b")
PICTURE = {"P1a": "sale", "P1b": "sale", "P2a": "books", "P2b": "books", "P3a": "cuts", "P3b": "cuts"}
ASKS = {"P1a": "event", "P1b": "lag", "P2a": "event", "P2b": "lag", "P3a": "event", "P3b": "lag"}
VERSION = 1
TEST = ("2024-01-01", "2026-09-30")
P_MAX = 0.05
CAL = (0.67, 1.5)          # actual over predicted, overall and in the top twentieth
TOP = 0.05
DECILES = 10
MIN_PEERS = 20
PLACEBO_FPR_MAX = 0.10
ORACLE_P_MAX = 0.01
N_FLIP, N_BOOT, N_PLACEBO, SEED = 10000, 2000, 200, 17
CLEAN = ("brain", "knowledge", "universe")
HERE = Path(__file__).resolve()
MODEL = "knowledge/mosaic_p1.json"
KIND = "mosaic"


def code_files() -> list[Path]:
    """What the numbers run through: this file, the mosaic and the price features under it, the bank's view, the
    series store, the release-window rule, the time rules, the sign-flip helpers, the scores, and the verdicts."""
    from brain import bank, events, series, timeutil
    from brain.research import eventtest, run, shocktest, stats
    return [HERE, *(Path(m.__file__).resolve()
                    for m in (mosaic, shocks, events, bank, series, timeutil, eventtest, shocktest, run, stats))]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def universe_shas() -> dict[str, str]:
    return {n: hashlib.sha256((config.ROOT / "universe" / f"{n}.json").read_bytes()).hexdigest()[:16]
            for n in mosaic.UNIVERSES}


def spec(tid: str) -> dict:
    model = mosaic.load()
    if model is None:
        raise Refused("fit the models first (python -m brain mosaic fit) and commit them")
    pic = PICTURE[tid]
    common = {"id": tid, "version": VERSION, "test": list(TEST), "picture": pic, "asks": ASKS[tid], "h": mosaic.H,
              "gap": mosaic.GAP, "universes": universe_shas(), "code_sha": code_sha(),
              "model_sha": mosaic.model_sha(), "train": model["train"], "feats": list(mosaic.PICTURES[pic]["feats"]),
              "habit": list(mosaic.PICTURES[pic]["habit"]), "top": TOP, "n_flip": N_FLIP, "n_boot": N_BOOT,
              "n_placebo": N_PLACEBO, "seed": SEED,
              "controls": {"placebo_fpr_max": PLACEBO_FPR_MAX, "oracle_p_max": ORACLE_P_MAX}}
    if ASKS[tid] == "event":
        return {**common, "gate": {"diff_gt": 0.0, "p_lt": P_MAX, "cal": list(CAL), "bh_q_lt": 0.10}}
    return {**common, "stat": "percentile", "deciles": DECILES, "min_peers": MIN_PEERS,
            "gate": {"stat_lt": 0.0, "p_lt": P_MAX, "bh_q_lt": 0.10}}


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
    files = [p.relative_to(root).as_posix() for p in code_files()] + [MODEL]
    changed = [x for x in _git("diff", "--name-only", commit, "HEAD", "--", *files).stdout.splitlines() if x.strip()]
    return f"code or the model changed since the declare commit {commit[:12]}: {', '.join(changed)}" if changed else None


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
    """Declare P1a to P3b (each once; a changed spec is refused: a change is a new id)."""
    trial = is_trial(trial)
    led, _ = paths(trial)
    if not trial and (dirty := tree_dirty()):
        raise Refused("commit brain/, knowledge/ and universe/ first; a declaration pins the committed code and "
                      "model: " + "; ".join(dirty))
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
            log(f"declared {tid} ({rec['spec_sha']}): TEST {TEST[0]}..{TEST[1]}; commit the ledger before running it")
    return out


# ---------- periods ----------
def periods(cal: np.ndarray, dates: np.ndarray, h: int) -> list[str]:
    """The dates to read: the first of ``dates`` (week ends), then each next one at least ``h`` sessions of ``cal``
    after the last taken. No two periods' h-session outcomes overlap."""
    out, last = [], None
    for d in sorted({str(x) for x in dates}):
        i = int(np.searchsorted(cal, d))
        if last is None or i - last >= h:
            out.append(d)
            last = i
    return out


def at_periods(tab: dict, keep: list[str]) -> dict:
    m = np.isin(tab["date"], np.array(keep))
    return {k: v[m] for k, v in tab.items()}


def by_period(period: np.ndarray, d: np.ndarray) -> tuple[list[str], np.ndarray]:
    ps = sorted(set(period.tolist()))
    return ps, np.array([d[period == p].mean() for p in ps])


def rank_in(period: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Each row's rank within its period, as a fraction (the highest chance is near 0)."""
    out = np.empty(len(p))
    for w in np.unique(period):
        idx = np.flatnonzero(period == w)
        o = np.argsort(-p[idx], kind="mergesort")
        rk = np.empty(len(idx))
        rk[o] = (np.arange(len(idx)) + 1) / len(idx)
        out[idx] = rk
    return out


# ---------- P1a, P2a: does the picture see the event coming? ----------
def event_controls(period, y, p, loss_base, rng) -> dict:
    """Placebo: the model's chances shuffled among each period's stocks. Oracle: a chance that knows the outcome."""
    groups = [np.flatnonzero(period == w) for w in np.unique(period)]
    hits = 0
    for _ in range(N_PLACEBO):
        perm = np.arange(len(y))
        for idx in groups:
            perm[idx] = rng.permutation(idx)
        _, d = by_period(period, loss_base - logloss(p[perm], y))
        hits += flip_p(d, 1, 999, rng) < P_MAX
    _, d = by_period(period, loss_base - logloss(np.where(y > 0, 0.9, 0.001), y))
    oracle_p = flip_p(d, 1, N_FLIP, rng)
    fpr = hits / N_PLACEBO
    return {"n_placebo": N_PLACEBO, "placebo_fpr": fpr, "oracle_p": oracle_p,
            "pass": bool(fpr <= PLACEBO_FPR_MAX and oracle_p <= ORACLE_P_MAX)}


def evaluate_event(tab: dict, part: dict, pic: str) -> dict:
    """``tab``: rows at the periods. ``part``: the picture's fitted model and baseline."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(tab[mosaic.PICTURES[pic]["y"]], float)
    period = np.asarray(tab["date"])
    p, pb = mosaic.chance(part["model"], tab["X"]), mosaic.chance(part["baseline"], tab["X"])
    loss, loss_b = logloss(p, y), logloss(pb, y)
    ps, d = by_period(period, loss_b - loss)
    pv = flip_p(d, 1, N_FLIP, rng)
    rank, rank_b = rank_in(period, p), rank_in(period, pb)
    top = rank <= TOP
    ratio = float(y.mean() / p.mean()) if len(y) else float("nan")
    top_ratio = float(y[top].mean() / p[top].mean()) if top.any() else float("nan")

    def capture(r, k):
        return float(y[r <= k].sum() / y.sum()) if y.sum() else float("nan")
    metrics = {"rows": len(y), "stocks": len(set(tab["t"].tolist())), "periods": len(ps),
               "first": ps[0] if ps else None, "last": ps[-1] if ps else None, "events": int(y.sum()),
               "loss": float(loss.mean()), "loss_base": float(loss_b.mean()),
               "diff": float(d.mean()) if len(d) else float("nan"), "interval90": boot_interval(d, N_BOOT, rng),
               "periods_better": int((d > 0).sum()), "p": pv, "rate": float(y.mean()), "mean_chance": float(p.mean()),
               "mean_base": float(pb.mean()), "cal_ratio": ratio,
               "top_rate": float(y[top].mean()) if top.any() else float("nan"),
               "top_chance": float(p[top].mean()) if top.any() else float("nan"), "top_ratio": top_ratio,
               "top_rate_base": float(y[rank_b <= TOP].mean()) if (rank_b <= TOP).any() else float("nan"),
               "auc": auc(p, y), "auc_base": auc(pb, y),
               "capture": {f"top{int(k * 100)}": capture(rank, k) for k in (0.01, 0.05, 0.10)},
               "capture_base": {f"top{int(k * 100)}": capture(rank_b, k) for k in (0.01, 0.05, 0.10)}}
    ft = mosaic.first_time(pic, tab["X"])
    if ft.any() and y[ft].sum():                # reported, not gated: where the habit is silent
        tf = rank_in(period[ft], p[ft]) <= TOP
        metrics["silent"] = {"rows": int(ft.sum()), "events": int(y[ft].sum()), "rate": float(y[ft].mean()),
                            "auc": auc(p[ft], y[ft]), "top_rate": float(y[ft][tf].mean()),
                            "top_held": float(y[ft][tf].sum() / y[ft].sum())}
    gate = {"diff_positive": metrics["diff"] > 0, "p_ok": pv < P_MAX, "cal_ok": CAL[0] <= ratio <= CAL[1],
            "top_cal_ok": CAL[0] <= top_ratio <= CAL[1]}
    gate["pass"] = bool(all(gate.values()))
    return {"metrics": metrics, "gate": gate, "controls": event_controls(period, y, p, loss_b, rng), "p": pv}


# ---------- P1b, P2b: do the flagged stocks lag? ----------
def flagged(tab: dict, part: dict) -> list[dict]:
    """Each period's top twentieth by the model's chance whose outcome is known, each with the outcomes of the
    period's unflagged stocks in its 60-session volatility decile."""
    period = np.asarray(tab["date"])
    p = mosaic.chance(part["model"], tab["X"])
    rank = rank_in(period, p)
    lv = np.asarray(tab["X"][:, mosaic.COLS.index("lv60")], float)
    fwd = np.asarray(tab["fwd"], float)
    out = []
    for w in np.unique(period):
        idx = np.flatnonzero((period == w) & np.isfinite(fwd))
        if len(idx) < DECILES * MIN_PEERS:
            continue
        flag = rank[idx] <= TOP
        q = np.quantile(lv[idx], np.linspace(0, 1, DECILES + 1))
        for i in idx[flag]:
            k = min(max(int(np.searchsorted(q, lv[i], side="right")) - 1, 0), DECILES - 1)
            peers = fwd[idx[~flag & (lv[idx] >= q[k]) & (lv[idx] <= q[k + 1])]]
            if len(peers) < MIN_PEERS:
                continue
            y = float(fwd[i])
            out.append({"t": str(tab["t"][i]), "date": str(w), "y": y, "peers": peers, "chance": float(p[i]),
                        "peer_mean": float(peers.mean()), "pct": beaten(peers, y),
                        "abn": y - float(peers.mean())})
    return out


def beaten(peers: np.ndarray, y: float) -> float:
    """The share of ``peers`` an outcome beat, a tie counting half: a peer's own outcome then scores one half on
    average, so the placebo (a random peer in the flagged stock's place) is centred on zero."""
    return float((peers < y).mean() + 0.5 * (peers == y).mean())


def lag_values(evs: list[dict], ys=None) -> np.ndarray:
    """Each flagged stock's share of peers beaten, less one half. ``ys`` replaces the outcomes (the controls)."""
    return np.array([beaten(ev["peers"], ev["y"] if ys is None else ys[k]) - 0.5 for k, ev in enumerate(evs)])


def lag_controls(evs: list[dict], period: np.ndarray, rng) -> dict:
    """Placebo: a random peer stands in for each flagged stock. Oracle: each at its peers' 10th percentile."""
    hits = 0
    for _ in range(N_PLACEBO):
        _, d = by_period(period, lag_values(evs, [float(rng.choice(ev["peers"])) for ev in evs]))
        hits += flip_p(d, -1, 999, rng) < P_MAX
    _, d = by_period(period, lag_values(evs, [float(np.quantile(ev["peers"], 0.1)) for ev in evs]))
    oracle_p = flip_p(d, -1, N_FLIP, rng)
    fpr = hits / N_PLACEBO
    return {"n_placebo": N_PLACEBO, "placebo_fpr": fpr, "oracle_p": oracle_p,
            "pass": bool(fpr <= PLACEBO_FPR_MAX and oracle_p <= ORACLE_P_MAX)}


def evaluate_lag(evs: list[dict]) -> dict:
    rng = np.random.default_rng(SEED)
    period = np.array([ev["date"] for ev in evs])
    ps, d = by_period(period, lag_values(evs)) if evs else ([], np.zeros(0))
    pv = flip_p(d, -1, N_FLIP, rng)
    y = np.array([ev["y"] for ev in evs])
    abn = np.array([ev["abn"] for ev in evs])
    pct = np.array([ev["pct"] for ev in evs])
    _, da = by_period(period, abn) if evs else ([], np.zeros(0))
    metrics = {"flagged": len(evs), "stocks": len({ev["t"] for ev in evs}), "periods": len(ps),
               "first": ps[0] if ps else None, "last": ps[-1] if ps else None,
               "stat": float(d.mean()) if len(d) else float("nan"), "interval90": boot_interval(d, N_BOOT, rng),
               "periods_negative": int((d < 0).sum()), "p": pv,
               "percentile": float(pct.mean()) if len(pct) else float("nan"),
               "share_behind": float((pct < 0.5).mean()) if len(pct) else float("nan"),
               "mean_outcome": float(y.mean()) if len(y) else float("nan"),
               "median_outcome": float(np.median(y)) if len(y) else float("nan"),
               "peer_mean": float(np.mean([ev["peer_mean"] for ev in evs])) if evs else float("nan"),
               "abnormal": float(da.mean()) if len(da) else float("nan"),
               "abnormal_interval90": boot_interval(da, N_BOOT, rng),
               "abnormal_p": flip_p(da, -1, N_FLIP, rng),
               "mean_chance": float(np.mean([ev["chance"] for ev in evs])) if evs else float("nan")}
    gate = {"negative": metrics["stat"] < 0, "p_ok": pv < P_MAX}
    gate["pass"] = bool(all(gate.values()))
    return {"metrics": metrics, "gate": gate, "controls": lag_controls(evs, period, rng), "p": pv}


# ---------- running ----------
def test_table(bank: Bank, tickers=None, log=print) -> dict:
    """TEST's rows at its periods."""
    tab = mosaic.table(bank, tickers or mosaic.universe_tickers(), TEST[0], TEST[1], log=log)
    cal = shocks.market(Bank(bank.root).view())[0]
    return at_periods(tab, periods(cal, tab["date"], mosaic.H))


def test(bank: Bank, *, trial: bool = False, log=print, tickers=None) -> list[dict]:
    """Run every declared test here that has no result yet, once."""
    from brain.research import run
    trial = is_trial(trial)
    led, res = paths(trial)
    recs = ledger.read(led)
    todo = []
    for tid in IDS:
        decl = _declared(recs, tid)
        if decl is None:
            raise Refused(f"{tid} is not declared")
        if any(r["op"] == "result" and r.get("kind") == KIND and r["id"] == tid for r in recs):
            continue
        s = decl["spec"]
        if spec_sha(s) != decl["spec_sha"]:
            raise Refused(f"{tid}: its declared spec does not match its spec_sha")
        if s["model_sha"] != mosaic.model_sha():
            raise Refused(f"{tid}: {MODEL} is not the declared model")
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
        raise Refused("every mosaic test already has a result; a rerun needs research withdraw and a new id")
    model = mosaic.load()
    tab = test_table(bank, tickers, log=log)
    out = []
    for decl in todo:
        tid, pic = decl["id"], PICTURE[decl["id"]]
        r = evaluate_event(tab, model[pic], pic) if ASKS[tid] == "event" else evaluate_lag(flagged(tab, model[pic]))
        rec = {"op": "result", "kind": KIND, "id": tid, "spec_sha": decl["spec_sha"], "gate_type": ASKS[tid],
               "picture": pic, **r, "at": now_iso(), **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            m = r["metrics"]
            if ASKS[tid] == "event":
                log(f"{tid} ({pic}): {m['rows']} stock-periods in {m['periods']} periods, {m['events']} events; log "
                    f"loss {m['loss']:.5f} vs the habit's {m['loss_base']:.5f} (difference {m['diff']:+.5f}, p "
                    f"{m['p']:.4f}); chance {m['mean_chance'] * 100:.2f}% vs actual {m['rate'] * 100:.2f}%; the top 5% "
                    f"held {m['capture']['top5'] * 100:.0f}% of the events (the habit's "
                    f"{m['capture_base']['top5'] * 100:.0f}%). Gate {'met' if r['gate']['pass'] else 'not met'}")
            else:
                log(f"{tid} ({pic}): {m['flagged']} flagged in {m['periods']} periods; share of peers beaten less "
                    f"half {m['stat']:+.4f} (p {m['p']:.4f}); against peers {m['abnormal'] * 100:+.2f}% over "
                    f"{mosaic.H} sessions. Gate {'met' if r['gate']['pass'] else 'not met'}")
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
    p = sub.add_parser("mosaictest", help="declare or run P1a-P3b, the mosaic's tests (docs/MOSAIC.md)")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.set_defaults(fn=_cmd)
