"""Declared tests of step 2 of the plan in research/POSTMORTEM-2026-11.md: tilt toward boring quality, and buy
after fear (docs/TILT.md).

**T1a, the quality tilt.** A list a trader could have held: on each month's last session, the 500 largest stocks
by market value known that day (weighted-average shares from SEC filings times the close, ``metrics.market_cap_any``)
among the union of ``universe/us_large``, ``us_broad``, ``us_small``, ``us_micro`` and ``us_watch`` (today's largest
SEC filers with XBRL) that have a price that day and at least three of the four traits below. Membership is decided
by what was known that day, not by today's market value; companies that later failed or were bought are still
missing, and the list's own gap to RSP (below) measures what that adds.

Each member's quality score is the mean of its cross-sectional ranks among the members on the traits it has:

- profitable: trailing-twelve-month net income over total assets (the latest balance sheet, at most 200 days old);
- not diluting: minus the change in weighted-average shares over about a year (both counts as known at the time,
  on one split basis, their periods 270-460 days apart);
- not a lottery ticket: minus the standard deviation of daily log returns over the last 252 sessions (at least 200);
- near its high: the close over the highest close of the last 252 sessions (at least 200).

The portfolio holds the top third of the members by score, equal weight, through the research harness
(``brain/research/harness.py``): the signal from data known at the month-end close, the fill at the next day's
close, 10 bps per unit of turnover. The benchmark is the list itself at equal weight. TEST is every month from
2019-01-01 whose outcome is known; TRAIN (2011-2018) is reported. The gate: TEST mean net excess over the list > 0,
the harness's permutation p < 0.05 (the score shuffled among the members each month, 5,000 draws), the controls
passing on this list (an oracle reaches p < 0.001; random signals redrawn every month reach p < 0.05 in 1%-10% of
200 draws), and Benjamini-Hochberg q < 0.10 across every gated test in the ledger.

Reported beside it, not gated: each trait alone; the same tilt by market value ("the index without its junk": the
list weighted by market value, less its bottom third by score) against the list by market value; all of them
against SPY and against RSP (the S&P 500 at equal weight, a real fund that held the companies that later failed or
were bought); and the list's own gap to RSP and to SPY, which is what the survivor-only data adds.

**T1b, buying after fear.** At each week's last session, a fear spike is the VIX closing at least 1.25 times the
mean of its previous 63 closes. For 13 weeks counting a spike's own week the portfolio holds SPY at 100%; otherwise
at 70%, the rest in Treasury bills (the 13-week bill rate, ^IRX). The weight decided at a week's last close takes
effect at the next session's close; each change costs 10 bps per unit. The benchmark is a constant mix holding SPY
at the same average weight over the window, so only the timing is tested. The statistic is the mean weekly return
over that mix. TEST is 1993-02-01 to 2010-12-31, weeks no session here had looked at (the lesson came from
2011-2026). p: the weight path shifted in a circle against the returns by at least 52 weeks, 10,000 draws,
one-sided. The gate: mean > 0, p < 0.05, the controls passing (an oracle at 100% exactly in the weeks SPY beats
bills reaches p < 0.01; random spike weeks at the observed rate reach p < 0.05 in 1%-10% of 200 draws), and BH
q < 0.10. Reported beside it, not gated: 2011-01-01 to the latest week (seen before), and the timed portfolio
against SPY held at 100%.

Results are written whatever they say.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from brain import config, metrics
from brain.bank import Bank, canon
from brain.research import harness, ledger
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.research.hypotheses import SPLIT
from brain.research.panel import build
from brain.research.signals import rank_cs
from brain.timeutil import days_between, now_iso

IDS = ("T1a", "T1b")
ASKS = {"T1a": "tilt", "T1b": "timing"}
KIND = "tilt"
VERSION = 1
CLEAN = ("brain", "knowledge", "universe")
HERE = Path(__file__).resolve()

# T1a
UNIVERSES = ("us_large", "us_broad", "us_small", "us_micro", "us_watch")
LIST_SIZE = 500
WINDOW = 252
MIN_SESSIONS = 200
MIN_TRAITS = 3
TRAITS = ("profit", "no_dilution", "calm", "near_high")
ASSETS_STALE_DAYS = 200
SHARES_GAP = (270, 460)
FRAC = 1 / 3
COST_BPS = 10.0
N_PERM = 5000
N_PERM_TRAIT = 1000
N_RANDOM = 200
REFS = ("SPY", "RSP")

# T1b
FEAR_RATIO = 1.25
FEAR_LOOKBACK = 63
FEAR_WEEKS = 13
W_CALM, W_FEAR = 0.70, 1.00
T1B_TEST = ("1993-02-01", "2010-12-31")
T1B_SEEN = "2011-01-01"
T1B_START = "1990-01-01"
N_SHIFT, MIN_SHIFT = 10000, 52
N_ORACLE_SHIFT, N_RANDOM_SHIFT = 2000, 500
ORACLE_P_MAX = 0.01
FPR = (0.01, 0.10)
P_MAX = 0.05
BH_Q = 0.10
SEED = 23


# ---------- what a declaration pins ----------
def code_files() -> list[Path]:
    """What the numbers run through: this file, the harness and panel, the metrics and the bank's view, the series
    store and the price source, the time rules, the ranks, and the verdicts."""
    from brain import bank, series, timeutil
    from brain.research import panel, run, signals, stats
    from brain.sources import prices
    return [HERE, *(Path(m.__file__).resolve()
                    for m in (harness, panel, signals, metrics, bank, series, prices, timeutil, run, stats))]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def universe_shas() -> dict[str, str]:
    return {n: hashlib.sha256((config.ROOT / "universe" / f"{n}.json").read_bytes()).hexdigest()[:16]
            for n in UNIVERSES}


def tickers() -> list[str]:
    out: set[str] = set()
    for n in UNIVERSES:
        out |= set(json.loads((config.ROOT / "universe" / f"{n}.json").read_text())["tickers"])
    return sorted(out)


def spec(tid: str) -> dict:
    common = {"id": tid, "version": VERSION, "asks": ASKS[tid], "code_sha": code_sha(), "seed": SEED,
              "gate": {"stat_gt": 0.0, "p_lt": P_MAX, "bh_q_lt": BH_Q}}
    if tid == "T1a":
        return {**common, "universes": universe_shas(), "list_size": LIST_SIZE, "traits": list(TRAITS),
                "window": WINDOW, "min_sessions": MIN_SESSIONS, "min_traits": MIN_TRAITS,
                "assets_stale_days": ASSETS_STALE_DAYS, "shares_gap": list(SHARES_GAP), "frac": FRAC,
                "cost_bps": COST_BPS, "split": dict(SPLIT), "n_perm": N_PERM, "n_random": N_RANDOM,
                "refs": list(REFS),
                "controls": {"oracle_p_lt": 0.001, "random_fpr": [0.01, 0.10]}}
    return {**common, "fear_ratio": FEAR_RATIO, "fear_lookback": FEAR_LOOKBACK, "fear_weeks": FEAR_WEEKS,
            "weights": [W_CALM, W_FEAR], "cost_bps": COST_BPS, "test": list(T1B_TEST), "seen_from": T1B_SEEN,
            "data_from": T1B_START, "n_shift": N_SHIFT, "min_shift": MIN_SHIFT,
            "controls": {"oracle_p_lt": ORACLE_P_MAX, "random_fpr": list(FPR), "n_random": N_RANDOM,
                         "n_random_shift": N_RANDOM_SHIFT, "n_oracle_shift": N_ORACLE_SHIFT}}


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
    """Declare T1a and T1b (each once; a changed spec is refused: a change is a new id)."""
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
            log(f"declared {tid} ({rec['spec_sha']}); commit the ledger before running it")
    return out


# ---------- T1a: the traits, the list, the score ----------
def profitability(view, t: str) -> float | None:
    """Trailing-twelve-month net income over the latest total assets (USD, at most ASSETS_STALE_DAYS old)."""
    ni = metrics.ttm(view, t, metrics.NET_INCOME)
    if ni is None or ni.inputs[0].unit != "USD":
        return None
    ps = view.periods(t, "us-gaap:Assets")
    if not ps:
        return None
    a = ps[-1]
    if a.unit != "USD" or not a.value or a.value <= 0 or days_between(a.as_of, view.t) > ASSETS_STALE_DAYS:
        return None
    r = ni.value / a.value
    return r if -1.0 <= r <= 1.0 else None


def share_growth(bank_t: Bank, view, t: str, book) -> float | None:
    """Weighted-average shares now over about a year earlier, minus one, both as known at the time and on one split
    basis; None unless the two periods end SHARES_GAP days apart."""
    now = metrics.latest_shares_any(view, t)
    if now is None:
        return None
    then = metrics.latest_shares_any(bank_t.view((date.fromisoformat(view.t) - timedelta(days=365)).isoformat()), t)
    if then is None or not SHARES_GAP[0] <= days_between(then.as_of, now.as_of) <= SHARES_GAP[1]:
        return None
    a = now.value * book.basis_after(t, now.known_at)
    b = then.value * book.basis_after(t, then.known_at)
    if a <= 0 or b <= 0:
        return None
    g = a / b - 1.0
    return g if -0.9 <= g <= 5.0 else None


def price_traits(adj: np.ndarray, rebal) -> dict[str, np.ndarray]:
    """``calm`` (minus the 252-session volatility of daily log returns) and ``near_high`` (the close over the
    252-session high) on each rebalance day, from prices up to and including it."""
    rebal = np.asarray(rebal)
    N = adj.shape[1]
    calm = np.full((len(rebal), N), np.nan)
    high = np.full((len(rebal), N), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for m, i in enumerate(rebal):
            lo = int(i) - WINDOW
            if lo < 0:
                continue
            x = adj[lo:int(i) + 1]
            lr = np.diff(np.log(x), axis=0)
            n = np.isfinite(lr).sum(0)
            ok = (n >= MIN_SESSIONS) & np.isfinite(x[-1]) & (x[-1] > 0)
            if not ok.any():
                continue
            sd = np.full(N, np.nan)
            sd[ok] = np.nanstd(lr[:, ok], axis=0, ddof=1)
            hi = np.full(N, np.nan)
            hi[ok] = np.nanmax(x[1:, ok], axis=0)
            calm[m, ok] = -sd[ok]
            nh = x[-1] / hi
            good = ok & np.isfinite(nh)
            high[m, good] = nh[good]
    return {"calm": calm, "near_high": high}


def fundamentals(bank: Bank, panel, log=None) -> dict[str, np.ndarray]:
    """``mcap``, ``profit`` and ``no_dilution`` per (rebalance, ticker), each ticker through its own Bank on the
    journal (its facts only, freed after it), each value from a view at the rebalance day."""
    M1, N = len(panel.rebal), len(panel.tickers)
    out = {k: np.full((M1, N), np.nan) for k in ("mcap", "profit", "no_dilution")}
    for j, t in enumerate(panel.tickers):
        b = Bank(bank.root, use_index=bank.use_index)
        for m, i in enumerate(panel.rebal):
            if not panel.listed[m, j]:
                continue
            v = b.view(str(panel.dates[i]))
            mc = metrics.market_cap_any(v, t, panel.book)
            if mc is not None and mc.value > 0:
                out["mcap"][m, j] = mc.value
            p = profitability(v, t)
            if p is not None:
                out["profit"][m, j] = p
            g = share_growth(b, v, t, panel.book)
            if g is not None:
                out["no_dilution"][m, j] = -g
        if log and (j + 1) % 250 == 0:
            log(f"  accounts read for {j + 1}/{N} stocks")
    return out


def membership(T: dict[str, np.ndarray], listed: np.ndarray, size: int = LIST_SIZE) -> np.ndarray:
    """The ``size`` largest by market value each month among the listed stocks with a market value and at least
    MIN_TRAITS traits."""
    have = sum(np.isfinite(T[k]).astype(int) for k in TRAITS)
    ok = listed & np.isfinite(T["mcap"]) & (have >= MIN_TRAITS)
    mem = np.zeros(ok.shape, dtype=bool)
    for m in range(ok.shape[0]):
        idx = np.flatnonzero(ok[m])
        if len(idx):
            mem[m, idx[np.argsort(-T["mcap"][m, idx], kind="stable")[:size]]] = True
    return mem


def score(T: dict[str, np.ndarray], mem: np.ndarray, traits=TRAITS, need: int = MIN_TRAITS) -> np.ndarray:
    """The mean of each member's cross-sectional ranks (in [-0.5, 0.5]) on the traits it has; NaN off the list."""
    S = np.full(mem.shape, np.nan)
    for m in range(mem.shape[0]):
        if not mem[m].any():
            continue
        ranks = np.vstack([rank_cs(T[k][m], mem[m]) for k in traits])
        cnt = np.isfinite(ranks).sum(0)
        ok = mem[m] & (cnt >= need)
        if ok.any():
            S[m, ok] = np.nanmean(ranks[:, ok], axis=0)
    return S


def cap_weighted(R: np.ndarray, mem: np.ndarray, mcap: np.ndarray, keep: np.ndarray, cost_bps: float) -> dict:
    """Holdings ``keep`` weighted by market value against the members weighted by market value (drift ignored,
    as in the harness)."""
    M, N = R.shape
    gross, bench, turn = np.full(M, np.nan), np.full(M, np.nan), np.zeros(M)
    prev = np.zeros(N)
    for m in range(M):
        r = np.nan_to_num(R[m], nan=0.0)
        if not mem[m].any():
            prev = np.zeros(N)
            continue
        wb = np.where(mem[m], np.nan_to_num(mcap[m]), 0.0)
        bench[m] = wb @ r / wb.sum()
        h = keep[m] & mem[m]
        if h.sum() < 3:
            prev = np.zeros(N)
            continue
        w = np.where(h, np.nan_to_num(mcap[m]), 0.0)
        w = w / w.sum()
        gross[m] = w @ r
        turn[m] = np.abs(w - prev).sum()
        prev = w
    return {"gross": gross, "bench": bench, "turnover": turn, "net": gross - turn * cost_bps / 1e4}


def bottom_third(S: np.ndarray, mem: np.ndarray) -> np.ndarray:
    """Each month's lowest third of the scored members."""
    out = np.zeros(S.shape, dtype=bool)
    for m in range(S.shape[0]):
        idx = np.flatnonzero(mem[m] & np.isfinite(S[m]))
        if len(idx) >= 3:
            k = round(len(idx) / 3)
            out[m, idx[np.argsort(S[m, idx], kind="stable")[:k]]] = True
    return out


def growth(r: np.ndarray) -> dict:
    """Geometric annual return, $1,000 grown and the worst fall of monthly or weekly returns ``r`` (NaN skipped)."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {"n": 0}
    curve = np.cumprod(1 + r)
    return {"n": len(r), "grown_1000": float(1000 * curve[-1]),
            "worst_fall": float((curve / np.maximum.accumulate(curve) - 1).min())}


def _annual(g: dict, per_year: int) -> float:
    return float((g["grown_1000"] / 1000) ** (per_year / g["n"]) - 1) if g.get("n") else float("nan")


def _ref_returns(panel, sym: str, df: pd.DataFrame | None = None) -> np.ndarray:
    """A reference fund's return over each rebalance's entry-to-exit window on the panel's calendar."""
    if df is None:
        s = pd.Series(panel.book.adj[sym], index=panel.book.dates[sym])
    else:
        s = pd.Series(df["adjclose"].to_numpy(float), index=df["date"].to_numpy(dtype=str))
    s = s[~s.index.duplicated(keep="last")].reindex(panel.dates).ffill(limit=3).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return s[panel.exit] / s[panel.entry] - 1


def masks(panel) -> tuple[np.ndarray, np.ndarray]:
    mo = panel.months
    start = mo >= SPLIT["first_rebalance"]
    return (np.flatnonzero(start & (mo <= SPLIT["train_end"])), np.flatnonzero(mo >= SPLIT["test_start"]))


def evaluate_tilt(bank: Bank, *, rsp: pd.DataFrame | None, log=print, n_random: int | None = None,
                  tickers_=None) -> dict:
    from brain.research import run
    n_random = N_RANDOM if n_random is None else n_random
    names = tickers_ or tickers()
    have = bank.view()
    names = [t for t in names if have.latest(t, "series.price_daily") is not None]
    panel = build(bank, names)
    M = len(panel.entry)
    train, test = masks(panel)
    if log:
        log(f"T1a: {len(names)} stocks, rebalances {panel.months[0]}..{panel.months[-1]}; reading accounts")
    T = fundamentals(bank, panel, log=log)
    T.update(price_traits(panel.adj, panel.rebal))
    mem = membership(T, panel.listed)
    S = score(T, mem)
    kw = {"frac": FRAC, "cost_bps": COST_BPS, "select": "top_frac"}
    te = harness.evaluate(S[:M], panel.R, mem[:M], test, n_perm=N_PERM, seed=8, **kw)
    tr = harness.evaluate(S[:M], panel.R, mem[:M], train, n_perm=N_PERM, seed=7, **kw)
    if log:
        log(f"T1a: TEST net excess {te['mean_net_excess'] * 100:+.3f}%/mo, p {te['perm_p']:.4f}; running controls")
    ctl = run.controls(dataclasses.replace(panel, listed=mem), test, n_random)
    alone = {}
    mem_test = mem[:M][test]
    for k in TRAITS:
        Sk = score(T, mem, traits=(k,), need=1)
        e = harness.evaluate(Sk[:M], panel.R, mem[:M], test, n_perm=N_PERM_TRAIT, seed=9, **kw)
        alone[k] = {"mean_net_excess": e.get("mean_net_excess"), "perm_p": e.get("perm_p"),
                    "coverage": float((np.isfinite(T[k][:M][test]) & mem_test).sum() / mem_test.sum())}
    pf = harness.portfolio(S[:M], panel.R, mem[:M], FRAC, COST_BPS, "top_frac")
    tilt_net = pf["gross"] - pf["turnover"] * COST_BPS / 1e4
    keep = mem & ~bottom_third(S, mem)
    cw = cap_weighted(panel.R, mem[:M], T["mcap"][:M], keep[:M], COST_BPS)
    spy = _ref_returns(panel, "SPY")
    refs = {"SPY": spy, "RSP": _ref_returns(panel, "RSP", rsp) if rsp is not None else np.full(M, np.nan)}
    series_ = {"tilt": tilt_net, "list_ew": pf["bench"], "list_cw": cw["bench"], "without_junk": cw["net"], **refs}
    window = {k: growth(v[test]) for k, v in series_.items()}
    ann = {k: _annual(g, 12) for k, g in window.items()}
    cwx = cw["net"][test] - cw["bench"][test]
    cwx = cwx[np.isfinite(cwx)]
    sizes = mem[:M][test].sum(1)
    stat = te.get("mean_net_excess", float("nan"))
    p = te.get("perm_p", float("nan"))
    met = {"tickers": len(names), "panel_digest": panel.digest(),
           "first": str(panel.months[test[0]]), "last": str(panel.months[test[-1]]),
           "months": len(test), "list_size_min": int(sizes.min()), "list_size_median": float(np.median(sizes)),
           "trait_coverage": {k: alone[k]["coverage"] for k in TRAITS},
           "test": te, "train": tr, "traits_alone": alone,
           "without_junk": {"mean_net_excess": float(cwx.mean()) if len(cwx) else float("nan"),
                            "months_won": int((cwx > 0).sum()), "avg_turnover": float(np.nanmean(cw["turnover"][test]))},
           "window": window, "annual": ann,
           "survivorship": {"list_ew_minus_rsp": ann["list_ew"] - ann["RSP"], "list_cw_minus_spy": ann["list_cw"] - ann["SPY"]}}
    gate = {"stat": stat, "p": p, "pass": bool(stat > 0 and p < P_MAX)}
    return {"p": p, "gate": gate, "controls": {**ctl, "pass": bool(ctl["pass"])}, "metrics": met}


# ---------- T1b: buying after fear ----------
def weekly(spy: pd.DataFrame, vix: pd.DataFrame, irx: pd.DataFrame) -> dict:
    """Each week's decision (the VIX's last session of an ISO week): a spike, the after-fear state and its weight,
    and the next week's SPY and bill returns from the next session's close to the session after the next decision."""
    vd = vix["date"].to_numpy(dtype=str)
    vc = vix["close"].to_numpy(float)
    iso = [date.fromisoformat(x).isocalendar()[:2] for x in vd]
    last = np.array([i for i in range(len(vd)) if i == len(vd) - 1 or iso[i + 1] != iso[i]], dtype=int)
    cs = np.concatenate([[0.0], np.cumsum(vc)])
    spike = np.zeros(len(last), dtype=bool)
    for k, i in enumerate(last):
        if i >= FEAR_LOOKBACK:
            spike[k] = vc[i] >= FEAR_RATIO * (cs[i] - cs[i - FEAR_LOOKBACK]) / FEAR_LOOKBACK
    after = np.array([spike[max(0, k - FEAR_WEEKS + 1): k + 1].any() for k in range(len(last))])
    w = np.where(after, W_FEAR, W_CALM)
    sd = spy["date"].to_numpy(dtype=str)
    sa = spy["adjclose"].to_numpy(float)
    idd = irx["date"].to_numpy(dtype=str)
    ic = irx["close"].to_numpy(float)
    dates = vd[last]
    pos = np.searchsorted(sd, dates, side="right")      # the first SPY session after each decision
    ipos = np.searchsorted(idd, dates, side="right") - 1  # the last bill rate on or before it
    r, rf = np.full(len(last), np.nan), np.full(len(last), np.nan)
    for k in range(len(last) - 1):
        e0, e1 = pos[k], pos[k + 1]
        if dates[k] < sd[0] or e0 >= len(sd) or e1 >= len(sd) or e1 <= e0 or ipos[k] < 0:
            continue
        r[k] = sa[e1] / sa[e0] - 1
        days = (date.fromisoformat(sd[e1]) - date.fromisoformat(sd[e0])).days
        rf[k] = (1 + ic[ipos[k]] / 100) ** (days / 365) - 1
    return {"date": dates, "spike": spike, "after": after, "w": w, "r": r, "rf": rf}


def _excess(w: np.ndarray, ex: np.ndarray, wbar: float) -> np.ndarray:
    dw = np.abs(np.diff(w, prepend=w[0]))
    return (w - wbar) * ex - dw * COST_BPS / 1e4


def shift_p(w: np.ndarray, ex: np.ndarray, n: int, rng) -> tuple[float, float]:
    """(mean excess over the constant mix, one-sided p from circular shifts of the weights by at least MIN_SHIFT)."""
    wbar = float(w.mean())
    obs = float(_excess(w, ex, wbar).mean())
    L = len(w)
    if L <= 2 * MIN_SHIFT:
        return obs, float("nan")
    offs = rng.integers(MIN_SHIFT, L - MIN_SHIFT + 1, n)
    null = np.array([_excess(np.roll(w, int(o)), ex, wbar).mean() for o in offs])
    return obs, float((1 + np.sum(null >= obs - 1e-15)) / (n + 1))


def _states(spike: np.ndarray) -> np.ndarray:
    return np.array([spike[max(0, k - FEAR_WEEKS + 1): k + 1].any() for k in range(len(spike))])


def window_stats(wk: dict, lo: str, hi: str | None) -> dict:
    sel = (wk["date"] >= lo) & ((wk["date"] <= hi) if hi else True) & np.isfinite(wk["r"]) & np.isfinite(wk["rf"])
    idx = np.flatnonzero(sel)
    w, r, rf, after = wk["w"][idx], wk["r"][idx], wk["rf"][idx], wk["after"][idx]
    ex = r - rf
    wbar = float(w.mean())
    strat = w * r + (1 - w) * rf - np.abs(np.diff(w, prepend=w[0])) * COST_BPS / 1e4
    mix = wbar * r + (1 - wbar) * rf
    episodes = int(np.sum(after & ~np.concatenate([[False], after[:-1]])))
    g = {"timed": growth(strat), "constant_mix": growth(mix), "spy": growth(r), "bills": growth(rf)}
    return {"weeks": len(idx), "first": str(wk["date"][idx[0]]), "last": str(wk["date"][idx[-1]]),
            "spike_weeks": int(wk["spike"][idx].sum()), "episodes": episodes, "share_after": float(after.mean()),
            "avg_weight": wbar,
            "spy_excess_after": float(ex[after].mean()) if after.any() else float("nan"),
            "spy_excess_calm": float(ex[~after].mean()) if (~after).any() else float("nan"),
            "mean_excess": float(_excess(w, ex, wbar).mean()), "window": g,
            "annual": {k: _annual(v, 52) for k, v in g.items()}, "_idx": idx}


def evaluate_timing(wk: dict, rng, n_shift: int | None = None, n_random: int | None = None) -> dict:
    n_shift = N_SHIFT if n_shift is None else n_shift
    n_random = N_RANDOM if n_random is None else n_random
    test = window_stats(wk, *T1B_TEST)
    seen = window_stats(wk, T1B_SEEN, None)
    idx = test.pop("_idx")
    seen_idx = seen.pop("_idx")
    w, ex = wk["w"][idx], (wk["r"] - wk["rf"])[idx]
    stat, p = shift_p(w, ex, n_shift, rng)
    test["p"] = p
    seen["mean_excess"], seen["p"] = shift_p(wk["w"][seen_idx], (wk["r"] - wk["rf"])[seen_idx], n_shift, rng)
    oracle = np.where(ex > 0, W_FEAR, W_CALM)
    _, oracle_p = shift_p(oracle, ex, N_ORACLE_SHIFT, rng)
    rate = float(wk["spike"][idx].mean())
    ps = []
    for _ in range(n_random):
        fake = np.where(_states(rng.random(len(idx)) < rate), W_FEAR, W_CALM)
        ps.append(shift_p(fake, ex, N_RANDOM_SHIFT, rng)[1])
    fpr = float(np.mean(np.array(ps) < P_MAX))
    ctl = {"oracle_p": oracle_p, "random_fpr": fpr, "n_random": n_random,
           "pass": bool(oracle_p < ORACLE_P_MAX and FPR[0] <= fpr <= FPR[1])}
    gate = {"stat": stat, "p": p, "pass": bool(stat > 0 and p < P_MAX)}
    return {"p": p, "gate": gate, "controls": ctl, "metrics": {"test": test, "seen": seen}}


def _digest(df: pd.DataFrame, col: str) -> dict:
    h = hashlib.sha256()
    h.update(df["date"].to_numpy(dtype=str).astype("U10").tobytes())
    h.update(np.round(df[col].to_numpy(float), 6).tobytes())
    return {"sha": h.hexdigest()[:16], "first": str(df["date"].iloc[0]), "last": str(df["date"].iloc[-1]),
            "rows": len(df)}


# ---------- running ----------
def test(bank: Bank, *, trial: bool = False, log=print) -> list[dict]:
    """Run every declared test here that has no result yet, once."""
    from brain.research import run
    from brain.sources import prices
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
        if not trial:
            if dirty := tree_dirty():
                raise Refused("the working tree has changes under brain/, knowledge/ or universe/: " + "; ".join(dirty))
            if why := lineage_problem(decl.get("commit")):
                raise Refused(f"{tid}: {why}")
            if s["code_sha"] != code_sha():
                raise Refused(f"{tid}: the code is not the declared code")
            if tid == "T1a" and s["universes"] != universe_shas():
                raise Refused(f"{tid}: a universe file changed since the declaration; declare a new id")
            if not committed_declaration(tid, decl["spec_sha"]):
                raise Refused(f"{tid}: commit the ledger with its declaration before running it")
        todo.append(decl)
    if not todo:
        raise Refused("every step-2 test already has a result; a rerun needs research withdraw and a new id")
    out = []
    for decl in todo:
        tid = decl["id"]
        rng = np.random.default_rng(SEED)
        if tid == "T1a":
            rsp = prices.fetch_daily("RSP", "2009-01-01")[0]
            r = evaluate_tilt(bank, rsp=rsp, log=log)
            r["data"] = {"RSP": _digest(rsp, "adjclose")}
        else:
            spy, vix, irx = (prices.fetch_daily(s_, T1B_START)[0] for s_ in ("SPY", "^VIX", "^IRX"))
            r = evaluate_timing(weekly(spy, vix, irx), rng)
            r["data"] = {"SPY": _digest(spy, "adjclose"), "^VIX": _digest(vix, "close"), "^IRX": _digest(irx, "close")}
        rec = {"op": "result", "kind": KIND, "id": tid, "spec_sha": decl["spec_sha"], "gate_type": ASKS[tid], **r,
               "at": now_iso(), **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            g = r["gate"]
            log(f"{tid}: statistic {g['stat'] * 100:+.4f}% per {'month' if tid == 'T1a' else 'week'}, p {g['p']:.4f}; "
                f"controls {'passed' if r['controls']['pass'] else 'FAILED'}; gate {'met' if g['pass'] else 'not met'}")
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
    p = sub.add_parser("tilttest", help="declare or run T1a-T1b: the quality tilt and buying after fear "
                                        "(docs/TILT.md)")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.set_defaults(fn=_cmd)
