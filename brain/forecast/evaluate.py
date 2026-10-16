"""Walk-forward forecast runs and their metrics (docs/FORECAST.md, sections 2 and 7).

A run, in order:

1. **Grid.** Forecast dates counted back from the anchor A in steps of h sessions, so that the latest date's
   window ends at A and each window starts where the next one ends (``grid``).
2. **Features.** One feature set over the whole grid since 2011 (``features.feature_set``), and one price
   panel for the range (``features.prices``), both at the run's pinned ``prices_asof``.
3. **Walk forward.** At each retrain date the model is fitted on the samples whose stored exit is on or
   before that date (``learn.fit``, ``learn.fit_range``) and forecasts every date up to the next retrain.
4. **Metrics.** Each one is computed per date across the stocks forecast that date, then over dates.
   Stocks move together within a week, so a date is the unit of evidence: every interval and every
   p-value here is date-level.

The gate (F1b, F1c) is a one-sided sign-flip test on the series of per-date size-neutral ICs. Two controls
check that test on the run's own data: the oracle (a forecast that knew the outcome) must reach its smallest
possible p, and the placebo (F1's own forecasts with each date's sign flipped by a coin) must pass about as
often as the 5% level says.
"""
from __future__ import annotations

import math
import re
import statistics
import time

import numpy as np

from brain.forecast import learn

SINCE = learn.LEARN["since"]
TRIAL_LAST = "2018-12-31"     # a trial's anchor: the last SPY session on or before this (TRAIN years)
TRIAL_START = "2015-01-01"    # a trial's gated windows start here instead of 2019-01-01
SIGN_FLIP = {"exact_max": 16, "draws": 20000, "seed": 7}
PLACEBO = {"draws": 200, "seed": 7, "reject_at": 0.05, "ok": (0.01, 0.10)}
PERMUTATION = {"draws": 1000, "seed": 7}   # the within-date permutation diagnostic
MIN_STOCKS = 3                 # a date needs this many stocks for a rank statistic
GATE_P = 0.05
Z90 = statistics.NormalDist().inv_cdf(0.95)
NORMAL_Q = {learn.qkey(q): statistics.NormalDist().inv_cdf(q) for q in learn.RANGE["quantiles"]}
QKEYS = [learn.qkey(q) for q in learn.RANGE["quantiles"]]
QLEVELS = dict(zip(QKEYS, learn.RANGE["quantiles"]))


def _features():
    from brain.forecast import features
    return features


def _think():
    from brain.forecast import think
    return think


# ---------- the grid and the anchor ----------
def grid(cal, anchor: str, h: int, since: str = SINCE) -> list[str]:
    """Forecast dates on SPY's calendar ``cal``, oldest first, dated on or after ``since``.

    d_k is the session h*k + 1 sessions before the anchor (k = 1, 2, ...). Its entry is the next session and
    its exit h sessions after that (the outcome rule), so d_1's exit is the anchor itself and each window
    begins at the close where the next one ends: the windows never overlap."""
    cal = np.asarray(cal).astype(str)
    a = int(np.searchsorted(cal, str(anchor)[:10]))
    if a >= len(cal) or cal[a] != str(anchor)[:10]:
        raise ValueError(f"anchor {anchor} is not a SPY session in the book")
    idx = np.arange(a - 1 - int(h), -1, -int(h))[::-1]
    return [str(cal[i]) for i in idx if cal[i] >= since[:10]]


def exit_of(cal, d: str, h: int) -> str | None:
    """The exit session of a window opened after the close of d (h sessions after the next session)."""
    cal = np.asarray(cal).astype(str)
    i = int(np.searchsorted(cal, d[:10], side="right")) + int(h)
    return str(cal[i]) if i < len(cal) else None


def trial_anchor(cal) -> str:
    """The last SPY session on or before 2018-12-31."""
    cal = np.asarray(cal).astype(str)
    i = int(np.searchsorted(cal, TRIAL_LAST, side="right")) - 1
    if i < 0:
        raise ValueError("no SPY session on or before 2018-12-31")
    return str(cal[i])


def find_anchor(book, tickers, ref: str = "SPY") -> str:
    """The last SPY session A for which every stock in ``tickers`` that was listed 5 sessions earlier (a
    close dated within 4 days of that session and at least 252 bars up to it) has a bar dated exactly A,
    with a positive close, in ``book``."""
    from brain.timeutil import days_between
    cal = np.asarray(book.dates[ref]).astype(str)
    own = {t: (np.asarray(book.dates[t]).astype(str), np.asarray(book.adj[t], dtype=float))
           for t in (str(x).upper() for x in tickers) if book.has(t) and t != ref}
    for a in range(len(cal) - 1, 4, -1):
        A, back = cal[a], cal[a - 5]
        ok = True
        for d, adj in own.values():
            i = int(np.searchsorted(d, back, side="right")) - 1
            if i < 0 or i + 1 < 252 or days_between(d[i], back) > 4:
                continue  # not listed 5 sessions earlier
            j = int(np.searchsorted(d, A))
            if j >= len(d) or d[j] != A or not (np.isfinite(adj[j]) and adj[j] > 0):
                ok = False
                break
        if ok:
            return str(A)
    raise ValueError("no session where every listed stock has a bar")


def plan(test: dict, cal, anchor: str, *, trial: bool = False, recent_periods: int = 8) -> dict:
    """The forecast dates and retrain cutoffs of one declared test.

    - ``periods``: the last that many grid dates (F1a, F1d), one model at the first of them.
    - ``start``: grid dates from ``start`` (TRIAL_START in a trial). An ``end`` other than ``<anchor>`` is
      F1a's first date, so F1b stops ``recent_periods`` dates short of the anchor. ``retrain`` names how
      often the model is refitted ("every 26 periods"), each time with cutoff = that forecast date."""
    h = int(test["horizon"])
    g = grid(cal, anchor, h)
    if "periods" in test:
        dates, every = g[-int(test["periods"]):], None
        start = dates[0] if dates else None
    else:
        start = TRIAL_START if trial else str(test["start"])
        pool = g if str(test.get("end", "<anchor>")) == "<anchor>" else g[:-int(recent_periods)]
        dates = [d for d in pool if d >= start]
        m = re.search(r"every (\d+) periods", str(test.get("retrain", "")))
        every = int(m.group(1)) if m else None
    retrain = dates[::every] if every else dates[:1]
    return {"h": h, "anchor": anchor, "grid": g, "dates": dates, "retrain": retrain, "start": start,
            "first": dates[0] if dates else None, "last": dates[-1] if dates else None,
            "last_exit": exit_of(cal, dates[-1], h) if dates else None, "n_dates": len(dates)}


# ---------- rank statistics on one date ----------
def rank(x) -> np.ndarray:
    """Ranks 1..n, ties sharing their average rank."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1] if n else np.zeros(0, dtype=int)
    ends = np.r_[starts[1:], n] if n else np.zeros(0, dtype=int)
    r = np.empty(n)
    r[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return r


def _unit(v: np.ndarray) -> np.ndarray | None:
    """``v`` centred and scaled to length 1; None when it has no spread."""
    c = v - v.mean()
    s = float(np.sqrt(c @ c))
    return c / s if s > 1e-12 * max(1.0, float(np.abs(v).max(initial=0.0))) * math.sqrt(len(v)) else None


def spearman(a, b) -> float:
    """Spearman rank correlation; 0 when either side has no spread (a flat forecast ranks nothing)."""
    ua, ub = _unit(rank(a)), _unit(rank(b))
    return 0.0 if ua is None or ub is None else float(ua @ ub)


def neutral_rank(p, size) -> np.ndarray | None:
    """p's rank less its least-squares fit on size's rank (with an intercept), centred and of length 1;
    None when nothing is left (a forecast that is a function of size alone)."""
    rp, rs = rank(p), rank(size)
    rp, rs = rp - rp.mean(), rs - rs.mean()
    ss = float(rs @ rs)
    resid = rp - (float(rs @ rp) / ss) * rs if ss > 0 else rp
    tol = 1e-9 * max(1.0, float(np.sqrt(rp @ rp)))
    if float(np.sqrt(resid @ resid)) <= tol:
        return None
    return _unit(resid)


def size_neutral_ic(p, e, size) -> float:
    """The IC after p's rank is replaced by its residual on size's rank: what p says beyond size."""
    a, b = neutral_rank(p, size), _unit(rank(e))
    return 0.0 if a is None or b is None else float(a @ b)


# ---------- date-level statistics ----------
def sign_flip(series, exact_max: int = SIGN_FLIP["exact_max"], draws: int = SIGN_FLIP["draws"],
              seed: int = SIGN_FLIP["seed"]) -> dict:
    """One-sided sign-flip test that the mean of ``series`` (or of each row of a 2-D batch) is above 0.

    Under the null each date's value is as likely to have the other sign. With K <= ``exact_max`` dates
    every one of the 2^K sign patterns is counted (p = share of patterns whose mean is at least the
    observed); else ``draws`` seeded draws, p = (1 + count) / (1 + draws). ``min_p`` is the smallest p the
    test can give: 1 / 2^K, or 1 / (1 + draws)."""
    x = np.atleast_2d(np.asarray(series, dtype=float))
    K = x.shape[1]
    if K == 0:
        out = {"p": np.full(len(x), np.nan), "min_p": float("nan"), "exact": True, "k": 0}
    else:
        if K <= exact_max:
            S = (((np.arange(2 ** K)[:, None] >> np.arange(K)) & 1) * 2 - 1).astype(float)
            exact = True
        else:
            S = np.random.default_rng(seed).integers(0, 2, (draws, K)).astype(float) * 2 - 1
            exact = False
        obs = x.mean(axis=1)
        null = S @ x.T / K                                 # (patterns, batch)
        count = (null >= (obs - 1e-12 * (1 + np.abs(obs)))[None, :]).sum(axis=0)
        p = count / len(S) if exact else (1 + count) / (1 + draws)
        out = {"p": p, "min_p": float(1 / len(S) if exact else 1 / (1 + draws)), "exact": exact, "k": K}
    if np.ndim(series) == 1:
        out["p"] = float(out["p"][0])
    else:
        out["p"] = [float(v) for v in out["p"]]
    out["mean"] = float(np.mean(x)) if np.ndim(series) == 1 and K else None
    return out


def interval(values) -> dict:
    """Mean over dates with its date-level 90% interval: mean +- 1.645 sd / sqrt(K)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    K = len(v)
    if K == 0:
        return {"mean": None, "sd": None, "lo90": None, "hi90": None, "k": 0}
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if K > 1 else float("nan")
    half = Z90 * sd / math.sqrt(K) if K > 1 else float("nan")
    return {"mean": m, "sd": _num(sd), "lo90": _num(m - half), "hi90": _num(m + half), "k": K}


def _num(x):
    return float(x) if x is not None and np.isfinite(x) else None


def label(iv: dict, detects: float | None = None, what: str = "mean IC") -> str:
    """A descriptive retest's reading: inconclusive when its 90% interval spans 0, else positive or negative."""
    lo, hi = iv.get("lo90"), iv.get("hi90")
    word = "inconclusive" if lo is None or hi is None or lo <= 0 <= hi else ("positive" if lo > 0 else "negative")
    size = f"; it can detect {what} >= {detects:g} at 80% power" if detects else ""
    return f"retest (descriptive): {word}{size}"


def brier_parts(p, y, clim) -> dict:
    """One date's Brier score and its split: level (mean forecast against the share that beat SPY) plus
    within-date (how each stock's forecast differs from the date's mean, against how its outcome did). The
    same for climatology, the constant ``clim``. level + within = brier."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    pb, yb = float(p.mean()), float(y.mean())
    return {"brier": float(np.mean((p - y) ** 2)), "level": (pb - yb) ** 2,
            "within": float(np.mean(((p - pb) - (y - yb)) ** 2)),
            "clim": float(np.mean((clim - y) ** 2)), "clim_level": (clim - yb) ** 2, "clim_within": yb * (1 - yb)}


def calibration_table(p, y, groups: int = 10) -> list[dict]:
    """Forecasts sorted and cut into ``groups`` near-equal bins: mean forecast, realised frequency, n."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    if len(p) == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    out = []
    for part in np.array_split(order, min(groups, len(p))):
        out.append({"lo": float(p[part].min()), "hi": float(p[part].max()), "mean_p": float(p[part].mean()),
                    "freq": float(y[part].mean()), "n": len(part)})
    return out


def pinball(u, pred: dict, levels: dict = QLEVELS) -> np.ndarray:
    """Per row, the mean over quantiles of the pinball loss of predicted log moves against the realised ``u``."""
    tot = np.zeros(len(u))
    for k, q in levels.items():
        d = u - pred[k]
        tot += np.maximum(q * d, (q - 1) * d)
    return tot / len(levels)


def placement(p, e) -> float | None:
    """Share of stocks the forecast places on the right side of the date's median excess (a forecast above
    its own median calls "above"). Stocks exactly at either median are not counted."""
    mp, me = np.median(p), np.median(e)
    counted = (p != mp) & (e != me)
    if not counted.any():
        return None
    right = ((p > mp) & (e > me)) | ((p < mp) & (e < me))
    return float(right[counted].mean())


def spread(p, e, halves: bool = False) -> float:
    """Top minus bottom fifth (or half) of the date's stocks by forecast: their mean excess."""
    n = len(p)
    k = max(1, n // 2 if halves else n // 5)
    order = np.argsort(p, kind="mergesort")
    return float(e[order[-k:]].mean() - e[order[:k]].mean())


# ---------- scoring a set of forecasts ----------
def _groups(dates) -> list[tuple[str, np.ndarray]]:
    d = np.asarray(dates).astype(str)
    u, inv = np.unique(d, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    cuts = np.cumsum(np.bincount(inv, minlength=len(u)))[:-1]
    return list(zip(u.tolist(), np.split(order, cuts)))


def score(S: dict, h: int, *, gated: bool, halves: bool = False, cal_groups: int = 10, detects: float | None = None,
          perm: dict = PERMUTATION) -> dict:
    """Every metric of section 7 for the forecasts ``S`` (arrays per sample: date, p, y, excess, size, mom,
    clim, u, sig, naive, zq). ``gated`` adds the sign-flip p, the within-date permutation diagnostic and
    the date series the controls need; otherwise each mean gets its date-level 90% interval and a label."""
    per = {k: [] for k in ("date", "n", "ic", "sn_ic", "size_ic", "size_sn_ic", "mom_ic", "mom_sn_ic", "spread",
                           "placement", "cover68", "cover90", "width68", "width90", "pinball", "pinball_naive",
                           "brier_gain", "within_gain")}
    parts, vecs, oracle = [], [], []
    sqh = math.sqrt(int(h))
    n_range = 0
    has_size = np.isfinite(S["size"]).any()
    has_mom = np.isfinite(S["mom"]).any()
    for d, idx in _groups(S["date"]):
        p, e, y = S["p"][idx], S["excess"][idx], S["y"][idx]
        size = np.nan_to_num(S["size"][idx])
        clim = float(S["clim"][idx][0])
        bp = brier_parts(p, y, clim)
        parts.append(bp)
        per["date"].append(d)
        per["n"].append(len(idx))
        per["brier_gain"].append(bp["clim"] - bp["brier"])
        per["within_gain"].append(bp["clim_within"] - bp["within"])
        ranked = len(idx) >= MIN_STOCKS
        nan = float("nan")
        per["ic"].append(spearman(p, e) if ranked else nan)
        per["sn_ic"].append(size_neutral_ic(p, e, size) if ranked and has_size else nan)
        per["size_ic"].append(spearman(-size, e) if ranked and has_size else nan)
        per["size_sn_ic"].append(size_neutral_ic(-size, e, size) if ranked and has_size else nan)
        mom = np.nan_to_num(S["mom"][idx])
        per["mom_ic"].append(spearman(mom, e) if ranked and has_mom else nan)
        per["mom_sn_ic"].append(size_neutral_ic(mom, e, size) if ranked and has_mom and has_size else nan)
        per["spread"].append(spread(p, e, halves) if ranked else nan)
        pl = placement(p, e) if ranked else None
        per["placement"].append(nan if pl is None else pl)
        if gated and ranked and has_size:
            a, b = neutral_rank(p, size), _unit(rank(e))
            if a is not None and b is not None:
                vecs.append((a, b))
            oracle.append(size_neutral_ic(rank(e), e, size))
        # the range: realised log move from the close of d against the band's log quantiles
        u, sg, nv = S["u"][idx], S["sig"][idx], S["naive"][idx]
        ok = np.isfinite(u) & np.isfinite(sg) & (sg > 0) & np.isfinite(nv) & (nv > 0)
        if ok.any():
            n_range += int(ok.sum())
            zq = S["zq"][idx][ok]
            s = sg[ok] * sqh
            lq = {k: zq[:, c] * s for c, k in enumerate(QKEYS)}
            uu = u[ok]
            per["cover68"].append(float(((uu >= lq["q16"]) & (uu <= lq["q84"])).mean()))
            per["cover90"].append(float(((uu >= lq["q05"]) & (uu <= lq["q95"])).mean()))
            per["width68"].append(float(np.mean(np.exp(lq["q84"]) - np.exp(lq["q16"]))))
            per["width90"].append(float(np.mean(np.exp(lq["q95"]) - np.exp(lq["q05"]))))
            per["pinball"].append(float(pinball(uu, lq).mean()))
            naive = {k: NORMAL_Q[k] * nv[ok] * sqh for k in QKEYS}
            per["pinball_naive"].append(float(pinball(uu, naive).mean()))
        else:
            for k in ("cover68", "cover90", "width68", "width90", "pinball", "pinball_naive"):
                per[k].append(nan)

    def mean(k):
        v = [x for x in per[k] if np.isfinite(x)]
        return float(np.mean(v)) if v else None

    B = {k: float(np.mean([bp[k] for bp in parts])) if parts else None for k in parts[0]} if parts else {}
    brier = {**B, "skill": (1 - B["brier"] / B["clim"]) if B and B["clim"] else None,
             "within_skill": (1 - B["within"] / B["clim_within"]) if B and B["clim_within"] else None,
             "base_rates": sorted({float(c) for c in np.unique(S["clim"])})}
    pb, pn = mean("pinball"), mean("pinball_naive")
    m = {
        "n_dates": len(per["date"]), "n_samples": len(S["p"]),
        "ic": interval(per["ic"]), "ic_size_neutral": interval(per["sn_ic"]),
        "baselines": {"size": {"ic": interval(per["size_ic"]), "ic_size_neutral": interval(per["size_sn_ic"]),
                               "score": "-log_mktcap"},
                      "momentum": {"ic": interval(per["mom_ic"]), "ic_size_neutral": interval(per["mom_sn_ic"]),
                                   "score": "mom_12_1"}},
        "spread": {**interval(per["spread"]), "what": "top half minus bottom half" if halves else
                   "top fifth minus bottom fifth"},
        "brier": brier, "brier_gain": interval(per["brier_gain"]), "within_gain": interval(per["within_gain"]),
        "placement": interval(per["placement"]),
        "calibration": calibration_table(S["p"], S["y"], cal_groups),
        "range": {"cover68": interval(per["cover68"]), "cover90": interval(per["cover90"]),
                  "width68": mean("width68"), "width90": mean("width90"), "pinball": pb, "pinball_naive": pn,
                  "pinball_skill": (1 - pb / pn) if pb is not None and pn else None, "n_samples": n_range},
        "per_date": {k: [_r6(x) for x in per[k]] if k != "date" else per[k] for k in PER_DATE_KEPT},
    }
    sn = [x for x in per["sn_ic"] if np.isfinite(x)]
    if gated:
        sf = sign_flip(sn)
        m["_sn"] = sn                       # full precision, for the controls
        m["ic_size_neutral"].update(p=sf["p"], min_p=sf["min_p"], exact=sf["exact"],
                                    within_date_perm_p=_within_date_perm(vecs, perm))
        m["_oracle"] = oracle
    else:
        m["label"] = label(m["ic_size_neutral"], detects)
    return m


PER_DATE_KEPT = ("date", "n", "ic", "sn_ic", "spread", "placement", "brier_gain", "within_gain", "cover68", "cover90")


def _r6(x):
    """A per-date value as the ledger keeps it: 6 decimals, None when undefined."""
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def _within_date_perm(vecs: list[tuple[np.ndarray, np.ndarray]], perm: dict = PERMUTATION) -> float | None:
    """Diagnostic only: each date's excess ranks shuffled across its stocks, the mean size-neutral IC
    recomputed. It ignores how stocks move together, so it is far too narrow to gate on."""
    if not vecs:
        return None
    rng = np.random.default_rng(perm["seed"])
    P = int(perm["draws"])
    null = np.zeros(P)
    obs = 0.0
    for a, b in vecs:
        obs += float(a @ b)
        null += rng.permuted(np.tile(b, (P, 1)), axis=1) @ a
    obs, null = obs / len(vecs), null / len(vecs)
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + P))


def gate_and_controls(m: dict, placebo: dict = PLACEBO) -> tuple[dict, dict]:
    """The declared gate (mean size-neutral IC > 0 with date-level p < 0.05) and its two controls:

    - oracle: p_beat = the realised excess rank. Its test must reach the smallest p the test can give.
    - placebo: F1's own forecasts with each date's sign flipped by a fair coin, ``draws`` times. 1 - p
      reverses every rank, so a flipped date's size-neutral IC is exactly minus its own; the gate's test
      must reject (p < 0.05) in 1%-10% of draws."""
    sn = np.asarray(m.pop("_sn", []), dtype=float)
    snm = m["ic_size_neutral"]
    gate = {"mean_size_neutral_ic": snm["mean"], "p": snm.get("p"), "k": len(sn),
            "pass": bool(snm["mean"] is not None and snm["mean"] > 0 and snm.get("p") is not None
                         and snm["p"] < GATE_P)}
    orc_series = np.asarray(m.pop("_oracle", []), dtype=float)
    orc = sign_flip(orc_series)
    o_ok = bool(orc["k"] > 0 and orc["p"] <= orc["min_p"] * (1 + 1e-9))
    if len(sn):
        flips = np.random.default_rng(placebo["seed"]).integers(0, 2, (placebo["draws"], len(sn))) * 2 - 1
        ps = np.asarray(sign_flip(flips * sn[None, :])["p"])
        rate = float(np.mean(ps < placebo["reject_at"]))
    else:
        rate = float("nan")
    lo, hi = placebo["ok"]
    p_ok = bool(np.isfinite(rate) and lo <= rate <= hi)
    ctl = {"oracle_p": orc["p"], "min_p": orc["min_p"],
           "oracle_mean_size_neutral_ic": float(orc_series.mean()) if len(orc_series) else None,
           "oracle_ok": o_ok, "placebo_reject_rate": _num(rate), "placebo_draws": int(placebo["draws"]),
           "placebo_ok": p_ok, "pass": bool(o_ok and p_ok)}
    return gate, ctl


# ---------- the walk-forward run ----------
class Data:
    """One feature set and its range inputs on one grid (anchor, h): what the tests on that grid share."""

    def __init__(self, fs, px, h: int):
        self.fs, self.px, self.h = fs, px, int(h)
        r, self.sig = learn.range_inputs(fs, px)
        with np.errstate(invalid="ignore"):
            self.u = np.log1p(r)            # the realised log move from the close of d to the close of the exit
        self.r = r
        self.naive = trailing_sd(fs, px)
        self.rows: dict[str, np.ndarray] = dict(_groups(fs.date)) if len(fs.date) else {}
        names = list(fs.names)
        n = len(fs.date)
        self.size = fs.X[:, names.index("log_mktcap")] if "log_mktcap" in names else np.full(n, np.nan)
        self.mom = fs.X[:, names.index("mom_12_1")] if "mom_12_1" in names else np.full(n, np.nan)


def trailing_sd(fs, px, n: int = 252) -> np.ndarray:
    """Per sample, the standard deviation of the stock's daily log returns over the 252 sessions to d: the
    naive normal band's volatility. NaN with fewer than 21 returns."""
    dates = np.asarray(px.dates).astype(str)
    adj = np.asarray(px.adj, dtype=float)
    col = {str(t).upper(): j for j, t in enumerate(px.tickers)}
    d = np.asarray(fs.date).astype(str)
    j = np.array([col.get(str(t).upper(), -1) for t in fs.ticker], dtype=int)
    i_d = np.searchsorted(dates, d, side="right") - 1
    out = np.full(len(d), np.nan)
    for i in np.unique(i_d):
        if i < 0:
            continue
        a = adj[max(0, i - n):i + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.diff(np.log(np.where(a > 0, a, np.nan)), axis=0)
        ok = np.isfinite(r)
        cnt = ok.sum(axis=0)
        mu = np.where(ok, r, 0.0).sum(axis=0) / np.maximum(cnt, 1)
        var = (np.where(ok, r - mu, 0.0) ** 2).sum(axis=0) / np.maximum(cnt - 1, 1)
        sd = np.where(cnt >= learn.MIN_SIGMA_RETURNS, np.sqrt(var), np.nan)
        rows = np.flatnonzero((i_d == i) & (j >= 0))
        out[rows] = sd[j[rows]]
    return out


def prepare(bank, tickers, cal, anchor: str, h: int, prices_asof: str, *, trial: bool = False, log=print,
            cache: dict | None = None) -> Data:
    """The feature set over the whole grid since 2011 and the price panel, at ``prices_asof``. In a trial the
    grid's last exit must be on or before 2018-12-31, so nothing later is ever computed."""
    g = grid(cal, anchor, h)
    if not g:
        raise ValueError(f"no grid dates before the anchor {anchor}")
    if trial and (exit_of(cal, g[-1], h) or "9999") > TRIAL_LAST:
        raise SystemExit(f"trial grid would end after {TRIAL_LAST}: refused (development data rule)")
    key = (str(anchor), int(h), str(prices_asof))
    if cache is not None and key in cache:
        return cache[key]
    F = _features()
    t0 = time.monotonic()
    pkey = ("prices", str(prices_asof))
    px = cache.get(pkey) if cache is not None else None
    if px is None:  # one price panel per prices_asof: the features and the range read the same one
        px = F.prices(bank, tickers, prices_asof)
        if cache is not None:
            cache[pkey] = px
    t1 = time.monotonic()
    fs = F.feature_set(bank, tickers, g, prices_asof, h, log=log, px=px)
    t2 = time.monotonic()
    data = Data(fs, px, h)
    if log:
        log(f"  h={h}: {len(fs.date)} samples on {len(g)} grid dates ({g[0]}..{g[-1]}); prices {t1 - t0:.0f}s, "
            f"features {t2 - t1:.0f}s{' (cached)' if not getattr(fs, 'seconds', 1) else ''}, range inputs {time.monotonic() - t2:.0f}s")
    data.seconds = {"prices": round(t1 - t0, 1), "features": round(t2 - t1, 1), "range_inputs": round(time.monotonic() - t2, 1)}
    if cache is not None:
        cache[key] = data
    return data


def _empty_samples() -> dict:
    return {k: [] for k in ("row", "date", "p", "clim", "zq")}


def walk(data: Data, dates: list[str], retrain: list[str], *, log=print, name: str = "") -> tuple[dict, list[dict], tuple]:
    """Forecasts on ``dates``: a model fitted at each retrain cutoff forecasts every date from it up to the
    next. Returns the samples (arrays: row, date, ticker, p, y, excess, size, mom, clim, u, sig, naive,
    zq), one record per model, and the last (model, range) fitted."""
    fs = data.fs
    acc = _empty_samples()
    models, last = [], (None, None)
    cuts = list(retrain)
    for j, c in enumerate(cuts):
        block = [d for d in dates if d >= c and (j + 1 == len(cuts) or d < cuts[j + 1])]
        t0 = time.monotonic()
        model = learn.fit(fs, c)
        rng = learn.fit_range(fs, c, r_total=data.r, sig=data.sig, h=data.h)
        sec = time.monotonic() - t0
        models.append({**model.card(), "cutoff": c, "n_calibration": model.n_calibration, "last_exit": model.last_exit,
                       "base_rate": model.base_rate, "platt": list(model.platt), "range": rng,
                       "forecast_dates": len(block), "seconds": round(sec, 1)})
        if log:
            log(f"  {name} model {j + 1}/{len(cuts)} through {c}: {model.rows} rows, base rate {model.base_rate:.3f}, "
                f"{sec:.0f}s; forecasts {len(block)} dates")
        zq = np.array([rng["q"][k] for k in QKEYS])
        for d in block:
            rows = data.rows.get(d)
            if rows is None or not len(rows):
                continue
            p = learn.predict(model, fs.X[rows])
            acc["row"].append(rows)
            acc["date"].append(np.full(len(rows), d, dtype="U10"))
            acc["p"].append(p)
            acc["clim"].append(np.full(len(rows), model.base_rate))
            acc["zq"].append(np.tile(zq, (len(rows), 1)))
        last = (model, rng)
    return _samples(data, acc), models, last


def _samples(data: Data, acc: dict) -> dict:
    fs = data.fs
    rows = np.concatenate(acc["row"]) if acc["row"] else np.zeros(0, dtype=int)
    S = {"row": rows, "date": np.concatenate(acc["date"]) if acc["date"] else np.zeros(0, dtype="U10"),
         "p": np.concatenate(acc["p"]) if acc["p"] else np.zeros(0),
         "clim": np.concatenate(acc["clim"]) if acc["clim"] else np.zeros(0),
         "zq": np.vstack(acc["zq"]) if acc["zq"] else np.zeros((0, len(QKEYS)))}
    S.update(ticker=np.asarray(fs.ticker)[rows].astype(str), y=np.asarray(fs.y, dtype=float)[rows],
             excess=np.asarray(fs.excess, dtype=float)[rows], size=data.size[rows], mom=data.mom[rows],
             u=data.u[rows], sig=data.sig[rows], naive=data.naive[rows])
    return S


def run(bank, test: dict, *, anchor: str, prices_asof: str, tickers, cal, trial: bool = False, log=print,
        cache: dict | None = None, recent_periods: int = 8, detects: float | None = None) -> dict:
    """One F1 test (F1a, F1b or F1c) from its plan to its metrics; gated tests also get their gate and
    controls. The pieces of a ledger result, without the bookkeeping."""
    t0 = time.monotonic()
    pl = plan(test, cal, anchor, trial=trial, recent_periods=recent_periods)
    if not pl["dates"]:
        raise SystemExit(f"{test['id']}: no forecast dates in its window")
    if trial and (pl["last_exit"] or "9999") > TRIAL_LAST:
        raise SystemExit(f"{test['id']}: trial window would end after {TRIAL_LAST}: refused")
    data = prepare(bank, tickers, cal, anchor, pl["h"], prices_asof, trial=trial, log=log, cache=cache)
    t1 = time.monotonic()
    S, models, _ = walk(data, pl["dates"], pl["retrain"], log=log, name=test["id"])
    t2 = time.monotonic()
    gated = test.get("gate") == "ic"
    m = score(S, pl["h"], gated=gated, detects=detects)
    out = {"metrics": m, "n_dates": m["n_dates"], "n_samples": m["n_samples"], "dropped": dict(data.fs.dropped),
           "model": models, "cache_key": data.fs.key, "prices_asof": prices_asof,
           "plan": {k: pl[k] for k in ("anchor", "start", "first", "last", "last_exit", "n_dates", "retrain")}}
    if gated:
        gate, ctl = gate_and_controls(m)
        out.update(p=gate["p"], gate=gate, controls=ctl)
    else:
        out["controls"] = {"pass": None, "note": "descriptive: no gate and no controls"}
    out["seconds"] = {**getattr(data, "seconds", {}), "fits": round(t2 - t1, 1),
                      "metrics": round(time.monotonic() - t2, 1), "total": round(time.monotonic() - t0, 1)}
    return out


def brier_difference(S: dict, p_other: np.ndarray) -> dict:
    """F1d's comparison: per date, the summed Brier score of F1 minus that of the other forecast (positive =
    the other was closer), the mean over dates with its date-level 90% interval, and the exact one-sided
    sign-flip p that the mean is above 0."""
    y, p1 = S["y"], S["p"]
    per, ds = [], []
    for d, idx in _groups(S["date"]):
        per.append(float(np.sum((p1[idx] - y[idx]) ** 2) - np.sum((p_other[idx] - y[idx]) ** 2)))
        ds.append(d)
    sf = sign_flip(per)                     # 8 dates: all 2^8 = 256 sign patterns
    iv = interval(per)
    return {**iv, "p": sf["p"], "min_p": sf["min_p"], "exact": sf["exact"], "per_date": per, "dates": ds,
            "label": label(iv).replace("retest (descriptive): ", "retest (descriptive): LLM vs F1 ")}


def run_llm(bank, test: dict, *, anchor: str, prices_asof: str, tickers, cal, trial: bool = False, client=None,
            log=print, cache: dict | None = None, think=None) -> dict:
    """F1d: F1a's model on F1a's dates for the test's stocks, and the local model's probability for each,
    given F1's forecast and a point-in-time snapshot. A failed answer counts as F1's and is counted."""
    from brain.metrics import PriceBook
    think = think or _think()
    t0 = time.monotonic()
    pl = plan(test, cal, anchor, trial=trial)
    if trial and (pl["last_exit"] or "9999") > TRIAL_LAST:
        raise SystemExit(f"{test['id']}: trial window would end after {TRIAL_LAST}: refused")
    data = prepare(bank, tickers, cal, anchor, pl["h"], prices_asof, trial=trial, log=log, cache=cache)
    fs = data.fs
    model = learn.fit(fs, pl["retrain"][0])
    rng = learn.fit_range(fs, pl["retrain"][0], r_total=data.r, sig=data.sig, h=data.h)
    want = [str(t).upper() for t in test["tickers"]]
    book = PriceBook(bank, [*want, "SPY"], prices_asof)
    zq = np.array([rng["q"][k] for k in QKEYS])
    acc = _empty_samples()
    p_llm, replies, failed, missing = [], [], 0, []
    tick = np.asarray(fs.ticker).astype(str)
    for d in pl["dates"]:
        rows = data.rows.get(d, np.zeros(0, dtype=int))
        at = {tick[r].upper(): int(r) for r in rows}
        for t in want:
            r = at.get(t)
            if r is None:
                missing.append([d, t])
                continue
            fc = learn.forecast(model, rng, ticker=t, d=d, h=data.h, close=float(fs.close[r]), sigma_d=float(data.sig[r]),
                                x_row=fs.X[r], entry=str(fs.entry[r]), exit=str(fs.exit[r]))
            res = think.llm_forecast(bank, book, t, d, fc, client=client)
            ok = bool(res.get("ok")) and isinstance(res.get("p_beat"), (int, float)) and math.isfinite(res["p_beat"])
            if not ok:
                failed += 1
            p_llm.append(float(res["p_beat"]) if ok else fc["p_beat"])
            replies.append({"date": d, "ticker": t, "ok": ok, "p_f1": fc["p_beat"], "p_beat": res.get("p_beat"),
                            "reason": res.get("reason"), "raw": res.get("raw"), "model": res.get("model"),
                            "seconds": res.get("seconds"), "error": res.get("error")})
            if log:
                log(f"  {test['id']} {d} {t}: F1 {fc['p_beat']:.3f}, model "
                    + (f"{float(res['p_beat']):.3f}" if ok else f"failed ({res.get('error')})"))
            acc["row"].append(np.array([r]))
            acc["date"].append(np.array([d], dtype="U10"))
            acc["p"].append(np.array([fc["p_beat"]]))
            acc["clim"].append(np.array([model.base_rate]))
            acc["zq"].append(zq[None, :])
    S = _samples(data, acc)
    pl_arr = np.asarray(p_llm, dtype=float)
    m_f1 = score(S, data.h, gated=False, halves=True, cal_groups=5)
    m_llm = score({**S, "p": pl_arr}, data.h, gated=False, halves=True, cal_groups=5)
    cmp = brier_difference(S, pl_arr)
    return {"metrics": {"f1": m_f1, "llm": m_llm, "comparison": cmp, "llm_failed": failed,
                        "llm_asked": len(replies), "missing": missing, "label": cmp["label"]},
            "n_dates": m_f1["n_dates"], "n_samples": m_f1["n_samples"], "dropped": dict(fs.dropped),
            "model": [{**model.card(), "cutoff": pl["retrain"][0], "base_rate": model.base_rate, "range": rng}],
            "cache_key": fs.key, "prices_asof": prices_asof, "replies": replies,
            "plan": {k: pl[k] for k in ("anchor", "start", "first", "last", "last_exit", "n_dates", "retrain")},
            "controls": {"pass": None, "note": "descriptive: no gate and no controls"},
            "seconds": {"total": round(time.monotonic() - t0, 1)}}
