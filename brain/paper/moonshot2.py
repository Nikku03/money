"""The time machine, second version (test P4): dynamic exits, a sturdier learner, and a post-mortem of
every trade, from every direction, fed into the next one (docs/PAPER.md).

Paper only, a simulation of the past. Same clock, universe, tradability check and costs as P3b
(brain/paper/moonshot.py). What changes:

Exits (``EXITS``, fixed in advance)
    E0 fixed            +20% / -10% / 10 sessions (P3b's rule)
    E1 scaled           target 2.5 and stop 1.25 times the stock's own 10-session volatility (clipped
                        to 8-60% and 4-30%), 10 sessions
    E2 trailing         no target: a stop at -10%, then trailing 2.5 daily volatilities (5-25%) below the
                        highest close, up to 15 sessions
    E3 breakeven        -10% stop; once a close is 10% up, the stop moves to the entry; target +30%;
                        15 sessions
    E4 follow-through   P3b's rule, but out at the 3rd close unless it is 3% up
    Stops move only after a close, so no day uses its own later prices. Every candidate is simulated
    under every exit, which is what lets hindsight teach the next choice.

Choosing the exit, by experience (``ExitChooser``)
    Conditions = market fear (VIX above or below its last year's median) x the stock's volatility
    tercile that week: 6 kinds of trade. For each, every exit's average net over the trader's own top
    10 candidates in each past week whose trades have closed, shrunk toward that exit's overall
    average (30 trades' worth). The best one is used. Until 30 weeks are known, E0.

Learning (``Ensemble``)
    The expected net return under each exit, by ridge regression on winsorized outcomes (+-40%),
    older weeks fading (half-life 52 weeks), bagged 8 times over weeks. A stock's score is the mean
    of the 8 predictions for the exit its conditions call for, less half their spread: what the models
    agree on. It buys up to 3, only those whose score beats 0 (costs are already in the outcomes).

Memory (``mem_*`` features)
    For each stock, its last year of resolved outcomes under E0: how often it hit the target, how
    often it was stopped, its mean net (shrunk toward the week's average, 5 trades' worth). The
    trader learns whether a stock's record should count, like any other feature.

Post-mortem (``diagnose``) of every trade it takes, once closed
    path: the best and worst it got (MFE, MAE), on which day; "never worked", "went right then
    reversed", "gapped through the stop", "timed out flat";
    exits: what each of the five would have made, the best one, and the regret of the one chosen;
    the call: its predicted net against the result;
    reasons: the three features that pushed the score most;
    influences: market, oil, bonds, dollar and the stock's own part (moonshot.explain);
    news: the company's filings during the trade.
    Its lessons reach the next trade three ways: the exit chooser (hindsight exits), the memory
    features (each stock's record), and the stake (``trust``: half size while its own last 26 weeks of
    trades have not beaten random picks under the same exits).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from brain.paper import moonshot as m1

EXITS = ("E0 fixed", "E1 scaled", "E2 trailing", "E3 breakeven", "E4 follow-through")
MAXH = 15
WIN = 0.40
BAGS = 8
SHRINK_EXIT, SHRINK_MEM = 30.0, 5.0
TOP_TRACK = 10
MEM = ("mem_hit", "mem_stop", "mem_net")
FEATURES = m1.FEATURES + MEM
PLAIN = {**m1.PLAIN, "mem_hit": "this stock often hit +20% this year", "mem_stop": "this stock often got stopped this year",
         "mem_net": "this stock's trades did well this year"}


# --- exits ------------------------------------------------------------------------------------------------
def rule_params(name: str, sig: np.ndarray) -> dict:
    """sig: each stock's daily volatility over the last 20 sessions."""
    n = len(sig)
    one = lambda v: np.full(n, v)
    s10 = sig * math.sqrt(10)
    if name == "E0 fixed":
        return {"target": one(0.20), "stop": one(0.10), "hold": 10}
    if name == "E1 scaled":
        return {"target": np.clip(2.5 * s10, 0.08, 0.60), "stop": np.clip(1.25 * s10, 0.04, 0.30), "hold": 10}
    if name == "E2 trailing":
        return {"target": None, "stop": one(0.10), "trail": np.clip(2.5 * sig, 0.05, 0.25), "hold": 15}
    if name == "E3 breakeven":
        return {"target": one(0.30), "stop": one(0.10), "breakeven": 0.10, "hold": 15}
    if name == "E4 follow-through":
        return {"target": one(0.20), "stop": one(0.10), "check": (2, 0.03), "hold": 10}
    raise ValueError(name)


def run_exit(o, h, l, c, d: int, prm: dict) -> tuple[np.ndarray, np.ndarray]:
    """Net return (after costs) and exit session of a trade bought at session d's open, every stock at
    once. o/h/l/c: arrays [session, stock]."""
    entry = o[d]
    n = len(entry)
    stop = entry * (1 - prm["stop"])
    tgt = entry * (1 + prm["target"]) if prm.get("target") is not None else np.full(n, np.inf)
    done = np.zeros(n, bool)
    px = np.full(n, np.nan)
    ex = np.full(n, d + prm["hold"] - 1)
    hc = entry.copy()
    for k in range(prm["hold"]):
        i = d + k
        if k > 0:
            gd = ~done & (o[i] <= stop)
            gu = ~done & ~gd & (o[i] >= tgt)
            px[gd | gu] = o[i][gd | gu]
            ex[gd | gu] = i
            done |= gd | gu
        s = ~done & (l[i] <= stop)
        px[s], ex[s] = stop[s], i
        done |= s
        t = ~done & (h[i] >= tgt)
        px[t], ex[t] = tgt[t], i
        done |= t
        hc = np.fmax(hc, c[i])                                   # stops move after the close, for tomorrow
        if prm.get("trail") is not None:
            stop = np.fmax(stop, hc * (1 - prm["trail"]))
        if prm.get("breakeven") is not None:
            stop = np.where(hc >= entry * (1 + prm["breakeven"]), np.fmax(stop, entry), stop)
        if prm.get("check") and k == prm["check"][0]:
            weak = ~done & (c[i] < entry * (1 + prm["check"][1]))
            px[weak], ex[weak] = c[i][weak], i
            done |= weak
    last = d + prm["hold"] - 1
    rest = ~done
    px[rest] = c[last][rest]
    return px / entry - 1 - 2 * m1.COST, ex


def all_exits(P: dict, stocks: list[str], d: int, sig: np.ndarray) -> dict:
    arr = {k: P[k][stocks].values for k in ("open", "high", "low", "close")}
    return {r: run_exit(arr["open"], arr["high"], arr["low"], arr["close"], d, rule_params(r, sig)) for r in EXITS}


def path_stats(P: dict, t: str, d: int, exit_i: int) -> dict:
    o = P["open"][t].values
    hi, lo = P["high"][t].values[d:exit_i + 1], P["low"][t].values[d:exit_i + 1]
    e = o[d]
    mfe, mae = float(hi.max() / e - 1), float(lo.min() / e - 1)
    return {"mfe": mfe, "mae": mae, "best_day": int(np.argmax(hi)), "worst_day": int(np.argmin(lo))}


# --- learning ---------------------------------------------------------------------------------------------
class Ensemble:
    """Per exit, ridge on winsorized net, bagged over weeks; score = mean - spread / 2."""

    def __init__(self, k: int, seed: int = 0, names: list[str] | None = None):
        self.k, self.rng = k, np.random.default_rng(seed)
        self.names = list(names or FEATURES)
        self.W = None
        self.mu, self.sd = np.zeros(k), np.ones(k)

    def fit(self, X, Y, week, now_week):
        if len(X) < 500:
            return
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([np.ones((len(X), 1)), np.clip((X - self.mu) / self.sd, -5, 5)])
        sw = 0.5 ** ((now_week - week) / m1.HALF_LIFE)
        weeks = np.unique(week)
        R = np.eye(Z.shape[1]) * 10.0
        R[0, 0] = 1e-6
        Yc = np.clip(Y, -WIN, WIN)
        W = []
        for _ in range(BAGS):
            pick = self.rng.choice(weeks, size=len(weeks))
            cnt = pd.Series(pick).value_counts()
            bw = sw * pd.Series(week).map(cnt).fillna(0).values
            A = (Z * bw[:, None]).T @ Z + R
            W.append(np.linalg.solve(A, (Z * bw[:, None]).T @ Yc))
        self.W = np.array(W)                                       # bags x features x exits

    def predict(self, X):
        if self.W is None:
            return None, None
        Z = np.hstack([np.ones((len(X), 1)), np.clip((X - self.mu) / self.sd, -5, 5)])
        P = np.einsum("nf,bfe->bne", Z, self.W)
        return P.mean(0), P.std(0)

    def reasons(self, x, exit_j: int, k: int = 3) -> list[tuple[str, float]]:
        z = np.clip((x - self.mu) / self.sd, -5, 5)
        contrib = z * self.W[:, 1:, exit_j].mean(0)
        top = np.argsort(-np.abs(contrib))[:k]
        return [(PLAIN.get(self.names[j], self.names[j]), float(contrib[j])) for j in top]


class ExitChooser:
    def __init__(self):
        self.rows = []                                             # (week, bucket, nets per exit)

    def add(self, week, buckets, nets):
        for b, n in zip(buckets, nets):
            self.rows.append((week, int(b), n))

    def choose(self, now_week, known_before) -> dict[int, int]:
        rows = [r for r in self.rows if r[0] < known_before]
        if len({r[0] for r in rows}) < 30:
            return {b: 0 for b in range(6)}
        allv = np.array([r[2] for r in rows])
        overall = allv.mean(0)
        out = {}
        for b in range(6):
            v = np.array([r[2] for r in rows if r[1] == b]).reshape(-1, len(EXITS))
            score = (v.sum(0) + SHRINK_EXIT * overall) / (len(v) + SHRINK_EXIT)
            out[b] = int(np.argmax(score))
        return out


def diagnose(P, t, d, k_exit, nets_all, chosen, pred, x, ens, events, j):
    ex_i = int(nets_all[EXITS[chosen]][1][j])
    net = float(nets_all[EXITS[chosen]][0][j])
    ps = path_stats(P, t, d, ex_i)
    by_exit = {r: float(nets_all[r][0][j]) for r in EXITS}
    best = max(by_exit, key=by_exit.get)
    if ps["mfe"] < 0.03:
        shape = "never worked (it never got 3% up)"
    elif ps["mfe"] >= 0.10 and net < 0:
        shape = "went right, then reversed (was 10%+ up, ended a loss)"
    elif net <= -0.10 - 2 * m1.COST - 1e-9 and P["open"][t].values[ex_i] < P["open"][t].values[d] * 0.9 and ex_i > d:
        shape = "gapped through the stop"
    elif net > 0.15:
        shape = "worked: a moonshot or close"
    elif abs(net) < 0.03:
        shape = "timed out flat"
    else:
        shape = "worked a little" if net > 0 else "drifted to a loss"
    why = m1.explain(P, t, d, ex_i, events)
    return {"date": str(P["close"].index[d]), "ticker": t, "exit": EXITS[chosen], "net": net,
            "shape": shape, "mfe": ps["mfe"], "mae": ps["mae"], "best_day": ps["best_day"],
            "by_exit": by_exit, "best_exit": best, "regret": by_exit[best] - net,
            "predicted": float(pred), "reasons": ens.reasons(x, chosen), "influences": why["parts"],
            "filings_during": why["filings_during"], "closed": str(P["close"].index[ex_i])}


def lesson(dg: dict) -> str:
    r = "; ".join(f"{n} ({v:+.3f})" for n, v in dg["reasons"])
    big = max(dg["influences"].items(), key=lambda kv: abs(kv[1]))
    alt = (f"{dg['best_exit']} would have made {dg['by_exit'][dg['best_exit']]:+.1%}" if dg["regret"] > 0.005
           else "its exit was the best of the five")
    return (f"{dg['ticker']} {dg['net']:+.1%} with {dg['exit']} ({dg['shape']}); predicted {dg['predicted']:+.1%}; "
            f"best {dg['mfe']:+.1%} on day {dg['best_day'] + 1}, worst {dg['mae']:+.1%}; {alt}; "
            f"largest influence {big[0]} ({big[1]:+.1%}); filed: {', '.join(dg['filings_during']) or 'nothing'}; "
            f"picked for: {r}")


# --- the simulation ---------------------------------------------------------------------------------------
def simulate(raw: dict, sec: dict, stocks: list[str], start: str = m1.START, shuffle: bool = False,
             seed: int = 0, journal: bool = True, record_all: bool = False, dossier=None) -> dict:
    """``record_all`` keeps, for every week, every stock's score and outcome too (for the review of the
    moonshots it missed, brain/paper/misses.py); it changes nothing the trader does. ``dossier``
    (brain/paper/dossier.Dossier, test P5) adds what the company's own reports say, as features, and
    writes each trade's dossier into the journal before the trade."""
    from brain.paper import dossier as ds
    names = list(FEATURES) + (list(ds.FEATURES) if dossier is not None else [])
    if dossier is not None:
        PLAIN.update(ds.PLAIN)
    P = m1.panels(raw)
    dates = P["close"].index.values
    stocks = [s for s in stocks if s in P["close"] and P["close"][s].iloc[:260].notna().sum() > 200]
    events = m1.filing_events(sec, dates)
    week = pd.Series(pd.to_datetime(dates)).dt.to_period("W").values
    firsts = [i for i in range(1, len(dates)) if week[i] != week[i - 1] and i >= 261 and i + MAXH <= len(dates)]
    s0 = int(np.searchsorted(dates, start))
    rng = np.random.default_rng(seed)
    ens, chooser = Ensemble(len(names), seed, names), ExitChooser()
    hist = {"X": [], "Y": [], "exit": [], "week": []}
    memo = []                                                      # (week, resolved_at, stock idx, hit, stop, net) under E0
    vixs = []
    trades, weeks, record = [], [], []
    every = []
    for wk, d in enumerate(firsts):
        F = m1.panel(P, stocks, events, sec.get("sic", {}), d)
        ok = m1.tradable(P, stocks, d) & np.isfinite(F.values).all(1)
        r = np.log(P["close"][stocks].iloc[d - 21:d]).diff().iloc[1:]
        sig = r.std().fillna(0.05).values
        ex = all_exits(P, stocks, d, sig)
        nets = np.column_stack([ex[e][0] for e in EXITS])
        resolved = np.max(np.column_stack([ex[e][1] for e in EXITS]), axis=1)
        ok &= np.isfinite(nets).all(1)
        # memory: each stock's resolved E0 record over the last year, shrunk to the cross-section
        past = [mm for mm in memo if mm[1] < d and mm[0] > wk - 52]
        hitv, stopv, netv = (np.zeros(len(stocks)), np.zeros(len(stocks)), np.zeros(len(stocks)))
        cnt = np.zeros(len(stocks))
        for mm in past:
            hitv[mm[2]] += mm[3]
            stopv[mm[2]] += mm[4]
            netv[mm[2]] += mm[5]
            cnt[mm[2]] += 1
        base = (np.array([mm[3] for mm in past]).mean() if past else 0.04,
                np.array([mm[4] for mm in past]).mean() if past else 0.3,
                np.array([mm[5] for mm in past]).mean() if past else 0.0)
        F["mem_hit"] = (hitv + SHRINK_MEM * base[0]) / (cnt + SHRINK_MEM)
        F["mem_stop"] = (stopv + SHRINK_MEM * base[1]) / (cnt + SHRINK_MEM)
        F["mem_net"] = (netv + SHRINK_MEM * base[2]) / (cnt + SHRINK_MEM)
        if dossier is not None:
            E = dossier.features(d, stocks)
            for col in ds.FEATURES:
                F[col] = E[col].values
        X = F[names].values
        vix = F["lvix"].iloc[0]
        vixs.append(vix)
        hi_fear = vix > np.median(vixs[-52:])
        vt = pd.Series(np.where(ok, sig, np.nan)).rank(pct=True).fillna(0.5).values
        bucket = (hi_fear * 3 + np.minimum(2, (vt * 3).astype(int))).astype(int)
        if d >= s0:
            HX = np.vstack(hist["X"]) if hist["X"] else np.zeros((0, len(names)))
            if len(HX):
                known = np.concatenate(hist["exit"]) < d
                ens.fit(HX[known], np.vstack(hist["Y"])[known], np.concatenate(hist["week"])[known], wk)
            choice = chooser.choose(wk, d)
            mean, spread = ens.predict(X)
            book, picks = [], []
            if mean is not None:
                cj = np.array([choice[b] for b in bucket])
                score = mean[np.arange(len(stocks)), cj] - 0.5 * spread[np.arange(len(stocks)), cj]
                score[~ok] = -np.inf
                order = np.argsort(score)[::-1]
                picks = [j for j in order[:m1.TOP] if score[j] > 0]
                top10 = [j for j in order[:TOP_TRACK] if ok[j]]
                chooser.add(wk, bucket[top10], nets[top10])
                for j in picks:
                    dg = diagnose(P, stocks[j], d, None, ex, cj[j], mean[j, cj[j]], X[j], ens, events, j) if journal else \
                        {"net": float(nets[j, cj[j]]), "exit": EXITS[cj[j]], "ticker": stocks[j], "date": str(dates[d])}
                    if journal:
                        dg["lesson"] = lesson(dg)
                        if dossier is not None:
                            dg["dossier"] = dossier.card(stocks[j], d)
                    trades.append(dg)
                    book.append(dg["net"])
                rnd = float(np.mean(nets[ok, :][np.arange(ok.sum()), cj[ok]]))
                if record_all:
                    every.append({"d": d, "ok": ok.copy(), "score": np.where(np.isfinite(score), score, np.nan).astype(np.float32),
                                  "picked": np.isin(np.arange(len(stocks)), picks), "X": X.astype(np.float32)})
            else:
                rnd = float(np.mean(nets[ok, 0]))
            edge_hist = [x for x in record[-26:]]
            trust = 1.0 if (len(edge_hist) < 8 or np.mean(edge_hist) > 0) else 0.5
            wnet = float(np.mean(book)) if book else 0.0
            if book:
                record.append(wnet - rnd)
            weeks.append({"date": str(dates[d]), "net": wnet, "random": rnd, "n": len(book), "trust": trust,
                          "exits": {EXITS[k]: int(sum(1 for b in bucket[ok] if choice[b] == k)) for k in range(len(EXITS))}})
        Y = nets[ok].copy()
        if shuffle:
            Y = Y[rng.permutation(len(Y))]
        hist["X"].append(X[ok])
        hist["Y"].append(Y)
        hist["exit"].append(resolved[ok])
        hist["week"].append(np.full(ok.sum(), wk))
        e0n, e0x = ex["E0 fixed"]
        for j in np.flatnonzero(ok):
            n0 = float(e0n[j])
            memo.append((wk, int(e0x[j]), j, float(n0 >= 0.20 - 2 * m1.COST - 1e-9), float(n0 <= -0.10 - 2 * m1.COST + 1e-9), n0))
    out = {"trades": trades, "weeks": weeks, "stocks": len(stocks), "shuffled": shuffle}
    if record_all:
        out.update(every=every, names=stocks)
    return out


def grade(run: dict, shuffled: dict, test: str = "P4") -> dict:
    """P4 or P5, as declared in docs/PAPER.md before each first run (the same three checks)."""
    w = run["weeks"]
    net = [x["net"] for x in w]
    edge = [x["net"] - x["random"] if x["n"] else 0.0 for x in w]
    vs = [a["net"] - b["net"] for a, b in zip(w, shuffled["weeks"])]
    t = lambda v: m1.tstat(v)
    (m, tt, p), (me, te, pe), (ms, ts, ps) = t(net), t(edge), t(vs)
    checks = {"beats random picks under the same exits (one-sided p < 0.05)": me > 0 and pe < 0.05,
              "makes money after costs (one-sided p < 0.05)": m > 0 and p < 0.05,
              "beats the same learner taught shuffled outcomes (p < 0.05)": ms > 0 and ps < 0.05}
    return {"test": test, "weeks": len(w), "traded_weeks": sum(1 for x in w if x["n"]), "trades": len(run["trades"]),
            "weekly_net": {"mean": m, "t": tt, "p": p}, "vs_random": {"mean": me, "t": te, "p": pe},
            "vs_shuffled": {"mean": ms, "t": ts, "p": ps}, "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL"}
