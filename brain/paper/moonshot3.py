"""The third machine (P7): what the first two taught, built to be judged once on years it has never
seen (docs/PAPER.md). Paper only, a simulation of the past.

Same clock, trade, exits, exit chooser (with the look-ahead fixed), tradability check, memory and
dossier as P5. What changes:

It studies the past before trading
    It learns from every week it has data for before it starts trading, not just the last six months:
    a trader starting in 2016 has studied 2011-2015.

Learning that holds up (``Stats``)
    The same model of the expected net under each exit (ridge on winsorized outcomes, older weeks
    fading, bagged 8 times over weeks), kept as each finished week's sufficient statistics, so ten
    years of weeks cost no more than one.
    - Its features are scaled once, from what it had seen when it first scores, and frozen.
    - Each refit chooses its own ridge strength (3, 30 or 300): fit on all but the latest 26 finished
      weeks, test on those 26.
    - It learns only from weeks whose every trade has finished before the decision's open.

It earns the right to trade (``gate``)
    Every week it also records what its top 3 would have made against random picks under the same
    exits (its shadow record), whether it traded or not. It trades only while that record, over the
    last 52 finished weeks (at least 26 of them), is positive with t above 1. Otherwise it stays out,
    and keeps scoring. It starts scoring 52 weeks before it may trade, so it has a record when trading
    opens.

Two new features, from the missed-moonshot review
    ind_ret5, ind_ret20: how the rest of its industry (2-digit SEC code, tradable stocks) moved over
    the last week and month, against SPY. The review found these on 2021-2026, which is why the
    held-out test is 2016-2020.
"""
from __future__ import annotations

import collections
import math

import numpy as np
import pandas as pd

from brain.paper import dossier as ds
from brain.paper import moonshot as m1
from brain.paper import moonshot2 as m2

NEW = ("ind_ret5", "ind_ret20")
MEMF = ("an_net", "an_hit", "own_n", "own_net")             # P8: what memory recalls (memory.Analogs, its own trades)
PLAIN = {**m2.PLAIN, **ds.PLAIN, "ind_ret5": "its industry rose last week", "ind_ret20": "its industry rose this month",
         "an_net": "situations like this one did well", "an_hit": "situations like this one often hit +20%",
         "own_n": "it has traded this stock before", "own_net": "its own trades of this stock did well"}
OWN_SHRINK = 3.0
FEATURES = tuple(m2.FEATURES) + tuple(ds.FEATURES) + NEW
LAMBDAS = (3.0, 30.0, 300.0)
VAL_WEEKS = 26
GATE_WEEKS, GATE_MIN, GATE_T = 52, 26, 1.0
SCORE_LEAD = 52


def industry(F: pd.DataFrame, sic2: np.ndarray, ok: np.ndarray) -> dict:
    out = {}
    for k in (5, 20):
        r = F[f"ret{k}"].values
        v = np.zeros(len(r))
        for g in set(sic2[ok]) - {""}:
            m = ok & (sic2 == g)
            n, tot = m.sum(), r[m].sum()
            idx = np.flatnonzero(sic2 == g)
            v[idx] = np.where(m[idx], (tot - r[idx]) / max(1, n - 1), tot / max(1, n)) if n > 1 else 0.0
        out[f"ind_ret{k}"] = v
    return out


class Stats:
    """Bagged ridge over per-week sufficient statistics, with its ridge strength chosen by validation."""

    def __init__(self, k: int, seed: int = 0, names=None):
        self.k, self.rng = k, np.random.default_rng(seed)
        self.names = list(names or FEATURES)
        self.mu = self.sd = None
        self.weeks = {}                                        # week -> (S, B, yy)
        self.W = None
        self.lam = None

    def set_scaler(self, X: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9

    def z(self, X):
        return np.hstack([np.ones((len(X), 1)), np.clip((X - self.mu) / self.sd, -5, 5)])

    def add_week(self, w, X, Y):
        Z, Yc = self.z(X), np.clip(Y, -m2.WIN, m2.WIN)
        self.weeks[w] = (Z.T @ Z, Z.T @ Yc, (Yc * Yc).sum(0))

    def _solve(self, ws, mult, lam):
        S = sum(mult[i] * self.weeks[w][0] for i, w in enumerate(ws))
        B = sum(mult[i] * self.weeks[w][1] for i, w in enumerate(ws))
        R = np.eye(S.shape[0]) * lam
        R[0, 0] = 1e-6
        return np.linalg.solve(S + R, B)

    def fit(self, now_wk: int):
        ws = sorted(self.weeks)
        if len(ws) < 26:
            return
        decay = np.array([0.5 ** ((now_wk - w) / m1.HALF_LIFE) for w in ws])
        if len(ws) >= VAL_WEEKS + 52:
            old, new = ws[:-VAL_WEEKS], ws[-VAL_WEEKS:]
            best = None
            for lam in LAMBDAS:
                Wl = self._solve(old, decay[:len(old)], lam)
                err = sum(self.weeks[w][2].sum() - 2 * (Wl * self.weeks[w][1]).sum()
                          + np.einsum("fe,fg,ge->", Wl, self.weeks[w][0], Wl) for w in new)
                if best is None or err < best[0]:
                    best = (err, lam)
            self.lam = best[1]
        else:
            self.lam = 30.0
        W = []
        for _ in range(m2.BAGS):
            cnt = collections.Counter(self.rng.choice(len(ws), size=len(ws)))
            mult = np.array([cnt.get(i, 0) for i in range(len(ws))]) * decay
            W.append(self._solve(ws, mult, self.lam))
        self.W = np.array(W)

    def predict(self, X):
        if self.W is None:
            return None, None
        P = np.einsum("nf,bfe->bne", self.z(X), self.W)
        return P.mean(0), P.std(0)

    def reasons(self, x, j, k=3):
        zz = np.clip((x - self.mu) / self.sd, -5, 5)
        c = zz * self.W[:, 1:, j].mean(0)
        top = np.argsort(-np.abs(c))[:k]
        return [(PLAIN.get(self.names[i], self.names[i]), float(c[i])) for i in top]


def gate(shadow: list, d: int) -> tuple[bool, float, float]:
    rec = [e for r, e in shadow if r < d][-GATE_WEEKS:]
    if len(rec) < GATE_MIN:
        return False, float("nan"), float("nan")
    m, t, _ = m1.tstat(rec)
    return (m > 0 and t > GATE_T), m, t


def simulate(raw: dict, sec: dict, facts: dict, stocks: list[str], start: str, end_data: str | None = None,
             shuffle: bool = False, seed: int = 0, journal: bool = True, memory: bool = False, live: bool = False) -> dict:
    """Trade from ``start``; see no data after ``end_data`` (the held-out test ends at 2020-12-31).

    ``memory`` (P8): before each decision it recalls situations like each stock's present one among
    everything it has watched (memory.Analogs: the 50 nearest finished cases of the last two years,
    their mean net and +20% rate under the fixed exit) and its own trades of that stock (how many,
    their mean net, shrunk to 0 by 3 trades' worth), and weighs them as features.
    ``live``: also score weeks whose future has not happened; their picks come back as ``pending``
    and nothing about them is learned. With both off, this is P7 exactly."""
    from brain.paper import memory as mem
    if end_data:
        raw = {t: df[df["date"].astype(str) <= end_data] for t, df in raw.items()}
    P = m1.panels(raw)
    dates = P["close"].index.values
    stocks = [s for s in stocks if s in P["close"] and P["close"][s].iloc[:260].notna().sum() > 200]
    events = m1.filing_events(sec, dates)
    dos = ds.Dossier({t: facts["facts"].get(t, []) for t in stocks}, facts["names"], P, events, sec.get("sic", {}))
    sic2 = np.array([str(sec.get("sic", {}).get(t) or "")[:2] for t in stocks])
    week = pd.Series(pd.to_datetime(dates)).dt.to_period("W").values
    firsts = [i for i in range(1, len(dates)) if week[i] != week[i - 1] and i >= 261 and (live or i + m2.MAXH <= len(dates))]
    s0 = int(np.searchsorted(dates, start))
    names = list(FEATURES) + (list(MEMF) if memory else [])
    analogs, own = (mem.Analogs(), []) if memory else (None, None)
    pending, pending_weeks = [], []
    w_start = next(w for w, d in enumerate(firsts) if d >= s0)
    w_score = max(26, w_start - SCORE_LEAD)
    rng = np.random.default_rng(seed)
    st, chooser = Stats(len(names), seed, names), m2.ExitChooser()
    raw_weeks = {}                                             # week -> (d, X, Y) until the scaler exists
    finished = collections.deque()                             # (resolved session, week)
    memo = collections.deque()
    vixs, shadow = [], []
    trades, weeks = [], []
    for wk, d in enumerate(firsts):
        F = m1.panel(P, stocks, events, sec.get("sic", {}), d)
        ok = m1.tradable(P, stocks, d) & np.isfinite(F.values).all(1)
        r = np.log(P["close"][stocks].iloc[d - 21:d]).diff().iloc[1:]
        sig = r.std().fillna(0.05).values
        complete = d + m2.MAXH <= len(dates)
        if complete:
            ex = m2.all_exits(P, stocks, d, sig)
            nets = np.column_stack([ex[e][0] for e in m2.EXITS])
            resolved = np.max(np.column_stack([ex[e][1] for e in m2.EXITS]), axis=1)
            ok &= np.isfinite(nets).all(1)
        else:                                                  # its future has not happened yet
            ex, nets, resolved = None, np.full((len(stocks), len(m2.EXITS)), np.nan), np.full(len(stocks), np.inf)
        while memo and memo[0][0] <= wk - 52:
            memo.popleft()
        past = [mm for mm in memo if mm[1] < d]
        hitv, stopv, netv, cnt = (np.zeros(len(stocks)) for _ in range(4))
        for mm in past:
            hitv[mm[2]] += mm[3]; stopv[mm[2]] += mm[4]; netv[mm[2]] += mm[5]; cnt[mm[2]] += 1
        base = (np.mean([mm[3] for mm in past]) if past else 0.04, np.mean([mm[4] for mm in past]) if past else 0.3,
                np.mean([mm[5] for mm in past]) if past else 0.0)
        F["mem_hit"] = (hitv + m2.SHRINK_MEM * base[0]) / (cnt + m2.SHRINK_MEM)
        F["mem_stop"] = (stopv + m2.SHRINK_MEM * base[1]) / (cnt + m2.SHRINK_MEM)
        F["mem_net"] = (netv + m2.SHRINK_MEM * base[2]) / (cnt + m2.SHRINK_MEM)
        E = dos.features(d, stocks)
        for c in ds.FEATURES:
            F[c] = E[c].values
        for c, v in industry(F, sic2, ok).items():
            F[c] = v
        X = Xbase = F[list(FEATURES)].values
        if memory:
            an_net, an_hit = analogs.recall(wk, d, Xbase)
            on, osum = np.zeros(len(stocks)), np.zeros(len(stocks))
            for r_at, j, n in own:
                if r_at < d:
                    on[j] += 1
                    osum[j] += n
            X = np.column_stack([Xbase, an_net, an_hit, on, osum / (on + OWN_SHRINK)])
        vixs.append(F["lvix"].iloc[0])
        hi_fear = vixs[-1] > np.median(vixs[-52:])
        vt = pd.Series(np.where(ok, sig, np.nan)).rank(pct=True).fillna(0.5).values
        bucket = (hi_fear * 3 + np.minimum(2, (vt * 3).astype(int))).astype(int)
        # learn every week whose trades all finished before this open
        while finished and finished[0][0] < d:
            _, w = finished.popleft()
            if st.mu is not None:
                dd, Xw, Yw = raw_weeks.pop(w)
                st.add_week(w, Xw, Yw)
        if wk >= w_score:
            if st.mu is None:
                st.set_scaler(np.vstack([v[1] for v in raw_weeks.values()]))
                for w in [w for w, v in raw_weeks.items() if v[0] + m2.MAXH - 1 < d]:
                    dd, Xw, Yw = raw_weeks.pop(w)
                    st.add_week(w, Xw, Yw)
            st.fit(wk)
            mean, spread = st.predict(X)
            if mean is not None and not complete:
                if d >= s0:
                    open_, gm, gt = gate(shadow, d)
                    choice = chooser.choose(wk, d)
                    cj = np.array([choice[b] for b in bucket])
                    score = mean[np.arange(len(stocks)), cj] - 0.5 * spread[np.arange(len(stocks)), cj]
                    score[~ok] = -np.inf
                    top3 = [j for j in np.argsort(score)[::-1][:m1.TOP] if ok[j]]
                    pending_weeks.append({"date": str(dates[d]), "gate": bool(open_), "gate_mean": gm, "gate_t": gt})
                    for j in ([j for j in top3 if score[j] > 0] if open_ else []):
                        pending.append({"date": str(dates[d]), "ticker": stocks[j], "exit": m2.EXITS[cj[j]], "sig": float(sig[j]),
                                        "score": float(score[j]), "predicted": float(mean[j, cj[j]]),
                                        "reasons": st.reasons(X[j], cj[j]), "dossier": dos.card(stocks[j], d)})
                continue
            if mean is not None:
                choice = chooser.choose(wk, d)
                cj = np.array([choice[b] for b in bucket])
                score = mean[np.arange(len(stocks)), cj] - 0.5 * spread[np.arange(len(stocks)), cj]
                score[~ok] = -np.inf
                order = np.argsort(score)[::-1]
                top3 = [j for j in order[:m1.TOP] if ok[j]]
                rnd = float(np.mean(nets[ok][np.arange(ok.sum()), cj[ok]]))
                shadow.append((d + m2.MAXH - 1, float(np.mean(nets[top3, cj[top3]])) - rnd))
                open_, gm, gt = gate(shadow[:-1], d)
                top10 = [j for j in order[:m2.TOP_TRACK] if ok[j]]
                chooser.add(wk, bucket[top10], nets[top10], resolved[top10])
                if d >= s0:
                    picks = [j for j in top3 if score[j] > 0] if open_ else []
                    book = []
                    for j in picks:
                        dg = m2.diagnose(P, stocks[j], d, None, ex, cj[j], mean[j, cj[j]], X[j], st, events, j) if journal else \
                            {"net": float(nets[j, cj[j]]), "exit": m2.EXITS[cj[j]], "ticker": stocks[j], "date": str(dates[d]),
                             "closed": str(dates[int(ex[m2.EXITS[cj[j]]][1][j])])}
                        if journal:
                            dg["lesson"] = m2.lesson(dg)
                            dg["dossier"] = dos.card(stocks[j], d)
                        trades.append(dg)
                        book.append(dg["net"])
                        if memory:
                            own.append((int(ex[m2.EXITS[cj[j]]][1][j]), j, float(nets[j, cj[j]])))
                    weeks.append({"date": str(dates[d]), "net": float(np.mean(book)) if book else 0.0, "random": rnd,
                                  "n": len(book), "gate": bool(open_), "gate_mean": gm, "gate_t": gt, "lam": st.lam})
        if not complete:
            continue
        if memory:
            e0 = ex["E0 fixed"][0]
            analogs.add(wk, int(d + m2.MAXH - 1), Xbase[ok], e0[ok], (e0[ok] >= 0.20 - 2 * m1.COST - 1e-9).astype(float))
        Y = nets[ok].copy()
        if shuffle:
            Y = Y[rng.permutation(len(Y))]
        raw_weeks[wk] = (d, X[ok], Y)
        finished.append((int(d + m2.MAXH - 1), wk))
        e0n, _ = ex["E0 fixed"]
        for j in np.flatnonzero(ok):
            n0 = float(e0n[j])
            memo.append((wk, int(d + m2.MAXH - 1), j, float(n0 >= 0.20 - 2 * m1.COST - 1e-9),
                         float(n0 <= -0.10 - 2 * m1.COST + 1e-9), n0))
    return {"trades": trades, "weeks": weeks, "stocks": len(stocks), "shuffled": shuffle, "start": start,
            "learned_from": str(dates[firsts[0]]), "end_data": end_data, "pending": pending,
            "pending_weeks": pending_weeks, "last_date": str(dates[-1]), "memory": memory}
