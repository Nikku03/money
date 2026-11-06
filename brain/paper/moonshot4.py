"""The fourth machine (P10): it learns how the market works first, and trades second (docs/PAPER.md).

Paper only, a simulation of the past. Same clock, trade, exits, exit chooser, tradability check,
gate, learner (``moonshot3.Stats``), company dossier, memory and attention over past situations as
P9 (``moonshot3.simulate(memory=True, attention=True)``), so the difference is the market model.

What changes
    Every week, before deciding, it steps its market model (brain/paper/market.py):
    - it learns what the market paid for in the week just finished, across every stock on its list,
      and scores the forecast it made a week ago;
    - it reads the forces running the market now, and attends to the past weeks that looked like it;
    - it forecasts what the market will pay for this week, and so each stock's expected move.
    Seven of its readings join P9's features (``MMF``): each stock's expected move, last week's push
    from its forces and its own move apart from them (week and month), and three about the whole
    market (its list's expected move this week, breadth, moonshot weather).

Lessons
    Each week's market lesson (what moved, what the market paid for and punished, how much of the
    list's moves its traits explained, what it had expected, its rank correlation), every quarter
    what it believes about how the market works, and for every trade, the market's view of that stock
    (its strongest forces, what its traits were expected to earn) beside P9's post-mortem, dossier and
    attended situations.

The control (``shuffle``) scrambles what it learns from, both the trades' outcomes (as before) and,
inside each week, which stock made which move, so its market model learns nothing real either.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from brain.paper import dossier as ds
from brain.paper import market as mk
from brain.paper import moonshot as m1
from brain.paper import moonshot2 as m2
from brain.paper import moonshot3 as m3

MMF = ("mm_exp", "mm_force5", "mm_own5", "mm_own20", "mm_list", "mm_breadth", "mm_moon")
PLAIN = {**m3.PLAIN, "mm_exp": "the market model expects it to beat its list this week",
         "mm_force5": "its forces pushed it up last week", "mm_own5": "it rose on its own last week",
         "mm_own20": "it rose on its own this month", "mm_list": "the market model expects its list to rise this week",
         "mm_breadth": "most of its list is above its 50-day average", "mm_moon": "many stocks are running 20%+ (moonshot weather)"}
BELIEF_EVERY = 13


def simulate(raw: dict, sec: dict, facts: dict, stocks: list[str], start: str, end_data: str | None = None,
             shuffle: bool = False, seed: int = 0, journal: bool = True, live: bool = False) -> dict:
    """P10: P9 (memory and attention) with the market model. ``raw`` must hold mk.NEEDED besides the
    stocks. Trade from ``start``; no data after ``end_data``; ``live`` as in moonshot3."""
    from brain.paper import attention as atn
    from brain.paper import memory as mem
    if end_data:
        raw = {t: df[df["date"].astype(str) <= end_data] for t, df in raw.items()}
    missing = [a for a in mk.NEEDED if a not in raw]
    if missing:
        raise ValueError(f"the market model needs {missing}")
    P = m1.panels(raw)
    dates = P["close"].index.values
    stocks = [s for s in stocks if s in P["close"] and s not in mk.NEEDED and P["close"][s].iloc[:260].notna().sum() > 200]
    events = m1.filing_events(sec, dates)
    dos = ds.Dossier({t: facts["facts"].get(t, []) for t in stocks}, facts["names"], P, events, sec.get("sic", {}))
    sic2 = np.array([str(sec.get("sic", {}).get(t) or "")[:2] for t in stocks])
    week = pd.Series(pd.to_datetime(dates)).dt.to_period("W").values
    firsts = [i for i in range(1, len(dates)) if week[i] != week[i - 1] and i >= 261 and (live or i + m2.MAXH <= len(dates))]
    s0 = int(np.searchsorted(dates, start))
    names = list(m3.FEATURES) + list(m3.MEMF) + list(m3.ATTF) + list(MMF)
    att = atn.AttentionMemory(len(m3.FEATURES), len(m3.ATTF), seed)
    mm = mk.MarketModel(P, stocks, events, seed, shuffle)
    analogs, own = mem.Analogs(), []
    pending, pending_weeks = [], []
    w_start = next(w for w, d in enumerate(firsts) if d >= s0)
    w_score = max(26, w_start - m3.SCORE_LEAD)
    rng = np.random.default_rng(seed)
    st, chooser = m3.Stats(len(names), seed, names), m2.ExitChooser()
    st_plain = PLAIN
    raw_weeks = {}
    finished = collections.deque()
    memo = collections.deque()
    vixs, shadow = [], []
    trades, weeks, beliefs, skill = [], [], [], []
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
        else:
            ex, nets, resolved = None, np.full((len(stocks), len(m2.EXITS)), np.nan), np.full(len(stocks), np.inf)
        while memo and memo[0][0] <= wk - 52:
            memo.popleft()
        past = [x for x in memo if x[1] < d]
        hitv, stopv, netv, cnt = (np.zeros(len(stocks)) for _ in range(4))
        for x in past:
            hitv[x[2]] += x[3]; stopv[x[2]] += x[4]; netv[x[2]] += x[5]; cnt[x[2]] += 1
        base = (np.mean([x[3] for x in past]) if past else 0.04, np.mean([x[4] for x in past]) if past else 0.3,
                np.mean([x[5] for x in past]) if past else 0.0)
        F["mem_hit"] = (hitv + m2.SHRINK_MEM * base[0]) / (cnt + m2.SHRINK_MEM)
        F["mem_stop"] = (stopv + m2.SHRINK_MEM * base[1]) / (cnt + m2.SHRINK_MEM)
        F["mem_net"] = (netv + m2.SHRINK_MEM * base[2]) / (cnt + m2.SHRINK_MEM)
        E = dos.features(d, stocks)
        for c in ds.FEATURES:
            F[c] = E[c].values
        for c, v in m3.industry(F, sic2, ok).items():
            F[c] = v
        Xbase = F[list(m3.FEATURES)].values
        # the market first: learn last week, read this one, forecast it
        mv = mm.week(d, wk, F, E, m1.tradable(P, stocks, d) & np.isfinite(F.values).all(1))
        skill.append({"date": str(dates[d]), "skill26": mv["skill"]})
        an_net, an_hit = analogs.recall(wk, d, Xbase)
        on, osum = np.zeros(len(stocks)), np.zeros(len(stocks))
        for r_at, j, n in own:
            if r_at < d:
                on[j] += 1
                osum[j] += n
        X = np.column_stack([Xbase, an_net, an_hit, on, osum / (on + m3.OWN_SHRINK)])
        if att.trained_at is None or wk - att.trained_at >= m3.ATT_EVERY:
            att.fit(d, wk)
        X = np.column_stack([X, att.recall(d, Xbase)[0]] + [mv[c] for c in MMF])
        if journal and wk % BELIEF_EVERY == 0 and d >= s0:
            beliefs.append({"date": str(dates[d]), **mm.belief()})
        vixs.append(F["lvix"].iloc[0])
        hi_fear = vixs[-1] > np.median(vixs[-52:])
        vt = pd.Series(np.where(ok, sig, np.nan)).rank(pct=True).fillna(0.5).values
        bucket = (hi_fear * 3 + np.minimum(2, (vt * 3).astype(int))).astype(int)
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
                    open_, gm, gt = m3.gate(shadow, d)
                    choice = chooser.choose(wk, d)
                    cj = np.array([choice[b] for b in bucket])
                    score = mean[np.arange(len(stocks)), cj] - 0.5 * spread[np.arange(len(stocks)), cj]
                    score[~ok] = -np.inf
                    top3 = [j for j in np.argsort(score)[::-1][:m1.TOP] if ok[j]]
                    pending_weeks.append({"date": str(dates[d]), "gate": bool(open_), "gate_mean": gm, "gate_t": gt,
                                          "market": mm.journal[-1]["lesson"] if mm.journal else None})
                    for j in ([j for j in top3 if score[j] > 0] if open_ else []):
                        pending.append({"date": str(dates[d]), "ticker": stocks[j], "exit": m2.EXITS[cj[j]], "sig": float(sig[j]),
                                        "score": float(score[j]), "predicted": float(mean[j, cj[j]]),
                                        "reasons": _reasons(st, X[j], cj[j], st_plain), "dossier": dos.card(stocks[j], d),
                                        "market_view": mm.view(j), "attended": m3.attended(att, d, Xbase[j])})
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
                open_, gm, gt = m3.gate(shadow[:-1], d)
                top10 = [j for j in order[:m2.TOP_TRACK] if ok[j]]
                chooser.add(wk, bucket[top10], nets[top10], resolved[top10])
                if d >= s0:
                    picks = [j for j in top3 if score[j] > 0] if open_ else []
                    book = []
                    for j in picks:
                        if journal:
                            dg = m2.diagnose(P, stocks[j], d, None, ex, cj[j], mean[j, cj[j]], X[j], st, events, j)
                            dg["reasons"] = _reasons(st, X[j], cj[j], st_plain)
                            dg["lesson"] = m2.lesson(dg)
                            dg["dossier"] = dos.card(stocks[j], d)
                            dg["attended"] = m3.attended(att, d, Xbase[j])
                            dg["market_view"] = mm.view(j)
                        else:
                            dg = {"net": float(nets[j, cj[j]]), "exit": m2.EXITS[cj[j]], "ticker": stocks[j],
                                  "date": str(dates[d]), "closed": str(dates[int(ex[m2.EXITS[cj[j]]][1][j])])}
                        trades.append(dg)
                        book.append(dg["net"])
                        own.append((int(ex[m2.EXITS[cj[j]]][1][j]), j, float(nets[j, cj[j]])))
                    weeks.append({"date": str(dates[d]), "net": float(np.mean(book)) if book else 0.0, "random": rnd,
                                  "n": len(book), "gate": bool(open_), "gate_mean": gm, "gate_t": gt, "lam": st.lam})
        if not complete:
            continue
        hit0 = (ex["E0 fixed"][0][ok] >= 0.20 - 2 * m1.COST - 1e-9).astype(float)
        att.add(wk, int(d + m2.MAXH - 1), Xbase[ok], np.column_stack([np.clip(nets[ok], -m2.WIN, m2.WIN), hit0]),
                [(stocks[j], str(dates[d])) for j in np.flatnonzero(ok)])
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
    H = mm.hist
    return {"trades": trades, "weeks": weeks, "stocks": len(stocks), "shuffled": shuffle, "start": start,
            "learned_from": str(dates[firsts[0]]), "end_data": end_data, "pending": pending,
            "pending_weeks": pending_weeks, "last_date": str(dates[-1]),
            "market": mm.journal, "beliefs": beliefs, "final_belief": mm.belief(),
            "market_record": _record(H),
            "skill": skill, "attention_val_loss": att.val_loss}


def _record(H: dict) -> list[dict]:
    return [{"week": w, "ic": ic, "ic_static": ics, "explained": r2, "ic_nogap": a, "ic_close": b, "ic_nogap_close": c}
            for w, ic, ics, r2, a, b, c in zip(H["date"], H["ic"], H["ic_static"], H["r2"], H["ic_nogap"], H["ic_close"],
                                               H["ic_nogap_close"])]


def market_pass(raw: dict, sec: dict, facts: dict, stocks: list[str], end_data: str | None = None, shuffle: bool = False,
                seed: int = 0) -> dict:
    """The market model alone, stepped exactly as ``simulate`` steps it (same list, weeks, tradable
    stocks, seed): its weekly record and lessons, without the trader. Its forecasts never depend on
    the trades, so they are the same as in a full run."""
    if end_data:
        raw = {t: df[df["date"].astype(str) <= end_data] for t, df in raw.items()}
    P = m1.panels(raw)
    dates = P["close"].index.values
    stocks = [s for s in stocks if s in P["close"] and s not in mk.NEEDED and P["close"][s].iloc[:260].notna().sum() > 200]
    events = m1.filing_events(sec, dates)
    dos = ds.Dossier({t: facts["facts"].get(t, []) for t in stocks}, facts["names"], P, events, sec.get("sic", {}))
    week = pd.Series(pd.to_datetime(dates)).dt.to_period("W").values
    firsts = [i for i in range(1, len(dates)) if week[i] != week[i - 1] and i >= 261 and i + m2.MAXH <= len(dates)]
    mm = mk.MarketModel(P, stocks, events, seed, shuffle)
    for wk, d in enumerate(firsts):
        F = m1.panel(P, stocks, events, sec.get("sic", {}), d)
        E = dos.features(d, stocks)
        mm.week(d, wk, F, E, m1.tradable(P, stocks, d) & np.isfinite(F.values).all(1))
    return {"market": mm.journal, "market_record": _record(mm.hist), "final_belief": mm.belief()}


def _reasons(st, x, j, plain, k=3):
    zz = np.clip((x - st.mu) / st.sd, -5, 5)
    c = zz * st.W[:, 1:, j].mean(0)
    top = np.argsort(-np.abs(c))[:k]
    return [(plain.get(st.names[i], st.names[i]), float(c[i])) for i in top]


def grade_market(run: dict, start: str, end: str | None = None) -> dict:
    """P10-M: does it understand the market? Its weekly rank correlation between each stock's expected
    move and the move that followed (open to next week's open), over every tradable stock on its list,
    in the weeks it traded (from ``start``). PASS if the mean is above 0, one-sided p < 0.05."""
    rec = [r for r in run["market_record"] if r["week"] >= start and (end is None or r["week"] <= end)]
    ic = [r["ic"] for r in rec if np.isfinite(r["ic"])]
    ics = [r["ic_static"] for r in rec if np.isfinite(r["ic_static"])]
    m, t, p = m1.tstat(ic)
    extra = {k: dict(zip(("mean", "t", "p"), m1.tstat([r[k] for r in rec if np.isfinite(r.get(k, np.nan))])))
             for k in ("ic_nogap", "ic_close", "ic_nogap_close")}
    return {"weeks": len(ic), "mean_rank_corr": m, "t": t, "p": p,
            "static_premia_only": dict(zip(("mean", "t", "p"), m1.tstat(ics))),
            "reported_only": {"without the morning gap": extra["ic_nogap"], "close to close": extra["ic_close"],
                              "without the gap, close to close": extra["ic_nogap_close"]},
            "explained": float(np.mean([r["explained"] for r in rec])) if rec else None,
            "verdict": "PASS" if (m > 0 and p < 0.05) else "FAIL"}
