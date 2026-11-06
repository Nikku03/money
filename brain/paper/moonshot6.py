"""P12: P11 with every data layer wired in (docs/PAPER.md). Paper only, a simulation of the past.

Same machine as P11 (``moonshot5``): it hunts only in weeks of high fear, among the most volatile
third, ranked by its lessons, ten picks with the breakeven exit, and it checks its own lessons. What
changes is what it sees: who is buying and who is betting against (``layers``: insider purchases and
sales, short-sale volume). And it sees whichever of those passed P11's rule on 2016-2020 and
2021-2026 (``EXTRA``).

Forward only, and without weight until a forward record tests them, each forward pick also carries
what the repo can see live and has no free history of: options' priced moves, headlines, guidance tags
in earnings releases, and alternative data. ``forward5`` records them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.paper import dossier as ds
from brain.paper import layers as ly
from brain.paper import market as mk
from brain.paper import moonshot as m1
from brain.paper import moonshot2 as m2
from brain.paper import moonshot5 as m5

# new lessons that passed P11's rule on both seen periods (filled from review, docs/PAPER.md P12)
EXTRA: dict[str, int] = {}
NEG = {"ins_buys": "no insider buying", "ins_buyers": "no insiders buying", "ins_sells": "insiders not selling",
       "ins_net": "insiders sold more than they bought", "short_r5": "little short selling this week",
       "short_r20": "little short selling this month", "short_chg": "short selling falling"}
PLAIN = {**m5.PLAIN, **{k: (ly.PLAIN[k] if s > 0 else NEG[k]) for k, s in EXTRA.items()}}


def simulate(raw: dict, sec: dict, facts: dict, stocks: list[str], start: str, layers: dict, end_data: str | None = None,
             shuffle: bool = False, seed: int = 0, journal: bool = True, live: bool = False) -> dict:
    """``layers``: {"insiders": rows for layers.Insiders, "short": rows for layers.ShortVolume}."""
    if end_data:
        raw = {t: df[df["date"].astype(str) <= end_data] for t, df in raw.items()}
    P = m1.panels(raw)
    dates = P["close"].index.values
    stocks = [s for s in stocks if s in P["close"] and s not in mk.NEEDED and P["close"][s].iloc[:260].notna().sum() > 200]
    events = m1.filing_events(sec, dates)
    dos = ds.Dossier({t: facts["facts"].get(t, []) for t in stocks}, facts["names"], P, events, sec.get("sic", {}))
    ins, sv = ly.Insiders(layers.get("insiders", {}), dates), ly.ShortVolume(layers.get("short", {}), P)
    week = pd.Series(pd.to_datetime(dates)).dt.to_period("W").values
    firsts = [i for i in range(1, len(dates)) if week[i] != week[i - 1] and i >= 261 and (live or i + m2.MAXH <= len(dates))]
    s0 = int(np.searchsorted(dates, start))
    rng = np.random.default_rng(seed)
    base = {} if shuffle else {**m5.LESSONS, **EXTRA}
    vixs, record, trades, weeks, pending, pending_weeks = [], [], [], [], [], []
    for d in firsts:
        F = m1.panel(P, stocks, events, sec.get("sic", {}), d)
        ok = m1.tradable(P, stocks, d) & np.isfinite(F.values).all(1)
        E = dos.features(d, stocks)
        for c in ds.FEATURES:
            F[c] = E[c].values
        L = pd.concat([ins.features(d, stocks), sv.features(d, stocks)], axis=1)
        for c in ly.TRAITS:
            F[c] = L[c].values                                  # may be missing (short volume): ranked as average
        vixs.append(float(F["lvix"].iloc[0]))
        complete = d + m2.MAXH <= len(dates)
        if complete:
            sig = np.log(P["close"][stocks].iloc[d - 21:d]).diff().iloc[1:].std().fillna(0.05).values
            ex = m2.all_exits(P, stocks, d, sig)
            nets = np.column_stack([ex[e][0] for e in m2.EXITS])
            okx = ok & np.isfinite(nets).all(1)
        else:
            ex, nets, okx = None, None, ok
        lessons = m5.health(record, d, base)
        sc, C = m5.score(F, okx, lessons)
        if shuffle:
            sc = rng.random(len(stocks))
        lv = F["lv20"].values
        vol_third = okx & (lv >= np.quantile(lv[okx], 2 / 3)) if okx.any() else okx
        hunt = True if shuffle else m5.fear(vixs)
        cand = np.flatnonzero(vol_third)
        picks = list(cand[np.argsort(-sc[cand])[:m5.TOP_N]]) if hunt else []
        if d >= s0 and not complete:
            pending_weeks.append({"date": str(dates[d]), "hunt": hunt, "set_aside": sorted(set(base) - set(lessons))})
            for j in picks:
                pending.append({"date": str(dates[d]), "ticker": stocks[j], "exit": m2.EXITS[m5.EXIT], "sig": 0.0,
                                "why": _why(C[j], lessons), "dossier": dos.card(stocks[j], d),
                                "layers": {c: (None if not np.isfinite(F[c].iloc[j]) else float(F[c].iloc[j])) for c in ly.TRAITS}})
            continue
        if not complete:
            continue
        if d >= s0:
            book = []
            for j in picks:
                tr = {"date": str(dates[d]), "ticker": stocks[j], "exit": m2.EXITS[m5.EXIT], "net": float(nets[j, m5.EXIT]),
                      "closed": str(dates[int(ex[m2.EXITS[m5.EXIT]][1][j])]), "score": float(sc[j])}
                if journal:
                    tr["why"] = _why(C[j], lessons)
                    tr["dossier"] = dos.card(stocks[j], d)
                    tr["layers"] = {c: (None if not np.isfinite(F[c].iloc[j]) else float(F[c].iloc[j])) for c in ly.TRAITS}
                    tr["by_exit"] = {e: float(nets[j, i]) for i, e in enumerate(m2.EXITS)}
                trades.append(tr)
                book.append(tr["net"])
            weeks.append({"date": str(dates[d]), "net": float(np.mean(book)) if book else 0.0,
                          "random": float(nets[okx, m5.EXIT].mean()), "n": len(book), "gate": bool(hunt),
                          "vix": float(np.exp(vixs[-1])), "set_aside": sorted(set(base) - set(lessons))})
        ics = {k: mk.spearman(*_finite(F[k].values.astype(float)[okx], nets[okx, m5.EXIT])) for k in base}
        record.append((int(d + m2.MAXH - 1), ics))
    return {"trades": trades, "weeks": weeks, "stocks": len(stocks), "shuffled": shuffle, "start": start, "end_data": end_data,
            "learned_from": str(dates[firsts[0]]), "pending": pending, "pending_weeks": pending_weeks,
            "last_date": str(dates[-1]), "lessons": base}


def _finite(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _why(c: np.ndarray, lessons: dict, k: int = 3) -> list[str]:
    if not len(lessons):
        return []
    names = list(lessons)
    top = np.argsort(-c)[:k]
    return [f"{PLAIN.get(names[i], names[i])} ({c[i]:+.1f} sd)" for i in top]
