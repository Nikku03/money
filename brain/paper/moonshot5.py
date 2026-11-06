"""P11, "the oracle": what every version taught, in one machine (docs/PAPER.md). Paper only, a
simulation of the past. It cannot see the future; the name is the request's, not a claim.

What the review of every version found (``review.py``). It covered every tradable stock in every week
of 2016-2020 (the 2015-chosen list) and 2021-2026 (the 2021-chosen list), the ground truth that P3-P10
were all trying to learn.

1. **What every version chased loses.** The traits that make a +20% run likelier make the −10%
   stop likelier still: volatility, lottery-like days, penny prices, biotech, share sales, big recent
   moves. Every version learned to find movers and bought them. Their picks sat:
   - 0.4 to 1.1 standard deviations above their list on volatility, and 0.8 to 1.2 above on
     lottery-like days;
   - 0.3 to 0.6 below it on profitability, and 0.4 to 1.2 below it on nearness to the 52-week high;
   - P7-P10 also bought gap-ups at the open, 0.8 to 1.3 above. P5, the best of them on an honest
     list, bought gap-downs.
2. **Sixteen lessons held in both periods,** for the fixed, trailing and follow-through exits alike
   (the same sign, |t| of at least 2 in each period, for each exit). They are ``LESSONS``: against
   volatility, lottery-like days, big recent moves, penny prices, biotech, share sales in prospect
   and dilution; for profitability, operating margin, cash runway, liquidity, results that usually
   lift the stock, and an older last report.
3. **The lessons tell the worst from the rest; they do not find winners.** Their top picks are calm
   stocks that rarely move. Among the stocks volatile enough to run 20%, they still separate the
   worst: the bottom five lost far more than the top five in both periods.
4. **When to hunt.** Moonshot trades on the whole list did better in weeks of high fear (the VIX above
   its past-year median) in both periods: +0.26% and +0.23% a week, against −0.20% and −0.41% in calm
   weeks.
5. **The exit.** The breakeven exit (E3) had the best average outcome on the whole list in both
   periods.
6. **Breadth.** An edge of tenths of a percent cannot show through three trades a week, so it buys ten.
   Scrambling the outcomes across all stock-weeks leaves no lesson and no fear effect.

The machine (``LESSONS``, ``fear``, ``score``, ``health``):
- **When:** it trades only in weeks of high fear (the VIX at the last close above the median of its
  last 52 weekly readings).
- **What:** it considers only the most volatile third of the tradable list, the stocks that can run
  20%+.
- **Rank:** it ranks them by the lessons, each trait's gaussianized rank with its sign, equally
  weighted.
- **Buy:** it buys the top ten at the open, each with the breakeven exit (E3): a −10% stop, moved to
  the entry once a close is 10% up; a +30% target; 15 sessions.
- **It keeps checking its lessons** on its own finished history. A lesson whose weekly rank
  correlation with the E3 outcome, over the last 104 finished weeks (at least 52), runs against its
  sign with t below −2 is set aside until it recovers.

**The control** (``shuffle``) is the same machine taught scrambled outcomes. Re-deriving the lessons
from scrambled outcomes gives no lesson and no fear effect. So the control trades every week, ranks
the volatile third at random, and uses E3.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from brain.paper import dossier as ds
from brain.paper import market as mk
from brain.paper import moonshot as m1
from brain.paper import moonshot2 as m2

# trait: sign. Frozen from review.lessons on 2016-2020 and 2021-2026 (docs/PAPER.md, P11 review).
LESSONS = {"lv20": -1, "lv60": -1, "lmax250": -1, "n10": -1, "a1": -1, "a5": -1, "pen": -1, "bio": -1,
           "offer30": -1, "f_dilution": -1, "ldv": 1, "f_op_margin": 1, "f_profitable": 1, "f_runway": 1,
           "f_stale": 1, "f_earn_sign": 1}
PLAIN = {"lv20": "calm lately", "lv60": "calm", "lmax250": "no lottery-like day this year", "n10": "few 10% days",
         "a1": "no big move yesterday", "a5": "no big move this week", "pen": "not a penny stock",
         "bio": "not a drug or biotech company", "offer30": "no share sale in prospect", "f_dilution": "not diluting",
         "ldv": "heavily traded", "f_op_margin": "profitable operations", "f_profitable": "made a profit",
         "f_runway": "long cash runway", "f_stale": "older last report", "f_earn_sign": "its results usually lift it"}
EXIT = 3                                     # E3 breakeven
TOP_N = 10
FEAR_WEEKS = 52
HEALTH_WEEKS, HEALTH_MIN, HEALTH_T = 104, 52, -2.0


def fear(vix_history: list[float]) -> bool:
    """High fear: the latest VIX reading above the median of the last FEAR_WEEKS readings (including it)."""
    return bool(len(vix_history) and vix_history[-1] > np.median(vix_history[-FEAR_WEEKS:]))


def score(X: pd.DataFrame, ok: np.ndarray, lessons: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each stock's lessons score (mean of sign x gaussianized rank among ``ok``) and the per-lesson
    contributions (stocks x lessons)."""
    if not lessons:
        return np.zeros(len(X)), np.zeros((len(X), 0))
    C = np.column_stack([s * mk.grank(X[k].values.astype(float), ok) for k, s in lessons.items()])
    return C.mean(1), C


def health(record: list, d: int, lessons: dict) -> dict:
    """The lessons still standing at session d: ``record`` holds (resolved session, {lesson: weekly rank
    correlation with the E3 outcome}); only weeks finished before d count."""
    done = [r for at, r in record if at < d][-HEALTH_WEEKS:]
    out = {}
    for k, s in lessons.items():
        x = np.array([r[k] * s for r in done if np.isfinite(r.get(k, np.nan))])
        if len(x) >= HEALTH_MIN and x.std(ddof=1) > 0 and x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) < HEALTH_T:
            continue
        out[k] = s
    return out


def simulate(raw: dict, sec: dict, facts: dict, stocks: list[str], start: str, end_data: str | None = None,
             shuffle: bool = False, seed: int = 0, journal: bool = True, live: bool = False) -> dict:
    if end_data:
        raw = {t: df[df["date"].astype(str) <= end_data] for t, df in raw.items()}
    P = m1.panels(raw)
    dates = P["close"].index.values
    stocks = [s for s in stocks if s in P["close"] and s not in mk.NEEDED and P["close"][s].iloc[:260].notna().sum() > 200]
    events = m1.filing_events(sec, dates)
    dos = ds.Dossier({t: facts["facts"].get(t, []) for t in stocks}, facts["names"], P, events, sec.get("sic", {}))
    week = pd.Series(pd.to_datetime(dates)).dt.to_period("W").values
    firsts = [i for i in range(1, len(dates)) if week[i] != week[i - 1] and i >= 261 and (live or i + m2.MAXH <= len(dates))]
    s0 = int(np.searchsorted(dates, start))
    rng = np.random.default_rng(seed)
    base = {} if shuffle else dict(LESSONS)
    vixs, record, trades, weeks, pending, pending_weeks = [], [], [], [], [], []
    for d in firsts:
        F = m1.panel(P, stocks, events, sec.get("sic", {}), d)
        ok = m1.tradable(P, stocks, d) & np.isfinite(F.values).all(1)
        E = dos.features(d, stocks)
        for c in ds.FEATURES:
            F[c] = E[c].values
        vixs.append(float(F["lvix"].iloc[0]))
        complete = d + m2.MAXH <= len(dates)
        if complete:
            sig = np.log(P["close"][stocks].iloc[d - 21:d]).diff().iloc[1:].std().fillna(0.05).values
            ex = m2.all_exits(P, stocks, d, sig)
            nets = np.column_stack([ex[e][0] for e in m2.EXITS])
            okx = ok & np.isfinite(nets).all(1)
        else:
            ex, nets, okx = None, None, ok
        lessons = health(record, d, base)
        sc, C = score(F, okx, lessons)
        if shuffle:
            sc = rng.random(len(stocks))
        lv = F["lv20"].values
        vol_third = okx & (lv >= np.quantile(lv[okx], 2 / 3)) if okx.any() else okx
        hunt = True if shuffle else fear(vixs)
        cand = np.flatnonzero(vol_third)
        picks = list(cand[np.argsort(-sc[cand])[:TOP_N]]) if hunt else []
        if d >= s0 and not complete:
            pending_weeks.append({"date": str(dates[d]), "hunt": hunt, "set_aside": sorted(set(base) - set(lessons))})
            for j in picks:
                pending.append({"date": str(dates[d]), "ticker": stocks[j], "exit": m2.EXITS[EXIT], "sig": 0.0,
                                "why": _why(C[j], lessons) if len(lessons) else [], "dossier": dos.card(stocks[j], d)})
            continue
        if not complete:
            continue
        if d >= s0:
            book = []
            for j in picks:
                tr = {"date": str(dates[d]), "ticker": stocks[j], "exit": m2.EXITS[EXIT], "net": float(nets[j, EXIT]),
                      "closed": str(dates[int(ex[m2.EXITS[EXIT]][1][j])]), "score": float(sc[j])}
                if journal:
                    tr["why"] = _why(C[j], lessons) if len(lessons) else []
                    tr["dossier"] = dos.card(stocks[j], d)
                    tr["by_exit"] = {e: float(nets[j, i]) for i, e in enumerate(m2.EXITS)}
                trades.append(tr)
                book.append(tr["net"])
            weeks.append({"date": str(dates[d]), "net": float(np.mean(book)) if book else 0.0,
                          "random": float(nets[okx, EXIT].mean()), "n": len(book), "gate": bool(hunt),
                          "vix": float(np.exp(vixs[-1])), "vix_median": float(np.exp(np.median(vixs[-FEAR_WEEKS:]))),
                          "volatile_third": float(nets[vol_third, EXIT].mean()),
                          "set_aside": sorted(set(base) - set(lessons))})
        # this week's lessons, scored once its trades finish (for the health check)
        ics = {k: mk.spearman(F[k].values.astype(float)[okx], nets[okx, EXIT]) for k in base}
        record.append((int(d + m2.MAXH - 1), ics))
    return {"trades": trades, "weeks": weeks, "stocks": len(stocks), "shuffled": shuffle, "start": start, "end_data": end_data,
            "learned_from": str(dates[firsts[0]]), "pending": pending, "pending_weeks": pending_weeks,
            "last_date": str(dates[-1]), "lessons": LESSONS if not shuffle else {}}


def _why(c: np.ndarray, lessons: dict, k: int = 3) -> list[str]:
    names = list(lessons)
    top = np.argsort(-c)[:k]
    return [f"{PLAIN[names[i]]} ({c[i]:+.1f} sd)" for i in top]
