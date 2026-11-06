"""What every version got right and wrong (the review behind P11, docs/PAPER.md).

``account``   a $1,000 account from a run's trades: each new trade takes 1/6 of the account's value
              (3 a week, held up to about two weeks), gains and losses compounding: the method every
              account figure in docs/PAPER.md uses.
``situations`` every tradable stock in every week of a period, with what each version could see
              (prices, filings, the dossier, its industry, its forces) and how each of the five exits
              would have turned out: the ground truth that all versions were trying to learn.
``lessons``   for each trait, how it related to each exit's outcome week by week (rank correlation),
              in each period separately. A lesson that is real should hold in both periods; one that
              flips sign was something a version learned from one period's noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from brain.paper import dossier as ds
from brain.paper import market as mk
from brain.paper import moonshot as m1
from brain.paper import moonshot2 as m2
from brain.paper import moonshot3 as m3

TRAITS = (list(m1.CONT) + ["pen", "offer30", "earn_win", "bio"] + list(ds.FEATURES) + ["ind_ret5", "ind_ret20"]
          + ["own5", "force5", "own20"] + list(mk.SENS))
STATE = ("vix", "vix_term", "spy_dd", "breadth", "list20", "moon", "jumpy", "SPY_20", "credit_20")


def account(trades: list[dict], start: float = 1000.0, slots: int = 6) -> tuple[float, float]:
    cash, op, eqs = start, [], []
    for t in sorted(trades, key=lambda t: t["date"]):
        for o in [o for o in op if o[0] < t["date"]]:
            cash += o[1] * (1 + o[2])
            op.remove(o)
        eq = cash + sum(o[1] for o in op)
        eqs.append(eq)
        stake = min(eq / slots, cash)
        cash -= stake
        op.append((t.get("closed") or t.get("exit"), stake, t["net"]))
    for o in op:
        cash += o[1] * (1 + o[2])
    e = np.array(eqs) if eqs else np.array([start])
    return float(cash), float((e / np.maximum.accumulate(e) - 1).min())


def situations(raw: dict, sec: dict, facts: dict, stocks: list[str], start: str, end_data: str | None = None) -> dict:
    """Per week from ``start``: the tradable stocks' traits, the week's market state, and every exit's
    net (with whether E0 hit +20% or was stopped)."""
    if end_data:
        raw = {t: df[df["date"].astype(str) <= end_data] for t, df in raw.items()}
    P = m1.panels(raw)
    dates = P["close"].index.values
    stocks = [s for s in stocks if s in P["close"] and s not in mk.NEEDED and P["close"][s].iloc[:260].notna().sum() > 200]
    events = m1.filing_events(sec, dates)
    dos = ds.Dossier({t: facts["facts"].get(t, []) for t in stocks}, facts["names"], P, events, sec.get("sic", {}))
    sic2 = np.array([str(sec.get("sic", {}).get(t) or "")[:2] for t in stocks])
    week = pd.Series(pd.to_datetime(dates)).dt.to_period("W").values
    s0 = int(np.searchsorted(dates, start))
    firsts = [i for i in range(1, len(dates)) if week[i] != week[i - 1] and i >= max(261, s0) and i + m2.MAXH <= len(dates)]
    mm = mk.MarketModel(P, stocks, events)
    out = []
    for d in firsts:
        F = m1.panel(P, stocks, events, sec.get("sic", {}), d)
        ok = m1.tradable(P, stocks, d) & np.isfinite(F.values).all(1)
        sig = np.log(P["close"][stocks].iloc[d - 21:d]).diff().iloc[1:].std().fillna(0.05).values
        ex = m2.all_exits(P, stocks, d, sig)
        nets = np.column_stack([ex[e][0] for e in m2.EXITS])
        ok &= np.isfinite(nets).all(1)
        E = dos.features(d, stocks)
        for c in ds.FEATURES:
            F[c] = E[c].values
        for c, v in m3.industry(F, sic2, ok).items():
            F[c] = v
        B, b = mm.exposures(d)
        f5, o5 = mm.decompose(d, B, b, 5)
        _, o20 = mm.decompose(d, B, b, 20)
        F["own5"], F["force5"], F["own20"] = o5, f5, o20
        for c, a in mk.SENS.items():
            F[c] = B[mm.tf.index(a)]
        s = dict(zip(mk.STATE, mm.state(d, ok, F["lv20"].values)))
        e0 = nets[:, 0]
        out.append({"date": str(dates[d]), "tickers": [stocks[j] for j in np.flatnonzero(ok)],
                    "X": F[list(TRAITS)].values[ok].astype(np.float32), "nets": nets[ok].astype(np.float32),
                    "hit": (e0[ok] >= m1.TARGET - 2 * m1.COST - 1e-9), "stop": (e0[ok] <= -m1.STOP - 2 * m1.COST + 1e-9),
                    "state": {k: float(s[k]) for k in STATE}})
    return {"weeks": out, "traits": list(TRAITS), "exits": list(m2.EXITS)}


def _ic(x, y):
    if len(x) < 30 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def lessons(sit: dict, outcome: int = 0) -> pd.DataFrame:
    """Each trait's weekly rank correlation with an exit's net (``outcome`` indexes EXITS), and with
    hitting +20% and being stopped: mean, t across weeks, and the share of weeks it was positive."""
    rows = []
    for k, name in enumerate(sit["traits"]):
        ic = np.array([_ic(w["X"][:, k], w["nets"][:, outcome]) for w in sit["weeks"]])
        ih = np.array([_ic(w["X"][:, k], w["hit"].astype(float)) for w in sit["weeks"]])
        ist = np.array([_ic(w["X"][:, k], w["stop"].astype(float)) for w in sit["weeks"]])
        ic, ih, ist = ic[np.isfinite(ic)], ih[np.isfinite(ih)], ist[np.isfinite(ist)]
        t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))) if len(ic) > 2 else np.nan
        rows.append({"trait": name, "ic": ic.mean(), "t": t, "pos": float((ic > 0).mean()),
                     "ic_hit": ih.mean(), "ic_stop": ist.mean()})
    return pd.DataFrame(rows).set_index("trait")
