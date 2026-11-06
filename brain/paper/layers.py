"""Data layers wired into the trader (P12, docs/PAPER.md): who is buying and who is betting against.

Insiders (``Insiders``)
    Open-market purchases (code P) and sales (code S) by a company's officers, directors and 10%
    owners, from SEC's Form 4/5 data sets, matched by CIK. A trade is visible from the session after its
    filing date. Over the last 63 sessions (about 90 days):
    - ins_buys     purchases;
    - ins_buyers   distinct insiders who bought (a cluster of buyers says more than one);
    - ins_sells    sales;
    - ins_net      log(1 + dollars bought) - log(1 + dollars sold).
    Insiders appear only in their public role, as counts; no name is kept by the trader.

Short-sale volume (``ShortVolume``)
    The share of a stock's FINRA-reported volume that was sold short, from FINRA's daily files (one per
    reporting facility before 2018-08, consolidated after). A day's file is visible from the next
    session. Over the last 5 and 20 sessions, and the change from the last 60:
    - short_r5, short_r20, short_chg.
    A stock's series counts only while FINRA's volume is consistent with its own trading volume (its
    median share of the stock's volume over the last 60 sessions between 3% and 120%). A reused ticker
    cannot then borrow another company's numbers. Otherwise the traits are missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INSIDER = ("ins_buys", "ins_buyers", "ins_sells", "ins_net")
SHORT = ("short_r5", "short_r20", "short_chg")
TRAITS = INSIDER + SHORT
PLAIN = {"ins_buys": "insiders bought in the open market", "ins_buyers": "several insiders bought",
         "ins_sells": "insiders sold", "ins_net": "insiders bought more than they sold",
         "short_r5": "heavily sold short this week", "short_r20": "heavily sold short this month",
         "short_chg": "short selling rising"}
WINDOW = 63


class Insiders:
    def __init__(self, rows: dict, dates: np.ndarray):
        """``rows``: ticker -> [(filed, trade date, code, shares, price, owner cik, relationship), ...]."""
        self.by = {}
        for t, rs in rows.items():
            vis = np.array([np.searchsorted(dates, r[0], side="right") for r in rs])
            buy = np.array([r[2] == "P" for r in rs])
            usd = np.array([(r[3] or 0) * (r[4] or 0) for r in rs], float)
            owner = np.array([str(r[5] or "") for r in rs])
            o = np.argsort(vis, kind="stable")
            self.by[t] = (vis[o], buy[o], usd[o], owner[o])

    def features(self, d: int, stocks: list[str]) -> pd.DataFrame:
        out = np.zeros((len(stocks), len(INSIDER)))
        for i, t in enumerate(stocks):
            if t not in self.by:
                continue
            vis, buy, usd, owner = self.by[t]
            m = (vis > d - WINDOW) & (vis <= d)
            if not m.any():
                continue
            b, s = m & buy, m & ~buy
            out[i] = (b.sum(), len(set(owner[b])), s.sum(), np.log1p(usd[b].sum()) - np.log1p(usd[s].sum()))
        return pd.DataFrame(out, index=stocks, columns=list(INSIDER))


class ShortVolume:
    def __init__(self, rows: dict, P: dict):
        """``rows``: ticker -> [(date, short volume, total volume), ...]; P: the price panels (for volume)."""
        dates = P["close"].index
        self.short, self.total = {}, {}
        for t, rs in rows.items():
            df = pd.DataFrame(rs, columns=["date", "s", "v"]).groupby("date").sum().reindex(dates)
            self.short[t], self.total[t] = df["s"].values, df["v"].values
        self.vol = P["vol"]

    def features(self, d: int, stocks: list[str]) -> pd.DataFrame:
        out = np.full((len(stocks), len(SHORT)), np.nan)
        for i, t in enumerate(stocks):
            if t not in self.short:
                continue
            s, v = self.short[t][d - 60:d], self.total[t][d - 60:d]          # files up to the day before the open
            own = self.vol[t].values[d - 60:d] if t in self.vol else None
            if own is None or np.isfinite(v).sum() < 40:
                continue
            share = np.nanmedian(v / np.where(own > 0, own, np.nan))
            if not (0.03 <= share <= 1.2):
                continue
            r = lambda k: np.nansum(s[-k:]) / max(1.0, np.nansum(v[-k:]))
            out[i] = (r(5), r(20), r(5) - r(60))
        return pd.DataFrame(out, index=stocks, columns=list(SHORT))
