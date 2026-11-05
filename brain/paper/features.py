"""What the trader can see on a decision day, built only from data dated on or before it.

All features are measured on the decision day's close and ranked across the stocks (cross-sectional
z-scores), so the trader compares stocks with each other, not with the past.

    own       mom_1w, mom_1m, mom_12_1: the stock's return in excess of SPY over the last 5 and 21
              sessions, and over the year before last month; vol_1m: its daily volatility; from_high:
              how far below its 52-week high it sits.
    dots      peers_1w: the last week's excess return of the stock's five nearest neighbours, the
              stocks it moved with most over the past year (measured, not assumed);
              oil_link, rates_link, dollar_link, market_link: the stock's sensitivity (beta, past
              year of daily returns) to oil, bonds, the dollar and the market, times how that asset
              moved in the last week. A stock that tracks oil, after oil jumped, scores high on oil_link.
    news      (live only: the headline feed has no history) counts of tagged headlines in the last
              day, by rule (news.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = ("mom_1w", "mom_1m", "mom_12_1", "vol_1m", "from_high", "peers_1w",
                  "oil_link", "rates_link", "dollar_link", "market_link")
LINKS = {"oil_link": "USO", "rates_link": "TLT", "dollar_link": "UUP", "market_link": "SPY"}
YEAR = 252


def zscore(x: pd.Series) -> pd.Series:
    s = x.std()
    return (x - x.mean()) / s if s and np.isfinite(s) else x * 0.0


def at(px: pd.DataFrame, i: int, stocks: list[str]) -> pd.DataFrame:
    """Features for ``stocks`` on row ``i`` of the closes table (that day's close), z-scored across stocks.
    Uses rows <= i only."""
    if i < YEAR + 21:
        raise ValueError("needs a year and a month of history")
    win = px.iloc[i - YEAR:i + 1]
    ret = win.pct_change().iloc[1:]
    spy = ret["SPY"]
    exc = ret[stocks].sub(spy, axis=0)
    p = px.iloc[: i + 1]
    f = pd.DataFrame(index=stocks)
    f["mom_1w"] = p[stocks].iloc[-1] / p[stocks].iloc[-6] - (p["SPY"].iloc[-1] / p["SPY"].iloc[-6])
    f["mom_1m"] = p[stocks].iloc[-1] / p[stocks].iloc[-22] - (p["SPY"].iloc[-1] / p["SPY"].iloc[-22])
    f["mom_12_1"] = p[stocks].iloc[-22] / p[stocks].iloc[-YEAR] - (p["SPY"].iloc[-22] / p["SPY"].iloc[-YEAR])
    f["vol_1m"] = ret[stocks].iloc[-21:].std()
    f["from_high"] = p[stocks].iloc[-1] / p[stocks].iloc[-YEAR:].max() - 1
    corr = exc.corr()
    week = exc.iloc[-5:].sum()
    peers = {}
    for s in stocks:
        near = corr[s].drop(s).dropna().nlargest(5).index
        peers[s] = week[near].mean() if len(near) else 0.0
    f["peers_1w"] = pd.Series(peers)
    for name, asset in LINKS.items():
        a = ret[asset]
        beta = ret[stocks].apply(lambda c: np.cov(c, a)[0, 1] / a.var() if a.var() > 0 else 0.0)
        f[name] = beta * a.iloc[-5:].sum()
    return f.astype(float).fillna(0.0).apply(zscore)


def forward(px: pd.DataFrame, i: int, stocks: list[str], horizon: int = 5) -> pd.Series:
    """Each stock's return in excess of SPY from row i's close to row i+horizon's close (the outcome)."""
    a, b = px.iloc[i], px.iloc[i + horizon]
    return (b[stocks] / a[stocks] - 1) - (b["SPY"] / a["SPY"] - 1)
