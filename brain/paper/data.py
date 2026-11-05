"""Daily adjusted closes for the trader's universe and the cross-market assets it watches."""
from __future__ import annotations

import json

import pandas as pd

from brain import config
from brain.sources import prices

# Cross-market assets: the "dots" outside the stock list. Each is an ETF with history since 2010.
MACRO = {"SPY": "the market", "USO": "oil", "TLT": "long-term Treasury bonds (rates, inverted)", "UUP": "the US dollar"}


def universe() -> list[str]:
    return json.loads(config.UNIVERSE.read_text())["tickers"]


def closes(tickers: list[str], start: str = "2014-01-01") -> pd.DataFrame:
    """Adjusted closes (splits and dividends), one column per ticker, on the dates SPY traded."""
    cols = {}
    for t in dict.fromkeys(["SPY", *tickers]):
        df = prices.fetch_daily(t, start=start)[0]
        cols[t] = df.set_index("date")["adjclose"].astype(float)
    out = pd.DataFrame(cols)
    out = out.loc[out["SPY"].notna()]
    return out.sort_index()
