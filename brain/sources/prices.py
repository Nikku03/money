"""Daily prices from Yahoo Finance's chart endpoint.

Unofficial and for personal research: Yahoo's terms forbid redistribution, so the
CSV snapshots stay in the gitignored home directory and are never committed.

``close`` is adjusted for splits only; ``adjclose`` for splits and dividends
(use it for returns). Both are back-adjusted to today's share basis.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd

from brain import series
from brain.bank import Fact
from brain.net import fetch

CHART = ("https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
         "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit&includeAdjustedClose=true")


def parse_chart(data: dict, now: float | None = None) -> tuple[pd.DataFrame, list[tuple[str, float]], dict]:
    res = data["chart"]["result"][0]
    meta = res["meta"]
    off = int(meta.get("gmtoffset", 0))
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or q["close"]
    df = pd.DataFrame({
        "date": [datetime.fromtimestamp(t + off, timezone.utc).strftime("%Y-%m-%d") for t in ts],
        "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
        "adjclose": adj, "volume": q["volume"],
    })
    df = df.dropna(subset=["close", "adjclose"]).drop_duplicates("date", keep="last").reset_index(drop=True)
    # Drop today's bar while the session is still open: it is not a close yet.
    reg = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    now = time.time() if now is None else now
    if len(df) and reg and reg.get("start", 0) <= now < reg.get("end", 0):
        today = datetime.fromtimestamp(reg["start"] + off, timezone.utc).strftime("%Y-%m-%d")
        if df["date"].iloc[-1] == today:
            df = df.iloc[:-1]
    splits = []
    for s in (res.get("events") or {}).get("splits", {}).values():
        d = datetime.fromtimestamp(int(s["date"]) + off, timezone.utc).strftime("%Y-%m-%d")
        splits.append((d, float(s["numerator"]) / float(s["denominator"])))
    splits.sort()
    keep = {k: meta.get(k) for k in ("currency", "exchangeName", "instrumentType", "longName", "exchangeTimezoneName")}
    return df, splits, keep


def fetch_daily(symbol: str, start: str = "2000-01-01"):
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    p2 = (int(time.time()) // 86400 + 1) * 86400  # next UTC midnight: same URL all day, so the cache works
    url = CHART.format(sym=symbol.upper(), p1=p1, p2=p2)
    r = fetch(url, ttl=6 * 3600)
    # Judge "session still open" at the time the body was fetched, not now: a body cached at midday and
    # re-read after the close still holds that midday bar, which is not a close.
    fetched = datetime.fromisoformat(r.fetched_at.replace("Z", "+00:00")).timestamp()
    df, splits, meta = parse_chart(json.loads(r.body), now=fetched)
    if df.empty:
        raise ValueError(f"no price rows for {symbol}")
    return df, splits, meta, r


def ingest(symbol: str, start: str = "2000-01-01") -> list[Fact]:
    df, splits, meta, r = fetch_daily(symbol, start)
    sha = series.store(df)
    src = {"name": "Yahoo Finance chart API (unofficial)", "url": f"https://finance.yahoo.com/quote/{symbol.upper()}/history"}
    out = [Fact(entity=symbol, attr="series.price_daily", value=sha, as_of=df["date"].iloc[-1],
                known_at=df["date"].iloc[-1], tier="market", source=src,
                context={"rows": len(df), "first": df["date"].iloc[0], "last": df["date"].iloc[-1],
                         "basis": "close=split-adjusted; adjclose=split+dividend-adjusted, today's share basis",
                         "splits": [[d, r_] for d, r_ in splits], **{k: v for k, v in meta.items() if v}},
                retrieved_at=r.fetched_at)]
    for d, ratio in splits:
        out.append(Fact(entity=symbol, attr="split", value=ratio, as_of=d, known_at=d, tier="market",
                        source=src, retrieved_at=r.fetched_at))
    return out
