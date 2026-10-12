"""Macro series from FRED (St. Louis Fed), via its official API. Context for briefs only.

Needs a free key: https://fred.stlouisfed.org/docs/api/api_key.html -> export FRED_API_KEY=...
(The fredgraph.csv download link rejects scripted clients; we use the sanctioned API instead.)

These are latest-vintage values: CPI, payrolls and GDP get revised, so they are not
point-in-time and must not feed a backtest. Market rates (DGS10 etc.) are not revised.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from brain import series
from brain.bank import Fact
from brain.net import fetch

URL = "https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}&file_type=json"
DEFAULT = {
    "DGS10": "10-year Treasury yield, %",
    "DGS3MO": "3-month Treasury yield, %",
    "T10Y2Y": "10y minus 2y Treasury spread, %",
    "FEDFUNDS": "Effective federal funds rate, %",
    "CPIAUCSL": "CPI, all urban consumers (index; revised)",
    "UNRATE": "Unemployment rate, % (revised)",
    "BAMLH0A0HYM2": "US high-yield option-adjusted spread, %",
}
REVISED = {"CPIAUCSL", "UNRATE"}


def ingest(sid: str) -> list[Fact]:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise SystemExit("FRED needs a free API key: https://fred.stlouisfed.org/docs/api/api_key.html\n"
                         "  export FRED_API_KEY=...")
    r = fetch(URL.format(sid=sid, key=key), ttl=12 * 3600)
    obs = json.loads(r.body)["observations"]
    df = pd.DataFrame({"date": [o["date"] for o in obs], "value": [o["value"] for o in obs]})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    sha = series.store(df)
    last = df["date"].iloc[-1]
    return [Fact(entity="MACRO", attr=f"series.fred.{sid}", value=sha, as_of=last, known_at=r.fetched_at,
                 tier="reported", source={"name": "FRED", "url": f"https://fred.stlouisfed.org/series/{sid}"},
                 context={"label": DEFAULT.get(sid, sid), "rows": len(df), "first": df["date"].iloc[0],
                          "last": last, "point_in_time": sid not in REVISED},
                 retrieved_at=r.fetched_at)]
