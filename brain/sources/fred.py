"""Macro series from FRED (St. Louis Fed), via its official API. Context for briefs only.

Needs a free key: https://fred.stlouisfed.org/docs/api/api_key.html -> export FRED_API_KEY=...
(The fredgraph.csv download link rejects scripted clients; we use the sanctioned API instead.)
``python -m brain macro`` runs :func:`ingest_all`: the Treasury yield curve always (no key
needed, ``brain/sources/treasury.py``), FRED only when a key is set. The key never leaves
this module unredacted: ``brain.net`` strips it from cached URLs and error messages.

These are latest-vintage values: CPI, payrolls and GDP get revised, so they are not
point-in-time and must not feed a backtest. Market rates (DGS10 etc.) are not revised.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from brain import series
from brain.bank import Bank, Fact
from brain.net import FetchError, fetch

URL = "https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}&file_type=json"
KEY_HELP = "https://fred.stlouisfed.org/docs/api/api_key.html"
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


def api_key() -> str:
    return os.environ.get("FRED_API_KEY", "").strip()


def ingest(sid: str) -> list[Fact]:
    key = api_key()
    if not key:
        raise SystemExit(f"FRED needs a free API key: {KEY_HELP}\n  export FRED_API_KEY=...")
    r = fetch(URL.format(sid=sid, key=key), ttl=12 * 3600)
    data = json.loads(r.body)
    if "observations" not in data:  # an API error body; its message never echoes the key, but make sure
        raise ValueError(f"FRED {sid}: {str(data.get('error_message', 'no observations')).replace(key, '***')}")
    obs = data["observations"]
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


def ingest_all(bank: Bank, sids: list[str] | None = None, log=print) -> dict:
    """Treasury curve always; FRED series only with FRED_API_KEY. A part that fails is logged and
    skipped. Returns {"added": {part: n}, "failed": {part: reason}, "fred_skipped": bool}."""
    from brain.sources import treasury
    added: dict[str, int] = {}
    failed: dict[str, str] = {}
    try:
        added["treasury"] = bank.add_many(treasury.ingest())
        log(f"treasury: {added['treasury']} new facts")
    except (FetchError, ValueError, KeyError) as e:
        failed["treasury"] = str(e)
        log(f"treasury: failed ({e})")
    if not api_key():
        log(f"FRED skipped: FRED_API_KEY is not set (free key: {KEY_HELP})")
        return {"added": added, "failed": failed, "fred_skipped": True}
    for sid in sids or list(DEFAULT):
        try:
            added[sid] = bank.add_many(ingest(sid))
            log(f"{sid}: {added[sid]} new facts")
        except (FetchError, ValueError, KeyError, IndexError) as e:
            failed[sid] = str(e)
            log(f"{sid}: failed ({e})")
    return {"added": added, "failed": failed, "fred_skipped": False}
