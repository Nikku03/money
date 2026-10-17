"""US Treasury daily par yield curve rates, from Treasury's own CSV download.

US government data, public domain, no key. One CSV per calendar year, newest row
first, dates as MM/DD/YYYY. The tenor columns change over the years (2 Mo from
2018, 4 Mo from 2022, 1.5 Month from 2025), so the header is read, not assumed.
We store oldest first with ISO dates and the tenors ordered by maturity.

Par yields are published after each business day's close and are not revised, so
a snapshot is known at the end of the last day it describes.
"""
from __future__ import annotations

import io
import re

import pandas as pd

from brain import series
from brain.bank import Fact
from brain.net import fetch
from brain.timeutil import today

URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
       "{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv")
SOURCE = {"name": "US Treasury",
          "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView"
                 "?type=daily_treasury_yield_curve"}
ATTR = "series.treasury.par_curve"

# The curve the terminal draws. 1.5 Month exists only since 2025 and is left out of it.
TENORS = ["1 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"]
YEARS = [0.083, 0.167, 0.25, 0.333, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]

_TENOR = re.compile(r"([\d.]+)\s*(mo|mos|month|months|yr|yrs|year|years)", re.I)


def tenor_years(label: str) -> float | None:
    """'3 Mo' -> 0.25, '1.5 Month' -> 0.125, '10 Yr' -> 10.0; None for anything else."""
    m = _TENOR.fullmatch(label.strip())
    if not m:
        return None
    n = float(m.group(1))
    return n / 12 if m.group(2).lower().startswith("mo") else n


def parse_csv(body: bytes) -> pd.DataFrame:
    """``date`` (ISO, ascending) plus one float column per tenor in the header; empty cells are NaN."""
    text = body.decode("utf-8-sig")
    if not text.strip():
        return pd.DataFrame({"date": pd.Series(dtype=str)})
    raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    out = pd.DataFrame({"date": pd.to_datetime(raw.iloc[:, 0].str.strip(), format="%m/%d/%Y").dt.strftime("%Y-%m-%d")})
    for c in sorted((c for c in raw.columns[1:] if tenor_years(c) is not None), key=tenor_years):
        out[c] = pd.to_numeric(raw[c].str.strip(), errors="coerce")
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


FIRST_YEAR = 2010   # the link tests (brain/chain/linktest.py) read the 10-year yield from 2010


def ingest(years: int | None = None) -> list[Fact]:
    """Every calendar year from FIRST_YEAR (or the last ``years``) to this one, as one snapshot fact. Past
    years are cached for 30 days, so a daily run fetches the current year."""
    this = int(today()[:4])
    years = this - FIRST_YEAR + 1 if years is None else years
    frames, fetched = [], []
    for y in range(this - years + 1, this + 1):
        r = fetch(URL.format(year=y), ttl=6 * 3600 if y == this else 30 * 86400)
        df = parse_csv(r.body)
        if len(df):
            frames.append(df)
            fetched.append(r.fetched_at)
    if not frames:
        raise ValueError("Treasury returned no yield-curve rows")
    df = pd.concat(frames, ignore_index=True)
    tenors = sorted((c for c in df.columns if c != "date"), key=tenor_years)
    df = df[["date", *tenors]].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    sha = series.store(df)
    first, last = df["date"].iloc[0], df["date"].iloc[-1]
    return [Fact(entity="MACRO", attr=ATTR, value=sha, unit="percent", as_of=last, known_at=last, tier="reported",
                 source=dict(SOURCE), context={"rows": len(df), "first": first, "last": last, "tenors": tenors},
                 retrieved_at=max(fetched))]
