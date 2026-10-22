"""Collect everything public the brain knows how to read about a ticker."""
from __future__ import annotations

import urllib.error

from brain.bank import Bank, Fact
from brain.net import FetchError
from brain.sources import news, prices, sec

PARTS = ("sec", "filings", "prices", "news")


def add_identity(bank: Bank, facts: list[Fact]) -> int:
    """Identity facts are re-read on every ingest; store only changes."""
    n = 0
    for f in facts:
        cur = bank.view().latest(f.entity, f.attr)
        if cur is None or cur.value != f.value:
            n += bank.add_many([f])
    return n


def ingest(bank: Bank, ticker: str, parts=PARTS, all_concepts: bool = False, log=print) -> dict:
    t = ticker.upper()
    counts = {}
    if "sec" in parts or "filings" in parts:
        try:
            sec.cik_for(t)
            has_sec = True
        except KeyError as e:
            log(f"  {t}: {e}; skipping SEC")
            has_sec = False
        if has_sec and "sec" in parts:
            try:
                counts["sec_facts"] = bank.add_many(sec.facts_for(t, all_concepts))
            except FetchError as e:
                # A fund (SPY) has a CIK but files no XBRL statements; its filings and prices still count.
                if not (isinstance(e.__cause__, urllib.error.HTTPError) and e.__cause__.code == 404):
                    raise
                log(f"  {t}: no XBRL financial statements at the SEC (a fund?); skipping them")
                counts["sec_facts"] = 0
        if has_sec and "filings" in parts:
            fil, ident = sec.filings_for(t)
            counts["filings"] = bank.add_many(fil)
            counts["identity"] = add_identity(bank, ident)
    if "prices" in parts:
        try:
            counts["prices"] = bank.add_many(prices.ingest(t))
        except (FetchError, ValueError) as e:
            # A delisted ticker (an acquisition that closed) has no price page, or a page with no bars
            # (prices.fetch_daily: "no price rows"); the other tickers still run.
            gone = (isinstance(e, ValueError) and str(e).startswith("no price rows")) or \
                (isinstance(e.__cause__, urllib.error.HTTPError) and e.__cause__.code == 404)
            if not gone:
                raise
            log(f"  {t}: no price history at the source (delisted?); skipping prices")
            counts["prices"] = 0
    if "news" in parts:
        try:
            counts["headlines"] = bank.add_many(news.ingest(t))
        except Exception as e:  # a feed outage must not stop an ingest
            log(f"  {t}: headlines unavailable ({e})")
    return counts
