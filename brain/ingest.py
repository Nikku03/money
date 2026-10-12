"""Collect everything public the brain knows how to read about a ticker."""
from __future__ import annotations

from brain.bank import Bank, Fact
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
            counts["sec_facts"] = bank.add_many(sec.facts_for(t, all_concepts))
        if has_sec and "filings" in parts:
            fil, ident = sec.filings_for(t)
            counts["filings"] = bank.add_many(fil)
            counts["identity"] = add_identity(bank, ident)
    if "prices" in parts:
        counts["prices"] = bank.add_many(prices.ingest(t))
    if "news" in parts:
        try:
            counts["headlines"] = bank.add_many(news.ingest(t))
        except Exception as e:  # a feed outage must not stop an ingest
            log(f"  {t}: headlines unavailable ({e})")
    return counts
