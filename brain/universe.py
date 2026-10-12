"""Build the research universe by a fixed, written rule, and freeze it in git.

The rule uses today's market values, so every member is a survivor. That bias is
stated in every result rather than hidden; see research/RESULTS.md.
"""
from __future__ import annotations

import hashlib
import json

from brain import config
from brain.bank import Bank
from brain.ingest import add_identity
from brain.net import FetchError
from brain.sources import prices, sec
from brain.timeutil import now_iso

PRICE_START_BY = "2010-01-04"


def rule_text(n: int) -> str:
    return (f"SEC company_tickers.json in file order (roughly descending market value); first ticker per CIK; "
            f"kept if SEC XBRL has us-gaap NetIncomeLoss and Assets from 10-K filings and Yahoo daily prices "
            f"start on or before {PRICE_START_BY}; the first {n} kept.")


def build(bank: Bank, n: int = 60, log=print) -> dict:
    rows = sec.ticker_table()
    listing_sha = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
    seen, kept, rejected = set(), [], []
    for row in rows:
        if len(kept) >= n:
            break
        if row["cik"] in seen:
            continue
        seen.add(row["cik"])
        t = row["ticker"]
        try:
            facts = sec.facts_for(t)
        except FetchError as e:
            rejected.append({"ticker": t, "reason": f"no SEC companyfacts ({e})"})
            continue
        tenk = {f.attr for f in facts if str(f.context.get("form", "")).startswith("10-K")}
        lacking = [c for c in ("us-gaap:NetIncomeLoss", "us-gaap:Assets") if c not in tenk]
        if lacking:
            rejected.append({"ticker": t, "reason": f"no 10-K {', '.join(lacking)} in SEC XBRL under this CIK "
                                                    f"(foreign filer, other taxonomy, or new CIK)"})
            continue
        try:
            px = prices.ingest(t)
        except (FetchError, ValueError, KeyError) as e:
            rejected.append({"ticker": t, "reason": f"no Yahoo prices ({e})"})
            continue
        first = px[0].context["first"]
        if first > PRICE_START_BY:
            rejected.append({"ticker": t, "reason": f"prices start {first}"})
            continue
        bank.add_many(facts)
        bank.add_many(px)
        fil, ident = sec.filings_for(t)
        bank.add_many(fil)
        add_identity(bank, ident)
        kept.append(t)
        log(f"  kept {t:<6} ({len(kept)}/{n})")
    bank.add_many(prices.ingest("SPY"))
    out = {
        "built_at": now_iso(), "rule": rule_text(n),
        "rule_short": "top SEC filers by market value with 10-K XBRL and prices since 2010",
        "sec_listing_sha": listing_sha, "n": len(kept), "tickers": kept, "rejected": rejected,
        "reference": "SPY (calendar and context only; not in the universe)",
        "bias": "survivorship: chosen by 2026 market value, all members listed since 2010",
    }
    config.UNIVERSE.parent.mkdir(parents=True, exist_ok=True)
    config.UNIVERSE.write_text(json.dumps(out, indent=1) + "\n")
    return out
