"""Build the research universe by a fixed, written rule, and freeze it in git.

The rule uses today's market values, so every member is a survivor. That bias is
stated in every result rather than hidden; see research/RESULTS.md.

Two universes: ``us_large.json`` (the first 60 kept; frozen, H1-H5 were tested on
it) and ``us_broad.json`` (every CIK among the first 1,500 rows of SEC's list that
meets the same rule), where less-watched stocks give slow-priced information a
chance to show.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from brain import config
from brain.bank import Bank
from brain.ingest import add_identity
from brain.net import FetchError
from brain.sources import prices, sec
from brain.timeutil import now_iso

PRICE_START_BY = "2010-01-04"
BROAD = config.ROOT / "universe" / "us_broad.json"
BROAD_RULE_SHORT = "SEC ranks 1-1500 with 10-K XBRL and prices since 2010"


class Rejected(Exception):
    """A ticker that fails the rule. ``transient``: the source gave up (throttled, down), so the
    rule could not be applied; the broad build waits and asks again rather than reject on that."""

    def __init__(self, reason: str, transient: bool = False):
        super().__init__(reason)
        self.transient = transient


def _transient(e: Exception) -> bool:
    return isinstance(e, FetchError) and "gave up" in str(e)


def rule_text(n: int) -> str:
    return (f"SEC company_tickers.json in file order (roughly descending market value); first ticker per CIK; "
            f"kept if SEC XBRL has us-gaap NetIncomeLoss and Assets from 10-K filings and Yahoo daily prices "
            f"start on or before {PRICE_START_BY}; the first {n} kept.")


def broad_rule_text(max_rank: int) -> str:
    return (f"SEC company_tickers.json in file order (roughly descending market value); the first {max_rank} "
            f"rows; first ticker per CIK; kept if SEC XBRL has us-gaap NetIncomeLoss and Assets from 10-K filings "
            f"and Yahoo daily prices start on or before {PRICE_START_BY}; every CIK that qualifies is kept.")


def qualify(t: str) -> tuple[list, list]:
    """The rule for one ticker: its SEC facts and Yahoo prices, or ``Rejected`` with the reason."""
    try:
        facts = sec.facts_for(t)
    except FetchError as e:
        raise Rejected(f"no SEC companyfacts ({e})", _transient(e))
    tenk = {f.attr for f in facts if str(f.context.get("form", "")).startswith("10-K")}
    lacking = [c for c in ("us-gaap:NetIncomeLoss", "us-gaap:Assets") if c not in tenk]
    if lacking:
        raise Rejected(f"no 10-K {', '.join(lacking)} in SEC XBRL under this CIK "
                       f"(foreign filer, other taxonomy, or new CIK)")
    try:
        px = prices.ingest(t)
    except (FetchError, ValueError, KeyError) as e:
        raise Rejected(f"no Yahoo prices ({e})", _transient(e))
    first = px[0].context["first"]
    if first > PRICE_START_BY:
        raise Rejected(f"prices start {first}")
    return facts, px


def _store(bank: Bank, t: str, facts: list, px: list, all_pages: bool = False) -> int:
    bank.add_many(facts)
    bank.add_many(px)
    fil, ident = sec.filings_for(t, all_pages=all_pages)
    # Oldest first, so each fact lands at the end of its attribute's sorted list (cheap inserts).
    n = bank.add_many(sorted(fil, key=lambda f: (f.known_at, f.as_of)))
    add_identity(bank, ident)
    return n


def _listing_sha(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]


def build(bank: Bank, n: int = 60, log=print) -> dict:
    rows = sec.ticker_table()
    listing_sha = _listing_sha(rows)
    seen, kept, rejected = set(), [], []
    for row in rows:
        if len(kept) >= n:
            break
        if row["cik"] in seen:
            continue
        seen.add(row["cik"])
        t = row["ticker"]
        try:
            facts, px = qualify(t)
        except Rejected as e:
            rejected.append({"ticker": t, "reason": str(e)})
            continue
        _store(bank, t, facts, px)
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


def _qualify_patiently(t: str, waits=(30, 120, 300), log=print, sleep=time.sleep) -> tuple[list, list]:
    for w in (*waits, None):
        try:
            return qualify(t)
        except Rejected as e:
            if not e.transient or w is None:
                raise
            log(f"  {t}: {e}; retrying in {w}s")
            sleep(w)


def build_broad(bank: Bank, max_rank: int = 1500, log=print, path: Path | None = None) -> dict:
    """Every CIK among the first ``max_rank`` rows of SEC's ticker list that meets the rule, with
    its SEC facts, every filing page and its prices ingested. Each ticker is written through its
    own Bank on the same journal, so memory stays at one company's facts however long the list."""
    rows = sec.ticker_table()
    listing_sha = _listing_sha(rows)
    seen, kept, rejected = set(), [], []
    t0 = time.time()
    for rank, row in enumerate(rows[:max_rank], 1):
        if row["cik"] in seen:
            continue
        seen.add(row["cik"])
        t = row["ticker"]
        try:
            facts, px = _qualify_patiently(t, log=log)
        except Rejected as e:
            rejected.append({"ticker": t, "reason": str(e)})
            log(f"  [{rank}/{max_rank}] reject {t:<6} {e}")
            continue
        except Exception as e:  # one bad response must not end a two-hour build
            rejected.append({"ticker": t, "reason": f"error reading sources ({type(e).__name__}: {e})"})
            log(f"  [{rank}/{max_rank}] reject {t:<6} error {type(e).__name__}: {e}")
            continue
        try:
            nf = _store(Bank(bank.root, use_index=bank.use_index), t, facts, px, all_pages=True)
        except Exception as e:  # kept by the rule; its filings can be re-read with `brain ingest`
            nf = f"filings failed ({type(e).__name__}: {e})"
        kept.append(t)
        log(f"  [{rank}/{max_rank}] kept {t:<6} ({len(kept)} kept, {len(rejected)} rejected, "
            f"{time.time() - t0:.0f}s) filings added: {nf}")
    bank.add_many(prices.ingest("SPY"))
    out = {
        "built_at": now_iso(), "rule": broad_rule_text(max_rank),
        "rule_short": BROAD_RULE_SHORT if max_rank == 1500 else f"SEC ranks 1-{max_rank} with 10-K XBRL and prices since 2010",
        "sec_listing_sha": listing_sha, "n": len(kept), "tickers": kept, "rejected": rejected,
        "reference": "SPY (calendar and context only; not in the universe)",
        "bias": "survivorship: chosen by 2026 market value, all members listed since 2010",
    }
    p = Path(path) if path else BROAD
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    return out
