"""SEC EDGAR: identity, filings list, and XBRL financial facts.

All public-domain US government data. known_at for a financial fact is the date of
the filing that first reported that value; a later filing reporting a different
value for the same period is kept as a restatement.
"""
from __future__ import annotations

import json

from brain import config
from brain.bank import Fact
from brain.net import fetch
from brain.timeutil import days_between

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/"

# Curated: enough for growth, margins, valuation, balance-sheet health, cash use.
# Everything else in companyfacts is still one flag away (--all-concepts).
CONCEPTS = {
    "us-gaap": [
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
        "CostOfRevenue", "GrossProfit", "OperatingIncomeLoss", "NetIncomeLoss",
        "EarningsPerShareBasic", "EarningsPerShareDiluted",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent", "StockholdersEquity",
        "CashAndCashEquivalentsAtCarryingValue", "LongTermDebt", "LongTermDebtNoncurrent",
        "NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment",
        "ResearchAndDevelopmentExpense", "PaymentsForRepurchaseOfCommonStock", "PaymentsOfDividends",
        "CommonStockDividendsPerShareDeclared",
    ],
    "dei": ["EntityCommonStockSharesOutstanding", "EntityPublicFloat"],
}
REVENUE = ["us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:SalesRevenueNet"]


def _get_json(url: str, ttl: float = 86400):
    r = fetch(url, user_agent=config.sec_user_agent(), ttl=ttl)
    return json.loads(r.body), r.fetched_at


def ticker_table() -> list[dict]:
    """SEC's ticker list, in file order (roughly descending market value)."""
    data, _ = _get_json(TICKERS_URL, ttl=7 * 86400)
    return [{"ticker": v["ticker"].upper(), "cik": int(v["cik_str"]), "title": v["title"]}
            for _, v in sorted(data.items(), key=lambda kv: int(kv[0]))]


def cik_for(ticker: str) -> int:
    t = ticker.upper()
    for row in ticker_table():
        if row["ticker"] == t:
            return row["cik"]
    raise KeyError(f"{ticker} is not in SEC's ticker list (foreign listings and ETFs often are not)")


def parse_companyfacts(ticker: str, data: dict, fetched_at: str, all_concepts: bool = False) -> list[Fact]:
    cik = int(data["cik"])
    out = []
    for ns, concepts in data.get("facts", {}).items():
        wanted = concepts.keys() if all_concepts else [c for c in CONCEPTS.get(ns, []) if c in concepts]
        for concept in wanted:
            for unit, entries in concepts[concept].get("units", {}).items():
                groups: dict[tuple, list] = {}
                for e in entries:
                    groups.setdefault((e.get("start"), e["end"]), []).append(e)
                for (start, end), es in groups.items():
                    # A period that ends after its own filing date is a tagging error in the source.
                    es = sorted((e for e in es if e["end"] <= e["filed"]), key=lambda e: (e["filed"], e["accn"]))
                    last_val = object()
                    for e in es:
                        if e["val"] == last_val:
                            continue  # a later filing repeating a known value adds nothing
                        last_val = e["val"]
                        ctx = {"form": e.get("form"), "fy": e.get("fy"), "fp": e.get("fp"), "accn": e["accn"]}
                        if start:
                            ctx["start"] = start
                            ctx["duration_days"] = days_between(start, end)
                        out.append(Fact(
                            entity=ticker, attr=f"{ns}:{concept}", value=e["val"], unit=unit,
                            as_of=end, known_at=e["filed"], tier="filed",
                            source={"name": "SEC EDGAR XBRL",
                                    "url": ARCHIVE.format(cik=cik, accn=e["accn"].replace("-", ""))},
                            context=ctx, retrieved_at=fetched_at,
                        ))
    return out


def parse_submissions(ticker: str, data: dict, fetched_at: str) -> tuple[list[Fact], dict]:
    cik = int(data["cik"])
    r = data["filings"]["recent"]
    facts = []
    for i in range(len(r["accessionNumber"])):
        accn = r["accessionNumber"][i]
        doc = r["primaryDocument"][i]
        url = ARCHIVE.format(cik=cik, accn=accn.replace("-", "")) + doc
        ctx = {"accn": accn}
        if r.get("items") and r["items"][i]:
            ctx["items"] = r["items"][i]
        if r.get("primaryDocDescription") and r["primaryDocDescription"][i]:
            ctx["description"] = r["primaryDocDescription"][i]
        # Accepted after hours, a filing carries the next business day's date and is public from then.
        # known_at is the later of acceptance and the official filing date.
        filing_date = r["filingDate"][i]
        accepted = r["acceptanceDateTime"][i]
        known = accepted if accepted and accepted[:10] >= filing_date else filing_date
        as_of = min(r["reportDate"][i] or filing_date, filing_date)
        facts.append(Fact(
            entity=ticker, attr="filing", value=r["form"][i], as_of=as_of, known_at=known,
            tier="filed", source={"name": "SEC EDGAR", "url": url}, context=ctx, retrieved_at=fetched_at,
        ))
    identity = {
        "identity.name": data.get("name"),
        "identity.cik": cik,
        "identity.sic": data.get("sic"),
        "identity.industry": data.get("sicDescription"),
        "identity.exchanges": ",".join(data.get("exchanges") or []),
        "identity.fiscal_year_end": data.get("fiscalYearEnd"),
        "identity.website": data.get("website") or None,
    }
    return facts, {k: v for k, v in identity.items() if v not in (None, "")}


def facts_for(ticker: str, all_concepts: bool = False) -> list[Fact]:
    cik = cik_for(ticker)
    data, fetched_at = _get_json(FACTS_URL.format(cik=cik))
    return parse_companyfacts(ticker, data, fetched_at, all_concepts)


def filings_for(ticker: str) -> tuple[list[Fact], list[Fact]]:
    cik = cik_for(ticker)
    url = SUBS_URL.format(cik=cik)
    data, fetched_at = _get_json(url)
    facts, ident = parse_submissions(ticker, data, fetched_at)
    id_facts = [Fact(entity=ticker, attr=k, value=v, as_of=fetched_at[:10], known_at=fetched_at,
                     tier="filed", source={"name": "SEC EDGAR submissions", "url": url},
                     retrieved_at=fetched_at) for k, v in ident.items()]
    return facts, id_facts
