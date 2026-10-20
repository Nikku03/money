"""Event tags by fixed rules: no model (docs/WEB.md).

A headline or summary is matched against a short list of patterns, one per kind of corporate event. A tag says
only that the words appear: "raises guidance" in a headline tags ``guidance_raised``. It implies no direction for
the stock and weighs nothing; what an event type does to prices is for a declared test to measure.
"""
from __future__ import annotations

import re

_W = r"(?:its |the |full[- ]year |annual |quarterly |fiscal |\w+ )?"
# Up to five words between a verb and "guidance", as press releases write it ("Raises Full-Year 2026 Revenue and EPS
# Outlook"), but never across a clause break, past a word that says the guidance did not move ("Raises Dividend;
# Reaffirms Guidance"), onto guidance for spending, costs or taxes, which is not a forecast of the business, or onto
# a feeling about the outlook ("increasing confidence in our outlook").
_GAP = (r"(?:(?!(?:re)?affirm|reiterat|maintain|confirm|narrow|dividend|unchanged|capital|capex|expens|cost|spend|tax|"
        r"confiden)[\w$%&/'’()-]+,?\s+){0,5}")
RULES: dict[str, str] = {
    # Not "price increases", a noun: "announced price increases fully offset the current forecast".
    "guidance_raised": rf"(?<!price )(?<!prices )\b(raises?|raised|raising|increases?|increased|increasing|lifts?|"
                       rf"lifted|boosts?|boosted|ups|upped|hikes?|hiked) {_GAP}(guidance|outlook|forecast)",
    "guidance_cut": rf"\b(cuts|cutting|lowers|lowered|lowering|slashes|slashed|reduces|reduced|reducing|trims|"
                    rf"trimmed) {_GAP}(guidance|outlook|forecast)|\bprofit warning\b",   # verbs only: not "lower than forecast"
    "earnings_beat": r"\b(beats?|tops?|topped|surpass(es|ed)?|exceeds?|exceeded) (wall street |analysts'? |the )?"
                     r"(estimates?|expectations|forecasts?|consensus)",
    "earnings_miss": r"\b(miss(es|ed)?|falls? short of|fell short of|lags?) (wall street |analysts'? |the )?"
                     r"(estimates?|expectations|forecasts?|consensus)",
    "acquisition": r"\b(to acquire|acquires|acquired|acquisition of|agrees? to buy|takeover|merger with|to merge)\b",
    "buyback": r"\b(buy-?backs?|share repurchases?|repurchase program|stock repurchase)\b",
    "dividend_raised": rf"\b(raises?|increases?|hikes?|boosts?) {_W}dividend",
    "dividend_cut": rf"\b(cuts?|suspends?|slashes|eliminates?) {_W}dividend",
    "lawsuit": r"\b(lawsuit|sued|sues|class action|antitrust suit)\b",
    "investigation": r"\b(investigation|probe|subpoena|charged by the sec|sec charges)\b",
    "approval": r"\b(fda approv\w*|wins? approval|receives? approval|granted approval|cleared by)\b",
    "rejection": r"\b(fda rejects?|complete response letter|denied approval|rejects? (the )?application)\b",
    "upgrade": r"\bupgrades?\b|\bupgraded\b|\braised to (buy|outperform|overweight)\b",
    "downgrade": r"\bdowngrades?\b|\bdowngraded\b|\bcut to (sell|underperform|underweight|neutral|hold)\b",
    "executive_change": r"\b(ceo|cfo|chief executive|chief financial officer)\b.{0,40}\b(steps? down|resigns?|resigned|"
                        r"departs?|retires?|to leave|ousted|named|appointed|hires?)\b",
    "layoffs": r"\b(layoffs?|job cuts|cuts? [\d,]+ jobs|workforce reduction|lay(s|ing)? off)\b",
    "product_launch": r"\b(launch(es|ed)?|unveils?|unveiled|introduces?|debuts?)\b",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in RULES.items()}


def tag(*texts: str) -> list[str]:
    """The event kinds whose pattern appears in any of ``texts``, in RULES order."""
    blob = " ".join(t for t in texts if t)
    return [k for k, rx in _COMPILED.items() if rx.search(blob)]
