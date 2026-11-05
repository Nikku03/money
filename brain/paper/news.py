"""Headlines into features: is it about this company, and what kind of event is it?

The per-ticker Yahoo feed (brain/sources/news.py) mixes in other companies' headlines (asked for NVDA
on 2026-11-04, its top five were other firms' earnings calls), so a headline counts for a stock only
if it names the company as its subject (``about``).

The first live run (2026-11-04) tagged "Top" as an earnings beat, a product "upgrade" as an analyst
upgrade, and a bank's rating of another company as news about the bank. The rules were tightened the
same day, before any of those trades' outcomes existed, and that first step was redone.

Each counted headline is tagged by fixed, readable rules; the tags are the trader's news features,
and their weights start at 0 and are learned only from outcomes, going forward (the feed holds no
history).
"""
from __future__ import annotations

import re

TAGS = {
    # analyst actions only: "upgrades X to buy", "raises price target", "moves to strong buy" (not a product upgrade)
    "news_upgrade": r"\b(upgrade[sd]\b[^.]{0,30}\bto\b|upgrade \w+( \w+)? to (buy|outperform|overweight)|(raises?|lifts?|boosts?) (its |the )?price target|moves? to (strong )?buy|"
                    r"(rated|rating|initiated|starts?)\b[^.]{0,25}\b(buy|outperform|overweight))",
    "news_downgrade": r"\b(downgrade[sd]\b[^.]{0,30}\bto\b|downgrade \w+( \w+)? to (sell|underperform|underweight)|(cuts?|lowers?|slashes?) (its |the )?price target|moves? to (strong )?sell|"
                      r"(rated|rating|initiated|starts?)\b[^.]{0,25}\b(sell|underperform|underweight))",
    # results against a yardstick, not any "top" or "beat"
    "news_beat": r"\b((beats?|tops?|surpass(es)?|exceeds?)\b[^.]{0,40}\b(estimates?|expectations|forecasts?|consensus|views?)|"
                 r"earnings (and \w+ )?beat|record (quarterly )?(revenue|sales|profit)s?)\b",
    "news_miss": r"\b((miss(es)?|falls? short of|trails?)\b[^.]{0,40}\b(estimates?|expectations|forecasts?|consensus|views?)|"
                 r"earnings miss|below (estimates|expectations))\b",
    "news_guide_up": r"\b(rais(es?|ed)|lift(s|ed)?|boost(s|ed)?|hike[sd]?) (its |full-year |annual )?(guidance|outlook|forecast)\b",
    "news_guide_down": r"\b(cuts?|lowers?|slashes?|withdraws?) (its |full-year |annual )?(guidance|outlook|forecast)\b",
    "news_approval": r"\b(approv(al|es|ed)|clearance|wins? (contract|deal|order))\b",
    "news_legal": r"\b(lawsuit|sued|probe|investigation|subpoena|antitrust|fine[ds]?|recall)\b",
    "news_buyback": r"\b(buyback|repurchase|dividend (increase|hike))\b",
    "news_layoffs": r"\b(layoffs?|job cuts|cuts? \d+ jobs|restructuring)\b",
}
NEWS_FEATURES = tuple(TAGS)
_SUFFIX = re.compile(r"\b(inc|incorporated|corp|corporation|co|company|plc|ltd|holdings?|group|class [a-c]|the)\b\.?", re.I)


def aliases(ticker: str, long_name: str | None) -> list[str]:
    """Words that mark a headline as about this company: the ticker and the name's first word(s)."""
    out = {ticker.upper().split("-")[0]}
    if long_name:
        core = _SUFFIX.sub("", long_name).replace(",", " ").split()
        if core:
            out.add(core[0])
            if len(core) > 1 and len(core[0]) <= 3:
                out.add(" ".join(core[:2]))
    return sorted(out, key=len, reverse=True)


ACTOR = r"\W+(starts|initiates|upgrades|downgrades|sees|says|raises|cuts|lifts|rates|keeps|reiterates|expects)\b"


def about(headline: str, names: list[str]) -> bool:
    """Names the company, as its subject: not a roundup of many companies, and not with the company as
    the analyst ("Wells Fargo starts X at overweight" is news about X, not about Wells Fargo)."""
    if headline.count(",") >= 3:                                   # "Top upgrades: A, B, C, D, ..." is a list
        return False
    for n in names:
        if re.search(rf"(?<![\w$]){re.escape(n)}(?!\w)", headline, re.I):
            if re.search(rf"(?<![\w$]){re.escape(n)}(\s\w+)?{ACTOR}", headline, re.I):
                return False
            return True
    return False


def tag(headline: str) -> list[str]:
    return [t for t, rx in TAGS.items() if re.search(rx, headline, re.I)]
