"""Headlines into features: is it about this company, and what kind of event is it?

The per-ticker Yahoo feed (brain/sources/news.py) mixes in other companies' headlines (asked for NVDA
on 2026-11-04, its top five were other firms' earnings calls), so a headline counts for a stock only
if it names the company. Each counted headline is tagged by fixed, readable rules; the tags are the
trader's news features, and their weights start at 0 and are learned only from outcomes, going
forward (the feed holds no history).
"""
from __future__ import annotations

import re

TAGS = {
    "news_upgrade": r"\b(upgrade[sd]?|raises? (its )?(price )?target|outperform|overweight|buy rating)\b",
    "news_downgrade": r"\b(downgrade[sd]?|cuts? (its )?(price )?target|underperform|underweight|sell rating)\b",
    "news_beat": r"\b(beats?|tops?|surpass(es)?|exceeds?|record (revenue|profit|sales))\b",
    "news_miss": r"\b(miss(es)?|falls? short|disappoint\w*|below (estimates|expectations))\b",
    "news_guide_up": r"\b(raises?|lifts?|boosts?|hikes?) (its |full-year |annual )?(guidance|outlook|forecast)\b",
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


def about(headline: str, names: list[str]) -> bool:
    return any(re.search(rf"(?<![\w$]){re.escape(n)}(?!\w)", headline, re.I) for n in names)


def tag(headline: str) -> list[str]:
    return [t for t, rx in TAGS.items() if re.search(rx, headline, re.I)]
