"""Headlines from Yahoo Finance's per-ticker RSS feed.

Stored as tier "reported" and unscored: a headline is a claim that something
happened, not evidence that it moves the price.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from brain.bank import Fact
from brain.net import fetch
from brain.timeutil import rfc822_to_iso

RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"


def parse_rss(ticker: str, body: bytes, fetched_at: str) -> list[Fact]:
    root = ET.fromstring(body)
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate")
        if not (title and link and pub):
            continue
        known = rfc822_to_iso(pub)
        out.append(Fact(entity=ticker, attr="headline", value=title, as_of=known[:10], known_at=known,
                        tier="reported", source={"name": urlparse(link).hostname or "unknown", "url": link},
                        context={"feed": "yahoo-rss"}, retrieved_at=fetched_at))
    return out


def ingest(ticker: str) -> list[Fact]:
    r = fetch(RSS.format(sym=ticker.upper()), ttl=3600)
    return parse_rss(ticker, r.body, r.fetched_at)
