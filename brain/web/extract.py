"""Fixed rules that take facts out of a page: no model, nothing inferred (docs/WEB.md).

- ``article``: a news story or press release, from its structured data (schema.org JSON-LD: headline,
  datePublished, author, publisher) and its meta tags (OpenGraph, article:published_time), then its text.
- ``links``: the page's links, optionally those whose address or text matches a pattern.
- ``tables``: every HTML table, as rows of cell text.

What a page does not state is left out: an article with no stated publication time gets none.
"""
from __future__ import annotations

import contextlib
import json
import re
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse

from brain import scout

ARTICLE_TYPES = {"newsarticle", "article", "reportagenewsarticle", "analysisnewsarticle", "blogposting",
                 "pressrelease", "report", "webpage"}
_PUBLISHED_META = ("article:published_time", "og:article:published_time", "datepublished", "pubdate",
                   "publishdate", "date", "dc.date", "sailthru.date", "parsely-pub-date")
_MODIFIED_META = ("article:modified_time", "og:updated_time", "datemodified", "last-modified")


class _Doc(HTMLParser):
    """One pass over a page: title, meta tags, JSON-LD blocks, links and tables."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.meta, self.ld, self.links, self.tables = "", {}, [], [], []
        self.canonical = ""
        self._in_title = self._in_ld = False
        self._buf: list[str] = []
        self._a: dict | None = None
        self._table: list | None = None
        self._row: list | None = None
        self._cell: list | None = None
        self._stack: list[list] = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "title":
            self._in_title, self._buf = True, []
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").strip().lower()
            if key and a.get("content") and key not in self.meta:
                self.meta[key] = a["content"].strip()
        elif tag == "link" and "canonical" in a.get("rel", "").lower() and a.get("href"):
            self.canonical = a["href"]
        elif tag == "script" and "ld+json" in a.get("type", "").lower():
            self._in_ld, self._buf = True, []
        elif tag == "a" and a.get("href"):
            self._a = {"href": a["href"], "text": []}
        elif tag == "table":
            if self._table is not None:
                self._stack.append(self._table)
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "title" and self._in_title:
            self.title, self._in_title = " ".join("".join(self._buf).split()), False
        elif tag == "script" and self._in_ld:
            self.ld.append("".join(self._buf))
            self._in_ld = False
        elif tag == "a" and self._a is not None:
            self.links.append({"href": self._a["href"], "text": " ".join("".join(self._a["text"]).split())})
            self._a = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = self._stack.pop() if self._stack else None

    def handle_data(self, data):
        if self._in_title or self._in_ld:
            self._buf.append(data)
        if self._a is not None:
            self._a["text"].append(data)
        if self._cell is not None:
            self._cell.append(data)


def parse(html: str) -> _Doc:
    d = _Doc()
    with contextlib.suppress(Exception):   # a broken page gives what was read before it broke
        d.feed(html or "")
        d.close()
    return d


def jsonld(doc: _Doc) -> list[dict]:
    """Every JSON-LD object on the page, @graph members and lists flattened."""
    out = []

    def walk(o):
        if isinstance(o, list):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            out.append(o)
            if "@graph" in o:
                walk(o["@graph"])
    for raw in doc.ld:
        try:
            walk(json.loads(raw.strip()))
        except ValueError:
            continue
    return out


def _types(o: dict) -> set[str]:
    t = o.get("@type")
    return {str(x).lower() for x in (t if isinstance(t, list) else [t]) if x}


def _name(v) -> str | None:
    """A person's or an organisation's name, from a string, an object or a list of them."""
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, dict):
        return _name(v.get("name"))
    if isinstance(v, list):
        names = [n for n in (_name(x) for x in v) if n]
        return ", ".join(names) or None
    return None


def article(html: str, url: str) -> dict:
    """What the page states about itself as a story: headline, publication and update times (ISO, UTC), author,
    publisher, description, section, whether it marks itself paywalled, and its text."""
    d = parse(html)
    objs = [o for o in jsonld(d) if _types(o) & ARTICLE_TYPES]
    objs.sort(key=lambda o: "webpage" in _types(o))   # a story before a generic page
    a = objs[0] if objs else {}
    m = d.meta
    published = scout.published_iso(a.get("datePublished") or next((m[k] for k in _PUBLISHED_META if k in m), None))
    modified = scout.published_iso(a.get("dateModified") or next((m[k] for k in _MODIFIED_META if k in m), None))
    text = scout.html_to_text(html)["text"]
    headline = (a.get("headline") or m.get("og:title") or m.get("twitter:title") or d.title or "").strip()
    return {
        "url": urldefrag(urljoin(url, d.canonical))[0] if d.canonical else url,
        "headline": " ".join(headline.split())[:400],
        "published": published,
        "modified": modified,
        "author": _name(a.get("author")) or m.get("author"),
        "publisher": _name(a.get("publisher")) or m.get("og:site_name"),
        "description": " ".join((a.get("description") or m.get("og:description") or m.get("description") or "").split())[:600],
        "section": a.get("articleSection") if isinstance(a.get("articleSection"), str) else m.get("article:section"),
        "paywalled": str(a.get("isAccessibleForFree", "")).lower() == "false",
        "structured": bool(objs),
        "text": text,
        "chars": len(text),
    }


def links(html: str, base: str, pattern: str | None = None, same_host: bool = False) -> list[dict]:
    """The page's http(s) links, absolute and without fragments, first occurrence only; with ``pattern``, those
    whose address or text matches it (a regular expression, case-insensitive)."""
    rx = re.compile(pattern, re.IGNORECASE) if pattern else None
    host = (urlparse(base).hostname or "").lower()
    seen, out = set(), []
    for ln in parse(html).links:
        u = urldefrag(urljoin(base, ln["href"]))[0]
        p = urlparse(u)
        if p.scheme not in ("http", "https") or u in seen:
            continue
        if same_host and (p.hostname or "").lower() != host:
            continue
        if rx and not (rx.search(u) or rx.search(ln["text"])):
            continue
        seen.add(u)
        out.append({"url": u, "text": ln["text"][:300]})
    return out


def feed(xml: str) -> list[dict]:
    """An RSS or Atom feed's items: link, title and stated publication time, in the feed's order."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml.encode("utf-8", "replace"))
    except ET.ParseError:
        return []
    out = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag not in ("item", "entry"):
            continue
        kids = {k.tag.rsplit("}", 1)[-1]: k for k in el}
        link = ""
        for k in el:   # Atom: the story is the rel="alternate" link (or one with no rel); images are enclosures
            if k.tag.rsplit("}", 1)[-1] == "link" and k.get("rel", "alternate") == "alternate":
                link = (k.get("href") or k.text or "").strip()
                break
        title = " ".join((kids["title"].text or "").split()) if "title" in kids else ""
        when = next((kids[k].text for k in ("pubDate", "published", "updated", "date") if k in kids and kids[k].text), None)
        if link.startswith(("http://", "https://")):
            out.append({"url": urldefrag(link)[0], "text": title[:300], "published": scout.published_iso(when)})
    return out


def tables(html: str) -> list[list[list[str]]]:
    """Every table with at least one row, as rows of cell text."""
    return parse(html).tables
