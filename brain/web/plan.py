"""What the reasoning engine sends the gateway to get, and what it keeps (docs/WEB.md).

The engine looks at what the bank lacks and writes tasks: a recipe, a URL, the ticker it is for and why. No model
is asked; the reasons are rules over the bank.

- **Stories behind headlines.** A headline in the bank (from a news feed) whose page has not been read: read it
  for what the page states about itself (``extract.article``), tag its events (``events.tag``) and keep it as a
  ``web.article`` fact.
- **Listed sources.** ``knowledge/web_sources.json`` names pages or feeds to watch for a ticker: a newsroom's RSS
  feed, a press-release list. The gateway reads it, keeps the links matching its pattern, and reads the newest.

Each kept fact cites the page it came from. Its ``known_at`` is the publication time the page states, or the
time it was read when it states none. A ``web.article`` is a report of what a page says: it weighs nothing in a
decision until a declared test has measured what such reports do.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from brain import config
from brain.bank import Bank, Fact
from brain.web import events, extract

ATTR = "web.article"
SOURCES = config.ROOT / "knowledge" / "web_sources.json"
PER_TICKER = 5
MAX_TEXT = 4000      # characters of a story's text kept in the fact's context


def sources(path=None) -> list[dict]:
    p = Path(path or SOURCES)
    if not p.exists():
        return []
    return [s for s in json.loads(p.read_text()).get("sources", []) if s.get("url") and s.get("ticker")]


def read_urls(bank: Bank, ticker: str) -> set[str]:
    """Pages already kept for ``ticker``, as asked and as finally read."""
    out = set()
    for f in bank.view().facts(ticker, ATTR):
        out.add(f.source.get("url", ""))
        out.add((f.context or {}).get("asked_url", ""))
    return out - {""}


def plan(bank: Bank, tickers, per_ticker: int = PER_TICKER, source_path=None) -> list[dict]:
    """The tasks the bank's gaps call for, newest headlines first. No network."""
    tasks = []
    for t in [x.upper() for x in tickers]:
        have = read_urls(bank, t)
        heads = sorted(bank.view().facts(t, "headline"), key=lambda f: f.known_at, reverse=True)
        n = 0
        for f in heads:
            url = f.source.get("url", "")
            if not url or url in have or n >= per_ticker:
                continue
            tasks.append({"recipe": "article", "ticker": t, "url": url,
                          "why": f"headline of {f.known_at[:10]} not read yet: \"{str(f.value)[:80]}\""})
            have.add(url)
            n += 1
        for s in sources(source_path):
            if s["ticker"].upper() == t:
                tasks.append({"recipe": "listing", "ticker": t, "url": s["url"], "pattern": s.get("pattern"),
                              "feed": bool(s.get("feed")), "follow": int(s.get("follow", 3)),
                              "why": f"listed source: {s.get('name', s['url'])}"})
    return tasks


def article_fact(ticker: str, page, art: dict) -> Fact:
    when = art["published"] or page.fetched_at
    tags = events.tag(art["headline"], art["description"])
    return Fact(entity=ticker.upper(), attr=ATTR, value=art["headline"] or "(no headline)", as_of=when[:10],
                known_at=when, tier="reported",
                source={"name": art["publisher"] or urlparse(page.final_url).hostname or "web", "url": page.final_url},
                context={"asked_url": page.url, "published": art["published"], "modified": art["modified"],
                         "published_stated": bool(art["published"]), "author": art["author"],
                         "description": art["description"], "section": art["section"], "events": tags,
                         "structured": art["structured"], "chars": art["chars"], "text": art["text"][:MAX_TEXT],
                         "read_by": f"gateway ({page.mode})"},
                retrieved_at=page.fetched_at)


def run(bank: Bank, tasks: list[dict], gateway, log=print) -> dict:
    """Carry out ``tasks`` through ``gateway``; keep what the rules extract. Returns what was kept and refused."""
    kept, refused = [], []
    queue = list(tasks)
    while queue:
        task = queue.pop(0)
        page = gateway.open(task["url"], feed=task.get("feed", False))
        if not page.ok:
            refused.append({**task, "error": page.error})
            if log:
                log(f"  {task['ticker']} {task['recipe']}: {task['url'][:90]} -> {page.error}")
            continue
        if task["recipe"] == "article":
            art = extract.article(page.html, page.final_url)
            if art["paywalled"]:
                refused.append({**task, "error": "the story marks itself as paywalled"})
                continue
            fact = article_fact(task["ticker"], page, art)
            n = bank.add_many([fact])
            kept.append({"ticker": fact.entity, "url": page.final_url, "headline": fact.value, "events":
                         fact.context["events"], "published": art["published"], "new": bool(n)})
            if log:
                log(f"  {fact.entity}: {fact.value[:80]} | {art['published'] or 'no stated time'} | "
                    f"{', '.join(fact.context['events']) or 'no event tags'}")
        elif task["recipe"] == "listing":
            have = read_urls(bank, task["ticker"])
            listed = (extract.feed(page.html) if task.get("feed")
                      else extract.links(page.html, page.final_url, task.get("pattern")))
            if task.get("feed") and task.get("pattern"):
                rx = re.compile(task["pattern"], re.IGNORECASE)
                listed = [ln for ln in listed if rx.search(ln["url"]) or rx.search(ln["text"])]
            found = [ln for ln in listed if ln["url"] not in have][:task.get("follow", 3)]
            if log:
                log(f"  {task['ticker']} listing {page.final_url[:80]}: {len(found)} new links")
            queue += [{"recipe": "article", "ticker": task["ticker"], "url": ln["url"],
                       "why": f"linked from {page.final_url}"} for ln in found]
    return {"kept": kept, "refused": refused, "pages": gateway.reader.fetched}
