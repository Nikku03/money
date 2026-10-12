"""Claims found on the open web, and their track record.

An agent browsing the web stores what it reads with ``note``: the text, the URL,
when it was published, and the direction it implies (up / down / none). A claim
is not evidence. ``score`` checks each directional claim against what the stock
then did relative to SPY, and reports hit rates per source. In the cell project,
curated literature signs agreed with measured knockdowns about half the time;
expect the same of market commentary until a source's record says otherwise.
"""
from __future__ import annotations

from urllib.parse import urlparse

import numpy as np

from brain.bank import Bank, Fact
from brain.metrics import PriceBook
from brain.research.stats import binom_two_sided
from brain.timeutil import now_iso

DIRECTIONS = {"up": 1, "down": -1, "none": 0}


def note(bank: Bank, ticker: str, text: str, url: str, *, tier: str = "opinion", direction: str = "none",
         known_at: str | None = None, horizon_days: int = 21, attr: str = "claim") -> Fact:
    if tier not in ("opinion", "reported"):
        raise ValueError("web claims are tier 'opinion' (views, forecasts) or 'reported' (factual news)")
    known = known_at or now_iso()
    if known > now_iso():
        raise ValueError("known_at is in the future")
    f = Fact(entity=ticker, attr=attr, value=text, as_of=known[:10], known_at=known, tier=tier,
             source={"name": urlparse(url).hostname or url, "url": url},
             context={"direction": DIRECTIONS[direction], "horizon_days": horizon_days}, retrieved_at=now_iso())
    bank.add(f)
    return f


def score(bank: Bank, benchmark: str = "SPY") -> dict:
    claims = []
    for e in bank.entities():
        claims += [f for f in bank.facts(e, "claim") if f.context.get("direction")]
    book = PriceBook(bank, sorted({c.entity for c in claims} | {benchmark}))
    rows, pending = [], []
    for c in claims:
        if not book.has(c.entity) or not book.has(benchmark):
            pending.append((c, "no prices"))
            continue
        d, a = book.dates[c.entity], book.adj[c.entity]
        i0 = int(np.searchsorted(d, c.known_at[:10], side="right"))  # first close strictly after publication
        i1 = i0 + int(c.context.get("horizon_days", 21))
        if i1 >= len(d):
            pending.append((c, "horizon not reached"))
            continue
        bd, ba = book.dates[benchmark], book.adj[benchmark]
        j0, j1 = np.searchsorted(bd, d[i0]), np.searchsorted(bd, d[i1])
        if j1 >= len(bd) or bd[j0] != d[i0] or bd[j1] != d[i1]:
            pending.append((c, "calendar mismatch"))
            continue
        excess = (a[i1] / a[i0] - 1) - (ba[j1] / ba[j0] - 1)
        rows.append({"ref": c.ref, "source": c.source["name"], "direction": c.context["direction"],
                     "excess": float(excess), "hit": bool(np.sign(excess) == c.context["direction"])})
    by = {}
    for r in rows:
        s = by.setdefault(r["source"], {"n": 0, "hits": 0})
        s["n"] += 1
        s["hits"] += r["hit"]
    for s in by.values():
        s["hit_rate"] = s["hits"] / s["n"]
        s["p_vs_coin"] = binom_two_sided(s["hits"], s["n"])
    return {"by_source": by, "scored": rows, "pending": [(c.ref, why) for c, why in pending]}
