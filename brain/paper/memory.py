"""The trader's memory: what it did, what it missed, what it believed, and recall of all three.

The memory is a bank (brain/bank.py) of its own, committed in research/paper/memory/, so it outlives
the containers the trader runs in. Every memory is a fact with the moment it became known:
- a trade is remembered from the session it closed (before then, its outcome did not exist);
- a missed moonshot from the session its move finished;
- a belief from the week it was held.
Recall goes through a View at the moment of the decision, so a trader in 2021 cannot recall
anything learned in 2022. Nothing is overwritten: a later lesson is a new fact.

    episodes   memory.trade      one per trade: entry, exit rule, net, shape, best and worst
                                 point, the exit that would have done best, its call against the
                                 result, its reasons, the influences, the filings, the lesson
    misses     memory.missed     one per missed moonshot: cause, factors, where it ranked it
    beliefs    memory.beliefs    what raised and lowered its odds, as it then believed (TRADER)

``recall`` gathers, for one company at one moment: its trades of it and what they taught, the
moonshots of it that it missed, and a written card for the journal. ``Analogs`` (used by the P8
machine) recalls situations like the present one among every stock it has watched and how they
turned out.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from brain import config
from brain.bank import Bank, Fact

ROOT = config.ROOT / "research" / "paper" / "memory"
TRADER = "TRADER"
MARKET = "MARKET"


def bank(root: Path | None = None) -> Bank:
    root = Path(root or ROOT)
    root.mkdir(parents=True, exist_ok=True)
    ign = root / ".gitignore"
    if not ign.exists():
        ign.write_text("# rebuildable caches; the journal shards are the memory\n*.idx\n*.lock\n")
    return Bank(root)


def _at(date: str) -> str:
    """A session's date as a known_at: after that day's close (21:00 UTC)."""
    return f"{date[:10]}T21:00:00Z"


def trade_fact(tr: dict, source: str) -> Fact:
    keep = ("exit", "net", "shape", "mfe", "mae", "best_day", "best_exit", "regret", "predicted", "reasons",
            "influences", "filings_during", "lesson", "by_exit")
    value = {k: tr[k] for k in keep if k in tr and tr[k] is not None}
    value["entry"] = tr["date"]
    return Fact(entity=tr["ticker"], attr="memory.trade", value=value, as_of=tr["date"][:10],
                known_at=_at(tr["closed"]), tier="derived", source={"name": f"paper trader: {source}"},
                context={"run": source})


def missed_fact(r: dict, source: str, closed: str) -> Fact:
    value = {k: r[k] for k in ("cause", "tags", "rank", "of", "move", "filed", "filed_on_jump", "overnight_share",
                                "industry_wave", "market", "volume_x", "jump_day") if k in r}
    return Fact(entity=r["ticker"], attr="memory.missed", value=value, as_of=r["date"][:10], known_at=_at(closed),
                tier="derived", source={"name": f"missed-moonshot review: {source}"}, context={"run": source})


def beliefs_fact(date: str, lessons: list, source: str) -> Fact:
    return Fact(entity=TRADER, attr="memory.beliefs", value={"lessons": [[n, round(float(w), 5)] for n, w in lessons]},
                as_of=date[:10], known_at=_at(date), tier="derived", source={"name": f"paper trader: {source}"},
                context={"run": source})


def market_fact(les: dict, source: str) -> Fact:
    """A week's market lesson (brain/paper/market.py), known from the open after the week, when its
    last move was known."""
    keep = ("market", "list_vs_market", "explained", "forces", "paid", "punished", "expected", "ic", "lesson")
    value = {k: (None if isinstance(les[k], float) and not np.isfinite(les[k]) else les[k]) for k in keep if k in les}
    return Fact(entity=MARKET, attr="memory.market", value=value, as_of=les["week"][:10],
                known_at=f"{les['known'][:10]}T14:35:00Z", tier="derived", source={"name": f"market model: {source}"},
                context={"run": source})


def market_recall(b: Bank, at: str, n: int = 4) -> list[dict]:
    """The latest ``n`` market lessons known at ``at``."""
    rows = sorted(b.view(at).facts(MARKET, "memory.market"), key=lambda f: f.as_of)[-n:]
    return [f.value | {"week": f.as_of} for f in rows]


def remember(b: Bank, facts: list[Fact]) -> int:
    return b.add_many(facts)


# --- recall -------------------------------------------------------------------------------------------
def recall(b: Bank, ticker: str, at: str) -> dict:
    """What the memory holds about ``ticker`` as known at ``at`` (a date or time)."""
    v = b.view(at)
    trades = [f.value for f in v.facts(ticker, "memory.trade")]
    missed = [f.value | {"date": f.as_of} for f in v.facts(ticker, "memory.missed")]
    nets = [t["net"] for t in trades]
    out = {"ticker": ticker, "at": at, "trades": len(trades), "missed": len(missed),
           "mean_net": float(np.mean(nets)) if nets else None,
           "shapes": {s: sum(1 for t in trades if t.get("shape") == s) for s in {t.get("shape") for t in trades}},
           "last_lessons": [t.get("lesson") for t in sorted(trades, key=lambda t: t["entry"])[-3:]],
           "missed_causes": {c: sum(1 for m in missed if m.get("cause") == c) for c in {m.get("cause") for m in missed}}}
    return out


def card(r: dict) -> str:
    """The memory, written for the journal before a trade."""
    if not r["trades"] and not r["missed"]:
        return f"memory: nothing remembered about {r['ticker']} yet"
    parts = []
    if r["trades"]:
        shapes = ", ".join(f"{n} {s}" for s, n in sorted(r["shapes"].items(), key=lambda kv: -kv[1]) if s)
        parts.append(f"traded {r['ticker']} {r['trades']} time(s) before, mean {r['mean_net']:+.1%} ({shapes})")
        if r["last_lessons"] and r["last_lessons"][-1]:
            parts.append("last lesson: " + r["last_lessons"][-1][:220])
    if r["missed"]:
        causes = ", ".join(f"{n} {c}" for c, n in sorted(r["missed_causes"].items(), key=lambda kv: -kv[1]) if c)
        parts.append(f"missed {r['missed']} moonshot(s) in it before ({causes})")
    return "memory: " + "; ".join(parts)


# --- situations like this one -------------------------------------------------------------------------
class Analogs:
    """Recall of past situations like the present one: the k nearest resolved candidates (every stock
    it watched, not only those it traded), in standardized feature space, and how they turned out.
    Only situations whose outcome was known before the decision are kept; the oldest are let go after
    ``keep_weeks`` weeks."""

    def __init__(self, k: int = 50, keep_weeks: int = 104, cap: int = 20000):
        self.k, self.keep, self.cap = k, keep_weeks, cap
        self.rows = []                                         # (week, resolved session, Z, net, hit)
        self.mu = self.sd = None

    def add(self, week: int, resolved: int, X: np.ndarray, net: np.ndarray, hit: np.ndarray):
        self.rows.append((week, resolved, X.astype(np.float32), net.astype(np.float32), hit.astype(np.float32)))

    def recall(self, now_week: int, d: int, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.rows = [r for r in self.rows if r[0] > now_week - self.keep]
        known, n = [], 0
        for r in reversed([r for r in self.rows if r[1] < d]):       # the most recent first, up to ``cap`` rows
            if n >= self.cap:
                break
            known.append(r)
            n += len(r[2])
        if sum(len(r[2]) for r in known) < self.k * 10:
            return np.zeros(len(X)), np.zeros(len(X))
        A = np.vstack([r[2] for r in known])
        if self.mu is None:
            self.mu, self.sd = A.mean(0), A.std(0) + 1e-9
        Az = np.clip((A - self.mu) / self.sd, -5, 5)
        Q = np.clip((X - self.mu) / self.sd, -5, 5).astype(np.float32)
        net, hit = np.concatenate([r[3] for r in known]), np.concatenate([r[4] for r in known])
        d2 = (Q * Q).sum(1)[:, None] - 2 * Q @ Az.T + (Az * Az).sum(1)[None, :]
        near = np.argpartition(d2, self.k, axis=1)[:, :self.k]
        return net[near].mean(1), hit[near].mean(1)


# --- filling it with what is already recorded ---------------------------------------------------------
def backfill(b: Bank, log=print) -> dict:
    """Remember the trades and misses already recorded in research/paper/ (each dated by when it closed)."""
    paper = config.ROOT / "research" / "paper"
    n = {}
    for name, src in (("P6_corrected_journal.jsonl", "P6 (corrected), 2021-chosen list, 2021-2026"),
                      ("P7_heldout_journal.jsonl", "P7 held-out, 2015-chosen list, 2016-2020")):
        p = paper / name
        if not p.exists():
            continue
        facts = [trade_fact(json.loads(x), src) for x in p.read_text().splitlines() if x.strip()]
        n[name] = remember(b, facts)
    # the market model's weekly lessons, one per week (the held-out run's, on another list over the same
    # weeks, stay in research/paper/P10_heldout_market.jsonl)
    for name, src in (("P10_development_market.jsonl", "P10 development, 2021-chosen list, 2011-2026"),):
        p = paper / name
        if p.exists():
            n[name] = remember(b, [market_fact(json.loads(x), src) for x in p.read_text().splitlines() if x.strip()])
    p = paper / "P4_moonshots.jsonl"
    if p.exists():
        facts = []
        for x in p.read_text().splitlines():
            r = json.loads(x)
            if r.get("caught"):
                continue
            # a moonshot is known once +20% was reached: within the 10 sessions after its entry; dated
            # conservatively at the 15th weekday after entry
            closed = str(np.busday_offset(r["date"][:10], 14, roll="forward"))
            facts.append(missed_fact(r, "P4 review, 2026-chosen list, 2021-2026", closed))
        n["P4_moonshots.jsonl (missed only)"] = remember(b, facts)
    log(f"remembered: {n}")
    return n
