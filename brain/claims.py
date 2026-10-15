"""Claims found on the open web, and their track record.

An agent browsing the web stores what it reads with ``note``: the text, the URL,
when it was published, and the direction it implies (up / down / none). A claim
is not evidence. ``score`` checks each directional claim against what the stock
then did relative to SPY, and reports hit rates per source. In the cell project,
curated literature signs agreed with measured knockdowns about half the time;
expect the same of market commentary until a source's record says otherwise.

``track_record`` is the stricter record the decision policy weighs (docs/DECIDE.md,
section 3.3): a call is scored only on sessions after the bank held it, calls made
in the same window count as one bet, and a source needs enough bets, a one-sided
test, a false-discovery correction across sources and a lower confidence bound on
its hit rate before its calls weigh anything. Its calls must also be priced: calls
whose window closed with no close for their stock count against it, so a record
cannot be made by fetching prices only for the calls that went its way.
"""
from __future__ import annotations

from urllib.parse import urlparse

import numpy as np

from brain.bank import Bank, Fact
from brain.metrics import PriceBook
from brain.research.stats import bh_adjust, binom_lower_bound, binom_one_sided_greater, binom_two_sided
from brain.timeutil import days_between, norm_known_at, now_iso, today

DIRECTIONS = {"up": 1, "down": -1, "none": 0}
STALE_CLOSE_DAYS = 4   # a stock's close older than this, at a benchmark session, is no close for it
# A call whose exit is this many benchmark sessions past, with still no outcome, is "unpriced": its stock's
# closes are missing (never ingested, not refreshed, delisted). It is scored as soon as they arrive, but a
# source is not proven while more than UNPRICED_MAX_SHARE of its closed calls are unpriced, so a record
# cannot be built by pricing only the calls that went its way.
GRACE_SESSIONS = 5
UNPRICED_MAX_SHARE = 0.10


def note(bank: Bank, ticker: str, text: str, url: str, *, tier: str = "opinion", direction: str = "none",
         known_at: str | None = None, horizon_days: int = 21, attr: str = "claim", noted_by: str | None = None) -> Fact:
    """``noted_by`` records who read the page when it was not a person or agent at the CLI (``brain scout``
    passes ``scout:<model>``), so claims can later be scored by who noted them as well as by source.
    The URL must be http(s) with a host: the source is named after that host, so no claim can take a
    name such as ``llm:...`` that the model's own judgments use."""
    u = urlparse(str(url).strip())
    if u.scheme.lower() not in ("http", "https") or not u.hostname:
        raise ValueError(f"a claim's source is an http(s) URL with a host, not {str(url)[:80]!r}")
    if tier not in ("opinion", "reported"):
        raise ValueError("web claims are tier 'opinion' (views, forecasts) or 'reported' (factual news)")
    raw = str(known_at).strip() if known_at else ""
    known = norm_known_at(raw) if raw else now_iso()
    # A time is compared in UTC, whatever offset the page wrote it with. A date alone stands for the end of
    # that UTC day, so today's date is accepted (a page that gives only its date) and a later one is not.
    if (raw[:10] > today()) if len(raw) == 10 else (known > now_iso()):
        raise ValueError("known_at is in the future")
    context = {"direction": DIRECTIONS[direction], "horizon_days": horizon_days}
    if noted_by:
        context["noted_by"] = noted_by
    f = Fact(entity=ticker, attr=attr, value=text, as_of=known[:10], known_at=known, tier=tier,
             source={"name": u.hostname, "url": url}, context=context, retrieved_at=now_iso())
    bank.add(f)
    return f


def score(bank: Bank, benchmark: str = "SPY") -> dict:
    """Every directional claim against SPY from the first close after publication, overlapping calls
    included, with a two-sided p per source. The terminal's CN and CLM and ``brain score-claims`` read
    this; the record a decision weighs is ``track_record``."""
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


# ---------- the record a decision weighs ----------
def anchor(f: Fact) -> str:
    """When the bank held a call: the later of its publication and its retrieval. A page noted a week after
    it appeared is scored from the week after, never on the move it may describe."""
    return max(f.known_at, f.retrieved_at or f.known_at)


def source_key(f: Fact) -> str:
    """A claim's source is its site and who noted it ("reuters.com via scout:qwen3:8b"); a judgment's is
    the model's own name ("llm:qwen3:8b@3f2a91c0")."""
    if f.attr == "judgment":
        return str(f.source.get("name"))
    return f"{f.source.get('name')} via {f.context.get('noted_by') or 'agent'}"


def outcome(book: PriceBook, ticker: str, anchor: str, horizon: int, benchmark: str = "SPY") -> dict | None:
    """Excess return over the benchmark from the first benchmark session strictly after anchor's date to
    `horizon` benchmark sessions later; the stock's close is its last close on or before each of those
    sessions (its own calendar may differ). {"entry": date, "exit": date, "excess": float}; None when the
    exit session has not happened in the book, either series lacks a close by then, or the stock's close is
    more than 4 days older than the session."""
    t, b = ticker.upper(), benchmark.upper()
    if not book.has(t) or not book.has(b):
        return None
    bd, ba = book.dates[b], book.adj[b]
    j0 = int(np.searchsorted(bd, anchor[:10], side="right"))
    j1 = j0 + int(horizon)
    if j1 >= len(bd):
        return None
    sd, sa = book.dates[t], book.adj[t]
    closes = []
    for j in (j0, j1):
        i = int(np.searchsorted(sd, bd[j], side="right")) - 1
        if i < 0 or days_between(str(sd[i]), str(bd[j])) > STALE_CLOSE_DAYS or not np.isfinite(sa[i]) or sa[i] <= 0:
            return None
        closes.append(float(sa[i]))
    if not (np.isfinite(ba[j0]) and np.isfinite(ba[j1]) and ba[j0] > 0):
        return None
    excess = (closes[1] / closes[0] - 1) - (float(ba[j1]) / float(ba[j0]) - 1)
    return {"entry": str(bd[j0]), "exit": str(bd[j1]), "excess": float(excess)}


def scored_at(book: PriceBook, ticker: str, anchor: str, horizon: int, t: str,
              benchmark: str = "SPY") -> tuple[str, dict | None]:
    """("scored", outcome) when the call's window closed by t and both closes are there; ("unpriced", None)
    when the benchmark's book at t holds GRACE_SESSIONS sessions past its exit and the stock still has no
    outcome; else ("pending", None). Whether a call is unpriced is read from the benchmark's calendar alone,
    so it cannot depend on what the stock did."""
    o = outcome(book, ticker, anchor, horizon, benchmark)
    if o is not None and o["exit"] <= t[:10]:  # no exit after t
        return "scored", o
    b = benchmark.upper()
    if o is None and book.has(b):
        bd = book.dates[b]
        held = int(np.searchsorted(bd, t[:10], side="right"))  # the benchmark's sessions up to t
        if int(np.searchsorted(bd, anchor[:10], side="right")) + int(horizon) + GRACE_SESSIONS < held:
            return "unpriced", None
    return "pending", None


def bets(calls: list[dict]) -> tuple[list[dict], int]:
    """Group scored calls into independent bets. `calls`: [{"ref", "ticker", "direction", "entry", "exit",
    "excess"}], any order. In entry order (ties by ref), a call opens a bet when its entry is after the exit
    of the last bet; calls with that same entry join it; any other call whose entry falls inside a kept bet's
    window is dropped. A bet's value is the mean of direction*excess over its calls; it is a hit when > 0,
    a miss when < 0, and unscored when exactly 0. Returns (bets, n_dropped); each bet:
    {"entry", "exit", "refs", "value", "hit"}."""
    kept: list[dict] = []
    dropped = 0
    for c in sorted(calls, key=lambda c: (c["entry"], c["ref"])):
        last = kept[-1] if kept else None
        if last is None or c["entry"] > last["exit"]:
            kept.append({"entry": c["entry"], "exit": c["exit"], "calls": [c]})
        elif c["entry"] == last["entry"]:
            last["calls"].append(c)
            last["exit"] = max(last["exit"], c["exit"])
        else:
            dropped += 1  # inside a kept bet's window: the same market move, counted once
    out = []
    for b in kept:
        value = float(sum(c["direction"] * c["excess"] for c in b["calls"]) / len(b["calls"]))
        out.append({"entry": b["entry"], "exit": b["exit"], "refs": [c["ref"] for c in b["calls"]],
                    "value": value, "hit": True if value > 0 else False if value < 0 else None})
    return out, dropped


def tally(bs: list[dict], alpha: float = 0.05) -> dict:
    """Bets' counts and tests: hits, hit rate, one-sided p against a coin, the Clopper-Pearson lower bound,
    and the mean signed excess. An unscored (exactly zero) bet counts in the mean only."""
    scored = [b for b in bs if b["hit"] is not None]
    n, k = len(scored), sum(1 for b in scored if b["hit"])
    return {"bets": n, "hits": k, "hit_rate": k / n if n else None,
            "p": binom_one_sided_greater(k, n) if n else None,
            "lower": binom_lower_bound(k, n, alpha) if n else None,
            "mean_signed_excess": float(np.mean([b["value"] for b in bs])) if bs else None}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def too_unpriced(unpriced: int, closed: int) -> bool:
    """True when more than UNPRICED_MAX_SHARE of the calls whose window has closed (``closed``, priced or
    not) have no outcome. A rare delisting fits under it; prices fetched only for the winners do not."""
    return unpriced > UNPRICED_MAX_SHARE * closed


def gate_rows(raw: dict[str, dict], gate: dict) -> dict[str, dict]:
    """Add q (Benjamini-Hochberg across every source with at least one bet), proven, weight and why to rows
    that hold ``tally``'s fields and the call counts. A source with too many unpriced calls
    (``too_unpriced``) is not proven, whatever its bets say, until their tickers are ingested."""
    tested = [k for k, r in raw.items() if r["bets"] >= 1]
    q_of = dict(zip(tested, bh_adjust([raw[k]["p"] for k in tested])))
    out = {}
    for k, r in raw.items():
        q = q_of.get(k)
        n, hits = r["bets"], r["hits"]
        mse = r["mean_signed_excess"]
        u = int(r.get("unpriced") or 0)
        closed = r["calls"] - r["pending"] - r["unscorable"]  # every call is one of unscorable, pending, unpriced, scored
        blocked = too_unpriced(u, closed)
        proven = bool(n >= gate["min_bets"] and r["p"] is not None and r["p"] < gate["p_lt"]
                      and q is not None and q < gate["bh_q"] and mse is not None and mse > gate["mean_signed_excess_gt"]
                      and not blocked)
        weight = max(0.0, 2 * r["lower"] - 1) if proven else 0.0
        unpriced = f"{u} with no price past {'its' if u == 1 else 'their'} exit"  # "3 with no price past their exit"
        if proven:
            why = f"proven: {n} bets, {r['hit_rate']:.0%} hit, weight {weight:.2f}"
            rest = [f"{r['pending']} pending"] * bool(r["pending"]) + [unpriced] * bool(u)
            why += f"; of its calls, {' and '.join(rest)}" if rest else ""
        elif n >= gate["min_bets"]:
            why = (f"{n} bets, {r['hit_rate']:.0%} hit, p {r['p']:.2f}, q {q:.2f} after "
                   f"{_plural(len(tested), 'source')}: not proven")
            if mse is not None and not mse > gate["mean_signed_excess_gt"]:
                why += f" (mean signed excess {mse * 100:+.2f}%)"
            if blocked:
                why += (f"; {u} of {closed} calls past their exit have no price for their stock (at most "
                        f"{UNPRICED_MAX_SHARE:.0%} may): ingest those tickers")
            elif u:
                why += f"; of its calls, {unpriced}"
        elif n == 0:
            rest = [f"{r['pending']} pending"] * bool(r["pending"] or not u) + [unpriced] * bool(u)
            why = (f"0 bets from {_plural(r['calls'], 'call')}, {', '.join(rest)}; needs {gate['min_bets']} bets"
                   if r["calls"] else f"0 bets; needs {gate['min_bets']} bets")
        else:
            why = (f"{_plural(n, 'bet')} ({_plural(hits, 'hit')}) from {_plural(r['calls'], 'call')}"
                   f"{', ' + unpriced if u else ''}; needs {gate['min_bets']} bets")
        out[k] = {**r, "unpriced": u, "q": q, "proven": proven, "weight": float(weight), "why": why}
    return out


def calls_at(bank: Bank, t: str, attrs=("claim", "judgment")) -> list[Fact]:
    """Every directional call in the bank as it stood at t: known and retrieved by then, not downgraded.
    Each entity is read through its own Bank on the journal, so memory stays at one entity's index."""
    out = []
    for e in bank.entities():
        b = Bank(bank.root, use_index=bank.use_index)
        have = set(b.attrs(e))
        v = b.view(t)
        for a in attrs:
            if a in have:
                out += [f for f in v.facts(e, a) if f.context.get("direction") and (f.retrieved_at or f.known_at) <= t]
    return out


def track_record(bank: Bank, t: str | None = None, attrs=("claim", "judgment"), benchmark: str = "SPY",
                 horizon: int = 21, gate: dict | None = None) -> dict[str, dict]:
    """Per source key (claims: "<name> via <noted_by or agent>"; judgments: the source name), the record as it
    stood at time t (default the end of today, as T_RUN):
       {"bets": int, "hits": int, "hit_rate": float | None, "p": float | None,       # one-sided vs a coin
        "q": float | None,                                                          # BH across every source with >= 1 bet
        "lower": float | None,                                                      # Clopper-Pearson one-sided lower bound
        "mean_signed_excess": float | None, "up": int, "down": int,                 # direction mix of the calls
        "calls": int, "dropped": int, "pending": int, "unpriced": int, "unscorable": int,
        "proven": bool, "weight": float, "why": str}
    Rules:
    - a call's anchor is max(known_at, retrieved_at): it is scored only on sessions after the bank held it;
    - only calls retrieved no later than t, on prices known at t (PriceBook asof=t), with horizon_days == horizon;
    - calls on unscorable symbols (decide.scorable) are skipped and counted in `unscorable`;
    - a call with no outcome is `unpriced` once SPY's book is GRACE_SESSIONS past its exit, else `pending`
      (``scored_at``); either is scored with its real outcome once its stock's closes are in the bank;
    - bets() over the scored calls;
    - proven = bets >= min_bets and p < p_lt and q < bh_q and mean_signed_excess > 0, and at most
      UNPRICED_MAX_SHARE of the calls past their exit are unpriced; weight = max(0, 2*lower - 1) when proven, else 0;
    - why is plain words, e.g. "3 bets (2 hits) from 11 calls; needs 30 bets",
      "41 bets, 66% hit, p 0.03, q 0.31 after 12 sources: not proven", "proven: 52 bets, 79% hit, weight 0.35".
    `gate` defaults to POLICY's source_gate."""
    from brain.decide import POLICY, scorable
    gate = gate or POLICY["source_gate"]
    t = t or today() + "T23:59:59Z"
    calls = [f for f in calls_at(bank, t, attrs) if int(f.context.get("horizon_days") or 21) == horizon]
    book = PriceBook(bank, sorted({f.entity for f in calls if scorable(f.entity)[0]} | {benchmark}), asof=t)
    by: dict[str, dict] = {}
    for f in calls:
        s = by.setdefault(source_key(f), {"calls": 0, "up": 0, "down": 0, "pending": 0, "unpriced": 0, "unscorable": 0,
                                          "scored": []})
        d = int(np.sign(f.context["direction"]))
        s["calls"] += 1
        s["up" if d > 0 else "down"] += 1
        if not scorable(f.entity)[0]:
            s["unscorable"] += 1
            continue
        status, o = scored_at(book, f.entity, anchor(f), horizon, t, benchmark)
        if o is None:
            s[status] += 1
            continue
        s["scored"].append({"ref": f.ref, "ticker": f.entity, "direction": d, **o})
    raw = {}
    for k, s in by.items():
        bs, dropped = bets(s.pop("scored"))
        raw[k] = {**tally(bs, gate["alpha_lower"]), **s, "dropped": dropped}
    return gate_rows(raw, gate)
