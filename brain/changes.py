"""Quiet changes: how much each 10-K and 10-Q differs from the same report a year earlier (docs/CHANGES.md).

Companies rewrite their reports a little every year. Most of the text stays; what changes (a new risk, a customer
no longer named, softer wording) is easy to miss, because almost nobody compares the two documents. Cohen, Malloy
and Nguyen ("Lazy Prices", Journal of Finance 2020) found that firms whose reports changed most did worse
afterwards, and slowly. This module measures the change; whether it predicts anything here is H10's question.

Each 10-K is paired with the 10-K whose period ended about a year earlier, each 10-Q with the 10-Q a year
earlier (the same fiscal quarter). The comparison is plain arithmetic on the words, with no model involved, so
it can be tested on history without a model's memory leaking in:

- ``cosine``: cosine similarity of the two documents' word counts (letters only, so numbers that change every
  year do not count; common words such as "the" and "of" left out, ``STOP``, or they swamp the rest). 1 means
  the same words in the same proportions.
- ``jaccard``: shared distinct words over all distinct words.
- ``new_share``: the share of the new report's sentences that were not in the old one (numbers and month names
  ignored, so "the three months ended July 27, 2025" matches the year before).

The result is a ``derived`` fact, ``text.similarity`` (value: the cosine), on the filer. It carries the newer
filing's ``as_of`` and ``known_at`` and cites both filings, so a view at t sees only comparisons whose newer
report was public by t.

Documents are fetched once from SEC and kept as their sentences under ``<home>/text`` (gzip). The raw document is
not kept.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter
from typing import Callable

from brain import config
from brain.bank import Bank, Fact
from brain.timeutil import days_between

FAMILY = {"10-K": "10-K", "10-K405": "10-K", "10-KT": "10-K", "10-Q": "10-Q"}  # amendments are skipped
PAIR_DAYS = (300, 430)      # the earlier report's period ends this many days before the newer one's
MAX_AGE_DAYS = 400          # ``latest``: a comparison older than this no longer speaks for the company
MIN_HISTORY = 4             # ``unusual``: earlier comparisons of the same form needed to say what is usual
METHOD = 1                  # bump when the text, sentence or similarity rules change; facts record it
ATTR = "text.similarity"
STOP = frozenset("""a an the and or but if of to in on at by for with from as is are was were be been being it its this
that these those we our us they their them he she his her you your not no nor so than too very can could may might
will would shall should must do does did done have has had having such which who whom whose what when where why
how all any both each few more most other some own same only into over under about above below up down out off
again further then once here there also per via upon within without between among through during before after
while""".split())
_MONTHS = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b")

# Sentences an analyst would want to see first when they are new: risks, troubles, who the company depends on.
RISK = re.compile(r"\b(going concern|material weakness|restat\w*|investigat\w*|subpoena\w*|indict\w*|"
                  r"wells notice|impair\w*|default\w*|covenant\w*|waiver|bankrupt\w*|liquidity|delist\w*|"
                  r"recall\w*|terminat\w*|non-renewal|did not renew|lost|loss of|no longer|discontinu\w*|"
                  r"downgrad\w*|cybersecurity incident|breach\w*|ransomware|lawsuit|litigation|settlement|"
                  r"tariff\w*|sanction\w*|export control\w*|strike|work stoppage|shortage\w*|sole source|"
                  r"single source|customer\w*|supplier\w*|concentration|accounted for)\b", re.I)


# ---------- text ----------
def sentences(paras: list[str]) -> list[str]:
    """Sentences of at least 30 characters and 4 words, in document order."""
    out = []
    for p in paras:
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", p):
            s = s.strip()
            if len(s) >= 30 and len(re.findall(r"[A-Za-z]{2,}", s)) >= 4:
                out.append(s)
    return out


def norm_sentence(s: str) -> str:
    """A sentence for set comparison: lower case, every number as '#', punctuation and spacing ignored."""
    s = _MONTHS.sub("#m", re.sub(r"\d+(?:[.,]\d+)*", "#", s.lower()))
    return re.sub(r"[^a-z#]+", " ", s).strip()


def word_counts(sents: list[str]) -> Counter:
    c: Counter = Counter()
    for s in sents:
        c.update(w for w in re.findall(r"[a-z]{2,}", s.lower()) if w not in STOP)
    return c


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return float("nan")
    small, big = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(v * big.get(k, 0) for k, v in small.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def jaccard(a: Counter, b: Counter) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else float("nan")


def compare(new: list[str], old: list[str]) -> dict:
    wn, wo = word_counts(new), word_counts(old)
    on = {norm_sentence(s) for s in old}
    nn = [norm_sentence(s) for s in new]
    fresh = sum(1 for s in nn if s not in on)
    return {"cosine": round(cosine(wn, wo), 6), "jaccard": round(jaccard(wn, wo), 6),
            "new_share": round(fresh / len(nn), 6) if nn else float("nan"),
            "words": sum(wn.values()), "prev_words": sum(wo.values()),
            "sentences": len(nn), "prev_sentences": len(on)}


def new_sentences(new: list[str], old: list[str], limit: int = 12) -> list[str]:
    """Sentences of the new report not in the old one: those naming a risk or a dependency first, then the longest."""
    on = {norm_sentence(s) for s in old}
    seen, fresh = set(), []
    for s in new:
        n = norm_sentence(s)
        if n not in on and n not in seen:
            seen.add(n)
            fresh.append(s)
    fresh.sort(key=lambda s: (-len(RISK.findall(s)), -len(s)))
    return fresh[:limit]


# ---------- documents ----------
def _text_path(accn: str):
    d = config.home() / "text"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{re.sub(r'[^0-9A-Za-z-]', '_', accn)}.json.gz"


def fetch_document(url: str) -> bytes:
    from brain.net import fetch
    return fetch(url, user_agent=config.sec_user_agent(), ttl=0, store=False).body


def document(filing: Fact, fetch_doc: Callable[[str], bytes] = fetch_document) -> list[str]:
    """The filing's sentences, from the text cache or fetched from SEC once."""
    from brain.graph import filing_text
    accn = str(filing.context.get("accn"))
    p = _text_path(accn)
    if p.exists():
        return json.loads(gzip.decompress(p.read_bytes()))["sentences"]
    sents = sentences(filing_text(fetch_doc(filing.source["url"])))
    p.write_bytes(gzip.compress(json.dumps({"accn": accn, "url": filing.source["url"], "method": METHOD,
                                            "sentences": sents}, ensure_ascii=False).encode()))
    return sents


# ---------- pairs and facts ----------
def reports(bank: Bank, ticker: str, since: str | None = None) -> list[Fact]:
    fs = [f for f in bank.facts(ticker, "filing") if f.value in FAMILY and (not since or f.as_of >= since)]
    return sorted(fs, key=lambda f: (f.as_of, f.known_at, f.id))


def pairs(fs: list[Fact]) -> list[tuple[Fact, Fact]]:
    """(newer, earlier) for each report with a same-family report whose period ended 300-430 days before it
    (the one nearest a year) and that was public before it. One pair per newer period: when a period was filed
    twice, the first filing."""
    out, done = [], set()
    for f in fs:
        key = (FAMILY[f.value], f.as_of)
        if key in done:
            continue
        cands = [g for g in fs if FAMILY[g.value] == FAMILY[f.value] and g.known_at < f.known_at
                 and PAIR_DAYS[0] <= days_between(g.as_of, f.as_of) <= PAIR_DAYS[1]]
        if cands:
            done.add(key)
            out.append((f, min(cands, key=lambda g: (abs(days_between(g.as_of, f.as_of) - 365), g.known_at))))
    return out


def similarity_fact(new: Fact, old: Fact, cmp: dict) -> Fact:
    ctx = {"form": FAMILY[new.value], "accn": new.context.get("accn"), "prev_accn": old.context.get("accn"),
           "prev_as_of": old.as_of, "method": METHOD, **{k: v for k, v in cmp.items() if k != "cosine"}}
    return Fact(entity=new.entity, attr=ATTR, value=cmp["cosine"], as_of=new.as_of, known_at=new.known_at,
                tier="derived", source={"name": f"computed from SEC EDGAR {new.value} text", "url": new.source["url"]},
                context=ctx, inputs=(new.ref, old.ref))


def build(bank: Bank, ticker: str, since: str | None = None, fetch_doc: Callable[[str], bytes] = fetch_document,
          log: Callable[[str], None] | None = print) -> dict:
    """Compare every pair of the ticker's reports in the bank not yet compared under this METHOD."""
    ticker = ticker.upper()
    done = {(f.context.get("accn"), f.context.get("prev_accn")) for f in bank.facts(ticker, ATTR)
            if f.context.get("method") == METHOD}
    todo = [(n, o) for n, o in pairs(reports(bank, ticker, since))
            if (n.context.get("accn"), o.context.get("accn")) not in done]
    from brain.net import FetchError
    out = {"compared": 0, "failed": []}
    facts = []
    for n, o in todo:
        try:
            cmp = compare(document(n, fetch_doc), document(o, fetch_doc))
        except (FetchError, OSError, KeyError) as e:
            out["failed"].append(f"{n.context.get('accn')}: {e}")
            continue
        if not cmp["sentences"] or not cmp["prev_sentences"]:
            out["failed"].append(f"{n.context.get('accn')}: no text")
            continue
        facts.append(similarity_fact(n, o, cmp))
        out["compared"] += 1
    bank.add_many(facts)
    if log:
        log(f"{ticker}: {out['compared']} compared" + (f", {len(out['failed'])} failed" if out["failed"] else ""))
    return out


def latest(view, ticker: str, max_age: int = MAX_AGE_DAYS) -> Fact | None:
    """The comparison of the ticker's latest report known at the view's time, if that report is recent."""
    fs = view.facts(ticker, ATTR, lambda f: f.context.get("method") == METHOD)
    if not fs:
        return None
    f = max(fs, key=lambda x: (x.known_at, x.as_of, x.id))
    return f if days_between(f.known_at, view.t) <= max_age else None


def unusual(view, ticker: str, max_age: int = MAX_AGE_DAYS) -> tuple[float, Fact] | None:
    """How unusual the ticker's latest report is for this company: its similarity minus the mean of the company's
    earlier comparisons of the same form, in their standard deviations (the earlier ones known before it). Below 0
    means it changed more than usual. Scored against the company's own same-form history because 10-Ks run more
    similar year to year than 10-Qs (medians about 0.99 and 0.97 on the large caps), and some companies rewrite
    more than others. None without a recent report or with fewer than ``MIN_HISTORY`` earlier ones."""
    f = latest(view, ticker, max_age)
    if f is None:
        return None
    past = [g.value for g in history(view, ticker)
            if g.context.get("form") == f.context.get("form") and g.known_at < f.known_at]
    if len(past) < MIN_HISTORY:
        return None
    mu = sum(past) / len(past)
    sd = math.sqrt(sum((x - mu) ** 2 for x in past) / len(past))
    if sd <= 0:
        return None
    return (f.value - mu) / sd, f


def history(view, ticker: str) -> list[Fact]:
    fs = view.facts(ticker, ATTR, lambda f: f.context.get("method") == METHOD)
    return sorted(fs, key=lambda x: (x.known_at, x.id))


# ---------- CLI ----------
def _tickers(a) -> list[str]:
    if a.universe:
        from brain.research.run import load_universe
        return load_universe(None if a.universe == "large" else "universe/us_broad.json")["tickers"]
    return [t.upper() for t in a.tickers]


def _bar(x: float) -> str:
    return "#" * max(0, min(20, int(round((1 - x) * 100))))


def _cmd(a) -> int:
    bank = Bank()
    if a.action == "build":
        tickers = _tickers(a)
        if not tickers:
            print("changes build TICKER... | --universe large")
            return 2
        bad = 0
        for t in tickers:
            out = build(Bank(bank.root), t, a.since)
            bad += bool(out["failed"])
        return 1 if bad else 0
    if a.action == "show":
        if len(a.tickers) != 1:
            print("changes show TICKER [--at DATE] [--n 12]")
            return 2
        t = a.tickers[0].upper()
        view = bank.view(a.at)
        hist = history(view, t)
        if not hist:
            print(f"{t}: no comparisons yet (brain ingest {t}; brain changes build {t})")
            return 1
        print(f"{t}: each report against the same report a year earlier (cosine of word counts; 1 = unchanged)")
        for f in hist[-12:]:
            c = f.context
            print(f"  {c['form']:<5} {f.as_of} filed {f.known_at[:10]}  similarity {f.value:.3f}  "
                  f"new sentences {c['new_share']:.0%}  {_bar(f.value)}")
        f = hist[-1]
        new = next((g for g in bank.facts(t, "filing") if g.ref == f.inputs[0]), None)
        old = next((g for g in bank.facts(t, "filing") if g.ref == f.inputs[1]), None)
        if new and old:
            print(f"\nNew in the {f.context['form']} for {f.as_of} (not in the one for {old.as_of}), "
                  f"risks and dependencies first:")
            for s in new_sentences(document(new), document(old), a.n):
                print(f"  - {s[:400]}")
        return 0
    if a.action == "recent":
        view = bank.view(a.at)
        tickers = _tickers(a) or bank.entities()
        rows = []
        for t in tickers:
            f = latest(view, t, max_age=a.days)
            if f is not None:
                rows.append(f)
        if not rows:
            print(f"no comparisons of reports filed in the last {a.days} days")
            return 0
        print(f"Reports filed in the {a.days} days to {view.t}, most changed first:")
        for f in sorted(rows, key=lambda x: x.value)[: a.n]:
            print(f"  {f.entity:<6} {f.context['form']:<5} {f.as_of} filed {f.known_at[:10]}  similarity "
                  f"{f.value:.3f}  new sentences {f.context['new_share']:.0%}")
        return 0
    return 2


def add_cli(sub) -> None:
    s = sub.add_parser("changes", help="quiet changes in 10-Ks and 10-Qs against the same report a year earlier "
                                       "(docs/CHANGES.md): build TICKER... | show TICKER | recent")
    s.add_argument("action", choices=["build", "show", "recent"])
    s.add_argument("tickers", nargs="*", metavar="TICKER")
    s.add_argument("--universe", choices=["large", "broad"], help="build/recent: every ticker of a universe")
    s.add_argument("--since", default="2008-01-01", help="build: reports for periods ending on or after this date")
    s.add_argument("--at", help="show/recent: as known at this time (default now)")
    s.add_argument("--days", type=int, default=14, help="recent: reports filed in this many days")
    s.add_argument("--n", type=int, default=12, help="show: new sentences to list; recent: rows")
    s.set_defaults(fn=_cmd)
