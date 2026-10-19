"""The company relationship graph: who sells to, buys from, competes or works with whom, read from 10-K text.

A model reads each 10-K as filed and names the companies it mentions as customers, suppliers, competitors or
partners (docs/GRAPH.md). The model is only a reader. Every relation it returns must quote the filing
verbatim, and the quote must contain the company's name. Anything else is dropped, so a relation the model
remembers from training but the filing does not state never enters the bank. (Which of the stated relations it
chooses to return can still lean on what it remembers; docs/GRAPH.md says how far that goes.)

Each kept relation is a ``derived`` fact on the filer, ``relation.<kind>``, whose value is the counterparty's
ticker (or ``?<normalised name>`` when SEC's ticker list has no match). It carries the filing's ``as_of`` and
``known_at`` and cites the filing fact in ``inputs``, so a view at t sees exactly the relations stated in
10-Ks public by t. A ``relation.scan`` fact per filing records that the filing was read and how many
relations it gave, so "read, none named" differs from "not read".

The graph at t takes, for each company, the relations in its latest 10-K read and known by t (if that 10-K is
at most ``STALE_DAYS`` old). A relation dropped from a later 10-K is gone from then on.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from typing import Callable

from brain import config
from brain.bank import Bank, Fact
from brain.timeutil import days_between

KINDS = ("customer", "supplier", "competitor", "partner")
# Seen from the counterparty: if A names B a customer, A is B's supplier.
INVERSE = {"customer": "supplier", "supplier": "customer", "competitor": "competitor", "partner": "partner"}
FORMS = ("10-K", "10-K405", "10-KT")   # amendments (10-K/A) are mostly Part III only and are skipped
STALE_DAYS = 550                        # a 10-K older than this at t no longer speaks for the company
MAX_PASSAGES = 60                       # per filing, the strongest by cue score, kept in document order
BATCH = 8                               # passages per model request
MAX_CHARS = 1500                        # a longer paragraph is cut at sentence ends into pieces of at most this
LLM_FIXED = {"temperature": 0, "seed": 7, "max_tokens": 1200}
DOC_TTL = 365 * 86400                   # a filed document never changes

SYSTEM = "You read SEC filings and answer with JSON only."
PROMPT = (
    "These passages come from {company}'s annual report (10-K). List every other company the passages name as "
    "{company}'s customer, supplier, competitor or partner.\n"
    "- customer: buys from {company}. supplier: sells to {company} (parts, manufacturing, services, licenses). "
    "competitor: competes with {company}. partner: a joint venture, alliance or collaboration.\n"
    "- Only companies named in the passages. Not unnamed ones (\"Customer A\", \"one distributor\"), not people, "
    "not government agencies, not {company} or its own subsidiaries.\n"
    "- quote: words copied exactly from the passage that name the company and show the relation.\n"
    "- share_pct: the percent of {company}'s revenue or purchases the passage gives for it, else null.\n"
    'Answer with JSON only: {{"relations": [{{"passage": 1, "name": "Company name as written", '
    '"relation": "customer", "share_pct": null, "quote": "exact words"}}]}}. An empty list is fine. /no_think\n\n'
    "{passages}"
)

_CUE = re.compile(r"\b(customers?|clients?|suppliers?|vendors?|supply|suppl(?:y|ies|ied)|manufactur\w*|foundr\w*|"
                  r"compet\w*|rivals?|partner\w*|alliances?|joint ventures?|collaborat\w*|licens\w*|distributors?|"
                  r"resellers?|accounted for|sole[- ]source|single[- ]source|purchases?|sales to)\b", re.I)
_SHARE = re.compile(r"accounted for (?:approximately |about |roughly )?\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*%\s+of "
                    r"(?:our |its |the company's )?(?:total |net |consolidated )?(?:revenue|sales|purchases)", re.I)
_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z0-9&]+(?:\s+[A-Z][a-zA-Z0-9&]+)*")
_GENERIC = re.compile(r"^(?:the\s+)?(?:(?:our|its|a|an|one|two|three|several|certain|other|major|largest|"
                      r"significant|key)\s+)*(?:customers?|clients?|suppliers?|vendors?|distributors?|competitors?|"
                      r"partners?|companies|company|manufacturers?|foundr(?:y|ies)|resellers?|retailers?|"
                      r"oems?|government|u\.?s\.? government)(?:\s+[a-z0-9])?$", re.I)
_SUFFIX = {"inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd", "limited", "plc",
           "llc", "lp", "llp", "holdings", "holding", "group", "sa", "ag", "nv", "se", "the", "de", "new"}


def prompt_sha() -> str:
    return hashlib.sha256((SYSTEM + "\n" + PROMPT).encode()).hexdigest()[:16]


# ---------- text ----------
def filing_text(body: bytes | str) -> list[str]:
    """A filed document (HTML, or the plain text of older filings) as paragraphs of collapsed whitespace."""
    t = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
    if re.search(r"(?i)<(p|div|br|td|font)\b", t):
        # HTML: line breaks in the source mean nothing; paragraphs end at block tags.
        t = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", t)
        t = re.sub(r"(?is)<ix:header\b.*?</ix:header>", " ", t)    # inline XBRL's hidden facts
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h[1-6]|table|td)>", "\n", t)
        t = re.sub(r"<[^>]+>", " ", t)
        blocks = html.unescape(t).replace("\xa0", " ").split("\n")
    else:
        # Plain text (older filings): lines are wrapped; paragraphs end at a blank line.
        blocks = re.split(r"\n\s*\n", html.unescape(re.sub(r"<[^>]+>", " ", t)))
    out = []
    for block in blocks:
        p = norm_space(block)
        if len(p) >= 40:
            out.append(p)
    return out


def norm_space(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip()


def _pieces(p: str) -> list[str]:
    if len(p) <= MAX_CHARS:
        return [p]
    out, cur = [], ""
    for s in re.split(r"(?<=[.;])\s+", p):
        if cur and len(cur) + 1 + len(s) > MAX_CHARS:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return [x[:MAX_CHARS] for x in out]


def score(p: str) -> int:
    """How strongly a passage looks like it names a relation: cue words, a revenue share, proper names."""
    cues = len(_CUE.findall(p))
    if not cues:
        return 0
    return cues + 3 * len(_SHARE.findall(p)) + min(5, len(_PROPER.findall(p)))


def passages(paras: list[str], limit: int = MAX_PASSAGES) -> list[str]:
    """The ``limit`` strongest relation passages, in the order the filing has them."""
    cand = [(i, x) for i, x in enumerate(q for p in paras for q in _pieces(p)) if score(x) >= 2]
    top = sorted(cand, key=lambda ix: (-score(ix[1]), ix[0]))[:limit]
    return [x for _, x in sorted(top)]


# ---------- names ----------
def norm_name(s: str) -> str:
    """A company name reduced for matching: lower case, '&' as 'and', no punctuation or legal suffixes."""
    s = s.lower().replace("&", " and ")
    s = re.sub(r"/[a-z]{2}/?", " ", s)        # SEC titles carry a state: 'APPLE INC /CA/'
    words = re.sub(r"[^a-z0-9]+", " ", s).split()
    while words and words[-1] in _SUFFIX:
        words.pop()
    while words and words[0] == "the":
        words.pop(0)
    return " ".join(words)


class Resolver:
    """Company name -> ticker, from SEC's ticker list (today's list: see docs/GRAPH.md on what that misses).
    A name matches when its normalised form equals a listed title's. Several tickers on one title (share
    classes) resolve to the first listed, which is the larger by SEC's ordering."""

    def __init__(self, table: list[dict]):
        self.by_name: dict[str, str] = {}
        self.tickers = set()
        for row in table:
            self.tickers.add(row["ticker"].upper())
            n = norm_name(row["title"])
            if n:
                self.by_name.setdefault(n, row["ticker"].upper())

    @classmethod
    def from_sec(cls) -> "Resolver":
        from brain.sources import sec
        return cls(sec.ticker_table())

    def resolve(self, name: str) -> str | None:
        return self.by_name.get(norm_name(name))


def node(name: str, resolver: Resolver | None) -> str:
    """The value a relation fact stores: a ticker, or '?' and the normalised name."""
    t = resolver.resolve(name) if resolver else None
    return t or "?" + norm_name(name)


# ---------- the model's answer ----------
def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%?\s*", v)
        return float(m.group(1)) if m else None
    return None


def check(name: str, kind: str, quote: str, passage: str, share=None) -> tuple[dict | None, str]:
    """One relation held against the text it claims to come from: (relation, "") or (None, why not).
    The quote must be in the passage word for word (whitespace and typographic quotes aside) and must contain
    the name. A share is kept only if its number is written in the quote."""
    name, kind, quote = norm_space(str(name or "")), str(kind or "").strip().lower(), norm_space(str(quote or ""))
    if kind not in KINDS:
        return None, "relation is not customer, supplier, competitor or partner"
    if len(name) < 2 or not norm_name(name) or _GENERIC.match(name) or re.match(r"(?i)^(our|its)\b", name):
        return None, "no company name"
    if not quote or quote.lower() not in norm_space(passage).lower():
        return None, "quote not in the passage"
    if name.lower() not in quote.lower():
        return None, "name not in the quote"
    pct = _num(share)
    if pct is not None:
        txt = f"{pct:g}"
        if not (0 < pct <= 100 and re.search(rf"(?<![\d.]){re.escape(txt)}(?:\.0+)?\s*(?:%|percent)", quote, re.I)):
            pct = None
    return {"name": name, "relation": kind, "quote": quote, "share_pct": pct}, ""


def parse(text: str, batch: list[str]) -> tuple[list[dict], dict[str, int]]:
    """The relations in one answer that pass ``check`` against their passage, and why the others failed."""
    from brain.judge import clean
    rejected: dict[str, int] = {}

    def drop(why: str) -> None:
        rejected[why] = rejected.get(why, 0) + 1

    s = clean(text)
    try:
        obj = json.loads(s[s.find("{"): s.rfind("}") + 1]) if "{" in s else None
    except ValueError:
        obj = None
    items = obj.get("relations") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        drop("no relations list in the answer")
        return [], rejected
    out = []
    for it in items:
        if not isinstance(it, dict):
            drop("not an object")
            continue
        n = it.get("passage")
        n = int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) or (isinstance(n, str) and n.isdigit()) \
            else None
        if n is None or not 1 <= n <= len(batch):
            drop("no such passage")
            continue
        rel, why = check(it.get("name"), it.get("relation"), it.get("quote"), batch[n - 1], it.get("share_pct"))
        if rel is None:
            drop(why)
        else:
            out.append(rel)
    return out, rejected


def ask(client, company: str, batch: list[str]) -> str:
    listed = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(batch, 1))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(company=company, passages=listed)}]
    reply = client.complete(msgs, temperature=LLM_FIXED["temperature"], seed=LLM_FIXED["seed"],
                            max_tokens=LLM_FIXED["max_tokens"])
    return str(reply["choices"][0]["message"].get("content") or "")


# ---------- facts ----------
def tenks(bank: Bank, ticker: str, since: str | None = None) -> list[Fact]:
    fs = [f for f in bank.facts(ticker, "filing") if f.value in FORMS and (not since or f.as_of >= since)]
    return sorted(fs, key=lambda f: (f.known_at, f.id))


def _relation_fact(filing: Fact, rel: dict, value: str, extractor: str) -> Fact:
    ctx = {"accn": filing.context.get("accn"), "form": filing.value, "name": rel["name"], "quote": rel["quote"],
           "share_pct": rel["share_pct"], "extractor": extractor}
    if extractor != "agent":
        ctx["prompt_sha"] = prompt_sha()
    return Fact(entity=filing.entity, attr=f"relation.{rel['relation']}", value=value, as_of=filing.as_of,
                known_at=filing.known_at, tier="derived",
                source={"name": f"SEC EDGAR {filing.value} text, read by {extractor}", "url": filing.source["url"]},
                context=ctx, inputs=(filing.ref,))


def _scan_fact(filing: Fact, n: int, n_passages: int, rejected: dict, text_sha: str, extractor: str) -> Fact:
    return Fact(entity=filing.entity, attr="relation.scan", value=n, as_of=filing.as_of, known_at=filing.known_at,
                tier="derived",
                source={"name": f"SEC EDGAR {filing.value} text, read by {extractor}", "url": filing.source["url"]},
                context={"accn": filing.context.get("accn"), "form": filing.value, "passages": n_passages,
                         "rejected": dict(sorted(rejected.items())), "text_sha": text_sha, "extractor": extractor,
                         "prompt_sha": prompt_sha()},
                inputs=(filing.ref,))


def fetch_document(url: str) -> bytes:
    from brain.net import fetch
    return fetch(url, user_agent=config.sec_user_agent(), ttl=DOC_TTL).body


def read_filing(filing: Fact, client, resolver: Resolver | None, company: str,
                fetch_doc: Callable[[str], bytes] = fetch_document) -> list[Fact]:
    """One 10-K read by the model: its relation facts (one per counterparty and kind) and its scan fact."""
    body = fetch_doc(filing.source["url"])
    paras = filing_text(body)
    ps = passages(paras)
    rejected: dict[str, int] = {}
    kept: dict[tuple, dict] = {}
    for i in range(0, len(ps), BATCH):
        batch = ps[i:i + BATCH]
        rels, rej = parse(ask(client, company, batch), batch)
        for k, v in rej.items():
            rejected[k] = rejected.get(k, 0) + v
        for r in rels:
            value = node(r["name"], resolver)
            if value == filing.entity.upper():
                rejected["the filer itself"] = rejected.get("the filer itself", 0) + 1
                continue
            key = (value, r["relation"])
            if key not in kept or (r["share_pct"] is not None and kept[key]["rel"]["share_pct"] is None):
                kept[key] = {"value": value, "rel": r}
    extractor = str(getattr(client, "model", "model"))
    out = [_relation_fact(filing, k["rel"], k["value"], extractor) for k in kept.values()]
    out.append(_scan_fact(filing, len(out), len(ps), rejected,
                          hashlib.sha256("\n".join(paras).encode()).hexdigest()[:16], extractor))
    return out


def scanned(bank: Bank, ticker: str, extractor: str) -> set[str]:
    """Accession numbers of the ticker's filings already read by this extractor and prompt."""
    return {f.context.get("accn") for f in bank.facts(ticker, "relation.scan")
            if f.context.get("extractor") == extractor and f.context.get("prompt_sha") == prompt_sha()}


def extract(bank: Bank, ticker: str, client, resolver: Resolver | None = None, *, since: str | None = None,
            max_filings: int | None = None, fetch_doc: Callable[[str], bytes] = fetch_document,
            log: Callable[[str], None] | None = print) -> dict:
    """Read the ticker's 10-Ks in the bank (``brain ingest`` lists them) not yet read by this model and prompt."""
    ticker = ticker.upper()
    done = scanned(bank, ticker, str(getattr(client, "model", "model")))
    todo = [f for f in tenks(bank, ticker, since) if f.context.get("accn") not in done]
    if max_filings is not None:
        todo = todo[-max_filings:] if max_filings > 0 else []
    name = (bank.view().latest(ticker, "identity.name") or None)
    company = str(name.value) if name else ticker
    from brain.llm import LLMError
    from brain.net import FetchError
    out = {"filings": 0, "relations": 0, "failed": []}
    for f in todo:
        try:
            facts = read_filing(f, client, resolver, company, fetch_doc)
        except (FetchError, LLMError, KeyError, OSError) as e:  # one failed filing must not stop the rest
            out["failed"].append(f"{f.context.get('accn')}: {e}")
            if log:
                log(f"{ticker} {f.as_of} {f.context.get('accn')}: failed ({e})")
            continue
        bank.add_many(facts)
        n = len(facts) - 1
        out["filings"] += 1
        out["relations"] += n
        if log:
            log(f"{ticker} {f.value} {f.as_of}: {n} relations from {facts[-1].context['passages']} passages")
    return out


def add_by_agent(bank: Bank, ticker: str, accn: str, name: str, kind: str, quote: str, share=None,
                 resolver: Resolver | None = None, fetch_doc: Callable[[str], bytes] = fetch_document) -> Fact:
    """An agent's own reading of a 10-K, held to the same rule: the quote must be in the filing and name the company.
    Raises ValueError when it is not, or when the bank has no such 10-K for the ticker."""
    filing = next((f for f in tenks(bank, ticker) if f.context.get("accn") == accn), None)
    if filing is None:
        raise ValueError(f"no 10-K {accn} for {ticker.upper()} in the bank (run brain ingest {ticker.upper()})")
    text = "\n".join(filing_text(fetch_doc(filing.source["url"])))
    # The quote may run across what the parser split into paragraphs, so it is checked against the whole text.
    rel, why = check(name, kind, quote, norm_space(text), share)
    if rel is None:
        raise ValueError(why)
    value = node(rel["name"], resolver)
    if value == filing.entity.upper():
        raise ValueError("the filer itself")
    f = _relation_fact(filing, rel, value, "agent")
    bank.add(f)
    return f


# ---------- the graph ----------
@dataclass(frozen=True)
class Edge:
    src: str          # the filer that states the relation
    dst: str          # the counterparty: a ticker, or '?name'
    relation: str     # as the filer states it: dst is src's customer, supplier, competitor or partner
    share_pct: float | None
    as_of: str
    known_at: str
    ref: str          # the relation fact, ENTITY/id


def latest_scan(view, ticker: str) -> Fact | None:
    """The fact that marks the latest 10-K read and known by the view's time, if that 10-K is not stale: its
    model scan, or an agent's relation when no model has read it."""
    read = view.facts(ticker, "relation.scan")
    for kind in KINDS:
        read += view.facts(ticker, f"relation.{kind}", lambda x: x.context.get("extractor") == "agent")
    if not read:
        return None
    f = max(read, key=lambda s: (s.as_of, s.known_at, s.attr == "relation.scan"))
    return f if days_between(f.as_of, view.t) <= STALE_DAYS else None


def edges_of(view, ticker: str) -> list[Edge]:
    """The relations the ticker's latest read 10-K states, as known at the view's time. Agent readings of that
    same 10-K count with the model's."""
    scan = latest_scan(view, ticker)
    if scan is None:
        return []
    accn = scan.context.get("accn")
    seen, out = set(), []
    for kind in KINDS:
        for f in view.facts(ticker, f"relation.{kind}", lambda x: x.context.get("accn") == accn):
            if (f.value, kind) in seen:
                continue
            seen.add((f.value, kind))
            out.append(Edge(f.entity, str(f.value), kind, f.context.get("share_pct"), f.as_of, f.known_at, f.ref))
    return out


def graph_at(bank: Bank, tickers, t: str | None = None) -> list[Edge]:
    """Every edge the given companies' 10-Ks state, as known at t."""
    view = bank.view(t)
    return [e for tk in tickers for e in edges_of(view, tk)]


def neighbours(edges: list[Edge], ticker: str) -> list[tuple[str, str, Edge]]:
    """(counterparty, its role for ``ticker``, edge) from both directions: what the ticker's own 10-K states, and
    what other companies' 10-Ks state about it (turned round: if A names the ticker a customer, A is its supplier)."""
    t = ticker.upper()
    out = []
    for e in edges:
        if e.src == t:
            out.append((e.dst, e.relation, e))
        elif e.dst == t:
            out.append((e.src, INVERSE[e.relation], e))
    return out


# ---------- CLI ----------
def _cmd(a) -> int:
    bank = Bank()
    if a.action == "extract":
        if not a.words:
            print("graph extract: name one or more tickers")
            return 2
        from brain import llm
        client = llm.settings()
        ok, why = client.available()
        if not ok:
            print(f"graph extract: no model to read with: {why}")
            return 1
        resolver = Resolver.from_sec()
        bad = 0
        for t in a.words:
            out = extract(bank, t, client, resolver, since=a.since, max_filings=a.max)
            print(f"{t.upper()}: {out['filings']} 10-Ks read, {out['relations']} relations"
                  + (f", {len(out['failed'])} failed" if out["failed"] else ""))
            bad += bool(out["failed"])
        return 1 if bad else 0
    if a.action == "add":
        if len(a.words) != 2 or not (a.name and a.relation and a.quote):
            print("graph add TICKER ACCESSION --name NAME --relation KIND --quote 'exact words' [--share PCT]")
            return 2
        try:
            f = add_by_agent(bank, a.words[0], a.words[1], a.name, a.relation, a.quote, a.share, Resolver.from_sec())
        except ValueError as e:
            print(f"graph add: refused: {e}")
            return 1
        print(f"stored {f.ref}: {f.entity} names {f.value} its {f.attr.split('.', 1)[1]}")
        return 0
    if a.action == "show":
        if len(a.words) != 1:
            print("graph show TICKER [--at DATE]")
            return 2
        t = a.words[0].upper()
        view = bank.view(a.at)
        scan = latest_scan(view, t)
        others = [x for x in bank.entities() if x != t]
        edges = edges_of(view, t) + [e for x in others for e in edges_of(view, x) if e.dst == t]
        print(f"{t} at {view.t}: " + (f"own 10-K for {scan.as_of} (filed {scan.known_at[:10]})" if scan
                                         else "no 10-K of its own read yet"))
        rows = neighbours(edges, t)
        if not rows:
            print("  no relations known")
        for other, role, e in sorted(rows, key=lambda r: (r[1], r[0])):
            pct = f" {e.share_pct:g}%" if e.share_pct is not None else ""
            said = "its 10-K" if e.src == t else f"{e.src}'s 10-K"
            print(f"  {role:<10} {other:<24}{pct:<6} per {said} {e.as_of}  [{e.ref}]")
        return 0
    if a.action == "stats":
        view = bank.view(a.at)
        n_co = n_edges = n_res = 0
        by_kind: dict[str, int] = {}
        for t in bank.entities():
            es = edges_of(view, t)
            if latest_scan(view, t) is None:
                continue
            n_co += 1
            n_edges += len(es)
            n_res += sum(not e.dst.startswith("?") for e in es)
            for e in es:
                by_kind[e.relation] = by_kind.get(e.relation, 0) + 1
        print(f"at {view.t}: {n_co} companies with a current 10-K read, {n_edges} relations "
              f"({n_res} to a listed ticker): " + ", ".join(f"{k} {by_kind.get(k, 0)}" for k in KINDS))
        return 0
    return 2


def add_cli(sub) -> None:
    s = sub.add_parser("graph", help="company relationship graph read from 10-Ks (docs/GRAPH.md): "
                                     "extract TICKER... | add TICKER ACCESSION | show TICKER | stats")
    s.add_argument("action", choices=["extract", "add", "show", "stats"])
    s.add_argument("words", nargs="*", metavar="TICKER | ACCESSION")
    s.add_argument("--since", help="extract: only 10-Ks for periods ending on or after this date")
    s.add_argument("--max", type=int, help="extract: at most this many 10-Ks per ticker, the latest")
    s.add_argument("--at", help="show/stats: the graph as known at this time (default now)")
    s.add_argument("--name", help="add: the company as the filing names it")
    s.add_argument("--relation", choices=KINDS, help="add: what it is to the filer")
    s.add_argument("--quote", help="add: words copied exactly from the 10-K that name it and show the relation")
    s.add_argument("--share", help="add: the percent of revenue or purchases, if the quote gives one")
    s.set_defaults(fn=_cmd)
