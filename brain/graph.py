"""The company relationship graph: who sells to, buys from, competes or works with whom, read from 10-K text.

A model reads each 10-K as filed and names the companies it mentions as customers, suppliers, competitors or
partners (docs/GRAPH.md). The model is only a reader. Every relation it returns must quote the filing
verbatim, and the quote must contain the company's name. Anything else is dropped, so a relation the model
remembers from training but the filing does not state never enters the bank. (Which of the stated relations it
chooses to return can still lean on what it remembers; docs/GRAPH.md says how far that goes.)

Each kept relation is a ``derived`` fact on the filer, ``relation.<kind>``, whose value is the counterparty's
normalised name, as the filing writes it (``Resolver`` maps names to tickers when the graph is read, so a better
matcher never rewrites a fact). It carries the filing's ``as_of`` and
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
import unicodedata
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
SCHEMA = 2                              # 2: relation values are names, resolved when read (1 stored tickers)

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
# Trailing words that name a unit of a company rather than the company ('Amazon Web Services', 'Bayer HealthCare').
DIVISION = {"web", "services", "service", "healthcare", "health", "financial", "finance", "capital", "credit",
            "leasing", "usa", "us", "america", "americas", "north", "international", "global", "worldwide",
            "technologies", "technology", "energy", "pharma", "pharmaceuticals", "aerospace", "aviation", "and", "of"}
_DEPOSITARY = re.compile(r"(?i)^\s*(citibank|jpmorgan chase bank|the bank of new york mellon|bank of new york mellon|"
                         r"deutsche bank trust)\b.*/\s*adr\b")
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
    """A company name reduced for matching: accents and apostrophes dropped, lower case, '&' as 'and', no
    punctuation or legal suffixes. 'Lowe's Companies, Inc.' and 'LOWES COMPANIES INC' both give 'lowes'."""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"(?<=\w)['’](?=\w)", "", s.lower()).replace("&", " and ")
    s = re.sub(r"/\s*adr\b.*$", " ", s)       # an ADR listing: 'Kioxia Holdings Corporation/ADR'
    s = re.sub(r"/[a-z]{2}/?", " ", s)        # SEC titles carry a state: 'APPLE INC /CA/'
    words = re.sub(r"[^a-z0-9]+", " ", s).split()
    while words and words[-1] in _SUFFIX:
        words.pop()
    while words and words[0] == "the":
        words.pop(0)
    return " ".join(words)


class Resolver:
    """Company name -> ticker, from SEC's ticker list (today's list: see docs/GRAPH.md on what that misses).
    Tried in order, on the normalised name; each answer says which rule matched:

    - ``exact``: it equals a listed title ('Lowe's Companies' -> LOW).
    - ``prefix``: it is the first whole words of exactly one listed company's title, and at least 4 characters
      ('Amazon' -> 'AMAZON COM INC', 'Cisco' -> 'CISCO SYSTEMS'; 'Mitsubishi' starts several, so no match).
    - ``shortened``: with trailing division words dropped ('USA', 'healthcare', 'capital', ...: ``DIVISION``), it
      equals a listed title ('Legend Biotech USA Inc.' -> LEGN). Exact only: a shortened name matched by prefix
      took 'Philip Morris USA' (Altria's) to PM and 'Banc of America Leasing' to Banc of California.

    Several tickers on one company (share classes, preferreds) resolve to the first listed, which SEC orders by size.
    ADRs that SEC lists under their depositary bank's name ('Citibank,N.A./ADR') are left out."""

    def __init__(self, table: list[dict]):
        self.exact: dict[str, str] = {}
        self.prefix: dict[str, set[str]] = {}
        first: dict[int, str] = {}
        for row in table:
            t = first.setdefault(int(row["cik"]), row["ticker"].upper())
            n = norm_name(row["title"])
            if not n or _DEPOSITARY.match(row["title"]):
                continue
            self.exact.setdefault(n, t)
            words = n.split()
            for k in range(1, len(words) + 1):
                self.prefix.setdefault(" ".join(words[:k]), set()).add(t)

    @classmethod
    def from_sec(cls) -> "Resolver":
        from brain.sources import sec
        return cls(sec.ticker_table())

    def _match(self, n: str) -> tuple[str | None, str]:
        if n in self.exact:
            return self.exact[n], "exact"
        hits = self.prefix.get(n, set())
        if len(n) >= 4 and len(hits) == 1:
            return next(iter(hits)), "prefix"
        return None, ""

    def match(self, name: str) -> tuple[str | None, str]:
        """(ticker, rule) or (None, "")."""
        n = norm_name(name)
        t, how = self._match(n)
        if t:
            return t, how
        words = n.split()
        while len(words) > 1 and words[-1] in DIVISION:
            words.pop()
            t = self.exact.get(" ".join(words))
            if t:
                return t, "shortened"
        return None, ""

    def resolve(self, name: str) -> str | None:
        return self.match(name)[0]


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
           "share_pct": rel["share_pct"], "extractor": extractor, "schema": SCHEMA}
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
                         "prompt_sha": prompt_sha(), "schema": SCHEMA},
                inputs=(filing.ref,))


def fetch_document(url: str) -> bytes:
    from brain.net import fetch
    return fetch(url, user_agent=config.sec_user_agent(), ttl=DOC_TTL).body


def _read(filing: Fact, fetch_doc: Callable[[str], bytes]) -> tuple[list[str], list[str]]:
    paras = filing_text(fetch_doc(filing.source["url"]))
    return paras, passages(paras)


def _facts(filing: Fact, rels: list[dict], rejected: dict[str, int], extractor: str, paras: list[str],
           ps: list[str]) -> list[Fact]:
    """Checked relations as facts, one per name and kind (the first, or the first with a share), and the filing's
    scan fact. Names are stored as written; which ticker each is (and whether it is the filer) is decided when
    the graph is read."""
    kept: dict[tuple, dict] = {}
    for r in rels:
        value = norm_name(r["name"])
        key = (value, r["relation"])
        if key not in kept or (r["share_pct"] is not None and kept[key]["rel"]["share_pct"] is None):
            kept[key] = {"value": value, "rel": r}
    out = [_relation_fact(filing, k["rel"], k["value"], extractor) for k in kept.values()]
    out.append(_scan_fact(filing, len(out), len(ps), rejected,
                          hashlib.sha256("\n".join(paras).encode()).hexdigest()[:16], extractor))
    return out


def read_filing(filing: Fact, client, company: str, fetch_doc: Callable[[str], bytes] = fetch_document) -> list[Fact]:
    """One 10-K read by the model: its relation facts (one per counterparty and kind) and its scan fact."""
    paras, ps = _read(filing, fetch_doc)
    rels, rejected = [], {}
    for i in range(0, len(ps), BATCH):
        batch = ps[i:i + BATCH]
        got, rej = parse(ask(client, company, batch), batch)
        rels += got
        for k, v in rej.items():
            rejected[k] = rejected.get(k, 0) + v
    return _facts(filing, rels, rejected, str(getattr(client, "model", "model")), paras, ps)


# ---------- an agent reading in the model's place ----------
def packet(filing: Fact, company: str, fetch_doc: Callable[[str], bytes] = fetch_document) -> dict:
    """What the model would be shown for one 10-K, for an agent to read instead: the same instructions and the
    same passages, numbered from 1 across the whole filing. The answer goes to ``record_answer``."""
    paras, ps = _read(filing, fetch_doc)
    head = PROMPT.split("{passages}")[0].replace(" /no_think", "").format(company=company).strip()
    return {"ticker": filing.entity, "accn": filing.context.get("accn"), "form": filing.value,
            "period": filing.as_of, "filed": filing.known_at, "company": company, "prompt_sha": prompt_sha(),
            "text_sha": hashlib.sha256("\n".join(paras).encode()).hexdigest()[:16],
            "instructions": head + ' Also give "ticker" and "accn" as above. Passage numbers are the ones below.',
            "passages": [{"n": i, "text": p} for i, p in enumerate(ps, 1)]}


def record_answer(bank: Bank, answer: str | dict, extractor: str = "agent", fetch_doc: Callable[[str], bytes] = fetch_document) -> list[Fact]:
    """An agent's answer to a ``packet``, held to the model's rules. The passages are rebuilt from the filing
    itself, never taken from the packet, so an edited packet cannot ground a quote. Raises ValueError when the
    answer names no 10-K the bank has for the ticker, or the 10-K was already read by this extractor and prompt."""
    obj = json.loads(answer) if isinstance(answer, str) else answer
    if not isinstance(obj, dict):
        raise ValueError("the answer is not a JSON object")
    ticker, accn = str(obj.get("ticker") or "").upper(), str(obj.get("accn") or "")
    filing = next((f for f in tenks(bank, ticker) if f.context.get("accn") == accn), None) if ticker else None
    if filing is None:
        raise ValueError(f"no 10-K {accn or '?'} for {ticker or '?'} in the bank")
    if accn in scanned(bank, ticker, extractor):
        raise ValueError(f"{ticker} {accn} was already read by {extractor} with this prompt")
    paras, ps = _read(filing, fetch_doc)
    rels, rejected = parse(json.dumps({"relations": obj.get("relations")}), ps)
    facts = _facts(filing, rels, rejected, extractor, paras, ps)
    bank.add_many(facts)
    return facts


def scanned(bank: Bank, ticker: str, extractor: str) -> set[str]:
    """Accession numbers of the ticker's filings already read by this extractor and prompt."""
    return {f.context.get("accn") for f in bank.facts(ticker, "relation.scan")
            if f.context.get("extractor") == extractor and f.context.get("prompt_sha") == prompt_sha()
            and f.context.get("schema") == SCHEMA}


def extract(bank: Bank, ticker: str, client, *, since: str | None = None,
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
            facts = read_filing(f, client, company, fetch_doc)
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
                 fetch_doc: Callable[[str], bytes] = fetch_document) -> Fact:
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
    f = _relation_fact(filing, rel, norm_name(rel["name"]), "agent")
    bank.add(f)
    return f


# ---------- the graph ----------
@dataclass(frozen=True)
class Edge:
    src: str          # the filer that states the relation
    dst: str          # the counterparty: a ticker, or '?' and its normalised name when none matches
    relation: str     # as the filer states it: dst is src's customer, supplier, competitor or partner
    share_pct: float | None
    as_of: str
    known_at: str
    ref: str          # the relation fact, ENTITY/id
    name: str = ""    # the counterparty as the filing writes it
    match: str = ""   # how dst was found: exact | prefix | shortened ('' when unresolved; Resolver.match)


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


def edges_of(view, ticker: str, resolver: Resolver | None = None) -> list[Edge]:
    """The relations the ticker's latest read 10-K states, as known at the view's time, with each counterparty
    resolved to a ticker where the resolver can. Agent readings of that same 10-K count with the model's. Two
    names that resolve to one ticker give one edge per kind (the first, or the first with a share); a name that
    resolves to the filer itself gives none."""
    scan = latest_scan(view, ticker)
    if scan is None:
        return []
    accn = scan.context.get("accn")
    kept: dict[tuple, Edge] = {}
    for kind in KINDS:
        for f in view.facts(ticker, f"relation.{kind}", lambda x: x.context.get("accn") == accn):
            name = str(f.context.get("name") or f.value)
            t, how = resolver.match(name) if resolver else (None, "")
            if t == f.entity:
                continue
            e = Edge(f.entity, t or "?" + str(f.value), kind, f.context.get("share_pct"), f.as_of, f.known_at,
                     f.ref, name, how)
            k = (e.dst, kind)
            if k not in kept or (e.share_pct is not None and kept[k].share_pct is None):
                kept[k] = e
    return list(kept.values())


def graph_at(bank: Bank, tickers, t: str | None = None, resolver: Resolver | None = None) -> list[Edge]:
    """Every edge the given companies' 10-Ks state, as known at t."""
    view = bank.view(t)
    return [e for tk in tickers for e in edges_of(view, tk, resolver)]


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
        bad = 0
        for t in a.words:
            out = extract(bank, t, client, since=a.since, max_filings=a.max)
            print(f"{t.upper()}: {out['filings']} 10-Ks read, {out['relations']} relations"
                  + (f", {len(out['failed'])} failed" if out["failed"] else ""))
            bad += bool(out["failed"])
        return 1 if bad else 0
    if a.action == "add":
        if len(a.words) != 2 or not (a.name and a.relation and a.quote):
            print("graph add TICKER ACCESSION --name NAME --relation KIND --quote 'exact words' [--share PCT]")
            return 2
        try:
            f = add_by_agent(bank, a.words[0], a.words[1], a.name, a.relation, a.quote, a.share)
        except ValueError as e:
            print(f"graph add: refused: {e}")
            return 1
        print(f"stored {f.ref}: {f.entity} names {f.value} its {f.attr.split('.', 1)[1]}")
        return 0
    if a.action == "packet":
        if not a.words or not a.dir:
            print("graph packet TICKER... --dir DIR [--max N | --since DATE]")
            return 2
        from pathlib import Path

        from brain.net import FetchError
        out_dir = Path(a.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        bad = 0
        for t in a.words:
            t = t.upper()
            done = scanned(bank, t, "agent")
            todo = [f for f in tenks(bank, t, a.since) if f.context.get("accn") not in done]
            todo = todo[-(a.max if a.max is not None else 1):] if not a.since or a.max is not None else todo
            name = bank.view().latest(t, "identity.name")
            for f in todo:
                try:
                    pk = packet(f, str(name.value) if name else t)
                except (FetchError, OSError) as e:
                    print(f"{t} {f.context.get('accn')}: failed ({e})")
                    bad += 1
                    continue
                path = out_dir / f"{t}_{pk['accn']}.json"
                path.write_text(json.dumps(pk, indent=1, ensure_ascii=False))
                print(f"{t} {f.value} {f.as_of}: {len(pk['passages'])} passages -> {path}")
            if not todo:
                print(f"{t}: no unread 10-K in the bank (brain ingest {t})")
        return 1 if bad else 0
    if a.action == "answer":
        if not a.words:
            print("graph answer FILE...  (each: {\"ticker\", \"accn\", \"relations\": [...]})")
            return 2
        from pathlib import Path
        bad = 0
        for path in a.words:
            try:
                facts = record_answer(bank, Path(path).read_text())
            except (ValueError, OSError) as e:
                print(f"{path}: refused: {e}")
                bad += 1
                continue
            scan = facts[-1]
            rej = ", ".join(f"{k} {v}" for k, v in scan.context["rejected"].items()) or "none"
            print(f"{scan.entity} {scan.as_of}: {scan.value} relations kept; dropped: {rej}")
        return 1 if bad else 0
    if a.action == "show":
        if len(a.words) != 1:
            print("graph show TICKER [--at DATE]")
            return 2
        t = a.words[0].upper()
        view, resolver = bank.view(a.at), Resolver.from_sec()
        scan = latest_scan(view, t)
        others = [x for x in bank.entities() if x != t]
        edges = edges_of(view, t, resolver) + [e for x in others for e in edges_of(view, x, resolver) if e.dst == t]
        print(f"{t} at {view.t}: " + (f"own 10-K for {scan.as_of} (filed {scan.known_at[:10]})" if scan
                                         else "no 10-K of its own read yet"))
        rows = neighbours(edges, t)
        if not rows:
            print("  no relations known")
        for other, role, e in sorted(rows, key=lambda r: (r[1], r[0])):
            pct = f" {e.share_pct:g}%" if e.share_pct is not None else ""
            said = "its 10-K" if e.src == t else f"{e.src}'s 10-K"
            how = f" ({e.match}: {e.name})" if e.src == t and e.match not in ("", "exact") else ""
            print(f"  {role:<10} {other:<24}{pct:<6} per {said} {e.as_of}{how}  [{e.ref}]")
        return 0
    if a.action == "stats":
        view, resolver = bank.view(a.at), Resolver.from_sec()
        n_co = 0
        es: list[Edge] = []
        for t in bank.entities():
            if latest_scan(view, t) is None:
                continue
            n_co += 1
            es += edges_of(view, t, resolver)
        res = [e for e in es if e.match]
        print(f"at {view.t}: {n_co} companies with a current 10-K read, {len(es)} relations")
        for k in KINDS:
            ks = [e for e in es if e.relation == k]
            print(f"  {k:<10} {len(ks):>5}, {sum(bool(e.match) for e in ks):>5} to a listed ticker")
        print("  matched by: " + ", ".join(f"{m} {sum(e.match == m for e in res)}"
                                           for m in ("exact", "prefix", "shortened")))
        return 0
    return 2


def add_cli(sub) -> None:
    s = sub.add_parser("graph", help="company relationship graph read from 10-Ks (docs/GRAPH.md): "
                                     "extract TICKER... | packet TICKER... --dir DIR | answer FILE... | "
                                     "add TICKER ACCESSION | show TICKER | stats")
    s.add_argument("action", choices=["extract", "packet", "answer", "add", "show", "stats"])
    s.add_argument("words", nargs="*", metavar="TICKER | ACCESSION | FILE")
    s.add_argument("--since", help="extract/packet: only 10-Ks for periods ending on or after this date")
    s.add_argument("--max", type=int, help="extract/packet: at most this many 10-Ks per ticker, the latest "
                                           "(packet: 1 unless --since is given)")
    s.add_argument("--dir", help="packet: where to write one JSON packet per 10-K for an agent to read")
    s.add_argument("--at", help="show/stats: the graph as known at this time (default now)")
    s.add_argument("--name", help="add: the company as the filing names it")
    s.add_argument("--relation", choices=KINDS, help="add: what it is to the filer")
    s.add_argument("--quote", help="add: words copied exactly from the 10-K that name it and show the relation")
    s.add_argument("--share", help="add: the percent of revenue or purchases, if the quote gives one")
    s.set_defaults(fn=_cmd)
