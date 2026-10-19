# The company relationship graph

`brain/graph.py` reads each company's 10-Ks and records whom it names as a customer, supplier, competitor or partner. The graph is the input for H9, a declared hypothesis that news travels slowly along customer–supplier links (step 2 of the plan below). This page covers step 1: building the graph honestly.

## Why it is built this way

A language model knows how 2019–2026 turned out. If it supplies relations from memory ("Apple buys from TSMC"), a backtest is handed the answers. So the model here is only a reader, and every relation it returns must be proved from the filing:

- **The quote is checked.** Each relation must come with words copied from the passage, and those words must appear in the passage exactly (whitespace and typographic quotes aside). Otherwise it is dropped.
- **The quote must contain the company's name.** A real quote that never mentions the name is dropped.
- **Unnamed counterparties are dropped.** "Customer A", "one distributor" and "our largest customer" don't count.
- **A revenue share is kept only if its number is in the quote.** "accounted for 14%" keeps 14; a share the quote doesn't state becomes null.
- **The filer itself is dropped** when its name resolves to its own ticker.

Every drop is counted by reason on the filing's `relation.scan` fact, so you can see how often the model tried to add something the text doesn't say.

**What this does not remove.** The checks prove that each kept relation is stated in the filing. They don't prove the model picked relations without leaning on memory: of the relations a filing states, it may tend to return the ones it knows mattered later. Two things limit that bias:

- passages are chosen by fixed cue words, not by the model;
- the model is never shown dates, prices or what happened next.

It is still a residual bias. Step 3 measures it before any result is trusted.

## Facts

For each kept relation, one `derived` fact on the filer:

| Field | Value |
|---|---|
| `attr` | `relation.customer`, `relation.supplier`, `relation.competitor` or `relation.partner`, as the filer states it (the counterparty is the filer's customer, etc.) |
| `value` | the counterparty's normalised name as the filing writes it (`taiwan semiconductor manufacturing`). Tickers are matched when the graph is read (Name matching, below), so a better matcher never rewrites a fact |
| `as_of`, `known_at` | the 10-K's own: the period it covers and when it became public |
| `inputs` | the 10-K's `filing` fact |
| `context` | `accn`, `form`, `name` as written, `quote`, `share_pct`, `extractor` (the model, or `agent`), `prompt_sha`, `schema` |

For each filing read, one `relation.scan` fact: the number of relations kept, the number of passages read, the drop counts by reason, the sha of the extracted text, the extractor, the prompt sha and the schema. A 10-K counts as read by an extractor only if a scan with the same prompt sha and schema exists, so changing the prompt rereads everything under a new sha. Nothing is overwritten.

Schema 1 stored tickers as values. Its facts from the first reading were downgraded (with that reason) and re-recorded from the same answers as schema 2.

## The graph at a time t

`graph.graph_at(bank, tickers, t)` reads only through `bank.view(t)`:

- For each company, it uses the latest 10-K read and public by t, and only if that 10-K covers a period ending at most 550 days before t. A company that stopped filing drops out.
- It returns that 10-K's relations, plus any an agent added for the same 10-K. With a `Resolver` each counterparty becomes a ticker where one matches (`?name` where none does). Two names for one company give one edge per kind, and a name that matches the filer itself gives none.
- A relation named in one 10-K and missing from the next is gone from the next filing date on.
- `graph.neighbours(edges, T)` adds the other direction. If A's 10-K names T a customer, then A is T's supplier.

**The leak test** (`tests/test_graph.py`): facts known after t, including a later 10-K and a backdated relation recorded later, change nothing in the graph at t.

## Commands

```
python -m brain ingest NVDA                     # lists the 10-Ks (filing facts) the graph reads
python -m brain graph extract NVDA AMD --since 2010-01-01 [--max 3]
python -m brain graph show NVDA [--at 2020-06-30]
python -m brain graph stats [--at 2020-06-30]
python -m brain graph packet NVDA AMD --dir packets/ [--max 3 | --since 2010-01-01]
python -m brain graph answer packets/NVDA_0001045810-26-000021.answer.json
python -m brain graph add NVDA 0001045810-25-000023 --name "..." --relation supplier \
    --quote "exact words from the 10-K" [--share 12]
```

- `extract` needs a model behind `BRAIN_LLM_URL` (`brain/llm.py`). It reads at temperature 0 with a fixed seed: up to 60 cue passages per 10-K, 8 per request.
- `packet` and `answer` let an agent read in the model's place. `packet` writes, for each 10-K, the model's instructions and the same numbered passages. The agent's answer, `{"ticker", "accn", "relations": [...]}` in the model's format, goes to `answer`, which rebuilds the passages from the filing itself (never from the packet file) and applies the same checks. It stores the result with `extractor: agent` and a scan fact, so "read, none named" counts as read.
- `add` is for an agent recording one relation it found anywhere in a 10-K. It is held to the same check, against the whole filing text, and is stored with `extractor: agent`.
- Both fetch the filing from SEC, which needs `SEC_USER_AGENT` with a contact email. SEC refuses requests without one.

## Name matching

`graph.Resolver` matches names against **today's** SEC ticker list. It first normalises them: accents and apostrophes dropped, lower case, `&` as "and", punctuation, legal suffixes, state tags like `/DE/` and `/ADR` removed. Each edge records which rule matched:

| Rule | Matches when | Example |
|---|---|---|
| `exact` | the normalised name equals a listed title | "Lowe's Companies, Inc." → LOW; "Kioxia Holdings Corporation" → KXIAY |
| `prefix` | it is the first whole words of exactly one listed company's title, and at least 4 characters | "Amazon" → AMZN ("AMAZON COM INC"); "Mitsubishi" starts several titles, so no match |
| `shortened` | with trailing division words dropped ("USA", "healthcare", "capital"), it equals a listed title | "Legend Biotech USA Inc." → LEGN |

A shortened name must match exactly. In the first reading, prefix matching after shortening took "Philip Morris USA" (Altria's) to PM and "Banc of America Leasing" to Banc of California. Some ADRs are listed under their depositary bank's name ("Citibank,N.A./ADR"); those titles are left out, since one of them took "Citibank, N.A." to an unrelated ADR. H9 can use `exact` matches alone if the other rules look loose.

Consequences:

- **Delisted, acquired and private counterparties stay unresolved** (`?name`). They remain in the graph as nodes; they just can't be priced. So the coverage of priceable links is lower for older years. This affects coverage only; no returns leak through it.
- **Abbreviations and renamings don't resolve.** "TSMC" stays `?tsmc` unless the filing writes the full name; "Google" doesn't reach Alphabet, and "GE Aerospace" doesn't reach SEC's "GENERAL ELECTRIC CO". A curated alias table would be a declared change.
- **Companies with no US listing stay unresolved**: Samsung Electronics, Airbus, Bayer, Komatsu.
- **A name reused by a different company later** would resolve to the later one. This is rare, and it's checked by eye in `graph show`.

## The first reading (2026-10-19)

Agents read the latest 10-K of each of the 60 `us_large` companies through `packet` and `answer`. Their answers are committed in `knowledge/graph/agent/`. Rebuild with `python -m brain ingest T` for each ticker (the `filing` facts), then `python -m brain graph answer knowledge/graph/agent/*.json`.

- **Relations.** 559 from 60 10-Ks: 72 customer, 75 supplier, 311 competitor, 101 partner. No answer failed the quote check, because the agents ran the same check before answering. So this reading measures nothing about memory; step 3 still does that.
- **Resolved to a listed ticker.** 233 (42%): 211 exact, 21 prefix, 1 shortened. 60 of those are customer or supplier edges.
- **Revenue shares.** Only 2 are stated for a named company: Walmart is 16% of P&G's revenue and 14% of PepsiCo's.
- **Most named.** Microsoft and Amazon (7 each), TSMC (6), NVIDIA and Broadcom (5). The most named unresolved are Samsung Electronics, Discover (now part of Capital One), OpenAI and Airbus.
- **No relations named.** 11 10-Ks: GOOGL, AMZN, CSCO, AMAT, UNH, HD, TXN, WFC, CRM, ETN, ABT. Some name their large customers only as "Customer A" (AMAT's 19% and 15%, CSCO's 11% channel partner). WFC's 10-K puts its business description in an exhibit the packet doesn't read (14 passages); JPM and C are thin for the same reason.

**Judgement calls**, recorded as the readers made them:
- Licensees count as customers (GE HealthCare licensing the GE name, Sony licensing from Disney).
- Parent companies named only as the owner of a counterparty are left out (AbbVie for Pharmacyclics, Cisco for Splunk).
- Acquisitions, executives' past employers, litigants and regulators are left out.

**Known limits:**
- A relation the 10-K says has ended can still be recorded. P&G's Glad venture with Clorox is recorded, though the filing says it expired in January 2026.
- A share given for a segment (Airbus at 29% of Pratt & Whitney) or a subsidiary (Walmart at 17.2% of McLane) is not a share of the filer's revenue, and was left empty.

## The plan

1. **The graph (this page).** Extraction, grounding, the point-in-time view, the leak test, and the first reading above. H9 needs history: each company's 10-Ks from 2010 on (about 16 per company). That is about 900 more 10-Ks for `us_large` and about 11,000 for `us_broad`. Report coverage (`graph stats --at` each year) before anything is tested.
2. **H9: propagation.** A signal in `brain/research/signals.py`: a company's score is its customers' last-month excess returns (and, as a separate variant, earnings surprises), weighted by revenue share where the 10-K gives one. It is declared in `hypotheses.py`, committed, then run through the same gate as H1–H8b. It counts in the false-discovery correction.
3. **Measure the model's memory bias.** Rerun extraction on a sample with the counterparty names masked to placeholders, then restored only after the model picks. Compare the two edge sets. A large gap means the chosen edges lean on memory, and H9's result must be read with that in mind.
4. **H9b: a graph neural network**, only if H9 shows something. It gets a new id and is declared before running.
