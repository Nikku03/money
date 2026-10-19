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
| `value` | the counterparty's ticker, or `?` plus its normalised name when SEC's ticker list has no match (`?intel` would mean "Intel Corporation" did not resolve) |
| `as_of`, `known_at` | the 10-K's own: the period it covers and when it became public |
| `inputs` | the 10-K's `filing` fact |
| `context` | `accn`, `form`, `name` as written, `quote`, `share_pct`, `extractor` (the model, or `agent`), `prompt_sha` |

For each filing read, one `relation.scan` fact: the number of relations kept, the number of passages read, the drop counts by reason, the sha of the extracted text, the extractor and the prompt sha. A 10-K counts as read by an extractor only if a scan with the same prompt sha exists, so changing the prompt rereads everything under a new sha. Nothing is overwritten.

## The graph at a time t

`graph.graph_at(bank, tickers, t)` reads only through `bank.view(t)`:

- For each company, it uses the latest 10-K read and public by t, and only if that 10-K covers a period ending at most 550 days before t. A company that stopped filing drops out.
- It returns that 10-K's relations, plus any an agent added for the same 10-K.
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

Names resolve to tickers by exact match after normalising (lower case, `&` as "and", punctuation, legal suffixes and state tags like `/DE/` removed) against **today's** SEC ticker list. Consequences:

- **Delisted, acquired and private counterparties stay unresolved** (`?name`). They remain in the graph as nodes; they just can't be priced. So the coverage of priceable links is lower for older years. This affects coverage only; no returns leak through it.
- **Abbreviations don't resolve.** "TSMC" stays `?tsmc` unless the filing writes the full name. A curated alias table would be a declared change.
- **A name reused by a different company later** would resolve to the later one. This is rare, and it's checked by eye in `graph show`.

## The plan

1. **The graph (this page).** Extraction, grounding, the point-in-time view, the leak test. Then read the broad universe's 10-Ks from 2010 on, and report coverage (`graph stats` by year) before anything is tested.
2. **H9: propagation.** A signal in `brain/research/signals.py`: a company's score is its customers' last-month excess returns (and, as a separate variant, earnings surprises), weighted by revenue share where the 10-K gives one. It is declared in `hypotheses.py`, committed, then run through the same gate as H1–H8b. It counts in the false-discovery correction.
3. **Measure the model's memory bias.** Rerun extraction on a sample with the counterparty names masked to placeholders, then restored only after the model picks. Compare the two edge sets. A large gap means the chosen edges lean on memory, and H9's result must be read with that in mind.
4. **H9b: a graph neural network**, only if H9 shows something. It gets a new id and is declared before running.
