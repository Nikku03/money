# Quiet changes in 10-Ks and 10-Qs

`brain/changes.py` compares every 10-K and 10-Q with the same report a year earlier and records how much it changed. The idea comes from Cohen, Malloy and Nguyen, "Lazy Prices" (Journal of Finance, 2020). Companies rewrite their reports a little each year. A new risk, a customer no longer named, softer wording: these changes are public, but almost nobody compares the two documents, so the market can be slow to notice them. The paper found that firms whose reports changed most did worse afterwards.

Whether that holds here is H10's question (`research/RESULTS.md`).

## What is measured

Each 10-K is paired with the 10-K whose period ended 300 to 430 days earlier (the one nearest a year). Each 10-Q is paired with the 10-Q of the same fiscal quarter a year earlier. The earlier report must have been public before the newer one. Amendments (10-K/A, 10-Q/A) are skipped.

| Measure | What it is |
|---|---|
| `cosine` (the fact's value) | cosine similarity of the two documents' word counts. Letters only, so numbers that change every year don't count. Common words ("the", "of", "and"; `changes.STOP`) are left out, because they swamp the rest. 1 means the same words in the same proportions. |
| `jaccard` | shared distinct words over all distinct words |
| `new_share` | the share of the new report's sentences that were not in the old one. Numbers and month names are ignored, so "the three months ended July 27, 2025" matches the year before. |

The comparison is arithmetic on the words, with no language model involved. So it can be tested on history without a model's memory leaking into the result.

Each comparison is a `derived` fact, `text.similarity`, on the filer:
- `as_of` and `known_at` are the newer report's.
- `inputs` cites both filings.
- `context` holds the form, both accession numbers, the other measures and `METHOD`.

A view at t sees only comparisons whose newer report was public by t. A change to the text, sentence or similarity rules is a new `METHOD`, and readers use only the current one.

Documents are fetched once from SEC and kept as their sentences under `<home>/text` (gzip, about 60 KB a report). The raw document isn't kept.

## Commands

```
python -m brain ingest NVDA                       # the filing list the comparisons pair
python -m brain changes build NVDA [--since 2008-01-01]
python -m brain changes build --universe large    # every us_large ticker
python -m brain changes show NVDA                 # its history, then the newest report's new sentences
python -m brain changes recent [--days 14]        # reports filed lately, most changed first
```

`show` lists the sentences of the newest report that weren't in the one a year earlier. Sentences naming a risk or a dependency (going concern, material weakness, investigation, impairment, default, customer, supplier, export controls and others: `changes.RISK`) come first. This is the part a person reads: what changed, in the company's own words.

## H10

H10 tests the change score on the 60-stock `us_large` universe. Each month, every stock's latest report (filed in the last 400 days) is scored against the company's own earlier comparisons of the same form: its similarity minus their mean, in their standard deviations (`changes.unusual`, with at least 4 earlier ones). H10 holds the third that changed least compared with their usual, the quiet ones. It compares them with all 60, after costs, with the same gate as every other hypothesis.

Why against the company's own history and not the raw similarity:
- **Forms differ.** On the large caps, 10-Ks run more similar year to year (median about 0.99) than 10-Qs (about 0.97). At a month-end, a raw score would mostly sort stocks by which form they filed last.
- **Companies differ.** Some rewrite more than others, every year.

This was found by looking at the comparisons by year and form, before H10 was declared and before any return was computed. The week-ahead look (`docs/AHEAD.md`) flags a report on the same score, at -1.5 or below.

It is declared in `brain/research/hypotheses.py`, committed before it runs, and counts in the false-discovery correction.

**Result (2026-10-19): FAIL.** On the 92 TEST months (2019 to 2026), the quiet third earned 0.16% a month less than all 60 after costs (permutation p 0.75; TRAIN 2011 to 2018: 0.27% a month less, p 0.91). The controls passed. On these large caps, a report that changed less than usual did not mark a stock that went on to beat the others. The score still shows what changed; it carries no weight as a signal.

Reading it:
- **Few stocks.** 60 stocks give little power: an edge of a few tenths of a percent a month can hide in the noise.
- **Survivorship.** The universe is today's survivors, so a firm whose reports changed because it was failing is missing. That works against finding the effect.
- **Not the paper's exact measure.** The paper sorts on the raw similarity. H10 sorts on similarity against the company's own usual, for the reasons above.
- **Short histories.** A company with a new SEC registration has fewer comparisons and no score until it has 4 of a form: BLK (new holding company, 2024), LIN (2018), DIS (2019), MRVL (2021), AVGO (2018), GOOGL (2015). It is not ranked in those months.
- **Wrapper filings.** Some banks (WFC, JPM, C) put most of their annual report in an exhibit the primary document doesn't contain. Their similarity is that of the wrapper.
