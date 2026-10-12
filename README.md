# money: a memory bank and an evidence engine for public market data

An agent with web access reads the internet. This repo is what it remembers, and the test every trading idea has to pass before it is believed.

```
 web pages, news ──(agent: `note`)──┐
 SEC EDGAR (filings, XBRL) ─────────┤
 Yahoo prices, FRED macro ──────────┼──► memory bank ──► brief on a target (every line cited)
                                    │    (sourced,   ──► research engine: declared hypotheses,
                                    │     timestamped)    matched nulls, TEST-only verdicts
                                    └─────────────────► claim scoring: is this source ever right?
```

Design carried over from the cell project (`Nikku03/cell`, `HANDOFF_REASONING_ENGINE_AND_MEMORY_BANK.md`):

| Lesson there | What it became here |
| --- | --- |
| Reasoning machinery was not the bottleneck; knowledge was | Most of the code is about getting facts right: sources, timestamps, restatements, share bases |
| Facts individually addressable, looked up by content | Every fact is content-addressed (id = hash), stored once, retrieved by entity and attribute |
| Write each intermediate result down (the "snap") | Derived numbers are written back as facts that cite their inputs |
| Journal must replay exactly | Append-only JSONL per entity; `check` verifies ids and replay |
| Literature signs ~50% right in a given cell line | Web claims are stored as claims, then scored per source against what the stock did |
| Matched controls, positive control first, predeclared gates | Shuffled-signal null, oracle control, random-signal calibration, gates in code before running |
| A generic learner beat hand-written rules | H5 is a ridge learner over the hand signals' ranks, judged by the same gate |
| Negatives recorded, overstatements withdrawn not deleted | `research/ledger.jsonl` is append-only and committed |

One rule did **not** carry over. The cell project split TRAIN and TEST by hashing the key. Market data is a time series and neighbouring days are correlated, so a hash split would leak. Here the split is by date.

## The one idea that matters most: point-in-time

Each fact has two dates. `as_of` is what it describes (the quarter that ended 2024-06-30). `known_at` is when the public could first know it (the 10-Q filed 2024-07-31). `bank.view(t)` shows only facts with `known_at <= t`, and every signal and metric reads through a view. A backtest on 2015-03-31 therefore sees 2015's filings, including numbers that were later restated, and never sees 2016's. Most backtests that look too good fail on exactly this.

## What the evidence says so far

First run, 2026-10-12 (`research/RESULTS.md`): **none of the five declared ideas beat a shuffled signal on the 2019–2026 TEST months after costs.**

| Idea | TEST net excess / month | Perm p | Verdict |
| --- | --- | --- | --- |
| 12-1 month momentum | +0.11% | 0.18 | FAIL |
| 1-month reversal | −0.01% | 0.21 | FAIL |
| Earnings yield (value) | −0.23% | 0.87 | FAIL |
| Low asset growth | −0.11% | 0.74 | FAIL |
| Ridge learner on all four | +0.19% | 0.051 | FAIL (BH q 0.26) |

The controls passed: the harness finds a planted edge (oracle p = 0.0005), and random signals pass 3% of the time, near the nominal 5%. So these FAILs are real negatives for this universe, not a broken instrument. As in the cell project, the generic learner came closest, but closest is not a pass.

What this means for making money: textbook signals applied to 60 mega-caps that everyone watches do not carry an edge you can detect after costs. Survivorship alone is worth more than any of them: the equal-weight universe returned 27% a year against SPY's 18%, purely because it was chosen with hindsight. That hindsight is exactly what a real trader doesn't have. Any edge is more likely to come from information the market prices slowly: better data (measured, specific facts, as in the cell project), less-watched stocks, or events. That needs new hypotheses, and each one goes through the same gate.

## Quick start

```
export SEC_USER_AGENT="Your Name your@email.com"
python -m unittest discover -s tests -t .
python -m brain ingest AAPL                 # SEC facts, filings, prices, headlines
python -m brain macro                       # FRED rates, spreads, CPI (needs FRED_API_KEY)
python -m brain brief AAPL                  # cited brief -> var/briefs/AAPL.md
python -m brain ask AAPL us-gaap:NetIncomeLoss --at 2015-06-30   # what was known then
python -m brain note AAPL "Analyst X cuts to sell on China demand" \
    --source https://... --known-at 2026-10-09T13:00:00Z --direction down
python -m brain score-claims                # hit rate per source, vs a coin flip
python -m brain check                       # invariants; exit 0 = OK
```

Research:

```
python -m brain universe --n 60             # rebuild the universe by its written rule (already frozen in git)
python -m brain ingest-universe             # refresh its data
python -m brain research declare            # declare hypotheses in brain/research/hypotheses.py, then commit
python -m brain research run                # run, append results to the ledger, write research/RESULTS.md
```

Requires Python 3.11+, numpy and pandas. Local state lives in `var/` (gitignored; set `MONEY_HOME` to move it).

## Layout

| Path | What |
| --- | --- |
| `brain/bank.py` | Fact, Bank (journal, replay, downgrade), View (point-in-time reads) |
| `brain/check.py` | Invariants: id integrity, replay, nothing known before it happened, derived facts known after their inputs, series files intact |
| `brain/sources/` | SEC EDGAR (XBRL facts, filings, identity), Yahoo daily prices and splits, FRED, Yahoo RSS headlines |
| `brain/metrics.py` | Point-in-time metrics (market value, earnings yield, asset growth, margins, FCF), shared by briefs and backtests |
| `brain/claims.py` | Store web claims; score them per source |
| `brain/brief.py` | Cited one-page brief; writes derived facts back |
| `brain/research/` | Monthly panel, signals, harness (portfolio, costs, permutation null), ledger, hypotheses, runner |
| `research/ledger.jsonl` | Every declaration and result, append-only (committed) |
| `research/RESULTS.md` | Current verdicts, generated from the ledger (committed) |
| `universe/us_large.json` | The research universe, its rule, and what the rule rejected (committed) |

## Data and licences

| Source | Used for | Terms |
| --- | --- | --- |
| SEC EDGAR | Financial statements, filings, identity | US government, public domain. Requires a User-Agent with contact details; at most 10 requests/s |
| Yahoo Finance chart API | Daily prices, splits | Unofficial, personal research only. Snapshots stay in `var/` and are never committed |
| Yahoo Finance RSS | Headlines | Links and titles only |
| FRED (official API, free key in `FRED_API_KEY`) | Macro context in briefs | Mostly public domain; revised series are flagged and kept out of backtests |

## Known limits

- Earnings yield covers 85% of TEST stock-months. Some companies tag net income or diluted shares with concepts outside the curated list (see the ledger note). The fix must be declared as a new hypothesis.

- **Survivorship.** The universe is chosen by today's market value, so every member survived and grew. Results report the size of the effect and treat any PASS as an upper bound. Fixing this needs delisted-company price data, which free sources don't provide.
- Annual fundamentals only for signals (latest 10-K). A trailing-twelve-month version needs quarter arithmetic (Q4 = FY − 9-month YTD).
- Form 4 insider filings are listed but not parsed. Headlines are stored but not scored.
- US only. The price source handles other exchanges (e.g. `RELIANCE.NS`), but filings come from SEC.
- No broker connection, by design.

Not financial advice.
