# Testing the chain engine's links

The chain engine (`brain/chain`) follows cause → effect links from `knowledge/links.json`: "oil up → airline fuel costs up → airline profits down → airline stocks down". The links are textbook knowledge. This page is how each one is measured on the bank's history before a chain counts for anything.

## What can be measured

A link is measured where the bank can see both ends of a chain:

- **The cause** is a daily series:
  - WTI crude (`CL=F`, the front-month future), monthly % change;
  - the 10-year Treasury par yield (US Treasury), monthly change in percentage points;
  - the US dollar index (`DX-Y.NYB`), monthly % change.
- **The effect** is a fixed basket of five stocks. It is measured as the basket's equal-weight total return minus SPY's.

A chain whose middle steps the bank can't see (fuel costs, profits, mortgage rates) is tested end to end. For example, oil → airline stocks covers the three links between them.

Today that gives five pairs:

| Pair | Basket | The links say |
|---|---|---|
| oil → airline stocks | DAL, UAL, LUV, AAL, ALK | down |
| oil → oil producer stocks | XOM, CVX, COP, EOG, OXY | up |
| 10-year yield → bank stocks | JPM, BAC, WFC, C, PNC | up |
| 10-year yield → tech stocks | AAPL, MSFT, NVDA, ORCL, ADBE | down |
| 10-year yield → homebuilder stocks | LEN, DHI, PHM, NVR, TOL | down |

The other links stay untested until the bank holds what would measure them. `python -m brain chain links` lists each link's test, or what it needs:

- **Inflation:** CPI with its release dates (FRED, with `FRED_API_KEY`).
- **Mortgage rates:** a mortgage-rate series.
- **The Strait disruption:** a dated list of shipping disruptions.
- **Exporters:** each company's share of foreign sales.

## Two tests per pair

Each pair is tested on month-end closes, on SPY's calendar:

1. **Same month** (descriptive, REPORT). Does the basket move with the cause in the same month, and by how much? The answer is given per typical move: per +10% oil, per +0.25 point on the 10-year yield, per +2% on the dollar.
2. **Next month** (gated). Does this month's change in the cause predict next month's basket return? This is the question that matters for acting on a link. A link the market prices at once shows up in the same month and not in the next.

## The rules

These are the research ledger's rules (`CLAUDE.md`).

- **Declared before running.** `python -m brain chain declare` writes each test's full spec to `research/ledger.jsonl`. The spec includes the path of links, the measures, the baskets, the windows, the statistics and the code's sha. Commit before running.
- **Frozen.** A declared spec never changes. A change is a new id.
- **Windows.** TRAIN is 2011–2018. TEST runs from 2019 to the last complete month at declare. The verdict reads TEST only; TRAIN is shown beside it.
- **The statistic.**
  - The slope of the basket's excess return on the cause's change.
  - Its p-value is one-sided, in the direction the links imply, from 5,000 shuffles of the cause's months.
  - Its 90% interval comes from resampling months.
- **Controls, on each test's TEST months.**
  - 200 random causes must reach p < 0.05 at most 10% of the time.
  - A cause equal to the effect must reach p < 0.01.
  - If either fails, the verdict is void.
- **The gate for a next-month test.** PASS when all four hold:
  - the slope points the way the links say;
  - p < 0.05;
  - the controls pass;
  - Benjamini–Hochberg q < 0.10 across every gated test in the ledger (the hypotheses, the gated forecast tests and these).
  - Same-month tests are descriptive and stay outside the correction.
- **Recorded whatever they say.** `python -m brain chain test` writes the results to the ledger and the Links section of `research/RESULTS.md`. Commit both. A mistake is withdrawn (`python -m brain research withdraw ID --reason "..."`), never deleted.
- **Refusals.** `test` refuses in any of these cases:
  - the working tree has changes under `brain/` or `knowledge/`;
  - the code differs from the declared code;
  - a test already has a result;
  - a window has fewer than 80% of its months measured. Run `python -m brain chain measure` to collect the data first.
- **Trial runs.** `--trial` (or any home other than `var/`) writes to `<home>/trial/` and reads TRAIN years only (2011–2014 against 2015–2018). Trials never touch the committed record or the TEST months.

## Reading the results

- **Survivorship.** The baskets are chosen today, so companies that failed or merged are missing.
- **Oil roll effects.** Oil is the front-month future, so its monthly change includes roll effects.
- **Same-month moves can't be traded on.** A strong same-month move with no next-month effect means the market prices the link as it happens. It tells you the link is real, not that it can be traded.
- **A FAIL is not the opposite.** It says the bank's history does not show the effect at this power. Eight years of TEST months is about 90 data points.
- **Results carry no weight in D1.** `brain chain follow` shows them beside each chain, but no link test feeds a decision. Policy D1 is frozen (`docs/DECIDE.md`).
