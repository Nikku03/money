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

Each pair is tested on SPY's calendar:

1. **Same month** (descriptive, REPORT). Does the basket's excess return move with the cause in the same month (month-end close to month-end close), and by how much? The answer is the slope times a typical move: percentage points of excess return per +10% oil, per +0.25 point on the 10-year yield, per +2% on the dollar. It is not the average excess return.
2. **Next month** (gated). Does this month's change in the cause (to its month-end close) predict the basket's excess return over the next month, from the close of that month's first session to its last? Starting a session later is the research harness's rule (signal at the month-end close, fill at the next session's close). It keeps a cause that closes after the stock market (oil, the dollar) from leaking into the effect. This is the question that matters for acting on a link. A link the market prices at once shows up in the same month and not in the next.

Stocks and SPY must have a price on the exact session; a cause may be up to 7 days old at the month's last session.

## The rules

These are the research ledger's rules (`CLAUDE.md`).

- **Declared before running.** `python -m brain chain declare` writes each test's full spec to `research/ledger.jsonl`. The spec includes:
  - the path of links, the measures and the baskets;
  - the windows and the statistics;
  - the false-discovery rule;
  - the code's sha;
  - `data_asof`: the time the series are read as of.

  Commit the ledger before running: `test` refuses a declaration that isn't in HEAD's ledger.
- **Frozen.** A declared spec never changes. Ids carry a version (`L1-oil-airlines-next`). A change to the design or the code after declare is a new `VERSION` in `brain/chain/linktest.py`, so new ids. `declare` refuses an existing id whose design has changed.
- **Pinned data.** `test` reads every series as the bank held it at `data_asof`. The result names each snapshot it read (`inputs`), so it can be reproduced.
- **Windows.** TRAIN is 2011–2018. TEST runs from 2019 to the last complete month at declare. The verdict reads TEST only; TRAIN is shown beside it.
- **The statistic.**
  - The slope of the basket's excess return on the cause's change.
  - Its p-value is one-sided, in the direction the links imply, from 5,000 shuffles of the cause's months.
  - Its 90% interval comes from resampling months.
- **Controls, on each test's TEST months.**
  - The real cause, shifted round by 3 months or more, must reach p < 0.05 in at most 10% of the shifts. A shift keeps the cause's own autocorrelation, fat tails and jumps but breaks any link. Too many hits means the shuffle null is too narrow for these series.
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
  - a window has fewer than 80% of its months measured, or misses its first or last month. Run `python -m brain chain measure` to collect the data first;
  - the declaration isn't committed, or its spec no longer matches its sha.
- **Trial runs.** `--trial` (or any home or ledger other than the committed ones) writes to `<home>/trial/` and reads TRAIN years only (2011–2014 against 2015–2018). A trial refuses a declaration that isn't a trial one, so it never computes TEST months, even from a copy of the real ledger.

## Reading the results

- **Survivorship.** The baskets are chosen today, so companies that failed or merged are missing.
- **Oil roll effects.** Oil is the front-month future, so its monthly change includes roll effects.
- **Same-month moves can't be traded on.** A strong same-month move with no next-month effect means the market prices the link as it happens. It tells you the link is real, not that it can be traded.
- **A FAIL is not the opposite.** It says the bank's history does not show the effect at this power. Eight years of TEST months is about 90 data points.
- **Results carry no weight in D1.** `brain chain follow` shows them beside each chain, but no link test feeds a decision. Policy D1 is frozen (`docs/DECIDE.md`).

## Links from a model

`python -m brain chain distill` asks whether the chain engine's knowledge can come from a language model alone.

- **What the model gets.** Only the names of the 19 states in `knowledge/links.json`: no links, no data.
- **What it's asked.** For each state, what it directly causes within about a month, and in which direction. The settings are temperature 0, seed 7, and `/no_think` for Qwen3.
- **Where its answers go.** They become its own link table, `<home>/knowledge/links.<model>.json`.

The report then answers three questions:

1. **Recall.** How many of the textbook links did the model write on its own, with the right sign, and what did it add?
2. **Reasoning.** Can the chain engine follow chains through the model's table? Each chain is checked against an exact lookup.
3. **Reality.** For each cause → basket pair the model's table implies and the bank can measure, its direction is set beside the recorded link tests above. Nothing new is measured: a pair the tests haven't covered stays untested until it is declared.

The model's table is its opinion, like a claim from a web page. It weighs nothing anywhere. On Colab, run it with the 32B model to compare models.
