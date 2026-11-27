# Research results

Generated 2026-11-27T06:29:37Z from `research/ledger.jsonl`. Every number below is copied from the ledger.

TRAIN 2011-01-31..2018-12-31, TEST from 2019-01-01. Monthly, equal weight against the equal-weight universe, 10 bps per unit turnover. Holdings: the top third by signal unless the hypothesis names another selection (Select column; see `brain/research/harness.py`).

Universes (each has its own panel and controls):

- `universe/us_large.json`: 60 US stocks (top SEC filers by market value with 10-K XBRL and prices since 2010). Hypotheses: H1-mom12_1, H2-rev1, H3-earnings_yield, H4-low_asset_growth, H5-ridge.
- `universe/us_broad.json`: 741 US stocks (SEC ranks 1-1500 with 10-K XBRL and prices since 2010). Hypotheses: H6-insider_buying, H3c-earnings_yield_ttm, H7b-earnings_reaction, H8b-sue, H3b-earnings_yield_ttm, H7-earnings_reaction, H8-sue.

Gate (declared before running): TEST mean net excess > 0, permutation p < 0.05, and Benjamini–Hochberg q < 0.10 across every hypothesis in the ledger and every gated forecast test (see Forecasts) and every gated link test (see Links) and every gated event test (see Events) and every gated move-size test (see Move sizes) and every gated shock test (see Shocks) and every gated weekly-range test (see Weekly ranges) and every gated mosaic test (see Mosaic) and every gated step-2 test (see Step 2) and every gated ripple test (see Ripples) and every gated earnings test (see Earnings: expected and what followed) and every gated promise test (see Promises: said and delivered) and every gated customer test (see Customers: who it sells to).

| Hypothesis | Universe | Select | TEST net excess / mo | Annualised | Info ratio | Months won | Perm p | BH q | TRAIN net excess / mo | TRAIN p | TEST coverage | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1-mom12_1 | us_large | top_frac | +0.11% | +1.3% | 0.12 | 42/92 | 0.1758 | 0.399 | +0.22% | 0.0148 | 100% | **FAIL** |
| H2-rev1 | us_large | top_frac | -0.01% | -0.1% | -0.01 | 49/92 | 0.2142 | 0.423 | +0.05% | 0.0722 | 100% | **FAIL** |
| H3-earnings_yield | us_large | top_frac | -0.23% | -2.7% | -0.31 | 46/92 | 0.8710 | 0.917 | -0.10% | 0.8714 | 85% | **FAIL** |
| H4-low_asset_growth | us_large | top_frac | -0.11% | -1.3% | -0.22 | 48/92 | 0.7363 | 0.912 | -0.09% | 0.7251 | 97% | **FAIL** |
| H5-ridge | us_large | top_frac | +0.19% | +2.3% | 0.27 | 48/92 | 0.0510 | 0.182 | +0.14% | 0.0546 | 100% | **FAIL** |
| H6-insider_buying | us_broad | positive | +0.00% | +0.0% | 0.01 | 48/92 | 0.2789 | 0.473 | +0.11% | 0.0080 | 100% | **FAIL** |
| H3c-earnings_yield_ttm | us_broad | top_frac | -0.11% | -1.3% | -0.25 | 48/92 | 0.8846 | 0.917 | -0.11% | 0.9282 | 94% | **FAIL** |
| H7b-earnings_reaction | us_broad | top_frac_vs_all | -0.24% | -2.8% | -0.55 | 36/92 | 0.5845 | 0.779 | -0.07% | 0.1838 | 36% | **FAIL** |
| H8b-sue | us_broad | top_frac | -0.05% | -0.6% | -0.20 | 48/92 | 0.4867 | 0.728 | -0.03% | 0.1058 | 92% | **FAIL** |
| H3b-earnings_yield_ttm | us_broad | top_frac | -0.11% | -1.3% | -0.25 | 48/92 | 0.8784 | – | -0.11% | 0.9342 | 94% | **WITHDRAWN** |
| H7-earnings_reaction | us_broad | top_frac_vs_all | -0.24% | -2.9% | -0.56 | 34/92 | 0.6027 | – | -0.07% | 0.1704 | 35% | **WITHDRAWN** |
| H8-sue | us_broad | top_frac | -0.06% | -0.7% | -0.23 | 48/92 | 0.5071 | – | -0.04% | 0.1378 | 92% | **WITHDRAWN** |

TEST coverage is the share of listed stock-months with a signal value; the rest are not ranked.

## Withdrawn

Run, then withdrawn for a mistake. The result stays in the ledger and still counts in the Benjamini–Hochberg correction (so its own q is not shown).

- H3b-earnings_yield_ttm: Code mistake found in review: a derived Q4 (fiscal year minus nine-month YTD) paired the year as first filed in the 10-K with the latest-known nine months, which can be a recast filed later (a comparative in next year's 10-Q), mixing two bases. Example: CAT Q4 2015 ProfitLoss came out -500M. Across the broad universe at three views, 129 of 9,050 derived Q4 NetIncomeLoss values change and 83 without a nine-month figure known at the 10-K are dropped. Fixed in metrics._quarters_of (the nine months as known when the year was filed); re-declared as H3c-earnings_yield_ttm.
- H7-earnings_reaction: Code mistake found in review: events.distinct treated every 8-K item 2.02 filing within 20 days of the previous filing as the same release, so results that followed a pre-announcement or preliminary revenue release were dropped (NVDA 2019-02-15 after 2019-01-28; ISRG's January results each year; CAT 2024-10-30), and the signal took the earlier filing's reaction. In the broad universe it kept 62,455 of 66,445 filings; the fix (a re-file within a day or an 8-K/A within 20 days is the same release, compared with the last release kept) keeps 65,253. Re-declared as H7b-earnings_reaction.
- H8-sue: Code mistake found in review: SUE's quarterly EPS used the derived Q4 that paired the fiscal year as first filed with a nine-month YTD recast later (see H3b's withdrawal). Fixed in metrics._quarters_of; re-declared as H8b-sue.

## Controls

`universe/us_large.json` (latest run, 2026-10-12):

- Oracle (next month's actual return as the signal): TEST p = 0.0005, gross excess +8.04%/mo. The harness can see a real edge.
- 200 random signals, redrawn every month: 3.0% reached p < 0.05 (expected about 5%). This is what the null is checked against: signals that change from month to month.
- Random scores held fixed per stock were not tried in this run. The null redraws its holdings every month, so for a signal that persists from month to month (fundamentals, slow-moving events) it is too narrow and the p-value too small, by an amount not measured here.
- Controls passed.

`universe/us_broad.json` (latest run, 2026-10-14):

- Oracle (next month's actual return as the signal): TEST p = 0.0005, gross excess +9.03%/mo. The harness can see a real edge.
- 200 random signals, redrawn every month: 4.0% reached p < 0.05 (expected about 5%). This is what the null is checked against: signals that change from month to month.
- 200 random scores held fixed per stock in every month: 5.5% reached p < 0.05. The null redraws its holdings every month, so for a signal that persists from month to month (fundamentals, slow-moving events) it is too narrow and the p-value too small; above about 5%, this is by how much. Reported, not part of the declared gate.
- Controls passed.

## Survivorship bias, measured

- `universe/us_large.json`: over TEST the equal-weight universe returned +27.4% a year; SPY returned +17.9%.
- `universe/us_broad.json`: over TEST the equal-weight universe returned +20.4% a year; SPY returned +17.9%.

The gap is mostly selection: each universe is chosen by today's market value, so every member is a survivor and a past winner. Comparing against the same universe cancels the level, not the tilt: survivorship still flatters momentum and growth, and hurts value. A PASS here is an upper bound.

## Ledger notes

- 2026-10-12: The five declarations above were written before the repository had a commit, so their commit field reads HEAD. They were declared against the code committed as a293ac1, which contains this ledger with only those declarations. git_state now records an empty commit when HEAD does not exist.
- 2026-10-12: H3 coverage diagnosis (after the TEST result, so no rerun): earnings yield is missing for 15% of TEST stock-months. Causes: no us-gaap WeightedAverageNumberOfDilutedSharesOutstanding (V, BRK-B; GOOGL, DIS, BLK, LIN, MRVL in earlier years) and NetIncomeLoss no longer tagged in 10-Ks (CAT since FY2010, MA and TMO since FY2013, SCCO, AVGO FY2025). A version that reads more concepts must be declared as a new hypothesis (H3b) and counts in the BH correction.
- 2026-10-13: H6-insider_buying's recorded run (2026-10-13T18:29:28Z) read SEC insider bulk data sets 2010q4 through 2026q3 (var/insiders_done.json, written 17:51:54Z), so the 90-day purchase window was full from the first rebalance (2011-01-31). The bulk loader's default start is now 2010q4 to match. No result changes.
- 2026-10-17: From the link tests (docs/LINKS.md) on, the false-discovery correction counts every gated link test as well: every hypothesis, every gated forecast test and every gated link test with a result, withdrawn ones included. F1b and F1c were declared with the narrower family; their q is now taken over this one.
- 2026-10-19: From the event tests (docs/EVENTS.md) on, the false-discovery correction also counts every gated event test: every hypothesis and every gated forecast, link and event test with a result, withdrawn ones included.
- 2026-11-16: T1a's reported gap of the list to SPY by market value (-5.8% a year) was labelled 'what the survivor-only data adds'. Checked after the run: the list by market value leaves out companies whose weighted-average share count is not read from their filings (Berkshire and Visa throughout, Alphabet at the start of TEST) and Exxon, whose accounts were never read (universe/us_broad.json rejected it). The gap is mostly that, not survivorship; the label was corrected in research/run.py. No declaration or result changes.

## Reading this

- FAIL means the claim did not beat a shuffled signal on unseen months after costs. It does not mean the opposite is true.
- `universe/us_large.json`: 92 TEST months and 60 stocks.
- `universe/us_broad.json`: 92 TEST months and 741 stocks.
- Few TEST months give little power: an edge of a few tenths of a percent a month can hide in this noise.
- Not financial advice. Nothing here places orders.

## Forecasts

F1 forecasts each stock's chance to beat SPY over the next h sessions, and the likely range of its price (docs/FORECAST.md). Every metric is computed per forecast date across the stocks forecast that date, then over dates. Stocks move together within a week, so every interval and p-value is date-level.

Gate (F1b, F1c; declared before running): mean size-neutral IC > 0 with a one-sided date-level sign-flip p < 0.05, Benjamini–Hochberg q < 0.10 across every hypothesis and gated forecast test in the ledger, and both controls passing. F1a and F1d are descriptive retests (REPORT): their 90% intervals are shown, they get no verdict and stay out of the correction.

| Test | h | Dates | Samples | IC | Size-neutral IC | p or 90% interval | BH q | Size IC | Momentum IC | Spread | Brier skill (within-date) | Placement | 68% band held | 90% band held | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1a-w5-recent | 5 | 2026-08-18..2026-10-07 (8) | 5928 | +0.040 | +0.042 | 90% -0.099..+0.183 | – | -0.046 | +0.014 | +0.52% | +0.99% (+0.13%) | 50.9% | 69.0% (62.8%-75.2%) | 91.9% (88.1%-95.7%) | **REPORT** |
| F1b-w5-test | 5 | 2019-01-03..2026-08-11 (383) | 283803 | +0.010 | +0.011 | p 0.0870 | 0.287 | +0.009 | +0.008 | -0.06% | -0.08% (+0.00%) | 50.4% | 67.8% (66.5%-69.1%) | 89.9% (89.0%-90.7%) | **FAIL** |
| F1c-m21-test | 21 | 2019-01-07..2026-09-15 (93) | 68913 | +0.008 | +0.014 | p 0.2096 | 0.423 | +0.020 | -0.001 | +0.23% | -0.42% (-0.01%) | 50.2% | 69.3% (67.0%-71.5%) | 90.9% (89.4%-92.4%) | **FAIL** |
| F1d-llm-w5-recent | 5 | 2026-08-18..2026-10-07 (8) | 96 | -0.003 | -0.011 | 90% -0.169..+0.147; Brier F1 - model +0.0097 (p 0.352) | – | +0.017 | -0.145 | -0.17% | +0.19% (+0.20%) | 37.5% | 76.0% (67.3%-84.8%) | 92.7% (82.5%-100.0%) | **REPORT** |

IC: Spearman correlation of the chance to beat SPY with what each stock then did beyond SPY. Size-neutral IC: the same after the forecast's rank is cleared of its fit on market value, since the universe holds today's survivors and small-then predicts rose-later. Size IC and Momentum IC: the same samples ranked by size alone (smaller first) and by 12-1 momentum alone. Spread: top fifth minus bottom fifth by forecast, mean excess per date (F1d: halves of 12). Brier skill: against the training base rate as a constant; the within-date part is the only one that speaks to ranking. Placement: share placed on the right side of the date's median. Bands: share of stocks whose close at the exit fell inside, per date (interval).

### What the range and the chance showed

- F1a-w5-recent (2026-08-18 to 2026-10-07, 8 dates): the 68% band held 69.0% of the time per date (date-level 90% interval 62.8% to 75.2%) and the 90% band 91.9% (88.1% to 95.7%), against 68% and 90% aimed for; its pinball loss was 1% above a naive normal band's. Then the chance to beat SPY had mean size-neutral IC +0.042 (90% interval -0.099..+0.183); retest (descriptive): inconclusive; it can detect mean IC >= 0.15 at 80% power.
- F1b-w5-test (2019-01-03 to 2026-08-11, 383 dates): the 68% band held 67.8% of the time per date (date-level 90% interval 66.5% to 69.1%) and the 90% band 89.9% (89.0% to 90.7%), against 68% and 90% aimed for; its pinball loss was 2% below a naive normal band's. Then the chance to beat SPY ranked stocks beyond size with mean IC +0.011 (date-level p 0.0870, q 0.287): FAIL. Controls passed: the oracle reached p 0.00005 (smallest possible 0.00005), so the test can reject, and 3.0% of 200 coin-flipped placebos passed (1%-10% expected): a check of the sign-flip test itself, not of the forecaster, since coin-flipped signs pass about 5% of the time for any series, a look-ahead leak included (the features' leak tests are what check for one). Within-date permutation p 0.0010 (a diagnostic: too narrow, since stocks move together).
- F1c-m21-test (2019-01-07 to 2026-09-15, 93 dates): the 68% band held 69.3% of the time per date (date-level 90% interval 67.0% to 71.5%) and the 90% band 90.9% (89.4% to 92.4%), against 68% and 90% aimed for; its pinball loss was 1% below a naive normal band's. Then the chance to beat SPY ranked stocks beyond size with mean IC +0.014 (date-level p 0.2096, q 0.423): FAIL. Controls passed: the oracle reached p 0.00005 (smallest possible 0.00005), so the test can reject, and 5.0% of 200 coin-flipped placebos passed (1%-10% expected): a check of the sign-flip test itself, not of the forecaster, since coin-flipped signs pass about 5% of the time for any series, a look-ahead leak included (the features' leak tests are what check for one). Within-date permutation p 0.0010 (a diagnostic: too narrow, since stocks move together).
- F1d-llm-w5-recent (2026-08-18 to 2026-10-07, 8 dates): the 68% band held 76.0% of the time per date (date-level 90% interval 67.3% to 84.8%) and the 90% band 92.7% (82.5% to 100.0%), against 68% and 90% aimed for; its pinball loss was 2% above a naive normal band's. Then the local model's Brier score against F1's on the same stocks: F1 minus model +0.0097 per date (90% interval -0.0245..+0.0440, exact sign-flip p 0.352); 0 of 96 answers failed and count as F1's; retest (descriptive): LLM vs F1 inconclusive.

### Calibration

F1a-w5-recent:

| Forecast range | Mean forecast | Beat SPY | n |
| --- | --- | --- | --- |
| 0.454-0.474 | 0.470 | 0.243 | 593 |
| 0.474-0.479 | 0.477 | 0.352 | 593 |
| 0.479-0.483 | 0.481 | 0.405 | 593 |
| 0.483-0.485 | 0.484 | 0.411 | 593 |
| 0.485-0.488 | 0.487 | 0.403 | 593 |
| 0.488-0.490 | 0.489 | 0.445 | 593 |
| 0.490-0.492 | 0.491 | 0.487 | 593 |
| 0.492-0.495 | 0.494 | 0.514 | 593 |
| 0.495-0.499 | 0.497 | 0.483 | 592 |
| 0.499-0.527 | 0.503 | 0.486 | 592 |

F1b-w5-test:

| Forecast range | Mean forecast | Beat SPY | n |
| --- | --- | --- | --- |
| 0.403-0.476 | 0.467 | 0.479 | 28381 |
| 0.476-0.479 | 0.479 | 0.514 | 28381 |
| 0.479-0.482 | 0.481 | 0.509 | 28381 |
| 0.482-0.486 | 0.484 | 0.479 | 28380 |
| 0.486-0.489 | 0.487 | 0.476 | 28380 |
| 0.489-0.493 | 0.491 | 0.494 | 28380 |
| 0.493-0.497 | 0.495 | 0.494 | 28380 |
| 0.497-0.502 | 0.499 | 0.496 | 28380 |
| 0.502-0.508 | 0.505 | 0.505 | 28380 |
| 0.508-0.597 | 0.521 | 0.472 | 28380 |

F1c-m21-test:

| Forecast range | Mean forecast | Beat SPY | n |
| --- | --- | --- | --- |
| 0.387-0.460 | 0.444 | 0.491 | 6892 |
| 0.460-0.469 | 0.465 | 0.488 | 6892 |
| 0.469-0.476 | 0.473 | 0.515 | 6892 |
| 0.476-0.481 | 0.479 | 0.506 | 6891 |
| 0.481-0.484 | 0.482 | 0.489 | 6891 |
| 0.484-0.487 | 0.486 | 0.496 | 6891 |
| 0.487-0.490 | 0.488 | 0.447 | 6891 |
| 0.490-0.497 | 0.493 | 0.498 | 6891 |
| 0.497-0.502 | 0.500 | 0.478 | 6891 |
| 0.502-0.592 | 0.536 | 0.434 | 6891 |

F1d-llm-w5-recent:

| Forecast range | Mean forecast | Beat SPY | n |
| --- | --- | --- | --- |
| 0.450-0.500 | 0.487 | 0.500 | 20 |
| 0.500-0.500 | 0.500 | 0.526 | 19 |
| 0.500-0.500 | 0.500 | 0.474 | 19 |
| 0.500-0.500 | 0.500 | 0.316 | 19 |
| 0.500-0.500 | 0.500 | 0.737 | 19 |

Forecasts are paper only. Nothing here places orders, and none of it is financial advice.

## Links

Tests of the chain engine's cause → effect links (`knowledge/links.json`, `docs/LINKS.md`). Each one measures a chain end to end where the bank can see both ends: the cause's monthly change, and a fixed basket's equal-weight total return minus SPY's. **Same month** is descriptive (REPORT): how far the basket's excess return moves with the cause, month-end to month-end. **Next month** is gated, declared before running: the basket's excess return over the next month, from its first session's close to its last, against this month's change in the cause. It passes when the slope points the way the links say, one-sided permutation p < 0.05, the controls pass, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger. TEST from 2019; TRAIN 2011–2018 is shown beside it. The effect is the slope times a typical move: percentage points of excess return per move, not the average excess return.

| Test | Cause → effect | Timing | TEST months | Effect per move, points (TEST) | 90% interval | p | BH q | TRAIN effect | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1-oil-airlines-same | oil_price_up → airline_stocks_down | same month | 93 (2019-01..2026-09) | +0.44% per +10% oil | -0.95%..+2.36% | 0.7417 | – | -3.13% | **REPORT** |
| L1-oil-airlines-next | oil_price_up → airline_stocks_down | next month | 93 (2019-01..2026-09) | -0.49% per +10% oil | -2.28%..+0.45% | 0.2190 | 0.423 | -1.76% | **FAIL** |
| L1-oil-producers-same | oil_price_up → oil_producer_stocks_up | same month | 93 (2019-01..2026-09) | +2.96% per +10% oil | +1.19%..+5.81% | 0.0002 | – | +2.90% | **REPORT** |
| L1-oil-producers-next | oil_price_up → oil_producer_stocks_up | next month | 93 (2019-01..2026-09) | -0.41% per +10% oil | -1.83%..+0.93% | 0.7489 | 0.912 | +0.20% | **FAIL** |
| L1-rates-homebuilders-same | interest_rates_up → homebuilder_stocks_down | same month | 93 (2019-01..2026-09) | -2.29% per +0.25 pt 10-year | -3.70%..-0.95% | 0.0018 | – | -0.50% | **REPORT** |
| L1-rates-homebuilders-next | interest_rates_up → homebuilder_stocks_down | next month | 93 (2019-01..2026-09) | +0.48% per +0.25 pt 10-year | -0.70%..+1.80% | 0.7395 | 0.912 | -0.26% | **FAIL** |
| L1-rates-banks-same | interest_rates_up → bank_stocks_up | same month | 93 (2019-01..2026-09) | +1.05% per +0.25 pt 10-year | +0.03%..+2.10% | 0.0266 | – | +3.44% | **REPORT** |
| L1-rates-banks-next | interest_rates_up → bank_stocks_up | next month | 93 (2019-01..2026-09) | +0.44% per +0.25 pt 10-year | -0.52%..+1.35% | 0.2088 | 0.423 | +0.15% | **FAIL** |
| L1-rates-tech-same | interest_rates_up → tech_stocks_down | same month | 93 (2019-01..2026-09) | -1.08% per +0.25 pt 10-year | -1.77%..-0.27% | 0.0180 | – | +0.42% | **REPORT** |
| L1-rates-tech-next | interest_rates_up → tech_stocks_down | next month | 93 (2019-01..2026-09) | -0.33% per +0.25 pt 10-year | -1.07%..+0.42% | 0.2587 | 0.473 | -0.24% | **FAIL** |

- **L1-oil-airlines-same**: for each month with oil up 10%, the airline stocks' return over SPY that month was 0.44 points higher (90% interval -0.95% to +2.36%; TRAIN -3.13%). The links say lower. REPORT.
- **L1-oil-airlines-next**: for each month with oil up 10%, the airline stocks' return over SPY the next month was 0.49 points lower (90% interval -2.28% to +0.45%; TRAIN -1.76%). The links say lower. FAIL.
- **L1-oil-producers-same**: for each month with oil up 10%, the oil producer stocks' return over SPY that month was 2.96 points higher (90% interval +1.19% to +5.81%; TRAIN +2.90%). The links say higher. REPORT.
- **L1-oil-producers-next**: for each month with oil up 10%, the oil producer stocks' return over SPY the next month was 0.41 points lower (90% interval -1.83% to +0.93%; TRAIN +0.20%). The links say higher. FAIL.
- **L1-rates-homebuilders-same**: for each month with the 10-year yield up 0.25 point, the homebuilder stocks' return over SPY that month was 2.29 points lower (90% interval -3.70% to -0.95%; TRAIN -0.50%). The links say lower. REPORT.
- **L1-rates-homebuilders-next**: for each month with the 10-year yield up 0.25 point, the homebuilder stocks' return over SPY the next month was 0.48 points higher (90% interval -0.70% to +1.80%; TRAIN -0.26%). The links say lower. FAIL.
- **L1-rates-banks-same**: for each month with the 10-year yield up 0.25 point, the bank stocks' return over SPY that month was 1.05 points higher (90% interval +0.03% to +2.10%; TRAIN +3.44%). The links say higher. REPORT.
- **L1-rates-banks-next**: for each month with the 10-year yield up 0.25 point, the bank stocks' return over SPY the next month was 0.44 points higher (90% interval -0.52% to +1.35%; TRAIN +0.15%). The links say higher. FAIL.
- **L1-rates-tech-same**: for each month with the 10-year yield up 0.25 point, the tech stocks' return over SPY that month was 1.08 points lower (90% interval -1.77% to -0.27%; TRAIN +0.42%). The links say lower. REPORT.
- **L1-rates-tech-next**: for each month with the 10-year yield up 0.25 point, the tech stocks' return over SPY the next month was 0.33 points lower (90% interval -1.07% to +0.42%; TRAIN -0.24%). The links say lower. FAIL.

Controls on each test's TEST months: the real cause, shifted round by 3 months or more (which keeps its autocorrelation and breaks any link), reached p < 0.05 in 0.0%–8.0% of 88 shifts (at most 10% allowed); a cause equal to the effect reached p ≤ 0.0020. All passed.

Reading this: the baskets are chosen today, so companies that failed or merged are missing (survivorship). Oil is the front-month future, so its monthly change includes roll effects. A same-month move says the link is priced as it happens; only a next-month PASS would say it can be acted on. Links the bank cannot measure stay untested: `python -m brain chain links` lists what each would need.


## Events

Tests of corporate events named in earnings press releases (SEC 8-K exhibit 99; `docs/EVENTS.md`). Fixed rules tag each release's opening text (`brain/web/events.py`). Each month, the tagged releases' mean excess return over SPY is compared with the month's other earnings releases; the statistic is the mean of those monthly differences, with a sign-flip p, one-sided in the direction the event implies (version 1) or two-sided (version 2). **Reaction** (days -1..+1) is descriptive: does the event move the stock? **Drift** (days +2..+21) is gated, declared before running: it passes when the difference points the way the event implies (either way for a two-sided test), p < 0.05, the controls pass, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger. TEST from 2019; TRAIN 2011–2018 is shown beside it.

| Test | Event | Days | TEST events (months) | vs other releases | 90% interval | p | BH q | vs SPY | TRAIN vs others | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1-guidance_raised-reaction | guidance_raised | -1..+1 | 174 (59) | +1.41% | -0.08%..+2.92% | 0.0579 | – | +2.28% | +1.28% | **REPORT** |
| E1-guidance_raised-drift | guidance_raised | +2..+21 | 174 (59) | -2.15% | -3.27%..-1.14% | 0.9997 | 1.000 | -1.10% | -0.51% | **FAIL** |
| E2-small-guidance_raised-reaction | guidance_raised | -1..+1 | 1071 (90) | +3.74% | +2.86%..+4.60% | 0.0000 | – | +3.97% | +3.35% | **REPORT** |
| E2-small-guidance_raised-drift | guidance_raised | +2..+21 | 1071 (90) | -0.35% | -1.15%..+0.41% | 0.4570 | 0.728 | -0.37% | -1.25% | **FAIL** |
| E2-small-guidance_cut-reaction | guidance_cut | -1..+1 | 91 (48) | -7.13% | -9.55%..-4.90% | 0.0000 | – | -6.68% | -5.33% | **REPORT** |
| E2-small-guidance_cut-drift | guidance_cut | +2..+21 | 91 (48) | +0.90% | -1.00%..+2.84% | 0.4749 | 0.728 | +0.71% | +1.02% | **FAIL** |
| E2-micro-guidance_raised-reaction | guidance_raised | -1..+1 | 319 (81) | +6.78% | +4.99%..+8.52% | 0.0000 | – | +6.72% | +4.29% | **REPORT** |
| E2-micro-guidance_raised-drift | guidance_raised | +2..+21 | 319 (81) | -1.25% | -2.77%..+0.20% | 0.1779 | 0.399 | -1.29% | -2.05% | **FAIL** |
| E2-micro-guidance_cut-reaction | guidance_cut | -1..+1 | 59 (38) | -9.20% | -12.32%..-6.07% | 0.0000 | – | -8.74% | -5.36% | **REPORT** |
| E2-micro-guidance_cut-drift | guidance_cut | +2..+21 | 59 (38) | -1.66% | -4.06%..+0.64% | 0.2717 | 0.473 | -1.47% | +0.14% | **FAIL** |

- **E1-guidance_raised-reaction**: over days -1..+1, releases tagged guidance_raised did +1.41% against the month's other releases (90% interval -0.08% to +2.92%) and +2.28% against SPY, over 174 releases in 59 months. The event implies up. REPORT.
- **E1-guidance_raised-drift**: over days +2..+21, releases tagged guidance_raised did -2.15% against the month's other releases (90% interval -3.27% to -1.14%) and -1.10% against SPY, over 174 releases in 59 months. The event implies up. FAIL.
- **E2-small-guidance_raised-reaction**: over days -1..+1, releases tagged guidance_raised did +3.74% against the month's other releases (90% interval +2.86% to +4.60%) and +3.97% against SPY, over 1071 releases in 90 months. Tested either way. A round trip in these stocks costs about 1%. 264 of the 32499 releases had no text read (no exhibit 99 found) and are left out of both groups. REPORT.
- **E2-small-guidance_raised-drift**: over days +2..+21, releases tagged guidance_raised did -0.35% against the month's other releases (90% interval -1.15% to +0.41%) and -0.37% against SPY, over 1071 releases in 90 months. Tested either way. A round trip in these stocks costs about 1%. 264 of the 32461 releases had no text read (no exhibit 99 found) and are left out of both groups. FAIL.
- **E2-small-guidance_cut-reaction**: over days -1..+1, releases tagged guidance_cut did -7.13% against the month's other releases (90% interval -9.55% to -4.90%) and -6.68% against SPY, over 91 releases in 48 months. Tested either way. A round trip in these stocks costs about 1%. 264 of the 32499 releases had no text read (no exhibit 99 found) and are left out of both groups. REPORT.
- **E2-small-guidance_cut-drift**: over days +2..+21, releases tagged guidance_cut did +0.90% against the month's other releases (90% interval -1.00% to +2.84%) and +0.71% against SPY, over 91 releases in 48 months. Tested either way. A round trip in these stocks costs about 1%. 264 of the 32461 releases had no text read (no exhibit 99 found) and are left out of both groups. FAIL.
- **E2-micro-guidance_raised-reaction**: over days -1..+1, releases tagged guidance_raised did +6.78% against the month's other releases (90% interval +4.99% to +8.52%) and +6.72% against SPY, over 319 releases in 81 months. Tested either way. A round trip in these stocks costs about 2%. 261 of the 18349 releases had no text read (no exhibit 99 found) and are left out of both groups. REPORT.
- **E2-micro-guidance_raised-drift**: over days +2..+21, releases tagged guidance_raised did -1.25% against the month's other releases (90% interval -2.77% to +0.20%) and -1.29% against SPY, over 319 releases in 81 months. Tested either way. A round trip in these stocks costs about 2%. 261 of the 18332 releases had no text read (no exhibit 99 found) and are left out of both groups. FAIL.
- **E2-micro-guidance_cut-reaction**: over days -1..+1, releases tagged guidance_cut did -9.20% against the month's other releases (90% interval -12.32% to -6.07%) and -8.74% against SPY, over 59 releases in 38 months. Tested either way. A round trip in these stocks costs about 2%. 261 of the 18349 releases had no text read (no exhibit 99 found) and are left out of both groups. REPORT.
- **E2-micro-guidance_cut-drift**: over days +2..+21, releases tagged guidance_cut did -1.66% against the month's other releases (90% interval -4.06% to +0.64%) and -1.47% against SPY, over 59 releases in 38 months. Tested either way. A round trip in these stocks costs about 2%. 261 of the 18332 releases had no text read (no exhibit 99 found) and are left out of both groups. FAIL.

Controls on each test's TEST months: untagged releases drawn at random in place of the tagged ones reached p < 0.05 in 2.0%–7.0% of 200 draws (at most 10% allowed); the month's best or worst releases in their place reached p ≤ 0.0005. All passed.

Reading this: a tag says only that the release's opening uses the words ("raises full-year guidance"); the rules miss other phrasings and sometimes tag a passing mention. The universe is chosen today (survivorship). A reaction with no drift means the market prices the event at once.


## Move sizes

Tests of the earnings move-size model M1 (`brain/moves.py`, `docs/MOVES.md`): the size of a release's move over days -1..+1 against SPY, whichever way, predicted as a range from what was known two sessions before. Per release the score is the mean pinball loss over the 10/25/50/75/90% quantiles; each month the history-only baseline's mean loss minus M1's, and the statistic is the mean of those monthly differences with a one-sided sign-flip p. The gate, declared before running: positive, p < 0.05, M1's 80% range holding 75%-85% of TEST releases, the controls passing, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger.

| Test | TEST releases (months) | Loss M1 | Loss baseline | Monthly difference | 90% interval | p | BH q | 80% range held | 50% range held | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1a | 17539 (33) | 1.587 | 1.611 | +0.028 | +0.022..+0.036 | 0.0001 | 0.001 | 77.1% | 47.9% | **PASS** |

- **M1a**: over 17539 releases from 2024-01 to 2026-09, M1's ranges scored 1.587 points of pinball loss against the baseline's 1.611 (better in 32 of 33 months). Its 80% range held 77.1% of the moves (the baseline's 77.0%) and its 50% range 47.9%. The chance of moving more than the usual move (median of the last eight) scored a Brier of 0.2421 against the baseline's 0.2456 and the base rate's 0.2497 (51.7% did). Controls: shuffled ranges reached p < 0.05 in 0.0% of 200 draws; the actual move as the middle reached p 0.0001. PASS.

Reading this: a size is not a direction. The universe is chosen today (survivorship), and the pinball loss is in points of move (0.5 is half a percentage point).


## Shocks

Tests of the shock radar S1 (`brain/shocks.py`, `docs/SHOCKS.md`): the chance that a stock moves 20% or more against SPY, either way, in one session of the next five, from what was known at a week's last close. Per stock-week the score is the log loss of the chance; each month the volatility-only baseline's mean loss minus S1's, and the statistic is the mean of those monthly differences with a one-sided sign-flip p. The gate, declared before running: positive, p < 0.05, S1's chances calibrated overall and in each week's top twentieth (actual over predicted within the declared band), the controls passing, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger.

| Test | TEST stock-weeks (months) | Loss S1 | Loss baseline | Monthly difference | 90% interval | p | BH q | Actual / predicted (all, top 5%) | Top 5% caught | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1a | 224166 (33) | 0.03196 | 0.03443 | +0.00247 | +0.00204..+0.00294 | 0.0001 | 0.001 | 1.24, 1.21 | 52% | **PASS** |

Tests of the earnings-aware radar S3 (`brain/catalysts.py`, `docs/SHOCKS.md`, Catalysts): S1 plus a report expected a quarter after the last one and the stock's earnings habit (the median move against SPY on its last eight report sessions), judged against S1 on the stock-weeks whose report is expected 1-8 days ahead. Per stock-week the score is the log loss of the chance; each month S1's mean loss minus S3's, and the statistic is the mean of those monthly differences with a one-sided sign-flip p. The gate, declared before running: positive, p < 0.05, S3's chances calibrated overall and in each week's top twentieth, the controls passing, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger.

| Test | TEST report-weeks (months) | Loss S3 | Loss S1 | Monthly difference | 90% interval | p | BH q | Actual / predicted (all, top 5%) | Top 5% caught (S3, S1) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S3a | 22732 (33) | 0.10694 | 0.11409 | +0.00715 | +0.00413..+0.01085 | 0.0003 | 0.002 | 1.59, 0.91 | 21%, 19% | **FAIL** |

Tests of what follows a shock: an up-shock episode opens on a session 20%+ above SPY without a results report and with no up-shock in the 30 days before. Its outcome is its excess return over the next 21 sessions, against the stocks in its 60-session volatility decile that did not shock that day. Per month, **percentile** averages the share of peers each episode beat less one half (the typical episode); **abnormal** averages each episode's outcome less its peers' mean (the average episode, before costs). The gate, declared before running: the mean of monthly values negative with a one-sided sign-flip p < 0.05, the controls passing, and Benjamini–Hochberg q < 0.10 across every gated test.

| Test | Statistic | TEST episodes (months) | Mean of monthly values | 90% interval | p | BH q | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S2a | percentile | 423 (33) | +0.0041 | -0.0281..+0.0347 | 0.5833 | 0.779 | **FAIL** |
| S2b | abnormal | 423 (33) | +0.0439 | -0.0223..+0.1133 | 0.8406 | 0.917 | **FAIL (controls)** |

- **S1a**: over 224166 stock-weeks of 1568 stocks from 2024-01 to 2026-09, 0.72% of stock-weeks had a 20%+ session (68% of those were up). S1's chances averaged 0.58% (the baseline's 0.72%) and beat the baseline's log loss in 31 of 33 months; AUC 0.905 against 0.857. Each week's top 1%, 5% and 10% by S1 held 22%, 52% and 68% of all shocks (the baseline's 21%, 45% and 59%); in S1's top 5% a shock came 7.5% of the time against a predicted 6.2%. Controls: shuffled chances reached p < 0.05 in 0.0% of 200 draws; an oracle reached p 0.0001. PASS.
- **S3a**: over 22732 report-week stock-weeks of 1534 stocks from 2024-01 to 2026-09, 2.63% had a 20%+ session (59% of those were up). S3's chances averaged 1.65% (S1's 1.17%) and beat S1's log loss in 23 of 33 months; AUC 0.835 against 0.818. Each week's top 5% and 10% of report weeks by S3 held 21% and 35% of their shocks (S1's 19% and 36%); in S3's top 5% a shock came 11.9% of the time against a predicted 13.1%. On every stock-week (223594; report weeks were 10% of them and held 37% of the shocks), S3's loss was 0.03120 against S1's 0.03190, better in 24 of 33 months (p 0.0002, not gated); each week's top 5% held 54% of all shocks (S1's 52%). Controls: shuffled chances reached p < 0.05 in 0.0% of 200 draws; an oracle reached p 0.0001. FAIL.
- **S2a** (percentile): 423 episodes of 191 stocks from 2024-01 to 2026-09. Over the next 21 sessions their median excess return was -1.7% and their mean +10.1%, against +3.9% for their peers: +6.26 points on average, beating 50.6% of peers (51% fell behind the middle one). Negative in 15 of 33 months. Jumps of 50%+: 42, -6.4 points; drug and biotech: 95, +1.1; under $5: 158, +15.2. Controls: a random peer in each episode's place reached p < 0.05 in 5.5% of 200 draws; an oracle reached p 0.0001. FAIL.
- **S2b** (abnormal): 423 episodes of 191 stocks from 2024-01 to 2026-09. Over the next 21 sessions their median excess return was -1.7% and their mean +10.1%, against +3.9% for their peers: +6.26 points on average, beating 50.6% of peers (51% fell behind the middle one). Negative in 13 of 33 months. Jumps of 50%+: 42, -6.4 points; drug and biotech: 95, +1.1; under $5: 158, +15.2. Controls: a random peer in each episode's place reached p < 0.05 in 19.5% of 200 draws; an oracle reached p 0.0001. FAIL (controls).

Reading this: a chance of a shock is a size, not a direction, and a fade measured against peers is not a trade: shorting a stock that just jumped costs borrow fees and risks a squeeze. The universe is chosen today (survivorship: companies bought out or delisted are missing, so shocks are undercounted and post-shock returns flattered).


## Weekly ranges

Tests of the scan's weekly range W1 (`brain/scan.py`, `docs/SCAN.md`): the range of a stock's return over the next five sessions, from what was known at a close (each week's last for W1a and W1b, every session's for W1c), on tradable stocks. Per stock-week the score is the mean pinball loss of the log return over the 5/16/50/84/95% quantiles; each month the baseline's mean loss (the range its 60-session volatility alone gives) minus W1's, and the statistic is the mean of those monthly differences with a one-sided sign-flip p. The gate, declared before running: positive, p < 0.05, W1's 68% range holding 63%-73% of TEST stock-weeks and its 90% range 86%-94%, the controls passing, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger.

| Test | TEST stock-weeks (months) | Loss W1 | Loss baseline | Monthly difference | 90% interval | p | BH q | 68% range held | 90% range held | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1a | 209939 (33) | 1.148 | 1.164 | +0.016 | +0.011..+0.021 | 0.0001 | 0.001 | 68.1% | 89.7% | **PASS** |
| W1b | 120721 (33) | 1.597 | 1.611 | +0.014 | +0.009..+0.018 | 0.0001 | 0.001 | 65.5% | 87.9% | **PASS** |
| W1c | 1592265 (33) | 1.329 | 1.346 | +0.017 | +0.014..+0.021 | 0.0001 | 0.001 | 67.4% | 89.1% | **PASS** |

- **W1a**: over 209939 stock-weeks of 1550 stocks from 2024-01 to 2026-09, W1's ranges scored 1.148 points of pinball loss against the baseline's 1.164 (better in 30 of 33 months). Its 68% range held 68.1% of the moves and its 90% range 89.7% (the baseline's 70.4% and 91.6%); 5.0% fell below the 90% range and 5.3% rose above it. By list (68% and 90%): broad 68.1% and 89.9% (105890); micro 68.2% and 89.3% (33981); small 67.9% and 89.5% (70068). With a report due: 64.7% and 87.1% (16642); without: 68.3% and 89.9%. The typical move was 2.7% either way, against a typical 68% range of ±4.0%. Within a week, W1's size ranked the stocks' actual moves with a rank correlation of 0.347 (the baseline's 0.300). Direction, reported and not tested: 51.9% of stock-weeks rose (the training years' share for stocks of the same size of move: 52.7%), and 49.1% ended above the range's middle. Controls: shuffled ranges reached p < 0.05 in 0.0% of 200 draws; a range that knows the outcome reached p 0.0001. PASS.
- **W1b**: over 120721 stock-weeks of 934 stocks from 2024-01 to 2026-09, W1's ranges scored 1.597 points of pinball loss against the baseline's 1.611 (better in 29 of 33 months). Its 68% range held 65.5% of the moves and its 90% range 87.9% (the baseline's 66.8% and 89.5%); 5.9% fell below the 90% range and 6.3% rose above it. By list (68% and 90%): watch ranks 1-1500 64.8% and 87.6% (49982); watch ranks 1501-3000 66.1% and 88.0% (70739). With a report due: 60.5% and 84.4% (9436); without: 66.0% and 88.1%. The typical move was 3.6% either way, against a typical 68% range of ±5.2%. Within a week, W1's size ranked the stocks' actual moves with a rank correlation of 0.383 (the baseline's 0.356). Direction, reported and not tested: 51.2% of stock-weeks rose (the training years' share for stocks of the same size of move: 52.5%), and 48.8% ended above the range's middle. Controls: shuffled ranges reached p < 0.05 in 0.0% of 200 draws; a range that knows the outcome reached p 0.0001. PASS.
- **W1c**: over 1592265 stock-weeks of 2483 stocks from 2024-01 to 2026-09, W1's ranges scored 1.329 points of pinball loss against the baseline's 1.346 (better in 28 of 33 months). Its 68% range held 67.4% of the moves and its 90% range 89.1% (the baseline's 68.9% and 90.5%); 5.2% fell below the 90% range and 5.7% rose above it. By list (68% and 90%): broad 68.3% and 90.0% (510196); micro 68.7% and 89.4% (163511); small 68.0% and 89.5% (336883); watch ranks 1-1500 65.1% and 87.6% (240846); watch ranks 1501-3000 66.2% and 88.1% (340829). With a report due: 65.2% and 87.3% (151046); without: 67.6% and 89.3%. The typical move was 3.0% either way, against a typical 68% range of ±4.4%. Within a week, W1's size ranked the stocks' actual moves with a rank correlation of 0.383 (the baseline's 0.347). Direction, reported and not tested: 51.6% of stock-weeks rose (the training years' share for stocks of the same size of move: 52.7%), and 49.0% ended above the range's middle. Controls: shuffled ranges reached p < 0.05 in 0.0% of 200 draws; a range that knows the outcome reached p 0.0001. PASS.

Reading this: a range is a size, not a direction; its middle sits near zero for every stock. The universe is chosen today (survivorship: companies bought out or delisted are missing, so the largest moves are undercounted), and the pinball loss is in points of log return.


## Mosaic

Tests of the mosaic's pictures (`brain/mosaic.py`, `docs/MOSAIC.md`): the chance that a filing revealing a hidden event (a prospectus that opens a sale of new shares; a late-filing notice or a notice that earlier accounts cannot be relied on; an 8-K reporting a restructuring or an impairment) comes in the next 20 sessions, from the public pieces known at a week's last close. TEST is read at periods at least 20 sessions apart, so no two outcomes overlap. Per stock the score is the log loss of the chance; each period the habit-only baseline's mean loss minus the model's, and the statistic is the mean of those period differences with a one-sided sign-flip p. The gate, declared before running: positive, p < 0.05, the chances calibrated overall and in each period's top twentieth (actual over predicted within the declared band), the controls passing, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger.

| Test | Picture | TEST stock-periods (periods) | Events | Loss | Loss habit | Period difference | 90% interval | p | BH q | Actual / predicted (all, top 5%) | Top 5% held | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1a | sale | 75141 (31) | 1027 | 0.05766 | 0.05932 | +0.00167 | +0.00095..+0.00236 | 0.0001 | 0.001 | 1.02, 0.91 | 38% | **PASS** |
| P2a | books | 75141 (31) | 283 | 0.01948 | 0.02048 | +0.00100 | +0.00060..+0.00141 | 0.0001 | 0.001 | 0.98, 0.97 | 54% | **PASS** |
| P3a | cuts | 75141 (31) | 363 | 0.02730 | 0.02829 | +0.00101 | +0.00057..+0.00148 | 0.0004 | 0.002 | 0.86, 0.80 | 34% | **PASS** |

Tests of whether the flagged stocks lag: each period's top twentieth by a picture's chance, each stock's return over SPY's across the next 20 sessions set against the period's unflagged stocks in its 60-session volatility decile. Per period, the share of peers each flagged stock beat less one half is averaged (the typical flagged stock). The gate, declared before running: the mean of the period values negative with a one-sided sign-flip p < 0.05, the controls passing, and Benjamini–Hochberg q < 0.10 across every gated test. The mean return against peers, which a trade would earn before costs, is reported beside it and is not gated.

| Test | Picture | Flagged (periods) | Mean of period values | 90% interval | p | BH q | Return against peers | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1b | sale | 3743 (31) | -0.0076 | -0.0270..+0.0117 | 0.2703 | 0.473 | +0.89% | **FAIL** |
| P2b | books | 3743 (31) | -0.0073 | -0.0199..+0.0049 | 0.1734 | 0.399 | +2.17% | **FAIL** |
| P3b | cuts | 3743 (31) | -0.0108 | -0.0237..+0.0027 | 0.0991 | 0.295 | -0.31% | **FAIL** |

- **P1a** (a sale of new shares): over 75141 stock-periods of 2500 stocks from 2024-01-05 to 2026-09-04, the event came within 20 sessions in 1.37% of them (1027). The model's chances averaged 1.35% (the habit's 1.26%) and beat the habit's log loss in 24 of 31 periods; AUC 0.875 against 0.857. Each period's top 1%, 5% and 10% by the model held 10%, 38% and 59% of the events (the habit's 9%, 32% and 54%); in the model's top 5% the event came 10.6% of the time against a predicted 11.5% (in the habit's top 5%: 8.8%). Where the company had no such event in three years (58462 stock-periods, 194 events, 0.33%), the model's top 5% there had the event 1.7% of the time and held 25% of those events (AUC 0.774; reported, not gated). Controls: shuffled chances reached p < 0.05 in 0.0% of 200 draws; an oracle reached p 0.0001. PASS.
- **P2a** (late or unreliable accounts): over 75141 stock-periods of 2500 stocks from 2024-01-05 to 2026-09-04, the event came within 20 sessions in 0.38% of them (283). The model's chances averaged 0.38% (the habit's 0.38%) and beat the habit's log loss in 25 of 31 periods; AUC 0.871 against 0.802. Each period's top 1%, 5% and 10% by the model held 35%, 54% and 64% of the events (the habit's 27%, 54% and 60%); in the model's top 5% the event came 4.1% of the time against a predicted 4.2% (in the habit's top 5%: 4.1%). Where the company had no such event in three years (68072 stock-periods, 112 events, 0.16%), the model's top 5% there had the event 0.7% of the time and held 21% of those events (AUC 0.766; reported, not gated). Controls: shuffled chances reached p < 0.05 in 0.0% of 200 draws; an oracle reached p 0.0001. PASS.
- **P3a** (a restructuring or a write-off): over 75141 stock-periods of 2500 stocks from 2024-01-05 to 2026-09-04, the event came within 20 sessions in 0.48% of them (363). The model's chances averaged 0.56% (the habit's 0.41%) and beat the habit's log loss in 20 of 31 periods; AUC 0.798 against 0.710. Each period's top 1%, 5% and 10% by the model held 12%, 34% and 52% of the events (the habit's 9%, 31% and 47%); in the model's top 5% the event came 3.3% of the time against a predicted 4.1% (in the habit's top 5%: 3.0%). Where the company had no such event in three years (65895 stock-periods, 175 events, 0.27%), the model's top 5% there had the event 1.1% of the time and held 20% of those events (AUC 0.707; reported, not gated). Controls: shuffled chances reached p < 0.05 in 0.0% of 200 draws; an oracle reached p 0.0001. PASS.
- **P1b** (a sale of new shares): 3743 flagged stock-periods of 427 stocks from 2024-01-05 to 2026-09-04, with a mean chance of 11.5%. Over the next 20 sessions their median return over SPY's was -1.4% and their mean +2.6%, against +1.7% for their peers. They beat 49.2% of peers (52% fell behind the middle one), negative in 14 of 31 periods. Against peers they returned +0.89% a period on average (90% interval -0.61%..+2.49%, one-sided p 0.8159; reported, not gated). Controls: a random peer in each flagged stock's place reached p < 0.05 in 7.5% of 200 draws; an oracle reached p 0.0001. FAIL.
- **P2b** (late or unreliable accounts): 3743 flagged stock-periods of 336 stocks from 2024-01-05 to 2026-09-04, with a mean chance of 4.2%. Over the next 20 sessions their median return over SPY's was -1.4% and their mean +3.5%, against +1.3% for their peers. They beat 49.3% of peers (50% fell behind the middle one), negative in 15 of 31 periods. Against peers they returned +2.17% a period on average (90% interval +0.57%..+3.79%, one-sided p 0.9798; reported, not gated). Controls: a random peer in each flagged stock's place reached p < 0.05 in 3.5% of 200 draws; an oracle reached p 0.0001. FAIL.
- **P3b** (a restructuring or a write-off): 3743 flagged stock-periods of 412 stocks from 2024-01-05 to 2026-09-04, with a mean chance of 4.1%. Over the next 20 sessions their median return over SPY's was -1.4% and their mean +0.6%, against +0.9% for their peers. They beat 48.9% of peers (51% fell behind the middle one), negative in 18 of 31 periods. Against peers they returned -0.31% a period on average (90% interval -1.27%..+0.62%, one-sided p 0.2903; reported, not gated). Controls: a random peer in each flagged stock's place reached p < 0.05 in 6.0% of 200 draws; an oracle reached p 0.0001. FAIL.

Reading this: a chance that a filing comes is not a direction, and a lag measured against peers is not a trade: shorting costs borrow fees and the flagged stocks are volatile. The universe is chosen today (survivorship: companies that failed or were bought are missing, so the events of the weakest companies are undercounted and the flagged stocks' returns flattered).


## Step 2: the quality tilt and buying after fear

Declared once each before running (`brain/research/tilttest.py`, `docs/TILT.md`). The gate: the statistic positive, p < 0.05, the controls passing, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger.

| Test | What | Statistic | p | BH q | Controls | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| T1a | top third by quality against the list, a month | -0.095% | 0.8490 | 0.917 | passed | **FAIL** |
| T1b | SPY timed after fear against a constant mix, a week | -0.012% | 0.8783 | 0.917 | passed | **FAIL** |

- **T1a** (the quality tilt): 93 TEST months from 2019-01-31 to 2026-09-30, a list of 500 stocks a month chosen by market value as known each month from 2537 candidates. The top third by quality beat the list by -0.095% a month after costs (permutation p 0.8490; 42 of 93 months won; turnover 33% a month). TRAIN (2011-2018): +0.075% a month, p 0.0062. Each trait alone over TEST (reported): calm -0.348% (p 1.000), near_high -0.144% (p 0.905), no_dilution +0.166% (p 0.001), profit +0.032% (p 0.239). By market value, the list without its bottom third beat the list by market value by +0.310% a month (reported). A year over TEST: the tilt +12.8%, the list at equal weight +13.5%, the list by market value +10.0%, the list without its junk +15.6%, SPY +15.8%, RSP (the S&P 500 at equal weight) +12.6%. The list's own gap, which is survivorship plus what the list leaves out: at equal weight against RSP +1.0% a year, by market value against SPY -5.8%. Controls: an oracle reached p 0.0005; random signals redrawn monthly reached p < 0.05 in 2.0% of draws, held fixed per stock in 5.5% (not gated: above about 5%, the null is too narrow for a slow signal by that much). FAIL.
- **T1b** (buying after fear): 935 TEST weeks from 1993-02-05 to 2010-12-31, 62 spike weeks in 16 episodes, 31% of weeks after fear. SPY beat bills by -0.001% a week after fear and +0.171% otherwise. The timed portfolio beat the constant mix by -0.0120% a week (p 0.8783). A year: timed +6.3%, constant mix +7.2%, SPY held at 100% +7.9%; worst fall -51% against SPY's -55%. Seen before, not gated (2011-01-07 to 2026-11-06, 827 weeks): SPY beat bills by +0.378% a week after fear and +0.053% otherwise; the timed portfolio against the constant mix +0.0217% a week (p 0.0030); a year timed +13.4%, SPY +13.7%. Controls: an oracle reached p 0.0005; random spike weeks reached p < 0.05 in 5.5% of 200 draws. FAIL.

Reading this: the list is chosen month by month by what was known then, but it holds only companies still filing today; RSP held the ones that failed or were bought. It also leaves out companies whose share count could not be read from their filings (several share classes: Berkshire, Visa, Alphabet at the start of TEST) or whose accounts were not read at all (Exxon), so by market value it is not the S&P 500, and its gap to SPY is mostly that. The traits came from published research and from looks at 2011-2026 on other lists, so T1a's TEST is not unseen: a PASS is an upper bound. T1b's TEST (1993-2010) had not been looked at. Paper only; nothing here places orders.


## Ripples

Declared before running (`brain/research/rippletest.py`, `docs/RIPPLES.md`). A stone is an own move (market removed) of 3+ usual days; its wave reaches each stock by how far that stock moved with it over the year before. TRAIN 2011-2023, TEST 2024-01..2026-09, the research universe. Gated tests pass on a positive statistic with p < 0.05, their controls, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger.

| Test | Question | Statistic | 90% interval | p | BH q | Controls | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1b | how high the waves travel (next five sessions' size) | +0.00604 | +0.00314..+0.00911 | 0.0011 | 0.005 | FAILED | **FAIL (controls)** |
| R1c | which way the waves travel (next five sessions' direction) | -0.0012 | -0.0137..+0.0109 | 0.5680 | 0.779 | passed | **FAIL** |
| R1d | do waves that add up move a stock more than waves that cancel | -0.0108 | -0.0255..+0.0036 | 0.8782 | 0.917 | passed | **FAIL** |

- **R1a** (reported, the waves on the day): 689 TEST sessions, 24.9 stones and 1184 stocks reached a session. The wave's height matched the stock's own move that day with a rank correlation of +0.237. Own move (usual days) by ring: first 0.747, second 0.633, beyond 0.638; on dry land (no link to any stone) 0.593. Wave height by ring: first 0.282, second 0.189, beyond 0.188. Waves absorbed (under a quarter of a usual day) 58.0%, felt (a quarter to one) 42.0%, a hit (one or more) 0.06%. Interference against the day's own move size: +0.121.
- **R1b** (size): 816086 TEST stock-days in 33 months. Squared error of the next five sessions' size: baseline 0.5702, with the waves 0.5640; better in 26 of 33 months. Next-five-session size (usual days of five sessions) after a wave absorbed 0.778, felt 0.792, a hit 0.997. Controls: shuffled waves reached p < 0.05 in 100.0% of draws; an oracle p 0.0001. FAIL (controls).
- **R1c** (direction): 693396 TEST stock-days with wave energy of half a usual day or more, 650 sessions. Monthly mean rank correlation of the wave with the next five sessions' own move -0.0012, positive in 14 of 33 months. The top fifth by wave against the bottom fifth: -0.001 usual days of five sessions. Reported beside it: the wave not yet moved (wave less own move) +0.0190; the stock's own move reversed +0.0187. Controls: shuffled waves reached p < 0.05 in 6.5% of draws; an oracle p 0.0001. FAIL.
- **R1d** (interference): 303616 TEST stock-days with wave energy of one usual day or more (95% of them more cancelled than added). Monthly mean rank correlation of interference with the next five sessions' size beyond the baseline -0.0108, positive in 11 of 33 months. Next-five-session size where waves added 0.783, where they cancelled 0.865. Controls: shuffled reached p < 0.05 in 5.5% of draws; an oracle p 0.0001. FAIL.

Reading this: the pond is the research universe's own stocks, chosen by today's lists (survivors). A wave on the day is a description, not a forecast; only R1b-R1d ask what follows. Paper only; nothing here places orders.


## Earnings: expected and what followed

Declared before running (`brain/research/earntest.py`, `docs/EARNINGS.md`). Every results report of the research, small, micro and newer-listing universes; the setup is what the price held at the close before the report. TRAIN 2011-2023, TEST 2024-01..2026-09. Per month the statistic is averaged (a busy month counts once), with a one-sided sign-flip p over the months. Gated tests pass on a positive statistic with p < 0.05, their controls, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger.

| Test | Question | Statistic | 90% interval | p | BH q | Controls | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| X1a | does the run into a report predict the reaction (priced in) | +0.0291 | +0.0126..+0.0459 | 0.0029 | 0.012 | passed | **PASS** |
| X1b | does a stock's habit at its reports predict the reaction | -0.0083 | -0.0263..+0.0080 | 0.7706 | 0.917 | passed | **FAIL** |
| X1c | do the most similar past setups predict the reaction (analogs) | +0.0085 | -0.0048..+0.0227 | 0.1576 | 0.399 | passed | **FAIL** |
| X1d | does the reaction keep going or come back over the next week | +0.0058 | -0.0149..+0.0266 | 0.3264 | 0.538 | passed | **FAIL** |
| X1e | what each kind of episode did (priced in, relief, ...) | +nan | +nan..+nan | – | – | – | **REPORT** |
| X1f | does the expectation score (run-up against habit) predict the reaction | +0.0195 | +0.0001..+0.0377 | 0.0519 | 0.182 | passed | **FAIL** |

- **X1a**: 26814 TEST episodes in 33 months; monthly rank correlation with the reaction +0.0291 (positive in 23 of 33 months). The top fifth by the prediction averaged +0.31% (51% rose), the bottom fifth +0.49% (49% rose). Controls: shuffled predictions reached p < 0.05 in 7.5% of draws; the oracle p 0.0005. PASS.
- **X1b**: 26814 TEST episodes in 33 months; monthly rank correlation with the reaction -0.0083 (positive in 13 of 33 months). The top fifth by the prediction averaged +0.27% (50% rose), the bottom fifth +0.38% (49% rose). Controls: shuffled predictions reached p < 0.05 in 6.0% of draws; the oracle p 0.0005. FAIL.
- **X1c**: 26814 TEST episodes in 33 months; monthly rank correlation with the reaction +0.0085 (positive in 15 of 33 months). The top fifth by the prediction averaged +0.81% (51% rose), the bottom fifth +0.34% (50% rose). Controls: shuffled predictions reached p < 0.05 in 3.0% of draws; the oracle p 0.0005. FAIL.
- **X1d**: 26814 TEST episodes in 33 months; monthly rank correlation with the next week's drift +0.0058 (positive in 17 of 33 months). The top fifth by the prediction averaged +0.05% (47% rose), the bottom fifth -0.34% (45% rose). Controls: shuffled predictions reached p < 0.05 in 5.5% of draws; the oracle p 0.0005. FAIL.
- **X1e**: 26814 TEST episodes by kind. surprise down: 7758 episodes, reaction -7.47%, next week -0.30%, kept going 54%; surprise up: 7650 episodes, reaction +8.21%, next week +0.06%, kept going 48%; no reaction: 4485 episodes, reaction -0.02%, next week -0.21%, kept going 51%; priced in: 1755 episodes, reaction -7.22%, next week -0.29%, kept going 55%; confirmed: 1735 episodes, reaction +8.85%, next week +0.11%, kept going 49%; relief: 1730 episodes, reaction +7.83%, next week -0.38%, kept going 45%; capitulation: 1701 episodes, reaction -7.36%, next week +0.06%, kept going 51%. REPORT.
- **X1f**: 26814 TEST episodes in 33 months; monthly rank correlation with the reaction +0.0195 (positive in 25 of 33 months). The top fifth by the prediction averaged +0.34% (52% rose), the bottom fifth +0.49% (49% rose). Controls: shuffled predictions reached p < 0.05 in 4.0% of draws; the oracle p 0.0005. FAIL.

## Promises: said and delivered

Declared before running (`brain/research/promisetest.py`, `docs/PROMISES.md`). A promise is revenue guidance in dollars read from an earnings release of the research universe; its outcome is the revenue first reported for the promised period, against the promise's midpoint. TRAIN is promises made 2011-2023, TEST 2024-01..2026-09. Per reporting season (a calendar quarter) the statistic is the rank correlation of the prediction with the outcome, averaged over seasons, with a one-sided sign-flip p over the seasons. Gated tests pass on a positive statistic with p < 0.05, their controls, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger. G1a is about what a company delivers, not about its stock.

| Test | Question | Statistic | 90% interval | p | BH q | Controls | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1a | does a company's habit of beating its own revenue promise carry to the next promise | +0.4743 | +0.4300..+0.5141 | 0.0007 | 0.004 | passed | **PASS** |
| G1b | the report session's move by what the release said | +nan | +nan..+nan | – | – | – | **REPORT** |
| G2a | does the habit of beating its own revenue promise carry to the next promise, on the small, micro and newer-listing universes | +0.4861 | +0.4521..+0.5212 | 0.0006 | 0.003 | passed | **PASS** |
| G2b | beyond its habit, is a promise that asks for more growth beaten by more | +0.0955 | +0.0230..+0.1656 | 0.0312 | 0.125 | passed | **FAIL** |
| G2c | after a release, do the next five sessions follow what was delivered against the habit | -0.0002 | -0.0465..+0.0388 | 0.5026 | 0.728 | passed | **FAIL** |
| G2d | after a release, do the next five sessions follow the change in the year's promise | +0.0337 | -0.0053..+0.0747 | 0.1075 | 0.301 | passed | **FAIL** |
| G3a | is a promise that asks for more growth followed by a better report session, three months on | +0.0381 | -0.0094..+0.0839 | 0.1151 | 0.307 | passed | **FAIL** |

- **G1a**: 670 quarterly promises in 11 seasons; rank correlation of its habit with the surprise on the new promise +0.4743 (positive in 11 of 11 seasons). The top fifth by the prediction delivered a median +5.13% against its midpoint, the bottom fifth +0.81%. Controls: shuffled predictions reached p < 0.05 in 6.5% of draws; the oracle p 0.0005. Reported, not gated: of 670 promises with a habit, the delivery was a median 2.41% from the midpoint and 1.43% from the midpoint moved by the habit; it fell inside the 2-in-3 range 77.3% of the time and inside the 9-in-10 range 95.7%; 3.9% missed the company's own range. PASS.
- **G1b**: 765 quarterly promises of 106 companies and 594 changes of a year's promise on TEST. delivered above its last promise: 285 releases, the report session +2.21% against the market (median +2.28%, 61% rose), the week after +0.73%; delivered inside it: 355 releases, the report session -0.23% against the market (median -0.54%, 48% rose), the week after +0.25%; delivered below it: 28 releases, the report session -6.30% against the market (median -2.86%, 18% rose), the week after +1.45%; delivered more against its midpoint than its habit: 359 releases, the report session +2.35% against the market (median +2.28%, 60% rose), the week after +0.74%; delivered less than its habit: 277 releases, the report session -2.04% against the market (median -2.33%, 40% rose), the week after +0.19%; raised the year's promise: 199 releases, the report session +3.29% against the market (median +2.69%, 62% rose), the week after +0.09%; held it: 333 releases, the report session +0.22% against the market (median +0.10%, 50% rose), the week after -0.37%; cut it: 62 releases, the report session -3.70% against the market (median -2.66%, 34% rose), the week after +1.65%. REPORT.
- **G2a**: 1357 quarterly promises of 299 companies in 11 seasons; rank correlation of the habit with the surprise on the new promise +0.4861 (positive in 11 of 11 seasons). The top fifth by habit delivered a median +4.56% against its midpoint, the bottom fifth +0.78%. Reported, not gated: the delivery was a median 2.34% from the midpoint and 1.37% from the midpoint moved by the habit; it fell inside the 2-in-3 range 77.2% of the time and inside the 9-in-10 range 95.1%; 4.6% missed the company's own range. Controls: shuffled predictions reached p < 0.05 in 2.0% of draws; the oracle p 0.0005. PASS.
- **G2b**: 1357 rows in 11 seasons; rank correlation of the stretch with the surprise less the habit +0.0955 (positive in 8 of 11 seasons). The fifth of promises asking for most delivered a median +0.29% beyond its habit, the fifth asking for least -0.39%. Controls: shuffled predictions reached p < 0.05 in 4.0% of draws; the oracle p 0.0020. FAIL.
- **G2c**: 1250 rows in 11 seasons; rank correlation of the delivery against the habit with the stock's next five sessions -0.0002 (positive in 7 of 11 seasons). After the top fifth the stock moved a mean -0.14% against the market, after the bottom fifth -0.36%. Controls: shuffled predictions reached p < 0.05 in 4.5% of draws; the oracle p 0.0010. FAIL.
- **G2d**: 1875 rows in 11 seasons; rank correlation of the change in the year's promise with the stock's next five sessions +0.0337 (positive in 7 of 11 seasons). The five sessions after the report session, against the market: raised (543): +0.31%, 49% rose; held (1060): -0.57%, 43% rose; cut (272): -0.92%, 44% rose. Controls: shuffled predictions reached p < 0.05 in 7.5% of draws; the oracle p 0.0010. FAIL.
- **G3a**: 2352 promised quarters of 374 companies in 11 seasons; rank correlation of the stretch with the move of the report session that delivered the quarter +0.0381 (positive in 8 of 11 seasons). That session, against the market: the third asking for least (788): -0.57%, 46% rose; the third asking for middle (776): +1.20%, 52% rose; the third asking for most (788): +0.90%, 49% rose. Controls: shuffled predictions reached p < 0.05 in 5.0% of draws; the oracle p 0.0010. FAIL.

## Customers: who it sells to

Declared before running (`brain/research/customertest.py`, `docs/CUSTOMERS.md`). A link is point-in-time: a company of the research universe named a listed customer in an annual report already filed (every report since 2010, read by fixed rules). An event is a results report of the customer while a link holds; events where the supplier itself reports nearby are left out. TRAIN is customers' reports of 2011-2023, TEST 2024-01..2026-09. Per reporting season the statistic is a rank correlation with the customer's report-session move, averaged over seasons, with a one-sided sign-flip p. Gated tests pass on a positive statistic with p < 0.05, their controls, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger. C1a is about the same session: it explains a move and is not a forecast.

| Test | Question | Statistic | 90% interval | p | BH q | Controls | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1a | on a customer's report session, does a supplier that names it move with it more than its industry peers that do not | +0.0616 | -0.0096..+0.1249 | 0.1001 | 0.295 | passed | **FAIL** |
| C1b | does a customer's report-session move tell its supplier's next five sessions | -0.0927 | -0.1487..-0.0334 | 0.9818 | 1.000 | passed | **FAIL** |

- **C1a**: 694 reports of a customer while a link held (118 links, 52 customers) in 11 seasons. Rank correlation of the customer's report-session move with its suppliers' move the same session +0.1089, with their industry peers' +0.0473: the link adds +0.0616 (positive in 7 of 11 seasons). After the customers' best fifth of reports (+7.87%), their suppliers moved +0.00% the same session (48% rose); after the worst fifth (-8.34%), -0.36% (41% rose). The same session: it explains a move, it is not a forecast. Controls: shuffled customer moves reached p < 0.05 in 0.0% of draws; the oracle p 0.0005. FAIL.
- **C1b**: 694 reports of a customer while a link held, in 11 seasons. Rank correlation of the customer's report-session move with its suppliers' next 5 sessions -0.0927 (positive in 2 of 11 seasons); the next 21 sessions -0.0267; industry peers' next 5 sessions -0.0595. Controls: shuffled customer moves reached p < 0.05 in 5.0% of draws; the oracle p 0.0015. FAIL.
