# Research results

Generated 2026-10-19T20:31:44Z from `research/ledger.jsonl`. Every number below is copied from the ledger.

TRAIN 2011-01-31..2018-12-31, TEST from 2019-01-01. Monthly, equal weight against the equal-weight universe, 10 bps per unit turnover. Holdings: the top third by signal unless the hypothesis names another selection (Select column; see `brain/research/harness.py`).

Universes (each has its own panel and controls):

- `universe/us_large.json`: 60 US stocks (top SEC filers by market value with 10-K XBRL and prices since 2010). Hypotheses: H1-mom12_1, H2-rev1, H3-earnings_yield, H4-low_asset_growth, H5-ridge, H10-text_unchanged.
- `universe/us_broad.json`: 741 US stocks (SEC ranks 1-1500 with 10-K XBRL and prices since 2010). Hypotheses: H6-insider_buying, H3c-earnings_yield_ttm, H7b-earnings_reaction, H8b-sue, H3b-earnings_yield_ttm, H7-earnings_reaction, H8-sue.

Gate (declared before running): TEST mean net excess > 0, permutation p < 0.05, and Benjamini–Hochberg q < 0.10 across every hypothesis in the ledger and every gated forecast test (see Forecasts) and every gated link test (see Links).

| Hypothesis | Universe | Select | TEST net excess / mo | Annualised | Info ratio | Months won | Perm p | BH q | TRAIN net excess / mo | TRAIN p | TEST coverage | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1-mom12_1 | us_large | top_frac | +0.11% | +1.3% | 0.12 | 42/92 | 0.1758 | 0.620 | +0.22% | 0.0148 | 100% | **FAIL** |
| H2-rev1 | us_large | top_frac | -0.01% | -0.1% | -0.01 | 49/92 | 0.2142 | 0.620 | +0.05% | 0.0722 | 100% | **FAIL** |
| H3-earnings_yield | us_large | top_frac | -0.23% | -2.7% | -0.31 | 46/92 | 0.8710 | 0.885 | -0.10% | 0.8714 | 85% | **FAIL** |
| H4-low_asset_growth | us_large | top_frac | -0.11% | -1.3% | -0.22 | 48/92 | 0.7363 | 0.885 | -0.09% | 0.7251 | 97% | **FAIL** |
| H5-ridge | us_large | top_frac | +0.19% | +2.3% | 0.27 | 48/92 | 0.0510 | 0.620 | +0.14% | 0.0546 | 100% | **FAIL** |
| H6-insider_buying | us_broad | positive | +0.00% | +0.0% | 0.01 | 48/92 | 0.2789 | 0.620 | +0.11% | 0.0080 | 100% | **FAIL** |
| H3c-earnings_yield_ttm | us_broad | top_frac | -0.11% | -1.3% | -0.25 | 48/92 | 0.8846 | 0.885 | -0.11% | 0.9282 | 94% | **FAIL** |
| H7b-earnings_reaction | us_broad | top_frac_vs_all | -0.24% | -2.8% | -0.55 | 36/92 | 0.5845 | 0.885 | -0.07% | 0.1838 | 36% | **FAIL** |
| H8b-sue | us_broad | top_frac | -0.05% | -0.6% | -0.20 | 48/92 | 0.4867 | 0.885 | -0.03% | 0.1058 | 92% | **FAIL** |
| H10-text_unchanged | us_large | top_frac | -0.16% | -1.9% | -0.39 | 42/92 | 0.7532 | 0.885 | -0.27% | 0.9066 | 95% | **FAIL** |
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

`universe/us_large.json` (latest run, 2026-10-19):

- Oracle (next month's actual return as the signal): TEST p = 0.0005, gross excess +8.04%/mo. The harness can see a real edge.
- 200 random signals, redrawn every month: 3.0% reached p < 0.05 (expected about 5%). This is what the null is checked against: signals that change from month to month.
- 200 random scores held fixed per stock in every month: 13.5% reached p < 0.05. The null redraws its holdings every month, so for a signal that persists from month to month (fundamentals, slow-moving events) it is too narrow and the p-value too small; above about 5%, this is by how much. Reported, not part of the declared gate.
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
| F1b-w5-test | 5 | 2019-01-03..2026-08-11 (383) | 283803 | +0.010 | +0.011 | p 0.0870 | 0.620 | +0.009 | +0.008 | -0.06% | -0.08% (+0.00%) | 50.4% | 67.8% (66.5%-69.1%) | 89.9% (89.0%-90.7%) | **FAIL** |
| F1c-m21-test | 21 | 2019-01-07..2026-09-15 (93) | 68913 | +0.008 | +0.014 | p 0.2096 | 0.620 | +0.020 | -0.001 | +0.23% | -0.42% (-0.01%) | 50.2% | 69.3% (67.0%-71.5%) | 90.9% (89.4%-92.4%) | **FAIL** |
| F1d-llm-w5-recent | 5 | 2026-08-18..2026-10-07 (8) | 96 | -0.003 | -0.011 | 90% -0.169..+0.147; Brier F1 - model +0.0097 (p 0.352) | – | +0.017 | -0.145 | -0.17% | +0.19% (+0.20%) | 37.5% | 76.0% (67.3%-84.8%) | 92.7% (82.5%-100.0%) | **REPORT** |

IC: Spearman correlation of the chance to beat SPY with what each stock then did beyond SPY. Size-neutral IC: the same after the forecast's rank is cleared of its fit on market value, since the universe holds today's survivors and small-then predicts rose-later. Size IC and Momentum IC: the same samples ranked by size alone (smaller first) and by 12-1 momentum alone. Spread: top fifth minus bottom fifth by forecast, mean excess per date (F1d: halves of 12). Brier skill: against the training base rate as a constant; the within-date part is the only one that speaks to ranking. Placement: share placed on the right side of the date's median. Bands: share of stocks whose close at the exit fell inside, per date (interval).

### What the range and the chance showed

- F1a-w5-recent (2026-08-18 to 2026-10-07, 8 dates): the 68% band held 69.0% of the time per date (date-level 90% interval 62.8% to 75.2%) and the 90% band 91.9% (88.1% to 95.7%), against 68% and 90% aimed for; its pinball loss was 1% above a naive normal band's. Then the chance to beat SPY had mean size-neutral IC +0.042 (90% interval -0.099..+0.183); retest (descriptive): inconclusive; it can detect mean IC >= 0.15 at 80% power.
- F1b-w5-test (2019-01-03 to 2026-08-11, 383 dates): the 68% band held 67.8% of the time per date (date-level 90% interval 66.5% to 69.1%) and the 90% band 89.9% (89.0% to 90.7%), against 68% and 90% aimed for; its pinball loss was 2% below a naive normal band's. Then the chance to beat SPY ranked stocks beyond size with mean IC +0.011 (date-level p 0.0870, q 0.620): FAIL. Controls passed: the oracle reached p 0.00005 (smallest possible 0.00005), so the test can reject, and 3.0% of 200 coin-flipped placebos passed (1%-10% expected): a check of the sign-flip test itself, not of the forecaster, since coin-flipped signs pass about 5% of the time for any series, a look-ahead leak included (the features' leak tests are what check for one). Within-date permutation p 0.0010 (a diagnostic: too narrow, since stocks move together).
- F1c-m21-test (2019-01-07 to 2026-09-15, 93 dates): the 68% band held 69.3% of the time per date (date-level 90% interval 67.0% to 71.5%) and the 90% band 90.9% (89.4% to 92.4%), against 68% and 90% aimed for; its pinball loss was 1% below a naive normal band's. Then the chance to beat SPY ranked stocks beyond size with mean IC +0.014 (date-level p 0.2096, q 0.620): FAIL. Controls passed: the oracle reached p 0.00005 (smallest possible 0.00005), so the test can reject, and 5.0% of 200 coin-flipped placebos passed (1%-10% expected): a check of the sign-flip test itself, not of the forecaster, since coin-flipped signs pass about 5% of the time for any series, a look-ahead leak included (the features' leak tests are what check for one). Within-date permutation p 0.0010 (a diagnostic: too narrow, since stocks move together).
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
| L1-oil-airlines-next | oil_price_up → airline_stocks_down | next month | 93 (2019-01..2026-09) | -0.49% per +10% oil | -2.28%..+0.45% | 0.2190 | 0.620 | -1.76% | **FAIL** |
| L1-oil-producers-same | oil_price_up → oil_producer_stocks_up | same month | 93 (2019-01..2026-09) | +2.96% per +10% oil | +1.19%..+5.81% | 0.0002 | – | +2.90% | **REPORT** |
| L1-oil-producers-next | oil_price_up → oil_producer_stocks_up | next month | 93 (2019-01..2026-09) | -0.41% per +10% oil | -1.83%..+0.93% | 0.7489 | 0.885 | +0.20% | **FAIL** |
| L1-rates-homebuilders-same | interest_rates_up → homebuilder_stocks_down | same month | 93 (2019-01..2026-09) | -2.29% per +0.25 pt 10-year | -3.70%..-0.95% | 0.0018 | – | -0.50% | **REPORT** |
| L1-rates-homebuilders-next | interest_rates_up → homebuilder_stocks_down | next month | 93 (2019-01..2026-09) | +0.48% per +0.25 pt 10-year | -0.70%..+1.80% | 0.7395 | 0.885 | -0.26% | **FAIL** |
| L1-rates-banks-same | interest_rates_up → bank_stocks_up | same month | 93 (2019-01..2026-09) | +1.05% per +0.25 pt 10-year | +0.03%..+2.10% | 0.0266 | – | +3.44% | **REPORT** |
| L1-rates-banks-next | interest_rates_up → bank_stocks_up | next month | 93 (2019-01..2026-09) | +0.44% per +0.25 pt 10-year | -0.52%..+1.35% | 0.2088 | 0.620 | +0.15% | **FAIL** |
| L1-rates-tech-same | interest_rates_up → tech_stocks_down | same month | 93 (2019-01..2026-09) | -1.08% per +0.25 pt 10-year | -1.77%..-0.27% | 0.0180 | – | +0.42% | **REPORT** |
| L1-rates-tech-next | interest_rates_up → tech_stocks_down | next month | 93 (2019-01..2026-09) | -0.33% per +0.25 pt 10-year | -1.07%..+0.42% | 0.2587 | 0.620 | -0.24% | **FAIL** |

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

