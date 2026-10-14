# Research results

Generated 2026-10-14T00:02:41Z from `research/ledger.jsonl`. Every number below is copied from the ledger.

TRAIN 2011-01-31..2018-12-31, TEST from 2019-01-01. Monthly, equal weight against the equal-weight universe, 10 bps per unit turnover. Holdings: the top third by signal unless the hypothesis names another selection (Select column; see `brain/research/harness.py`).

Universes (each has its own panel and controls):

- `universe/us_large.json`: 60 US stocks (top SEC filers by market value with 10-K XBRL and prices since 2010). Hypotheses: H1-mom12_1, H2-rev1, H3-earnings_yield, H4-low_asset_growth, H5-ridge.
- `universe/us_broad.json`: 741 US stocks (SEC ranks 1-1500 with 10-K XBRL and prices since 2010). Hypotheses: H6-insider_buying, H3c-earnings_yield_ttm, H7b-earnings_reaction, H8b-sue, H3b-earnings_yield_ttm, H7-earnings_reaction, H8-sue.

Gate (declared before running): TEST mean net excess > 0, permutation p < 0.05, and Benjamini–Hochberg q < 0.10 across every hypothesis in the ledger.

| Hypothesis | Universe | Select | TEST net excess / mo | Annualised | Info ratio | Months won | Perm p | BH q | TRAIN net excess / mo | TRAIN p | TEST coverage | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1-mom12_1 | us_large | top_frac | +0.11% | +1.3% | 0.12 | 42/92 | 0.1758 | 0.837 | +0.22% | 0.0148 | 100% | **FAIL** |
| H2-rev1 | us_large | top_frac | -0.01% | -0.1% | -0.01 | 49/92 | 0.2142 | 0.837 | +0.05% | 0.0722 | 100% | **FAIL** |
| H3-earnings_yield | us_large | top_frac | -0.23% | -2.7% | -0.31 | 46/92 | 0.8710 | 0.885 | -0.10% | 0.8714 | 85% | **FAIL** |
| H4-low_asset_growth | us_large | top_frac | -0.11% | -1.3% | -0.22 | 48/92 | 0.7363 | 0.885 | -0.09% | 0.7251 | 97% | **FAIL** |
| H5-ridge | us_large | top_frac | +0.19% | +2.3% | 0.27 | 48/92 | 0.0510 | 0.612 | +0.14% | 0.0546 | 100% | **FAIL** |
| H6-insider_buying | us_broad | positive | +0.00% | +0.0% | 0.01 | 48/92 | 0.2789 | 0.837 | +0.11% | 0.0080 | 100% | **FAIL** |
| H3c-earnings_yield_ttm | us_broad | top_frac | -0.11% | -1.3% | -0.25 | 48/92 | 0.8846 | 0.885 | -0.11% | 0.9282 | 94% | **FAIL** |
| H7b-earnings_reaction | us_broad | top_frac_vs_all | -0.24% | -2.8% | -0.55 | 36/92 | 0.5845 | 0.885 | -0.07% | 0.1838 | 36% | **FAIL** |
| H8b-sue | us_broad | top_frac | -0.05% | -0.6% | -0.20 | 48/92 | 0.4867 | 0.885 | -0.03% | 0.1058 | 92% | **FAIL** |
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

## Reading this

- FAIL means the claim did not beat a shuffled signal on unseen months after costs. It does not mean the opposite is true.
- `universe/us_large.json`: 92 TEST months and 60 stocks.
- `universe/us_broad.json`: 92 TEST months and 741 stocks.
- Few TEST months give little power: an edge of a few tenths of a percent a month can hide in this noise.
- Not financial advice. Nothing here places orders.
