# Research results

Generated 2026-10-12T17:41:50Z from `research/ledger.jsonl`. Every number below is copied from the ledger.

Universe: 60 US large caps (top SEC filers by market value with 10-K XBRL and prices since 2010). TRAIN 2011-01-31..2018-12-31, TEST from 2019-01-01. Top third by signal vs the equal-weight universe, monthly, 10 bps per unit turnover.

Gate (declared before running): TEST mean net excess > 0, permutation p < 0.05, and Benjamini–Hochberg q < 0.10 across every hypothesis in the ledger.

| Hypothesis | TEST net excess / mo | Annualised | Info ratio | Months won | Perm p | BH q | TRAIN net excess / mo | TRAIN p | TEST coverage | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1-mom12_1 | +0.11% | +1.3% | 0.12 | 42/92 | 0.1758 | 0.357 | +0.22% | 0.0148 | 100% | **FAIL** |
| H2-rev1 | -0.01% | -0.1% | -0.01 | 49/92 | 0.2142 | 0.357 | +0.05% | 0.0722 | 100% | **FAIL** |
| H3-earnings_yield | -0.23% | -2.7% | -0.31 | 46/92 | 0.8710 | 0.871 | -0.10% | 0.8714 | 85% | **FAIL** |
| H4-low_asset_growth | -0.11% | -1.3% | -0.22 | 48/92 | 0.7363 | 0.871 | -0.09% | 0.7251 | 97% | **FAIL** |
| H5-ridge | +0.19% | +2.3% | 0.27 | 48/92 | 0.0510 | 0.255 | +0.14% | 0.0546 | 100% | **FAIL** |

TEST coverage is the share of listed stock-months with a signal value; the rest are not ranked.

## Controls

- Oracle (next month's actual return as the signal): TEST p = 0.0005, gross excess +8.04%/mo. The harness can see a real edge.
- 200 random signals: 3.0% reached p < 0.05 (expected about 5%). The null is calibrated.
- Controls passed.

## Survivorship bias, measured

Over TEST the equal-weight universe returned +27.4% a year; SPY returned +17.9%. The gap is mostly selection: the universe is today's largest companies, so every member is a survivor and a past winner. Comparing against the same universe cancels the level, not the tilt: survivorship still flatters momentum and growth, and hurts value. A PASS here is an upper bound.

## Ledger notes

- 2026-10-12: The five declarations above were written before the repository had a commit, so their commit field reads HEAD. They were declared against the code committed as a293ac1, which contains this ledger with only those declarations. git_state now records an empty commit when HEAD does not exist.
- 2026-10-12: H3 coverage diagnosis (after the TEST result, so no rerun): earnings yield is missing for 15% of TEST stock-months. Causes: no us-gaap WeightedAverageNumberOfDilutedSharesOutstanding (V, BRK-B; GOOGL, DIS, BLK, LIN, MRVL in earlier years) and NetIncomeLoss no longer tagged in 10-Ks (CAT since FY2010, MA and TMO since FY2013, SCCO, AVGO FY2025). A version that reads more concepts must be declared as a new hypothesis (H3b) and counts in the BH correction.

## Reading this

- FAIL means the claim did not beat a shuffled signal on unseen months after costs. It does not mean the opposite is true.
- 92 TEST months and 60 stocks give little power: an edge of a few tenths of a percent a month can hide in this noise.
- Not financial advice. Nothing here places orders.
