# Catalysts and surprise moves: what came before them, and what helps predict them

Written Nov 11, 2026. **Exploration**: it read 2011-2023 only. It was finished, and the test S3a declared, before
any 2024-2026 data was read for it. The exploration on its own counts as no evidence.

## The question

The user asked to predict catalysts and surprise moves. S1 (`docs/SHOCKS.md`) already ranks every stock by its
chance of a 20%+ session against SPY in the next five, and passed S1a. This exploration asks two things:
- **what came with each surprise:** a results report, another SEC filing, or nothing at all;
- **whether knowing the catalyst helps S1:** whether it ranks the surprises better.

## The data

- **Stocks:** the research universe, 1,572 US stocks (`us_broad`, `us_small`, `us_micro`).
- **Units:** S1's rows, one stock at one week's last close: 1,060,191 stock-weeks in 2011-2023.
- **Surprise sessions:** 9,669 sessions that beat or trailed SPY by 20% or more.
- **Protocol:** fit 2011-2019 (733,442 stock-weeks); judge 2020-2023 (326,749).
- **Catalysts:** from each company's SEC filings, each counted from the session at whose close it was first known
  (`events.event_index`).

  | Catalyst | What it is |
  |---|---|
  | Results report | 8-K item 2.02 |
  | Deal | 8-K items 1.01, 2.01, 5.01 |
  | Distress | 8-K items 1.03, 2.04, 2.05, 2.06, 3.01, 4.01, 4.02, and late-filing notices (NT 10-K, NT 10-Q) |
  | Executive change | 8-K item 5.02 |
  | Press release | 8-K items 7.01 and 8.01 |
  | Share sale | 424B, S-1, S-3, F-1, F-3 |
  | Merger filing | DEFM14A, PREM14A, SC 14D9, SC TO-T and 425 |
  | Activist stake | SC 13D |
  | Reverse split | the bank's split facts |
  | Lockup expiry | 180 days after a listing's 424B4 or 424B1 |
  | Age | how long the stock has been listed |

## What came with each surprise

| What was filed that session | 2011-2019 | Up | 2020-2023 | Up |
|---|---|---|---|---|
| **Nothing at all** | **78.9%** | 63% | **72.8%** | 69% |
| Results report | 12.1% | 50% | 16.6% | 58% |
| Press release (8-K 7.01, 8.01) | 3.9% | 53% | 4.9% | 56% |
| Deal (8-K 1.01, 2.01, 5.01) | 2.1% | 68% | 2.3% | 77% |
| Share sale | 1.1% | 26% | 1.5% | 31% |
| Executive change | 0.6% | 60% | 0.6% | 68% |
| Distress filing or late-filing notice | 0.7% | 47% | 0.6% | 59% |
| Merger filing, other 8-K | 0.6% | | 0.9% | |
| **Sessions** | **6,623** | 60% | **3,046** | 66% |

- **Most surprises come with nothing filed.** About three in four had no SEC filing that session. The news came
  through press wires, social media or sector moves, or there was none at all. 40-42% were in stocks under $5.
- **Results reports are the one big catalyst known in advance**, at 12-17% of surprises.
- **Up and down are not predictable from the catalyst.** Share sales are mostly down; everything else leans up,
  as surprises do overall.

## Did the catalysts help S1? Mostly not

Fitted on 2011-2019 and judged on 2020-2023, S1 plus twelve catalyst features:

| | Log loss | AUC | Top 5% caught | Better than S1 in |
|---|---|---|---|---|
| S1 | 0.03390 | 0.8954 | 50.2% | |
| S1 + every catalyst above | 0.03390 | 0.8967 | 50.3% | 18 of 48 months (p 0.46) |

- **Most catalysts carried no weight.** Deals, distress filings, executive changes, merger filings, activist stakes,
  reverse splits and lockups each moved the loss by less than 0.00001 when dropped.
- **Press releases hurt out of sample.**
- **Only one helped:** a report expected a quarter after the last one.

## Where S1 is weakest: report weeks

In 2020-2023, S1 had each kind of surprise in its week's top 5% this often:

| Kind | Sessions | In S1's top 5% |
|---|---|---|
| Nothing filed | 2,193 | 57% |
| Another filing | 324 | 49% |
| **Results report** | **504** | **34%** |

S1 sees cheap, jumpy stocks well. It misses earnings surprises in stocks that are calm between reports. What helps
there is each stock's own **earnings habit**: the median move against SPY on its last eight report sessions.

| 2020-2023 report weeks: the stock's habit | Stock-weeks | A 20%+ session | S1 said | S1 + habit said |
|---|---|---|---|---|
| Below the median | 16,290 | 0.50% | 0.90% | 0.74% |
| Median to 80th percentile | 9,774 | 1.23% | 1.64% | 1.70% |
| 80th to 95th | 4,886 | 3.38% | 2.84% | 3.90% |
| Top 5% | 1,630 | 6.13% | 4.94% | 9.96% |

**S3: S1 plus a report expected a quarter after the last one, plus the earnings habit.**

| 2020-2023, fit on 2011-2019 | Log loss | AUC | Top 5% caught | Earnings surprises in top 5% | Better than S1 in |
|---|---|---|---|---|---|
| S1 | 0.033901 | 0.8954 | 50.2% | 34% | |
| S1 + a report a quarter on | 0.033859 | 0.8960 | 50.2% | 36% | 27 of 48 months (p 0.30) |
| **S3** | **0.033789** | **0.8984** | **51.0%** | **39%** | 31 of 48 months (p 0.094) |

- **In report weeks only** (32,602 stock-weeks, 469 surprises), S3 beat S1 in 30 of 48 months (p 0.012). Its loss
  was 0.06652 against 0.06762.
- **Elsewhere it changed nothing:** the loss was 0.03016 for both.
- **S3 ran high in report weeks:** it said 1.97% against 1.44% actual (S1 said 1.62%). Refitted on 2011-2023, as
  declared, it may run closer.

## Declared

**S3a:** in the weeks a stock's report is expected, S3's chance of a 20%+ session beats S1's.
- **Test period:** 2024-01 to 2026-09, research universe.
- **Score:** monthly log loss, one-sided sign-flip p < 0.05.
- **Calibration:** S3's chances must match the actual rate within 0.67-1.5, overall and in each week's top 5%.
- **Controls** and the false-discovery correction as for S1a.
- **Code:** `brain/catalysts.py` and `brain/research/catalysttest.py`.

## Limits

- **Survivorship**, as for S1: the universe is chosen today, so takeovers and collapses are undercounted.
- **SEC filings are only part of the news.** Press wires and social media, where three in four surprises start, are
  not in the bank's history.
- **FDA decisions, trial readouts and medical meetings are not in it either.** No point-in-time history of them is
  available here; `research/PHARMA-CATALYSTS-2026-11.md` lists the known ones by hand.
- **A report's date is estimated,** from last year's or last quarter's date, not the announced one.
- **A size is not a direction.** Nothing here trades.
