# Learning from smart money's wins: what was visible before, and did it predict?

Written Nov 1, 2026. **Exploratory**, 2014-2023 only; 2024 onward untouched. Follows
`HISTORY-SMART-MONEY-2026-11.md`, which found that copying these players after disclosure did not pay. The
question here: when they won, what could the public see beforehand, and does that pattern pick winners in general?

## Method

Start where the winners provably had skill. Activist-style 13D filers did: their targets jump about 5% when the
stake becomes public. Politicians' records were indistinguishable from luck, so their "wins" have nothing to teach.
Then:

1. **Rebuild the public picture.** For each of the 1,572 research-universe companies (`universe/us_*.json`), at each
   month-end from Dec 2014 to Aug 2023: prices and volume up to that day, and fundamentals as filed by that day
   (the bank's `known_at`), none older than 18 months.
2. **Profile the winners' picks against everyone else.** Each feature becomes a percentile within its month (50 =
   average). The profile is the targets' average in the month before their initial 13D: 386 by fund-like filers,
   74 by the well-known activists of the history note, 702 by anyone, none with a 13D in the year before.
3. **Learn on old years, check on newer ones.** A logistic model on the percentiles was trained on 2015-2019 to
   predict a 13D in the next three months, then tested on 2020-2023.
4. **Would it have paid?** Each month, an equal-weight basket of the top 10% of scores, against the average
   universe stock.

## What the targets looked like beforehand

| Feature, percentile among all companies that month | Fund-like filers (386) | Well-known activists (74) |
|---|---|---|
| 12-month return | 35 (z -10.3) | **28** (z -6.7) |
| 2-year return | 34 (z -10.8) | 32 (z -5.5) |
| Distance below the 52-week high | 34 (z -11.2) | 33 (z -4.9) |
| Daily volatility, 3 months | **67** (z +11.3) | 63 (z +4.0) |
| Volume, last month vs the 6 before | 54 (z +2.8) | **65** (z +4.3) |
| Trading value (liquidity) | 43 (z -5.1) | 60 (z +3.0) |
| Market value | 36 (z -8.9) | 53 |
| Return on assets | 38 (z -7.4) | 48 |
| Book value / market value | 56 (z +4.0) | 54 |

Targets were beaten-down, jumpy stocks well below their highs. The well-known activists' targets also show a volume
surge in the month before the filing: the activist's own buying, which must be disclosed within 10 days of
crossing 5%. Fund-like filers also lean to smaller, less profitable companies. Cash, debt, sales growth and last
month's return set targets apart from other companies little or not at all.

## Could it be learned?

| Trained 2015-2019, tested 2020-2023 | AUC | Top 10% of scores held | Became a target within 3 months |
|---|---|---|---|
| Fund-like filers (685 / 422 events) | 0.70 | 29% of the targets | 1.9% of top-10% stocks vs 0.6% of all |
| Well-known activists (124 / 84) | 0.70 | 31% | 0.4% vs 0.1% |
| Any 13D (1,259 / 745) | 0.66 | 23% | 2.6% vs 1.1% |

The profile predicts targets about three times better than chance. Even so, about 98% of the stocks it flags are
never targeted: the event is rare.

## Did it pay? No. One year made the backtest look good.

The top-10% basket beat the average stock by +2.15% a month in 2020-2023 (t +2.08). Broken down:

| Period | Excess return a month |
|---|---|
| 2015-2019 (the model saw targets, never returns) | +0.46% (t +0.87) |
| 2020 | **+6.71%** (t +2.35): beaten-down stocks rebounding after the March crash |
| 2021 / 2022 / 2023 | +0.14% / +0.78% / +0.35%, none significant |
| 2020-2023 without the stocks that actually got a 13D | +2.06% |

None of the profit came from activists. It came from one rebound year, and the universe makes it worse: every member
is still listed in 2026, so beaten-down companies that later failed are missing, and buying beaten-down stocks looks
better than it was. One trait at a time (most volatile, worst year, smallest, furthest below high) earned -0.4% to
+0.2% a month over 2015-2023.

## Footprints of big buyers: volume surges

Each month's stocks were split into ten groups by last month's volume against the six months before.

| | Next month vs the average stock |
|---|---|
| Biggest surge (top 10%) | +0.52% (t +2.20) |
| Middle | -0.30% (t -3.12) |
| Biggest drop (bottom 10%) | +0.38% (t +2.17) |
| A surge with an ordinary price month (2,880 cases) | +0.46% (t +1.93); 13D within 3 months 1.6% vs 1.6% for all |

Unusual volume either way was followed by slightly better months than ordinary volume, which is not a direction
signal. With this many looks, |t| near 2 is expected by chance. Quiet accumulation did not foretell activists.

## The famous case: the market before the February 2020 crash

From Jan 17 to Feb 19, 2020, SPY rose 1.9% to a record. Over the same weeks China (FXI) fell 5.9%, oil 9.0%, copper
8.3%, energy stocks 7.2% and airlines 4.4%, while long Treasuries rose 5.6% and gold 3.6%. SPY then fell 33.7% by
Mar 23. The warning was public.

Tested as a rule over 2005-2023, it gives no edge. The rule: China -8%, copper -5% and long Treasuries +3% over a
month, with SPY within 3% of its high. It fired 5 times, counting once per three months:

| Date | SPY over the next 63 sessions | Worst drop within them |
|---|---|---|
| 2013-04-15 | +9.6% | -0.6% |
| 2014-01-30 | +6.0% | -2.8% |
| 2015-07-27 | -0.4% | -9.4% |
| 2020-02-03 | -8.4% | -30.8% |
| 2021-07-08 | +6.7% | -1.4% |

Two warnings were right and three were false alarms. On all days with SPY within 3% of its high, a 10% drop
followed within 63 sessions 12% of the time.

## What to rely on, from everything tested so far

- **Rely on size, not direction.** What options price for a move is about right (`CALLS-REVIEW-2026-11.md`), and
  the forecaster's 68% and 90% bands held (F1).
- **News is priced fast.** A 13D moves its target about 5% within two days, most of it before the next open.
  Being first is the edge; what is left after that is small and unreliable.
- **Hindsight always finds a story.** Every famous win has visible precursors. The precursors here were real
  (targets do look different; the 2020 warning was there) but rare, shared by many non-events, and unprofitable
  when tested across all stocks and years.
- **No direction signal has survived a test here yet.** H1-H8, L1, E1/E2, politicians, activists, these profiles.

## Biases

- **Survivorship:** members listed 2010-2026 only, which inflates rebounds.
- **Hindsight:** the activist list is picked by today's fame.
- **Many looks:** about 60 statistics here, so expect several |t| near 2 by chance.
- **Missing data:** no news history (2015-2023 news per company is not in the bank) and no options history.
