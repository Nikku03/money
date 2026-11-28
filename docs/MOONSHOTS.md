# The moonshot watch: where the next big jumps come from

Asked for on 2026-11-28: "i don't need just the revenue guess, i want to predict every moonshots before they
happen or atleast most of them."

A **moonshot** here is a session in which a stock beats SPY by 20% or more; a **crash** is one 20% or more below.

```
python -m brain moonshots list [--top 25]   # the latest recorded scan's likeliest names, and what they are
python -m brain moonshots score             # every judged recorded scan: jumps caught, crashes, paper return
```

## What can be said, and what cannot

The list is the scan's own: S1's chance of a 20%+ session either way in the next five (`docs/SHOCKS.md`; S1a PASS
on 2024-2026: each week's top 1%, 5% and 10% held 22%, 52% and 68% of all such sessions, and 68% of them were up).
Nothing new is fitted. Three things were measured before building it, on weeks of 2011-2023 only (1,058,157
stock-weeks of 1,561 stocks; S1 refitted on 2011-2019 and read on 2020-2023, which it had not seen):

**1. Where the jumps come from can be said** (`research/moonshots/explore_train.py`).

| Each week's list, 2020-2023 | Share of all jumps | A jump that week | A crash that week |
|---|---|---|---|
| top 1% of stocks | 25% | 15.0% of its names | 9.7% |
| top 5% | 53% | 6.1% | 3.0% |
| top 10% | 68% | 3.9% | 1.9% |
| top 20% | 82% | 2.4% | 1.2% |

Of all jumps of 2011-2023, 44% came from stocks under $5 (6% of stock-weeks), 15% from drug and biotech
companies, 9% from a week with a results report due, and 41% from none of the three.

**2. Which way cannot.** A model fitted to jumps alone ranked them no better than S1 (54% of jumps in its top 5%
against 53%; the same in every year). Rankings built to keep jumps and drop crashes (the jump chance less the
crash chance, or over the any-shock chance) lost more jumps than crashes: 44% and 34% of jumps in the top 5%.
What makes a jump likely makes a crash likely; every list that holds the jumps holds about one crash for two
jumps. No jump model was declared: it would add nothing to S1.

**3. Buying the list looked good on paper, and the paper cannot be trusted**
(`research/moonshots/explore_buy_train.py`).

| The top 5% each week, equal parts, five sessions against SPY | 2011-2019 | 2020-2023 |
|---|---|---|
| mean | +0.77% (t +5.8) | +1.14% (t +3.6) |
| median pick | -0.70% | -0.82% |
| picks that rose | 45% | 46% |
| weeks above zero | 59% | 57% |
| worst week, best week | -7.5%, +13.0% | -13.7%, +18.8% |
| the weeks with a jump, with a crash, the rest | +26.6%, -30.4%, -0.50% | +32.3%, -28.5%, -0.31% |

The usual pick lost; a few jumps paid for many small losses, in every year from 2011 to 2023. Two things flatter
that mean, and neither can be removed from old data:

- **The dead are missing.** The bank holds only companies still listed today. A $1 stock of 2015 that is here
  survived; those that went to zero are not in the rows. On lists of the riskiest stocks that is a large bias.
- **These stocks are costly to trade.** Nearly half the jumps are in stocks under $5, where the gap between the
  buying and the selling price can be several percent a round trip, more than the mean itself.

So the paper figure is a lead, not a finding, and it is not presented as an edge.

## The scoreboard

`moonshots score` uses recorded scans only (`research/scans/`): each list was written before its five sessions
traded, whatever became of its stocks afterwards, so the dead are not missing. For each judged scan it gives the
jumps and crashes among all stocks, and for the top 25, 5%, 10% and 20% by chance: the jumps and crashes held,
and the mean and median return of the list against SPY, bought in equal parts.

- The first nine judged scans (2026-11-09 to 2026-11-19, about two weeks of a falling market for speculative
  stocks): the top 20% held 49 of 62 jumps (79%) and 35 of 52 crashes, and the top 5% returned a mean -6.2% a
  reading against SPY (median pick -7.7%). The lists caught the jumps and lost money.
- **The reading, fixed on 2026-11-28:** week-end scans from 2026-12-04 on (one week is one reading, since a
  week's stocks move together); the list is the top 5%; the value is its mean five-session return against SPY.
  Under 30 weekly readings the record says nothing, either way. When it has 30, the question "does buying the
  list pay" is declared as a test and judged once.
- Paper returns, before the cost of trading.

## What it is for

A list says where to look. Most of its names will not jump that week (about 1 in 16 of the top 5%), about a third
of the big moves on it are crashes, and nothing in this repo says which. It is not a direction and not a trade.
What it gives the person is a short map of where the week's moonshots will most likely come from, to be read with
the rest: a stock's dates (`brain file`), what it needs (`brain mosaic`), its report (`brain expect`).

What could add to it, none of it built: dated events with a known day (FDA decisions, trial results), which the
bank does not hold yet; option prices and short interest, which need a paid data source.
