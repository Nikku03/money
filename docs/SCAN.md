# The market scan: how far each stock is likely to move, and why

`python -m brain scan run` reads every stock the research universe holds and prints, for each, the range its price
is likely to stay in over the next five sessions, its chance of a 20%+ day, and what makes the range wide.

It does not say which way a stock will go. Nothing in this repo can, and the scan says so on every run.

## Why size and not direction

Every declared test in the repo is one of two kinds.

| Question | Tests | Passed |
|---|---|---|
| Which way will it go? | 36 | 0 |
| How far will it move? | 3 | 3 |

- **Which way** (all FAIL):
  - nine signals from prices, accounts, insider buying and earnings (H1-H8);
  - "quiet changes" in filings (H10);
  - the forecaster's chance to beat SPY (F1b, F1c);
  - five cause-and-effect links and five earnings drifts;
  - the fade after a jump (S2a, S2b);
  - twelve verdicts on the paper trader (P1-P13).
- **How far** (all PASS): earnings moves (M1a), 20%+ days (S1a) and the weekly range below (W1a). The
  forecaster's price bands also held their odds, reported beside F1b and F1c.

H10 is on the `claude/eloquent-cori-9mojj5` branch and the paper trader on `claude/sleepy-goldberg-2ma18z`; the
rest are in `research/RESULTS.md`.

So the scan reports what has been shown to be predictable: the size of the move, and where the large ones are
likely to land.

## What a line says

```
  CATX   2 in 3: -27.0%..+34.0%   9 in 10: -45.0%..+80.0%  20%+ day 47.0%   why: results due within eight days; ...
```

- **2 in 3, 9 in 10:** the ranges the stock's price change over the next five sessions should fall in that often
  (W1, below). The range is not symmetric: a wild stock can rise further than it can fall.
- **20%+ day:** the chance that one of those sessions moves 20% or more against SPY, either way (S1,
  `docs/SHOCKS.md`).
- **why:** the inputs that widen this stock's range most, in its own numbers: how much it moves a day, a report
  due, a low price or thin trading, a recent run or fall, a share-sale filing, a drug company, a fearful market.

Stocks are listed widest range first. A stock the scan cannot read honestly is left out and counted with the
reason.

## W1: the weekly range

A row is one stock at one close. Everything in it was known at that close.

- **Inputs:** S1's 28 (volatility over 20, 60 and 250 sessions, the largest moves of the year, recent moves and
  returns, price, dollar volume, distance from the 52-week high and low, recent 8-Ks, a share-sale filing, a report
  expected within eight days, drug or biotech, the VIX), plus two:
  - `lv5`, the stock's variance over its last five sessions;
  - `fearvol`, the VIX times the stock's 250-session volatility. Market fear widens a calm stock's range more than
    a wild one's.
- **The size:** least squares on the log of the next five sessions' squared daily returns. Its square root,
  `sigma`, is the size of an ordinary week for that stock.
- **The shape:** the five-session return over `sigma`, as it was distributed in the training years. It is kept
  apart for each tenth of `sigma`, and within each tenth for weeks with a report due (a group with fewer than 500
  training rows takes its tenth's shape).
- **The baseline it must beat:** the same method on the 60-session volatility alone.
- **Fitted once** on the research universe's tradable stock-weeks of 2011-2023 (`knowledge/scan_w1.json`).

**Tradable** means, at that close: a bar that day, trades on each of the last 20 sessions, at least $500,000 a day
over them (the paper trader's rule, P3b), and no data error in the last 250 sessions (a day beyond +500% or -95%,
or a close under $0.05). The same rule decides which rows the model learns from, which it is tested on, and which
stocks the scan lists.

## The declared test, W1a

`brain/research/rangetest.py`, declared before it ran. TEST is every tradable research-universe stock-week ending
2024-01-01 to 2026-09-30, which the model was not fitted on.

- Per stock-week the score is the pinball loss of the return over the 5%, 16%, 50%, 84% and 95% points: lower is
  better, and it rewards a range that is both right and tight. Per month, the baseline's mean loss minus W1's. The
  statistic is the mean of the monthly differences, with a one-sided sign-flip p.
- **It passes when:**
  - the difference is positive with p < 0.05;
  - the 2-in-3 range holds 63%-73% of TEST stock-weeks, and the 9-in-10 range 86%-94%;
  - the controls pass: W1's ranges shuffled among each week's stocks reach p < 0.05 in at most 10% of draws, and a
    range that knows the outcome reaches p < 0.01;
  - Benjamini-Hochberg q < 0.10 across every gated test in the ledger.
- The middle of the range is reported, not tested: W1a is not a direction test.

**Dry runs on years W1 had not seen, inside 2011-2023** (exploration; no evidence on their own):

| Fitted on | Checked on | 2-in-3 held | 9-in-10 held | Loss against the baseline's | Months better | p |
|---|---|---|---|---|---|---|
| 2011-2015 | 2016-2019 | 67.9% | 89.2% | 0.930 against 0.950 | 45 of 48 | 0.0001 |
| 2011-2019 | 2020-2023 | 70.4% | 91.3% | 1.295 against 1.325 | 32 of 48 | 0.0041 |

- In 2020-2023 the ranges of the calmest tenth of stocks were too narrow (the 2-in-3 range held 59%) and those of
  wilder stocks a little wide (72%). In 2016-2019 every tenth held 66%-69%.
- With a report due, the 2-in-3 range held 65% in 2016-2019 and 73% in 2020-2023.
- The range's middle said nothing about direction: 51.4% and 48.3% of stock-weeks ended above it.

## Results

**W1a: PASS** (2026-11-10, `research/RESULTS.md`). Run once, as declared: 209,939 tradable stock-weeks of 1,550
stocks, January 2024 to September 2026.

| | Aimed for | Held |
|---|---|---|
| The 2-in-3 range | 68% | 68.1% |
| The 9-in-10 range | 90% | 89.7% |

- **Every kind of stock:**
  - large and mid caps 68.1% and 89.9%; small caps 67.9% and 89.5%; micro caps 68.2% and 89.3%;
  - every tenth of size, calm to wild: 66%-70% and 89%-91%.
- **Tighter than volatility alone:** a pinball loss of 1.148 against 1.164, better in 30 of 33 months (p 0.0001,
  BH q 0.001). The 60-session volatility's own ranges were too wide (70.4% and 91.6%).
- **It ranks the movers:** within a week, the rank correlation between W1's size and the actual move was 0.347
  (volatility alone: 0.300).
- **Report weeks ran a little narrow:** with a report due the ranges held 64.7% and 87.1%.
- **Tails were even:** 5.0% of moves fell below the 9-in-10 range and 5.3% rose above it.
- **No direction, as expected:** 51.9% of stock-weeks rose, and 49.1% ended above the range's middle.
- **Controls passed:** shuffled ranges never reached p < 0.05 in 200 draws.

**Looked at after the fact** (the same TEST stock-weeks cut other ways; descriptive, not verdicts):

| Cut | 2-in-3 held | 9-in-10 held |
|---|---|---|
| VIX under 15 | 66.4% | 89.0% |
| VIX 15-20 | 67.7% | 89.7% |
| VIX 20-25 | 69.1% | 89.1% |
| VIX 25-30 | 74.6% | 94.4% |
| VIX 30 or more | 82.9% | 97.2% |
| After the stock's calmest weeks (a tenth of them) | 65.8% | 87.4% |
| After its wildest weeks (one in twenty) | 73.8% | 92.1% |
| S1's chance of a 20%+ day at 10% or more (597 stock-weeks) | 63.8% | 86.3% |

- **In a frightened market the ranges are too wide.** Fear fades faster than W1 allows.
- **After a wild week they are a little wide, after a calm one a little narrow.**
- **For the likeliest shock names they are a little narrow.**
- **S1's highest chances held up** on its own TEST set: where it said 20%-30%, a 20%+ day came 22.5% of the time
  (240 stock-weeks); 30%-50%, 38.0% (208); 50% or more, 55.6% (63).

## Using it

```
python -m brain scan run                 # the whole research universe at the bank's latest close, widest first
python -m brain scan run --refresh       # re-read daily prices and latest filings first (about 20 minutes)
python -m brain scan run --sort shock    # likeliest 20%+ day first
python -m brain scan run CATX AAPL       # named tickers, in full (they need prices in the bank)
python -m brain scan run TSLA --refresh  # a ticker outside the research universe: read, and marked as untested
python -m brain scan run --record        # also write research/scans/<close>.json, so the scan can be scored
python -m brain scan score               # every recorded scan whose five sessions have closed
python -m brain scan fit                 # 2011-2023 only; a refit is a new model and a new test id
```

- Quote a line as printed. A range is a size: never present its middle, or the share of weeks that rose, as a
  call.
- A scan is recorded before any of its five sessions trades, once, and never rewritten. `scan score` judges it
  against the bank's prices: how often each range held, and the 20%+ days against their chances.
- One week's stocks move together, so a week is one reading. Under 30 judged scans the record says little; the
  tested evidence is W1a and S1a.

## Limits

- **No direction.** The middle of every range sits near zero. It is a little above zero because the list holds
  only companies that survived to today.
- **Survivorship.** The universe is chosen today: companies bought out or delisted are missing, and so are their
  largest moves.
- **Not the whole market.** The research universe is 1,565 US stocks with prices since 2010. A company listed
  after that is not in it, and those are often the wildest. A named ticker outside it (Tesla, Meta, Coinbase) gets
  a line marked "outside the tested list": W1 and S1 were neither fitted nor tested on it.
- **Report dates are estimates:** last year's date moved on 52 weeks.
- **A range is not a trade.** Options already charge for the move they expect. Whether a size forecast beats that
  price is unproven (`docs/MOVES.md`).
- **Paper only.** Nothing here places an order.
