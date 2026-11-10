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

**What the scan watches** (`scan_tickers`):
- **The research universe:** 1,565 US stocks with prices since 2010 (`us_broad`, `us_small`, `us_micro`). W1 was
  fitted on them, and W1a tested it on them.
- **The newer listings:** 965 companies of SEC ranks 1-3000 that the research lists left out only because they
  listed after 2010 (`universe/us_watch.json`).
  - 392 are large and mid caps (Meta, Tesla, Palantir, CrowdStrike, Uber); 573 are small caps.
  - W1 was not fitted on them. W1b tests its ranges on them, with W1a's gate and years.
  - Until W1b has a result, each line of theirs says so.

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

**W1b: PASS, a little narrow** (2026-11-10). The same model, gate and years on the newer listings, which W1 was
not fitted on: 120,721 tradable stock-weeks of 934 stocks.

| | Aimed for | Held |
|---|---|---|
| The 2-in-3 range | 68% | 65.5% |
| The 9-in-10 range | 90% | 87.9% |

- **Narrow, but inside the declared bands.** It still beat volatility alone: 1.597 against 1.611, better in 29 of
  33 months (p 0.0001, BH q 0.001).
- **For these stocks, a move outside the 9-in-10 range comes about one week in eight,** not one in ten (5.9% below
  it, 6.3% above it).
- **With a report due, the ranges held 60.5% and 84.4%.** Read a newer listing's report week as wider than printed.
- **No direction:** 48.8% of stock-weeks ended above the range's middle.

**W1c: PASS** (2026-11-10). The same model and gate, read at every session's close instead of each week's last,
on both lists: 1,592,265 readings of 2,483 stocks, January 2024 to September 2026.

| | Aimed for | Held |
|---|---|---|
| The 2-in-3 range | 68% | 67.4% |
| The 9-in-10 range | 90% | 89.1% |

- **A scan on any day is as good as a Friday's.** It beat volatility alone: 1.329 against 1.346, better in 28 of
  33 months (p 0.0001, BH q 0.001).
- **By list:** large and mid caps 68.3% and 90.0%; small caps 68.0% and 89.5%; micro caps 68.7% and 89.4%; newer
  listings 65.1%-66.2% and 87.6%-88.1%.
- **With a report due, the ranges held 65.2% and 87.3%.**
- **Readings overlap:** one day's five sessions share four with the next day's, so 1.6 million readings are far
  fewer independent ones. The test counts each month as one reading.
- **No direction:** 49.0% of readings ended above the range's middle.

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
python -m brain scan replay --from 2024-11-04 [--save]   # the scan as it would have read each week since
python -m brain scan fit                 # 2011-2023 only; a refit is a new model and a new test id
```

- Quote a line as printed. A range is a size: never present its middle, or the share of weeks that rose, as a
  call.
- A scan is recorded before any of its five sessions trades, and a recorded row is never changed. A later run
  may add stocks the record lacks until 13:30 UTC on the next weekday (before the US open), listed under `added`
  with their time. `scan score` judges it against the bank's prices: how often each range held, and the 20%+
  days against their chances.
- One week's stocks move together, so a week is one reading. Under 30 judged scans the record says little; the
  tested evidence is W1a and S1a.

## Through the day: a scan at every close, checks in between

The market keeps moving between closes: news after the close, trading before the open, the session itself.
The scan handles that in two ways.

**A scan at every close.**
- `python -m brain scan run --refresh --record` re-reads every stock's prices and filings and records the day's
  scan, every trading day, not only on Fridays.
- W1c tests the ranges read at every session's close (Results).

**Checks between scans.** `python -m brain scan watch [--record]` holds the latest recorded scan against what has
happened since its close:
- **each stock's latest trade** (Yahoo's five-minute chart, trading before the open and after the close included),
  against the close the scan read and the range it recorded;
- **the market:** SPY, QQQ, IWM, S&P 500 futures (which trade through the night) and the VIX.
  - A move of 1.5% or more in SPY or the futures is flagged as market-wide: then every range is likely too
    narrow until the next scan. The replay's worst weeks were market-wide.
  - A VIX up 15% or more is flagged as fear rising;
- **filings since the close** (SEC's per-company filing lists), for every stock outside its range and for each of
  the scan's 25 likeliest 20%+ days: a results report, a share sale, other 8-Ks.

A check forecasts nothing. A flag says where a range is already used up, or where news came in; the next close's
scan re-reads every range from the new facts.

| Command | What it does |
|---|---|
| `scan watch` | one check of every stock in the latest scan (about 20 minutes: Yahoo is read at most twice a second) |
| `scan watch --only-top 300` | the 300 widest ranges and the 300 likeliest 20%+ days only (a few minutes) |
| `scan watch --every 30 --record` | a check every 30 minutes until Ctrl+C, each one written to `research/scans/updates/<close>/` |

`--every` is for a computer that stays on; it waits at least 15 minutes between starts. SEC's live filings feed
sits under `/cgi-bin`, which SEC's robots.txt disallows, so the checks read each company's own filing list.

**The schedule.** Scheduled tasks run in the session that holds the bank, on weekdays, in New York time:

| Time | What |
|---|---|
| 8:07 | a check: overnight news and trading before the open |
| 12:13 | a check: the session so far |
| 20:23 | a scan: prices and filings re-read, the day's scan recorded, scans whose five sessions closed judged; then a check of trading and filings after the close |

Each run is committed (`research/scans/` and `research/scans/updates/`). If the session's machine is reset, its
bank is empty: a check then finds no closes, and the next `scan run --refresh` re-reads the prices.

## The replay: two years as if live

`brain scan replay` (`brain/scanreplay.py`) reads the scan at each week's last close with only what was known then,
and judges each week against what followed. It is descriptive, not a test.

Over the 104 weeks from November 2024 to October 2026 (`research/SCAN-REPLAY-2026-11.md`):

| | Result |
|---|---|
| 2-in-3 range held | 68.8% |
| 9-in-10 range held | 90.0% |
| Weeks the 2-in-3 range held 44%-86% | 9 in 10 |
| The 25 widest ranges moved, either way | 11.8% on average (every stock: 4.1%) |
| Weeks they moved more than the average stock | 104 of 104 |
| S1's 25 likeliest with a 20%+ day | 242 of 2,600 (219 expected) |

- **Ranges broke in market-wide moves:** in the April 2025 tariff crash week, 20% of stocks stayed inside the
  2-in-3 range.
- **As a buy list the top movers were a lottery.** Five of 2,600 picks (SBET, CAPR, QUBT, QMCO, SLNH) carried
  the gains; without the three best weeks it lost after costs.

## The trader replay: every close, the open and midday

`brain scan trader` (`brain/scantrader.py`) replays the same two years the way a trader would work, session by
session, with the frozen W1 model and only what was known at each moment. Its rules were fixed and committed
before it ran, and it runs once:

| When | What the trader does |
|---|---|
| After every close | reads the scan: each stock's range for the next five sessions |
| At the open | checks the market: SPY 1.5% or more below the last close, or the VIX 15% or more above it, cuts the book to half for the day; sells stocks that open through their floor; then trades to the new book |
| At midday (12:30 New York) | the same market check with the midday prices (Yahoo's hourly chart) |
| All day | a standing stop at each stock's 9-in-10 floor (the scan's 5% quantile from the last close); a stock stopped out is not bought for five sessions, and one already through its floor at the open is not bought |
| At the close | marks everything; cash earns the 13-week Treasury bill rate (^IRX) |

- **Books:** the scan's 25 widest ranges (the weekly replay's buy list, a misuse: a range is a size, not a
  direction); every tradable stock; and the S&P 500 (SPY).
- **Each book runs fixed (read once a week, fully invested) and as the trader,** with each step in between,
  so each change shows on its own:

  | Step | Adds |
  |---|---|
  | 0 | the fixed weekly reading (SPY: bought and held) |
  | 1 | the scan read after every close (a mover already held stays while among the 50 widest) |
  | 2 | sizing by risk: each stock's money in proportion to one over its range, and the whole book cut in proportion when its ranges run wider than their median over the last 250 sessions (never borrowing) |
  | 3 | the standing stops |
  | 4 | the market checks at the open and at midday: the trader |

- **Costs, a side:** 0.05% for stocks trading $100 million a day or more, 0.1% from $20 million, 0.2% from
  $5 million, 0.4% below; SPY 0.01%. A stop fills 0.5% under its level, never under the day's low.
- **The first session is November 11, 2024,** the first with Yahoo's hourly prices (they reach back two years).
- **What else it reports:**
  - the ranges read after every close against the week's fixed reading on the same five sessions;
  - how often stocks broke their ranges on the days the market checks fired.

## Limits

- **No direction.** The middle of every range sits near zero. It is a little above zero because the list holds
  only companies that survived to today.
- **Survivorship.** The universe is chosen today: companies bought out or delisted are missing, and so are their
  largest moves.
- **Not the whole market.** The scan watches the research universe and the newer listings of SEC ranks 1-3000.
  - Micro caps that listed after 2010 are not watched, and neither are foreign companies filing 20-Fs.
  - A named ticker outside both lists gets a line marked "outside the scan's lists": W1 and S1 were neither
    fitted nor tested on it.
  - S1's shock chances were fitted and tested on the research universe only.
- **Report dates are estimates:** last year's date moved on 52 weeks.
- **The checks between scans are not forecasts.** They compare the latest trade with the last close's range.
  A price before the open or after the close can come from a few shares, and many small stocks trade little then.
- **A range is not a trade.** Options already charge for the move they expect. Whether a size forecast beats that
  price is unproven (`docs/MOVES.md`).
- **Paper only.** Nothing here places an order.
