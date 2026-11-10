# The scan as a trader: two years, session by session

Run on 2026-11-10 with `python -m brain scan trader --save` (`brain/scantrader.py`). The rules were committed
in `ba3f6e7` before the replay ran, and it ran once.

- **Model:** the frozen W1 model `b18d8557a496682a`, fitted on 2011-2023.
- **Span:** 500 sessions, 2024-11-11 to 2026-11-09.
- **Stocks:** 2,537 plus SPY.
- **Midday prices:** Yahoo's hourly chart. 2,350 stocks have 450 or more sessions of them.
- **Results:** `research/scans/replays/trader-2024-11-11_2026-11-09.json` holds each book's daily value and
  holdings, and the trade logs.

It is descriptive, not a test. The scan's declared tests are W1a-W1c and S1a.

## What the trader did

| When | What |
|---|---|
| After every close | read the scan: each stock's range for the next five sessions |
| At the open | checked the market, sold stocks that opened through their floor, then traded to the new book |
| The market check | SPY 1.5% or more below the last close, or the VIX 15% or more above it, cut the book to half for the day |
| At midday (12:30 New York) | the same market check, with the midday prices |
| All day | a standing stop at each stock's 9-in-10 floor; a stock stopped out was not bought again for five sessions |
| At the close | marked everything; cash earned the 13-week Treasury bill rate |

Each book ran fixed (read once a week, always fully invested) and as the trader, with each step in between.

## $1 in each book, after costs

| | Book: rules | $1 → | A year | Worst fall | Sharpe | Beta | Money held | Costs paid | Without its 5 best sessions |
|---|---|---|---|---|---|---|---|---|---|
| M0 | 25 movers: fixed weekly | 5.30 | +131.8% | -50% | 1.58 | 2.29 | 100% | 0.39 | 2.84 |
| M1 | + read after every close | 3.92 | +99.1% | -52% | 1.35 | 2.28 | 100% | 0.28 | 2.18 |
| M2 | + sized by risk | 3.45 | +86.6% | -51% | 1.29 | 2.04 | 96% | 0.34 | 2.05 |
| M3 | + standing stops | 3.40 | +85.2% | -51% | 1.29 | 2.05 | 96% | 0.34 | 2.04 |
| M4 | + market checks: **the trader** | 2.83 | +69.0% | -52% | 1.14 | 1.99 | 93% | 0.35 | 1.71 |
| A0 | every tradable stock: fixed weekly | 1.38 | +17.8% | -24% | 0.74 | 0.99 | 100% | 0.01 | 1.12 |
| A1 | + read after every close | 1.39 | +18.1% | -23% | 0.75 | 1.00 | 100% | 0.02 | 1.12 |
| A2 | + sized by risk | 1.15 | +7.4% | -20% | 0.29 | 0.70 | 93% | 0.05 | 1.00 |
| A3 | + standing stops | 1.13 | +6.4% | -19% | 0.23 | 0.66 | 93% | 0.05 | 0.99 |
| A4 | + market checks: **the trader** | 1.10 | +5.1% | -18% | 0.15 | 0.63 | 91% | 0.08 | 0.96 |
| S0 | S&P 500 (SPY): bought and held | 1.28 | +13.4% | -19% | 0.60 | 1.00 | 100% | 0.00 | 1.04 |
| S1 | + sized by risk, read after every close | 1.20 | +9.9% | -16% | 0.47 | 0.80 | 93% | 0.00 | 1.05 |
| S2 | + market checks: **the trader** | 1.21 | +10.3% | -14% | 0.51 | 0.75 | 91% | 0.00 | 1.05 |

- "Costs paid" is in dollars per $1 started.
- **Real funds over the same days** (the close of 2024-11-08 to the close of 2026-11-09):

  | Fund | $1 → |
  |---|---|
  | SPY | 1.29 |
  | Total market (VTI) | 1.28 |
  | Equal-weight S&P 500 (RSP) | 1.24 |
  | Small caps (IWM) | 1.21 |
  | Micro caps (IWC) | 1.34 |

## What it shows

1. **In every book the trader ended with less than the fixed version.**

   | Book | Trader | Fixed |
   |---|---|---|
   | S&P 500 | $1.21 | $1.28 |
   | Every stock | $1.10 | $1.38 |
   | The 25 movers | $2.83 | $5.30 |

2. **Where the book was spread out, its falls were smaller.**
   - S&P 500: the worst fall was 14% against 19%, and the swings 13% a year against 17%.
   - Every stock: the worst fall was 18% against 24%.
   - Not in the 25 movers: 52% against 50%.
3. **It gave up more return than risk.** Return for each unit of risk (Sharpe) fell:
   - S&P 500: 0.51 against 0.60;
   - every stock: 0.15 against 0.74.
4. **Step by step:**
   - **Reading after every close** changed almost nothing when the book held every stock ($1.39 against
     $1.38). For the 25 movers it ended lower ($3.92 against $5.30). That book is a lottery: without its five
     best sessions, the two end at $2.18 and $2.84.
   - **Sizing by risk cost the most.**
     - Every stock: $1.15 against $1.39. One over the range puts the money in the calmest stocks, and in
       these two years the wild ones ran.
     - S&P 500: $1.20 against $1.28. Holding less while ranges were wide meant holding less in the rebounds.
   - **Standing stops** fired 10,095 times in the every-stock book and 68 times among the movers. They cost
     a little: $1.13 against $1.15, and $3.40 against $3.45. The movers' stops sold at a median 21% below
     the buying price.
   - **The market checks** fired on 27 sessions: 12 at the open and 15 at midday.
     - On the S&P 500 they gained a cent ($1.21 against $1.20) and trimmed the worst fall from 16% to 14%.
     - On the stock books they cost: $1.10 against $1.13, and $2.83 against $3.40.
     - **The cuts tended to sell just before a rebound.** Over the five sessions after a cut, SPY rose
       0.69% on average, against 0.28% after any session. After the cuts of April 4, 7 and 21, 2025, it rose
       5.7%, 6.9% and 7.2%.
5. **Ranges read after every close were barely better than the week's fixed reading.**

   | Monday to Thursday (922,882 readings) | Read every close | Fixed weekly |
   |---|---|---|
   | 2-in-3 range held | 67.9% | 67.5% |
   | 9-in-10 range held | 89.4% | 88.9% |
   | Pinball loss | 0.0137 | 0.0138 |

   - After a 2% market day or a 15% VIX jump since the fixed reading (94,827 readings), both ran wide: the
     2-in-3 range held 74.1% (read every close) and 72.0% (fixed weekly). The loss was 0.0131 against 0.0132.
     In a frightened market the ranges are too wide (`docs/SCAN.md`, Results), and a fresh reading widens
     them further.
6. **The checks say something about size, not direction.**
   - **Market checks:** on midday-flag sessions, 14.6% of stocks ended the five sessions outside their
     9-in-10 range. On open-flag sessions it was 11.4%, and on the other sessions 10.5%.
   - **Stocks already outside their 2-in-3 range at the open** ended the five sessions outside their 9-in-10
     range 45.0% of the time (11,127 readings), against 10.7% for every stock. That is what `scan watch`
     flags live: where a range is already used up, not which way the stock goes next.

## Quarter by quarter

| Quarter | SPY held | SPY trader | Every stock, fixed | Every stock, trader | Movers, fixed | Movers, trader |
|---|---|---|---|---|---|---|
| 2024 Q4 (from Nov 11) | -2.0% | -1.8% | -3.8% | -5.2% | +47.3% | +35.7% |
| 2025 Q1 | -4.3% | -3.6% | -6.9% | -5.7% | -26.1% | -26.5% |
| 2025 Q2 | +10.8% | +8.6% | +10.3% | +5.8% | +79.4% | +47.3% |
| 2025 Q3 | +8.1% | +7.3% | +11.8% | +6.7% | +74.8% | +83.9% |
| 2025 Q4 | +2.7% | +1.7% | +3.7% | +0.4% | -6.3% | -13.7% |
| 2026 Q1 | -4.4% | -5.0% | +1.8% | -1.3% | +50.0% | +39.2% |
| 2026 Q2 | +15.1% | +13.3% | +18.9% | +12.1% | +77.9% | +44.6% |
| 2026 Q3 | +2.4% | +2.0% | -3.2% | -4.6% | -24.6% | -21.8% |
| 2026 Q4 (to Nov 9) | -1.3% | -1.4% | +3.1% | +3.3% | -17.7% | -22.7% |

The trader lost less than the S&P 500 in two of its four falling quarters. It gained less in every rising one.

## The movers' log

| | The trader (M4) | Fixed (M0) |
|---|---|---|
| Closed positions | 592 | 837 |
| Median closed position | -1.1% | -0.7% |
| Mean closed position | +10.3% | +7.5% |
| Positions that rose | 46% | 49% |

- **Most positions lost money; a few carried the book.**
  - The trader's best: TNGX +717%, BW +555%, AGL +553%, UAMY +525%.
  - Fixed's best: BW +2,912%, AXTI +1,949%, TNGX +639%, RXT +549%, RGTI +476%, QBTS +354%.
- **Worst:** QMCO -88% and -95%, USBC -81%, SANA -71%, EYPT -87%.
- **These were real runs, not split errors.** The winners' price histories have no session beyond the bank's
  data-error rule (+500% or -95% in a day).

## Limits

- **The lists flatter the stock books.**
  - They were chosen in 2026, the newer listings by SEC rank, which tracks size. So they hold stocks that
    later soared, and miss those that collapsed and were delisted.
  - Two-thirds of the movers bought (66% fixed, 64% trader) came from the newer listings.
  - The every-stock book's $1.38 beat every real fund above.
  - Compare fixed against trader inside a book. The S&P 500 book has no picking at all and is the fairest.
- **One path of one market.** These two years held one crash (April 2025) and quick rebounds, a market that
  punishes cutting after a fall. A slow fall, like 2008 or 2022, would treat the cuts differently, and this
  replay cannot say how.
- **Fills are estimates:**
  - stops come from the day's low, and when in the day the low came is not known past midday;
  - the midday price is the last hourly bar by 12:30;
  - costs are estimated by dollar volume;
  - no shorting, no borrowing, and cash earns the bill rate.
- **The rules were set once, before the run, and not tuned.** Other thresholds would give other numbers.
  Trying many and keeping the best would only fit this one path.
- **Paper only.** Nothing here placed an order.
