# The market scan, replayed week by week: November 2024 to October 2026

Written Nov 10, 2026. Descriptive: the declared tests of these models are W1a and S1a (`research/RESULTS.md`).
A replay adds no verdict.

## What was replayed

- **The scan as it would have run live:** at each week's last close, from Nov 8, 2024 to Oct 30, 2026 (104
  weeks), using the frozen W1 (weekly range) and S1 (shock radar) models. Both were fitted on 2011-2023, so every
  week here is out of sample.
- **Point in time:** each reading uses only prices to that close and filings public by then. The stocks listed
  are the research universe's tradable ones that week, about 1,485 a week, 154,397 stock-weeks in all.
- **Judged against what followed:** the next five sessions for the ranges and the 20%+ days; for the buy-list
  illustration, the next week's reading (four sessions in a holiday week), so weeks never overlap.
- **Rerun:** `python -m brain scan replay --from 2024-11-04 --save` writes
  `research/scans/replays/replay-2024-11-04_2026-12-31.json`, every week's record.

The first 99 weeks fall inside W1a's TEST window (January 2024 to September 2026). The last five (October 2026)
are new.

## Results

**Ranges: honest over two years, broken in market crashes.**

| | Aimed for | Held |
|---|---|---|
| The 2-in-3 range | 68% | 68.8% |
| The 9-in-10 range | 90% | 90.0% |

- **Tails were even:** 5.2% of stock-weeks fell below the 9-in-10 floor and 4.8% rose above its ceiling.
- **Week to week it swings:** in nine weeks of ten the 2-in-3 range held between 44% and 86%.
  - In the 76 weeks SPY moved less than 2% either way, it held 70.9%.
  - In the other 28 it held 63.0%.
- **Its worst weeks were market-wide moves.** Every stock moved together, so most left their ranges at once:

  | Week read | 2-in-3 range held | SPY over the next five sessions |
  |---|---|---|
  | Mar 28, 2025 (the April tariff crash) | 20% | -9.1% |
  | Dec 13, 2024 | 34% | -1.8% |
  | Oct 3, 2025 | 41% | -2.4% |
  | Jan 30, 2026 | 43% | -0.2% |
  | Jul 25, 2025 | 43% | -2.4% |

**Movers: the predicted big movers moved about three times as far, every week.**
- Each week's 25 widest ranges moved 11.8% on average, either way. The average listed stock moved 4.1%.
- They moved more in all 104 weeks.
- 42% of them moved 10% or more in the week, against 8% of all stocks.

**Shocks: the radar's picks had 20%+ days about fifteen times as often as the average stock.**
- S1's 25 likeliest each week: 242 of 2,600 picks (9.3%) had a 20%+ day against SPY. The average stock-week:
  0.64%.
- S1 expected 219 among its picks; across every stock-week it expected 781, and 992 came. Its chances ran about
  a fifth low, as S1a found.

**Direction: none, as designed.**
- The 25 widest ranges rose in 46.6% of their weeks; every listed stock in 51.6%.
- 48.3% of all stock-weeks ended above the middle of their range.

**By quarter:**

| Quarter | Weeks | 2-in-3 held | 9-in-10 held | 25 widest moved | Average stock moved | 20%+ days among picks (expected) |
|---|---|---|---|---|---|---|
| 2024 Q4 | 8 | 68.2% | 89.3% | 15.0% | 3.6% | 25 (17.0) |
| 2025 Q1 | 13 | 63.3% | 85.4% | 12.7% | 4.4% | 38 (34.1) |
| 2025 Q2 | 13 | 75.1% | 94.0% | 14.7% | 4.1% | 35 (36.8) |
| 2025 Q3 | 13 | 71.0% | 91.6% | 12.9% | 3.7% | 33 (26.6) |
| 2025 Q4 | 13 | 71.6% | 91.6% | 11.2% | 3.8% | 32 (25.6) |
| 2026 Q1 | 13 | 65.5% | 87.7% | 9.5% | 4.5% | 20 (24.7) |
| 2026 Q2 | 13 | 62.5% | 87.6% | 11.3% | 4.7% | 31 (21.4) |
| 2026 Q3 | 13 | 73.0% | 92.8% | 8.4% | 3.6% | 20 (22.1) |
| 2026 Q4 (5 weeks) | 5 | 67.7% | 89.0% | 12.4% | 4.5% | 8 (10.3) |

**The five new weeks, after W1a's test window** (October 2026):

| Week read | 2-in-3 held | 9-in-10 held | SPY | 25 widest moved | Average stock moved |
|---|---|---|---|---|---|
| Oct 2 | 70.3% | 94.4% | +1.2% | 10.5% | 3.5% |
| Oct 9 | 59.6% | 83.0% | -2.0% | 15.8% | 4.7% |
| Oct 16 | 72.5% | 93.3% | -0.4% | 11.1% | 4.1% |
| Oct 23 | 75.9% | 91.9% | -0.6% | 10.9% | 4.3% |
| Oct 30 | 60.3% | 82.6% | -2.7% | 13.7% | 6.1% |

## A misuse, shown on purpose: the scan's top movers as a buy list

If someone bought each week's 25 widest ranges at the close and sold them at the next week's close:

| $1 from Nov 8, 2024 | Became | Worst fall from a peak |
|---|---|---|
| The 25 widest ranges, before costs | $2.75 | -45% |
| The same, after 0.5% a round trip | $1.64 | -50% |
| Every listed stock, equal weight | $1.26 | -19.5% |
| SPY | $1.26 | -17% |

On paper that beats the market. It is not an edge:

- **A few lottery tickets carry it.**
  - The typical pick lost 0.8% a week.
  - Without the best 1% of picks (26 of 2,600), the average pick lost 0.13% a week.
  - Its gains came from a few explosions: SBET +1,041% in one week (May 2025), CAPR +403%, QUBT +163%,
    QMCO +156%, SLNH +155%.
- **Take away its three best weeks and it loses** after costs ($0.83); before costs it makes $1.37.
- **It is twice the market's swings:** its weekly return moved 2.1 times SPY's. What is left after that (+0.78%
  a week, t 1.14) is within chance, and so is its edge over every listed stock (+1.03% a week, t 1.44).
- **Survivorship favours exactly this bet.** The list was chosen in 2026, so a volatile stock that went to zero
  in these years is missing, and its zero is missing with it.
- **The years before say the same.** In 2011-2023, where W1 was fitted, the same buy-list, after the same costs,
  turned $1 into $4.20, against $5.70 for every listed stock. Its before-cost figure there, $124, is flattered by
  the survivorship above, and counts a few holiday sessions twice (five-session windows).
- **The paper trader's runs agree** (`docs/PAPER.md` on `claude/sleepy-goldberg-2ma18z`):
  - buying the 3 jumpiest stocks each week lost 0.47% a week in P3b (2021-2026);
  - its review found volatility worked against every version under its exits, in all three periods it checked.

The scan locates size. Which way a stock goes, it does not know.

## Limits

- **Survivorship:** companies delisted or bought out between 2024 and 2026 are missing, and so are their largest
  moves.
- **One window:** 104 weeks, most of them already inside W1a's test. The five October weeks are the only new
  ones.
- **Trading costs are estimates:** 0.5% a round trip for these stocks may be low; spreads on a $3 stock run
  wider.
- **Paper only.** Nothing here places an order.
