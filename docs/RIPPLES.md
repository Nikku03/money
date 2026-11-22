# Ripples in the pond (`brain/ripple.py`)

The person's picture (2026-11-22): the market is a pond and every piece of news is a stone. A big stone makes a high
wave. The wave travels and hits what is around it; whatever it hits either absorbs it or takes a hit. When waves
from several stones meet they add (constructive) or cancel (destructive), and the result travels on. The goal: how
far and how high the waves travel, what they hit, how hard, and what happens where they meet.

```
python -m brain ripple map                      # the stones and waves on the latest check
python -m brain ripple map --date 2026-11-20    # on one session's close-to-close moves
python -m brain rippletest declare|test         # the declared tests R1a-R1d
```

## The pieces

| Picture | In the bank |
|---|---|
| The pond | A year of daily returns with the market's part taken out, each stock in its own usual days. How closely two stocks moved together over that year is the water between them. |
| Dry land | A stock with no link of 0.20 or more to any of the day's stones: no wave reaches it. |
| A stone | A stock whose own move (market removed) is 3 or more of its usual days. Its push is capped at 3 usual days: past that, the move is its own news, which its neighbours did not share over the year. |
| A wave | What the year of returns says a stock moves when the day's stones move as they did, all at once (a regression on the stones together, so stones that share one cause are not counted once each). |
| Height | The wave in the stock's own usual days, times 0.24: over 2012-2016 (867,000 stock-days, inside the training years), stocks moved about a quarter of what the year's links say, the same day. |
| Energy | All the water that reached a stock, whatever its direction. |
| Interference | Height over energy: 1 when every stone pushed the same way (constructive), near 0 when they cancelled (destructive). |
| Rings | First ring: a partner of a stone (a link of 0.30 or more). Second: a partner of a partner. |
| Withstanding | A wave under a quarter of a usual day is absorbed (lost in ordinary noise); a quarter to one is felt; one or more is a hit. |

## What the waves did in the training years (2011-2023, reported, not tested)

Measured while building, on the training years only:

- **They travel a short way.** On a typical session, 27 stones sent waves to about 830 stocks.
  - The first ring moved 0.78 usual days.
  - The second ring moved 0.64, and stocks beyond it 0.65.
  - Stocks on dry land moved 0.59.
  - So the wave is felt next to the stone and mostly gone one step out.
- **Most are absorbed.** 59% of waves were under a quarter of a usual day, 41% were felt, and 0.13% were a hit.
- **On the day, waves match moves.** The rank correlation between a wave's height and the stock's own move that day
  was +0.21. That describes how stocks move together; it is not a forecast.
- **Neither direction nor interference showed anything for the next five sessions** in the training years:
  - direction: a rank correlation of +0.0015, p 0.34;
  - interference: +0.002, p 0.31.

## The declared tests (`brain/research/rippletest.py`, results in `research/RESULTS.md`)

The setup:

- **Universe:** the research universe (1,565 stocks with prices since 2010).
- **Years:** TRAIN 2011-2023 (the size models are fitted here); TEST 2024-01 to 2026-09, which no choice here looked
  at.
- **Pond:** rebuilt every 21 sessions from the 250 sessions before it.
- **Outcome:** each reached stock's own move over the next five sessions.

The tests:

- **R1a (reported):** the waves on the day: the match of height and own move, the move by ring and on dry land, and
  how many waves are absorbed, felt or a hit.
- **R1b, how high the waves travel.** Does a stock's wave today tell how big its next five sessions are, beyond its
  own recent moves, its own move today and the market's recent size? Two least-squares models are fitted on TRAIN,
  with and without the wave's height and energy. The statistic is the monthly gain in squared error on TEST.
- **R1c, which way they travel.** Does the wave's sign tell the next five sessions' direction, as a wave still
  arriving would? The statistic is the monthly mean of each session's rank correlation, among stocks with wave energy
  of half a usual day or more.
- **R1d, interference.** Among stocks with wave energy of one usual day or more, do waves that added up leave a
  bigger next five sessions than waves that cancelled, beyond the baseline?

Each gated test passes on a positive statistic with a one-sided sign-flip p under 0.05 over months, its controls (a
placebo that shuffles waves within each session; an oracle), and Benjamini-Hochberg q under 0.10 across every gated
test in the ledger. Results are recorded whatever they say.

## Results on TEST (2024-01 to 2026-09, run once, `research/RESULTS.md`)

| Test | Question | Result | Verdict |
|---|---|---|---|
| R1a | the waves on the day (reported) | height matched the day's own move, rank correlation +0.237; first ring 0.747 usual days, second 0.633, beyond 0.638, dry land 0.593; 58% absorbed, 42% felt, 0.06% a hit | reported |
| R1b | how high the waves travel over the next five sessions | the waves cut the squared error by 1.1% (p 0.0011, 26 of 33 months), but shuffling the waves among each session's stocks did as well in every draw | **FAIL (controls)** |
| R1c | which way they travel | rank correlation -0.0012 (p 0.57) | **FAIL** |
| R1d | waves that add against waves that cancel | -0.0108 (p 0.88) | **FAIL** |

What this says:

- **On the day, a wave reaches one ring.** Partners of a stone moved 26% more than stocks on dry land; partners of
  partners 7% more.
- **It is mostly absorbed.** At the same-day calibration, almost no stock outside the stones takes a hit.
- **The day's storm matters, where the waves went does not.** After the run, outside the test, the size gain was
  split: the day's waves as a whole cut the error by 1.02%, and where they went added 0.06%. A session full of
  stones means bigger moves next week for every stock, and the scan's own recent-move inputs already carry most of
  that.
- **Neither the direction of a wave nor how waves meet says anything about the next five sessions.** A stock's own
  move reversing (+0.019) explains what little the wave not yet moved showed.

## Reading a map

A map lists the day's stones, then the stocks the waves reach highest. For each stock it gives:

- the wave's height and how the stock takes it (absorbed, felt, a hit);
- the energy and the interference;
- the ring;
- the stock's own move;
- the stones that sent the largest parts of its wave.

On Friday Nov 20, 2026 the highest waves (about 0.8 of a usual day) reached mortgage lenders, homebuilders and building
suppliers. They came mostly from Rocket and Invesco Mortgage (rate-linked) and from Spectrum Brands and Gap. 11 of the top 12
moved up that day against the market.

A map describes how the day's moves fit the year's links. R1b-R1d failed: it does not say what comes next.
