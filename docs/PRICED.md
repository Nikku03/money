# What the options already price (V1, O1; `brain/priced.py`)

Every direction signal tested in this repo has failed. Move size is different: it persists, and three
tested forecasts of it held their stated odds (the forecaster's bands, M1, S1). A size forecast is worth
something only where it disagrees with what the options already charge for the move, and only if that
disagreement survives the cost of trading options.

The first look, at M1's 99 earnings calls of one week (`docs/MOVES.md`):
- the options had priced most of the moves, and ranked their sizes as well as M1;
- where M1 disagreed, it leaned the right way (p 0.08).

That is not proven. These two tests, declared here before their first call, decide it.

```
python -m brain priced week      # on a weekend: judge expired calls, record V1 and O1 from Friday's close
python -m brain priced status    # both records so far; under 30 judged weeks they say nothing
```

## V1: any week, every stock with weekly options (declared 2026-11-09, before its first call)

- **When:** each weekend, from the last Friday's close. A job on Saturday sees no newer price than
  Friday's.
- **Who:** every stock on `universe/us_broad.json` with an option expiring within the next seven days
  (normally the next Friday).
- **What the options price:** the at-the-money straddle's mid over the stock's price, from CBOE's
  delayed quotes at Friday's close. Only usable quotes count: each leg's spread at most half its mid.
- **Our forecast** of the coming week's move, either way (`Har`, `knowledge/priced_v1.json`):
  - It reads the stock's own variance over the last 5, 22 and 66 sessions, and the VIX.
  - It was fitted once on 2011-2018 (297,882 stock-weeks of the list) and frozen. The size of a
    five-session move is that forecast times the training years' own distribution.
  - **Checked on 2019-2022, years it did not see:** its 50%, 80% and 90% sizes held 52.9%, 83.0% and
    92.1% of the moves (slightly cautious). Its rank correlation with the actual size was +0.347,
    against +0.309 for the 66-session volatility alone.
- **Events stay out.** A stock whose options price an event is recorded but left out of the test: its
  front expiry's at-the-money volatility is 15%+ above the next expiry's, usually an earnings report.
  Earnings are O1's.
- **The sides:** each week, the fifth of the test stocks with the highest forecast / priced ratio are
  "buy" (the straddle looks cheap), the lowest fifth "sell". Ranking within the week means a bias every
  stock shares cannot make the result, for instance options being dear on average or the forecast
  running a little high.
- **Judged at the expiry's close.** A straddle held to expiry pays its intrinsic value |close − strike|.
  Per dollar of stock:
  - the buyer gets payoff − (mid + half the quoted spread);
  - the seller gets (mid − half the spread) − payoff.
- **PASS only if both hold** over 30 judged weeks with at least five test calls each (one-sided
  p < 0.05, with the week as the unit):
  1. **Information:** the mean weekly rank correlation between forecast / priced and actual move /
     priced is above 0. Our forecast sees something the options' price does not.
  2. **Money:** following the sides (buys and sells, equal weight), after half the spread, makes money.
- **Reported, not graded:** always selling (what the options' usual premium pays a seller), and how
  often moves beat the priced move.

## O1: earnings (declared 2026-11-09, before its first call)

- **When and who:** each weekend, the coming week's likely reporters among the broad list and the 1,000
  largest SEC-listed companies (the earnings radar's index, read afresh each week). Only those with
  usable straddle quotes whose options price the report count (the radar's volatility bump).
- **What is compared:**
  - M1's expected move for the report (`docs/MOVES.md`);
  - the options' priced move for it: the straddle's priced move less the stock's ordinary days to
    expiry (the radar's `event_move`).
- **The side:** M1's, the 20% margin of the radar.
- **Judged** at the straddle's expiry close, as V1.
- **PASS only if both hold** over 30 judged weeks with at least five calls each: the rank correlation
  (M1 / priced against actual move / priced) and following M1's side after half the spread, each above
  0 at one-sided p < 0.05.

## Stated in advance

- **No free history of option prices exists here.** So neither test can look back: both run forward
  from 2026-11-14. Thirty judged weeks of V1 take about seven months; O1's thirty earnings weeks about
  a year.
- **The mid may not be a price anyone gets,** and paying half the quoted spread is an estimate.
  Dividends and early exercise are ignored for a one-week at-the-money straddle.
- **A weekly straddle seller can lose many times the premium in one week.** "Always selling" is
  reported to show the options' usual premium, not as a strategy.
- **Paper only.** Nothing here places an order.
