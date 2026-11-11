# Shocks (S1): where the unexpected is likely to land

Every direction signal tested in this repo has failed. How much a stock moves is different: it persists (M1,
`docs/MOVES.md`). S1 asks the same of the extreme moves people call surprises: a drug company up 200% on an FDA
designation, a stock up 70% on a bid, a country fund down 20% after an election.

## What can and cannot be predicted

- **Not the day and the direction.** If public data said a stock would jump tomorrow, buyers would pay for it today.
- **Where and roughly when.** Shocks are not spread evenly. They land on small, cheap, jumpy stocks; on drug
  companies waiting for an answer; on stocks that moved hard lately; on stocks about to report or about to sell
  shares. S1 ranks every stock each week by its chance of a shock.
- **What follows.** Whether a shock tends to hold or fade is a separate, testable question: S2a and S2b below.

## What it predicts

The chance that a stock moves 20% or more against SPY, either way, in one session of the next five. A row is one
stock at one week's last close; every input was known then (`brain/shocks.py` lists them): its volatility over 20,
60 and 250 sessions, its largest move and its count of 10%+ and 20%+ days over the past year, the latest moves, its
returns over 5, 20 and 60 sessions, its price and dollar volume, a volume surge, its distance from the 52-week high
and low, recent 8-Ks, a recent prospectus or shelf filing (a share sale is near), a report expected within eight days
(last year's date moved on 52 weeks), whether it is a drug or biotech company, and the VIX.

The model is a logistic regression with a small ridge, fitted on the research universe's stock-weeks of 2011-2023
(`knowledge/shocks_s1.json`). The baseline is the same fit on the volatility history alone.

## The declared tests

`brain/research/shocktest.py`, declared before they ran. TEST is 2024-01-01 to 2026-09-30, which no model or rule
here was built on.

**S1a, the radar.**
- Every research-universe stock-week whose five sessions have closed.
- Per stock-week the score is the log loss of the chance; per month, the baseline's mean loss minus S1's. The
  statistic is the mean of the monthly differences, with a one-sided sign-flip p.
- It passes when:
  - the difference is positive with p < 0.05;
  - S1's mean chance is within 0.67-1.5 of the actual rate, overall and in each week's top twentieth;
  - the controls pass: S1's chances shuffled within each week reach p < 0.05 in at most 10% of draws, and an oracle
    reaches p < 0.01.

**S2a and S2b, what follows a jump.** In 2011-2023 a stock that jumped 20%+ above SPY without a results report then
did worse than equally jumpy stocks over the next month, every year (`research/SHOCKS-2026-11.md`). The tests ask
the same of 2024-2026.
- An episode opens on a session 20%+ above SPY:
  - with no results report (an 8-K with item 2.02) that session;
  - with no up-shock in the 30 days before.
- Its outcome is its excess return over the next 21 sessions. Peers are the stocks in its 60-session volatility decile,
  read the close before, that did not shock that day. Volatile stocks move more and drift down in log terms, so an
  unmatched average would mislead.
- **S2a**, the typical episode: per month, the share of its peers each episode beat, less one half.
- **S2b**, the average episode, which is what a trade would earn before costs: per month, each episode's outcome
  less its peers' mean.
- Each passes when:
  - its mean of monthly values is negative with a one-sided sign-flip p < 0.05;
  - a random peer standing in for each episode reaches p < 0.05 in at most 10% of draws;
  - an oracle that puts each episode at its peers' 10th percentile reaches p < 0.01.

All three also need Benjamini-Hochberg q < 0.10 across every gated test in the ledger.

**Results (2026-11-03, `research/RESULTS.md`).**
- **S1a: PASS.**
  - Better than volatility alone in 31 of 33 months; each week's top 5% held 52% of the shocks.
  - Its chances ran about a fifth low (0.72% actual against 0.58% predicted).
- **S2a: FAIL.** The 2011-2023 fade after a jump did not repeat.
- **S2b: FAIL (controls).**

```
python -m brain shocks fit          # 2011-2023 only; commit knowledge/shocks_s1.json
python -m brain shocktest declare   # S1a, S2a, S2b; commit the ledger
python -m brain shocktest test      # commit the ledger and research/RESULTS.md, whatever they say
```

## Catalysts: what comes with a surprise, and S3

`research/CATALYSTS-2026-11.md` (2011-2023 only) sorted every 20%+ session by what the company filed that session:

| Filed that session | 2011-2019 | 2020-2023 |
|---|---|---|
| Nothing at all | 79% | 73% |
| Results report (8-K 2.02) | 12% | 17% |
| Other filings | 9% | 10% |

- Most surprises come with nothing filed: the news came through press wires or social media, or there was none.
- Deals, distress filings, executive changes, press releases, merger filings, activist stakes, reverse splits and
  lockups added nothing to S1.
- **Results reports are the one large catalyst known in advance, and S1 is weakest there.** It had only a third of
  earnings-day surprises in its week's top 5%.
- **What helps is each stock's earnings habit:** the median move against SPY on its last eight report sessions.

**S3** (`brain/catalysts.py`, `knowledge/shocks_s3.json`) is S1 plus:
- `earnq`: a report expected 1-8 days ahead, a quarter after the last one;
- the habit (`ehab_known`, `ehab_l`);
- the habit when a report is due (`ehab_win`).

It is fitted on the same stock-weeks of 2011-2023:
```
python -m brain catalysts fit
python -m brain catalysttest declare      # commit the code, both models and the ledger first
python -m brain catalysttest test         # once; commit the ledger and RESULTS.md whatever they say
```

**S3a**, declared before it ran, asks whether S3 beats S1 in the weeks a stock's report is expected:
- **Test period:** 2024-01 to 2026-09, research universe.
- **Score:** monthly log loss, one-sided sign-flip p < 0.05.
- **Calibration:** S3's chances must match the actual rate within 0.67-1.5, overall and in each week's top 5%.
- **Controls** and the false-discovery correction as for S1a.
- **Reported, not gated:** every stock-week.

## Using it

```
python -m brain shocks radar              # the research universe, highest chance first
python -m brain shocks radar SUPN CAPR    # named tickers (they need prices in the bank)
```

Each line is a chance, not a call. A stock's chance of a shock says nothing about its direction, and an option's
price already charges for most of what S1 sees. Any call made from it goes into `brain predict` first.

**The live record.**
- The radar's top 25 on Tuesday Nov 3's close went into `research/s1_calls.jsonl` as shock calls, judged through
  Nov 10 (`BRAIN_PREDICTIONS=research/s1_calls.jsonl python -m brain predict score`).
- A call is withdrawn, with its reason, only for bad input found before its first session closes. On Nov 3 that took
  out two:
  - BSAI, whose price history jumps 52,074% in one day (an unadjusted split or a reused ticker);
  - BSCL, which had no Nov 3 bar and no trades on 17 of its last 20 sessions.

## Limits

- **Survivorship.** The universe is chosen today: companies bought out or delisted are missing, and so are their
  biggest days, the takeover jumps and the collapses. Shocks are undercounted, more so among the smallest stocks.
- **The smallest stocks are mostly outside it.** The micro list stops at SEC rank 4,500 and needs prices since 2010,
  so recent listings, where many 100%+ days happen, are not in it.
- **Industry codes are today's**, and a report's expected date is a year-old date moved on, not the announced one.
- **Broken price histories can top the list.** An unadjusted split, a reused ticker or a stock that has stopped
  trading feeds the radar wrong inputs, and the radar does not flag them yet. Check a name's prices before quoting
  its chance.
- **Thin over-the-counter names rank high.** Some top names trade a few thousand dollars a day, where one small
  trade can make a 20% session. Their chances are honest, but such a shock means little.
- **A fade against peers is not a trade.** Shorting a stock that just jumped costs borrow fees and risks a squeeze,
  and among stocks under $5 a few keep exploding.
- **A size is not a direction**, and nothing here trades.
