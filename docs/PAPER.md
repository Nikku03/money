# A paper trader that learns from its own trades (`brain/paper`)

**Paper only.** Nothing here places an order, connects to a broker or uses real money (CLAUDE.md).
Its trades are its own record. They are not advice, and they carry no weight in `brain decide`
unless a pre-declared test passes.

```
python -m brain paper replay --json replay.json   # walk-forward over 2015-2026, graded by P1
python -m brain paper step                        # one live day: close a due book, learn, open the next
python -m brain paper status                      # the live record (says nothing under 30 closed weeks)
```

## What it does

Every 5 sessions it scores the 60 stocks of `universe/us_large.json` and may open a long-short
book: long the 5 highest scores, short the 5 lowest, held 5 sessions. Long-short cancels the
market's moves, so only stock picking counts. It trades only when the predicted spread beats the
round-trip cost of 0.4% (0.1% each way, both legs).

- **What it sees** (`features.py`, all on or before the decision close, z-scored across stocks):
  - its own momentum (1 week, 1 month, 12-1), volatility, and distance from the 52-week high;
  - the dots: its five measured neighbours' last week, and its measured sensitivity to oil (USO),
    bonds (TLT), the dollar (UUP) and the market (SPY), times how each moved last week;
  - live only: today's headlines that name the company, tagged by fixed rules (`news.py`: upgrade,
    downgrade, beat, miss, guidance up or down, approval, legal, buyback, layoffs).
- **How it learns** (`learner.py`): a ridge-regularized linear model, updated by recursive least
  squares when each book closes. It learns from the outcome of every stock it scored, traded or not,
  and never sees an outcome before the book that produced it has closed. Old lessons fade (0.995 per
  week), so it can adapt when markets change. Every weight starts at zero, so it trades on nothing
  until outcomes pull a weight away from zero.
- **Live** (`live.py`): each step is appended to `research/paper/ledger.jsonl` and the learner and
  open book to `research/paper/state.json`, before outcomes are known. Commit both after each step.

## The test, declared before the first run on real data (P1)

Written and committed before `brain paper replay` was run on real prices. It is never edited; a
changed test is P2.

- **Data:** Yahoo adjusted daily closes, 2014 to the latest close, for the 60 stocks plus SPY, USO,
  TLT and UUP. Decisions every 5th session from the first with a year and a month of history.
- **Test period:** every decision week after the learner's first 52 updates, about a year of
  warm-up, through the last week whose outcome is known.
- **Measure:** weekly net P&L of the book after costs, with weeks it stays flat counted as 0.
- **PASS only if all three hold:**
  1. the mean weekly net P&L is above 0, with a one-sided t-test p below 0.05 (the weeks do not
     overlap);
  2. the mean beats the same learner taught with shuffled outcomes (each week's outcomes permuted
     across stocks);
  3. the mean is above 0 in both halves of the test period.
- **Otherwise FAIL.** A FAIL is recorded and stays. A PASS would still face the repo's
  false-discovery correction across every hypothesis already run (40-odd, nearly all FAIL) before
  anything relied on it, and the live record would have to agree.

Known biases, stated in advance:
- **Survivorship.** The 60 stocks were chosen by their 2026 size, so their past looks better than a
  real investor's choices did then. A long-short book is less exposed to this, but not immune.
- **No news in the replay.** The headline feed has no history, so news weights can only be learned
  live, going forward.
- **Prior expectations.** The repo's record says to expect little: momentum, reversal, value and
  earnings-reaction hypotheses all failed (`research/RESULTS.md`).

## P1 result, recorded 2026-11-05: **FAIL** (`research/paper/P1.json`)

Test period 2016-02-16 to 2026-10-27, 539 weeks, of which it traded 533. Figures are per week,
after costs:

| book | mean | t | one-sided p | annualized | Sharpe | total | worst drawdown |
|---|---|---|---|---|---|---|---|
| **the learning trader** | **−0.171%** | −0.81 | 0.79 | −8.5% | −0.25 | **−79%** | −87.5% |
| first half | −0.256% | −1.05 | 0.85 | | | −60% | |
| second half | −0.087% | −0.25 | 0.60 | | | −49% | |
| shuffled-outcome control | +0.032% | +0.47 | 0.32 | +1.7% | +0.15 | +11% | −32% |
| 12-1 momentum book, no learning | −0.308% | −1.38 | 0.92 | −14.8% | −0.43 | −91% | −94% |
| SPY, held (context) | +0.322% | +3.11 | 0.001 | +18.2% | +0.96 | +382% | −29% |
| the 60 stocks, held equally (context; survivorship-biased) | +0.480% | +4.32 | <0.001 | +28.3% | +1.34 | +1002% | −27% |

None of the three checks held: the mean is below 0, it is worse than the shuffled control, and it
is negative in both halves.

Looked at after the fact (not part of P1, and not a claim):

- **Before costs**, the trader made +0.227% a week (t 1.06, p 0.14): no evidence of an edge. The
  0.4% weekly cost turned noise into a steady loss.
- **It was overconfident.** Its median predicted spread was +0.83% a week; it got +0.23%. So its
  cost bar almost never held it back. The shuffled control, knowing nothing, predicted small spreads
  and stayed flat in 451 of 539 weeks, which is why it lost less.
- **The learned weights are tiny** (all under 0.002 per z-score). It found nothing durable: its
  strongest leanings (towards volatile stocks and 12-1 momentum, away from stocks that move with the
  market) are the kind that come and go.
- **This agrees with the repo's record:** momentum, reversal, value and earnings-drift hypotheses all
  failed (`research/RESULTS.md`).

A fix for the overconfidence (trade only when the learner's own uncertainty allows it), or anything
else, is a new test, P2, declared before it runs. P1 stays as recorded.

## Live record

Started 2026-11-04 (first step after that day's close) with **every weight at zero**, not seeded
from the replay that failed. Each closed week updates it; its news weights can only be learned this
way. `brain paper status` reports it; under 30 closed weeks it says nothing. On the first live day
the news rules mis-tagged several headlines. They were tightened, and the day's step redone, before
any outcome existed (`news.py`).

## The second test, declared before its first run (P2): prices and historical company news

Asked to start from the past, with news, the trader gets the only dated news history available here
for free: each company's SEC EDGAR filings since 2010, with the moment each became public. Written
and committed before any filing history was fetched or the replay run.

- **Added features** (`filings.py`): per stock, the filings public after the previous decision close
  and by this one, counted by group (clipped at 2). The groups come from the 8-K form's own item list,
  fixed in advance:
  - `sec_results` 2.02, `sec_deal` 1.01/1.02/2.01, `sec_exec` 5.02;
  - `sec_restructure` 2.05/2.06, `sec_trouble` 4.01/4.02/3.01/1.03/2.04;
  - `sec_disclosure` 7.01/8.01, `sec_capital` 2.03/3.02;
  - `sec_activist`: SC 13D and 13D/A.
- **Everything else is as P1:** the same prices, universe, learner, costs and test period rule, and
  the same three checks (mean > 0 with one-sided p < 0.05; beats the same learner taught shuffled
  outcomes, now with the filings too; mean > 0 in both halves). FAIL otherwise.
- **The live trader then continues from P2's learner** (its weights and their uncertainty), whatever
  the verdict, as asked. The headline features it never saw in history start at 0. The handover is
  recorded in the ledger as a `restart` event.
- **Expectation, stated in advance:** filings become public before or at the reaction, and the repo's
  earnings-event drift tests all failed (E1, E2), so little is expected beyond P1.
