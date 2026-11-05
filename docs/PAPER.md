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

## P2 result, recorded 2026-11-05: **FAIL** (`research/paper/P2.json`)

The filing history: 19,604 tagged filings across all 60 companies, read from SEC EDGAR. Test period
and weeks as P1 (539 weeks; it traded in 537). Figures are per week, after costs:

| book | mean | t | one-sided p | total | worst drawdown |
|---|---|---|---|---|---|
| **trader, prices + filings** | **−0.217%** | −1.08 | 0.86 | **−83%** | −88% |
| first half | −0.458% | −1.99 | 0.98 | −76% | |
| second half | +0.024% | +0.07 | 0.47 | −28% | |
| shuffled-outcome control (with filings) | −0.052% | −0.46 | 0.68 | −37% | −63% |
| P1, prices only (for comparison) | −0.171% | −0.81 | 0.79 | −79% | −88% |

None of the three checks held. Adding the news history made it slightly worse, not better.

Looked at after the fact (not claims):

- **Before costs** it made +0.18% a week (t 0.90).
- **Its largest learned weight** is +0.028 on `sec_trouble` (restatements, auditor changes,
  delisting notices), that is, buy troubled stocks. Those filings are rare among these 60 large
  companies, so this is most likely fitted to a handful of events. The ridge prior shrinks rare
  features less than it should when their few outcomes happen to agree.

## The live trader now starts from the past

On 2026-11-05 the live trader was restarted from P2's learner, as declared: 591 weekly lessons,
2015 to 2026-10-27, with the headline features at 0 (ledger event `restart`). The first live book, on
2026-11-04's close, went long MRVL, STX, AMD, MU and INTC and short BRK-B, COST, KO, WMT and ABT,
with a predicted spread of +1.12%. That is its learned lean toward volatile stocks over steady ones.
This is the paper decision of a model that failed both declared tests; it is recorded to be scored,
and it is not a recommendation.

# The time machine: a moonshot hunter living from 2021 (P3, `brain/paper/moonshot.py`)

Asked for: a trader dropped into Monday 4 January 2021 at the opening bell, taking short-term paper
trades aimed at moonshots, learning from every result why the stock moved, and using those lessons
on its next trades. Paper only, a simulation of the past.

## The rules, fixed before the first run

- **Universe:** the 510 companies of `universe/us_small.json`.
  **Survivorship, stated plainly:** the list was drawn up in 2026, so every member survived to 2026.
  A trader in 2021 could not have known which would, so the list inflates every absolute result. Only
  the comparison with random picks from the same list is fair, and that is what P3 tests.
- **Clock:** each week's first session, at the open, from 2021-01-04 to the last week whose trades
  have finished. It knows:
  - prices up to the previous close, and that morning's opening prices;
  - SEC filings public before 09:30 New York time.
  Before 2021 it may study the outcomes it can see from mid-2020 (its features need a year of prices
  first), as someone living then could. It takes no trades before 2021-01-04.
- **The trade:** buy the 3 stocks it rates likeliest to moonshot, at the open.
  - Exit at +20% (the moonshot), at −10%, or at the close of the 10th session, whichever comes first.
  - A day touching both levels counts as the stop. An open beyond a level exits at that open.
  - 0.25% cost each way.
- **What it sees:**
  - jumpiness (volatility, its largest move and its 10%+ days over the year);
  - recent moves against SPY, and the morning's gap;
  - distance from its 52-week high and low, price, dollar volume and volume surge;
  - last week's SEC filings by 8-K group, a share sale in prospect, earnings likely during the hold;
  - drug or biotech company or not;
  - the market's last week and the VIX.
- **How it learns:** the chance of hitting +20% before −10% within 10 sessions, by ridge logistic
  regression. It learns from every stock it watched whose trade would have resolved before this
  open, not only its own three, and refits each week, with older outcomes fading (half-life 52 weeks).
- **Why it moved:** each closed trade's move is split into the market's part, oil's, bonds' and the
  dollar's (the stock's sensitivity over the prior year times each asset's move), and the stock's own
  remainder, plus the filings it made during the trade. This is attribution, what the move went with,
  not proof of cause. A journal keeps every lesson, and every quarter what the learner currently
  believes raises and lowers the odds.

## P3: PASS only if all three hold

1. The weekly mean net return of its trades beats the expected return of 3 random picks from the same
   list that week, the same rules applied to every stock (one-sided t-test over weeks, p < 0.05).
2. The weekly mean net return is above 0 (one-sided p < 0.05).
3. It beats the same learner taught shuffled outcomes (outcomes permuted within each week),
   one-sided p < 0.05.

FAIL otherwise. Reported but not part of the test: its hit rate against the base rate, its hit rate
year by year (is it getting smarter?), the first and second halves, and buying the 3 jumpiest stocks
(size without direction, as the shock radar S1 does).

**Expectation, stated in advance:** S1 has shown that size can be predicted, but every direction
test in this repo has failed. A moonshot needs direction, so P3 will likely fail. It is run anyway,
and recorded either way.

## P3 result, recorded 2026-11-05: **FAIL**, withdrawn as flawed (`research/paper/P3.json`)

303 weeks from 2021-01-04 and 909 trades. Its weekly net was +0.73% (t 0.99, p 0.16). It beat random
picks by +0.76% a week (p 0.15) and the shuffled learner by +0.76% (p 0.16). None of the three checks
held.

**Withdrawn as a mistake in the design, not deleted.** P3 never asked whether a stock could actually
be bought. It picked MALG, a stock with no recorded volume on 86% of days whose "price" was a stale
quote jumping between levels, in 99 of its 909 trades, because stale jumps look like volatility.
Its numbers stay above, for the record.

## P3b, declared before its first run: P3 with a tradability check

Identical to P3 in every rule, check and comparison, with one addition fixed in advance:

- **Tradable:** at each open, a stock counts only if it traded on every one of the last 20 sessions
  and averaged at least $500,000 a day in dollar volume. This applies to what it buys, to what it
  learns from (an untradable stock's outcome is not a real outcome), and to the random-pick and
  shuffled controls alike.
- **The same three checks**, at one-sided p < 0.05: beats random picks from the tradable list; makes
  money after costs; beats the shuffled learner.

## P3b result, recorded 2026-11-05: **FAIL** (`research/paper/P3b.json`; every trade and its lesson in `P3b_journal.jsonl`)

303 weeks from 2021-01-04, 909 trades, tradable stocks only.

| | per week | t | one-sided p |
|---|---|---|---|
| **its trades, after costs** | **−0.13%** | −0.24 | 0.60 |
| minus random picks from the same tradable list | −0.03% | −0.05 | 0.52 |
| minus the same learner taught shuffled outcomes | +0.59% | +1.06 | 0.15 |
| random picks (context) | −0.10% | −0.52 | |
| the 3 jumpiest stocks each week (context) | −0.47% | −1.02 | |

None of the three checks held. Without the stale-quote stock, P3's small positive number disappears:
the trader did no better than random picks.

What it did and did not learn (descriptive, not claims):

- **It finds big movers, not winners.**
  - 27.5% of its trades hit +20%, against a 3.8% base rate for any tradable stock that week. That
    is the size effect the shock radar S1 already passed on.
  - But 56% hit the −10% stop first: the stocks it picks move hard in both directions. Its median
    trade lost 10.5%.
- **Its learned beliefs are about size, not direction.** Odds rise with volatility, with earnings
  likely during the trade, with a high VIX, and with a big move earlier in the year. They fall with a
  higher share price and with a results filing already made. Nothing in them says up rather than
  down.
- **Overconfident.** It predicted a 47% chance on average and got 27.5%.
- **"Smarter with each trade" is not shown.**
  - Hit rate by year: 28%, 21%, 23%, 30%, 31%, 33%. Net per trade by year: +0.8%, −3.0%, −1.0%,
    +0.5%, +0.7%, +1.5%.
  - The second half of the test was better than the first (+1.0% a week, t 1.30, p 0.10), but that is
    within chance, and the market regime changed too (2022 was a bear market for small caps).
- **Why the moves happened, as attribution:**
  - About 71% of each trade's move was the stock itself, not the market, oil, bonds or the dollar.
  - Results filings were present in 18% of the moonshots and 14% of the stop-outs: news came with big
    moves in both directions.
  - The biggest winners and the biggest losers were the same kind of trade. APPS hit +54% and +70%
    around results, and lost 57% around results.
- **Buying the jumpiest stocks loses most** (−0.47% a week): lottery-like stocks underperform, a
  known effect.

The conclusion matches every direction test in this repo: public data, including a company's own
filings, predicts how much a stock will move, not which way.

## As a $1,000 account (computed 2026-11-05)

Each new trade takes 1/6 of the account's current value (3 trades a week, held up to 2 weeks, so
up to 6 open at once), with costs included, gains and losses compounding, and nothing added.

| | 2021-01-04 | end (2026-11-04 trades) | best | worst fall from a peak |
|---|---|---|---|---|
| **P3b, tradable stocks only** | $1,000 | **$551 (−45%)** | $1,410 | −82% (low $252) |
| random picks from the same list, same sizing | $1,000 | $817 (−18%) | | |
| P3, withdrawn (included the stale-quote stock) | $1,000 | $1,295 (+30%) | $1,595 | −69% |
| SPY, held (dividends reinvested) | $1,000 | $2,206 (+121%) | | |

- **P3b year by year** (the account at each year's first trade): 2021 $1,000; 2022 $1,184;
  2023 $480; 2024 $364; 2025 $329; 2026 $409; end $551.
- **Its trades:** 339 winners made +$5,268 and 564 losers lost −$5,718. The best trade was +$98
  (NLST, April 2021, +50%); the worst −$42 (NVAX, August 2022, −35%).
- **The withdrawn P3's profit came from the stock nobody could trade:** its best trade was MALG,
  +233% on a stale quote.

# Version 2: dynamic exits, sturdier learning, and a post-mortem of every trade (P4, `moonshot2.py`)

Asked for: a dynamic exit strategy, learning that is harder to fool, and an analysis of every trade
from every direction (why it won, why it lost), used on the next trade. Same clock, 510 small caps,
tradability check and 0.25% costs as P3b. Written and committed before its first run.

- **Five exits, fixed in advance:**

  | exit | rule |
  |---|---|
  | E0 fixed | +20%, −10%, 10 sessions (P3b) |
  | E1 scaled | target 2.5 and stop 1.25 times the stock's own 10-session volatility (8-60% and 4-30%), 10 sessions |
  | E2 trailing | no target; −10% stop, then trailing 2.5 daily volatilities (5-25%) below the highest close, up to 15 sessions |
  | E3 breakeven | −10% stop; once a close is 10% up, the stop moves to the entry; +30% target; 15 sessions |
  | E4 follow-through | as E0, but out at the 3rd close unless at least 3% up |

  Stops move only after a close, so no day uses its own later prices. Every candidate is simulated
  under all five.
- **The exit is chosen by experience.** There are 6 kinds of trade: market fear (VIX above or below
  its last year's median) × the stock's volatility tercile. For each kind, every exit's average over
  the trader's own top-10 candidates of past weeks (closed trades only), shrunk toward that exit's
  overall average, 30 trades' worth. The best is used. E0 until 30 weeks are known.
- **Sturdier learning.** It predicts the expected net under each exit: ridge regression on outcomes
  winsorized at ±40%, older weeks fading (half-life 52 weeks), bagged 8 times over weeks. It scores a
  stock as the mean of the 8 predictions for its chosen exit, less half their spread. It buys up to 3,
  only those scoring above 0, so it may sit out a week.
- **Memory of each stock.** Its last year of resolved outcomes under E0 (how often it hit +20%, how
  often it was stopped, its mean net, shrunk toward the average, 5 trades' worth) enter as features.
- **A post-mortem of every trade it takes:**
  - the path: its best and worst point and on which day; never worked, went right then reversed,
    gapped through the stop, timed out flat, or worked;
  - every exit's result in hindsight, the best one, and the regret of the one chosen;
  - its predicted net against the result;
  - the three features that drove the pick;
  - the market's, oil's, bonds' and the dollar's parts and the stock's own;
  - the company's filings during the trade.

  These lessons reach the next trade through the exit chooser (hindsight exits), the memory features
  (each stock's record), and the stake. The stake is half size for the $1,000 account while its last
  26 weeks of trades have not beaten random picks under the same exits. The stake does not enter P4.

## P4: PASS only if all three hold

Weekly mean net of its book (weeks it sits out count as 0):
1. beats 3 random tradable picks that week, under the exits the chooser gave their kinds of trade
   (one-sided p < 0.05);
2. is above 0 (one-sided p < 0.05);
3. beats the same machine taught shuffled outcomes (one-sided p < 0.05).

FAIL otherwise.

**Stated in advance:** this is the trader's fifth declared test on overlapping data (P1, P2, P3, P3b,
P4). Five tries raise the odds that one passes by luck, so a P4 PASS would be checked against that
before anything relied on it. By the repo's record, direction has never been predictable from public
data. Exits can change the shape of the losses, not create an edge that is not there.

## P4 result, recorded 2026-11-05: **FAIL**, two of three checks (`research/paper/P4.json`; every trade's post-mortem in `P4_journal.jsonl`)

302 weeks from 2021-01-04. It traded in 299 of them, 877 trades in all.

| check, per week (one-sided) | result | |
|---|---|---|
| beats random picks under the same exits | +1.33%, t 2.46, p 0.007 | held |
| makes money after costs | +1.28%, t 2.12, p 0.017 | held |
| beats the same machine taught shuffled outcomes | +0.84%, t 1.37, p 0.085 | **failed** |

Declared verdict: **FAIL**. This is the closest any version has come, so it was checked harder, after
the fact. None of these checks changes the verdict:

- **A few trades carry it.**

  | edge over random picks, per week | |
  |---|---|
  | all 877 trades | +1.33% (p 0.007) |
  | without its 5 best | +0.88% (p 0.034) |
  | without its 10 best | +0.59% (p 0.11) |
  | without its 20 best | +0.13% (p 0.39) |

  The best was Digital Turbine (APPS), +136% under the trailing exit in August 2024.
- **It all comes from 2024-2026.** The edge over random picks by year was −0.87%, +0.51%, −0.30%,
  +2.99%, +3.74% and +2.04% a week. It did not get better steadily; it did well once small, jumpy
  stocks near their highs (quantum computing, drones, AI suppliers) ran in 2024-25.
- **Survivorship can explain that.** The 510 stocks were chosen by their 2026 size. A stock that
  ran in 2024-25 and stayed up is on the list; one that ran and then collapsed below the cutoff is
  not. A trader leaning toward stocks near their 52-week highs gains from exactly that, and random
  picks gain less. This bias was stated before the run, and the 2024-26 edge is what it would look
  like.
- **The learning itself shows little.** The machine taught shuffled outcomes still made +0.44% a week:
  the exits and the stock memory (both learned from real outcomes in both machines) and the
  stock list do part of the work. Among the trades it took, its predicted return did not track the
  realized one (correlation −0.03). It was overconfident: +2.8% predicted, +1.4% realized per trade.
- **Five tests.** p 0.007 × 5 trader tests = 0.035, so the random-picks check would survive a
  Bonferroni correction on its own. But the declared rule needed all three checks.

What the post-mortems say (877 trades):

- **Shapes:**
  - never worked 23%, worked a little 22%, a moonshot or close 16%;
  - drifted to a loss 14%, timed out flat 13%;
  - went right then reversed 8%, gapped through the stop 4%.
- **Exits:** it chose the volatility-scaled exit most (546), then breakeven (139), trailing (102) and
  fixed (90). In hindsight the best exit varied evenly (fixed 248, trailing 201, follow-through 179,
  scaled 142, breakeven 107). The average regret was 6.2% a trade: no single exit is right, and the
  chooser could not tell in advance which one would be.
- **Influences:** 62% of each move was the stock itself. Results filings came with 23% of the
  winners and 14% of the losers.
- **Its reasons:** most picks leaned on high fear (VIX), being near the 52-week high, and a gap up
  at the open: momentum in jumpy stocks.

**As a $1,000 account** (1/6 of the account per trade, half stakes while its last 26 weeks trailed
random picks): $1,000 → **$2,723 (+172%)**, against SPY's $2,206 over the same period. On the way it
fell to $642 by early 2023 (a −48% fall from its peak). The account at each year's start:

| 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| $1,000 | $777 | $642 | $677 | $1,221 | $2,256 |

With full stakes it ends at $3,340. Because of the survivorship bias above, these figures overstate
what a trader in 2021 could have had.

## Every moonshot that happened, and why (`misses.py`; `research/paper/P4_moonshots.jsonl`)

The trader studied its own trades. This review asks "why?" of every moonshot that happened, caught
or missed: each tradable stock that, bought at a week's first open, reached +20% before −10% within
10 sessions. It uses P4's run, re-run with every stock's weekly score recorded; the result is
identical to the declared run. This is hindsight: it explains, it does not predict.

**How many it caught.** About 19 moonshots happened each week. Over 302 weeks there were 5,729; the
trader caught 154 (2.7%). Three random picks a week would catch about 0.6%, so it did about 4.4
times better than chance and still missed 97%.

| year | moonshots | caught | in its top 10 | in its top tenth | ranking skill (AUC; 0.5 = coin) |
|---|---|---|---|---|---|
| 2021 | 1,049 | 30 | 84 | 231 | 0.57 |
| 2022 | 845 | 18 | 43 | 115 | 0.52 |
| 2023 | 776 | 18 | 65 | 204 | 0.56 |
| 2024 | 995 | 34 | 78 | 200 | 0.56 |
| 2025 | 1,055 | 28 | 64 | 215 | 0.56 |
| 2026 (to Oct) | 1,009 | 26 | 77 | 271 | 0.62 |

Its ranking is a little better than a coin and did not improve steadily. 2026's 0.62 is its best year,
but one year is not a trend. Of the missed ones, 19% were in its top tenth that morning, 36% in its
top quarter, and 44% in its bottom half: for those, it saw nothing. Only 6 were in its top 3 and left
out for scoring under its bar.

**Why the missed moonshots happened** (primary cause, first that fits):

| cause | missed | caught |
|---|---|---|
| steady climb, no single jump | 22% | 12% |
| mixed (a big day plus a climb) | 27% | 33% |
| industry wave (median 50% of its industry up 10%+ with it) | 18% | 8% |
| jump with no SEC filing (news this system cannot see) | 16% | 38% |
| earnings jump | 12% | 8% |
| deal or activist jump | 1% | 1% |
| other filed-news jump | 2% | 1% |
| market wave | 2% | 0% |

Factors present among the missed (several at once):
- 35% were steady climbs, 31% one jump;
- 30% had earnings expected in the window;
- the market was up 3%+ in 21%;
- an industry wave in 21%;
- mostly overnight in 16%;
- volume tripled or more in 7%.

What this says:

- **It catches surprises in jumpy stocks and misses trends and waves.** Its catches are mostly jumps
  with no filing behind them, in the cheap, jumpy stocks it favours. It misses most steady climbs and
  industry waves. The largest waves were July 2024, April 2026, November 2024 and February 2021,
  months when whole industries moved together.
- **What the missed moonshots looked like.** Against an ordinary stock, they had more 10%+ days and
  a huge move earlier in the year, a big move that week, were further from their 52-week high,
  and were more often under $5. Against its picks, they were less extreme: further up over the past
  week, month and quarter, nearer their highs, and higher priced. So the missed ones are jumpy too,
  but trending, where its picks are the very jumpiest.
- **Earnings were knowable and not used.** 71% of the missed earnings jumps had earnings expected in
  the window, yet the median one ranked at the 37th percentile. A report date says a big move is
  likely, but not which way.
- **Some causes are outside its sight.** The biggest misses were jumps of 75-107% (log) in a few
  days, mostly overnight on huge volume: NKTR twice in June 2025, REPX in January 2021, OMER in
  October 2025 (a deal), and CAR in October 2021 (earnings, ranked 484 of 486). The news that moved
  most of them came from outside its filings: trial results, press releases, a short squeeze.

**What reaches the next trade.** The trader already learns from every stock's outcome, missed
moonshots included; that is how its model is fitted. The review adds what the model cannot learn on
its own: causes it has no feature for. Two candidates, both found by looking at these same years:
1. an industry-wave feature: how its industry moved over the last week, measured at the open;
2. trend inside jumpy stocks.

Adding them and testing on 2021-2026 would be testing on the data that suggested them, so the only
clean test of them is forward, from now.

# Know the company before trading it (P5, `dossier.py`)

Asked for: the trader should know the company inside out before it trades. Before each trade it
builds a dossier from what was public before that open:
- the company's own financial reports (SEC XBRL), each number counted from the session after the
  day it was filed;
- its filing history;
- its own price history, including how it reacted to its last four results.

The dossier is written into the journal for every trade, before the trade, and its numbers enter
the learner as features:

| feature | what it says |
|---|---|
| `f_size`, `f_ps` | market value; price against sales |
| `f_rev_growth` | revenue against the same quarter a year before |
| `f_gross_margin`, `f_op_margin`, `f_profitable` | margins; a profit or a loss last quarter |
| `f_rnd` | R&D against revenue |
| `f_runway` | cash over the quarterly operating burn (20 quarters if it does not burn) |
| `f_debt`, `f_current` | long-term debt over assets; current assets over current liabilities |
| `f_dilution` | shares outstanding against a year before |
| `f_stale` | how old its latest report is |
| `f_earn_abs`, `f_earn_sign` | its last four results moves: their mean size and mean direction |
| `f_fin_missing` | it reports no revenue |

A missing number takes that week's median.

**Not included:** the industry-wave and trend ideas from the missed-moonshot review. They were found
by looking at 2021-2026, so testing them on those years would be testing on the answers. P5 adds
only the company's own public facts.

## P5: PASS only if all three hold

Everything as P4 (exits, chooser, ensemble, memory, tradability, costs, controls), plus the dossier:
1. beats random picks under the same exits (one-sided p < 0.05);
2. makes money after costs (one-sided p < 0.05);
3. beats the same machine taught shuffled outcomes (one-sided p < 0.05).

FAIL otherwise. Reported beside it: whether the dossier changed P4's result, and which dossier facts
the learner came to weigh.

**Stated in advance:**
- This is the trader's sixth declared test on overlapping data, so a pass would be weighed against
  six tries.
- The survivorship bias of the 2026-chosen list still applies to every absolute number.
- A company's published accounts are known to everyone who trades it, and are old news by the time
  they are filed. Knowing them is necessary for judging risk (runway, dilution), but they are not
  expected to tell which way the next two weeks go.

## P5 result, recorded 2026-11-05: **FAIL**, two of three checks (`research/paper/P5.json`; every trade with its dossier and post-mortem in `P5_journal.jsonl`)

The first attempt stopped on a sorting bug before producing any result. It was fixed and the test
run unchanged. The changed code reproduces P4's run exactly without the dossier.

300 of 302 weeks traded, 889 trades.

| check, per week (one-sided) | result | |
|---|---|---|
| beats random picks under the same exits | +1.76%, t 3.66, p 0.0001 | held |
| makes money after costs | +1.70%, t 3.16, p 0.0008 | held |
| beats the same machine taught shuffled outcomes | +0.84%, t 1.50, p 0.066 | **failed** |

Declared verdict: **FAIL**. Checked after the fact, where P4 was fragile:

- **Not carried by a few trades.** The edge over random picks was +1.76% a week with all trades,
  +1.19% without its 10 best (p 0.003), and +0.82% without its 20 best (p 0.02).
- **Positive every year:** +0.15%, +1.97%, +1.92%, +2.66%, +0.84% and +3.37% a week, 2021 to 2026.
  P4's edge came only from 2024-26.
- **Six tests.** p 0.0001 × 6 = 0.0006 for the random-picks check.
- **The dossier's own contribution is not proven.** P5 beat P4 by +0.42% a week (t 0.84, p 0.20).
- **The shuffled machine still made money.** The machine taught shuffled outcomes earned about
  +0.86% a week, because its exits and stock memory learn from real outcomes. Part of the edge
  therefore comes from those, and from the stock list, not from learning which stocks will win.
- **Survivorship is still the most likely explanation for the rest.** These 510 stocks were chosen
  by their 2026 size, and the trader leans toward stocks near their highs and toward bigger
  companies, which is what survivors look like in hindsight.

**What it learned to weigh, from the dossier.** Of the dossier facts, "bigger company" was among a
pick's top three reasons 199 times, "latest report is old" 99, "issued many new shares" 30,
"expensive against sales" 21, and "heavy R&D" 20. Bigger companies, an old report, new shares and a
long runway leaned toward a pick; operating profitability and "no revenue" leaned away. The pulls are
small next to the price features, whose top reasons stayed high fear (VIX), near the 52-week high and
a gap up at the open.

**Dossier limits, seen in the journal:**
- A reverse split makes the share count fall and reads as "shares −63% on a year ago" (SBET,
  June 2025), when the company in fact consolidated its shares.
- Runway under one quarter printed as "about 0 quarters"; it now reads "under 1 quarter".
- The dossier has no text: no business description, no risk factors, no going-concern warning, no
  insider trades. Those need each filing's documents, not the financial data.

**As a $1,000 account** (1/6 per trade, half stakes while its last 26 weeks trailed random picks):
$1,000 → **$3,814**, against SPY's $2,206 over the same period. Its worst fall from a peak was −36%.

| 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| $1,000 | $944 | $1,094 | $1,501 | $2,133 | $2,052 |

Inflated by survivorship, as above.

## Where this leaves the trader

Across six declared tests:
- P1 to P3b failed outright;
- P4 and P5 beat random picks but not their shuffled twins;
- P5's edge is the steadiest yet, and the stock list's survivorship can explain it.

The question history cannot settle is whether any of it survives without hindsight. Only a forward
test can answer that: P5's machine running live, on paper, from a list chosen today, with its
dossiers and post-mortems written before each outcome.

# P6: the same machine on a stock list chosen in 2021 (declared before the list was built)

Every historical test so far used stocks chosen by their 2026 size, so their history contains
hindsight: P5's edge could be the machine exploiting that bias. P6 changes nothing in the machine
(P5's code, as committed). It changes only the list, which is chosen with what a trader knew at
the close of 2020-12-31. Written and committed before any price for the new list was fetched.

**The list rule (all inputs dated on or before 2020-12-31):**
1. **Pool:** every company in SEC's current ticker list: the first ticker per CIK, a plain ticker
   (1-5 letters, optionally a class letter), about 7,986 names. Today's list contains companies that
   have since shrunk to almost nothing, so losers are in the pool, not only survivors who grew.
2. **Traded through 2020:** Yahoo daily prices for at least 240 of 2020's sessions, and trades on
   every one of the last 20 sessions of 2020.
3. **Small but tradable then:** a median daily dollar volume from 2020-07-01 to 2020-12-31 between
   $1 million and $20 million, and a close on 2020-12-31 of at least $1.
4. **An operating company then:** a 10-K or 10-Q filed with the SEC before 2021-01-04 (this excludes
   funds and trusts).

**The machine and the test:**
- P5 exactly: exits, exit chooser, ensemble, memory, dossier, tradability check, costs, controls,
  start on 2021-01-04.
- **PASS only if all three hold**, one-sided p < 0.05:
  1. beats random picks from the same list under the same exits;
  2. makes money after costs;
  3. beats the same machine taught shuffled outcomes.

  FAIL otherwise.

**What this does not fix, stated in advance:**
- A company that is no longer on SEC's list or has no Yahoo prices today is missing: those delisted,
  acquired or bankrupt between 2021 and 2026. Bankruptcies missing flatter a long-only trader.
  Acquisitions missing (often a jump of 20%+) count against it.
- The test period is the same one P1-P5 were developed on, and this is the trader's seventh test.
  The stock list is new, and the machine is frozen.

A clean forward test from today, on a list chosen today, is set up alongside it.

# F1: the forward test (declared 2026-11-05, before its first week)

The only test hindsight cannot reach. P5's machine, frozen as committed, trades on paper from the
week of **2026-11-09**, on `universe/us_small.json`, a list built on 2026-10-12, before the start.

```
SEC_USER_AGENT='...' python -m brain paper forward step     # after each close; commit research/paper/forward/
python -m brain paper forward status
```

- **Each week's first session:** its picks (or that it stays out), with each pick's dossier, exit
  rule, predicted net and reasons, are appended to `research/paper/forward/ledger.jsonl`. They are
  written that evening, from data before that morning's open, before any of their outcome exists.
- **Every later session:** each open trade is followed under its own exit rule, and its close is
  appended when it happens.
- **The ledger is append-only:** a week once written is never rewritten. The machine learns each week
  from everything resolved by then, as in the historical runs.
- **The test, read once 30 weeks have fully closed (about June 2027).** It needs all three checks at
  one-sided p < 0.05:
  1. beats random picks from the list under the same exits (computed afterwards by re-running the
     frozen machine over the same weeks, which must reproduce the ledger's picks);
  2. makes money after costs;
  3. beats the machine taught shuffled outcomes.

  Before then, `status` reports the record but claims nothing from it.

## P6 result, recorded 2026-11-05: **FAIL** (`research/paper/P6.json`, `P6_journal.jsonl`)

The frozen P5 machine on `universe/p6_2021.json`: 1,097 small operating companies, chosen from
7,986 using only 2020's data. It covers 302 weeks from 2021-01-04; it traded in 289 of them, 846
trades in all.

| per week (one-sided) | P5, 2026-chosen list | **P6, 2021-chosen list** |
|---|---|---|
| net after costs | +1.70% (p 0.0008) | **+0.17% (t 0.37, p 0.35)** |
| minus random picks, same exits | +1.76% (p 0.0001) | **+0.45% (t 1.04, p 0.15)** |
| minus the shuffled machine | +0.84% (p 0.066) | +0.87% (t 1.33, p 0.09) |
| edge without its 10 best trades | +1.19% | **−0.10%** |
| edge without its 20 best trades | +0.82% | **−0.50%** |
| $1,000 by 2026 (SPY: $2,206) | $3,814 | **$1,123**, worst fall −30% |

None of the three checks held. Edge over random picks by year: +1.23%, +0.85%, −0.28%, +1.85%,
−0.63% and −0.58% a week. Of its trades, 37% won and the median lost 2.2%; "never worked" was the
commonest shape (244 of 846).

**What this says about everything before it.** The same machine, unchanged, lost its edge when the
stock list stopped knowing the future. P5's steady edge was mostly the 2026-chosen list's
survivorship: a list of companies known to have grown into 2026 rewards a trader that leans toward
bigger stocks near their highs. On a list a trader could actually have had in January 2021, the
machine's picks are indistinguishable from random ones, after costs and without its few luckiest
trades.

What remains true:
- It finds stocks that are about to move a lot (the size effect, as S1 and P3b found).
- Its exits and post-mortems describe trades well.
- Nothing in seven declared tests shows that it knows which way a stock will go.

F1, the forward test, keeps running as declared, and will judge the machine on weeks no one has
seen.

# P7: develop on 2021-2026, judge once on 2016-2020 (declared 2026-11-05, before any 2016-2020 data was fetched)

Every machine so far was built and judged on 2021-2026, so an improvement found by studying those
years may only fit them. P7 separates the two:

- **Development (in-sample, iterated freely):** the 2021-2026 weeks on `universe/p6_2021.json`.
  Results there are reported as in-sample and prove nothing.
- **Held-out test (run once):** trading weeks from 2016-01-04 to 2020-12-28 (the last first-session
  whose exits finish in 2020's data), on a list chosen with 2015's data. No moonshot machine has
  seen small-company data before 2021. The machine is frozen at a commit, named in this file, before
  the held-out run, and is never changed after it. Whatever the result, it is recorded.

**The held-out list, by P6's rule shifted back six years:** SEC's current ticker list, first plain
ticker per CIK.
- at least 240 Yahoo sessions in 2015, and trades on each of the last 20 sessions of 2015;
- a median daily dollar volume from 2015-07-01 to 2015-12-31 between $1M and $20M;
- a 2015-12-31 close of at least $1;
- a 10-K or 10-Q filed before 2016-01-04.

Its bias is the same as P6's: companies gone from SEC's list or Yahoo today are missing, and over
10 years more of them are gone. The held-out machine learns from history before 2016-01-04 only
(prices from 2011).

**The held-out test, PASS only if all three hold** (one-sided p < 0.05, weekly, weeks it stays out
count as 0):
1. beats random picks from the list under the same exits;
2. makes money after costs;
3. beats the same machine taught shuffled outcomes.

FAIL otherwise. This is the eighth declared test of a trader here.

## Correction, 2026-11-05: a look-ahead in P4-P6's exit chooser

The exit chooser filtered past trades by comparing a week number with a session number, so every
past week counted, including the last one or two, whose trades had not finished at the decision's
open. The stake rule did the same. Both now count only trades finished before the open (with a test).
- F1's live decisions were not affected: in live mode, every week the chooser used was complete.
- The original results stay recorded above. Re-run with the fix (`research/paper/P4_corrected.json`
  and the like):

| test | as first run | corrected | verdict |
|---|---|---|---|
| P4 | +1.28%/wk (p 0.017); vs random +1.33% (p 0.007) | **+0.65%/wk (p 0.12); vs random +0.76% (p 0.063); vs shuffled +0.10% (p 0.43)** | FAIL, now on all three |
| P5 | +1.70%/wk (p 0.0008); vs random +1.76% (p 0.0001) | **+1.51%/wk (p 0.0027); vs random +1.58% (p 0.0004); vs shuffled +0.70% (p 0.12)** | FAIL, two of three |
| P6 | +0.17%/wk (p 0.35); vs random +0.45% (p 0.15) | **+0.23%/wk (p 0.32); vs random +0.49% (p 0.15)** | FAIL |

About half of P4's edge was the leak. P5's shrank a little, and P6, already flat, barely moved. The
conclusions stand: P5's edge is on the 2026-chosen list only, and P6 shows it does not survive a
list chosen in 2021. With full stakes, $1,000 became $1,716 (P4), $2,966 (P5) and $1,524 (P6),
against SPY's $2,206.

## P7 development (in-sample: 2021-2026 on the 2021-chosen list), recorded 2026-11-05

It learned from 2011-06 onward and traded from 2021-01-04. In-sample, so this proves nothing; it is
here because it decided what happened next.

- **Per week after costs:** −0.10% (p 0.64). Against random picks −0.08%; against the shuffled
  machine −0.17%.
- **Trading:** the gate let it trade 123 of 302 weeks, and none in 2022 or 2026. Its 368 trades
  averaged −0.27% (median −2.6%, 41% winners).
- **Its own validation chose the strongest ridge (300) in 280 of 302 refits:** by its own test, the
  features carry next to no signal.
- **The gate did not help.** The shadow record (its top 3 against random over the last 52 finished
  weeks) was positive 75% of the time, yet the weeks it then traded lost. A good past year of picks
  did not predict a good next week.

**Decision: freeze, do not tune.** Nothing is broken, and the machine's own validation says there is
no signal. Adjusting it until 2021-2026 looked good would be fitting those years. P7 is frozen at
commit **465a14e** (moonshot3.py, moonshot2.py, moonshot.py, dossier.py as of that commit) and run
once on the held-out 2016-2020, as declared.

## P7 held-out result, recorded 2026-11-05: **FAIL** (`research/paper/P7_heldout.json`, `P7_heldout_journal.jsonl`)

Run once, as declared: machine frozen at 465a14e, the 815 companies of `universe/p7_2015.json`,
learning from 2011-06, trading 2016-01-04 to 2020-12-07 (the last week whose exits finished in 2020),
no data after 2020-12-31.

| per week (one-sided) | result |
|---|---|
| net after costs | +0.06% (t 0.23, p 0.41) |
| minus random picks, same exits | −0.06% (t −0.26, p 0.60) |
| minus the shuffled machine | +0.24% (t 0.66, p 0.26) |

None of the three checks held.

- **Trading:** the gate opened in only 36 of 258 weeks (17 in 2016, 19 in 2020, none in 2017-2019).
- **Trades:** its 108 averaged +0.43% (median −4.8%, 34% winners).
- **Validation:** it chose the strongest ridge in 233 of 258 refits.
- **As a $1,000 account:** $1,000 → $1,046, against SPY's $2,054 over the same five years.

**Where eight declared tests leave this trader.** On a stock list chosen without hindsight, in
years it was never developed on, the best machine built here does no better than random picks,
and stays out of the market most weeks because its own record never earns it the right to trade.
P5's edge on the 2026-chosen list was survivorship. P4's was half look-ahead. The parts that hold
up are:
- the size effect (it finds stocks about to move a lot, either way);
- exits that change the shape of the losses;
- dossiers and post-mortems that explain each trade.

None of them predicts direction.

F1 keeps running forward, as declared, on the frozen P5 machine.

# Memory (`memory.py`), and F2: a machine that decides with it (declared 2026-11-05)

**The memory bank.** The trader keeps a committed bank of its own (`research/paper/memory/`, the
repo's `Bank` with its point-in-time `View`). It holds three kinds of memory:
- **episodes:** every trade, known from the session it closed: its result, how the move went, the
  exit that would have been best, its call against the result, its reasons, the influences, the
  filings, the lesson;
- **misses:** every missed moonshot, known from when its move finished, with its causes and where it
  was ranked;
- **beliefs:** what it held, dated.

Recall reads through a View at the decision, so nothing learned later can be recalled earlier.
Backfilled with 843 P6 trades (corrected run), 108 P7 held-out trades and 5,575 missed moonshots from
the P4 review.

```
python -m brain paper memory recall SNEX                    # what it remembers now
python -m brain paper memory recall SNEX --at 2022-01-01    # what it remembered then
```

**F1 now remembers.** Each pick's memory card (recalled as of that morning) is written beside it in
the ledger, and each closed trade is added to the bank. F1's decisions are unchanged: frozen P5.

**P8: the P7 machine deciding with memory** (`moonshot3.simulate(memory=True)`). Before each decision
it recalls, as features:
- the 50 nearest finished situations of the last two years among everything it watched, with their
  mean net and +20% rate;
- its own trades of the stock: how many, and their mean net, shrunk by 3 trades' worth.

With memory off, the code reproduces the frozen P7 exactly.

**F2, declared before its first week.** P8 trades forward on paper from the week of 2026-11-09, on
`universe/us_small.json`, beside F1, with the same ledger rules (`research/paper/forward2/`), the same
exit-following, the same memory bank, and the same daily schedule.
- It studies everything back to 2011 before each decision.
- It may stay out, by its gate.
- **Test, read once 30 weeks have fully closed (about June 2027).** It needs all three checks at
  one-sided p < 0.05: beats random picks under the same exits; makes money after costs; beats the
  machine taught shuffled outcomes (both computed afterwards by re-running the frozen code over the
  same weeks, which must reproduce the ledger).

The held-out years 2016-2020 have now been used once, by P7, so they are no longer a clean test for
P8. F2 is P8's test.

## P8 development (in-sample, 2021-2026 on the 2021-chosen list), and F2's dry run, 2026-11-05

**P8 development.** With memory, P8 did worse in-sample than P7 without it:
- −0.27% a week after costs (p 0.88), against P7's −0.10%;
- against random picks −0.31% (P7 −0.08%);
- 231 trades in 79 weeks, mean −0.91% (median −3.2%, 38% winners).

Memory was used ("it has traded this stock before" was among a pick's top reasons 37 times, "this
stock often got stopped this year" 48), but recalling situations like the present one did not pick
better stocks. Like every other feature here, the past cases it recalls do not predict direction.
In-sample, this proves nothing either way. F2 stands as declared and is P8's only test.

**F2's dry run.** The real forward list, real inputs, P8 in live mode, no ledger written: inputs in
6.6 minutes, the machine in 5. On the latest weeks (2026-10-19 to 2026-11-02) its gate was shut
(shadow record +0.3% to +0.5% a week, t 0.3 to 0.5, under the bar of 1). F2 will likely begin by
staying out, as declared, until its record earns trades.

# Attention (`attention.py`), and F3: the machine with attention over memory (declared 2026-11-05)

**What attention adds.** P8 recalled the 50 nearest past situations by a fixed distance, every
feature counting the same. Attention learns what "near" should mean: one scaled dot-product head,
query and key projections (8 dimensions) learned by gradient descent.
- **Values:** how each remembered situation turned out, its net under each exit and whether it hit
  +20%.
- **Training:** only on finished situations. Queries are from the latest 26 finished weeks, keys at
  least 3 weeks older still, and the latest fifth of query weeks is held out to stop training at
  its best.
- **Retraining:** every 13 weeks.
- **At a decision:** it attends over everything finished before that open, and each pick names the
  past situations it attended to most.
- **Sharpness is bounded** (each projection's norm is held at 6 or under). A first version put 91% of
  one pick's attention on a single week of March 2020: matching an outlier, not learning from many
  cases. The bound was chosen on a synthetic problem with a planted answer, never on market data.
- **Tests:** with the answer planted in one feature, attention learns to look at that feature (its
  recall correlates 0.83 with the truth, where uniform attention gives about 0). It never attends to
  a situation that had not finished by the decision.

**P9 = P8 + attention** (`moonshot3.simulate(memory=True, attention=True)`): attention's recall
(each exit's net, the +20% rate) enters as six features. With attention off, the code reproduces P7
and P8 exactly.

**F3, declared before its first week.** P9 trades forward on paper from the week of 2026-11-09, on
`universe/us_small.json`, beside F1 and F2, with the same ledger rules (`research/paper/forward3/`),
exit-following, memory bank, daily schedule, and test. That test is read once 30 weeks have fully
closed, and needs all three checks at one-sided p < 0.05. P9's in-sample development run on
2021-2026 is reported when it finishes, and proves nothing either way.
