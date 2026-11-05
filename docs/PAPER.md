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
