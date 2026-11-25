# Earnings: what was expected, what followed, and why

Asked for on 2026-11-25: "to predict the future of something, you need to know the past and present, not just what
but how and why too. That's what we are missing." The person asked for this to start with results reports.

Every failed direction test in this repo asked "this happened, so what comes next?" (news then a drift, oil then
airlines, a wave then the next day). The market prices the "what" within the day. A stock moves on the gap between
what happens and what was expected, and the brain kept no record of what was expected. That week showed it:
- HPE, Workday, Nutanix and Semtech beat estimates and fell.
- Dell fell 12% the session before its report.
- Ambarella fell 20% in the session before its own bad news.

## The past: the episode library (`brain/episodes.py`)

```
python -m brain expect build          # every report of the four universes, about 5 minutes (var/episodes/)
```

**What an episode is.** One earnings release: an 8-K item 2.02 from 2011 on, for the research, small, micro and
newer-listing universes, about 107,000 in all. Its setup is what the price held at the close before the report,
when a trade "before the report" is made:

| Field | What it measures |
|---|---|
| `run5`, `run21`, `run63` | how far the stock ran against the market over 5, 21 and 63 sessions, in its own usual moves for that span |
| `hi52` | how far below its 52-week high it sits |
| `med8`, `last` | the size of its usual report move (the median of its last eight) and of its last one |
| `mean8`, `up8` | how it did at its last eight reports: the mean signed move, and the share that rose |
| `vix`, `spy21` | the market's mood: the VIX, and SPY over the last month |

**What happened.**
- `r1`: the move against the market from the close before the report to the report session's close.
- `car3`: days -1..+1.
- `post5` and `post21`: the drift over the next week and month.

**Why, in a fixed form.** The setup is "ran up", "sold off" or "flat"; the reaction is "rose", "fell" or none.
Together they make the episode's name:

| Name | Setup | Reaction | What it means |
|---|---|---|---|
| priced in | ran up | fell | the good news was already in the price |
| relief | sold off | rose | the fear was already in the price |
| confirmed | ran up | rose | |
| capitulation | sold off | fell | |
| surprise up, surprise down | flat | rose, fell | |
| no reaction | | under a quarter of its usual move | |

**Limits.**
- There is no point-in-time consensus history in the bank, and no history of option prices, so the expectation is
  the price's own: the run-up, the stock's habit at its reports, the mood.
- The press-release text was lost in a reset of the machine, so past episodes carry no guidance tags yet. Re-reading
  releases from SEC (`brain releases`) would add them.

## On TRAIN years (2011-2023), before any test

`research/earnings/explore_train.py`; its logs are beside it. The monthly rank correlation of each part of the setup
with `r1`, over 154 months:

| Setup part | Correlation with `r1` | t |
|---|---|---|
| run-up, week (`run5`) | -0.038 | -6.4 |
| run-up, month (`run21`) | -0.038 | -6.4 |
| run-up, quarter (`run63`) | -0.030 | |
| habit, share that rose (`up8`) | +0.021 | +3.4 |
| VIX, the market's month | about 0 | |

**Run-up.** The further a stock ran into its report, the worse it did on the report session. The top fifth by
`run21` averaged -0.01% against the market, the second fifth +0.41%.

**Habit.** Stocks that usually rise at their reports kept doing so.

**After the reaction.** The reaction kept going a little over the next month (+0.023, t +3.5), as the post-earnings
drift is known to. That is the idea H7b tested monthly, and it FAILED on 2019-2026. Over the next week the
continuation was weaker (+0.012).

**The expectation score.** Judged on 2019-2023 from 2011-2018, still inside TRAIN:
- The run-up alone gave +0.034.
- The habit alone gave +0.028.
- The 100 most similar past setups gave +0.022.
- The expectation score, `z(-run21) + z(-run5) + z(up8)` (equal weights, nothing fitted), gave +0.050 on 2011-2018
  and +0.052 on 2019-2023.

These are small numbers. The difference between the top and bottom fifths is a few tenths of a percent per report,
and a one-session trade around a report pays its spread twice. Whether this is an edge is for the declared test.

## The declared tests (`brain/research/earntest.py`)

```
python -m brain earntest declare      # after committing the code
python -m brain earntest test         # once; then commit the ledger and research/RESULTS.md, whatever they say
```

TEST is 2024-01..2026-09.

| Test | Prediction | Outcome |
|---|---|---|
| X1a (priced in) | `-run21` | `r1` |
| X1b (habit) | `up8` | `r1` |
| X1c (analogs) | the mean `r1` of the 100 nearest earlier episodes | `r1` |
| X1d (after the reaction) | the reaction in its usual moves | `post5` |
| X1e (reported, not gated) | TEST episodes by name, with what followed | |
| X1f (expectation score) | the expectation score | `r1` |

**How a test passes.**
- The statistic is the monthly rank correlation, averaged over months, and must be positive.
- A one-sided sign-flip p over the months must be under 0.05.
- Placebo: shuffling within a month must reach p < 0.05 in at most 10% of draws.
- Oracle: knowing the outcome must reach p < 0.01.
- Benjamini–Hochberg q must be under 0.10 across every gated test in the ledger.

The results are in `research/RESULTS.md`, section "Earnings: expected and what followed".

## The present: before a report (`brain/expect.py`)

```
python -m brain expect show DELL CRM            # the card
python -m brain expect record CRM --report 2026-12-02   # write it down before the report; judged on r1
python -m brain expect score                    # the forward record (says nothing under 30 weeks)
```

The card shows:
- The stock's setup now, in words.
- Its expectation score and where that falls among 2011-2023's reports, with how that fifth reacted.
- The 100 most similar past setups (any stock, before today): how often they rose, how far they moved, their names,
  and the week after, split by whether they rose or fell.
- The stock's own last reports, by name.
- The declared tests' verdicts as recorded.

A score or a lean whose test has not passed is untested. Neither is a call until it is recorded, here or with
`brain predict add`.

**In the desk's case file.** Each mover with a results report since the close gets an `[e:TICKER]` line: its setup
before the report and how the most similar past setups reacted, to cite in the note's reasoning.
