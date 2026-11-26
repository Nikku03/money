# Promises: what a company said it would deliver, and whether it did

Asked for on 2026-11-25: "if we know the company in and out, how it works ... its next projects and what it need,
and according to timeline, trace every fabric of the company, we can tell how much stock will move, and if it can
deliver, and all the other stuff it failed on."

The first promise a company makes in public, on a date, with a number, is its revenue guidance: "we expect revenue
of $70 million to $75 million next quarter". The next report says whether it delivered. This is the first layer of
the company file: a record, per company, of what it said and what it did.

```
python -m brain releases --broad          # read every earnings release with its forward-looking passages (hours)
python -m brain promises build            # the library, from the bank (var/promises/, under a minute)
python -m brain promises show NVDA AMBA   # a company's card
python -m brain promisetest declare|test  # the declared tests G1a, G1b (the research universe)
python -m brain promisetest2 declare|test # G2a-G2d (the small, micro and newer-listing universes)
```

## The promise (`brain/promises.py`)

Reader 3 of `brain/sources/releases.py` keeps, beside each release's opening, its forward-looking passages: the
text around words like "outlook", "guidance", "expects" when a figure stands beside them. Fixed rules, no model,
read those for revenue guidance in dollars:

| Form | Example |
|---|---|
| a range | "revenue in the range of $555 million to $565 million" |
| a midpoint with a tolerance | "revenue of $2.70 billion, plus or minus $100 million", "± 5%" |
| a single figure | "expects 2023 revenue to be $42.7 billion" |

- A figure counts only when a forward word governs it and nothing in between says it already happened.
- Its horizon is the next quarter or the fiscal year, by the words beside it and by its size against the company's
  own revenue. A figure the size of a part of the company (a segment, one product) is dropped.
- A pre-announcement of the quarter just ended is not a promise.

**Limits.**
- Only revenue, only in dollars. Guidance given as a growth rate ("up 5% to 7%"), as earnings per share, or in a
  table without units is not read. Many large companies give none.
- The rules were tuned by reading promises from every year, 2024-2026 included, for accuracy of reading. No
  relation between a record and an outcome was looked at on those years before the tests ran.
- Early years are thin: SEC's list of a company's filings reaches back unevenly, so most promises are from 2017 on.

## The delivery

- A quarter's promise is held against the revenue reported for the first fiscal quarter ending after the release;
  a year's against the first fiscal year ending after it (SEC XBRL).
- The figure is the one **first** reported, not a later restatement. A company that splits in two restates its
  past without the part it sold, which would turn every old promise into a miss.
- `surprise` is the delivery over the promise's midpoint, less one. `beat` is above the top of the range, `missed`
  below the bottom, else `met`.
- A delivery further than -30% or +50% from the promise is a misread or a bad account figure: kept as a mismatch,
  not graded. That removes real disasters too, so the record understates the worst misses.
- An outcome is known when the accounts were filed, never before.

## The record

At any date, a company's record is its last eight quarterly promises whose outcomes were known by then (at least
four): how many it kept, how many it beat, and its usual surprise (the median).

Its **habit** is the median surprise of its last four quarterly deliveries, the quarter it has just reported
included. On 2011-2023 this said the most about the next delivery (below), and it is what G1a tests.

- `stretch` says how much a new promise asks: its midpoint's growth over the year-ago quarter, less the growth the
  company last reported.
- A yearly promise is compared with the company's previous promise for the same fiscal year: raised, cut or held
  (a move of 1% or more).

## On TRAIN years (2011-2023), before any test

`research/promises/explore_train.py` and `explore_habit_train.py`; their logs are beside them. The research
universe: 5,576 promises of 266 of its 741 companies. Each number is a rank correlation taken inside a reporting
season (a calendar quarter) and averaged over the seasons.

**Companies promise low.** Of 1,863 graded quarterly promises, 78% were delivered above the midpoint, by a median
+1.85%; 8.9% fell below the bottom of the range.

**What says how the next promise will go.**

| Known when the promise is made | Against the surprise on it | t |
|---|---|---|
| the median surprise of its last four deliveries, the latest included (the **habit**) | +0.315 | +11.7 |
| the surprise it has just delivered | +0.291 | +8.7 |
| the median of its last eight outcomes on file | +0.155 | +5.7 |
| the stretch (less is better) | +0.071 | +2.2 |

The habit also came closest to the delivery: a median 1.96% off, against 2.55% for the promise's midpoint alone.

**The stock, on the day the promised quarter is reported.** The record says nothing: the habit +0.004, the share of
promises it beat +0.021. The delivery itself, known only afterwards, +0.262.

**The stock, on the day of a release (the same day: an explanation, not a forecast).**

| What the release said | Against the report session's move | t |
|---|---|---|
| what it delivered against its last promise | +0.237 | +9.3 |
| that, less the habit the promise was made under | +0.213 | +8.9 |
| the change in its promise for the year | +0.229 | +7.2 |
| the stretch of its new promise | +0.027 | +1.0 |

**The stock, after a release.** Weak: the change in the year's promise against the next five sessions +0.054
(t +1.5; -0.056 on 2011-2018, +0.102 on 2019-2023). Nothing for the stretch or the delivery.

## The declared tests (`brain/research/promisetest.py`)

```
python -m brain promisetest declare      # after committing the code
python -m brain promisetest test         # once; then commit the ledger and research/RESULTS.md, whatever they say
```

TEST is promises made 2024-01..2026-09. The statistic is the season's rank correlation, averaged over seasons, with
a one-sided sign-flip p over the seasons, a placebo, an oracle, and Benjamini–Hochberg q < 0.10 across every gated
test in the ledger.

Run once on 2026-11-25 (`research/RESULTS.md`, section "Promises: said and delivered"):

| Test | 2024-01..2026-09 | p | BH q | Verdict |
|---|---|---|---|---|
| G1a the habit carries to the next promise | +0.474, positive in 11 of 11 seasons, 670 promises | 0.0007 | 0.003 | **PASS** |
| G1b the report session's move by what the release said | reported, not gated | | | REPORT |

**What G1a's pass means.**
- A company's own number is not the best guess of what it will deliver. Its number moved by its habit is: the
  delivery was a median 2.41% from the midpoint and 1.43% from the midpoint moved by the habit.
- The fifth of promises with the highest habit was delivered a median +5.13% above the midpoint; the lowest fifth
  +0.81%.
- The ranges fixed on 2011-2023 were wider than 2024-2026 needed: the 2-in-3 range held 77.3% of deliveries and the
  9-in-10 range 95.7% (reported, not gated). They are cautious, not calibrated.
- Only 3.9% of these promises fell below the company's own range.

**What it does not mean.**
- It is about what a company delivers, not about its stock. On TRAIN the habit said nothing about the reaction on
  the day the quarter is reported, and that was not declared.
- It covers a part of the market: 106 of the 741 companies made a quarterly revenue promise in dollars on
  2024-2026, and 195 made a quarterly or a yearly one.
- It leaves out the worst cases: 15 of the 765 quarterly promises on 2024-2026 (2%) were mismatches, not graded.
  Some are misreads, some are companies that changed shape (GE, Western Digital and Honeywell split; Dollar Tree
  sold Family Dollar).

**G1b: the day of a release, by what it said (2024-2026, the same day).**

| The release | Releases | Report session, against the market | Rose | The week after |
|---|---|---|---|---|
| delivered above the top of its last promise | 285 | +2.21% | 61% | +0.73% |
| delivered inside it | 355 | -0.23% | 48% | +0.25% |
| delivered below it | 28 | -6.30% (median -2.86%) | 18% | +1.45% |
| delivered more against its midpoint than its habit | 359 | +2.35% | 60% | +0.74% |
| delivered less than its habit | 277 | -2.04% | 40% | +0.19% |
| raised the year's promise | 199 | +3.29% | 62% | +0.09% |
| held it | 333 | +0.22% | 50% | -0.37% |
| cut it | 62 | -3.70% | 34% | +1.65% |

- **Why a company can keep its promise and fall.** Keeping the promise is the normal case, so it is already in the
  price. What moves the stock is the delivery against the habit, and the new promise.
- These are averages of the same day's move, known only once the release is out. They explain a move. They are not
  a forecast, and the "week after" column is not tested (after a cut it was up, not down).

**Held back at first.** The stretch against the delivery and the change of the year's promise against the next
five sessions had too few cases on this universe (eleven seasons of about 65 quarterly promises): a FAIL would
have said nothing. They were tested on the other universes, below.

## The other universes: G2a-G2d (`brain/research/promisetest2.py`)

```
python -m brain releases --band small     # then --band micro, then the tickers of universe/us_watch.json (hours)
python -m brain promises build            # every universe: 19,547 promises of 1,016 companies
python -m brain promisetest2 declare|test
```

The small, micro and newer-listing universes together (1,796 companies, none of the research universe), whose
releases were re-read on 2026-11-25/26. The research universe is left out because G1b had printed its week after
a release.

**On promises made 2011-2023** (`research/promises/explore_bands_train.py` and its log; 3,993 quarterly promises of
405 companies):

| | Small | Micro | Newer listings | All three |
|---|---|---|---|---|
| quarterly promises | 1,166 | 673 | 2,154 | 3,993 |
| delivered above the midpoint | 66% | 68% | 84% | 76% |
| missed their own range | 15.9% | 17.9% | 6.6% | 11.1% |
| the habit against the next surprise | +0.307 | too few a season | +0.415 | +0.351 (40 of 40 seasons) |
| the stretch against the surprise less the habit | +0.110 | | +0.174 | +0.150 |
| the delivery against the habit, against the next five sessions | | | | +0.074 (t +3.0) |
| the change in the year's promise, against the next five sessions | +0.024 | | +0.070 | +0.059 (t +2.3) |

- Small and micro companies miss their own range about twice as often as large ones (16% to 18% against 8.9%).
- Beyond the habit, a promise that asks for more was beaten by more, in every universe. The first reading ("less
  is better", +0.071) came from the habit itself: companies that beat by a lot guide low.

**Run once on 2026-11-26** (`research/RESULTS.md`), promises made 2024-01..2026-09:

| Test | 2024-01..2026-09 | p | BH q | Verdict |
|---|---|---|---|---|
| G2a the habit carries to the next promise | +0.486, positive in 11 of 11 seasons, 1,357 promises of 299 companies | 0.0006 | 0.003 | **PASS** |
| G2b beyond the habit, asking for more is beaten by more | +0.096, 8 of 11 seasons | 0.031 | 0.12 | **FAIL** |
| G2c the next five sessions follow the delivery against the habit | -0.000, 7 of 11 seasons | 0.50 | 0.73 | **FAIL** |
| G2d the next five sessions follow the change in the year's promise | +0.034, 7 of 11 seasons | 0.11 | 0.30 | **FAIL** |

- **G2a.** The habit carries on other companies as it did on the research universe (+0.474 there). The fifth of
  promises with the highest habit was delivered a median +4.56% above the midpoint, the lowest fifth +0.78%. The
  delivery was a median 2.34% from the midpoint and 1.37% from the midpoint moved by the habit. The ranges held
  77.2% and 95.1% (cautious, as before); 4.6% missed the company's own range.
- **G2b.** The sign held (the fifth asking for most delivered +0.29% beyond its habit, the fifth asking for least
  -0.39%) and p was 0.031, but it did not clear the false-discovery bar across the ledger's tests. Not a finding.
  What can be said: nothing here shows that a promise asking for more falls short.
- **G2c, G2d.** The stock does not follow in the week after. After a raise the five sessions gave +0.31% against
  the market (49% rose), after a cut -0.92% (44% rose), held -0.57%: the order is as expected and too weak to
  pass. A release is priced on its session (G1b). Do not present a raise, a cut or a beat as a direction for the
  days after.

## The card and the desk

```
python -m brain promises show NVDA
```

The card gives the company's record (its last eight quarterly promises on file: kept, beat, missed), its open
promises with how much growth each asks for, where its habit puts the delivery (with the two ranges), its last
graded promises, and G1a's verdict as recorded. The before-the-report card (`brain expect show`, `docs/EARNINGS.md`)
carries the same lines.

- In the days between a release and the accounts of the quarter it reports, the habit is one delivery behind: the
  bank takes the delivered revenue from the accounts, not from the release.
- Quote the card as printed. A delivery is not a direction for the stock.

**In the desk's case file** (`docs/DESK.md`): each mover with a results report since the close gets a
`[g:TICKER]` line: what it had promised for the quarter it has just reported, its record and where its habit put
the delivery, its new promise and how much that asks for, and whether the year's promise was raised or cut. The
figures are read by fixed rules: check them against the release before quoting them.

## How well the reader reads

- A spot-check of 16 promises drawn at random from 2015-2023: 13 read correctly, 2 misreads caught by the mismatch
  rule, 1 misread graded (a past quarter's revenue, "4Q19 revenue of $1.2 billion", taken as a promise). A rule
  added after it (a quarter named with its year that is already over is a result) removed 25 such rows from
  2011-2023 and no real promise among those checked.
- The largest misses it finds are known ones: Seagate and AMD in 2022, Pfizer's 2023 cut, Apple at the start of
  COVID.

## What this is the first layer of

The person's direction is a file on each company: how it works, what it promised, what it needs, who it trades
with, and its dates, so that what it can deliver can be judged. This layer is the promise and the delivery for
revenue. Not built yet: guidance given as growth rates or earnings per share; what a company needs against what it
has (cash against planned spending); a timeline of its known dates; its earnings calls.
