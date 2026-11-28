# Memory: where the work stands

Read this first in every session. It is the agent's notebook for this repo: what the person wants, what exists, what
was found, what is running and what is open. Details live in the files it points to; search them when needed.
Updated 2026-11-16 (three sessions in the memory bank; the diagnosis; step 2 tested and failed). The log of what happened, by date, is `memory/JOURNAL.md`.

## Keeping this alive

- This file is the always-read page of the memory bank (`docs/MEMORY.md`). The rest is searched:
  `python -m brain memory recall "..."` over every turn of the conversation since 2026-11-10 (the person's
  messages, replies, reasoning, tool calls and results, compaction summaries), the journal, kept facts, the docs,
  the research notes and the desk notes, ranked by words and by meaning (a small embedding model, `pip install
  fastembed`, approved by the person 2026-11-15). Start a session with `python -m brain memory start`.
- The talk holds three sessions (about 14,450 passages, every one embedded for search by meaning since
  2026-11-16). This one (`session_018E1bTF9g2P9K8brvhcocRv`, word for word since 2026-11-10,
  summaries before) and `session_01MSaQVTPoBwSsjdSBSk9kdd` ("Biological neuron game with dimensional
  connections", 2026-11-04 to 11-09, imported 2026-11-15). That one began with a game for grown neurons and became
  the self-improving paper traders F1-F5 (with memory and attention, "start the account from $1000", "don't
  cheat", "wire in all") and the weekly options checks V1/O1; it lives on branch `claude/sleepy-goldberg-2ma18z`
  and its scheduled tasks still run there. And `session_018ANuE3nDWJsR8htWjJDJhG` ("High variability trading ML
  model", 2026-07-29 to 10-21, 9,212 passages, 160 of the person's messages): where the trading work began ("a
  safe high variability trading ML model" following catalysts; moonshots of 10%+, about 5 trades a week; options;
  a Markov chain as a mass-emotion detector; trial results calendars). `memory recall` searches all three.
- Copies live in the Google Drive folder `money-brain memory`, one file per save, named with the date; the newest
  is current. One upload from here carries about 3 MB, so `save` writes larger files in parts. Newest full set
  (2026-11-28): `MEMORY`, `JOURNAL`, `claims`, `notes`, `talk 2026-11-28 1of2/2of2.jsonl.gz` and `vectors 2026-11-28
  1of7..7of7.npz`, all uploaded by file reference. After a reset, download the newest `talk` and every `vectors`
  part of the same date and run `memory load --talk FILE ... --vectors FILE ...`; whatever has no vector is
  embedded again by `memory index` (about two hours for everything), and recall
  by words works meanwhile.
- After meaningful work: rewrite what changed here, add to `JOURNAL.md`, `memory save`, commit, push, upload.
- The local home's market data (2.7 GB of SEC data, 3.1 GB of prices) is public and rebuilt by commands
  (`memory status` lists them). Yahoo prices are never committed or uploaded.

## The person

- Not technical. Wants plain, short answers; says "in simple" when an answer is too dense.
- Asks big, open goals ("make an all around mosaic") and expects them built, tested and running.
- Wants the honest result, including when an idea fails. Wants things to run by themselves (scheduled checks).
- The aim they stated (2026-11-13): use public information to find what only insiders know. Legal "mosaic"
  inference from public pieces only; no inside information, ever.
- Their email is private: never put it in a request, a file or a commit.

## Ground rules for this session's work

- Repo `Nikku03/money`, branch `claude/upbeat-turing-z9jk2b`. Commit and push there; no pull request unless asked.
  End commits with the attribution lines the session gives.
- `export SEC_USER_AGENT="money-brain research admin@example.com"` (generic; never the person's email).
- Everything is paper. Declare a test and commit it before running it; never edit a declared test. Record any call
  to the person (`predict add`, or a desk note's calls) before its outcome.
- Size calls only when they disagree with what the options already price (promised 2026-11-13: repeating the
  options' priced move is no surprise).
- `CLAUDE.md` holds the full rules; `docs/` explains each part.

## What exists (one line each; the doc has the rest)

| Part | Command | Doc |
|---|---|---|
| Bank of public facts | `ingest`, `brief`, `note` | `README.md` |
| Weekly ranges W1 + 20%-day chance S1 for ~2,450 stocks | `scan run --refresh --record` | `docs/SCAN.md` |
| Between-scan checks with "why it moved" | `scan watch --record` | `docs/SCAN.md` |
| Earnings move size M1 against options | `moves predict`, `earnings radar` | `docs/MOVES.md` |
| Shock radar S1; S3 with earnings habit | `shocks radar`, `catalysts fit` | `docs/SHOCKS.md` |
| Desk: case file + reasoned note held to rules | `desk build`, `desk note`, `desk score` | `docs/DESK.md` |
| Partners, drivers, stories (connections) | used by the desk | `brain/connect.py` |
| Mosaic: share sale, late accounts, cuts | `mosaic read`, `mosaic show`, `mosaic radar --record`, `mosaic score` | `docs/MOSAIC.md` |
| Trader replays | `scan trader [--one-trade]` | `research/SCAN-TRADER-2026-11.md` |
| Memory bank: recall anything said or done | `memory start`, `memory recall`, `memory remember`, `memory save`, `memory load` | `docs/MEMORY.md` |
| Earnings: what was expected before a report, the past like it, what followed and why | `expect build/show/radar/record/score`, `earntest` | `docs/EARNINGS.md` |
| Promises: what a company said it would deliver (revenue guidance) and whether it did | `releases --broad/--band`, `promises build/show`, `promisetest` | `docs/PROMISES.md` |
| Spiking neurons (side project, not the market): hand-wired Pong/Catch/Runner, school; cells lif/izh/hh | `neurons play/catch/run --cells`, `neurons ladder`, `neurons exchange`, `neurons train [--no-delays]` | `docs/NEURONS.md` |

## What was found (verdicts in `research/RESULTS.md`)

- **PASS (sizes and events):** W1a, W1b, W1c (the weekly ranges hold their odds); S1a (shock radar ranks
  surprises); M1a (earnings move size); P1a, P2a, P3a (the mosaic sees share sales, late accounts and cuts coming,
  better than each company's habit, and calibrated).
- **PASS, the first on which way (2026-11-25): X1a.** The further a stock runs into its results report, the more
  often it falls on the report session (2024-2026 monthly rank IC +0.029, p 0.003, q 0.013). The typical reaction,
  not the average: the fifth that ran up most rose 49% of the time, median -0.13%, mean +0.49% on big winners. A
  tilt in the odds, not a trade. After the fact it is mostly the everyday pullback: ordinary days of the same
  stocks showed +0.037 on 2024-2026 (report days +0.029). X1b habit, X1c analogs, X1d week after, X1f score
  (p 0.052) FAILED.
- **FAIL (every direction test):** H1-H8 (momentum, value, insider buying, earnings surprise), L1 (cause → effect
  links next month), E1/E2 (news drift), S2a/S2b (fade after a jump), S3a (calibration), P1b-P3b (mosaic-flagged
  stocks did not lag). No test of which way a stock goes has passed.
- **FAIL (2026-11-26/27), more direction tests:** C1a, C1b (a customer's results and its supplier's stock),
  G2c, G2d (the week after a release), G3a (what a promise asks for, against the report day that delivers it:
  +0.038 on 2024-2026, p 0.115). 71 verdicts in the ledger, 11 PASS.
- **The price knows (2011-2023, not declared):** before a report the stock's own run says how the delivery
  will come against the habit (+0.15) and says nothing about the report session. The market is ahead of the
  report; what is public is in the price.
- **Exploratory, not declared:** partners' results and late reports say almost nothing about a stock's own report
  (`research/MOSAIC-2026-11.md`); smart-money footprints (politicians, 13D copying, volume) did not pay
  (`research/HISTORY-SMART-MONEY-2026-11.md`, `research/PRECURSORS-2026-11.md`); dynamic trading and trading less
  did not beat a fixed weekly reading (`research/SCAN-TRADER-2026-11.md`).
- **Options:** my size calls on NVDA, AMD, DELL matched what options already priced: no edge.

## The diagnosis (2026-11-16, `research/POSTMORTEM-2026-11.md`)

Asked: "why do we keep failing and how do we make it the best strategy?" Read all three sessions. About seventy
ideas run to a verdict; the eight that passed are all about size or which filing comes; every direction or profit
test failed. Five causes: (1) the question itself, direction of liquid stocks from public news; (2) the bet is
priced too (options, spreads and binary odds charge what we know; costs take the rest); (3) each win was an
artifact (survivor lists, look-ahead, small samples, best-of-many, no costs); (4) hunting where odds are worst
(moonshots, the most volatile stocks, weekly options); (5) hopping between ideas before any forward test matures.
What survived: sizes, events, "boring beats exciting", buying after fear spikes, expensive stocks reversing into
earnings (55.5%, weak), size forecast against option prices (V1/O1 running), and one researched drug readout
entered before it (KOD +174%, n=1). Best design: index core; tilt only with what held, declared once against SPY;
size engine for risk and option choice; one specialist line of binary events entered before, small and many; one
idea at a time, checked against the memory bank first.

## Step 2 tested (2026-11-16, `docs/TILT.md`): both FAILED

- The person said "start with step 2". Declared once each and run: T1a, the quality tilt (profitable, not diluting,
  calm, near its high; top third of the 500 largest by market value known each month) lost to its own list on
  2019-2026 (-0.095% a month, p 0.85) after winning on 2011-2018 (p 0.006). T1b, SPY at 100% for 13 weeks after a
  VIX spike (1.25x its 63-day mean) and 70% otherwise, lost on 1993-2010, never looked at before (p 0.88); it won
  only on 2011-2026, the years the lesson came from. Neither beat SPY (2019-2026: SPY +15.8% a year; the best
  variant, the index by market value without its bottom third, +15.6%).
- "Boring beats exciting" and "buy after fear" were features of their era. What is left of the plan: index core,
  size engine, the binary-event line, forward records. One lead, not a finding: "not diluting" alone +0.17% a
  month (p 0.001), one of four traits looked at after the run; it needs its own forward test.
- Known gap in our lists: companies with several share classes (Berkshire, Visa, Alphabet before its count is read)
  get no market value, and Exxon's accounts are never read, so a list by market value is not the S&P 500.

## Running now

- Scheduled tasks firing into this session (New York time, weekdays): `trig_01Jzn1R7gdNyBDpUPN1rnxWo` 8:07 before
  the open; `trig_01Bdk1YEDujykdhnHVkEYCTG` 12:13 midday; `trig_01UzoxK1NLyqQyoZUu1nmgRK` 22:23 after the close.
  Each runs the scan or the check, builds the desk's case file and writes a reasoned note.
- Two other scheduled tasks belong to another branch (`claude/sleepy-goldberg-2ma18z`: V1/O1 weekly, F1-F5
  daily). Do not touch them.

## Open records (judged later; quote their score commands before claiming anything)

- Desk calls (`research/desk_calls.jsonl`), due 2026-11-20: NVDA moves 7.5%+ (chance 0.50), AMD 9%+ (0.42),
  DELL 9.5%+ (0.40). `desk score`.
- Mosaic calls (`research/mosaic_calls.jsonl`): 75, the top 25 of each picture at the close of 2026-11-13, judged
  after 20 sessions (mid-December). `mosaic score`.
- Against the analysts (`research/against_calls.jsonl`): the person asked 2026-11-27 "ok, track it and tell me
  after the reports". 9 lines recorded 2026-11-27 morning (BOX, OKTA, NTAP, ZS, FIVE, VEEV, LULU, AVGO, PLAB), more
  that evening and on Nov 28 (reporters Nov 30-Dec 17 with an open promise and a habit). Scheduled tasks fire into
  this session: `trig_01UaQT7NqEhRekAZsLthoGKc` Nov 28 15:07Z (fetch and record ADBE, PL, JBL, MU and any missed);
  readouts `trig_01KqTfqo9a7BV2DbiGyYn4K1` Dec 4 21:47Z, `trig_017xyVhRi6R3VesPZshHrHwM` Dec 11 21:47Z,
  `trig_01XPfpqjAH9qnMberV6vTmEM` Dec 18 21:47Z. Judging needs the quarter's accounts in the bank: `brain ingest T
  --skip filings news prices` per ticker, then `promises build` with NO tickers (with tickers it overwrites the
  library), then `against score`. Accounts reach the bank 0-15 days after a release; until then compare with the
  release's revenue by hand, labelled.
- S1 calls (`research/s1_calls.jsonl`, 27), predictions (`research/predictions.jsonl`, 31), M1 calls (99).

## The neuron ladder (2026-11-20, at the person's request; `docs/NEURONS.md`)

- The person shared He et al. 2024 (Nat Comput Sci, doi 10.1038/s43588-024-00674-9: rich Hodgkin-Huxley cells
  vs more simple LIF cells) and asked to compare it with "our brain model" (the spiking circuits, not the
  market), then to "add the missing parts but keep the good things". Brought `brain/neurons` over from
  `claude/sleepy-goldberg-2ma18z` unchanged and added HH and LIF cells beside Izhikevich (default; every old
  result reproduces exactly), an exchange-rate measurement and timing-task training.
- Found: HH caught 87% in hand-wired Catch (Izh 75%, LIF 28%); Izh alone has Runner at 0 crashes (wiring tuned
  on it). Copying one HH cell: LIF x1 0.34, Izh 0.47, LIF x2 0.53, x4 0.50, x8 0.53 (Gamma); none close.
  Trained on timing: HH and LIF need the same number of cells (n=4: 86%/89%), Izh a few points behind;
  trained delays add 0-5 points for every cell. HH costs 5-10x CPU. The paper's claim did not show up here.

## The pond and its ripples (2026-11-22, at the person's request; `docs/RIPPLES.md`)

- The person's picture: news is a stone in a pond, its wave travels, hits or is absorbed, waves interfere. Built
  `brain/ripple.py` (joint reading of the day's stones on a year of market-adjusted returns, calibrated KAPPA 0.24 on
  2012-2016) and `ripple map`; declared R1a-R1d (78f4283) and ran them once on 2024-2026.
- Result: on the day a wave reaches one ring (first ring 0.747 usual days against 0.593 on dry land) and is mostly
  absorbed (58% absorbed, 42% felt, 0.06% a hit). After the day: R1b (size) FAIL on controls (the day's storm cut the
  error 1.02%, where waves went 0.06%), R1c (direction) FAIL, R1d (interference) FAIL.
- Then asked to "adjust the strategy to how market works". Measured on 2011-2023 only (`research/ripples/`): quiet
  stocks catch waves overnight, nothing after the next open; no weekly busy-to-quiet flow (IC +0.003); earnings
  read-through priced on the day (later +0.005, small over-reaction -0.006); storm widening fixes W1 on 2011-2023 but
  W1 was already calibrated across storms on 2024-2026 (after the fact), so not adopted. Map now shows the day's storm
  and stocks that moved against their waves (own news likely).
- The person said "test it": forward test R2a-R2c declared 2026-11-22 (96b1e39), recorder `brain/againstwave.py`,
  record `research/ripple_calls.jsonl`; first reading 2026-11-20 (36 stocks against their waves, 2 with news).
  Judged once at 30 weekly readings (about June 2027) with `python -m brain againsttest test`.
- STANDING STEP: at every after-close check, after `scan run --refresh --record`, run `python -m brain ripple record`
  and commit research/ripple_calls.jsonl (a reading must be written before the next session closes; the before-open
  check is the backup slot). Adding it to the after-close task's prompt was refused by the permission check
  (2026-11-22), so it is done by hand until the person allows update_trigger.

## Attention (2026-11-23, at the person's request: "give the brain attention"; `docs/ATTENTION.md`)

- `python -m brain attention` after `desk build`: scores movers, against-the-wave stocks and watched stocks on seven
  fixed-weight parts (softmax), halves stray prints (one or two distinct prices since the close), and digs into the
  top six (filings, stories under web rules, memory recall, the pond). Focus kept in `research/attention/`.
- STANDING STEP: at every check, run `python -m brain attention` after `desk build` and start the note from it.

## Earnings: expected and what followed (2026-11-25, at the person's request; `docs/EARNINGS.md`)

- The person: "to predict the future of something, you need to know the past and present, not just what but how and
  why too", then "start with earnings reports and do all the things and test overnight".
- Built `brain/episodes.py` (107,583 reports since 2011, four universes; setup at the close before: run5/21/63 in
  usual moves, hi52, med8, mean8, up8, vix, spy21; r1, car3, post5, post21; a fixed name: priced in, relief,
  confirmed, capitulation, surprise up/down, no reaction), `brain/expect.py` (card, radar, forward record
  `research/earn_calls.jsonl`, an `[e:TICKER]` case-file line), `brain/research/earntest.py` (X1a-X1f).
- TRAIN 2011-2023 exploration logs in `research/earnings/`. The season so far (same-industry reports before) showed
  nothing on TRAIN; not declared. No past release text in the bank (lost in a reset); expectations are price-based.
- Declared and run once on 2026-11-25: X1a PASS; X1b, X1c, X1d, X1f FAIL; X1e reported.
- First forward record: Deere (DE) before its report of 2026-11-25. STANDING STEP to propose: at each after-close
  check, `expect radar --report <next session>` and record the confirmed reporters (`expect record`).

## Promises: said and delivered (2026-11-25, at the person's request; `docs/PROMISES.md`)

- The person's direction: know each company in and out (how it works, what it promised, what it needs, who it
  trades with, its dates) to judge whether it can deliver and how far the stock moves. They said yes to starting
  with the promise-keeping record.
- Reader 3 (`brain/sources/releases.py`) keeps each release's forward-looking passages. All 741 research-universe
  companies were re-read from SEC on 2026-11-25 (31,996 releases). `brain/promises.py` reads revenue guidance in
  dollars by fixed rules and holds it against revenue as first reported: 5,576 promises of 266 companies.
- The habit = the median surprise of a company's last four quarterly deliveries, the latest included (chosen on
  TRAIN: `research/promises/explore_habit_train.log`). Ranges around it: BAND68 (-3.5%, +3.0%), BAND90 (±8.1%).
- Declared and run once on 2026-11-25: **G1a PASS** (the habit carries to the next promise: +0.474, 11 of 11
  seasons, p 0.0007, q 0.003; delivery a median 1.43% from the habit-moved midpoint against 2.41% from the
  midpoint). G1b reported (same day): less than its habit -2.04% (40% rose), more +2.35%; year's promise cut -3.70%,
  raised +3.29%; delivered inside its range -0.23%.
- About deliveries, not the stock. Not declared (nothing on TRAIN): the record against the reaction on the day the
  quarter is reported. HELD BACK for a later id (too few 2024-2026 cases in this universe): the stretch against
  the delivery (TRAIN +0.071), the change of the year's promise against the next five sessions (TRAIN +0.054).
- Coverage: 106 of 741 companies made a quarterly dollar revenue promise on 2024-2026; 2% mismatches not graded.
- Seen on TRAIN (broad), new on 2026-11-26: after taking the habit out, a promise that asks for MORE delivered
  more, not less (minus the stretch against the surprise less the habit: -0.133, t -5.0, 6 of 30 seasons
  positive). The raw +0.071 came from the habit (big beaters guide low). So "asks for more, falls short" is not
  what the old years show.
- Every universe's releases are re-read with reader 3 as of 2026-11-26 (broad 741, small 510, micro 332, newer
  listings 965; the newer listings in four batches sized to finish inside the two-hour limit). The library
  (`promises build`): 19,547 promises of 1,016 companies.
- TRAIN on the three other universes (`research/promises/explore_bands_train.log`): 3,993 quarterly promises of
  405 companies; small and micro miss their own range about twice as often as large companies (16-18% against
  8.9%); the habit +0.351 (40 of 40 seasons); beyond the habit a promise that asks for more was beaten by more
  (+0.150), so the first "less is better" reading came from the habit itself.
- **G2a-G2d**, declared and run once on 2026-11-26 (`brain/research/promisetest2.py`; 1,796 companies of the
  small, micro and newer-listing universes; the research universe left out because G1b had printed its week
  after a release): **G2a PASS** (the habit carries: +0.486, 11 of 11 seasons, 1,357 promises of 299
  companies, p 0.0006, q 0.003; delivery a median 1.37% from the habit-moved midpoint against 2.34%).
  **G2b FAIL** (the stretch beyond the habit +0.096, p 0.031, q 0.12: the sign held, not past the bar).
  **G2c FAIL** (the five sessions after a release against the delivery less the habit: -0.000). **G2d FAIL**
  (against the change in the year's promise: +0.034, p 0.11; after a cut -0.92%, after a raise +0.31%).
  So the habit is the one promise finding, now on two sets of companies; the stock after a release follows
  nothing tested. The card and the desk line quote all five verdicts. 70 verdicts in the ledger, 11 PASS.
- Sized, not built (`research/promises/survey_forms.log`, loose patterns, an upper bound, 2024-2026): 199 of 741
  companies gave a dollar revenue promise; 215 others gave revenue growth in percent; 262 others gave EPS; with
  all three at most 523 of 741. EPS guidance is mostly adjusted EPS, which the accounts do not hold: its
  delivery would have to be read from the next release.
- The desk's case file now carries `[g:TICKER]` for a reporting mover (first live use: the next check with
  reporters). `promisetest` does not pin `research/run.py`.

## Ahead of whom? Analysts, the price, and G3a (2026-11-27; `docs/PROMISES.md`)

The person, 2026-11-27: "i want it to be ahead of everyone and not just ahead, but oppotunistic and accurate".

- **Against the analysts (a trial, not a test).** The Alpha Vantage connector's `EARNINGS_ESTIMATES` gives
  analysts' revenue and EPS estimates per fiscal quarter back to 2017 (a free key: 25 requests a day, one a
  second; each answer is about 12,000 tokens, so fetch through a fresh subagent that writes
  `var/consensus/full/TICKER.json`). On 11 companies and 214 quarters ending by 2023: analysts sit +0.21% above
  the company's midpoint (they repeat it), the habit says +2.12%, revenue beat the analysts in 81%; median miss
  1.88% for the analysts against 1.36% for the midpoint moved by the habit; the habit's guess was closer in 62%.
  The report session followed the delivery against the habit (+0.31) a little better than the beat against the
  analysts (+0.28), and a beat the habit foresaw brought nothing (-0.07). Ahead of the analysts is not ahead of
  the price. `research/promises/trial_analysts_train.py`.
- **What warns of a broken habit (2011-2023, `research/promises/explore_break_train.py`).** Twelve things known
  before a report. The stock's run says how the delivery comes against the habit and nothing about the session
  (priced). The season so far: priced. One lead on the session: the stretch (+0.061, t +4.2).
- **G3a FAIL (declared and run 2026-11-27, `brain/research/promisetest3.py`).** The stretch against the report
  session that delivers the promise, the four universes, 2024-01..2026-09: +0.038, 8 of 11 seasons, p 0.115,
  q 0.31. The third asking for least -0.57% (46% rose), the middle +1.20%, the most +0.90%. Not a finding.
- **The forward record (`brain/against.py`, `research/against_calls.jsonl`).** `against add TICKER --analysts X
  --report D` before a report; `against score` after `promises build`; under 30 judged lines it says nothing.
  Recorded 2026-11-27: BOX, OKTA, NTAP (Dec 1), ZS, FIVE (Dec 2), VEEV (Dec 3), LULU (Dec 8), AVGO, PLAB (Dec 9).
  OXM and ZUMZ had no report date yet; their estimates are in `var/consensus/full`. Each week: fetch the
  estimates of the companies about to report that have an open promise and a habit, and record them.
- **The second guess, G4a PASS (2026-11-28; `brain/research/promisetest4.py`, `promises.guess2`).** The person
  asked "how can we make it better?". On 2011-2023 a large habit overshoots (over 8%: delivered under the habit's
  guess 85% of the time) and the fixed ranges misfit steady and erratic companies. The second guess: delivery =
  +1.0% + 0.50 x habit; ranges by the company's own spread (`BAND2`). On 2024-2026: mean miss 2.33% against 2.48%,
  11 of 11 seasons, p 0.0009, q 0.004; over 8% habits 3.46% against 4.60%. G4b (reported): ranges held 78.9% and
  94.2% (wide), and 87/77/71/74% across spreads where the fixed ones held 94/80/64/39%. 73 verdicts, 12 PASS.
  The card prints both guesses; `against second` added the second guess to all 24 recorded lines on 2026-11-28,
  before any report (Micron: $67.2 bn against the plain habit's $71.6 bn and the analysts' $61.8 bn).
- **What I offered next (2026-11-28), not started:** read guidance given as growth rates and as earnings per share
  (today only revenue in dollars, about 1 company in 7); a paid data plan for analysts' estimates and option
  prices on every company (needs the person).
- **What I told the person:** ahead of the news and of the price: no test supports it; ahead of the analysts'
  published number on what a company delivers: the trial says yes; accurate: the sizes are (the ranges hold their
  odds), my direction calls are a coin flip (13 of 25); opportunistic means speaking only when a tested piece
  says something rare and staying silent otherwise. What would move it: a data key without the daily limit
  (analysts' estimates and option prices for every company), and time for the forward records.

## Reddit, by company and topic (2026-11-27; `docs/REDDIT.md`, `brain/reddit.py`)

- The person asked to track company employees' LinkedIn, Twitter and Reddit. I declined following people (their own
  hard rule; logins and terms block LinkedIn and X); when they said "if they post it publicly ... there is no
  illegal or ethical issue", I agreed reading public posts is legal and trading on truly public facts is not
  insider trading, but kept to company-level counting. They said "ok, set up the reddit one".
- Built: `brain reddit check|collect|show|top` (8 tests). Site-wide search per company by its common name
  (conservative names: `AMBIGUOUS`, `ALIASES`, `CASED`), keep posts that name it, store id, forum, time, score,
  comments, topic tags; no username, no text. Key: the person's Reddit app (client ID as username, secret as
  password) saved through the secure form for host `www.reddit.com` (basic); the proxy adds it.
- Status 2026-11-27: no key. Reddit's apps page told the person "Max clients have been reached"; Reddit approves new
  API apps by hand since November 2025. I told them to look for an existing app under "developed applications" or
  apply through Reddit's help centre. When a key exists: `reddit check`, then `reddit collect` at every
  after-close check; declare a test after about eight weeks of record.

## Moonshots (2026-11-28; `docs/MOONSHOTS.md`, `brain/moonshots.py`)

- The person: "i don't need just the revenue guess, i want to predict every moonshots before they happen or atleast
  most of them." A moonshot: a session 20%+ above SPY.
- Measured on 2011-2023 only (`research/moonshots/`, 1,058,157 stock-weeks; S1 refitted on 2011-2019, read on
  2020-2023): the top 5/10/20% of stocks each week held 53/68/82% of jumps; a jump-only model ranks no better than
  S1 (not declared); jumps and crashes cannot be told apart (about 2 jumps to 1 crash in every list). 44% of jumps
  are in stocks under $5, 15% in drug companies, 9% in results weeks, 41% in none of these.
- Buying the top 5% weekly: mean +0.77% (2011-2019) and +1.14% (2020-2023) a week against SPY on paper, median pick
  -0.8%, 46% rose, positive every year. NOT trusted: survivorship (the bank holds only companies listed today) and
  trading costs on penny stocks. A lead for the forward record, never an edge.
- Built `moonshots list` (the latest scan's likeliest names with tags) and `moonshots score` (recorded scans only:
  jumps and crashes held, paper return). First nine scans (Nov 9-19): top 20% held 49 of 62 jumps (79%); the top
  5% lost a mean 6.2% a reading. Reading fixed 2026-11-28: week-end scans from 2026-12-04, top 5%, mean return
  against SPY; 30 weekly readings before it says anything, then declare and judge once.
- Old-years rows are kept outside git: `var/moon_train.npz` (features, outcomes) and `var/moon_train_fwd.npz`.
- Not built, offered: dated events (FDA decisions, trial results); option prices and short interest (paid data).

## The company file (2026-11-26; `docs/FILE.md`)

- `python -m brain file TICKER` (`brain/file.py`, about 20 s a company): eight sections on one page: what it is;
  its accounts and what the price pays (P/E on the fiscal year and on the last four quarters, price to sales);
  what it promised and delivered; what it needs and what is on file (the mosaic); who it moves with; its dates;
  how far it can move (W1, S1, M1; the run into a report only within 14 days of it); what is not in the file yet.
- It adds no claim of its own: each line is an existing card with its recorded verdict. Start here when the
  person asks about a company.
- Not in it yet (the person's "every fabric"): suppliers by name and unnamed customers, projects and what they
  need, guidance as growth or EPS, earnings calls, any tested direction.
- Seen while building, not fixed: the brief shows NVDA's capex from fiscal 2012 (the company changed the tag;
  the file reads both tags and drops an item older than the fiscal year).

## Customers: who it sells to (2026-11-26; `docs/CUSTOMERS.md`)

- The person asked "who it trades with". `brain/customers.py` reads each company's latest 10-K by fixed rules:
  a named customer and its share, the largest unnamed share and the count of 10%+ customers, the top five or
  ten, the U.S. government's share, or the report's own "no customer brings a tenth". One `customers.major`
  fact a report (reader 4), the sentences quoted; candidate passages cached in `var/customers/chunks/` so
  `customers read --cached [--again]` replays a new reader without SEC. `customers build` makes the library
  with the reverse index (who names a company). The company file's section 5 carries the card.
- Research universe, latest reports (741 read in about 10 minutes, 0 failed): 62 name a known customer (104
  links, 77 with a share, 96 to a listed company); 149 state a largest share (median 19%); 191 a count; 93 say
  none; 434 nothing read. Walmart is named by 15 companies, AT&T 6, Apple 5.
- Precision was bought by reading: reader 1 (any known name in a customer sentence) was wrong for about one
  company in five (NASDAQ, rating agencies, brands a dealer sells, vendors, own segments). Reader 4 takes a name
  only where a customer stands; all 104 links were read against their sentences. It misses more than it invents.
- Every year's 10-K since 2010 was read on 2026-11-26 (12,169 reports, 0 failed; reader 5 after a sample of 45
  pairs showed a subsidiary's share, a growth figure and a spun-off parent read as customers): 2,271
  point-in-time links of 179 suppliers to 147 listed customers (`customers build --all-years`,
  `customers.links`).
- On TRAIN (customers' reports 2011-2023, `research/customers/explore_train.log`): the same session +0.220 for
  linked suppliers against +0.075 for industry peers; the next five sessions +0.040 (+0.099 on 2011-2018, +0.010
  on 2019-2023); the supplier's own later report -0.028 (not declared).
- Declared and run once on 2026-11-26 (`brain/research/customertest.py`): **C1a FAIL** (the day: the link added
  +0.062, linked +0.109 against peers +0.047, 7 of 11 seasons, p 0.10, q 0.30) and **C1b FAIL** (the next five
  sessions -0.093, 2 of 11 seasons positive, p 0.98). The read-through on the day shrank from the old years;
  after the day there is nothing to follow. The reversal in C1b was not declared: not a finding. The card and
  the desk's `[c:TICKER]` line quote both verdicts. Small, micro and newer-listing universes are not read.
- DO NOT run `brain ingest` or `sec.filings_for(all_pages=True)` over a universe: the older pages of the large
  banks' filing lists hold over a hundred thousand note prospectuses (JPM 133,000 424B2), the bank's journal
  for them grows to 75 MB, and `mosaic read` (READ_FROM 2008) would then try to fetch every one. Done by mistake
  on 2026-11-26 for 175 companies and undone the same hour (820,818 just-added filing facts removed from the
  local journals, originals copied first; the 3,598 older annual reports kept). `customers.older_reports` takes
  the annual reports only. Guarded since: `sec.filings_for` keeps the latest page only for a filer with more
  than 20,000 older filings (`MAX_OLDER`), so `brain ingest JPM` is safe.

## Open questions and offers (no answer yet)

- Asked 2026-11-25: (1) OK to withdraw R2a-R2c and re-declare them as new ids? Their pin on `brain/research/run.py`
  broke when the earnings kind was added, so they would be refused at judgment; the restart loses 3 recorded
  sessions. (2) Which companies does the person care about most (a short list for deep files)? (3) Can they get a
  data key for analyst forecasts and option prices?
  Said again 2026-11-27: the attached Alpha Vantage connector works for analysts' estimates but its key is a
  free one (25 requests a day); a paid plan on that same connector, or a key saved in the environment's
  settings (never pasted into the chat), would let the trial become a test on every company.

- Asked 2026-11-14: add `mosaic read` and `mosaic radar --record` to the after-close check? (Without it the
  mosaic's prospectuses and accounts go stale.)
- Found 2026-11-14: the check measures bitcoin and S&P futures from Thursday when Yahoo's day bar is missing or
  still open, so "bitcoin -6%" was really about -2%; and it reads zero-volume after-hours prints (SAR +6.5% was
  one such print). Fix offered, not made.
- Offered earlier, not taken up: a test of whether a stock's news reaches its partners the next day (knock-on);
  an industry lead-lag test; the earnings radar in the 8:07 check; S3 recalibrated going forward; the trader
  replayed on 2022; a better headline picker.

## The market thread (last desk note: `research/desk/2026-11-20/2026-11-21T0427Z.json`, after Friday's close)

- Thu 2026-11-19: Anthropic (ANT) priced $83, opened $111.50, closed $97.17 (+17%); -4.7% after hours. The market
  reversed from +1.3% before the open to S&P 500 -2.02% (7,175.83, sixth loss in a row, longest since April 2024),
  Nasdaq -2.25%, Dow -2.19% (below 50,000); all 11 sectors down; VIX near 29. AMD -7.7% on a Hunterbrook report.
  Health insurers down a third day (UNH $330.48, -19.8% since Monday's $412.02); earlier notes miscounted the days
  (said third/fourth), corrected in the Thursday close note. Security software up against the market (Palo Alto
  beat and raised). After hours: Intuit -9.8% (lower outlook; Mailchimp sold for $3.68bn vs $12bn), Ross -5.2%,
  Gap +3.6%.
- Data gap (resolved): Yahoo posted Thursday's daily bar with an empty close for about 880 stocks; the closes
  appeared by 06:30Z Nov 20 and the scan of 2026-11-19 was re-run in full (2,451 stocks, 86 left out; 7fd7bcd).
  If it happens again: poll, then `scan run --refresh --record` before the next open, commit, refresh the explorer.
- Fri 2026-11-20: global AI-chip liquidation. Taiex -6.2% (-17.5% week, worst since 1991), KOSPI -5.2%, Nikkei
  -5.1%. US close: SPY -3.64%, QQQ -5.67%, DIA -1.98% (SPY and DIA in correction; DIA's worst week since March
  2020); seventh S&P 500 loss in a row. NVDA -14.01% to $125.53, MU/AMD -13%+, AVGO -12.2% (Google reportedly cut
  Broadcom TPU plans; unconfirmed). AI data-center hosts -15-18%. Gold -4%, bitcoin -10%, 10-year 4.91%, VIX 35.89.
  Fed Governor Cook warned against waiting on rate increases. Anthropic closed $88.91. After hours: CrowdStrike
  reported in talks to buy Zscaler (ZS +16%, CRWD -5%).
- Desk calls: the first three (NVDA, AMD, DELL size calls) were judged right at Friday's close (-22.9%, -20.4%,
  -18.9% from Nov 13), but at chances 0.50/0.42/0.40 (Brier 0.316). Open: KYMR move 15%+ either way from $124.06
  by Nov 27 (0.45), on its Phase 2b readout Monday Nov 23 at 8 a.m. ET.
- Scan score: the week of Nov 13-20 held 29.9% (2-in-3) and 62.8% (9-in-10), the worst so far; mean of 5 judged
  scans 54.2% and 83.3%.
- Check flaws known (fix offered, not made): after the close every driver line (futures, oil, gold) is measured
  from the previous day's bar; the oil line jumps when Yahoo's front WTI contract rolls (Nov 18: about 1-1.5
  points); single early pre-market prints count as moves (Nov 18: NWN, HEI, NEOG, WELL); dividends read as moves.
- Open with the person: the next research step (drug-trial line, a forward record of companies not issuing
  shares, or no new tests), and whether to fix the check's stale-baseline flaw.

## The scan explorer (built 2026-11-18 at the person's request)

- A private page: https://claude.ai/artifact/EyLgFmNo5ZyYFsKt6Hoq3H (`docs/EXPLORER.md`, `brain/explorer.py`,
  `web/explorer/index.html`). Perspective 5.6.1 grid of every scanned stock: ranges, 20% day chance, move since the
  latest check, where it sits, why it moved, desk themes as filters. Only percents, never prices. If Perspective
  cannot start in the page it shows a plain table after 20 seconds.
- Refresh: `python -m brain explorer build` (about 3 minutes the first time per scan), then publish with the
  Artifact tool with url https://claude.ai/artifact/EyLgFmNo5ZyYFsKt6Hoq3H, file_path web/explorer/index.html, files
  {"data.json": "var/explorer/data.json"} (the vendored lib/ files stay). Always pass `url`: on 2026-11-18 a publish
  without it, after a summary, made a separate broken page (https://claude.ai/artifact/8vgPd7A27CAQwfknCor3d4, no
  Perspective files; the person was told and asked whether to delete it). The tool may ask to read the published
  data.json first. Do this after every check: the person accepted "refreshed after each check".
- The before-the-open task's prompt now includes the refresh (step 6). Updating the midday and after-close tasks'
  prompts was refused by the permission check (2026-11-18); until the person allows it, refresh the page by hand
  at the end of those checks.
