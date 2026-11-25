# Journal

What was asked and what was done, newest last. One entry per working day; details are in the commits and the files
named. `memory/MEMORY.md` says where things stand now.

## Before 2026-11-13

Asked and done, in order: an analysis of CATX; whether its options were expensive; a review of the robot traders;
predictions of move sizes (M1); a scan of the whole market (W1 ranges, S1 shock chances); a two-year backtest;
small and mid caps added (`universe/us_small.json`, `us_micro.json`, `us_watch.json`); the scan run all the time
(three scheduled checks a day); a dynamic trader replay, then one trade a day (both below the fixed weekly reading);
plain explanations of how trades and move sizes work; a catalyst model S3 (ranked better, failed calibration, S3a
FAIL); "the market runs on chains of cause and effect, news and catalysts": the why-layer for each flagged move
(`brain/why.py`).

## 2026-11-13

- Asked: "it is only scratching the surface: seeing news, filings, charts, not reasoning or connecting them."
  Built the desk (`brain/desk.py`, `brain/connect.py`, `docs/DESK.md`): a case file per check and a reasoned note
  held to rules. Wrote the first note (Nvidia chief's resignation and the AI hardware markdown). The three
  scheduled checks now end in a note.
- Asked: "knowing reports are coming and stocks will move is nothing special; where is the surprise?" Agreed: my
  NVDA, AMD, DELL size calls matched what options priced. Promised to make size calls only when they disagree.
- Asked: "so our target is using public information to find things only someone with insider info can." Explained
  the legal mosaic and proposed testing cash runway + a shelf on file + a recent jump.

## 2026-11-14

- After-close check and desk note (`research/desk/2026-11-13/2026-11-14T0444Z.json`). Found the check's bitcoin
  and futures lines were measured from Thursday; offered a fix.
- Asked: "make an all around mosaic." Built `brain/mosaic.py`: 40-odd public pieces per stock, the first page of
  12,783 prospectuses read (most sell bonds), accounts for 2,514 companies. Three pictures: a share sale, late or
  unreliable accounts, a restructuring or write-off. Declared P1a-P3b, ran them on 2024-2026: the chances beat
  each company's habit and were calibrated (P1a, P2a, P3a PASS); the flagged stocks did not lag (P1b, P2b, P3b
  FAIL). Recorded 75 live calls. Asked whether to add the mosaic to the nightly check: no answer yet.
- Asked: "in simple." Explained: it predicts what a company will announce, not which way the stock goes.

## 2026-11-15

- Asked: use the memory bank as an extended context, saved to Google Drive, so nothing is lost when the machine
  resets. Built `memory/` (this journal, `MEMORY.md`, the claims the agent noted) and `brain/memory.py`
  (`memory save`, `memory load`, `memory status`), and a copy on Drive in the folder `money-brain memory`.
- Then: "I want a memory bank that replaces your context window, so you can remember more." Explained that the
  window itself is fixed and built a library beside it: `memory capture` keeps every turn of the conversation
  (1,712 passages since 2026-11-10, emails and keys masked), `memory recall` ranks the conversation, notes and
  docs against a question (BM25 with a short synonym list), `memory show` gives a passage in full, `memory
  remember` keeps a fact, `memory start` opens a session. Installing a meaning-based search model was refused by
  the permission system; not worked around; recall works without it. The packed talk goes to Drive with the notes.
- Then: "yes, install the smarter search; how many tokens can it save in 2-3 GB?" Installed fastembed with the
  model `BAAI/bge-small-en-v1.5` (65 MB). Recall now fuses a ranking by words with a ranking by meaning. Long
  messages are kept in pieces of about 900 characters so each has one meaning. Vectors are saved to Drive with the
  talk. Measured: 2-3 GB holds about 210-320 million tokens locally with vectors, 340-520 million packed on Drive,
  1.4-2.1 billion as packed text alone; at five days' pace, 3 GB is over ten years. Embedding runs at about two
  passages a second, so the vectors are worth keeping; past about 100,000 passages search needs an index.
- Then: "do we have this conversation in the memory bank? Add that session too." Yes for this one (since
  2026-11-10 word for word). Imported the other trading session, `session_01MSaQVTPoBwSsjdSBSk9kdd`, through the
  remote-session tools: 38 saved pages, 2,277 passages, 2026-11-04 to 11-09, with the person's 34 messages there
  (its reasoning is not shared). Built `memory import --session`.
- Then: "there's also another session named High variability trading ML model; add that too." Imported
  `session_018ANuE3nDWJsR8htWjJDJhG`: 251 saved pages, 9,212 passages, 2026-07-29 to 10-21, 160 of the person's
  messages. Found that documents and skills loaded into a session looked like the person's words; now only
  messages sent from the person's app count as theirs (both imported sessions re-read with that rule).

## 2026-11-16

- Asked: "now that you have the full context in the memory bank, analyse why we keep failing and how to make it
  an absolute best strategy." Read the three sessions' summaries and reports through the memory bank and wrote
  `research/POSTMORTEM-2026-11.md`: about seventy ideas, eight passes (all size or events), five causes, what
  survived, the best design. Proposed next: one declared test of "quality tilt plus buying after fear" against SPY,
  and a graveyard check before any new idea.
- Found the older session's data-provider API key in plain text in the imported talk. Strengthened the mask
  (named keys, NAME_KEY=value, a key seen once is masked everywhere), rewrote the stored talk (none left),
  uploaded a clean copy to Drive and moved the earlier upload to Drive's trash. Told the person; rotating the key
  is their call.
- The embedding of the two imported sessions finished (9,505 passages, about an hour and a half): recall by
  meaning now covers all three sessions. Drive refused the vectors (13 MB): one upload from here carries about
  3 MB. `memory save` now writes a large file in parts under 2.8 MB and `memory load` takes several files. The
  talk of all three sessions is on Drive; the five vector parts are not, and the person was asked whether to
  send them.
- Before-the-open check and desk note (`research/desk/2026-11-13/2026-11-16T1339Z.json`): AI hardware marked down
  again before Nvidia reports; crypto stocks at their usual rates after bitcoin's weekend fall; CG Oncology's
  result and UroGen beside it; two flags were dividends leaving the price (a third flaw in the check). No calls.
- Midday check and desk note (`research/desk/2026-11-13/2026-11-16T1738Z.json`): the AI sell-off spread to the
  whole build-out and most buyers; Trump Media's debt deadline; Medicaid insurers up; UroGen reversed, so the
  morning's inference did not hold; IEP's fall is more than its distribution. No calls.
- Asked: "yes upload them, and start with step 2". Uploaded the five vector parts, the talk in two parts and the
  claims to Drive. Built `brain/research/tilttest.py` (T1a the quality tilt, T1b buying after fear), checked the
  memory bank for earlier tries (never declared alone), counted the data without returns, declared, committed, ran
  once: both FAILED, controls passed. Corrected one misleading label in RESULTS.md after the run (the list's gap to
  SPY is mostly companies the list leaves out, not survivorship) with a ledger note.
- After-close scan of 2026-11-16 (2,452 stocks, 85 left out), the after-hours check and a desk note
  (`research/desk/2026-11-16/2026-11-17T0430Z.json`): the AI markdown closed at its lows with a new chip-export rule
  for five countries; the check's market-wide flag was its stale-baseline flaw again. No calls.
- 2026-11-17: before-the-open, midday and after-close checks with desk notes. Midday: software sold as a group
  before Anthropic's offering. After the close: Nvidia beat ($115.8bn) and guided $130bn; chips rallied after hours.
  The person asked about Perspective (an open-source data grid and pivot table); explained it is a viewer, not an
  edge, and offered a scan explorer; no answer yet.

## 2026-11-18

- Asked: "yes build the scan explorer page". Built `brain/explorer.py` (joins the latest scan, check, case file and
  desk note into var/explorer/data.json; percents only), the page `web/explorer/index.html` (Perspective 5.6.1,
  vendored with its wasm), tests and `docs/EXPLORER.md`; checked it in a local browser on desktop and phone; published
  it privately at https://claude.ai/artifact/EyLgFmNo5ZyYFsKt6Hoq3H.
- Added the page refresh to the before-the-open task's prompt. The same change to the midday and after-close tasks
  was refused by the permission check; told the person, and the refresh is done by hand at those checks until
  they allow it.
- Wednesday's three checks with desk notes (before the open, midday, after the close) and the close scan of
  2026-11-18 (2,446 stocks, 91 left out). The explorer page was refreshed after each check, by hand for midday and
  after the close (passing its url). A refresh without `url` made a separate broken page; told the person.
  Two of the morning's inferences were wrong and were corrected in the later notes.

## 2026-11-19 to 2026-11-20

- Thursday's three checks and notes: before the open (Anthropic's price lifted AI; Sellas priced its share sale, which
  the mosaic had flagged at 7.3% on Monday), midday (the reversal on Anthropic's first trade; AMD short report;
  staffing and IT services down, inferred AI reading), after the close (a 2% fall; Intuit, Ross, Gap after hours).
- The close scan of 2026-11-19 covered only 1,580 stocks: Yahoo's Thursday bars lacked closes for about 880. A poll
  waits for them; the container restarted once during the wait (the bank survived).
- The full scan of 2026-11-19 was re-run at 06:30Z when Yahoo posted the missing closes (2,451 stocks, 86 left
  out; 7fd7bcd); explorer refreshed.

## 2026-11-20: the paper and the neuron ladder

- The person asked to compare He et al. 2024 (PMID 39152312; abstract only, full text paywalled) with "our brain
  model". First answered about the market engine; the person said to compare with the brain model itself, so
  compared it with the spiking circuits (`brain/neurons`). Then asked: "add the missing parts in our current brain
  model, but also keep the good things".
- Brought the neuron code over unchanged (tests pass); added `--cells lif|izh|hh` (calibrated gains), `neurons
  ladder` (every game on each cell), `neurons exchange` (fitting LIF circuits to one HH cell) and `neurons train`
  (timing task, delays trained or fixed); 20 new tests; results in `research/neurons/` and `docs/NEURONS.md`.
- Verdict: richer cells did not need fewer cells on this simulator; HH helped hand-wired Catch; delays (our
  wiring's strength) helped every cell; HH costs 5-10x CPU.
- Friday Nov 20's three checks with desk notes: before the open (the chip selloff ran through Asia), midday (the
  whole AI hardware chain fell, a market-wide move) and after the close (SPY -3.64%, QQQ -5.67%; the desk's first
  three calls judged right; a new call on Kymera's Phase 2b readout). The scan of 2026-11-20 recorded in full.

## 2026-11-22: what was won and lost; the pond

- The person asked what was won and lost across the tests and scores (answered from RESULTS.md, the postmortem, the
  trader replays and every score command), asked for it in simple words, then for every suggestion and call with its
  outcome and the picks ranked.
- The person proposed the pond-and-ripples picture. Built the ripple engine and map, declared R1a-R1d, ran once:
  ripples travel one ring on the day and are mostly absorbed; nothing about the following five sessions passed.
- Asked to adjust the ripple strategy to how markets work. Measured wave speed by liquidity, from the next open,
  busy-to-quiet weekly flow, earnings read-through and the storm on W1, all on 2011-2023; none leaves a tradable wave
  after the day. Added the storm reading and the moved-against-its-wave flag to the map.
- The person said to test whether stocks that moved against their wave keep going or bounce back. Built the recorder
  and the forward tests R2a-R2c, declared them before any record, recorded the first session (2026-11-20). The
  after-close task could not be changed (permission refused); the record is run by hand at each close.

## 2026-11-23: attention

- The person asked what the brain can do, trace and reason, then said "give the brain attention". Built
  `brain/attention.py`: what deserves a closer look at a check and the digging on it; tests, `docs/ATTENTION.md`.
  On Nov 20's after-hours check it put Kymera, Zscaler and CrowdStrike first and halved N-able's single-trade move.


## 2026-11-23 to 2026-11-25: the trade question, the catalysts, and earnings expectations

- Nov 23-24 checks with desk notes: Nvidia reinstated its CEO (S&P 500 +4.73% Monday); chips gave much of it back
  Tuesday while retailers and drug makers rose; Ambarella's memory-cost forecast; software beats met by falls.
- The person asked for a trade to hold overnight on catalysts. Answer: no tested direction; recorded a Dell
  move-size call (10833f5c, 4.5%+ either way from $296.45 by Nov 25, 75%). Then FDA dates and trial readouts: BBIO
  and COGT approved early; Savara overdue.
- The person said the brain lacks the how and why. Built the earnings episode library, the card before a report
  and the declared tests X1a-X1f (`docs/EARNINGS.md`). X1a PASS (a run-up makes a fall on the report session more
  likely; not a lower average); the rest FAILED. Recorded Deere's card before its Nov 25 report.
- The person said to know each company in and out, to judge whether it can deliver; yes to the promise-keeping
  record first. Re-read every earnings release of the 741 research-universe companies from SEC with its
  forward-looking passages (reader 3; about four hours, restarted once at the two-hour limit). Built
  `brain/promises.py` (revenue promises in dollars against revenue as first reported), the card, the desk's
  `[g:TICKER]` line and `brain/research/promisetest.py`.
- On 2011-2023 only: companies promise low (78% delivered above the midpoint); the median of the last four
  deliveries says most about the next. Declared G1a and G1b, ran once: G1a PASS (+0.474, 11 of 11 seasons). The
  day's move follows the delivery against the habit and the change in the year's promise (G1b, an explanation).
  Two questions held back for more companies (the stretch; the price after a raise or a cut). Started the small
  band's re-read.
- Fixed on the way: three event-test fixtures broken by reader 3; past quarters' sales read as promises (a
  quarter named with its year that is over is a result); the year-ago quarter of an open promise.
