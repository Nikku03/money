# Memory: where the work stands

Read this first in every session. It is the agent's notebook for this repo: what the person wants, what exists, what
was found, what is running and what is open. Details live in the files it points to; search them when needed.
Updated 2026-11-15 (memory bank added). The log of what happened, by date, is `memory/JOURNAL.md`.

## Keeping this alive

- This file is the always-read page of the memory bank (`docs/MEMORY.md`). The rest is searched:
  `python -m brain memory recall "..."` over every turn of the conversation since 2026-11-10 (the person's
  messages, replies, reasoning, tool calls and results, compaction summaries), the journal, kept facts, the docs,
  the research notes and the desk notes. Start a session with `python -m brain memory start`.
- Copies live in the Google Drive folder `money-brain memory`, one file per save, named with the date; the newest
  is current. After a reset, download the newest `talk` file and run `memory load --talk FILE`.
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

## What was found (verdicts in `research/RESULTS.md`)

- **PASS (sizes and events):** W1a, W1b, W1c (the weekly ranges hold their odds); S1a (shock radar ranks
  surprises); M1a (earnings move size); P1a, P2a, P3a (the mosaic sees share sales, late accounts and cuts coming,
  better than each company's habit, and calibrated).
- **FAIL (every direction test):** H1-H8 (momentum, value, insider buying, earnings surprise), L1 (cause → effect
  links next month), E1/E2 (news drift), S2a/S2b (fade after a jump), S3a (calibration), P1b-P3b (mosaic-flagged
  stocks did not lag). No test of which way a stock goes has passed.
- **Exploratory, not declared:** partners' results and late reports say almost nothing about a stock's own report
  (`research/MOSAIC-2026-11.md`); smart-money footprints (politicians, 13D copying, volume) did not pay
  (`research/HISTORY-SMART-MONEY-2026-11.md`, `research/PRECURSORS-2026-11.md`); dynamic trading and trading less
  did not beat a fixed weekly reading (`research/SCAN-TRADER-2026-11.md`).
- **Options:** my size calls on NVDA, AMD, DELL matched what options already priced: no edge.

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
- S1 calls (`research/s1_calls.jsonl`, 27), predictions (`research/predictions.jsonl`, 31), M1 calls (99).

## Open questions and offers (no answer yet)

- Asked 2026-11-14: add `mosaic read` and `mosaic radar --record` to the after-close check? (Without it the
  mosaic's prospectuses and accounts go stale.)
- Found 2026-11-14: the check measures bitcoin and S&P futures from Thursday when Yahoo's day bar is missing or
  still open, so "bitcoin -6%" was really about -2%; and it reads zero-volume after-hours prints (SAR +6.5% was
  one such print). Fix offered, not made.
- Offered earlier, not taken up: a test of whether a stock's news reaches its partners the next day (knock-on);
  an industry lead-lag test; the earnings radar in the 8:07 check; S3 recalibrated going forward; the trader
  replayed on 2022; a better headline picker.

## The market thread (last desk note: `research/desk/2026-11-13/2026-11-14T0444Z.json`)

- Nvidia's chief resigned (week of 2026-11-09) amid a scandal with export-control probes (Megaspeed, Singapore);
  AI hardware sold off together; Nvidia reports Tue 2026-11-17, its first without him.
- Rates: 10-year yield ~5.17%, wholesale inflation 6.5% a year, December Fed hike priced near 90%; dollar strong.
- Crypto sold off Friday evening; crypto-heavy stocks followed at their usual rates.
- Threads to settle: Nvidia's report and its dependents (Dell, CoreWeave, Super Micro); AMD and Broadcom with or
  against Nvidia; CD -18.7% unexplained; Vir's $10 offering closes 2026-11-16; Modine's cooling division talks
  with Samsung; Global Payments' -12% unexplained; Inhibrx's $150m at-the-market sale before its data on
  2026-11-19/20.
