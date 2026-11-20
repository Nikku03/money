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
  is current. One upload from here carries about 3 MB, so `save` writes larger files in parts. On Drive now: the
  talk of all three sessions (`talk 2026-11-16d`, keys masked) and the vectors of this session only
  (`vectors 2026-11-15c.npz`). The vectors of all three sessions (13 MB, five parts) are NOT on Drive: the upload
  was refused for size and the person was asked (2026-11-16) whether to send the five parts. After a reset,
  download the newest `talk` and every `vectors` part there is and run `memory load --talk FILE --vectors FILE
  ...`; whatever has no vector is embedded again by `memory index` (about two hours for everything), and recall
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
| Spiking neurons (side project, not the market): hand-wired Pong/Catch/Runner, school; cells lif/izh/hh | `neurons play/catch/run --cells`, `neurons ladder`, `neurons exchange`, `neurons train [--no-delays]` | `docs/NEURONS.md` |

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

## Open questions and offers (no answer yet)

- Asked 2026-11-14: add `mosaic read` and `mosaic radar --record` to the after-close check? (Without it the
  mosaic's prospectuses and accounts go stale.)
- Found 2026-11-14: the check measures bitcoin and S&P futures from Thursday when Yahoo's day bar is missing or
  still open, so "bitcoin -6%" was really about -2%; and it reads zero-volume after-hours prints (SAR +6.5% was
  one such print). Fix offered, not made.
- Offered earlier, not taken up: a test of whether a stock's news reaches its partners the next day (knock-on);
  an industry lead-lag test; the earnings radar in the 8:07 check; S3 recalibrated going forward; the trader
  replayed on 2022; a better headline picker.

## The market thread (last desk note: `research/desk/2026-11-19/2026-11-20T0507Z.json`, after Thursday's close)

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
- Desk calls judged at Fri Nov 20's close against Nov 13's: NVDA 7.5%+ (0.50; $162.89; Thu $145.98, -10.38%),
  AMD 9%+ (0.42; $532.84; Thu $489.19, -8.19%), DELL 9.5%+ (0.40; $342.98; Thu about $312.54, -8.88%).
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
