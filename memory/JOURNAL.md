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
