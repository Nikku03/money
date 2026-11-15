# The memory bank: a library beside the context window

A model's context window is fixed, like a desk. Long conversations are summarized and details drop out, and the
machine this runs on is reset now and then. Nothing can make the desk bigger. What `brain/memory.py` adds is a
library beside it: everything said and done is kept, searched in a second, and the pages that matter are brought
back to the desk when they are needed. The library has no size limit and is kept on the person's Google Drive.

```
python -m brain memory start                    # first thing in a session: where things stand, the latest asks
python -m brain memory recall "what did we decide about the mosaic"   # the passages that answer a question
python -m brain memory show ID                  # one passage in full; a tool call comes with its result
python -m brain memory remember "a fact" --topic drive   # keep a fact no file states
python -m brain memory index                    # embed passages not embedded yet (recall does it too)
python -m brain memory save                     # pack it all for Drive (and save the bank's claims)
python -m brain memory load [--talk F] [--vectors F]     # after a reset: merge the Drive copies back
python -m brain memory import --session ID PAGES...       # another session's history, from saved event pages
python -m brain memory status                   # what the local home holds, and whether it was reset
```

## What the library holds

| Shelf | What | Where |
|---|---|---|
| **talk** | every turn of the conversation: the person's messages, the agent's replies, its reasoning, each tool call and the start and end of its result, and the summaries written when the context was compacted | local home, `var/memory/talk.jsonl`; on Drive |
| **notes** | `MEMORY.md` (where the work stands, read first), `JOURNAL.md` (what happened, by date), `notes.jsonl` (facts kept with `remember`) | `memory/`, committed; on Drive |
| **docs** | `CLAUDE.md`, `README.md`, `docs/`, the `research/` notes and results, the desk's recorded notes, each module's opening docstring | the repo, read in place |
| **claims** | the claims the agent noted from pages it read: the one part of the bank that cannot be downloaded again | `memory/claims.jsonl`; on Drive |

`capture` (run by `recall`, `start` and `save` on their own) reads the session transcripts on this machine from
where it last stopped. Email addresses and keys are masked before anything is kept. Long tool results keep their
first 1,800 and last 700 characters. A long message is kept in pieces of about 900 characters, so each piece has
one meaning to rank by; `show` puts a piece back with the rest of its message.

## How recall works

Two rankings, fused:

- **By words** (BM25): rare words that the question and the passage share count most. A question is widened by a
  short list of words that mean the same here (prefer and want, flaw and bug, sale and offering).
- **By meaning**: a small embedding model (`BAAI/bge-small-en-v1.5`, 65 MB, on the CPU; `pip install fastembed`)
  turns every passage and the question into 384 numbers, and passages whose numbers point the same way rank high.
  "Was there a mistake in how crypto was measured" finds "bitcoin's -6% was measured from Thursday evening" with
  almost no word in common. Each passage is embedded once; the vectors are kept in `var/memory/vectors.npz`.
- **Fused** by reciprocal rank: a passage near the top of either ranking comes near the top, one near the top of
  both comes first. The person's messages, the agent's replies, summaries and notes count a little more than tool
  output.
- `--since 2026-11-14` keeps only what happened from that date; `--kind you` only the person's messages;
  `--words-only` skips the model. Without the model installed, recall ranks by words alone and says so.
- Recall leaves out its own calls and output, so asking twice does not find the first asking.

## Other sessions

The person works in more than one session. Another session's history comes through the remote-session tools,
not the transcripts on this machine:

1. `list_sessions` (mine) finds it; `list_events` with `kinds: ["user", "assistant"]` and `limit: 100` reads its
   newest page, and `before_id` set to a page's `first_id` reads the page before, until `has_more` is false or
   missing. Each page is too large to show and is saved to a file.
2. `python -m brain memory import --session SESSION_ID FILE ...` adds every page to the talk, each passage
   labelled with the session's name. A page imported twice adds nothing.

That session's messages, replies, tool calls, results and summaries come through; its reasoning does not (it is
not shared between sessions). System notices (hook messages, image notes, notifications) are not taken for the
person's words. Imported so far: `session_01MSaQVTPoBwSsjdSBSk9kdd` (2026-11-04 to 11-09, the paper traders
F1-F5 and the options checks V1/O1 on branch `claude/sleepy-goldberg-2ma18z`).

## Keeping it across resets

- `memory/` is committed and pushed. The talk is too large and too raw for the repo, so it lives in the local
  home and, packed, on Drive.
- After meaningful work: update `MEMORY.md` and `JOURNAL.md`, run `memory save`, commit `memory/`, and upload what
  it lists to the Drive folder `money-brain memory`, each as a new file named with the date. The newest is current.
- After a reset: `memory start` says whether the home was reset. `pip install fastembed`. Download the newest
  `talk` and `vectors` files from the Drive folder (Drive's download saves each as a file) and run
  `memory load --talk FILE --vectors FILE`: embedding everything again would take half an hour now, and days at
  the scale below. Then rebuild the market data:

1. `python -m brain scan run --refresh`: prices and the latest filings of every scan stock (about 40 minutes).
2. `python -m brain mosaic read`: company accounts and what each prospectus sells (1-2 hours from empty).

## How much it can hold

Measured on the first five days (2026-11-10 to 2026-11-15): about 330,000 tokens of kept text (9% of the raw
transcript: tool results are clipped), 5.0 bytes a token in the local file and 1.4 packed, plus 808 bytes of vector
for each passage of about 185 tokens (4.4 bytes a token).

| Space | Local, with vectors | On Drive, packed, with vectors | On Drive, packed, text only |
|---|---|---|---|
| 2 GB | about 210 million tokens | about 340 million | about 1.4 billion |
| 3 GB | about 320 million tokens | about 520 million | about 2.1 billion |

That is 200 to 500 times a one-million-token context window. At the pace of those five days (about 66,000 kept
tokens a day) 3 GB would take over ten years to fill.

Space is not what limits it. At that size:

- **Embedding** runs at about two passages a second on this machine's CPU, so new passages each day take a minute
  or two, but embedding a whole 3 GB library from nothing would take weeks. That is why the vectors are kept on
  Drive.
- **Search** reads every passage for each question. Under about 100,000 passages that takes a second or two; past
  that it needs an index built once and updated (not built yet; the library holds a few thousand passages).

## What it is not

- It does not make the context window larger, and it does not think. It finds and returns text; the agent still
  has to ask it, and a question in the wrong words can miss.
- Yahoo prices are never stored in it. The person's email is masked, and nothing private about anyone is kept.
