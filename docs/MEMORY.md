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
python -m brain memory save                     # pack it all for Drive (and save the bank's claims)
python -m brain memory load [--talk FILE]       # after a reset: merge the Drive copy back, restore the claims
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
first 1,800 and last 700 characters.

## How recall works

- Every passage is ranked against the question by BM25: rare words that the question and the passage share count
  most. The person's messages, the agent's replies, summaries and notes count a little more than tool output.
- A question is widened by a short list of words that mean the same here (prefer and want, flaw and bug, sale and
  offering). Concrete words work best: a ticker, a test id, a file name, a date.
- `--since 2026-11-14` keeps only what happened from that date; `--kind you` only the person's messages.
- Recall leaves out its own calls and output, so asking twice does not find the first asking.
- It finds words, not meanings. A model that ranks by meaning (an embedding model) would find more; installing one
  needs the person's permission, and recall works without it.

## Keeping it across resets

- `memory/` is committed and pushed. The talk is too large and too raw for the repo, so it lives in the local
  home and, packed, on Drive.
- After meaningful work: update `MEMORY.md` and `JOURNAL.md`, run `memory save`, commit `memory/`, and upload what
  it lists to the Drive folder `money-brain memory`, each as a new file named with the date. The newest is current.
- After a reset: `memory start` says whether the home was reset. Download the newest `talk` file from the Drive
  folder (Drive's download saves it as a file) and run `memory load --talk FILE`. Then rebuild the market data:

1. `python -m brain scan run --refresh`: prices and the latest filings of every scan stock (about 40 minutes).
2. `python -m brain mosaic read`: company accounts and what each prospectus sells (1-2 hours from empty).

## What it is not

- It does not make the context window larger, and it does not think. It finds and returns text; the agent still
  has to ask it, and a question in the wrong words can miss.
- Yahoo prices are never stored in it. The person's email is masked, and nothing private about anyone is kept.
