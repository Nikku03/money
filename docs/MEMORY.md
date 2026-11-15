# Memory between sessions

The machine this runs on is reset now and then. The repository comes back from git; the local home (`var/`) does
not. The agent's working memory is also limited: long conversations are summarized, and details drop out. So the
agent keeps a notebook it can come back to.

```
python -m brain memory status   # what the local home holds, and whether it was reset
python -m brain memory save     # write the agent's own records to memory/, list the files to keep on Drive
python -m brain memory load     # after a reset: put them back, and say how to rebuild the rest
```

## What is kept where

| What | Where | After a reset |
|---|---|---|
| Where the work stands: the person, the rules, what exists, what was found, what is open | `memory/MEMORY.md` | read first |
| What was asked and done, by date | `memory/JOURNAL.md` | searched when needed |
| Claims the agent noted from pages it read | `memory/claims.jsonl` | `memory load` puts them back |
| Code, tests, verdicts, recorded calls, desk notes, scans | the repository | already there |
| SEC filings and accounts, prospectus readings, prices | the local home | rebuilt from public sources |

`memory/` is committed. The agent also keeps a copy on the person's Google Drive, in the folder
`money-brain memory`. Each save is a new file named with its date; the newest is the current one. That copy
matters when a session starts from a checkout without `memory/`, or when the person wants to read it.

## Rebuilding the local home

`memory load` says when the home looks reset, and lists the commands:

1. `python -m brain scan run --refresh`: prices and the latest filings of every scan stock (about 40 minutes).
2. `python -m brain mosaic read`: company accounts and what each prospectus sells (1-2 hours from empty).
3. `python -m brain mosaic radar`: the mosaic's model is committed, so it works as soon as its data is back.

Rerunning an old research test needs the full filing histories and data sets it names in its own doc.

## What it is not

- It does not make the agent's working memory larger. It is a notebook the agent reads at the start and searches
  when it needs something it no longer holds.
- Yahoo prices are never committed or uploaded. Nothing private about anyone is written into it, and the
  person's email is never stored.
