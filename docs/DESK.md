# The desk: reasoning about a check, not listing it

A check between scans (`docs/SCAN.md`) says which stocks have left their ranges and prints one headline beside each.
That is a list. It does not say that a seed company's warning about Brazil is why a chemical maker fell, that a stock
"with no news" fell because bitcoin did, or that a headline was about another company. The desk is where that
reasoning happens, and where it is held to rules.

```
python -m brain scan watch --record     # the check (about 25 minutes)
python -m brain desk build              # the case file for it (about 3 minutes)
python -m brain desk show NVDA CRWV     # the same, with those movers' stories and filings in full
python -m brain desk template           # the layout of a note
python -m brain desk check note.json    # what is wrong with a note, if anything
python -m brain desk note note.json     # record it, with its calls
python -m brain desk last               # the last recorded note
python -m brain desk score              # how the desk's calls have done
```

## Who does what

| Step | Who | What it is |
|---|---|---|
| The case file | the engine, by rule | everything that bears on the day's moves, each item with an id |
| The note | a reasoner | what happened and why, citing the case file |
| The rules | the engine | a note that breaks them is not recorded |
| The score | the engine | the note's calls, judged after their dates |

The reasoner is the agent working in this repo (`CLAUDE.md`), which can also read the pages and filings a case file
points to. A language model given the same case file could write the note too; nothing in the rules depends on who
writes it.

## The case file

`brain/desk.py` builds it from the latest recorded check. Nothing in it is a conclusion.

- **The backdrop.**
  - The market line, and the drivers (oil, the 10-year yield, the dollar, gold, bitcoin) against their usual day.
  - Scheduled macro releases, industries moving together, and how many stocks are outside their ranges.
  - Market-wide stories: round-ups carried by many feeds or naming many stocks.
- **Each mover's move, split.** For the 30 stocks furthest outside their ranges:
  - what the market carried (the stock's beta times SPY's move);
  - what each driver carried, for the drivers the stock has clearly moved with over the past year;
  - its industry's median move;
  - what is left over.
- **Its partners.** The stocks whose daily returns, with the market's part taken out, moved most with its own over
  the last 250 sessions (`brain/connect.py`), and how each moved today. Partners come from prices, not from industry
  codes: Corteva's closest partner is FMC, Super Micro's are Dell and Hewlett Packard Enterprise, and CoreWeave's are
  the bitcoin miners that host AI servers.
- **Its stories.** The items in its news feed since the close, with their summaries; which other stocks each one
  names; and stories in its feed that name another company, marked as such.
- **Its filings.** The opening text of each 8-K's press release and of each share-sale filing since the close.
- **Its accounts.** Market value, cash, long-term debt, the last four quarters' revenue and net income, and for a
  company losing money how many quarters its cash covers (SEC XBRL).
- **Its last sessions,** and the earlier checks that flagged it.
- **Clusters.** Movers tied by a story that names one of them, or by a year of moving together and the same
  direction today. A tie is a lead. Whether the cause is shared is for the note.
- **Driver reach.** When a driver moves 1.5 times its usual day, the stocks that moved most with it over the year
  and what they did today.
- **Memory.** The last note's themes and open questions, so the reasoning continues from check to check.

The case file holds Yahoo's prices, so it stays in the local home (`var/desk/`), like the list of every checked
stock's move that each check now keeps for it (`var/scanwatch/`). The repo keeps the note.

## The note, and the rules it is held to

A note has a backdrop, themes, singles, the unexplained, calls and open questions (`desk template`).

| Rule | Why |
|---|---|
| Every cause cites items of the case file, or pages the reasoner read and listed under `read` | nothing is asserted from memory |
| A cause is marked `reported` or `inferred`; a reported one must cite a story, a filing or a page | a source's statement and the reasoner's own inference stay apart |
| A theme ties at least two stocks and says the cause, how it reaches each of them, and what lies beneath | a shared headline is not a mechanism |
| Every stock outside its 9-in-10 range is in a theme, a single, or the unexplained list | "no news found" is said out loud, not skipped |
| A read-through (a stock the theme may also reach) is printed as untested | it is a hypothesis |
| Every forward-looking statement is a call with its chance and its date, recorded before the outcome | `CLAUDE.md`: a call is recorded first |
| A page under `read` that names a ticker is kept in the bank as a claim | `CLAUDE.md`: what is read goes into the bank with its source |
| A note is recorded once and never edited | like every record here |

The recorded note (`research/desk/<scan date>/<check>.json`) carries the text of every item it cites, so it can be
read without its case file.

## What it is, and what it is not

- **It explains.** A note says why stocks moved together and what the numbers underneath are. It reads sources the
  check only listed.
- **It is not evidence.** A good story is not a tested signal. In this repo, links between causes and stocks
  explained the same month and failed the next (L1), results-day news showed no drift after it (E1, E2), and no
  direction test has passed.
- **Its forward claims are scored.** The desk's calls go to `research/desk_calls.jsonl` and `desk score` judges them
  like any other call (`docs/PREDICT.md`). Under 30 judged calls the record says nothing, either way. A read-through
  that names no call is an observation to check at the next note, not a forecast.
- **Partners describe the past year.** Two stocks that moved together for a year can stop. Whether a stock's news
  reaches its partners the next day is a hypothesis nobody has tested here yet; the way to test it is a declared test
  (`CLAUDE.md`, "Turning an idea into a tested claim").
- **Public information only,** read under the web rules. Executives appear in their public role only. Nothing about
  private individuals is kept.
