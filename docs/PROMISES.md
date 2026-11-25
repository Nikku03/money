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
python -m brain promisetest declare|test  # the declared tests G1a-G1d
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
four): how many it kept, how many it beat, and its **usual surprise** (the median).

- `stretch` says how much a new promise asks: its midpoint's growth over the year-ago quarter, less the growth the
  company last reported.
- A yearly promise is compared with the company's previous promise for the same fiscal year: raised, cut or held
  (a move of 1% or more).

RESULTS_PLACEHOLDER
