# The company file: everything the bank holds on one company, on one page

Asked for on 2026-11-25: "if we know the company in and out, how it works and doing internally, the pe ratio ...
who it trades with ... its next projects and what it need, and according to timeline, trace every fabric of the
company, we can tell how much stock will move, and if it can deliver."

```
python -m brain file NVDA            # about 20 seconds a company
python -m brain file NVDA AMBA DE
```

The file adds nothing of its own (`brain/file.py`). Each section is one of the engine's cards, printed as that
card prints it, with the verdict of the test behind it as recorded in `research/RESULTS.md`.

| Section | What it shows | From | Tested by |
|---|---|---|---|
| 1. What it is | industry, exchange, fiscal year end, market value | the bank's identity facts, `brain/metrics.py` | facts, not a claim |
| 2. Its accounts | the latest fiscal year and quarter as filed; margins; cash and debt; P/E on the fiscal year and on the last four quarters; price to sales | SEC XBRL (`brain ingest`) | facts, not a claim |
| 3. What it promised, and whether it delivered | its record of revenue promises, its open promises, where its habit puts the next delivery | `brain/promises.py` (`docs/PROMISES.md`) | G1a |
| 4. What it needs and what is on file | cash against what operations use, a registration to sell shares, past sales, late filings, officer changes; the chance of a share sale, late accounts or a write-off in 20 sessions | `brain/mosaic.py` (`docs/MOSAIC.md`) | P1a, P2a, P3a (and P1b-P3b: no direction) |
| 5. Who it moves with | the stocks that moved most with it over a year, and their latest results | `brain/connect.py` | R1b-R1d: what follows a partner's move |
| 6. Its dates | its last four reports with how the stock took them; a guess at the next report; its fiscal year end | `brain/episodes.py` | the reaction's name is a description |
| 7. How far it can move | the range for the next five sessions, the chance of a 20%+ day, the size of its next report's move, and (within 14 days of a report) the run into it | `brain/scan.py`, `brain/shocks.py`, `brain/moves.py`, `brain/expect.py` | W1a or W1b, S1a, M1a, X1a |
| 8. Not in the file yet | what the engine does not read | | |

## Rules for quoting it

- Quote a line as printed, with its verdict. A FAIL beside a line means the line describes; it does not forecast.
- Section 7 gives sizes. None says which way. No test of a direction has passed.
- Section 5's partners are stocks that moved together, whatever their business. They are not the company's
  customers or suppliers.
- Section 2's P/E is the price over earnings as filed (GAAP). A one-off gain or loss sits in it. The two P/E
  figures can differ widely for that reason: read both.
- Section 2's cash leaves out investments; section 4's "liquid assets" counts them.
- The next report in section 6 is a guess: a year after last year's. `python -m brain expect radar --confirmed`
  reads the week's calendar.
- An account item last tagged years ago (a company that changed how it labels a line) is left out, not shown as
  this year's.
- A section whose piece is missing says so ("not available", "no ... in the bank") and the file goes on.

## What it does not hold yet

The person's direction is to trace every fabric of a company. Still missing:

- **Customers and suppliers by name.** A 10-K must name a customer of 10% or more of revenue; that text is not
  read yet.
- **Projects and what they need.** Spending a company has planned against the cash it has.
- **Guidance as a growth rate or as earnings per share.** Section 3 reads revenue in dollars only. A count on
  the research universe's releases of 2024-2026 (`research/promises/survey_forms.py`; loose patterns, so an upper
  bound): 199 of 741 companies gave a revenue promise in dollars; 215 others gave revenue growth in percent and
  262 others gave earnings per share (the two groups overlap). With all three forms the record would reach at
  most 523 of the 741 companies, against 199 now.
- **Earnings calls.**
- **A direction.** Every direction test so far has failed or was not gated.
