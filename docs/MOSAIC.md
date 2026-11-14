# The mosaic: what public pieces say about things only insiders know yet

People inside a company know some things before the public does: that it is about to sell shares, that its accounts
will be late, that it is about to cut back. Trading on that knowledge is illegal, and nothing here uses it. But
each of those things becomes public later through a dated SEC filing, and before it does the public already holds
pieces that bear on it. A mosaic sets the pieces side by side and asks what they point to.

```
python -m brain mosaic read             # read each prospectus's first page: shares, bonds, or a resale (resumes)
python -m brain mosaic fit              # fit the pictures on 2011-2023
python -m brain mosaic show VIR CADL    # one stock's pieces, each with its filing, and each picture's chance
python -m brain mosaic radar            # the stocks each picture points to most, with the pieces that say so
python -m brain mosaictest declare      # the declared tests (commit, then: mosaictest test)
```

## The pieces

Every piece is public, dated, and read as it stood at a close (`brain/mosaic.py` lists them all).

| Piece | Where it comes from |
|---|---|
| Quarters of cash left | SEC accounts (XBRL): liquid assets over what operations used in the latest year |
| A registration to sell securities on file, or just declared effective | S-3, S-1, F-3 and EFFECT filings |
| Past sales of new shares, and how long ago | 424B5, 424B4 and 424B1 prospectuses whose first page sells shares |
| Bonds or preferred shares sold; holders registering to resell | the same forms, read by their first page; 424B3, 424B7 |
| A jump in the price, the run over one and three months, the distance from the year's high | prices |
| Results just reported, or due within eight days | 8-K item 2.02, and last year's date |
| Late-filing notices; a report overdue against the company's own habit | NT 10-K, NT 10-Q; 10-K and 10-Q dates |
| A change of auditor; a notice not to rely on past accounts; amended reports | 8-K items 4.01 and 4.02; 10-K/A, 10-Q/A |
| A listing-rule notice | 8-K item 3.01 |
| Officers or directors arriving or leaving | 8-K item 5.02 |
| Letters between the SEC's staff and the company | UPLOAD and CORRESP, from the day they are made public |
| Past restructurings and write-offs | 8-K items 2.05 and 2.06 |
| A net loss; liabilities above assets | SEC accounts |

Most prospectuses of the sale forms sell bonds, not shares (utilities, banks and property companies file them all
the time). So `mosaic read` reads the first page of each one and files it as shares, debt, preferred or a resale
by holders. Only shares sold by the company count as a sale.

## The pictures

A picture is one hidden event, the filing that will reveal it, and a chance that it comes in the next 20 sessions.

| Picture | What insiders know | The filing that reveals it |
|---|---|---|
| **SALE** | the company is about to sell new shares | a prospectus for common shares that opens a deal |
| **BOOKS** | the accounts will be late, or the old ones are wrong | a late-filing notice, or an 8-K item 4.02 |
| **CUTS** | it is about to cut back or write assets off | an 8-K item 2.05 or 2.06 |

- An event *opens* when no filing of its kind came in the ten sessions before: a deal's preliminary and final
  prospectus are one deal.
- Each picture has a **model** of all its pieces and a **baseline** that knows only the company's habit: how often
  and how lately it has done this before. Both are ridge logistic regressions fitted on 2011-2023
  (`knowledge/mosaic_p1.json`). In a logistic regression each piece multiplies the odds, which is what a mosaic is.
- The card and the radar say which pieces raise a stock's chance most.

**Not offered: a coming takeover or a failure.** The universe is the companies listed today, so the ones that were
bought or that failed are missing, and such a picture could not be tested. The card shows the pieces (a holder
of 5% or more who may seek influence, for one) without a chance.

**Looked at and not declared: the coming quarter.** Whether a stock's results move could be read from its partners'
results, or from its report being late, was looked at on 2011-2023 (`research/MOSAIC-2026-11.md`). Both said almost
nothing, so no test was spent on them. The card still lists the partners' latest results.

## The declared tests

`brain/research/mosaictest.py`, declared before they ran. TEST is 2024-01-01 to 2026-09-30, read at periods at
least 20 sessions apart so that no two outcomes overlap. The fit read nothing after 2023, as a piece or an outcome.

**Does the picture see the event coming?** P1a (SALE), P2a (BOOKS), P3a (CUTS).
- Per stock the score is the log loss of the chance; per period, the baseline's mean loss minus the model's. The
  statistic is the mean of the period differences, with a one-sided sign-flip p.
- It passes when:
  - the difference is positive with p < 0.05: the pieces add to the habit;
  - the model's mean chance is within 0.67-1.5 of the actual rate, overall and in each period's top twentieth;
  - the controls pass: shuffled chances reach p < 0.05 in at most 10% of draws, and an oracle reaches p < 0.01.
- Reported beside it: the same ranking among companies with no such event in three years, where the habit is
  silent and only the other pieces can see anything.

**Do the flagged stocks then lag?** P1b (SALE), P2b (BOOKS), P3b (CUTS).
- Each period's top twentieth by the picture's chance is flagged.
- A flagged stock's outcome is its return over SPY's across the next 20 sessions, set against the period's
  unflagged stocks in its 60-session volatility decile: the share of them it beat, less one half.
- It passes when the mean of the period values is negative with a one-sided sign-flip p < 0.05 and the controls
  pass (a random peer in each flagged stock's place; an oracle at its peers' 10th percentile).
- The mean return against those peers, which a trade would earn before costs, is reported beside it.

All six also need Benjamini-Hochberg q < 0.10 across every gated test in the ledger. The results are in
`research/RESULTS.md`, under Mosaic.

## What it is, and what it is not

- **It is legal inference from public filings.** Every piece cites its filing. Nothing here is, or needs, inside
  information; if something looks leaked, it is not stored (`CLAUDE.md`).
- **A chance that a filing comes is not a direction.** Whether a flagged stock then lags is its own test. Quote a
  chance with its test's verdict, as the card prints it.
- **The habit is most of it.** A company that has sold shares four times in three years will likely do it again;
  the baseline already knows that. What the test asks is whether the other pieces add to it.
- **Survivorship.** Companies that failed or were bought are missing, so the weakest companies' events are
  undercounted and the flagged stocks' returns are flattered.
- **Today's industry codes and today's universe** are read back into the past, as in the shock radar.
- Nothing here trades.
