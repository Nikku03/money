# Event tests: does a kind of corporate event move the stock?

The first event tests ask about guidance: when a company **raises** its guidance in an earnings press release, does the stock move?

**Version 1 (E1) didn't test cuts.** Among the 60 stocks' releases since 2011 the rules find only 10 guidance cuts, 4 of them in TEST, too few to say anything.

**Version 2 (E2) asks the same of small and micro caps,** raises and cuts both: see [Small and micro caps](#version-2-small-and-micro-caps).

## The events

- **Where they come from.** Every earnings 8-K (item 2.02) in the bank has its press release as exhibit 99.1. `python -m brain releases` reads each release's opening from SEC EDGAR: the headline, the dateline and the highlights.
  - The SEC's rules apply: our `SEC_USER_AGENT`, and under 10 requests a second through the shared SEC bucket.
  - Each release is kept as a `release.earnings` fact: its title, its opening text, and the exhibit's URL.
  - Its `known_at` is the filing's acceptance time, so a test sees the release only once it was public.
- **How they are tagged.** Fixed rules (`brain/web/events.py`) tag the opening text, with no model.
  - **guidance_raised:** "raises full-year 2026 revenue and EPS outlook", "increases fiscal 2025 guidance".
  - **guidance_cut:** "lowers full-year outlook", "cuts guidance", "profit warning". Verbs only: "lower than forecast" and "the lower end of our guidance" are not cuts.
  - Guidance for spending, costs or taxes doesn't count either way: a lower capex outlook isn't bad news.
  - A phrase can't reach across a clause break or past words that say guidance didn't move: "reaffirms", "maintains", "unchanged". "Raises dividend; reaffirms guidance" is not a raise.
  - A release tagged both ways ("raises revenue guidance, lowers EPS outlook") counts as neither.
  - The rules are applied at test time to the stored text, so the declared code fixes them.
- **Universe.** Version 1: the 60 stocks of `universe/us_large.json`, from December 2010 on. Version 2: the small and micro caps below.

## The tests

Each event kind is tested in two windows. Day 0 is the first close at or after the filing became public (`brain/events.py`).

| Window | Days | Gate | Asks |
|---|---|---|---|
| reaction | -1..+1 | REPORT | Does the event move the stock when it comes out? |
| drift | +2..+21 | gated | Does it keep moving afterwards, when an investor could still act on it? |

- **The comparison.** Each month, the tagged releases' mean excess return over SPY minus the mean of the month's other earnings releases.
  - Every release moves in an earnings season. Only the gap says whether the event matters.
  - The move against SPY alone is shown beside it.
- **The statistic.** The mean of the monthly differences: a month counts once, however many releases it has. Its p-value is a sign flip, 20,000 draws. Version 1 makes it one-sided, in the direction the event implies: up for a raise, down for a cut. Version 2 makes it two-sided (below).
- **Controls, on each test's TEST months.**
  - **Placebo:** untagged releases drawn at random in place of the tagged ones must reach p < 0.05 in at most 10% of 200 draws.
  - **Oracle:** the month's best (or worst) releases in their place must reach p < 0.01.
  - If either fails, the verdict is void.
- **Gate (drift).** PASS when all four hold:
  - the difference points the way the event implies (version 1; either way counts in version 2, and the result says which);
  - p < 0.05;
  - the controls pass;
  - Benjamini–Hochberg q < 0.10 across every gated test in the ledger: hypotheses, forecasts, links and these.
- **The usual rules.**
  - Declared before running (`python -m brain events declare`, then commit), and frozen: a change is a new `VERSION`.
  - TRAIN 2011–2018, TEST from 2019 to two months before the latest prices, so every drift window has closed.
  - The data is read as the bank held it at declare.
  - Results are written to the ledger and the Events section of `research/RESULTS.md` whatever they say (`python -m brain events test`, then commit).
  - `--trial` reads TRAIN years only, in the trial ledger.

## Version 2: small and micro caps

E1 found a raise lifting a large cap by about 1.4 points on days -1..+1 against the month's other releases, then the stock giving back about 2 points over the next month: the other way from the drift E1 declared. Small and micro caps are where a slow reaction is most often claimed, so version 2 asks there.

- **Universes.** By the same rule as the broad universe, on lower rows of SEC's `company_tickers.json` (roughly by market value): SEC XBRL from 10-Ks, and Yahoo daily prices since 2010.
  - **small:** rows 1501–3000, `python -m brain universe --band small` → `universe/us_small.json`.
  - **micro:** rows 3001–4500, `python -m brain universe --band micro` → `universe/us_micro.json`.
- **The lists are in the repo** (`universe/us_small.json`, 510 stocks, and `us_micro.json`, 332, built 2026-10-19). A rebuild starts again from the first row and re-reads SEC's listing, which changes daily, so it can give a different list. Each declared test pins its list's sha and refuses a changed file: a rebuilt band is a new experiment with new ids. The listing was re-read after `us_broad.json` was built, so 11 stocks near row 1500 are in both.
- **Releases.** `python -m brain releases --band small`, then `--band micro`. That's tens of thousands of releases at under 10 SEC requests a second, so it takes hours. Re-running it resumes: releases already held are skipped.
- **Declaring.** `python -m brain events declare --band small` once that band's releases are read, and the same for micro. A band is refused while more than 2% of its earnings filings have no release read. The declaration pins what the test will read (`data_sha`: every release and its window return), so a fact downgraded or added later makes the test refuse rather than quietly change.
- **Unread releases.** About 1 release in 5 is filed under a name that doesn't say it's exhibit 99 ("q2fy18pr.htm", "ex_956944.htm"). The first reader kept those with no text. The reader now also reads the filing's index page, which states each document's type (EX-99.1), so it finds them. Running `brain releases` again reads the ones kept empty, and the empty fact is downgraded as superseded, not deleted.
  - Any release still without text is left out of both groups in version 2, and RESULTS.md says how many.
  - Version 1 counted them among the "other releases". That dilutes its comparison a little, and its results stand as recorded.
- **Tests.** `E2-small-…` and `E2-micro-…`, for guidance_raised and guidance_cut, each in the reaction and drift windows.
  - **Either way.** The drift p is two-sided. E1's drift pointed the other way, so a direction declared in advance would only be a guess.
  - **Cost.** A round trip in such stocks costs about 1% (small) and 2% (micro) in spread and impact. RESULTS.md states it beside each result: a drift smaller than that couldn't be acted on, whatever its p.
  - Everything else is as above: the same rules, windows, controls (the oracle uses the month's best releases), TRAIN/TEST split, and the one false-discovery correction across the ledger.
- **Bias.**
  - **Survivorship.** Each universe is chosen by today's market value, and every member has been listed since 2010. Companies that failed or were bought out are missing, and small caps fail more often.
  - **Reported events only.** A small company that never issues guidance never appears.

## Not tested: analyst downgrades

Rating changes are not filed anywhere public, and the free histories of them sit behind access tokens meant to stop automated reading, which the repo's rules forbid getting around.

The web gateway (`docs/WEB.md`) tags downgrades and upgrades in headlines as they are published, so a record builds from now on. Once it has a few hundred tagged headlines, a downgrade test can be declared on that record.
