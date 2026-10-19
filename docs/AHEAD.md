# Looking ahead: this week's events and what moved beneath the surface

`python -m brain ahead` lists what is coming in the next days for the companies you watch, who each event ripples to, and what is already public but little read.

```
python -m brain ahead --universe large            # the 60 large caps, next 7 days
python -m brain ahead NVDA MU LRCX --days 14
python -m brain ahead --universe large --at 2026-07-15   # as the bank knew it then
```

Each run is saved to `<home>/ahead/<start>.json`, so what was flagged can later be checked against what happened.

## Scheduled

The date of these events is known in advance; the outcome isn't.

- **Earnings releases.** Expected from the company's own history: the release (an 8-K with item 2.02) about a year earlier, plus 52 weeks, shifted by how its dates have drifted year over year.
  - Each row says how regular the company has been ("same weekday every year", or "usually within N days").
  - A quarter whose release is already out is not listed.
  - Companies announce their dates in press releases, which the bank doesn't read. So a row is an estimate, and it says so.
- **FOMC decisions.** From the Federal Reserve's published calendar (`federalreserve.gov`), including which meetings publish projections.
- **Market dates.** Option expiries (the third Friday; March, June, September and December are the quarterly ones), and month and quarter ends.

BLS release dates (CPI, jobs) are not included: bls.gov refuses automated requests, and the repo doesn't get around bot checks.

## Beneath the surface

- **Planned insider sales.** Form 144 notices are filed before an affiliate sells restricted or control shares, so they announce sales that haven't happened yet. The count per company in the last week is shown. The sellers are not named here.
- **Quiet changes.** A report filed in the last two weeks is flagged when its similarity to the same report a year earlier (`docs/CHANGES.md`) is at least 1.5 standard deviations below the company's own usual. "Unusual for this company" matters more than the raw number, since some companies rewrite more than others. `brain changes show T` lists the new sentences.

## Ripple

For each company with an event, `ahead` lists its listed customers, suppliers, partners and competitors from the relationship graph (`docs/GRAPH.md`). It uses the company's own 10-K and other companies' 10-Ks about it. If NVIDIA names Micron a supplier, Micron's row shows NVIDIA as a customer.

## What this is not

It's a map of where to look, not a forecast. The quiet-change score was tested as H10 on the 60 large caps and failed (`docs/CHANGES.md`): it shows what changed, not which way the stock goes. Planned insider sales and the ripple are not tested yet, and each would be a declared hypothesis of its own before anything relies on it.
