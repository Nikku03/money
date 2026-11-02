# Move sizes (M1): how far a stock moves on its earnings, whichever way

Every direction signal tested in this repo has failed (H1-H8, the links, the event drifts, the forecaster's chance
to beat SPY). How *much* a stock moves is different: it persists, and the forecaster's price bands held their
stated odds. M1 predicts that size for an earnings release, so it can be set against what the options charge.

## What it predicts

The size of the release's move over days -1..+1 against SPY, either way: the unsigned `car_3d` of
`brain/events.py`, from the close two sessions before the release's first session to the close one session after.
M1 gives a range: the 10%, 25%, 50%, 75% and 90% points, and the chance of moving more than any given size.

It reads four things known at the close two sessions before the release:

| Feature | What |
|---|---|
| `med8` | the median move of the stock's last eight releases (at least four) |
| `last` | the most recent of those moves |
| `day_vol` | the standard deviation of its daily moves against SPY over the 160 sessions before, release days left out |
| `vix` | the VIX close |

The model is least squares on logs. The spread of its misses over the training years sets the range; the same spread
gives the chance of beating a size. The baseline it must beat is the same fit on `med8` alone: the stock's own history.

## Fitted once, tested once

```
python -m brain moves fit            # fits M1 and the baseline on 2011-2023, saves knowledge/moves_m1.json
python -m brain movetest declare     # declares M1a, pinning the code, the model's sha and the universe files
# commit brain/, knowledge/ and research/ledger.jsonl
python -m brain movetest test        # runs M1a once on 2024-01-01..2026-09-30; commit the ledger and RESULTS.md
```

- **The years.** The fit reads only releases through 2023-12-31. Exploration before the fit used only 2011-2023
  (trained on 2011-2019, checked on 2020-2023). The releases of 2024 onward were seen only as the earnings radar's
  recent "usual moves", never fitted or scored.
- **The test.** M1a scores each TEST release with the mean pinball loss over the five quantiles: lower is better,
  and it rewards a range that is both right and tight. Each month, the baseline's mean loss minus M1's; the
  statistic is the mean of those monthly differences, with a one-sided sign-flip p.
- **The gate.** M1a passes when:
  - the difference is positive and p < 0.05;
  - M1's 80% range holds between 75% and 85% of TEST releases;
  - the controls pass: ranges shuffled within the month reach p < 0.05 in at most 10% of draws, and the actual
    move used as the middle reaches p < 0.01;
  - Benjamini-Hochberg q < 0.10 across every gated test in the ledger.
- **Expectation, stated before running.** The range should hold its odds, as the forecaster's bands did. The gain
  over history alone should be modest: a stock's own past moves already carry most of what is knowable.

## Using it

```
python -m brain moves predict AAPL CE --implied 4.3
```

This prints each ticker's range for its next report and, with `--implied`, the chance of moving more than the
options price. The options' priced move comes from `brain earnings radar`, which also shows M1's range beside it.
How to read the chance:

- **Low chance** (the options price a move M1 rarely sees): the options are dear, and the edge, if any, is the
  seller's. Only a defined-risk spread fits a small account, and selling carries the rare large loss.
- **High chance:** the options are cheap, and the buyer's side has the edge.
- **In between:** no bet.

Every such call goes into `brain predict` first, with M1's chance as its probability. M1's own calls, one per
confirmed report with options, are kept apart from the agent's in `research/m1_calls.jsonl`:

```
BRAIN_PREDICTIONS=research/m1_calls.jsonl python -m brain predict score --refresh
```

Each is a `move` call: the options' priced move for the report is the threshold, and M1's chance is the
probability. It is made at the close two sessions before the release's first session and judged at the close one
session after it. That is M1's own window, measured on the stock's raw price rather than against SPY.

M1a's PASS (`research/RESULTS.md`, Move sizes) is about the range against the stock's own history. Whether M1 reads
the options' price better than the options do is unproven until thirty of these calls are judged. A size is not a
direction, and nothing here trades.
