# The agent's calls, and the world radar

Two tools for working with a person day to day. Neither is a tested signal.

## `brain predict`: every call, written before the outcome

When the agent tells someone "I lean down on PYPL into its report", that is a forecast, and it is only worth what
its record shows. So each one is written down first, with the chance it is right and the day it is judged:

```
python -m brain predict add PYPL down --prob 0.55 --by 2026-10-27 \
    --thesis "branded checkout stuck at 1-2%; stock ran up with the sector" \
    --source https://... --trade "PYPL 2026-10-30 52 put" --paid 0.82
python -m brain predict close <id> --exit 1.40      # the paper trade's exit, per share
python -m brain predict list [--open] [--refresh]
python -m brain predict score [--refresh]
```

- **The call.** `up` or `down` (optionally by at least `--threshold` percent), or `move` (at least
  `--threshold` percent either way). `--prob` is the chance the call is right. The start price is the bank's
  latest close for the ticker (ingest it first) unless `--price` is given.
- **Judging.** A call is judged on the last daily close on or before its `--by` date, once that close is in
  the bank (the price source keeps a session out until it has closed): hit or miss, and its Brier score,
  `(prob - outcome)^2`. A coin flip at 50% scores 0.25; lower is better, and only calibrated confidence earns it.
- **Paper trades.** A call can carry the trade riding on it: what was bought and what it cost. `close` records
  the exit; `score` adds up the profit and loss on paper. Nothing is ever traded.
- **The record.** `research/predictions.jsonl`, append-only and committed. A call is never edited; a mistake is
  withdrawn with `predict withdraw <id> --reason "..."` and stays on file. Fewer than 30 judged calls say
  nothing either way, the same bar the engine sets for any source (CLAUDE.md, "Claims are not evidence").

## `brain radar`: what moved hard, world-wide

```
python -m brain radar --refresh      # re-read world markets and US-listed country funds, then scan
python -m brain radar --all          # every market, flagged or not
```

It reads the world markets of `brain/markets.py` and about two dozen US-listed country funds (Brazil, Korea,
India, Indonesia, Taiwan, ...), and flags any whose move over a day, a week or a month, or whose fall from its
three-month high, passes its kind's threshold (indices and funds 3% / 7% / 10% / 15%; FX, commodities, crypto
and volatility have their own). A symbol whose last close is more than five days old is marked STALE. It
reports what already happened, early and in one place; it forecasts nothing.
