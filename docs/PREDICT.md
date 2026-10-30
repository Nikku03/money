# The agent's calls, the world radar and the earnings radar

Three tools for working with a person day to day. None is a tested signal.

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

## `brain earnings`: who reports next, how far they usually move, and what the options price

```
python -m brain earnings index [--ingest-missing]   # weekly, slow: report dates and past reactions, saved in the home
python -m brain earnings radar [--days 5] [--budget 100]
python -m brain earnings misses [--date YYYY-MM-DD] # a day's big movers among 8-K filers, and whether the radar had them
```

- **Who.** The research universe (`universe/`) plus the largest ~1,000 SEC-listed companies (the "top" band).
  `--ingest-missing` reads filings and prices for any the bank lacks.
- **When.** Last year's same-season report plus 52 weeks (the same weekday), from the companies' own 8-K item
  2.02 filings. "Likely" when that lands in the window, "possible" (shown `~date?`) within 8 days of it:
  companies often move a report a week. The radar first re-reads each candidate's latest filings, so one
  that already reported this season drops out. A date is an estimate until the company announces its own.
- **How far.** The median of the stock's last eight report reactions (`car_3d`, days -1..+1 vs SPY).
- **What the options price.** The at-the-money straddle on the first expiry after the report (Cboe's public
  delayed quotes, `brain/sources/options.py`, one request every 2.5 s). When that expiry is weeks away, the
  ordinary days in between are stripped out with the stock's normal daily volatility, so the report's own
  priced move is compared on the same three-session basis as its history. `CHEAP` means the stock moved more
  than that after at least 60% of its last reports (5 of 8), `EXPENSIVE` after at most 30% (2 of 8). A strangle at the
  priced width shows the cheapest two-sided bet; `[fits budget]` marks a bet at or under `--budget`.
- **Is a report really coming.** When the first expiry's at-the-money volatility is under 1.15 times the next
  expiry's, the options see no event in between: the row says `[check the date]`, and its verdict means
  nothing (an estimate a quarter off, as for a company that reports in December). Likely dates are listed
  before possible ones; replayed on Oct 29, 2026, sorting the likely list by usual move put 8 of its 20 big
  movers in the top 100, about twice what a random order would.
- **Did it work.** Each radar run is saved; `misses` lists one session's movers of 10%+ among companies that
  filed an 8-K that day or the evening before (SEC's daily form index), with the cause their filing names, and
  whether a radar made before that day had them in its window. Movers with no 8-K (trial data, rumours,
  sector moves) are outside it.

A size is not a direction, and no part of this has been tested as a signal. When the user acts on a cheap
reading, record it with `brain predict add ... move --threshold <the priced move>` first.
