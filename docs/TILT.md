# Step 2: the quality tilt and buying after fear

`research/POSTMORTEM-2026-11.md` looked back over about seventy ideas in three sessions. Every attempt to predict
which way a stock goes, or to make money from it, failed; every version lost to holding the S&P 500. Its plan
starts with an index core, then step 2: **tilt only with what held, declared once, against SPY, on a list a trader
could have held.** What held, in exploratory looks that were never declared on their own:

- **Boring beats exciting.** Profitable, cash-rich, near the 52-week high, not diluting: 15 of 16 such lessons kept
  their sign in 2013-2015, a period they were not learned on. The traders built on them still lost, but those
  traders hunted only among the most volatile third of stocks, which is what the lessons say to avoid.
- **Buying after fear.** Trades taken in weeks after fear spiked made +0.51% a week; in calm weeks, -0.62%.

The same ideas are in the published research: high profitability (Novy-Marx 2013), low volatility (the "low-risk
anomaly"), nearness to the 52-week high (George and Hwang 2004) and low share issuance (Pontiff and Woodgate 2008)
were each documented on data before 2013.

```
python -m brain tilttest declare      # once, with the code committed; then commit the ledger
python -m brain tilttest test         # once; then commit the ledger and research/RESULTS.md, whatever they say
```

The code is `brain/research/tilttest.py`; its docstring is the full declaration.

## T1a: the quality tilt

**The list.** On each month's last session, the 500 largest stocks by market value *as known that day* (shares
from SEC filings times the close). Candidates are today's largest SEC filers with XBRL accounts (the union of the
five universe files, about 2,500 stocks), so a company that was small in 2011 joins the list only when it became
large. A stock needs a price that day and at least three of the four traits.

**The traits**, each a rank among the 500 that month, then averaged into a quality score:

| Trait | Measured as |
|---|---|
| Profitable | net income over the last four quarters, over total assets |
| Not diluting | minus the growth in share count over about a year |
| Not a lottery ticket | minus the volatility of daily returns over the last year |
| Near its high | the close over the highest close of the last year |

**The portfolio.** The top third by score, equal weight, rebalanced monthly, 10 bps of cost per unit of turnover,
through the same harness as H1-H8 (the signal at a month's close, the trade at the next day's close).

**The test.** TEST is every month from January 2019 whose outcome is known. The benchmark is the list itself at
equal weight, so whatever the list as a whole earned cancels and only the tilt is judged. p comes from shuffling the
score among the 500 each month, 5,000 times. The gate: beat the list after costs, p < 0.05, the controls passing,
and a false-discovery q < 0.10 across every test ever run here.

**Against SPY.** Reported beside the verdict, not gated: the tilt, the list, and "the index without its junk" (the
list weighted by market value, less its bottom third by score) each against SPY and against RSP, the S&P 500 at
equal weight. RSP is a real fund that held the companies that later failed or were bought; our list cannot, because
it is built from companies still filing today. The list's own gap to RSP measures how much that flatters it.

**Honest limits.** The traits were picked from research and from our own earlier looks at 2011-2026 (on other
lists), so TEST is not unseen: a PASS is an upper bound, and a FAIL weighs more. The clean check is a forward
record, month by month.

## T1b: buying after fear

**The rule.** At each week's last close, a fear spike is the VIX closing at least 25% above its average of the
previous 63 sessions (about three months). For 13 weeks after a spike, hold SPY at 100%; otherwise hold 70% and keep
30% in Treasury bills. The change takes effect at the next session's close and costs 10 bps per unit.

**The test.** TEST is February 1993 to December 2010: weeks none of our sessions had looked at (the lesson came from
2011-2026). The benchmark is a constant mix holding SPY at the same average weight, so only the *timing* is judged,
not how much stock is held on average. p comes from sliding the 100%/70% pattern in a circle against the returns by
at least a year, 10,000 times. The gate: beat the constant mix, p < 0.05, the controls passing (an oracle that
knows each week's outcome must be seen; random spike weeks must pass about 5% of the time), and q < 0.10.

**Reported beside it, not gated:** 2011 to now (seen before), and the timed portfolio against SPY held at 100%,
which is what a person would actually compare with: holding 30% in bills most of the time costs return in a rising
market.

## Results

Written after the run, whatever it says.
