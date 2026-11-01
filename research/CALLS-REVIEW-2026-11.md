# Review of the first calls, Oct 26 - Oct 30, 2026

Written Nov 1, 2026, from `research/predictions.jsonl` (`python -m brain predict score`), the options study of
Oct 24 and the bank's closes. Ten calls are far too few to judge skill (under 30 says nothing); this records
what each call rested on, what decided it, and what to change.

## The recorded calls

| Call | Chance | Outcome | What decided it |
|---|---|---|---|
| PYPL down into Q3 | 55% | miss, +2.0% | EPS beat ($1.45 vs $1.31), FY guide raised to ~$5.60 from ~$5.38; branded checkout was weak (1%), as called, but did not matter |
| PYPL down 5.9%+ (52 put, paid 0.82) | 28% | miss; put sold at 0.10, -$72 | as above, and the options' event premium drained away the morning after |
| KO up into Q3 | 55% | miss, -1.6% | beat (EPS $0.91 vs $0.88, volume +5%), yet the stock slipped: a beat is not a rise |
| KO move 3.8%+ (strangle) | 25% | miss | small reaction |
| CNC up into Q3 | 60% | hit, +1.0% | peers beat and raised; much was already in the price (+13% the month before) |
| UPS down into Q3 | 55% | miss, +5.8% | beat ($1.69 vs $1.63), 2026 outlook raised, inbound volume recovered |
| BBAI down into Q3 | 55% | miss, +16.0% | revenue +31%, FY revenue guide raised |
| IBIT move 5%+ by Oct 30 | 38% | miss, -4.1% | the calmer last month, flagged at the time, was the better guide than the year |
| COIN move 7.5%+ by Oct 30 | 50% | hit, -10.7% | moved more than the options priced, as it had after 5 of its last 8 reports |
| EWZ back to 38 by Dec 18 | 22% | open (32.46 on Oct 30) | |

Judged 9: 2 right (22%), stated chance 47% on average, Brier 0.212 (a coin flip scores 0.250, helped by the
low-chance calls that duly missed). If the stated chances were honest, 2 or fewer hits had a 12% chance; the five
plain direction calls went 1 for 5 (12% if they were really 55-60%).

## The options study (52 Monday/Tuesday reporters, quotes of Oct 23)

- 18 of 52 moved more over three sessions than their options priced for the report: on average the options were
  priced about right or rich, not cheap.
- The 22 straddles expiring Oct 30, bought at Oct 23's mid and held to expiry: 7 won, average -32%.
- The cheap/fair/expensive reading from eight past reports barely sorted them: cheap 4 of 7 moved more than
  priced, fair 8 of 32, expensive 6 of 13. The biggest moves came from names read fair or expensive (GLXY +28%,
  BBAI +25%, MIR +25%, FFIV -22%).
- Exit timing mattered: GLXY was +28% three sessions after its report, but its straddle held to Friday lost 21%.

Usual-move sizes for 39 of the week's reporters: 18 moved more than their usual (average) move; actual moves ran
from 0x (MSFT) to 3.9x (NOW -30% against a usual 7.7%). A usual move is the middle of a wide spread, not a forecast.

## Calls that were made but not recorded

- Brazil after the Lula win: "political crashes often keep sliding; wait 1-2 days". EWZ fell another 4.9% by
  Oct 30. Right, but not in the record.
- SPCX Q3 revenue: estimate $12.84B, actual $14.575B, outside the stated 90% band. Direction right, size wrong:
  the band was too narrow.
- Korea "bounces hardest if AI fears fade": +6.6% on Oct 28, then given back in Friday's AI sell-off (EWY flat).

## What to change

1. No direction call from "it usually falls after reports": 0 for 3 here (PYPL, UPS, BBAI), and the same idea
   failed its tests (H7b; the E1/E2 drift tests).
2. In a beat-and-raise season a down call needs a specific reason (a pre-announcement, a peer's guide-down, a
   known loss), not a mood. Guidance decides the reaction, and a shrinking share count makes an EPS guide easy
   to raise (PYPL).
3. A beat is not a rise (KO): what was expected and what was already in the price decide the move.
4. Options priced the event about right or rich. Buy only with a clear margin, size small, and sell the session
   after the report.
5. Read a usual move as a spread, 0x to 4x, not a number.
6. Check the market first (`brain radar`, index futures): a 2% index day or a sector rotation moves single names
   more than a company thesis.
7. Before any options reading: a pending takeover (ROKU), the volatility bump (MDB), weekend quotes, stale filings.
8. Record every call to the person, including the informal ones, before the outcome.
9. Keep scoring. Ten calls say nothing yet; thirty will start to.
