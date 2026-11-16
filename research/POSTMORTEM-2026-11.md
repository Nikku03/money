# Why it keeps failing: three sessions, one record

Written Nov 16, 2026, from the memory bank (`docs/MEMORY.md`): every result recorded in three sessions between
2026-07-29 and 2026-11-15. Nothing here is a new test; each number is as that session recorded it.

| Session | Dates | What it tried |
|---|---|---|
| `session_018ANuE3nDWJsR8htWjJDJhG` "High variability trading ML model" (repo `integratedai`) | Jul 29 - Oct 21 | catalyst and moonshot models, filing latency, options around earnings, drug-trial readouts |
| `session_01MSaQVTPoBwSsjdSBSk9kdd` (branch `claude/sleepy-goldberg-2ma18z`) | Nov 4 - 9 | thirteen self-improving paper traders, P1-P13 |
| `session_018E1bTF9g2P9K8brvhcocRv` (this repo) | Oct 12 - now | declared tests of signals, links, events, ranges, shocks, the mosaic |

## The scoreboard

About seventy ideas were run to a verdict.

**Passed, on data never seen (8 declared tests, all here):** the size of moves (W1a, W1b, W1c, M1a), where shocks
land (S1a), and which filing is coming (P1a, P2a, P3a). None says which way a price goes.

**Failed: every test of direction or profit.**

- *Signals on stocks:* momentum, reversal, value, asset growth, insider buying, earnings reaction and surprise
  (H1-H8); cause-and-effect links a month ahead (L1); news drift (E1, E2); the fade after a jump (S2a, S2b);
  mosaic-flagged stocks lagging (P1b-P3b); copying politicians and activists; volume surges; partners' results.
- *Catalysts and filings (Jul-Aug):* the wide replication (t +0.39 against a required 2.0); 2015-2026 (-0.15% a
  trade); the latency window (16% fillable, 92% of filings arrive with the market closed); buying the open after
  an 8-K (-2.55% over ten days, 1,861 trades); defence contracts; FDA and trial news after the fact (-2.20% on
  940 trades); relative volume; filing text for direction (five times); seventeen exit rules; an 18-question
  checklist (0 of 18 repeated); short interest; quiet accumulation (ran backwards).
- *Exotic models:* Ramanujan's partition and pi formulas; the REM residual model; permutation entropy (narrower
  than sampling noise); the Mobius loop to peers (51%).
- *Options:* buying before scheduled events (three measurements); the butterfly (break-even $0.063 a leg against
  a $0.05 tick); the short vertical spread (+0.8% gross, -18.0% net of $0.05 a leg).
- *The thirteen traders:* each failed on years it did not learn from. $1,000 became $170 to $1,524 where SPY
  made $2,054 to $4,820 over the same years.

## Five causes

1. **The question.** Nearly every idea asked which way a liquid US stock would go, from public information. That
   is the most contested question in markets. Public news is in the price by the next trade: the move is in the
   gap, and the gap cannot be bought.
2. **The bet is priced too.** Where something true was found, the instrument already charged for it. The credit
   spread needed a 61% win rate and the signal supplied 61.9%. Earnings options charged about what earnings
   delivered. A drug readout paying +174% or -60% breaks even at 26%, about the odds on offer. Costs took the rest.
3. **Each win was an artifact.** The "$3,000" trader (P5) held a stock list chosen in 2026 and had a look-ahead
   bug; on lists a trader could have held it averaged +0.26% a week, t 1.0. A pilot's +9.86% on 11 trades became
   -2.20% on 940. A 61.9% hit rate on 118 events became 50.7% on 4,917. One "2,842 events" was 16. Edges gross
   of costs lived in contracts too thin to trade. The tests caught every one, which is the system working.
4. **The hunt went where the odds are worst.** Moonshots, the most volatile stocks and weekly options are where
   buyers overpay for a lottery ticket, costs are largest and samples smallest. "Chasing volatile stocks loses
   after the stops" held in all three periods it was checked on.
5. **Hopping.** Seventy ideas in fifteen weeks, most judged within days and dropped. The few hints were left as
   footnotes while the next idea began, some dead ideas were proposed again, and the forward tests that could
   settle anything need thirty weeks each.

## What survived, and how far

| Finding | Evidence | Missing |
|---|---|---|
| Move sizes and shock odds | passed on 2024-2026 | says nothing of direction |
| Which filing is coming (share sale, late accounts, cuts) | passed on 2024-2026 | flagged stocks did not lag |
| Boring beats exciting: profitable, cash-rich, near the high, not diluting | 15 of 16 lessons kept their sign in a third period | the traders built on them still lost to random picks |
| Buying after fear spikes | +0.51% a week in fear weeks, -0.62% in calm ones; held in 2013-2015 | never declared as its own test |
| Expensive stocks (price to sales of 8 or more) reverse into earnings | 55.5% on 656 unseen events, p 0.0055 | +0.5% a trade, interval crossing zero; borrow cost unmeasured |
| Our size forecast against the options' price | 7 disagreements moved 1.46 times the priced move, p 0.08; M1's first 99 calls showed no skill | V1 and O1 are running; they need real quotes and 30 weeks |
| A researched drug readout, entered before it | one real trade, KOD, +174% | entering after the readout lost on 940 trades; entering before was never tested |

## What the best version looks like

No strategy is certain to beat the market, and this record gives no reason to expect a large edge. The best
design it supports:

1. **Hold the index as the core.** Every version lost to it. The bar is beating SPY, not beating zero.
2. **Tilt only with what held:** away from lottery stocks and diluters, toward quality, and add after fear spikes.
   Declare it once, against SPY, on a list a trader could have held.
3. **Use the size engine where it is proven:** position size, stops, and which option to use. V1 and O1 decide
   whether its disagreements with option prices pay.
4. **One specialist line: binary events entered before, small and many.** A probability for every pivotal readout,
   written down before it, set against the odds the options imply, scored after fifty of them.
5. **One idea at a time.** Search the memory bank before testing; require a repeat on a second period; wait for
   thirty forward bets; buy the two missing data sets (option quotes, prices of delisted stocks) if it is to go
   further.
