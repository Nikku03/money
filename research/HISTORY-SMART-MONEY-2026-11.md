# What the public footprints of "smart money" were worth, 2015-2023

Written Nov 1, 2026. **Exploratory.** No hypothesis was declared before looking, so nothing here is a tested
claim. Trades and filings from 2024 on were not looked at: they are kept for a declared test.

The question: had we copied politicians' disclosed stock trades, or activist funds' Schedule 13D stakes, as soon as
each became public, would we have beaten the market?

## Data

| Group | Source | Usable |
|---|---|---|
| Senators | [Senate Stock Watcher](https://github.com/timothycarambat/senate-stock-watcher-data)'s public copy of the Senate's periodic transaction reports, joined to each report's received date | 4,214 stock buys and sales by 34 senators, reports filed 2015-2020 |
| House members | The House Clerk's yearly index ([2020FD.zip](https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2020FD.zip) ... 2023) and its 2,055 electronic transaction reports (PDFs), read with `pdftotext` | 14,795 stock trades and 512 option trades, reports filed 2020-2023 |
| Activist-style stakes | EDGAR's quarterly form indexes ([2019 Q1](https://www.sec.gov/Archives/edgar/full-index/2019/QTR1/form.gz) ...), initial `SC 13D` filings 2015-2023; the subject is the party with a current SEC ticker | 4,528 of 11,809 filings; 138 by well-known activist funds |

Prices are daily closes adjusted for splits and dividends (Yahoo, kept in the bank's home, not committed).

## How it was measured

- **Excess return**: the stock's return minus SPY's over the same sessions (for 13D targets, mostly small
  companies, minus IWM, the Russell 2000 fund).
- **What they got**: from the close on the trade date. If politicians traded on what they knew, it shows here.
- **What a copier could get**: from the close of the first session after the report reached the public. This is
  the number that matters for us.
- **Sales** count as bets the stock will lag; a sale is right when the excess afterwards is negative.
- **Clustering**: trades bunch (one report lists many; members trade the same names in the same weeks), so each
  calendar month's trades are averaged first and the t-statistic is taken across months. With about forty
  slices, one |t| near 2 is expected by chance.
- **Luck**: each member's record is compared with 2,000 draws of the same number of trades at random from all
  members' trades. The question is whether the best record beats the best record that luck alone produces.

## Politicians

The median Senate report reached the public 20 days after the trade, the median House report 29 days after
(10% of House trades over 230 days, from a few members' catch-up filings). The House columns below are trades
dated 2019-2023.

| Excess vs SPY | Senate buys (2,175) | Senate sales (2,039) | House buys (6,989) | House sales (7,677) |
|---|---|---|---|---|
| From the trade, 21 sessions | -0.32%, t -0.55 | +0.10% | -0.14%, t -0.29 | +0.36% |
| From the trade, 63 sessions | -0.09% | +0.28% | +0.13% | +0.90% |
| Copying after the report, 21 sessions | +0.01%, t -0.73 | +0.01% | +0.11%, t +0.20 | +0.10% |
| Copying after the report, 63 sessions | -0.30%, t -0.13 | -0.01% | -0.01%, t -0.05 | +0.32% |
| Copying after the report, 126 sessions | +0.83%, t +0.63 | -0.73% | -0.17%, t -0.46 | +0.16% |
| Share that beat SPY over 63 sessions | 46% | 47% | 47% | 46% |

- **No edge**, either from the trade date or for a copier. The stocks they sold did as well as the market or
  better.
- **By size, by owner.** No slice of either chamber shows a copier's edge: $1k-15k, $15k-50k, over $50k; self,
  spouse, joint. The largest House buys (over $50k, 480 trades) came closest: +0.90% over 63 sessions, 54% beat
  SPY, t +1.19.
- **The best traders are lucky ones.** The best House copier's record (68 members with 30+ trades, +5.3% per
  trade over 63 sessions) is below the best that luck gives (+5.8% median). The best senator's (+2.4%) is within
  luck too (+1.9% median, +3.7% at the 95th percentile). The one near miss: the best senator over 21 sessions
  (+2.2%, beaten by luck 6% of the time), one of five such checks.
- **Persistence is weak.** Members' ranks in the first half of each period barely predict the second half: House
  2019-2021 to 2022-2023, 28 members, rank correlation +0.19, top half +0.07% vs bottom half -0.48%. Senate
  2012-2016 to 2017-2020, 11 members, rank correlation -0.43 (63 sessions) and +0.38 (21 sessions).
- **Options.** The House's 512 option trades, read as bets on the underlying, were right 50% of the time for a
  copier over 63 sessions (+0.26%).
- **The best-known household.** Speaker Pelosi's reports, 90 trades with options, 2020-2023, scored -0.89% for a
  copier over 63 sessions. Their 53 bullish trades did well: +3.1% in the month after the trade, 70% right;
  copied after the report, +0.2% over 21 sessions and +2.8% over 63 sessions. Most were calls on the largest tech
  companies in years when those led the market, so the record cannot tell skill from that tilt.

This agrees with the studies of the same years. [Belmont, Sacerdote, Sehgal and Van Hoek
(2022)](https://ideas.repec.org/a/eee/pubeco/v207y2022ics0047272722000044.html) find, for 2012-2020, that
House buys lagged the market by 0.26% over six months and that the results look like random stock picking. The
outperformance found in earlier years ([about 6% a year for the House before 2012](https://www.cato.org/sites/cato.org/files/2020-08/RB231.pdf))
did not survive the STOCK Act. One recent paper claims that members' trades beat their peers' after they become
party or committee leaders ([Wei and Zhou, NBER w34524](https://www.nber.org/papers/w34524)); that subgroup is
too small in this data to check.

## Activist-style stakes (Schedule 13D)

Excess vs IWM; the filing day is day 0.

| | All initial 13Ds (4,528) | Price $5+, filed by a fund-like name (1,682) | Well-known activist funds (138) |
|---|---|---|---|
| Run-up, 20 sessions before | +3.25% mean, -0.85% median | +2.84% | +1.82%, t 2.2 |
| Reaction, close before to close after day 1 | +1.43%, t 6.1 | +1.41%, t 5.3 | **+5.08%** (median +4.01%), 78% up, t 7.9 |
| Buy at the next morning's open, 21 sessions | +0.15% | +0.11% | +3.05% (median +1.29%), t 1.9 |
| From day 1's close, 63 sessions | -2.0% | -2.8% | +0.36% |
| From day 1's close, 126 sessions | -6.7%, t -3.7 | -7.1% | -1.2% |

- **The pop belongs to whoever was first.** The well-known activist funds' filings moved their targets about 5% in
  two days, in every year from 2015 to 2023 (yearly averages +2.4% to +8.8%). Most of it came on the filing day
  itself (+1.8%, open to close, before subtracting the index) and overnight after an evening filing (+1.9%). From
  the next morning's open, the rest of that day added +0.5%.
- **After that, little that holds up.** The next-morning buyer's +3.05% over a month (t 1.9) rests on a few
  stocks: one rose 143%. Without the top five it is +1.0%. Over three and six months, nothing.
- **Most 13Ds are not activists.** All filings together: +1.4% on the filing, then nothing for a copier, then a
  slide against small companies over six months. Many filers are insiders, lenders or holders of distressed
  companies.
- The size of the pop agrees with [Brav, Jiang, Partnoy and Thomas
  (2008)](https://corpgov.law.harvard.edu/wp-content/uploads/2008/03/hedge_fund_activism-final.pdf): about 7%
  over days -20 to +20 for 2001-2006, with no reversal in the following year.

## What can bias it

- **Survivorship.** Companies no longer listed have no price history: 900 of 5,447 Senate trades, 1,826 of
  17,088 House trades and 5,791 of 11,809 13D filings (no listed party) dropped out. For 13D targets this removes
  many takeovers, often the activist's best outcome, so the long-run 13D numbers lean low.
- **Hindsight in the activist list.** The 27 fund names were picked by today's fame, which favours funds that did
  well. A declared test must fix the list before it looks at 2024 onward.
- **Ticker mapping.** Old tickers were mapped to current ones where the company is the same (Facebook to Meta,
  Anthem to Elevance...); trades whose company name does not match the ticker's current owner were dropped (298).
- **Ranges and paper.** Reports give amount ranges, not sizes; every trade counts once. Paper House reports
  (about 18%, scanned images) were not read.

## What it means for us

1. A politician's disclosed buy is not a reason to buy. On average it carried nothing, at the trade or after the
   report, and no member's record stands out from luck.
2. An activist's 13D is news, priced within hours. Copying it the next morning has not reliably paid. Only someone
   reading EDGAR as filings land could take part of the pop, and only after hours, where trading is thin.
3. The general lesson: once a footprint is public, either the price has already moved or there was nothing in it.
   The edge, where there was one, belonged to whoever moved first.

Candidate declared tests on the untouched 2024-2026 data, if wanted: the well-known activists' 13Ds bought at the
next open and held 21 sessions (the list fixed first); and the largest congressional buys (over $50k), copied
after the report and held 63 sessions. Neither looks strong enough here to expect a pass.
