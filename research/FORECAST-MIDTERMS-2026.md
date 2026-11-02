# My own 2026 midterm forecast, written before any vote is counted

Written Mon Nov 2, 2026, about 20:50 UTC, the day before the election. Committed to git before polls close, so it can be
judged honestly. It is my model, built from public polls and ratings. Markets and other forecasters are shown only for
comparison. Exploratory: no declared test.

## Result

| Outcome | My chance | Polymarket (Nov 2, 20:07 UTC) | Silver Bulletin (late Oct, press reports) |
|---|---|---|---|
| Democrats win the House | **91%** | 95.5% | about 95% |
| Democrats win the Senate (51+ seats of their own) | **79%** | 77.5% | about 76% |
| ...counting independents who might side with them (Osborn NE, Bodnar MT) | 81% | | |
| Democrats win both | **78%** | | |
| Democratic House, Republican Senate | 13% | | |
| Republicans keep both | 8% | | |
| Republican House, Democratic Senate | 1% | | |

- **Senate seats:** the most likely outcome is **53 Democrats** (80% range 49-55).
- **House seats:** the most likely outcome is **235 Democrats** (80% range 218-253). A majority needs 218.

## How the model works

**Senate.**
- **Polls.** Every general-election poll on each race's Wikipedia page, dated Aug 1 to Nov 2 (16 races, 451 polls).
  - The average uses polls ending Oct 5 or later.
  - Only each pollster's latest poll counts; if it published versions, they are averaged, and likely voters are
    preferred.
  - Weights: recency (half-life 10 days); sample size (square root, capped); population (registered voters 0.8,
    adults 0.5); a party sponsor (D) or (R) 0.6.
- **Experts.** The 12 raters on the 2026 Senate elections page (Cook, DDHQ, Economist, FPO, Fox, IE, MS NOW, NBC, RCP,
  Sabato, Silver, Split Ticket). Their calls are turned into margins: Safe 15, Likely 8, Lean 4, Tilt 2, Tossup 0.
- **Blend.** The weight on polls is n / (n + 2), with n the number of pollsters. A race with 27 pollsters (Texas) is
  93% polls; a race with one poll is a third polls.
- **Uncertainty.**
  - A national polling error shared by every race: standard deviation 3.0 points.
  - A race error: standard deviation 4.0 + 3.0/sqrt(1 + n), so about 4.6-7 points.
  - Both are drawn from fat-tailed (t, 6 degrees of freedom) distributions; 200,000 simulated elections.
- **Seat arithmetic.**
  - Democrats hold 34 seats not up this year and 9 safe seats that are (CO, DE, IL, MA, NJ, NM, OR, RI, VA).
  - They need 8 more from the 15 races where a Democrat is the challenger or holds a competitive seat.
  - At 50-50 the Republican vice president breaks the tie, so 51 means control.

**House.**
- **Districts.** All 435, with their 2026 Cook PVI on the new maps, from the 2026 House elections page.
- **The middle seat.** The 218th most Democratic district is R+2, so on a uniform swing Democrats need to win the
  national vote by about 4 points.
- **Each district's margin.** The generic ballot (the average of five aggregators on Nov 1: Democrats +8.7) plus twice
  the PVI, plus 2.5 points for an incumbent running.
- **The 147 rated districts.** That margin is blended 50/50 with the ten raters' average call (Cook, IE, Sabato,
  Economist, Split Ticket, DDHQ, FPO, Fox, Silver, RCP).
- **Uncertainty.** A national error with standard deviation 3.5 points, and a district error of 5 (rated) or 6 (other)
  points; fat tails; 100,000 simulations.
- **Both chambers.** One national error shared between House and Senate.

## The Senate races

"Polls" is my weighted average of the D-side margin; "Move" is the average margin from Oct 12 on minus Sep 10 - Oct 11
(positive: toward the Democrat or independent).

| State | Matchup (D or I vs R) | Polls | Pollsters | Experts | My margin | Chance D/I wins | Move |
|---|---|---|---|---|---|---|---|
| North Carolina (open R) | Cooper vs Whatley | +10.5 | 11 | +8.5 | +10.2 | 96% | -0.1 |
| Georgia (D) | Ossoff vs Collins | +8.4 | 6 | +9.4 | +8.6 | 93% | +0.5 |
| New Hampshire (open D) | Pappas vs Sununu | +9.0 | 5 | +7.6 | +8.6 | 93% | +0.7 |
| Minnesota (open D) | Flanagan vs Tafoya | +6.0 | 9 | +6.0 | +6.0 | 87% | +1.8 |
| Michigan (open D) | El-Sayed vs Rogers | +5.5 | 15 | +3.8 | +5.3 | 85% | +1.8 |
| Maine (R) | Jackson vs Collins | +4.9 | 13 | +3.0 | +4.6 | 81% | +3.0 |
| Texas (open R) | Talarico vs Paxton | +3.2 | 27 | +2.5 | +3.2 | 74% | +0.3 |
| Kansas (R) | Hamilton vs Marshall | +3.5 | 7 | +0.7 | +2.9 | 71% | +2.3 |
| Ohio special (R) | Brown vs Husted | +2.6 | 18 | +1.8 | +2.6 | 69% | 0.0 |
| Alaska (R) | Peltola vs Sullivan | +1.5 | 6 | +1.3 | +1.5 | 61% | 0.0 |
| Iowa (open R) | Turek vs Hinson | 0.0 | 11 | -1.0 | -0.1 | 49% | -1.6 |
| South Carolina (appointed R) | Andrews vs Graham | -1.3 | 8 | -3.2 | -1.7 | 37% | +2.6 |
| Nebraska (R) | Osborn (I) vs Ricketts | -1.5 | 7 | -4.5 | -2.2 | 34% | +0.7 |
| Louisiana (open R) | Davis vs Letlow | 0.0 | 1 | -10.6 | -7.1 | 13% | |
| Mississippi (R) | Colom vs Hyde-Smith | -3.0 | 1 | -9.2 | -7.1 | 13% | |
| Florida special (R) | (no polls listed) | | 0 | -8.2 | -8.2 | 12% | |
| Montana (open R) | Bodnar (I) vs Alme | -7.5 | 4 | -11.5 | -8.8 | 7% | +1.1 |

**The path.**
- If Democrats hold Georgia, New Hampshire, Minnesota and Michigan, they need 4 flips.
- They lead in North Carolina, Maine, Texas, Kansas, Ohio and Alaska, so there is room to lose two and still win.
- Iowa is even.
- South Carolina and Nebraska are within the polls' usual miss.

## The campaign's movement

- **Markets.** Polymarket's Democratic Senate odds rose from 28% (Aug 2025) to 50.5% (Sep 1, 2026), 62.5% (Oct 1),
  70.5% (Oct 30) and 77.5% (Nov 2). House odds rose from 88.5% on Sep 1 to 95.5%.
- **Senate polls.** Of the 14 races with enough polls, 10 moved toward Democrats in October: Maine +3.0, South Carolina
  +2.6, Kansas +2.3, Michigan and Minnesota +1.8. Ohio, Alaska and North Carolina were flat, and Iowa moved 1.6 toward
  Republicans.
- **Generic ballot.** Democrats +8.7 (DDHQ +8.9, FiftyPlusOne +8.6, NBC +8.0, RCP +8.9, Silver Bulletin +8.8; Nov 1).
- **Early vote (University of Florida tracking, reported about Oct 30).**
  - In states with party registration, registered Democrats cast about 46% of early ballots and Republicans 30%.
  - Ohio: 41% Democrats against 28% Republicans, against 26% and 36% at the same point in 2024.
  - Florida is the exception, with Republicans ahead.
  - Early voters lean Democratic in most years, so this is a sign of enthusiasm, not a count.
- **Special elections, 2025-26.** Democrats ran 11-13 points ahead of the 2024 presidential margin on average
  (Downballot, Multistate). Brookings' broader count gives 4.5 points. Before the 2018 midterms the Downballot figure
  was 10.6.
- **Trump's net approval.** About -22 on Oct 8 in the Silver Bulletin average (as reported by a prediction-market page
  that resolves on it); not confirmed for November.

## What could make it wrong

| If polls are off by | Democratic Senate chance | Democratic House chance |
|---|---|---|
| 3 points too Democratic (as in 2020) | 45% | 69% |
| none (central) | 79% | 91% |
| 3 points too Republican (as in some 2018 and 2022 races) | 95% | 98% |

- **The Senate hinges on a systematic polling error.** A 2020-sized miss toward Republicans would make it a coin flip.
- **The House.** It survives most misses because the national lead (about 9) is twice what the map requires (about 4).
- **Independents.** Osborn (Nebraska) and Bodnar (Montana) have not said they would caucus with Democrats. A 50-50
  Senate plus an independent would be decided by that choice.
- **Thin races.** Louisiana, Mississippi and Florida rest mostly on the raters; a surprise there is not well measured.
- **Shared inputs.** The raters' and my poll average draw on the same polls, so the blend is less independent than it
  looks.

## Sources

- Wikipedia (article pages only, per robots.txt):
  - "2026 United States Senate elections" (ratings table);
  - each race's "2026 United States Senate election in ..." page (poll lists);
  - "2026 United States House of Representatives elections" (generic ballot aggregates, district PVIs);
  - "2026 United States House of Representatives election ratings".
- Polymarket: "Which party will win the Senate in 2026?" and "...the House in 2026?" (daily price history).
- Early vote: University of Florida Election Lab via Scripps News and Spectrum News (late October).
- Special elections: Brookings, Multistate, the Downballot via reporting.
- Silver Bulletin and Newsweek reports (late October).
