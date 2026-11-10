# Working rules for an agent using this repo as its brain

This repo is a memory bank and an evidence engine. You (an agent with web access) are the part that reads the open internet. Everything you learn goes into the bank with its source and timestamp. Nothing becomes a trading belief until the engine has tested it.

## Setup

```
export SEC_USER_AGENT="Your Name your@email.com"   # SEC fair-access policy; required
python -m unittest discover -s tests -t .          # must pass
python -m brain check                              # bank invariants; exit 0 = OK
```

## Looking into a target (a company or ticker)

1. `python -m brain ingest TICKER`: SEC XBRL financials, the filings list, daily prices and headlines.
2. `python -m brain markets` (world indices, FX, commodities, crypto) and `python -m brain macro` (the US Treasury curve; FRED series too when `FRED_API_KEY` is set).
3. Read the web for what the bank lacks: earnings call coverage, guidance, product news, regulatory actions, analyst views. For each finding:
   ```
   python -m brain note TICKER "what the page says, in one line" \
       --source URL --known-at <publication time, ISO> \
       --tier reported|opinion --direction up|down|none
   ```
   - `--source` is the page's http(s) URL; anything else is refused.
   - `reported` is a factual event ("FDA approved X"). `opinion` is a view or forecast ("analyst says buy").
   - `--known-at` is when the page was published, not when you read it. If you can't find a publication time, leave it out (it defaults to now).
   - `--direction` is what the claim implies for the stock relative to the market over about a month. Use `none` when it implies nothing.
   The engine can also read the web by itself, with no model: `python -m brain web plan` shows what it would fetch from the bank's gaps (the stories behind headlines, the sources in `knowledge/web_sources.json`), and `python -m brain web run` fetches it under the web rules below and keeps each `web.article` with its source and stated publication time (`docs/WEB.md`).
4. `python -m brain brief TICKER`: a cited one-page summary. Quote numbers from the brief, not from memory.
5. `python -m brain decide TICKER`: a paper decision under policy D1 (`docs/DECIDE.md`), weighing only tested evidence and citing every input. Quote its action, score and each item's weight and why as the card prints them. `--no-llm` skips the model; `--read` lets it read the web first. `python -m brain decide score` shows how past decisions and every source's calls did.
6. For a person at a screen, `python -m brain terminal` serves the same bank as a keyboard-driven terminal (`docs/TERMINAL.md`). Its `ASK` analyst follows the same rules as you: bank-only facts, cited ids, verdicts quoted as recorded.
7. `python -m brain forecast TICKER [--horizon 5|21] [--llm]`: a paper forecast (68% and 90% price bands, the chance to beat SPY, the drivers), recorded as a fact (`docs/FORECAST.md`). Quote the bands, the chance and the Record lines as the card prints them: the chance has no tested edge unless its declared test (F1b, F1c) says PASS. On the real bank it refuses until F1a-F1c have recorded results.
8. `python -m brain moves predict TICKER [--implied X]`: the likely size of its next earnings move, either way, and the chance of moving more than the options' priced move X% (M1, `docs/MOVES.md`; `brain earnings radar` shows the same beside each priced move). Quote the range and the chance as printed. M1a (PASS in `research/RESULTS.md`) tested the range against the stock's own history; the chance against the options' price showed no skill in its first 99 live calls (`research/m1_calls.jsonl`, one week of reports; `docs/MOVES.md`): it did no better than treating the options as fairly priced. A size is not a direction.
9. `python -m brain shocks radar [TICKER ...]`: each stock's chance of a 20%+ session against SPY, either way, in the next five (S1, `docs/SHOCKS.md`). S1a (PASS in `research/RESULTS.md`) tested it on 2024-2026: each week's top 5% held half the shocks, and its chances ran about a fifth low. A chance of a shock is a size, not a direction. The fade after a jump (S2a, S2b) FAILED: do not present it as an edge. Its live calls are in `research/s1_calls.jsonl`. A broken price history (an unadjusted split, a reused ticker, no recent trades) can top the list, so check a name's prices before quoting its chance.
10. `python -m brain scan run [TICKER ...] [--refresh] [--record]`: the market scan (`docs/SCAN.md`). For every tradable stock of the research universe it prints the range its price change should stay in over the next five sessions (W1), S1's chance of a 20%+ day, and why the range is wide. W1a (PASS in `research/RESULTS.md`) tested the ranges on 2024-2026: the 2-in-3 range held 68.1% and the 9-in-10 range 89.7%; with a report due they ran a little narrow (64.7% and 87.1%). Quote a line as printed. A range is a size: never present its middle, or how often such stocks rose, as a direction. Stocks with thin or broken prices are left out with the reason. `--record` writes the scan to `research/scans/` before its sessions trade; `python -m brain scan score` judges recorded scans.

## Turning an idea into a tested claim

1. Write it as a signal in `brain/research/signals.py` (point-in-time: it may only use `ctx.adj` and `ctx.view`).
2. Add a hypothesis to `brain/research/hypotheses.py` with a new id. Never edit a declared one; make `H6b` instead.
3. `python -m brain research declare`, then commit the code and the ledger **before** running.
4. `python -m brain research run H6`, then commit the ledger and `research/RESULTS.md`, whatever the result.
5. A FAIL is recorded and stays. A mistake is withdrawn (`research withdraw H6 --reason "..."`), never deleted.

Every hypothesis ever run counts in the false-discovery correction, so trying twenty variants costs something. That is the point.

The chain engine's cause → effect links (`knowledge/links.json`) are tested the same way (`docs/LINKS.md`): `python -m brain chain measure`, `python -m brain chain declare`, commit, `python -m brain chain test`, commit the ledger and `research/RESULTS.md`. A new link or basket is a new test id; a declared one is never edited.

Corporate events are tested the same way (`docs/EVENTS.md`): `python -m brain releases` reads each earnings release's opening from SEC EDGAR, fixed rules tag its events, then `python -m brain events declare`, commit, `python -m brain events test`, commit. Small and micro caps: the lists are in `universe/` (a rebuild is a new experiment); `python -m brain releases --band small|micro` (hours; resumes if re-run), then `python -m brain events declare --band small|micro`.

## Decisions

- The model's own call (a `judgment` fact, made by `brain decide`) is a source like any web page. It carries weight only through its scored record (`brain decide score`, terminal `CLM` and `DEC`), never because it reads well. The same holds for the claims you note.
- Policy D1 is never edited. A change to its weights, thresholds or gates is a new policy, D2, with its own sha; decisions already made stay scored under the policy that made them.
- Report NO_EDGE plainly: it means no tested evidence carries weight. The lean shown beside it is unproven and not evidence; never present it as a call.
- Decisions are paper. They are recorded and scored, never traded.
- Any call you make to the person (a direction, a size, a trade idea) is recorded with `python -m brain predict add` before its outcome is known, with the chance you give it (`docs/PREDICT.md`). Quote `python -m brain predict score` before claiming any record, good or bad; under 30 judged calls it says nothing yet.
- Check the world before a market call: `python -m brain radar --refresh` flags markets and country funds that moved hard.
- Before talking about the week's reports, run `python -m brain earnings radar` (likely reporters, usual moves, what options price) and, the day after, `python -m brain earnings misses` to see what it missed (`docs/PREDICT.md`).

## Hard rules

- Collect public information about companies, markets and the economy only. Executives appear only in their public role: filings, Form 4s, earnings calls. Do not compile information on private individuals.
- Respect sources: identify yourself to SEC, stay under 10 requests/s, and don't get around paywalls, logins, robots.txt or bot checks.
- No material non-public information. If something looks leaked or confidential, do not store it.
- This repo places no orders. Do not add code that connects to a broker or trades real money.
- Claims are not evidence. Run `python -m brain decide score` before trusting any source's direction calls. Its track records are the only thing that gives a source weight. Calls in one 21-session window count as one bet, and a source needs 30 bets, a one-sided test and a false-discovery correction across sources (terminal `CLM`, 'the record D1 weighs'). `python -m brain score-claims` counts every call, overlapping ones included, so it overstates the evidence. In the related cell project, curated literature signs were right about half the time; assume the same of market commentary until a source's record shows otherwise.
- Report negatives as plainly as positives. If every hypothesis fails, say so.

## Alternative data: where the legal and ethical lines are

These sources watch the physical world. They stay useful and legal only inside these lines; code and agents must not cross them.

- **Aircraft (ADS-B).** Track only aircraft whose FAA registered owner is the company itself, a corporate asset. Never track aircraft registered to individuals. Never try to see through trusts, LLCs or privacy programs (FAA PIA/LADD) to find who owns or rides in an aircraft. Do not infer who was on board. Records are airport to airport, with times. Tails in `var/aviation_optout.txt` are never tracked.
- **Ships (AIS).** Count commercial vessels (cargo, tanker) in watched port areas. Pleasure and sailing craft are dropped. Store aggregate counts per area, not individual tracks.
- **Cameras.** Only official government traffic cameras published for public viewing under open terms (Transport for London JamCams, Caltrans). Never private, unsecured or login-protected cameras. Count vehicles only. Frames are processed in memory and discarded, never stored. No detection or identification of people, faces or number plates.
- **Web reading.** Respect robots.txt, identify the client, rate-limit, and never get around paywalls, logins or bot checks.
- Credit the sources: adsb.lol (ODbL), TfL Open Data, Caltrans, aisstream.io, OurAirports, FAA.
