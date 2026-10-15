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
4. `python -m brain brief TICKER`: a cited one-page summary. Quote numbers from the brief, not from memory.
5. `python -m brain decide TICKER`: a paper decision under policy D1 (`docs/DECIDE.md`), weighing only tested evidence and citing every input. Quote its action, score and each item's weight and why as the card prints them. `--no-llm` skips the model; `--read` lets it read the web first. `python -m brain decide score` shows how past decisions and every source's calls did.
6. For a person at a screen, `python -m brain terminal` serves the same bank as a keyboard-driven terminal (`docs/TERMINAL.md`). Its `ASK` analyst follows the same rules as you: bank-only facts, cited ids, verdicts quoted as recorded.

## Turning an idea into a tested claim

1. Write it as a signal in `brain/research/signals.py` (point-in-time: it may only use `ctx.adj` and `ctx.view`).
2. Add a hypothesis to `brain/research/hypotheses.py` with a new id. Never edit a declared one; make `H6b` instead.
3. `python -m brain research declare`, then commit the code and the ledger **before** running.
4. `python -m brain research run H6`, then commit the ledger and `research/RESULTS.md`, whatever the result.
5. A FAIL is recorded and stays. A mistake is withdrawn (`research withdraw H6 --reason "..."`), never deleted.

Every hypothesis ever run counts in the false-discovery correction, so trying twenty variants costs something. That is the point.

## Decisions

- The model's own call (a `judgment` fact, made by `brain decide`) is a source like any web page. It carries weight only through its scored record (`brain decide score`, terminal `CLM` and `DEC`), never because it reads well. The same holds for the claims you note.
- Policy D1 is never edited. A change to its weights, thresholds or gates is a new policy, D2, with its own sha; decisions already made stay scored under the policy that made them.
- Report NO_EDGE plainly: it means no tested evidence carries weight. The lean shown beside it is unproven and not evidence; never present it as a call.
- Decisions are paper. They are recorded and scored, never traded.

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
