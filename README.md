# money: a memory bank and an evidence engine for public market data

An agent with web access reads the internet. This repo is what it remembers, and the test every trading idea has to pass before it is believed.

```
 web pages, news ──(agent `note`, or `scout`)─┐
 SEC EDGAR: filings, XBRL, Forms 4/5 ─────────┤
 Yahoo prices, world markets ─────────────────┤
 US Treasury curve, FRED macro ───────────────┼──► memory bank ──► MONEY TERMINAL (local, keyboard-driven; every number cited)
 FAA registry + adsb.lol (company aircraft) ──┤    (sourced,   ──► analyst: an open-weight model on your machine, reading the bank through tools
 aisstream.io AIS (vessels per port area) ────┤     timestamped)──► research engine: declared hypotheses, matched nulls
 TfL / Caltrans cameras (vehicle counts) ─────┘                 ──► watchlist rules -> alerts
                                                                ──► claim scoring: is this source ever right?
                                                                ──► paper decisions (policy D1): weigh only tested evidence
```

Design carried over from the cell project (`Nikku03/cell`, `HANDOFF_REASONING_ENGINE_AND_MEMORY_BANK.md`):

| Lesson there | What it became here |
| --- | --- |
| Reasoning machinery was not the bottleneck; knowledge was | Most of the code is about getting facts right: sources, timestamps, restatements, share bases |
| Facts individually addressable, looked up by content | Every fact is content-addressed (id = hash), stored once, retrieved by entity and attribute |
| Write each intermediate result down (the "snap") | Derived numbers are written back as facts that cite their inputs |
| Journal must replay exactly | Append-only JSONL per entity; `check` verifies ids and replay |
| Literature signs ~50% right in a given cell line | Web claims are stored as claims, then scored per source against what the stock did |
| Matched controls, positive control first, predeclared gates | Shuffled-signal null, oracle control, random-signal calibration, gates in code before running |
| A generic learner beat hand-written rules | H5 is a ridge learner over the hand signals' ranks, judged by the same gate |
| Negatives recorded, overstatements withdrawn not deleted | `research/ledger.jsonl` is append-only and committed |

One rule did **not** carry over. The cell project split TRAIN and TEST by hashing the key. Market data is a time series and neighbouring days are correlated, so a hash split would leak. Here the split is by date.

## The one idea that matters most: point-in-time

Each fact has two dates. `as_of` is what it describes (the quarter that ended 2024-06-30). `known_at` is when the public could first know it (the 10-Q filed 2024-07-31). `bank.view(t)` shows only facts with `known_at <= t`, and every signal and metric reads through a view. A backtest on 2015-03-31 therefore sees 2015's filings, including numbers that were later restated, and never sees 2016's. Most backtests that look too good fail on exactly this. The same holds for the new data: an insider trade is known when its Form 4 was filed, not when it was traded; an earnings reaction's 20-day drift is `None`, never partial, until those 20 sessions have closed.

## What the evidence says so far

Twelve hypotheses have been declared and run (`research/RESULTS.md`); three of those were withdrawn after review found code mistakes behind them and were re-run under new ids. **None beat a shuffled signal on the 2019–2026 TEST months after costs.**

| Idea | Universe | TEST net excess / month | Perm p | Verdict |
| --- | --- | --- | --- | --- |
| H1 12-1 month momentum | 60 large caps | +0.11% | 0.18 | FAIL |
| H2 1-month reversal | 60 large caps | −0.01% | 0.21 | FAIL |
| H3 earnings yield (value) | 60 large caps | −0.23% | 0.87 | FAIL |
| H4 low asset growth | 60 large caps | −0.11% | 0.74 | FAIL |
| H5 ridge learner on H1–H4 | 60 large caps | +0.19% | 0.051 | FAIL (BH q 0.61) |
| H3c trailing-twelve-month earnings yield | 741 stocks | −0.11% | 0.88 | FAIL |
| H6 insider buying (officers and directors, 90 days) | 741 stocks | +0.00% | 0.28 | FAIL |
| H7b earnings-day reaction drift | 741 stocks | −0.24% | 0.58 | FAIL |
| H8b standardized unexpected earnings (SUE) | 741 stocks | −0.05% | 0.49 | FAIL |
| H3b, H7, H8 (first versions) | 741 stocks | −0.11%, −0.24%, −0.06% | 0.88, 0.60, 0.51 | withdrawn |

H3b, H7 and H8 were withdrawn, not deleted. A derived fourth quarter mixed a fiscal year as first filed with nine months recast later (H3b, H8). Earnings releases that came within 20 days of a pre-announcement were dropped as duplicates (H7). The corrected versions were declared and committed before they ran, on the same price snapshots, and came out within a few hundredths of a percent of the first versions. The withdrawn runs still count in the false-discovery correction.

The controls passed on both universes: the harness finds a planted edge (oracle p = 0.0005), and random signals pass 3–4% of the time, near the nominal 5%. Random scores held fixed per stock pass 5.5% of the time on the broad universe. That is the case for slow signals like earnings yield and SUE, and it too is near nominal. On the 60-stock universe that check was not run (a smaller test found it too lenient there), so read the large-cap p-values as optimistic. These are real negatives for these universes, not a broken instrument.

H6 is the instructive one. On the TRAIN months (2011–2018) insider buying earned +0.11% a month with p = 0.008, the kind of result that gets written up. On the months it had never seen it earned nothing. That is why the split exists. Every hypothesis ever run counts in the false-discovery correction, so H5's q rose from 0.26 to 0.61 as H3b–H8b were added, withdrawn ones included; trying more variants costs something, as it should.

What this means for making money: textbook signals, including the well-known event ones (insider buying, post-earnings drift, SUE), do not carry an edge you can detect after costs in these universes. Both universes are chosen by today's market value, so they are survivors; the equal-weight broad universe returned 20% a year over TEST against SPY's 18%, mostly for that reason. Any edge is more likely to come from information the market prices slowly, and each new idea goes through the same gate. The alternative data below (aircraft, ships, cameras) is collected so that ideas about it can be tested once enough history exists; none has been tested yet.

## Decisions (paper only)

```
python -m brain decide AAPL                 # weigh the tested evidence on AAPL under policy D1; record a paper decision
python -m brain decide AAPL --no-llm        # deterministic only: no web reading, no model call
python -m brain decide score                # how past decisions did against SPY, and every source's track record
```

`brain decide` turns what the bank holds into one paper call per stock: BUY (expect it to beat SPY over the next 21 trading days), SELL (expect it to trail SPY), NO_EDGE (nothing tested points either way) or NO_DATA (the bank lacks recent prices; nothing is recorded). When a model is reachable it can first read public pages about the company (`--read`) and make its own call, which is stored as a `judgment` fact citing the facts it used. A fixed, declared policy (D1, [`docs/DECIDE.md`](docs/DECIDE.md)) then gives each piece of evidence a weight from its tested record, and the decision goes into the bank with every input cited. Twenty-one trading days later it is scored against what the stock did.

**Today it says NO_EDGE for every stock.** That is the honest answer: every research hypothesis above failed its TEST gate, and no source (web site or model) has a scored record yet, so nothing carries weight. The card shows the sources' unweighted lean beside the decision, labelled as not evidence. The answer changes on its own when a hypothesis passes, or when a source earns weight.

**How a source earns weight.** Every directional call, a web claim or the model's own judgment, is scored against SPY from the first session after the bank held it. Calls made in one 21-day window count as one bet, since they rise and fall together. A source needs 30 such bets, a one-sided p < 0.05 against a coin, a false-discovery correction across every source (Benjamini-Hochberg q < 0.10) and a positive mean excess. Its weight is then max(0, 2L − 1), where L is the 95% lower bound of its hit rate. At about one bet a month that takes two and a half years, and a source needs roughly 27 hits in 30 bets to move a decision on its own. The model is a source like any web site: its calls weigh nothing until its own record clears that gate, and a new model or prompt starts a new record.

D1 is never edited: a change is D2, with its own hash, and scores are reported per policy. Decisions are paper only. Nothing here places orders or connects to a broker, and none of it is financial advice.

## Forecasts (paper only)

```
python -m brain forecast AAPL               # a weather-style forecast for the next 5 sessions, recorded as a fact
python -m brain forecast AAPL --horizon 21 --llm   # 21 sessions; also ask the local model for its own chance
python -m brain forecast score              # how recorded forecasts did once their window closed
```

`brain forecast` is a weather report for one stock after SPY's latest close: the likely range of its price at the end of the next 5 (or 21) sessions, as a 68% band and a 90% band; the chance it beats SPY over that time, against a coin's 50%; the main drivers of that chance; and the record of such forecasts. The forecaster, F1, is a small model (a logistic regression and gradient-boosted trees, numpy only) that learns from 2011 on how about forty price, behavioural, fundamental and market-regime variables relate to what stocks did next. Every input is read as the bank stood after the close of the forecast day, and the range comes from each stock's own recent volatility. With `--llm` the local model reads a point-in-time snapshot of the stock, F1's numbers and a primer on how markets and people behave ([`docs/MARKET_PRIMER.md`](docs/MARKET_PRIMER.md)), and gives its own chance, recorded as a second fact.

**How it is tested.** Four tests are declared in the research ledger before they run, with the code, the primer and the model pinned ([`docs/FORECAST.md`](docs/FORECAST.md)). F1b (weekly, 2019 to two months ago) and F1c (21 sessions, 2019 to now) are the verdicts: their chance to beat SPY must rank stocks by what they then did beyond what size alone explains, with a date-level p < 0.05, a false-discovery correction shared with every research hypothesis, and two controls. F1a (the last two months) and F1d (the local model against F1 on 12 stocks over those weeks) are descriptive retests, shown with their uncertainty and no verdict. The ranges are scored by how often the price ended inside the band, per date. The expectation, written down before any run: the range should be about right, and the chance close to a coin.

**Where the record comes from.** The card's Record lines quote the declared runs' results from `research/ledger.jsonl` as they stand; until a test has run, the card says the forecast is untested. Development used only synthetic banks and real data from 2011-2018, so the held-out years are computed once, by the recorded runs. On the real bank, or any copy of it, `brain forecast TICKER` refuses until F1a, F1b and F1c have results in the committed ledger, since a live model would otherwise train on those years first; `--llm` also waits for F1d's result or its withdrawal. Forecasts are paper only and not financial advice.

## The chain engine

`python -m brain chain follow oil_price_up` follows cause→effect links (`knowledge/links.json`) step by step with the bump-node dendritic network and its snap, copied from Nikku03/cell (`brain/chain/engine`). It trains in about 20 seconds and was exact on fresh tables up to 16 steps. Each chain is shown beside an exact lookup. The links are textbook knowledge, so a chain says something about a stock only once its links have been measured on the bank's history.

`python -m brain chain measure`, then `declare`, commit, `test`, commit, measures each chain the bank can see end to end (`docs/LINKS.md`):
- It covers oil → airline and oil producer stocks, and the 10-year yield → bank, tech and homebuilder stocks.
- Each pair is tested twice: does the basket move with the cause that month, and does it move the next month? The next-month test is the one that matters for acting on a link.
- `brain chain links` shows each link's test, or what would measure it. `brain chain follow` shows the results beside each chain.

## Event tests

`python -m brain releases` reads each earnings press release's opening from SEC EDGAR. Fixed rules tag the events it names, such as guidance raised or guidance cut. `python -m brain events declare`, commit, `python -m brain events test`, commit, then tests whether those events move the stock (`docs/EVENTS.md`):
- **Reaction:** the 3 days around the release.
- **Drift:** the following month.

Each is measured against the same month's other earnings releases.

## The web gateway

`python -m brain web run` lets the reasoning engine read the web without a language model (`docs/WEB.md`):
- **The engine chooses:** it picks what to fetch from the bank's gaps, such as the story behind each headline and company newsroom feeds.
- **The browser fetches:** a real browser, browser-use with no agent, renders the pages, under the repo's web rules. It identifies itself, checks robots.txt, reads public hosts only, and stops at logins, paywalls and bot checks.
- **Fixed rules extract:** the headline, the stated publication time, author, publisher and text, plus tags for corporate events. Each is stored with its source.
- **Without a browser:** `--http` reads over plain HTTP.

## What the brain collects

| Area | What | Command |
| --- | --- | --- |
| Filings and fundamentals | SEC XBRL facts, every filing back to the first EDGAR one, identity | `brain ingest T` |
| Quarterly earnings | Discrete quarters (Q4 derived as the fiscal year minus nine months when it is not reported), trailing twelve months, SUE, EPS YoY | terminal `T ERN` |
| Earnings reactions | Earnings 8-Ks (item 2.02); 3-day excess return around the release, then 20- and 60-day drift against SPY | terminal `T ERN` |
| Insider trades | Open-market purchases (P) and sales (S) from SEC Forms 4 and 5: quarterly bulk data sets from 2010q4, plus each new Form 4 | `brain insiders T`, terminal `T INS` |
| More stocks | `universe/us_broad.json`: 741 stocks from SEC's first 1,500 filers by the same written rule as the 60-stock universe | `brain universe --broad`, `brain ingest-universe --broad` |
| Corporate aircraft | Aircraft the FAA lists as registered to the company itself, and their airport-to-airport flights | `brain jets link T`, `brain jets poll`, terminal `T JET` |
| Shipping | Commercial vessels (cargo and tanker) counted in port areas, as aggregate counts; other and untyped vessels are tallied beside the count | `brain ships sample AREA`, terminal `SHIP` |
| Traffic cameras | Vehicles counted in official public traffic-camera frames near a place (a company's plant, office or depot) | `brain cams near LAT LON`, `brain cams sample ID`, terminal `CAM` |
| Watchlist and alerts | Tickers, company aircraft, port areas and camera sites; rules for price moves, filings, insider buys, earnings reactions, landings, unusual counts, headlines | `brain watch ...`, `brain alerts ...`, terminal `WL`, `ALRT` |
| Analyst and scout | An open-weight model (Ollama, llama.cpp, vLLM, LM Studio) answers from the bank; the scout reads public web pages and notes claims with their sources | terminal `ASK`, `brain scout T` |

## The legal and ethical lines

They are in [`CLAUDE.md`](CLAUDE.md) ("Hard rules" and "Alternative data") and they are requirements. The code enforces them rather than asking nicely:

- **Public information only.** Executives appear only in their public role, as their filings state it. Insider records keep name, SEC CIK, relationship and title; addresses, signatures and remarks are never read into the bank. Nothing on private individuals.
- **Aircraft.** Only aircraft whose FAA registrant is a corporation or LLC whose name equals the company's SEC name word for word, legal form included (WEST LAKE HOLDINGS INC is not WESTLAKE CORP; an LLC registrant matches only a company that is itself an LLC). Individuals, trusts, other LLCs and co-owners never match; there is no fuzzy matching. Aircraft in the FAA's LADD and PIA privacy programs are dropped the moment they are seen and never polled again. Tails or hexes in `var/aviation_optout.txt` (`brain jets optout N123AB`) are never linked, polled or shown, and what was recorded about them before (registration, flights, the alerts citing them) is withdrawn. Stored: airport to airport, with times. No tracks, no callsigns, no guess at who was on board.
- **Ships.** Counts per area by type and status; the count is of commercial vessels (cargo, tanker). Pleasure and sailing craft are dropped, a vessel whose type was not heard is never in the count, and no vessel's identity or track is stored.
- **Cameras.** Only official government traffic cameras published for public viewing under open terms: Transport for London JamCams and Caltrans. A camera is sampled only by its id in the provider's own listing, from the provider's own image host; there is no way to point the sampler at another URL, a private camera or a login-protected one. Frames are counted in memory and dropped, never written to disk. The detector reports only the four vehicle classes and never reads the person class (the other classes are read only to veto a box that is a train or a boat, not a bus); people, faces and number plates are not detected at all.
- **Web reading.** The scout obeys robots.txt, names itself, waits a second between requests to a host, refuses private and local addresses, and reports a paywall, login or bot check as blocked and moves on. It can only note a claim from a page it actually read in that run.
- **SEC.** Every request carries `SEC_USER_AGENT`, and all SEC requests from every process using the same `var/` take turns through one lock, so the machine stays under 10 per second even with two terminal jobs running.
- **No orders.** Nothing connects to a broker.

## The terminal

```
python -m brain terminal                    # opens http://127.0.0.1:8765
```

A keyboard-driven market terminal over the memory bank: type `AAPL GP 5Y SPY` and press Enter. Four panes, a command line with autocomplete and history, world clocks, a ticker tape, and an alert count in the top bar. Full spec and user guide: [`docs/TERMINAL.md`](docs/TERMINAL.md).

| Function | What |
| --- | --- |
| `DES` `GP` `FA` `CF` `CN` | description, price chart (with comparisons), 10 years of financials, SEC filings, news and web claims |
| `INS` | insider trades, newest filing first, with 90-day buy and sell totals and the H6 verdict |
| `ERN` | earnings by quarter: EPS, EPS YoY, SUE, 3-day reaction, 20- and 60-day drift, with the H7b and H8b verdicts |
| `JET` | the company's own registered aircraft and their recorded flights |
| `SHIP` `CAM` | vessel counts per port area, vehicle counts per camera, each with a z-score; `SHIP AREA` and `CAM ID` chart one |
| `WL` `ALRT` | the watchlist with last price, 1-day move and latest alert; notice and high alerts newest first, `ALRT ALL` with the info ones (`WL ADD T`, `ALRT ACK ID`) |
| `EV` | the brain's verdict: where a stock ranks on every tested signal (H1-H4 against the 60-stock universe, H3c, H6, H7b and H8b against the 741-stock one), and whether that signal passed its held-out test |
| `BRIEF` `SHOW` | the cited one-page brief; any fact with its source |
| `WEI` `MOV` `CURVE` `ECO` | world markets, universe movers, Treasury yield curve, macro |
| `RES` `CLM` | research verdicts per universe; per source, the track record decisions weigh (independent bets) and the count of every call |
| `DEC` | paper decisions under policy D1: `T DEC` the latest with its evidence weights, the unproven lean, the explanation and the history; `DEC` every stock's latest, the scorecard and every source's track record; `T DEC NEW` decides afresh in a background job |
| `FCST` | paper forecasts: `T FCST` the latest bands drawn on a 3-month price chart, the chance to beat SPY against a coin, the drivers, the declared tests' lines and live scoring; `FCST` every stock's latest; `T FCST NEW` forecasts afresh in a background job |
| `ASK` | ask the analyst, an open-weight model on a server you run. It answers only from the bank, through read-only tools that run terminal commands, and cites fact ids. Its tools cover DES, GP, FA, CF, CN, EV, SHOW, RES, WEI, MOV, CURVE, ECO, CLM, INS, ERN, JET, SHIP, CAM, WL, ALRT, DEC and FCST (reading only). Under each answer the pane flags cited ids and numbers that none of its tool results hold |
| `INGEST` `JOBS` | collect data in the background: a ticker, `MARKETS`, `MACRO`, `UNIVERSE`, `INSIDERS`, `WATCH` |

It listens on 127.0.0.1 only and requires a per-session token on every API call; a Host check blocks DNS rebinding, and the page's CSP allows no external requests.

## Setup

Required: Python 3.11+, numpy and pandas, and a contact for SEC.

```
pip install -e .                            # numpy, pandas
export SEC_USER_AGENT="Your Name your@email.com"
python -m unittest discover -s tests -t .   # no network; must pass
```

Optional, each enabling one part. Nothing else needs a key.

| Setting | Enables | Notes |
| --- | --- | --- |
| `FRED_API_KEY` | FRED macro series in `macro`, `ECO` and briefs | Free key from the St. Louis Fed |
| `AISSTREAM_API_KEY` | `ships sample`, watched port areas | Free key from aisstream.io |
| `TFL_APP_KEY` | TfL camera listing with a higher request allowance | Works without it |
| `pip install opencv-python-headless websockets` (or `pip install -e ".[alt]"`) | Vehicle counting (OpenCV DNN) and the AIS WebSocket | The YOLOX model (35.9 MB) downloads once to `var/models/` from Hugging Face and is checked against a pinned sha256 |
| `BRAIN_LLM_URL`, `BRAIN_LLM_MODEL`, `BRAIN_LLM_KEY` | The terminal's `ASK`, `brain scout`, and `brain decide`'s reading, judgment and explanation | Defaults: `http://127.0.0.1:11434/v1` (Ollama) and `qwen3:8b`. Any OpenAI-compatible server with tool calling works; standard-library HTTP, no vendor SDK |
| `BRAIN_LLM_CTX`, `BRAIN_LLM_TIMEOUT` | Fit the analyst to the server | Context window in tokens (default 16384), seconds per request (default 300) |
| `BRAIN_SEARX_URL` | `web_search` for the scout | A SearXNG instance with the JSON format enabled. Without it the scout cannot search: it reads pages the bank links to (headlines, filings) and URLs you put in the question, e.g. `brain scout AAPL "Read https://... and note the reported results"` |
| `BRAIN_NTFY_URL` | Push alerts to a phone | e.g. `https://ntfy.sh/<your-topic>`; notice and high alerts only |
| `MONEY_HOME` | Where local state lives | Default `var/` (gitignored) |

The analyst and scout run on a free, open-weight model on your own machine. With [Ollama](https://ollama.com):

```
OLLAMA_CONTEXT_LENGTH=16384 ollama serve    # Ollama's default context is smaller and it cuts silently
ollama pull qwen3:8b                        # Apache-2.0; about 5 GB, runs on CPU (slowly) or a GPU
python -m brain terminal                    # the top bar shows ANALYST ON when the model answers
```

Or point `BRAIN_LLM_URL` at llama.cpp's server, vLLM or LM Studio (the URL ends in `/v1`) and set `BRAIN_LLM_MODEL` to a model it lists. Smaller models answer faster and make more mistakes; check every number against the fact it cites (`SHOW`).

For the scout's web search, run SearXNG locally, for example `docker run -d -p 8888:8080 searxng/searxng`, add `json` to `search: formats` in its `settings.yml`, and `export BRAIN_SEARX_URL=http://127.0.0.1:8888`.

## Quick start

```
python -m brain ingest AAPL                 # SEC facts, filings, prices, headlines
python -m brain insiders AAPL               # insider trades: bulk history (first run fetches ~700 MB of SEC data sets, cached), then recent Form 4s
python -m brain markets                     # world indices, FX, commodities, crypto (32 symbols)
python -m brain macro                       # US Treasury curve; FRED series too if FRED_API_KEY is set
python -m brain brief AAPL                  # cited brief -> var/briefs/AAPL.md (a month's first ranks all 741 broad stocks: about a minute)
python -m brain ask AAPL us-gaap:NetIncomeLoss --at 2015-06-30   # what was known then
python -m brain note AAPL "Analyst X cuts to sell on China demand" \
    --source https://... --known-at 2026-10-09T13:00:00Z --direction down
python -m brain scout AAPL                  # the model reads public pages and notes claims with sources
python -m brain score-claims                # every call per source vs a coin flip, overlapping ones included (decide score shows the record that gives weight)
python -m brain decide AAPL                 # a paper decision under policy D1 (today: NO_EDGE; see Decisions)
python -m brain forecast AAPL               # a paper forecast: price bands and the chance to beat SPY (see Forecasts)
python -m brain check                       # invariants; exit 0 = OK (about 2.5 minutes on a 1.9 GB bank)
```

Watchlist and alerts:

```
python -m brain watch add AAPL NVDA         # tickers (the kind word "ticker" is optional)
python -m brain jets link COST              # find aircraft registered to Costco itself
python -m brain watch add aircraft COST     # record their landings, airport to airport
python -m brain watch add area LA-LONG-BEACH    # or any box: watch add area NAME --bbox LAT0 LON0 LAT1 LON1
python -m brain cams near 51.4816 -0.1446 --km 1    # official cameras near a place, nearest first
python -m brain watch add site "Apple Battersea" --ticker AAPL --near 51.4816 -0.1446 --max 2
python -m brain watch rule headlines off    # headlines are info alerts and can be many; thresholds: watch rule NAME VALUE
python -m brain watch run --every 15m       # refresh + check forever, one line per alert; Ctrl+C stops
python -m brain alerts                      # list (* = not acknowledged); alerts check | ack ID | ack-all
```

An alert says where to look. It is not evidence that a price will move.

Research:

```
python -m brain universe --broad            # rebuild universe/us_broad.json by its rule (about 40 minutes; frozen in git)
python -m brain ingest-universe --broad     # refresh its data
python -m brain insiders                    # bulk insider data sets for every universe ticker, from 2010q4 (H6's first rebalance looks back 90 days)
python -m brain research declare            # declare hypotheses in brain/research/hypotheses.py, then commit
python -m brain research run                # run, append results to the ledger, write research/RESULTS.md
```

Local state lives in `var/` (gitignored; set `MONEY_HOME` to move it). With the broad universe and insider history it is about 2 GB.

## What was tested here, and what was not

Everything runs offline in the unit tests (fixtures, a fake OpenAI-compatible server, synthetic Form 4s, AIS messages and detector outputs). Against the real sources, on 2026-10-13:

- **Worked:** SEC filings, XBRL and Form 4s (live and the bulk data sets, 423,076 trades); the 741-stock universe build; earnings tables and reactions (AAPL, NVDA, JPM, HRI spot-checked against reported figures); FAA registry links (Costco 5 aircraft, Caterpillar 2, Apple none); adsb.lol position reads; TfL and Caltrans camera listings and vehicle counts; the watchlist loop, alerts and the terminal's new functions; `brain check` on the full bank (0 errors).
- **Only partly:** no company aircraft landed while it was being polled, so landing detection is tested on recorded position sequences, not on a live landing. adsb.lol answered one burst with HTTP 429; the poller now slows down and retries once. The analyst and scout were run against llama.cpp's server (llama-cpp-python) with Qwen3-1.7B on a CPU, not Ollama with qwen3:8b. Given a press-release URL, the scout read the page (robots.txt checked) and noted three claims with that URL and the page's publication date. The small model picked the right tools and reached the right verdict ("no hypothesis passed"), but it miscounted the hypotheses in one answer and misread a net income figure in another, and it took 3–15 minutes per question. Use the largest model your machine runs, and check every number with `SHOW`.
- **Decisions (2026-10-15, on a copy of the bank):** `brain decide AAPL MSFT NVDA --no-llm` recorded three NO_EDGE decisions: every hypothesis FAIL, no source with a record. The first run took about 2 minutes while the broad ranking was computed; later ones take a second or two per stock. SPY, ^GSPC and a mistyped ticker gave NO_DATA and recorded nothing, and `brain check` found 0 errors with the decisions and a judgment in the bank. With Qwen3-1.7B on llama.cpp's server (CPU), the judge read the bank through one round of tool calls, then called AAPL up (confidence 0.6), citing 5 facts that all resolved. Its reasons were weak (it compared Apple's revenue with "SPY's"), and that is why a call weighs nothing until its source has a record. A second judgment cited dates instead of fact ids and was not recorded. One explanation passed the checks though loosely worded; another failed them and the template was used. With `--read --max-pages 2` and no search engine, the model tried two Apple pages (one answered 404, the other 403, which was left alone) and noted nothing. A decision with the model took 12 to 20 minutes. In the terminal, `AAPL DEC NEW` ran as a job, `AAPL DEC` then showed the new decision, and a page reload started no job.
- **Not tested live:** aisstream.io (no key here; the WebSocket path is tested against recorded messages only), SearXNG search (no instance here), ntfy push.

## Layout

| Path | What |
| --- | --- |
| `brain/bank.py` | Fact, Bank (journal, replay, downgrade, lazy per-attribute reads, safe across processes), View (point-in-time reads) |
| `brain/check.py` | Invariants: id integrity, replay, sidecar indexes, nothing known before it happened, derived facts known after their inputs, series files intact |
| `brain/sources/` | SEC EDGAR (XBRL facts, every filing, identity), insider trades (`insiders.py`), Yahoo prices and splits, Treasury curve, FRED, headlines, aircraft (`aviation.py`), AIS (`shipping.py`), traffic cameras (`cameras.py`) |
| `brain/metrics.py` | Point-in-time metrics: market value, earnings yield, asset growth, margins, FCF, quarters, TTM, SUE |
| `brain/events.py` | Earnings events, reactions and the earnings table |
| `brain/universe.py` | The universe rules: 60 large caps and the 741-stock broad set |
| `brain/vision.py` | Vehicle counter: YOLOX on OpenCV DNN, frames in memory only |
| `brain/watch.py` | Watchlist, alert rules, refresh loop, ntfy push |
| `brain/llm.py` | Client for any OpenAI-compatible chat server (standard library only) |
| `brain/scout.py` | The model reading the web: search, robots-respecting page reader, claims with sources |
| `brain/net.py` | Polite HTTP: user agent, per-host pacing, SEC turns shared across processes, cache |
| `brain/claims.py` | Store web claims; score them per source; the track record a decision weighs |
| `brain/decide.py` | Policy D1: weigh the evidence, record a paper decision, score decisions later |
| `brain/judge.py` | The model's steps in a decision: its own call (a judgment fact) and a checked explanation |
| `brain/forecast/` | Forecasts: point-in-time features and the label (`features.py`), the learner and the range (`learn.py`, `gbm.py`), walk-forward tests and their metrics (`evaluate.py`), the local model's snapshot and probability (`think.py`), the card, the fact and live scoring (`live.py`) |
| `brain/brief.py` | Cited one-page brief; writes derived facts back |
| `brain/research/` | Monthly panel, signals, harness (portfolio, costs, nulls per select mode), ledger, hypotheses, runner; the declared forecast tests (`forecast_tests.py`) |
| `brain/terminal/` | The terminal: command grammar, data functions, local HTTP server and jobs, analyst, static frontend |
| `brain/markets.py` | The world-markets symbol list and its collector |
| `docs/TERMINAL.md`, `docs/EXPANSION.md`, `docs/DECIDE.md`, `docs/FORECAST.md`, `docs/MARKET_PRIMER.md` | Terminal spec; the v2 spec (insiders, earnings, universe, alternative data, alerts, open model); the decision contract (policy D1, judgments, scoring); the forecast contract (features, learner, declared tests F1a-F1d); the primer the local model reads |
| `research/ledger.jsonl` | Every declaration and result, append-only (committed) |
| `research/RESULTS.md` | Current verdicts, generated from the ledger (committed) |
| `universe/us_large.json`, `universe/us_broad.json` | The research universes, their rules, and what each rule rejected (committed) |

## Data and licences

| Source | Used for | Terms |
| --- | --- | --- |
| SEC EDGAR (submissions, XBRL, Form 4 XML) | Financial statements, filings, identity, live insider trades | US government, public domain. User-Agent with contact details required; at most 10 requests/s |
| SEC Insider Transactions Data Sets | Insider trade history (Forms 3/4/5, quarterly zips) | US government, public domain |
| Yahoo Finance chart API | Daily prices and splits for stocks, indices, FX, commodities, crypto | Unofficial, personal research only. Snapshots stay in `var/` and are never committed |
| Yahoo Finance RSS | Headlines | Links and titles only |
| US Treasury daily par yield curve | `CURVE`, `ECO`, macro context | US government, public domain; no key |
| FRED (official API, free key) | Macro context | Mostly public domain; revised series are flagged and kept out of backtests |
| FAA Aircraft Registry (`ReleasableAircraft.zip`) | Which aircraft a company itself owns | US government, public domain |
| adsb.lol | Live aircraft positions, one aircraft per request | Open Database License (ODbL); credit adsb.lol |
| OurAirports | Nearest airport to a landing | Public domain |
| aisstream.io | AIS vessel messages in a box, summarized to counts | Free API key; credit aisstream.io |
| Transport for London JamCams | Camera listing and frames (London) | TfL Open Data: "Powered by TfL Open Data" |
| Caltrans CWWP2 CCTV | Camera listing and frames (California) | Public data, no charge, Caltrans Conditions of Use |
| YOLOX-S via the OpenCV model zoo (Hugging Face `opencv/object_detection_yolox`) | Vehicle counts | Apache-2.0 |
| Qwen3 (default `qwen3:8b`) or any open-weight model you choose | Analyst, scout and the decision judge | Qwen3: Apache-2.0. Runs on your machine; nothing leaves it unless `BRAIN_LLM_URL` points elsewhere |
| SearXNG (self-hosted) | Web search for the scout | AGPL-3.0 software; results link to their own sites, read under their robots.txt |
| ntfy | Optional alert push | Apache-2.0 / GPL-2.0 software; ntfy.sh or self-hosted |

## Known limits

- **Survivorship.** Both universes are chosen by today's market value, so every member survived and grew. Results report the size of the effect and treat any PASS as an upper bound. Fixing this needs delisted-company price data, which free sources don't provide.
- **Alternative data has no history yet.** Aircraft flights, vessel counts and vehicle counts start when you start collecting them. No hypothesis on them can be tested until there are years of samples, and a single camera frame is a noisy count.
- **Coverage.** H7b has a value for 36% of TEST stock-months (an earnings release in the 31 days before the rebalance); quarterly EPS is missing where a company tags it outside the curated XBRL concepts.
- The analyst checks its own answer, not its reasoning: the ASK pane flags cited ids its tools never returned and numbers none of its tool results hold, but a number it computed correctly from cited facts is flagged too, and one it copied from the wrong row is not.
- Headline alerts are many (every new headline is an `info` alert), so they are off for a new watchlist (`brain watch rule headlines on` turns them on); the terminal's alert count and `ALRT` leave info alerts out, and `ALRT ALL` lists them.
- US filings only. The price source handles other exchanges (e.g. `RELIANCE.NS`), but filings come from SEC. Cameras: London and California only.
- No broker connection, by design.
- The terminal is single-user and local. Prices are end-of-day snapshots, not a live feed.

Not financial advice.
