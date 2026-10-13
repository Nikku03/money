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

Nine hypotheses have been declared and run (`research/RESULTS.md`). **None beat a shuffled signal on the 2019–2026 TEST months after costs.**

| Idea | Universe | TEST net excess / month | Perm p | Verdict |
| --- | --- | --- | --- | --- |
| H1 12-1 month momentum | 60 large caps | +0.11% | 0.18 | FAIL |
| H2 1-month reversal | 60 large caps | −0.01% | 0.21 | FAIL |
| H3 earnings yield (value) | 60 large caps | −0.23% | 0.87 | FAIL |
| H4 low asset growth | 60 large caps | −0.11% | 0.74 | FAIL |
| H5 ridge learner on H1–H4 | 60 large caps | +0.19% | 0.051 | FAIL (BH q 0.46) |
| H3b trailing-twelve-month earnings yield | 741 stocks | −0.11% | 0.88 | FAIL |
| H6 insider buying (officers and directors, 90 days) | 741 stocks | +0.00% | 0.28 | FAIL |
| H7 earnings-day reaction drift | 741 stocks | −0.24% | 0.60 | FAIL |
| H8 standardized unexpected earnings (SUE) | 741 stocks | −0.06% | 0.51 | FAIL |

The controls passed on both universes: the harness finds a planted edge (oracle p = 0.0005), and random signals pass 3–4% of the time, near the nominal 5%. So these are real negatives for these universes, not a broken instrument.

H6 is the instructive one. On the TRAIN months (2011–2018) insider buying earned +0.11% a month with p = 0.008, the kind of result that gets written up. On the months it had never seen it earned nothing. That is why the split exists. Every hypothesis ever run counts in the false-discovery correction, so H5's q rose from 0.26 to 0.46 when H3b–H8 were added; trying more variants costs something, as it should.

What this means for making money: textbook signals, including the well-known event ones (insider buying, post-earnings drift, SUE), do not carry an edge you can detect after costs in these universes. Both universes are chosen by today's market value, so they are survivors; the equal-weight broad universe returned 20% a year over TEST against SPY's 18%, mostly for that reason. Any edge is more likely to come from information the market prices slowly, and each new idea goes through the same gate. The alternative data below (aircraft, ships, cameras) is collected so that ideas about it can be tested once enough history exists; none has been tested yet.

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
| `ERN` | earnings by quarter: EPS, EPS YoY, SUE, 3-day reaction, 20- and 60-day drift, with the H7 and H8 verdicts |
| `JET` | the company's own registered aircraft and their recorded flights |
| `SHIP` `CAM` | vessel counts per port area, vehicle counts per camera, each with a z-score; `SHIP AREA` and `CAM ID` chart one |
| `WL` `ALRT` | the watchlist with last price, 1-day move and latest alert; notice and high alerts newest first, `ALRT ALL` with the info ones (`WL ADD T`, `ALRT ACK ID`) |
| `EV` | the brain's verdict: where a stock ranks on every tested signal (H1-H4 against the 60-stock universe, H3c, H6, H7b and H8b against the 741-stock one), and whether that signal passed its held-out test |
| `BRIEF` `SHOW` | the cited one-page brief; any fact with its source |
| `WEI` `MOV` `CURVE` `ECO` | world markets, universe movers, Treasury yield curve, macro |
| `RES` `CLM` | research verdicts per universe; web-claim track records per source |
| `ASK` | ask the analyst, an open-weight model on a server you run. It answers only from the bank, through read-only tools that run terminal commands, and cites fact ids. Its tools cover DES, GP, FA, CF, CN, EV, SHOW, RES, WEI, MOV, CURVE, ECO, CLM, INS, ERN, JET, SHIP, CAM, WL and ALRT (reading only). Under each answer the pane flags cited ids and numbers that none of its tool results hold |
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
| `BRAIN_LLM_URL`, `BRAIN_LLM_MODEL`, `BRAIN_LLM_KEY` | The terminal's `ASK` and `brain scout` | Defaults: `http://127.0.0.1:11434/v1` (Ollama) and `qwen3:8b`. Any OpenAI-compatible server with tool calling works; standard-library HTTP, no vendor SDK |
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
python -m brain score-claims                # hit rate per source, vs a coin flip
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
| `brain/claims.py` | Store web claims; score them per source |
| `brain/brief.py` | Cited one-page brief; writes derived facts back |
| `brain/research/` | Monthly panel, signals, harness (portfolio, costs, nulls per select mode), ledger, hypotheses, runner |
| `brain/terminal/` | The terminal: command grammar, data functions, local HTTP server and jobs, analyst, static frontend |
| `brain/markets.py` | The world-markets symbol list and its collector |
| `docs/TERMINAL.md`, `docs/EXPANSION.md` | Terminal spec; the v2 spec (insiders, earnings, universe, alternative data, alerts, open model) |
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
| Qwen3 (default `qwen3:8b`) or any open-weight model you choose | Analyst and scout | Qwen3: Apache-2.0. Runs on your machine; nothing leaves it unless `BRAIN_LLM_URL` points elsewhere |
| SearXNG (self-hosted) | Web search for the scout | AGPL-3.0 software; results link to their own sites, read under their robots.txt |
| ntfy | Optional alert push | Apache-2.0 / GPL-2.0 software; ntfy.sh or self-hosted |

## Known limits

- **Survivorship.** Both universes are chosen by today's market value, so every member survived and grew. Results report the size of the effect and treat any PASS as an upper bound. Fixing this needs delisted-company price data, which free sources don't provide.
- **Alternative data has no history yet.** Aircraft flights, vessel counts and vehicle counts start when you start collecting them. No hypothesis on them can be tested until there are years of samples, and a single camera frame is a noisy count.
- **Coverage.** H7 has a value for 35% of TEST stock-months (an earnings release in the 31 days before the rebalance); quarterly EPS is missing where a company tags it outside the curated XBRL concepts.
- The analyst checks its own answer, not its reasoning: the ASK pane flags cited ids its tools never returned and numbers none of its tool results hold, but a number it computed correctly from cited facts is flagged too, and one it copied from the wrong row is not.
- Headline alerts are many (every new headline is an `info` alert), so they are off for a new watchlist (`brain watch rule headlines on` turns them on); the terminal's alert count and `ALRT` leave info alerts out, and `ALRT ALL` lists them.
- US filings only. The price source handles other exchanges (e.g. `RELIANCE.NS`), but filings come from SEC. Cameras: London and California only.
- No broker connection, by design.
- The terminal is single-user and local. Prices are end-of-day snapshots, not a live feed.

Not financial advice.
