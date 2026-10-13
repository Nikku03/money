# Expansion v2: spec

Adds insider trades, earnings reactions, quarterly fundamentals, a broader universe, aircraft, shipping and traffic-camera data, a watchlist with alerts, and an open-source model in place of Claude for the analyst. This file is the contract between the modules. Shapes and names here are normative; if code and this file disagree, fix one of them in the same change.

The legal and ethical lines are in `CLAUDE.md` ("Alternative data"). They are requirements, not suggestions; each module below restates the ones it enforces.

The bank (`brain/bank.py`) reads lazily per attribute and is safe for several processes, so a large universe is fine. Every new fact goes through `Bank.add_many` and obeys the point-in-time rules: `as_of` is what the fact describes, `known_at` is when the public could first know it.

---

## 1. SEC depth

### 1.1 All filings, not only the latest ~1,000 (`brain/sources/sec.py`)
`filings_for` also fetches the older pages listed in the submissions JSON under `filings.files` (`https://data.sec.gov/submissions/<name>`), so 8-K history reaches back to the company's first EDGAR filings. Same fact shape as today (attr `filing`).

### 1.2 More XBRL concepts (`brain/sources/sec.py`)
Add to `CONCEPTS["us-gaap"]`: `ProfitLoss`, `NetIncomeLossAvailableToCommonStockholdersBasic`, `NetIncomeLossAvailableToCommonStockholdersDiluted`, `WeightedAverageNumberOfSharesOutstandingBasic`, `WeightedAverageNumberOfShareOutstandingBasicAndDiluted`, `PaymentsToAcquireProductiveAssets`, `PaymentsOfDividendsCommonStock`, `RevenueFromContractWithCustomerIncludingAssessedTax`, `SalesRevenueGoodsNet`, `CostOfGoodsAndServicesSold`. Existing functions that feed declared hypotheses (`metrics.earnings_yield`, `metrics.asset_growth`, `metrics.market_cap`) must keep their exact behaviour: new concepts are used only by new functions.

### 1.3 Quarterly and trailing-twelve-month (`brain/metrics.py`, new functions only)
```python
@dataclass
class Quarter:                 # one fiscal quarter of a flow concept
    start: str; end: str       # ISO dates
    value: float
    known_at: str              # the later of its inputs' known_at
    inputs: list[Fact]         # 1 fact (reported quarter) or 2 (Q4 = fiscal year minus 9-month YTD)
    derived: bool

def quarterly(view, entity, attrs) -> list[Quarter]            # discrete quarters, oldest first
def ttm(view, entity, attrs) -> M | None                       # sum of the last 4 contiguous quarters; None if stale (> 200 days) or gappy
def sue(view, entity) -> M | None                              # (EPS_q - EPS_{q-4}) / sd of that difference over the 8 prior quarters (>= 6 needed)
def earnings_yield_ttm(view, entity, book) -> M | None         # TTM net income (NetIncomeLoss, then ProfitLoss, then ...AvailableToCommonStockholdersBasic) / market value; same sanity bound as earnings_yield
```
Quarters: a reported quarter has duration 80–100 days. Q4 = annual (330–400 days, 10-K) minus the 9-month YTD (260–280 days) ending at Q3, known when both are known. Concepts merge by priority (as `metrics.annual` does).

### 1.4 Earnings events and reactions (`brain/events.py`)
```python
def earnings_events(view, ticker) -> list[Fact]     # attr "filing", form 8-K or 8-K/A, items contain "2.02"; oldest first
def reaction(book, view, ticker, event, bench="SPY") -> dict | None
    # {"date": event day (first close at or after known_at), "car_3d": excess return day -1 .. day +1 vs bench,
    #  "drift_20d": excess return day +2 .. day +21, "drift_60d": ..day +61 (None when not yet known at view.t),
    #  "event": event.ref}
def earnings_table(view, ticker, book) -> list[dict]
    # newest first: {"period_end", "filed", "eps", "eps_yoy", "sue", "car_3d", "drift_20d", "drift_60d", "cite"}
```
Point-in-time: a drift window that ends after `view.t` is `None`, never partial. A quarter's release is the last distinct earnings 8-K within 90 days after the quarter ended that was public by the day its EPS was filed (New York dates), else the first one after it: a pre-announcement gives way to the results that followed, and a guidance 8-K after the 10-Q is not taken.

### 1.5 Insider trades (`brain/sources/insiders.py`)
Source: SEC "Insider Transactions Data Sets" (Forms 3/4/5, quarterly zips, public domain), `https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{YYYY}q{Q}_form345.zip`, plus live Form 4 XML for the current quarter (`https://www.sec.gov/Archives/edgar/data/<cik>/<accn-no-dashes>/<primary xml>`).
```python
def parse_dataset(zip_bytes: bytes, cik_to_ticker: dict[int, str], fetched_at: str) -> list[Fact]
def ingest_bulk(bank, cik_to_ticker, start: str = "2010q4", log=print) -> dict   # quarters done, facts added; skips quarters already ingested (remember them in var/insiders_done.json)
def parse_form4(xml: bytes, ticker: str, accn: str, filed: str, url: str) -> list[Fact]
def ingest_live(bank, ticker, days: int = 120) -> int                            # Form 4s filed in the last `days` not yet in the bank
```
Fact: entity ticker, attr `insider.trade`, tier `filed`, `value` = shares (positive), `unit` "shares", `as_of` = transaction date, `known_at` = filing date (bulk: end of that day; live: acceptance time), source `{"name": "SEC Form 4", "url": <filing index URL>}`, context `{"code": "P"|"S", "price": float|None, "value_usd": float|None, "owner": str, "owner_cik": str, "relationship": "officer"|"director"|"ten_percent_owner"|"other" (comma-joined if several), "title": str|None, "accn": str, "ownership": "D"|"I"}`. Only open-market purchases (`P`) and sales (`S`) are kept. Insiders appear in their public role only, as the filings state it. The default `start`, 2010q4, holds the first day of H6's first look-back: the 2011-01-31 rebalance counts purchases filed in the 90 days before it.

### 1.6 Broader universe (`brain/universe.py`)
`build_broad(bank, max_rank=1500, log=print)` writes `universe/us_broad.json`: every CIK among the first 1,500 rows of SEC's ticker list (file order is roughly market value) that meets the existing rule (10-K XBRL with `NetIncomeLoss` and `Assets`, Yahoo prices since 2010-01-04). Same fields as `us_large.json`; `rule_short` "SEC ranks 1-1500 with 10-K XBRL and prices since 2010". `universe/us_large.json` stays frozen (the H1–H5 results use it). CLI: `python -m brain universe --broad`.

### 1.7 New hypotheses (`brain/research/`)
Declared before any run (CLAUDE.md). All on `universe/us_broad.json`, same split, costs, benchmark and gate as H1–H5:

| id | signal | select | claim |
| --- | --- | --- | --- |
| `H3b-earnings_yield_ttm` | `earnings_yield_ttm` | top third | TTM earnings yield with broader concepts beats the universe next month |
| `H6-insider_buying` | `insider_buying`: dollar value of open-market purchases by officers and directors in the 90 days to the rebalance day (by filing date) | `positive` | stocks with insider buying beat the universe next month |
| `H7-earnings_reaction` | `earnings_reaction`: `car_3d` of an earnings event known in the 31 days to the rebalance day | `top_frac_vs_all` | strong earnings-day reactions keep drifting the same way next month |
| `H8-sue` | `sue` from the latest quarter known at the rebalance day | top third | standardized unexpected earnings predicts next month's excess return |

`select` lives in the hypothesis dict only (adding keys to `COMMON` would change H1–H5's declared spec hashes, which `research run` would then refuse). Modes: `top_frac` (default, today's behaviour: top third of stocks with a value; the null shuffles among stocks with a value), `top_frac_vs_all` (same holdings; the null draws the same number of names from all listed stocks, so "has an event at all" is part of what is tested), `positive` (every listed stock with signal > 0, at least 3; the null draws the same number from all listed stocks). `research run` builds one panel per universe and must reproduce H1–H5 exactly (same panel digest) when re-run.

---

## 2. Alternative data

### 2.1 Aircraft (`brain/sources/aviation.py`)
- FAA registry: `https://registry.faa.gov/database/ReleasableAircraft.zip` (public domain; cache 7 days). `MASTER.txt`: N-number, owner `NAME`, `MODE S CODE HEX`, `MFR MDL CODE`, `YEAR MFR`, `TYPE REGISTRANT`. Model names from `ACFTREF.txt`.
- `company_aircraft(name: str) -> list[dict]`: aircraft whose registrant type is a corporation (or LLC) and whose normalized owner name equals the company's normalized SEC name word for word, legal form included (normalizing evens out case, punctuation, `&`, a leading THE, the SEC's `/DE/` tail and how the legal form is spelled: CORPORATION = CORP, COMPANY = CO, INCORPORATED = INC, LIMITED = LTD; no word is dropped or joined, and HOLDINGS is part of the name). An LLC registrant matches only a company whose SEC name is itself an LLC. Never registrants of type individual. No fuzzy expansion to subsidiaries, trusts or other LLCs. Links record the rule they were made under (`MATCH_RULE`); one made under an older rule is re-checked before it is polled.
- `link(bank, ticker) -> int`: writes facts attr `aircraft.registered`, entity ticker, tier `reported`, value = N-number, context `{"hex", "model", "year", "owner"}`, source FAA registry URL.
- Positions: adsb.lol open data (ODbL), `https://api.adsb.lol/v2/hex/<hex>` (one per request, ≤ 1 request/s). `poll(bank, state_path=var/aviation_state.json) -> int`: for each registered hex of the watched companies, read the current position; detect take-off (ground → airborne) and landing (airborne → ground, or last seen below 3,000 ft then gone for > 20 min); on a landing write attr `aircraft.flight`, entity ticker, tier `reported`, value = arrival airport ident, context `{"hex", "reg", "from", "to", "from_name", "to_name", "departed", "arrived"}`, `as_of` = arrival date, `known_at` = when the landing was observed, source adsb.lol. Nearest airport within 5 km from OurAirports (`https://davidmegginson.github.io/ourairports-data/airports.csv`, public domain; large, medium and small airports; cache 30 days).
- Opt-out: any registration or hex in `var/aviation_optout.txt` is skipped everywhere.

### 2.2 Shipping (`brain/sources/shipping.py`)
- aisstream.io WebSocket `wss://stream.aisstream.io/v0/stream`, free key in `AISSTREAM_API_KEY`; subscribe `{"APIKey": key, "BoundingBoxes": [[[lat_min, lon_min], [lat_max, lon_max]]], "FilterMessageTypes": ["PositionReport", "ShipStaticData"]}`. Optional dependency `websockets`.
- `AREAS`: built-in port areas `{"name", "bbox": [lat_min, lon_min, lat_max, lon_max]}`: LA-LONG-BEACH, OAKLAND, HOUSTON, NEW-YORK-NJ, ROTTERDAM, SINGAPORE, SHANGHAI, NINGBO, BUSAN, JEBEL-ALI, SUEZ, PANAMA, HORMUZ. Watchlist areas add to these.
- `summarize(messages: list[dict], area: dict, window_s: int) -> dict` (pure, tested on fixtures): unique MMSIs; type from ShipStaticData (`Type` 70–79 cargo, 80–89 tanker, other types other; 36/37 pleasure/sailing dropped; no static data heard: untyped); `vessels` counts commercial vessels only (cargo + tanker, `basis: "cargo+tanker"`), and the navigational status (1 anchored, 5 moored, else underway) is of those; other and untyped are tallies beside the count, never in it (an untyped vessel could be a pleasure craft).
- `sample(bank, area, seconds=DEFAULT_SECONDS) -> Fact` (420 s, longer than the six-minute ShipStaticData cycle): entity `AREA.<NAME>`, attr `ships.count`, tier `reported`, value = commercial vessels seen, context `{"cargo", "tanker", "other", "untyped", "anchored", "moored", "underway", "window_s", "bbox", "basis"}`, `as_of` = sample date, `known_at` = sample end, source aisstream.io. Samples compare (z-scores) only with the same window, box and basis.

### 2.3 Traffic cameras (`brain/sources/cameras.py`, `brain/vision.py`)
- Providers (official, open): TfL JamCams `https://api.tfl.gov.uk/Place/Type/JamCam` (TfL Open Data; optional `TFL_APP_KEY`), Caltrans `https://cwwp2.dot.ca.gov/data/d{N}/cctv/cctvStatusD{NN}.json` for districts 1–12. No other providers unless added to `PROVIDERS` with their licence.
- `cameras(provider=None) -> list[dict]` `{"id": "TFL.00002.00865" | "CALTRANS.D04.tv102...", "provider", "name", "lat", "lon", "image_url", "licence"}` (cache 1 day); `near(lat, lon, km=1.0) -> list[dict]` sorted by distance.
- `brain/vision.py`: `VehicleCounter` using OpenCV DNN with YOLOX (Apache-2.0) from Hugging Face `opencv/object_detection_yolox`, file `object_detection_yolox_2022nov.onnx`, downloaded to `var/models/` and pinned by sha256. `count(image_bytes) -> {"car", "truck", "bus", "motorcycle", "vehicles"}` (COCO classes; score ≥ 0.5, NMS 0.5). No other classes are reported. Optional dependency `opencv-python-headless`; without it the sampler says so and writes nothing.
- `sample(bank, camera_ids) -> int`: fetch each image into memory, count, discard the bytes; fact entity `CAM.<id>`, attr `camera.vehicles`, tier `reported`, value = vehicles, context `{"car", "truck", "bus", "motorcycle", "provider", "name", "lat", "lon", "detector": "yolox-2022nov", "threshold": 0.5, "decode": 2}` (`decode` 2: a box whose best non-person class is not a vehicle, a train or a boat, is vetoed; samples compare, for z-scores, only with the same detector, threshold and decode), `as_of` = sample date, `known_at` = sample time, source `{"name": provider, "url": provider page}`. Images are never written to disk.

---

## 3. Watchlist and alerts (`brain/watch.py`)
Watchlist `var/watchlist.json`:
```json
{"tickers": ["AAPL"], "aircraft_of": ["AAPL"], "areas": [{"name": "LA-LONG-BEACH"}],
 "sites": [{"name": "Tesla Fremont", "ticker": "TSLA", "cameras": ["CALTRANS.D04.tv..."]}],
 "rules": {"price_move": 0.05, "insider_buy_usd": 100000, "earnings_car": 0.05, "zscore": 3.0,
           "forms": ["8-K", "10-K", "10-Q", "SC 13D", "S-1"]}}
```
```python
def load() -> dict; def save(w: dict) -> None
def add(kind, value, **kw) / remove(kind, value)                 # kind: ticker | aircraft | area | site
def check(bank, since: str | None = None) -> list[Fact]          # evaluate rules on facts known after `since` (default: last check, kept in var/alerts_state.json)
def refresh(bank, log=print) -> dict                             # ingest watched tickers (prices, filings, news, live insiders), poll aircraft, sample areas and site cameras
def run(every_s: int = 900, log=print)                           # refresh + check forever; one line per alert; Ctrl+C stops
def notify(alerts) -> None                                       # optional push via ntfy (BRAIN_NTFY_URL, e.g. https://ntfy.sh/<topic>)
```
Alert = fact entity (ticker / `AREA.*` / `CAM.*`), attr `alert`, tier `derived`, value = one-line message, context `{"rule", "severity": "info"|"notice"|"high"}`, inputs = the facts that triggered it. Rules: `price_move` (|1D| ≥ threshold), `filing` (a watched form), `insider_buy` (P by officer/director ≥ USD threshold), `earnings` (new earnings event; high if |car_3d| ≥ threshold), `aircraft` (a company aircraft landed), `ships` / `camera` (latest count's z-score vs the previous 30 samples ≥ threshold), `headline` (new headline, info; off for a new watchlist, `brain watch rule headlines on` turns it on). Acknowledged alert ids live in `var/alerts_state.json`.

CLI: `brain watch add|remove|list ...`, `brain watch run [--every 15m]`, `brain alerts [check|list|ack ID|ack-all]`, `brain insiders [--start 2010q4] [TICKER...]`, `brain jets link TICKER | poll`, `brain ships sample AREA [--seconds 420]`, `brain cams near LAT LON [--km 1] | sample ID...`.

---

## 4. Open-source model (`brain/llm.py`, `brain/terminal/analyst.py`, `brain/scout.py`)
- The analyst runs on any server that speaks the OpenAI-compatible chat-completions API with tool calling: Ollama (default), llama.cpp server, vLLM, LM Studio. No Anthropic dependency.
- Settings: `BRAIN_LLM_URL` (default `http://127.0.0.1:11434/v1`), `BRAIN_LLM_MODEL` (default `qwen3:8b`, Apache-2.0), optional `BRAIN_LLM_KEY`.
- `brain/llm.py`: `available() -> (bool, reason)` (GET `<url>/models`, model listed); `chat(messages, tools=None, temperature=0.2, timeout=300) -> dict` (the `choices[0].message`), standard library HTTP only.
- `brain/terminal/analyst.py` keeps its public API (`available()`, `ask(bank_reader, question, ticker=None, history=None, client=None)`), its tool set and rules, its validation and result shape; tools are sent in OpenAI function format; the loop runs at most 8 rounds; tool calls written as text (some small models do this) are parsed when they are a JSON object naming a known tool. Result `model` is the served model name.
- `brain/scout.py` (the brain reading the open web): `scout(bank, ticker, question=None, max_pages=8, log=print) -> dict`. Tools: the analyst's read-only bank tools, `web_search(query)` via a SearXNG instance (`BRAIN_SEARX_URL`; unavailable without it), `read_page(url)` (robots.txt respected via `urllib.robotparser`, our user agent, 1 request/s per host, HTML stripped to text, 20,000 characters max), and `note_claim(text, url, direction, tier, published)` which calls `brain.claims.note`. `note_claim` only accepts a URL that `read_page` actually read in this run. CLI: `brain scout TICKER ["question"]`. The terminal's ASK stays read-only.

---

## 5. Terminal (`brain/terminal/`)
New functions, same response envelope and `table` view unless noted:

| Code | Scope | What |
| --- | --- | --- |
| `INS` | security | insider trades, newest first: date filed, owner, relationship, code, shares, price, value; note with 90-day buy/sell totals |
| `ERN` | security | `events.earnings_table` rows; columns period end, filed, EPS, EPS YoY, SUE, 3-day reaction, 20-day and 60-day drift |
| `JET` | security | company aircraft (registry) and recent flights (from, to, departed, arrived) |
| `SHIP` | global | latest count per watched area with its z-score; `SHIP AREA` shows that area's history (view `chart` with `series` of counts) |
| `CAM` | global | latest vehicle count per watched site camera; `CAM ID` shows that camera's history (view `chart`) |
| `WL` | global | watchlist with last price, 1D, latest alert per row |
| `ALRT` | global | alerts newest first; unacknowledged ones flagged; `ALRT ACK ID` acknowledges |

The top bar shows the number of unacknowledged alerts; clicking it runs `ALRT`. `INGEST` accepts `INSIDERS` and `WATCH` (runs `watch.refresh`).

---

## 6. Tests
Every module gets unit tests on fixtures (no network): insider dataset rows and a Form 4 XML; quarterly Q4 derivation and TTM; SUE; earnings reactions with a point-in-time cut; FAA MASTER rows (corporate vs individual registrants, opt-out); landing detection from a sequence of positions; AIS message summaries; camera listings; the vehicle counter's post-processing on a synthetic output tensor; alert rules on synthetic facts; the LLM client and analyst loop against a local fake OpenAI-compatible server; scout's robots.txt and "only URLs it read" rules. Network tests are opt-in and never part of the suite.
