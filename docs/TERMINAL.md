# Terminal: spec and user guide

A keyboard-driven market terminal served locally by `python -m brain terminal`. Everything it shows comes from the memory bank (`brain/bank.py`) through point-in-time views, and the evidence engine (`brain/research/`) judges every signal it shows. An analyst (Claude, through tools that read the bank) answers questions in plain language and cites fact ids.

Inspired by professional market terminals; it uses no third-party branding. Product name in the UI: **MONEY TERMINAL**.

This document is the contract between the data layer, the server and the frontend. Shapes here are normative. If code and this file disagree, fix one of them in the same change.

---

## 1. Commands

Grammar: `[TICKER] FUNCTION [ARGS...]`, then Enter (the "GO" key). Case-insensitive. Also accepted:
- function first: `GP AAPL 5Y` = `AAPL GP 5Y`
- a bare ticker: `AAPL` = `AAPL DES`
- the noise words `US`, `EQUITY`, `EQUITIES`, `<GO>`, `GO` anywhere are ignored (`AAPL US EQUITY GP` works)
- market symbols as tickers: `^GSPC GP 5Y`, `EURUSD=X GP`, `BTC-USD DES`

| Code | Scope | Args | What |
| --- | --- | --- | --- |
| `HELP` | global | – | list of functions and keys |
| `DES` | security | – | description: identity, price summary, key derived metrics, latest filings and headlines |
| `GP` | security | `[RANGE] [CMP...]` | price chart. RANGE in `1M 3M 6M YTD 1Y 2Y 5Y 10Y MAX` (default `1Y`). Up to 3 comparison tickers (rebased to 100) |
| `FA` | security | – | financial analysis: up to 10 fiscal years of annual statements + ratios |
| `CF` | security | `[FORM]` | company filings; optional form filter (`10-K`, `8-K`, `4` ...) |
| `CN` | security | – | company news: headlines (unscored) + claims noted from the web with their source's record |
| `EV` | security | – | evidence: where the ticker ranks on each tested signal, and that signal's TEST verdict |
| `BRIEF` | security | – | the full cited brief (`brain/brief.py`) as markdown |
| `SHOW` | both | `ID` or `ENTITY/ID` | one fact by id prefix: full JSON and source link. Searches the ticker's facts, then `MACRO`, then the whole bank, so a citation opens from any pane |
| `WEI` | global | – | world markets: equity indices, volatility, FX, commodities, crypto |
| `MOV` | global | – | research-universe movers: last session and 1 month |
| `CURVE` | global | – | US Treasury par yield curve: latest, 1 month ago, 1 year ago, key spreads |
| `ECO` | global | – | macro table: Treasury spreads + FRED series if ingested |
| `RES` | global | – | research ledger verdicts |
| `CLM` | global | – | web-claim scoring per source |
| `ASK` | global | `QUESTION...` | ask the analyst (Claude) a question; it reads the bank through tools |
| `INGEST` | security or global | `TICKER` / `MARKETS` / `MACRO` / `UNIVERSE` | start a background collection job |
| `JOBS` | global | – | background jobs and their logs |

Unknown function → error `unknown function 'XYZ'` with a hint listing the closest codes (`closest: GP, CF`, or `HELP` when nothing is close). A security function without a ticker → error `FA needs a ticker, e.g. AAPL FA`.

When a line reads both ways, `[TICKER] FUNCTION` wins over function-first, so tickers that are also codes work (`CF DES`, `GP GP 1Y`). A GP range, a CF token that looks like a form (has a digit and no `.^=`), a SHOW fact id and the INGEST keywords are never taken as the ticker, so `GP 5Y` is an error rather than a chart of "5Y". Tickers must match the ingest pattern (§3).

---

## 2. Data layer (`brain/terminal/`)

### `commands.py`

```python
@dataclass
class Command:
    fn: str                 # upper-case function code, e.g. "GP"
    ticker: str | None      # upper-case, e.g. "AAPL", "^GSPC"
    args: list[str]         # remaining tokens, upper-cased except for ASK (kept verbatim)
    raw: str

FUNCTIONS: dict[str, dict]  # code -> {"scope": "security"|"global"|"both", "args": str, "desc": str}

def parse(line: str) -> Command          # raises CommandError(message, hint) on bad input
def complete(prefix: str, tickers: list[str]) -> list[dict]   # [{"text","kind":"function"|"ticker","desc"}], max 12
```

Ticker detection: the first token that is not a known function code and is not a noise word is the ticker, unless the function is global (then all remaining tokens are args). `ASK` keeps the rest of the line verbatim as one arg.

### `data.py`

```python
def run(bank: Bank, cmd: Command) -> dict     # dispatch; never raises for user errors
def run_line(bank: Bank, line: str) -> dict   # parse + run; a parse error is an ok:false result too
```

`JOBS`, `ASK` and `INGEST` need the server (jobs, the model), so `data.run` answers them `ok: false`; the server handles them itself (§3).

Point in time: every function reads the bank through a view at the end of the current UTC day (`T23:59:59Z`), not at the current second. A dated close counts as known at the end of its day, so a European close collected this evening would otherwise stay hidden until midnight UTC. Nothing dated after today becomes visible, and `brain/sources/prices.py` drops the bar of a session that is still open.

Every response is a JSON-serialisable dict:

```json
{"ok": true, "fn": "GP", "ticker": "AAPL", "args": ["5Y"], "title": "AAPL — Apple Inc. · price 5Y",
 "asof": "2026-10-09", "view": "chart", "data": {...}}
```
or
```json
{"ok": false, "fn": "FA", "ticker": "XYZ", "error": "no SEC facts for XYZ", "hint": "INGEST XYZ"}
```

`asof` is the latest date the data describes. `view` tells the frontend how to render `data`. A citation (`cite`) is the first 8 characters of a fact id; `ref` is `ENTITY/fullid`. Numbers are plain JSON numbers (not formatted strings) except where a field says `display`. Missing numbers are `null`, never NaN.

#### view `help` (HELP)
`data = {"functions": [{"code","scope","args","desc"}], "keys": [{"key","desc"}]}`

#### view `des` (DES)
```json
{"ticker": "AAPL", "name": "Apple Inc.", "kind": "equity|index|fx|commodity|crypto|other",
 "identity": [{"label": "Industry", "value": "Electronic Computers", "cite": "cc3ce885"}],
 "price": {"close": 336.64, "date": "2026-10-09", "chg_1d": 0.0123, "ret_1m": 0.031, "ret_3m": 0.062,
           "ret_ytd": 0.21, "ret_12m": 0.309, "vol_12m": 0.247, "from_high_12m": -0.013, "high_12m": 341.0,
           "low_12m": 210.2, "cite": "9de710d2"} | null,
 "metrics": [{"key": "market_cap", "label": "Market value", "value": 4.95e12, "unit": "USD", "as_of": "2026-10-09",
              "cite": "292044a9", "inputs": ["112e4f55", "9de710d2"]}],
 "filings": [{"date": "2026-07-31", "form": "10-Q", "desc": "", "url": "...", "cite": "7a85f912"}],
 "headlines": [{"time": "2026-10-09T13:00:00Z", "title": "...", "url": "...", "source": "finance.yahoo.com", "cite": "..."}]}
```
`metrics` = the derived facts the brief writes (market_cap, earnings_yield, pe, revenue_growth, net_margin, gross_margin, free_cash_flow, fcf_yield, asset_growth), computed through `brain.metrics` and written to the bank with `metrics.to_fact` (same as `brain/brief.py`). `pe` is display-only (1 / earnings_yield, no fact of its own; it carries a `display` string and cites the earnings-yield fact). Every metric row carries `as_of`, the date its derived fact describes (a fiscal-year end for the statement metrics, the price date for market value and the yields), and the frontend shows it next to the value; the brief's Derived lines show the same date. No metric is shown from a stale annual figure: a metric whose newest annual input is more than `brain.metrics.STALE_DAYS` (550) days older than the view is left out, as market value, earnings yield and asset growth already are, and `fcf_yield` is computed only from a fresh free cash flow. (A company that stops tagging a concept would otherwise show a figure from years ago as current.) `price` also has `chg_5d`. Up to 5 filings and 5 headlines. The filings are the company's own reports, `brain.brief.notable_filing`: 10-K, 10-Q, 8-K, 20-F, 40-F, 6-K, 10-KT, 10-QT, DEF 14A and their amendments (the brief's "Recent filings" uses the same rule). Insider and ownership reports and offering paperwork stay in `CF`; a bank files several 424B2s a day, which would otherwise fill the list. Identity values are shown as filed, except that repeated exchanges are dropped (`NYSE,NYSE,NYSE` → `NYSE`). Non-equities (indices, FX, ...) have empty identity/metrics/filings. `kind` comes from `brain/markets.py` for watched symbols, else `equity` when SEC identity facts exist, else `other` (ETFs such as SPY).

Return windows count trading sessions: 1D, 5D = 5, 1M = 21, 3M = 63, 12M = 252 (as in the brief); crypto trades every day and uses 7, 30, 91 and 365. YTD runs from the last close of the previous year.

#### view `chart` (GP)
```json
{"ticker": "AAPL", "range": "1Y",
 "series": [{"date": "2025-10-09", "close": 254.0, "adj": 252.1, "volume": 41000000}],
 "compare": [{"ticker": "SPY", "points": [{"date": "...", "value": 100.0}]}],
 "rebased": [{"date": "...", "value": 100.0}],
 "stats": {"first": 254.0, "last": 336.6, "ret": 0.325, "high": 341.0, "low": 210.2, "high_date": "...", "low_date": "..."},
 "splits": [["2020-08-31", 4.0]], "cite": "9de710d2", "kind": "equity"}
```
`series` uses `close` (split-adjusted) for the price axis; `ret` uses `adj` (total return). Downsample evenly to at most 1,500 points, always keeping the first and last. `rebased` is the main ticker's `adj` rebased to 100 at the range start (for comparing); `compare` tickers are rebased the same way on the main ticker's dates (forward-filled, NaN dropped). Missing comparison tickers are listed in `data.missing` (always present, possibly empty). `splits` lists every split, not only those in the range. YTD starts on the first trading day of the year of the series' last date.

#### view `table` (FA, CF, WEI, MOV, ECO, RES, CLM, JOBS)
```json
{"columns": [{"key": "fy", "label": "FY end", "type": "text|number|pct|money|date|link|ticker|verdict", "digits": 2}],
 "rows": [{"fy": "2025-09-27", "_cite": "650dff20", "_url": "...", "_group": "Income statement"}],
 "groups": ["Income statement", "Balance sheet"],
 "note": "free text shown under the table"}
```
Column types tell the frontend how to format: `pct` is a fraction (0.123 → +12.3%), `money` is a raw number (4.95e12 → 4.95T), `ticker` cells are clickable (run `<T> DES`), `link` cells hold a URL, `verdict` is PASS/FAIL/... and is colour-coded, `date` is an ISO date or timestamp. Row keys starting with `_` are metadata: `_cite` (fact id prefix), `_url`, `_group` (section header), `_ticker`, and:
- `_cites`: `{column key: cite}`, one citation per cell (FA).
- `_type`: overrides a numeric column's type for this row's cells (FA's columns are `money`; its EPS row is `number`, its ratio rows `pct`).
- `_digits`: decimals for this row's `number` cells (WEI gives FX rows under 10 four decimals: EUR/USD 1.1213).

- **FA**: one row per item, one column per fiscal year (newest first, at most 10). Items: Revenue, Gross profit, Operating income, Net income, EPS diluted, Operating cash flow, Capex, Free cash flow, R&D, Buybacks, Dividends paid, Total assets, Total liabilities, Equity, Cash, Long-term debt; then ratios Gross margin, Operating margin, Net margin, Revenue growth, ROE (net income / equity). Groups: `Income statement`, `Cash flow`, `Balance sheet`, `Ratios` (R&D sits in the income statement). Use `brain.metrics.annual` (flows: `flow=True`, balance sheet: `flow=False`) through `bank.view()`. Each cell's citation goes in a parallel row key `_cites: {"2025-09-27": "650dff20"}`; free cash flow and ratios are computed, so they have none. Columns are fiscal-year end dates, matched across statements within 7 days. Rows with no data are left out. ROE is blank when equity ≤ 0, revenue growth when the prior year's revenue ≤ 0. EPS is put on today's share basis: a figure filed before a later split is divided by it (Apple's FY2017 9.21 → 2.30 after the 2020 4:1 split), the note names the splits, and the citation keeps the as-filed number.
- **CF**: columns date, form, desc (8-K item names, as in `brain/brief.py` `ITEMS_8K`), link. Up to 200 rows, newest first. Optional form filter from args; a filter matches amendments too (`8-K` shows `8-K/A`).
- **WEI**: columns symbol (ticker), name, last, chg_1d (pct), chg_5d (pct), ret_1m (pct), ret_ytd (pct), date. Groups in this order: Americas, Europe, Asia-Pacific, Volatility, FX, Commodities, Crypto. Symbols come from `brain/markets.py`. Symbols not yet ingested are omitted, and `note` says `INGEST MARKETS`.
- **MOV**: groups `Top gainers, last session`, `Top losers, last session`, `Top 1 month`, `Bottom 1 month`; 8 rows each; columns symbol (ticker), name, last, chg_1d, ret_1m. Universe from `universe/us_large.json`.
- **ECO**: Treasury 3M, 2Y, 10Y, 30Y yields; spreads 10Y−2Y and 10Y−3M (latest, 1 month ago, 1 year ago); then FRED series if ingested (`brain/sources/fred.py` DEFAULT). Columns series, latest, m1, y1 (numbers), date, source; groups `Treasury yields`, `Treasury spreads`, `FRED`. Treasury rows are labelled `Treasury 10Y par yield, %` and `Treasury 10Y − 2Y, pp` so they never clash with FRED's labels.
- **RES**: columns id, claim, test_net (pct, label `TEST net/mo`), perm_p (number, 4 digits), bh_q (3 digits), verdict (verdict). From `brain.research.run.verdicts()`; declared hypotheses without a result show `not run`. `bh_q` is Benjamini–Hochberg over every hypothesis that has a result, withdrawn ones included (they were tried, so withdrawing one never lowers the others' q); a withdrawn row itself shows `bh_q: null`. `note`: controls status and the survivorship line.
- **CLM**: columns source, n, hits, hit_rate (pct), p_vs_coin; from `brain.claims.score`.
- **JOBS** is served by the server (§3), same table shape.

Empty data: RES with no results and CLM with no scored claims are `ok: true` with a note (no INGEST fixes them). CN for a ticker with data but no news returns empty lists. Anything else missing is `ok: false` with the command that fixes it: `INGEST <T>`, `INGEST MARKETS` (watched symbols), `INGEST MACRO`, `INGEST UNIVERSE` (MOV, and EV when the universe panel cannot be built). FA and CF on a market symbol hint `<T> GP`.

#### view `news` (CN)
`data = {"headlines": [{"time","title","url","source","cite"}], "claims": [{"time","text","url","source","tier","direction": 1|-1|0,"cite","record": {"n","hit_rate"} | null}]}` (newest first, up to 50 each).

#### view `evidence` (EV)
```json
{"ticker": "AAPL", "rebalance_date": "2026-09-30", "universe_size": 60,
 "signals": [{"signal": "mom12_1", "hypothesis": "H1-mom12_1", "value": 0.25, "percentile": 53, "in_top_third": false,
              "verdict": "FAIL", "test_net_excess": 0.0011, "perm_p": 0.1758, "claim": "..."}],
 "controls_pass": true, "summary": "0 of 4 signals passed their TEST gate. ..."}
```
Same computation as `brain/brief.py` `_signals_section` (reuse it via a shared helper, `brain.brief.evidence`; do not duplicate the maths). Signal rows also carry `bh_q`; a withdrawn hypothesis has verdict `WITHDRAWN...` and `bh_q: null`. The signals rank stocks, so EV on a watched market symbol (index, FX, commodity, crypto, volatility) is `ok: false` with hint `<T> GP`.

#### view `markdown` (BRIEF)
`data = {"markdown": "..."}` (from `brain.brief.build`).

#### view `fact` (SHOW)
`data = {"fact": {...Fact.to_dict()...}, "downgraded": false, "inputs": [{"ref","attr","value","cite"}]}`

#### view `curve` (CURVE)
```json
{"tenors": ["1 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"],
 "years": [0.083, 0.167, 0.25, 0.333, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
 "curves": [{"label": "Latest", "date": "2026-10-09", "yields": [4.13, null, ...]},
            {"label": "1 month ago", "date": "...", "yields": [...]},
            {"label": "1 year ago", "date": "...", "yields": [...]}],
 "spreads": [{"label": "10Y − 2Y", "latest": 0.44, "m1": 0.40, "y1": 0.10}], "cite": "..."}
```
Yields in percent. The `1.5 Month` column (only in recent years) is dropped (it stays in the stored series). Spreads, in percentage points: `10Y − 2Y`, `10Y − 3M`, `30Y − 5Y`.

#### Markets and Treasury collection
- `brain/markets.py`: `MARKETS: list[dict]` with `symbol`, `name`, `group`, `kind`; `ingest(bank, log=print) -> dict` that calls `brain.sources.prices.ingest(symbol, start="2015-01-01")` per symbol and adds the facts (skipping and logging symbols that fail). Symbols:
  Americas `^GSPC` S&P 500, `^DJI` Dow Jones Industrial Average, `^IXIC` Nasdaq Composite, `^RUT` Russell 2000, `^GSPTSE` S&P/TSX Composite, `^BVSP` Bovespa;
  Europe `^FTSE` FTSE 100, `^GDAXI` DAX, `^FCHI` CAC 40, `^STOXX50E` Euro Stoxx 50;
  Asia-Pacific `^N225` Nikkei 225, `^HSI` Hang Seng, `000001.SS` Shanghai Composite, `^NSEI` Nifty 50, `^BSESN` BSE Sensex, `^AXJO` S&P/ASX 200, `^KS11` KOSPI;
  Volatility `^VIX` CBOE Volatility Index;
  FX `DX-Y.NYB` US Dollar Index, `EURUSD=X` EUR/USD, `GBPUSD=X` GBP/USD, `JPY=X` USD/JPY, `CNY=X` USD/CNY, `INR=X` USD/INR;
  Commodities `CL=F` WTI crude, `BZ=F` Brent crude, `GC=F` Gold, `SI=F` Silver, `HG=F` Copper, `NG=F` Natural gas;
  Crypto `BTC-USD` Bitcoin, `ETH-USD` Ether.
- `brain/sources/treasury.py`: `URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"`. `parse_csv(body: bytes) -> pd.DataFrame` (columns `date` ISO + one column per tenor; header read dynamically because tenors differ across years; dates are `MM/DD/YYYY`, newest first; empty cells → NaN). `ingest(years=2) -> list[Fact]`: current and previous calendar year, concatenated, sorted ascending, stored with `brain.series.store`, one fact: entity `MACRO`, attr `series.treasury.par_curve`, tier `reported`, source `{"name": "US Treasury", "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve"}`, `as_of` = last date, `known_at` = last date (end of day). US government data, public domain; no key.
- CLI: `python -m brain markets` (ingest MARKETS) and `python -m brain macro` (Treasury always; FRED series only when `FRED_API_KEY` is set, otherwise print one line saying FRED was skipped). `python -m brain terminal [--host 127.0.0.1] [--port 8765] [--no-browser]` calls `brain.terminal.server.serve(host, port, open_browser)`.

---

## 3. Server (`brain/terminal/server.py`, `brain/terminal/jobs.py`)

Standard library only (`http.server.ThreadingHTTPServer`). One `Bank` held in server state, guarded by a `threading.RLock`; every request that touches the bank holds the lock.

```python
def make_server(host: str = "127.0.0.1", port: int = 8765, bank: Bank | None = None) -> tuple[ThreadingHTTPServer, str]
    # returns (server, token); port 0 picks a free port (tests)
def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None
```

Routes:

| Method | Path | Body / query | Response |
| --- | --- | --- | --- |
| GET | `/` | – | `static/index.html` with `{{TOKEN}}` replaced by the session token |
| GET | `/static/app.js`, `/static/style.css` | – | the file (fixed allow-list; nothing else under static is served) |
| GET | `/api/run` | `cmd=<command line>` | `data.run` result (JOBS handled by the server) |
| GET | `/api/complete` | `q=<prefix>` | `{"items": commands.complete(q, tickers)}` with tickers = bank entities |
| GET | `/api/status` | – | `{"entities": n, "tickers": [...], "universe": n, "jobs_running": n, "analyst": {"available": bool, "reason": str}, "time": iso}` |
| GET | `/api/fact` | `ticker`, `id` | same as `SHOW` |
| POST | `/api/ask` | `{"question": str, "ticker": str \| null, "history": [{"role","content"}]}` | `analyst.ask(...)` result (§4) |
| POST | `/api/ingest` | `{"target": "AAPL" \| "MARKETS" \| "MACRO" \| "UNIVERSE"}` | `{"ok": true, "job": {...}}` |
| GET | `/api/jobs` | – | `{"jobs": [...]}` newest first |

Successful `/api/status`, `/api/complete` and `/api/jobs` responses also carry `"ok": true`.

`/api/run` handles three functions itself instead of passing them to `data.run`:
- `JOBS`: the jobs table below.
- `INGEST T1 [T2 ...]`: validates every target (at most 10) before queuing any, then returns the JOBS table titled `INGEST KO: job j1 running`. A target already queued or running returns its existing job.
- `ASK [question]`: opens the analyst pane and never calls the model (a GET must not spend money): `view: "ask"`, `data = {"question": str, "ticker": str | null, "available": bool, "reason": str}`. The pane then POSTs the question to `/api/ask`.

`/api/fact` requires `ticker` to match the ticker pattern and `id` to be 1–20 hex characters (else 400) and runs `SHOW` directly, so a ticker that is also a function code still works. `/api/ask` takes a question of at most 4,000 characters and keeps the last 50 history messages; the analyst reads the last 10, at most 8,000 characters each, so the frontend sends no more than that and keeps the whole body under 60 KB.

The bank is read in full before `serve()` prints the URL (about 5 s for 60 stocks and the world markets). A Bank otherwise loads each entity on first use, and MOV and EV touch the whole universe, so their first call would hold the bank lock and stall every pane for several seconds. After a job, the fresh Bank is likewise read in full before the swap, outside the lock, and panes keep answering from the old one meanwhile. A reload that started earlier never replaces one that started later.

Security (all required):
- Bind `127.0.0.1` by default. If `--host` is anything else, print a warning that the terminal has no user accounts.
- **Host check**: reject (403) any request whose `Host` header is not `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>` when bound to loopback. This blocks DNS-rebinding.
- **Token**: every `/api/*` request must carry header `X-Terminal-Token: <token>` (constant-time compare, `hmac.compare_digest`); else 401. The token is random per server start (`secrets.token_urlsafe(32)`), injected only into the HTML page. No CORS headers are ever sent.
- POST bodies: `Content-Type: application/json`, max 64 KB, else 400/413.
- Responses: `Content-Type: application/json; charset=utf-8`, `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`; the HTML page also gets `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'` and `Referrer-Policy: no-referrer`.
- Errors inside a handler → 500 with `{"ok": false, "error": "internal error"}` and the traceback logged to stderr, never sent to the client.
- Ingest targets are validated: a ticker matches `^[A-Za-z0-9.^=\-]{1,15}$`, or one of the keywords. Jobs run as subprocesses with an argument list (never a shell): `[sys.executable, "-m", "brain", "ingest", T]`, `["markets"]`, `["macro"]`, `["ingest-universe"]`, with `cwd` = repo root and the server's environment. At most 2 concurrent jobs; others queue. Each job keeps the last 200 log lines. When a job finishes, the server replaces its `Bank` with a fresh `Bank()` under the lock, so new facts appear.
- Job record: `{"id","target","status": "queued|running|done|failed","started","finished","returncode","log": [str]}`. `JOBS` renders them as a table (columns id, target, status (text), started, finished (date, UTC), last log line; ticker targets carry `_ticker`); its note gives the running and queued counts and the newest job's last 10 log lines. Ids are `j1`, `j2`, ...; at most 50 jobs wait in the queue and the last 100 are remembered. On shutdown, queued jobs are cancelled and running children are left to finish.

---

## 4. Analyst (`brain/terminal/analyst.py`)

```python
def available() -> tuple[bool, str]       # SDK importable and credentials resolvable; reason if not
def ask(bank_reader, question: str, ticker: str | None = None, history: list | None = None) -> dict
```

`bank_reader` is a callable `fn(cmd_line: str) -> dict` that runs a terminal command against the bank (the server passes `run_line(..., read_only=True)`, which holds the lock and refuses `INGEST` and `ASK`). The analyst's tools are thin wrappers over terminal functions: `describe` (DES), `price_history` (GP, sampled to 60 points), `financials` (FA), `filings` (CF), `news` (CN), `evidence` (EV), `show_fact` (SHOW), `research_results` (RES), `world_markets` (WEI), `movers` (MOV), `yield_curve` (CURVE), `macro` (ECO), `claim_scores` (CLM). They read the bank, with one exception that BRIEF shares: `describe` records the derived metrics DES computes (§2, view `des`), so the ids it cites resolve with `show_fact`. Nothing else is written and no job starts.

Tool inputs are validated (tickers ≤ 15 characters without spaces, fact ids of 4–20 hexadecimal characters) before any command runs; a rejected input goes back to the model as an `is_error` tool result (the SDK's `ToolError`, which it does not log as a traceback). Tool results are compacted before the model sees them: GP to 60 points, tables to 40 rows, CN to 20 headlines and 20 claims, every string to 600 characters (ending `… [N more characters cut]`), and a serialized result to 40,000 characters (then `{"truncated": true, "note", "partial"}` with the first 40,000 characters as a string). A stored headline or a noted claim can be any length; real results are under 15,000 characters.

`available()` builds a client without sending anything and checks that it found an API key, a token, or a profile or federation config the SDK can load. A config directory with no usable profile is not credentials, and a named profile that is missing (`ANTHROPIC_PROFILE=work`) is reported with the SDK's message instead of failing halfway through a question.

Model `claude-opus-5-5` through the SDK tool runner, effort `high`, adaptive thinking (the default), server-side refusal fallback (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`), automatic prompt caching, at most 12 tool rounds. Install with `pip install anthropic` (or `pip install -e .[analyst]`).

Result: `{"ok": true, "answer": "<markdown>", "tool_calls": [{"name": "describe", "input": "AAPL DES"}], "model": "...", "stop_reason": "...", "usage": {"input_tokens","output_tokens","cache_read_input_tokens","requests"}}` or `{"ok": false, "error": "...", "hint": "..."}` (e.g. no credentials: error `analyst unavailable`, hint `no Anthropic credentials: export ANTHROPIC_API_KEY=...`).
- `usage` covers the whole answer: every request the tool runner made and, within each, every fallback attempt (`usage.iterations`; the top-level usage of a response covers only the attempt that produced it). `input_tokens` includes tokens written to and read from the prompt cache; `cache_read_input_tokens` is the part read from it; `requests` counts the model requests.
- `stop_reason: "max_tokens"`: the answer ends with `*(answer cut off at the length limit)*`. `"tool_use"`: the 12-round limit ended the run while the model still wanted tools; whatever text came with the last call is a preamble, so the answer always ends with `*(stopped after 12 tool rounds without a final answer; ask a narrower question)*`.
- `"refusal"`: `ok: false`, error `the model declined this request`, hint = `stop_details.explanation`, plus `tool_calls` and `usage`.
- API errors: 401 → `Anthropic rejected the credentials`, 429 → `rate limited by the Anthropic API`, other statuses → `Anthropic API error <status>` (hint: the API's message), network → `could not reach the Anthropic API`, credentials that fail during the request → `no usable Anthropic credentials`, any other SDK error → `Anthropic SDK error (<class>)`; none of them raises.

Rules the analyst follows (system prompt): answer only from tool results; cite fact ids as `[abcd1234]`; separate filed/market facts from claims and opinions; quote research verdicts plainly, including FAILs; say what to INGEST when data is missing; no orders, no promises of returns.

---

## 5. Frontend (`brain/terminal/static/`)

Plain HTML/CSS/JS, no external requests (works offline, passes the CSP above). Files: `index.html`, `app.js`, `style.css`.

- **Look**: black background, amber function text, white values, green up / red down, monospace throughout (`ui-monospace, "SF Mono", Menlo, Consolas, monospace`), dense 12–13px tables, 1px dividers. No logos.
- **Layout**: top bar (MONEY TERMINAL, world clocks for New York, London, Mumbai, Tokyo, UTC; bank stats; job indicator; analyst availability), a command line, a 2×2 grid of panes, a bottom ticker tape of WEI rows that scrolls slowly and refreshes every 60 s. Panes resize with the window; at widths under 900px the grid becomes one column.
- **Panes**: each has a header (pane number, current command, title, `asof`, a reload button, a maximise toggle) and a body. The title drops what the command beside it already says (`world markets` under `WEI`, `Apple Inc. · description` under `AAPL DES`); the full title is its tooltip. Clicking a pane focuses it; the command line targets the focused pane. Default layout on first load: 1 `WEI`, 2 `MOV`, 3 `SPY GP 1Y`, 4 `ASK`. The layout (each pane's command) is saved in `localStorage` (wrapped in try/catch) and restored. A reload, a restored layout or the refresh after a job never starts a job or asks the model again: `INGEST X` reruns as `JOBS`, `ASK question` as `ASK`.
- **Command line**: Enter runs; Esc clears; Up/Down walks history (saved, last 100); typing shows an autocomplete list from `/api/complete` (debounced 120 ms; Tab or click accepts). `F1` runs HELP. `Ctrl+1..4` focuses a pane. `Ctrl+Enter` runs the command in the next pane instead.
- **Rendering by `view`**:
  - `table`: sticky header; group header rows; numbers right-aligned; `pct` coloured by sign; `ticker` cells clickable (run `<T> DES` in the same pane; Shift-click: next pane); citations shown as a small dim `[abcd1234]` that opens `SHOW` in a popover.
  - `chart`: canvas line chart of `close` with date and price axes, volume bars along the bottom, crosshair + tooltip on hover (date, close, change from start), range buttons (`1M 3M 6M YTD 1Y 2Y 5Y 10Y MAX`) in the pane header that rerun the command; with comparisons, plot `rebased` and each `compare` series (each a different colour, legend). Redraw on resize. Device-pixel-ratio aware.
  - `curve`: canvas yield curves (x = years on a square-root scale, labelled by tenor), one line per curve, legend, spread table under it.
  - `des`: two-column key/value layout plus mini tables for filings and headlines.
  - `news`, `evidence`, `fact`, `help`: dedicated layouts using the same table styles.
  - `markdown`: rendered by a small safe renderer (escape HTML first; then headings, bold, lists, tables, links restricted to `http:`/`https:` with `rel="noopener noreferrer" target="_blank"`, inline code, `[abcd1234]` citations made clickable).
  - `ask` (pane running `ASK`): chat log, input at the bottom, "thinking…" indicator while waiting, each answer rendered as markdown with a collapsible list of tool calls. Status shows when the analyst is unavailable, with the hint. History sent with a question: earlier successful turns only, the last 10 messages, 8,000 characters each, dropping the oldest pairs until the body fits in 60 KB. The chat lives in memory per pane and is not saved.
- **Errors**: `ok: false` renders the error in red and the hint as a clickable command when it reads as one (a function code first or second); `closest: GP, CF` becomes one link per code with the unknown word replaced. Other hints (`export ANTHROPIC_API_KEY=...`) stay text.
- **Safety**: all server text is inserted with `textContent` or through the escaping markdown renderer. No `innerHTML` with unescaped data, no `eval`, no inline event handlers (CSP).
- Every API call sends `X-Terminal-Token` read from `<meta name="terminal-token" content="{{TOKEN}}">`.

---

## 6. Tests

- `tests/test_terminal_commands.py`: grammar, noise words, function-first, bare ticker, ASK verbatim, errors with hints, completion.
- `tests/test_terminal_data.py`: synthetic bank (see `tests/util.py`) → each function returns the documented shape; GP downsampling keeps endpoints and ≤1,500 points; comparisons rebased; FA columns newest first; WEI/MOV/CURVE with synthetic series; missing data → `ok: false` with a hint; no NaN anywhere in JSON (`json.dumps(..., allow_nan=False)` must succeed).
- `tests/test_treasury.py`: CSV parsing with both old (no `1.5 Month`) and new headers.
- `tests/test_terminal_server.py`: server on port 0 in a thread; 401 without token; 403 with a foreign Host; `/` injects the token; `/static/../x` and unknown static files 404; `/api/run` works; bad JSON 400; oversize body 413; ingest target validation; JOBS listing.
- `tests/test_analyst.py`: no credentials (also an empty config directory, a missing named profile) → `ok: false` with hint; with a fake client, tool calls dispatch to terminal commands and the answer comes back; the SDK's real tool runner on a mock transport: refusal, max_tokens, the tool-round limit, 401/429/500/529 and a network error, a rejected tool input as an `is_error` result with no logged traceback, history in the request, usage summed over requests and fallback attempts; tool results capped in size; the server's reader writes only derived facts and refuses INGEST and ASK.
- `tests/test_terminal_static.py`: the page has the token meta, loads only `/static/app.js` and `/static/style.css`, and has no inline script, style or handlers; `app.js` uses no `innerHTML`, `eval` or string timers, calls only routes the server defines, renders every view, and lists the same function codes and ranges as `commands.py`.
