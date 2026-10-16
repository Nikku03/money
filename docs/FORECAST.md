# Forecasts: a weather report for a stock

`brain forecast AAPL` gives a weather-style forecast for one stock over the next 5 or 21 trading sessions:

- **the likely range of its price** (a 68% band and a 90% band);
- **the chance it beats SPY** over that time;
- **the main drivers** behind that chance;
- **its own record**: how such forecasts did on held-out history and live.

There are two forecasters.

- **F1, the learner.** A numpy-only model learns from history how much each of about forty variables matters:
  - price patterns;
  - fundamentals;
  - behavioural signals (how people over- and under-react);
  - the market regime.
- **F1-llm, "thinking".** The local open-weight model reads a point-in-time snapshot of the stock, F1's numbers and a primer on how markets and people work (`docs/MARKET_PRIMER.md`). It then gives its own probability.

Everything is tested before it is believed. The tests in section 7 are declared in the research ledger, with their code, primer and model pinned, and committed **before** they run. Forecasts are paper only: nothing here trades.

**Expectation, stated before running.**
- **The range** should be roughly right, because how much a stock moves (its volatility) persists.
- **The chance to beat SPY** will likely be close to a coin. Every single signal tested here so far failed, and public patterns are priced quickly.
- **Detection limits.** Weekly ICs (section 7) vary by about 0.16 to 0.20 from week to week, so each test can only detect a mean IC of at least roughly the following, at 80% power and one-sided 5%:

  | Test | What it detects |
  | --- | --- |
  | the last two months (F1a, 8 weeks) | mean IC >= 0.15 |
  | 2019-2026 weekly (F1b, about 390 weeks) | mean IC >= 0.025 |
  | 2019-2026 monthly (F1c, about 90 windows) | mean IC >= 0.05 |

  So F1a and F1d are reported as descriptive retests, with their uncertainty, not as pass/fail. The verdicts come from F1b and F1c.

---

## 1. Files and owner groups

| File | What | Group |
| --- | --- | --- |
| `brain/forecast/__init__.py`, `brain/forecast/features.py` (new) | point-in-time feature matrices, the outcome rule (sections 2, 3) | features |
| `brain/forecast/learn.py`, `brain/forecast/gbm.py` (new) | logistic, numpy gradient boosting, calibration, range model (sections 4, 5) | learn |
| `brain/forecast/evaluate.py`, `brain/research/forecast_tests.py` (new); `brain/research/run.py` (verdicts/report integration) | walk-forward runs, metrics, controls, ledger, pinning (sections 7, 8) | learn |
| `docs/MARKET_PRIMER.md`, `brain/forecast/think.py` (new); `brain/judge.py` (primer in the judge's prompt) | the LLM's knowledge, its snapshot and probability, model identity (section 9) | think |
| `brain/forecast/live.py` (new), `brain/cli.py`, `brain/terminal/*`, `docs/TERMINAL.md` | CLI card, the `forecast` fact, live scoring, terminal FCST (section 10) | product |
| `README.md`, `CLAUDE.md`, this file | docs | integration |

Each group edits only its files and codes against this contract for the others. Tests go in `tests/test_forecast_<group>.py`.

**Development data rule.** Until the declared runs, no code may compute a forecast, feature-based metric or LLM answer on the real bank for any date after 2018-12-31.
- Development uses synthetic banks, or real data in the TRAIN years (2011-2018) only.
- Trial mode (section 8) enforces this.
- The held-out years are computed once, by the recorded runs.
- Live forecasts enforce it too: on the real bank (MONEY_HOME unset or `var/`) `brain forecast TICKER` refuses until the committed ledger holds a non-trial result for each of F1a, F1b and F1c (section 10).

---

## 2. Timing, outcome, samples

**Forecast date `d`.** A forecast is made after the close of a SPY session `d`. It reads facts through `bank.view(T(d))`, with `T(d) = d + "T23:59:59Z"`: filings, insider trades, earnings events, headlines.

**Prices.**
- All prices come from **one `PriceBook` per run, built at the run's pinned `prices_asof`**, plus the same snapshots' daily CSVs for high, low and volume (`series.load`).
- Every array is sliced to bars dated `<= d` for features.
- The bank's price snapshots were all fetched recently, so a book at `asof = T(d)` would have no prices for past `d`.
- ^VIX comes from rows dated `<= d` in its latest snapshot (history from 2015).

**Outcome** (`features.outcome`, the only label function):
```python
def outcome(book, ticker, d, h) -> dict | None:   # claims.outcome(book, ticker, T(d), h, "SPY") verbatim
    # entry = the first SPY session strictly after d; exit = h SPY sessions after the entry;
    # {"entry", "exit", "excess"}; None when a close is missing or more than 4 days stale
```
- The window starts at the **next** session's close. Something public after the close of `d` but stamped on `d`, such as an after-close 10-Q, a same-day Form 4 or an evening headline, is never inside a window it could have predicted.
- `y = 1` when `excess > 0`. An exactly-zero excess is dropped.
- Every sample stores its `exit`.
- **Every** cutoff uses the stored exit, never `d + h` arithmetic: training, the calibration slice, range quantiles, the live model.

**Universe.**
- `universe/us_broad.json` minus SPY.
- A stock is listed at `d` when it has a close dated within 4 days of `d` and at least 252 bars up to `d`.
- The universe is chosen by today's market value, so it holds survivors: "small then" predicts "rose later". Section 7 measures and removes the size effect before any gate.

**Forecast grid** (`evaluate.grid`).
- Every `h`-th SPY session counted back from an anchor session `A`: `d_k` is the session `k*h + 1` sessions before `A` (k = 1, 2, ...).
- The entry is the session after `d_k` and the exit `h` sessions after the entry, so `d_1`'s exit is `A` itself and each window starts at the close where the next one ends. The windows never overlap.

**Training samples.** The same grid, back to 2011-01-01. A model trained at cutoff `C` uses only samples whose stored `exit <= C`.

---

## 3. Features (`brain/forecast/features.py`)

```python
FEATURES_VERSION = 1
FEATURES: list[tuple[str, str, str]]          # (name, group P|B|F|R, plain-words description): the 40 below plus 9 indicators
def prices(bank, tickers, prices_asof) -> Prices          # dates, adj, close, high, low, volume (T, N); spy (T,); vix series
def outcome(book, ticker, d, h) -> dict | None
def stock_features(px, i) -> dict[str, np.ndarray]       # at calendar index i (= d); (N,), NaN when undefined
def fundamental_features(bank, tickers, month_ends, book) -> dict[str, np.ndarray]   # (M, N) at view(T(month_end))
def event_features(bank, tickers, d, book) -> dict[str, np.ndarray]                  # insider and earnings at view(T(d))
def regime_features(px, i) -> dict[str, float]
def feature_set(bank, tickers, grid, prices_asof, h, *, cache=True, workers=None, log=None, px=None) -> FeatureSet
    # X (S, F), y, excess, exit, date and ticker per sample; names, groups; dropped counts
    # also raw (float32, before transforms), r_total, close, entry, pending (listed rows whose window is open), key
```

**P: price patterns** (bars dated `<= d`):
- `r_5`, `r_21`, `r_63`, `r_126`: past returns.
- `mom_12_1`: the return from `d-252` to `d-21` sessions.
- `ma_50`, `ma_200`: close over its 50- and 200-session mean, minus 1.
- `rsi_14`: Wilder's 14-session RSI.
- `vol_21`, `vol_63`: the standard deviation of daily log returns.
- `vol_ratio`: `vol_21 / vol_63`.
- `beta_252`: the slope against SPY's daily returns over 252 sessions.
- `idio_vol_63`: the residual standard deviation after removing `beta_252`, over 63 sessions.
- `skew_63`.
- `range_21`: the mean of `(high-low)/close` over 21 sessions.

**B: behavioural:**
- `dist_52w_high`, `dist_52w_low`: anchoring.
- `max_ret_21`: lottery preference.
- `min_ret_21`.
- `rev_5`, `rev_21`: minus `r_5` and minus `r_21` (short-term overreaction).
- `ltr_3y`: minus the return from `d-756` to `d-252` (long-term overreaction).
- `abn_volume`: log of the 5-session mean volume over the 63-session mean volume (attention).
- `turnover_trend`: log of the 21-session mean dollar volume over the 252-session mean.
- `insider_buy_90d`: log1p of the USD of code-P purchases by officers and directors in Form 4/5 filings known at `T(d)` within 90 days. `signals.insider_buying`'s rule.
- `earnings_car_3d`: the 3-day reaction (`events.reaction`) to the latest distinct earnings release known at `T(d)` whose reaction window closed by `d`, within 63 sessions. Otherwise 0, with `has_earnings_recent = 0`.

**F: fundamentals.**
- Computed at each month-end view (`view(T(month_end))`, month-ends on SPY's calendar), with market values from the run's book at that date.
- A forecast on `d` uses the last SPY session before the first day of `d`'s month (`features.month_end_index`), so knowing which month-end applies never needs a bar after `d`. A forecast on a month's last session therefore uses the month-end before it.
- Carried forward to later forecast dates until the next month-end; never carried back.
- The features:
  - `earnings_yield_ttm` (`metrics.earnings_yield_ttm`);
  - `log_mktcap` (`metrics.market_cap_any`);
  - `sales_growth` (`metrics.revenue_growth`);
  - `gross_profitability` (`metrics.ratio` of gross profit to assets, latest fiscal year; when the bank has no current `GrossProfit`, revenue minus cost of revenue, `CostOfRevenue` else `CostOfGoodsAndServicesSold`, over assets);
  - `asset_growth` (`metrics.asset_growth`);
  - `accruals` (net income minus operating cash flow, over assets);
  - `leverage` (liabilities over assets; when the bank has no current `Liabilities`, 1 minus `StockholdersEquity` over assets);
  - `sue` (`metrics.sue`; **month-end only**).
- An implementer uses the bank's existing concept lists and `metrics` helpers. A concept with no helper is computed with `metrics.annual`, and the concept names are listed in `features.py`.
- Each is NaN when the bank lacks it, with a `has_<name>` indicator (group F). `has_earnings_recent` (group B) is the indicator for `earnings_car_3d`. Indicators stay 0/1 and are not ranked.

**R: regime** (the same for every stock at `d`): `vix`, `vix_chg_21`, `spy_r_21`, `spy_r_252`, `spy_above_200`, `mkt_vol_21`. The Treasury curve is not used: the bank holds it only from 2025.

**Transforms** (per date):
- P, B and F are ranked across listed stocks into [-0.5, 0.5], ties sharing their mean rank. NaN becomes 0, and the indicators carry the gap.
- R features are z-scored by the mean and standard deviation of their own values on earlier grid dates only, with at least 52 earlier dates (else 0).

**Cache** (`<home>/cache/forecast/`). The key holds:
- `FEATURES_VERSION`;
- the sha of `features.py`;
- the universe sha, grid, h and `prices_asof`;
- the price-snapshot ids;
- every ticker's journal shard state (size and inode, as `signals._shard_state`);
- the sha of the modules the features read (`features.deps_sha`: metrics, events, claims, sources/insiders, research/signals, series).

A backfill or a code change recomputes. The key goes into each result.

---

## 4. The learner (`brain/forecast/learn.py`, `brain/forecast/gbm.py`)

```python
LEARN = {
  "logistic": {"l2": 1.0, "iters": 200},
  "gbm": {"trees": 120, "depth": 3, "lr": 0.05, "bins": 64, "min_leaf": 200, "subsample": 0.8, "colsample": 0.8, "seed": 7},
  "blend": "mean of the two probabilities",
  "calibration": "Platt scaling on the training samples with the latest 20% of exits; models refit on the rest first",
  "max_rows": 150000,          # training rows, by a date-stratified draw seeded 7 when more
  "since": "2011-01-01",
}
def fit(fs: FeatureSet, cutoff: str, spec=LEARN) -> Model     # only samples with exit <= cutoff; deterministic
def predict(model, X) -> np.ndarray                           # P(y = 1) in (0, 1)
def drivers(model, x_row, names, k=5) -> list[dict]          # [{"name", "group", "push"}]: logistic coefficient * value
def to_json(model) -> dict; def from_json(d) -> Model         # plain JSON, no pickle
```
- **GBM.** Histogram gradient boosting for log loss in numpy:
  - quantile bins from the training rows;
  - depth-limited trees built from `np.bincount` histograms, using the parent-minus-sibling trick;
  - shrinkage;
  - seeded row and column subsampling.
- **Hyperparameters** are fixed here, frozen in each spec, and never tuned on any test window.

---

## 5. The range (`learn.py`)

```python
RANGE = {"ewma_lambda": 0.94, "lookback": 252, "quantiles": [0.05, 0.16, 0.5, 0.84, 0.95]}
def sigma(px, i) -> np.ndarray                 # (N,) EWMA daily log-return volatility at d
def fit_range(fs, cutoff, px) -> dict          # quantiles of z = log(1 + r) / (sigma_d * sqrt(h)), samples with exit <= cutoff
def band(close_d, sigma_d, h, q) -> dict       # prices close_d * exp(q * sigma_d * sqrt(h)) for each quantile
```
`r` is the stock's own total return from the close of `d` to the close at the exit (`learn.range_inputs`, from the run's price panel), the same move the band is quoted for (section 6). It is one session longer than `FeatureSet.r_total` (entry to exit), which the range does not use. The 68% band is `[q16, q84]`, the 90% band `[q05, q95]`. Bands are on total return, so they swing with the whole market.

---

## 6. A forecast
```python
{"ticker", "date": d, "horizon": h, "entry": date, "exit": date, "close": float, "p_beat": float,
 "band": {"q05", "q16", "q50", "q84", "q95"}, "move68_pct": [lo, hi], "move90_pct": [lo, hi],
 "drivers": [{"name", "group", "push"}], "model": {"id": "F1", "sha", "trained_through": C, "rows": int}}
```
`date` is `d`. The band is for the close at `exit`, measured from the close at `d`.

---

## 7. Declared tests (`brain/research/forecast_tests.py`)

```python
FORECAST_TESTS = [
  {"id": "F1a-w5-recent", "forecaster": "F1", "horizon": 5, "periods": 8, "gate": "report",
   "retrain": "once, at cutoff = the first forecast date d_8",
   "claim": "Retest on the last 8 non-overlapping weeks (about two months, held out): how F1's chance-to-beat-SPY "
            "forecasts and ranges did. Descriptive: detects mean IC >= 0.15 at 80% power."},
  {"id": "F1b-w5-test", "forecaster": "F1", "horizon": 5, "start": "2019-01-01", "end": "<F1a's d_8>", "gate": "ic",
   "retrain": "every 26 periods (about half a year), cutoff = that forecast date",
   "claim": "From 2019 to the start of F1a's window, F1's weekly forecasts rank stocks by what they then did beyond size: "
            "mean size-neutral IC > 0."},
  {"id": "F1c-m21-test", "forecaster": "F1", "horizon": 21, "start": "2019-01-01", "end": "<anchor>", "gate": "ic",
   "retrain": "every 6 periods",
   "claim": "From 2019 to the anchor, F1's 21-session forecasts have mean size-neutral IC > 0."},
  {"id": "F1d-llm-w5-recent", "forecaster": "F1-llm", "horizon": 5, "periods": 8, "gate": "report",
   "tickers": ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "JPM", "CVX", "JNJ", "PG", "WMT", "KO", "CAT"],
   "retrain": "F1a's model",
   "claim": "Retest on F1a's 8 weeks and these 12 stocks: does the local model's probability, given F1's forecast, a "
            "point-in-time snapshot and the market primer, have a lower Brier score than F1 alone? Descriptive."},
]
```
**Common spec fields** (`brain forecast declare` fills them in and writes them into the spec, so the spec sha covers them):
- `"kind": "forecast"`, plus `universe`, `universe_sha`, `LEARN`, `RANGE`, `FEATURES` (names and `FEATURES_VERSION`).
- `"anchor"`, `"prices_asof"` and `"code_sha"`: the sha of every file under `brain/forecast/`, plus the modules the features read (`brain/metrics.py`, `brain/events.py`, `brain/claims.py`, `brain/sources/insiders.py`, `brain/research/signals.py`, `brain/series.py`), `brain/research/forecast_tests.py` and `docs/MARKET_PRIMER.md`.
- `"gate"` details.
- **For F1d:** `"llm"`: `{"model", "digest", "cutoff", "temperature": 0, "seed": 7, "primer_sha", "prompt_sha", "snapshot_version"}`.
  - `digest` is the server's weights digest: Ollama `/api/show`, or the sha256 of a local GGUF file when the server reports a path.
  - `cutoff` is the model's stated training-data cutoff, passed as `--llm-cutoff` at declare. Declare refuses unless it is earlier than F1a's first forecast date.

**Anchor `A`.** The last SPY session for which every universe stock that was listed 5 sessions earlier has a bar dated exactly `A` in the run's book. It is pinned at declare with `prices_asof = now`.
- F1a's 8 weeks are the 8 grid dates before `A`, so their exits run up to `A` (`d_1`'s exit is `A`).
- F1b ends at F1a's first date, so they share no week.
- F1c's grid is counted back from `A`.

**Metrics** (`brain/forecast/evaluate.py`). Per date, then over dates:
- **IC.** The Spearman rank correlation between `p_beat` and `excess` across listed stocks, per date.
- **Size-neutral IC.** The same, after replacing `p_beat`'s rank by its residual from a least-squares fit on `log_mktcap`'s rank within that date. Survivorship rewards small stocks here, so this is the gated IC.
- **Baselines on the same samples:** size alone (`-log_mktcap`) and momentum alone (`mom_12_1`), each with its IC and size-neutral IC.
- **Spread.** Top-quintile minus bottom-quintile mean excess per date (F1d: top half minus bottom half of 12).
- **Brier score.**
  - Against climatology, the training base rate of `y` as a constant: `brier_skill = 1 - brier / brier_clim`.
  - Decomposed per date into a level term (mean `p_beat` against that date's share beating SPY) and a within-date term. Only the within-date term speaks to ranking: weekly breadth (28%-70% beat SPY) dominates the level term.
- **Placement.** The share of stocks a forecast places correctly above or below that date's median excess.
- **Calibration table.** Deciles of `p_beat` (F1d: quintiles): mean forecast, realised frequency, n.
- **Range.** Coverage of the 68% and 90% bands, per date, with the mean over dates and a date-level 90% interval (the sd of the per-date coverage over sqrt(K)). Also the mean band width, and the pinball loss against a naive normal band from trailing 252-session volatility. A market-wide move pushes most stocks out of their bands in the same week, so only the date-level interval is honest.

**Gate** (`"ic"`: F1b and F1c):
- The mean size-neutral IC is > 0, with **date-level** p < 0.05: a one-sided sign-flip test on the series `IC_1..IC_K`, exact for K <= 16, else 20,000 Monte Carlo draws seeded 7.
- BH q < 0.10 across every hypothesis and gated forecast test in the ledger, withdrawn ones included.
- The within-date permutation p is reported as a diagnostic only. It ignores how stocks move together and is several times too narrow.

**Controls** (per gated run):
- **Oracle.** `p_beat` = the realised excess rank. It must reach the test's minimum attainable p.
- **Placebo.** F1's own `p_beat` with its sign flipped by a fair coin on each date, 200 draws. The gate's test must reject in 1%-10% of draws.

Otherwise the result is FAIL (controls).

**Descriptive tests** (`"report"`: F1a and F1d) are recorded with the same metrics, plus each mean's date-level 90% interval. They have no verdict and are kept out of BH. They are labelled "retest (descriptive)": "inconclusive" when the interval spans 0, otherwise "positive" or "negative", with the detectable size stated.

**F1d's comparison.**
- The mean over dates of the per-date summed Brier difference (F1 minus LLM), with an exact per-date sign-flip p (2^8 = 256 draws).
- A failed LLM answer counts as F1's answer, and the failure is counted.

---

## 8. Ledger, pinning, trials

**Declare.** `brain forecast declare --llm-cutoff YYYY-MM` refuses unless the working tree is clean for `brain/` and `docs/MARKET_PRIMER.md`. Without `--llm-cutoff` it declares F1a-F1c and leaves F1d undeclared, saying so. A later declare reuses the anchor and `prices_asof` of the tests already declared, so the family keeps one anchor. It then appends, per undeclared test:
```python
{"op": "declare", "kind": "forecast", "id", "spec_sha", "spec", "at", **ledger.git_state()}
```
**Run.** `brain forecast test ID...` refuses unless:
- the test is declared and its spec sha matches;
- the current `code_sha` equals the declared one, and the tree is clean;
- HEAD is the declare commit or a descendant that leaves those files unchanged;
- for F1d, the live model's name and digest and the primer and prompt shas equal the spec's;
- no result exists yet for that `(id, spec_sha)`. A rerun needs `research withdraw` and a new id.

It then appends:
```python
{"op": "result", "kind": "forecast", "id", "spec_sha", "at", **git_state, "metrics", "p" (gated only), "controls",
 "n_dates", "n_samples", "dropped", "model", "cache_key", "prices_asof"}
```
F1d's result also stores every raw model reply. After the run, RESULTS.md is regenerated.

**verdicts.**
- `research.run.verdicts()` puts every gated forecast result and every hypothesis result into one BH pass: `p` for forecasts, `perm_p` for hypotheses; withdrawn included.
- A gated test passes on its gate, q < 0.10 and its controls.
- Descriptive results get the verdict `REPORT`, outside BH.

**RESULTS.md** gains a "Forecasts" section:
- a row per test (IC, size-neutral IC with its date-level p or interval, q, the baselines' ICs, spread, Brier skill against climatology and its within-date term, placement, range coverage per date with interval, n);
- the calibration table;
- a plain-words line on what the range and the chance showed.

**Trials.** With `--trial`, a non-default `BRAIN_LEDGER`, or a MONEY_HOME other than the default:
- every anchor is replaced by the last session of 2018 (TRAIN), and F1b, F1c and F1d run on TRAIN windows only: F1b's and F1c's windows start at 2015-01-01 (`evaluate.TRIAL_START`) instead of 2019-01-01;
- a run refuses if any grid's last exit would fall after 2018-12-31;
- the ledger written is the trial one: `BRAIN_LEDGER` and `BRAIN_RESULTS` when they point elsewhere, else `<MONEY_HOME>/trial/`; never the committed files;
- a trial is plumbing, not evidence: it may run on an uncommitted tree (the clean-tree and HEAD-lineage refusals become warnings), while the code sha, spec sha, one-result and F1d identity checks still apply.

The held-out windows are computed only by the recorded runs.

---

## 9. Thinking: the primer and the LLM forecast (`docs/MARKET_PRIMER.md`, `brain/forecast/think.py`)

**`docs/MARKET_PRIMER.md`** is plain English, about 2,000 words at most: the knowledge the model is given.
- **How prices form.** Expectations, not results, move prices. Public news is priced quickly. A stock's move is the market's move plus its own.
- **What moves a stock.** Earnings against expectations, guidance, rates, liquidity, flows.
- **Base rates.** About half of stocks beat SPY in any week or month. Typical moves follow from volatility. Confident calls are usually wrong.
- **Known patterns** (momentum, short- and long-term reversal, value, profitability, low volatility, post-earnings drift, 52-week-high anchoring, lottery stocks, insider buying, attention) and the published evidence that they weakened after publication.
- **Human nature.** Overconfidence, herding, loss aversion and the disposition effect, anchoring, recency, narrative, overreaction and underreaction.
- **How to reason.**
  1. Start from the base rate.
  2. Ask what is already priced in.
  3. Move only for evidence with a tested record.
  4. Keep most probabilities between 0.45 and 0.55; 0.60 is a strong view.
- **This bank's own tests.** Only TRAIN-period numbers (2011-2018) from RESULTS.md, or plain words ("every hypothesis failed its TEST gate"), with no figures from later months.
- **What not to do.** Never use knowledge of events after the snapshot date. Never infer anything about private individuals.

The primer cites no market event, price, result or publication dated after 2026-06-30. A test checks that it contains no date in or after the declared windows.

```python
SNAPSHOT_VERSION = 1
def snapshot(bank, book, ticker, d, forecast) -> str
    # at T(d) only; the ticker symbol, no name, since the bank's identity facts are newer than the windows:
    # - returns over 1w/1m/3m/12m, 52w high/low distance and volatility, from the book sliced to <= d;
    # - the latest fiscal year's key figures and the last 4 quarters' EPS and SUE, from F1's month-end view
    #   (the last session before d's month, features.month_end_index);
    # - the last earnings release known at T(d) and its 3-day reaction if closed;
    # - insider buys and sells in 90 days (counts, USD);
    # - F1's forecast (p_beat, the 68% band, the top drivers).
    # No headlines: the bank holds none before 2026-10-09, so the tested input and the served input stay the same.
def llm_identity(client) -> dict      # {"model", "digest"}: Ollama /api/show; a GGUF path's sha256 for llama.cpp
def llm_forecast(bank, book, ticker, d, forecast, client=None, deadline=None) -> dict
    # {"ok", "p_beat", "reason", "raw", "model", "seconds", "error", "down"}; never raises
```
**`llm_forecast`.**
- One request, no tools, `temperature 0`, seed 7, `max_tokens 300`.
- **System message:** the primer, plus the instructions "Give the probability that this stock beats SPY over the next 5 sessions after the snapshot; reply with only a JSON object {"p_beat": 0.0-1.0, "reason": "one or two sentences"}; never use knowledge of events after the snapshot date".
- For the Qwen3 family, the user message ends with `/no_think`, its documented switch.
- `PROMPT` holds the system template, the user template and that rule. `prompt_sha` hashes it.
- **Parsing** follows `judge.assess`'s robust order with one repair. `p_beat` is clipped to [0.01, 0.99].
- **On failure** the sample takes F1's `p_beat`, and `llm_failed` is counted.

**The judge.** `brain/judge.py` includes the primer in its system prompt, so `JUDGE` and `JUDGE_SHA` change and the judge becomes a new source.

---

## 10. Product: live forecasts (`brain/forecast/live.py`, CLI, terminal)

**Model.** `live.model(bank)` trains F1 by F1a's rule exactly: every realised sample since 2011-01-01, `max_rows`, the same calibration and range fit, at cutoff = the latest session. It is cached as JSON under `<home>/models/forecast/<cache key>.json`, and retrained when the newest realised exit is more than 5 sessions past the cached one.

**`brain forecast AAPL [--horizon 5|21] [--llm]`** prints the card:
```
AAPL  after 2026-10-15, next 5 sessions                forecast F1 · paper only · not financial advice
Close 342.00 on 2026-10-15
Likely range   68%: 331 - 353  (-3.2% .. +3.3%)     90%: 323 - 361  (-5.6% .. +5.6%)
Chance it beats SPY: 51%   (a coin is 50%)
Drivers: dist_52w_high +0.08, mom_12_1 +0.05, abn_volume -0.04, ...
Record: last 2 months (F1a): IC +0.01 (90% interval -0.10..+0.12), inconclusive. 2019-2026 (F1b): IC ..., p ..., FAIL.
        Range: 68% band held 66% of the time (interval 58-74%). Live forecasts scored: 0.
```
- It records a `forecast` fact: entity the ticker, attr `forecast`, tier `derived`, value `"p_beat 0.51; 68% 331-353"`, known_at `T(d)`.
  - **Context:** section 6's dict.
  - **Inputs:** the ticker's and SPY's price snapshot refs, plus the fundamental facts read.
- `--llm` adds the model's probability, recorded as a second fact with `context.forecaster "F1-llm"`.

**`brain forecast score`** scores live forecasts once their exit has passed:
- band coverage per date;
- Brier against climatology, with the within-date term;
- placement;
- IC per date.

**Terminal.**
- **`T FCST`:** the bands drawn on a 3-month price chart's canvas, the chance against the coin, the drivers, the F1 tests' lines and live scoring.
- **`T FCST NEW`:** a background job of kind `forecast` (target `FCST T`, argv `["forecast", T]`), with `DEC NEW`'s job rules. Reruns show `T FCST` and never start a job.
- **ASK:** a read-only analyst tool `forecast` maps to `T FCST`.

---

## 11. Tests (`tests/test_forecast_*.py`)

- **Features.**
  - Hand-computed values on a synthetic bank.
  - **The leak test, for every feature.** Appending bars dated after `d` to the same snapshot (with older adjclose re-rounded within 1e-6), or facts known after `T(d)`, changes nothing at `d`.
  - A fact stamped exactly `T(d)` never overlaps its label window.
  - Fundamentals carry forward only.
  - R z-scores use earlier dates only.
  - The cache key changes with the code and with a backfilled shard.
- **Learn.**
  - The GBM fits a nonlinear rule better than logistic.
  - Determinism.
  - A planted sample whose stored exit is `C + 1` is never used at cutoff `C`, including in the calibration slice and the range fit.
  - Range coverage on synthetic data with a known volatility.
- **Evaluate.**
  - The date-level sign-flip p against exact values.
  - Size neutralisation removes a pure-size forecast's IC.
  - The oracle reaches the minimum p; the placebo rejects about 5%.
  - Brier decomposition.
  - Non-overlapping grids; F1a and F1b share no week.
  - Anchors pinned.
- **Ledger.**
  - Refusals: undeclared, spec changed, code sha changed, dirty tree, a second result, an F1d model mismatch.
  - BH includes gated forecasts and excludes descriptive ones.
  - Trials shift anchors to 2018 and never write the real ledger.
- **Think.**
  - The snapshot is point-in-time.
  - The primer has no date in the windows.
  - Parse and fallback.
  - `/no_think` for Qwen3.
  - The digest refusal.
  - The judge carries the primer.
- **Product.** The card, the forecast fact passing `check.run`, live scoring, the terminal view and job, reruns, and the analyst tool.

---

## 12. As built (integration notes)

Where the code settled a point this contract left open, or chose differently, it is listed here. Sections 1-11 above already say what the code does where it changed them (the grid, the range's move, the month-end rule, the fallbacks, `code_sha`, trials).

**Features** (`features.py`).
- `FEATURES` has 49 entries: the 40 named in section 3 plus 9 indicators (`has_earnings_recent`, and `has_<name>` for each F feature). `FEATURES_VERSION` is 1.
- `ma_50`, `ma_200` use the adjusted close; `dist_52w_*` the split-adjusted close; `range_21` split-adjusted high, low and close. Dollar volume is split-adjusted close times volume.
- `rsi_14` is Wilder's over the last 250 changes, seeded by the mean of the first 14. `vix_chg_21` is in points; `spy_above_200` is 0/1; `mkt_vol_21` is the sd of SPY's daily log returns.
- `beta_252` and `idio_vol_63` use simple returns; `vol_*` and `skew_63` log returns (biased Fisher-Pearson skew). A window statistic needs 80% of its values.
- Closes are forward-filled up to 3 sessions on SPY's calendar; high, low and volume are not. The calendar starts 2007-01-01. "Within 63 sessions" counts SPY sessions from the release's reaction day.
- Insider and earnings features read each ticker's facts once, through the view at the last grid date, then keep those known by `T(d)` for each date: the same set as `view(T(d))`, tested equal on every grid date.
- `prices()` drops SPY and ^VIX from the universe and reads the one book in chunks of 64 tickers at the same `asof`. `feature_set` takes `cache`, `workers` (forked processes; `$BRAIN_FORECAST_WORKERS`, else up to 4), `log`, and `px` (a panel already built for the same tickers and `asof`; `evaluate.prepare` builds it once and passes it).
- The cache is an uncompressed npz (no pickle) of about 224 MB for the weekly 2011-2018 universe.
- R z-scores need 52 earlier grid dates. On the monthly grid every R feature is 0 until about 2015, and `vix` (bank history from 2015) until about 2019; on the weekly grid R is 0 through 2011 and `vix` until about 2016. The learner barely sees the regime in TRAIN: this is a known weakness, kept as declared.

**Learner** (`learn.py`, `gbm.py`).
- Logistic: Newton's method with step halving, minimising summed log loss plus `l2/2 * |w|^2`, intercept unpenalised; `iters` is the maximum.
- GBM: no leaf L2, a hessian floor of 1e-6, NaN treated as 0.
- Platt scaling is a two-parameter logistic on the blend's log-odds, fitted on the latest 20% of exits (the fewest latest exit dates holding at least 20% of rows) with both models fitted on the rest; the final models are refitted on all rows and keep that map.
- The stratified draw gives each date the same share (largest remainders, earlier dates first on ties), keys seeded 7.
- `forecast(model, rng, ticker=, d=, h=, close=, sigma_d=, x_row=, entry=, exit=)` returns section 6's dict.

**Evaluate and the ledger** (`evaluate.py`, `forecast_tests.py`, `research/run.py`).
- Placebo: `size-neutral-IC(1 - p) = -size-neutral-IC(p)` on every date, so each draw flips the per-date series by its coins and runs the gate's own test. The oracle must reach `min_p` (1/2^K exact, else 1/20001). Within-date permutation diagnostic: 1,000 draws seeded 7.
- Placement counts "above" when a forecast is above the date's median forecast; stocks exactly at either median are not counted. The spread uses n//5 stocks per tail (n//2 for F1d). Brier climatology is each model's own training base rate. Pinball loss is in log-return units over the 5 quantiles, against a normal band from the 252-session sd of daily log returns. Calibration bins are equal-count.
- Specs also hold `gate_details`, `window` (first, last, last_exit, n_dates, start: from the calendar only) and `trial`. Results also hold `gate_type`, `gate`, `anchor`, `trial`, `plan`, `seconds` and, for F1d, every reply. `metrics.per_date` keeps 10 per-date series at 6 decimals, so a gated result line is 50-100 KB. NaN is written as null.
- `run.verdicts(path)` returns forecast rows too (`kind: "forecast"`; verdict PASS, FAIL, "FAIL (controls)", REPORT or WITHDRAWN); `run.forecast_verdicts`, `run.is_forecast` and `run.is_gated_forecast` go with it. Readers keyed by hypothesis (brief, decide, the terminal's EV lines) leave forecast rows out; RES lists them with their verdict and q. RESULTS.md gains `## Forecasts` after the hypotheses.
- `prompt_sha` is `think.prompt_sha()` (sha of `think.PROMPT` in canonical JSON); `primer_sha` is `think.primer_sha()` (sha of the primer's bytes). The model's identity is `{"model": judge.canonical_model(the id the server lists), "digest": "sha256:..."}`: the GGUF file's sha256 when the server lists a path (llama.cpp), else the weights blob Ollama's `/api/show` names. F1d's test also compares `snapshot_version`.

**Thinking** (`think.py`, `judge.py`).
- The snapshot shows SPY's return beside the stock's over each window, a typical h-session move (daily volatility times sqrt(h)), and F1's drivers with their plain-words descriptions. It shows only ratios, never price levels: the book's closes are on today's split basis, and a level would tell of later splits.
- Seed 7 goes in the request body through `llm.Client.complete(..., seed=7)`; a stand-in client is asked without it.
- `PROMPT` (hashed by `prompt_sha`) holds the system and user templates, the repair text, the Qwen3 `/no_think` rule, temperature 0, seed 7, `max_tokens` 300, the clip, one repair and the default horizon. The system template has a `{primer}` slot: the primer is pinned by `primer_sha`.
- The judge's system prompt is its rules, one line saying the primer's numbers are background and never cited, the primer, then the answer format (JUDGE_SHA `a74a865f`). The judge does not read forecasts: the `forecast` tool, the attribute and FCST are hidden from it, as DECIDE.md keeps the engine's own outputs from it.
- The primer adds about 11.4k characters to the judge's prompt; keep the judge's server at a 16k context (`BRAIN_LLM_CTX`). The forecaster's own prompt is about 3.9k tokens and fits 8k.

**Live** (`live.py`, CLI, terminal).
- The live grid is `evaluate.grid(cal, d, h)` (F1a's grid with `A = d`), plus `d` itself so the feature set's pending rows give the cross-section at `d`. With a cached model, one date is computed (`feature_set([d])`) and the R columns are z-scored over the same grid's earlier dates; a test checks both routes give the same inputs. Live retrains do not use the feature cache: the grid moves every session.
- The model cache key is the sha of `brain/forecast/*.py`, `features.deps_sha`, `FEATURES_VERSION` and names, `LEARN`, `RANGE`, `h` and the universe, not the date. A cached model is retrained when the newest realised exit has moved more than 5 sessions, when it was trained after `d`, when its file is unreadable, or with `--refit`.
- The fact's `known_at` is the later of `T(d)` and its latest input's `known_at`. Inputs also cite ^VIX's snapshot. Its context is section 6's dict plus `forecaster`, `made_at` (each run is its own fact), `close_date`, `sigma`, `naive_sd`, `base_rate`, `range_q`, the training window, `model_key`, `prices_asof` and `in_universe`. `entry` and `exit` are the expected weekdays; scoring reads the real calendar.
- `--llm`: a failed or unusable answer is still recorded as an F1-llm fact, carrying F1's `p_beat` with `llm_failed: true` and the error, as F1d counts it.
- Live scoring counts one forecast per (stock, forecaster, horizon, date), the latest recorded. A window that opens inside an earlier counted one is "overlapping" and a forecast recorded after its entry session's close (20:00 UTC) is "late"; neither is scored. Metrics come from `evaluate.score` without size, so there is no size-neutral IC live; IC needs dates with 3 or more stocks.
- Refusals: SPY and market symbols; a stock not listed at `d` (with the reason); SPY's latest close more than 7 days old; a horizon other than 5 or 21. A ticker outside `us_broad` is still forecast, ranked with the universe, and the card says the model never trained on it.
- The card says "size-neutral IC" (the gated statistic) where section 10's example says "IC", names the range line's test ("Range (F1a): 68% band held 66% of the time (interval 58%-74%)"), and adds the model's answer with `--llm`, the model's sha and training window, an "untested" line until a test has a result, and a closing line: the chance has no tested edge unless the record says PASS.
- The CLI adds `--refit` and `--json`. `brain forecast declare|test` are the declared tests' commands (`forecast_tests.add_cli` on the same parser); `--trial` and `--llm-cutoff` go only with them.
- `FCST` alone lists each stock's latest forecast, live scoring and the declared tests' lines.

**Expected wall-clock time of the recorded runs** (4 CPUs, from the TRAIN-year trials; see the integration report): weekly features over about 820 grid dates about 8 minutes, F1b's 15 fits about 5-6 minutes, F1a under a minute, monthly features about 3-4 minutes, F1c about 4 minutes: about 20-25 minutes for F1a-F1c with about 4-5 GB of memory at peak. F1d adds 96 local-model requests on F1a's model, about 35 seconds each with Qwen3-1.7B on 4 CPU threads (about an hour); an 8B model is several times slower.
