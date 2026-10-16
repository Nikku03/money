# Forecasts: a weather report for a stock

`brain forecast AAPL` gives a weather-style forecast for one stock over the next 5 or 21 trading sessions:

- **the likely range of its price** at the end of the horizon (a 68% band and a 90% band);
- **the chance it beats SPY** over the horizon;
- **the main drivers** behind that chance;
- **its own record**: how such forecasts did on held-out history and live (calibration, as a weather service reports it).

Two forecasters:

- **F1 (the learner).** A numpy-only model learns from history how much each of about forty variables matters:
  - price patterns;
  - fundamentals;
  - behavioural signals (how people over- and under-react);
  - the market regime.
- **F1-llm ("thinking").** The local open-weight model reads a point-in-time snapshot of the stock, F1's numbers and a primer on how markets and people work (`docs/MARKET_PRIMER.md`), then gives its own probability.

Everything is tested before it is believed. The tests in section 7 are declared in the research ledger and committed **before** they run. They include the last two months, held out. Every test counts in the false-discovery correction with the hypotheses H1-H8b. Forecasts are paper only: nothing here trades.

**Expectation, stated before running.**
- The range forecast should be roughly calibrated: volatility is persistent, so how much a stock moves is predictable.
- The chance-to-beat-SPY forecast will likely be close to a coin. Every single signal tested here so far failed, and public patterns are priced quickly.
- The tests report what happens either way.

---

## 1. Files and owner groups

| File | What | Group |
| --- | --- | --- |
| `brain/forecast/__init__.py`, `brain/forecast/features.py` (new) | point-in-time feature matrices (section 3) | features |
| `brain/forecast/learn.py`, `brain/forecast/gbm.py` (new) | logistic, numpy gradient boosting, calibration, range model (sections 4, 5) | learn |
| `brain/forecast/evaluate.py`, `brain/research/forecast_tests.py` (new); `brain/research/run.py` (verdicts/report integration) | walk-forward runs, metrics, controls, ledger (sections 7, 8) | learn |
| `docs/MARKET_PRIMER.md`, `brain/forecast/think.py` (new); `brain/judge.py` (primer in the judge's prompt) | the LLM's knowledge, its point-in-time snapshot and probability (section 9) | think |
| `brain/forecast/live.py` (new), `brain/cli.py`, `brain/terminal/*`, `docs/TERMINAL.md` | CLI card, the `forecast` fact, live scoring, terminal FCST (section 10) | product |
| `README.md`, `CLAUDE.md`, this file | docs | integration |

Owner rules: each group edits only its files, and codes against this contract for the others. Tests live in `tests/test_forecast_*.py`, one file per group.

---

## 2. Timing, outcome, samples

**Forecast time.**
- A forecast is made at the close of a SPY session `d`. Everything it reads is as the bank stood at `T(d) = d + "T23:59:59Z"`: `bank.view(T(d))` and `PriceBook(..., asof=T(d))`.
- Price features use daily bars dated `<= d` only.

**Outcome.**
- The window runs from the close of `d` to the close of the session `h` SPY sessions later (`h = 5` or `21`).
- A stock's close at a session is its last close on or before it, and is missing when that close is more than 4 days older, as in `claims.outcome`.
- `excess = (stock_exit/stock_entry - 1) - (SPY_exit/SPY_entry - 1)`, with dividend-adjusted closes, and `y = 1` when `excess > 0`.
- A sample with a missing outcome is dropped and counted.

**Universe.**
- `universe/us_broad.json`, minus SPY.
- A stock is listed at `d` when it has a close within 4 days of `d` and at least 252 sessions of history.
- The universe is chosen by today's market value, so it holds survivors. Results report this, as RESULTS.md does.

**Forecast dates.**
- Every `h`-th SPY session, counted back from an anchor session `A`, so the windows never overlap: `d_k = A - k*h` sessions, with `k = 1, 2, ...`.
- Each window `(d_k, d_k + h]` ends where the next begins.
- The anchor of every declared test is pinned in its spec (section 7).

**Training samples.**
- The same grid, extended back to 2011-01-01: weekly for `h = 5`, every 21 sessions for `h = 21`.
- A model trained at cutoff `C` uses only samples whose exit session is `<= C`, so every label it learns from was known at `C`.

---

## 3. Features (`brain/forecast/features.py`)

```python
def price_matrices(bank, tickers, asof) -> Prices      # dates (T,), adj (T,N), close (T,N), high, low, volume (T,N); SPY adj (T,)
def stock_features(px: Prices, i: int) -> dict[str, np.ndarray]      # at calendar index i (= d), each (N,), NaN when undefined
def fundamental_features(bank, tickers, month_ends) -> dict[str, np.ndarray]   # (M, N), each at view(T(month_end))
def regime_features(bank, px: Prices, i: int) -> dict[str, float]
def feature_matrix(bank, tickers, dates, asof) -> FeatureSet   # X (S, F) over (date, ticker) samples, names, groups, keys
FEATURES: list[tuple[str, str, str]]   # (name, group P|F|B|R, plain-words description), the frozen list below
```

**P: price patterns** (from adjusted closes `<= d`):
- `r_5`, `r_21`, `r_63`, `r_126`: past returns over 5, 21, 63 and 126 sessions.
- `mom_12_1`: the return from `d-252` to `d-21`.
- `ma_50`, `ma_200`: close over its 50- and 200-session mean, minus 1.
- `rsi_14`: the 14-session RSI (Wilder).
- `vol_21`, `vol_63`: the standard deviation of daily log returns.
- `vol_ratio`: `vol_21 / vol_63`.
- `beta_252` and `idio_vol_63`: the slope against SPY's daily returns over 252 sessions, and the residual standard deviation over 63.
- `skew_63`: the skewness of daily returns.
- `range_21`: the mean of `(high-low)/close` over 21 sessions.

**B: behavioural** (how people react):
- `dist_52w_high`, `dist_52w_low`: close over the 252-session high (low) minus 1. Anchoring: people judge a price against its recent peak.
- `max_ret_21`: the largest daily return in 21 sessions. Lottery preference: people overpay for stocks that just jumped.
- `min_ret_21`: the largest daily drop in 21 sessions.
- `rev_5`, `rev_21`: minus `r_5` and minus `r_21`. Short-term overreaction that reverses.
- `ltr_3y`: minus the return from `d-756` to `d-252`. Long-term overreaction.
- `abn_volume`: log of the 5-session mean volume over the 63-session mean volume. Attention.
- `turnover_trend`: log of the 21-session mean dollar volume over the 252-session mean.
- `insider_buy_90d`: log1p of the USD of open-market purchases (code P) by officers and directors in Form 4/5 filings known in the 90 days to `d`. `signals.insider_buying`'s rule, at `d`.
- `earnings_car_3d`: the 3-day reaction to the latest earnings release known by `d` whose window closed by `d`, if within 63 sessions, else 0 with `has_earnings_recent = 0`. Underreaction: drift after news.
- `sue`: `metrics.sue` at `view(T(d))`, from the latest month-end view (see F).

**F: fundamentals.** Computed at each month-end view (the last SPY session of each month, `view(T(month_end))`) and carried forward to forecast dates until the next month-end. Never carried back.
- `earnings_yield_ttm` (`metrics.earnings_yield_ttm`).
- `log_mktcap` (`metrics.market_cap_any`, log).
- `sales_growth`: the latest fiscal year's revenue against the year before (`metrics.revenue_growth`).
- `gross_profitability`: gross profit over total assets, latest fiscal year.
- `asset_growth` (`metrics.asset_growth`).
- `accruals`: (net income minus operating cash flow) over total assets, latest fiscal year.
- `leverage`: total liabilities over total assets.
- `sue` (`metrics.sue`).

Each is NaN when the bank lacks it, and has a matching `has_<name>` indicator.

**R: market regime** (the same for every stock at `d`):
- `vix`: ^VIX close at `d`.
- `vix_chg_21`.
- `spy_r_21`, `spy_r_252`, `spy_above_200`.
- `curve_10y_3m`: the 10-year minus 3-month Treasury par yield on the latest curve known at `T(d)` (MACRO).
- `mkt_vol_21`: SPY's 21-session realised volatility.

A regime feature with no data at `d` is NaN.

**Transforms** (in `feature_matrix`, per date):
- P, B and F features are ranked across the listed stocks at `d` into [-0.5, 0.5]. NaN becomes 0, and the `has_` indicators carry the gap.
- R features are z-scored with the mean and standard deviation of their own past values on the forecast grid. They are expanding: only dates `< d`, with at least 52 earlier values (else 0).

Nothing is fitted on future data.

**Caching.** Feature matrices for a (universe sha, grid, asof) may be cached under `<home>/cache/forecast/`. The key includes the bank's price-snapshot ids, so new data invalidates them.

---

## 4. The learner (`brain/forecast/learn.py`, `brain/forecast/gbm.py`)

```python
LEARN = {                       # frozen per model id; part of every declared spec
  "logistic": {"l2": 1.0, "iters": 200},
  "gbm": {"trees": 150, "depth": 3, "lr": 0.05, "bins": 64, "min_leaf": 200, "subsample": 0.8, "colsample": 0.8, "seed": 7},
  "blend": "mean of the two probabilities",
  "calibration": "Platt scaling fitted on the most recent 20% (by date) of the training samples, models refit on the rest first",
  "max_rows": 200000,           # training rows, sampled by date-stratified seeded draw when more
}
def fit(X, y, dates, spec=LEARN) -> Model          # deterministic for a given input and seed
def predict(model, X) -> np.ndarray                 # P(y = 1), in (0, 1)
def drivers(model, x_row, names, k=5) -> list[dict] # [{"name", "group", "push": +/-float}]: logistic coefficient * value
```
- **GBM.** Histogram gradient boosting for log loss, in numpy: quantile bins per feature from the training rows, depth-limited trees from bin histograms (`np.bincount`), shrinkage, row and column subsampling with a seeded `np.random.default_rng`.
- **Hyperparameters.** They are fixed here and frozen in each test's spec. No tuning on any test window.

---

## 5. The range (`learn.py`)

```python
RANGE = {"ewma_lambda": 0.94, "lookback": 252, "quantiles": [0.05, 0.16, 0.5, 0.84, 0.95]}
def sigma(px, i) -> np.ndarray      # (N,) EWMA daily log-return volatility at d
def fit_range(z) -> dict            # empirical quantiles of z = log(1+r_h) / (sigma * sqrt(h)) over training samples
def band(close_d, sigma_d, h, q) -> dict   # {"q05","q16","q50","q84","q95"} prices: close_d * exp(q * sigma_d * sqrt(h))
```
- The 68% band is `[q16, q84]` and the 90% band is `[q05, q95]`.
- The quantiles come from the same training samples as the learner. Fat tails and drift come with them.

---

## 6. A forecast
```python
{"ticker", "date": d, "horizon": h, "close": float, "p_beat": float,
 "band": {"q05", "q16", "q50", "q84", "q95"},          # prices
 "move68_pct": [lo, hi], "move90_pct": [lo, hi],         # the same bands as returns
 "drivers": [{"name", "group", "push"}],                  # top 5 by |push|
 "model": {"id": "F1", "sha", "trained_through": C, "rows": int}}
```

---

## 7. Declared tests (`brain/research/forecast_tests.py`)

```python
FORECAST_TESTS = [
  {"id": "F1a-w5-recent", "kind": "forecast", "forecaster": "F1", "horizon": 5, "anchor": <pinned>, "periods": 8,
   "retrain": "once, at the first forecast date",
   "claim": "Over the last 8 non-overlapping weeks (about two months, held out), F1's chance-to-beat-SPY forecasts rank "
            "stocks by what they then did: mean weekly rank correlation (IC) > 0."},
  {"id": "F1b-w5-test", "kind": "forecast", "forecaster": "F1", "horizon": 5, "anchor": <F1a's first forecast date>,
   "start": "2019-01-01", "retrain": "every 13 periods",
   "claim": "Over every non-overlapping week from 2019 to the start of F1a's window, F1's forecasts have mean weekly IC > 0."},
  {"id": "F1c-m21-test", "kind": "forecast", "forecaster": "F1", "horizon": 21, "anchor": <pinned>, "start": "2019-01-01",
   "retrain": "every 3 periods",
   "claim": "Over every non-overlapping 21-session window from 2019 to the anchor, F1's forecasts have mean IC > 0."},
  {"id": "F1d-llm-w5-recent", "kind": "forecast", "forecaster": "F1-llm", "horizon": 5, "anchor": <F1a's anchor>,
   "periods": 8, "tickers": [12 fixed names, below],
   "claim": "On the same 8 weeks and 12 stocks, the local model's probability, given F1's forecast, a point-in-time "
            "snapshot and the market primer, has a lower Brier score than F1 alone."},
]
```
Every test also carries `"universe": BROAD, "universe_sha": BROAD_SHA`, `"learn": LEARN`, `"range": RANGE`, `"features"` (the FEATURES names) and `"gate"` (below). The spec sha covers all of it.

**Anchors.**
- `brain forecast declare` pins each anchor to the last SPY session in the bank at that moment, for which every universe stock's outcome is known. The anchor is written into the spec and ledger before any run.
- F1a's window is the 8 weeks before its anchor.
- F1b ends where F1a's window starts, so the two never share a week.

**F1d sample.**
- The tickers: AAPL, NVDA, MSFT, AMZN, GOOGL, JPM, XOM, JNJ, PG, WMT, KO, CAT. Each must be in us_broad; `declare` refuses otherwise.
- The model: whatever `BRAIN_LLM_MODEL` names at run time, recorded with the result.
- The run refuses unless the model's training data predates the window. The declared model must state a training cutoff (the `llm_cutoff` spec field, set at declare from a CLI flag) earlier than the window's first date. Otherwise the model may know what happened.

**Metrics** (`brain/forecast/evaluate.py`), per forecast date and pooled:
- **IC.** Spearman rank correlation between `p_beat` and `excess` across the listed stocks, per date. The headline is the mean over dates.
- **Spread.** Top-quintile minus bottom-quintile mean excess per date (F1d: top half minus bottom half).
- **Brier score** of `p_beat` against `y`, against a constant 0.5 (`brier_skill = 1 - brier/0.25`), and log loss.
- **Hit rate** of `p_beat > 0.5` calls.
- **Calibration table:** deciles of `p_beat` (F1d: quintiles), mean forecast and realised frequency, n.
- **Range:** coverage of the 68% and 90% bands, the mean band width, and the pinball loss against a naive band. The naive band is a normal band from the trailing 252-session daily volatility.
- **Baselines** on the same samples: momentum alone (`mom_12_1` rank as `p_beat`'s order) for IC.

**Gates.**
- **F1a, F1b, F1c:** mean IC > 0, and the permutation p < 0.05. The permutation shuffles `p_beat` across stocks within each date, 5000 times, one-sided on the mean IC. In addition, BH q < 0.10 across every hypothesis and forecast test in the ledger.
- **F1d:** the mean Brier difference (F1 minus LLM) > 0. Its p (< 0.05) comes from a paired sign-flip permutation over the (date, ticker) samples, 5000 draws. BH applies as above.
- **Controls, per run:**
  - an oracle (`p_beat` = the realised excess rank) must reach p < 0.001;
  - 200 random forecasts must reach p < 0.05 in 1%-10% of draws;
  - otherwise the result is recorded as FAIL (controls).
- Range coverage is reported and never gated: it makes no claim of an edge.

---

## 8. Ledger and results

- **Commands.**
  - `brain forecast declare` appends `{"op": "declare", "kind": "forecast", "id", "spec_sha", "spec", "at", **git_state}` for each undeclared test, using the same `ledger` functions as hypotheses.
  - `brain forecast test ID...` refuses an undeclared or changed spec and runs the test. It appends `{"op": "result", "kind": "forecast", "id", "spec_sha", "at", **git_state, "metrics": {...}, "p": float, "controls": {...}, "n_dates", "n_samples", "dropped", "model": {...}}`.
  - The command then regenerates RESULTS.md.
  - Withdrawal works as for hypotheses: `brain research withdraw ID --reason`.
- **verdicts.**
  - `research.run.verdicts()` includes forecast results in one Benjamini-Hochberg pass with every hypothesis: p = `perm_p` for hypotheses and `p` for forecasts; withdrawn ones included.
  - Verdict PASS needs all of: its gate (section 7), q < 0.10, and controls passed.
- **RESULTS.md** gains a "Forecasts" section:
  - a table per test (IC, its p, q, spread, Brier skill, hit rate, range coverage 68%/90%, n);
  - the calibration table;
  - a plain-words line on what the range and the chance each showed.
- **Trial runs.** `BRAIN_LEDGER` and `BRAIN_RESULTS` point a trial run at scratch copies, so it never writes the committed ledger.

---

## 9. Thinking: the primer and the LLM forecast (`docs/MARKET_PRIMER.md`, `brain/forecast/think.py`)

**`docs/MARKET_PRIMER.md`.** At most about 2,500 words, plain English, the knowledge the model is given:
- **How prices form.** Expectations, not results, move prices. Public news is priced quickly. A stock's move is the market's move plus its own.
- **What moves a stock.** Earnings against expectations, guidance, rates, liquidity, flows.
- **Base rates.** About half of stocks beat SPY in any week or month. Typical weekly and monthly moves follow from volatility. Confident calls are usually wrong.
- **Known patterns and how they decayed:** momentum, short-term and long-term reversal, value, profitability, low volatility, post-earnings drift, 52-week-high anchoring, lottery stocks, insider buying, attention.
- **Human nature:** overconfidence, herding, loss aversion and the disposition effect, anchoring, recency, narrative, overreaction and underreaction.
- **How to reason.**
  1. Start from the base rate (50%).
  2. Ask what is already priced in.
  3. Move the probability only for evidence with a tested record.
  4. Keep most probabilities between 0.45 and 0.55; 0.60 is a strong view.
- **What this bank's own tests found:** every hypothesis failed, with the RESULTS.md numbers quoted as of writing.
- **What not to do:** never use knowledge of events after the snapshot date; never infer anything about private individuals.

```python
def snapshot(bank, ticker: str, d: str, forecast: dict) -> str
    # point-in-time text at T(d), built from view(T(d)) and PriceBook(asof=T(d)) only:
    # - name and industry;
    # - returns over 1w/1m/3m/12m, 52w high/low distance, volatility;
    # - the latest fiscal year's key figures and the last 4 quarters' EPS and SUE (metrics at the view);
    # - the last earnings release and its 3-day reaction if closed;
    # - insider buys and sells in 90 days (counts, USD);
    # - headlines known in the 7 days to T(d), at most 10 titles;
    # - F1's forecast (p_beat, the 68% band, the top drivers).
def llm_forecast(bank, ticker, d, forecast, client=None, deadline=None) -> dict
    # {"ok", "p_beat": float | None, "reason": str, "raw", "model", "seconds", "error"}; never raises
```
`llm_forecast` makes one request, with no tools:
- **System message:** the primer, plus the instructions "Give the probability that this stock beats SPY over the next 5 sessions; reply with only a JSON object {"p_beat": 0.0-1.0, "reason": "one or two sentences"}; never use knowledge of events after the snapshot date".
- **User message:** the snapshot.

The reply is parsed as `judge.assess` parses, with the same robust order and one repair. `p_beat` is clipped to [0.01, 0.99]. On failure the sample falls back to F1's `p_beat` and is counted as `llm_failed`. The fallback keeps the paired test honest: a failed answer is F1's answer.

**The judge.** `brain/judge.py` includes the primer in the judge's system prompt, so `JUDGE` and `JUDGE_SHA` change and the judge is a new source.

---

## 10. Product: live forecasts (`brain/forecast/live.py`, CLI, terminal)

**Model.** `live.model(bank)` trains F1 on every sample realised by now: the latest 2 years, capped at `max_rows`. It caches the model under `<home>/models/forecast/<sha>.json` (plain JSON, no pickle), keyed by the feature data's snapshot ids, and retrains when the newest realised sample is more than 5 sessions newer than the cached one.

**`brain forecast AAPL [--horizon 5|21] [--llm]`** prints the card:
```
AAPL  next 5 sessions (to about 2026-10-23)          forecast F1 · paper only · not financial advice
Close 342.00 on 2026-10-15
Likely range   68%: 331 - 353  (-3.2% .. +3.3%)     90%: 323 - 361  (-5.6% .. +5.6%)
Chance it beats SPY: 51%   (a coin is 50%)
Drivers: dist_52w_high +0.8, mom_12_1 +0.5, abn_volume -0.4, ...
Record: F1a (last 8 weeks) IC +0.01, p 0.41: FAIL. F1b (2019-2026) ... Live forecasts scored: 0.
```
- It records a `forecast` fact: entity ticker, attr `forecast`, tier `derived`, value `"p_beat 0.51; 68% 331-353"`, known_at `T(d)`.
  - **Context:** the section 6 dict.
  - **Inputs:** the ticker's and SPY's price snapshot refs, plus the fundamentals facts read.
- `--llm` adds the model's probability beside F1's. That is recorded too, as a second fact with `context.forecaster "F1-llm"`.

**`brain forecast score`** scores live forecasts whose horizon has passed:
- coverage of the 68% and 90% bands;
- the Brier score and calibration of `p_beat`;
- IC per date.

The weather service checking itself.

**Terminal.**
- **`T FCST`:** the latest forecast. It shows:
  - the bands as a chart: the 68% and 90% ranges over the horizon, drawn on the price chart's canvas;
  - the chance against the coin;
  - the drivers;
  - the F1 tests' verdicts;
  - live calibration.
- **`T FCST NEW`:** a background job of kind `forecast` (target `FCST T`, argv `["forecast", T]`), with the same job rules as `DEC NEW`. Reruns show `T FCST` and never start a job.
- **ASK:** a read-only analyst tool `forecast` mapping to `T FCST`.

---

## 11. Tests (`tests/test_forecast_*.py`)

- **Features.**
  - Each feature on a synthetic panel against a hand computation.
  - No feature at `d` changes when bars, filings or insider facts after `T(d)` are added: the leak test, run for every feature.
  - Fundamentals are carried forward, not back.
  - The R z-scores use only earlier dates.
- **Learn.**
  - GBM fits a known nonlinear rule better than logistic.
  - Determinism under a seed.
  - Calibration on a held-out slice.
  - Training never uses a sample whose exit is after the cutoff (a sample whose label is planted to leak is ignored).
- **Range.** Coverage on synthetic data with a known volatility is about 68% and 90%.
- **Evaluate.**
  - IC and the permutation p on known cases.
  - The oracle passes and random forecasts fail.
  - Non-overlapping dates.
  - F1a and F1b share no week.
  - The anchors are pinned.
  - An undeclared or changed spec is refused.
  - BH includes forecasts.
  - RESULTS.md gets the section.
  - `BRAIN_LEDGER` keeps a trial off the real ledger.
- **Think.**
  - The snapshot is point-in-time (facts after `T(d)` never appear).
  - Parse and fallback.
  - The model-cutoff refusal.
  - The judge carries the primer.
- **Product.** The card, the forecast fact passing `check.run`, live scoring, the terminal view and job, reruns, and the analyst tool.
