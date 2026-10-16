"""The forecast product (docs/FORECAST.md, section 10): live.model's training rule and JSON cache, the card, the
forecast fact passing check.run, --llm, live scoring, the development data rule and the CLI.

The engine (features, learn, evaluate, think) is written in parallel, so most tests run against ``Stub``: the
contract's interfaces in a few lines (one ranked price feature and one regime feature, a chance that is a fixed
function of the first, fixed range quantiles). ``RealEngineTest`` runs the real engine on a small synthetic bank.
Every test uses a temporary MONEY_HOME and a scratch ledger; nothing reads prices after the synthetic ones."""
import io
import json
import math
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from brain import claims, config, series
from brain.bank import Bank, Fact
from brain.check import run as check_run
from brain.forecast import live
from brain.metrics import PriceBook
from brain.timeutil import today
from tests.util import TempHome, filed

END = pd.Timestamp(today()) - pd.offsets.BDay(1)
DAYS = pd.bdate_range(end=END, periods=600).strftime("%Y-%m-%d").tolist()
UNI = ["AAA", "BBB", "CCC", "DDD", "EEE"]
Z = {"q05": -1.645, "q16": -1.0, "q50": 0.0, "q84": 1.0, "q95": 1.645}


# ---------- synthetic prices ----------
def snap(entity, dates, close, known=None):
    close = np.asarray(close, dtype=float)
    df = pd.DataFrame({"date": dates, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                       "adjclose": close, "volume": np.full(len(dates), 1000)})
    return Fact(entity=entity, attr="series.price_daily", value=series.store(df), as_of=dates[-1],
                known_at=known or dates[-1], tier="market", source={"name": "test", "url": "https://example.test/px"},
                context={"rows": len(dates), "first": dates[0], "last": dates[-1], "splits": []})


def random_paths(n, seed=5):
    rng = np.random.default_rng(seed)
    out = {"SPY": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)),
           "^VIX": 15 + np.abs(np.cumsum(rng.normal(0, 0.2, n)))}
    for k, t in enumerate(UNI):
        out[t] = (40 + 10 * k) * np.cumprod(1 + rng.normal(0.0002, 0.02, n))
    return out


def add_prices(bank, paths=None, upto=None, days=DAYS, only=None):
    paths = paths or random_paths(len(days))
    upto = upto or len(days)
    bank.add_many([snap(t, days[:upto], v[:upto]) for t, v in paths.items() if only is None or t in only])
    return paths


# ---------- the stand-in engine ----------
def rank_unit(x, ok):
    out = np.zeros(len(x))
    ok = ok & np.isfinite(x)
    n = int(ok.sum())
    if n >= 2:
        r = pd.Series(x[ok]).rank(method="average").to_numpy()
        out[ok] = (r - 1) / (n - 1) - 0.5
    return out


class StubPx:
    def __init__(self, bank, tickers, asof):
        self.tickers, self.asof = [str(t).upper() for t in tickers], asof
        self.book = PriceBook(bank, [*self.tickers, "SPY", "^VIX"], asof)
        self.dates = np.asarray(self.book.dates["SPY"]).astype(str)
        self.adj = np.full((len(self.dates), len(self.tickers)), np.nan)
        self.close = self.adj.copy()
        where = pd.Index(self.dates)
        for j, t in enumerate(self.tickers):
            if self.book.has(t):
                pos = where.get_indexer(self.book.dates[t])
                m = pos >= 0
                self.adj[pos[m], j] = np.asarray(self.book.adj[t])[m]
                self.close[pos[m], j] = np.asarray(self.book.close[t])[m]

    def index(self, d):
        i = int(np.searchsorted(self.dates, str(d)[:10]))
        if i >= len(self.dates) or self.dates[i] != str(d)[:10]:
            raise KeyError(d)
        return i

    def listed(self, i):
        return np.isfinite(self.adj[i]) & (np.isfinite(self.adj[:i + 1]).sum(axis=0) >= 252)


class StubModel(SimpleNamespace):
    def card(self):
        return {"id": "F1", "sha": "stub0000", "trained_through": self.trained_through, "rows": self.rows}


class Stub:
    """features, learn and evaluate as live.py calls them (docs/FORECAST.md sections 2-5 and 7)."""
    FEATURES_VERSION = 1
    NAMES = ["r_21", "vix"]
    GROUPS = ["P", "R"]
    REGIME_NAMES = ["vix"]
    LEARN = {"since": "2011-01-01", "max_rows": 150000}
    RANGE = {"quantiles": [0.05, 0.16, 0.5, 0.84, 0.95]}
    QKEYS = list(Z)

    def __init__(self):
        self.fits, self.sets, self.prices_built = [], [], 0

    # features
    def deps_sha(self):
        return "stub"

    def prices(self, bank, tickers, asof):
        self.prices_built += 1
        return StubPx(bank, tickers, asof)

    def regime_features(self, px, i):
        return {"vix": float(i)}

    def zscore_earlier(self, v, min_dates=52):
        out = np.zeros(len(v))
        for g in range(len(v)):
            prev = v[:g]
            if len(prev) >= min_dates and prev.std(ddof=1) > 0:
                out[g] = (v[g] - prev.mean()) / prev.std(ddof=1)
        return out

    def month_end_index(self, dates, i):
        return int(np.searchsorted(dates, str(dates[i])[:7] + "-01", side="left")) - 1

    def feature_set(self, bank, tickers, grid, asof, h, cache=True, log=None, px=None):
        self.sets.append({"grid": [str(g) for g in grid], "asof": asof, "h": int(h), "cache": cache})
        px = px if px is not None else self.prices(bank, tickers, asof)
        gr = sorted({str(g)[:10] for g in grid})
        rows = [px.index(g) for g in gr]
        z = self.zscore_earlier(np.array([float(i) for i in rows]))
        bd = np.asarray(px.book.dates["SPY"]).astype(str)
        S = {k: [] for k in ("X", "y", "exit", "date", "ticker", "excess")}
        P = {k: [] for k in ("X", "date", "ticker", "close")}
        for g, (d, i) in enumerate(zip(gr, rows)):
            ok = px.listed(i)
            r21 = px.adj[i] / px.adj[i - 21] - 1 if i >= 21 else np.full(len(px.tickers), np.nan)
            rk = rank_unit(r21, ok)
            open_window = int(np.searchsorted(bd, d, side="right")) + int(h) >= len(bd)
            for j in np.flatnonzero(ok):
                x, t = [rk[j], z[g]], px.tickers[j]
                if open_window:
                    for k, v in (("X", x), ("date", d), ("ticker", t), ("close", px.close[i, j])):
                        P[k].append(v)
                    continue
                o = claims.outcome(px.book, t, d + "T23:59:59Z", int(h), "SPY")
                if o is None or o["excess"] == 0:
                    continue
                for k, v in (("X", x), ("y", int(o["excess"] > 0)), ("exit", o["exit"]), ("date", d), ("ticker", t),
                             ("excess", o["excess"])):
                    S[k].append(v)
        return SimpleNamespace(
            X=np.array(S["X"], dtype=float).reshape(-1, 2), y=np.array(S["y"], dtype=int),
            exit=np.array(S["exit"], dtype="U10"), date=np.array(S["date"], dtype="U10"),
            ticker=np.array(S["ticker"]), excess=np.array(S["excess"]), dropped={"pending": len(P["date"])},
            pending={"X": np.array(P["X"], dtype=float).reshape(-1, 2), "date": np.array(P["date"], dtype="U10"),
                     "ticker": np.array(P["ticker"]), "close": np.array(P["close"], dtype=float)},
            tickers=list(px.tickers), names=self.NAMES, groups=self.GROUPS)

    # learn
    def fit(self, fs, cutoff):
        self.fits.append(str(cutoff))
        use = np.asarray(fs.exit) <= str(cutoff)
        assert use.all(), "a live model trains on realised samples only"
        return StubModel(base_rate=float(np.mean(fs.y[use])) if use.any() else 0.5, rows=int(use.sum()),
                         trained_through=str(cutoff))

    def fit_range(self, fs, cutoff, px=None):
        return {"q": dict(Z), "h": 5, "n": len(fs.exit), "trained_through": str(cutoff)}

    def sigma(self, px, i):
        return np.full(len(px.tickers), 0.02)

    def forecast(self, model, rng, *, ticker, d, h, close, sigma_d, x_row, entry=None, exit=None, k=5):
        p = 0.5 + 0.2 * float(x_row[0])
        band = {q: close * math.exp(z * sigma_d * math.sqrt(h)) for q, z in rng["q"].items()}
        pct = lambda lo, hi: [round((band[lo] / close - 1) * 100, 2), round((band[hi] / close - 1) * 100, 2)]  # noqa: E731
        drivers = [{"name": "r_21", "group": "P", "push": 0.8 * float(x_row[0])}] if x_row[0] else []
        return {"ticker": ticker, "date": d, "horizon": h, "entry": entry, "exit": exit, "close": close, "p_beat": p,
                "band": band, "move68_pct": pct("q16", "q84"), "move90_pct": pct("q05", "q95"), "drivers": drivers,
                "model": model.card()}

    def to_json(self, m):
        return {"kind": "F1", "base_rate": m.base_rate, "rows": m.rows, "trained_through": m.trained_through}

    def from_json(self, d):
        if d.get("kind") != "F1":
            raise ValueError("not an F1 model")
        return StubModel(base_rate=d["base_rate"], rows=d["rows"], trained_through=d["trained_through"])

    # evaluate
    def grid(self, cal, anchor, h, since="2011-01-01"):
        cal = np.asarray(cal).astype(str)
        a = int(np.searchsorted(cal, anchor))
        return [str(cal[i]) for i in np.arange(a - 1 - h, -1, -h)[::-1] if cal[i] >= since]

    def exit_of(self, cal, d, h):
        cal = np.asarray(cal).astype(str)
        i = int(np.searchsorted(cal, d, side="right")) + h
        return str(cal[i]) if i < len(cal) else None

    def score(self, S, h, **kw):
        from brain.forecast import evaluate
        return evaluate.score(S, h, **kw)


class ForecastHome(TempHome):
    """A temporary home, a scratch ledger and RESULTS.md, and the stand-in engine."""
    STUB = True

    def setUp(self):
        super().setUp()
        home = Path(self._tmp.name)
        self.ledger = home / "ledger.jsonl"
        self.patches = [mock.patch.object(config, "LEDGER", self.ledger),
                        mock.patch.object(config, "RESULTS", home / "RESULTS.md"),
                        mock.patch.object(live, "universe_tickers", lambda: list(UNI))]
        if self.STUB:
            self.stub = Stub()
            self.patches += [mock.patch.object(live, "_features", lambda: self.stub),
                             mock.patch.object(live, "_learn", lambda: self.stub),
                             mock.patch.object(live, "_evaluate", lambda: self.stub)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        super().tearDown()

    def write_ledger(self, recs):
        self.ledger.write_text("".join(json.dumps(r) + "\n" for r in recs))


# ---------- the model ----------
class ModelTest(ForecastHome):
    def setUp(self):
        super().setUp()
        for upto in (len(DAYS) - 6, len(DAYS) - 3, len(DAYS) - 1, len(DAYS)):  # a book at each T(d) these tests use
            add_prices(self.bank, upto=upto)
        self.cal = np.array(DAYS)

    def test_trained_by_f1as_rule_and_cached_as_json(self):
        m = live.model(self.bank, 5, tickers=UNI, cal=self.cal)
        d = DAYS[-1]
        self.assertTrue(m.retrained)
        self.assertEqual(self.stub.fits, [d])  # cutoff: the latest session
        fs_call = self.stub.sets[0]
        self.assertEqual(fs_call["grid"], self.stub.grid(self.cal, d, 5) + [d])  # F1a's grid counted back from d, and d
        self.assertEqual((fs_call["asof"], fs_call["h"], fs_call["cache"]), (d + "T23:59:59Z", 5, False))
        self.assertEqual(m.meta["newest_exit"], d)  # the last grid date's window ends at d
        self.assertEqual(m.path.parent, Path(self._tmp.name) / "models" / "forecast")
        body = json.loads(m.path.read_text())
        self.assertEqual(body["key"], m.key)
        self.assertEqual(body["model"]["trained_through"], d)
        self.assertEqual(body["meta"]["trained_through"], d)
        again = live.model(self.bank, 5, tickers=UNI, cal=self.cal)
        self.assertFalse(again.retrained)
        self.assertEqual(self.stub.fits, [d])  # read from the cache: not trained again
        self.assertEqual(again.model.trained_through, d)

    def test_retrained_once_the_newest_exit_moves_more_than_five_sessions(self):
        live.model(self.bank, 5, tickers=UNI, cal=self.cal[:-6])
        self.assertEqual(self.stub.fits, [DAYS[-7]])
        live.model(self.bank, 5, tickers=UNI, cal=self.cal[:-1])  # 5 sessions later: still fresh
        self.assertEqual(len(self.stub.fits), 1)
        m = live.model(self.bank, 5, tickers=UNI, cal=self.cal)  # 6: retrained
        self.assertEqual(self.stub.fits, [DAYS[-7], DAYS[-1]])
        self.assertTrue(m.retrained)
        live.model(self.bank, 5, tickers=UNI, cal=self.cal[:-3])  # a model that saw sessions after d is not used
        self.assertEqual(self.stub.fits[-1], DAYS[-4])

    def test_the_key_names_the_code_the_spec_h_and_the_universe(self):
        k = live.model_key(UNI, 5)
        self.assertNotEqual(k["sha"], live.model_key(UNI, 21)["sha"])
        self.assertNotEqual(k["sha"], live.model_key(UNI[:-1], 5)["sha"])
        with mock.patch.object(live, "code_sha", lambda: "edited"):
            self.assertNotEqual(k["sha"], live.model_key(UNI, 5)["sha"])
        with mock.patch.object(Stub, "LEARN", {"since": "2012-01-01", "max_rows": 150000}):
            self.assertNotEqual(k["sha"], live.model_key(UNI, 5)["sha"])
        self.assertEqual(k["sha"], live.model_key(list(UNI), 5)["sha"])
        self.assertNotIn(DAYS[-1], json.dumps(k))  # not the date: a cached model serves later days while fresh

    def test_a_damaged_cache_is_retrained(self):
        m = live.model(self.bank, 5, tickers=UNI, cal=self.cal)
        m.path.write_text("{not json")
        live.model(self.bank, 5, tickers=UNI, cal=self.cal)
        self.assertEqual(len(self.stub.fits), 2)
        live.model(self.bank, 5, tickers=UNI, cal=self.cal, refit=True)
        self.assertEqual(len(self.stub.fits), 3)

    def test_a_cached_models_cross_section_is_the_same_as_a_fresh_ones(self):
        fresh = live.model(self.bank, 5, tickers=UNI, cal=self.cal)
        a = live.cross_section(self.bank, fresh, UNI, DAYS[-1], 5, self.cal)
        cached = live.model(self.bank, 5, tickers=UNI, cal=self.cal)
        b = live.cross_section(self.bank, cached, UNI, DAYS[-1], 5, self.cal)
        self.assertEqual(a["ticker"], b["ticker"])
        np.testing.assert_allclose(a["X"], b["X"])  # R z-scored by the same grid's earlier dates
        self.assertNotEqual(float(np.abs(a["X"][:, 1]).max()), 0.0)


# ---------- the card and the fact ----------
class ForecastTest(ForecastHome):
    def setUp(self):
        super().setUp()
        add_prices(self.bank)

    def forecast(self, *tickers, **kw):
        return live.run(self.bank, list(tickers), universe=UNI, **kw)

    def test_the_fact_passes_check_and_cites_its_inputs(self):
        [r] = self.forecast("bbb")
        self.assertTrue(r["ok"], r)
        f = self.bank.get(r["ref"])
        d = DAYS[-1]
        self.assertEqual((f.entity, f.attr, f.tier, f.as_of, f.known_at), ("BBB", "forecast", "derived", d, d + "T23:59:59Z"))
        c = f.context
        for k in ("ticker", "date", "horizon", "entry", "exit", "close", "p_beat", "band", "move68_pct", "move90_pct",
                  "drivers", "model"):
            self.assertIn(k, c)
        self.assertEqual(set(c["band"]), set(Z))
        self.assertEqual((c["forecaster"], c["horizon"], c["date"]), ("F1", 5, d))
        self.assertEqual(f.value, f"p_beat {c['p_beat']:.2f}; 68% {c['band']['q16']:.2f}-{c['band']['q84']:.2f}")
        self.assertEqual(c["entry"], str(np.busday_offset(np.datetime64(d), 1)))  # expected sessions: weekdays
        snaps = {e: PriceBook(self.bank, [e], d + "T23:59:59Z").snapshot[e].ref for e in ("BBB", "SPY", "^VIX")}
        self.assertEqual(set(f.inputs), set(snaps.values()))
        self.assertEqual(f.source, {"name": "brain.forecast", "forecaster": "F1"})
        errors, _ = check_run(Bank())
        self.assertEqual(errors, [])

    def test_fundamental_facts_read_are_cited(self):
        y = int(DAYS[-1][:4]) - 1
        known = f"{y + 1}-02-20"
        assets = filed("AAA", "us-gaap:Assets", 900e6, f"{y}-12-31", known, form="10-K")
        liab = filed("AAA", "us-gaap:Liabilities", 400e6, f"{y}-12-31", known, form="10-K")
        later = filed("AAA", "us-gaap:Liabilities", 999e6, f"{y}-12-31", DAYS[-1] + "T12:00:00Z", form="10-K/A")
        self.bank.add_many([assets, liab, later])
        [r] = self.forecast("AAA")
        f = self.bank.get(r["ref"])
        self.assertIn(assets.ref, f.inputs)
        self.assertIn(liab.ref, f.inputs)
        self.assertNotIn(later.ref, f.inputs)  # known after the month-end the features read
        self.assertEqual(check_run(Bank())[0], [])

    def test_the_card(self):
        [r] = self.forecast("AAA")
        text = live.card(r)
        lines = text.splitlines()
        d = DAYS[-1]
        self.assertTrue(lines[0].startswith(f"AAA  after {d}, next 5 sessions"), lines[0])
        self.assertTrue(lines[0].endswith("forecast F1 · paper only · not financial advice"))
        self.assertTrue(lines[1].startswith("Close ") and lines[1].endswith(f" on {d}"), lines[1])
        b = r["forecast"]["band"]
        self.assertIn(f"Likely range   68%: {b['q16']:.2f} - {b['q84']:.2f}  (", lines[2])
        self.assertIn(f"90%: {b['q05']:.2f} - {b['q95']:.2f}", lines[2])
        self.assertEqual(lines[3], f"Chance it beats SPY: {r['forecast']['p_beat'] * 100:.0f}%   (a coin is 50%)")
        self.assertTrue(lines[4].startswith("Drivers: "), lines[4])
        self.assertIn("Record: last 2 months (F1a): not run yet.", text)
        self.assertIn("        2019-2026 (F1b): not run yet.", text)
        self.assertIn("this forecast is untested", text)
        self.assertIn("Live forecasts scored: 0.", text)
        self.assertIn(f"Recorded [{r['ref'].split('/')[1][:8]}]", text)
        self.assertEqual(r["model"]["trained_through"], d)

    def test_the_record_quotes_the_declared_tests(self):
        def result(i, gate_type, metrics, **kw):
            return {"op": "result", "kind": "forecast", "id": i, "spec_sha": "s", "at": "2026-10-16T00:00:00Z",
                    "gate_type": gate_type, "metrics": metrics, "plan": {"first": "2019-01-04", "last": "2026-08-07",
                                                                   "last_exit": "2026-08-14"}, "trial": False, **kw}
        rng = {"cover68": {"mean": 0.66, "lo90": 0.58, "hi90": 0.74}, "cover90": {"mean": 0.88, "lo90": 0.8, "hi90": 0.95}}
        f1a = result("F1a-w5-recent", "report", {"ic_size_neutral": {"mean": 0.01, "lo90": -0.10, "hi90": 0.12},
                                                 "range": rng, "label": "retest (descriptive): inconclusive; it can "
                                                                        "detect mean IC >= 0.15 at 80% power"})
        f1b = result("F1b-w5-test", "ic", {"ic_size_neutral": {"mean": 0.004}, "range": rng}, p=0.31,
                     gate={"pass": False}, controls={"pass": True})
        self.write_ledger([f1a, f1b])
        rec = live.record_lines(5)
        self.assertEqual(rec["lines"], [
            "last 2 months (F1a): size-neutral IC +0.01 (90% interval -0.10..+0.12), inconclusive",
            "2019-2026 (F1b): size-neutral IC +0.004, p 0.310, FAIL",
            "Range (F1a): 68% band held 66% of the time (interval 58%-74%)"])
        self.assertEqual(rec["tests"], {"F1a-w5-recent": "REPORT", "F1b-w5-test": "FAIL"})
        self.assertEqual(live.record_lines(21)["lines"][0], "2019-2026 (F1c): not run yet")
        with self.ledger.open("a") as fh:
            fh.write(json.dumps({"op": "withdraw", "id": "F1b-w5-test", "reason": "a mistake"}) + "\n")
        self.assertIn("2019-2026 (F1b): withdrawn (a mistake)", live.record_lines(5)["lines"])
        [r] = self.forecast("AAA")
        self.assertIn("Record: last 2 months (F1a): size-neutral IC +0.01", live.card(r))

    def test_refusals(self):
        out = self.forecast("SPY", "^GSPC", "a b")
        self.assertEqual([r["ok"] for r in out], [False, False, False])
        self.assertIn("benchmark", out[0]["error"])
        self.assertIn("market symbol", out[1]["error"])
        self.assertIn("not a ticker", out[2]["error"])
        short = snap("NEWCO", DAYS[-100:], np.linspace(10, 12, 100))
        self.bank.add(short)
        [r] = self.forecast("NEWCO")
        self.assertFalse(r["ok"])
        self.assertIn("NEWCO has 100 sessions of prices", r["error"])
        [r] = self.forecast("ZZZQ")
        self.assertEqual((r["ok"], r["hint"]), (False, "INGEST ZZZQ"))
        self.assertIn("no prices for ZZZQ", r["error"])
        self.assertEqual(self.bank.facts("NEWCO", "forecast"), [])
        with mock.patch.object(live, "today", lambda: str(np.busday_offset(np.datetime64(DAYS[-1]), 8))):
            with self.assertRaises(live.ForecastError) as cm:
                self.forecast("AAA")
        self.assertEqual(cm.exception.hint, "INGEST SPY")
        with self.assertRaises(live.ForecastError):
            self.forecast("AAA", h=10)

    def test_no_spy_no_forecast(self):
        b = Bank(Path(self._tmp.name) / "other")
        add_prices(b, only=["AAA"])
        with self.assertRaises(live.ForecastError) as cm:
            live.run(b, ["AAA"], universe=UNI)
        self.assertEqual(cm.exception.hint, "INGEST SPY")

    def test_a_ticker_outside_the_universe_is_ranked_with_it_and_says_so(self):
        p = random_paths(len(DAYS), seed=9)
        self.bank.add(snap("OUTSIDE", DAYS, p["AAA"]))
        [r] = self.forecast("OUTSIDE")
        self.assertTrue(r["ok"], r)
        self.assertFalse(r["extra"]["in_universe"])
        self.assertIn("not in universe/us_broad.json", live.card(r))

    def test_llm_answer_recorded_as_a_second_fact(self):
        seen = []

        def ask(bank, book, t, d, fc, client=None):
            seen.append((t, d, fc["p_beat"]))
            return {"ok": True, "p_beat": 0.61, "reason": "Momentum is mild.", "raw": '{"p_beat": 0.61}',
                    "model": "qwen3:8b", "seconds": 1.0, "error": None}
        with mock.patch.object(live, "_think", lambda: SimpleNamespace(llm_forecast=ask)):
            [r] = self.forecast("CCC", llm=True)
        f1, f2 = self.bank.get(r["ref"]), self.bank.get(r["llm_ref"])
        self.assertEqual(seen, [("CCC", DAYS[-1], f1.context["p_beat"])])
        self.assertEqual((f2.context["forecaster"], f2.context["p_beat"], f2.context["llm_failed"]), ("F1-llm", 0.61, False))
        self.assertEqual(f2.source["forecaster"], "F1-llm")
        self.assertEqual(f2.context["llm"]["f1_p_beat"], f1.context["p_beat"])
        self.assertIn(f1.ref, f2.inputs)
        self.assertEqual(f2.context["band"], f1.context["band"])
        self.assertTrue(f2.value.startswith("p_beat 0.61; 68% "))
        self.assertIn("The model's chance (F1-llm, qwen3:8b): 61%. Momentum is mild.", live.card(r))
        self.assertEqual(check_run(Bank())[0], [])

    def test_a_failed_llm_answer_counts_as_f1s_and_is_flagged(self):
        for think in (SimpleNamespace(llm_forecast=lambda *a, **k: {"ok": False, "error": "server down", "p_beat": None}),
                      SimpleNamespace(llm_forecast=lambda *a, **k: 1 / 0),
                      SimpleNamespace(llm_forecast=lambda *a, **k: {"ok": True, "p_beat": 7.0})):
            with mock.patch.object(live, "_think", lambda: think):
                [r] = self.forecast("DDD", llm=True)
            f1, f2 = self.bank.get(r["ref"]), self.bank.get(r["llm_ref"])
            self.assertEqual(f2.context["p_beat"], f1.context["p_beat"])
            self.assertTrue(f2.context["llm_failed"])
            self.assertTrue(f2.context["llm"]["error"])
            self.assertIn("gave no usable answer", live.card(r))
        g = live.score(self.bank)["groups"]["F1-llm h=5"]
        self.assertEqual((g["forecasts"], g["llm_failed"]), (1, 1))  # one per stock, date and horizon: the latest


class GuardTest(ForecastHome):
    def test_the_real_bank_waits_for_the_declared_tests(self):
        led = Path(self._tmp.name) / "committed.jsonl"
        real = Path(self._tmp.name) / "var"
        result = lambda i, trial=False: {"op": "result", "kind": "forecast", "id": i, "trial": trial}  # noqa: E731
        with mock.patch.object(live, "REAL_HOME", real), mock.patch.object(live, "COMMITTED_LEDGER", led), \
                mock.patch.dict(os.environ, {"MONEY_HOME": str(real)}):
            self.assertIn("development data rule", live.held_out_guard())
            with self.assertRaises(live.ForecastError) as cm:
                live.run(self.bank, ["AAA"], universe=UNI)
            self.assertIn("F1a-w5-recent, F1b-w5-test, F1c-m21-test have no recorded result", cm.exception.message)
            with self.assertRaises(live.ForecastError):
                live.model(self.bank, 5, tickers=UNI)  # the model alone too
            self.assertEqual(self.stub.fits, [])  # refused before anything was read or trained
            led.write_text("".join(json.dumps(result(i, trial=True)) + "\n" for i in live.GUARD_TESTS))
            self.assertIsNotNone(live.held_out_guard())  # trial results do not count
            led.write_text("".join(json.dumps(result(i)) + "\n" for i in live.GUARD_TESTS[:2]))
            self.assertIn("F1c-m21-test has no recorded result", live.held_out_guard())
            led.write_text("".join(json.dumps(result(i)) + "\n" for i in live.GUARD_TESTS))
            self.assertIsNone(live.held_out_guard())
        self.assertIsNone(live.held_out_guard())  # any other MONEY_HOME: a synthetic bank


# ---------- live scoring ----------
def score_paths(n=len(DAYS)):
    """SPY flat; from session 500 AAA rises 1% a day and BBB falls 1% a day; the rest flat."""
    k = np.arange(n)
    up = np.where(k >= 500, 1.01 ** np.maximum(k - 500, 0), 1.0)
    return {"SPY": np.full(n, 100.0), "^VIX": np.full(n, 15.0), "AAA": 50 * up, "BBB": 60 / up,
            "CCC": np.full(n, 30.0), "DDD": np.full(n, 20.0), "EEE": np.full(n, 25.0)}


def forecast_fact(bank, t, i, p, sigma=0.02, h=5, forecaster="F1", made=None, base_rate=0.5, **extra):
    """A forecast as live.run records it, on session DAYS[i], made at ``made`` (default that evening)."""
    d = DAYS[i]
    book = PriceBook(bank, [t, "SPY"], d + "T23:59:59Z")
    close = float(book.close[t][list(book.dates[t]).index(d)])
    band = {q: close * math.exp(z * sigma * math.sqrt(h)) for q, z in Z.items()}
    sess = live.expected_sessions(d, h)  # weekdays, as live.run writes them (the synthetic calendar has no holidays)
    fc = {"ticker": t, "date": d, "horizon": h, "entry": sess[0], "exit": sess[h], "close": close, "p_beat": p, "band": band,
          "move68_pct": [round((band["q16"] / close - 1) * 100, 2), round((band["q84"] / close - 1) * 100, 2)],
          "move90_pct": [round((band["q05"] / close - 1) * 100, 2), round((band["q95"] / close - 1) * 100, 2)],
          "drivers": [{"name": "mom_12_1", "group": "P", "push": 0.05}],
          "model": {"id": "F1", "sha": "abcd1234abcd1234", "trained_through": d, "rows": 1000}}
    ex = {"close_date": d, "sigma": sigma, "naive_sd": sigma, "base_rate": base_rate, "range_q": dict(Z), **extra}
    with mock.patch.object(live, "now_iso", lambda: made or d + "T22:00:00Z"):
        f = live.fact_for(fc, ex, [book.snapshot[t], book.snapshot["SPY"]], forecaster,
                          llm={"ok": False, "error": "x"} if forecaster == "F1-llm" else None)
    bank.add(f)
    return f


class ScoreTest(ForecastHome):
    def setUp(self):
        super().setUp()
        p = score_paths()
        for i in (510, 512, 530, 540, len(DAYS) - 2):  # the snapshots each forecast read, as daily ingests leave them
            add_prices(self.bank, paths={k: v for k, v in p.items() if k != "CCC" or i <= 541}, upto=i + 1)
        add_prices(self.bank, paths={k: v for k, v in p.items() if k != "CCC"})
        self.bank.add(snap("CCC", DAYS[:542], p["CCC"][:542]))   # no close after session 541
        self.f = {
            "a1": forecast_fact(self.bank, "AAA", 510, 0.6),                 # scored: outside 68%, inside 90%, beat
            "b1": forecast_fact(self.bank, "BBB", 510, 0.4, sigma=0.04),     # scored the same date: inside both, trailed
            "a2": forecast_fact(self.bank, "AAA", 512, 0.6),                 # opens inside a1's window: overlapping
            "a3": forecast_fact(self.bank, "AAA", 530, 0.6, made=DAYS[532] + "T10:00:00Z"),  # recorded late
            "a4": forecast_fact(self.bank, "AAA", 540, 0.7),                 # scored, a second date
            "a5": forecast_fact(self.bank, "AAA", len(DAYS) - 2, 0.5),      # pending: its window is open
            "c1": forecast_fact(self.bank, "CCC", 540, 0.5),                 # unpriced: CCC has no close at the exit
            "l1": forecast_fact(self.bank, "AAA", 510, 0.6, forecaster="F1-llm", llm_failed=True),
        }

    def test_statuses_and_outcomes(self):
        sc = live.score(self.bank)
        rows = {r["ref"]: r for r in sc["rows"]}
        st = {k: rows[f.ref]["status"] for k, f in self.f.items()}
        self.assertEqual(st, {"a1": "scored", "b1": "scored", "a2": "overlapping", "a3": "late", "a4": "scored",
                              "a5": "pending", "c1": "unpriced", "l1": "scored"})
        a1 = rows[self.f["a1"].ref]
        self.assertEqual((a1["entry"], a1["exit"]), (DAYS[511], DAYS[516]))
        self.assertAlmostEqual(a1["close_exit"], 50 * 1.01 ** 16, places=5)
        self.assertEqual((a1["in68"], a1["in90"], a1["beat"]), (False, True, True))
        b1 = rows[self.f["b1"].ref]
        self.assertEqual((b1["in68"], b1["in90"], b1["beat"]), (True, True, False))
        g = sc["groups"]["F1 h=5"]
        self.assertEqual({k: g[k] for k in ("forecasts", "scored", "dates", "overlapping", "late", "pending", "unpriced")},
                         {"forecasts": 7, "scored": 3, "dates": 2, "overlapping": 1, "late": 1, "pending": 1,
                          "unpriced": 1})
        self.assertEqual(sc["groups"]["F1-llm h=5"]["llm_failed"], 1)
        self.assertEqual([r["date"] for r in sc["rows"]], sorted((r["date"] for r in sc["rows"]), reverse=True))

    def test_metrics_per_date_then_over_dates(self):
        m = live.score(self.bank)["groups"]["F1 h=5"]["metrics"]
        self.assertEqual((m["n_dates"], m["n_samples"]), (2, 3))
        self.assertAlmostEqual(m["range"]["cover68"]["mean"], 0.25)   # dates: 1 of 2, then 0 of 1
        self.assertAlmostEqual(m["range"]["cover90"]["mean"], 1.0)
        self.assertAlmostEqual(m["brier"]["brier"], (0.16 + 0.09) / 2)  # per date, then the mean over dates
        self.assertAlmostEqual(m["brier"]["clim"], 0.25)
        self.assertAlmostEqual(m["brier"]["skill"], 0.5)
        self.assertEqual(m["ic"]["k"], 0)  # no date has 3 stocks

    def test_one_forecast_per_stock_date_and_horizon_counts(self):
        again = forecast_fact(self.bank, "AAA", 510, 0.9, made=DAYS[510] + "T23:00:00Z")  # rerun that evening
        sc = live.score(self.bank)
        refs = {r["ref"] for r in sc["rows"]}
        self.assertIn(again.ref, refs)
        self.assertNotIn(self.f["a1"].ref, refs)  # the latest recorded wins
        self.assertEqual(sc["groups"]["F1 h=5"]["forecasts"], 7)

    def test_scorecard_and_card_line(self):
        text = live.scorecard(live.score(self.bank))
        self.assertIn("F1 h=5: 7 forecasts; 3 scored on 2 dates, 1 overlapping, 1 late, 1 pending, 1 unpriced", text)
        self.assertIn("68% band held 25% per date", text)
        self.assertIn("Brier skill +50.0% against the training base rate", text)
        self.assertIn("1 model answers failed (counted as F1's)", text)
        self.assertEqual(live.live_summary(self.bank, 5)["line"],
                         "Live forecasts scored: 3 on 2 dates: 68% band held 25%, Brier skill +50.0% against the base rate.")
        self.assertIn("No forecasts recorded yet", live.scorecard({"groups": {}}))


# ---------- the command line ----------
class CliTest(ForecastHome):
    def setUp(self):
        super().setUp()
        add_prices(self.bank)

    def cli(self, *argv):
        from brain import cli
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_forecast_score_and_errors(self):
        code, out, err = self.cli("forecast", "AAA", "BBB")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.count("forecast F1 · paper only · not financial advice"), 2)
        self.assertEqual(len(Bank().facts("AAA", "forecast")), 1)
        code, out, _ = self.cli("forecast", "AAA", "--horizon", "21", "--json")
        self.assertEqual((code, json.loads(out)[0]["forecast"]["horizon"]), (0, 21))
        code, out, _ = self.cli("forecast", "score")
        self.assertEqual(code, 0)
        self.assertIn("F1 h=5: 2 forecasts", out)
        self.assertIn("F1 h=21: 1 forecast;", out)
        code, out, err = self.cli("forecast", "SPY")
        self.assertEqual(code, 1)
        self.assertIn("benchmark", err)
        for argv in (("forecast",), ("forecast", "AAA", "--trial"), ("forecast", "score", "AAA"),
                     ("forecast", "declare", "--llm"), ("forecast", "score", "--llm")):
            with self.subTest(argv=argv):
                code, _, err = self.cli(*argv)
                self.assertEqual(code, 1)
                self.assertTrue(err.startswith("brain forecast: "), err)

    def test_declare_and_test_go_to_the_declared_tests(self):
        from brain.research import forecast_tests
        seen = []
        with mock.patch.object(forecast_tests, "cmd", lambda a: seen.append((list(a.words), a.trial, a.llm_cutoff)) or 0):
            self.assertEqual(self.cli("forecast", "declare", "--trial", "--llm-cutoff", "2025-01")[0], 0)
            self.assertEqual(self.cli("forecast", "test", "F1a-w5-recent", "--trial")[0], 0)
        self.assertEqual(seen, [(["declare"], True, "2025-01"), (["test", "F1a-w5-recent"], True, None)])

    def test_registered_as_a_plugin(self):
        import argparse
        from brain import cli
        sub = argparse.ArgumentParser().add_subparsers()
        self.assertIn("brain.forecast.live", cli.add_plugins(sub))
        out = io.StringIO()
        with self.assertRaises(SystemExit) as cm, redirect_stdout(out):
            cli.main(["forecast", "-h"])
        self.assertEqual(cm.exception.code, 0)
        for opt in ("--horizon", "--llm", "--llm-cutoff", "--trial"):  # the card's and the declared tests' options
            self.assertIn(opt, out.getvalue())


# ---------- the real engine on a synthetic bank ----------
class RealEngineTest(ForecastHome):
    """features, learn and evaluate as they stand, on eight synthetic stocks over 1,100 sessions."""
    STUB = False

    def test_forecast_card_fact_and_the_cached_models_answer(self):
        days = pd.bdate_range(end=END, periods=1100).strftime("%Y-%m-%d").tolist()
        rng = np.random.default_rng(21)
        uni = [f"S{k}" for k in range(8)]
        facts = [snap("SPY", days, 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(days)))),
                 snap("^VIX", days, 15 + np.abs(np.cumsum(rng.normal(0, 0.2, len(days)))))]
        facts += [snap(t, days, 50 * np.cumprod(1 + rng.normal(0.0002, 0.02, len(days)))) for t in uni]
        self.bank.add_many(facts)
        [r] = live.run(self.bank, ["S3"], universe=uni)
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["model"]["retrained"])
        fc = r["forecast"]
        self.assertTrue(0 < fc["p_beat"] < 1)
        b = fc["band"]
        self.assertTrue(b["q05"] < b["q16"] < b["q50"] < b["q84"] < b["q95"])
        self.assertTrue(b["q16"] < fc["close"] < b["q84"])
        self.assertEqual(check_run(Bank())[0], [])
        [again] = live.run(self.bank, ["S3"], universe=uni)  # the cached model and a one-date cross-section
        self.assertFalse(again["model"]["retrained"])
        self.assertAlmostEqual(again["forecast"]["p_beat"], fc["p_beat"], places=12)
        for q in Z:
            self.assertAlmostEqual(again["forecast"]["band"][q], b[q], places=9)
        self.assertEqual(again["forecast"]["drivers"], fc["drivers"])
        self.assertIn("Chance it beats SPY:", live.card(again))


if __name__ == "__main__":
    import unittest
    unittest.main()
