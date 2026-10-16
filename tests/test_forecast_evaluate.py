"""Walk-forward metrics, the gate and its controls, and the forecast ledger (docs/FORECAST.md, sections 7, 8
and 11: Evaluate, Ledger). features.py and think.py are stubbed: these tests code against the contract."""
import argparse
import contextlib
import dataclasses
import itertools
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from brain import config, universe
from brain.forecast import evaluate, learn
from brain.research import forecast_tests as ft
from brain.research import ledger, run
from tests.util import TempHome, price_snapshot

SMALL = {**learn.LEARN, "gbm": {**learn.LEARN["gbm"], "trees": 8, "min_leaf": 40}}


# ---------- a synthetic market and stand-ins for features.py and think.py ----------
F1D = ft.BY_ID["F1d-llm-w5-recent"]["tickers"]


def market(n=24, start="2009-01-01", end="2019-06-28", seed=1):
    """F1d's 12 stocks and n - 12 others, with their own volatilities and betas on one market."""
    days = pd.bdate_range(start, end).strftime("%Y-%m-%d").to_numpy()
    T = len(days)
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.01, 0.03, n)
    m = rng.normal(0.0003, 0.008, T)
    r = m[:, None] * rng.uniform(0.7, 1.3, n) + rng.normal(0, 1, (T, n)) * vol
    adj = 50 * np.exp(np.cumsum(r, axis=0))
    return SimpleNamespace(tickers=[*F1D, *(f"S{j:02d}" for j in range(n - len(F1D)))], dates=days, adj=adj,
                           close=adj.copy(),
                           spy=100 * np.exp(np.cumsum(m)), size=rng.normal(size=n))


def _rank_unit(x):
    out = np.zeros(len(x))
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    out[:] = r / (len(x) - 1) - 0.5
    return out


class StubFeatures:
    """features.py as the contract writes it, on a synthetic market. ``planted`` mixes the window's own excess
    into one feature, so a learner can find it. Every grid it is asked for is kept in ``calls``."""
    FEATURES_VERSION = 1
    NAMES = ("r_5", "mom_12_1", "log_mktcap", "planted", "has_sue")
    FEATURES = tuple((n, g, "") for n, g in zip(NAMES, ["P", "P", "F", "B", "F"]))

    def __init__(self, px, planted=0.0):
        self.px, self.planted, self.calls = px, planted, []

    def prices(self, bank, tickers, prices_asof, **kw):
        return self.px

    def feature_set(self, bank, tickers, grid, prices_asof, h, log=None, **kw):
        self.calls.append((list(grid), int(h)))
        px, rng = self.px, np.random.default_rng(2)
        pos = {d: i for i, d in enumerate(px.dates)}
        rows = []
        for d in grid:
            i = pos[d]
            if i < 252 or i + 1 + h >= len(px.dates):
                continue
            e0, e1 = i + 1, i + 1 + h
            exc = px.adj[e1] / px.adj[e0] - 1 - (px.spy[e1] / px.spy[e0] - 1)
            X = np.column_stack([_rank_unit(px.adj[i] / px.adj[i - 5]), _rank_unit(px.adj[i - 21] / px.adj[i - 252]),
                                 _rank_unit(px.size), _rank_unit(self.planted * exc + rng.normal(size=len(exc)) * 0.01),
                                 (rng.random(len(exc)) < 0.5).astype(float)])
            for j in range(len(exc)):
                rows.append((X[j], exc[j], px.adj[e1, j] / px.adj[e0, j] - 1, px.close[i, j], px.dates[e0], px.dates[e1],
                             d, px.tickers[j]))
        cols = list(zip(*rows)) if rows else [[]] * 8
        return SimpleNamespace(
            X=np.asarray(cols[0], dtype=float).reshape(-1, len(self.NAMES)), excess=np.asarray(cols[1], dtype=float),
            y=(np.asarray(cols[1]) > 0).astype(np.int8), r_total=np.asarray(cols[2], dtype=float),
            close=np.asarray(cols[3], dtype=float), entry=np.asarray(cols[4]), exit=np.asarray(cols[5]),
            date=np.asarray(cols[6]), ticker=np.asarray(cols[7]), names=list(self.NAMES),
            groups=[g for _, g, _ in self.FEATURES], dropped={"unlisted": 0, "no_outcome": 0, "zero_excess": 0,
                                                             "pending": 0},
            key={"sha": "stub", "n_grid": len(grid)}, h=int(h), tickers=list(px.tickers))


class StubThink:
    PROMPT = (("system", "primer + rules"), ("user", "snapshot"))
    SNAPSHOT_VERSION = 1

    def __init__(self, digest="sha256:aaa", fail_every=4):
        self.digest, self.fail_every, self.asked, self.no_think = digest, fail_every, [], []

    @staticmethod
    def is_qwen3(model):
        return str(model).startswith("qwen3")

    def llm_identity(self, client):
        return {"model": "qwen3:8b", "digest": self.digest}

    def primer_sha(self):
        return "primer0"

    def llm_forecast(self, bank, book, ticker, d, forecast, client=None, deadline=None, no_think=None):
        self.asked.append((d, ticker, forecast["p_beat"]))
        self.no_think.append(no_think)
        if len(self.asked) % self.fail_every == 0:
            return {"ok": False, "p_beat": None, "reason": None, "raw": "not json", "model": "qwen3:8b",
                    "seconds": 0.1, "error": "no JSON object"}
        return {"ok": True, "p_beat": 0.55, "reason": "base rate", "raw": '{"p_beat": 0.55}', "model": "qwen3:8b",
                "seconds": 0.1, "error": None}


# ---------- per-date and date-level statistics ----------
class SignFlipTest(unittest.TestCase):
    def brute(self, x):
        x = np.asarray(x, dtype=float)
        means = [np.mean(np.array(s) * x) for s in itertools.product([1, -1], repeat=len(x))]
        return np.mean(np.array(means) >= x.mean() - 1e-12)

    def test_exact_p_matches_every_sign_pattern(self):
        self.assertAlmostEqual(evaluate.sign_flip([1, 2, 3])["p"], 1 / 8)
        self.assertAlmostEqual(evaluate.sign_flip([1, -2, 3])["p"], 3 / 8)     # sums 6, 4, 2 reach 2
        rng = np.random.default_rng(0)
        for k in (5, 9, 12):
            x = rng.normal(0.02, 0.1, k)
            out = evaluate.sign_flip(x)
            self.assertTrue(out["exact"])
            self.assertAlmostEqual(out["p"], self.brute(x))
            self.assertAlmostEqual(out["min_p"], 1 / 2 ** k)
        batch = evaluate.sign_flip(np.vstack([[1, 2, 3], [1, -2, 3]]))
        self.assertEqual(batch["p"], [1 / 8, 3 / 8])

    def test_more_than_16_dates_use_20000_seeded_draws(self):
        x = np.full(40, 0.05)
        out = evaluate.sign_flip(x)
        self.assertFalse(out["exact"])
        self.assertEqual(out["p"], 1 / 20001)
        self.assertEqual(out["min_p"], 1 / 20001)
        y = np.random.default_rng(3).normal(0, 0.1, 40)
        self.assertEqual(evaluate.sign_flip(y)["p"], evaluate.sign_flip(y)["p"])   # seeded 7
        self.assertGreater(evaluate.sign_flip(-x)["p"], 0.99)


class RankStatisticsTest(unittest.TestCase):
    def test_ranks_share_ties(self):
        np.testing.assert_array_equal(evaluate.rank([3, 1, 3, 2]), [3.5, 1, 3.5, 2])
        self.assertAlmostEqual(evaluate.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(evaluate.spearman([1, 1, 1], [1, 2, 3]), 0.0)          # a flat forecast ranks nothing

    def test_size_neutralisation_removes_a_pure_size_forecast(self):
        rng = np.random.default_rng(1)
        size = rng.normal(size=300)
        excess = -0.5 * size + rng.normal(size=300)        # survivors: small then, rose later
        p = 1 / (1 + np.exp(size))                          # "small beats": a function of size alone
        self.assertGreater(evaluate.spearman(p, excess), 0.3)
        self.assertAlmostEqual(evaluate.size_neutral_ic(p, excess, size), 0.0)
        info = rng.normal(size=300)
        excess2 = excess + info
        self.assertGreater(evaluate.size_neutral_ic(info + 0.01 * size, excess2, size), 0.3)
        # flipping the forecast flips its size-neutral IC exactly (what the placebo relies on)
        q = rng.random(300)
        self.assertAlmostEqual(evaluate.size_neutral_ic(1 - q, excess2, size), -evaluate.size_neutral_ic(q, excess2, size))

    def test_placement_spread_and_calibration(self):
        p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.5, 0.55])
        e = np.array([-3, -2, -1, 1, 2, 3, 4, 5, 0, 0.5])
        # medians 0.525 and 0.75: (0.4, 1) and (0.55, 0.5) are on the wrong sides, the other eight right
        self.assertAlmostEqual(evaluate.placement(p, e), 0.8)
        self.assertEqual(evaluate.placement(p, 10 * (p - 0.5)), 1.0)
        self.assertAlmostEqual(evaluate.spread(p, e), np.mean([4, 5]) - np.mean([-3, -2]))
        self.assertAlmostEqual(evaluate.spread(p, e, halves=True), np.mean([2, 3, 4, 5, 0.5]) - np.mean([-3, -2, -1, 1, 0]))
        t = evaluate.calibration_table(p, (e > 0).astype(float), groups=5)
        self.assertEqual([c["n"] for c in t], [2] * 5)
        self.assertEqual(t[0]["freq"], 0.0)
        self.assertEqual(t[-1]["freq"], 1.0)

    def test_brier_splits_into_level_and_within_date(self):
        rng = np.random.default_rng(2)
        for _ in range(5):
            p, y = rng.random(50), (rng.random(50) < 0.3).astype(float)
            b = evaluate.brier_parts(p, y, 0.5)
            self.assertAlmostEqual(b["level"] + b["within"], b["brier"])
            self.assertAlmostEqual(b["clim_level"] + b["clim_within"], b["clim"])
            self.assertAlmostEqual(b["clim_within"], y.mean() * (1 - y.mean()))

    def test_interval_and_label(self):
        iv = evaluate.interval([0.1, 0.2, 0.3, None, float("nan")])
        self.assertEqual(iv["k"], 3)
        self.assertAlmostEqual(iv["lo90"], 0.2 - 1.6448536 * 0.1 / np.sqrt(3), places=6)
        self.assertIn("positive", evaluate.label(iv, 0.15))
        self.assertIn("mean IC >= 0.15", evaluate.label(iv, 0.15))
        self.assertIn("inconclusive", evaluate.label(evaluate.interval([-0.1, 0.1, 0.05])))
        self.assertIn("negative", evaluate.label(evaluate.interval([-0.1, -0.2, -0.3])))


def samples(K=60, n=50, seed=0, signal=0.0, h=5):
    """Forecast samples as walk() returns them: K dates, n stocks, a known volatility for the range."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2015-01-02", periods=K, freq="W-FRI").strftime("%Y-%m-%d").to_numpy(), n)
    S = K * n
    market = np.repeat(rng.normal(0, 0.05, K), n)          # a common move every date
    size = rng.normal(size=S)
    info = rng.normal(size=S)
    excess = 0.01 * (signal * info + rng.normal(size=S))
    sig = np.full(S, 0.02)
    u = market + rng.normal(size=S) * sig * np.sqrt(h + 1)  # from the close of d to the exit: h + 1 daily returns
    # the band as fit_range learns it on that move: normal quantiles, quoted with sqrt(h)
    zq = np.tile([evaluate.NORMAL_Q[k] * np.sqrt((h + 1) / h) for k in evaluate.QKEYS], (S, 1))
    p = 0.5 + 0.05 * np.tanh(info + rng.normal(size=S))
    return {"date": dates, "ticker": np.tile([f"T{j}" for j in range(n)], K), "p": p, "y": (excess > 0).astype(float),
            "excess": excess, "size": size, "mom": rng.normal(size=S), "clim": np.full(S, 0.5), "u": u, "sig": sig,
            "naive": sig, "zq": zq}


class ScoreAndControlsTest(unittest.TestCase):
    def test_oracle_reaches_the_minimum_p_and_placebo_rejects_about_five_percent(self):
        m = evaluate.score(samples(K=60), 5, gated=True)
        gate, ctl = evaluate.gate_and_controls(m)
        self.assertTrue(ctl["oracle_ok"])
        self.assertEqual(ctl["oracle_p"], ctl["min_p"])
        self.assertEqual(ctl["min_p"], 1 / 20001)
        self.assertTrue(0.01 <= ctl["placebo_reject_rate"] <= 0.10)
        self.assertTrue(ctl["pass"])
        self.assertFalse(gate["pass"])                         # no signal: the gate stays shut
        # over many more draws the placebo rejects at the test's level
        _, many = evaluate.gate_and_controls(evaluate.score(samples(K=60), 5, gated=True),
                                             {**evaluate.PLACEBO, "draws": 3000})
        self.assertAlmostEqual(many["placebo_reject_rate"], 0.05, delta=0.015)
        # exact branch: 10 dates, the oracle reaches 1/1024
        _, ctl10 = evaluate.gate_and_controls(evaluate.score(samples(K=10), 5, gated=True))
        self.assertEqual(ctl10["oracle_p"], 1 / 1024)
        self.assertTrue(ctl10["oracle_ok"])

    def test_the_placebo_checks_the_test_and_cannot_see_a_leak(self):
        # A random sign pattern gives p < 0.05 about 5% of the time for any series: the placebo passes for pure
        # noise and for a look-ahead leak alike (every date +0.3). It checks the test's arithmetic only.
        rng = np.random.default_rng(5)
        for series in (rng.normal(0, 0.18, 192), np.full(192, 0.3) + rng.normal(0, 0.01, 192)):
            _, ctl = evaluate.gate_and_controls({"ic_size_neutral": {"mean": float(series.mean()), "p": 0.5},
                                                 "_sn": list(series), "_oracle": list(np.abs(series) + 1)})
            self.assertTrue(ctl["placebo_ok"])
            self.assertAlmostEqual(ctl["placebo_reject_rate"], 0.05, delta=0.035)
        self.assertIn("not of the forecaster", ft.GATE_DETAILS["ic"]["controls"]["placebo"])

    def test_a_planted_signal_passes_the_gate(self):
        S = samples(K=40, signal=0.5)
        S["p"] = 0.5 + 0.05 * np.tanh(S["excess"] * 50 + np.random.default_rng(1).normal(size=len(S["p"])))
        m = evaluate.score(S, 5, gated=True)
        gate, ctl = evaluate.gate_and_controls(m)
        self.assertTrue(gate["pass"])
        self.assertTrue(ctl["pass"])
        self.assertLess(m["ic_size_neutral"]["within_date_perm_p"], 0.01)
        self.assertGreater(m["ic_size_neutral"]["mean"], 0.2)

    def test_the_naive_band_spans_the_h_plus_1_returns_it_is_scored_on(self):
        # u runs from the close of d to the close of the exit: h + 1 daily returns. With iid normal returns of a
        # known sd, F1's learned band (normal quantiles over h + 1 returns, quoted with sqrt(h)) and the naive
        # normal band of the same sd are the same band, so neither beats the other.
        h, K, n, sd = 5, 40, 300, 0.02
        rng = np.random.default_rng(11)
        S = K * n
        u = rng.normal(0, sd, (S, h + 1)).sum(axis=1)
        zq = np.tile([evaluate.NORMAL_Q[k] * np.sqrt((h + 1) / h) for k in evaluate.QKEYS], (S, 1))
        dates = np.repeat(pd.bdate_range("2015-01-02", periods=K, freq="W-FRI").strftime("%Y-%m-%d").to_numpy(), n)
        excess = rng.normal(size=S)
        Sx = {"date": dates, "p": np.full(S, 0.5), "y": (excess > 0).astype(float), "excess": excess,
              "size": rng.normal(size=S), "mom": rng.normal(size=S), "clim": np.full(S, 0.5), "u": u,
              "sig": np.full(S, sd), "naive": np.full(S, sd), "zq": zq}
        rg = evaluate.score(Sx, h, gated=False)["range"]
        self.assertAlmostEqual(rg["pinball_skill"], 0.0, places=9)
        self.assertAlmostEqual(rg["cover68"]["mean"], 0.68, delta=0.01)

    def test_range_coverage_is_per_date_with_a_date_level_interval(self):
        m = evaluate.score(samples(K=80), 5, gated=False)
        c68 = m["range"]["cover68"]
        self.assertEqual(c68["k"], 80)
        per = m["per_date"]["cover68"]
        self.assertAlmostEqual(c68["mean"], np.mean(per))
        self.assertAlmostEqual(c68["hi90"] - c68["mean"], 1.6448536 * np.std(per, ddof=1) / np.sqrt(80), places=6)
        # the common move pushes whole dates out of their bands: per-date coverage swings far more widely
        # than 50 independent stocks would
        self.assertGreater(np.std(per), 2 * np.sqrt(0.68 * 0.32 / 50))
        self.assertLess(m["range"]["cover68"]["lo90"], 0.68)
        self.assertIsNotNone(m["range"]["pinball_skill"])
        self.assertAlmostEqual(m["range"]["pinball_skill"], 0.0)   # same normal band as the naive one
        self.assertIn("label", m)
        b = m["brier"]
        self.assertAlmostEqual(b["level"] + b["within"], b["brier"])
        self.assertAlmostEqual(b["skill"], 1 - b["brier"] / b["clim"])
        self.assertEqual(len(m["calibration"]), 10)
        for k in ("size", "momentum"):
            self.assertIn("ic_size_neutral", m["baselines"][k])
        self.assertAlmostEqual(m["baselines"]["size"]["ic_size_neutral"]["mean"], 0.0)


class GridTest(unittest.TestCase):
    cal = pd.bdate_range("2009-01-01", "2019-06-28").strftime("%Y-%m-%d").to_numpy()

    def windows(self, dates, h):
        """Each date's window as the sessions after its entry up to its exit (the return's days)."""
        out = []
        for d in dates:
            i = int(np.searchsorted(self.cal, d, side="right"))
            out.append(set(self.cal[i + 1: i + 1 + h]))
        return out

    def test_grid_windows_never_overlap_and_end_at_the_anchor(self):
        A = "2019-06-14"
        for h in (5, 21):
            g = evaluate.grid(self.cal, A, h)
            self.assertEqual(evaluate.exit_of(self.cal, g[-1], h), A)
            self.assertGreaterEqual(g[0], "2011-01-01")
            w = self.windows(g, h)
            for a, b in itertools.pairwise(w):
                self.assertFalse(a & b)
            for d0, d1 in itertools.pairwise(g):
                i0 = int(np.searchsorted(self.cal, d0))
                self.assertEqual(self.cal[i0 + h], d1)          # every h-th session
                self.assertEqual(evaluate.exit_of(self.cal, d0, h), self.cal[int(np.searchsorted(self.cal, d1)) + 1])
        with self.assertRaises(ValueError):
            evaluate.grid(self.cal, "2019-06-15", 5)              # a Saturday is no anchor

    def test_recent_and_test_windows_share_no_week(self):
        A = "2019-06-14"
        a = evaluate.plan(ft.BY_ID["F1a-w5-recent"], self.cal, A)
        b = evaluate.plan(ft.BY_ID["F1b-w5-test"], self.cal, A)
        c = evaluate.plan(ft.BY_ID["F1c-m21-test"], self.cal, A)
        self.assertEqual(a["n_dates"], 8)
        self.assertEqual(a["last_exit"], A)
        self.assertEqual(a["retrain"], [a["first"]])            # once, at d_8
        self.assertTrue(set().union(*self.windows(a["dates"], 5)).isdisjoint(set().union(*self.windows(b["dates"], 5))))
        self.assertLess(b["last"], a["first"])
        self.assertLessEqual(b["last_exit"], self.cal[int(np.searchsorted(self.cal, a["first"])) + 1])
        self.assertGreaterEqual(b["first"], "2019-01-01")
        self.assertEqual(b["retrain"], b["dates"][::26])
        self.assertEqual(c["retrain"], c["dates"][::6])
        self.assertEqual(c["last_exit"], A)
        t = evaluate.plan(ft.BY_ID["F1b-w5-test"], self.cal, evaluate.trial_anchor(self.cal), trial=True)
        self.assertEqual(evaluate.trial_anchor(self.cal), "2018-12-31")
        self.assertGreaterEqual(t["first"], evaluate.TRIAL_START)
        self.assertLessEqual(t["last_exit"], "2018-12-31")


class AnchorTest(TempHome):
    def test_anchor_is_the_last_session_every_listed_stock_has(self):
        from brain.metrics import PriceBook
        days = pd.bdate_range("2017-01-02", "2019-03-29").strftime("%Y-%m-%d").tolist()
        self.bank.add(price_snapshot("SPY", days, list(np.linspace(100, 120, len(days)))))
        self.bank.add(price_snapshot("AAA", days, list(np.linspace(10, 12, len(days)))))
        self.bank.add(price_snapshot("BBB", days[:-2], list(np.linspace(10, 12, len(days) - 2))))  # two bars short
        self.bank.add(price_snapshot("NEW", days[-100:], list(np.linspace(10, 12, 100))))           # not listed yet
        book = PriceBook(self.bank, ["SPY", "AAA", "BBB", "NEW"])
        self.assertEqual(evaluate.find_anchor(book, ["AAA", "NEW"]), days[-1])
        self.assertEqual(evaluate.find_anchor(book, ["AAA", "BBB", "NEW"]), days[-3])


def later(t, seconds=60):
    """A retrieval time ``seconds`` after ``t``: a fetch made after the declaration."""
    return (pd.Timestamp(t) + pd.Timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


class PinnedPricesTest(TempHome):
    """A price snapshot's known_at is its last bar's date, not its fetch time: a book pinned at prices_asof
    (``retrieved_by``) reads only the snapshots fetched by then."""

    def fetched(self, t, days, close, at):
        return dataclasses.replace(price_snapshot(t, days, close), retrieved_at=at)

    def test_a_pinned_book_never_reads_a_later_fetch(self):
        from brain.forecast import features
        from brain.metrics import PriceBook
        days = pd.bdate_range("2017-01-02", "2018-12-14").strftime("%Y-%m-%d").tolist()
        line = list(np.linspace(100, 110, len(days)))
        self.bank.add_many([self.fetched("SPY", days, line, "2018-12-14T22:00:00Z"),
                            self.fetched("AAA", days, list(np.linspace(10, 12, len(days))), "2018-12-14T22:00:00Z")])
        asof = "2018-12-17T15:00:00Z"                                   # the declaration: prices_asof = now
        at_declare = PriceBook(self.bank, ["SPY"], asof, retrieved_by=asof)
        key0 = features.cache_key(self.bank, ["AAA", "HALT"], days[-30:], asof, 5, retrieved_by=asof)
        # later that day a re-fetch with the same last bar and revised adjusted closes; a stock first fetched a week on
        self.bank.add_many([self.fetched("SPY", days, [x * 0.99 for x in line], "2018-12-17T17:00:00Z"),
                            self.fetched("HALT", days[:-10], [5.0] * (len(days) - 10), "2018-12-24T12:00:00Z")])
        self.assertAlmostEqual(PriceBook(self.bank, ["SPY"], asof).adj["SPY"][-1], 108.9)   # unpinned: reads it
        book = PriceBook(self.bank, ["SPY", "HALT"], asof, retrieved_by=asof)
        self.assertEqual(book.snapshot["SPY"].id, at_declare.snapshot["SPY"].id)
        self.assertAlmostEqual(book.adj["SPY"][-1], 110.0)
        self.assertFalse(book.has("HALT"))
        px = features.prices(self.bank, ["AAA", "HALT"], asof, retrieved_by=asof)
        self.assertEqual(px.retrieved_by, asof)
        self.assertEqual(px.snapshots()["SPY"], at_declare.snapshot["SPY"].id)
        self.assertNotIn("HALT", px.snapshots())
        key1 = features.cache_key(self.bank, ["AAA", "HALT"], days[-30:], asof, 5, retrieved_by=asof)
        self.assertEqual(key1["snapshots_sha"], key0["snapshots_sha"])
        self.assertEqual(key1["snapshots_sha"],
                         features.snapshots_sha(self.bank, ["AAA", "HALT"], asof, retrieved_by=asof))
        self.assertNotEqual(features.cache_key(self.bank, ["AAA", "HALT"], days[-30:], asof, 5)["snapshots_sha"],
                            key0["snapshots_sha"])                      # unpinned, the re-fetch is read


# ---------- the ledger: declare, test, refusals, trials, verdicts ----------
class Pipeline(TempHome):
    def setUp(self):
        super().setUp()
        self.home = Path(self._tmp.name)
        self.px = market()
        days = list(self.px.dates)
        self.bank.add(price_snapshot("SPY", days, list(self.px.spy)))
        for j, t in enumerate(self.px.tickers):
            self.bank.add(price_snapshot(t, days, list(self.px.adj[:, j])))
        self.uni = self.home / "us_broad.json"
        self.uni.write_text(json.dumps({"tickers": [*self.px.tickers, "SPY"], "rule_short": "synthetic"}))
        self.features = StubFeatures(self.px)
        self.think = StubThink()
        self.stack = contextlib.ExitStack()
        for target, attr, new in [(universe, "BROAD", self.uni), (evaluate, "_features", lambda: self.features),
                                  (learn, "LEARN", SMALL), (ft, "_think", lambda: self.think),
                                  (evaluate, "_think", lambda: self.think), (ft, "_client", lambda c=None: "client")]:
            self.stack.enter_context(mock.patch.object(target, attr, new))
        self.addCleanup(self.stack.close)
        self.log = []

    def real_mode(self, dirty=(), lineage=None):
        """As if run on the committed home and ledger: no trial, a clean tree, HEAD descending from the declare."""
        st = contextlib.ExitStack()
        st.enter_context(mock.patch.object(ft, "is_trial", lambda flag=False: False))
        st.enter_context(mock.patch.object(config, "LEDGER", self.home / "ledger.jsonl"))
        st.enter_context(mock.patch.object(config, "RESULTS", self.home / "RESULTS.md"))
        st.enter_context(mock.patch.object(ft, "tree_dirty", lambda: list(dirty)))
        st.enter_context(mock.patch.object(ft, "lineage_problem", lambda commit, llm=False: lineage))
        return st

    def declare(self, **kw):
        return ft.declare(self.bank, log=self.log.append, **kw)

    def run_tests(self, ids, **kw):
        return ft.test(self.bank, ids, log=self.log.append, **kw)

    def test_trials_shift_anchors_to_2018_and_never_write_the_real_ledger(self):
        # The committed ledger's paths, as when BRAIN_LEDGER and BRAIN_RESULTS are unset: the trial is one by its
        # MONEY_HOME alone, and must still never write them (a BRAIN_LEDGER in the environment changes nothing here).
        self.stack.enter_context(mock.patch.object(config, "LEDGER", ft.DEFAULT_LEDGER))
        self.stack.enter_context(mock.patch.object(config, "RESULTS", ft.DEFAULT_RESULTS))
        real = [(p, p.read_bytes() if p.exists() else None) for p in (ft.DEFAULT_LEDGER, ft.DEFAULT_RESULTS)]
        self.assertTrue(ft.is_trial())                                  # a MONEY_HOME other than var/
        led, res = ft.paths(True)
        self.assertEqual(led, self.home / "trial" / "ledger.jsonl")
        new = self.declare(llm_cutoff="2025-06")
        self.assertEqual({r["spec"]["anchor"] for r in new}, {"2018-12-31"})
        self.assertTrue(all(r["spec"]["trial"] for r in new))
        self.assertTrue(all(r["spec"]["window"]["last_exit"] <= "2018-12-31" for r in new))
        self.assertTrue(any("not earlier than F1a" in m for m in self.log))  # a trial warns, a real declare refuses
        out = self.run_tests([t["id"] for t in ft.FORECAST_TESTS])
        self.assertEqual([r["id"] for r in out], [t["id"] for t in ft.FORECAST_TESTS])
        for r in out:
            self.assertEqual(r["anchor"], "2018-12-31")
            self.assertTrue(r["trial"])
            self.assertLessEqual(r["plan"]["last_exit"], "2018-12-31")
        for grid_, h in self.features.calls:                           # nothing later was ever computed
            self.assertLessEqual(evaluate.exit_of(self.px.dates, grid_[-1], h), "2018-12-31")
        self.assertEqual(len(self.features.calls), 2)                   # one weekly, one monthly feature set
        for p, before in real:
            self.assertEqual(p.read_bytes() if p.exists() else None, before)
        recs = ledger.read(led)
        self.assertEqual([r["op"] for r in recs], ["declare"] * 4 + ["result"] * 4)
        self.assertIn("## Forecasts", res.read_text())
        # F1d: every reply kept, failures counted as F1's answer, an exact 2^8 sign flip
        d = next(r for r in out if r["id"].startswith("F1d"))
        self.assertEqual(len(d["replies"]), d["metrics"]["llm_asked"])
        self.assertEqual(d["metrics"]["llm_failed"], sum(not x["ok"] for x in d["replies"]))
        self.assertGreater(d["metrics"]["llm_failed"], 0)
        for x in d["replies"]:
            if not x["ok"]:
                self.assertIsNone(x["p_beat"])
        c = d["metrics"]["comparison"]
        self.assertTrue(c["exact"])
        self.assertAlmostEqual(c["p"] * 256, round(c["p"] * 256))
        self.assertEqual(len(c["per_date"]), 8)
        self.assertEqual({t for _, t, _ in self.think.asked}, set(F1D))
        self.assertEqual(len(self.think.asked), 8 * 12)
        self.assertEqual(set(self.think.no_think), {True})      # the declaration's pinned /no_think decision, each time

    def test_declare_pins_anchor_prices_code_and_model(self):
        with self.real_mode():
            new = self.declare(llm_cutoff="2010-01")
            sp = {r["id"]: r["spec"] for r in new}
            self.assertEqual(set(sp), set(ft.BY_ID))
            self.assertEqual({s["anchor"] for s in sp.values()}, {self.px.dates[-1]})   # every stock has its last bar
            self.assertEqual(len({s["prices_asof"] for s in sp.values()}), 1)
            for s in sp.values():
                self.assertEqual(s["kind"], "forecast")
                # F1d's code_sha also covers its LLM files
                self.assertEqual(s["code_sha"], ft.code_sha(llm=s["forecaster"] == "F1-llm"))
                self.assertEqual(s["LEARN"], json.loads(json.dumps(SMALL)))
                self.assertEqual(s["FEATURES"]["names"], list(StubFeatures.NAMES))
                self.assertEqual(s["universe"], "universe/us_broad.json")
                self.assertEqual(s["universe_sha"], run.universe_sha("universe/us_broad.json"))
            self.assertEqual(sp["F1d-llm-w5-recent"]["llm"], {"model": "qwen3:8b", "digest": "sha256:aaa",
                                                              "no_think": True, "cutoff": "2010-01",
                                                              "temperature": 0, "seed": 7, "primer_sha": "primer0",
                                                              "prompt_sha": ft._sha(StubThink.PROMPT),
                                                              "snapshot_version": 1})
            self.assertEqual(sp["F1b-w5-test"]["window"]["last_exit"],
                             self.px.dates[int(np.searchsorted(self.px.dates, sp["F1a-w5-recent"]["window"]["first"])) + 1])
            self.assertEqual(self.declare(llm_cutoff="2010-01"), [])     # already declared: nothing new
            # new prices since, fetched after the declaration (more sessions and revised closes): the declared
            # anchor, prices_asof and the book the run reads stay as pinned
            asof = sp["F1a-w5-recent"]["prices_asof"]
            pinned = list(ft.calendar(self.bank, asof))
            days = pd.bdate_range("2009-01-01", "2019-09-30").strftime("%Y-%m-%d").tolist()
            self.bank.add(dataclasses.replace(price_snapshot("SPY", days, list(np.linspace(90, 130, len(days)))),
                                              retrieved_at=later(asof)))
            self.assertEqual(ft.declared("F1b-w5-test", config.LEDGER)["spec"]["anchor"], self.px.dates[-1])
            self.assertEqual(list(ft.calendar(self.bank, asof)), pinned)          # the later fetch is not read
            self.assertEqual(pinned, list(self.px.dates))
            out = self.run_tests(["F1a-w5-recent"])
            self.assertEqual(out[0]["anchor"], self.px.dates[-1])
            self.assertEqual(out[0]["prices_asof"], asof)
            self.assertEqual(out[0]["plan"]["last_exit"], self.px.dates[-1])
            # a snapshot the pin cannot place in time (no retrieval time, known before prices_asof) changes the book
            # the run would read: refused, not run on other prices than the declaration's
            j = self.px.tickers.index("S00")
            self.bank.add(price_snapshot("S00", list(self.px.dates), list(self.px.adj[:, j] * 1.01)))
            with self.assertRaises(ft.Refused) as cm:
                self.run_tests(["F1b-w5-test"])
            self.assertIn("price snapshots", str(cm.exception))
            self.assertEqual(sp["F1b-w5-test"]["snapshots_sha"], sp["F1a-w5-recent"]["snapshots_sha"])

    def test_declare_refusals(self):
        with self.real_mode(dirty=["?? brain/forecast/new.py"]):
            with self.assertRaises(ft.Refused) as cm:
                self.declare()
            self.assertIn("not clean", str(cm.exception))
        with self.real_mode():
            with self.assertRaises(ft.Refused) as cm:
                self.declare(llm_cutoff="2019-05")                       # after F1a's first date
            self.assertIn("not earlier than F1a", str(cm.exception))
            with self.assertRaises(ft.Refused):
                self.declare(llm_cutoff="May 2019")
            self.think.digest = ""
            with self.assertRaises(ft.Refused) as cm:
                self.declare(llm_cutoff="2010-01")
            self.assertIn("digest", str(cm.exception))
            self.assertFalse(config.LEDGER.exists())                     # refused before writing anything
            new = self.declare()                                         # no cutoff: F1d left undeclared
            self.assertEqual([r["id"] for r in new], ["F1a-w5-recent", "F1b-w5-test", "F1c-m21-test"])
            self.assertTrue(any("F1d not declared" in m for m in self.log))

    def test_test_refusals(self):
        with self.real_mode():
            with self.assertRaises(ft.Refused) as cm:
                self.run_tests(["F1a-w5-recent"])
            self.assertIn("not declared", str(cm.exception))
            self.declare(llm_cutoff="2010-01")
            with self.assertRaises(ft.Refused):
                self.run_tests(["F9-nothing"])
            with mock.patch.object(ft, "code_sha", lambda llm=False: "0" * 16):
                with self.assertRaises(ft.Refused) as cm:
                    self.run_tests(["F1a-w5-recent"])
                self.assertIn("code has changed", str(cm.exception))
            changed = {**ft.BY_ID, "F1a-w5-recent": {**ft.BY_ID["F1a-w5-recent"], "periods": 9}}
            with mock.patch.object(ft, "BY_ID", changed):
                with self.assertRaises(ft.Refused) as cm:
                    self.run_tests(["F1a-w5-recent"])
                self.assertIn("no longer matches", str(cm.exception))
            with mock.patch.object(learn, "LEARN", {**SMALL, "max_rows": 1}), self.assertRaises(ft.Refused):
                self.run_tests(["F1a-w5-recent"])
            self.think.digest = "sha256:bbb"
            with self.assertRaises(ft.Refused) as cm:
                self.run_tests(["F1d-llm-w5-recent"])
            self.assertIn("digest", str(cm.exception))
            self.think.digest = "sha256:aaa"
            with mock.patch.object(StubThink, "PROMPT", (("system", "edited"),)):
                with self.assertRaises(ft.Refused) as cm:
                    self.run_tests(["F1d-llm-w5-recent"])
                self.assertIn("prompt_sha", str(cm.exception))
            self.assertFalse(any(r["op"] == "result" for r in ledger.read(config.LEDGER)))
            self.run_tests(["F1a-w5-recent"])
            with self.assertRaises(ft.Refused) as cm:                     # one result per id and spec
                self.run_tests(["F1a-w5-recent"])
            self.assertIn("already has a result", str(cm.exception))
        with self.real_mode(dirty=[" M brain/forecast/learn.py"]):
            with self.assertRaises(ft.Refused) as cm:
                self.run_tests(["F1b-w5-test"])
            self.assertIn("not clean", str(cm.exception))
        with self.real_mode(lineage="HEAD is not the declare commit abc or a descendant of it"):
            with self.assertRaises(ft.Refused) as cm:
                self.run_tests(["F1b-w5-test"])
            self.assertIn("descendant", str(cm.exception))
        self.assertEqual(sum(r["op"] == "result" for r in ledger.read(self.home / "ledger.jsonl")), 1)

    def test_no_declaration_after_an_overlapping_result(self):
        with self.real_mode():
            self.assertEqual([r["id"] for r in self.declare()], ["F1a-w5-recent", "F1b-w5-test", "F1c-m21-test"])
            self.assertTrue(any("once any of them has a result" in m for m in self.log))
            self.run_tests(["F1a-w5-recent"])
            before = config.LEDGER.read_bytes()
            # F1d would pin a model, primer and prompt chosen with F1a's numbers on the same 8 weeks in view
            self.think.digest = "sha256:chosen-after-seeing-F1a"
            with self.assertRaises(ft.Refused) as cm:
                self.declare(llm_cutoff="2010-01")
            self.assertIn("F1d-llm-w5-recent", str(cm.exception))
            self.assertIn("overlaps F1a-w5-recent", str(cm.exception))
            self.assertIn("brain research withdraw F1d-llm-w5-recent", str(cm.exception))
            self.assertEqual(config.LEDGER.read_bytes(), before)               # refused before writing anything
            # recorded as dropped: never declared, and the declare says so
            ledger.withdraw("F1d-llm-w5-recent", "dropped: not declared before F1a's result", config.LEDGER)
            self.log.clear()
            self.assertEqual(self.declare(llm_cutoff="2010-01"), [])
            self.assertTrue(any("withdrawn (dropped) before it was declared" in m for m in self.log))
            self.assertIsNone(ft.declared("F1d-llm-w5-recent", config.LEDGER))
            # a withdrawn declared test is not run either
            ledger.withdraw("F1c-m21-test", "a mistake", config.LEDGER)
            with self.assertRaises(ft.Refused) as cm:
                self.run_tests(["F1c-m21-test"])
            self.assertIn("withdrawn (a mistake)", str(cm.exception))

    def test_a_withdrawn_result_does_not_block_and_a_trial_only_warns(self):
        with self.real_mode():
            self.declare(ids=["F1a-w5-recent"])
            self.run_tests(["F1a-w5-recent"])
            with self.assertRaises(ft.Refused):                                # F1b's last window ends at F1a's entry
                self.declare(ids=["F1b-w5-test"])
            ledger.withdraw("F1a-w5-recent", "a mistake in the code", config.LEDGER)   # recorded, with its reason
            self.assertEqual([r["id"] for r in self.declare(ids=["F1b-w5-test"])], ["F1b-w5-test"])
        # a trial (TRAIN years) declares late with a warning: plumbing, not evidence
        self.declare()
        self.run_tests(["F1a-w5-recent"])
        self.log.clear()
        self.assertEqual([r["id"] for r in self.declare(llm_cutoff="2010-01")], ["F1d-llm-w5-recent"])
        self.assertTrue(any(m.startswith("trial: F1d-llm-w5-recent: its window") for m in self.log))

    def test_gated_results_carry_p_controls_and_enter_the_report(self):
        with self.real_mode():
            self.declare()
            out = {r["id"]: r for r in self.run_tests(["F1a-w5-recent", "F1b-w5-test", "F1c-m21-test"])}
        b = out["F1b-w5-test"]
        for k in ("metrics", "p", "controls", "n_dates", "n_samples", "dropped", "model", "cache_key", "prices_asof",
                  "spec_sha", "commit"):
            self.assertIn(k, b)
        self.assertEqual(b["kind"], "forecast")
        self.assertEqual(b["controls"]["min_p"], 1 / 2 ** b["gate"]["k"] if b["gate"]["k"] <= 16 else 1 / 20001)
        self.assertNotIn("p", out["F1a-w5-recent"])
        self.assertIsNone(out["F1a-w5-recent"]["controls"]["pass"])
        for m in b["model"]:
            self.assertLessEqual(m["last_exit"], m["cutoff"])
        text = (self.home / "RESULTS.md").read_text()
        self.assertIn("## Forecasts", text)
        self.assertIn("| F1b-w5-test | 5 |", text)
        self.assertIn("| F1a-w5-recent | 5 |", text)
        self.assertIn("**REPORT** |", text)
        self.assertIn("### Calibration", text)
        self.assertIn("the 68% band held", text)
        # the placebo checks the sign-flip test, not the forecaster: RESULTS.md says so and claims nothing more
        self.assertIn("a check of the sign-flip test itself, not of the forecaster", text)
        self.assertIn("a check of the test itself", ft.summary(b))

    def test_cli_registers_declare_and_test(self):
        sub = argparse.ArgumentParser().add_subparsers(dest="cmd")
        self.assertIs(ft.add_cli(sub), ft.cmd)
        a = sub.choices["forecast"].parse_args(["test", "F1a-w5-recent", "--trial"])
        self.assertEqual((a.words, a.trial, a.fn), (["test", "F1a-w5-recent"], True, ft.cmd))
        own = argparse.ArgumentParser()
        own.add_argument("words", nargs="*")
        self.assertIs(ft.add_cli(own), ft.cmd)                          # the product's forecast parser
        a = own.parse_args(["declare", "--llm-cutoff", "2025-01"])
        self.assertEqual(a.llm_cutoff, "2025-01")
        import io
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(ft, "declare", lambda bank, **kw: []), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            self.assertEqual(ft.cmd(a), 0)
            self.assertEqual(ft.cmd(argparse.Namespace(words=["score"])), 2)   # not this module's word
        self.assertIn("nothing new to declare", out.getvalue())
        self.assertIn("trial: TRAIN years only", err.getvalue())


class CodeShaTest(unittest.TestCase):
    """What ``code_sha`` and the HEAD-lineage check cover: a change to any of it after the declare is refused."""
    F1_MODULES = ("brain/bank.py", "brain/timeutil.py", "brain/sources/sec.py", "brain/research/run.py",
                  "brain/metrics.py", "brain/research/forecast_tests.py", "docs/MARKET_PRIMER.md", "brain/forecast")
    LLM_MODULES = ("brain/judge.py", "brain/llm.py", "brain/terminal/analyst.py")

    def test_the_point_in_time_view_and_the_models_helpers_are_pinned(self):
        f1, f1d = ft._code_paths(), ft._code_paths(llm=True)
        for p in self.F1_MODULES:
            self.assertIn(p, f1)
            self.assertIn(p, f1d)
        for p in self.LLM_MODULES:                                     # F1d's alone: F1's tests do not wait on them
            self.assertNotIn(p, f1)
            self.assertIn(p, f1d)
        self.assertNotEqual(ft.code_sha(), ft.code_sha(llm=True))

    def edited(self, rel):
        target, real = (config.ROOT / rel).resolve(), Path.read_bytes

        def read(p):
            b = real(p)
            return b + b"\n# edited after the declare\n" if Path(p).resolve() == target else b
        return mock.patch.object(Path, "read_bytes", read)

    def test_an_edit_to_the_view_changes_the_shas_and_the_feature_cache_key(self):
        from brain.forecast import features
        before = (ft.code_sha(), ft.code_sha(llm=True), features.deps_sha())
        for rel in ("brain/bank.py", "brain/timeutil.py", "brain/sources/sec.py"):
            with self.subTest(rel=rel), self.edited(rel):
                after = (ft.code_sha(), ft.code_sha(llm=True), features.deps_sha())
                for a, b in zip(before, after):
                    self.assertNotEqual(a, b)
        with self.edited("brain/research/run.py"):                    # the verdicts: the tests' code, not a feature
            self.assertNotEqual(ft.code_sha(), before[0])
            self.assertEqual(features.deps_sha(), before[2])
        for rel in self.LLM_MODULES:
            with self.subTest(rel=rel), self.edited(rel):
                self.assertEqual(ft.code_sha(), before[0])
                self.assertNotEqual(ft.code_sha(llm=True), before[1])


def frec(i, p=None, gate=True, ctl=True, gated=True):
    r = {"op": "result", "kind": "forecast", "id": i, "spec_sha": "s", "at": "2026-11-01T00:00:00Z",
         "gate_type": "ic" if gated else "report", "metrics": {}, "n_dates": 10, "n_samples": 100,
         "controls": {"pass": ctl} if gated else {"pass": None}}
    if gated:
        r.update(p=p, gate={"pass": gate})
    return r


def hrec(i, p):
    return {"op": "result", "id": i, "spec_sha": "x", "test": {"perm_p": p, "mean_net_excess": 0.01},
            "controls": {"pass": True}}


class VerdictsTest(unittest.TestCase):
    def test_bh_counts_gated_forecasts_and_leaves_descriptive_ones_out(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.jsonl"
            ledger.append(hrec("H1", 0.02), p)
            ledger.append(hrec("H2", 0.04), p)
            alone = {r["id"]: r for r in run.verdicts(p)}
            self.assertAlmostEqual(alone["H1"]["bh_q"], 0.04)
            ledger.append(frec("F1a", gated=False), p)                  # descriptive: outside BH
            same = {r["id"]: r for r in run.verdicts(p)}
            self.assertAlmostEqual(same["H1"]["bh_q"], 0.04)
            self.assertEqual(same["F1a"]["verdict"], "REPORT")
            self.assertTrue(np.isnan(same["F1a"]["bh_q"]))
            ledger.append(frec("F1b", p=0.01), p)                       # gated: in BH
            rows = {r["id"]: r for r in run.verdicts(p)}
            self.assertAlmostEqual(rows["H1"]["bh_q"], 0.03)            # three tests now: 0.02 * 3 / 2
            self.assertAlmostEqual(rows["F1b"]["bh_q"], 0.03)           # 0.01 * 3 / 1, capped by the next
            self.assertEqual(rows["F1b"]["verdict"], "PASS")
            ledger.append(frec("F1c", p=0.9, ctl=False), p)
            ledger.withdraw("F1c", "a code mistake", p)
            rows = {r["id"]: r for r in run.verdicts(p)}
            self.assertAlmostEqual(rows["F1b"]["bh_q"], 0.04)           # withdrawn F1c still counts: 0.01 * 4 / 1
            self.assertTrue(rows["F1c"]["verdict"].startswith("WITHDRAWN"))
            ledger.append(frec("F1e", p=0.001, ctl=False), p)
            ledger.append(frec("F1f", p=0.2, gate=False), p)
            rows = {r["id"]: r for r in run.verdicts(p)}
            self.assertEqual(rows["F1e"]["verdict"], "FAIL (controls)")
            self.assertEqual(rows["F1f"]["verdict"], "FAIL")
            self.assertEqual({r["id"] for r in run.forecast_verdicts(p)}, {"F1a", "F1b", "F1c", "F1e", "F1f"})
            self.assertEqual({r["id"] for r in run.verdicts(p) if not run.is_forecast(r)}, {"H1", "H2"})


if __name__ == "__main__":
    unittest.main()
