"""Step 2's tests (brain/research/tilttest.py): the quality tilt and buying after fear, on made-up data."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from brain.research import ledger, run, tilttest
from brain.research.tilttest import W_CALM, W_FEAR


class TraitsTest(unittest.TestCase):
    def test_a_calm_stock_at_its_high_scores_above_a_wild_one_far_below_it(self):
        rng = np.random.default_rng(0)
        n = 300
        calm = 100 * np.exp(np.cumsum(np.full(n, 0.001)))                     # steady and at its high
        wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
        fall = np.concatenate([np.linspace(100, 200, 200), np.linspace(200, 120, 100)])
        young = np.concatenate([np.full(200, np.nan), np.linspace(10, 11, 100)])  # 100 sessions: too few
        adj = np.column_stack([calm, wild, fall, young])
        t = tilttest.price_traits(adj, [n - 1])
        self.assertGreater(t["calm"][0, 0], t["calm"][0, 1])
        self.assertAlmostEqual(t["near_high"][0, 0], 1.0)
        self.assertAlmostEqual(t["near_high"][0, 2], 120 / 200, places=6)
        self.assertTrue(np.isnan(t["calm"][0, 3]) and np.isnan(t["near_high"][0, 3]))
        self.assertTrue(np.isnan(tilttest.price_traits(adj, [100])["calm"]).all())    # not 252 sessions back

    def test_the_list_is_the_largest_by_value_that_day_among_stocks_with_three_traits(self):
        mcap = np.array([[50.0, 40, 30, 20, 10, 60]])
        full = np.ones((1, 6))
        T = {"mcap": mcap, "profit": full.copy(), "no_dilution": full.copy(), "calm": full.copy(),
             "near_high": full.copy()}
        T["profit"][0, 5] = T["no_dilution"][0, 5] = np.nan                    # the largest has only two traits
        listed = np.array([[True, True, True, False, True, True]])             # the fourth has no price that day
        mem = tilttest.membership(T, listed, size=3)
        self.assertEqual(np.flatnonzero(mem[0]).tolist(), [0, 1, 2])

    def test_the_score_averages_the_ranks_of_the_traits_a_member_has(self):
        mem = np.array([[True, True, True, False]])
        T = {"profit": np.array([[3.0, 2.0, 1.0, 9.0]]), "no_dilution": np.array([[3.0, 2.0, np.nan, 9.0]]),
             "calm": np.array([[1.0, 2.0, 3.0, 9.0]]), "near_high": np.array([[3.0, 2.0, 1.0, 9.0]])}
        S = tilttest.score(T, mem)
        self.assertTrue(np.isnan(S[0, 3]))                                      # off the list
        self.assertGreater(S[0, 0], S[0, 1])
        self.assertGreater(S[0, 1], S[0, 2])
        self.assertAlmostEqual(S[0, 0], np.mean([0.5, 0.5, -0.5, 0.5]))
        self.assertTrue(np.isnan(tilttest.score(T, mem, need=4)[0, 2]))         # the third has only three

    def test_the_index_without_its_junk_drops_the_bottom_third_and_weights_by_value(self):
        S = np.array([[0.4, 0.1, -0.2, -0.4, 0.3, 0.0]])
        mem = np.ones((1, 6), dtype=bool)
        low = tilttest.bottom_third(S, mem)
        self.assertEqual(np.flatnonzero(low[0]).tolist(), [2, 3])
        R = np.array([[0.10, 0.00, -0.50, -0.50, 0.02, 0.01]])
        mcap = np.array([[4.0, 1, 1, 1, 2, 1]])
        cw = tilttest.cap_weighted(R, mem, mcap, mem & ~low, cost_bps=10)
        self.assertAlmostEqual(cw["gross"][0], (0.4 + 0 + 0.04 + 0.01) / 8)
        self.assertAlmostEqual(cw["bench"][0], (0.4 + 0 - 0.5 - 0.5 + 0.04 + 0.01) / 10)
        self.assertAlmostEqual(cw["net"][0], cw["gross"][0] - 1.0 * 10 / 1e4)  # a first month turns over 100%


def _days(n: int, start: str = "2000-01-03") -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


class FearTest(unittest.TestCase):
    def _frames(self, vix_close, spy_adj=None, rate=5.0):
        d = _days(len(vix_close))
        spy = pd.DataFrame({"date": d, "adjclose": spy_adj if spy_adj is not None else np.linspace(100, 120, len(d))})
        return spy, pd.DataFrame({"date": d, "close": vix_close}), pd.DataFrame({"date": d, "close": [rate] * len(d)})

    def test_a_spike_puts_the_next_thirteen_weeks_at_full_weight(self):
        v = np.full(400, 15.0)
        v[200:] = 15.0
        days = _days(400)
        fri = next(i for i in range(200, 210) if pd.Timestamp(days[i]).weekday() == 4)
        v[fri] = 30.0                                                           # twice its 63-day mean
        wk = tilttest.weekly(*self._frames(v))
        k = int(np.flatnonzero(wk["date"] == days[fri])[0])
        self.assertTrue(wk["spike"][k] and wk["spike"].sum() == 1)
        self.assertEqual(wk["w"][k - 1], W_CALM)
        self.assertTrue((wk["w"][k:k + tilttest.FEAR_WEEKS] == W_FEAR).all())
        self.assertEqual(wk["w"][k + tilttest.FEAR_WEEKS], W_CALM)

    def test_a_week_s_return_starts_at_the_session_after_the_decision(self):
        n = 200
        spy_adj = 100 * 1.001 ** np.arange(n)
        wk = tilttest.weekly(*self._frames(np.full(n, 15.0), spy_adj=spy_adj, rate=0.0))
        days = _days(n)
        k = 5
        d0, d1 = wk["date"][k], wk["date"][k + 1]
        e0, e1 = days.index(d0) + 1, days.index(d1) + 1
        self.assertAlmostEqual(wk["r"][k], spy_adj[e1] / spy_adj[e0] - 1)
        self.assertEqual(wk["rf"][k], 0.0)
        self.assertTrue(np.isnan(wk["r"][-1]))                                  # no next week yet

    def test_timing_alone_is_tested_and_an_oracle_is_seen(self):
        rng = np.random.default_rng(1)
        ex = rng.normal(0.001, 0.02, 600)
        stat, p = tilttest.shift_p(np.full(600, 0.85), ex, 500, rng)            # a constant weight: no timing
        self.assertAlmostEqual(stat, 0.0)
        oracle = np.where(ex > 0, W_FEAR, W_CALM)
        stat, p = tilttest.shift_p(oracle, ex, 500, rng)
        self.assertGreater(stat, 0)
        self.assertLess(p, 0.01)
        noise = np.where(rng.random(600) < 0.3, W_FEAR, W_CALM)
        self.assertGreater(tilttest.shift_p(noise, ex, 500, rng)[1], 0.01)

    def test_windows_count_episodes_and_compare_with_spy_held_at_full_weight(self):
        wk = {"date": np.array(_days(300)[::5][:60]), "spike": np.zeros(60, bool), "after": np.zeros(60, bool),
              "w": np.full(60, W_CALM), "r": np.full(60, 0.002), "rf": np.full(60, 0.001)}
        wk["after"][10:23] = wk["after"][40:53] = True
        wk["spike"][10] = wk["spike"][40] = True
        wk["w"] = np.where(wk["after"], W_FEAR, W_CALM)
        s = tilttest.window_stats(wk, wk["date"][0], None)
        self.assertEqual((s["episodes"], s["spike_weeks"], s["weeks"]), (2, 2, 60))
        self.assertAlmostEqual(s["share_after"], 26 / 60)
        self.assertAlmostEqual(s["spy_excess_after"], 0.001)
        self.assertGreater(s["window"]["spy"]["grown_1000"], s["window"]["constant_mix"]["grown_1000"])


class DeclareTest(unittest.TestCase):
    def test_the_specs_pin_the_rules_and_differ(self):
        a, b = tilttest.spec("T1a"), tilttest.spec("T1b")
        self.assertEqual((a["list_size"], a["frac"], a["cost_bps"], a["traits"]),
                         (500, 1 / 3, 10.0, ["profit", "no_dilution", "calm", "near_high"]))
        self.assertEqual((b["test"], b["weights"], b["fear_ratio"], b["fear_weeks"]),
                         (["1993-02-01", "2010-12-31"], [0.7, 1.0], 1.25, 13))
        self.assertEqual(tilttest.spec_sha(a), tilttest.spec_sha(tilttest.spec("T1a")))
        self.assertNotEqual(tilttest.spec_sha(a), tilttest.spec_sha(b))
        self.assertEqual(set(a["universes"]), set(tilttest.UNIVERSES))


def _tilt_result(tid: str, p: float, stat: float) -> dict:
    g = {"n": 90, "grown_1000": 2000.0, "worst_fall": -0.2}
    if tid == "T1a":
        ann = {k: 0.1 for k in ("tilt", "list_ew", "list_cw", "without_junk", "SPY", "RSP")}
        te = {"mean_net_excess": stat, "perm_p": p, "months_beating_benchmark": 50, "n_months": 92, "avg_turnover": 0.2}
        m = {"tickers": 2500, "first": "2019-01-31", "last": "2026-09-30", "months": 92, "list_size_median": 500,
             "test": te, "train": {"mean_net_excess": 0.001, "perm_p": 0.2},
             "traits_alone": {k: {"mean_net_excess": 0.0, "perm_p": 0.5} for k in tilttest.TRAITS},
             "without_junk": {"mean_net_excess": 0.0005}, "annual": ann,
             "survivorship": {"list_ew_minus_rsp": 0.02, "list_cw_minus_spy": 0.01}}
        ctl = {"oracle_test_perm_p": 0.0005, "random_false_positive_rate": 0.05,
               "persistent_random_false_positive_rate": 0.06, "pass": True}
    else:
        w = {"weeks": 930, "first": "1993-02-05", "last": "2010-12-31", "spike_weeks": 40, "episodes": 18,
             "share_after": 0.3, "spy_excess_after": 0.003, "spy_excess_calm": 0.001, "mean_excess": stat, "p": p,
             "annual": {"timed": 0.08, "constant_mix": 0.07, "spy": 0.09},
             "window": {"timed": g, "spy": g, "constant_mix": g}}
        m = {"test": w, "seen": {**w, "first": "2011-01-07"}}
        ctl = {"oracle_p": 0.0005, "random_fpr": 0.05, "n_random": 200, "pass": True}
    return {"op": "result", "kind": "tilt", "id": tid, "spec_sha": "x", "gate_type": tilttest.ASKS[tid], "p": p,
            "gate": {"stat": stat, "p": p, "pass": stat > 0 and p < 0.05}, "controls": ctl, "metrics": m,
            "at": "2026-11-17T00:00:00Z"}


class VerdictTest(unittest.TestCase):
    def test_step_two_results_count_in_the_correction_and_reach_the_results_page(self):
        with tempfile.TemporaryDirectory() as d:
            led, res = os.path.join(d, "ledger.jsonl"), os.path.join(d, "RESULTS.md")
            from pathlib import Path
            ledger.append(_tilt_result("T1a", 0.001, 0.002), Path(led))
            ledger.append(_tilt_result("T1b", 0.40, -0.0001), Path(led))
            v = {r["id"]: r for r in run.verdicts(Path(led))}
            self.assertEqual(v["T1a"]["verdict"], "PASS")
            self.assertEqual(v["T1b"]["verdict"], "FAIL")
            self.assertAlmostEqual(v["T1a"]["bh_q"], 0.002)                    # two tests: q = p * 2 / 1
            text = run.report(Path(led), Path(res))
            self.assertIn("## Step 2: the quality tilt and buying after fear", text)
            self.assertIn("**T1a** (the quality tilt)", text)
            self.assertIn("**T1b** (buying after fear)", text)


if __name__ == "__main__":
    unittest.main()


class FlowTest(unittest.TestCase):
    """The whole declare-and-test flow, in trial mode, on a made-up bank: the glue between the parts runs, the
    ledger gets one result per test, and the results page is written."""

    def setUp(self):
        from tests.util import TempHome
        self.home = TempHome()
        self.home.setUp()
        self.bank = self.home.bank

    def tearDown(self):
        self.home.tearDown()

    def _bank(self, names):
        from tests.util import filed, price_snapshot
        rng = np.random.default_rng(5)
        days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2009-01-02", "2026-11-13")]
        n = len(days)
        facts = [price_snapshot("SPY", days, list(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, n)))))]
        for j, t in enumerate(names):
            px = 20 * np.exp(np.cumsum(rng.normal(2e-4, 0.01 + 0.002 * j, n)))
            facts.append(price_snapshot(t, days, list(px)))
            for q in pd.date_range("2008-12-31", "2026-09-30", freq="QE"):
                end = q.strftime("%Y-%m-%d")
                known = (q + pd.Timedelta(days=40)).strftime("%Y-%m-%dT21:00:00Z")
                start = (q - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
                facts.append(filed(t, "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
                                   1e8 * (1 + 0.01 * j) ** ((q.year - 2008) * 4 + q.quarter), end, known,
                                   form="10-Q", start=start, duration_days=90))
        self.bank.add_many(facts)

    def test_declare_then_test_writes_one_result_each_and_the_page(self):
        from unittest import mock

        from brain.sources import prices
        names = [f"T{i:02d}" for i in range(24)]
        self._bank(names)
        days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("1990-01-02", "2026-11-13")]
        rng = np.random.default_rng(9)
        vix = 15 + np.abs(np.cumsum(rng.normal(0, 0.5, len(days)))) % 30
        frames = {"SPY": pd.DataFrame({"date": days, "adjclose": 50 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, len(days)))),
                                       "close": 1.0}),
                  "^VIX": pd.DataFrame({"date": days, "close": vix, "adjclose": vix}),
                  "^IRX": pd.DataFrame({"date": days, "close": 3.0, "adjclose": 3.0}),
                  "RSP": pd.DataFrame({"date": days, "adjclose": np.linspace(10, 60, len(days)), "close": 1.0})}

        def fetch(sym, start="2000-01-01", ttl=0):
            return frames[sym], [], {}, None
        with mock.patch.object(prices, "fetch_daily", fetch), mock.patch.object(tilttest, "tickers", lambda: names), \
                mock.patch.object(tilttest, "N_RANDOM", 4), mock.patch.object(tilttest, "N_SHIFT", 200), \
                mock.patch.object(tilttest, "N_PERM", 200), mock.patch.object(tilttest, "N_PERM_TRAIT", 50):
            tilttest.declare(trial=True, log=None)
            out = tilttest.test(self.bank, trial=True, log=None)
        self.assertEqual([r["id"] for r in out], ["T1a", "T1b"])
        a, b = out
        self.assertEqual(a["metrics"]["tickers"], 24)
        self.assertEqual(a["metrics"]["list_size_median"], 24)
        self.assertTrue(np.isfinite(a["p"]) and np.isfinite(b["p"]))
        self.assertEqual(b["metrics"]["test"]["first"][:4], "1993")
        from brain.research.forecast_tests import paths
        led, res = paths(True)
        self.assertEqual(sum(1 for r in ledger.read(led) if r["op"] == "result"), 2)
        self.assertIn("## Step 2", res.read_text())
        with self.assertRaises(SystemExit):                                     # once only
            tilttest.test(self.bank, trial=True, log=None)
