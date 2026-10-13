import unittest

import numpy as np

from brain.research import harness, signals
from brain.research.panel import build
from tests.util import TempHome, price_snapshot


class HarnessTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.M, self.N = 96, 30
        self.R = rng.normal(0.01, 0.06, (self.M, self.N))
        self.listed = np.ones((self.M, self.N), bool)
        self.months = np.arange(self.M)

    def test_oracle_is_found(self):
        ev = harness.evaluate(self.R, self.R, self.listed, self.months, frac=1 / 3, cost_bps=10, n_perm=1000)
        self.assertLess(ev["perm_p"], 0.002)
        self.assertGreater(ev["mean_net_excess"], 0.02)

    def test_random_signals_pass_at_nominal_rate(self):
        rng = np.random.default_rng(1)
        ps = [harness.evaluate(rng.random((self.M, self.N)), self.R, self.listed, self.months,
                               frac=1 / 3, cost_bps=10, n_perm=300, seed=s)["perm_p"] for s in range(120)]
        fpr = np.mean(np.array(ps) < 0.05)
        self.assertLess(fpr, 0.12)
        self.assertGreater(np.mean(ps), 0.35)

    def test_costs_are_charged_on_turnover(self):
        S = np.tile(np.arange(self.N, dtype=float), (self.M, 1))  # constant ranking: one buy, then hold
        pf = harness.portfolio(S, self.R, self.listed, 1 / 3, 10)
        self.assertAlmostEqual(pf["turnover"][0], 1.0)
        self.assertAlmostEqual(pf["turnover"][1:].sum(), 0.0)
        self.assertAlmostEqual(pf["excess"][0] - pf["net"][0], 0.001)


class PanelTimingTest(TempHome):
    def test_entry_is_day_after_month_end_and_learner_uses_realised_months_only(self):
        import pandas as pd
        days = pd.bdate_range("2019-01-01", "2021-12-31").strftime("%Y-%m-%d").tolist()
        rng = np.random.default_rng(2)
        for t in ["A", "B", "C", "D", "SPY"]:
            px = list(100 * np.cumprod(1 + rng.normal(0, 0.01, len(days))))
            self.bank.add(price_snapshot(t, days, px))
        p = build(self.bank, ["A", "B", "C", "D"], start="2019-01-01")
        for m in range(len(p.entry)):
            self.assertEqual(p.entry[m], p.rebal[m] + 1)
            self.assertNotEqual(p.dates[p.rebal[m]][:7], p.dates[p.rebal[m] + 1][:7])
            self.assertEqual(p.exit[m], p.rebal[m + 1] + 1)
        feats = signals.compute(self.bank, p, ["rev1", "mom12_1"])
        S, used = signals.ridge_learner(feats, p, min_months=3)
        self.assertTrue(used)
        for m, ks in used.items():
            self.assertTrue(all(p.exit[k] <= p.rebal[m] for k in ks))
            self.assertNotIn(m - 1, ks)  # last month's outcome is not known on this rebalance day


def legacy_portfolio(S, R, listed, frac, cost_bps):
    """The harness as H1-H5 were run (commit 08e83bf), kept here to pin the default mode to it."""
    M, N = R.shape
    W, gross, bench, turn, prev = np.zeros((M, N)), np.full(M, np.nan), np.full(M, np.nan), np.zeros(M), np.zeros(N)
    for m in range(M):
        valid = listed[m] & np.isfinite(S[m])
        n = int(valid.sum())
        r = np.nan_to_num(R[m], nan=0.0)
        if listed[m].any():
            bench[m] = r[listed[m]].mean()
        if n < 3:
            prev = np.zeros(N)
            continue
        k = max(1, int(round(frac * n)))
        idx = np.flatnonzero(valid)
        hold = idx[np.argsort(-S[m, idx], kind="stable")[:k]]
        W[m, hold] = 1.0 / k
        gross[m] = r[hold].mean()
        turn[m] = np.abs(W[m] - prev).sum()
        prev = W[m]
    return W, gross, bench, turn


def legacy_perm_null(S, R, listed, frac, months, n_perm, seed):
    rng = np.random.default_rng(seed)
    total, used = np.zeros(n_perm), 0
    for m in months:
        valid = listed[m] & np.isfinite(S[m])
        n = int(valid.sum())
        if n < 3:
            continue
        k = max(1, int(round(frac * n)))
        r = np.nan_to_num(R[m], nan=0.0)
        rv, b = r[valid], r[listed[m]].mean()
        keys = rng.random((n_perm, n))
        pick = np.argpartition(keys, k - 1, axis=1)[:, :k]
        total += rv[pick].mean(axis=1) - b
        used += 1
    return total / max(used, 1)


class SelectModesTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.M, self.N = 96, 60
        self.R = rng.normal(0.01, 0.06, (self.M, self.N))
        self.R[rng.random(self.R.shape) < 0.02] = np.nan
        self.listed = rng.random((self.M, self.N)) < 0.9
        self.months = np.arange(self.M)

    def test_default_mode_is_bit_for_bit_the_declared_harness(self):
        rng = np.random.default_rng(12)
        S = rng.normal(size=(self.M, self.N))
        S[rng.random(S.shape) < 0.3] = np.nan
        S[5, :] = np.nan  # a month with no values
        S[6, 3:] = np.nan  # a month with three
        pf = harness.portfolio(S, self.R, self.listed, 1 / 3, 10)
        W, gross, bench, turn = legacy_portfolio(S, self.R, self.listed, 1 / 3, 10)
        for a, b in ((pf["W"], W), (pf["gross"], gross), (pf["bench"], bench), (pf["turnover"], turn)):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(harness.perm_null(S, self.R, self.listed, 1 / 3, self.months, 300, 8),
                                      legacy_perm_null(S, self.R, self.listed, 1 / 3, self.months, 300, 8))
        self.assertEqual(harness.evaluate(S, self.R, self.listed, self.months, frac=1 / 3, cost_bps=10, n_perm=200),
                         harness.evaluate(S, self.R, self.listed, self.months, frac=1 / 3, cost_bps=10, n_perm=200,
                                          select="top_frac"))

    def test_choose(self):
        s = np.array([np.nan, 5.0, 0.0, -1.0, 2.0, 3.0, np.nan, 4.0])
        listed = np.array([True, True, True, True, True, True, True, False])
        hold, pool = harness.choose(s, listed, 1 / 3, "top_frac")
        self.assertEqual(list(hold), [1, 5])  # top third of the 5 listed names with a value
        self.assertEqual(list(np.flatnonzero(pool)), [1, 2, 3, 4, 5])
        hold, pool = harness.choose(s, listed, 1 / 3, "top_frac_vs_all")
        self.assertEqual(list(hold), [1, 5])
        self.assertEqual(list(np.flatnonzero(pool)), [0, 1, 2, 3, 4, 5, 6])
        hold, pool = harness.choose(s, listed, 1 / 3, "positive")
        self.assertEqual(list(hold), [1, 4, 5])  # 7.0 is above 0 but not listed
        self.assertEqual(list(np.flatnonzero(pool)), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(harness.choose(np.where(s > 2, s, 0.0), listed, 1 / 3, "positive"), (None, None))  # 2 names
        with self.assertRaises(ValueError):
            harness.choose(s, listed, 1 / 3, "top_half")

    def test_positive_holds_every_name_above_zero(self):
        S = np.zeros((self.M, self.N))
        S[:, :5] = 1.0
        pf = harness.portfolio(S, self.R, self.listed, 1 / 3, 10, select="positive")
        for m in range(self.M):
            want = self.listed[m, :5]
            if want.sum() >= 3:
                np.testing.assert_array_equal(pf["W"][m] > 0, np.r_[want, np.zeros(self.N - 5, bool)])
            else:
                self.assertTrue(np.isnan(pf["gross"][m]))

    def _presence_edge(self):
        """Names with an 'event' (a value at all) earn 2% more; the value itself carries nothing."""
        rng = np.random.default_rng(13)
        has = rng.random((self.M, self.N)) < 0.25
        R = rng.normal(0.0, 0.05, (self.M, self.N)) + 0.02 * has
        S = np.where(has, rng.normal(size=has.shape), np.nan)
        return S, R

    def test_vs_all_null_tests_having_a_value_and_top_frac_does_not(self):
        S, R = self._presence_edge()
        listed = np.ones_like(self.listed)
        kw = dict(frac=1 / 3, cost_bps=10, n_perm=500)
        self.assertLess(harness.evaluate(S, R, listed, self.months, select="top_frac_vs_all", **kw)["perm_p"], 0.01)
        self.assertGreater(harness.evaluate(S, R, listed, self.months, select="top_frac", **kw)["perm_p"], 0.05)
        pos = np.where(np.isfinite(S), 1.0, 0.0)
        self.assertLess(harness.evaluate(pos, R, listed, self.months, select="positive", **kw)["perm_p"], 0.01)

    def test_new_nulls_are_calibrated(self):
        rng = np.random.default_rng(14)
        kw = dict(frac=1 / 3, cost_bps=10, n_perm=200)
        ps = {"top_frac_vs_all": [], "positive": []}
        for s in range(80):
            has = rng.random((self.M, self.N)) < 0.2
            S = np.where(has, rng.normal(size=has.shape), np.nan)
            ps["top_frac_vs_all"].append(harness.evaluate(S, self.R, self.listed, self.months, select="top_frac_vs_all",
                                                          seed=s, **kw)["perm_p"])
            ps["positive"].append(harness.evaluate(np.where(has, 1.0, 0.0), self.R, self.listed, self.months,
                                                   select="positive", seed=s, **kw)["perm_p"])
        for mode, p in ps.items():
            self.assertLess(np.mean(np.array(p) < 0.05), 0.13, mode)
            self.assertGreater(np.mean(p), 0.35, mode)
