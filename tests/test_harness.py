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
