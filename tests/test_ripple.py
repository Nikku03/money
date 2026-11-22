"""The ripple engine (brain/ripple.py) and its tests' statistics (brain/research/rippletest.py), on made-up data."""
from __future__ import annotations

import unittest

import numpy as np

from brain import ripple
from brain.research import rippletest as rt
from brain.research import run


def two_groups(n_days=250, seed=0):
    """Two groups of five stocks: each group shares a factor; the groups do not share one."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.01, n_days)
    fa, fb = rng.normal(0, 0.01, n_days), rng.normal(0, 0.01, n_days)
    R = np.empty((n_days, 10))
    for j in range(10):
        f = fa if j < 5 else fb
        R[:, j] = m + f + rng.normal(0, 0.008, n_days)
    return R, m


class Pond(unittest.TestCase):
    def setUp(self):
        R, m = two_groups()
        self.p = ripple.pond(R, m)

    def test_links_inside_groups_only(self):
        C = self.p["C"]
        self.assertGreater(C[0, 1], 0.4)
        self.assertLess(abs(C[0, 6]), 0.2)
        self.assertTrue(self.p["wet"][0, 1])
        self.assertFalse(self.p["wet"][0, 6])

    def test_a_stone_raises_its_own_group_only(self):
        z = np.zeros(10)
        z[0] = 4.0
        S = ripple.stones(self.p, z * self.p["s"])
        self.assertEqual(S.tolist(), [0])
        w = ripple.waves(self.p, z, S)
        self.assertTrue((w["H"][1:5] > 0.5 * ripple.KAPPA).all())
        self.assertTrue((np.abs(w["H"][5:]) < 0.3 * ripple.KAPPA).all())
        self.assertEqual(w["ring"][0], 0)
        self.assertTrue((w["ring"][1:5] == 1).all())
        self.assertTrue(w["stone"][0] and not w["stone"][1])

    def test_opposite_stones_cancel(self):
        z = np.zeros(10)
        z[0], z[1] = 4.0, -4.0
        S = np.array([0, 1])
        w = ripple.waves(self.p, z, S)
        self.assertLess(w["I"][2], 0.2)            # destructive
        z[1] = 4.0
        w2 = ripple.waves(self.p, z, S)
        self.assertGreater(w2["I"][2], 0.9)        # constructive
        self.assertGreater(abs(w2["H"][2]), abs(w["H"][2]))

    def test_shared_stones_are_not_counted_twice(self):
        z = np.zeros(10)
        z[0] = 3.0
        one = ripple.waves(self.p, z, np.array([0]))["H"][3]
        z[1] = 3.0
        two = ripple.waves(self.p, z, np.array([0, 1]))["H"][3]
        self.assertGreater(two, one)
        self.assertLess(two, 1.6 * one)            # the second stone mostly repeats the first

    def test_a_stones_push_is_capped(self):
        z = np.zeros(10)
        z[0] = 3.0
        a = ripple.waves(self.p, z, np.array([0]))["H"][2]
        z[0] = 30.0
        b = ripple.waves(self.p, z, np.array([0]))["H"][2]
        self.assertAlmostEqual(a, b, places=5)

    def test_no_stones(self):
        w = ripple.waves(self.p, np.zeros(10), np.array([], int))
        self.assertEqual(float(np.abs(w["H"]).sum()), 0.0)

    def test_withstand(self):
        self.assertEqual(ripple.withstand(0.1), "absorbed")
        self.assertEqual(ripple.withstand(-0.5), "felt")
        self.assertEqual(ripple.withstand(1.5), "a hit")

    def test_contributions(self):
        z = np.zeros(10)
        z[0], z[1] = 4.0, 2.0
        S = np.array([0, 1])
        w = ripple.waves(self.p, z, S)
        c = ripple.contributions(w, z, S, 2)
        self.assertEqual({i for i, _ in c}, {0, 1})
        self.assertAlmostEqual(sum(a for _, a in c), float(w["H"][2]), places=4)


class Days(unittest.TestCase):
    def test_days_yields_waves_and_outcomes(self):
        R, m = two_groups(400, seed=1)
        R[330, 0] += 0.08                          # a stone on day 330
        cal = np.array([f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(400)])
        px = {"cal": cal, "tickers": [f"T{i}" for i in range(10)], "R": R, "m": m}
        out = list(ripple.days(px, cal[320], cal[340]))
        d = next(x for x in out if x["date"] == cal[330])
        self.assertIn(0, d["S"].tolist())
        self.assertTrue((d["H"][1:5] > 0).all())
        self.assertTrue(np.isfinite(d["Y"][1:]).all())


class Statistics(unittest.TestCase):
    def test_daily_ic(self):
        day = np.repeat(np.arange(5), 30)
        a = np.tile(np.arange(30.0), 5)
        ds, ics = rt.daily_ic(day, a, a)
        self.assertEqual(len(ds), 5)
        self.assertTrue(np.allclose(ics, 1.0))
        ds, ics = rt.daily_ic(day, a, -a)
        self.assertTrue(np.allclose(ics, -1.0))

    def test_shuffle_keeps_each_day(self):
        rng = np.random.default_rng(0)
        day = np.repeat(np.arange(3), 10)
        x = np.arange(30.0)
        y = rt.shuffled_within_day(day, x, rng)
        for d in range(3):
            self.assertEqual(sorted(y[day == d]), sorted(x[day == d]))

    def test_direction_finds_a_real_push(self):
        rng = np.random.default_rng(1)
        n_days, per = 120, 60
        day = np.repeat(np.arange(n_days), per)
        H = rng.normal(0, 1, n_days * per)
        rows = {"day": day, "month": np.array([f"2024-{1 + d // 10:02d}" for d in day]), "H": H,
                "W": np.abs(H) + 0.6, "z": rng.normal(0, 1, len(H)), "Y": 0.3 * H + rng.normal(0, 1, len(H))}
        old = rt.N_PLACEBO
        rt.N_PLACEBO = 20
        try:
            r = rt.evaluate_direction(rows)
        finally:
            rt.N_PLACEBO = old
        self.assertGreater(r["metrics"]["ic"], 0.1)
        self.assertLess(r["p"], 0.01)
        self.assertTrue(r["gate"]["pass"])

    def test_report_section_renders(self):
        r = {"id": "R1c", "kind": "ripple", "gate_type": "direction", "p": 0.4, "bh_q": 0.6, "verdict": "FAIL",
             "controls": {"pass": True, "placebo_fpr": 0.05, "oracle_p": 0.0001},
             "metrics": {"ic": 0.001, "interval90": [-0.01, 0.01], "rows": 10, "sessions": 5, "months": 3,
                         "months_positive": 1, "spread_top_bottom": 0.0, "ic_gap_reported": 0.0,
                         "ic_own_reversal_reported": 0.0}}
        text = "\n".join(run._ripple_section([r]))
        self.assertIn("## Ripples", text)
        self.assertIn("R1c", text)
        self.assertTrue(run.is_gated_ripple(r))


if __name__ == "__main__":
    unittest.main()
