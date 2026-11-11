"""The earnings-aware shock radar S3 (brain/catalysts.py) and its test S3a (brain/research/catalysttest.py)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from brain import catalysts
from brain.research import catalysttest as ct


def days(n=400, start="2020-01-01"):
    return pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").to_numpy()


class ReportFeaturesTest(unittest.TestCase):
    def test_the_habit_is_known_from_the_second_report_and_counts_each_move_once_its_session_closed(self):
        d = days()
        e = np.full(len(d), 0.001)
        reps = [50, 113, 176, 239, 302]
        for k, size in zip(reps, (0.10, -0.30, 0.20, 0.05, 0.40)):
            e[k] = size
        f = catalysts.report_features(d, e, reps, np.zeros(len(d)))
        self.assertTrue(np.isnan(f["ehab"][112]))                       # one report so far: not yet a habit
        self.assertEqual(f["ehab_known"][112], 0)
        self.assertAlmostEqual(f["ehab"][113], 0.20, places=12)         # median of 0.10 and 0.30, from that close
        self.assertAlmostEqual(f["ehab"][175], 0.20, places=12)
        self.assertAlmostEqual(f["ehab"][176], 0.20, places=12)         # 0.10, 0.30, 0.20
        self.assertAlmostEqual(f["ehab"][239], 0.15, places=12)         # 0.10, 0.30, 0.20, 0.05
        self.assertAlmostEqual(f["ehab_l"][239], np.log(0.15), places=12)

    def test_a_report_a_quarter_after_the_last_is_due_one_to_eight_days_ahead(self):
        d = days()
        f = catalysts.report_features(d, np.zeros(len(d)), [50], np.zeros(len(d)))
        exp = np.datetime64(d[50]) + np.timedelta64(91, "D")
        due = [i for i in range(len(d)) if f["earnq"][i] > 0]
        self.assertTrue(due)
        for i in due:
            gap = (exp - np.datetime64(d[i])).astype(int)
            self.assertTrue(1 <= gap <= 8, (d[i], gap))
        self.assertEqual(f["earnq"][50], 0)

    def test_the_habit_counts_toward_the_chance_only_when_a_report_is_due(self):
        d = days()
        e = np.full(len(d), 0.001)
        e[[50, 113]] = (0.2, 0.2)
        win = np.zeros(len(d))
        win[150] = 1.0                                                   # S1's own estimate says due
        f = catalysts.report_features(d, e, [50, 113], win)
        self.assertAlmostEqual(f["ehab_win"][150], 0.2, places=12)
        self.assertEqual(f["ehab_win"][151], 0.0 if f["earnq"][151] == 0 else 0.2)
        self.assertEqual(f["ehab_win"][120], 0.0)


def part(w0: float, w1: float) -> dict:
    return {"feats": ["x"], "lo": [-50.0], "hi": [50.0], "mu": [0.0], "sd": [1.0], "w": [w0, w1]}


def synthetic_rows(seed=3, months=24, per_week=60):
    """Report-week and other rows whose shocks follow x: a model that reads x should win."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(months * 4):
        date = str(np.datetime64("2024-01-05") + np.timedelta64(7 * k, "D"))
        for j in range(per_week):
            x = rng.normal()
            p = 1 / (1 + np.exp(-(-3.0 + 1.5 * x)))
            rows.append({"t": f"T{j}", "date": date, "x": float(x), "y": float(rng.random() < p), "up": 1.0,
                         "due": float(j % 2 == 0)})
    return rows


class S3aTest(unittest.TestCase):
    def test_a_radar_that_reads_the_cause_passes_and_one_that_does_not_fails(self):
        rows = synthetic_rows()
        base = float(np.mean([r["y"] for r in rows]))
        s1 = {"s1": part(np.log(base / (1 - base)), 0.0)}               # the base rate only
        good = ct.evaluate(rows, s1, {"s3": part(-3.0, 1.5)})
        self.assertTrue(good["gate"]["pass"], good["gate"])
        self.assertTrue(good["controls"]["pass"], good["controls"])
        self.assertGreater(good["metrics"]["months_better"], good["metrics"]["months"] * 0.8)
        self.assertEqual(good["metrics"]["stock_weeks"], len([r for r in rows if r["due"] > 0]))
        self.assertIn("all_weeks", good["metrics"])
        same = ct.evaluate(rows, s1, {"s3": s1["s1"]})
        self.assertFalse(same["gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
