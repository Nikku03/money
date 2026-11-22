"""Stocks that moved against their wave: selection, outcomes and the forward tests' judging (R2a-R2c)."""
from __future__ import annotations

import unittest

import numpy as np

from brain import againstwave as aw
from brain import ripple
from brain.research import againsttest as at


class Select(unittest.TestCase):
    def test_against_and_matched_controls(self):
        #           0 against  1 with     2 ctrl     3 ctrl news 4 stone  5 too small 6 wave-against-but-small
        z = np.array([-2.0, 1.5, -2.2, -2.1, -4.0, -0.4, -1.8])
        H = np.array([+0.8, 0.6, 0.1, 0.0, 0.0, 0.0, +0.1])
        stone = np.array([False, False, False, False, True, False, False])
        news = np.array([False, False, False, True, False, False, False])
        against, controls = aw.select(z, H, stone, news)
        self.assertEqual(against, [0])
        self.assertEqual(sorted(controls[0]), [2, 6])    # same sign, close in size, same news, no wave against
        self.assertNotIn(3, controls[0])                 # different news status
        self.assertNotIn(4, controls[0])                 # a stone

    def test_controls_capped_and_ordered(self):
        n = 30
        z = np.full(n, -1.5)
        z[0] = -2.0
        H = np.zeros(n)
        H[0] = 0.9
        z[1:] = -2.0 + np.linspace(-0.4, 0.4, n - 1)
        _, c = aw.select(z, H, np.zeros(n, bool), np.zeros(n, bool))
        self.assertEqual(len(c[0]), aw.N_CONTROLS)
        sizes = [abs(abs(z[j]) - 2.0) for j in c[0]]
        self.assertEqual(sizes, sorted(sizes))


class Outcome(unittest.TestCase):
    def test_outcome_in_usual_days(self):
        r = np.full(aw.H, 0.01)
        m = np.full(aw.H, 0.005)
        y = aw.outcome(r, m, beta=1.0, s=0.01)
        self.assertAlmostEqual(y, 5 * 0.005 / (0.01 * np.sqrt(aw.H)))
        r[2] = np.nan
        self.assertTrue(np.isnan(aw.outcome(r, m, 1.0, 0.01)))


def row(date, news, sign, y, yc, rec_at=None, nxt=None):
    return {"date": date, "recorded_at": rec_at or f"{date}T23:00:00Z", "t": "X", "news": news, "sign": sign, "y": y,
            "y_controls": yc, "n_controls": 5, "next_session": nxt or date}


class Judge(unittest.TestCase):
    def test_eligible_drops_late_and_early_records(self):
        rows = [row("2026-11-20", False, 1, 0.1, 0.0, rec_at="2026-11-22T10:00:00Z", nxt="2026-11-23"),
                row("2026-11-24", False, 1, 0.1, 0.0, rec_at="2026-11-25T21:00:00Z", nxt="2026-11-25"),
                row("2026-11-19", False, 1, 0.1, 0.0, rec_at="2026-11-19T23:00:00Z", nxt="2026-11-20")]
        kept = at.eligible(rows)
        self.assertEqual([r["date"] for r in kept], ["2026-11-20"])

    def test_values_per_test(self):
        rows = [row("2026-11-23", False, -1, 0.5, 0.1), row("2026-11-24", True, 1, 0.4, 0.0)]
        v, _ = at.values(rows, "R2a")
        self.assertEqual(v.tolist(), [0.5])             # fell, then rose 0.5: bounced back
        v, _ = at.values(rows, "R2b")
        self.assertEqual(v.tolist(), [0.4])             # rose with news, kept going
        v, _ = at.values(rows, "R2c")
        self.assertAlmostEqual(v[0], 0.5 - 0.1)

    def test_weekly_and_evaluate(self):
        rng = np.random.default_rng(0)
        days = np.arange("2027-01-04", "2027-09-01", dtype="datetime64[D]")
        rows = [row(str(d), False, -1, float(rng.normal(0.3, 0.5)), 0.0) for d in days if d.astype(object).weekday() < 5]
        r = at.evaluate(rows, "R2a")
        self.assertGreaterEqual(r["metrics"]["weeks"], at.MIN_WEEKS)
        self.assertGreater(r["metrics"]["mean"], 0.1)
        self.assertTrue(r["gate"]["pass"])
        few = at.evaluate(rows[:20], "R2a")
        self.assertFalse(few["gate"]["weeks_ok"])


class Thresholds(unittest.TestCase):
    def test_against_uses_the_maps_rule(self):
        self.assertEqual(ripple.how_it_moved(0.3, -1.2, False), "against its wave")
        self.assertEqual(ripple.how_it_moved(0.2, -1.2, False), "with the water")


if __name__ == "__main__":
    unittest.main()
