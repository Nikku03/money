"""The promise tests (brain/research/promisetest.py): the table, the statistics and the report."""
from __future__ import annotations

import unittest

import numpy as np

from brain import promises as pr
from brain.research import promisetest as pt


def promise(t, made, surprise, *, known=None, period_end=None, outcome_known=None, stretch=0.0, horizon="quarter",
            **more):
    d = {"t": t, "made": made, "known_at": known or made + "T21:05:00Z", "horizon": horizon, "low": 95.0,
         "high": 105.0, "mid": 100.0, "point": False, "stretch": stretch}
    if surprise is not None:
        d.update(surprise=surprise, grade=pr.grade(100.0 * (1 + surprise), 95.0, 105.0), period_end=period_end,
                 outcome_known=outcome_known, actual=100.0 * (1 + surprise))
    elif period_end:
        d["period_end"] = period_end
    d.update(more)
    return d


def company_rows(t, surprises, start_year=2020):
    """One quarterly promise a quarter, made a month after the quarter starts; its outcome known 95 days later."""
    out = []
    for k, s in enumerate(surprises):
        y, q = start_year + k // 4, k % 4
        made = f"{y}-{q * 3 + 2:02d}-10"
        end = f"{y}-{q * 3 + 3:02d}-28"
        yk, qk = (y, q + 1) if q < 3 else (y + 1, 0)
        known = f"{yk}-{qk * 3 + 2:02d}-12T12:00:00Z"
        out.append(promise(t, made, s, period_end=end, outcome_known=known))
    return out


class Table(unittest.TestCase):
    def test_season(self):
        self.assertEqual(pt.season("2025-02-10"), "2025Q1")
        self.assertEqual(pt.season("2025-12-31"), "2025Q4")

    def test_the_record_uses_only_outcomes_known_when_the_promise_was_made(self):
        lib = company_rows("AAA", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        Q, Y = pt.table(lib, [])
        self.assertEqual(len(Q), 6)
        self.assertEqual(Y, [])
        # the fifth promise is made two days before the fourth's outcome is filed: three known, no record yet
        self.assertIsNone(Q[4]["usual"])
        # the sixth sees the first four outcomes, not its own or the fifth's
        self.assertAlmostEqual(Q[5]["usual"], float(np.median([0.01, 0.02, 0.03, 0.04])))
        # what the same release delivered against the promise before it
        self.assertAlmostEqual(Q[5]["got"], 0.05)
        self.assertIsNone(Q[0]["got"])

    def test_the_report_session_and_the_year_s_change(self):
        lib = company_rows("AAA", [0.01, 0.02])
        lib += [promise("AAA", "2020-02-10", None, horizon="year", period_end="2020-12-31", mid=400.0),
                promise("AAA", "2020-05-10", None, horizon="year", period_end="2020-12-31", mid=420.0)]
        pr.changes(lib)
        eps = [{"t": "AAA", "date": "2020-02-11", "pre": "2020-02-10", "r1": 0.03, "post5": 0.01},
               {"t": "AAA", "date": "2020-05-11", "pre": "2020-05-08", "r1": -0.02, "post5": 0.004},
               {"t": "BBB", "date": "2020-05-11", "pre": "2020-05-08", "r1": 0.5, "post5": 0.5}]
        Q, Y = pt.table(lib, eps)
        self.assertEqual([r["r1"] for r in Q], [0.03, -0.02])
        self.assertEqual(len(Y), 1)                                    # the first promise for a year has no change
        self.assertAlmostEqual(Y[0]["change"], 0.05)
        self.assertEqual((Y[0]["changed"], Y[0]["post5"]), ("raised", 0.004))
        Q2, _ = pt.table(lib, eps, tickers=["BBB"])
        self.assertEqual(Q2, [])


class Statistics(unittest.TestCase):
    def data(self, n_seasons=12, n=40, slope=0.5, seed=5):
        rng = np.random.default_rng(seed)
        g = np.repeat([f"20{20 + k // 4}Q{k % 4 + 1}" for k in range(n_seasons)], n)
        x = rng.normal(size=len(g))
        y = slope * x + rng.normal(size=len(g))
        return g, x, y

    def test_season_ic_needs_enough_rows(self):
        g, x, y = self.data(n=10)
        gs, d = pt.season_ic(g, x, y)
        self.assertEqual(len(d), 0)                                    # under MIN_N a season
        gs, d = pt.season_ic(g, x, y, min_n=10)
        self.assertEqual(len(d), 12)

    def test_a_real_relation_passes_with_its_controls(self):
        g, x, y = self.data()
        m = pt.ic_test(g, x, y, np.random.default_rng(1), n_placebo=40)
        self.assertGreater(m["stat"], 0.2)
        self.assertLess(m["p"], 0.01)
        self.assertTrue(m["controls"]["pass"])
        self.assertEqual(m["seasons"], 12)

    def test_no_relation_does_not(self):
        g, x, y = self.data(slope=0.0)
        m = pt.ic_test(g, x, y, np.random.default_rng(1), n_placebo=40)
        self.assertGreater(m["p"], 0.05)

    def test_missing_values_are_dropped(self):
        g, x, y = self.data()
        x[::7] = np.nan
        m = pt.ic_test(g, x, y, np.random.default_rng(1), n_placebo=10)
        self.assertEqual(m["rows"], int(np.isfinite(x).sum()))


class Reports(unittest.TestCase):
    def test_the_habit_s_ranges(self):
        rows = [{"usual": 0.02, "surprise": 0.02 + e, "grade": "met"} for e in np.linspace(-0.10, 0.10, 41)]
        rows.append({"usual": None, "surprise": 0.5, "grade": "beat"})
        h = pt.habit_report(rows)
        self.assertEqual(h["rows"], 41)
        want = np.mean([(pr.BAND68[0] <= e <= pr.BAND68[1]) for e in np.linspace(-0.10, 0.10, 41)])
        self.assertAlmostEqual(h["in68"], float(want))
        self.assertLessEqual(h["in68"], h["in90"])

    def test_by_what_the_release_said(self):
        Q = [{"got_grade": "beat", "r1": 0.04, "post5": 0.01}, {"got_grade": "beat", "r1": -0.02, "post5": None},
             {"got_grade": "missed", "r1": -0.08, "post5": -0.01}, {"got_grade": None, "r1": 0.5, "post5": 0.0}]
        Y = [{"changed": "cut", "r1": -0.05, "post5": 0.0}]
        out = pt.by_what(Q, Y)
        self.assertEqual(out["delivered"]["beat"]["n"], 2)
        self.assertAlmostEqual(out["delivered"]["beat"]["r1"], 0.01)
        self.assertAlmostEqual(out["delivered"]["beat"]["rose"], 0.5)
        self.assertEqual(out["year"]["cut"]["n"], 1)
        self.assertNotIn("met", out["delivered"])
        self.assertEqual(out["habit"], {})
        Q = [{"got": 0.01, "usual": 0.03, "r1": -0.04, "post5": 0.0}, {"got": 0.05, "usual": 0.03, "r1": 0.02, "post5": 0.0},
             {"got": 0.05, "usual": None, "r1": 0.5, "post5": 0.0}]
        out = pt.by_what(Q, [])
        self.assertEqual((out["habit"]["above"]["n"], out["habit"]["below"]["n"]), (1, 1))
        self.assertAlmostEqual(out["habit"]["below"]["r1"], -0.04)


class Evaluate(unittest.TestCase):
    def test_on_a_synthetic_library(self):
        rng = np.random.default_rng(3)
        lib, eps = [], []
        for c in range(60):
            habit = rng.normal(0.02, 0.02)
            rows = company_rows(f"C{c:02d}", [habit + rng.normal(0, 0.005) for _ in range(26)], start_year=2020)
            for r in rows:
                r["stretch"] = float(rng.normal())
                eps.append({"t": r["t"], "date": r["made"], "pre": r["made"], "r1": float(rng.normal(0, 0.03)),
                            "post5": float(rng.normal(0, 0.02))})
            lib += rows
        out = pt.evaluate(lib, eps, rng=np.random.default_rng(1))
        self.assertGreater(out["G1a"]["stat"], 0.3)                     # the habit was planted
        self.assertLess(out["G1a"]["p"], 0.01)
        self.assertGreater(out["G1b"]["p"], 0.01)                       # the stretch was noise
        self.assertEqual(out["G1c"]["seasons"], 0)                      # no yearly promises here
        self.assertGreater(out["G1a"]["habit"]["rows"], 100)
        self.assertLess(out["G1a"]["habit"]["miss_habit"], out["G1a"]["habit"]["miss_midpoint"])
        self.assertIn("quarterly promises", pt.words("G1a", out["G1a"]))
        self.assertIn("releases", pt.words("G1d", out["G1d"]))

    def test_the_spec_pins_the_code_and_the_ranges(self):
        s = pt.spec("G1a")
        self.assertEqual(s["band68"], list(pr.BAND68))
        self.assertEqual(s["code_sha"], pt.code_sha())
        self.assertIsNone(pt.spec("G1d")["gate"])
        self.assertNotEqual(pt.spec_sha(s), pt.spec_sha(pt.spec("G1b")))


if __name__ == "__main__":
    unittest.main()
