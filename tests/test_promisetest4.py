"""The second guess of a delivery (brain/research/promisetest4.py; promises.guess2): the rows, the statistics and
the report."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from brain import promises as pr
from brain.research import promisetest as pt
from brain.research import promisetest4 as p4
from brain.research import run
from brain.research.forecast_tests import Refused, paths
from tests.test_promisetest import company_rows, promise
from tests.util import TempHome


def planted(seed=3, companies=80, pull=True, quarters=16, start_year=2023):
    """Companies with a level of their own. With ``pull`` the next delivery is what PULL says of the habit plus
    noise (a large habit overshoots); without it the next delivery is the habit itself plus a little noise."""
    rng = np.random.default_rng(seed)
    lib = []
    for c in range(companies):
        level = rng.normal(0.03, 0.04)
        s = []
        for k in range(quarters):
            if k < 4:
                s.append(level + rng.normal(0, 0.02))
            else:
                h = float(np.median(s[-4:]))
                s.append(pr.PULL[0] + pr.PULL[1] * h + rng.normal(0, 0.01) if pull else h + rng.normal(0, 0.002))
        lib += company_rows(f"C{c:02d}", s, start_year=start_year)
    return lib


class Guess(unittest.TestCase):
    def test_the_habit_pulled_in_and_ranges_by_the_spread(self):
        g = pr.guess2(0.16, 0.08)
        self.assertAlmostEqual(g["usual"], 0.09)                       # +1.0% + 0.50 x 16%
        self.assertAlmostEqual(g["lo68"], 0.09 - 0.02 - 0.5 * 0.08)
        self.assertAlmostEqual(g["hi68"], 0.09 + 0.0175 + 0.6 * 0.08)
        self.assertAlmostEqual(g["lo90"], 0.09 - 0.04 - 0.9 * 0.08)
        self.assertAlmostEqual(g["hi90"], 0.09 + 0.0375 + 1.1 * 0.08)
        steady, wild = pr.guess2(0.02, 0.002), pr.guess2(0.02, 0.06)
        self.assertLess(steady["hi68"] - steady["lo68"], wild["hi68"] - wild["lo68"])
        self.assertAlmostEqual(pr.guess2(-0.02, 0.0)["usual"], 0.0)    # a negative habit is pulled up

    def test_the_spread_is_that_of_the_habits_surprises(self):
        rows = company_rows("AAA", [0.01, 0.03, 0.05, 0.07, 0.20], start_year=2020)
        made = rows[4]["made"]
        self.assertAlmostEqual(pr.habit(rows, made), 0.04)
        self.assertAlmostEqual(pr.spread(rows, made), float(np.std([0.01, 0.03, 0.05, 0.07])))
        self.assertIsNone(pr.spread(rows, rows[3]["made"]))            # three deliveries: none
        self.assertEqual(pr.recent(rows, made), [0.01, 0.03, 0.05, 0.07])

    def test_the_card_s_expectation_carries_the_second_guess(self):
        rows = company_rows("AAA", [0.10, 0.12, 0.14, 0.16], start_year=2020) + [promise("AAA", "2021-02-10", None)]
        e = pr.expected(rows, "2021-03-01T00:00:00Z")
        self.assertAlmostEqual(e["point"], 113.0)                      # the plain habit: +13%
        self.assertAlmostEqual(e["second"]["usual"], 0.01 + 0.5 * 0.13)
        self.assertAlmostEqual(e["second"]["point"], 100 * (1 + 0.075))
        self.assertLess(e["second"]["lo90"], e["second"]["lo68"])
        self.assertLess(e["second"]["hi68"], e["second"]["hi90"])


class Rows(unittest.TestCase):
    def test_a_row_needs_a_habit_and_a_promise_made_inside_the_span(self):
        lib = company_rows("AAA", [0.02] * 8, start_year=2023)         # made 2023-02 .. 2024-11
        got = p4.deliveries(lib)
        self.assertEqual([r["made"] for r in got], ["2024-02-10", "2024-05-10", "2024-08-10", "2024-11-10"])
        self.assertEqual(got[0]["season"], "2024Q1")
        self.assertAlmostEqual(got[0]["habit"], 0.02)
        self.assertAlmostEqual(got[0]["spread"], 0.0)
        self.assertAlmostEqual(got[0]["g"]["usual"], 0.02)             # +1% + half of 2%
        self.assertEqual(len(p4.deliveries(lib, span=("2023-01-01", "2023-12-31"))), 0)   # no habit yet in 2023
        self.assertEqual(p4.deliveries(lib, tickers={"ZZZ"}), [])

    def test_the_last_promise_for_a_quarter_counts_and_a_mismatch_does_not(self):
        lib = company_rows("AAA", [0.02] * 6, start_year=2023)
        again = dict(lib[5], made="2024-06-01", known_at="2024-06-01T12:00:00Z", surprise=0.09)
        got = p4.deliveries(lib + [again])
        self.assertEqual([(r["made"], r["s"]) for r in got], [("2024-02-10", 0.02), ("2024-06-01", 0.09)])
        bad = dict(lib[5], mismatch=True)
        bad.pop("grade")
        self.assertEqual([r["made"] for r in p4.deliveries(lib[:5] + [bad])], ["2024-02-10"])

    def test_the_universe_is_the_four_together(self):
        sets = {"us_broad": {"B"}, "us_small": {"A", "B"}, "us_micro": {"C"}, "us_watch": {"D"}}
        with mock.patch.object(p4, "_tickers", lambda n: sets[n]):
            self.assertEqual(p4.universe(), ["A", "B", "C", "D"])


class Statistics(unittest.TestCase):
    def test_where_large_habits_overshoot_the_pulled_in_guess_wins(self):
        with mock.patch.object(pt, "N_PLACEBO", 40), mock.patch.object(pt, "N_FLIP", 2000), \
                mock.patch.object(pt, "N_BOOT", 300):
            m = p4.evaluate(planted(pull=True))
        a = m["G4a"]
        self.assertGreater(a["stat"], 0)
        self.assertLess(a["p"], 0.01)
        self.assertEqual(a["seasons_positive"], a["seasons"])
        self.assertLess(a["mean_second"], a["mean_plain"])
        self.assertTrue(a["controls"]["pass"])
        self.assertLessEqual(a["controls"]["placebo_fpr"], 0.15)
        self.assertIn("in its favour", p4.words("G4a", a))
        self.assertIn("2-in-3 range held", p4.words("G4b", m["G4b"]))

    def test_where_the_plain_habit_is_right_it_does_not(self):
        with mock.patch.object(pt, "N_PLACEBO", 20), mock.patch.object(pt, "N_FLIP", 2000), \
                mock.patch.object(pt, "N_BOOT", 300):
            a = p4.evaluate(planted(pull=False))["G4a"]
        self.assertLess(a["stat"], 0)
        self.assertGreater(a["p"], 0.5)

    def test_paired_test_is_per_season(self):
        rng = np.random.default_rng(0)
        g = np.array(["2024Q1"] * 30 + ["2024Q2"] * 30 + ["2024Q3"] * 5)          # the third is too small
        d = np.concatenate([np.full(30, 0.01), np.full(30, 0.03), np.full(5, -9.0)])
        with mock.patch.object(pt, "N_FLIP", 500), mock.patch.object(pt, "N_BOOT", 100):
            m = p4.paired_test(g, d, rng)
        self.assertEqual((m["seasons"], m["seasons_positive"]), (2, 2))
        self.assertAlmostEqual(m["stat"], 0.02)

    def test_coverage_by_the_company_s_own_spread(self):
        def row(s, habit, spread):
            return {"s": s, "habit": habit, "spread": spread, "g": pr.guess2(habit, spread)}
        rows = [row(0.02, 0.02, 0.0), row(0.20, 0.02, 0.005), row(0.02, 0.02, 0.03), row(-0.30, 0.02, 0.10)]
        c = p4.coverage(rows)
        self.assertEqual(c["all"]["n"], 4)
        self.assertAlmostEqual(c["all"]["in68"], 0.5)
        self.assertEqual([b["n"] for b in c["by_spread"]], [2, 1, 1])              # 0-1%, 2-5%, over 5%
        self.assertAlmostEqual(c["by_spread"][0]["in68"], 0.5)
        self.assertAlmostEqual(c["by_spread"][0]["fixed68"], 0.5)
        self.assertEqual(p4.coverage([])["all"], None)


class TrialRun(TempHome):
    def test_declare_then_test_writes_the_ledger_and_the_report(self):
        lib = planted(companies=60)
        names = sorted({r["t"] for r in lib})
        with mock.patch.object(p4, "universe", lambda: names), mock.patch.object(pt, "N_PLACEBO", 20), \
                mock.patch.object(p4, "universe_sha", lambda: "x" * 16):
            with self.assertRaises(Refused):
                p4.test(self.bank, log=None, lib=lib)                             # not declared yet
            self.assertEqual(len(p4.declare(log=None)), 2)
            self.assertEqual(p4.declare(log=None), [])                            # declared once
            out = p4.test(self.bank, log=None, lib=lib)
            self.assertEqual([r["id"] for r in out], ["G4a", "G4b"])
            self.assertEqual(p4.test(self.bank, log=None, lib=lib), [])           # run once
        by = {r["id"]: r for r in out}
        self.assertTrue(by["G4a"]["gate"]["pass"])
        self.assertEqual(by["G4b"]["gate_type"], "report")
        self.assertIsNone(by["G4b"]["p"])
        led, res = paths(True)
        text = res.read_text()
        self.assertIn("| G4a |", text)
        self.assertIn("| G4b |", text)
        got = {r["id"]: r["verdict"] for r in run.verdicts(led) if r.get("kind") == "promise"}
        self.assertEqual(got["G4a"], "PASS")
        self.assertEqual(got["G4b"], "REPORT")


if __name__ == "__main__":
    unittest.main()
