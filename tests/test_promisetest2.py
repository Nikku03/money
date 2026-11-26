"""The promise tests on the other universes (brain/research/promisetest2.py): the statistics and the report."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from brain.research import promisetest as pt
from brain.research import promisetest2 as p2
from brain.research import run
from brain.research.forecast_tests import Refused, paths
from tests.test_promisetest import company_rows
from tests.util import TempHome


def planted(seed=4, companies=60, drift=0.0):
    """Companies with a habit that carries; a stretch that adds to it; a stock that follows the delivery against
    the habit by ``drift`` over the five sessions after."""
    rng = np.random.default_rng(seed)
    lib, eps = [], []
    for c in range(companies):
        habit = rng.normal(0.02, 0.02)
        rows = company_rows(f"C{c:02d}", [0.0] * 26, start_year=2020)
        for r in rows:
            r["stretch"] = float(rng.normal())
            s = habit + 0.004 * r["stretch"] + rng.normal(0, 0.004)
            r.update(surprise=float(s), actual=100.0 * (1 + s), grade="beat" if s > 0.05 else "met")
        for k, r in enumerate(rows):
            got = rows[k - 1]["surprise"] - habit if k else 0.0
            eps.append({"t": r["t"], "date": r["made"], "pre": r["made"], "r1": float(rng.normal(0, 0.03)),
                        "post5": float(drift * got + rng.normal(0, 0.02))})
        lib += rows
    return lib, eps


class Statistics(unittest.TestCase):
    def test_the_universe_leaves_the_research_universe_out(self):
        sets = {"us_small": {"A", "B"}, "us_micro": {"C"}, "us_watch": {"D", "B"}, "us_broad": {"B", "Z"}}
        with mock.patch.object(p2, "_tickers", lambda n: sets[n]):
            self.assertEqual(p2.universe(), ["A", "C", "D"])

    def test_a_habit_and_a_stretch_that_carry_and_a_stock_that_does_not_follow(self):
        lib, eps = planted()
        with mock.patch.object(pt, "N_PLACEBO", 20), mock.patch.object(pt, "N_FLIP", 2000), \
                mock.patch.object(pt, "N_BOOT", 300):
            m = p2.evaluate(lib, eps)
        self.assertGreater(m["G2a"]["stat"], 0.5)
        self.assertLess(m["G2a"]["p"], 0.01)
        self.assertGreater(m["G2b"]["stat"], 0.3)             # beyond the habit, asking for more delivered more
        self.assertLess(m["G2b"]["p"], 0.01)
        self.assertGreater(m["G2c"]["p"], 0.05)               # the stock did not follow
        self.assertEqual(m["G2d"]["rows"], 0)                 # no yearly promises in these rows
        self.assertIn("surprise less the habit", p2.words("G2b", m["G2b"]))
        self.assertIn("next five sessions", p2.words("G2c", m["G2c"]))

    def test_a_stock_that_follows_the_delivery(self):
        lib, eps = planted(drift=3.0)
        with mock.patch.object(pt, "N_PLACEBO", 10), mock.patch.object(pt, "N_FLIP", 2000), \
                mock.patch.object(pt, "N_BOOT", 300):
            m = p2.evaluate(lib, eps)
        self.assertGreater(m["G2c"]["stat"], 0.2)
        self.assertLess(m["G2c"]["p"], 0.01)

    def test_after_a_change_of_the_years_promise(self):
        y = [{"changed": "raised", "post5": 0.02}, {"changed": "raised", "post5": -0.01},
             {"changed": "cut", "post5": -0.03}, {"changed": "held", "post5": None}]
        got = p2.after_change(y)
        self.assertEqual(got["raised"]["n"], 2)
        self.assertAlmostEqual(got["raised"]["post5"], 0.005)
        self.assertEqual(got["cut"]["rose"], 0.0)
        self.assertNotIn("held", got)


class TrialRun(TempHome):
    def test_declare_then_test_writes_the_ledger_and_the_report(self):
        lib, eps = planted(companies=50)
        names = sorted({r["t"] for r in lib})
        with mock.patch.object(p2, "universe", lambda: names), mock.patch.object(pt, "N_PLACEBO", 20), \
                mock.patch.object(p2, "universe_sha", lambda: "x" * 16):
            with self.assertRaises(Refused):
                p2.test(self.bank, log=None, lib=lib, eps=eps)                # not declared yet
            self.assertEqual(len(p2.declare(log=None)), 4)
            self.assertEqual(p2.declare(log=None), [])                        # declared once
            out = p2.test(self.bank, log=None, lib=lib, eps=eps)
            self.assertEqual([r["id"] for r in out], list(p2.IDS))
            self.assertEqual(p2.test(self.bank, log=None, lib=lib, eps=eps), [])   # run once
        by = {r["id"]: r for r in out}
        self.assertTrue(by["G2a"]["gate"]["pass"])
        self.assertEqual(by["G2c"]["gate_type"], "drift")
        led, res = paths(True)
        text = res.read_text()
        self.assertIn("## Promises: said and delivered", text)
        self.assertIn("| G2a |", text)
        self.assertIn("| G2d |", text)
        got = {r["id"]: r["verdict"] for r in run.verdicts(led) if r.get("kind") == "promise"}
        self.assertEqual(got["G2a"], "PASS")
        self.assertIn(got["G2c"], ("FAIL", "FAIL (controls)", "PASS"))


if __name__ == "__main__":
    unittest.main()
