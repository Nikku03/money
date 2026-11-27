"""The promise that asks for more, against the report session that delivers it (brain/research/promisetest3.py)."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from brain.research import promisetest as pt
from brain.research import promisetest3 as p3
from brain.research import run
from brain.research.forecast_tests import Refused, paths
from tests.test_promisetest import company_rows
from tests.util import TempHome


def planted(seed=5, companies=60, slope=0.0, quarters=12, start_year=2024):
    """Companies whose report session moves by ``slope`` times the stretch of the promise it delivers. A promise is
    made on the 10th of a quarter's second month and delivered by the next release, 43 to 46 days after the
    quarter's end."""
    rng = np.random.default_rng(seed)
    lib, eps = [], []
    for c in range(companies):
        rows = company_rows(f"C{c:02d}", [0.0] * quarters, start_year=start_year)
        for r in rows:
            r["stretch"] = float(rng.normal())
        for k in range(1, len(rows)):
            eps.append({"t": rows[k]["t"], "date": rows[k]["made"], "pre": rows[k]["made"],
                        "r1": float(slope * rows[k - 1]["stretch"] + rng.normal(0, 0.03))})
        lib += rows
    return lib, eps


class Deliveries(unittest.TestCase):
    def test_a_quarter_is_delivered_by_the_first_report_session_after_its_end(self):
        lib = company_rows("AAA", [0.0, 0.0, 0.0], start_year=2024)        # ends 03-28, 06-28, 09-28
        for r, s in zip(lib, (0.1, -0.2, 0.3)):
            r["stretch"] = s
        eps = [{"t": "AAA", "date": "2024-03-30", "r1": 0.5},              # 2 days after the end: too soon
               {"t": "AAA", "date": "2024-05-10", "r1": 0.01},             # delivers the first quarter
               {"t": "AAA", "date": "2024-08-10", "r1": -0.02},            # the second
               {"t": "AAA", "date": "2025-02-10", "r1": 0.09}]             # 135 days after the third: too late
        got = p3.deliveries(lib, eps)
        self.assertEqual([(r["period_end"], r["date"], r["stretch"], r["r1"]) for r in got],
                         [("2024-03-28", "2024-05-10", 0.1, 0.01), ("2024-06-28", "2024-08-10", -0.2, -0.02)])
        self.assertEqual(got[0]["season"], "2024Q2")

    def test_the_last_promise_for_a_quarter_counts_and_late_or_mismatched_ones_do_not(self):
        a, = company_rows("AAA", [0.0], start_year=2024)                    # made 02-10 for the quarter to 03-28
        a["stretch"] = 0.1
        again = dict(a, made="2024-03-01", known_at="2024-03-01T12:00:00Z", stretch=0.4)
        after = dict(a, made="2024-04-02", known_at="2024-04-02T12:00:00Z", stretch=0.9)   # the quarter had ended
        eps = [{"t": "AAA", "date": "2024-05-10", "r1": 0.01}]
        got = p3.deliveries([a, again, after], eps)
        self.assertEqual([r["stretch"] for r in got], [0.4])
        self.assertEqual(p3.deliveries([dict(a, mismatch=True)], eps), [])
        self.assertEqual(p3.deliveries([a], eps, tickers={"ZZZ"}), [])
        self.assertEqual(p3.deliveries([a], [{"t": "AAA", "date": "2024-05-10", "r1": None}]), [])

    def test_the_universe_is_the_four_together(self):
        sets = {"us_broad": {"B", "Z"}, "us_small": {"A", "B"}, "us_micro": {"C"}, "us_watch": {"D"}}
        with mock.patch.object(p3, "_tickers", lambda n: sets[n]):
            self.assertEqual(p3.universe(), ["A", "B", "C", "D", "Z"])


class Statistics(unittest.TestCase):
    def test_a_session_that_follows_the_stretch(self):
        lib, eps = planted(slope=0.01)
        with mock.patch.object(pt, "N_PLACEBO", 20), mock.patch.object(pt, "N_FLIP", 2000), \
                mock.patch.object(pt, "N_BOOT", 300):
            m = p3.evaluate(lib, eps)["G3a"]
        self.assertGreater(m["stat"], 0.2)
        self.assertLess(m["p"], 0.01)
        self.assertEqual(m["companies"], 60)
        self.assertGreater(m["thirds"]["most"]["mean"], m["thirds"]["least"]["mean"])
        self.assertIn("report session that delivered the quarter", p3.words("G3a", m))
        self.assertIn("the third asking for most", p3.words("G3a", m))

    def test_a_session_that_does_not(self):
        lib, eps = planted(slope=0.0)
        with mock.patch.object(pt, "N_PLACEBO", 20), mock.patch.object(pt, "N_FLIP", 2000), \
                mock.patch.object(pt, "N_BOOT", 300):
            m = p3.evaluate(lib, eps)["G3a"]
        self.assertGreater(m["p"], 0.05)

    def test_only_sessions_inside_the_span_count(self):
        lib, eps = planted(slope=0.01, quarters=8, start_year=2022)         # sessions of 2022-05 to 2023-11
        with mock.patch.object(pt, "N_PLACEBO", 5), mock.patch.object(pt, "N_FLIP", 500), \
                mock.patch.object(pt, "N_BOOT", 100):
            self.assertEqual(p3.evaluate(lib, eps)["G3a"]["rows"], 0)
            self.assertGreater(p3.evaluate(lib, eps, span=p3.TRAIN)["G3a"]["rows"], 300)

    def test_thirds_are_taken_inside_each_season(self):
        rows = [{"season": "2024Q2", "stretch": s, "r1": r} for s, r in
                ((-3, -0.02), (-2, -0.01), (-1, 0.0), (0, 0.0), (1, 0.01), (2, 0.0), (3, 0.02), (4, 0.03), (5, 0.04))]
        rows += [{"season": "2024Q3", "stretch": 9.0, "r1": 0.5}, {"season": "2024Q3", "stretch": None, "r1": 0.5}]
        got = p3.thirds(rows)
        self.assertEqual((got["least"]["n"], got["middle"]["n"], got["most"]["n"]), (3, 3, 3))   # 2024Q3 too small
        self.assertAlmostEqual(got["least"]["mean"], -0.01)
        self.assertAlmostEqual(got["most"]["rose"], 1.0)


class TrialRun(TempHome):
    def test_declare_then_test_writes_the_ledger_and_the_report(self):
        lib, eps = planted(companies=50, slope=0.01)
        names = sorted({r["t"] for r in lib})
        with mock.patch.object(p3, "universe", lambda: names), mock.patch.object(pt, "N_PLACEBO", 20), \
                mock.patch.object(p3, "universe_sha", lambda: "x" * 16):
            with self.assertRaises(Refused):
                p3.test(self.bank, log=None, lib=lib, eps=eps)                # not declared yet
            self.assertEqual(len(p3.declare(log=None)), 1)
            self.assertEqual(p3.declare(log=None), [])                        # declared once
            out = p3.test(self.bank, log=None, lib=lib, eps=eps)
            self.assertEqual([r["id"] for r in out], ["G3a"])
            self.assertEqual(p3.test(self.bank, log=None, lib=lib, eps=eps), [])   # run once
        self.assertTrue(out[0]["gate"]["pass"])
        self.assertEqual(out[0]["gate_type"], "drift")
        led, res = paths(True)
        text = res.read_text()
        self.assertIn("## Promises: said and delivered", text)
        self.assertIn("| G3a |", text)
        got = {r["id"]: r["verdict"] for r in run.verdicts(led) if r.get("kind") == "promise"}
        self.assertEqual(got["G3a"], "PASS")


if __name__ == "__main__":
    unittest.main()
