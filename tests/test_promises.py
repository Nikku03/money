"""The promise record (brain/promises.py): reading revenue guidance, its horizon, its grade and the record."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from brain import promises as pr


def one(text):
    c = pr.candidates(text)
    return c[0] if c else None


class Reading(unittest.TestCase):
    def test_a_range(self):
        c = one("For the first quarter of 2020, Fortinet currently expects: • Revenue in the range of $555 million "
                "to $565 million")
        self.assertEqual((c["low"], c["high"], c["point"], c["words"]), (555e6, 565e6, False, "quarter"))

    def test_a_midpoint_with_a_tolerance(self):
        c = one("The Company is providing the following guidance for its fiscal second quarter 2026: •Revenue of "
                "$2.70 billion, plus or minus $100 million •Non-GAAP diluted EPS of $2.75")
        self.assertAlmostEqual(c["low"], 2.6e9)
        self.assertAlmostEqual(c["high"], 2.8e9)
        c = one("the following guidance for its fiscal second quarter 2020: • Revenue of $2.72 billion, plus or minus 5%")
        self.assertAlmostEqual(c["low"], 2.72e9 * 0.95)
        c = one("For the fourth quarter of fiscal 2023, we are forecasting revenue of $2.70 billion, +/- $100 million")
        self.assertAlmostEqual(c["high"], 2.8e9)

    def test_a_bullet_after_a_colon(self):
        c = one("NVIDIA's outlook for the first quarter of fiscal 2027 is as follows: • Revenue is expected to be "
                "$43.0 billion, plus or minus 2%. • GAAP gross margins are expected to be 71.8%")
        self.assertAlmostEqual(c["low"], 43.0e9 * 0.98)
        self.assertEqual(c["words"], "quarteryear")           # "fiscal 2027" sits beside it: the size decides

    def test_a_single_figure_only_when_a_forward_word_governs_it(self):
        c = one("The company now expects 2023 revenue to be $42.7 billion.")
        self.assertEqual((c["low"], c["point"], c["label"]), (42.7e9, True, 2023))
        c = one("•First quarter revenue guidance of approximately $7.6 billion")
        self.assertEqual(c["low"], 7.6e9)
        # a bullet of reported sales under an outlook heading is not a promise
        self.assertIsNone(one("RAISES 2019 ADJUSTED EPS AND FREE CASH FLOW OUTLOOK Strong sales and operating profit "
                              "drive performance in Q3; • Sales of $19.5 billion"))
        self.assertIsNone(one("Raises Midpoint of Full-Year EPS Guidance Range –Sales of $10.1 billion"))

    def test_parts_of_a_company_and_past_results_are_not_promises(self):
        self.assertIsNone(one("better than expected commercial bookings growth of 28% and Microsoft Cloud revenue of "
                              "$23.4 billion"))
        self.assertIsNone(one("with an expected industry-leading churn of 0.78% •Mobility service revenues of "
                              "$16.5 billion to $16.7 billion"))
        self.assertIsNone(one("Raises 2022 Revenue Guidance for Comirnaty to Approximately $32 billion"))
        self.assertIsNone(one("Revenue was $5.0 billion, above the high end of our guidance."))
        self.assertIsNone(one("for 2024 we expect revenue performance in line with our model and about $12 billion "
                              "in free cash flow"))
        self.assertIsNone(one("we expect cost of revenue of $2.0 billion to $2.1 billion"))
        self.assertIsNone(one("we expect sales growth of more than $15 billion"))


    def test_a_quarter_named_with_its_year(self):
        c = one("Sets Outlook for Full Year 2020 Moody's Corporation 4Q19 revenue of $1.2 billion")
        self.assertEqual(c["qlabel"], (4, 2019))
        self.assertTrue(pr.ended_before(c["qlabel"], "2020-02-12"))            # a result, not a promise
        c = one("For the first quarter of 2020, Fortinet currently expects: • Revenue in the range of $555 million "
                "to $565 million")
        self.assertEqual(c["qlabel"], (1, 2020))
        self.assertFalse(pr.ended_before(c["qlabel"], "2020-02-06"))
        self.assertEqual(pr.quarter_label("Q4 FY22: Revenue of", 0, 30), (4, 2022))
        self.assertEqual(pr.quarter_label("fourth quarter of fiscal 2023, we are forecasting", 0, 60), (4, 2023))
        self.assertIsNone(pr.quarter_label("next quarter revenue of", 0, 30))
        # a quarter named after it without a year governs the figure
        t = "Reports Fourth Quarter 2020 Results; Expects First Quarter Revenue of $500 million to $520 million"
        self.assertIsNone(one(t)["qlabel"])
        # the label of the sentence before does not reach this one
        c = one("Sales rose 5% from the fourth quarter of 2017. • We expect full year 2018 reported revenue to be "
                "between $3.425 billion and $3.525 billion")
        self.assertIsNone(c["qlabel"])
        self.assertFalse(pr.ended_before(None, "2020-02-06"))


class Horizon(unittest.TestCase):
    def test_by_words_and_size(self):
        q = {"low": 95.0, "high": 105.0, "words": "quarter"}
        self.assertEqual(pr.horizon(q, 98.0, 90.0, 380.0), "quarter")
        self.assertIsNone(pr.horizon({**q, "low": 28.0, "high": 32.0}, 98.0, 90.0, 380.0))     # a part of it
        y = {"low": 400.0, "high": 420.0, "words": "year"}
        self.assertEqual(pr.horizon(y, 98.0, 90.0, 380.0), "year")
        self.assertIsNone(pr.horizon({**y, "words": "quarter"}, 98.0, 90.0, 380.0))
        self.assertEqual(pr.horizon({**y, "words": ""}, 98.0, 90.0, 380.0), "year")           # the size decides
        self.assertEqual(pr.horizon({**q, "words": "quarteryear"}, 98.0, 90.0, 380.0), "quarter")

    def test_grade(self):
        self.assertEqual(pr.grade(106.0, 95.0, 105.0), "beat")
        self.assertEqual(pr.grade(105.2, 95.0, 105.0), "met")          # within the rounding tolerance
        self.assertEqual(pr.grade(94.0, 95.0, 105.0), "missed")


class Record(unittest.TestCase):
    def rows(self):
        out = []
        for k, (g, s) in enumerate([("beat", .05), ("met", .01), ("missed", -.04), ("met", .0), ("beat", .06)]):
            out.append({"t": "X", "horizon": "quarter", "grade": g, "surprise": s,
                        "outcome_known": f"2025-{k * 2 + 1:02d}-20T00:00:00Z"})
        return out

    def test_only_outcomes_known_by_then(self):
        rows = self.rows()
        self.assertIsNone(pr.record(rows, "2025-06-01T00:00:00Z"))            # three known: too few
        r = pr.record(rows, "2025-08-01T00:00:00Z")
        self.assertEqual(r["n"], 4)
        self.assertAlmostEqual(r["kept"], 0.75)
        self.assertEqual(r["last"], "met")
        self.assertEqual(pr.record(rows, "2026-01-01T00:00:00Z")["n"], 5)

    def test_changes(self):
        rows = [{"t": "X", "horizon": "year", "period_end": "2025-12-31", "mid": 100.0},
                {"t": "X", "horizon": "year", "period_end": "2025-12-31", "mid": 104.0},
                {"t": "X", "horizon": "year", "period_end": "2025-12-31", "mid": 98.0},
                {"t": "X", "horizon": "year", "period_end": "2026-12-31", "mid": 110.0}]
        pr.changes(rows)
        self.assertEqual([r.get("changed") for r in rows], [None, "raised", "cut", None])


class Expected(unittest.TestCase):
    def rows(self):
        out = []
        for k, s in enumerate([.02, .04, .03, .05]):
            out.append({"t": "X", "horizon": "quarter", "grade": "met", "surprise": s, "made": f"2025-{k * 3 + 1:02d}-15",
                        "known_at": f"2025-{k * 3 + 1:02d}-15T21:05:00Z", "mid": 100.0,
                        "outcome_known": f"2025-{k * 3 + 3:02d}-20T00:00:00Z"})
        out.append({"t": "X", "horizon": "quarter", "made": "2026-01-15", "known_at": "2026-01-15T21:05:00Z",
                    "mid": 200.0, "low": 190.0, "high": 210.0, "point": False})
        return out

    def test_the_habit_moves_the_midpoint(self):
        e = pr.expected(self.rows(), "2026-02-01T00:00:00Z")
        self.assertAlmostEqual(e["usual"], 0.035)
        self.assertAlmostEqual(e["point"], 207.0)
        self.assertAlmostEqual(e["lo68"], 200.0 * (1.035 + pr.BAND68[0]))
        self.assertLess(e["lo90"], e["lo68"])
        self.assertGreater(e["hi90"], e["hi68"])

    def test_none_without_a_record_or_with_a_stale_promise(self):
        rows = self.rows()
        self.assertIsNone(pr.expected(rows, "2026-01-01T00:00:00Z"))            # the open promise is not made yet
        self.assertIsNone(pr.expected(rows, "2026-09-01T00:00:00Z"))            # made more than OPEN_DAYS ago
        self.assertIsNone(pr.expected(rows[2:], "2026-02-01T00:00:00Z"))        # two outcomes: no record

    def test_after_the_close(self):
        self.assertTrue(pr.after_close("2026-11-24T21:05:00Z", "2026-11-24"))   # 4:05 p.m. in New York
        self.assertFalse(pr.after_close("2026-11-24T13:00:00Z", "2026-11-24"))  # before that day's open
        self.assertTrue(pr.after_close("2026-11-25T11:00:00Z", "2026-11-24"))
        self.assertTrue(pr.after_close("2026-07-21T20:05:00Z", "2026-07-21"))   # summer time: 4:05 p.m.
        self.assertFalse(pr.after_close("2026-07-21T19:30:00Z", "2026-07-21"))


class FirstReported(unittest.TestCase):
    def test_the_earliest_version_and_the_largest_concept(self):
        f = lambda attr, end, v, known: SimpleNamespace(attr=attr, as_of=end, value=v, known_at=known,
                                                        context={"start": "2024-01-01"})
        got = pr._first([f("Revenues", "2024-03-31", 100.0, "2024-05-01"),
                         f("Revenues", "2024-03-31", 60.0, "2025-05-01"),            # a later restatement
                         f("SalesRevenueNet", "2024-03-31", 20.0, "2024-05-01")])     # a part of the total
        self.assertEqual(got["2024-03-31"][:2], (100.0, "2024-05-01"))


if __name__ == "__main__":
    unittest.main()
