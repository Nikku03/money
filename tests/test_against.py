"""The engine against the analysts (brain/against.py): a line before the report, judged after it."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from brain import against
from tests.test_promisetest import company_rows, promise

NOW = "2021-08-20T12:00:00Z"


def rows(habit=0.02):
    """Six delivered quarters of 2020-2021 (each +habit), then an open promise made 2021-08-10 (midpoint 100)."""
    done = company_rows("AAA", [habit] * 6, start_year=2020)        # the last made 2021-05-10, to 2021-06-28
    return done + [promise("AAA", "2021-08-10", None)]


class Line(unittest.TestCase):
    def test_the_habit_s_guess_beside_the_analysts(self):
        c = against.line(rows(), "AAA", 100.5, "2021-11-09", n=7, source="a page", now=NOW)
        self.assertEqual(c["id"], "AAA-2021-08-10")
        self.assertAlmostEqual(c["point"], 102.0)
        self.assertAlmostEqual(c["gap"], 102.0 / 100.5 - 1)
        self.assertEqual((c["analysts_n"], c["source"], c["made_at"]), (7, "a page", NOW))
        self.assertLess(c["lo68"], c["point"])
        self.assertIn("the analysts expect $0.0 mn (7 analysts)", against.words(c))

    def test_refused_once_the_report_date_has_come_or_without_a_habit_or_with_wrong_units(self):
        with self.assertRaises(SystemExit):
            against.line(rows(), "AAA", 100.5, "2021-08-20", now=NOW)            # today is not ahead
        with self.assertRaises(SystemExit):
            against.line(rows()[3:], "AAA", 100.5, "2021-11-09", now=NOW)        # three deliveries: no habit
        with self.assertRaises(SystemExit):
            against.line(rows()[:-1], "AAA", 100.5, "2021-11-09", now=NOW)       # no open promise
        with self.assertRaises(SystemExit):
            against.line(rows(), "AAA", 100.5e6, "2021-11-09", now=NOW)          # dollars against millions
        with self.assertRaises(ValueError):
            against.line(rows(), "AAA", 100.5, "next week", now=NOW)

    def test_a_line_is_written_once(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calls.jsonl"
            c = against.line(rows(), "AAA", 100.5, "2021-11-09", now=NOW)
            against.add(c, p)
            with self.assertRaises(SystemExit):
                against.add(c, p)
            self.assertEqual([r["id"] for r in against.read_calls(p)], ["AAA-2021-08-10"])
            self.assertEqual(against.read_calls(Path(d) / "none.jsonl"), [])


class Judge(unittest.TestCase):
    def setUp(self):
        self.call = against.line(rows(), "AAA", 100.5, "2021-11-09", now=NOW)
        self.done = rows()[:-1] + [promise("AAA", "2021-08-10", 0.025, period_end="2021-09-28",
                                           outcome_known="2021-11-12T12:00:00Z")]

    def test_waiting_until_the_revenue_is_on_file(self):
        j, = against.judge(rows(), [self.call])
        self.assertIsNone(j["actual"])
        s = against.score(rows(), path=Path("/nonexistent/none.jsonl"), eps=[])
        self.assertEqual((s["recorded"], s["judged"]), (0, 0))
        self.assertIn("Nothing judged yet", " ".join(against.score_words(s)))

    def test_who_was_closer(self):
        eps = [{"t": "AAA", "date": "2021-11-10", "known_at": "2021-11-09T21:05:00Z", "r1": 0.03}]
        j, = against.judge(self.done, [self.call], eps)
        self.assertTrue(j["valid"])
        self.assertAlmostEqual(j["actual"], 102.5)
        self.assertEqual(j["closer"], "engine")                   # 102.0 against 100.5, delivered 102.5
        self.assertTrue(j["beat_analysts"] and j["in68"] and j["in90"])
        self.assertEqual(j["r1"], 0.03)
        low = dict(self.call, analysts=102.4, gap=102.0 / 102.4 - 1)
        self.assertEqual(against.judge(self.done, [low], eps)[0]["closer"], "analysts")
        same = dict(self.call, analysts=103.0)                    # both 0.5 away
        self.assertEqual(against.judge(self.done, [same], eps)[0]["closer"], "tie")

    def test_a_line_written_after_the_release_is_void(self):
        eps = [{"t": "AAA", "date": "2021-08-19", "known_at": "2021-08-18T21:05:00Z", "r1": 0.5},   # not its quarter
               {"t": "AAA", "date": "2021-11-10", "known_at": "2021-11-09T21:05:00Z", "r1": 0.03}]
        late = dict(self.call, made_at="2021-11-10T09:00:00Z")
        j, = against.judge(self.done, [late], eps)
        self.assertFalse(j["valid"])
        j, = against.judge(self.done, [dict(self.call, made_at="2021-11-13T00:00:00Z")], [])   # no episode: the accounts
        self.assertFalse(j["valid"])

    def test_score_counts_only_valid_lines_and_says_nothing_under_thirty(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calls.jsonl"
            against.add(self.call, p)
            against.add(dict(self.call, id="AAA-void", made_at="2021-11-20T00:00:00Z"), p)
            s = against.score(self.done, p, [])
        self.assertEqual((s["recorded"], s["judged"], s["void"]), (2, 1, 1))
        self.assertEqual((s["engine_closer"], s["analysts_closer"], s["beat_analysts"]), (1.0, 0.0, 1.0))
        text = " ".join(against.score_words(s))
        self.assertIn("void 1", text)
        self.assertIn("Under 30 judged lines this says nothing yet", text)


if __name__ == "__main__":
    unittest.main()
