"""The moonshot watch (brain/moonshots.py): the lists, a scan judged, the scoreboard's words."""
from __future__ import annotations

import unittest

from brain import moonshots as ms


def rec(asof="2026-12-04", n=40):
    """A recorded scan of ``n`` stocks, S00 the likeliest; the last one has no chance."""
    rows = {f"S{k:02d}": [-0.1, -0.05, 0.05, 0.1, round(0.30 - 0.005 * k, 4)] for k in range(n)}
    rows["NOCH"] = [-0.1, -0.05, 0.05, 0.1, None]
    return {"asof": asof, "rows": rows}


class Lists(unittest.TestCase):
    def test_cut_is_a_count_or_a_share(self):
        self.assertEqual(ms.cut(2452, 25), 25)
        self.assertEqual(ms.cut(10, 25), 10)
        self.assertEqual(ms.cut(2452, 0.05), 123)
        self.assertEqual(ms.cut(40, 0.20), 8)

    def test_ranked_by_chance_without_the_stocks_that_have_none(self):
        r = ms.ranked(rec())
        self.assertEqual([t for t, _ in r[:3]], ["S00", "S01", "S02"])
        self.assertEqual(len(r), 40)
        tie = {"asof": "x", "rows": {"B": [0, 0, 0, 0, 0.1], "A": [0, 0, 0, 0, 0.1]}}
        self.assertEqual([t for t, _ in ms.ranked(tie)], ["A", "B"])

    def test_a_week_end_is_the_last_session_of_its_week(self):
        cal = ["2026-11-25", "2026-11-27", "2026-11-30", "2026-12-01"]
        self.assertTrue(ms.week_end("2026-11-27", cal))             # Friday; the next session is Monday
        self.assertFalse(ms.week_end("2026-11-25", cal))
        self.assertFalse(ms.week_end("2026-12-01", cal))            # the latest session: not known yet


class Judge(unittest.TestCase):
    def setUp(self):
        # S00 jumps (+40%), S01 crashes (-30%), S30 jumps outside every small list; the rest drift down 1%
        self.moves = {f"S{k:02d}": (-0.01, False, False) for k in range(40)}
        self.moves.update(S00=(0.40, True, False), S01=(-0.30, False, True), S30=(0.25, True, False))
        self.moves.pop("S39")                                       # no whole window: left out

    def test_a_scan_against_what_happened(self):
        j = ms.judge_scan(rec(), self.moves)
        self.assertEqual((j["n"], j["jumps"], j["crashes"]), (39, 2, 1))
        top = j["lists"]["top 5%"]                                  # 2 of 39
        self.assertEqual((top["n"], top["jumps"], top["crashes"]), (2, 1, 1))
        self.assertAlmostEqual(top["mean"], 0.05)                   # +40% and -30%
        self.assertAlmostEqual(top["rose"], 0.5)
        wide = j["lists"]["top 20%"]                                # 8 of 39
        self.assertEqual((wide["n"], wide["jumps"]), (8, 1))
        self.assertAlmostEqual(wide["median"], -0.01)
        self.assertEqual(j["lists"]["top 25"]["n"], 25)

    def test_the_scoreboard_says_nothing_under_thirty_weekly_readings(self):
        a = {**ms.judge_scan(rec("2026-12-04"), self.moves), "weekly": True}
        b = {**ms.judge_scan(rec("2026-12-07"), self.moves), "weekly": False}
        old = {**ms.judge_scan(rec("2026-11-13"), self.moves), "weekly": True}      # before the reading was fixed
        text = "\n".join(ms.score_lines({"scans": [old, a, b], "open": 2}))
        self.assertIn("Recorded scans judged: 3 (2 still open)", text)
        self.assertIn("Weekly readings from 2026-12-04 (1)", text)
        self.assertIn("held 1 of 2 jumps (50%) and 1 of 1 crashes", text)
        self.assertIn("Under 30 weekly readings this record says nothing yet", text)
        self.assertIn("not a direction", text)
        m = ms.summary([a, b], "top 5%")
        self.assertEqual((m["n"], m["jumps"], m["all_jumps"], m["crashes"]), (2, 2, 4, 2))
        self.assertAlmostEqual(m["mean"], 0.05)
        self.assertIn("No recorded scan has closed", ms.score_lines({"scans": [], "open": 3})[0])


if __name__ == "__main__":
    unittest.main()
