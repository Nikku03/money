"""Attention (brain/attention.py): the parts, the weights, the stray-print check and the reading helpers."""
from __future__ import annotations

import json
import unittest

import numpy as np

from brain import attention as at


class Parts(unittest.TestCase):
    def test_surprise(self):
        self.assertEqual(at.surprise(0.02, [-0.10, 0.10], None), 0.0)           # well inside its range
        self.assertAlmostEqual(at.surprise(0.10, [-0.10, 0.10], None), 1 / 3)   # at its edge
        self.assertEqual(at.surprise(-0.30, [-0.10, 0.10], None), 1.0)          # far beyond
        self.assertAlmostEqual(at.surprise(None, None, 1.5), 0.5)               # no range: usual days

    def test_unexplained(self):
        self.assertAlmostEqual(at.unexplained(-0.10, -0.08, None), 0.8)
        self.assertEqual(at.unexplained(-0.10, -0.08, "its own move, no news found"), 1.0)
        self.assertEqual(at.unexplained(None, None, None), 0.0)

    def test_news(self):
        s, why = at.news_score(["8-K 4.02"], 0)
        self.assertEqual(s, 1.0)
        self.assertIn("accounts no longer to be relied on", why[0])
        s, _ = at.news_score(["results report (8-K 2.02)"], 2)
        self.assertEqual(s, 1.0)                                                # 0.7 + stories, capped
        s, _ = at.news_score([], 0)
        self.assertEqual(s, 0.0)

    def test_fragile(self):
        m = {"mosaic": {"pictures": {"sale": {"name": "a sale of new shares", "chance": 0.10, "usual": 0.01}}},
             "numbers": {"cash_covers_quarters": 1.5}}
        s, why = at.fragile_score(m)
        self.assertEqual(s, 1.0)
        self.assertEqual(len(why), 2)
        self.assertEqual(at.fragile_score({})[0], 0.0)

    def test_reach(self):
        self.assertEqual(at.reach_score(0), 0.0)
        self.assertEqual(at.reach_score(40), 1.0)
        self.assertLess(at.reach_score(5), at.reach_score(20))


class Weights(unittest.TestCase):
    def test_weights_sum_to_one_and_order(self):
        c = [{"t": "A", "score": 0.5}, {"t": "B", "score": 0.2}, {"t": "C", "score": 0.4}]
        out = at.weigh(c)
        self.assertEqual([x["t"] for x in out], ["A", "C", "B"])
        self.assertAlmostEqual(sum(x["attention"] for x in out), 1.0)
        self.assertGreater(out[0]["attention"], out[1]["attention"])
        self.assertAlmostEqual(sum(at.WEIGHTS.values()), 1.0)


class FakeResponse:
    def __init__(self, body):
        self.body = body


def chart(ts, closes):
    return lambda url, ttl=0: FakeResponse(json.dumps(
        {"chart": {"result": [{"timestamp": ts, "indicators": {"quote": [{"close": closes}]}}]}}).encode())


class StrayPrint(unittest.TestCase):
    def test_one_late_print_is_thin(self):
        from datetime import UTC, datetime
        close = int(datetime(2026, 11, 20, 21, 0, tzinfo=UTC).timestamp())
        ts = [close - 300, close, close + 300, close + 600, close + 900, close + 1200]
        # the close at 10.0; after it, nothing moves until one print at 8.8, repeated by the chart
        f = chart(ts, [10.0, 10.0, 10.0, 9.9, 8.8, 8.8])
        n = at.trading_behind("X", "2026-11-20T21:00:00Z", "2026-11-21T04:00:00Z", -0.12, f)
        self.assertEqual(n, 1)
        cands = [{"t": "X", "score": 0.6, "move": -0.12, "reasons": []}]
        at.check_trading(cands, "2026-11-20T21:00:00Z", 1, "2026-11-21T04:00:00Z", f)
        self.assertAlmostEqual(cands[0]["score"], 0.3)
        self.assertIn("stray print", cands[0]["reasons"][-1])

    def test_a_real_move_is_not_thin(self):
        from datetime import UTC, datetime
        close = int(datetime(2026, 11, 20, 21, 0, tzinfo=UTC).timestamp())
        ts = [close - 300] + [close + 300 * k for k in range(1, 9)]
        f = chart(ts, [10.0, 11.5, 11.6, 11.7, 11.4, 11.8, 11.65, 11.55, 11.75])
        self.assertGreater(at.trading_behind("X", "2026-11-20T21:00:00Z", None, 0.16, f), at.THIN_BARS)

    def test_unreadable_is_none(self):
        def boom(url, ttl=0):
            raise OSError("no network")
        self.assertIsNone(at.trading_behind("X", "2026-11-20T21:00:00Z", None, 0.1, boom))


class Reading(unittest.TestCase):
    def test_key_sentences(self):
        text = ("Acme Corp reported revenue of $4.2 billion, up 12%. The weather was pleasant. "
                "The company now expects full-year growth of 8% to 9%, below its earlier outlook. "
                "Another sentence without anything in it at all.")
        k = at.key_sentences(text, ["Acme"])
        self.assertTrue(any("$4.2 billion" in s for s in k))
        self.assertTrue(any("expects" in s for s in k))
        self.assertFalse(any("weather" in s for s in k))

    def test_watched(self):
        case = {"asof": "2026-11-20", "moves": {"AAA": 0.01, "BBB": -0.02},
                "memory": {"themes": [{"name": "t", "members": ["AAA"]}], "threads": ["Watch BBB into Friday."]}}
        w = at.watched(case)
        self.assertIn("AAA", w)
        self.assertIn("BBB", w)


if __name__ == "__main__":
    unittest.main()
