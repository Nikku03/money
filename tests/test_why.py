"""Why a stock moved (brain/why.py): its industry's move, sector-wide moves, the chain links, the news, the verdict."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from brain import why


def ind(**codes):
    return {t: {"sic": c, "name": f"industry {c}"} for t, c in codes.items()}


class IndustryTest(unittest.TestCase):
    def test_the_finest_level_with_enough_peers_and_never_the_stock_itself(self):
        codes = {f"S{k}": "3674" for k in range(9)}                     # nine semiconductor stocks
        codes.update({f"C{k}": "3672" for k in range(3)})                # three circuit-board makers
        moves = {t: 0.08 for t in codes if t.startswith("S")}
        moves.update({"S0": 0.30, "C0": 0.01, "C1": 0.02, "C2": 0.03})
        ctx = why.industry_context(moves, ind(**codes))
        self.assertEqual((ctx["S0"]["level"], ctx["S0"]["code"], ctx["S0"]["n"]), (4, "3674", 8))
        self.assertAlmostEqual(ctx["S0"]["median"], 0.08)                 # its own +30% left out
        self.assertEqual((ctx["C0"]["level"], ctx["C0"]["code"], ctx["C0"]["n"]), (3, "367", 11))
        self.assertEqual(ctx["C0"]["name"], "industry 3674")             # the group's most common description

    def test_an_industry_moving_together_is_flagged(self):
        codes = {f"S{k}": "3674" for k in range(8)}
        codes.update({f"B{k}": "6022" for k in range(8)})
        moves = {t: (0.06 if t.startswith("S") else 0.004) for t in codes}
        sec = why.sector_moves(moves, ind(**codes))
        self.assertEqual([(x["code"], x["n"]) for x in sec], [("3674", 8)])
        self.assertAlmostEqual(sec[0]["median"], 0.06)


class AttributeTest(unittest.TestCase):
    def test_what_carried_a_move(self):
        self.assertEqual(why.attribute(0.20, 0.01, 0.01, news=True), "its own news")
        self.assertEqual(why.attribute(0.12, 0.015, 0.08, news=False), "its industry")
        self.assertEqual(why.attribute(0.025, 0.02, None, news=False), "the market")
        self.assertEqual(why.attribute(0.25, 0.015, 0.02, news=False), "its own move, no news found")
        self.assertEqual(why.attribute(0.10, 0.01, 0.07, news=True), "its industry")   # news, but the industry moved too


class LinksTest(unittest.TestCase):
    def test_industries_the_chain_engine_reaches(self):
        self.assertEqual(why.link_for("1311")[0], "oil producers")
        self.assertEqual(why.link_for("3674")[0], "tech")                # by the three-digit prefix 367
        self.assertEqual(why.link_for("6022")[0], "banks")
        self.assertIsNone(why.link_for("2834"))
        note = why.link_note({"sic": "1311"}, [{"symbol": "CL=F", "name": "oil", "unit": "pct", "move": 0.031}],
                             {"L1-oil-producers": {"per_move": 0.0296, "move_label": "+10% oil", "same": "REPORT",
                                                   "next": "FAIL"}})
        self.assertEqual(note, "chain: oil producers; oil +3.1% since the close; recorded link +2.96% per +10% oil in "
                               "the same month (REPORT); the next month FAIL")


class DayAndNewsTest(unittest.TestCase):
    def test_the_close_in_utc_and_the_days_macro_releases(self):
        self.assertEqual(why.close_utc("2026-11-10"), "2026-11-10T21:00:00Z")   # 16:00 New York in winter
        self.assertEqual(why.close_utc("2026-07-10"), "2026-07-10T20:00:00Z")   # and in summer
        self.assertIn("the jobs report", why.macro_today("2026-11-06"))
        self.assertEqual(why.macro_today("2026-11-11"), [])

    def test_headlines_since_the_close_newest_first(self):
        def fake(t):
            return [SimpleNamespace(value="old", known_at="2026-11-10T15:00:00Z", source={"name": "a"}),
                    SimpleNamespace(value="new", known_at="2026-11-11T12:00:00Z", source={"name": "b"}),
                    SimpleNamespace(value="newer", known_at="2026-11-11T13:00:00Z", source={"name": "c"})]
        got = why.headlines_since("X", "2026-11-10T21:00:00Z", ingest=fake)
        self.assertEqual(got, ["newer (c, 2026-11-11T13:00Z)", "new (b, 2026-11-11T12:00Z)"])

    def test_drivers_in_percent_and_the_yield_in_points(self):
        px = {"CL=F": (61.2, 60.0), "^TNX": (4.10, 4.15), "DX-Y.NYB": (99.0, 100.0)}
        got = why.drivers("2026-11-10", quote=lambda s: {"price": px[s][0]} if s in px else None,
                          close_on=lambda s, day: px[s][1] if s in px else None)
        self.assertEqual([d["name"] for d in got], ["oil", "10-year yield", "dollar"])
        self.assertAlmostEqual(got[0]["move"], 0.02)
        self.assertAlmostEqual(got[1]["move"], -0.05)


if __name__ == "__main__":
    unittest.main()
