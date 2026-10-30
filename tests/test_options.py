"""Cboe delayed option quotes (brain/sources/options.py): parsing OCC symbols, the straddle, the strangle."""
import unittest

from brain.sources import options


def chain_json(spot=50.0):
    def o(sym, bid, ask):
        return {"option": sym, "bid": bid, "ask": ask, "iv": 0.5, "open_interest": 10, "volume": 1}
    return {"timestamp": "2026-10-29 20:00:00", "data": {"symbol": "XYZ", "current_price": spot, "options": [
        o("XYZ261030C00050000", 1.9, 2.1), o("XYZ261030P00050000", 1.8, 2.0),
        o("XYZ261030C00054000", 0.45, 0.55), o("XYZ261030P00046000", 0.40, 0.50),
        o("XYZ261030C00055000", 0.0, 0.30),                       # no bid: unusable
        o("XYZ261106C00050000", 2.5, 2.7), o("XYZ261106P00050000", 2.4, 2.6),
        o("BAD", 1, 1)]}}


class OptionsTest(unittest.TestCase):
    def setUp(self):
        self.c = options.parse(chain_json())

    def test_parse(self):
        self.assertEqual(self.c["spot"], 50.0)
        self.assertEqual(sorted(self.c["chain"]), ["2026-10-30", "2026-11-06"])
        self.assertEqual(self.c["chain"]["2026-10-30"][50.0]["C"]["ask"], 2.1)

    def test_first_expiry(self):
        self.assertEqual(options.first_expiry(self.c["chain"], "2026-10-30"), "2026-10-30")
        self.assertEqual(options.first_expiry(self.c["chain"], "2026-10-31"), "2026-11-06")
        self.assertIsNone(options.first_expiry(self.c["chain"], "2026-12-01"))

    def test_straddle(self):
        s = options.straddle(self.c, "2026-10-30")
        self.assertEqual(s["strike"], 50.0)
        self.assertAlmostEqual(s["mid"], 3.9)
        self.assertAlmostEqual(s["cost"], 410.0)
        self.assertAlmostEqual(s["implied_pct"], 7.8)

    def test_strangle_skips_unusable_legs(self):
        g = options.strangle(self.c, "2026-10-30", 8.0)
        self.assertEqual((g["call"], g["put"]), (54.0, 46.0))
        self.assertAlmostEqual(g["cost"], 105.0)
        self.assertFalse(options.usable({"bid": 0.1, "ask": 0.5}))
        self.assertTrue(options.usable({"bid": 1.0, "ask": 1.2}))


if __name__ == "__main__":
    unittest.main()
