"""World radar (brain/radar.py): moves from daily closes, flags by kind, the scan's order and the CLI."""
import contextlib
import io

import pandas as pd

from brain import cli, radar
from tests.util import TempHome, price_snapshot


def days(n, end="2026-10-23"):
    return pd.bdate_range(end=end, periods=n).strftime("%Y-%m-%d").tolist()


class RadarTest(TempHome):
    def test_moves_and_flags(self):
        closes = [100.0] * 60 + [100.0, 96.0, 90.0, 80.0]
        m = radar.moves(closes)
        self.assertAlmostEqual(m["d1"], -11.111, places=2)
        self.assertAlmostEqual(m["from_high"], -20.0)
        f = radar.flags("fund", m)
        self.assertTrue(any(x.startswith("day") for x in f) and any("3-month high" in x for x in f))
        self.assertEqual(radar.flags("fund", radar.moves([100.0] * 64)), [])
        self.assertIsNone(radar.moves([1.0] * 10))

    def test_fx_thresholds_are_tighter(self):
        m = radar.moves([1.0] * 63 + [1.02])
        self.assertTrue(radar.flags("fx", m))
        self.assertFalse(radar.flags("index", m))

    def test_scan_orders_by_severity_and_marks_missing_and_stale(self):
        d = days(70)
        self.bank.add_many([price_snapshot("EWZ", d, [42.0] * 69 + [34.0]),
                            price_snapshot("EWY", d, [150.0] * 70),
                            price_snapshot("^KS11", days(70, end="2026-10-09"), [6000.0] * 69 + [5000.0])])
        rows = radar.scan(self.bank, today="2026-10-26")
        have = [r for r in rows if not r.get("missing")]
        self.assertEqual(have[0]["symbol"], "EWZ")
        self.assertTrue(have[0]["flags"])
        ks = next(r for r in have if r["symbol"] == "^KS11")
        self.assertTrue(ks["stale"])
        self.assertEqual(next(r for r in have if r["symbol"] == "EWY")["flags"], [])
        self.assertTrue(any(r.get("missing") for r in rows))

    def test_cli(self):
        self.bank.add_many([price_snapshot("EWZ", days(70), [42.0] * 69 + [34.0])])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["radar"])
        self.assertIn("EWZ", out.getvalue())
        self.assertIn("Brazil", out.getvalue())


if __name__ == "__main__":
    import unittest
    unittest.main()
