"""The scan explorer's data (brain/explorer.py): one row per recorded stock, its ranges in percent, its move from the
latest check and where that move sits; only numbers the repository already records."""
from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from brain import explorer
from tests.util import TempHome


class WhereTest(unittest.TestCase):
    def test_a_move_is_placed_against_both_ranges(self):
        args = (-0.05, 0.06, -0.10, 0.12)
        self.assertEqual(explorer.where(0.01, *args), explorer.INSIDE)
        self.assertEqual(explorer.where(-0.07, *args), explorer.OUT68)
        self.assertEqual(explorer.where(0.15, *args), explorer.OUT90)
        self.assertEqual(explorer.where(None, *args), explorer.NONE)
        self.assertEqual(explorer.where(float("nan"), *args), explorer.NONE)


class TableTest(unittest.TestCase):
    def test_rows_turn_recorded_log_ranges_into_percent_and_carry_why(self):
        recorded = {"rows": {"BBB": [math.log(0.9), math.log(0.95), math.log(1.05), math.log(1.1), 0.012],
                             "AAA": [math.log(0.8), math.log(0.9), math.log(1.1), math.log(1.3), None]},
                    "newer_listings": ["BBB"]}
        ind = {"AAA": {"sic": "3674", "name": "Semiconductors & Related Devices", "company": "Aaa Corp"}}
        cols = explorer.table(recorded, ind, {"AAA": "a $3.00 stock"}, {"AAA": 0.35, "BBB": float("nan")},
                              {"AAA": {"t": "AAA", "why": "its own news", "headlines": ["Aaa wins a contract"]}})
        self.assertEqual(cols["Ticker"], ["AAA", "BBB"])
        self.assertEqual(list(cols), list(explorer.COLUMNS))
        self.assertEqual((cols["2-in-3 low %"][0], cols["2-in-3 high %"][0]), (-10.0, 10.0))
        self.assertEqual(cols["9-in-10 high %"][0], 30.0)
        self.assertEqual(cols["Typical move %"][0], 10.0)
        self.assertEqual(cols["Move now %"], [35.0, None])
        self.assertEqual(cols["Where now"], [explorer.OUT90, explorer.NONE])
        self.assertEqual(cols["20% day chance %"], [None, 1.2])
        self.assertEqual(cols["Listed after 2010"], ["no", "yes"])
        self.assertEqual(cols["Sector"][0], "Electronic equipment")
        self.assertEqual(cols["Why it moved"][0], "its own news; headline: Aaa wins a contract")
        self.assertEqual(cols["Why the range is wide"], ["a $3.00 stock", ""])


class BuildTest(TempHome):
    def test_build_joins_the_scan_its_check_and_its_note_without_prices(self):
        with tempfile.TemporaryDirectory() as scans, tempfile.TemporaryDirectory() as root:
            os.environ["BRAIN_SCANS"] = scans
            try:
                Path(scans, "2026-11-17.json").write_text(json.dumps({
                    "asof": "2026-11-17", "skipped": 3, "newer_listings": [],
                    "rows": {"AAA": [-0.1, -0.05, 0.05, 0.1, 0.01]}}))
                up = Path(scans, "updates", "2026-11-17")
                up.mkdir(parents=True)
                (up / "2026-11-18T0432Z.json").write_text(json.dumps({
                    "at": "2026-11-18T04:32:06Z", "checked": 1, "outside68": 1, "outside90": 0, "fear_up": False,
                    "market_wide": False, "flags": [{"t": "AAA", "why": "its own news", "headlines": []}],
                    "market": [{"name": "S&P 500 (SPY)", "move": 0.007, "session": "after the close"}]}))
                case = Path(os.environ["MONEY_HOME"], "desk", "2026-11-17")
                case.mkdir(parents=True)
                (case / "2026-11-18T0432Z.json").write_text(json.dumps({"check": "2026-11-18T0432Z",
                                                                        "moves": {"AAA": 0.07}}))
                note = Path(root, "research", "desk", "2026-11-17")
                note.mkdir(parents=True)
                (note / "2026-11-18T0432Z.json").write_text(json.dumps({"check": "2026-11-18T0432Z", "note": {
                    "themes": [{"name": "A theme", "kind": "reported", "members": ["AAA"]}], "singles": []}}))
                with mock.patch.object(explorer.config, "ROOT", Path(root)), \
                        mock.patch.object(explorer, "reasons", lambda bank, asof, tickers, log=None: {"AAA": "why"}):
                    out = explorer.build(self.bank, out=Path(root, "data.json"), log=None)
                d = json.loads(out.read_text())
            finally:
                os.environ.pop("BRAIN_SCANS", None)
        self.assertEqual(d["scan"]["asof"], "2026-11-17")
        self.assertEqual(d["check"]["line"], "S&P 500 (SPY) +0.7%")
        self.assertEqual(d["check"]["session"], "after the close")
        self.assertEqual(d["note"]["themes"][0]["members"], ["AAA"])
        self.assertEqual(d["rows"]["Move now %"], [7.0])
        self.assertEqual(d["rows"]["Where now"], [explorer.OUT68])
        self.assertEqual(d["rows"]["Why it moved"], ["its own news"])
        self.assertNotIn("close", d["rows"])                      # no prices in the page's data
        self.assertNotIn("price", json.dumps(d["columns"]).lower())


class FilesTest(unittest.TestCase):
    def test_every_file_the_page_loads_is_published_beside_it(self):
        m = explorer.files_map()
        self.assertIn("data.json", m)
        self.assertIn("lib/server/dist/wasm/perspective-server.wasm", m)
        self.assertIn("lib/viewer/dist/wasm/perspective-viewer.wasm", m)
        page = explorer.page().read_text()
        for path in ("lib/client/dist/cdn/perspective.js", "lib/viewer/dist/cdn/perspective-viewer.js",
                     "lib/viewer-datagrid/dist/cdn/perspective-viewer-datagrid.js",
                     "lib/viewer/dist/css/pro.css", "data.json"):
            self.assertIn(path, page)
            self.assertIn(path, m)


if __name__ == "__main__":
    unittest.main()
