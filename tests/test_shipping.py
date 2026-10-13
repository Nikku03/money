import argparse
import asyncio
import importlib.util
import json
import os
import threading
import unittest
from unittest import mock

from brain import config
from brain.sources import shipping as sh
from tests.util import TempHome

AREA = {"name": "TEST-PORT", "bbox": [10.0, 20.0, 11.0, 21.0]}


def pos(mmsi, lat=10.5, lon=20.5, nav=0, t="2026-10-13 10:00:00.0 +0000 UTC"):
    return {"MessageType": "PositionReport",
            "MetaData": {"MMSI": mmsi, "ShipName": "X", "latitude": lat, "longitude": lon, "time_utc": t},
            "Message": {"PositionReport": {"UserID": mmsi, "Latitude": lat, "Longitude": lon,
                                           "NavigationalStatus": nav, "Sog": 0.1}}}


def static(mmsi, typ):
    return {"MessageType": "ShipStaticData", "MetaData": {"MMSI": mmsi},
            "Message": {"ShipStaticData": {"UserID": mmsi, "Type": typ, "Name": "X", "ImoNumber": 1}}}


MESSAGES = [
    static(211000001, 70), pos(211000001, nav=1),                       # cargo, anchored
    static(211000002, 79), pos(211000002, nav=0),                       # cargo, underway ...
    pos(211000002, nav=5, t="2026-10-13 10:04:00.0 +0000 UTC"),         # ... then moored (latest wins)
    static(311000003, 84), pos(311000003, nav=5),                       # tanker, moored
    static(411000004, 37), pos(411000004, nav=0),                       # pleasure craft: dropped
    static(411000005, 36), pos(411000005, nav=0),                       # sailing: dropped
    static(511000006, 52), pos(511000006, nav=0),                       # tug: other, not counted
    pos(611000007, nav=15),                                             # no static data yet: untyped, not counted
    static(711000008, 70), pos(711000008, lat=12.0, nav=0),             # outside the box: not counted
    pos(3669999, nav=0),                                                # not a ship station MMSI
    static(211000009, 71),                                              # static data but no position: not seen
    {"MessageType": "PositionReport", "Message": {}},                   # malformed
]


class SummarizeTest(unittest.TestCase):
    def test_counts(self):
        s = sh.summarize(MESSAGES, AREA, 300)
        self.assertEqual(s["vessels"], 3)  # commercial: cargo and tanker
        self.assertEqual((s["cargo"], s["tanker"], s["other"], s["untyped"]), (2, 1, 1, 1))
        self.assertEqual((s["anchored"], s["moored"], s["underway"]), (1, 2, 0))  # of the commercial vessels
        self.assertEqual((s["window_s"], s["bbox"], s["basis"]), (300, AREA["bbox"], "cargo+tanker"))

    def test_untyped_vessels_are_never_counted(self):
        # A pleasure craft whose static data fell outside the window, beside one whose static data was heard.
        msgs = [pos(366000001), pos(366000002), static(366000002, 37)]
        s = sh.summarize(msgs, AREA, 300)
        self.assertEqual((s["vessels"], s["other"], s["untyped"]), (0, 0, 1))
        self.assertEqual((s["anchored"], s["moored"], s["underway"]), (0, 0, 0))

    def test_default_window_outlasts_the_static_data_cycle(self):
        import inspect
        self.assertGreaterEqual(sh.DEFAULT_SECONDS, 6 * 60 + 30)
        self.assertEqual(inspect.signature(sh.sample).parameters["seconds"].default, sh.DEFAULT_SECONDS)

    def test_aggregates_only(self):
        s = sh.summarize(MESSAGES, AREA, 300)
        self.assertEqual(set(s), {"vessels", "cargo", "tanker", "other", "untyped", "anchored", "moored",
                                  "underway", "window_s", "bbox", "basis"})
        text = json.dumps(s)
        for m in ("211000001", "311000003"):
            self.assertNotIn(m, text)

    def test_fact_shape(self):
        f = sh.to_fact(sh.summarize(MESSAGES, AREA, 300), AREA, "2026-10-13T10:05:00Z")
        self.assertEqual((f.entity, f.attr, f.value, f.tier), ("AREA.TEST-PORT", "ships.count", 3, "reported"))
        self.assertEqual((f.as_of, f.known_at), ("2026-10-13", "2026-10-13T10:05:00Z"))
        self.assertLessEqual({"cargo", "tanker", "other", "anchored", "moored", "underway", "window_s", "bbox",
                              "basis"}, set(f.context))
        self.assertNotIn("vessels", f.context)
        self.assertEqual(f.source["name"], "aisstream.io")


class AreasTest(TempHome):
    def test_builtin(self):
        names = [a["name"] for a in sh.AREAS]
        self.assertEqual(names, ["LA-LONG-BEACH", "OAKLAND", "HOUSTON", "NEW-YORK-NJ", "ROTTERDAM", "SINGAPORE",
                                 "SHANGHAI", "NINGBO", "BUSAN", "JEBEL-ALI", "SUEZ", "PANAMA", "HORMUZ"])
        for a in sh.AREAS:
            self.assertTrue(sh._valid_bbox(a["bbox"]), a)
        self.assertEqual(sh.find_area("la-long-beach")["name"], "LA-LONG-BEACH")
        with self.assertRaises(KeyError):
            sh.find_area("ATLANTIS")

    def test_watchlist_areas_add(self):
        (config.home() / "watchlist.json").write_text(json.dumps({"areas": [
            {"name": "LA-LONG-BEACH"}, {"name": "Savannah", "bbox": [31.9, -81.2, 32.2, -80.8]},
            {"name": "BROKEN", "bbox": [5, 5, 1, 1]}]}))
        names = [a["name"] for a in sh.areas()]
        self.assertIn("SAVANNAH", names)
        self.assertNotIn("BROKEN", names)
        self.assertEqual(names.count("LA-LONG-BEACH"), 1)

    def test_no_key(self):
        with mock.patch.dict(os.environ, {"AISSTREAM_API_KEY": ""}):
            with self.assertRaises(sh.Unavailable) as cm:
                sh.sample(self.bank, "SINGAPORE", seconds=1)
        self.assertIn("AISSTREAM_API_KEY", str(cm.exception))
        self.assertEqual(self.bank.facts("AREA.SINGAPORE", "ships.count"), [])


@unittest.skipUnless(importlib.util.find_spec("websockets"), "websockets not installed")
class SampleTest(TempHome):
    """sample() against a local fake aisstream server (no network)."""

    def setUp(self):
        super().setUp()
        import websockets
        self.got = []
        ready = threading.Event()
        self.loop = asyncio.new_event_loop()

        async def handler(ws):
            self.got.append(json.loads(await ws.recv()))
            for m in MESSAGES:
                await ws.send(json.dumps(m))
            await ws.wait_closed()

        async def main():
            self.server = await websockets.serve(handler, "127.0.0.1", 0)
            self.port = self.server.sockets[0].getsockname()[1]
            ready.set()
            await self.server.wait_closed()

        self.thread = threading.Thread(target=lambda: self.loop.run_until_complete(main()), daemon=True)
        self.thread.start()
        ready.wait(5)

    def tearDown(self):
        self.loop.call_soon_threadsafe(self.server.close)
        self.thread.join(5)
        super().tearDown()

    def test_sample_writes_one_aggregate_fact(self):
        f = sh.sample(self.bank, AREA, seconds=1, key="k", url=f"ws://127.0.0.1:{self.port}")
        sub = self.got[0]
        self.assertEqual(sub["APIKey"], "k")
        self.assertEqual(sub["BoundingBoxes"], [[[10.0, 20.0], [11.0, 21.0]]])
        self.assertEqual(sub["FilterMessageTypes"], ["PositionReport", "ShipStaticData"])
        self.assertEqual(f.value, 3)
        self.assertEqual([g.id for g in self.bank.facts("AREA.TEST-PORT", "ships.count")], [f.id])


class CliTest(unittest.TestCase):
    def test_subcommands(self):
        p = argparse.ArgumentParser()
        sh.add_cli(p.add_subparsers(dest="cmd"))
        a = p.parse_args(["ships", "sample", "HOUSTON", "--seconds", "60"])
        self.assertEqual((a.area, a.seconds, a.fn), ("HOUSTON", 60, sh._cmd_sample))
        self.assertEqual(p.parse_args(["ships", "sample", "HOUSTON"]).seconds, sh.DEFAULT_SECONDS)
        self.assertEqual(p.parse_args(["ships", "areas"]).fn, sh._cmd_areas)


if __name__ == "__main__":
    unittest.main()
