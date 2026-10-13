import argparse
import contextlib
import gzip
import importlib.util
import io
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from brain import config, vision
from brain.sources import cameras as cams
from tests.util import TempHome


def tfl_place(pid, name, lat, lon, available="true", img=None):
    img = img or f"https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/{pid}.jpg"
    return {"$type": "Tfl.Api.Presentation.Entities.Place", "id": f"JamCams_{pid}", "commonName": name,
            "placeType": "JamCam", "lat": lat, "lon": lon, "additionalProperties": [
                {"category": "payload", "key": "available", "value": available},
                {"category": "payload", "key": "imageUrl", "value": img},
                {"category": "payload", "key": "videoUrl", "value": img.replace(".jpg", ".mp4")}]}


TFL = [
    tfl_place("00001.01251", "Tower Bridge Rd", 51.5007, -0.0762),
    tfl_place("00002.00865", "A406 Billet Upass E", 51.6012, -0.0215),
    tfl_place("00001.09999", "Out of service", 51.5008, -0.0763, available="false"),
    tfl_place("00001.08888", "Odd host", 51.5009, -0.0764, img="https://example.com/cam.jpg"),
    {"id": "JamCams_00001.07777", "commonName": "no position", "additionalProperties": []},
]


def caltrans_row(slug, name, lat, lon, in_service="true", district=4):
    return {"cctv": {"index": "1", "location": {"district": str(district), "locationName": name,
                                                "nearbyPlace": "Fremont", "latitude": str(lat), "longitude": str(lon)},
                     "inService": in_service, "imageData": {"static": {
                         "currentImageURL": f"https://cwwp2.dot.ca.gov/data/d{district}/cctv/image/{slug}/{slug}.jpg"}}}}


CALTRANS = {"data": [
    caltrans_row("tv512i880atmowry", "TV512 -- I-880 : At Mowry Ave", 37.5291, -121.9894),
    caltrans_row("tv513i880atstevenson", "TV513 -- I-880 : At Stevenson", 37.5101, -121.9690),
    caltrans_row("tv999off", "TV999 -- off", 37.5292, -121.9895, in_service="false"),
]}


def fake_listing(url):
    if "tfl.gov.uk" in url:
        return TFL
    if "cctvStatusD04" in url:
        return CALTRANS
    return {"data": []}


class ListingTest(TempHome):
    def test_parse_tfl(self):
        cs = cams.parse_tfl(TFL)
        self.assertEqual([c["id"] for c in cs], ["TFL.00001.01251", "TFL.00002.00865"])
        self.assertEqual(set(cs[0]), {"id", "provider", "name", "lat", "lon", "image_url", "licence"})
        self.assertEqual(cs[0]["provider"], "TFL")
        self.assertIn("TfL Open Data", cs[0]["licence"])

    def test_parse_caltrans(self):
        cs = cams.parse_caltrans(CALTRANS, 4)
        self.assertEqual([c["id"] for c in cs], ["CALTRANS.D04.tv512i880atmowry", "CALTRANS.D04.tv513i880atstevenson"])
        self.assertEqual(cs[0]["name"], "TV512 -- I-880 : At Mowry Ave (Fremont)")
        self.assertEqual((cs[0]["lat"], cs[0]["lon"]), (37.5291, -121.9894))

    def test_cameras_cached_a_day_and_near_sorted(self):
        with mock.patch.object(cams, "_get_json", side_effect=fake_listing) as g:
            everything = cams.cameras()
            n_calls = g.call_count
            self.assertEqual(n_calls, 1 + 12)  # TfL once, Caltrans districts 1-12
            self.assertEqual(len(everything), 4)
            hits = cams.near(37.5291, -121.9894, km=5)
            self.assertEqual(g.call_count, n_calls)  # served from the cached listing
        self.assertEqual([c["id"] for c in hits], ["CALTRANS.D04.tv512i880atmowry", "CALTRANS.D04.tv513i880atstevenson"])
        self.assertEqual(hits[0]["km"], 0.0)
        self.assertLess(hits[1]["km"], 5)
        self.assertEqual(cams.near(37.5291, -121.9894, km=1), hits[:1])
        self.assertEqual(cams.near(51.5007, -0.0762, km=1, provider="tfl")[0]["id"], "TFL.00001.01251")

    def test_tfl_key_stays_off_disk(self):
        seen = []
        with mock.patch.dict(os.environ, {"TFL_APP_KEY": "sekrit"}), \
                mock.patch.object(cams, "_get_json", side_effect=lambda u: seen.append(u) or TFL):
            cams.cameras("TFL")
        self.assertIn("app_key=sekrit", seen[0])
        for p in Path(self._tmp.name).rglob("*"):
            if p.is_file():
                self.assertNotIn(b"sekrit", p.read_bytes())

    def test_outage_is_not_cached_as_an_empty_listing(self):
        with mock.patch.object(cams, "_get_json", side_effect=cams.FetchError("HTTP 503")):
            with self.assertRaises(cams.FetchError):
                cams.cameras("CALTRANS")
        self.assertFalse((config.home() / "cache" / "cameras_CALTRANS.json").exists())

    def test_unknown_provider(self):
        with self.assertRaises(KeyError):
            cams.cameras("SOMEONE")

    def test_only_official_hosts(self):
        cam = cams.parse_tfl(TFL)[0]
        self.assertTrue(cams.official(cam))
        self.assertFalse(cams.official({**cam, "image_url": "http://192.168.1.10/snapshot.jpg"}))
        self.assertFalse(cams.official({**cam, "provider": "INSECAM"}))
        with self.assertRaises(PermissionError):
            cams.fetch_frame({**cam, "image_url": "https://example.com/x.jpg"})


MARKER = b"\xff\xd8\xff\xe0FRAME-MARKER-" + bytes(range(256)) * 4  # recognizable "image" bytes


class FakeCounter:
    def __init__(self):
        self.seen = 0

    def count(self, image_bytes):
        self.seen += 1
        if image_bytes == b"garbage":
            raise ValueError("not a decodable image")
        return {"car": 3, "truck": 1, "bus": 0, "motorcycle": 1, "vehicles": 5}


def all_bytes_under(root: Path) -> list[tuple[Path, bytes]]:
    out = []
    for p in root.rglob("*"):
        if p.is_file():
            b = p.read_bytes()
            if p.suffix == ".gz":
                b = gzip.decompress(b)
            out.append((p, b))
    return out


class SampleTest(TempHome):
    def setUp(self):
        super().setUp()
        with mock.patch.object(cams, "_get_json", side_effect=fake_listing):
            cams.cameras()

    def test_fact_shape_and_nothing_on_disk(self):
        counter = FakeCounter()
        n = cams.sample(self.bank, ["CALTRANS.D04.tv512i880atmowry", "tfl.00001.01251", "TFL.00000.00000"],
                        counter=counter, frame=lambda cam: MARKER, log=lambda *_: None)
        self.assertEqual((n, counter.seen), (2, 2))
        f = self.bank.facts("CAM.CALTRANS.D04.TV512I880ATMOWRY", "camera.vehicles")[0]
        self.assertEqual((f.attr, f.value, f.tier, f.unit), ("camera.vehicles", 5, "reported", "vehicles"))
        self.assertEqual(set(f.context), {"car", "truck", "bus", "motorcycle", "provider", "name", "lat", "lon",
                                          "detector", "threshold", "decode"})
        self.assertEqual((f.context["detector"], f.context["threshold"], f.context["decode"]), ("yolox-2022nov", 0.5, 2))
        self.assertEqual(f.source, {"name": "Caltrans CCTV", "url": cams.PROVIDERS["CALTRANS"]["page"]})
        self.assertEqual(f.as_of, f.known_at[:10])
        for p, b in all_bytes_under(Path(config.home())):
            self.assertNotIn(b"FRAME-MARKER", b, f"frame bytes found in {p}")
            self.assertNotIn(p.suffix.lower(), {".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    def test_default_fetch_path_keeps_frames_in_memory(self):
        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        urls = []

        def urlopen(req, timeout=None):
            urls.append(req.full_url)
            return Resp(MARKER)

        with mock.patch.object(cams.urllib.request, "urlopen", side_effect=urlopen), \
                mock.patch.object(cams, "MIN_INTERVAL", 0):
            n = cams.sample(self.bank, ["TFL.00001.01251"], counter=FakeCounter(), log=lambda *_: None)
        self.assertEqual(n, 1)
        self.assertEqual(urls, ["https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.01251.jpg"])
        for p, b in all_bytes_under(Path(config.home())):
            self.assertNotIn(b"FRAME-MARKER", b, f"frame bytes found in {p}")

    def test_bad_frame_skipped(self):
        n = cams.sample(self.bank, ["TFL.00001.01251"], counter=FakeCounter(), frame=lambda cam: b"garbage",
                        log=lambda *_: None)
        self.assertEqual(n, 0)

    def test_without_opencv_nothing_is_written(self):
        with mock.patch.object(vision, "_cv2", side_effect=vision.Unavailable("no cv2")):
            with self.assertRaises(vision.Unavailable):
                cams.sample(self.bank, ["TFL.00001.01251"], frame=lambda cam: MARKER, log=lambda *_: None)
        self.assertEqual(self.bank.facts("CAM.TFL.00001.01251", "camera.vehicles"), [])

    def test_cli_without_opencv_says_so(self):
        p = argparse.ArgumentParser()
        cams.add_cli(p.add_subparsers(dest="cmd"))
        a = p.parse_args(["cams", "sample", "TFL.00001.01251"])
        out = io.StringIO()
        with mock.patch.object(vision, "_cv2", side_effect=vision.Unavailable("needs OpenCV")), \
                contextlib.redirect_stdout(out):
            self.assertEqual(a.fn(a), 1)
        self.assertIn("nothing written", out.getvalue())

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "opencv not installed")
    def test_placeholder_frame_writes_no_zero(self):
        import cv2
        from tests.test_vision import FakeNet, synthetic
        ok, card = cv2.imencode(".jpg", np.full((260, 320, 3), 255, np.uint8))
        logged = []
        n = cams.sample(self.bank, ["CALTRANS.D04.tv512i880atmowry"], counter=vision.VehicleCounter(net=FakeNet(synthetic([]))),
                        frame=lambda cam: card.tobytes(), log=logged.append)
        self.assertEqual(n, 0)
        self.assertIn("placeholder", logged[0])

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "opencv not installed")
    def test_real_counter_path_writes_no_image(self):
        import cv2
        from tests.test_vision import FakeNet, row_at, synthetic
        img = np.zeros((360, 640, 3), np.uint8)
        img[100:200, 100:300] = 200
        ok, jpg = cv2.imencode(".jpg", img)
        raw = synthetic([(row_at(200, 150), 200, 150, 200, 100, 0.9, 2, 0.9)])
        counter = vision.VehicleCounter(net=FakeNet(raw))
        before = {p for p, _ in all_bytes_under(Path(config.home()))}
        n = cams.sample(self.bank, ["TFL.00002.00865"], counter=counter, frame=lambda cam: jpg.tobytes(),
                        log=lambda *_: None)
        self.assertEqual(n, 1)
        self.assertEqual(self.bank.facts("CAM.TFL.00002.00865", "camera.vehicles")[0].context["car"], 1)
        for p, b in all_bytes_under(Path(config.home())):
            self.assertNotIn(jpg.tobytes()[:64], b, f"frame bytes found in {p}")
            self.assertFalse(b.startswith(b"\xff\xd8\xff"), f"a JPEG at {p}")
        after = {p for p, _ in all_bytes_under(Path(config.home()))}
        self.assertFalse({p for p in after - before if "bank" not in p.parts})  # only the journal grew


class CliTest(unittest.TestCase):
    def test_subcommands(self):
        p = argparse.ArgumentParser()
        cams.add_cli(p.add_subparsers(dest="cmd"))
        a = p.parse_args(["cams", "near", "51.5", "-0.12", "--km", "2", "--provider", "tfl"])
        self.assertEqual((a.lat, a.lon, a.km, a.provider, a.fn), (51.5, -0.12, 2.0, "TFL", cams._cmd_near))
        a = p.parse_args(["cams", "sample", "TFL.00001.01251", "CALTRANS.D04.tv1"])
        self.assertEqual((a.ids, a.fn), (["TFL.00001.01251", "CALTRANS.D04.tv1"], cams._cmd_sample))


if __name__ == "__main__":
    unittest.main()
