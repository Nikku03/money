import hashlib
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from brain import vision
from tests.util import TempHome

HAS_CV2 = importlib.util.find_spec("cv2") is not None
PERSON = 0  # COCO


def synthetic(dets):
    """A YOLOX output (8400, 85) with the given detections placed at chosen anchors.
    dets: (row, cx, cy, w, h, objectness, class id, class prob), box in input pixels."""
    grid, stride = vision.anchors()
    raw = np.zeros((len(grid), 85), dtype=np.float32)
    raw[:, 2:4] = np.log(1.0)
    for row, cx, cy, w, h, obj, cls, p in dets:
        s = stride[row, 0]
        raw[row, 0:2] = [cx / s - grid[row, 0], cy / s - grid[row, 1]]
        raw[row, 2:4] = [np.log(w / s), np.log(h / s)]
        raw[row, 4] = obj
        raw[row, 5 + cls] = p
    return raw


def row_at(cx, cy, stride=8):
    """Index of the stride-`stride` anchor containing (cx, cy)."""
    off = {8: 0, 16: 80 * 80, 32: 80 * 80 + 40 * 40}[stride]
    n = 640 // stride
    return off + int(cy // stride) * n + int(cx // stride)


class PostprocessTest(unittest.TestCase):
    def test_counts_vehicles_only(self):
        raw = synthetic([
            (row_at(100, 100), 100, 100, 60, 40, 0.9, 2, 0.95),     # car
            (row_at(108, 100), 104, 101, 60, 40, 0.9, 2, 0.80),     # same car again: suppressed
            (row_at(300, 300), 300, 300, 120, 80, 0.8, 7, 0.90),    # truck
            (row_at(200, 400), 200, 400, 50, 50, 0.9, 3, 0.90),     # motorcycle
            (row_at(500, 100), 500, 100, 100, 60, 0.5, 5, 0.50),    # bus at 0.25: below threshold
            (row_at(400, 500), 400, 500, 30, 80, 0.99, PERSON, 0.99),  # a person: never reported
            (row_at(600, 600), 600, 600, 30, 30, 0.99, 16, 0.99),   # a dog: never reported
        ])
        dets = vision.postprocess(raw)
        self.assertEqual(sorted(d["class"] for d in dets), ["car", "motorcycle", "truck"])
        self.assertEqual(vision.tally(dets), {"car": 1, "truck": 1, "bus": 0, "motorcycle": 1, "vehicles": 3})
        car = [d for d in dets if d["class"] == "car"][0]
        np.testing.assert_allclose(car["box"], [70, 80, 130, 120], atol=1e-3)
        self.assertAlmostEqual(car["score"], 0.9 * 0.95, places=5)

    def test_person_column_is_never_read(self):
        raw = synthetic([(row_at(100, 100), 100, 100, 60, 40, 0.9, 2, 0.95)])
        base = vision.postprocess(raw)
        raw[:, 5 + PERSON] = np.nan  # if the person column were read, scores would go NaN
        self.assertEqual(vision.postprocess(raw), base)

    def test_one_object_scored_as_car_and_truck_counts_once(self):
        raw = synthetic([
            (row_at(100, 100), 100, 100, 60, 40, 0.9, 2, 0.90),
            (row_at(108, 100), 102, 100, 60, 40, 0.9, 7, 0.85),
        ])
        self.assertEqual(vision.tally(vision.postprocess(raw))["vehicles"], 1)

    def test_ratio_maps_back_to_the_frame(self):
        raw = synthetic([(row_at(100, 100), 100, 100, 60, 40, 0.9, 2, 0.95)])
        d = vision.postprocess(raw, ratio=0.5)[0]
        np.testing.assert_allclose(d["box"], [140, 160, 260, 240], atol=1e-3)

    def test_empty_and_shape(self):
        self.assertEqual(vision.postprocess(np.zeros((8400, 85), np.float32)), [])
        with self.assertRaises(ValueError):
            vision.postprocess(np.zeros((10, 85), np.float32))

    def test_nms(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
        self.assertEqual(vision.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5), [0, 2])
        self.assertEqual(vision.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.9), [0, 1, 2])


class LiveFrameTest(unittest.TestCase):
    def test_live_frame(self):
        rng = np.random.default_rng(1)
        road = rng.integers(40, 180, (260, 320, 3)).astype(np.uint8)
        self.assertIsNone(vision.live_frame(road))
        night = np.full((260, 320, 3), 12, np.uint8)
        night[100:110, 50:60] = 250  # headlights
        night += rng.integers(0, 8, night.shape).astype(np.uint8)
        self.assertIsNone(vision.live_frame(night))
        card = np.full((260, 320, 3), 250, np.uint8)
        card[100:140, 40:280] = (110, 30, 20)
        self.assertIn("placeholder", vision.live_frame(card))
        self.assertIn("blank", vision.live_frame(np.full((260, 320, 3), 128, np.uint8)))


class ModelFileTest(TempHome):
    def test_download_checks_sha256(self):
        src = Path(self._tmp.name) / "src.onnx"
        src.write_bytes(b"model bytes")
        dest = vision.model_path()
        with self.assertRaises(vision.Unavailable):
            vision.download(src.as_uri(), dest, "0" * 64, log=lambda *_: None)
        self.assertFalse(dest.exists())
        self.assertEqual(list(dest.parent.iterdir()), [])  # no partial file left
        good = hashlib.sha256(b"model bytes").hexdigest()
        self.assertEqual(vision.download(src.as_uri(), dest, good, log=lambda *_: None), dest)
        self.assertEqual(dest.read_bytes(), b"model bytes")

    def test_model_lives_under_home_models(self):
        self.assertEqual(vision.model_path().parent.name, "models")
        self.assertTrue(str(vision.model_path()).startswith(self._tmp.name))
        self.assertEqual(len(vision.MODEL_SHA256), 64)

    def test_without_opencv(self):
        with mock.patch.dict("sys.modules", {"cv2": None}):
            with self.assertRaises(vision.Unavailable) as cm:
                vision.VehicleCounter(net=object())
        self.assertIn("opencv-python-headless", str(cm.exception))


class FakeNet:
    def __init__(self, raw):
        self.raw, self.inputs = raw, []

    def setInput(self, blob):
        self.inputs.append(blob.shape)

    def forward(self):
        return self.raw[np.newaxis]


@unittest.skipUnless(HAS_CV2, "opencv not installed")
class CounterTest(TempHome):
    def test_count_from_jpeg_bytes(self):
        import cv2
        scene = np.random.default_rng(0).integers(0, 200, (480, 1280, 3)).astype(np.uint8)
        ok, jpg = cv2.imencode(".jpg", scene)
        raw = synthetic([(row_at(100, 100), 100, 100, 60, 40, 0.9, 7, 0.95)])
        net = FakeNet(raw)
        c = vision.VehicleCounter(net=net)
        self.assertEqual(c.count(jpg.tobytes()), {"car": 0, "truck": 1, "bus": 0, "motorcycle": 0, "vehicles": 1})
        self.assertEqual(net.inputs, [(1, 3, 640, 640)])
        d = c.detect(jpg.tobytes())[0]  # 1280 wide -> ratio 0.5
        np.testing.assert_allclose(d["box"], [140, 160, 260, 240], atol=1e-3)
        with self.assertRaises(ValueError):
            c.count(b"not an image")

    def test_placeholder_frames_are_not_counted(self):
        import cv2
        card = np.full((260, 320, 3), 255, np.uint8)  # Caltrans' white "Temporarily Unavailable" card
        cv2.putText(card, "Temporarily", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (110, 30, 20), 3)
        card[225:] = 60  # caption bar
        ok, jpg = cv2.imencode(".jpg", card)
        net = FakeNet(synthetic([]))
        with self.assertRaises(vision.NotLive):
            vision.VehicleCounter(net=net).count(jpg.tobytes())
        self.assertEqual(net.inputs, [])  # never reached the model
        ok, blank = cv2.imencode(".jpg", np.full((260, 320, 3), 30, np.uint8))
        with self.assertRaises(vision.NotLive):
            vision.VehicleCounter(net=net).count(blank.tobytes())

    def test_letterbox(self):
        img, r = vision.letterbox(np.zeros((480, 1280, 3), np.uint8))
        self.assertEqual((img.shape, r), ((640, 640, 3), 0.5))
        self.assertEqual(img[0, 0, 0], 0.0)
        self.assertEqual(img[400, 0, 0], 114.0)  # padding below the 240-row image


if __name__ == "__main__":
    unittest.main()
