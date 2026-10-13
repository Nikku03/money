"""Count vehicles in a traffic-camera frame, in memory, with an open model.

YOLOX-S (Apache-2.0) as packaged by the OpenCV model zoo on Hugging Face
(``opencv/object_detection_yolox``), run through OpenCV's DNN module on the CPU. The file is
downloaded once to ``<home>/models/`` and checked against a pinned sha256 before every load.

The lines from CLAUDE.md ("Alternative data"), enforced here: the counter takes image bytes, never
a path, and writes nothing. Only the four COCO vehicle classes are scored at all; the person class
and every other class are not read from the network output, so nothing about people can be
reported. No faces, no number plates, no tracking across frames.

Optional dependency: ``opencv-python-headless`` (``pip install money-brain[alt]``). Without it
:class:`VehicleCounter` raises :class:`Unavailable` with that hint; the post-processing below is
plain numpy and needs no OpenCV.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

from brain import config

MODEL_REPO = "opencv/object_detection_yolox"
MODEL_REVISION = "78c368f74ce73ee28fc7a1be418a598c71b58b52"
MODEL_FILE = "object_detection_yolox_2022nov.onnx"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}"
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
MODEL_SIZE = 35_858_002
DETECTOR = "yolox-2022nov"
INPUT = 640
STRIDES = (8, 16, 32)
SCORE, NMS = 0.5, 0.5
VEHICLES = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}  # COCO class ids; nothing else is read
MAX_IMAGE_BYTES = 10 * 1024 * 1024
PLACEHOLDER_WHITE = 0.5  # share of near-white pixels above which a frame is a provider's notice, not a view


class NotLive(ValueError):
    """The frame is a provider placeholder ("Temporarily Unavailable") or blank: nothing to count."""


class Unavailable(RuntimeError):
    """The vehicle counter cannot run here (OpenCV missing, model not obtainable)."""


def _cv2():
    try:
        import cv2
    except ImportError as e:
        raise Unavailable("vehicle counting needs OpenCV: pip install opencv-python-headless "
                          "(or pip install money-brain[alt])") from e
    return cv2


# ---------- the model file ----------
def model_path() -> Path:
    return config.home() / "models" / MODEL_FILE


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str, log=print) -> Path:
    """Fetch ``url`` to ``dest`` once, identifying ourselves; keep it only if the sha256 matches."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    req = urllib.request.Request(url, headers={"User-Agent": config.WEB_USER_AGENT})
    h = hashlib.sha256()
    log(f"  downloading {url}")
    try:
        with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as fh:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                h.update(chunk)
                fh.write(chunk)
        if h.hexdigest() != sha256:
            raise Unavailable(f"{dest.name}: sha256 {h.hexdigest()} does not match the pinned {sha256}; not used")
        os.replace(tmp, dest)
    except OSError as e:
        raise Unavailable(f"could not download {url}: {e}") from e
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def ensure_model(log=print) -> Path:
    p = model_path()
    if p.exists() and _sha256(p) == MODEL_SHA256:
        return p
    if p.exists():
        log(f"  {p} does not match the pinned sha256; downloading again")
    return download(MODEL_URL, p, MODEL_SHA256, log)


# ---------- pre- and post-processing ----------
def letterbox(rgb: np.ndarray, size: int = INPUT) -> tuple[np.ndarray, float]:
    """Scale into a size x size canvas (top-left, padded with 114) as the model was trained."""
    cv2 = _cv2()
    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(h * r), int(w * r)
    out = np.full((size, size, 3), 114.0, dtype=np.float32)
    out[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return out, r


def anchors(size: int = INPUT) -> tuple[np.ndarray, np.ndarray]:
    """Grid cell (x, y) and stride for each of the model's output rows (8400 at 640)."""
    grids, strides = [], []
    for s in STRIDES:
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    """Greedy non-maximum suppression on x1, y1, x2, y2 boxes; indices kept, best first."""
    order = list(np.argsort(-scores, kind="stable"))
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        if not order:
            break
        rest = np.array(order)
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = [int(j) for j, o in zip(rest, iou) if o <= thr]
    return keep


def postprocess(raw: np.ndarray, ratio: float = 1.0, score: float = SCORE, nms_thr: float = NMS,
                size: int = INPUT) -> list[dict]:
    """Vehicle detections from one YOLOX output, shape (rows, 85): box regression (4), objectness, then
    80 class probabilities. Only the vehicle class columns are read. Suppression runs across the vehicle
    classes together, so one pickup scored as both car and truck counts once."""
    raw = np.asarray(raw, dtype=np.float32).reshape(-1, raw.shape[-1])
    grid, stride = anchors(size)
    if len(raw) != len(grid):
        raise ValueError(f"expected {len(grid)} output rows for a {size}px input, got {len(raw)}")
    names = list(VEHICLES)
    cls = raw[:, [5 + VEHICLES[n] for n in names]]
    scores = raw[:, 4:5] * cls
    best = scores.argmax(1)
    top = scores.max(1)
    m = top >= score
    if not m.any():
        return []
    xy = (raw[m, :2] + grid[m]) * stride[m]
    wh = np.exp(np.clip(raw[m, 2:4], -20, 20)) * stride[m]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / ratio
    keep = nms(boxes, top[m], nms_thr)
    return [{"class": names[int(best[m][i])], "score": float(top[m][i]), "box": [float(x) for x in boxes[i]]}
            for i in keep]


def tally(dets: list[dict]) -> dict:
    out = {n: 0 for n in ("car", "truck", "bus", "motorcycle")}
    for d in dets:
        out[d["class"]] += 1
    out["vehicles"] = sum(out.values())
    return out


def live_frame(img: np.ndarray) -> str | None:
    """Why ``img`` (H x W x 3) is not a live camera view, or None when it is. Caltrans serves a white
    "Temporarily Unavailable" card in place of a dead camera's frame: about 90% near-white, where live
    road scenes measured 0-7%. Counting such a card would record a false zero. The bottom 15% (the
    caption bar) is left out of the measure."""
    body = img[: max(1, int(img.shape[0] * 0.85))]
    if float((body.min(axis=2) >= 235).mean()) >= PLACEHOLDER_WHITE:
        return "a provider placeholder (mostly white), not a live view"
    if float(body.std()) < 4.0:
        return "a blank frame"
    return None


# ---------- the counter ----------
class VehicleCounter:
    """``count(image_bytes)`` -> {"car", "truck", "bus", "motorcycle", "vehicles"}. ``net`` is for tests: any
    object with OpenCV's ``setInput`` / ``forward``."""

    def __init__(self, net=None, score: float = SCORE, nms_thr: float = NMS, log=print):
        self.cv2 = _cv2()
        self.score, self.nms_thr = score, nms_thr
        if net is None:
            net = self.cv2.dnn.readNet(str(ensure_model(log)))
        self.net = net

    def detect(self, image_bytes: bytes) -> list[dict]:
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(f"image larger than {MAX_IMAGE_BYTES} bytes")
        bgr = self.cv2.imdecode(np.frombuffer(image_bytes, np.uint8), self.cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("not a decodable image")
        why = live_frame(bgr)
        if why:
            raise NotLive(why)
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        del bgr
        img, ratio = letterbox(rgb)
        del rgb
        self.net.setInput(np.transpose(img, (2, 0, 1))[np.newaxis])
        out = self.net.forward()
        return postprocess(out[0] if out.ndim == 3 else out, ratio, self.score, self.nms_thr)

    def count(self, image_bytes: bytes) -> dict:
        return tally(self.detect(image_bytes))

