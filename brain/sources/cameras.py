"""Official traffic cameras: find them near a place, and count the vehicles they show.

The lines from CLAUDE.md ("Alternative data"), enforced here:

- Only providers in ``PROVIDERS``: government traffic cameras published for public viewing under
  open terms (Transport for London JamCams, Caltrans). A camera is sampled only by its id from
  that provider's own listing, and only from an image URL on the provider's published image host
  (``image_prefixes``); there is no way to pass an arbitrary URL. Adding a provider means adding
  it here with its licence.
- Frames are fetched into memory, counted and dropped. Nothing is written to disk: not the image,
  not a crop, not an HTTP cache entry. Listings are cached as parsed JSON (no keys in them).
- Vehicle classes only (``brain.vision``). No people, faces or number plates.

Credit the sources: "Powered by TfL Open Data"; Caltrans CWWP2.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode, urlparse

from brain import config
from brain.bank import Bank, Fact
from brain.net import FetchError
from brain.timeutil import now_iso

LISTING_TTL = 86400
CALTRANS_DISTRICTS = range(1, 13)
PROVIDERS = {
    "TFL": {
        "name": "TfL JamCams",
        "page": "https://api.tfl.gov.uk/",
        "licence": "TfL Open Data (Powered by TfL Open Data)",
        "listing": "https://api.tfl.gov.uk/Place/Type/JamCam",
        "image_prefixes": ("https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/",),
    },
    "CALTRANS": {
        "name": "Caltrans CCTV",
        "page": "https://cwwp2.dot.ca.gov/documentation/cctv/cctv.htm",
        "licence": "Caltrans CWWP2 public data, no charge (Caltrans Conditions of Use)",
        "listing": "https://cwwp2.dot.ca.gov/data/d{n}/cctv/cctvStatusD{n:02d}.json",
        "image_prefixes": ("https://cwwp2.dot.ca.gov/data/",),
    },
}
MIN_INTERVAL = 1.0  # between requests to one host
MAX_IMAGE_BYTES = 10 * 1024 * 1024
_last: dict[str, float] = {}


# ---------- HTTP straight into memory ----------
def _get(url: str, accept: str = "*/*", max_bytes: int | None = None, timeout: float = 30) -> bytes:
    """GET into memory; never cached on disk (frames must not be, and the TfL key stays off disk)."""
    host = urlparse(url).hostname or ""
    wait = _last.get(host, 0) + MIN_INTERVAL - time.time()
    if wait > 0:
        time.sleep(wait)
    _last[host] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": config.WEB_USER_AGENT, "Accept": accept})
    safe = re.sub(r"(app_key|api_key)=[^&]+", r"\1=***", url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(max_bytes + 1) if max_bytes else r.read()
    except urllib.error.HTTPError as e:
        raise FetchError(f"{safe}: HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise FetchError(f"{safe}: {e}") from e
    if max_bytes and len(body) > max_bytes:
        raise FetchError(f"{safe}: larger than {max_bytes} bytes")
    return body


def _get_json(url: str):
    return json.loads(_get(url, accept="application/json"))


# ---------- listings ----------
def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_tfl(data: list) -> list[dict]:
    """TfL Place/Type/JamCam -> cameras. Cameras TfL marks unavailable are left out."""
    p = PROVIDERS["TFL"]
    out = []
    for place in data or []:
        props = {a.get("key"): a.get("value") for a in place.get("additionalProperties") or []}
        pid = str(place.get("id", ""))
        m = re.fullmatch(r"JamCams_(\d+\.\d+)", pid)
        lat, lon = _num(place.get("lat")), _num(place.get("lon"))
        if not m or lat is None or lon is None or str(props.get("available", "true")).lower() != "true":
            continue
        img = props.get("imageUrl") or ""
        if not img.startswith(p["image_prefixes"]):
            continue
        out.append({"id": f"TFL.{m.group(1)}", "provider": "TFL", "name": place.get("commonName") or pid,
                    "lat": lat, "lon": lon, "image_url": img, "licence": p["licence"]})
    return out


def parse_caltrans(data: dict, district: int) -> list[dict]:
    """One Caltrans district's cctvStatus JSON -> cameras. Cameras not in service are left out."""
    p = PROVIDERS["CALTRANS"]
    out = []
    for row in (data or {}).get("data", []):
        c = row.get("cctv") or {}
        loc = c.get("location") or {}
        img = (((c.get("imageData") or {}).get("static") or {}).get("currentImageURL") or "").strip()
        lat, lon = _num(loc.get("latitude")), _num(loc.get("longitude"))
        if str(c.get("inService", "")).lower() != "true" or lat is None or lon is None:
            continue
        m = re.search(r"/cctv/image/([A-Za-z0-9_-]+)/[^/]+\.jpe?g$", img)
        if not m or not img.startswith(p["image_prefixes"]):
            continue
        name = loc.get("locationName") or m.group(1)
        if loc.get("nearbyPlace"):
            name = f"{name} ({loc['nearbyPlace']})"
        out.append({"id": f"CALTRANS.D{int(district):02d}.{m.group(1)}", "provider": "CALTRANS", "name": name,
                    "lat": lat, "lon": lon, "image_url": img, "licence": p["licence"]})
    return out


def _listing_path(provider: str) -> Path:
    d = config.home() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"cameras_{provider}.json"


def _fetch_listing(provider: str) -> list[dict]:
    p = PROVIDERS[provider]
    if provider == "TFL":
        key = os.environ.get("TFL_APP_KEY", "").strip()
        return parse_tfl(_get_json(p["listing"] + ("?" + urlencode({"app_key": key}) if key else "")))
    out, failed = [], []
    for n in CALTRANS_DISTRICTS:
        try:
            out += parse_caltrans(_get_json(p["listing"].format(n=n)), n)
        except (FetchError, ValueError) as e:
            failed.append(f"district {n}: {e}")
    if len(failed) == len(CALTRANS_DISTRICTS):  # an outage: do not cache an empty listing
        raise FetchError("; ".join(failed[:3]))
    for f in failed:
        print(f"  Caltrans {f} (listing unavailable; the rest is used)")
    return out


def cameras(provider: str | None = None) -> list[dict]:
    """Every camera of one provider (or all), from listings cached one day."""
    names = [provider.upper()] if provider else list(PROVIDERS)
    out = []
    for name in names:
        if name not in PROVIDERS:
            raise KeyError(f"unknown camera provider {provider!r}; known: {', '.join(PROVIDERS)}")
        path = _listing_path(name)
        try:
            cached = json.loads(path.read_text())
            fresh = time.time() - cached["epoch"] < LISTING_TTL
        except (OSError, ValueError, KeyError, TypeError):
            cached, fresh = None, False
        if fresh:
            out += cached["cameras"]
            continue
        try:
            cams = _fetch_listing(name)
        except (FetchError, ValueError) as e:
            if cached:  # a stale listing beats none
                out += cached["cameras"]
                continue
            raise FetchError(f"{name} camera listing unavailable: {e}") from e
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"epoch": time.time(), "fetched_at": now_iso(), "cameras": cams}))
        os.replace(tmp, path)
        out += cams
    return out


def km_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    la1, la2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(h, 1.0)))


def near(lat: float, lon: float, km: float = 1.0, provider: str | None = None) -> list[dict]:
    """Cameras within ``km``, nearest first, each with its distance as "km"."""
    hits = []
    for c in cameras(provider):
        d = km_between(lat, lon, c["lat"], c["lon"])
        if d <= km:
            hits.append({**c, "km": round(d, 3)})
    return sorted(hits, key=lambda c: (c["km"], c["id"]))


def official(cam: dict) -> bool:
    p = PROVIDERS.get(str(cam.get("provider", "")).upper())
    return bool(p) and str(cam.get("image_url", "")).startswith(p["image_prefixes"])


# ---------- sampling ----------
def fetch_frame(cam: dict) -> bytes:
    """One frame of an official camera, into memory only."""
    if not official(cam):
        raise PermissionError(f"{cam.get('id')}: not an official open camera image URL; refused")
    return _get(cam["image_url"], accept="image/jpeg,image/*", max_bytes=MAX_IMAGE_BYTES)


def sample(bank: Bank, camera_ids: list[str], counter=None, frame=fetch_frame, log=print) -> int:
    """Count vehicles in the current frame of each camera; one ``camera.vehicles`` fact per camera.
    Frames are discarded after counting. Raises ``brain.vision.Unavailable`` (writing nothing) when
    OpenCV or the model is missing. Returns facts written."""
    from brain import vision
    counter = counter or vision.VehicleCounter(log=log)
    by_id = {}
    for pv in sorted({str(c).split(".")[0].upper() for c in camera_ids} & set(PROVIDERS)):
        try:
            by_id.update((c["id"].upper(), c) for c in cameras(pv))
        except FetchError as e:
            log(f"  {e}")
    facts = []
    for cid in camera_ids:
        cam = by_id.get(str(cid).upper())
        if cam is None:
            log(f"  {cid}: not in any official camera listing; skipped")
            continue
        p = PROVIDERS[cam["provider"]]
        try:
            img = frame(cam)
        except (FetchError, PermissionError) as e:
            log(f"  {cid}: {e}")
            continue
        t = now_iso()
        try:
            n = counter.count(img)
        except ValueError as e:
            log(f"  {cid}: {e}")
            continue
        finally:
            del img  # the frame is gone once counted
        facts.append(Fact(entity=f"CAM.{cam['id']}", attr="camera.vehicles", value=int(n["vehicles"]),
                          unit="vehicles", as_of=t[:10], known_at=t, tier="reported",
                          source={"name": p["name"], "url": p["page"]},
                          context={"car": int(n["car"]), "truck": int(n["truck"]), "bus": int(n["bus"]),
                                   "motorcycle": int(n["motorcycle"]), "provider": cam["provider"],
                                   "name": cam["name"], "lat": cam["lat"], "lon": cam["lon"],
                                   "detector": vision.DETECTOR, "threshold": vision.SCORE,
                                   "decode": vision.DECODE},
                          retrieved_at=t))
    return bank.add_many(facts)


# ---------- CLI ----------
def _cmd_near(a) -> int:
    try:
        hits = near(a.lat, a.lon, a.km, a.provider)
    except (FetchError, KeyError) as e:
        print(e)
        return 1
    for c in hits:
        print(f"{c['km']:>6.2f} km  {c['id']:<40} {c['name']}")
    print(f"{len(hits)} official camera{'s' if len(hits) != 1 else ''} within {a.km} km")
    return 0


def _cmd_sample(a) -> int:
    from brain import vision
    bank = Bank()
    try:
        n = sample(bank, a.ids)
    except vision.Unavailable as e:
        print(f"{e}\nnothing written")
        return 1
    v = bank.view()
    for cid in a.ids:
        f = v.latest(f"CAM.{cid}", "camera.vehicles")
        if f is not None:
            c = f.context
            print(f"{cid}: {f.value} vehicle{'s' if f.value != 1 else ''} (car {c['car']}, truck {c['truck']}, bus {c['bus']}, "
                  f"motorcycle {c['motorcycle']}) at {f.known_at} [{f.id[:8]}]")
    print(f"{n} sample{'s' if n != 1 else ''} written")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("cams", help="official traffic cameras (TfL, Caltrans): find, count vehicles")
    cs = p.add_subparsers(dest="cams_cmd", required=True)
    s = cs.add_parser("near", help="cameras near a point, nearest first")
    s.add_argument("lat", type=float)
    s.add_argument("lon", type=float)
    s.add_argument("--km", type=float, default=1.0)
    s.add_argument("--provider", choices=sorted(PROVIDERS), type=str.upper)
    s.set_defaults(fn=_cmd_near)
    s = cs.add_parser("sample", help="count vehicles in each camera's current frame (frames are not kept)")
    s.add_argument("ids", nargs="+")
    s.set_defaults(fn=_cmd_sample)
