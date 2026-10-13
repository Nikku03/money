"""Commercial vessels in watched port areas, counted from AIS (aisstream.io).

The lines from CLAUDE.md ("Alternative data"), enforced here: only aggregate counts per area are
stored, never an individual vessel's track or identity. Messages are summarized in memory and
dropped; the fact holds counts by type and navigational status. The count itself (the fact's
value) is of commercial vessels only: cargo (AIS ship types 70-79) and tankers (80-89). Pleasure
craft and sailing vessels (36, 37) are dropped from every count. Only ship stations (MMSI
2xxxxxxxx to 7xxxxxxxx) count, not base stations, aids to navigation or SAR aircraft.

A vessel's type comes from its ShipStaticData broadcast (every six minutes), so the default window
(``DEFAULT_SECONDS``) is longer than that. A vessel heard without one could be a pleasure craft: it
is never in the count, only tallied as "untyped". Other typed vessels (tugs, passenger ships,
fishing vessels) are tallied as "other", outside the count too.

Source (credit it): aisstream.io, free key in ``AISSTREAM_API_KEY``. Needs the optional
``websockets`` package (``pip install money-brain[alt]``).
"""
from __future__ import annotations

import asyncio
import json
import os
import time

from brain import config
from brain.bank import Bank, Fact
from brain.timeutil import now_iso

AIS_URL = "wss://stream.aisstream.io/v0/stream"
SOURCE = {"name": "aisstream.io", "url": "https://aisstream.io/"}
CARGO, TANKER = range(70, 80), range(80, 90)
PLEASURE = {36, 37}  # sailing, pleasure craft: never counted
ANCHORED, MOORED = 1, 5  # AIS navigational status
DEFAULT_SECONDS = 420  # longer than the six-minute ShipStaticData cycle, so most vessels are typed
BASIS = "cargo+tanker"  # what the value counts; samples on another basis are not comparable

# bbox: [lat_min, lon_min, lat_max, lon_max], covering the terminals and the usual anchorages.
AREAS: list[dict] = [
    {"name": "LA-LONG-BEACH", "bbox": [33.60, -118.35, 33.80, -118.05]},
    {"name": "OAKLAND", "bbox": [37.70, -122.42, 37.85, -122.25]},
    {"name": "HOUSTON", "bbox": [29.25, -95.30, 29.78, -94.60]},
    {"name": "NEW-YORK-NJ", "bbox": [40.45, -74.20, 40.72, -73.95]},
    {"name": "ROTTERDAM", "bbox": [51.85, 3.50, 52.10, 4.55]},
    {"name": "SINGAPORE", "bbox": [1.15, 103.60, 1.35, 104.10]},
    {"name": "SHANGHAI", "bbox": [30.55, 121.45, 31.50, 122.40]},
    {"name": "NINGBO", "bbox": [29.80, 121.75, 30.05, 122.30]},
    {"name": "BUSAN", "bbox": [35.00, 128.75, 35.15, 129.15]},
    {"name": "JEBEL-ALI", "bbox": [24.95, 54.90, 25.10, 55.15]},
    {"name": "SUEZ", "bbox": [29.85, 32.25, 31.35, 32.65]},
    {"name": "PANAMA", "bbox": [8.85, -79.95, 9.45, -79.45]},
    {"name": "HORMUZ", "bbox": [25.90, 55.80, 26.80, 57.00]},
]


class Unavailable(RuntimeError):
    """AIS sampling cannot run here (no key, no websockets)."""


def _valid_bbox(b) -> bool:
    try:
        la0, lo0, la1, lo1 = (float(x) for x in b)
    except (TypeError, ValueError):
        return False
    return -90 <= la0 < la1 <= 90 and -180 <= lo0 < lo1 <= 180


def areas() -> list[dict]:
    """Built-in areas plus watchlist areas that carry their own bbox (var/watchlist.json)."""
    out = [dict(a) for a in AREAS]
    names = {a["name"] for a in out}
    try:
        w = json.loads((config.home() / "watchlist.json").read_text())
    except (OSError, ValueError):
        w = {}
    for a in w.get("areas", []):
        name = str(a.get("name", "")).upper()
        if name and name not in names and _valid_bbox(a.get("bbox")):
            out.append({"name": name, "bbox": [float(x) for x in a["bbox"]]})
            names.add(name)
    return out


def find_area(name_or_area) -> dict:
    """A built-in or watchlist area by name, or a {"name", "bbox"} dict checked and normalized."""
    if isinstance(name_or_area, dict):
        if not _valid_bbox(name_or_area.get("bbox")):
            raise ValueError(f"area needs bbox [lat_min, lon_min, lat_max, lon_max]: {name_or_area}")
        return {"name": str(name_or_area["name"]).upper(), "bbox": [float(x) for x in name_or_area["bbox"]]}
    key = str(name_or_area).upper()
    for a in areas():
        if a["name"] == key:
            return a
    raise KeyError(f"unknown area {name_or_area!r}; known: {', '.join(a['name'] for a in areas())}")


def _ship_mmsi(x) -> int | None:
    try:
        m = int(x)
    except (TypeError, ValueError):
        return None
    return m if 200_000_000 <= m < 800_000_000 else None


def _inside(bbox, lat, lon) -> bool:
    try:
        return bbox[0] <= float(lat) <= bbox[2] and bbox[1] <= float(lon) <= bbox[3]
    except (TypeError, ValueError):
        return False


def summarize(messages: list[dict], area: dict, window_s: int) -> dict:
    """Aggregate counts from aisstream messages. Pure; nothing about any one vessel is returned.

    A vessel is seen when it reported a position inside the bbox. Type from its ShipStaticData:
    70-79 cargo, 80-89 tanker, 36/37 dropped, any other type "other", no static data heard "untyped".
    ``vessels`` (the fact's value) is cargo + tanker; status, from each one's latest position report
    (1 anchored, 5 moored, else underway), is of those too. "other" and "untyped" are tallies beside it."""
    bbox = area["bbox"]
    types: dict[int, int] = {}
    status: dict[int, tuple[str, int, int]] = {}  # mmsi -> (time, order, nav status)
    for n, m in enumerate(messages):
        kind = m.get("MessageType")
        body = (m.get("Message") or {}).get(kind) or {}
        meta = m.get("MetaData") or {}
        mmsi = _ship_mmsi(body.get("UserID", meta.get("MMSI")))
        if mmsi is None:
            continue
        if kind == "ShipStaticData":
            try:
                types[mmsi] = int(body.get("Type"))
            except (TypeError, ValueError):
                pass
        elif kind == "PositionReport":
            lat = body.get("Latitude", meta.get("latitude"))
            lon = body.get("Longitude", meta.get("longitude"))
            if not _inside(bbox, lat, lon):
                continue
            try:
                nav = int(body.get("NavigationalStatus", 15))
            except (TypeError, ValueError):
                nav = 15
            key = (str(meta.get("time_utc", "")), n)
            if mmsi not in status or key >= status[mmsi][:2]:
                status[mmsi] = (*key, nav)
    out = {"vessels": 0, "cargo": 0, "tanker": 0, "other": 0, "untyped": 0,
           "anchored": 0, "moored": 0, "underway": 0, "window_s": int(window_s), "bbox": list(bbox),
           "basis": BASIS}
    for mmsi, (_, _, nav) in status.items():
        t = types.get(mmsi)
        if t is None:  # type not heard: possibly a pleasure craft, so never counted
            out["untyped"] += 1
            continue
        if t in PLEASURE:
            continue
        if t in CARGO:
            out["cargo"] += 1
        elif t in TANKER:
            out["tanker"] += 1
        else:
            out["other"] += 1
            continue
        out["vessels"] += 1
        out["anchored" if nav == ANCHORED else "moored" if nav == MOORED else "underway"] += 1
    return out


def to_fact(summary: dict, area: dict, ended_at: str) -> Fact:
    ctx = {k: v for k, v in summary.items() if k != "vessels"}
    return Fact(entity=f"AREA.{area['name']}", attr="ships.count", value=summary["vessels"], unit="vessels",
                as_of=ended_at[:10], known_at=ended_at, tier="reported", source=dict(SOURCE), context=ctx,
                retrieved_at=ended_at)


async def _collect(url: str, key: str, bbox: list[float], seconds: float) -> list[dict]:
    try:
        import websockets
    except ImportError as e:
        raise Unavailable("AIS sampling needs the websockets package: pip install websockets "
                          "(or pip install money-brain[alt])") from e
    sub = {"APIKey": key, "BoundingBoxes": [[[bbox[0], bbox[1]], [bbox[2], bbox[3]]]],
           "FilterMessageTypes": ["PositionReport", "ShipStaticData"]}
    out: list[dict] = []
    deadline = time.monotonic() + seconds
    async with websockets.connect(url, open_timeout=30, close_timeout=5,
                                  user_agent_header=config.WEB_USER_AGENT) as ws:
        await ws.send(json.dumps(sub))
        while (left := deadline - time.monotonic()) > 0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=left)
            except asyncio.TimeoutError:
                break
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if isinstance(msg, dict) and msg.get("error"):
                raise Unavailable(f"aisstream.io refused the subscription: {msg['error']}")
            if isinstance(msg, dict):
                out.append(msg)
    return out


def sample(bank: Bank, area, seconds: int = DEFAULT_SECONDS, key: str | None = None, url: str = AIS_URL) -> Fact:
    """Listen to AIS in the area for ``seconds``, write one ``ships.count`` fact (commercial vessels: cargo
    and tanker), return it."""
    key = key or os.environ.get("AISSTREAM_API_KEY", "").strip()
    if not key:
        raise Unavailable("AIS sampling needs a free aisstream.io key: export AISSTREAM_API_KEY=...")
    a = find_area(area)
    msgs = asyncio.run(_collect(url, key, a["bbox"], seconds))
    ended = now_iso()
    f = to_fact(summarize(msgs, a, seconds), a, ended)
    del msgs  # individual vessel messages are not kept
    bank.add(f)
    return f


# ---------- CLI ----------
def _cmd_sample(a) -> int:
    try:
        f = sample(Bank(), a.area, a.seconds)
    except (Unavailable, KeyError) as e:
        print(e)
        return 1
    c = f.context
    print(f"{f.entity}: {f.value} commercial vessels in {c['window_s']} s (cargo {c['cargo']}, tanker {c['tanker']}; "
          f"anchored {c['anchored']}, moored {c['moored']}, underway {c['underway']}); also seen: other {c['other']}, "
          f"untyped {c['untyped']} [{f.id[:8]}]")
    return 0


def _cmd_areas(a) -> int:
    for x in areas():
        print(f"{x['name']:<16} {x['bbox']}")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("ships", help="count commercial vessels in port areas (aisstream.io AIS)")
    ss = p.add_subparsers(dest="ships_cmd", required=True)
    s = ss.add_parser("sample", help="listen to one area and store the vessel counts")
    s.add_argument("area")
    s.add_argument("--seconds", type=int, default=DEFAULT_SECONDS,
                   help=f"listening window (default {DEFAULT_SECONDS}: longer than the six-minute static-data cycle)")
    s.set_defaults(fn=_cmd_sample)
    s = ss.add_parser("areas", help="list the port areas")
    s.set_defaults(fn=_cmd_areas)
