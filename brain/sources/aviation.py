"""Corporate aircraft: what the FAA says a company owns, and its airport-to-airport flights.

The lines from CLAUDE.md ("Alternative data"), enforced here rather than left to the caller:

- Only aircraft whose FAA registered owner is the company itself: a corporation or LLC registrant
  (``CORPORATE``) whose name *equals* the company's SEC name word for word, legal form included.
  Normalizing evens out spelling only: case, punctuation, "&" for AND, a leading THE, the SEC's
  state-of-incorporation tail ("/DE/") and how the legal form is written (CORPORATION = CORP,
  COMPANY = CO, INCORPORATED = INC, LIMITED = LTD). No word is ever dropped or joined to the next,
  so WEST LAKE HOLDINGS INC is not WESTLAKE CORP, GAP HOLDINGS LLC is not GAP INC and APA LLC is
  not APA CORP. An LLC registrant matches only a company whose SEC name is itself an LLC.
  Individuals (type 1), co-owners, partnerships and government registrants never match, whatever
  their name; a registration listing other co-owner names is skipped too. No fuzzy matching, no
  expansion to subsidiaries, trusts or other LLCs, no attempt to see through privacy programs.
- Aircraft that adsb.lol flags as FAA LADD or PIA (the owner asked not to be tracked) are dropped
  the moment we see the flag, and never polled or linked again.
- A registration or Mode S hex in ``var/aviation_optout.txt`` is skipped everywhere: never linked,
  never polled, never written.
- What was recorded about an aircraft before it was opted out or seen in LADD/PIA (its
  registration, its flights and the aircraft alerts citing them) is withdrawn: ``forget`` downgrades
  those facts, from ``jets optout``, ``link`` and every ``poll``.
- What is stored is a flight record from airport to airport with times. No track, no callsign, no
  inference about who was on board. The poll state keeps only each aircraft's last observation,
  needed to tell a landing, and live position responses are never cached on disk.
- Registrations are re-checked against a registry download at most a week old (``LINK_MAX_AGE_S``,
  measured from the download) before polling, so an aircraft the company sold stops being polled.
  When the registry cannot be fetched, a link whose download is older than ``LINK_HARD_AGE_S``, or
  that was made under an older owner-name rule (``MATCH_RULE``), is not polled at all.
- One poll at a time per home (a lock around the poll state), and every process's adsb.lol requests
  share one pace (``<home>/rate.adsb.lol``), so two pollers together stay at one request a second.

Sources (credit them): FAA Aircraft Registry (public domain), adsb.lol (ODbL), OurAirports
(public domain).
"""
from __future__ import annotations

import contextlib
import csv
import fcntl
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from brain import config
from brain.bank import Bank, Fact, View
from brain.net import FetchError, fetch
from brain.timeutil import now_iso

REGISTRY_URL = "https://registry.faa.gov/database/ReleasableAircraft.zip"
REGISTRY_TTL = 7 * 86400
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
AIRPORTS_TTL = 30 * 86400
ADSB_URL = "https://api.adsb.lol/v2/hex/{hex}"
ADSB_INTERVAL = 1.0  # adsb.lol: at most one request per second, from all processes together
ADSB_MAX_INTERVAL = 8.0  # after a 429 the spacing doubles, up to this, for the rest of the process
ADSB_RETRY_S = 5.0   # wait before the one retry of a 429 when the answer names no Retry-After
FAA_SOURCE = {"name": "FAA Aircraft Registry", "url": REGISTRY_URL}
ADSB_SOURCE = {"name": "adsb.lol (ODbL)", "url": "https://api.adsb.lol/"}

CORPORATE = {"3": "corporation", "7": "LLC", "8": "non-citizen corporation"}  # FAA TYPE REGISTRANT
VALID_STATUS = {"V", "T"}  # FAA STATUS CODE: valid registration (T: valid, from a trainee)
# Ways of writing a legal form, and the one spelling they compare as. HOLDINGS is not a legal form: it
# is part of a name, and GAP HOLDINGS LLC is another company than GAP INC.
LEGAL_FORMS = {"INC": "INC", "INCORPORATED": "INC", "CORP": "CORP", "CORPORATION": "CORP", "CO": "CO",
               "COMPANY": "CO", "LTD": "LTD", "LIMITED": "LTD", "PLC": "PLC", "LLC": "LLC"}
MATCH_RULE = 2  # the owner-name rule links are made under; a link made under another is re-checked first
AIRPORT_TYPES = ("large_airport", "medium_airport", "small_airport")
MAJOR_TYPES = ("large_airport", "medium_airport")  # the only fields an approach from afar is read as heading to
AIRPORT_KM = 5.0         # seen on the ground this close to a field: at that field
LOW_FT = 3000            # last seen below this height above a field, then gone for GONE_S: landed there
GONE_S = 20 * 60
LOST_S = 12 * 3600       # airborne and unseen this long: we no longer know where it is
GLIDE_DEG, GLIDE_MIN_DEG, GLIDE_MAX_DEG = 3.0, 2.0, 6.0  # an approach path, and the angles still taken as one
APPROACH_KM = 20.0       # LOW_FT above a field on a 3 degree path is about 17 km out
NEAR_KM = 3.0            # this close to a field, any height below LOW_FT above it counts (overhead, the pattern)
TRACK_DEG = 45.0         # farther out, the field must lie within this of the aircraft's track, when known
CLIMB_FPM = 300          # last seen climbing faster than this (ft/min): not landing
VERT_FPM = 4000          # quickest likely climb or descent, ft/min
STOP_S = 15 * 60         # shortest likely stop on the ground
HOP_S = 10 * 60          # shortest likely flight from one field to another
MIN_LEG_S = 10 * 60      # last seen low this soon after taking off, and not descending: climbing out
CLIMB_KM_PER_KFT = 10.0  # farthest a departing aircraft gets from its field per 1,000 ft it has climbed
FT_M = 0.3048
LINK_MAX_AGE_S = 7 * 86400    # registry data older than this is re-checked before polling
LINK_HARD_AGE_S = 14 * 86400  # registry data this old that cannot be refreshed: not polled
DB_PIA, DB_LADD = 4, 8   # readsb dbFlags bits
UNTRACKED = "untracked"  # downgrade reasons of ``forget`` start with this

_last_live = 0.0
_live_interval = ADSB_INTERVAL
_registry_cache: dict = {}
_airports_cache: dict = {}


# ---------- names ----------
def normalize_name(name: str) -> str:
    """Comparable form of a company name, for an exact match and nothing looser.

    Upper case; SEC's state-of-incorporation tail ("/DE", "/NEW/") dropped; "&" read as AND; other
    punctuation splits words; a leading THE dropped; the trailing legal form written one way
    (CORPORATION -> CORP, COMPANY -> CO, ...). Every other word stays, in order, space-separated:
    "AMAZON.COM, INC." equals "AMAZON COM INC", but "WEST LAKE HOLDINGS INC" never equals "WESTLAKE CORP".
    """
    s = re.sub(r"\s*/[A-Z.]*/?\s*$", "", str(name).upper().strip())
    s = s.replace("&", " AND ")
    words = [w for w in re.split(r"[^A-Z0-9]+", s.replace("'", "")) if w]
    if words[:1] == ["THE"] and len(words) > 1:
        words = words[1:]
    if len(words) > 3 and words[-3:] == ["L", "L", "C"]:  # "L.L.C."
        words[-3:] = ["LLC"]
    i = len(words)
    while i > 0 and words[i - 1] in LEGAL_FORMS:
        i -= 1
    return " ".join(words[:i] + [LEGAL_FORMS[w] for w in words[i:]])


def is_llc(normalized: str) -> bool:
    return normalized.rsplit(" ", 1)[-1] == "LLC"


def norm_reg(reg: str) -> str:
    """'n123ab' / '123AB' -> 'N123AB'."""
    r = str(reg).strip().upper()
    return r if r.startswith("N") else "N" + r


# ---------- opt-out ----------
def optout_path() -> Path:
    return config.home() / "aviation_optout.txt"


def optout() -> set[str]:
    """Registrations and hex codes never to track: one per line, '#' starts a comment."""
    p = optout_path()
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        tok = line.split("#", 1)[0].strip().upper()
        if tok:
            out.add(tok)
            if not tok.startswith("N") and not re.fullmatch(r"[0-9A-F]{6}", tok):
                out.add("N" + tok)
    return out


def opted_out(reg: str | None, hex_: str | None, out: set[str] | None = None) -> bool:
    out = optout() if out is None else out
    keys = set()
    if reg:
        keys |= {norm_reg(reg), norm_reg(reg)[1:]}
    if hex_:
        keys.add(str(hex_).strip().upper())
    return bool(keys & out)


def blocked_hexes(state_path: Path | str | None = None) -> set[str]:
    """Mode S codes the poller found in the FAA's LADD or PIA programs (the owner asked not to be tracked)."""
    st = _read_json(Path(state_path) if state_path else default_state_path(), {})
    try:
        return {str(h).lower() for h in (st.get("blocked") or {})}
    except AttributeError:
        return set()


def untracked(reg: str | None, hex_: str | None, out: set[str] | None = None,
              blocked: set[str] | None = None) -> bool:
    """True when this aircraft must not be tracked or shown: opted out, or in LADD/PIA."""
    blocked = blocked_hexes() if blocked is None else blocked
    return opted_out(reg, hex_, out) or str(hex_ or "").strip().lower() in blocked


def add_optout(token: str, bank: Bank | None = None) -> int:
    """Never track this registration or hex again, and withdraw what was recorded about it (``forget``).
    Returns the number of facts withdrawn."""
    p = optout_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(token.strip().upper() + "\n")
    return forget(bank if bank is not None else Bank())


def forget(bank: Bank, tickers: list[str] | None = None, blocked: set[str] | None = None) -> int:
    """Downgrade what the bank holds about aircraft that must not be tracked (opted out, or in LADD/PIA):
    their ``aircraft.registered`` and ``aircraft.flight`` facts, and the aircraft alerts that cite them.
    ``tickers`` defaults to every company ever linked or watched. Returns how many facts were downgraded."""
    out = optout()
    blocked = blocked_hexes() if blocked is None else {str(h).lower() for h in blocked}
    if not out and not blocked:
        return 0
    if tickers is None:
        tickers = list(_read_json(_links_path(), {})) + watched()
    n = 0
    for t in dict.fromkeys(str(x).upper() for x in tickers):
        gone: set[str] = set()  # refs of untracked facts, withdrawn now or before
        hexes: set[str] = set()
        for attr in ("aircraft.registered", "aircraft.flight"):
            live = {f.id for f in bank.facts(t, attr)}
            for f in bank.facts(t, attr, include_downgraded=True):
                reg = f.value if attr == "aircraft.registered" else f.context.get("reg")
                hex_ = str(f.context.get("hex") or "").lower()
                if opted_out(reg, hex_, out):
                    why = f"{UNTRACKED}: {reg or hex_} is in aviation_optout.txt"
                elif hex_ in blocked:
                    why = f"{UNTRACKED}: {hex_} is in the FAA LADD/PIA program (the owner asked not to be tracked)"
                else:
                    continue
                gone.add(f.ref)
                hexes.add(hex_)
                if f.id in live:
                    bank.downgrade(t, f.id, why)
                    n += 1
        if not gone:
            continue
        for f in bank.facts(t, "alert"):
            if f.context.get("rule") != "aircraft":
                continue
            cited = {str(k).split(":")[1].lower() for k in f.context.get("keys") or []
                     if str(k).startswith("aircraft:") and str(k).count(":") >= 2}
            if set(f.inputs) & gone or cited & hexes:
                bank.downgrade(t, f.id, f"{UNTRACKED}: cites an aircraft that is no longer tracked")
                n += 1
    return n


# ---------- FAA registry ----------
def _rows(raw: bytes):
    text = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", errors="replace", newline="")
    reader = csv.reader(text)
    header = [h.strip() for h in next(reader)]
    for row in reader:
        yield {h: (row[i].strip() if i < len(row) else "") for i, h in enumerate(header) if h}


def _member(z: zipfile.ZipFile, name: str) -> str:
    for n in z.namelist():
        if n.upper().endswith(name.upper()):
            return n
    raise KeyError(f"{name} not in the FAA registry archive")


def parse_registry(zip_bytes: bytes) -> dict[str, list[dict]]:
    """Normalized owner name -> aircraft, for corporate and LLC registrants with a valid registration.

    Only the fields needed to identify the aircraft are kept (no addresses). Rows of any other
    registrant type are read past and dropped here, before anything can match them."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        models = {}
        for r in _rows(z.read(_member(z, "ACFTREF.txt"))):
            models[r.get("CODE", "")] = " ".join(x for x in (r.get("MFR", ""), r.get("MODEL", "")) if x)
        index: dict[str, list[dict]] = {}
        for r in _rows(z.read(_member(z, "MASTER.txt"))):
            kind = CORPORATE.get(r.get("TYPE REGISTRANT", ""))
            if kind is None or r.get("STATUS CODE", "") not in VALID_STATUS:
                continue
            if any(r.get(f"OTHER NAMES({i})") for i in range(1, 6)):
                continue  # co-owners listed: not the company alone
            hex_ = r.get("MODE S CODE HEX", "").lower()
            owner = r.get("NAME", "")
            if not (owner and re.fullmatch(r"[0-9a-f]{6}", hex_)):
                continue
            year = r.get("YEAR MFR", "")
            index.setdefault(normalize_name(owner), []).append({
                "reg": norm_reg(r["N-NUMBER"]), "hex": hex_, "model": models.get(r.get("MFR MDL CODE", "")) or None,
                "year": int(year) if year.isdigit() else None, "owner": owner, "registrant": kind,
            })
    return index


def registry() -> tuple[dict[str, list[dict]], str]:
    """The parsed registry and when it was fetched (downloaded at most weekly)."""
    r = fetch(REGISTRY_URL, ttl=REGISTRY_TTL)
    if _registry_cache.get("fetched_at") != r.fetched_at:
        _registry_cache.clear()
        _registry_cache.update(index=parse_registry(r.body), fetched_at=r.fetched_at)
    return _registry_cache["index"], _registry_cache["fetched_at"]


def company_aircraft(name: str, index: dict[str, list[dict]] | None = None) -> list[dict]:
    """Aircraft registered to the company itself: a corporate/LLC registrant whose normalized owner name
    equals the normalized company name (legal form included), an LLC registrant only for a company that
    is itself an LLC, and not opted out or in LADD/PIA. ``index`` defaults to the current FAA registry."""
    if index is None:
        index, _ = registry()
    out, blocked = optout(), blocked_hexes()
    key = normalize_name(name)
    llc = is_llc(key)
    hits = [dict(a) for a in index.get(key, [])
            if (a["registrant"] == "LLC") == llc and not untracked(a["reg"], a["hex"], out, blocked)]
    return sorted(hits, key=lambda a: a["reg"])


# ---------- links: which aircraft each watched company owns ----------
def _links_path() -> Path:
    return config.home() / "aviation_links.json"


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _write_json(p: Path, obj) -> None:
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, p)


@contextlib.contextmanager
def _locked(p: Path):
    """An exclusive lock on ``<p>.lock``, shared by every process using this home."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.with_name(p.name + ".lock").open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def company_name(bank: Bank, ticker: str) -> str:
    f = bank.view().latest(ticker, "identity.name")
    if f is None:
        raise KeyError(f"no identity.name for {ticker.upper()} in the bank; run: python -m brain ingest {ticker.upper()}")
    return str(f.value)


def link(bank: Bank, ticker: str, index: dict[str, list[dict]] | None = None, fetched_at: str | None = None) -> int:
    """Write the company's registered aircraft as facts (``aircraft.registered``) and remember them as the
    set ``poll`` watches. Returns how many aircraft are registered to the company now. The facts of
    one registry download share its date, so re-linking against the same download adds nothing."""
    t = ticker.upper()
    name = company_name(bank, t)
    if index is None:
        index, fetched_at = registry()
    fetched_at = fetched_at or now_iso()
    planes = company_aircraft(name, index)
    bank.add_many(Fact(entity=t, attr="aircraft.registered", value=a["reg"], as_of=fetched_at[:10],
                       known_at=fetched_at, tier="reported", source=dict(FAA_SOURCE),
                       context={"hex": a["hex"], "model": a["model"], "year": a["year"], "owner": a["owner"]},
                       retrieved_at=fetched_at) for a in planes)
    with _locked(_links_path()):
        links = _read_json(_links_path(), {})
        links[t] = {"name": name, "linked_at": now_iso(), "registry_fetched_at": fetched_at, "rule": MATCH_RULE,
                    "aircraft": [{"reg": a["reg"], "hex": a["hex"]} for a in planes]}
        _write_json(_links_path(), links)
    forget(bank, [t])  # tails opted out since an earlier link of the same download
    return len(planes)


def registered(view: View, ticker: str) -> list[Fact]:
    """The company's aircraft as of the latest registry download visible in ``view`` (one fact per tail),
    without opted-out or LADD/PIA aircraft. Do not use ``View.periods`` for this attribute: tails linked
    the same day share a period key."""
    fs = view.facts(ticker, "aircraft.registered")
    if not fs:
        return []
    last = max(f.as_of for f in fs)
    out, blocked = optout(), blocked_hexes()
    best: dict[str, Fact] = {}
    for f in fs:
        if f.as_of == last and (f.value not in best or f.known_at >= best[f.value].known_at):
            best[f.value] = f
    return sorted((f for f in best.values() if not untracked(f.value, f.context.get("hex"), out, blocked)),
                  key=lambda f: f.value)


def flights(view: View, ticker: str) -> list[Fact]:
    """Recorded flights (``aircraft.flight``), newest arrival first, without opted-out or LADD/PIA aircraft."""
    out, blocked = optout(), blocked_hexes()
    fs = [f for f in view.facts(ticker, "aircraft.flight")
          if not untracked(f.context.get("reg"), f.context.get("hex"), out, blocked)]
    return sorted(fs, key=lambda f: (f.context.get("arrived") or f.as_of, f.known_at), reverse=True)


# ---------- airports ----------
def parse_airports(body: bytes) -> dict:
    text = io.StringIO(body.decode("utf-8-sig", errors="replace"))
    ident, name, lat, lon, elev, major = [], [], [], [], [], []
    for r in csv.DictReader(text):
        if r.get("type") not in AIRPORT_TYPES:
            continue
        try:
            la, lo = float(r["latitude_deg"]), float(r["longitude_deg"])
        except (TypeError, ValueError):
            continue
        try:
            el = float(r.get("elevation_ft") or 0)
        except ValueError:
            el = 0.0
        ident.append(r.get("ident") or r.get("gps_code") or "")
        name.append(r.get("name") or "")
        lat.append(la)
        lon.append(lo)
        elev.append(el)
        major.append(r.get("type") in MAJOR_TYPES)
    return {"ident": ident, "name": name, "lat": np.radians(lat), "lon": np.radians(lon),
            "elev": np.array(elev, dtype=float), "major": np.array(major, dtype=bool)}


def airports() -> dict:
    r = fetch(AIRPORTS_URL, ttl=AIRPORTS_TTL)
    if _airports_cache.get("fetched_at") != r.fetched_at:
        _airports_cache.clear()
        _airports_cache.update(table=parse_airports(r.body), fetched_at=r.fetched_at)
    return _airports_cache["table"]


def _km(lat1, lon1, lat2, lon2):
    """Great-circle km; the second point may be arrays in radians, the first is in degrees."""
    la, lo = np.radians(lat1), np.radians(lon1)
    h = np.sin((lat2 - la) / 2) ** 2 + np.cos(la) * np.cos(lat2) * np.sin((lon2 - lo) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(h, 1.0)))


def _bearing(lat, lon, lat2, lon2):
    """Initial bearing in degrees from (lat, lon) in degrees to points in radians."""
    la, lo = np.radians(lat), np.radians(lon)
    y = np.sin(lon2 - lo) * np.cos(lat2)
    x = np.cos(la) * np.sin(lat2) - np.sin(la) * np.cos(lat2) * np.cos(lon2 - lo)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def nearest_airport(table: dict, lat: float, lon: float, km: float = AIRPORT_KM) -> dict | None:
    """The closest large, medium or small airport within ``km``, else None."""
    if lat is None or lon is None or not len(table["lat"]):
        return None
    d = _km(lat, lon, table["lat"], table["lon"])
    i = int(np.argmin(d))
    if d[i] > km:
        return None
    return {"ident": table["ident"][i], "name": table["name"][i], "km": float(d[i])}


def landing_airport(table: dict, last: dict) -> dict | None:
    """The field an aircraft last seen airborne at ``last`` ({"alt", "lat", "lon", "track", "rate"}) was
    landing at, or None. Height is taken above each field (OurAirports elevation), not above sea level,
    and must be below LOW_FT. The nearest such field within NEAR_KM wins (overhead, in the pattern, short
    final). Farther out, up to APPROACH_KM, only a large or medium airport the aircraft is on an approach
    path to counts: GLIDE_MIN_DEG to GLIDE_MAX_DEG above it and, when the track is known, ahead of it; the
    one closest to a 3 degree path wins. So an approach to a big airport that passes a small strip is not
    a landing at the strip, and a private strip far off is never guessed. Last seen climbing: None."""
    alt, lat, lon = last.get("alt"), last.get("lat"), last.get("lon")
    if alt is None or lat is None or lon is None or not len(table["lat"]):
        return None
    if last.get("rate") is not None and last["rate"] > CLIMB_FPM:
        return None
    d = _km(lat, lon, table["lat"], table["lon"])
    n = len(d)
    h = float(alt) - np.asarray(table.get("elev", np.zeros(n)), dtype=float)
    low = (h >= -300) & (h < LOW_FT)
    near = low & (d <= NEAR_KM)
    if near.any():
        i = int(np.argmin(np.where(near, d, np.inf)))
        return {"ident": table["ident"][i], "name": table["name"][i], "km": float(d[i])}
    ang = np.degrees(np.arctan2(h * FT_M, np.maximum(d, 0.01) * 1000.0))
    ok = low & np.asarray(table.get("major", np.ones(n, dtype=bool)), dtype=bool) & (d <= APPROACH_KM) \
        & (ang >= GLIDE_MIN_DEG) & (ang <= GLIDE_MAX_DEG)
    if last.get("track") is not None:
        off = np.abs((_bearing(lat, lon, table["lat"], table["lon"]) - float(last["track"]) + 180.0) % 360.0 - 180.0)
        ok &= off <= TRACK_DEG
    if not ok.any():
        return None
    i = int(np.argmin(np.where(ok, np.abs(ang - GLIDE_DEG), np.inf)))
    return {"ident": table["ident"][i], "name": table["name"][i], "km": float(d[i])}


# ---------- positions ----------
def _retry_after(e: urllib.error.HTTPError) -> float:
    try:
        return min(60.0, max(1.0, float(e.headers.get("Retry-After") or ADSB_RETRY_S)))
    except (TypeError, ValueError, AttributeError):  # an HTTP date, or no headers
        return ADSB_RETRY_S


def _live_turn() -> None:
    """Wait for the next adsb.lol turn and claim it. The time of the last request by any process using
    this home lives in ``<home>/rate.adsb.lol``, read and written under an exclusive lock, so two pollers
    keep one pace together instead of one each."""
    global _last_live
    p = config.home() / "rate.adsb.lol"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            with contextlib.suppress(ValueError):
                _last_live = max(_last_live, float(fh.read().strip() or 0))
            wait = _last_live + _live_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            _last_live = time.time()
            fh.seek(0)
            fh.truncate()
            fh.write(repr(_last_live))
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _live_json(url: str, retry: bool = True) -> dict:
    """GET JSON straight into memory: live positions are never written to the HTTP cache. adsb.lol
    publishes no fixed rate, it limits by load: a 429 slows every later request down and is retried once."""
    global _live_interval
    _live_turn()
    req = urllib.request.Request(url, headers={"User-Agent": config.WEB_USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429 and retry:
            _live_interval = min(ADSB_MAX_INTERVAL, _live_interval * 2)
            time.sleep(_retry_after(e))
            return _live_json(url, retry=False)
        raise FetchError(f"{url}: HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
        raise FetchError(f"{url}: {e}") from e


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def observe(ac: dict, now: float) -> dict:
    """One adsb.lol aircraft record -> {"t", "ground", "alt", "lat", "lon", "track", "rate", "flags"}.
    ``ground`` is None when the record gives no altitude; ``rate`` is the vertical rate in ft/min."""
    alt = ac.get("alt_baro", ac.get("alt_geom"))
    ground = None
    if alt == "ground":
        ground, alt = True, 0
    elif isinstance(alt, (int, float)):
        ground, alt = False, float(alt)
    else:
        alt = None
    lat, lon = ac.get("lat"), ac.get("lon")
    if lat is None and isinstance(ac.get("lastPosition"), dict):
        lat, lon = ac["lastPosition"].get("lat"), ac["lastPosition"].get("lon")
    seen = ac.get("seen_pos", ac.get("seen", 0)) or 0
    track = _num(ac.get("track", ac.get("true_heading")))
    rate = _num(ac.get("baro_rate", ac.get("geom_rate")))
    return {"t": now - float(seen), "ground": ground, "alt": alt, "lat": lat, "lon": lon, "track": track,
            "rate": rate, "flags": int(ac.get("dbFlags") or 0)}


def _stop_fits(st: dict, obs: dict) -> bool:
    """Could the aircraft, last seen airborne as ``st``, have landed somewhere, stood and taken off again
    unseen before this sighting? Then the leg it was on is not the one it is on now."""
    if "t" not in st:
        return False
    after = 0.0 if obs["ground"] else float(obs.get("alt") or 0)
    need = STOP_S + 60.0 * (float(st.get("alt") or 0) + after) / VERT_FPM + (HOP_S if obs["ground"] else 0)
    return obs["t"] - st["t"] > need


def _left_from(st: dict, obs: dict) -> bool:
    """Did an aircraft last seen on the ground as ``st`` take off from there to be where ``obs`` is? Yes
    when it was seen there within GONE_S, or is no farther from it than a climb to its height takes it;
    after a long unseen spell farther away, it may have flown elsewhere first."""
    if obs["t"] - st.get("t", obs["t"]) <= GONE_S:
        return True
    if None in (st.get("lat"), st.get("lon"), obs.get("lat"), obs.get("lon")):
        return False
    reach = max(NEAR_KM, CLIMB_KM_PER_KFT * float(obs.get("alt") or 0) / 1000.0)
    return float(_km(obs["lat"], obs["lon"], np.radians(st["lat"]), np.radians(st["lon"]))) <= reach


def _climbing_out(st: dict) -> bool:
    """Last seen within MIN_LEG_S of its takeoff and not seen descending: low because it had just left,
    not because it was landing (else a climb-out that went quiet reads as a landing back where it began)."""
    if not st.get("departed") or (st.get("rate") is not None and st["rate"] < -CLIMB_FPM):
        return False
    return st["t"] - _epoch(st["departed"]) < MIN_LEG_S


def step(st: dict | None, obs: dict | None, now: float, table: dict) -> tuple[dict, dict | None]:
    """Advance one aircraft's state by one poll. Returns (new state, event); an event is
    {"kind": "takeoff", ...} or {"kind": "landing", "from", "from_name", "to", "to_name", "departed",
    "arrived"}. A landing is seen on the ground near a field, or inferred when the aircraft was last
    seen low on the way to one (``landing_airport``) and has been gone for GONE_S. Where the aircraft
    could have stopped somewhere unseen (``_stop_fits``, ``_left_from``), the origin is left unknown
    rather than carried over from an earlier leg."""
    st = dict(st or {"phase": "unknown"})
    if obs is None:
        if st["phase"] == "air" and "t" in st:
            gone = now - st["t"]
            if gone > GONE_S and st.get("alt") is not None:
                apt = landing_airport(table, st)
                if apt is not None and not _climbing_out(st):
                    ev = _landing(st, apt, st["t"])
                    return _on_ground(st["t"], apt, st.get("lat"), st.get("lon")), ev
                if apt is None and st["alt"] < LOW_FT:
                    return {"phase": "unknown", "t": st["t"]}, None
            if gone > LOST_S:
                return {"phase": "unknown", "t": st["t"]}, None
        return st, None
    if obs["ground"] is None:
        return st, None
    if obs["ground"]:
        apt = nearest_airport(table, obs["lat"], obs["lon"])
        ev = None
        if st["phase"] == "air" and apt:
            if _stop_fits(st, obs):
                st = {"phase": "air", "from": None, "from_name": None, "departed": None}
            ev = _landing(st, apt, obs["t"])
        return _on_ground(obs["t"], apt, obs["lat"], obs["lon"]), ev
    ev = None
    if st["phase"] == "ground":
        known = bool(st.get("airport")) and _left_from(st, obs)
        st = {"phase": "air", "from": st.get("airport") if known else None,
              "from_name": st.get("airport_name") if known else None,
              "departed": _iso(obs["t"]) if known else None}
        ev = {"kind": "takeoff", "from": st["from"], "departed": st["departed"]}
    elif st["phase"] != "air" or _stop_fits(st, obs):
        st = {"phase": "air", "from": None, "from_name": None, "departed": None}
    st.update(t=obs["t"], alt=obs["alt"], lat=obs["lat"], lon=obs["lon"], track=obs.get("track"),
              rate=obs.get("rate"))
    return st, ev


def _on_ground(t: float, apt: dict | None, lat, lon) -> dict:
    return {"phase": "ground", "t": t, "airport": apt["ident"] if apt else None,
            "airport_name": apt["name"] if apt else None, "lat": lat, "lon": lon}


def _landing(st: dict, apt: dict, t: float) -> dict:
    return {"kind": "landing", "from": st.get("from"), "from_name": st.get("from_name"), "to": apt["ident"],
            "to_name": apt["name"], "departed": st.get("departed"), "arrived": _iso(t)}


def default_state_path() -> Path:
    return config.home() / "aviation_state.json"


def watched() -> list[str]:
    """Tickers in the watchlist's ``aircraft_of`` (var/watchlist.json)."""
    w = _read_json(config.home() / "watchlist.json", {})
    return [str(t).upper() for t in w.get("aircraft_of", [])]


def _data_age(lk: dict) -> float:
    """Seconds since the registry download a link was made from."""
    try:
        return time.time() - _epoch(str(lk.get("registry_fetched_at") or lk.get("linked_at")))
    except (TypeError, ValueError):
        return float("inf")


def _current_links(bank: Bank, tickers: list[str], log) -> dict[str, list[dict]]:
    links = _read_json(_links_path(), {})
    stale = [t for t in tickers
             if t not in links or links[t].get("rule") != MATCH_RULE or _data_age(links[t]) > LINK_MAX_AGE_S]
    if stale:
        try:
            index, fetched_at = registry()
            for t in stale:
                try:
                    link(bank, t, index, fetched_at)
                except KeyError as e:
                    log(f"  {t}: {e}")
        except FetchError as e:
            log(f"  FAA registry unavailable ({e}); using existing links")
        links = _read_json(_links_path(), {})
    out = {}
    for t in tickers:
        lk = links.get(t)
        if lk is None:
            continue
        if lk.get("rule") != MATCH_RULE:
            log(f"  {t}: registrations were matched under an older owner-name rule and could not be re-checked; "
                "not polling its aircraft")
            continue
        if _data_age(lk) > LINK_HARD_AGE_S:
            log(f"  {t}: registrations not re-checked against a registry download of the last "
                f"{LINK_HARD_AGE_S // 86400} days; not polling its aircraft")
            continue
        out[t] = lk["aircraft"]
    return out


def poll(bank: Bank, state_path: Path | str | None = None, tickers: list[str] | None = None,
         get=None, table: dict | None = None, now: float | None = None, log=print) -> int:
    """Read the current position of every registered aircraft of the watched companies once, and write
    an ``aircraft.flight`` fact for each landing detected. Returns facts written. One poll runs at a
    time per state file: a second waits for the first, so neither undoes the other's LADD blocks."""
    sp = Path(state_path) if state_path else default_state_path()
    tickers = [t.upper() for t in (tickers or watched())]
    if not tickers:
        log("  no companies to poll: pass tickers, or add one with: python -m brain watch add aircraft TICKER")
        return 0
    with _locked(sp):
        return _poll(bank, sp, tickers, get, table, now, log)


def _poll(bank: Bank, sp: Path, tickers: list[str], get, table, now, log) -> int:
    links = _current_links(bank, tickers, log)
    for t in tickers:
        if t in links and not links[t]:
            log(f"  {t}: the FAA registry lists no aircraft registered to the company itself; nothing to poll")
    out = optout()
    planes: dict[str, dict] = {}
    for t, lst in links.items():
        for a in lst:
            planes.setdefault(a["hex"], {"reg": a["reg"], "tickers": []})["tickers"].append(t)
    state = _read_json(sp, {"aircraft": {}, "blocked": {}})
    state.setdefault("aircraft", {})
    state.setdefault("blocked", {})
    get = get or (lambda h: _live_json(ADSB_URL.format(hex=h)))
    table = table if table is not None else (airports() if planes else {"lat": []})
    facts = []
    read = 0
    for hex_, p in sorted(planes.items()):
        if opted_out(p["reg"], hex_, out):
            state["aircraft"].pop(hex_, None)
            continue
        if hex_ in state["blocked"]:
            continue
        try:
            data = get(hex_)
        except FetchError as e:
            log(f"  {p['reg']}: {e}")
            continue
        read += 1
        t_now = now if now is not None else time.time()
        acs = [a for a in data.get("ac") or [] if str(a.get("hex", "")).lower().lstrip("~") == hex_]
        obs = observe(acs[0], t_now) if acs else None
        if obs and obs["flags"] & (DB_PIA | DB_LADD):
            state["blocked"][hex_] = "FAA LADD/PIA: the owner asked not to be tracked"
            state["aircraft"].pop(hex_, None)
            log(f"  {p['reg']}: in FAA LADD/PIA; dropped and never polled again")
            continue
        st, ev = step(state["aircraft"].get(hex_), obs, t_now, table)
        state["aircraft"][hex_] = st
        if ev and ev["kind"] == "landing":
            seen = _iso(t_now)
            for tk in p["tickers"]:
                facts.append(Fact(entity=tk, attr="aircraft.flight", value=ev["to"], as_of=ev["arrived"][:10],
                                  known_at=seen, tier="reported", source=dict(ADSB_SOURCE),
                                  context={"hex": hex_, "reg": p["reg"], "from": ev["from"], "to": ev["to"],
                                           "from_name": ev["from_name"], "to_name": ev["to_name"],
                                           "departed": ev["departed"], "arrived": ev["arrived"]},
                                  retrieved_at=seen))
            log(f"  {p['reg']}: {ev['from'] or '?'} -> {ev['to']} arrived {ev['arrived']}")
    # An aircraft no company here or in the watchlist owns any more: forget its last position. A poll of a
    # few tickers by hand keeps the watchlist's aircraft, or the loop would miss their next landing.
    links_now = _read_json(_links_path(), {})
    keep = set(planes) | {a["hex"] for t in watched() for a in (links_now.get(t) or {}).get("aircraft", [])
                          if (links_now.get(t) or {}).get("rule") == MATCH_RULE
                          and not opted_out(a.get("reg"), a["hex"], out)}
    for hex_ in [h for h in state["aircraft"] if h not in keep]:
        state["aircraft"].pop(hex_)
    _write_json(sp, state)
    if planes:
        log(f"  {read} of {len(planes)} aircraft read from adsb.lol")
    n = bank.add_many(facts)
    withdrawn = forget(bank, blocked=set(state["blocked"]) | blocked_hexes())
    if withdrawn:
        log(f"  {withdrawn} fact{'s' if withdrawn != 1 else ''} about opted-out or LADD/PIA aircraft withdrawn")
    return n


# ---------- CLI ----------
def _cmd_link(a) -> int:
    bank = Bank()
    try:
        index, fetched_at = registry()
    except FetchError as e:
        print(f"FAA registry unavailable: {e}")
        return 1
    for t in a.tickers:
        t = t.upper()
        try:
            n = link(bank, t, index, fetched_at)
        except KeyError as e:
            print(f"{t}: {e}")
            continue
        name = company_name(bank, t)
        print(f"{t} ({name}): {n} aircraft registered to the company (FAA registry {fetched_at[:10]})")
        current = {x["reg"] for x in (_read_json(_links_path(), {}).get(t) or {}).get("aircraft", [])}
        for f in registered(bank.view(), t) if n else []:
            if f.value not in current:
                continue
            c = f.context
            print(f"  {f.value:<8} hex {c['hex']}  {c['model'] or '?'}  {c['year'] or ''}  owner {c['owner']}")
    return 0


def _cmd_poll(a) -> int:
    n = poll(Bank(), tickers=a.tickers or None)
    print(f"{n} flight{'s' if n != 1 else ''} recorded")
    return 0


def _cmd_optout(a) -> int:
    n = add_optout(a.token)
    print(f"{a.token.upper()} added to {optout_path()}; it will never be linked or polled"
          + (f"; {n} recorded fact{'s' if n != 1 else ''} about it withdrawn" if n else ""))
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("jets", help="company-registered aircraft (FAA) and their flights (adsb.lol)")
    js = p.add_subparsers(dest="jets_cmd", required=True)
    s = js.add_parser("link", help="find aircraft the FAA lists as registered to the company itself")
    s.add_argument("tickers", nargs="+")
    s.set_defaults(fn=_cmd_link)
    s = js.add_parser("poll", help="read positions once; record landings airport to airport")
    s.add_argument("tickers", nargs="*", help="default: the watchlist's aircraft_of")
    s.set_defaults(fn=_cmd_poll)
    s = js.add_parser("optout", help="never track this registration or Mode S hex")
    s.add_argument("token")
    s.set_defaults(fn=_cmd_optout)
