"""Corporate aircraft: what the FAA says a company owns, and its airport-to-airport flights.

The lines from CLAUDE.md ("Alternative data"), enforced here rather than left to the caller:

- Only aircraft whose FAA registrant type is a corporation or LLC (``CORPORATE``) and whose
  registered owner name, normalized, *equals* the company's SEC name. Individuals (type 1),
  co-owners, partnerships and government registrants never match, whatever their name; a
  registration listing other co-owner names is skipped too. No fuzzy matching, no expansion to
  trusts or other LLCs, no attempt to see through privacy programs.
- Aircraft that adsb.lol flags as FAA LADD or PIA (the owner asked not to be tracked) are dropped
  the moment we see the flag, and never polled again.
- A registration or Mode S hex in ``var/aviation_optout.txt`` is skipped everywhere: never linked,
  never polled, never written.
- What is stored is a flight record from airport to airport with times. No track, no callsign, no
  inference about who was on board. The poll state keeps only each aircraft's last observation,
  needed to tell a landing, and live position responses are never cached on disk.
- Registrations are re-checked against the registry at least weekly (``LINK_MAX_AGE_S``) before
  polling, so an aircraft the company sold stops being polled.

Sources (credit them): FAA Aircraft Registry (public domain), adsb.lol (ODbL), OurAirports
(public domain).
"""
from __future__ import annotations

import csv
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
ADSB_INTERVAL = 1.0  # adsb.lol: at most one request per second
FAA_SOURCE = {"name": "FAA Aircraft Registry", "url": REGISTRY_URL}
ADSB_SOURCE = {"name": "adsb.lol (ODbL)", "url": "https://api.adsb.lol/"}

CORPORATE = {"3": "corporation", "7": "LLC", "8": "non-citizen corporation"}  # FAA TYPE REGISTRANT
VALID_STATUS = {"V", "T"}  # FAA STATUS CODE: valid registration (T: valid, from a trainee)
SUFFIXES = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LLC", "LTD", "LIMITED", "PLC",
            "HOLDINGS"}
AIRPORT_TYPES = ("large_airport", "medium_airport", "small_airport")
AIRPORT_KM = 5.0
LOW_FT = 3000            # last seen below this, then gone for GONE_S: landed
GONE_S = 20 * 60
LOST_S = 12 * 3600       # airborne and unseen this long: we no longer know where it is
LINK_MAX_AGE_S = 7 * 86400
LINK_HARD_AGE_S = 30 * 86400  # a link this old that cannot be refreshed is not polled
DB_PIA, DB_LADD = 4, 8   # readsb dbFlags bits

_last_live = 0.0
_registry_cache: dict = {}
_airports_cache: dict = {}


# ---------- names ----------
def normalize_name(name: str) -> str:
    """Comparable form of a company name, for an exact match and nothing looser.

    Upper case; SEC's state-of-incorporation tail ("/DE", "/NEW/") dropped; "&" read as AND; other
    punctuation splits words; a leading THE and trailing legal-form words (INC, CORP, CO, LLC, LTD,
    HOLDINGS, ...) dropped; words joined without spaces, so "AMAZON.COM" equals "AMAZON COM".
    """
    s = re.sub(r"\s*/[A-Z.]*/?\s*$", "", str(name).upper().strip())
    s = s.replace("&", " AND ")
    words = [w for w in re.split(r"[^A-Z0-9]+", s.replace("'", "")) if w]
    if words[:1] == ["THE"] and len(words) > 1:
        words = words[1:]
    while len(words) > 1 and (words[-1] in SUFFIXES or words[-1] == "AND"):
        words.pop()
    return "".join(words)


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


def add_optout(token: str) -> None:
    p = optout_path()
    with p.open("a") as fh:
        fh.write(token.strip().upper() + "\n")


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
    """Aircraft registered to the company itself: corporate/LLC registrant, normalized owner name equal
    to the normalized company name, not opted out. ``index`` defaults to the current FAA registry."""
    if index is None:
        index, _ = registry()
    out = optout()
    key = normalize_name(name)
    hits = [dict(a) for a in index.get(key, []) if not opted_out(a["reg"], a["hex"], out)]
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
    links = _read_json(_links_path(), {})
    links[t] = {"name": name, "linked_at": now_iso(), "registry_fetched_at": fetched_at,
                "aircraft": [{"reg": a["reg"], "hex": a["hex"]} for a in planes]}
    _write_json(_links_path(), links)
    return len(planes)


def registered(view: View, ticker: str) -> list[Fact]:
    """The company's aircraft as of the latest registry download visible in ``view`` (one fact per tail).
    Do not use ``View.periods`` for this attribute: tails linked the same day share a period key."""
    fs = view.facts(ticker, "aircraft.registered")
    if not fs:
        return []
    last = max(f.as_of for f in fs)
    best: dict[str, Fact] = {}
    for f in fs:
        if f.as_of == last and (f.value not in best or f.known_at >= best[f.value].known_at):
            best[f.value] = f
    return sorted(best.values(), key=lambda f: f.value)


def flights(view: View, ticker: str) -> list[Fact]:
    """Recorded flights (``aircraft.flight``), newest arrival first."""
    fs = view.facts(ticker, "aircraft.flight")
    return sorted(fs, key=lambda f: (f.context.get("arrived") or f.as_of, f.known_at), reverse=True)


# ---------- airports ----------
def parse_airports(body: bytes) -> dict:
    text = io.StringIO(body.decode("utf-8-sig", errors="replace"))
    ident, name, lat, lon = [], [], [], []
    for r in csv.DictReader(text):
        if r.get("type") not in AIRPORT_TYPES:
            continue
        try:
            la, lo = float(r["latitude_deg"]), float(r["longitude_deg"])
        except (TypeError, ValueError):
            continue
        ident.append(r.get("ident") or r.get("gps_code") or "")
        name.append(r.get("name") or "")
        lat.append(la)
        lon.append(lo)
    return {"ident": ident, "name": name, "lat": np.radians(lat), "lon": np.radians(lon)}


def airports() -> dict:
    r = fetch(AIRPORTS_URL, ttl=AIRPORTS_TTL)
    if _airports_cache.get("fetched_at") != r.fetched_at:
        _airports_cache.clear()
        _airports_cache.update(table=parse_airports(r.body), fetched_at=r.fetched_at)
    return _airports_cache["table"]


def nearest_airport(table: dict, lat: float, lon: float, km: float = AIRPORT_KM) -> dict | None:
    """The closest large, medium or small airport within ``km``, else None."""
    if lat is None or lon is None or not len(table["lat"]):
        return None
    la, lo = np.radians(lat), np.radians(lon)
    h = np.sin((table["lat"] - la) / 2) ** 2 + np.cos(la) * np.cos(table["lat"]) * np.sin((table["lon"] - lo) / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(h, 1.0)))
    i = int(np.argmin(d))
    if d[i] > km:
        return None
    return {"ident": table["ident"][i], "name": table["name"][i], "km": float(d[i])}


# ---------- positions ----------
def _live_json(url: str) -> dict:
    """GET JSON straight into memory: live positions are never written to the HTTP cache."""
    global _last_live
    wait = _last_live + ADSB_INTERVAL - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_live = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": config.WEB_USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise FetchError(f"{url}: HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
        raise FetchError(f"{url}: {e}") from e


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def observe(ac: dict, now: float) -> dict:
    """One adsb.lol aircraft record -> {"t", "ground", "alt", "lat", "lon", "flags"}. ``ground`` is None
    when the record gives no altitude."""
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
    return {"t": now - float(seen), "ground": ground, "alt": alt, "lat": lat, "lon": lon,
            "flags": int(ac.get("dbFlags") or 0)}


def step(st: dict | None, obs: dict | None, now: float, table: dict) -> tuple[dict, dict | None]:
    """Advance one aircraft's state by one poll. Returns (new state, event); an event is
    {"kind": "takeoff", ...} or {"kind": "landing", "from", "from_name", "to", "to_name", "departed",
    "arrived"}. A landing needs an airport within AIRPORT_KM; otherwise nothing is recorded."""
    st = dict(st or {"phase": "unknown"})
    if obs is None:
        if st["phase"] == "air" and "t" in st:
            gone = now - st["t"]
            if gone > GONE_S and st.get("alt") is not None and st["alt"] < LOW_FT:
                apt = nearest_airport(table, st.get("lat"), st.get("lon"))
                if apt is None:
                    return {"phase": "unknown", "t": st["t"]}, None
                ev = _landing(st, apt, st["t"])
                return _on_ground(st["t"], apt, st.get("lat"), st.get("lon")), ev
            if gone > LOST_S:
                return {"phase": "unknown", "t": st["t"]}, None
        return st, None
    if obs["ground"] is None:
        return st, None
    if obs["ground"]:
        apt = nearest_airport(table, obs["lat"], obs["lon"])
        ev = _landing(st, apt, obs["t"]) if st["phase"] == "air" and apt else None
        return _on_ground(obs["t"], apt, obs["lat"], obs["lon"]), ev
    ev = None
    if st["phase"] == "ground":
        st = {"phase": "air", "from": st.get("airport"), "from_name": st.get("airport_name"),
              "departed": _iso(obs["t"])}
        ev = {"kind": "takeoff", "from": st["from"], "departed": st["departed"]}
    elif st["phase"] != "air":
        st = {"phase": "air", "from": None, "from_name": None, "departed": None}
    st.update(t=obs["t"], alt=obs["alt"], lat=obs["lat"], lon=obs["lon"])
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


def _current_links(bank: Bank, tickers: list[str], log) -> dict[str, list[dict]]:
    links = _read_json(_links_path(), {})
    stale = [t for t in tickers if t not in links or time.time() - _epoch(links[t]["linked_at"]) > LINK_MAX_AGE_S]
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
        if time.time() - _epoch(lk["linked_at"]) > LINK_HARD_AGE_S:
            log(f"  {t}: registrations not re-checked in 30 days; not polling its aircraft")
            continue
        out[t] = lk["aircraft"]
    return out


def poll(bank: Bank, state_path: Path | str | None = None, tickers: list[str] | None = None,
         get=None, table: dict | None = None, now: float | None = None, log=print) -> int:
    """Read the current position of every registered aircraft of the watched companies once, and write
    an ``aircraft.flight`` fact for each landing detected. Returns facts written."""
    sp = Path(state_path) if state_path else default_state_path()
    tickers = [t.upper() for t in (tickers or watched())]
    if not tickers:
        log("  no companies to poll: pass tickers, or add one with: python -m brain watch add aircraft TICKER")
        return 0
    links = _current_links(bank, tickers, log)
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
    for hex_ in [h for h in state["aircraft"] if h not in planes]:
        state["aircraft"].pop(hex_)  # no longer a watched company's aircraft: forget its last position
    _write_json(sp, state)
    return bank.add_many(facts)


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
        for f in registered(bank.view(), t) if n else []:
            c = f.context
            print(f"  {f.value:<8} hex {c['hex']}  {c['model'] or '?'}  {c['year'] or ''}  owner {c['owner']}")
    return 0


def _cmd_poll(a) -> int:
    n = poll(Bank(), tickers=a.tickers or None)
    print(f"{n} flight{'s' if n != 1 else ''} recorded")
    return 0


def _cmd_optout(a) -> int:
    add_optout(a.token)
    print(f"{a.token.upper()} added to {optout_path()}; it will never be linked or polled")
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
