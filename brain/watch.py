"""Watchlist and alerts: what to keep an eye on, and the rules that say when to look.

The watchlist (``<home>/watchlist.json``) names tickers, companies whose own aircraft to follow,
port areas and official-camera sites. ``refresh`` collects what is new about them; ``check`` runs
the rules over facts that became known since the last check and writes each alert back to the
bank as a derived fact that cites the facts behind it. An alert is not evidence that anything
will move a price; it says where to look.

Every alert carries rule keys: the rule and what it was about (a filing's accession number, a
price bar's date, a sample's id). A key that has alerted once never alerts again, however often
the same facts are read, and because the keys live in the bank that holds across processes. So
``check`` can re-read a window generously: a fact counts as new when it became known after the
previous check minus ``GRACE_DAYS``, because sources publish late (an EDGAR list cached for an
hour shows filings accepted before the last check). Rules read through a view at the check's
time, so an alert is never known before its inputs. Acknowledged alert ids and the time of the
last check live in ``<home>/alerts_state.json``.

The alternative-data lines in CLAUDE.md hold here too: aircraft alerts say airport to airport and
nothing about who flew; a registration or hex in ``aviation_optout.txt`` never alerts; sites take
only cameras of the official open providers.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from brain import config
from brain.bank import Bank, Fact, View
from brain.timeutil import days_between, norm_known_at, now_iso

DEFAULT_RULES = {"price_move": 0.05, "insider_buy_usd": 100000, "earnings_car": 0.05, "zscore": 3.0,
                 "forms": ["8-K", "10-K", "10-Q", "SC 13D", "S-1"], "headlines": True}
KINDS = {"ticker": "tickers", "tickers": "tickers", "aircraft": "aircraft_of", "aircraft_of": "aircraft_of",
         "area": "areas", "areas": "areas", "site": "sites", "sites": "sites"}
SEVERITY = {"info": 0, "notice": 1, "high": 2}
FIRST_LOOKBACK_DAYS = 3   # the first check ever looks this far back
GRACE_DAYS = 3            # later checks re-read this far before the previous one (late publishers)
REACTION_DAYS = 10        # an earnings release's 3-day reaction is looked for this long after it
PRICE_BARS = 5            # at most this many new daily bars are checked per snapshot
Z_WINDOW, Z_MIN = 30, 10  # z-score against the previous 30 comparable samples; at least 10 needed
AREA_SECONDS = 300
FILINGS_TTL = 900         # the recent-filings list is re-read at most every 15 minutes
MIN_EVERY_S = 60
NTFY_TIMEOUT = 10
NTFY_MAX_BYTES = 4000     # ntfy turns longer bodies into attachments
OFFICIAL_CAMERAS = ("TFL", "CALTRANS")  # used only if brain.sources.cameras cannot be imported
ITEMS = {"1.01": "material agreement", "1.02": "agreement ended", "1.03": "bankruptcy",
         "2.01": "acquisition or disposal", "2.02": "results", "2.03": "new debt", "2.05": "restructuring",
         "2.06": "impairment", "3.01": "delisting notice", "4.01": "auditor change",
         "4.02": "past financials not to be relied on", "5.01": "change in control",
         "5.02": "officer or director change", "5.07": "shareholder vote", "7.01": "Reg FD", "8.01": "other events",
         "9.01": "exhibits"}
HIGH_ITEMS = {"1.03", "3.01", "4.02", "5.01"}


# ---------- files ----------
def watchlist_path() -> Path:
    return config.home() / "watchlist.json"


def state_path() -> Path:
    return config.home() / "alerts_state.json"


@contextlib.contextmanager
def _locked(name: str):
    """An exclusive lock shared by every process using this home (the terminal, a ``watch run``)."""
    with (config.home() / name).open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _write_json(p: Path, obj) -> None:
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, p)


# ---------- the watchlist ----------
def _norm_area(a) -> dict:
    a = {"name": a} if isinstance(a, str) else dict(a)
    a["name"] = str(a.get("name", "")).strip().upper()
    return a


def _norm_site(s: dict) -> dict:
    s = dict(s)
    s["name"] = str(s.get("name", "")).strip()
    if s.get("ticker"):
        s["ticker"] = str(s["ticker"]).strip().upper()
    s["cameras"] = list(dict.fromkeys(str(c).strip() for c in s.get("cameras") or [] if str(c).strip()))
    return s


def _normalize(raw: dict) -> dict:
    w = dict(raw)  # keys this module does not know are kept
    for k in ("tickers", "aircraft_of"):
        w[k] = list(dict.fromkeys(str(t).strip().upper() for t in raw.get(k) or [] if str(t).strip()))
    w["areas"] = [a for a in (_norm_area(x) for x in raw.get("areas") or []) if a["name"]]
    w["sites"] = [s for s in (_norm_site(x) for x in raw.get("sites") or [] if isinstance(x, dict)) if s["name"]]
    w["rules"] = {**DEFAULT_RULES, **(raw.get("rules") or {})}
    return w


def load() -> dict:
    """The watchlist, with every section present and default rules filled in. A file that is not
    valid JSON is an error, not an empty watchlist: saving over it would lose what it held."""
    p = watchlist_path()
    try:
        raw = json.loads(p.read_text())
    except FileNotFoundError:
        raw = {}
    except ValueError as e:
        raise ValueError(f"{p} is not valid JSON ({e}); fix or remove it") from e
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must hold a JSON object")
    return _normalize(raw)


def save(w: dict) -> None:
    _write_json(watchlist_path(), _normalize(w))


def _kind(kind: str) -> str:
    try:
        return KINDS[kind.lower()]
    except KeyError:
        raise ValueError(f"unknown kind {kind!r}; use ticker, aircraft, area or site") from None


def _symbol(value: str) -> str:
    v = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=\-]{0,19}", v):
        raise ValueError(f"not a ticker: {value!r}")
    return v


def _bbox(b) -> list[float]:
    try:
        la0, lo0, la1, lo1 = (float(x) for x in b)
    except (TypeError, ValueError):
        raise ValueError(f"bbox is [lat_min, lon_min, lat_max, lon_max], not {b!r}") from None
    if not (-90 <= la0 < la1 <= 90 and -180 <= lo0 < lo1 <= 180):
        raise ValueError(f"bbox {b!r}: need -90 <= lat_min < lat_max <= 90 and -180 <= lon_min < lon_max <= 180")
    return [la0, lo0, la1, lo1]


def _builtin_areas() -> set[str] | None:
    try:
        from brain.sources import shipping
    except Exception:  # not there (or mid-edit): an area cannot be checked against the list
        return None
    return {a["name"] for a in shipping.AREAS}


def _area(name: str, kw: dict) -> dict:
    a = {"name": str(name).strip().upper()}
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{0,39}", a["name"]):
        raise ValueError(f"area names are letters, digits, '-' and '_': {name!r}")
    if kw.get("bbox") is not None:
        a["bbox"] = _bbox(kw["bbox"])
    else:
        known = _builtin_areas()
        if known is not None and a["name"] not in known:
            raise ValueError(f"{a['name']} is not a built-in port area ({', '.join(sorted(known))}); give its bbox")
    if kw.get("seconds") is not None:
        s = int(kw["seconds"])
        if not 10 <= s <= 3600:
            raise ValueError("an area's sampling window is 10-3600 seconds")
        a["seconds"] = s
    return a


def _camera_providers() -> set[str]:
    try:
        from brain.sources.cameras import PROVIDERS
        return set(PROVIDERS)
    except Exception:
        return set(OFFICIAL_CAMERAS)


def _camera_id(cid: str, providers: set[str]) -> str:
    cid = str(cid).strip()
    if cid.split(".", 1)[0].upper() not in providers or not re.fullmatch(r"[A-Za-z]+\.[A-Za-z0-9._-]+", cid):
        raise ValueError(f"{cid!r}: sites take only official open traffic cameras ({', '.join(sorted(providers))}), "
                         f"by the id their listing gives (python -m brain cams near LAT LON)")
    return cid


def _site(name: str, kw: dict, old: dict | None) -> dict:
    s = dict(old or {"name": str(name).strip()})
    s["cameras"] = list(s.get("cameras") or [])
    if not s["name"]:
        raise ValueError("a site needs a name")
    if kw.get("ticker"):
        s["ticker"] = _symbol(kw["ticker"])
    providers = _camera_providers()
    have = {c.upper() for c in s["cameras"]}
    for c in kw.get("cameras") or []:
        c = _camera_id(c, providers)
        if c.upper() not in have:
            s["cameras"].append(c)
            have.add(c.upper())
    for k in ("lat", "lon"):
        if kw.get(k) is not None:
            s[k] = float(kw[k])
    if not s["cameras"]:
        raise ValueError(f"site {s['name']!r} needs at least one camera")
    return s


def add(kind: str, value: str, **kw) -> dict:
    """Add a ticker, a company whose aircraft to follow, an area (``bbox=``, ``seconds=``) or a site
    (``ticker=``, ``cameras=[ids]``, ``lat=``, ``lon=``). Adding a site that exists adds its cameras.
    Returns the saved watchlist."""
    key = _kind(kind)
    with _locked("watchlist.json.lock"):
        w = load()
        if key in ("tickers", "aircraft_of"):
            v = _symbol(value)
            if v not in w[key]:
                w[key].append(v)
        elif key == "areas":
            a = _area(value, kw)
            w["areas"] = [x for x in w["areas"] if x["name"] != a["name"]] + [a]
        else:
            old = next((s for s in w["sites"] if s["name"].casefold() == str(value).strip().casefold()), None)
            s = _site(value, kw, old)
            w["sites"] = [x for x in w["sites"] if x is not old] + [s]
        save(w)
    return load()


def remove(kind: str, value: str) -> dict:
    """Take an entry off the watchlist (KeyError if it is not there). Alerts already written stay."""
    key = _kind(kind)
    with _locked("watchlist.json.lock"):
        w = load()
        if key in ("tickers", "aircraft_of"):
            v = str(value).strip().upper()
            keep = [x for x in w[key] if x != v]
        elif key == "areas":
            v = str(value).strip().upper()
            keep = [x for x in w["areas"] if x["name"] != v]
        else:
            v = str(value).strip()
            keep = [x for x in w["sites"] if x["name"].casefold() != v.casefold()]
        if len(keep) == len(w[key]):
            raise KeyError(f"{v} is not in the watchlist's {key}")
        w[key] = keep
        save(w)
    return load()


def watched_entities(w: dict) -> list[str]:
    """Every bank entity the watchlist's rules look at."""
    out = w["tickers"] + w["aircraft_of"] + [f"AREA.{a['name']}" for a in w["areas"]]
    out += [f"CAM.{c}".upper() for s in w["sites"] for c in s["cameras"]]
    return list(dict.fromkeys(out))


# ---------- alert state: last check, acknowledged ids ----------
def _read_state() -> dict:
    p = state_path()
    try:
        s = json.loads(p.read_text())
    except FileNotFoundError:
        s = {}
    except ValueError as e:
        raise ValueError(f"{p} is not valid JSON ({e}); fix or remove it") from e
    return {**s, "last_check": s.get("last_check"), "acked": list(s.get("acked") or []),
            "entities": list(s.get("entities") or [])}


@contextlib.contextmanager
def _state():
    """Read-modify-write of the state file under its lock."""
    with _locked("alerts_state.json.lock"):
        s = _read_state()
        yield s
        s["acked"] = sorted(set(s["acked"]))
        s["entities"] = sorted(set(s["entities"]))
        _write_json(state_path(), s)


def acked() -> set[str]:
    return set(_read_state()["acked"])


def alerts(bank: Bank, unacked_only: bool = False, limit: int | None = None,
           entities: list[str] | None = None) -> list[Fact]:
    """Alerts newest first: of ``entities``, or of everything watched plus anything that ever alerted."""
    if entities is None:
        try:
            w = load()
        except ValueError:
            w = _normalize({})
        entities = list(dict.fromkeys(watched_entities(w) + _read_state()["entities"]))
    done = acked() if unacked_only else set()
    out = [f for e in dict.fromkeys(x.upper() for x in entities) for f in bank.facts(e, "alert") if f.id not in done]
    out.sort(key=lambda f: (f.known_at, f.id), reverse=True)
    return out[:limit] if limit else out


def unacked(bank: Bank) -> list[Fact]:
    return alerts(bank, unacked_only=True)


def ack(bank: Bank, ident: str) -> Fact:
    """Acknowledge one alert by id, id prefix or ENTITY/id ref. KeyError if none matches, ValueError if
    several do."""
    pre = str(ident).strip().rpartition("/")[2].lower()
    if not re.fullmatch(r"[0-9a-f]{4,20}", pre):
        raise KeyError(f"{ident!r} is not an alert id (at least 4 hex characters)")
    hits = [f for f in alerts(bank) if f.id.startswith(pre)]
    if not hits:
        raise KeyError(f"no alert with id {pre}")
    if len({f.id for f in hits}) > 1:
        raise ValueError(f"{pre} matches {len(hits)} alerts; give more of the id")
    with _state() as s:
        s["acked"].append(hits[0].id)
    return hits[0]


def ack_all(bank: Bank) -> int:
    ids = [f.id for f in unacked(bank)]
    if ids:
        with _state() as s:
            s["acked"] += ids
    return len(ids)


# ---------- rules ----------
def _shift(iso: str, days: float) -> str:
    t = datetime.fromisoformat(norm_known_at(iso).replace("Z", "+00:00")) + timedelta(days=days)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _sev(x: float, thr: float) -> str:
    return "high" if x >= 2 * thr else "notice"


def _on(v) -> bool:
    return v is not None and v is not False


class _Check:
    """One evaluation: the view at the check's time, the window of new facts, the keys in use."""

    def __init__(self, bank: Bank, now: str, lower: str, rules: dict):
        self.bank, self.now, self.lower, self.rules = bank, now, lower, rules
        self.view: View = bank.view(now)
        self.out: list[Fact] = []
        self._used: dict[str, set[str]] = {}
        self._books: dict[str, object] = {}

    def new(self, entity: str, attr: str, where=None) -> list[Fact]:
        """Facts that became known inside the window, oldest first."""
        fs = [f for f in self.view.facts(entity, attr, where) if f.known_at > self.lower]
        return sorted(fs, key=lambda f: (f.known_at, f.id))

    def used(self, entity: str) -> set[str]:
        e = entity.upper()
        if e not in self._used:
            keys = set()
            for f in self.bank.facts(e, "alert", include_downgraded=True):  # a withdrawn alert is not re-raised
                keys.update(f.context.get("keys") or [])
            self._used[e] = keys
        return self._used[e]

    def emit(self, entity: str, rule: str, severity: str, message: str, as_of: str, inputs: list[Fact],
             keys: list[str], **extra) -> Fact | None:
        used = self.used(entity)
        if any(k in used for k in keys):
            return None
        f = Fact(entity=entity, attr="alert", value=message, as_of=as_of, known_at=self.now, tier="derived",
                 source={"name": "brain.watch", "rule": rule},
                 context={"rule": rule, "severity": severity, "keys": list(keys), **extra},
                 inputs=tuple(dict.fromkeys(g.ref for g in inputs)), retrieved_at=self.now)
        used.update(keys)
        self.out.append(f)
        return f

    def book(self, ticker: str):
        if ticker not in self._books:
            from brain.metrics import PriceBook
            self._books[ticker] = PriceBook(self.bank, [ticker, "SPY"], asof=self.now)
        return self._books[ticker]


def _price_move(c: _Check, t: str) -> None:
    """|1-day return| >= rules.price_move on a daily bar of the latest snapshot. A bar is new when its
    date, read as the end of that day (the bank's rule for a close), falls inside the window."""
    thr = c.rules.get("price_move")
    cur = c.view.latest(t, "series.price_daily")
    if not _on(thr) or cur is None or cur.known_at <= c.lower:
        return
    from brain import series  # pandas: imported only when a rule needs it, not on every CLI start
    df = series.load(cur.value)
    dates, close, adj = df["date"].to_numpy(dtype=str), df["close"].to_numpy(float), df["adjclose"].to_numpy(float)
    n = len(dates)
    for i in range(max(1, n - PRICE_BARS), n):
        if not c.lower < norm_known_at(dates[i]) <= c.now or not adj[i - 1] > 0:
            continue
        r = float(adj[i] / adj[i - 1] - 1)
        if abs(r) >= thr:
            c.emit(t, "price_move", _sev(abs(r), thr),
                   f"{t} {r:+.1%} on {dates[i]} (close {close[i]:,.2f}, previous session {dates[i - 1]})",
                   str(dates[i]), [cur], [f"price:{t}:{dates[i]}"], move=round(r, 6))


def _form_key(form: str) -> str:
    f = re.sub(r"\s+", " ", str(form).strip().upper())
    f = re.sub(r"^SCHEDULE ", "SC ", f)  # EDGAR's structured 13D/13G forms are named SCHEDULE 13D since 2024
    return f[:-2] if f.endswith("/A") else f


def _items(f: Fact) -> list[str]:
    return [x.strip() for x in str(f.context.get("items") or "").split(",") if x.strip()]


def _is_earnings(f: Fact) -> bool:
    return str(f.value).upper() in ("8-K", "8-K/A") and "2.02" in _items(f)


def _filing(c: _Check, t: str) -> None:
    """A filing of a watched form (amendments too). Earnings releases go to the earnings rule instead."""
    want = {_form_key(x) for x in c.rules.get("forms") or []}
    if not want:
        return
    for f in c.new(t, "filing", lambda f: _form_key(f.value) in want):
        if _is_earnings(f):
            continue
        items = _items(f)
        what = "; ".join(f"{x} {ITEMS[x]}" if x in ITEMS else x for x in items)
        sev = "high" if HIGH_ITEMS & set(items) else "notice"
        c.emit(t, "filing", sev, f"{t} filed {f.value}{f' (items {what})' if what else ''} on {f.known_at[:10]}",
               f.as_of, [f], [f"filing:{f.context.get('accn') or f.id}"], form=str(f.value))


def _officer_or_director(f: Fact) -> bool:
    return bool(set(str(f.context.get("relationship") or "").split(",")) & {"officer", "director"})


def _insider_buy(c: _Check, t: str) -> None:
    """Open-market purchases (code P) by officers or directors, summed per filing, >= rules.insider_buy_usd."""
    thr = c.rules.get("insider_buy_usd")
    if not _on(thr):
        return
    by_filing: dict[str, dict[tuple, Fact]] = {}
    for f in c.new(t, "insider.trade", lambda f: f.context.get("code") == "P" and _officer_or_director(f)):
        x = f.context
        key = (x.get("owner_cik"), f.as_of, f.value, x.get("price"), x.get("ownership"))
        by_filing.setdefault(x.get("accn") or f.id, {}).setdefault(key, f)  # a bulk and a live copy count once
    for accn, trades in by_filing.items():
        fs = list(trades.values())
        usd = sum(float(f.context.get("value_usd") or 0.0) for f in fs)
        if usd < thr:
            continue
        who = []
        for f in fs:
            x = f.context
            role = ", ".join(r for r in str(x.get("relationship") or "").split(",") if r)
            w = f"{x.get('owner') or 'unnamed'} ({role}{', ' + x['title'] if x.get('title') else ''})"
            if w not in who:
                who.append(w)
        days = sorted({f.as_of for f in fs})
        when = days[0] if len(days) == 1 else f"{days[0]} to {days[-1]}"
        shares = sum(float(f.value) for f in fs)
        c.emit(t, "insider_buy", _sev(usd, thr),
               f"{t} insider buying: {'; '.join(who)} bought {shares:,.0f} shares for ${usd:,.0f} on {when}, "
               f"filed {max(f.known_at for f in fs)[:10]}", days[-1], fs, [f"insider:{accn}"], value_usd=round(usd, 2))


def _earnings(c: _Check, t: str) -> None:
    """A new earnings release (notice), and its 3-day excess reaction vs SPY once complete (high when
    |car_3d| >= rules.earnings_car). A release first seen with its reaction complete alerts once."""
    from brain import events
    thr = c.rules.get("earnings_car")
    evs = [e for e in events.distinct(events.earnings_events(c.view, t))
           if e.known_at > c.lower or days_between(e.known_at, c.now) <= REACTION_DAYS]
    for e in evs:
        accn = e.context.get("accn") or e.id
        k_event, k_car = f"earnings:{accn}", f"earnings-car:{accn}"
        book = c.book(t)
        rx = events.reaction(book, c.view, t, e) if book.has(t) and book.has("SPY") else None
        car = rx["car_3d"] if rx else None
        inputs = [e] + ([book.snapshot[t], book.snapshot["SPY"]] if car is not None else [])
        big = car is not None and _on(thr) and abs(car) >= thr
        if e.known_at > c.lower:
            tail = f"; 3-day reaction {car:+.1%} vs SPY" if car is not None else ""
            c.emit(t, "earnings", "high" if big else "notice",
                   f"{t} earnings release (8-K item 2.02) filed {e.known_at[:10]}{tail}", e.as_of, inputs,
                   [k_event] + ([k_car] if car is not None else []), car_3d=car)
        if big:
            c.emit(t, "earnings", "high",
                   f"{t} {car:+.1%} vs SPY over the 3 days around its earnings release of {rx['date']}",
                   rx["date"], inputs, [k_car], car_3d=car)


def _headline(c: _Check, t: str) -> None:
    if not c.rules.get("headlines", True):
        return
    for f in c.new(t, "headline"):
        c.emit(t, "headline", "info", f"{t} headline: {f.value} ({f.source.get('name', '?')})", f.as_of, [f],
               [f"headline:{f.source.get('url') or f.id}"])


def _opted_out():
    try:
        from brain.sources.aviation import opted_out
        return opted_out
    except Exception:  # the aviation module unavailable: read the opt-out file the same way, more simply
        p = config.home() / "aviation_optout.txt"
        toks = {ln.split("#", 1)[0].strip().upper() for ln in p.read_text().splitlines()} if p.exists() else set()
        return lambda reg, hex_: bool({str(reg or "").upper(), str(reg or "").upper().lstrip("N"),
                                       str(hex_ or "").upper()} & (toks - {""}))


def _aircraft(c: _Check, t: str) -> None:
    """A company-registered aircraft landed: airport to airport, with times, nothing about who flew."""
    out = _opted_out()
    for f in c.new(t, "aircraft.flight"):
        x = f.context
        if out(x.get("reg"), x.get("hex")):
            continue
        to = f"{x.get('to') or f.value}" + (f" ({x['to_name']})" if x.get("to_name") else "")
        frm = f"{x.get('from') or 'unknown'}" + (f" ({x['from_name']})" if x.get("from_name") else "")
        c.emit(t, "aircraft", "notice",
               f"{t} aircraft {x.get('reg') or x.get('hex')} landed at {to} from {frm}; "
               f"departed {x.get('departed') or '?'}, arrived {x.get('arrived') or f.known_at}",
               f.as_of, [f], [f"aircraft:{x.get('hex') or f.id}:{x.get('arrived') or f.known_at}"])


def _zscore(c: _Check, entity: str, attr: str, rule: str, same, describe) -> None:
    """Each new sample vs the previous Z_WINDOW comparable ones (same window or detector): |z| >=
    rules.zscore alerts. Fewer than Z_MIN prior samples, or no spread among them, says nothing."""
    thr = c.rules.get("zscore")
    if not _on(thr):
        return
    import numpy as np
    facts = sorted(c.view.facts(entity, attr), key=lambda f: (f.known_at, f.id))
    for i, f in enumerate(facts):
        if f.known_at <= c.lower:
            continue
        prior = []
        for j in range(i - 1, -1, -1):  # walk back: a year of 15-minute samples is long
            if same(facts[j], f):
                prior.append(facts[j])
                if len(prior) == Z_WINDOW:
                    break
        prior.reverse()
        if len(prior) < Z_MIN:
            continue
        vals = np.array([float(g.value) for g in prior])
        sd = float(vals.std(ddof=1))
        if not sd > 0:
            continue
        z = (float(f.value) - float(vals.mean())) / sd
        if abs(z) >= thr:
            c.emit(entity, rule, _sev(abs(z), thr),
                   f"{describe(f)}, z {z:+.1f} vs the previous {len(prior)} samples (mean {vals.mean():.1f})",
                   f.as_of, [f, *prior], [f"{rule}:{f.id}"], z=round(z, 2))


def _ships(c: _Check, entity: str) -> None:
    def describe(f: Fact) -> str:
        x = f.context
        return (f"{entity}: {f.value} vessels in {x.get('window_s')} s (cargo {x.get('cargo')}, "
                f"tanker {x.get('tanker')}) at {f.known_at[:16]}Z")
    same = lambda g, f: (g.context.get("window_s"), g.context.get("bbox")) == (f.context.get("window_s"), f.context.get("bbox"))
    _zscore(c, entity, "ships.count", "ships", same, describe)


def _camera(c: _Check, entity: str, site: dict) -> None:
    label = site["name"] + (f", {site['ticker']}" if site.get("ticker") else "")
    describe = lambda f: f"{entity} ({label}): {f.value} vehicles at {f.known_at[:16]}Z"
    same = lambda g, f: (g.context.get("detector"), g.context.get("threshold")) == \
        (f.context.get("detector"), f.context.get("threshold"))
    _zscore(c, entity, "camera.vehicles", "camera", same, describe)


TICKER_RULES = (_price_move, _filing, _insider_buy, _earnings, _headline)


def check(bank: Bank, since: str | None = None, *, now: str | None = None, log=print) -> list[Fact]:
    """Run every rule over facts known after ``since`` (default: the last check minus ``GRACE_DAYS``;
    on the first check, ``FIRST_LOOKBACK_DAYS``), write the alerts, return them. A rule that fails on
    one entity is logged and the others carry on."""
    now = norm_known_at(now) if now else now_iso()
    w = load()
    with _locked("alerts.check.lock"):  # one check at a time, so two cannot raise the same alert
        last = _read_state()["last_check"]
        if since is not None:
            lower = norm_known_at(since)
        elif last:
            lower = _shift(last, -GRACE_DAYS)
        else:
            lower = _shift(now, -FIRST_LOOKBACK_DAYS)
        c = _Check(bank, now, lower, w["rules"])
        jobs = [(fn, t, ()) for t in w["tickers"] for fn in TICKER_RULES]
        jobs += [(_aircraft, t, ()) for t in w["aircraft_of"]]
        jobs += [(_ships, f"AREA.{a['name']}", ()) for a in w["areas"]]
        jobs += [(_camera, f"CAM.{cam}".upper(), (s,)) for s in w["sites"] for cam in s["cameras"]]
        for fn, entity, args in jobs:
            try:
                fn(c, entity, *args)
            except Exception as e:
                log(f"  alert rule {getattr(fn, '__name__', 'rule').lstrip('_')} on {entity} failed "
                    f"({type(e).__name__}: {e})")
        bank.add_many(c.out)
        with _state() as s:
            s["last_check"] = max(filter(None, [s["last_check"], now]))
            s["entities"] += [f.entity for f in c.out]
    return sorted(c.out, key=lambda f: (f.as_of, f.entity, f.id))


def line(f: Fact) -> str:
    """One alert as one line: when, severity, short id, message."""
    return f"{f.known_at[:16]}Z {f.context.get('severity', '?'):<6} [{f.id[:8]}] {f.value}"


# ---------- collecting ----------
def _src_filings(bank: Bank, t: str, log) -> int:
    """The whole filing history the first time, then only EDGAR's recent page, re-read every 15 minutes."""
    from brain.ingest import add_identity, ingest
    from brain.sources import sec
    if not bank.facts(t, "filing"):
        return ingest(bank, t, ("filings",), log=log).get("filings", 0)
    fil, ident = sec.filings_for(t, all_pages=False, ttl=FILINGS_TTL)
    add_identity(bank, ident)
    return bank.add_many(fil)


def _src_insiders(bank: Bank, t: str, log) -> int:
    from brain.sources import insiders
    return insiders.ingest_live(bank, t)


def _src_prices(bank: Bank, t: str, log) -> int:
    from brain.ingest import ingest
    return ingest(bank, t, ("prices",), log=log).get("prices", 0)


def _src_news(bank: Bank, t: str, log) -> int | None:
    from brain.ingest import ingest
    return ingest(bank, t, ("news",), log=log).get("headlines")


def _src_aircraft(bank: Bank, tickers: list[str], log) -> int:
    from brain.sources import aviation
    return aviation.poll(bank, tickers=tickers, log=log)


def _src_area(bank: Bank, area: dict, log) -> int:
    from brain.sources import shipping
    spec = {"name": area["name"], "bbox": area["bbox"]} if area.get("bbox") else area["name"]
    return shipping.sample(bank, spec, seconds=int(area.get("seconds") or AREA_SECONDS)).value


def _src_cameras(bank: Bank, ids: list[str], log) -> int:
    from brain.sources import cameras
    return cameras.sample(bank, ids, log=log)


def _unavailable(e: BaseException) -> bool:
    """The source cannot run here at all (missing module, package, key or model), as opposed to failing once."""
    return isinstance(e, ImportError) or type(e).__name__ == "Unavailable"


def refresh(bank: Bank, log=print) -> dict:
    """Collect what is new for everything watched. A source that fails, or cannot run here, costs one
    log line and the rest carry on."""
    w = load()
    out = {"tickers": {}, "benchmark": None, "aircraft": None, "areas": {}, "cameras": None, "skipped": [],
           "errors": []}

    def guard(label: str, fn, *args):
        try:
            return fn(bank, *args, log)
        except (Exception, SystemExit) as e:  # SystemExit: config.sec_user_agent() exits when it is unset
            msg = f"{label}: {type(e).__name__}: {e}"
            (out["skipped"] if _unavailable(e) else out["errors"]).append(msg)
            log(f"  {msg}")
            if _unavailable(e):
                raise _Skip from e
            return None

    def skip(msg: str) -> None:
        out["skipped"].append(msg)
        log(f"  {msg}")

    sec_ok = bool(os.environ.get("SEC_USER_AGENT", "").strip())
    if w["tickers"] and not sec_ok:
        skip("filings and insider trades skipped: SEC needs export SEC_USER_AGENT='Your Name your@email.com'")
    sources = [("filings", _src_filings), ("insiders", _src_insiders)] if sec_ok else []
    sources += [("prices", _src_prices), ("headlines", _src_news)]
    dead: set[str] = set()
    for t in w["tickers"]:
        counts = out["tickers"][t] = {}
        for name, fn in sources:
            if name in dead:
                continue
            try:
                counts[name] = guard(f"{t} {name}", fn, t)
            except _Skip:
                dead.add(name)  # it will not work for the next ticker either
    if w["tickers"] and "SPY" not in w["tickers"] and "prices" not in dead:
        with contextlib.suppress(_Skip):
            out["benchmark"] = guard("SPY prices (benchmark)", _src_prices, "SPY")
    if w["aircraft_of"]:
        with contextlib.suppress(_Skip):
            out["aircraft"] = guard("aircraft", _src_aircraft, w["aircraft_of"])
    if w["areas"]:
        if not os.environ.get("AISSTREAM_API_KEY", "").strip():
            skip("port areas skipped: AIS needs a free aisstream.io key (export AISSTREAM_API_KEY=...)")
        else:
            for a in w["areas"]:
                try:
                    out["areas"][a["name"]] = guard(f"area {a['name']}", _src_area, a)
                except _Skip:
                    break
    ids = list(dict.fromkeys(c for s in w["sites"] for c in s["cameras"]))
    if ids:
        with contextlib.suppress(_Skip):
            out["cameras"] = guard("cameras", _src_cameras, ids)
    return out


def summary(out: dict) -> str:
    """refresh's result as one line: what each source added ("-" when it did not run or failed)."""
    def n(v) -> str:
        return "-" if v is None else f"+{v}"
    parts = [f"{t} " + " ".join(f"{k} {n(v)}" for k, v in c.items()) for t, c in (out.get("tickers") or {}).items()]
    if out.get("aircraft") is not None:
        parts.append(f"flights {n(out['aircraft'])}")
    parts += [f"{a} vessels {v}" for a, v in (out.get("areas") or {}).items() if v is not None]
    if out.get("cameras") is not None:
        parts.append(f"camera samples {n(out['cameras'])}")
    bad = len(out.get("errors") or []) + len(out.get("skipped") or [])
    return ("; ".join(parts) or "nothing watched") + (f" ({bad} skipped or failed, above)" if bad else "")


class _Skip(Exception):
    """Raised by refresh's guard when a source cannot run here, so its remaining calls are not tried."""


# ---------- pushing and looping ----------
def notify(alerts_: list[Fact], *, min_severity: str = "notice", log=None) -> None:
    """Push alerts at or above ``min_severity`` to ntfy (BRAIN_NTFY_URL, e.g. https://ntfy.sh/<topic>) as one
    plain-text POST. Does nothing without the variable; never raises."""
    url = os.environ.get("BRAIN_NTFY_URL", "").strip()
    try:
        keep = [f for f in alerts_ or [] if SEVERITY.get(f.context.get("severity"), 0) >= SEVERITY[min_severity]]
        if not url or not keep:
            return
        if not url.startswith(("https://", "http://")):
            raise ValueError("BRAIN_NTFY_URL must be an http(s) URL")
        top = max(keep, key=lambda f: SEVERITY.get(f.context.get("severity"), 0)).context.get("severity")
        body = "\n".join(line(f) for f in keep).encode()
        if len(body) > NTFY_MAX_BYTES:
            body = body[:NTFY_MAX_BYTES - 40].rsplit(b"\n", 1)[0] + f"\n... ({len(keep)} alerts in all)".encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Title": f"money-brain: {len(keep)} alert{'s' if len(keep) != 1 else ''}",
            "Priority": {"high": "high", "notice": "default"}.get(top, "low"),
            "Content-Type": "text/plain; charset=utf-8", "User-Agent": config.WEB_USER_AGENT})
        with urllib.request.urlopen(req, timeout=NTFY_TIMEOUT) as r:
            r.read()
    except Exception as e:
        if log:
            log(f"  ntfy push failed ({type(e).__name__}: {e})")


def run(every_s: int = 900, log=print, *, cycles: int | None = None, sleep=time.sleep) -> None:
    """refresh + check every ``every_s`` seconds (at least ``MIN_EVERY_S``), one line per alert, until
    Ctrl+C. A failed refresh or check is logged and the loop goes on."""
    every_s = max(MIN_EVERY_S, int(every_s))
    n = 0
    try:
        while True:
            start = time.monotonic()
            bank = Bank()  # a fresh one each cycle: memory stays bounded, other processes' writes are read
            try:
                got = refresh(bank, log=log)
                log(f"{now_iso()[:16]}Z refreshed: {summary(got)}")
            except Exception as e:
                log(f"refresh failed ({type(e).__name__}: {e})")
            try:
                new = check(bank, log=log)
                log(f"{now_iso()[:16]}Z checked: {len(new)} new alert{'s' if len(new) != 1 else ''}")
                for f in new:
                    log(line(f))
                notify(new, log=log)
            except Exception as e:
                log(f"check failed ({type(e).__name__}: {e})")
            n += 1
            if cycles is not None and n >= cycles:
                return
            sleep(max(0.0, every_s - (time.monotonic() - start)))
    except KeyboardInterrupt:
        log("stopped")


# ---------- CLI ----------
def parse_every(text: str) -> int:
    """'900', '90s', '15m', '1h30m', '1d' -> seconds (at least MIN_EVERY_S)."""
    s = str(text).strip().lower()
    if re.fullmatch(r"\d+", s):
        n = int(s)
    else:
        parts = re.findall(r"(\d+)([smhd])", s)
        if not parts or "".join(a + b for a, b in parts) != s:
            raise argparse.ArgumentTypeError(f"an interval like 900, 90s, 15m or 1h, not {text!r}")
        n = sum(int(a) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[b] for a, b in parts)
    if n < MIN_EVERY_S:
        raise argparse.ArgumentTypeError(f"at least {MIN_EVERY_S} seconds between rounds")
    return n


def _print_watchlist(w: dict) -> None:
    print(f"tickers      {' '.join(w['tickers']) or '-'}")
    print(f"aircraft_of  {' '.join(w['aircraft_of']) or '-'}")
    areas = [a["name"] + (f" {a['bbox']}" if a.get("bbox") else "") + (f" {a['seconds']} s" if a.get("seconds") else "")
             for a in w["areas"]]
    print(f"areas        {'; '.join(areas) or '-'}")
    if not w["sites"]:
        print("sites        -")
    for i, s in enumerate(w["sites"]):
        tk = f" ({s['ticker']})" if s.get("ticker") else ""
        print(f"{'sites' if i == 0 else '':<12} {s['name']}{tk}: {', '.join(s['cameras'])}")
    r = w["rules"]
    shown = [f"{k} {','.join(v) if isinstance(v, list) else 'on' if v is True else 'off' if v in (None, False) else v}"
             for k, v in r.items()]
    print(f"rules        {'  '.join(shown)}")


def _split_kind(values: list[str]) -> tuple[str, list[str]]:
    """``[KIND] VALUE...``: the kind word may be left out for tickers (``watch add AAPL MSFT``). A lone
    kind word is a ticker (SITE is SiteOne Landscape Supply)."""
    if len(values) > 1 and values[0].lower() in KINDS:
        return _kind(values[0]), values[1:]
    return "tickers", values


def _cmd_add(a) -> int:
    kind, vals = _split_kind(a.values)
    try:
        if kind in ("tickers", "aircraft_of"):
            for v in vals:
                add(kind, v)
        elif kind == "areas":
            if len(vals) != 1:
                raise ValueError("one area at a time")
            add(kind, vals[0], bbox=a.bbox, seconds=a.seconds)
        else:
            name, cams = " ".join(vals), list(a.cameras or [])
            if a.near:
                from brain.sources import cameras
                hits = cameras.near(a.near[0], a.near[1], a.km)[:a.max]
                if not hits:
                    raise ValueError(f"no official camera within {a.km} km of {a.near[0]}, {a.near[1]}")
                for h in hits:
                    print(f"  {h['km']:.2f} km  {h['id']}  {h['name']}")
                cams += [h["id"] for h in hits]
            add(kind, name, ticker=a.ticker, cameras=cams, lat=a.near[0] if a.near else None,
                lon=a.near[1] if a.near else None)
    except (ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 1
    except Exception as e:  # e.g. a camera listing that could not be fetched
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    _print_watchlist(load())
    return 0


def _cmd_remove(a) -> int:
    kind, vals = _split_kind(a.values)
    vals = [" ".join(vals)] if kind == "sites" else vals
    status = 0
    for v in vals:
        try:
            remove(kind, v)
        except KeyError as e:
            print(e.args[0], file=sys.stderr)
            status = 1
    _print_watchlist(load())
    return status


def _cmd_list(a) -> int:
    w = load()
    if a.json:
        print(json.dumps(w, indent=1))
    else:
        _print_watchlist(w)
    return 0


def _rule_value(name: str, text: str):
    t = text.strip()
    if t.lower() in ("off", "none", "null", ""):
        return None
    if name == "forms":
        return [x.strip() for x in t.split(",") if x.strip()]
    if name == "headlines":
        if t.lower() not in ("on", "off", "true", "false", "yes", "no"):
            raise ValueError("headlines is on or off")
        return t.lower() in ("on", "true", "yes")
    v = float(t)
    if v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


def _cmd_rule(a) -> int:
    if a.name not in DEFAULT_RULES:
        print(f"unknown rule {a.name!r}; rules: {', '.join(DEFAULT_RULES)}", file=sys.stderr)
        return 1
    try:
        v = _rule_value(a.name, " ".join(a.value))
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    with _locked("watchlist.json.lock"):
        w = load()
        w["rules"][a.name] = v
        save(w)
    _print_watchlist(load())
    return 0


def _cmd_run(a) -> int:
    print(f"watching every {a.every} s; Ctrl+C stops", flush=True)
    run(a.every, log=lambda m: print(m, flush=True), cycles=1 if a.once else None)
    return 0


def _cmd_alerts(a) -> int:
    bank = Bank()
    if a.action == "check":
        new = check(bank, since=a.since)
        for f in new:
            print(line(f))
        notify(new, log=print)
        print(f"{len(new)} new alert{'s' if len(new) != 1 else ''}")
        return 0
    if a.action == "ack":
        if not a.ids:
            print("ack needs an alert id (see: python -m brain alerts list)", file=sys.stderr)
            return 1
        status = 0
        for i in a.ids:
            try:
                f = ack(bank, i)
                print(f"acknowledged {line(f)}")
            except (KeyError, ValueError) as e:
                print(e.args[0], file=sys.stderr)
                status = 1
        return status
    if a.action == "ack-all":
        n = ack_all(bank)
        print(f"{n} alert{'s' if n != 1 else ''} acknowledged")
        return 0
    done = acked()
    rows = alerts(bank, unacked_only=a.unacked, limit=a.n)
    for f in rows:
        print(f"{' ' if f.id in done else '*'} {line(f)}")
    if rows:
        print(f"{len(unacked(bank))} unacknowledged (*)")
    else:
        print("no unacknowledged alerts" if a.unacked else "no alerts yet (python -m brain alerts check)")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("watch", help="watchlist: tickers, company aircraft, port areas, camera sites; run the loop")
    ws = p.add_subparsers(dest="watch_cmd", required=True)
    s = ws.add_parser("add", help="add to the watchlist: [ticker] T..., aircraft T..., area NAME, site NAME")
    s.add_argument("values", nargs="+", metavar="[KIND] VALUE",
                   help="KIND is ticker (the default), aircraft, area or site")
    s.add_argument("--bbox", nargs=4, type=float, metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
                   help="area: its box, when it is not a built-in port area")
    s.add_argument("--seconds", type=int, help="area: AIS listening window per sample (default 300)")
    s.add_argument("--ticker", help="site: the company it is about")
    s.add_argument("--cameras", nargs="+", metavar="ID", help="site: official camera ids (brain cams near LAT LON)")
    s.add_argument("--near", nargs=2, type=float, metavar=("LAT", "LON"), help="site: take the nearest official cameras")
    s.add_argument("--km", type=float, default=1.0, help="site --near: search radius (default 1 km)")
    s.add_argument("--max", type=int, default=3, help="site --near: at most this many cameras (default 3)")
    s.set_defaults(fn=_cmd_add)
    s = ws.add_parser("remove", help="take an entry off the watchlist (alerts already written stay)")
    s.add_argument("values", nargs="+", metavar="[KIND] VALUE",
                   help="KIND is ticker (the default), aircraft, area or site")
    s.set_defaults(fn=_cmd_remove)
    s = ws.add_parser("list", help="show the watchlist and its rules")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_cmd_list)
    s = ws.add_parser("rule", help="set a rule's threshold (off disables it); forms take a comma list")
    s.add_argument("name", choices=list(DEFAULT_RULES))
    s.add_argument("value", nargs="+")
    s.set_defaults(fn=_cmd_rule)
    s = ws.add_parser("run", help="refresh and check forever, one line per alert; Ctrl+C stops")
    s.add_argument("--every", type=parse_every, default=900, help="900, 15m, 1h ... (default 15m)")
    s.add_argument("--once", action="store_true", help="one round, then stop")
    s.set_defaults(fn=_cmd_run)

    s = sub.add_parser("alerts", help="alerts from the watchlist's rules: check, list, ack ID, ack-all")
    s.add_argument("action", nargs="?", default="list", choices=["check", "list", "ack", "ack-all"])
    s.add_argument("ids", nargs="*", help="ack: alert ids or prefixes")
    s.add_argument("--since", help="check: evaluate facts known after this time (default: the last check)")
    s.add_argument("--unacked", action="store_true", help="list: only unacknowledged alerts")
    s.add_argument("-n", type=int, default=50, help="list: at most this many (default 50)")
    s.set_defaults(fn=_cmd_alerts)
