"""Polite HTTP: identify ourselves, rate-limit per host, retry transient errors,
cache bodies (gzip) under <home>/cache so re-runs do not hammer sources.

SEC counts every sec.gov host as one site and every process on a machine as one client, and
the terminal runs two jobs at once (an INGEST beside INGEST INSIDERS, say). So SEC requests take
turns through a lock file in the home, across processes and hosts, and stay under 10 per second
however many processes ask."""
from __future__ import annotations

import contextlib
import fcntl
import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

from brain import config
from brain.timeutil import now_iso

MIN_INTERVAL = {"sec.gov": 0.15, "data.sec.gov": 0.15, "www.sec.gov": 0.15}  # SEC: <= 10 req/s
DEFAULT_INTERVAL = 0.5
SHARED = {"sec.gov"}  # rate buckets whose turns are shared by every process using this home
_last: dict[str, float] = {}


@dataclass
class Response:
    url: str
    body: bytes
    fetched_at: str
    cached: bool


class FetchError(RuntimeError):
    pass


def redact(url: str) -> str:
    return re.sub(r"(api_key|apikey|token)=[^&]+", r"\1=***", url)


def _cache_paths(url: str):
    key = hashlib.sha256(url.encode()).hexdigest()
    d = config.home() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.gz", d / f"{key}.json"


def bucket(host: str) -> str:
    """The rate-limit bucket of a host: all of sec.gov is one, any other host is its own."""
    return "sec.gov" if host == "sec.gov" or host.endswith(".sec.gov") else host


def wait_turn(host: str, gap: float | None = None) -> None:
    """Sleep until this host's bucket may be asked again, then claim the turn. A shared bucket's last
    request time lives in ``<home>/rate.<bucket>``, read and written under an exclusive lock, so
    processes queue behind each other instead of each keeping its own pace."""
    b = bucket(host)
    gap = MIN_INTERVAL.get(host, DEFAULT_INTERVAL) if gap is None else gap
    if b not in SHARED:
        wait = _last.get(b, 0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[b] = time.time()
        return
    p = config.home() / f"rate.{b}"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            with contextlib.suppress(ValueError):
                _last[b] = max(_last.get(b, 0), float(fh.read().strip() or 0))
            wait = _last.get(b, 0) + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[b] = time.time()
            fh.seek(0)
            fh.truncate()
            fh.write(repr(_last[b]))
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def fetch(url: str, *, user_agent: str | None = None, ttl: float = 86400, retries: int = 4) -> Response:
    body_p, meta_p = _cache_paths(url)
    if ttl > 0 and body_p.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if time.time() - meta["epoch"] < ttl:
            return Response(redact(url), gzip.decompress(body_p.read_bytes()), meta["fetched_at"], True)

    host = urlparse(url).hostname or ""
    gap = MIN_INTERVAL.get(host, DEFAULT_INTERVAL)
    req = urllib.request.Request(url, headers={
        "User-Agent": user_agent or config.WEB_USER_AGENT,
        "Accept-Encoding": "gzip",
    })
    err = None
    for attempt in range(retries):
        wait_turn(host, gap)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
            fetched_at = now_iso()
            body_p.write_bytes(gzip.compress(body))
            meta_p.write_text(json.dumps({"url": redact(url), "fetched_at": fetched_at, "epoch": time.time()}))
            return Response(redact(url), body, fetched_at, False)
        except urllib.error.HTTPError as e:
            err = e
            if e.code not in (429, 500, 502, 503, 504):
                raise FetchError(f"{redact(url)}: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2 ** (attempt + 1))
    raise FetchError(f"{redact(url)}: gave up after {retries} tries ({err})")
