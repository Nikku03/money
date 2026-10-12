"""Polite HTTP: identify ourselves, rate-limit per host, retry transient errors,
cache bodies (gzip) under <home>/cache so re-runs do not hammer sources."""
from __future__ import annotations

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
        wait = _last.get(host, 0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[host] = time.time()
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
