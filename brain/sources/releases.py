"""Earnings press releases from SEC filings: each release's opening text, point-in-time (docs/EVENTS.md).

For each earnings 8-K (item 2.02) in the bank, the filing's folder listing names its documents; the press release
is exhibit 99.1. Its opening is read (an HTTP range request: the headline, the dateline and the highlights come
first) and kept as a ``release.earnings`` fact: the title, the opening text and the event tags fixed rules find
in it (``brain.web.events``). Its known_at is the filing's own acceptance time, so a test reads a release only
from the moment it was public. The opening text is kept, so the rules can be re-applied without fetching again.

SEC's fair-access rules apply: SEC_USER_AGENT on every request, and the shared SEC bucket in ``brain/net.py``
keeps every process together under 10 requests a second.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from brain import config, net
from brain import events as earnings
from brain.bank import Bank, Fact
from brain.timeutil import now_iso
from brain.web import events as tags

ATTR = "release.earnings"
SINCE = "2010-12-01"
HEAD_BYTES = 200_000     # the start of a release; its financial tables come later
OPENING_CHARS = 8000     # text kept: the headline, the highlights, the quotes and the outlook section
_EX991 = re.compile(r"(ex|exhibit|dex)[-_ ]?99[-_.]?0?1(?!\d)", re.IGNORECASE)
_EX99 = re.compile(r"(ex|exhibit|dex)[-_ ]?99", re.IGNORECASE)
_DOC = re.compile(r"\.(htm|html|txt)$", re.IGNORECASE)
_WRAPPER = re.compile(r"<(TYPE|SEQUENCE|FILENAME|DESCRIPTION)>[^\n<]*", re.IGNORECASE)   # EDGAR's SGML lines
_FOLDER = re.compile(r"/Archives/edgar/data/(\d+)/(\d{18})/")


def exhibit_name(names: list[str]) -> str | None:
    """The press release among a filing's documents: exhibit 99.1, else the first exhibit 99."""
    docs = [n for n in names if _DOC.search(n) and "index" not in n.lower()]
    for rx in (_EX991, _EX99):
        hit = [n for n in docs if rx.search(n)]
        if hit:
            return min(hit, key=len)
    return None


def _get(url: str, max_bytes: int | None = None, retries: int = 4) -> bytes:
    """One SEC request on the shared bucket; with ``max_bytes``, only the start of the document."""
    headers = {"User-Agent": config.sec_user_agent()}
    if max_bytes:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"
    err = None
    for attempt in range(retries):
        net.wait_turn(urlparse(url).hostname or "www.sec.gov")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read(max_bytes) if max_bytes else r.read()
        except urllib.error.HTTPError as e:
            err = e
            if e.code not in (429, 500, 502, 503, 504):
                raise net.FetchError(f"{url}: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2 ** (attempt + 1))
    raise net.FetchError(f"{url}: gave up after {retries} tries ({err})")


def opening(html: str) -> tuple[str, str]:
    """(title, opening text) of a release: its first line and its first OPENING_CHARS characters."""
    from brain import scout
    text = scout.html_to_text(_WRAPPER.sub(" ", html))["text"]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = next((ln for ln in lines if len(ln) > 15 and not ln.lower().startswith(("exhibit", "ex-99", "ex 99"))),
                 lines[0] if lines else "")
    return title[:300], " ".join(text.split())[:OPENING_CHARS]


def read_release(filing: Fact, fetched_at: str | None = None) -> Fact:
    """The release fact for one earnings filing (two SEC requests: the folder listing, then the release)."""
    m = _FOLDER.search(filing.source.get("url", ""))
    if not m:
        raise ValueError(f"{filing.ref}: no EDGAR folder in its URL")
    folder = f"https://www.sec.gov/Archives/edgar/data/{m.group(1)}/{m.group(2)}/"
    names = [i["name"] for i in json.loads(_get(folder + "index.json"))["directory"]["item"]]
    name = exhibit_name(names)
    title, text, url = "(no exhibit 99 in the filing)", "", folder
    if name:
        url = folder + name
        raw = _get(url, HEAD_BYTES).decode("utf-8", "replace")
        title, text = opening(raw)
    return Fact(entity=filing.entity, attr=ATTR, value=title or "(untitled)", as_of=filing.as_of,
                known_at=filing.known_at, tier="filed", source={"name": "SEC EDGAR 8-K exhibit 99", "url": url},
                context={"accn": filing.context.get("accn"), "filing": filing.ref, "exhibit": name,
                         "opening_chars": OPENING_CHARS,
                         "events": tags.tag(text), "opening": text},
                retrieved_at=fetched_at or now_iso())


WORKERS = 6   # requests in flight at once; the shared SEC bucket still spaces them under 10 a second


def collect(bank: Bank, tickers, since: str = SINCE, log=print, workers: int = WORKERS) -> dict:
    """Read every earnings release since ``since`` that the bank does not hold yet."""
    from concurrent.futures import ThreadPoolExecutor

    def one(f):
        try:
            return read_release(f), None
        except Exception as e:  # noqa: BLE001 - one filing's trouble is recorded, the rest still run
            return None, {"filing": f.ref, "why": f"{type(e).__name__}: {str(e)[:120]}"}
    view = bank.view()
    kept, failed = 0, []
    for t in [x.upper() for x in tickers]:
        have = {f.context.get("accn") for f in view.facts(t, ATTR) if f.context.get("opening_chars") == OPENING_CHARS}
        todo = [f for f in earnings.distinct(earnings.earnings_events(view, t))
                if f.known_at >= since and f.context.get("accn") not in have]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(one, todo))
        out = [r for r, _ in done if r is not None]
        failed += [w for _, w in done if w is not None]
        kept += bank.add_many(out)
        if log:
            log(f"{t}: {len(out)} releases read, {sum(bool(x.context['events']) for x in out)} with event tags")
    return {"kept": kept, "failed": failed}


def _cmd(a) -> int:
    from brain import universe
    path = universe.BANDS[a.band][2] if a.band else universe.BROAD if a.broad else config.UNIVERSE
    tickers = [w.upper() for w in a.tickers] or json.loads(path.read_text())["tickers"]
    out = collect(Bank(), tickers, a.since)
    print(f"{out['kept']} new release facts; {len(out['failed'])} filings failed")
    for f in out["failed"][:10]:
        print(f"  {f['filing']}: {f['why']}")
    return 1 if out["failed"] and not out["kept"] else 0


def add_cli(sub) -> None:
    s = sub.add_parser("releases", help="earnings press releases (SEC 8-K exhibit 99): opening text and event tags")
    s.add_argument("tickers", nargs="*", help="default: the research universe (universe/us_large.json)")
    s.add_argument("--broad", action="store_true", help="the broad universe (universe/us_broad.json)")
    s.add_argument("--band", choices=["small", "micro"], help="universe/us_small.json or us_micro.json")
    s.add_argument("--since", default=SINCE, help=f"first filing date (default {SINCE})")
    s.set_defaults(fn=_cmd)
