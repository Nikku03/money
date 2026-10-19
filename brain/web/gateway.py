"""The reasoning engine's gateway to the web: a browser with no language model (docs/WEB.md).

The engine decides what it needs and hands the gateway exact instructions, a URL and a recipe (``plan.py``). The
gateway opens the page, renders it, and hands the HTML back with where and when it was read. Fixed rules
(``extract.py``) take the facts out; nothing here guesses or reads between the lines.

The page is rendered by a real browser, driven through browser-use's page API (no agent, no model), when
browser-use and a Chromium are installed; otherwise it is fetched over plain HTTP. Either way the repo's web
rules hold (CLAUDE.md):

- **It says who it is.** Our user agent on every request. In the browser, ``navigator.webdriver`` stays true:
  browser-use's default flag that hides automation is removed, and its telemetry, captcha solver and add-on
  extensions (an ad blocker, a cookie-banner clicker) are off, so pages render as they are.
- **Public hosts only.** Every request the browser makes, redirects and page assets included, is checked and
  failed when its host is local or private (``scout.vet_host``, ``scout.resolve_public``).
- **robots.txt** for our agent on every page and redirect, a pause between requests to one host (longer when the
  site asks for a crawl-delay), and a page budget.
- **It stops** at a login, a paywall or a bot check and records why. It never gets around one.
"""
from __future__ import annotations

import asyncio
import contextlib
import glob
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from urllib.parse import urldefrag, urljoin, urlparse

from brain import config, scout
from brain.timeutil import now_iso

SETTLE = 1.0          # seconds for scripts to fill the page after it has loaded
LOAD_SECONDS = 30.0   # how long a page may take to load
MAX_PAGES = 40
HIDE_AUTOMATION = "--disable-blink-features=AutomationControlled"   # browser-use's default; removed
DECLARE_AUTOMATION = ["--enable-automation", "--enable-blink-features=AutomationControlled"]
CHROMIUM_GLOBS = ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",              # Playwright's, preinstalled
                  os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/chrome"))  # playwright install
# No usage reports from this machine, and quiet logs; set before browser-use is first imported.
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["BROWSER_USE_CLOUD_SYNC"] = "false"
os.environ.setdefault("BROWSER_USE_LOGGING_LEVEL", "error")
_LOGIN = re.compile(r"<input[^>]+type=[\"']?password", re.IGNORECASE)
FEED_TYPES = ("application/rss+xml", "application/atom+xml", "application/xml", "text/xml", "text/plain")
_SKIP_SCHEMES = ("data", "blob", "about", "chrome", "devtools")


def chromium() -> str | None:
    """A Chromium to drive: BRAIN_CHROMIUM, one on PATH, or Playwright's."""
    p = os.environ.get("BRAIN_CHROMIUM", "").strip()
    if p and os.path.exists(p):
        return p
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        if hit := shutil.which(name):
            return hit
    for g in CHROMIUM_GLOBS:
        if hits := sorted(glob.glob(g)):
            return hits[-1]
    return None


def browser_available() -> tuple[bool, str]:
    import importlib.util
    if importlib.util.find_spec("browser_use") is None:
        return False, "browser-use is not installed (pip install browser-use)"
    if not chromium():
        return False, "no Chromium found (set BRAIN_CHROMIUM to its path)"
    return True, ""


@dataclass
class Page:
    """One page as the gateway read it, or why it did not."""
    ok: bool
    url: str                       # as asked
    final_url: str = ""            # after redirects
    status: int = 0
    html: str = ""
    fetched_at: str = ""
    mode: str = ""                 # "browser" or "http"
    error: str = ""
    refused: list = field(default_factory=list)   # requests refused while rendering it: {"url", "why"}


def _host_problem(host: str) -> str | None:
    """Why a host may not be read (local, private, not a plain name), or None."""
    try:
        scout.vet_host(host)
        scout.resolve_public(host, 443)
    except PermissionError as e:
        return str(e)
    except OSError as e:
        return f"{host} does not resolve ({type(e).__name__})"
    return None


class Gateway:
    """Opens pages for the reasoning engine. ``mode``: "browser", "http" or "auto" (the browser when it is
    available). ``host_problem(host) -> str | None`` says why a host may not be read; the default is scout's
    public-host check (tests pass their own). ``reader`` is a ``scout.PageReader``: its robots.txt cache,
    pacing and page budget are the gateway's."""

    def __init__(self, mode: str = "auto", max_pages: int = MAX_PAGES, reader=None, host_problem=None,
                 settle: float = SETTLE):
        if mode == "auto":
            mode = "browser" if browser_available()[0] else "http"
        if mode == "browser" and not (ok := browser_available())[0]:
            raise RuntimeError(ok[1])
        self.mode, self.settle = mode, settle
        self.reader = reader or scout.PageReader(max_pages=max_pages)
        self._host_problem = host_problem or _host_problem
        self._hosts: dict[str, str | None] = {}
        self._loop = None
        self._session = None
        self._refused: dict[str, list] = {}   # page session id -> requests refused

    # ---------- the rules ----------
    def host_problem(self, host: str) -> str | None:
        host = (host or "").lower()
        if host not in self._hosts:
            self._hosts[host] = self._host_problem(host)
        return self._hosts[host]

    def url_problem(self, url: str, document: bool = True) -> str | None:
        """Why ``url`` may not be fetched: not http(s), a non-public host, or (a page) robots.txt."""
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return "the gateway reads http(s) URLs only"
        if why := self.host_problem(u.hostname):
            return why
        if document:
            ok, why = self.reader.robots(url)
            if not ok:
                return why if "local or private" in why or "not a plain host" in why else f"{why} (respecting it)"
        return None

    def _pace(self, host: str) -> None:
        r = self.reader
        wait = r._last.get(host, -1e9) + max(r.min_interval, r._delay.get(host, 0.0)) - r.clock()
        if wait > 0:
            r.sleep(wait)
        r._last[host] = r.clock()

    def _judge(self, page: Page) -> Page:
        """Refuse what the rules say not to read: a login, a paywall, a bot check, an error status."""
        if page.status in scout.BLOCKED_STATUS:
            page.ok, page.error = False, (f"blocked (HTTP {page.status}, {scout.BLOCKED_STATUS[page.status]}); "
                                          "not getting around it")
        elif page.status not in (0, 200):
            page.ok, page.error = False, f"HTTP {page.status}"
        else:
            doc = scout.html_to_text(page.html)
            if scout._CHALLENGE_TITLE.search(doc["title"]) or (len(doc["text"]) < 500 and scout._CHALLENGE_MARK.search(page.html)):
                page.ok, page.error = False, "blocked: the page is a bot check; not getting around it"
            elif scout._NOT_FREE.search(page.html):
                page.ok, page.error = False, "the page marks itself as paywalled; not reading behind it"
            elif _LOGIN.search(page.html) and len(doc["text"]) < 1500:
                page.ok, page.error = False, "a login page; the gateway never logs in"
        if not page.ok:
            page.html = ""
        return page

    # ---------- opening a page ----------
    def open(self, url: str, feed: bool = False) -> Page:
        """Read one page by the rules. Never raises for the page's own trouble: the Page says what happened.
        ``feed``: an RSS or Atom feed, always read over plain HTTP (there is nothing to render)."""
        url = urldefrag(str(url or "").strip())[0]
        if urlparse(url).scheme not in ("http", "https"):
            return Page(False, url, error="not read: the gateway reads http(s) URLs only")
        if self.reader.fetched >= self.reader.max_pages:
            return Page(False, url, error=f"page budget used ({self.reader.max_pages} pages)")
        if why := self.url_problem(url):
            return Page(False, url, error=f"not read: {why}")
        self.reader.fetched += 1
        if self.mode == "browser":
            self._pace((urlparse(url).hostname or "").lower())   # over HTTP, reader._get paces each request
        host = (urlparse(url).hostname or "").lower()
        use_browser = (self.mode == "browser" and not feed
                       and not (host == "sec.gov" or host.endswith(".sec.gov")))   # SEC gets its own user agent
        try:
            page = self._browser(url) if use_browser else self._http(url, feed)
        except Exception as e:  # noqa: BLE001 - network or browser trouble is a result, not a crash
            return Page(False, url, mode=self.mode, error=f"could not read it ({type(e).__name__}: {str(e)[:160]})")
        if page.ok and page.final_url and page.final_url != url and (why := self.url_problem(page.final_url)):
            page.ok, page.html, page.error = False, "", f"redirected to a page the rules refuse: {why}"
        return self._judge(page) if page.ok and not feed else page

    def _http(self, url: str, feed: bool = False) -> Page:
        target = url
        for _ in range(scout.MAX_REDIRECTS + 1):
            status, hdrs, body = self.reader._get(target)   # our user agent; the connection pins the vetted address
            if status in (301, 302, 303, 307, 308) and hdrs.get("location"):
                target = urldefrag(urljoin(target, hdrs["location"]))[0]
                if why := self.url_problem(target):
                    return Page(False, url, final_url=target, status=status, mode="http",
                                error=f"redirected to a page the rules refuse: {why}")
                continue
            break
        else:
            return Page(False, url, mode="http", error="too many redirects")
        ctype = hdrs.get("content-type", "").split(";")[0].strip().lower()
        if status in scout.BLOCKED_STATUS or status != 200:
            return Page(True, url, final_url=target, status=status, mode="http")   # _judge words the refusal
        ok_types = FEED_TYPES if feed else ("text/html", "application/xhtml+xml")
        if ctype and ctype not in ok_types:
            return Page(False, url, final_url=target, status=status, mode="http", error=f"not a web page ({ctype})")
        html = body[:scout.MAX_BYTES].decode(scout._charset(hdrs.get("content-type", "")), "replace")
        return Page(True, url, final_url=target, status=status, html=html, fetched_at=now_iso(), mode="http")

    # ---------- the browser ----------
    def _run(self, coro):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def _browser(self, url: str) -> Page:
        return self._run(self._render(url))

    async def _start(self) -> None:
        if self._session is not None:
            return
        import logging

        from browser_use import BrowserProfile, BrowserSession
        for name in ("browser_use", "bubus", "cdp_use"):
            logging.getLogger(name).setLevel(logging.WARNING)
        profile = BrowserProfile(headless=True, executable_path=chromium(), user_agent=config.WEB_USER_AGENT,
                                 captcha_solver=False, enable_default_extensions=False,
                                 ignore_default_args=[HIDE_AUTOMATION],
                                 args=[*DECLARE_AUTOMATION, "--no-sandbox"], block_ip_addresses=False)
        self._session = BrowserSession(browser_profile=profile)
        await self._session.start()
        client = self._session.cdp_client
        loop = asyncio.get_running_loop()

        def on_paused(event, session_id=None):
            rid = event.get("requestId")
            req_url = (event.get("request") or {}).get("url", "")
            is_doc = event.get("resourceType") == "Document"

            async def decide():
                why = await loop.run_in_executor(None, self._request_problem, req_url, is_doc)
                try:
                    if why:
                        self._refused.setdefault(session_id or "", []).append({"url": req_url[:200], "why": why})
                        await client.send.Fetch.failRequest(params={"requestId": rid, "errorReason": "BlockedByClient"},
                                                            session_id=session_id)
                    else:
                        await client.send.Fetch.continueRequest(params={"requestId": rid}, session_id=session_id)
                except Exception:  # noqa: BLE001, S110 - the page may already be gone
                    pass
            asyncio.ensure_future(decide())
        client.register.Fetch.requestPaused(on_paused)

    def _request_problem(self, url: str, is_doc: bool) -> str | None:
        """Every request the browser makes passes through here: page assets need a public host, and a page
        (the first load and every redirect) also robots.txt."""
        if urlparse(url).scheme in _SKIP_SCHEMES:
            return None
        return self.url_problem(url, document=is_doc)

    async def _render(self, url: str) -> Page:
        await self._start()
        page = await self._session.new_page()
        sid = await page._ensure_session()
        try:
            await page._client.send.Fetch.enable(params={"patterns": [{"urlPattern": "*"}]}, session_id=sid)
            try:
                await page.goto(url)
            except RuntimeError:
                if not self._refused.get(sid):
                    raise
                why = self._refused[sid][0]["why"]
                return Page(False, url, mode="browser", refused=self._refused.get(sid, []), error=f"not read: {why}")
            end = time.monotonic() + LOAD_SECONDS
            while time.monotonic() < end and await page.evaluate("() => document.readyState") != "complete":
                await asyncio.sleep(0.25)
            await asyncio.sleep(self.settle)
            raw = await page.evaluate(
                "() => JSON.stringify({url: location.href, type: document.contentType, "
                "html: document.documentElement.outerHTML, "
                "status: ((performance.getEntriesByType('navigation')[0] || {}).responseStatus) || 0})")
            got = json.loads(raw)
        finally:
            refused = self._refused.pop(sid, [])
            with contextlib.suppress(Exception):
                await self._session.close_page(page)
        final = urldefrag(got.get("url") or url)[0]
        if final.startswith("chrome-error:"):
            why = refused[0]["why"] if refused else "the browser could not load it"
            return Page(False, url, mode="browser", refused=refused, error=f"not read: {why}")
        if got.get("type") not in ("text/html", "application/xhtml+xml"):
            return Page(False, url, final_url=final, mode="browser", refused=refused,
                        error=f"not a web page ({got.get('type') or 'unknown type'})")
        return Page(True, url, final_url=final, status=int(got.get("status") or 0), html=got.get("html") or "",
                    fetched_at=now_iso(), mode="browser", refused=refused)

    def close(self) -> None:
        if self._session is not None:
            with contextlib.suppress(Exception):
                self._run(self._session.kill())
            self._session = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
