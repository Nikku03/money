"""Local HTTP server for MONEY TERMINAL.

Every page on the open web can make a browser send requests to localhost, so a
local server is only as private as its checks. Four of them, all required:

- Host check: a request must name the loopback address and port we bound.
  This defeats DNS rebinding, where an attacker's domain re-resolves to
  127.0.0.1 and the browser then treats our pages as theirs.
- Token: a random value per server start, put only into the HTML page. Every
  /api/* call must echo it. Another origin cannot read our page (no CORS
  headers are ever sent), so it cannot learn the token.
- JSON bodies only, at most 64 KB: a cross-site form cannot send
  application/json, and nothing large is ever needed.
- A strict Content-Security-Policy: the page runs only its own script.

The bank is not thread-safe, so one ``Bank`` lives in server state behind an
RLock and every request that reads it holds the lock. Ingest runs in child
processes (``jobs.py``); when one finishes, a fresh Bank is swapped in.
"""
from __future__ import annotations

import functools
import hmac
import importlib
import ipaddress
import json
import re
import secrets
import socket
import socketserver
import sys
import threading
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from brain import config
from brain.bank import Bank
from brain.terminal import commands, data, jobs
from brain.timeutil import now_iso, today

STATIC = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/static/style.css": ("style.css", "text/css; charset=utf-8")}  # nothing else is served
MAX_BODY = 64 * 1024
MAX_DISCARD = 1024 * 1024  # an oversize body is read and dropped up to this, so the 413 reaches the client
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'")
HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
           "Content-Security-Policy": CSP, "Referrer-Policy": "no-referrer"}
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1")
HOST_HEADER = re.compile(r"(\[[0-9a-f:.]+\]|[a-z0-9.\-]+):(\d{1,5})")
FACT_ID = re.compile(r"[0-9A-Fa-f]{1,20}")
TICKERS_TTL = 10.0  # seconds; completion asks on every keystroke
MAX_INGEST_TARGETS = 10
MAX_QUESTION = 4000
MAX_HISTORY = 50  # older chat turns are dropped; the analyst reads fewer still

API = {  # path -> (method, handler)
    "/api/run": ("GET", "api_run"),
    "/api/complete": ("GET", "api_complete"),
    "/api/status": ("GET", "api_status"),
    "/api/fact": ("GET", "api_fact"),
    "/api/jobs": ("GET", "api_jobs"),
    "/api/ask": ("POST", "api_ask"),
    "/api/ingest": ("POST", "api_ingest"),
}


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _is_ip(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def _analyst():
    """Imported on use, never at module level: a missing or broken analyst must not stop the terminal."""
    return importlib.import_module("brain.terminal.analyst")


def _log_exc(where: str) -> None:
    print(f"[terminal] error in {where}:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)


# ---------- JSON ----------
def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    return str(o)


def dumps(obj) -> str:
    try:
        return json.dumps(obj, allow_nan=False, default=_default, ensure_ascii=False)
    except ValueError:  # NaN somewhere (the browser's JSON.parse rejects it): send null instead
        return json.dumps(data.clean(obj), allow_nan=False, default=_default, ensure_ascii=False)


def warm(bank: Bank) -> Bank:
    """Read every journal shard now. A Bank loads an entity on first use, and MOV or EV touch the whole
    universe (about 100 MB of journal, 6-7 s): done lazily under the bank lock, that first call stalls
    every pane, after each start and again after each finished job."""
    for e in bank.entities():
        bank.attrs(e)
    return bank


def _universe_size() -> int:
    try:
        return len(json.loads(config.UNIVERSE.read_text())["tickers"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


# ---------- server ----------
class TerminalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host: str, port: int, bank: Bank | None = None):
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        self.host = host
        self.loopback = is_loopback(host)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.bank = bank if bank is not None else Bank()
        self.static_dir = STATIC
        self.quiet = False  # True silences per-request error lines (tests); tracebacks are always logged
        self.jobs = jobs.JobManager(on_finish=self.reload_bank)
        self._tickers: tuple[float, list[str]] | None = None
        self._generation = self._swapped = 0  # bank reloads started / last one swapped in
        self._analyst_logged = False
        super().__init__((host, port), Handler)

    def server_bind(self):
        # HTTPServer.server_bind calls socket.getfqdn, a reverse DNS lookup that can stall; we need neither.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = str(self.server_address[0]), int(self.server_address[1])

    def server_close(self):
        self.jobs.shutdown()
        super().server_close()

    # ---------- state ----------
    def reload_bank(self, job: dict | None = None) -> None:
        """A job wrote to the journal from another process; re-read it from scratch. The fresh Bank is
        read in full before the swap and outside the lock, so panes keep answering from the old one
        meanwhile. A reload that started earlier never replaces one that started later."""
        with self.lock:
            root = self.bank.root
            self._generation += 1
            gen = self._generation
        fresh = warm(Bank(root))
        with self.lock:
            if gen > self._swapped:
                self.bank, self._swapped, self._tickers = fresh, gen, None

    def tickers(self) -> list[str]:
        """Bank entities, cached briefly. Holds the lock."""
        with self.lock:
            now = time.monotonic()
            if self._tickers is None or now - self._tickers[0] > TICKERS_TTL:
                self._tickers = (now, sorted(self.bank.entities()))
            return list(self._tickers[1])

    def host_allowed(self, value: str) -> bool:
        m = HOST_HEADER.fullmatch(value.strip().lower())
        if not m or int(m.group(2)) != self.server_port:
            return False
        name = m.group(1).strip("[]")
        if name in LOOPBACK_NAMES:
            return True
        if self.loopback:
            return False
        # Bound beyond loopback: DNS rebinding needs a domain name, so accept IP literals and our own host name.
        return name == self.host.lower() or _is_ip(name)

    def analyst_status(self) -> tuple[bool, str]:
        try:
            ok, reason = _analyst().available()
            return bool(ok), str(reason or "")
        except Exception as e:  # noqa: BLE001 - a broken analyst must never break the terminal
            if not self._analyst_logged:
                self._analyst_logged = True
                _log_exc("analyst")
            return False, f"the analyst failed to load ({type(e).__name__}); see the server log"

    # ---------- commands ----------
    def run_line(self, line: str, read_only: bool = False) -> dict:
        """One terminal command. ``read_only`` (the analyst's tools) refuses INGEST and ASK."""
        line = (line or "").strip()
        if not line:
            return {"ok": False, "fn": None, "ticker": None, "args": [], "error": "empty command", "hint": "HELP"}
        try:
            cmd = commands.parse(line)
        except commands.CommandError as e:
            return {"ok": False, "fn": None, "ticker": None, "args": [], "error": e.message, "hint": e.hint}
        if read_only and cmd.fn in ("INGEST", "ASK"):
            return {"ok": False, "fn": cmd.fn, "ticker": cmd.ticker, "args": cmd.args,
                    "error": f"{cmd.fn} is not available here (read-only)", "hint": "HELP"}
        if cmd.fn == "JOBS":
            return self.jobs_view(cmd)
        if cmd.fn == "INGEST":
            return self.ingest_command(cmd)
        if cmd.fn == "ASK":
            return self.ask_view(cmd)
        with self.lock:
            return data.run(self.bank, cmd)

    def ask_view(self, cmd) -> dict:
        """Opens the analyst pane. A GET never calls the model (slow, and it costs money): the pane
        posts ``data.question`` to /api/ask itself."""
        ok, reason = self.analyst_status()
        question = " ".join(cmd.args).strip()
        return {"ok": True, "fn": "ASK", "ticker": cmd.ticker, "args": list(cmd.args), "title": "ASK · analyst",
                "asof": today(), "view": "ask",
                "data": {"question": question, "ticker": cmd.ticker, "available": ok, "reason": reason}}

    def jobs_view(self, cmd, title: str = "Background jobs") -> dict:
        return {"ok": True, "fn": cmd.fn, "ticker": cmd.ticker, "args": list(cmd.args), "title": title,
                "asof": today(), "view": "table", "data": jobs.table(self.jobs.list_jobs())}

    def ingest_command(self, cmd) -> dict:
        targets = ([cmd.ticker] if cmd.ticker else []) + list(cmd.args)
        err = {"ok": False, "fn": cmd.fn, "ticker": cmd.ticker, "args": list(cmd.args)}
        if not targets:
            return {**err, "error": "INGEST needs a target: a ticker, MARKETS, MACRO or UNIVERSE",
                    "hint": "INGEST AAPL"}
        if len(targets) > MAX_INGEST_TARGETS:
            return {**err, "error": f"at most {MAX_INGEST_TARGETS} targets at once", "hint": "JOBS"}
        try:
            for t in targets:  # all valid, or none queued
                jobs.validate(t)
            started = [self.jobs.submit(t) for t in targets]
        except jobs.JobError as e:
            return {**err, "error": e.message, "hint": e.hint}
        title = "INGEST " + ", ".join(f"{j['target']}: job {j['id']} {j['status']}" for j in started)
        return self.jobs_view(cmd, title=title)


def make_server(host: str = "127.0.0.1", port: int = 8765, bank: Bank | None = None) -> tuple[ThreadingHTTPServer, str]:
    """(server, token). Port 0 picks a free port. Call ``serve_forever`` to run it."""
    server = TerminalServer(host, port, bank)
    return server, server.token


def _url_host(host: str) -> str:
    if host in ("", "0.0.0.0"):
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    return f"[{host}]" if ":" in host else host


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    try:
        server, _ = make_server(host, port)
    except OSError as e:
        raise SystemExit(f"cannot listen on {host}:{port}: {e.strerror or e}. Try another --port.")
    url = f"http://{_url_host(host)}:{server.server_port}/"
    # Read the whole bank before announcing the URL: loading it while serving would slow every request
    # that arrives meanwhile (the reads and the requests take turns on one interpreter lock).
    t0 = time.monotonic()
    print("reading the memory bank...", end=" ", flush=True)
    warm(server.bank)
    print(f"{len(server.tickers())} entities in {time.monotonic() - t0:.1f}s", flush=True)
    if not server.loopback:
        print(f"WARNING: listening on {host or 'all interfaces'}, not loopback. The terminal has no user "
              f"accounts: anyone who can reach this port can open it, read the bank and start ingest jobs.",
              file=sys.stderr)
    print(f"MONEY TERMINAL on {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - no browser is fine; the URL is printed
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
    finally:
        server.server_close()


# ---------- requests ----------
class BadRequest(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _param(q: dict, name: str) -> str:
    return (q.get(name) or [""])[0]


class Handler(BaseHTTPRequestHandler):
    server: TerminalServer
    _sent, _consumed = False, True  # reset per request in _handle; protocol errors never drain
    timeout = 60  # a client that stalls mid-request does not hold a thread forever

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    # ---------- plumbing ----------
    def _handle(self, method: str) -> None:
        self._sent, self._consumed = False, False
        try:
            self._route(method)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except BadRequest as e:
            self.close_connection = True
            self._json(e.code, {"ok": False, "error": e.message})
        except Exception:  # noqa: BLE001 - log it; the client learns nothing but "internal error"
            _log_exc(f"{method} {urlsplit(self.path).path}")
            if not self._sent:
                self.close_connection = True
                self._json(500, {"ok": False, "error": "internal error"})

    def _route(self, method: str) -> None:
        if not self._host_ok():
            return self._json(403, {"ok": False, "error": "forbidden: unexpected Host header"})
        url = urlsplit(self.path)
        path = url.path
        if path.startswith("/api/"):
            if not self._token_ok():
                return self._json(401, {"ok": False, "error": "missing or wrong X-Terminal-Token"})
            route = API.get(path)
            if route is None:
                return self._json(404, {"ok": False, "error": "not found"})
            want, name = route
            if method != want:
                return self._json(405, {"ok": False, "error": f"use {want}"}, {"Allow": want})
            if method == "POST":
                return getattr(self, name)(self._read_json())
            try:
                q = parse_qs(url.query, keep_blank_values=True, max_num_fields=20)
            except ValueError:
                raise BadRequest(400, "too many query parameters")
            return getattr(self, name)(q)
        if method != "GET":
            return self._json(405, {"ok": False, "error": "use GET"}, {"Allow": "GET"})
        if path == "/":
            return self._page()
        if path in STATIC_FILES:
            return self._static(*STATIC_FILES[path])
        return self._json(404, {"ok": False, "error": "not found"})

    def _host_ok(self) -> bool:
        values = self.headers.get_all("Host") or []
        return len(values) == 1 and self.server.host_allowed(values[0])

    def _token_ok(self) -> bool:
        given = self.headers.get("X-Terminal-Token") or ""
        return hmac.compare_digest(given.encode("utf-8", "replace"), self.server.token.encode())

    def _read_json(self) -> dict:
        if self.headers.get("Transfer-Encoding"):
            raise BadRequest(400, "send a Content-Length body, not chunked")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise BadRequest(400, "Content-Length required")
        if length < 0:
            raise BadRequest(400, "bad Content-Length")
        if length > MAX_BODY:
            raise BadRequest(413, f"body larger than {MAX_BODY // 1024} KB")
        self._consumed = True
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise BadRequest(400, "incomplete body")
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise BadRequest(400, "Content-Type must be application/json")
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise BadRequest(400, "body is not valid JSON")
        if not isinstance(obj, dict):
            raise BadRequest(400, "body must be a JSON object")
        return obj

    def _drain(self) -> None:
        """Read and drop a body we will not use. Closing a socket with unread input makes the kernel send
        a reset, and the client may then lose our response (a 401 or 413) before reading it."""
        if self._consumed:
            return
        self._consumed = True
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        left = min(length, MAX_DISCARD)
        try:
            while left > 0:
                chunk = self.rfile.read(min(left, 65536))
                if not chunk:
                    break
                left -= len(chunk)
        except OSError:
            pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self._drain()
        self._sent = True
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in {**HEADERS, **(extra or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj, extra: dict | None = None) -> None:
        self._send(code, dumps(obj).encode("utf-8"), "application/json; charset=utf-8", extra)

    def version_string(self) -> str:
        return "MoneyTerminal"  # no Python version in the Server header

    def send_error(self, code, message=None, explain=None):
        """Protocol errors (bad request line, unsupported method) as JSON with the same headers."""
        self.close_connection = True
        try:
            phrase = HTTPStatus(code).phrase
        except ValueError:
            phrase = "error"
        self._json(code, {"ok": False, "error": message or phrase})

    def log_request(self, code="-", size="-"):
        if isinstance(code, HTTPStatus):
            code = code.value
        if str(code).isdigit() and int(code) >= 400:  # successful requests are not worth a line each
            super().log_request(code, size)

    def log_message(self, format, *args):
        if not getattr(self.server, "quiet", False):
            super().log_message(format, *args)

    # ---------- pages ----------
    def _page(self) -> None:
        try:
            html = (self.server.static_dir / "index.html").read_text(encoding="utf-8")
        except FileNotFoundError:
            return self._json(404, {"ok": False, "error": "frontend missing: brain/terminal/static/index.html"})
        self._send(200, html.replace("{{TOKEN}}", self.server.token).encode("utf-8"), "text/html; charset=utf-8")

    def _static(self, name: str, ctype: str) -> None:
        try:
            body = (self.server.static_dir / name).read_bytes()
        except FileNotFoundError:
            return self._json(404, {"ok": False, "error": "not found"})
        self._send(200, body, ctype)

    # ---------- API ----------
    def api_run(self, q: dict) -> None:
        self._json(200, self.server.run_line(_param(q, "cmd")))

    def api_complete(self, q: dict) -> None:
        items = commands.complete(_param(q, "q"), self.server.tickers())
        self._json(200, {"ok": True, "items": items})

    def api_status(self, q: dict) -> None:
        tickers = self.server.tickers()
        ok, reason = self.server.analyst_status()
        self._json(200, {"ok": True, "entities": len(tickers), "tickers": tickers, "universe": _universe_size(),
                         "jobs_running": self.server.jobs.running(), "analyst": {"available": ok, "reason": reason},
                         "time": now_iso()})

    def api_fact(self, q: dict) -> None:
        ticker, fid = _param(q, "ticker").strip(), _param(q, "id").strip().strip("[]")
        if not (jobs.TICKER.fullmatch(ticker) and FACT_ID.fullmatch(fid)):
            raise BadRequest(400, "need ticker=<symbol>&id=<hex fact id prefix>")
        # Built directly, not parsed: a ticker that is also a function code (CF, GP) would parse as the function.
        cmd = commands.Command(fn="SHOW", ticker=ticker.upper(), args=[fid.upper()], raw=f"{ticker} SHOW {fid}")
        with self.server.lock:
            res = data.run(self.server.bank, cmd)
        self._json(200, res)

    def api_jobs(self, q: dict) -> None:
        self._json(200, {"ok": True, "jobs": self.server.jobs.list_jobs()})

    def api_ingest(self, body: dict) -> None:
        try:
            job = self.server.jobs.submit(body.get("target"))
        except jobs.JobError as e:
            return self._json(400, {"ok": False, "error": e.message, "hint": e.hint})
        self._json(200, {"ok": True, "job": job})

    def api_ask(self, body: dict) -> None:
        question, ticker, history = body.get("question"), body.get("ticker") or None, body.get("history") or []
        if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION:
            raise BadRequest(400, f"question must be a non-empty string of at most {MAX_QUESTION} characters")
        if ticker is not None and not (isinstance(ticker, str) and jobs.TICKER.fullmatch(ticker)):
            raise BadRequest(400, "ticker must be a symbol or null")
        if not (isinstance(history, list) and all(
                isinstance(h, dict) and h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str)
                for h in history)):
            raise BadRequest(400, "history must be a list of {role: user|assistant, content: str} messages")
        history = history[-MAX_HISTORY:]
        fallback = f"{ticker.upper()} BRIEF" if ticker else "HELP"
        try:
            analyst = _analyst()
        except Exception:  # noqa: BLE001
            _log_exc("analyst import")
            return self._json(200, {"ok": False, "error": "the analyst is not available (it failed to load)",
                                    "hint": fallback})
        reader = functools.partial(self.server.run_line, read_only=True)  # takes the bank lock per call
        try:
            res = analyst.ask(reader, question, ticker=ticker.upper() if ticker else None, history=history)
        except Exception:  # noqa: BLE001
            _log_exc("analyst.ask")
            res = {"ok": False, "error": "the analyst failed (internal error)", "hint": fallback}
        if not isinstance(res, dict):
            res = {"ok": False, "error": "the analyst returned nothing usable", "hint": fallback}
        self._json(200, res)
