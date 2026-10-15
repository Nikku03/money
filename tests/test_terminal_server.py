"""Terminal server and background jobs.

A real server runs on port 0 in a thread; requests go through urllib with an
explicit Host header (and no proxy). Jobs run a harmless child command injected
into the JobManager, so nothing here touches the network.
"""
import contextlib
import http.client
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

from brain import cli
from brain.bank import Bank
from brain.terminal import commands, data, jobs, server
from tests.util import TempHome, filed

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route 127.0.0.1 through a proxy
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'")


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def fake_argv(target: str) -> list[str]:
    return py(f"print('fake ingest {target}')")


def wait_until(cond, timeout=20.0):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            return False
        time.sleep(0.01)
    return True


class ServerTest(TempHome):
    def setUp(self):
        super().setUp()
        self.bank.add(filed("AAPL", "us-gaap:Revenues", 1.0, "2024-09-28", "2024-11-01"))
        self.static = Path(self._tmp.name) / "static"
        self.static.mkdir()
        (self.static / "index.html").write_text('<meta name="terminal-token" content="{{TOKEN}}"><p>MONEY</p>')
        (self.static / "app.js").write_text("console.log('app');")
        (self.static / "style.css").write_text("body{}")
        (self.static / "secret.txt").write_text("not served")
        (Path(self._tmp.name) / "x").write_text("outside static")
        self.server, self.token = server.make_server("127.0.0.1", 0, bank=self.bank)
        self.server.static_dir = self.static
        self.server.quiet = True
        self.server.jobs.argv_for = fake_argv
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.assertTrue(self.server.jobs.wait(20))
        self.server.server_close()
        self.thread.join(5)
        super().tearDown()

    # ---------- helpers ----------
    def req(self, path, method="GET", body=None, headers=None, host=None, token=True):
        h = {"Host": host or f"127.0.0.1:{self.port}"}
        if token:
            h["X-Terminal-Token"] = self.token if token is True else token
        h.update(headers or {})
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=body, method=method, headers=h)
        try:
            with OPENER.open(r, timeout=20) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, e.read()

    def get_json(self, path, **kw):
        code, headers, body = self.req(path, **kw)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        return code, json.loads(body)

    def post_json(self, path, obj, **kw):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        code, headers, raw = self.req(path, method="POST", body=body,
                                      headers={"Content-Type": "application/json", **kw.pop("headers", {})}, **kw)
        return code, json.loads(raw)

    def raw(self, data: bytes, wait: float = 10) -> bytes:
        """Send bytes as they are; everything the server answers before closing."""
        with socket.create_connection(("127.0.0.1", self.port), timeout=wait) as c:
            c.sendall(data)
            out = b""
            while chunk := c.recv(65536):
                out += chunk
            return out

    def raw_get(self, path):
        """http.client sends the path exactly as given (no normalisation)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            c.putrequest("GET", path, skip_host=True)
            c.putheader("Host", f"127.0.0.1:{self.port}")
            c.endheaders()
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    # ---------- security ----------
    def test_api_needs_the_token(self):
        self.assertEqual(self.get_json("/api/status", token=False)[0], 401)
        self.assertEqual(self.get_json("/api/status", token="x" * 43)[0], 401)
        self.assertEqual(self.get_json("/api/jobs", token=self.token[:-1])[0], 401)
        self.assertEqual(self.post_json("/api/ingest", {"target": "AAPL"}, token=False)[0], 401)
        self.assertEqual(self.get_json("/api/status")[0], 200)

    def test_foreign_host_is_forbidden(self):
        for host in ("evil.example", f"evil.example:{self.port}", f"127.0.0.1:{self.port + 1}", "127.0.0.1",
                     f"127.0.0.2:{self.port}", f"rebind.attacker.test:{self.port}"):
            with self.subTest(host=host):
                self.assertEqual(self.req("/api/status", host=host)[0], 403)
                self.assertEqual(self.req("/", host=host, token=False)[0], 403)
        for host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}", f"LOCALHOST:{self.port}"):
            with self.subTest(host=host):
                self.assertEqual(self.req("/api/status", host=host)[0], 200)

    def test_page_injects_token_and_sets_security_headers(self):
        code, headers, body = self.req("/", token=False, headers={"Origin": "https://evil.example"})
        self.assertEqual(code, 200)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertIn(f'content="{self.token}"', body.decode())
        self.assertNotIn("{{TOKEN}}", body.decode())
        self.assertEqual(headers["Content-Security-Policy"], CSP)
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertFalse([k for k in headers if k.lower().startswith("access-control-")])

    def test_no_other_site_can_load_our_files(self):
        # A page elsewhere could otherwise probe for a running terminal with <script src> or <link>.
        for path, token in (("/static/app.js", False), ("/static/style.css", False), ("/", False),
                            ("/api/status", True)):
            with self.subTest(path=path):
                code, headers, _ = self.req(path, token=token)
                self.assertEqual(code, 200)
                self.assertEqual(headers["Cross-Origin-Resource-Policy"], "same-origin")
                self.assertEqual(headers["Cross-Origin-Opener-Policy"], "same-origin")

    def test_absolute_form_targets_are_refused_and_answered(self):
        for target in ("http://[x/api/status", f"http://evil.example:{self.port}/", "*"):
            with self.subTest(target=target):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    out = self.raw(f"GET {target} HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n\r\n".encode())
                self.assertTrue(out.startswith(b"HTTP/1.0 400"), out[:80])
                self.assertNotIn("Traceback", err.getvalue())

    def test_json_headers_and_no_cors(self):
        code, headers, _ = self.req("/api/status", headers={"Origin": "https://evil.example"})
        self.assertEqual(code, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertFalse([k for k in headers if k.lower().startswith("access-control-")])
        code, headers, _ = self.req("/api/status", method="OPTIONS", headers={"Origin": "https://evil.example"})
        self.assertNotEqual(code, 200)
        self.assertFalse([k for k in headers if k.lower().startswith("access-control-")])

    def test_static_allow_list(self):
        code, headers, body = self.req("/static/app.js", token=False)
        self.assertEqual((code, body), (200, b"console.log('app');"))
        self.assertTrue(headers["Content-Type"].startswith("text/javascript"))
        code, headers, _ = self.req("/static/style.css", token=False)
        self.assertEqual(code, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/css"))
        for path in ("/static/../x", "/static/secret.txt", "/static/index.html", "/static/%2e%2e/x",
                     "/static/app.js/", "/static//app.js", "/static/", "/x", "/index.html", "/../x"):
            with self.subTest(path=path):
                code, body = self.raw_get(path)
                self.assertEqual(code, 404)
                self.assertNotIn(b"outside static", body)
                self.assertNotIn(b"not served", body)

    def test_post_body_rules(self):
        self.assertEqual(self.post_json("/api/ingest", b"{not json")[0], 400)
        self.assertEqual(self.post_json("/api/ingest", b"[1, 2]")[0], 400)
        self.assertEqual(self.post_json("/api/ingest", b"\xff\xfe")[0], 400)
        self.assertEqual(self.post_json("/api/ingest", b"[" * 60000)[0], 400)
        code, _, raw = self.req("/api/ingest", method="POST", body=b'{"target": "AAPL"}',
                                headers={"Content-Type": "text/plain"})
        self.assertEqual(code, 400)
        code, out = self.post_json("/api/ask", json.dumps({"question": "x" * (server.MAX_BODY + 1)}).encode())
        self.assertEqual(code, 413)
        self.assertFalse(out["ok"])
        self.assertEqual(self.req("/api/ask", method="GET")[0], 405)
        self.assertEqual(self.post_json("/api/run", {})[0], 405)
        self.assertEqual(self.get_json("/api/nope")[0], 404)
        body = b'{"target": "AAPL"}'
        out = self.raw(f"POST /api/ingest HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nX-Terminal-Token: {self.token}\r\n"
                       f"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n".encode()
                       + f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n")
        self.assertTrue(out.startswith(b"HTTP/1.0 400"), out[:80])
        self.assertIn(b"not chunked", out)
        self.assertEqual(self.server.jobs.list_jobs(), [])

    def test_a_request_must_arrive_within_the_deadline(self):
        def drip(head: bytes, step: float, seconds: float = 8):
            """Send ``head``, then one more byte every ``step`` seconds until the server answers or closes;
            (what it sent, seconds taken)."""
            t0 = time.monotonic()
            with socket.create_connection(("127.0.0.1", self.port), timeout=10) as c:
                c.sendall(head)
                c.settimeout(step)
                while time.monotonic() - t0 < seconds:
                    try:
                        return c.recv(65536), time.monotonic() - t0
                    except TimeoutError:
                        try:
                            c.sendall(b" ")
                        except OSError:
                            break
                return None, time.monotonic() - t0
        start = (f"POST /api/ask HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nX-Terminal-Token: {self.token}\r\n"
                 f"Content-Type: application/json\r\n").encode()
        err = io.StringIO()
        with mock.patch.object(server, "REQUEST_SECONDS", 1.0), mock.patch.object(server.Handler, "timeout", 0.6), \
                contextlib.redirect_stderr(err):
            # a body byte every 0.3 s never trips the per-read timeout, but the whole request runs out of time
            out, took = drip(start + b"Content-Length: 1000\r\n\r\n{", 0.3)
            self.assertTrue(out and out.startswith(b"HTTP/1.0 408"), out and out[:80])
            self.assertLess(took, 4)
            # a body that stops: 408 too, and no traceback (it is the client's doing, not ours)
            out, took = drip(start + b"Content-Length: 1000\r\n\r\n{", 5)
            self.assertTrue(out and out.startswith(b"HTTP/1.0 408"), out and out[:80])
            # headers that trickle in: the connection is closed
            out, took = drip(start + b"X-Pad: ", 0.3)
            self.assertEqual(out, b"")
            self.assertLess(took, 4)
        self.assertNotIn("Traceback", err.getvalue())
        self.assertEqual(self.get_json("/api/status")[0], 200)  # the server is still fine

    def test_connections_beyond_the_cap_are_closed(self):
        with mock.patch.object(server, "MAX_CONNECTIONS", 2):
            srv, _ = server.make_server("127.0.0.1", 0, bank=self.bank)
        srv.quiet = True
        t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        t.start()
        try:
            port = srv.server_port
            idle = [socket.create_connection(("127.0.0.1", port), timeout=10) for _ in range(2)]
            try:
                self.assertTrue(wait_until(lambda: srv._connections._value == 0))  # two threads wait on them
                with socket.create_connection(("127.0.0.1", port), timeout=10) as extra:
                    self.assertEqual(extra.recv(100), b"")  # closed at once, no thread started
            finally:
                for c in idle:
                    c.close()
            self.assertTrue(wait_until(lambda: srv._connections._value == 2))  # slots come back
            with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
                c.sendall(f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode())
                self.assertTrue(c.recv(100).startswith(b"HTTP/1.0 200"))
        finally:
            srv.shutdown()
            srv.server_close()
            t.join(5)

    def test_internal_errors_hide_details(self):
        def boom(bank, cmd):
            raise RuntimeError("secret detail /home/user")
        err = io.StringIO()
        with mock.patch.object(data, "run", boom), contextlib.redirect_stderr(err):
            code, headers, body = self.req("/api/run?cmd=HELP")
        self.assertEqual(code, 500)
        self.assertEqual(json.loads(body), {"ok": False, "error": "internal error"})
        self.assertNotIn(b"secret", body)
        self.assertNotIn(b"Traceback", body)
        self.assertIn("Traceback", err.getvalue())
        self.assertIn("secret detail", err.getvalue())

    # ---------- routes ----------
    def test_run_dispatches_to_data_layer_under_the_bank(self):
        seen = []

        def fake_run(bank, cmd):
            seen.append((bank, cmd, self.server.lock._is_owned()))
            return {"ok": True, "fn": cmd.fn, "ticker": cmd.ticker, "args": cmd.args, "title": "t", "asof": None,
                    "view": "help", "data": {"x": float("nan"), "n": np.int64(3)}}
        with mock.patch.object(data, "run", fake_run):
            code, out = self.get_json("/api/run?cmd=HELP")
            self.assertEqual(self.get_json("/api/fact?ticker=AAPL&id=ab12")[0], 200)
        self.assertEqual(code, 200)
        self.assertTrue(out["ok"])
        self.assertEqual(out["data"], {"x": None, "n": 3})  # NaN never reaches the browser
        self.assertIs(seen[0][0], self.server.bank)
        self.assertEqual(seen[0][1].fn, "HELP")
        self.assertEqual([s[2] for s in seen], [True, True])  # the bank lock is held while the data layer runs

    def test_run_maps_command_errors(self):
        def bad(line):
            raise commands.CommandError("unknown function 'XYZ'", "XY: did you mean ...")
        with mock.patch.object(commands, "parse", bad):
            code, out = self.get_json("/api/run?cmd=XYZ")
        self.assertEqual(code, 200)
        self.assertEqual((out["ok"], out["error"], out["hint"]), (False, "unknown function 'XYZ'", "XY: did you mean ..."))
        code, out = self.get_json("/api/run?cmd=")
        self.assertEqual((code, out["ok"], out["hint"]), (200, False, "HELP"))

    def test_run_with_the_real_parser_and_data_layer(self):
        code, out = self.get_json("/api/run?cmd=HELP")
        self.assertEqual(code, 200)
        self.assertTrue(out["ok"])
        self.assertEqual(out["view"], "help")
        json.dumps(out, allow_nan=False)
        code, out = self.get_json("/api/run?cmd=AAPL%20US%20EQUITY%20XYZQ")
        self.assertEqual((code, out["ok"], out["error"]), (200, False, "unknown function 'XYZQ'"))
        self.assertTrue(out["hint"])

    def test_complete_passes_bank_entities(self):
        seen = []

        def fake_complete(prefix, tickers):
            seen.append((prefix, tickers))
            return [{"text": "AAPL", "kind": "ticker", "desc": ""}]
        with mock.patch.object(commands, "complete", fake_complete):
            code, out = self.get_json("/api/complete?q=aa")
        self.assertEqual(code, 200)
        self.assertEqual(out["items"], [{"text": "AAPL", "kind": "ticker", "desc": ""}])
        self.assertEqual(seen, [("aa", ["AAPL"])])

    def test_status_shape(self):
        code, out = self.get_json("/api/status")
        self.assertEqual(code, 200)
        for k in ("entities", "tickers", "universe", "jobs_running", "jobs_queued", "jobs_active", "jobs_finished",
                  "analyst", "time"):
            self.assertIn(k, out)
        self.assertEqual((out["entities"], out["tickers"], out["jobs_running"]), (1, ["AAPL"], 0))
        self.assertEqual((out["jobs_queued"], out["jobs_active"], out["jobs_finished"]), (0, [], 0))
        self.assertIsInstance(out["analyst"]["available"], bool)
        self.assertIsInstance(out["analyst"]["reason"], str)
        self.assertTrue(out["time"].endswith("Z"))

    def test_fact_runs_show(self):
        seen = []

        def fake_run(bank, cmd):
            seen.append(cmd)
            return {"ok": True, "fn": "SHOW", "ticker": cmd.ticker, "view": "fact", "data": {}}
        with mock.patch.object(data, "run", fake_run):
            self.assertEqual(self.get_json("/api/fact?ticker=CF&id=ab12cd34")[0], 200)
            self.assertEqual(self.get_json("/api/fact?ticker=AAPL&id=zz")[0], 400)
            self.assertEqual(self.get_json("/api/fact?ticker=AAPL")[0], 400)
            self.assertEqual(self.get_json("/api/fact?ticker=A%20B&id=ab")[0], 400)
        self.assertEqual(len(seen), 1)
        self.assertEqual((seen[0].fn, seen[0].ticker, [a.lower() for a in seen[0].args]), ("SHOW", "CF", ["ab12cd34"]))

    # ---------- analyst ----------
    def test_ask_without_analyst_module(self):
        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"brain.terminal.analyst": None}), contextlib.redirect_stderr(err):
            code, out = self.post_json("/api/ask", {"question": "how is AAPL?", "ticker": "AAPL", "history": []})
            self.assertEqual(code, 200)
            self.assertFalse(out["ok"])
            self.assertTrue(out["hint"])
            code, status = self.get_json("/api/status")
            self.assertEqual(code, 200)
            self.assertFalse(status["analyst"]["available"])
            self.assertTrue(status["analyst"]["reason"])

    def test_ask_reads_bank_read_only(self):
        fake = types.ModuleType("brain.terminal.analyst")
        got = {}

        def ask(bank_reader, question, ticker=None, history=None):
            got.update(question=question, ticker=ticker, history=history,
                       jobs=bank_reader("JOBS"), ingest=bank_reader("INGEST MSFT"))
            return {"ok": True, "answer": "fine [abcd1234]", "tool_calls": [], "model": "fake", "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1}}
        fake.ask, fake.available = ask, lambda: (True, "")
        with mock.patch.dict(sys.modules, {"brain.terminal.analyst": fake}):
            code, out = self.post_json("/api/ask", {"question": "q?", "ticker": "aapl",
                                                    "history": [{"role": "user", "content": "hi"}]})
            self.assertTrue(self.get_json("/api/status")[1]["analyst"]["available"])
        self.assertEqual((code, out["ok"], out["answer"]), (200, True, "fine [abcd1234]"))
        self.assertEqual((got["question"], got["ticker"], got["history"]), ("q?", "AAPL", [{"role": "user", "content": "hi"}]))
        self.assertTrue(got["jobs"]["ok"])
        self.assertFalse(got["ingest"]["ok"])  # the analyst's tools cannot start jobs
        self.assertEqual(self.server.jobs.list_jobs(), [])

    def test_ask_command_opens_the_pane_without_calling_the_model(self):
        fake = types.ModuleType("brain.terminal.analyst")
        fake.available = lambda: (False, "no Anthropic credentials")
        fake.ask = mock.Mock(side_effect=AssertionError("GET must not call the model"))
        with mock.patch.dict(sys.modules, {"brain.terminal.analyst": fake}):
            code, out = self.get_json("/api/run?cmd=ASK%20why%20is%20the%20curve%20flat%3F")
        self.assertEqual((code, out["ok"], out["fn"], out["view"]), (200, True, "ASK", "ask"))
        self.assertEqual(out["data"]["question"], "why is the curve flat?")
        self.assertEqual((out["data"]["available"], out["data"]["reason"]), (False, "no Anthropic credentials"))
        fake.ask.assert_not_called()

    def test_ask_keeps_recent_history_only(self):
        fake = types.ModuleType("brain.terminal.analyst")
        fake.ask = mock.Mock(return_value={"ok": True, "answer": "a"})
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(60)]
        with mock.patch.dict(sys.modules, {"brain.terminal.analyst": fake}):
            code, out = self.post_json("/api/ask", {"question": "q", "ticker": None, "history": history})
        self.assertEqual((code, out["ok"]), (200, True))
        sent = fake.ask.call_args.kwargs["history"]
        self.assertEqual((len(sent), sent[-1]["content"]), (server.MAX_HISTORY, "59"))

    def test_ask_validates_input(self):
        fake = types.ModuleType("brain.terminal.analyst")
        fake.ask = mock.Mock(return_value={"ok": True, "answer": "a"})
        with mock.patch.dict(sys.modules, {"brain.terminal.analyst": fake}):
            for body in ({}, {"question": ""}, {"question": 5}, {"question": "q", "ticker": "A B"},
                         {"question": "q", "history": "x"}, {"question": "q", "history": [{"role": "system", "content": "x"}]},
                         {"question": "q", "history": [{"role": "user", "content": 5}]},
                         {"question": "x" * (server.MAX_QUESTION + 1)}):
                with self.subTest(body=str(body)[:80]):
                    self.assertEqual(self.post_json("/api/ask", body)[0], 400)
            fake.ask.assert_not_called()
            self.assertEqual(self.post_json("/api/ask", {"question": "x" * server.MAX_QUESTION})[0], 200)

    def test_read_only_refuses_ingest_and_ask(self):
        for line in ("ASK why?", "INGEST AAPL", "AAPL DEC NEW", "DEC NEW AAPL"):
            with self.subTest(line=line):
                out = self.server.run_line(line, read_only=True)
                self.assertFalse(out["ok"])
                self.assertIn("read-only", out["error"])
        self.assertTrue(self.server.run_line("ASK why?")["ok"])
        self.assertEqual(self.server.jobs.list_jobs(), [])

    # ---------- v2: writes, alerts, alternative-data facts ----------
    def _alert(self, entity="AAPL", severity="high"):
        from brain.bank import Fact
        f = Fact(entity=entity, attr="alert", value=f"{entity} moved", as_of="2026-01-02",
                 known_at="2026-01-02T10:00:00Z", tier="derived", source={"name": "brain.watch", "rule": "price_move"},
                 context={"rule": "price_move", "severity": severity, "keys": [f"price:{entity}"]})
        self.bank.add(f)
        return f

    def test_watchlist_and_acks_change_only_from_the_command_line(self):
        from brain import watch
        code, out = self.get_json("/api/run?cmd=WL%20ADD%20aapl%20msft")
        self.assertEqual((code, out["ok"], out["fn"], out["args"]), (200, True, "WL", ["ADD", "AAPL", "MSFT"]))
        self.assertEqual(watch.load()["tickers"], ["AAPL", "MSFT"])
        f = self._alert()
        self.assertEqual(self.get_json("/api/status")[1]["alerts"], {"unacked": 1, "high": 1, "info": 0})
        for line in ("WL DEL AAPL", f"ALRT ACK {f.id[:8]}", "ALRT ACK ALL"):  # the analyst's reader
            with self.subTest(line=line):
                out = self.server.run_line(line, read_only=True)
                self.assertFalse(out["ok"])
                self.assertIn("is not available here (read-only)", out["error"])
        self.assertEqual(watch.load()["tickers"], ["AAPL", "MSFT"])
        self.assertTrue(self.server.run_line("ALRT", read_only=True)["ok"])  # reading is fine
        self.assertTrue(self.server.run_line("WL", read_only=True)["ok"])
        code, out = self.get_json(f"/api/run?cmd=ALRT%20ACK%20{f.id[:8]}")
        self.assertEqual((out["ok"], out["data"]["unacked"]), (True, 0))
        self.assertEqual(watch.acked(), {f.id})
        self.assertEqual(self.get_json("/api/status")[1]["alerts"], {"unacked": 0, "high": 0, "info": 0})
        out = self.get_json("/api/run?cmd=WL%20DEL%20MSFT")[1]
        self.assertTrue(out["ok"])
        self.assertEqual(watch.load()["tickers"], ["AAPL"])

    def test_writes_hold_the_bank_lock(self):
        seen = []

        def fake_run(bank, cmd, writes=False):
            seen.append((cmd.fn, writes, self.server.lock._is_owned()))
            return {"ok": True, "fn": cmd.fn, "ticker": None, "args": cmd.args, "title": "t", "asof": None,
                    "view": "table", "data": {}}
        with mock.patch.object(data, "run", fake_run):
            self.get_json("/api/run?cmd=WL%20ADD%20AAPL")
            self.get_json("/api/run?cmd=ALRT")
        self.assertEqual(seen, [("WL", True, True), ("ALRT", False, True)])

    def test_status_carries_alerts_and_the_model(self):
        with mock.patch.dict(os.environ, {"BRAIN_LLM_MODEL": "llama3.1:8b"}):
            out = self.get_json("/api/status")[1]
        self.assertEqual(out["alerts"], {"unacked": 0, "high": 0, "info": 0})
        self.assertEqual(out["analyst"]["model"], "llama3.1:8b")
        self._alert("MSFT", "notice")
        from brain import watch
        watch.save({"tickers": ["MSFT"]})
        self.assertEqual(self.get_json("/api/status")[1]["alerts"], {"unacked": 1, "high": 0, "info": 0})
        self._alert("MSFT", "info")  # a headline: counted apart, not in the number the top bar shows
        self.assertEqual(self.get_json("/api/status")[1]["alerts"], {"unacked": 1, "high": 0, "info": 1})
        watch.state_path().write_text("{broken")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, out = self.get_json("/api/status")  # a broken state file never breaks the status poll
            self.assertEqual(self.get_json("/api/status")[0], 200)
        self.assertEqual((code, out["alerts"]), (200, {"unacked": None, "high": None, "info": None}))
        self.assertEqual(err.getvalue().count("alert count unavailable"), 1)  # logged once, as one line
        self.assertNotIn("Traceback", err.getvalue())
        out = self.get_json("/api/run?cmd=ALRT")[1]
        self.assertFalse(out["ok"])
        self.assertIn("not valid JSON", out["error"])

    def test_files_of_the_wrong_shape_are_user_errors_not_500s(self):
        from brain import watch
        cases = [(watch.state_path(), "[]", ("ALRT", "WL", "ALRT ACK ALL", "WL ADD MSFT")),
                 (watch.state_path(), '{"acked": 5}', ("ALRT", "WL", "ALRT ACK ALL", "WL ADD MSFT")),
                 (watch.watchlist_path(), '{"tickers": ["AAPL"], "areas": 5}', ("WL", "WL ADD MSFT")),
                 (watch.watchlist_path(), '{"tickers": "AAPL"}', ("WL", "WL ADD MSFT"))]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            for path, text, lines in cases:
                path.write_text(text)
                for line in lines:
                    with self.subTest(text=text, line=line):
                        code, out = self.get_json("/api/run?cmd=" + line.replace(" ", "%20"))
                        self.assertEqual((code, out["ok"]), (200, False))
                        self.assertIn(str(path), out["error"])
                        self.assertIn("fix or remove it", out["error"])
                self.assertEqual(path.read_text(), text)  # never saved over
                code, out = self.get_json("/api/status")
                self.assertEqual(code, 200)
                self.assertTrue(self.get_json("/api/run?cmd=HELP")[1]["ok"])
                path.unlink()
        self.assertNotIn("Traceback", err.getvalue())

    def test_fact_for_ports_and_cameras(self):
        from brain.bank import Fact
        f = Fact(entity="AREA.LA-LONG-BEACH", attr="ships.count", value=12, as_of="2026-01-02",
                 known_at="2026-01-02T10:00:00Z", tier="reported", source={"name": "aisstream.io", "url": "https://aisstream.io/"})
        self.bank.add(f)
        code, out = self.get_json(f"/api/fact?ticker=AREA.LA-LONG-BEACH&id={f.id[:8]}")
        self.assertEqual((code, out["ok"], out["data"]["fact"]["id"]), (200, True, f.id))
        self.assertEqual(out["args"], [f"AREA.LA-LONG-BEACH/{f.id[:8].upper()}"])
        for bad in ("AREA.LA%20B", "AREA./x", "CAM.A%2FB", "AREA." + "X" * 101):
            with self.subTest(bad=bad):
                self.assertEqual(self.get_json(f"/api/fact?ticker={bad}&id=ab12")[0], 400)

    def test_ingest_insiders_and_watch(self):
        out = self.server.run_line("INGEST INSIDERS WATCH")
        self.assertTrue(out["ok"], out)
        self.assertTrue(self.server.jobs.wait(20))
        rows = self.get_json("/api/run?cmd=JOBS")[1]["data"]["rows"]
        self.assertEqual(sorted(r["target"] for r in rows), ["INSIDERS", "WATCH"])
        self.assertTrue(all("_ticker" not in r for r in rows))  # keywords, not tickers
        out = self.get_json("/api/run?cmd=INGEST")[1]
        self.assertIn("INSIDERS or WATCH", out["error"])

    # ---------- jobs ----------
    def test_ingest_target_validation(self):
        bad = ["AAPL; rm -rf /", "../etc", "A" * 16, "AAPL\n", "", " AAPL", "$(id)", "-h", "--all-concepts",
               "AA PL", "ÄPPLE", 5, None, ["AAPL"]]
        for t in bad:
            with self.subTest(target=t):
                code, out = self.post_json("/api/ingest", {"target": t})
                self.assertEqual(code, 400)
                self.assertFalse(out["ok"])
        self.assertEqual(self.server.jobs.list_jobs(), [])

    def test_ingest_runs_job_and_refreshes_bank(self):
        bank, refreshed = self.server.bank, []
        real = bank.refresh
        bank.refresh = lambda: refreshed.append(1) or real()
        code, out = self.post_json("/api/ingest", {"target": "brk.b"})
        self.assertEqual(code, 200)
        self.assertTrue(out["ok"])
        job = out["job"]
        self.assertEqual(set(job), {"id", "target", "kind", "status", "started", "finished", "returncode", "log"})
        self.assertEqual(job["kind"], "ingest")
        self.assertEqual(job["target"], "BRK.B")
        self.assertTrue(self.server.jobs.wait(20))
        code, out = self.get_json("/api/jobs")
        j = out["jobs"][0]
        self.assertEqual((j["status"], j["returncode"], j["log"]), ("done", 0, ["fake ingest BRK.B"]))
        self.assertIs(self.server.bank, bank)  # the same Bank, brought up to date in place
        self.assertEqual(refreshed, [1])

    def test_reload_reads_only_what_other_processes_appended(self):
        bank = self.server.bank
        self.bank.add(filed("MSFT", "us-gaap:Revenues", 2.0, "2024-06-30", "2024-07-30"))
        self.server.reload_bank()
        self.assertEqual(len(bank.facts("MSFT", "us-gaap:Revenues")), 1)
        self.assertEqual(bank.facts("ZZZ", "us-gaap:Revenues"), [])  # read once while empty
        aapl_size = bank.shard("AAPL").stat().st_size
        msft_indexed = bank._idx["MSFT"].size
        child = Bank(bank.root)  # a job writing from another process
        child.add(filed("MSFT", "us-gaap:Revenues", 3.0, "2024-09-30", "2024-10-30"))
        child.add(filed("NVDA", "us-gaap:Revenues", 4.0, "2024-07-28", "2024-08-28"))
        child.add(filed("ZZZ", "us-gaap:Revenues", 5.0, "2024-07-28", "2024-08-28"))
        scanned, real_scan = [], Bank._scan
        with mock.patch.object(Bank, "_scan", lambda b, e, ix, fh, st: scanned.append((e, ix.size)) or real_scan(b, e, ix, fh, st)):
            self.server.reload_bank()
        self.assertNotIn("AAPL", [e for e, _ in scanned])  # an unchanged shard is not read again
        self.assertEqual([at for e, at in scanned if e == "MSFT"], [msft_indexed])  # only the appended tail
        self.assertEqual(bank.shard("AAPL").stat().st_size, aapl_size)
        self.assertEqual(len(bank.facts("MSFT", "us-gaap:Revenues")), 2)
        self.assertEqual(len(bank.facts("ZZZ", "us-gaap:Revenues")), 1)  # not the empty read from before
        self.assertEqual(len(bank.facts("NVDA", "us-gaap:Revenues")), 1)
        full = Bank(bank.root)
        for e in full.entities():
            self.assertEqual((bank.digest(e), bank.attrs(e)), (full.digest(e), full.attrs(e)))
        self.assertEqual(self.server.tickers(), ["AAPL", "MSFT", "NVDA", "ZZZ"])

    def test_reloads_run_one_at_a_time_and_a_burst_shares_one(self):
        gate, started, calls = threading.Event(), threading.Event(), []
        bank = self.server.bank
        real_refresh = bank.refresh

        def slow_refresh():
            calls.append(1)
            started.set()
            self.assertTrue(gate.wait(10))
            return real_refresh()
        with mock.patch.object(bank, "refresh", slow_refresh):
            first = threading.Thread(target=self.server.reload_bank)
            first.start()
            self.assertTrue(started.wait(10))
            Bank(self.bank.root).add(filed("MSFT", "us-gaap:Revenues", 2.0, "2024-06-30", "2024-07-30"))
            burst = [threading.Thread(target=self.server.reload_bank) for _ in range(4)]
            for t in burst:
                t.start()
            self.assertTrue(wait_until(lambda: self.server._reloads_asked == 5))
            self.assertEqual(len(calls), 1)  # the burst waits: never two reloads at once
            gate.set()
            for t in [first, *burst]:
                t.join(10)
                self.assertFalse(t.is_alive())
        self.assertEqual(len(calls), 2)  # the first, then one for the whole burst
        self.assertEqual(len(self.server.bank.facts("MSFT")), 1)

    def test_a_job_counts_as_running_until_its_facts_are_visible(self):
        child = ("from brain.bank import Bank, Fact\n"
                 "Bank().add(Fact(entity='ZZTEST', attr='us-gaap:Revenues', value=5.0, as_of='2024-09-28', "
                 "known_at='2024-11-01', tier='filed', source={'name': 'test', 'url': 'https://example.test/r'}))\n"
                 "print('added')")
        self.server.jobs.argv_for = lambda t: py(child)
        self.assertEqual(self.get_json("/api/run?cmd=ZZTEST%20FA")[1]["ok"], False)  # caches ZZTEST as empty
        polls, bank = [], self.server.bank
        real_refresh = bank.refresh

        def slow_refresh():
            time.sleep(0.5)  # taking in a big job's output can take a moment
            return real_refresh()
        with mock.patch.object(bank, "refresh", slow_refresh):
            self.assertTrue(self.post_json("/api/ingest", {"target": "ZZTEST"})[1]["ok"])
            while True:  # what the page does: poll until the finished count rises, then refresh its panes
                status = self.get_json("/api/status")[1]
                polls.append(status["jobs_running"])
                if status["jobs_finished"]:
                    break
                self.assertEqual((status["jobs_running"], status["jobs_active"]), (1, ["ZZTEST"]))
                time.sleep(0.02)
            self.assertEqual((status["jobs_running"], status["jobs_active"], status["jobs_finished"]), (0, [], 1))
            with self.server.lock:
                self.assertEqual(len(self.server.bank.facts("ZZTEST")), 1)
            self.assertIn("ZZTEST", status["tickers"])
        self.assertEqual(self.get_json("/api/jobs")[1]["jobs"][0]["status"], "done")
        self.assertGreater(len(polls), 1)

    def test_jobs_listing_via_run(self):
        code, out = self.get_json("/api/run?cmd=JOBS")
        self.assertEqual((code, out["ok"], out["view"], out["data"]["rows"]), (200, True, "table", []))
        self.assertIn("INGEST", out["data"]["note"])
        self.assertTrue(self.get_json("/api/run?cmd=INGEST%20MARKETS")[1]["ok"])
        self.assertTrue(self.post_json("/api/ingest", {"target": "macro"})[1]["ok"])
        self.assertTrue(self.server.jobs.wait(20))
        code, out = self.get_json("/api/run?cmd=JOBS")
        self.assertEqual(out["view"], "table")
        self.assertEqual([c["key"] for c in out["data"]["columns"]],
                         ["id", "target", "status", "started", "finished", "last"])
        rows = out["data"]["rows"]
        self.assertEqual([r["target"] for r in rows], ["MACRO", "MARKETS"])  # newest first
        self.assertEqual([r["status"] for r in rows], ["done", "done"])
        self.assertEqual(rows[0]["last"], "fake ingest MACRO")
        self.assertEqual([j["target"] for j in self.get_json("/api/jobs")[1]["jobs"]], ["MACRO", "MARKETS"])
        code, out = self.get_json("/api/run?cmd=INGEST%20A%3BB")
        self.assertFalse(out["ok"])

    def test_ingest_queues_all_targets_or_none(self):
        many = " ".join(f"T{i}" for i in range(server.MAX_INGEST_TARGETS + 1))
        out = self.server.run_line(f"INGEST {many}")
        self.assertEqual((out["ok"], out["error"]), (False, f"at most {server.MAX_INGEST_TARGETS} targets at once"))
        out = self.server.run_line("INGEST AAPL MSFT A;B")
        self.assertFalse(out["ok"])
        self.assertEqual(self.server.jobs.list_jobs(), [])
        # No room for all of them: none is queued, rather than the first few with an error.
        real = self.server.jobs
        self.server.jobs = jobs.JobManager(max_running=0, argv_for=fake_argv)
        try:
            for i in range(jobs.MAX_QUEUED - 5):
                self.server.jobs.submit(f"Q{i}")
            out = self.server.run_line("INGEST " + " ".join(f"A{i}" for i in range(10)))
            self.assertEqual((out["ok"], out["hint"]), (False, "JOBS"))
            self.assertEqual(self.server.jobs.queued(), jobs.MAX_QUEUED - 5)
            out = self.server.run_line("INGEST A1 A2 A3 A4 A5 a5 Q0")  # duplicates and a queued target take no room
            self.assertTrue(out["ok"])
            self.assertEqual(self.server.jobs.queued(), jobs.MAX_QUEUED)
        finally:
            self.server.jobs.shutdown()
            self.server.jobs = real


class HostRulesTest(TempHome):
    def test_non_loopback_bind_accepts_ip_literals_only(self):
        srv, _ = server.make_server("0.0.0.0", 0)
        try:
            p = srv.server_port
            self.assertFalse(srv.loopback)
            self.assertTrue(srv.host_allowed(f"localhost:{p}"))
            self.assertTrue(srv.host_allowed(f"192.168.1.20:{p}"))
            self.assertFalse(srv.host_allowed(f"rebind.attacker.test:{p}"))
            self.assertFalse(srv.host_allowed(f"192.168.1.20:{p + 1}"))
        finally:
            srv.server_close()

    def test_the_printed_url_is_accepted(self):
        srv, _ = server.make_server("127.0.0.2", 0)
        try:
            p = srv.server_port
            self.assertTrue(srv.loopback)
            self.assertTrue(srv.host_allowed(f"127.0.0.2:{p}"))  # what serve() prints for --host 127.0.0.2
            self.assertTrue(srv.host_allowed(f"127.0.0.1:{p}"))
            self.assertFalse(srv.host_allowed(f"127.0.0.3:{p}"))
            self.assertFalse(srv.host_allowed(f"rebind.attacker.test:{p}"))
            srv.host = "::1"  # no IPv6 here to bind; the check reads only these
            self.assertTrue(srv.host_allowed(f"[0:0:0:0:0:0:0:1]:{p}"))
            self.assertTrue(srv.host_allowed(f"[::1]:{p}"))
            srv.host = "::ffff:127.0.0.1"
            self.assertTrue(srv.host_allowed(f"[::ffff:7f00:1]:{p}"))  # how a browser writes that address
            self.assertFalse(srv.host_allowed(f"[::ffff:7f00:2]:{p}"))
            srv.host, srv.server_port = "127.0.0.1", 80  # a browser leaves out the default port
            self.assertTrue(srv.host_allowed("127.0.0.1"))
            self.assertTrue(srv.host_allowed("localhost"))
            self.assertTrue(srv.host_allowed("127.0.0.1:80"))
            self.assertFalse(srv.host_allowed("127.0.0.2"))
            self.assertFalse(srv.host_allowed("evil.example"))
            self.assertFalse(srv.host_allowed("127.0.0.1:8080"))
        finally:
            srv.server_close()


class ServeTest(TempHome):
    def _serve(self, host, open_browser):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(server.TerminalServer, "serve_forever", side_effect=KeyboardInterrupt), \
                mock.patch("webbrowser.open") as wb, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            server.serve(host, 0, open_browser)
        return out.getvalue(), err.getvalue(), wb

    def test_serve_prints_url_and_stops_cleanly(self):
        out, err, wb = self._serve("127.0.0.1", True)
        self.assertIn("http://127.0.0.1:", out)
        self.assertNotIn("WARNING", err)
        wb.assert_called_once()
        self.assertTrue(wb.call_args[0][0].startswith("http://127.0.0.1:"))

    def test_serve_reports_a_busy_port(self):
        busy, _ = server.make_server("127.0.0.1", 0)
        try:
            with self.assertRaises(SystemExit) as cm, mock.patch("webbrowser.open") as wb:
                server.serve("127.0.0.1", busy.server_port, True)
            self.assertIn("cannot listen", str(cm.exception))
            wb.assert_not_called()
        finally:
            busy.server_close()

    def test_serve_warns_off_loopback_and_respects_no_browser(self):
        out, err, wb = self._serve("0.0.0.0", False)
        self.assertIn("WARNING", err)
        self.assertIn("no user accounts", err)
        self.assertIn("BRAIN_LLM_URL", err)  # ASK runs the configured model server
        self.assertIn("change the watchlist, acknowledge alerts", err)  # the two commands that write
        self.assertNotIn("Anthropic", err)
        wb.assert_not_called()

    def test_serve_says_which_jobs_outlive_it(self):
        gate = Path(self._tmp.name) / "go"
        started = []

        def run_then_stop(srv, *a, **kw):
            srv.jobs.argv_for = lambda t: py(f"import os, time\nprint('working')\n"
                                             f"while not os.path.exists({str(gate)!r}): time.sleep(0.02)")
            srv.jobs.submit("AAPL")
            started.append(srv)
            self.assertTrue(wait_until(lambda: srv.jobs.list_jobs()[0]["log"]))
            raise KeyboardInterrupt
        err = io.StringIO()
        with mock.patch.object(server.TerminalServer, "serve_forever", autospec=True, side_effect=run_then_stop), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            server.serve("127.0.0.1", 0, False)
        try:
            self.assertIn("INGEST AAPL is still running and will finish on its own; its output: ", err.getvalue())
            log = err.getvalue().split("its output: ")[1].split()[0]
            self.assertIn("working", Path(log).read_text())
        finally:
            gate.touch()
            self.assertTrue(started[0].jobs.wait(20))

    def test_out_of_range_port_is_a_message_not_a_traceback(self):
        for port in (70000, -1):
            with self.subTest(port=port), self.assertRaises(SystemExit) as cm:
                server.serve("127.0.0.1", port, False)
            self.assertIn("0-65535", str(cm.exception))
        err = io.StringIO()
        with mock.patch.object(server, "serve") as serve, contextlib.redirect_stderr(err), \
                self.assertRaises(SystemExit) as cm:
            cli.main(["terminal", "--port", "70000", "--no-browser"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("0-65535", err.getvalue())
        serve.assert_not_called()
        with mock.patch.object(server, "serve") as serve:
            cli.main(["terminal", "--port", "0", "--no-browser"])
        serve.assert_called_once_with("127.0.0.1", 0, False)


class JobManagerTest(TempHome):
    def test_validate_and_argv(self):
        self.assertEqual(jobs.validate("aapl"), "AAPL")
        self.assertEqual(jobs.validate("^gspc"), "^GSPC")
        self.assertEqual(jobs.validate("EURUSD=X"), "EURUSD=X")
        self.assertEqual(jobs.validate("markets"), "MARKETS")
        exe = sys.executable
        self.assertEqual(jobs.argv_for("AAPL"), [exe, "-m", "brain", "ingest", "AAPL"])
        self.assertEqual(jobs.argv_for("MARKETS"), [exe, "-m", "brain", "markets"])
        self.assertEqual(jobs.argv_for("MACRO"), [exe, "-m", "brain", "macro"])
        self.assertEqual(jobs.argv_for("UNIVERSE"), [exe, "-m", "brain", "ingest-universe"])
        self.assertEqual(jobs.validate("insiders"), "INSIDERS")
        self.assertEqual(jobs.argv_for("INSIDERS"), [exe, "-m", "brain", "insiders"])
        self.assertEqual(jobs.argv_for("WATCH"), [exe, "-m", "brain", "watch", "run", "--once"])
        for t in ("AAPL\n", "A/B", "A B", "x" * 16, "", "-X", "A;B", "A|B", "A&B", "`id`"):
            with self.subTest(t=t), self.assertRaises(jobs.JobError):
                jobs.validate(t)

    def test_at_most_two_run_and_the_rest_queue(self):
        gate = Path(self._tmp.name) / "go"
        wait = py(f"import os, time\nwhile not os.path.exists({str(gate)!r}): time.sleep(0.02)\nprint('released')")
        finished = []
        m = jobs.JobManager(on_finish=finished.append, argv_for=lambda t: wait)
        ids = [m.submit(t)["id"] for t in ("A", "B", "C")]
        self.assertEqual([m.get(i)["status"] for i in ids], ["running", "running", "queued"])
        self.assertEqual((m.running(), m.queued()), (2, 1))
        self.assertEqual(m.submit("a")["id"], ids[0])  # same target while running: the same job
        gate.touch()
        self.assertTrue(m.wait(20))
        self.assertEqual([m.get(i)["status"] for i in ids], ["done"] * 3)
        self.assertEqual(sorted(f["id"] for f in finished), sorted(ids))
        self.assertEqual([j["id"] for j in m.list_jobs()], ids[::-1])

    def test_log_keeps_last_200_lines_and_failures_are_recorded(self):
        m = jobs.JobManager(argv_for=lambda t: py("import sys\nfor i in range(250): print('line', i)\nsys.exit(3)"))
        jid = m.submit("X")["id"]
        self.assertTrue(m.wait(20))
        j = m.get(jid)
        self.assertEqual((j["status"], j["returncode"], len(j["log"])), ("failed", 3, 200))
        self.assertEqual((j["log"][0], j["log"][-1]), ("line 50", "line 249"))
        self.assertTrue(j["started"] <= j["finished"])

    def test_log_lines_as_a_text_pipe_would_read_them(self):
        code = ("import sys, time\n"
                "w = sys.stdout.buffer\n"
                "w.write(b'caf\\xc3'); w.flush(); time.sleep(0.3)\n"  # a character split across two reads
                "w.write(b'\\xa9\\r'); w.flush(); time.sleep(0.3)\n"  # and a \\r\\n split across two
                "w.write(b'\\nwin\\r\\nold mac\\rbad \\xff byte\\nno newline at the end')\n")
        m = jobs.JobManager(argv_for=lambda t: py(code))
        jid = m.submit("X")["id"]
        self.assertTrue(m.wait(20))
        self.assertEqual(m.get(jid)["log"], ["café", "win", "old mac", "bad \ufffd byte", "no newline at the end"])

    def test_job_that_cannot_start_fails_cleanly(self):
        m = jobs.JobManager(argv_for=lambda t: [str(Path(self._tmp.name) / "no-such-program")])
        jid = m.submit("X")["id"]
        self.assertTrue(m.wait(20))
        j = m.get(jid)
        self.assertEqual((j["status"], j["returncode"]), ("failed", None))
        self.assertIn("could not run job", j["log"][-1])

    def test_runs_without_a_shell_in_repo_root(self):
        m = jobs.JobManager(argv_for=lambda t: py("import os, sys\nprint(os.getcwd())\nprint(sys.argv[1:])") + [t])
        jid = m.submit("A=B")["id"]
        self.assertTrue(m.wait(20))
        log = m.get(jid)["log"]
        self.assertEqual(Path(log[0]).resolve(), Path(jobs.config.ROOT).resolve())
        self.assertEqual(log[1], "['A=B']")

    def test_job_reads_as_running_until_on_finish_returns(self):
        gate, seen = threading.Event(), []

        def on_finish(job):
            seen.append((job["status"], job["returncode"], m.get(job["id"])["status"], m.running()))
            self.assertTrue(gate.wait(10))
        m = jobs.JobManager(on_finish=on_finish, argv_for=fake_argv)
        jid = m.submit("A")["id"]
        self.assertTrue(wait_until(lambda: seen))
        self.assertEqual(seen, [("done", 0, "running", 1)])  # on_finish gets the outcome; clients still see running
        self.assertEqual((m.get(jid)["status"], m.get(jid)["finished"], m.running()), ("running", None, 1))
        gate.set()
        self.assertTrue(m.wait(20))
        j = m.get(jid)
        self.assertEqual((j["status"], j["returncode"], j["log"], m.running()), ("done", 0, ["fake ingest A"], 0))

    def test_submit_many_is_all_or_nothing(self):
        m = jobs.JobManager(max_running=0, argv_for=fake_argv)
        try:
            for i in range(jobs.MAX_QUEUED - 2):
                m.submit(f"Q{i}")
            with self.assertRaises(jobs.JobError):
                m.submit_many(["A", "B", "C"])
            with self.assertRaises(jobs.JobError):
                m.submit_many(["A", "B;"])
            self.assertEqual(m.queued(), jobs.MAX_QUEUED - 2)
            got = m.submit_many(["A", "b", "B", "Q0"])
            self.assertEqual([j["target"] for j in got], ["A", "B", "B", "Q0"])
            self.assertEqual(got[1]["id"], got[2]["id"])
            self.assertEqual(m.queued(), jobs.MAX_QUEUED)
            with self.assertRaises(jobs.JobError):
                m.submit("C")
            self.assertEqual(m.submit("q1")["target"], "Q1")  # already queued: no room needed
        finally:
            m.shutdown()

    def test_finished_jobs_beyond_the_limit_are_forgotten(self):
        m = jobs.JobManager(argv_for=lambda t: [str(Path(self._tmp.name) / "no-such-program")])
        for i in range(jobs.KEEP_JOBS + 5):
            m.submit(f"T{i}")
            self.assertTrue(m.wait(20))
        m.submit("LAST")
        self.assertTrue(m.wait(20))
        ids = [j["id"] for j in m.list_jobs()]
        self.assertEqual(len(ids), jobs.KEEP_JOBS)
        self.assertEqual((ids[0], ids[-1]), (f"j{jobs.KEEP_JOBS + 6}", "j7"))  # newest first; the oldest are gone

    def test_running_jobs_outlive_the_server(self):
        """A child must finish its ingest when the server stops: its output goes to a file, not a pipe that
        dies with the server, and it runs in its own session, out of reach of Ctrl+C in the server's terminal."""
        marker = Path(self._tmp.name) / "marker"
        child = ("import os, sys, time\n"
                 "for i in range(12):\n    print('step', i); time.sleep(0.1)\n"
                 f"open({str(marker)!r}, 'w').write(str(os.getsid(0)))\n")
        parent = ("import sys, time\n"
                  "from brain.terminal import jobs\n"
                  f"m = jobs.JobManager(argv_for=lambda t: [sys.executable, '-c', {child!r}])\n"
                  "m.submit('UNIVERSE')\n"
                  "time.sleep(0.4)\n"
                  "left = m.shutdown()\n"
                  "print(left[0][0], left[0][1])\n")
        out = subprocess.run([sys.executable, "-c", parent], cwd=jobs.config.ROOT, capture_output=True, text=True,
                             timeout=60, env={**os.environ, "TMPDIR": self._tmp.name})
        target, log = out.stdout.split()  # the server has exited here
        self.assertEqual(target, "UNIVERSE", out.stderr)
        self.assertTrue(wait_until(lambda: marker.exists() and marker.read_text()), "the child died with the server")
        self.assertNotEqual(int(marker.read_text()), os.getsid(0))
        self.assertEqual(Path(log).parent, Path(self._tmp.name))
        self.assertIn("step 11", Path(log).read_text())  # its output is kept for the user

    def test_shutdown_cancels_queued(self):
        gate = Path(self._tmp.name) / "go"
        wait = py(f"import os, time\nwhile not os.path.exists({str(gate)!r}): time.sleep(0.02)")
        m = jobs.JobManager(max_running=1, argv_for=lambda t: wait)
        a, b = m.submit("A")["id"], m.submit("B")["id"]
        m.shutdown()
        self.assertEqual(m.get(b)["status"], "failed")
        with self.assertRaises(jobs.JobError):
            m.submit("C")
        gate.touch()
        self.assertTrue(m.wait(20))
        self.assertEqual(m.get(a)["status"], "done")

    def test_table_shape(self):
        t = jobs.table([{"id": "j2", "target": "MARKETS", "status": "running", "started": "s", "finished": None,
                         "returncode": None, "log": ["a", "b"]},
                        {"id": "j1", "target": "AAPL", "status": "done", "started": "s", "finished": "f",
                         "returncode": 0, "log": []}])
        self.assertEqual([r["last"] for r in t["rows"]], ["b", ""])
        self.assertEqual(t["rows"][1]["_ticker"], "AAPL")
        self.assertNotIn("_ticker", t["rows"][0])
        self.assertEqual([c["type"] for c in t["columns"]], ["text", "text", "text", "date", "date", "text"])
        json.dumps(t, allow_nan=False)


class DumpsTest(unittest.TestCase):
    def test_non_finite_numbers_become_null(self):
        out = json.loads(server.dumps({"a": float("nan"), "b": [np.float64("inf"), np.int64(2)], "c": np.float32(1.5)}))
        self.assertEqual(out, {"a": None, "b": [None, 2], "c": 1.5})


if __name__ == "__main__":
    unittest.main()
