"""Terminal server and background jobs.

A real server runs on port 0 in a thread; requests go through urllib with an
explicit Host header (and no proxy). Jobs run a harmless child command injected
into the JobManager, so nothing here touches the network.
"""
import contextlib
import http.client
import io
import json
import sys
import threading
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

from brain.terminal import commands, data, jobs, server
from tests.util import TempHome, filed

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route 127.0.0.1 through a proxy
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'")


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def fake_argv(target: str) -> list[str]:
    return py(f"print('fake ingest {target}')")


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
            seen.append((bank, cmd))
            return {"ok": True, "fn": cmd.fn, "ticker": cmd.ticker, "args": cmd.args, "title": "t", "asof": None,
                    "view": "help", "data": {"x": float("nan"), "n": np.int64(3)}}
        with mock.patch.object(data, "run", fake_run):
            code, out = self.get_json("/api/run?cmd=HELP")
        self.assertEqual(code, 200)
        self.assertTrue(out["ok"])
        self.assertEqual(out["data"], {"x": None, "n": 3})  # NaN never reaches the browser
        self.assertIs(seen[0][0], self.server.bank)
        self.assertEqual(seen[0][1].fn, "HELP")

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
        for k in ("entities", "tickers", "universe", "jobs_running", "analyst", "time"):
            self.assertIn(k, out)
        self.assertEqual((out["entities"], out["tickers"], out["jobs_running"]), (1, ["AAPL"], 0))
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
        for body in ({}, {"question": ""}, {"question": 5}, {"question": "q", "ticker": "A B"},
                     {"question": "q", "history": "x"}, {"question": "q", "history": [{"role": "system", "content": "x"}]}):
            with self.subTest(body=body):
                self.assertEqual(self.post_json("/api/ask", body)[0], 400)

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

    def test_ingest_runs_job_and_swaps_bank(self):
        old = self.server.bank
        code, out = self.post_json("/api/ingest", {"target": "brk.b"})
        self.assertEqual(code, 200)
        self.assertTrue(out["ok"])
        job = out["job"]
        self.assertEqual(set(job), {"id", "target", "status", "started", "finished", "returncode", "log"})
        self.assertEqual(job["target"], "BRK.B")
        self.assertTrue(self.server.jobs.wait(20))
        code, out = self.get_json("/api/jobs")
        j = out["jobs"][0]
        self.assertEqual((j["status"], j["returncode"], j["log"]), ("done", 0, ["fake ingest BRK.B"]))
        self.assertIsNot(self.server.bank, old)
        self.assertEqual(self.server.bank.root, old.root)

    def test_reload_reads_the_whole_bank_and_never_goes_back(self):
        self.bank.add(filed("MSFT", "us-gaap:Revenues", 2.0, "2024-06-30", "2024-07-30"))
        self.server.reload_bank()
        fresh = self.server.bank
        self.assertIsNot(fresh, self.bank)
        self.assertEqual(set(fresh._facts), {"AAPL", "MSFT"})  # read before the swap, not on first request
        # A reload that started earlier but finishes later must not replace a newer bank.
        gate, started = threading.Event(), threading.Event()
        real_warm = server.warm

        def slow_warm(bank):
            started.set()
            gate.wait(10)
            return real_warm(bank)
        with mock.patch.object(server, "warm", slow_warm):
            t = threading.Thread(target=self.server.reload_bank)
            t.start()
            self.assertTrue(started.wait(10))
        self.server.reload_bank()  # starts later, finishes first
        newest = self.server.bank
        gate.set()
        t.join(10)
        self.assertIs(self.server.bank, newest)

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
        wb.assert_not_called()


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
