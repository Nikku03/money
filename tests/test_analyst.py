import functools
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from brain.terminal import analyst, server
from brain.timeutil import today
from tests.util import TempHome, filed, price_snapshot


class FakeRunner:
    """Stands in for client.beta.messages.tool_runner: calls one tool, then answers."""

    def __init__(self, kw, tool_name, tool_input):
        self.kw, self.tool_name, self.tool_input = kw, tool_name, tool_input
        self.tool_output = None

    def __iter__(self):
        tool = next(t for t in self.kw["tools"] if t.name == self.tool_name)
        yield SimpleNamespace(stop_reason="tool_use", model=analyst.MODEL, usage=None,
                              content=[SimpleNamespace(type="tool_use", name=self.tool_name, input=self.tool_input)])
        self.tool_output = json.loads(tool.call(self.tool_input))
        yield SimpleNamespace(stop_reason="end_turn", model=analyst.MODEL,
                              usage=SimpleNamespace(input_tokens=1200, output_tokens=80),
                              content=[SimpleNamespace(type="text", text="AAPL closed at 336.64 [9de710d2].")])


class FakeClient:
    def __init__(self, tool_name="describe", tool_input=None):
        self.kw = None
        self.runner = None
        outer = self

        class Messages:
            def tool_runner(self, **kw):
                outer.kw = kw
                outer.runner = FakeRunner(kw, tool_name, tool_input or {"ticker": "aapl"})
                return outer.runner

        self.beta = SimpleNamespace(messages=Messages())


class AnalystTest(unittest.TestCase):
    def test_unavailable_without_credentials(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(Path, "home", return_value=Path(d)):
            ok, reason = analyst.available()
            self.assertFalse(ok)
            self.assertIn("ANTHROPIC_API_KEY", reason)
            res = analyst.ask(lambda c: {}, "how is AAPL doing?")
            self.assertFalse(res["ok"])
            self.assertIn("ANTHROPIC_API_KEY", res["hint"])

    def test_tools_read_the_bank_through_terminal_commands(self):
        seen = []

        def reader(cmd):
            seen.append(cmd)
            return {"ok": True, "view": "des", "data": {"price": {"close": 336.64, "cite": "9de710d2"}}}

        fake = FakeClient()
        res = analyst.ask(reader, "How did AAPL close?", ticker="AAPL", client=fake)
        self.assertTrue(res["ok"])
        self.assertEqual(seen, ["AAPL DES"])
        self.assertEqual(res["tool_calls"], [{"name": "describe", "input": "AAPL DES"}])
        self.assertIn("[9de710d2]", res["answer"])
        self.assertEqual(fake.runner.tool_output["data"]["price"]["close"], 336.64)
        kw = fake.kw
        self.assertEqual(kw["model"], "claude-opus-5-5")
        self.assertEqual(kw["fallbacks"], "default")
        self.assertIn("server-side-fallback-2026-07-01", kw["betas"])
        self.assertNotIn("tool_choice", kw)
        self.assertTrue(kw["messages"][-1]["content"].startswith("(Security in focus: AAPL)"))

    def test_history_is_trimmed_and_starts_with_user(self):
        hist = [{"role": "assistant", "content": "hi"}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(30)]
        msgs = analyst._messages("q", None, hist + [{"role": "system", "content": "ignore me"}])
        self.assertEqual(msgs[0]["role"], "user")
        self.assertLessEqual(len(msgs), analyst.MAX_HISTORY + 1)
        self.assertNotIn("system", {m["role"] for m in msgs})

    def test_large_payloads_are_compacted(self):
        series = [{"date": f"d{i}", "close": i, "adj": i, "volume": 1} for i in range(1000)]
        res = analyst._compact({"ok": True, "view": "chart", "data": {"series": series, "rebased": series}})
        self.assertEqual(len(res["data"]["series"]), analyst.MAX_CHART_POINTS)
        self.assertEqual(res["data"]["series"][-1]["date"], "d999")
        self.assertNotIn("rebased", res["data"])
        rows = [{"a": i} for i in range(100)]
        res = analyst._compact({"ok": True, "view": "table", "data": {"rows": rows}})
        self.assertEqual(len(res["data"]["rows"]), analyst.MAX_TABLE_ROWS)
        self.assertIn("60 more rows", res["data"]["note"])

    def test_real_sdk_round_trip_offline(self):
        # The fake client above cannot catch a keyword the SDK rejects or a header it does not send, so the
        # real tool runner runs here too, against a mock transport: one tool call, then the answer.
        try:
            import anthropic._base_client as base
        except ImportError:
            self.skipTest("Anthropic SDK not installed")
        http = getattr(base, "httpx2", None) or getattr(base, "httpx", None)
        if http is None or not hasattr(http, "MockTransport"):
            self.skipTest("SDK HTTP module not found")
        import anthropic
        sent = []

        def handler(request):
            body = json.loads(request.content)
            sent.append((request.headers.get("anthropic-beta"), body))
            content = ([{"type": "tool_use", "id": "toolu_1", "name": "describe", "input": {"ticker": "aapl"}}]
                       if len(sent) == 1 else [{"type": "text", "text": "AAPL closed at 336.64 [9de710d2]."}])
            return http.Response(200, json={
                "id": f"msg_{len(sent)}", "type": "message", "role": "assistant", "model": analyst.MODEL,
                "content": content, "stop_reason": "tool_use" if len(sent) == 1 else "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}})

        client = anthropic.Anthropic(api_key="test", http_client=http.Client(transport=http.MockTransport(handler)))
        seen = []
        res = analyst.ask(lambda c: seen.append(c) or {"ok": True, "view": "des", "data": {}}, "How did AAPL close?",
                          ticker="AAPL", client=client)
        self.assertTrue(res["ok"], res)
        self.assertEqual((seen, res["answer"]), (["AAPL DES"], "AAPL closed at 336.64 [9de710d2]."))
        self.assertEqual(len(sent), 2)
        beta, body = sent[0]
        self.assertIn("server-side-fallback-2026-07-01", beta)
        self.assertEqual((body["model"], body["fallbacks"], body["output_config"]), (analyst.MODEL, "default", {"effort": "high"}))
        self.assertEqual(len(body["tools"]), 13)
        tool_results = [b for b in sent[1][1]["messages"][-1]["content"] if b["type"] == "tool_result"]
        self.assertEqual(len(tool_results), 1)

    def test_bad_tool_input_never_reaches_the_bank(self):
        # ToolError: the SDK's runner hands the message to Claude as an is_error tool_result (see the
        # mock-transport test below), without logging a traceback as it does for other exceptions.
        from anthropic.lib.tools import ToolError
        seen, calls = [], []
        tools = {t.name: t for t in analyst._tools(lambda c: seen.append(c) or {}, calls)}
        for name, bad in [("describe", {"ticker": "AAPL DES; RES"}), ("show_fact", {"ticker": "AAPL", "fact_id": "../x"}),
                          ("show_fact", {"ticker": "AAPL", "fact_id": "zzzz"}), ("show_fact", {"ticker": "AAPL", "fact_id": "ÉÉÉÉ"}),
                          ("show_fact", {"ticker": "AAPL", "fact_id": "abc"}),
                          ("price_history", {"ticker": "AAPL", "compare": "SPY WAYTOOLONGSYMBOL123"})]:
            with self.assertRaises(ToolError, msg=bad):
                tools[name].call(bad)
        self.assertEqual(seen, [])
        self.assertEqual(calls, [])
        tools["show_fact"].call({"ticker": "aapl", "fact_id": "[9DE710D2]"})
        self.assertEqual(seen, ["AAPL SHOW 9de710d2"])

    def test_tool_results_are_capped_in_size(self):
        # One long stored headline or claim used to go to the model whole, on every news call.
        huge = "x" * 200_000
        res = {"ok": True, "fn": "CN", "ticker": "ZZ", "view": "news",
               "data": {"headlines": [{"title": huge, "cite": "abcd1234"}],
                        "claims": [{"text": huge, "source": huge, "cite": "abcd1235"}]}}
        tools = {t.name: t for t in analyst._tools(lambda c: res, [])}
        out = tools["news"].call({"ticker": "zz"})
        self.assertLess(len(out), 3000)
        back = json.loads(out)
        self.assertEqual(back["data"]["headlines"][0]["cite"], "abcd1234")
        self.assertTrue(back["data"]["headlines"][0]["title"].endswith(f"[{200_000 - analyst.MAX_TEXT} more characters cut]"))
        self.assertTrue(back["data"]["claims"][0]["text"].startswith("x" * analyst.MAX_TEXT))
        # Many fields, each under the per-string limit, are cut as a whole and say so.
        rows = [{"desc": "y" * 500, "_cite": f"{i:08x}"} for i in range(analyst.MAX_TABLE_ROWS)]
        wide = {"ok": True, "fn": "CF", "ticker": "ZZ", "view": "table",
                "data": {"rows": rows, "columns": [{"key": f"c{i}", "label": "z" * 500} for i in range(200)]}}
        out = analyst._serialize(wide)
        self.assertLessEqual(len(out), analyst.MAX_RESULT_CHARS + 1000)
        back = json.loads(out)
        self.assertTrue(back["truncated"])
        self.assertIn("only the first", back["note"])
        self.assertEqual((back["fn"], back["ticker"]), ("CF", "ZZ"))
        # Real-sized results are unchanged.
        small = {"ok": True, "view": "des", "data": {"metrics": [{"key": "market_cap", "value": 4.95e12, "cite": "292044a9"}]}}
        self.assertEqual(json.loads(analyst._serialize(small)), small)

    def test_bank_reader_exception_becomes_a_tool_error_payload(self):
        calls = []

        def reader(cmd):
            raise RuntimeError("bank on fire")
        tools = {t.name: t for t in analyst._tools(reader, calls)}
        out = json.loads(tools["describe"].call({"ticker": "AAPL"}))
        self.assertEqual(out, {"ok": False, "error": "RuntimeError: bank on fire"})
        self.assertEqual(calls, [{"name": "describe", "input": "AAPL DES"}])


def _http():
    try:
        import anthropic._base_client as base
    except ImportError:
        return None
    http = getattr(base, "httpx2", None) or getattr(base, "httpx", None)
    return http if http is not None and hasattr(http, "MockTransport") else None


def _msg(n, content, stop="end_turn", usage=None, **extra):
    return {"id": f"msg_{n}", "type": "message", "role": "assistant", "model": analyst.MODEL, "content": content,
            "stop_reason": stop, "stop_sequence": None, "usage": usage or {"input_tokens": 10, "output_tokens": 5}, **extra}


def _tool_use(n, name="describe", inp=None):
    return {"type": "tool_use", "id": f"toolu_{n}", "name": name, "input": inp if inp is not None else {"ticker": "aapl"}}


class RealRunnerTest(unittest.TestCase):
    """The SDK's own tool runner against a mock transport: what each kind of response becomes."""

    def setUp(self):
        self.http = _http()
        if self.http is None:
            self.skipTest("Anthropic SDK (with its HTTP module) not installed")
        import anthropic
        self.anthropic = anthropic

    def run_ask(self, reply, question="How did AAPL close?", history=None, reader=None):
        """reply(n, body) -> a message dict, (status, error dict), or an exception for the n-th request."""
        http, sent, seen = self.http, [], []

        def handler(request):
            body = json.loads(request.content)
            sent.append(body)
            r = reply(len(sent), body)
            if isinstance(r, Exception):
                raise r
            status, payload = r if isinstance(r, tuple) else (200, r)
            return http.Response(status, json=payload)

        client = self.anthropic.Anthropic(api_key="test", max_retries=0,
                                          http_client=http.Client(transport=http.MockTransport(handler)))
        res = analyst.ask(reader or (lambda c: seen.append(c) or {"ok": True, "view": "des", "data": {}}),
                          question, ticker="AAPL", history=history, client=client)
        return res, sent, seen

    def test_stopping_at_the_tool_round_limit_is_not_an_answer(self):
        # Text next to the last tool call is a preamble ("Let me also check..."), not the answer.
        res, sent, _ = self.run_ask(lambda n, b: _msg(n, [{"type": "text", "text": f"Let me also check the curve ({n})."},
                                                             _tool_use(n, "yield_curve", {})], stop="tool_use",
                                                         usage={"input_tokens": 100 * n, "output_tokens": 5}))
        self.assertEqual(len(sent), analyst.MAX_ITERATIONS)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["stop_reason"], "tool_use")
        self.assertIn(f"stopped after {analyst.MAX_ITERATIONS} tool rounds without a final answer", res["answer"])
        self.assertEqual(res["usage"]["input_tokens"], sum(100 * n for n in range(1, analyst.MAX_ITERATIONS + 1)))
        self.assertEqual(res["usage"]["output_tokens"], 5 * analyst.MAX_ITERATIONS)
        self.assertEqual(res["usage"]["requests"], analyst.MAX_ITERATIONS)

    def test_refusal_is_not_ok_and_keeps_the_explanation_and_tool_calls(self):
        def reply(n, body):
            if n == 1:
                return _msg(n, [_tool_use(n)], stop="tool_use")
            return _msg(n, [], stop="refusal", stop_details={"type": "refusal", "category": "cyber",
                                                              "explanation": "This request was declined."})
        res, _, _ = self.run_ask(reply)
        self.assertFalse(res["ok"])
        self.assertEqual(res["hint"], "This request was declined.")
        self.assertEqual(res["tool_calls"], [{"name": "describe", "input": "AAPL DES"}])

    def test_max_tokens_marks_the_answer_cut_off(self):
        res, _, _ = self.run_ask(lambda n, b: _msg(n, [{"type": "text", "text": "AAPL closed at"}], stop="max_tokens"))
        self.assertTrue(res["ok"])
        self.assertTrue(res["answer"].startswith("AAPL closed at"))
        self.assertIn("cut off at the length limit", res["answer"])

    def test_api_errors_become_ok_false_results(self):
        def err(status, kind):
            return lambda n, b: (status, {"type": "error", "error": {"type": kind, "message": f"{kind} happened"}})
        cases = [(err(401, "authentication_error"), "Anthropic rejected the credentials"),
                 (err(429, "rate_limit_error"), "rate limited by the Anthropic API"),
                 (err(500, "api_error"), "Anthropic API error 500"),
                 (err(529, "overloaded_error"), "Anthropic API error 529"),
                 (lambda n, b: self.http.ConnectError("connection refused"), "could not reach the Anthropic API")]
        for reply, error in cases:
            with self.subTest(error=error):
                res, _, seen = self.run_ask(reply)
                self.assertEqual((res["ok"], res["error"]), (False, error))
                self.assertIn("hint", res)
                self.assertEqual(seen, [])

    def test_rejected_tool_input_reaches_claude_as_an_error_without_a_traceback(self):
        def reply(n, body):
            if n == 1:
                return _msg(n, [_tool_use(1, "describe", {"ticker": "AAPL DES; RES"}),
                                _tool_use(2, "show_fact", {"ticker": "AAPL", "fact_id": "zzzz"})], stop="tool_use")
            return _msg(n, [{"type": "text", "text": "I could not look that up."}])
        with self.assertNoLogs("anthropic", level="ERROR"):
            res, sent, seen = self.run_ask(reply)
        self.assertTrue(res["ok"], res)
        self.assertEqual(seen, [])
        results = [b for b in sent[1]["messages"][-1]["content"] if b["type"] == "tool_result"]
        self.assertEqual([r.get("is_error") for r in results], [True, True])
        self.assertIn("not a ticker", json.dumps(results[0]["content"]))
        self.assertIn("not a fact id", json.dumps(results[1]["content"]))

    def test_history_reaches_the_request(self):
        hist = [{"role": "user", "content": "What about MSFT?"}, {"role": "assistant", "content": "MSFT closed at 500 [aaaa1111]."}]
        _, sent, _ = self.run_ask(lambda n, b: _msg(n, [{"type": "text", "text": "ok"}]), history=hist)
        msgs = sent[0]["messages"]
        self.assertEqual([(m["role"], m["content"]) for m in msgs[:2]], [(h["role"], h["content"]) for h in hist])
        self.assertEqual(msgs[2]["role"], "user")
        self.assertIn("How did AAPL close?", json.dumps(msgs[2]["content"]))

    def test_usage_counts_every_request_cache_reads_and_fallback_attempts(self):
        def reply(n, body):
            if n == 1:
                return _msg(n, [_tool_use(n)], stop="tool_use",
                            usage={"input_tokens": 100, "output_tokens": 5, "cache_creation_input_tokens": 2000,
                                   "cache_read_input_tokens": 0})
            # The first model declined, a fallback served: top-level usage is the serving attempt only.
            it = [{"type": "message", "input_tokens": 50, "output_tokens": 3, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 2000},
                  {"type": "fallback_message", "model": "claude-opus-5", "input_tokens": 60, "output_tokens": 7,
                   "cache_creation_input_tokens": 2100, "cache_read_input_tokens": 0}]
            return _msg(n, [{"type": "text", "text": "done"}],
                        usage={"input_tokens": 60, "output_tokens": 7, "cache_creation_input_tokens": 2100,
                               "cache_read_input_tokens": 0, "iterations": it})
        res, sent, _ = self.run_ask(reply)
        self.assertEqual(len(sent), 2)
        self.assertEqual(res["usage"], {"input_tokens": 100 + 2000 + 50 + 2000 + 60 + 2100, "output_tokens": 15,
                                        "cache_read_input_tokens": 2000, "requests": 2})


class CredentialsTest(unittest.TestCase):
    def env(self, home, **env):
        return [mock.patch.dict(os.environ, env, clear=True), mock.patch.object(Path, "home", return_value=Path(home))]

    def check(self, home, expect_ok, **env):
        p = self.env(home, **env)
        with p[0], p[1]:
            ok, reason = analyst.available()
            res = analyst.ask(lambda c: {}, "how is AAPL doing?")  # must not raise
        self.assertEqual(ok, expect_ok, reason)
        return reason, res

    def test_an_empty_config_directory_is_not_credentials(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".config" / "anthropic").mkdir(parents=True)
            reason, res = self.check(d, False)
        self.assertIn("ANTHROPIC_API_KEY", reason)
        self.assertEqual((res["ok"], res["error"]), (False, "analyst unavailable"))
        self.assertIn("ANTHROPIC_API_KEY", res["hint"])

    def test_a_missing_named_profile_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            reason, res = self.check(d, False, ANTHROPIC_PROFILE="work")
        self.assertIn("no usable Anthropic credentials", reason)
        self.assertFalse(res["ok"])
        self.assertIn("work", res["hint"])

    def test_an_identity_token_file_alone_is_not_credentials(self):
        with tempfile.TemporaryDirectory() as d:
            self.check(d, False, ANTHROPIC_IDENTITY_TOKEN_FILE=str(Path(d) / "token"))

    def test_an_api_key_is_available(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.env(d, ANTHROPIC_API_KEY="sk-test")
            with p[0], p[1]:
                self.assertEqual(analyst.available(), (True, ""))


class ReaderTest(TempHome):
    """What the server's reader for the analyst may do to the bank: DES records derived metrics (as BRIEF
    does, so the ids it cites resolve), and nothing else is written; INGEST and ASK are refused."""

    def test_the_analysts_reader_writes_only_derived_facts(self):
        end = pd.Timestamp(today()) - pd.offsets.BDay(3)
        days = pd.bdate_range(end - pd.DateOffset(years=2), end).strftime("%Y-%m-%d").tolist()
        self.bank.add(price_snapshot("AAPL", days, [100.0 + 0.1 * i for i in range(len(days))]))
        last_fy = end.year - 1 if end >= pd.Timestamp(f"{end.year}-02-15") else end.year - 2
        for y in (last_fy - 1, last_fy):
            for attr, v in (("us-gaap:Revenues", 100e9), ("us-gaap:NetIncomeLoss", 25e9)):
                self.bank.add(filed("AAPL", attr, v * (1.1 if y == last_fy else 1), f"{y}-12-31", f"{y + 1}-02-10",
                                    form="10-K", start=f"{y}-01-01", duration_days=364))
        shard = self.bank.shard("AAPL")
        before = shard.read_text().splitlines()
        srv, _ = server.make_server("127.0.0.1", 0, bank=self.bank)
        try:
            reader = functools.partial(srv.run_line, read_only=True)  # what /api/ask passes to analyst.ask
            self.assertTrue(reader("AAPL DES")["ok"])
            self.assertFalse(reader("ASK what now?")["ok"])
            self.assertFalse(reader("INGEST MSFT")["ok"])
            self.assertEqual(srv.jobs.list_jobs(), [])
        finally:
            srv.server_close()
        added = [json.loads(line) for line in shard.read_text().splitlines()[len(before):]]
        self.assertTrue(added, "DES records the derived metrics it shows")
        self.assertTrue(all(r["op"] == "add" and r["fact"]["tier"] == "derived" and r["fact"]["attr"].startswith("derived.")
                            for r in added), added)
        self.assertEqual(sorted(p.name for p in self.bank.root.glob("*.jsonl")), ["AAPL.jsonl"])
