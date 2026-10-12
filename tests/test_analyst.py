import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from brain.terminal import analyst


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

    def test_bad_tool_input_never_reaches_the_bank(self):
        # The SDK's runner turns a tool exception into an is_error tool_result for Claude.
        seen, calls = [], []
        tools = {t.name: t for t in analyst._tools(lambda c: seen.append(c) or {}, calls)}
        for name, bad in [("describe", {"ticker": "AAPL DES; RES"}), ("show_fact", {"ticker": "AAPL", "fact_id": "../x"}),
                          ("price_history", {"ticker": "AAPL", "compare": "SPY WAYTOOLONGSYMBOL123"})]:
            with self.assertRaises(ValueError):
                tools[name].call(bad)
        self.assertEqual(seen, [])
        self.assertEqual(calls, [])
