import functools
import json
import unittest

import pandas as pd

from brain import llm
from brain.terminal import analyst, server
from brain.timeutil import today
from tests.test_llm import FakeOpenAI, closed_port_url
from tests.util import TempHome, filed, price_snapshot


def tool_call(n, name="describe", args=None, raw=None):
    return {"id": f"call_{n}", "type": "function",
            "function": {"name": name, "arguments": raw if raw is not None else json.dumps(args if args is not None else {"ticker": "aapl"})}}


def calls_msg(*calls, content=None):
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def text_msg(text):
    return {"role": "assistant", "content": text}


def completion(msg, finish="stop", usage=None, model="served-model"):
    return {"id": "x", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 10}}


class AskCase(unittest.TestCase):
    def run_ask(self, reply, question="How did AAPL close?", history=None, reader=None, ticker="AAPL", **client_kw):
        """reply(n, body) -> the n-th response (see FakeOpenAI). Returns (result, request bodies, commands run)."""
        seen = []
        reader = reader or (lambda c: seen.append(c) or {"ok": True, "view": "des",
                                                         "data": {"price": {"close": 336.64, "cite": "9de710d2"}}})
        with FakeOpenAI(reply) as srv:
            res = analyst.ask(reader, question, ticker=ticker, history=history, client=srv.client(**client_kw))
        return res, srv.requests, seen


class AnalystLoopTest(AskCase):
    def test_tools_read_the_bank_through_terminal_commands(self):
        def reply(n, body):
            return calls_msg(tool_call(1)) if n == 1 else text_msg("AAPL closed at 336.64 [9de710d2].")
        res, sent, seen = self.run_ask(reply)
        self.assertTrue(res["ok"], res)
        self.assertEqual(seen, ["AAPL DES"])
        self.assertEqual(res["tool_calls"], [{"name": "describe", "input": "AAPL DES"}])
        self.assertEqual(res["answer"], "AAPL closed at 336.64 [9de710d2].")
        self.assertEqual((res["model"], res["stop_reason"], res["rounds"]), ("served-model", "stop", 2))
        self.assertEqual(res["citations"], {"verified": ["9de710d2"], "unverified": []})
        self.assertEqual(res["usage"], {"input_tokens": 2000, "output_tokens": 20, "cache_read_input_tokens": None,
                                        "requests": 2})
        first, second = sent
        self.assertEqual(first["model"], "fake-model")
        self.assertEqual(first["tool_choice"], "auto")
        self.assertEqual(len(first["tools"]), 13)
        self.assertEqual({t["type"] for t in first["tools"]}, {"function"})
        names = [t["function"]["name"] for t in first["tools"]]
        self.assertEqual(names[:3], ["describe", "price_history", "financials"])
        self.assertEqual(first["messages"][0]["role"], "system")
        for rule in ("cite", "[abcd1234]", "third-party text", "do not place orders", "FAIL", "INGEST"):
            self.assertIn(rule, first["messages"][0]["content"])
        self.assertTrue(first["messages"][-1]["content"].startswith("(Security in focus: AAPL)"))
        # The tool's result went back as a tool message answering the call's id.
        tool_msgs = [m for m in second["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["call_1"])
        self.assertEqual(json.loads(tool_msgs[0]["content"])["data"]["price"]["close"], 336.64)
        asst = [m for m in second["messages"] if m["role"] == "assistant"][-1]
        self.assertEqual(asst["tool_calls"][0]["function"], {"name": "describe", "arguments": '{"ticker": "aapl"}'})

    def test_tool_schemas_are_openai_functions(self):
        specs = {t.name: t.spec() for t in analyst._tools(lambda c: {}, [])}
        self.assertEqual(len(specs), 13)
        ph = specs["price_history"]["function"]["parameters"]
        self.assertEqual((ph["type"], ph["required"]), ("object", ["ticker"]))
        self.assertEqual(set(ph["properties"]), {"ticker", "period", "compare"})
        self.assertEqual(specs["research_results"]["function"]["parameters"]["properties"], {})
        self.assertEqual(specs["show_fact"]["function"]["parameters"]["required"], ["ticker", "fact_id"])

    def test_tool_calls_written_as_text_are_parsed(self):
        # Small models served without a tool parser write the call into the content.
        def reply(n, body):
            if n == 1:
                return text_msg('<tool_call>\n{{"name": "describe", "arguments": {"ticker": "AAPL"}}}\n</tool_call>')
            if n == 2:
                return text_msg('I will check the curve.\n```json\n{"name": "yield_curve", "arguments": {}}\n```')
            return text_msg("Done [9de710d2].")
        res, sent, seen = self.run_ask(reply)
        self.assertTrue(res["ok"], res)
        self.assertEqual(seen, ["AAPL DES", "CURVE"])
        asst = [m for m in sent[1]["messages"] if m["role"] == "assistant"][-1]
        self.assertEqual(asst["content"], "")  # the call is sent back structured, not twice
        self.assertEqual(asst["tool_calls"][0]["function"]["name"], "describe")
        self.assertEqual(res["answer"], "Done [9de710d2].")

    def test_json_naming_no_known_tool_is_an_answer(self):
        res, sent, seen = self.run_ask(lambda n, b: text_msg('The record is {"name": "Apple", "cik": 320193}.'))
        self.assertEqual((res["ok"], seen, len(sent)), (True, [], 1))
        self.assertIn("320193", res["answer"])

    def test_bad_arguments_and_unknown_tools_go_back_as_errors(self):
        def reply(n, body):
            if n == 1:
                return calls_msg(tool_call(1, raw='{"ticker": "AAPL"'),            # not JSON
                                 tool_call(2, raw='["AAPL"]'),                      # not an object
                                 tool_call(3, "buy_stock", {"ticker": "AAPL"}),     # no such tool
                                 tool_call(4, "describe", {"ticker": "AAPL DES; RES"}),
                                 tool_call(5, "show_fact", {"ticker": "AAPL", "fact_id": "zzzz"}),
                                 tool_call(6, "describe", {}))                      # missing argument
            return text_msg("I could not look that up.")
        res, sent, seen = self.run_ask(reply)
        self.assertTrue(res["ok"], res)
        self.assertEqual(seen, [])
        self.assertEqual(res["tool_calls"], [])
        out = {m["tool_call_id"]: json.loads(m["content"]) for m in sent[1]["messages"] if m["role"] == "tool"}
        self.assertEqual(len(out), 6)
        self.assertTrue(all(o["ok"] is False for o in out.values()))
        self.assertIn("not valid JSON", out["call_1"]["error"])
        self.assertIn("must be a JSON object", out["call_2"]["error"])
        self.assertIn("no tool named 'buy_stock'", out["call_3"]["error"])
        self.assertIn("not a ticker", out["call_4"]["error"])
        self.assertIn("not a fact id", out["call_5"]["error"])
        self.assertIn("describe needs ticker", out["call_6"]["error"])
        # A server would refuse its own invalid JSON echoed back: the history carries "{}" instead.
        asst = [m for m in sent[1]["messages"] if m["role"] == "assistant"][-1]
        self.assertEqual(asst["tool_calls"][0]["function"]["arguments"], "{}")

    def test_a_broken_text_tool_call_is_an_error_not_an_answer(self):
        def reply(n, body):
            if n == 1:
                return text_msg('<tool_call>\n{"name": "describe", "arguments": {"ticker": AAPL}}\n</tool_call>')
            return text_msg("Sorry.")
        res, sent, seen = self.run_ask(reply)
        self.assertEqual((res["answer"], seen), ("Sorry.", []))
        tool = [m for m in sent[1]["messages"] if m["role"] == "tool"][0]
        self.assertIn("not valid JSON", json.loads(tool["content"])["error"])

    def test_stopping_at_the_round_limit_is_not_an_answer(self):
        res, sent, seen = self.run_ask(lambda n, b: calls_msg(tool_call(n, "yield_curve", {}),
                                                              content=f"Let me also check the curve ({n})."))
        self.assertEqual(len(sent), analyst.MAX_ROUNDS)
        self.assertEqual(analyst.MAX_ROUNDS, 8)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["stop_reason"], "tool_calls")
        self.assertIn(f"stopped after {analyst.MAX_ROUNDS} tool rounds without a final answer", res["answer"])
        self.assertEqual(len(seen), analyst.MAX_ROUNDS - 1)  # the last round's calls are not run: nobody would read them
        self.assertEqual(res["usage"]["requests"], analyst.MAX_ROUNDS)
        self.assertEqual(res["usage"]["input_tokens"], 1000 * analyst.MAX_ROUNDS)
        # The last request tells the model to answer now.
        self.assertEqual(sent[-1]["messages"][-1], {"role": "user", "content": analyst.LAST_ROUND})
        self.assertNotIn(analyst.LAST_ROUND, json.dumps(sent[-2]["messages"]))

    def test_length_marks_the_answer_cut_off_and_empty_is_not_ok(self):
        res, _, _ = self.run_ask(lambda n, b: completion(text_msg("AAPL closed at"), finish="length"))
        self.assertTrue(res["ok"])
        self.assertTrue(res["answer"].startswith("AAPL closed at"))
        self.assertIn("cut off at the length limit", res["answer"])
        self.assertEqual(res["stop_reason"], "length")
        res, _, _ = self.run_ask(lambda n, b: text_msg("   "))
        self.assertEqual((res["ok"], res["error"]), (False, "the model returned an empty answer"))

    def test_thinking_is_not_part_of_the_answer(self):
        res, _, _ = self.run_ask(lambda n, b: text_msg("<think>The user wants {\"name\": \"describe\"}.</think>\n\nNo data."))
        self.assertEqual(res["answer"], "No data.")
        res, _, _ = self.run_ask(lambda n, b: text_msg("hidden reasoning</think>Answer [abcd1234]."))
        self.assertEqual(res["answer"], "Answer [abcd1234].")

    def test_server_errors_become_ok_false_results(self):
        cases = [(lambda n, b: (401, {"error": {"message": "bad key"}}), "the model server rejected the request", "BRAIN_LLM_KEY"),
                 (lambda n, b: (404, {"error": "model 'qwen3:8b' not found, try pulling it first"}),
                  "the model server does not know this model or path", "ollama pull"),
                 (lambda n, b: (500, {"error": {"message": "CUDA out of memory"}}), "model server error 500", "CUDA"),
                 (lambda n, b: (200, {"object": "chat.completion", "choices": []}), "the model server's answer was not chat-completions JSON", "/v1")]
        for reply, error, hint in cases:
            with self.subTest(error=error):
                res, _, seen = self.run_ask(reply)
                self.assertEqual((res["ok"], res["error"]), (False, error))
                self.assertIn(hint, res["hint"])
                self.assertEqual((seen, res["tool_calls"]), ([], []))

    def test_an_unreachable_server_names_the_setting_and_ollama(self):
        res = analyst.ask(lambda c: {}, "how is AAPL doing?", client=llm.Client(url=closed_port_url()))
        self.assertEqual((res["ok"], res["error"]), (False, "could not reach the model server"))
        self.assertIn("BRAIN_LLM_URL", res["hint"])
        self.assertIn("ollama", res["hint"])
        self.assertEqual(analyst.ask(lambda c: {}, "  ")["error"], "empty question")

    def test_available_asks_the_configured_server(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"BRAIN_LLM_URL": closed_port_url(), "BRAIN_LLM_MODEL": "qwen3:8b"}):
            ok, reason = analyst.available()
        self.assertFalse(ok)
        self.assertIn("BRAIN_LLM_URL", reason)
        self.assertIn("ollama pull qwen3:8b", reason)
        with FakeOpenAI(models=("qwen3:8b",)) as srv, \
                mock.patch.dict(os.environ, {"BRAIN_LLM_URL": srv.url, "BRAIN_LLM_MODEL": "qwen3:8b"}):
            self.assertEqual(analyst.available(), (True, ""))

    def test_an_error_after_tool_calls_keeps_them(self):
        def reply(n, body):
            return calls_msg(tool_call(1)) if n == 1 else (503, {"error": "loading model"})
        res, _, seen = self.run_ask(reply)
        self.assertFalse(res["ok"])
        self.assertEqual(res["tool_calls"], [{"name": "describe", "input": "AAPL DES"}])
        self.assertEqual(res["usage"]["requests"], 1)

    def test_history_reaches_the_request(self):
        hist = [{"role": "user", "content": "What about MSFT?"}, {"role": "assistant", "content": "MSFT closed at 500 [aaaa1111]."}]
        _, sent, _ = self.run_ask(lambda n, b: text_msg("ok"), history=hist)
        msgs = sent[0]["messages"]
        self.assertEqual([(m["role"], m["content"]) for m in msgs[1:3]], [(h["role"], h["content"]) for h in hist])
        self.assertEqual(msgs[3]["role"], "user")
        self.assertIn("How did AAPL close?", msgs[3]["content"])

    def test_a_context_overflow_is_retried_once_to_fit_what_the_server_said(self):
        big = {"ok": True, "view": "des", "data": {"blob": ["x" * 500] * 60, "cite": "9de710d2"}}

        def reply(n, body):
            if n == 1:
                return calls_msg(tool_call(1), tool_call(2, "financials", {"ticker": "AAPL"}))
            if n == 2:
                return (400, {"error": {"message": "Requested tokens (40000) exceed context window of 4096"}})
            return text_msg("fits now")
        res, sent, seen = self.run_ask(reply, reader=lambda c: big, ctx=200_000)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["answer"], "fits now")
        self.assertEqual(len(sent), 3)
        self.assertLess(len(json.dumps(sent[2]["messages"])), len(json.dumps(sent[1]["messages"])))
        self.assertLess(len(json.dumps(sent[2]["messages"])), 4096 * 4)


class FitTest(unittest.TestCase):
    def convo(self):
        hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"h{i} " + "y" * 3000} for i in range(6)]
        q = {"role": "user", "content": "the question"}
        r1 = [{"role": "assistant", "content": "", "tool_calls": [tool_call(1)]},
              {"role": "tool", "tool_call_id": "call_1", "content": "a" * 5000}]
        r2 = [{"role": "assistant", "content": "", "tool_calls": [tool_call(2)]},
              {"role": "tool", "tool_call_id": "call_2", "content": "b" * 5000}]
        return hist + [q] + r1 + r2, len(hist)

    def test_nothing_is_cut_when_it_fits(self):
        convo, n = self.convo()
        self.assertEqual(analyst.fit(convo, 10 ** 6, n), convo)

    def test_older_history_then_older_results_go_first(self):
        convo, n = self.convo()
        out = analyst.fit(convo, 12_000, n)
        self.assertEqual([m["content"][:2] for m in out if m["role"] != "tool"][:2], ["h4", "h5"])
        self.assertEqual(out[0]["role"], "user")
        tools = [m["content"] for m in out if m["role"] == "tool"]
        self.assertEqual(tools[0], analyst.ELIDED)
        self.assertEqual(tools[1], "b" * 5000)  # the latest round stays whole while it can
        self.assertIn({"role": "user", "content": "the question"}, out)
        self.assertEqual(convo[-1]["content"], "b" * 5000)  # the input is not changed

    def test_the_latest_results_are_cut_last(self):
        convo, n = self.convo()
        out = analyst.fit(convo, 3_000, n)
        self.assertEqual([m for m in out if m["content"].startswith("h")], [])
        latest = [m["content"] for m in out if m["role"] == "tool"][-1]
        self.assertTrue(latest.startswith("b" * 500))
        self.assertIn("characters cut to fit the context", latest)
        self.assertIn({"role": "user", "content": "the question"}, out)

    def test_budget_learns_the_servers_token_ratio(self):
        b = analyst.Budget(8192, fixed_chars=6000)
        before = b.chars()
        b.measured(20_000, 10_000)  # 2 characters per token: denser than the first guess
        self.assertEqual(b.ratio, 2.0)
        self.assertLess(b.chars(), before)
        b.measured(40_000, 10_000)  # never raised again
        self.assertEqual(b.ratio, 2.0)
        self.assertGreaterEqual(b.result_chars(), analyst.MIN_RESULT_CHARS)
        self.assertLessEqual(analyst.Budget(10 ** 6).result_chars(), analyst.MAX_RESULT_CHARS)


class ToolsTest(unittest.TestCase):
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
        seen, calls = [], []
        tools = {t.name: t for t in analyst._tools(lambda c: seen.append(c) or {}, calls)}
        for name, bad in [("describe", {"ticker": "AAPL DES; RES"}), ("show_fact", {"ticker": "AAPL", "fact_id": "../x"}),
                          ("show_fact", {"ticker": "AAPL", "fact_id": "zzzz"}), ("show_fact", {"ticker": "AAPL", "fact_id": "ÉÉÉÉ"}),
                          ("show_fact", {"ticker": "AAPL", "fact_id": "abc"}),
                          ("price_history", {"ticker": "AAPL", "compare": "SPY WAYTOOLONGSYMBOL123"}),
                          ("price_history", {"ticker": "AAPL", "period": "1Y SPY"}),
                          ("filings", {"ticker": "AAPL", "form": "10-K\nRES"}),
                          ("describe", {}), ("describe", "AAPL")]:
            with self.assertRaises(analyst.ToolError, msg=bad):
                tools[name].call(bad)
        self.assertEqual(seen, [])
        self.assertEqual(calls, [])
        tools["show_fact"].call({"ticker": "aapl", "fact_id": "[9DE710D2]"})
        tools["price_history"].call({"ticker": "aapl", "period": None, "compare": "spy qqq", "extra": 1})
        tools["filings"].call({"ticker": "aapl", "form": "8-k"})
        self.assertEqual(seen, ["AAPL SHOW 9de710d2", "AAPL GP 1Y SPY QQQ", "AAPL CF 8-K"])

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
        wide = {"ok": True, "fn": "DES", "ticker": "ZZ", "view": "des",
                "data": {f"k{i}": "y" * 500 for i in range(200)}}
        out = analyst._serialize(wide)
        self.assertLessEqual(len(out), analyst.MAX_RESULT_CHARS + 1000)
        back = json.loads(out)
        self.assertTrue(back["truncated"])
        self.assertIn("only the first", back["note"])
        self.assertEqual((back["fn"], back["ticker"]), ("DES", "ZZ"))
        # A table that size keeps whole rows and columns instead.
        rows = [{"desc": "y" * 500, "_cite": f"{i:08x}"} for i in range(analyst.MAX_TABLE_ROWS)]
        wide = {"ok": True, "fn": "CF", "ticker": "ZZ", "view": "table",
                "data": {"rows": rows, "columns": [{"key": f"c{i}", "label": "z" * 500} for i in range(200)]}}
        back = json.loads(analyst._serialize(wide))
        self.assertNotIn("truncated", back)
        self.assertEqual(back["data"]["rows"][0]["_cite"], "00000000")
        self.assertIn("cut to fit the model's context", back["data"]["note"])
        # A smaller context gives a smaller cap.
        small_cap = {t.name: t for t in analyst._tools(lambda c: wide, [], cap=lambda: 5000)}
        self.assertLessEqual(len(small_cap["filings"].call({"ticker": "zz"})), 5000)
        # Real-sized results are unchanged, but for big whole numbers, which get thousands separators.
        small = {"ok": True, "view": "des", "data": {"metrics": [{"key": "market_cap", "value": 4.95e12, "cite": "292044a9"},
                                                                 {"key": "pe", "value": 31.5, "cite": "292044aa"}]}}
        back = json.loads(analyst._serialize(small))
        self.assertEqual(back["data"]["metrics"][0]["value"], "4,950,000,000,000")
        back["data"]["metrics"][0]["value"] = 4.95e12
        self.assertEqual(back, small)

    def test_big_whole_numbers_are_written_for_reading(self):
        self.assertEqual(analyst._readable({"a": [112010000000, -93736000000, 7.46, 2025, 1_000_000, 1234567.5, True,
                                                  "112010000000", None]}),
                         {"a": ["112,010,000,000", "-93,736,000,000", 7.46, 2025, "1,000,000", 1234567.5, True,
                                "112010000000", None]})

    def test_bank_reader_exception_becomes_a_tool_error_payload(self):
        calls = []

        def reader(cmd):
            raise RuntimeError("bank on fire")
        tools = {t.name: t for t in analyst._tools(reader, calls)}
        out = json.loads(tools["describe"].call({"ticker": "AAPL"}))
        self.assertEqual(out, {"ok": False, "error": "RuntimeError: bank on fire"})
        self.assertEqual(calls, [{"name": "describe", "input": "AAPL DES"}])

    def test_citations_are_checked_against_the_tool_results(self):
        results = ['{"cite": "9de710d2"}', '{"_cites": {"2025": "4dd81b1a"}}']
        answer = "Net income [4DD81B1A]; close [9de710d2, 9de710d2]; revenue [deadbeef]; see [note] and [1]."
        self.assertEqual(analyst.check_citations(answer, results),
                         {"verified": ["4dd81b1a", "9de710d2"], "unverified": ["deadbeef"]})
        self.assertEqual(analyst.check_citations("no cites", results), {"verified": [], "unverified": []})

    def test_wide_tables_lose_columns_then_rows_not_whole_rows_of_text(self):
        cols = [{"key": "item", "label": "Item"}] + [{"key": f"y{i}", "label": f"y{i}"} for i in range(10)]
        rows = [{"item": f"row{r}", "_cites": {f"y{i}": f"{r:04x}{i:04x}" for i in range(10)},
                 **{f"y{i}": 123456789000 + i for i in range(10)}} for r in range(20)]
        res = {"ok": True, "fn": "FA", "view": "table", "data": {"columns": cols, "rows": rows, "note": "n"}}
        out = json.loads(analyst._serialize(res, 3000))
        data = out["data"]
        self.assertLessEqual(len(json.dumps(out)), 3000)
        self.assertEqual([c["key"] for c in data["columns"]], ["item", "y0", "y1", "y2"])
        self.assertEqual(set(data["rows"][0]), {"item", "_cites", "y0", "y1", "y2"})
        self.assertEqual(set(data["rows"][0]["_cites"]), {"y0", "y1", "y2"})
        self.assertIn("first 4 of 11 columns", data["note"])
        self.assertIn(f"of 20 rows", data["note"])
        self.assertEqual(data["rows"][0]["item"], "row0")

    def test_text_tool_call_shapes(self):
        known = {"describe", "yield_curve"}
        cases = [('<tool_call>\n{"name": "describe", "arguments": {"ticker": "AAPL"}}\n</tool_call>', ["describe"]),
                 ('{"type": "function", "function": {"name": "describe", "arguments": "{\\"ticker\\": \\"AAPL\\"}"}}', ["describe"]),
                 ('<|python_tag|>{"name": "yield_curve", "parameters": {}}', ["yield_curve"]),
                 ('{"name": "describe", "arguments": {"ticker": "A"}}\n{"name": "yield_curve"}', ["describe", "yield_curve"]),
                 ('Apple {"name": "Apple Inc."} reported [abcd1234].', []),
                 ("no json here", [])]
        for text, names in cases:
            with self.subTest(text=text):
                self.assertEqual([c["name"] for c in analyst.text_tool_calls(text, known)], names)


class ReaderTest(TempHome):
    """What the server's reader for the analyst may do to the bank: DES records derived metrics (as BRIEF
    does, so the ids it cites resolve), and nothing else is written; INGEST and ASK are refused."""

    def add_aapl(self):
        end = pd.Timestamp(today()) - pd.offsets.BDay(3)
        days = pd.bdate_range(end - pd.DateOffset(years=2), end).strftime("%Y-%m-%d").tolist()
        self.bank.add(price_snapshot("AAPL", days, [100.0 + 0.1 * i for i in range(len(days))]))
        last_fy = end.year - 1 if end >= pd.Timestamp(f"{end.year}-02-15") else end.year - 2
        for y in (last_fy - 1, last_fy):
            for attr, v in (("us-gaap:Revenues", 100e9), ("us-gaap:NetIncomeLoss", 25e9)):
                self.bank.add(filed("AAPL", attr, v * (1.1 if y == last_fy else 1), f"{y}-12-31", f"{y + 1}-02-10",
                                    form="10-K", start=f"{y}-01-01", duration_days=364))

    def test_the_analysts_reader_writes_only_derived_facts(self):
        self.add_aapl()
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

    def test_bank_reader_is_read_only_like_the_servers(self):
        self.add_aapl()
        read = analyst.bank_reader(self.bank)
        self.assertTrue(read("AAPL FA")["ok"])
        for line in ("INGEST MSFT", "ASK what now?", "JOBS", "", "NOPE NOPE NOPE"):
            self.assertFalse(read(line)["ok"], line)

    def test_a_whole_answer_against_a_bank(self):
        self.add_aapl()

        def reply(n, body):
            if n == 1:
                return calls_msg(tool_call(1, "financials", {"ticker": "AAPL"}))
            tool = [m for m in body["messages"] if m["role"] == "tool"][-1]
            cite = json.loads(tool["content"])["data"]["rows"][0].get("_cite") or "none"
            return text_msg(f"See [{cite}].")
        with FakeOpenAI(reply) as srv:
            res = analyst.ask(analyst.bank_reader(self.bank), "AAPL revenue?", client=srv.client())
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["tool_calls"], [{"name": "financials", "input": "AAPL FA"}])
        self.assertRegex(res["answer"], r"\[[0-9a-f]{8}\]|\[none\]")


if __name__ == "__main__":
    unittest.main()
