import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from brain import llm


class FakeOpenAI:
    """A local OpenAI-compatible server. ``reply(n, body)`` gives the n-th chat response: a message dict
    (wrapped into a completion), a full completion (has "choices"), or (status, payload)."""

    def __init__(self, reply=None, models=("fake-model",), model_status=200):
        self.reply = reply or (lambda n, body: {"role": "assistant", "content": "ok"})
        self.models, self.model_status = list(models), model_status
        self.requests: list[dict] = []   # chat bodies, in order
        self.headers: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, payload):
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):  # the client gave up (timeout tests)
                    pass

            def do_GET(self):
                outer.headers.append(dict(self.headers))
                if self.path == "/v1/models":
                    return self._send(outer.model_status, {"object": "list", "data": [
                        {"id": m, "object": "model"} for m in outer.models]})
                self._send(404, {"error": {"message": "not found"}})

            def do_POST(self):
                outer.headers.append(dict(self.headers))
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != "/v1/chat/completions":
                    return self._send(404, {"error": {"message": "not found"}})
                outer.requests.append(body)
                r = outer.reply(len(outer.requests), body)
                status, payload = r if isinstance(r, tuple) else (200, r)
                if status == 200 and isinstance(payload, dict) and "choices" not in payload:
                    finish = "tool_calls" if payload.get("tool_calls") else "stop"
                    payload = {"id": f"c{len(outer.requests)}", "object": "chat.completion", "model": "served-model",
                               "choices": [{"index": 0, "message": payload, "finish_reason": finish}],
                               "usage": {"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010}}
                self._send(status, payload)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads, self.httpd.block_on_close = True, False
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def client(self, **kw) -> llm.Client:
        return llm.Client(url=self.url, model=kw.pop("model", "fake-model"), **kw)


def closed_port_url() -> str:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}/v1"


class ClientTest(unittest.TestCase):
    def test_settings_come_from_the_environment(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            c = llm.Client.from_env()
        self.assertEqual((c.url, c.model, c.key, c.timeout, c.ctx),
                         ("http://127.0.0.1:11434/v1", "qwen3:8b", None, 300.0, 16384))
        env = {"BRAIN_LLM_URL": "http://box:8000/v1/", "BRAIN_LLM_MODEL": "llama3.1:8b", "BRAIN_LLM_KEY": "k",
               "BRAIN_LLM_TIMEOUT": "60", "BRAIN_LLM_CTX": "8192"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = llm.Client.from_env()
        self.assertEqual((c.url, c.model, c.key, c.timeout, c.ctx), ("http://box:8000/v1", "llama3.1:8b", "k", 60.0, 8192))

    def test_chat_returns_the_message_and_sends_tools_with_auto_choice(self):
        tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
        with FakeOpenAI(lambda n, b: {"role": "assistant", "content": "hello"}) as srv:
            msg = srv.client(key="secret").chat([{"role": "user", "content": "hi"}], tools=tools, max_tokens=50)
            plain = srv.client().chat([{"role": "user", "content": "hi"}])
        self.assertEqual(msg, {"role": "assistant", "content": "hello"})
        self.assertEqual(plain["content"], "hello")
        body = srv.requests[0]
        self.assertEqual((body["model"], body["temperature"], body["stream"], body["max_tokens"], body["tool_choice"]),
                         ("fake-model", 0.2, False, 50, "auto"))
        self.assertEqual(body["tools"], tools)
        self.assertNotIn("tools", srv.requests[1])
        self.assertNotIn("tool_choice", srv.requests[1])
        auth = [h.get("Authorization") for h in srv.headers]
        self.assertEqual(auth, ["Bearer secret", None])

    def test_module_level_chat_uses_the_environment(self):
        with FakeOpenAI(lambda n, b: {"role": "assistant", "content": "env"}) as srv, \
                mock.patch.dict(os.environ, {"BRAIN_LLM_URL": srv.url, "BRAIN_LLM_MODEL": "m2"}):
            self.assertEqual(llm.chat([{"role": "user", "content": "x"}])["content"], "env")
        self.assertEqual(srv.requests[0]["model"], "m2")

    def test_available_needs_the_server_and_the_model(self):
        with FakeOpenAI(models=("qwen3:8b", "other")) as srv:
            self.assertEqual(srv.client(model="qwen3:8b").available(), (True, ""))
            self.assertEqual(srv.client(model="qwen3").available(), (False, mock.ANY))
            ok, why = srv.client(model="mistral").available()
            self.assertFalse(ok)
            self.assertIn("'mistral' is not on the server", why)
            self.assertIn("qwen3:8b, other", why)
            self.assertIn("BRAIN_LLM_MODEL", why)
        with FakeOpenAI(models=("qwen3:latest",)) as srv:
            self.assertTrue(srv.client(model="qwen3").available()[0])  # Ollama's implicit tag
        ok, why = llm.Client(url=closed_port_url()).available(timeout=2)
        self.assertFalse(ok)
        self.assertIn("BRAIN_LLM_URL", why)
        self.assertIn("ollama", why)

    def test_module_available_is_cached_briefly(self):
        with FakeOpenAI(models=("m",)) as srv, mock.patch.dict(os.environ, {"BRAIN_LLM_URL": srv.url, "BRAIN_LLM_MODEL": "m"}):
            llm._available_cache.clear()
            self.assertEqual(llm.available(), (True, ""))
            self.assertEqual(llm.available(), (True, ""))
            self.assertEqual(len(srv.headers), 1)
            srv.models = []
            self.assertFalse(llm.available(max_age=0)[0])
            self.assertEqual(len(srv.headers), 2)
        llm._available_cache.clear()

    def test_errors_are_typed_and_explained(self):
        cases = [
            ((401, {"error": {"message": "bad key"}}), "http", 401, "rejected", "BRAIN_LLM_KEY"),
            ((404, {"error": "model 'qwen3:8b' not found"}), "http", 404, "does not know", "ollama pull"),
            ((400, {"error": {"message": "registry.ollama.ai/library/gemma:2b does not support tools"}}), "http", 400,
             "does not support tool calling", "BRAIN_LLM_MODEL"),
            ((400, {"error": {"message": "Requested tokens (9000) exceed context window of 8192"}}), "http", 400,
             "context window", "BRAIN_LLM_CTX"),
            ((500, {"detail": "boom"}), "http", 500, "model server error 500", "boom"),
            ((200, {"object": "chat.completion", "choices": []}), "bad_response", None, "not chat-completions", "/v1"),
        ]
        for reply, kind, status, error, hint in cases:
            with self.subTest(error=error), FakeOpenAI(lambda n, b, r=reply: r) as srv:
                c = srv.client()
                with self.assertRaises(llm.LLMError) as cm:
                    c.chat([{"role": "user", "content": "x"}])
                self.assertEqual((cm.exception.kind, cm.exception.status), (kind, status))
                e, h = llm.describe_error(cm.exception, c)
                self.assertIn(error, e)
                self.assertIn(hint, h)
        with self.assertRaises(llm.LLMError) as cm:
            llm.Client(url=closed_port_url()).chat([{"role": "user", "content": "x"}])
        self.assertEqual(cm.exception.kind, "unreachable")
        e, h = llm.describe_error(cm.exception, llm.Client())
        self.assertEqual(e, "could not reach the model server")
        self.assertIn("BRAIN_LLM_URL", h)
        self.assertIn("ollama serve", h)

    def test_context_limit_is_read_from_the_servers_message(self):
        for text, n in [("Requested tokens (9000) exceed context window of 8192", 8192),
                        ("This model's maximum context length is 4096 tokens. However, you requested 5000", 4096),
                        ("something else", None)]:
            self.assertEqual(llm.LLMError("http", text, 400).context_limit(), n)
        self.assertIsNone(llm.LLMError("unreachable", "context window of 8192").context_limit())

    def test_a_slow_server_times_out(self):
        import time
        with FakeOpenAI(lambda n, b: time.sleep(1.5) or {"role": "assistant", "content": "late"}) as srv:
            with self.assertRaises(llm.LLMError) as cm:
                srv.client(timeout=0.3).chat([{"role": "user", "content": "x"}])
        self.assertEqual(cm.exception.kind, "timeout")
        self.assertIn("BRAIN_LLM_TIMEOUT", llm.describe_error(cm.exception, srv.client())[1])

    def test_the_key_stays_on_its_origin_across_redirects(self):
        # urllib copies Authorization to any redirect target; a hop to another host must not carry the key.
        seen = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                seen.append((self.headers.get("Host"), self.path, self.headers.get("Authorization")))
                port = self.server.server_port
                where = {"/v1/models": f"http://localhost:{port}/elsewhere/models",   # another origin
                         "/v2/models": "/v2/moved/models"}.get(self.path)             # the same origin
                if where:
                    self.send_response(302)
                    self.send_header("Location", where)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                raw = json.dumps({"data": [{"id": "m"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        try:
            port = httpd.server_port
            with mock.patch.dict(os.environ, {"no_proxy": "*", "NO_PROXY": "*"}):
                self.assertEqual(llm.Client(url=f"http://127.0.0.1:{port}/v1", key="sk-test-123").models(), ["m"])
                self.assertEqual(llm.Client(url=f"http://127.0.0.1:{port}/v2", key="sk-test-123").models(), ["m"])
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(seen, [(f"127.0.0.1:{port}", "/v1/models", "Bearer sk-test-123"),
                                (f"localhost:{port}", "/elsewhere/models", None),
                                (f"127.0.0.1:{port}", "/v2/models", "Bearer sk-test-123"),
                                (f"127.0.0.1:{port}", "/v2/moved/models", "Bearer sk-test-123")])

    def test_a_server_answering_this_process_is_busy_not_off(self):
        # llama.cpp's server answers one request at a time: a status probe during an answer timed out and the
        # terminal showed ANALYST OFF next to an analyst that was thinking.
        go, started = threading.Event(), threading.Event()

        def reply(n, body):
            started.set()
            go.wait(10)
            return {"role": "assistant", "content": "done"}
        with FakeOpenAI(reply, models=("m",)) as srv, \
                mock.patch.dict(os.environ, {"BRAIN_LLM_URL": srv.url, "BRAIN_LLM_MODEL": "m"}):
            llm._available_cache.clear()
            t = threading.Thread(target=lambda: llm.chat([{"role": "user", "content": "x"}]))
            t.start()
            try:
                self.assertTrue(started.wait(10))
                self.assertEqual(llm.available(max_age=0), (True, llm.BUSY))
                self.assertTrue(llm.Client.from_env().busy())
                self.assertEqual([h for h in srv.headers if "Content-Length" not in h], [])  # no probe sent
            finally:
                go.set()
                t.join(10)
            self.assertFalse(llm.Client.from_env().busy())
            self.assertEqual(llm.available(max_age=0), (True, ""))
        llm._available_cache.clear()
        # A probe that times out is not a question: raising BRAIN_LLM_TIMEOUT would not help it.
        import socket
        quiet = socket.socket()
        quiet.bind(("127.0.0.1", 0))
        quiet.listen(5)  # accepts connections, never answers
        try:
            ok, why = llm.Client(url=f"http://127.0.0.1:{quiet.getsockname()[1]}/v1").available(timeout=0.5)
        finally:
            quiet.close()
        self.assertFalse(ok)
        self.assertIn("GET /models within 0.5 s", why)
        self.assertIn("busy with another request", why)
        self.assertNotIn("BRAIN_LLM_TIMEOUT", why)

    def test_loopback_bypasses_proxies(self):
        with mock.patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                                          "no_proxy": "", "NO_PROXY": ""}):
            with FakeOpenAI(lambda n, b: {"role": "assistant", "content": "direct"}) as srv:
                self.assertEqual(srv.client().chat([{"role": "user", "content": "x"}])["content"], "direct")


if __name__ == "__main__":
    unittest.main()
