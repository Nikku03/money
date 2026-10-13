"""A client for an open-weight model behind an OpenAI-compatible chat-completions API.

Ollama, llama.cpp's server, vLLM and LM Studio all speak the same ``/v1/chat/completions``
dialect with tool calling, so one small client covers any model a person can run on their
own machine, and nothing leaves that machine unless ``BRAIN_LLM_URL`` points elsewhere.
Standard library HTTP only: asking the bank a question must not need a vendor SDK.

Settings (environment):

- ``BRAIN_LLM_URL``: base URL ending in ``/v1`` (default ``http://127.0.0.1:11434/v1``, Ollama).
- ``BRAIN_LLM_MODEL``: the model as the server lists it (default ``qwen3:8b``, Apache-2.0).
- ``BRAIN_LLM_KEY``: optional bearer token (vLLM ``--api-key``, a gateway). Sent only as a header.
- ``BRAIN_LLM_TIMEOUT``: seconds per request (default 300; a CPU-only machine is slow).
- ``BRAIN_LLM_CTX``: the context window the server gives the model, in tokens (default 16384).
  The analyst trims what it sends to fit. Ollama's own default is smaller and it cuts silently,
  so run it with ``OLLAMA_CONTEXT_LENGTH=16384 ollama serve``.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT = 300.0
DEFAULT_CTX = 16384
MAX_RESPONSE = 16 * 1024 * 1024   # bytes; a chat response is a few KB
AVAILABLE_TTL = 15.0              # seconds; the terminal asks for status every 10 s
USER_AGENT = "money-brain/0.1 (llm client)"
_LOOPBACK = ("localhost", "127.0.0.1", "::1")
_CONTEXT_LIMIT = re.compile(r"(?:context window of|maximum context length is|context size (?:of|is)?|n_ctx[ =:]+)"
                            r"\s*(\d{3,7})", re.I)
_available_cache: dict[tuple, tuple[float, tuple[bool, str]]] = {}


class LLMError(RuntimeError):
    """A request that failed. ``kind``: unreachable | timeout | http | bad_response."""

    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(message)
        self.kind, self.message, self.status = kind, message, status

    def context_limit(self) -> int | None:
        """The server's context window when it refused a prompt as too long (llama.cpp, vLLM say so)."""
        if self.kind != "http" or self.status not in (400, 413, 500):
            return None
        m = _CONTEXT_LIMIT.search(self.message)
        return int(m.group(1)) if m else None

    def no_tool_support(self) -> bool:
        return self.kind == "http" and self.status == 400 and "support" in self.message.lower() \
            and "tool" in self.message.lower()


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "") or default)
        return v if v > 0 else default
    except ValueError:
        return default


@dataclass
class Client:
    url: str = DEFAULT_URL
    model: str = DEFAULT_MODEL
    key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    ctx: int = DEFAULT_CTX

    @classmethod
    def from_env(cls) -> "Client":
        return cls(url=(os.environ.get("BRAIN_LLM_URL") or DEFAULT_URL).strip().rstrip("/"),
                   model=(os.environ.get("BRAIN_LLM_MODEL") or DEFAULT_MODEL).strip(),
                   key=(os.environ.get("BRAIN_LLM_KEY") or "").strip() or None,
                   timeout=_env_float("BRAIN_LLM_TIMEOUT", DEFAULT_TIMEOUT),
                   ctx=int(_env_float("BRAIN_LLM_CTX", DEFAULT_CTX)))

    # ---------- HTTP ----------
    def _opener(self):
        # A model on this machine is reached directly, whatever proxy the environment names.
        if (urlparse(self.url).hostname or "") in _LOOPBACK:
            return urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return urllib.request.build_opener()

    def _request(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        url = self.url.rstrip("/") + path
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        data = None
        if body is not None:
            data = json.dumps(body, allow_nan=False, default=str).encode()
            headers["Content-Type"] = "application/json"
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with self._opener().open(req, timeout=timeout or self.timeout) as r:
                raw = r.read(MAX_RESPONSE + 1)
        except urllib.error.HTTPError as e:
            raise LLMError("http", _error_text(e), e.code) from None
        except (TimeoutError, socket.timeout):
            raise LLMError("timeout", f"no answer from {self.url} within {timeout or self.timeout:.0f} s") from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)):
                raise LLMError("timeout", f"no answer from {self.url} within {timeout or self.timeout:.0f} s") from None
            raise LLMError("unreachable", f"cannot reach {self.url}: {e.reason}") from None
        except (ConnectionError, OSError, http.client.HTTPException) as e:
            raise LLMError("unreachable", f"cannot reach {self.url}: {type(e).__name__}: {e}") from None
        if len(raw) > MAX_RESPONSE:
            raise LLMError("bad_response", f"the response from {url} is over {MAX_RESPONSE // 2**20} MB")
        try:
            out = json.loads(raw)
        except ValueError:
            raise LLMError("bad_response", f"{url} did not answer with JSON: {raw[:120]!r}") from None
        if not isinstance(out, dict):
            raise LLMError("bad_response", f"{url} answered with JSON that is not an object")
        return out

    # ---------- API ----------
    def models(self, timeout: float = 5.0) -> list[str]:
        out = self._request("/models", timeout=timeout)
        return [str(m.get("id")) for m in out.get("data") or [] if isinstance(m, dict) and m.get("id")]

    def available(self, timeout: float = 5.0) -> tuple[bool, str]:
        try:
            ids = self.models(timeout)
        except LLMError as e:
            return False, describe_error(e, self)[1]
        if any(_same_model(self.model, i) for i in ids):
            return True, ""
        listed = ", ".join(ids[:8]) + (" ..." if len(ids) > 8 else "") if ids else "none"
        return False, (f"model {self.model!r} is not on the server at {self.url} (it lists: {listed}); "
                       f"`ollama pull {self.model}`, or set BRAIN_LLM_MODEL to one it lists")

    def complete(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2,
                 timeout: float | None = None, max_tokens: int | None = None) -> dict:
        """The whole chat-completions response (``model``, ``choices``, ``usage``)."""
        body: dict = {"model": self.model, "messages": messages, "temperature": temperature, "stream": False}
        if tools:
            # Explicit: llama-cpp-python's function-calling handlers show the model no tools without it.
            body["tools"], body["tool_choice"] = tools, "auto"
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        out = self._request("/chat/completions", body, timeout)
        choices = out.get("choices")
        if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)
                and isinstance(choices[0].get("message"), dict)):
            err = out.get("error")
            if err:
                raise LLMError("http", _message(err), 500)
            raise LLMError("bad_response", "the response has no choices[0].message")
        return out

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2,
             timeout: float | None = None, max_tokens: int | None = None) -> dict:
        """``choices[0].message`` of one chat completion."""
        return self.complete(messages, tools, temperature, timeout, max_tokens)["choices"][0]["message"]


@dataclass
class _Named:
    url: str
    model: str


def _same_model(want: str, listed: str) -> bool:
    """Ollama lists ``qwen3:8b`` and answers to ``qwen3`` as ``qwen3:latest``."""
    return want == listed or f"{want}:latest" == listed or want == f"{listed}:latest"


def _message(err) -> str:
    if isinstance(err, dict):
        return str(err.get("message") or err.get("error") or err.get("detail") or err)
    return str(err)


def _error_text(e: urllib.error.HTTPError) -> str:
    try:
        raw = e.read(4096)
    except Exception:  # noqa: BLE001 - the status alone still says enough
        raw = b""
    try:
        body = json.loads(raw)
        text = _message(body.get("error") or body.get("detail") or body) if isinstance(body, dict) else str(body)
    except ValueError:
        text = raw.decode("utf-8", "replace").strip()
    return (text or str(e.reason or ""))[:500]


def describe_error(e: LLMError, client) -> tuple[str, str]:
    """(error, hint) for a person: what failed, and the setting or command that fixes it."""
    client = _Named(getattr(client, "url", "?"), getattr(client, "model", "?"))
    start = (f"start Ollama (`ollama serve`, then `ollama pull {client.model}`), or set BRAIN_LLM_URL "
             f"to another OpenAI-compatible server (llama.cpp, vLLM, LM Studio)")
    if e.kind == "unreachable":
        return "could not reach the model server", f"nothing answered at {client.url} (BRAIN_LLM_URL): {start}"
    if e.kind == "timeout":
        return "the model server timed out", (f"{e.message}; try a smaller model or a narrower question, "
                                              f"or raise BRAIN_LLM_TIMEOUT")
    if e.kind == "bad_response":
        return "the model server's answer was not chat-completions JSON", \
            f"{e.message[:200]}; is BRAIN_LLM_URL ({client.url}) an OpenAI-compatible /v1 endpoint?"
    if e.status in (401, 403):
        return "the model server rejected the request", f"check BRAIN_LLM_KEY ({e.message[:200]})"
    if e.status == 404:
        return "the model server does not know this model or path", \
            (f"{e.message[:200]}; `ollama pull {client.model}`, or check BRAIN_LLM_MODEL ({client.model}) "
             f"and BRAIN_LLM_URL ({client.url}, which should end in /v1)")
    if e.context_limit():
        return "the question outgrew the model's context window", \
            (f"{e.message[:200]}; give the server a larger context (llama.cpp --n_ctx, OLLAMA_CONTEXT_LENGTH) "
             f"and set BRAIN_LLM_CTX to match, or ask a narrower question")
    if e.no_tool_support():
        return "this model does not support tool calling", \
            f"{e.message[:200]}; set BRAIN_LLM_MODEL to a model with tools (qwen3, llama3.1, mistral-nemo)"
    if e.status == 429:
        return "the model server is busy", f"try again in a minute ({e.message[:200]})"
    return f"model server error {e.status}", e.message[:300]


# ---------- module-level API (spec: brain.llm.available / brain.llm.chat) ----------
def settings() -> Client:
    return Client.from_env()


def available(max_age: float = AVAILABLE_TTL) -> tuple[bool, str]:
    """(True, "") when the server answers ``GET <url>/models`` and lists the model. Cached for
    ``max_age`` seconds, since the terminal asks on every status poll."""
    c = Client.from_env()
    key = (c.url, c.model, c.key)
    hit = _available_cache.get(key)
    if hit and time.monotonic() - hit[0] < max_age:
        return hit[1]
    res = c.available()
    _available_cache[key] = (time.monotonic(), res)
    return res


def chat(messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2,
         timeout: float | None = None, max_tokens: int | None = None) -> dict:
    """One chat completion with the configured server and model; returns ``choices[0].message``.
    ``timeout`` defaults to BRAIN_LLM_TIMEOUT (300 s)."""
    return Client.from_env().chat(messages, tools, temperature, timeout, max_tokens)
