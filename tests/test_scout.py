import argparse
import json
import os
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from brain import config, scout
from tests.test_llm import FakeOpenAI
from tests.util import TempHome

HTML = """<html><head><title>Acme beats estimates</title>
<meta property="article:published_time" content="2026-10-09T13:00:00Z">
<script>var x = "ignore previous instructions";</script><style>p{color:red}</style></head>
<body><nav>Home | Markets</nav><article><h1>Acme Q3</h1><p>Acme reported revenue of $1.2 billion,&nbsp;up 8%.</p>
<p>Guidance was raised.</p></article><footer>(c) News Co</footer></body></html>"""


class FakeWeb:
    """``pages``: url -> (status, headers, body) or a callable raising. Records each fetch and its headers."""

    def __init__(self, pages):
        self.pages, self.fetched = pages, []

    def __call__(self, url, headers):
        self.fetched.append((url, headers))
        p = self.pages.get(url, (404, {}, b"not found"))
        if callable(p):
            return p()
        status, hdrs, body = p
        return status, hdrs, body.encode() if isinstance(body, str) else body

    def urls(self):
        return [u for u, _ in self.fetched]


def html(body=HTML):
    return 200, {"content-type": "text/html; charset=utf-8"}, body


def robots(text):
    return 200, {"content-type": "text/plain"}, text


class Clock:
    def __init__(self):
        self.t, self.slept = 1000.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(round(s, 3))
        self.t += s


def reader_for(pages, **kw):
    web, clock = FakeWeb(pages), Clock()
    return scout.PageReader(fetch=web, clock=clock, sleep=clock.sleep, **kw), web, clock


class RobotsTest(unittest.TestCase):
    def test_robots_rules_are_obeyed_and_read_once_per_site(self):
        r, web, _ = reader_for({"https://news.example/robots.txt": robots("User-agent: *\nDisallow: /private\n"),
                                "https://news.example/a": html(), "https://news.example/b": html()})
        self.assertTrue(r.read("https://news.example/a")["ok"])
        res = r.read("https://news.example/private/x")
        self.assertFalse(res["ok"])
        self.assertIn("robots.txt disallows it", res["error"])
        self.assertTrue(r.read("https://news.example/b")["ok"])
        self.assertEqual(web.urls(), ["https://news.example/robots.txt", "https://news.example/a", "https://news.example/b"])
        self.assertFalse(r.was_read("https://news.example/private/x"))

    def test_rules_for_our_agent_apply(self):
        r, web, _ = reader_for({"https://x.example/robots.txt": robots("User-agent: money-brain\nDisallow: /\n\n"
                                                                       "User-agent: *\nAllow: /\n"),
                                "https://x.example/a": html()})
        self.assertIn("disallows", r.read("https://x.example/a")["error"])
        self.assertEqual(web.urls(), ["https://x.example/robots.txt"])

    def test_missing_robots_allows_and_forbidden_or_broken_robots_does_not(self):
        cases = [((404, {}, "nope"), True), ((410, {}, ""), True), ((403, {}, "no"), False), ((401, {}, ""), False),
                 ((503, {}, "down"), False)]
        for robots_reply, allowed in cases:
            with self.subTest(robots=robots_reply[0]):
                r, web, _ = reader_for({"https://y.example/robots.txt": robots_reply, "https://y.example/a": html()})
                self.assertEqual(r.read("https://y.example/a")["ok"], allowed)
                self.assertEqual("https://y.example/a" in web.urls(), allowed)

        def boom():
            raise OSError("connection reset")
        r, web, _ = reader_for({"https://z.example/robots.txt": boom, "https://z.example/a": html()})
        res = r.read("https://z.example/a")
        self.assertFalse(res["ok"])
        self.assertIn("robots.txt could not be read", res["error"])

    def test_one_request_per_second_per_host_and_crawl_delay(self):
        r, web, clock = reader_for({"https://a.example/robots.txt": robots("User-agent: *\nAllow: /\n"),
                                    "https://a.example/1": html(), "https://a.example/2": html(),
                                    "https://b.example/robots.txt": robots("User-agent: *\nCrawl-delay: 5\n"),
                                    "https://b.example/1": html()})
        r.read("https://a.example/1")
        r.read("https://a.example/2")
        self.assertEqual(clock.slept, [1.0, 1.0])  # robots -> page 1 -> page 2, one second apart
        r.read("https://b.example/1")
        self.assertEqual(clock.slept, [1.0, 1.0, 5.0])  # a new host: no wait for robots, then its crawl delay

    def test_our_user_agent_and_secs_contact_agent(self):
        r, web, _ = reader_for({"https://www.sec.gov/robots.txt": (404, {}, ""), "https://www.sec.gov/about": html(),
                                "https://n.example/robots.txt": (404, {}, ""), "https://n.example/a": html()})
        with mock.patch.dict(os.environ, {"SEC_USER_AGENT": "Test Person test@example.com"}):
            r.read("https://www.sec.gov/about")
            r.read("https://n.example/a")
        agents = {u: h["User-Agent"] for u, h in web.fetched}
        self.assertEqual(agents["https://www.sec.gov/about"], "Test Person test@example.com")
        self.assertEqual(agents["https://n.example/a"], config.WEB_USER_AGENT)


class PageTest(unittest.TestCase):
    def ok_site(self, **pages):
        base = {"https://p.example/robots.txt": (404, {}, "")}
        base.update({f"https://p.example/{k}": v for k, v in pages.items()})
        return reader_for(base)

    def test_html_becomes_text_with_title_and_publication_time(self):
        r, _, _ = self.ok_site(a=html())
        res = r.read("https://p.example/a#top")
        self.assertTrue(res["ok"])
        self.assertEqual((res["title"], res["published"]), ("Acme beats estimates", "2026-10-09T13:00:00Z"))
        self.assertIn("Acme reported revenue of $1.2 billion, up 8%.\nGuidance was raised.", res["text"])
        for gone in ("ignore previous", "color:red", "Home | Markets", "News Co"):
            self.assertNotIn(gone, res["text"])
        self.assertTrue(r.was_read("https://P.example/a/"))

    def test_long_pages_are_cut_to_20000_characters(self):
        r, _, _ = self.ok_site(long=(200, {"content-type": "text/plain"}, "w " * 30_000))
        res = r.read("https://p.example/long")
        self.assertEqual((len(res["text"]), res["truncated"], res["chars"]), (scout.MAX_PAGE_CHARS, True, 59_999))

    def test_paywalls_logins_and_bot_checks_are_left_alone(self):
        challenge = "<html><head><title>Just a moment...</title></head><body>checking</body></html>"
        paywalled = '<html><script type="application/ld+json">{"isAccessibleForFree": "False"}</script><p>teaser</p></html>'
        r, web, _ = self.ok_site(login=(401, {}, ""), pay=(402, {}, ""), deny=(403, {}, ""), slow=(429, {}, ""),
                                 bot=html(challenge), wall=html(paywalled),
                                 pdf=(200, {"content-type": "application/pdf"}, b"%PDF"))
        for page, words in [("login", "login"), ("pay", "paywall"), ("deny", "access denied"), ("slow", "rate"),
                            ("bot", "bot check"), ("wall", "paywalled"), ("pdf", "not a text page")]:
            with self.subTest(page=page):
                res = r.read(f"https://p.example/{page}")
                self.assertFalse(res["ok"])
                self.assertIn(words, res["error"])
                self.assertFalse(r.was_read(f"https://p.example/{page}"))
        self.assertEqual(web.urls().count("https://p.example/deny"), 1)  # no retry
        self.assertIn("access denied", r.read("https://p.example/deny")["error"])  # asking again: same answer, no fetch
        self.assertEqual(web.urls().count("https://p.example/deny"), 1)

    def test_a_captcha_on_a_page_with_content_is_not_a_bot_check(self):
        page = HTML.replace("</article>", '<div class="g-recaptcha"></div></article>') + "<p>" + "story " * 200 + "</p>"
        r, _, _ = self.ok_site(a=html(page), b=html('<html><body><div class="cf-turnstile"></div></body></html>'))
        self.assertTrue(r.read("https://p.example/a")["ok"])
        self.assertIn("bot check", r.read("https://p.example/b")["error"])

    def test_local_addresses_and_other_schemes_are_refused(self):
        r, web, _ = reader_for({})
        for url in ("http://127.0.0.1:8765/api/status", "http://localhost/x", "http://[::1]/", "http://10.0.0.5/",
                    "http://169.254.169.254/latest/meta-data", "file:///etc/passwd", "ftp://x.example/a"):
            with self.subTest(url=url):
                self.assertFalse(r.read(url)["ok"])
        self.assertEqual(web.fetched, [])

    def test_redirects_are_checked_hop_by_hop(self):
        r, web, _ = reader_for({
            "https://r.example/robots.txt": (404, {}, ""),
            "https://r.example/go": (301, {"location": "https://other.example/story"}, ""),
            "https://other.example/robots.txt": robots("User-agent: *\nDisallow: /story\n"),
            "https://r.example/local": (302, {"location": "http://127.0.0.1/admin"}, ""),
            "https://r.example/fine": (302, {"location": "/final"}, ""),
            "https://r.example/final": html()})
        self.assertIn("disallows", r.read("https://r.example/go")["error"])
        self.assertIn("local or private", r.read("https://r.example/local")["error"])
        res = r.read("https://r.example/fine")
        self.assertEqual((res["ok"], res["url"]), (True, "https://r.example/final"))
        self.assertTrue(r.was_read("https://r.example/fine") and r.was_read("https://r.example/final"))
        self.assertNotIn("https://other.example/story", web.urls())
        self.assertNotIn("http://127.0.0.1/admin", web.urls())

    def test_page_budget(self):
        r, web, _ = reader_for({"https://p.example/robots.txt": (404, {}, ""), "https://p.example/1": html(),
                                "https://p.example/2": html()}, max_pages=1)
        self.assertTrue(r.read("https://p.example/1")["ok"])
        self.assertTrue(r.read("https://p.example/1")["ok"])  # the same page again is not a new fetch
        self.assertIn("page budget used", r.read("https://p.example/2")["error"])
        self.assertEqual(web.urls().count("https://p.example/1"), 1)


def call(n, name, args):
    return {"id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class ScoutTest(TempHome):
    def web(self):
        return reader_for({"https://news.example/robots.txt": robots("User-agent: *\nDisallow: /secret\n"),
                           "https://news.example/acme": html()})

    def run_scout(self, reply, search=None, pages=None, **kw):
        reader, web, _ = reader_for(pages) if pages else self.web()
        logs = []
        with FakeOpenAI(reply) as srv:
            out = scout.scout(self.bank, "acme", "What happened at Acme?", client=srv.client(), search=search,
                              reader=reader, log=logs.append, **kw)
        return out, srv.requests, logs

    def tool_results(self, body):
        return {m["tool_call_id"]: json.loads(m["content"]) for m in body["messages"] if m["role"] == "tool"}

    def test_only_urls_read_in_this_run_can_be_noted(self):
        def reply(n, body):
            if n == 1:
                return {"role": "assistant", "content": None, "tool_calls": [
                    call(1, "note_claim", {"text": "Acme raised guidance", "url": "https://news.example/acme",
                                           "direction": "up", "tier": "reported"}),
                    call(2, "read_page", {"url": "https://news.example/acme"}),
                    call(3, "read_page", {"url": "https://news.example/secret/memo"})]}
            if n == 2:
                return {"role": "assistant", "content": None, "tool_calls": [
                    call(4, "note_claim", {"text": "Acme   raised\nguidance", "url": "https://news.example/acme",
                                           "direction": "up", "tier": "reported", "published": "2026-10-09T13:00:00Z"}),
                    call(5, "note_claim", {"text": "Leaked memo", "url": "https://news.example/secret/memo",
                                           "direction": "down", "tier": "reported"}),
                    call(6, "note_claim", {"text": "x", "url": "https://news.example/acme", "direction": "sideways",
                                           "tier": "reported"}),
                    call(7, "note_claim", {"text": "y", "url": "https://news.example/acme", "direction": "none",
                                           "tier": "reported", "published": "2099-01-01"})]}
            return {"role": "assistant", "content": "Noted one claim."}
        out, sent, logs = self.run_scout(reply)
        self.assertTrue(out["ok"], out)
        first = self.tool_results(sent[1])
        self.assertIn("only a URL that read_page read", first["call_1"]["error"])
        self.assertTrue(first["call_2"]["ok"])
        self.assertIn("robots.txt disallows it", first["call_3"]["error"])
        second = self.tool_results(sent[2])
        self.assertTrue(second["call_4"]["ok"])
        self.assertIn("only a URL that read_page read", second["call_5"]["error"])
        self.assertIn("direction is up, down or none", second["call_6"]["error"])
        self.assertIn("in the future", second["call_7"]["error"])
        claims = self.bank.facts("ACME", "claim")
        self.assertEqual(len(claims), 1)
        c = claims[0]
        self.assertEqual((c.value, c.known_at, c.tier, c.source["url"]),
                         ("Acme raised guidance", "2026-10-09T13:00:00Z", "reported", "https://news.example/acme"))
        self.assertEqual(c.context, {"direction": 1, "horizon_days": 21, "noted_by": "scout:fake-model"})
        self.assertEqual([n["id"] for n in out["notes"]], [c.id[:8]])
        self.assertEqual([(p["url"], p["ok"]) for p in out["pages"]],
                         [("https://news.example/acme", True), ("https://news.example/secret/memo", False)])
        self.assertEqual(out["answer"], "Noted one claim.")
        self.assertTrue(any(line.startswith("noted [") for line in logs))

    def test_a_deadline_ends_the_loop_and_the_output_keeps_its_shape(self):
        # brain decide gives the scout 15 minutes. Past the deadline the next request is the last, as at
        # max_rounds; run_loop's trace (results, convo) is not handed on.
        def reply(n, body):
            return {"role": "assistant", "content": None,
                    "tool_calls": [call(n, "read_page", {"url": "https://news.example/acme"})]}
        out, sent, _ = self.run_scout(reply, deadline=time.monotonic() - 1)
        self.assertEqual(len(sent), 2)
        self.assertEqual(out["stop_reason"], "deadline")
        self.assertIn("stopped at the time limit", out["answer"])
        self.assertNotIn("results", out)
        self.assertNotIn("convo", out)
        self.assertEqual([p["url"] for p in out["pages"]], ["https://news.example/acme"])

    def test_the_noter_is_the_model_by_its_canonical_name(self):
        # llama.cpp's server lists a model by its file path; the claim's noter is the model, not the path.
        def reply(n, body):
            if n == 1:
                return {"role": "assistant", "content": None, "tool_calls": [
                    call(1, "read_page", {"url": "https://news.example/acme"})]}
            if n == 2:
                return {"role": "assistant", "content": None, "tool_calls": [
                    call(2, "note_claim", {"text": "Acme raised guidance", "url": "https://news.example/acme",
                                           "direction": "up", "tier": "reported"})]}
            return {"role": "assistant", "content": "Noted one claim."}
        reader, _, _ = self.web()
        with FakeOpenAI(reply) as srv:
            out = scout.scout(self.bank, "acme", client=srv.client(model="/models/Qwen3-1.7B-Q8_0.gguf"),
                              reader=reader, log=lambda m: None)
        self.assertTrue(out["ok"], out)
        (c,) = self.bank.facts("ACME", "claim")
        self.assertEqual(c.context["noted_by"], "scout:qwen3-1.7b-q8_0")

    def test_claims_are_never_dated_before_the_page_that_makes_them(self):
        # A small model easily enters the date of the event a page describes; score-claims would then score
        # the claim on the very move it reports.
        undated = "<html><head><title>Acme</title></head><body><p>Acme jumped after earnings.</p></body></html>"
        pages = {"https://news.example/robots.txt": (404, {}, ""), "https://news.example/acme": html(),
                 "https://news.example/undated": html(undated)}

        def reply(n, body):
            if n == 1:
                return {"role": "assistant", "content": None, "tool_calls": [
                    call(1, "read_page", {"url": "https://news.example/acme"}),
                    call(2, "read_page", {"url": "https://news.example/undated"})]}
            if n == 2:
                return {"role": "assistant", "content": None, "tool_calls": [
                    call(3, "note_claim", {"text": "Acme jumped after earnings", "url": "https://news.example/acme",
                                           "direction": "up", "tier": "reported", "published": "2024-01-02"}),
                    call(4, "note_claim", {"text": "Acme jumped", "url": "https://news.example/undated",
                                           "direction": "up", "tier": "reported", "published": "2025-01-02"}),
                    call(5, "note_claim", {"text": "Acme updated", "url": "https://news.example/acme",
                                           "direction": "none", "tier": "reported",
                                           "published": "2026-10-09T18:00:00Z"})]}
            return {"role": "assistant", "content": "done"}
        before = scout.now_iso()
        out, sent, _ = self.run_scout(reply, pages=pages)
        res = self.tool_results(sent[2])
        self.assertEqual(res["call_3"]["known_at"], "2026-10-09T13:00:00Z")  # the page's own time, not 2024
        self.assertIn("earlier than the page's own publication time", res["call_3"]["note"])
        self.assertGreaterEqual(res["call_4"]["known_at"], before)  # no date on the page: when it was read
        self.assertIn("states no publication time", res["call_4"]["note"])
        self.assertEqual(res["call_5"]["known_at"], "2026-10-09T18:00:00Z")  # later than the page's: kept
        self.assertNotIn("note", res["call_5"])
        stored = {c.value: c.known_at for c in self.bank.facts("ACME", "claim")}
        self.assertEqual(stored["Acme jumped after earnings"], "2026-10-09T13:00:00Z")
        self.assertGreaterEqual(stored["Acme jumped"], before)
        self.assertIn("never the date of the event a page describes", sent[0]["messages"][0]["content"])

    def test_claims_are_filed_under_the_page_as_finally_read(self):
        # A page reached through a redirector on a reputable host is that other site's claim.
        pages = {"https://reputable.example/robots.txt": (404, {}, ""),
                 "https://reputable.example/out?to=acme": (302, {"location": "https://other.example/acme"}, ""),
                 "https://other.example/robots.txt": (404, {}, ""), "https://other.example/acme": html()}

        def reply(n, body):
            if n == 1:
                return {"role": "assistant", "content": None, "tool_calls": [
                    call(1, "read_page", {"url": "https://reputable.example/out?to=acme"})]}
            if n == 2:
                return {"role": "assistant", "content": None, "tool_calls": [
                    call(2, "note_claim", {"text": "Acme beat", "url": "https://reputable.example/out?to=acme",
                                           "direction": "up", "tier": "reported"})]}
            return {"role": "assistant", "content": "done"}
        out, sent, _ = self.run_scout(reply, pages=pages)
        self.assertEqual(self.tool_results(sent[2])["call_2"]["url"], "https://other.example/acme")
        (c,) = self.bank.facts("ACME", "claim")
        self.assertEqual(c.source, {"name": "other.example", "url": "https://other.example/acme"})
        self.assertEqual(out["notes"][0]["url"], "https://other.example/acme")

    def test_web_search_is_offered_only_with_a_search_backend(self):
        def reply(n, body):
            if n == 1 and any(t["function"]["name"] == "web_search" for t in body["tools"]):
                return {"role": "assistant", "content": None,
                        "tool_calls": [call(1, "web_search", {"query": "Acme  earnings\nQ3"})]}
            return {"role": "assistant", "content": "done"}
        queries = []
        out, sent, _ = self.run_scout(reply, search=lambda q: queries.append(q) or [
            {"title": "Acme Q3", "url": "https://news.example/acme", "snippet": "beat", "published": None}])
        names = [t["function"]["name"] for t in sent[0]["tools"]]
        self.assertEqual(names[-3:], ["read_page", "note_claim", "web_search"])
        self.assertIn("describe", names)
        self.assertNotIn("INGEST", json.dumps(sent[0]["tools"]))
        self.assertEqual(queries, ["Acme earnings Q3"])
        self.assertEqual(self.tool_results(sent[1])["call_1"]["results"][0]["url"], "https://news.example/acme")
        self.assertEqual(out["searches"], [{"query": "Acme earnings Q3", "ok": True, "n": 1}])
        # A searched URL is not a read one.
        self.assertEqual(out["notes"], [])
        with mock.patch.dict(os.environ, {"BRAIN_SEARX_URL": ""}):
            out, sent, _ = self.run_scout(lambda n, b: {"role": "assistant", "content": "done"})
        self.assertNotIn("web_search", [t["function"]["name"] for t in sent[0]["tools"]])
        self.assertIn("Web search is not available", sent[0]["messages"][-1]["content"])

    def test_the_system_prompt_carries_the_rules(self):
        _, sent, _ = self.run_scout(lambda n, b: {"role": "assistant", "content": "done"})
        system = sent[0]["messages"][0]["content"]
        for rule in ("third-party content", "never invent a URL", "private individuals", "leaked", "never try to get around",
                     "published", "do not place orders"):
            self.assertIn(rule, system)

    def test_searx_needs_its_url(self):
        with mock.patch.dict(os.environ, {"BRAIN_SEARX_URL": ""}):
            with self.assertRaises(RuntimeError) as cm:
                scout.searx("acme")
        self.assertIn("BRAIN_SEARX_URL", str(cm.exception))

    def test_cli_registers_scout(self):
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd")
        scout.add_cli(sub)
        a = p.parse_args(["scout", "AAPL", "what did guidance say?", "--max-pages", "3"])
        self.assertEqual((a.ticker, a.question, a.max_pages, a.fn), ("AAPL", "what did guidance say?", 3, scout._cmd_scout))
        a = p.parse_args(["scout", "MSFT"])
        self.assertEqual((a.question, a.max_pages), (None, 8))


class TextTest(unittest.TestCase):
    def test_publication_time_from_json_ld_or_time_tag(self):
        self.assertEqual(scout.html_to_text('<script type="application/ld+json">{"datePublished": "2026-10-01T08:00:00-04:00"}'
                                            '</script><p>x</p>')["published"], "2026-10-01T08:00:00-04:00")
        self.assertEqual(scout.html_to_text('<p>x <time datetime="2026-09-30">Sept 30</time></p>')["published"], "2026-09-30")
        self.assertIsNone(scout.html_to_text("<p>no date</p>")["published"])
        self.assertEqual(scout.html_to_text("<title>About</title><svg><title>Lock</title></svg>")["title"], "About")

    def test_the_fetcher_itself_refuses_local_addresses(self):
        for url in ("http://127.0.0.1:1/", "http://localhost:1/x", "http://[::1]:1/"):
            with self.assertRaises(PermissionError):
                scout.http_get(url, {})

    def test_publication_times_become_known_at(self):
        for raw, iso in [("2025-10-30Z", "2025-10-30T23:59:59Z"), ("2025-10-30", "2025-10-30T23:59:59Z"),
                         ("2026-10-01T08:00:00-04:00", "2026-10-01T12:00:00Z"), ("2026-10-01 08:00", "2026-10-01T08:00:00Z"),
                         ("Oct 30, 2025", None), ("", None), (None, None)]:
            self.assertEqual(scout.published_iso(raw), iso, raw)

    def test_private_literal(self):
        for host in ("localhost", "127.0.0.1", "[::1]", "10.1.2.3", "192.168.0.1", "169.254.169.254", "printer.local", ""):
            self.assertTrue(scout._private_literal(host), host)
        for host in ("www.sec.gov", "8.8.8.8", "example.com"):
            self.assertFalse(scout._private_literal(host), host)

    def test_hosts_that_are_not_plain_or_not_public_are_refused(self):
        refused = ["%31%32%37.0.0.1", "%6c%6f%63%61%6c%68%6f%73%74", "%31%36%39.254.169.254",  # urllib unquotes these
                   "100.64.0.1", "100.100.100.100",                       # shared/CGNAT: Tailscale's tailnet
                   "::ffff:127.0.0.1", "64:ff9b::7f00:1", "2002:7f00:1::", "::127.0.0.1", "fe80::1%eth0",
                   "2606:4700::1111%25x", "198.18.0.1", "0.0.0.0", "224.0.0.1",
                   "127.1", "2130706433", "0x7f.0.0.1", "0177.0.0.1", "\uff11\uff12\uff17.\uff10.\uff10.\uff11",
                   "intranet", "metadata.google.internal", "nas.lan", "a..b", "a b.example", "x@y.example"]
        for host in refused:
            with self.subTest(host=host), self.assertRaises(PermissionError):
                scout.vet_host(host)
        self.assertEqual(scout.vet_host("WWW.SEC.GOV."), "www.sec.gov")
        self.assertEqual(scout.vet_host("b\u00fccher.example"), "xn--bcher-kva.example")  # IDNA, as the connection
        self.assertEqual(scout.vet_host("[2606:4700::1111]"), "2606:4700::1111")
        self.assertEqual(scout.vet_host("8.8.8.8"), "8.8.8.8")
        for url in ("http://%31%32%37.0.0.1:1/", "http://user@example.com/", "file:///etc/passwd"):
            with self.subTest(url=url), self.assertRaises(PermissionError):
                scout.http_get(url, {})

    def test_only_common_text_encodings_are_decoded(self):
        # Python's punycode codec is quadratic: a 3 MB body named charset=punycode took minutes.
        for ctype, name in [("text/html; charset=punycode", "utf-8"), ("text/plain; charset=rot13", "utf-8"),
                            ('text/html; charset="ISO-8859-1"', "iso8859-1"), ("text/html; charset=windows-1252", "cp1252"),
                            ("text/html; charset=Shift_JIS", "shift_jis"), ("text/html", "utf-8"),
                            ("text/html; charset=nonsense", "utf-8")]:
            self.assertEqual(scout._charset(ctype), name, ctype)
        body = b"a" * 400_000 + b"-" + b"9" * 400_000
        r, _, _ = reader_for({"https://p.example/robots.txt": (404, {}, ""),
                              "https://p.example/p": (200, {"content-type": "text/plain; charset=punycode"}, body)})
        t0 = time.monotonic()
        res = r.read("https://p.example/p")
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertTrue(res["ok"])
        self.assertTrue(res["text"].startswith("aaaa"))


class _Local:
    """A local HTTP server for the fetcher's own tests; records each request's path and Host header."""

    def __init__(self, handler=None):
        self.hits = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.hits.append((self.path, self.headers.get("Host")))
                if handler:
                    return handler(self)
                if self.path == "/robots.txt":
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = b"internal-only data"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def fake_dns(table, calls=None):
    """getaddrinfo answering from ``table`` (name -> address, or a list of addresses served in turn)."""
    real = socket.getaddrinfo

    def lookup(host, port, *a, **kw):
        if host in table:
            ans = table[host]
            if isinstance(ans, list):
                ans = ans[len([c for c in (calls or []) if c == host]) % len(ans)]
            if calls is not None:
                calls.append(host)
            if ans is None:
                raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ans, port or 0))]
        return real(host, port, *a, **kw)
    return lookup


class FetchTest(unittest.TestCase):
    """The real fetcher (``http_get``): what it checks is what it connects to, and one fetch has one deadline."""

    def setUp(self):
        p = mock.patch.object(scout, "_proxy_for", return_value=None)  # direct, whatever this machine's proxies
        p.start()
        self.addCleanup(p.stop)

    def test_encoded_hosts_never_reach_a_local_server(self):
        srv = _Local()
        self.addCleanup(srv.close)
        r = scout.PageReader(min_interval=0)
        for host in ("%31%32%37.0.0.1", "%6c%6f%63%61%6c%68%6f%73%74", "127.0.0.1", "localhost", "2130706433"):
            with self.subTest(host=host):
                url = f"http://{host}:{srv.port}/secret"
                self.assertFalse(r.read(url)["ok"])
                self.assertFalse(r.was_read(url))
        # The same through a redirect from a public page: each hop is checked by the fetcher itself.
        real = scout.http_get

        def fetch(url, headers):
            if url.startswith("https://public.example/"):
                if url.endswith("/robots.txt"):
                    return 404, {}, b""
                return 302, {"location": f"http://%31%32%37.0.0.1:{srv.port}/via-redirect"}, b""
            return real(url, headers)
        r = scout.PageReader(fetch=fetch, min_interval=0)
        res = r.read("https://public.example/article")
        self.assertFalse(res["ok"])
        self.assertIn("not a plain host name", res["error"])
        self.assertFalse(r.was_read("https://public.example/article"))
        self.assertEqual(srv.hits, [])

    def test_the_connection_goes_to_the_address_that_was_checked(self):
        # DNS rebinding: a zero-TTL name answers a public address to the check and loopback to the connection.
        lookups, connected = [], []

        def connect(addr, *a, **kw):
            connected.append(addr)
            raise ConnectionRefusedError("test: no network")
        with mock.patch("socket.getaddrinfo", fake_dns({"rebind.example": ["93.184.216.34", "127.0.0.1"]}, lookups)), \
                mock.patch("socket.create_connection", connect):
            with self.assertRaises(ConnectionRefusedError):
                scout.http_get("http://rebind.example:8080/x", {})
            self.assertEqual((lookups, connected), (["rebind.example"], [("93.184.216.34", 8080)]))
            with self.assertRaises(PermissionError):  # the next answer is loopback: refused, nothing connects
                scout.http_get("http://rebind.example:8080/x", {})
            self.assertEqual(connected, [("93.184.216.34", 8080)])
        with mock.patch("socket.getaddrinfo", lambda *a, **k: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 80))]), \
                mock.patch("socket.create_connection", connect):
            with self.assertRaises(PermissionError):  # every address must be public, not only the first
                scout.http_get("http://mixed.example/x", {})
        self.assertEqual(len(connected), 1)

    def test_unresolvable_names_fail_closed_unless_a_proxy_resolves_them(self):
        connected = []

        def connect(addr, *a, **kw):
            connected.append(addr)
            raise ConnectionRefusedError("test")
        with mock.patch("socket.getaddrinfo", fake_dns({"nowhere.example": None, "local.example": "127.0.0.1"})), \
                mock.patch("socket.create_connection", connect):
            with self.assertRaises(socket.gaierror):
                scout.http_get("http://nowhere.example/x", {})
            self.assertEqual(connected, [])
            with mock.patch.object(scout, "_proxy_for", return_value="http://proxy.example:3128"):
                with self.assertRaises(ConnectionRefusedError):
                    scout.http_get("http://nowhere.example/x", {})
                self.assertEqual(connected, [("proxy.example", 3128)])
                with self.assertRaises(PermissionError):  # resolvable here, to a local address: refused even so
                    scout.http_get("http://local.example/x", {})
        self.assertEqual(len(connected), 1)

    def test_a_pinned_fetch_keeps_the_name_for_the_host_header(self):
        srv = _Local()
        self.addCleanup(srv.close)
        with mock.patch("socket.getaddrinfo", fake_dns({"site.example": "127.0.0.1"})), \
                mock.patch.object(scout, "_public_ip", return_value=True):  # let the test server count as public
            status, hdrs, body = scout.http_get(f"http://site.example:{srv.port}/a?b=1", {"User-Agent": "t"})
        self.assertEqual((status, body, hdrs["content-type"]), (200, b"internal-only data", "text/plain"))
        self.assertEqual(srv.hits, [("/a?b=1", f"site.example:{srv.port}")])

    def test_a_fetch_has_one_wall_clock_limit(self):
        # A server that drips one byte every 0.2 s never trips a per-read timeout; the whole fetch is bounded.
        def drip(h):
            if h.path == "/headers":
                try:
                    h.wfile.write(b"HTTP/1.1 200 OK\r\n")
                    for _ in range(40):
                        time.sleep(0.2)
                        h.wfile.write(b"X: y\r\n")
                        h.wfile.flush()
                except OSError:
                    pass
                return
            h.send_response(200)
            h.send_header("Content-Type", "text/plain")
            h.send_header("Content-Length", "40")
            h.end_headers()
            for _ in range(40):
                time.sleep(0.2)
                try:
                    h.wfile.write(b"x")
                    h.wfile.flush()
                except OSError:
                    return
        srv = _Local(drip)
        self.addCleanup(srv.close)
        with mock.patch("socket.getaddrinfo", fake_dns({"drip.example": "127.0.0.1"})), \
                mock.patch.object(scout, "_public_ip", return_value=True):
            for path in ("/body", "/headers"):
                with self.subTest(path=path):
                    t0 = time.monotonic()
                    with self.assertRaises(TimeoutError):
                        scout.http_get(f"http://drip.example:{srv.port}{path}", {}, timeout=1.0)
                    self.assertLess(time.monotonic() - t0, 2.5)


if __name__ == "__main__":
    unittest.main()
