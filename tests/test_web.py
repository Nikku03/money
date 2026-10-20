import http.client
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from brain import scout
from brain.bank import Fact
from brain.web import events, extract, plan
from brain.web.gateway import Gateway, Page, _host_problem, browser_available
from tests.util import TempHome

ARTICLE = """<html><head><title>Acme | Example News</title>
<meta property="og:site_name" content="Example News">
<script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"WebPage","name":"x"},
{"@type":"NewsArticle","headline":"Acme raises full-year guidance after record quarter",
"datePublished":"2026-10-15T09:05:00-04:00","author":[{"@type":"Person","name":"Jane Roe"}],
"publisher":{"@type":"Organization","name":"Example News"},"description":"Acme lifted its outlook."}]}</script>
</head><body><h1>Acme raises full-year guidance</h1><p id="x">{body}</p>
<script>document.getElementById('x').textContent += ' Rendered by script.';</script></body></html>"""
B_ARTICLE = """<html><head><title>Acme to acquire Beta</title><meta property="og:title" content="Acme to acquire Beta for $2 billion">
<meta property="article:published_time" content="2026-10-16T12:00:00Z"><meta name="description" content="A deal."></head>
<body><p>{body}</p></body></html>"""
LONG = "Acme Corp reported results. " * 40
PAGES = {
    "/robots.txt": (200, "text/plain", "User-agent: *\nDisallow: /private/\n"),
    "/a.html": (200, "text/html", ARTICLE.replace("{body}", LONG)),
    "/b.html": (200, "text/html", B_ARTICLE.replace("{body}", LONG)),
    "/private/x.html": (200, "text/html", "<p>secret</p>"),
    "/captcha.html": (200, "text/html", "<html><head><title>Just a moment...</title></head><body>checking</body></html>"),
    "/login.html": (200, "text/html", "<html><body><form><input type='password' name='p'></form></body></html>"),
    "/paywall.html": (200, "text/html", '<html><head><script type="application/ld+json">{"@type":"NewsArticle",'
                      '"headline":"h","isAccessibleForFree":"False"}</script></head><body>' + LONG + "</body></html>"),
    "/list.html": (200, "text/html", ('<a href="/a.html">Acme guidance</a> <a href="/b.html">Acme deal</a> '
                                      '<a href="https://elsewhere.example/c">other</a>')),
    "/gone.html": (404, "text/html", "no"),
    "/pic.png": (200, "image/png", "not really a png"),
    "/feed.xml": (200, "application/atom+xml", ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Acme news'
                                                '</title><link rel="alternate" href="https://acme.example/story-1"/>'
                                                '<link rel="enclosure" href="https://acme.example/story-1.jpg"/>'
                                                '<updated>2026-10-16T10:00:00Z</updated></entry></feed>')),
}
REDIRECTS = {"/redir": "/a.html", "/to-private": "http://10.0.0.1/admin"}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.seen.append((self.path, self.headers.get("User-Agent")))
        if self.path in REDIRECTS:
            self.send_response(302)
            self.send_header("Location", REDIRECTS[self.path])
            self.end_headers()
            return
        status, ctype, body = PAGES.get(self.path, (404, "text/plain", "missing"))
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def local_fetch(url, headers, timeout=10):
    """scout.http_get for the test server only: no public-address pinning, no redirects followed."""
    from urllib.parse import urlparse
    u = urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    c.request("GET", u.path or "/", headers=headers)
    try:
        r = c.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    finally:
        c.close()


def local_ok(host):
    return None if host in ("127.0.0.1", "localhost") else _host_problem(host)


class Server:
    def __enter__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.seen = []
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        return self

    def __exit__(self, *e):
        self.httpd.shutdown()


def gateway(mode, **kw):
    reader = scout.PageReader(max_pages=kw.pop("max_pages", 40), fetch=local_fetch, min_interval=0)
    return Gateway(mode=mode, reader=reader, host_problem=local_ok, settle=0.2, **kw)


class ExtractTest(unittest.TestCase):
    def test_article_from_json_ld_and_from_meta(self):
        a = extract.article(ARTICLE.replace("{body}", LONG), "https://news.example/a")
        self.assertEqual(a["headline"], "Acme raises full-year guidance after record quarter")
        self.assertEqual(a["published"], "2026-10-15T13:05:00Z")       # stated with an offset, kept in UTC
        self.assertEqual((a["author"], a["publisher"]), ("Jane Roe", "Example News"))
        self.assertTrue(a["structured"])
        b = extract.article(B_ARTICLE.replace("{body}", LONG), "https://news.example/b")
        self.assertEqual((b["headline"], b["published"], b["structured"]),
                         ("Acme to acquire Beta for $2 billion", "2026-10-16T12:00:00Z", False))
        none = extract.article("<html><title>Plain</title><p>text</p></html>", "https://x.example/")
        self.assertIsNone(none["published"])                              # nothing stated, nothing invented

    def test_links_and_tables(self):
        ls = extract.links(PAGES["/list.html"][2], "https://news.example/list", same_host=True)
        self.assertEqual([x["url"] for x in ls], ["https://news.example/a.html", "https://news.example/b.html"])
        self.assertEqual(len(extract.links(PAGES["/list.html"][2], "https://news.example/", "deal")), 1)
        t = extract.tables("<table><tr><th>Q</th><th>EPS</th></tr><tr><td>Q3</td><td>1.98</td></tr></table>")
        self.assertEqual(t, [[["Q", "EPS"], ["Q3", "1.98"]]])

    def test_feed_items_take_the_story_link(self):
        items = extract.feed(PAGES["/feed.xml"][2])
        self.assertEqual(items, [{"url": "https://acme.example/story-1", "text": "Acme news",
                                  "published": "2026-10-16T10:00:00Z"}])
        rss = extract.feed('<rss><channel><item><title>T</title><link>https://a.example/x</link>'
                           '<pubDate>Thu, 15 Oct 2026 13:00:00 GMT</pubDate></item></channel></rss>')
        self.assertEqual(rss[0]["url"], "https://a.example/x")

    def test_event_tags(self):
        self.assertEqual(events.tag("Acme raises full-year guidance"), ["guidance_raised"])
        self.assertEqual(events.tag("Acme to acquire Beta", "and launches a buyback"),
                         ["acquisition", "buyback", "product_launch"])
        self.assertIn("downgrade", events.tag("Analyst downgrades Acme to neutral"))
        self.assertIn("earnings_beat", events.tag("Acme tops Wall Street estimates"))
        self.assertEqual(events.tag("Acme shares were little changed"), [])
        for not_a_raise in ("gives us increased confidence in our outlook for the remainder of the year",
                            "announced price increases fully offset the current forecast for raw material costs"):
            self.assertNotIn("guidance_raised", events.tag(not_a_raise))
        self.assertIn("guidance_raised", events.tag("Company raises 2019 guidance for net sales, Adjusted EBITDA and EPS"))
        for not_a_change in ("downtime events that modestly lowered sales below previous guidance",
                             "energy prices which dramatically increased since our previous guidance",
                             "we are reducing our 2023 cash burn guidance", "an increase in the forecasted costs to complete",
                             "two rate cuts in 2025, however, they also forecasted 6 rate cuts"):
            self.assertEqual([k for k in events.tag(not_a_change) if k.startswith("guidance")], [], not_a_change)
        self.assertIn("guidance_cut", events.tag("cutting its sales forecasts for the year"))
        self.assertIn("guidance_raised", events.tag("we today raise our previous guidance for full-year 2023"))


class GatewayRules:
    """The same rules, through whichever mode a subclass sets."""
    mode = "http"

    def test_reads_by_the_rules_and_refuses_what_they_refuse(self):
        with Server() as s, gateway(self.mode) as gw:
            p = gw.open(s.base + "/redir")
            self.assertTrue(p.ok, p.error)
            self.assertEqual(p.final_url, s.base + "/a.html")
            self.assertIn("Acme raises", extract.article(p.html, p.final_url)["headline"])
            for path, word in (("/private/x.html", "robots.txt"), ("/captcha.html", "bot check"),
                               ("/login.html", "login"), ("/paywall.html", "paywalled"), ("/gone.html", "404"),
                               ("/to-private", "local or private"), ("/pic.png", "not a web page")):
                q = gw.open(s.base + path)
                self.assertFalse(q.ok, path)
                self.assertIn(word, q.error, (path, q.error))
                self.assertEqual(q.html, "")
            self.assertNotIn("/private/x.html", [x[0] for x in s.httpd.seen])   # never even asked for
            f = gw.open(s.base + "/feed.xml", feed=True)
            self.assertTrue(f.ok, f.error)
            self.assertEqual(f.mode, "http")                                       # a feed is never rendered
            uas = {ua for _, ua in s.httpd.seen}
            self.assertTrue(all("money-brain" in (ua or "") for ua in uas), uas)  # says who it is, every time


class HttpGatewayTest(GatewayRules, unittest.TestCase):
    mode = "http"

    def test_budget_and_non_web_urls(self):
        with Server() as s, gateway("http", max_pages=1) as gw:
            self.assertTrue(gw.open(s.base + "/a.html").ok)
            self.assertIn("budget", gw.open(s.base + "/b.html").error)
            self.assertIn("http(s)", gw.open("file:///etc/passwd").error)


@unittest.skipUnless(browser_available()[0], "browser-use or Chromium not installed")
class BrowserGatewayTest(GatewayRules, unittest.TestCase):
    mode = "browser"

    def test_the_browser_renders_scripts_and_declares_itself(self):
        with Server() as s, gateway("browser") as gw:
            p = gw.open(s.base + "/a.html")
            self.assertTrue(p.ok, p.error)
            self.assertEqual((p.mode, p.status), ("browser", 200))
            self.assertIn("Rendered by script.", p.html)
            wd = gw._run(self._webdriver(gw, s.base + "/b.html"))
            self.assertEqual(wd, "true")   # navigator.webdriver: the site can see a program drives the browser

    @staticmethod
    async def _webdriver(gw, url):
        page = await gw._session.new_page(url)
        try:
            import asyncio
            await asyncio.sleep(0.5)
            return await page.evaluate("() => String(navigator.webdriver)")
        finally:
            await gw._session.close_page(page)


class FakeGateway:
    def __init__(self, pages):
        self.pages, self.asked = pages, []
        self.reader = type("R", (), {"fetched": 0})()

    def open(self, url, feed=False):
        self.asked.append(url)
        self.reader.fetched += 1
        html = self.pages.get(url)
        if html is None:
            return Page(False, url, error="not read: robots.txt disallows it (respecting it)")
        return Page(True, url, final_url=url, status=200, html=html, fetched_at="2026-10-17T10:00:00Z", mode="http")


class PlanTest(TempHome):
    def headline(self, t, title, url, when):
        return Fact(entity=t, attr="headline", value=title, as_of=when[:10], known_at=when, tier="reported",
                    source={"name": "finance.yahoo.com", "url": url}, context={"feed": "yahoo-rss"})

    def test_plan_reads_unread_headlines_then_keeps_cited_facts(self):
        self.bank.add_many([self.headline("ACME", "Acme raises guidance", "https://news.example/a", "2026-10-15T13:00:00Z"),
                            self.headline("ACME", "Acme to acquire Beta", "https://news.example/b", "2026-10-16T12:00:00Z"),
                            self.headline("ACME", "Old news", "https://news.example/old", "2026-09-01T12:00:00Z")])
        tasks = plan.plan(self.bank, ["acme"], per_ticker=2, source_path="/nonexistent.json")
        self.assertEqual([t["url"] for t in tasks], ["https://news.example/b", "https://news.example/a"])  # newest first
        gw = FakeGateway({"https://news.example/a": ARTICLE.replace("{body}", LONG),
                          "https://news.example/b": B_ARTICLE.replace("{body}", LONG)})
        out = plan.run(self.bank, tasks, gw, log=None)
        self.assertEqual(len(out["kept"]), 2)
        facts = {f.source["url"]: f for f in self.bank.view().facts("ACME", plan.ATTR)}
        a = facts["https://news.example/a"]
        self.assertEqual((a.known_at, a.tier, a.source["name"]), ("2026-10-15T13:05:00Z", "reported", "Example News"))
        self.assertEqual(a.context["events"], ["guidance_raised"])
        self.assertEqual(facts["https://news.example/b"].context["events"], ["acquisition"])
        self.assertEqual([t["url"] for t in plan.plan(self.bank, ["ACME"], per_ticker=2, source_path="/nonexistent.json")],
                         ["https://news.example/old"])   # what was read is not asked for again

    def test_a_page_with_no_stated_time_is_known_when_read(self):
        self.bank.add_many([self.headline("ACME", "x", "https://news.example/p", "2026-10-15T13:00:00Z")])
        gw = FakeGateway({"https://news.example/p": "<html><title>Plain</title><body>" + LONG + "</body></html>"})
        plan.run(self.bank, plan.plan(self.bank, ["ACME"], source_path="/nonexistent.json"), gw, log=None)
        [f] = self.bank.view().facts("ACME", plan.ATTR)
        self.assertEqual((f.known_at, f.context["published_stated"]), ("2026-10-17T10:00:00Z", False))

    def test_listed_sources_are_followed(self):
        import json
        import os
        src = os.path.join(os.environ["MONEY_HOME"], "sources.json")
        with open(src, "w") as fh:
            json.dump({"sources": [{"ticker": "ACME", "url": "https://acme.example/news", "pattern": "release",
                                    "follow": 1}]}, fh)
        gw = FakeGateway({"https://acme.example/news": '<a href="/release-1">Release one</a><a href="/about">About</a>',
                          "https://acme.example/release-1": B_ARTICLE.replace("{body}", LONG)})
        out = plan.run(self.bank, plan.plan(self.bank, ["ACME"], source_path=src), gw, log=None)
        self.assertEqual(gw.asked, ["https://acme.example/news", "https://acme.example/release-1"])
        self.assertEqual([k["url"] for k in out["kept"]], ["https://acme.example/release-1"])
