"""Reddit read by company and topic (brain/reddit.py): names, topics, what is kept, the API client, the cards."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

from brain import reddit

TABLE = [{"ticker": "COST", "cik": 1, "title": "Costco Wholesale Corp /NEW"},
         {"ticker": "AMD", "cik": 2, "title": "Advanced Micro Devices Inc"},
         {"ticker": "ADI", "cik": 3, "title": "Analog Devices Inc"},
         {"ticker": "GOOGL", "cik": 4, "title": "Alphabet Inc."},
         {"ticker": "GOOG", "cik": 4, "title": "Alphabet Inc."},
         {"ticker": "TGT", "cik": 5, "title": "TARGET CORP"},
         {"ticker": "FYBR", "cik": 6, "title": "Frontier Communications Parent, Inc."},
         {"ticker": "ULCC", "cik": 7, "title": "Frontier Group Holdings, Inc."},
         {"ticker": "NOW", "cik": 8, "title": "ServiceNow, Inc."},
         {"ticker": "WMS", "cik": 9, "title": "Advanced Drainage Systems, Inc."}]


def post(title, body="", sub="wallstreetbets", pid="abc", created=1764000000, author="someone"):
    return {"id": pid, "title": title, "selftext": body, "subreddit": sub, "created_utc": created, "score": 12,
            "num_comments": 3, "author": author, "permalink": f"/r/{sub}/comments/{pid}/x/"}


class Names(unittest.TestCase):
    def test_common_names(self):
        n = reddit.names_for(["COST", "AMD", "ADI", "GOOGL", "TGT", "FYBR", "NOW", "ZZZ"], TABLE)
        self.assertEqual(n["COST"], "Costco")
        self.assertEqual(n["AMD"], "Advanced Micro Devices")         # "Advanced" starts two companies' names
        self.assertEqual(n["GOOGL"], "Google")                       # an alias: the name people use
        self.assertIsNone(n["TGT"])                                  # "Target" is an ordinary word
        self.assertEqual(n["FYBR"], "Frontier Communications Parent")  # "Frontier" is two companies
        self.assertEqual(n["ADI"], "Analog Devices")                 # "Analog" is an ordinary word
        self.assertEqual(n["NOW"], "Servicenow")
        self.assertIsNone(n["ZZZ"])

    def test_query_and_matching(self):
        self.assertEqual(reddit.query_for("COST", "Costco"), '"Costco" OR COST')
        self.assertEqual(reddit.query_for("NOW", "Servicenow"), '"Servicenow"')       # NOW is a word
        self.assertEqual(reddit.query_for("TGT", None), "TGT")
        self.assertEqual(reddit.query_for("ON", None), '"$ON"')
        self.assertTrue(reddit.names_it("costco cut hours", "COST", "Costco"))
        self.assertTrue(reddit.names_it("bought $NOW calls", "NOW", "Servicenow"))
        self.assertFalse(reddit.names_it("right NOW the market", "NOW", "Servicenow"))
        self.assertTrue(reddit.names_it("TGT earnings", "TGT", None))
        self.assertFalse(reddit.names_it("target practice", "TGT", None))
        self.assertFalse(reddit.names_it("Costcos everywhere", "COST", "Costco"))
        self.assertTrue(reddit.names_it("Meta cut 5% of staff", "META", "Meta"))
        self.assertFalse(reddit.names_it("the meta has shifted", "META", "Meta"))      # a capital is required
        self.assertTrue(reddit.names_it("buying META here", "META", "Meta"))


class Topics(unittest.TestCase):
    def test_fixed_rules(self):
        self.assertEqual(reddit.topics_in("Big layoffs announced, severance is 8 weeks"), ["layoffs"])
        self.assertEqual(reddit.topics_in("Got a RIF notice today"), ["layoffs"])
        self.assertEqual(reddit.topics_in("the rif between them"), [])              # lower case: another word
        self.assertEqual(reddit.topics_in("My offer was rescinded after the hiring freeze"), ["hiring_freeze"])
        self.assertEqual(reddit.topics_in("the app is down again, outage for hours"), ["outage"])
        self.assertEqual(reddit.topics_in("the stock is down 5% today"), [])        # a price, not an outage
        self.assertEqual(reddit.topics_in("bought the $50 strike calls"), [])       # an option, not a strike
        self.assertEqual(reddit.topics_in("workers are on strike at the plant"), ["strike"])
        self.assertEqual(reddit.topics_in("I recall that the CEO said"), [])
        self.assertEqual(reddit.topics_in("voluntary recall of 2 million units"), ["recall"])
        self.assertEqual(reddit.topics_in("they delayed the launch again"), ["delay"])
        self.assertEqual(reddit.topics_in("my flight was delayed"), [])
        self.assertEqual(reddit.topics_in("the launch was delayed to March"), ["delay"])
        self.assertEqual(reddit.topics_in("closing 150 stores next year"), ["closures"])
        self.assertEqual(reddit.topics_in("the CFO is stepping down"), ["leader_leaving"])
        self.assertEqual(reddit.topics_in("RTO mandate and a bonus cut"), ["morale"])
        self.assertEqual(sorted(reddit.topics_in("reorg and layoffs")), ["layoffs", "reorg"])


class Record(unittest.TestCase):
    def test_kept_without_author_or_text(self):
        r = reddit.record_of(post("Costco layoffs?", "heard about cuts", sub="Costco"), "COST", "Costco", "T")
        self.assertEqual(set(r), {"id", "t", "sub", "created", "score", "comments", "kind", "topics", "fetched_at"})
        self.assertNotIn("someone", json.dumps(r))
        self.assertNotIn("heard", json.dumps(r))
        self.assertEqual((r["kind"], r["topics"], r["sub"]), ("other", ["layoffs"], "costco"))
        self.assertEqual(r["created"], datetime.fromtimestamp(1764000000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.assertIsNone(reddit.record_of(post("nothing about it"), "COST", "Costco", "T"))
        self.assertEqual(reddit.record_of(post("COST calls"), "COST", "Costco", "T")["kind"], "investing")


class FakeResponse(io.BytesIO):
    def __init__(self, body, headers=None):
        super().__init__(json.dumps(body).encode())
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    def __init__(self, pages, token_status=200):
        self.pages, self.calls, self.token_status = list(pages), [], token_status

    def __call__(self, req, timeout=30):
        self.calls.append(req)
        if req.full_url == reddit.TOKEN_URL:
            if self.token_status != 200:
                raise urllib.error.HTTPError(req.full_url, self.token_status, "no", {}, None)
            return FakeResponse({"access_token": "tok", "expires_in": 3600})
        return FakeResponse(self.pages.pop(0), {"X-Ratelimit-Remaining": "90", "X-Ratelimit-Reset": "30"})


def listing(posts, after=None):
    return {"data": {"children": [{"kind": "t3", "data": p} for p in posts], "after": after}}


class Client(unittest.TestCase):
    def test_token_then_search_two_pages(self):
        op = FakeOpener([listing([post("Costco a", pid="1")], after="t3_1"), listing([post("Costco b", pid="2")])])
        c = reddit.Client(opener=op, sleep=lambda s: None)
        got = c.search('"Costco" OR COST', pages=2)
        self.assertEqual([p["id"] for p in got], ["1", "2"])
        self.assertEqual(op.calls[0].full_url, reddit.TOKEN_URL)
        self.assertNotIn("Authorization", op.calls[0].headers)       # the proxy adds the saved key
        self.assertEqual(op.calls[1].get_header("Authorization"), "bearer tok")
        self.assertIn("after=t3_1", op.calls[2].full_url)
        self.assertIn("sort=new", op.calls[1].full_url)

    def test_no_key(self):
        with self.assertRaises(reddit.NoKey):
            reddit.token(FakeOpener([], token_status=401))

    def test_collect_keeps_new_posts_once(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "posts.jsonl"
            pages = [listing([post("Costco layoffs", pid="1", sub="Costco"), post("unrelated", pid="2")]),
                     listing([post("Costco layoffs", pid="1", sub="Costco"), post("COST to the moon", pid="3")])]
            op = FakeOpener(pages)
            c = reddit.Client(opener=op, sleep=lambda s: None)
            r1 = reddit.collect(["COST"], {"COST": "Costco"}, c, pages=1, p=p, log=None)
            r2 = reddit.collect(["COST"], {"COST": "Costco"}, c, pages=1, p=p, log=None)
            rows = reddit.load(p)
        self.assertEqual((r1["kept"], r2["kept"]), (1, 1))
        self.assertEqual(sorted(r["id"] for r in rows), ["1", "3"])


class Cards(unittest.TestCase):
    def test_show_and_top(self):
        asof = datetime(2026, 11, 27, 12, tzinfo=UTC)
        rows = [{"id": "1", "t": "COST", "sub": "costco", "created": "2026-11-26T10:00:00Z", "score": 5,
                 "comments": 2, "kind": "other", "topics": ["layoffs"], "fetched_at": "x"},
                {"id": "2", "t": "COST", "sub": "stocks", "created": "2026-11-25T10:00:00Z", "score": 1,
                 "comments": 0, "kind": "investing", "topics": [], "fetched_at": "x"},
                {"id": "3", "t": "AMD", "sub": "stocks", "created": "2026-10-01T10:00:00Z", "score": 1,
                 "comments": 0, "kind": "investing", "topics": ["delay"], "fetched_at": "x"}]
        text = "\n".join(reddit.show(rows, "COST", 30, asof))
        self.assertIn("2 posts name it (1 in investing forums, 1 elsewhere)", text)
        self.assertIn("layoffs 1", text)
        self.assertIn("https://www.reddit.com/comments/1", text)
        self.assertIn("no person is followed", text)
        self.assertIn("no post kept", reddit.show(rows, "AMD", 7, asof)[0])
        t = "\n".join(reddit.top(rows, 7, asof=asof))
        self.assertIn("COST", t)
        self.assertNotIn("AMD", t)


if __name__ == "__main__":
    unittest.main()
