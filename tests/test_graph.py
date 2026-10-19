import json

from brain import graph
from brain.bank import Fact
from tests.util import TempHome

TABLE = [{"ticker": "AAPL", "cik": 320193, "title": "Apple Inc."},
         {"ticker": "TSM", "cik": 1046179, "title": "TAIWAN SEMICONDUCTOR MANUFACTURING CO LTD"},
         {"ticker": "GOOGL", "cik": 1652044, "title": "Alphabet Inc."},
         {"ticker": "GOOG", "cik": 1652044, "title": "Alphabet Inc."},
         {"ticker": "AMD", "cik": 2488, "title": "ADVANCED MICRO DEVICES INC"},
         {"ticker": "CHIP", "cik": 1, "title": "Chipco Corp /DE/"}]

DOC_2015 = """<html><head><title>x</title></head><body>
<p>Item 1. Business. We design graphics processors for gaming, data centers and automobiles around the world.</p>
<p>Sales to Apple Inc. accounted for 14% of our net revenue in fiscal 2015, and sales to one distributor
accounted for 11%.</p>
<p>We do not own fabrication facilities. Taiwan Semiconductor Manufacturing Company Limited manufactures
substantially all of our wafers.</p>
<p>Our competitors include Advanced Micro Devices, Inc. and several smaller companies.</p>
<p>The weather in Santa Clara was pleasant all year and nothing here names a relation of any kind at all.</p>
</body></html>"""

DOC_2016 = """<html><body>
<p>Our competitors include Advanced Micro Devices, Inc. and Intel Corporation, among other companies.</p>
</body></html>"""

ANSWER_2015 = {"relations": [
    {"passage": 1, "name": "Apple Inc.", "relation": "customer", "share_pct": 14,
     "quote": "Sales to Apple Inc. accounted for 14% of our net revenue"},
    {"passage": 1, "name": "one distributor", "relation": "customer", "share_pct": 11,
     "quote": "sales to one distributor accounted for 11%"},
    {"passage": 2, "name": "Taiwan Semiconductor Manufacturing Company Limited", "relation": "supplier",
     "share_pct": None, "quote": "Taiwan Semiconductor Manufacturing Company Limited manufactures substantially all of our wafers"},
    {"passage": 3, "name": "Advanced Micro Devices, Inc.", "relation": "competitor", "share_pct": None,
     "quote": "Our competitors include Advanced Micro Devices, Inc."},
    # Remembered, not written: Samsung is in no passage.
    {"passage": 2, "name": "Samsung Electronics", "relation": "supplier", "share_pct": None,
     "quote": "Samsung Electronics manufactures our memory"},
    {"passage": 9, "name": "Google", "relation": "customer", "share_pct": None, "quote": "Google"},
]}

ANSWER_2016 = {"relations": [
    {"passage": 1, "name": "Advanced Micro Devices, Inc.", "relation": "competitor", "share_pct": None,
     "quote": "Our competitors include Advanced Micro Devices, Inc. and Intel Corporation"},
    {"passage": 1, "name": "Intel Corporation", "relation": "competitor", "share_pct": None,
     "quote": "Advanced Micro Devices, Inc. and Intel Corporation"},
]}


class FakeClient:
    model = "fake-reader"

    def __init__(self, answers):
        self.answers, self.calls = answers, 0

    def complete(self, messages, **kw):
        self.calls += 1
        text = messages[1]["content"]
        key = "2016" if "Intel Corporation" in text else "2015"
        return {"choices": [{"message": {"content": "<think>reading</think>" + json.dumps(self.answers[key])}}]}


def tenk(ticker, as_of, known_at, accn, doc):
    return Fact(entity=ticker, attr="filing", value="10-K", as_of=as_of, known_at=known_at, tier="filed",
                source={"name": "SEC EDGAR", "url": f"https://www.sec.gov/Archives/{accn}/{doc}"},
                context={"accn": accn})


DOCS = {"https://www.sec.gov/Archives/0001-15-000001/a2015.htm": DOC_2015.encode(),
        "https://www.sec.gov/Archives/0001-16-000001/a2016.htm": DOC_2016.encode()}


class TextTest(TempHome):
    def test_filing_text_and_passages(self):
        paras = graph.filing_text(DOC_2015)
        self.assertFalse(any("<" in p for p in paras))
        ps = graph.passages(paras)
        self.assertEqual(len(ps), 3)
        self.assertTrue(ps[0].startswith("Sales to Apple Inc. accounted for 14%"))
        self.assertFalse(any("weather" in p for p in ps))
        self.assertEqual(len(graph.passages(paras, limit=1)), 1)

    def test_plain_text_filings_join_wrapped_lines(self):
        doc = ("ITEM 1. BUSINESS\n\nSales to Apple Inc. accounted for 14% of our net\nrevenue in fiscal 1999, "
               "more than any other customer.\n\nNothing else.")
        self.assertIn("Sales to Apple Inc. accounted for 14% of our net revenue in fiscal 1999, more than any other "
                      "customer.", graph.filing_text(doc))

    def test_long_paragraph_is_cut_at_sentences(self):
        long = " ".join(["Our customers buy many products from us every single year."] * 60)
        pieces = graph._pieces(long)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p) <= graph.MAX_CHARS for p in pieces))

    def test_names_resolve_against_sec_titles(self):
        r = graph.Resolver(TABLE)
        self.assertEqual(r.resolve("Apple Inc."), "AAPL")
        self.assertEqual(r.resolve("Taiwan Semiconductor Manufacturing Company Limited"), "TSM")
        self.assertEqual(r.resolve("Alphabet"), "GOOGL")
        self.assertEqual(r.resolve("Chipco Corporation"), "CHIP")
        self.assertIsNone(r.resolve("TSMC"))
        self.assertEqual(graph.node("Intel Corporation", r), "?intel")


class CheckTest(TempHome):
    P = "Sales to Apple Inc. accounted for 14% of our net revenue in fiscal 2015."

    def test_grounded_relation_is_kept(self):
        rel, why = graph.check("Apple Inc.", "Customer", "Sales to  Apple Inc. accounted for 14%", self.P, "14%")
        self.assertEqual(why, "")
        self.assertEqual((rel["relation"], rel["share_pct"]), ("customer", 14.0))

    def test_ungrounded_relations_are_dropped(self):
        cases = [("Samsung", "supplier", "Samsung supplies memory", "quote not in the passage"),
                 ("Apple Inc.", "customer", "accounted for 14% of our net revenue", "name not in the quote"),
                 ("one customer", "customer", "accounted for 14%", "no company name"),
                 ("Customer A", "customer", "accounted for 14%", "no company name"),
                 ("our largest customer", "customer", "accounted for 14%", "no company name"),
                 ("Apple Inc.", "friend", "Sales to Apple Inc.", "relation is not customer, supplier, competitor or partner")]
        for name, kind, quote, why in cases:
            self.assertEqual(graph.check(name, kind, quote, self.P), (None, why), name)

    def test_share_must_be_written_in_the_quote(self):
        q = "Sales to Apple Inc. accounted for 14%"
        self.assertIsNone(graph.check("Apple Inc.", "customer", q, self.P, 20)[0]["share_pct"])
        self.assertIsNone(graph.check("Apple Inc.", "customer", "Sales to Apple Inc.", self.P, 14)[0]["share_pct"])
        self.assertIsNone(graph.check("Apple Inc.", "customer", q, self.P, 1.4)[0]["share_pct"])

    def test_parse_reports_why(self):
        rels, rej = graph.parse(json.dumps(ANSWER_2015), graph.passages(graph.filing_text(DOC_2015)))
        self.assertEqual([r["name"] for r in rels], ["Apple Inc.", "Taiwan Semiconductor Manufacturing Company Limited",
                                                    "Advanced Micro Devices, Inc."])
        self.assertEqual(rej, {"no company name": 1, "quote not in the passage": 1, "no such passage": 1})
        self.assertEqual(graph.parse("no idea", ["x"]), ([], {"no relations list in the answer": 1}))


class ExtractTest(TempHome):
    def setUp(self):
        super().setUp()
        self.f15 = tenk("NVDA", "2015-01-25", "2015-03-12T16:00:00Z", "0001-15-000001", "a2015.htm")
        self.f16 = tenk("NVDA", "2016-01-31", "2016-03-17T16:00:00Z", "0001-16-000001", "a2016.htm")
        self.bank.add_many([self.f15, self.f16])
        self.resolver = graph.Resolver(TABLE)
        self.client = FakeClient({"2015": ANSWER_2015, "2016": ANSWER_2016})

    def run_extract(self, **kw):
        return graph.extract(self.bank, "nvda", self.client, self.resolver, fetch_doc=DOCS.__getitem__, log=None, **kw)

    def test_facts_carry_the_filing_time_and_cite_it(self):
        out = self.run_extract()
        self.assertEqual(out, {"filings": 2, "relations": 5, "failed": []})
        rels = self.bank.facts("NVDA", "relation.customer")
        self.assertEqual([(f.value, f.context["share_pct"]) for f in rels], [("AAPL", 14.0)])
        f = rels[0]
        self.assertEqual((f.known_at, f.as_of, f.tier, f.inputs), (self.f15.known_at, self.f15.as_of, "derived",
                                                                   (self.f15.ref,)))
        self.assertIn("Apple Inc. accounted for 14%", f.context["quote"])
        scan = [s for s in self.bank.facts("NVDA", "relation.scan") if s.context["accn"] == "0001-15-000001"][0]
        self.assertEqual((scan.value, scan.context["passages"]), (3, 3))
        self.assertEqual(scan.context["rejected"], {"no company name": 1, "no such passage": 1,
                                                    "quote not in the passage": 1})

    def test_bank_invariants_hold(self):
        from brain.check import run
        self.run_extract()
        errors, _ = run(self.bank)
        self.assertEqual(errors, [])

    def test_rerun_reads_nothing_twice(self):
        self.run_extract()
        calls = self.client.calls
        self.assertEqual(self.run_extract(), {"filings": 0, "relations": 0, "failed": []})
        self.assertEqual(self.client.calls, calls)

    def test_max_filings_takes_the_latest(self):
        self.assertEqual(self.run_extract(max_filings=1)["filings"], 1)
        self.assertEqual({s.context["accn"] for s in self.bank.facts("NVDA", "relation.scan")}, {"0001-16-000001"})

    def test_a_failed_fetch_is_reported_and_the_rest_read(self):
        docs = {k: v for k, v in DOCS.items() if "2016" in k}
        out = graph.extract(self.bank, "NVDA", self.client, self.resolver, fetch_doc=docs.__getitem__, log=None)
        self.assertEqual((out["filings"], len(out["failed"])), (1, 1))

    def test_the_graph_sees_only_what_was_filed_by_t(self):
        self.run_extract()
        self.assertEqual(graph.graph_at(self.bank, ["NVDA"], "2015-03-12T15:59:59Z"), [])
        g15 = graph.graph_at(self.bank, ["NVDA"], "2015-06-30")
        self.assertEqual(sorted((e.dst, e.relation) for e in g15),
                         [("AAPL", "customer"), ("AMD", "competitor"), ("TSM", "supplier")])
        # The 2016 10-K replaces the 2015 one: Apple and TSMC are no longer named.
        g16 = graph.graph_at(self.bank, ["NVDA"], "2016-06-30")
        self.assertEqual(sorted((e.dst, e.relation) for e in g16), [("?intel", "competitor"), ("AMD", "competitor")])
        # A 10-K that is too old speaks for nobody.
        self.assertEqual(graph.graph_at(self.bank, ["NVDA"], "2018-06-30"), [])

    def test_later_facts_change_nothing_at_an_earlier_time(self):
        """The leak test: whatever is added known after t, the graph at t is the same."""
        self.bank.add(self.f15)
        graph.extract(self.bank, "NVDA", self.client, self.resolver, fetch_doc=DOCS.__getitem__, log=None,
                      max_filings=0)
        self.bank.add_many(graph.read_filing(self.f15, self.client, self.resolver, "NVIDIA", DOCS.__getitem__))
        before = graph.graph_at(self.bank, ["NVDA"], "2015-12-31")
        self.bank.add_many(graph.read_filing(self.f16, self.client, self.resolver, "NVIDIA", DOCS.__getitem__))
        self.bank.add(Fact(entity="NVDA", attr="relation.customer", value="GOOGL", as_of="2015-01-25",
                           known_at="2017-01-01", tier="derived", source={"name": "x", "url": "https://x.test"},
                           context={"accn": "0001-15-000001", "extractor": "agent"}))
        self.assertEqual(graph.graph_at(self.bank, ["NVDA"], "2015-12-31"), before)

    def test_neighbours_turn_edges_round(self):
        self.run_extract()
        edges = graph.graph_at(self.bank, ["NVDA"], "2015-06-30")
        roles = {(o, r) for o, r, _ in graph.neighbours(edges, "AAPL")}
        self.assertEqual(roles, {("NVDA", "supplier")})
        self.assertEqual({(o, r) for o, r, _ in graph.neighbours(edges, "TSM")}, {("NVDA", "customer")})

    def test_an_agent_is_held_to_the_same_rule(self):
        with self.assertRaisesRegex(ValueError, "quote not in the passage"):
            graph.add_by_agent(self.bank, "NVDA", "0001-15-000001", "Samsung", "supplier", "Samsung makes our memory",
                               resolver=self.resolver, fetch_doc=DOCS.__getitem__)
        with self.assertRaisesRegex(ValueError, "no 10-K"):
            graph.add_by_agent(self.bank, "NVDA", "9999", "Apple Inc.", "customer", "Apple Inc.",
                               fetch_doc=DOCS.__getitem__)
        f = graph.add_by_agent(self.bank, "NVDA", "0001-15-000001", "Apple Inc.", "customer",
                               "Sales to Apple Inc. accounted for 14% of our net revenue", "14",
                               resolver=self.resolver, fetch_doc=DOCS.__getitem__)
        self.assertEqual((f.value, f.context["share_pct"], f.context["extractor"]), ("AAPL", 14.0, "agent"))
        # With no model scan of that 10-K, the agent's reading alone puts it in the graph.
        g = graph.graph_at(self.bank, ["NVDA"], "2015-06-30")
        self.assertEqual([(e.dst, e.relation) for e in g], [("AAPL", "customer")])


class AgentPacketTest(TempHome):
    def setUp(self):
        super().setUp()
        self.f15 = tenk("NVDA", "2015-01-25", "2015-03-12T16:00:00Z", "0001-15-000001", "a2015.htm")
        self.bank.add(self.f15)
        self.resolver = graph.Resolver(TABLE)

    def test_packet_shows_what_the_model_would_see(self):
        pk = graph.packet(self.f15, "NVIDIA CORP", DOCS.__getitem__)
        self.assertEqual((pk["ticker"], pk["accn"], pk["prompt_sha"]), ("NVDA", "0001-15-000001", graph.prompt_sha()))
        self.assertEqual([p["text"] for p in pk["passages"]], graph.passages(graph.filing_text(DOC_2015)))
        self.assertIn("NVIDIA CORP's customer, supplier, competitor or partner", pk["instructions"])
        self.assertNotIn("{", pk["instructions"].split("Answer with JSON")[0])

    def test_answer_is_held_to_the_model_rules(self):
        answer = dict(ANSWER_2015, ticker="nvda", accn="0001-15-000001")
        facts = graph.record_answer(self.bank, json.dumps(answer), self.resolver, fetch_doc=DOCS.__getitem__)
        self.assertEqual(sorted(f.value for f in facts[:-1]), ["AAPL", "AMD", "TSM"])
        self.assertEqual({f.context["extractor"] for f in facts}, {"agent"})
        scan = facts[-1]
        self.assertEqual((scan.attr, scan.value), ("relation.scan", 3))
        self.assertEqual(scan.context["rejected"], {"no company name": 1, "no such passage": 1,
                                                    "quote not in the passage": 1})
        self.assertEqual(len(graph.graph_at(self.bank, ["NVDA"], "2015-06-30")), 3)
        with self.assertRaisesRegex(ValueError, "already read"):
            graph.record_answer(self.bank, answer, self.resolver, fetch_doc=DOCS.__getitem__)

    def test_answer_for_an_unknown_filing_is_refused(self):
        for bad in ({"ticker": "NVDA", "accn": "nope", "relations": []}, {"relations": []}, []):
            with self.assertRaises(ValueError):
                graph.record_answer(self.bank, bad, self.resolver, fetch_doc=DOCS.__getitem__)

    def test_a_read_with_nothing_named_still_counts_as_read(self):
        facts = graph.record_answer(self.bank, {"ticker": "NVDA", "accn": "0001-15-000001", "relations": []},
                                    self.resolver, fetch_doc=DOCS.__getitem__)
        self.assertEqual([(f.attr, f.value) for f in facts], [("relation.scan", 0)])
        self.assertIsNotNone(graph.latest_scan(self.bank.view("2015-06-30"), "NVDA"))
        self.assertEqual(graph.graph_at(self.bank, ["NVDA"], "2015-06-30"), [])
