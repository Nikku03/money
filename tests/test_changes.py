from brain import changes
from brain.bank import Fact
from tests.util import TempHome

OLD = ("<html><body><p>We sell chips to many customers. Our largest customer is a cloud provider in the United States. "
       "For the three months ended July 28, 2024, revenue was $30.0 billion. We face competition from many firms in "
       "every market we serve.</p></body></html>")
NEW = ("<html><body><p>We sell chips to many customers. Our largest customer is a cloud provider in the United States. "
       "For the three months ended July 27, 2025, revenue was $46.7 billion. We face competition from many firms in "
       "every market we serve. There is substantial doubt about our ability to continue as a going concern this year."
       "</p></body></html>")


def report(form, as_of, known_at, accn):
    return Fact(entity="ACME", attr="filing", value=form, as_of=as_of, known_at=known_at, tier="filed",
                source={"name": "SEC EDGAR", "url": f"https://www.sec.gov/Archives/{accn}.htm"}, context={"accn": accn})


class TextTest(TempHome):
    def test_sentences_and_their_normal_form(self):
        from brain.graph import filing_text
        s = changes.sentences(filing_text(NEW))
        self.assertEqual(len(s), 5)
        self.assertEqual(changes.norm_sentence("For the three months ended July 27, 2025, revenue was $46.7 billion."),
                         changes.norm_sentence("For the three months ended April 28, 2024, revenue was $30.0 billion."))

    def test_similarity_measures(self):
        a = changes.word_counts(["The chips sell well", "chips again"])
        self.assertNotIn("the", a)
        self.assertAlmostEqual(changes.cosine(a, a), 1.0)
        self.assertEqual(changes.cosine(a, changes.word_counts(["Oranges grow"])), 0.0)
        self.assertEqual(changes.jaccard(a, a), 1.0)
        from brain.graph import filing_text
        new, old = changes.sentences(filing_text(NEW)), changes.sentences(filing_text(OLD))
        cmp = changes.compare(new, old)
        self.assertLess(cmp["cosine"], 1.0)
        self.assertAlmostEqual(cmp["new_share"], 1 / 5)   # only the going-concern sentence is new
        self.assertEqual(changes.new_sentences(new, old), [new[-1]])


class PairTest(TempHome):
    def test_each_report_meets_the_same_report_a_year_earlier(self):
        fs = [report("10-K", "2023-12-31", "2024-02-20", "k23"), report("10-Q", "2024-03-31", "2024-05-01", "q124"),
              report("10-Q", "2024-06-30", "2024-08-01", "q224"), report("10-K", "2024-12-31", "2025-02-20", "k24"),
              report("10-Q", "2025-03-31", "2025-05-01", "q125"), report("10-Q", "2025-06-30", "2025-08-01", "q225"),
              report("10-K/A", "2024-12-31", "2025-04-01", "ka24")]
        self.bank.add_many(fs)
        got = [(n.context["accn"], o.context["accn"]) for n, o in changes.pairs(changes.reports(self.bank, "ACME"))]
        self.assertEqual(got, [("k24", "k23"), ("q125", "q124"), ("q225", "q224")])

    def test_an_earlier_report_filed_later_is_not_a_pair(self):
        fs = [report("10-K", "2023-12-31", "2025-06-01", "late"), report("10-K", "2024-12-31", "2025-02-20", "k24")]
        self.assertEqual(changes.pairs(sorted(fs, key=lambda f: f.as_of)), [])


class BuildTest(TempHome):
    def setUp(self):
        super().setUp()
        self.old = report("10-Q", "2024-07-28", "2024-08-28T20:00:00Z", "q24")
        self.new = report("10-Q", "2025-07-27", "2025-08-27T20:00:00Z", "q25")
        self.bank.add_many([self.old, self.new])
        self.docs = {self.old.source["url"]: OLD.encode(), self.new.source["url"]: NEW.encode()}
        self.fetched = []

    def fetch(self, url):
        self.fetched.append(url)
        return self.docs[url]

    def test_build_stores_a_cited_point_in_time_fact(self):
        out = changes.build(self.bank, "acme", fetch_doc=self.fetch, log=None)
        self.assertEqual(out, {"compared": 1, "failed": []})
        f = self.bank.facts("ACME", changes.ATTR)[0]
        self.assertEqual((f.known_at, f.as_of, f.tier, f.inputs), (self.new.known_at, self.new.as_of, "derived",
                                                                   (self.new.ref, self.old.ref)))
        self.assertEqual((f.context["form"], f.context["prev_accn"], f.context["method"]), ("10-Q", "q24", changes.METHOD))
        from brain.check import run
        self.assertEqual(run(self.bank)[0], [])

    def test_documents_are_fetched_once_and_pairs_compared_once(self):
        changes.build(self.bank, "ACME", fetch_doc=self.fetch, log=None)
        self.assertEqual(changes.build(self.bank, "ACME", fetch_doc=self.fetch, log=None)["compared"], 0)
        self.assertEqual(len(self.fetched), 2)

    def test_latest_sees_only_what_was_public(self):
        changes.build(self.bank, "ACME", fetch_doc=self.fetch, log=None)
        self.assertIsNone(changes.latest(self.bank.view("2025-08-27T19:59:59Z"), "ACME"))
        self.assertIsNotNone(changes.latest(self.bank.view("2025-09-30"), "ACME"))
        self.assertIsNone(changes.latest(self.bank.view("2026-12-31"), "ACME"))   # too old to speak for it

    def test_a_document_with_no_text_is_reported(self):
        self.docs[self.new.source["url"]] = b"<html><body><p>x</p></body></html>"
        out = changes.build(self.bank, "ACME", fetch_doc=self.fetch, log=None)
        self.assertEqual((out["compared"], out["failed"]), (0, ["q25: no text"]))
