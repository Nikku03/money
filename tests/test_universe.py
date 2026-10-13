import json
import unittest
from pathlib import Path
from unittest import mock

from brain import universe
from brain.bank import Bank, Fact
from brain.net import FetchError
from brain.sources import sec
from tests.util import TempHome, filed


def _cols(rows):
    keys = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "primaryDocument",
            "primaryDocDescription", "items"]
    return {k: [r.get(k, "") for r in rows] for k in keys}


class FilingPagesTest(unittest.TestCase):
    def test_older_pages_are_read_and_have_the_same_shape(self):
        recent = _cols([{"accessionNumber": "0001-26-000002", "filingDate": "2026-01-30", "reportDate": "2026-01-29",
                         "acceptanceDateTime": "2026-01-30T02:30:33.000Z", "form": "8-K", "primaryDocument": "r.htm",
                         "items": "2.02,9.01"}])
        older = _cols([{"accessionNumber": "0001-01-000001", "filingDate": "2001-04-20", "reportDate": "",
                        "acceptanceDateTime": "", "form": "8-K", "primaryDocument": "", "items": "5,7"},
                       {"accessionNumber": "0001-01-000000", "filingDate": "2001-02-01", "reportDate": "2000-12-31",
                        "acceptanceDateTime": "2001-02-01T16:00:00.000Z", "form": "10-K", "primaryDocument": "k.txt"}])
        data = {"cik": 7, "name": "Co", "filings": {"recent": recent, "files": [
            {"name": "CIK0000000007-submissions-001.json", "filingCount": 2},
            {"name": "../../evil.json"}, {"name": "https://elsewhere.test/x.json"}]}}
        calls = []

        def fake(url, ttl=86400):
            calls.append((url, ttl))
            return (data if url == sec.SUBS_URL.format(cik=7) else older), "2026-10-13T00:00:00Z"

        with mock.patch.object(sec, "cik_for", return_value=7), mock.patch.object(sec, "_get_json", side_effect=fake):
            fil, ident = sec.filings_for("T")
            recent_only, _ = sec.filings_for("T", all_pages=False)
        self.assertEqual([u for u, _ in calls], [sec.SUBS_URL.format(cik=7),
                                                 "https://data.sec.gov/submissions/CIK0000000007-submissions-001.json",
                                                 sec.SUBS_URL.format(cik=7)])
        self.assertGreaterEqual(calls[1][1], 7 * 86400)  # older pages: a week-old copy is fine
        self.assertEqual([f.context["accn"] for f in fil], ["0001-26-000002", "0001-01-000001", "0001-01-000000"])
        self.assertEqual(len(recent_only), 1)
        old8k = fil[1]
        self.assertEqual((old8k.attr, old8k.value, old8k.as_of, old8k.known_at), ("filing", "8-K", "2001-04-20",
                                                                                   "2001-04-20T23:59:59Z"))
        self.assertEqual(old8k.context, {"accn": "0001-01-000001", "items": "5,7"})
        self.assertEqual(old8k.source["url"], "https://www.sec.gov/Archives/edgar/data/7/000101000001/")
        self.assertEqual(fil[2].as_of, "2000-12-31")
        self.assertEqual(fil[2].known_at, "2001-02-01T16:00:00Z")
        self.assertIn("identity.name", {f.attr for f in ident})

    def test_recent_block_parses_exactly_as_before(self):
        # The refactor into pages must not change a single fact id already in banks.
        data = {"cik": 2, "filings": {"recent": _cols([
            {"accessionNumber": "0001-24-1", "filingDate": "2024-05-01", "reportDate": "2024-03-31",
             "acceptanceDateTime": "2024-05-01T20:15:00.000Z", "form": "10-Q", "primaryDocument": "q.htm",
             "primaryDocDescription": "10-Q", "items": ""}])}}
        fs, _ = sec.parse_submissions("T", data, "2026-01-01T00:00:00Z")
        want = Fact(entity="T", attr="filing", value="10-Q", as_of="2024-03-31", known_at="2024-05-01T20:15:00.000Z",
                    tier="filed", source={"name": "SEC EDGAR", "url": "https://www.sec.gov/Archives/edgar/data/2/0001241/q.htm"},
                    context={"accn": "0001-24-1", "description": "10-Q"})
        self.assertEqual(fs[0].id, want.id)

    def test_new_concepts_are_collected(self):
        for c in ("ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic",
                  "WeightedAverageNumberOfShareOutstandingBasicAndDiluted", "CostOfGoodsAndServicesSold"):
            self.assertIn(c, sec.CONCEPTS["us-gaap"])
        self.assertEqual(sec.CONCEPTS["us-gaap"][:3], ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                                                       "SalesRevenueNet"])


class BuildBroadTest(TempHome):
    def setUp(self):
        super().setUp()
        self.rows = [{"ticker": "AAA", "cik": 1, "title": "A"}, {"ticker": "AAB", "cik": 1, "title": "A class B"},
                     {"ticker": "FOR", "cik": 2, "title": "Foreign"}, {"ticker": "NEW", "cik": 3, "title": "New"},
                     {"ticker": "BBB", "cik": 4, "title": "B"}, {"ticker": "ERR", "cik": 5, "title": "Err"},
                     {"ticker": "CCC", "cik": 6, "title": "C"}]

    def _facts(self, t):
        if t == "FOR":
            return [filed(t, "ifrs:Profit", 1.0, "2020-12-31", "2021-03-01", form="20-F")]
        if t == "ERR":
            raise FetchError("HTTP 404")
        return [filed(t, "us-gaap:NetIncomeLoss", 5.0, "2020-12-31", "2021-02-01", form="10-K", start="2020-01-01",
                      duration_days=365),
                filed(t, "us-gaap:Assets", 50.0, "2020-12-31", "2021-02-01", form="10-K")]

    def _px(self, t):
        first = "2015-06-01" if t == "NEW" else "2005-01-03"
        return [Fact(entity=t, attr="series.price_daily", value="0" * 64, as_of="2026-10-09", known_at="2026-10-09",
                     tier="market", source={"name": "test", "url": "https://example.test/px"},
                     context={"first": first, "last": "2026-10-09", "splits": []})]

    def _filings(self, t, all_pages=True, ttl=86400):
        self.pages.append((t, all_pages))
        return ([Fact(entity=t, attr="filing", value="8-K", as_of=d, known_at=d, tier="filed",
                      source={"name": "SEC EDGAR", "url": f"https://example.test/{t}/{d}"}, context={"accn": d})
                 for d in ("2026-01-30", "2001-04-20")],
                [Fact(entity=t, attr="identity.name", value=t, as_of="2026-10-13", known_at="2026-10-13",
                      tier="filed", source={"name": "SEC EDGAR submissions", "url": "https://example.test/s"})])

    def test_rule_rank_cutoff_and_output(self):
        self.pages = []
        out_path = Path(self._tmp.name) / "us_broad.json"
        logs = []
        with mock.patch.object(sec, "ticker_table", return_value=self.rows), \
             mock.patch.object(sec, "facts_for", side_effect=self._facts), \
             mock.patch.object(universe.prices, "ingest", side_effect=self._px), \
             mock.patch.object(sec, "filings_for", side_effect=self._filings):
            out = universe.build_broad(self.bank, max_rank=6, log=logs.append, path=out_path)
        self.assertEqual(out["tickers"], ["AAA", "BBB"])  # CCC is rank 7, past the cutoff; AAB repeats CIK 1
        reasons = {r["ticker"]: r["reason"] for r in out["rejected"]}
        self.assertEqual(set(reasons), {"FOR", "NEW", "ERR"})
        self.assertIn("no 10-K", reasons["FOR"])
        self.assertIn("prices start 2015-06-01", reasons["NEW"])
        self.assertIn("no SEC companyfacts", reasons["ERR"])
        saved = json.loads(out_path.read_text())
        self.assertEqual(saved, out)
        frozen = json.loads((universe.config.ROOT / "universe" / "us_large.json").read_text())
        self.assertEqual(set(saved) - {"max_rank"}, set(frozen))  # same fields as us_large.json
        self.assertIn("first 6 rows", saved["rule"])
        self.assertEqual(self.pages, [("AAA", True), ("BBB", True)])  # every filing page, for kept tickers only
        b = Bank(self.bank.root)
        self.assertEqual(len(b.facts("AAA", "filing")), 2)
        self.assertEqual(len(b.facts("BBB", "us-gaap:Assets")), 1)
        self.assertEqual(b.facts("FOR", "filing"), [])
        self.assertEqual(len(logs), 5)

    def test_default_rule_text_and_path(self):
        self.assertIn("first 1500 rows", universe.broad_rule_text(1500))
        self.assertEqual(universe.BROAD.name, "us_broad.json")
        self.assertEqual(universe.BROAD_RULE_SHORT, "SEC ranks 1-1500 with 10-K XBRL and prices since 2010")


class CliPluginTest(unittest.TestCase):
    def test_missing_and_broken_plugins_do_not_break_the_cli(self):
        import argparse
        import sys
        import types
        from brain import cli
        good = types.ModuleType("brain_test_good")
        good.add_cli = lambda sub: sub.add_parser("goodcmd").set_defaults(fn=lambda a: 7)
        bad = types.ModuleType("brain_test_bad")

        def boom(sub):
            raise RuntimeError("half-written")
        bad.add_cli = boom
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd")
        with mock.patch.dict(sys.modules, {"brain_test_good": good, "brain_test_bad": bad}), \
             mock.patch("sys.stderr"):
            done = cli.add_plugins(sub, ("brain_test_good", "brain_test_bad", "brain_test_absent_module"))
        self.assertEqual(done, ["brain_test_good"])
        self.assertEqual(p.parse_args(["goodcmd"]).fn(None), 7)


if __name__ == "__main__":
    unittest.main()


class TransientTest(unittest.TestCase):
    def test_a_throttled_source_is_asked_again_not_taken_as_a_rejection(self):
        calls = []

        def facts(t):
            calls.append(t)
            if len(calls) < 3:
                raise FetchError("https://x: gave up after 4 tries (HTTP Error 429)")
            return [filed(t, "us-gaap:NetIncomeLoss", 1.0, "2020-12-31", "2021-02-01", form="10-K"),
                    filed(t, "us-gaap:Assets", 1.0, "2020-12-31", "2021-02-01", form="10-K")]
        px = [Fact(entity="T", attr="series.price_daily", value="0" * 64, as_of="2026-10-09", known_at="2026-10-09",
                   tier="market", source={"name": "t", "url": "https://example.test/px"}, context={"first": "2001-01-02"})]
        slept = []
        with mock.patch.object(sec, "facts_for", side_effect=facts), \
             mock.patch.object(universe.prices, "ingest", return_value=px):
            f, p = universe._qualify_patiently("T", waits=(1, 2, 3), log=lambda m: None, sleep=slept.append)
        self.assertEqual((len(calls), slept), (3, [1, 2]))
        with mock.patch.object(sec, "facts_for", side_effect=FetchError("https://x: HTTP 404")):
            with self.assertRaises(universe.Rejected) as cm:
                universe._qualify_patiently("T", waits=(1,), log=lambda m: None, sleep=slept.append)
        self.assertFalse(cm.exception.transient)
        self.assertEqual(slept, [1, 2])  # a 404 is an answer, not a reason to wait
