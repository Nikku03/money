import io
import json
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from brain.bank import Fact
from brain.net import FetchError, Response
from brain.sources import insiders
from tests.util import TempHome

# Made-up people and companies: the fixtures mirror the data sets' layout, not anyone's trades.
SUB = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DATE_OF_ORIG_SUB", "NO_SECURITIES_OWNED",
       "NOT_SUBJECT_SEC16", "FORM3_HOLDINGS_REPORTED", "FORM4_TRANS_REPORTED", "DOCUMENT_TYPE", "ISSUERCIK",
       "ISSUERNAME", "ISSUERTRADINGSYMBOL", "REMARKS", "AFF10B5ONE"]
OWN = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE", "RPTOWNER_TXT",
       "RPTOWNER_STREET1", "RPTOWNER_STREET2", "RPTOWNER_CITY", "RPTOWNER_STATE", "RPTOWNER_ZIPCODE",
       "RPTOWNER_STATE_DESC", "FILE_NUMBER"]
TRN = ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE", "SECURITY_TITLE_FN", "TRANS_DATE", "TRANS_DATE_FN",
       "DEEMED_EXECUTION_DATE", "DEEMED_EXECUTION_DATE_FN", "TRANS_FORM_TYPE", "TRANS_CODE", "EQUITY_SWAP_INVOLVED",
       "EQUITY_SWAP_TRANS_CD_FN", "TRANS_TIMELINESS", "TRANS_TIMELINESS_FN", "TRANS_SHARES", "TRANS_SHARES_FN",
       "TRANS_PRICEPERSHARE", "TRANS_PRICEPERSHARE_FN", "TRANS_ACQUIRED_DISP_CD", "TRANS_ACQUIRED_DISP_CD_FN",
       "SHRS_OWND_FOLWNG_TRANS", "SHRS_OWND_FOLWNG_TRANS_FN", "VALU_OWND_FOLWNG_TRANS", "VALU_OWND_FOLWNG_TRANS_FN",
       "DIRECT_INDIRECT_OWNERSHIP", "DIRECT_INDIRECT_OWNERSHIP_FN", "NATURE_OF_OWNERSHIP", "NATURE_OF_OWNERSHIP_FN"]


def tsv(cols, rows):
    lines = ["\t".join(cols)] + ["\t".join(str(r.get(c, "")) for c in cols) for r in rows]
    return ("\n".join(lines) + "\n").encode()


def dataset(subs, owners, trans) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", tsv(SUB, subs))
        z.writestr("REPORTINGOWNER.tsv", tsv(OWN, owners))
        z.writestr("NONDERIV_TRANS.tsv", tsv(TRN, trans))
    return buf.getvalue()


def sample_zip() -> bytes:
    subs = [
        {"ACCESSION_NUMBER": "0000000001-24-000001", "FILING_DATE": "03-APR-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "0000000111", "ISSUERNAME": "Example Corp", "ISSUERTRADINGSYMBOL": "EXC"},
        {"ACCESSION_NUMBER": "0000000001-24-000002", "FILING_DATE": "10-APR-2024", "DOCUMENT_TYPE": "4/A",
         "ISSUERCIK": "0000000111"},                                    # amendment: skipped
        {"ACCESSION_NUMBER": "0000000002-24-000003", "FILING_DATE": "05-APR-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "0000000999"},                                    # issuer not asked for
        {"ACCESSION_NUMBER": "0000000001-24-000004", "FILING_DATE": "12-APR-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "0000000111"},                                    # joint filers
    ]
    owners = [
        {"ACCESSION_NUMBER": "0000000001-24-000001", "RPTOWNERCIK": "0000000501", "RPTOWNERNAME": "Doe Jane",
         "RPTOWNER_RELATIONSHIP": "Director,Officer", "RPTOWNER_TITLE": "Chief Executive Officer",
         "RPTOWNER_STREET1": "1 Private Lane", "RPTOWNER_CITY": "Somewhere"},
        {"ACCESSION_NUMBER": "0000000001-24-000002", "RPTOWNERCIK": "0000000501", "RPTOWNERNAME": "Doe Jane",
         "RPTOWNER_RELATIONSHIP": "Director,Officer"},
        {"ACCESSION_NUMBER": "0000000001-24-000004", "RPTOWNERCIK": "0000000601", "RPTOWNERNAME": "Example Fund LP",
         "RPTOWNER_RELATIONSHIP": "TenPercentOwner"},
        {"ACCESSION_NUMBER": "0000000001-24-000004", "RPTOWNERCIK": "0000000602", "RPTOWNERNAME": "Roe Rich",
         "RPTOWNER_RELATIONSHIP": "Director"},
    ]
    t = lambda accn, code, date, shares, price, own="D": {
        "ACCESSION_NUMBER": accn, "TRANS_CODE": code, "TRANS_DATE": date, "TRANS_SHARES": shares,
        "TRANS_PRICEPERSHARE": price, "DIRECT_INDIRECT_OWNERSHIP": own, "SECURITY_TITLE": "Common Stock"}
    trans = [
        t("0000000001-24-000001", "P", "01-APR-2024", "1000.0", "25.5"),
        t("0000000001-24-000001", "S", "02-APR-2024", "200.0", "", "I"),   # no price reported
        t("0000000001-24-000001", "S", "02-APR-2024", "200.0", "", "I"),   # a second, identical lot
        t("0000000001-24-000001", "M", "02-APR-2024", "5000.0", "0.0"),    # option exercise: not a trade
        t("0000000001-24-000001", "P", "09-APR-2024", "10.0", "25.0"),     # dated after its own filing: dropped
        t("0000000001-24-000002", "P", "01-APR-2024", "1000.0", "25.5"),
        t("0000000002-24-000003", "P", "01-APR-2024", "1.0", "1.0"),
        t("0000000001-24-000004", "P", "11-APR-2024", "300.0", "20.0"),
        t("0000000001-24-000004", "P", "11-APR-0024", "300.0", "20.0"),   # a typo in the year: dropped
        t("0000000001-24-000004", "P", "11-APR-2024", "lots", "20.0"),     # no number: dropped
    ]
    return dataset(subs, owners, trans)


FORM4 = b"""<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0609</schemaVersion>
  <documentType>4</documentType>
  <issuer><issuerCik>0000000111</issuerCik><issuerName>Example Corp</issuerName><issuerTradingSymbol>EXC</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0000000701</rptOwnerCik><rptOwnerName>Poe Pat</rptOwnerName></reportingOwnerId>
    <reportingOwnerAddress><rptOwnerStreet1>2 Private Road</rptOwnerStreet1><rptOwnerCity>Elsewhere</rptOwnerCity></reportingOwnerAddress>
    <reportingOwnerRelationship><isDirector>false</isDirector><isOfficer>true</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner><isOther>0</isOther><officerTitle>CFO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-10-06</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>2399</value></transactionShares>
        <transactionPricePerShare><value>40.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-10-06</value></transactionDate>
      <transactionCoding><transactionCode>F</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>100</value></transactionShares></transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-10-07</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>50</value></transactionShares>
        <transactionPricePerShare><footnoteId id="F1"/></transactionPricePerShare></transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>I</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <derivativeTable><derivativeTransaction><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
  </derivativeTransaction></derivativeTable>
  <ownerSignature><signatureName>/s/ Poe Pat</signatureName></ownerSignature>
</ownershipDocument>
"""


class ParseDatasetTest(unittest.TestCase):
    def test_open_market_trades_of_asked_issuers(self):
        fs = insiders.parse_dataset(sample_zip(), {111: "EXC"}, "2026-10-13T00:00:00Z")
        self.assertEqual([(f.context["accn"][-1], f.context["code"], f.as_of, f.context.get("lot")) for f in fs],
                         [("1", "P", "2024-04-01", None), ("1", "S", "2024-04-02", None), ("1", "S", "2024-04-02", 2),
                          ("4", "P", "2024-04-11", None)])
        self.assertEqual(len({f.id for f in fs}), 4)  # identical lots stay two facts
        buy, sell, _, joint = fs
        self.assertEqual((buy.entity, buy.attr, buy.tier, buy.unit, buy.value), ("EXC", "insider.trade", "filed", "shares", 1000.0))
        self.assertEqual(buy.known_at, "2024-04-03T23:59:59Z")  # end of the filing day
        self.assertEqual(buy.source, {"name": "SEC Form 4", "url": "https://www.sec.gov/Archives/edgar/data/111/"
                                      "000000000124000001/0000000001-24-000001-index.htm"})
        self.assertEqual(buy.context, {"code": "P", "price": 25.5, "value_usd": 25500.0, "owner": "Doe Jane",
                                       "owner_cik": "0000000501", "relationship": "officer,director",
                                       "title": "Chief Executive Officer", "accn": "0000000001-24-000001",
                                       "ownership": "D"})
        self.assertEqual((sell.context["price"], sell.context["value_usd"], sell.context["ownership"]), (None, None, "I"))
        self.assertEqual(joint.context["owner"], "Example Fund LP; Roe Rich")
        self.assertEqual(joint.context["relationship"], "director,ten_percent_owner")
        # Nothing about where an insider lives reaches the bank.
        self.assertNotIn("Private", json.dumps([f.to_dict() for f in fs]))

    def test_quarters(self):
        self.assertEqual(insiders.quarters("2025q3", today="2026-04-02"), ["2025q3", "2025q4", "2026q1"])
        self.assertEqual(insiders.quarters("2026q2", today="2026-04-02"), [])
        with self.assertRaises(ValueError):
            insiders.quarters("2025-3")


class ParseForm4Test(unittest.TestCase):
    def test_form4_xml(self):
        url = insiders.index_url(111, "0000000111-26-000009")
        fs = insiders.parse_form4(FORM4, "EXC", "0000000111-26-000009", "2026-10-09T02:30:32Z", url)
        self.assertEqual([(f.context["code"], f.value) for f in fs], [("P", 2399.0), ("S", 50.0)])
        p, s = fs
        self.assertEqual(p.known_at, "2026-10-09T02:30:32Z")  # acceptance time
        self.assertEqual(p.context, {"code": "P", "price": 40.0, "value_usd": 95960.0, "owner": "Poe Pat",
                                     "owner_cik": "0000000701", "relationship": "officer", "title": "CFO",
                                     "accn": "0000000111-26-000009", "ownership": "D"})
        self.assertIsNone(s.context["price"])
        self.assertEqual(s.context["ownership"], "I")
        self.assertEqual(p.source["url"], url)
        self.assertNotIn("Private", json.dumps([f.to_dict() for f in fs]))
        amended = FORM4.replace(b"<documentType>4</documentType>", b"<documentType>4/A</documentType>")
        self.assertEqual(insiders.parse_form4(amended, "EXC", "x", "2026-10-09", url), [])

    def test_xml_url_from_the_rendered_view(self):
        f = Fact(entity="EXC", attr="filing", value="4", as_of="2026-10-06", known_at="2026-10-09", tier="filed",
                 source={"name": "SEC EDGAR", "url": "https://www.sec.gov/Archives/edgar/data/111/000011126000009/"
                                                     "xslF345X06/form4.xml"})
        self.assertEqual(insiders.xml_url(f), "https://www.sec.gov/Archives/edgar/data/111/000011126000009/form4.xml")


class IngestTest(TempHome):
    def _resp(self, body):
        return Response("u", body, "2026-10-13T00:00:00Z", False)

    def test_bulk_remembers_quarters_per_issuer_and_skips_filings_read_live(self):
        z = sample_zip()
        calls = []

        def fake_fetch(url, **kw):
            calls.append(url)
            if "2024q2" in url and "datastandardsinnovation" in url:  # the newer folder
                return self._resp(z)
            raise FetchError(f"{url}: HTTP 404")
        live = Fact(entity="EXC", attr="insider.trade", value=300.0, unit="shares", as_of="2024-04-11",
                    known_at="2024-04-12T13:00:00Z", tier="filed", source={"name": "SEC Form 4", "url": "https://x"},
                    context={"code": "P", "accn": "0000000001-24-000004", "price": 20.0, "value_usd": 6000.0})
        self.bank.add(live)
        with mock.patch.object(insiders, "fetch", side_effect=fake_fetch), \
             mock.patch.object(insiders, "quarters", return_value=["2024q1", "2024q2"]), \
             mock.patch.dict("os.environ", {"SEC_USER_AGENT": "test test@example.test"}):
            out = insiders.ingest_bulk(self.bank, {111: "EXC"}, "2024q1", log=lambda m: None)
            self.assertEqual((out["quarters"], out["unavailable"], out["facts"]), (["2024q2"], ["2024q1"], 3))
            again = insiders.ingest_bulk(self.bank, {111: "EXC"}, "2024q1", log=lambda m: None)
            self.assertEqual((again["skipped"], again["facts"]), (["2024q2"], 0))
            more = insiders.ingest_bulk(self.bank, {111: "EXC", 999: "OTH"}, "2024q1", log=lambda m: None)
            self.assertEqual((more["quarters"], more["facts"]), (["2024q2"], 1))  # the new issuer only
        self.assertEqual(sum("2024q2" in c for c in calls), 4)  # both folders, twice
        self.assertEqual(sum("2024q1" in c for c in calls), 6)  # never found: asked in both folders each run
        done = json.loads((Path(self._tmp.name) / "insiders_done.json").read_text())
        self.assertEqual(done, {"2024q2": [111, 999]})
        accns = sorted(f.context["accn"] for f in self.bank.facts("EXC", "insider.trade"))
        self.assertEqual(accns, ["0000000001-24-000001"] * 3 + ["0000000001-24-000004"])
        self.assertEqual(len(insiders.trades(self.bank.view(), "EXC")), 4)

    def test_default_start_covers_h6s_first_lookback(self):
        # H6's first TRAIN rebalance counts purchases filed in the 90 days before it; a rebuild that follows
        # the documented command (no --start) must read the data set holding the first of those days.
        import argparse
        from datetime import date, timedelta
        from brain.research import hypotheses, signals
        first = date.fromisoformat(hypotheses.SPLIT["first_rebalance"]) - timedelta(days=signals.INSIDER_DAYS)
        need = f"{first.year}q{(first.month - 1) // 3 + 1}"
        p = argparse.ArgumentParser()
        insiders.add_cli(p.add_subparsers())
        self.assertEqual(p.parse_args(["insiders"]).start, need)
        asked = []
        with mock.patch.object(insiders, "_fetch_dataset", side_effect=lambda q, log: asked.append(q)):
            insiders.ingest_bulk(self.bank, {111: "EXC"}, log=lambda m: None)
        self.assertEqual(asked[0], need)

    def test_rereading_a_quarter_adds_only_what_is_missing(self):
        # A filing read before lots were numbered holds one copy of two identical lots; a re-read adds
        # the second and nothing else.
        first = insiders.parse_dataset(sample_zip(), {111: "EXC"}, "2026-10-13T00:00:00Z")
        self.bank.add_many(f for f in first if "lot" not in f.context)
        with mock.patch.object(insiders, "fetch", return_value=self._resp(sample_zip())), \
             mock.patch.object(insiders, "quarters", return_value=["2024q2"]), \
             mock.patch.dict("os.environ", {"SEC_USER_AGENT": "test test@example.test"}):
            out = insiders.ingest_bulk(self.bank, {111: "EXC"}, "2024q2", log=lambda m: None)
        self.assertEqual(out["facts"], 1)
        self.assertEqual(len(self.bank.facts("EXC", "insider.trade")), 4)

    def test_live_reads_new_form4s_once(self):
        from brain.sources import sec
        filings = [Fact(entity="EXC", attr="filing", value=form, as_of="2026-10-06", known_at=known, tier="filed",
                        source={"name": "SEC EDGAR", "url": f"https://www.sec.gov/Archives/edgar/data/111/{a}/xslF345X06/form4.xml"},
                        context={"accn": a})
                   for form, known, a in [("4", "2026-10-09T02:30:32Z", "0000000111-26-000009"),
                                          ("4/A", "2026-10-10T02:30:32Z", "0000000111-26-000010"),
                                          ("4", "2020-01-01T00:00:00Z", "0000000111-20-000001")]]
        fetched = []
        with mock.patch.object(sec, "filings_for", return_value=(filings, [])), \
             mock.patch.object(sec, "cik_for", return_value=111), \
             mock.patch.object(insiders, "_now", return_value=datetime(2026, 10, 13, tzinfo=timezone.utc)), \
             mock.patch.object(insiders, "fetch", side_effect=lambda url, **kw: fetched.append(url) or self._resp(FORM4)), \
             mock.patch.dict("os.environ", {"SEC_USER_AGENT": "test test@example.test"}):
            n = insiders.ingest_live(self.bank, "EXC", days=120)
            again = insiders.ingest_live(self.bank, "EXC", days=120)
        self.assertEqual((n, again), (2, 0))
        self.assertEqual(fetched, ["https://www.sec.gov/Archives/edgar/data/111/0000000111-26-000009/form4.xml"])
        self.assertEqual(len(self.bank.facts("EXC", "filing")), 3)


class SignalTest(TempHome):
    def test_insider_buying_counts_officer_and_director_purchases_in_90_days(self):
        import pandas as pd
        from brain.research import signals
        from brain.research.panel import build
        from tests.util import price_snapshot
        days = pd.bdate_range("2023-01-02", "2024-12-31").strftime("%Y-%m-%d").tolist()
        for t in ("EXC", "SPY"):
            self.bank.add(price_snapshot(t, days, [10.0] * len(days)))
        fs = insiders.parse_dataset(sample_zip(), {111: "EXC"}, "2026-10-13T00:00:00Z")
        self.bank.add_many(fs)
        self.bank.add_many(insiders.parse_form4(FORM4.replace(b"2026-10-0", b"2024-04-0"), "EXC", "live-1",
                                                "2024-04-12T02:00:00Z", "https://x"))
        p = build(self.bank, ["EXC"], start="2023-01-01")
        S = signals.compute(self.bank, p, ["insider_buying"])["insider_buying"]
        months = [str(p.dates[i])[:7] for i in p.rebal]
        val = dict(zip(months, S[:, 0]))
        # April 2024: Doe's 25,500 (officer, director) + Roe's 6,000 (director, joint filing) + Poe's 95,960 (officer).
        self.assertAlmostEqual(val["2024-04"], 25500 + 6000 + 95960)
        self.assertAlmostEqual(val["2024-06"], 25500 + 6000 + 95960)
        self.assertEqual(val["2024-07"], 0.0)  # filed more than 90 days before the July rebalance
        self.assertEqual(val["2024-03"], 0.0)  # not filed yet
        tot = insiders.buy_sell_totals(self.bank.view("2024-04-30"), "EXC")
        self.assertEqual((tot["buys"], tot["sells"]), (3, 3))
