import math
from unittest import mock

from brain import series
from brain.net import Response
from brain.sources import fred, treasury
from tests.util import TempHome

# Before 2018 there was no 2 Mo, before 2022 no 4 Mo, before 2025 no 1.5 Month. Files come newest first.
OLD = b'''Date,"1 Mo","3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"
12/30/2016,0.44,0.51,0.62,0.85,1.20,1.47,1.93,2.25,2.45,2.79,3.06
12/29/2016,0.39,0.47,0.62,0.85,1.22,1.49,1.96,2.30,2.49,2.82,3.08'''

NEW = '''﻿Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"
01/03/2017,4.44,,4.35,4.34,4.31,4.25,4.18,4.28,4.32,4.41,4.51,4.60,4.88,4.82
01/02/2017,4.45,4.40,4.36,4.36,4.31,4.25,4.17,4.25,4.29,4.38,4.47,4.57,4.86,4.79
'''.encode()


class ParseTest(TempHome):
    def test_old_header(self):
        df = treasury.parse_csv(OLD)
        self.assertEqual(list(df.columns), ["date", "1 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr",
                                            "10 Yr", "20 Yr", "30 Yr"])
        self.assertEqual(list(df["date"]), ["2016-12-29", "2016-12-30"])  # ascending, ISO
        self.assertAlmostEqual(df["10 Yr"].iloc[-1], 2.45)

    def test_new_header_with_bom_and_empty_cells(self):
        df = treasury.parse_csv(NEW)
        self.assertEqual(list(df.columns[:4]), ["date", "1 Mo", "1.5 Month", "2 Mo"])
        self.assertIn("4 Mo", df.columns)
        self.assertEqual(df["date"].iloc[0], "2017-01-02")
        self.assertTrue(math.isnan(df["1.5 Month"].iloc[1]))  # empty cell
        self.assertAlmostEqual(df["1.5 Month"].iloc[0], 4.40)

    def test_empty_body(self):
        self.assertEqual(len(treasury.parse_csv(b"")), 0)

    def test_tenor_years(self):
        self.assertAlmostEqual(treasury.tenor_years("3 Mo"), 0.25)
        self.assertAlmostEqual(treasury.tenor_years("1.5 Month"), 0.125)
        self.assertEqual(treasury.tenor_years("30 Yr"), 30.0)
        self.assertIsNone(treasury.tenor_years("Date"))
        self.assertEqual(len(treasury.TENORS), len(treasury.YEARS))


class IngestTest(TempHome):
    def fake_fetch(self, url, **kw):
        body = OLD if "/2016/" in url else NEW
        return Response(url, body, "2017-01-04T12:00:00Z", False)

    def test_two_years_concatenated_and_stored(self):
        with mock.patch.object(treasury, "fetch", self.fake_fetch), \
             mock.patch.object(treasury, "today", lambda: "2017-01-04"):
            facts = treasury.ingest()
        self.assertEqual(len(facts), 1)
        f = facts[0]
        self.assertEqual((f.entity, f.attr, f.tier), ("MACRO", "series.treasury.par_curve", "reported"))
        self.assertEqual(f.source["name"], "US Treasury")
        self.assertEqual(f.as_of, "2017-01-03")
        self.assertEqual(f.known_at, "2017-01-03T23:59:59Z")
        df = series.load(f.value)
        self.assertEqual(list(df["date"]), ["2016-12-29", "2016-12-30", "2017-01-02", "2017-01-03"])
        self.assertEqual(list(df.columns[:4]), ["date", "1 Mo", "1.5 Month", "2 Mo"])  # ordered by maturity
        self.assertTrue(math.isnan(df["2 Mo"].iloc[0]))  # the old file had no 2 Mo column
        self.assertEqual(f.context["rows"], 4)
        self.bank.add_many(facts)
        self.assertEqual(self.bank.view().latest("MACRO", treasury.ATTR).value, f.value)


class MacroTest(TempHome):
    def test_fred_skipped_without_key(self):
        logs = []
        fake = [treasury.Fact(entity="MACRO", attr=treasury.ATTR, value="x" * 64, as_of="2017-01-03",
                              known_at="2017-01-03", tier="reported", source=dict(treasury.SOURCE))]
        with mock.patch.dict("os.environ", {"FRED_API_KEY": ""}), \
             mock.patch.object(treasury, "ingest", lambda: fake), \
             mock.patch.object(fred, "ingest", side_effect=AssertionError("FRED must not be called")):
            out = fred.ingest_all(self.bank, log=logs.append)
        self.assertTrue(out["fred_skipped"])
        self.assertEqual(out["added"], {"treasury": 1})
        self.assertEqual(sum("FRED skipped" in line for line in logs), 1)

    def test_one_failure_does_not_stop_the_rest(self):
        def boom():
            raise fred.FetchError("https://example.test: HTTP 503")
        with mock.patch.dict("os.environ", {"FRED_API_KEY": "k"}), \
             mock.patch.object(treasury, "ingest", boom), \
             mock.patch.object(fred, "ingest", lambda sid: []):
            out = fred.ingest_all(self.bank, ["DGS10"], log=lambda s: None)
        self.assertIn("treasury", out["failed"])
        self.assertEqual(out["added"], {"DGS10": 0})

    def test_fred_error_body_never_echoes_the_key(self):
        body = b'{"error_code": 400, "error_message": "bad key secretkey123"}'
        with mock.patch.dict("os.environ", {"FRED_API_KEY": "secretkey123"}), \
             mock.patch.object(fred, "fetch", lambda url, **kw: Response(url, body, "2017-01-04T00:00:00Z", False)):
            with self.assertRaises(ValueError) as cm:
                fred.ingest("DGS10")
        self.assertNotIn("secretkey123", str(cm.exception))
