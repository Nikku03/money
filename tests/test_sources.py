import unittest

from brain.sources import news, prices, sec


class SecParseTest(unittest.TestCase):
    def test_companyfacts_keeps_first_report_and_restatements(self):
        e = lambda val, filed, accn, end="2019-12-31", start="2019-01-01", form="10-K": {
            "start": start, "end": end, "val": val, "accn": accn, "fy": 2019, "fp": "FY", "form": form, "filed": filed}
        data = {"cik": 1, "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
            e(100, "2020-02-20", "a-1"),
            e(100, "2021-02-20", "a-2"),             # repeated in next year's 10-K: dropped
            e(95, "2022-02-20", "a-3"),              # restated: kept
            e(7, "2019-06-01", "a-4", end="2020-01-31"),  # ends after it was filed: tagging error, dropped
        ]}}, "Ignored": {"units": {"USD": [e(1, "2020-02-20", "a-5")]}}}}}
        fs = sec.parse_companyfacts("T", data, "2026-01-01T00:00:00Z")
        self.assertEqual([(f.value, f.known_at[:10]) for f in fs], [(100, "2020-02-20"), (95, "2022-02-20")])
        self.assertEqual(fs[0].context["duration_days"], 364)
        self.assertTrue(fs[0].source["url"].endswith("/1/a1/"))

    def test_submissions(self):
        data = {"cik": 2, "name": "Co", "sic": "1", "sicDescription": "Things", "exchanges": ["NYSE"],
                "fiscalYearEnd": "1231", "filings": {"recent": {
                    "accessionNumber": ["0001-24-1"], "filingDate": ["2024-05-01"], "reportDate": ["2024-03-31"],
                    "acceptanceDateTime": ["2024-05-01T20:15:00.000Z"], "form": ["10-Q"],
                    "primaryDocument": ["q.htm"], "primaryDocDescription": ["10-Q"], "items": [""]}}}
        fs, ident = sec.parse_submissions("T", data, "2026-01-01T00:00:00Z")
        self.assertEqual(fs[0].known_at, "2024-05-01T20:15:00Z")
        self.assertEqual(fs[0].as_of, "2024-03-31")
        self.assertEqual(ident["identity.exchanges"], "NYSE")

    def test_after_hours_filing_is_known_on_its_filing_date(self):
        data = {"cik": 2, "filings": {"recent": {
            "accessionNumber": ["0001-26-1"], "filingDate": ["2026-07-06"], "reportDate": ["2026-07-06"],
            "acceptanceDateTime": ["2026-07-03T02:17:14.000Z"], "form": ["4"],
            "primaryDocument": ["f4.xml"], "primaryDocDescription": [""], "items": [""]}}}
        fs, _ = sec.parse_submissions("T", data, "2026-10-01T00:00:00Z")
        self.assertEqual(fs[0].known_at, "2026-07-06T23:59:59Z")
        self.assertLessEqual(fs[0].as_of, fs[0].known_at[:10])


class YahooParseTest(unittest.TestCase):
    def chart(self, now):
        day = 86400
        start = 1700000000 - 1700000000 % day + 14 * 3600  # 14:00 UTC
        ts = [start, start + day, start + 2 * day]
        return {"chart": {"result": [{
            "meta": {"gmtoffset": -18000, "currency": "USD",
                     "currentTradingPeriod": {"regular": {"start": ts[2], "end": ts[2] + 6 * 3600}}},
            "timestamp": ts,
            "indicators": {"quote": [{"open": [1, None, 3], "high": [1, None, 3], "low": [1, None, 3],
                                      "close": [1, None, 3], "volume": [5, None, 7]}],
                           "adjclose": [{"adjclose": [0.9, None, 2.9]}]},
            "events": {"splits": {str(ts[0]): {"date": ts[0], "numerator": 4, "denominator": 1}}},
        }]}}, ts

    def test_parse_drops_nulls_and_open_bar(self):
        data, ts = self.chart(0)
        df, splits, _ = prices.parse_chart(data, now=ts[2] + 3600)  # session open: last bar is not a close
        self.assertEqual(len(df), 1)
        self.assertEqual(splits[0][1], 4.0)
        df, _, _ = prices.parse_chart(data, now=ts[2] + 7 * 3600)  # after the close
        self.assertEqual(len(df), 2)


class RssParseTest(unittest.TestCase):
    def test_rss(self):
        body = b"""<?xml version="1.0"?><rss><channel>
        <item><title>Co beats</title><link>https://news.example.com/a</link>
        <pubDate>Fri, 09 Oct 2026 13:00:00 +0000</pubDate></item>
        <item><title>no date</title><link>https://x</link></item></channel></rss>"""
        fs = news.parse_rss("T", body, "2026-10-10T00:00:00Z")
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].known_at, "2026-10-09T13:00:00Z")
        self.assertEqual(fs[0].source["name"], "news.example.com")
        self.assertEqual(fs[0].tier, "reported")
