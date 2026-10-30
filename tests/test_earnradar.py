"""Earnings radar (brain/earnradar.py): the index, the date estimate, the options verdict, the snapshot and
the misses report. Quotes, the SEC index and ingests are stood in for; nothing touches the network."""
import contextlib
import io
import json
from datetime import date
from unittest import mock

import pandas as pd

from brain import cli
from brain import earnradar as er
from brain.sources import options, sec
from tests.test_options import chain_json
from tests.util import TempHome, filed, price_snapshot


class EstimateTest(TempHome):
    def test_timing_in_new_york(self):
        self.assertEqual(er.timing("2025-10-30T11:02:00Z"), "before open")
        self.assertEqual(er.timing("2025-10-29T20:05:00Z"), "after close")
        self.assertEqual(er.timing("2025-10-30T00:30:00Z"), "after close")   # 20:30 New York the day before
        self.assertEqual(er.timing("2025-10-29T17:00:00Z"), "during session")

    def test_next_report(self):
        row = {"reports": ["2025-07-30T20:05:00Z", "2025-10-30T20:05:00Z", "2026-07-29T20:05:00Z"]}
        got = er.next_report(row, date(2026, 10, 26), 5)
        self.assertEqual(got, {"date": "2026-10-29", "timing": "after close", "basis": "2025-10-30",
                               "confidence": "likely"})
        self.assertEqual(er.next_report(row, date(2026, 11, 2), 5)["confidence"], "possible")   # 4 days early
        self.assertIsNone(er.next_report(row, date(2026, 11, 9), 5))     # well outside the window
        row["reports"].append("2026-10-20T20:05:00Z")                      # already reported this season
        self.assertIsNone(er.next_report(row, date(2026, 10, 26), 5))

    def test_reaction_day_and_verdict(self):
        self.assertEqual(er.reaction_day("2026-10-30", "after close"), "2026-11-02")   # Friday night -> Monday
        self.assertEqual(er.reaction_day("2026-10-29", "before open"), "2026-10-29")
        self.assertEqual(er.verdict(5, 8), "cheap")
        self.assertEqual(er.verdict(2, 8), "expensive")
        self.assertEqual(er.verdict(3, 8), "fair")
        self.assertEqual(er.verdict(1, 3), "too few reports")

    def test_event_move_strips_ordinary_days_to_expiry(self):
        self.assertEqual(er.sessions("2026-10-26", "2026-11-20"), 20)
        self.assertEqual(er.event_move(10.0, None, "2026-10-26", "2026-11-20"), 10.0)
        self.assertAlmostEqual(er.event_move(10.0, 2.0, "2026-10-26", "2026-11-20"), 7.53, places=1)
        self.assertGreater(er.event_move(10.0, 2.0, "2026-10-30", "2026-10-30"), 10.0)   # two ordinary days added


class IndexTest(TempHome):
    def test_build_index_reads_reports_and_reactions(self):
        days = pd.bdate_range("2025-01-01", "2025-12-31").strftime("%Y-%m-%d").tolist()
        px = [100.0] * len(days)
        i = days.index("2025-07-31")
        px = px[:i] + [110.0] * (len(days) - i)              # +10% on the report after Jul 30's close
        self.bank.add_many([price_snapshot("X", days, px), price_snapshot("SPY", days, [400.0] * len(days)),
                            filed("X", "filing", "8-K", "2025-07-30", "2025-07-30T20:05:00Z", items="2.02,9.01"),
                            filed("X", "filing", "8-K", "2025-08-05", "2025-08-05T20:05:00Z", items="5.02")])
        out = er.build_index(self.bank, {"X": "small", "Y": "micro"}, log=lambda m: None)
        self.assertEqual(out["skipped_no_filings"], 1)
        row = out["tickers"]["X"]
        self.assertEqual(row["reports"], ["2025-07-30T20:05:00Z"])
        self.assertAlmostEqual(row["moves"][0], 10.0, places=1)
        self.assertEqual(json.loads(er.index_path().read_text())["tickers"]["X"]["band"], "small")
        self.assertIsNotNone(row["day_vol"])


    def test_refresh_drops_a_company_that_already_reported(self):
        days = pd.bdate_range("2025-01-01", "2026-10-23").strftime("%Y-%m-%d").tolist()
        self.bank.add_many([price_snapshot("X", days, [100.0] * len(days)), price_snapshot("SPY", days, [400.0] * len(days)),
                            filed("X", "filing", "8-K", "2025-10-30", "2025-10-30T11:00:00Z", items="2.02")])
        idx = er.build_index(self.bank, {"X": "small"}, log=lambda m: None)
        self.assertEqual(er.candidates(idx, date(2026, 10, 26), 5), {"X": "small"})
        fresh = [filed("X", "filing", "8-K", "2026-10-22", "2026-10-22T11:00:00Z", items="2.02")]
        with mock.patch.object(sec, "filings_for", return_value=(fresh, [])):
            self.assertEqual(er.refresh(self.bank, {"X": "small"}, log=lambda m: None), 1)
        self.assertEqual(er.candidates(er.load_index(), date(2026, 10, 26), 5), {})


class RadarTest(TempHome):
    def index(self):
        return {"tickers": {
            "XYZ": {"band": "small", "reports": ["2025-10-30T11:00:00Z"], "moves": [9, -12, 3, 10, -8.5, 2, 15, -1]},
            "CALM": {"band": "broad", "reports": ["2025-10-30T11:00:00Z"], "moves": [1, -1, 2, 1, -2, 1, 1, 0.5]},
            "LATER": {"band": "broad", "reports": ["2025-11-20T11:00:00Z"], "moves": [5, 5, 5, 5]}}}

    def test_radar_with_quotes(self):
        rows = er.radar(date(2026, 10, 26), 5, 120.0, index=self.index(), log=lambda m: None,
                        chain_for=lambda t: options.parse(chain_json()))
        self.assertEqual([r["ticker"] for r in rows], ["XYZ", "CALM"])
        x = rows[0]
        self.assertEqual((x["date"], x["react"], x["timing"]), ("2026-10-29", "2026-10-29", "before open"))
        self.assertAlmostEqual(x["straddle"]["implied_pct"], 7.8)
        self.assertEqual((x["beat"], x["verdict"]), (5, "cheap"))
        self.assertTrue(x["fits_budget"])                     # the $105 strangle
        self.assertFalse(x["date_check"])                     # front volatility 1.6x the next: a report is priced
        self.assertEqual(rows[1]["verdict"], "expensive")
        snaps = list(er.snap_dir().glob("*.json"))
        self.assertEqual(len(snaps), 1)
        self.assertEqual(json.loads(snaps[0].read_text())["rows"][0]["ticker"], "XYZ")

    def test_options_that_barely_price_a_report_flag_the_date(self):
        idx = {"tickers": {"MDB": {"band": "top", "reports": ["2025-10-30T11:00:00Z"], "moves": [23, -25, 20, 30, -22, 24, 21, -26]},
                           "POSS": {"band": "top", "reports": ["2025-11-07T11:00:00Z"], "moves": [40] * 8}}}
        rows = er.radar(date(2026, 10, 26), 5, 1000.0, index=idx, log=lambda m: None,
                        chain_for=lambda t: options.parse(chain_json(front_iv=0.5, back_iv=0.5)))
        self.assertEqual([r["ticker"] for r in rows], ["MDB", "POSS"])        # likely before possible
        self.assertTrue(rows[0]["date_check"])                                 # flat volatility: no report priced
        self.assertEqual(rows[1]["confidence"], "possible")

    def test_radar_keeps_going_when_quotes_fail(self):
        def boom(t):
            raise RuntimeError("HTTP 429")
        rows = er.radar(date(2026, 10, 26), 5, index=self.index(), log=lambda m: None, chain_for=boom)
        self.assertIn("HTTP 429", rows[0]["options_error"])

    def test_cli_radar_without_options(self):
        er.index_path().write_text(json.dumps(self.index()))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["earnings", "radar", "--start", "2026-10-26", "--no-options", "--no-refresh", "--min-move", "5"])
        self.assertIn("XYZ", out.getvalue())
        self.assertNotIn("CALM", out.getvalue())


class UniverseTest(TempHome):
    def test_top_band_comes_from_secs_list_and_skips_odd_tickers(self):
        table = [{"ticker": "NOW", "cik": 1, "title": ""}, {"ticker": "BRK-B", "cik": 2, "title": ""},
                 {"ticker": "CVNA", "cik": 3, "title": ""}]
        with mock.patch.object(sec, "ticker_table", return_value=table), mock.patch.object(er, "TOP", 2):
            self.assertEqual(er.universe(("top",)), {"NOW": "top"})


class MissesTest(TempHome):
    def test_form_index_and_cause(self):
        text = ("Form Type   Company Name   CIK   Date Filed   File Name\n" + "-" * 40 + "\n"
                "8-K         ADT Inc.                         1703056     20261029    edgar/data/1703056/0001.txt\n"
                "10-Q        Someone                          1           20261029    edgar/data/1/0002.txt\n")
        rows = er.parse_form_index(text)
        self.assertEqual([(r["form"], r["cik"]) for r in rows], [("8-K", 1703056)])
        self.assertEqual(er.cause({"2.02", "9.01"}), "earnings")
        self.assertEqual(er.cause({"8.01"}), "other news")
        self.assertEqual(er.cause({"1.01"}), "deal or agreement")

    def test_misses_flags_a_radar_hit_and_a_miss(self):
        days = ["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29"]
        self.bank.add_many([price_snapshot("AAA", days, [10, 10, 10, 13.0]), price_snapshot("BBB", days, [20, 20, 20, 15.0]),
                            price_snapshot("CCC", days, [5, 5, 5, 5.1]),
                            filed("AAA", "filing", "8-K", "2026-10-28", "2026-10-28T20:05:00Z", items="2.02,9.01"),
                            filed("BBB", "filing", "8-K", "2026-10-29", "2026-10-29T13:20:00Z", items="8.01,9.01")])
        er.snap_dir().mkdir(parents=True, exist_ok=True)
        (er.snap_dir() / "s.json").write_text(json.dumps({"made_at": "2026-10-26T12:00:00Z", "start": "2026-10-26",
                                                          "days": 5, "rows": [{"ticker": "AAA", "react": "2026-10-29"}]}))
        (er.snap_dir() / "late.json").write_text(json.dumps({"made_at": "2026-10-30T12:00:00Z", "start": "2026-10-26",
                                                             "days": 5, "rows": [{"ticker": "BBB"}]}))  # made after the day
        table = [{"ticker": "AAA", "cik": 1, "title": ""}, {"ticker": "BBB", "cik": 2, "title": ""},
                 {"ticker": "CCC", "cik": 3, "title": ""}]
        with mock.patch.object(sec, "ticker_table", return_value=table):
            rows = er.misses(self.bank, date(2026, 10, 29), only_universe=False, log=lambda m: None,
                             filers=lambda d: [{"cik": 1}, {"cik": 2}, {"cik": 3}], ingest_fn=lambda t: None)
        self.assertEqual([(r["ticker"], r["cause"], r["flagged"]) for r in rows],
                         [("AAA", "earnings", True), ("BBB", "other news", False)])
        self.assertAlmostEqual(rows[0]["move_pct"], 30.0)


if __name__ == "__main__":
    import unittest
    unittest.main()
