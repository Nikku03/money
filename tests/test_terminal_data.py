import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from brain import brief, config, series
from brain.bank import Bank, Fact
from brain.claims import note
from brain.metrics import PriceBook
from brain.research.hypotheses import HYPOTHESES
from brain.net import Response
from brain.sources import prices, treasury
from brain.terminal import data
from brain.terminal.commands import Command, parse
from brain.timeutil import today
from tests.util import TempHome, filed, price_snapshot

# Synthetic dates end a few sessions before today, so views at "now" see everything and stay fresh.
END = pd.Timestamp(today()) - pd.offsets.BDay(3)
DAYS = pd.bdate_range(END - pd.DateOffset(years=11, months=6), END).strftime("%Y-%m-%d").tolist()
CRYPTO_DAYS = pd.date_range(END - pd.DateOffset(years=2), END).strftime("%Y-%m-%d").tolist()
LAST_FY = END.year - 1 if END >= pd.Timestamp(f"{END.year}-02-15") else END.year - 2
# A 4:1 split in the middle of FA's ten-year window, relative to today so the tests never age out.
SPLIT_DATE = f"{LAST_FY - 5}-08-31"
UNIVERSE = ["AAPL", "BBB", "CCC", "DDD", "EEE", "FFF"]


def walk(rng, n, start=50.0, sd=0.015):
    return list(start * np.cumprod(1 + rng.normal(0.0004, sd, n)))


def day(offset: int) -> str:
    return DAYS[offset]


def build_bank(bank: Bank) -> dict:
    rng = np.random.default_rng(7)
    px = {}
    for t in [*UNIVERSE, "SPY", "^GSPC"]:
        px[t] = walk(rng, len(DAYS))
        adj = [p * (0.98 + 0.02 * i / len(DAYS)) for i, p in enumerate(px[t])]  # dividends: adj below close early on
        bank.add(price_snapshot(t, DAYS, px[t], adj=adj, splits=[(SPLIT_DATE, 4.0)] if t == "AAPL" else ()))
    bank.add(price_snapshot("EURUSD=X", DAYS, walk(rng, len(DAYS), 1.1, 0.004)))
    px["BTC-USD"] = walk(rng, len(CRYPTO_DAYS), 30000.0, 0.03)
    bank.add(price_snapshot("BTC-USD", CRYPTO_DAYS, px["BTC-USD"]))

    ident = [("identity.name", "Apple Inc."), ("identity.industry", "Electronic Computers"), ("identity.cik", 320193)]
    bank.add_many(filed("AAPL", a, v, DAYS[-30], DAYS[-30]) for a, v in ident)
    flows = {"us-gaap:Revenues": 100e9, "us-gaap:GrossProfit": 40e9, "us-gaap:OperatingIncomeLoss": 30e9,
             "us-gaap:NetIncomeLoss": 25e9, "us-gaap:EarningsPerShareDiluted": 1.5,
             "us-gaap:NetCashProvidedByUsedInOperatingActivities": 28e9,
             "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment": 5e9,
             "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": 4e9}
    stocks = {"us-gaap:Assets": 300e9, "us-gaap:StockholdersEquity": 80e9}
    facts = []
    for k, y in enumerate(range(LAST_FY - 12, LAST_FY + 1)):
        g = 1.05 ** k
        for a, v in flows.items():
            facts.append(filed("AAPL", a, v * g, f"{y}-12-31", f"{y + 1}-02-10", form="10-K",
                               start=f"{y}-01-01", duration_days=364))
        for a, v in stocks.items():
            facts.append(filed("AAPL", a, v * g, f"{y}-12-31", f"{y + 1}-02-10", form="10-K"))
    bank.add_many(facts)

    def filing(form, k, items=""):
        return Fact(entity="AAPL", attr="filing", value=form, as_of=day(-k), known_at=day(-k), tier="filed",
                    source={"name": "SEC EDGAR", "url": f"https://www.sec.gov/x/{form}-{k}.htm"},
                    context={"items": items} if items else {})
    bank.add_many([filing("10-K", 40), filing("8-K", 41, "2.02,9.01"), filing("4", 5), filing("4", 6),
                   filing("8-K", 10, "5.02"), filing("10-Q", 20), filing("8-K/A", 12, "5.02")])
    bank.add_many(Fact(entity="AAPL", attr="headline", value=f"Headline {k}", as_of=day(-k),
                       known_at=day(-k)[:10] + "T13:00:00Z", tier="reported",
                       source={"name": "news.example.com", "url": f"https://news.example.com/{k}"})
                  for k in range(1, 9))
    note(bank, "AAPL", "Analyst says buy", "https://analyst.example.com/a", direction="up",
         known_at=day(-60) + "T12:00:00Z")

    # Treasury curve over two years, with the 1.5 Month column the curve leaves out.
    tdays = [d for d in DAYS if d >= str((END - pd.DateOffset(years=2)).date())]
    curve = pd.DataFrame({"date": tdays})
    for j, tn in enumerate(["1 Mo", "1.5 Month", *treasury.TENORS[1:]]):
        curve[tn] = [3.0 + 0.1 * j + 0.0005 * i for i in range(len(tdays))]
    curve.loc[0, "4 Mo"] = np.nan
    sha = series.store(curve)
    bank.add(Fact(entity="MACRO", attr=treasury.ATTR, value=sha, unit="percent", as_of=tdays[-1], known_at=tdays[-1],
                  tier="reported", source=dict(treasury.SOURCE)))
    fred_df = pd.DataFrame({"date": tdays, "value": [4.0 + 0.001 * i for i in range(len(tdays))]})
    bank.add(Fact(entity="MACRO", attr="series.fred.DGS10", value=series.store(fred_df), as_of=tdays[-1],
                  known_at=tdays[-1], tier="reported", source={"name": "FRED", "url": "https://fred.stlouisfed.org/series/DGS10"},
                  context={"label": "10-year Treasury yield, %", "point_in_time": True}))
    return {"px": px, "curve": curve}


def ledger_records() -> list[dict]:
    at = "2026-01-01T00:00:00Z"
    ctl = {"pass": True, "oracle_test_perm_p": 0.0005, "random_false_positive_rate": 0.03, "n_random": 200}
    ctx = {"spy_ann_return_test": 0.18, "universe_ew_ann_return_test": 0.27}
    recs = [{"op": "declare", "id": h, "spec_sha": "s", "spec": {"claim": f"claim {h}"}, "at": at}
            for h in ("H1-mom12_1", "H2-rev1", "H3-earnings_yield", "H4-low_asset_growth")]
    for h, net, p in (("H1-mom12_1", 0.0011, 0.18), ("H2-rev1", -0.0001, 0.21), ("H4-low_asset_growth", -0.001, 0.7)):
        recs.append({"op": "result", "id": h, "spec_sha": "s", "at": at, "test": {"mean_net_excess": net, "perm_p": p},
                     "train": {}, "controls": ctl, "context": ctx})
    recs.append({"op": "withdraw", "id": "H4-low_asset_growth", "reason": "test withdrawal", "at": at})
    return recs


class SyntheticBankTest(unittest.TestCase):
    """One synthetic bank for the class; each test only reads it (DES and BRIEF add derived facts)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        home = Path(cls._tmp.name)
        cls._old = os.environ.get("MONEY_HOME")
        os.environ["MONEY_HOME"] = str(home)
        (home / "universe.json").write_text(json.dumps({"tickers": UNIVERSE, "rule_short": "synthetic"}))
        (home / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger_records()))
        cls._patches = [mock.patch.object(config, "UNIVERSE", home / "universe.json"),
                        mock.patch.object(config, "LEDGER", home / "ledger.jsonl")]
        for p in cls._patches:
            p.start()
        cls.bank = Bank()
        cls.syn = build_bank(cls.bank)

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        if cls._old is None:
            os.environ.pop("MONEY_HOME", None)
        else:
            os.environ["MONEY_HOME"] = cls._old
        cls._tmp.cleanup()

    def run_ok(self, line):
        r = data.run_line(self.bank, line)
        self.assertTrue(r["ok"], (line, r.get("error"), r.get("hint")))
        json.dumps(r, allow_nan=False)  # never NaN in a response
        for k in ("fn", "ticker", "args", "title", "asof", "view", "data"):
            self.assertIn(k, r)
        return r

    def run_err(self, line):
        r = data.run_line(self.bank, line)
        self.assertFalse(r["ok"], line)
        self.assertTrue(r["error"])
        json.dumps(r, allow_nan=False)
        return r

    # ---------- global ----------
    def test_help(self):
        d = self.run_ok("HELP")["data"]
        self.assertIn("GP", [f["code"] for f in d["functions"]])
        self.assertTrue(all({"code", "scope", "args", "desc"} <= set(f) for f in d["functions"]))
        self.assertTrue(d["keys"] and all({"key", "desc"} <= set(k) for k in d["keys"]))

    def test_server_only_functions(self):
        for line in ("JOBS", "ASK what now?", "INGEST AAPL"):
            self.assertTrue(self.run_err(line)["hint"])

    def test_unknown_and_missing_ticker(self):
        r = data.run(self.bank, Command(fn="XYZ", ticker="AAPL", args=[], raw="AAPL XYZ"))
        self.assertEqual(r["error"], "unknown function 'XYZ'")
        r = data.run(self.bank, Command(fn="FA", ticker=None, args=[], raw="FA"))
        self.assertEqual((r["error"], r["hint"]), ("FA needs a ticker, e.g. AAPL FA", "AAPL FA"))
        self.assertEqual(self.run_err("FA")["hint"], "AAPL FA")  # parse errors come back the same way
        self.assertFalse(data.run(self.bank, Command(fn="DES", ticker="a;b", args=[], raw=""))["ok"])

    # ---------- DES ----------
    def test_des_equity(self):
        r = self.run_ok("aapl")
        self.assertEqual(r["view"], "des")
        self.assertEqual(r["title"], "AAPL — Apple Inc. · description")
        d = r["data"]
        self.assertEqual((d["name"], d["kind"]), ("Apple Inc.", "equity"))
        self.assertEqual([i["label"] for i in d["identity"]], ["Industry", "SEC CIK"])
        p = d["price"]
        self.assertEqual(p["date"], DAYS[-1])
        self.assertAlmostEqual(p["close"], self.syn["px"]["AAPL"][-1], places=4)
        for k in ("chg_1d", "ret_1m", "ret_3m", "ret_ytd", "ret_12m", "vol_12m", "from_high_12m", "high_12m",
                  "low_12m", "cite"):
            self.assertIsNotNone(p[k], k)
        keys = [m["key"] for m in d["metrics"]]
        for k in ("market_cap", "earnings_yield", "pe", "revenue_growth", "net_margin", "gross_margin",
                  "free_cash_flow", "fcf_yield", "asset_growth"):
            self.assertIn(k, keys)
        mc = d["metrics"][keys.index("market_cap")]
        self.assertTrue(mc["cite"] and len(mc["inputs"]) == 2)
        # every row is dated: the price date for market value and the yields, the fiscal-year end otherwise
        as_of = {m["key"]: m["as_of"] for m in d["metrics"]}
        for k in ("market_cap", "earnings_yield", "pe", "fcf_yield"):
            self.assertEqual(as_of[k], DAYS[-1], k)
        for k in ("revenue_growth", "net_margin", "gross_margin", "free_cash_flow", "asset_growth"):
            self.assertEqual(as_of[k], f"{LAST_FY}-12-31", k)
        written = self.bank.facts("AAPL", "derived.market_cap")  # written back, as the brief does
        self.assertIn(mc["cite"], [f.id[:8] for f in written])
        self.assertNotIn("derived.pe", self.bank.attrs("AAPL"))  # P/E is display-only
        self.assertEqual([f["form"] for f in d["filings"]], ["8-K", "8-K/A", "10-Q", "10-K", "8-K"])  # no Form 4
        self.assertEqual(d["filings"][-1]["desc"], "results, exhibits")
        self.assertEqual([h["title"] for h in d["headlines"]], [f"Headline {k}" for k in range(1, 6)])

    def test_des_market_symbols(self):
        d = self.run_ok("^GSPC DES")["data"]
        self.assertEqual((d["kind"], d["name"]), ("index", "S&P 500"))
        self.assertEqual((d["identity"], d["metrics"], d["filings"]), ([], [], []))
        self.assertIsNotNone(d["price"])
        self.assertEqual(self.run_ok("EURUSD=X")["data"]["kind"], "fx")
        self.assertEqual(self.run_ok("SPY DES")["data"]["kind"], "other")
        btc = self.run_ok("BTC-USD DES")["data"]
        a = self.syn["px"]["BTC-USD"]
        self.assertEqual(btc["kind"], "crypto")
        self.assertAlmostEqual(btc["price"]["ret_1m"], a[-1] / a[-31] - 1, places=4)  # 30 sessions: trades daily

    # ---------- GP ----------
    def test_gp_downsampled_with_endpoints(self):
        r = self.run_ok("AAPL GP MAX")
        d = r["data"]
        self.assertEqual(r["view"], "chart")
        self.assertLessEqual(len(d["series"]), data.MAX_POINTS)
        self.assertGreater(len(DAYS), data.MAX_POINTS)
        self.assertEqual((d["series"][0]["date"], d["series"][-1]["date"]), (DAYS[0], DAYS[-1]))
        dates = [p["date"] for p in d["series"]]
        self.assertEqual(dates, sorted(set(dates)))
        px = self.syn["px"]["AAPL"]
        self.assertAlmostEqual(d["stats"]["first"], px[0], places=4)
        self.assertAlmostEqual(d["stats"]["high"], max(px), places=4)
        self.assertEqual(d["stats"]["high_date"], DAYS[int(np.argmax(px))])
        self.assertEqual(d["splits"], [[SPLIT_DATE, 4.0]])
        self.assertEqual((d["kind"], d["cite"] is not None), ("equity", True))
        self.assertEqual(d["rebased"][0]["value"], 100.0)
        self.assertEqual(len(d["rebased"]), len(d["series"]))

    def test_gp_ranges(self):
        d = self.run_ok("AAPL GP")["data"]
        self.assertEqual(d["range"], "1Y")
        start = str((END - pd.DateOffset(years=1)).date())
        self.assertEqual(d["series"][0]["date"], min(x for x in DAYS if x >= start))
        ytd = self.run_ok("GP AAPL YTD")["data"]
        self.assertEqual(ytd["series"][0]["date"], max(x for x in DAYS if x[:4] < DAYS[-1][:4]))  # last close of last year
        self.assertAlmostEqual(ytd["stats"]["ret"], self.run_ok("AAPL DES")["data"]["price"]["ret_ytd"], places=10)
        self.assertEqual(len(self.run_ok("AAPL GP 1M")["data"]["series"]), sum(
            x >= str((END - pd.DateOffset(months=1)).date()) for x in DAYS))
        self.assertIn("one range", self.run_err("AAPL GP 1Y 5Y")["error"])
        self.assertIn("at most 3", self.run_err("AAPL GP SPY BBB CCC DDD")["error"])
        for bad in ("3Y", "2M", "1W", "20Y"):  # a range we do not draw is an error, never a comparison ticker
            r = self.run_err(f"AAPL GP {bad}")
            self.assertEqual((r["error"], r["hint"]), (f"unknown range {bad}; ranges: 1M 3M 6M YTD 1Y 2Y 5Y 10Y MAX",
                                                       "AAPL GP 1Y"))
        self.assertTrue(self.run_err("GP 3Y AAPL")["error"].startswith("unknown range 3Y"))

    def test_gp_compare_rebased(self):
        r = self.run_ok("AAPL GP 1Y SPY BTC-USD ZZZZ")
        d = r["data"]
        self.assertEqual([c["ticker"] for c in d["compare"]], ["SPY", "BTC-USD"])
        self.assertEqual(d["missing"], ["ZZZZ"])
        self.assertIn("vs SPY, BTC-USD", r["title"])
        main = [p["date"] for p in d["series"]]
        for c in d["compare"]:
            self.assertEqual(c["points"][0]["value"], 100.0)
            self.assertEqual([p["date"] for p in c["points"]], main)  # on the main ticker's dates
        spy = self.syn["px"]["SPY"]
        i0 = DAYS.index(main[0])
        adj = [p * (0.98 + 0.02 * i / len(DAYS)) for i, p in enumerate(spy)]
        self.assertAlmostEqual(d["compare"][0]["points"][-1]["value"], adj[-1] / adj[i0] * 100, places=3)

    def test_gp_compare_that_starts_later(self):
        # BTC-USD has two years of history: on MAX every line is 100 on its first date, so the legend's
        # returns (last / 100 - 1) all cover the same period.
        d = self.run_ok("AAPL GP MAX SPY BTC-USD")["data"]
        spy, btc = d["compare"]
        first_btc = min(x for x in DAYS if x >= CRYPTO_DAYS[0])
        self.assertEqual((spy["start"], btc["start"]), (DAYS[0], first_btc))
        base = d["rebase_date"]
        self.assertTrue(first_btc <= base <= DAYS[DAYS.index(first_btc) + 5], base)  # the first shown date
        at = lambda pts: next(p["value"] for p in pts if p["date"] == base)
        self.assertEqual((at(d["rebased"]), at(spy["points"]), at(btc["points"])), (100.0, 100.0, 100.0))
        self.assertEqual(btc["points"][0]["date"], base)
        adj = {p["date"]: p["adj"] for p in d["series"]}
        self.assertAlmostEqual(d["rebased"][-1]["value"] / 100 - 1, adj[DAYS[-1]] / adj[base] - 1, places=10)
        a = self.syn["px"]["BTC-USD"]
        self.assertAlmostEqual(btc["points"][-1]["value"] / 100 - 1, a[-1] / a[CRYPTO_DAYS.index(base)] - 1, places=8)
        self.assertEqual(self.run_ok("AAPL GP MAX")["data"]["rebase_date"], DAYS[0])

    def test_gp_missing(self):
        self.assertEqual(self.run_err("ZZZZ GP")["hint"], "INGEST ZZZZ")
        self.assertEqual(self.run_err("^FTSE GP 5Y")["hint"], "INGEST MARKETS")

    # ---------- FA, CF, CN ----------
    def test_fa(self):
        r = self.run_ok("AAPL FA")
        d = r["data"]
        fys = [c["key"] for c in d["columns"][1:]]
        self.assertEqual(len(fys), 10)
        self.assertEqual(fys, sorted(fys, reverse=True))  # newest first
        self.assertEqual(fys[0], f"{LAST_FY}-12-31")
        self.assertEqual(r["asof"], fys[0])
        rows = {row["item"]: row for row in d["rows"]}
        self.assertEqual(d["groups"], ["Income statement", "Cash flow", "Balance sheet", "Ratios"])
        self.assertEqual(set(rows["Revenue"]["_cites"]), set(fys))
        g = 1.05 ** 12
        self.assertAlmostEqual(rows["Revenue"][fys[0]], 100e9 * g, delta=1)
        self.assertAlmostEqual(rows["Free cash flow"][fys[0]], 23e9 * g, delta=1)
        self.assertAlmostEqual(rows["Net margin"][fys[0]], 0.25)
        self.assertAlmostEqual(rows["ROE"][fys[0]], 25 / 80)
        self.assertAlmostEqual(rows["Revenue growth"][fys[0]], 0.05)
        self.assertEqual(rows["EPS diluted"]["_type"], "number")
        self.assertEqual(rows["Gross margin"]["_type"], "pct")
        self.assertNotIn("R&D", rows)  # no data, no row
        self.assertNotIn("_type", d["note"])  # the note is for people, not about row keys
        err = self.run_err("ZZZZ FA")
        self.assertEqual((err["error"], err["hint"]), ("no SEC facts for ZZZZ", "INGEST ZZZZ"))

    def test_fa_eps_on_todays_share_basis(self):
        # Synthetic AAPL splits 4:1 on SPLIT_DATE; each FY's 10-K is filed on Feb 10 of the next year.
        d = self.run_ok("AAPL FA")["data"]
        eps = next(r for r in d["rows"] if r["item"] == "EPS diluted")
        before = [fy for fy in eps if fy[:2] == "20" and fy < SPLIT_DATE[:4] + "-01-01"]  # filed before the split
        self.assertTrue(before)
        for fy in [c["key"] for c in d["columns"][1:]]:
            k = int(fy[:4]) - (LAST_FY - 12)
            filed_value = 1.5 * 1.05 ** k
            self.assertAlmostEqual(eps[fy], filed_value / 4 if fy in before else filed_value, places=6)
            cited = next(f for f in self.bank.facts("AAPL", "us-gaap:EarningsPerShareDiluted")
                         if f.id.startswith(eps["_cites"][fy]))
            self.assertAlmostEqual(cited.value, filed_value, places=6)  # the citation keeps the as-filed number
        self.assertIn(f"4:1 on {SPLIT_DATE}", d["note"])

    def test_cf(self):
        d = self.run_ok("AAPL CF")["data"]
        self.assertEqual(len(d["rows"]), 7)
        self.assertEqual([c["key"] for c in d["columns"]], ["date", "form", "desc", "link"])
        dates = [r["date"] for r in d["rows"]]
        self.assertEqual(dates, sorted(dates, reverse=True))
        d = self.run_ok("AAPL CF 8-K")["data"]
        self.assertEqual([r["form"] for r in d["rows"]], ["8-K", "8-K/A", "8-K"])
        self.assertTrue(d["note"].startswith("3 of 3 form 8-K filings"))
        self.assertTrue(all(r["_cite"] and r["_url"] == r["link"] for r in d["rows"]))
        self.assertEqual(self.run_ok("AAPL CF 4")["data"]["rows"][0]["form"], "4")
        self.assertEqual(self.run_err("SPY CF")["hint"], "INGEST SPY")

    def test_cn(self):
        d = self.run_ok("AAPL CN")["data"]
        self.assertEqual(d["headlines"][0]["title"], "Headline 1")
        self.assertEqual(len(d["headlines"]), 8)
        c = d["claims"][0]
        self.assertEqual((c["direction"], c["tier"], c["source"]), (1, "opinion", "analyst.example.com"))
        self.assertEqual(c["record"]["n"], 1)
        self.assertEqual(self.run_ok("SPY CN")["data"], {"headlines": [], "claims": []})

    # ---------- EV, BRIEF, SHOW ----------
    def test_ev(self):
        r = self.run_ok("AAPL EV")
        d = r["data"]
        self.assertEqual(r["view"], "evidence")
        self.assertEqual((d["ticker"], d["universe_size"], d["controls_pass"]), ("AAPL", 6, True))
        sig = {s["signal"]: s for s in d["signals"]}
        self.assertEqual(list(sig), ["mom12_1", "rev1", "earnings_yield", "low_asset_growth"])
        self.assertEqual((sig["mom12_1"]["verdict"], sig["mom12_1"]["test_net_excess"]), ("FAIL", 0.0011))
        self.assertEqual(sig["earnings_yield"]["verdict"], "not run")
        self.assertTrue(sig["low_asset_growth"]["verdict"].startswith("WITHDRAWN"))
        self.assertIsNone(sig["low_asset_growth"]["bh_q"])  # NaN in the ledger maths, null on the wire
        self.assertTrue(0 <= sig["mom12_1"]["percentile"] <= 100)
        self.assertEqual(sig["mom12_1"]["claim"], next(h["claim"] for h in HYPOTHESES if h["id"] == "H1-mom12_1"))
        self.assertTrue(d["summary"].startswith("0 of 4 signals passed"))
        self.assertEqual(self.run_err("ZZZZ EV")["hint"], "INGEST ZZZZ")
        for sym in ("^GSPC", "EURUSD=X", "BTC-USD"):  # stock signals do not rank an index, a currency or a coin
            err = self.run_err(f"{sym} EV")
            self.assertIn("market symbol", err["error"])
            self.assertEqual(err["hint"], f"{sym} GP")

    def test_brief_matches_ev(self):
        md = self.run_ok("AAPL BRIEF")["data"]["markdown"]
        self.assertTrue(md.startswith("# AAPL — Apple Inc."))
        ev = self.run_ok("AAPL EV")["data"]
        self.assertIn(f"last month-end close ({ev['rebalance_date']})", md)
        mom = ev["signals"][0]
        self.assertIn(f"| mom12_1 | {mom['value']:+.3f} | {mom['percentile']} |", md)

    def test_show(self):
        f = self.bank.facts("AAPL", "us-gaap:Revenues")[0]
        d = self.run_ok(f"AAPL SHOW {f.id[:8]}")["data"]
        self.assertEqual(d["fact"]["id"], f.id)
        self.assertFalse(d["downgraded"])
        self.run_ok("AAPL DES")
        mc = self.bank.facts("AAPL", "derived.market_cap")[0]
        d = self.run_ok(f"SHOW AAPL/{mc.id}")["data"]
        self.assertEqual(len(d["inputs"]), 2)
        self.assertTrue(all(i["attr"] and i["cite"] for i in d["inputs"]))
        curve = self.bank.facts("MACRO", treasury.ATTR)[0]
        r = self.run_ok(f"SHOW {curve.id[:10]}")  # no ticker: found among MACRO facts
        self.assertEqual(r["data"]["fact"]["entity"], "MACRO")
        self.assertTrue(self.run_ok(f"AAPL SHOW {curve.id[:8]}")["ok"])  # a CURVE citation from an AAPL pane
        self.assertIn("no fact", self.run_err("AAPL SHOW 0000ffff")["error"])
        self.assertIn("not a fact id", self.run_err("AAPL SHOW AB/C1")["error"])

    # ---------- WEI, MOV, CURVE, ECO ----------
    def test_wei(self):
        r = self.run_ok("WEI")
        d = r["data"]
        self.assertEqual([row["symbol"] for row in d["rows"]], ["^GSPC", "EURUSD=X", "BTC-USD"])
        self.assertEqual(d["groups"], ["Americas", "FX", "Crypto"])
        self.assertIn("INGEST MARKETS", d["note"])
        self.assertEqual([c["key"] for c in d["columns"]],
                         ["symbol", "name", "last", "chg_1d", "chg_5d", "ret_1m", "ret_ytd", "date"])
        rows = {row["symbol"]: row for row in d["rows"]}
        self.assertEqual(rows["EURUSD=X"]["_digits"], 4)
        self.assertEqual(rows["^GSPC"]["_ticker"], "^GSPC")
        a = self.syn["px"]["BTC-USD"]
        self.assertAlmostEqual(rows["BTC-USD"]["chg_5d"], a[-1] / a[-8] - 1, places=4)
        self.assertEqual(r["asof"], max(DAYS[-1], CRYPTO_DAYS[-1]))

    def test_mov(self):
        d = self.run_ok("MOV")["data"]
        self.assertEqual(d["groups"], ["Top gainers, last session", "Top losers, last session", "Top 1 month",
                                       "Bottom 1 month"])
        by = {g: [r for r in d["rows"] if r["_group"] == g] for g in d["groups"]}
        self.assertTrue(all(0 < len(v) <= 8 for v in by.values()))
        up = [r["chg_1d"] for r in by["Top gainers, last session"]]
        down = [r["chg_1d"] for r in by["Top losers, last session"]]
        self.assertEqual(up, sorted(up, reverse=True))
        self.assertEqual(down, sorted(down))
        self.assertTrue(all(x > 0 for x in up) and all(x < 0 for x in down))  # a gainer rose; a loser fell
        self.assertEqual(len(up) + len(down), len(UNIVERSE))
        self.assertIn("Apple Inc.", [r["name"] for r in d["rows"]])

    def test_curve(self):
        r = self.run_ok("CURVE")
        d = r["data"]
        curve = self.syn["curve"]
        self.assertEqual(d["tenors"], treasury.TENORS)
        self.assertEqual(len(d["years"]), 13)
        self.assertEqual([c["label"] for c in d["curves"]], ["Latest", "1 month ago", "1 year ago"])
        latest = d["curves"][0]
        self.assertEqual(latest["date"], curve["date"].iloc[-1])
        self.assertEqual(len(latest["yields"]), 13)  # 1.5 Month dropped
        self.assertAlmostEqual(latest["yields"][1], curve["2 Mo"].iloc[-1])
        self.assertLessEqual(d["curves"][1]["date"], str((pd.Timestamp(latest["date"]) - pd.DateOffset(months=1)).date()))
        sp = {s["label"]: s for s in d["spreads"]}
        self.assertAlmostEqual(sp["10Y − 2Y"]["latest"], curve["10 Yr"].iloc[-1] - curve["2 Yr"].iloc[-1], places=4)
        self.assertIsNotNone(sp["10Y − 3M"]["y1"])
        self.assertTrue(d["cite"])

    def test_eco(self):
        d = self.run_ok("ECO")["data"]
        self.assertEqual(d["groups"], ["Treasury yields", "Treasury spreads", "FRED"])
        rows = {r["series"]: r for r in d["rows"]}
        self.assertAlmostEqual(rows["Treasury 10Y par yield, %"]["latest"], self.syn["curve"]["10 Yr"].iloc[-1])
        self.assertEqual(rows["Treasury 10Y par yield, %"]["_group"], "Treasury yields")
        self.assertIn("Treasury 10Y − 2Y, pp", rows)
        fred = [r for r in d["rows"] if r["_group"] == "FRED"]
        self.assertEqual(len(fred), 1)
        self.assertTrue(all(r["_cite"] for r in d["rows"]))

    # ---------- RES, CLM ----------
    def test_res(self):
        d = self.run_ok("RES")["data"]
        rows = {r["id"]: r for r in d["rows"]}
        self.assertEqual(rows["H1-mom12_1"]["verdict"], "FAIL")
        self.assertEqual(rows["H1-mom12_1"]["claim"], "claim H1-mom12_1")
        self.assertEqual(rows["H3-earnings_yield"]["verdict"], "not run")
        self.assertIsNone(rows["H4-low_asset_growth"]["bh_q"])
        self.assertEqual([c["type"] for c in d["columns"]], ["text", "text", "text", "pct", "number", "number", "verdict"])
        self.assertEqual({r["universe"] for r in d["rows"]}, {"us_large"})
        self.assertIn("Controls passed on us_large", d["note"])
        self.assertIn("+27.0% a year", d["note"])

    def test_res_reports_each_universe_once(self):
        broad = {"op": "result", "id": "H6-insider_buying", "spec_sha": "s", "at": "2026-02-01T00:00:00Z",
                 "universe": "universe/us_broad.json", "test": {"mean_net_excess": 0.0001, "perm_p": 0.3}, "train": {},
                 "controls": {"pass": True, "oracle_test_perm_p": 0.0005, "random_false_positive_rate": 0.05,
                              "n_random": 200},
                 "context": {"spy_ann_return_test": 0.18, "universe_ew_ann_return_test": 0.31}}
        declare = {"op": "declare", "id": "H6-insider_buying", "spec_sha": "s", "at": "2026-01-15T00:00:00Z",
                   "spec": {"claim": "insiders", "universe": "universe/us_broad.json"}}
        path = Path(self._tmp.name) / "ledger2.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in ledger_records() + [declare, broad]))
        with mock.patch.object(config, "LEDGER", path):
            d = self.run_ok("RES")["data"]
        rows = {r["id"]: r for r in d["rows"]}
        self.assertEqual((rows["H6-insider_buying"]["universe"], rows["H1-mom12_1"]["universe"]), ("us_broad", "us_large"))
        self.assertEqual(d["note"].count("Controls passed on "), 2)
        self.assertIn("Controls passed on us_broad", d["note"])
        self.assertIn("+31.0% a year", d["note"])

    def test_clm(self):
        d = self.run_ok("CLM")["data"]
        self.assertEqual(d["rows"][0]["source"], "analyst.example.com")
        self.assertEqual(d["rows"][0]["n"], 1)
        self.assertIn("pending", d["note"])

    def test_no_nan_anywhere(self):
        for line in ("HELP", "AAPL", "SPY", "AAPL GP 10Y SPY", "^GSPC GP MAX", "AAPL FA", "AAPL CF", "AAPL CN",
                     "AAPL EV", "BBB EV", "AAPL BRIEF", "WEI", "MOV", "CURVE", "ECO", "RES", "CLM"):
            self.run_ok(line)


class EmptyBankTest(TempHome):
    def test_missing_data_hints(self):
        with mock.patch.object(config, "UNIVERSE", Path(self._tmp.name) / "none.json"), \
             mock.patch.object(config, "LEDGER", Path(self._tmp.name) / "none.jsonl"):
            cases = {"XYZ": "INGEST XYZ", "XYZ DES": "INGEST XYZ", "XYZ FA": "INGEST XYZ", "XYZ CN": "INGEST XYZ",
                     "XYZ BRIEF": "INGEST XYZ", "^GSPC DES": "INGEST MARKETS", "WEI": "INGEST MARKETS",
                     "MOV": "INGEST UNIVERSE", "CURVE": "INGEST MACRO", "ECO": "INGEST MACRO"}
            for line, hint in cases.items():
                r = data.run_line(self.bank, line)
                self.assertFalse(r["ok"], line)
                self.assertEqual(r["hint"], hint, line)
            self.assertTrue(data.run_line(self.bank, "RES")["ok"])  # an empty ledger is a fact, not an error
            self.assertTrue(data.run_line(self.bank, "CLM")["ok"])

    def test_ingest_hint_round_trips_for_tickers_that_are_codes(self):
        # CF Industries, Movado, RPC...: the hint for a code-named ticker must start a job for that ticker.
        for code in ("CF", "FA", "GP", "MOV", "RES", "ECO", "CLM", "GO"):
            r = data.run_line(self.bank, f"{code} DES")
            self.assertEqual((r["fn"], r["ticker"], r["error"], r["hint"]),
                             ("DES", code, f"nothing in the bank for {code}", f"INGEST {code}"))
            cmd = parse(r["hint"])
            self.assertEqual((cmd.fn, cmd.ticker, cmd.args), ("INGEST", code, []), code)

    def test_mov_session_survives_one_stock_ingested_after_the_close(self):
        days = pd.bdate_range(end=pd.Timestamp(today()) - pd.offsets.BDay(3), periods=40).strftime("%Y-%m-%d").tolist()
        nxt = (pd.Timestamp(days[-1]) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        moves = {"AAA": 0.02, "BBB": 0.01, "CCC": -0.01, "DDD": -0.03, "EEE": 0.0}
        for t, m in moves.items():
            self.bank.add(price_snapshot(t, days, [100.0] * 39 + [100.0 * (1 + m)]))
        self.bank.add(price_snapshot("KKK", days + [nxt], [100.0] * 40 + [100.1]))  # re-ingested after the close
        uni = Path(self._tmp.name) / "uni.json"
        uni.write_text(json.dumps({"tickers": [*moves, "KKK"]}))
        with mock.patch.object(config, "UNIVERSE", uni):
            r = data.run_line(self.bank, "MOV")
        d = r["data"]
        by = {g: [row["symbol"] for row in d["rows"] if row["_group"] == g] for g in d["groups"]}
        self.assertEqual(by["Top gainers, last session"], ["AAA", "BBB"])
        self.assertEqual(by["Top losers, last session"], ["DDD", "CCC"])
        self.assertIn(f"last session {days[-1]}", d["note"])
        self.assertIn(f"1 stocks whose last close is not on {days[-1]} are left out", d["note"])
        self.assertEqual(by["Top 1 month"][0], "AAA")
        self.assertEqual(r["asof"], nxt)

    def test_display_rules_for_banks(self):
        # A bank lists one exchange per listed security and files offering paperwork daily.
        days = pd.bdate_range(end=pd.Timestamp(today()) - pd.offsets.BDay(2), periods=30).strftime("%Y-%m-%d").tolist()
        self.bank.add(price_snapshot("BNK", days, [50.0 + i for i in range(30)]))
        self.bank.add_many([filed("BNK", "identity.name", "BANK CORP", days[0], days[0]),
                            filed("BNK", "identity.exchanges", "NYSE,NYSE,NYSE", days[0], days[0])])
        forms = ["424B2", "FWP", "4/A", "SCHEDULE 13G/A", "10-Q", "13F-HR", "8-K", "3", "144", "10-K/A"]
        self.bank.add_many(Fact(entity="BNK", attr="filing", value=form, as_of=days[-1 - k], known_at=days[-1 - k],
                                tier="filed", source={"name": "SEC EDGAR", "url": f"https://www.sec.gov/x/{k}.htm"})
                           for k, form in enumerate(forms))
        d = data.run_line(self.bank, "BNK DES")["data"]
        self.assertEqual({i["label"]: i["value"] for i in d["identity"]}["Exchange"], "NYSE")
        self.assertEqual([f["form"] for f in d["filings"]], ["10-Q", "8-K", "10-K/A"])
        self.assertEqual(len(data.run_line(self.bank, "BNK CF")["data"]["rows"]), len(forms))  # CF still lists all
        with mock.patch.object(config, "UNIVERSE", Path(self._tmp.name) / "none.json"):
            md = brief.build(self.bank, "BNK")
        self.assertIn("- Exchange: NYSE [", md)
        self.assertIn("**10-Q**", md)
        self.assertNotIn("424B", md)
        self.assertNotIn("**4/A**", md)
        self.assertNotIn("13G", md)

    def test_todays_close_is_visible(self):
        # A close dated today is known at the end of today (UTC); the terminal still shows it now.
        days = pd.date_range(end=pd.Timestamp(today()), periods=40).strftime("%Y-%m-%d").tolist()
        self.bank.add(price_snapshot("^FTSE", days, [100.0 + i for i in range(40)]))
        self.assertEqual(data.run_line(self.bank, "WEI")["data"]["rows"][0]["date"], today())
        self.assertEqual(data.run_line(self.bank, "^FTSE DES")["data"]["price"]["date"], today())
        self.assertEqual(data.run_line(self.bank, "^FTSE GP 1M")["asof"], today())

    def test_price_stats_windows(self):
        d = np.array(pd.bdate_range("2024-01-01", periods=300).strftime("%Y-%m-%d"))
        a = np.linspace(100.0, 200.0, 300)
        s = brief.price_stats(d, a * 2, a)
        self.assertAlmostEqual(s["ret_1m"], a[-1] / a[-22] - 1)
        self.assertAlmostEqual(s["ret_12m"], a[-1] / a[-253] - 1)
        prior = int(np.flatnonzero(d < d[-1][:4] + "-01-01")[-1])
        self.assertAlmostEqual(s["ret_ytd"], a[-1] / a[prior] - 1)
        self.assertEqual((s["high_12m"], s["close"]), (400.0, 400.0))
        self.assertAlmostEqual(s["vol_12m"], np.diff(np.log(a[-253:])).std() * np.sqrt(252))
        self.assertIsNone(brief.price_stats(d[:0], a[:0], a[:0]))

    def test_price_stats_never_relabels_a_shorter_window(self):
        d = np.array(pd.bdate_range("2024-03-01", periods=120).strftime("%Y-%m-%d"))
        a = np.linspace(100.0, 160.0, 120)
        c = a * 0.9  # closes differ from dividend-adjusted closes
        one = brief.price_stats(d[:1], c[:1], a[:1])  # a single close is still shown
        self.assertEqual((one["close"], one["date"]), (90.0, "2024-03-01"))
        self.assertTrue(np.isnan(one["chg_1d"]))
        s = brief.price_stats(d, c, a)  # four months of history, all in 2024
        self.assertAlmostEqual(s["ret_1m"], a[-1] / a[-22] - 1)
        for k in ("ret_12m", "vol_12m", "high_12m", "low_12m", "from_high_12m", "ret_ytd"):
            self.assertTrue(np.isnan(s[k]), k)  # not "since listing" under a 12-month or YTD label
        d2 = np.array(pd.bdate_range("2023-01-02", periods=300).strftime("%Y-%m-%d"))
        a2 = 100 * np.exp(np.sin(np.arange(300) / 20))
        c2 = a2 * np.linspace(1.2, 1.0, 300)  # dividends: closes drift away from the adjusted series
        full = brief.price_stats(d2, c2, a2)
        self.assertAlmostEqual(full["high_12m"], c2[-252:].max())
        self.assertAlmostEqual(full["from_high_12m"], c2[-1] / c2[-252:].max() - 1)  # same basis as the high

    def test_clean(self):
        out = data.clean({"a": float("nan"), "b": [np.float64("inf"), np.int64(3), (1, np.bool_(True))],
                          "c": np.array([1.5, np.nan])})
        self.assertEqual(out, {"a": None, "b": [None, 3, [1, True]], "c": [1.5, None]})
        self.assertIsInstance(out["b"][1], int)
        json.dumps(out, allow_nan=False)

    def test_downsample(self):
        for n in (1, 2, 1500, 1501, 4000):
            idx = data.downsample(n)
            self.assertLessEqual(len(idx), 1500)
            self.assertEqual((idx[0], idx[-1]), (0, n - 1))
            self.assertTrue(np.all(np.diff(idx) > 0))
        self.assertEqual(len(data.downsample(4000)), 1500)
        self.assertTrue(math.isclose(np.diff(data.downsample(3000)).mean(), 2999 / 1499))


class CachedChartTest(TempHome):
    """docs/TERMINAL.md section 2 relies on prices.py dropping the bar of a session that is still open."""

    def chart(self, start):
        day = 86400
        ts = [start - 4 * day, start - 3 * day, start]  # Thursday, Friday, then Monday's session in progress
        return {"chart": {"result": [{
            "meta": {"gmtoffset": -14400, "currency": "USD", "longName": "Test Co",
                     "regularMarketTime": start + 4 * 3600,
                     "currentTradingPeriod": {"regular": {"start": start, "end": start + 23400}}},
            "timestamp": ts,
            "indicators": {"quote": [{"open": [10, 11, 12], "high": [10, 11, 12], "low": [10, 11, 12],
                                      "close": [10.0, 11.0, 12.0], "volume": [5, 6, 7]}],
                           "adjclose": [{"adjclose": [10.0, 11.0, 12.0]}]},
        }]}}

    def test_midday_body_reread_after_the_close_keeps_no_open_bar(self):
        start = int(pd.Timestamp("2026-10-12T13:30:00Z").timestamp())
        body = json.dumps(self.chart(start)).encode()
        cached = Response("https://query1.finance.yahoo.com/v8/finance/chart/TST", body,
                          "2026-10-12T17:32:04Z", True)  # fetched at midday, served from the cache
        after_close = start + 23400 + 3600
        with mock.patch.object(prices, "fetch", return_value=cached), \
             mock.patch.object(prices, "time", mock.Mock(time=mock.Mock(return_value=after_close))):
            facts = prices.ingest("TST")
        snap = facts[0]
        self.assertEqual((snap.as_of, snap.context["last"]), ("2026-10-09", "2026-10-09"))
        self.assertEqual(series.load(snap.value)["close"].tolist(), [10.0, 11.0])
        self.assertEqual(snap.retrieved_at, "2026-10-12T17:32:04Z")
        # The same body fetched after the close does hold the day's close.
        done = Response(cached.url, body, "2026-10-12T21:05:00Z", True)
        with mock.patch.object(prices, "fetch", return_value=done), \
             mock.patch.object(prices, "time", mock.Mock(time=mock.Mock(return_value=after_close))):
            df, _, _, _ = prices.fetch_daily("TST")
        self.assertEqual(df["date"].iloc[-1], "2026-10-12")



# ---------- v2: insider trades, earnings, aircraft, ports, cameras, watchlist, alerts ----------
TODAY = pd.Timestamp(today())
V2_DAYS = pd.bdate_range(TODAY - pd.DateOffset(years=7), END).strftime("%Y-%m-%d").tolist()
V2_YEARS = list(range(TODAY.year - 5, TODAY.year - 1))  # fully reported, whatever today's date
LA_BBOX = [33.60, -118.35, 33.80, -118.05]


def ago(days: int, hour: str = "T21:00:00Z") -> str:
    return (TODAY - pd.Timedelta(days=days)).strftime("%Y-%m-%d") + hour


def trade(t, code, shares, price, days, owner="Doe Jane", rel="director", title=None, accn=None, known=None, lot=None):
    ctx = {"code": code, "price": price, "value_usd": round(shares * price, 2), "owner": owner, "owner_cik": "0001",
           "relationship": rel, "title": title, "accn": accn or f"acc-{days}-{code}", "ownership": "D"}
    if lot:
        ctx["lot"] = lot
    return Fact(entity=t, attr="insider.trade", value=float(shares), unit="shares", as_of=ago(days + 2)[:10],
                known_at=known or ago(days), tier="filed",
                source={"name": "SEC Form 4", "url": f"https://www.sec.gov/Archives/edgar/data/1/{ctx['accn']}-index.htm"},
                context=ctx)


def ships(name, n, at, bbox=LA_BBOX, window=300):
    from brain.sources import shipping
    summary = {"vessels": n, "cargo": n // 2, "tanker": n // 4, "other": n - n // 2 - n // 4, "untyped": 0,
               "anchored": 1, "moored": n // 2, "underway": n - 1 - n // 2, "window_s": window, "bbox": list(bbox)}
    return shipping.to_fact(summary, {"name": name, "bbox": list(bbox)}, at)


def camera(cid, n, at, name="Example Rd / Depot"):
    return Fact(entity=f"CAM.{cid}", attr="camera.vehicles", value=n, unit="vehicles", as_of=at[:10], known_at=at,
                tier="reported", source={"name": "TfL JamCams", "url": "https://api.tfl.gov.uk/"},
                context={"car": n - 2, "truck": 1, "bus": 1, "motorcycle": 0, "provider": "TFL", "name": name,
                         "lat": 51.5, "lon": -0.1, "detector": "yolox-2022nov", "threshold": 0.5})


def alert(entity, rule, severity, msg, at, inputs=()):
    return Fact(entity=entity, attr="alert", value=msg, as_of=at[:10], known_at=at, tier="derived",
                source={"name": "brain.watch", "rule": rule},
                context={"rule": rule, "severity": severity, "keys": [f"{rule}:{msg}"]},
                inputs=tuple(f.ref for f in inputs), retrieved_at=at)


AIRPORTS = {"KSJC": "San Jose Intl", "KTEB": "Teterboro"}
SHIP_COUNTS = [10, 11, 9, 10, 12, 10, 11, 9, 10, 11, 10, 12, 10, 30]
CAM_COUNTS = [20, 22, 19, 21, 20, 23, 18, 20, 21, 22, 19, 21]


def build_v2(bank: Bank, home: Path) -> dict:
    from tests.test_quarterly import fiscal_year, known_dates
    rng = np.random.default_rng(11)
    px = {t: walk(rng, len(V2_DAYS)) for t in ("XCO", "SPY")}
    for t, v in px.items():
        bank.add(price_snapshot(t, V2_DAYS, v))
    bank.add(price_snapshot("NOSEC", V2_DAYS[-30:], px["SPY"][-30:]))  # prices only, like an ETF
    bank.add_many([filed("XCO", "identity.name", "Example Co", V2_DAYS[0], V2_DAYS[0]),
                   filed("XCO", "identity.cik", 1234, V2_DAYS[0], V2_DAYS[0]),
                   filed("ZCO", "identity.name", "Zed Co", V2_DAYS[0], V2_DAYS[0]),
                   filed("ZCO", "identity.cik", 5678, V2_DAYS[0], V2_DAYS[0])])
    # quarterly EPS: Q1-Q3 from 10-Qs and Q4 only as the year minus nine months; a release after each report
    eps = "us-gaap:EarningsPerShareDiluted"
    for k, y in enumerate(V2_YEARS):
        bank.add_many(fiscal_year("XCO", eps, y, [1.0 + 0.1 * k, 1.1 + 0.1 * k, 0.9 + 0.12 * k, 1.3 + 0.15 * k],
                                  known_dates(y)))
        for kd in known_dates(y):
            bank.add(Fact(entity="XCO", attr="filing", value="8-K", as_of=kd, known_at=kd + "T21:00:00Z", tier="filed",
                          source={"name": "SEC EDGAR", "url": f"https://www.sec.gov/x/8k-{kd}.htm"},
                          context={"items": "2.02,9.01", "accn": f"8k-{kd}"}))
    # insider trades: inside and outside the 90-day window, one filing read both live and in bulk
    live = trade("XCO", "P", 10000, 50.0, 30, owner="Doe Jane", rel="director", accn="acc-live",
                 known=ago(30, "T14:05:00Z"))
    trades = [live, trade("XCO", "P", 10000, 50.0, 30, owner="Doe Jane", rel="director", accn="acc-live",
                          known=ago(30, "T23:59:59Z")),  # the bulk copy of the same filing
              trade("XCO", "S", 2000, 55.0, 10, owner="Roe Richard", rel="officer", title="CFO"),
              trade("XCO", "P", 1000, 52.0, 40, owner="Big Fund LP", rel="ten_percent_owner"),
              trade("XCO", "P", 500, 40.0, 200, owner="Doe Jane", rel="director,officer", title="CEO")]
    bank.add_many(trades)
    # aircraft: two tails registered to the company; one of them opted out
    reg_at = ago(5, "T10:00:00Z")
    planes = [Fact(entity="XCO", attr="aircraft.registered", value=r, as_of=reg_at[:10], known_at=reg_at, tier="reported",
                   source={"name": "FAA Aircraft Registry", "url": "https://registry.faa.gov/database/ReleasableAircraft.zip"},
                   context={"hex": h_, "model": "GULFSTREAM G280", "year": 2018, "owner": "EXAMPLE CO"})
              for r, h_ in (("N1XC", "a00001"), ("N2XC", "a00002"))]
    flights = [Fact(entity="XCO", attr="aircraft.flight", value=to, as_of=arr[:10], known_at=arr, tier="reported",
                    source={"name": "adsb.lol (ODbL)", "url": "https://api.adsb.lol/"},
                    context={"hex": h_, "reg": r, "from": frm, "to": to, "from_name": AIRPORTS[frm],
                             "to_name": AIRPORTS[to], "departed": dep, "arrived": arr})
               for r, h_, frm, to, dep, arr in (
                   ("N1XC", "a00001", "KSJC", "KTEB", ago(3, "T12:00:00Z"), ago(3, "T17:10:00Z")),
                   ("N1XC", "a00001", "KTEB", "KSJC", ago(2, "T12:00:00Z"), ago(2, "T18:00:00Z")),
                   ("N2XC", "a00002", "KSJC", "KTEB", ago(1, "T12:00:00Z"), ago(1, "T17:00:00Z")))]
    bank.add_many(planes + flights)
    (home / "aviation_optout.txt").write_text("N2XC  # asked not to be tracked\n")
    (home / "aviation_links.json").write_text(json.dumps({"XCO": {"name": "Example Co", "registry_fetched_at": reg_at,
                                                                  "aircraft": [{"reg": "N1XC", "hex": "a00001"},
                                                                               {"reg": "N2XC", "hex": "a00002"}]}}))
    # port areas: a watched one with a jump in its last sample, and one sampled but not watched
    la = [ships("LA-LONG-BEACH", n, ago(len(SHIP_COUNTS) - i, "T12:00:00Z")) for i, n in enumerate(SHIP_COUNTS)]
    bank.add_many(la + [ships("TESTBOX", 4, ago(1, "T09:00:00Z"), bbox=[1, 2, 3, 4])])
    cams = [camera("TFL.00001.00001", n, ago(len(CAM_COUNTS) - i, "T08:00:00Z")) for i, n in enumerate(CAM_COUNTS)]
    bank.add_many(cams + [camera("TFL.00009.00009", 7, ago(1, "T07:00:00Z"), name="Elsewhere")])
    from brain import watch
    watch.save({"tickers": ["XCO", "SPY"], "aircraft_of": ["XCO"], "areas": [{"name": "LA-LONG-BEACH"}],
                "sites": [{"name": "Depot", "ticker": "XCO", "cameras": ["TFL.00001.00001", "TFL.00001.00002"]}]})
    alerts = [alert("XCO", "insider_buy", "high", "XCO insider buying: Doe Jane bought", ago(29, "T15:00:00Z"), [live]),
              alert("XCO", "aircraft", "notice", "XCO aircraft N1XC landed at KSJC", ago(2, "T18:30:00Z"), flights[1:2]),
              alert("AREA.LA-LONG-BEACH", "ships", "high", "AREA.LA-LONG-BEACH: 30 vessels, z +9.9", ago(1, "T12:05:00Z"),
                    la[-1:])]
    bank.add_many(alerts)
    return {"px": px, "trades": trades, "flights": flights, "la": la, "cams": cams, "alerts": alerts}


def v2_ledger() -> list[dict]:
    at = "2026-01-01T00:00:00Z"
    ctl = {"pass": True, "oracle_test_perm_p": 0.0005, "random_false_positive_rate": 0.03, "n_random": 200}
    ctx = {"spy_ann_return_test": 0.18, "universe_ew_ann_return_test": 0.27}
    recs = [{"op": "declare", "id": h, "spec_sha": "s", "spec": {"claim": f"claim {h}"}, "at": at}
            for h in ("H6-insider_buying", "H7-earnings_reaction", "H8-sue")]
    recs += [{"op": "result", "id": h, "spec_sha": "s", "at": at, "test": {"mean_net_excess": net, "perm_p": p},
              "train": {}, "controls": ctl, "context": ctx}
             for h, net, p in (("H6-insider_buying", -0.0012, 0.43), ("H7-earnings_reaction", 0.0004, 0.31))]
    return recs


class V2Base(unittest.TestCase):
    """A synthetic home with every kind of v2 data; tests that change the watchlist or acknowledge alerts
    restore what they changed."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.home = Path(cls._tmp.name)
        cls._old = os.environ.get("MONEY_HOME")
        os.environ["MONEY_HOME"] = str(cls.home)
        (cls.home / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in v2_ledger()))
        (cls.home / "universe.json").write_text(json.dumps({"tickers": ["XCO", "ZCO"]}))
        from brain import universe
        cls._patches = [mock.patch.object(config, "UNIVERSE", cls.home / "universe.json"),
                        mock.patch.object(config, "LEDGER", cls.home / "ledger.jsonl"),
                        mock.patch.object(universe, "BROAD", cls.home / "none.json")]
        for p in cls._patches:
            p.start()
        cls.bank = Bank()
        cls.syn = build_v2(cls.bank, cls.home)

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        if cls._old is None:
            os.environ.pop("MONEY_HOME", None)
        else:
            os.environ["MONEY_HOME"] = cls._old
        cls._tmp.cleanup()

    def run_ok(self, line, writes=False):
        r = data.run(self.bank, parse(line), writes=writes)
        self.assertTrue(r["ok"], (line, r.get("error"), r.get("hint")))
        json.dumps(r, allow_nan=False)
        for k in ("fn", "ticker", "args", "title", "asof", "view", "data"):
            self.assertIn(k, r)
        if r["view"] == "table":
            d = r["data"]
            keys = {c["key"] for c in d["columns"]}
            for row in d["rows"]:
                self.assertLessEqual({k for k in row if not k.startswith("_")}, keys, line)
                for k, cmd in row.get("_cmds", {}).items():
                    self.assertIn(k, keys)
                    self.assertTrue(parse(cmd).fn, cmd)  # every row command parses
        return r

    def run_err(self, line, writes=False):
        r = data.run_line(self.bank, line) if not writes else data.run(self.bank, parse(line), writes=True)
        self.assertFalse(r["ok"], line)
        self.assertTrue(r["error"])
        json.dumps(r, allow_nan=False)
        return r


class V2FunctionsTest(V2Base):
    def test_help_and_parse(self):
        codes = [f["code"] for f in self.run_ok("HELP")["data"]["functions"]]
        for c in ("INS", "ERN", "JET", "SHIP", "CAM", "WL", "ALRT"):
            self.assertIn(c, codes)

    # ---------- INS ----------
    def test_ins(self):
        r = self.run_ok("XCO INS")
        d = r["data"]
        self.assertEqual(r["title"], "XCO — Example Co · insider trades · 90 days: bought $552,000, sold $110,000")
        self.assertEqual([c["key"] for c in d["columns"]],
                         ["filed", "traded", "owner", "relationship", "code", "shares", "price", "value", "ownership"])
        self.assertEqual(len(d["rows"]), 4)  # the live and bulk copies of one filing count once
        filed_ = [row["filed"] for row in d["rows"]]
        self.assertEqual(filed_, sorted(filed_, reverse=True))
        self.assertEqual(r["asof"], filed_[0])
        first = d["rows"][0]
        self.assertEqual((first["code"], first["owner"], first["relationship"], first["value"]),
                         ("S", "Roe Richard", "officer (CFO)", 110000.0))
        self.assertTrue(first["_cite"] and first["_url"].startswith("https://www.sec.gov/"))
        self.assertIn("10% owner", [row["relationship"] for row in d["rows"]])
        self.assertTrue({row["code"] for row in d["rows"]} <= {"P", "S"})
        # 90 days: the director's $500,000 and the 10% owner's $52,000 bought, $110,000 sold
        self.assertIn("bought $552,000 in 2 trades and sold $110,000 in 1", d["note"])
        self.assertIn("officers and directors bought $500,000", d["note"])
        self.assertIn("Research: H6-insider_buying FAIL (TEST net excess -0.12% a month, permutation p 0.430)", d["note"])

    def test_ins_missing_and_wrong_kinds(self):
        self.assertEqual(self.run_err("^GSPC INS")["hint"], "^GSPC GP")
        self.assertEqual(self.run_err("NOSEC INS")["hint"], "INGEST NOSEC")  # prices only: no SEC filer
        self.assertEqual(self.run_err("QQQQ INS")["hint"], "INGEST QQQQ")
        r = self.run_err("ZCO INS")  # in a research universe, never read
        self.assertEqual((r["error"], r["hint"]), ("no insider trades for ZCO in the bank", "INGEST INSIDERS"))
        with mock.patch.object(config, "UNIVERSE", self.home / "none.json"):
            self.assertEqual(self.run_err("ZCO INS")["hint"], "python -m brain insiders ZCO")
        done = self.home / "insiders_done.json"
        done.write_text(json.dumps({"2025q1": [5678]}))
        try:
            d = self.run_ok("ZCO INS")["data"]  # read, and nothing was filed: an empty list, not an error
        finally:
            done.unlink()
        self.assertEqual(d["rows"], [])
        self.assertIn("No open-market purchases or sales", d["note"])

    # ---------- ERN ----------
    def test_ern(self):
        from brain import events
        r = self.run_ok("XCO ERN")
        d = r["data"]
        self.assertEqual([c["key"] for c in d["columns"]], ["period_end", "filed", "release", "eps", "eps_from", "eps_yoy",
                                                            "sue", "car_3d", "drift_20d", "drift_60d"])
        self.assertEqual(d["columns"][4]["type"], "flag")
        view = self.bank.view(today() + "T23:59:59Z")
        table = events.earnings_table(view, "XCO", PriceBook(self.bank, ["XCO", "SPY"], asof=today()))
        self.assertEqual(len(d["rows"]), len(table))
        self.assertEqual(len(d["rows"]), 4 * len(V2_YEARS))
        ends = [row["period_end"] for row in d["rows"]]
        self.assertEqual(ends, sorted(ends, reverse=True))
        self.assertEqual(ends[0], f"{V2_YEARS[-1]}-12-31")
        for row, t in zip(d["rows"], table):
            for k in ("eps", "eps_yoy", "sue", "car_3d", "drift_20d", "drift_60d"):
                if t[k] is None:
                    self.assertIsNone(row[k], k)
                else:
                    self.assertAlmostEqual(row[k], t[k], places=12, msg=k)
            self.assertEqual(row["eps_from"], "FY − 9M" if row["period_end"].endswith("12-31") else "")
        q4 = d["rows"][0]
        self.assertAlmostEqual(q4["eps"], 1.3 + 0.15 * (len(V2_YEARS) - 1))
        self.assertEqual(q4["release"], f"{V2_YEARS[-1] + 1}-02-20")
        self.assertIsNotNone(q4["car_3d"])  # windows long closed
        self.assertIsNotNone(q4["drift_60d"])
        self.assertEqual(q4["_cites"]["car_3d"], q4["_cites"]["release"])
        self.assertTrue(q4["_url"].endswith(f"8k-{V2_YEARS[-1] + 1}-02-20.htm"))
        self.assertEqual(len(q4["_refs"]), 3)  # the year, the nine months, the release
        self.assertIsNotNone(d["rows"][4]["sue"])  # enough history by the last year
        self.assertIn("Research: H7-earnings_reaction FAIL", d["note"])
        self.assertIn("Research: H8-sue has no result yet", d["note"])
        self.assertEqual(self.run_err("^GSPC ERN")["hint"], "^GSPC GP")
        self.assertEqual(self.run_err("ZCO ERN")["hint"], "INGEST ZCO")

    # ---------- JET ----------
    def test_jet(self):
        r = self.run_ok("XCO JET")
        d = r["data"]
        self.assertEqual(d["groups"], ["Registered aircraft", "Flights, newest first"])
        planes = [row for row in d["rows"] if row["_group"] == "Registered aircraft"]
        flights = [row for row in d["rows"] if row["_group"] == "Flights, newest first"]
        self.assertEqual([p_["reg"] for p_ in planes], ["N1XC"])  # N2XC opted out: not shown anywhere
        self.assertEqual([(f["from"], f["to"]) for f in flights],
                         [("KTEB Teterboro", "KSJC San Jose Intl"), ("KSJC San Jose Intl", "KTEB Teterboro")])
        self.assertEqual(flights[0]["departed"], ago(2, "T12:00:00Z"))
        self.assertEqual(flights[0]["model"], "GULFSTREAM G280")
        self.assertEqual(planes[0]["year"], "2018")
        self.assertEqual([c["key"] for c in d["columns"]], ["reg", "model", "year", "from", "to", "departed", "arrived"])
        self.assertNotIn("N2XC", json.dumps(r))
        self.assertIn("who was on board is not known and not inferred", d["note"])
        self.assertIn("2 flights recorded", d["note"])
        self.assertEqual(r["asof"], flights[0]["arrived"][:10])

    def test_jet_sold_aircraft_and_missing_links(self):
        links = self.home / "aviation_links.json"
        old = links.read_text()
        links.write_text(json.dumps({"XCO": {"registry_fetched_at": ago(0, "T01:00:00Z"), "aircraft": []}}))
        try:
            d = self.run_ok("XCO JET")["data"]  # a newer registry download no longer lists N1XC
        finally:
            links.write_text(old)
        self.assertEqual([row for row in d["rows"] if row["_group"] == "Registered aircraft"], [])
        self.assertEqual(len(d["columns"]), 7)  # its flights stay on record
        self.assertIn("No aircraft are registered in Example Co's own name", d["note"])
        state = self.home / "aviation_state.json"  # the poller saw N1XC's LADD flag: gone from JET, flights too
        state.write_text(json.dumps({"aircraft": {}, "blocked": {"A00001": "FAA LADD/PIA"}}))
        try:
            d = self.run_ok("XCO JET")["data"]
        finally:
            state.unlink()
        self.assertEqual(d["rows"], [])
        self.assertNotIn("N1XC", json.dumps(d))
        self.assertEqual(self.run_ok("WL")["data"]["rows"][2]["last"], 1)  # restored: one aircraft again
        r = self.run_err("ZCO JET")
        self.assertEqual(r["hint"], "python -m brain jets link ZCO")
        self.assertEqual(self.run_err("^GSPC JET")["hint"], "^GSPC GP")

    # ---------- SHIP ----------
    def test_ship_table(self):
        r = self.run_ok("SHIP")
        d = r["data"]
        self.assertEqual(d["groups"], ["Watched areas", "Other sampled areas"])
        rows = {row["area"]: row for row in d["rows"]}
        la = rows["LA-LONG-BEACH"]
        prior = np.array(SHIP_COUNTS[:-1], dtype=float)
        self.assertAlmostEqual(la["z"], (30 - prior.mean()) / prior.std(ddof=1))
        self.assertEqual((la["vessels"], la["n"], la["window_s"], la["_group"]), (30, len(SHIP_COUNTS), 300, "Watched areas"))
        self.assertEqual((la["cargo"], la["tanker"], la["moored"]), (15, 7, 15))
        self.assertEqual(la["_cmds"], {"area": "SHIP LA-LONG-BEACH"})
        self.assertEqual(la["_entity"], "AREA.LA-LONG-BEACH")
        box = rows["TESTBOX"]
        self.assertEqual((box["_group"], box["vessels"], box["z"]), ("Other sampled areas", 4, None))  # one sample: no z
        self.assertIn("pleasure and sailing craft are never counted", d["note"])

    def test_ship_area_chart(self):
        r = self.run_ok("SHIP LA-LONG-BEACH")
        d = r["data"]
        self.assertEqual((r["view"], d["kind"], d["unit"], d["entity"]), ("chart", "count", "vessels", "AREA.LA-LONG-BEACH"))
        self.assertEqual([x["value"] for x in d["series"]], SHIP_COUNTS)
        self.assertEqual(d["series"][-1]["parts"]["cargo"], 15)
        self.assertTrue(all(x["cite"] for x in d["series"]))
        st = d["stats"]
        self.assertEqual((st["last"], st["high"], st["low"], st["n"], st["z_samples"]), (30, 30, 9, 14, 13))
        self.assertAlmostEqual(st["mean"], float(np.mean(SHIP_COUNTS)))
        self.assertEqual(st["high_date"], d["series"][-1]["date"])
        self.assertIn("SHIP — LA-LONG-BEACH", r["title"])
        self.assertEqual(self.run_err("SHIP HOUSTON")["hint"], "python -m brain watch add area HOUSTON")
        self.assertIn("unknown area NOPE", self.run_err("SHIP NOPE")["error"])
        self.assertEqual(self.run_err("SHIP A B")["hint"], "SHIP A")
        self.assertIn("not an area name", self.run_err("SHIP .X")["error"])

    def test_ship_chart_downsamples(self):
        many = [ships("BIGAREA", 10 + i % 5, (TODAY - pd.Timedelta(minutes=15 * (2000 - i))).strftime("%Y-%m-%dT%H:%M:%SZ"))
                for i in range(2000)]
        self.bank.add_many(many)
        d = self.run_ok("SHIP BIGAREA")["data"]
        self.assertLessEqual(len(d["series"]), data.MAX_POINTS)
        self.assertEqual((d["series"][0]["date"], d["series"][-1]["date"]), (many[0].known_at, many[-1].known_at))
        self.assertEqual(d["stats"]["n"], 2000)

    # ---------- CAM ----------
    def test_cam(self):
        d = self.run_ok("CAM")["data"]
        self.assertEqual(d["groups"], ["Depot · XCO", "Other sampled cameras"])
        rows = {row["camera"]: row for row in d["rows"]}
        main = rows["TFL.00001.00001"]
        prior = np.array(CAM_COUNTS[:-1], dtype=float)
        self.assertAlmostEqual(main["z"], (CAM_COUNTS[-1] - prior.mean()) / prior.std(ddof=1))
        self.assertEqual((main["vehicles"], main["car"], main["n"], main["name"]), (21, 19, 12, "Example Rd / Depot"))
        self.assertEqual(main["_cmds"], {"camera": "CAM TFL.00001.00001"})
        self.assertEqual(rows["TFL.00001.00002"]["n"], 0)  # watched, never sampled
        self.assertNotIn("vehicles", rows["TFL.00001.00002"])
        self.assertEqual(rows["TFL.00009.00009"]["_group"], "Other sampled cameras")
        self.assertIn("no people, faces or number plates", d["note"])
        placeholder = camera("TFL.00009.00010", 0, "2026-10-12T10:00:00Z")  # a provider's "unavailable" frame
        self.bank.add(placeholder)
        self.bank.downgrade(placeholder.entity, placeholder.id, "placeholder frame")
        self.assertNotIn("TFL.00009.00010", {row["camera"] for row in self.run_ok("CAM")["data"]["rows"]})

    def test_cam_chart(self):
        for line in ("CAM TFL.00001.00001", "CAM cam.tfl.00001.00001"):
            r = self.run_ok(line)
            self.assertEqual((r["view"], r["data"]["kind"], r["data"]["unit"]), ("chart", "count", "vehicles"))
            self.assertEqual([x["value"] for x in r["data"]["series"]], CAM_COUNTS)
            self.assertEqual(set(r["data"]["series"][0]["parts"]), {"car", "truck", "bus", "motorcycle"})
        self.assertEqual(self.run_err("CAM TFL.00001.00002")["hint"], "INGEST WATCH")
        self.assertEqual(self.run_err("CAM TFL.7")["hint"], "python -m brain cams sample TFL.7")
        self.assertIn("not a camera id", self.run_err("CAM ../X")["error"])

    # ---------- WL ----------
    def test_wl(self):
        r = self.run_ok("WL")
        d = r["data"]
        self.assertEqual(d["groups"], ["Tickers", "Company aircraft", "Port areas", "Camera sites"])
        by = {(row["_group"], row["item"]): row for row in d["rows"]}
        xco = by[("Tickers", "XCO")]
        self.assertAlmostEqual(xco["last"], self.syn["px"]["XCO"][-1], places=4)
        self.assertAlmostEqual(xco["chg_1d"], self.syn["px"]["XCO"][-1] / self.syn["px"]["XCO"][-2] - 1, places=6)
        self.assertEqual((xco["name"], xco["measure"], xco["_ticker"]), ("Example Co", "close", "XCO"))
        self.assertTrue(xco["alert"].startswith("XCO insider buying"))  # the aircraft alert belongs to its own row
        self.assertEqual(xco["_cites"]["alert"], self.syn["alerts"][0].id[:8])
        self.assertIsNone(by[("Tickers", "SPY")]["alert"])
        news = alert("XCO", "headline", "info", "XCO headline: something", ago(0, "T00:00:01Z"))
        self.bank.add(news)  # newer, but only info: the insider alert stays the row's
        try:
            again = {(row["_group"], row["item"]): row for row in self.run_ok("WL")["data"]["rows"]}
            self.assertTrue(again[("Tickers", "XCO")]["alert"].startswith("XCO insider buying"))
        finally:
            self.bank.downgrade("XCO", news.id, "test fixture")
        jets = by[("Company aircraft", "XCO")]
        self.assertEqual((jets["last"], jets["_digits"], jets["_cmds"]), (1, 0, {"item": "XCO JET"}))
        self.assertTrue(jets["alert"].startswith("XCO aircraft N1XC landed"))
        area = by[("Port areas", "LA-LONG-BEACH")]
        self.assertEqual((area["last"], area["_entity"], area["_cmds"]), (30, "AREA.LA-LONG-BEACH", {"item": "SHIP LA-LONG-BEACH"}))
        self.assertTrue(area["alert"].startswith("AREA.LA-LONG-BEACH: 30 vessels"))
        cam = by[("Camera sites", "TFL.00001.00001")]
        self.assertEqual((cam["last"], cam["name"]), (21, "Depot · XCO"))
        self.assertEqual([c["key"] for c in d["columns"]],
                         ["item", "name", "last", "chg_1d", "alert", "alert_at", "measure", "date"])
        self.assertIsNone(by[("Camera sites", "TFL.00001.00002")]["last"])
        self.assertIn("3 unacknowledged alerts (ALRT)", d["note"])
        self.assertTrue(next(c for c in d["columns"] if c["key"] == "alert")["nowrap"])
        self.assertIn("a 1-day move of 5% or more", d["note"])

    def test_writes_need_the_server(self):
        for line in ("WL ADD AAPL", "WL DEL XCO", "ALRT ACK ALL", "ALRT ACK abcd"):
            r = self.run_err(line)
            self.assertIn("runs only from the terminal's own command line", r["error"])
            self.assertEqual(r["hint"], line.split()[0])
        self.assertTrue(data.is_write(parse("WL ADD AAPL")))
        self.assertFalse(data.is_write(parse("WL")))
        self.assertFalse(data.is_write(parse("ALRT")))
        from brain import watch
        self.assertEqual(watch.load()["tickers"], ["XCO", "SPY"])  # nothing changed

    def test_bad_arguments(self):
        self.assertEqual(self.run_err("WL FOO")["hint"], "WL")
        self.assertEqual(self.run_err("ALRT FOO")["hint"], "ALRT")
        for line in ("ALRT ACK", "ALRT ACK xyz!", "ALRT ACK 12", "ALRT ACK ab12 cd34"):
            self.assertIn("ALRT ACK takes one alert id", self.run_err(line, writes=True)["error"], line)
        self.assertIn("needs tickers", self.run_err("WL ADD", writes=True)["error"])

    # ---------- ALRT ----------
    def test_alrt(self):
        r = self.run_ok("ALRT")
        d = r["data"]
        self.assertEqual([c["key"] for c in d["columns"]], ["new", "at", "severity", "entity", "rule", "message", "ack"])
        self.assertEqual([c["type"] for c in d["columns"]][:3], ["flag", "date", "severity"])
        times = [row["at"] for row in d["rows"]]
        self.assertEqual(times, sorted(times, reverse=True))
        self.assertEqual((d["unacked"], d["high"]), (3, 2))
        self.assertTrue(all(row["new"] == "NEW" and row["ack"] == "ACK" for row in d["rows"]))
        area = d["rows"][0]
        self.assertEqual((area["entity"], area["severity"], area["rule"]), ("AREA.LA-LONG-BEACH", "high", "ships"))
        self.assertEqual(area["_cmds"]["entity"], "SHIP LA-LONG-BEACH")
        self.assertTrue(area["_cmds"]["ack"].startswith("ALRT ACK ") and len(area["_cmds"]["ack"].split()[-1]) == 12)
        self.assertEqual(d["rows"][1]["_ticker"], "XCO")
        self.assertEqual(r["asof"], times[0][:10])
        self.assertIn("3 of 3 unacknowledged", d["note"])

    def test_alrt_puts_high_before_info_within_one_check(self):
        at = ago(0, "T00:00:02Z")
        same = [alert("XCO", "headline", "info", "XCO headline: zzz", at), alert("XCO", "filing", "high", "XCO filed 8-K", at),
                alert("XCO", "price_move", "notice", "XCO -6.0%", at)]
        self.bank.add_many(same)
        try:
            rows = self.run_ok("ALRT")["data"]["rows"]
        finally:
            for f in same:
                self.bank.downgrade("XCO", f.id, "test fixture")
        self.assertEqual([r["severity"] for r in rows[:3]], ["high", "notice", "info"])
        self.assertEqual(rows[3]["at"], self.syn["alerts"][2].known_at)  # older checks after

    def test_no_nan_anywhere(self):
        for line in ("XCO INS", "XCO ERN", "XCO JET", "SHIP", "SHIP LA-LONG-BEACH", "CAM", "CAM TFL.00001.00001", "WL",
                     "ALRT"):
            self.run_ok(line)


class V2WritesTest(V2Base):
    """Commands that change the watchlist and the alert state, as the server runs them."""

    def test_wl_add_and_del(self):
        from brain import watch
        r = self.run_ok("WL ADD nvda aapl nvda", writes=True)
        self.assertEqual((r["fn"], r["args"]), ("WL", ["ADD", "NVDA", "AAPL", "NVDA"]))
        self.assertEqual(r["title"], "WL — watchlist · added NVDA, AAPL")
        self.assertEqual(watch.load()["tickers"], ["XCO", "SPY", "NVDA", "AAPL"])
        items = [row["item"] for row in r["data"]["rows"] if row["_group"] == "Tickers"]
        self.assertEqual(items, ["XCO", "SPY", "NVDA", "AAPL"])
        r = self.run_ok("WL DEL NVDA AAPL ZZZ", writes=True)
        self.assertEqual(r["title"], "WL — watchlist · removed NVDA, AAPL; not on it: ZZZ")
        self.assertEqual(watch.load()["tickers"], ["XCO", "SPY"])
        self.assertIn("is not a ticker", self.run_err("WL ADD A;B", writes=True)["error"])

    def test_wl_with_a_broken_file_is_an_error_and_is_kept(self):
        from brain import watch
        p = watch.watchlist_path()
        old = p.read_text()
        p.write_text("{not json")
        try:
            for line in ("WL", "WL ADD AAPL", "SHIP", "CAM"):
                r = self.run_err(line, writes=line.startswith("WL ADD"))
                self.assertIn("is not valid JSON", r["error"])
            self.assertEqual(p.read_text(), "{not json")  # never overwritten
        finally:
            p.write_text(old)

    def test_ack_one_then_all(self):
        from brain import watch
        state = watch.state_path()
        try:
            first = self.syn["alerts"][0]
            r = self.run_ok(f"ALRT ACK {first.id[:10]}", writes=True)
            self.assertEqual(r["title"], f"ALRT — alerts · acknowledged {first.id[:8]}")
            d = r["data"]
            self.assertEqual(d["unacked"], 2)
            row = next(x for x in d["rows"] if x["_cite"] == first.id[:8])
            self.assertEqual((row["new"], row["ack"], row.get("_dim"), "ack" in row["_cmds"]), ("", None, True, False))
            self.assertIn(first.id, watch.acked())
            self.assertIn("2 unacknowledged alerts", self.run_ok("WL")["data"]["note"])
            r = self.run_err("ALRT ACK 0000ffff", writes=True)
            self.assertEqual((r["error"], r["hint"]), ("no alert with id 0000ffff", "ALRT"))
            r = self.run_ok("ALRT ACK ALL", writes=True)
            self.assertEqual((r["title"], r["data"]["unacked"]), ("ALRT — alerts · acknowledged 2 alerts", 0))
            self.assertTrue(all(x.get("_dim") for x in r["data"]["rows"]))
        finally:
            state.unlink(missing_ok=True)


class V2EmptyHomeTest(TempHome):
    def test_global_functions_say_what_to_do(self):
        with mock.patch.object(config, "LEDGER", Path(self._tmp.name) / "none.jsonl"):
            for line, words in (("SHIP", "No port areas are watched or sampled yet. Built-in areas: LA-LONG-BEACH"),
                                ("CAM", "No camera sites are watched or sampled yet"),
                                ("WL", "The watchlist is empty. WL ADD AAPL adds a ticker"),
                                ("ALRT", "No alerts yet")):
                r = data.run_line(self.bank, line)
                self.assertTrue(r["ok"], (line, r.get("error")))
                self.assertEqual(r["data"]["rows"], [])
                self.assertIn(words, r["data"]["note"])
                json.dumps(r, allow_nan=False)
            self.assertEqual(data.run_line(self.bank, "ALRT")["data"]["unacked"], 0)
            self.assertEqual(data.run_line(self.bank, "AAPL INS")["hint"], "INGEST AAPL")
            self.assertEqual(data.run_line(self.bank, "AAPL ERN")["hint"], "INGEST AAPL")


if __name__ == "__main__":
    unittest.main()
