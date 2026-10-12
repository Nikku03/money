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
        self.assertEqual([c["type"] for c in d["columns"]], ["text", "text", "pct", "number", "number", "verdict"])
        self.assertIn("Controls passed", d["note"])
        self.assertIn("+27.0% a year", d["note"])

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


if __name__ == "__main__":
    unittest.main()
