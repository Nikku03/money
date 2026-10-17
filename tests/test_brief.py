"""The brief's quarterly earnings and insider sections, and the broad-universe rows of its evidence."""
import json
import math
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from brain import brief, config, events, universe
from brain.bank import Fact
from brain.metrics import PriceBook
from brain.research import hypotheses, signals
from brain.research.panel import build as build_panel
from tests.test_quarterly import EPS, q
from tests.util import TempHome, filed, price_snapshot

H1_H4 = ["H1-mom12_1", "H2-rev1", "H3-earnings_yield", "H4-low_asset_growth"]
WITHDRAWN = ["H3b-earnings_yield_ttm", "H7-earnings_reaction", "H8-sue"]  # as the ledger records them
BROAD_IDS = ["H6-insider_buying", "H3c-earnings_yield_ttm", "H7b-earnings_reaction", "H8b-sue"]


def filing(t, form, known, items="", accn=None):
    return Fact(entity=t, attr="filing", value=form, as_of=known[:10], known_at=known, tier="filed",
                source={"name": "SEC EDGAR", "url": f"https://example.test/{accn or known}"},
                context={"accn": accn or known, **({"items": items} if items else {})})


def trade(t, code, known, traded, shares, price, accn, owner="Doe Jane", rel="officer,director",
          title="Chief Executive Officer"):
    # Made-up people: the fixtures mirror a filing's fields, not anyone's trades.
    return Fact(entity=t, attr="insider.trade", value=float(shares), unit="shares", as_of=traded, known_at=known,
                tier="filed", source={"name": "SEC Form 4", "url": f"https://example.test/f4/{accn}"},
                context={"code": code, "price": price, "value_usd": None if price is None else round(shares * price, 2),
                         "owner": owner, "owner_cik": "0000000501", "relationship": rel, "title": title, "accn": accn,
                         "ownership": "D"})


class BriefSectionsTest(TempHome):
    def setUp(self):
        super().setUp()
        days = pd.bdate_range("2023-01-02", "2024-09-27").strftime("%Y-%m-%d").tolist()  # known by the brief's date
        rng = np.random.default_rng(9)
        for t in ("X", "SPY", "^GSPC", "Y"):
            self.bank.add(price_snapshot(t, days, list(50 * np.cumprod(1 + rng.normal(0, 0.01, len(days))))))
        ends = pd.date_range("2021-12-31", periods=11, freq="QE").strftime("%Y-%m-%d").tolist()
        for t in ("X", "Y"):
            for v, end in zip([0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.3, 1.5, 1.6, 1.0, 1.4], ends):
                start = (pd.Timestamp(end) - pd.offsets.QuarterBegin(startingMonth=1)).strftime("%Y-%m-%d")
                known = (pd.Timestamp(end) + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
                self.bank.add(q(t, EPS, v, start, end, known, unit="USD/shares"))
        self.release = filing("X", "8-K", "2024-05-02T00:30:21Z", "2.02,9.01", "r1")
        self.bank.add_many([filing("X", "8-K", "2024-04-09T12:00:00Z", "2.02", "warn"), self.release,
                            filing("X", "8-K", "2024-07-31T13:00:00Z", "2.02", "r2")])
        tr = [trade("X", "P", f"2024-08-{d:02d}", f"2024-08-{d - 2:02d}", 1000 + d, 20.0, f"a{d}") for d in range(5, 17)]
        tr.append(trade("X", "S", "2024-09-20", "2024-09-18", 500, None, "s1", owner="Roe Rich", rel="director", title=None))
        tr.append(trade("X", "P", "2024-05-01", "2024-04-29", 999, 10.0, "old"))  # filed over 90 days before
        tr.append(trade("X", "P", "2024-09-10", "2024-09-06", 100, 30.0, "own", owner="Example Fund LP",
                        rel="ten_percent_owner", title=None))
        self.bank.add_many(tr)
        self.home = Path(self._tmp.name)
        self.p = mock.patch.object(config, "UNIVERSE", self.home / "none.json")  # no signal ranking here
        self.p.start()

    def tearDown(self):
        self.p.stop()
        super().tearDown()

    def test_quarterly_earnings_section(self):
        md = brief.build(self.bank, "X", "2024-09-30")
        sec = md.split("## Quarterly earnings, last 8 quarters (SEC filings)")[1].split("\n## ")[0]
        rows = [r for r in sec.splitlines() if r.startswith("| 20")]
        view = self.bank.view("2024-09-30")
        table = events.earnings_table(view, "X", PriceBook(self.bank, ["X", "SPY"], asof="2024-09-30"))
        self.assertEqual(len(rows), 8)
        self.assertEqual([r.split(" | ")[0][2:] for r in rows], [r["period_end"] for r in table[:8]])
        q1 = next(r for r in table if r["period_end"] == "2024-03-31")
        line = next(r for r in rows if r.startswith("| 2024-03-31 |"))
        self.assertIn(f"| 2024-05-05 | 2024-05-02 | 1.00 | {brief.pct(q1['eps_yoy'])} |", line)  # the results, not the warning
        self.assertIn(f"| {brief.pct(q1['car_3d'])} | {brief.pct(q1['drift_20d'])} |", line)
        self.assertIn(f"[{self.release.id[:8]}]", line)
        self.assertIn(f"[{q1['cite'][0].partition('/')[2][:8]}]", line)  # the EPS fact is cited too
        self.assertIsNotNone(q1["car_3d"])
        self.assertIn("| 2022-09-30 | 2022-11-04 | – | 1.20 | – | – | – | – |", md)  # nothing to compare or react to
        self.assertNotIn("| 2022-06-30 |", md)  # the ninth quarter back

    def test_insider_section_totals_trades_and_roles(self):
        md = brief.build(self.bank, "X", "2024-09-30")
        sec = md.split("## Insider trades, last 90 days (SEC Forms 4 and 5)")[1].split("\n## ")[0]
        buys = sum((1000 + d) * 20.0 for d in range(5, 17)) + 100 * 30.0
        self.assertIn(f"- Bought ${brief.money(buys)} in 13 trades; sold $0.00 in 1. Officers and directors bought "
                      f"${brief.money(buys - 3000.0)}", sec)
        rows = [r for r in sec.splitlines() if r.startswith("| 2024")]
        self.assertEqual(len(rows), 10)
        self.assertTrue(rows[0].startswith("| 2024-09-20 | 2024-09-18 | Roe Rich | director | S, sale | 500 | – | – |"))
        self.assertIn("| Example Fund LP | 10% owner | P, buy | 100 | 30.00 | 3.00K |", sec)
        self.assertIn("| Doe Jane | officer, director (Chief Executive Officer) | P, buy |", sec)
        self.assertIn("4 more in the window", sec)
        self.assertNotIn("2024-04-29", sec)          # filed before the window
        self.assertNotIn("0000000501", md)           # a filing's role and name, nothing else about the person
        self.assertNotIn("not parsed yet", md)

    def test_no_trades_and_market_symbols(self):
        md = brief.build(self.bank, "Y", "2024-09-30")
        self.assertIn("- No insider trades in the bank for Y: `python -m brain insiders Y` reads them.", md)
        self.bank.add(filed("Y", "identity.cik", 222, "2024-01-01", "2024-01-01"))
        (self.home / "insiders_done.json").write_text(json.dumps({"2024q2": [222]}))
        md = brief.build(self.bank, "Y", "2024-09-30")
        self.assertIn("- No open-market purchases or sales filed in the 90 days to 2024-09-30.", md)
        self.assertIn("## Quarterly earnings", md)
        idx = brief.build(self.bank, "^GSPC", "2024-09-30")
        self.assertNotIn("## Quarterly earnings", idx)
        self.assertNotIn("## Insider trades", idx)
        early = brief.build(self.bank, "X", "2024-05-03")  # point-in-time: Q1 2024 EPS not filed yet
        self.assertNotIn("| 2024-03-31 |", early)
        self.assertIn("| 2023-12-31 |", early)


def _universe_bank(bank, large, broad):
    """Prices 2009..mid-2021; fiscal-year facts for ``large`` (H1-H4); quarterly EPS, net income, shares,
    earnings 8-Ks and director purchases for ``broad`` (H3c, H6, H7b, H8b)."""
    days = pd.bdate_range("2009-01-01", "2021-06-30").strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(3)
    for t in [*broad, "SPY"]:
        bank.add(price_snapshot(t, days, list(50 * np.cumprod(1 + rng.normal(0.0003, 0.015, len(days))))))
    for i, t in enumerate(large):
        for y in range(2009, 2021):
            bank.add_many([
                filed(t, "us-gaap:NetIncomeLoss", 10.0 + i + y % 3, f"{y}-12-31", f"{y + 1}-02-20",
                      form="10-K", start=f"{y}-01-01", duration_days=364),
                filed(t, "us-gaap:Assets", 100.0 * (1 + 0.05 * i) ** (y - 2008), f"{y}-12-31", f"{y + 1}-02-20", form="10-K"),
                filed(t, "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding", 10.0, f"{y}-12-31",
                      f"{y + 1}-02-20", form="10-K", start=f"{y}-01-01", duration_days=364)])
    ends = pd.date_range("2009-03-31", "2021-03-31", freq="QE")
    for i, t in enumerate(broad):
        facts = []
        for k, e in enumerate(ends):
            start = (e - pd.offsets.QuarterBegin(startingMonth=1)).strftime("%Y-%m-%d")
            end, known = e.strftime("%Y-%m-%d"), (e + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
            dur = (e - pd.Timestamp(start)).days
            eps = 1.0 + 0.1 * i + 0.3 * rng.normal()
            ni = filed(t, "us-gaap:ProfitLoss", eps * 10, end, known, form="10-Q", start=start, duration_days=dur)
            facts += [filed(t, "us-gaap:EarningsPerShareDiluted", eps, end, known, form="10-Q", start=start,
                            duration_days=dur),
                      Fact(**{**{k_: getattr(ni, k_) for k_ in ("entity", "attr", "value", "as_of", "known_at", "tier",
                                                                 "source", "context")}, "unit": "USD"}),
                      filing(t, "8-K", known + "T21:30:00Z", "2.02,9.01", f"{t}-{k}")]
            if (k + i) % 2 == 0:
                facts.append(trade(t, "P", known, known, 100, 50.0 * (1 + i), f"f4-{t}-{k}", rel="director", title=None))
        for y in range(2009, 2021):
            facts.append(filed(t, "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic", 10.0, f"{y}-12-31",
                               f"{y + 1}-02-20", form="10-K", start=f"{y}-01-01", duration_days=364))
        facts.append(filed(t, "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic", 10.0, "2021-03-31",
                           "2021-05-05", form="10-Q", start="2021-01-01", duration_days=89))
        bank.add_many(facts)


class BroadEvidenceTest(TempHome):
    def setUp(self):
        super().setUp()
        self.home = Path(self._tmp.name)
        self.large = [f"L{i}" for i in range(5)]
        self.broad = [*self.large, *(f"B{i}" for i in range(5, 11))]
        _universe_bank(self.bank, self.large, self.broad)
        (self.home / "large.json").write_text(json.dumps({"tickers": self.large, "rule_short": "synthetic"}))
        (self.home / "broad.json").write_text(json.dumps({"tickers": self.broad, "rule_short": "synthetic broad"}))
        ctl = {"pass": True, "oracle_test_perm_p": 0.0005, "random_false_positive_rate": 0.04, "n_random": 200}
        recs = [{"op": "result", "id": h, "spec_sha": "s", "at": "2026-01-01T00:00:00Z", "controls": ctl,
                 "test": {"mean_net_excess": net, "perm_p": p}, "train": {}, "context": {}}
                for h, net, p in (("H1-mom12_1", 0.0011, 0.18), ("H6-insider_buying", 0.00002, 0.28))]
        recs += [{"op": "withdraw", "id": h, "reason": "a code mistake", "at": "2026-01-02T00:00:00Z"} for h in WITHDRAWN]
        (self.home / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        self.patches = [mock.patch.object(config, "UNIVERSE", self.home / "large.json"),
                        mock.patch.object(config, "LEDGER", self.home / "ledger.jsonl"),
                        mock.patch.object(universe, "BROAD", self.home / "broad.json")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        super().tearDown()

    def test_broad_rows_rank_like_the_backtest_and_leave_h1_h4_alone(self):
        ev = brief.evidence(self.bank, "B7", budget_s=None)
        self.assertEqual([r["hypothesis"] for r in ev["signals"]], H1_H4 + BROAD_IDS)
        self.assertEqual(ev["broad"]["status"], "complete")
        self.assertEqual((ev["broad"]["universe"], ev["broad"]["universe_size"]), ("universe/us_broad.json", 11))
        # H1-H4: the rows they were before broad rows existed.
        with mock.patch.object(universe, "BROAD", self.home / "missing.json"):
            alone = brief.evidence(self.bank, "B7", budget_s=None)
        self.assertEqual(ev["signals"][:4], alone["signals"])
        self.assertEqual(alone["broad"]["status"], "unavailable")
        self.assertTrue(alone["summary"].startswith("0 of 4 signals"))
        self.assertTrue(ev["summary"].startswith("0 of 8 signals"))
        # The broad values are the backtest's on its last rebalance day.
        panel = build_panel(self.bank, self.broad)
        S = signals.compute(self.bank, panel, list(signals.PER_TICKER))
        j = self.broad.index("B7")
        sel = {h["id"]: h for h in hypotheses.HYPOTHESES}
        for r in ev["signals"][4:]:
            v = S[r["signal"]][-1]
            self.assertAlmostEqual(r["value"], v[j], msg=r["signal"])
            ok = np.isfinite(v) & panel.listed[-1]
            self.assertEqual(r["percentile"], round(float((v[ok] < v[j]).sum() / (ok.sum() - 1)) * 100))
            self.assertEqual(r["select"], sel[r["hypothesis"]]["select"])
            self.assertEqual(r["universe"], "universe/us_broad.json")
            self.assertIsInstance(r["held"], bool)
        rows = {r["hypothesis"]: r for r in ev["signals"]}
        self.assertEqual((rows["H6-insider_buying"]["verdict"], rows["H8b-sue"]["verdict"]), ("FAIL", "not run"))
        self.assertEqual(rows["H6-insider_buying"]["held"], rows["H6-insider_buying"]["value"] > 0)
        json.dumps(ev, allow_nan=False)

    def test_budget_cache_and_what_invalidates_it(self):
        ev = brief.evidence(self.bank, "B7", budget_s=0)  # only the asked-for stock fits
        b = ev["broad"]
        self.assertEqual((b["status"], b["computed"], b["total"]), ("pending", 1, 11))
        for r in ev["signals"][4:]:
            self.assertIsNotNone(r["value"])
            self.assertIsNone(r["percentile"])
            self.assertIsNone(r["held"])
        self.assertIn("not complete yet (1 of 11 stocks)", ev["summary"])
        md = brief.build(self.bank, "B7", evidence_budget_s=0)
        self.assertIn("Ranking not complete:", md)
        full = brief.evidence(self.bank, "B7", budget_s=None)
        self.assertEqual(full["broad"]["status"], "complete")
        again = brief.evidence(self.bank, "B7", budget_s=0)  # all cached
        self.assertEqual(again["broad"]["status"], "complete")
        self.assertEqual(again["signals"], full["signals"])
        md = brief.build(self.bank, "B7", evidence_budget_s=0)
        self.assertIn("| insider_buying | H6-insider_buying | positive |", md)
        d = full["broad"]["rebalance_date"]
        # Known after the rebalance day: that day's view cannot see it, the cache stands.
        self.bank.add(filing("B9", "8-K", "2021-06-15T21:30:00Z", "2.02", "late"))
        self.assertEqual(brief.evidence(self.bank, "B7", budget_s=0)["broad"]["status"], "complete")
        # A backfill known before it (an insider data set read later) changes what that view sees.
        self.bank.add(trade("B9", "P", f"{d[:8]}10", f"{d[:8]}08", 1000, 40.0, "backfill", rel="officer", title="CFO"))
        self.assertEqual(brief.evidence(self.bank, "B7", budget_s=0)["broad"]["status"], "pending")
        before = {r["hypothesis"]: r for r in full["signals"]}["H6-insider_buying"]
        after = brief.evidence(self.bank, "B9", budget_s=None)
        h6 = {r["hypothesis"]: r for r in after["signals"]}["H6-insider_buying"]
        panel = build_panel(self.bank, self.broad)
        S = signals.compute(self.bank, panel, ["insider_buying"])["insider_buying"]
        self.assertAlmostEqual(h6["value"], S[-1][self.broad.index("B9")])
        self.assertGreater(h6["value"], 40000.0 - 1e-6)
        self.assertTrue(math.isfinite(before["value"]))

    def test_a_bank_without_the_broad_universe_gets_no_broad_rows(self):
        (self.home / "broad.json").write_text(json.dumps({"tickers": self.broad + [f"Z{i}" for i in range(20)]}))
        ev = brief.evidence(self.bank, "L1", budget_s=None)
        self.assertEqual([r["hypothesis"] for r in ev["signals"]], H1_H4)
        self.assertEqual(ev["broad"]["status"], "unavailable")
        self.assertIn("only 11 of its 31 stocks are in the bank, too few to rank", ev["broad"]["reason"])
        md = brief.build(self.bank, "L1")
        self.assertIn("Not ranked against `universe/us_broad.json`: only 11 of its 31 stocks are in the bank, too few to rank", md)
