import contextlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from brain import config, universe
from brain.bank import Fact
from brain.research import harness, hypotheses, ledger, run, signals
from tests.util import TempHome, filed, price_snapshot

H1_H5 = ["H1-mom12_1", "H2-rev1", "H3-earnings_yield", "H4-low_asset_growth", "H5-ridge"]
NEW = ["H3b-earnings_yield_ttm", "H6-insider_buying", "H7-earnings_reaction", "H8-sue"]


def usd(f: Fact) -> Fact:
    return Fact(**{**{k: getattr(f, k) for k in ("entity", "attr", "value", "as_of", "known_at", "tier", "source",
                                                  "context")}, "unit": "USD"})


def _strip_volatile(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("at", "commit", "brain_dirty")}


class DeclaredSpecsTest(TempHome):
    def test_h1_h5_specs_are_unchanged_and_new_ones_are_well_formed(self):
        recorded = {r["id"]: r["spec_sha"] for r in ledger.read(config.ROOT / "research" / "ledger.jsonl")
                    if r["op"] == "declare"}
        by_id = {h["id"]: h for h in hypotheses.HYPOTHESES}
        for hid in H1_H5:  # adding H3b..H8 (and select modes) must not move a declared hash
            self.assertEqual(ledger.spec_sha(by_id[hid]), recorded[hid], hid)
            self.assertNotIn("select", by_id[hid])
        self.assertEqual(len(by_id), len(hypotheses.HYPOTHESES))
        for hid in NEW:
            h = ledger.spec(by_id[hid])
            self.assertEqual(h["universe"], "universe/us_broad.json")
            self.assertIn(h["select"], harness.SELECT)
            self.assertIn(h["signal"], signals.PER_TICKER)
            self.assertEqual((h["split"], h["gate"], h["cost_bps"], h["frac"]),
                             (hypotheses.SPLIT, hypotheses.GATE, 10, 1 / 3))
        self.assertEqual({by_id[h]["select"] for h in NEW}, {"top_frac", "positive", "top_frac_vs_all"})
        self.assertEqual(run.universe_path("universe/us_broad.json"), universe.BROAD)


class RunEndToEndTest(TempHome):
    def setUp(self):
        super().setUp()
        self.home = Path(self._tmp.name)
        days = pd.bdate_range("2009-01-01", "2021-06-30").strftime("%Y-%m-%d").tolist()
        rng = np.random.default_rng(3)
        self.large = [f"T{i}" for i in range(9)]
        self.broad = [*self.large, "T9", "T10", "T11"]
        for t in [*self.broad, "SPY"]:
            px = list(50 * np.cumprod(1 + rng.normal(0.0003, 0.015, len(days))))
            self.bank.add(price_snapshot(t, days, px))
        for i, t in enumerate(self.large):
            for y in range(2009, 2021):
                self.bank.add_many([
                    filed(t, "us-gaap:NetIncomeLoss", 10.0 + i + y % 3, f"{y}-12-31", f"{y + 1}-02-20",
                          form="10-K", start=f"{y}-01-01", duration_days=364),
                    filed(t, "us-gaap:Assets", 100.0 * (1 + 0.05 * i) ** (y - 2008), f"{y}-12-31", f"{y + 1}-02-20", form="10-K"),
                    filed(t, "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding", 10.0, f"{y}-12-31",
                          f"{y + 1}-02-20", form="10-K", start=f"{y}-01-01", duration_days=364),
                ])
        ends = pd.date_range("2009-03-31", "2021-03-31", freq="QE")
        for i, t in enumerate(self.broad):
            facts = []
            for k, e in enumerate(ends):
                start = (e - pd.offsets.QuarterBegin(startingMonth=1)).strftime("%Y-%m-%d")
                end, known = e.strftime("%Y-%m-%d"), (e + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
                dur = (e - pd.Timestamp(start)).days
                eps = 1.0 + 0.1 * i + 0.3 * rng.normal()
                facts += [filed(t, "us-gaap:EarningsPerShareDiluted", eps, end, known, form="10-Q", start=start,
                                duration_days=dur),
                          usd(filed(t, "us-gaap:ProfitLoss", eps * 10, end, known, form="10-Q", start=start,
                                    duration_days=dur)),
                          Fact(entity=t, attr="filing", value="8-K", as_of=known, known_at=known + "T21:30:00Z",
                               tier="filed", source={"name": "SEC EDGAR", "url": f"https://example.test/{t}/{k}"},
                               context={"accn": f"{t}-{k}", "items": "2.02,9.01"})]
                if (k + i) % 2 == 0:  # a director buys now and then
                    facts.append(Fact(entity=t, attr="insider.trade", value=100.0, unit="shares", as_of=known,
                                      known_at=known, tier="filed", source={"name": "SEC Form 4", "url": "https://x"},
                                      context={"code": "P", "price": 50.0, "value_usd": 5000.0 * (1 + i),
                                               "owner": "Doe Jane", "owner_cik": "1", "relationship": "director",
                                               "title": None, "accn": f"f4-{t}-{k}", "ownership": "D"}))
            facts.append(filed(t, "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic", 10.0, "2021-03-31",
                               "2021-05-05", form="10-Q", start="2021-01-01", duration_days=89))
            for y in range(2009, 2021):
                facts.append(filed(t, "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic", 10.0, f"{y}-12-31",
                                   f"{y + 1}-02-20", form="10-K", start=f"{y}-01-01", duration_days=364))
            self.bank.add_many(facts)
        self.large_file = self.home / "universe.json"
        self.large_file.write_text(json.dumps({"tickers": self.large, "rule_short": "synthetic"}))
        self.broad_file = self.home / "us_broad.json"
        self.broad_file.write_text(json.dumps({"tickers": self.broad, "rule_short": "synthetic broad"}))

    def patched(self, real_sha: bool = False) -> contextlib.ExitStack:
        """Ledger, results and both universes in the temp home. The universe sha check sees the
        declared value unless ``real_sha`` (the synthetic file's sha is not the real one)."""
        st = contextlib.ExitStack()
        st.enter_context(mock.patch.object(config, "LEDGER", self.home / "ledger.jsonl"))
        st.enter_context(mock.patch.object(config, "RESULTS", self.home / "RESULTS.md"))
        st.enter_context(mock.patch.object(config, "UNIVERSE", self.large_file))
        st.enter_context(mock.patch.object(universe, "BROAD", self.broad_file))
        st.enter_context(mock.patch.dict("brain.research.hypotheses.GATE", {"n_perm": 200}))
        if not real_sha:
            st.enter_context(mock.patch.object(run, "universe_sha", lambda name=None: hypotheses.BROAD_SHA))
        return st

    def test_declared_hypotheses_run_and_report(self):
        with self.patched():
            with self.assertRaises(SystemExit):  # undeclared: refused
                run.run(self.bank, n_random=5)
            ledger.declare_all()
            alone = run.run(self.bank, H1_H5, n_random=20)
            out = run.run(self.bank, n_random=20)
        self.assertEqual([r["id"] for r in out], H1_H5 + NEW)
        for r in out:  # synthetic releases come 35 days after each quarter: one month in three has one
            self.assertGreater(r["test"]["n_months"], 6 if r["id"] == "H7-earnings_reaction" else 20, r["id"])
            self.assertGreater(r["test_signal_coverage"], 0.2, r["id"])
        # H1-H5 on their own panel come out exactly as when they ran alone.
        for a, b in zip(alone, out[:5]):
            self.assertEqual(_strip_volatile(a), _strip_volatile(b))
            self.assertNotIn("universe", b)
        new = {r["id"]: r for r in out[5:]}
        self.assertEqual({r["universe"] for r in new.values()}, {"universe/us_broad.json"})
        self.assertEqual({r["n_tickers"] for r in new.values()}, {12})
        self.assertNotEqual(new["H8-sue"]["panel_digest"], out[0]["panel_digest"])
        self.assertAlmostEqual(new["H6-insider_buying"]["test_signal_coverage"], 1.0)  # 0 is a value
        text = (self.home / "RESULTS.md").read_text()
        for hid in H1_H5 + NEW:
            self.assertIn(hid, text)
        self.assertIn("`universe/us_broad.json`: 12 US stocks (synthetic broad)", text)
        self.assertIn("| H6-insider_buying | us_broad | positive |", text)

    def test_pinned_prices_reproduce_a_run_after_new_prices_arrive(self):
        with self.patched():
            ledger.declare_all()
            first = run.run(self.bank, ["H1-mom12_1"], n_random=5)[0]
            days = pd.bdate_range("2009-01-01", "2021-09-30").strftime("%Y-%m-%d").tolist()
            for t in [*self.large, "SPY"]:  # a later price snapshot: three more months
                self.bank.add(price_snapshot(t, days, list(np.linspace(40, 60, len(days)))))
            moved = run.run(self.bank, ["H1-mom12_1"], n_random=5)[0]
            pinned = run.run(self.bank, ["H1-mom12_1"], n_random=5, asof="2021-07-01")[0]
        self.assertNotEqual(moved["panel_digest"], first["panel_digest"])
        self.assertEqual(pinned["panel_digest"], first["panel_digest"])
        self.assertEqual(pinned["test"], first["test"])
        self.assertEqual(pinned["prices_asof"], "2021-07-01")

    def test_a_changed_universe_file_is_refused(self):
        hyps = [{**h, "universe_sha": "0" * 16} if h["id"] == "H8-sue" else h for h in hypotheses.HYPOTHESES]
        with self.patched(real_sha=True), mock.patch.object(run, "HYPOTHESES", hyps), \
                mock.patch.object(ledger, "HYPOTHESES", hyps):
            ledger.declare_all()
            with self.assertRaises(SystemExit) as cm:
                run.run(self.bank, ["H8-sue"], n_random=5)
        self.assertIn("declare a new id", str(cm.exception))
