import contextlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from brain import config, universe
from brain.bank import Fact
from brain.research import harness, hypotheses, ledger, run, signals
from tests.util import TempHome, filed, price_snapshot

H1_H5 = ["H1-mom12_1", "H2-rev1", "H3-earnings_yield", "H4-low_asset_growth", "H5-ridge"]
NEW = ["H3b-earnings_yield_ttm", "H6-insider_buying", "H7-earnings_reaction", "H8-sue",
       "H3c-earnings_yield_ttm", "H7b-earnings_reaction", "H8b-sue"]
LARGE_NEW = ["H10-text_similarity"]   # on the large universe, so run with H1-H5's group


def usd(f: Fact) -> Fact:
    return Fact(**{**{k: getattr(f, k) for k in ("entity", "attr", "value", "as_of", "known_at", "tier", "source",
                                                  "context")}, "unit": "USD"})


def _strip_volatile(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("at", "commit", "brain_dirty")}


class DeclaredSpecsTest(TempHome):
    def test_h1_h5_specs_are_unchanged_and_new_ones_are_well_formed(self):
        recorded = {r["id"]: r["spec_sha"] for r in ledger.read(config.ROOT / "research" / "ledger.jsonl")
                    if r["op"] == "declare" and r.get("kind") is None}  # forecast and link pins are checked at run time
        by_id = {h["id"]: h for h in hypotheses.HYPOTHESES}
        for hid in recorded:  # a declared spec is never edited: adding H3b..H8b (and select modes) moves no hash
            self.assertEqual(ledger.spec_sha(by_id[hid]), recorded[hid], hid)
        for hid in H1_H5:
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
        for hid in LARGE_NEW:
            h = ledger.spec(by_id[hid])
            self.assertEqual((h["universe"], h["select"]), ("universe/us_large.json", "top_frac"))
            self.assertIn(h["signal"], signals.PER_TICKER)


    def test_a_withdrawn_hypothesis_gives_way_to_its_successor(self):
        ids = lambda gone: [h["id"] for h in hypotheses.current(gone)]  # noqa: E731
        self.assertEqual(ids(()), [h["id"] for h in hypotheses.HYPOTHESES])
        shown = ids({"H7-earnings_reaction": "x", "H8-sue": "x", "H8b-sue": "x", "H2-rev1": "x"})
        self.assertNotIn("H7-earnings_reaction", shown)
        self.assertIn("H7b-earnings_reaction", shown)
        self.assertNotIn("H8-sue", shown)       # H8b follows it, withdrawn too: the last one stays
        self.assertIn("H8b-sue", shown)
        self.assertIn("H2-rev1", shown)         # no successor: its withdrawal is still shown

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
            for y in range(2009, 2022):   # report-to-report similarity, one per quarter (brain.changes)
                for q, (end, filed_on) in enumerate([("03-31", "05-05"), ("06-30", "08-05"), ("09-30", "11-05"),
                                                     ("12-31", "02-20")]):
                    k = f"{y + (q == 3)}-{filed_on}"
                    if k > "2021-06-30":
                        continue
                    self.bank.add(Fact(entity=t, attr="text.similarity", value=float(rng.uniform(0.9, 1.0)),
                                       as_of=f"{y}-{end}", known_at=k, tier="derived",
                                       source={"name": "test", "url": "https://example.test/text"},
                                       inputs=(f"{t}/x",), context={"form": "10-K" if q == 3 else "10-Q",
                                                                    "method": 1, "new_share": 0.3}))
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
        self.assertEqual([r["id"] for r in out], H1_H5 + LARGE_NEW + NEW)
        for r in out:  # synthetic releases come 35 days after each quarter: one month in three has one
            self.assertGreater(r["test"]["n_months"], 6 if r["id"].startswith("H7") else 20, r["id"])
            self.assertGreater(r["test_signal_coverage"], 0.2, r["id"])
        # H1-H5 on their own panel come out exactly as when they ran alone.
        for a, b in zip(alone, out[:5]):
            self.assertEqual(_strip_volatile(a), _strip_volatile(b))
            self.assertNotIn("universe", b)
        new = {r["id"]: r for r in out[5 + len(LARGE_NEW):]}
        self.assertNotIn("universe", out[5])   # H10 runs on the large universe, with H1-H5's panel
        self.assertEqual(out[5]["panel_digest"], out[0]["panel_digest"])
        self.assertEqual({r["universe"] for r in new.values()}, {"universe/us_broad.json"})
        self.assertEqual({r["n_tickers"] for r in new.values()}, {12})
        self.assertNotEqual(new["H8-sue"]["panel_digest"], out[0]["panel_digest"])
        self.assertAlmostEqual(new["H6-insider_buying"]["test_signal_coverage"], 1.0)  # 0 is a value
        text = (self.home / "RESULTS.md").read_text()
        for hid in H1_H5 + LARGE_NEW + NEW:
            self.assertIn(hid, text)
        self.assertIn("`universe/us_broad.json`: 12 US stocks (synthetic broad)", text)
        self.assertIn("| H6-insider_buying | us_broad | positive |", text)
        self.assertNotIn("The null is calibrated.", text)  # the control tests monthly redraws, nothing more
        self.assertIn("20 random scores held fixed per stock in every month:", text)
        for r in out:
            self.assertIn("persistent_random_false_positive_rate", r["controls"])

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


class ControlsTest(unittest.TestCase):
    def test_a_random_score_fixed_per_stock_is_reported_not_gated(self):
        # Stocks with their own mean return: holding the same names every month spreads results wider than
        # the null's monthly redraws, so a random score fixed per stock "passes" too often.
        rng = np.random.default_rng(11)
        M, N = 60, 45
        R = rng.normal(0, 0.03, N) + rng.normal(0, 0.05, (M, N))
        panel = SimpleNamespace(R=R, listed=np.ones((M + 1, N), bool), entry=np.arange(M))
        ctl = run.controls(panel, np.arange(M), n_random=40)
        self.assertGreater(ctl["persistent_random_false_positive_rate"], ctl["random_false_positive_rate"] + 0.1)
        self.assertEqual(ctl["pass"], ctl["oracle_test_perm_p"] < 0.001 and 0.01 <= ctl["random_false_positive_rate"] <= 0.10)

    def test_results_recorded_before_the_check_say_it_was_not_tried(self):
        import tempfile
        ctl = {"pass": True, "oracle_test_perm_p": 0.0005, "oracle_test_mean_gross_excess": 0.08,
               "random_false_positive_rate": 0.03, "n_random": 200}  # as H1-H8 were recorded
        te = {"mean_net_excess": -0.001, "ann_net_excess": -0.012, "info_ratio": -0.2, "months_beating_benchmark": 40,
              "n_months": 92, "perm_p": 0.7}
        rec = {"op": "result", "id": "H1-mom12_1", "spec_sha": "s", "at": "2026-01-01T00:00:00Z", "n_tickers": 60,
               "test": te, "train": {"mean_net_excess": 0.001, "perm_p": 0.2}, "controls": ctl,
               "context": {"spy_ann_return_test": 0.18, "universe_ew_ann_return_test": 0.27}}
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "ledger.jsonl"
            led.write_text(json.dumps(rec) + "\n")
            with mock.patch.object(config, "LEDGER", led), mock.patch.object(config, "RESULTS", Path(d) / "R.md"):
                text = run.report()
        self.assertNotIn("The null is calibrated.", text)
        self.assertIn("200 random signals, redrawn every month: 3.0% reached p < 0.05", text)
        self.assertIn("Random scores held fixed per stock were not tried in this run.", text)

    def test_report_lists_withdrawals_and_takes_controls_from_the_latest_run(self):
        import tempfile
        te = {"mean_net_excess": -0.001, "ann_net_excess": -0.012, "info_ratio": -0.2, "months_beating_benchmark": 40,
              "n_months": 92, "perm_p": 0.7}
        base = {"op": "result", "spec_sha": "s", "n_tickers": 60, "test": te, "train": {"mean_net_excess": 0.001, "perm_p": 0.2},
                "context": {"spy_ann_return_test": 0.18, "universe_ew_ann_return_test": 0.27}}
        old = {"oracle_test_perm_p": 0.0005, "oracle_test_mean_gross_excess": 0.08, "pass": True,
               "random_false_positive_rate": 0.03, "n_random": 200}
        recs = [{**base, "id": "H1-mom12_1", "at": "2026-01-01T00:00:00Z", "controls": old},
                {**base, "id": "H2-rev1", "at": "2026-02-01T00:00:00Z",
                 "controls": {**old, "persistent_random_false_positive_rate": 0.055}},
                {"op": "withdraw", "id": "H1-mom12_1", "reason": "a code mistake: details", "at": "2026-02-02T00:00:00Z"}]
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "ledger.jsonl"
            led.write_text("".join(json.dumps(r) + "\n" for r in recs))
            with mock.patch.object(config, "LEDGER", led), mock.patch.object(config, "RESULTS", Path(d) / "R.md"):
                text = run.report()
        self.assertIn("| 40/92 | 0.7000 | – | +0.10% |", text)  # counted in BH, its own q not shown
        self.assertIn("**WITHDRAWN** |", text)
        self.assertIn("- H1-mom12_1: a code mistake: details", text)
        self.assertIn("(latest run, 2026-02-01)", text)
        self.assertNotIn("Random scores held fixed per stock were not tried in this run.", text)

