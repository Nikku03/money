"""The shock radar (brain/shocks.py) on made-up prices whose shocks are known. Nothing touches the network."""
import json
import os

import numpy as np
import pandas as pd

from brain import shocks
from brain.research import shocktest
from tests.util import TempHome, filed, price_snapshot


def synthetic(n=700, start="2015-01-01", shock_at=(400,), size=0.30, ordinary=0.01):
    """Business days from ``start``; the stock alternates +/-``ordinary`` against a flat SPY and jumps by ``size``
    on each session in ``shock_at``."""
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").to_numpy()
    r = np.where(np.arange(n) % 2 == 0, ordinary, -ordinary)
    for i in shock_at:
        r[i] = size
    adj = 20 * np.cumprod(1 + r)
    return dates, adj


def build(dates, adj, reports=(), news=(), offer=(), bio=1.0, vol=1e6):
    n = len(dates)
    return shocks.frame(dates, adj.copy(), adj, np.full(n, vol), np.full(n, 400.0), np.full(n, 18.0),
                        list(reports), list(news), list(offer), bio)


class FrameTest(TempHome):
    def test_outcome_looks_ahead_five_sessions_and_features_do_not(self):
        d, adj = synthetic(shock_at=(400,))
        f = build(d, adj)
        # the five sessions before the shock see it coming in their outcome; the shock day itself does not
        self.assertTrue(all(f["y"][i] == 1 for i in range(395, 400)))
        self.assertEqual(f["y"][394], 0)
        self.assertEqual(f["y"][400], 0)
        self.assertEqual(f["up"][399], 1)
        self.assertEqual(f["down"][399], 0)
        # features at 399 know nothing of it; at 400 they do
        self.assertLess(f["a1"][399], 0.02)
        self.assertAlmostEqual(f["a1"][400], 0.30, places=9)
        self.assertEqual(f["n20"][399], 0)
        self.assertEqual(f["n20"][400], 1)
        self.assertEqual(f["big1"][400], 1)
        self.assertAlmostEqual(np.exp(f["lmax250"][400]), 0.30 + 1e-4, places=6)
        self.assertTrue(np.isnan(f["y"][-1]))                     # the last window has not closed

    def test_a_data_error_is_not_a_shock_and_voids_its_window(self):
        d, adj = synthetic(shock_at=(400,), size=9.0)            # +900% in a day: a bad bar
        f = build(d, adj)
        self.assertTrue(np.isnan(f["a1"][400]))
        self.assertTrue(all(np.isnan(f["y"][i]) for i in range(395, 400)))
        self.assertEqual(f["y"][394], 0)

    def test_filings_and_reports_count_from_when_they_were_known(self):
        d, adj = synthetic()
        f = build(d, adj, news=[300, 302], offer=[310])
        self.assertEqual(f["news5"][299], 0)
        self.assertEqual(f["news5"][302], 2)
        self.assertEqual(f["news5"][305], 1)
        self.assertEqual(f["offer30"][309], 0)
        self.assertEqual(f["offer30"][310], 1)
        self.assertEqual(f["offer30"][339], 1)
        self.assertEqual(f["offer30"][340], 0)
        self.assertEqual(f["bio"][0], 1.0)
        self.assertEqual(f["bio_known"][0], 1.0)
        g = build(d, adj, bio=None)
        self.assertEqual((g["bio"][0], g["bio_known"][0]), (0.0, 0.0))

    def test_report_window_uses_last_years_date(self):
        dates = pd.bdate_range("2020-01-01", "2021-12-31").strftime("%Y-%m-%d").to_numpy()
        win, known = shocks.earn_window(dates, ["2020-05-07"])
        d = list(dates)
        # 2020-05-07 + 364 days = 2021-05-06; a session sees it when it lands 1-8 days ahead
        self.assertEqual(win[d.index("2021-04-27")], 0)
        self.assertEqual(win[d.index("2021-04-28")], 1)            # 8 days ahead
        self.assertEqual(win[d.index("2021-05-05")], 1)            # 1 day ahead
        self.assertEqual(win[d.index("2021-05-06")], 0)            # the day itself is not ahead
        self.assertEqual(known[d.index("2021-04-27")], 0)          # no estimate within reach yet
        self.assertEqual(known[d.index("2021-04-28")], 1)

    def test_week_ends_are_each_completed_weeks_last_session(self):
        dates = pd.bdate_range("2024-01-01", "2024-01-24").strftime("%Y-%m-%d").to_numpy()
        ends = shocks.week_ends(dates)
        self.assertEqual([dates[i] for i in ends], ["2024-01-05", "2024-01-12", "2024-01-19"])

    def test_rows_skip_unknown_features_and_open_outcomes(self):
        d, adj = synthetic()
        f = build(d, adj, reports=["2015-03-02"])
        rows = shocks.rows_at(f, d, [10, 450, len(d) - 2], "XYZ")
        self.assertEqual([r["date"] for r in rows], [d[450]])      # too little history; window still open
        live = shocks.rows_at(f, d, [len(d) - 2], "XYZ", outcome=False)
        self.assertEqual(len(live), 1)
        self.assertNotIn("y", live[0])


class ModelTest(TempHome):
    def test_fit_finds_the_driver_and_is_calibrated(self):
        rng = np.random.default_rng(3)
        n = 40000
        rows = []
        for _ in range(n):
            x = {k: float(rng.normal()) for k in shocks.FEATS}
            p = 1 / (1 + np.exp(-(-4.0 + 1.2 * x["lv60"])))
            rows.append({**x, "y": float(rng.random() < p)})
        m = shocks.fit(rows)
        w = dict(zip(m["feats"], m["w"][1:]))
        self.assertGreater(w["lv60"], 0.9)
        self.assertLess(max(abs(v) for k, v in w.items() if k != "lv60"), 0.15)
        p = shocks.chances(m, rows)
        self.assertAlmostEqual(float(p.mean()), float(np.mean([r["y"] for r in rows])), delta=0.002)
        hi = p > np.quantile(p, 0.9)
        self.assertAlmostEqual(float(np.mean([r["y"] for r, h in zip(rows, hi) if h])), float(p[hi].mean()), delta=0.02)
        b = shocks.fit(rows, shocks.BASELINE)
        self.assertEqual(b["feats"], list(shocks.BASELINE))


class DeclaredTestTrial(TempHome):
    def setUp(self):
        super().setUp()
        self._model = os.environ.get("BRAIN_SHOCKS_MODEL")
        os.environ["BRAIN_SHOCKS_MODEL"] = os.path.join(self._tmp.name, "shocks_s1.json")

    def tearDown(self):
        if self._model is None:
            os.environ.pop("BRAIN_SHOCKS_MODEL", None)
        else:
            os.environ["BRAIN_SHOCKS_MODEL"] = self._model
        super().tearDown()

    def test_fit_declare_and_test_in_a_trial(self):
        rng = np.random.default_rng(5)
        d = pd.bdate_range("2010-01-04", "2026-10-30").strftime("%Y-%m-%d").to_numpy()
        facts, tickers = [], ["AAA", "BBB", "CCC", "DDD"]
        for k, t in enumerate(tickers):
            sd, q = 0.005 * (k + 1), 0.002 * (k + 1) ** 2          # jumpier stocks have more shocks
            r = rng.normal(0, sd, len(d))
            hit = rng.random(len(d)) < q
            r[hit] = rng.choice([-0.25, 0.3], hit.sum())
            adj = list(10 * np.cumprod(1 + r))
            facts.append(price_snapshot(t, list(d), adj))
            facts.append(filed(t, "filing", "8-K", "2015-03-02", "2015-03-02T21:00:00Z", items="2.02"))
        facts += [price_snapshot("SPY", list(d), [400.0] * len(d)), price_snapshot("^VIX", list(d), [20.0] * len(d))]
        self.bank.add_many(facts)
        model = shocks.fit_and_save(self.bank, tickers=tickers, log=None)
        self.assertGreater(model["s1"]["n"], 2000)
        self.assertGreater(model["s1"]["rate"], 0.01)
        top = shocks.radar(self.bank, tickers)
        self.assertEqual(len(top), 4)
        self.assertGreater(top[0]["chance"], top[-1]["chance"])
        self.assertTrue(shocktest.is_trial(False))                         # a temp home: never the committed ledger
        rec = shocktest.declare(log=None)
        self.assertEqual(rec["id"], "S1a")
        self.assertIsNone(shocktest.declare(log=None))                     # declared once
        out = shocktest.test(self.bank, log=None, tickers=tickers)
        m = out["metrics"]
        self.assertGreater(m["stock_weeks"], 400)
        self.assertEqual(m["first"][:4], "2024")
        self.assertIn("pass", out["gate"])
        self.assertGreater(m["auc_s1"], 0.6)
        led, res = shocktest.paths(True)
        self.assertIn("## Shocks", res.read_text())
        self.assertEqual([json.loads(x)["op"] for x in led.read_text().splitlines()], ["declare", "result"])
        with self.assertRaises(SystemExit):
            shocktest.test(self.bank, log=None, tickers=tickers)           # one result per id


class ScoreTest(TempHome):
    def test_capture_and_ranks(self):
        week = np.array(["w1"] * 4 + ["w2"] * 4)
        p = np.array([0.4, 0.1, 0.2, 0.3, 0.05, 0.5, 0.2, 0.1])
        rk = shocktest.week_rank(week, p)
        np.testing.assert_allclose(rk, [0.25, 1.0, 0.75, 0.5, 1.0, 0.25, 0.5, 0.75])
        sc = {"rank": rk, "y": np.array([1, 0, 0, 0, 0, 1, 0, 1.0])}
        cap, rate = shocktest.capture(sc, 0.25)
        self.assertAlmostEqual(cap, 2 / 3)
        self.assertAlmostEqual(rate, 1.0)
        self.assertAlmostEqual(shocktest.auc(np.array([0.1, 0.9, 0.5]), np.array([0, 1, 0.0])), 1.0)
