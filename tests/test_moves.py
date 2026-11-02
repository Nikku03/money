"""Earnings move size (brain/moves.py) and its declared test (brain/research/movetest.py), on made-up prices whose
moves are known. Nothing touches the network or the committed ledger."""
import json
import os

import numpy as np
import pandas as pd

from brain import moves
from brain.research import movetest
from tests.util import TempHome, filed, price_snapshot


def synthetic(n=900, every=63, jumps=None, start="2019-01-01", ordinary=0.01):
    """Business days from ``start``; the stock alternates +/-``ordinary`` and jumps on each release's first session.
    Returns (dates, adj, firsts, known)."""
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").to_numpy()
    r = np.where(np.arange(n) % 2 == 0, ordinary, -ordinary)
    firsts = list(range(200, n - 2, every))
    jumps = jumps or [0.04, -0.06, 0.08, -0.05, 0.10, -0.03, 0.07, -0.09, 0.05, -0.04, 0.06]
    for k, i in enumerate(firsts):
        r[i] = jumps[k % len(jumps)]
    adj = 100 * np.cumprod(1 + r)
    known = [f"{dates[i - 1]}T21:00:00Z" for i in firsts]     # after the close: the next session is the first
    return dates, adj, firsts, known


class FeatureTest(TempHome):
    def test_moves_and_features_are_point_in_time(self):
        d, adj, firsts, known = synthetic()
        flat = np.full(len(d), 400.0)
        rows = moves.release_rows(d, adj, d, flat, d, np.full(len(d), 20.0), firsts, known)
        want = [abs(adj[i + 1] / adj[i - 2] - 1) for i in firsts]          # close of day -2 to close of day +1
        np.testing.assert_allclose([r["move"] for r in rows], want)         # |car_3d| against a flat SPY
        self.assertAlmostEqual(want[0], 1.04 * 0.99 * 0.99 - 1, places=9)   # the jump and two ordinary days
        self.assertNotIn("med8", rows[3])                                   # three earlier moves: too few
        r4 = rows[4]
        self.assertAlmostEqual(r4["med8"], float(np.median(want[:4])), places=12)
        self.assertAlmostEqual(r4["last"], want[3], places=12)
        self.assertAlmostEqual(r4["day_vol"], 0.01, places=3)              # release days left out
        self.assertEqual(r4["vix"], 20.0)

    def test_fit_and_chances(self):
        rng = np.random.default_rng(1)
        med = rng.uniform(0.02, 0.15, 3000)
        rows = [{"move": m * float(np.exp(rng.normal(0, 0.5))), "med8": m, "last": m, "day_vol": m / 4, "vix": 20.0}
                for m in med]
        base = moves.fit(rows, moves.BASELINE)
        self.assertAlmostEqual(base["beta"][1], 1.0, delta=0.1)            # the move scales with its usual move
        q = moves.quantiles(base, {"med8": 0.05})
        self.assertLess(q[0.1], q[0.5])
        self.assertLess(q[0.5], q[0.9])
        held = np.mean([moves.quantiles(base, r)[0.1] <= r["move"] <= moves.quantiles(base, r)[0.9] for r in rows[:500]])
        self.assertAlmostEqual(held, 0.8, delta=0.06)
        lo, hi = moves.chance_above(base, {"med8": 0.05}, 0.02), moves.chance_above(base, {"med8": 0.05}, 0.20)
        self.assertGreater(lo, hi)
        self.assertGreater(moves.pinball({0.1: 0.0, 0.25: 0.0, 0.5: 0.0, 0.75: 0.0, 0.9: 0.0}, 0.1),
                           moves.pinball({0.1: 0.08, 0.25: 0.09, 0.5: 0.1, 0.75: 0.11, 0.9: 0.12}, 0.1))


class DeclaredTestTrial(TempHome):
    def setUp(self):
        super().setUp()
        self._model = os.environ.get("BRAIN_MOVES_MODEL")
        os.environ["BRAIN_MOVES_MODEL"] = os.path.join(self._tmp.name, "moves_m1.json")

    def tearDown(self):
        if self._model is None:
            os.environ.pop("BRAIN_MOVES_MODEL", None)
        else:
            os.environ["BRAIN_MOVES_MODEL"] = self._model
        super().tearDown()

    def test_fit_declare_and_test_in_a_trial(self):
        facts = []
        tickers = ["AAA", "BBB", "CCC"]
        for k, t in enumerate(tickers):
            d, adj, _, known = synthetic(n=1900, start="2018-01-01", ordinary=0.005 * (k + 1),
                                              jumps=[x * (k + 1) for x in (0.02, -0.03, 0.04, -0.025, 0.05, -0.015)])
            facts.append(price_snapshot(t, list(d), list(adj)))
            facts += [filed(t, "filing", "8-K", ka[:10], ka, items="2.02") for ka in known]
        facts += [price_snapshot("SPY", list(d), [400.0] * len(d)), price_snapshot("^VIX", list(d), [20.0] * len(d))]
        self.bank.add_many(facts)
        model = moves.fit_and_save(self.bank, tickers=tickers, log=None)
        self.assertGreater(model["m1"]["n"], 20)
        self.assertTrue(movetest.is_trial(False))                          # a temp home: never the committed ledger
        rec = movetest.declare(log=None)
        self.assertEqual(rec["id"], "M1a")
        self.assertIsNone(movetest.declare(log=None))                      # declared once
        out = movetest.test(self.bank, log=None, tickers=tickers)
        m = out["metrics"]
        self.assertGreater(m["releases"], 5)
        self.assertEqual(m["first"][:4], "2024")
        self.assertIn("pass", out["gate"])
        led, res = movetest.paths(True)
        self.assertIn("## Move sizes", res.read_text())
        self.assertEqual([json.loads(x)["op"] for x in led.read_text().splitlines()], ["declare", "result"])
        with self.assertRaises(SystemExit):
            movetest.test(self.bank, log=None, tickers=tickers)            # one result per id


if __name__ == "__main__":
    import unittest
    unittest.main()
