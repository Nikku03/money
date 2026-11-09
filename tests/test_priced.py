import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from brain import priced as pr


def chain(spot=100.0, front="2026-11-20", nxt="2026-11-27", iv_front=0.3, iv_next=0.3, half=0.1):
    leg = lambda mid, iv: {"bid": mid - half, "ask": mid + half, "iv": iv, "open_interest": 10.0, "volume": 1.0}
    return {"spot": spot, "chain": {front: {100.0: {"C": leg(2.0, iv_front), "P": leg(2.0, iv_front)},
                                            105.0: {"C": leg(0.5, iv_front), "P": leg(5.4, iv_front)}},
                                    nxt: {100.0: {"C": leg(3.0, iv_next), "P": leg(3.0, iv_next)}}}}


class PricedTest(unittest.TestCase):
    def test_forecast_inputs_need_66_sessions(self):
        self.assertIsNone(pr.har_x(np.full(50, 100.0), 20))
        x = pr.har_x(100 * np.exp(np.cumsum(np.full(80, 0.01))), 20)
        self.assertAlmostEqual(x[1], math.log(1e-4 + 1e-10), places=6)          # 1% a day, every day

    def test_fit_learns_that_volatility_persists(self):
        rng = np.random.default_rng(0)
        rows = []
        for s in rng.uniform(0.005, 0.04, 200):
            c = 100 * np.exp(np.cumsum(rng.normal(0, s, 400)))
            rows += pr.samples(c, np.full(400, 20.0))
        m = pr.fit(rows)
        calm, wild = [pr.har_x(100 * np.exp(np.cumsum(rng.normal(0, s, 80))), 20) for s in (0.005, 0.04)]
        self.assertGreater(m.sigma(wild), 4 * m.sigma(calm))
        self.assertLess(m.chance_above(calm, 0.10), m.chance_above(wild, 0.10))

    def test_straddle_quote_at_the_money_with_its_spread(self):
        q = pr.straddle_quote(chain(), "2026-11-20", 101.0)
        self.assertEqual((q["strike"], q["mid"], q["half_spread"]), (100.0, 4.0, 0.2))

    def test_sides_rank_within_the_week_and_skip_events(self):
        calls = [{"ratio": r, "event": False} for r in np.linspace(0.5, 1.5, 10)] + [{"ratio": 9.0, "event": True}]
        pr.sides(calls)
        self.assertEqual([c["side"] for c in calls[:2]], ["sell", "sell"])
        self.assertEqual([c["side"] for c in calls[8:10]], ["buy", "buy"])
        self.assertEqual(calls[10]["side"], "none")                              # an event week is O1's

    def test_outcome_pays_intrinsic_value_less_the_spread(self):
        call = {"ticker": "X", "start_date": "2026-11-13", "by": "2026-11-20", "start_price": 100.0, "priced_pct": 4.0,
                "straddle": {"strike": 100.0, "mid": 4.0, "half_spread": 0.2}}
        o = pr.outcome(call, 110.0)
        self.assertAlmostEqual(o["buyer"], 0.10 - 0.042)
        self.assertAlmostEqual(o["seller"], 0.038 - 0.10)
        self.assertTrue(o["beat"])

    def test_record_picks_next_fridays_expiry_and_flags_events(self):
        m = pr.Har([1.96, 0.165, 0.055, 0.532, 0.297], np.linspace(0, 3, 100))
        closes = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 80)))
        calls = pr.record_v1("2026-11-13", lambda t: (closes, 101.0), lambda t: chain(iv_front=0.6 if t == "E" else 0.3),
                             20.0, ["A", "E"], m)
        self.assertEqual([c["by"] for c in calls], ["2026-11-20", "2026-11-20"])
        self.assertEqual([c["event"] for c in calls], [False, True])
        self.assertAlmostEqual(calls[0]["priced_pct"], 4.0 / 101.0 * 100, places=2)

    def test_judges_only_expired_calls_and_scores_the_weeks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "v1.jsonl"
            calls = []
            for w, wk in enumerate(("2026-11-13", "2026-11-20", "2026-11-27")):
                for k in range(10):
                    calls.append({"type": "call", "test": "V1", "ticker": f"T{k}", "start_date": wk, "by": f"exp{w}",
                                  "start_price": 100.0, "priced_pct": 4.0, "ratio": 0.5 + 0.1 * k, "event": False,
                                  "side": "buy" if k >= 8 else "sell" if k < 2 else "none", "made_at": wk,
                                  "straddle": {"strike": 100.0, "mid": 4.0, "half_spread": 0.1}})
            pr.append(p, calls)
            n = pr.judge(p, lambda t, d: 100.0 + 2.0 * int(t[1:]), "exp9")   # the higher the ratio, the bigger the move
            self.assertEqual(n, 30)
            self.assertEqual(pr.judge(p, lambda t, d: 100.0, "exp9"), 0)        # judged once
            e = pr.evaluate(p, "V1")
            self.assertEqual(e["weeks"], 3)
            self.assertAlmostEqual(e["rank_corr"]["mean"], 1.0)
            self.assertGreater(e["follow_sides"]["mean"], 0)
            self.assertIn("too early", e["verdict"])
            with tempfile.TemporaryDirectory() as tmp2:
                q = Path(tmp2) / "v.jsonl"
                pr.append(q, calls[:1])
                self.assertEqual(pr.judge(q, lambda t, d: 101.0, "exp0"), 0)    # not past its expiry yet


if __name__ == "__main__":
    unittest.main()
