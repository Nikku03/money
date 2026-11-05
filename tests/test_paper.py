"""The learning paper trader (brain/paper): point-in-time features, the learner, the replay, news tags."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from brain.paper import features, news, replay
from brain.paper.learner import Learner


def fake_prices(n_days=700, stocks=("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"), plant=0.0, seed=0):
    """Random walks; with ``plant``, next week's return follows last week's mom_1w (a planted edge)."""
    rng = np.random.default_rng(seed)
    cols = list(stocks) + ["SPY", "USO", "TLT", "UUP"]
    r = rng.normal(0, 0.01, size=(n_days, len(cols)))
    if plant:
        for t in range(10, n_days):
            past = r[t - 10:t - 5, :len(stocks)].sum(0)
            r[t, :len(stocks)] += plant * (past - past.mean()) / 5
    px = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), columns=cols,
                      index=pd.bdate_range("2020-01-01", periods=n_days).strftime("%Y-%m-%d"))
    return px, list(stocks)


class FeatureTest(unittest.TestCase):
    def test_point_in_time(self):
        px, stocks = fake_prices()
        i = 400
        a = features.at(px, i, stocks)
        later = px.copy()
        later.iloc[i + 1:] *= 1.5                       # change the future: today's features must not move
        b = features.at(later, i, stocks)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(list(a.columns), list(features.PRICE_FEATURES))

    def test_forward_is_excess_return(self):
        px, stocks = fake_prices()
        y = features.forward(px, 300, stocks, 5)
        want = px[stocks].iloc[305] / px[stocks].iloc[300] - px["SPY"].iloc[305] / px["SPY"].iloc[300]
        np.testing.assert_allclose(y.values, want.values)


class LearnerTest(unittest.TestCase):
    def test_recovers_a_linear_rule(self):
        rng = np.random.default_rng(0)
        L = Learner(["a", "b", "c"], forget=1.0)
        for _ in range(200):
            X = rng.normal(size=(20, 3))
            L.update(X, 0.01 * X[:, 1] + 0.001 * rng.normal(size=20))
        self.assertAlmostEqual(L.w[1], 0.01, places=3)
        self.assertLess(abs(L.w[0]) + abs(L.w[2]), 0.001)

    def test_round_trips(self):
        L = Learner(["a", "b"])
        L.update(np.eye(2), np.array([0.01, -0.01]))
        M = Learner.load(L.state())
        np.testing.assert_allclose(M.w, L.w)


class ReplayTest(unittest.TestCase):
    def test_finds_a_planted_edge_and_not_noise(self):
        px, stocks = fake_prices(plant=0.6, n_days=1400)
        noise_px, _ = fake_prices(plant=0.0, n_days=1400, seed=1)
        with mock.patch("brain.paper.data.universe", return_value=stocks):
            g = replay.grade(replay.run(px), replay.run(px, shuffle=True))
            gn = replay.grade(replay.run(noise_px), replay.run(noise_px, shuffle=True))
        self.assertEqual(g["verdict"], "PASS", g["checks"])
        self.assertEqual(gn["verdict"], "FAIL", gn["checks"])

    def test_never_trades_before_learning(self):
        px, stocks = fake_prices()
        with mock.patch("brain.paper.data.universe", return_value=stocks):
            r = replay.run(px)
        self.assertFalse(r["weeks"][0]["trade"])


class NewsTest(unittest.TestCase):
    def test_relevance_and_tags(self):
        names = news.aliases("NVDA", "NVIDIA Corporation")
        self.assertTrue(news.about("Nvidia beats estimates and raises guidance", names))
        self.assertFalse(news.about("Royalty Pharma plc Q3 2026 Earnings Call Summary", names))
        tags = set(news.tag("Nvidia beats estimates and raises its guidance"))
        self.assertEqual(tags, {"news_beat", "news_guide_up"})
        self.assertIn("news_downgrade", news.tag("Analyst downgrades Apple to underweight"))
        self.assertEqual(news.tag("Apple unveils new colors"), [])

    def test_first_live_day_mistakes(self):
        # paraphrases of 2026-11-04's mis-tags
        self.assertEqual(news.tag("Top analyst reports for Microsoft and Marvell"), [])
        self.assertEqual(news.tag("Apple may turn Siri into a paid upgrade"), [])
        self.assertEqual(news.tag("Siri could become a paid upgrade, as the CEO plans to charge for it"), [])
        self.assertIn("news_upgrade", news.tag("Ratings agency upgrades KLA to A"))
        self.assertIn("news_beat", news.tag("Qualcomm tops fourth-quarter estimates"))
        self.assertIn("news_upgrade", news.tag("Salesforce moves to strong buy after upgrade"))
        self.assertFalse(news.about("Wells Fargo starts On Holding at overweight", news.aliases("WFC", "Wells Fargo & Company")))
        self.assertFalse(news.about("Why Morgan Stanley sees 70% upside in a chip stock", news.aliases("MS", "Morgan Stanley")))
        self.assertFalse(news.about("Upgrades and downgrades today: Adobe, Apple, Exxon, Fiserv, Visa", news.aliases("V", "Visa Inc.")))
        self.assertTrue(news.about("Amgen raised guidance after an earnings beat", news.aliases("AMGN", "Amgen Inc.")))
        self.assertEqual(set(news.tag("Amgen raised guidance after an earnings beat")), {"news_guide_up", "news_beat"})


if __name__ == "__main__":
    unittest.main()
