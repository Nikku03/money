"""The learning paper trader (brain/paper): point-in-time features, the learner, the replay, news tags."""
import math
from pathlib import Path
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


class FilingsTest(unittest.TestCase):
    def test_counts_are_point_in_time(self):
        from brain.paper import filings
        dates = pd.bdate_range("2024-01-01", periods=20).strftime("%Y-%m-%d")
        hist = {"A": [("2024-01-08T15:00:00Z", {"sec_results"}),      # 10:00 New York, Jan 8: counts at Jan 8's close
                      ("2024-01-08T22:00:00Z", {"sec_exec"})]}        # 17:00 New York: after the close, counts Jan 9
        idx = filings.index_events(hist, dates)
        i8 = list(dates).index("2024-01-08")
        c = filings.counts(idx, ["A"], i8 - 5, i8)
        self.assertEqual(c[0, filings.FILING_FEATURES.index("sec_results")], 1)
        self.assertEqual(c[0, filings.FILING_FEATURES.index("sec_exec")], 0)
        c9 = filings.counts(idx, ["A"], i8, i8 + 1)
        self.assertEqual(c9[0, filings.FILING_FEATURES.index("sec_exec")], 1)

    def test_learner_extends_without_forgetting(self):
        L = Learner(["a", "b"])
        L.update(np.eye(2), np.array([0.01, -0.01]))
        w, P = L.w.copy(), L.P.copy()
        L.extend(["a", "b", "c"])
        np.testing.assert_allclose(L.w[:2], w)
        np.testing.assert_allclose(L.P[:2, :2], P)
        self.assertEqual(L.w[2], 0.0)


class MoonshotTest(unittest.TestCase):
    def _P(self, o, h, l, c):
        idx = pd.Index([f"d{i:02d}" for i in range(len(c))])
        mk = lambda v: pd.DataFrame({"X": v}, index=idx, dtype=float)
        return {"open": mk(o), "high": mk(h), "low": mk(l), "close": mk(c)}

    def test_exit_rules(self):
        from brain.paper import moonshot as m
        n = m.HOLD + 2
        flat = [100.0] * n
        # target hit on day 2: +20% gross
        P = self._P(flat, [100, 105, 121] + [100] * (n - 3), [99] * n, flat)
        O = m.outcomes(P, ["X"], 0)
        self.assertTrue(O["hit"].iloc[0])
        self.assertAlmostEqual(O["net"].iloc[0], m.TARGET - 2 * m.COST)
        self.assertEqual(O["exit"].iloc[0], 2)
        # both touched the same day: the stop counts
        P = self._P(flat, [100, 125] + [100] * (n - 2), [100, 85] + [99] * (n - 2), flat)
        O = m.outcomes(P, ["X"], 0)
        self.assertFalse(O["hit"].iloc[0])
        self.assertAlmostEqual(O["net"].iloc[0], -m.STOP - 2 * m.COST)
        # gap down through the stop: out at the open, worse than the stop
        o = [100, 80] + [80] * (n - 2)
        P = self._P(o, [100] + [80] * (n - 1), [99] + [79] * (n - 1), [100] + [80] * (n - 1))
        self.assertAlmostEqual(m.outcomes(P, ["X"], 0)["net"].iloc[0], -0.20 - 2 * m.COST)
        # neither: out at the close of the last day held
        c = [100.0] * (m.HOLD - 1) + [105.0] * 3
        P = self._P(flat, [106] * n, [99] * n, c)
        self.assertAlmostEqual(m.outcomes(P, ["X"], 0)["net"].iloc[0], 0.05 - 2 * m.COST)

    def test_before_the_bell(self):
        from brain.paper import moonshot as m
        dates = np.array(["2021-01-04", "2021-01-05", "2021-01-06"])
        self.assertEqual(m.visible_index(dates, "2021-01-05T12:00:00Z"), 1)   # 07:00 New York: that day's open
        self.assertEqual(m.visible_index(dates, "2021-01-05T15:00:00Z"), 2)   # 10:00 New York: the next open

    def test_learner_ranks_a_planted_signal(self):
        from brain.paper import moonshot as m
        rng = np.random.default_rng(0)
        X = rng.normal(size=(3000, len(m.FEATURES)))
        y = (rng.random(3000) < 1 / (1 + np.exp(-(-2 + 1.5 * X[:, 0])))).astype(float)
        L = m.Learner()
        L.fit(X, y, np.zeros(3000))
        self.assertGreater(L.w[1], 1.0)
        self.assertEqual(L.lessons(1)[0][0], m.PLAIN[m.FEATURES[0]])


class TradableTest(unittest.TestCase):
    def test_ghost_stocks_are_not_tradable(self):
        from brain.paper import moonshot as m
        n = 30
        idx = pd.Index([f"d{i:02d}" for i in range(n)])
        vol = pd.DataFrame({"LIVE": [200_000.0] * n, "GHOST": [0.0] * 25 + [100.0] * 5, "THIN": [1000.0] * n}, index=idx)
        raw = pd.DataFrame({"LIVE": [5.0] * n, "GHOST": [3.5] * n, "THIN": [5.0] * n}, index=idx)
        ok = m.tradable({"vol": vol, "raw": raw}, ["LIVE", "GHOST", "THIN"], 25)
        self.assertEqual(ok.tolist(), [True, False, False])


class DynamicExitTest(unittest.TestCase):
    def _arr(self, o, h, l, c):
        return [np.array(x, float).reshape(-1, 1) for x in (o, h, l, c)]

    def test_trailing_stop_follows_the_high_close(self):
        from brain.paper import moonshot2 as m
        n = 16
        c = [100, 110, 130, 150, 140, 120] + [120] * (n - 6)
        o = [100] + c[:-1]
        h = [x + 1 for x in c]
        l = [100, 109, 129, 149, 139, 119] + [119] * (n - 6)
        prm = {"target": None, "stop": np.array([0.10]), "trail": np.array([0.10]), "hold": 15}
        net, ex = m.run_exit(*self._arr(o, h, l, c), 0, prm)
        # after the 150 close the stop sits at 135: day 5 (low 119) hits it
        self.assertAlmostEqual(net[0], 1.35 - 1 - 2 * 0.0025, places=6)
        self.assertEqual(ex[0], 5)

    def test_breakeven_and_follow_through(self):
        from brain.paper import moonshot2 as m
        n = 16
        c = [112, 105, 99] + [99] * (n - 3)
        o = [100] + c[:-1]
        h = [113, 106, 100] + [100] * (n - 3)
        l = [100, 104, 98] + [98] * (n - 3)
        prm = {"target": np.array([0.30]), "stop": np.array([0.10]), "breakeven": 0.10, "hold": 15}
        net, ex = m.run_exit(*self._arr(o, h, l, c), 0, prm)
        self.assertAlmostEqual(net[0], -2 * 0.0025, places=6)          # out at the entry, not at -10%
        c2 = [101, 101.5, 102] + [130] * (n - 3)
        prm = {"target": np.array([0.20]), "stop": np.array([0.10]), "check": (2, 0.03), "hold": 10}
        net, ex = m.run_exit(*self._arr([100] + c2[:-1], [x + 0.5 for x in c2], [x - 0.5 for x in c2], c2), 0, prm)
        self.assertEqual(ex[0], 2)                                     # no follow-through by the 3rd close: out
        self.assertAlmostEqual(net[0], 0.02 - 2 * 0.0025, places=6)

    def test_stop_never_uses_the_same_days_close(self):
        from brain.paper import moonshot2 as m
        # day 0: a close of 150 must not raise the stop for day 0 itself (the low of 95 is before it)
        c = [150] + [150] * 15
        prm = {"target": None, "stop": np.array([0.10]), "trail": np.array([0.10]), "hold": 15}
        net, _ = m.run_exit(*self._arr([100] + [150] * 15, [151] * 16, [95] + [149] * 15, c), 0, prm)
        self.assertGreater(net[0], 0.3)


class MissesTest(unittest.TestCase):
    def test_anatomy_and_causes(self):
        from brain.paper import misses
        o = np.array([10.0, 10.1, 13.0, 13.1, 13.2])
        c = np.array([10.1, 10.2, 13.0, 13.2, 13.3])
        a = misses.anatomy(o, c, 0, 4)
        self.assertEqual(a["shape"], "one jump")
        self.assertEqual(a["jump_kind"], "gap")
        self.assertEqual(a["jump_at"], 2)
        self.assertGreater(a["overnight_share"], 0.8)
        self.assertEqual(misses.classify(a, {"sec_results"}, {"sec_results"}, 0, 0, 0, 1), "earnings jump")
        self.assertEqual(misses.classify(a, set(), set(), 0, 0, 0, 1), "jump with no filing (news not in its filings)")
        steady = misses.anatomy(np.linspace(10, 12, 11), np.linspace(10.1, 12.2, 11), 0, 10)
        self.assertEqual(steady["shape"], "steady climb")
        self.assertEqual(misses.classify(steady, set(), set(), 0.5, 5, 0.0, 1), "industry wave")
        self.assertEqual(misses.classify(steady, set(), set(), 0.0, 5, 0.0, 1), "steady climb")

    def test_auc(self):
        from brain.paper import misses
        self.assertAlmostEqual(misses.auc(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1, 1, 0, 0])), 1.0)
        self.assertAlmostEqual(misses.auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([1, 0, 1, 0])), 0.5)


class DossierTest(unittest.TestCase):
    def _dossier(self):
        from brain.paper import dossier as ds
        dates = pd.bdate_range("2021-01-01", periods=500).strftime("%Y-%m-%d")
        px = pd.DataFrame({"X": np.full(500, 10.0)}, index=dates)
        P = {"close": px, "raw": px}
        facts = {"X": [
            ("us-gaap:Revenues", 100e6, "2021-01-01", "2021-03-31", "2021-05-10", "10-Q", 89),
            ("us-gaap:Revenues", 150e6, "2022-01-01", "2022-03-31", "2022-05-10", "10-Q", 89),
            ("us-gaap:CashAndCashEquivalentsAtCarryingValue", 40e6, None, "2022-03-31", "2022-05-10", "10-Q", 0),
            ("us-gaap:NetCashProvidedByUsedInOperatingActivities", -20e6, "2022-01-01", "2022-03-31", "2022-05-10", "10-Q", 89),
            ("dei:EntityCommonStockSharesOutstanding", 10e6, None, "2022-05-01", "2022-05-10", "10-Q", 0),
            ("dei:EntityCommonStockSharesOutstanding", 5e6, None, "2021-05-01", "2021-05-10", "10-Q", 0),
            ("us-gaap:Revenues", 99e6, None, "2021-03-31", "2021-05-10", "10-Q", 0)]}   # ties, with no start
        return ds, ds.Dossier(facts, {"X": "Example Corp"}, P, {"X": []}, {"X": "2834"}), list(dates)

    def test_point_in_time(self):
        ds, D, dates = self._dossier()
        before = dates.index("2022-05-10")                   # filed that day: visible from the next session
        self.assertEqual(D.facts("X", before)["revenue_q"], 100e6)
        self.assertEqual(D.facts("X", before + 1)["revenue_q"], 150e6)

    def test_numbers_and_card(self):
        ds, D, dates = self._dossier()
        d = dates.index("2022-05-10") + 1
        n = ds.Dossier.numbers(D.facts("X", d))
        self.assertAlmostEqual(math.exp(n["f_rev_growth"]), 1.5, places=6)
        self.assertAlmostEqual(math.expm1(n["f_runway"]), 40e6 / (20e6 / 89 * 91), places=6)
        self.assertAlmostEqual(math.exp(n["f_dilution"]), 2.0, places=6)
        card = D.card("X", d)
        self.assertIn("Example Corp", card)
        self.assertIn("+50% on a year ago", card)
        self.assertIn("shares +100% on a year ago", card)

    def test_nonsense_numbers_count_as_missing(self):
        from brain.paper import dossier as ds
        f = {"revenue_q": 1e6, "revenue_q_year_ago": None, "gross_profit_q": None, "operating_income_q": None,
             "net_income_q": None, "rnd_q": None, "cash": -5e6, "operating_cash_flow_q": -1e6, "long_term_debt": None,
             "assets": 1e7, "current_assets": -1.0, "current_liabilities": 2e6, "shares": 1e6, "shares_year_ago": -3.0,
             "mcap": 1e7, "sessions_since_report": 10, "results_moves": []}
        n = ds.Dossier.numbers(f)
        self.assertIsNone(n["f_runway"])
        self.assertIsNone(n["f_current"])
        self.assertIsNone(n["f_dilution"])

    def test_card_with_cash_but_no_cash_flow(self):
        ds, D, dates = self._dossier()
        D.by["X"].pop("us-gaap:NetCashProvidedByUsedInOperatingActivities")
        self.assertIn("cash $40.0M", D.card("X", dates.index("2022-05-10") + 1))


class ForwardTest(unittest.TestCase):
    def test_follow_waits_for_the_exit(self):
        from brain.paper import forward
        idx = pd.Index([f"2026-11-{9 + i:02d}" for i in range(5)])
        mk = lambda v: pd.DataFrame({"X": v}, index=idx, dtype=float)
        flat = [10.0] * 5
        P = {"open": mk(flat), "high": mk([10.5] * 5), "low": mk([9.8] * 5), "close": mk(flat)}
        self.assertEqual(forward.follow(P, "X", 0, "E0 fixed", 0.03), (None, None))   # still open
        P["low"] = mk([9.8, 9.8, 8.5, 9.0, 9.0])                                       # stopped on day 3
        net, when = forward.follow(P, "X", 0, "E0 fixed", 0.03)
        self.assertAlmostEqual(net, -0.10 - 2 * 0.0025)
        self.assertEqual(when, "2026-11-11")


class NoPeekTest(unittest.TestCase):
    def test_exit_chooser_ignores_unfinished_trades(self):
        from brain.paper import moonshot2 as m
        ch = m.ExitChooser()
        good = np.array([[0.0, 0.0, 0.5, 0.0, 0.0]])                 # E2 would have been great...
        for w in range(40):
            ch.add(w, [0], np.array([[0.1, 0.0, 0.0, 0.0, 0.0]]), [w * 5 + 14])
        ch.add(40, [0] * 1000, np.repeat(good, 1000, 0), [999] * 1000)  # ...but those trades finish at session 999
        self.assertEqual(ch.choose(41, 300)[0], 0)                    # at session 300 it cannot know that
        self.assertEqual(ch.choose(41, 1000)[0], 2)


class P7Test(unittest.TestCase):
    def test_gate_needs_a_finished_record(self):
        from brain.paper import moonshot3 as m
        good = [(i, 0.02 + 0.001 * (i % 3)) for i in range(40)]
        self.assertTrue(m.gate(good, 100)[0])
        self.assertFalse(m.gate(good[:20], 100)[0])                    # too short a record
        self.assertFalse(m.gate(good, 10)[0])                           # those weeks had not finished by session 10
        self.assertFalse(m.gate([(i, -0.01) for i in range(40)], 100)[0])

    def test_ridge_strength_follows_the_signal(self):
        from brain.paper import moonshot3 as m
        rng = np.random.default_rng(0)
        for signal, want in ((0.05, 3.0), (0.0, 300.0)):
            st = m.Stats(4, seed=0)
            st.set_scaler(rng.normal(size=(500, 4)))
            for w in range(120):
                X = rng.normal(size=(50, 4))
                Y = np.column_stack([signal * X[:, 0] + 0.02 * rng.normal(size=50)] * 5)
                st.add_week(w, X, Y)
            st.fit(120)
            self.assertEqual(st.lam, want)


class MemoryTest(unittest.TestCase):
    def test_recall_respects_time(self):
        import tempfile
        from brain.paper import memory as mem
        with tempfile.TemporaryDirectory() as tmp:
            b = mem.bank(Path(tmp))
            tr = {"date": "2021-03-01", "closed": "2021-03-12", "ticker": "XYZ", "exit": "E0 fixed", "net": -0.105,
                  "shape": "never worked (it never got 3% up)", "lesson": "XYZ -10.5%: it never got going"}
            mem.remember(b, [mem.trade_fact(tr, "test")])
            self.assertEqual(mem.recall(b, "XYZ", "2021-03-11")["trades"], 0)    # still open on the 11th
            r = mem.recall(b, "XYZ", "2021-03-13")
            self.assertEqual(r["trades"], 1)
            self.assertIn("traded XYZ 1 time(s) before, mean -10.5%", mem.card(r))
            self.assertIn("it never got going", mem.card(r))

    def test_analogs_recall_similar_situations(self):
        from brain.paper import memory as mem
        rng = np.random.default_rng(0)
        A = mem.Analogs(k=20)
        for w in range(30):
            X = rng.normal(size=(100, 3))
            A.add(w, w * 5 + 10, X, np.where(X[:, 0] > 1, 0.2, -0.05), (X[:, 0] > 1).astype(float))
        net, hit = A.recall(30, 1000, np.array([[2.0, 0, 0], [-2.0, 0, 0]]))
        self.assertGreater(net[0], 0.1)
        self.assertLess(net[1], 0.0)
        self.assertEqual(A.recall(30, 5, np.zeros((1, 3)))[0][0], 0.0)              # nothing resolved by session 5
