"""The scan as a trader (brain/scantrader.py): midday prices, stops, market checks, sizing, weekly vs daily."""
from __future__ import annotations

import json
import unittest
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

import numpy as np

from brain import scantrader as st

NY = ZoneInfo("America/New_York")
WIDE = np.log([0.9, 0.95, 1.0, 1.05, 1.1])           # a stock's five quantiles: its floor 10% under the close


def chart(bars: list[tuple[str, float]]) -> bytes:
    ts = [int(datetime.fromisoformat(t).replace(tzinfo=NY).timestamp()) for t, _ in bars]
    return json.dumps({"chart": {"result": [{"timestamp": ts,
                                             "indicators": {"quote": [{"close": [c for _, c in bars]}]}}]}}).encode()


def market(n: int = 12, m: int = 3, price: float = 10.0) -> st.Data:
    """``m`` stocks and SPY over ``n`` sessions, flat prices, every stock tradable with WIDE ranges."""
    dates = np.array([f"2025-03-{k + 1:02d}" for k in range(n)])
    full = lambda v: np.full((n, m + 1), v, float)
    D = st.Data(dates=dates, names=[f"S{k}" for k in range(m)] + ["SPY"], open=full(price), low=full(price),
                mid=full(price), close=full(price), has=np.ones((n, m + 1), bool), dv=full(1e12),
                q=np.tile(WIDE.astype(np.float32), (n, m + 1, 1)), l5=full(0.0), vix_open=np.full(n, 20.0),
                vix_mid=np.full(n, 20.0), vix_close=np.full(n, 20.0), rf=np.zeros(n),
                week_end=np.array([k % 5 == 4 for k in range(n)]))
    D.q[:, m, :] = np.nan                               # SPY is not one of the scan's stocks
    for a in (D.open, D.low, D.mid, D.close):
        a[:, m] = 100.0
    return D


def free(dv, spy):
    return np.zeros(len(dv))


class MiddayTest(unittest.TestCase):
    def test_the_last_bar_that_ended_by_half_past_twelve(self):
        stock = chart([("2025-01-15T09:30", 10.0), ("2025-01-15T10:30", 11.0), ("2025-01-15T11:30", 12.0),
                       ("2025-01-15T12:30", 13.0), ("2025-07-15T11:30", 20.0), ("2025-07-15T12:30", 21.0)])
        self.assertEqual(st.parse_midday(stock), {"2025-01-15": 12.0, "2025-07-15": 20.0})   # winter and summer
        vix = chart([("2025-01-15T03:00", 18.0), ("2025-01-15T11:00", 19.0), ("2025-01-15T12:00", 25.0),
                     ("2025-01-15T20:00", 30.0)])
        self.assertEqual(st.parse_midday(vix), {"2025-01-15": 19.0})       # on the hour: the 11:00 bar


class StopsTest(unittest.TestCase):
    def test_gaps_morning_and_afternoon_breaks_and_the_wait_before_buying_again(self):
        D = market()
        D.open[3, 0] = D.low[3, 0] = D.mid[3, 0] = 8.5                     # S0 opens through its 9.0 floor
        D.close[3, 0] = 8.6
        D.low[4, 1], D.mid[4, 1] = 8.8, 9.5                                 # S1 breaks it in the afternoon
        D.low[5, 2], D.mid[5, 2] = 8.7, 8.9                                 # S2 is through it by midday
        p = st.Policy("T", "movers", "test", daily=True, stops=True)
        R = st.readings(D)
        with mock.patch.object(st, "cost_side", free):
            r = st.simulate(D, R, p, 1, 11, log_trades=True)
        sells = [(t["date"], t["stock"], t["when"]) for t in r["trades"] if t["action"] == "sell"]
        self.assertEqual(sells, [("2025-03-04", "S0", "open"), ("2025-03-05", "S1", "afternoon"),
                                 ("2025-03-06", "S2", "morning")])
        self.assertEqual(r["counts"]["stops"], 3)
        buys = [(t["date"], t["stock"]) for t in r["trades"] if t["action"] == "buy"]
        self.assertIn(("2025-03-10", "S0"), buys)                           # back after five sessions ...
        self.assertNotIn(("2025-03-09", "S0"), buys)                        # ... not before
        # nav[k] is session 1 + k. S0 lost 15% on its third; the rest went into S1 and S2, which then filled at
        # the floor less the slip (each time the book's money moved into the stocks still allowed)
        fill = 9.0 * (1 - st.STOP_SLIP) / 10
        self.assertAlmostEqual(r["nav"][2], 1 - 0.15 / 3, places=9)
        self.assertAlmostEqual(r["nav"][3], 0.95 / 2 * fill + 0.95 / 2, places=9)
        self.assertAlmostEqual(r["nav"][4], (0.95 / 2 * fill + 0.95 / 2) * fill, places=9)

    def test_a_stock_already_through_its_floor_at_the_open_is_not_bought(self):
        D = market()
        D.open[1, 0] = 8.0                                                  # through 9.0 at the first open
        p = st.Policy("T", "movers", "test", daily=True, stops=True)
        with mock.patch.object(st, "cost_side", free):
            r = st.simulate(D, st.readings(D), p, 1, 3, log_trades=True)
        self.assertNotIn(("2025-03-02", "S0"), [(t["date"], t["stock"]) for t in r["trades"] if t["action"] == "buy"])


class MarketCheckTest(unittest.TestCase):
    def test_a_fall_at_the_open_or_a_fear_jump_at_midday_cuts_the_book_to_half(self):
        D = market()
        D.open[3, -1] = 97.5                                                # SPY -2.5% at the open
        D.vix_mid[6] = 24.0                                                 # the VIX +20% at midday
        p = st.Policy("T", "spy", "test", daily=True, market=True)
        with mock.patch.object(st, "cost_side", free):
            r = st.simulate(D, st.readings(D), p, 1, 11, log_trades=True)
        g = r["gross"]
        up = 100 / 97.5                                                     # the cut half missed the rebound
        self.assertAlmostEqual(g[2], 0.5 * up / (0.5 + 0.5 * up), places=6)  # session 3, at its close
        self.assertAlmostEqual(g[3], 1.0, places=6)                         # back at the next open
        self.assertAlmostEqual(g[5], 0.5, places=6)                         # session 6: cut at midday
        self.assertEqual((r["counts"]["cuts_open"], r["counts"]["cuts_midday"]), (1, 1))
        cuts = [t for t in r["trades"] if t["action"] == "cut"]
        self.assertEqual([(t["date"], t["when"]) for t in cuts], [("2025-03-04", "open"), ("2025-03-07", "midday")])
        self.assertIn("SPY -2.5%", cuts[0]["why"])
        # the open fell 2.5%, half was sold there, and the session closed back at 100
        self.assertAlmostEqual(r["nav"][2], 0.975 * 0.5 + 0.975 * 0.5 * up, places=9)

    def test_cash_earns_the_bill_rate(self):
        D = market()
        D.rf[:] = 0.0001
        D.open[1, -1] = 97.0                                                # cut on the first day
        p = st.Policy("T", "spy", "test", daily=True, market=True)
        with mock.patch.object(st, "cost_side", free):
            r = st.simulate(D, st.readings(D), p, 1, 1)
        self.assertAlmostEqual(r["nav"][0], 1.0001 * (0.5 + 0.5 * 100 / 97.0), places=8)


class SizingTest(unittest.TestCase):
    def test_weights_follow_one_over_the_range_and_the_book_shrinks_when_ranges_widen(self):
        D = market(m=2)
        D.q[:, 1, :] = (2 * WIDE).astype(np.float32)                        # S1's range twice S0's
        R = st.readings(D)
        R["usual"]["movers"][:] = 0.05
        p = st.Policy("T", "movers", "test", daily=True, risk=True)
        w, gross = st.target(D, R, p, 3, 4, np.zeros(3, bool), np.full(3, -1))
        self.assertAlmostEqual(w[0] / w[1], 2.0, places=5)
        hm = 2 / (1 / R["sig"][3, 0] + 1 / R["sig"][3, 1])
        self.assertAlmostEqual(gross, min(1.0, 0.05 / hm), places=6)
        R["usual"]["movers"][:] = 1.0
        self.assertEqual(st.target(D, R, p, 3, 4, np.zeros(3, bool), np.full(3, -1))[1], 1.0)   # never borrows

    def test_costs_by_dollar_volume(self):
        c = st.cost_side(np.array([2e8, 5e7, 1e7, 1e6, np.nan, 1e12]), 5)
        self.assertEqual(c.tolist(), [0.0005, 0.001, 0.002, 0.004, 0.004, st.SPY_COST])


class WeeklyTest(unittest.TestCase):
    def test_the_fixed_book_changes_only_after_a_weeks_last_close(self):
        D = market(m=2)
        D.q[:, 0, :] = (2 * WIDE).astype(np.float32)                        # S0 the widest at first ...
        D.q[2:, 1, :] = (3 * WIDE).astype(np.float32)                       # ... S1 from the close of session 2
        with mock.patch.object(st, "cost_side", free), mock.patch.object(st, "TOP", 1), \
                mock.patch.object(st, "KEEP", 1):
            weekly = st.simulate(D, st.readings(D), st.Policy("W", "movers", "w"), 1, 11, log_trades=True)
            daily = st.simulate(D, st.readings(D), st.Policy("D", "movers", "d", daily=True), 1, 11, log_trades=True)
        first_buy = lambda r: next(t["date"] for t in r["trades"] if t["action"] == "buy" and t["stock"] == "S1")
        self.assertEqual(first_buy(daily), "2025-03-04")                    # the open after session 2's close
        self.assertEqual(first_buy(weekly), "2025-03-06")                   # the open after the week's last close


class AccuracyTest(unittest.TestCase):
    def test_fresh_ranges_against_the_weeks_fixed_reading(self):
        D = market(n=15, m=4)
        D.l5[:] = 0.2
        D.q[:, :4, :] = np.log([0.7, 0.78, 1.0, 1.3, 1.4]).astype(np.float32)       # fresh: wide enough
        D.q[D.week_end, :4, :] = WIDE.astype(np.float32)                            # the fixed reading: too narrow
        a = st.accuracy(D, 5)
        self.assertEqual((a["midweek"]["fresh68"], a["midweek"]["fixed68"]), (1.0, 0.0))
        self.assertEqual(a["all"]["sessions"], 10)


class OneTradeTest(unittest.TestCase):
    """One order a day at most (simulate_one)."""

    def movers(self, n: int = 12) -> st.Data:
        D = market(n=n, m=3)
        for k, f in enumerate((1.2, 1.1, 1.0)):                             # S0 the widest, then S1, then S2
            D.q[:, k, :] = (f * WIDE).astype(np.float32)
        return D

    def run_one(self, D, p, top=3, keep=3, R=None):
        with mock.patch.object(st, "cost_side", free), mock.patch.object(st, "TOP", top), \
                mock.patch.object(st, "KEEP", keep):
            return st.simulate_one(D, R or st.readings(D), p, 1, len(D.dates) - 1, log_trades=True)

    M5 = st.Policy("T", "movers", "one", daily=True, stops=True, market=True, one=True)

    def test_the_book_fills_one_stock_a_day(self):
        r = self.run_one(self.movers(), self.M5)
        buys = [(t["date"], t["stock"]) for t in r["trades"] if t["action"] == "buy"]
        self.assertEqual(buys, [("2025-03-02", "S0"), ("2025-03-03", "S1"), ("2025-03-04", "S2")])
        self.assertEqual(r["counts"]["most_orders_a_day"], 1)
        self.assertAlmostEqual(r["gross"][2], 1.0, places=9)                # a third of the account each

    def test_a_stop_comes_first_and_a_market_check_holds_off_buying(self):
        D = self.movers()
        D.open[2, 0] = 8.0                                                  # S0 opens through its 8.81 floor
        D.open[4, -1] = 97.0                                                # SPY -3% at the open of session 4
        r = self.run_one(D, self.M5)
        day = lambda d: [(t["stock"], t["action"], t["when"]) for t in r["trades"] if t["date"] == d]
        self.assertEqual(day("2025-03-03"), [("S0", "sell", "open")])      # not the planned buy of S1
        self.assertEqual(day("2025-03-04"), [("S1", "buy", "open")])
        self.assertEqual(day("2025-03-05"), [])                             # the check: no buying today
        self.assertEqual(day("2025-03-06"), [("S2", "buy", "open")])
        self.assertEqual(r["counts"]["most_orders_a_day"], 1)

    def test_a_standing_stop_fills_only_on_a_day_with_no_order_yet(self):
        D = self.movers()
        D.low[2, 0] = 8.5                                                   # S0 dips through its floor ...
        D.low[4, 0], D.mid[4, 0] = 8.5, 9.5                                 # ... on a buying day, then a quiet one
        r = self.run_one(D, self.M5)
        sells = [(t["date"], t["stock"], t["when"]) for t in r["trades"] if t["action"] == "sell"]
        self.assertEqual(sells, [("2025-03-05", "S0", "afternoon")])        # the day S1 was bought it held on
        self.assertEqual(r["counts"]["most_orders_a_day"], 1)

    def test_spy_trades_only_when_far_from_its_target_or_on_a_check(self):
        D = market()
        D.open[4, -1] = 97.0                                                # a fall at the open of session 4
        D.vix_mid[7] = 24.0                                                 # fear at midday in session 7
        p = st.Policy("T", "spy", "one", daily=True, risk=True, market=True, one=True)
        R = st.readings(D)
        R["usual"]["spy"][:] = 1.0
        R["risk"]["spy"][:] = 1.0                                           # the target: fully in ...
        R["risk"]["spy"][9] = 1.0 / 0.95                                    # ... 95% after session 9 (within the band)
        r = self.run_one(D, p, R=R)
        got = [(t["date"], t["action"], t["when"]) for t in r["trades"]]
        self.assertEqual(got, [("2025-03-02", "buy", "open"), ("2025-03-05", "sell", "open"),
                               ("2025-03-06", "buy", "open"), ("2025-03-08", "sell", "midday"),
                               ("2025-03-09", "buy", "open")])
        self.assertEqual(r["counts"]["most_orders_a_day"], 1)


if __name__ == "__main__":
    unittest.main()
