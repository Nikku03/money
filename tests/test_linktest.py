import math
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from brain import chain, series
from brain.bank import Fact
from brain.chain import linktest
from brain.research import run
from brain.research.forecast_tests import Refused
from brain.sources import treasury
from tests.util import TempHome

# Two sessions a month, shared by every series: the 2nd (the month's first) and the 28th (its last).
DATES = [f"{y}-{m:02d}-{d:02d}" for y in range(2010, 2019) for m in range(1, 13) for d in (2, 28)]


def snap(entity, close):
    close = np.asarray(close, dtype=float)
    df = pd.DataFrame({"date": DATES, "open": close, "high": close, "low": close, "close": close, "adjclose": close,
                       "volume": np.full(len(DATES), 1000)})
    return Fact(entity=entity, attr="series.price_daily", value=series.store(df), as_of=DATES[-1], known_at=DATES[-1],
                tier="market", source={"name": "test", "url": "https://example.test/px"},
                context={"rows": len(DATES), "first": DATES[0], "last": DATES[-1], "splits": []})


def world(seed=3):
    """Oil moves airlines against it in the same month, and oil producers with it over the next month from that
    month's first session; the 10-year yield moves nothing."""
    rng = np.random.default_rng(seed)
    n = len(DATES)
    spy = rng.normal(0.004, 0.028, n)
    oil = rng.normal(0.0, 0.056, n)
    oil[0] = 0.0
    level = 60 * np.cumprod(1 + oil)
    ends = list(range(1, n, 2))                         # each month's 28th
    lead = np.zeros(n)
    for k in range(1, len(ends) - 1):                   # month k's oil change, earned 2nd -> 28th of month k + 1
        lead[ends[k + 1]] = 0.3 * (level[ends[k]] / level[ends[k - 1]] - 1)
    px = {"SPY": 100 * np.cumprod(1 + spy), "CL=F": level}
    for node, ex in (("airline_stocks_down", -0.2 * oil), ("oil_producer_stocks_up", lead)):
        for k, t in enumerate(linktest.NODES[node]["tickers"]):
            px[t] = (30 + k) * np.cumprod(1 + spy + ex + rng.normal(0, 0.004, n))
    for node in ("bank_stocks_up", "tech_stocks_down", "homebuilder_stocks_down"):
        for k, t in enumerate(linktest.NODES[node]["tickers"]):
            px[t] = (30 + k) * np.cumprod(1 + spy + rng.normal(0, 0.02, n))
    curve = pd.DataFrame({"date": DATES, "10 Yr": 2.5 + np.cumsum(rng.normal(0, 0.14, n))})
    tre = Fact(entity="MACRO", attr=treasury.ATTR, value=series.store(curve), unit="percent", as_of=DATES[-1],
               known_at=DATES[-1], tier="reported", source=dict(treasury.SOURCE),
               context={"rows": n, "first": DATES[0], "last": DATES[-1], "tenors": ["10 Yr"]})
    return [snap(t, v) for t, v in px.items()] + [tre]


class PairsTest(unittest.TestCase):
    def test_the_links_give_five_measured_pairs_with_their_signs(self):
        got = {(p["cause"], p["effect"]): p["sign"] for p in linktest.pairs(chain.load_links())}
        self.assertEqual(got, {("oil_price_up", "airline_stocks_down"): -1, ("oil_price_up", "oil_producer_stocks_up"): 1,
                               ("interest_rates_up", "homebuilder_stocks_down"): -1,
                               ("interest_rates_up", "bank_stocks_up"): 1, ("interest_rates_up", "tech_stocks_down"): -1})
        ids = [t["id"] for t in linktest.tests()]
        self.assertEqual(len(ids), 10)
        self.assertIn("L1-oil-airlines-next", ids)
        air = next(p for p in linktest.pairs(chain.load_links()) if p["effect"] == "airline_stocks_down")
        self.assertEqual([s["effect"] for s in air["path"]],
                         ["airline_fuel_costs_up", "airline_profits_down", "airline_stocks_down"])

    def test_what_is_not_measured_says_what_would_measure_it(self):
        self.assertIsNone(linktest.needs("oil_price_up"))
        self.assertIn("FRED", linktest.needs("inflation_up"))
        self.assertIn("quarterly", linktest.needs("airline_profits_down"))

    def test_a_cause_reached_only_through_another_series_is_not_paired_past_it(self):
        links = [{"cause": "oil_price_up", "effect": "interest_rates_up", "sign": 1},
                 {"cause": "interest_rates_up", "effect": "bank_stocks_up", "sign": 1}]
        self.assertEqual([(p["cause"], p["effect"]) for p in linktest.pairs(links)],
                         [("interest_rates_up", "bank_stocks_up")])


class DataTest(unittest.TestCase):
    def test_month_ends_and_stale_values(self):
        cal = ["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28"]
        ends = linktest.month_ends(cal)
        self.assertEqual(ends, {"2020-01": "2020-01-31", "2020-02": "2020-02-28"})
        got = linktest.sample(["2020-01-20", "2020-02-27"], [1.0, 2.0], ends)
        self.assertEqual(got, {"2020-02": 2.0})  # January's last value is 11 days old
        self.assertEqual(linktest.sample(["2020-01-31", "2020-02-27"], [1.0, 2.0], ends, stale=0), {"2020-01": 1.0})
        self.assertEqual(linktest.month_starts(cal), {"2020-01": "2020-01-30", "2020-02": "2020-02-27"})

    def test_next_month_pairs_this_months_cause_with_next_months_effect(self):
        mo = {"months": ["2019-01", "2019-02", "2019-03"], "x": np.array([1.0, 2.0, 3.0]),
              "y": np.array([10.0, 20.0, 30.0]), "y_next": np.array([11.0, 21.0, 31.0]),
              "members": np.array([5, 5, 5]), "members_next": np.array([5, 4, 5])}
        nxt = linktest.window_pairs(mo, "next", "2019-02", "2019-03")
        self.assertEqual((list(nxt["x"]), list(nxt["y"]), nxt["months"]), ([1.0, 2.0], [21.0, 31.0], ["2019-02", "2019-03"]))
        self.assertEqual(nxt["members"], 4.5)
        same = linktest.window_pairs(mo, "same", "2019-02", "2019-03")
        self.assertEqual((list(same["x"]), list(same["y"])), ([2.0, 3.0], [20.0, 30.0]))

    def test_the_statistics_and_controls(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 80)
        y = -0.5 * x + rng.normal(0, 0.5, 80)
        self.assertAlmostEqual(linktest.slope(x, y), -0.5, delta=0.15)
        self.assertLess(linktest.perm_p(x, y, -1, 999, rng), 0.01)
        self.assertGreater(linktest.perm_p(x, y, 1, 999, rng), 0.9)  # one-sided: the other way is no evidence
        ctl = linktest.controls(x, y, -1, rng, {**linktest.CONTROLS, "n_perm": 200})
        self.assertTrue(ctl["pass"], ctl)
        self.assertEqual(ctl["n_shifts"], 75)
        self.assertLess(ctl["oracle_p"], 0.01)
        # Two independent but slow-moving series: the shuffle null is too narrow, and the shifts say so.
        a, b = np.zeros(120), np.zeros(120)
        for i in range(1, 120):
            a[i], b[i] = 0.95 * a[i - 1] + rng.normal(), 0.95 * b[i - 1] + rng.normal()
        slow = linktest.controls(a, b, 1 if linktest.slope(a, b) > 0 else -1, rng, {**linktest.CONTROLS, "n_perm": 200})
        self.assertGreater(slow["shift_fpr"], 0.10)
        self.assertFalse(slow["pass"])


class DeclareAndTestTest(TempHome):
    def setUp(self):
        super().setUp()
        self.bank.add_many(world())

    def run_trial(self):
        self.assertTrue(linktest.is_trial())  # a temporary home never writes the committed ledger
        new = linktest.declare(self.bank, trial=True, log=None)
        self.assertEqual(len(new), 10)
        self.assertEqual(new[0]["spec"]["test"], linktest.TRIAL_SPLIT["test"])
        return linktest.test(self.bank, trial=True, log=None)

    def test_results_verdicts_and_the_results_file(self):
        recs = {r["id"]: r for r in self.run_trial()}
        led, res = linktest.paths(True)
        rows = {r["id"]: r for r in run.verdicts(led) if run.is_link(r)}
        same = recs["L1-oil-airlines-same"]["metrics"]["test"]
        self.assertAlmostEqual(same["per_move"], -0.02, delta=0.005)  # -0.2 a unit of oil, per 10%
        self.assertLess(same["p"], 0.01)
        self.assertEqual(rows["L1-oil-airlines-same"]["verdict"], "REPORT")
        self.assertFalse(math.isfinite(rows["L1-oil-airlines-same"]["bh_q"]))  # descriptive: outside the correction
        self.assertEqual(rows["L1-oil-producers-next"]["verdict"], "PASS")
        self.assertGreater(recs["L1-oil-airlines-next"]["metrics"]["test"]["p"], 0.05)
        self.assertEqual(rows["L1-oil-airlines-next"]["verdict"], "FAIL")
        self.assertTrue(all(r["controls"]["pass"] for r in recs.values()))
        text = res.read_text()
        self.assertIn("## Links", text)
        self.assertIn("| L1-oil-producers-next |", text)
        self.assertEqual(set(recs["L1-oil-producers-next"]["inputs"]), {*linktest.NODES["oil_producer_stocks_up"]["tickers"],
                                                                        "SPY", "CL=F"})
        self.assertIn("for each month with oil up 10%, the oil producer stocks' return over SPY the next month was", text)
        self.assertFalse(any(run.is_hypothesis(r) for r in rows.values()))
        self.assertEqual(linktest.test(self.bank, trial=True, log=None), [])  # nothing left to run

    def test_chain_follow_shows_the_tests_on_its_chains(self):
        self.run_trial()
        led, _ = linktest.paths(True)
        net = chain.train(steps=40, log=None)
        rows = chain.follow(net, chain.load_links(), "oil_price_up", 4, linktest.status(led))
        air = next(r for r in rows if r["exact"][-1] == "airline_stocks_down")
        self.assertEqual([t["id"] for t in air["tests"]], ["L1-oil-airlines-same", "L1-oil-airlines-next"])
        self.assertEqual(air["status"], "tested")
        rows = chain.follow(net, chain.load_links(), "strait_disruption", 4, linktest.status(led))
        self.assertTrue(all(r["status"] == "untested" for r in rows))  # its first link is not measured

    def test_refusals(self):
        with self.assertRaises(Refused):
            linktest.test(self.bank, ["L1-nope-next"], trial=True, log=None)
        linktest.declare(self.bank, trial=True, log=None)
        s = next(r["spec"] for r in linktest._records(linktest.paths(True)[0]) if r["op"] == "declare")
        with self.assertRaises(Refused):  # no measured TEST months: refused, nothing recorded
            linktest.evaluate(linktest.monthly(self.bank, s), {**s, "test": ["2030-01", "2030-12"]})
        led = linktest.paths(True)[0]
        real = dict(next(r for r in linktest._records(led) if r["op"] == "declare"))
        real_spec = {**real["spec"], "trial": False, "test": ["2019-01", "2026-09"], "id": "L1-real-next"}
        linktest.ledger.append({**real, "id": "L1-real-next", "spec": real_spec, "spec_sha": linktest.spec_sha(real_spec)}, led)
        with self.assertRaises(Refused):  # a trial never reads TEST years, even from a copy of the real ledger
            linktest.test(self.bank, ["L1-real-next"], trial=True, log=None)
        bad = {**real, "id": "L1-bad-next", "spec": {**real["spec"], "id": "L1-bad-next", "sign": -real["spec"]["sign"]}}
        linktest.ledger.append(bad, led)
        with self.assertRaises(Refused):  # the spec no longer matches its sha
            linktest.test(self.bank, ["L1-bad-next"], trial=True, log=None)
        with mock.patch.object(linktest, "VERSION", 1), mock.patch.dict(linktest.NODES["airline_stocks_down"],
                                                                       {"tickers": ["DAL", "UAL"]}), \
             self.assertRaises(Refused):  # a changed design under a declared id
            linktest.declare(self.bank, trial=True, log=None)
