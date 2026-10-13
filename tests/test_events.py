import numpy as np
import pandas as pd

from brain import events
from brain.bank import Fact
from brain.metrics import PriceBook
from tests.test_quarterly import EPS, q
from tests.util import TempHome, price_snapshot

DAYS = pd.bdate_range("2023-01-02", "2024-12-31").strftime("%Y-%m-%d").tolist()


def filing(t, form, known, items="", accn=None):
    return Fact(entity=t, attr="filing", value=form, as_of=known[:10], known_at=known, tier="filed",
                source={"name": "SEC EDGAR", "url": f"https://example.test/{accn or known}"},
                context={"accn": accn or known, **({"items": items} if items else {})})


class EventsTest(TempHome):
    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(5)
        self.spy = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(DAYS)))
        self.x = 50 * np.cumprod(1 + rng.normal(0, 0.02, len(DAYS)))
        self.bank.add(price_snapshot("SPY", DAYS, list(self.spy)))
        self.bank.add(price_snapshot("X", DAYS, list(self.x)))
        self.book = PriceBook(self.bank, ["X", "SPY"])
        self.release = filing("X", "8-K", "2024-05-02T00:30:21Z", "2.02,9.01", "a1")  # 16:30 New York, May 1
        self.bank.add_many([
            self.release,
            filing("X", "8-K/A", "2024-05-06T12:00:00Z", "2.02", "a2"),           # amends the same release
            filing("X", "8-K", "2024-03-01T15:00:00Z", "5.02", "a3"),             # an officer change
            filing("X", "10-Q", "2024-05-03T20:00:00Z", "", "a4"),
            filing("X", "8-K", "2024-07-31T13:00:00Z", "2.02", "a5"),             # 09:00 New York: before the open
        ])

    def i(self, d):
        return DAYS.index(d)

    def test_earnings_events_and_distinct_releases(self):
        evs = events.earnings_events(self.bank.view("2024-12-31"), "X")
        self.assertEqual([e.context["accn"] for e in evs], ["a1", "a2", "a5"])
        self.assertEqual([e.context["accn"] for e in events.distinct(evs)], ["a1", "a5"])
        self.assertEqual(events.earnings_events(self.bank.view("2024-05-01"), "X"), [])  # known only after 00:30Z May 2

    def test_event_day_is_first_close_at_or_after_publication(self):
        d = np.array(DAYS)
        self.assertEqual(DAYS[events.event_index(d, "2024-05-02T00:30:21Z")], "2024-05-02")  # after the close
        self.assertEqual(DAYS[events.event_index(d, "2024-07-31T13:00:00Z")], "2024-07-31")  # before the close
        self.assertEqual(DAYS[events.event_index(d, "2024-07-31T20:30:00Z")], "2024-08-01")  # 16:30 EDT
        self.assertEqual(DAYS[events.event_index(d, "2024-06-01T14:00:00Z")], "2024-06-03")  # a Saturday

    def test_reaction_windows_and_point_in_time_cut(self):
        e = self.i("2024-05-02")
        ex = lambda a, b: (self.x[b] / self.x[a] - 1) - (self.spy[b] / self.spy[a] - 1)
        full = events.reaction(self.book, self.bank.view("2024-12-31"), "X", self.release)
        self.assertEqual(full["date"], "2024-05-02")
        self.assertAlmostEqual(full["car_3d"], ex(e - 2, e + 1))
        self.assertAlmostEqual(full["drift_20d"], ex(e + 1, e + 21))
        self.assertAlmostEqual(full["drift_60d"], ex(e + 1, e + 61))
        self.assertEqual(full["event"], self.release.ref)
        # Thirty sessions later the 60-day drift has not happened yet: None, not a partial number.
        mid = events.reaction(self.book, self.bank.view(DAYS[e + 30]), "X", self.release)
        self.assertAlmostEqual(mid["car_3d"], full["car_3d"])
        self.assertAlmostEqual(mid["drift_20d"], full["drift_20d"])
        self.assertIsNone(mid["drift_60d"])
        self.assertIsNone(events.reaction(self.book, self.bank.view(DAYS[e + 20]), "X", self.release)["drift_20d"])
        self.assertIsNone(events.reaction(self.book, self.bank.view(DAYS[e]), "X", self.release)["car_3d"])
        self.assertIsNone(events.reaction(self.book, self.bank.view("2024-05-01"), "X", self.release))  # not public yet

    def test_earnings_table(self):
        ends = pd.date_range("2021-12-31", periods=11, freq="QE").strftime("%Y-%m-%d").tolist()
        eps = [0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.3, 1.5, 1.6, 1.0, 1.4]
        for v, end in zip(eps, ends):
            start = (pd.Timestamp(end) - pd.offsets.QuarterBegin(startingMonth=1)).strftime("%Y-%m-%d")
            known = (pd.Timestamp(end) + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
            self.bank.add(q("X", EPS, v, start, end, known, unit="USD/shares"))
        rows = events.earnings_table(self.bank.view("2024-12-31"), "X", self.book)
        self.assertEqual([r["period_end"] for r in rows], ends[::-1])
        self.assertEqual(set(rows[0]), {"period_end", "filed", "eps", "eps_yoy", "sue", "car_3d", "drift_20d",
                                        "drift_60d", "cite"})
        last, prev = rows[0], rows[1]  # Q2 2024 matches the 2024-07-31 release; Q1 2024 the May 1 release
        self.assertAlmostEqual(last["eps_yoy"], 1.4 / 1.3 - 1)
        self.assertAlmostEqual(prev["eps_yoy"], 1.0 / 1.2 - 1)
        self.assertEqual(prev["filed"], "2024-05-05")
        full = events.reaction(self.book, self.bank.view("2024-12-31"), "X", self.release)
        self.assertAlmostEqual(prev["car_3d"], full["car_3d"])
        self.assertIn(self.release.ref, prev["cite"])
        self.assertIsNone(rows[3]["car_3d"])  # 2023 quarters: no release in the bank
        self.assertIsNone(rows[-1]["eps_yoy"])
        self.assertIsNotNone(last["sue"])  # six YoY differences in the 8 quarters before it
        self.assertIsNone(prev["sue"])      # only five
        self.assertIsNone(rows[-1]["sue"])

    def test_release_is_the_last_one_public_by_the_eps_filing(self):
        ends = pd.date_range("2022-12-31", periods=7, freq="QE").strftime("%Y-%m-%d").tolist()
        for v, end in zip([1.0, 1.1, 1.2, 1.3, 1.2, 1.3, 1.4], ends):  # EPS filed 35 days after each quarter
            start = (pd.Timestamp(end) - pd.offsets.QuarterBegin(startingMonth=1)).strftime("%Y-%m-%d")
            known = (pd.Timestamp(end) + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
            self.bank.add(q("X", EPS, v, start, end, known, unit="USD/shares"))
        pre1 = filing("X", "8-K", "2024-04-09T12:00:00Z", "2.02,9.01", "pre1")  # Q1 2024 pre-announcement
        guid = filing("X", "8-K", "2024-06-12T12:00:00Z", "2.02,9.01", "guid")  # guidance, after Q1's 10-Q
        pre3 = filing("X", "8-K", "2023-10-10T12:00:00Z", "2.02", "pre3")       # Q3 2023 interim update
        res3 = filing("X", "8-K", "2023-11-05T00:30:00Z", "2.02", "res3")       # 20:30 New York on the 10-Q's day
        late4 = filing("X", "8-K", "2024-02-20T12:00:00Z", "2.02", "late4")     # Q4 2023: only after its EPS
        self.bank.add_many([pre1, guid, pre3, res3, late4])
        rows = {r["period_end"]: r for r in events.earnings_table(self.bank.view("2024-12-31"), "X", self.book)}
        q1 = rows["2024-03-31"]
        self.assertEqual(q1["filed"], "2024-05-05")
        self.assertEqual(q1["cite"][-1], self.release.ref)  # the results, not the warning before or the guidance after
        full = events.reaction(self.book, self.bank.view("2024-12-31"), "X", self.release)
        self.assertAlmostEqual(q1["car_3d"], full["car_3d"])
        self.assertEqual(rows["2023-09-30"]["cite"][-1], res3.ref)   # same New York day as the filing counts
        self.assertEqual(rows["2023-12-31"]["cite"][-1], late4.ref)  # nothing by the filing: the first after it
        self.assertEqual(rows["2024-06-30"]["cite"][-1], events.earnings_events(self.bank.view("2024-12-31"), "X")[-1].ref)
        self.assertEqual(len(rows["2023-03-31"]["cite"]), 1)         # no release in its window: EPS only

    def test_earnings_reaction_signal_is_point_in_time(self):
        from brain.research import signals
        from brain.research.panel import build
        p = build(self.bank, ["X"], start="2023-01-01")
        S = signals.compute(self.bank, p, ["earnings_reaction"])["earnings_reaction"][:, 0]
        val = {str(p.dates[i])[:7]: S[m] for m, i in enumerate(p.rebal)}
        full = lambda ev: events.reaction(self.book, self.bank.view("2024-12-31"), "X", ev)["car_3d"]
        july = events.earnings_events(self.bank.view("2024-12-31"), "X")[-1]
        self.assertTrue(np.isnan(val["2024-04"]))
        self.assertAlmostEqual(val["2024-05"], full(self.release))  # the release, not its amendment
        self.assertTrue(np.isnan(val["2024-06"]))                   # known 58 days before the June rebalance
        self.assertTrue(np.isnan(val["2024-07"]))                   # July 31 release: day +1 not closed yet
        self.assertAlmostEqual(val["2024-08"], full(july))
