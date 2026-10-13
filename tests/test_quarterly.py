import numpy as np

from brain import metrics
from brain.bank import Fact
from brain.metrics import PriceBook
from tests.util import TempHome, filed, price_snapshot

NI = "us-gaap:NetIncomeLoss"
EPS = "us-gaap:EarningsPerShareDiluted"


def q(t, attr, v, start, end, known, form="10-Q", unit="USD"):
    from brain.timeutil import days_between
    f = filed(t, attr, v, end, known, form=form, start=start, duration_days=days_between(start, end))
    return Fact(entity=f.entity, attr=f.attr, value=f.value, as_of=f.as_of, known_at=f.known_at, tier=f.tier,
                source=f.source, context=f.context, unit=unit)


def fiscal_year(t, attr, y, quarters, known, *, report_q4=False):
    """Calendar fiscal year y: Q1-Q3 from 10-Qs, plus 6- and 9-month YTD, plus the 10-K year."""
    s = [f"{y}-01-01", f"{y}-04-01", f"{y}-07-01", f"{y}-10-01"]
    e = [f"{y}-03-31", f"{y}-06-30", f"{y}-09-30", f"{y}-12-31"]
    out = [q(t, attr, quarters[i], s[i], e[i], known[i]) for i in range(3)]
    out.append(q(t, attr, quarters[0] + quarters[1], s[0], e[1], known[1]))
    out.append(q(t, attr, sum(quarters[:3]), s[0], e[2], known[2]))
    out.append(q(t, attr, sum(quarters), s[0], e[3], known[3], form="10-K"))
    if report_q4:
        out.append(q(t, attr, quarters[3], s[3], e[3], known[3], form="10-K"))
    return out


def known_dates(y):
    return [f"{y}-05-01", f"{y}-08-01", f"{y}-11-01", f"{y + 1}-02-20"]


class QuarterlyTest(TempHome):
    def test_q4_is_year_minus_nine_months_known_when_both_are(self):
        self.bank.add_many(fiscal_year("X", NI, 2023, [10.0, 20.0, 30.0, 45.0], known_dates(2023)))
        before = metrics.quarterly(self.bank.view("2024-02-19"), "X", NI)
        self.assertEqual([x.end for x in before], ["2023-03-31", "2023-06-30", "2023-09-30"])
        qs = metrics.quarterly(self.bank.view("2024-03-01"), "X", NI)
        self.assertEqual([x.value for x in qs], [10.0, 20.0, 30.0, 45.0])
        q4 = qs[-1]
        self.assertTrue(q4.derived)
        self.assertEqual((q4.start, q4.end), ("2023-10-01", "2023-12-31"))
        self.assertEqual(len(q4.inputs), 2)
        self.assertEqual({f.context["duration_days"] for f in q4.inputs}, {364, 272})
        self.assertEqual(q4.known_at, "2024-02-20T23:59:59Z")
        self.assertFalse(any(x.derived for x in qs[:3]))
        self.assertEqual([len(x.inputs) for x in qs[:3]], [1, 1, 1])

    def test_a_reported_q4_is_used_as_is(self):
        self.bank.add_many(fiscal_year("X", NI, 2023, [10.0, 20.0, 30.0, 45.0], known_dates(2023), report_q4=True))
        q4 = metrics.quarterly(self.bank.view("2024-03-01"), "X", NI)[-1]
        self.assertFalse(q4.derived)
        self.assertEqual(q4.value, 45.0)

    def test_restated_quarter_latest_known_wins_and_concepts_merge_by_priority(self):
        self.bank.add_many(fiscal_year("X", NI, 2023, [10.0, 20.0, 30.0, 45.0], known_dates(2023)))
        self.bank.add(q("X", NI, 11.0, "2023-01-01", "2023-03-31", "2024-05-01"))  # restated a year later
        self.bank.add(q("X", "us-gaap:ProfitLoss", 99.0, "2023-04-01", "2023-06-30", "2023-08-01"))
        self.bank.add(q("X", "us-gaap:ProfitLoss", 7.0, "2024-01-01", "2024-03-31", "2024-05-01"))
        v = self.bank.view("2024-06-01")
        qs = metrics.quarterly(v, "X", [NI, "us-gaap:ProfitLoss"])
        self.assertEqual([x.value for x in qs], [11.0, 20.0, 30.0, 45.0, 7.0])  # Q2 keeps NetIncomeLoss; 2024Q1 falls back
        self.assertEqual(metrics.quarterly(self.bank.view("2024-03-01"), "X", NI)[0].value, 10.0)


class TtmTest(TempHome):
    def test_sum_of_four_contiguous_quarters(self):
        self.bank.add_many(fiscal_year("X", NI, 2023, [10.0, 20.0, 30.0, 45.0], known_dates(2023)))
        self.bank.add(q("X", NI, 15.0, "2024-01-01", "2024-03-31", "2024-05-01"))
        m = metrics.ttm(self.bank.view("2024-03-01"), "X", NI)
        self.assertEqual((m.value, m.as_of), (105.0, "2023-12-31"))
        m = metrics.ttm(self.bank.view("2024-05-02"), "X", NI)
        self.assertEqual((m.value, m.as_of), (110.0, "2024-03-31"))
        self.assertEqual(len(m.inputs), 5)  # Q2, Q3 reported; Q4 = year - 9M; 2024Q1
        self.assertLessEqual(max(f.known_at for f in m.inputs), "2024-05-02")

    def test_gappy_or_stale_is_none(self):
        self.bank.add_many(fiscal_year("X", NI, 2023, [10.0, 20.0, 30.0, 45.0], known_dates(2023)))
        # 2024: Q2 only (Q1 missing): the last four are not contiguous.
        self.bank.add(q("X", NI, 15.0, "2024-04-01", "2024-06-30", "2024-08-01"))
        self.assertIsNone(metrics.ttm(self.bank.view("2024-08-02"), "X", NI))
        self.assertIsNotNone(metrics.ttm(self.bank.view("2024-07-01"), "X", NI))
        self.assertIsNone(metrics.ttm(self.bank.view("2024-07-19"), "X", NI))  # Q4 2023 ended > 200 days ago
        self.assertIsNone(metrics.ttm(self.bank.view("2023-12-01"), "X", NI))  # three quarters only

    def test_units_must_agree(self):
        self.bank.add_many(fiscal_year("X", NI, 2023, [10.0, 20.0, 30.0, 45.0], known_dates(2023)))
        self.bank.add(q("X", NI, 15.0, "2024-01-01", "2024-03-31", "2024-05-01", unit="EUR"))
        self.assertIsNone(metrics.ttm(self.bank.view("2024-05-02"), "X", NI))


class EarningsYieldTtmTest(TempHome):
    def setUp(self):
        super().setUp()
        dates = ["2024-02-28", "2024-02-29", "2024-03-01"]
        self.bank.add(price_snapshot("X", dates, [10.0, 10.0, 10.0]))
        self.book = PriceBook(self.bank, ["X"])

    def test_broader_concepts_and_old_metric_untouched(self):
        # The company tags ProfitLoss, not NetIncomeLoss, and basic shares only.
        self.bank.add_many(fiscal_year("X", "us-gaap:ProfitLoss", 2023, [1.0, 2.0, 3.0, 4.0], known_dates(2023)))
        self.bank.add(filed("X", "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic", 50.0, "2023-12-31",
                            "2024-02-20", form="10-K", start="2023-01-01", duration_days=364))
        v = self.bank.view("2024-03-01")
        m = metrics.earnings_yield_ttm(v, "X", self.book)
        self.assertAlmostEqual(m.value, 10.0 / 500.0)
        self.assertIn("us-gaap:WeightedAverageNumberOfSharesOutstandingBasic", {f.attr for f in m.inputs})
        # The declared H3 metric reads only NetIncomeLoss and diluted shares, exactly as before.
        self.assertIsNone(metrics.earnings_yield(v, "X", self.book))
        self.assertIsNone(metrics.market_cap(v, "X", self.book))

    def test_bound_and_currency(self):
        self.bank.add_many(fiscal_year("X", NI, 2023, [1000.0, 1000.0, 1000.0, 1000.0], known_dates(2023)))
        self.bank.add(filed("X", "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding", 50.0, "2023-12-31",
                            "2024-02-20", form="10-K", start="2023-01-01", duration_days=364))
        self.assertIsNone(metrics.earnings_yield_ttm(self.bank.view("2024-03-01"), "X", self.book))  # 4000/500 > 1


class SueTest(TempHome):
    EPS_PATH = [0.50, 0.55, 0.60, 0.70, 0.62, 0.66, 0.75, 0.80, 0.70, 0.80, 0.85, 0.90, 0.95]

    def _eps(self, values, t="X", first_year=2021, split_at=None, ratio=2.0):
        """Quarterly EPS from 10-Qs, filed 30 days after each quarter end; values filed after
        ``split_at`` are on the post-split basis (divided by ``ratio``)."""
        import pandas as pd
        ends = pd.date_range(f"{first_year}-03-31", periods=len(values), freq="QE").strftime("%Y-%m-%d").tolist()
        facts = []
        for v, e in zip(values, ends):
            start = (pd.Timestamp(e) - pd.offsets.QuarterBegin(startingMonth=1)).strftime("%Y-%m-%d")
            known = (pd.Timestamp(e) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            post = split_at is not None and known > split_at
            facts.append(q(t, EPS, v / ratio if post else v, start, e, known, unit="USD/shares"))
        return facts, ends

    def test_value_matches_the_definition(self):
        facts, ends = self._eps(self.EPS_PATH)
        self.bank.add_many(facts)
        v = self.bank.view("2024-05-15")
        m = metrics.sue(v, "X")
        e = self.EPS_PATH
        diffs = [e[k] - e[k - 4] for k in range(4, 12)]
        want = (e[12] - e[8]) / np.std(diffs, ddof=1)
        self.assertAlmostEqual(m.value, want)
        self.assertEqual(m.as_of, ends[-1])
        self.assertLessEqual(max(f.known_at for f in m.inputs), v.cutoff)

    def test_needs_six_prior_differences_and_a_fresh_quarter(self):
        facts, ends = self._eps(self.EPS_PATH[:10])  # quarter 10 has 5 prior YoY differences
        self.bank.add_many(facts)
        self.assertIsNone(metrics.sue(self.bank.view("2023-08-15"), "X"))
        facts, _ = self._eps(self.EPS_PATH[:11])
        self.bank.add_many(facts)
        self.assertIsNotNone(metrics.sue(self.bank.view("2023-11-15"), "X"))
        self.assertIsNone(metrics.sue(self.bank.view("2024-08-15"), "X"))  # latest quarter ended > 200 days ago

    def test_a_split_does_not_move_it(self):
        facts, _ = self._eps(self.EPS_PATH)
        self.bank.add_many(facts)
        plain = metrics.sue(self.bank.view("2024-05-15"), "X").value
        for t in ("Y", "Z"):  # a 2:1 split lands mid-series; Y's bank knows of it, Z's does not
            self.bank.add_many(self._eps(self.EPS_PATH, t=t, split_at="2022-10-15")[0])
        self.bank.add(Fact(entity="Y", attr="split", value=2.0, as_of="2022-10-15", known_at="2022-10-15",
                           tier="market", source={"name": "test", "url": "https://example.test/split"}))
        v = self.bank.view("2024-05-15")
        self.assertAlmostEqual(metrics.sue(v, "Y").value, plain)
        # Without the split the basis change looks like an earnings collapse.
        self.assertLess(metrics.sue(v, "Z").value, plain - 1)
