from brain import metrics
from brain.metrics import PriceBook
from tests.util import TempHome, filed, price_snapshot


class MetricsTest(TempHome):
    def setUp(self):
        super().setUp()
        b = self.bank
        b.add_many([
            filed("X", "us-gaap:NetIncomeLoss", 50.0, "2019-12-31", "2020-02-20", form="10-K", start="2019-01-01", duration_days=364),
            filed("X", "us-gaap:NetIncomeLoss", 12.0, "2020-03-31", "2020-04-30", form="10-Q", start="2020-01-01", duration_days=90),
            filed("X", "us-gaap:Assets", 1000.0, "2018-12-31", "2019-02-20", form="10-K"),
            filed("X", "us-gaap:Assets", 1100.0, "2019-12-31", "2020-02-20", form="10-K"),
            # 100 shares reported 2020-02-20; a 2:1 split on 2020-03-02 doubles them.
            filed("X", "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding", 100.0, "2019-12-31", "2020-02-20",
                  form="10-K", start="2019-01-01", duration_days=364),
        ])
        # Yahoo-style prices back-adjusted for the split: true pre-split close 20 shows as 10.
        dates = ["2020-02-27", "2020-02-28", "2020-03-02", "2020-03-03"]
        b.add(price_snapshot("X", dates, [10.0, 10.0, 10.0, 10.0], splits=[("2020-03-02", 2.0)]))
        self.book = PriceBook(b, ["X"])

    def test_annual_ignores_quarters(self):
        a = metrics.annual(self.bank.view("2020-06-01"), "X", "us-gaap:NetIncomeLoss")
        self.assertEqual([f.value for f in a], [50.0])

    def test_market_cap_is_split_consistent(self):
        # Before the split: 100 shares x true price 20 = 2000. After: 200 x 10 = 2000.
        before = metrics.market_cap(self.bank.view("2020-02-28"), "X", self.book)
        after = metrics.market_cap(self.bank.view("2020-03-03"), "X", self.book)
        self.assertAlmostEqual(before.value, 2000.0)
        self.assertAlmostEqual(after.value, 2000.0)

    def test_earnings_yield_and_asset_growth(self):
        v = self.bank.view("2020-03-03")
        self.assertAlmostEqual(metrics.earnings_yield(v, "X", self.book).value, 50 / 2000)
        self.assertAlmostEqual(metrics.asset_growth(v, "X").value, 0.1)
        self.assertIsNone(metrics.asset_growth(self.bank.view("2020-02-19"), "X"))

    def test_price_reads_stop_at_view_date(self):
        d, _ = self.book.adj_upto(self.bank.view("2020-02-28"), "X")
        self.assertEqual(list(d), ["2020-02-27", "2020-02-28"])

    def test_derived_fact_cites_inputs(self):
        v = self.bank.view("2020-03-03")
        m = metrics.earnings_yield(v, "X", self.book)
        f = metrics.to_fact("X", "earnings_yield", m)
        self.assertEqual(f.tier, "derived")
        self.assertEqual(len(f.inputs), 3)
        for ref in f.inputs:
            self.assertLessEqual(self.bank.get(ref).known_at, f.known_at)
