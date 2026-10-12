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


class AnnualMergeTest(TempHome):
    def test_concepts_merge_by_priority_not_by_filing_date(self):
        # Both debt concepts are filed every year, the total one later; the row must not alternate.
        for y, (noncur, total) in {2021: (80.0, 90.0), 2022: (70.0, 85.0), 2023: (60.0, 75.0)}.items():
            self.bank.add_many([
                filed("X", "us-gaap:LongTermDebtNoncurrent", noncur, f"{y}-12-31", f"{y + 1}-02-10", form="10-K"),
                filed("X", "us-gaap:LongTermDebt", total, f"{y}-12-31", f"{y + 1}-02-11" if y % 2 else f"{y + 1}-02-09",
                      form="10-K"),
            ])
        s = metrics.annual(self.bank.view("2024-06-01"), "X", ["us-gaap:LongTermDebtNoncurrent", "us-gaap:LongTermDebt"],
                           flow=False)
        self.assertEqual([f.value for f in s], [80.0, 70.0, 60.0])
        self.assertEqual({f.attr for f in s}, {"us-gaap:LongTermDebtNoncurrent"})

    def test_older_concept_fills_years_the_preferred_one_lacks(self):
        self.bank.add_many([
            filed("X", "us-gaap:SalesRevenueNet", 100.0, "2016-12-31", "2017-02-10", form="10-K", start="2016-01-01", duration_days=365),
            filed("X", "us-gaap:Revenues", 120.0, "2018-12-31", "2019-02-10", form="10-K", start="2018-01-01", duration_days=364),
        ])
        s = metrics.annual(self.bank.view("2019-06-01"), "X", ["us-gaap:Revenues", "us-gaap:SalesRevenueNet"])
        self.assertEqual([(f.as_of, f.value) for f in s], [("2016-12-31", 100.0), ("2018-12-31", 120.0)])


class StalenessTest(TempHome):
    def test_ratios_growth_and_fcf_refuse_old_fiscal_years(self):
        for y in (2008, 2009):
            self.bank.add_many([
                filed("X", "us-gaap:Revenues", 100.0 + y - 2008, f"{y}-12-31", f"{y + 1}-02-10", form="10-K",
                      start=f"{y}-01-01", duration_days=364),
                filed("X", "us-gaap:NetIncomeLoss", 10.0, f"{y}-12-31", f"{y + 1}-02-10", form="10-K",
                      start=f"{y}-01-01", duration_days=364),
                filed("X", "us-gaap:NetCashProvidedByUsedInOperatingActivities", 20.0, f"{y}-12-31", f"{y + 1}-02-10",
                      form="10-K", start=f"{y}-01-01", duration_days=364),
                filed("X", "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", 5.0, f"{y}-12-31", f"{y + 1}-02-10",
                      form="10-K", start=f"{y}-01-01", duration_days=364),
            ])
        fresh, stale = self.bank.view("2010-06-01"), self.bank.view("2026-06-01")
        net_margin = lambda v: metrics.ratio(v, "X", "us-gaap:NetIncomeLoss", "us-gaap:Revenues")
        self.assertAlmostEqual(net_margin(fresh).value, 10 / 101)
        self.assertIsNotNone(metrics.revenue_growth(fresh, "X"))
        self.assertEqual(metrics.free_cash_flow(fresh, "X").value, 15.0)
        # Sixteen years on, the same numbers are history, not current metrics.
        self.assertIsNone(net_margin(stale))
        self.assertIsNone(metrics.revenue_growth(stale, "X"))
        self.assertIsNone(metrics.free_cash_flow(stale, "X"))
