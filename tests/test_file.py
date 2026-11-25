"""The company file (brain/file.py): each section from the bank, a missing piece said plainly, nothing of its own."""
from __future__ import annotations

from unittest import mock

from brain import file as cf
from brain.metrics import PriceBook
from tests.util import TempHome, filed, price_snapshot

REV, NI = "us-gaap:Revenues", "us-gaap:NetIncomeLoss"
CAPEX = "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"
SHARES = "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"


def year(attr, value, y, **ctx):
    return filed("X", attr, value, f"{y}-12-31", f"{y + 1}-02-20", form="10-K", start=f"{y}-01-01", duration_days=364,
                 **ctx)


class Words(TempHome):
    def test_money(self):
        self.assertEqual(cf.money(3.77e12), "$3.77 trillion")
        self.assertEqual(cf.money(-38.16e6), "$-38.16 million")
        self.assertEqual(cf.money(299000), "$299,000")

    def test_the_next_report_is_a_year_after_the_first_whose_anniversary_has_not_passed(self):
        eps = [{"date": d} for d in ("2025-11-19", "2026-02-25", "2026-05-20", "2026-08-26", "2026-11-18")]
        self.assertEqual(cf.next_report(eps, "2026-11-25"), ("2027-02-24", "2026-02-25"))
        # a report due three days ago by last year's date may still be coming
        self.assertEqual(cf.next_report(eps[:-1], "2026-11-21"), ("2026-11-18", "2025-11-19"))
        # once it is out, the anniversary of the report before it is no guess: too close to the one just made
        self.assertEqual(cf.next_report(eps, "2026-11-19")[0], "2027-02-24")
        self.assertIsNone(cf.next_report([], "2026-11-25"))

    def test_partners_are_not_customers(self):
        out = cf.moves_with(["  partners the stocks that moved most with it over the past year: AAA (0.41)."])
        self.assertIn("AAA (0.41)", out[0])
        self.assertIn("not its customers or suppliers", out[1])
        self.assertIn("no partners", cf.moves_with([])[0])

    def test_a_stock_outside_the_scan_says_so_and_a_size_is_not_a_direction(self):
        with mock.patch("brain.scanwatch.latest_scan", return_value=None):
            out = cf.how_far(self.bank, "X", None)
        self.assertIn("not in the latest recorded scan", out[0])
        self.assertEqual(out[-1], "  These are sizes. None says which way.")

    def test_the_scan_line_is_quoted_with_its_test(self):
        rec = {"asof": "2026-11-24", "rows": {"X": [-0.128, -0.069, 0.077, 0.132, 0.012]}, "newer_listings": ["X"]}
        with mock.patch("brain.scanwatch.latest_scan", return_value=(None, rec)), \
                mock.patch.object(cf, "verdict", side_effect=lambda *ids: ", ".join(f"{i} PASS" for i in ids)):
            out = cf.how_far(self.bank, "X", None)
        self.assertIn("2 times in 3 between -6.9% and +7.7%; 9 times in 10 between -12.8% and +13.2% (W1b PASS; a newer "
                      "listing)", out[0])
        self.assertIn("1.2% (S1a PASS; tested on the research universe only)", out[1])

    def test_the_run_into_a_report_shows_only_when_the_report_is_near(self):
        setup = {"run21": 2.5, "run5": 0.4, "run63": 1.0, "hi52": -0.02, "med8": 0.05, "last": 0.04, "up8": 0.5,
                 "mean8": 0.1, "vix": 15.0, "spy21": 0.01}
        with mock.patch("brain.scanwatch.latest_scan", return_value=None):
            far = cf.how_far(self.bank, "X", setup, 60)
            near = cf.how_far(self.bank, "X", setup, 3)
        self.assertFalse(any("ran up" in x for x in far))
        self.assertTrue(any("with its report about 3 days away, into the report it ran up" in x for x in near))


class Accounts(TempHome):
    def setUp(self):
        super().setUp()
        self.bank.add_many([
            year(REV, 100.0e6, 2018), year(REV, 120.0e6, 2019), year(NI, 24.0e6, 2019),
            year(CAPEX, 9.0e6, 2012),                      # last tagged years ago: not this year's
            year(SHARES, 10.0e6, 2019),
            filed("X", "us-gaap:CashAndCashEquivalentsAtCarryingValue", 30.0e6, "2019-12-31", "2020-02-20", form="10-K"),
            filed("X", "identity.industry", "Widgets", "2020-01-01", "2020-01-01"),
            filed("X", "identity.fiscal_year_end", "1231", "2020-01-01", "2020-01-01"),
        ])
        dates = ["2020-02-27", "2020-02-28", "2020-03-02", "2020-03-03"]
        self.bank.add(price_snapshot("X", dates, [48.0, 48.0, 48.0, 48.0]))
        self.view = self.bank.view("2020-03-03")
        self.book = PriceBook(self.bank, ["X"], asof="2020-03-03")

    def test_the_year_its_margins_and_what_the_price_pays(self):
        out = "\n".join(cf.accounts(self.view, "X", self.book))
        self.assertIn("fiscal year to 2019-12-31 (filed 2020-02-20): revenue $120.00 million (+20.0% on the year)", out)
        self.assertIn("net income $24.00 million (20.0% of revenue)", out)
        self.assertIn("cash $30.00 million", out)
        self.assertNotIn("plant and equipment", out)       # the 2012 figure is not shown as this year's
        # 10 million shares at $48 = $480 million: 20 times earnings of $24 million, 4 times revenue of $120 million
        self.assertIn("20.0 times the latest fiscal year's earnings (P/E)", out)
        self.assertIn("4.0 times the year's revenue", out)

    def test_what_it_is(self):
        out = cf.what_it_is(self.view, "X", self.book)
        self.assertEqual(out[0], "  Widgets; fiscal year ends 12-31")
        self.assertIn("market value $480.00 million", out[1])

    def test_a_loss_has_no_pe(self):
        self.bank.add(year(NI, -5.0e6, 2019, accn="restated"))
        self.bank.add(filed("X", NI, -5.0e6, "2019-12-31", "2020-02-25", form="10-K/A", start="2019-01-01",
                            duration_days=364))
        out = "\n".join(cf.accounts(self.bank.view("2020-03-03"), "X", self.book))
        self.assertIn("no P/E: the latest fiscal year was a loss", out)


class WholeFile(TempHome):
    def test_an_unknown_company_gets_every_section_and_no_error(self):
        with mock.patch("brain.scanwatch.latest_scan", return_value=None):
            out = cf.build(self.bank, "zzzz", asof="2026-11-25")
        self.assertTrue(out[0].startswith("THE COMPANY FILE: ZZZZ"))
        heads = [x for x in out if x[:2] in {f"{k}." for k in range(1, 9)}]
        self.assertEqual(len(heads), 8)
        # the mosaic stops without market prices: the file says so and goes on
        self.assertIn("  not available: no SPY or ^VIX prices in the bank", out)
        self.assertEqual(sum("not available" in x for x in out), 1, out)
        self.assertIn("  no annual accounts in the bank (python -m brain ingest ZZZZ)", out)
        self.assertIn("  no results report in the bank for it", out)
        self.assertTrue(any("a tested direction: no test of which way a stock goes has passed" in x for x in out))
