"""How stocks are connected (brain/connect.py): partners from a year of prices, drivers, and stories that tie stocks
together, on made-up returns whose links are known. Nothing touches the network."""
import unittest

import numpy as np
import pandas as pd

from brain import connect
from tests.test_scan import snapshot
from tests.util import TempHome


def world(n=250, seed=5):
    """A market, a shared factor for A and B beyond it, and C on its own."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.01, n)
    f = rng.normal(0, 0.02, n)
    a = 1.5 * m + f + rng.normal(0, 0.005, n)
    b = 0.5 * m + f + rng.normal(0, 0.005, n)
    c = 1.0 * m + rng.normal(0, 0.02, n)
    return m, np.column_stack([a, b, c])


class PartnersTest(unittest.TestCase):
    def test_partners_share_more_than_the_market(self):
        m, R = world()
        E, beta = connect.residuals(R, m)
        self.assertAlmostEqual(beta[0], 1.5, delta=0.3)
        self.assertAlmostEqual(beta[2], 1.0, delta=0.3)
        C = connect.correlations(E)
        self.assertGreater(C[0, 1], 0.9)                                  # A and B share the factor
        self.assertLess(abs(C[0, 2]), 0.2)                                # A and C share only the market, taken out
        self.assertTrue(np.isnan(C[0, 0]))
        self.assertEqual([t for t, _ in connect.partners(C, ["A", "B", "C"], "A")], ["B"])
        self.assertEqual(connect.partners(C, ["A", "B", "C"], "C"), [])
        self.assertEqual(connect.partners(C, ["A", "B", "C"], "Z"), [])

    def test_one_shared_jump_is_not_a_link(self):
        rng = np.random.default_rng(2)
        m = rng.normal(0, 0.01, 250)
        R = rng.normal(0, 0.02, (250, 2))
        R[100] = (0.60, 0.60)                                             # one day both jumped 60%
        raw = np.corrcoef(R[:, 0], R[:, 1])[0, 1]
        C = connect.correlations(connect.residuals(R, m)[0])
        self.assertGreater(raw, connect.MIN_CORR)                         # the plain correlation would call it a link
        self.assertLess(C[0, 1], connect.MIN_CORR)

    def test_too_few_shared_sessions_is_no_link(self):
        m, R = world()
        R[: 250 - connect.MIN_OBS + 5, 1] = np.nan                        # B listed late: under MIN_OBS sessions
        E, beta = connect.residuals(R, m)
        self.assertTrue(np.isnan(E[:, 1]).all())
        self.assertTrue(np.isnan(beta[1]))
        self.assertTrue(np.isnan(connect.correlations(E)[0, 1]))


class DriversTest(unittest.TestCase):
    def test_a_stock_that_moves_with_a_driver_is_found_and_one_that_does_not_is_left_out(self):
        rng = np.random.default_rng(9)
        n = 250
        m, btc, oil = rng.normal(0, 0.01, n), rng.normal(0, 0.03, n), rng.normal(0, 0.02, n)
        miner = 1.2 * m + 2.0 * btc + rng.normal(0, 0.01, n)
        bank = 0.9 * m + rng.normal(0, 0.01, n)
        expo = connect.exposures(np.column_stack([miner, bank]), m, {"BTC-USD": btc, "CL=F": oil})
        self.assertEqual(expo["symbols"], ["BTC-USD", "CL=F"])
        self.assertAlmostEqual(expo["beta"][0, 0], 2.0, delta=0.15)
        self.assertGreater(expo["t"][0, 0], 20)
        self.assertAlmostEqual(expo["market"][0], 1.2, delta=0.3)
        self.assertAlmostEqual(expo["usual"]["BTC-USD"], 0.03, delta=0.005)
        got = connect.carried(expo, 0, {"BTC-USD": -0.05, "CL=F": 0.02})
        self.assertEqual([x["driver"] for x in got], ["BTC-USD"])         # oil never moved it: not listed
        self.assertAlmostEqual(got[0]["implied"], -0.10, delta=0.01)
        self.assertEqual(connect.carried(expo, 1, {"BTC-USD": -0.05, "CL=F": 0.02}), [])
        self.assertEqual([t for t, _, _ in connect.most_exposed(expo, ["MINER", "BANK"], "BTC-USD", 5)], ["MINER"])
        self.assertEqual(connect.most_exposed(expo, ["MINER", "BANK"], "GC=F", 5), [])

    def test_a_driver_is_read_at_its_last_close_on_or_before_each_session(self):
        cal = np.array(["2026-11-06", "2026-11-09", "2026-11-10"])        # Friday, Monday, Tuesday
        frames = {
            "BTC-USD": pd.DataFrame({"date": ["2026-11-05", "2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09",
                                              "2026-11-10"], "close": [100.0, 100.0, 90.0, 95.0, 110.0, 121.0]}),
            "^TNX": pd.DataFrame({"date": ["2026-11-06", "2026-11-09", "2026-11-10"], "close": [4.00, 4.10, 4.05]}),
            "GC=F": pd.DataFrame({"date": ["2026-11-06", "2026-11-10"], "close": [50.0, 55.0]})}   # no bar on Monday

        def fetch(sym, start):
            if sym not in frames:
                raise ValueError("no such driver here")
            return frames[sym]

        d = connect.driver_changes(cal, fetch=fetch)
        self.assertEqual(sorted(d), ["BTC-USD", "GC=F", "^TNX"])           # the two with no data are left out
        np.testing.assert_allclose(d["BTC-USD"], [0.10, 0.10])            # Friday to Monday spans the weekend
        np.testing.assert_allclose(d["^TNX"], [0.10, -0.05])              # in points
        np.testing.assert_allclose(d["GC=F"], [0.0, 0.10])


class BookTest(TempHome):
    def test_the_book_reads_the_sessions_up_to_the_scan_s_close_and_drops_a_bad_day(self):
        d = pd.bdate_range("2025-01-02", "2026-11-12").strftime("%Y-%m-%d").to_numpy()
        up = 100 * np.cumprod(np.full(len(d), 1.001))
        a = 20 * np.cumprod(np.full(len(d), 1.002))
        a[-10:] *= 9.0                                                    # +800% in a day: a data error
        self.bank.add_many([snapshot("SPY", d, up), snapshot("^VIX", d, np.full(len(d), 15.0)), snapshot("AAA", d, a)])
        b = connect.book(self.bank, ["AAA", "NONE"], "2026-11-09", lookback=50)
        self.assertEqual((b["R"].shape, len(b["cal"])), ((50, 2), 51))
        self.assertEqual(b["cal"][-1], "2026-11-09")                      # nothing after the scan's close
        np.testing.assert_allclose(b["m"], 0.001, rtol=1e-4)            # stored prices are rounded
        self.assertEqual(int(np.isnan(b["R"][:, 0]).sum()), 1)            # the bad day
        self.assertTrue(np.isnan(b["R"][:, 1]).all())                     # a ticker the bank does not hold
        again = connect.cached_book(self.bank, ["AAA", "NONE"], "2026-11-09", lookback=50)
        cached = connect.cached_book(self.bank, ["AAA", "NONE"], "2026-11-09", lookback=50)
        np.testing.assert_allclose(again["R"], cached["R"], rtol=1e-6, equal_nan=True)
        self.assertEqual(list(cached["cal"]), list(b["cal"]))


NAMES = {"NVDA": "NVIDIA CORP", "CRWV": "CoreWeave, Inc.", "SMCI": "Super Micro Computer, Inc.",
         "GPN": "GLOBAL PAYMENTS INC", "TGT": "TARGET CORP", "VIR": "Vir Biotechnology, Inc.",
         "HTFL": "Heartflow, Inc.", "AMD": "ADVANCED MICRO DEVICES INC", "FMC": "FMC CORP"}


class StoriesTest(unittest.TestCase):

    def test_the_words_a_headline_would_use_for_a_company(self):
        self.assertEqual(connect.name_key("NVIDIA CORP"), "NVIDIA")
        self.assertEqual(connect.name_key("Super Micro Computer, Inc."), "Super Micro")
        self.assertEqual(connect.name_key("GLOBAL PAYMENTS INC"), "GLOBAL PAYMENTS")
        self.assertIsNone(connect.name_key("TARGET CORP"))                # too ordinary alone, and nothing follows
        self.assertIsNone(connect.name_key(None))

    def test_which_stocks_a_headline_names(self):
        def named(title):
            return sorted(connect.named(title, NAMES))
        self.assertEqual(named("Why CoreWeave Stock Is in the Nvidia Danger Zone"), ["CRWV", "NVDA"])
        self.assertEqual(named("HeartFlow (HTFL) Q3 2026 Earnings Call Transcript"), ["HTFL"])
        self.assertEqual(named("AMD Stocks Fall 6.46% After 30% Weekly Surge"), ["AMD"])
        self.assertEqual(named("Super Micro Slides 7%; Dell Falls 4%"), ["SMCI"])
        self.assertEqual(named("Super Bowl Ads Hit a Record as Global Demand Rises"), [])     # one ordinary word
        self.assertEqual(named("Retailers Miss Their Sales Target After a Virus Scare"), [])  # not Target, not Vir
        self.assertEqual(named("Vir Biotechnology Announces Pricing of Public Offering"), ["VIR"])
        self.assertEqual(named("FMC Guidance Cuts Due to Patent Expiration"), ["FMC"])
        self.assertEqual(named("The CEO Met the FDA as the SEC Looked On"), [])

    def test_stories_are_listed_once_and_a_round_up_is_a_wrap(self):
        def item(title, path, at="2026-11-13T15:00:00Z"):
            return {"title": title, "summary": "", "url": f"https://news.test/{path}?src=rss", "source": "news.test",
                    "at": at}
        both = item("Why CoreWeave Stock Is in the Nvidia Danger Zone", "a")
        feeds = {"NVDA": [both, item("Stock Market Today: Futures Fall", "b"), item("An old story on Nvidia", "c",
                                                                                 "2026-11-12T10:00:00Z")],
                 "CRWV": [{**both, "url": "https://news.test/a?src=other"}],
                 "SMCI": [item("Nvidia, AMD, Super Micro and More Stocks to Know", "d")]}
        got = connect.stories(feeds, "2026-11-12T21:00:00Z", NAMES)
        self.assertEqual(len(got), 3)                                     # the old story is before the close
        tie = next(s for s in got if s["url"].endswith("a?src=rss"))
        self.assertEqual((tie["feeds"], sorted(tie["names"]), tie["wrap"]), (["NVDA", "CRWV"], ["CRWV", "NVDA"], False))
        self.assertEqual(connect.story_links(tie, {"NVDA", "CRWV", "SMCI"}), ["CRWV", "NVDA"])
        self.assertTrue(next(s for s in got if "Market Today" in s["title"])["wrap"])       # titled like a wrap
        self.assertTrue(next(s for s in got if "More Stocks" in s["title"])["wrap"])        # names three stocks
        many = {t: [both] for t in ("A", "B", "C", "D", "E")}
        self.assertTrue(connect.stories(many, "2026-11-12T21:00:00Z", {})[0]["wrap"])       # carried by five feeds

    def test_clusters_join_what_any_chain_of_ties_links(self):
        got = connect.clusters(["A", "B", "C", "D", "E"], [("A", "B"), ("B", "C"), ("D", "Z")])
        self.assertEqual(got, [["A", "B", "C"]])                          # D's tie is to a non-mover; E has none
        self.assertEqual(connect.clusters(["A", "B"], []), [])


if __name__ == "__main__":
    unittest.main()
