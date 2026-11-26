"""Who a company sells to (brain/customers.py): the customer sentences of an annual report, the shares they state,
the listed companies they name, and who names whom. The sentences are worded as real reports word them."""
from __future__ import annotations

import unittest

from brain import customers as cu
from tests.util import TempHome

TICKERS = [
    {"ticker": "AAPL", "title": "Apple Inc."}, {"ticker": "MSFT", "title": "MICROSOFT CORP"},
    {"ticker": "WMT", "title": "Walmart Inc."}, {"ticker": "CVS", "title": "CVS HEALTH Corp"},
    {"ticker": "TM", "title": "TOYOTA MOTOR CORP/"}, {"ticker": "GM", "title": "General Motors Co"},
    {"ticker": "CSCO", "title": "CISCO SYSTEMS, INC."}, {"ticker": "NVDA", "title": "NVIDIA CORP"},
    {"ticker": "NOK", "title": "NOKIA CORP"}, {"ticker": "AMZN", "title": "AMAZON COM INC"},
    {"ticker": "ORCL", "title": "ORACLE CORP"}, {"ticker": "ISRG", "title": "INTUITIVE SURGICAL INC"},
    {"ticker": "SYK", "title": "STRYKER CORP"}, {"ticker": "TGT", "title": "TARGET CORP"},
    {"ticker": "DOW", "title": "DOW INC."}, {"ticker": "CAH", "title": "CARDINAL HEALTH INC"},
    {"ticker": "COST", "title": "COSTCO WHOLESALE CORP /NEW"}, {"ticker": "NDAQ", "title": "NASDAQ, INC."},
    {"ticker": "DIS", "title": "Walt Disney Co"}, {"ticker": "RY", "title": "ROYAL BANK OF CANADA"},
    {"ticker": "GEF", "title": "GREIF, INC."}, {"ticker": "POOL", "title": "POOL CORP"},
    {"ticker": "PTC", "title": "PTC Inc."}, {"ticker": "DUK", "title": "Duke Energy CORP"},
    {"ticker": "ULCC", "title": "Frontier Group Holdings, Inc.", "cik": 1}, {"ticker": "KR", "title": "KROGER CO"},
    {"ticker": "FYBR", "title": "Frontier Communications Parent, Inc.", "cik": 2},
]
TABLE = cu.name_table(TICKERS)


def read(text, own=None):
    return cu.read(text, TABLE, own)


def named(text, own=None):
    return [(x["ticker"] or x["name"], x["share"]) for x in read(text, own)["named"]]


class Names(unittest.TestCase):
    def test_a_name_is_its_distinguishing_words(self):
        self.assertEqual(cu.core("COSTCO WHOLESALE CORP /NEW"), ("COSTCO", "WHOLESALE"))
        self.assertEqual(cu.core("Lowe's Companies, Inc."), ("LOWES",))
        self.assertEqual(cu.core("The Home Depot, Inc."), ("HOME", "DEPOT"))
        self.assertEqual(TABLE[("CVS", "HEALTH")], "CVS")
        self.assertEqual(TABLE[("AMAZON",)], "AMZN")            # a short name a report uses
        self.assertIsNone(TABLE[("SAMSUNG",)])                  # named, not listed in the United States

    def test_the_longest_name_wins_and_the_company_is_not_its_own_customer(self):
        s = "Cardinal Health | Business Customers Our largest customer, CVS Health, accounted for 28 percent of revenue."
        self.assertEqual([(x[0], x[1]) for x in cu.names_in(s, TABLE, own="CAH")], [("CVS Health", "CVS")])
        self.assertEqual(len(cu.names_in(s, TABLE)), 2)

    def test_a_word_that_is_also_a_company(self):
        # "Target" as a goal, "DoW" as the Department of War: not Target Corporation, not Dow Inc.
        self.assertEqual(cu.names_in("Revenue (weighted 33%) at Target was the measure used.", TABLE), [])
        self.assertEqual(cu.names_in("including 63% from the Department of War (DoW), sales were up", TABLE), [])
        got = cu.names_in("Sales to Walmart and Target accounted for 22% and 11% of net sales.", TABLE)
        self.assertEqual([x[1] for x in got], ["WMT", "TGT"])
        self.assertEqual([x[1] for x in cu.names_in("sales to Target Corporation were 12%", TABLE)], ["TGT"])


class Roles(unittest.TestCase):
    """A known name counts only where a customer stands."""

    def got(self, s, own=None):
        return [(x[1] or x[0]) for x in cu.customers_in(s, TABLE, own)]

    def test_where_a_customer_stands(self):
        self.assertEqual(self.got("Sales to AT&T were $500.7 million, or 10.5% of total revenue, in fiscal 2025."), ["T"])
        self.assertEqual(self.got("In fiscal 2026, Walgreens accounted for approximately 23% of revenue."), ["Walgreens"])
        self.assertEqual(self.got("our top four customers by total revenue were T-Mobile (18%), AT&T (17%), Verizon "
                                  "Wireless (14%) and Telefonica (10%)."), ["TMUS", "T", "VZ"])
        self.assertEqual(self.got("Our product sales to three large wholesalers, McKesson Corporation, Cencora, Inc. "
                                  "and Cardinal Health, Inc., each individually accounted for more than 10% of total "
                                  "revenues."), ["MCK", "COR", "CAH"])
        self.assertEqual(self.got("In fiscal 2026, revenues from Apple and Samsung each comprised 10% or more of our "
                                  "consolidated revenues."), ["AAPL", "Samsung"])
        self.assertEqual(self.got("Percentage of consolidated net sales to customers: McKesson Corporation * * 10 % "
                                  "Cencora, Inc. 11 % 12 % 10 %"), ["MCK", "COR"])

    def test_where_it_does_not(self):
        for s in ("the last reported closing sales price of our common stock on The NASDAQ Global Select Market was "
                  "$20.16 per share, which represented a premium of approximately 1.10%",
                  "During 2025, sales of Microsoft products accounted for approximately 17% of our consolidated net "
                  "sales to customers.",
                  "Apple, Inc. 12 % 12 % 11 % (1) Revenue generated from products purchased from this vendor was less "
                  "than 10% of consolidated revenue.",
                  "One customer in the Pool business represented approximately 18% of our consolidated net sales.",
                  "Revenue increases of 53 % are net of PTC flow backs to customers.",
                  "For 2025, Greif accounted for approximately 18% of the Company’s consolidated total assets and "
                  "approximately 4% of the Company’s consolidated net sales.",
                  "condominium units adjacent to the Four Seasons Resort Orlando at Walt Disney World Resort, which "
                  "represented 2% of total revenues from customers in 2025.",
                  "71% of our dealership revenue was generated from premium brands, such as Toyota and Honda, sold "
                  "to customers."):
            self.assertEqual(self.got(s), [], s)

    def test_two_companies_with_one_name_are_no_ones_name(self):
        self.assertNotIn(("FRONTIER",), TABLE)
        self.assertEqual(self.got("Sales to customers: Frontier 9.0% 8.5% 10.7%"), [])

    def test_a_written_name_ends_at_its_last_word(self):
        got = cu.customers_in("The Company’s largest customers, Walmart Inc. and The Kroger Co., accounted for "
                              "approximately 36% of volume and approximately 29% of the Company’s total net sales.", TABLE)
        self.assertEqual([(x[0], x[1]) for x in got], [("Walmart Inc", "WMT"), ("Kroger Co", "KR")])


class Shares(unittest.TestCase):
    def test_the_reporting_threshold_is_not_a_share(self):
        self.assertEqual(cu.shares("two customers that accounted for more than 10% of our revenue in 2025"), [])
        self.assertEqual(cu.shares("the customer representing 10% or more of revenue was WT, which accounted for "
                                   "approximately 70% of total revenue"), [0.70])
        self.assertEqual(cu.shares("Customer A 20 % 16 % * Amount was less than 10% of total."), [0.20, 0.16])

    def test_receivables_are_left_out_of_a_sentence(self):
        s = ("For fiscal 2026, sales to two of our customers represented 24.7% of our total net sales, and as of "
             "April 30, 2026, these two customers accounted for 30.1% of our accounts receivable.")
        self.assertEqual(cu.shares(cu.revenue_part(s)), [0.247])

    def test_how_many_customers(self):
        self.assertEqual(cu.how_many("sales to one direct customer represented 22% of total revenue and sales to "
                                     "another direct customer represented 14% of total revenue"), 2)
        self.assertEqual(cu.how_many("For fiscal year 2025, sales to four customers represented 20.9 %, 11.5 %, "
                                     "11.3 %, and 11.1 % of total net sales."), 4)
        self.assertEqual(cu.how_many("we had four, two and two customers, respectively, that each contributed 10% or "
                                     "more of our revenues in fiscal 2026, 2025 and 2024"), 4)
        self.assertEqual(cu.how_many("there were two customers (one distributor and one direct customer) whose "
                                     "revenues represented 10% or more of total net revenue"), 2)
        self.assertEqual(cu.how_many("Our largest customer accounted for 19% of sales."), 1)


class Reading(unittest.TestCase):
    def test_one_named_customer_takes_the_latest_year(self):
        r = read("Filler sentence here. During fiscal 2026, fiscal 2025, and fiscal 2024, Apple, through sales to "
                 "multiple distributors, in the aggregate accounted for 61 %, 67 %, and 69 % of the Company’s net "
                 "revenue, respectively. More text follows here.")
        self.assertEqual((r["named"][0]["ticker"], r["named"][0]["share"], r["top"], r["n_major"]),
                         ("AAPL", 0.61, 0.61, 1))
        # the years running upwards: the last percentage is the latest
        self.assertEqual(named("In 2024 and 2025, Walmart Inc. accounted for 18.4% and 19.0% of our net sales, "
                               "respectively."), [("WMT", 0.19)])

    def test_several_named_customers_take_their_own_shares(self):
        self.assertEqual(named("Net sales from our largest two customers, Intuitive Surgical SARL and Stryker "
                               "Corporation, were 24.3% and 21.5%, respectively, of our total net sales for 2025."),
                         [("ISRG", 0.243), ("SYK", 0.215)])
        self.assertEqual(named("During fiscal year 2026, Cisco Systems, Inc., NVIDIA Corporation, Nokia Corporation "
                               "and Amazon.com, Inc. contributed 19.9%, 16.3%, 10.7% and 10.5%, respectively, of our "
                               "revenues."), [("CSCO", 0.199), ("NVDA", 0.163), ("NOK", 0.107), ("AMZN", 0.105)])

    def test_a_list_with_an_unknown_name_is_not_lined_up(self):
        # Digicomm is not a known company: which percentage is Microsoft's cannot be told from this sentence
        self.assertEqual(named("In 2025, Digicomm and Microsoft represented 53.1 % and 28.8 % of its revenue, "
                               "respectively, as customers."), [("MSFT", None)])
        self.assertEqual(named("In 2024, Microsoft, Digicomm and Oracle represented 43.7 %, 35.1 % and 12.4 % of "
                               "revenue from customers."), [("MSFT", None), ("ORCL", None)])
        # each with its own share after it can be told
        self.assertEqual(named("In 2025, Digicomm represented 53.1% of our revenue and Microsoft represented 28.8% "
                               "of our revenue."), [("MSFT", 0.288)])

    def test_a_share_before_the_name(self):
        self.assertEqual(named("In 2025 and 2024, approximately 39% and 40% of our total consolidated sales, "
                               "respectively, were to Airbus, and its related subcontractors and approximately 13% and "
                               "15% of our total consolidated sales, respectively, were to Boeing and its related "
                               "subcontractors, our customers."), [("Airbus", 0.39), ("BA", 0.13)])
        self.assertEqual(named("Sales to these customers represented approximately 71% of total revenue, with 33% sold "
                               "to Canadian Imperial Bank of Commerce, 13% sold to the Royal Bank of Canada."),
                         [("RY", 0.13)])

    def test_a_share_of_a_group_a_product_or_a_subsidiary_is_no_ones_own(self):
        self.assertEqual(named("Collectively, our net sales to these four customers, including Apple, were 31 percent "
                               "of our consolidated net sales in 2025."), [("AAPL", None)])
        self.assertEqual(named("The Company’s largest customers, Walmart Inc. and The Kroger Co., accounted for "
                               "approximately 36% of volume and approximately 29% of the Company’s total net sales."),
                         [("WMT", None), ("KR", None)])
        self.assertEqual(read("Product sales during 2025 to McKesson Corporation, Cardinal Health and Cencora "
                              "represented approximately 48%, 22% and 24%, respectively, of total ARISTADA and "
                              "ARISTADA INITIO gross sales to customers.")["sentences"], [])
        self.assertEqual(read("Transco’s largest customer was Duke Energy Corporation, which accounted for "
                              "approximately 9 percent of its operating revenue.")["sentences"], [])

    def test_a_table_after_a_colon(self):
        r = read("In 2025, the Company had three automotive customers, which individually accounted for 10% or more "
                 "of consolidated net sales as follows: Toyota Motor Company Volkswagen Group General Motors 2025 "
                 "18 % 11 % 10 % 2024 19 % 13 % 11 % 2023 18 % 14 % 10 % (8) COMPREHENSIVE INCOME")
        self.assertEqual([(x["ticker"] or x["name"], x["share"]) for x in r["named"]],
                         [("TM", 0.18), ("Volkswagen Group", 0.11), ("GM", 0.10)])
        self.assertEqual((r["top"], r["n_major"]), (0.18, 3))
        r = read("Customers whose revenues represented 10% or more of total net revenue: Customer A 14% 13% * "
                 "Distributor A 37% 34% 24% *Less than 10% of net revenue.")
        self.assertEqual((r["top"], r["n_major"], r["named"]), (0.37, 2, []))

    def test_a_table_of_names_and_a_star(self):
        # no customer word in the row itself: two known names, each with its percentages, are a customer table
        self.assertEqual(named("2025 2024 2023 Cardinal Health, Inc. (“Cardinal Health”) 29 % 29 % 28 % Cencora, Inc. "
                               "(“Cencora”) 21 % 21 % 22 % McKesson Corporation (“McKesson”) 24 % 23 % 24 % 65 "
                               "Revenues Recognized were as shown."), [("CAH", 0.29), ("MCK", 0.24), ("COR", 0.21)])
        # a star: below a tenth in the latest year, so not this year's customer
        self.assertEqual(named("Percentage of consolidated net sales to customers: McKesson Corporation * * 10 % "
                               "Cencora, Inc. 11 % 12 % 10 % Cardinal Health, Inc. * * 5 %"), [("COR", 0.11)])

    def test_a_report_that_says_no_customer_brings_a_tenth(self):
        r = read("Total Revenue was up. No sales to an individual customer or country other than the United States "
                 "accounted for more than 10% of revenue for fiscal years 2026, 2025, or 2024. More text here.")
        self.assertIn("No sales to an individual customer", r["none"])
        self.assertEqual((r["top"], r["named"]), (None, []))
        # "no other customer" means there is one: not this
        r = read("Our largest customer, Walmart Inc., accounted for 19% of net sales. No other customer accounted "
                 "for 10 percent or more of our consolidated net sales.")
        self.assertIsNone(r["none"])
        self.assertEqual(r["top"], 0.19)

    def test_unnamed_customers_and_the_top_ten(self):
        r = read("For the fiscal year ended June 30, 2026, sales to one customer represented 28.1 % of total net "
                 "sales. In addition, net revenue from our ten (10) largest customers represented 82% of our total "
                 "net revenue for fiscal 2026. Our five largest customers accounted for approximately 37% of our net "
                 "revenue.")
        self.assertEqual((r["top"], r["n_major"], r["top5"], r["top10"], r["named"]), (0.281, 1, 0.37, 0.82, []))

    def test_a_name_without_the_word_customer_needs_its_share_after_it(self):
        self.assertEqual(named("During fiscal 2026, Walmart Inc. and its affiliates (Walmart) accounted for 22 percent "
                               "of our consolidated net sales."), [("WMT", 0.22)])
        self.assertEqual(named("Net sales 54 % 57 % 30 % 66 % (a) Includes Walmart Inc. and its affiliates."), [])

    def test_what_is_not_a_customer_share(self):
        for s in ("Sales to customers in Asia accounted for approximately 88% of our total revenue in fiscal 2026.",
                  "No other single customer or customer group represented more than 10% of fiscal 2026 sales.",
                  "The five largest customers of Assembly Components accounted for approximately 53% of segment "
                  "sales for 2025.",
                  "Three direct customers accounted for 25 %, 18 %, and 13 % of our accounts receivable balance.",
                  "Revenue grew on a 8% increase in revenues from existing customers within our networking business.",
                  "For 2025, approximately 61% of Assembly Components’ net sales were to domestic customers."):
            r = read(s)
            self.assertEqual((r["top"], r["top5"], r["named"], r["sentences"]), (None, None, [], []), s)

    def test_the_government(self):
        r = read("In 2025, 72% of our $75.0 billion in sales were from the U.S. Government, either as a prime "
                 "contractor or as a subcontractor (including 63% from the Department of War (DoW)) and 28% were "
                 "from international customers.")
        self.assertEqual((r["government"], r["top"], r["named"]), (0.72, 0.72, []))


class Library(TempHome):
    def test_who_names_whom_and_the_card(self):
        lib = [{"t": "SWKS", "filed": "2026-11-10", "year_end": "2026-10-02", "url": "https://example.test/swks",
                "top": 0.61, "n_major": 1, "top5": None, "top10": None, "government": None,
                "named": [{"name": "Apple", "ticker": "AAPL", "share": 0.61}],
                "sentences": ["Apple accounted for 61 % of the Company’s net revenue."]},
               {"t": "CRUS", "filed": "2026-05-21", "year_end": "2026-03-28", "url": "https://example.test/crus",
                "top": 0.91, "n_major": 1, "top5": None, "top10": 0.96, "government": None,
                "named": [{"name": "Apple Inc", "ticker": "AAPL", "share": 0.91}], "sentences": []}]
        self.assertEqual([(r["t"], r["share"]) for r in cu.named_by(lib, "aapl")], [("CRUS", 0.91), ("SWKS", 0.61)])
        card = "\n".join(cu.render("SWKS", lib))
        self.assertIn("named: Apple (AAPL) 61% of revenue", card)
        self.assertIn("its largest stated customer share is 61%; 1 customer of a tenth or more", card)
        self.assertIn('"Apple accounted for 61 % of the Company’s net revenue."', card)
        self.assertIn("source: https://example.test/swks", card)
        card = "\n".join(cu.render("AAPL", lib))
        self.assertIn("its annual report has not been read", card)
        self.assertIn("named as a customer by: CRUS (91% of its revenue); SWKS (61% of its revenue)", card)
        self.assertIn("not tested", card)
        lib.append({"t": "MSFT", "filed": "2026-07-30", "year_end": "2026-06-30", "url": "https://example.test/msft",
                    "top": None, "n_major": None, "top5": None, "top10": None, "government": None, "named": [],
                    "none": "No sales to an individual customer accounted for more than 10% of revenue.",
                    "sentences": []})
        self.assertIn("its report says no customer brings a tenth of its revenue", "\n".join(cu.render("MSFT", lib)))

    def test_links_are_known_from_the_filing_until_the_next_report(self):
        def row(t, known, named):
            return {"t": t, "filed": known[:10], "known_at": known, "named": named}
        hist = [row("SWKS", "2024-11-15T21:00:00Z", [{"name": "Apple", "ticker": "AAPL", "share": 0.69}]),
                row("SWKS", "2025-11-14T21:00:00Z", [{"name": "Apple", "ticker": "AAPL", "share": 0.67},
                                                     {"name": "Samsung", "ticker": None, "share": 0.1}]),
                row("OLD", "2015-02-20T21:00:00Z", [{"name": "Walmart", "ticker": "WMT", "share": None}])]
        got = cu.links(hist)
        self.assertEqual([(x["s"], x["c"], x["share"], x["known_at"][:10], x["until"][:10]) for x in got],
                         [("SWKS", "AAPL", 0.69, "2024-11-15", "2025-11-14"),
                          ("SWKS", "AAPL", 0.67, "2025-11-14", "2027-02-13"),      # 456 days after its last report
                          ("OLD", "WMT", None, "2015-02-20", "2016-05-21")])

    def test_the_desks_line(self):
        lib = [{"t": "SWKS", "filed": "2026-11-10", "top": 0.61, "n_major": 1,
                "named": [{"name": "Apple", "ticker": "AAPL", "share": 0.61}]},
               {"t": "SMCI", "filed": "2026-08-31", "top": 0.281, "n_major": 1, "named": []},
               {"t": "MSFT", "filed": "2026-07-29", "top": None, "n_major": None, "named": []}]
        moves = {"AAPL": -0.032, "SWKS": -0.051}
        self.assertEqual(cu.line(lib, "swks", moves.get),
                         "sells to Apple (AAPL, -3.2% since the close) 61% of revenue (annual report filed "
                         "2026-11-10). Read from annual reports; a read-through is untested")
        self.assertEqual(cu.line(lib, "AAPL", moves.get),
                         "named as a customer by SWKS (61% of its revenue, -5.1% since the close). Read from annual "
                         "reports; a read-through is untested")
        self.assertIn("its largest customer, unnamed, brings 28% of revenue", cu.line(lib, "SMCI"))
        self.assertIsNone(cu.line(lib, "MSFT"))
        self.assertIsNone(cu.line(lib, "ZZZZ"))

    def test_the_library_keeps_the_latest_reading(self):
        from brain.bank import Fact
        for known, top in (("2025-11-10", 0.67), ("2026-11-10", 0.61)):
            self.bank.add(Fact(entity="SWKS", attr=cu.ATTR, value=top, unit="ratio", as_of=known, known_at=known,
                               tier="filed", source={"name": "SEC EDGAR 10-K", "url": "https://example.test/" + known},
                               context={"accn": known, "reader": cu.READER, "top": top, "n_major": 1, "top5": None,
                                        "top10": None, "government": None, "named": [], "sentences": []}))
        rows = cu.build(self.bank, ["swks", "none"])
        self.assertEqual([(r["t"], r["top"], r["filed"]) for r in rows], [("SWKS", 0.61, "2026-11-10")])
        every = cu.build(self.bank, ["swks"], history=True)
        self.assertEqual([(r["top"], r["filed"]) for r in every], [(0.67, "2025-11-10"), (0.61, "2026-11-10")])
        p = cu.save(rows, self.bank.home / "c.json.gz") if hasattr(self.bank, "home") else None
        if p:
            self.assertEqual(cu.load(p), rows)


if __name__ == "__main__":
    unittest.main()
