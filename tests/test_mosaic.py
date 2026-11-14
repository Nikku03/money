"""The mosaic (brain/mosaic.py) and its declared tests (brain/research/mosaictest.py), on made-up filings, accounts
and prices whose answers are known. Nothing touches the network."""
from __future__ import annotations

import unittest
from itertools import pairwise

import numpy as np
import pandas as pd

from brain import mosaic
from brain.research import mosaictest as mt
from brain.research import run
from tests.test_scan import snapshot
from tests.util import TempHome, filed


def days(n=900, start="2019-01-02"):
    return pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").to_numpy()


class KindsTest(unittest.TestCase):
    def test_a_filing_is_read_by_its_form_and_an_8k_by_its_items(self):
        self.assertEqual(mosaic.kinds("424B5", set()), ["prospectus"])        # what it sells is on its first page
        self.assertEqual(mosaic.kinds("424B3", set()), ["resale"])           # holders selling, not the company
        self.assertEqual(mosaic.kinds("424B2", set()), [])                   # banks' notes: not a share sale
        self.assertEqual(mosaic.kinds("S-3ASR", set()), ["shelf"])
        self.assertEqual(mosaic.kinds("S-8", set()), [])                     # an employee plan is not a shelf
        self.assertEqual(mosaic.kinds("NT 10-K", set()), ["nt"])
        self.assertEqual(mosaic.kinds("10-Q", set()), ["periodic"])
        self.assertEqual(mosaic.kinds("10-K/A", set()), ["amend"])
        self.assertEqual(mosaic.kinds("8-K", {"2.02", "9.01"}), ["results"])
        self.assertEqual(sorted(mosaic.kinds("8-K", {"4.01", "4.02", "5.02"})), ["auditor", "nonrel", "officer"])
        self.assertEqual(mosaic.kinds("8-K", {"2.05", "2.06"}), ["cut"])     # one filing, one cut
        self.assertEqual(mosaic.kinds("10-K", {"4.02"}), ["periodic"])       # items count on an 8-K only
        self.assertEqual(mosaic.kinds("4", set()), [])

    def test_a_filing_before_the_calendar_keeps_its_distance(self):
        cal = days()
        self.assertEqual(mosaic.position(cal, "2019-01-04T15:00:00Z"), 2)     # in session: that close
        self.assertEqual(mosaic.position(cal, "2019-01-04T22:00:00Z"), 3)     # after the close: the next one
        k = mosaic.position(cal, "2018-01-02T15:00:00Z")                      # a year before the first session
        self.assertTrue(-256 <= k <= -250, k)


class OfferingTest(unittest.TestCase):
    def test_the_first_page_says_what_a_prospectus_sells(self):
        kind = lambda text: mosaic.offering_kind(text)[0]
        self.assertEqual(kind("PROSPECTUS 7,142,857 Shares Common Stock We are offering 7,142,857 shares"), "shares")
        self.assertEqual(kind("Up to $150,000,000 Common Stock We may offer and sell shares from time to time"), "shares")
        self.assertEqual(kind("2,000,000 American Depositary Shares representing ordinary shares"), "shares")
        self.assertEqual(kind("£350,000,000 5.000% Notes due 2029 We are offering"), "debt")
        self.assertEqual(kind("$500,000,000 4.75% Convertible Senior Notes due 2030, convertible into common stock"),
                         "debt")                                              # named first: the notes
        self.assertEqual(kind("8,000,000 Depositary Shares each representing a 1/40th interest in a share of Series A "
                              "Preferred Stock"), "preferred")
        self.assertEqual(kind("10,000,000 Shares Common Stock The selling stockholders are offering 10,000,000 shares. "
                              "We are not selling any shares and we will not receive any proceeds"), "holders")
        self.assertEqual(kind("We are offering 5,000,000 shares of common stock and the selling stockholders 1,000,000. "
                              "We will not receive any proceeds from the shares sold by the selling stockholders"),
                         "shares")
        self.assertEqual(kind("25,000,000 Shares Common Stock We expect to enter into a forward sale agreement. We will "
                              "not receive any proceeds from the sale by the forward seller"), "shares")
        self.assertEqual(kind("Table of Contents"), "unknown")
        self.assertIn("7,142,857 Shares", mosaic.offering_kind("PROSPECTUS 7,142,857 Shares Common Stock")[1])


class CountingTest(unittest.TestCase):
    def test_counts_and_gaps_look_back_only_and_see_filings_before_the_calendar(self):
        c = mosaic.count_in(10, [2, 2, 5, -3], 4)
        self.assertEqual(c.tolist(), [1, 0, 2, 2, 2, 3, 1, 1, 1, 0])          # -3 is inside the window at 0 only
        s = mosaic.since(10, [2, 5], 100)
        self.assertEqual(s.tolist(), [100, 100, 0, 1, 2, 0, 1, 2, 3, 4])
        self.assertEqual(mosaic.since(4, [-5], 100).tolist(), [5, 6, 7, 8])
        self.assertEqual(mosaic.since(4, [-5], 6).tolist(), [5, 6, 6, 6])     # capped

    def test_a_deal_s_second_prospectus_does_not_open_a_new_deal(self):
        self.assertEqual(mosaic.fresh([50, 51, 58, 70, 200]), [50, 70, 200])  # 58 follows 51 within ten sessions
        self.assertEqual(mosaic.fresh([]), [])

    def test_a_report_is_overdue_from_three_sessions_after_last_year_s_until_it_arrives(self):
        late = mosaic.overdue(700, [100, 362])                               # last year's at 100: expected at 352
        self.assertEqual(late[354], 0)                                        # two sessions: inside the grace
        self.assertEqual(late[355], 3)
        self.assertEqual(late[361], 9)
        self.assertEqual(late[362], 0)                                        # it arrived
        on_time = mosaic.overdue(700, [100, 350])
        self.assertEqual(on_time[340:420].max(), 0)
        never = mosaic.overdue(700, [100])
        self.assertEqual(never[352 + 60], 60)
        self.assertEqual(never[352 + 61], 0)                                  # after sixty sessions the habit is gone
        self.assertEqual(mosaic.overdue(700, []).max(), 0)


class AccountsTest(unittest.TestCase):
    def test_the_latest_period_known_at_each_session_and_a_restatement_from_when_it_was_known(self):
        val, end = mosaic.step(10, [(2, 100, 5.0), (6, 190, 7.0), (4, 100, 6.0), (8, 50, 1.0)])
        self.assertTrue(np.isnan(val[:2]).all())
        self.assertEqual(val[2:].tolist(), [5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0])   # 6.0 restates; 1.0 is older
        self.assertEqual(end[9], 190)
        self.assertEqual(mosaic.step(3, [(-5, 10, 2.0)])[0].tolist(), [2.0, 2.0, 2.0])  # known before the calendar

    def test_a_year_to_date_flow_becomes_the_trailing_year_s_rate(self):
        raw = [(10, 1000, 365, -400.0),            # the full year to day 1000
               (20, 1090, 90, -10.0),              # this year's first quarter: one large receipt
               (5, 725, 90, -90.0)]                # last year's first quarter
        got = {round(e): r for _, e, r in mosaic.quarter_rates(raw)}
        self.assertAlmostEqual(got[1000], -400 * 91.25 / 365)
        self.assertAlmostEqual(got[1090], (-400 - 10 + 90) / 4)               # not -10 a quarter
        self.assertAlmostEqual(mosaic.quarter_rates([(20, 1090, 90, -10.0)])[0][2], -10 * 91.25 / 90)

    def test_the_runway_is_liquid_assets_over_the_burn_and_stale_accounts_are_not_read(self):
        d = np.arange(100, 500)
        acc = {"cash": [(0, 90, 50.0)], "cur_a": [(0, 90, 120.0)], "cur_l": [(0, 90, 20.0)],
               "op_cf": [(0, 90.09, -25.0)], "net": [(0, 90.09, -30.0)], "equity": [(0, 90, -5.0)]}
        m = mosaic.money(400, d, acc)
        self.assertEqual((m["acct"][0], m["burn"][0], m["loss"][0], m["negeq"][0]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["runway"][0], 4.0)                           # 120 - 20 = 100 liquid, 25 a quarter
        self.assertAlmostEqual(m["lrun"][0], np.log1p(4.0))
        self.assertEqual((m["acct"][399], m["burn"][399]), (0, 0))            # 409 days after the period: stale
        self.assertAlmostEqual(m["lrun"][399], np.log1p(mosaic.RUN_CAP))
        calm = mosaic.money(5, np.arange(100, 105), {**acc, "op_cf": [(0, 90.09, 25.0)]})
        self.assertEqual(calm["burn"][0], 0)
        self.assertAlmostEqual(calm["lrun"][0], np.log1p(mosaic.RUN_CAP))
        self.assertEqual(mosaic.money(5, np.arange(5), {})["acct"].tolist(), [0] * 5)


class PiecesTest(unittest.TestCase):
    def test_the_pieces_read_only_what_was_filed_by_each_close(self):
        mk = {"sale": [100, 101, 400], "shelf": [90], "effect": [95], "results": [98], "nt": [300],
              "nonrel": [305], "auditor": [250], "officer": [280, 285], "periodic": [50, 110]}
        p = mosaic.pieces(600, mk)
        self.assertEqual((p["sale_never"][99], p["sale10"][99], p["sales3y"][99]), (1, 0, 0))
        self.assertEqual((p["sale_never"][100], p["sale10"][100]), (0, 1))
        self.assertAlmostEqual(p["sales3y"][399], np.log1p(1))                # two prospectuses, one deal
        self.assertAlmostEqual(p["sales3y"][400], np.log1p(2))
        self.assertEqual((p["shelf60"][89], p["shelf60"][90], p["shelf60"][150], p["shelf3y"][150]), (0, 1, 0, 1))
        self.assertEqual((p["effect60"][95], p["earn5"][98], p["earn5"][102], p["earn5"][103]), (1, 1, 1, 0))
        self.assertEqual((p["nt250"][299], p["aud250"][299]), (0, 1))
        self.assertAlmostEqual(p["nt750"][300], np.log1p(1))
        self.assertAlmostEqual(p["nt_gap"][310], np.log1p(5))                 # the 4.02 at 305 counts too
        self.assertAlmostEqual(p["off60"][285], np.log1p(2))
        self.assertEqual(p["late10"][302 + 5], 5)                             # the report of 50 has no successor

    def test_the_outcome_is_a_deal_opening_in_the_next_twenty_sessions(self):
        n = 200
        adj, b = np.linspace(10, 20, n), np.full(n, 100.0)
        o = mosaic.outcomes(n, {"sale": [100, 101, 108], "nt": [150], "nonrel": [50]}, adj, b)
        self.assertEqual((o["y_sale"][79], o["y_sale"][80], o["y_sale"][99], o["y_sale"][100]), (0, 1, 1, 0))
        self.assertEqual(o["y_sale"][105], 0)                                 # 108 belongs to the deal of 100
        self.assertEqual((o["y_books"][30], o["y_books"][49], o["y_books"][50], o["y_books"][130]), (1, 1, 0, 1))
        self.assertTrue(np.isnan(o["y_sale"][-mosaic.H:]).all())              # those windows have not closed
        self.assertAlmostEqual(o["fwd"][10], adj[30] / adj[10] - 1)
        self.assertTrue(np.isnan(o["fwd"][-1]))


def bank_with(self, tickers, n=900):
    """A bank where each ticker has prices, and the first has a sale filing and burning accounts."""
    d = days(n)
    rng = np.random.default_rng(4)
    facts = [snapshot("SPY", d, 100 * np.cumprod(1 + rng.normal(0, 0.01, n))),
             snapshot("^VIX", d, np.full(n, 15.0))]
    for t in tickers:
        facts.append(snapshot(t, d, 20 * np.cumprod(1 + rng.normal(0, 0.03, n))))
    t0 = tickers[0]
    src = {"name": "test", "url": "https://example.test/f"}
    from brain.bank import Fact
    for k, (day, form, items) in enumerate(((d[700], "S-3", None), (d[720], "424B5", None),
                                            (d[760], "8-K", "2.02,9.01"), (d[765], "NT 10-Q", None),
                                            (d[600], "424B5", None), (d[650], "424B5", None))):
        facts.append(Fact(entity=t0, attr="filing", value=form, as_of=day, known_at=day + "T15:00:00Z", tier="filed",
                          source=src, context={"accn": f"a{k}", **({"items": items} if items else {})}))
    for accn, day, sells in (("a1", d[720], "shares"), ("a4", d[600], "debt")):   # a5 has not been read
        facts.append(Fact(entity=t0, attr=mosaic.OFFER, value=sells, as_of=day, known_at=day + "T15:00:00Z",
                          tier="filed", source=src, context={"accn": accn, "form": "424B5", "reader": mosaic.READER}))
    facts += [filed(t0, mosaic.CASH, 40e6, d[690], d[695] + "T23:59:59Z"),
              filed(t0, mosaic.OP_CF, -30e6, d[690], d[695] + "T23:59:59Z", duration_days=91, start=d[630])]
    self.bank.add_many(facts)
    return d


class TableTest(TempHome):
    def test_rows_are_weekly_point_in_time_and_carry_the_outcome(self):
        d = bank_with(self, ["AAA", "BBB"])
        tab = mosaic.table(self.bank, ["AAA", "BBB", "NONE"], d[400], d[880], log=None)
        self.assertEqual(sorted(set(tab["t"].tolist())), ["AAA", "BBB"])
        self.assertEqual(tab["X"].shape[1], len(mosaic.COLS))
        a = tab["t"] == "AAA"
        col = lambda k: tab["X"][a][:, mosaic.COLS.index(k)]
        dates = tab["date"][a]
        self.assertTrue((col("shelf3y")[dates < d[700]] == 0).all())           # not before it was filed
        self.assertTrue((col("shelf3y")[dates >= d[700]] == 1).all())
        self.assertTrue((col("burn")[dates > d[696]][:20] == 1).all())
        self.assertTrue((col("burn")[dates <= d[695]] == 0).all())
        y = tab["y_sale"][a]
        self.assertTrue((y[(dates >= d[700]) & (dates < d[720])] == 1).all())  # the sale of d[720] is ahead
        self.assertTrue((y[dates >= d[720]] == 0).all())
        self.assertTrue((y[dates < d[700]] == 0).all())                        # bonds, and a prospectus not read
        self.assertTrue((col("debt250")[(dates >= d[600]) & (dates < d[600 + 252])] == 1).all())
        self.assertTrue((col("sale_never")[dates < d[720]] == 1).all())
        self.assertEqual(len(mosaic.unread(self.bank.view(), "AAA")), 1)
        yb = tab["y_books"][a]
        self.assertTrue((yb[(dates >= d[745]) & (dates < d[765])] == 1).all())
        self.assertTrue((tab["y_sale"][tab["t"] == "BBB"] == 0).all())
        self.assertLessEqual(max(tab["date"]), d[899 - mosaic.H])              # later windows have not closed

    def test_accounts_are_read_again_only_after_a_new_report(self):
        d = bank_with(self, ["AAA", "BBB"])
        from brain.bank import Fact
        src = {"name": "test", "url": "https://example.test/f"}
        report = lambda t, day: Fact(entity=t, attr="filing", value="10-Q", as_of=day, known_at=day + "T21:00:00Z",
                                     tier="filed", source=src, context={"accn": "q" + day})
        self.assertFalse(mosaic.stale_accounts(self.bank.view(), "BBB"))       # it files no reports
        self.bank.add_many([report("AAA", d[695])])
        self.assertFalse(mosaic.stale_accounts(self.bank.view(), "AAA"))       # its accounts came with that report
        self.bank.add_many([report("AAA", d[760]), report("BBB", d[760])])
        self.assertTrue(mosaic.stale_accounts(self.bank.view(), "AAA"))        # a newer report than its accounts
        self.assertTrue(mosaic.stale_accounts(self.bank.view(), "BBB"))        # a report and no accounts at all

    def test_the_card_and_the_radar_speak_from_a_fitted_model(self):
        import os
        d = bank_with(self, ["AAA", "BBB", "CCC"])
        path = os.path.join(self._tmp.name, "mosaic.json")
        os.environ["BRAIN_MOSAIC_MODEL"] = path
        os.environ["BRAIN_LEDGER"] = os.path.join(self._tmp.name, "ledger.jsonl")
        try:
            from importlib import reload

            from brain import config
            reload(config)
            model = mosaic.fit_and_save(self.bank, tickers=["AAA", "BBB", "CCC"], log=None)
            self.assertEqual(model["train"], [mosaic.TRAIN_START, mosaic.TRAIN_END])
            rows = mosaic.radar(self.bank, ["AAA", "BBB"])
            self.assertEqual([r["t"] for r in rows], ["AAA", "BBB"])
            self.assertTrue(all(0 <= r["sale"] <= 1 and 0 <= r["books"] <= 1 for r in rows))
            text = "\n".join(mosaic.card(self.bank, "AAA"))
            self.assertIn("MOSAIC for AAA", text)
            self.assertIn("424B5, " + d[720], text)
            self.assertIn("1 prospectus not read yet", text)
            self.assertIn("NT 10-Q, " + d[765], text)
            self.assertIn("P1a not run yet", text)
            self.assertIn("Not offered: a takeover or a failure", text)
            self.assertIn("not enough price history", mosaic.card(self.bank, "ZZZ")[0])
            calls = os.path.join(self._tmp.name, "calls.jsonl")
            early = mosaic.radar(self.bank, ["AAA", "BBB"], asof=d[705])       # fifteen sessions before AAA's sale
            self.assertEqual(mosaic.record(early, calls), 6)                  # two stocks, three pictures
            self.assertEqual(mosaic.record(early, calls), 0)                  # a close is recorded once
            self.assertEqual(mosaic.record(rows, calls), 6)                   # the latest close: still open
            sc = mosaic.score(self.bank, calls)
            self.assertEqual((sc["sale"]["judged"], sc["sale"]["came"], sc["sale"]["open"]), (2, 1, 2))
            self.assertEqual((sc["books"]["came"], sc["cuts"]["came"]), (0, 0))
            self.assertIn("Under 30 judged calls this says nothing yet", mosaic.score_lines(sc)[0])
            before = mosaic.radar(self.bank, ["AAA"], asof=d[710])[0]         # ten sessions before the sale
            self.assertEqual((before["date"], before["sale_never"], before["shelf3y"]), (d[710], 1.0, 1.0))
            self.assertEqual(mosaic.radar(self.bank, ["AAA"], asof=d[725])[0]["sale_never"], 0.0)
            from brain import desk
            z = desk.mosaic_of(self.bank, ["AAA", "ZZZ"], d[710])
            self.assertEqual(list(z), ["AAA"])
            self.assertEqual(z["AAA"]["date"], d[710])
            self.assertEqual(sorted(z["AAA"]["pictures"]), ["books", "cuts", "sale"])
            line = desk._mosaic_line({"mosaic": z["AAA"]})
            self.assertIn("before this move", line)
            self.assertIn("not a direction", line)
            self.assertIsNone(desk._mosaic_line({"mosaic": None}))
        finally:
            os.environ.pop("BRAIN_MOSAIC_MODEL", None)
            os.environ.pop("BRAIN_LEDGER", None)
            reload(config)


def synthetic(seed=2, n_periods=30, per=400, signal=True):
    """Rows where the event depends on one piece (lrun) and flagged stocks lag when ``signal``."""
    rng = np.random.default_rng(seed)
    cal = days(n_periods * 25 + 60)
    X = np.zeros((n_periods * per, len(mosaic.COLS)), np.float32)
    lrun, lv = mosaic.COLS.index("lrun"), mosaic.COLS.index("lv60")
    date, y, fwd = [], [], []
    for k in range(n_periods):
        x = rng.normal(0, 1, per)
        X[k * per:(k + 1) * per, lrun] = x
        X[k * per:(k + 1) * per, lv] = rng.normal(-3.5, 0.4, per)
        p = 1 / (1 + np.exp(-(-3 + 1.5 * x)))
        y.append((rng.random(per) < p).astype(float))
        fwd.append(rng.normal(0, 0.1, per) - (0.05 * (x > 1.5) if signal else 0))
        date += [cal[k * 25]] * per
    tab = {"t": np.array([f"T{i % per}" for i in range(n_periods * per)]), "date": np.array(date), "X": X,
           "y_sale": np.concatenate(y), "y_books": np.concatenate(y), "fwd": np.concatenate(fwd)}
    feats = list(mosaic.SALE_FEATS)

    def part(w_lrun, w0):
        w = [w0] + [0.0] * len(feats)
        w[1 + feats.index("lrun")] = w_lrun
        return {"feats": feats, "lo": [-50.0] * len(feats), "hi": [50.0] * len(feats), "mu": [0.0] * len(feats),
                "sd": [1.0] * len(feats), "w": w}
    return cal, tab, {"model": part(1.5, -3.0), "baseline": part(0.0, float(np.log(0.09 / 0.91)))}


class TestsTest(unittest.TestCase):
    def test_periods_do_not_overlap(self):
        cal = days(300)
        weeks = cal[4::5]
        got = mt.periods(cal, weeks, 20)
        idx = [int(np.searchsorted(cal, d)) for d in got]
        self.assertEqual(idx[0], 4)
        self.assertTrue(all(b - a >= 20 for a, b in pairwise(idx)))
        self.assertTrue(all(b - a == 20 for a, b in pairwise(idx)))        # every fourth week here

    def test_a_model_that_knows_the_piece_beats_the_habit_and_a_model_that_knows_nothing_does_not(self):
        _, tab, part = synthetic()
        r = mt.evaluate_event(tab, part, "sale")
        self.assertTrue(r["gate"]["pass"], r["gate"])
        self.assertTrue(r["controls"]["pass"], r["controls"])
        self.assertGreater(r["metrics"]["capture"]["top5"], r["metrics"]["capture_base"]["top5"])
        self.assertGreater(r["metrics"]["auc"], 0.7)
        blind = mt.evaluate_event(tab, {"model": part["baseline"], "baseline": part["baseline"]}, "sale")
        self.assertFalse(blind["gate"]["pass"])
        hot = {"model": {**part["model"], "w": [x + (3.0 if i == 0 else 0) for i, x in enumerate(part["model"]["w"])]},
               "baseline": part["baseline"]}
        self.assertFalse(mt.evaluate_event(tab, hot, "sale")["gate"]["cal_ok"])     # ranks well, chances too high

    def test_flagged_stocks_that_lag_are_found_and_ones_that_do_not_are_not(self):
        _, tab, part = synthetic()
        evs = mt.flagged(tab, part)
        self.assertEqual(len(evs), 30 * 20)                                    # the top twentieth of 400, 30 periods
        r = mt.evaluate_lag(evs)
        self.assertTrue(r["gate"]["pass"], r["metrics"])
        self.assertTrue(r["controls"]["pass"], r["controls"])
        self.assertLess(r["metrics"]["abnormal"], -0.03)
        _, calm, _ = synthetic(signal=False)
        self.assertFalse(mt.evaluate_lag(mt.flagged(calm, part))["gate"]["pass"])
        self.assertEqual(mt.evaluate_lag([])["gate"]["pass"], False)

    def test_the_results_page_states_each_test_in_words(self):
        _, tab, part = synthetic()
        ev = {"op": "result", "kind": "mosaic", "id": "P1a", "gate_type": "event", "picture": "sale",
              **mt.evaluate_event(tab, part, "sale"), "bh_q": 0.01, "verdict": "PASS"}
        lg = {"op": "result", "kind": "mosaic", "id": "P1b", "gate_type": "lag", "picture": "sale",
              **mt.evaluate_lag(mt.flagged(tab, part)), "bh_q": 0.01, "verdict": "FAIL"}
        text = "\n".join(run._mosaic_section([ev, lg]))
        self.assertIn("## Mosaic", text)
        self.assertIn("| P1a | sale |", text)
        self.assertIn("**P1b** (a sale of new shares)", text)
        self.assertIn("not a trade", text)
        self.assertTrue(run.is_gated_mosaic(ev) and run.is_gated_mosaic(lg))
        self.assertEqual(run._mosaic_section([]), [])


if __name__ == "__main__":
    unittest.main()
