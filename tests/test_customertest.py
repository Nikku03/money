"""The customer-link tests (brain/research/customertest.py): the events, the statistics and the report."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from brain import customers as cu
from brain.research import customertest as ct
from brain.research import run
from brain.research.forecast_tests import Refused, paths
from tests.util import TempHome


def link(s, c, known="2023-11-15T21:00:00Z", until="2025-02-13", share=0.3):
    return {"s": s, "c": c, "share": share, "known_at": known, "until": until}


def report(t, d, r1, pre=None):
    return {"t": t, "date": d, "pre": pre or d[:8] + f"{int(d[8:]) - 1:02d}", "r1": r1}


class Events(unittest.TestCase):
    def test_season(self):
        self.assertEqual(ct.season("2025-02-10"), "2025Q1")
        self.assertEqual(ct.season("2025-12-31"), "2025Q4")

    def test_a_link_counts_from_its_filing_until_its_end(self):
        eps = {"C": [report("C", "2023-11-15", 0.01), report("C", "2024-02-10", 0.02), report("C", "2025-03-10", 0.03)]}
        rows = ct.events([link("S", "C")], eps, lambda *a, **k: 0.5, ("2023-01-01", "2026-12-31"))
        # the report on the filing's own day is before the link is known; the one after its end is past it
        self.assertEqual([(r["date"], r["c_r1"], r["same"], r["after"]) for r in rows], [("2024-02-10", 0.02, 0.5, 0.5)])
        self.assertEqual(ct.events([link("S", "C")], eps, lambda *a, **k: 0.5, ("2025-01-01", "2026-12-31")), [])

    def test_a_supplier_that_reports_nearby_is_left_out_of_the_moves(self):
        eps = {"C": [report("C", "2024-02-10", 0.02), report("C", "2024-05-10", -0.01)],
               "S": [report("S", "2024-02-13", 0.05)]}
        rows = ct.events([link("S", "C")], eps, lambda *a, **k: 0.5, ct.TEST)
        self.assertEqual([(r["date"], r["same"], r["after"], r["long"]) for r in rows],
                         [("2024-02-10", None, None, None), ("2024-05-10", 0.5, 0.5, 0.5)])

    def test_the_move_asks_for_the_same_session_and_the_sessions_after(self):
        asked = []

        def move(t, start, sessions=None, end=None):
            asked.append((t, start, sessions, end))
            return 0.0
        ct.events([link("S", "C")], {"C": [report("C", "2024-02-12", 0.02, pre="2024-02-09")]}, move, ct.TEST)
        self.assertEqual(asked, [("S", "2024-02-09", None, "2024-02-12"), ("S", "2024-02-12", ct.AFTER, None),
                                 ("S", "2024-02-12", ct.LONG, None)])

    def test_peers_are_of_the_suppliers_industry_and_do_not_name_the_customer(self):
        sic = {"S1": "367", "S2": "367", "P1": "367", "P2": "367", "P3": "367", "P4": "367", "C": "357", "Z": "283"}
        got = ct.peers_of([link("S1", "C"), link("S2", "C")], sic)
        # S2 names the same customer: it is not S1's peer; three peers a link, the first in ticker order
        self.assertEqual([x["s"] for x in got if x["c"] == "C"][:3], ["P1", "P2", "P3"])
        self.assertEqual(len(got), 6)
        self.assertEqual(ct.peers_of([link("Z", "C")], sic), [])

    def test_the_digest_follows_the_links(self):
        a = [link("S1", "C"), link("S2", "C")]
        self.assertEqual(ct.links_digest(a), ct.links_digest(list(reversed(a))))
        self.assertNotEqual(ct.links_digest(a), ct.links_digest(a[:1]))


def planted(seed=3, seasons=11, day=0.12, after=0.0):
    """Linked suppliers move with the customer on the day (``day`` of its move) and after it (``after``); peers
    do not."""
    rng = np.random.default_rng(seed)
    linked, peer = [], []
    for k in range(seasons):
        s = f"{2024 + k // 4}Q{k % 4 + 1}"
        for _ in range(60):
            c = float(rng.normal(0, 0.05))
            linked.append({"s": "S", "c": "C", "season": s, "c_r1": c, "same": day * c + float(rng.normal(0, 0.01)),
                           "after": after * c + float(rng.normal(0, 0.03)), "long": float(rng.normal(0, 0.06))})
            for _ in range(2):
                peer.append({"s": "P", "c": "C", "season": s, "c_r1": c, "same": float(rng.normal(0, 0.01)),
                             "after": float(rng.normal(0, 0.03)), "long": float(rng.normal(0, 0.06))})
    return linked, peer


class Statistics(unittest.TestCase):
    def test_a_link_that_is_real_on_the_day_and_says_nothing_after(self):
        linked, peer = planted()
        with mock.patch.object(ct, "N_FLIP", 2000), mock.patch.object(ct, "N_BOOT", 300):
            m = ct.evaluate(linked, peer, n_placebo=30)
        a, b = m["C1a"], m["C1b"]
        self.assertEqual((a["seasons"], a["events"], a["peer_events"]), (11, 660, 1320))
        self.assertGreater(a["linked"], 0.3)
        self.assertLess(abs(a["peers"]), 0.1)
        self.assertAlmostEqual(a["stat"], a["linked"] - a["peers"])
        self.assertLess(a["p"], 0.01)
        self.assertTrue(a["controls"]["pass"])
        self.assertGreater(a["fifths"]["best"]["same"], 0)
        self.assertLess(a["fifths"]["worst"]["same"], 0)
        self.assertGreater(b["p"], 0.05)
        self.assertIn("the link adds", ct.words("C1a", a))
        self.assertIn("it explains a move, it is not a forecast", ct.words("C1a", a))
        self.assertIn(f"next {ct.AFTER} sessions", ct.words("C1b", b))

    def test_what_the_peers_do_too_is_not_the_links(self):
        # suppliers and peers alike move with the customer: the link adds nothing
        linked, peer = planted(day=0.12)
        for r in peer:
            r["same"] = 0.12 * r["c_r1"] + (r["same"])
        with mock.patch.object(ct, "N_FLIP", 2000), mock.patch.object(ct, "N_BOOT", 300):
            a = ct.evaluate(linked, peer, n_placebo=10)["C1a"]
        self.assertGreater(a["linked"], 0.3)
        self.assertLess(abs(a["stat"]), 0.1)
        self.assertGreater(a["p"], 0.01)

    def test_a_season_with_too_few_events_is_left_out(self):
        linked, peer = planted(seasons=3)
        linked = [r for r in linked if r["season"] != "2024Q2"] + [r for r in linked if r["season"] == "2024Q2"][:5]
        with mock.patch.object(ct, "N_FLIP", 500), mock.patch.object(ct, "N_BOOT", 100):
            m = ct.evaluate(linked, peer, n_placebo=5)
        self.assertEqual((m["C1a"]["seasons"], m["C1b"]["seasons"]), (2, 2))


class TrialRun(TempHome):
    def test_declare_then_test_writes_the_ledger_and_the_report(self):
        rng = np.random.default_rng(8)
        customers_ = [f"C{k}" for k in range(5)]
        suppliers = [f"S{k:02d}" for k in range(30)]
        named = {s: customers_[k % 5] for k, s in enumerate(suppliers)}
        hist = []
        for s in suppliers:
            for known in ("2023-11-15T21:00:00Z", "2025-01-15T21:00:00Z"):
                hist.append({"t": s, "filed": known[:10], "known_at": known,
                             "named": [{"name": named[s], "ticker": named[s], "share": 0.3}]})
        eps, r1 = [], {}
        for k in range(11):
            y, q = 2024 + k // 4, k % 4
            for c in customers_:
                d = f"{y}-{q * 3 + 2:02d}-10"
                r1[(c, d)] = float(rng.normal(0, 0.05))
                eps.append({"t": c, "date": d, "pre": d[:8] + "09", "r1": r1[(c, d)]})
        sic = {**{s: "367" for s in suppliers}, **{f"P{k}": "367" for k in range(6)}}

        def move(t, start, sessions=None, end=None):
            if end is not None and t in named:                   # the same session: a supplier follows its customer
                return 0.15 * r1[(named[t], end)] + float(rng.normal(0, 0.01))
            return float(rng.normal(0, 0.02))
        names = sorted(set(suppliers) | set(customers_))
        with mock.patch.object(ct, "universe", lambda: names), mock.patch.object(ct, "N_PLACEBO", 20):
            with self.assertRaises(Refused):
                ct.test(self.bank, log=None, hist=hist, eps=eps, sic=sic, move=move)       # not declared yet
            self.assertEqual(len(ct.declare(log=None, hist=hist)), 2)
            self.assertEqual(ct.declare(log=None, hist=hist), [])                         # declared once
            with self.assertRaises(Refused):                                              # other links: refused
                ct.test(self.bank, log=None, hist=hist[:-1], eps=eps, sic=sic, move=move)
            out = ct.test(self.bank, log=None, hist=hist, eps=eps, sic=sic, move=move)
            self.assertEqual([r["id"] for r in out], list(ct.IDS))
            self.assertEqual(ct.test(self.bank, log=None, hist=hist, eps=eps, sic=sic, move=move), [])   # run once
        by = {r["id"]: r for r in out}
        self.assertTrue(by["C1a"]["gate"]["pass"])
        # the last report read is of 2025-01-15: its links hold 456 days, to 2026-04-15, so nine seasons of eleven
        self.assertEqual(by["C1a"]["metrics"]["seasons"], 9)
        led, res = paths(True)
        text = res.read_text()
        self.assertIn("## Customers: who it sells to", text)
        self.assertIn("| C1a |", text)
        self.assertIn("| C1b |", text)
        got = {r["id"]: r["verdict"] for r in run.verdicts(led) if r.get("kind") == "customer"}
        self.assertEqual(got["C1a"], "PASS")
        self.assertIn(got["C1b"], ("FAIL", "PASS", "FAIL (controls)"))
        self.assertEqual(len(cu.links(hist)), 60)


if __name__ == "__main__":
    unittest.main()
