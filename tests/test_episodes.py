"""Earnings episodes (brain/episodes.py), their declared tests (brain/research/earntest.py) and the card before a
report (brain/expect.py): the setup, the names, the analogs, the statistics and the forward record."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from brain import episodes as ep
from brain import expect
from brain.research import earntest as et


def series(n=400, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    b = np.cumprod(1 + rng.normal(0.0003, 0.01, n)) * 100
    a = b * np.cumprod(1 + rng.normal(drift, 0.015, n))
    v = np.full(n, 18.0)
    return a, b, v


class Names(unittest.TestCase):
    def test_why(self):
        self.assertEqual(ep.why(1.5, -0.05, 0.06), "priced in")
        self.assertEqual(ep.why(-1.5, 0.05, 0.06), "relief")
        self.assertEqual(ep.why(1.5, 0.05, 0.06), "confirmed")
        self.assertEqual(ep.why(-1.5, -0.05, 0.06), "capitulation")
        self.assertEqual(ep.why(0.2, 0.05, 0.06), "surprise up")
        self.assertEqual(ep.why(0.2, -0.05, 0.06), "surprise down")
        self.assertEqual(ep.why(1.5, 0.001, 0.06), "no reaction")    # under a quarter of its usual move


class Setup(unittest.TestCase):
    def test_needs_history(self):
        a, b, v = series()
        self.assertIsNone(ep.setup(a, b, v, 300, [(100, 0.05), (160, -0.03)]))   # two earlier reports only

    def test_only_closed_reports_count(self):
        a, b, v = series()
        prev = [(60, 0.05), (120, -0.03), (180, 0.04), (240, 0.06), (299, 0.10)]
        s = ep.setup(a, b, v, 299, prev)          # the report at 299 has not closed its window (299 + 1 > 299)
        self.assertIsNotNone(s)
        self.assertEqual(s["n_prev"], 4)
        self.assertAlmostEqual(s["med8"], float(np.median([0.05, 0.03, 0.04, 0.06])))
        self.assertAlmostEqual(s["up8"], 0.75)

    def test_run_is_in_usual_moves(self):
        a, b, v = series()
        a2 = a.copy()
        a2[290:] *= 1.10                          # a 10% jump against the market in the last ten sessions
        prev = [(60, 0.05), (120, -0.03), (180, 0.04), (240, 0.06)]
        s1, s2 = ep.setup(a, b, v, 299, prev), ep.setup(a2, b, v, 299, prev)
        self.assertGreater(s2["run21"], s1["run21"] + 1)


class Rows(unittest.TestCase):
    def test_episode_rows_point_in_time(self):
        a, b, v = series(500)
        dates = np.array([str(np.datetime64("2020-01-01") + k) for k in range(500)])
        firsts = [60, 120, 180, 240, 300, 360, 499]
        rows = ep.episode_rows(dates, a, b, v, firsts, ["x"] * len(firsts))
        last = rows[-1]
        self.assertIsNone(last["post5"])          # its week has not closed
        self.assertIsNotNone(last["r1"])
        self.assertIn("why", rows[-2])
        self.assertEqual(rows[-2]["pre"], str(dates[359]))
        self.assertAlmostEqual(rows[-2]["r1"], (a[360] / a[359] - 1) - (b[360] / b[359] - 1))


def library(n=3000, seed=1, edge=0.0):
    """A synthetic library: r1 = edge * (-run21) + noise, spread over months."""
    rng = np.random.default_rng(seed)
    out = []
    months = [f"{y}-{m:02d}" for y in range(2011, 2027) for m in range(1, 13)]
    for k in range(n):
        m = months[k % len(months)]
        if m > "2026-09":
            m = "2025-06"
        f = {x: float(rng.normal()) for x in ep.FEATS}
        f["med8"] = 0.06
        r1 = edge * -f["run21"] * 0.02 + float(rng.normal(0, 0.05))
        out.append({"t": f"T{k % 50}", "date": f"{m}-15", "pre": f"{m}-14", **f, "r1": r1,
                    "post5": float(rng.normal(0, 0.03)), "post21": float(rng.normal(0, 0.06)),
                    "why": ep.why(f["run21"], r1, 0.06)})
    return out


class Statistics(unittest.TestCase):
    def test_monthly_ic(self):
        month = np.array(["2024-01"] * 40 + ["2024-02"] * 40)
        x = np.arange(80, dtype=float)
        ms, d = et.monthly_ic(month, x, x)
        self.assertEqual(ms, ["2024-01", "2024-02"])
        self.assertTrue(np.allclose(d, 1.0))
        ms, d = et.monthly_ic(month[:50], x[:50], x[:50])      # the second month has only 10
        self.assertEqual(ms, ["2024-01"])

    def test_ic_test_finds_a_real_edge_and_controls_pass(self):
        rng = np.random.default_rng(0)
        month = np.repeat([f"2024-{m:02d}" for m in range(1, 13)] + [f"2025-{m:02d}" for m in range(1, 13)], 80)
        x = rng.normal(size=len(month))
        y = 0.3 * x + rng.normal(size=len(month))
        r = et.ic_test(month, x, y, rng, n_placebo=40)
        self.assertGreater(r["stat"], 0.1)
        self.assertLess(r["p"], 0.01)
        self.assertTrue(r["controls"]["pass"])
        r0 = et.ic_test(month, rng.normal(size=len(month)), y, rng, n_placebo=10)
        self.assertGreater(r0["p"], 0.01)

    def test_analogs_use_only_earlier_months(self):
        lib = ep.complete(library(1200, edge=1.0))
        train = [r for r in lib if r["date"] <= "2023-12-31"]
        test = [r for r in lib if "2024-01-01" <= r["date"] <= "2026-09-30"]
        pred = et.analog_predictions(train, lib, test, k=50)
        self.assertEqual(len(pred), len(test))
        self.assertTrue(np.isfinite(pred).all())
        # a test episode's prediction must not change when a same-month episode's outcome changes
        lib2 = [dict(r) for r in lib]
        first = test[0]
        for r in lib2:
            if r["date"][:7] == first["date"][:7] and r is not first:
                r["r1"] = 9.9
        test2 = [r for r in lib2 if "2024-01-01" <= r["date"] <= "2026-09-30"]
        pred2 = et.analog_predictions([r for r in lib2 if r["date"] <= "2023-12-31"], lib2, test2, k=50)
        self.assertAlmostEqual(pred[0], pred2[0])

    def test_by_why(self):
        rows = [{"why": "priced in", "r1": -0.05, "post5": -0.01, "post21": 0.0},
                {"why": "priced in", "r1": -0.03, "post5": 0.02, "post21": 0.01}]
        r = et.by_why(rows)["priced in"]
        self.assertEqual(r["n"], 2)
        self.assertAlmostEqual(r["kept_going"], 0.5)


class Evaluate(unittest.TestCase):
    def test_score_and_evaluate_on_synthetic_library(self):
        lib = library(4000, edge=1.0)
        train = [r for r in lib if r["date"] <= "2023-12-31"]
        sc = et.score_scale(train)
        lo = et.expectation_score({"run21": 3.0, "run5": 3.0, "up8": 0.0}, sc)
        hi = et.expectation_score({"run21": -3.0, "run5": -3.0, "up8": 1.0}, sc)
        self.assertGreater(hi, lo)
        old = (et.N_PLACEBO, et.N_FLIP, et.N_BOOT)
        et.N_PLACEBO, et.N_FLIP, et.N_BOOT = 5, 500, 200
        try:
            res = et.evaluate(lib)
        finally:
            et.N_PLACEBO, et.N_FLIP, et.N_BOOT = old
        self.assertEqual(set(res), set(et.IDS))
        self.assertGreater(res["X1a"]["stat"], 0)          # the synthetic edge is -run21
        self.assertIn("priced in", res["X1e"]["by_why"])
        for tid in ("X1a", "X1b", "X1c", "X1d", "X1f"):
            self.assertIn("controls", res[tid])
            self.assertTrue(et.words(tid, res[tid]))
        self.assertTrue(et.words("X1e", res["X1e"]))


class Card(unittest.TestCase):
    def test_analogs_and_record(self):
        lib = library(2000)
        s = {f: 0.0 for f in ep.FEATS}
        s.update({"run21": 2.0, "med8": 0.06, "asof": "2026-11-24"})
        a = expect.analogs(s, lib, k=100)
        self.assertEqual(a["n"], 100)
        self.assertTrue(0 <= a["rose"] <= 1)
        self.assertIn("priced in", a["why"].keys() | {"priced in"})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calls.jsonl"
            self.assertEqual(expect.read_calls(p), [])
            p.write_text('{"id": "X-2026-12-01", "t": "X", "report": "2026-12-01", "asof": "2026-11-30", '
                         '"lean": 0.01}\n')
            j = expect.judge([{"t": "X", "pre": "2026-11-30", "r1": 0.02, "why": "surprise up"}], expect.read_calls(p))
            self.assertAlmostEqual(j[0]["r1"], 0.02)
            s2 = expect.score([{"t": "X", "pre": "2026-11-30", "r1": 0.02, "why": "surprise up"}], p)
            self.assertEqual(s2["judged"], 1)
            self.assertEqual(s2["sign_hit"], 1.0)


if __name__ == "__main__":
    unittest.main()
