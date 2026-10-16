"""F1's learner and range model (docs/FORECAST.md, sections 4, 5 and 11: Learn)."""
import json
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from brain.forecast import gbm, learn

SMALL = {**learn.LEARN, "gbm": {**learn.LEARN["gbm"], "trees": 25, "min_leaf": 40}}


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def feature_set(n_dates=60, per_date=80, seed=0, start="2012-01-02", h=5):
    """A stand-in FeatureSet: weekly dates, exits h+1 sessions on, a planted nonlinear rule in x0, x1."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_dates * 6 + 10).strftime("%Y-%m-%d").to_numpy()
    dates = days[: n_dates * 6: 6]
    exits = days[6: n_dates * 6 + 6: 6]
    S = n_dates * per_date
    X = rng.uniform(-0.5, 0.5, (S, 6))
    X[:, 5] = (rng.random(S) < 0.4).astype(float)
    z = 1.5 * np.sign(X[:, 0]) * np.sign(X[:, 1]) + 0.8 * X[:, 2]
    y = (rng.random(S) < gbm.sigmoid(z)).astype(np.int8)
    names = ["x0", "x1", "x2", "x3", "x4", "has_x"]
    return SimpleNamespace(
        X=X, y=y, excess=np.where(y == 1, 0.01, -0.01), r_total=rng.normal(0, 0.03, S),
        date=np.repeat(dates, per_date), exit=np.repeat(exits, per_date),
        ticker=np.tile([f"T{j}" for j in range(per_date)], n_dates), names=names,
        groups=["P", "P", "B", "F", "R", "F"], h=h)


def take(fs, mask):
    out = SimpleNamespace(**vars(fs))
    for k in ("X", "y", "excess", "r_total", "date", "exit", "ticker"):
        setattr(out, k, getattr(fs, k)[mask])
    return out


class GBMTest(unittest.TestCase):
    def test_boosting_fits_a_nonlinear_rule_better_than_logistic(self):
        rng = np.random.default_rng(1)
        n = 8000
        X = rng.uniform(-0.5, 0.5, (2 * n, 5))
        y = (rng.random(2 * n) < gbm.sigmoid(2.0 * np.sign(X[:, 0]) * np.sign(X[:, 1]))).astype(float)
        tr, te = slice(0, n), slice(n, None)
        g = gbm.fit(X[tr], y[tr], **learn.LEARN["gbm"])
        coef, b = learn._logistic(X[tr], y[tr], 1.0, 200)
        ll_gbm = logloss(y[te], gbm.predict(g, X[te]))
        ll_log = logloss(y[te], gbm.sigmoid(X[te] @ coef + b))
        self.assertLess(ll_gbm, ll_log - 0.03)          # XOR: no line separates it
        self.assertAlmostEqual(ll_log, np.log(2), delta=0.01)

    def test_histograms_by_parent_minus_sibling_match_direct_counts(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(3000, 4))
        edges = gbm.bin_edges(X, 16)
        XI = gbm.binned(X, edges).astype(np.intp)
        g, h = rng.normal(size=3000), rng.random(3000)
        cols = np.array([0, 2, 3])
        parent = gbm._hist(XI, cols, np.arange(3000), g, h, 16)
        left = np.flatnonzero(XI[1] <= 7)
        right = np.setdiff1d(np.arange(3000), left)
        sib = parent.minus(gbm._hist(XI, cols, left, g, h, 16))
        direct = gbm._hist(XI, cols, right, g, h, 16)
        np.testing.assert_allclose(sib.G, direct.G, atol=1e-10)
        np.testing.assert_allclose(sib.H, direct.H, atol=1e-10)
        np.testing.assert_array_equal(sib.C, direct.C)
        self.assertLessEqual(max(len(e) for e in edges), 15)  # at most bins - 1 quantile edges

    def test_min_leaf_and_depth_are_respected(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(size=(2000, 3))
        y = (X[:, 0] > 0.5).astype(float)
        m = gbm.fit(X, y, trees=5, depth=2, lr=0.1, bins=32, min_leaf=300, subsample=1.0, colsample=1.0, seed=7)
        for t in m["trees"]:
            self.assertLessEqual(len(t["feature"]), 7)   # a depth-2 tree has at most 7 nodes
        leaves = gbm._apply(m["trees"][0], 2, len(X), lambda f, node: X[np.arange(len(X)), f]
                            < np.asarray(m["trees"][0]["threshold"])[node])
        _, counts = np.unique(leaves, return_counts=True)
        self.assertGreaterEqual(counts.min(), 300)


class FitTest(unittest.TestCase):
    def test_fit_is_deterministic_and_round_trips_through_json(self):
        fs = feature_set()
        a, b = learn.fit(fs, "2013-12-31", SMALL), learn.fit(fs, "2013-12-31", SMALL)
        self.assertEqual(learn.to_json(a), learn.to_json(b))
        d = json.loads(json.dumps(learn.to_json(a)))           # plain JSON, no pickle
        back = learn.from_json(d)
        np.testing.assert_array_equal(learn.predict(back, fs.X), learn.predict(a, fs.X))
        self.assertEqual(back.sha, a.sha)
        p = learn.predict(a, fs.X)
        self.assertTrue(((p > 0) & (p < 1)).all())
        d["coef"][0] += 1.0
        with self.assertRaises(ValueError):                    # a tampered model is refused
            learn.from_json(d)

    def test_a_sample_whose_exit_is_after_the_cutoff_is_never_used(self):
        fs = feature_set()
        C = "2013-06-28"
        after = (pd.Timestamp(C) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")   # C + 1
        rng = np.random.default_rng(9)
        k = 400
        planted = SimpleNamespace(**vars(fs))
        # extreme rows dated before C, with exit C + 1: in the latest exits, where the calibration slice is drawn
        planted.X = np.vstack([fs.X, rng.uniform(5, 9, (k, fs.X.shape[1]))])
        planted.y = np.concatenate([fs.y, np.ones(k, dtype=np.int8)])
        planted.excess = np.concatenate([fs.excess, np.full(k, 0.5)])
        planted.r_total = np.concatenate([fs.r_total, np.full(k, 3.0)])
        planted.date = np.concatenate([fs.date, np.full(k, "2013-06-21")])
        planted.exit = np.concatenate([fs.exit, np.full(k, after)])
        planted.ticker = np.concatenate([fs.ticker, [f"P{j}" for j in range(k)]])
        sig = np.full(len(planted.y), 0.02)
        clean = take(fs, fs.exit <= C)
        self.assertEqual(learn.to_json(learn.fit(planted, C, SMALL)), learn.to_json(learn.fit(clean, C, SMALL)))
        self.assertEqual(learn.fit_range(planted, C, r_total=planted.r_total, sig=sig, h=5),
                         learn.fit_range(clean, C, r_total=clean.r_total, sig=sig[: len(clean.y)], h=5))
        # the same rows with exit = C are used: the cutoff is the stored exit, on or before
        planted.exit = np.concatenate([fs.exit, np.full(k, C)])
        self.assertNotEqual(learn.fit(planted, C, SMALL).sha, learn.fit(clean, C, SMALL).sha)
        self.assertNotEqual(learn.fit_range(planted, C, r_total=planted.r_total, sig=sig, h=5)["q"],
                            learn.fit_range(clean, C, r_total=clean.r_total, sig=sig[: len(clean.y)], h=5)["q"])

    def test_calibration_slice_is_the_latest_fifth_of_exits(self):
        ex = np.repeat(np.array(["2014-01-03", "2014-01-10", "2014-01-17", "2014-01-24", "2014-01-31"]), 10)
        cal = learn.calibration_split(ex)
        self.assertEqual(set(ex[cal]), {"2014-01-31"})        # 20% of rows exactly: one date
        ex2 = np.repeat(np.array(["2014-01-03", "2014-01-10", "2014-01-17"]), [50, 30, 15])
        self.assertEqual(set(ex2[learn.calibration_split(ex2)]), {"2014-01-17", "2014-01-10"})  # at least 20%
        self.assertFalse(learn.calibration_split(np.array(["2014-01-03"] * 9)).any())

    def test_draw_is_stratified_by_date_and_seeded(self):
        d = np.repeat(np.array(["a", "b", "c"]), [600, 300, 100])
        pick = learn.stratified_draw(d, 500)
        self.assertEqual(len(pick), 500)
        self.assertEqual(dict(zip(*np.unique(d[pick], return_counts=True))), {"a": 300, "b": 150, "c": 50})
        np.testing.assert_array_equal(pick, learn.stratified_draw(d, 500))
        np.testing.assert_array_equal(learn.stratified_draw(d, 5000), np.arange(1000))

    def test_max_rows_caps_the_training_rows(self):
        fs = feature_set()
        m = learn.fit(fs, "2013-12-31", {**SMALL, "max_rows": 1000})
        self.assertEqual(m.rows, 1000)
        self.assertEqual(m.trained_through, "2013-12-31")
        self.assertLessEqual(m.last_exit, "2013-12-31")

    def test_drivers_are_coefficient_times_value(self):
        fs = feature_set()
        m = learn.fit(fs, "2013-12-31", SMALL)
        x = np.array([0.4, -0.3, 0.5, 0.0, 0.1, 1.0])
        out = learn.drivers(m, x, fs.names, k=3)
        self.assertEqual(len(out), 3)
        pushes = {n: c * v for n, c, v in zip(fs.names, m.coef, x) if c * v != 0}
        best = sorted(pushes, key=lambda n: -abs(pushes[n]))[:3]
        self.assertEqual([d["name"] for d in out], best)
        for d in out:
            self.assertAlmostEqual(d["push"], pushes[d["name"]])
            self.assertEqual(d["group"], dict(zip(fs.names, fs.groups))[d["name"]])
        self.assertNotIn("x3", [d["name"] for d in learn.drivers(m, x, fs.names, k=6)])   # 0 pushes nothing

    def test_forecast_has_the_contract_shape(self):
        fs = feature_set()
        m = learn.fit(fs, "2013-12-31", SMALL)
        rng = {"q": {"q05": -1.7, "q16": -1.0, "q50": 0.0, "q84": 1.0, "q95": 1.7}}
        f = learn.forecast(m, rng, ticker="T1", d="2014-01-03", h=5, close=100.0, sigma_d=0.02, x_row=fs.X[0],
                           entry="2014-01-06", exit="2014-01-13")
        self.assertEqual(set(f), {"ticker", "date", "horizon", "entry", "exit", "close", "p_beat", "band", "move68_pct",
                                  "move90_pct", "drivers", "model"})
        self.assertEqual(set(f["band"]), {"q05", "q16", "q50", "q84", "q95"})
        self.assertAlmostEqual(f["band"]["q84"], 100 * np.exp(0.02 * np.sqrt(5)))
        self.assertEqual(set(f["model"]), {"id", "sha", "trained_through", "rows"})
        self.assertEqual(f["model"]["id"], "F1")
        json.dumps(f)


class RangeTest(unittest.TestCase):
    def test_sigma_is_the_ewma_of_past_returns_only(self):
        rng = np.random.default_rng(4)
        T = 400
        r = rng.normal(0, 0.02, (T, 2))
        adj = 100 * np.exp(np.cumsum(r, axis=0))
        px = SimpleNamespace(adj=adj, dates=np.arange(T), tickers=["A", "B"])
        i = 300
        rets = np.diff(np.log(adj[i - 252: i + 1, 0]))
        w = 0.94 ** np.arange(252)[::-1]
        self.assertAlmostEqual(learn.sigma(px, i)[0], np.sqrt((w * rets ** 2).sum() / w.sum()))
        later = adj.copy()
        later[i + 1:] *= 3.0                                  # what happens after d changes nothing at d
        np.testing.assert_array_equal(learn.sigma(px, i), learn.sigma(SimpleNamespace(adj=later), i))
        self.assertTrue(np.isnan(learn.sigma(px, 10)).all())  # fewer than 21 returns

    def test_bands_cover_as_aimed_on_a_known_volatility(self):
        rng = np.random.default_rng(5)
        T, N, h = 2600, 60, 5
        vol = rng.uniform(0.01, 0.03, N)
        lr = rng.normal(0, 1, (T, N)) * vol
        adj = 50 * np.exp(np.cumsum(lr, axis=0))
        days = pd.bdate_range("2009-01-01", periods=T).strftime("%Y-%m-%d").to_numpy()
        px = SimpleNamespace(adj=adj, dates=days, tickers=[f"S{j}" for j in range(N)])
        idx = np.arange(300, T - h - 2, h + 1)
        date = np.repeat(days[idx], N)
        exit_ = np.repeat(days[idx + 1 + h], N)
        fs = SimpleNamespace(date=date, exit=exit_, ticker=np.tile(px.tickers, len(idx)), h=h)
        r, sig = learn.range_inputs(fs, px)
        np.testing.assert_allclose(r, adj[idx + 1 + h].ravel() / adj[idx].ravel() - 1)   # close of d to the exit
        mid = days[idx[len(idx) // 2]]
        fit = learn.fit_range(fs, mid, px)
        self.assertEqual(fit, learn.fit_range(fs, mid, r_total=r, sig=sig))
        # d to exit is h + 1 sessions: the quantiles of z are the normal's times sqrt((h + 1) / h)
        self.assertAlmostEqual(fit["q"]["q84"], 0.9945 * np.sqrt(6 / 5), delta=0.05)
        later = date > mid
        z = np.log1p(r[later]) / (sig[later] * np.sqrt(h))
        cover68 = np.mean((z >= fit["q"]["q16"]) & (z <= fit["q"]["q84"]))
        cover90 = np.mean((z >= fit["q"]["q05"]) & (z <= fit["q"]["q95"]))
        self.assertAlmostEqual(cover68, 0.68, delta=0.02)
        self.assertAlmostEqual(cover90, 0.90, delta=0.015)
        b = learn.band(100.0, 0.02, 5, fit)
        self.assertLess(b["q05"], b["q16"])
        self.assertLess(b["q84"], b["q95"])
        self.assertAlmostEqual(b["q16"], 100 * np.exp(fit["q"]["q16"] * 0.02 * np.sqrt(5)))
        with self.assertRaises(ValueError):
            learn.fit_range(fs, mid)                      # needs px, or r_total and sig


if __name__ == "__main__":
    unittest.main()
