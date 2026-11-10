"""The market scan (brain/scan.py) and its declared test (brain/research/rangetest.py) on made-up prices whose
sizes are known. Nothing touches the network."""
import json
import os

import numpy as np
import pandas as pd

from brain import scan, series, shocks
from brain.bank import Fact
from brain.research import rangetest
from tests.util import TempHome, filed


def snapshot(entity, dates, close, volume=1e6):
    vol = np.full(len(dates), float(volume)) if np.isscalar(volume) else np.asarray(volume, float)
    df = pd.DataFrame({"date": list(dates), "open": close, "high": close, "low": close, "close": close,
                       "adjclose": close, "volume": vol})
    return Fact(entity=entity, attr="series.price_daily", value=series.store(df), as_of=dates[-1], known_at=dates[-1],
                tier="market", source={"name": "test", "url": "https://example.test/px"},
                context={"rows": len(dates), "first": dates[0], "last": dates[-1], "splits": []})


def walk(n=400, step=0.01, price=20.0):
    r = np.where(np.arange(n) % 2 == 0, step, -step)
    return price * np.cumprod(1 + r)


class ExtrasTest(TempHome):
    def test_the_outcome_looks_five_sessions_ahead(self):
        c = walk()
        f = scan.extras(c, c, np.full(len(c), 1e6), np.ones(len(c), bool))
        self.assertAlmostEqual(f["l5"][100], np.log(c[105] / c[100]), places=12)
        self.assertAlmostEqual(f["rv"][100], float(np.sum(np.diff(np.log(c[100:106])) ** 2)), places=12)
        self.assertTrue(np.isnan(f["l5"][-5:]).all())                    # those windows have not closed
        self.assertTrue(np.isfinite(f["l5"][-6]))
        self.assertAlmostEqual(f["lv5"][100], 0.5 * np.log(np.mean(np.diff(np.log(c[95:101])) ** 2) + 1e-8), places=9)
        self.assertTrue(f["ok"][100])
        self.assertEqual(f["why"][100], 0)

    def test_a_data_error_voids_its_window_and_the_year_after(self):
        c = walk(700)
        c[200:] *= 12.0                                                   # +1100% in a day: a bad bar
        f = scan.extras(c, c, np.full(len(c), 1e6), np.ones(len(c), bool))
        self.assertTrue(np.isnan(f["l5"][195:200]).all())
        self.assertTrue(np.isfinite(f["l5"][194]))
        self.assertTrue(np.isfinite(f["l5"][200]))                        # the window after it is whole
        self.assertEqual(f["why"][199], 0)
        self.assertEqual(f["why"][200], 4)
        self.assertEqual(f["why"][200 + scan.ERROR_DAYS - 1], 4)          # for the year its features look back on
        self.assertEqual(f["why"][200 + scan.ERROR_DAYS], 0)

    def test_untradable_rows_say_why(self):
        c = walk()
        n = len(c)
        has = np.ones(n, bool)
        has[300] = False                                                  # no bar that day
        vol = np.full(n, 1e6)
        vol[300] = np.nan
        vol[150] = 0.0                                                    # a session without trades
        f = scan.extras(c, c, vol, has)
        self.assertEqual(f["why"][300], 1)
        self.assertEqual(f["why"][301], 2)                                # the gap counts against the next 20 sessions
        self.assertEqual(f["why"][320], 0)
        self.assertEqual(f["why"][150], 2)
        self.assertEqual(f["why"][169], 2)
        self.assertEqual(f["why"][170], 0)
        self.assertTrue(np.isnan(f["l5"][295]))                           # its window ends on the missing day
        self.assertTrue(np.isnan(f["l5"][300]))                           # and one read on it does not start
        self.assertTrue(np.isfinite(f["l5"][296]))
        thin = scan.extras(c, c, np.full(n, 100.0), np.ones(n, bool))    # about $2,000 a day
        self.assertEqual(thin["why"][100], 3)
        self.assertFalse(thin["ok"][100])


def planted(n, rng, drive="lv60", skew=0.0):
    """Rows whose five-session variance is set by one feature; z is normal, shifted by ``skew`` for the wildest."""
    X = rng.normal(size=(n, len(scan.FEATS)))
    ls = -3.2 + 0.5 * X[:, scan.FEATS.index(drive)]
    z = rng.normal(size=n) + skew * (ls > np.quantile(ls, 0.9))
    daily = rng.normal(size=(n, scan.H)) * np.exp(ls)[:, None] / np.sqrt(scan.H)
    return X.astype(np.float32), z * np.exp(ls), (daily ** 2).sum(1), ls


class ModelTest(TempHome):
    def test_fit_finds_the_driver_and_its_ranges_hold(self):
        rng = np.random.default_rng(4)
        X, l5, rv, _ = planted(60000, rng)
        m = scan.fit(X, l5, rv)
        w = dict(zip(m["feats"], m["w"][1:]))
        self.assertGreater(w["lv60"], 0.9)                                 # log variance moves 1 per unit of lv60
        self.assertLess(max(abs(v) for k, v in w.items() if k != "lv60"), 0.05)
        Xn, yn, _, ls = planted(40000, rng)
        q = scan.quantiles(m, Xn)
        self.assertAlmostEqual(float(((yn >= q[:, 1]) & (yn <= q[:, 3])).mean()), 0.68, delta=0.015)
        self.assertAlmostEqual(float(((yn >= q[:, 0]) & (yn <= q[:, 4])).mean()), 0.90, delta=0.01)
        s, tenth = scan.scale(m, Xn)
        self.assertGreater(float(np.corrcoef(np.log(s), ls)[0, 1]), 0.98)
        self.assertEqual(sorted(set(tenth.tolist())), list(range(scan.BUCKETS)))
        b = scan.fit(X, l5, rv, scan.BASELINE)
        self.assertEqual(b["feats"], ["lv60"])

    def test_each_tenth_keeps_its_own_shape(self):
        rng = np.random.default_rng(6)
        X, l5, rv, _ = planted(60000, rng, skew=0.5)
        m = scan.fit(X, l5, rv)
        self.assertGreater(m["up"][9], 0.62)                               # the wildest tenth leans up, as planted
        self.assertAlmostEqual(m["up"][0], 0.5, delta=0.03)
        q = scan.quantiles(m, X[:2000])
        _, tenth = scan.scale(m, X[:2000])
        self.assertGreater(float(np.median(q[tenth == 9, 2])), 0.0)       # and so does its middle
        self.assertAlmostEqual(float(np.median(q[tenth == 0, 2])), 0.0, delta=0.005)

    def test_report_weeks_get_their_own_shape_when_there_are_enough(self):
        rng = np.random.default_rng(7)
        X, l5, rv, _ = planted(80000, rng)
        due = X[:, scan.FEATS.index("earn_win")] > 0
        l5 = np.where(due, l5 * 2.0, l5)                                  # report weeks: twice the move, same variance told
        m = scan.fit(X, l5, rv, split="earn_win")
        self.assertEqual(len(m["z"]), 2 * scan.BUCKETS)
        q = scan.quantiles(m, X)
        width = q[:, 3] - q[:, 1]
        s, group = scan.scale(m, X)
        self.assertTrue(((group >= scan.BUCKETS) == due).all())
        self.assertAlmostEqual(float(np.median(width[due] / s[due]) / np.median(width[~due] / s[~due])), 2.0, delta=0.15)
        for part in (due, ~due):
            self.assertAlmostEqual(float(((l5 >= q[:, 1]) & (l5 <= q[:, 3]))[part].mean()), 0.68, delta=0.02)
        thin = scan.fit(X[:6000], l5[:6000], rv[:6000], split="earn_win")  # about 300 report rows a tenth: too few
        self.assertEqual(thin["z"][3], thin["z"][3 + scan.BUCKETS])        # so they take their tenth's shape

    def test_chance_of_a_move_falls_with_its_size(self):
        rng = np.random.default_rng(8)
        X, l5, rv, _ = planted(60000, rng)
        m = scan.fit(X, l5, rv)
        c5, c10, c40 = (scan.chance_beyond(m, X[:300], x) for x in (0.02, 0.05, 0.4))
        self.assertTrue((c5 >= c10).all() and (c10 >= c40).all())
        said = scan.chance_beyond(m, X[:5000], 0.05).mean()
        got = (np.abs(np.expm1(l5[:5000])) >= 0.05).mean()
        self.assertAlmostEqual(float(said), float(got), delta=0.03)

    def test_reasons_name_what_widens_the_range(self):
        rng = np.random.default_rng(9)
        X, l5, rv, _ = planted(20000, rng, drive="earn_win")
        m = scan.fit(X, l5, rv)
        row = {k: 0.0 for k in scan.FEATS}
        row.update(earn_win=2.5, lv20=np.log(0.03), logp=np.log(12.0), ldv=np.log(5e6), vix=np.log(18.0))
        why = scan.reasons(m, row, {"date": "2026-11-12", "timing": "after close"})
        self.assertEqual(why, ["results due within eight days (about 2026-11-12, after close)"])
        self.assertGreater(scan.pulls(m, row)["report"], 1.0)
        self.assertEqual(scan.reasons(m, row), ["results due within eight days (by last year's date)"])
        self.assertIn("results came out on 2026-11-05", scan.reasons(m, row, {"out": "2026-11-05"})[0])
        row["earn_win"] = -2.5
        self.assertEqual(scan.reasons(m, row), [])                         # nothing widens it


class LiveHelpersTest(TempHome):
    def test_report_dates_say_due_or_already_out(self):
        from unittest import mock

        from brain import earnradar
        idx = {"tickers": {"DUE": {"reports": ["2025-08-06T20:05:00Z", "2025-11-12T21:05:00Z"]},
                           "OUT": {"reports": ["2025-11-13T21:05:00Z", "2026-11-05T21:05:00Z"]},
                           "FAR": {"reports": ["2025-09-30T12:00:00Z"]}}}
        with mock.patch.object(earnradar, "load_index", return_value=idx):
            got = scan.report_dates("2026-11-09", ["DUE", "OUT", "FAR", "NONE"])
        self.assertEqual(got["DUE"]["date"], "2026-11-11")                # last year's date moved on 52 weeks
        self.assertEqual(got["DUE"]["timing"], "after close")
        self.assertEqual(got["OUT"], {"out": "2026-11-05"})
        self.assertEqual(sorted(got), ["DUE", "OUT"])
        with mock.patch.object(earnradar, "load_index", side_effect=FileNotFoundError):
            self.assertEqual(scan.report_dates("2026-11-09", ["DUE"]), {})

    def test_a_results_filing_in_the_bank_means_the_report_is_out(self):
        due = {"date": "2026-11-11", "timing": "after close"}
        self.assertEqual(scan.report_note({"date": "2026-11-09", "last_report": "2026-11-05"}, due), {"out": "2026-11-05"})
        self.assertEqual(scan.report_note({"date": "2026-11-09", "last_report": "2026-08-06"}, due), due)
        self.assertEqual(scan.report_note({"date": "2026-11-09"}, due), due)
        self.assertIsNone(scan.report_note({"date": "2026-11-09", "last_report": None}, None))

    def test_refresh_reads_prices_then_filings_and_counts_outages(self):
        from unittest import mock

        from brain.sources import prices, sec
        seen = []

        def px(t, ttl):
            seen.append(("px", t))
            if t == "BAD":
                raise OSError("down")
            return []

        def fil(t, all_pages, ttl):
            seen.append(("fil", t, all_pages))
            return [], []
        with mock.patch.object(prices, "ingest", px), mock.patch.object(sec, "filings_for", fil):
            r = scan.refresh(self.bank, ["aaa", "BAD"], log=None)
        self.assertEqual(seen, [("px", "SPY"), ("px", "^VIX"), ("px", "AAA"), ("px", "BAD"),
                                ("fil", "AAA", False), ("fil", "BAD", False)])
        self.assertEqual(r, {"symbols": 4, "failed": {"prices": 1, "filings": 0}})


class TrialTest(TempHome):
    def setUp(self):
        super().setUp()
        self._env = {k: os.environ.get(k) for k in ("BRAIN_SCAN_MODEL", "BRAIN_SHOCKS_MODEL", "BRAIN_SCANS")}
        os.environ["BRAIN_SCAN_MODEL"] = os.path.join(self._tmp.name, "scan_w1.json")
        os.environ["BRAIN_SHOCKS_MODEL"] = os.path.join(self._tmp.name, "shocks_s1.json")
        os.environ["BRAIN_SCANS"] = os.path.join(self._tmp.name, "scans")

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        super().tearDown()

    def test_fit_scan_record_score_and_the_declared_test(self):
        rng = np.random.default_rng(12)
        d = pd.bdate_range("2010-01-04", "2026-10-16").strftime("%Y-%m-%d").to_numpy()   # both ends are in the past:
        tickers = [f"T{k:02d}" for k in range(12)]                                        # a view sees them
        facts = []
        for k, t in enumerate(tickers):
            sd = 0.006 * (1 + k % 4)                                       # four sizes of daily move
            r = rng.normal(0, sd, len(d))
            facts.append(snapshot(t, d, 20 * np.cumprod(1 + r)))
            facts.append(filed(t, "filing", "8-K", "2015-03-02", "2015-03-02T21:00:00Z", items="2.02"))
        facts.append(snapshot("THIN", d, 20 * np.cumprod(1 + rng.normal(0, 0.01, len(d))), volume=50.0))
        facts += [snapshot("SPY", d, np.full(len(d), 400.0)), snapshot("^VIX", d, np.full(len(d), 20.0))]
        self.bank.add_many(facts)
        model = scan.fit_and_save(self.bank, tickers=tickers + ["THIN"], log=None)
        self.assertGreater(model["w1"]["n"], 6000)
        self.assertEqual(model["baseline"]["feats"], ["lv60"])
        shocks.fit_and_save(self.bank, tickers=tickers, log=None)

        res = scan.live(self.bank, tickers + ["THIN", "NOPE"])
        self.assertEqual(res["asof"], d[-1])
        self.assertEqual(len(res["rows"]), len(tickers))
        self.assertEqual({s["t"]: s["why"] for s in res["skipped"]},
                         {"THIN": "under $500,000 traded a day", "NOPE": "too little price history in the bank"})
        half = {r["t"]: r["half"] for r in res["rows"]}
        self.assertGreater(half["T03"], 2 * half["T00"])                   # four times the daily move: a wider range
        self.assertEqual(res["rows"][0]["half"], max(half.values()))       # widest first
        r0 = res["rows"][0]
        self.assertTrue(r0["lo90"] < r0["lo68"] < r0["mid"] < r0["hi68"] < r0["hi90"])
        self.assertTrue(0 <= r0["shock"] <= 1)
        self.assertIn("2 in 3:", scan.line(model["w1"], r0))

        # recorded once, then judged when its five sessions are in the bank
        early = scan.live(scan.Bank(self.bank.root), tickers)
        p = scan.record(early)
        self.assertTrue(p.exists())
        self.assertIsNone(scan.record(early))
        self.assertEqual([x["judged"] for x in scan.score(self.bank)], [False])
        d2 = pd.bdate_range("2010-01-04", "2026-10-30").strftime("%Y-%m-%d").to_numpy()
        more = []
        for t in tickers:
            old = series.load(self.bank.view().latest(t, "series.price_daily").value)["close"].to_numpy()
            more.append(snapshot(t, d2, np.r_[old, old[-1] * np.cumprod(1 + rng.normal(0, 0.01, len(d2) - len(d)))]))
        more += [snapshot("SPY", d2, np.full(len(d2), 400.0)), snapshot("^VIX", d2, np.full(len(d2), 20.0))]
        self.bank.add_many(more)
        judged = scan.score(scan.Bank(self.bank.root))
        self.assertEqual(len(judged), 1)
        j = judged[0]
        self.assertTrue(j["judged"])
        self.assertEqual((j["asof"], j["through"], j["n"]), (d[-1], d2[len(d) + 4], len(tickers)))
        self.assertTrue(0 <= j["cover68"] <= j["cover90"] <= 1)
        rec = json.loads(p.read_text())
        t0 = tickers[0]
        got = scan.outcome(scan.Bank(self.bank.root).view(), t0, shocks.market(scan.Bank(self.bank.root).view()), len(d) - 1)
        px = series.load(scan.Bank(self.bank.root).view().latest(t0, "series.price_daily").value)["close"].to_numpy()
        self.assertAlmostEqual(got[0], float(np.log(px[len(d) + 4] / px[len(d) - 1])), places=9)
        self.assertEqual(len(rec["rows"][t0]), 5)

        # the declared test, as a trial
        self.assertTrue(rangetest.is_trial(False))
        decl = rangetest.declare(log=None)
        self.assertEqual(decl["id"], "W1a")
        self.assertIsNone(rangetest.declare(log=None))                     # declared once
        out = rangetest.test(self.bank, log=None, tickers=tickers)
        m = out["metrics"]
        self.assertGreater(m["stock_weeks"], 1000)
        self.assertEqual(m["first"][:4], "2024")
        self.assertAlmostEqual(m["cover68"], 0.68, delta=0.05)
        self.assertAlmostEqual(m["cover90"], 0.90, delta=0.04)
        self.assertGreater(m["rank_w1"], 0.2)
        self.assertIn("pass", out["gate"])
        self.assertTrue(out["controls"]["oracle_p"] < 0.01)
        led, res_path = rangetest.paths(True)
        text = res_path.read_text()
        self.assertIn("## Weekly ranges", text)
        self.assertIn("| W1a |", text)
        self.assertEqual([json.loads(x)["op"] for x in led.read_text().splitlines()], ["declare", "result"])
        with self.assertRaises(SystemExit):
            rangetest.test(self.bank, log=None, tickers=tickers)           # one result


class PinballTest(TempHome):
    def test_pinball_rewards_the_right_quantile(self):
        y = np.array([0.0, 0.1])
        q = np.array([[0.0] * 5, [0.0] * 5])
        loss = rangetest.pinball(q, y)
        self.assertEqual(loss[0], 0.0)
        self.assertAlmostEqual(loss[1], 0.1 * float(np.mean(scan.QS)), places=12)
        self.assertEqual(rangetest.monthly(np.array(["2024-01", "2024-01", "2024-02"]), np.array([1.0, 3.0, 5.0]))[1].tolist(),
                         [2.0, 5.0])
