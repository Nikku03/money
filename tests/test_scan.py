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
        self.assertEqual(scan.reasons(m, {**row, "date": "2026-11-09"}, {"date": "2026-11-09", "timing": "after close"}),
                         ["results due about now (expected 2026-11-09, after close)"])
        row["earn_win"] = -2.5
        self.assertEqual(scan.reasons(m, row), [])                         # nothing widens it


class ReasonsTest(TempHome):
    def part(self, heavy):
        """A model whose only weights are on ``heavy`` (every feature centred on 0 with unit spread)."""
        w = [0.0] + [1.0 if k in heavy else 0.0 for k in scan.FEATS]
        n = len(scan.FEATS)
        return {"feats": list(scan.FEATS), "lo": [-99.0] * n, "hi": [99.0] * n, "mu": [0.0] * n, "sd": [1.0] * n, "w": w}

    def row(self, **kw):
        return {**{k: 0.0 for k in scan.FEATS}, "lv5": np.log(0.04), "logp": np.log(20.0), "ldv": np.log(5e7), **kw}

    def test_swings_are_said_from_the_last_five_sessions(self):
        m = self.part({"lv250"})
        self.assertEqual(scan.reasons(m, self.row(lv250=2.0, n10=12.0)),
                         ["moved about 4.0% a day over its last five sessions, with 12 days of 10%+ in a year"])
        self.assertEqual(scan.reasons(m, self.row(lv250=2.0, a1=0.41)),
                         ["moved about 4.0% a day over its last five sessions, 41% against the market on the last one"])

    def test_price_and_volume_are_named_only_when_true(self):
        m = self.part({"logp2"})
        self.assertEqual(scan.reasons(m, self.row(logp2=2.0, logp=np.log(1.5))), ["a $1.50 stock"])
        self.assertEqual(scan.reasons(m, self.row(logp2=2.0, volsurge=0.9)), ["its trading volume has jumped this week"])
        self.assertEqual(scan.reasons(m, self.row(logp2=2.0, ldv=np.log(2e6))), ["thinly traded ($2.0M a day)"])
        self.assertEqual(scan.reasons(m, self.row(logp2=2.0)), [])          # $50M a day at $20: nothing to say

    def test_runs_filings_drugs_and_fear(self):
        self.assertEqual(scan.reasons(self.part({"ret20"}), self.row(ret20=np.log(1.5))), ["up 50% against the market in a month"])
        self.assertEqual(scan.reasons(self.part({"hi52"}), self.row(hi52=1.0, lo52=0.01)), ["at its 52-week low"])
        self.assertEqual(scan.reasons(self.part({"offer30"}), self.row(offer30=1.0)), ["a share-sale filing in the last 30 sessions"])
        self.assertEqual(scan.reasons(self.part({"bio"}), self.row(bio=1.0)), ["a drug or biotech company"])
        self.assertEqual(scan.reasons(self.part({"vix"}), self.row(vix=np.log(30.0))), ["market fear is high (VIX 30)"])
        both = scan.reasons(self.part({"vix", "bio", "offer30", "ret20"}), self.row(vix=4.0, bio=3.0, offer30=2.0, ret20=1.0), top=2)
        self.assertEqual(both, ["market fear is high (VIX 55)", "a drug or biotech company"])   # the two widest pulls


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


class ModelHome(TempHome):
    """A temp home whose fitted models and recorded scans stay in it."""

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


class TrialTest(ModelHome):
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
        self.assertEqual(scan.opens_after(early["asof"]), "2026-10-19T13:30:00Z")   # a Friday's scan: Monday's open
        self.assertIsNone(scan.record(early, now="2026-10-19T13:30:00Z"))          # too late: nothing written
        part = {**early, "rows": early["rows"][:5]}
        p = scan.record(part, now="2026-10-17T10:00:00Z")
        self.assertTrue(p.exists())
        first = json.loads(p.read_text())
        self.assertEqual(len(first["rows"]), 5)
        self.assertIsNone(scan.record(part, now="2026-10-17T11:00:00Z"))           # nothing new to add
        self.assertEqual(scan.record(early, now="2026-10-18T09:00:00Z"), p)        # the rest, added before the open
        both = json.loads(p.read_text())
        self.assertEqual(len(both["rows"]), len(tickers))
        self.assertEqual({t: both["rows"][t] for t in first["rows"]}, first["rows"])  # recorded rows unchanged
        self.assertEqual(both["added"][0]["at"], "2026-10-18T09:00:00Z")
        self.assertEqual(len(both["added"][0]["tickers"]), len(tickers) - 5)
        self.assertEqual([x["judged"] for x in scan.score(self.bank)], [False])
        (p.parent / "notes.json").write_text("{}")                         # anything else in the folder is not a scan
        self.assertEqual(len(scan.score(self.bank)), 1)
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
        decl = rangetest.declare(log=None, ids=["W1a"])
        self.assertEqual([x["id"] for x in decl], ["W1a"])
        self.assertEqual(rangetest.declare(log=None, ids=["W1a"]), [])      # declared once
        out = rangetest.test(self.bank, log=None, tickers=tickers)
        self.assertEqual([x["id"] for x in out], ["W1a"])                   # W1b was not declared here
        out = out[0]
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
        self.assertEqual([x["id"] for x in rangetest.declare(log=None, ids=["W1b"])], ["W1b"])  # the other test,
        self.assertEqual(rangetest.spec("W1b")["universes"], rangetest.universe_shas(("us_watch",)))  # declared apart
        w1b = rangetest.test(self.bank, log=None, tickers=tickers)
        self.assertEqual([x["id"] for x in w1b], ["W1b"])
        self.assertIn("| W1b |", res_path.read_text())
        self.assertEqual([x["id"] for x in rangetest.declare(log=None)], ["W1c"])   # every session's close
        self.assertEqual(rangetest.spec("W1c")["readings"], "every session's close")
        w1c = rangetest.test(self.bank, log=None, tickers=tickers)[0]["metrics"]
        self.assertGreater(w1c["stock_weeks"], 4 * m["stock_weeks"])          # about five readings a week
        self.assertAlmostEqual(w1c["cover68"], 0.68, delta=0.05)


class PinballTest(TempHome):
    def test_pinball_rewards_the_right_quantile(self):
        y = np.array([0.0, 0.1])
        q = np.array([[0.0] * 5, [0.0] * 5])
        loss = rangetest.pinball(q, y)
        self.assertEqual(loss[0], 0.0)
        self.assertAlmostEqual(loss[1], 0.1 * float(np.mean(scan.QS)), places=12)
        self.assertEqual(rangetest.monthly(np.array(["2024-01", "2024-01", "2024-02"]), np.array([1.0, 3.0, 5.0]))[1].tolist(),
                         [2.0, 5.0])


class ReplayTest(ModelHome):
    def test_replay_reads_each_week_as_live_and_judges_it(self):
        from brain import scanreplay
        rng = np.random.default_rng(21)
        d = pd.bdate_range("2010-01-04", "2026-10-16").strftime("%Y-%m-%d").to_numpy()
        tickers = [f"T{k:02d}" for k in range(30)]
        facts = []
        for k, t in enumerate(tickers):
            sd = 0.005 * (1 + k % 5)                                       # five sizes of daily move
            r = rng.normal(0, sd, len(d))
            if k == 29:
                r[len(d) - 40] = 0.6                                       # one 60% day, in a week of the replay
            facts.append(snapshot(t, d, 20 * np.cumprod(1 + r)))
        spy = 400 * np.cumprod(1 + np.where(np.arange(len(d)) % 2, 0.001, -0.0005))
        facts += [snapshot("SPY", d, spy), snapshot("^VIX", d, np.full(len(d), 20.0))]
        self.bank.add_many(facts)
        scan.fit_and_save(self.bank, tickers=tickers, log=None)
        shocks.fit_and_save(self.bank, tickers=tickers, log=None)
        res = scanreplay.run(self.bank, "2026-01-01", "2026-12-31", tickers=tickers, log=None)
        s, weeks = res["summary"], res["weeks"]
        self.assertEqual(s["weeks"], len(weeks))
        self.assertGreater(s["weeks"], 35)
        self.assertEqual(weeks[0]["week"][:4], "2026")
        self.assertLessEqual(weeks[-1]["week"], d[-6])                    # only weeks whose five sessions closed
        self.assertTrue(all(25 <= w["stocks"] <= 30 for w in weeks))       # a stock that drifted under $0.50 trades too
        self.assertEqual(s["stock_weeks"], sum(w["stocks"] for w in weeks))  # little to list, and is left out
        self.assertAlmostEqual(s["held68"], 0.68, delta=0.06)
        self.assertAlmostEqual(s["held90"], 0.90, delta=0.05)
        self.assertGreater(s["top_abs"], s["all_abs"])                    # the widest ranges moved most
        self.assertEqual(s["shock_picks"], 25 * s["weeks"])
        self.assertGreaterEqual(s["shock_all"], 1)                        # the planted 60% day is seen
        i = list(d).index(weeks[3]["week"])
        self.assertAlmostEqual(weeks[3]["spy"], spy[i + 5] / spy[i] - 1, places=9)
        j = list(d).index(weeks[4]["week"])                                # the next reading: held to it, no overlap
        self.assertAlmostEqual(weeks[3]["spy_ret"], spy[j] / spy[i] - 1, places=9)
        first, last = list(d).index(weeks[0]["week"]), list(d).index(weeks[-1]["week"])
        self.assertAlmostEqual(scanreplay.growth([w["spy_ret"] for w in weeks])[0],
                               spy[last + 5] / spy[first], places=6)      # every session counted once
        end, fall = scanreplay.growth([0.1, -0.5, 0.2])
        self.assertAlmostEqual(end, 1.1 * 0.5 * 1.2)
        self.assertAlmostEqual(fall, -0.5)
        q = scanreplay.by_quarter(weeks)
        self.assertEqual(sum(x["weeks"] for x in q), s["weeks"])
        self.assertEqual(q[0]["quarter"], "2026Q1")


class WatchListTest(TempHome):
    def test_the_scan_watches_the_research_lists_then_the_newer_listings(self):
        listed = shocks.universe_tickers()
        both = scan.scan_tickers()
        self.assertEqual(both[:len(listed)], listed)
        self.assertEqual(len(both), len(set(both)))
        newer = scan.watch_tickers()
        self.assertTrue(set(newer).isdisjoint(listed))                     # no research stock is a newer listing
        self.assertEqual(set(both), set(listed) | set(newer))
        d = json.loads((scan.config.ROOT / "universe" / "us_watch.json").read_text())
        self.assertEqual(d["n"], len(d["tickers"]))
        self.assertEqual(set(d["bands"]), set(d["tickers"]))
        self.assertTrue(scan.watch_status().startswith("W1b ") or scan.watch_status() == "untested until W1b runs")


class BetweenScansTest(TempHome):
    def setUp(self):
        super().setUp()
        self._old_scans = os.environ.get("BRAIN_SCANS")
        os.environ["BRAIN_SCANS"] = os.path.join(self._tmp.name, "scans")

    def tearDown(self):
        if self._old_scans is None:
            os.environ.pop("BRAIN_SCANS", None)
        else:
            os.environ["BRAIN_SCANS"] = self._old_scans
        super().tearDown()

    def chart(self, closes, start, periods):
        ts = [start + 300 * k for k in range(len(closes))]
        return json.dumps({"chart": {"result": [{"meta": {"currentTradingPeriod": periods}, "timestamp": ts,
                                                  "indicators": {"quote": [{"close": closes}]}}]}}).encode()

    def test_the_latest_trade_and_the_part_of_the_day_it_falls_in(self):
        from brain import scanwatch
        day = 1_794_000_000 - 1_794_000_000 % 86400                       # a UTC midnight
        periods = {"pre": {"start": day + 9 * 3600, "end": day + 14.5 * 3600},
                   "regular": {"start": day + 14.5 * 3600, "end": day + 21 * 3600},
                   "post": {"start": day + 21 * 3600, "end": day + 25 * 3600}}
        q = scanwatch.parse_quote(self.chart([10.0, 10.5, None], day + 10 * 3600, periods))
        self.assertEqual((q["price"], q["session"]), (10.5, "before the open"))   # a missing last bar is skipped
        self.assertEqual(scanwatch.parse_quote(self.chart([11.0], day + 15 * 3600, periods))["session"], "in session")
        self.assertEqual(scanwatch.parse_quote(self.chart([12.0], day + 22 * 3600, periods))["session"], "after the close")
        self.assertEqual(scanwatch.parse_quote(self.chart([9.0], day + 3600, periods))["session"], "overnight")
        self.assertIsNone(scanwatch.parse_quote(self.chart([None], day + 15 * 3600, periods)))

    def test_moves_judged_against_the_recorded_range(self):
        from brain import scanwatch
        rng = [np.log(0.85), np.log(0.92), np.log(1.09), np.log(1.18)]
        self.assertEqual({k: v for k, v in scanwatch.judge(np.log(1.05), rng).items() if k != "used"},
                         {"outside68": False, "outside90": False})
        self.assertAlmostEqual(scanwatch.judge(np.log(1.05), rng)["used"], np.log(1.05) / np.log(1.09))
        self.assertTrue(scanwatch.judge(np.log(1.12), rng)["outside68"])
        self.assertFalse(scanwatch.judge(np.log(1.12), rng)["outside90"])
        self.assertTrue(scanwatch.judge(np.log(0.80), rng)["outside90"])

    def test_filings_since_the_close_and_what_they_are(self):
        from brain import scanwatch
        fs = [filed("X", "filing", "8-K", "2026-11-09", "2026-11-09T20:10:00Z", items="2.02,9.01"),   # 15:10 ET: before
              filed("X", "filing", "8-K", "2026-11-09", "2026-11-09T21:30:00Z", items="2.02,9.01"),   # 16:30 ET: after
              filed("X", "filing", "424B5", "2026-11-10", "2026-11-10T12:00:00Z"),
              filed("X", "filing", "8-K", "2026-11-10", "2026-11-10T13:00:00Z", items="1.01,8.01,9.01")]
        got = scanwatch.since_close(fs, "2026-11-09")
        self.assertEqual([f.known_at for f in got], [f.known_at for f in fs[1:]])
        self.assertEqual([scanwatch.describe(f) for f in got],
                         ["results report (8-K 2.02)", "share-sale filing (424B5)", "8-K 1.01, 8.01"])

    def test_watch_flags_stocks_outside_their_range_and_a_market_wide_move(self):
        from unittest import mock

        from brain import scanwatch
        d = pd.bdate_range("2025-01-02", "2026-11-09").strftime("%Y-%m-%d").to_numpy()
        self.bank.add_many([snapshot(t, d, np.full(len(d), 20.0)) for t in ("AAA", "BBB", "CCC")])
        r = [round(float(x), 4) for x in (np.log(0.85), np.log(0.92), np.log(1.09), np.log(1.18))]
        p = scanwatch.scan.scans_dir() / "2026-11-09.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"asof": "2026-11-09", "rows": {"AAA": r + [0.01], "BBB": r + [0.30], "CCC": r + [0.02]}}))
        prices = {"AAA": 24.0, "BBB": 20.4, "CCC": 21.0, "SPY": 98.0, "QQQ": 99.0, "IWM": 100.0, "ES=F": 97.9,
                  "^VIX": 23.2}
        with mock.patch.object(scanwatch, "quote", lambda s: {"price": prices[s], "at": "2026-11-10T15:00Z",
                                                                "session": "in session"}), \
                mock.patch.object(scanwatch, "market_close_on", lambda s, day: 20.0 if s == "^VIX" else 100.0), \
                mock.patch.object(scanwatch, "filings_since", lambda t, asof: ["results report (8-K 2.02) at x"] if t == "AAA" else []), \
                mock.patch.object(scanwatch, "TOP_SHOCKS", 1):                # the scan's single likeliest 20%+ day
            res = scanwatch.watch(self.bank)
        self.assertEqual((res["asof"], res["checked"], res["outside68"], res["outside90"]), ("2026-11-09", 3, 1, 1))
        self.assertTrue(res["market_wide"])                                   # futures -2.1% since the close
        self.assertTrue(res["fear_up"])                                       # VIX +16%
        top = res["flags"][0]
        self.assertEqual((top["t"], top["outside90"], top["filings"]), ("AAA", True, ["results report (8-K 2.02) at x"]))
        self.assertIn("BBB", [f["t"] for f in res["flags"]])                 # the scan's likeliest 20%+ day, inside
        self.assertNotIn("CCC", [f["t"] for f in res["flags"]])              # inside, nothing to say
        out = scanwatch.record(res)
        self.assertEqual(out.parent.name, "2026-11-09")
        self.assertEqual(json.loads(out.read_text())["outside90"], 1)

    def test_a_loop_checks_again_after_its_interval(self):
        from types import SimpleNamespace
        from unittest import mock

        from brain import scanwatch
        calls, naps = [], []
        res = {"asof": "2026-11-09", "at": "2026-11-10T15:00:00Z", "listed": 0, "market": [], "checked": 0,
               "outside68": 0, "outside90": 0, "market_wide": False, "fear_up": False, "flags": []}
        with mock.patch.object(scanwatch, "watch", lambda *a, **k: calls.append(1) or res), \
                mock.patch.object(scanwatch, "show", lambda *a, **k: None):
            scanwatch.cmd(SimpleNamespace(tickers=[], only_top=None, every=5, record=False), sleep=naps.append, cycles=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(naps), 2)
        self.assertTrue(all(14 * 60 < x <= 15 * 60 for x in naps))          # never more often than every 15 minutes
