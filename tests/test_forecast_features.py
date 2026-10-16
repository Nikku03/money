import math
import os
import shutil
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from brain import claims, events, metrics, series
from brain.bank import Fact
from brain.forecast import features as F
from brain.research import signals
from brain.sources import insiders
from brain.timeutil import norm_known_at
from tests.test_quarterly import EPS, NI, fiscal_year, known_dates
from tests.util import TempHome, filed

DAYS = pd.bdate_range("2012-01-02", "2016-12-30").strftime("%Y-%m-%d").tolist()   # 1304 sessions
STOCKS = ["A", "B", "C", "D", "E", "PENNY"]
NAMES_IN_CONTRACT = [
    "r_5", "r_21", "r_63", "r_126", "mom_12_1", "ma_50", "ma_200", "rsi_14", "vol_21", "vol_63", "vol_ratio",
    "beta_252", "idio_vol_63", "skew_63", "range_21",
    "dist_52w_high", "dist_52w_low", "max_ret_21", "min_ret_21", "rev_5", "rev_21", "ltr_3y", "abn_volume",
    "turnover_trend", "insider_buy_90d", "earnings_car_3d",
    "earnings_yield_ttm", "log_mktcap", "sales_growth", "gross_profitability", "asset_growth", "accruals", "leverage",
    "sue",
    "vix", "vix_chg_21", "spy_r_21", "spy_r_252", "spy_above_200", "mkt_vol_21",
]


def make_grid(dates, anchor, h, since=None):
    """Every h-th session counted back from the anchor, oldest first, on or after ``since``."""
    a = dates.index(anchor)
    return [dates[i] for i in range(a, -1, -h)[::-1] if since is None or dates[i] >= since]


def snap(entity, dates, close, adj=None, high=None, low=None, volume=None, known=None, splits=()):
    close = np.asarray(close, dtype=float)
    df = pd.DataFrame({"date": dates, "open": close, "high": close if high is None else high,
                       "low": close if low is None else low, "close": close,
                       "adjclose": close if adj is None else adj,
                       "volume": np.full(len(dates), 1000) if volume is None else volume})
    sha = series.store(df)
    return Fact(entity=entity, attr="series.price_daily", value=sha, as_of=dates[-1], known_at=known or dates[-1],
                tier="market", source={"name": "test", "url": "https://example.test/px"},
                context={"rows": len(dates), "first": dates[0], "last": dates[-1], "splits": [list(s) for s in splits]})


def trade(t, known, as_of, usd, code="P", rel="officer", accn="x1", owner="9", lot=None):
    ctx = {"code": code, "price": 10.0, "value_usd": usd, "owner": "Officer", "owner_cik": owner, "relationship": rel,
           "title": None, "accn": accn, "ownership": "D"}
    if lot:
        ctx["lot"] = lot
    return Fact(entity=t, attr=insiders.ATTR, value=usd / 10.0, unit="shares", as_of=as_of, known_at=known,
                tier="filed", source={"name": "SEC Form 4", "url": f"https://example.test/{accn}"}, context=ctx)


def filing(t, form, known, items="", accn=None):
    return Fact(entity=t, attr="filing", value=form, as_of=known[:10], known_at=known, tier="filed",
                source={"name": "SEC EDGAR", "url": f"https://example.test/{accn or known}"},
                context={"accn": accn or known, **({"items": items} if items else {})})


def annual(t, attr, v, y, known, instant=False):
    end = f"{y}-12-31"
    if instant:
        return filed(t, attr, v, end, known, form="10-K")
    return filed(t, attr, v, end, known, form="10-K", start=f"{y}-01-01", duration_days=364)


class Synthetic(TempHome):
    """Six stocks on 1,304 sessions (one a penny stock), SPY, ^VIX from session 300, a late listing (LATE)
    and a stock with a long gap (GAP); fundamentals for A. ``FIRST`` cuts the first price book short."""
    FIRST = None

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(11)
        n = len(DAYS)
        self.spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
        self.px = {}
        for k, t in enumerate(STOCKS):
            base = 2.0 if t == "PENNY" else 50.0 + 10 * k
            c = base * np.cumprod(1 + 0.8 * (self.spy / np.roll(self.spy, 1) - 1) * (k % 3) + rng.normal(0, 0.02, n))
            c[0] = base
            spread = np.abs(rng.normal(0, 0.01, n))
            self.px[t] = {"close": c, "adj": c * 0.9, "high": c * (1 + spread), "low": c * (1 - spread),
                          "volume": rng.integers(1000, 5000, n).astype(float)}
        self.vix = 15 + np.cumsum(rng.normal(0, 0.3, n))
        self.gap_close = 30 * np.cumprod(1 + rng.normal(0, 0.02, n))
        self.late_from = 1000
        self.gap = (900, 912)
        self.write_prices(self.FIRST or n)
        self.add_company_facts()

    def write_prices(self, upto, scale=1.0, known=None):
        d = DAYS[:upto]
        known = known or d[-1]
        facts = [snap("SPY", d, self.spy[:upto], known=known),
                 snap("^VIX", d[300:], self.vix[300:upto], known=known)]
        for t in STOCKS:
            p = self.px[t]
            facts.append(snap(t, d, p["close"][:upto], adj=np.round(p["adj"][:upto] * scale, 6), high=p["high"][:upto],
                              low=p["low"][:upto], volume=p["volume"][:upto], known=known))
        if upto > self.late_from:
            late = self.px["B"]["close"][:upto] * 1.1
            facts.append(snap("LATE", d[self.late_from:], late[self.late_from:], known=known))
        keep = [i for i in range(upto) if not self.gap[0] <= i < self.gap[1]]
        facts.append(snap("GAP", [d[i] for i in keep], self.gap_close[keep], known=known))
        self.bank.add_many(facts)

    def add_company_facts(self):
        b = self.bank
        for y in (2012, 2013, 2014, 2015):
            b.add_many(fiscal_year("A", NI, y, [10e6 + y, 12e6, 9e6 + 3 * y, 15e6], known_dates(y)))
            eps = [0.10 + 0.01 * (y - 2012) + 0.003 * q * (-1) ** (q + y) for q in range(4)]
            b.add_many(fiscal_year("A", EPS, y, eps, known_dates(y)))
        for y, (rev, gp, ocf, assets, liab, sh) in {2014: (400e6, 150e6, 40e6, 900e6, 500e6, 1.0e7),
                                                    2015: (460e6, 170e6, 30e6, 1000e6, 600e6, 1.1e7)}.items():
            known = f"{y + 1}-02-20"
            b.add_many([annual("A", "us-gaap:Revenues", rev, y, known), annual("A", "us-gaap:GrossProfit", gp, y, known),
                        annual("A", "us-gaap:NetCashProvidedByUsedInOperatingActivities", ocf, y, known),
                        annual("A", "us-gaap:Assets", assets, y, known, instant=True),
                        annual("A", "us-gaap:Liabilities", liab, y, known, instant=True),
                        annual("A", "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding", sh, y, known)])

    def i(self, d):
        return DAYS.index(d)

    def asof(self):
        return F.T(DAYS[-1])

    def grid(self, anchor=None, h=5, since=None):
        return make_grid(DAYS, anchor or DAYS[-1], h, since=since or DAYS[300])

    def tickers(self):
        return [*STOCKS, "LATE", "GAP"]


class PricesAndLabelTest(Synthetic):
    def test_features_are_exactly_the_contract_list(self):
        names = [n for n, _, _ in F.FEATURES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([n for n in names if n not in F.INDICATORS], NAMES_IN_CONTRACT)
        self.assertEqual(F.INDICATORS, {"has_earnings_recent", *(f"has_{n}" for n in F.FUND_NAMES)})
        self.assertTrue(all(g in "PBFR" and len(g) == 1 and desc for _, g, desc in F.FEATURES))
        groups = {n: g for n, g, _ in F.FEATURES}
        self.assertEqual({n for n in names if groups[n] == "R"}, set(NAMES_IN_CONTRACT[-6:]))
        self.assertEqual({n for n in names if groups[n] == "F" and n not in F.INDICATORS}, set(NAMES_IN_CONTRACT[26:34]))
        self.assertEqual(F.FEATURES_VERSION, 1)

    def test_prices_from_one_book_with_high_low_volume(self):
        px = F.prices(self.bank, ["SPY", "a", *STOCKS[1:], "^VIX", "LATE", "GAP"], self.asof())
        self.assertEqual(px.tickers, [*STOCKS, "LATE", "GAP"])  # SPY and ^VIX are context, not stocks
        self.assertEqual(list(px.dates), [d for d in DAYS if d >= F.PRICE_START])
        j = px.tickers.index("A")
        p = self.px["A"]
        np.testing.assert_allclose(px.adj[:, j], np.round(p["adj"], 6), atol=1e-12)
        np.testing.assert_allclose(px.close[:, j], np.round(p["close"], 6), atol=1e-12)
        np.testing.assert_allclose(px.high[:, j], np.round(p["high"], 6), atol=1e-12)
        np.testing.assert_allclose(px.low[:, j], np.round(p["low"], 6), atol=1e-12)
        np.testing.assert_allclose(px.volume[:, j], p["volume"])
        np.testing.assert_allclose(px.spy, np.round(self.spy, 6), atol=1e-12)
        self.assertTrue(np.isnan(px.vix[:300]).all())
        np.testing.assert_allclose(px.vix[300:], np.round(self.vix[300:], 6), atol=1e-12)
        self.assertEqual({t: f.id for t, f in px.book.snapshot.items()},
                         {t: self.bank.view(self.asof()).latest(t, "series.price_daily").id for t in [*px.tickers, "SPY", "^VIX"]})
        # The gap: closes filled for 3 sessions, then NaN; high/low/volume never filled.
        g, (g0, g1) = px.tickers.index("GAP"), self.gap
        self.assertTrue(np.isfinite(px.adj[g0:g0 + 3, g]).all() and np.isnan(px.adj[g0 + 3:g1, g]).all())
        self.assertTrue(np.isnan(px.volume[g0:g1, g]).all())

    def test_listed_rule(self):
        px = F.prices(self.bank, self.tickers(), self.asof())
        late, gap = px.tickers.index("LATE"), px.tickers.index("GAP")
        self.assertFalse(px.listed(self.late_from + 250)[late])       # 251 bars
        self.assertTrue(px.listed(self.late_from + 251)[late])        # 252 bars
        g0, g1 = self.gap
        self.assertTrue(px.listed(g0 + 1)[gap])                       # last close 2 sessions back, within 4 days
        self.assertFalse(px.listed(g0 + 5)[gap])                      # last close a week back
        self.assertTrue(px.listed(g1)[gap])
        self.assertTrue(px.listed(250).sum() == 0 and px.listed(251)[:len(STOCKS)].all())   # 251, then 252 bars

    def test_outcome_is_claims_outcome_from_the_next_session(self):
        px = F.prices(self.bank, STOCKS, self.asof())
        d = DAYS[700]
        o = F.outcome(px.book, "A", d, 5)
        self.assertEqual(o, claims.outcome(px.book, "A", d + "T23:59:59Z", 5, "SPY"))
        self.assertEqual((o["entry"], o["exit"]), (DAYS[701], DAYS[706]))
        a, s = np.round(self.px["A"]["adj"], 6), np.round(self.spy, 6)
        self.assertAlmostEqual(o["excess"], (a[706] / a[701] - 1) - (s[706] / s[701] - 1), places=12)
        self.assertIsNone(F.outcome(px.book, "A", DAYS[-5], 5))   # the exit is not in the book yet

    def test_hand_computed_price_and_regime_features(self):
        px = F.prices(self.bank, STOCKS, self.asof())
        i, j = 1000, 0
        a, c = px.adj[:, j], px.close[:, j]
        hi, lo, v, s = px.high[:, j], px.low[:, j], px.volume[:, j], px.spy
        sf = F.stock_features(px, i)
        sr = a[i - 251:i + 1] / a[i - 252:i] - 1
        mr = s[i - 251:i + 1] / s[i - 252:i] - 1
        beta = np.polyfit(mr, sr, 1)[0]
        x63 = np.diff(np.log(a[i - 63:i + 1]))
        m = x63.mean()
        gains = losses = None
        ch = np.diff(a[i - 250:i + 1])
        for k, x in enumerate(ch):
            g, lo_ = max(x, 0.0), max(-x, 0.0)
            if k < 14:
                gains, losses = (gains or 0) + g / 14, (losses or 0) + lo_ / 14
            else:
                gains, losses = (gains * 13 + g) / 14, (losses * 13 + lo_) / 14
        exp = {
            "r_5": a[i] / a[i - 5] - 1, "r_21": a[i] / a[i - 21] - 1, "r_63": a[i] / a[i - 63] - 1,
            "r_126": a[i] / a[i - 126] - 1, "mom_12_1": a[i - 21] / a[i - 252] - 1,
            "ma_50": a[i] / a[i - 49:i + 1].mean() - 1, "ma_200": a[i] / a[i - 199:i + 1].mean() - 1,
            "rsi_14": 100 - 100 / (1 + gains / losses),
            "vol_21": np.std(np.diff(np.log(a[i - 21:i + 1])), ddof=1), "vol_63": np.std(x63, ddof=1),
            "beta_252": beta, "idio_vol_63": np.std(sr[-63:] - beta * mr[-63:], ddof=1),
            "skew_63": ((x63 - m) ** 3).mean() / ((x63 - m) ** 2).mean() ** 1.5,
            "range_21": ((hi - lo) / c)[i - 20:i + 1].mean(),
            "dist_52w_high": c[i] / c[i - 251:i + 1].max() - 1, "dist_52w_low": c[i] / c[i - 251:i + 1].min() - 1,
            "max_ret_21": sr[-21:].max(), "min_ret_21": sr[-21:].min(),
            "rev_5": -(a[i] / a[i - 5] - 1), "rev_21": -(a[i] / a[i - 21] - 1),
            "ltr_3y": -(a[i - 252] / a[i - 756] - 1),
            "abn_volume": math.log(v[i - 4:i + 1].mean() / v[i - 62:i + 1].mean()),
            "turnover_trend": math.log((c * v)[i - 20:i + 1].mean() / (c * v)[i - 251:i + 1].mean()),
        }
        exp["vol_ratio"] = exp["vol_21"] / exp["vol_63"]
        self.assertEqual(set(sf), set(exp))
        for k, e in exp.items():
            self.assertAlmostEqual(float(sf[k][j]), float(e), places=9, msg=k)
        r = F.regime_features(px, i)
        vx = px.vix
        self.assertAlmostEqual(r["vix"], vx[i])
        self.assertAlmostEqual(r["vix_chg_21"], vx[i] - vx[i - 21])
        self.assertAlmostEqual(r["spy_r_21"], s[i] / s[i - 21] - 1)
        self.assertAlmostEqual(r["spy_r_252"], s[i] / s[i - 252] - 1)
        self.assertEqual(r["spy_above_200"], float(s[i] > s[i - 199:i + 1].mean()))
        self.assertAlmostEqual(r["mkt_vol_21"], np.std(np.diff(np.log(s[i - 21:i + 1])), ddof=1))
        early = F.stock_features(px, 300)
        self.assertTrue(np.isnan(early["ltr_3y"]).all() and np.isfinite(early["r_126"]).all())
        self.assertTrue(math.isnan(F.regime_features(px, 299)["vix"]))

    def test_rank_and_zscore_transforms(self):
        x = np.array([3.0, 1.0, 1.0, np.nan, 7.0, 5.0])
        valid = np.array([True, True, True, True, True, False])
        r = F.rank_unit(x, valid)
        np.testing.assert_allclose(r, [1 / 6, -1 / 3, -1 / 3, 0.0, 0.5, 0.0])   # ties share their mean rank
        np.testing.assert_array_equal(F.rank_unit(np.zeros(5), np.ones(5, bool)), np.zeros(5))
        v = np.arange(60.0) ** 1.5
        v[55] = np.nan
        z = F.zscore_earlier(v)
        self.assertTrue((z[:52] == 0).all() and z[55] == 0)
        prev = v[:57][np.isfinite(v[:57])]
        self.assertAlmostEqual(z[57], (v[57] - prev.mean()) / prev.std(ddof=1))
        v2 = v.copy()
        v2[58:] = -1e9                                                 # later dates never move an earlier z
        np.testing.assert_array_equal(F.zscore_earlier(v2)[:58], z[:58])


class EventsAndFundamentalsTest(Synthetic):
    def test_insider_buying_is_signals_rule(self):
        b, d0 = self.bank, "2015-06-15"
        b.add_many([
            trade("A", "2015-06-01T21:00:00Z", "2015-05-29", 50_000, accn="a1"),
            trade("A", "2015-06-01T23:59:59Z", "2015-05-29", 50_000, accn="a1"),       # the same lot, read in bulk
            trade("A", "2015-06-02T23:59:59Z", "2015-06-01", 20_000, accn="a2", lot=None),
            trade("A", "2015-06-02T23:59:59Z", "2015-06-01", 20_000, accn="a2", lot=2),  # a second identical lot
            trade("A", "2015-05-01T23:59:59Z", "2015-04-30", 7_000, rel="director", accn="a3"),
            trade("A", "2015-05-02T23:59:59Z", "2015-04-30", 900_000, rel="ten_percent_owner", accn="a4"),
            trade("A", "2015-05-03T23:59:59Z", "2015-04-30", 800_000, code="S", accn="a5"),
            trade("A", "2015-02-01T23:59:59Z", "2015-01-30", 60_000, accn="a6"),        # over 90 days before d0
            trade("A", "2015-06-15T23:59:59Z", "2015-06-15", 1_000, accn="a7"),         # stamped exactly T(d0)
            trade("A", "2015-06-16T00:00:00Z", "2015-06-15", 5e6, accn="a8"),           # one second later
        ])
        px = F.prices(b, STOCKS, self.asof())
        for d in ["2015-03-02", "2015-05-01", "2015-06-01", "2015-06-02", d0, "2015-06-16", "2015-08-30", "2015-09-30"]:
            want = signals.insider_buying(SimpleNamespace(date=d, view=b.view(d)), "A")
            got = F.event_features(b, ["A"], d, px.book)["insider_buy_90d"][0]
            self.assertAlmostEqual(got, math.log1p(want), places=12, msg=d)
        self.assertAlmostEqual(F.event_features(b, ["A"], d0, px.book)["insider_buy_90d"][0],
                               math.log1p(50_000 + 40_000 + 7_000 + 1_000))
        # One feature set over many dates reads each date through its own view, as signals does.
        g = [x for x in DAYS[self.i("2015-01-02"):self.i("2015-10-30"):3]]
        fs = F.feature_set(b, STOCKS, g, self.asof(), 5, cache=False)
        a = fs.ticker == "A"
        for d, x in zip(fs.date[a], fs.raw[a, fs.col("insider_buy_90d")]):
            want = signals.insider_buying(SimpleNamespace(date=str(d), view=b.view(str(d))), "A")
            self.assertAlmostEqual(float(x), math.log1p(want), places=4, msg=str(d))

    def test_earnings_reaction_latest_closed_release_within_63_sessions(self):
        b = self.bank
        b.add_many([filing("A", "8-K", "2015-04-29T20:30:00Z", "2.02,9.01", "e1"),   # 16:30 New York: moves 04-30
                    filing("A", "8-K/A", "2015-05-05T12:00:00Z", "2.02", "e2"),      # amends it
                    filing("A", "8-K", "2015-07-29T13:00:00Z", "2.02", "e3"),        # before the open: moves 07-29
                    filing("A", "8-K", "2015-06-10T13:00:00Z", "5.02", "e4")])       # not earnings
        px = F.prices(b, STOCKS, self.asof())
        a, s = px.adj[:, 0], px.spy
        e1, e3 = self.i("2015-04-30"), self.i("2015-07-29")
        car = lambda e: (a[e + 1] / a[e - 2] - 1) - (s[e + 1] / s[e - 2] - 1)
        ef = lambda d: F.event_features(b, ["A"], d, px.book)
        self.assertEqual(ef(DAYS[e1 - 1])["has_earnings_recent"][0], 0.0)
        self.assertEqual(ef(DAYS[e1])["has_earnings_recent"][0], 0.0)              # its window closes at e + 1
        r = ef(DAYS[e1 + 1])
        self.assertAlmostEqual(r["earnings_car_3d"][0], car(e1), places=12)
        self.assertEqual(r["has_earnings_recent"][0], 1.0)
        self.assertAlmostEqual(ef(DAYS[e1 + 63])["earnings_car_3d"][0], car(e1), places=12)
        self.assertEqual(ef(DAYS[e3])["has_earnings_recent"][0], float(e3 - e1 <= 63))  # e3 open: e1 if near
        self.assertAlmostEqual(ef(DAYS[e3 + 1])["earnings_car_3d"][0], car(e3), places=12)
        self.assertAlmostEqual(ef(DAYS[e3 + 63])["earnings_car_3d"][0], car(e3), places=12)
        late = ef(DAYS[e3 + 64])
        self.assertEqual((late["earnings_car_3d"][0], late["has_earnings_recent"][0]), (0.0, 0.0))
        # The same as reading every release through view(T(d)) afresh, for each date.
        literal = {}
        for d in [DAYS[k] for k in (e1 - 1, e1, e1 + 1, e1 + 63, e1 + 64, e3 - 1, e3, e3 + 1, e3 + 63, e3 + 64)] + \
                DAYS[e1 - 3:e3 + 70:7]:
            v = b.view(d)
            want = (0.0, 0.0)
            for e in reversed(events.distinct(events.earnings_events(v, "A"))):
                rr = events.reaction(px.book, v, "A", e)
                if rr is None or rr["car_3d"] is None:
                    continue
                if self.i(d) - self.i(rr["date"]) <= 63:
                    want = (rr["car_3d"], 1.0)
                break
            got = ef(d)
            self.assertEqual((got["earnings_car_3d"][0], got["has_earnings_recent"][0]), want, d)
            literal[d] = want
        fs = F.feature_set(b, STOCKS, sorted(x for x in literal if x in DAYS), self.asof(), 5, cache=False)
        a = fs.ticker == "A"
        for d, car, has in zip(fs.date[a], fs.raw[a, fs.col("earnings_car_3d")], fs.raw[a, fs.col("has_earnings_recent")]):
            self.assertAlmostEqual(float(car), literal[str(d)][0], places=6, msg=str(d))
            self.assertEqual(float(has), literal[str(d)][1], str(d))

    def test_hand_computed_fundamentals_at_a_month_end(self):
        px = F.prices(self.bank, STOCKS, self.asof())
        me = "2016-03-31"
        ff = F.fundamental_features(self.bank, ["A", "B"], [me], px.book)
        close = px.close[self.i(me), 0]
        ni15 = sum([10e6 + 2015, 12e6, 9e6 + 3 * 2015, 15e6])
        exp = {"log_mktcap": math.log(1.1e7 * close), "sales_growth": 460 / 400 - 1, "gross_profitability": 0.17,
               "asset_growth": 1000 / 900 - 1, "accruals": (ni15 - 30e6) / 1000e6, "leverage": 0.6,
               "earnings_yield_ttm": ni15 / (1.1e7 * close)}
        for k, e in exp.items():
            self.assertAlmostEqual(ff[k][0, 0], e, places=10, msg=k)
            self.assertEqual(ff[f"has_{k}"][0, 0], 1.0)
        self.assertAlmostEqual(ff["sue"][0, 0], metrics.sue(self.bank.view(me), "A").value)
        self.assertTrue(np.isfinite(ff["sue"][0, 0]))
        for k in F.FUND_NAMES:                       # B has no filings: NaN, with its indicator 0
            self.assertTrue(np.isnan(ff[k][0, 1]) and ff[f"has_{k}"][0, 1] == 0.0, k)

    def test_gross_profit_and_leverage_fallbacks(self):
        b = self.bank
        known = "2016-02-20"
        c = [annual("C", "us-gaap:Revenues", 300e6, 2015, known), annual("C", "us-gaap:CostOfRevenue", 180e6, 2015, known),
             annual("C", "us-gaap:Assets", 600e6, 2015, known, instant=True),
             annual("C", "us-gaap:StockholdersEquity", 240e6, 2015, known, instant=True)]
        d = [annual("D", "us-gaap:Revenues", 100e6, 2015, known),
             annual("D", "us-gaap:CostOfGoodsAndServicesSold", 70e6, 2015, known),
             annual("D", "us-gaap:Assets", 200e6, 2015, known, instant=True),
             annual("D", "us-gaap:StockholdersEquity", 250e6, 2015, known, instant=True)]   # negative liabilities: kept
        a = [annual("A", "us-gaap:CostOfRevenue", 999e6, 2015, known),                       # A tags GrossProfit
             annual("A", "us-gaap:StockholdersEquity", 1e6, 2015, known, instant=True)]     # and Liabilities
        old = [annual("E", "us-gaap:GrossProfit", 50e6, 2012, "2013-02-20"),                 # stale: falls back
               annual("E", "us-gaap:Assets", 500e6, 2012, "2013-02-20", instant=True),
               annual("E", "us-gaap:Revenues", 90e6, 2015, known), annual("E", "us-gaap:CostOfRevenue", 40e6, 2015, known),
               annual("E", "us-gaap:Assets", 400e6, 2015, known, instant=True)]
        b.add_many(c + d + a + old)
        px = F.prices(b, STOCKS, self.asof())
        ff = F.fundamental_features(b, ["A", "C", "D", "E"], ["2016-03-31"], px.book)
        self.assertAlmostEqual(ff["gross_profitability"][0, 0], 0.17)            # GrossProfit first
        self.assertAlmostEqual(ff["leverage"][0, 0], 0.6)                        # Liabilities first
        self.assertAlmostEqual(ff["gross_profitability"][0, 1], (300 - 180) / 600)
        self.assertAlmostEqual(ff["leverage"][0, 1], 1 - 240 / 600)
        self.assertAlmostEqual(ff["gross_profitability"][0, 2], (100 - 70) / 200)  # the second cost concept
        self.assertAlmostEqual(ff["leverage"][0, 2], 1 - 250 / 200)
        self.assertAlmostEqual(ff["gross_profitability"][0, 3], (90 - 40) / 400)
        self.assertTrue(np.isnan(ff["leverage"][0, 3]) and ff["has_leverage"][0, 3] == 0.0)
        # Before the 10-K is known, nothing: the fallbacks read the same view.
        early = F.fundamental_features(b, ["C"], ["2016-01-29"], px.book)
        self.assertTrue(np.isnan(early["gross_profitability"][0, 0]) and np.isnan(early["leverage"][0, 0]))
        # The facts each value came from, for a forecast to cite.
        vals, facts = F.fundamentals_with_inputs(b.view(F.T("2016-03-31")), "C", px.book)
        self.assertEqual({f.id for f in facts}, {f.id for f in c})
        self.assertEqual(vals, F.fundamentals_at(b.view(F.T("2016-03-31")), "C", px.book))

    def test_fundamentals_carry_forward_only(self):
        b = self.bank
        b.add(annual("A", "us-gaap:Liabilities", 800e6, 2015, "2016-04-15", instant=True))   # restated mid-April
        fs = F.feature_set(b, STOCKS, self.grid(since="2016-01-01"), self.asof(), 5, cache=False)
        lev = fs.raw[:, fs.col("leverage")]
        a = fs.ticker == "A"
        by = dict(zip(fs.date[a], lev[a]))
        before = [d for d in by if "2016-04-01" <= d < "2016-05-01"]    # after the restatement, too
        after = [d for d in by if d >= "2016-05-01"]
        self.assertTrue(any(d > "2016-04-15" for d in before))
        self.assertTrue(before and after)
        self.assertTrue(all(abs(by[d] - 0.6) < 1e-12 for d in before))   # April's dates use March's month-end
        self.assertTrue(all(abs(by[d] - 0.8) < 1e-12 for d in after))    # from May, April's month-end
        april_end = max(d for d in DAYS if d.startswith("2016-04"))
        self.assertEqual(F.month_end_index(np.array(DAYS), self.i(april_end)), self.i("2016-03-31"))
        self.assertEqual(F.month_end_index(np.array(DAYS), self.i("2016-05-02")), self.i(april_end))
        # The carried values are the month-end view's own.
        px = F.prices(b, STOCKS, self.asof())
        me = F.fundamental_features(b, ["A"], ["2016-03-31", april_end], px.book)
        self.assertEqual(list(me["leverage"][:, 0]), [0.6, 0.8])


class FeatureSetTest(Synthetic):
    def test_samples_rows_and_drops(self):
        self.bank.add(snap("TWIN", DAYS, self.spy, known=DAYS[-1]))   # SPY itself: every excess is exactly 0
        g = self.grid()
        fs = F.feature_set(self.bank, [*self.tickers(), "TWIN", "NOPRICE"], g, self.asof(), 5, cache=False)
        self.assertEqual(fs.X.shape, (len(fs.y), len(F.FEATURES)))
        self.assertEqual(fs.names, F.NAMES)
        self.assertEqual(fs.groups, F.GROUPS)
        self.assertTrue(np.all(fs.entry > fs.date) and np.all(fs.exit > fs.entry))
        self.assertTrue(np.all((fs.excess > 0) == (fs.y == 1)) and np.all(fs.excess != 0))
        self.assertNotIn("TWIN", set(fs.ticker))
        self.assertGreater(fs.dropped["zero_excess"], 100)
        self.assertNotIn("NOPRICE", set(fs.ticker))
        self.assertGreater(fs.dropped["unlisted"], len(g))            # NOPRICE, LATE before its 252nd bar, ...
        self.assertEqual(fs.dropped["pending"], len(fs.pending["ticker"]))
        # Exit = h sessions after the entry, so the last two grid dates' windows are still open.
        self.assertEqual(set(fs.pending["date"]), {g[-1], g[-2]})
        self.assertEqual(fs.pending["X"].shape, (len(fs.pending["ticker"]), len(F.FEATURES)))
        self.assertGreater(fs.dropped["no_outcome"], 0)               # GAP: its exit close is stale
        ranked = [c for c, (n, grp, _) in enumerate(F.FEATURES) if grp != "R" and n not in F.INDICATORS]
        self.assertTrue(np.all(np.abs(fs.X[:, ranked]) <= 0.5))
        self.assertTrue(np.isin(fs.X[:, [fs.col(n) for n in F.INDICATORS]], [0.0, 1.0]).all())
        # One date: ranks over every listed stock, including those whose outcome is dropped later.
        d = g[150]
        px = F.prices(self.bank, fs.tickers, self.asof())
        i = px.index(d)
        r5 = F.stock_features(px, i)["r_5"]
        want = F.rank_unit(r5, px.listed(i))
        rows = fs.date == d
        for t, x in zip(fs.ticker[rows], fs.X[rows, fs.col("r_5")]):
            self.assertAlmostEqual(x, want[fs.tickers.index(t)], places=12)
        k = np.flatnonzero((fs.ticker == "A") & rows)[0]
        o = F.outcome(px.book, "A", d, 5)
        self.assertEqual((fs.entry[k], fs.exit[k]), (o["entry"], o["exit"]))
        self.assertAlmostEqual(fs.excess[k], o["excess"], places=14)
        a = px.adj[:, 0]
        self.assertAlmostEqual(fs.r_total[k], a[px.index(o["exit"])] / a[px.index(o["entry"])] - 1, places=10)
        self.assertTrue(np.all(fs.realised(g[100]) == (fs.exit <= g[100])))
        # R columns: z-scores over earlier grid dates only.
        reg = np.array([F.regime_features(px, px.index(x))["spy_r_21"] for x in g])
        z = F.zscore_earlier(reg)
        for gi in (10, 60, 120, 200):
            at = np.flatnonzero(fs.date == g[gi])
            if len(at):
                self.assertTrue(np.allclose(fs.X[at, fs.col("spy_r_21")], z[gi]))

    def test_a_fact_stamped_exactly_t_of_d_never_overlaps_its_window(self):
        g = self.grid()
        d = g[180]
        stamp = d + "T23:59:59Z"
        self.bank.add_many([trade("A", stamp, d, 2e6, accn="late-form-4"),
                            filing("A", "8-K", stamp, "2.02", "after-close-release")])
        fs = F.feature_set(self.bank, STOCKS, g, self.asof(), 5, cache=False)
        k = np.flatnonzero((fs.ticker == "A") & (fs.date == d))[0]
        self.assertAlmostEqual(fs.raw[k, fs.col("insider_buy_90d")], math.log1p(2e6), places=4)   # known on d
        entry_close_utc = pd.Timestamp(fs.entry[k] + " 16:00", tz="America/New_York").tz_convert("UTC")
        self.assertGreater(fs.entry[k], d)
        self.assertLess(pd.Timestamp(stamp), entry_close_utc)        # public before the window's first close
        # The release moves the entry session, so its reaction lies before the window, never inside it.
        e = events.event_index(np.array(DAYS), stamp)
        self.assertEqual(DAYS[e], fs.entry[k])

    def test_workers_and_cache(self):
        g = self.grid(since=DAYS[700])
        one = F.feature_set(self.bank, self.tickers(), g, self.asof(), 5, workers=1)
        two = F.feature_set(self.bank, self.tickers(), g, self.asof(), 5, workers=2, cache=False)
        for k in ("X", "excess", "exit", "date", "ticker"):
            np.testing.assert_array_equal(getattr(one, k), getattr(two, k))
        np.testing.assert_array_equal(one.raw, two.raw)
        hit = F.feature_set(self.bank, self.tickers(), g, self.asof(), 5)
        self.assertEqual(hit.seconds, 0.0)
        self.assertEqual(hit.key, one.key)
        for k in ("X", "raw", "y", "excess", "r_total", "close", "entry", "exit", "date", "ticker"):
            np.testing.assert_array_equal(getattr(hit, k), getattr(one, k))
        for k in one.pending:
            np.testing.assert_array_equal(hit.pending[k], one.pending[k])
        self.assertEqual(hit.dropped, one.dropped)
        self.assertTrue(list((self.bank.root.parent / "cache" / "forecast").glob("features-*.npz")))

    def test_cache_key_changes_with_the_code_and_a_backfilled_shard(self):
        g = self.grid(since=DAYS[1000])
        args = (self.tickers(), g, self.asof(), 5)
        k0 = F.cache_key(self.bank, *args)
        self.assertEqual(F.cache_key(self.bank, *args), k0)
        self.assertNotEqual(F.cache_key(self.bank, *args[:-1], 21)["sha"], k0["sha"])
        self.assertNotEqual(F.cache_key(self.bank, args[0], g[:-1], *args[2:])["sha"], k0["sha"])
        self.assertNotEqual(F.cache_key(self.bank, args[0], g, F.T(DAYS[-2]), 5)["sha"], k0["sha"])
        edited = os.path.join(self._tmp.name, "features_edited.py")
        shutil.copy(F.__file__, edited)
        with open(edited, "a") as fh:
            fh.write("\n# a change to the code\n")
        with mock.patch.object(F, "_SOURCE", F.Path(edited)):
            self.assertNotEqual(F.cache_key(self.bank, *args)["sha"], k0["sha"])
        fs0 = F.feature_set(self.bank, *args)
        d = g[5]
        before = fs0.raw[(fs0.ticker == "A") & (fs0.date == d), fs0.col("insider_buy_90d")][0]
        # A backfill: an old purchase, known long before d, appended to A's shard after the cache was written.
        self.bank.add(trade("A", norm_known_at(DAYS[1005]), DAYS[1004], 3e5, accn="backfill"))
        k1 = F.cache_key(self.bank, *args)
        self.assertNotEqual(k1["sha"], k0["sha"])
        self.assertNotEqual(k1["shards_sha"], k0["shards_sha"])
        fs1 = F.feature_set(self.bank, *args)
        self.assertGreater(fs1.seconds, 0.0)                         # recomputed, not read from the cache
        after = fs1.raw[(fs1.ticker == "A") & (fs1.date == d), fs1.col("insider_buy_90d")][0]
        self.assertEqual(before, 0.0)
        self.assertAlmostEqual(after, math.log1p(3e5), places=4)

    def test_a_prebuilt_price_panel_is_reused(self):
        g = self.grid(since=DAYS[1100])
        px = F.prices(self.bank, self.tickers(), self.asof())
        a = F.feature_set(self.bank, self.tickers(), g, self.asof(), 5, cache=False, px=px)
        b = F.feature_set(self.bank, self.tickers(), g, self.asof(), 5, cache=False)
        np.testing.assert_array_equal(a.X, b.X)
        with self.assertRaises(ValueError):
            F.feature_set(self.bank, STOCKS, g, self.asof(), 5, cache=False, px=px)


class LeakTest(Synthetic):
    """Appending bars dated after d to the same snapshots (older adjclose re-rounded within 1e-6), or facts
    known after T(d), changes no feature at d."""
    FIRST = 1100                         # the first book ends at DAYS[1099]

    def test_no_feature_at_d_changes(self):
        b = self.bank
        d = DAYS[self.FIRST - 1]
        first = F.T(d)
        g = make_grid(DAYS, d, 5, since=DAYS[300])
        self.assertEqual(g[-1], d)
        me = DAYS[F.month_end_index(np.array(DAYS), self.i(d))]
        b.add(filing("C", "8-K", first, "2.02", "evening-release"))   # stamped exactly T(d): its reaction is after d
        fs1 = F.feature_set(b, self.tickers(), g, first, 5, cache=False)
        px1 = F.prices(b, STOCKS, first)
        raw1, reg1 = F.stock_features(px1, self.i(d)), F.regime_features(px1, self.i(d))
        ev1 = F.event_features(b, STOCKS, d, px1.book)
        fund1 = F.fundamental_features(b, STOCKS, [me], px1.book)

        # Later: longer snapshots of every series (SPY and ^VIX too), older adjclose re-adjusted and re-rounded,
        # and facts known one second after T(d) of every kind the bank-based features read.
        self.write_prices(len(DAYS), scale=1 - 4e-7, known=DAYS[-1])
        after = norm_known_at(pd.Timestamp(first) + pd.Timedelta(seconds=1))
        b.add_many([
            trade("A", after, d, 9e6, accn="later-buy"), trade("B", after, d, 9e6, rel="director", accn="later-b"),
            filing("A", "8-K", after, "2.02", "later-8k"), filing("B", "8-K", after, "2.02", "later-8k-b"),
            annual("A", "us-gaap:Liabilities", 990e6, 2015, after, instant=True),
            annual("A", "us-gaap:Assets", 5e9, 2015, after, instant=True),
            annual("A", "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding", 9e9, 2015, after),
            annual("A", "us-gaap:GrossProfit", 1e9, 2015, after),
            annual("A", "us-gaap:NetCashProvidedByUsedInOperatingActivities", 7e8, 2015, after),
            annual("A", NI, -5e8, 2015, after), annual("B", "us-gaap:Revenues", 1e9, 2015, after),
            # what the gross-profit and leverage fallbacks read, for a stock that had none of it at d
            annual("B", "us-gaap:CostOfRevenue", 4e8, 2015, after), annual("B", "us-gaap:Assets", 2e9, 2015, after,
                                                                               instant=True),
            annual("B", "us-gaap:StockholdersEquity", 5e8, 2015, after, instant=True),
            Fact(entity="A", attr="split", value=2.0, as_of=after[:10], known_at=after, tier="market",
                 source={"name": "test", "url": "https://example.test/split"}),
        ])
        b.add_many(fiscal_year("A", EPS, 2016, [9.0, 9.0, 9.0, 9.0], [after] * 4))
        later = F.T(DAYS[-1])
        px2 = F.prices(b, STOCKS, later)
        self.assertGreater(len(px2.dates), len(px1.dates))
        self.assertFalse(np.array_equal(px2.adj[:self.FIRST], px1.adj))      # re-rounded, not the same numbers
        raw2, reg2 = F.stock_features(px2, self.i(d)), F.regime_features(px2, self.i(d))
        for n in raw1:
            # Re-rounding moves a $2 stock's skew_63 by up to ~1e-4; a leak moves values by far more.
            np.testing.assert_allclose(raw2[n], raw1[n], rtol=1e-4, atol=1e-4, err_msg=n)
        for n in reg1:
            self.assertAlmostEqual(reg2[n], reg1[n], places=7, msg=n)
        ev2 = F.event_features(b, STOCKS, d, px2.book)
        fund2 = F.fundamental_features(b, STOCKS, [me], px2.book)
        for n in ev1:
            np.testing.assert_allclose(ev2[n], ev1[n], rtol=1e-6, atol=1e-9, err_msg=n)
        for n in fund1:
            np.testing.assert_allclose(fund2[n], fund1[n], rtol=1e-6, atol=1e-9, err_msg=n)
        self.assertTrue(np.isfinite(fund1["leverage"][0, 0]) and ev1["has_earnings_recent"].sum() == 0)

        # Every column of every (date, stock) row of the feature set, realised or still pending in the first book.
        # The second grid runs on past d, so the later facts and bars are in its reach.
        g2 = g + DAYS[self.FIRST - 1 + 5::5]
        fs2 = F.feature_set(b, self.tickers(), g2, later, 5, cache=False)

        def rows(fs):
            out = {(x, t): (a, r) for x, t, a, r in zip(fs.date, fs.ticker, fs.X, fs.raw)}
            p = fs.pending
            out.update({(x, t): (a, r) for x, t, a, r in zip(p["date"], p["ticker"], p["X"], p["raw"])})
            return out
        r1, r2 = rows(fs1), rows(fs2)
        self.assertGreater(len(fs1.pending["ticker"]), 0)
        r2 = {k: v for k, v in r2.items() if k[0] <= d}
        self.assertLessEqual(set(r1), set(r2))
        self.assertGreater(fs2.raw[(fs2.ticker == "A") & (fs2.date > d), F.NAMES.index("insider_buy_90d")].max(), 15)
        common = sorted(r1)
        self.assertIn((d, "A"), common)
        self.assertGreater(len(common), 1000)
        x1, x2 = (np.array([r[k][0] for k in common]) for r in (r1, r2))
        w1, w2 = (np.array([r[k][1] for k in common], dtype=float) for r in (r1, r2))
        for c, n in enumerate(F.NAMES):
            np.testing.assert_allclose(x2[:, c], x1[:, c], atol=1e-6, err_msg=n)
            np.testing.assert_allclose(w2[:, c], w1[:, c], rtol=1e-4, atol=1e-4, equal_nan=True, err_msg=n)
        self.assertTrue(np.isfinite(w1[:, F.NAMES.index("vix")]).any())
