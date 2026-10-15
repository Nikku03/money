import argparse
import io
import json
import os
from contextlib import redirect_stdout
from unittest import mock

import pandas as pd

from brain import check as invariants
from brain import config, watch
from brain.bank import Fact
from tests.util import TempHome, filed, price_snapshot

# Made-up companies and people; the facts mirror the shapes in docs/EXPANSION.md, not anyone's trades.
DAYS = pd.bdate_range("2026-03-02", periods=40).strftime("%Y-%m-%d").tolist()
CAM = "TFL.00001.00001"
SITE = {"name": "Plant One", "ticker": "X", "cameras": [CAM]}
AREA = {"name": "TEST-PORT", "bbox": [10.0, 20.0, 11.0, 21.0]}


def at(day: str, hhmm: str = "12:00") -> str:
    return f"{day}T{hhmm}:00Z"


def filing(t, form, known, items="", accn=None):
    return filed(t, "filing", form, known[:10], known, accn=accn or known, **({"items": items} if items else {}))


def trade(t, shares, price, known, accn="0001-26-000001", code="P", rel="director", owner="DOE JANE", title=None,
          day=None, ownership="D", lot=None):
    return Fact(entity=t, attr="insider.trade", value=shares, unit="shares", as_of=day or known[:10], known_at=known,
                tier="filed", source={"name": "SEC Form 4", "url": f"https://example.test/{accn}-index.htm"},
                context={"code": code, "price": price, "value_usd": round(shares * price, 2), "owner": owner,
                         "owner_cik": "0000000001", "relationship": rel, "title": title, "accn": accn,
                         "ownership": ownership, **({"lot": lot} if lot else {})})


def sample(entity, attr, value, known, **ctx):
    return Fact(entity=entity, attr=attr, value=value, as_of=known[:10], known_at=known, tier="reported",
                source={"name": "test feed", "url": "https://example.test/feed"}, context=ctx)


def ships(value, known, window_s=300, **extra):
    return sample("AREA.TEST-PORT", "ships.count", value, known, window_s=window_s, bbox=AREA["bbox"], cargo=value,
                  tanker=0, other=0, anchored=0, moored=0, underway=value, **extra)


def vehicles(value, known, **extra):
    return sample(f"CAM.{CAM}", "camera.vehicles", value, known, car=value, truck=0, bus=0, motorcycle=0,
                  provider="TFL", name="Test Rd", lat=51.5, lon=-0.1, detector="yolox-2022nov", threshold=0.5, **extra)


def flight(t, known, reg="N100XX", hex_="a00001"):
    return sample(t, "aircraft.flight", "KSJC", known, hex=hex_, reg=reg, **{"from": "KTEB"}, to="KSJC",
                  from_name="Teterboro", to_name="San Jose Intl", departed=at(known[:10], "08:00"),
                  arrived=known)


def headline(t, title, known, url):
    return Fact(entity=t, attr="headline", value=title, as_of=known[:10], known_at=known, tier="reported",
                source={"name": "news.example", "url": url}, context={"feed": "yahoo-rss"})


class WatchTest(TempHome):
    def setUp(self):
        super().setUp()
        self.log = []
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for k in ("BRAIN_NTFY_URL", "AISSTREAM_API_KEY", "SEC_USER_AGENT"):
            os.environ.pop(k, None)

    def tearDown(self):
        self.env.stop()
        super().tearDown()

    def watching(self, **w):
        watch.save({"tickers": [], "aircraft_of": [], "areas": [], "sites": [], **w})

    def check(self, now, **kw):
        return watch.check(self.bank, now=now, log=self.log.append, **kw)

    def rules(self, alerts):
        return sorted(f.context["rule"] for f in alerts)

    def assert_cited(self, alerts):
        for f in alerts:
            self.assertEqual(f.tier, "derived")
            self.assertTrue(f.inputs, f.value)
            for ref in f.inputs:
                g = self.bank.get(ref)
                self.assertIsNotNone(g, ref)
                self.assertLessEqual(g.known_at, f.known_at)
        errors, _ = invariants.run(self.bank)
        self.assertEqual(errors, [])


class WatchlistTest(WatchTest):
    def test_defaults_when_missing(self):
        w = watch.load()
        self.assertEqual((w["tickers"], w["aircraft_of"], w["areas"], w["sites"]), ([], [], [], []))
        self.assertEqual(w["rules"], watch.DEFAULT_RULES)

    def test_add_remove_round_trip(self):
        watch.add("ticker", "aapl")
        watch.add("ticker", "AAPL")
        watch.add("aircraft", "aapl")
        watch.add("area", "la-long-beach")
        watch.add("area", "my-port", bbox=[1, 2, 3, 4], seconds=120)
        watch.add("site", "Plant One", ticker="x", cameras=[CAM])
        w = watch.add("site", "plant one", cameras=["CALTRANS.D04.tv102", CAM.lower()])
        self.assertEqual(w["tickers"], ["AAPL"])
        self.assertEqual(w["aircraft_of"], ["AAPL"])
        self.assertEqual(w["areas"], [{"name": "LA-LONG-BEACH"}, {"name": "MY-PORT", "bbox": [1.0, 2.0, 3.0, 4.0],
                                                                  "seconds": 120}])
        self.assertEqual(w["sites"], [{"name": "Plant One", "ticker": "X", "cameras": [CAM, "CALTRANS.D04.tv102"]}])
        on_disk = json.loads((config.home() / "watchlist.json").read_text())
        self.assertEqual(on_disk["aircraft_of"], ["AAPL"])  # the shape aviation.watched() and shipping.areas() read
        self.assertEqual(watch.watched_entities(w), ["AAPL", "AREA.LA-LONG-BEACH", "AREA.MY-PORT", f"CAM.{CAM}",
                                                     "CAM.CALTRANS.D04.TV102"])
        w = watch.remove("site", "PLANT ONE")
        w = watch.remove("area", "my-port")
        w = watch.remove("ticker", "aapl")
        self.assertEqual((w["tickers"], w["sites"], [a["name"] for a in w["areas"]]), ([], [], ["LA-LONG-BEACH"]))
        with self.assertRaises(KeyError):
            watch.remove("ticker", "AAPL")

    def test_refusals(self):
        with self.assertRaises(ValueError):
            watch.add("area", "NOWHERE")                          # not built in and no bbox
        with self.assertRaises(ValueError):
            watch.add("area", "BOX", bbox=[3, 2, 1, 4])           # lat_min > lat_max
        with self.assertRaises(ValueError):
            watch.add("site", "Lobby", cameras=["HOMECAM.1"])     # not an official open provider
        with self.assertRaises(ValueError):
            watch.add("site", "Empty")                            # no camera
        with self.assertRaises(ValueError):
            watch.add("plane", "X")
        self.assertEqual(watch.load()["sites"], [])

    def test_unknown_keys_and_rules_survive(self):
        (config.home() / "watchlist.json").write_text(json.dumps({"tickers": ["x"], "note": "mine",
                                                                 "rules": {"zscore": 4}}))
        watch.add("ticker", "Y")
        w = json.loads((config.home() / "watchlist.json").read_text())
        self.assertEqual((w["note"], w["tickers"], w["rules"]["zscore"], w["rules"]["price_move"]),
                         ("mine", ["X", "Y"], 4, 0.05))

    def test_corrupt_watchlist_is_an_error_not_empty(self):
        (config.home() / "watchlist.json").write_text("{nope")
        with self.assertRaises(ValueError):
            watch.load()

    def test_headlines_are_off_for_a_new_watchlist_and_kept_in_an_old_one(self):
        p = config.home() / "watchlist.json"
        self.assertFalse(watch.load()["rules"]["headlines"])
        watch.add("ticker", "X")
        self.assertIs(json.loads(p.read_text())["rules"]["headlines"], False)  # saved as off
        p.write_text(json.dumps({"tickers": ["X"], "rules": {"zscore": 4}}))  # written before the default changed
        self.assertIs(watch.load()["rules"]["headlines"], True)
        watch.add("ticker", "Y")
        self.assertIs(json.loads(p.read_text())["rules"]["headlines"], True)  # and it stays on when saved
        p.write_text(json.dumps({"tickers": ["X"]}))
        self.assertIs(watch.load()["rules"]["headlines"], True)
        p.write_text(json.dumps({"tickers": ["X"], "rules": {"headlines": False}}))
        self.assertIs(watch.load()["rules"]["headlines"], False)

    def test_fields_of_the_wrong_shape_are_errors_naming_the_field(self):
        p = config.home() / "watchlist.json"
        cases = {'{"tickers": "AAPL"}': "tickers must be a list of strings",
                 '{"tickers": [5]}': "tickers must be a list of strings",
                 '{"tickers": ["A;B"]}': "tickers: not a ticker",
                 '{"tickers": ["AAPL"], "areas": 5}': "areas must be a list",
                 '{"areas": [7]}': "areas: each entry is a name",
                 '{"areas": [{"name": "BOX", "bbox": [3, 2, 1, 4]}]}': "areas: BOX: bbox",
                 '{"sites": {"name": "x"}}': "sites must be a list",
                 '{"sites": ["Plant"]}': "sites: each entry is",
                 '{"sites": [{"name": "Plant", "cameras": "TFL.1"}]}': "sites: Plant: cameras must be a list",
                 '{"sites": [{"name": "Lobby", "cameras": ["HOMECAM.1"]}]}': "sites: Lobby: 'HOMECAM.1'",
                 '{"rules": []}': "rules must be an object",
                 '{"rules": {"zscore": "3"}}': "rules.zscore must be a number",
                 '{"rules": {"forms": "8-K"}}': "rules.forms must be a list",
                 '{"rules": {"headlines": "on"}}': "rules.headlines must be true or false"}
        for text, words in cases.items():
            with self.subTest(text=text):
                p.write_text(text)
                with self.assertRaises(ValueError) as e:
                    watch.load()
                self.assertIn(str(p), str(e.exception))
                self.assertIn(words, str(e.exception))
                self.assertIn("fix or remove it", str(e.exception))
                with self.assertRaises(ValueError):
                    watch.add("ticker", "MSFT")
                self.assertEqual(p.read_text(), text)  # never saved over
        p.write_text(json.dumps({"tickers": [" aapl ", ""], "rules": {"zscore": None, "headlines": False}, "x": 1}))
        w = watch.load()
        self.assertEqual((w["tickers"], w["rules"]["zscore"], w["x"]), (["AAPL"], None, 1))


class PriceRuleTest(WatchTest):
    def test_move_alerts_once_per_bar_and_cites_the_snapshot(self):
        self.watching(tickers=["X"])
        closes = [100.0] * 29 + [106.0]
        snap = price_snapshot("X", DAYS[:30], closes)
        self.bank.add(snap)
        out = self.check(at(DAYS[30], "01:00"))
        self.assertEqual(self.rules(out), ["price_move"])
        a = out[0]
        self.assertEqual((a.entity, a.as_of, a.context["severity"], a.inputs), ("X", DAYS[29], "notice", (snap.ref,)))
        self.assertIn("+6.0%", a.value)
        self.assert_cited(out)
        self.assertEqual(self.check(at(DAYS[30], "02:00")), [])
        # the same bars re-fetched as a different snapshot: no second alert for that day
        self.bank.add(price_snapshot("X", DAYS[:30], closes, adj=[c * 0.99 for c in closes]))
        self.assertEqual(self.check(at(DAYS[30], "03:00")), [])
        # a new day falls 12%: high
        self.bank.add(price_snapshot("X", DAYS[:31], closes + [106 * 0.88]))
        out = self.check(at(DAYS[31], "01:00"))
        self.assertEqual([(f.as_of, f.context["severity"]) for f in out], [(DAYS[30], "high")])

    def test_small_moves_and_old_bars_do_not_alert(self):
        self.watching(tickers=["X"])
        closes = [100.0] * 20 + [110.0] + [110.0] * 9   # the jump is weeks before the check
        self.bank.add(price_snapshot("X", DAYS[:30], [c * (1.01 if i == 29 else 1) for i, c in enumerate(closes)]))
        self.assertEqual(self.check(at(DAYS[30], "01:00")), [])

    def test_a_close_is_not_seen_before_its_day_ends(self):
        self.watching(tickers=["X"])
        self.bank.add(price_snapshot("X", DAYS[:30], [100.0] * 29 + [90.0]))
        self.assertEqual(self.check(at(DAYS[29], "21:00")), [])   # known_at is the end of that UTC day
        self.assertEqual(self.rules(self.check(at(DAYS[30], "00:30"))), ["price_move"])


class FilingAndInsiderRuleTest(WatchTest):
    def test_watched_forms_alert_and_earnings_releases_are_left_to_the_earnings_rule(self):
        self.watching(tickers=["X"])
        k = at(DAYS[10], "14:00")
        self.bank.add_many([
            filing("X", "8-K", k, "5.02,9.01", "a1"),
            filing("X", "10-Q", k, "", "a2"),
            filing("X", "4", k, "", "a3"),                         # not a watched form
            filing("X", "SCHEDULE 13D/A", k, "", "a4"),            # EDGAR's newer name for SC 13D, amended
            filing("X", "8-K", k, "1.03", "a5"),                   # bankruptcy: high
            filing("X", "8-K", k, "2.02,9.01", "a6"),              # an earnings release
            filing("X", "10-K", at(DAYS[1]), "", "a7"),            # before the first check's look-back
        ])
        out = [f for f in self.check(at(DAYS[10], "18:00")) if f.context["rule"] == "filing"]
        self.assertEqual(sorted(f.context["form"] for f in out), ["10-Q", "8-K", "8-K", "SCHEDULE 13D/A"])
        self.assertIn("5.02 officer or director change", next(f.value for f in out if "5.02" in f.value))
        self.assertEqual(next(f.context["severity"] for f in out if "1.03" in f.value), "high")
        self.assert_cited(out)
        self.assertEqual(self.check(at(DAYS[11], "18:00")), [])

    def test_insider_buys_by_officers_and_directors_summed_per_filing(self):
        self.watching(tickers=["X"])
        k = at(DAYS[10], "21:00")
        self.bank.add_many([
            trade("X", 1000, 60.0, k, "f1", day=DAYS[8]), trade("X", 1000, 61.0, k, "f1", day=DAYS[9]),  # $121k
            trade("X", 10000, 50.0, k, "f2", rel="ten_percent_owner"),     # not an officer or director
            trade("X", 10000, 50.0, k, "f3", code="S", rel="officer"),     # a sale
            trade("X", 100, 50.0, k, "f4", rel="officer"),                 # $5k: below the threshold
            trade("X", 5000, 50.0, k, "f5", rel="director,ten_percent_owner", owner="ROE RICH", title="Chair"),
        ])
        out = self.check(at(DAYS[11], "12:00"))
        self.assertEqual(self.rules(out), ["insider_buy", "insider_buy"])
        by = {f.context["keys"][0]: f for f in out}
        self.assertEqual(by["insider:f1"].context["value_usd"], 121000.0)
        self.assertEqual(len(by["insider:f1"].inputs), 2)
        self.assertIn(f"on {DAYS[8]} to {DAYS[9]}", by["insider:f1"].value)
        self.assertEqual(by["insider:f5"].context["severity"], "high")    # $250k >= 2x $100k
        self.assertIn("ROE RICH (director, ten_percent_owner, Chair)", by["insider:f5"].value)
        self.assert_cited(out)
        # a later copy of the same filing (the quarterly data set) does not alert again
        self.bank.add(trade("X", 1000, 60.0, at(DAYS[12], "23:59"), "f1", day=DAYS[8]))
        self.assertEqual(self.check(at(DAYS[13], "12:00")), [])

    def test_repeated_identical_lots_each_count(self):
        # One filing's six purchases at one price: 3, 1 and 2 shares held indirectly, 3 directly, then 1 and 2
        # indirectly again. The repeats are distinct transactions (lot 2), so 12 shares, not 9.
        self.watching(tickers=["X"], rules={"insider_buy_usd": 6000})
        k, p = at(DAYS[10], "21:00"), 576.33
        self.bank.add_many([trade("X", n, p, k, "f9", day=DAYS[9], ownership=o, lot=lot)
                            for n, o, lot in ((3, "I", None), (1, "I", None), (2, "I", None), (3, "D", None),
                                              (1, "I", 2), (2, "I", 2))])
        out = self.check(at(DAYS[11], "12:00"))
        self.assertEqual(self.rules(out), ["insider_buy"])
        self.assertEqual(out[0].context["value_usd"], 6915.96)
        self.assertEqual(len(out[0].inputs), 6)
        self.assertIn("bought 12 shares for $6,916", out[0].value)

    def test_a_filing_read_live_and_in_bulk_in_one_check_counts_once(self):
        # Both copies are new in the same check: the sum is one copy's, lots included, never both.
        self.watching(tickers=["X"])
        live, bulk = at(DAYS[10], "14:05"), at(DAYS[10], "23:59")
        for known in (live, bulk):
            self.bank.add_many([trade("X", 500, 60.0, known, "f7", day=DAYS[9]),               # $60k in all
                                trade("X", 500, 60.0, known, "f7", day=DAYS[9], lot=2),
                                trade("X", 1000, 60.0, known, "f6", day=DAYS[9]),              # $120k in all
                                trade("X", 1000, 60.0, known, "f6", day=DAYS[9], lot=2)])
        out = self.check(at(DAYS[11], "12:00"))
        self.assertEqual([f.context["keys"] for f in out], [["insider:f6"]])  # f7 stays below $100k
        self.assertEqual(out[0].context["value_usd"], 120000.0)
        self.assertEqual(out[0].context["severity"], "notice")
        self.assertIn("bought 2,000 shares for $120,000", out[0].value)


class EarningsRuleTest(WatchTest):
    E = 30

    def prices(self, t, upto):
        self.bank.add(price_snapshot("SPY", DAYS[:upto + 1], [100.0] * (upto + 1)))
        closes = [50.0 if i < self.E else 55.0 for i in range(upto + 1)]  # +10% on the release day
        self.bank.add(price_snapshot(t, DAYS[:upto + 1], closes))

    def release(self, t):
        f = filing(t, "8-K", at(DAYS[self.E], "13:00"), "2.02,9.01", f"r-{t}")  # 09:00 New York: before the open
        self.bank.add(f)
        return f

    def test_release_then_its_reaction(self):
        self.watching(tickers=["X"], rules={"price_move": None})
        self.prices("X", self.E - 1)
        ev = self.release("X")
        out = self.check(at(DAYS[self.E], "20:00"))
        self.assertEqual([(f.context["rule"], f.context["severity"], f.context["car_3d"]) for f in out],
                         [("earnings", "notice", None)])
        self.assertEqual(out[0].inputs, (ev.ref,))
        self.prices("X", self.E + 1)                     # day +1 has closed: the 3-day window is complete
        out = self.check(at(DAYS[self.E + 2], "01:00"))
        self.assertEqual([(f.context["rule"], f.context["severity"]) for f in out], [("earnings", "high")])
        self.assertAlmostEqual(out[0].context["car_3d"], 0.10, places=6)
        self.assertEqual(out[0].context["keys"], ["earnings-car:r-X"])
        self.assertEqual(len(out[0].inputs), 3)          # the release and both price snapshots
        self.assertIn("+10.0% vs SPY", out[0].value)
        self.assert_cited(out)
        self.assertEqual(self.check(at(DAYS[self.E + 3], "01:00")), [])

    def test_a_release_first_seen_with_its_reaction_alerts_once(self):
        self.watching(tickers=["Y"], rules={"price_move": None})
        self.prices("Y", self.E + 1)
        self.release("Y")
        out = self.check(at(DAYS[self.E + 2], "01:00"))
        self.assertEqual([(f.context["rule"], f.context["severity"]) for f in out], [("earnings", "high")])
        self.assertEqual(out[0].context["keys"], ["earnings:r-Y", "earnings-car:r-Y"])
        self.assertIn("3-day reaction +10.0% vs SPY", out[0].value)
        self.assertEqual(self.check(at(DAYS[self.E + 2], "02:00")), [])

    def test_a_quiet_reaction_stays_a_notice(self):
        self.watching(tickers=["Z"], rules={"price_move": None, "earnings_car": 0.5})
        self.prices("Z", self.E + 1)
        self.release("Z")
        out = self.check(at(DAYS[self.E + 2], "01:00"))
        self.assertEqual([f.context["severity"] for f in out], ["notice"])


class AltDataRuleTest(WatchTest):
    def test_ship_counts_need_ten_prior_samples(self):
        self.watching(areas=[AREA])
        base = [100, 102, 98, 101, 99, 103, 97, 100, 102]          # nine priors
        self.bank.add_many([ships(v, at(DAYS[0], f"{h:02d}:00")) for h, v in enumerate(base)])
        self.bank.add(ships(160, at(DAYS[0], "10:00")))
        self.assertEqual(self.check(at(DAYS[0], "11:00")), [])     # only nine before it
        self.bank.add(ships(170, at(DAYS[0], "12:00")))             # ten before it now (the 160 is one of them)
        self.bank.add(ships(400, at(DAYS[0], "13:00"), window_s=60))  # another window length: not comparable
        out = self.check(at(DAYS[0], "14:00"))
        self.assertEqual([(f.entity, f.context["rule"]) for f in out], [("AREA.TEST-PORT", "ships")])
        self.assertEqual(len(out[0].inputs), 11)
        self.assertGreater(out[0].context["z"], 3)
        self.assert_cited(out)
        self.assertEqual(self.check(at(DAYS[0], "15:00")), [])

    def test_ship_counts_on_another_basis_are_not_compared(self):
        # Samples written before the count became commercial-only (no "basis") counted every typed vessel:
        # a commercial-only count after them is no drop, and alerts only against samples on its own basis.
        self.watching(areas=[AREA])
        self.bank.add_many([ships(v, at(DAYS[0], f"{h:02d}:00")) for h, v in enumerate(
            [200, 204, 198, 201, 199, 203, 197, 200, 202, 201, 199])])
        self.bank.add(ships(100, at(DAYS[0], "12:00"), basis="cargo+tanker"))
        self.assertEqual(self.check(at(DAYS[0], "13:00")), [])  # 11 older samples, none on its basis
        new = [100, 102, 98, 101, 99, 103, 97, 100, 102]
        self.bank.add_many([ships(v, at(DAYS[1], f"{h:02d}:00"), basis="cargo+tanker") for h, v in enumerate(new)])
        self.bank.add(ships(160, at(DAYS[1], "10:00"), basis="cargo+tanker"))
        out = self.check(at(DAYS[1], "11:00"))
        self.assertEqual([f.context["rule"] for f in out], ["ships"])
        self.assertEqual(len(out[0].inputs), 11)  # the sample and its ten priors on the same basis
        self.assertIn("commercial vessels", out[0].value)

    def test_camera_spike_names_the_site_and_flat_history_says_nothing(self):
        self.watching(sites=[SITE])
        quiet = [5, 6, 4, 5, 7, 5, 6, 4, 5, 6, 5, 4]
        self.bank.add_many([vehicles(v, at(DAYS[1], f"{h:02d}:00")) for h, v in enumerate(quiet)])
        self.bank.add(vehicles(40, at(DAYS[1], "13:00")))
        out = self.check(at(DAYS[1], "14:00"))
        self.assertEqual([(f.entity, f.context["rule"], f.context["severity"]) for f in out],
                         [(f"CAM.{CAM}", "camera", "high")])
        self.assertIn("(Plant One, X): 40 vehicles", out[0].value)
        self.watching(sites=[{**SITE, "cameras": ["TFL.00001.00002"]}])
        cam2 = "CAM.TFL.00001.00002"
        self.bank.add_many([Fact(**{**vehicles(3, at(DAYS[2], f"{h:02d}:00")).content(), "entity": cam2})
                            for h in range(12)])
        self.bank.add(Fact(**{**vehicles(9, at(DAYS[2], "13:00")).content(), "entity": cam2}))
        self.assertEqual(self.check(at(DAYS[2], "14:00")), [])     # no spread among the priors: no z-score

    def test_camera_counts_from_another_decode_are_not_compared(self):
        from brain import vision
        self.watching(sites=[SITE])
        quiet = [5, 6, 4, 5, 7, 5, 6, 4, 5, 6, 5, 4]  # read before the decode was recorded
        self.bank.add_many([vehicles(v, at(DAYS[1], f"{h:02d}:00")) for h, v in enumerate(quiet)])
        self.bank.add(vehicles(40, at(DAYS[1], "13:00"), decode=vision.DECODE))
        self.assertEqual(self.check(at(DAYS[1], "14:00")), [])  # no earlier sample read the same way
        self.bank.add_many([vehicles(v, at(DAYS[2], f"{h:02d}:00"), decode=vision.DECODE) for h, v in enumerate(quiet)])
        self.bank.add(vehicles(40, at(DAYS[2], "13:00"), decode=vision.DECODE))
        out = self.check(at(DAYS[2], "14:00"))
        self.assertEqual([f.context["rule"] for f in out], ["camera"])
        self.assertEqual(len(out[0].inputs), 14)  # the sample, the first 40 and the 12 quiet ones read the same way

    def test_landings_alert_and_opted_out_tails_do_not(self):
        self.watching(aircraft_of=["X"])
        self.bank.add_many([flight("X", at(DAYS[3], "15:00")), flight("X", at(DAYS[3], "16:00"), "N200YY", "a00002"),
                            flight("X", at(DAYS[3], "16:30"), "N300ZZ", "a00003")])
        (config.home() / "aviation_optout.txt").write_text("N200YY  # asked not to be tracked\n")
        (config.home() / "aviation_state.json").write_text(json.dumps({"aircraft": {}, "blocked": {"A00003": "LADD"}}))
        out = self.check(at(DAYS[3], "17:00"))
        self.assertEqual([f.context["rule"] for f in out], ["aircraft"])
        self.assertIn("N100XX landed at KSJC (San Jose Intl) from KTEB (Teterboro)", out[0].value)
        self.assertEqual((out[0].context["reg"], out[0].context["hex"]), ("N100XX", "a00001"))
        self.assert_cited(out)

    def test_alerts_about_a_tail_opted_out_later_are_not_listed(self):
        self.watching(aircraft_of=["X"])
        self.bank.add_many([flight("X", at(DAYS[3], "15:00")), flight("X", at(DAYS[3], "16:00"), "N200YY", "a00002")])
        by = {f.context["reg"]: f for f in self.check(at(DAYS[3], "17:00"))}
        bare = lambda f, **kw: Fact(**{**f.content(), "known_at": at(DAYS[3], "17:01"), **kw, "context": {  # noqa: E731
            **{k: v for k, v in f.context.items() if k not in ("reg", "hex")}, **kw.pop("context", {})}})
        # alerts written before alerts named their aircraft: one found by its rule key, one by the flight it cites
        self.bank.add_many([bare(by["N100XX"]), bare(by["N200YY"], context={"keys": ["aircraft-old"]})])
        self.assertEqual(len(watch.alerts(self.bank)), 4)
        (config.home() / "aviation_optout.txt").write_text("N100XX\n")  # by hand: nothing withdrawn yet
        shown = watch.alerts(self.bank)
        self.assertEqual(len(shown), 2)
        self.assertNotIn("N100XX", " ".join(f.value for f in shown))
        (config.home() / "aviation_state.json").write_text(json.dumps({"blocked": {"a00002": "FAA LADD/PIA"}}))
        self.assertEqual(watch.alerts(self.bank), [])
        self.assertEqual(watch.unacked(self.bank), [])
        self.assertEqual(watch.ack_all(self.bank), 0)

    def test_headlines_are_off_by_default_and_info_when_on(self):
        self.watching(tickers=["X"])                                 # a new watchlist: headlines off
        self.bank.add(headline("X", "X opens a plant", at(DAYS[4], "09:00"), "https://news.example/1"))
        self.assertEqual(self.check(at(DAYS[4], "10:00")), [])
        self.watching(tickers=["X"], rules={"headlines": True})  # python -m brain watch rule headlines on
        out = self.check(at(DAYS[4], "11:00"))
        self.assertEqual([(f.context["rule"], f.context["severity"]) for f in out], [("headline", "info")])
        self.watching(tickers=["X"], rules={"headlines": False})
        self.bank.add(headline("X", "X again", at(DAYS[4], "11:30"), "https://news.example/2"))
        self.assertEqual(self.check(at(DAYS[4], "12:00")), [])


class AckAndStateTest(WatchTest):
    def setUp(self):
        super().setUp()
        self.watching(tickers=["X"], rules={"headlines": True})
        self.bank.add_many([headline("X", f"story {i}", at(DAYS[5], f"0{i}:00"), f"https://news.example/{i}")
                            for i in range(1, 4)])
        self.first = self.check(at(DAYS[5], "12:00"))

    def test_acknowledged_alerts_stay_acknowledged(self):
        self.assertEqual(len(self.first), 3)
        one = watch.ack(self.bank, self.first[0].id[:8])
        self.assertEqual(one.id, self.first[0].id)
        self.assertEqual({f.id for f in watch.unacked(self.bank)}, {f.id for f in self.first[1:]})
        self.bank.add(headline("X", "story 4", at(DAYS[5], "13:00"), "https://news.example/4"))
        new = self.check(at(DAYS[5], "14:00"))
        self.assertEqual(len(new), 1)
        self.assertIn(one.id, watch.acked())
        self.assertEqual(watch.ack_all(self.bank), 3)
        self.assertEqual(watch.unacked(self.bank), [])
        self.assertEqual(self.check(at(DAYS[6], "14:00")), [])
        self.assertEqual(len(watch.alerts(self.bank)), 4)
        self.assertEqual([f.known_at for f in watch.alerts(self.bank)][0], at(DAYS[5], "14:00"))  # newest first

    def test_ack_refusals(self):
        with self.assertRaises(KeyError):
            watch.ack(self.bank, "ffffffff")
        with self.assertRaises(KeyError):
            watch.ack(self.bank, "zz")
        f = self.first[0]
        self.assertEqual(watch.ack(self.bank, f"{f.entity}/{f.id}").id, f.id)  # a full ref works too

    def test_alerts_of_a_removed_ticker_are_still_listed(self):
        watch.remove("ticker", "X")
        self.assertEqual(len(watch.alerts(self.bank)), 3)

    def test_a_state_file_of_the_wrong_shape_is_an_error_naming_the_field(self):
        p = config.home() / "alerts_state.json"
        for text, words in (("[]", "must hold a JSON object"), ('{"acked": 5}', "acked must be a list of strings"),
                            ('{"acked": [1]}', "acked must be a list of strings"),
                            ('{"entities": "X"}', "entities must be a list of strings"),
                            ('{"last_check": 5}', "last_check must be a time"),
                            ('{"last_check": "soon"}', "last_check 'soon' is not a time")):
            with self.subTest(text=text):
                p.write_text(text)
                for fn in (watch.acked, lambda: watch.alerts(self.bank), lambda: watch.ack_all(self.bank),
                           lambda: self.check(at(DAYS[6], "12:00"))):
                    with self.assertRaises(ValueError) as e:
                        fn()
                    self.assertIn(str(p), str(e.exception))
                    self.assertIn(words, str(e.exception))
                self.assertEqual(p.read_text(), text)

    def test_explicit_since_and_state(self):
        state = json.loads((config.home() / "alerts_state.json").read_text())
        self.assertEqual((state["last_check"], state["entities"]), (at(DAYS[5], "12:00"), ["X"]))
        self.assertEqual(self.check(at(DAYS[5], "12:30"), since="2000-01-01"), [])  # keys already used

    def test_one_failing_rule_does_not_stop_the_others(self):
        self.bank.add(headline("X", "story 9", at(DAYS[5], "15:00"), "https://news.example/9"))
        def _price_move(c, t):
            raise RuntimeError("boom")

        with mock.patch.object(watch, "TICKER_RULES", (_price_move, watch._headline)):
            out = self.check(at(DAYS[5], "16:00"))
        self.assertEqual(self.rules(out), ["headline"])
        self.assertTrue(any("price_move on X failed" in m for m in self.log))


class NotifyTest(WatchTest):
    def alerts(self):
        self.watching(tickers=["X"], rules={"headlines": True})
        self.bank.add(headline("X", "story", at(DAYS[5], "01:00"), "https://news.example/1"))
        self.bank.add(price_snapshot("X", DAYS[:6], [100.0] * 5 + [80.0]))
        return self.check(at(DAYS[6], "01:00"))

    def test_nothing_without_the_variable(self):
        out = self.alerts()
        with mock.patch("urllib.request.urlopen") as op:
            watch.notify(out)
        op.assert_not_called()

    def test_posts_plain_text_with_a_short_timeout(self):
        out = self.alerts()
        os.environ["BRAIN_NTFY_URL"] = "https://ntfy.example/topic"
        with mock.patch("urllib.request.urlopen") as op:
            watch.notify(out)
        req, = op.call_args.args
        self.assertEqual((req.full_url, req.get_method()), ("https://ntfy.example/topic", "POST"))
        self.assertLessEqual(op.call_args.kwargs["timeout"], 10)
        body = req.data.decode()
        self.assertIn("-20.0%", body)
        self.assertNotIn("headline", body)                 # info alerts are not pushed by default
        self.assertEqual(req.get_header("Priority"), "high")
        self.assertTrue(req.get_header("Content-type").startswith("text/plain"))

    def test_never_raises(self):
        out = self.alerts()
        os.environ["BRAIN_NTFY_URL"] = "https://ntfy.example/topic"
        with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
            watch.notify(out, log=self.log.append)
        self.assertTrue(any("ntfy push failed" in m for m in self.log))
        os.environ["BRAIN_NTFY_URL"] = "file:///etc/passwd"
        with mock.patch("urllib.request.urlopen") as op:
            watch.notify(out)
        op.assert_not_called()


class RefreshAndRunTest(WatchTest):
    def test_refresh_survives_failing_and_unavailable_sources(self):
        self.watching(tickers=["A", "B"], aircraft_of=["A"], areas=[AREA], sites=[SITE])
        os.environ["SEC_USER_AGENT"] = "test test@example.test"

        class Unavailable(RuntimeError):
            pass

        def prices(bank, t, log):
            if t == "A":
                raise ValueError("no price rows for A")
            return 1

        insiders = mock.Mock(side_effect=ModuleNotFoundError("No module named 'brain.sources.insiders'"))
        with mock.patch.object(watch, "_src_filings", return_value=2), \
                mock.patch.object(watch, "_src_insiders", insiders), \
                mock.patch.object(watch, "_src_prices", side_effect=prices), \
                mock.patch.object(watch, "_src_news", return_value=3), \
                mock.patch.object(watch, "_src_aircraft", side_effect=RuntimeError("adsb down")), \
                mock.patch.object(watch, "_src_area") as area, \
                mock.patch.object(watch, "_src_cameras", side_effect=Unavailable("needs OpenCV")):
            out = watch.refresh(self.bank, log=self.log.append)
        self.assertEqual(out["tickers"]["B"], {"filings": 2, "prices": 1, "headlines": 3})
        self.assertEqual(out["tickers"]["A"], {"filings": 2, "prices": None, "headlines": 3})
        self.assertEqual(insiders.call_count, 1)            # missing module: tried once, not per ticker
        self.assertEqual(out["benchmark"], 1)
        area.assert_not_called()                             # no AIS key
        self.assertEqual(len(out["errors"]), 2)
        self.assertEqual(len(out["skipped"]), 3)             # insiders, areas, cameras: one line each
        self.assertEqual(len(self.log), 5)

    def test_without_sec_identity_sec_sources_are_skipped_with_one_line(self):
        self.watching(tickers=["A", "B"])
        with mock.patch.object(watch, "_src_filings") as fil, mock.patch.object(watch, "_src_insiders") as ins, \
                mock.patch.object(watch, "_src_prices", return_value=0), \
                mock.patch.object(watch, "_src_news", return_value=0):
            watch.refresh(self.bank, log=self.log.append)
        fil.assert_not_called()
        ins.assert_not_called()
        self.assertEqual(len(self.log), 1)
        self.assertIn("SEC_USER_AGENT", self.log[0])

    def test_area_sampling_stops_after_unavailable(self):
        self.watching(areas=[AREA, {"name": "LA-LONG-BEACH"}])
        os.environ["AISSTREAM_API_KEY"] = "k"

        class Unavailable(RuntimeError):
            pass

        with mock.patch.object(watch, "_src_area", side_effect=Unavailable("needs websockets")) as area:
            out = watch.refresh(self.bank, log=self.log.append)
        self.assertEqual(area.call_count, 1)
        self.assertEqual(len(out["skipped"]), 1)

    def test_run_survives_a_failed_refresh_and_prints_one_line_per_alert(self):
        self.watching(tickers=["X"], rules={"headlines": True})
        self.bank.add_many([headline("X", f"story {i}", f"2026-10-0{i}T09:00:00Z", f"https://news.example/{i}")
                            for i in range(1, 3)])
        calls = []

        def refresh(bank, log=print):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("network down")
            return {}

        real_check = watch.check
        with mock.patch.object(watch, "refresh", side_effect=refresh), \
                mock.patch.object(watch, "check", side_effect=lambda bank, log=print: real_check(
                    bank, now="2026-10-03T12:00:00Z", log=log)), \
                mock.patch.object(watch, "notify") as push:
            watch.run(60, log=self.log.append, cycles=2, sleep=lambda s: None)
        self.assertEqual(len(calls), 2)
        self.assertTrue(any("refresh failed" in m for m in self.log))
        lines = [m for m in self.log if "headline:" in m]
        self.assertEqual(len(lines), 2)                       # two alerts, one line each, on the first round only
        self.assertEqual(push.call_count, 2)

    def test_prices_are_read_with_a_short_ttl(self):
        from brain.sources import prices
        seen = []

        def fetch(url, ttl):  # what prices.ingest asks the HTTP cache for
            seen.append(ttl)
            raise RuntimeError("stop here")

        with mock.patch.object(prices, "fetch", fetch):
            with self.assertRaises(RuntimeError):
                watch._src_prices(self.bank, "x", self.log.append)
            with self.assertRaises(RuntimeError):
                prices.ingest("X")  # an ordinary ingest keeps the long cache
        self.assertEqual(seen, [watch.PRICES_TTL, prices.TTL])
        self.assertLessEqual(watch.PRICES_TTL, 900)
        self.assertGreater(prices.TTL, watch.PRICES_TTL)

    def test_ctrl_c_stops_the_loop(self):
        self.watching()

        def interrupt(s):
            raise KeyboardInterrupt

        with mock.patch.object(watch, "refresh", return_value={}):
            watch.run(60, log=self.log.append, sleep=interrupt)
        self.assertEqual(self.log[-1], "stopped")


class CliTest(WatchTest):
    def cli(self, *argv):
        p = argparse.ArgumentParser()
        watch.add_cli(p.add_subparsers(dest="cmd", required=True))
        a = p.parse_args(argv)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = a.fn(a)
        return code, buf.getvalue()

    def test_every(self):
        self.assertEqual([watch.parse_every(x) for x in ("900", "90s", "15m", "1h30m")], [900, 90, 900, 5400])
        for bad in ("5s", "15 minutes", "m"):
            with self.assertRaises(argparse.ArgumentTypeError):
                watch.parse_every(bad)

    def test_watch_and_alerts_commands(self):
        self.assertEqual(self.cli("watch", "add", "ticker", "x", "y")[0], 0)
        self.assertIn("headlines off", self.cli("watch", "list")[1])  # a new watchlist
        self.assertEqual(self.cli("watch", "rule", "headlines", "on")[0], 0)
        self.assertEqual(self.cli("watch", "add", "site", "Plant", "One", "--ticker", "x", "--cameras", CAM)[0], 0)
        self.assertEqual(self.cli("watch", "add", "area", "box", "--bbox", "1", "2", "3", "4")[0], 0)
        self.assertEqual(self.cli("watch", "rule", "price_move", "0.03")[0], 0)
        self.assertEqual(self.cli("watch", "rule", "forms", "8-K,", "S-1")[0], 0)
        code, text = self.cli("watch", "list")
        self.assertIn("tickers      X Y", text)
        self.assertIn("Plant One (X): " + CAM, text)
        w = watch.load()
        self.assertEqual((w["rules"]["price_move"], w["rules"]["forms"]), (0.03, ["8-K", "S-1"]))
        self.assertEqual(self.cli("watch", "remove", "site", "plant", "one")[0], 0)
        self.bank.add(headline("X", "story", "2026-10-12T09:00:00Z", "https://news.example/1"))
        with mock.patch.object(watch, "now_iso", return_value="2026-10-12T12:00:00Z"):
            code, text = self.cli("alerts", "check")
        self.assertIn("1 new alert", text)
        code, text = self.cli("alerts")
        self.assertIn("* 2026-10-12T12:00Z info", text)
        aid = watch.alerts(self.bank)[0].id
        self.assertEqual(self.cli("alerts", "ack", aid[:6])[0], 0)
        self.assertEqual(self.cli("alerts", "list", "--unacked")[1].strip(), "no unacknowledged alerts")
        self.assertEqual(self.cli("alerts", "ack-all")[1].strip(), "0 alerts acknowledged")

    def test_run_once_says_it_is_one_round(self):
        with mock.patch.object(watch, "run") as run:
            code, text = self.cli("watch", "run", "--once")
        self.assertEqual((code, text.splitlines()[0]), (0, "one round of refresh and check"))
        self.assertEqual(run.call_args.kwargs["cycles"], 1)
        with mock.patch.object(watch, "run") as run:
            text = self.cli("watch", "run", "--every", "15m")[1]
        self.assertEqual(text.splitlines()[0], "watching every 900 s; Ctrl+C stops")
        self.assertIsNone(run.call_args.kwargs["cycles"])

    def test_refresh_summary(self):
        self.assertEqual(watch.summary({}), "nothing watched")
        out = {"tickers": {"X": {"filings": 2, "prices": None}}, "aircraft": 0, "areas": {"LA": 12},
               "cameras": None, "skipped": ["port areas skipped"], "errors": []}
        self.assertEqual(watch.summary(out), "X filings +2 prices -; flights +0; LA vessels 12 (1 skipped or failed, above)")

    def test_kind_defaults_to_ticker(self):
        self.assertEqual(self.cli("watch", "add", "aapl", "msft")[0], 0)
        self.assertEqual(self.cli("watch", "add", "site")[0], 0)  # a lone kind word is a ticker (SiteOne)
        self.assertEqual(self.cli("watch", "add", "aircraft", "aapl")[0], 0)
        w = watch.load()
        self.assertEqual((w["tickers"], w["aircraft_of"]), (["AAPL", "MSFT", "SITE"], ["AAPL"]))
        self.assertEqual(self.cli("watch", "remove", "msft")[0], 0)
        self.assertEqual(self.cli("watch", "remove", "ticker", "site")[0], 0)
        self.assertEqual(watch.load()["tickers"], ["AAPL"])


class AlertHelperTest(WatchTest):
    """watch.alert: one alert written as check writes them, for alerts raised outside the rules."""

    def test_writes_like_check_dedupes_by_key_records_the_entity_and_pushes(self):
        self.bank.add(price_snapshot("X", DAYS[:6], [100.0] * 6))
        snap = self.bank.facts("X", "series.price_daily")[0]
        os.environ["BRAIN_NTFY_URL"] = "https://ntfy.example/topic"
        with mock.patch("urllib.request.urlopen") as op:
            f = watch.alert(self.bank, "x", "decision", "notice", "X decision NO_EDGE -> BUY (policy D1, paper)", [snap, snap],
                            ["decision:X/abc"], at(DAYS[6], "23:59"), policy="D1")
        self.assertEqual((f.entity, f.attr, f.tier, f.value, f.known_at, f.as_of),
                         ("X", "alert", "derived", "X decision NO_EDGE -> BUY (policy D1, paper)", at(DAYS[6], "23:59"), DAYS[6]))
        self.assertEqual(f.source, {"name": "brain.watch", "rule": "decision"})
        self.assertEqual(f.context, {"rule": "decision", "severity": "notice", "keys": ["decision:X/abc"], "policy": "D1"})
        self.assertEqual(f.inputs, (snap.ref,))
        self.assertTrue(f.retrieved_at)
        self.assertEqual(op.call_count, 1)
        self.assertIn("X decision NO_EDGE -> BUY", op.call_args.args[0].data.decode())
        self.assertIn("X", json.loads(watch.state_path().read_text())["entities"])
        self.assertEqual([g.ref for g in watch.alerts(self.bank)], [f.ref])  # listed though X is not watched
        self.assertIsNone(watch.alert(self.bank, "X", "decision", "notice", "again", [snap], ["decision:X/abc"],
                                      at(DAYS[7])))
        self.assertEqual(len(self.bank.facts("X", "alert")), 1)
        self.assert_cited([f])
        with self.assertRaises(ValueError):
            watch.alert(self.bank, "X", "decision", "urgent", "m", [snap], ["k2"], at(DAYS[7]))

    def test_a_malformed_state_file_is_an_error_before_anything_is_written(self):
        self.bank.add(price_snapshot("X", DAYS[:6], [100.0] * 6))
        snap = self.bank.facts("X", "series.price_daily")[0]
        watch.state_path().write_text("[1, 2]")
        with self.assertRaises(ValueError):
            watch.alert(self.bank, "X", "decision", "notice", "m", [snap], ["k"], at(DAYS[6]))
        self.assertEqual(self.bank.facts("X", "alert"), [])
