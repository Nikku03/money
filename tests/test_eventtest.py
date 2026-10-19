import json
import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from brain import config, series
from brain.bank import Fact
from brain.research import eventtest, run
from brain.research.forecast_tests import Refused
from brain.sources import releases as reader
from tests.util import TempHome

DAYS = [d.date().isoformat() for d in pd.bdate_range("2010-10-01", "2019-03-29")]
TICKERS = [f"S{i:02d}" for i in range(30)]
OPENINGS = {"raised": "Acme reports results and raises full-year 2018 revenue and EPS guidance",
            "cut": "Acme reports results; lowers full-year outlook on weak demand",
            "both": "Acme raises revenue guidance and lowers EPS outlook",
            "none": "Acme reports third quarter results"}


def snap(entity, close):
    df = pd.DataFrame({"date": DAYS, "open": close, "high": close, "low": close, "close": close, "adjclose": close,
                       "volume": np.full(len(DAYS), 1000)})
    return Fact(entity=entity, attr="series.price_daily", value=series.store(df), as_of=DAYS[-1], known_at=DAYS[-1],
                tier="market", source={"name": "test", "url": "https://example.test/px"},
                context={"rows": len(DAYS), "first": DAYS[0], "last": DAYS[-1], "splits": []})


def world(seed=5):
    """Releases every 63 sessions; 'raised' ones move +2% on days -1..+1 and +3% more over +2..+21, 'cut' ones the
    mirror image, the rest nothing. Releases are filed after the close (21:00Z), so day 0 is the next session."""
    rng = np.random.default_rng(seed)
    n = len(DAYS)
    spy = rng.normal(0.0003, 0.01, n)
    facts = [snap("SPY", 100 * np.cumprod(1 + spy))]
    for k, t in enumerate(TICKERS):
        r = spy + rng.normal(0, 0.012, n)
        for j, i in enumerate(range(20 + k, n - 30, 63)):
            kind = rng.choice(["raised", "cut", "both", "none", "none", "none", "none"])
            s = {"raised": 1, "cut": -1}.get(kind, 0)
            r[i + 1] += s * 0.02            # day 0: the session after the filing
            r[i + 3: i + 23] += s * 0.03 / 20   # days +2..+21
            when = f"{DAYS[i]}T21:00:00Z"
            facts.append(Fact(entity=t, attr="filing", value="8-K", as_of=DAYS[i], known_at=when, tier="filed",
                              source={"name": "SEC EDGAR", "url": f"https://www.sec.gov/f/{t}/{j}"},
                              context={"accn": f"{t}-{j}", "items": "2.02,9.01"}, retrieved_at="2019-04-01T00:00:00Z"))
            facts.append(Fact(entity=t, attr="release.earnings", value="Acme results", as_of=DAYS[i], known_at=when,
                              tier="filed", source={"name": "SEC EDGAR 8-K exhibit 99", "url": f"https://www.sec.gov/x/{t}/{j}"},
                              context={"accn": f"{t}-{j}", "opening": OPENINGS[kind], "events": [],
                                       "opening_chars": reader.OPENING_CHARS},
                              retrieved_at="2019-04-01T00:00:00Z"))
        facts.append(snap(t, 50 * np.cumprod(1 + r)))
    return facts


class EventTest(TempHome):
    def setUp(self):
        super().setUp()
        self.bank.add_many(world())
        self.uni = os.path.join(os.environ["MONEY_HOME"], "uni.json")
        with open(self.uni, "w") as fh:
            json.dump({"tickers": TICKERS}, fh)
        self.patches = [mock.patch.object(eventtest, "UNIVERSE", self.uni),
                        mock.patch.object(eventtest, "SUITES", [{"version": 1, "tag": None, "universe": None,
                                                                 "events": {"guidance_raised": 1}}])]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        super().tearDown()

    def test_declared_trial_finds_what_was_planted(self):
        new = eventtest.declare(self.bank, trial=True, log=None)
        self.assertEqual([r["id"] for r in new], ["E1-guidance_raised-reaction", "E1-guidance_raised-drift"])
        out = {r["id"]: r for r in eventtest.test(self.bank, trial=True, log=None)}
        led, res = eventtest.paths(True)
        rows = {r["id"]: r for r in run.verdicts(led) if run.is_event(r)}
        up = out["E1-guidance_raised-drift"]["metrics"]["test"]
        self.assertAlmostEqual(up["diff"], 0.03, delta=0.01)
        self.assertAlmostEqual(out["E1-guidance_raised-reaction"]["metrics"]["test"]["diff"], 0.02, delta=0.01)
        self.assertEqual(rows["E1-guidance_raised-drift"]["verdict"], "PASS")
        cut = eventtest.spec({"id": "E1-guidance_cut-drift", "event": "guidance_cut", "window": "drift", "sign": -1},
                             "2018-12", True, "2019-05-01T00:00:00Z")   # the same machinery, the other way
        r = eventtest.evaluate(eventtest.releases(self.bank, cut), cut)
        self.assertAlmostEqual(r["metrics"]["test"]["diff"], -0.03, delta=0.01)
        self.assertTrue(r["gate"]["pass"])
        self.assertEqual(rows["E1-guidance_raised-reaction"]["verdict"], "REPORT")
        self.assertTrue(all(r["controls"]["pass"] for r in out.values()))
        self.assertIn("## Events", res.read_text())
        self.assertFalse(any(run.is_hypothesis(r) for r in rows.values()))

    def test_a_release_tagged_both_ways_is_neither(self):
        s = eventtest.spec(eventtest.tests()[0], "2018-12", True, "2019-05-01T00:00:00Z")
        self.assertFalse(eventtest.is_tagged({"kinds": {"guidance_raised", "guidance_cut"}}, s))
        self.assertTrue(eventtest.is_tagged({"kinds": {"guidance_raised", "product_launch"}}, s))

    def test_no_planted_effect_no_pass(self):
        rng = np.random.default_rng(1)   # tagged and untagged releases drawn alike: nothing to find
        rels = [{"ticker": "X", "month": f"20{15 + i // 12:02d}-{i % 12 + 1:02d}", "x": float(rng.normal(0, 0.05)),
                 "kinds": {"guidance_raised"} if j == 0 else set(), "ref": ""} for i in range(48) for j in range(6)]
        rels += [{**r, "month": r["month"].replace("2015", "2011").replace("2016", "2012").replace("2017", "2013")
                  .replace("2018", "2014")} for r in rels]
        s = eventtest.spec(eventtest.tests()[1], "2018-12", True, "2019-05-01T00:00:00Z")
        r = eventtest.evaluate(rels, s)
        self.assertGreater(r["metrics"]["test"]["p"], 0.05)
        self.assertFalse(r["gate"]["pass"])

    def test_two_sided_finds_either_way_and_version_two_ids(self):
        t = {"id": "E2-small-guidance_raised-drift", "event": "guidance_raised", "window": "drift", "sign": 0,
             "version": 2, "suite": "small", "universe": self.uni, "cost_round_trip": 0.01}
        s = eventtest.spec(t, "2018-12", True, "2019-05-01T00:00:00Z")
        self.assertEqual((s["two_sided"], s["cost_round_trip"], s["version"]), (True, 0.01, 2))
        r = eventtest.evaluate(eventtest.releases(self.bank, s), s)
        self.assertTrue(r["gate"]["pass"])                          # +3% planted: found without saying which way
        down = {**s, "event": "guidance_cut"}
        r = eventtest.evaluate(eventtest.releases(self.bank, down), down)
        self.assertLess(r["metrics"]["test"]["diff"], 0)
        self.assertTrue(r["gate"]["pass"])
        self.assertEqual(eventtest.test_id("guidance_cut", "drift", 2, "micro"), "E2-micro-guidance_cut-drift")
        one = eventtest.spec({"id": "E1-guidance_raised-drift", "event": "guidance_raised", "window": "drift",
                              "sign": 1}, "2018-12", True, "2019-05-01T00:00:00Z")
        self.assertNotIn("two_sided", one)                          # version 1 specs are unchanged

    def v2_suite(self):
        return {"version": 2, "tag": "small", "universe": self.uni, "events": {"guidance_raised": 0, "guidance_cut": 0},
                "cost_round_trip": 0.01}

    def test_version_two_declares_pinned_and_runs_two_sided(self):
        with mock.patch.object(eventtest, "SUITES", [self.v2_suite()]):
            new = eventtest.declare(self.bank, trial=True, log=None, bands=["small"])
            self.assertEqual(len(new), 4)
            s = new[0]["spec"]
            self.assertEqual((s["universe_sha"], s["n_tickers"]), (run.universe_sha(self.uni), 30))
            self.assertTrue(s["data_sha"])
            out = {r["id"]: r for r in eventtest.test(self.bank, trial=True, log=None)}
        cut = out["E2-small-guidance_cut-drift"]
        self.assertLess(cut["metrics"]["test"]["diff"], 0)       # planted -3%: found, and the sign says which way
        self.assertTrue(cut["gate"]["pass"])
        self.assertEqual(cut["coverage"], {"releases": cut["coverage"]["releases"], "unread": 0})
        text = eventtest.paths(True)[1].read_text()
        self.assertIn("two-sided", text)
        self.assertIn("costs about 1%", text)

    def test_version_two_refuses_a_changed_universe_or_changed_data(self):
        with mock.patch.object(eventtest, "SUITES", [self.v2_suite()]):
            eventtest.declare(self.bank, trial=True, log=None)
            with open(self.uni) as fh:
                orig = fh.read()
            with open(self.uni, "w") as fh:
                json.dump({"tickers": TICKERS[:12]}, fh)   # the band rebuilt after declaring
            with self.assertRaises(Refused):
                eventtest.test(self.bank, trial=True, log=None)
            with open(self.uni, "w") as fh:
                fh.write(orig)
            f = self.bank.facts("S03", "release.earnings")[0]
            self.bank.downgrade("S03", f.id, "test: a release found wrong after declaring")
            with self.assertRaises(Refused):                 # the data read is no longer the data pinned
                eventtest.test(self.bank, trial=True, log=None)
            self.assertFalse(any(r["op"] == "result" for r in eventtest.ledger.read(eventtest.paths(True)[0])))

    def test_a_band_is_declared_only_once_its_releases_are_read(self):
        when = "2018-06-04T21:00:00Z"
        self.bank.add_many([Fact(entity="S05", attr="filing", value="8-K", as_of=when[:10], known_at=when,
                                 tier="filed", source={"name": "SEC EDGAR", "url": "https://www.sec.gov/f/S05/new"},
                                 context={"accn": "S05-new", "items": "2.02"}, retrieved_at=when)])
        self.assertIsNone(eventtest.collection_problem(self.bank, self.uni))   # 1 unread of hundreds: within 2%
        many = [Fact(entity=t, attr="filing", value="8-K", as_of="2018-07-02", known_at=f"2018-07-0{2 + k % 3}T21:00:00Z",
                     tier="filed", source={"name": "SEC EDGAR", "url": f"https://www.sec.gov/g/{t}/{k}"},
                     context={"accn": f"{t}-late-{k}", "items": "2.02"}, retrieved_at="2018-07-06T00:00:00Z")
                for k, t in enumerate(TICKERS)]
        self.bank.add_many(many)
        self.assertIn("not read yet", eventtest.collection_problem(self.bank, self.uni))
        with mock.patch.object(eventtest, "SUITES", [{"version": 1, "tag": None, "universe": None,
                                                      "events": {"guidance_raised": 1}}, self.v2_suite()]):
            with self.assertRaises(Refused):                 # nothing is written: not even the ready version 1 ids
                eventtest.declare(self.bank, trial=True, log=None)
            self.assertEqual(eventtest.ledger.read(eventtest.paths(True)[0]), [])
            with self.assertRaises(Refused):
                eventtest.declare(self.bank, trial=True, log=None, bands=["micro"])

    def test_version_two_leaves_unread_releases_out_of_both_groups(self):
        s = eventtest.spec({"id": "E2-small-guidance_raised-drift", "event": "guidance_raised", "window": "drift",
                            "sign": 0, "version": 2, "suite": "small", "universe": self.uni, "cost_round_trip": 0.01},
                           "2018-12", True, "2019-05-01T00:00:00Z")
        rels = eventtest.releases(self.bank, s)
        unread = [{**r, "kinds": set(), "read": False, "x": 0.5, "ref": r["ref"] + "u"} for r in rels[:40]]
        a, b = eventtest.evaluate(rels, s), eventtest.evaluate(rels + unread, s)
        self.assertEqual(a["metrics"], b["metrics"])
        self.assertEqual(b["coverage"]["unread"], 40)
        one = {**s, "version": 1}
        self.assertNotEqual(eventtest.evaluate(rels + unread, one)["metrics"], a["metrics"])   # version 1 as it was

    def test_refusals(self):
        with self.assertRaises(Refused):
            eventtest.test(self.bank, ["E1-nope-drift"], trial=True, log=None)
        s = eventtest.spec(eventtest.tests()[1], "2018-12", True, "2019-05-01T00:00:00Z")
        with self.assertRaises(Refused):   # no releases collected at all
            eventtest.evaluate([], s)
        few = [{"ticker": "X", "month": m, "x": 0.01, "kinds": {"guidance_raised"} if j == 0 else set(), "ref": ""}
               for m in ("2015-03", "2016-03", "2011-05") for j in range(5)]
        r = eventtest.evaluate(few, s)   # too few events: recorded, and it cannot pass
        self.assertTrue(r["thin"])
        self.assertFalse(r["gate"]["pass"])
        led = eventtest.paths(True)[0]
        real = {**s, "trial": False, "test": ["2019-01", "2026-08"], "id": "E1-real-drift"}
        eventtest.ledger.append({"op": "declare", "kind": "event", "id": "E1-real-drift", "spec": real,
                                 "spec_sha": eventtest.spec_sha(real)}, led)
        with self.assertRaises(Refused):   # a trial never reads TEST years, even from a copy of the real ledger
            eventtest.test(self.bank, ["E1-real-drift"], trial=True, log=None)


if __name__ == "__main__":
    unittest.main()


class FrozenSpecs(unittest.TestCase):
    def test_the_declared_version_one_specs_are_what_the_code_builds_now(self):
        led = config.ROOT / "research" / "ledger.jsonl"
        declared = {r["id"]: r["spec"] for r in eventtest.ledger.read(led) if r.get("op") == "declare"
                    and r.get("kind") == "event" and r["spec"].get("version") == 1}
        if not declared:
            self.skipTest("no version 1 event test in the ledger")
        v1 = [su for su in eventtest.SUITES if su["version"] == 1]
        for t in eventtest.tests(v1):
            if t["id"] in declared:
                now = eventtest.spec(t, "2026-08", False, "x")
                self.assertEqual({k: v for k, v in declared[t["id"]].items() if k not in eventtest.VARYING},
                                 {k: v for k, v in now.items() if k not in eventtest.VARYING}, t["id"])
