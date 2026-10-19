import json
import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from brain import series
from brain.bank import Fact
from brain.research import eventtest, run
from brain.research.forecast_tests import Refused
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
            facts.append(Fact(entity=t, attr="release.earnings", value="Acme results", as_of=DAYS[i], known_at=when,
                              tier="filed", source={"name": "SEC EDGAR 8-K exhibit 99", "url": f"https://www.sec.gov/x/{t}/{j}"},
                              context={"accn": f"{t}-{j}", "opening": OPENINGS[kind], "events": []},
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
        self.patch = mock.patch.object(eventtest, "UNIVERSE", self.uni)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
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
