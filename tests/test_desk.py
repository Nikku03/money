"""The desk (brain/desk.py): the case file built for a check and the rules a reasoned note is held to, on a
made-up market of five stocks whose links are known. Nothing touches the network."""
import json
import os
from unittest import mock

import numpy as np
import pandas as pd

from brain import connect, desk, scanwatch
from brain.bank import Fact
from tests.test_scan import snapshot
from tests.util import TempHome

ASOF, AT = "2026-11-09", "2026-11-10T15:00:00Z"
STAMP = "2026-11-10T1500Z"
FAR = "2099-12-31"
NAMES = {"AAA": "ALPHA WIDGETS INC", "BBB": "BETA MACHINES CORP", "CCC": "GAMMA FOODS INC", "DDD": "DELTA DRUGS INC",
         "EEE": "EPSILON MINING INC"}


def ident(t: str, attr: str, value: str) -> Fact:
    return Fact(entity=t, attr=attr, value=value, as_of="2025-01-01", known_at="2025-01-01T00:00:00Z", tier="filed",
                source={"name": "test", "url": "https://example.test/id"})


def item(title: str, path: str, at: str = "2026-11-10T13:00:00Z", summary: str = "") -> dict:
    return {"title": title, "summary": summary, "url": f"https://news.test/{path}", "source": "news.test", "at": at}


def flag(t: str, move: float, out90: bool, filings=()) -> dict:
    return {"t": t, "move": move, "at": "2026-11-10T15:00Z", "session": "in session", "outside68": True,
            "outside90": out90, "used": 2.0, "range68": [-0.05, 0.06], "range90": [-0.10, 0.12], "shock": 0.01,
            "likeliest": False, "filings": list(filings), "headlines": [], "industry": None, "own": move + 0.01,
            "link": None, "why": "its own news" if filings else "its own move, no news found"}


class DeskCase(TempHome):
    def setUp(self):
        super().setUp()
        self._env = {k: os.environ.get(k) for k in ("BRAIN_SCANS", "BRAIN_DESK", "BRAIN_DESK_CALLS")}
        os.environ["BRAIN_SCANS"] = os.path.join(self._tmp.name, "scans")
        os.environ["BRAIN_DESK"] = os.path.join(self._tmp.name, "notes")
        os.environ["BRAIN_DESK_CALLS"] = os.path.join(self._tmp.name, "desk_calls.jsonl")
        rng = np.random.default_rng(11)
        d = pd.bdate_range("2025-06-02", ASOF).strftime("%Y-%m-%d").to_numpy()
        n = len(d)
        m, f = rng.normal(0, 0.01, n), rng.normal(0, 0.02, n)
        self.btc = rng.normal(0, 0.03, n)
        rets = {"SPY": m, "AAA": 1.2 * m + f + rng.normal(0, 0.004, n), "BBB": 0.8 * m + f + rng.normal(0, 0.004, n),
                "CCC": m + rng.normal(0, 0.02, n), "DDD": m + rng.normal(0, 0.02, n),
                "EEE": m + 1.5 * self.btc + rng.normal(0, 0.005, n)}
        facts = [snapshot(t, d, 50 * np.cumprod(1 + r)) for t, r in rets.items()]
        facts.append(snapshot("^VIX", d, np.full(n, 15.0)))
        for t, name in NAMES.items():
            facts += [ident(t, "identity.sic", "3674"), ident(t, "identity.name", name),
                      ident(t, "identity.industry", "Semiconductors & Related Devices")]
        self.bank.add_many(facts)
        self.dates = d
        scans = scanwatch.scan.scans_dir()
        scans.mkdir(parents=True)
        (scans / f"{ASOF}.json").write_text(json.dumps({"asof": ASOF, "rows": {t: [-0.1, -0.05, 0.06, 0.12, 0.01]
                                                                                 for t in NAMES}}))
        self.res = {
            "asof": ASOF, "scan": f"{ASOF}.json", "at": AT, "checked": 5, "listed": 5, "outside68": 4, "outside90": 3,
            "market_wide": False, "fear_up": False, "macro_today": [], "sectors": [], "why_counts": {"its own news": 1},
            "market": [{"symbol": "SPY", "name": "S&P 500 (SPY)", "move": -0.01, "at": "2026-11-10T15:00Z",
                        "session": "in session"}],
            "drivers": [{"symbol": "BTC-USD", "name": "bitcoin", "unit": "pct", "move": -0.06}],
            "flags": [flag("AAA", -0.20, True, ["results report (8-K 2.02) at 2026-11-09T21:10Z"]),
                      flag("BBB", -0.11, True), flag("DDD", 0.15, True), flag("EEE", -0.09, False)],
            "moves": {"AAA": -0.20, "BBB": -0.11, "CCC": 0.002, "DDD": 0.15, "EEE": -0.09}}
        self.feeds = {
            "AAA": [item("Alpha Widgets Cuts Its Outlook on Weak Brazil Demand", "a", summary="Orders fell 30%."),
                    item("Stock Market Today: Futures Slip", "w"),
                    item("Alpha Widgets Names a New Plant Manager", "old", at="2026-11-08T10:00:00Z")],
            "BBB": [item("Alpha Widgets Cuts Its Outlook on Weak Brazil Demand", "a")],
            "DDD": [], "EEE": [item("Bitcoin Falls 6% Overnight", "b")]}

    def tearDown(self):
        for k, v in self._env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        super().tearDown()

    def build(self, **k):
        with mock.patch.object(connect, "driver_changes", lambda cal: {"BTC-USD": self.btc[-(len(cal) - 1):]}):
            return desk.build(self.bank, self.res, self.res["moves"], STAMP, texts=False, accounts=False,
                              feed=lambda t: self.feeds[t], **k)

    def note(self, **over) -> dict:
        n = {"check": STAMP, "by": "agent", "backdrop": "The market is down 1% with bitcoin down 6%.",
             "backdrop_evidence": ["k", "d:bitcoin"],
             "themes": [{"name": "Brazil demand", "kind": "reported", "cause": "Alpha cut its outlook on Brazil.",
                         "mechanism": "Beta sells into the same market.", "beneath": "Farm incomes in Brazil.",
                         "members": ["AAA", "BBB"], "evidence": ["s1", "p:AAA~BBB"],
                         "read_through": [{"t": "CCC", "why": "it also sells there", "evidence": ["m:AAA"]}],
                         "watch": "Beta's own outlook next week."}],
             "singles": [{"t": "EEE", "kind": "inferred", "cause": "Bitcoin's fall carried it.",
                          "beneath": "It has moved 1.5% for each 1% in bitcoin.", "evidence": ["x:EEE"]}],
             "unexplained": [{"t": "DDD", "tried": "no filing, no story, its partners are flat"}],
             "calls": [{"ticker": "CCC", "call": "move", "threshold": 5.0, "prob": 0.3, "by": FAR,
                        "thesis": "Brazil reaches it too", "from": "Brazil demand"}],
             "threads": ["Does Beta confirm the Brazil weakness?"]}
        n.update(over)
        return n


class CaseFileTest(DeskCase):
    def test_a_check_s_moves_stay_out_of_the_repo_and_are_read_back(self):
        p = scanwatch.record(self.res)
        self.assertNotIn("moves", json.loads(p.read_text()))             # the committed record: flags only
        self.assertTrue(str(scanwatch.moves_path(ASOF, STAMP)).startswith(self._tmp.name))
        rec, moves, stamp = desk.latest_watch()
        self.assertEqual((stamp, rec["asof"], moves["CCC"]), (STAMP, ASOF, 0.002))
        os.remove(scanwatch.moves_path(ASOF, STAMP))
        self.assertIsNone(desk.latest_watch()[1])                         # an older check: its flags only

    def test_the_case_file_splits_each_move_and_finds_what_ties_the_movers(self):
        case = self.build()
        self.assertEqual([m["t"] for m in case["movers"]], ["AAA", "BBB", "DDD", "EEE"])
        a, e = case["movers"][0], case["movers"][3]
        self.assertEqual((a["partners"][0]["t"], a["partners"][0]["mover"]), ("BBB", True))   # a year of moving together
        self.assertGreater(a["partners"][0]["corr"], 0.9)
        self.assertAlmostEqual(a["market_part"], 1.2 * -0.01, delta=0.004)
        self.assertEqual(a["drivers"], [])                                # bitcoin never moved it
        self.assertEqual([x["name"] for x in e["drivers"]], ["bitcoin"])
        self.assertAlmostEqual(e["drivers"][0]["implied"], 1.5 * -0.06, delta=0.01)
        self.assertAlmostEqual(e["left"], -0.09 - e["market_part"] - e["drivers"][0]["implied"], places=9)
        self.assertLess(abs(e["left"]), 0.02)                             # bitcoin and the market carried nearly all
        self.assertEqual(len(a["last"]), desk.LAST)
        self.assertEqual(a["last"][-1][0], ASOF)
        story = next(s for s in case["stories"] if "Brazil" in s["title"])
        self.assertEqual((story["feeds"], story["names"], story["wrap"]), (["AAA", "BBB"], ["AAA"], False))
        self.assertEqual(a["stories"], [story["id"]])                     # the story before the close is left out
        self.assertEqual(case["movers"][1]["stories"], [story["id"]])     # in Beta's feed, though it names Alpha
        self.assertEqual([case["stories"][int(i[1:]) - 1]["title"] for i in case["wraps"]],
                         ["Stock Market Today: Futures Slip"])
        self.assertEqual(case["clusters"][0]["members"], ["AAA", "BBB"])
        self.assertEqual({x["by"] for x in case["clusters"][0]["ties"]}, {story["id"], "p:AAA~BBB"})
        self.assertEqual(case["reach"][0]["driver"], "bitcoin")           # a move of twice its usual day
        self.assertEqual(case["reach"][0]["stocks"][0]["t"], "EEE")
        self.assertIsNone(case["memory"])

    def test_every_item_has_an_id_and_the_text_shows_it(self):
        case = self.build()
        ev = desk.evidence(case)
        for i in ("k", "d:bitcoin", "m:AAA", "x:EEE", "h:AAA", "p:AAA~BBB", "s1", "c1", "r:bitcoin"):
            self.assertIn(i, ev)
        self.assertIn("bitcoin carried", ev["x:EEE"]["text"])
        self.assertEqual(ev["s1"]["url"], "https://news.test/a")
        text = desk.render(case)
        for i in ("[k]", "[m:DDD]", "[c1]", "[s1]", "[r:bitcoin]"):
            self.assertIn(i, text)
        self.assertIn("no filing since the close and no story in its feed", text)       # DDD
        self.assertIn("Orders fell 30%.", desk.render(case, detail=["AAA"]))
        p = desk.save_case(case)
        self.assertTrue(str(p).startswith(self._tmp.name))                # a working file, outside the repo
        self.assertEqual(desk.load_case(STAMP)["check"], STAMP)

    def test_a_stock_whose_move_was_not_kept_is_read_at_the_time_of_the_case(self):
        res = {**self.res, "flags": [f for f in self.res["flags"] if f["t"] != "BBB"]}
        close = scanwatch.close_on(self.bank.view(), "BBB", ASOF)
        asked = []

        def quote(t):
            asked.append(t)
            return {"price": close * 0.9, "at": "x", "session": "in session"}
        with mock.patch.object(connect, "driver_changes", lambda cal: {}):
            case = desk.build(self.bank, res, {}, STAMP, texts=False, accounts=False, feed=lambda t: [], quote=quote)
        partner = case["movers"][0]["partners"][0]                        # an older check kept only its flags
        self.assertEqual((partner["t"], partner["mover"]), ("BBB", False))
        self.assertAlmostEqual(partner["move"], -0.10, places=6)
        self.assertEqual(asked, ["BBB"])                                  # read once, and only the partner
        self.assertIsNone(case["drivers"][0]["usual"])                    # no driver history: no usual day
        self.assertEqual(case["reach"], [])
        self.assertEqual(case["clusters"], [])                            # one mover of the pair: no cluster

    def test_earlier_checks_that_flagged_a_stock(self):
        scanwatch.record({**self.res, "at": "2026-11-10T13:00:00Z"})
        scanwatch.record({**self.res, "at": "2026-11-10T17:00:00Z"})     # after this check: not "before"
        seen = desk.seen_before(ASOF, STAMP)
        self.assertEqual(seen["AAA"], ["11-10 13:00Z -20.0%"])

    def test_a_stale_balance_sheet_figure_is_not_shown(self):
        def fact(attr, value, as_of):
            return Fact(entity="AAA", attr=attr, value=value, as_of=as_of, known_at=as_of + "T00:00:00Z", tier="filed",
                        source={"name": "test", "url": "https://example.test/x"}, unit="USD")
        self.bank.add_many([fact("us-gaap:CashAndCashEquivalentsAtCarryingValue", 5e8, "2026-09-30"),
                            fact("us-gaap:LongTermDebt", 9e8, "2016-03-31")])
        with mock.patch("brain.sources.sec.facts_for", lambda t: []):
            got = desk.numbers(self.bank, "AAA")
        self.assertEqual((got["cash"], got["cash_at"]), (5e8, "2026-09-30"))
        self.assertNotIn("debt", got)


class NoteTest(DeskCase):
    def test_a_note_that_cites_its_causes_and_covers_the_big_movers_passes(self):
        self.assertEqual(desk.check(self.note(), self.build()), [])

    def test_what_a_note_is_held_to(self):
        case = self.build()

        def wrong(**over):
            return " | ".join(desk.check(self.note(**over), case))
        theme = self.note()["themes"][0]
        self.assertIn("is for the check", wrong(check="2026-11-10T0900Z"))
        self.assertIn("'by' is empty", wrong(by=""))
        self.assertIn("'backdrop' is empty", wrong(backdrop=" "))
        self.assertIn("backdrop: cites nothing", wrong(backdrop_evidence=[]))
        self.assertIn("cites 's99'", wrong(themes=[{**theme, "evidence": ["s99"]}]))
        self.assertIn("cites no story, filing or page", wrong(themes=[{**theme, "evidence": ["p:AAA~BBB"]}]))
        self.assertEqual(wrong(themes=[{**theme, "kind": "inferred", "evidence": ["p:AAA~BBB"]}]), "")
        self.assertIn("'mechanism' is empty", wrong(themes=[{**theme, "mechanism": ""}]))
        self.assertIn("ties at least two stocks", wrong(themes=[{**theme, "members": ["AAA"]}]))
        self.assertIn("ZZZ is not a stock of the scan", wrong(themes=[{**theme, "members": ["AAA", "ZZZ"]}]))
        self.assertIn("read-through CCC: 'why' is empty",
                      wrong(themes=[{**theme, "read_through": [{"t": "CCC", "why": "", "evidence": ["m:AAA"]}]}]))
        self.assertIn("DDD is outside its 9-in-10 range", wrong(unexplained=[]))
        self.assertIn("'tried' is empty", wrong(unexplained=[{"t": "DDD", "tried": ""}]))
        self.assertIn("'kind' is one of", wrong(singles=[{"t": "EEE", "kind": "sure", "cause": "x", "beneath": "y",
                                                           "evidence": ["x:EEE"]}]))
        call = self.note()["calls"][0]
        self.assertIn("'prob' is the chance", wrong(calls=[{**call, "prob": 1.0}]))
        self.assertIn("'by' is the date", wrong(calls=[{**call, "by": "next week"}]))
        self.assertIn("'from' names no theme", wrong(calls=[{**call, "from": "a hunch"}]))
        self.assertIn("'call' is one of", wrong(calls=[{**call, "call": "buy"}]))
        self.assertIn("the case file has no move for CCC", " ".join(desk.check(
            self.note(), {**case, "moves": {t: v for t, v in case["moves"].items() if t != "CCC"}})))
        self.assertEqual(wrong(calls=[{**call, "ticker": "SPY", "price": 512.5}]), "")   # a price given at the call
        # a page the reasoner read is evidence once it is listed, and it can report a cause
        page = "https://news.test/brazil"
        self.assertIn("neither in the case file nor a page", wrong(themes=[{**theme, "evidence": [page]}]))
        self.assertEqual(wrong(themes=[{**theme, "evidence": [page]}],
                               read=[{"url": page, "says": "Brazil farm incomes fell 12%.", "ticker": "AAA"}]), "")
        self.assertIn("'says' is empty", wrong(read=[{"url": page, "says": ""}]))
        self.assertIn("not an http(s) page", wrong(read=[{"url": "llm:guess", "says": "x"}]))

    def test_a_note_is_recorded_once_with_its_calls_and_the_items_it_cites(self):
        case = self.build()
        noted = []

        def claim(bank, ticker, text, url, **k):
            noted.append((ticker, url, k["noted_by"]))
            return Fact(entity=ticker, attr="claim", value=text, as_of="2026-11-10", known_at="2026-11-10T12:00:00Z",
                        tier="reported", source={"name": "news.test", "url": url})
        page = {"url": "https://news.test/brazil", "says": "Brazil farm incomes fell 12%.", "ticker": "AAA",
                "published": "2026-11-10T12:00:00Z"}
        rec = desk.record_note(self.bank, case, self.note(read=[page, {"url": "https://news.test/x", "says": "y"}]),
                               claim=claim)
        self.assertEqual(noted, [("AAA", "https://news.test/brazil", "desk")])          # a page with a ticker
        self.assertEqual(rec["claims_noted"], 1)
        calls = [json.loads(x) for x in desk.calls_path().read_text().splitlines()]
        self.assertEqual([(c["ticker"], c["call"], c["prob"], c["by"], c["threshold_pct"]) for c in calls],
                         [("CCC", "move", 0.3, FAR, 5.0)])
        close = scanwatch.close_on(self.bank.view(), "CCC", ASOF)
        self.assertAlmostEqual(calls[0]["start_price"], close * 1.002, places=3)   # the price at the check, not the close
        self.assertIn(f"desk {STAMP}, from Brazil demand", calls[0]["thesis"])
        self.assertEqual(calls[0]["sources"], ["https://news.test/a"])
        self.assertEqual(rec["calls"], [calls[0]["id"]])
        kept = json.loads((desk.notes_dir() / ASOF / f"{STAMP}.json").read_text())
        self.assertEqual(sorted(kept["evidence"]), ["d:bitcoin", "k", "m:AAA", "p:AAA~BBB", "s1", "x:EEE"])
        self.assertIn("Brazil", kept["evidence"]["s1"]["text"])           # the note stands without its case file
        with self.assertRaisesRegex(ValueError, "already recorded"):
            desk.record_note(self.bank, case, self.note(calls=[]), claim=claim)
        text = desk.show_note(kept)
        for part in ("Theme: Brazil demand (reported)", "May also reach CCC (untested)", "Unexplained:", "DDD",
                     "chance 30%", "Open questions"):
            self.assertIn(part, text)
        s = desk.score(self.bank)
        self.assertEqual((s["calls"], s["judged"], s["enough"]), (1, 0, False))

    def test_a_note_that_fails_is_not_recorded_and_makes_no_call(self):
        with self.assertRaisesRegex(ValueError, "DDD is outside its 9-in-10 range"):
            desk.record_note(self.bank, self.build(), self.note(unexplained=[]))
        self.assertFalse(desk.calls_path().exists())
        self.assertEqual(list(desk.notes_dir().glob("*/*.json")), [])

    def test_the_next_case_file_carries_the_last_note(self):
        desk.record_note(self.bank, self.build(), self.note(calls=[]))
        self.assertIsNone(self.build()["memory"])                         # the same check: nothing before it
        later = {**self.res, "at": "2026-11-10T19:00:00Z"}
        with mock.patch.object(connect, "driver_changes", lambda cal: {}):
            case = desk.build(self.bank, later, later["moves"], "2026-11-10T1900Z", texts=False, accounts=False,
                              feed=lambda t: [])
        self.assertEqual(case["memory"]["check"], STAMP)
        self.assertEqual(case["memory"]["themes"][0]["members"], ["AAA", "BBB"])
        self.assertEqual(case["memory"]["threads"], ["Does Beta confirm the Brazil weakness?"])
        self.assertIn("Brazil demand", desk.evidence(case)["t1"]["text"])
        self.assertIn("MEMORY", desk.render(case))
