"""The terminal's DEC (docs/DECIDE.md section 5): T DEC and DEC alone over recorded decisions, T DEC NEW as a
background job in a slot of its own, refused to the read-only reader, the WL Decision column, and CN and CLM
showing the record D1 weighs beside every call. Decisions are written as facts of the contract's shape, so
nothing here runs the engine or a model; brain.decide's reading and scoring functions are the real ones."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from brain import claims, config, decide, watch
from brain.bank import Bank, Fact
from brain.sources import aviation
from brain.terminal import commands, data, jobs, server
from brain.terminal.commands import parse
from brain.timeutil import now_iso, today
from tests.util import TempHome, filed, price_snapshot

END = pd.Timestamp(today()) - pd.offsets.BDay(3)
DAYS = pd.bdate_range(end=END, periods=160).strftime("%Y-%m-%d").tolist()
JUDGE = "llm:fake@abcd1234"
SITE = "analyst.example.com"


def claim(t, direction, known, horizon=21, text="Analyst says buy"):
    return Fact(entity=t, attr="claim", value=text, as_of=known[:10], known_at=known, tier="opinion",
                source={"name": SITE, "url": f"https://{SITE}/{t}/{known}"},
                context={"direction": direction, "horizon_days": horizon}, retrieved_at=known)


def judgment(t, direction, known, inputs):
    return Fact(entity=t, attr="judgment", value=f"{t}: up on margins", as_of=known[:10], known_at=known, tier="opinion",
                source={"name": JUDGE}, context={"direction": direction, "horizon_days": 21, "confidence": 0.6,
                                                 "reasons": [], "unverified_numbers": [], "policy": "D1"},
                inputs=tuple(f.ref for f in inputs), retrieved_at=known)


def alert(t, rule, msg, at, **ctx):
    return Fact(entity=t, attr="alert", value=msg, as_of=at[:10], known_at=at, tier="derived",
                source={"name": "brain.watch", "rule": rule},
                context={"rule": rule, "severity": "notice", "keys": [f"{rule}:{msg}"], **ctx}, retrieved_at=at)


def decision(t, action, day, lean=0, sha=None, **ctx):
    """A decision fact as decide records it (docs/DECIDE.md section 2.2), made at the end of ``day``. ``lean`` is
    a direction, or the whole lean dict."""
    base = {"policy": "D1", "policy_sha": sha or decide.POLICY_SHA, "git": {"commit": "c0ffee", "brain_dirty": False},
            "horizon_days": 21, "direction": decide.DIRECTION[action], "score": float(decide.DIRECTION[action]),
            "threshold": 0.5, "signal_score": 0.0, "source_term": {"id": None, "direction": 0, "weight": 0.0},
            "items": [], "lean": lean if isinstance(lean, dict) else
            {"direction": lean, "up": max(lean, 0), "down": max(-lean, 0), "sources": abs(lean)},
            "context_refs": [], "price": None, "judgment": None, "read": None, "explanation": "", "explained_by": "template",
            "explain_error": None, "unverified_numbers": [], "notes": []}
    base.update(ctx)
    return Fact(entity=t, attr="decision", value=action, as_of=day, known_at=day + "T23:59:59Z", tier="derived",
                source={"name": "brain.decide", "policy": "D1"}, context=base, inputs=(f"{t}/0",),
                retrieved_at=day + "T22:00:00Z")


def build(bank: Bank, home: Path) -> dict:
    n = len(DAYS)
    snaps = {"XCO": price_snapshot("XCO", DAYS, [50 * 1.002 ** k for k in range(n)]),   # beats a flat SPY
             "ZCO": price_snapshot("ZCO", DAYS, [80 * 0.998 ** k for k in range(n)]),   # trails it
             "SPY": price_snapshot("SPY", DAYS, [400.0] * n)}
    bank.add_many(snaps.values())
    bank.add_many([filed("XCO", "identity.name", "Example Co", DAYS[0], DAYS[0]),
                   filed("ZCO", "identity.name", "Zed Co", DAYS[0], DAYS[0])])
    now = today() + "T12:00:00Z"
    fresh, stale = claim("XCO", 1, now), claim("XCO", 1, DAYS[-40] + "T12:00:00Z", text="Old analyst note")
    judged = judgment("XCO", 1, now, [snaps["XCO"]])
    moved = alert("XCO", "price_move", "XCO moved 6% in a day", DAYS[-1] + "T21:00:00Z")
    # a company aircraft alert about a tail opted out since: SHOW withholds it, so DEC must too
    landed = alert("XCO", "aircraft", "XCO aircraft N2XC landed at KTEB", DAYS[-2] + "T18:00:00Z", reg="N2XC", hex="a00002")
    (home / "aviation_optout.txt").write_text("N2XC  # asked not to be tracked\n")
    bank.add_many([fresh, stale, judged, moved, landed])
    record = {"bets": 0, "hits": 0, "hit_rate": None, "p": None, "q": None, "lower": None, "mean_signed_excess": None,
              "up": 1, "down": 0, "calls": 1, "dropped": 0, "pending": 1, "unscorable": 0, "proven": False,
              "weight": 0.0, "why": "0 bets from 1 call, 1 pending; needs 30 bets"}
    items = [{"kind": "signal", "id": "H1-mom12_1", "direction": 0, "weight": 0.0,
              "why": "FAIL on TEST (net excess +0.11%/mo, p 0.18)", "refs": [],
              "record": {"verdict": "FAIL", "universe": "universe/us_large.json", "rebalance_date": DAYS[-10],
                         "coverage": {"ranked": 60, "universe_size": 60}, "held": False}},
             {"kind": "source", "id": f"{SITE} via agent", "direction": 1, "weight": 0.0, "why": record["why"],
              "refs": [fresh.ref], "record": record},
             {"kind": "source", "id": JUDGE, "direction": 1, "weight": 0.0, "why": "0 bets; needs 30 bets",
              "refs": [judged.ref], "record": None}]
    notes = [{"ref": stale.ref, "why": "stale: entry 39 sessions before this decision"},
             {"ref": moved.ref, "why": "alert (price_move); alert rules are untested"},
             {"ref": landed.ref, "why": "alert (aircraft); alert rules are untested"}]
    last = snaps["XCO"]
    latest = decision("XCO", "NO_EDGE", today(), lean={"direction": 1, "up": 2, "down": 0, "sources": 2}, items=items,
                      context_refs=[n["ref"] for n in notes], context_notes=notes,
                      price={"ref": last.ref, "date": DAYS[-1], "close": 50 * 1.002 ** (n - 1)},
                      judgment={"ref": judged.ref, "error": None}, read={"claims_noted": 1, "pages_read": 3, "error": None},
                      explanation="NO_EDGE: nothing tested carries weight, so the score is 0.00 and BUY needs +0.50.",
                      explained_by=JUDGE, unverified_numbers=["7%"], notes=[decide.NO_WEIGHT_NOTE])
    facts = [decision("XCO", "BUY", DAYS[-100], lean=1),         # scored: its own bet, a hit
             decision("XCO", "BUY", DAYS[-95], lean=1),          # inside that bet's window: overlapping
             decision("ZCO", "SELL", DAYS[-60], lean=-1),        # a separate bet, a hit
             decision("ZCO", "SELL", DAYS[-30], sha="deadbeefdeadbeef"),  # made under an edited D1
             latest]                                             # today: pending
    bank.add_many(facts)
    return {"fresh": fresh, "stale": stale, "judged": judged, "moved": moved, "landed": landed, "latest": latest,
            "old": facts[:2], "snap": last}


class DecBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.home = Path(cls._tmp.name)
        cls._old = os.environ.get("MONEY_HOME")
        os.environ["MONEY_HOME"] = str(cls.home)
        cls._patches = [mock.patch.object(config, "LEDGER", cls.home / "ledger.jsonl")]
        for p in cls._patches:
            p.start()
        cls.bank = Bank()
        cls.syn = build(cls.bank, cls.home)

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        if cls._old is None:
            os.environ.pop("MONEY_HOME", None)
        else:
            os.environ["MONEY_HOME"] = cls._old
        cls._tmp.cleanup()

    def run_ok(self, line, writes=False):
        r = data.run(self.bank, parse(line), writes=writes)
        self.assertTrue(r["ok"], (line, r.get("error"), r.get("hint")))
        json.dumps(r, allow_nan=False)
        for k in ("fn", "ticker", "args", "title", "asof", "view", "data"):
            self.assertIn(k, r)
        return r

    def run_err(self, line, writes=False):
        r = data.run(self.bank, parse(line), writes=writes)
        self.assertFalse(r["ok"], line)
        self.assertTrue(r["error"])
        json.dumps(r, allow_nan=False)
        return r

    def assertTableShape(self, t, line):
        keys = {c["key"] for c in t["columns"]}
        for row in t["rows"]:
            self.assertLessEqual({k for k in row if not k.startswith("_")}, keys, line)
            for k, cmd in row.get("_cmds", {}).items():
                self.assertIn(k, keys)
                c = parse(cmd)
                self.assertFalse(c.fn == "DEC" and c.args, f"{cmd}: a row command must never start a job")


class DecOneTest(DecBase):
    def test_the_latest_decision_with_its_evidence(self):
        r = self.run_ok("XCO DEC")
        self.assertEqual((r["view"], r["asof"], r["fn"], r["ticker"]), ("decision", today(), "DEC", "XCO"))
        self.assertEqual(r["title"], "XCO — Example Co · decision · NO_EDGE")
        d = r["data"]
        self.assertEqual((d["action"], d["label"], d["score"], d["threshold"]), ("NO_EDGE", "NO_EDGE: do nothing", 0.0, 0.5))
        self.assertEqual((d["policy"], d["policy_sha"], d["policy_current"]), ("D1", decide.POLICY_SHA, True))
        self.assertEqual((d["as_of"], d["cite"]), (today(), self.syn["latest"].id[:8]))
        self.assertEqual(d["price"]["cite"], self.syn["snap"].id[:8])
        items = {i["id"]: i for i in d["items"]}
        self.assertEqual(list(items), ["H1-mom12_1", f"{SITE} via agent", JUDGE])  # every item, weight 0 included
        self.assertEqual(items["H1-mom12_1"]["cites"], [])
        self.assertEqual(items[f"{SITE} via agent"]["cites"], [{"entity": "XCO", "cite": self.syn["fresh"].id[:8]}])
        self.assertEqual(items[JUDGE]["cites"], [{"entity": "XCO", "cite": self.syn["judged"].id[:8]}])
        self.assertEqual(items[f"{SITE} via agent"]["record"]["why"], "0 bets from 1 call, 1 pending; needs 30 bets")
        for i in d["items"]:
            self.assertEqual(set(i), {"kind", "id", "direction", "weight", "why", "cites", "record"})
        self.assertEqual([(c["cite"], c["attr"]) for c in d["context"]],
                         [(self.syn["stale"].id[:8], "claim"), (self.syn["moved"].id[:8], "alert")])
        self.assertEqual(d["context"][0]["why"], "stale: entry 39 sessions before this decision")
        self.assertEqual((d["lean"]["direction"], d["lean"]["text"]), (1, "up, 2 of 2 sources"))
        self.assertEqual((d["explained_by"], d["unverified_numbers"], d["explain_error"]), (JUDGE, ["7%"], None))
        self.assertTrue(d["explanation"].startswith("NO_EDGE: nothing tested"))
        self.assertEqual(d["judgment"], {"cite": self.syn["judged"].id[:8], "error": None})
        self.assertEqual(d["read"], {"claims_noted": 1, "pages_read": 3, "error": None})
        self.assertEqual(d["notes"], [decide.NO_WEIGHT_NOTE])

    def test_context_refs_go_through_shows_withholding(self):
        landed = self.syn["landed"]
        self.assertFalse(data.run_line(self.bank, f"XCO SHOW {landed.id[:8]}")["ok"])  # SHOW withholds it
        r = self.run_ok("XCO DEC")
        self.assertEqual(r["data"]["withheld"], 1)
        self.assertNotIn(landed.id[:8], json.dumps(r))
        self.assertNotIn("N2XC", json.dumps(r))

    def test_history_with_outcomes(self):
        h = self.run_ok("XCO DEC")["data"]["history"]
        self.assertTableShape(h, "XCO DEC")
        self.assertEqual([c["key"] for c in h["columns"]],
                         ["as_of", "action", "score", "lean", "excess", "hit", "status", "policy"])
        rows = h["rows"]
        self.assertEqual([r["as_of"] for r in rows], [today(), DAYS[-95], DAYS[-100]])  # newest first
        self.assertEqual([r["status"] for r in rows], ["pending", "overlapping", "scored"])
        self.assertEqual([r["hit"] for r in rows], [None, "hit", "hit"])
        self.assertIsNone(rows[0]["excess"])
        self.assertAlmostEqual(rows[2]["excess"], 1.002 ** 21 - 1, places=6)  # closes are stored to 6 decimals
        self.assertEqual(rows[2]["_cite"], self.syn["old"][0].id[:8])
        self.assertEqual((rows[0]["lean"], rows[0]["policy"]), ("up, 2 of 2 sources", "D1"))
        z = self.run_ok("ZCO DEC")["data"]["history"]["rows"]
        self.assertEqual(z[0]["policy"], "D1?deadbeefdeadbeef")  # an edited policy is never reported as D1
        self.assertFalse(self.run_ok("ZCO DEC")["data"]["policy_current"])

    def test_no_decision_yet(self):
        r = self.run_err("NOSUCH DEC")
        self.assertEqual((r["error"], r["hint"]), ("no decision recorded for NOSUCH yet", "NOSUCH DEC NEW"))
        hint = parse(r["hint"])
        self.assertEqual((hint.fn, hint.ticker, hint.args), ("DEC", "NOSUCH", ["NEW"]))  # the pane offers it as a click
        for t in ("SPY", "^GSPC"):  # D1 decides on stocks only: no job would help
            r = self.run_err(f"{t} DEC")
            self.assertTrue(r["error"].startswith(f"no decisions for {t}: "), r["error"])
            self.assertEqual(r["hint"], "DEC")

    def test_dec_new_never_runs_in_the_data_layer(self):
        self.assertTrue(data.is_write(parse("XCO DEC NEW")))
        self.assertFalse(data.is_write(parse("XCO DEC")))
        for writes in (False, True):
            r = self.run_err("XCO DEC NEW", writes=writes)
            self.assertIn("starts a background decision job", r["error"])
            self.assertEqual(r["hint"], "XCO DEC")
        r = data.run(self.bank, commands.Command(fn="DEC", ticker=None, args=["NEW"]))
        self.assertEqual((r["ok"], r["hint"]), (False, "DEC"))
        r = data.run(self.bank, commands.Command(fn="DEC", ticker="XCO", args=["FOO"]))
        self.assertEqual((r["ok"], r["error"]), (False, "DEC takes no arguments, or NEW"))


class DecAllTest(DecBase):
    def test_every_stock_the_scorecard_and_the_source_records(self):
        r = self.run_ok("DEC")
        self.assertEqual((r["view"], r["ticker"], r["asof"]), ("decisions", None, today()))
        d = r["data"]
        for k in ("latest", "scorecard", "sources"):
            self.assertTableShape(d[k], f"DEC {k}")
        latest = {row["ticker"]: row for row in d["latest"]["rows"]}
        self.assertEqual(list(latest), ["XCO", "ZCO"])
        x = latest["XCO"]
        self.assertEqual((x["action"], x["score"], x["lean"], x["as_of"], x["n"]),
                         ("NO_EDGE", 0.0, "up, 2 of 2 sources", today(), 3))
        self.assertEqual((x["_cmds"], x["_cite"], x["_ticker"]), ({"ticker": "XCO DEC"}, self.syn["latest"].id[:8], "XCO"))
        self.assertEqual(latest["ZCO"]["action"], "SELL")
        card = d["scorecard"]
        d1 = f"D1 · sha {decide.POLICY_SHA}"
        self.assertEqual(card["groups"], [d1, "D1?deadbeefdeadbeef · sha deadbeefdeadbeef · not the code's: the policy was edited"])
        rows = {(row["_group"], row["stream"]): row for row in card["rows"]}
        acts = rows[(d1, "BUY/SELL actions")]
        self.assertEqual((acts["bets"], acts["hits"], acts["dropped"], acts["proven"]), (2, 2, 1, "no"))
        self.assertEqual(rows[(d1, "NO_EDGE (what the stocks did)")]["bets"], 0)
        self.assertIn("D1: 1 pending", card["note"])
        self.assertIn("never a hit rate", card["note"])
        src = {row["source"]: row for row in d["sources"]["rows"]}
        self.assertEqual(set(src), {f"{SITE} via agent", JUDGE})  # claims and the model's judgments
        self.assertEqual((src[JUDGE]["bets"], src[JUDGE]["proven"], src[JUDGE]["weight"]), (0, "no", 0.0))
        self.assertIn("needs 30 bets", src[JUDGE]["why"])
        self.assertIn("The record D1 weighs", d["sources"]["note"])

    def test_an_empty_bank_answers_with_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = data.run(Bank(Path(tmp) / "bank"), parse("DEC"))
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"]["latest"]["rows"], [])
        self.assertIn("T DEC NEW makes one", r["data"]["latest"]["note"])
        self.assertIn("No directional calls", r["data"]["sources"]["note"])


class OtherScreensTest(DecBase):
    def test_wl_decision_column(self):
        watch.save({"tickers": ["XCO", "SPY"]})
        try:
            d = self.run_ok("WL")["data"]
        finally:
            watch.watchlist_path().unlink()
        self.assertTableShape(d, "WL")
        self.assertEqual([c["key"] for c in d["columns"]],
                         ["item", "name", "last", "chg_1d", "decision", "alert", "alert_at", "measure", "date"])
        rows = {row["item"]: row for row in d["rows"]}
        self.assertEqual(rows["XCO"]["decision"], "NO_EDGE")
        self.assertEqual(rows["XCO"]["_cites"]["decision"], self.syn["latest"].id[:8])
        self.assertEqual(rows["XCO"]["_cmds"]["decision"], "XCO DEC")
        self.assertIsNone(rows["SPY"]["decision"])  # empty when none was recorded
        self.assertNotIn("decision", rows["SPY"].get("_cites", {}))

    def test_cn_shows_the_record_d1_weighs_beside_every_call(self):
        c = next(x for x in self.run_ok("XCO CN")["data"]["claims"] if x["cite"] == self.syn["fresh"].id[:8])
        self.assertEqual(c["gate_key"], f"{SITE} via agent")
        self.assertEqual(set(c["gate"]), {"bets", "hits", "hit_rate", "q", "proven", "weight", "why"})
        # the old claim's window has closed (one bet); today's is pending
        self.assertEqual((c["gate"]["bets"], c["gate"]["proven"], c["gate"]["weight"]), (1, False, 0.0))
        self.assertEqual(c["gate"]["why"], "1 bet (1 hit) from 2 calls; needs 30 bets")
        self.assertIn("record", c)  # every call, overlapping: claims.score, kept beside it
        proven = {f"{SITE} via agent": {"bets": 52, "hits": 41, "hit_rate": 41 / 52, "p": 1e-5, "q": 2e-5, "lower": 0.68,
                                        "mean_signed_excess": 0.01, "up": 40, "down": 12, "calls": 60, "dropped": 8,
                                        "pending": 0, "unscorable": 0, "proven": True, "weight": 0.36,
                                        "why": "proven: 52 bets, 79% hit, weight 0.36"}}
        with mock.patch.object(claims, "track_record", return_value=proven) as tr:
            c = next(x for x in self.run_ok("XCO CN")["data"]["claims"] if x["cite"] == self.syn["fresh"].id[:8])
        self.assertTrue(tr.called)
        self.assertEqual((c["gate"]["proven"], c["gate"]["weight"], c["gate"]["why"]),
                         (True, 0.36, "proven: 52 bets, 79% hit, weight 0.36"))
        self.assertEqual(self.run_ok("ZCO CN")["data"]["claims"], [])

    def test_clm_shows_both_records_labelled(self):
        d = self.run_ok("CLM")["data"]
        self.assertTableShape(d, "CLM")
        self.assertEqual(d["groups"], ["The record D1 weighs", "Every call, overlapping"])
        weighed = {r["source"]: r for r in d["rows"] if r["_group"] == "The record D1 weighs"}
        every = {r["source"]: r for r in d["rows"] if r["_group"] == "Every call, overlapping"}
        self.assertEqual(set(weighed), {f"{SITE} via agent", JUDGE})
        self.assertEqual(set(every), {SITE})  # claims.score counts claims, overlapping ones included
        self.assertEqual(every[SITE]["calls"], 1)  # the stale claim's window has closed; today's is pending
        for k in ("bets", "q", "proven", "weight", "why"):
            self.assertIn(k, weighed[JUDGE])
        self.assertIn("The record D1 weighs (claims.track_record)", d["note"])
        self.assertIn("Every call, overlapping", d["note"])

    def test_no_nan_anywhere(self):
        for line in ("XCO DEC", "ZCO DEC", "DEC", "XCO CN", "CLM", "HELP"):
            self.run_ok(line)
        codes = [f["code"] for f in self.run_ok("HELP")["data"]["functions"]]
        self.assertIn("DEC", codes)


class ShowWithholdsInputsTest(TempHome):
    """SHOW of a fact made from an aircraft fact withheld since: a real decision that cited a company aircraft's
    landing alert while its tail was tracked, and a judgment that cited the flight. Once the tail opts out,
    neither names it, however SHOW is reached (the DEC header's citation, /api/fact, the analyst's show_fact)."""

    def setUp(self):
        super().setUp()
        end = pd.Timestamp(today()) - pd.offsets.BDay(1)
        days = pd.bdate_range(end=end, periods=60).strftime("%Y-%m-%d").tolist()
        self.snap = price_snapshot("XCO", days, [50.0] * 60)
        at = days[-1] + "T18:00:00Z"
        self.flight = Fact(entity="XCO", attr="aircraft.flight", value="KTEB", as_of=at[:10], known_at=at,
                           tier="reported", source={"name": "adsb.lol"},
                           context={"hex": "a00002", "reg": "N2XC", "from": "KBOS", "to": "KTEB", "departed": at,
                                    "arrived": at}, retrieved_at=at)
        self.landed = Fact(entity="XCO", attr="alert", value="XCO aircraft N2XC landed at KTEB from KBOS",
                           as_of=at[:10], known_at=at, tier="derived", source={"name": "brain.watch", "rule": "aircraft"},
                           inputs=(self.flight.ref,), retrieved_at=at,
                           context={"rule": "aircraft", "severity": "notice", "keys": ["aircraft:a00002:" + at],
                                    "reg": "N2XC", "hex": "a00002"})
        self.bank.add_many([self.snap, price_snapshot("SPY", days, [400.0] * 60), self.flight,
                            filed("XCO", "identity.name", "Example Co", days[0], days[0])])

    def assertNoTail(self, out, where):
        text = (out if isinstance(out, str) else json.dumps(out)).lower()
        for tail in ("n2xc", "a00002"):
            self.assertNotIn(tail, text, where)

    def test_a_decision_that_cited_a_landing_alert(self):
        self.bank.add(self.landed)
        d = decide.decide(self.bank, "XCO", do_judge=False, do_explain=False, log=lambda m: None)
        self.assertTrue(d["ref"], d["notes"])
        self.assertIn(self.landed.ref, self.bank.get(d["ref"]).inputs)  # tracked when decided: an input
        cite = d["ref"].partition("/")[2][:8]
        before = data.run_line(self.bank, f"XCO SHOW {cite}")["data"]
        self.assertEqual(before["withheld"], 0)
        self.assertIn(self.landed.value, [i["value"] for i in before["inputs"]])
        aviation.optout_path().write_text("N2XC\n")  # what `brain jets optout N2XC` does: the list, then forget
        self.assertEqual(aviation.forget(self.bank, tickers=["XCO"]), 2)  # the flight and the alert
        from brain.terminal import analyst
        srv, _ = server.make_server("127.0.0.1", 0, bank=self.bank)
        srv.quiet = True
        try:
            reader = lambda line: srv.run_line(line, read_only=True)  # noqa: E731
            tools = {t.name: t for t in analyst._tools(reader, [])}
            seen = {"SHOW": data.run_line(self.bank, f"XCO SHOW {cite}"),
                    "show_fact": tools["show_fact"].call({"ticker": "XCO", "fact_id": cite}),
                    "DEC": srv.run_line("XCO DEC")}
        finally:
            srv.server_close()
        r = seen["SHOW"]
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["data"]["withheld"], 1)
        self.assertNotIn(self.landed.ref, [i["ref"] for i in r["data"]["inputs"]])
        self.assertIn(self.snap.ref, [i["ref"] for i in r["data"]["inputs"]])  # the rest is listed
        self.assertEqual(seen["DEC"]["data"]["withheld"], 1)
        for where, out in seen.items():
            self.assertNoTail(out, where)

    def test_a_judgment_that_cited_a_flight(self):
        at = self.flight.known_at
        j = Fact(entity="XCO", attr="judgment", value="XCO: up; its jet N2XC flew to KTEB", as_of=at[:10], known_at=at,
                 tier="opinion", source={"name": JUDGE}, inputs=(self.flight.ref, self.snap.ref), retrieved_at=at,
                 context={"direction": 1, "horizon_days": 21, "confidence": 0.6, "policy": "D1", "unverified_numbers": [],
                          "reasons": [{"text": "n2xc landed at KTEB from KBOS", "cites": [self.flight.ref]},
                                      {"text": "the close held at 50", "cites": [self.snap.ref]}]})
        self.bank.add(j)
        before = data.run_line(self.bank, f"XCO SHOW {j.id[:8]}")["data"]
        self.assertEqual((before["withheld"], len(before["inputs"])), (0, 2))
        aviation.optout_path().write_text("N2XC  # asked not to be tracked\n")  # by hand: nothing withdrawn yet
        r = data.run_line(self.bank, f"XCO SHOW {j.id[:8]}")
        self.assertTrue(r["ok"], r)
        d = r["data"]
        self.assertEqual(d["withheld"], 1)
        self.assertEqual([i["ref"] for i in d["inputs"]], [self.snap.ref])
        self.assertEqual(d["fact"]["value"], "XCO: up; its jet [withheld] flew to KTEB")
        self.assertEqual([x["text"] for x in d["fact"]["context"]["reasons"]],  # a reason resting on the flight goes
                         ["[withheld: it cites an aircraft that is no longer tracked]", "the close held at 50"])
        self.assertNoTail(r, "SHOW of the judgment")
        self.assertFalse(data.run_line(self.bank, f"XCO SHOW {self.flight.id[:8]}")["ok"])  # the flight itself
        self.assertEqual(data.run_line(self.bank, f"XCO SHOW {self.snap.id[:8]}")["data"]["withheld"], 0)


def fake_argv(target: str) -> list[str]:
    return [sys.executable, "-c", f"print('fake job {target}')"]


def printing(*lines: str, code: int = 0) -> list[str]:
    """A job's argv that prints ``lines`` and exits with ``code``."""
    return [sys.executable, "-c", "import sys\n" + "".join(f"print({x!r})\n" for x in lines) + f"sys.exit({code})"]


def no_data_card(target: str) -> list[str]:
    """What ``brain decide T`` prints when it decides nothing: no prices, and no ingest could fetch them."""
    t = jobs.ticker_of(target)
    return printing(f"ingest of {t} failed (FetchError: HTTP 404)",
                    f"{t}  NO_DATA: cannot decide   (policy D1 · paper only · not financial advice)",
                    f"Nothing decided: no prices for {t}: ingest {t}.", "Why: There is no data to decide on.",
                    "Not recorded: NO_DATA has nothing to score against.")


class DecideJobTest(TempHome):
    def test_argv_target_and_validation(self):
        exe = sys.executable
        self.assertEqual(jobs.decide_target("aapl"), "DEC AAPL")
        self.assertEqual(jobs.validate("dec brk.b"), "DEC BRK.B")
        self.assertEqual(jobs.argv_for("DEC AAPL"), [exe, "-m", "brain", "decide", "AAPL", "--ingest", "--read"])
        self.assertEqual(jobs.argv_for("AAPL"), [exe, "-m", "brain", "ingest", "AAPL"])
        self.assertEqual(jobs.validate("DEC"), "DEC")  # Diversified Energy: an ingest of the ticker DEC
        self.assertEqual((jobs.kind_of("DEC AAPL"), jobs.kind_of("DEC"), jobs.kind_of("MARKETS")),
                         ("decide", "ingest", "ingest"))
        self.assertEqual((jobs.ticker_of("DEC AAPL"), jobs.ticker_of("AAPL"), jobs.ticker_of("WATCH")), ("AAPL", "AAPL", None))
        self.assertEqual((jobs.is_ticker("DEC AAPL"), jobs.is_ticker("AAPL"), jobs.is_ticker("MACRO")), (False, True, False))
        for t in ("DEC -X", "DEC A;B", "DEC  AAPL", "DEC ", "DEC A B", "DEC AAPL\n"):
            with self.subTest(t=t), self.assertRaises(jobs.JobError):
                jobs.validate(t)

    def test_one_job_per_kind_and_ticker_in_a_slot_of_its_own(self):
        gate = Path(self._tmp.name) / "go"
        wait = [sys.executable, "-c", f"import os, time\nwhile not os.path.exists({str(gate)!r}): time.sleep(0.02)"]
        m = jobs.JobManager(argv_for=lambda t: wait)
        try:
            ing = [m.submit(t) for t in ("A", "B", "C")]               # two ingest slots: C waits
            dec = [m.submit(jobs.decide_target(t)) for t in ("A", "B")]  # one decision slot: B waits
            self.assertEqual([j["status"] for j in ing], ["running", "running", "queued"])
            self.assertEqual([(j["status"], j["kind"]) for j in dec], [("running", "decide"), ("queued", "decide")])
            self.assertEqual((m.running(), m.queued()), (3, 2))  # the decision never takes an ingest slot
            self.assertEqual(m.submit("dec a")["id"], dec[0]["id"])  # the same (kind, ticker): the same job
            self.assertNotEqual(m.submit("a")["id"], dec[0]["id"])   # an ingest of A is another kind
            self.assertEqual(m.submit("a")["id"], ing[0]["id"])
            # what /api/status reports: the page tells a queued decision from a finished one by these
            self.assertEqual(m.summary(), {"running": 3, "queued": 2, "active": ["A", "B", "C", "DEC A", "DEC B"],
                                           "finished": 0})
            gate.touch()
            self.assertTrue(m.wait(30))
            self.assertEqual({j["status"] for j in m.list_jobs()}, {"done"})
            self.assertEqual({j["kind"] for j in m.list_jobs()}, {"ingest", "decide"})
            self.assertEqual(m.summary(), {"running": 0, "queued": 0, "active": [], "finished": 5})
        finally:
            gate.touch()
            m.shutdown()

    def test_decision_jobs_wait_their_turn_without_holding_up_ingest(self):
        order, lock = [], threading.Lock()
        gate = Path(self._tmp.name) / "release"

        def argv(t):
            code = (f"import os, time\nprint('start {t}')\n"
                    + (f"while not os.path.exists({str(gate)!r}): time.sleep(0.02)\n" if t.startswith("DEC") else ""))
            return [sys.executable, "-c", code]

        def finished(job):
            with lock:
                order.append(job["target"])
        m = jobs.JobManager(on_finish=finished, argv_for=argv)
        try:
            m.submit_many(["DEC A", "DEC B"])
            m.submit_many(["X", "Y", "Z"])
            end = time.monotonic() + 30
            while len(order) < 3 and time.monotonic() < end:  # every ingest finishes while a decision blocks
                time.sleep(0.02)
            self.assertEqual(sorted(order), ["X", "Y", "Z"])
            self.assertEqual([j["status"] for j in m.list_jobs() if j["kind"] == "decide"], ["queued", "running"])
            gate.touch()
            self.assertTrue(m.wait(30))
            self.assertEqual(order[3:], ["DEC A", "DEC B"])  # one at a time, in order
        finally:
            gate.touch()
            m.shutdown()

    def test_the_jobs_table_shows_a_decision_job(self):
        t = jobs.table([{"id": "j2", "target": "DEC AAPL", "kind": "decide", "status": "running", "started": "s",
                         "finished": None, "returncode": None, "log": ["AAPL  NO_EDGE: do nothing"]},
                        {"id": "j1", "target": "AAPL", "kind": "ingest", "status": "done", "started": "s",
                         "finished": "f", "returncode": 0, "log": []}])
        dec, ing = t["rows"]
        self.assertEqual((dec["target"], dec["_ticker"], dec["_cmds"]), ("DEC AAPL", "AAPL", {"target": "AAPL DEC"}))
        self.assertEqual((ing["_ticker"], "_cmds" in ing), ("AAPL", False))
        self.assertIn("1 decision at once", t["note"])
        self.assertIn("Log of j2 (DEC AAPL)", t["note"])
        empty = jobs.table([])
        self.assertIn("AAPL DEC NEW", empty["note"])
        odd = jobs.table([{"id": "j3", "target": "DEC ASK", "kind": "decide", "status": "done", "started": "s",
                           "finished": "f", "returncode": 0, "log": []}])["rows"][0]
        self.assertNotIn("_cmds", odd)  # `ASK DEC` would put a question to the model: no command at all


class DecServerTest(TempHome):
    def setUp(self):
        super().setUp()
        self.server, _ = server.make_server("127.0.0.1", 0, bank=self.bank)
        self.server.quiet = True
        self.server.jobs.argv_for = fake_argv

    def tearDown(self):
        self.assertTrue(self.server.jobs.wait(20))
        self.server.server_close()
        super().tearDown()

    def test_dec_new_starts_a_decide_job_and_answers_with_jobs(self):
        out = self.server.run_line("aapl dec new")
        self.assertTrue(out["ok"], out)
        self.assertEqual((out["fn"], out["ticker"], out["args"], out["view"]), ("DEC", "AAPL", ["NEW"], "table"))
        self.assertTrue(out["title"].startswith("DEC NEW AAPL: job j1 "), out["title"])
        self.assertIn("AAPL DEC shows the decision", out["title"])
        self.assertEqual(out["data"]["rows"][0]["target"], "DEC AAPL")
        again = self.server.run_line("DEC NEW AAPL")  # de-duplicated while queued or running
        self.assertTrue(again["title"].startswith("DEC NEW AAPL: job j1 "), again["title"])
        self.assertTrue(self.server.jobs.wait(20))
        job = self.server.jobs.list_jobs()[0]
        self.assertEqual((job["target"], job["kind"], job["status"], job["log"]),
                         ("DEC AAPL", "decide", "done", ["fake job DEC AAPL"]))
        self.assertEqual(len(self.server.jobs.list_jobs()), 1)
        jobs_view = self.server.run_line("JOBS")
        self.assertEqual(jobs_view["data"]["rows"][0]["_cmds"], {"target": "AAPL DEC"})

    def test_dec_says_when_the_last_decision_job_recorded_nothing(self):
        self.server.jobs.argv_for = no_data_card
        self.assertTrue(self.server.run_line("ZZZQ DEC NEW")["ok"])
        self.assertTrue(self.server.jobs.wait(20))
        self.assertEqual(self.server.jobs.list_jobs()[0]["status"], "done")  # NO_DATA exits 0
        why = "the last ZZZQ DEC NEW (job j1, done) recorded nothing: no prices for ZZZQ: ingest ZZZQ"
        out = self.server.run_line("ZZZQ DEC")
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], f"no decision recorded for ZZZQ yet; {why}")
        self.assertEqual(out["hint"], "JOBS")  # not ZZZQ DEC NEW again: that job would end the same way
        self.assertEqual(self.server.run_line("ZZZQ DEC", read_only=True)["hint"], "JOBS")  # the analyst reads it too
        self.bank.add(price_snapshot("ZZZQ", DAYS[-30:], [10.0] * 30))
        self.bank.add(decision("ZZZQ", "NO_EDGE", DAYS[-1]))  # an older decision: shown, with the note beside it
        out = self.server.run_line("ZZZQ DEC")
        self.assertTrue(out["ok"], out)
        self.assertEqual((out["data"]["as_of"], out["data"]["job_note"]), (DAYS[-1], why))
        mine = decision("ZZZQ", "NO_EDGE", today())
        self.bank.add(Fact.from_dict({**mine.to_dict(), "retrieved_at": now_iso()}))  # one a job records: no note
        out = self.server.run_line("ZZZQ DEC")
        self.assertEqual(out["data"]["as_of"], today())
        self.assertNotIn("job_note", out["data"])
        self.assertNotIn("job_note", self.server.run_line("DEC")["data"])

    def test_dec_says_when_a_decision_job_is_at_work_or_failed(self):
        gate = Path(self._tmp.name) / "go"
        self.server.jobs.argv_for = lambda t: [
            sys.executable, "-c", f"import os, sys, time\nwhile not os.path.exists({str(gate)!r}): time.sleep(0.02)\n"
            "print('Traceback (most recent call last):')\nprint('RuntimeError: boom')\nsys.exit(1)"]
        self.bank.add(price_snapshot("XCO", DAYS[-30:], [10.0] * 30))
        self.bank.add(decision("XCO", "SELL", DAYS[-1]))
        try:
            self.assertTrue(self.server.run_line("XCO DEC NEW")["ok"])
            self.assertTrue(self.server.run_line("ZCO DEC NEW")["ok"])  # one decision at a time: ZCO waits
            out = self.server.run_line("XCO DEC")
            self.assertTrue(out["ok"], out)
            self.assertEqual((out["data"]["action"], out["data"]["job_note"]),
                             ("SELL", "a new decision on XCO is running (job j1); XCO DEC shows it when it is done"))
            out = self.server.run_line("ZCO DEC")
            self.assertEqual((out["ok"], out["hint"]), (False, "JOBS"))
            self.assertEqual(out["error"], "no decision recorded for ZCO yet; a new decision on ZCO is queued (job j2); "
                                           "ZCO DEC shows it when it is done")
            self.assertEqual(self.server.run_line("XCO DEC FOO")["error"], "DEC takes no arguments, or NEW")
        finally:
            gate.touch()
        self.assertTrue(self.server.jobs.wait(20))
        out = self.server.run_line("XCO DEC")
        self.assertEqual(out["data"]["job_note"], "the last XCO DEC NEW (job j1, failed) recorded nothing: RuntimeError: boom")

    def test_the_real_argv_of_a_decision_job(self):
        self.server.jobs.argv_for = jobs.argv_for
        with mock.patch.object(jobs.JobManager, "_follow", lambda m, job: (job.log.append(" ".join(job.argv[1:])), 0)[1]):
            self.assertTrue(self.server.run_line("MSFT DEC NEW")["ok"])
            self.assertTrue(self.server.jobs.wait(20))
        self.assertEqual(self.server.jobs.list_jobs()[0]["log"], ["-m brain decide MSFT --ingest --read"])

    def test_refusals(self):
        for line in ("DEC NEW", "SPY DEC NEW", "^GSPC DEC NEW"):
            with self.subTest(line=line):
                out = self.server.run_line(line)
                self.assertFalse(out["ok"])
        self.assertEqual(self.server.run_line("DEC NEW")["error"], "DEC NEW needs a ticker, e.g. AAPL DEC NEW")
        self.assertIn("benchmark", self.server.run_line("SPY DEC NEW")["error"])
        out = self.server.run_line("AAPL DEC NEW", read_only=True)  # the analyst's reader
        self.assertEqual((out["ok"], out["error"]), (False, "DEC NEW is not available here (read-only)"))
        self.assertEqual(self.server.jobs.list_jobs(), [])
        self.assertFalse(self.server.run_line("AAPL DEC", read_only=True)["ok"])  # reading is allowed: no decision yet
        self.assertEqual(self.server.run_line("AAPL DEC", read_only=True)["hint"], "AAPL DEC NEW")

    def test_the_analyst_reads_decisions_and_cannot_start_one(self):
        from brain.terminal import analyst
        self.bank.add(price_snapshot("XCO", DAYS[-30:], [10.0] * 30))
        self.bank.add(decision("XCO", "NO_EDGE", today(), explanation="Nothing tested carries weight."))
        reader = lambda line: self.server.run_line(line, read_only=True)  # noqa: E731
        calls = []
        tools = {t.name: t for t in analyst._tools(reader, calls)}
        out = json.loads(tools["decision"].call({"ticker": "xco"}))
        self.assertEqual((out["ok"], out["view"], out["data"]["action"]), (True, "decision", "NO_EDGE"))
        self.assertEqual(calls, [{"name": "decision", "input": "XCO DEC"}])
        self.assertFalse(reader("XCO DEC NEW")["ok"])
        self.assertEqual(self.server.jobs.list_jobs(), [])


class DecHttpTest(TempHome):
    """/api/ingest takes ingest targets only: a decision starts from the command line, T DEC NEW."""

    def test_api_ingest_refuses_a_decision_target(self):
        import urllib.error
        import urllib.request
        srv, token = server.make_server("127.0.0.1", 0, bank=self.bank)
        srv.quiet = True
        srv.jobs.argv_for = fake_argv
        th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        th.start()
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for target, code in (("DEC AAPL", 400), ("dec aapl", 400), ("DEC", 200)):
                req = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/api/ingest", method="POST",
                                             data=json.dumps({"target": target}).encode(),
                                             headers={"Host": f"127.0.0.1:{srv.server_port}", "X-Terminal-Token": token,
                                                      "Content-Type": "application/json"})
                try:
                    with opener.open(req, timeout=20) as resp:
                        got = resp.status
                except urllib.error.HTTPError as e:
                    with e:
                        got = e.code
                self.assertEqual(got, code, target)
            self.assertTrue(srv.jobs.wait(20))
            self.assertEqual([j["target"] for j in srv.jobs.list_jobs()], ["DEC"])  # the ticker DEC, ingested
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(5)


if __name__ == "__main__":
    unittest.main()
