"""The decision engine (docs/DECIDE.md section 3): policy D1, signal rules, sources, the data gate, recording,
change alerts, the model steps' availability, scoring decisions and the CLI. The model steps are stood in for
by a fake brain.judge, so nothing here needs a model server."""
import contextlib
import hashlib
import io
import json
import math
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from brain import brief, check, claims, cli, config, decide, series, universe, watch
from brain.bank import Bank, Fact
from brain.metrics import PriceBook
from brain.research.hypotheses import BROAD, current
from brain.timeutil import now_iso, today
from tests.util import TempHome, price_snapshot

H1, H6 = "H1-mom12_1", "H6-insider_buying"
WITHDRAWN = ["H3b-earnings_yield_ttm", "H7-earnings_reaction", "H8-sue"]  # as the ledger records them
LARGE = "universe/us_large.json"


def sessions(n: int = 300, end: str | None = None) -> list[str]:
    """n weekdays ending today (today itself included even on a weekend: a close ingested this evening)."""
    end = end or today()
    days = pd.bdate_range(end=end, periods=n).strftime("%Y-%m-%d").tolist()
    return days if days[-1] == end else days[1:] + [end]


def claim(t, host, direction, known, horizon=21, noted_by=None, retrieved=None):
    """A claim as claims.note writes it, with its retrieval time set (note stamps the real now)."""
    known = known if "T" in known else known + "T15:00:00Z"
    ctx = {"direction": direction, "horizon_days": horizon, **({"noted_by": noted_by} if noted_by else {})}
    return Fact(entity=t, attr="claim", value=f"{host} says {direction}", as_of=known[:10], known_at=known,
                tier="opinion", source={"name": host, "url": f"https://{host}/{t}/{known}"}, context=ctx,
                retrieved_at=retrieved or known)


def row(hid, verdict="FAIL", held=None, broad=False, net=-0.0011, p=0.88):
    r = {"signal": hid.split("-", 1)[1], "hypothesis": hid, "value": 0.1, "percentile": 50, "in_top_third": None,
         "verdict": verdict, "test_net_excess": net, "perm_p": p, "bh_q": 0.9, "claim": ""}
    if broad:
        r.update(universe=BROAD, universe_size=6, select="positive", held=held)
    else:
        r["in_top_third"] = held
    return r


class FakeJudge:
    """Stands in for brain.judge: records what it was asked."""

    def __init__(self, ready=True, assess=None, explain=None):
        self.ready = ready
        self.asked, self.explained = [], []
        self._assess, self._explain = assess, explain

    def llm_ready(self):
        return (True, "") if self.ready else (False, "nothing answered at http://127.0.0.1:11434/v1")

    def assess(self, bank, ticker, client=None, log=print, deadline=None):
        self.asked.append(ticker)
        if self._assess:
            return self._assess(bank, ticker)
        return {"ok": False, "fact": None, "ref": None, "direction": None, "error": "no call", "down": None}

    def explain(self, decision, client=None, deadline=None):
        self.explained.append(decision["ticker"])
        if self._explain:
            return self._explain(decision)
        return {"text": self.template_explanation(decision), "by": "template", "unverified_numbers": [], "error": None,
                "down": None}

    @staticmethod
    def template_explanation(decision):
        return f"{decision['ticker']}: {decision['action']} (template)"


class DecideCase(TempHome):
    def setUp(self):
        super().setUp()
        self.home = Path(self._tmp.name)
        self.days = sessions()
        self.t_run = today() + "T23:59:59Z"
        rng = np.random.default_rng(5)
        for t in ("X", "Y", "SPY"):
            self.bank.add(price_snapshot(t, self.days, list(50 * np.cumprod(1 + rng.normal(0, 0.01, len(self.days))))))
        self.month_end = max(d for d in self.days if d[:7] < today()[:7])
        (self.home / "large.json").write_text(json.dumps({"tickers": ["X", "Y", "Z", "W"]}))
        (self.home / "broad.json").write_text(json.dumps({"tickers": ["X", "Y", "Z", "W", "V", "U"]}))
        recs = [{"op": "withdraw", "id": h, "reason": "a code mistake", "at": "2026-01-02T00:00:00Z"} for h in WITHDRAWN]
        (self.home / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        self.ev = self.evidence()
        self.judge = FakeJudge(ready=False)
        self.patches = [mock.patch.object(config, "UNIVERSE", self.home / "large.json"),
                        mock.patch.object(config, "LEDGER", self.home / "ledger.jsonl"),
                        mock.patch.object(universe, "BROAD", self.home / "broad.json"),
                        mock.patch.object(brief, "evidence", side_effect=self.fake_evidence),
                        mock.patch.object(decide, "_judge_module", side_effect=lambda: self.judge)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        super().tearDown()

    def fake_evidence(self, bank, t, budget_s=None, asof=None):
        self.evidence_asof = asof  # decide reads the ranking at T_RUN, like every other read in the run
        return self.ev

    def evidence(self, rows=None, date=None, cov=4, bcov=6, bdate=None):
        """What brief.evidence returns: H1-H4 against the large universe, H6 and kin against the broad one."""
        return {"ticker": "X", "rebalance_date": date or self.month_end, "universe": LARGE, "universe_size": 4,
                "coverage": {"ranked": cov, "universe_size": 4},
                "signals": rows if rows is not None else [row(H1), row(H6, broad=True, held=False)],
                "broad": {"universe": BROAD, "status": "complete", "universe_size": 6,
                          "rebalance_date": bdate or self.month_end, "computed": 6, "total": 6,
                          "coverage": {"ranked": bcov, "universe_size": 6}}}

    def go(self, t="X", **kw):
        kw = {"do_judge": False, "do_explain": False, "log": lambda m: None, **kw}
        return decide.decide(self.bank, t, **kw)

    def items(self, d) -> dict:
        return {i["id"]: i for i in d["items"]}


class PolicyTest(DecideCase):
    def test_d1_is_frozen(self):
        self.assertEqual(decide.POLICY_SHA, "bef26a9d157cbcf5")  # D1 is never edited: a change is D2
        self.assertEqual(decide.POLICY_SHA,
                         hashlib.sha256(json.dumps(decide.POLICY, sort_keys=True).encode()).hexdigest()[:16])
        self.assertEqual((decide.POLICY["id"], decide.POLICY["threshold"], decide.POLICY["horizon_days"]), ("D1", 0.5, 21))
        # The unpriced rule lives in claims.py; D1 names its values so a change to them is a new policy too.
        gate = decide.POLICY["source_gate"]
        self.assertEqual((gate["grace_sessions"], gate["unpriced_max_share"]),
                         (claims.GRACE_SESSIONS, claims.UNPRICED_MAX_SHARE))

    def test_scorable(self):
        for t in ("SPY", "^GSPC", "EURUSD=X", "BTC-USD", "^VIX", "gc=f"):
            self.assertFalse(decide.scorable(t)[0], t)
            self.assertTrue(decide.scorable(t)[1])
        for t in ("AAPL", "QQQ", "VOO"):
            self.assertEqual(decide.scorable(t), (True, ""))

    def test_thresholds(self):
        self.assertEqual([decide.action_for(s) for s in (0.5, 1.0 - 0.5, 1.0, -0.5, -1.0, 0.49, -0.49, 0.0)],
                         ["BUY", "BUY", "BUY", "SELL", "SELL", "NO_EDGE", "NO_EDGE", "NO_EDGE"])


class SignalRulesTest(DecideCase):
    def test_each_rule_failing_alone_names_itself(self):
        passing = dict(rows=[row(H1, "PASS", held=True, net=0.004, p=0.01)])
        d = self.go(**{})
        self.assertEqual(d["action"], "NO_EDGE")
        self.assertEqual(self.evidence_asof, self.t_run)  # the ranking is read at T_RUN, as the data gate is
        self.ev = self.evidence(**passing)
        d = self.go()
        h1 = self.items(d)[H1]
        self.assertEqual((d["action"], h1["weight"], h1["direction"]), ("BUY", 1.0, 1))
        self.assertIn("tests D1, not the hypothesis", h1["why"])
        self.assertEqual(h1["record"], {"verdict": "PASS", "universe": LARGE, "rebalance_date": self.month_end,
                                        "coverage": {"ranked": 4, "universe_size": 4}, "held": True})
        cases = {
            "a": (self.evidence(rows=[row(H1, "FAIL", held=True, net=-0.0011, p=0.88)]),
                  "FAIL on TEST (net excess -0.11%/mo, p 0.88)"),
            "c": (self.evidence(date="2026-01-30", **passing), "ranking as of 2026-01-30: ingest SPY and the universe"),
            "d": (self.evidence(cov=1, **passing), "ranking covers 1 of 4 stocks"),
            "e": (self.evidence(rows=[row(H1, "PASS", held=False)]), "PASS, but its backtest would not hold this stock"),
        }
        for rule, (ev, why) in cases.items():
            with self.subTest(rule=rule):
                self.ev = ev
                h1 = self.items(self.go())[H1]
                self.assertEqual((h1["weight"], h1["direction"], h1["why"]), (0.0, 0, why))
        self.ev = self.evidence(rows=[row(H1, "not run", held=True)])
        self.assertEqual(self.items(self.go())[H1]["why"], "not run")
        # (b): the stock is ranked (evidence adds it) but its universe file does not list it
        self.ev = self.evidence(**passing)
        (self.home / "large.json").write_text(json.dumps({"tickers": ["Y", "Z", "W"]}))
        h1 = self.items(self.go())[H1]
        self.assertEqual((h1["weight"], h1["why"]), (0.0, "PASS on universe/us_large.json, which does not list this stock"))

    def test_broad_rows_use_the_broad_ranking(self):
        self.ev = self.evidence(rows=[row(H6, "PASS", held=True, broad=True)])
        h6 = self.items(self.go())[H6]
        self.assertEqual((h6["weight"], h6["record"]["universe"], h6["record"]["coverage"]["universe_size"]), (1.0, BROAD, 6))
        self.ev = self.evidence(rows=[row(H6, "PASS", held=True, broad=True)], bcov=2)
        self.assertEqual(self.items(self.go())[H6]["why"], "ranking covers 2 of 6 stocks")
        self.ev = self.evidence(rows=[row(H6, "PASS", held=True, broad=True)], date="2026-01-30")  # H1-H4's date only
        self.assertEqual(self.items(self.go())[H6]["weight"], 1.0)

    def test_missing_rows_still_get_items(self):
        self.ev = {**self.evidence(rows=[row(H1)]), "broad": {"universe": BROAD, "status": "unavailable", "rows": [],
                                                              "reason": "the bank holds 3 of its 741 stocks"}}
        d = self.go()
        it = self.items(d)
        self.assertEqual([i["id"] for i in d["items"] if i["kind"] == "signal"], [h["id"] for h in current(WITHDRAWN)])
        self.assertEqual(it["H5-ridge"]["why"], "not run; not ranked per stock")
        self.assertEqual(it["H6-insider_buying"]["why"], "not run; the bank holds 3 of its 741 stocks")
        self.assertNotIn("H7-earnings_reaction", it)  # withdrawn, with a successor
        for i in d["items"]:
            self.assertEqual(set(i), {"kind", "id", "direction", "weight", "why", "refs", "record"})
        with mock.patch.object(brief, "evidence", side_effect=brief.NoEvidence("no research universe (none.json)")):
            d = self.go()
        self.assertEqual([(i["kind"], i["id"], i["weight"]) for i in d["items"]], [("signal", "evidence", 0.0)])
        self.assertIn("no research universe", d["items"][0]["why"])


class SourcesTest(DecideCase):
    def test_lag_window_and_other_horizons(self):
        d0 = self.days
        facts = [claim("X", "a.example", 1, d0[-1 - 5]),            # entry 5 sessions before: counts
                 claim("X", "a.example", 1, d0[-1 - 2], noted_by="scout:qwen3:8b"),
                 claim("X", "b.example", -1, d0[-1 - 6]),           # 6: stale
                 claim("X", "c.example", 1, d0[-1 - 18]),           # 18: stale
                 claim("X", "d.example", 1, d0[-1 - 30]),           # its horizon is over: not even context
                 claim("X", "e.example", 1, d0[-1 - 3], horizon=63),
                 claim("X", "f.example", 0, d0[-1 - 1]),            # no direction, in the window
                 claim("X", "g.example", 0, d0[-1 - 9])]            # no direction, outside it
        self.bank.add_many(facts)
        d = self.go()
        src = {i["id"]: i for i in d["items"] if i["kind"] == "source"}
        self.assertEqual(set(src), {"a.example via agent", "a.example via scout:qwen3:8b"})
        self.assertEqual(src["a.example via agent"]["refs"], [facts[0].ref])
        why = {c["ref"]: c["why"] for c in d["context_notes"]}
        self.assertEqual(why, {facts[2].ref: "stale: entry 6 sessions before this decision",
                               facts[3].ref: "stale: entry 18 sessions before this decision",
                               facts[5].ref: "horizon 63, not D1's 21", facts[6].ref: "no direction"})
        self.assertEqual(set(d["context_refs"]), set(why))
        f = decide.latest(self.bank, "X")
        self.assertEqual(set(f.inputs), {d["price"]["ref"], facts[0].ref, facts[1].ref, *why})
        self.assertEqual(src["a.example via agent"]["why"], "0 bets from 1 call, 1 pending; needs 30 bets")

    def test_a_claim_retrieved_late_anchors_late(self):
        late = claim("X", "a.example", 1, self.days[-40], retrieved=self.days[-1] + "T12:00:00Z")
        self.bank.add(late)
        d = self.go()
        self.assertEqual([i["id"] for i in d["items"] if i["kind"] == "source"], ["a.example via agent"])

    def test_source_term_largest_proven_weight_and_disagreement(self):
        def rec(w):
            return {"bets": 40, "hits": 33, "hit_rate": 0.825, "p": 1e-5, "q": 1e-4, "lower": (w + 1) / 2,
                    "mean_signed_excess": 0.01, "up": 40, "down": 0, "calls": 40, "dropped": 0, "pending": 0,
                    "unscorable": 0, "proven": True, "weight": w, "why": f"proven: weight {w:.2f}"}
        self.bank.add_many([claim("X", "a.example", 1, self.days[-2]), claim("X", "b.example", 1, self.days[-3])])
        rows = {"a.example via agent": rec(0.3), "b.example via agent": rec(0.3)}
        with mock.patch.object(decide, "track_record", return_value=rows):
            d = self.go()
        self.assertEqual((d["action"], d["score"], d["source_term"]),
                         ("NO_EDGE", 0.3, {"id": "a.example via agent", "direction": 1, "weight": 0.3}))
        self.assertNotIn(decide.NO_WEIGHT_NOTE, d["notes"])
        rows["b.example via agent"] = rec(0.6)
        with mock.patch.object(decide, "track_record", return_value=rows):
            d = self.go()
        self.assertEqual((d["action"], d["score"], d["source_term"]["id"]), ("BUY", 0.6, "b.example via agent"))
        # b's calls now net down (+1 -1 -1): the two proven sources disagree, so neither counts
        self.bank.add_many([claim("X", "b.example", -1, self.days[-2]), claim("X", "b.example", -1, self.days[-4])])
        with mock.patch.object(decide, "track_record", return_value=rows):
            d = self.go()
        self.assertEqual((d["action"], d["score"], d["source_term"]), ("NO_EDGE", 0.0, {"id": None, "direction": 0, "weight": 0.0}))
        self.assertEqual(d["lean"], {"direction": 0, "up": 1, "down": 1, "sources": 2})
        # a PASS signal and a proven source against it: 1 - 0.6 = 0.4, short of BUY; the source alone sells
        rows["a.example via agent"] = rec(0.0) | {"proven": False}
        self.ev = self.evidence(rows=[row(H1, "PASS", held=True)])
        with mock.patch.object(decide, "track_record", return_value=rows):
            d = self.go()
        self.assertEqual((d["action"], round(d["score"], 6)), ("NO_EDGE", 0.4))
        self.ev = self.evidence()
        with mock.patch.object(decide, "track_record", return_value=rows):
            self.assertEqual(self.go()["action"], "SELL")

    def test_combine_alone(self):
        s = lambda i, d, w: decide.item("source", i, d, w)  # noqa: E731
        self.assertEqual(decide.combine([s("a", 1, 0.3), s("b", 1, 0.3)])[2], 0.3)
        self.assertEqual(decide.combine([s("a", 1, 0.3), s("b", -1, 0.3)])[2], 0.0)
        self.assertEqual(decide.combine([s("a", 1, 0.3), s("b", -1, 0.0)])[2], 0.3)  # weight 0 does not disagree
        sig = decide.item("signal", "H1", 1, 1.0)
        self.assertEqual(decide.combine([sig, s("a", -1, 0.7)]), (1.0, {"id": "a", "direction": -1, "weight": 0.7}, 1.0 - 0.7))

    def test_lean(self):
        self.bank.add_many([claim("X", "a.example", 1, self.days[-2]), claim("X", "b.example", 1, self.days[-2]),
                            claim("X", "c.example", -1, self.days[-3])])
        d = self.go()
        self.assertEqual(d["lean"], {"direction": 1, "up": 2, "down": 1, "sources": 3})
        self.assertEqual(d["action"], "NO_EDGE")  # the lean is not evidence
        self.assertEqual(d["notes"][0], decide.NO_WEIGHT_NOTE)
        self.assertIn("Lean (unproven, not evidence): up, 2 up and 1 down of 3 sources", decide.card(d))

    def test_alerts_in_the_week_are_context(self):
        snap = self.bank.facts("X", "series.price_daily")[0]
        recent = watch.alert(self.bank, "X", "price_move", "notice", "X moved", [snap], ["price_move:x:1"],
                             self.days[-2] + "T23:59:59Z")
        watch.alert(self.bank, "X", "price_move", "notice", "X moved before", [snap], ["price_move:x:0"],
                    self.days[-20] + "T23:59:59Z")
        watch.alert(self.bank, "X", "headline", "info", "X: a headline", [snap], ["headline:x:1"],
                    self.days[-2] + "T23:59:59Z")  # info alerts are the headlines: never context
        d = self.go()
        self.assertEqual(d["context_refs"], [recent.ref])
        self.assertIn("alert rules are untested", d["context_notes"][0]["why"])


class DataGateTest(DecideCase):
    def setUp(self):
        super().setUp()
        self.judge = FakeJudge(ready=True)
        self.ingests = []
        p = mock.patch("brain.ingest.ingest", side_effect=lambda bank, t, log=print, **kw: self.ingests.append(t) or {})
        p.start()
        self.patches.append(p)

    def assert_no_data(self, t, why, **kw):
        d = self.go(t, do_judge=True, do_explain=True, do_read=True, **kw)
        self.assertEqual((d["ok"], d["action"], d["ref"], d["notes"][0]), (True, "NO_DATA", None, why))
        self.assertEqual((self.judge.asked, self.judge.explained), ([], []))
        self.assertEqual(self.bank.facts(t, "decision"), [])
        self.assertIn("NO_DATA", d["explanation"])
        return d

    def test_no_prices_stale_prices_stale_spy(self):
        with mock.patch("brain.scout.scout") as scout:
            self.assert_no_data("Q", "no prices for Q: ingest Q")
            old = sessions(200, end=self.days[-11])
            self.bank.add(price_snapshot("OLD", old, [10.0] * len(old)))
            self.assert_no_data("OLD", f"latest close {old[-1]}: ingest OLD")
            scout.assert_not_called()
        self.assertEqual(self.ingests, [])

    def test_stale_spy(self):
        fresh = Bank(self.home / "fresh")
        old = sessions(200, end=self.days[-11])
        fresh.add_many([price_snapshot("X", self.days, [10.0] * len(self.days)), price_snapshot("SPY", old, [10.0] * len(old))])
        self.bank = fresh
        self.assert_no_data("X", "SPY prices are stale: ingest SPY")

    def test_unscorable_symbols_are_not_ingested_read_or_judged(self):
        self.bank.add(price_snapshot("^GSPC", self.days, [10.0] * len(self.days)))
        with mock.patch("brain.scout.scout") as scout, mock.patch("brain.sources.prices.ingest") as px:
            for t in ("SPY", "^GSPC"):
                d = self.assert_no_data(t, decide.scorable(t)[1], do_ingest=True)
                self.assertIsNone(d["price"])
            scout.assert_not_called()
            px.assert_not_called()
        self.assertEqual(self.ingests, [])

    def test_a_close_ingested_after_the_session_is_used_and_check_passes(self):
        self.assertIsNone(self.bank.view(now_iso()).latest("X", "series.price_daily"))  # known only at day's end
        d = self.go()
        self.assertEqual((d["action"], d["price"]["date"]), ("NO_EDGE", today()))
        f = decide.latest(self.bank, "X")
        self.assertEqual((f.known_at, f.as_of, f.tier), (self.t_run, today(), "derived"))
        self.assertEqual(check.run(Bank(self.bank.root))[0], [])

    def test_an_ingest_failure_is_a_note(self):
        with mock.patch("brain.ingest.ingest", side_effect=SystemExit("SEC EDGAR requires a User-Agent")), \
                mock.patch("brain.sources.prices.ingest", side_effect=RuntimeError("feed down")) as px:
            state = {}
            d = self.go(do_ingest=True, llm_state=state)
            self.go("Y", do_ingest=True, llm_state=state)
        self.assertEqual(d["action"], "NO_EDGE")
        self.assertIn("ingest of X failed (SystemExit: SEC EDGAR requires a User-Agent); decided on what the bank holds",
                      d["notes"])
        self.assertIn("ingest of SPY prices failed (RuntimeError: feed down)", d["notes"])
        self.assertEqual(px.call_count, 2)  # it failed, so the next ticker tries again
        with mock.patch("brain.sources.prices.ingest", return_value=[]) as px:
            state = {}
            self.go(do_ingest=True, llm_state=state)
            self.go("Y", do_ingest=True, llm_state=state)
        self.assertEqual(px.call_count, 1)  # SPY once per run
        self.assertEqual(self.ingests, ["X", "Y"])


class RecordTest(DecideCase):
    def judged(self, bank, ticker, retrieved=None):
        snap = bank.view(self.t_run).latest(ticker, "series.price_daily")
        f = Fact(entity=ticker, attr="judgment", value="up: the quarter beat", as_of=self.t_run[:10], known_at=self.t_run,
                 tier="opinion", source={"name": "llm:fake@1234abcd"},
                 context={"direction": 1, "horizon_days": 21, "confidence": 0.7, "reasons": [], "unverified_numbers": [],
                          "model": "fake", "server_model": "fake", "judge_sha": "1234abcd", "policy": "D1"},
                 inputs=(snap.ref,), retrieved_at=retrieved or now_iso())
        bank.add(f)
        return {"ok": True, "fact": f, "ref": f.ref, "direction": 1, "error": None, "down": None}

    def test_a_judgment_written_after_t_run_is_neither_weighed_nor_cited(self):
        """A run that crosses midnight UTC: the judge stamps its call known at T_RUN but writes it the next day,
        and the scout's last claim is written then too. gather weighs only calls retrieved by T_RUN, so the
        decision cites neither and says why; both stay in the bank, scored from their own anchors."""
        after = (date.fromisoformat(self.t_run[:10]) + timedelta(days=1)).isoformat()
        self.judge = FakeJudge(ready=True, assess=lambda bank, t: self.judged(bank, t, retrieved=after + "T00:04:00Z"))
        late = claim("X", "late.example", 1, self.days[-1] + "T22:00:00Z", retrieved=after + "T00:02:00Z")

        def scout(bank, ticker, **kw):
            bank.add(late)
            return {"ok": True, "notes": [{"id": late.id[:8]}], "pages": [{"ok": True}]}
        with mock.patch("brain.scout.scout", side_effect=scout):
            d = self.go(do_read=True, do_judge=True)
        (jf,) = self.bank.facts("X", "judgment")
        self.assertEqual(d["judgment"], {"ref": None, "error": f"written after {self.t_run}"})
        self.assertIn(f"judgment {jf.ref} was written after {self.t_run} (the run crossed midnight UTC): it is recorded "
                      "and scored as its source's call, but not weighed or cited in this decision", d["notes"])
        self.assertIn(f"1 claim the scout noted after {self.t_run} (the run crossed midnight UTC) is not weighed here: "
                      "they count from the next run", d["notes"])
        self.assertEqual(([i for i in d["items"] if i["kind"] == "source"], d["lean"]["sources"]), ([], 0))
        f = decide.latest(self.bank, "X")
        self.assertNotIn(jf.ref, f.inputs)
        self.assertNotIn(late.ref, f.inputs)
        self.assertEqual(d["read"], {"claims_noted": 1, "pages_read": 1, "error": None})
        self.assertEqual(check.run(Bank(self.bank.root))[0], [])

    def test_the_recorded_decision_reloads_plain_and_passes_check(self):
        self.judge = FakeJudge(ready=True, assess=self.judged)
        self.ev = self.evidence(rows=[row(H1, "PASS", held=True, net=np.float64(0.004), p=np.float64(0.01)),
                                      row(H6, broad=True, held=np.bool_(False), net=float("nan"))])
        self.bank.add(claim("X", "a.example", -1, self.days[-2]))
        d = self.go(do_judge=True, do_explain=True)
        self.assertEqual(d["action"], "BUY")
        fresh = Bank(self.bank.root)
        (f,) = fresh.facts("X", "decision")
        self.assertEqual(f.ref, d["ref"])

        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
            else:
                self.assertTrue(x is None or type(x) in (bool, int, float, str), (type(x), x))
                if isinstance(x, float):
                    self.assertTrue(math.isfinite(x))
        walk(f.context)
        signal, term, score = decide.combine(f.context["items"])
        self.assertEqual((score, signal, term), (f.context["score"], f.context["signal_score"], f.context["source_term"]))
        self.assertEqual(decide.action_for(score), f.value)
        jref = d["judgment"]["ref"]
        self.assertIn(jref, f.inputs)
        self.assertIn("llm:fake@1234abcd", self.items(d))
        self.assertEqual(self.judge.explained, ["X"])
        self.assertEqual(check.run(Bank(self.bank.root))[0], [])
        self.assertEqual(decide.as_dict(f), d)
        for k in ("ok", "ticker", "action", "direction", "score", "threshold", "signal_score", "source_term", "items",
                  "lean", "context_refs", "price", "judgment", "read", "explanation", "explained_by", "explain_error",
                  "unverified_numbers", "ref", "policy", "policy_sha", "notes"):
            self.assertIn(k, d)
        self.assertEqual(f.context["git"].keys(), {"commit", "brain_dirty"})
        self.assertEqual(decide.tickers_with_decisions(self.bank), ["X"])

    def test_the_change_alert(self):
        watch.add("ticker", "X")
        first = self.go()
        self.assertEqual(self.bank.facts("X", "alert"), [])  # a first decision raises none
        self.ev = self.evidence(rows=[row(H1, "PASS", held=True)])
        with mock.patch.object(watch, "notify") as push:
            second = self.go()
        (a,) = self.bank.facts("X", "alert")
        self.assertEqual(a.value, "X decision NO_EDGE -> BUY (policy D1, paper)")
        self.assertEqual((a.context["rule"], a.context["severity"], a.context["keys"]),
                         ("decision", "notice", [f"decision:{second['ref']}"]))
        self.assertEqual((a.inputs, a.known_at), ((second["ref"],), self.t_run))
        push.assert_called_once()
        self.assertIn("X", json.loads(watch.state_path().read_text())["entities"])
        self.assertNotEqual(first["ref"], second["ref"])
        self.go()  # BUY again: no change, no alert
        self.assertEqual(len(self.bank.facts("X", "alert")), 1)
        self.assertEqual(check.run(Bank(self.bank.root))[0], [])
        self.ev = self.evidence()
        self.go("Y")  # not on the watchlist: no alert either way
        watch.watchlist_path().write_text("{not json")
        d = self.go()
        self.assertEqual(d["action"], "NO_EDGE")
        self.assertTrue(any(n.startswith("no change alert: ") for n in d["notes"]))
        self.assertEqual(len(self.bank.facts("X", "alert")), 1)

    def test_history_and_latest(self):
        self.assertIsNone(decide.latest(self.bank, "X"))
        a = self.go()
        self.ev = self.evidence(rows=[row(H1, "PASS", held=True)])
        b = self.go()
        self.assertEqual([f.ref for f in decide.history(self.bank, "X")], [b["ref"], a["ref"]])
        self.assertEqual(decide.latest(self.bank, "X").value, "BUY")


class ClausesTest(DecideCase):
    """Clauses of section 3 that change what is recorded, each with a test that fails when it breaks."""

    def test_sessions_past_spys_last_close_count_as_weekdays(self):
        end = pd.bdate_range(end=today(), periods=4)[0].strftime("%Y-%m-%d")  # 3 weekdays before T_RUN's date
        days = sessions(300, end=end)
        b = Bank(self.home / "lagged")  # X and SPY both end there: inside the 7-day gate on any day of the week
        b.add_many([price_snapshot(t, days, [10.0] * len(days)) for t in ("X", "SPY")])
        five, six = claim("X", "a.example", 1, days[-3]), claim("X", "b.example", 1, days[-4])
        b.add_many([five, six])
        self.bank = b
        d = self.go()
        # The decision's entry is 3 weekdays past SPY's last session (the bank does not hold them yet), so a
        # call entering 2 sessions before that last one is 5 sessions back, and one a session earlier is 6.
        self.assertEqual([i["refs"] for i in d["items"] if i["kind"] == "source"], [[five.ref]])
        self.assertEqual({c["ref"]: c["why"] for c in d["context_notes"]},
                         {six.ref: "stale: entry 6 sessions before this decision"})

    def test_the_data_gate_names_the_stock_before_spy(self):
        old = sessions(200, end=self.days[-11])
        b = Bank(self.home / "stale")
        b.add_many([price_snapshot("OLD", old, [10.0] * len(old)), price_snapshot("SPY", old, [10.0] * len(old))])
        self.bank = b
        self.assertEqual(self.go("OLD")["notes"][0], f"latest close {old[-1]}: ingest OLD")  # SPY is stale too
        self.assertEqual(self.go("Q")["notes"][0], "no prices for Q: ingest Q")

    def test_an_engine_fault_is_not_ok_records_nothing_and_exits_1(self):
        with mock.patch.object(decide, "gather", side_effect=RuntimeError("boom")):
            d = self.go()
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(["decide", "X", "--no-llm"])
        self.assertEqual((d["ok"], d["action"], d["ref"]), (False, "NO_DATA", None))
        self.assertIn("decide failed (RuntimeError: boom); nothing recorded", d["notes"])
        self.assertEqual(code, 1)
        self.assertIn("decide failed (RuntimeError: boom); nothing recorded", out.getvalue())
        self.assertEqual(Bank(self.bank.root).facts("X", "decision"), [])

    def test_the_change_alert_compares_under_the_same_policy_only(self):
        watch.add("ticker", "X")
        first = self.go()
        prev = decide.latest(self.bank, "X")
        other = Fact(entity="X", attr="decision", value="BUY", as_of=prev.as_of, known_at=prev.known_at, tier="derived",
                     source=dict(prev.source, policy="D2"), context=dict(prev.context, policy="D2", policy_sha="0" * 16),
                     inputs=prev.inputs, retrieved_at=prev.retrieved_at)
        self.bank.add(other)
        with mock.patch.object(watch, "notify"):
            second = self.go()  # NO_EDGE again under D1: the D2 BUY recorded in between is not a change
        self.assertEqual((first["action"], second["action"]), ("NO_EDGE", "NO_EDGE"))
        self.assertEqual(self.bank.facts("X", "alert"), [])


class ModelStepsTest(DecideCase):
    def test_a_timeout_stops_the_runs_later_model_steps(self):
        down = "the model server timed out: no answer within 300 s"
        self.judge = FakeJudge(ready=True, assess=lambda bank, t: {"ok": False, "fact": None, "ref": None,
                                                                    "error": down, "down": down})
        state = {}
        with mock.patch("brain.scout.scout", return_value={"ok": True, "notes": [{"id": "1"}], "pages": [{"ok": True}],
                                                          "answer": "done"}) as scout:
            d = self.go(do_read=True, do_judge=True, do_explain=True, llm_state=state)
            e = self.go("Y", do_read=True, do_judge=True, do_explain=True, llm_state=state)
        self.assertEqual(state["down"], down)
        self.assertEqual((self.judge.asked, self.judge.explained, scout.call_count), (["X"], [], 1))
        self.assertEqual(d["read"], {"claims_noted": 1, "pages_read": 1, "error": None})
        self.assertIn("Read: 1 page read, 1 claim noted", decide.card(d))
        self.assertEqual(d["judgment"], {"ref": None, "error": down})
        self.assertEqual(e["judgment"], {"ref": None, "error": f"the model failed earlier in this run ({down})"})
        self.assertEqual((d["explained_by"], e["explained_by"]), ("template", "template"))
        self.assertTrue(any("explanation by template: the model failed earlier in this run" in n for n in d["notes"]))
        for step in ("read", "judgment"):
            self.assertTrue(any(n.startswith(f"{step} skipped: the model failed earlier in this run") for n in e["notes"]))
        self.assertEqual((d["action"], e["action"]), ("NO_EDGE", "NO_EDGE"))

    def test_a_scout_timeout_without_a_down_field_counts(self):
        self.judge = FakeJudge(ready=True)
        state = {}
        with mock.patch("brain.scout.scout", return_value={"ok": False, "error": "the model server timed out",
                                                          "notes": [], "pages": []}):
            d = self.go(do_read=True, do_judge=True, llm_state=state)
        self.assertEqual(state["down"], "the model server timed out")
        self.assertEqual(self.judge.asked, [])
        self.assertEqual(d["read"]["error"], "the model server timed out")

    def test_no_model_reachable(self):
        self.judge = FakeJudge(ready=False)
        with mock.patch("brain.scout.scout") as scout:
            d = self.go(do_read=True, do_judge=True, do_explain=True)
        scout.assert_not_called()
        self.assertEqual(self.judge.asked, [])
        self.assertTrue(d["judgment"]["error"].startswith("no model reachable (nothing answered at"))
        self.assertIn("read skipped: no model reachable (nothing answered at http://127.0.0.1:11434/v1)", d["notes"])
        self.assertIn("judgment skipped: no model reachable (as above)", d["notes"])
        self.assertEqual((d["explained_by"], d["explanation"]), ("template", "X: NO_EDGE (template)"))
        self.assertIsNotNone(d["ref"])

    def test_without_brain_judge_the_engine_still_decides(self):
        with mock.patch.object(decide, "_judge_module", side_effect=ImportError("No module named 'brain.judge'")):
            d = self.go(do_judge=True, do_explain=True)
        self.assertEqual(d["action"], "NO_EDGE")
        self.assertTrue(d["judgment"]["error"].startswith("brain.judge unavailable (ImportError"))
        self.assertIn("No evidence carries weight yet", d["explanation"])

    def test_a_rejected_explanation_says_why_on_the_card(self):
        why = "the explanation does not state the action NO_EDGE"
        self.judge = FakeJudge(ready=True, explain=lambda dec: {"text": "X: NO_EDGE (template)", "by": "template",
                                                                 "unverified_numbers": [], "error": why, "down": None})
        d = self.go(do_explain=True)
        self.assertEqual((d["explained_by"], d["explain_error"]), ("template", why))
        self.assertIn(f"(written by the template: the model's explanation was not used, {why})", decide.card(d))

    def test_a_model_explanation_is_kept_and_cannot_change_the_action(self):
        self.judge = FakeJudge(ready=True, explain=lambda dec: {"text": "Do nothing: no evidence has a record.",
                                                                 "by": "llm:fake", "unverified_numbers": [], "error": None})
        d = self.go(do_explain=True)
        self.assertEqual((d["action"], d["explained_by"], d["explanation"]),
                         ("NO_EDGE", "llm:fake", "Do nothing: no evidence has a record."))


def decision(t, action, day, lean=0, sha=None, commit="c0ffee"):
    """A decision fact as decide records it, made on ``day`` (scoring reads only these fields)."""
    return Fact(entity=t, attr="decision", value=action, as_of=day, known_at=day + "T23:59:59Z", tier="derived",
                source={"name": "brain.decide", "policy": "D1"},
                context={"policy": "D1", "policy_sha": sha or decide.POLICY_SHA, "horizon_days": 21,
                         "direction": decide.DIRECTION[action], "lean": {"direction": lean, "up": 0, "down": 0, "sources": 0},
                         "git": {"commit": commit, "brain_dirty": False}},
                inputs=(f"{t}/0",), retrieved_at=day + "T22:00:00Z")


class ScoreTest(TempHome):
    def setUp(self):
        super().setUp()
        self.days = pd.bdate_range("2024-01-01", periods=200).strftime("%Y-%m-%d").tolist()
        n = len(self.days)
        for i, t in enumerate("ABCDEFGH"):
            self.bank.add(price_snapshot(t, self.days, [100 * 1.002 ** k for k in range(n)]))  # beats a flat SPY
        self.bank.add(price_snapshot("SPY", self.days, [100.0] * n))
        self.t = self.days[-1] + "T23:59:59Z"

    def test_three_streams_one_run_one_bet_sha_mismatch_and_no_edge_apart(self):
        d = self.days
        facts = [decision(t, "BUY", d[10], lean=1) for t in "ABC"]          # one run, three tickers: one bet
        facts += [decision("D", "BUY", d[15], lean=1)]                      # inside that bet's window
        facts += [decision(t, "NO_EDGE", d[5]) for t in "EFG"]              # NO_EDGE before and around the BUYs
        facts += [decision("H", "SELL", d[40], lean=-1)]                    # a later, separate bet: a miss
        facts += [decision("A", "BUY", d[60], sha="deadbeefdeadbeef")]      # made under an edited D1
        facts += [decision("B", "BUY", d[-5])]                              # its window has not closed
        self.bank.add_many(facts)
        s = decide.score(self.bank, self.t)
        p = s["policies"]["D1"]
        self.assertEqual((p["actions"]["bets"], p["actions"]["hits"], p["actions"]["dropped"]), (2, 1, 1))
        self.assertEqual((p["lean"]["bets"], p["lean"]["hits"], p["lean"]["dropped"]), (2, 1, 1))
        self.assertEqual(p["no_edge"]["bets"], 1)
        self.assertAlmostEqual(p["no_edge"]["mean_excess"], 1.002 ** 21 - 1)
        self.assertEqual((p["pending"], p["commits"], p["sha"]), (1, ["c0ffee"], decide.POLICY_SHA))
        self.assertFalse(p["actions"]["proven"])
        self.assertIn("D1?deadbeefdeadbeef", s["policies"])
        self.assertEqual(s["policies"]["D1?deadbeefdeadbeef"]["actions"]["bets"], 1)
        rows = {(r["ticker"], r["as_of"]): r for r in s["rows"]}
        self.assertEqual(rows[("A", d[10])]["stream_status"], "scored")
        self.assertEqual(rows[("D", d[15])]["stream_status"], "overlapping")
        self.assertEqual(rows[("B", d[-5])]["stream_status"], "pending")
        self.assertEqual(rows[("H", d[40])]["hit"], False)
        self.assertIsNone(rows[("E", d[5])]["hit"])  # NO_EDGE has no hit, only what the stock did
        self.assertEqual(rows[("E", d[5])]["stream_status"], "scored")
        self.assertEqual(s["rows"][0]["as_of"], d[-5])  # newest first
        self.assertEqual(s["gate"], decide.POLICY["source_gate"])
        json.dumps(s, allow_nan=False)
        text = decide.scorecard(s, {})
        self.assertIn("D1?deadbeefdeadbeef", text)
        self.assertIn("not a hit rate", text)

    def test_a_decision_with_no_price_past_its_exit_is_unpriced(self):
        d = self.days
        # NOPX's prices were never ingested; LAG's stop before its exit, which SPY passed 2 sessions ago.
        self.bank.add(price_snapshot("LAG", d[:-9], [100.0] * (len(d) - 9)))
        self.bank.add_many([decision("A", "BUY", d[10], lean=1), decision("NOPX", "BUY", d[40], lean=1),
                            decision("LAG", "SELL", d[-24])])
        s = decide.score(self.bank, self.t)
        p = s["policies"]["D1"]
        rows = {r["ticker"]: r for r in s["rows"]}
        self.assertEqual({t: r["stream_status"] for t, r in rows.items()},
                         {"A": "scored", "NOPX": "unpriced", "LAG": "pending"})
        self.assertEqual((p["pending"], p["unpriced"], p["actions"]["unpriced"], p["lean"]["unpriced"]), (1, 1, 1, 1))
        self.assertEqual(p["actions"]["bets"], 1)
        text = decide.scorecard(s, {})
        self.assertIn("3 decisions, 1 pending, 1 with no price past their exit", text)
        self.assertIn("1 with no price past its exit (ingest their tickers)", text)
        # Once NOPX's prices arrive it is scored on what it did: nothing is frozen out.
        self.bank.add(price_snapshot("NOPX", d, [100 * 1.002 ** k for k in range(len(d))]))
        p = decide.score(self.bank, self.t)["policies"]["D1"]
        self.assertEqual((p["unpriced"], p["actions"]["bets"]), (0, 2))

    def test_too_many_unpriced_decisions_bar_a_proven_stream(self):
        gate = decide.POLICY["source_gate"]
        won = [{"entry": f"2024-{m:02d}-01", "exit": f"2024-{m:02d}-28", "refs": [str(m)], "value": 0.01, "hit": True}
               for m in range(1, 13)] * 3  # 36 hits
        self.assertTrue(decide._stream(won, 0, gate, priced=36, unpriced=4)["proven"])    # 4 of 40: within 10%
        self.assertFalse(decide._stream(won, 0, gate, priced=36, unpriced=5)["proven"])   # 5 of 41: over it
        self.assertEqual(decide._stream(won, 0, gate, priced=36, unpriced=5)["unpriced"], 5)

    def test_point_in_time(self):
        short = self.days[:21]  # the prices as they stood on days[20]
        self.bank.add_many([price_snapshot("A", short, [100 * 1.002 ** k for k in range(21)]),
                            price_snapshot("SPY", short, [100.0] * 21), decision("A", "BUY", self.days[10])])
        early = self.days[20] + "T23:59:59Z"
        self.assertEqual(decide.score(self.bank, early)["policies"]["D1"]["pending"], 1)  # exit after t
        self.assertEqual(decide.score(self.bank, self.days[5] + "T23:59:59Z")["rows"], [])  # made after t
        self.assertEqual(decide.score(self.bank, self.t)["policies"]["D1"]["actions"]["bets"], 1)


class CliTest(DecideCase):
    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["decide", *argv])
        return code, out.getvalue(), err.getvalue()

    def test_decide_no_llm_prints_the_card_and_records_a_checked_decision(self):
        self.bank.add(claim("X", "reuters.com", 1, self.days[-2], noted_by="scout:qwen3"))
        code, out, _ = self.cli("X", "--no-llm")
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(lines[0], "X  NO_EDGE: do nothing   (policy D1 · paper only · not financial advice)")
        self.assertEqual(lines[1], "Score 0.00; BUY needs +0.50, SELL -0.50. No evidence carries weight yet.")
        self.assertIn("Evidence                              dir   weight  why", out)
        self.assertIn("signal  H1-mom12_1                    none  0       FAIL on TEST (net excess -0.11%/mo, p 0.88)", out)
        self.assertIn("source  reuters.com via scout:qwen3   up    0       0 bets from 1 call, 1 pending; needs 30 bets", out)
        self.assertIn("Lean (unproven, not evidence): up, 1 of 1 sources", out)
        f = decide.latest(Bank(self.bank.root), "X")
        self.assertIn(f"Recorded X/{f.id[:8]}. Scored after 21 trading days: `brain decide score`.", out)
        self.assertEqual(self.judge.asked, [])
        self.assertEqual(check.run(Bank(self.bank.root))[0], [])

        code, out, _ = self.cli("show", "X")
        self.assertEqual(code, 0)
        self.assertIn(f"Recorded X/{f.id[:8]}.", out)
        code, out, _ = self.cli("show", "X", "--all", "--json")
        self.assertEqual((code, [r["ref"] for r in json.loads(out)]), (0, [f.ref]))
        code, out, _ = self.cli("score")
        self.assertEqual(code, 0)
        self.assertIn("D1 (sha bef26a9d157cbcf5", out)
        self.assertIn("reuters.com via scout:qwen3", out)
        code, out, _ = self.cli("score", "--json")
        self.assertEqual(json.loads(out)["score"]["policies"]["D1"]["pending"], 1)
        code, out, _ = self.cli("policy")
        self.assertEqual(code, 0)
        self.assertIn("POLICY_SHA bef26a9d157cbcf5", out)
        self.assertEqual(json.loads(self.cli("policy", "--json")[1])["policy"], decide.POLICY)

    def test_json_watchlist_and_usage_errors(self):
        watch.add("ticker", "Y")
        code, out, _ = self.cli("X", "SPY", "--watchlist", "--no-llm", "--json")
        got = json.loads(out)
        self.assertEqual((code, [(d["ticker"], d["action"]) for d in got]),
                         (0, [("X", "NO_EDGE"), ("SPY", "NO_DATA"), ("Y", "NO_EDGE")]))
        for argv in ((), ("show",), ("show", "X", "Y"), ("score", "X"), ("X", "--all"), ("bad ticker",),
                     ("policy", "--ingest")):
            with self.subTest(argv=argv):
                self.assertEqual(self.cli(*argv)[0], 1)
        self.assertEqual(self.cli("show", "Q")[0], 1)  # nothing recorded for Q


class CheckRuleTest(TempHome):
    """brain check: a model's own call (an opinion from llm:) needs no URL but follows the derived-input rules."""

    def setUp(self):
        super().setUp()
        self.days = sessions(30, end="2026-03-31")
        self.snap = price_snapshot("X", self.days, [10.0] * 30)
        self.bank.add(self.snap)

    def opinion(self, name, inputs, known="2026-04-01T23:59:59Z", url=None):
        src = {"name": name, **({"url": url} if url else {})}
        return Fact(entity="X", attr="judgment", value="up", as_of=known[:10], known_at=known, tier="opinion",
                    source=src, context={"direction": 1, "horizon_days": 21}, inputs=tuple(inputs), retrieved_at=known)

    def errors(self):
        return check.run(Bank(self.bank.root))[0]

    def test_llm_opinions(self):
        ok = self.opinion("llm:qwen3:8b@3f2a91c0", [self.snap.ref])
        self.bank.add(ok)
        self.assertEqual(self.errors(), [])
        bare = self.opinion("llm:qwen3:8b@3f2a91c0", [], known="2026-04-02T23:59:59Z")
        early = self.opinion("llm:qwen3:8b@3f2a91c0", [self.snap.ref], known="2026-03-30T12:00:00Z")
        ghost = self.opinion("llm:qwen3:8b@3f2a91c0", ["X/0123456789abcdef0123"], known="2026-04-03T23:59:59Z")
        elsewhere = self.opinion("llm:qwen3:8b@3f2a91c0", ["MACRO/0123456789abcdef0123"], known="2026-04-04T23:59:59Z")
        human = self.opinion("someone", [], known="2026-04-05T23:59:59Z")
        self.bank.add_many([bare, early, ghost, elsewhere, human])
        errs = self.errors()
        self.assertEqual(len(errs), 5, errs)
        self.assertTrue(any(bare.id in e and "model opinion cites no inputs" in e for e in errs))
        self.assertTrue(any(early.id in e and "known before its input" in e for e in errs))
        self.assertTrue(any(ghost.id in e and "not in bank" in e for e in errs))
        self.assertTrue(any(elsewhere.id in e and "not in bank" in e for e in errs))
        self.assertTrue(any(human.id in e and "without a source url" in e for e in errs))


class RealEvidenceTest(TempHome):
    """brief.evidence's coverage fields, and the signal rules on a ranking it really computed."""

    def setUp(self):
        super().setUp()
        from tests.test_brief import _universe_bank
        self.home = Path(self._tmp.name)
        self.large = [f"L{i}" for i in range(5)]
        self.broad = [*self.large, *(f"B{i}" for i in range(5, 11))]
        _universe_bank(self.bank, self.large, self.broad)  # prices to 2021-06-30
        (self.home / "large.json").write_text(json.dumps({"tickers": self.large}))
        (self.home / "broad.json").write_text(json.dumps({"tickers": self.broad}))
        ctl = {"pass": True, "oracle_test_perm_p": 0.0005, "random_false_positive_rate": 0.04, "n_random": 200}
        recs = [{"op": "result", "id": h, "spec_sha": "s", "at": "2026-01-01T00:00:00Z", "controls": ctl,
                 "test": {"mean_net_excess": net, "perm_p": p}, "train": {}, "context": {}}
                for h, net, p in (("H1-mom12_1", 0.0011, 0.18), ("H6-insider_buying", 0.01, 0.001))]
        recs += [{"op": "withdraw", "id": h, "reason": "a code mistake", "at": "2026-01-02T00:00:00Z"} for h in WITHDRAWN]
        (self.home / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        self.patches = [mock.patch.object(config, "UNIVERSE", self.home / "large.json"),
                        mock.patch.object(config, "LEDGER", self.home / "ledger.jsonl"),
                        mock.patch.object(universe, "BROAD", self.home / "broad.json")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        super().tearDown()

    def test_coverage_fields_and_rules_on_a_real_ranking(self):
        ev = brief.evidence(self.bank, "B7", budget_s=0)  # only the asked-for stock fits in the budget
        self.assertEqual(ev["coverage"], {"ranked": 5, "universe_size": 5})
        self.assertEqual((ev["broad"]["status"], ev["broad"]["coverage"]), ("pending", {"ranked": 1, "universe_size": 11}))
        ev = brief.evidence(self.bank, "B7", budget_s=None)
        self.assertEqual(ev["broad"]["coverage"], {"ranked": 11, "universe_size": 11})
        json.dumps(ev, allow_nan=False)
        # The prices end on 2021-06-30, a month-end with no later session, so the ranking is from 2021-05-31.
        t = "2021-07-02T23:59:59Z"
        book = PriceBook(self.bank, ["B7", "SPY"], asof=t)
        self.assertEqual((ev["rebalance_date"], decide.last_month_end(book, t)), ("2021-05-31", "2021-06-30"))
        it = {i["id"]: i for i in decide.signal_items(self.bank, "B7", t, book)}
        self.assertEqual(it["H6-insider_buying"]["record"]["verdict"], "PASS")
        self.assertEqual(it["H6-insider_buying"]["why"], "ranking as of 2021-05-31: ingest SPY and the universe")
        self.assertEqual(it["H1-mom12_1"]["why"], "FAIL on TEST (net excess +0.11%/mo, p 0.18)")
        self.assertEqual(it["H1-mom12_1"]["record"]["coverage"], {"ranked": 5, "universe_size": 5})
        # Seen on June 30 itself, before July's first close, May's ranking is the current one, and the
        # remaining rules decide.
        t = "2021-06-30T23:59:59Z"
        book = PriceBook(self.bank, ["SPY"], asof=t)
        it = {i["id"]: i for i in decide.signal_items(self.bank, "B7", t, book)}
        row = {r["hypothesis"]: r for r in ev["signals"]}["H6-insider_buying"]
        h6 = it["H6-insider_buying"]
        self.assertEqual(h6["record"], {"verdict": "PASS", "universe": BROAD, "rebalance_date": "2021-05-31",
                                        "coverage": {"ranked": 11, "universe_size": 11}, "held": row["held"]})
        if row["held"]:
            self.assertEqual((h6["weight"], h6["direction"]), (1.0, 1))
        else:
            self.assertEqual((h6["weight"], h6["why"]), (0.0, "PASS, but its backtest would not hold this stock"))
        # A stock outside the universe is ranked (evidence adds it), but the universe file does not list it.
        spy = self.bank.facts("SPY", "series.price_daily")[0]
        self.bank.add(Fact(**{**{k: getattr(spy, k) for k in ("attr", "value", "as_of", "known_at", "tier", "source",
                                                               "context")}, "entity": "OUT"}))
        it = {i["id"]: i for i in decide.signal_items(self.bank, "OUT", t, book)}
        self.assertEqual((it["H6-insider_buying"]["weight"], it["H6-insider_buying"]["why"]),
                         (0.0, "PASS on universe/us_broad.json, which does not list this stock"))


class MonthOpeningTest(RealEvidenceTest):
    """The first session of a month, after the close and before midnight UTC (docs/DECIDE.md 3.2, rule (c))."""

    def test_a_month_opening_close_ingested_this_evening_marks_the_month_end(self):
        ctl = {"pass": True, "oracle_test_perm_p": 0.0005, "random_false_positive_rate": 0.04, "n_random": 200}
        recs = [{"op": "result", "id": h, "spec_sha": "s", "at": "2026-01-01T00:00:00Z", "controls": ctl,
                 "test": {"mean_net_excess": 0.01, "perm_p": 0.001}, "train": {}, "context": {}}
                for h in ("H1-mom12_1", "H6-insider_buying")]
        recs += [{"op": "withdraw", "id": h, "reason": "a code mistake", "at": "2026-01-02T00:00:00Z"} for h in WITHDRAWN]
        (self.home / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        for t in [*self.broad, "SPY"]:  # this evening's ingest: July 1st's close, known at the end of its day
            df = series.load(self.bank.facts(t, "series.price_daily")[-1].value)
            self.bank.add(price_snapshot(t, [*df["date"], "2021-07-01"], [*df["close"], float(df["close"].iloc[-1])]))
        evening, t = "2021-07-01T21:00:00Z", "2021-07-01T23:59:59Z"
        with mock.patch("brain.bank.now_iso", return_value=evening):
            book = PriceBook(self.bank, ["SPY"], asof=t)
            self.assertEqual(decide.last_month_end(book, t), "2021-06-30")
            # read at "now", July 1st's close is not visible yet, and June's month-end is not marked
            self.assertEqual(brief.evidence(self.bank, "L0", budget_s=None, asof=evening)["rebalance_date"], "2021-05-31")
            self.assertEqual(brief.evidence(self.bank, "L0", budget_s=None, asof=evening)["broad"]["rebalance_date"],
                             "2021-05-31")
            evs = {s: brief.evidence(self.bank, s, budget_s=None, asof=t) for s in self.large}
            self.assertEqual({ev["rebalance_date"] for ev in evs.values()}, {"2021-06-30"})
            # the broad rows (H6 here) rank from the same month-end as the H1-H4 rows
            self.assertEqual({ev["broad"]["rebalance_date"] for ev in evs.values()}, {"2021-06-30"})
            held = [s for s, ev in evs.items() if {r["hypothesis"]: r for r in ev["signals"]}[H1]["in_top_third"]]
            self.assertTrue(held)
            it = {i["id"]: i for i in decide.signal_items(self.bank, held[0], t, book)}
        h1 = it[H1]
        self.assertEqual((h1["weight"], h1["direction"], h1["record"]["rebalance_date"]), (1.0, 1, "2021-06-30"))
        self.assertIn("PASS on TEST", h1["why"])
        h6 = it["H6-insider_buying"]
        self.assertEqual(h6["record"]["rebalance_date"], "2021-06-30")
        self.assertNotIn("ranking as of", h6["why"])


class FullLoopTest(DecideCase):
    """decide with the real brain.judge, against a scripted local OpenAI-compatible server."""

    def test_judge_and_explain_through_the_real_judge_module(self):
        from brain import judge
        from tests.test_analyst import calls_msg, text_msg, tool_call
        from tests.test_judge import call_json, price_cite
        from tests.test_llm import FakeOpenAI
        self.patches.pop().stop()  # the real brain.judge this time, not the stand-in

        def reply(n, body):
            if body["messages"][0]["content"] == judge.EXPLAIN_SYSTEM:
                return text_msg("X: NO_EDGE, so do nothing. No evidence carries weight yet.")
            if not any(m["role"] == "tool" for m in body["messages"]):
                return calls_msg(tool_call(1, "describe", {"ticker": "X"}))
            return text_msg(call_json("up", [price_cite(body)]))
        with FakeOpenAI(reply) as srv, mock.patch.object(judge, "llm_ready", return_value=(True, "")):
            d = decide.decide(self.bank, "X", client=srv.client(), log=lambda m: None)
        self.assertEqual(d["action"], "NO_EDGE")
        jf = self.bank.get(d["judgment"]["ref"])
        self.assertEqual((jf.attr, jf.known_at, jf.context["direction"]), ("judgment", self.t_run, 1))
        src = [i for i in d["items"] if i["kind"] == "source"]
        self.assertEqual([(i["id"], i["direction"], i["weight"], i["refs"]) for i in src],
                         [(jf.source["name"], 1, 0.0, [jf.ref])])
        self.assertTrue(jf.source["name"].startswith("llm:fake-model@"))
        self.assertEqual(d["lean"], {"direction": 1, "up": 1, "down": 0, "sources": 1})
        self.assertEqual((d["explained_by"], d["explanation"]),
                         ("llm:fake-model", "X: NO_EDGE, so do nothing. No evidence carries weight yet."))
        self.assertIn(jf.ref, decide.latest(self.bank, "X").inputs)
        self.assertEqual(check.run(Bank(self.bank.root))[0], [])
