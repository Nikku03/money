"""The decision engine (docs/DECIDE.md section 3): policy D1, signal rules, sources, the data gate, recording,
change alerts, the model steps' availability, scoring decisions and the CLI. The model steps are stood in for
by a fake brain.judge, so nothing here needs a model server."""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from brain import brief, check, cli, config, decide, universe, watch
from brain.bank import Bank, Fact
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
                        mock.patch.object(brief, "evidence", side_effect=lambda bank, t, budget_s=None: self.ev),
                        mock.patch.object(decide, "_judge_module", side_effect=lambda: self.judge)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        super().tearDown()

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
        self.assertEqual(decide.POLICY_SHA, "67f1a0f75992fd06")  # D1 is never edited: a change is D2
        self.assertEqual(decide.POLICY_SHA,
                         hashlib.sha256(json.dumps(decide.POLICY, sort_keys=True).encode()).hexdigest()[:16])
        self.assertEqual((decide.POLICY["id"], decide.POLICY["threshold"], decide.POLICY["horizon_days"]), ("D1", 0.5, 21))

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
    def judged(self, bank, ticker):
        snap = bank.view(self.t_run).latest(ticker, "series.price_daily")
        f = Fact(entity=ticker, attr="judgment", value="up: the quarter beat", as_of=self.t_run[:10], known_at=self.t_run,
                 tier="opinion", source={"name": "llm:fake@1234abcd"},
                 context={"direction": 1, "horizon_days": 21, "confidence": 0.7, "reasons": [], "unverified_numbers": [],
                          "model": "fake", "server_model": "fake", "judge_sha": "1234abcd", "policy": "D1"},
                 inputs=(snap.ref,), retrieved_at=now_iso())
        bank.add(f)
        return {"ok": True, "fact": f, "ref": f.ref, "direction": 1, "error": None, "down": None}

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
        self.assertEqual(d["judgment"], {"ref": None, "error": down})
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
        self.assertTrue(d["judgment"]["error"].startswith("judgment skipped: no model reachable"))
        self.assertEqual((d["explained_by"], d["explanation"]), ("template", "X: NO_EDGE (template)"))
        self.assertIsNotNone(d["ref"])

    def test_without_brain_judge_the_engine_still_decides(self):
        with mock.patch.object(decide, "_judge_module", side_effect=ImportError("No module named 'brain.judge'")):
            d = self.go(do_judge=True, do_explain=True)
        self.assertEqual(d["action"], "NO_EDGE")
        self.assertTrue(d["judgment"]["error"].startswith("judgment skipped: brain.judge unavailable"))
        self.assertIn("No evidence carries weight yet", d["explanation"])

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
        self.assertIn("D1 (sha 67f1a0f75992fd06", out)
        self.assertIn("reuters.com via scout:qwen3", out)
        code, out, _ = self.cli("score", "--json")
        self.assertEqual(json.loads(out)["score"]["policies"]["D1"]["pending"], 1)
        code, out, _ = self.cli("policy")
        self.assertEqual(code, 0)
        self.assertIn("POLICY_SHA 67f1a0f75992fd06", out)
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
