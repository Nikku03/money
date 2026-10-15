import contextlib
import io
from datetime import datetime, timedelta, timezone

import pandas as pd

from brain import claims, cli
from brain.bank import Bank, Fact
from brain.metrics import PriceBook
from brain.timeutil import today
from tests.util import TempHome, price_snapshot


class ClaimsTest(TempHome):
    def test_claims_are_scored_against_the_market_after_publication(self):
        days = pd.bdate_range("2024-01-01", periods=60).strftime("%Y-%m-%d").tolist()
        up = [100 * 1.01 ** i for i in range(60)]     # beats a flat market
        flat = [100.0] * 60
        self.bank.add(price_snapshot("X", days, up))
        self.bank.add(price_snapshot("SPY", days, flat))
        claims.note(self.bank, "X", "X will rally", "https://good.example/a", direction="up", known_at=days[5])
        claims.note(self.bank, "X", "X will fall", "https://bad.example/b", direction="down", known_at=days[5])
        claims.note(self.bank, "X", "too recent", "https://bad.example/c", direction="down", known_at=days[50])
        claims.note(self.bank, "X", "no view", "https://good.example/d", known_at=days[5])
        s = claims.score(self.bank)
        self.assertEqual(s["by_source"]["good.example"], {"n": 1, "hits": 1, "hit_rate": 1.0, "p_vs_coin": 1.0})
        self.assertEqual(s["by_source"]["bad.example"]["hits"], 0)
        self.assertEqual(len(s["pending"]), 1)

    def test_future_claims_are_refused(self):
        with self.assertRaises(ValueError):
            claims.note(self.bank, "X", "t", "https://x.example", known_at="2999-01-01T00:00:00Z")

    def test_a_publication_time_is_compared_in_utc_whatever_its_offset(self):
        now = datetime.now(timezone.utc)
        ahead = (now + timedelta(hours=3)).astimezone(timezone(timedelta(hours=-5))).isoformat(timespec="seconds")
        with self.assertRaises(ValueError):  # 3 hours ahead, though its wall-clock text sorts below now's
            claims.note(self.bank, "X", "t", "https://x.example/a", known_at=ahead)
        before = (now - timedelta(minutes=30)).astimezone(timezone(timedelta(hours=2))).isoformat(timespec="seconds")
        f = claims.note(self.bank, "X", "t", "https://x.example/b", known_at=before)  # 30 minutes ago, east of UTC
        self.assertEqual(f.known_at, (now - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.assertLessEqual(f.known_at, f.retrieved_at)
        self.assertEqual(f.as_of, f.known_at[:10])  # the UTC date, not the page's local one
        g = claims.note(self.bank, "X", "t", "https://x.example/c", known_at=today())  # a page that gives only its date
        self.assertEqual(g.known_at, today() + "T23:59:59Z")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        with self.assertRaises(ValueError):
            claims.note(self.bank, "X", "t", "https://x.example/d", known_at=tomorrow)
        self.assertEqual(len(self.bank.facts("X", "claim")), 2)

    def test_note_takes_only_http_urls_with_a_host(self):
        for url in ("llm:qwen3:8b@3f2a91c0", "ftp://x.example/a", "file:///etc/passwd", "https://", "x.example/a",
                    "javascript:alert(1)"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    claims.note(self.bank, "X", "t", url)
        f = claims.note(self.bank, "X", "t", "http://Plain.Example/a")
        self.assertEqual(f.source, {"name": "plain.example", "url": "http://Plain.Example/a"})
        self.assertEqual(self.bank.facts("X", "claim"), [f])

    def test_brain_note_says_why_it_refuses_a_source(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cli.main(["note", "Y", "Reuters says buy", "--source", "Reuters", "--direction", "up"]), 1)
        self.assertIn("http(s) URL", err.getvalue())
        self.assertEqual(Bank().facts("Y", "claim"), [])


def call(t, host, direction, known, retrieved=None, horizon=21, noted_by=None, attr="claim"):
    """A claim (or judgment) with its retrieval time set; claims.note stamps the real now."""
    known = known if "T" in known else known + "T15:00:00Z"
    ctx = {"direction": direction, "horizon_days": horizon, **({"noted_by": noted_by} if noted_by else {})}
    src = {"name": host} if attr == "judgment" else {"name": host, "url": f"https://{host}/{t}/{known}"}
    return Fact(entity=t, attr=attr, value=f"{host}: {direction}", as_of=known[:10], known_at=known,
                tier="opinion", source=src, context=ctx, retrieved_at=retrieved or known)


class TrackRecordTest(TempHome):
    """claims.outcome, bets and track_record: the record decision policy D1 weighs (docs/DECIDE.md 3.3)."""

    def setUp(self):
        super().setUp()
        self.days = pd.bdate_range("2024-01-01", periods=300).strftime("%Y-%m-%d").tolist()
        self.n = len(self.days)
        self.bank.add(price_snapshot("SPY", self.days, [100.0] * self.n))
        self.t = self.days[-1] + "T23:59:59Z"

    def rising(self, t, rate=0.002):
        self.bank.add(price_snapshot(t, self.days, [100 * (1 + rate) ** k for k in range(self.n)]))

    def test_outcome_from_the_first_session_after_the_anchor(self):
        self.rising("X")
        book = PriceBook(self.bank, ["X", "SPY"], asof=self.t)
        o = claims.outcome(book, "X", self.days[5] + "T15:00:00Z", 21)
        self.assertEqual((o["entry"], o["exit"]), (self.days[6], self.days[27]))
        self.assertAlmostEqual(o["excess"], 1.002 ** 21 - 1)
        self.assertIsNone(claims.outcome(book, "X", self.days[-10], 21))  # the exit has not happened
        self.assertIsNone(claims.outcome(book, "NONE", self.days[5], 21))
        gap = [d for i, d in enumerate(self.days) if not 6 <= i <= 14]  # no close for two weeks around the entry
        self.bank.add(price_snapshot("G", gap, [100.0] * len(gap)))
        book = PriceBook(self.bank, ["G", "SPY"], asof=self.t)
        self.assertIsNone(claims.outcome(book, "G", self.days[10], 21))  # its last close is over 4 days old
        self.assertIsNotNone(claims.outcome(book, "G", self.days[8], 21))  # 4 days old is still its close
        self.assertIsNotNone(claims.outcome(book, "G", self.days[30], 21))

    def test_the_anchor_a_claim_retrieved_after_its_window_is_not_scored_on_it(self):
        up_then_down = [100 * 1.01 ** k if k <= 30 else 100 * 1.01 ** 30 * 0.99 ** (k - 30) for k in range(self.n)]
        self.bank.add(price_snapshot("X", self.days, up_then_down))
        late = call("X", "late.example", 1, self.days[5], retrieved=self.days[40] + "T10:00:00Z")
        self.bank.add(late)
        self.assertEqual(claims.anchor(late), self.days[40] + "T10:00:00Z")
        r = claims.track_record(self.bank, self.t)["late.example via agent"]
        self.assertEqual((r["bets"], r["hits"]), (1, 0))  # scored on the fall after day 40, not the rise after day 5
        self.assertLess(r["mean_signed_excess"], 0)

    def test_thirty_tickers_on_one_day_are_one_bet_and_a_call_inside_it_is_dropped(self):
        for i in range(30):
            self.rising(f"T{i}")
        self.bank.add_many([call(f"T{i}", "herd.example", 1, self.days[5], noted_by="scout:m") for i in range(30)])
        r = claims.track_record(self.bank, self.t)["herd.example via scout:m"]
        self.assertEqual((r["bets"], r["hits"], r["calls"], r["dropped"], r["proven"]), (1, 1, 30, 0, False))
        self.assertEqual(r["why"], "1 bet (1 hit) from 30 calls; needs 30 bets")
        self.bank.add_many([call("T0", "herd.example", 1, self.days[12], noted_by="scout:m"),   # inside that bet's window
                            call("T1", "herd.example", 1, self.days[40], noted_by="scout:m")])  # after it: a second bet
        r = claims.track_record(self.bank, self.t)["herd.example via scout:m"]
        self.assertEqual((r["bets"], r["hits"], r["calls"], r["dropped"], r["pending"], r["up"], r["down"]),
                         (2, 2, 32, 1, 0, 32, 0))
        self.assertFalse(r["proven"])
        self.assertEqual(r["weight"], 0.0)
        self.assertEqual(r["why"], "2 bets (2 hits) from 32 calls; needs 30 bets")
        self.assertEqual(set(r), {"bets", "hits", "hit_rate", "p", "q", "lower", "mean_signed_excess", "up", "down",
                                  "calls", "dropped", "pending", "unpriced", "unscorable", "proven", "weight", "why"})

    def test_bets_and_an_exact_zero_bet(self):
        c = lambda ref, d, entry, exit_, x: {"ref": ref, "ticker": "X", "direction": d, "entry": entry, "exit": exit_,  # noqa: E731
                                             "excess": x}
        bs, dropped = claims.bets([c("b", 1, "2024-01-02", "2024-02-01", 0.02), c("a", -1, "2024-01-02", "2024-02-01", 0.02),
                                   c("c", 1, "2024-01-10", "2024-02-09", 0.5), c("d", 1, "2024-02-02", "2024-03-01", -0.01)])
        self.assertEqual(dropped, 1)
        self.assertEqual([(b["entry"], b["refs"], b["value"], b["hit"]) for b in bs],
                         [("2024-01-02", ["a", "b"], 0.0, None), ("2024-02-02", ["d"], -0.01, False)])
        t = claims.tally(bs)
        self.assertEqual((t["bets"], t["hits"], t["hit_rate"]), (1, 0, 0.0))  # the zero bet is unscored
        self.assertAlmostEqual(t["mean_signed_excess"], -0.005)
        self.assertEqual(claims.bets([]), ([], 0))
        self.assertEqual(claims.tally([])["p"], None)

    def gate(self):
        from brain.decide import POLICY
        return POLICY["source_gate"]

    def raw(self, bets, hits, p, lower, mse):
        return {"bets": bets, "hits": hits, "hit_rate": hits / bets if bets else None, "p": p, "lower": lower,
                "mean_signed_excess": mse, "up": bets, "down": 0, "calls": bets, "dropped": 0, "pending": 0,
                "unscorable": 0}

    def test_bh_across_sources_removes_a_lone_p_of_0_04_among_20(self):
        lone = self.raw(40, 27, 0.04, 0.55, 0.01)
        alone = claims.gate_rows({"lone": lone}, self.gate())["lone"]
        self.assertTrue(alone["proven"])
        self.assertAlmostEqual(alone["weight"], 0.1)
        self.assertEqual(alone["why"], "proven: 40 bets, 68% hit, weight 0.10")
        crowd = {"lone": lone, **{f"s{i}": self.raw(1, 0, 1.0, 0.0, -0.01) for i in range(19)}}
        rows = claims.gate_rows(crowd, self.gate())
        self.assertAlmostEqual(rows["lone"]["q"], 0.8)
        self.assertFalse(rows["lone"]["proven"])
        self.assertEqual(rows["lone"]["weight"], 0.0)
        self.assertEqual(rows["lone"]["why"], "40 bets, 68% hit, p 0.04, q 0.80 after 20 sources: not proven")
        none = claims.gate_rows({"z": self.raw(0, 0, None, None, None) | {"calls": 0}}, self.gate())["z"]
        self.assertEqual((none["q"], none["why"]), (None, "0 bets; needs 30 bets"))

    def test_hits_with_a_negative_mean_signed_excess_are_not_proven(self):
        r = claims.gate_rows({"s": self.raw(40, 30, 0.001, 0.6, -0.002)}, self.gate())["s"]
        self.assertFalse(r["proven"])
        self.assertIn("mean signed excess -0.20%", r["why"])
        self.assertTrue(claims.gate_rows({"s": self.raw(40, 30, 0.001, 0.6, 0.002)}, self.gate())["s"]["proven"])

    def test_point_in_time(self):
        short = self.days[:41]  # the prices as they stood on days[40]
        self.bank.add_many([price_snapshot("X", short, [100 * 1.002 ** k for k in range(41)]),
                            price_snapshot("SPY", short, [100.0] * 41)])
        self.rising("X")
        self.bank.add_many([call("X", "a.example", 1, self.days[5]),                       # exit day 27: scored at day 40
                            call("X", "a.example", 1, self.days[30]),                      # exit day 52: after day 40
                            call("X", "a.example", 1, self.days[2], retrieved=self.days[45] + "T09:00:00Z")])
        at40 = claims.track_record(self.bank, self.days[40] + "T23:59:59Z")["a.example via agent"]
        self.assertEqual((at40["calls"], at40["bets"], at40["pending"]), (2, 1, 1))  # nothing retrieved after t
        later = claims.track_record(self.bank, self.t)["a.example via agent"]
        self.assertEqual((later["calls"], later["bets"], later["dropped"], later["pending"]), (3, 2, 1, 0))

    def test_unscorable_symbols_and_other_horizons_and_judgments(self):
        self.rising("X")
        self.bank.add(price_snapshot("^GSPC", self.days, [100.0] * self.n))
        self.bank.add_many([call("SPY", "a.example", 1, self.days[5]), call("^GSPC", "a.example", -1, self.days[5]),
                            call("X", "a.example", 1, self.days[5], horizon=63),
                            call("X", "llm:m@12345678", 1, self.days[5], attr="judgment"),
                            call("X", "b.example", 0, self.days[5])])
        rows = claims.track_record(self.bank, self.t)
        self.assertEqual(set(rows), {"a.example via agent", "llm:m@12345678"})
        a = rows["a.example via agent"]
        self.assertEqual((a["calls"], a["unscorable"], a["bets"], a["up"], a["down"]), (2, 2, 0, 1, 1))
        self.assertEqual(a["why"], "0 bets from 2 calls, 0 pending; needs 30 bets")
        self.assertEqual((rows["llm:m@12345678"]["bets"], rows["llm:m@12345678"]["hits"]), (1, 1))
        self.assertEqual(claims.track_record(self.bank, self.t, horizon=63)["a.example via agent"]["bets"], 1)

    def test_a_call_with_no_price_past_its_exit_is_unpriced_not_pending(self):
        self.rising("X")
        live = call("X", "a.example", 1, self.days[-30])          # exit 21 sessions on, 8 before SPY's last close
        # LAG's prices stop 10 sessions before the end: its exit, 2 sessions before SPY's last, has no close
        # yet, but SPY is still within GRACE_SESSIONS of that exit, so the call waits.
        self.bank.add(price_snapshot("LAG", self.days[:-9], [100.0] * (self.n - 9)))
        recent = call("LAG", "a.example", 1, self.days[-24])
        # D stops trading 10 sessions after the call on it (delisted after a fall): its exit has no close.
        self.bank.add(price_snapshot("D", self.days[:61], [100.0] * 51 + [100.0 * 0.8 ** k for k in range(1, 11)]))
        gone = call("D", "a.example", 1, self.days[50])
        never = call("NEVER", "a.example", 1, self.days[50])     # its prices were never ingested
        self.bank.add_many([live, recent, gone, never])
        book = PriceBook(self.bank, ["X", "LAG", "D", "NEVER", "SPY"], asof=self.t)
        self.assertEqual(claims.scored_at(book, "X", claims.anchor(live), 21, self.t)[0], "scored")
        self.assertEqual(claims.scored_at(book, "LAG", claims.anchor(recent), 21, self.t), ("pending", None))
        self.assertEqual(claims.scored_at(book, "D", claims.anchor(gone), 21, self.t), ("unpriced", None))
        self.assertEqual(claims.scored_at(book, "NEVER", claims.anchor(never), 21, self.t), ("unpriced", None))
        r = claims.track_record(self.bank, self.t)["a.example via agent"]
        self.assertEqual((r["calls"], r["bets"], r["pending"], r["unpriced"]), (4, 1, 1, 2))
        self.assertEqual(r["why"], "1 bet (1 hit) from 4 calls, 2 with no price past their exit; needs 30 bets")
        # Read back at an earlier time, SPY's book was not yet GRACE_SESSIONS past D's exit: pending then.
        at = self.days[50 + 1 + 21 + claims.GRACE_SESSIONS - 1] + "T23:59:59Z"
        self.bank.add(price_snapshot("SPY", self.days[:50 + 1 + 21 + claims.GRACE_SESSIONS],
                                     [100.0] * (50 + 1 + 21 + claims.GRACE_SESSIONS)))
        self.assertEqual(claims.scored_at(PriceBook(self.bank, ["D", "SPY"], asof=at), "D", claims.anchor(gone), 21, at)[0],
                         "pending")

    def test_prices_ingested_only_for_the_winners_do_not_make_a_record(self):
        """A source calls two stocks up in each of 35 windows: one rises, the other falls further, so every
        bet is a miss. Prices for the fallers are never ingested. Their calls are unpriced, not pending, so
        the source is not proven on its winners alone; once those prices come in, the calls are scored on
        what the stocks really did."""
        days = pd.bdate_range("2020-01-01", periods=900).strftime("%Y-%m-%d").tolist()
        n = len(days)
        self.bank.add(price_snapshot("SPY", days, [100.0] * n))
        losers = {}
        for w in range(35):
            i = 10 + w * 23  # one window per 23 sessions, so each is its own bet
            self.bank.add_many([call(f"W{w}", "coin.example", 1, days[i]), call(f"L{w}", "coin.example", 1, days[i])])
            self.bank.add(price_snapshot(f"W{w}", days, [100.0 * 1.003 ** min(max(0, k - i), 30) for k in range(n)]))
            losers[f"L{w}"] = [100.0 * 0.99 ** min(max(0, k - i), 30) for k in range(n)]
        t = days[-1] + "T23:59:59Z"
        r = claims.track_record(self.bank, t)["coin.example via agent"]
        self.assertEqual((r["calls"], r["bets"], r["hits"], r["unpriced"], r["pending"]), (70, 35, 35, 35, 0))
        self.assertTrue(r["p"] < 0.05 and r["q"] < 0.10 and r["mean_signed_excess"] > 0)  # every other test passes
        self.assertEqual((r["proven"], r["weight"]), (False, 0.0))
        self.assertIn("35 of 70 calls past their exit have no price for their stock (at most 10% may): "
                      "ingest those tickers", r["why"])
        for t_, path in losers.items():  # the fallers' prices arrive late: scored, not frozen out
            self.bank.add(price_snapshot(t_, days, path))
        r = claims.track_record(self.bank, t)["coin.example via agent"]
        self.assertEqual((r["bets"], r["hits"], r["unpriced"], r["proven"]), (35, 0, 0, False))
        self.assertLess(r["mean_signed_excess"], 0)

    def test_a_few_unpriced_calls_do_not_bar_a_record_and_are_shown(self):
        ok = self.raw(40, 30, 0.001, 0.6, 0.002) | {"calls": 44, "pending": 2, "unpriced": 2}
        r = claims.gate_rows({"s": ok}, self.gate())["s"]
        self.assertTrue(r["proven"])
        self.assertEqual(r["why"], "proven: 40 bets, 75% hit, weight 0.20; of its calls, 2 pending and 2 with no "
                                   "price past their exit")
        many = ok | {"calls": 47, "unpriced": 5}  # 5 of the 45 calls past their exit: over 10%
        r = claims.gate_rows({"s": many}, self.gate())["s"]
        self.assertEqual((r["proven"], r["weight"]), (False, 0.0))
        self.assertIn("5 of 45 calls past their exit have no price for their stock", r["why"])
        self.assertTrue(claims.too_unpriced(5, 45))
        self.assertFalse(claims.too_unpriced(4, 45))

