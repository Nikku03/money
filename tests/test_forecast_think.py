"""The thinking forecaster (docs/FORECAST.md sections 9 and 11, Think): the primer, the point-in-time snapshot,
the request, parsing and its fallback, and the model's identity."""
import hashlib
import json
import re
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from brain import config, judge, llm, series
from brain.bank import Bank, Fact
from brain.forecast import think
from brain.metrics import PriceBook
from brain.research import forecast_tests
from brain.sources import insiders
from tests.test_analyst import completion, text_msg
from tests.test_llm import FakeOpenAI, closed_port_url
from tests.test_quarterly import EPS, NI, fiscal_year, known_dates
from tests.util import TempHome, filed

ROOT = config.ROOT
DAYS = pd.bdate_range("2015-01-02", "2017-12-29").strftime("%Y-%m-%d").tolist()
D = "2017-06-15"
T_D = D + "T23:59:59Z"
SPLIT = "2017-09-01"          # 2-for-1, after D: today's price basis halves every close before it
NAME = "Example Widgets Corp"
HEADLINE = "Example Widgets beats estimates and raises guidance"
FC = {"ticker": "XCO", "date": D, "horizon": 5, "entry": "2017-06-16", "exit": "2017-06-23", "close": 7777.77,
      "p_beat": 0.5234, "band": {"q05": 7333.33, "q16": 7555.55, "q50": 7779.91, "q84": 7891.23, "q95": 7999.97},
      "move68_pct": [-3.12, 3.41], "move90_pct": [-5.6, 5.9],
      "drivers": [{"name": "dist_52w_high", "group": "B", "push": 0.081},
                  {"name": "mom_12_1", "group": "P", "push": -0.0412}],
      "model": {"id": "F1", "sha": "abc", "trained_through": "2017-06-01", "rows": 1000}}
GOOD = '{"p_beat": 0.53, "reason": "Close to the base rate; F1 leans slightly up."}'


# ---------- a synthetic bank ----------
def walk(seed: int, start: float) -> np.ndarray:
    r = np.random.default_rng(seed).normal(0.0004, 0.012, len(DAYS))
    return np.round(start * np.cumprod(1 + r), 2)


def snap(entity, dates, close, known, splits=()):
    close = np.asarray(close, dtype=float)
    df = pd.DataFrame({"date": dates, "open": close, "high": close, "low": close, "close": close, "adjclose": close,
                       "volume": np.full(len(dates), 1000)})
    return Fact(entity=entity, attr="series.price_daily", value=series.store(df), as_of=dates[-1], known_at=known,
                tier="market", source={"name": "test", "url": "https://example.test/px"},
                context={"rows": len(dates), "first": dates[0], "last": dates[-1], "splits": [list(s) for s in splits]})


def annual(attr, v, y, known, instant=False):
    end = f"{y}-12-31"
    if instant:
        return filed("XCO", attr, v, end, known, form="10-K")
    return filed("XCO", attr, v, end, known, form="10-K", start=f"{y}-01-01", duration_days=364)


def filing(known, items="2.02", form="8-K"):
    return Fact(entity="XCO", attr="filing", value=form, as_of=known[:10], known_at=known, tier="filed",
                source={"name": "SEC EDGAR", "url": f"https://example.test/{known}"},
                context={"accn": known, "items": items})


def trade(known, usd, code="P", rel="officer", accn="a1"):
    ctx = {"code": code, "price": 10.0, "value_usd": usd, "owner": "Officer", "owner_cik": "9", "relationship": rel,
           "title": None, "accn": accn, "ownership": "D"}
    return Fact(entity="XCO", attr=insiders.ATTR, value=usd / 10.0, unit="shares", as_of=known[:10], known_at=known,
                tier="filed", source={"name": "SEC Form 4", "url": f"https://example.test/{accn}"}, context=ctx)


def company_facts() -> list[Fact]:
    """XCO's filings, earnings releases, insider trades, name and a headline, some known after T(D)."""
    out = []
    eps = {2013: [1.0, 1.1, 1.2, 1.3], 2014: [1.1, 1.0, 1.4, 1.2], 2015: [1.3, 1.2, 1.5, 1.6],
           2016: [1.2, 1.5, 1.4, 1.9], 2017: [1.6, 1.7, 1.5, 2.0]}
    for y, qs in eps.items():
        out += fiscal_year("XCO", EPS, y, qs, known_dates(y))
        out += fiscal_year("XCO", NI, y, [q * 1e9 for q in qs], known_dates(y))
        out += fiscal_year("XCO", "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding", y,
                           [3e9 if y < 2017 or i < 2 else 6e9 for i in range(4)], known_dates(y))
    for y, rev in ((2014, 100e9), (2015, 110e9), (2016, 121e9), (2017, 130e9)):
        k = f"{y + 1}-02-20"
        out += [annual("us-gaap:Revenues", rev, y, k), annual("us-gaap:GrossProfit", rev * 0.4, y, k),
                annual("us-gaap:NetCashProvidedByUsedInOperatingActivities", rev * 0.06, y, k),
                annual("us-gaap:Assets", rev * 2, y, k, instant=True),
                annual("us-gaap:Liabilities", rev * 1.1, y, k, instant=True)]
    for k in ("2016-01-28T21:05:00Z", "2016-04-28T21:05:00Z", "2016-07-28T21:05:00Z", "2016-10-27T21:05:00Z",
              "2017-02-02T21:05:00Z", "2017-04-27T20:05:00Z", "2017-07-27T20:05:00Z"):
        out.append(filing(k))
    out += [trade("2017-05-10T20:00:00Z", 250_000.0, accn="b1"),
            trade("2017-06-01T20:00:00Z", 1_200_000.0, code="S", rel="director", accn="s1"),
            trade(T_D, 100_000.0, accn="b2"),                                  # known exactly at T(d): counts
            trade("2017-06-16T00:00:00Z", 9_000_000.0, accn="b3"),             # one second later: never
            trade("2017-01-02T20:00:00Z", 77_000.0, accn="old")]              # outside the 90 days
    out.append(Fact(entity="XCO", attr="identity.name", value=NAME, as_of="2015-01-01", known_at="2015-01-01",
                    tier="filed", source={"name": "SEC EDGAR", "url": "https://example.test/name"}))
    out.append(Fact(entity="XCO", attr="headline", value=HEADLINE, as_of="2017-06-14", known_at="2017-06-14T12:00:00Z",
                    tier="reported", source={"name": "news", "url": "https://example.test/news"}))
    out.append(Fact(entity="XCO", attr="split", value=2.0, as_of=SPLIT, known_at=SPLIT + "T12:00:00Z", tier="filed",
                    source={"name": "test", "url": "https://example.test/split"}))
    return out


class Market(TempHome):
    """XCO and SPY on 2015-2017 sessions. ``self.bank`` holds everything, fetched at the end of 2017 on today's
    price basis (XCO split 2-for-1 on 2017-09-01, after D). ``cut()`` is the same world as it stood at T(D)."""

    def setUp(self):
        super().setUp()
        self.spy, self.xco = walk(3, 200.0), walk(5, 60.0)
        self.raw = company_facts()
        today_basis = np.where(np.array(DAYS) < SPLIT, self.xco / 2, self.xco)
        self.bank.add(snap("SPY", DAYS, self.spy, DAYS[-1]))
        self.bank.add(snap("XCO", DAYS, today_basis, DAYS[-1], splits=[(SPLIT, 2.0)]))
        self.bank.add_many(self.raw)
        self.book = PriceBook(self.bank, ["XCO", "SPY"], DAYS[-1] + "T23:59:59Z")

    def cut(self) -> tuple[Bank, PriceBook]:
        """Only facts known by T(D), and prices up to D as they then traded (no split yet)."""
        root = Path(self._tmp.name) / "cut"
        b = Bank(root)
        n = DAYS.index(D) + 1
        b.add(snap("SPY", DAYS[:n], self.spy[:n], T_D))
        b.add(snap("XCO", DAYS[:n], self.xco[:n], T_D))
        b.add_many([f for f in self.raw if f.known_at <= T_D])
        return b, PriceBook(b, ["XCO", "SPY"], T_D)


class SnapshotTest(Market):
    def test_point_in_time(self):
        full = think.snapshot(self.bank, self.book, "XCO", D, FC)
        cut, book = self.cut()
        self.assertEqual(full, think.snapshot(cut, book, "xco", D, FC))
        # Facts known one second after T(d) change nothing: an earnings release, a restated year, insider buys.
        late = "2017-06-16T00:00:00Z"
        self.bank.add_many([filing(late), annual("us-gaap:Revenues", 1.0, 2016, late), trade(late, 5e9, accn="z9"),
                            filed("XCO", EPS, 9.9, "2017-03-31", late, form="10-Q/A", start="2017-01-01",
                                  duration_days=89)])
        self.assertEqual(full, think.snapshot(Bank(self.bank.root), self.book, "XCO", D, FC))
        # The request carries exactly this snapshot.
        user = think.messages(self.bank, self.book, "XCO", D, FC, "fake-model")[1]["content"]
        self.assertTrue(user.startswith(full + "\n\n"), user[:200])

    def test_what_it_holds_and_leaves_out(self):
        s = think.snapshot(self.bank, self.book, "XCO", D, FC)
        lines = s.splitlines()
        self.assertEqual(lines[0], f"Snapshot of XCO after the close of {D}. Only facts public by then; "
                                   f"no company name and no headlines.")
        i = DAYS.index(D)
        x = self.xco

        def pct(v):
            return f"{v * 100:+.1f}%"
        self.assertIn(f"1 week {pct(x[i] / x[i - 5] - 1)} (SPY {pct(self.spy[i] / self.spy[i - 5] - 1)})", s)
        self.assertIn(f"12 months {pct(x[i] / x[i - 252] - 1)}", s)
        w = x[i - 251: i + 1]
        self.assertIn(f"Close against its 52-week high {pct(x[i] / w.max() - 1)}, against its 52-week low "
                      f"{pct(x[i] / w.min() - 1)}.", s)
        v = x[i - 63: i + 1]
        sd = np.log(v[1:] / v[:-1]).std(ddof=1)
        self.assertIn(f"Volatility: {sd * 100:.2f}% a day over the last 63 sessions", s)
        self.assertIn(f"a typical 5-session move is about {sd * np.sqrt(5) * 100:.1f}%", s)
        # Fundamentals from the month-end view before D's month (F1's rule); the 2017 10-K is a year away.
        self.assertIn("Fundamentals (filings known at the month-end view, 2017-05-31):", s)
        self.assertIn("Latest fiscal year (ended 2016-12-31, filed 2017-02-20): revenue 121.00B (+10.0% on the year "
                      "before)", s)
        self.assertIn("liabilities over assets 0.55; gross profit over assets 0.20", s)
        eps = next(ln for ln in lines if ln.startswith("- EPS by quarter"))
        self.assertEqual(re.findall(r"(\d{4}-\d{2}-\d{2}): ", eps), ["2016-06-30", "2016-09-30", "2016-12-31", "2017-03-31"])
        self.assertIn("2017-03-31: 1.60 (+33% on a year earlier, SUE", eps)
        self.assertIn("Market value ", s)
        # The latest release known at T(d), with its closed 3-day reaction; insider trades in the 90 days to d,
        # one filed exactly at T(d) included, one a second later not.
        self.assertIn("Last earnings release: an 8-K (item 2.02) filed 2017-04-27, 49 days before; 3-day reaction "
                      "against SPY ", s)
        self.assertIn(f"Insider trades (Forms 4 and 5) filed in the 90 days to {D}: 2 open-market buys (350.00K USD), "
                      f"1 sale (1.20M USD).", s)
        # Never: the company's name, a headline, a price level (today's basis or as traded), or anything of F1's
        # forecast: shown it, small models copied F1's number, so the model is asked for its own view.
        for gone in (NAME, "Example Widgets", HEADLINE, f"{x[i]:.2f}", f"{x[i] / 2:.2f}", "7777.77", "7891.23", "7333",
                     "F1", "0.523", "-3.1%", "dist_52w_high", "mom_12_1"):
            self.assertNotIn(gone, s)

    def test_month_end_view_and_open_windows(self):
        before = think.snapshot(self.bank, self.book, "XCO", D, FC)
        # A 10-K amendment filed after the month-end but before d reaches the snapshot only next month.
        self.bank.add(annual("us-gaap:Revenues", 125e9, 2016, "2017-06-05T12:00:00Z"))
        b = Bank(self.bank.root)
        self.assertEqual(before, think.snapshot(b, self.book, "XCO", D, FC))
        self.assertIn("revenue 125.00B", think.snapshot(b, self.book, "XCO", "2017-07-05", FC))
        # The day after a release, its 3-day window is still open.
        s = think.snapshot(b, self.book, "XCO", "2017-04-28", FC)
        self.assertIn("Last earnings release: an 8-K (item 2.02) filed 2017-04-27, 1 day before; its 3-day reaction "
                      "window had not closed.", s)
        # No forecast (the 5-session default), a 21-session forecast, a stock with no prices: still a snapshot.
        self.assertIn("a typical 5-session move", think.snapshot(b, self.book, "XCO", D, None))
        s21 = think.snapshot(b, self.book, "XCO", D, {**FC, "horizon": 21})
        self.assertIn("a typical 21-session move", s21)
        self.assertIn("Prices: none in the bank up to this date.", think.snapshot(b, self.book, "NOPE", D, FC))
        self.assertIn("filed in the 90 days to 2016-06-15: 0 open-market buys, 0 sales.",
                      think.snapshot(b, self.book, "XCO", "2016-06-15", FC))

    def test_snapshot_version_and_messages(self):
        self.assertEqual(think.SNAPSHOT_VERSION, 2)
        system, user = think.messages(self.bank, self.book, "XCO", D, FC, "llama3.1:8b")
        self.assertEqual(system["role"], "system")
        self.assertTrue(system["content"].startswith(judge.PRIMER))
        for rule in ("Give the probability that this stock beats SPY over the next 5 sessions after the snapshot",
                     'reply with only a JSON object, no other text: {"p_beat": 0.0-1.0, "reason": "one or two sentences"}',
                     "Never use knowledge of events after the snapshot date."):
            self.assertIn(rule.lower(), system["content"].lower())
        self.assertTrue(user["content"].endswith("Give the probability that XCO beats SPY over the next 5 sessions "
                                                 f"after {D}. Reply with only the JSON object."))
        self.assertIn("next 21 sessions", think.messages(self.bank, self.book, "XCO", D, {**FC, "horizon": 21})[0]["content"])


class PrimerTest(unittest.TestCase):
    TEXT = (ROOT / "docs" / "MARKET_PRIMER.md").read_text(encoding="utf-8")

    def test_no_date_in_or_after_the_declared_windows(self):
        starts = [t["start"] for t in forecast_tests.FORECAST_TESTS if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(t.get("start")))]
        first = min(starts)
        self.assertEqual(first, "2019-01-01")
        years = [int(y) for y in re.findall(r"(?<!\d)(1[89]\d\d|2\d\d\d)(?!\d)", self.TEXT)]
        self.assertTrue(years)
        self.assertLess(max(years), int(first[:4]), sorted(set(years)))
        self.assertLessEqual(max(years), 2026)
        self.assertEqual(re.findall(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}", self.TEXT), [])
        for month in ("January", "February", "March", "April", "June", "July", "August", "September", "October",
                      "November", "December"):
            self.assertNotIn(month, self.TEXT)

    def test_the_banks_own_figures_are_train_only(self):
        rows = [ln.split("|")[1:-1] for ln in (ROOT / "research" / "RESULTS.md").read_text().splitlines()
                if re.match(r"\| H\w", ln)]
        head = next(ln for ln in (ROOT / "research" / "RESULTS.md").read_text().splitlines()
                    if ln.startswith("| Hypothesis"))
        cols = [c.strip() for c in head.split("|")[1:-1]]
        cell = [{c: v.strip() for c, v in zip(cols, r)} for r in rows]
        self.assertGreater(len(cell), 5)
        train = {r["TRAIN net excess / mo"].replace("%", "") for r in cell} | {r["TRAIN p"] for r in cell}
        test_only = {r[c] for r in cell for c in ("Months won", "Perm p", "BH q", "Annualised", "Info ratio")
                     if r[c] not in ("–", "-")}
        for v in test_only:  # no TEST figure anywhere in the primer
            self.assertIsNone(re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", self.TEXT), v)
        section = self.TEXT.split("## This bank's own tests", 1)[1].split("\n## ", 1)[0]
        allowed = {"2011", "2018", "60", "741", "12", "1", "0.05"} | {v.lstrip("+") for v in train} | train
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?", section)
        self.assertTrue(any("0.0148" == n for n in nums), nums)
        for n in nums:
            self.assertTrue(n in allowed or n.lstrip("+-") in allowed, f"{n} is not a TRAIN figure from RESULTS.md")
        self.assertIn("every hypothesis failed its TEST gate", section)

    def test_size_and_contents(self):
        self.assertLessEqual(len(self.TEXT.split()), 2000)
        low = self.TEXT.lower()
        for topic in ("expectations", "priced", "market's move", "guidance", "rates", "liquidity", "flows",
                      "about half", "volatility", "confident calls are usually wrong",
                      "momentum", "short-term reversal", "long-term reversal", "value", "profitability",
                      "low volatility", "post-earnings drift", "52-week high", "lottery", "insider buying", "attention",
                      "after publication", "overconfidence", "herding", "loss aversion", "disposition effect",
                      "anchoring", "recency", "narrative", "overreaction", "underreaction",
                      "start from the base rate", "already priced in", "tested record", "0.45", "0.55",
                      "0.60 is a strong view", "never use knowledge of events after the snapshot date",
                      "private individuals"):
            self.assertIn(topic, low, topic)
        self.assertEqual(think.primer(), self.TEXT.strip())

    def test_primer_and_prompt_pins(self):
        self.assertEqual(think.primer_sha(), hashlib.sha256((ROOT / "docs" / "MARKET_PRIMER.md").read_bytes()).hexdigest()[:16])
        self.assertEqual(think.prompt_sha(), forecast_tests._sha(think.PROMPT))
        self.assertEqual(forecast_tests.llm_pins(think), {"primer_sha": think.primer_sha(), "prompt_sha": think.prompt_sha(),
                                                          "snapshot_version": 2})
        for k in ("system", "user", "repair", "qwen3", "no_think", "temperature", "seed", "max_tokens"):
            self.assertIn(k, think.PROMPT)
        self.assertEqual((think.PROMPT["temperature"], think.PROMPT["seed"], think.PROMPT["max_tokens"]), (0, 7, 300))
        self.assertIn("{primer}", think.PROMPT["system"])  # the primer is pinned by its own sha
        # Frozen: a new prompt is a new F1d test. Change this on purpose.
        self.assertEqual(think.prompt_sha(), "9cf7aa3ed50e27bb")


class ParseTest(unittest.TestCase):
    def test_robust_order(self):
        cases = [
            (GOOD, 0.53),
            ("```json\n" + GOOD + "\n```", 0.53),
            ("<think>\nthe base rate is 0.5 {not json}\n</think>\n\n" + GOOD, 0.53),
            ("Here you go: " + GOOD + " Hope that helps.", 0.53),
            ('{"answer": {"p_beat": 0.48, "reason": "nested"}}', 0.48),
            ('{"p_beat": "55%", "reason": "x"}', 0.55),
            ('{"p_beat": 55, "reason": "a percent written as a number"}', 0.55),
            ('{"p_beat": "0.61"}', 0.61),
            ('{"p_beat": 0}', 0.01),
            ('{"p_beat": 1}', 0.99),
            ('{"p_beat": 0.999}', 0.99),
            ('{"p_beat": 0.123456, "reason": "rounded"}', 0.1235),
        ]
        for text, want in cases:
            with self.subTest(text=text[:40]):
                call, why = think.parse(text)
                self.assertEqual(why, "")
                self.assertEqual(call["p_beat"], want)
        call, _ = think.parse('{"p_beat": 0.5, "reason": "line one\\n   line two ' + "x" * 400 + '"}')
        self.assertTrue(call["reason"].startswith("line one line two"))
        self.assertLessEqual(len(call["reason"]), 300)
        self.assertEqual(think.parse('{"p_beat": 0.5, "reason": 7}')[0]["reason"], "")

    def test_failures_say_why(self):
        for text, why in [("I think it is about even.", "holds no JSON object"),
                          ('{"probability": 0.5}', "no JSON object with a p_beat"),
                          ('{"p_beat": true}', "is not a probability"),
                          ('{"p_beat": 150}', "is not a probability"),
                          ('{"p_beat": -0.2}', "is not a probability"),
                          ('{"p_beat": "high"}', "is not a probability"),
                          ('{"p_beat": NaN}', "is not a probability"),
                          ('{"p_beat": 0.55, "reason": "cut off', "no JSON object with a p_beat"),
                          ("", "holds no JSON object")]:
            with self.subTest(text=text):
                call, got = think.parse(text)
                self.assertIsNone(call)
                self.assertIn(why, got)


class LLMForecastTest(Market):
    def ask(self, reply, model="qwen3:8b", **kw):
        with FakeOpenAI(reply, models=(model,)) as srv:
            res = think.llm_forecast(self.bank, self.book, "XCO", D, FC, client=srv.client(model=model), **kw)
        return res, srv.requests

    def test_one_request_no_tools_temperature_0_seed_7(self):
        res, sent = self.ask(lambda n, b: text_msg("<think>\n\n</think>\n\n" + GOOD))
        self.assertEqual(len(sent), 1)
        body = sent[0]
        self.assertEqual((body["model"], body["temperature"], body["seed"], body["max_tokens"]), ("qwen3:8b", 0, 7, 300))
        self.assertNotIn("tools", body)
        system, user = body["messages"]
        self.assertEqual(system, {"role": "system", "content": think.SYSTEM.format(primer=judge.PRIMER, horizon=5)})
        self.assertIn(judge.PRIMER, system["content"])
        self.assertTrue(user["content"].endswith(" /no_think"))
        self.assertIn(think.snapshot(self.bank, self.book, "XCO", D, FC), user["content"])
        self.assertTrue(res["ok"], res)
        self.assertEqual((res["p_beat"], res["reason"], res["error"], res["down"]),
                         (0.53, "Close to the base rate; F1 leans slightly up.", None, None))
        self.assertEqual(res["model"], "served-model")
        self.assertTrue(res["raw"].startswith("<think>"))
        self.assertIsInstance(res["seconds"], float)
        self.assertEqual(set(res), {"ok", "p_beat", "reason", "raw", "model", "seconds", "error", "down"})

    def test_no_think_only_for_qwen3(self):
        for model, suffix in (("qwen3:8b", True), ("/models/Qwen3-1.7B-Q8_0.gguf", True), ("Qwen/Qwen3-8B", True),
                              ("qwen2.5:7b", False), ("llama3.1:8b", False), ("qwen3.5:9b", False)):
            with self.subTest(model=model):
                res, sent = self.ask(lambda n, b: text_msg(GOOD), model=model)
                self.assertTrue(res["ok"])
                self.assertEqual(sent[0]["messages"][1]["content"].endswith(" /no_think"), suffix)
                self.assertEqual(sent[0]["messages"][1]["content"].count("/no_think"), int(suffix))
                self.assertEqual(think.is_qwen3(model), suffix)

    def test_no_think_follows_the_served_model_not_the_name_asked_for(self):
        # llama.cpp serves its one GGUF whatever model name a request carries, and F1d's identity pins that
        # file: the family that decides /no_think is the served model's, never BRAIN_LLM_MODEL's text.
        for served, asked, suffix in (("/models/Qwen3-1.7B-Q8_0.gguf", "local-llama", True),
                                      ("/models/Llama-3.1-8B-Q8_0.gguf", "qwen3:8b", False),
                                      ("qwen3:8b", "qwen3:8b", True)):
            with self.subTest(served=served, asked=asked):
                with FakeOpenAI(lambda n, b: text_msg(GOOD), models=(served,)) as srv:
                    res = think.llm_forecast(self.bank, self.book, "XCO", D, FC, client=srv.client(model=asked))
                self.assertTrue(res["ok"], res)
                self.assertEqual(srv.requests[0]["messages"][1]["content"].endswith(" /no_think"), suffix)
                self.assertEqual(think.is_qwen3(served), suffix)
        # a declared test passes its pinned decision, and that is what is sent
        with FakeOpenAI(lambda n, b: text_msg(GOOD), models=("qwen3:8b",)) as srv:
            think.llm_forecast(self.bank, self.book, "XCO", D, FC, client=srv.client(model="qwen3:8b"), no_think=False)
        self.assertFalse(srv.requests[0]["messages"][1]["content"].endswith(" /no_think"))
        # a server whose model list cannot be read: the client's own name decides
        with FakeOpenAI(lambda n, b: text_msg(GOOD), models=("qwen3:8b",), model_status=500) as srv:
            think.llm_forecast(self.bank, self.book, "XCO", D, FC, client=srv.client(model="qwen3:8b"))
        self.assertTrue(srv.requests[0]["messages"][1]["content"].endswith(" /no_think"))

    def test_one_repair_then_the_answer(self):
        res, sent = self.ask(lambda n, b: text_msg("About even, maybe 0.52." if n == 1 else GOOD))
        self.assertEqual(len(sent), 2)
        msgs = sent[1]["messages"]
        self.assertEqual(msgs[:2], sent[0]["messages"])
        self.assertEqual(msgs[2], {"role": "assistant", "content": "About even, maybe 0.52."})
        self.assertEqual(msgs[3], {"role": "user", "content": think.REPAIR + " /no_think"})
        self.assertEqual((sent[1]["seed"], sent[1]["temperature"], sent[1]["max_tokens"]), (7, 0, 300))
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["p_beat"], 0.53)
        self.assertEqual(res["raw"], "About even, maybe 0.52.\n--- repair ---\n" + GOOD)

    def test_a_failed_answer_is_reported_never_a_number(self):
        res, sent = self.ask(lambda n, b: text_msg("I cannot say."), model="llama3.1:8b")
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[1]["messages"][3]["content"], think.REPAIR)
        self.assertFalse(res["ok"])
        self.assertIsNone(res["p_beat"])
        self.assertIn("even after one repair: the answer holds no JSON object", res["error"])
        self.assertIsNone(res["down"])
        # Cut off at the length limit, twice.
        res, sent = self.ask(lambda n, b: completion(text_msg('{"p_beat": 0.5, "reason": "a long'), finish="length"))
        self.assertEqual((res["ok"], len(sent)), (False, 2))
        self.assertIn("cut off at the length limit", res["error"])

    def test_it_never_raises(self):
        res = think.llm_forecast(self.bank, self.book, "XCO", D, FC, client=llm.Client(url=closed_port_url(), model="m"))
        self.assertFalse(res["ok"])
        self.assertIsNone(res["p_beat"])
        self.assertIn("could not reach the model server", res["down"])
        self.assertEqual(res["error"], res["down"])
        res, sent = self.ask(lambda n, b: (500, {"error": {"message": "model crashed"}}))
        self.assertEqual((res["ok"], len(sent)), (False, 1))
        self.assertIn("model crashed", res["error"])
        with mock.patch.object(self.bank, "view", side_effect=RuntimeError("disk on fire")):
            res, sent = self.ask(lambda n, b: text_msg(GOOD))
        self.assertEqual((res["ok"], res["error"], len(sent)), (False, "RuntimeError: disk on fire", 0))
        res, sent = self.ask(lambda n, b: text_msg(GOOD), deadline=time.monotonic() - 1)
        self.assertEqual((res["ok"], res["error"], len(sent)), (False, "no time left to ask the model", 0))
        res, sent = self.ask(lambda n, b: text_msg("not json"), deadline=time.time() + 60)
        self.assertEqual((res["ok"], len(sent)), (False, 2))


# ---------- identity ----------
class FakeOllama:
    """Ollama's two endpoints the identity reads: GET /v1/models and POST /api/show."""

    def __init__(self, models=("qwen3:8b",), show=None, show_status=200):
        self.models, self.show, self.show_status, self.shown = list(models), show or {}, show_status, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, payload):
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._send(200, {"object": "list", "data": [{"id": m} for m in outer.models]})
                self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != "/api/show":
                    return self._send(404, {"error": "not found"})
                outer.shown.append(body)
                self._send(outer.show_status, outer.show)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class IdentityTest(unittest.TestCase):
    BLOB = "ab" * 32

    def test_llama_cpp_gguf_path_is_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "Qwen3-Tiny-Q8_0.gguf"
            p.write_bytes(b"GGUF" + bytes(range(256)) * 5000)
            want = "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
            with FakeOpenAI(models=(str(p),)) as srv:
                ident = think.llm_identity(srv.client(model="anything"))
                self.assertEqual((ident["model"], ident["digest"], ident["why"]), ("qwen3-tiny-q8_0", want, None))
                self.assertIn(str(p.resolve()), ident["via"])
                # /no_think is pinned by the served file's family, whatever name the client asks for
                self.assertEqual(forecast_tests.llm_identity(srv.client(model="anything"), think=think),
                                 {"model": "qwen3-tiny-q8_0", "digest": want, "no_think": True})
            p.write_bytes(b"GGUF other weights")  # new weights, same path: a new digest
            with FakeOpenAI(models=(str(p),)) as srv:
                self.assertNotEqual(think.llm_identity(srv.client())["digest"], want)

    def test_ollama_show_names_the_weights_blob(self):
        show = {"modelfile": f"# Modelfile\nFROM /root/.ollama/models/blobs/sha256-{self.BLOB}\nTEMPLATE x\n",
                "details": {"family": "qwen3"}}
        with FakeOllama(models=("qwen3:8b", "llama3.1:8b"), show=show) as srv:
            ident = think.llm_identity(llm.Client(url=srv.url, model="qwen3:8b"))
            self.assertEqual(ident, {"model": "qwen3:8b", "digest": f"sha256:{self.BLOB}", "via": "ollama /api/show",
                                     "why": None})
            self.assertEqual(srv.shown[0]["model"], "qwen3:8b")
        with FakeOllama(models=("qwen3:latest",), show={"digest": "cd" * 32}) as srv:
            ident = think.llm_identity(llm.Client(url=srv.url, model="qwen3"))
            self.assertEqual((ident["model"], ident["digest"]), ("qwen3", "sha256:" + "cd" * 32))

    def test_the_digest_refusal(self):
        for models, show_status, model in ((("my-alias",), 404, "my-alias"), (("/no/such/file.gguf",), 404, "x"),
                                           (("a", "b"), 404, "c"), (("qwen3:8b",), 200, "qwen3:8b")):
            with self.subTest(models=models), FakeOllama(models=models, show={"details": {}}, show_status=show_status) as srv:
                c = llm.Client(url=srv.url, model=model)
                ident = think.llm_identity(c)
                self.assertIsNone(ident["digest"])
                self.assertTrue(ident["why"])
                with self.assertRaises(forecast_tests.Refused) as cm:
                    forecast_tests.llm_identity(c, think=think)
                self.assertIn("digest", str(cm.exception))
        with self.assertRaises(llm.LLMError):
            think.llm_identity(llm.Client(url=closed_port_url(), model="m"))
        with self.assertRaises(forecast_tests.Refused):
            forecast_tests.llm_identity(llm.Client(url=closed_port_url(), model="m"), think=think)


if __name__ == "__main__":
    unittest.main()
