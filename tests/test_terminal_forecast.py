"""The terminal's FCST (docs/FORECAST.md section 10, docs/TERMINAL.md): T FCST and FCST alone over recorded
forecasts, T FCST NEW as a background job of kind ``forecast`` with DEC NEW's rules (its own slot, one per
ticker, reruns never start one), refused to the read-only reader, the analyst's forecast tool, and RES with
forecast results in the ledger. Forecasts are written as facts of the contract's shape (``live.fact_for``), so
nothing here trains a model."""
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from brain import config
from brain.forecast import live
from brain.terminal import commands, data, jobs, server
from brain.terminal.commands import FUNCTIONS, CommandError, complete, fcst_line, parse
from brain.timeutil import now_iso
from tests.test_forecast_product import DAYS, add_prices, forecast_fact, score_paths, snap
from tests.util import TempHome, filed


def fake_argv(target):
    return [sys.executable, "-c", f"print('fake job {target}')"]


class FcstGrammarTest(unittest.TestCase):
    def check(self, line, fn, ticker, args=()):
        c = parse(line)
        self.assertEqual((c.fn, c.ticker, c.args), (fn, ticker, list(args)), line)

    def test_fcst(self):
        self.check("AAPL FCST", "FCST", "AAPL")
        self.check("fcst aapl", "FCST", "AAPL")
        self.check("FCST", "FCST", None)  # every stock's latest forecast and live scoring
        self.check("AAPL FCST NEW", "FCST", "AAPL", ["NEW"])
        self.check("fcst new aapl", "FCST", "AAPL", ["NEW"])  # NEW is FCST's keyword, never the ticker
        self.check("FCST AAPL NEW", "FCST", "AAPL", ["NEW"])
        self.check("AAPL US EQUITY FCST NEW <GO>", "FCST", "AAPL", ["NEW"])
        self.check("NEW FCST", "FCST", "NEW")
        self.check("CF FCST NEW", "FCST", "CF", ["NEW"])
        self.check("FCST DES", "DES", "FCST")  # a ticker named FCST: [TICKER] FUNCTION wins
        self.check("DEC FCST NEW", "FCST", "DEC", ["NEW"])
        self.check("FCST DEC NEW", "DEC", "FCST", ["NEW"])
        self.check("INGEST FCST", "INGEST", "FCST")
        with self.assertRaises(CommandError) as cm:
            parse("FCST NEW")
        self.assertEqual((cm.exception.message, cm.exception.hint), ("FCST NEW needs a ticker, e.g. AAPL FCST NEW", "FCST"))
        for line, hint in (("AAPL FCST FOO", "AAPL FCST"), ("FCST NEW NEW", "FCST"), ("AAPL FCST NEW 5Y", "AAPL FCST")):
            with self.assertRaises(CommandError, msg=line) as cm:
                parse(line)
            self.assertEqual((cm.exception.message, cm.exception.hint), ("FCST takes no arguments, or NEW", hint))
        self.assertEqual((fcst_line("AAPL"), fcst_line("CF", new=True), fcst_line("NEW")),
                         ("AAPL FCST", "CF FCST NEW", "NEW FCST"))
        for t in ("INGEST", "SHOW", "ASK", "HELP"):
            self.assertEqual((fcst_line(t), fcst_line(t, new=True)), (None, None), t)
        self.assertEqual((FUNCTIONS["FCST"]["scope"], FUNCTIONS["FCST"]["args"]), ("both", "[NEW]"))
        self.assertEqual(list(FUNCTIONS).index("FCST"), list(FUNCTIONS).index("DEC") + 1)

    def test_completion(self):
        tickers = ["AAPL", "ABT", "SPY"]
        texts = lambda q: [d["text"] for d in complete(q, tickers)]  # noqa: E731
        self.assertEqual(texts("AAPL FCST "), ["NEW"])
        self.assertEqual(complete("AAPL FCST N", tickers)[0],
                         {"text": "NEW", "kind": "keyword",
                          "desc": "forecast afresh in a background job (paper only): T FCST NEW"})
        self.assertEqual(texts("FCST "), ["NEW"])
        self.assertEqual(texts("FCST A"), ["AAPL", "ABT"])
        self.assertEqual(texts("AAPL FCST NEW "), [])
        self.assertIn("FCST", texts("AAPL F"))
        self.assertEqual(complete("AAPL DEC N", tickers)[0]["desc"], "decide afresh in a background job (paper only): T DEC NEW")


class FcstBase(TempHome):
    """AAA and BBB with recorded forecasts (one scored, one pending, the model's answer beside one), SPY flat."""

    def setUp(self):
        super().setUp()
        home = Path(self._tmp.name)
        self.patches = [mock.patch.object(config, "LEDGER", home / "ledger.jsonl"),
                        mock.patch.object(config, "RESULTS", home / "RESULTS.md")]
        for p in self.patches:
            p.start()
        p = score_paths()
        for i in (510, len(DAYS) - 2):
            add_prices(self.bank, paths=p, upto=i + 1)
        add_prices(self.bank, paths=p)
        self.bank.add(filed("AAA", "identity.name", "Triple A Co", DAYS[0], DAYS[0]))
        self.old = forecast_fact(self.bank, "AAA", 510, 0.6)
        self.new = forecast_fact(self.bank, "AAA", len(DAYS) - 2, 0.55)
        self.llm = forecast_fact(self.bank, "AAA", len(DAYS) - 2, 0.62, forecaster="F1-llm")
        self.long = forecast_fact(self.bank, "AAA", len(DAYS) - 2, 0.52, h=21)
        self.bbb = forecast_fact(self.bank, "BBB", 510, 0.4, sigma=0.04)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        super().tearDown()

    def run_line(self, line, writes=False):
        return data.run(self.bank, parse(line), writes=writes)


class FcstOneTest(FcstBase):
    def test_the_latest_forecasts_with_their_bands_and_the_chart(self):
        out = self.run_line("AAA FCST")
        self.assertTrue(out["ok"], out)
        self.assertEqual((out["view"], out["fn"], out["ticker"]), ("forecast", "FCST", "AAA"))
        self.assertEqual(out["title"], "AAA — Triple A Co · forecast · 55% to beat SPY")
        d = out["data"]
        self.assertEqual([f["horizon"] for f in d["forecasts"]], [5, 21])
        f5 = d["forecasts"][0]
        self.assertEqual((f5["date"], f5["p_beat"], f5["cite"], f5["forecaster"]), (DAYS[-2], 0.55, self.new.id[:8], "F1"))
        self.assertEqual(set(f5["band"]), {"q05", "q16", "q50", "q84", "q95"})
        self.assertEqual(f5["llm"]["p_beat"], 0.62)  # the model's answer on the same date sits beside F1's
        self.assertEqual(f5["llm"]["cite"], self.llm.id[:8])
        self.assertEqual(f5["outcome"]["status"], "pending")
        self.assertEqual({k for k, v in f5["outcome"].items() if v is not None}, {"status"})
        self.assertIn("last 2 months (F1a): not run yet", f5["record"])
        self.assertIn("the model on F1a's weeks (F1d): not run yet", f5["record"])
        self.assertIsNone(d["forecasts"][1]["llm"])
        self.assertEqual(d["retrieved_at"], self.new.retrieved_at)
        # the chart: three months of closes before the newest forecast, then the windows' sessions (weekdays)
        dates = [x["date"] for x in d["series"]]
        self.assertEqual(dates[-1], DAYS[-1])
        self.assertLessEqual(dates[0], DAYS[-60])
        self.assertTrue(all(x > DAYS[-1] for x in d["future"]))
        self.assertEqual(d["future"][-1], self.long.context["exit"])  # the 21-session window's expected exit
        self.assertEqual(d["cite"], self.bank.view(DAYS[-1] + "T23:59:59Z").latest("AAA", "series.price_daily").id[:8])
        hist = d["history"]["rows"]
        self.assertEqual([(r["date"], r["horizon"], r["forecaster"]) for r in hist],
                         [(DAYS[-2], 5, "F1"), (DAYS[-2], 5, "F1-llm"), (DAYS[-2], 21, "F1"), (DAYS[510], 5, "F1")])
        self.assertEqual((hist[-1]["status"], hist[-1]["in68"]), ("scored", "out"))
        live_rows = {r["stream"]: r for r in d["live"]["rows"]}
        self.assertEqual((live_rows["F1 h=5"]["forecasts"], live_rows["F1 h=5"]["scored"]), (3, 2))
        self.assertEqual(live_rows["F1 h=5"]["cover68"], "50%")  # a share, not an excess: plain text, unsigned
        self.assertEqual((hist[0]["chance"], hist[0]["p_beat"]), ("55%", 0.55))
        self.assertIn("paper only", d["note"].lower())

    def test_a_scored_forecast_shows_its_outcome(self):
        out = self.run_line("BBB FCST")
        f = out["data"]["forecasts"][0]
        self.assertEqual(f["outcome"]["status"], "scored")
        self.assertEqual((f["outcome"]["in68"], f["outcome"]["in90"], f["outcome"]["beat"]), (True, True, False))
        self.assertEqual(f["outcome"]["exit"], DAYS[516])

    def test_no_forecast_yet(self):
        self.bank.add(snap("CCC", DAYS, np.full(len(DAYS), 30.0)))
        out = self.run_line("CCC FCST")
        self.assertEqual((out["ok"], out["error"], out["hint"]), (False, "no forecast recorded for CCC yet", "CCC FCST NEW"))
        for t in ("SPY", "^GSPC"):
            out = self.run_line(f"{t} FCST")
            self.assertEqual((out["ok"], out["hint"]), (False, "FCST"), out)
            self.assertTrue(out["error"].startswith(f"no forecasts for {t}: "))
        self.assertEqual(self.run_line("ASK FCST")["fn"], "ASK")  # ASK first is always ASK: no FCST on a ticker ASK

    def test_fcst_new_never_runs_in_the_data_layer(self):
        for writes in (False, True):
            out = self.run_line("AAA FCST NEW", writes=writes)
            self.assertEqual((out["ok"], out["hint"]), (False, "AAA FCST"))
            self.assertEqual(out["error"], "FCST NEW starts a background forecast job; it runs only from the terminal's "
                                           "own command line")
        self.assertTrue(data.is_write(parse("AAA FCST NEW")))
        self.assertFalse(data.is_write(parse("AAA FCST")))
        out = data.run(self.bank, commands.Command(fn="FCST", ticker="AAA", args=["FOO"]))
        self.assertEqual((out["ok"], out["error"]), (False, "FCST takes no arguments, or NEW"))

    def test_no_nan_anywhere(self):
        text = json.dumps(self.run_line("AAA FCST"), allow_nan=False)
        self.assertIn("forecast", text)


class FcstAllTest(FcstBase):
    def test_every_stock_live_scoring_and_the_declared_tests(self):
        out = self.run_line("FCST")
        self.assertTrue(out["ok"], out)
        self.assertEqual((out["view"], out["ticker"]), ("forecasts", None))
        rows = out["data"]["latest"]["rows"]
        self.assertEqual([(r["ticker"], r["date"], r["horizon"]) for r in rows], [("AAA", DAYS[-2], 5), ("BBB", DAYS[510], 5)])
        self.assertEqual(rows[0]["_cmds"], {"ticker": "AAA FCST"})
        self.assertEqual(rows[1]["status"], "scored")
        self.assertEqual((rows[0]["chance"], rows[1]["chance"]), ("55%", "40%"))
        self.assertEqual([r["horizon"] for r in out["data"]["record"]], [5, 21])
        self.assertIn("2019-2026 (F1c): not run yet", out["data"]["record"][1]["lines"])
        self.assertEqual({r["stream"] for r in out["data"]["live"]["rows"]}, {"F1 h=5", "F1 h=21", "F1-llm h=5"})

    def test_an_empty_bank_answers_with_notes(self):
        bank = type(self.bank)(Path(self._tmp.name) / "empty")
        out = data.run(bank, parse("FCST"))
        self.assertTrue(out["ok"], out)
        self.assertIn("T FCST NEW makes one", out["data"]["latest"]["note"])
        self.assertEqual(out["data"]["live"]["rows"], [])


class FcstResTest(FcstBase):
    def test_res_lists_forecast_results_without_breaking(self):
        recs = [{"op": "declare", "kind": "forecast", "id": "F1b-w5-test", "spec_sha": "s",
                 "spec": {"claim": "F1 ranks stocks beyond size", "universe": "universe/us_broad.json"}, "at": "x"},
                {"op": "result", "kind": "forecast", "id": "F1b-w5-test", "spec_sha": "s", "at": "2026-10-16T00:00:00Z",
                 "gate_type": "ic", "p": 0.4, "gate": {"pass": False}, "controls": {"pass": True}, "metrics": {}},
                {"op": "declare", "kind": "forecast", "id": "F1c-m21-test", "spec_sha": "t",
                 "spec": {"claim": "monthly", "universe": "universe/us_broad.json"}, "at": "x"}]
        config.LEDGER.write_text("".join(json.dumps(r) + "\n" for r in recs))
        out = self.run_line("RES")
        self.assertTrue(out["ok"], out)
        rows = {r["id"]: r for r in out["data"]["rows"]}
        self.assertEqual((rows["F1b-w5-test"]["verdict"], rows["F1b-w5-test"]["perm_p"]), ("FAIL", None))
        self.assertEqual(rows["F1c-m21-test"]["verdict"], "not run")
        self.assertIn("Forecast tests", out["data"]["note"])


class ForecastJobTest(TempHome):
    def test_target_argv_and_validation(self):
        exe = sys.executable
        self.assertEqual(jobs.forecast_target("aapl"), "FCST AAPL")
        self.assertEqual(jobs.validate("fcst brk.b"), "FCST BRK.B")
        self.assertEqual(jobs.argv_for("FCST AAPL"), [exe, "-m", "brain", "forecast", "AAPL"])
        self.assertEqual((jobs.validate("FCST"), jobs.kind_of("FCST")), ("FCST", "ingest"))  # the ticker FCST
        self.assertEqual((jobs.kind_of("FCST AAPL"), jobs.ticker_of("FCST AAPL"), jobs.is_ticker("FCST AAPL")),
                         ("forecast", "AAPL", False))
        for t in ("FCST -X", "FCST A;B", "FCST  AAPL", "FCST ", "FCST A B", "FCST AAPL\n"):
            with self.subTest(t=t), self.assertRaises(jobs.JobError):
                jobs.validate(t)

    def test_its_own_slot_one_per_ticker(self):
        gate = Path(self._tmp.name) / "go"
        wait = [sys.executable, "-c", f"import os, time\nwhile not os.path.exists({str(gate)!r}): time.sleep(0.02)"]
        m = jobs.JobManager(argv_for=lambda t: wait)
        try:
            ing = [m.submit(t) for t in ("A", "B")]
            dec = m.submit(jobs.decide_target("A"))
            fc = [m.submit(jobs.forecast_target(t)) for t in ("A", "B")]   # one forecast slot: B waits
            self.assertEqual([j["status"] for j in ing], ["running", "running"])
            self.assertEqual(dec["status"], "running")  # a decision never waits behind a forecast
            self.assertEqual([(j["status"], j["kind"]) for j in fc], [("running", "forecast"), ("queued", "forecast")])
            self.assertEqual(m.submit("fcst a")["id"], fc[0]["id"])  # de-duplicated while queued or running
            self.assertEqual(m.summary()["active"], ["A", "B", "DEC A", "FCST A", "FCST B"])
            gate.touch()
            self.assertTrue(m.wait(30))
            self.assertEqual({j["kind"] for j in m.list_jobs()}, {"ingest", "decide", "forecast"})
        finally:
            gate.touch()
            m.shutdown()

    def test_forecasts_wait_their_turn_without_holding_up_ingest(self):
        order, lock = [], threading.Lock()
        gate = Path(self._tmp.name) / "release"

        def argv(t):
            return [sys.executable, "-c", "import os, time\n" + (f"while not os.path.exists({str(gate)!r}): time.sleep(0.02)\n"
                                                                if t.startswith("FCST") else "")]

        def finished(job):
            with lock:
                order.append(job["target"])
        m = jobs.JobManager(on_finish=finished, argv_for=argv)
        try:
            m.submit_many(["FCST A", "FCST B"])
            m.submit_many(["X", "Y", "DEC Z"])
            end = time.monotonic() + 30
            while len(order) < 3 and time.monotonic() < end:
                time.sleep(0.02)
            self.assertEqual(sorted(order), ["DEC Z", "X", "Y"])
            gate.touch()
            self.assertTrue(m.wait(30))
            self.assertEqual(order[3:], ["FCST A", "FCST B"])
        finally:
            gate.touch()
            m.shutdown()

    def test_the_jobs_table_shows_a_forecast_job(self):
        t = jobs.table([{"id": "j2", "target": "FCST AAPL", "kind": "forecast", "status": "running", "started": "s",
                         "finished": None, "returncode": None, "log": ["training F1 (h=5)"]}])
        row = t["rows"][0]
        self.assertEqual((row["_ticker"], row["_cmds"]), ("AAPL", {"target": "AAPL FCST"}))
        self.assertIn("1 forecast and 1 decision at once", t["note"])
        self.assertIn("AAPL FCST NEW", jobs.table([])["note"])
        odd = jobs.table([{"id": "j3", "target": "FCST ASK", "kind": "forecast", "status": "done", "started": "s",
                           "finished": "f", "returncode": 0, "log": []}])["rows"][0]
        self.assertNotIn("_cmds", odd)


class FcstServerTest(FcstBase):
    def setUp(self):
        super().setUp()
        self.server, _ = server.make_server("127.0.0.1", 0, bank=self.bank)
        self.server.quiet = True
        self.server.jobs.argv_for = fake_argv

    def tearDown(self):
        self.assertTrue(self.server.jobs.wait(20))
        self.server.server_close()
        super().tearDown()

    def test_fcst_new_starts_a_forecast_job_and_answers_with_jobs(self):
        out = self.server.run_line("ccc fcst new")
        self.assertTrue(out["ok"], out)
        self.assertEqual((out["fn"], out["ticker"], out["args"], out["view"]), ("FCST", "CCC", ["NEW"], "table"))
        self.assertTrue(out["title"].startswith("FCST NEW CCC: job j1 "), out["title"])
        self.assertIn("CCC FCST shows the forecast", out["title"])
        again = self.server.run_line("FCST NEW CCC")
        self.assertTrue(again["title"].startswith("FCST NEW CCC: job j1 "), again["title"])
        self.assertTrue(self.server.jobs.wait(20))
        job = self.server.jobs.list_jobs()[0]
        self.assertEqual((job["target"], job["kind"], job["status"]), ("FCST CCC", "forecast", "done"))
        self.assertEqual(len(self.server.jobs.list_jobs()), 1)
        self.assertEqual(self.server.run_line("JOBS")["data"]["rows"][0]["_cmds"], {"target": "CCC FCST"})

    def test_the_real_argv(self):
        self.server.jobs.argv_for = jobs.argv_for
        with mock.patch.object(jobs.JobManager, "_follow", lambda m, job: (job.log.append(" ".join(job.argv[1:])), 0)[1]):
            self.assertTrue(self.server.run_line("MSFT FCST NEW")["ok"])
            self.assertTrue(self.server.jobs.wait(20))
        self.assertEqual(self.server.jobs.list_jobs()[0]["log"], ["-m brain forecast MSFT"])

    def test_refusals(self):
        self.assertEqual(self.server.run_line("FCST NEW")["error"], "FCST NEW needs a ticker, e.g. AAPL FCST NEW")
        self.assertIn("benchmark", self.server.run_line("SPY FCST NEW")["error"])
        self.assertIn("market symbol", self.server.run_line("^GSPC FCST NEW")["error"])
        out = self.server.run_line("AAA FCST NEW", read_only=True)
        self.assertEqual((out["ok"], out["error"]), (False, "FCST NEW is not available here (read-only)"))
        self.assertEqual(self.server.jobs.list_jobs(), [])
        self.assertTrue(self.server.run_line("AAA FCST", read_only=True)["ok"])  # reading is allowed

    def test_fcst_says_when_the_last_job_is_at_work_failed_or_recorded(self):
        gate = Path(self._tmp.name) / "go"
        self.server.jobs.argv_for = lambda t: [
            sys.executable, "-c", f"import os, sys, time\nwhile not os.path.exists({str(gate)!r}): time.sleep(0.02)\n"
            "print('training F1 (h=5)')\nprint('brain forecast: no forecast for CCC on 2026-10-15: no prices (INGEST CCC)', "
            "file=sys.stderr)\nsys.exit(1)"]
        try:
            self.assertTrue(self.server.run_line("AAA FCST NEW")["ok"])
            self.assertTrue(self.server.run_line("CCC FCST NEW")["ok"])  # one forecast at a time: CCC waits
            out = self.server.run_line("AAA FCST")
            self.assertEqual(out["data"]["job_note"],
                             "a new forecast on AAA is running (job j1); AAA FCST shows it when it is done")
            out = self.server.run_line("CCC FCST")
            self.assertEqual((out["ok"], out["hint"]), (False, "JOBS"))
            self.assertEqual(out["error"], "no forecast recorded for CCC yet; a new forecast on CCC is queued (job j2); "
                                           "CCC FCST shows it when it is done")
        finally:
            gate.touch()
        self.assertTrue(self.server.jobs.wait(20))
        out = self.server.run_line("CCC FCST")
        self.assertEqual(out["error"], "no forecast recorded for CCC yet; the last CCC FCST NEW (job j2, failed) recorded "
                                       "nothing: no forecast for CCC on 2026-10-15: no prices (INGEST CCC)")
        self.assertIn("recorded nothing", self.server.run_line("AAA FCST")["data"]["job_note"])
        with mock.patch.object(live, "now_iso", lambda: now_iso()):
            forecast_fact(self.bank, "AAA", len(DAYS) - 2, 0.57, made=now_iso())  # one a job records: no note
        out = self.server.run_line("AAA FCST")
        self.assertEqual(out["data"]["forecasts"][0]["p_beat"], 0.57)
        self.assertNotIn("job_note", out["data"])

    def test_the_analyst_reads_forecasts_and_cannot_start_one(self):
        from brain.terminal import analyst
        reader = lambda line: self.server.run_line(line, read_only=True)  # noqa: E731
        calls = []
        tools = {t.name: t for t in analyst._tools(reader, calls)}
        self.assertIn("Quote the bands and the chance as recorded", tools["forecast"].spec()["function"]["description"])
        out = json.loads(tools["forecast"].call({"ticker": "aaa"}))
        self.assertEqual((out["ok"], out["view"], out["data"]["forecasts"][0]["p_beat"]), (True, "forecast", 0.55))
        self.assertEqual(calls, [{"name": "forecast", "input": "AAA FCST"}])
        with self.assertRaises(analyst.ToolError):
            tools["forecast"].call({"ticker": "AAA FCST NEW"})
        self.assertFalse(reader("AAA FCST NEW")["ok"])
        self.assertFalse(analyst.bank_reader(self.bank)("AAA FCST NEW")["ok"])
        self.assertEqual(self.server.jobs.list_jobs(), [])


class FcstHttpTest(TempHome):
    def test_api_ingest_refuses_a_forecast_target(self):
        import urllib.error
        import urllib.request
        srv, token = server.make_server("127.0.0.1", 0, bank=self.bank)
        srv.quiet = True
        srv.jobs.argv_for = fake_argv
        th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        th.start()
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for target, code in (("FCST AAPL", 400), ("fcst aapl", 400), ("FCST", 200)):
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
            self.assertEqual([j["target"] for j in srv.jobs.list_jobs()], ["FCST"])  # the ticker FCST, ingested
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(5)


if __name__ == "__main__":
    unittest.main()
