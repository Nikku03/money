"""The agent's call record (brain/predict.py): recording, judging against later closes, paper trades,
withdrawals, the score and the CLI. Prices are test snapshots; the record goes to a scratch file."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest import mock

from brain import cli, config, predict
from tests.util import TempHome, price_snapshot

MADE = "2025-10-26T15:00:00Z"


class PredictTest(TempHome):
    def setUp(self):
        super().setUp()
        self.path = Path(tempfile.mkdtemp()) / "predictions.jsonl"
        self.bank.add_many([price_snapshot("PYPL", ["2025-10-22", "2025-10-23", "2025-10-26"], [57.5, 57.0, 54.6])])

    def later(self, closes):
        dates = ["2025-10-22", "2025-10-23", "2025-10-26", "2025-10-27", "2025-10-28"][: 3 + len(closes)]
        self.bank.add_many([price_snapshot("PYPL", dates, [57.5, 57.0, 54.6, *closes])])

    def add(self, call="down", prob=0.6, by="2025-10-27", **kw):
        return predict.add(self.bank, "pypl", call, prob, by, "thesis", made_at=MADE, path=self.path, **kw)

    def test_start_price_is_the_banks_latest_close(self):
        rec = self.add()
        self.assertEqual((rec["ticker"], rec["start_price"], rec["start_date"]), ("PYPL", 54.6, "2025-10-26"))
        self.assertEqual(predict.read(self.path)[0]["id"], rec["id"])

    def test_refuses_bad_input(self):
        for kw in ({"call": "sideways"}, {"prob": 1.0}, {"prob": 0}, {"by": "2025-10-25"},
                   {"by": "2025-10-26"},  # that close is already in the bank: the outcome is known
                   {"call": "move"}, {"threshold": -1}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.add(**kw)

    def test_a_call_judged_today_is_fine_before_todays_close(self):
        rec = predict.add(self.bank, "PYPL", "down", 0.55, "2025-10-27", "t", made_at="2025-10-27T09:00:00Z",
                          path=self.path)
        self.assertEqual((rec["start_date"], rec["start_price"]), ("2025-10-26", 54.6))
        with self.assertRaises(ValueError):   # no closes and no price given
            predict.add(self.bank, "ZZZZ", "up", 0.6, "2025-10-27", "t", made_at=MADE, path=self.path)

    def test_open_until_the_judging_close_is_in_the_bank(self):
        self.add()
        self.assertEqual(predict.state(self.bank, self.path)[0]["status"], "open")

    def test_down_call_hit_and_brier(self):
        self.add(prob=0.6)
        self.later([51.0])
        r = predict.state(self.bank, self.path)[0]
        self.assertEqual(r["status"], "hit")
        self.assertAlmostEqual(r["move_pct"], (51.0 / 54.6 - 1) * 100, places=2)
        self.assertAlmostEqual(r["brier"], 0.16)

    def test_threshold_and_move_calls(self):
        self.add(call="down", threshold=7.0)             # needs 50.78 or lower: misses at 51.0
        self.add(call="move", prob=0.4, threshold=5.0)   # 6.6% either way: hits
        self.add(call="up", prob=0.3)                    # fell: misses
        self.later([51.0])
        got = [r["status"] for r in predict.state(self.bank, self.path)]
        self.assertEqual(got, ["miss", "hit", "miss"])

    def test_judged_on_the_close_of_its_date_not_a_later_one(self):
        self.add(by="2025-10-27")
        self.later([56.0, 40.0])      # up on the 27th, crash on the 28th
        self.assertEqual(predict.state(self.bank, self.path)[0]["status"], "miss")

    def test_paper_trade_close_and_score(self):
        rec = self.add(trade={"desc": "PYPL 2025-10-31 52 put", "paid": 0.82})
        with self.assertRaises(ValueError):
            predict.close("ffffffff", 1.0, path=self.path)
        predict.close(rec["id"][:4], 1.5, "sold at the open", path=self.path)
        with self.assertRaises(ValueError):
            predict.close(rec["id"], 1.6, path=self.path)          # closed once
        self.later([51.0])
        rows = predict.state(self.bank, self.path)
        self.assertAlmostEqual(rows[0]["trade_pnl"], 68.0)
        s = predict.score(rows)
        self.assertEqual((s["judged"], s["hits"], s["trades_closed"], s["trade_wins"]), (1, 1, 1, 1))
        self.assertAlmostEqual(s["trade_pnl"], 68.0)
        self.assertFalse(s["enough"])

    def test_withdrawn_calls_leave_the_score_but_stay_on_file(self):
        rec = self.add()
        with self.assertRaises(ValueError):
            predict.withdraw(rec["id"], " ", path=self.path)
        predict.withdraw(rec["id"], "wrong ticker", path=self.path)
        self.later([51.0])
        rows = predict.state(self.bank, self.path)
        self.assertEqual(rows[0]["status"], "withdrawn")
        s = predict.score(rows)
        self.assertEqual((s["withdrawn"], s["judged"]), (1, 0))
        self.assertEqual([r["type"] for r in predict.read(self.path)], ["call", "withdraw"])

    def test_the_same_call_twice_is_refused(self):
        self.add()
        with self.assertRaises(ValueError):
            self.add()

    def test_cli_add_list_score(self):
        out = io.StringIO()
        with mock.patch.object(config, "PREDICTIONS", self.path), contextlib.redirect_stdout(out):
            cli.main(["predict", "add", "PYPL", "down", "--prob", "0.55", "--by", "2099-01-02",
                      "--thesis", "checkout growth stalls", "--trade", "PYPL 52 put", "--paid", "0.82"])
            cli.main(["predict", "list"])
            cli.main(["predict", "score"])
        text = out.getvalue()
        self.assertIn("Recorded", text)
        self.assertIn("checkout growth stalls", text)
        self.assertIn("this record says nothing yet", text)
        self.assertEqual(json.loads(self.path.read_text().splitlines()[0])["prob"], 0.55)


if __name__ == "__main__":
    import unittest
    unittest.main()
