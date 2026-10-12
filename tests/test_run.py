import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from brain import config
from brain.research import ledger, run
from tests.util import TempHome, filed, price_snapshot


class RunEndToEndTest(TempHome):
    def test_declared_hypotheses_run_and_report(self):
        home = Path(self._tmp.name)
        days = pd.bdate_range("2009-01-01", "2021-06-30").strftime("%Y-%m-%d").tolist()
        rng = np.random.default_rng(3)
        tickers = [f"T{i}" for i in range(9)]
        for t in [*tickers, "SPY"]:
            px = list(50 * np.cumprod(1 + rng.normal(0.0003, 0.015, len(days))))
            self.bank.add(price_snapshot(t, days, px))
        for i, t in enumerate(tickers):
            for y in range(2009, 2021):
                self.bank.add_many([
                    filed(t, "us-gaap:NetIncomeLoss", 10.0 + i + y % 3, f"{y}-12-31", f"{y + 1}-02-20",
                          form="10-K", start=f"{y}-01-01", duration_days=364),
                    filed(t, "us-gaap:Assets", 100.0 * (1 + 0.05 * i) ** (y - 2008), f"{y}-12-31", f"{y + 1}-02-20", form="10-K"),
                    filed(t, "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding", 10.0, f"{y}-12-31",
                          f"{y + 1}-02-20", form="10-K", start=f"{y}-01-01", duration_days=364),
                ])
        uni = home / "universe.json"
        uni.write_text(json.dumps({"tickers": tickers, "rule_short": "synthetic"}))
        with mock.patch.object(config, "LEDGER", home / "ledger.jsonl"), \
             mock.patch.object(config, "RESULTS", home / "RESULTS.md"), \
             mock.patch.object(config, "UNIVERSE", uni), \
             mock.patch.dict("brain.research.hypotheses.GATE", {"n_perm": 200}):
            with self.assertRaises(SystemExit):  # undeclared: refused
                run.run(self.bank, n_random=5)
            ledger.declare_all()
            out = run.run(self.bank, n_random=20)
            self.assertEqual(len(out), 5)
            for r in out:
                self.assertGreater(r["test"]["n_months"], 20)
                self.assertGreater(r["test_signal_coverage"], 0.5)
            text = (home / "RESULTS.md").read_text()
            self.assertIn("H5-ridge", text)
            self.assertEqual(len(run.verdicts()), 5)
