import tempfile
import unittest
from pathlib import Path
from unittest import mock

from brain import config
from brain.research import ledger, run


def result(hid, p, net=0.01):
    return {"op": "result", "id": hid, "spec_sha": "x", "test": {"perm_p": p, "mean_net_excess": net},
            "controls": {"pass": True}}


class VerdictsTest(unittest.TestCase):
    def test_withdrawn_hypotheses_still_count_in_the_correction(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.jsonl"
            with mock.patch.object(config, "LEDGER", p):
                ledger.append(result("A", 0.02))
                ledger.append(result("B", 0.04))
                alone = {r["id"]: r for r in run.verdicts()}
                self.assertAlmostEqual(alone["A"]["bh_q"], 0.04)
                ledger.append(result("C", 0.9))
                ledger.withdraw("C", "bug in signal")
                rows = {r["id"]: r for r in run.verdicts()}
                # C is withdrawn but was tried: A's q uses 3 tests, not 2.
                self.assertAlmostEqual(rows["A"]["bh_q"], 0.06)
                self.assertTrue(rows["C"]["verdict"].startswith("WITHDRAWN"))
                self.assertEqual(rows["A"]["verdict"], "PASS")
