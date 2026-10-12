import tempfile
import unittest
from pathlib import Path

from brain.research import ledger


class LedgerTest(unittest.TestCase):
    def test_declared_specs_are_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.jsonl"
            orig = ledger.HYPOTHESES
            try:
                ledger.HYPOTHESES = [{"id": "Hx", "signal": "rev1", "claim": "a"}]
                self.assertEqual(len(ledger.declare_all(p)), 1)
                self.assertEqual(ledger.declare_all(p), [])
                ledger.HYPOTHESES = [{"id": "Hx", "signal": "rev1", "claim": "changed after seeing data"}]
                with self.assertRaises(SystemExit):
                    ledger.declare_all(p)
            finally:
                ledger.HYPOTHESES = orig
            ledger.withdraw("Hx", "test", p)
            self.assertEqual(ledger.withdrawn(p), {"Hx": "test"})
