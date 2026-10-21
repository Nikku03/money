from pathlib import Path

from brain import catalysts
from tests.util import TempHome

BASE = {"ticker": "ACME", "event": "Phase 3 at a congress", "event_date": "2999-01-01",
        "question": "OS hazard ratio at or below 0.75"}


class CatalystsTest(TempHome):
    def setUp(self):
        super().setUp()
        self.path = Path(self._tmp.name) / "calls.jsonl"

    def test_call_resolve_score(self):
        catalysts.call({**BASE, "id": "a", "p": 0.9}, self.path)
        catalysts.call({**BASE, "id": "b", "p": 0.2}, self.path)
        self.assertEqual(catalysts.score(self.path), {"n": 0})
        catalysts.resolve("a", True, "https://example.test/a", path=self.path)
        catalysts.resolve("b", True, "https://example.test/b", path=self.path)
        s = catalysts.score(self.path)
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["brier"], (0.01 + 0.64) / 2)

    def test_calls_are_never_rewritten(self):
        catalysts.call({**BASE, "id": "a", "p": 0.5}, self.path)
        with self.assertRaisesRegex(ValueError, "exists"):
            catalysts.call({**BASE, "id": "a", "p": 0.6}, self.path)
        catalysts.resolve("a", False, "https://example.test/a", path=self.path)
        with self.assertRaisesRegex(ValueError, "settled"):
            catalysts.resolve("a", True, "https://example.test/a", path=self.path)

    def test_a_call_comes_before_its_event_and_an_outcome_cites_a_page(self):
        with self.assertRaisesRegex(ValueError, "already past"):
            catalysts.call({**BASE, "id": "late", "p": 0.5, "event_date": "2000-01-01"}, self.path)
        with self.assertRaisesRegex(ValueError, "probability"):
            catalysts.call({**BASE, "id": "x", "p": 1.5}, self.path)
        catalysts.call({**BASE, "id": "a", "p": 0.5}, self.path)
        with self.assertRaisesRegex(ValueError, "http"):
            catalysts.resolve("a", True, "my memory", path=self.path)

    def test_a_voided_call_is_kept_but_not_scored(self):
        catalysts.call({**BASE, "id": "a", "p": 0.5}, self.path)
        catalysts.void("a", "threshold not reported at the presentation", self.path)
        self.assertEqual(catalysts.state(self.path)["a"]["void"], "threshold not reported at the presentation")
        self.assertEqual(catalysts.score(self.path), {"n": 0})
