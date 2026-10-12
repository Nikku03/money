import json

from brain import check
from brain.bank import Bank, Fact, LookAheadError
from tests.util import TempHome, filed


class BankTest(TempHome):
    def test_id_ignores_retrieval_time(self):
        a = filed("aapl", "x", 1, "2020-01-01", "2020-02-01")
        b = Fact(**{**a.__dict__, "retrieved_at": "2026-01-01T00:00:00Z"})
        self.assertEqual(a.id, b.id)
        self.assertEqual(a.entity, "AAPL")
        self.assertEqual(a.known_at, "2020-02-01T23:59:59Z")

    def test_duplicate_is_not_stored_twice(self):
        f = filed("A", "x", 1, "2020-01-01", "2020-02-01")
        self.assertTrue(self.bank.add(f))
        self.assertFalse(self.bank.add(f))
        self.assertEqual(len(self.bank.facts("A")), 1)

    def test_view_hides_the_future_and_restatements_until_known(self):
        orig = filed("A", "ni", 100, "2019-12-31", "2020-02-15", form="10-K")
        restated = filed("A", "ni", 90, "2019-12-31", "2021-02-15", form="10-K")
        self.bank.add_many([orig, restated])
        self.assertIsNone(self.bank.view("2020-02-14").latest("A", "ni"))
        self.assertEqual(self.bank.view("2020-02-15").latest("A", "ni").value, 100)
        self.assertEqual(self.bank.view("2021-01-01").latest("A", "ni").value, 100)
        self.assertEqual(self.bank.view("2021-02-16").latest("A", "ni").value, 90)

    def test_require_not_after(self):
        v = self.bank.view("2020-01-01")
        v.require_not_after("2020-01-01")
        with self.assertRaises(LookAheadError):
            v.require_not_after("2020-01-02")

    def test_replay_and_downgrade_persist(self):
        f = filed("A", "x", 1, "2020-01-01", "2020-02-01")
        g = filed("A", "x", 2, "2020-03-01", "2020-04-01")
        self.bank.add_many([f, g])
        self.bank.downgrade("A", g.id, "contradicted by measurement")
        self.assertEqual(self.bank.view().latest("A", "x").value, 1)
        fresh = Bank(self.bank.root)
        self.assertEqual(fresh.digest("A"), self.bank.digest("A"))
        self.assertTrue(fresh.is_downgraded("A", g.id))
        self.assertEqual(len(fresh.facts("A", include_downgraded=True)), 2)

    def test_check_catches_tampering_and_time_travel(self):
        good = filed("A", "x", 1, "2020-01-01", "2020-02-01")
        early = filed("A", "y", 1, "2020-06-30", "2020-06-01")  # known before the period ended
        self.bank.add_many([good, early])
        derived = Fact(entity="A", attr="derived.z", value=2.0, as_of="2020-02-01", known_at="2020-01-15",
                       tier="derived", source={"name": "t"}, inputs=(good.ref,))
        self.bank.add(derived)
        errors, _ = check.run(Bank(self.bank.root))
        self.assertTrue(any("before as_of" in e for e in errors))
        self.assertTrue(any("known before its input" in e for e in errors))

        p = self.bank.shard("A")
        lines = p.read_text().splitlines()
        rec = json.loads(lines[0])
        rec["fact"]["value"] = 999
        p.write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n")
        errors, _ = check.run(Bank(self.bank.root))
        self.assertTrue(any("stored id" in e for e in errors))

    def test_check_passes_clean_bank(self):
        f = filed("A", "x", 1, "2020-01-01", "2020-02-01")
        self.bank.add(f)
        d = Fact(entity="A", attr="derived.z", value=2.0, as_of="2020-02-01", known_at="2020-02-02",
                 tier="derived", source={"name": "t"}, inputs=(f.ref,))
        self.bank.add(d)
        errors, _ = check.run(Bank(self.bank.root))
        self.assertEqual(errors, [])
