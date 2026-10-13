"""The lazy, indexed bank: reads one attribute at a time, shares a journal across processes,
and treats its sidecar index as a cache that can always be rebuilt."""
import json
import multiprocessing as mp
from unittest import mock

from brain import check
from brain.bank import Bank, Fact
from tests.util import TempHome, filed


def _writer(root, tag, n):
    b = Bank(root)
    for i in range(n):
        b.add(Fact(entity="SHARED", attr=f"x.{tag}", value=i, as_of="2020-01-01", known_at="2020-02-01",
                   tier="filed", source={"name": "t", "url": "https://example.test"}))


class LazyBankTest(TempHome):
    def setUp(self):
        super().setUp()
        self.bank.add_many([filed("A", "price", i, f"2020-01-{i + 1:02d}", f"2020-01-{i + 1:02d}") for i in range(5)]
                           + [filed("A", "filing", "10-K", "2020-02-01", "2020-02-02")])

    def test_reads_only_the_attribute_asked_for(self):
        b = Bank(self.bank.root)
        self.assertEqual(len(b.facts("A", "filing")), 1)
        self.assertEqual({f.attr for f in b._facts["A"].values()}, {"filing"})
        self.assertEqual(b.attrs("A"), ["filing", "price"])  # known from the index without reading facts
        self.assertEqual(len(b.view().facts("A", "price")), 5)

    def test_index_is_reused_and_only_appended_bytes_are_scanned(self):
        Bank(self.bank.root).facts("A", "price")
        with mock.patch.object(Bank, "_scan") as scan:
            self.assertEqual(len(Bank(self.bank.root).facts("A", "price")), 5)
            scan.assert_not_called()
        reader = Bank(self.bank.root)
        reader.facts("A", "price")
        Bank(self.bank.root).add(filed("A", "price", 99, "2020-03-01", "2020-03-01"))  # another process
        reader.refresh()
        self.assertEqual(len(reader.facts("A", "price")), 6)

    def test_rewritten_journal_is_noticed_and_the_index_rebuilt(self):
        p = self.bank.shard("A")
        Bank(self.bank.root).facts("A", "price")
        p.write_text(p.read_text().replace('"value":0', '"value":1000'))  # same length? no: rewritten in place
        b = Bank(self.bank.root)
        self.assertEqual(sorted(f.value for f in b.facts("A", "price"))[-1], 1000)
        errors, _ = check.run(Bank(self.bank.root))
        self.assertTrue(any("stored id" in e for e in errors))  # the edit is still caught as tampering

    def test_wrong_offsets_heal_by_rebuilding(self):
        Bank(self.bank.root).facts("A", "price")
        ip = self.bank.shard("A").with_suffix(".jsonl.idx")
        d = json.loads(ip.read_text())
        d["attrs"]["price"], d["attrs"]["filing"] = d["attrs"]["filing"], d["attrs"]["price"]
        ip.write_text(json.dumps(d))
        b = Bank(self.bank.root)
        self.assertEqual(len(b.facts("A", "price")), 5)
        self.assertEqual(len(b.facts("A", "filing")), 1)
        self.assertEqual(b.journal_errors, [])

    def test_reingest_does_not_duplicate_lines_for_unread_attributes(self):
        p = self.bank.shard("A")
        before = p.read_text().count("\n")
        b = Bank(self.bank.root)  # has read nothing yet
        self.assertEqual(b.add_many([filed("A", "price", i, f"2020-01-{i + 1:02d}", f"2020-01-{i + 1:02d}")
                                     for i in range(5)]), 0)
        self.assertEqual(p.read_text().count("\n"), before)

    def test_a_torn_last_line_stays_alone(self):
        p = self.bank.shard("A")
        p.write_bytes(p.read_bytes() + b'{"fact":{"as_of":"2020')  # a writer died mid-line
        b = Bank(self.bank.root)
        b.add(filed("A", "price", 7, "2020-04-01", "2020-04-01"))
        fresh = Bank(self.bank.root)
        self.assertEqual(len(fresh.facts("A", "price")), 6)
        self.assertTrue(any("unparseable" in e for e in fresh.journal_errors))

    def test_concurrent_writers_from_two_processes(self):
        ctx = mp.get_context("fork")
        ps = [ctx.Process(target=_writer, args=(str(self.bank.root), t, 150)) for t in ("a", "b")]
        for q in ps:
            q.start()
        for q in ps:
            q.join(60)
            self.assertEqual(q.exitcode, 0)
        fresh = Bank(self.bank.root)
        self.assertEqual((len(fresh.facts("SHARED", "x.a")), len(fresh.facts("SHARED", "x.b"))), (150, 150))
        self.assertEqual(fresh.journal_errors, [])
        lines = self.bank.shard("SHARED").read_text().splitlines()
        self.assertEqual(len(lines), 300)
        self.assertTrue(all(json.loads(line)["op"] == "add" for line in lines))
        scratch = Bank(self.bank.root)  # an index rebuilt from scratch agrees with the shared one
        self.bank.shard("SHARED").with_suffix(".jsonl.idx").unlink()
        self.assertEqual(scratch.digest("SHARED"), fresh.digest("SHARED"))

    def test_find_by_id_prefix_reads_one_attribute(self):
        f = self.bank.facts("A", "filing")[0]
        b = Bank(self.bank.root)
        self.assertEqual([x.id for x in b.find("A", f.id[:8])], [f.id])
        self.assertEqual({x.attr for x in b._facts["A"].values()}, {"filing"})
        self.assertEqual(b.get(f.ref).value, "10-K")
        self.assertIsNone(b.get("A/" + "0" * 20))


class CheckIndexTest(TempHome):
    def test_check_catches_an_index_that_hides_a_record(self):
        self.bank.add_many([filed("A", "price", i, f"2020-01-{i + 1:02d}", f"2020-01-{i + 1:02d}") for i in range(3)])
        Bank(self.bank.root).facts("A", "price")
        self.assertEqual(check.run(Bank(self.bank.root))[0], [])
        ip = self.bank.shard("A").with_suffix(".jsonl.idx")
        d = json.loads(ip.read_text())
        d["attrs"]["price"] = d["attrs"]["price"][:-3]  # one record silently dropped from the index
        ip.write_text(json.dumps(d))
        errors, _ = check.run(Bank(self.bank.root))
        self.assertTrue(any("index disagrees" in e for e in errors), errors)
