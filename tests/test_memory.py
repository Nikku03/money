"""Memory between sessions (brain/memory.py): the agent's claims leave the bank and come back after a reset."""
from __future__ import annotations

import os
import tempfile

from brain import memory
from brain.bank import Bank
from tests.util import TempHome, filed


def claim(t, text, day):
    return filed(t, "claim", text, day, day + "T12:00:00Z", tier="reported", direction="none")


class MemoryTest(TempHome):
    def setUp(self):
        super().setUp()
        self._mem = tempfile.TemporaryDirectory()
        os.environ["BRAIN_MEMORY"] = self._mem.name

    def tearDown(self):
        os.environ.pop("BRAIN_MEMORY", None)
        self._mem.cleanup()
        super().tearDown()

    def test_claims_survive_a_reset_of_the_home(self):
        self.bank.add_many([claim("INBX", "a $150m at-the-market agreement", "2026-11-13"),
                            claim("MOD", "talks over the cooling division", "2026-11-14"),
                            filed("MOD", "us-gaap:Assets", 1.0, "2026-09-30", "2026-11-01T00:00:00Z")])
        self.assertEqual(memory.export_claims(self.bank), 2)            # accounts can be read again; claims cannot
        lines = (memory.memory_dir() / memory.CLAIMS).read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("INBX", lines[0])                                  # oldest first
        with tempfile.TemporaryDirectory() as fresh:                     # the machine was reset
            os.environ["MONEY_HOME"] = fresh
            b = Bank()
            self.assertEqual(memory.claims(b), [])
            self.assertEqual(memory.import_claims(b), 2)
            self.assertEqual(memory.import_claims(b), 0)                 # not added twice
            got = memory.claims(Bank())
            self.assertEqual([(f.entity, f.value) for f in got],
                             [("INBX", "a $150m at-the-market agreement"), ("MOD", "talks over the cooling division")])
            os.environ["MONEY_HOME"] = self._tmp.name

    def test_status_says_when_the_home_is_empty_and_files_carry_their_hash(self):
        st = memory.status(self.bank)
        self.assertTrue(st["empty"])
        self.assertIsNone(st["spy_last"])
        self.assertEqual(memory.import_claims(self.bank), 0)              # nothing saved yet: nothing to load
        (memory.memory_dir() / "MEMORY.md").write_text("where things stand\n")
        f = memory.files()
        self.assertEqual([x["name"] for x in f], ["MEMORY.md"])
        self.assertEqual(len(f[0]["sha256"]), 64)
        self.assertTrue(f[0]["path"].endswith("MEMORY.md"))
