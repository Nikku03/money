"""Memory between sessions (brain/memory.py): the agent's claims leave the bank and come back after a reset."""
from __future__ import annotations

import json
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


def line(kind, uid, at, content, **extra):
    return {"type": kind, "uuid": uid, "timestamp": at, "sessionId": "s1", "message": {"content": content}, **extra}


class TalkTest(MemoryTest):
    def setUp(self):
        super().setUp()
        self._tx = tempfile.TemporaryDirectory()
        self.tx = os.path.join(self._tx.name, "session.jsonl")
        os.environ["BRAIN_TRANSCRIPTS"] = os.path.join(self._tx.name, "*.jsonl")

    def tearDown(self):
        os.environ.pop("BRAIN_TRANSCRIPTS", None)
        self._tx.cleanup()
        super().tearDown()

    def write(self, lines, mode="w"):
        with open(self.tx, mode) as fh:
            fh.writelines(json.dumps(x) + "\n" for x in lines)

    def test_the_conversation_is_kept_with_its_kinds_and_without_emails_or_keys(self):
        self.write([
            line("user", "u0", "2026-11-10T16:45:00Z", "Summary: the scan W1 passed its test.", isCompactSummary=True),
            line("user", "u1", "2026-11-14T16:26:00Z", "make an all around mosaic. mail me at someone@example.com",
                 origin={"kind": "human"}),
            line("user", "u2", "2026-11-14T16:26:01Z", "<system-reminder>noise</system-reminder>", isMeta=True),
            line("assistant", "a1", "2026-11-14T16:27:00Z", [
                {"type": "thinking", "thinking": "Three pictures: share sale, late accounts, cuts."},
                {"type": "text", "text": "Building the mosaic now."},
                {"type": "tool_use", "id": "t1", "name": "Bash",
                 "input": {"command": "export KEY=1; curl 'https://x.test/?api_key=abc123'", "description": "Fetch"}}]),
            line("user", "u3", "2026-11-14T16:27:05Z", [{"type": "tool_result", "tool_use_id": "t1",
                                                          "content": "x" * 5000}]),
        ])
        self.assertEqual(memory.capture(), 6)
        talk = memory.read_talk()
        self.assertEqual([c["kind"] for c in talk], ["summary", "you", "thought", "me", "tool", "result"])
        you = next(c for c in talk if c["kind"] == "you")
        self.assertIn("[email]", you["text"])
        self.assertNotIn("example.com", you["text"])
        tool = next(c for c in talk if c["kind"] == "tool")
        self.assertIn("api_key=[secret]", tool["text"])
        result = next(c for c in talk if c["kind"] == "result")
        self.assertLess(len(result["text"]), 2700)                      # head and tail of a long result
        self.assertIn("characters left out", result["text"])
        self.assertEqual([x["kind"] for x in memory.find(result["id"])], ["tool", "result"])   # a result and its call
        self.assertEqual(memory.capture(), 0)                            # nothing new: nothing kept twice
        self.write([line("user", "u4", "2026-11-15T20:08:00Z", "save it to my drive", origin={"kind": "human"})], "a")
        self.assertEqual(memory.capture(), 1)                            # only the new line is read

    def test_recall_finds_the_passage_and_leaves_its_own_calls_out(self):
        self.write([
            line("user", "u1", "2026-11-13T22:28:00Z", "our target is finding what only insiders know",
                 origin={"kind": "human"}),
            line("assistant", "a1", "2026-11-13T22:29:00Z", [{"type": "text", "text": "Prices moved on Thursday."}]),
            line("assistant", "a2", "2026-11-15T20:00:00Z", [{"type": "tool_use", "id": "t9", "name": "Bash", "input": {
                "command": "python -m brain memory recall insiders know target"}}]),
        ])
        (memory.memory_dir() / "MEMORY.md").write_text("# Memory\n\n## The person\n\nWants plain, short answers.\n")
        talk = {"you", "me", "tool", "result", "thought", "summary"}       # the repo's own docs left aside here
        hits = memory.recall("what did they say about insiders", kinds=talk)
        self.assertEqual(hits[0]["kind"], "you")
        self.assertTrue(all("brain memory" not in h["text"] for h in hits))
        prefers = memory.recall("what does the person prefer about explanations", kinds=talk | {"note"})
        self.assertEqual(prefers[0]["src"], str(memory.memory_dir() / "MEMORY.md"))
        self.assertEqual(memory.recall("insiders", since="2026-11-14"), [])
        self.assertEqual(memory.recall("the and of"), [])                # nothing but common words

    def test_a_kept_fact_is_recalled_and_the_talk_comes_back_from_its_packed_copy(self):
        memory.remember("Drive folder for memory is called money-brain memory", topic="drive")
        hit = memory.recall("which drive folder holds the memory", kinds={"note"})[0]
        self.assertEqual((hit["kind"], hit["head"]), ("note", "drive"))
        self.write([line("user", "u1", "2026-11-14T16:26:00Z", "make an all around mosaic",
                         origin={"kind": "human"})])
        memory.capture()
        packed = memory.pack_talk()
        os.remove(memory.talk_dir() / memory.TALK)                       # the machine was reset
        self.assertEqual(memory.read_talk(), [])
        self.assertEqual(memory.merge_talk(packed), 1)
        self.assertEqual(memory.merge_talk(packed), 0)
        self.assertEqual(memory.read_talk()[0]["text"], "make an all around mosaic")

    def test_markdown_is_cut_by_heading_and_carries_its_headings(self):
        text = "# Top\n\nintro\n\n## Part\n\n" + "\n\n".join(f"paragraph {i} " + "word " * 80 for i in range(12))
        parts = memory.split_markdown(text, "x.md", limit=600)
        self.assertEqual(parts[0]["head"], "Top")
        self.assertTrue(all(p["head"] == "Top > Part" for p in parts[1:]))
        self.assertGreater(len(parts), 3)
        self.assertTrue(all(p["text"].startswith(p["head"]) for p in parts))
