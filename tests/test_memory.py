"""Memory between sessions (brain/memory.py): the agent's claims leave the bank and come back after a reset."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

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
        memory._model = False                                            # words alone here; MeaningTest brings one

    def tearDown(self):
        memory._model = None
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
        self.assertEqual(memory.capture(), 8)                            # the long result is kept in 3 pieces
        talk = memory.read_talk()
        self.assertEqual(list(dict.fromkeys(c["kind"] for c in talk)), ["summary", "you", "thought", "me", "tool",
                                                                        "result"])
        you = next(c for c in talk if c["kind"] == "you")
        self.assertIn("[email]", you["text"])
        self.assertNotIn("example.com", you["text"])
        tool = next(c for c in talk if c["kind"] == "tool")
        self.assertIn("api_key=[secret]", tool["text"])
        result = next(c for c in talk if c["kind"] == "result")
        whole = memory.find(result["id"])                                # a result comes with its call
        self.assertEqual([x["kind"] for x in whole], ["tool", "result", "result", "result"])
        kept = "".join(x["text"] for x in whole[1:])
        self.assertLess(len(kept), 2700)                                 # head and tail of a long result
        self.assertIn("characters left out", kept)
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
        import base64  # Drive's download: JSON, base64 inside
        wrapped = memory.talk_dir() / "download.txt"
        wrapped.write_text(json.dumps({"id": "x", "title": "talk.jsonl.gz",
                                       "content": base64.b64encode(packed.read_bytes()).decode()}))
        os.remove(memory.talk_dir() / memory.TALK)
        self.assertEqual(memory.merge_talk(wrapped), 1)

    def test_markdown_is_cut_by_heading_and_carries_its_headings(self):
        text = "# Top\n\nintro\n\n## Part\n\n" + "\n\n".join(f"paragraph {i} " + "word " * 80 for i in range(12))
        parts = memory.split_markdown(text, "x.md", limit=600)
        self.assertEqual(parts[0]["head"], "Top")
        self.assertTrue(all(p["head"] == "Top > Part" for p in parts[1:]))
        self.assertGreater(len(parts), 3)
        self.assertTrue(all(p["text"].startswith(p["head"]) for p in parts))


class FakeModel:
    """Meaning, made up: each text is a count of the groups its words fall in, so 'wants' and 'prefer' are close."""
    GROUPS = (("prefer", "want", "wants", "like"), ("short", "plain", "simple", "brief"), ("bitcoin", "crypto"),
              ("mistake", "flaw", "wrong", "error"), ("measured", "counted", "read"))

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        low = text.lower()
        v = np.array([sum(low.count(w) for w in g) for g in self.GROUPS] + [0.01], float)
        return v / np.linalg.norm(v)

    def embed(self, texts, batch_size=32):
        texts = list(texts)
        self.calls += len(texts)
        return [self._vec(t) for t in texts]

    def query_embed(self, texts):
        return [self._vec(t) for t in texts]


class MeaningTest(TalkTest):
    def setUp(self):
        super().setUp()
        self.model = FakeModel()
        memory._model = self.model

    def tearDown(self):
        memory._model = None
        super().tearDown()

    def test_meaning_finds_what_shares_no_words_and_vectors_are_kept(self):
        self.write([
            line("assistant", "a1", "2026-11-14T04:47:00Z", [{"type": "text", "text":
                 "I found a flaw: bitcoin's -6% was measured from Thursday evening."}]),
            line("assistant", "a2", "2026-11-14T05:00:00Z", [{"type": "text", "text": "The scan covered 2450 stocks."}]),
        ])
        talk = {"me"}
        by_words = memory.recall("how was crypto counted", kinds=talk, meaning=False)
        self.assertEqual(by_words, [])                                   # not one word in common
        memory.vectors(memory.library())                                 # index: every passage embedded once
        hits = memory.recall("how was crypto counted", kinds=talk)
        self.assertEqual(hits[0]["text"][:15], "I found a flaw:")
        self.assertEqual(hits[0]["by"], "words and meaning")
        first = self.model.calls
        memory.recall("bitcoin", kinds=talk)
        self.assertEqual(self.model.calls, first)                        # nothing new: nothing embedded again
        self.write([line("user", "u9", "2026-11-15T20:00:00Z", "keep answers short", origin={"kind": "human"})], "a")
        memory.recall("bitcoin", kinds=talk)
        self.assertEqual(self.model.calls, first + 1)                    # only the new passage

    def test_without_a_model_recall_ranks_by_words(self):
        memory._model = False
        self.write([line("user", "u1", "2026-11-14T16:26:00Z", "make an all around mosaic", origin={"kind": "human"})])
        hits = memory.recall("mosaic", kinds={"you"})
        self.assertEqual((hits[0]["by"], hits[0]["text"]), ("words", "make an all around mosaic"))

    def test_fusion_puts_a_passage_high_in_both_rankings_first(self):
        a, b, c = ({"id": i, "kind": "me", "text": i} for i in "abc")
        got = [x["id"] for _, x in memory.fuse([(3, a), (2, b), (1, c)], [(0.9, b), (0.8, c), (0.7, a)], 3)]
        self.assertEqual(got, ["b", "a", "c"])

    def test_a_long_message_is_kept_in_pieces_and_shown_whole(self):
        long = "\n\n".join(f"paragraph {i}: " + "detail " * 60 for i in range(8))
        self.write([line("assistant", "a1", "2026-11-14T16:27:00Z", [{"type": "text", "text": long}])])
        memory.capture()
        parts = memory.read_talk()
        self.assertGreater(len(parts), 3)
        self.assertTrue(all(len(p["text"]) <= memory.PART_CHARS * 1.5 for p in parts))
        self.assertEqual(len({p["of"] for p in parts}), 1)
        whole = memory.find(parts[2]["id"])
        self.assertEqual([p["part"] for p in whole], list(range(len(parts))))
        self.assertEqual(" ".join(p["text"] for p in whole).count("paragraph"), 8)

    def test_vectors_come_back_from_their_saved_copy(self):
        memory._model = FakeModel()
        self.write([line("user", "u1", "2026-11-14T16:26:00Z", "keep it plain and short", origin={"kind": "human"})])
        memory.vectors(memory.library())
        saved = memory.talk_dir() / "saved.npz"
        os.replace(memory.talk_dir() / memory.VECTORS, saved)            # the machine was reset
        self.assertGreater(memory.merge_vectors(saved), 0)
        self.assertEqual(memory.merge_vectors(saved), 0)
        before = memory._model.calls
        memory.recall("plain", kinds={"you"})
        self.assertEqual(memory._model.calls, before)                    # nothing had to be embedded again

    def test_recall_embeds_a_few_new_passages_and_leaves_the_rest_for_index(self):
        memory._model = FakeModel()
        self.write([line("user", f"u{i}", f"2026-11-14T16:{i:02d}:00Z", f"plain note {i}", origin={"kind": "human"})
                    for i in range(5)])
        memory.capture()
        every = memory.library({"you"})
        v = memory.vectors(every, limit=2)
        self.assertEqual(memory._model.calls, 2)
        self.assertEqual(int((np.abs(v).sum(1) > 0).sum()), 2)            # the two newest; the rest wait
        self.assertEqual([x["text"] for x, row in zip(every, v) if np.abs(row).sum() > 0], ["plain note 3", "plain note 4"])
        memory.vectors(every)
        self.assertEqual(memory._model.calls, 5)                           # index embeds the rest, once

    def test_another_session_s_saved_pages_join_the_talk(self):
        def ev(kind, uid, at, content):
            return {"created_at": at, kind: {"uuid": uid, "internal_anthropic_catchall": {
                "message": {"content": content}, "session_id": "session_other", "timestamp": at}}}
        page = {"ccr": {"data": [
            ev("user", "o1", "2026-11-05T10:00:00Z", "run the paper traders every evening"),
            ev("user", "o2", "2026-11-05T10:00:01Z", "<task-notification>done</task-notification>"),
            ev("user", "o3", "2026-11-05T09:00:00Z", "This session is being continued from a previous conversation."),
            ev("assistant", "o4", "2026-11-05T10:01:00Z", [
                {"type": "thinking", "thinking": "", "signature": "x"},
                {"type": "text", "text": "F1-F5 step nightly at 5:47pm New York."}]),
            {"created_at": "2026-11-05T10:02:00Z", "system": {"uuid": "o5"}}], "has_more": False}}
        path = os.path.join(self._tx.name, "page-001.txt")
        with open(path, "w") as fh:
            fh.write('<other-session nonce="n" untrusted="true">\nAnother session\'s record:\n    '
                     + json.dumps(page) + '\n</other-session nonce="n">')
        self.assertEqual(memory.import_events([path]), 3)               # the person, a summary, a reply
        self.assertEqual(memory.import_events([path]), 0)
        talk = memory.read_talk()
        self.assertEqual([c["kind"] for c in talk], ["summary", "you", "me"])
        self.assertTrue(all(c["session"] == "session_other" for c in talk))
        hit = memory.recall("when do the paper traders run", kinds={"me"})[0]
        self.assertIn("5:47pm", hit["text"])
