import re
import unittest
from pathlib import Path

from brain import claims, decide
from brain.decide import POLICY
from brain.research.stats import binom_lower_bound, binom_one_sided_greater
from tests.util import TempHome

ROOT = Path(__file__).resolve().parent.parent
DOCS = ("README.md", "CLAUDE.md", "docs/DECIDE.md", "docs/TERMINAL.md", "docs/EXPANSION.md")
GATE = POLICY["source_gate"]

# The forms of a track record's `why` that state counts (brain.claims.gate_rows).
NOT_PROVEN = re.compile(r"(\d+) bets, (\d+)% hit, p (\d\.\d\d), q (\d\.\d\d) after (\d+) sources: not proven")
PROVEN = re.compile(r"proven: (\d+) bets, (\d+)% hit, weight (\d\.\d\d)")
FEW = re.compile(r"(\d+) bets? \((\d+) hits?\) from (\d+) calls?; needs \d+ bets")
NONE = re.compile(r"\b0 bets(?: from (\d+) calls?, (\d+) pending)?; needs \d+ bets")


def row(bets, hits, calls=None, pending=0, p=None):
    """A source's counts as gate_rows reads them, with p and the lower bound computed from bets and hits."""
    return {"bets": bets, "hits": hits, "hit_rate": hits / bets if bets else None,
            "p": p if p is not None else binom_one_sided_greater(hits, bets) if bets else None,
            "lower": binom_lower_bound(hits, bets, GATE["alpha_lower"]) if bets else None,
            "mean_signed_excess": 0.01 if bets else None, "up": bets, "down": 0,
            "calls": bets if calls is None else calls, "dropped": 0, "pending": pending, "unpriced": 0,
            "unscorable": 0}


def why(rows):
    return claims.gate_rows(rows, GATE)["src"]["why"]


def hit_counts(n, pct):
    """Every hit count whose rate over n bets prints as pct% (gate_rows writes the rate to whole percents)."""
    return [k for k in range(n + 1) if f"{k / n:.0%}" == f"{pct}%"]


def printed(m):
    """What gate_rows prints for the counts an example states: one string per count that fits it."""
    if m.re is NOT_PROVEN:
        n, pct, _, q, s = int(m[1]), int(m[2]), m[3], float(m[4]), int(m[5])
        # The other sources each have one bet at p = q, so BH gives this source q whenever its p allows it.
        others = {f"other{i}": row(1, 0, p=q) for i in range(s - 1)}
        return {why({"src": row(n, k), **others}) for k in hit_counts(n, pct)}
    if m.re is PROVEN:
        n, pct = int(m[1]), int(m[2])
        return {why({"src": row(n, k)}) for k in hit_counts(n, pct)}
    if m.re is FEW:
        return {why({"src": row(int(m[1]), int(m[2]), calls=int(m[3]))})}
    calls, pending = int(m[1] or 0), int(m[2] or 0)
    return {why({"src": row(0, 0, calls=calls, pending=pending)})}


def block(text, start, end):
    """The lines of text from the one holding start to the next one holding end."""
    i = text.index(start)
    return text[i:text.index(end, i) + len(end)]


class WhyExamplesTest(unittest.TestCase):
    """A reader checks the gate's arithmetic against the docs' examples, so each must be what the code prints
    for the counts it states. "41 bets, 66% hit" can only be 27 hits, whose one-sided p is 0.0298: an
    example reading "p 0.02" could never be printed."""

    def test_every_why_example_is_what_gate_rows_prints(self):
        seen = set()
        for name in DOCS:
            text = (ROOT / name).read_text()
            for form in (NOT_PROVEN, PROVEN, FEW, NONE):
                for m in form.finditer(text):
                    seen.add(form.pattern)
                    outs = printed(m)
                    self.assertIn(m[0], outs, f"{name}: {m[0]!r} cannot come out of gate_rows; "
                                              f"for those counts it prints {sorted(outs)}")
        self.assertEqual(seen, {NOT_PROVEN.pattern, PROVEN.pattern, FEW.pattern, NONE.pattern})

    def test_the_corrected_examples(self):
        self.assertEqual(hit_counts(41, 66), [27])
        self.assertEqual(hit_counts(52, 79), [41])
        self.assertAlmostEqual(binom_one_sided_greater(27, 41), 0.0298, places=4)
        self.assertAlmostEqual(2 * binom_lower_bound(41, 52) - 1, 0.3485, places=4)
        text = (ROOT / "docs/DECIDE.md").read_text()
        self.assertIn('"41 bets, 66% hit, p 0.03, q 0.31 after 12 sources: not proven"', text)
        self.assertIn('"proven: 52 bets, 79% hit, weight 0.35"', text)

    def test_a_wrong_example_is_caught(self):
        for wrong in ("41 bets, 66% hit, p 0.02, q 0.31 after 12 sources: not proven",
                      "proven: 52 bets, 79% hit, weight 0.36",
                      "41 bets, 66% hit, p 0.03, q 0.40 after 12 sources: not proven",   # q above 12 * p
                      "31 bets (20 hits) from 40 calls; needs 30 bets"):                   # it has 30 bets
            m = next(f.search(wrong) for f in (NOT_PROVEN, PROVEN, FEW) if f.search(wrong))
            self.assertNotIn(m[0], printed(m), wrong)


class ShapesTest(TempHome):
    """DECIDE.md's shapes for a track record row and a policy's scorecard list the keys the code returns."""

    def test_track_record_row_keys(self):
        claims.note(self.bank, "X", "X will rise", "https://a.example/x", direction="up",
                    known_at="2026-01-05T00:00:00Z")
        keys = set(claims.track_record(self.bank)["a.example via agent"])
        shape = block((ROOT / "docs/DECIDE.md").read_text(), '{"bets": int, "hits": int', '"why": str}')
        self.assertEqual(set(re.findall(r'"(\w+)":', shape)), keys)

    def test_score_policy_keys(self):
        d1 = decide.score(self.bank)["policies"]["D1"]
        text = (ROOT / "docs/DECIDE.md").read_text()
        shape = block(text, 'def score(bank: Bank, t: str | None = None) -> dict:', '"rows"')
        self.assertEqual(set(re.findall(r'"(\w+)":', shape)) - {"policies", "D1", "rows"}, set(d1))
        actions = block(shape, '"actions": {', "}")
        self.assertEqual(set(re.findall(r'"(\w+)"', actions)) - {"actions"}, set(d1["actions"]))
        statuses = block(text, '"stream_status":', "}")
        self.assertIn('"unpriced"', statuses)


class ScoreClaimsWordingTest(unittest.TestCase):
    """`brain score-claims` counts every call, overlapping ones included, so it overstates the evidence; the
    record that gives a source weight is claims.track_record (`brain decide score`, CLM's first group). A doc
    that names score-claims must say what it counts, and the README must say where the weighed record is."""

    def test_docs_say_score_claims_counts_overlapping_calls(self):
        for name in DOCS:
            for line in (ROOT / name).read_text().splitlines():
                if "score-claims" in line:
                    self.assertIn("overlapping", line, f"{name}: {line.strip()[:120]}")

    def test_readme_points_from_score_claims_to_decide_score(self):
        text = (ROOT / "README.md").read_text()
        line = next(ln for ln in text.splitlines() if ln.startswith("python -m brain score-claims"))
        self.assertIn("decide score", line)
        self.assertIn("python -m brain decide score", text)


if __name__ == "__main__":
    unittest.main()
