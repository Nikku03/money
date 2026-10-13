import re
import unittest
from pathlib import Path

from brain.sources import insiders

ROOT = Path(__file__).resolve().parent.parent
QUARTER = re.compile(r"\b20\d\d[qQ][1-4]\b")


class InsiderStartDocsTest(unittest.TestCase):
    """The docs name the quarter `brain insiders` starts from by default. Following them on a fresh home has
    to read the same data sets the recorded H6 run read, so they must name the code's default."""

    def test_docs_name_the_default_start_quarter(self):
        for name in ("README.md", "docs/EXPANSION.md"):
            text = (ROOT / name).read_text()
            found = QUARTER.findall(text)
            self.assertTrue(found, f"{name} names no insider start quarter")
            self.assertEqual(sorted(set(q.lower() for q in found)), [insiders.DEFAULT_START], name)

    def test_readme_research_steps_say_where_insider_history_starts(self):
        text = (ROOT / "README.md").read_text()
        research = text[text.index("Research:"):]
        line = next(ln for ln in research.splitlines() if ln.startswith("python -m brain insiders"))
        self.assertIn(insiders.DEFAULT_START, line)


if __name__ == "__main__":
    unittest.main()
