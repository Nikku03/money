"""The terminal's static frontend, read as files: CSP-safe markup, no unsafe DOM APIs, and in step with
the server's routes and the command grammar. Nothing here needs a browser."""
import re
import unittest
from html.parser import HTMLParser

from brain.terminal import commands, server

STATIC = server.STATIC
VIEWS = {"help", "des", "chart", "table", "news", "evidence", "markdown", "fact", "curve", "ask"}


class _Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[tuple[str, dict]] = []
        self.script_text = ""
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        self._in_script = tag == "script"

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, text):
        if self._in_script:
            self.script_text += text


class StaticFilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (STATIC / "index.html").read_text(encoding="utf-8")
        cls.js = (STATIC / "app.js").read_text(encoding="utf-8")
        cls.css = (STATIC / "style.css").read_text(encoding="utf-8")
        cls.page = _Page()
        cls.page.feed(cls.html)

    def test_served_files_exist(self):
        for name, _ in server.STATIC_FILES.values():
            self.assertTrue((STATIC / name).is_file(), name)

    def test_page_is_csp_safe(self):
        metas = [a for t, a in self.page.tags if t == "meta" and a.get("name") == "terminal-token"]
        self.assertEqual([m.get("content") for m in metas], ["{{TOKEN}}"])
        refs = [a.get("src") or a.get("href") for t, a in self.page.tags if t in ("script", "link")]
        self.assertEqual(sorted(r for r in refs if not r.startswith("data:")), ["/static/app.js", "/static/style.css"])
        self.assertFalse(self.page.script_text.strip(), "inline script")
        self.assertNotIn("style", [t for t, _ in self.page.tags], "inline <style>")
        for tag, attrs in self.page.tags:
            self.assertNotIn("style", attrs, tag)
            self.assertFalse([k for k in attrs if k.startswith("on")], f"inline handler on <{tag}>")

    def test_script_uses_no_unsafe_apis(self):
        code = re.sub(r"/\*[\s\S]*?\*/|^\s*//.*$", "", self.js, flags=re.M)  # comments may name what is avoided
        for pat in (r"\.innerHTML\b", r"\.outerHTML\b", r"insertAdjacentHTML", r"document\.write", r"\beval\s*\(",
                    r"new\s+Function\b", r"setAttribute\(\s*[\"']style", r"set(?:Timeout|Interval)\(\s*[\"'`]"):
            self.assertIsNone(re.search(pat, code), pat)
        self.assertIn("X-Terminal-Token", self.js)
        self.assertNotRegex(self.css, r"@import|url\(\s*[\"']?https?:")

    def test_script_matches_the_server_and_grammar(self):
        called = set(re.findall(r"[\"'](/api/[a-z]+)", self.js))
        self.assertTrue(called)
        self.assertLessEqual(called, set(server.API), "app.js calls a route the server does not have")
        funcs = re.search(r"const FUNCS = \[([^\]]*)\]", self.js).group(1)
        self.assertEqual(re.findall(r"\"([A-Z]+)\"", funcs), list(commands.FUNCTIONS))
        ranges = re.search(r"const RANGES = \[([^\]]*)\]", self.js).group(1)
        self.assertEqual(tuple(re.findall(r"\"([0-9A-Z]+)\"", ranges)), commands.RANGES)
        views = re.search(r"const VIEWS = \{([^}]*)\}", self.js).group(1)
        self.assertEqual(set(re.findall(r"(\w+):\s*view\w+", views)), VIEWS)


if __name__ == "__main__":
    unittest.main()
