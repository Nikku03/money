"""The terminal's static frontend, read as files: CSP-safe markup, no unsafe DOM APIs, and in step with
the server's routes and the command grammar. Nothing here needs a browser: the behaviour tests run app.js
under node against a small fake DOM and a fake fetch (skipped when node is not installed)."""
import json
import re
import shutil
import subprocess
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


def css_rules(css: str) -> dict[str, dict[str, str]]:
    """Selector -> declarations, merged in source order (rules inside @media included)."""
    out: dict[str, dict[str, str]] = {}
    for sels, body in re.findall(r"([^{}]+)\{([^{}]*)\}", re.sub(r"/\*[\s\S]*?\*/", "", css)):
        decls = dict((k.strip(), v.strip()) for k, _, v in (d.partition(":") for d in body.split(";")) if k.strip())
        for sel in sels.split(","):
            out.setdefault(" ".join(sel.split()), {}).update(decls)
    return out


class StylesheetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = css_rules((STATIC / "style.css").read_text(encoding="utf-8"))

    def decl(self, sel, prop):
        return self.rules.get(sel, {}).get(prop)

    def test_long_tokens_wrap_instead_of_widening(self):
        # one 3,000-letter claim or a 600-letter word in an answer must not push columns off-screen
        for sel in (".tbl td.wrap", ".tbl td .wrap", ".md", ".turn-q"):
            self.assertEqual(self.decl(sel, "overflow-wrap"), "anywhere", sel)
        self.assertEqual(self.decl(".md pre", "overflow-x"), "auto")

    def test_pane_header_keeps_its_buttons(self):
        # nine range buttons in a 500px header used to push reload and maximise out of the clipped header
        self.assertEqual(self.decl(".pane-head > .btn", "flex"), "0 0 auto")
        self.assertEqual(self.decl(".pane-tools", "min-width"), "0")
        self.assertEqual(self.decl(".pane-tools", "overflow-x"), "auto")
        self.assertEqual(self.decl(".pane-tools > *", "flex"), "0 0 auto")
        self.assertEqual(self.decl(".pane-asof", "min-width"), "0")
        title, asof = self.decl(".pane-title", "flex").split(), self.decl(".pane-asof", "flex").split()
        self.assertGreater(float(title[1]), float(asof[1]), "the title gives way before the date")
        # the date shrinks when the header is short of room; it is not hidden on narrow screens (rules in @media merge)
        self.assertNotEqual(self.decl(".pane-asof", "display"), "none")

    def test_fact_attributes_break_into_lines_not_letters(self):
        self.assertEqual(self.decl(".fact .tbl td.attr", "min-width"), "calc(30ch + 16px)")  # 30 characters and the padding
        for sel in (".fact .mono-wrap", ".fact-head .attr"):
            self.assertEqual(self.decl(sel, "overflow-wrap"), "anywhere", sel)
            self.assertIsNone(self.decl(sel, "word-break"), sel)

    def test_group_labels_stay_in_view(self):
        self.assertEqual(self.decl(".tbl tr.grp .grp-label", "position"), "sticky")
        self.assertEqual(self.decl(".tbl tr.grp td:first-child", "position"), "static")


NODE = shutil.which("node")

# Loads app.js with its final `boot();` replaced by an accessor into the module scope, under a fake DOM:
# elements keep children, classes, attributes and listeners; fetch answers from canned responses and
# records every call. The test's code runs as an async function body and returns JSON.
HARNESS = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");
const input = JSON.parse(fs.readFileSync(0, "utf8"));

class FakeNode {
  constructor() { this.childNodes = []; this.parentNode = null; }
  append(...xs) {
    for (let x of xs) {
      if (!(x instanceof FakeNode)) x = new FakeText(String(x));
      if (x.parentNode) x.parentNode.childNodes = x.parentNode.childNodes.filter((c) => c !== x);
      x.parentNode = this;
      this.childNodes.push(x);
    }
  }
  replaceChildren(...xs) { for (const c of this.childNodes) c.parentNode = null; this.childNodes = []; this.append(...xs); }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  get isConnected() { return true; }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(""); }
  set textContent(v) { this.replaceChildren(); if (String(v) !== "") this.append(String(v)); }
}
class FakeText extends FakeNode {
  constructor(t) { super(); this.data = t; }
  get textContent() { return this.data; }
  set textContent(v) { this.data = String(v); }
}
class FakeElement extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.className = "";
    this.attributes = {};
    this.dataset = {};
    this.listeners = {};
    this.style = { setProperty() {} };
    this.scrolledIntoView = 0;
  }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  getAttribute(k) { return k in this.attributes ? this.attributes[k] : null; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  removeEventListener() {}
  get classList() {
    const el = this;
    const list = () => el.className.split(/\s+/).filter(Boolean);
    const set = (on, c) => { el.className = (on ? [...new Set([...list(), c])] : list().filter((x) => x !== c)).join(" "); };
    return {
      add: (...cs) => cs.forEach((c) => set(true, c)),
      remove: (...cs) => cs.forEach((c) => set(false, c)),
      contains: (c) => list().includes(c),
      toggle: (c, force) => { const on = force === undefined ? !list().includes(c) : !!force; set(on, c); return on; },
    };
  }
  querySelectorAll(sel) { return all(this).filter((e) => e.tagName === sel.toUpperCase()); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  getBoundingClientRect() { return this.rect || { top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 }; }
  scrollIntoView() { this.scrolledIntoView += 1; }
  focus() {}
  closest() { return null; }
  setSelectionRange() {}
  cloneNode() { return new FakeElement(this.tagName); }
}
function all(root) {
  const out = [];
  const walk = (n) => { for (const c of n.childNodes) if (c instanceof FakeElement) { out.push(c); walk(c); } };
  walk(root);
  return out;
}

const byId = new Map();
const docListeners = {};
const storage = new Map();
globalThis.Node = FakeNode;
globalThis.document = {
  createElement: (t) => new FakeElement(t),
  createTextNode: (t) => new FakeText(String(t)),
  querySelector: (sel) => {
    if (sel.startsWith("meta")) return null;
    if (!byId.has(sel)) byId.set(sel, new FakeElement("div"));
    return byId.get(sel);
  },
  addEventListener: (type, fn) => { (docListeners[type] = docListeners[type] || []).push(fn); },
  documentElement: new FakeElement("html"),
  body: new FakeElement("body"),
  activeElement: null,
};
globalThis.window = {
  localStorage: {
    getItem: (k) => (storage.has(k) ? storage.get(k) : null),
    setItem: (k, v) => { storage.set(k, String(v)); },
  },
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
  addEventListener() {},
  devicePixelRatio: 1,
  location: { reload() {} },
};
for (const [k, v] of Object.entries(input.storage || {})) storage.set(k, JSON.stringify(v));
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "", fontFamily: "monospace" });
globalThis.setTimeout = () => 0;  // no polls or timers: the test drives every step
globalThis.clearTimeout = () => {};
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};
globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};
globalThis.ResizeObserver = class { observe() {} disconnect() {} };

const calls = [];
globalThis.fetch = async (url, init = {}) => {
  const u = new URL(url, "http://terminal.test");
  calls.push({ path: u.pathname, cmd: u.searchParams.get("cmd"), body: init.body ? JSON.parse(init.body) : null });
  let data = { ok: false, error: "no canned response" };
  if (u.pathname === "/api/run" && input.responses && input.responses[u.searchParams.get("cmd")]) data = input.responses[u.searchParams.get("cmd")];
  else if (u.pathname === "/api/status") data = input.status || { ok: true, jobs_running: 0, analyst: {} };
  else if (u.pathname === "/api/ask") data = { ok: true, answer: "An answer." };
  return { ok: true, status: 200, json: async () => JSON.parse(JSON.stringify(data)) };
};

const tail = /\n\s*boot\(\);\s*\n\}\)\(\);\s*$/;
let src = fs.readFileSync(input.app, "utf8");
if (!tail.test(src)) throw new Error("app.js no longer ends with `boot(); })();`: update the harness");
src = src.replace(tail, "\n  globalThis.__app = (expr) => eval(expr);\n})();\n");
vm.runInThisContext(src, { filename: "app.js" });
const app = globalThis.__app;
const settle = async () => { for (let k = 0; k < 50; k += 1) await new Promise((r) => setImmediate(r)); };
const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
new AsyncFunction("app", "calls", "settle", "all", "document", "window", "input", "docListeners", input.code)(
  app, calls, settle, all, document, window, input, docListeners)
  .then((out) => process.stdout.write(JSON.stringify(out === undefined ? null : out)))
  .catch((e) => { process.stderr.write(String((e && e.stack) || e)); process.exit(1); });
"""


def run_app(code: str, **data):
    """Run `code` (an async JS function body) against app.js; returns what it returns."""
    payload = json.dumps({"app": str(STATIC / "app.js"), "code": code, **data})
    proc = subprocess.run([NODE, "-e", HARNESS], input=payload, capture_output=True, text=True, timeout=120)
    if proc.returncode:
        raise AssertionError("node: " + proc.stderr[-3000:])
    return json.loads(proc.stdout)


def server_res(line: str, view: str = "table", data: dict | None = None) -> dict:
    """What /api/run answers for `line`, as far as the frontend cares: the server's own parse."""
    try:
        c = commands.parse(line)
    except commands.CommandError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "fn": c.fn, "ticker": c.ticker, "args": c.args, "view": view,
            "data": data if data is not None else {"columns": [], "rows": []}}


@unittest.skipUnless(NODE, "node is not installed")
class AppBehaviourTest(unittest.TestCase):
    def assertSafe(self, line, why):
        """`line`, sent by the page on its own, must neither start a job nor put a question to the model."""
        try:
            c = commands.parse(line)
        except commands.CommandError:
            return
        self.assertNotEqual(c.fn, "INGEST", f"{why}: {line!r} starts a job")
        self.assertFalse(c.fn == "ASK" and c.args, f"{why}: {line!r} asks the model again")

    def test_rerun_lines_never_ingest_or_ask(self):
        # tickers that are also codes (CF, GP, FA) put the code first: `CF INGEST` ingests CF
        lines = ["CF INGEST", "GP INGEST", "FA INGEST", "INGEST CF", "AAPL INGEST", "ingest markets", "GO INGEST",
                 "CF US EQUITY INGEST", "CF ASK what is the margin trend", "GP ASK q", "AAPL ASK why now",
                 "ASK what changed", "ASK INGEST AAPL", "INGEST ASK", "AAPL GP 5Y SPY", "CF DES", "WEI"]
        res = {line: server_res(line) for line in lines}
        out = run_app("""
          const safeRerun = app("safeRerun"), rerunLine = app("rerunLine");
          const o = {};
          for (const line of input.lines) o[line] = [safeRerun(line), rerunLine(line, input.res[line]), rerunLine(line, null)];
          return o;
        """, lines=lines, res=res)
        for line, reruns in out.items():
            for again in reruns:
                self.assertSafe(again, f"rerun of {line!r}")
        self.assertEqual(out["CF INGEST"][0], "JOBS")
        self.assertEqual(out["GP ASK q"][0], "GP ASK")
        self.assertEqual(out["CF ASK what is the margin trend"][0], "CF ASK")
        self.assertEqual(out["ASK INGEST AAPL"][0], "ASK")
        for line in ("AAPL GP 5Y SPY", "CF DES", "WEI"):
            self.assertEqual(out[line], [line] * 3)

    def test_restored_layout_never_ingests_or_asks(self):
        # a layout saved by an earlier page may hold the raw lines
        saved = {"panes": ["CF INGEST", "CF ASK what is the margin trend", "GP ASK why", "AAPL DES"], "focus": 1}
        out = run_app('return app("loadLayout")();', storage={"money-terminal.layout.v1": saved})
        self.assertEqual(out["lines"], ["JOBS", "CF ASK", "GP ASK", "AAPL DES"])
        self.assertEqual(out["focus"], 1)

    def test_job_refresh_reload_and_restore_do_not_start_another_ingest(self):
        rows = [{"id": "j1", "target": "CF", "status": "queued"}]
        responses = {"CF INGEST": server_res("CF INGEST", data={"columns": [], "rows": rows}), "JOBS": server_res("JOBS")}
        out = run_app("""
          const Pane = app("Pane"), panes = app("panes");
          for (let i = 0; i < 4; i += 1) panes.push(new Pane(i));
          await panes[0].run("CF INGEST");  // typed once: one job
          await settle();
          await app("refreshStatus")();  // the job ends: panes refresh
          await settle();
          await app("refreshStatus")();
          await settle();
          panes[0].reload();  // the reload button
          await settle();
          const saved = JSON.parse(window.localStorage.getItem("money-terminal.layout.v1"));
          return { runs: calls.filter((c) => c.path === "/api/run").map((c) => c.cmd), saved, restored: app("loadLayout")().lines };
        """, responses=responses)
        runs = [r for r in out["runs"] if r != "WEI"]  # the ticker tape refreshes after a job
        self.assertEqual(runs[0], "CF INGEST")
        for line in runs[1:] + out["saved"]["panes"] + out["restored"]:
            self.assertSafe(line, "after the job")
        self.assertEqual(runs.count("CF INGEST"), 1, runs)
        self.assertEqual(out["saved"]["panes"][0], "JOBS")

    def test_reload_does_not_ask_the_question_again(self):
        def ask(line, question):  # the worst case: the server reads `GP ASK ...` as ASK with GP in focus
            return {"ok": True, "fn": "ASK", "ticker": line.split()[0] if line.split()[0] != "ASK" else None,
                    "args": [question] if question else [], "view": "ask",
                    "data": {"available": True, "question": question, "ticker": None}}
        responses = {"GP ASK why did it fall": ask("GP ASK", "why did it fall"), "GP ASK": ask("GP ASK", ""),
                     "ASK what changed": ask("ASK", "what changed"), "ASK": ask("ASK", "")}
        out = run_app("""
          const Pane = app("Pane"), panes = app("panes");
          for (let i = 0; i < 4; i += 1) panes.push(new Pane(i));
          await panes[0].run("GP ASK why did it fall");
          await panes[1].run("ASK what changed");
          await settle();
          panes[0].reload();
          panes[1].reload();
          await settle();
          const saved = JSON.parse(window.localStorage.getItem("money-terminal.layout.v1"));
          return { asks: calls.filter((c) => c.path === "/api/ask").map((c) => c.body.question), saved: saved.panes };
        """, responses=responses)
        self.assertEqual(out["asks"], ["why did it fall", "what changed"])
        self.assertEqual(out["saved"][:2], ["GP ASK", "ASK"])

    def test_markdown_blocks_render_in_linear_time(self):
        # each of these took seconds (the heading cubically) and froze the tab while BRIEF rendered
        md = "\n".join(["Analyst note", "# a" + " " * 3000 + "x", "- " + " " * 40000 + "x y",
                        "| a |", " " * 40000 + "x", "## Title ##", "# C#", "### ###", "####### seven"])
        out = run_app("""
          const t0 = process.hrtime.bigint();
          const root = app("renderMarkdown")(input.md, { ticker: "MSFT" });
          const ms = Number(process.hrtime.bigint() - t0) / 1e6;
          return { ms, heads: all(root).filter((e) => /^H[1-6]$/.test(e.tagName)).map((e) => [e.tagName, e.textContent]) };
        """, md=md)
        self.assertLess(out["ms"], 250)
        self.assertEqual(out["heads"], [["H1", "a" + " " * 3000 + "x"], ["H2", "Title"], ["H1", "C#"], ["H3", ""]])

    def test_links_keep_balanced_parentheses(self):
        md = ("Wiki https://en.wikipedia.org/wiki/Apple_(company) end\n\n"
              "[ok](https://ok.example/a_(b)) and (see https://x.example/a). Done https://y.example/b.")
        out = run_app("""
          const root = app("renderMarkdown")(input.md, { ticker: "AAPL" });
          return { hrefs: all(root).filter((e) => e.tagName === "A").map((e) => e.href),
                   paras: all(root).filter((e) => e.tagName === "P").map((e) => e.textContent) };
        """, md=md)
        self.assertEqual(out["hrefs"], ["https://en.wikipedia.org/wiki/Apple_(company)", "https://ok.example/a_(b)",
                                        "https://x.example/a", "https://y.example/b"])
        self.assertEqual(out["paras"], ["Wiki https://en.wikipedia.org/wiki/Apple_(company) end",
                                        "ok and (see https://x.example/a). Done https://y.example/b."])

    def test_price_line_and_tooltip_follow_the_close(self):
        # VZ 5Y: the close fell 20% while the total return with dividends was +9.9%
        series = [{"date": "2021-10-11", "close": 52.18, "adj": 40.0, "volume": 1000},
                  {"date": "2024-01-02", "close": 45.0, "adj": 41.0, "volume": 1000},
                  {"date": "2026-10-09", "close": 41.65, "adj": 43.956, "volume": 1000}]
        out = run_app("""
          const model = (s, ret) => app("chartModel")({ ticker: "VZ", series: s, stats: { ret } });
          const m = model(input.series, 0.0989);
          const up = model(input.series.slice().reverse().map((x, i) => ({ ...x, date: input.series[i].date })), -0.05);
          return { color: m.lines[0].color, upColor: up.lines[0].color, tip: app("priceTipRows")(m, 2) };
        """, series=series)
        self.assertEqual(out["color"], "down")
        self.assertEqual(out["upColor"], "up")
        self.assertEqual(out["tip"][:3], [["Close", "41.65", ""], ["From start", "-20.18%", "down"],
                                          ["Total return", "+9.89%", "up"]])

    def test_tables_wrap_long_descriptions_and_names(self):
        desc = "results, officer/director change, charter/bylaw change, shareholder vote, exhibits"
        attr = "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"
        out = run_app("""
          const pane = { body: document.createElement("div"), toolsEl: document.createElement("span"), disposers: [], run() {} };
          app("viewDes")(pane, { ok: true, ticker: "BRK-B", data: { ticker: "BRK-B", filings: [
            { date: "2026-05-04", form: "8-K", desc: input.desc, url: "https://www.sec.gov/x/d159922d8k.htm", cite: "abcd1234" }] } });
          const fact = app("factBody")({ fact: { entity: "AAPL", attr: "derived.market_cap", value: 1 },
            inputs: [{ ref: "AAPL/0011", cite: "00112233", attr: input.attr, value: 15408095000 }] }, pane);
          const grp = app("gridTable")([{ label: "Item" }, { label: "FY2025" }], [{ group: "Balance sheet" }, { cells: ["Assets", "1"] }]);
          const cls = (root, text) => all(root).filter((e) => e.tagName === "TD" && e.textContent === text).map((e) => e.className);
          const g = all(grp).find((e) => e.tagName === "TR" && e.className === "grp");
          const name = all(fact).find((e) => e.tagName === "TD" && e.textContent === input.attr).firstChild;
          return { desc: cls(pane.body, input.desc), attr: cls(fact, input.attr),
                   pieces: name.childNodes.filter((n) => n.tagName !== "WBR").map((n) => n.textContent),
                   group: g.firstChild.childNodes.map((e) => [e.tagName, e.className, e.textContent]) };
        """, desc=desc, attr=attr)
        self.assertEqual(len(out["desc"]), 1)
        self.assertIn("wrap", out["desc"][0].split())
        self.assertEqual(len(out["attr"]), 1)
        self.assertIn("attr", out["attr"][0].split())
        self.assertNotIn("grow", out["attr"][0].split())
        self.assertEqual(out["pieces"], ["us-gaap:", "Weighted", "Average", "Number", "Of", "Diluted", "Shares", "Outstanding"])
        self.assertEqual(out["group"], [["SPAN", "grp-label", "Balance sheet"]])

    def test_des_metrics_show_the_date_they_describe(self):
        metrics = [{"key": "market_cap", "label": "Market value", "value": 4.95e12, "unit": "USD", "as_of": "2026-10-09",
                    "cite": "292044a9", "inputs": []},
                   {"key": "net_margin", "label": "Net margin", "value": 0.24, "unit": "ratio", "as_of": "2025-09-27",
                    "cite": "11223344", "inputs": []}]
        out = run_app("""
          const pane = { body: document.createElement("div"), toolsEl: document.createElement("span"), disposers: [], run() {} };
          app("viewDes")(pane, { ok: true, ticker: "AAPL", data: { ticker: "AAPL", metrics: input.metrics } });
          return all(pane.body).filter((e) => e.tagName === "TR")
            .map((tr) => tr.childNodes.filter((td) => td.className === "x").map((td) => td.textContent));
        """, metrics=metrics)
        self.assertEqual(out, [["2026-10-09"], ["2025-09-27"]])

    def test_keyboard_focus_and_runs_reveal_a_pane_below_the_fold(self):
        responses = {"AAPL DES": server_res("AAPL DES", view="des", data={"ticker": "AAPL"})}
        out = run_app("""
          const Pane = app("Pane"), panes = app("panes");
          for (let i = 0; i < 4; i += 1) panes.push(new Pane(i));
          const grid = document.querySelector("#grid");
          const scrolled = () => panes.map((p) => p.el.scrolledIntoView);
          const res = {};
          grid.scrollHeight = 900; grid.clientHeight = 900;  // the 2x2 grid: everything is in view
          app("bindKeys")();
          const key = (k) => docListeners.keydown.forEach((f) => f({ key: k, code: "Digit" + k, ctrlKey: true, preventDefault() {}, target: null }));
          key("4");
          res.wide = scrolled();
          grid.scrollHeight = 2100; grid.clientHeight = 820;  // one column under 900px: the grid scrolls
          key("4");
          res.ctrl4 = scrolled();
          app("focusPane")(2);  // a click focuses a pane that is in view already
          res.click = scrolled();
          app("booted = true");
          await panes[1].run("AAPL DES");  // Ctrl+Enter or Shift-click: output goes to another pane
          await panes[2].run("AAPL DES", true);  // a quiet refresh after a job
          res.run = scrolled();
          return res;
        """, responses=responses)
        self.assertEqual(out["wide"], [0, 0, 0, 0])
        self.assertEqual(out["ctrl4"], [0, 0, 0, 1])
        self.assertEqual(out["click"], [0, 0, 0, 1])
        self.assertEqual(out["run"], [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
