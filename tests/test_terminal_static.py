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
VIEWS = {"help", "des", "chart", "table", "news", "evidence", "markdown", "fact", "curve", "ask", "decision", "decisions",
         "forecast", "forecasts"}


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
        # prose keeps its measure; a wide table (the brief's insider trades, 9 columns) may use the pane's width
        self.assertEqual(self.decl(".md > *", "max-width"), "128ch")
        self.assertEqual((self.decl(".md > .md-table", "max-width"), self.decl(".md .md-table", "overflow-x")),
                         ("100%", "auto"))

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
  calls.push({ path: u.pathname, cmd: u.searchParams.get("cmd"), url: u.searchParams.get("ticker"),
               body: init.body ? JSON.parse(init.body) : null });
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
        self.assertFalse(c.fn == "DEC" and c.args, f"{why}: {line!r} starts a decision job")
        self.assertFalse(c.fn == "FCST" and c.args, f"{why}: {line!r} starts a forecast job")

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

    def test_dec_new_reruns_as_the_decision_it_made(self):
        lines = ["AAPL DEC NEW", "dec new aapl", "DEC AAPL NEW", "AAPL US EQUITY DEC NEW <GO>", "GO DEC NEW", "CF DEC NEW",
                 "DEC DEC NEW", "DEC NEW DEC", "NEW DEC NEW", "DEC NEW INGEST", "DEC NEW ASK", "DEC INGEST", "DEC NEW",
                 "AAPL DEC", "NEW DEC", "DEC", "DEC DES", "dec  new   aapl"]
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
        for line in ("AAPL DEC NEW", "DEC AAPL NEW", "AAPL US EQUITY DEC NEW <GO>", "dec new aapl", "dec  new   aapl"):
            for again in out[line]:
                c = commands.parse(again)
                self.assertEqual((c.fn, c.ticker, c.args), ("DEC", "AAPL", []), (line, again))
        self.assertEqual(out["AAPL DEC NEW"], ["AAPL DEC"] * 3)
        self.assertEqual(out["dec new aapl"][1], "AAPL DEC")  # the server's reading, once it answered
        self.assertEqual(out["GO DEC NEW"], ["GO DEC"] * 3)  # the ticker GO is kept
        self.assertEqual(out["CF DEC NEW"], ["CF DEC"] * 3)
        self.assertEqual(out["NEW DEC NEW"][0], "NEW DEC")  # before the code, NEW was the ticker
        self.assertEqual(out["DEC NEW INGEST"], ["JOBS"] * 3)  # dropping NEW must not make INGEST the function
        self.assertEqual(out["DEC NEW"][0], "DEC")
        for line in ("AAPL DEC", "NEW DEC", "DEC", "DEC DES"):
            self.assertEqual(out[line], [line] * 3)

    def test_dec_new_starts_polling_and_the_refresh_shows_the_decision(self):
        rows = [{"id": "j1", "target": "DEC AAPL", "status": "running"}]
        responses = {"AAPL DEC NEW": server_res("AAPL DEC NEW", data={"columns": [], "rows": rows}),
                     "AAPL DEC": server_res("AAPL DEC", view="decision", data={"ticker": "AAPL", "action": "NO_EDGE"}),
                     "DEC": server_res("DEC", view="decisions", data={}), "WL": server_res("WL"), "JOBS": server_res("JOBS")}
        out = run_app("""
          const Pane = app("Pane"), panes = app("panes");
          for (let i = 0; i < 4; i += 1) panes.push(new Pane(i));
          await panes[1].run("DEC");
          await panes[2].run("WL");
          await panes[0].run("AAPL DEC NEW");  // typed once: one job
          await settle();
          const polling = app("jobsRunning");
          await app("refreshStatus")();  // the job ends: the decision panes refresh
          await settle();
          await app("refreshStatus")();
          await settle();
          panes[0].reload();  // the reload button
          await settle();
          const saved = JSON.parse(window.localStorage.getItem("money-terminal.layout.v1"));
          return { runs: calls.filter((c) => c.path === "/api/run").map((c) => c.cmd), polling, saved: saved.panes,
                   restored: app("loadLayout")().lines, shown: panes[0].res.view };
        """, responses=responses)
        runs = [r for r in out["runs"] if r != "WEI"]  # the ticker tape refreshes after a job
        self.assertEqual(runs[:3], ["DEC", "WL", "AAPL DEC NEW"])
        self.assertEqual(out["polling"], 1)  # a DEC NEW result counts as a running job, as INGEST does
        self.assertEqual(runs.count("AAPL DEC NEW"), 1, runs)
        for line in runs[3:] + out["saved"] + out["restored"]:
            self.assertSafe(line, "after the job")
        self.assertIn("AAPL DEC", runs[3:])  # the refresh after the job
        self.assertEqual(runs[3:].count("DEC"), 1)  # the scorecard and the WL decision column refresh too
        self.assertEqual(runs[3:].count("WL"), 1)
        self.assertEqual(out["saved"][0], "AAPL DEC")
        self.assertEqual(out["restored"][0], "AAPL DEC")
        self.assertEqual(out["shown"], "decision")

    def test_a_dec_new_pane_waits_for_its_own_job(self):
        # an ingest of MSFT finishes while the decision on AAPL still runs: the DEC NEW pane keeps its job table
        rows = [{"id": "j2", "target": "DEC AAPL", "status": "running"}, {"id": "j1", "target": "MSFT", "status": "running"}]
        responses = {"AAPL DEC NEW": server_res("AAPL DEC NEW", data={"columns": [], "rows": rows}),
                     "AAPL DEC": server_res("AAPL DEC", view="decision", data={"ticker": "AAPL", "action": "NO_EDGE"}),
                     "JOBS": server_res("JOBS")}
        for baseline in (True, False):  # with the finished count of an earlier poll, and before any poll
            with self.subTest(baseline=baseline):
                out = run_app("""
                  const Pane = app("Pane"), panes = app("panes");
                  for (let i = 0; i < 4; i += 1) panes.push(new Pane(i));
                  const st = (running, active, finished) => ({ ok: true, jobs_running: running, jobs_queued: 0,
                                                               jobs_active: active, jobs_finished: finished, analyst: {} });
                  if (input.baseline) { input.status = st(0, [], 5); await app("refreshStatus")(); }
                  await panes[0].run("AAPL DEC NEW");
                  await panes[1].run("JOBS");
                  await settle();
                  const runs = () => calls.filter((c) => c.path === "/api/run").map((c) => c.cmd);
                  input.status = st(1, ["DEC AAPL"], 6);   // only the MSFT ingest finished
                  await app("refreshStatus")();
                  await settle();
                  const mid = runs(), midView = panes[0].res.fn + " " + (panes[0].res.args || []).join(" ");
                  input.status = st(0, [], 7);             // now the decision is done too
                  await app("refreshStatus")();
                  await settle();
                  return { mid, midView, runs: runs(), view: panes[0].res.view };
                """, responses=responses, baseline=baseline)
                mid = [r for r in out["mid"] if r != "WEI"]
                self.assertEqual(mid.count("AAPL DEC"), 0, mid)  # not while DEC AAPL runs
                self.assertEqual(out["midView"], "DEC NEW")
                self.assertIn("JOBS", mid[2:])  # the JOBS pane refreshed after the ingest finished
                runs = [r for r in out["runs"] if r != "WEI"]
                self.assertEqual(runs.count("AAPL DEC"), 1, runs)  # once its own job is done
                self.assertEqual(runs.count("AAPL DEC NEW"), 1, runs)
                self.assertEqual(out["view"], "decision")

    def test_a_queued_decision_is_not_taken_for_a_finished_one(self):
        # one decision at a time: MSFT waits behind AAPL, and the running count stays 1 as one hands over to the other
        a = [{"id": "j1", "target": "DEC AAPL", "status": "running"}]
        m = [{"id": "j2", "target": "DEC MSFT", "status": "queued"}, *a]
        responses = {"AAPL DEC NEW": server_res("AAPL DEC NEW", data={"columns": [], "rows": a}),
                     "MSFT DEC NEW": server_res("MSFT DEC NEW", data={"columns": [], "rows": m}),
                     "AAPL DEC": server_res("AAPL DEC", view="decision", data={"ticker": "AAPL", "action": "NO_EDGE"}),
                     "MSFT DEC": server_res("MSFT DEC", view="decision", data={"ticker": "MSFT", "action": "NO_EDGE"})}
        out = run_app("""
          const Pane = app("Pane"), panes = app("panes");
          for (let i = 0; i < 4; i += 1) panes.push(new Pane(i));
          const st = (running, queued, active, finished) => ({ ok: true, jobs_running: running, jobs_queued: queued,
                                                               jobs_active: active, jobs_finished: finished, analyst: {} });
          const runs = () => calls.filter((c) => c.path === "/api/run").map((c) => c.cmd).filter((c) => c !== "WEI");
          input.status = st(0, 0, [], 0);
          await app("refreshStatus")();
          await panes[0].run("AAPL DEC NEW");
          await panes[1].run("MSFT DEC NEW");
          await settle();
          const steps = [];
          for (const s of [st(1, 1, ["DEC AAPL", "DEC MSFT"], 0),   // nothing finished
                           st(1, 0, ["DEC MSFT"], 1),               // AAPL's is done, MSFT's started
                           st(0, 0, [], 2)]) {                      // MSFT's is done
            input.status = s;
            await app("refreshStatus")();
            await settle();
            steps.push({ runs: runs(), polling: app("jobsRunning"), lines: panes.slice(0, 2).map((p) => p.line) });
          }
          return steps;
        """, responses=responses)
        start = ["AAPL DEC NEW", "MSFT DEC NEW"]
        self.assertEqual(out[0]["runs"], start)  # a queued job is not a finished one
        self.assertEqual(out[0]["polling"], 2)  # queued counts as active: the fast poll goes on
        self.assertEqual(out[1]["runs"], start + ["AAPL DEC"])  # AAPL's pane, and only it
        self.assertEqual(out[2]["runs"], start + ["AAPL DEC", "AAPL DEC", "MSFT DEC"])  # a DEC pane refreshes; MSFT's now
        self.assertEqual(out[2]["lines"], ["AAPL DEC", "MSFT DEC"])
        self.assertEqual(out[2]["polling"], 0)

    def test_dec_shows_the_servers_job_note(self):
        d = {"ticker": "ZZZQ", "action": "NO_EDGE", "label": "NO_EDGE: do nothing", "score": 0.0, "threshold": 0.5,
             "items": [], "context": [], "withheld": 0, "lean": {}, "notes": [], "history": {"columns": [], "rows": []},
             "job_note": "the last ZZZQ DEC NEW (job j3, done) recorded nothing: no prices for ZZZQ: ingest ZZZQ"}
        out = run_app("""
          const p = new (app("Pane"))(0), q = new (app("Pane"))(1);
          p.show({ ok: true, fn: "DEC", ticker: "ZZZQ", args: [], view: "decision", data: input.d });
          q.show({ ok: true, fn: "DEC", ticker: "ZZZQ", args: [], view: "decision", data: { ...input.d, job_note: null } });
          const notes = (x) => all(x.body).filter((e) => e.className === "dec-job").map((e) => e.textContent);
          const shown = { p: notes(p), q: notes(q) };
          const link = all(p.body).find((e) => e.className === "dec-job").childNodes.find((e) => e.className === "cmdlink");
          link.listeners.click[0]({ stopPropagation() {} });  // runs JOBS in this pane
          await settle();
          return { ...shown, ran: calls.filter((c) => c.path === "/api/run").map((c) => c.cmd) };
        """, d=d)
        self.assertEqual(out["p"], [d["job_note"] + ". JOBS"])
        self.assertEqual(out["q"], [])
        self.assertEqual(out["ran"], ["JOBS"])

    def test_fcst_new_reruns_as_the_forecast_it_made(self):
        lines = ["AAPL FCST NEW", "fcst new aapl", "FCST AAPL NEW", "AAPL US EQUITY FCST NEW <GO>", "GO FCST NEW",
                 "CF FCST NEW", "FCST FCST NEW", "FCST NEW FCST", "NEW FCST NEW", "FCST NEW INGEST", "FCST NEW ASK",
                 "FCST NEW", "AAPL FCST", "FCST", "FCST DES", "DEC FCST NEW", "FCST DEC NEW", "fcst  new   aapl"]
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
        for line in ("AAPL FCST NEW", "FCST AAPL NEW", "AAPL US EQUITY FCST NEW <GO>", "fcst new aapl", "fcst  new   aapl"):
            for again in out[line]:
                c = commands.parse(again)
                self.assertEqual((c.fn, c.ticker, c.args), ("FCST", "AAPL", []), (line, again))
        self.assertEqual(out["AAPL FCST NEW"], ["AAPL FCST"] * 3)
        self.assertEqual(out["CF FCST NEW"], ["CF FCST"] * 3)
        self.assertEqual(out["FCST NEW INGEST"], ["JOBS"] * 3)
        self.assertEqual(out["DEC FCST NEW"], ["DEC FCST"] * 3)  # the ticker DEC's forecast
        self.assertEqual(out["FCST DEC NEW"], ["FCST DEC"] * 3)  # the ticker FCST's decision
        for line in ("AAPL FCST", "FCST", "FCST DES"):
            self.assertEqual(out[line], [line] * 3)

    def test_fcst_new_polls_waits_for_its_own_job_then_shows_the_forecast(self):
        rows = [{"id": "j2", "target": "FCST AAPL", "status": "running"}, {"id": "j1", "target": "MSFT", "status": "running"}]
        responses = {"AAPL FCST NEW": server_res("AAPL FCST NEW", data={"columns": [], "rows": rows}),
                     "AAPL FCST": server_res("AAPL FCST", view="forecast", data={"ticker": "AAPL", "forecasts": []}),
                     "FCST": server_res("FCST", view="forecasts", data={}), "JOBS": server_res("JOBS")}
        out = run_app("""
          const Pane = app("Pane"), panes = app("panes");
          for (let i = 0; i < 4; i += 1) panes.push(new Pane(i));
          const st = (running, active, finished) => ({ ok: true, jobs_running: running, jobs_queued: 0,
                                                       jobs_active: active, jobs_finished: finished, analyst: {} });
          input.status = st(0, [], 3);
          await app("refreshStatus")();
          await panes[1].run("FCST");
          await panes[0].run("AAPL FCST NEW");  // typed once: one job
          await settle();
          const polling = app("jobsRunning");
          const runs = () => calls.filter((c) => c.path === "/api/run").map((c) => c.cmd).filter((c) => c !== "WEI");
          input.status = st(1, ["FCST AAPL"], 4);   // only the MSFT ingest finished
          await app("refreshStatus")();
          await settle();
          const mid = runs(), midView = panes[0].res.fn + " " + (panes[0].res.args || []).join(" ");
          input.status = st(0, [], 5);              // the forecast is done too
          await app("refreshStatus")();
          await settle();
          panes[0].reload();
          await settle();
          const saved = JSON.parse(window.localStorage.getItem("money-terminal.layout.v1"));
          return { polling, mid, midView, runs: runs(), view: panes[0].res.view, saved: saved.panes };
        """, responses=responses)
        self.assertEqual(out["polling"], 2)  # a FCST NEW result counts its jobs as running, as INGEST does
        self.assertEqual(out["mid"].count("AAPL FCST"), 0, out["mid"])  # not while FCST AAPL runs
        self.assertEqual(out["midView"], "FCST NEW")
        self.assertEqual(out["runs"].count("AAPL FCST NEW"), 1, out["runs"])
        self.assertIn("AAPL FCST", out["runs"])
        self.assertGreaterEqual(out["runs"].count("FCST"), 2)  # the FCST pane refreshes when a job ends
        for line in out["runs"][2:] + out["saved"]:
            self.assertSafe(line, "after the job")
        self.assertEqual((out["view"], out["saved"][0]), ("forecast", "AAPL FCST"))

    FORECAST = {
        "ticker": "AAPL", "name": "Apple Inc.", "cite": "a1b2c3d4", "retrieved_at": "2026-10-15T22:00:00Z",
        "series": [{"date": "2026-10-13", "close": 100.0}, {"date": "2026-10-14", "close": 101.0},
                   {"date": "2026-10-15", "close": 102.0}],
        "future": ["2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"],
        "forecasts": [
            {"forecaster": "F1", "horizon": 5, "date": "2026-10-15", "entry": "2026-10-16", "exit": "2026-10-22",
             "close": 102.0, "close_date": "2026-10-15", "p_beat": 0.55,
             "band": {"q05": 92.0, "q16": 97.0, "q50": 102.5, "q84": 108.0, "q95": 112.0},
             "move68_pct": [-4.9, 5.88], "move90_pct": [-9.8, 9.8],
             "drivers": [{"name": "mom_12_1", "group": "P", "push": 0.08}, {"name": "abn_volume", "group": "B", "push": -0.04}],
             "model": {"id": "F1", "sha": "abcdef1234567890", "trained_through": "2026-10-15", "rows": 150000},
             "cite": "f0f0f0f0", "outcome": {"status": "pending"},
             "llm": {"p_beat": 0.6, "ok": True, "reason": "Mild momentum.", "model": "qwen3:8b", "cite": "e1e1e1e1"},
             "record": ["last 2 months (F1a): not run yet", "2019-2026 (F1b): not run yet"]},
            {"forecaster": "F1", "horizon": 21, "date": "2026-10-14", "close": 101.0, "p_beat": 0.48,
             "band": {"q05": 80.0, "q16": 90.0, "q50": 101.0, "q84": 112.0, "q95": 120.0}, "exit": "2026-11-13",
             "outcome": {"status": "scored", "exit": "2026-10-20", "close_exit": 95.0, "in68": True}, "drivers": [],
             "cite": "f1f1f1f1", "record": []}],
        "live": {"columns": [{"key": "stream", "label": "Forecaster", "type": "text"}], "rows": [{"stream": "F1 h=5"}]},
        "history": {"columns": [{"key": "date", "label": "Date", "type": "date"}], "rows": [{"date": "2026-10-15"}]},
        "note": "Paper only, not financial advice.",
        "job_note": "a new forecast on AAPL is running (job j3); AAPL FCST shows it when it is done"}

    def test_forecast_chart_model(self):
        out = run_app("""
          const m = app("forecastModel")(input.d);
          return { dates: m.dates, pts: m.lines[0].pts.length, kind: m.kind, ref: m.ref, extent: m.extent,
                   cones: m.cones.map((c) => [c.horizon, c.color, c.i0, c.i1, c.realised, c.in68]),
                   tip: m.tipRows(7), tipStart: m.tipRows(2) };
        """, d=self.FORECAST)
        self.assertEqual(out["dates"][:3], ["2026-10-13", "2026-10-14", "2026-10-15"])
        self.assertEqual(len(out["dates"]), 8)  # three closes, then the window's sessions
        self.assertEqual((out["pts"], out["kind"], out["ref"]), (3, "forecast", 102.0))
        # the 5-session cone from the close on 10-15 to its exit; the 21-session one is scored: its real exit is used
        self.assertEqual(out["cones"], [[5, "cyan", 2, 7, None, False], [21, "magenta", 1, 5, 95.0, True]])
        self.assertEqual(out["extent"], [92.0, 112.0, None, 80.0, 120.0, 95.0])
        self.assertEqual(out["tip"][0], ["5 sessions after 2026-10-15", "exit 2026-10-22", "t-cyan"])
        self.assertEqual(out["tip"][1], ["68% band", "97.00 – 108", ""])
        self.assertEqual(out["tipStart"][0], ["Close", "102.00", ""])

    def test_the_bands_are_drawn_on_the_price_canvas(self):
        out = run_app("""
          const calls2 = [];
          const ctx = new Proxy({}, { get: (o, k) => (k in o ? o[k] : k === "measureText" ? () => ({ width: 20 })
                                        : k === "createLinearGradient" ? () => ({ addColorStop() {} })
                                        : (...a) => calls2.push([k, ...a.map((x) => (typeof x === "number" ? Math.round(x) : x))])),
                                      set: (o, k, v) => { o[k] = v; return true; } });
          const canvas = () => { const c = document.createElement("canvas"); c.getContext = () => ctx; return c; };
          const st = { wrap: { clientWidth: 800, clientHeight: 300 }, base: canvas(), over: canvas(),
                       model: app("forecastModel")(input.d) };
          app("drawPrice")(st);
          return { fills: calls2.filter((c) => c[0] === "fill").length, texts: calls2.filter((c) => c[0] === "fillText").map((c) => c[1]),
                   arcs: calls2.filter((c) => c[0] === "arc").length, lo: st.geo.lo, hi: st.geo.hi };
        """, d=self.FORECAST)
        self.assertGreaterEqual(out["fills"], 5)  # two cones per forecast, the close's area, the realised dot
        for label in ("97.00", "108", "90.00", "112"):  # each 68% band's edges, labelled at its exit
            self.assertIn(label, out["texts"])
        self.assertEqual(out["arcs"], 1)  # the scored forecast's close at its exit
        self.assertLessEqual(out["lo"], 80.0)  # the axis makes room for every band
        self.assertGreaterEqual(out["hi"], 120.0)

    def test_forecast_view(self):
        out = run_app("""
          const p = new (app("Pane"))(0);
          p.show({ ok: true, fn: "FCST", ticker: "AAPL", args: [], view: "forecast", title: "AAPL — Apple Inc. · forecast", data: input.d });
          const els = all(p.body);
          const cls = (c) => els.filter((e) => e.className === c).map((e) => e.textContent);
          const meter = els.find((e) => e.className === "fc-meter");
          const fill = meter.childNodes.find((e) => String(e.className).startsWith("fc-fill"));
          return { heads: cls("sec-h"), chance: cls("fc-chance"), llm: cls("fc-llm"), job: cls("dec-job"),
                   record: cls("fc-record"), legend: cls("legend"), charts: els.filter((e) => String(e.className).includes("fc-chart")).length,
                   fill: [fill.className, fill.style.left, fill.style.width], text: p.body.textContent };
        """, d=self.FORECAST)
        self.assertEqual(out["charts"], 1)
        self.assertIn("Next 5 sessions after 2026-10-15F1 · model abcdef12, trained through 2026-10-15", out["heads"])
        self.assertIn("Next 21 sessions after 2026-10-14", " ".join(out["heads"]))
        self.assertIn("Live scoringevery recorded forecast, once its exit has passed", out["heads"])
        self.assertTrue(any(h.startswith("History") for h in out["heads"]))
        self.assertTrue(out["chance"][0].startswith("Chance it beats SPY55%a coin is 50%"), out["chance"])
        self.assertEqual(out["fill"], ["fc-fill up", "50.0%", "5.0%"])  # from the coin to the forecast
        self.assertEqual(out["llm"], ["The model (F1-llm, qwen3:8b): 60%. Mild momentum. [e1e1e1e1]"])
        self.assertEqual(out["job"], [self.FORECAST["job_note"] + ". JOBS"])
        self.assertEqual(out["record"][0], "last 2 months (F1a): not run yet2019-2026 (F1b): not run yet")
        self.assertIn("5 sessions: 68% 97.00 – 108, 90% 92.00 – 112", out["legend"][0])
        for text in ("paper only · not financial advice", "pending: the window has not closed", "mom_12_1",
                     "closed at 95.00 on 2026-10-20: inside the 68% band", "(-4.9% .. +5.9%)"):
            self.assertIn(text, out["text"])

    def test_forecasts_view(self):
        d = {"latest": {"columns": [{"key": "ticker", "label": "Stock", "type": "ticker"}],
                        "rows": [{"ticker": "AAPL", "_ticker": "AAPL", "_cmds": {"ticker": "AAPL FCST"}}]},
             "live": {"columns": [], "rows": [], "note": "No forecasts recorded yet."},
             "record": [{"horizon": 5, "lines": ["last 2 months (F1a): not run yet"]}, {"horizon": 21, "lines": ["x"]}],
             "note": "Paper forecasts."}
        out = run_app("""
          const p = new (app("Pane"))(0);
          p.show({ ok: true, fn: "FCST", ticker: null, args: [], view: "forecasts", data: input.d });
          const text = p.body.textContent;
          const link = all(p.body).find((e) => e.className === "cmdlink");
          link.listeners.click[0]({ stopPropagation() {} });
          await settle();
          return { text, ran: calls.filter((c) => c.path === "/api/run").map((c) => c.cmd) };
        """, d=d)
        for text in ("Latest forecast per stock", "Live scoring", "The declared tests", "5 sessions",
                     "last 2 months (F1a): not run yet", "No forecasts recorded yet.", "Paper forecasts."):
            self.assertIn(text, out["text"])
        self.assertEqual(out["ran"], ["AAPL FCST"])  # a row's command reads the forecast; it starts no job

    def test_show_says_when_inputs_are_withheld(self):
        fact = {"id": "d0d0d0d0d0d0d0d0d0d0", "entity": "XCO", "attr": "decision", "value": "NO_EDGE", "as_of": "2026-10-15",
                "known_at": "2026-10-15T23:59:59Z", "tier": "derived", "source": {"name": "brain.decide"}, "context": {},
                "inputs": ["XCO/9de710d29de710d29de7", "XCO/58171cf4e8d0fa5042a5"]}
        inputs = [{"ref": "XCO/9de710d29de710d29de7", "attr": "series.price_daily", "value": "abc", "cite": "9de710d2"}]
        out = run_app("""
          const Pane = app("Pane");
          const p = new Pane(0), q = new Pane(1), r = new Pane(2);
          const res = (data) => ({ ok: true, fn: "SHOW", ticker: "XCO", args: ["d0d0d0d0"], view: "fact", data });
          p.show(res({ fact: input.fact, downgraded: false, inputs: input.inputs, withheld: 1 }));
          q.show(res({ fact: input.fact, downgraded: false, inputs: [], withheld: 2 }));
          r.show(res({ fact: input.fact, downgraded: false, inputs: input.inputs, withheld: 0 }));
          const heads = (x) => all(x.body).filter((e) => e.className === "sec-h").map((e) => e.textContent);
          return { p: p.body.textContent, q: q.body.textContent, r: r.body.textContent, qheads: heads(q) };
        """, fact=fact, inputs=inputs)
        self.assertIn("1 input is not shown: about an aircraft that is no longer tracked", out["p"])
        self.assertIn("[9de710d2]", out["p"])
        self.assertIn("2 inputs are not shown", out["q"])  # every input withheld: the section still says so
        self.assertIn("Inputsthe facts this one was derived from", out["qheads"])
        self.assertNotIn("not shown", out["r"])

    def test_decision_view(self):
        hist = {"columns": [{"key": "as_of", "label": "Date", "type": "date"}, {"key": "action", "label": "Action", "type": "text"},
                            {"key": "status", "label": "Scored", "type": "text"}],
                "rows": [{"as_of": "2026-10-15", "action": "NO_EDGE", "status": "pending", "_cite": "d0d0d0d0"},
                         {"as_of": "2026-06-01", "action": "BUY", "status": "overlapping", "_cite": "d1d1d1d1"}],
                "groups": [], "note": "Scored against SPY."}
        d = {"ticker": "AAPL", "name": "Apple Inc.", "action": "NO_EDGE", "label": "NO_EDGE: do nothing", "direction": 0,
             "score": 0.0, "threshold": 0.5, "signal_score": 0.0, "source_term": {"id": None, "direction": 0, "weight": 0.0},
             "policy": "D1", "policy_sha": "bef26a9d157cbcf5", "policy_current": True, "horizon_days": 21,
             "as_of": "2026-10-15", "cite": "d0d0d0d0", "price": {"date": "2026-10-14", "close": 247.1, "cite": "9de710d2"},
             "items": [{"kind": "signal", "id": "H1-mom12_1", "direction": 0, "weight": 0.0, "why": "FAIL on TEST", "cites": [], "record": None},
                       {"kind": "source", "id": "reuters.com via scout:qwen3:8b", "direction": 1, "weight": 0.0,
                        "why": "2 bets (1 hit) from 3 calls; needs 30 bets",
                        "cites": [{"entity": "AAPL", "cite": "c1c1c1c1"}, {"entity": "AAPL", "cite": "c2c2c2c2"}], "record": {}}],
             "context": [{"entity": "AAPL", "cite": "e1e1e1e1", "attr": "alert", "why": "alert (price_move); alert rules are untested"}],
             "withheld": 1, "lean": {"direction": 1, "up": 1, "down": 0, "sources": 1, "text": "up, 1 of 1 source"},
             "explanation": "Do nothing: no tested evidence carries weight.", "explained_by": "llm:qwen3:8b@3f2a91c0",
             "explain_error": None, "unverified_numbers": ["7%"], "judgment": {"cite": "f1f1f1f1", "error": None},
             "read": {"claims_noted": 2, "pages_read": 5, "error": None},
             "notes": ["No evidence carries weight yet: every hypothesis failed its TEST gate or was not run, and no source has a proven record."],
             "history": hist}
        out = run_app("""
          const Pane = app("Pane");
          const p = new Pane(0), q = new Pane(1), r = new Pane(2);
          p.show({ ok: true, fn: "DEC", ticker: "AAPL", args: [], view: "decision", title: "AAPL — Apple Inc. · decision · NO_EDGE", data: input.d });
          q.show({ ok: true, fn: "DEC", ticker: "MSFT", args: [], view: "decision", data: { ...input.d, ticker: "MSFT", explained_by: "template",
            unverified_numbers: [], explain_error: "a contrary action word (buy)", withheld: 0, policy_current: false,
            read: { claims_noted: 1, pages_read: 1, error: "no model reachable (nothing answered)" },
            context: Array.from({ length: 12 }, (_, k) => ({ entity: "MSFT", cite: `e${k}e${k}e${k}e${k}`.slice(0, 8), attr: "alert", why: "alert (headline)" })) } });
          r.show({ ok: true, fn: "DEC", ticker: null, args: [], view: "decisions", data: { latest: input.hist, scorecard: { columns: [], rows: [], note: "no bets yet" },
            sources: input.hist, note: "Paper decisions. RES has the verdicts." } });
          const cites = all(p.body).filter((e) => e.className === "cite").map((e) => e.textContent);
          const cite = all(p.body).find((e) => e.className === "cite" && e.textContent === "[c2c2c2c2]");
          cite.listeners.click[0]({ stopPropagation() {} });
          await settle();
          const heads = (x) => all(x.body).filter((e) => e.className === "sec-h").map((e) => e.textContent);
          return { text: p.body.textContent, cites, heads: heads(p), warn: all(p.body).filter((e) => e.className === "ask-warn").map((e) => e.textContent),
                   template: q.body.textContent, qheads: heads(q), qwarn: all(q.body).filter((e) => e.className === "ask-warn").length,
                   folded: all(q.body).filter((e) => e.tagName === "SUMMARY").map((e) => e.textContent),
                   qrows: all(q.body).filter((e) => e.tagName === "DETAILS").map((e) => all(e).filter((x) => x.tagName === "TR").length),
                   overview: heads(r), facts: calls.filter((c) => c.path === "/api/fact").map((c) => c.url),
                   rows: all(p.body).filter((e) => e.tagName === "TR").length, view: p.body.className };
        """, d=d, hist=hist)
        self.assertEqual(out["view"], "pane-body view-decision")
        t = out["text"]
        for words in ("NO_EDGE: do nothing", "paper only · not financial advice", "Score 0.00; BUY needs +0.50, SELL -0.50",
                      "policy D1", "as of 2026-10-15", "FAIL on TEST", "2 bets (1 hit) from 3 calls; needs 30 bets",
                      "up, 1 of 1 source", "Do nothing: no tested evidence carries weight.", "No evidence carries weight yet",
                      "5 pages read, 2 claims noted", "1 cited fact is not shown", "Signals 0.00 + source term 0 = score 0.00."):
            self.assertIn(words, t)
        self.assertIn("Leanunproven, not evidence", out["heads"])
        self.assertIn("Explanationmodel-written (llm:qwen3:8b@3f2a91c0); the decision is the table", out["heads"])
        self.assertEqual(len(out["warn"]), 1)
        self.assertIn("1 number in this explanation is in none of the decision's own numbers: 7%", out["warn"][0])
        for c in ("[d0d0d0d0]", "[9de710d2]", "[c1c1c1c1]", "[c2c2c2c2]", "[e1e1e1e1]", "[f1f1f1f1]", "[d1d1d1d1]"):
            self.assertIn(c, out["cites"])  # the decision, its price, each item's refs, the context, the judgment, history
        self.assertEqual(out["facts"], ["AAPL"])
        self.assertIn("Explanationwritten from the table by a template; the decision is the table", out["qheads"])
        self.assertIn("The model's explanation was not used: a contrary action word (buy)", out["template"])
        self.assertIn("1 page read, 1 claim noted: no model reachable (nothing answered)", out["template"])
        self.assertIn("not the code's policy", out["template"])
        self.assertEqual(out["qwarn"], 0)
        self.assertEqual(out["folded"], ["2 more"])  # a week of headline alerts folds after the first 10
        self.assertEqual(out["qrows"], [3])  # the header and the 2 rows under it
        self.assertIn("recorded [f1f1f1f1]", t)
        self.assertEqual(out["overview"], ["Latest decision per stockpaper only · click a stock for its decision",
                                           "Scorecarddecisions scored against SPY as independent bets",
                                           "Source track recordsthe record D1 weighs"])

    def test_news_shows_the_record_d1_weighs_beside_every_call(self):
        claims = [{"time": "2026-10-14T12:00:00Z", "text": "Analyst says buy", "url": "https://a.example/x", "source": "a.example",
                   "tier": "opinion", "direction": 1, "cite": "c1c1c1c1", "record": {"n": 11, "hit_rate": 0.55},
                   "gate_key": "a.example via agent",
                   "gate": {"bets": 3, "hits": 2, "hit_rate": 0.6667, "q": 0.41, "proven": False, "weight": 0.0,
                            "why": "3 bets (2 hits) from 11 calls; needs 30 bets"}},
                  {"time": "2026-10-13T12:00:00Z", "text": "Old note", "url": "https://b.example/y", "source": "b.example",
                   "tier": "reported", "direction": 0, "cite": "c2c2c2c2", "record": None, "gate_key": "b.example via agent", "gate": None}]
        out = run_app("""
          const Pane = app("Pane");
          const p = new Pane(0);
          p.show({ ok: true, fn: "CN", ticker: "AAPL", args: [], view: "news", data: { headlines: [], claims: input.claims } });
          const ths = all(p.body).filter((e) => e.tagName === "TH").map((e) => e.textContent);
          const gates = all(p.body).filter((e) => e.tagName === "TD" && /\\bgate\\b/.test(e.className)).map((e) => [e.textContent, e.title || ""]);
          return { ths, gates };
        """, claims=claims)
        self.assertIn("Every call, overlapping", out["ths"])
        self.assertIn("The record D1 weighs", out["ths"])
        self.assertEqual(out["gates"][0][0], "3 bets · 67% hit · q 0.41 · not proven, weight 03 bets (2 hits) from 11 calls; needs 30 bets")
        self.assertEqual(out["gates"][0][1], "a.example via agent: the record D1 weighs")
        self.assertEqual(out["gates"][1][0], "no record")

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


    # ---------- v2 ----------
    def test_writes_rerun_as_the_list_they_showed(self):
        lines = ["WL ADD NVDA", "wl del aapl", "AAPL WL ADD MSFT", "WL ADD GP", "ALRT ACK 1a2b3c4d", "ALRT ACK ALL",
                 "WL", "ALRT", "SHIP LA-LONG-BEACH", "CAM TFL.00002.00865", "AAPL INS"]
        res = {line: server_res(line) for line in lines}
        out = run_app("""
          const safeRerun = app("safeRerun"), rerunLine = app("rerunLine");
          const o = {};
          for (const line of input.lines) o[line] = [safeRerun(line), rerunLine(line, input.res[line]), rerunLine(line, null)];
          return { o, restored: app("loadLayout")().lines };
        """, lines=lines, res=res, storage={"money-terminal.layout.v1": {
            "panes": ["WL ADD NVDA", "ALRT ACK ALL", "WL DEL AAPL", "AAPL INS"], "focus": 0}})
        for line in ("WL ADD NVDA", "wl del aapl", "AAPL WL ADD MSFT", "WL ADD GP"):
            self.assertEqual(out["o"][line], ["WL"] * 3, line)
        for line in ("ALRT ACK 1a2b3c4d", "ALRT ACK ALL"):
            self.assertEqual(out["o"][line], ["ALRT"] * 3, line)
        for line in ("WL", "ALRT", "SHIP LA-LONG-BEACH", "CAM TFL.00002.00865", "AAPL INS"):
            self.assertEqual(out["o"][line], [line] * 3, line)
        self.assertEqual(out["restored"], ["WL", "ALRT", "WL", "AAPL INS"])

    def test_table_rows_carry_commands_entities_and_severity(self):
        d = {"columns": [{"key": "new", "label": "", "type": "flag"}, {"key": "severity", "label": "Severity", "type": "severity"},
                         {"key": "entity", "label": "Entity", "type": "ticker"}, {"key": "message", "label": "Alert", "type": "text"},
                         {"key": "ack", "label": "", "type": "text"}],
             "rows": [{"new": "NEW", "severity": "high", "entity": "AREA.LA-LONG-BEACH", "message": "30 vessels", "ack": "ACK",
                       "_cite": "abcd1234", "_entity": "AREA.LA-LONG-BEACH",
                       "_cmds": {"entity": "SHIP LA-LONG-BEACH", "ack": "ALRT ACK abcd1234ef01"}},
                      {"new": "", "severity": "notice", "entity": "LA-X", "message": "seen", "ack": None, "_dim": True,
                       "_ticker": "AAPL", "_cmds": {"entity": "SHIP LA-X"}}],
             "groups": [], "note": "INGEST WATCH collects; see RES and ALRT, not RES-X."}
        out = run_app("""
          const runs = [];
          const pane = { run: (line) => runs.push(line) };
          const root = app("dataTable")(input.d, { pane, ticker: null });
          const trs = all(root).filter((e) => e.tagName === "TR" && e.parentNode.tagName === "TBODY");
          const links = all(root).filter((e) => e.className === "cmdlink");
          for (const b of links) b.listeners.click[0]({ stopPropagation() {}, shiftKey: false });
          const cite = all(root).find((e) => e.className === "cite");
          cite.listeners.click[0]({ stopPropagation() {} });
          await settle();
          return { runs, rowCls: trs.map((tr) => tr.className), labels: links.map((b) => b.textContent),
                   sev: all(root).filter((e) => e.tagName === "TD" && /\\bsev\\b/.test(e.className)).map((e) => [e.className, e.textContent]),
                   flags: all(root).filter((e) => e.className === "flag").map((e) => e.textContent),
                   fact: calls.filter((c) => c.path === "/api/fact").length,
                   factUrl: calls.filter((c) => c.path === "/api/fact").map((c) => c.url) };
        """, d=d)
        self.assertEqual(out["rowCls"], ["", "dimrow"])
        # the row command wins over the ticker type: LA-X reads as a ticker but runs SHIP
        self.assertEqual(out["labels"], ["AREA.LA-LONG-BEACH", "ACK", "LA-X", "INGEST WATCH", "RES", "ALRT"])
        self.assertEqual(out["runs"], ["SHIP LA-LONG-BEACH", "ALRT ACK abcd1234ef01", "SHIP LA-X", "INGEST WATCH", "RES", "ALRT"])
        self.assertEqual(out["sev"], [["sev sev-high", "HIGH"], ["sev sev-notice", "NOTICE"]])
        self.assertEqual(out["flags"], ["NEW"])
        self.assertEqual((out["fact"], out["factUrl"]), (1, ["AREA.LA-LONG-BEACH"]))  # the row's entity, not MACRO

    def test_citations_of_ports_and_cameras_look_in_their_entity(self):
        out = run_app("""
          await app("showFact")("AREA.LA-LONG-BEACH", "abcd1234", null);
          await app("showFact")("CAM.CALTRANS.D04.TVB04I880ATTRUCKSCALES", "abcd1234", null);
          await app("showFact")("not an entity!", "abcd1234", null);
          return calls.filter((c) => c.path === "/api/fact").map((c) => c.url);
        """)
        self.assertEqual(out, ["AREA.LA-LONG-BEACH", "CAM.CALTRANS.D04.TVB04I880ATTRUCKSCALES", "MACRO"])

    def test_count_chart(self):
        series = [{"date": "2026-10-12T22:00:00Z", "value": 10, "cite": "a1b2c3d4", "parts": {"cargo": 5, "tanker": 2}},
                  {"date": "2026-10-13T04:00:00Z", "value": 12, "cite": "a1b2c3d5", "parts": {"cargo": 6, "tanker": 3}},
                  {"date": "2026-10-13T10:00:00Z", "value": 30, "cite": "a1b2c3d6", "parts": {"cargo": 15, "tanker": 7}}]
        d = {"entity": "AREA.LA-LONG-BEACH", "name": "LA-LONG-BEACH", "kind": "count", "unit": "vessels", "series": series,
             "stats": {"first": 10, "last": 30, "high": 30, "low": 10, "high_date": series[2]["date"],
                       "low_date": series[0]["date"], "mean": 17.33, "n": 3, "z": 9.9, "z_samples": 2},
             "cite": "a1b2c3d6", "source": {"name": "aisstream.io", "url": "https://aisstream.io/"}}
        out = run_app("""
          const m = app("countModel")(input.d);
          const pane = { body: document.createElement("div"), toolsEl: document.createElement("span"), disposers: [], run() {} };
          app("viewChart")(pane, { ok: true, fn: "SHIP", args: ["LA-LONG-BEACH"], view: "chart", data: input.d });
          const strip = all(pane.body).find((e) => e.className === "chart-stats");
          return { tip: m.tipRows(2), digits: m.digits, ref: m.ref, label: m.label(m.dates[2]), color: m.lines[0].color,
                   ticks: app("timeTicks")(m.dates, 600).map((t) => t.label), tools: pane.toolsEl.childNodes.length,
                   strip: strip.childNodes.map((s) => s.textContent).filter(Boolean) };
        """, d=d)
        self.assertEqual(out["tip"], [["Vessels", "30", ""], ["cargo", "15", "dim"], ["tanker", "7", "dim"]])
        self.assertEqual((out["digits"], out["ref"], out["label"], out["color"]), (0, 17.33, "2026-10-13 10:00", "amber"))
        self.assertEqual(out["ticks"], ["22:00", "04:00", "10:00"])  # samples within a day: labelled by the hour
        self.assertEqual(out["tools"], 0)  # no price ranges on a count chart
        self.assertEqual(out["strip"][:3], ["Last30", "z+9.9", "Mean17.3"])

    def test_count_axis_is_whole_numbers(self):
        out = run_app("""
          const nice = app("niceTicks");
          return { flat: nice(2, 4, 6, 1), wide: nice(0, 300, 6, 1), price: nice(2.94, 3.06, 6) };
        """)
        self.assertEqual((out["flat"]["ticks"], out["flat"]["decimals"]), ([2, 3, 4], 0))  # one value, 3 cars
        self.assertEqual((out["wide"]["step"], out["wide"]["decimals"]), (50, 0))
        self.assertEqual(out["price"]["decimals"], 3)  # prices keep their fractions (step 0.025)

    def test_alert_count_in_the_top_bar(self):
        responses = {"ALRT": server_res("ALRT", data={"columns": [], "rows": [], "unacked": 2}), "WL": server_res("WL")}
        out = run_app("""
          const Pane = app("Pane"), panes = app("panes");
          for (let i = 0; i < 4; i += 1) panes.push(new Pane(i));
          await panes[0].run("ALRT");
          await panes[1].run("WL");
          await settle();
          const el = document.querySelector("#st-alerts");
          const seen = [];
          const show = app("showAlerts");
          show({ unacked: 2, high: 1 }); seen.push([el.textContent, el.className]);
          const before = calls.length;
          show({ unacked: 2, high: 1 });
          const same = calls.length - before;
          show({ unacked: 0, high: 0 }); seen.push([el.textContent, el.className]);
          await settle();
          const reran = calls.slice(before).filter((c) => c.path === "/api/run").map((c) => c.cmd);
          show({ unacked: 3, high: 0 }); seen.push([el.textContent, el.className]);
          await settle();
          const mark = calls.length;
          show({ unacked: 3, high: 0, info: 2 });  // a headline came in: the count stays, the panes refresh
          const infoTitle = el.title, infoText = el.textContent;
          await settle();
          const reranInfo = calls.slice(mark).filter((c) => c.path === "/api/run").map((c) => c.cmd);
          show(null); seen.push([el.textContent, el.className]);
          return { seen, same, reran, infoTitle, infoText, reranInfo };
        """, responses=responses)
        self.assertEqual(out["seen"], [["ALRT 2", "st st-btn bad"], ["ALRT 0", "st st-btn"], ["ALRT 3", "st st-btn warn"],
                                       ["ALRT ?", "st st-btn bad"]])
        self.assertEqual(out["same"], 0)  # no change, no refresh
        self.assertEqual(sorted(out["reran"]), ["ALRT", "WL"])  # a change refreshes the panes that show alerts
        self.assertEqual(out["infoText"], "ALRT 3")  # notice and high only
        self.assertTrue(out["infoTitle"].endswith("(and 2 info: ALRT ALL)"), out["infoTitle"])
        self.assertEqual(sorted(out["reranInfo"]), ["ALRT", "WL"])

    def test_nowrap_columns_cut_long_text(self):
        long = "COST headline: " + "x" * 120
        out = run_app("""
          const root = app("dataTable")({ columns: [{ key: "a", label: "A", type: "text", nowrap: true },
                                                    { key: "b", label: "B", type: "text" }],
                                          rows: [{ a: input.long, b: input.long }] }, { pane: null });
          const tds = all(root).filter((e) => e.tagName === "TD");
          return tds.map((td) => [td.className, td.title || "", td.textContent.length]);
        """, long=long)
        self.assertNotIn("wrap", out[0][0].split())
        self.assertEqual(out[0][1], long)  # the whole text on hover
        self.assertIn("wrap", out[1][0].split())

    def test_alrt_offers_ack_all(self):
        out = run_app("""
          const runs = [];
          const pane = { body: document.createElement("div"), toolsEl: document.createElement("span"), disposers: [],
                         run: (l) => runs.push(l) };
          app("viewTable")(pane, { ok: true, fn: "ALRT", args: [], data: { columns: [], rows: [], unacked: 3, info: 2,
                                                                          note: "2 info alerts: ALRT ALL lists them." } });
          const b = pane.toolsEl.childNodes[0];
          b.listeners.click[0]({ stopPropagation() {} });
          const note = all(pane.body).filter((e) => e.className === "cmdlink").map((e) => e.textContent);
          const every = { body: document.createElement("div"), toolsEl: document.createElement("span"), disposers: [], run() {} };
          app("viewTable")(every, { ok: true, fn: "ALRT", args: ["ALL"], data: { columns: [], rows: [], unacked: 5, info: 2 } });
          const none = { body: document.createElement("div"), toolsEl: document.createElement("span"), disposers: [], run() {} };
          app("viewTable")(none, { ok: true, fn: "ALRT", args: [], data: { columns: [], rows: [], unacked: 0 } });
          return { label: b.textContent, title: b.title, everyTitle: every.toolsEl.childNodes[0].title, runs, note,
                   none: none.toolsEl.childNodes.length };
        """)
        self.assertEqual((out["label"], out["runs"], out["none"]), ("ACK ALL", ["ALRT ACK ALL"], 0))
        self.assertEqual(out["title"], "Acknowledge all 3 unacknowledged alerts, and the 2 info ones not listed (ALRT ACK ALL)")
        self.assertEqual(out["everyTitle"], "Acknowledge all 5 unacknowledged alerts (ALRT ACK ALL)")
        self.assertEqual(out["note"], ["ALRT ALL"])  # the note's ALRT ALL runs as one command

    def test_evidence_groups_rows_by_universe_with_held_and_pending(self):
        row = {"signal": "mom12_1", "hypothesis": "H1-mom12_1", "value": 0.25, "percentile": 80, "in_top_third": True,
               "verdict": "FAIL", "test_net_excess": 0.0011, "perm_p": 0.17, "bh_q": 0.5, "claim": ""}
        broad = [{**row, "signal": "insider_buying", "hypothesis": "H6-insider_buying", "value": 2500000.0,
                  "percentile": None, "in_top_third": None, "held": None, "select": "positive",
                  "universe": "universe/us_broad.json", "universe_size": 741},
                 {**row, "signal": "sue", "hypothesis": "H8-sue", "value": 1.5, "percentile": 90, "in_top_third": True,
                  "held": False, "select": "top_frac", "universe": "universe/us_broad.json", "universe_size": 741}]
        data = {"ticker": "AAPL", "rebalance_date": "2026-09-30", "universe": "universe/us_large.json", "universe_size": 60,
                "signals": [row, *broad], "controls_pass": True, "summary": "0 of 3 signals passed.",
                "broad": {"universe": "universe/us_broad.json", "status": "pending", "universe_size": 741, "computed": 40, "total": 741,
                          "rebalance_date": "2026-09-30"}}
        out = run_app("""
          const pane = { body: document.createElement("div"), toolsEl: document.createElement("span"), disposers: [], run() {} };
          app("viewEvidence")(pane, { ok: true, fn: "EV", ticker: "AAPL", args: [], view: "evidence", data: input.data });
          const trs = all(pane.body).filter((e) => e.tagName === "TR");
          const done = { body: document.createElement("div"), toolsEl: document.createElement("span"), disposers: [], run() {} };
          const only = { ...input.data, signals: [input.data.signals[0]],
                         broad: { universe: "universe/us_broad.json", status: "unavailable", universe_size: 741, reason: "the bank holds 3 of its 741 stocks" } };
          app("viewEvidence")(done, { ok: true, fn: "EV", ticker: "AAPL", args: [], view: "evidence", data: only });
          return { rows: trs.map((tr) => tr.childNodes.map((td) => td.textContent)),
                   meta: all(pane.body).filter((e) => e.className === "ev-meta").map((e) => e.textContent),
                   groups: all(done.body).filter((e) => e.className === "grp").length,
                   note: all(done.body).filter((e) => e.className === "note").map((e) => e.textContent) };
        """, data=data)
        rows = out["rows"]
        self.assertEqual(rows[0][:6], ["Signal", "Hypothesis", "Select", "Value", "Pctl", "Held"])
        self.assertEqual(rows[1], ["us_large · 60 stocks"])
        self.assertEqual(rows[2][:3], ["mom12_1", "H1-mom12_1", "top_frac"])
        self.assertEqual(rows[2][5], "yes")  # H1-H4: held is the top third
        self.assertEqual(rows[3], ["us_broad · 741 stocks"])
        self.assertIn("40 of 741 stocks so far", rows[4][0])
        self.assertEqual((rows[5][0], rows[5][2], rows[5][3], rows[5][4], rows[5][5]),
                         ("insider_buying", "positive", "$2.50M", "—", "—"))  # pending: a value, no rank, no held
        self.assertEqual((rows[6][0], rows[6][5]), ("sue", "no"))  # the row's own held, not the top third
        self.assertIn("us_broad ranking 40/741", out["meta"][0])
        self.assertEqual(out["groups"], 0)  # one universe: no group rows
        self.assertIn("Not ranked against us_broad: the bank holds 3 of its 741 stocks.", out["note"][0])

    def test_answer_flags_unverified_numbers_and_links_only_verified_citations(self):
        res = {"ok": True, "answer": "Revenue rose to $391B [9de710d2]; the margin is 46% [deadbeef0011].",
               "model": "qwen", "usage": {}, "stop_reason": "stop", "tool_calls": [],
               "citations": {"verified": ["9de710d2"], "unverified": ["deadbeef0011", "not an id"]},
               "unverified_numbers": ["46%", {"text": "12.5x"}, 7, "46%", None]}
        out = run_app("""
          const pane = { body: document.createElement("div"), toolsEl: document.createElement("span"), disposers: [], run() {} };
          const turn = (r) => ({ q: "why?", ticker: "AAPL", t0: 0, t1: 1500, res: r, timers: [] });
          const el = app("turnEl")(turn(input.res), pane);
          const cls = (c) => all(el).filter((e) => e.className === c);
          const links = cls("cite");
          links[links.length - 1].listeners.click[0]({ stopPropagation() {} });  // the Sources line's
          await settle();
          const plain = app("turnEl")(turn({ ok: true, answer: "No numbers [9de710d2].", tool_calls: [] }), pane);
          return { warn: cls("ask-warn").map((e) => e.textContent), links: links.map((b) => b.textContent),
                   flagged: cls("cite-bad").map((e) => [e.textContent, e.tagName]),
                   cites: cls("ask-cites").map((e) => e.textContent),
                   facts: calls.filter((c) => c.path === "/api/fact").map((c) => c.url),
                   plain: all(plain).filter((e) => /ask-warn|ask-cites|cite-bad/.test(e.className)).length,
                   plainLinks: all(plain).filter((e) => e.className === "cite").length };
        """, res=res)
        self.assertEqual(len(out["warn"]), 1)
        self.assertIn("3 numbers in this answer are in none of its tool results: 46%, 12.5x, 7.", out["warn"][0])
        self.assertEqual(out["links"], ["[9de710d2]", "[9de710d2]"])  # in the text, and under it
        self.assertEqual(out["flagged"], [["[deadbeef]", "SPAN"], ["[deadbeef]", "SPAN"]])  # never a button
        self.assertTrue(out["cites"][0].startswith("Sources [9de710d2]"))
        self.assertIn("not in this answer's tool results", out["cites"][0])
        self.assertEqual(out["facts"], ["AAPL"])
        self.assertEqual((out["plain"], out["plainLinks"]), (0, 1))  # an answer without the fields: as before

    def test_analyst_busy_is_shown_as_busy(self):
        out = run_app("""
          await app("refreshStatus")();
          const ae = document.querySelector("#st-analyst");
          return { bar: ae.textContent, cls: ae.className, title: ae.title };
        """, status={"ok": True, "jobs_running": 0, "alerts": {"unacked": 0, "high": 0},
                     "analyst": {"available": True, "reason": "busy answering a question", "model": "qwen"}})
        self.assertEqual(out["bar"], "ANALYST BUSY")
        self.assertIn("warn", out["cls"])
        self.assertIn("qwen, busy answering a question", out["title"])

    def test_analyst_off_says_how_to_start_it(self):
        reason = ("nothing answered at http://127.0.0.1:11434/v1 (BRAIN_LLM_URL): start Ollama (`ollama serve`, "
                  "then `ollama pull qwen3:8b`), or set BRAIN_LLM_URL to another OpenAI-compatible server")
        out = run_app("""
          await app("refreshStatus")();
          const ae = document.querySelector("#st-analyst");
          const Pane = app("Pane");
          const off = new Pane(0), on = new Pane(1);
          off.show({ ok: true, fn: "ASK", args: [], view: "ask", data: { available: false, reason: input.reason, model: "qwen3:8b" } });
          on.show({ ok: true, fn: "ASK", args: [], view: "ask", data: { available: true, reason: "", model: "qwen3:8b" } });
          const text = (p) => p.body.textContent;
          return { bar: ae.textContent, title: ae.title, off: text(off), on: text(on),
                   setup: all(off.body).filter((e) => e.className === "ask-setup").length,
                   onSetup: all(on.body).filter((e) => e.className === "ask-setup").length };
        """, reason=reason, status={"ok": True, "jobs_running": 0, "analyst": {"available": False, "reason": reason},
                                    "alerts": {"unacked": 0, "high": 0}})
        self.assertEqual(out["bar"], "ANALYST OFF · OLLAMA / BRAIN_LLM_URL")
        self.assertIn(reason, out["title"])
        self.assertIn(reason, out["off"])
        self.assertEqual((out["setup"], out["onSetup"]), (1, 0))
        for words in ("ollama pull qwen3:8b", "BRAIN_LLM_URL", "BRAIN_LLM_MODEL", "llama.cpp, vLLM, LM Studio"):
            self.assertIn(words, out["off"])
        self.assertIn("qwen3:8b. Answers only from the bank's tools", out["on"])
        self.assertNotIn("Anthropic", out["off"] + out["on"] + out["title"])


if __name__ == "__main__":
    unittest.main()
