/* MONEY TERMINAL front end.
 *
 * Plain DOM, no framework and no build step: the page has to run offline under a strict
 * Content-Security-Policy (script-src 'self', no inline code), so every handler is attached
 * here, and every piece of server text goes in through textContent. The markdown renderer
 * builds DOM nodes from text rather than HTML strings, so there is no innerHTML anywhere in
 * this file. Payload shapes are specified in docs/TERMINAL.md, sections 2 and 5.
 */
"use strict";

(() => {
  // ---------- constants ----------
  const $ = (sel, root = document) => root.querySelector(sel);
  const TOKEN = ($('meta[name="terminal-token"]') || {}).content || "";
  const FUNCS = ["HELP", "DES", "GP", "FA", "CF", "CN", "EV", "BRIEF", "INS", "ERN", "JET", "SHOW", "WEI", "MOV",
    "CURVE", "ECO", "RES", "CLM", "SHIP", "CAM", "WL", "ALRT", "ASK", "INGEST", "JOBS"];
  const SECURITY_FNS = new Set(["DES", "GP", "FA", "CF", "CN", "EV", "BRIEF", "INS", "ERN", "JET"]);
  const RANGES = ["1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "10Y", "MAX"];
  const DEFAULT_LAYOUT = ["WEI", "MOV", "SPY GP 1Y", "ASK"];
  const N_PANES = 4;
  const TICKER_RE = /^[A-Za-z0-9.^=-]{1,15}$/;
  const ALT_ENTITY_RE = /^(?:AREA|CAM)\.[A-Za-z0-9._-]{1,100}$/;  // port areas and cameras: facts, not securities
  const HEX_ID = /^[0-9a-fA-F]{4,20}$/;
  const KEY_LAYOUT = "money-terminal.layout.v1";
  const KEY_HISTORY = "money-terminal.history.v1";
  const HISTORY_MAX = 100;
  const COMPLETE_DELAY = 120;
  const TAPE_EVERY = 60000;
  const STATUS_EVERY = 10000;
  const JOBS_POLL = 1500;  // status poll while jobs run, so a finished job shows promptly
  const ASK_HISTORY = 10;  // messages; the analyst reads no more (analyst.MAX_HISTORY)
  const ASK_MSG = 8000;  // characters per earlier message; the analyst reads no more
  const ASK_BODY = 60000;  // bytes; the server refuses bodies over 64 KB
  const NUMERIC = new Set(["number", "pct", "money", "signed"]);
  const SEVERITIES = new Set(["info", "notice", "high"]);
  const PALETTE = ["amber", "cyan", "magenta", "lime"];
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  const panes = [];
  let focused = 0;
  let booted = false;  // set once the restored layout has been sent: only later runs scroll a pane into view
  let lastTicker = "AAPL";  // the security that HELP's function codes run against

  // ---------- storage (private windows and blocked storage just forget) ----------
  const store = {
    get(key, fallback) {
      try {
        const raw = window.localStorage.getItem(key);
        return raw == null ? fallback : JSON.parse(raw);
      } catch (e) {
        return fallback;
      }
    },
    set(key, value) {
      try {
        window.localStorage.setItem(key, JSON.stringify(value));
      } catch (e) { /* nothing to remember with */ }
    },
  };

  // ---------- DOM helpers ----------
  function h(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }

  function button(label, title, onClick, cls = "btn") {
    const b = h("button", cls, label);
    b.type = "button";
    if (title) {
      b.title = title;
      b.setAttribute("aria-label", title);
    }
    b.addEventListener("click", onClick);
    return b;
  }

  function dash() {
    return h("span", "faint", "—");
  }

  /** Only absolute http(s) URLs become links; anything else is shown as text. */
  function safeUrl(u) {
    if (typeof u !== "string" || !u) return null;
    try {
      const url = new URL(u);
      return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
    } catch (e) {
      return null;
    }
  }

  function extLink(text, url, cls) {
    const href = safeUrl(url);
    if (!href) return h("span", cls || "", text);
    const a = h("a", cls || "", text);
    a.href = href;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.title = href;
    return a;
  }

  function linkLabel(url) {
    try {
      const u = new URL(url);
      const last = u.pathname.split("/").filter(Boolean).pop() || u.hostname;
      return last.length > 30 ? last.slice(0, 29) + "…" : last;
    } catch (e) {
      return "link";
    }
  }

  function section(title, sub) {
    const s = h("div", "sec");
    const head = h("div", "sec-h", title);
    if (sub) head.append(h("span", "sec-sub", sub));
    s.append(head);
    return s;
  }

  // ---------- server ----------
  async function api(path, body) {
    const init = { headers: { "X-Terminal-Token": TOKEN }, cache: "no-store", credentials: "same-origin" };
    if (body !== undefined) {
      init.method = "POST";
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    let resp;
    try {
      resp = await fetch(path, init);
    } catch (e) {
      return { ok: false, error: "cannot reach the terminal server", hint: "", offline: true };
    }
    let data = null;
    try {
      data = await resp.json();
    } catch (e) {
      data = null;
    }
    if (resp.status === 401) {
      return { ok: false, error: "the server did not accept this page's session token; it was probably restarted. Reload the page.",
        hint: "", expired: true };
    }
    if (!resp.ok || !data || typeof data !== "object") {
      return { ok: false, error: (data && data.error) || `server error (HTTP ${resp.status})`, hint: (data && data.hint) || "" };
    }
    return data;
  }

  const runCmd = (line) => api("/api/run?" + new URLSearchParams({ cmd: line }));

  // ---------- formatting ----------
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);

  function fixed(v, d) {
    if (Math.abs(v) < 0.5 * 10 ** -d) v = 0;  // no "-0.00"
    return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  function fmtNum(v, d = 2) {
    return isNum(v) ? fixed(v, d) : null;
  }

  function fmtSigned(v, d = 2) {
    if (!isNum(v)) return null;
    const s = fixed(v, d);
    return !s.startsWith("-") && /[1-9]/.test(s) ? "+" + s : s;
  }

  function fmtPct(v, d = 2, signed = true) {
    if (!isNum(v)) return null;
    const s = fixed(v * 100, d);
    return (signed && !s.startsWith("-") && /[1-9]/.test(s) ? "+" : "") + s + "%";
  }

  function fmtMoney(v) {
    if (!isNum(v)) return null;
    const a = Math.abs(v);
    for (const [div, unit] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]]) {
      if (a >= div) return fixed(v / div, 2) + unit;
    }
    return fixed(v, 2);
  }

  /** up / down / flat from a formatted value, so colour always matches what is shown. */
  function signOf(text) {
    if (typeof text !== "string") return "";
    if (text.startsWith("+")) return "up";
    if (text.startsWith("-")) return "down";
    return "flat";
  }

  function fmtTime(s) {
    if (typeof s !== "string") return s == null ? "" : String(s);
    const m = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})/.exec(s);
    return m ? `${m[1]} ${m[2]}` : s;
  }

  function priceDigits(v) {
    return isNum(v) && Math.abs(v) < 10 ? 4 : 2;
  }

  function tickerOk(t) {
    return typeof t === "string" && TICKER_RE.test(t);
  }

  /** A bank entity a citation can be looked up in: a ticker, or a port area or camera (AREA.X, CAM.Y). */
  function entityOk(t) {
    return tickerOk(t) || (typeof t === "string" && ALT_ENTITY_RE.test(t));
  }

  // ---------- clickable pieces ----------
  function focusedPane() {
    return panes[focused];
  }

  function nextPane(p) {
    return panes[((p ? p.i : focused) + 1) % N_PANES];
  }

  /** A command the user can click: runs in `pane` (Shift: the next pane). */
  function cmdLink(line, pane, label) {
    const b = button(label || line, `Run ${line} (Shift-click: next pane)`, (e) => {
      e.stopPropagation();
      const target = pane || focusedPane();
      (e.shiftKey ? nextPane(target) : target).run(line);
    }, "cmdlink");
    return b;
  }

  function tickerEl(sym, pane, label) {
    const b = button(label || sym, `${sym} DES (Shift-click: next pane)`, (e) => {
      e.stopPropagation();
      const target = pane || focusedPane();
      (e.shiftKey ? nextPane(target) : target).run(`${sym} DES`);
    }, "tk");
    return b;
  }

  function citeEl(id, ticker) {
    if (!id) return null;
    const short = String(id).slice(0, 8);
    const b = button(`[${short}]`, `Fact ${short}: show it and its source`, (e) => {
      e.stopPropagation();
      showFact(ticker, short, b);
    }, "cite");
    return b;
  }

  function verdictEl(v) {
    const s = String(v);
    const cls = /^PASS/i.test(s) ? "pass" : /^FAIL/i.test(s) ? "fail" : "other";
    return h("span", "verdict " + cls, s);
  }

  // ---------- tables ----------
  /* A cell is null (shown as a dash), a string, a Node, or {text|node, cls, title, onClick}. */
  function fillCell(td, c) {
    if (c === null || c === undefined) {
      td.append(dash());
      return;
    }
    if (typeof c === "string" || typeof c === "number") {
      td.textContent = String(c);
      return;
    }
    if (c instanceof Node) {
      td.append(c);
      return;
    }
    if (c.node) td.append(c.node);
    else if (c.text === null || c.text === undefined) td.append(dash());
    else td.textContent = String(c.text);
    if (c.cls) td.classList.add(...c.cls.split(" ").filter(Boolean));
    if (c.title) td.title = c.title;
    if (c.onClick) td.addEventListener("click", c.onClick);
  }

  /** cols: [{label, cls, title}]; rows: [{cells, cls} | {group} | {full: Node, cls}] */
  function gridTable(cols, rows, cls) {
    const table = h("table", "tbl" + (cls ? " " + cls : ""));
    if (cols) {
      const tr = h("tr");
      for (const c of cols) {
        const th = h("th", c.cls || "", c.label);
        if (c.title) th.title = c.title;
        if (c.minCh) table.style.setProperty("--grow-min", c.minCh + "ch");  // the slack column never shrinks below this
        tr.append(th);
      }
      const thead = h("thead");
      thead.append(tr);
      table.append(thead);
    }
    const tbody = h("tbody");
    const span = cols ? cols.length : 1;
    for (const r of rows) {
      const tr = h("tr", r.cls || "");
      if (r.group !== undefined || r.full !== undefined) {
        if (r.group !== undefined) tr.className = "grp";
        const td = h("td");
        td.colSpan = span;
        if (r.group !== undefined) td.append(h("span", "grp-label", r.group));  // sticky: stays in view when scrolled sideways
        else td.append(r.full);
        tr.append(td);
      } else {
        r.cells.forEach((c, i) => {
          const td = h("td", (cols && cols[i] && cols[i].cls) || "");
          fillCell(td, c);
          tr.append(td);
        });
      }
      tbody.append(tr);
    }
    table.append(tbody);
    return table;
  }

  function kvTable(rows) {
    const t = h("table", "kv");
    for (const r of rows) {
      const tr = h("tr");
      tr.append(h("td", "k", r.label));
      const v = h("td", "v" + (r.num ? " num" : "") + (r.cls ? " " + r.cls : ""));
      if (r.value === null || r.value === undefined) v.append(dash());
      else if (r.value instanceof Node) v.append(r.value);
      else v.textContent = String(r.value);
      tr.append(v);
      if (r.extra !== undefined) tr.append(h("td", "x", r.extra || ""));
      const c = h("td", "c");
      const ce = r.cite ? citeEl(r.cite, r.ticker) : null;
      if (ce) c.append(ce);
      tr.append(c);
      t.append(tr);
    }
    return t;
  }

  /** The text column with the longest content (not a timestamp) takes the slack, so numbers line up at the right. */
  function growColumn(cols, rows) {
    let best = -1;
    let bestLen = 0;
    cols.forEach((c, i) => {
      if ((c.type || "text") !== "text") return;
      const vals = rows.map((r) => String(r[c.key] == null ? "" : r[c.key]));
      if (vals.filter((v) => /^\d{4}-\d{2}-\d{2}/.test(v)).length > vals.length / 2) return;
      let total = 0;
      for (const v of vals) total += v.length;
      const avg = rows.length ? total / rows.length : 0;
      if (avg > bestLen) {
        best = i;
        bestLen = avg;
      }
    });
    return best;
  }

  function cellFor(v, type, digits, row, ctx) {
    if (v === null || v === undefined) return null;
    switch (type) {
      case "number":
        return { text: isNum(v) ? fmtNum(v, digits == null ? 2 : digits) : String(v) };
      case "signed": {
        const t = isNum(v) ? fmtSigned(v, digits == null ? 2 : digits) : String(v);
        return { text: t, cls: signOf(t) };
      }
      case "pct": {
        if (!isNum(v)) return { text: String(v) };
        const t = fmtPct(v, digits == null ? 2 : digits);
        return { text: t, cls: signOf(t) };
      }
      case "money":
        return { text: isNum(v) ? fmtMoney(v) : String(v) };
      case "date":
        return { text: fmtTime(String(v)) };
      case "link":
        return safeUrl(v) ? extLink(linkLabel(v) + " ↗", v) : String(v);
      case "ticker": {
        const sym = tickerOk(row._ticker) ? row._ticker : String(v);
        return tickerOk(sym) ? tickerEl(sym, ctx.pane, String(v)) : String(v);
      }
      case "verdict":
        return verdictEl(v);
      case "severity": {
        const s = String(v).toLowerCase();
        return { text: s.toUpperCase(), cls: "sev sev-" + (SEVERITIES.has(s) ? s : "other") };
      }
      case "flag":
        return v === "" ? "" : h("span", "flag", String(v));
      default: {
        const s = typeof v === "object" ? JSON.stringify(v) : String(v);
        return s.length > 48 ? { text: s, cls: "wrap" } : s;
      }
    }
  }

  /** Spec view `table`: columns typed for formatting, rows with `_` metadata, optional groups and note. */
  function dataTable(d, ctx) {
    const cols = Array.isArray(d.columns) ? d.columns : [];
    const rows = Array.isArray(d.rows) ? d.rows : [];
    const wrap = h("div", "tbl-wrap");
    if (!cols.length || !rows.length) {
      wrap.append(d.note ? noteEl(d.note, ctx) : h("div", "empty", "No rows."));
      return wrap;
    }
    const hasLinkCol = cols.some((c) => c.type === "link");
    const showSrc = rows.some((r) => r._cite || (r._url && !hasLinkCol));
    const grow = growColumn(cols, rows);
    const head = cols.map((c, i) => ({
      label: c.label == null ? c.key : c.label,
      cls: [NUMERIC.has(c.type) ? "num" : "", i === grow ? "grow fit" : ""].join(" ").trim(),
      minCh: i === grow ? Math.min(22, Math.max(8, ...rows.map((r) => String(r[c.key] == null ? "" : r[c.key]).length))) + 1 : 0,
    }));
    if (showSrc) head.push({ label: "Src", cls: "src", title: "The fact behind the row: click to show it" });

    // the entity a row's citations belong to: a port area or camera, the row's ticker, or the pane's
    const rowTicker = (r) => (entityOk(r._entity) ? r._entity : tickerOk(r._ticker) ? r._ticker : ctx.ticker);
    const body = [];
    const emit = (r) => {
      const cells = cols.map((c) => {
        const type = r._type && NUMERIC.has(c.type) ? r._type : c.type || "text";
        const digits = r._digits != null && type === "number" ? r._digits : c.digits;
        let cell = cellFor(r[c.key], type, digits, r, ctx);
        if (c.nowrap && cell && typeof cell === "object" && !(cell instanceof Node) && cell.cls === "wrap") {
          cell = { text: cell.text, title: cell.text };  // one line, cut with an ellipsis; the whole text on hover
        }
        const cmd = r._cmds && typeof r._cmds[c.key] === "string" ? r._cmds[c.key] : null;
        if (cmd && r[c.key] != null && r[c.key] !== "") cell = cmdLink(cmd, ctx.pane, String(r[c.key]));
        const cid = r._cites && r._cites[c.key];
        if (cid && cell !== null) {
          const obj = typeof cell === "string" ? { text: cell } : cell instanceof Node ? { node: cell } : cell;
          obj.cls = ((obj.cls || "") + " has-cite").trim();
          obj.title = `Fact ${cid}: click to show it and its source`;
          obj.onClick = (e) => showFact(rowTicker(r), cid, e.currentTarget);
          return obj;
        }
        return cell;
      });
      if (showSrc) {
        const s = h("span", "nowrap");
        const ce = citeEl(r._cite, rowTicker(r));
        if (ce) s.append(ce);
        if (r._url && !hasLinkCol && safeUrl(r._url)) s.append(extLink("↗", r._url, "ext"));
        cells.push(s);
      }
      body.push({ cells, cls: r._dim ? "dimrow" : "" });
    };

    let groups = Array.isArray(d.groups) && d.groups.length ? d.groups.slice() : [];
    if (!groups.length) groups = [...new Set(rows.map((r) => r._group).filter(Boolean))];
    if (groups.length) {
      const known = new Set(groups);
      rows.filter((r) => !known.has(r._group)).forEach(emit);
      for (const g of groups) {
        const rs = rows.filter((r) => r._group === g);
        if (!rs.length) continue;
        body.push({ group: g });
        rs.forEach(emit);
      }
    } else {
      rows.forEach(emit);
    }
    wrap.append(gridTable(head, body));
    if (d.note) wrap.append(noteEl(d.note, ctx));
    return wrap;
  }

  /** Notes are plain text; the INGEST commands they mention, and RES, ALRT and ALRT ALL, are clickable. */
  function noteEl(text, ctx) {
    const el = h("div", "note");
    const re = /\bINGEST (?:MARKETS|MACRO|UNIVERSE|INSIDERS|WATCH)\b|\b(?:RES|ALRT(?: ALL)?)\b(?![-\w])/g;
    const s = String(text);
    let at = 0;
    for (const m of s.matchAll(re)) {
      el.append(document.createTextNode(s.slice(at, m.index)));
      el.append(cmdLink(m[0], ctx && ctx.pane));
      at = m.index + m[0].length;
    }
    el.append(document.createTextNode(s.slice(at)));
    return el;
  }

  // ---------- errors ----------
  function looksLikeCommand(s) {
    const w = String(s).trim().split(/\s+/);
    if (!w[0] || w.length > 16) return false;
    if (FUNCS.includes(w[0])) return true;
    return w.length > 1 && FUNCS.includes(w[1]) && TICKER_RE.test(w[0]) && w[0] === w[0].toUpperCase();
  }

  /** "unknown function 'BAR'" with hint "closest: BRIEF": offer the line with BAR swapped for BRIEF. */
  function fixLine(line, res, code) {
    const m = /unknown function '([^']+)'/.exec((res && res.error) || "");
    if (!m || !line) return code;
    const words = line.trim().split(/\s+/);
    const i = words.findIndex((w) => w.toUpperCase() === m[1].toUpperCase());
    if (i < 0) return code;
    words[i] = code;
    return words.join(" ");
  }

  function errorBox(res, pane, line) {
    res = res || {};
    const box = h("div", "err");
    box.append(h("div", "err-label", "ERROR"));
    box.append(h("div", "err-msg", res.error || "something went wrong"));
    const hint = typeof res.hint === "string" ? res.hint.trim() : "";
    if (hint) {
      const row = h("div", "err-hint");
      const m = /^closest:\s*(.+)$/i.exec(hint);
      if (m) {
        row.append(h("span", "", "Did you mean "));
        for (const code of m[1].split(/[,\s]+/).filter((c) => FUNCS.includes(c.toUpperCase()))) {
          row.append(cmdLink(fixLine(line, res, code.toUpperCase()), pane));
        }
      } else if (looksLikeCommand(hint)) {
        row.append(h("span", "", "Try "), cmdLink(hint, pane));
      } else {
        row.append(h("span", "", hint));
      }
      box.append(row);
    }
    if (res.expired) {
      const row = h("div", "err-hint");
      row.append(button("Reload the page", "Reload with a fresh session token", () => window.location.reload(), "cmdlink"));
      box.append(row);
    }
    return box;
  }

  // ---------- markdown (DOM nodes from text; no HTML is ever parsed) ----------
  const INLINE = new RegExp([
    "(`+)([^`]|[^`][\\s\\S]*?[^`])\\1(?!`)",                // 1,2 code
    "\\*\\*(?=\\S)([\\s\\S]+?)\\*\\*",                     // 3 bold
    "\\[([^\\]\\n]+)\\]\\(((?:[^()\\s]|\\([^()\\s]*\\))+)\\)",   // 4,5 link; the URL may hold (balanced) parentheses
    "\\[([0-9a-fA-F]{8,20})\\]",                           // 6 citation
    "(https?:\\/\\/(?:[^\\s<>()\\[\\]]|\\([^\\s<>()\\[\\]]*\\))*(?:[^\\s<>()\\[\\].,;:!?'\"]|\\([^\\s<>()\\[\\]]*\\)))",  // 7 bare URL, ditto
    "(?<![\\w*])\\*(?=\\S)([^*\\n]+?)\\*(?![\\w*])",         // 8 emphasis
  ].join("|"), "g");

  function inline(parent, text, ctx) {
    const s = String(text);
    let at = 0;
    for (const m of s.matchAll(INLINE)) {
      if (m.index > at) parent.append(document.createTextNode(s.slice(at, m.index)));
      at = m.index + m[0].length;
      if (m[2] !== undefined) parent.append(h("code", "", m[2].replace(/^ (.*) $/, "$1")));
      else if (m[3] !== undefined) {
        const b = h("strong");
        inline(b, m[3], ctx);
        parent.append(b);
      } else if (m[4] !== undefined) parent.append(extLink(m[4], m[5]));
      else if (m[6] !== undefined) {  // an id the analyst's tool results never returned is flagged, not linked
        const id = m[6].toLowerCase();
        parent.append(ctx.unverified && ctx.unverified.has(id) ? unverifiedCite(id) : citeEl(id, ctx.ticker));
      }
      else if (m[7] !== undefined) parent.append(extLink(m[7], m[7]));
      else if (m[8] !== undefined) {
        const em = h("em");
        inline(em, m[8], ctx);
        parent.append(em);
      }
    }
    if (at < s.length) parent.append(document.createTextNode(s.slice(at)));
    return parent;
  }

  // Block patterns run on every line of claims and answers, so none may backtrack: no two runs that
  // can trade the same characters (`\s*\|?\s*`, `.*?\s*#*\s*$`), and `[^\n]` rather than `.`, which
  // stops at U+2028 and makes `$` retry. A heading is parsed without a regex for the same reason.
  const LIST_ITEM = /^(\s*)([-*+]|\d{1,3}[.)])\s+([^\n]*)$/;
  const BLOCK_START = /^\s*(#{1,6}\s|```|>|[-*+]\s|\d{1,3}[.)]\s|\|)/;
  const TABLE_RULE = /^\s*(?:\|\s*)?:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*(?:\|\s*)?$/;

  /** `## Title ##` → {level: 2, text: "Title"}; closing #'s count only after a space (`# C#` keeps its #). */
  function heading(line) {
    const m = /^#{1,6}(?=\s)/.exec(line);
    if (!m) return null;
    let text = line.slice(m[0].length).trim();
    let end = text.length;
    while (end > 0 && text[end - 1] === "#") end -= 1;
    if (end === 0) text = "";
    else if (end < text.length && /\s/.test(text[end - 1])) text = text.slice(0, end).trimEnd();
    return { level: m[0].length, text };
  }

  function splitRow(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
    return s.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
  }

  function mdTable(lines, ctx) {
    const header = splitRow(lines[0]);
    const align = splitRow(lines[1]).map((c) => (/^:-+:$/.test(c) ? "center" : /-:$/.test(c) ? "right" : ""));
    const body = lines.slice(2).map(splitRow);
    const numericCol = header.map((_, i) => body.length > 0 && body.every((r) => /^[-+−]?[$€£]?[\d.,]+\s?[%BMKTx]?$|^[-–—]?$/.test(r[i] || "")) );
    const cols = header.map((c, i) => ({ label: c, cls: align[i] === "right" || numericCol[i] ? "num" : "" }));
    const rows = body.map((r) => ({
      cells: header.map((_, i) => {
        const span = h("span");
        inline(span, r[i] || "", ctx);
        return span;
      }),
    }));
    const wrap = h("div", "md-table");
    const t = gridTable(cols.map((c) => ({ label: "", cls: c.cls })), rows);
    // header cells can carry markdown too
    const ths = t.querySelectorAll("th");
    header.forEach((c, i) => inline(ths[i], c, ctx));
    wrap.append(t);
    return wrap;
  }

  function mdList(lines, i, indent, ctx) {
    const ordered = /^\s*\d/.test(lines[i]);
    const list = h(ordered ? "ol" : "ul");
    while (i < lines.length) {
      const m = LIST_ITEM.exec(lines[i]);
      if (!m) {
        if (/^\s+\S/.test(lines[i]) && list.lastChild && !BLOCK_START.test(lines[i])) {
          list.lastChild.append(document.createTextNode(" "));
          inline(list.lastChild, lines[i].trim(), ctx);
          i += 1;
          continue;
        }
        break;
      }
      const ind = m[1].replace(/\t/g, "    ").length;
      if (ind < indent) break;
      if (ind <= indent + 1 && /^\d/.test(m[2]) !== ordered) break;  // "1." after "-" starts a new list
      if (ind > indent + 1) {
        const [sub, j] = mdList(lines, i, ind, ctx);
        let host = list.lastChild;
        if (!host) {
          host = h("li");
          list.append(host);
        }
        host.append(sub);
        i = j;
        continue;
      }
      const li = h("li");
      inline(li, m[3], ctx);
      list.append(li);
      i += 1;
    }
    return [list, i];
  }

  function renderMarkdown(src, ctx) {
    const root = h("div", "md");
    const lines = String(src == null ? "" : src).replace(/\r\n?/g, "\n").split("\n");
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) {
        i += 1;
        continue;
      }
      let m;
      if (/^\s*```/.test(line)) {
        const code = [];
        i += 1;
        while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
        i += 1;
        const pre = h("pre");
        pre.append(h("code", "", code.join("\n")));
        root.append(pre);
        continue;
      }
      if ((m = heading(line))) {
        root.append(inline(h("h" + m.level), m.text, ctx));
        i += 1;
        continue;
      }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
        root.append(h("hr"));
        i += 1;
        continue;
      }
      if (/^\s*\|/.test(line) && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1])) {
        const block = [line, lines[i + 1]];
        i += 2;
        while (i < lines.length && /^\s*\|/.test(lines[i])) block.push(lines[i++]);
        root.append(mdTable(block, ctx));
        continue;
      }
      if (/^\s*>/.test(line)) {
        const quote = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
        const bq = h("blockquote");
        bq.append(...renderMarkdown(quote.join("\n"), ctx).childNodes);
        root.append(bq);
        continue;
      }
      if ((m = LIST_ITEM.exec(line))) {
        const [list, j] = mdList(lines, i, m[1].length, ctx);
        root.append(list);
        i = j;
        continue;
      }
      const para = [line.trim()];
      i += 1;
      while (i < lines.length && lines[i].trim() && !BLOCK_START.test(lines[i])) para.push(lines[i++].trim());
      root.append(inline(h("p"), para.join(" "), ctx));
    }
    return root;
  }

  // ---------- views ----------
  function viewHelp(p, res) {
    const d = res.data || {};
    const root = h("div", "help");
    const intro = h("div", "help-intro");
    intro.append(document.createTextNode("Type "), h("span", "kbd", "[TICKER] FUNCTION [ARGS]"),
      document.createTextNode(" then Enter. Function first, a bare ticker and US EQUITY <GO> also work. Try: "));
    ["AAPL DES", "AAPL GP 5Y SPY MSFT", "AAPL FA", "AAPL INS", "AAPL ERN", "WEI", "CURVE", "RES", "WL", "ALRT", "ASK"]
      .forEach((ex, k) => intro.append(...(k ? [document.createTextNode(" · ")] : []), cmdLink(ex, p)));
    root.append(intro);

    const cols = h("div", "help-cols");
    const fs = section("Functions", "click a code to run it");
    const fnRows = (d.functions || []).map((f) => ({
      cells: [fnButton(f, p), { text: f.scope, cls: "dim" }, { text: f.args || "", cls: "dim" }, { text: f.desc, cls: "wrap" }],
    }));
    fs.append(gridTable([{ label: "Code" }, { label: "Scope" }, { label: "Args" }, { label: "Description", cls: "grow" }], fnRows));
    const ks = section("Keys");
    ks.append(gridTable([{ label: "Key" }, { label: "Action", cls: "grow" }],
      (d.keys || []).map((k) => ({ cells: [{ node: h("span", "kbd", k.key) }, { text: k.desc, cls: "wrap" }] }))));
    cols.append(fs, ks);
    root.append(cols);
    p.body.append(root);
  }

  function fnButton(f, p) {
    const code = String(f.code || "");
    return button(code, `${code}: ${f.desc || ""}`, (e) => {
      e.stopPropagation();
      if (code === "INGEST" || code === "SHOW") {
        prefill(code === "SHOW" ? `${lastTicker} SHOW ` : "INGEST ");
      } else if (SECURITY_FNS.has(code)) {
        p.run(`${lastTicker} ${code}`);
      } else {
        p.run(code);
      }
    }, "tk");
  }

  function viewDes(p, res) {
    const d = res.data || {};
    const t = d.ticker || res.ticker;
    const root = h("div", "des");
    const top = h("div", "des-top");
    top.append(h("span", "des-tk", t), h("span", "des-name", d.name || ""));
    if (d.kind) top.append(h("span", "badge", d.kind));
    root.append(top);

    const px = d.price;
    if (px) {
      const q = h("div", "des-quote");
      const chg = fmtPct(px.chg_1d);
      q.append(h("span", "des-last", fmtNum(px.close, priceDigits(px.close))));
      if (chg) q.append(h("span", "des-chg " + signOf(chg), chg));
      q.append(h("span", "dim", `close ${px.date || ""}`));
      const ce = citeEl(px.cite, t);
      if (ce) q.append(ce);
      root.append(q);
    }

    const cols = h("div", "des-cols");
    const left = h("div");
    const right = h("div");
    if (px) {
      const s = section("Price", "returns include dividends");
      const pct = (label, v, signed = true) => {
        const txt = fmtPct(v, 2, signed);
        return { label, value: txt, num: true, cls: signed ? signOf(txt) : "" };
      };
      s.append(kvTable([
        pct("1 day", px.chg_1d), pct("5 days", px.chg_5d), pct("1 month", px.ret_1m), pct("3 months", px.ret_3m),
        pct("Year to date", px.ret_ytd), pct("12 months", px.ret_12m),
        { label: "12M high", value: fmtNum(px.high_12m, priceDigits(px.high_12m)), num: true },
        { label: "12M low", value: fmtNum(px.low_12m, priceDigits(px.low_12m)), num: true },
        pct("From 12M high", px.from_high_12m), pct("Volatility 12M", px.vol_12m, false),
      ]));
      left.append(s);
    } else {
      const s = section("Price");
      s.append(h("div", "empty", "No prices in the bank."));
      left.append(s);
    }
    if ((d.identity || []).length) {
      const s = section("Identity", "SEC");
      s.append(kvTable(d.identity.map((r) => ({
        label: r.label,
        value: safeUrl(String(r.value)) ? extLink(String(r.value), String(r.value)) : r.value,
        cite: r.cite, ticker: t,
      }))));
      right.append(s);
    }
    if ((d.metrics || []).length) {
      const s = section("Derived metrics", "written to the bank, citing inputs");
      s.append(kvTable(d.metrics.map((m) => {
        const v = metricText(m);
        const growth = /growth/.test(m.key || "");
        // the date the figure describes: a fiscal-year end, or the price date for market value and the yields
        const asOf = typeof m.as_of === "string" ? m.as_of.slice(0, 10) : "";
        return { label: m.label || m.key, value: v, num: true, cls: growth ? signOf(v) : "", extra: asOf, cite: m.cite, ticker: t };
      })));
      right.append(s);
    }
    if (!right.childNodes.length) {
      const s = section("Identity");
      s.append(h("div", "empty", d.kind && d.kind !== "equity" ? "Market symbol: prices only, no SEC filings." : "No SEC identity facts."));
      right.append(s);
    }
    cols.append(left, right);
    root.append(cols);

    if ((d.filings || []).length) {
      const s = section("Latest reports", "10-K, 10-Q, 8-K and proxy; CF lists every filing");
      s.append(gridTable([{ label: "Date" }, { label: "Form" }, { label: "Description", cls: "grow" }, { label: "Document" }, { label: "", cls: "src" }],
        d.filings.map((f) => ({
          cells: [f.date, { text: f.form, cls: "amber" }, { text: f.desc || "", cls: "wrap" },  // an 8-K's item list wraps
            safeUrl(f.url) ? extLink(linkLabel(f.url) + " ↗", f.url) : null, citeEl(f.cite, t)],
        }))));
      root.append(s);
    }
    if ((d.headlines || []).length) {
      const s = section("Headlines", "reported, unscored");
      s.append(headlineTable(d.headlines, t));
      root.append(s);
    }
    p.body.append(root);
  }

  function metricText(m) {
    if (m.display) return String(m.display) + (m.unit === "x" ? "x" : "");
    if (!isNum(m.value)) return null;
    if (m.unit === "USD") return fmtMoney(m.value);
    if (m.unit === "ratio") return fmtPct(m.value, 2, /growth/.test(m.key || ""));
    if (m.unit === "x") return fixed(m.value, 1) + "x";
    return fmtNum(m.value, 2) + (m.unit ? " " + m.unit : "");
  }

  function headlineTable(list, ticker) {
    return gridTable([{ label: "Time (UTC)" }, { label: "Headline", cls: "grow" }, { label: "Source" }, { label: "", cls: "src" }],
      list.map((x) => ({
        cells: [fmtTime(x.time), { node: extLink(x.title || "(untitled)", x.url, "wrap") , cls: "wrap" }, { text: x.source || "", cls: "dim" },
          citeEl(x.cite, ticker)],
      })));
  }

  function viewTable(p, res) {
    const d = res.data || {};
    if (res.fn === "ALRT" && isNum(d.unacked) && d.unacked > 0) {
      // the default list leaves info alerts out, but ACK ALL acknowledges them too
      const hidden = (res.args || [])[0] !== "ALL" && isNum(d.info) && d.info > 0 ? `, and the ${d.info} info ones not listed` : "";
      p.toolsEl.append(button("ACK ALL", `Acknowledge all ${d.unacked} unacknowledged alerts${hidden} (ALRT ACK ALL)`, (e) => {
        e.stopPropagation();
        p.run("ALRT ACK ALL");
      }, "btn"));
    }
    p.body.append(dataTable(d, { pane: p, ticker: res.ticker }));
  }

  function viewNews(p, res) {
    const d = res.data || {};
    const t = res.ticker;
    const root = h("div", "news");
    const claims = d.claims || [];
    const cs = section("Claims noted from the web", "a direction is not evidence until its source's record beats a coin flip (CLM)");
    if (claims.length) {
      cs.append(gridTable(
        [{ label: "Time (UTC)" }, { label: "Dir" }, { label: "Tier" }, { label: "Claim", cls: "grow" }, { label: "Source" }, { label: "Record", cls: "num" }, { label: "", cls: "src" }],
        claims.map((c) => {
          const dir = c.direction > 0 ? { text: "▲ up", cls: "dir up" } : c.direction < 0 ? { text: "▼ down", cls: "dir down" } : { text: "–", cls: "dir flat" };
          const rec = c.record && isNum(c.record.n)
            ? { text: `${c.record.n} · ${isNum(c.record.hit_rate) ? fmtPct(c.record.hit_rate, 0, false) : "—"}`, cls: "num", title: "claims scored · hit rate" }
            : { text: "no record", cls: "num faint" };
          return { cells: [fmtTime(c.time), dir, { text: c.tier || "", cls: "tier " + (c.tier || "") },
            { node: extLink(c.text || "", c.url), cls: "wrap" }, { text: c.source || "", cls: "dim" }, rec, citeEl(c.cite, t)] };
        })));
    } else {
      cs.append(h("div", "empty", "No claims noted from the web for this security."));
    }
    const hs = section("Headlines", "reported, unscored: a headline is not evidence");
    if ((d.headlines || []).length) hs.append(headlineTable(d.headlines, t));
    else hs.append(h("div", "empty", "No headlines in the bank."));
    root.append(cs, hs);
    p.body.append(root);
  }

  /** EV: the H1-H4 rows against the research universe, then the hypotheses declared on the broad one
   * (H3c, H6, H7b, H8b: a withdrawn hypothesis gives way to its successor), each group under its universe. "Held" is what that hypothesis' backtest would hold: the
   * top third by signal, or for select `positive` every stock above 0. A broad ranking still being filled
   * (`broad.status` pending) shows the stock's value without a rank. */
  function viewEvidence(p, res) {
    const d = res.data || {};
    const b = d.broad && typeof d.broad === "object" ? d.broad : {};
    const root = h("div", "evidence");
    const head = h("div", "ev-head");
    head.append(h("div", "ev-summary", d.summary || ""));
    const meta = h("div", "ev-meta");
    const ctl = d.controls_pass === true ? h("span", "badge ok", "controls pass")
      : d.controls_pass === false ? h("span", "badge bad", "controls failed") : h("span", "badge", "controls n/a");
    const size = (n) => (n == null ? "—" : String(n));
    const uni = (u) => String(u || "").replace(/^.*\//, "").replace(/\.json$/, "");  // universe/us_broad.json -> us_broad
    meta.append(ctl, h("span", "", `Ranked at ${d.rebalance_date || "—"}`),
      h("span", "", `${uni(d.universe) || "Universe"} ${size(d.universe_size)} stocks`));
    if (b.status === "complete") meta.append(h("span", "", `${uni(b.universe)} ${size(b.universe_size)} stocks`));
    else if (b.status === "pending") {
      const pend = h("span", "badge warn", `${uni(b.universe) || "broad"} ranking ${size(b.computed)}/${size(b.total)}`);
      pend.title = "The month's first broad ranking reads every stock's filings; run EV again to go on where it stopped";
      meta.append(pend);
    }
    head.append(meta);
    root.append(head);

    const signals = Array.isArray(d.signals) ? d.signals : [];
    const groups = [[uni(d.universe) || "research universe", d.universe_size, signals.filter((s) => !s.universe)]];
    const broad = signals.filter((s) => s.universe);
    if (broad.length) groups.push([uni(b.universe || broad[0].universe), b.universe_size ?? broad[0].universe_size, broad]);
    const rows = [];
    for (const [name, n, list] of groups) {
      if (groups.length > 1) rows.push({ group: `${name} · ${size(n)} stocks` });
      if (list === broad && b.status === "pending") {
        rows.push({ full: document.createTextNode(`Ranking not complete: ${size(b.computed)} of ${size(b.total)} stocks so far. `
          + "This stock's value is shown without a rank until it is; EV again goes on where it stopped."), cls: "sub" });
      }
      for (const s of list) {
        const pctl = isNum(s.percentile) ? s.percentile : null;
        let rank = null;
        if (pctl !== null) {
          const wrap = h("span", "nowrap");
          wrap.append(document.createTextNode(String(pctl).padStart(3, " ")));
          const bar = h("span", "bar");
          const fill = h("span", "fill" + (s.in_top_third ? " top" : ""));
          fill.style.width = Math.max(0, Math.min(100, pctl)) + "%";
          bar.append(fill);
          wrap.append(bar);
          rank = wrap;
        }
        const held = s.universe ? s.held : s.in_top_third;
        const tn = fmtPct(s.test_net_excess, 2);
        const value = s.signal === "insider_buying" && isNum(s.value) ? "$" + fmtMoney(s.value) : fmtNum(s.value, 4);  // USD bought
        rows.push({ cells: [
          { text: s.signal, cls: "amber" }, s.hypothesis || "", { text: s.select || "top_frac", cls: "dim" },
          { text: value, cls: "num" }, rank,
          held == null ? null : held ? { text: "yes", cls: "amber" } : { text: "no", cls: "dim" },
          verdictEl(s.verdict || "not run"), tn === null ? null : { text: tn, cls: "num " + signOf(tn) },
          { text: fmtNum(s.perm_p, 4), cls: "num" }, { text: fmtNum(s.bh_q, 3), cls: "num" },
        ] });
        if (s.claim) rows.push({ full: document.createTextNode(s.claim), cls: "sub" });
      }
    }
    root.append(gridTable([
      { label: "Signal" }, { label: "Hypothesis" }, { label: "Select", title: "How the backtest picks holdings: top_frac (top third), top_frac_vs_all, positive (every stock above 0)" },
      { label: "Value", cls: "num" }, { label: "Pctl", title: "Percentile in the hypothesis' universe" },
      { label: "Held", title: "What the backtest would hold: the top third by signal; for select positive, every stock above 0" },
      { label: "Verdict" }, { label: "TEST net/mo", cls: "num" }, { label: "Perm p", cls: "num" }, { label: "BH q", cls: "num" },
    ], rows));
    let note = "Percentile is the security's rank on each signal across its hypothesis' universe at the last month-end. "
      + "A verdict is the signal's TEST result after costs against a shuffled-signal null; a high rank on a FAIL signal is not evidence of an edge.";
    if (b.status === "unavailable" && b.reason && b.universe_size) note += ` Not ranked against ${uni(b.universe)}: ${b.reason}.`;
    root.append(noteEl(note, { pane: p }));
    p.body.append(root);
  }

  function viewMarkdown(p, res) {
    p.body.append(renderMarkdown((res.data || {}).markdown, { pane: p, ticker: res.ticker }));
  }

  function factValue(v, unit) {
    if (isNum(v)) {
      const raw = v.toLocaleString("en-US", { maximumFractionDigits: 8 });
      return unit === "USD" && Math.abs(v) >= 1e4 ? `${fmtMoney(v)}  (${raw})` : raw;
    }
    if (v !== null && typeof v === "object") return h("span", "mono-wrap", JSON.stringify(v));
    return v === null || v === undefined ? null : h("span", "mono-wrap", String(v));
  }

  function factBody(d, pane) {
    const f = (d && d.fact) || {};
    const t = entityOk(f.entity) ? f.entity : null;
    const root = h("div", "fact");
    const head = h("div", "fact-head");
    head.append(tickerOk(t) && t !== "MACRO" ? tickerEl(t, pane) : h("span", "amber", f.entity || ""), h("span", "attr", f.attr || ""));
    if (f.tier) head.append(h("span", "badge", f.tier));
    if (d.downgraded) head.append(h("span", "badge bad", "downgraded"));
    root.append(head);

    const src = f.source && typeof f.source === "object" ? f.source : {};
    const srcEl = h("span");
    if (src.name) srcEl.append(document.createTextNode(String(src.name)));
    if (safeUrl(src.url)) {
      if (src.name) srcEl.append(document.createTextNode("  "));
      srcEl.append(extLink(src.url, src.url, "mono-wrap"));
    }
    for (const [k, v] of Object.entries(src)) {
      if (k !== "name" && k !== "url") srcEl.append(h("span", "dim", `  ${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`));
    }
    const rows = [
      { label: "Value", value: factValue(f.value, f.unit) },
      { label: "Unit", value: f.unit || null },
      { label: "As of", value: f.as_of || null },
      { label: "Known at", value: f.known_at || null },
      { label: "Source", value: srcEl.childNodes.length ? srcEl : null },
      { label: "Fact id", value: f.id || null },
      { label: "Retrieved", value: f.retrieved_at || null },
    ];
    const ctx = f.context && typeof f.context === "object" ? f.context : {};
    for (const [k, v] of Object.entries(ctx)) {
      rows.push({ label: k, value: h("span", "mono-wrap dim", typeof v === "string" ? v : JSON.stringify(v)) });
    }
    root.append(kvTable(rows));

    if ((d.inputs || []).length) {
      const s = section("Inputs", "the facts this one was derived from");
      s.append(gridTable([{ label: "Fact" }, { label: "Attribute", cls: "attr" }, { label: "Value", cls: "grow" }],
        d.inputs.map((x) => {
          const ent = String(x.ref || "").split("/")[0];
          let val = x.value;
          if (isNum(val)) val = val.toLocaleString("en-US", { maximumFractionDigits: 6 });
          else if (val !== null && typeof val === "object") val = JSON.stringify(val);
          const sv = val === null || val === undefined ? null : String(val);
          return { cells: [citeEl(x.cite, entityOk(ent) ? ent : t), { node: nameEl(x.attr || "", "mono-wrap"), cls: "amber" },
            sv === null ? null : { node: h("span", "mono-wrap", sv.length > 160 ? sv.slice(0, 159) + "…" : sv), title: sv }] };
        })));
      root.append(s);
    }
    const raw = h("details", "raw");
    raw.append(h("summary", "", "JSON"), h("pre", "json", JSON.stringify(f, null, 2)));
    root.append(raw);
    return root;
  }

  /** A long dotted or camel-case name (us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding) with soft
   * wrap points between its words, so a narrow column breaks it into words rather than letters. */
  function nameEl(text, cls) {
    const el = h("span", cls);
    String(text).split(/(?<=[a-z0-9])(?=[A-Z])|(?<=[:._/])/).forEach((part, k) => {
      if (k) el.append(h("wbr"));
      el.append(document.createTextNode(part));
    });
    return el;
  }

  function viewFact(p, res) {
    p.body.append(factBody(res.data || {}, p));
  }

  // ---------- canvas charts ----------
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim() || "#888";
  const fontFamily = () => getComputedStyle(document.body).fontFamily || "monospace";

  function rgba(hex, a) {
    const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
    if (!m) return hex;
    return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})`;
  }

  /** Size the backing store to CSS pixels x devicePixelRatio, so lines stay sharp on any screen. */
  function setupCanvas(cv, w, hgt) {
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.round(w * dpr));
    const H = Math.max(1, Math.round(hgt * dpr));
    if (cv.width !== W || cv.height !== H) {
      cv.width = W;
      cv.height = H;
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, hgt);
    return ctx;
  }

  /** Round axis values between lo and hi; `minStep` 1 keeps a count's axis to whole numbers. */
  function niceTicks(lo, hi, maxTicks, minStep = 0) {
    const raw = Math.max((hi - lo) / Math.max(1, maxTicks), minStep);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = Math.max(minStep, [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) || 10 * mag);
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(Number(v.toPrecision(12)));
    const str = String(Number(step.toPrecision(6)));
    const decimals = str.includes("e-") ? Number(str.split("e-")[1]) : str.includes(".") ? str.split(".")[1].length : 0;
    return { ticks, step, decimals };
  }

  function crisp(v) {
    return Math.round(v) + 0.5;
  }

  /** Month-boundary ticks for daily data, sized to the width: years, quarters or months; days for short spans. */
  function timeTicks(dates, width) {
    const n = dates.length;
    const out = [];
    if (n < 2) return out;
    const maxTicks = Math.max(2, Math.floor(width / 78));
    const span = (Date.parse(dates[n - 1]) - Date.parse(dates[0])) / 864e5;
    if (!(span >= 75)) {
      // samples taken through the day (ships, cameras) are labelled by the hour over a day or two
      const k = Math.max(1, Math.ceil(n / maxTicks));
      const hours = span < 2 && /T\d{2}:\d{2}/.test(dates[0]);
      let prev = "";
      for (let i = 0; i < n; i += k) {
        const label = hours ? dates[i].slice(11, 16) : `${MONTHS[+dates[i].slice(5, 7) - 1]} ${+dates[i].slice(8, 10)}`;
        if (label !== prev) out.push({ i, label });
        prev = label;
      }
      return out;
    }
    const months = span / 30.44;
    const step = [1, 2, 3, 6, 12, 24, 36, 60, 120].find((s) => months / s <= maxTicks) || 120;
    for (let i = 1; i < n; i += 1) {
      if (dates[i].slice(0, 7) === dates[i - 1].slice(0, 7)) continue;
      const y = +dates[i].slice(0, 4);
      const mo = +dates[i].slice(5, 7) - 1;
      if (step >= 12) {
        if (mo === 0 && y % (step / 12) === 0) out.push({ i, label: String(y) });
      } else if (mo % step === 0) {
        out.push({ i, label: mo === 0 ? String(y) : MONTHS[mo], strong: mo === 0 });
      }
    }
    return out;
  }

  /** One chart: a base canvas redrawn on resize, an overlay redrawn on hover, and a DOM tooltip. */
  function chartFrame(p, cls) {
    const wrap = h("div", "chart-wrap" + (cls ? " " + cls : ""));
    const base = h("canvas", "base");
    const over = h("canvas", "over");
    const tip = h("div", "tip");
    tip.hidden = true;
    base.setAttribute("aria-hidden", "true");
    over.setAttribute("role", "img");
    wrap.append(base, over, tip);
    const st = { wrap, base, over, tip, geo: null, draw: null, hover: null };
    let frame = 0;
    const redraw = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        if (st.draw && wrap.isConnected) {
          st.draw(st);
          tip.hidden = true;
        }
      });
    };
    st.redraw = redraw;
    const ro = new ResizeObserver(redraw);
    ro.observe(wrap);
    redrawers.add(redraw);
    p.disposers.push(() => {
      ro.disconnect();
      cancelAnimationFrame(frame);
      redrawers.delete(redraw);
    });
    const move = (e) => {
      if (!st.geo || !st.hover) return;
      const r = over.getBoundingClientRect();
      st.hover(st, e.clientX - r.left, e.clientY - r.top);
    };
    over.addEventListener("pointermove", move);
    over.addEventListener("pointerdown", move);
    over.addEventListener("pointerleave", () => {
      if (st.geo) setupCanvas(over, st.geo.W, st.geo.H);
      tip.hidden = true;
    });
    return st;
  }

  function placeTip(st, x, y) {
    const { tip, geo } = st;
    tip.hidden = false;
    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    let left = x + 14;
    if (left + tw > geo.plot.x + geo.plot.w) left = x - tw - 14;
    let top = Math.max(geo.plot.y + 2, Math.min(y - th / 2, geo.H - th - 24));
    tip.style.left = Math.max(2, left) + "px";
    tip.style.top = Math.max(2, top) + "px";
  }

  function tipRow(k, v, cls) {
    const r = h("div", "tip-r");
    r.append(h("span", "k", k), h("span", cls || "", v));
    return r;
  }

  function axisTag(ctx, x, y, text, bg, fg) {
    ctx.font = `600 11px ${fontFamily()}`;
    const w = ctx.measureText(text).width + 8;
    ctx.fillStyle = bg;
    ctx.fillRect(x, Math.round(y - 8), w, 16);
    ctx.fillStyle = fg;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillText(text, x + 4, y + 0.5);
    return w;
  }

  // price chart (GP)
  function chartModel(d) {
    const S = Array.isArray(d.series) ? d.series : [];
    const dates = S.map((x) => String(x.date));
    const idx = new Map(dates.map((x, i) => [x, i]));
    const compare = (d.compare || []).filter((c) => c && Array.isArray(c.points) && c.points.length);
    const rebased = compare.length > 0;
    const asLine = (label, color, pts, width, extra) => {
      const values = new Map(pts);
      return { label, color, pts, values, width, ...extra };
    };
    let lines;
    if (rebased) {
      const pts = (d.rebased || []).map((x) => [idx.get(x.date), x.value]).filter(([i, v]) => i !== undefined && isNum(v));
      lines = [asLine(d.ticker, PALETTE[0], pts, 1.8)].concat(compare.map((c, k) => asLine(c.ticker, PALETTE[(k + 1) % PALETTE.length],
        c.points.map((x) => [idx.get(x.date), x.value]).filter(([i, v]) => i !== undefined && isNum(v)), 1.3)));
    } else {
      // coloured by the line it draws (close), not by total return: a dividend payer can fall and still return
      const pts = S.map((x, i) => [i, x.close]).filter(([, v]) => isNum(v));
      const fell = pts.length > 1 && pts[pts.length - 1][1] < pts[0][1];
      lines = [asLine(d.ticker, fell ? "down" : "up", pts, 1.5, { fill: true })];
    }
    const vols = rebased ? null : S.map((x) => (isNum(x.volume) && x.volume > 0 ? x.volume : null));
    return { S, dates, lines, rebased, vols: vols && vols.some((v) => v !== null) ? vols : null, kind: d.kind };
  }

  function drawPrice(st) {
    const m = st.model;
    const W = st.wrap.clientWidth;
    const H = st.wrap.clientHeight;
    if (W < 60 || H < 60 || m.dates.length < 1) return;
    const ctx = setupCanvas(st.base, W, H);
    setupCanvas(st.over, W, H);
    const font = fontFamily();
    let lo = Infinity;
    let hi = -Infinity;
    for (const l of m.lines) for (const [, v] of l.pts) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (m.rebased) {
      lo = Math.min(lo, 100);
      hi = Math.max(hi, 100);
    }
    if (!isFinite(lo)) return;
    const count = m.kind === "count";  // vessels, vehicles: whole numbers, so at least one either side
    const pad = Math.max((hi - lo) * 0.06 || Math.abs(hi) * 0.02 || 1, count ? 1 : 0);
    lo -= pad;
    hi += pad;

    const top = 10;
    const bottom = 20;
    const volH = m.vols ? Math.max(24, Math.round((H - top - bottom) * 0.16)) : 0;
    const gap = m.vols ? 6 : 0;
    const priceH = H - top - bottom - volH - gap;
    const { ticks, decimals } = niceTicks(lo, hi, Math.max(2, Math.floor(priceH / 44)), count ? 1 : 0);
    const dec = decimals;
    const tagDigits = m.rebased ? 1 : m.digits != null ? m.digits : priceDigits(hi);
    ctx.font = `600 11px ${font}`;
    const labelW = Math.max(...ticks.map((v) => ctx.measureText(fixed(v, dec)).width), ctx.measureText(fixed(hi, tagDigits)).width + 8);
    const axisW = Math.ceil(labelW) + 16;
    const plot = { x: 8, y: top, w: W - axisW - 8, h: priceH + gap + volH };
    const n = m.dates.length;
    const xAt = (i) => plot.x + (n <= 1 ? plot.w / 2 : (i / (n - 1)) * plot.w);
    const yAt = (v) => plot.y + (1 - (v - lo) / (hi - lo)) * priceH;
    st.geo = { W, H, plot, priceH, xAt, yAt, lo, hi, n, dec, tagDigits };
    ctx.font = `11px ${font}`;

    // grid and price axis
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#191919";
    ctx.fillStyle = css("dim");
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    for (const v of ticks) {
      const y = crisp(yAt(v));
      if (y < plot.y - 1 || y > plot.y + priceH + 1) continue;
      ctx.beginPath();
      ctx.moveTo(plot.x, y);
      ctx.lineTo(plot.x + plot.w, y);
      ctx.stroke();
      ctx.fillText(fixed(v, dec), plot.x + plot.w + 8, y);
    }
    // time axis
    ctx.textBaseline = "top";
    ctx.textAlign = "center";
    let lastRight = -Infinity;
    for (const t of timeTicks(m.dates, plot.w)) {
      const x = crisp(xAt(t.i));
      ctx.strokeStyle = t.strong ? "#262626" : "#171717";
      ctx.beginPath();
      ctx.moveTo(x, plot.y);
      ctx.lineTo(x, plot.y + plot.h);
      ctx.stroke();
      const w = ctx.measureText(t.label).width;
      const lx = Math.max(x, w / 2 + 1);  // keep the first label inside the canvas
      if (lx - w / 2 < lastRight + 8 || lx + w / 2 > plot.x + plot.w + axisW) continue;
      ctx.fillStyle = t.strong ? css("text") : css("dim");
      ctx.fillText(t.label, lx, plot.y + plot.h + 5);
      lastRight = lx + w / 2;
    }
    // frame
    ctx.strokeStyle = css("rule-strong");
    ctx.beginPath();
    ctx.moveTo(crisp(plot.x + plot.w), plot.y);
    ctx.lineTo(crisp(plot.x + plot.w), plot.y + plot.h);
    ctx.moveTo(plot.x, crisp(plot.y + plot.h));
    ctx.lineTo(plot.x + plot.w, crisp(plot.y + plot.h));
    ctx.stroke();

    // reference: start price, 100 when rebased, or the model's own (a count chart's mean)
    const ref = m.rebased ? 100 : isNum(m.ref) ? m.ref : m.lines[0].pts.length ? m.lines[0].pts[0][1] : null;
    if (ref !== null) {
      ctx.save();
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = "#4a4a4a";
      ctx.beginPath();
      ctx.moveTo(plot.x, crisp(yAt(ref)));
      ctx.lineTo(plot.x + plot.w, crisp(yAt(ref)));
      ctx.stroke();
      ctx.restore();
    }

    // volume
    if (m.vols) {
      const vmax = Math.max(...m.vols.filter((v) => v !== null));
      const vy = plot.y + priceH + gap;
      const bw = Math.max(1, (plot.w / n) * 0.7);
      const up = rgba(css("up"), 0.45);
      const down = rgba(css("down"), 0.45);
      m.vols.forEach((v, i) => {
        if (v === null) return;
        const bh = Math.max(1, (v / vmax) * volH);
        const prev = i > 0 ? m.S[i - 1].close : m.S[i].close;
        ctx.fillStyle = m.S[i].close >= prev ? up : down;
        ctx.fillRect(xAt(i) - bw / 2, vy + volH - bh, bw, bh);
      });
      ctx.strokeStyle = "#1c1c1c";
      ctx.beginPath();
      ctx.moveTo(plot.x, crisp(vy - gap / 2));
      ctx.lineTo(plot.x + plot.w, crisp(vy - gap / 2));
      ctx.stroke();
    }

    // lines (comparisons first, main on top)
    for (const l of m.lines.slice().reverse()) {
      if (!l.pts.length) continue;
      const color = css(l.color);
      ctx.beginPath();
      l.pts.forEach(([i, v], k) => (k ? ctx.lineTo(xAt(i), yAt(v)) : ctx.moveTo(xAt(i), yAt(v))));
      ctx.lineJoin = "round";
      ctx.lineWidth = l.width;
      ctx.strokeStyle = color;
      ctx.stroke();
      if (l.fill) {
        const g = ctx.createLinearGradient(0, plot.y, 0, plot.y + priceH);
        g.addColorStop(0, rgba(color, 0.2));
        g.addColorStop(1, rgba(color, 0));
        ctx.lineTo(xAt(l.pts[l.pts.length - 1][0]), plot.y + priceH);
        ctx.lineTo(xAt(l.pts[0][0]), plot.y + priceH);
        ctx.closePath();
        ctx.fillStyle = g;
        ctx.fill();
      }
    }
    // last-value tags on the axis, nudged apart
    const tags = m.lines.filter((l) => l.pts.length).map((l) => {
      const v = l.pts[l.pts.length - 1][1];
      return { y: yAt(v), text: fixed(v, tagDigits), color: css(l.color) };
    }).sort((a, b) => a.y - b.y);
    for (let k = 1; k < tags.length; k += 1) if (tags[k].y - tags[k - 1].y < 16) tags[k].y = tags[k - 1].y + 16;
    for (const t of tags) axisTag(ctx, plot.x + plot.w + 1, Math.min(Math.max(t.y, plot.y + 8), plot.y + priceH), t.text, t.color, "#000");
  }

  function hoverPrice(st, x, y) {
    const g = st.geo;
    const m = st.model;
    const ctx = setupCanvas(st.over, g.W, g.H);
    if (x < g.plot.x || x > g.plot.x + g.plot.w || y < g.plot.y || y > g.plot.y + g.plot.h) {
      st.tip.hidden = true;
      return;
    }
    const i = Math.max(0, Math.min(g.n - 1, Math.round(((x - g.plot.x) / g.plot.w) * (g.n - 1))));
    const xi = crisp(g.xAt(i));
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = rgba(css("amber"), 0.6);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xi, g.plot.y);
    ctx.lineTo(xi, g.plot.y + g.plot.h);
    if (y <= g.plot.y + g.priceH) {
      ctx.moveTo(g.plot.x, crisp(y));
      ctx.lineTo(g.plot.x + g.plot.w, crisp(y));
    }
    ctx.stroke();
    ctx.restore();
    for (const l of m.lines) {
      const v = l.values.get(i);
      if (!isNum(v)) continue;
      ctx.beginPath();
      ctx.arc(g.xAt(i), g.yAt(v), 3, 0, Math.PI * 2);
      ctx.fillStyle = css(l.color);
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.stroke();
    }
    if (y <= g.plot.y + g.priceH) {
      const v = g.lo + (1 - (y - g.plot.y) / g.priceH) * (g.hi - g.lo);
      axisTag(ctx, g.plot.x + g.plot.w + 1, y, fixed(v, g.tagDigits), css("amber"), "#000");
    }
    // date tag on the time axis
    const date = m.label ? m.label(m.dates[i]) : m.dates[i];
    ctx.font = `600 11px ${fontFamily()}`;
    const dw = ctx.measureText(date).width + 8;
    const dx = Math.max(g.plot.x, Math.min(g.xAt(i) - dw / 2, g.plot.x + g.plot.w - dw));
    ctx.fillStyle = css("amber");
    ctx.fillRect(dx, g.plot.y + g.plot.h + 2, dw, 16);
    ctx.fillStyle = "#000";
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillText(date, dx + 4, g.plot.y + g.plot.h + 10.5);

    const tip = st.tip;
    tip.replaceChildren(h("div", "tip-d", date));
    if (m.rebased) {
      for (const l of m.lines) {
        const v = l.values.get(i);
        const chg = isNum(v) ? fmtPct(v / 100 - 1) : null;
        tip.append(tipRow(l.label, isNum(v) ? `${fixed(v, 1)}  ${chg}` : "—", "t-" + l.color));
      }
    } else {
      for (const [k, v, cls] of (m.tipRows ? m.tipRows(i) : priceTipRows(m, i))) tip.append(tipRow(k, v, cls));
    }
    placeTip(st, g.xAt(i), y);
  }

  /** Tooltip rows for day i of a single-security chart: the close and its change from the first close
   * (what the line shows), then the total return with dividends, labelled as such, when adj is known. */
  function priceTipRows(m, i) {
    const s = m.S[i] || {};
    const rows = [["Close", isNum(s.close) ? fmtNum(s.close, priceDigits(s.close)) : "—", ""]];
    const first = m.lines[0] && m.lines[0].pts.length ? m.lines[0].pts[0][1] : null;
    const chg = isNum(s.close) && isNum(first) && first ? fmtPct(s.close / first - 1) : null;
    rows.push(["From start", chg || "—", signOf(chg)]);
    const a0 = m.S.find((x) => isNum(x.adj));
    if (a0 && a0.adj && isNum(s.adj)) {
      const tr = fmtPct(s.adj / a0.adj - 1);
      rows.push(["Total return", tr, signOf(tr)]);
    }
    if (isNum(s.volume) && s.volume > 0) rows.push(["Volume", fmtMoney(s.volume), ""]);
    return rows;
  }

  // sample counts (SHIP AREA, CAM ID): the price chart's frame, one point per sample
  function countModel(d) {
    const S = Array.isArray(d.series) ? d.series : [];
    const dates = S.map((x) => String(x.date));
    const pts = S.map((x, i) => [i, x.value]).filter(([, v]) => isNum(v));
    const line = { label: d.name || d.entity || "", color: "amber", pts, values: new Map(pts), width: 1.5, fill: true };
    const unit = String(d.unit || "count");
    return {
      S, dates, lines: [line], rebased: false, vols: null, kind: "count", digits: 0,
      ref: d.stats && isNum(d.stats.mean) ? d.stats.mean : null,
      label: fmtTime,
      tipRows: (i) => {
        const s = S[i] || {};
        const rows = [[unit[0].toUpperCase() + unit.slice(1), isNum(s.value) ? fmtNum(s.value, 0) : "—", ""]];
        for (const [k, v] of Object.entries(s.parts && typeof s.parts === "object" ? s.parts : {})) {
          if (isNum(v)) rows.push([k, fmtNum(v, 0), "dim"]);
        }
        return rows;
      },
    };
  }

  function viewCountChart(p, res) {
    const d = res.data || {};
    const st = d.stats || {};
    const strip = h("div", "chart-stats");
    const kv = (k, v, cls) => {
      const s = h("span");
      s.append(h("span", "k", k), h("span", "v " + (cls || ""), v == null ? "—" : v));
      strip.append(s);
    };
    const model = countModel(d);
    kv("Last", fmtNum(st.last, 0));
    const z = fmtSigned(st.z, 1);
    kv("z", z, z ? signOf(z) : "");
    kv("Mean", fmtNum(st.mean, 1));
    kv("High", st.high_date ? `${fmtNum(st.high, 0)} ${fmtTime(st.high_date)}` : fmtNum(st.high, 0));
    kv("Low", st.low_date ? `${fmtNum(st.low, 0)} ${fmtTime(st.low_date)}` : fmtNum(st.low, 0));
    kv("Samples", isNum(st.n) ? String(st.n) : null);
    const c = citeEl(d.cite, entityOk(d.entity) ? d.entity : null);
    if (c) strip.append(c);
    p.body.append(strip);
    if (!model.dates.length) {
      p.body.append(h("div", "empty", "No samples."));
      return;
    }
    const frame = chartFrame(p);
    frame.model = model;
    frame.draw = drawPrice;
    frame.hover = hoverPrice;
    frame.over.setAttribute("aria-label", `${d.name || d.entity || ""}: ${d.unit || "count"} per sample`);
    p.body.append(frame.wrap);
    const src = d.source && typeof d.source === "object" ? d.source : {};
    const lg = h("div", "legend");
    lg.append(h("span", "dim", `${d.unit || "count"} per sample; dashed line: the mean. z: the newest sample against up to 30 `
      + "earlier comparable ones."));
    if (src.name) lg.append(extLink(String(src.name), src.url, "dim"));
    p.body.append(lg);
  }

  function viewChart(p, res) {
    const d = res.data || {};
    if (d.kind === "count") {
      viewCountChart(p, res);
      return;
    }
    const t = d.ticker || res.ticker;
    const extra = (res.args || []).filter((a) => !RANGES.includes(String(a).toUpperCase()));
    for (const r of RANGES) {
      const b = button(r, `${t} GP ${r}`, (e) => {
        e.stopPropagation();
        p.run([t, "GP", r, ...extra].join(" "));
      }, "btn" + (r === d.range ? " on" : ""));
      p.toolsEl.append(b);
    }
    const st = d.stats || {};
    const strip = h("div", "chart-stats");
    const kv = (k, v, cls) => {
      const s = h("span");
      s.append(h("span", "k", k), h("span", "v " + (cls || ""), v == null ? "—" : v));
      strip.append(s);
    };
    const model = chartModel(d);
    const ret = fmtPct(st.ret);
    kv("Last", fmtNum(st.last, priceDigits(st.last)));
    kv(`${d.range || ""} total return`, ret, signOf(ret));
    kv("High", st.high_date ? `${fmtNum(st.high, priceDigits(st.high))} ${st.high_date}` : fmtNum(st.high));
    kv("Low", st.low_date ? `${fmtNum(st.low, priceDigits(st.low))} ${st.low_date}` : fmtNum(st.low));
    if (model.rebased) kv("Rebased", "100");
    const first = model.dates[0];
    const splits = (d.splits || []).filter((s) => Array.isArray(s) && first && String(s[0]) >= first);
    if (splits.length) kv("Splits", splits.map((s) => `${s[0]} ${s[1]}:1`).join(", "));
    if ((d.missing || []).length) {
      const s = h("span");
      s.append(h("span", "k", "Not in bank"));
      for (const x of d.missing) s.append(cmdLink(`INGEST ${x}`, p));
      strip.append(s);
    }
    const c = citeEl(d.cite, t);
    if (c) strip.append(c);
    p.body.append(strip);

    if (!model.dates.length) {
      p.body.append(h("div", "empty", "No price points in this range."));
      return;
    }
    const frame = chartFrame(p);
    frame.model = model;
    frame.draw = drawPrice;
    frame.hover = hoverPrice;
    frame.over.setAttribute("aria-label", `${t} price chart, ${d.range}`);
    p.body.append(frame.wrap);
    if (model.rebased) {
      const lg = h("div", "legend");
      for (const l of model.lines) {
        const it = h("span", "li");
        it.append(h("span", "sw c-" + l.color), h("span", "t-" + l.color, l.label));
        const last = l.pts.length ? l.pts[l.pts.length - 1][1] : null;
        const chg = isNum(last) ? fmtPct(last / 100 - 1) : null;
        if (chg) it.append(h("span", signOf(chg), chg));
        lg.append(it);
      }
      p.body.append(lg);
    }
  }

  // yield curve (CURVE)
  const CURVE_STYLE = [{ color: "amber", width: 2, dash: [] }, { color: "cyan", width: 1.4, dash: [] }, { color: "magenta", width: 1.3, dash: [4, 3] }];
  const tenorYears = (t) => {
    const m = /([\d.]+)\s*(Mo|Yr)/i.exec(String(t));
    return m ? Number(m[1]) / (/mo/i.test(m[2]) ? 12 : 1) : 1;
  };
  const shortTenor = (t) => String(t).replace(/\s*Mo(nth)?s?$/i, "M").replace(/\s*Yr|\s*Years?$/i, "Y");

  function drawCurve(st) {
    const d = st.model;
    const W = st.wrap.clientWidth;
    const H = st.wrap.clientHeight;
    if (W < 60 || H < 60) return;
    const ctx = setupCanvas(st.base, W, H);
    setupCanvas(st.over, W, H);
    const font = fontFamily();
    const years = d.years;
    const xs = years.map((y) => Math.sqrt(y));
    let lo = Infinity;
    let hi = -Infinity;
    for (const c of d.curves) for (const v of c.yields) if (isNum(v)) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    if (!isFinite(lo)) return;
    const pad = (hi - lo) * 0.12 || 0.25;
    lo -= pad;
    hi += pad;
    const { ticks, decimals } = niceTicks(lo, hi, Math.max(2, Math.floor((H - 40) / 40)));
    const dec = Math.max(2, decimals);
    ctx.font = `11px ${font}`;
    const axisW = Math.ceil(Math.max(40, ...ticks.map((v) => ctx.measureText(fixed(v, dec) + "%").width))) + 16;
    const plot = { x: 14, y: 12, w: W - axisW - 26, h: H - 12 - 24 };
    const x0 = xs[0];
    const x1 = xs[xs.length - 1];
    const xAt = (k) => plot.x + ((xs[k] - x0) / (x1 - x0 || 1)) * plot.w;
    const yAt = (v) => plot.y + (1 - (v - lo) / (hi - lo)) * plot.h;
    st.geo = { W, H, plot, xAt, yAt, lo, hi, n: xs.length, dec };

    ctx.lineWidth = 1;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    for (const v of ticks) {
      const y = crisp(yAt(v));
      if (y < plot.y - 1 || y > plot.y + plot.h + 1) continue;
      ctx.strokeStyle = "#191919";
      ctx.beginPath();
      ctx.moveTo(plot.x, y);
      ctx.lineTo(plot.x + plot.w, y);
      ctx.stroke();
      ctx.fillStyle = css("dim");
      ctx.fillText(fixed(v, dec) + "%", plot.x + plot.w + 8, y);
    }
    ctx.textBaseline = "top";
    ctx.textAlign = "center";
    let lastRight = -Infinity;
    d.tenors.forEach((tn, k) => {
      const x = crisp(xAt(k));
      ctx.strokeStyle = "#151515";
      ctx.beginPath();
      ctx.moveTo(x, plot.y);
      ctx.lineTo(x, plot.y + plot.h);
      ctx.stroke();
      const label = shortTenor(tn);
      const w = ctx.measureText(label).width;
      if (x - w / 2 < lastRight + 6) return;
      ctx.fillStyle = css("dim");
      ctx.fillText(label, x, plot.y + plot.h + 6);
      lastRight = x + w / 2;
    });
    ctx.strokeStyle = css("rule-strong");
    ctx.beginPath();
    ctx.moveTo(crisp(plot.x + plot.w), plot.y);
    ctx.lineTo(crisp(plot.x + plot.w), plot.y + plot.h);
    ctx.moveTo(plot.x, crisp(plot.y + plot.h));
    ctx.lineTo(plot.x + plot.w, crisp(plot.y + plot.h));
    ctx.stroke();

    d.curves.slice().reverse().forEach((c) => {
      const s = CURVE_STYLE[d.curves.indexOf(c)] || CURVE_STYLE[2];
      const color = css(s.color);
      const pts = c.yields.map((v, k) => [k, v]).filter(([, v]) => isNum(v));
      ctx.save();
      ctx.setLineDash(s.dash);
      ctx.lineWidth = s.width;
      ctx.strokeStyle = color;
      ctx.lineJoin = "round";
      ctx.beginPath();
      pts.forEach(([k, v], j) => (j ? ctx.lineTo(xAt(k), yAt(v)) : ctx.moveTo(xAt(k), yAt(v))));
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = color;
      for (const [k, v] of pts) {
        ctx.beginPath();
        ctx.arc(xAt(k), yAt(v), s.width >= 2 ? 2.6 : 2, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    const latest = d.curves[0];
    const lastK = latest.yields.map((v, k) => (isNum(v) ? k : -1)).filter((k) => k >= 0).pop();
    if (lastK !== undefined) axisTag(ctx, plot.x + plot.w + 1, yAt(latest.yields[lastK]), fixed(latest.yields[lastK], 2) + "%", css("amber"), "#000");
  }

  function hoverCurve(st, x, y) {
    const g = st.geo;
    const d = st.model;
    const ctx = setupCanvas(st.over, g.W, g.H);
    if (x < g.plot.x - 10 || x > g.plot.x + g.plot.w + 10 || y < g.plot.y || y > g.plot.y + g.plot.h) {
      st.tip.hidden = true;
      return;
    }
    let k = 0;
    for (let j = 1; j < g.n; j += 1) if (Math.abs(g.xAt(j) - x) < Math.abs(g.xAt(k) - x)) k = j;
    const xk = crisp(g.xAt(k));
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = rgba(css("amber"), 0.6);
    ctx.beginPath();
    ctx.moveTo(xk, g.plot.y);
    ctx.lineTo(xk, g.plot.y + g.plot.h);
    ctx.stroke();
    ctx.restore();
    const tip = st.tip;
    tip.replaceChildren(h("div", "tip-d", d.tenors[k]));
    const ref = d.curves[0].yields[k];
    d.curves.forEach((c, j) => {
      const v = c.yields[k];
      const s = CURVE_STYLE[j] || CURVE_STYLE[2];
      if (isNum(v)) {
        ctx.beginPath();
        ctx.arc(g.xAt(k), g.yAt(v), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = css(s.color);
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.stroke();
      }
      let txt = isNum(v) ? fixed(v, 2) + "%" : "—";
      if (j > 0 && isNum(v) && isNum(ref)) txt += `  ${fmtSigned((ref - v) * 100, 0)}bp`;
      tip.append(tipRow(c.label, txt, "t-" + s.color));
    });
    placeTip(st, g.xAt(k), y);
  }

  function viewCurve(p, res) {
    const d = res.data || {};
    const curves = (d.curves || []).filter((c) => c && Array.isArray(c.yields));
    const tenors = d.tenors || [];
    const lg = h("div", "legend");
    curves.forEach((c, j) => {
      const s = CURVE_STYLE[j] || CURVE_STYLE[2];
      const it = h("span", "li");
      it.append(h("span", "sw c-" + s.color + (s.dash.length ? " dash" : "")), h("span", "t-" + s.color, c.label), h("span", "dim", c.date || ""));
      lg.append(it);
    });
    const c = citeEl(d.cite, "MACRO");
    if (c) lg.append(c);
    p.body.append(lg);
    if (!curves.length || !tenors.length) {
      p.body.append(h("div", "empty", "No curve points."));
      return;
    }
    const years = Array.isArray(d.years) && d.years.length === tenors.length && d.years.every(isNum)
      ? d.years : tenors.map(tenorYears);
    const frame = chartFrame(p, "curve-wrap");
    frame.model = { ...d, curves, years };
    frame.draw = drawCurve;
    frame.hover = hoverCurve;
    frame.over.setAttribute("aria-label", "US Treasury par yield curve");
    p.body.append(frame.wrap);

    // tenor table: each curve, then changes from the older curves in basis points
    const cols = [{ key: "tenor", label: "Tenor", type: "text" }];
    curves.forEach((cv, j) => cols.push({ key: "c" + j, label: cv.label, type: "number", digits: 2 }));
    const deltaLabel = (label) => {
      const m = /^(\d+)\s*(month|year|week|day)/i.exec(label);
      return m ? `Δ ${m[1]}${m[2][0].toUpperCase()} bp` : `Δ ${label} bp`;
    };
    curves.slice(1).forEach((cv, j) => cols.push({ key: "d" + (j + 1), label: deltaLabel(cv.label), type: "signed", digits: 0 }));
    const rows = tenors.map((tn, k) => {
      const r = { tenor: tn };
      curves.forEach((cv, j) => { r["c" + j] = cv.yields[k]; });
      curves.slice(1).forEach((cv, j) => {
        const a = curves[0].yields[k];
        const b = cv.yields[k];
        r["d" + (j + 1)] = isNum(a) && isNum(b) ? Math.round((a - b) * 10000) / 100 : null;
      });
      return r;
    });
    const tables = h("div", "curve-tables");
    const left = section("Par yields", "percent");
    left.append(dataTable({ columns: cols, rows }, { pane: p, ticker: "MACRO" }));
    const right = section("Spreads", "percentage points; negative = inverted");
    const sp = (d.spreads || []).map((s) => ({ label: s.label, latest: s.latest, m1: s.m1, y1: s.y1 }));
    right.append(dataTable({ columns: [{ key: "label", label: "Spread", type: "text" }, { key: "latest", label: "Latest", type: "signed", digits: 2 },
      { key: "m1", label: "1M ago", type: "signed", digits: 2 }, { key: "y1", label: "1Y ago", type: "signed", digits: 2 }], rows: sp }, { pane: p }));
    tables.append(left, right);
    p.body.append(tables);
  }

  // ---------- the analyst pane ----------
  function viewAsk(p, res) {
    const d = res.data || {};
    const chat = p.chat;
    if (tickerOk(d.ticker)) chat.ticker = d.ticker.toUpperCase();
    const root = h("div", "ask");
    const status = h("div", "ask-status" + (d.available ? "" : " off"));
    if (d.available) {
      status.append(h("span", "badge ok", "analyst ready"),
        h("span", "", (d.model ? `${d.model}. ` : "") + "Answers only from the bank's tools and cites fact ids."
          + (chat.ticker ? ` Security in focus: ${chat.ticker}.` : "")));
    } else {
      status.append(h("span", "badge bad", "analyst unavailable"), h("span", "", d.reason || "not configured"));
    }
    chat.setup = d.available ? null : setupEl(d.model);
    const log = h("div", "ask-log");
    const form = h("div", "ask-form");
    const input = h("textarea", "ask-input");
    input.rows = 2;
    input.maxLength = 4000;
    input.placeholder = "Ask about the bank, e.g. How have AAPL's margins and free cash flow changed? (Enter to send, Shift+Enter for a new line)";
    input.setAttribute("aria-label", "Question for the analyst");
    const send = button("ASK", "Send the question (Enter)", () => submit(), "ask-send");
    const clear = button("CLEAR", "Clear this conversation", () => {
      if (chat.busy) return;
      chat.turns = [];
      paint();
    }, "ask-clear");
    const actions = h("div", "ask-actions");
    actions.append(send, clear);
    form.append(h("span", "qp", "Q›"), input, actions);
    root.append(status, log, form);
    p.body.append(root);

    const submit = () => {
      const q = input.value.trim();
      if (!q || chat.busy) return;
      input.value = "";
      askQuestion(p, q);
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        submit();
      }
    });
    const paint = () => {
      log.replaceChildren();
      if (chat.setup) log.append(chat.setup);
      if (!chat.turns.length) log.append(askIntro(p, input));
      for (const t of chat.turns) log.append(turnEl(t, p));
      log.scrollTop = log.scrollHeight;
      send.disabled = !!chat.busy;
    };
    p.ask = { log, input, send, paint };
    p.disposers.push(() => { p.ask = null; });
    paint();
    if (d.question) askQuestion(p, d.question);
  }

  /** How to bring the analyst up: it runs on an open-weight model behind an OpenAI-compatible server. */
  function setupEl(model) {
    const m = model || "qwen3:8b";
    const box = h("div", "ask-setup");
    box.append(h("div", "sec-h", "Start the analyst"));
    const ol = h("ol");
    const li = (...parts) => {
      const el = h("li");
      for (const x of parts) el.append(typeof x === "string" ? document.createTextNode(x) : x);
      ol.append(el);
    };
    li("Ollama (the default): install it from ollama.com, run ", h("code", "", "OLLAMA_CONTEXT_LENGTH=16384 ollama serve"),
      ", then ", h("code", "", `ollama pull ${m}`), ".");
    li("Or another OpenAI-compatible server (llama.cpp, vLLM, LM Studio): set ", h("code", "", "BRAIN_LLM_URL"),
      " to its address ending in /v1 and ", h("code", "", "BRAIN_LLM_MODEL"), " to a model it lists (",
      h("code", "", "BRAIN_LLM_KEY"), " if it wants a key), then restart the terminal.");
    li("Run ASK again: the terminal checks the server every few seconds. Nothing leaves this machine unless BRAIN_LLM_URL points elsewhere.");
    box.append(ol);
    return box;
  }

  function askIntro(p, input) {
    const box = h("div", "ask-intro");
    box.append(document.createTextNode("The analyst reads the memory bank through the same functions as this terminal "
      + "(DES, FA, GP, CF, CN, EV, RES, WEI, CURVE ...) and cites the facts it used as [abcd1234]. "
      + "It quotes research verdicts as they are, FAILs included, and places no orders. Examples:"));
    const t = p.chat.ticker || lastTicker;
    for (const q of [`What do the filings say about ${t}'s margins and free cash flow over five years?`,
      `Does any tested signal favour ${t} right now?`,
      "How has the Treasury curve moved over the last year?"]) {
      box.append(button(q, "Put this question in the box", () => {
        input.value = q;
        input.focus();
      }, "cmdlink"));
    }
    return box;
  }

  function turnEl(turn, p) {
    const el = h("div", "turn");
    const q = h("div", "turn-q");
    q.append(h("span", "qp", "Q›"), h("span", "", turn.q));
    el.append(q);
    const a = h("div", "turn-a");
    const res = turn.res;
    if (!res) {
      const th = h("div", "thinking");
      const el2 = h("span", "elapsed");
      th.append(document.createTextNode("thinking"), h("span", "dots"), el2);
      const tick = () => { el2.textContent = Math.round((Date.now() - turn.t0) / 1000) + "s"; };
      tick();
      const timer = setInterval(() => {
        if (turn.res) clearInterval(timer);
        else tick();
      }, 1000);
      turn.timers.push(timer);
      a.append(th);
    } else if (res.ok) {
      const cites = res.citations && typeof res.citations === "object" ? res.citations : {};
      const unverified = factIds(cites.unverified);
      const ctx = { pane: p, ticker: turn.ticker || "MACRO", unverified: new Set(unverified) };
      a.append(renderMarkdown(res.answer || "", ctx));
      const nums = answerNumbers(res.unverified_numbers);
      if (nums.length) a.append(numbersWarning(nums));
      const verified = factIds(cites.verified).filter((x) => !ctx.unverified.has(x));
      if (verified.length || unverified.length) a.append(sourcesEl(verified, unverified, ctx));
    } else {
      a.append(errorBox(res, p));
    }
    const calls = res && Array.isArray(res.tool_calls) ? res.tool_calls : [];
    if (calls.length) {
      const det = h("details", "tools");
      det.append(h("summary", "", `${calls.length} tool call${calls.length === 1 ? "" : "s"}`));
      const ol = h("ol");
      for (const c of calls) {
        const li = h("li");
        const cmd = typeof c.input === "string" ? c.input : JSON.stringify(c.input);
        li.append(h("span", "tn", c.name || "tool"));
        if (typeof c.input === "string" && looksLikeCommand(c.input)) li.append(cmdLink(c.input, nextPane(p), c.input));
        else li.append(h("span", "", cmd));
        ol.append(li);
      }
      det.append(ol);
      a.append(det);
    }
    if (res && res.ok) {
      const meta = h("div", "turn-meta");
      const u = res.usage || {};
      if (res.model) meta.append(h("span", "", res.model));
      if (isNum(u.input_tokens) || isNum(u.output_tokens)) {
        meta.append(h("span", "", `${isNum(u.input_tokens) ? u.input_tokens.toLocaleString("en-US") : "?"} in · ${isNum(u.output_tokens) ? u.output_tokens.toLocaleString("en-US") : "?"} out`));
      }
      if (res.stop_reason) meta.append(h("span", "", res.stop_reason));
      if (turn.t1) meta.append(h("span", "", `${((turn.t1 - turn.t0) / 1000).toFixed(1)}s`));
      a.append(meta);
    }
    el.append(a);
    return el;
  }

  const MAX_SHOWN_IDS = 40;
  const MAX_SHOWN_NUMBERS = 12;

  /** Fact ids from the analyst's `citations` lists: hex strings, lower case, each once. */
  function factIds(list) {
    if (!Array.isArray(list)) return [];
    return [...new Set(list.filter((x) => typeof x === "string" && HEX_ID.test(x)).map((x) => x.toLowerCase()))];
  }

  /** The analyst's `unverified_numbers`: numbers written in the answer that none of its tool results hold,
   * as strings ("4.95T", "23.1%"); a number, or an object with text, number or value, is read too. */
  function answerNumbers(list) {
    if (!Array.isArray(list)) return [];
    const out = [];
    for (const x of list) {
      const v = typeof x === "string" || typeof x === "number" ? x
        : x && typeof x === "object" ? (x.text ?? x.number ?? x.value) : null;
      const t = v === null || v === undefined ? "" : String(v).trim();
      if (t && !out.includes(t)) out.push(t);
    }
    return out;
  }

  /** A warning under an answer: numbers it states that its tool results do not hold (made up, or computed). */
  function numbersWarning(nums) {
    const box = h("div", "ask-warn");
    const n = nums.length;
    const shown = nums.slice(0, MAX_SHOWN_NUMBERS).join(", ") + (n > MAX_SHOWN_NUMBERS ? `, and ${n - MAX_SHOWN_NUMBERS} more` : "");
    box.append(h("span", "badge warn", "unverified"),
      h("span", "", ` ${n} number${n === 1 ? "" : "s"} in this answer ${n === 1 ? "is" : "are"} in none of its tool results: `),
      h("span", "nums", shown),
      h("span", "", ". The model may have computed or made them up: check them against the cited facts."));
    return box;
  }

  /** The fact ids an answer cites: those its tool results returned open SHOW; the others are flagged, not linked. */
  function sourcesEl(verified, unverified, ctx) {
    const box = h("div", "ask-cites");
    if (verified.length) {
      box.append(h("span", "dim", "Sources "));
      for (const id of verified.slice(0, MAX_SHOWN_IDS)) box.append(citeEl(id, ctx.ticker), document.createTextNode(" "));
      if (verified.length > MAX_SHOWN_IDS) box.append(h("span", "dim", `+${verified.length - MAX_SHOWN_IDS}`));
    }
    if (unverified.length) {
      const bad = h("span", "warn");
      bad.append(document.createTextNode(`${verified.length ? " · " : ""}not in this answer's tool results (from an earlier answer, or made up): `));
      for (const id of unverified.slice(0, MAX_SHOWN_IDS)) bad.append(unverifiedCite(id), document.createTextNode(" "));
      box.append(bad);
    }
    return box;
  }

  function unverifiedCite(id) {
    const el = h("span", "cite-bad", `[${String(id).slice(0, 8)}]`);
    el.title = `Fact id ${id} is in none of this answer's tool results: from an earlier answer, or made up`;
    return el;
  }

  /** Earlier successful turns as chat history, cut the way the analyst reads them (the last
   * ASK_HISTORY messages, ASK_MSG characters each) and then to fit the server's 64 KB body limit,
   * oldest question-and-answer pair first. One long answer must not make every later question a 413. */
  function askHistory(turns, body) {
    let hist = [];
    for (const t of turns) {
      if (t.res && t.res.ok && t.res.answer) {
        hist.push({ role: "user", content: t.q.slice(0, ASK_MSG) }, { role: "assistant", content: t.res.answer.slice(0, ASK_MSG) });
      }
    }
    hist = hist.slice(-ASK_HISTORY);
    const size = (x) => new TextEncoder().encode(JSON.stringify({ ...body, history: x })).length;
    while (hist.length && size(hist) > ASK_BODY) hist = hist.slice(2);
    return hist;
  }

  async function askQuestion(p, question) {
    const chat = p.chat;
    if (chat.busy) return;
    const body = { question, ticker: chat.ticker || null };
    body.history = askHistory(chat.turns, body);
    const turn = { q: question, ticker: chat.ticker, t0: Date.now(), t1: 0, res: null, timers: [] };
    chat.turns.push(turn);
    chat.busy = true;
    if (p.ask) p.ask.paint();
    const res = await api("/api/ask", body);
    turn.res = res;
    turn.t1 = Date.now();
    turn.timers.forEach(clearInterval);
    chat.busy = false;
    if (p.ask) p.ask.paint();
  }

  const VIEWS = {
    help: viewHelp, des: viewDes, chart: viewChart, table: viewTable, news: viewNews, evidence: viewEvidence,
    markdown: viewMarkdown, fact: viewFact, curve: viewCurve, ask: viewAsk,
  };

  // ---------- panes ----------
  const redrawers = new Set();

  function canonical(res) {
    if (!res || !res.fn) return null;
    const parts = res.ticker ? [res.ticker, res.fn] : [res.fn];
    if (res.fn !== "ASK") parts.push(...(res.args || []));
    return parts.join(" ");
  }

  /** A title without what the command beside it already says: under WEI, "WEI — world markets" reads
   * "world markets"; under AAPL DES, "AAPL — Apple Inc. · description" reads "Apple Inc. · description". */
  function paneTitle(title, cmd) {
    const t = String(title || "");
    for (const lead of String(cmd || "").split(" ").slice(0, 2)) {  // the ticker, or the code (AAPL ASK)
      for (const sep of [" — ", " · "]) {
        if (lead && t.startsWith(lead + sep)) return t.slice(lead.length + sep.length);
      }
    }
    return t;
  }

  const NOISE = new Set(["US", "EQUITY", "EQUITIES", "<GO>", "GO"]);

  /** What a pane may rerun on its own (reload, restore, refresh): never a job start, never a
   * question to the model. Read from the line itself, since a pending line has no result yet.
   * The server takes the function from the first or second word, and `[TICKER] FUNCTION` wins
   * when both are codes (`CF INGEST` ingests CF), so INGEST or ASK in either place counts,
   * whatever the other word is. Dropping every noise word can only move a code forward. */
  function safeRerun(line) {
    const w = String(line).trim().split(/\s+/).map((x) => x.toUpperCase()).filter((x) => !NOISE.has(x));
    if (w[0] === "ASK") return "ASK";  // first on a line, ASK takes the rest as the question
    if (w[0] === "INGEST" || w[1] === "INGEST") return "JOBS";
    if (w[1] === "ASK") return `${w[0]} ASK`;
    // a watchlist edit or an acknowledgement is done once: rerun as the list it showed
    if ((w[0] === "WL" || w[1] === "WL") && (w.includes("ADD") || w.includes("DEL"))) return "WL";
    if ((w[0] === "ALRT" || w[1] === "ALRT") && w.includes("ACK")) return "ALRT";
    return String(line).trim();
  }

  const WRITES = { WL: ["ADD", "DEL"], ALRT: ["ACK"] };

  /** The line a pane reruns: the server's own reading wins once it has answered this line. */
  function rerunLine(line, res) {
    if (res && res.ok && res.fn === "INGEST") return "JOBS";
    if (res && res.ok && WRITES[res.fn] && (res.args || []).length && WRITES[res.fn].includes(String(res.args[0]))) return res.fn;
    return safeRerun(line);
  }

  class Pane {
    constructor(i) {
      this.i = i;
      this.line = "";
      this.res = null;
      this.resLine = null;  // the line `res` answered
      this.seq = 0;
      this.disposers = [];
      this.chat = { turns: [], ticker: null, busy: false };
      this.ask = null;

      this.el = h("section", "pane");
      this.el.setAttribute("aria-label", `Pane ${i + 1}`);
      const head = h("header", "pane-head");
      this.numEl = h("span", "pane-num", String(i + 1));
      this.cmdEl = h("span", "pane-cmd");
      this.titleEl = h("span", "pane-title");
      this.asofEl = h("span", "pane-asof");
      this.toolsEl = h("span", "pane-tools");
      this.toolsEl.addEventListener("wheel", (e) => {  // a narrow header scrolls its range buttons sideways
        const t = this.toolsEl;
        if (!e.deltaY || t.scrollWidth <= t.clientWidth) return;
        t.scrollLeft += e.deltaY;
        e.preventDefault();
      }, { passive: false });
      this.busyEl = h("span", "pane-busy", "■");
      this.busyEl.hidden = true;
      const reload = button("↻", "Rerun this pane's command", (e) => {
        e.stopPropagation();
        this.reload();
      }, "btn icon");
      const max = button("⤢", "Maximise or restore this pane (double-click the header)", (e) => {
        e.stopPropagation();
        toggleMax(this);
      }, "btn icon");
      head.append(this.numEl, this.cmdEl, this.titleEl, this.asofEl, this.toolsEl, this.busyEl, reload, max);
      head.addEventListener("dblclick", (e) => {
        if (!e.target.closest("button")) toggleMax(this);
      });
      this.body = h("div", "pane-body");
      this.el.append(head, this.body);
      this.el.addEventListener("pointerdown", () => focusPane(this.i), true);
    }

    async run(line, quiet = false) {
      line = String(line || "").trim();
      if (!line) return;
      const seq = ++this.seq;
      this.line = line;
      if (!quiet) {
        this.cmdEl.textContent = /^(\S+\s+)?ASK\b/i.test(line) ? "ASK" : line.toUpperCase();
        this.setBusy(true);
      }
      if (!quiet && booted) reveal(this);  // the user sent this here: show it, even below the fold
      const res = await runCmd(line);
      if (res && res.ok && res.fn === "INGEST") jobsStarted(res);  // even if a newer command owns the pane
      if (seq !== this.seq) return;  // a newer command owns this pane
      this.setBusy(false);
      // HELP's codes and the analyst's example questions use the last company shown, not SPY or ^GSPC
      const kind = res && res.data && res.data.kind;
      if (res && res.ok && tickerOk(res.ticker) && res.fn !== "ASK" && (!kind || kind === "equity")) lastTicker = res.ticker;
      const keepScroll = quiet ? this.body.scrollTop : 0;
      this.show(res);
      this.body.scrollTop = keepScroll;
      saveLayout();
    }

    reload(quiet = false) {
      if (this.line) this.run(this.safeLine(), quiet);
    }

    /** This pane's line made safe to rerun, using the result only if it answered this line. */
    safeLine() {
      return rerunLine(this.line, this.resLine === this.line ? this.res : null);
    }

    setBusy(on) {
      this.el.classList.toggle("loading", on);
      this.busyEl.hidden = !on;
    }

    dispose() {
      for (const f of this.disposers.splice(0)) {
        try {
          f();
        } catch (e) { /* already gone */ }
      }
    }

    show(res) {
      this.dispose();
      this.res = res;
      this.resLine = this.line;
      const ok = !!(res && res.ok);
      const view = ok && VIEWS[res.view] ? res.view : "error";
      this.body.className = "pane-body view-" + view;
      this.body.replaceChildren();
      this.toolsEl.replaceChildren();
      const canon = canonical(res);
      this.cmdEl.textContent = canon || this.line.toUpperCase();
      this.cmdEl.title = this.cmdEl.textContent;
      this.titleEl.textContent = ok ? paneTitle(res.title, this.cmdEl.textContent) : "";
      this.titleEl.title = ok ? res.title || "" : "";
      this.asofEl.textContent = ok && res.asof ? `as of ${res.asof}` : "";
      this.asofEl.title = this.asofEl.textContent;  // a narrow header cuts it short
      if (!ok) {
        this.body.append(errorBox(res, this, this.line));
        return;
      }
      if (view === "error") {
        this.body.append(errorBox({ error: `this terminal cannot show view '${res.view}'` }, this));
        return;
      }
      try {
        VIEWS[view](this, res);
      } catch (e) {
        console.error(e);
        this.body.replaceChildren(errorBox({ error: `could not render ${view}: ${e.message}` }, this));
      }
    }

    saveLine() {
      return this.line ? this.safeLine() : "";
    }
  }

  function focusPane(i, show = false) {
    if (!panes[i]) return;
    if (focused !== i) {
      focused = i;
      saveLayout();
    }
    panes.forEach((p, k) => p.el.classList.toggle("focused", k === i));
    $("#target").textContent = "P" + (i + 1);
    if (show) reveal(panes[i]);
  }

  /** In the one-column layout (under 900px) the grid scrolls, so a pane chosen from the keyboard or sent
   * a command can be below the fold: scroll it into view. Not on a click, which is in view already. */
  function reveal(p) {
    const grid = $("#grid");
    if (p && grid && grid.scrollHeight > grid.clientHeight) p.el.scrollIntoView({ block: "nearest" });
  }

  function toggleMax(p) {
    const grid = $("#grid");
    const on = !p.el.classList.contains("max");
    panes.forEach((q) => q.el.classList.remove("max"));
    grid.classList.toggle("maxed", on);
    if (on) p.el.classList.add("max");
  }

  function saveLayout() {
    if (panes.length !== N_PANES) return;
    store.set(KEY_LAYOUT, { panes: panes.map((p) => p.saveLine() || ""), focus: focused });
  }

  function loadLayout() {
    const saved = store.get(KEY_LAYOUT, null);
    const ok = saved && Array.isArray(saved.panes) && saved.panes.length === N_PANES
      && saved.panes.every((s) => typeof s === "string" && s.length <= 500);
    // a layout saved by an older page may hold `CF INGEST` or a question: restoring never reruns those
    const lines = ok ? saved.panes.map((s, i) => safeRerun(s) || DEFAULT_LAYOUT[i]) : DEFAULT_LAYOUT.slice();
    const focus = ok && Number.isInteger(saved.focus) && saved.focus >= 0 && saved.focus < N_PANES ? saved.focus : 0;
    return { lines, focus };
  }

  // ---------- command line, history, completion ----------
  const cmdInput = () => $("#cmd");
  let history = store.get(KEY_HISTORY, []);
  if (!Array.isArray(history)) history = [];
  history = history.filter((s) => typeof s === "string" && s.length <= 500).slice(-HISTORY_MAX);
  let histPos = history.length;
  let histDraft = "";
  const ac = { items: [], sel: -1, moved: false, seq: 0, timer: 0 };

  function pushHistory(line) {
    if (history[history.length - 1] !== line) history.push(line);
    if (history.length > HISTORY_MAX) history = history.slice(-HISTORY_MAX);
    histPos = history.length;
    store.set(KEY_HISTORY, history);
  }

  function prefill(text) {
    const input = cmdInput();
    input.value = text;
    input.focus();
    input.setSelectionRange(text.length, text.length);
    scheduleComplete();
  }

  function submit(target) {
    const input = cmdInput();
    const line = input.value.trim();
    hideAc();
    if (!line) return;
    pushHistory(line);
    input.value = "";
    histDraft = "";
    target.run(line);
  }

  function histStep(delta) {
    const input = cmdInput();
    if (!history.length) return;
    if (histPos === history.length) histDraft = input.value;
    histPos = Math.max(0, Math.min(history.length, histPos + delta));
    input.value = histPos === history.length ? histDraft : history[histPos];
    input.setSelectionRange(input.value.length, input.value.length);
  }

  function acOpen() {
    return !$("#ac").hidden && ac.items.length > 0;
  }

  function hideAc() {
    ac.seq += 1;
    clearTimeout(ac.timer);
    ac.items = [];
    ac.sel = -1;
    ac.moved = false;
    const list = $("#ac");
    list.hidden = true;
    list.replaceChildren();
    cmdInput().setAttribute("aria-expanded", "false");
  }

  function paintAc() {
    const list = $("#ac");
    list.replaceChildren();
    ac.items.forEach((it, k) => {
      const li = h("li", (it.kind === "function" ? "ac-fn" : it.kind === "keyword" ? "ac-kw" : "ac-tk") + (k === ac.sel ? " sel" : ""));
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", k === ac.sel ? "true" : "false");
      li.append(h("span", "ac-text", it.text), h("span", "ac-kind", it.kind || ""), h("span", "ac-desc", it.desc || ""));
      li.addEventListener("mousedown", (e) => e.preventDefault());  // keep focus in the command line
      li.addEventListener("click", () => acceptCompletion(k, true));
      list.append(li);
    });
    list.hidden = ac.items.length === 0;
    cmdInput().setAttribute("aria-expanded", ac.items.length ? "true" : "false");
    const sel = list.children[ac.sel];
    if (sel) sel.scrollIntoView({ block: "nearest" });
  }

  function scheduleComplete() {
    clearTimeout(ac.timer);
    ac.timer = setTimeout(fetchCompletions, COMPLETE_DELAY);
  }

  async function fetchCompletions() {
    const input = cmdInput();
    const q = input.value;
    const seq = ++ac.seq;
    if (!q.trim() || document.activeElement !== input) {
      hideAc();
      return;
    }
    const res = await api("/api/complete?" + new URLSearchParams({ q }));
    if (seq !== ac.seq || input.value !== q || document.activeElement !== input) return;
    ac.items = res && res.ok !== false && Array.isArray(res.items)
      ? res.items.filter((x) => x && typeof x.text === "string").slice(0, 12) : [];
    ac.sel = ac.items.length ? 0 : -1;
    ac.moved = false;
    paintAc();
  }

  function acceptCompletion(k, refetch) {
    const it = ac.items[k];
    if (!it) return;
    const input = cmdInput();
    const m = /^([\s\S]*?)(\S*)$/.exec(input.value);
    input.value = m[1] + it.text + " ";
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
    hideAc();
    if (refetch) scheduleComplete();
  }

  function moveSel(delta) {
    const n = ac.items.length;
    ac.sel = (ac.sel + delta + n) % n;
    ac.moved = true;
    paintAc();
  }

  function bindCommandLine() {
    const input = cmdInput();
    input.addEventListener("input", () => {
      ac.moved = false;
      histPos = history.length;
      scheduleComplete();
    });
    input.addEventListener("blur", () => setTimeout(() => {
      if (document.activeElement !== input) hideAc();
    }, 0));
    input.addEventListener("keydown", (e) => {
      if (e.isComposing) return;
      switch (e.key) {
        case "Enter":
          e.preventDefault();
          if (acOpen() && ac.moved && ac.sel >= 0) acceptCompletion(ac.sel, false);
          submit(e.ctrlKey ? nextPane(focusedPane()) : focusedPane());
          break;
        case "Tab":
          if (acOpen()) {
            e.preventDefault();
            acceptCompletion(Math.max(ac.sel, 0), true);
          }
          break;
        case "ArrowDown":
          e.preventDefault();
          if (acOpen()) moveSel(1);
          else histStep(1);
          break;
        case "ArrowUp":
          e.preventDefault();
          if (acOpen()) moveSel(-1);
          else histStep(-1);
          break;
        case "Escape":
          e.preventDefault();
          input.value = "";
          histPos = history.length;
          hideAc();
          break;
        default:
          break;
      }
    });
    $("#go").addEventListener("click", () => submit(focusedPane()));
  }

  function bindKeys() {
    document.addEventListener("keydown", (e) => {
      if (e.key === "F1") {
        e.preventDefault();
        focusedPane().run("HELP");
        return;
      }
      const digit = /^Digit([1-4])$/.exec(e.code || "") || /^([1-4])$/.exec(e.key);
      if (e.ctrlKey && !e.altKey && !e.metaKey && !e.shiftKey && digit) {
        e.preventDefault();
        focusPane(Number(digit[1]) - 1, true);
        cmdInput().focus();
        return;
      }
      if (e.key === "Escape" && !$("#popover").hidden) {
        closePopover();
        return;
      }
      const t = e.target;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
      if (!typing && !e.ctrlKey && !e.metaKey && !e.altKey && e.key.length === 1 && e.key !== " ") {
        cmdInput().focus();  // type anywhere: keys go to the command line
      }
    });
  }

  // ---------- fact popover (SHOW) ----------
  let popSeq = 0;
  let popAnchor = null;

  function placePopover() {
    const el = $("#popover");
    if (el.hidden) return;
    const r = popAnchor && popAnchor.isConnected ? popAnchor.getBoundingClientRect() : { left: 40, right: 40, top: 40, bottom: 40 };
    const w = el.offsetWidth;
    const hgt = el.offsetHeight;
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    const left = Math.max(8, Math.min(r.left, vw - w - 8));
    let top = r.bottom + 4;
    if (top + hgt > vh - 8) top = Math.max(8, r.top - hgt - 4);
    el.style.left = left + "px";
    el.style.top = top + "px";
  }

  function closePopover() {
    popSeq += 1;
    const el = $("#popover");
    el.hidden = true;
    el.replaceChildren();
    popAnchor = null;
  }

  async function showFact(ticker, id, anchor) {
    if (!HEX_ID.test(String(id))) return;
    const seq = ++popSeq;
    popAnchor = anchor;
    const el = $("#popover");
    const head = h("div", "pop-head");
    const tools = h("span", "pane-tools");
    head.append(h("span", "pop-title", `FACT ${id}`), h("span", "spacer"), tools,
      button("×", "Close (Esc)", closePopover, "btn icon"));
    const body = h("div", "pop-body");
    body.append(h("div", "empty", "loading…"));
    el.replaceChildren(head, body);
    el.hidden = false;
    placePopover();
    const t = entityOk(ticker) ? ticker : "MACRO";
    const res = await api("/api/fact?" + new URLSearchParams({ ticker: t, id }));
    if (seq !== popSeq) return;
    body.replaceChildren(res && res.ok ? factBody(res.data || {}, focusedPane()) : errorBox(res, focusedPane()));
    const ent = res && res.ok && res.data && res.data.fact ? res.data.fact.entity : null;
    if (entityOk(ent)) {
      const line = tickerOk(ent) ? `${ent} SHOW ${id}` : `SHOW ${ent}/${id}`;  // AREA.X and CAM.Y are no tickers
      tools.append(button("OPEN IN PANE", `Run ${line} in pane ${focused + 1}`, () => {
        closePopover();
        focusedPane().run(line);
      }, "btn"));
    }
    placePopover();
  }

  function bindPopover() {
    document.addEventListener("pointerdown", (e) => {
      const el = $("#popover");
      if (!el.hidden && !el.contains(e.target) && !(e.target.closest && e.target.closest(".cite, .has-cite"))) closePopover();
    }, true);
    window.addEventListener("resize", placePopover);
  }

  // ---------- top bar: clocks and status ----------
  const ZONES = [
    { label: "NY", tz: "America/New_York", open: [570, 960] },
    { label: "LDN", tz: "Europe/London", open: [480, 990] },
    { label: "MUM", tz: "Asia/Kolkata", open: [555, 930] },
    { label: "TKY", tz: "Asia/Tokyo", open: [540, 930] },
    { label: "UTC", tz: "UTC", open: null },
  ];

  function buildClocks() {
    const box = $("#clocks");
    for (const z of ZONES) {
      try {
        z.fmt = new Intl.DateTimeFormat("en-GB", { timeZone: z.tz, weekday: "short", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" });
      } catch (e) {
        continue;
      }
      const el = h("span", "clock");
      z.dot = h("span", "dot");
      z.time = h("span", "ct");
      el.append(h("span", "cz", z.label), z.time);
      if (z.open) {
        el.append(z.dot);
        el.title = `${z.tz}. The square is lit during the main exchange's regular hours (weekdays; holidays not included).`;
      } else {
        el.title = "Coordinated Universal Time";
      }
      box.append(el);
    }
    tickClocks();
    setInterval(tickClocks, 1000);
  }

  function tickClocks() {
    const now = new Date();
    for (const z of ZONES) {
      if (!z.fmt) continue;
      const parts = Object.fromEntries(z.fmt.formatToParts(now).map((x) => [x.type, x.value]));
      z.time.textContent = `${parts.hour}:${parts.minute}:${parts.second}`;
      if (z.open) {
        const mins = Number(parts.hour) * 60 + Number(parts.minute);
        const open = !["Sat", "Sun"].includes(parts.weekday) && mins >= z.open[0] && mins < z.open[1];
        z.dot.classList.toggle("open", open);
      }
    }
  }

  let jobsRunning = 0;
  let statusTimer = 0;
  let alertsSeen = null;  // the counts at the last status poll

  /** The top bar's alert count: notice and high alerts, what ALRT lists (info ones, headlines, only in
   * the tooltip: ALRT ALL lists them). A change in either refreshes the panes showing ALRT or WL. */
  function showAlerts(al) {
    const el = $("#st-alerts");
    const a = al && typeof al === "object" ? al : {};
    if (!isNum(a.unacked)) {
      el.textContent = "ALRT ?";
      el.className = "st st-btn bad";
      el.title = "The alert state could not be read: run ALRT to see why";
      return;
    }
    const info = isNum(a.info) ? a.info : 0;
    el.textContent = `ALRT ${a.unacked}`;
    el.className = "st st-btn" + (a.high > 0 ? " bad" : a.unacked > 0 ? " warn" : "");
    el.title = `${a.unacked} unacknowledged alert${a.unacked === 1 ? "" : "s"}${a.high ? `, ${a.high} high` : ""}: click to run ALRT`
      + (info ? ` (and ${info} info: ALRT ALL)` : "");
    const seen = `${a.unacked}/${info}`;
    if (alertsSeen !== null && seen !== alertsSeen) {
      for (const p of panes) if (p.res && p.res.ok && (p.res.fn === "ALRT" || p.res.fn === "WL")) p.reload(true);
    }
    alertsSeen = seen;
  }

  /** Poll status every STATUS_EVERY ms, or every JOBS_POLL ms while jobs are active. */
  function scheduleStatus(ms) {
    clearTimeout(statusTimer);
    statusTimer = setTimeout(async () => {
      await refreshStatus();
      scheduleStatus(jobsRunning > 0 ? JOBS_POLL : STATUS_EVERY);
    }, ms);
  }

  /** This page just started jobs (an INGEST result). A job can finish within a second, between two polls,
   * so count it now: the next poll that sees fewer running then refreshes JOBS and errored panes. */
  function jobsStarted(res) {
    const rows = (res && res.data && Array.isArray(res.data.rows)) ? res.data.rows : [];
    const active = rows.filter((r) => r.status === "running" || r.status === "queued").length;
    jobsRunning = Math.max(jobsRunning, active || 1);
    scheduleStatus(JOBS_POLL);
  }

  async function refreshStatus() {
    const s = await api("/api/status");
    const conn = $("#st-conn");
    if (!s || !s.ok) {
      conn.textContent = s && s.expired ? "SESSION EXPIRED: RELOAD" : "SERVER OFFLINE";
      conn.title = (s && s.error) || "";
      conn.hidden = false;
      return;
    }
    conn.hidden = true;
    const bank = $("#st-bank");
    bank.replaceChildren(document.createTextNode("BANK "), h("b", "", s.entities == null ? "–" : String(s.entities)),
      document.createTextNode(" · UNIV "), h("b", "", s.universe == null ? "–" : String(s.universe)));
    const running = Number(s.jobs_running) || 0;
    const jobs = $("#st-jobs");
    jobs.textContent = running ? `JOBS ${running} RUNNING` : "JOBS 0";
    jobs.className = "st st-btn" + (running ? " busy" : "");
    const an = s.analyst || {};
    const ae = $("#st-analyst");
    // the analyst is an open-weight model on a local server: when it is off, say what to start or set
    // available with a reason: the server is busy answering a question (brain.llm.BUSY), so it was not probed
    const busy = !!an.available && !!an.reason;
    ae.textContent = an.available ? (busy ? "ANALYST BUSY" : "ANALYST ON") : "ANALYST OFF · OLLAMA / BRAIN_LLM_URL";
    ae.className = "st st-btn " + (an.available ? (busy ? "warn" : "ok") : "bad");
    ae.title = an.available ? `The analyst (ASK) is available${an.model ? `: ${an.model}` : ""}${busy ? `, ${an.reason}` : ""}. Click to open it.`
      : `The analyst is unavailable: ${an.reason || "not configured"}. Click for how to start it.`;
    showAlerts(s.alerts);

    if (running < jobsRunning) {
      // a job finished: new facts may answer panes that were missing data (an INGEST pane reruns as JOBS)
      for (const p of panes) {
        if (!p.res || !p.res.ok || p.res.fn === "JOBS" || p.res.fn === "INGEST") p.reload(true);
      }
      refreshTape();
    } else if (running > 0) {
      for (const p of panes) if (p.res && (p.res.fn === "JOBS" || p.res.fn === "INGEST")) p.reload(true);
    }
    jobsRunning = running;
  }

  // ---------- ticker tape ----------
  let tapeKey = "";

  async function refreshTape() {
    const res = await runCmd("WEI");
    const track = $("#tape-track");
    if (!res || !res.ok || !res.data || !Array.isArray(res.data.rows) || !res.data.rows.length) {
      tapeKey = "";
      track.className = "tape-track";
      track.replaceChildren();
      const msg = h("span", "tape-msg");
      msg.append(h("span", "amber", "WEI "), document.createTextNode((res && res.error) || "no world-market rows"));
      if (res && res.hint && looksLikeCommand(res.hint)) msg.append(cmdLink(res.hint, null));
      track.append(msg);
      return;
    }
    const rows = res.data.rows;
    const key = JSON.stringify(rows.map((r) => [r.symbol, r.last, r.chg_1d]));
    if (key === tapeKey) return;  // unchanged: keep scrolling without a jump
    tapeKey = key;
    const seq = h("span", "tape-seq");
    for (const r of rows) {
      const sym = tickerOk(r._ticker) ? r._ticker : r.symbol;
      const it = h("span", "tape-item");
      it.dataset.sym = String(sym);
      it.title = `${r.name || sym}: ${sym} DES (Shift-click: next pane)`;
      const chg = fmtPct(r.chg_1d);
      const arrow = signOf(chg) === "up" ? "▲" : signOf(chg) === "down" ? "▼" : "";
      it.append(h("span", "tape-sym", r.symbol), h("span", "tape-name", r.name || ""),
        h("span", "tape-last", fmtNum(r.last, r._digits != null ? r._digits : priceDigits(r.last)) || "—"),
        h("span", signOf(chg), chg ? `${arrow}${chg}` : "—"));
      seq.append(it);
    }
    const copy = seq.cloneNode(true);
    copy.setAttribute("aria-hidden", "true");
    track.replaceChildren(seq, copy);
    track.className = "tape-track";
    requestAnimationFrame(() => {
      const w = seq.getBoundingClientRect().width;
      track.style.animationDuration = Math.max(30, Math.round(w / 40)) + "s";
      track.className = "tape-track run";
    });
  }

  function bindTape() {
    $("#tape-track").addEventListener("click", (e) => {
      const it = e.target.closest && e.target.closest(".tape-item");
      if (!it || !tickerOk(it.dataset.sym)) return;
      const p = focusedPane();
      (e.shiftKey ? nextPane(p) : p).run(`${it.dataset.sym} DES`);
    });
  }

  // ---------- boot ----------
  function watchPixelRatio() {
    let mq = null;
    const arm = () => {
      if (mq) mq.removeEventListener("change", onChange);
      mq = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
      mq.addEventListener("change", onChange);
    };
    const onChange = () => {
      for (const r of redrawers) r();
      arm();
    };
    arm();
  }

  function boot() {
    const grid = $("#grid");
    for (let i = 0; i < N_PANES; i += 1) {
      const p = new Pane(i);
      panes.push(p);
      grid.append(p.el);
    }
    const { lines, focus } = loadLayout();
    lines.forEach((line, i) => panes[i].run(line));
    booted = true;
    focusPane(focus);
    bindCommandLine();
    bindKeys();
    bindPopover();
    bindTape();
    buildClocks();
    watchPixelRatio();
    $("#st-jobs").addEventListener("click", () => focusedPane().run("JOBS"));
    $("#st-alerts").addEventListener("click", () => focusedPane().run("ALRT"));
    $("#st-analyst").addEventListener("click", () => focusedPane().run("ASK"));
    refreshStatus();
    refreshTape();
    scheduleStatus(STATUS_EVERY);
    setInterval(refreshTape, TAPE_EVERY);
    cmdInput().focus();
  }

  boot();
})();
