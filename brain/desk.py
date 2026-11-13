"""The desk: a case file for every check, and the reasoned note written from it (docs/DESK.md).

A check between scans (``brain/scanwatch.py``) says which stocks have left their ranges and prints a headline beside
each. That is a list, not an explanation. The desk takes the next two steps.

1. **The case file** (``build``). Everything that bears on the day's moves, gathered by rule, each item with an id:
   the market and the drivers (oil, the 10-year yield, the dollar, gold, bitcoin) against their usual day; each
   mover's move split into the market's part, its drivers' part and what is left; its partners (the stocks it has
   moved with for a year, whatever their industry) and how they moved today; the stories in its news feed with
   their summaries, and which other movers each story names; the opening text of its filings since the close; its
   size, cash and losses from its own accounts; its last sessions; the checks that already flagged it; every checked stock's move (`q:TICKER`); clusters of
   movers tied by a story or by a year of moving together; and the last note's themes and open questions.
2. **The note** (``record_note``). A reasoner reads the case file, and what it points to, and writes down what
   happened and why: the backdrop, the themes that tie movers together (the cause, the mechanism, what lies beneath,
   who else is exposed), each lone mover's cause, and what stays unexplained. The engine holds the note to rules: every cause cites
   items of the case file or pages the reasoner read; a cause is marked reported (a source states it) or inferred
   (the reasoner's own); every stock outside its 9-in-10 range is explained or listed as unexplained; and every
   forward-looking statement is a call, recorded with its chance before its outcome (``brain/predict.py``, in
   ``research/desk_calls.jsonl``) and scored after. A call starts from the stock's price at the check, not from the
   last close, so none of its outcome is known when it is made.

The engine does not reason, and the reasoner does not get to assert. The reasoner is the agent working in this repo
(CLAUDE.md), or a model given the same case file. A note is an explanation. It is not evidence, and its calls carry
weight only through their scored record.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from brain import config, connect, predict, scan, scanwatch, why
from brain.bank import Bank, canon
from brain.timeutil import days_between, now_iso

MOVERS = 30             # movers a case file covers, the furthest outside their ranges first
STORIES = 4             # stories kept for each mover (those that name it first)
WRAPS = 8               # market-wide stories kept for the backdrop
TEXTS = 12              # movers whose filings since the close are opened
FILINGS_EACH = 2        # filings opened for one mover
OPENING = 1800          # characters of a filing's text kept
NUMBERS = 12            # movers whose accounts are read
STALE_DAYS = 200        # a balance-sheet figure older than this is not shown
LAST = 5                # sessions of each mover's own history shown
EXPOSED = 6             # stocks shown for a driver that moved
DRIVER_BIG = 1.5        # a driver's move this many times its usual day is shown with the stocks it reaches
CLUSTER_CORR = 0.40     # two movers that moved the same way today are tied when their year's correlation is this
KINDS = ("reported", "inferred")
CALLS = ("up", "down", "move", "shock")
_ITEM = re.compile(r"Item\s+\d\.\d\d", re.IGNORECASE)


def cases_dir() -> Path:
    return config.home() / "desk"


def notes_dir() -> Path:
    return Path(os.environ.get("BRAIN_DESK") or config.ROOT / "research" / "desk")


def calls_path() -> Path:
    return Path(os.environ.get("BRAIN_DESK_CALLS") or config.ROOT / "research" / "desk_calls.jsonl")


def _pct(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:+.{d}f}%"


def _money(x: float | None) -> str:
    if x is None:
        return "n/a"
    for size, unit in ((1e9, "bn"), (1e6, "m")):
        if abs(x) >= size:
            return f"${x / size:,.1f} {unit}"
    return f"${x:,.0f}"


# ---------- what the case file reads ----------
def latest_watch() -> tuple[dict, dict | None, str]:
    """The newest recorded check, every checked stock's move from it when the local home still has them, and the
    check's stamp."""
    files = sorted(scan.scans_dir().glob("updates/????-??-??/*.json"), key=lambda p: (p.parent.name, p.name))
    if not files:
        raise SystemExit("no recorded check yet: python -m brain scan watch --record")
    p = files[-1]
    rec = json.loads(p.read_text())
    mp = scanwatch.moves_path(rec["asof"], p.stem)
    return rec, (json.loads(mp.read_text()) if mp.exists() else None), p.stem


def seen_before(asof: str, stamp: str, days: int = 2) -> dict[str, list[str]]:
    """The earlier checks (this scan's and the scan before's) that found each stock outside its 2-in-3 range."""
    out: dict[str, list[str]] = {}
    dirs = sorted(d for d in (scan.scans_dir() / "updates").glob("????-??-??") if d.name <= asof)[-days:]
    for d in dirs:
        for p in sorted(d.glob("*.json")):
            if d.name == asof and p.stem >= stamp:
                continue
            try:
                flags = json.loads(p.read_text()).get("flags") or []
            except ValueError:
                continue
            for f in flags:
                if f.get("outside68"):
                    out.setdefault(f["t"], []).append(f"{p.stem[5:10]} {p.stem[11:13]}:{p.stem[13:15]}Z {_pct(f['move'])}")
    return out


def filings_of(t: str, asof: str) -> list:
    """A stock's 8-Ks and share-sale filings first public after the close of ``asof``, the oldest first."""
    from brain.sources import sec
    try:
        facts, _ = sec.filings_for(t, all_pages=False, ttl=600)
    except Exception:  # noqa: BLE001 - one company's outage must not stop the case file
        return []
    new = [f for f in facts if f.attr == "filing" and str(f.value).upper().startswith(scanwatch.FILING_FORMS)]
    seen, out = set(), []
    for f in sorted(scanwatch.since_close(new, asof), key=lambda f: f.known_at):
        key = (f.context or {}).get("accn") or f.known_at
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def filing_text(f) -> dict:
    """The opening of what a filing says: an 8-K's press release (exhibit 99.1), else its own text from its first
    item; a share-sale filing's first page."""
    from brain.sources import releases
    url = (f.source or {}).get("url", "")
    try:
        if str(f.value).upper().startswith("8-K"):
            r = releases.read_release(f)
            if (r.context or {}).get("opening"):
                return {"title": str(r.value), "text": r.context["opening"][:OPENING], "url": r.source["url"]}
        raw = releases._get(url, releases.HEAD_BYTES).decode("utf-8", "replace")
        title, text = releases.opening(raw)
        m = _ITEM.search(text)
        return {"title": title, "text": text[m.start() if m else 0:][:OPENING], "url": url}
    except Exception as e:  # noqa: BLE001 - a filing that cannot be read is listed without its text
        return {"title": "", "text": "", "url": url, "error": f"{type(e).__name__}: {e}"[:200]}


def numbers(bank: Bank, t: str) -> dict | None:
    """A company's size and cash from its own accounts (SEC XBRL): market value, cash, long-term debt, the last
    four quarters' revenue and net income, and for a company losing money how many quarters its cash covers."""
    from brain import metrics
    from brain.sources import sec
    try:
        bank.add_many(sec.facts_for(t))
    except Exception:  # noqa: BLE001, S110 - no accounts at the SEC (a fund, a foreign filer): what the bank has
        pass
    view = bank.view()
    out: dict = {}
    try:
        mc = metrics.market_cap_any(view, t, metrics.PriceBook(bank, [t]))
    except Exception:  # noqa: BLE001
        mc = None
    if mc is not None:
        out["market_value"] = float(mc.value)
    def fresh(f):
        return f is not None and days_between(f.as_of, view.t) <= STALE_DAYS

    cash = view.latest(t, "us-gaap:CashAndCashEquivalentsAtCarryingValue")
    if fresh(cash):
        out["cash"], out["cash_at"] = float(cash.value), cash.as_of
    debt = max((f for f in (view.latest(t, "us-gaap:LongTermDebt"), view.latest(t, "us-gaap:LongTermDebtNoncurrent"))
                if fresh(f)), key=lambda f: f.as_of, default=None)
    if debt is not None:
        out["debt"], out["debt_at"] = float(debt.value), debt.as_of
    for key, attrs in (("revenue_4q", metrics.REVENUE), ("net_income_4q", metrics.NET_INCOME)):
        try:
            m = metrics.ttm(view, t, attrs)
        except Exception:  # noqa: BLE001
            m = None
        if m is not None:
            out[key], out[key + "_to"] = float(m.value), m.as_of
    if out.get("net_income_4q", 0) < 0 and out.get("cash"):
        out["cash_covers_quarters"] = round(out["cash"] / (-out["net_income_4q"] / 4), 1)
    return out or None


def memory(before: tuple[str, str] | None = None) -> dict | None:
    """The last recorded note (before the check ``before``: its scan date and stamp): its themes, threads and when."""
    files = sorted(notes_dir().glob("????-??-??/*.json"), key=lambda p: (p.parent.name, p.name))
    if before:
        files = [p for p in files if (p.parent.name, p.stem) < before]
    if not files:
        return None
    rec = json.loads(files[-1].read_text())
    n = rec["note"]
    return {"check": rec["check"], "asof": rec["asof"], "backdrop": n.get("backdrop"),
            "themes": [{k: th.get(k) for k in ("name", "cause", "kind", "members", "watch")} for th in n.get("themes", [])],
            "threads": n.get("threads") or []}


# ---------- the case file ----------
def build(bank: Bank, rec: dict | None = None, moves: dict | None = None, stamp: str | None = None, *,
          texts: bool = True, accounts: bool = True, log=None, feed=None, quote=None) -> dict:
    """The case file for one check (the latest recorded one when none is given)."""
    if rec is None:
        rec, moves, stamp = latest_watch()
    stamp = stamp or scanwatch.stamp(rec)
    asof = rec["asof"]
    say = log or (lambda m: None)
    rows = json.loads((scan.scans_dir() / rec["scan"]).read_text())["rows"]
    tickers = list(rows)
    ix = {t: j for j, t in enumerate(tickers)}
    view = bank.view()
    ind = why.industry_map(view, tickers)
    names = {t: (ind.get(t) or {}).get("company") for t in tickers}

    say("  a year of daily returns")
    b = connect.cached_book(bank, tickers, asof)
    E, _ = connect.residuals(b["R"], b["m"])
    C = connect.correlations(E)
    expo = connect.exposures(b["R"], b["m"], connect.driver_changes(b["cal"]))

    moves = dict(moves or {})
    for f in rec["flags"]:
        moves.setdefault(f["t"], f["move"])
    quote = quote or scanwatch.quote

    def move_of(t: str) -> float | None:
        if t not in moves:              # an older check kept only its flags: read the stock now
            try:
                q, c = quote(t), scanwatch.close_on(view, t, asof)
                moves[t] = q["price"] / c - 1 if q and c else None
            except Exception:  # noqa: BLE001
                moves[t] = None
        return moves[t]

    spy = next((m["move"] for m in rec["market"] if m["symbol"] == "SPY"), 0.0)
    drivers = []
    for d in rec.get("drivers") or []:
        usual = expo["usual"].get(d["symbol"])
        drivers.append({**d, "usual": usual, "times": abs(d["move"]) / usual if usual else None})
    dmoves = {d["symbol"]: d["move"] for d in drivers}
    dname = {d["symbol"]: d["name"] for d in drivers}

    flags = [f for f in rec["flags"] if f["outside68"] and f["t"] in ix]
    seen = seen_before(asof, stamp)
    movers = []
    for f in flags[:MOVERS]:
        t, j = f["t"], ix[f["t"]]
        mb = expo["market"][j]
        carried = connect.carried(expo, j, dmoves)
        parts = (mb * spy if np.isfinite(mb) else 0.0) + sum(c["implied"] for c in carried)
        c = f.get("industry")
        movers.append({
            "t": t, "company": names.get(t), "industry": (ind.get(t) or {}).get("name"), "move": f["move"],
            "session": f["session"], "outside90": bool(f["outside90"]), "range68": f["range68"], "range90": f["range90"],
            "why": f.get("why"), "industry_move": c["median"] if c else None, "industry_name": c["name"] if c else None,
            "industry_n": c["n"] if c else None, "own": f.get("own"),
            "market_beta": float(mb) if np.isfinite(mb) else None,
            "market_part": float(mb * spy) if np.isfinite(mb) else None,
            "drivers": [{**x, "name": dname.get(x["driver"], x["driver"])} for x in carried],
            "left": float(f["move"] - parts),
            "partners": [{"t": p, "corr": round(r, 2), "move": move_of(p)} for p, r in connect.partners(C, tickers, t)],
            "last": [[str(d), round(float(e), 4)] for d, e in zip(b["cal"][1:][-LAST:], E[-LAST:, j]) if np.isfinite(e)],
            "seen": seen.get(t, [])[-6:], "filed": f.get("filings") or [], "filings": [], "stories": [], "numbers": None})
    mset = {m["t"] for m in movers}
    for m in movers:
        for p in m["partners"]:
            p["mover"] = p["t"] in mset

    say(f"  the news feeds of {len(movers)} movers")
    feed = feed or connect.feed_items
    feeds = {}
    for m in movers:
        try:
            feeds[m["t"]] = feed(m["t"])
        except Exception:  # noqa: BLE001 - one feed's outage must not stop the case file
            feeds[m["t"]] = []
    cand = {t: names.get(t) for t in mset | {p["t"] for m in movers for p in m["partners"]}}
    found = connect.stories(feeds, why.close_utc(asof), cand)
    stories, sid = [], {}

    def keep(s: dict) -> str:
        key = connect.story_key(s["url"])
        if key not in sid:
            sid[key] = f"s{len(stories) + 1}"
            stories.append({"id": sid[key], **s})
        return sid[key]

    for m in movers:
        mine = [s for s in found if m["t"] in s["names"] and not s["wrap"]]
        rest = [s for s in found if m["t"] in s["feeds"] and m["t"] not in s["names"] and not s["wrap"]]
        m["stories"] = [keep(s) for s in (mine + ([] if mine else rest))[:STORIES]]
    wraps = [keep(s) for s in [s for s in found if s["wrap"]][:WRAPS]]

    # movers tied by a story that names one of them, or by a year of moving together and the same way today
    ties = []
    for s in stories:
        if not s["wrap"] and set(s["names"]) & mset:
            linked = connect.story_links(s, mset)
            ties += [{"a": a, "b": c, "by": s["id"]} for i, a in enumerate(linked) for c in linked[i + 1:]]
    for m in movers:
        for p in m["partners"]:
            if p["mover"] and m["t"] < p["t"] and p["corr"] >= CLUSTER_CORR and (m["move"] > 0) == ((p["move"] or 0) > 0):
                ties.append({"a": m["t"], "b": p["t"], "by": f"p:{m['t']}~{p['t']}"})
    clusters = []
    for k, group in enumerate(connect.clusters([m["t"] for m in movers], [(x["a"], x["b"]) for x in ties])):
        inside = set(group)
        near: dict[str, list[str]] = {}
        for m in movers:
            if m["t"] in inside:
                for p in m["partners"]:
                    if not p["mover"]:
                        near.setdefault(p["t"], []).append(m["t"])
        exposed = sorted(near, key=lambda t: (-len(near[t]), -abs(move_of(t) or 0)))[:EXPOSED]
        clusters.append({"id": f"c{k + 1}", "members": group,
                         "ties": [x for x in ties if x["a"] in inside and x["b"] in inside],
                         "exposed": [{"t": t, "move": move_of(t), "partner_of": near[t]} for t in exposed]})

    reach = []
    for d in drivers:
        if d["times"] and d["times"] >= DRIVER_BIG:
            top = connect.most_exposed(expo, tickers, d["symbol"], EXPOSED)
            reach.append({"driver": d["name"], "stocks": [{"t": t, "beta": round(bb, 3), "t_stat": round(tt, 1),
                                                           "implied": bb * d["move"], "move": move_of(t)}
                                                          for t, bb, tt in top]})

    if texts:
        say("  the filings since the close")
        for m in [m for m in movers if m["filed"]][:TEXTS]:
            for k, f in enumerate(filings_of(m["t"], asof)[-FILINGS_EACH:]):
                m["filings"].append({"id": f"f:{m['t']}:{k + 1}", "what": scanwatch.describe(f), "at": f.known_at,
                                     **filing_text(f)})
    if accounts:
        say("  the accounts of the largest movers")
        for m in movers[:NUMBERS]:
            try:
                m["numbers"] = numbers(bank, m["t"])
            except Exception:  # noqa: BLE001 - one company's accounts must not stop the case file
                m["numbers"] = None

    case = {"check": stamp, "asof": asof, "at": rec["at"], "built_at": now_iso(), "scan": rec["scan"],
            "market": rec["market"], "market_wide": rec["market_wide"], "fear_up": rec["fear_up"], "drivers": drivers,
            "macro_today": rec.get("macro_today") or [], "sectors": rec.get("sectors") or [],
            "counts": {"checked": rec["checked"], "outside68": rec["outside68"], "outside90": rec["outside90"],
                       "why": rec.get("why_counts") or {}, "beyond": max(0, len(flags) - len(movers))},
            "movers": movers, "stories": stories, "wraps": wraps, "clusters": clusters, "reach": reach,
            "memory": memory((asof, stamp)), "moves": {t: v for t, v in moves.items() if v is not None}}
    return json.loads(canon(case))


def save_case(case: dict) -> Path:
    """The case file in the local home: a working file (it holds Yahoo's prices), rebuilt at every check."""
    p = cases_dir() / case["asof"] / f"{case['check']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(case))
    return p


def load_case(stamp: str | None = None) -> dict:
    files = sorted(cases_dir().glob("????-??-??/*.json"), key=lambda p: (p.parent.name, p.name))
    if stamp:
        files = [p for p in files if p.stem == stamp]
    if not files:
        raise SystemExit("no case file: python -m brain desk build")
    return json.loads(files[-1].read_text())


# ---------- evidence: every item of a case file a note can cite ----------
def case_moves(case: dict) -> dict[str, float]:
    """Every stock's move since the close that a case file holds: all the check read, when the check kept them."""
    return case.get("moves") or {}


def _mover_line(m: dict) -> str:
    tag = "OUTSIDE its 9-in-10 range" if m["outside90"] else "outside its 2-in-3 range"
    return (f"{m['t']} ({m['company'] or 'name unknown'}; {m['industry'] or 'industry unknown'}) {_pct(m['move'])} since "
            f"the close ({m['session']}), {tag} ({_pct(m['range68'][0], 0)}..{_pct(m['range68'][1], 0)})")


def _parts_line(m: dict) -> str:
    out = []
    if m["market_part"] is not None:
        out.append(f"the market carried {_pct(m['market_part'])} (beta {m['market_beta']:.2f})")
    for x in m["drivers"]:
        unit = "per point" if x["driver"] == "^TNX" else "per 1%"
        size = x["beta"] if x["driver"] == "^TNX" else x["beta"] / 100
        out.append(f"{x['name']} carried {_pct(x['implied'])} ({_pct(size, 2)} {unit}, t {x['t']:.1f})")
    if m["industry_move"] is not None:
        out.append(f"its industry ({m['industry_name']}, {m['industry_n']} stocks) moved {_pct(m['industry_move'])}")
    out.append(f"left over after the market and its drivers {_pct(m['left'])}")
    return "; ".join(out)


def _numbers_line(m: dict) -> str | None:
    n = m.get("numbers")
    if not n:
        return None
    out = []
    if "market_value" in n:
        out.append(f"market value {_money(n['market_value'])}")
    if "cash" in n:
        out.append(f"cash and equivalents {_money(n['cash'])} ({n['cash_at']}; investments not counted)")
    if "debt" in n:
        out.append(f"long-term debt {_money(n['debt'])} ({n['debt_at']})")
    if "revenue_4q" in n:
        out.append(f"revenue {_money(n['revenue_4q'])} over the four quarters to {n['revenue_4q_to']}")
    if "net_income_4q" in n:
        out.append(f"net income {_money(n['net_income_4q'])} over the four quarters to {n['net_income_4q_to']}")
    if "cash_covers_quarters" in n:
        out.append(f"that cash alone covers about {n['cash_covers_quarters']} quarters of such losses")
    return "; ".join(out)


def evidence(case: dict) -> dict[str, dict]:
    """Every citable item of a case file by its id: its one-line text, and its page when it has one."""
    ev: dict[str, dict] = {}
    if case["market"]:
        ev["k"] = {"text": "Market: " + ", ".join(f"{m['name']} {_pct(m['move'])}" for m in case["market"])
                   + f" ({case['market'][0]['session']})"}
    for d in case["drivers"]:
        pts = d["unit"] == "points"
        move = f"{d['move']:+.2f} pt" if pts else _pct(d["move"])
        usual = "" if not d.get("usual") else (f" (its usual day: {d['usual']:.2f} pt)" if pts else
                                               f" (its usual day: {d['usual'] * 100:.1f}%)")
        ev[f"d:{d['name']}"] = {"text": f"{d['name']} {move} since the close{usual}"}
    for s in case["sectors"]:
        ev[f"g:{s['code']}"] = {"text": f"{s['name']}: the median of its {s['n']} checked stocks moved {_pct(s['median'])}"}
    for r in case["reach"]:
        ev[f"r:{r['driver']}"] = {"text": f"the stocks that moved most with {r['driver']} over the year, today: " + ", ".join(
            f"{x['t']} {_pct(x['move'])} (its rate implies {_pct(x['implied'])})" for x in r["stocks"])}
    for s in case["stories"]:
        ev[s["id"]] = {"text": f"{s['at'][:16]}Z {s['source']}: {s['title']}" + (f" -- {s['summary']}" if s["summary"] else ""),
                       "url": s["url"]}
    for m in case["movers"]:
        t = m["t"]
        ev[f"m:{t}"] = {"text": _mover_line(m)}
        ev[f"x:{t}"] = {"text": f"{t}: {_parts_line(m)}"}
        if m["last"] or m["seen"]:
            ev[f"h:{t}"] = {"text": f"{t} against the market, last sessions: "
                            + ", ".join(f"{d[5:]} {_pct(e)}" for d, e in m["last"])
                            + (f"; flagged before: {'; '.join(m['seen'])}" if m["seen"] else "")}
        for p in m["partners"]:
            ev[f"p:{t}~{p['t']}"] = {"text": f"{t} and {p['t']}: market-adjusted daily returns correlated {p['corr']:.2f} "
                                     f"over the last {connect.LOOKBACK} sessions; {p['t']} {_pct(p['move'])} since the close"}
        if (line := _numbers_line(m)):
            ev[f"n:{t}"] = {"text": f"{t} accounts: {line}"}
        for f in m["filings"]:
            ev[f["id"]] = {"text": f"{t} {f['what']} at {f['at'][:16]}Z: {f['title']}"
                           + (f" -- {f['text']}" if f["text"] else ""), "url": f["url"]}
    for c in case["clusters"]:
        ev[c["id"]] = {"text": "movers tied together: " + ", ".join(c["members"]) + " ("
                       + "; ".join(f"{x['a']}~{x['b']} by {x['by']}" for x in c["ties"]) + ")"}
    for k, th in enumerate((case.get("memory") or {}).get("themes") or []):
        ev[f"t{k + 1}"] = {"text": f"the last note ({case['memory']['check']}), theme {th['name']!r}: {th['cause']}"}
    for t, v in case_moves(case).items():      # any checked stock's move can be cited, mover or not
        ev[f"q:{t}"] = {"text": f"{t} {_pct(v)} since the close of {case['asof']}, as the check read it"}
    return ev


# ---------- reading a case file ----------
def render(case: dict, detail: list[str] | None = None) -> str:
    """The case file as text: the overview, or with ``detail`` the full items of those movers."""
    ev = evidence(case)
    L = [f"CASE FILE for the check {case['check']} (against the close of {case['asof']}; built {case['built_at'][:16]}Z)",
         "Every [id] can be cited in a note. The engine gathered this by rule; nothing here is a conclusion.", "",
         "BACKDROP", f"  [k] {ev['k']['text']}" if "k" in ev else "  no market reading"]
    if case["market_wide"]:
        L.append("  A market-wide move was flagged.")
    if case["fear_up"]:
        L.append("  Rising fear was flagged (the VIX up 15% or more).")
    L += [f"  [d:{d['name']}] {ev['d:' + d['name']]['text']}" for d in case["drivers"]]
    if case["macro_today"]:
        L.append("  Scheduled today: " + ", ".join(case["macro_today"]))
    L += [f"  [g:{s['code']}] {ev['g:' + s['code']]['text']}" for s in case["sectors"]]
    c = case["counts"]
    L.append(f"  {c['checked']} stocks checked: {c['outside90']} outside their 9-in-10 range, {c['outside68']} outside "
             f"their 2-in-3 range (over five full sessions about 10% and 32% end outside). "
             + (f"{c['beyond']} of them are beyond this case file's {MOVERS} movers." if c["beyond"] else ""))
    if case["wraps"]:
        L.append("  Market-wide stories:")
        L += [f"    [{i}] {ev[i]['text'][:330]}" for i in case["wraps"]]
    if case["reach"]:
        L += ["", (f"DRIVER REACH (a driver that moved {DRIVER_BIG} times its usual day, and the stocks that moved most "
                   "with it over the year)")]
        L += [f"  [r:{r['driver']}] {ev['r:' + r['driver']]['text']}" for r in case["reach"]]
    if case["clusters"]:
        L += ["", ("CLUSTERS (movers tied by a story or by a year of moving together; a tie is a lead, the cause is "
                   "for the note)")]
        mv = {m["t"]: m["move"] for m in case["movers"]}
        for cl in case["clusters"]:
            L.append(f"  [{cl['id']}] " + ", ".join(f"{t} {_pct(mv[t])}" for t in cl["members"]))
            L.append("       ties: " + "; ".join(f"{x['a']}~{x['b']} [{x['by']}]" for x in cl["ties"][:14]))
            if cl["exposed"]:
                L.append("       partners of its members, not outside their own ranges: " + ", ".join(
                    f"{x['t']} {_pct(x['move'])} (of {'/'.join(x['partner_of'])})" for x in cl["exposed"]))
    L += ["", "MOVERS"]
    for m in case["movers"]:
        t = m["t"]
        L.append(f"  [m:{t}] {ev['m:' + t]['text']}")
        L.append(f"     [x:{t}] {_parts_line(m)}")
        if f"h:{t}" in ev:
            L.append(f"     [h:{t}] " + ev[f"h:{t}"]["text"].split(": ", 1)[1])
        if m["partners"]:
            L.append("     partners [p:" + t + "~X]: " + ", ".join(
                f"{p['t']} {p['corr']:.2f} {_pct(p['move'])}" + ("*" if p["mover"] else "") for p in m["partners"]))
        if f"n:{t}" in ev:
            L.append(f"     [n:{t}] " + ev[f"n:{t}"]["text"].split(": ", 1)[1])
        full = bool(detail) and t in detail
        for f in m["filings"]:
            body = f["text"] if full else f["text"][:260]
            L.append(f"     [{f['id']}] {f['what']} at {f['at'][:16]}Z: {f['title'][:160]}" + (f" -- {body}" if body else "")
                     + (f"  <{f['url']}>" if full else ""))
        if m["filed"] and not m["filings"]:
            L.append("     filed since the close (not opened): " + "; ".join(m["filed"]))
        for i in m["stories"]:
            s = next(s for s in case["stories"] if s["id"] == i)
            mark = "" if t in s["names"] else " (in its feed; does not name it)"
            L.append(f"     [{i}] {ev[i]['text'] if full else ev[i]['text'][:230]}{mark}" + (f"  <{s['url']}>" if full else ""))
        if not m["stories"] and not m["filed"]:
            L.append("     no filing since the close and no story in its feed")
    mem = case.get("memory")
    if mem:
        L += ["", f"MEMORY (the last note, on the check {mem['check']})"]
        L += [f"  [t{k + 1}] {th['name']} ({th['kind']}): {th['cause']} Members: {', '.join(th['members'] or [])}."
              + (f" Watch: {th['watch']}" if th.get("watch") else "") for k, th in enumerate(mem["themes"])]
        L += [f"  open: {x}" for x in mem["threads"]]
    L.append("")
    L.append("* a partner that is itself a mover. A partner link says two stocks moved together for a year, not why.")
    return "\n".join(L)


# ---------- the note ----------
def _cited(ids, ev: dict, pages: set[str], where: str, need_source: bool, problems: list[str]) -> None:
    ids = list(ids or [])
    if not ids:
        problems.append(f"{where}: cites nothing")
    for i in ids:
        if i not in ev and i not in pages:
            problems.append(f"{where}: cites {i!r}, which is neither in the case file nor a page listed under 'read'")
    if need_source and not any(i in pages or re.match(r"(s\d+|f:)", i) for i in ids if i in ev or i in pages):
        problems.append(f"{where}: marked reported, but cites no story, filing or page that reports it")


def check(note: dict, case: dict) -> list[str]:
    """What is wrong with a note against its case file; an empty list means it can be recorded."""
    ev = evidence(case)
    problems: list[str] = []
    known = set(json.loads((scan.scans_dir() / case["scan"]).read_text())["rows"])
    if note.get("check") != case["check"]:
        problems.append(f"the note is for the check {note.get('check')!r}; the case file is {case['check']!r}")
    if not str(note.get("by") or "").strip():
        problems.append("'by' is empty: who reasoned (the agent, or a model's name)")
    pages = set()
    for k, r in enumerate(note.get("read") or []):
        url = str(r.get("url") or "")
        if not re.match(r"https?://[^/\s]+", url):
            problems.append(f"read[{k}]: its url is not an http(s) page")
        if not str(r.get("says") or "").strip():
            problems.append(f"read[{k}]: 'says' is empty (what the page states, in one line)")
        pages.add(url)
    if not str(note.get("backdrop") or "").strip():
        problems.append("'backdrop' is empty: what the market as a whole is doing, and why")
    _cited(note.get("backdrop_evidence"), ev, pages, "backdrop", False, problems)

    covered: set[str] = set()
    names = set()
    for k, th in enumerate(note.get("themes") or []):
        where = f"theme {th.get('name') or k!r}"
        names.add(th.get("name"))
        for key in ("name", "cause", "mechanism", "beneath"):
            if not str(th.get(key) or "").strip():
                problems.append(f"{where}: '{key}' is empty")
        if th.get("kind") not in KINDS:
            problems.append(f"{where}: 'kind' is one of {KINDS}")
        members = [str(t).upper() for t in th.get("members") or []]
        if len(members) < 2:
            problems.append(f"{where}: a theme ties at least two stocks (a lone mover goes under 'singles')")
        for t in members:
            if t not in known:
                problems.append(f"{where}: {t} is not a stock of the scan")
        covered |= set(members)
        _cited(th.get("evidence"), ev, pages, where, th.get("kind") == "reported", problems)
        for x in th.get("read_through") or []:
            t = str(x.get("t") or "").upper()
            if t not in known:
                problems.append(f"{where}: read-through {t or '?'} is not a stock of the scan")
            if not str(x.get("why") or "").strip():
                problems.append(f"{where}: read-through {t}: 'why' is empty")
            _cited(x.get("evidence"), ev, pages, f"{where}, read-through {t}", False, problems)
    for k, s in enumerate(note.get("singles") or []):
        t = str(s.get("t") or "").upper()
        where = f"single {t or k}"
        if t not in known:
            problems.append(f"{where}: not a stock of the scan")
        for key in ("cause", "beneath"):
            if not str(s.get(key) or "").strip():
                problems.append(f"{where}: '{key}' is empty")
        if s.get("kind") not in KINDS:
            problems.append(f"{where}: 'kind' is one of {KINDS}")
        covered.add(t)
        _cited(s.get("evidence"), ev, pages, where, s.get("kind") == "reported", problems)
    for k, u in enumerate(note.get("unexplained") or []):
        t = str(u.get("t") or "").upper()
        if t not in known:
            problems.append(f"unexplained {t or k}: not a stock of the scan")
        if not str(u.get("tried") or "").strip():
            problems.append(f"unexplained {t}: 'tried' is empty (what was looked at)")
        covered.add(t)
    for m in case["movers"]:
        if m["outside90"] and m["t"] not in covered:
            problems.append(f"{m['t']} is outside its 9-in-10 range and the note neither explains it nor lists it as "
                            "unexplained")
    at_check = case_moves(case)
    for k, c in enumerate(note.get("calls") or []):
        where = f"call {k + 1} ({c.get('ticker')})"
        if c.get("call") not in CALLS:
            problems.append(f"{where}: 'call' is one of {CALLS}")
        if not isinstance(c.get("prob"), (int, float)) or not 0 < c["prob"] < 1:
            problems.append(f"{where}: 'prob' is the chance it is right, strictly between 0 and 1")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(c.get("by") or "")):
            problems.append(f"{where}: 'by' is the date it is judged on, YYYY-MM-DD")
        if not str(c.get("thesis") or "").strip():
            problems.append(f"{where}: 'thesis' is empty")
        if c.get("from") not in names and str(c.get("from") or "").upper() not in covered:
            problems.append(f"{where}: 'from' names no theme or stock of this note")
        t = str(c.get("ticker") or "").upper()
        if t not in known and t != predict.BENCH:
            problems.append(f"{where}: not a stock of the scan")
        if t not in at_check and not (isinstance(c.get("price"), (int, float)) and c["price"] > 0):
            problems.append(f"{where}: the case file has no move for {t}, so give 'price': its price as the call is "
                            "made (a call starts from now, not from the last close)")
    return problems


def record_note(bank: Bank, case: dict, note: dict, *, calls_file: Path | None = None, claim=None) -> dict:
    """Record a note that passes ``check``: its calls first (each before its outcome, in the desk's own calls
    file), then the pages it read as claims in the bank (CLAUDE.md), then the note with the items it cites."""
    problems = check(note, case)
    if problems:
        raise ValueError("the note cannot be recorded:\n  - " + "\n  - ".join(problems))
    path = notes_dir() / case["asof"] / f"{case['check']}.json"
    if path.exists():
        raise ValueError(f"a note for the check {case['check']} is already recorded ({path}); a note is never edited")
    ev = evidence(case)
    cited: set[str] = set(note.get("backdrop_evidence") or [])
    for th in note.get("themes") or []:
        cited |= set(th.get("evidence") or [])
        for x in th.get("read_through") or []:
            cited |= set(x.get("evidence") or [])
    for s in note.get("singles") or []:
        cited |= set(s.get("evidence") or [])
    calls, at_check, view = [], case_moves(case), bank.view()
    for c in note.get("calls") or []:
        t = str(c["ticker"]).upper()
        price = c.get("price")
        if price is None:               # the stock's price at the check: its close, moved by what the check read
            close = scanwatch.close_on(view, t, case["asof"])
            if close is None:
                raise ValueError(f"call on {t}: the bank has no close for it on {case['asof']}; give 'price'")
            price = round(close * (1 + at_check[t]), 4)
        made = predict.add(bank, t, c["call"], float(c["prob"]), c["by"],
                           f"desk {case['check']}, from {c['from']}: {c['thesis']}",
                           threshold=float(c.get("threshold") or 0.0), price=float(price),
                           sources=[ev[i]["url"] for i in sorted(cited) if i in ev and ev[i].get("url")][:6],
                           path=calls_file or calls_path())
        calls.append(made["id"])
    if claim is None:
        from brain import claims
        claim = claims.note
    noted = 0
    for r in note.get("read") or []:
        if r.get("ticker"):
            try:
                bank.add(claim(bank, str(r["ticker"]).upper(), str(r["says"]), r["url"], tier=r.get("tier") or "reported",
                               direction=r.get("direction") or "none", known_at=r.get("published"), noted_by="desk"))
                noted += 1
            except ValueError:
                pass            # a page with a bad date is still listed in the note; it is just not kept as a claim
    rec = {"check": case["check"], "asof": case["asof"], "at": case["at"], "recorded_at": now_iso(),
           "case_sha": hashlib.sha256(canon(case).encode()).hexdigest()[:16], "note": note,
           "evidence": {i: ev[i] for i in sorted(cited) if i in ev}, "calls": calls, "claims_noted": noted}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return {**rec, "path": str(path)}


def show_note(rec: dict) -> str:
    """A recorded note as text."""
    n = rec["note"]
    L = [f"Desk note for the check {rec['check']} (against the close of {rec['asof']}), reasoned by {n['by']}.",
         "It explains; it is not evidence. Reported: a source states the cause. Inferred: the reasoner's own.", "",
         f"Backdrop: {n['backdrop']} [{', '.join(n.get('backdrop_evidence') or [])}]"]
    for th in n.get("themes") or []:
        L += ["", f"Theme: {th['name']} ({th['kind']}). Stocks: {', '.join(th['members'])}.",
              f"  Cause: {th['cause']} [{', '.join(th['evidence'])}]", f"  How it reaches them: {th['mechanism']}",
              f"  Beneath: {th['beneath']}"]
        L += [f"  May also reach {x['t']} (untested): {x['why']} [{', '.join(x.get('evidence') or [])}]"
              for x in th.get("read_through") or []]
        if th.get("watch"):
            L.append(f"  Watch: {th['watch']}")
    for s in n.get("singles") or []:
        L += ["", f"{s['t']} ({s['kind']}): {s['cause']} [{', '.join(s['evidence'])}]", f"  Beneath: {s['beneath']}"]
    if n.get("unexplained"):
        L += ["", "Unexplained:"] + [f"  {u['t']}: looked at {u['tried']}" for u in n["unexplained"]]
    if n.get("calls"):
        L += ["", f"Calls recorded before their outcome ({calls_path().name}):"]
        L += [f"  {i}  {c['ticker']} {c['call']}" + (f" {c['threshold']}%" if c.get("threshold") else "")
              + f" by {c['by']}, chance {c['prob']:.0%}: {c['thesis']}" for i, c in zip(rec["calls"], n["calls"])]
    if n.get("threads"):
        L += ["", "Open questions for the next check:"] + [f"  - {x}" for x in n["threads"]]
    return "\n".join(L)


def score(bank: Bank, path: Path | None = None) -> dict:
    """The desk's calls judged (``brain/predict.py``): under MIN_CALLS judged calls the record says nothing."""
    return predict.score(predict.state(bank, calls_path() if path is None else path))


TEMPLATE = {
    "check": "<the case file's check>", "by": "agent",
    "backdrop": "what the market as a whole is doing, and why", "backdrop_evidence": ["k"],
    "read": [{"url": "https://...", "says": "what the page states, in one line", "published": "ISO time",
              "ticker": "optional: keeps it as a claim in the bank", "tier": "reported|opinion",
              "direction": "up|down|none"}],
    "themes": [{"name": "short name", "kind": "reported|inferred", "cause": "what happened",
                "mechanism": "how it reaches each stock", "beneath": "the underlying driver, with numbers",
                "members": ["AAA", "BBB"], "evidence": ["s1", "p:AAA~BBB"],
                "read_through": [{"t": "CCC", "why": "why it may reach it", "evidence": ["p:AAA~CCC"]}],
                "watch": "what would confirm it or kill it"}],
    "singles": [{"t": "DDD", "kind": "reported|inferred", "cause": "what happened", "beneath": "what lies under it",
                 "evidence": ["f:DDD:1"]}],
    "unexplained": [{"t": "EEE", "tried": "what was looked at"}],
    "calls": [{"ticker": "CCC", "call": "move|up|down|shock", "threshold": 5.0, "prob": 0.35, "by": "YYYY-MM-DD",
               "thesis": "one line", "from": "the theme's name, or a stock of the note",
               "price": "only for a stock the case file has no move for: its price now"}],
    "threads": ["an open question for the next check"]}


# ---------- commands ----------
def _cmd(a) -> int:
    bank = Bank()
    if a.action == "build":
        case = build(bank, texts=not a.no_texts, accounts=not a.no_accounts, log=lambda m: print(m, flush=True))
        print(render(case))
        print(f"\ncase file {save_case(case)}\nA mover in full: python -m brain desk show TICKER ...   "
              "The note's layout: python -m brain desk template")
        return 0
    if a.action == "show":
        print(render(load_case(a.check), detail=[w.upper() for w in a.words] or None))
        return 0
    if a.action == "template":
        print(json.dumps(TEMPLATE, indent=1))
        return 0
    if a.action in ("check", "note"):
        if not a.words:
            print(f"usage: brain desk {a.action} NOTE.json")
            return 2
        note = json.loads(Path(a.words[0]).read_text())
        case = load_case(note.get("check"))
        if a.action == "check":
            problems = check(note, case)
            print("\n".join(f"- {p}" for p in problems) if problems else "the note passes; record it with: brain desk note")
            return 1 if problems else 0
        try:
            rec = record_note(bank, case, note)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        print(show_note(rec))
        print(f"\nrecorded {rec['path']}; commit it with {calls_path().name}")
        return 0
    if a.action == "last":
        files = sorted(notes_dir().glob("????-??-??/*.json"), key=lambda p: (p.parent.name, p.name))
        if not files:
            print("no note recorded yet")
            return 0
        print(show_note(json.loads(files[-1].read_text())))
        return 0
    s = score(bank)
    print(f"Desk calls {s['calls']}: {s['judged']} judged, {s['open']} open, {s['withdrawn']} withdrawn.")
    if s["judged"]:
        print(f"Judged: {s['hits']} right of {s['judged']} ({s['hit_rate']:.0%}); stated chance on average "
              f"{s['mean_prob']:.0%}. Brier {s['brier']:.3f} (a coin flip scores 0.250; lower is better).")
    if not s["enough"]:
        print(f"Fewer than {s['min_calls']} judged calls: this record says nothing yet, either way.")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("desk", help="the case file for a check and the reasoned note written from it (docs/DESK.md)")
    p.add_argument("action", choices=["build", "show", "template", "check", "note", "last", "score"])
    p.add_argument("words", nargs="*", help="show: tickers to read in full; check, note: the note's JSON file")
    p.add_argument("--check", help="show: a case file by its check stamp (default: the latest)")
    p.add_argument("--no-texts", action="store_true", help="build: do not open the filings")
    p.add_argument("--no-accounts", action="store_true", help="build: do not read the accounts")
    p.set_defaults(fn=_cmd)
