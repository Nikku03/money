"""Before a report: what the price expects, the past episodes most like it, what followed them and why
(docs/EARNINGS.md).

The present (``setup_now``): how far the stock has run into the report against the market, in its own usual moves;
how far below its 52-week high it sits; its habit at its last eight reports (the usual size, the mean signed
reaction, how often it rose); the VIX and the market's last month. The past (``analogs``): the episodes of the
library (``brain/episodes.py``) whose setups were closest, from before today, with how they reacted on the report
session, what followed over the next week and month, and why, in the library's fixed names (priced in, relief,
confirmed, capitulation, surprise up or down). The stock's own last reports, named the same way.

Whether any of this predicts is for the declared tests X1a-X1e (``brain/research/earntest.py``) and the forward
record: the card prints their verdicts as recorded and labels the analogs' lean untested until X1c passes. A lean is
not a call; a call made from a card is recorded first (``record``, or ``brain predict add``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from brain import config, episodes
from brain.bank import Bank
from brain.timeutil import now_iso

K = 100              # as X1c (brain/research/earntest.py)
CALLS = config.ROOT / "research" / "earn_calls.jsonl"
TESTS = ("X1a", "X1b", "X1c", "X1d", "X1f")


def _library(lib: list[dict] | None) -> list[dict]:
    return lib if lib is not None else episodes.load()


def own(bank: Bank, ticker: str, book=None) -> tuple[list[dict], dict | None]:
    """The stock's past episodes and its setup at the bank's latest close."""
    from brain.metrics import PriceBook
    t = ticker.upper()
    view = bank.view()
    book = book or PriceBook(bank, [t, episodes.BENCH, episodes.VIX])
    rows = episodes.ticker_episodes(view, book, t)
    if not (book.has(t) and book.has(episodes.BENCH) and book.has(episodes.VIX)):
        return rows, None
    dates, adj = book.adj_upto(view, t)
    bdates, badj = book.adj_upto(view, episodes.BENCH)
    vdates, _ = book.adj_upto(view, episodes.VIX)
    bclose = episodes.aligned(dates, bdates, badj)
    vix = episodes.aligned(dates, vdates, book.close[episodes.VIX][:len(vdates)])
    from brain import events
    evs = events.distinct(events.earnings_events(view, t))
    prev = []
    for e in evs:
        i = events.event_index(dates, e.known_at)
        if i - 2 >= 0 and i + 1 < len(dates):
            x = episodes._excess(adj, bclose, i - 2, i + 1)
            if x is not None:
                prev.append((i, x))
    p = len(dates) - 1
    s = episodes.setup(adj, bclose, vix, p, prev)
    if s:
        s.update({"t": t, "asof": str(dates[p]), "price": float(book.close[t][:len(dates)][p])})
        pending = [e for e in evs if events.event_index(dates, e.known_at) > p]
        if pending:
            s["pending"] = pending[-1].known_at     # a report is out, its session has not closed
    return rows, s


def analogs(setup: dict, lib: list[dict] | None = None, k: int = K, before: str | None = None) -> dict | None:
    """The ``k`` library episodes closest to ``setup`` from before ``before`` (default: its date), and what they did."""
    lib = episodes.complete(_library(lib), "r1")
    before = before or setup.get("asof") or "9999"
    pool = [r for r in lib if r["date"] < before]
    if len(pool) < k:
        return None
    train = [r for r in pool if episodes.TRAIN_START <= r["date"] <= episodes.TRAIN_END] or pool
    med, mad = episodes.scale(episodes.matrix(train))
    P = np.clip((episodes.matrix(pool) - med) / mad, -6, 6)
    x = np.clip((np.array([float(setup[f]) for f in episodes.FEATS]) - med) / mad, -6, 6)
    nn = episodes.nearest(P, x, k)
    near = [pool[i] for i in nn]
    r1 = np.array([r["r1"] for r in near])
    post5 = np.array([r["post5"] for r in near if r.get("post5") is not None])
    kinds: dict[str, int] = {}
    for r in near:
        kinds[r.get("why") or "?"] = kinds.get(r.get("why") or "?", 0) + 1
    up = [r for r in near if r["r1"] > 0 and r.get("post5") is not None]
    dn = [r for r in near if r["r1"] < 0 and r.get("post5") is not None]
    return {"n": len(near), "mean_r1": float(r1.mean()), "median_r1": float(np.median(r1)),
            "rose": float((r1 > 0).mean()), "abs_r1": float(np.median(np.abs(r1))),
            "post5": float(post5.mean()) if len(post5) else None,
            "post5_after_up": float(np.mean([r["post5"] for r in up])) if up else None,
            "post5_after_down": float(np.mean([r["post5"] for r in dn])) if dn else None,
            "why": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
            "examples": [{"t": r["t"], "date": r["date"], "why": r.get("why"), "r1": r["r1"],
                          "run21": r["run21"]} for r in near[:8]],
            "base_rose": float(np.mean([r["r1"] > 0 for r in pool])),
            "base_abs": float(np.median([abs(r["r1"]) for r in pool]))}


def score_view(setup: dict, lib: list[dict] | None = None) -> dict | None:
    """The expectation score (X1f: z(-run21) + z(-run5) + z(up8), TRAIN-scaled) of ``setup``, where it falls among
    the TRAIN years' reports, and how each fifth of them reacted."""
    from brain.research import earntest
    train = [r for r in episodes.complete(_library(lib), "r1")
             if episodes.TRAIN_START <= r["date"] <= episodes.TRAIN_END]
    if len(train) < 1000:
        return None
    sc = earntest.score_scale(train)
    x = np.array([earntest.expectation_score(r, sc) for r in train])
    y = np.array([r["r1"] for r in train])
    cuts = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    fifth = [{"mean": float(y[np.searchsorted(cuts, x) == k].mean()),
              "rose": float((y[np.searchsorted(cuts, x) == k] > 0).mean())} for k in range(5)]
    v = earntest.expectation_score(setup, sc)
    return {"score": v, "pct": float((x < v).mean()), "fifth": int(np.searchsorted(cuts, v)), "fifths": fifth}


def verdicts() -> dict[str, str]:
    """The recorded verdicts of the declared tests X1a-X1d ("not run" when absent)."""
    try:
        from brain.research import run
        v = {r["id"]: r["verdict"].split(":")[0] for r in run.verdicts() if r.get("kind") == "earn"}
    except Exception:  # noqa: BLE001 - a card without verdicts says so
        v = {}
    return {t: v.get(t, "not run") for t in TESTS}


def setup_words(s: dict) -> str:
    ran = ("ran up" if s["run21"] >= episodes.RAN else "sold off" if s["run21"] <= -episodes.RAN
           else "kept close to the market")
    return (f"into the report it {ran}: {s['run21']:+.1f} of its usual months against the market over 21 sessions "
            f"({s['run5']:+.1f} over 5, {s['run63']:+.1f} over 63); {abs(s['hi52']) * 100:.0f}% below its 52-week high; "
            f"its usual report move {s['med8'] * 100:.1f}%, last {s['last'] * 100:.1f}%; at its last eight reports "
            f"it rose {s['up8'] * 100:.0f}% of the time, mean {s['mean8']:+.2f} of its usual move; VIX {s['vix']:.1f}, "
            f"the market {s['spy21'] * 100:+.1f}% over the last month")


def card(bank: Bank, ticker: str, lib: list[dict] | None = None) -> dict:
    rows, s = own(bank, ticker)
    lib = _library(lib)
    a = analogs(s, lib) if s else None
    sv = score_view(s, lib) if s else None
    past = [r for r in rows if r.get("why")][-8:]
    return {"t": ticker.upper(), "setup": s, "analogs": a, "score": sv, "own": past, "verdicts": verdicts()}


def render(c: dict) -> list[str]:
    t, s, a = c["t"], c["setup"], c["analogs"]
    L = [f"{t}: before its next report (the setup at the close of {s['asof'] if s else '?'})"]
    if s and s.get("pending"):
        L[0] = (f"{t}: a report was filed at {s['pending'][:16]}Z and its session has not closed; this is the setup "
                f"at the close before it ({s['asof']})")
    if not s:
        return L + ["  not enough history: it needs four earlier reports, 160 ordinary sessions and the VIX"]
    L.append("  now: " + setup_words(s))
    if (sv := c.get("score")):
        f = sv["fifths"]
        names = ("lowest", "second", "middle", "fourth", "highest")
        L.append(f"  expectation score {sv['score']:+.2f} ({names[sv['fifth']]} fifth of 2011-2023's reports; high means "
                 f"it has not run up and usually rises at reports): that fifth rose {f[sv['fifth']]['rose'] * 100:.0f}% "
                 f"of the time on the report session, mean {f[sv['fifth']]['mean'] * 100:+.2f}% (highest fifth "
                 f"{f[4]['rose'] * 100:.0f}%, {f[4]['mean'] * 100:+.2f}%; lowest {f[0]['rose'] * 100:.0f}%, "
                 f"{f[0]['mean'] * 100:+.2f}%)")
    if a:
        L.append(f"  the {a['n']} most similar past setups (any stock, before today): on the report session "
                 f"{a['rose'] * 100:.0f}% rose (all reports: {a['base_rose'] * 100:.0f}%), mean "
                 f"{a['mean_r1'] * 100:+.2f}% against the market, median size {a['abs_r1'] * 100:.1f}% (all: "
                 f"{a['base_abs'] * 100:.1f}%)")
        L.append("  why they moved: " + ", ".join(f"{k} {v}" for k, v in list(a["why"].items())[:6]))
        if a["post5_after_up"] is not None and a["post5_after_down"] is not None:
            L.append(f"  the week after: those that rose then moved {a['post5_after_up'] * 100:+.2f}%, those that fell "
                     f"{a['post5_after_down'] * 100:+.2f}% (against the market)")
        L.append("  closest: " + "; ".join(f"{e['t']} {e['date']} {e['why']} {e['r1'] * 100:+.1f}%"
                                         for e in a["examples"][:5]))
    if c["own"]:
        L.append("  its own last reports: " + "; ".join(
            f"{r['date']} {r['why']} {r['r1'] * 100:+.1f}%" + (f", week after {r['post5'] * 100:+.1f}%"
                                                               if r.get("post5") is not None else "")
            for r in c["own"][-6:]))
    v = c["verdicts"]
    L.append("  tests (research/RESULTS.md): " + ", ".join(f"{k} {x}" for k, x in v.items())
             + ". A score or a lean whose test has not passed is untested; neither is a call until recorded.")
    if v.get("X1a") == "PASS":
        L.append("  how to read X1a's PASS: the further a stock runs into its report, the more often it falls on the report "
                 "session (its typical reaction is lower), but not its average: on 2024-2026 the fifth that ran up most "
                 "rose 49% of the time with a median of -0.13%, yet averaged +0.49% on its big winners "
                 "(research/earnings/after_the_fact_x1a.log). A tilt in the odds, not a trade")
    return L


# ---------- the forward record ----------
def read_calls(path: Path | None = None) -> list[dict]:
    p = path or CALLS
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def record(bank: Bank, ticker: str, report: str, lib: list[dict] | None = None, path: Path | None = None) -> dict:
    """Write the card's numbers for a coming report before it is public: judged later on its r1."""
    c = card(bank, ticker, lib)
    if not c["setup"] or not c["analogs"]:
        raise SystemExit(f"{ticker}: no setup or no analogs")
    if c["setup"].get("pending"):
        raise SystemExit(f"{ticker}: a report is already public ({c['setup']['pending'][:16]}Z); a record comes before it")
    if report <= c["setup"]["asof"]:
        raise SystemExit(f"{ticker}: the report ({report}) must come after the setup's close ({c['setup']['asof']})")
    a = c["analogs"]
    sv = c.get("score") or {}
    rec = {"id": f"{c['t']}-{report}", "t": c["t"], "report": report, "made_at": now_iso(),
           "asof": c["setup"]["asof"], "price": c["setup"]["price"],
           "setup": {k: c["setup"][k] for k in (*episodes.FEATS, "day_vol", "last")},
           "lean": a["mean_r1"], "rose": a["rose"], "abs": a["abs_r1"], "why": a["why"], "verdicts": c["verdicts"],
           "score": sv.get("score"), "score_pct": sv.get("pct")}
    p = path or CALLS
    p.parent.mkdir(parents=True, exist_ok=True)
    if any(r["id"] == rec["id"] for r in read_calls(p)):
        raise SystemExit(f"{rec['id']} is already recorded")
    with p.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def judge(lib: list[dict], calls: list[dict]) -> list[dict]:
    """Each recorded call with its outcome, from the library's episode on that report (None until it is in)."""
    by = {(r["t"], r["pre"]): r for r in lib if r.get("r1") is not None}
    out = []
    for c in calls:
        ep = by.get((c["t"], c["asof"]))
        out.append({**c, "r1": ep["r1"] if ep else None, "why_now": ep.get("why") if ep else None})
    return out


def score(lib: list[dict] | None = None, path: Path | None = None) -> dict:
    j = [x for x in judge(_library(lib), read_calls(path)) if x["r1"] is not None]
    if not j:
        return {"recorded": len(read_calls(path)), "judged": 0}
    lean = np.array([x["lean"] for x in j])
    r1 = np.array([x["r1"] for x in j])
    hit = float(np.mean(np.sign(lean) == np.sign(r1)))
    weeks = len({np.datetime64(x["report"], "W") for x in j})
    return {"recorded": len(read_calls(path)), "judged": len(j), "weeks": weeks, "sign_hit": hit,
            "mean_r1_when_lean_up": float(r1[lean > 0].mean()) if (lean > 0).any() else None,
            "mean_r1_when_lean_down": float(r1[lean < 0].mean()) if (lean < 0).any() else None}


# ---------- who reports soon ----------
_DAY = __import__("re").compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday), ([A-Z][a-z]+) (\d{1,2})$")
_ROW = __import__("re").compile(r"\(([A-Z][A-Z.\-]{0,6})\)$")
_ALSO = __import__("re").compile(r"^(Before the open|After the close|Time not announced): ([A-Z][A-Z.\-, ]+)\.?$")
_SESSIONS = {"Before the open": "before", "After the close": "after", "Time not announced": "unknown"}


def parse_calendar(text: str, year: int) -> dict[str, dict[str, list[str]]]:
    """A weekly earnings calendar article's text (StockTitan's "Earnings Calendar for the Week of ...") as
    {date: {"before": [...], "after": [...], "unknown": [...]}}: the company rows (``Name (TICKER)``) under each day
    and session heading, and the "Also reporting" lines."""
    from datetime import datetime
    out: dict[str, dict[str, list[str]]] = {}
    day, session = None, None
    for raw in text.splitlines():
        line = raw.strip()
        if (m := _DAY.match(line)):
            try:
                day = datetime.strptime(f"{m.group(2)} {m.group(3)} {year}", "%B %d %Y").date().isoformat()
            except ValueError:
                day = None
            session = None
            continue
        if day is None:
            continue
        if line in _SESSIONS:
            session = _SESSIONS[line]
            continue
        if (m := _ALSO.match(line)):
            ses = _SESSIONS[m.group(1)]
            for t in (x.strip().rstrip(".") for x in m.group(2).split(",")):
                if t and t not in out.setdefault(day, {}).setdefault(ses, []):
                    out[day][ses].append(t)
            continue
        if line.startswith("Source:") or line.startswith("Sec."):
            session = None if line.startswith("Sec.") else session
            if line.startswith("Sec."):
                day = None
            continue
        if session and (m := _ROW.search(line)) and m.group(1) not in ("ET", "EPS"):
            t = m.group(1).rstrip(".")
            if t not in out.setdefault(day, {}).setdefault(session, []):
                out[day][session].append(t)
    return out


def calendar_url(monday: str) -> str:
    from datetime import date
    d = date.fromisoformat(monday)
    return f"https://www.stocktitan.net/articles/earnings-calendar-week-of-{d.strftime('%B').lower()}-{d.day}-{d.year}"


def confirmed(session_date: str, gateway=None) -> dict[str, str] | None:
    """The tickers whose report session (the session that trades on the report) is ``session_date``, from the weekly
    calendar of its week and the week before: ticker -> "before the open on D" or "after the close on D-1". None
    when no calendar page could be read (the web rules apply; a refused page is not got around)."""
    from datetime import date, timedelta
    from brain.web import extract
    from brain.web.gateway import Gateway
    d = date.fromisoformat(session_date)
    mondays = sorted({(d - timedelta(days=d.weekday())).isoformat(),
                      (d - timedelta(days=1) - timedelta(days=(d - timedelta(days=1)).weekday())).isoformat()})
    cal: dict[str, dict[str, list[str]]] = {}
    g = gateway or Gateway(mode="http")
    read = False
    try:
        for mon in mondays:
            u = calendar_url(mon)
            pg = g.open(u)
            if not pg.ok:
                continue
            read = True
            for k, v in parse_calendar(extract.article(pg.html, u)["text"], date.fromisoformat(mon).year).items():
                for ses, ts in v.items():
                    cal.setdefault(k, {}).setdefault(ses, []).extend(ts)
    finally:
        if gateway is None:
            g.close()
    if not read:
        return None
    out = {t: f"before the open on {session_date}" for t in cal.get(session_date, {}).get("before", [])}
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    for t in cal.get(prev.isoformat(), {}).get("after", []):
        out.setdefault(t, f"after the close on {prev.isoformat()}")
    return out



def likely(lib: list[dict], start: str, end: str, slack: int = 6) -> dict[str, str]:
    """Tickers whose report a year before fell within [start - 1 year - slack, end - 1 year + slack] days and that have
    not reported in the 45 days to ``start``: ticker -> last year's report session (the guess's basis)."""
    from datetime import date, timedelta
    s0, e0 = date.fromisoformat(start), date.fromisoformat(end)
    lo, hi = (s0 - timedelta(days=364 + slack)).isoformat(), (e0 - timedelta(days=364 - slack)).isoformat()
    recent = (s0 - timedelta(days=45)).isoformat()
    by: dict[str, list[str]] = {}
    for r in lib:
        by.setdefault(r["t"], []).append(r["date"])
    out = {}
    for t, ds in by.items():
        ds.sort()
        if any(recent <= d < start for d in ds):
            continue
        hit = [d for d in ds if lo <= d <= hi]
        if hit:
            out[t] = hit[-1]
    return out


def radar(bank: Bank, lib: list[dict], start: str, end: str, limit: int = 40) -> list[dict]:
    """Cards for the stocks likely to report between ``start`` and ``end`` (by last year's dates), biggest usual
    report moves first."""
    cand = likely(lib, start, end)
    cards = []
    for t, basis in cand.items():
        try:
            c = card(bank, t, lib)
        except Exception:  # noqa: BLE001 - one stock's missing prices must not stop the radar
            continue
        if c["setup"] and c["analogs"]:
            c["basis"] = basis
            cards.append(c)
    cards.sort(key=lambda c: -c["setup"]["med8"])
    return cards[:limit]


# ---------- the case file ----------
def desk_line(bank: Bank, ticker: str, asof: str, lib: list[dict]) -> str | None:
    """For a mover with a results report since the close: the setup before it and its analogs, one line."""
    rows, _ = own(bank, ticker)
    ep = next((r for r in reversed(rows) if r.get("pre") == asof), None)
    s = None
    if ep and all(ep.get(f) is not None for f in episodes.FEATS):
        s = {**ep, "asof": ep["date"]}
    else:
        _, s = own(bank, ticker)
    if not s:
        return None
    a = analogs(s, lib, before=asof)
    out = "before the report: " + setup_words(s)
    if a:
        out += (f". Of the {a['n']} most similar past setups, {a['rose'] * 100:.0f}% rose on the report session "
                f"(all reports {a['base_rose'] * 100:.0f}%), mean {a['mean_r1'] * 100:+.2f}%; their kinds: "
                + ", ".join(f"{k} {v}" for k, v in list(a["why"].items())[:4]))
    v = verdicts()
    return out + (f". Tests (research/RESULTS.md): the run into a report X1a {v['X1a']} (a run-up makes a fall on the "
                  f"report session more likely, not a lower average); the similar past setups X1c {v['X1c']}; the "
                  f"expectation score X1f {v['X1f']}")


def _cmd(a) -> int:
    bank = Bank()
    if a.action == "build":
        rows = episodes.build(bank)
        p = episodes.save(rows)
        print(f"{len(rows)} episodes written to {p}")
        return 0
    lib = episodes.load()
    if not lib and a.action != "score":
        print("no episode library yet: run `python -m brain expect build` (about 15 minutes)")
        return 1
    if a.action == "show":
        for t in a.tickers:
            print("\n".join(render(card(bank, t, lib))))
            print()
    elif a.action == "record":
        if not a.report:
            print("--report YYYY-MM-DD is required")
            return 1
        for t in a.tickers:
            r = record(bank, t, a.report, lib)
            print(f"recorded {r['id']}: lean {r['lean'] * 100:+.2f}%, {r['rose'] * 100:.0f}% of the analogs rose; "
                  f"commit {CALLS.relative_to(config.ROOT)}")
    elif a.action == "radar":
        from datetime import date, timedelta
        asof = bank.view().t[:10]
        start = a.report or (date.fromisoformat(asof) + timedelta(days=1)).isoformat()
        end = (date.fromisoformat(start) + timedelta(days=max(a.days, 1) - 1)).isoformat()
        if a.confirmed:
            conf = confirmed(start)
            if conf is None:
                print(f"no weekly calendar could be read for {start}; nothing recorded")
                return 1
            cards = []
            for t, when in sorted(conf.items()):
                try:
                    c = card(bank, t, lib)
                except Exception:  # noqa: BLE001 - a company outside the bank is skipped
                    continue
                if c["setup"] and c["analogs"]:
                    c["basis"] = when
                    cards.append(c)
            cards.sort(key=lambda c: -c["setup"]["med8"])
            print(f"Confirmed to report for the session of {start} (the weekly calendar): {len(conf)} listed, "
                  f"{len(cards)} with a setup in the bank")
        else:
            cards = radar(bank, lib, start, end)
            print(f"Likely to react to a report between {start} and {end} (last year's dates; a guess, not a "
                  f"calendar): {len(cards)} with a setup")
        for c in cards:
            s, an, sv = c["setup"], c["analogs"], c.get("score") or {}
            out = f"  [out {s['pending'][:16]}Z]" if s.get("pending") else ""
            print(f"  {c['t']:6s} usual move {s['med8'] * 100:4.1f}%  run21 {s['run21']:+.1f}  rose at {s['up8'] * 100:3.0f}% "
                  f"of its last 8  score {sv.get('score', float('nan')):+.2f}  analogs: {an['rose'] * 100:.0f}% rose, mean "
                  f"{an['mean_r1'] * 100:+.2f}%  ({'' if a.confirmed else 'last year '}{c['basis']}){out}")
            if a.record and not s.get("pending"):
                try:
                    record(bank, c["t"], start, lib)
                except SystemExit as e:
                    print(f"    not recorded: {e}")
        if a.record:
            print(f"recorded before the reports; commit {CALLS.relative_to(config.ROOT)}")
        v = verdicts()
        print(f"Tests (research/RESULTS.md): run-up X1a {v['X1a']} (more often down after a run-up, not a lower average), "
              f"analogs X1c {v['X1c']}, score X1f {v['X1f']}. A lean is not a call; [out] marks a report already public.")
    elif a.action == "score":
        s = score(lib)
        print(json.dumps(s, indent=1))
        if s.get("weeks", 0) < 30:
            print("Fewer than 30 weeks of judged reports: this record says nothing yet, either way.")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("expect", help="before a report: what the price expects, the past setups like it, what "
                                      "followed and why (docs/EARNINGS.md)")
    p.add_argument("action", choices=["build", "show", "record", "score", "radar"])
    p.add_argument("tickers", nargs="*")
    p.add_argument("--report", help="record: the report's session, YYYY-MM-DD (radar: the session they likely react)")
    p.add_argument("--days", type=int, default=1, help="radar: how many calendar days ahead to look (default 1)")
    p.add_argument("--record", action="store_true", help="radar: record each card before its report")
    p.add_argument("--confirmed", action="store_true",
                   help="radar: only the companies a published weekly calendar lists for that session")
    p.set_defaults(fn=_cmd)
