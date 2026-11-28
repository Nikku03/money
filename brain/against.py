"""The engine against the analysts (docs/PROMISES.md): before a company reports, where its habit puts the revenue it
will deliver, set beside the analysts' estimate, written down before the outcome and judged after it.

    python -m brain against add TICKER --analysts 329.0e6 --report 2026-12-01 [--n 7] [--source "..."]
    python -m brain against show            # the lines still waiting for their report
    python -m brain against score           # who was closer, once the revenue is on file

The engine's guess is ``promises.expected``: the midpoint of the company's open quarterly promise moved by its
habit (G1a, G2a). The analysts' estimate is a figure the agent reads from a data connector or a page and passes in;
the engine fetches none. A line is refused once its report date has come, and at judging a line counts only when
the release that delivered the quarter became known after the line was written. Analysts mostly repeat the
company's own midpoint, and the habit says the company usually beats it: on eleven companies and 214 quarters of
2017-2023 the habit's guess was closer than the analysts' in 62% (``research/promises/trial_analysts_train.py``).
That was a trial on a few companies; this file is its forward record. It is about what a company delivers. It is
not a direction: on those quarters the stock did not rise more where the habit foresaw a larger beat.

Since 2026-11-28 a line also carries the second guess (the habit pulled in, ranges by the company's own spread;
G4a PASS). ``against second`` adds it to the lines written before it existed, each stamped with the time it was
added, and refuses a line whose report date has come.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from brain import config, episodes, promises
from brain.timeutil import now_iso

CALLS = config.ROOT / "research" / "against_calls.jsonl"
MIN_JUDGED = 30
SANE = (0.5, 2.0)            # an analysts' figure this far from the midpoint is a slip of units
AFTER_END = (5, 100)         # the delivering report session comes this many days after the quarter's end
TOL = 0.0005                 # misses closer than this are a tie


def _records(path: Path | None = None) -> list[dict]:
    p = path or CALLS
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def read_calls(path: Path | None = None) -> list[dict]:
    """The lines, without the second guesses added to them later."""
    return [r for r in _records(path) if not r.get("type")]


def read_seconds(path: Path | None = None) -> dict[str, dict]:
    """Second guesses added to earlier lines: line id -> the first one added."""
    out: dict[str, dict] = {}
    for r in _records(path):
        if r.get("type") == "second":
            out.setdefault(r["id"], r)
    return out


SECOND = ("usual", "spread", "point", "lo68", "hi68", "lo90", "hi90")


def line(rows: list[dict], ticker: str, analysts: float, report: str, *, n: int | None = None,
         source: str | None = None, now: str | None = None) -> dict:
    """The record for one company before its report (pure, given its promise rows). Refuses without an open
    promise and a habit, once the report date has come, or when the analysts' figure cannot be the same quantity."""
    now = now or now_iso()
    e = promises.expected(rows, now)
    if e is None:
        raise SystemExit(f"{ticker}: no open quarterly promise with a habit on file")
    date.fromisoformat(report)
    if report <= now[:10]:
        raise SystemExit(f"{ticker}: the report date ({report}) has come; a record is written before it")
    if not (SANE[0] <= analysts / e["mid"] <= SANE[1]):
        raise SystemExit(f"{ticker}: the analysts' figure ({analysts:,.0f}) is not near the promise's midpoint "
                         f"({e['mid']:,.0f}); check its units")
    return {"id": f"{ticker}-{e['made']}", "t": ticker, "made": e["made"], "report": report, "made_at": now,
            "mid": e["mid"], "habit": e["usual"], "point": e["point"], "lo68": e["lo68"], "hi68": e["hi68"],
            "lo90": e["lo90"], "hi90": e["hi90"], "analysts": float(analysts), "analysts_n": n, "source": source,
            "gap": e["point"] / analysts - 1, "second": {k: e["second"][k] for k in SECOND}}


def second_for(call: dict, rows: list[dict], now: str | None = None) -> dict:
    """The second guess for a line written before it existed (pure, given the company's promise rows). Refuses once
    the report date has come, or when the open promise is no longer the one the line was written for."""
    now = now or now_iso()
    if call["report"] <= now[:10]:
        raise SystemExit(f"{call['id']}: the report date ({call['report']}) has come; nothing is added after it")
    e = promises.expected(rows, now)
    if e is None or e["made"] != call["made"]:
        raise SystemExit(f"{call['id']}: its promise of {call['made']} is no longer the open one")
    return {"type": "second", "id": call["id"], "at": now, **{k: e["second"][k] for k in SECOND}}


def add_second(rec: dict, path: Path | None = None) -> dict:
    p = path or CALLS
    if rec["id"] in read_seconds(p) or any(c["id"] == rec["id"] and c.get("second") for c in read_calls(p)):
        raise SystemExit(f"{rec['id']} already has a second guess")
    with p.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def add(rec: dict, path: Path | None = None) -> dict:
    p = path or CALLS
    if any(r["id"] == rec["id"] for r in read_calls(p)):
        raise SystemExit(f"{rec['id']} is already recorded")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def judge(lib: list[dict], calls: list[dict], eps: list[dict] | None = None,
          seconds: dict[str, dict] | None = None) -> list[dict]:
    """Each line with its outcome once the promised quarter's revenue is on file (``actual`` None until then).
    ``valid`` is False when the release that delivered the quarter was already known when the line was written
    (from the episode library when it has that report, else from the accounts' date). A line's second guess (its
    own, or one added later and given in ``seconds``) is judged the same way, valid from the time it was written."""
    by = {}
    for r in lib:
        if r["horizon"] == "quarter" and r.get("actual") is not None and not r.get("mismatch"):
            by[(r["t"], r["made"])] = r
    ep_t: dict[str, list[dict]] = {}
    for e in sorted(eps or [], key=lambda e: e["date"]):
        ep_t.setdefault(e["t"], []).append(e)
    out = []
    for c in calls:
        r = by.get((c["t"], c["made"]))
        if r is None:
            out.append({**c, "actual": None})
            continue
        d0 = date.fromisoformat(r["period_end"])
        ep = next((e for e in ep_t.get(c["t"], [])
                   if AFTER_END[0] <= (date.fromisoformat(e["date"]) - d0).days <= AFTER_END[1]), None)
        known = (ep or {}).get("known_at") or r["outcome_known"]
        a = r["actual"]
        em, am = abs(a / c["point"] - 1), abs(a / c["analysts"] - 1)
        row = {**c, "actual": a, "valid": bool(known > c["made_at"]), "engine_miss": em, "analysts_miss": am,
               "closer": "tie" if abs(em - am) < TOL else "engine" if em < am else "analysts",
               "beat_analysts": bool(a > c["analysts"]), "in68": bool(c["lo68"] <= a <= c["hi68"]),
               "in90": bool(c["lo90"] <= a <= c["hi90"]), "r1": (ep or {}).get("r1")}
        g = c.get("second") or (seconds or {}).get(c["id"])
        if g:
            sm = abs(a / g["point"] - 1)
            row.update(second_valid=bool(known > (g.get("at") or c["made_at"])), second_miss=sm,
                       second_closer="tie" if abs(sm - am) < TOL else "second" if sm < am else "analysts",
                       second_in68=bool(g["lo68"] <= a <= g["hi68"]), second_in90=bool(g["lo90"] <= a <= g["hi90"]))
        out.append(row)
    return out


def score(lib: list[dict] | None = None, path: Path | None = None, eps: list[dict] | None = None) -> dict:
    calls = read_calls(path)
    j = judge(lib if lib is not None else promises.load(), calls, eps if eps is not None else episodes.load(),
              read_seconds(path))
    done = [x for x in j if x["actual"] is not None]
    ok = [x for x in done if x["valid"]]
    out = {"recorded": len(calls), "judged": len(ok), "void": len(done) - len(ok)}
    two = [x for x in ok if x.get("second_valid")]
    if two:
        out["second"] = {"judged": len(two), "closer": float(np.mean([x["second_closer"] == "second" for x in two])),
                         "analysts_closer": float(np.mean([x["second_closer"] == "analysts" for x in two])),
                         "miss": float(np.median([x["second_miss"] for x in two])),
                         "plain_miss": float(np.median([x["engine_miss"] for x in two])),
                         "analysts_miss": float(np.median([x["analysts_miss"] for x in two])),
                         "in68": float(np.mean([x["second_in68"] for x in two])),
                         "in90": float(np.mean([x["second_in90"] for x in two]))}
    if ok:
        out.update(engine_closer=float(np.mean([x["closer"] == "engine" for x in ok])),
                   analysts_closer=float(np.mean([x["closer"] == "analysts" for x in ok])),
                   engine_miss=float(np.median([x["engine_miss"] for x in ok])),
                   analysts_miss=float(np.median([x["analysts_miss"] for x in ok])),
                   beat_analysts=float(np.mean([x["beat_analysts"] for x in ok])),
                   in68=float(np.mean([x["in68"] for x in ok])), in90=float(np.mean([x["in90"] for x in ok])))
    return out


def _usd(x: float) -> str:
    return f"${x / 1e9:,.2f} bn" if abs(x) >= 1e9 else f"${x / 1e6:,.1f} mn"


def words(c: dict) -> str:
    n = f" ({c['analysts_n']} analysts)" if c.get("analysts_n") else ""
    return (f"{c['t']} reports {c['report']}: it promised {_usd(c['mid'])} (midpoint, {c['made']}); its habit "
            f"({c['habit'] * 100:+.1f}%) puts the delivery near {_usd(c['point'])}, 2 in 3 between {_usd(c['lo68'])} "
            f"and {_usd(c['hi68'])}; the analysts expect {_usd(c['analysts'])}{n}. The habit's guess is "
            f"{c['gap'] * 100:+.1f}% against the analysts'." + second_words(c.get("second"), c))


def second_words(g: dict | None, c: dict) -> str:
    if not g:
        return ""
    return (f" The second guess (the habit pulled in to {g['usual'] * 100:+.1f}%): {_usd(g['point'])}, 2 in 3 between "
            f"{_usd(g['lo68'])} and {_usd(g['hi68'])}; {(g['point'] / c['analysts'] - 1) * 100:+.1f}% against the "
            f"analysts'.")


def score_words(s: dict) -> list[str]:
    L = [f"Recorded {s['recorded']}, judged {s['judged']}" + (f", void {s['void']} (the report was out)" if s["void"] else "")]
    if not s["judged"]:
        return L + ["Nothing judged yet: a line is judged when its quarter's revenue is on file (python -m brain "
                    "promises build)."]
    L.append(f"The habit's guess was closer in {s['engine_closer'] * 100:.0f}%, the analysts in "
             f"{s['analysts_closer'] * 100:.0f}%; median miss {s['engine_miss'] * 100:.2f}% against "
             f"{s['analysts_miss'] * 100:.2f}%. The revenue beat the analysts in {s['beat_analysts'] * 100:.0f}%. The "
             f"2-in-3 range held {s['in68'] * 100:.0f}%, the 9-in-10 range {s['in90'] * 100:.0f}%.")
    if (g := s.get("second")):
        L.append(f"The second guess (the habit pulled in), {g['judged']} lines: closer than the analysts in "
                 f"{g['closer'] * 100:.0f}%, the analysts in {g['analysts_closer'] * 100:.0f}%; median miss "
                 f"{g['miss'] * 100:.2f}% (the plain habit {g['plain_miss'] * 100:.2f}%, the analysts "
                 f"{g['analysts_miss'] * 100:.2f}%). Its 2-in-3 range held {g['in68'] * 100:.0f}%, its 9-in-10 range "
                 f"{g['in90'] * 100:.0f}%.")
    if s["judged"] < MIN_JUDGED:
        L.append(f"Under {MIN_JUDGED} judged lines this says nothing yet.")
    return L


def _cmd(a) -> int:
    if a.action == "add":
        if not a.ticker or a.analysts is None or not a.report:
            raise SystemExit("against add TICKER --analysts X --report YYYY-MM-DD")
        t = a.ticker.upper()
        rows = [r for r in promises.load() if r["t"] == t]
        print("recorded: " + words(add(line(rows, t, a.analysts, a.report, n=a.n, source=a.source))))
        return 0
    if a.action == "second":
        lib, have, n = promises.load(), read_seconds(), 0
        for c in read_calls():
            if c.get("second") or c["id"] in have:
                continue
            try:
                g = add_second(second_for(c, [r for r in lib if r["t"] == c["t"]]))
            except SystemExit as e:
                print(f"not added: {e}")
                continue
            n += 1
            print(f"added: {c['t']} reports {c['report']}:{second_words(g, c)}")
        print(f"{n} second guesses added")
        return 0
    if a.action == "show":
        seconds = read_seconds()
        j = judge(promises.load(), read_calls(), episodes.load(), seconds)
        wait = [x for x in j if x["actual"] is None]
        for c in sorted(wait, key=lambda c: c["report"]):
            print(words(c) + ("" if c.get("second") else second_words(seconds.get(c["id"]), c)))
        print(f"{len(wait)} waiting for their report, {len(j) - len(wait)} with an outcome (against score)")
        return 0
    for ln in score_words(score()):
        print(ln)
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("against", help="the habit's guess of a company's revenue against the analysts', recorded "
                                       "before its report and judged after")
    p.add_argument("action", choices=["add", "second", "show", "score"])
    p.add_argument("ticker", nargs="?")
    p.add_argument("--analysts", type=float, help="the analysts' revenue estimate for the promised quarter, in dollars")
    p.add_argument("--report", help="the date the company reports (YYYY-MM-DD), still ahead")
    p.add_argument("--n", type=int, help="how many analysts")
    p.add_argument("--source", help="where the estimate was read")
    p.set_defaults(fn=_cmd)
