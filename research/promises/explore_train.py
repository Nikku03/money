"""TRAIN-only exploration of the promise record (brain/promises.py): promises made 2011-01-01..2023-12-31, the
research universe.

Run before any declared test, and never on a later promise:
    PYTHONPATH=. python research/promises/explore_train.py > research/promises/explore_train.log

Each line is the rank correlation of something known at the time with an outcome, taken inside each reporting
season (the calendar quarter; at least MIN_N rows) and averaged over the seasons, with its t.
A. Delivery: what is known when a quarterly promise is made, against how the delivered revenue compared with its
   midpoint (``surprise``).
B. The day it delivers: what is known at the close before the report of the promised quarter, against that
   report session's move against the market (``r1`` of the episode library).
C. The day of the promise (same day, an explanation and not a forecast): the report session's move against what
   the release said.
D. After the promise: what the release said, against the week and the month after the report session.
E. Where the delivery falls around the midpoint moved by the habit (for promises.BAND68 and BAND90).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date

import numpy as np

from brain import episodes, promises
from brain.research import promisetest as pt

TRAIN = pt.TRAIN
MIN_N = pt.MIN_N


def ic(group, pred, y, min_n=MIN_N):
    group, pred, y = np.array(group), np.array(pred, float), np.array(y, float)
    _, d = pt.season_ic(group, pred, y, min_n)
    if len(d) < 3:
        return float("nan"), float("nan"), len(d), 0
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d), int((d > 0).sum())


def line(name, group, pred, y):
    k = np.isfinite(np.array(pred, float)) & np.isfinite(np.array(y, float))
    m, t, n, pos = ic(group, pred, y)
    print(f"  {name:<62} {m:+.3f}  t {t:+.1f}  ({pos} of {n} seasons positive, {int(k.sum())} rows)")


def col(rows, k, sign=1.0):
    return pt._col(rows, k, sign)


def main() -> int:
    names = set(pt.universe())
    lib = [r for r in promises.load() if r["t"] in names]
    eps = [e for e in episodes.load() if e["t"] in names]
    print(f"{len(lib)} promises of {len({r['t'] for r in lib})} companies of the research universe "
          f"({len(names)} companies)")
    Q, Y = pt.table(lib, eps, names)
    Q = [r for r in Q if TRAIN[0] <= r["made"] <= TRAIN[1]]
    Y = [r for r in Y if TRAIN[0] <= r["made"] <= TRAIN[1]]

    # B: the day the promised quarter is reported, with the record known at the close before it
    by_t = defaultdict(list)
    for r in lib:
        by_t[r["t"]].append(r)
    ep_t = defaultdict(list)
    for e in eps:
        ep_t[e["t"]].append(e)
    for v in ep_t.values():
        v.sort(key=lambda e: e["date"])
    post21 = {(e["t"], e["date"]): e.get("post21") for e in eps}
    B = []
    for t, rs in by_t.items():
        rs.sort(key=lambda r: r["known_at"])
        for r in rs:
            if r["horizon"] != "quarter" or not r.get("period_end") or not (TRAIN[0] <= r["made"] <= TRAIN[1]):
                continue
            here = pt._first_after(ep_t.get(t, []), r["made"], pt.HERE_DAYS)
            there = pt._first_after(ep_t.get(t, []), r["period_end"], 100)
            if there is None or there is here or there["date"] > TRAIN[1]:
                continue
            rec = promises.record(rs, there["pre"] + "T20:00:00Z")
            B.append({"season": pt.season(there["date"]), "r1": there["r1"],
                      "usual": rec["surprise"] if rec else None, "beat": rec["beat"] if rec else None,
                      "missed": rec["missed"] if rec else None, "stretch": r.get("stretch"),
                      "surprise": r.get("surprise")})
    for r in Q:
        r["got_vs_usual"] = (r["got"] - r["usual"]) if (r.get("got") is not None and r.get("usual") is not None) else None
    for r in Y:
        e = pt._first_after(ep_t.get(r["t"], []), r["made"], pt.HERE_DAYS)
        r["post21"] = post21.get((r["t"], e["date"])) if e else None
    for r in Q:
        e = pt._first_after(ep_t.get(r["t"], []), r["made"], pt.HERE_DAYS)
        r["post21"] = post21.get((r["t"], e["date"])) if e else None

    for name, lo, hi in (("TRAIN 2011-2023", "2011", "2023"), ("  2011-2018", "2011", "2018"),
                         ("  2019-2023", "2019", "2023")):
        q = [r for r in Q if lo <= r["made"][:4] <= hi]
        y = [r for r in Y if lo <= r["made"][:4] <= hi]
        b = [r for r in B if lo <= r["season"][:4] <= hi]
        s = col(q, "surprise")
        g = np.isfinite(s)
        print(f"\n{name}: {len(q)} quarterly promises of {len({r['t'] for r in q})} companies ({int(g.sum())} graded, "
              f"{int(np.isfinite(col(q, 'usual')).sum())} with a record), {len(y)} changes of a year's promise")
        if not g.any():
            continue
        print(f"  graded: {np.mean(s[g] > 0) * 100:.0f}% above the midpoint; median {np.median(s[g]) * 100:+.2f}%; "
              f"missed the range {np.mean([r['grade'] == 'missed' for r in q if r.get('grade')]) * 100:.1f}%")
        gq = [r["season"] for r in q]
        print(" A. Delivery (against the surprise on the promise)")
        line("its usual surprise (median of its last 8)", gq, col(q, "usual"), s)
        line("minus the stretch (growth asked less growth just done)", gq, col(q, "stretch", -1), s)
        line("what it delivered against its last promise", gq, col(q, "got"), s)
        print(" B. The day it delivers (against that report session's move)")
        gb = [r["season"] for r in b]
        line("its usual surprise", gb, col(b, "usual"), col(b, "r1"))
        line("the share it beat", gb, col(b, "beat"), col(b, "r1"))
        line("minus the share it missed", gb, col(b, "missed", -1), col(b, "r1"))
        line("minus the stretch of the promise", gb, col(b, "stretch", -1), col(b, "r1"))
        line("(oracle) the surprise it then delivered", gb, col(b, "surprise"), col(b, "r1"))
        print(" C. The day of the promise (same day; an explanation, not a forecast)")
        line("the stretch of the new promise", gq, col(q, "stretch"), col(q, "r1"))
        line("what it delivered against its last promise", gq, col(q, "got"), col(q, "r1"))
        line("that, less its usual surprise", gq, col(q, "got_vs_usual"), col(q, "r1"))
        gy = [r["season"] for r in y]
        line("the change in its promise for the year", gy, col(y, "change"), col(y, "r1"))
        print(" D. After the promise (the week, then the month after the report session)")
        for out in ("post5", "post21"):
            line(f"the stretch of the new promise -> {out}", gq, col(q, "stretch"), col(q, out))
            line(f"what it delivered against its last promise -> {out}", gq, col(q, "got"), col(q, out))
            line(f"that, less its usual surprise -> {out}", gq, col(q, "got_vs_usual"), col(q, out))
            line(f"the change in its promise for the year -> {out}", gy, col(y, "change"), col(y, out))
        if name.startswith("TRAIN"):
            u = col(q, "usual")
            k = np.isfinite(u) & g
            e = s[k] - u[k]
            qs = np.quantile(e, [0.05, 1 / 6, 0.5, 5 / 6, 0.95])
            print(" E. The surprise less the usual surprise (promises with a record)")
            print(f"  {int(k.sum())} rows; 5% {qs[0] * 100:+.2f}%, 1-in-6 {qs[1] * 100:+.2f}%, median {qs[2] * 100:+.2f}%, "
                  f"5-in-6 {qs[3] * 100:+.2f}%, 95% {qs[4] * 100:+.2f}%")
            print(f"  median distance of the delivery from the midpoint {np.median(np.abs(s[k])) * 100:.2f}%, from the "
                  f"midpoint moved by the habit {np.median(np.abs(e)) * 100:.2f}%")
            raised = [r for r in y if r["changed"] == "raised"]
            cut = [r for r in y if r["changed"] == "cut"]
            for nm, rr in (("raised", raised), ("cut", cut)):
                a = np.array([r["r1"] for r in rr])
                p5 = col(rr, "post5")
                print(f"  year's promise {nm}: {len(rr)} releases, report session {a.mean() * 100:+.2f}% "
                      f"({np.mean(a > 0) * 100:.0f}% rose), the week after {np.nanmean(p5) * 100:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
