"""TRAIN-only exploration of the promise record on the other universes (small, micro, the newer listings), for the
questions G1 held back: promises made 2011-01-01..2023-12-31.

Run before any declared test, and never on a later promise:
    PYTHONPATH=. python research/promises/explore_bands_train.py us_small us_micro us_watch \
        > research/promises/explore_bands_train.log

Each line is the rank correlation of something known at the time with an outcome, taken inside each reporting
season (the calendar quarter; at least MIN_N rows) and averaged over the seasons, with its t. It is printed for each
universe named and for all of them together.
A. Delivery: the habit and the stretch against the surprise on the new promise; the stretch against the surprise
   less the habit (does the stretch add to the habit?).
B. After a release: what it said against the week and the month after the report session.
C. The same day (an explanation, not a forecast).
D. For 2024-01..2026-09, counts only: how many promises and seasons a test would have. No outcome of those years is
   read here.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from brain import config, episodes, promises
from brain.research import promisetest as pt

TRAIN, TEST, MIN_N = pt.TRAIN, pt.TEST, pt.MIN_N


def tickers(name: str) -> set[str]:
    return {t.upper() for t in json.loads((config.ROOT / "universe" / f"{name}.json").read_text())["tickers"]}


def ic(group, pred, y):
    _, d = pt.season_ic(np.array(group), np.array(pred, float), np.array(y, float), MIN_N)
    if len(d) < 3:
        return float("nan"), float("nan"), len(d), 0
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d), int((d > 0).sum())


def line(name, group, pred, y):
    k = np.isfinite(np.array(pred, float)) & np.isfinite(np.array(y, float))
    m, t, n, pos = ic(group, pred, y)
    print(f"  {name:<66} {m:+.3f}  t {t:+.1f}  ({pos} of {n} seasons positive, {int(k.sum())} rows)")


def counts(name, group, pred, usable):
    """TEST years, counts only: rows with the prediction and an outcome on file, and seasons with MIN_N of them."""
    group, k = np.array(group), np.isfinite(np.array(pred, float)) & np.array(usable, bool)
    per = [int((k & (group == g)).sum()) for g in sorted(set(group.tolist()))]
    print(f"  {name:<66} {int(k.sum())} rows, {sum(n >= MIN_N for n in per)} seasons with {MIN_N}+ "
          f"(median {int(np.median([n for n in per if n >= MIN_N] or [0]))} rows a season)")


def report(label: str, names: set[str], lib: list[dict], eps: list[dict]) -> None:
    lib = [r for r in lib if r["t"] in names]
    eps = [e for e in eps if e["t"] in names]
    post21 = {(e["t"], e["date"]): e.get("post21") for e in eps}
    ep_t: dict[str, list[dict]] = {}
    for e in sorted(eps, key=lambda e: e["date"]):
        ep_t.setdefault(e["t"], []).append(e)
    Q, Y = pt.table(lib, eps, names)
    for r in Q + Y:
        e = pt._first_after(ep_t.get(r["t"], []), r["made"], pt.HERE_DAYS)
        r["post21"] = post21.get((r["t"], e["date"])) if e else None
    for r in Q:
        ok = r.get("got") is not None and r.get("got_habit") is not None
        r["got_vs_habit"] = (r["got"] - r["got_habit"]) if ok else None
        ok = r.get("surprise") is not None and r.get("habit") is not None
        r["beyond_habit"] = (r["surprise"] - r["habit"]) if ok else None
    print(f"\n===== {label}: {len(names)} companies, {len(lib)} promises of {len({r['t'] for r in lib})} of them =====")
    for name, lo, hi in (("TRAIN 2011-2023", "2011", "2023"), ("  2011-2018", "2011", "2018"),
                         ("  2019-2023", "2019", "2023")):
        q = [r for r in Q if TRAIN[0] <= r["made"] <= TRAIN[1] and lo <= r["made"][:4] <= hi]
        y = [r for r in Y if TRAIN[0] <= r["made"] <= TRAIN[1] and lo <= r["made"][:4] <= hi]
        s = pt._col(q, "surprise")
        g = np.isfinite(s)
        print(f"\n{name}: {len(q)} quarterly promises of {len({r['t'] for r in q})} companies ({int(g.sum())} graded, "
              f"{int(np.isfinite(pt._col(q, 'habit')).sum())} with a habit), {len(y)} changes of a year's promise")
        if not g.any():
            continue
        missed = np.mean([r["grade"] == "missed" for r in q if r.get("grade")]) * 100
        print(f"  graded: {np.mean(s[g] > 0) * 100:.0f}% above the midpoint; median {np.median(s[g]) * 100:+.2f}%; "
              f"missed the range {missed:.1f}%")
        gq, gy = [r["season"] for r in q], [r["season"] for r in y]
        print(" A. Delivery (against the surprise on the promise)")
        line("its habit (median of its last 4 deliveries)", gq, pt._col(q, "habit"), s)
        line("minus the stretch (growth asked less growth just done)", gq, pt._col(q, "stretch", -1), s)
        line("minus the stretch, against the surprise less the habit", gq, pt._col(q, "stretch", -1),
             pt._col(q, "beyond_habit"))
        print(" B. After a release (the week, then the month after the report session)")
        for out in ("post5", "post21"):
            line(f"the change in its promise for the year -> {out}", gy, pt._col(y, "change"), pt._col(y, out))
            line(f"what it delivered against its last promise -> {out}", gq, pt._col(q, "got"), pt._col(q, out))
            line(f"that, less the habit -> {out}", gq, pt._col(q, "got_vs_habit"), pt._col(q, out))
            line(f"the stretch of the new promise -> {out}", gq, pt._col(q, "stretch"), pt._col(q, out))
        print(" C. The same day (an explanation, not a forecast)")
        line("what it delivered against its last promise", gq, pt._col(q, "got"), pt._col(q, "r1"))
        line("that, less the habit that promise was made under", gq, pt._col(q, "got_vs_habit"), pt._col(q, "r1"))
        line("the change in its promise for the year", gy, pt._col(y, "change"), pt._col(y, "r1"))
        if name.startswith("TRAIN"):
            for nm in ("raised", "held", "cut"):
                rr = [r for r in y if r["changed"] == nm]
                if rr:
                    a = np.array([r["r1"] for r in rr])
                    print(f"  year's promise {nm}: {len(rr)} releases, report session {a.mean() * 100:+.2f}% "
                          f"({np.mean(a > 0) * 100:.0f}% rose), the week after "
                          f"{np.nanmean(pt._col(rr, 'post5')) * 100:+.2f}%, the month after "
                          f"{np.nanmean(pt._col(rr, 'post21')) * 100:+.2f}%")
    q = [r for r in Q if TEST[0] <= r["made"] <= TEST[1]]
    y = [r for r in Y if TEST[0] <= r["made"] <= TEST[1]]
    print(f"\n D. {TEST[0]}..{TEST[1]}, counts only: {len(q)} quarterly promises of {len({r['t'] for r in q})} "
          f"companies, {len(y)} changes of a year's promise of {len({r['t'] for r in y})}")
    gq, gy = [r["season"] for r in q], [r["season"] for r in y]
    counts("the habit, with a graded outcome", gq, pt._col(q, "habit"), [r.get("grade") is not None for r in q])
    counts("the stretch, with a graded outcome", gq, pt._col(q, "stretch"), [r.get("grade") is not None for r in q])
    counts("the change of a year's promise, with the week after", gy, pt._col(y, "change"),
           [r.get("post5") is not None for r in y])
    counts("a delivery against its habit, with the week after", gq,
           [r["got_vs_habit"] if r["got_vs_habit"] is not None else np.nan for r in q],
           [r.get("post5") is not None for r in q])


def main(argv: list[str]) -> int:
    names = argv or ["us_small", "us_micro", "us_watch"]
    lib, eps = promises.load(), episodes.load()
    sets = {n: tickers(n) for n in names}
    for n in names:
        report(n, sets[n], lib, eps)
    if len(names) > 1:
        report(" + ".join(names), set().union(*sets.values()), lib, eps)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
