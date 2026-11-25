"""TRAIN-only (promises made 2011-2023): which simple measure of a company's habit says most about the surprise
on its next quarterly promise. Run after explore_train.py showed that the surprise the company has just delivered
(public in the release that makes the new promise) says more than the median of its last eight filed outcomes.

    PYTHONPATH=. python research/promises/explore_habit_train.py > research/promises/explore_habit_train.log
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from brain import episodes, promises
from brain.research import promisetest as pt


def main() -> int:
    names = set(pt.universe())
    lib = [r for r in promises.load() if r["t"] in names]
    by_t = defaultdict(list)
    for r in lib:
        if r["horizon"] == "quarter":
            by_t[r["t"]].append(r)
    rows = []
    for t, rs in by_t.items():
        rs.sort(key=lambda r: r["known_at"])
        hist = []                                   # surprises delivered so far, in order, as each release showed them
        prev = None
        for r in rs:
            got = None
            if prev is not None and prev.get("surprise") is not None and prev["period_end"] < r["made"]:
                got = prev["surprise"]
                hist.append(got)
            elif prev is not None:
                hist.append(None)                   # a gap: the promise before has no graded outcome
            if pt.TRAIN[0] <= r["made"] <= pt.TRAIN[1] and r.get("surprise") is not None:
                rec = promises.record(rs, r["known_at"])
                seen = [h for h in hist if h is not None]
                rows.append({"season": pt.season(r["made"]), "y": r["surprise"], "made": r["made"],
                             "usual": rec["surprise"] if rec else None, "got": got,
                             "med8": float(np.median(seen[-8:])) if len(seen) >= 4 else None,
                             "med4": float(np.median(seen[-4:])) if len(seen) >= 4 else None,
                             "mean4": float(np.mean(seen[-4:])) if len(seen) >= 4 else None,
                             "mean2": float(np.mean(seen[-2:])) if len(seen) >= 2 and got is not None else None,
                             "n_seen": len(seen)})
            prev = r
    for r in rows:
        r["half"] = (0.5 * r["got"] + 0.5 * r["med8"]) if (r["got"] is not None and r["med8"] is not None) else None
    g = np.array([r["season"] for r in rows])
    y = np.array([r["y"] for r in rows])
    print(f"{len(rows)} graded quarterly promises made in TRAIN, the research universe")
    print("rank correlation with the surprise on the new promise (per season, averaged), and how far the delivery "
          "fell from the midpoint moved by the measure (median)")
    for k, label in (("usual", "median of its last 8 outcomes filed by then (the record)"),
                     ("got", "the surprise it has just delivered (in the release)"),
                     ("mean2", "mean of its last 2 delivered, the latest included"),
                     ("med4", "median of its last 4 delivered, the latest included"),
                     ("mean4", "mean of its last 4 delivered, the latest included"),
                     ("med8", "median of its last 8 delivered, the latest included"),
                     ("half", "half the latest, half the median of its last 8")):
        x = np.array([r[k] if r[k] is not None else np.nan for r in rows], float)
        ok = np.isfinite(x)
        _, d = pt.season_ic(g, x, y)
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        e = y[ok] - x[ok]
        qs = np.quantile(e, [0.05, 1 / 6, 5 / 6, 0.95])
        halves = []
        for lo, hi in (("2011", "2018"), ("2019", "2023")):
            kk = np.array([lo <= r["made"][:4] <= hi for r in rows])
            _, dh = pt.season_ic(g[kk], x[kk], y[kk])
            halves.append(f"{dh.mean():+.3f}" if len(dh) else "n/a")
        print(f"  {label:<58} {d.mean():+.3f} t {t:+.1f} ({int((d > 0).sum())} of {len(d)}; halves {halves[0]}, "
              f"{halves[1]}; {int(ok.sum())} rows) | off by {np.median(np.abs(e)) * 100:.2f}% (midpoint alone "
              f"{np.median(np.abs(y[ok])) * 100:.2f}%) | residual 5% {qs[0] * 100:+.2f} 1-in-6 {qs[1] * 100:+.2f} "
              f"5-in-6 {qs[2] * 100:+.2f} 95% {qs[3] * 100:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
