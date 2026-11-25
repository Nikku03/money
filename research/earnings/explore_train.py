"""Exploration on TRAIN years only (2011-2023): which parts of the setup before a report relate to the reaction
(r1, the trade made before the report and closed the next session) and to the drift after it (post5, post21).
TEST years (2024 on) are not read here. Usage: PYTHONPATH=. python research/earnings/explore_train.py"""
from __future__ import annotations

import sys

import numpy as np

from brain import episodes as ep

TRAIN = ("2011-01-01", "2023-12-31")


def ranks(x):
    o = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return o / max(len(x) - 1, 1)


def monthly_ic(rows, feat, key, min_n=30):
    by = {}
    for r in rows:
        by.setdefault(r["date"][:7], []).append(r)
    ics = []
    for m, rs in sorted(by.items()):
        x = np.array([float(r[feat]) for r in rs])
        y = np.array([float(r[key]) for r in rs])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_n:
            continue
        ics.append(np.corrcoef(ranks(x[ok]), ranks(y[ok]))[0, 1])
    ics = np.array(ics)
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 2 else float("nan")
    return ics.mean(), t, len(ics), (ics > 0).mean()


def quintiles(rows, feat, key):
    x = np.array([float(r[feat]) for r in rows])
    y = np.array([float(r[key]) for r in rows])
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    q = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    b = np.searchsorted(q, x)
    return [(float(np.mean(y[b == k])) * 100, float(np.mean(y[b == k] > 0)) * 100, int((b == k).sum())) for k in range(5)]


def main():
    rows = [r for r in ep.load() if TRAIN[0] <= r["date"] <= TRAIN[1]]
    full = ep.complete(rows, "r1")
    print(f"TRAIN episodes: {len(rows)}; with a full setup and r1: {len(full)}")
    print(f"r1 mean {np.mean([r['r1'] for r in full]) * 100:+.3f}%, rose {np.mean([r['r1'] > 0 for r in full]) * 100:.1f}%")
    for key in ("r1", "post5", "post21"):
        rs = [r for r in full if r.get(key) is not None]
        print(f"\n== {key}: {len(rs)} episodes; monthly rank IC (mean, t, months, share positive)")
        for f in (*ep.FEATS, "day_vol", "last"):
            ic, t, n, pos = monthly_ic(rs, f, key)
            print(f"  {f:7s} {ic:+.4f}  t {t:+5.2f}  {n} months  {pos * 100:.0f}% positive")
    print("\n== r1 by quintile of run21 (mean %, % rose, n)")
    for i, x in enumerate(quintiles(full, "run21", "r1")):
        print("  ", i + 1, f"{x[0]:+.3f}% {x[1]:.1f}% {x[2]}")
    print("\n== r1 by quintile of mean8 (habit)")
    for i, x in enumerate(quintiles(full, "mean8", "r1")):
        print("  ", i + 1, f"{x[0]:+.3f}% {x[1]:.1f}% {x[2]}")
    # the reaction and what follows it
    rs = [r for r in full if r.get("post5") is not None and r.get("post21") is not None]
    for r in rs:
        r["z1"] = r["r1"] / r["med8"] if r["med8"] > 0 else 0.0
        r["against"] = -np.sign(r["r1"]) * np.sign(r["run21"]) * min(abs(r["run21"]), 3)
    for key in ("post5", "post21"):
        ic, t, n, pos = monthly_ic(rs, "z1", key)
        print(f"\n== {key} vs the reaction z1: IC {ic:+.4f} t {t:+.2f} ({n} months, {pos * 100:.0f}% +)")
    print("\n== by why (count, r1 mean, post5 mean, post21 mean, post5 % same sign as r1)")
    kinds = {}
    for r in rs:
        kinds.setdefault(r.get("why"), []).append(r)
    for k, v in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        a = np.array([[x["r1"], x["post5"], x["post21"]] for x in v])
        same = np.mean(np.sign(a[:, 0]) == np.sign(a[:, 1]))
        print(f"  {str(k):14s} {len(v):6d} r1 {a[:, 0].mean() * 100:+.2f}%  post5 {a[:, 1].mean() * 100:+.3f}%  "
              f"post21 {a[:, 2].mean() * 100:+.3f}%  continued {same * 100:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
