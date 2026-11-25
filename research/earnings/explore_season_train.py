"""TRAIN years only (2011-2023): does the season so far reach the next report? For each episode, the mean reaction
(in their own usual report moves) of the same-industry companies (SEC code, three digits, else two) whose report
sessions fell in the 30 days before its close-before-the-report; then its rank correlation with the episode's own
reaction, alone and beside the expectation score. TEST years are not read.
Usage: PYTHONPATH=. python research/earnings/explore_season_train.py"""
from __future__ import annotations

import bisect
from datetime import date, timedelta

import numpy as np

from brain import episodes as ep
from brain import why
from brain.bank import Bank
from brain.research import earntest as et

TRAIN_END = "2023-12-31"
WINDOW = 30


def season(rows: list[dict], sic: dict[str, str], digits: int, window: int = WINDOW, min_peers: int = 2) -> np.ndarray:
    groups: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        code = sic.get(r["t"])
        if code and r.get("med8") and r["med8"] > 0:
            groups.setdefault(code[:digits], []).append((r["date"], r["r1"] / r["med8"], r["t"]))
    for g in groups.values():
        g.sort()
    out = np.full(len(rows), np.nan)
    for k, r in enumerate(rows):
        code = sic.get(r["t"])
        if not code:
            continue
        g = groups.get(code[:digits], [])
        lo = (date.fromisoformat(r["pre"]) - timedelta(days=window)).isoformat()
        a = bisect.bisect_left(g, (lo,))
        b = bisect.bisect_right(g, (r["pre"], float("inf")))
        vals = [z for d, z, t in g[a:b] if t != r["t"] and d <= r["pre"]]
        if len(vals) >= min_peers:
            out[k] = float(np.mean(np.clip(vals, -4, 4)))
    return out


def main():
    import sys
    from pathlib import Path
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = [r for r in ep.complete(ep.load(src), "r1") if r["date"] <= TRAIN_END]
    view = Bank().view()
    sic = {t: i["sic"] for t, i in why.industry_map(view, sorted({r["t"] for r in rows})).items()}
    print(f"TRAIN episodes {len(rows)}; with an industry code {sum(r['t'] in sic for r in rows)}")
    month = np.array([r["date"][:7] for r in rows])
    y = np.array([r["r1"] for r in rows])
    sc = et.score_scale(rows)
    score = np.array([et.expectation_score(r, sc) for r in rows])
    for digits in (3, 2):
        s = season(rows, sic, digits)
        ok = np.isfinite(s)
        print(f"\n{digits}-digit industries: {ok.sum()} episodes with 2+ peers reported in the 30 days before")
        for lo, hi in (("2011", "2018"), ("2019", "2023")):
            k = ok & (month >= lo) & (month <= hi + "-12")
            ms, d = et.monthly_ic(month[k], s[k], y[k])
            print(f"  {lo}-{hi} season vs r1: IC {d.mean():+.4f} t {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f} "
                  f"({len(d)} months, {np.mean(d > 0) * 100:.0f}% +)")
            z = lambda x: (x - np.nanmedian(x)) / (np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826)
            comb = z(score[k]) + z(s[k])
            ms, d = et.monthly_ic(month[k], comb, y[k])
            ms2, d2 = et.monthly_ic(month[k], score[k], y[k])
            print(f"  {lo}-{hi} score alone IC {d2.mean():+.4f}; score + season IC {d.mean():+.4f} "
                  f"t {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f}")


if __name__ == "__main__":
    main()
