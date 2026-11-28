"""TRAIN-only exploration (promises made 2011-2023, the four universes): a better guess of the delivery than the
plain habit, and ranges that fit each company.

    PYTHONPATH=. python research/promises/explore_guess_train.py > research/promises/explore_guess_train.log

Asked on 2026-11-28 ("how can we make it better?"), after the forward record against the analysts was set up and
Micron's guess stood 16% over its own promise. Fitted on promises made 2011-2018 and checked on 2019-2023; nothing
of 2024 or later is read.

1. The size of the habit against what was then delivered: do large habits overshoot?
2. A pulled-in habit: delivery = C + K x habit, C and K by least absolute miss on 2011-2018.
3. Ranges by the company's own spread (the standard deviation of its last four surprises): the fixed ranges
   against ranges HALF = A + B x spread, each side fitted by pinball loss on 2011-2018.
"""
import json

import numpy as np

from brain import config, promises

names = set()
for u in ("us_broad", "us_small", "us_micro", "us_watch"):
    names |= {t.upper() for t in json.loads((config.ROOT / "universe" / f"{u}.json").read_text())["tickers"]}
by = {}
for r in promises.load():
    if r["t"] in names:
        by.setdefault(r["t"], []).append(r)
rows = []
for t, rs in by.items():
    rs = sorted(rs, key=lambda r: r["known_at"])
    last = {}
    for r in rs:
        if r["horizon"] == "quarter" and r.get("grade") and r.get("period_end") and not r.get("mismatch"):
            last[r["period_end"]] = r
    for pe, r in sorted(last.items()):
        if not ("2011-01-01" <= r["made"] <= "2023-12-31"):
            continue
        h = promises.habit(rs, r["made"])
        if h is None:
            continue
        prev = [x["surprise"] for k, x in sorted(last.items()) if k < r["made"]][-4:]
        rows.append({"t": t, "made": r["made"], "h": h, "s": r["surprise"], "sd": float(np.std(prev))})
h = np.array([r["h"] for r in rows])
s = np.array([r["s"] for r in rows])
sd = np.array([r["sd"] for r in rows])
yr = np.array([r["made"][:4] for r in rows])
fit, new = yr <= "2018", yr > "2018"
print(f"{len(rows)} graded quarterly promises with a habit, of {len({r['t'] for r in rows})} companies; "
      f"{int(fit.sum())} made 2011-2018, {int(new.sum())} made 2019-2023")

print("\n1. By the size of the habit")
for lo, hi in ((-1, 0), (0, .01), (.01, .02), (.02, .04), (.04, .08), (.08, 1)):
    m = (h > lo) & (h <= hi)
    print(f"  habit {lo * 100:+.0f}%..{hi * 100:+.0f}%: {int(m.sum()):>5} promises; median habit {np.median(h[m]) * 100:+.2f}%, "
          f"median delivery {np.median(s[m]) * 100:+.2f}%; delivered less than the habit {np.mean(s[m] < h[m]) * 100:.0f}%; "
          f"median miss of the habit's guess {np.median(np.abs(s[m] - h[m])) * 100:.2f}%")


def miss(pred, m):
    return float(np.median(np.abs(s[m] - pred[m])) * 100)


def mean_miss(pred, m):
    return float(np.mean(np.abs(s[m] - pred[m])) * 100)


print("\n2. A pulled-in habit")
best = min((mean_miss(c + k * h, fit), k, c) for k in np.arange(0.2, 1.21, 0.05) for c in np.arange(-0.01, 0.0201, 0.001))
K, C = round(float(best[1]), 2), round(float(best[2]), 3)
bm = min((miss(c + k * h, fit), k, c) for k in np.arange(0.2, 1.21, 0.05) for c in np.arange(-0.01, 0.0201, 0.001))
print(f"  least mean miss on 2011-2018: delivery = {C * 100:+.1f}% + {K:.2f} x habit; least median miss: "
      f"{bm[2] * 100:+.1f}% + {bm[1]:.2f} x habit")
for C_, K_, nm in ((0.010, 0.50, "+1.0% + 0.50 x habit"), (C, K, "the least-mean-miss line")):
    g = C_ + K_ * h
    for label, m in (("2011-2018", fit), ("2019-2023", new)):
        closer = np.mean(np.abs(s[m] - g[m]) < np.abs(s[m] - h[m])) * 100
        big = m & (h > .08)
        print(f"  {nm}, {label}: median miss {miss(g, m):.2f}% against the plain habit {miss(h, m):.2f}% and the midpoint "
              f"{miss(np.zeros_like(h), m):.2f}%; mean miss {mean_miss(g, m):.2f}% against {mean_miss(h, m):.2f}%; closer "
              f"than the plain habit in {closer:.0f}%; habits over 8% ({int(big.sum())}): {miss(g, big):.2f}% against "
              f"{miss(h, big):.2f}%")
by_year = []
g = 0.010 + 0.50 * h
for y in sorted(set(yr.tolist())):
    m = yr == y
    if m.sum() >= 40:
        by_year.append(f"{y}:{mean_miss(h, m) - mean_miss(g, m):+.2f}")
print("  +1.0% + 0.50 x habit, mean miss saved by year (points):", " ".join(by_year))

print("\n3. Ranges by the company's own spread (around the pulled-in guess, +1.0% + 0.50 x habit)")
res = s - g


def pin(q, pred, y):
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


grid_a, grid_b = np.arange(0.0, 0.0601, 0.0025), np.arange(0.0, 3.01, 0.1)
band = {}
for q in (0.05, 1 / 6, 5 / 6, 0.95):
    sign = -1 if q < 0.5 else 1
    band[q] = min((pin(q, sign * (a + b * sd[fit]), res[fit]), a, b) for a in grid_a for b in grid_b)[1:]
    print(f"  quantile {q:.3f}: {'-' if sign < 0 else '+'}({band[q][0] * 100:.2f}% + {band[q][1]:.1f} x spread)")
for label, m in (("2011-2018", fit), ("2019-2023", new)):
    lo68, hi68 = -(band[1 / 6][0] + band[1 / 6][1] * sd), band[5 / 6][0] + band[5 / 6][1] * sd
    lo90, hi90 = -(band[.05][0] + band[.05][1] * sd), band[.95][0] + band[.95][1] * sd
    print(f"  {label}: scaled ranges held {np.mean((res[m] >= lo68[m]) & (res[m] <= hi68[m])) * 100:.1f}% and "
          f"{np.mean((res[m] >= lo90[m]) & (res[m] <= hi90[m])) * 100:.1f}%; the fixed ranges around the plain habit "
          f"{np.mean((s[m] - h[m] >= -.035) & (s[m] - h[m] <= .03)) * 100:.1f}% and {np.mean(np.abs(s[m] - h[m]) <= .081) * 100:.1f}%")
    for a, b in ((0, .01), (.01, .02), (.02, .05), (.05, 1)):
        k = m & (sd > a) & (sd <= b)
        print(f"    own spread {a * 100:.0f}%..{b * 100:.0f}% ({int(k.sum())}): scaled {np.mean((res[k] >= lo68[k]) & (res[k] <= hi68[k])) * 100:.0f}% "
              f"and {np.mean((res[k] >= lo90[k]) & (res[k] <= hi90[k])) * 100:.0f}%; fixed "
              f"{np.mean((s[k] - h[k] >= -.035) & (s[k] - h[k] <= .03)) * 100:.0f}% and {np.mean(np.abs(s[k] - h[k]) <= .081) * 100:.0f}%; "
              f"median width of the scaled 2-in-3 range {np.median(hi68[k] - lo68[k]) * 100:.1f} points (fixed 6.5)")
