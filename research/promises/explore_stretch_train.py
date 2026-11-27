"""TRAIN-only closer look at the one lead of ``explore_break_train.py``: the stretch of a promise against the report
session that delivers it. Report sessions of 2011-01-01..2023-12-31 only.

    PYTHONPATH=. python research/promises/explore_stretch_train.py > research/promises/explore_stretch_train.log

By half, by universe, by its two parts (the growth asked for; the growth just done), by thirds inside each season,
by year, with the run into the report held apart, and against the run itself. Declared afterwards as G3a
(``brain/research/promisetest3.py``); its last line is G3a's own statistic on these years and the row counts, no
outcomes, of 2024-2026.
"""
import importlib.util
from collections import Counter

import numpy as np

from brain import config, episodes, promises
from brain.research import promisetest as pt
from brain.research import promisetest3 as p3

spec = importlib.util.spec_from_file_location("xb", config.ROOT / "research/promises/explore_break_train.py")
xb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xb)
lib, eps = promises.load(), episodes.load()
sets = {n: xb.tickers(n) for n in xb.UNIVERSES}
every = set().union(*sets.values())
rows = xb.rows_of(lib, eps, every, {})
asks = {(r["t"], r["made"], r.get("period_end")): (r.get("asks"), r.get("did")) for r in lib if r["horizon"] == "quarter"}
for r in rows:
    a = asks.get((r["t"], r["made"], r["pe"]), (None, None))
    r["asks"], r["did"] = a
    r["neg_did"] = -a[1] if a[1] is not None else None
tr = [r for r in rows if pt.TRAIN[0] <= r["date"] <= pt.TRAIN[1] and r["r1"] is not None]
print(len(tr), "deliveries in TRAIN with a session;", sum(r["stretch"] is not None for r in tr), "with a stretch")


def ic(q, k, y="r1"):
    g = np.array([r["season"] for r in q])
    _, d = pt.season_ic(g, xb.col(q, k), xb.col(q, y), pt.MIN_N)
    if len(d) < 3:
        return "too few"
    return f"{d.mean():+.3f} t {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.1f} ({int((d > 0).sum())} of {len(d)})"


for nm, q in (("all", tr), ("2011-2018", [r for r in tr if r["date"] < "2019"]),
              ("2019-2023", [r for r in tr if r["date"] >= "2019"]),
              *((u, [r for r in tr if r["t"] in sets[u]]) for u in xb.UNIVERSES),
              ("not the research universe", [r for r in tr if r["t"] not in sets["us_broad"]])):
    print(f"{nm:<28} stretch -> session {ic(q, 'stretch')} | asks -> {ic(q, 'asks')} | minus did -> "
          f"{ic(q, 'neg_did')} | rows {sum(r['stretch'] is not None for r in q)}")
w = [r for r in tr if r["stretch"] is not None]
s, r1 = xb.col(w, "stretch"), xb.col(w, "r1")
third, g = np.zeros(len(w), int), np.array([r["season"] for r in w])
for se in set(g.tolist()):
    i = np.where(g == se)[0]
    if len(i) < 9:
        third[i] = -1
        continue
    c = np.quantile(s[i], [1 / 3, 2 / 3])
    third[i] = np.where(s[i] <= c[0], 0, np.where(s[i] >= c[1], 2, 1))
for k, nm in enumerate(("lowest third (asks for less than it just did)", "middle third", "highest third (asks for more)")):
    m = third == k
    print(f"  {nm:<48} {int(m.sum())} deliveries: stretch median {np.median(s[m]) * 100:+.1f} points, the session "
          f"{r1[m].mean() * 100:+.2f}% (median {np.median(r1[m]) * 100:+.2f}%, {np.mean(r1[m] > 0) * 100:.0f}% rose)")
yr = np.array([r["date"][:4] for r in w])
print("  highest less lowest third, by year:",
      " ".join(f"{y}:{(r1[(third == 2) & (yr == y)].mean() - r1[(third == 0) & (yr == y)].mean()) * 100:+.1f}"
               for y in sorted(set(yr.tolist()))))
run = xb.col(w, "run21")
for lo, hi, nm in ((-9, -0.5, "ran down"), (-0.5, 0.5, "flat"), (0.5, 9, "ran up")):
    q = [w[i] for i in np.where((run > lo) & (run <= hi))[0]]
    print(f"  run of 21 sessions {nm:<9} {len(q)} rows: stretch -> session {ic(q, 'stretch')}")
print("  stretch against the run of 63 sessions into the report:", ic(w, "stretch", "run63"), "| 21:",
      ic(w, "stretch", "run21"))
print("  stretch against delivery beyond the habit:", ic(w, "stretch", "beyond"), "| against the raw surprise:",
      ic(w, "stretch", "surprise"))
print("  stretch 5%, 25%, 50%, 75%, 95%:", np.round(np.quantile(s, [.05, .25, .5, .75, .95]) * 100, 1))
m = p3.evaluate(lib, eps, p3.universe(), span=p3.TRAIN)["G3a"]
print("\nG3a's own statistic on TRAIN:", p3.words("G3a", m), f"p {m['p']:.4f}")
test = [r for r in p3.deliveries(lib, eps, p3.universe()) if p3.TEST[0] <= r["date"] <= p3.TEST[1] and r["stretch"] is not None]
c = Counter(r["season"] for r in test)
print(f"2024-01..2026-09, counts only: {len(test)} rows of {len({r['t'] for r in test})} companies;",
      " ".join(f"{k}:{v}" for k, v in sorted(c.items())))
