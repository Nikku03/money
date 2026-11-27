"""A trial, not a test: does the engine guess a company's revenue better than the analysts do? Quarters ending by
2023-12-31 only.

    PYTHONPATH=. python research/promises/trial_analysts_train.py > research/promises/trial_analysts_train.log

The analysts' quarterly revenue estimates come from the Alpha Vantage connector (EARNINGS_ESTIMATES; a free key, 25
requests a day), fetched on 2026-11-27 for eleven companies with long promise records and kept outside git in
``var/consensus`` (TICKER.json: fiscal quarter end -> estimate; full/TICKER.json: -> {rev, rev_n, eps, eps_n}).
The vendor's figure for a past quarter is its last estimate before the report; when it was frozen is not stated.
The engine's guess (the promise's midpoint moved by the habit) was fixed when the promise was made, about three
months earlier, so the comparison leans toward the analysts.

Printed per company and together: the median miss of the midpoint, of the midpoint moved by the habit and of the
analysts; where the analysts sit against the midpoint; how often the revenue beat the analysts; and whether the
report session followed the beat against the analysts, the delivery against the habit, or the beat the habit
predicted beforehand. Eleven companies is a trial: nothing here is declared.
"""
import json
from datetime import date

import numpy as np

from brain import config, episodes, promises

END = "2023-12-31"
here = config.home() / "consensus"


def rank(x):
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(rank(a), rank(b))[0, 1]) if len(a) > 3 else float("nan")


est = {}
for p in sorted(here.glob("*.json")):
    est[p.stem] = {date.fromisoformat(k): float(v) for k, v in json.loads(p.read_text()).items() if v}
for p in sorted((here / "full").glob("*.json")):
    est[p.stem] = {date.fromisoformat(k): float(v["rev"]) for k, v in json.loads(p.read_text()).items()
                   if v and v.get("rev")}
by, ep = {}, {}
for r in promises.load():
    by.setdefault(r["t"], []).append(r)
for e in episodes.load():
    if e["t"] in est:
        ep.setdefault(e["t"], []).append(e)
out = []
for t, es in sorted(est.items()):
    rows, last = by.get(t, []), {}
    for r in rows:
        if (r["horizon"] == "quarter" and r.get("grade") and not r.get("mismatch") and r.get("period_end")
                and r["period_end"] <= END):
            if r["period_end"] not in last or r["known_at"] > last[r["period_end"]]["known_at"]:
                last[r["period_end"]] = r
    for pe, r in sorted(last.items()):
        d = date.fromisoformat(pe)
        near = sorted((abs((k - d).days), k) for k in es if abs((k - d).days) <= 10)
        if not near:
            continue
        after = [e for e in ep.get(t, []) if pe < e["date"] and (date.fromisoformat(e["date"]) - d).days <= 100]
        out.append({"t": t, "pe": pe, "mid": r["mid"], "habit": promises.habit(rows, r["made"]),
                    "cons": es[near[0][1]], "actual": r["actual"],
                    "r1": min(after, key=lambda e: e["date"])["r1"] if after else None})


def report(w, name):
    if len(w) < 4:
        print(f"{name}: {len(w)} quarters with a habit, too few")
        return
    em = np.array([o["actual"] / o["mid"] - 1 for o in w])
    eh = np.array([o["actual"] / (o["mid"] * (1 + o["habit"])) - 1 for o in w])
    ec = np.array([o["actual"] / o["cons"] - 1 for o in w])
    gap = np.array([o["mid"] * (1 + o["habit"]) / o["cons"] - 1 for o in w])
    cm = np.array([o["cons"] / o["mid"] - 1 for o in w])
    print(f"{name}: {len(w)} quarters | median miss: midpoint {np.median(np.abs(em)):.2%}, habit-moved "
          f"{np.median(np.abs(eh)):.2%}, analysts {np.median(np.abs(ec)):.2%} | analysts over the midpoint "
          f"{np.median(cm):+.2%}, the habit {np.median([o['habit'] for o in w]):+.2%} | beat the analysts "
          f"{np.mean(ec > 0):.0%} | habit-moved closer than the analysts {np.mean(np.abs(eh) < np.abs(ec)):.0%} | "
          f"predicted beat against actual beat {rho(gap, ec):+.2f}")


w = [o for o in out if o["habit"] is not None and abs(o["cons"] / o["mid"] - 1) < 0.3]
print(f"companies {len({o['t'] for o in out})}, quarters {len(out)}, with a habit {len(w)}")
for t in sorted({o["t"] for o in w}):
    report([o for o in w if o["t"] == t], t)
report(w, "ALL")
r = [o for o in w if o["r1"] is not None]
r1 = np.array([o["r1"] for o in r])
ec = np.array([o["actual"] / o["cons"] - 1 for o in r])
eh = np.array([o["actual"] / (o["mid"] * (1 + o["habit"])) - 1 for o in r])
gap = np.array([o["mid"] * (1 + o["habit"]) / o["cons"] - 1 for o in r])
print(f"the report session ({len(r)} quarters) moved with the beat against the analysts {rho(ec, r1):+.2f}; with the "
      f"delivery against the habit {rho(eh, r1):+.2f}; with the beat predicted beforehand {rho(gap, r1):+.2f}")
hi = gap > np.median(gap)
print(f"  predicted beat above its median: the session {np.mean(r1[hi]):+.2%} ({np.mean(r1[hi] > 0):.0%} rose); below: "
      f"{np.mean(r1[~hi]):+.2%} ({np.mean(r1[~hi] > 0):.0%} rose)")
