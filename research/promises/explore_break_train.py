"""TRAIN-only exploration: what warns, before a report, that a company will break its habit?

The report session follows the delivery against the habit (G1b; +0.30 the same day on these years), so anything
that says beforehand whether a company will deliver more or less than its habit is a lead on the session itself.
Deliveries whose report session fell in 2011-01-01..2023-12-31 only; no later outcome is read (section D counts).

    PYTHONPATH=. python research/promises/explore_break_train.py > research/promises/explore_break_train.log

A row is one quarterly promise with its delivery: the promise's midpoint, the habit when it was made, the revenue
first reported, and the report session that delivered it (the first report session after the quarter's end).
``beyond`` is the surprise on the promise less the habit. Each line is the rank correlation of something known at
the close before that session with (a) ``beyond`` and (b) the session's move against the market, inside each
season (the calendar quarter of the session, at least MIN_N rows), averaged over the seasons, with its t.

Known before the session:
  the season so far     the mean ``beyond`` of the other companies whose report session came earlier in the season
                        (all of them; those of the same 2-digit SIC industry)
  the last season       the mean ``beyond`` of every delivery of the season before
  its last break        the delivery it made in the release that gave this promise, less the habit then
  the stretch           the growth the promise asks for less the growth just done (G2b)
  the range's width     (high - low) / midpoint; a point promise is 0
  the year's promise    the change of its promise for the year in the same release
  the run               the stock against the market over the 5, 21 and 63 sessions to the close before, in its
                        own usual moves (the episode library)
  after the last report the stock's 21 sessions after the release that gave the promise
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date

import numpy as np

from brain import config, episodes, promises
from brain.bank import Bank
from brain.research import customertest as ct
from brain.research import promisetest as pt

TRAIN, TEST, MIN_N = pt.TRAIN, pt.TEST, pt.MIN_N
UNIVERSES = ("us_broad", "us_small", "us_micro", "us_watch")
AFTER_END = (5, 100)        # the delivering report session comes this many days after the quarter's end
SEASON_MIN, IND_MIN = 10, 3


def tickers(name: str) -> set[str]:
    return {t.upper() for t in json.loads((config.ROOT / "universe" / f"{name}.json").read_text())["tickers"]}


def rows_of(lib: list[dict], eps: list[dict], names: set[str], sic: dict[str, str]) -> list[dict]:
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in lib:
        if r["t"] in names:
            by_t[r["t"]].append(r)
    ep_t: dict[str, list[dict]] = defaultdict(list)
    for e in sorted(eps, key=lambda e: e["date"]):
        if e["t"] in names:
            ep_t[e["t"]].append(e)
    year_change = {(r["t"], r["made"]): r["change"] for r in lib
                   if r["horizon"] == "year" and r.get("change") is not None}
    out = []
    for t in sorted(by_t):
        rs = sorted(by_t[t], key=lambda r: r["known_at"])
        last: dict[str, dict] = {}
        for r in rs:
            if r["horizon"] == "quarter" and r.get("period_end") and not r.get("mismatch"):
                last[r["period_end"]] = r                    # a quarter promised twice counts by its last promise
        prev = None
        for pe in sorted(last):
            r = last[pe]
            d0 = date.fromisoformat(pe)
            ep = next((e for e in ep_t.get(t, [])
                       if AFTER_END[0] <= (date.fromisoformat(e["date"]) - d0).days <= AFTER_END[1]), None)
            made_ep = pt._first_after(ep_t.get(t, []), r["made"], pt.HERE_DAYS)
            habit = promises.habit(rs, r["made"])
            got_vs = None
            if prev is not None and prev.get("surprise") is not None and prev["period_end"] < r["made"]:
                h0 = promises.habit(rs, prev["made"])
                got_vs = prev["surprise"] - h0 if h0 is not None else None
            if ep is not None:
                out.append({
                    "t": t, "made": r["made"], "pe": pe, "date": ep["date"], "season": pt.season(ep["date"]),
                    "sic2": sic.get(t, "")[:2], "habit": habit, "surprise": r.get("surprise"),
                    "beyond": (r["surprise"] - habit) if r.get("surprise") is not None and habit is not None else None,
                    "r1": ep.get("r1"), "run5": ep.get("run5"), "run21": ep.get("run21"), "run63": ep.get("run63"),
                    "last_break": got_vs, "stretch": r.get("stretch"),
                    "width": (r["high"] - r["low"]) / r["mid"] if r["mid"] else None,
                    "year": year_change.get((t, r["made"])),
                    "post21": made_ep.get("post21") if made_ep else None})
            prev = r
    return out


def add_season(rows: list[dict]) -> None:
    """The season so far and the last season, from deliveries whose report session came strictly earlier."""
    by_s: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["beyond"] is not None:
            by_s[r["season"]].append(r)
    seasons = sorted(by_s)
    mean = {s: float(np.mean([r["beyond"] for r in by_s[s]])) for s in seasons}
    before = {s: (mean[seasons[i - 1]] if i else None) for i, s in enumerate(seasons)}
    for s in seasons:
        by_s[s].sort(key=lambda r: r["date"])
    for r in rows:
        r["last_season"] = before.get(r["season"])
        early = [x for x in by_s.get(r["season"], []) if x["date"] < r["date"] and x["t"] != r["t"]]
        r["so_far"] = float(np.mean([x["beyond"] for x in early])) if len(early) >= SEASON_MIN else None
        ind = [x["beyond"] for x in early if x["sic2"] and x["sic2"] == r["sic2"]]
        r["so_far_ind"] = float(np.mean(ind)) if len(ind) >= IND_MIN else None
        r["so_far_ind_less"] = (r["so_far_ind"] - r["so_far"]) if r["so_far_ind"] is not None and r["so_far"] is not None else None


def col(rows, k, sign=1.0):
    return np.array([sign * r[k] if r.get(k) is not None else np.nan for r in rows], float)


def ic(group, pred, y):
    _, d = pt.season_ic(np.array(group), pred, y, MIN_N)
    if len(d) < 3:
        return float("nan"), float("nan"), len(d), 0
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d), int((d > 0).sum())


def line(name, rows, k):
    g = [r["season"] for r in rows]
    p = col(rows, k)
    a, b = ic(g, p, col(rows, "beyond")), ic(g, p, col(rows, "r1"))
    n = int((np.isfinite(p) & np.isfinite(col(rows, "beyond"))).sum())
    print(f"  {name:<44} delivery beyond its habit {a[0]:+.3f} t {a[1]:+.1f} ({a[3]} of {a[2]}) | "
          f"the session {b[0]:+.3f} t {b[1]:+.1f} ({b[3]} of {b[2]}) | {n} rows")


FEATS = (("the season so far, every company", "so_far"), ("the season so far, its industry", "so_far_ind"),
         ("its industry less every company", "so_far_ind_less"), ("the last season", "last_season"),
         ("its last break", "last_break"), ("the stretch", "stretch"), ("the range's width", "width"),
         ("the change in the year's promise", "year"), ("the run, 5 sessions", "run5"),
         ("the run, 21 sessions", "run21"), ("the run, 63 sessions", "run63"),
         ("the 21 sessions after the last report", "post21"))


def report(label: str, rows: list[dict]) -> None:
    print(f"\n===== {label} =====")
    for name, lo, hi in (("TRAIN 2011-2023", "2011", "2023"), ("  2011-2018", "2011", "2018"),
                         ("  2019-2023", "2019", "2023")):
        q = [r for r in rows if TRAIN[0] <= r["date"] <= TRAIN[1] and lo <= r["date"][:4] <= hi]
        w = [r for r in q if r["beyond"] is not None]
        print(f"\n{name}: {len(q)} deliveries of {len({r['t'] for r in q})} companies, {len(w)} with a habit")
        if len(w) < 100:
            continue
        b, r1 = col(w, "beyond"), col(w, "r1")
        k = np.isfinite(r1)
        print(f"  beyond its habit: median {np.median(b) * 100:+.2f}%, 1 in 6 below {np.quantile(b, 1 / 6) * 100:+.2f}%, "
              f"1 in 6 above {np.quantile(b, 5 / 6) * 100:+.2f}%; the same day against the session "
              f"{ic([r['season'] for r in w], b, r1)[0]:+.3f}")
        for nm, key in FEATS:
            line(nm, q, key)
        if name.startswith("TRAIN"):
            for nm, key in (("the season so far, every company", "so_far"), ("the season so far, its industry", "so_far_ind")):
                p = col(w, key)
                ok = np.isfinite(p) & k
                if ok.sum() < 200:
                    continue
                cut = np.quantile(p[ok], [1 / 3, 2 / 3])
                for part, m in (("lowest third", ok & (p <= cut[0])), ("middle third", ok & (p > cut[0]) & (p < cut[1])),
                                ("highest third", ok & (p >= cut[1]))):
                    print(f"    {nm}, {part}: {int(m.sum())} deliveries, beyond the habit {np.median(b[m]) * 100:+.2f}% "
                          f"({np.mean(b[m] > 0) * 100:.0f}% above), the session {np.mean(r1[m]) * 100:+.2f}% "
                          f"({np.mean(r1[m] > 0) * 100:.0f}% rose)")
    q = [r for r in rows if TEST[0] <= r["date"] <= TEST[1]]
    per = defaultdict(int)
    for r in q:
        if r["habit"] is not None and r.get("so_far_n_ok", True):
            per[r["season"]] += 1
    print(f"\n D. {TEST[0]}..{TEST[1]}, counts only: {len(q)} deliveries of {len({r['t'] for r in q})} companies; "
          f"with a habit {sum(per.values())}, in {sum(n >= MIN_N for n in per.values())} seasons with {MIN_N}+")


def main(argv: list[str]) -> int:
    lib, eps = promises.load(), episodes.load()
    sets = {n: tickers(n) for n in UNIVERSES}
    every = set().union(*sets.values())
    sic = ct.sic_of(Bank(), sorted(every))
    rows = rows_of(lib, eps, every, sic)
    add_season(rows)
    print(f"{len(rows)} quarterly promises with a delivering report session, {len({r['t'] for r in rows})} companies; "
          f"an industry code for {sum(1 for r in rows if r['sic2'])}")
    report("the four universes together", rows)
    broad = [r for r in rows if r["t"] in sets["us_broad"]]
    add_season(broad)
    report("the research universe alone (its own season so far)", broad)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
