"""TRAIN-only exploration of customer links (brain/customers.py): customers' reports of 2011-01-01..2023-12-31.

Run before any declared test, and never on a later report:
    PYTHONPATH=. python research/customers/explore_train.py > research/customers/explore_train.log

A link is point-in-time: supplier S named listed customer C in an annual report already filed, and S has not filed
its next one (brain.customers.links). For each results report of C while a link holds, C's report-session move
against the market (``r1`` of the episode library) is set against what S did:
  same   S's move against the market over the same session (an explanation, not a forecast)
  after  S's move against the market over the next 5 and the next 21 sessions
  own    S's own report-session move, when S reports 1 to 45 days after C
Events where S itself reports from the session before C's to five sessions after are left out of same and after.
Each line is the rank correlation inside a season (the calendar quarter of C's report, at least MIN_N events),
averaged over the seasons, with its t.
Controls: "peers" are stocks of S's industry (the same first three digits of its SIC code) that do not name C, up
to PEERS a link, read the same way. What a link adds is the linked line less the peers line.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date

import numpy as np

from brain import config, customers, episodes
from brain.bank import Bank
from brain.metrics import PriceBook
from brain.research import promisetest as pt

TRAIN = ("2011-01-01", "2023-12-31")
MIN_N = 20
PEERS = 3
OWN_DAYS = 45
BENCH = "SPY"


def season(d: str) -> str:
    return f"{d[:4]}Q{(int(d[5:7]) - 1) // 3 + 1}"


def ic(group, pred, y):
    _, d = pt.season_ic(np.array(group), np.array(pred, float), np.array(y, float), MIN_N)
    if len(d) < 3:
        return float("nan"), float("nan"), len(d), 0
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d), int((d > 0).sum())


def line(name, rows, key):
    rows = [r for r in rows if r.get(key) is not None]
    m, t, n, pos = ic([r["season"] for r in rows], [r["c_r1"] for r in rows], [r[key] for r in rows])
    print(f"  {name:<58} {m:+.3f}  t {t:+.1f}  ({pos} of {n} seasons positive, {len(rows)} events)")


class Prices:
    """Each stock's adjusted closes with the market's on the same days."""

    def __init__(self, bank: Bank, tickers):
        self.view = bank.view()
        self.book = PriceBook(bank, sorted(set(tickers) | {BENCH}))
        self.bd, self.ba = self.book.adj_upto(self.view, BENCH)
        self.cache: dict[str, tuple | None] = {}

    def get(self, t: str):
        if t not in self.cache:
            if not self.book.has(t):
                self.cache[t] = None
            else:
                d, a = self.book.adj_upto(self.view, t)
                self.cache[t] = (d.astype(str), a, episodes.aligned(d, self.bd, self.ba))
        return self.cache[t]

    def move(self, t: str, start: str, sessions: int | None = None, end: str | None = None) -> float | None:
        """t's move against the market from the close of ``start`` to the close ``sessions`` later (or of ``end``)."""
        got = self.get(t)
        if got is None:
            return None
        d, a, b = got
        i = int(np.searchsorted(d, start))
        if i >= len(d) or d[i] != start:
            return None
        j = i + sessions if sessions is not None else int(np.searchsorted(d, end))
        if j >= len(d) or (end is not None and d[j] != end):
            return None
        return episodes._excess(a, b, i, j)


def events(links_, eps_by, px: Prices, pairs) -> list[dict]:
    """One row a (link, report of C): what S did. ``pairs`` yields (S, C, share, known_at, until)."""
    out = []
    for s, c, share, known, until in pairs:
        own = [e["date"] for e in eps_by.get(s, [])]
        for e in eps_by.get(c, []):
            if not (TRAIN[0] <= e["date"] <= TRAIN[1]) or not (known[:10] < e["pre"] and e["date"] <= until[:10]):
                continue
            row = {"s": s, "c": c, "share": share, "date": e["date"], "season": season(e["date"]), "c_r1": e["r1"]}
            d0, d1 = date.fromisoformat(e["pre"]), date.fromisoformat(e["date"])
            near = any(-4 <= (date.fromisoformat(x) - d1).days <= 8 for x in own)
            if not near:
                row["same"] = px.move(s, e["pre"], end=e["date"])
                row["after5"] = px.move(s, e["date"], sessions=5)
                row["after21"] = px.move(s, e["date"], sessions=21)
            nxt = [x for x in eps_by.get(s, []) if 1 <= (date.fromisoformat(x["date"]) - d1).days <= OWN_DAYS]
            row["own"] = nxt[0]["r1"] if nxt else None
            out.append(row)
    return out


def main() -> int:
    bank = Bank()
    hist = customers.load(customers.history_path())
    L = customers.links(hist)
    names = {t.upper() for t in json.loads((config.ROOT / "universe" / "us_broad.json").read_text())["tickers"]}
    eps = [e for e in episodes.load() if e.get("r1") is not None]
    eps_by: dict[str, list[dict]] = defaultdict(list)
    for e in sorted(eps, key=lambda e: e["date"]):
        eps_by[e["t"]].append(e)
    view = bank.view()
    sic = {}
    for t in names:
        f = view.latest(t, "identity.sic")
        if f is not None:
            sic[t] = str(f.value)[:3]
    by_sic: dict[str, list[str]] = defaultdict(list)
    for t, k in sorted(sic.items()):
        by_sic[k].append(t)
    linked = {(x["s"], x["c"]) for x in L}
    real = [(x["s"], x["c"], x["share"], x["known_at"], x["until"]) for x in L]
    peers = []
    for s, c, share, known, until in real:
        pool = [p for p in by_sic.get(sic.get(s, ""), []) if p != s and p != c and (p, c) not in linked]
        peers += [(p, c, share, known, until) for p in pool[:PEERS]]
    px = Prices(bank, {x[0] for x in real} | {x[0] for x in peers})
    years = sorted({x["known_at"][:4] for x in L})
    print(f"{len(hist)} annual reports of {len({r['t'] for r in hist})} companies; {len(L)} point-in-time links of "
          f"{len({x['s'] for x in L})} suppliers to {len({x['c'] for x in L})} listed customers, filed "
          f"{years[0] if years else '-'}..{years[-1] if years else '-'}; {sum(x['share'] is not None for x in L)} "
          f"with a share")
    ev, pv = events(L, eps_by, px, real), events(L, eps_by, px, peers)
    for name, lo, hi in (("TRAIN 2011-2023", "2011", "2023"), ("  2011-2018", "2011", "2018"),
                         ("  2019-2023", "2019", "2023")):
        e = [r for r in ev if lo <= r["date"][:4] <= hi]
        p = [r for r in pv if lo <= r["date"][:4] <= hi]
        print(f"\n{name}: {len(e)} reports of a customer while a link held ({len({(r['s'], r['c']) for r in e})} "
              f"links, {len({r['c'] for r in e})} customers); {len(p)} for industry peers")
        if not e:
            continue
        print(" Linked suppliers (against the customer's report-session move)")
        line("same session", e, "same")
        line("next 5 sessions", e, "after5")
        line("next 21 sessions", e, "after21")
        line("the supplier's own report, 1-45 days later", e, "own")
        big = [r for r in e if (r["share"] or 0) >= 0.20]
        small = [r for r in e if r["share"] is not None and r["share"] < 0.20]
        print(f" By the customer's share of the supplier's revenue ({len(big)} events at a fifth or more, "
              f"{len(small)} below, {len(e) - len(big) - len(small)} not stated)")
        line("a fifth or more: same session", big, "same")
        line("a fifth or more: next 5 sessions", big, "after5")
        line("a fifth or more: its own report", big, "own")
        line("below a fifth: same session", small, "same")
        line("below a fifth: next 5 sessions", small, "after5")
        print(" Industry peers that do not name the customer")
        line("same session", p, "same")
        line("next 5 sessions", p, "after5")
        line("next 21 sessions", p, "after21")
        line("the peer's own report, 1-45 days later", p, "own")
        if name.startswith("TRAIN"):
            k = [r for r in e if r.get("same") is not None]
            if len(k) >= 50:
                q = np.quantile([r["c_r1"] for r in k], [0.2, 0.8])
                for nm, rr in (("the customer's best fifth of reports", [r for r in k if r["c_r1"] >= q[1]]),
                               ("its worst fifth", [r for r in k if r["c_r1"] <= q[0]])):
                    a5 = [r["after5"] for r in rr if r.get("after5") is not None]
                    print(f"  {nm}: {len(rr)} events, customer {np.mean([r['c_r1'] for r in rr]) * 100:+.2f}%, "
                          f"supplier the same session {np.mean([r['same'] for r in rr]) * 100:+.2f}% "
                          f"({np.mean([r['same'] > 0 for r in rr]) * 100:.0f}% rose), the next 5 sessions "
                          f"{np.mean(a5) * 100:+.2f}%")
    t0, t1 = "2024-01-01", "2026-09-30"
    n = sum(1 for s, c, share, known, until in real for x in eps_by.get(c, [])
            if t0 <= x["date"] <= t1 and known[:10] < x["pre"] and x["date"] <= until[:10])
    per = defaultdict(int)
    for s, c, share, known, until in real:
        for x in eps_by.get(c, []):
            if t0 <= x["date"] <= t1 and known[:10] < x["pre"] and x["date"] <= until[:10]:
                per[season(x["date"])] += 1
    print(f"\n{t0}..{t1}, counts only: {n} reports of a customer while a link held, in "
          f"{sum(v >= MIN_N for v in per.values())} seasons of {MIN_N}+ (no outcome of those years is read here)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
