"""After the fact (X1a judged): is the run-up effect about reports, or the short-term pullback of any day?
For each report episode, an ordinary day of the same stock 10 sessions before its close-before-report (no report
within 3 sessions of it): the same 21-session run in usual moves, and the next session's move against the market.
The monthly rank correlation of -run21 with the next move, report days against ordinary days, TRAIN and TEST.
Also X1a with the run ending one session earlier (leaving out the last day before the report).
Usage: PYTHONPATH=. python research/earnings/after_the_fact_ordinary_days.py"""
import math
import numpy as np
from brain import episodes as ep, events
from brain.bank import Bank
from brain.metrics import PriceBook
from brain.research import earntest as et

def main():
    lib = ep.complete(ep.load(), "r1")
    by = {}
    for r in lib:
        by.setdefault(r["t"], []).append(r)
    bank = Bank(); view = bank.view()
    tick = sorted(by)
    rows_rep, rows_ord, rows_lag = [], [], []
    for c in range(0, len(tick), 200):
        part = tick[c:c + 200]
        book = PriceBook(bank, [*part, "SPY"])
        bd, ba = book.adj_upto(view, "SPY")
        for t in part:
            if not book.has(t):
                continue
            dates, adj = book.adj_upto(view, t)
            b = ep.aligned(dates, bd, ba)
            idx = {d: k for k, d in enumerate(dates)}
            reps = sorted(idx[r["date"]] for r in by[t] if r["date"] in idx)
            for r in by[t]:
                p = idx.get(r["pre"])
                if p is None:
                    continue
                dv = r["day_vol"]
                rows_rep.append((r["date"][:7], -r["run21"], r["r1"]))
                # the run ending one session earlier
                x = ep._excess(adj, b, p - 22, p - 1)
                if x is not None:
                    rows_lag.append((r["date"][:7], -x / (dv * math.sqrt(21)), r["r1"]))
                q = p - 10
                if q - 22 < 0 or any(abs(q - i) <= 3 for i in reps):
                    continue
                run = ep._excess(adj, b, q - 21, q)
                nxt = ep._excess(adj, b, q, q + 1)
                if run is None or nxt is None:
                    continue
                rows_ord.append((str(dates[q + 1])[:7], -run / (dv * math.sqrt(21)), nxt))
    for name, lo, hi in (("TRAIN", "2011-01", "2023-12"), ("TEST", "2024-01", "2026-09")):
        for label, rows in (("report day", rows_rep), ("ordinary day 10 sessions before", rows_ord),
                            ("report day, run ending a session earlier", rows_lag)):
            R = [x for x in rows if lo <= x[0] <= hi]
            m = np.array([x[0] for x in R]); a = np.array([x[1] for x in R]); y = np.array([x[2] for x in R])
            ms, d = et.monthly_ic(m, a, y)
            print(f"{name} {label}: {len(R)} rows, monthly IC {d.mean():+.4f} "
                  f"(t {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f}, {len(d)} months)")

if __name__ == "__main__":
    main()
