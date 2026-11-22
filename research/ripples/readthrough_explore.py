"""TRAIN only (2011-2023): an early reporter's results wave reaches peers that report later. At the peer's own
results, is the read-through confirmed (under-reaction) or reversed (over-reaction)?"""
import warnings, time
import numpy as np
import pandas as pd
warnings.simplefilter("ignore")
from brain.bank import Bank
from brain import ripple, shocks, why

T0 = time.time()
bank = Bank(); view = bank.view()
tickers = shocks.universe_tickers()
px = ripple.prices(bank, tickers)
cal, R, m = px["cal"], px["R"], px["m"]
N = len(tickers)
rep = [set() for _ in range(N)]
for j, t in enumerate(tickers):
    reps, _, _ = shocks.filings(view, t, cal)
    for d in reps:
        k = np.searchsorted(cal, d)
        if k < len(cal) and cal[k] == d:
            rep[j].add(int(k))
ind = why.industry_map(view, tickers)
sic3 = np.array([ind.get(t, {}).get("sic", "")[:3] for t in tickers])
first, last = "2011-01-01", "2023-12-31"
idx = np.flatnonzero((cal >= first) & (cal <= last)); idx = idx[idx >= ripple.LOOKBACK + 60]
Z = np.full(R.shape, np.nan, np.float32)
blocks = list(range(idx[0], idx[-1] + 1, ripple.REBUILD))
partner = {}
for b0 in blocks:
    p = ripple.pond(R[b0 - ripple.LOOKBACK:b0], m[b0 - ripple.LOOKBACK:b0])
    hi = min(len(cal), b0 + ripple.REBUILD + 60)
    Z[b0:hi] = np.where(np.isnan(Z[b0:hi]), (R[b0:hi] - p["beta"] * m[b0:hi, None]) / p["s"], Z[b0:hi])
    partner[b0] = p["C"] >= 0.25
print("data", round(time.time() - T0), "s", flush=True)
recs = []
for b0 in blocks:
    P = partner[b0]
    for t in range(b0, min(b0 + ripple.REBUILD, idx[-1] + 1)):
        if t + 45 >= len(cal):
            break
        for i in range(N):
            if t not in rep[i] or not np.isfinite(Z[t, i]) or abs(Z[t, i]) < 2:
                continue
            for j in np.flatnonzero(P[:, i] | ((sic3 == sic3[i]) & (sic3 != ""))):
                if j == i:
                    continue
                later = [r for r in rep[j] if t + 3 <= r <= t + 40]
                if not later or any(t - 2 <= r <= t + 2 for r in rep[j]):
                    continue
                r = min(later)
                x = Z[t, j]; between = np.nansum(Z[t + 1:r - 1, j]); y = np.nansum(Z[r - 1:r + 2, j])
                if np.isfinite(x) and np.isfinite(y):
                    recs.append((cal[t][:7], t, i, j, float(Z[t, i]), float(x), float(between), float(y),
                                 bool(P[j, i]), bool(sic3[j] == sic3[i])))
df = pd.DataFrame(recs, columns=["month", "t", "i", "j", "zi", "x", "between", "y", "partner", "same_ind"])
print("leader-peer pairs", len(df), round(time.time() - T0), "s")


def ic(a, b, g):
    mm = g.groupby("month").apply(lambda h: h[a].rank().corr(h[b].rank()) if len(h) >= 15 else np.nan).dropna()
    return mm.mean(), mm.mean() / (mm.std() / np.sqrt(len(mm))), len(mm)


for name, g in (("all", df), ("partners", df[df.partner]), ("same industry, not partners", df[~df.partner & df.same_ind])):
    print(f"\n{name}: {len(g)} pairs")
    for a, b, lab in (("zi", "x", "leader's surprise -> peer's same-day move (the transfer)"),
                      ("zi", "between", "leader's surprise -> peer's drift until its own report"),
                      ("zi", "y", "leader's surprise -> peer's own report reaction"),
                      ("x", "y", "peer's same-day transfer move -> its own report reaction")):
        v, tv, n = ic(a, b, g)
        print(f"  {lab:62s} IC {v:+.4f} (t {tv:+.1f}, {n} months)")
    # partial: own-report reaction on leader's surprise and the peer's transfer move, pooled
    X = np.column_stack([np.ones(len(g)), np.sign(g.zi), np.clip(g.x, -5, 5)])
    b = np.linalg.lstsq(X, np.clip(g.y, -10, 10), rcond=None)[0]
    print(f"  pooled: own-report reaction = {b[0]:+.3f} {b[1]:+.3f} x sign(leader) {b[2]:+.3f} x transfer move")
print("done", round(time.time() - T0), "s")
