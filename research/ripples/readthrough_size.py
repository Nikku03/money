"""TRAIN only: does news that reached a company early (through peers' reports) shrink its own results-day move?"""
import warnings, time
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from brain.bank import Bank
from brain import ripple, shocks, why
T0 = time.time()
bank = Bank(); view = bank.view(); tickers = shocks.universe_tickers()
px = ripple.prices(bank, tickers); cal, R, m = px["cal"], px["R"], px["m"]; N = len(tickers)
rep = []
for t in tickers:
    reps, _, _ = shocks.filings(view, t, cal)
    rep.append(sorted(int(np.searchsorted(cal, d)) for d in reps if np.searchsorted(cal, d) < len(cal) and cal[np.searchsorted(cal, d)] == d))
ind = why.industry_map(view, tickers); sic3 = np.array([ind.get(t, {}).get("sic", "")[:3] for t in tickers])
idx = np.flatnonzero((cal >= "2011-01-01") & (cal <= "2023-12-31")); idx = idx[idx >= ripple.LOOKBACK + 60]
Z = np.full(R.shape, np.nan, np.float32); blocks = list(range(idx[0], idx[-1] + 1, ripple.REBUILD)); Cb = {}
for b0 in blocks:
    p = ripple.pond(R[b0 - ripple.LOOKBACK:b0], m[b0 - ripple.LOOKBACK:b0])
    hi = min(len(cal), b0 + ripple.REBUILD + 3)
    Z[b0:hi] = np.where(np.isnan(Z[b0:hi]), (R[b0:hi] - p["beta"] * m[b0:hi, None]) / p["s"], Z[b0:hi])
    Cb[b0] = p["C"]
blk = np.full(len(cal), -1)
for b0 in blocks:
    blk[b0:b0 + ripple.REBUILD] = b0
reported_on = {}
for i in range(N):
    for k in rep[i]:
        reported_on.setdefault(k, []).append(i)
recs = []
for j in range(N):
    r_list = rep[j]
    for n, r in enumerate(r_list):
        if r < idx[0] or r > idx[-1] or r + 2 >= len(cal) or blk[r] < 0:
            continue
        y = np.nansum(Z[r - 1:r + 2, j]) if np.isfinite(Z[r - 1:r + 2, j]).all() else np.nan
        past = [abs(np.nansum(Z[q - 1:q + 2, j])) for q in r_list[max(0, n - 8):n] if q - 1 >= 0 and np.isfinite(Z[q - 1:q + 2, j]).all()]
        if not np.isfinite(y) or len(past) < 4:
            continue
        habit = float(np.mean(past))
        C = Cb[blk[r]]
        A = T = 0.0; k_leaders = 0
        for t in range(max(0, r - 40), r - 2):
            for i in reported_on.get(t, []):
                if i == j or not (C[j, i] >= 0.25 or (sic3[i] == sic3[j] and sic3[j])):
                    continue
                if not (np.isfinite(Z[t, i]) and np.isfinite(Z[t, j])):
                    continue
                A += abs(float(Z[t, j])); T += max(float(C[j, i]), 0) * abs(float(Z[t, i])); k_leaders += 1
        recs.append((cal[r][:7], abs(y), habit, A, T, k_leaders))
df = pd.DataFrame(recs, columns=["month", "absy", "habit", "A", "T", "k"])
print("reports", len(df), "with any earlier peer report", int((df.k > 0).sum()), round(time.time() - T0), "s")
df["rel"] = np.log(df.absy + 0.25) - np.log(df.habit + 0.25)
for col in ("A", "T", "k"):
    mm = df.groupby("month").apply(lambda h: h[col].rank().corr(h["rel"].rank()) if len(h) >= 20 else np.nan).dropna()
    print(f"IC of {col} (earlier read-through) with the report move relative to habit: {mm.mean():+.4f} (t {mm.mean() / (mm.std() / np.sqrt(len(mm))):+.1f}, {len(mm)} months)")
q = df[df.k > 0]
q["Aq"] = pd.qcut(q.A, 5, labels=False, duplicates="drop")
print(q.groupby("Aq")[["absy", "habit"]].mean().assign(ratio=lambda x: x.absy / x.habit).round(3))
