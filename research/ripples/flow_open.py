"""TRAIN only: does the day-1 arrival in quiet receivers survive buying at the next open?"""
import warnings, time
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from brain.bank import Bank
from brain import ripple, shocks
T0 = time.time()
bank = Bank(); view = bank.view(); tickers = shocks.universe_tickers()
px = ripple.prices(bank, tickers); cal, R, m = px["cal"], px["R"], px["m"]; N = len(tickers)
OP = np.full(R.shape, np.nan); CL = np.full(R.shape, np.nan); DV = np.full(R.shape, np.nan)
for j, t in enumerate(tickers):
    df = shocks._series(view, t)
    if df is not None and len(df) > 60:
        c = shocks._aligned(df, "close", cal); o = shocks._aligned(df, "open", cal); a = shocks._aligned(df, "adjclose", cal)
        f = a / c                                   # adjustment factor, so open and close are on one basis
        OP[:, j] = o * f; CL[:, j] = a; DV[:, j] = c * shocks._aligned(df, "volume", cal)
spy = shocks._series(view, "SPY")
so = shocks._aligned(spy, "open", cal) * shocks._aligned(spy, "adjclose", cal) / shocks._aligned(spy, "close", cal)
sc = shocks._aligned(spy, "adjclose", cal)
LIQ = pd.DataFrame(DV).rolling(60, min_periods=40).median().shift(1).rank(axis=1, pct=True).to_numpy()
idx = np.flatnonzero((cal >= "2011-01-01") & (cal <= "2023-12-31")); idx = idx[idx >= ripple.LOOKBACK + 60]
Z = np.full(R.shape, np.nan, np.float32); blocks = list(range(idx[0], idx[-1] + 1, ripple.REBUILD)); info = {}
for b0 in blocks:
    p = ripple.pond(R[b0 - ripple.LOOKBACK:b0], m[b0 - ripple.LOOKBACK:b0])
    hi = min(len(cal), b0 + ripple.REBUILD + 6)
    Z[b0:hi] = np.where(np.isnan(Z[b0:hi]), (R[b0:hi] - p["beta"] * m[b0:hi, None]) / p["s"], Z[b0:hi])
    info[b0] = (p["C"] >= ripple.RING, p["beta"], p["s"])
recs = []
for b0 in blocks:
    P, beta, s = info[b0]
    for t in range(b0, min(b0 + ripple.REBUILD, idx[-1] + 1)):
        if t + 5 >= len(cal):
            break
        z = Z[t]; S = np.flatnonzero(np.isfinite(z) & (np.abs(z) >= 3))
        if not len(S):
            continue
        stone = np.zeros(N, bool); stone[S] = True
        for i in S:
            J = np.flatnonzero(P[:, i] & ~stone)
            if not len(J):
                continue
            sg = np.sign(z[i])
            # receiver's open(t+1) -> close(t+1) and -> close(t+5), less beta x SPY over the same window, in usual days
            r1 = (CL[t + 1, J] / OP[t + 1, J] - 1) - beta[J] * (sc[t + 1] / so[t + 1] - 1)
            r5 = (CL[t + 5, J] / OP[t + 1, J] - 1) - beta[J] * (sc[t + 5] / so[t + 1] - 1)
            gap = (OP[t + 1, J] / CL[t, J] - 1) - beta[J] * (so[t + 1] / sc[t] - 1)
            recs.append(np.column_stack([np.full(len(J), t), np.full(len(J), LIQ[t, i]), LIQ[t, J], z[J] * sg,
                                         gap / s[J] * sg, r1 / s[J] * sg, r5 / s[J] * sg, s[J]]))
A = pd.DataFrame(np.vstack(recs), columns=["t", "liq_i", "liq_j", "d0", "gap", "o2c1", "o2c5", "s"])
A = A[np.isfinite(A).all(1)]
A["month"] = [cal[int(x)][:7] for x in A.t]
print("pairs", len(A), round(time.time() - T0), "s")
def summ(g, lab):
    mm = g.groupby("month")[["gap", "o2c1", "o2c5"]].mean()
    tt = {c: mm[c].mean() / (mm[c].std() / np.sqrt(len(mm))) for c in mm}
    pct = (g.o2c5 * g.s).mean() * 100
    print(f"{lab:46s} n {len(g):7d} overnight gap {g.gap.mean():+.3f} (t {tt['gap']:+.1f})  open->close day1 {g.o2c1.mean():+.3f} "
          f"(t {tt['o2c1']:+.1f})  open->close day5 {g.o2c5.mean():+.3f} (t {tt['o2c5']:+.1f}) = {pct:+.3f}%")
summ(A, "all")
busy_quiet = (A.liq_i >= 2/3) & (A.liq_j < 1/3)
summ(A[busy_quiet], "busy sender -> quiet receiver")
summ(A[busy_quiet & (A.d0.abs() < 0.2)], "  ... receiver did not move on day 0")
summ(A[(A.liq_j < 1/3)], "quiet receivers, any sender")
summ(A[(A.liq_j < 1/3) & (A.d0.abs() < 0.2)], "  ... did not move on day 0")
summ(A[(A.liq_j >= 2/3)], "busy receivers")
