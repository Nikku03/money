"""Exploration on TRAIN only (2011-2023): how fast do waves arrive, and where are they slow?
A. Stone -> first-ring partner arrival profiles (days 0..+5), split by sender/receiver liquidity, news, size, sign.
B. Weekly industry lead-lag: leaders' (most traded) week -> followers' (least traded) next week."""
import warnings, time, json
import numpy as np
import pandas as pd
warnings.simplefilter("ignore")
from brain.bank import Bank
from brain import ripple, shocks, connect, why

T0 = time.time()
bank = Bank(); view = bank.view()
tickers = shocks.universe_tickers()
px = ripple.prices(bank, tickers)
cal, R, m = px["cal"], px["R"], px["m"]
N = len(tickers)
# dollar volume, 60-session median ending the day before
DV = np.full(R.shape, np.nan)
for j, t in enumerate(tickers):
    df = shocks._series(view, t)
    if df is not None and len(df) > 60:
        c = shocks._aligned(df, "close", cal); v = shocks._aligned(df, "volume", cal)
        DV[:, j] = c * v
DV60 = pd.DataFrame(DV).rolling(60, min_periods=40).median().shift(1).to_numpy()
LIQ = pd.DataFrame(DV60).rank(axis=1, pct=True).to_numpy()   # 0 = least traded
# results sessions
res_days = np.zeros(R.shape, bool)
for j, t in enumerate(tickers):
    reps, _, _ = shocks.filings(view, t, cal)
    for d in reps:
        k = np.searchsorted(cal, d)
        if k < len(cal) and cal[k] == d:
            res_days[k, j] = True
print("data", round(time.time() - T0), "s", flush=True)

first, last = "2011-01-01", "2023-12-31"
idx = np.flatnonzero((cal >= first) & (cal <= last))
idx = idx[idx >= ripple.LOOKBACK + 60]
# pass 1: point-in-time z for every day (betas/s from the block's pond)
Z = np.full(R.shape, np.nan, np.float32)
blocks = list(range(idx[0], idx[-1] + 1, ripple.REBUILD))
ponds = {}
for b0 in blocks:
    p = ripple.pond(R[b0 - ripple.LOOKBACK:b0], m[b0 - ripple.LOOKBACK:b0])
    hi = min(len(cal), b0 + ripple.REBUILD + 6)
    Z[b0:hi] = np.where(np.isnan(Z[b0:hi]), (R[b0:hi] - p["beta"] * m[b0:hi, None]) / p["s"], Z[b0:hi])
    ponds[b0] = (p["C"] >= ripple.RING)  # first-ring partner mask (positive links)
print("ponds", round(time.time() - T0), "s", flush=True)

rows = []
for b0 in blocks:
    P = ponds[b0]
    for t in range(b0, min(b0 + ripple.REBUILD, idx[-1] + 1)):
        if t + 5 >= len(cal):
            break
        z = Z[t]
        S = np.flatnonzero(np.isfinite(z) & (np.abs(z) >= 3))
        if not len(S):
            continue
        stone = np.zeros(N, bool); stone[S] = True
        for i in S:
            J = np.flatnonzero(P[:, i] & ~stone)
            if not len(J):
                continue
            sg = np.sign(z[i])
            fut = Z[t:t + 6, J] * sg                       # day 0..+5, signed by the stone
            rows.append(np.column_stack([
                np.full(len(J), t), np.full(len(J), i), J, np.full(len(J), z[i]),
                np.full(len(J), LIQ[t, i]), LIQ[t, J], np.full(len(J), res_days[t, i] or res_days[t - 1, i]),
                res_days[t:t + 6, J].any(0), fut.T]))
A = np.vstack(rows).astype(float)
cols = ["t", "i", "j", "zi", "liq_i", "liq_j", "news_i", "news_j_ahead"] + [f"d{k}" for k in range(6)]
df = pd.DataFrame(A, columns=cols)
df = df[np.isfinite(df[[f"d{k}" for k in range(6)]]).all(1) & np.isfinite(df.liq_i) & np.isfinite(df.liq_j)]
df["month"] = [cal[int(x)][:7] for x in df.t]
df["after"] = df[[f"d{k}" for k in range(1, 6)]].sum(1)
print("pairs", len(df), round(time.time() - T0), "s", flush=True)


def summ(g, label):
    mm = g.groupby("month")[["d0", "d1", "after"]].mean()
    t_after = mm["after"].mean() / (mm["after"].std() / np.sqrt(len(mm))) if len(mm) > 2 else np.nan
    print(f"{label:48s} n {len(g):7d}  day0 {g.d0.mean():+.3f}  day1 {g.d1.mean():+.3f}  days1-5 {g.after.mean():+.3f}"
          f" (t {t_after:+.1f}, {len(mm)} months)")


print("\nA. Arrival of the wave at first-ring partners, in the receiver's usual days, signed by the stone")
summ(df, "all")
df["rl"] = pd.cut(df.liq_j, [0, 1/3, 2/3, 1.0001], labels=["quiet", "middle", "busy"], include_lowest=True)
df["sl"] = pd.cut(df.liq_i, [0, 1/3, 2/3, 1.0001], labels=["quiet", "middle", "busy"], include_lowest=True)
for s_ in ("busy", "middle", "quiet"):
    for r_ in ("busy", "middle", "quiet"):
        summ(df[(df.sl == s_) & (df.rl == r_)], f"sender {s_:6s} -> receiver {r_}")
summ(df[df.news_i == 1], "sender on results day")
summ(df[df.news_i == 0], "sender without results")
summ(df[(df.news_i == 1) & (df.liq_i > df.liq_j + 0.3)], "results, sender much busier than receiver")
summ(df[(df.news_i == 1) & (df.liq_i > df.liq_j + 0.3) & (df.news_j_ahead == 0)], "  ... receiver no own results within 5 days")
summ(df[df.zi > 0], "up stones")
summ(df[df.zi < 0], "down stones")
for lo, hi in ((3, 5), (5, 8), (8, 99)):
    summ(df[(df.zi.abs() >= lo) & (df.zi.abs() < hi)], f"stone size {lo}-{hi} usual days")
# receiver under-reacted on day 0 (moved less than a fifth of a usual day with the stone)
summ(df[(df.d0 < 0.2) & (df.liq_i > df.liq_j + 0.3)], "receiver barely moved day 0, sender busier")

print("\nB. Weekly industry lead-lag (leaders = top 30% by trading, followers = bottom 30%, same 3-digit SIC)")
ind = why.industry_map(view, tickers)
sic3 = np.array([ind.get(t, {}).get("sic", "")[:3] for t in tickers])
# weekly blocks of 5 sessions; residual returns in usual days summed
W = []
wk_idx = idx[::5]
recs = []
for a, b in zip(wk_idx[:-2], wk_idx[1:-1]):
    this = np.nansum(Z[a:b], axis=0); nxt = np.nansum(Z[b:b + 5], axis=0)
    ok_this = np.isfinite(Z[a:b]).sum(0) == (b - a); ok_n = np.isfinite(Z[b:b + 5]).sum(0) == 5
    liq = LIQ[a]
    for g in np.unique(sic3):
        if not g:
            continue
        mem = np.flatnonzero((sic3 == g) & ok_this & ok_n & np.isfinite(liq))
        if len(mem) < 8:
            continue
        q = np.quantile(liq[mem], [0.3, 0.7])
        L = mem[liq[mem] >= q[1]]; F = mem[liq[mem] <= q[0]]
        if len(L) < 2 or len(F) < 2:
            continue
        lead = this[L].mean()
        for j in F:
            recs.append((cal[a][:7], a, lead, this[j], nxt[j]))
wk = pd.DataFrame(recs, columns=["month", "week", "lead", "own", "next"])
print("follower-weeks", len(wk))
ics = []
for w_, g in wk.groupby("week"):
    if len(g) >= 20:
        ics.append((g.month.iloc[0], g["lead"].rank().corr(g["next"].rank()), (g["lead"] - g["own"]).rank().corr(g["next"].rank()),
                    g["own"].rank().corr(g["next"].rank())))
ic = pd.DataFrame(ics, columns=["month", "lead", "gap", "own"]).groupby("month").mean()
for c in ("lead", "gap", "own"):
    print(f"  IC of {c:5s} with followers' next week: {ic[c].mean():+.4f} (t {ic[c].mean() / (ic[c].std() / np.sqrt(len(ic))):+.1f}, {len(ic)} months)")
print("done", round(time.time() - T0), "s")
