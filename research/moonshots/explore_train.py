"""TRAIN-only exploration (weeks ending 2011-2023): can jumps be told from crashes before they happen?

    PYTHONPATH=. python research/moonshots/explore_train.py var/moon_train.npz > research/moonshots/explore_train.log

Asked on 2026-11-28: "i want to predict every moonshots before they happen or atleast most of them." S1 (the shock
radar) gives the chance of a 20%+ session against SPY in the next five, either way; on 2024-2026 its top 10% of
stocks each week held 68% of such sessions, and 68% of them were up. A moonshot is the up kind. The rows are S1's
own (its universes, its features, one row a stock a week), built by ``shocks.dataset`` for 2011-2023 with the up
and down outcomes. Models are S1's ridge logistic regression on S1's features, fitted on weeks of 2011-2019 and
read on weeks of 2020-2023; no week of 2024 or later is read.

1. How many weeks hold a jump, a crash, both.
2. Three rankings read on 2020-2023: S1 refitted on 2011-2019 (any shock), a jump model (up only), and the jump
   model's chance less a crash model's. For each: the share of all jumps in each week's top 1%, 5%, 10% and 20%,
   the jump rate there, and how many crashes sit beside the jumps.
3. What separates a coming jump from a coming crash: the two models' standardized weights, side by side.
4. Is the chance right: predicted against actual in the top twentieth and overall.
"""
import sys

import numpy as np

from brain import shocks
from brain.research.shocktest import auc, week_rank

d = np.load(sys.argv[1], allow_pickle=False)
X, up, down, y, date = d["X"].astype(float), d["up"].astype(float), d["down"].astype(float), d["y"].astype(float), d["date"]
feats = [str(f) for f in d["feats"]]
fit, new = date <= "2019-12-31", date >= "2020-01-01"
print(f"{len(y)} stock-weeks of {len(set(d['t'].tolist()))} stocks, 2011-2023; fitted on {int(fit.sum())} (2011-2019), read on "
      f"{int(new.sum())} (2020-2023)")
print("\n1. Weeks with a 20%+ session against SPY in the next five")
for nm, m in (("2011-2019", fit), ("2020-2023", new)):
    print(f"  {nm}: any {y[m].mean() * 100:.2f}%, a jump {up[m].mean() * 100:.2f}%, a crash {down[m].mean() * 100:.2f}%, both "
          f"{(up[m] * down[m]).mean() * 100:.3f}%; of the weeks with any, a jump in {up[m].sum() / y[m].sum() * 100:.0f}%")


def model(target):
    Z = X[fit]
    lo, hi = np.percentile(Z, shocks.CLIP[0], axis=0), np.percentile(Z, shocks.CLIP[1], axis=0)
    Z = np.clip(Z, lo, hi)
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    w = shocks._logit(np.column_stack([np.ones(len(Z)), (Z - mu) / sd]), target[fit], shocks.LAM)
    def chance(M):
        Zm = (np.clip(M, lo, hi) - mu) / sd
        return 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(Zm)), Zm]) @ w, -30, 30)))
    return w, chance


w_any, c_any = model(y)
w_up, c_up = model(up)
w_dn, c_dn = model(down)
p_any, p_up, p_dn = c_any(X[new]), c_up(X[new]), c_dn(X[new])
wk, U, D = date[new], up[new], down[new]
print("\n2. Read on 2020-2023")
for nm, s in (("S1 refitted (any shock)", p_any), ("the jump model", p_up), ("jump less crash", p_up - p_dn),
              ("jump over any", p_up / np.maximum(p_any, 1e-9))):
    rk = week_rank(wk, s)
    line = f"  {nm:<24} AUC for a jump {auc(s, U):.3f};"
    for k in (0.01, 0.05, 0.10, 0.20):
        top = rk <= k
        line += (f" top {k * 100:.0f}%: {U[top].sum() / U.sum() * 100:.0f}% of jumps, a jump in {U[top].mean() * 100:.1f}%, "
                 f"a crash in {D[top].mean() * 100:.1f}%;")
    print(line)
print("  by year, the share of jumps in each week's top 5% (S1 refitted, then the jump model):",
      " ".join(f"{yr}: {U[(week_rank(wk, p_any) <= .05) & (np.char.startswith(wk, yr))].sum() / max(U[np.char.startswith(wk, yr)].sum(), 1) * 100:.0f}%/"
               f"{U[(week_rank(wk, p_up) <= .05) & (np.char.startswith(wk, yr))].sum() / max(U[np.char.startswith(wk, yr)].sum(), 1) * 100:.0f}%"
               for yr in ("2020", "2021", "2022", "2023")))
print("\n3. Standardized weights (a jump, a crash, the difference), largest differences first")
diff = w_up[1:] - w_dn[1:]
for i in np.argsort(-np.abs(diff)):
    print(f"  {feats[i]:<11} jump {w_up[1 + i]:+.3f}  crash {w_dn[1 + i]:+.3f}  difference {diff[i]:+.3f}")
print("\n4. Calibration of the jump model on 2020-2023")
rk = week_rank(wk, p_up)
print(f"  overall: predicted {p_up.mean() * 100:.2f}%, actual {U.mean() * 100:.2f}%; top twentieth: predicted "
      f"{p_up[rk <= .05].mean() * 100:.1f}%, actual {U[rk <= .05].mean() * 100:.1f}%; top hundredth: predicted "
      f"{p_up[rk <= .01].mean() * 100:.1f}%, actual {U[rk <= .01].mean() * 100:.1f}%")
both = (rk <= .05)
print(f"  in the jump model's top twentieth: {int(both.sum())} stock-weeks, {int(U[both].sum())} with a jump, {int(D[both].sum())} with a crash")
