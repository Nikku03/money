"""TRAIN-only exploration (weeks ending 2011-2023): what did the radar's top list return, and how are jumps spread?

    PYTHONPATH=. python research/moonshots/explore_buy_train.py var/moon_train.npz var/moon_train_fwd.npz \
        > research/moonshots/explore_buy_train.log

The second file holds each stock-week's return against SPY over the next five sessions (the product of its daily
excess returns as the radar defines them). The ranking is S1 refitted on weeks of 2011-2019 and read on 2020-2023;
the 2011-2019 lines use the same model in sample and are marked so.

1. Buying each week's top list in equal parts: the mean and the median five-session return against SPY, how many
   rose, and the week-by-week record (the mean of weekly means, its t, the share of weeks above zero).
2. The same for the names the list does not hold.
3. Where the jumps are: of all jumps, how many sat in stocks under $5, in drug companies, in a results week.
"""
import sys

import numpy as np

from brain import shocks
from brain.research.shocktest import week_rank

d, f = np.load(sys.argv[1], allow_pickle=False), np.load(sys.argv[2])
X, up, down, y, date = d["X"].astype(float), d["up"].astype(float), d["down"].astype(float), d["y"].astype(float), d["date"]
feats = [str(x) for x in d["feats"]]
fwd = f["fwd"]
fit = date <= "2019-12-31"
Z = X[fit]
lo, hi = np.percentile(Z, shocks.CLIP[0], axis=0), np.percentile(Z, shocks.CLIP[1], axis=0)
mu, sd = np.clip(Z, lo, hi).mean(0), np.clip(Z, lo, hi).std(0)
sd = np.where(sd > 0, sd, 1.0)
w = shocks._logit(np.column_stack([np.ones(len(Z)), (np.clip(Z, lo, hi) - mu) / sd]), y[fit], shocks.LAM)
p = 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), (np.clip(X, lo, hi) - mu) / sd]) @ w, -30, 30)))
rk = week_rank(date, p)


def weekly(mask):
    ws = sorted(set(date[mask].tolist()))
    m = np.array([fwd[mask & (date == wk)].mean() for wk in ws])
    return m


print("1. Buying each week's list in equal parts, the next five sessions against SPY")
for nm, span in (("2020-2023 (not fitted here)", ~fit), ("2011-2019 (fitted here)", fit)):
    print(f" {nm}")
    for label, m in (("top 1%", rk <= .01), ("top 5%", rk <= .05), ("top 10%", rk <= .10), ("top 20%", rk <= .20),
                     ("the other 80%", rk > .20), ("every stock", rk <= 1)):
        k = m & span
        wm = weekly(k)
        t = wm.mean() / (wm.std(ddof=1) / np.sqrt(len(wm)))
        print(f"  {label:<14} {int(k.sum()):>7} stock-weeks: mean {fwd[k].mean() * 100:+.2f}%, median {np.median(fwd[k]) * 100:+.2f}%, "
              f"rose {np.mean(fwd[k] > 0) * 100:.0f}%; a jump in {up[k].mean() * 100:.1f}%, a crash in {down[k].mean() * 100:.1f}%; "
              f"by week: mean {wm.mean() * 100:+.2f}% (t {t:+.1f}), {np.mean(wm > 0) * 100:.0f}% of {len(wm)} weeks above zero, "
              f"worst week {wm.min() * 100:+.1f}%, best {wm.max() * 100:+.1f}%")
    k = (rk <= .05) & span
    print(f"  inside the top 5%: the weeks with a jump returned a mean {fwd[k & (up > 0)].mean() * 100:+.1f}%, the weeks with a "
          f"crash {fwd[k & (down > 0) & (up == 0)].mean() * 100:+.1f}%, the rest {fwd[k & (y == 0)].mean() * 100:+.2f}%")
    yrs = sorted(set(x[:4] for x in date[span].tolist()))
    print("  top 5% by year (mean a week):", " ".join(f"{yr}:{weekly((rk <= .05) & np.char.startswith(date, yr)).mean() * 100:+.2f}" for yr in yrs))

print("\n3. Where the jumps are (2011-2023)")
col = {k: X[:, feats.index(k)] for k in ("pen", "bio", "earn_win", "logp")}
J = up > 0
print(f"  {int(J.sum())} stock-weeks with a jump, {J.mean() * 100:.2f}% of all")
for nm, m in (("under $5", col["pen"] > 0), ("drug and biotech companies", col["bio"] > 0), ("a results report due in the window", col["earn_win"] > 0),
              ("none of the three", (col["pen"] == 0) & (col["bio"] == 0) & (col["earn_win"] == 0))):
    print(f"  {nm:<36} {m.mean() * 100:>5.1f}% of stock-weeks, {J[m].sum() / J.sum() * 100:>5.1f}% of jumps; a jump in {J[m].mean() * 100:.2f}% of them")
