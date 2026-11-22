"""TRAIN only: the day's storm (how many stones, how big) against next week's move sizes for every stock."""
import warnings, time
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from brain.bank import Bank
from brain import ripple, shocks
T0 = time.time()
bank = Bank(); tickers = shocks.universe_tickers()
px = ripple.prices(bank, tickers)
v = bank.view(); vix = shocks._aligned(shocks._series(v, shocks.VIX), "adjclose", px["cal"])
rows = []
for d in ripple.days(px, "2011-01-01", "2023-12-31"):
    ok = np.isfinite(d["Y"]) & np.isfinite(d["own60"]) & np.isfinite(d["own5"]) & np.isfinite(d["z"])
    if ok.sum() < 200:
        continue
    k = int(np.searchsorted(px["cal"], d["date"]))
    storm_n = len(d["S"]) / ok.sum()                      # share of stocks that are stones
    disp = float(np.sqrt(np.nanmean(d["z"][ok] ** 2)))      # cross-sectional size of own moves today
    rows.append((d["date"][:7], storm_n, disp, d["mkt5"], vix[k],
                 float(np.mean(np.abs(d["Y"][ok]))), float(np.mean(d["own5"][ok])), float(np.mean(d["own60"][ok]))))
df = pd.DataFrame(rows, columns=["month", "stones", "disp", "mkt5", "vix", "absY", "own5", "own60"])
print("sessions", len(df), round(time.time() - T0), "s")
X0 = np.column_stack([np.ones(len(df)), np.log(df.own5), np.log(df.own60), df.mkt5, np.log(df.vix)])
X1 = np.column_stack([X0, np.log(df.disp), np.log(df.stones + 1e-3)])
y = np.log(df.absY)
for name, X in (("own recent moves, market, VIX", X0), ("+ the day's storm", X1)):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"next week's average move size, explained by {name}: R2 {r2:.3f}")
df["q"] = pd.qcut(df.disp, 5, labels=False)
print(df.groupby("q")[["stones", "disp", "absY", "own5", "vix"]].mean().round(3))
# within the same VIX level: storm still matters?
df["vq"] = pd.qcut(df.vix, 3, labels=["calm", "middle", "fearful"])
print(df.groupby(["vq", "q"])["absY"].mean().unstack().round(3))
