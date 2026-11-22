"""Does the day's storm widen W1's ranges usefully, beyond W1's own inputs? Fit on TRAIN 2011-2023 (W1's own years).
Then, reported after the fact (2024-2026 has been seen), the same widening on W1a's TEST years."""
import warnings, time, json
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from brain.bank import Bank
from brain import ripple, shocks, scan
from brain.research import rangetest
T0 = time.time()
bank = Bank(); tickers = shocks.universe_tickers()
px = ripple.prices(bank, tickers)
disp = {}
for d in ripple.days(px, "2011-01-01", "2026-09-30", h=0 if False else 5):
    ok = np.isfinite(d["z"])
    if ok.sum() >= 200:
        disp[d["date"]] = (float(np.sqrt(np.mean(np.clip(d["z"][ok], -10, 10) ** 2))), len(d["S"]) / ok.sum())
print("storm readings", len(disp), round(time.time() - T0), "s", flush=True)
model = scan.load()
def load(first, last):
    d = scan.dataset(bank, tickers, first, last, log=None)
    keep = np.array([x in disp for x in d["date"]])
    y = np.asarray(d["l5"], float)[keep]; X = d["X"][keep]; dates = np.array(d["date"])[keep]
    q = scan.quantiles(model["w1"], X)
    sd = np.array([disp[x][0] for x in dates])
    return {"y": y, "q": q, "sd": sd, "date": dates}
tr = load("2011-01-01", "2023-12-31")
print("TRAIN stock-weeks", len(tr["y"]), round(time.time() - T0), "s", flush=True)
med = float(np.median(np.unique(tr["sd"])))
def widen(q, sd, a, b):
    g = np.exp(a + b * (np.log(sd) - np.log(med)))[:, None]
    mid = q[:, 2:3]
    return mid + g * (q - mid)
def loss(q, y):
    return rangetest.pinball(q, y).mean()
base = loss(tr["q"], tr["y"])
best = (base, 0.0, 0.0)
for a in np.arange(-0.2, 0.21, 0.02):
    for b in np.arange(-0.4, 1.21, 0.05):
        l = loss(widen(tr["q"], tr["sd"], a, b), tr["y"])
        if l < best[0]:
            best = (l, float(a), float(b))
print(f"TRAIN: W1 pinball {base:.6f}; widened by the storm {best[0]:.6f} (a {best[1]:+.2f}, b {best[2]:+.2f}); gain {(base - best[0]) / base * 100:.2f}%")
def cover(q, y):
    return float(((y >= q[:, 1]) & (y <= q[:, 3])).mean()), float(((y >= q[:, 0]) & (y <= q[:, 4])).mean())
tr["qq"] = pd.qcut(tr["sd"], 5, labels=False, duplicates="drop")
print("TRAIN coverage by storm fifth (2-in-3, 9-in-10): W1 -> widened")
qw = widen(tr["q"], tr["sd"], best[1], best[2])
for k in range(5):
    m_ = tr["qq"] == k
    print(f"  fifth {k}: storm {np.median(tr['sd'][m_]):.2f}  W1 {cover(tr['q'][m_], tr['y'][m_])}  widened {cover(qw[m_], tr['y'][m_])}")
json.dump({"a": best[1], "b": best[2], "median_storm": med, "train_gain_pct": (base - best[0]) / base * 100},
          open("/tmp/claude-0/-home-user-money/c7630967-e348-5e70-a567-0fa9b42903e6/scratchpad/storm_fit.json", "w"))
te = load("2024-01-01", "2026-09-30")
qwt = widen(te["q"], te["sd"], best[1], best[2])
print(f"\nAFTER THE FACT (seen years) 2024-2026: W1 pinball {loss(te['q'], te['y']):.6f}, widened {loss(qwt, te['y']):.6f}")
te["qq"] = np.searchsorted(np.quantile(tr["sd"], [0.2, 0.4, 0.6, 0.8]), te["sd"])
for k in range(5):
    m_ = te["qq"] == k
    if m_.any():
        print(f"  fifth {k}: n {m_.sum()}  W1 {cover(te['q'][m_], te['y'][m_])}  widened {cover(qwt[m_], te['y'][m_])}")
print("done", round(time.time() - T0), "s")
