import warnings, numpy as np, pandas as pd
warnings.simplefilter("ignore")
from brain.bank import Bank
from brain import ripple, shocks
px = ripple.prices(Bank(), shocks.universe_tickers())
rows = []
for d in ripple.days(px, "2011-01-01", "2023-12-31"):
    ok = np.isfinite(d["z"]) & np.isfinite(d["Y"])
    if ok.sum() >= 200:
        rows.append((float(np.sqrt(np.mean(np.clip(d["z"][ok], -10, 10) ** 2))), float(np.mean(np.abs(d["Y"][ok])))))
df = pd.DataFrame(rows, columns=["storm", "absY"])
edges = np.quantile(df.storm, [0.2, 0.4, 0.6, 0.8])
df["q"] = np.searchsorted(edges, df.storm)
rel = df.groupby("q").absY.mean() / df.absY.mean()
print("edges", np.round(edges, 3).tolist(), "next-week size relative to an average day", np.round(rel.values, 3).tolist(), "days", len(df))
