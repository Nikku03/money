import numpy as np, time
from brain.bank import Bank
from brain import ripple, shocks
t0=time.time()
px=ripple.prices(Bank(), shocks.universe_tickers())
print("prices", round(time.time()-t0), "s", px["R"].shape, flush=True)
xs, ys = [], []
n=0
for d in ripple.days(px, "2012-01-01", "2016-12-31"):
    ok=(~d["stone"])&(d["W"]>0)&np.isfinite(d["z"])
    if ok.sum():
        xs.append(d["H"][ok]); ys.append(d["z"][ok])
    n+=1
x=np.concatenate(xs).astype(float); y=np.concatenate(ys).astype(float)
k=float((x*y).sum()/(x*x).sum())
print("sessions", n, "rows", len(x), "kappa (slope of own move on raw wave)", round(k,4), "corr", round(np.corrcoef(x,y)[0,1],4), round(time.time()-t0),"s")
for lo,hi in ((0,0.5),(0.5,1),(1,2),(2,4),(4,99)):
    m=(np.abs(x)>=lo)&(np.abs(x)<hi)
    if m.sum(): print(f"|raw wave| {lo}-{hi}: n {m.sum()}, mean own move in the wave's direction {np.mean(y[m]*np.sign(x[m])):.3f}, mean |raw| {np.mean(np.abs(x[m])):.3f}")
