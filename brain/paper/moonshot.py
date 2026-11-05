"""The time machine: a paper trader dropped into Monday 4 January 2021 at the opening bell, hunting
moonshots, learning from every trade why the stock moved as it did (docs/PAPER.md, test P3).

Paper only, and a simulation of the past: nothing here places an order.

The clock
    Each week's first session, at the open. It knows prices up to the previous close, that morning's
    opening prices, and SEC filings public before 09:30 New York time. Nothing later.

The trade (fixed in advance)
    Buy the TOP stocks it rates likeliest to moonshot, at the open. Exit at +TARGET (the moonshot),
    at -STOP, or at the close of the HOLD-th session, whichever comes first. A day that touches both
    is counted as the stop (the cautious reading); an open beyond a level exits at the open. COST is
    paid each way.

What it sees (``panel``): each stock's jumpiness (volatility over 20 and 60 sessions, its largest
move and count of 10%+ days over a year), its recent moves (1, 5, 20, 60 sessions, against SPY),
the morning's gap, distance from its 52-week high and low, price and dollar volume, volume surge,
the SEC filings it made public in the last week (by 8-K item group), a share sale in prospect
(prospectus or shelf in 30 days), earnings likely within the hold (last year's results filing, a
year on), whether it is a drug or biotech company, and the market's own week and the VIX.

What it learns (``Learner``): the chance that a stock hits +TARGET before -STOP within HOLD sessions,
by ridge-penalized logistic regression on every stock it watched whose outcome has resolved, not
only the ones it bought, refitted each week, with older outcomes fading (half-life HALF_LIFE weeks).

Why it moved (``explain``): each closed trade's move split into the market's part, oil's, bonds' and
the dollar's (each the stock's own sensitivity over the year before, times that asset's move), and
the stock's own remainder; plus the company's filings during the trade. That is attribution, what
the move went with, not proof of cause.
"""
from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from brain.paper import filings as fl

TARGET, STOP, HOLD, COST, TOP = 0.20, 0.10, 10, 0.0025, 3
LIQUID_DAYS, LIQUID_DOLLARS = 20, 500_000   # P3b: traded every one of the last 20 sessions, $500k a day on average
HALF_LIFE = 52.0          # weeks
RIDGE = 1.0
START = "2021-01-04"
NY = ZoneInfo("America/New_York")
OFFER = ("424B", "S-1", "S-3", "F-1", "F-3")
BIO_SIC = {"2834", "2835", "2836", "8731"}
MACRO = {"SPY": "the market", "USO": "oil", "TLT": "bonds", "UUP": "the dollar"}
CONT = ("lv20", "lv60", "lmax250", "n10", "a1", "a5", "ret5", "ret20", "ret60", "gap", "hi52", "lo52",
        "logp", "ldv", "volsurge")
FLAGS = ("pen", "offer30", "earn_win", "bio") + tuple(fl.GROUPS)
MARKET = ("spy5", "lvix")
FEATURES = CONT + FLAGS + MARKET
PLAIN = {"lv20": "jumpy lately (20-day volatility)", "lv60": "jumpy (60-day volatility)",
         "lmax250": "had a huge one-day move this year", "n10": "many 10%+ days this year",
         "a1": "big move yesterday", "a5": "big move this week", "ret5": "up this week vs the market",
         "ret20": "up this month vs the market", "ret60": "up this quarter vs the market",
         "gap": "gapped up at this morning's open", "hi52": "near its 52-week high", "lo52": "far above its 52-week low",
         "logp": "higher share price", "ldv": "more dollar volume (liquid)", "volsurge": "volume surging",
         "pen": "penny stock (under $5)", "offer30": "share sale in prospect (shelf/prospectus)",
         "earn_win": "earnings likely during the trade", "bio": "drug or biotech company",
         "spy5": "the market rose last week", "lvix": "fear is high (VIX)",
         **{g: "filed: " + g.replace("sec_", "").replace("_", " ") for g in fl.GROUPS}}


# --- data -------------------------------------------------------------------------------------------------
def panels(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Aligned daily panels on SPY's dates: adjusted open/high/low/close, raw close, volume."""
    dates = raw["SPY"]["date"].astype(str).values
    out = {k: {} for k in ("open", "high", "low", "close", "raw", "vol")}
    for t, df in raw.items():
        d = df.set_index(df["date"].astype(str))
        f = (d["adjclose"] / d["close"]).replace([np.inf, -np.inf], np.nan)
        out["open"][t], out["high"][t], out["low"][t] = d["open"] * f, d["high"] * f, d["low"] * f
        out["close"][t], out["raw"][t], out["vol"][t] = d["adjclose"], d["close"], d["volume"]
    return {k: pd.DataFrame(v).reindex(dates).astype(float) for k, v in out.items()}


def visible_index(dates: np.ndarray, known_at: str) -> int:
    """The first session whose OPEN comes after ``known_at``: before 09:30 New York, that day's open;
    after, the next session's."""
    t = datetime.fromisoformat(known_at.replace("Z", "+00:00")).astimezone(NY)
    side = "left" if (t.hour, t.minute) < (9, 30) else "right"
    return int(np.searchsorted(dates, t.date().isoformat(), side=side))


def filing_events(sec: dict, dates: np.ndarray) -> dict[str, list[tuple[int, set]]]:
    out = {}
    for t, rows in sec["filings"].items():
        ev = []
        for known, form, items in rows:
            labels = set()
            if form in ("8-K", "8-K/A"):
                codes = {x.strip() for x in items.split(",") if x.strip()}
                labels = {g for g, c in fl.GROUPS.items() if codes & c}
            if form.startswith(OFFER):
                labels.add("offer")
            if form in fl.ACTIVIST:
                labels.add("sec_activist")
            if labels:
                ev.append((visible_index(dates, known), labels))
        out[t] = ev
    return out


# --- what it sees -----------------------------------------------------------------------------------------
def panel(P: dict, stocks: list[str], events: dict, sic: dict, d: int) -> pd.DataFrame:
    """Features at the open of session ``d``, from data before it (and that open)."""
    c, o, v, raw = P["close"], P["open"], P["vol"], P["raw"]
    lo = d - 260
    C = c.iloc[lo:d]
    r = np.log(C[stocks]).diff().iloc[1:]
    spy = np.log(C["SPY"]).diff().iloc[1:]
    ex = r.sub(spy, axis=0)
    f = pd.DataFrame(index=stocks)
    f["lv20"] = np.log(ex.iloc[-20:].std() + 1e-6)
    f["lv60"] = np.log(ex.iloc[-60:].std() + 1e-6)
    f["lmax250"] = np.log(ex.abs().max() + 1e-6)
    f["n10"] = (ex.abs() > np.log(1.1)).sum()
    f["a1"] = ex.iloc[-1].abs()
    f["a5"] = ex.iloc[-5:].abs().max()
    for k in (5, 20, 60):
        f[f"ret{k}"] = ex.iloc[-k:].sum()
    prev = c[stocks].iloc[d - 1]
    f["gap"] = np.log(o[stocks].iloc[d] / prev)
    f["hi52"] = np.log(prev / C[stocks].max())
    f["lo52"] = np.log(prev / C[stocks].min())
    rawp = raw[stocks].iloc[d - 1]
    f["logp"] = np.log(rawp.clip(lower=0.01))
    dv = (raw[stocks].iloc[d - 20:d] * v[stocks].iloc[d - 20:d]).mean()
    f["ldv"] = np.log(dv.clip(lower=1.0))
    f["volsurge"] = np.log((v[stocks].iloc[d - 5:d].mean() + 1) / (v[stocks].iloc[d - 60:d].mean() + 1))
    f["pen"] = (rawp < 5).astype(float)
    for g in ("offer30", "earn_win", "bio", *fl.GROUPS):
        f[g] = 0.0
    for t in stocks:
        for k, labels in events.get(t, []):
            if d - 5 < k <= d:
                for g in labels & set(fl.GROUPS):
                    f.at[t, g] = 1.0
            if "offer" in labels and d - 30 < k <= d:
                f.at[t, "offer30"] = 1.0
            if "sec_results" in labels and d - 252 <= k < d - 252 + HOLD:     # results a year ago, this window
                f.at[t, "earn_win"] = 1.0
        f.at[t, "bio"] = float(str(sic.get(t)) in BIO_SIC)
    f["spy5"] = float(spy.iloc[-5:].sum())
    f["lvix"] = float(np.log(c["^VIX"].iloc[d - 1])) if "^VIX" in c else 0.0
    return f[list(FEATURES)].astype(float)


# --- the trade --------------------------------------------------------------------------------------------
def outcomes(P: dict, stocks: list[str], d: int) -> pd.DataFrame:
    """For every stock, the trade bought at session d's open: hit (the target first), net return after
    costs, and the session it closed (when its lesson can be learned)."""
    o, h, l, c = (P[k][stocks].values for k in ("open", "high", "low", "close"))
    entry = o[d]
    n = len(stocks)
    done = np.zeros(n, bool)
    ret = np.full(n, np.nan)
    hit = np.zeros(n, bool)
    exit_at = np.full(n, d + HOLD - 1)
    up, dn = entry * (1 + TARGET), entry * (1 - STOP)
    for k in range(HOLD):
        i = d + k
        if i >= len(c):
            break
        if k > 0:                                                  # gaps through a level: out at the open
            g_dn = ~done & (o[i] <= dn)
            g_up = ~done & ~g_dn & (o[i] >= up)
            ret[g_dn], ret[g_up] = o[i][g_dn] / entry[g_dn] - 1, o[i][g_up] / entry[g_up] - 1
            hit |= g_up
            exit_at[g_dn | g_up] = i
            done |= g_dn | g_up
        s = ~done & (l[i] <= dn)                                   # the stop first, if both are touched
        t = ~done & ~s & (h[i] >= up)
        ret[s], ret[t] = -STOP, TARGET
        hit |= t
        exit_at[s | t] = i
        done |= s | t
    last = min(d + HOLD - 1, len(c) - 1)
    rest = ~done
    ret[rest] = c[last][rest] / entry[rest] - 1
    valid = np.isfinite(entry) & np.isfinite(ret) & (d + HOLD - 1 < len(c))
    return pd.DataFrame({"hit": hit.astype(float), "net": ret - 2 * COST, "exit": exit_at, "valid": valid},
                        index=stocks)


# --- the learner ------------------------------------------------------------------------------------------
class Learner:
    """Ridge logistic regression on standardized features, refitted on every resolved observation, older
    ones weighted down by age (half-life HALF_LIFE weeks)."""

    def __init__(self):
        self.w = np.zeros(len(FEATURES) + 1)
        self.mu = np.zeros(len(FEATURES))
        self.sd = np.ones(len(FEATURES))
        self.n = 0

    def _z(self, X):
        return np.clip((X - self.mu) / self.sd, -5, 5)

    def fit(self, X: np.ndarray, y: np.ndarray, age_weeks: np.ndarray) -> None:
        if len(y) < 200 or y.sum() < 5:
            return
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([np.ones((len(y), 1)), self._z(X)])
        sw = 0.5 ** (age_weeks / HALF_LIFE)
        w = self.w.copy()
        R = np.eye(Z.shape[1]) * RIDGE
        R[0, 0] = 0
        for _ in range(25):
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            g = Z.T @ (sw * (p - y)) + R @ w
            H = (Z * (sw * p * (1 - p))[:, None]).T @ Z + R
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-7:
                break
        self.w, self.n = w, len(y)

    def prob(self, X: np.ndarray) -> np.ndarray:
        Z = np.hstack([np.ones((len(X), 1)), self._z(X)])
        return 1 / (1 + np.exp(-np.clip(Z @ self.w, -30, 30)))

    def lessons(self, k: int = 6) -> list[tuple[str, float]]:
        """The features that most raise and lower its moonshot odds, in plain words (weight per sd)."""
        order = np.argsort(self.w[1:])
        pick = list(order[::-1][:k]) + list(order[:k])
        return [(PLAIN[FEATURES[j]], float(self.w[1 + j])) for j in pick]


# --- why it moved -----------------------------------------------------------------------------------------
def explain(P: dict, t: str, d: int, exit_i: int, events: dict) -> dict:
    """The trade's move, split: each macro asset's part (the stock's sensitivity over the year before,
    times the asset's move over the hold) and the stock's own remainder; and its filings meanwhile."""
    c = P["close"]
    hist = np.log(c.iloc[d - 251:d]).diff().iloc[1:]
    total = math.log(c[t].iloc[exit_i] / P["open"][t].iloc[d])
    parts = {}
    X = hist[list(MACRO)].values
    yv = hist[t].values
    ok = np.isfinite(X).all(1) & np.isfinite(yv)
    beta = np.linalg.lstsq(np.c_[np.ones(ok.sum()), X[ok]], yv[ok], rcond=None)[0][1:]
    for b, (a, name) in zip(beta, MACRO.items()):
        move = math.log(c[a].iloc[exit_i] / P["open"][a].iloc[d]) if np.isfinite(P["open"][a].iloc[d]) else 0.0
        parts[name] = float(b * move)
    parts["the stock itself"] = float(total - sum(parts.values()))
    during = sorted({g for k, labels in events.get(t, []) if d < k <= exit_i + 1 for g in labels})
    return {"move": float(total), "parts": parts, "filings_during": during}


def lesson_text(t: str, tr: dict, why: dict) -> str:
    big = max(why["parts"].items(), key=lambda kv: abs(kv[1]))
    filed = (", ".join(x.replace("sec_", "").replace("_", " ") for x in why["filings_during"])) or "no filings"
    res = "hit the moonshot target" if tr["hit"] else ("stopped out" if tr["net"] <= -STOP else "timed out")
    return (f"{t} {res} ({tr['net']:+.1%} net): moved {why['move']:+.1%}; largest part {big[0]} "
            f"({big[1]:+.1%}); during the trade: {filed}")


# --- the simulation ---------------------------------------------------------------------------------------
def tradable(P: dict, stocks: list[str], d: int) -> np.ndarray:
    """Could it have been bought at session d's open? It traded on every one of the last LIQUID_DAYS
    sessions and averaged LIQUID_DOLLARS a day. (P3 had no such check, and picked a stock with no
    volume on 86% of days in 99 of its 909 trades.)"""
    v = P["vol"][stocks].iloc[d - LIQUID_DAYS:d]
    dv = (P["raw"][stocks].iloc[d - LIQUID_DAYS:d] * v).mean()
    return ((v > 0).all() & (dv >= LIQUID_DOLLARS)).values


def simulate(raw: dict, sec: dict, stocks: list[str], start: str = START, shuffle: bool = False,
             seed: int = 0, journal: bool = True, liquid: bool = False) -> dict:
    """``liquid`` (P3b): only stocks that could have been traded are bought or learned from."""
    P = panels(raw)
    dates = P["close"].index.values
    stocks = [s for s in stocks if s in P["close"] and P["close"][s].iloc[:260].notna().sum() > 200]
    events = filing_events(sec, dates)
    week = pd.Series(pd.to_datetime(dates)).dt.to_period("W").values
    firsts = [i for i in range(1, len(dates)) if week[i] != week[i - 1] and i >= 261 and i + HOLD <= len(dates)]
    s0 = int(np.searchsorted(dates, start))
    rng = np.random.default_rng(seed)
    L = Learner()
    seen_X, seen_y, seen_exit, seen_week = [], [], [], []
    trades, weeks, notes = [], [], []
    for wk, d in enumerate(firsts):
        F = panel(P, stocks, events, sec.get("sic", {}), d)
        O = outcomes(P, stocks, d)
        ok = O["valid"].values & np.isfinite(F.values).all(1)
        if liquid:
            ok &= tradable(P, stocks, d)
        if d >= s0:
            Xa = np.vstack(seen_X) if seen_X else np.zeros((0, len(FEATURES)))
            if len(Xa):
                ya, ea, wa = np.concatenate(seen_y), np.concatenate(seen_exit), np.concatenate(seen_week)
                known = ea < d                                      # only outcomes resolved before this open
                L.fit(Xa[known], ya[known], (wk - wa[known]).astype(float))
            p = L.prob(F.values)
            p[~ok] = -1
            pick = np.argsort(p)[::-1][:TOP]
            pick = [j for j in pick if ok[j]]
            book = []
            for j in pick:
                t = stocks[j]
                tr = {"date": str(dates[d]), "ticker": t, "p": float(p[j]), "hit": bool(O["hit"].iloc[j]),
                      "net": float(O["net"].iloc[j]), "exit": str(dates[int(O["exit"].iloc[j])])}
                if journal:
                    why = explain(P, t, d, int(O["exit"].iloc[j]), events)
                    tr["why"] = why
                    tr["lesson"] = lesson_text(t, tr, why)
                trades.append(tr)
                book.append(tr["net"])
            vol_pick = np.argsort(np.where(ok, F["lv20"].values, -np.inf))[::-1][:TOP]
            weeks.append({"date": str(dates[d]), "net": float(np.mean(book)) if book else 0.0,
                          "random": float(O["net"].values[ok].mean()), "hit_rate_all": float(O["hit"].values[ok].mean()),
                          "jumpiest": float(O["net"].values[vol_pick].mean()), "n_fit": L.n})
            if journal and (len(weeks) % 13 == 1):
                notes.append({"date": str(dates[d]), "learned_from": L.n, "lessons": L.lessons()})
        y = O["hit"].values[ok]
        if shuffle:
            y = rng.permutation(y)
        seen_X.append(F.values[ok])
        seen_y.append(y)
        seen_exit.append(O["exit"].values[ok])
        seen_week.append(np.full(ok.sum(), wk))
    return {"trades": trades, "weeks": weeks, "notes": notes, "stocks": len(stocks), "shuffled": shuffle, "liquid": liquid,
            "final_lessons": L.lessons(10)}


def tstat(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float(x.mean()) if len(x) else 0.0, 0.0, 1.0
    t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    return float(x.mean()), float(t), float(0.5 * math.erfc(t / math.sqrt(2)))


def grade(run: dict, shuffled: dict, test: str = "P3") -> dict:
    """P3 or P3b, as declared in docs/PAPER.md before each first run (the same three checks)."""
    w = run["weeks"]
    net = [x["net"] for x in w]
    edge = [x["net"] - x["random"] for x in w]
    vs_shuf = [a["net"] - b["net"] for a, b in zip(w, shuffled["weeks"])]
    m, t, p = tstat(net)
    me, te, pe = tstat(edge)
    ms, ts, ps = tstat(vs_shuf)
    half = len(w) // 2
    tr = run["trades"]
    checks = {"beats random picks from the same list (weekly, one-sided p < 0.05)": me > 0 and pe < 0.05,
              "makes money after costs (weekly mean > 0, one-sided p < 0.05)": m > 0 and p < 0.05,
              "beats the same learner taught shuffled outcomes (p < 0.05)": ms > 0 and ps < 0.05}
    return {"test": test, "weeks": len(w), "trades": len(tr),
            "weekly_net": {"mean": m, "t": t, "p": p}, "vs_random": {"mean": me, "t": te, "p": pe},
            "vs_shuffled": {"mean": ms, "t": ts, "p": ps},
            "hit_rate": float(np.mean([x["hit"] for x in tr])) if tr else 0.0,
            "base_hit_rate": float(np.mean([x["hit_rate_all"] for x in w])),
            "first_half": tstat(net[:half]), "second_half": tstat(net[half:]),
            "hit_rate_by_year": {y: float(np.mean([x["hit"] for x in tr if x["date"][:4] == y]))
                                 for y in sorted({x["date"][:4] for x in tr})},
            "jumpiest_weekly": tstat([x["jumpiest"] for x in w]), "random_weekly": tstat([x["random"] for x in w]),
            "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}
