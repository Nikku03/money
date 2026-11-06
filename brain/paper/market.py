"""How the market works (P10, docs/PAPER.md): what runs it, how that reaches each stock, what it
pays for, and how that changes. Paper only, a simulation of the past.

The machines before it learned which trades went well. This one first learns the market, every week,
from every stock on its list, and only then asks what that means for a trade.

The forces (``FORCES``, ``LEVELS``, ``state``)
    The market; big tech; small companies; value stocks; rates (long and medium Treasuries, the
    10-year and 3-month yields and the curve between them); credit (junk bonds against Treasuries);
    the dollar; oil; gold; emerging markets; fear (the VIX, and the VIX against its 3-month version:
    panic now against later); seventeen sectors and industries, from biotech to regional banks. And
    the crowd on its own list: breadth (how many are above their 50-day average), the list's moves
    against the market, dispersion, moonshot weather (how many ran 20%+ in the last two weeks), how
    jumpy they are, earnings season, and January.

How they reach each stock (``exposures``, ``decompose``)
    Each stock's sensitivity to each force over its last year of daily moves: ridge regression on the
    forces, each force net of the market. A move then splits into its forces' part and its own part.

What the market pays for (``learn``)
    Every week, the list's moves (open to next week's open) regressed on 31 traits (``TRAITS``):
    the company's own (size, valuation, growth, margins, R&D, runway, debt, dilution, how its results
    usually go), its price behaviour (volatility, lottery-like days, momentum, distance from its high,
    liquidity, share price, volume surge, the morning's gap), its sensitivities (to the market, rates,
    oil, the dollar, credit, small caps, tech, biotech), last week's push from its forces and its own
    move apart from them (does the market catch up, does a stock's own news fade), and catalysts
    (results due, a share sale in prospect, biotech). Each coefficient is what the market paid that
    week for one standard deviation of a trait. The share of the week's moves the traits explain says
    how much of the week was forces and how much each stock's own news.

How that changes (``forecast``)
    Next week's payoffs are forecast from what each trait paid over the long run, the last year,
    quarter, month and week (persistence), and from attention over every past week
    (``StateAttention``): which past markets looked like this one, and what the market paid in the
    week after them. Learned query and key projections as in P9, but over market weeks, with its
    sharpness bounded the same way. A pooled ridge (``dynamics``) learns how much each source has
    been worth, from the forecasts it actually made at the time.

A stock's expected move (``mm_exp``) is its traits times the forecast payoffs. Every week the model
also scores how well its last forecast ranked the list (its skill record) and writes the week's
lesson: what moved, what the market paid for, and what it had expected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata

FORCES = {"SPY": "the market", "QQQ": "big tech", "IWM": "small companies", "IWD": "value stocks",
          "TLT": "long Treasuries (falling rates)", "HYG": "junk bonds (credit appetite)", "UUP": "the dollar",
          "USO": "oil", "GLD": "gold", "EEM": "emerging markets", "XLK": "tech", "XLF": "finance", "XLE": "energy",
          "XLV": "health care", "XLI": "industrials", "XLY": "consumer spending", "XLP": "consumer staples",
          "XLU": "utilities", "XLB": "materials", "IYR": "real estate", "XBI": "biotech", "KRE": "regional banks",
          "SMH": "semiconductors", "GDX": "gold miners", "XME": "metals and mining", "XOP": "oil and gas producers",
          "XRT": "retail", "ITB": "homebuilders"}
LEVELS = ("^VIX", "^VIX3M", "^TNX", "^IRX")
NEEDED = tuple(FORCES) + ("IEF",) + LEVELS
EQUITY = ("QQQ", "IWM", "IWD", "EEM", "XBI", "KRE", "SMH", "XLE", "XLU", "XLF", "XLK")   # in the state, net of the market
MACRO = ("SPY", "TLT", "UUP", "USO", "GLD")
STATE = (tuple(f"{a}_{w}" for a in MACRO + EQUITY + ("credit",) for w in (5, 20))
         + ("vix", "vix_term", "tnx", "tnx_chg20", "curve", "spy_dd", "spy_vol",
            "breadth", "list5", "list20", "dispersion", "moon", "jumpy", "earn_season", "january"))
PLAIN_STATE = {"SPY": "the market", "TLT": "long Treasuries", "UUP": "the dollar", "USO": "oil", "GLD": "gold",
               "credit": "credit (junk bonds against Treasuries)", **{a: FORCES[a] + " against the market" for a in EQUITY}}
SENS = {"b_mkt": "SPY", "b_rates": "TLT", "b_oil": "USO", "b_dollar": "UUP", "b_credit": "HYG", "b_small": "IWM",
        "b_tech": "QQQ", "b_bio": "XBI"}
TRAITS = {"f_size": "bigger companies", "f_ps": "expensive against sales", "f_rev_growth": "growing revenue",
          "f_op_margin": "profitable operations", "f_rnd": "heavy R&D", "f_runway": "long cash runway",
          "f_debt": "indebted", "f_dilution": "issuing shares", "f_earn_sign": "results usually lift it",
          "lv20": "volatile stocks", "lmax250": "lottery-like stocks (a huge day this year)", "ret60": "up this quarter",
          "hi52": "near the 52-week high", "ldv": "heavily traded", "logp": "high share price",
          "volsurge": "volume surging", "gap": "gapped up at the open",
          "own5": "its own move last week, apart from its forces", "force5": "last week's push from its forces",
          "own20": "its own move last month, apart from its forces",
          "b_mkt": "high market beta", "b_rates": "moves with bond prices", "b_oil": "moves with oil",
          "b_dollar": "moves with the dollar", "b_credit": "moves with credit appetite", "b_small": "moves with small caps",
          "b_tech": "moves with tech", "b_bio": "moves with biotech",
          "earn_win": "results due", "offer30": "share sale in prospect", "bio": "drug and biotech companies"}
SERIES = tuple(TRAITS) + ("list_vs_market", "market")      # what it forecasts each week
PLAIN_SERIES = {**TRAITS, "list_vs_market": "its list against the market", "market": "the market"}
BETA_DAYS, BETA_RIDGE = 252, 100.0
XS_RIDGE = 0.1                     # cross-sectional ridge, per stock
MIN_WEEKS = 52                     # weeks of payoffs before it forecasts
ATT_MIN, ATT_EVERY = 104, 13       # weeks before attention trains; retrain interval
DYN = ("long run", "last year, beyond the long run", "last quarter, beyond the year", "last month, beyond the quarter",
       "last week, beyond the month", "attention, beyond the long run")
D, STEPS, LR, L2, NORM = 8, 300, 0.01, 1e-3, 6.0


def grank(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Gaussianized ranks among ``ok`` (ties share a rank); 0 elsewhere and where missing."""
    z = np.zeros(len(x))
    m = ok & np.isfinite(x)
    n = int(m.sum())
    if n < 10:
        return z
    z[m] = ndtri((rankdata(x[m]) - 0.5) / n)
    return z


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


class StateAttention:
    """Attention over past market weeks: a query from this week's state, keys from every earlier week's
    state, values what the market paid in the week after each. Each week attends only to weeks whose
    payoffs were known before its own decision (strictly earlier rows)."""

    def __init__(self, d_in: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.Wq = self.rng.normal(scale=0.05, size=(D, d_in))
        self.Wk = self.rng.normal(scale=0.05, size=(D, d_in))
        self.mu = self.sd = self.vsd = None
        self.trained_at, self.val_loss, self.base_loss = None, None, None

    def _z(self, S):
        return (np.clip((S - self.mu) / self.sd, -5, 5) / np.sqrt(S.shape[1])).astype(float)

    @staticmethod
    def _attend(Zq, Zk, Wq, Wk, V, mask):
        Q, K = Zq @ Wq.T, Zk @ Wk.T
        L = Q @ K.T / np.sqrt(D)
        L = np.where(mask, L, -1e9)
        L -= L.max(1, keepdims=True)
        A = np.exp(L) * mask
        A /= A.sum(1, keepdims=True)
        return Q, K, A, A @ V

    def fit(self, S: np.ndarray, V: np.ndarray, at: int) -> None:
        """Train on rows 0..T-1 (all finished). Query q attends to rows < q; the latest fifth of
        queries is held out to stop training at its best."""
        T = len(S)
        self.mu, self.sd = S.mean(0), S.std(0) + 1e-9
        self.vsd = V.std(0) + 1e-9
        Z, Vs = self._z(S), V / self.vsd
        q_all = np.arange(26, T)
        cut = q_all[int(len(q_all) * 0.8)]
        tr, va = q_all[q_all < cut], q_all[q_all >= cut]
        keys = np.arange(T)
        mtr, mva = keys[None, :] < tr[:, None], keys[None, :] < va[:, None]
        W = {"Wq": self.Wq.copy(), "Wk": self.Wk.copy()}
        m = {k: np.zeros_like(v) for k, v in W.items()}
        v2 = {k: np.zeros_like(v) for k, v in W.items()}

        def loss(Wd, qs, mask):
            return float(((self._attend(Z[qs], Z, Wd["Wq"], Wd["Wk"], Vs, mask)[3] - Vs[qs]) ** 2).mean())

        flat = {"Wq": np.zeros_like(self.Wq), "Wk": np.zeros_like(self.Wk)}
        self.base_loss = loss(flat, va, mva)                  # uniform attention: the mean of all past weeks
        best = (loss(W, va, mva), {k: x.copy() for k, x in W.items()})
        for step in range(1, STEPS + 1):
            Q, K, A, Yh = self._attend(Z[tr], Z, W["Wq"], W["Wk"], Vs, mtr)
            dY = 2 * (Yh - Vs[tr]) / Yh.size
            dA = dY @ Vs.T
            dL = A * (dA - (dA * A).sum(1, keepdims=True)) / np.sqrt(D)
            g = {"Wq": (dL @ K).T @ Z[tr] + L2 * W["Wq"], "Wk": (dL.T @ Q).T @ Z + L2 * W["Wk"]}
            for k in W:                                        # Adam, then the norm bound
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                W[k] -= LR * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v2[k] / (1 - 0.999 ** step)) + 1e-8)
                n = np.linalg.norm(W[k])
                if n > NORM:
                    W[k] *= NORM / n
            if step % 25 == 0:
                vl = loss(W, va, mva)
                if vl < best[0]:
                    best = (vl, {k: x.copy() for k, x in W.items()})
        self.Wq, self.Wk = best[1]["Wq"], best[1]["Wk"]
        self.val_loss, self.trained_at = best[0], at

    def recall(self, s: np.ndarray, S: np.ndarray, V: np.ndarray, top: int = 3):
        """Attention-weighted payoffs for state ``s`` over the rows S, V (all earlier weeks): the recall,
        how many weeks it effectively attended to, and the ``top`` weeks (row, weight)."""
        if self.mu is None or not len(S):
            return V.mean(0) if len(V) else None, float(len(V)), []
        _, _, A, _ = self._attend(self._z(s[None, :]), self._z(S), self.Wq, self.Wk, V, np.ones((1, len(S)), bool))
        a = A[0]
        idx = np.argsort(-a)[:top]
        return a @ V, float(1.0 / (a * a).sum()), [(int(i), float(a[i])) for i in idx]


class MarketModel:
    """The market model, stepped once per week at the decision's open (see the module text)."""

    def __init__(self, P: dict, stocks: list[str], events: dict, seed: int = 0, shuffle: bool = False):
        self.P, self.stocks, self.events, self.shuffle = P, stocks, events, shuffle
        self.rng = np.random.default_rng(seed + 7)
        self.dates = P["close"].index.values
        cl = P["close"]
        self.tf = list(FORCES)
        fc = cl[self.tf + ["IEF"]].ffill()
        self.Rf = np.log(fc[self.tf]).diff().fillna(0.0).values           # forces, daily log returns
        self.logf = np.log(fc)
        self.lev = cl[list(LEVELS)].ffill()
        self.Rs = np.log(cl[stocks]).diff().values                       # stocks, daily log returns (NaN missing)
        self.logc = np.log(cl[stocks]).values
        self.open_ = P["open"][stocks].values
        self.spy_open = P["open"]["SPY"].values
        self.res = {t: np.array(sorted(k for k, lab in events.get(t, []) if "sec_results" in lab)) for t in stocks}
        self.att = StateAttention(len(STATE), seed)
        self.hist = {"date": [], "S": [], "F": [], "A": [], "ic": [], "ic_static": [], "r2": [], "n": []}
        self.prev = None
        self.coef = np.zeros(len(DYN))
        self._U = []
        self.journal, self.beliefs = [], []
        self.last = {}

    # --- how the forces reach each stock -------------------------------------------------------------
    def exposures(self, d: int):
        X = self.Rf[d - BETA_DAYS + 1:d].copy()                        # returns into closes d-251 .. d-1
        spy = X[:, 0] - X[:, 0].mean()
        b = np.r_[0.0, [(X[:, k] - X[:, k].mean()) @ spy / (spy @ spy) for k in range(1, X.shape[1])]]
        Xo = X - np.outer(X[:, 0], b) * (np.arange(X.shape[1]) > 0)
        Xo -= Xo.mean(0)
        s = Xo.std(0) + 1e-12
        Xs = Xo / s
        Y = np.nan_to_num(self.Rs[d - BETA_DAYS + 1:d])
        Y = Y - Y.mean(0)
        B = np.linalg.solve(Xs.T @ Xs + BETA_RIDGE * np.eye(len(s)), Xs.T @ Y) / s[:, None]
        return B, b

    def decompose(self, d: int, B: np.ndarray, b: np.ndarray, w: int):
        X = self.Rf[d - w:d]
        Xo = X - np.outer(X[:, 0], b) * (np.arange(X.shape[1]) > 0)
        force = Xo.sum(0) @ B
        total = np.nansum(self.Rs[d - w:d], axis=0)
        return force, total - force

    # --- the state of the market ------------------------------------------------------------------------
    def state(self, d: int, ok: np.ndarray, lv20: np.ndarray) -> np.ndarray:
        lf, lev = self.logf, self.lev
        s = {}
        for w in (5, 20):
            spy = lf["SPY"].iloc[d - 1] - lf["SPY"].iloc[d - 1 - w]
            for a in MACRO:
                s[f"{a}_{w}"] = lf[a].iloc[d - 1] - lf[a].iloc[d - 1 - w]
            for a in EQUITY:
                s[f"{a}_{w}"] = lf[a].iloc[d - 1] - lf[a].iloc[d - 1 - w] - spy
            s[f"credit_{w}"] = (lf["HYG"].iloc[d - 1] - lf["HYG"].iloc[d - 1 - w]) - (lf["IEF"].iloc[d - 1] - lf["IEF"].iloc[d - 1 - w])
        vix, v3 = lev["^VIX"].iloc[d - 1], lev["^VIX3M"].iloc[d - 1]
        s["vix"] = np.log(vix)
        s["vix_term"] = np.log(vix / v3) if v3 > 0 else 0.0
        s["tnx"] = lev["^TNX"].iloc[d - 1] / 10
        s["tnx_chg20"] = (lev["^TNX"].iloc[d - 1] - lev["^TNX"].iloc[d - 21]) / 10
        s["curve"] = (lev["^TNX"].iloc[d - 1] - lev["^IRX"].iloc[d - 1]) / 10
        spyc = lf["SPY"].iloc[d - 252:d]
        s["spy_dd"] = spyc.iloc[-1] - spyc.max()
        s["spy_vol"] = np.log(np.std(np.diff(spyc.values[-21:])) * np.sqrt(252) + 1e-6)
        lc = self.logc
        ma50 = np.nanmean(lc[d - 50:d], axis=0)
        s["breadth"] = float(np.mean(lc[d - 1][ok] > ma50[ok])) if ok.any() else 0.5
        spy5, spy20 = s["SPY_5"], s["SPY_20"]
        r5, r20 = lc[d - 1] - lc[d - 6], lc[d - 1] - lc[d - 21]
        s["list5"] = float(np.nanmean(r5[ok]) - spy5) if ok.any() else 0.0
        s["list20"] = float(np.nanmean(r20[ok]) - spy20) if ok.any() else 0.0
        s["dispersion"] = float(np.nanstd(r5[ok])) if ok.any() else 0.0
        run = np.nanmax(lc[d - 10:d], axis=0) - lc[d - 11]
        s["moon"] = float(np.nanmean(run[ok] >= np.log(1.2))) if ok.any() else 0.0
        s["jumpy"] = float(np.nanmedian(lv20[ok])) if ok.any() else 0.0
        s["earn_season"] = float(np.mean([((v > d - 10) & (v <= d)).any() if len(v) else False
                                          for v, o in zip(self.res.values(), ok) if o])) if ok.any() else 0.0
        s["january"] = float(str(self.dates[d])[5:7] == "01")
        return np.array([s[k] for k in STATE], float)

    # --- what the market paid for last week --------------------------------------------------------------
    def learn(self, d: int) -> None:
        """At this open, last week's moves are complete: what did the market pay for, and how did the
        forecast made a week ago do?"""
        if self.prev is None:
            return
        pv = self.prev
        r = np.log(self.open_[d] / self.open_[pv["d"]])
        valid = pv["ok"] & np.isfinite(r)
        if valid.sum() < 30:
            self.prev = None
            return
        r = np.clip(r, -0.4, 0.4)
        y = np.zeros(len(r))
        y[valid] = r[valid] - r[valid].mean()
        ic = spearman(pv["exp"][valid], r[valid]) if pv["exp"] is not None else float("nan")
        ics = spearman(pv["static"][valid], r[valid]) if pv["static"] is not None else float("nan")
        if self.shuffle:                                      # the control: what each stock did is scrambled
            y[valid] = self.rng.permutation(y[valid])
        Z = pv["Z"][valid]
        n = len(Z)
        f = np.linalg.solve(Z.T @ Z + XS_RIDGE * n * np.eye(Z.shape[1]), Z.T @ y[valid])
        resid = y[valid] - Z @ f
        r2 = float(1 - (resid @ resid) / max(1e-12, y[valid] @ y[valid]))
        r2 = 1 - (1 - r2) * (n - 1) / max(1, n - Z.shape[1] - 1)       # adjusted for the number of traits
        spy = float(np.log(self.spy_open[d] / self.spy_open[pv["d"]]))
        lvm = float(r[valid].mean() - spy)
        H = self.hist
        H["date"].append(str(self.dates[pv["d"]]))
        H["S"].append(pv["s"])
        H["F"].append(np.r_[f, lvm, spy])
        # before it forecasts, its recall is uniform attention: the mean of every week so far
        H["A"].append(pv["att"] if pv["att"] is not None else
                      (np.mean(H["F"], 0) if H["F"] else np.zeros(len(SERIES))))
        H["ic"].append(ic)
        H["ic_static"].append(ics)
        H["r2"].append(r2)
        H["n"].append(n)
        self.journal.append(self._lesson(pv, d, f, r2, spy, lvm, ic))
        self.prev = None

    # --- forecasting what it will pay -------------------------------------------------------------------
    def _inputs(self, F: np.ndarray, A: np.ndarray, t: int) -> np.ndarray:
        """For row t (payoffs F[t] unknown), from F[:t]: each series' long-run mean, then what each shorter
        span (52, 13, 4 weeks, the last week) adds beyond the next longer one, and what the attention
        recall made at the time adds beyond the long run: (series x 6). Increments, so that a source
        that only repeats another (attention while it is still uniform) carries nothing."""
        past = F[:t]
        lr, y, q, m = past.mean(0), past[-52:].mean(0), past[-13:].mean(0), past[-4:].mean(0)
        return np.stack([lr, y - lr, q - y, m - q, past[-1] - m, A[t] - lr], 1)

    def dynamics(self) -> None:
        """How much each source of a forecast has been worth: a ridge pooled over every series and every
        week from the 26th, on the inputs as they stood before each week (each series in units of its
        own spread). The inputs of a finished week never change, so they are kept."""
        F, A = np.array(self.hist["F"]), np.array(self.hist["A"])
        T = len(F)
        while len(self._U) < T:
            t = len(self._U)
            self._U.append(self._inputs(F, A, t) if t >= 26 else None)
        sd = F.std(0) + 1e-12
        X = np.vstack([self._U[t] / sd[:, None] for t in range(26, T)])
        y = np.concatenate([F[t] / sd for t in range(26, T)])
        s = X.std(0) + 1e-12
        Xs = X / s
        self.coef = np.linalg.solve(Xs.T @ Xs + 0.01 * len(y) * np.eye(X.shape[1]), Xs.T @ y) / s

    def forecast(self, s: np.ndarray, wk: int):
        H = self.hist
        T = len(H["F"])
        if T < MIN_WEEKS:
            return None, None, None
        F, S = np.array(H["F"]), np.array(H["S"])
        if T >= ATT_MIN and (self.att.trained_at is None or wk - self.att.trained_at >= ATT_EVERY):
            self.att.fit(S, F, wk)
        att, eff, top = self.att.recall(s, S, F)
        self.dynamics()
        A = np.vstack([np.array(H["A"]), att[None, :]])
        u = self._inputs(np.vstack([F, np.zeros((1, F.shape[1]))]), A, T)
        sd = F.std(0) + 1e-12
        fhat = (u / sd[:, None]) @ self.coef * sd
        self.last = {"att_eff": eff, "att_top": [(H["date"][i], w) for i, w in top]}
        return fhat, att, F.mean(0)

    # --- each week ------------------------------------------------------------------------------------------
    def week(self, d: int, wk: int, F: pd.DataFrame, E: pd.DataFrame, ok: np.ndarray) -> dict:
        """At session d's open: learn last week, read the state, forecast, and each stock's view.
        Returns per-stock features (arrays) and keeps what next week's lesson needs."""
        self.learn(d)
        B, b = self.exposures(d)
        f5, o5 = self.decompose(d, B, b, 5)
        _, o20 = self.decompose(d, B, b, 20)
        cols = {**{c: E[c].values for c in TRAITS if c.startswith("f_")},
                **{c: F[c].values for c in ("lv20", "lmax250", "ret60", "hi52", "ldv", "logp", "volsurge", "gap",
                                            "earn_win", "offer30", "bio")},
                "own5": o5, "force5": f5, "own20": o20,
                **{c: B[self.tf.index(a)] for c, a in SENS.items()}}
        Z = np.column_stack([grank(np.asarray(cols[c], float), ok) for c in TRAITS])
        s = self.state(d, ok, F["lv20"].values)
        fhat, att, lr = self.forecast(s, wk)
        k = len(TRAITS)
        exp = Z @ fhat[:k] if fhat is not None else np.zeros(len(ok))
        static = Z @ lr[:k] if lr is not None else None
        self.prev = {"d": d, "ok": ok.copy(), "Z": Z, "s": s, "exp": exp if fhat is not None else None,
                     "static": static, "att": att, "fhat": fhat}
        self.Z, self.B = Z, B
        ics = [x for x in self.hist["ic"][-26:] if np.isfinite(x)]
        return {"mm_exp": exp, "mm_force5": f5, "mm_own5": o5, "mm_own20": o20,
                "mm_list": np.full(len(ok), fhat[k] + fhat[k + 1] if fhat is not None else 0.0),
                "mm_breadth": np.full(len(ok), s[STATE.index("breadth")]),
                "mm_moon": np.full(len(ok), s[STATE.index("moon")]),
                "skill": float(np.mean(ics)) if ics else float("nan")}

    # --- words ----------------------------------------------------------------------------------------------
    def view(self, j: int, k: int = 3) -> dict:
        """One stock, as the market model sees it at this open: its strongest forces and what its traits
        are expected to earn this week."""
        pv = self.prev
        B = self.B[:, j]
        sd = self.Rf[pv["d"] - BETA_DAYS + 1:pv["d"]].std(0)       # a force's daily spread over the same year
        top = np.argsort(-np.abs(B) * sd)[:k]
        out = {"forces": [f"{FORCES[self.tf[i]]} (beta {B[i]:+.2f})" for i in top]}
        if pv["fhat"] is not None:
            c = pv["Z"][j] * pv["fhat"][:len(TRAITS)]
            top = np.argsort(-np.abs(c))[:k]
            out["expected"] = float(c.sum())
            out["why"] = [f"{list(TRAITS.values())[i]} ({pv['Z'][j][i]:+.1f} sd, {c[i]:+.2%})" for i in top]
        return out

    def _lesson(self, pv: dict, d: int, f: np.ndarray, r2: float, spy: float, lvm: float, ic: float) -> dict:
        names = list(TRAITS.values())
        order = np.argsort(f)
        paid = [(names[i], float(f[i])) for i in order[::-1][:3] if f[i] > 0]
        punished = [(names[i], float(f[i])) for i in order[:3] if f[i] < 0]
        lf, a0 = self.logf, pv["d"] - 1                       # the forces' moves during the week, close to close
        mv = {a: float(lf[a].iloc[d - 1] - lf[a].iloc[a0]) for a in MACRO + EQUITY + ("HYG", "IEF")}
        moves = [(PLAIN_STATE[a], mv[a]) for a in MACRO] + [(PLAIN_STATE[a], mv[a] - mv["SPY"]) for a in EQUITY]
        moves.append((PLAIN_STATE["credit"], mv["HYG"] - mv["IEF"]))
        moves = sorted(moves, key=lambda kv: -abs(kv[1]))[:4]
        exp = None
        if pv["fhat"] is not None:
            eo = np.argsort(pv["fhat"][:len(TRAITS)])
            exp = [(names[i], float(pv["fhat"][i])) for i in eo[::-1][:3]]
        text = (f"Week of {self.dates[pv['d']]}: the market {spy:+.1%}, its list {lvm:+.1%} against it; "
                f"{', '.join(f'{n} {v:+.1%}' for n, v in moves)}. "
                f"The traits explained {r2:.0%} of the list's moves. "
                f"Paid for: {', '.join(f'{n} ({v:+.2%})' for n, v in paid) or 'nothing'}; "
                f"punished: {', '.join(f'{n} ({v:+.2%})' for n, v in punished) or 'nothing'}. "
                + (f"It had expected {', '.join(f'{n} ({v:+.2%})' for n, v in exp)} to pay; its ranking of the list "
                   f"came out with a rank correlation of {ic:+.2f}." if exp is not None and np.isfinite(ic)
                   else "It was not forecasting yet."))
        return {"week": str(self.dates[pv["d"]]), "known": str(self.dates[d]), "market": spy, "list_vs_market": lvm, "explained": r2,
                "forces": moves, "paid": paid, "punished": punished, "expected": exp,
                "ic": ic, "attended": self.last.get("att_top") if exp is not None else None, "lesson": text}

    def belief(self) -> dict:
        """What it believes about how the market works: long-run payoffs and how forecasts are made."""
        F = np.array(self.hist["F"]) if self.hist["F"] else np.zeros((0, len(SERIES)))
        names = list(TRAITS.values())
        lr = F[:, :len(TRAITS)].mean(0) if len(F) else np.zeros(len(TRAITS))
        t = lr / (F[:, :len(TRAITS)].std(0) / np.sqrt(max(1, len(F))) + 1e-12) if len(F) else lr
        order = np.argsort(t)
        ics = [x for x in self.hist["ic"] if np.isfinite(x)]
        return {"weeks": len(F), "rewarded": [(names[i], float(lr[i]), float(t[i])) for i in order[::-1][:5]],
                "punished": [(names[i], float(lr[i]), float(t[i])) for i in order[:5]],
                "dynamics": dict(zip(DYN, map(float, self.coef))),
                "skill": {"weeks": len(ics), "mean_ic": float(np.mean(ics)) if ics else None},
                "explained": float(np.mean(self.hist["r2"][-52:])) if self.hist["r2"] else None,
                "attention": {"weeks_attended": self.last.get("att_eff"), "val_loss": self.att.val_loss,
                              "uniform_loss": self.att.base_loss}}
