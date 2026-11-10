"""The market scan (docs/SCAN.md): for every stock the bank follows, how far it is likely to move over the next five
sessions, how likely a 20%+ session is, whether a results report is due, and why. Size, not direction.

Every direction test in this repo has failed; size has passed each time it was tested (the forecaster's bands, M1,
S1). The scan puts the tested size forecasts in one place and says plainly that it does not know which way.

**W1, the weekly range.** A row is one stock at one close; everything in it was known at that close.

- The scale: ``log(sum of the next five squared daily log returns) = x . w``, least squares with a small ridge on
  S1's features (``brain/shocks.py``) plus two: ``lv5``, the stock's own variance over the last five sessions, and
  ``fearvol``, the log VIX times ``lv250`` (market fear moves a calm stock's range more than a wild one's). The
  features are standardized and clipped at the training years' 0.1 and 99.9 percentiles. ``sigma = exp(x . w / 2)``
  is the size of an ordinary five-session move for that stock that week.
- The shape: the five-session log return over ``sigma`` (``z``), whose distribution is the training years' own, kept
  apart for each tenth of ``sigma`` (calm stocks and wild ones are skewed differently) and, within each tenth, for
  weeks with a report due (``earn_win``: a report makes a week's move a jump or nothing). A quantile of the return
  is ``exp(z_q * sigma) - 1``.
- The baseline is the same fit on ``lv60`` alone, with one shape per tenth: the stock's 60-session volatility.

Both live in ``knowledge/scan_w1.json``, fitted on TRAIN_START..TRAIN_END; the years after are the declared test's
(``brain/research/rangetest.py``).

**Tradable rows only** (the paper trader's rule, P3b): a bar on the reading day, trades on each of the last 20
sessions, an average of $500,000 a day over them, and no data error (a day beyond +500% or -95%, or a close under
$0.05) in the last 250 sessions. A stock that fails is listed apart with the reason and gets no forecast.

Nothing here says which way a stock will go, and nothing here trades.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from brain import config, shocks
from brain.bank import Bank, canon

H = shocks.H
TRAIN_START, TRAIN_END = shocks.TRAIN_START, shocks.TRAIN_END
FEATS = (*shocks.FEATS, "lv5", "fearvol")
BASELINE = ("lv60",)
SPLIT = "earn_win"          # W1 keeps z apart for rows where this feature is above zero
MIN_SHAPE = 500             # training rows a split group needs for a shape of its own
QS = (0.05, 0.16, 0.5, 0.84, 0.95)          # the 90% range, the 68% range and the middle
GRID = tuple(round(0.005 * k, 3) for k in range(1, 200))   # quantiles of z kept for each tenth of sigma
BUCKETS = 10
LAM = 1.0
CLIP = shocks.CLIP
FLOOR = 4e-6                # the least realized variance counted: a 0.2% week
TRADE_DAYS, MIN_DV = 20, 500_000.0
ERROR_DAYS = 250
DEFAULT_MODEL = config.ROOT / "knowledge" / "scan_w1.json"
WATCH = "us_watch"          # the scan's newer listings: SEC ranks 1-3000 that listed after 2010 (universe/us_watch.json)
CHUNK = shocks.CHUNK


# ---------- arrays for one stock (pure) ----------
def extras(close: np.ndarray, adj: np.ndarray, vol: np.ndarray, has: np.ndarray) -> dict[str, np.ndarray]:
    """What W1 needs beyond S1's features, for every session of one stock. ``close`` and ``adj`` are aligned to
    the market's sessions (a missing day carries the last close); ``vol`` is the day's own volume (NaN on a
    missing day) and ``has`` says whether the stock had a bar that day.

    - ``l5``, ``rv``: the log return and the sum of squared daily log returns over the next H sessions; NaN when
      the window is not closed, holds a data error, or ends on a day without a bar.
    - ``lv5``: the log of the root mean squared daily log return over the last five sessions.
    - ``ok``: tradable at this close (the module's rule); ``why``: 0 when it is, else the first rule it fails
      (1 no bar that day, 2 a session without trades in the last 20, 3 under $500,000 a day, 4 a data error in
      the last 250 sessions)."""
    n = len(close)
    with np.errstate(all="ignore"):
        r = np.r_[np.nan, adj[1:] / adj[:-1] - 1]
        bad = (r > shocks.BAD_UP) | (r < shocks.BAD_DOWN) | (close < shocks.MIN_PRICE)
        lr = np.where(bad | ~np.isfinite(r) | (r <= -1), np.nan, np.log1p(np.where(r > -1, r, 0.0)))
    fin = np.isfinite(lr)
    c1 = np.r_[0.0, np.cumsum(np.where(fin, lr, 0.0))]
    c2 = np.r_[0.0, np.cumsum(np.where(fin, lr, 0.0) ** 2)]
    cm = np.r_[0.0, np.cumsum(~fin)]
    l5, rv = np.full(n, np.nan), np.full(n, np.nan)
    i = np.arange(max(n - H, 0))
    whole = ((cm[i + H + 1] - cm[i + 1]) == 0) & has[i + H] & has[i]
    l5[i] = np.where(whole, c1[i + H + 1] - c1[i + 1], np.nan)
    rv[i] = np.where(whole, c2[i + H + 1] - c2[i + 1], np.nan)
    traded = has & (np.nan_to_num(vol) > 0)
    with np.errstate(all="ignore"):
        t20 = shocks._roll(traded.astype(float), TRADE_DAYS, "sum")
        dv = shocks._roll(np.where(traded, close * np.nan_to_num(vol), 0.0), TRADE_DAYS, "mean")
        err = shocks._roll(bad.astype(float), ERROR_DAYS, "sum", 1)
        lv5 = 0.5 * np.log(shocks._roll(np.where(fin, lr, np.nan) ** 2, 5, "mean", 4) + 1e-8)
    why = np.where(~has, 1, np.where(~(t20 >= TRADE_DAYS), 2, np.where(~(dv >= MIN_DV), 3,
                                                                    np.where(np.nan_to_num(err) > 0, 4, 0))))
    return {"l5": l5, "rv": rv, "lv5": lv5, "ok": why == 0, "why": why.astype(float)}


WHY = {1: "no price on the reading day", 2: "a session without trades in the last 20",
       3: "under $500,000 traded a day", 4: "a data error in its last year of prices"}


# ---------- from the bank ----------
def _own(df, col: str, cal: np.ndarray) -> np.ndarray:
    """A column on the market's sessions with no carrying forward: NaN where the stock had no bar."""
    import pandas as pd
    s = pd.Series(df[col].to_numpy(float), index=df["date"].to_numpy(dtype=str))
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(cal).to_numpy()


def ticker_frame(view, t: str, mk) -> tuple[np.ndarray, dict] | None:
    """S1's features and W1's extras for every session of ``t``, as known at ``view``."""
    cal, bclose, vclose = mk
    df = shocks._series(view, t)
    if df is None or len(df) < shocks.MIN_RET + 60:
        return None
    close, adj, vol = (shocks._aligned(df, c, cal) for c in ("close", "adjclose", "volume"))
    reports, news, offer = shocks.filings(view, t, cal)
    f = shocks.frame(cal, close, adj, vol, bclose, vclose, reports, news, offer, shocks.is_bio(view, t))
    own = _own(df, "close", cal)
    f.update(extras(close, adj, _own(df, "volume", cal), np.isfinite(own)))
    f["fearvol"] = f["vix"] * f["lv250"]
    return cal, f


def dataset(bank: Bank, tickers: list[str], first: str, last: str, log=print, step: str = "week") -> dict:
    """Tradable stock-weeks whose week ends in [first, last] and whose five sessions have closed, as arrays:
    ``t`` and ``date`` (lists), ``X`` (one column per FEATS), ``l5``, ``rv``. ``step="day"`` reads every session's
    close instead of each week's last. A fresh bank per CHUNK tickers keeps memory flat."""
    T, D, X, L, R = [], [], [], [], []
    for c in range(0, len(tickers), CHUNK):
        b = Bank(bank.root)
        v = b.view()
        mk = shocks.market(v)
        if mk is None:
            raise SystemExit(f"no {shocks.BENCH} or {shocks.VIX} prices in the bank: "
                             f"python -m brain ingest {shocks.BENCH} {shocks.VIX}")
        ends = np.arange(len(mk[0])) if step == "day" else shocks.week_ends(mk[0])
        ends = ends[(mk[0][ends] >= first) & (mk[0][ends] <= last)]
        for t in tickers[c:c + CHUNK]:
            got = ticker_frame(v, t, mk)
            if not got:
                continue
            cal, f = got
            x = np.column_stack([f[k][ends] for k in FEATS])
            keep = np.isfinite(x).all(1) & f["ok"][ends] & np.isfinite(f["l5"][ends]) & np.isfinite(f["rv"][ends])
            if keep.any():
                T += [t] * int(keep.sum())
                D += [str(d) for d in cal[ends][keep]]
                X.append(x[keep].astype(np.float32))
                L.append(f["l5"][ends][keep])
                R.append(f["rv"][ends][keep])
        if log:
            log(f"{min(c + CHUNK, len(tickers))}/{len(tickers)} tickers, {len(T)} stock-weeks")
        del b, v
        gc.collect()
    if not T:
        return {"t": [], "date": [], "X": np.zeros((0, len(FEATS)), np.float32), "l5": np.zeros(0), "rv": np.zeros(0)}
    return {"t": T, "date": D, "X": np.concatenate(X), "l5": np.concatenate(L), "rv": np.concatenate(R)}


# ---------- the model ----------
def _cols(feats) -> list[int]:
    return [FEATS.index(k) for k in feats]


def _group(X: np.ndarray, ls: np.ndarray, edges, split: str | None) -> np.ndarray:
    """Which shape a row takes: its tenth of the training scales, plus BUCKETS where ``split`` is above zero."""
    b = np.searchsorted(np.asarray(edges), ls, side="right")
    return b + BUCKETS * (np.asarray(X)[:, FEATS.index(split)] > 0) if split else b


def fit(X: np.ndarray, l5: np.ndarray, rv: np.ndarray, feats=FEATS, lam: float = LAM, split: str | None = None) -> dict:
    """The scale by ridge least squares, then z's quantiles for each tenth of the fitted scale (and, with
    ``split``, apart for the rows where that feature is above zero)."""
    Z = np.asarray(X, float)[:, _cols(feats)]
    lo, hi = np.percentile(Z, CLIP[0], axis=0), np.percentile(Z, CLIP[1], axis=0)
    Z = np.clip(Z, lo, hi)
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    A = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
    pen = lam * np.eye(A.shape[1])
    pen[0, 0] = 0.0
    w = np.linalg.solve(A.T @ A + pen, A.T @ np.log(np.maximum(rv, FLOOR)))
    ls = 0.5 * (A @ w)
    z = l5 / np.exp(ls)
    edges = np.quantile(ls, np.arange(1, BUCKETS) / BUCKETS)
    b = _group(X, ls, edges, split)
    tenth = b % BUCKETS
    # a split group too thin for its own shape (under MIN_SHAPE rows) takes its whole tenth's
    use = [b == k if (b == k).sum() >= MIN_SHAPE else tenth == k % BUCKETS for k in range(BUCKETS * (2 if split else 1))]
    if any(m.sum() < 2 for m in use):
        raise SystemExit("too few stock-weeks to fit a shape for each tenth")
    return {"feats": list(feats), "lo": lo.tolist(), "hi": hi.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
            "w": w.tolist(), "lam": lam, "edges": edges.tolist(), "split": split,
            "z": [np.quantile(z[m], GRID).tolist() for m in use],
            "up": [float((z[m] > 0).mean()) for m in use], "n": len(z)}


def scale(part: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's sigma and which shape it takes (``_group``)."""
    Z = np.clip(np.asarray(X, float)[:, _cols(part["feats"])], part["lo"], part["hi"])
    A = np.column_stack([np.ones(len(Z)), (Z - np.array(part["mu"])) / np.array(part["sd"])])
    ls = 0.5 * (A @ np.array(part["w"]))
    return np.exp(ls), _group(X, ls, part["edges"], part.get("split"))


def z_at(part: dict, bucket: np.ndarray, q: float) -> np.ndarray:
    """z's q-quantile for each row's tenth (linear between the kept quantiles, flat beyond them)."""
    zq = np.array([np.interp(q, GRID, g) for g in part["z"]])
    return zq[bucket]


def quantiles(part: dict, X: np.ndarray, qs=QS) -> np.ndarray:
    """The five-session log return's quantiles, one row per stock-week and one column per q."""
    s, b = scale(part, X)
    return np.column_stack([z_at(part, b, q) * s for q in qs])


def chance_beyond(part: dict, X: np.ndarray, size: float) -> np.ndarray:
    """The chance the five-session move is at least ``size`` (a fraction of the price), either way."""
    s, b = scale(part, X)
    out = np.empty(len(s))
    g = np.array(GRID)
    for k in range(len(s)):
        z = np.array(part["z"][b[k]])
        up = 1 - np.interp(np.log1p(size) / s[k], z, g, left=0.0, right=1.0)
        down = np.interp(np.log1p(-size) / s[k], z, g, left=0.0, right=1.0) if size < 1 else 0.0
        out[k] = min(max(up + down, 0.0), 1.0)
    return out


def model_path() -> Path:
    """The fitted models' file: ``knowledge/scan_w1.json``, or BRAIN_SCAN_MODEL (tests, trials)."""
    return Path(os.environ.get("BRAIN_SCAN_MODEL") or DEFAULT_MODEL)


def load(path: Path | None = None) -> dict | None:
    p = Path(path or model_path())
    return json.loads(p.read_text()) if p.exists() else None


def model_sha(path: Path | None = None) -> str:
    return hashlib.sha256(Path(path or model_path()).read_bytes()).hexdigest()[:16]


def fit_and_save(bank: Bank, path: Path | None = None, log=print, tickers=None) -> dict:
    """Fit W1 and the baseline on the universe's tradable stock-weeks from TRAIN_START to TRAIN_END and save both.
    Nothing after TRAIN_END is read into the fit: those years belong to the declared test."""
    d = dataset(bank, tickers or shocks.universe_tickers(), TRAIN_START, TRAIN_END, log=log)
    if not len(d["l5"]):
        raise SystemExit("no stock-weeks to fit on")
    out = {"name": "W1", "train": [TRAIN_START, TRAIN_END], "universes": list(shocks.UNIVERSES), "h": H,
           "w1": fit(d["X"], d["l5"], d["rv"], FEATS, split=SPLIT), "baseline": fit(d["X"], d["l5"], d["rv"], BASELINE)}
    p = Path(path or model_path())
    p.write_text(canon(out) + "\n")
    if log:
        log(f"fitted on {len(d['l5'])} stock-weeks ({TRAIN_START}..{TRAIN_END}); saved {p}")
    return out


# ---------- the scan ----------
GROUPS = {"swings": ("lv5", "lv20", "lv60", "lv250", "lmax250", "lmax2", "n20", "n10", "a1", "a5", "big1"),
          "report": ("earn_win", "earn_known"), "price": ("logp", "logp2", "pen", "ldv", "ldv2", "volsurge"),
          "run": ("ret5", "ret20", "ret60", "hi52", "lo52"), "filings": ("news5", "offer30"),
          "drug": ("bio", "bio_known"), "fear": ("vix", "fearvol")}
MIN_PULL = 0.05            # a group's push on the log variance worth naming


def pulls(part: dict, row: dict) -> dict[str, float]:
    """How far each group of features pushes this stock's log variance above (or below) an average stock-week's."""
    feats = part["feats"]
    z = np.clip(np.array([row[k] for k in feats], float), part["lo"], part["hi"])
    c = (z - np.array(part["mu"])) / np.array(part["sd"]) * np.array(part["w"][1:])
    return {g: float(sum(c[feats.index(k)] for k in ks if k in feats)) for g, ks in GROUPS.items()}


def _pct(x: float, signed: bool = False) -> str:
    return f"{x * 100:+.0f}%" if signed else f"{x * 100:.0f}%"


def reasons(part: dict, row: dict, report: dict | None = None, top: int = 3) -> list[str]:
    """Why the range is as wide as it is: the groups that widen it most, each said with the stock's own numbers."""
    out = []
    for g, c in sorted(pulls(part, row).items(), key=lambda kv: -kv[1]):
        if c < MIN_PULL or len(out) >= top:
            break
        if g == "swings":
            s = f"moved about {np.exp(row['lv5']) * 100:.1f}% a day over its last five sessions"
            if row["a1"] >= 0.10:
                s += f", {_pct(row['a1'])} against the market on the last one"
            elif row["n10"] >= 5:
                s += f", with {row['n10']:.0f} days of 10%+ in a year"
            out.append(s)
        elif g == "report":
            if row["earn_win"] > 0:
                if report and report.get("out"):
                    out.append(f"results came out on {report['out']} (last year's came later, so the range still "
                               "allows for a report)")
                elif report and report["date"] <= str(row.get("date", "")):
                    out.append(f"results due about now (expected {report['date']}, {report['timing']})")
                else:
                    when = f" (about {report['date']}, {report['timing']})" if report else " (by last year's date)"
                    out.append("results due within eight days" + when)
        elif g == "price":
            price, dv = float(np.exp(row["logp"])), float(np.exp(row["ldv"]))
            if price < 5:
                out.append(f"a ${price:.2f} stock")
            elif row["volsurge"] >= 0.5:
                out.append("its trading volume has jumped this week")
            elif dv < 5e6:
                out.append(f"thinly traded (${dv / 1e6:.1f}M a day)")
        elif g == "run":
            if row["ret20"] <= -0.2:
                out.append(f"down {_pct(1 - np.exp(row['ret20']))} against the market in a month")
            elif row["ret20"] >= 0.2:
                out.append(f"up {_pct(np.exp(row['ret20']) - 1)} against the market in a month")
            elif row["lo52"] <= 0.03:
                out.append("at its 52-week low")
            elif row["hi52"] >= -0.03:
                out.append("at its 52-week high")
            else:
                out.append(f"{_pct(-row['hi52'])} below its 52-week high")
        elif g == "filings":
            out.append("a share-sale filing in the last 30 sessions" if row["offer30"] > 0
                       else "company news filed this week (8-K)")
        elif g == "drug":
            if row["bio"] > 0:
                out.append("a drug or biotech company")
        elif g == "fear":
            out.append(f"market fear is high (VIX {np.exp(row['vix']):.0f})")
    return out


def watch_tickers() -> list[str]:
    p = config.ROOT / "universe" / f"{WATCH}.json"
    return json.loads(p.read_text())["tickers"] if p.exists() else []


def scan_tickers() -> list[str]:
    """What the scan watches: the research universe (W1's fitting and W1a's test), then the newer listings."""
    out = shocks.universe_tickers()
    have = set(out)
    return out + [t for t in watch_tickers() if t not in have]


def watch_status() -> str:
    """How far the ranges are tested on the newer listings: W1b's verdict as the ledger records it."""
    try:
        from brain.research import run
        v = next((r["verdict"] for r in run.verdicts() if r.get("kind") == "range" and r.get("id") == "W1b"), None)
    except Exception:  # noqa: BLE001 - a missing or unreadable ledger says untested
        v = None
    return f"W1b {v.split(':')[0]}" if v else "untested until W1b runs"


def live(bank: Bank, tickers: list[str], log=None) -> dict:
    """The scan at the bank's latest market close: one row per tradable stock, widest range first, and the stocks
    left out with the reason."""
    model = load()
    if model is None:
        raise SystemExit("no fitted model: run python -m brain scan fit")
    s1 = shocks.load()
    rows, skipped, asof = [], [], None
    for c in range(0, len(tickers), CHUNK):
        b = Bank(bank.root)
        v = b.view()
        mk = shocks.market(v)
        if mk is None:
            raise SystemExit(f"no {shocks.BENCH} or {shocks.VIX} prices in the bank: "
                             f"python -m brain ingest {shocks.BENCH} {shocks.VIX}")
        asof = str(mk[0][-1])
        i = len(mk[0]) - 1
        for t in (x.upper() for x in tickers[c:c + CHUNK]):
            got = ticker_frame(v, t, mk)
            if not got:
                skipped.append({"t": t, "why": "too little price history in the bank"})
                continue
            f = got[1]
            if f["why"][i]:
                skipped.append({"t": t, "why": WHY[int(f["why"][i])]})
                continue
            row = {k: float(f[k][i]) for k in FEATS}
            if not all(np.isfinite(x) for x in row.values()):
                skipped.append({"t": t, "why": "too little history for its features"})
                continue
            if row["earn_win"] > 0:                     # is this season's report already in the bank?
                reps = shocks.filings(v, t, mk[0])[0]
                row["last_report"] = reps[-1] if reps else None
            rows.append({"t": t, "date": asof, **row})
        if log:
            log(f"{min(c + CHUNK, len(tickers))}/{len(tickers)} tickers")
        del b, v
        gc.collect()
    if rows:
        X = np.array([[r[k] for k in FEATS] for r in rows], float)
        q = np.expm1(quantiles(model["w1"], X))
        s, tenth = scale(model["w1"], X)
        ch = shocks.chances(s1["s1"], rows) if s1 else [None] * len(rows)
        for k, r in enumerate(rows):
            r.update(lo90=float(q[k, 0]), lo68=float(q[k, 1]), mid=float(q[k, 2]), hi68=float(q[k, 3]),
                     hi90=float(q[k, 4]), half=float((q[k, 3] - q[k, 1]) / 2), sigma=float(s[k]),
                     tenth=int(tenth[k] % BUCKETS), up=float(model["w1"]["up"][tenth[k]]),
                     shock=None if ch[k] is None else float(ch[k]))
    rows.sort(key=lambda r: -r["half"])
    return {"asof": asof, "rows": rows, "skipped": skipped, "w1_sha": model_sha(),
            "s1_sha": shocks.model_sha() if s1 else None}


RECENT_REPORT = 12          # days: a report this recent is this season's, already out


def report_dates(asof: str, tickers: list[str]) -> dict[str, dict]:
    """What the earnings index knows of each ticker's report around ``asof``: its estimate of the next one in the
    eight days after (``date``, ``timing``), or that this season's is already out (``out``). Empty if the index
    is missing."""
    from datetime import date, timedelta

    from brain import earnradar
    try:
        idx = earnradar.load_index()["tickers"]
    except FileNotFoundError:
        return {}
    day = date.fromisoformat(asof)
    out = {}
    for t in tickers:
        row = idx.get(t)
        if not row:
            continue
        reps = row.get("reports") or []
        last = earnradar.ny_date(reps[-1]) if reps else None
        if last and 0 <= (day - last).days <= RECENT_REPORT:
            out[t] = {"out": last.isoformat()}
        elif nr := earnradar.next_report(row, day + timedelta(days=1), 8):
            out[t] = nr
    return out


def report_note(r: dict, known: dict | None) -> dict | None:
    """What to say of a row's report: out already when the bank holds a results 8-K from the last RECENT_REPORT
    days (the row's own filings, then the index), else the index's estimate of the date."""
    from datetime import date
    last = r.get("last_report")
    if last and 0 <= (date.fromisoformat(r["date"]) - date.fromisoformat(last)).days <= RECENT_REPORT:
        return {"out": last}
    return known


def refresh(bank: Bank, tickers: list[str], log=print) -> dict:
    """Re-read what the scan stands on: the market's and each ticker's daily prices, then each company's latest
    filings (SEC's recent page), so a new 8-K or share-sale filing counts. One symbol's outage does not stop it."""
    from brain.sources import prices, sec
    todo = [shocks.BENCH, shocks.VIX, *[t.upper() for t in tickers]]
    bad = {"prices": 0, "filings": 0}
    for k, t in enumerate(todo):
        try:
            bank.add_many(prices.ingest(t, ttl=3600))
        except Exception:  # noqa: BLE001
            bad["prices"] += 1
        if log and (k + 1) % 200 == 0:
            log(f"  prices: {k + 1}/{len(todo)}")
    for k, t in enumerate(todo[2:]):
        try:
            bank.add_many(sec.filings_for(t, all_pages=False, ttl=6 * 3600)[0])
        except Exception:  # noqa: BLE001
            bad["filings"] += 1
        if log and (k + 1) % 200 == 0:
            log(f"  filings: {k + 1}/{len(todo) - 2}")
    return {"symbols": len(todo), "failed": bad}


def line(part: dict, r: dict, report: dict | None = None) -> str:
    why = "; ".join(reasons(part, r, report)) or "nothing unusual: its ordinary swings"
    shock = "" if r["shock"] is None else f"  20%+ day {r['shock'] * 100:4.1f}%"
    return (f"  {r['t']:6} 2 in 3: {r['lo68'] * 100:+6.1f}%..{r['hi68'] * 100:+6.1f}%   9 in 10: "
            f"{r['lo90'] * 100:+6.1f}%..{r['hi90'] * 100:+6.1f}%{shock}   why: {why}")


# ---------- the record ----------
def scans_dir() -> Path:
    return Path(os.environ.get("BRAIN_SCANS") or config.ROOT / "research" / "scans")


OPEN_UTC = "13:30"           # no record after this on the next weekday: before the US open in winter and summer


def opens_after(asof: str) -> str:
    """The moment after which the scan of ``asof`` may no longer be written to: the next weekday at OPEN_UTC."""
    from datetime import date, timedelta
    d = date.fromisoformat(asof) + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return f"{d.isoformat()}T{OPEN_UTC}:00Z"


def record(res: dict, now: str | None = None) -> Path | None:
    """Write the scan for its close: each stock's range (log returns) and shock chance, before any of its five
    sessions has traded. A recorded row is never changed. Before the next session opens, a later run may add the
    stocks the record lacks (``added`` says when); after it, nothing is written. None when nothing was written."""
    from brain.timeutil import now_iso
    now = now or now_iso()
    if now >= opens_after(res["asof"]):
        return None
    p = scans_dir() / f"{res['asof']}.json"
    rows = {r["t"]: [round(float(np.log1p(r[k])), 4) for k in ("lo90", "lo68", "hi68", "hi90")]
            + [None if r["shock"] is None else round(r["shock"], 4)] for r in sorted(res["rows"], key=lambda r: r["t"])}
    watch = set(watch_tickers()) - set(shocks.universe_tickers())
    if p.exists():
        old = json.loads(p.read_text())
        new = {t: v for t, v in rows.items() if t not in old["rows"]}
        if not new:
            return None
        old["rows"].update(new)
        old["added"] = [*old.get("added", []), {"at": now, "tickers": sorted(new), "skipped": len(res["skipped"])}]
        old["newer_listings"] = sorted(set(old.get("newer_listings", [])) | {t for t in new if t in watch})
        p.write_text(canon(old) + "\n")
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(canon({"asof": res["asof"], "made_at": now, "h": H, "w1_sha": res["w1_sha"],
                        "s1_sha": res["s1_sha"], "skipped": len(res["skipped"]), "rows": rows,
                        "newer_listings": sorted(t for t in rows if t in watch)}) + "\n")
    return p


def outcome(view, t: str, mk, i0: int) -> tuple[float, float] | None:
    """A stock's log return over the H sessions after index ``i0`` and whether one of them moved 20%+ against SPY;
    None if the window is not whole."""
    cal, bclose, _ = mk
    df = shocks._series(view, t)
    if df is None or i0 + H >= len(cal):
        return None
    close, adj = (shocks._aligned(df, c, cal) for c in ("close", "adjclose"))
    f = extras(close, adj, _own(df, "volume", cal), np.isfinite(_own(df, "close", cal)))
    if not np.isfinite(f["l5"][i0]):
        return None
    w = slice(i0, i0 + H + 1)
    e = adj[w][1:] / adj[w][:-1] - bclose[w][1:] / bclose[w][:-1]
    return float(f["l5"][i0]), float(np.any(np.abs(e) >= shocks.SHOCK))


def score(bank: Bank, log=None) -> list[dict]:
    """Every recorded scan whose five sessions have closed, judged against the bank's prices: how often the ranges
    held, and the shock chances against what happened."""
    out = []
    v = bank.view()
    mk = shocks.market(v)
    if mk is None:
        return out
    cal = [str(x) for x in mk[0]]
    for p in sorted(scans_dir().glob("????-??-??.json")):       # one file per recorded close
        rec = json.loads(p.read_text())
        if rec["asof"] not in cal or cal.index(rec["asof"]) + H >= len(cal):
            out.append({"asof": rec["asof"], "judged": False, "n": len(rec["rows"])})
            continue
        i0 = cal.index(rec["asof"])
        in68 = in90 = n = 0
        ch, hit = [], []
        for t, (lo90, lo68, hi68, hi90, shock) in rec["rows"].items():
            got = outcome(v, t, mk, i0)
            if got is None:
                continue
            n += 1
            in68 += lo68 <= got[0] <= hi68
            in90 += lo90 <= got[0] <= hi90
            if shock is not None:
                ch.append(shock)
                hit.append(got[1])
        top = np.argsort(-np.array(ch))[:25] if ch else []
        out.append({"asof": rec["asof"], "judged": True, "through": cal[i0 + H], "n": n, "listed": len(rec["rows"]),
                    "cover68": in68 / n if n else float("nan"), "cover90": in90 / n if n else float("nan"),
                    "shock_said": float(np.mean(ch)) if ch else float("nan"),
                    "shock_rate": float(np.mean(hit)) if hit else float("nan"),
                    "top25_said": float(np.mean(np.array(ch)[top])) if len(top) else float("nan"),
                    "top25_rate": float(np.mean(np.array(hit)[top])) if len(top) else float("nan")})
        if log:
            log(f"judged {rec['asof']}")
    return out


# ---------- command line ----------
HEAD = ("What it says: how far each stock is likely to move over the next {h} sessions, and why. What it does not "
        "say: which way. No direction test in this repo has passed, and the middle of every range sits near zero.")


def _cmd(a) -> int:
    if a.action == "replay":
        from brain import scanreplay
        return scanreplay.cmd(a)
    if a.action == "watch":
        from brain import scanwatch
        return scanwatch.cmd(a)
    if a.action == "trader":
        from brain import scantrader
        return scantrader.cmd(a)
    bank = Bank()
    if a.action == "fit":
        fit_and_save(bank)
        return 0
    if a.action == "score":
        rows = score(bank)
        if not rows:
            print("no recorded scans yet: python -m brain scan run --record")
            return 0
        done = [r for r in rows if r["judged"]]
        for r in rows:
            if not r["judged"]:
                print(f"  {r['asof']}: {r['n']} stocks, its five sessions have not closed")
                continue
            print(f"  {r['asof']} to {r['through']}: {r['n']} stocks; the 2-in-3 range held {r['cover68'] * 100:.1f}%, "
                  f"the 9-in-10 range {r['cover90'] * 100:.1f}%; a 20%+ day came to {r['shock_rate'] * 100:.2f}% "
                  f"(said {r['shock_said'] * 100:.2f}%), and to {r['top25_rate'] * 100:.0f}% of the 25 likeliest "
                  f"(said {r['top25_said'] * 100:.0f}%)")
        if done:
            print(f"Scans judged: {len(done)}. Mean: the 2-in-3 range held "
                  f"{np.mean([r['cover68'] for r in done]) * 100:.1f}%, the 9-in-10 range "
                  f"{np.mean([r['cover90'] for r in done]) * 100:.1f}%.")
        print("One week's stocks move together, so a week is one reading: under 30 judged scans this record says "
              "little. The tested evidence is W1a and S1a in research/RESULTS.md.")
        return 0
    if a.refresh:
        r = refresh(bank, a.tickers or scan_tickers(), log=lambda m: print(m, flush=True))
        print(f"re-read {r['symbols']} symbols; not read: {r['failed']['prices']} prices, {r['failed']['filings']} "
              "filing lists")
        bank = Bank()
    named = [t.upper() for t in a.tickers]
    res = live(bank, named or scan_tickers())
    model = load()
    part = model["w1"]
    rows = res["rows"]
    reports = report_dates(res["asof"], [r["t"] for r in rows])
    listed, watch = set(shocks.universe_tickers()), set(watch_tickers())
    newer = sum(r["t"] in watch and r["t"] not in listed for r in rows)
    status = watch_status()
    print(f"Market scan at the close of {res['asof']} (W1 ranges, fitted {model['train'][0]}..{model['train'][1]}; "
          f"S1 shock chances). {len(rows)} tradable stocks"
          + (f", {newer} of them listed after 2010 (ranges {status})" if newer else "")
          + f"; {len(res['skipped'])} left out.")
    print(HEAD.format(h=H))
    if rows:
        ups = np.mean([r["up"] for r in rows])
        print(f"Typical stock here: 2 in 3 within ±{np.median([r['half'] for r in rows]) * 100:.1f}%. In the "
              f"training years stocks like these rose in {ups * 100:.0f}% of weeks, whatever the scan said.")
    shown = rows if named else sorted(rows, key=lambda r: -(r["shock"] or 0))[:a.top] if a.sort == "shock" else rows[:a.top]
    for r in shown:
        mark = ("" if r["t"] in listed else f"  [listed after 2010: ranges {status}]" if r["t"] in watch
                else "  [outside the scan's lists: W1 and S1 were neither fitted nor tested on it]")
        print(line(part, r, report_note(r, reports.get(r["t"]))) + mark)
    if named:
        for s in res["skipped"]:
            print(f"  {s['t']:6} left out: {s['why']}")
        if any(s["why"] == WHY[1] for s in res["skipped"]) and not a.refresh:
            print("  (a price older than the market's latest close: add --refresh to re-read these tickers)")
    elif res["skipped"]:
        why: dict[str, int] = {}
        for s in res["skipped"]:
            why[s["why"]] = why.get(s["why"], 0) + 1
        print("Left out: " + "; ".join(f"{n} {w}" for w, n in sorted(why.items(), key=lambda kv: -kv[1])))
    if a.record:
        p = record(res)
        print(f"recorded {p}; commit it before the next session opens" if p
              else f"nothing written: the scan of {res['asof']} already holds these stocks, or its next session "
                   f"has opened ({opens_after(res['asof'])})")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("scan", help="the market scan: each stock's likely move over five sessions, and why (docs/SCAN.md)")
    p.add_argument("action", choices=["run", "fit", "score", "replay", "watch", "trader"])
    p.add_argument("tickers", nargs="*", help="run: only these tickers, in full (they need prices in the bank)")
    p.add_argument("--top", type=int, default=25, help="run: how many of the widest ranges to list")
    p.add_argument("--sort", choices=["move", "shock"], default="move", help="run: widest range first, or likeliest 20%+ day")
    p.add_argument("--refresh", action="store_true", help="run: re-read the daily prices and latest filings first (about 20 minutes for the whole list)")
    p.add_argument("--record", action="store_true", help="run: write the scan to research/scans/ so it can be scored; "
                   "watch: write the check to research/scans/updates/<close>/")
    p.add_argument("--from", dest="first", default="2024-11-04", help="replay: the first week's end on or after this; "
                   "trader: the first session (never before 2024-11-11, the first with midday prices)")
    p.add_argument("--to", dest="last", default="2026-12-31", help="replay, trader: the last session on or before this")
    p.add_argument("--save", action="store_true", help="replay: write research/scans/replays/replay-<from>_<to>.json; "
                   "trader: write research/scans/replays/trader-<from>_<to>.json")
    p.add_argument("--only-top", dest="only_top", type=int, help="watch: only the N widest ranges and N likeliest 20%%+ days")
    p.add_argument("--every", type=int, help="watch: check again every N minutes (at least 15) until Ctrl+C")
    p.add_argument("--one-trade", dest="one_trade", action="store_true",
                   help="trader: one order a day at most (M5, S3), beside the fixed books and the trader with no limit")
    p.set_defaults(fn=_cmd)
