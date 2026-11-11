"""The earnings-aware shock radar (S3): S1 plus what a trader knows about a stock's next report (docs/SHOCKS.md,
"Catalysts").

S1 (brain/shocks.py) ranks every stock by its chance of a 20%+ session against SPY in the next five. Over 2011-2023
(research/CATALYSTS-2026-11.md) most such sessions came with nothing filed at the SEC that session (73-79%). Results
reports were the one large catalyst known in advance (12-17%), and S1 had only a third of those sessions in its
week's top 5%. Deals, distress filings, executive changes, press releases, merger filings, activist stakes, reverse
splits and lockups added nothing to S1 over 2020-2023. S3 adds only what is known about the next report:

- ``earnq``: a report is expected 1-8 days ahead, a quarter (91 days) after the last one. S1's ``earn_win`` moves
  last year's report on 52 weeks; a quarter's cadence also catches companies whose dates moved or that are new;
- ``ehab_known`` and ``ehab_l``: the stock's earnings habit, the median move against SPY on its last eight report
  sessions (at least two), as a log;
- ``ehab_win``: that habit when a report is expected (``earn_win`` or ``earnq``), else 0.

Everything is known at the week's last close; a report's move counts once its session has closed. The model is S1's
logistic regression with these four features added, fitted on the same stock-weeks, 2011-2023
(``knowledge/shocks_s3.json``). Its declared test, S3a, reads 2024-2026 (``brain/research/catalysttest.py``).

A chance of a shock is a size, not a direction. Nothing here trades.
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

ADDED = ("earnq", "ehab_known", "ehab_l", "ehab_win")
FEATS3 = (*shocks.FEATS, *ADDED)
HABIT_N, HABIT_MIN = 8, 2   # the last eight report sessions; at least two before the habit counts
QUARTER = 91                # days from one report to the next
AHEAD = 8                   # a report expected within this many days is due
DEFAULT_MODEL = config.ROOT / "knowledge" / "shocks_s3.json"


def excess(adj: np.ndarray, bclose: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Each session's return over SPY's, with data errors set to NaN (the rule of ``shocks.frame``)."""
    with np.errstate(all="ignore"):
        r = np.r_[np.nan, adj[1:] / adj[:-1] - 1]
        rb = np.r_[np.nan, bclose[1:] / bclose[:-1] - 1]
    bad = (r > shocks.BAD_UP) | (r < shocks.BAD_DOWN) | (close < shocks.MIN_PRICE)
    return np.where(bad, np.nan, r - rb)


def report_features(dates: np.ndarray, e: np.ndarray, report_idx, earn_win: np.ndarray) -> dict[str, np.ndarray]:
    """S3's added features for every session, from the report sessions (the session at whose close each results
    report was first known, which is the session it moved) and the excess returns ``e``. Also ``ehab`` (the habit,
    NaN until known) and ``due`` (a report expected within AHEAD days), for reading."""
    n = len(dates)
    d = np.asarray(dates).astype("datetime64[D]")
    rk = np.array(sorted({int(k) for k in report_idx if 0 <= int(k) < n}), int)
    earnq, hab, cnt = np.zeros(n), np.full(n, np.nan), np.zeros(n)
    if len(rk):
        pos = np.searchsorted(rk, np.arange(n), side="right") - 1
        ok = pos >= 0
        exp = d[rk[np.maximum(pos, 0)]] + np.timedelta64(QUARTER, "D")
        earnq = (ok & (exp > d) & (exp <= d + np.timedelta64(AHEAD, "D"))).astype(float)
        seen: list[float] = []
        for j, k in enumerate(rk):
            if np.isfinite(e[k]):
                seen.append(abs(float(e[k])))
            nxt = rk[j + 1] if j + 1 < len(rk) else n
            if seen:
                hab[k:nxt] = float(np.median(seen[-HABIT_N:]))
                cnt[k:nxt] = min(len(seen), HABIT_N)
    known = (cnt >= HABIT_MIN) & np.isfinite(hab)
    due = np.maximum(np.nan_to_num(earn_win), earnq) > 0
    return {"earnq": earnq, "ehab_known": known.astype(float),
            "ehab_l": np.where(known, np.log(np.maximum(np.nan_to_num(hab, nan=1e-3), 1e-3)), 0.0),
            "ehab_win": np.where(known & due, np.nan_to_num(hab), 0.0),
            "ehab": np.where(known, hab, np.nan), "due": due.astype(float)}


def ticker_frame(view, t: str, mk) -> tuple[np.ndarray, dict] | None:
    """S1's features and outcome for every session of ``t`` (``shocks.ticker_frame``), with S3's added."""
    got = shocks.ticker_frame(view, t, mk)
    if not got:
        return None
    cal, f = got
    df = shocks._series(view, t)
    close, adj = shocks._aligned(df, "close", cal), shocks._aligned(df, "adjclose", cal)
    reports, _, _ = shocks.filings(view, t, cal)
    idx = np.searchsorted(cal, np.array(reports, dtype=cal.dtype)) if reports else []
    f.update(report_features(cal, excess(adj, mk[1], close), idx, f["earn_win"]))
    return cal, f


def rows_at(f: dict[str, np.ndarray], dates: np.ndarray, idx, ticker: str, outcome: bool = True) -> list[dict]:
    """Rows at the sessions ``idx`` whose S3 features are all finite (and, with ``outcome``, whose window closed)."""
    out = []
    for i in idx:
        row = {k: float(f[k][i]) for k in FEATS3}
        if not all(np.isfinite(v) for v in row.values()):
            continue
        if outcome:
            if not np.isfinite(f["y"][i]):
                continue
            row.update(y=float(f["y"][i]), up=float(f["up"][i]), down=float(f["down"][i]))
        row["due"] = float(f["due"][i])
        row["ehab"] = float(f["ehab"][i]) if np.isfinite(f["ehab"][i]) else None
        out.append({"t": ticker, "date": str(dates[i]), **row})
    return out


def dataset(bank: Bank, tickers: list[str], first: str, last: str, log=print) -> list[dict]:
    """Weekly rows (each week's last session) in [first, last] whose five-session outcome is known."""
    out = []
    for c in range(0, len(tickers), shocks.CHUNK):
        b = Bank(bank.root)
        v = b.view()
        mk = shocks.market(v)
        if mk is None:
            raise SystemExit(f"no {shocks.BENCH} or {shocks.VIX} prices in the bank")
        ends = shocks.week_ends(mk[0])
        ends = ends[(mk[0][ends] >= first) & (mk[0][ends] <= last)]
        for t in tickers[c:c + shocks.CHUNK]:
            got = ticker_frame(v, t, mk)
            if got:
                out += rows_at(got[1], got[0], ends, t)
        if log:
            log(f"{min(c + shocks.CHUNK, len(tickers))}/{len(tickers)} tickers, {len(out)} stock-weeks")
        del b, v
        gc.collect()
    return out


def model_path() -> Path:
    """The fitted S3 file: ``knowledge/shocks_s3.json``, or BRAIN_SHOCKS3_MODEL (tests, trials)."""
    return Path(os.environ.get("BRAIN_SHOCKS3_MODEL") or DEFAULT_MODEL)


def load(path: Path | None = None) -> dict | None:
    p = Path(path or model_path())
    return json.loads(p.read_text()) if p.exists() else None


def model_sha(path: Path | None = None) -> str:
    return hashlib.sha256(Path(path or model_path()).read_bytes()).hexdigest()[:16]


def chances(model: dict, rows: list[dict]) -> np.ndarray:
    return shocks.chances(model["s3"], rows)


def fit_and_save(bank: Bank, path: Path | None = None, log=print, tickers=None) -> dict:
    """Fit S3 on S1's stock-weeks of TRAIN_START..TRAIN_END and save it. Nothing after TRAIN_END is read."""
    rows = dataset(bank, tickers or shocks.universe_tickers(), shocks.TRAIN_START, shocks.TRAIN_END, log=log)
    if not rows:
        raise SystemExit("no stock-weeks to fit on")
    due = [r for r in rows if r["due"] > 0]
    out = {"name": "S3", "train": [shocks.TRAIN_START, shocks.TRAIN_END], "universes": list(shocks.UNIVERSES),
           "h": shocks.H, "shock": shocks.SHOCK, "added": list(ADDED), "s3": shocks.fit(rows, FEATS3),
           "due_rows": len(due), "due_rate": float(np.mean([r["y"] for r in due])) if due else None}
    p = Path(path or model_path())
    p.write_text(canon(out) + "\n")
    if log:
        log(f"fitted S3 on {len(rows)} stock-weeks ({shocks.TRAIN_START}..{shocks.TRAIN_END}); {len(due)} with a "
            f"report expected (shock rate {out['due_rate'] * 100:.2f}% a week there); saved {p}")
    return out


def _cmd(a) -> int:
    if a.action == "fit":
        fit_and_save(Bank())
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("catalysts", help="the earnings-aware shock radar S3 (docs/SHOCKS.md, Catalysts)")
    p.add_argument("action", choices=["fit"])
    p.set_defaults(fn=_cmd)
