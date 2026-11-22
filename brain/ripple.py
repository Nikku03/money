"""Ripples in the pond: how far a piece of news travels through the market, and where waves meet (docs/RIPPLES.md).

The person's picture: the market is a pond and every piece of news is a stone. A big stone makes a high wave. The
wave travels, hits what is around it, and when two waves meet they add (constructive) or cancel (destructive).

Made concrete on the bank's daily closes:

- **The pond** is a year of daily returns with the market's part taken out (``connect.residuals``), each in its own
  usual days (``s``). How well two stocks' returns moved together over that year (``connect.correlations``, ``C``) is
  the water between them. A stock with no link of at least ``EDGE`` to any of the day's stones is on dry land: no wave
  reaches it.
- **A stone** is a stock whose own move (market removed) is at least ``STONE_Z`` of its usual days. Its push is capped
  at ``CAP`` usual days: past that, a move is the stone's own news, which its partners did not share in the year.
- **The waves are read together.** Stones that fall on the same day often share one cause (Friday's rate fall lifted
  housing, mortgage lenders and retailers at once), and adding each stone's wave alone would count that cause once per
  stone. So the wave at stock j is what the year of returns says j moves when the day's stones move as they did,
  all of them at once: the reading ``sum_i b[j, i] * z[i]`` with ``b[j, :] = C[j, S] (C[S, S] + RIDGE I)^-1`` (the
  regression of j on the stones, in usual days, shrunk by ``RIDGE``). A year's correlations overstate how much a
  stock follows a stone on the stone's day (big moves are mostly the stone's own news), so the wave's height is
  ``H[j] = KAPPA * reading``: KAPPA is how much of the reading stocks moved the same day over 2012-2016. ``H`` is in
  j's own usual days: what j has to absorb. Each stone's part is ``KAPPA * b[j, i] * z[i]``.
- **Energy** ``W[j] = sum_i |b[j, i] * z[i]|`` is how much water reached j, whatever the signs.
- **Interference** ``I[j] = |H[j]| / W[j]``: 1 when every stone's part pushed the same way (constructive), near 0 when
  they cancelled (destructive).
- **Rings:** a stock is in a stone's first ring when it is one of its partners (correlation ``RING`` or more), in the
  second when it is a partner of a partner, beyond otherwise.
- **Withstanding:** a wave under a quarter of a usual day is absorbed (lost in the stock's ordinary noise); a quarter
  to one is felt; one or more is a hit.

What a ripple says on the day it happens is a description of how the market moved together, not a forecast. Whether
the waves still travel after the day (how far and how high over the next five sessions, and which way) is what the
declared tests R1b, R1c and R1d ask (``brain/research/rippletest.py``). Until they pass, a ripple map explains; it
does not predict.
"""
from __future__ import annotations

import warnings

import numpy as np

from brain import connect

LOOKBACK = connect.LOOKBACK     # sessions behind the pond
STONE_Z = 3.0                   # a stone: a move (market removed) of at least this many usual days
EDGE = 0.20                     # links weaker than this carry no wave
RING = connect.MIN_CORR         # a partner: the first ring around a stone
ABSORB, HIT = 0.25, 1.0         # wave height (usual days): under ABSORB absorbed, from HIT a hit
CAP = 3.0                       # a stone pushes with at most this many usual days (the links were measured so)
RIDGE = 0.25                    # shrinkage of the joint reading of the stones
KAPPA = 0.24                    # how much of the joint reading a stock moved the same day, on average: the slope of
                                # its own move on the reading over 2012-2016 (866,547 stock-days, inside TRAIN); the
                                # wave's height is the reading times KAPPA, so it is in usual days that did happen
H = 5                           # sessions ahead in the tests


# ---------- the pond ----------
def pond(R: np.ndarray, m: np.ndarray, edge: float = EDGE) -> dict:
    """The pond from a window of daily returns ``R`` (sessions x stocks) and the market's ``m``: each stock's beta
    to the market and usual day (market removed), and the links between stocks (``C``; ``wet``: links of at least
    ``edge``)."""
    E, beta = connect.residuals(R, m)
    with np.errstate(all="ignore"):
        s = np.nanstd(E, axis=0)
    s[~np.isfinite(s) | (s <= 0)] = np.nan
    C = connect.correlations(E)
    C = np.where(np.isfinite(C), C, 0.0).astype(np.float32)
    ok = np.isfinite(s) & np.isfinite(beta)
    C[~ok, :] = 0.0
    C[:, ~ok] = 0.0
    return {"beta": np.where(ok, beta, np.nan), "s": np.where(ok, s, np.nan), "C": C, "wet": np.abs(C) >= edge}


def own_moves(p: dict, r: np.ndarray, m: float) -> np.ndarray:
    """Each stock's own move on a day: its return less its beta times the market's."""
    return r - p["beta"] * m


def stones(p: dict, e: np.ndarray, stone_z: float = STONE_Z) -> np.ndarray:
    """The stocks whose own move is at least ``stone_z`` of their usual days."""
    with np.errstate(all="ignore"):
        z = e / p["s"]
    return np.flatnonzero(np.isfinite(z) & (np.abs(z) >= stone_z))


def joint(p: dict, S: np.ndarray) -> np.ndarray:
    """``b``: each stock's regression on the stones ``S`` jointly (stocks x stones, in usual days), zero for a stock
    on dry land and for a stone's own row."""
    if not len(S):
        return np.zeros((len(p["s"]), 0), np.float32)
    CSS = p["C"][np.ix_(S, S)].astype(float)
    np.fill_diagonal(CSS, 1.0)
    M = np.linalg.inv(CSS + RIDGE * np.eye(len(S)))
    b = (p["C"][:, S].astype(float) @ M).astype(np.float32)
    b[~p["wet"][:, S].any(1)] = 0.0
    b[S] = 0.0
    return b


def waves(p: dict, z: np.ndarray, S: np.ndarray) -> dict:
    """The waves from the stones ``S`` (their own moves ``z[S]`` in usual days) at every stock: the wave's height
    ``H`` and energy ``W`` in each stock's usual days, the interference ``I``, the ring each stock is in, and ``b``
    (each stone's part per unit of its move)."""
    n = len(z)
    stone = np.zeros(n, bool)
    stone[S] = True
    if not len(S):
        zero = np.zeros(n)
        return {"H": zero, "W": zero, "I": np.full(n, np.nan), "ring": np.full(n, 3), "stone": stone,
                "b": np.zeros((n, 0), np.float32)}
    zS = np.clip(np.nan_to_num(z[S]), -CAP, CAP).astype(np.float32)
    b = joint(p, S)
    Hh = KAPPA * (b @ zS)
    W = KAPPA * (np.abs(b) @ np.abs(zS))
    with np.errstate(all="ignore"):
        I = np.where(W > 0, np.abs(Hh) / W, np.nan)
    P = np.abs(p["C"]) >= RING
    r1 = P[:, S].any(1)
    r2 = (P.astype(np.float32) @ r1.astype(np.float32)) > 0
    ring = np.where(r1, 1, np.where(r2, 2, 3))
    ring[stone] = 0
    return {"H": Hh, "W": W, "I": I, "ring": ring, "stone": stone, "b": b}


def contributions(w: dict, z: np.ndarray, S: np.ndarray, j: int, k: int = 4) -> list[tuple[int, float]]:
    """The ``k`` stones whose parts of stock ``j``'s wave are largest, each in j's usual days (signed)."""
    if not len(S):
        return []
    a = KAPPA * w["b"][j] * np.clip(np.nan_to_num(z[S]), -CAP, CAP)
    o = np.argsort(-np.abs(a))[:k]
    return [(int(S[i]), float(a[i])) for i in o if a[i] != 0]


def withstand(h: float) -> str:
    """How a stock takes a wave of height ``h`` (its usual days)."""
    a = abs(h)
    return "absorbed" if a < ABSORB else "felt" if a < HIT else "a hit"


# ---------- history, for the tests ----------
def prices(bank, tickers: list[str], log=None) -> dict:
    """Daily returns of ``tickers`` and of SPY on SPY's calendar from 2010 (data errors removed)."""
    from brain import shocks
    v = bank.view()
    mk = shocks.market(v)
    if mk is None:
        raise SystemExit(f"no {shocks.BENCH} prices in the bank: python -m brain ingest {shocks.BENCH}")
    cal, bclose, _ = mk
    A = np.full((len(cal), len(tickers)), np.nan)
    Cl = np.full((len(cal), len(tickers)), np.nan)
    for j, t in enumerate(tickers):
        df = shocks._series(v, t)
        if df is not None and len(df) > 60:
            A[:, j], Cl[:, j] = shocks._aligned(df, "adjclose", cal), shocks._aligned(df, "close", cal)
        if log and (j + 1) % 500 == 0:
            log(f"{j + 1}/{len(tickers)} price series")
    with np.errstate(all="ignore"):
        R = np.vstack([np.full((1, len(tickers)), np.nan), A[1:] / A[:-1] - 1])
        R[(R > shocks.BAD_UP) | (R < shocks.BAD_DOWN) | (Cl < shocks.MIN_PRICE)] = np.nan
        m = np.r_[np.nan, bclose[1:] / bclose[:-1] - 1]
    return {"cal": np.asarray(cal).astype(str), "tickers": list(tickers), "R": R, "m": m}


REBUILD = 21   # sessions between ponds: each pond is built from the LOOKBACK sessions before its first day


def days(px: dict, first: str, last: str, *, stone_z: float = STONE_Z, edge: float = EDGE, h: int = H,
         log=None):
    """Every session from ``first`` to ``last`` whose ``h`` sessions ahead have closed, with its waves and what
    followed. Yields one dict per session: its date, the stones, and per stock: the wave's height ``H``, energy ``W``,
    interference ``I``, ring, the stock's own move that day in usual days ``z``, its own recent moves (``own5``,
    ``own60``: root mean square of its usual-day moves over 5 and 60 sessions), the market's recent size ``mkt5``, and
    ``Y``: its own move over the next ``h`` sessions in usual days of h sessions (sum of own moves / (s * sqrt(h)))."""
    cal, R, m = px["cal"], px["R"], px["m"]
    idx = np.flatnonzero((cal >= first) & (cal <= last))
    idx = idx[idx + h < len(cal)]
    idx = idx[idx >= LOOKBACK + 60]
    if not len(idx):
        return
    built_at, p, Z = None, None, None
    for t in idx:
        if built_at is None or t - built_at >= REBUILD:
            built_at = t
            p = pond(R[t - LOOKBACK:t], m[t - LOOKBACK:t], edge)
            with np.errstate(all="ignore"):
                lo = max(1, t - 60)
                Z = (R[lo:min(len(cal), t + REBUILD + h + 1)] - p["beta"] * m[lo:min(len(cal), t + REBUILD + h + 1), None]) / p["s"]
            z0 = lo
            sm = np.nanstd(m[t - LOOKBACK:t])
            if log:
                log(f"pond for {cal[t]}: {int(np.isfinite(p['s']).sum())} stocks, "
                    f"{int((p['C'] != 0).sum() // 2)} links")
        k = t - z0
        z = Z[k]
        e = z * p["s"]
        S = stones(p, e, stone_z)
        w = waves(p, z, S)
        w.pop("b")
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            own5 = np.sqrt(np.nanmean(Z[k - 4:k + 1] ** 2, axis=0))
            own60 = np.sqrt(np.nanmean(Z[max(0, k - 59):k + 1] ** 2, axis=0))
            fut = Z[k + 1:k + 1 + h]
            Y = np.where(np.isfinite(fut).sum(0) == h, np.nansum(fut, axis=0) / np.sqrt(h), np.nan)
            mkt5 = float(np.sqrt(np.nanmean(m[t - 4:t + 1] ** 2)) / sm) if sm > 0 else np.nan
        yield {"date": str(cal[t]), "S": S, "z": z, "Y": Y, "own5": own5, "own60": own60, "mkt5": mkt5, **w}


# ---------- the live map ----------
def live(bank, asof: str, moves: dict[str, float], market_move: float, tickers: list[str] | None = None) -> dict:
    """The waves on the check's moves since the close of ``asof`` (``moves``: ticker -> move, ``market_move``:
    SPY's), on the pond of the year before."""
    tickers = sorted(t for t in (tickers or moves) if moves.get(t) is not None)
    b = connect.cached_book(bank, tickers, asof)
    p = pond(b["R"], b["m"])
    r = np.array([moves[t] for t in tickers], float)
    e = own_moves(p, r, market_move)
    S = stones(p, e)
    with np.errstate(all="ignore"):
        z = e / p["s"]
    w = waves(p, z, S)
    return {"asof": asof, "tickers": tickers, "pond": p, "e": e, "z": z, "S": S, **w}


def summary(res: dict, k: int = 25) -> list[str]:
    """The live map in words: the stones, then the stocks the waves reach highest, with who sent them."""
    t, p, e, S = res["tickers"], res["pond"], res["e"], res["S"]
    with np.errstate(all="ignore"):
        z = e / p["s"]
    out = [f"Ripples on the check against the close of {res['asof']}: {len(S)} stones (own moves of {STONE_Z:g}+ usual "
           f"days, market removed) among {int(np.isfinite(z).sum())} stocks."]
    for i in sorted(S, key=lambda i: -abs(z[i]))[:k]:
        out.append(f"  stone {t[i]:6s} own move {e[i] * 100:+.1f}% ({z[i]:+.1f} usual days)")
    hit = np.flatnonzero(~res["stone"] & (res["W"] > 0))
    order = hit[np.argsort(-np.abs(res["H"][hit]))][:k]
    out.append("Where the waves reach highest (height and energy in the stock's own usual days; interference 1 = all "
               "waves the same way, 0 = they cancel):")
    for j in order:
        c = contributions(res, z, S, j)
        moved = z[j]
        out.append(f"  {t[j]:6s} wave {res['H'][j]:+.1f} ({withstand(res['H'][j])}), energy {res['W'][j]:.1f}, "
                   f"interference {res['I'][j]:.2f}, ring {res['ring'][j]}; its own move {moved:+.1f}; from "
                   + ", ".join(f"{t[i]} {a:+.1f}" for i, a in c))
    out.append("A wave on the day describes how stocks moved together; whether waves keep travelling after the day is "
               "what R1b-R1d test (docs/RIPPLES.md).")
    return out


def day_map(bank, date: str, tickers: list[str]) -> dict:
    """The waves on one session's close-to-close moves (``date``), on the pond of the year before it."""
    b = connect.book(bank, tickers, date, LOOKBACK + 1)
    if str(b["cal"][-1]) != date:
        raise SystemExit(f"{date} is not a session in the bank (its last session up to then is {b['cal'][-1]})")
    p = pond(b["R"][:-1], b["m"][:-1])
    e = own_moves(p, b["R"][-1], float(b["m"][-1]))
    S = stones(p, e)
    with np.errstate(all="ignore"):
        z = e / p["s"]
    return {"asof": f"the session of {date} (close to close)", "tickers": list(tickers), "pond": p, "e": e, "z": z,
            "S": S, **waves(p, z, S)}


def _cmd(a) -> int:
    import json

    from brain.bank import Bank
    from brain.explorer import latest_case, latest_scan
    sc = latest_scan()
    if sc is None:
        raise SystemExit("no recorded scan: python -m brain scan run --refresh --record")
    case = latest_case(sc[1]["asof"])
    if case is None:
        raise SystemExit("no case file for the latest scan: python -m brain desk build")
    moves = {t: v for t, v in (case.get("moves") or {}).items() if v is not None}
    bank = Bank()
    if a.date:
        res = day_map(bank, a.date, sorted(moves))
    else:
        spy = next((m["move"] for m in case.get("market") or [] if m.get("symbol") == "SPY"), None)
        if spy is None:
            raise SystemExit("the case file has no SPY move")
        res = live(bank, sc[1]["asof"], moves, float(spy))
        res["asof"] = f"{sc[1]['asof']} (the check {case.get('check')})"
    for line in summary(res, a.top):
        print(line)
    if a.json:
        t = res["tickers"]
        rows = [{"t": t[j], "wave": float(res["H"][j]), "energy": float(res["W"][j]),
                 "interference": None if not np.isfinite(res["I"][j]) else float(res["I"][j]),
                 "ring": int(res["ring"][j]), "stone": bool(res["stone"][j])} for j in range(len(t))]
        with open(a.json, "w") as f:
            json.dump({"asof": res["asof"], "rows": rows}, f)
        print(f"wrote {a.json}")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("ripple", help="the waves news sends through stocks that move together (docs/RIPPLES.md)")
    s = p.add_subparsers(dest="ripple_cmd", required=True)
    q = s.add_parser("map", help="stones and the waves they send, on the latest check or one session's closes")
    q.add_argument("--date", help="a session's close-to-close moves instead of the latest check (YYYY-MM-DD)")
    q.add_argument("--top", type=int, default=25)
    q.add_argument("--json", help="write every stock's wave here")
    q.set_defaults(fn=_cmd)
