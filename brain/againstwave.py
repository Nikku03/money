"""Stocks that moved against their wave: a forward record for the declared tests R2a-R2c (docs/RIPPLES.md).

At each close, the ripple map (``ripple.day_map``) shows which stocks felt a wave of at least ``ripple.ABSORB`` usual
days one way and moved ``ripple.AGAINST`` or more the other (``ripple.how_it_moved``: "against its wave"). Each is
recorded with whether it had company news that session (an 8-K or a results report filed at the SEC, as the bank
holds them), and with up to ``N_CONTROLS`` stocks of the same day that moved the same way by about as much (within
``MATCH`` usual days), with the same news status, but with no wave against them. The record is written before the
next session closes; five sessions later the bank's closes say what followed (``outcomes``).

The questions, declared before the first record (``brain/research/againsttest.py``):

- **R2a:** without news, do they bounce back (their next five sessions' own move opposite to the day's)?
- **R2b:** with news, do they keep going?
- **R2c:** does moving against a wave matter: do they bounce back more than the matched stocks that moved alike
  without a wave against them?

A record is a reading, not a forecast; nothing here says which way until the tests are judged.
"""
from __future__ import annotations

import json

import numpy as np

from brain import config, ripple, shocks
from brain.timeutil import now_iso

CALLS = config.ROOT / "research" / "ripple_calls.jsonl"
N_CONTROLS = 10
MATCH = 0.5           # a control's own move is within this many usual days of the stock's
H = ripple.H          # sessions judged


def select(z: np.ndarray, Hh: np.ndarray, stone: np.ndarray, news: np.ndarray) -> tuple[list[int], dict[int, list[int]]]:
    """The stocks that moved against their waves, and for each its matched controls: the same sign of own move,
    within MATCH usual days of its size, the same news status, not a stone, and no wave against them (absorbed or
    the same way). Controls are the closest in size first, then by position (deterministic)."""
    ok = np.isfinite(z) & np.isfinite(Hh)
    against = [j for j in np.flatnonzero(ok & ~stone)
               if ripple.how_it_moved(float(Hh[j]), float(z[j]), False) == "against its wave"]
    aset = set(against)
    no_wave_against = ok & ~stone & ((np.abs(Hh) < ripple.ABSORB) | (np.sign(Hh) == np.sign(z)))
    controls = {}
    for j in against:
        c = np.flatnonzero(no_wave_against & (np.sign(z) == np.sign(z[j])) & (np.abs(np.abs(z) - abs(z[j])) <= MATCH)
                           & (news == news[j]))
        c = [int(x) for x in c if int(x) not in aset]
        c.sort(key=lambda x: (abs(abs(z[x]) - abs(z[j])), x))
        controls[int(j)] = c[:N_CONTROLS]
    return [int(j) for j in against], controls


def news_flags(view, tickers: list[str], cal: np.ndarray, k: int, which) -> np.ndarray:
    """Whether each of ``which`` (indices into ``tickers``) had an 8-K or a results report for session ``k``."""
    out = np.zeros(len(tickers), bool)
    d = str(cal[k])
    for j in which:
        reports, news, offer = shocks.filings(view, tickers[j], cal)
        out[j] = d in reports or k in news
    return out


def recorded_dates(path=None) -> set[str]:
    p = path or CALLS
    if not p.exists():
        return set()
    return {json.loads(ln)["date"] for ln in p.read_text().splitlines() if ln.strip()}


def record(bank, date: str | None = None, tickers: list[str] | None = None, path=None, log=print) -> dict | None:
    """Record the stocks that moved against their waves on ``date`` (default: the bank's latest session). Refused
    for a session that is not the bank's latest (its outcome has begun) or that is already recorded."""
    from brain import scan
    view = bank.view()
    mk = shocks.market(view)
    if mk is None:
        raise SystemExit(f"no {shocks.BENCH} prices in the bank: python -m brain scan run --refresh")
    cal = np.asarray(mk[0]).astype(str)
    date = date or str(cal[-1])
    if date != str(cal[-1]):
        raise SystemExit(f"{date} is not the bank's latest session ({cal[-1]}): a record is made before its outcome")
    if date in recorded_dates(path):
        if log:
            log(f"{date} is already recorded")
        return None
    tickers = sorted(tickers or scan.scan_tickers())
    res = ripple.day_map(bank, date, tickers)
    p = res["pond"]
    z, Hh, stone = res["z"], res["H"], res["stone"]
    k = int(np.searchsorted(cal, date))
    # news is read for the stocks that could be picked: a move of AGAINST - MATCH usual days or more, not a stone
    cand = np.flatnonzero(np.isfinite(z) & ~stone & (np.abs(z) >= ripple.AGAINST - MATCH))
    news = news_flags(view, tickers, cal, k, cand)
    against, controls = select(z, Hh, stone, news)

    def row(j):
        return {"t": tickers[j], "z": round(float(z[j]), 4), "H": round(float(Hh[j]), 4), "s": round(float(p["s"][j]), 6),
                "beta": round(float(p["beta"][j]), 4), "news": bool(news[j]), "pct": round(float(res["e"][j]) * 100, 3)}
    rec = {"date": date, "recorded_at": now_iso(), "stocks": int(np.isfinite(z).sum()),
           "storm": round(ripple.storm(z)["size"], 4),
           "against": [{**row(j), "controls": [[tickers[c], round(float(z[c]), 4), round(float(p["s"][c]), 6),
                                                 round(float(p["beta"][c]), 4)] for c in controls[j]]}
                       for j in against]}
    target = path or CALLS
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    if log:
        n_news = sum(r["news"] for r in rec["against"])
        log(f"recorded {date}: {len(against)} stocks moved against their waves ({n_news} with company news), "
            f"{sum(len(r['controls']) for r in rec['against'])} matched controls; written to {target}")
    return rec


def read(path=None) -> list[dict]:
    p = path or CALLS
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def outcome(r: np.ndarray, m: np.ndarray, beta: float, s: float) -> float:
    """The own move over the H sessions after the record: the sum of daily returns less beta times the market's, in
    usual days of H sessions. NaN when a session is missing."""
    e = r - beta * m
    if len(e) != H or not np.isfinite(e).all() or not (s > 0):
        return float("nan")
    return float(e.sum() / (s * np.sqrt(H)))


def outcomes(bank, recs: list[dict]) -> list[dict]:
    """Each recorded stock's outcome and its controls' mean outcome, for records whose H sessions have closed."""
    view = bank.view()
    mk = shocks.market(view)
    cal = np.asarray(mk[0]).astype(str)
    spy = shocks._aligned(shocks._series(view, shocks.BENCH), "adjclose", cal)
    cache: dict[str, np.ndarray] = {}

    def rets(t: str, k: int) -> np.ndarray:
        if t not in cache:
            df = shocks._series(view, t)
            cache[t] = shocks._aligned(df, "adjclose", cal) if df is not None else np.full(len(cal), np.nan)
        a = cache[t][k:k + H + 1]
        return a[1:] / a[:-1] - 1 if len(a) == H + 1 else np.full(H, np.nan)

    out = []
    for rec in recs:
        k = int(np.searchsorted(cal, rec["date"]))
        if k >= len(cal) or cal[k] != rec["date"] or k + H >= len(cal):
            continue
        sm = spy[k:k + H + 1]
        m = sm[1:] / sm[:-1] - 1
        for a in rec["against"]:
            y = outcome(rets(a["t"], k), m, a["beta"], a["s"])
            ys = [outcome(rets(c[0], k), m, c[3], c[2]) for c in a["controls"]]
            ys = [v for v in ys if np.isfinite(v)]
            out.append({"date": rec["date"], "recorded_at": rec["recorded_at"], "t": a["t"], "news": a["news"],
                        "sign": float(np.sign(a["z"])), "y": y, "y_controls": float(np.mean(ys)) if ys else float("nan"),
                        "n_controls": len(ys), "next_session": str(cal[k + 1])})
    return out


def status(bank, path=None) -> list[str]:
    """How the record stands: sessions and weeks recorded, how many judged. Interim means are shown but say nothing
    until the declared tests are judged (30 weekly readings)."""
    recs = read(path)
    if not recs:
        return ["no records yet: python -m brain ripple record (after each close)"]
    rows = outcomes(bank, recs)
    weeks = {np.datetime64(r["date"], "W") for r in recs}
    jw = {np.datetime64(r["date"], "W") for r in rows}
    out = [f"{len(recs)} sessions recorded ({min(r['date'] for r in recs)} to {max(r['date'] for r in recs)}), "
           f"{len(weeks)} weeks; {len(rows)} stocks judged in {len(jw)} weeks. The declared tests (R2a-R2c) are judged "
           f"once at 30 weekly readings; before that these numbers say nothing."]
    for lab, sel, sgn in (("without news, bounce back", lambda r: not r["news"], -1),
                          ("with news, keep going", lambda r: r["news"], 1)):
        v = [sgn * r["sign"] * r["y"] for r in rows if sel(r) and np.isfinite(r["y"])]
        if v:
            out.append(f"  {lab}: {len(v)} stocks, mean {np.mean(v):+.3f} usual days of five sessions")
    d = [(-r["sign"] * r["y"]) - (-r["sign"] * r["y_controls"]) for r in rows
         if np.isfinite(r["y"]) and np.isfinite(r["y_controls"])]
    if d:
        out.append(f"  bounce back beyond matched controls: {len(d)} stocks, mean {np.mean(d):+.3f}")
    return out
