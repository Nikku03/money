"""The moonshots that happened: which it caught, which it missed, and why each one happened.

A moonshot here is P4's target: a tradable stock bought at a week's first open that reached +20%
before -10% within 10 sessions. Every one of them, caught or not, is asked three questions.

Did the trader see it?
    Its rank that morning among the tradable stocks (by the score the trader used), whether it was
    in the top 3 but under the bar (score <= 0, "seen but not trusted"), and the features that set it
    apart from what the trader picked.

Why did it happen? Every factor at once, from the prices and filings of the trade's window:
    shape        one jump (one session or one overnight gap made 60%+ of the move) or a steady climb
                 (no step over 40%);
    overnight    the share of the move made in gaps between a close and the next open (news lands
                 outside trading hours);
    news         the company's SEC filings that became public in the window, and on the jump itself;
                 a jump with no filing is marked "news not in its filings": the cause is outside what
                 this system can see;
    industry     how many of its industry's tradable stocks (2-digit SEC code) rose 10%+ over the same
                 window: a wave lifts many boats;
    market       SPY's own move over the window;
    volume       trading volume in the window over its prior 60-session average (a rush in, or a squeeze);
    foreseeable  whether earnings were already expected in the window (earn_win at the open);
    influences   the market's, oil's, bonds', the dollar's and the stock's own parts (moonshot.explain).
The primary cause is the first that fits, in this order: earnings jump, deal jump, other filed-news
jump, jump with no filing, industry wave, market wave, steady climb, mixed.

Is it getting better at seeing them?
    Year by year, how well its scores separate the week's moonshots from the rest (the AUC: 0.5 is a
    coin, 1 is perfect), and how many fell in its top 3, top 10 and top tenth.

This reads what happened after each open, so it is hindsight: it explains, it does not predict.
Anything it suggests the trader should look at is a new feature for a new declared test, and only a
forward (live) test of it would be clean, since the suggestion came from these same years.
"""
from __future__ import annotations

import collections
import math

import numpy as np
import pandas as pd

from brain.paper import moonshot as m1
from brain.paper import moonshot2 as m2

CAUSES = ("earnings jump", "deal jump", "other filed-news jump", "jump with no filing (news not in its filings)",
          "industry wave", "market wave", "steady climb", "mixed")


def anatomy(o: np.ndarray, c: np.ndarray, d: int, ex: int) -> dict:
    """Split the move from session d's open to session ex's close into overnight gaps and sessions."""
    steps = [("session", d, math.log(c[d] / o[d]))]
    for k in range(d + 1, ex + 1):
        steps.append(("gap", k, math.log(o[k] / c[k - 1])))
        steps.append(("session", k, math.log(c[k] / o[k])))
    total = math.log(c[ex] / o[d])
    big = max(steps, key=lambda s: s[2])
    gaps = sum(s[2] for s in steps if s[0] == "gap")
    share = big[2] / total if total > 0 else 0.0
    return {"total": total, "jump_share": share, "jump_kind": big[0], "jump_at": big[1],
            "overnight_share": gaps / total if total > 0 else 0.0,
            "shape": "one jump" if share >= 0.6 else ("steady climb" if share < 0.4 else "mixed")}


def classify(a: dict, filed_on_jump: set, filed: set, wave: float, peers: int, market: float, own_share: float) -> str:
    if a["shape"] == "one jump":
        if "sec_results" in filed_on_jump:
            return "earnings jump"
        if filed_on_jump & {"sec_deal", "sec_activist"}:
            return "deal jump"
        if filed_on_jump:
            return "other filed-news jump"
        return "jump with no filing (news not in its filings)"
    if peers >= 3 and wave >= 0.3:
        return "industry wave"
    if market >= 0.03 and own_share < 0.5:
        return "market wave"
    if a["shape"] == "steady climb":
        return "steady climb"
    return "mixed"


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def review(raw: dict, sec: dict, rec: dict) -> dict:
    """``rec``: moonshot2.simulate(..., record_all=True)'s every/names."""
    P = m1.panels(raw)
    dates = P["close"].index.values
    names = rec["names"]
    events = m1.filing_events(sec, dates)
    sic2 = np.array([str(sec.get("sic", {}).get(t) or "")[:2] for t in names])
    o, c, v = (P[k][names].values for k in ("open", "close", "vol"))
    spy_o, spy_c = P["open"]["SPY"].values, P["close"]["SPY"].values
    rows, yearly = [], collections.defaultdict(lambda: {"y": [], "s": [], "top3": 0, "top10": 0, "dec": 0, "n": 0, "caught": 0})
    zsum = collections.defaultdict(lambda: np.zeros(len(m2.FEATURES)))
    zcnt = collections.Counter()
    for wk in rec["every"]:
        d, ok, score, picked, X = wk["d"], wk["ok"], wk["score"], wk["picked"], wk["X"]
        O = m1.outcomes(P, names, d)
        hit = (O["hit"].values > 0) & ok
        sc = np.where(ok, score, np.nan)
        order = np.argsort(-np.nan_to_num(sc, nan=-np.inf))
        rank = np.empty(len(names), int)
        rank[order] = np.arange(1, len(names) + 1)
        n_ok = int(ok.sum())
        yr = str(dates[d])[:4]
        Y = yearly[yr]
        Y["y"] += list(hit[ok].astype(int))
        Y["s"] += list(np.nan_to_num(sc[ok], nan=-1e9))
        Z = (X[ok] - X[ok].mean(0)) / (X[ok].std(0) + 1e-9)
        Zfull = np.zeros_like(X)
        Zfull[ok] = Z
        for g, mask in (("all", ok), ("picked", picked & ok), ("missed", hit & ~picked), ("caught", hit & picked)):
            zsum[g] += Zfull[mask].sum(0)
            zcnt[g] += int(mask.sum())
        for j in np.flatnonzero(hit):
            t = names[j]
            ex = int(O["exit"].iloc[j])
            a = anatomy(o[:, j], c[:, j], d, ex)
            filed, on_jump = set(), set()
            for k, labels in events.get(t, []):
                if d < k <= ex + 1:
                    filed |= labels
                # a gap into session j: news public by j's open; a session j: news by j's open or during it
                # (public at the next open)
                on = (k == a["jump_at"]) if a["jump_kind"] == "gap" else (a["jump_at"] <= k <= a["jump_at"] + 1)
                if on:
                    on_jump |= labels
            same = (sic2 == sic2[j]) & ok & (np.arange(len(names)) != j) & (sic2 != "")
            pm = np.log(c[ex][same] / o[d][same])
            wave = float((pm >= math.log(1.10)).mean()) if same.sum() else 0.0
            mkt = math.log(spy_c[ex] / spy_o[d])
            vol = float(np.nanmean(v[d:ex + 1, j]) / (np.nanmean(v[d - 60:d, j]) + 1))
            why = m1.explain(P, t, d, ex, events)
            own = abs(why["parts"]["the stock itself"]) / (sum(abs(x) for x in why["parts"].values()) + 1e-12)
            cause = classify(a, on_jump, filed, wave, int(same.sum()), mkt, own)
            tags = [a["shape"]]
            if a["overnight_share"] >= 0.6:
                tags.append("mostly overnight")
            if int(same.sum()) >= 3 and wave >= 0.3:
                tags.append(f"industry wave ({wave:.0%} of {int(same.sum())} peers up 10%+)")
            if mkt >= 0.03:
                tags.append(f"market up {mkt:+.1%}")
            if vol >= 3:
                tags.append(f"volume x{vol:.0f}")
            if X[j, m2.FEATURES.index("earn_win")] > 0:
                tags.append("earnings were expected")
            rows.append({"date": str(dates[d]), "ticker": t, "caught": bool(picked[j]), "rank": int(rank[j]), "of": n_ok,
                         "score": float(score[j]), "seen_not_trusted": bool(rank[j] <= m1.TOP and not picked[j]),
                         "move": a["total"], "cause": cause, "tags": tags, "filed": sorted(filed),
                         "filed_on_jump": sorted(on_jump), "overnight_share": a["overnight_share"], "industry_wave": wave,
                         "market": mkt, "volume_x": vol, "own_share": own, "jump_day": a["jump_at"] - d + 1})
            Y["n"] += 1
            Y["caught"] += int(picked[j])
            Y["top3"] += int(rank[j] <= 3)
            Y["top10"] += int(rank[j] <= 10)
            Y["dec"] += int(rank[j] <= max(1, n_ok // 10))
    by_year = {y: {"moonshots": v["n"], "caught": v["caught"], "in_top3": v["top3"], "in_top10": v["top10"],
                   "in_top_tenth": v["dec"], "auc": auc(np.array(v["s"]), np.array(v["y"]))} for y, v in sorted(yearly.items())}
    zmean = {g: zsum[g] / max(1, zcnt[g]) for g in zsum}
    blind = sorted(((m2.PLAIN[f], float(zmean["missed"][i] - zmean["picked"][i]), float(zmean["missed"][i]))
                    for i, f in enumerate(m2.FEATURES)), key=lambda x: -abs(x[1]))
    return {"rows": rows, "by_year": by_year, "blind_spots": blind[:12],
            "causes": {k: collections.Counter(r["cause"] for r in rows if not r["caught"])[k] for k in CAUSES},
            "causes_caught": {k: collections.Counter(r["cause"] for r in rows if r["caught"])[k] for k in CAUSES}}
