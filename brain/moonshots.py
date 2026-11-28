"""The moonshot watch (docs/MOONSHOTS.md): where the next 20%+ jumps are likely to come from, and an honest
scoreboard of what the list caught and what buying it would have returned.

    python -m brain moonshots list [--top 25]     # the latest recorded scan's likeliest names, with what they are
    python -m brain moonshots score               # every judged recorded scan: jumps caught, crashes, paper return

Asked for on 2026-11-28: "i want to predict every moonshots before they happen or atleast most of them." A moonshot
here is a session in which a stock beats SPY by 20% or more. The list is the scan's own (S1's chance of a 20%+
session either way in the next five, tested by S1a): nothing new is fitted. What was learned building it, on weeks
of 2011-2023 only (``research/moonshots/``):

- Where they come from can be said: each week's top 20% of stocks by S1 held 82% of the jumps, the top 5% 53%.
- Which way cannot: a model fitted to jumps alone ranks them no better than S1 (53% against 54% in the top 5%), and
  every list that holds the jumps holds the crashes beside them, about one crash for two jumps.
- Buying the top 5% each week returned a mean +0.8% to +1.1% a week against SPY on paper while the usual pick lost
  (median -0.8%, 46% rose): a few jumps paid for many small losses. That paper figure cannot be trusted: the bank
  holds only companies still listed today, so those that failed are missing, and these stocks are costly to trade.

So the scoreboard here uses recorded scans only: each list was written before its sessions traded, whatever became
of its stocks. One week's stocks move together, so a week is one reading; under MIN_WEEKS weekly readings the
record says nothing. A list says where to look. It is not a direction, and it is not a trade.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np

from brain import scan, shocks
from brain.bank import Bank

LISTS = (("top 25", 25), ("top 5%", 0.05), ("top 10%", 0.10), ("top 20%", 0.20))
MIN_WEEKS = 30
START = "2026-12-04"        # the first weekly scan read after the lists and this reading were fixed (2026-11-28)


def cut(n: int, k: float) -> int:
    """How many names a list holds among ``n`` ranked stocks: a count as given, or a share rounded."""
    return int(min(n, k)) if k >= 1 else int(round(n * k))


def ranked(rec: dict) -> list[tuple[str, float]]:
    """A recorded scan's stocks with a shock chance, the likeliest first (ties by ticker)."""
    rows = [(t, r[4]) for t, r in rec["rows"].items() if r[4] is not None]
    return sorted(rows, key=lambda x: (-x[1], x[0]))


def week_end(asof: str, cal: list[str]) -> bool:
    """Is ``asof`` the last session of its calendar week in ``cal`` (a weekly reading)?"""
    i = cal.index(asof)
    if i + 1 >= len(cal):
        return False
    a, b = date.fromisoformat(asof).isocalendar(), date.fromisoformat(cal[i + 1]).isocalendar()
    return (a[0], a[1]) != (b[0], b[1])


def judge_scan(rec: dict, moves: dict[str, tuple[float, bool, bool]]) -> dict:
    """One recorded scan against what happened (pure). ``moves`` gives each stock's return against SPY over the
    five sessions and whether one of them was a 20%+ jump, a 20%+ crash. Stocks without a whole window are left
    out of every count."""
    rows = [(t, c) for t, c in ranked(rec) if t in moves]
    ret = np.array([moves[t][0] for t, _ in rows])
    up = np.array([moves[t][1] for t, _ in rows], bool)
    dn = np.array([moves[t][2] for t, _ in rows], bool)
    out = {"asof": rec["asof"], "n": len(rows), "jumps": int(up.sum()), "crashes": int(dn.sum()),
           "all_mean": float(ret.mean()) if len(rows) else float("nan"), "lists": {}}
    for name, k in LISTS:
        m = cut(len(rows), k)
        if not m:
            continue
        out["lists"][name] = {"n": m, "jumps": int(up[:m].sum()), "crashes": int(dn[:m].sum()),
                              "mean": float(ret[:m].mean()), "median": float(np.median(ret[:m])),
                              "rose": float(np.mean(ret[:m] > 0)), "said": float(np.mean([c for _, c in rows[:m]]))}
    return out


def moves_for(view, mk, recs: list[dict]) -> dict[str, dict[str, tuple[float, bool, bool]]]:
    """For each judged scan (by its date): every ranked stock's five-session return against SPY and its jump and
    crash flags, from the bank's prices. Each stock's prices are read once."""
    cal = [str(x) for x in mk[0]]
    bclose = mk[1]
    out: dict[str, dict] = {r["asof"]: {} for r in recs}
    tick = sorted({t for r in recs for t, _ in ranked(r)})
    for t in tick:
        df = shocks._series(view, t)
        if df is None:
            continue
        adj = shocks._aligned(df, "adjclose", np.array(cal))
        for r in recs:
            row = r["rows"].get(t)
            if not row or row[4] is None:
                continue
            i0 = cal.index(r["asof"])
            a, b = adj[i0:i0 + shocks.H + 1], bclose[i0:i0 + shocks.H + 1]
            if len(a) < shocks.H + 1 or not np.all(np.isfinite(a)) or a[0] <= 0:
                continue
            e = a[1:] / a[:-1] - b[1:] / b[:-1]
            out[r["asof"]][t] = (float(a[-1] / a[0] - b[-1] / b[0]), bool(e.max() >= shocks.SHOCK),
                                 bool(e.min() <= -shocks.SHOCK))
    return out


def score(bank: Bank) -> dict:
    """Every recorded scan whose five sessions have closed, judged; ``weekly`` marks the week-end scans."""
    v = bank.view()
    mk = shocks.market(v)
    if mk is None:
        return {"scans": [], "open": 0}
    cal = [str(x) for x in mk[0]]
    recs, waiting = [], 0
    for p in sorted(scan.scans_dir().glob("????-??-??.json")):
        rec = json.loads(p.read_text())
        if rec["asof"] in cal and cal.index(rec["asof"]) + shocks.H < len(cal):
            recs.append(rec)
        else:
            waiting += 1
    moves = moves_for(v, mk, recs)
    scans = [{**judge_scan(r, moves[r["asof"]]), "weekly": week_end(r["asof"], cal)} for r in recs]
    return {"scans": scans, "open": waiting}


def summary(scans: list[dict], name: str) -> dict | None:
    got = [(s, s["lists"][name]) for s in scans if name in s["lists"]]
    if not got:
        return None
    j, c = sum(s["jumps"] for s, _ in got), sum(s["crashes"] for s, _ in got)
    return {"n": len(got), "mean": float(np.mean([x["mean"] for _, x in got])),
            "median": float(np.mean([x["median"] for _, x in got])), "rose": float(np.mean([x["rose"] for _, x in got])),
            "jumps": sum(x["jumps"] for _, x in got), "all_jumps": j, "crashes": sum(x["crashes"] for _, x in got),
            "all_crashes": c, "up_weeks": float(np.mean([x["mean"] > 0 for _, x in got])),
            "worst": float(min(x["mean"] for _, x in got)), "best": float(max(x["mean"] for _, x in got))}


def score_lines(res: dict) -> list[str]:
    scans = res["scans"]
    if not scans:
        return [f"No recorded scan has closed its five sessions yet ({res['open']} waiting)."]
    L = [f"Recorded scans judged: {len(scans)} ({res['open']} still open). A moonshot is a session 20%+ above SPY; a "
         f"crash 20%+ below."]
    for s in scans[-6:]:
        x = s["lists"].get("top 5%")
        if x:
            L.append(f"  {s['asof']}{' (week end)' if s['weekly'] else ''}: {s['jumps']} jumps and {s['crashes']} crashes "
                     f"among {s['n']} stocks; the top 5% ({x['n']}) held {x['jumps']} jumps and {x['crashes']} crashes "
                     f"and returned {x['mean'] * 100:+.1f}% against SPY (median {x['median'] * 100:+.1f}%)")
    weekly = [s for s in scans if s["weekly"] and s["asof"] >= START]
    for label, pool in ((f"All {len(scans)} judged scans (they overlap: several a week)", scans),
                        (f"Weekly readings from {START} ({len(weekly)})", weekly)):
        if not pool:
            L.append(f"{label}: none yet.")
            continue
        L.append(f"{label}:")
        for name, _ in LISTS:
            m = summary(pool, name)
            if not m:
                continue
            share = f"{m['jumps'] / m['all_jumps'] * 100:.0f}%" if m["all_jumps"] else "none"
            L.append(f"  {name:<8} held {m['jumps']} of {m['all_jumps']} jumps ({share}) and {m['crashes']} of "
                     f"{m['all_crashes']} crashes; bought in equal parts: mean {m['mean'] * 100:+.2f}% a reading against "
                     f"SPY, median pick {m['median'] * 100:+.2f}%, {m['rose'] * 100:.0f}% of picks rose; "
                     f"{m['up_weeks'] * 100:.0f}% of readings above zero (worst {m['worst'] * 100:+.1f}%, best "
                     f"{m['best'] * 100:+.1f}%)")
    if len(weekly) < MIN_WEEKS:
        L.append(f"Under {MIN_WEEKS} weekly readings this record says nothing yet, either way. Paper returns, before "
                 f"the cost of trading; a list is where to look, not a direction (docs/MOONSHOTS.md).")
    return L


def what_it_is(view, mk, t: str) -> dict:
    """Under $5, a drug company, a results report due: the three places most jumps come from."""
    got = shocks.ticker_frame(view, t, mk)
    if not got:
        return {}
    f = got[1]
    return {"pen": bool(f["pen"][-1] > 0), "bio": bool(f["bio"][-1] > 0), "report": bool(f["earn_win"][-1] > 0)}


def list_lines(bank: Bank, top: int = 25) -> list[str]:
    from brain import scanwatch
    got = scanwatch.latest_scan()
    if not got:
        return ["no recorded scan: python -m brain scan run --refresh --record"]
    rec = got[1]
    rows = ranked(rec)
    v = bank.view()
    mk = shocks.market(v)
    n5, n20 = cut(len(rows), 0.05), cut(len(rows), 0.20)
    L = [f"Moonshot watch from the scan of {rec['asof']}: {len(rows)} stocks with a chance. A chance is of a session "
         f"20%+ away from SPY in the next five, either way (S1; S1a PASS on 2024-2026, research/RESULTS.md).",
         f"On 2011-2023 about two such sessions in three were jumps and one a crash, and nothing here told them "
         f"apart. Each week's top 20% (here {n20} names) held about four jumps in five; the top 5% ({n5}) about half."]
    for t, c in rows[:top]:
        w = what_it_is(v, mk, t) if mk else {}
        tags = [x for x, on in (("under $5", w.get("pen")), ("a drug company", w.get("bio")),
                                ("a results report due", w.get("report"))) if on]
        L.append(f"  {t:<6} {c * 100:>5.1f}%" + (f"   {', '.join(tags)}" if tags else ""))
    said = float(np.mean([c for _, c in rows[:top]])) if rows else float("nan")
    L.append(f"The top {min(top, len(rows))} average {said * 100:.1f}%: most of them will not have such a session, and the "
             f"usual pick on such lists lost against SPY (median -0.8% a week on 2011-2023). A list is where to look, "
             f"not a direction and not a trade; the record: python -m brain moonshots score.")
    return L


def _cmd(a) -> int:
    bank = Bank()
    if a.action == "list":
        print("\n".join(list_lines(bank, a.top)))
    else:
        print("\n".join(score_lines(score(bank))))
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("moonshots", help="where the next 20%+ jumps are likely to come from (the scan's list), and "
                                         "the record of what the list caught and returned")
    p.add_argument("action", choices=["list", "score"])
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(fn=_cmd)
