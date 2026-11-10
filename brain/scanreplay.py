"""A replay of the market scan (docs/SCAN.md): the scan as it would have run at each week's last close, with the
frozen W1 and S1 models and only what was known at that close, judged against the five sessions that followed.

Descriptive. W1a and S1a are the declared tests of these models; a replay adds no verdict. It answers how the scan
would have read, week after week, to someone using it live:

- **Ranges:** how often each week's 2-in-3 and 9-in-10 ranges held, and how often a stock fell below the 9-in-10
  range's floor (the scan's "bad week" line: one week in twenty, if it is honest).
- **Movers:** each week's 25 widest ranges against every listed stock: how far they actually moved, and how many
  moved 10% or more either way.
- **Shocks:** each week's 25 likeliest 20%+ days by S1 against what happened.
- **As a buy list (a misuse, shown on purpose):** the 25 widest ranges bought at the close and sold five sessions
  later, against every listed stock and against SPY. The scan reads size, not direction, so nothing here should
  favour them; the replay shows whether it did.

Each week counts once: a week's stocks move together, so a week is one reading, not 1,500.
"""
from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np

from brain import config, scan, shocks
from brain.bank import Bank, canon

TOP = 25
BIG = 0.10                  # a "big" five-session move, either way
COST = 0.005                # a round trip in these stocks, both ways together, for the buy-list illustration


def rows(bank: Bank, tickers: list[str], first: str, last: str, log=print) -> dict:
    """Tradable stock-weeks ending in [first, last] whose five sessions have closed, with what the scan read and what
    followed: ``t``, ``date``, ``X`` (scan.FEATS), ``l5`` (the five-session log return), ``shock`` (1 when one of
    the five sessions moved 20%+ against SPY), ``spy5`` (SPY's five-session log return), and ``hold`` and
    ``spy_hold``: the stock's and SPY's log return from this reading to the next week's (four sessions in a holiday
    week), so weeks bought and sold in turn do not overlap."""
    T, D, X, L, Y, M, HS, HM = [], [], [], [], [], [], [], []
    for c in range(0, len(tickers), scan.CHUNK):
        b = Bank(bank.root)
        v = b.view()
        mk = shocks.market(v)
        if mk is None:
            raise SystemExit(f"no {shocks.BENCH} or {shocks.VIX} prices in the bank")
        cal, spy = mk[0], mk[1]
        every = shocks.week_ends(cal)
        nxt = np.r_[every[1:], len(cal)]                   # each reading's next one
        keep_w = (cal[every] >= first) & (cal[every] <= last) & (every + scan.H < len(cal)) & (nxt < len(cal))
        ends, nxt = every[keep_w], nxt[keep_w]
        with np.errstate(all="ignore"):
            spy5 = np.log(spy[ends + scan.H] / spy[ends])
            spyh = np.log(spy[nxt] / spy[ends])
        for t in tickers[c:c + scan.CHUNK]:
            got = scan.ticker_frame(v, t, mk)
            if not got:
                continue
            f = got[1]
            x = np.column_stack([f[k][ends] for k in scan.FEATS])
            adj = shocks._aligned(shocks._series(v, t), "adjclose", cal)
            with np.errstate(all="ignore"):
                hold = np.log(adj[nxt] / adj[ends])
            keep = (np.isfinite(x).all(1) & f["ok"][ends] & np.isfinite(f["l5"][ends]) & np.isfinite(f["y"][ends])
                    & np.isfinite(spy5) & np.isfinite(hold) & np.isfinite(spyh))
            if keep.any():
                T += [t] * int(keep.sum())
                D += [str(d) for d in cal[ends][keep]]
                X.append(x[keep].astype(np.float32))
                L.append(f["l5"][ends][keep])
                Y.append(f["y"][ends][keep])
                M.append(spy5[keep])
                HS.append(hold[keep])
                HM.append(spyh[keep])
        if log:
            log(f"{min(c + scan.CHUNK, len(tickers))}/{len(tickers)} tickers, {len(T)} stock-weeks")
        del b, v
        gc.collect()
    if not T:
        return {"t": [], "date": [], "X": np.zeros((0, len(scan.FEATS)), np.float32), "l5": np.zeros(0),
                "shock": np.zeros(0), "spy5": np.zeros(0), "hold": np.zeros(0), "spy_hold": np.zeros(0)}
    return {"t": T, "date": D, "X": np.concatenate(X), "l5": np.concatenate(L), "shock": np.concatenate(Y),
            "spy5": np.concatenate(M), "hold": np.concatenate(HS), "spy_hold": np.concatenate(HM)}


def predict(d: dict, w1: dict, s1: dict | None) -> dict:
    """The scan's reading of each stock-week: W1's quantiles (log returns, scan.QS) and S1's chance of a 20%+ day."""
    X = np.asarray(d["X"], float)
    q = scan.quantiles(w1, X)
    k = len(shocks.FEATS)
    ch = shocks.chances(s1, [dict(zip(shocks.FEATS, x[:k])) for x in X]) if s1 else np.full(len(X), np.nan)
    return {"q": q, "half": (np.expm1(q[:, 3]) - np.expm1(q[:, 1])) / 2, "shock_p": ch}


def weekly(d: dict, pred: dict, top: int = TOP) -> list[dict]:
    """One record per week: what the scan said and what happened. Ranges and moves are judged on the five sessions
    the scan read; the buy-list returns (``top_ret``, ``all_ret``, ``spy_ret``) run to the next week's reading."""
    y, q = np.asarray(d["l5"], float), pred["q"]
    r = np.expm1(y)
    h = np.expm1(np.asarray(d["hold"], float))
    date = np.array(d["date"])
    out = []
    for w in sorted(set(d["date"])):
        i = np.flatnonzero(date == w)
        movers = i[np.argsort(-pred["half"][i], kind="mergesort")[:top]]
        shocky = i[np.argsort(-pred["shock_p"][i], kind="mergesort")[:top]]
        out.append({
            "week": w, "stocks": len(i), "spy": float(np.expm1(d["spy5"][i[0]])),
            "spy_ret": float(np.expm1(d["spy_hold"][i[0]])),
            "held68": float(np.mean((y[i] >= q[i, 1]) & (y[i] <= q[i, 3]))),
            "held90": float(np.mean((y[i] >= q[i, 0]) & (y[i] <= q[i, 4]))),
            "below90": float(np.mean(y[i] < q[i, 0])), "above90": float(np.mean(y[i] > q[i, 4])),
            "all_abs": float(np.mean(np.abs(r[i]))), "all_big": float(np.mean(np.abs(r[i]) >= BIG)),
            "all_ret": float(np.mean(h[i])),
            "top_abs": float(np.mean(np.abs(r[movers]))), "top_big": float(np.mean(np.abs(r[movers]) >= BIG)),
            "top_ret": float(np.mean(h[movers])), "top_said": float(np.mean(pred["half"][movers])),
            "top_below90": float(np.mean(y[movers] < q[movers, 0])),
            "shock_said": float(np.sum(pred["shock_p"][shocky])), "shock_hits": int(np.sum(d["shock"][shocky])),
            "shock_all_said": float(np.sum(pred["shock_p"][i])), "shock_all": int(np.sum(d["shock"][i])),
        })
    return out


def _t(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float(x.mean()) if len(x) else float("nan"), float("nan")
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def growth(rets) -> tuple[float, float]:
    """$1 compounded through weekly returns: where it ends and its worst fall from a peak."""
    v = np.cumprod(1 + np.asarray(rets, float))
    peak = np.maximum.accumulate(np.r_[1.0, v])[1:]
    return float(v[-1]) if len(v) else 1.0, float((v / peak - 1).min()) if len(v) else 0.0


def summary(weeks: list[dict]) -> dict:
    """The replay over all its weeks, each week counted once."""
    if not weeks:
        return {"weeks": 0}
    a = {k: np.array([w[k] for w in weeks], float) for k in weeks[0] if k != "week"}
    top_net = a["top_ret"] - COST
    out = {"weeks": len(weeks), "first": weeks[0]["week"], "last": weeks[-1]["week"],
           "stock_weeks": int(a["stocks"].sum()), "stocks_per_week": float(a["stocks"].mean()),
           "held68": float(np.average(a["held68"], weights=a["stocks"])),
           "held90": float(np.average(a["held90"], weights=a["stocks"])),
           "below90": float(np.average(a["below90"], weights=a["stocks"])),
           "above90": float(np.average(a["above90"], weights=a["stocks"])),
           "weeks_68_in_60_76": int(((a["held68"] >= 0.60) & (a["held68"] <= 0.76)).sum()),
           "weeks_90_in_85_95": int(((a["held90"] >= 0.85) & (a["held90"] <= 0.95)).sum()),
           "worst68": sorted(({"week": w["week"], "held68": w["held68"], "held90": w["held90"], "spy": w["spy"]}
                              for w in weeks), key=lambda w: w["held68"])[:5],
           "widest68": sorted(({"week": w["week"], "held68": w["held68"], "held90": w["held90"], "spy": w["spy"]}
                               for w in weeks), key=lambda w: -w["held68"])[:3],
           "top_abs": float(a["top_abs"].mean()), "all_abs": float(a["all_abs"].mean()),
           "top_said": float(a["top_said"].mean()),
           "top_big": float(a["top_big"].mean()), "all_big": float(a["all_big"].mean()),
           "weeks_top_moved_more": int((a["top_abs"] > a["all_abs"]).sum()),
           "top_below90": float(a["top_below90"].mean()),
           "shock_hits": int(a["shock_hits"].sum()), "shock_said": float(a["shock_said"].sum()),
           "shock_all": int(a["shock_all"].sum()), "shock_all_said": float(a["shock_all_said"].sum()),
           "shock_picks": int(len(weeks) * TOP)}
    for name, x in (("top", a["top_ret"]), ("top_net", top_net), ("all", a["all_ret"]), ("spy", a["spy_ret"]),
                    ("top_vs_all", a["top_ret"] - a["all_ret"])):
        m, t = _t(x)
        end, dd = growth(x)
        out[name] = {"mean": m, "t": t, "end": end, "worst_fall": dd, "worst_week": float(x.min()),
                     "best_week": float(x.max()), "up_weeks": int((x > 0).sum())}
    return out


def by_quarter(weeks: list[dict]) -> list[dict]:
    out: dict[str, list[dict]] = {}
    for w in weeks:
        y, m = w["week"][:4], int(w["week"][5:7])
        out.setdefault(f"{y}Q{(m - 1) // 3 + 1}", []).append(w)
    return [{"quarter": k, "weeks": len(v),
             "held68": float(np.average([w["held68"] for w in v], weights=[w["stocks"] for w in v])),
             "held90": float(np.average([w["held90"] for w in v], weights=[w["stocks"] for w in v])),
             "top_abs": float(np.mean([w["top_abs"] for w in v])), "all_abs": float(np.mean([w["all_abs"] for w in v])),
             "shock_hits": int(sum(w["shock_hits"] for w in v)), "shock_said": float(sum(w["shock_said"] for w in v)),
             "top_ret": growth([w["top_ret"] for w in v])[0] - 1, "all_ret": growth([w["all_ret"] for w in v])[0] - 1,
             "spy": growth([w["spy_ret"] for w in v])[0] - 1} for k, v in sorted(out.items())]


def run(bank: Bank, first: str, last: str, tickers=None, log=print) -> dict:
    model, s1 = scan.load(), shocks.load()
    if model is None:
        raise SystemExit("no fitted model: run python -m brain scan fit")
    d = rows(bank, tickers or shocks.universe_tickers(), first, last, log=log)
    weeks = weekly(d, predict(d, model["w1"], s1["s1"] if s1 else None))
    return {"first": first, "last": last, "w1_sha": scan.model_sha(), "s1_sha": shocks.model_sha() if s1 else None,
            "w1_train": model["train"], "summary": summary(weeks), "quarters": by_quarter(weeks), "weeks": weeks}


def save(res: dict) -> Path:
    p = config.ROOT / "research" / "scans" / "replays" / f"replay-{res['first']}_{res['last']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(canon(res) + "\n")
    return p


def _pct(x: float, signed: bool = False, d: int = 1) -> str:
    return f"{x * 100:+.{d}f}%" if signed else f"{x * 100:.{d}f}%"


def show(res: dict) -> None:
    s = res["summary"]
    if not s["weeks"]:
        print("no closed weeks in that window")
        return
    print(f"Scan replay, {s['first']} to {s['last']}: {s['weeks']} weeks, {s['stock_weeks']} stock-weeks (about "
          f"{s['stocks_per_week']:.0f} a week). W1 fitted {res['w1_train'][0]}..{res['w1_train'][1]}, so every week "
          "here is out of sample. Descriptive: the declared tests are W1a and S1a.")
    print(f"  Ranges: the 2-in-3 range held {_pct(s['held68'])} and the 9-in-10 range {_pct(s['held90'])}; "
          f"{_pct(s['below90'])} fell below the 9-in-10 floor and {_pct(s['above90'])} rose above its ceiling. "
          f"Weeks with the 2-in-3 range holding 60%-76%: {s['weeks_68_in_60_76']} of {s['weeks']}.")
    print("  Worst weeks for the 2-in-3 range: " + "; ".join(
        f"{w['week']} {_pct(w['held68'], d=0)} (SPY {_pct(w['spy'], True)})" for w in s["worst68"]))
    print(f"  Movers: each week's {TOP} widest ranges moved {_pct(s['top_abs'])} on average either way (said "
          f"±{_pct(s['top_said'])}), every listed stock {_pct(s['all_abs'])}; {_pct(s['top_big'], d=0)} of them "
          f"moved 10%+, against {_pct(s['all_big'], d=0)} of all. They moved more in {s['weeks_top_moved_more']} of "
          f"{s['weeks']} weeks.")
    print(f"  Shocks: S1's {TOP} likeliest each week had {s['shock_hits']} 20%+ days in {s['shock_picks']} picks "
          f"(its chances summed to {s['shock_said']:.0f}); across every stock-week, {s['shock_all']} against "
          f"{s['shock_all_said']:.0f} said.")
    t, a, m = s["top"], s["all"], s["spy"]
    print(f"  As a buy list (a misuse): the {TOP} widest ranges bought each week turned $1 into ${t['end']:.2f} "
          f"before costs (${s['top_net']['end']:.2f} after {COST * 100:.1f}% a round trip; worst fall "
          f"{_pct(t['worst_fall'], d=0)}, worst week {_pct(t['worst_week'], True)}); every listed stock "
          f"${a['end']:.2f}; SPY ${m['end']:.2f}. Their weekly edge over every listed stock: "
          f"{_pct(s['top_vs_all']['mean'], True, 2)} (t {s['top_vs_all']['t']:+.2f}).")
    print("A range is a size, not a direction. Survivorship: the list was chosen in 2026, so stocks that were "
          "delisted or bought out are missing.")


def cmd(a) -> int:
    res = run(Bank(), a.first, a.last)
    show(res)
    if a.save:
        print(f"saved {save(res)}")
    return 0
