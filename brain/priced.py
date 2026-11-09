"""What the options already price (docs/PRICED.md): our predicted move sizes against the options' price,
recorded before the outcome and judged at the options' expiry. Paper only: nothing here trades.

The idea: we can predict how much a stock will move, not which way. Options already price an expected
move. A size forecast is worth something only where it disagrees with that price, a surprise the
options have not priced, and only if the disagreement survives the cost of trading options.

Two declared tests, recorded weekly from a Friday's close (the job runs on Saturday, before any new
price exists):

V1, any week (``record_v1``)
    For every stock on universe/us_broad.json with an option expiring the next Friday:
    - the at-the-money straddle's mid and quoted spread (CBOE delayed quotes, Friday's close);
    - our forecast of the size of the coming week's move (``Har``: the stock's own variance over the
      last 5, 22 and 66 sessions and the VIX, fitted once on 2011-2018 and frozen in
      knowledge/priced_v1.json).
    Stocks whose options price an event (the front expiry's at-the-money volatility 15%+ above the
    next one's, usually earnings) are recorded but left out of the test; earnings are O1's. Each week,
    the fifth whose straddles look cheapest against the forecast are "buy", the dearest fifth "sell".
    Ranking within the week means a bias every stock shares, such as options being dear on average,
    cannot make the result.

O1, earnings (``record_o1``)
    For the coming week's reporters with usable quotes (``earnradar.radar``): M1's expected move for the
    report against the options' priced move for it, and M1's side (20% margin), with the straddle.

Judging (``judge``) at the expiry's close: the move, whether it beat the priced move, and the straddle's
result for a buyer and a seller per dollar of stock, paying half the quoted spread on the way in. A
straddle held to expiry pays its intrinsic value, so there is no spread on the way out.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from brain import config
from brain.timeutil import now_iso

V1_PATH = config.ROOT / "research" / "v1_calls.jsonl"
O1_PATH = config.ROOT / "research" / "o1_calls.jsonl"
MODEL_PATH = config.ROOT / "knowledge" / "priced_v1.json"
UNIVERSE = config.ROOT / "universe" / "us_broad.json"
EVENT_BUMP = 1.15          # front at-the-money vol this far above the next expiry's: the options price an event
FIFTH = 0.2
MIN_WEEKS = 30
SOURCE = "https://cdn-api.cboe.com/api/global/delayed_quotes/options/{symbol}.json"


# --- the size forecast ----------------------------------------------------------------------------------
def har_x(closes: np.ndarray, vix: float) -> np.ndarray | None:
    """The forecast's inputs at a close: log variance of daily log returns over 5, 22 and 66 sessions, and
    the VIX as a daily variance. ``closes`` ends at that close."""
    c = np.asarray(closes, float)
    if len(c) < 67 or not np.isfinite(c[-67:]).all() or (c[-67:] <= 0).any() or not vix or vix <= 0:
        return None
    r = np.diff(np.log(c[-67:]))
    return np.array([1.0, *(math.log(float(np.mean(r[-k:] ** 2)) + 1e-10) for k in (5, 22, 66)),
                     math.log((vix / 100.0) ** 2 / 252.0)])


class Har:
    """log(sum of the next five squared daily returns) = x . beta; the five-session return's size is that
    sigma times a standardized draw whose distribution is the training years' own (``z`` quantiles)."""

    def __init__(self, beta, z_grid, fit_info=None):
        self.beta = np.asarray(beta, float)
        self.z = np.sort(np.asarray(z_grid, float))
        self.info = fit_info or {}

    def sigma(self, x: np.ndarray) -> float:
        return float(math.exp(0.5 * float(x @ self.beta)))

    def expected_abs(self, x: np.ndarray) -> float:
        """The expected size of the five-session move, as a fraction."""
        return self.sigma(x) * float(np.mean(self.z))

    def chance_above(self, x: np.ndarray, size: float) -> float:
        """The chance the five-session move, either way, is at least ``size`` (a fraction)."""
        k = size / self.sigma(x)
        p = 1.0 - np.searchsorted(self.z, k, side="left") / len(self.z)
        return float(min(max(p, 1e-4), 1 - 1e-4))

    def to_json(self) -> dict:
        return {"beta": self.beta.tolist(), "z_grid": self.z.tolist(), "info": self.info}

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "Har":
        d = json.loads(Path(path).read_text())
        return cls(d["beta"], d["z_grid"], d.get("info"))


def samples(closes: np.ndarray, vix: np.ndarray, every: int = 5):
    """(x, y, |five-session return|) every ``every`` sessions along one stock's aligned closes."""
    out = []
    for t in range(67, len(closes) - 5, every):
        x = har_x(closes[:t + 1], float(vix[t]))
        nxt = closes[t:t + 6]
        if x is None or not np.isfinite(nxt).all() or (nxt <= 0).any():
            continue
        r = np.diff(np.log(nxt))
        out.append((x, math.log(float(np.sum(r ** 2)) + 1e-10), abs(nxt[-1] / nxt[0] - 1)))
    return out


def fit(rows: list) -> Har:
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    sig = np.exp(0.5 * (X @ beta))
    z = np.array([r[2] for r in rows]) / sig
    grid = np.quantile(z, np.linspace(0.0005, 0.9995, 2000))
    return Har(beta, grid, {"rows": len(rows), "resid_sd": float(np.std(y - X @ beta)), "mean_abs_z": float(np.mean(z))})


# --- the options' side ------------------------------------------------------------------------------------
def straddle_quote(c: dict, expiry: str, spot: float) -> dict | None:
    """The at-the-money straddle on ``expiry`` nearest ``spot``: strike, mid and half the quoted spread
    (both legs), from usable quotes only."""
    from brain.sources import options
    legs = c["chain"].get(expiry) or {}
    both = [k for k, v in legs.items() if "C" in v and "P" in v]
    if not both or not spot:
        return None
    k = min(both, key=lambda s: abs(s - spot))
    call, put = legs[k]["C"], legs[k]["P"]
    if not (options.usable(call) and options.usable(put)):
        return None
    mid = (call["bid"] + call["ask"]) / 2 + (put["bid"] + put["ask"]) / 2
    half = ((call["ask"] - call["bid"]) + (put["ask"] - put["bid"])) / 2
    return {"expiry": expiry, "strike": float(k), "mid": round(mid, 4), "half_spread": round(half, 4),
            "call": [call["bid"], call["ask"]], "put": [put["bid"], put["ask"]]}


def event_ratio(c: dict, expiry: str) -> float | None:
    from brain.sources import options
    later = sorted(e for e in c["chain"] if e > expiry)
    a = options.atm_iv(c, expiry)
    b = options.atm_iv(c, later[0]) if later else None
    return round(a / b, 3) if a and b else None


def sides(calls: list[dict]) -> None:
    """Within one week's test calls, the cheapest fifth by forecast / priced is 'buy', the dearest 'sell'."""
    test = [c for c in calls if not c["event"]]
    order = sorted(test, key=lambda c: c["ratio"])
    n = int(len(order) * FIFTH)
    for c in calls:
        c["side"] = "none"
    for c in order[:n]:
        c["side"] = "sell"
    for c in order[len(order) - n:] if n else []:
        c["side"] = "buy"


def record_v1(friday: str, closes_for, chain_for, vix: float, tickers: list[str], model: Har, log=print) -> list[dict]:
    """Calls for the week after ``friday`` (whose close every input ends at). ``closes_for(t)`` -> (dividend-
    adjusted closes through that Friday, for the forecast; that Friday's traded close, the straddle's start);
    ``chain_for(t)`` -> an options chain (brain.sources.options.parse format)."""
    fri = date.fromisoformat(friday)
    calls = []
    for t in tickers:
        try:
            got = closes_for(t)
            if got is None:
                continue
            closes, s0 = got
            x = har_x(closes, vix)
            if x is None or not s0:
                continue
            c = chain_for(t)
            exp = next((e for e in sorted(c["chain"]) if friday < e <= (fri + timedelta(days=7)).isoformat()), None)
            if not exp:
                continue
            s0 = float(s0)
            q = straddle_quote(c, exp, s0)
            if not q:
                continue
            priced = q["mid"] / s0
            # options run to the expiry's close; the forecast covers five sessions, scaled to the sessions left
            days = int(np.busday_count(friday, exp)) or 1
            scale = math.sqrt(days / 5)
            exp_abs = model.expected_abs(x) * scale
            ev = event_ratio(c, exp)
            calls.append({"type": "call", "test": "V1", "made_at": now_iso(), "ticker": t, "start_date": friday,
                          "start_price": s0, "by": exp, "priced_pct": round(priced * 100, 3),
                          "expected_pct": round(exp_abs * 100, 3), "ratio": round(exp_abs / priced, 4),
                          "prob": round(model.chance_above(x, priced / scale), 4), "event": bool(ev and ev >= EVENT_BUMP),
                          "event_ratio": ev, "straddle": q, "source": SOURCE.format(symbol=t)})
        except Exception as e:  # noqa: BLE001 - one symbol's refusal must not stop the week
            log(f"  {t}: skipped ({type(e).__name__})")
    sides(calls)
    return calls


def record_o1(start: date, log=print, max_options: int = 250) -> list[dict]:
    """Calls for the coming week's earnings reporters (the radar's likely ones with usable quotes and M1's view)."""
    from brain import earnradar
    from brain.sources import options
    rows = earnradar.radar(start, 5, with_options=True, max_options=max_options, log=lambda m: None)
    calls = []
    for r in rows:
        m, s = r.get("m1"), r.get("straddle")
        if not (m and s) or r.get("date_check") or r.get("confidence") != "likely":
            continue
        try:
            c = options.fetch_chain(r["ticker"])
            q = straddle_quote(c, s["expiry"], r["spot"])
        except Exception:  # noqa: BLE001
            q = None
        if not q:
            continue
        calls.append({"type": "call", "test": "O1", "made_at": now_iso(), "ticker": r["ticker"],
                      "start_date": None, "start_price": float(r["spot"]), "by": s["expiry"], "report": r["date"],
                      "timing": r["timing"], "react": r["react"], "priced_pct": round(float(r["event_pct"]), 3),
                      "straddle_pct": float(s["implied_pct"]), "expected_pct": round(float(m["expected"]), 3),
                      "ratio": round(float(m["expected"]) / float(r["event_pct"]), 4) if r["event_pct"] else None,
                      "prob": round(float(m["chance"]), 4), "side": m["side"] or "none", "event": True,
                      "straddle": q, "source": SOURCE.format(symbol=r["ticker"])})
    return calls


# --- the ledgers ------------------------------------------------------------------------------------------
def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).exists() else []


def append(path: Path, recs: list[dict]) -> int:
    have = {(r.get("type"), r.get("ticker"), r.get("start_date"), r.get("by")) for r in read(path)}
    new = [r for r in recs if (r["type"], r["ticker"], r.get("start_date"), r["by"]) not in have]
    if new:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a") as f:
            for r in new:
                f.write(json.dumps(r, default=float, separators=(",", ":")) + "\n")
    return len(new)


def outcome(call: dict, close_by: float) -> dict:
    """A call's result at its expiry's close: the move, whether it beat the priced move, and the straddle's
    result per dollar of stock for a buyer and a seller, half the quoted spread paid on the way in."""
    s0, q = call["start_price"], call["straddle"]
    move = close_by / s0 - 1
    payoff = abs(close_by - q["strike"]) / s0
    return {"type": "judged", "ticker": call["ticker"], "start_date": call.get("start_date"), "by": call["by"],
            "close": float(close_by), "move_pct": round(move * 100, 3), "beat": abs(move) * 100 >= call["priced_pct"],
            "buyer": round(payoff - (q["mid"] + q["half_spread"]) / s0, 5),
            "seller": round((q["mid"] - q["half_spread"]) / s0 - payoff, 5)}


def judge(path: Path, close_on, today: str) -> int:
    """Judge every call whose expiry has closed by ``today``; ``close_on(ticker, date)`` -> that close or None."""
    recs = read(path)
    done = {(r["ticker"], r.get("start_date"), r["by"]) for r in recs if r["type"] == "judged"}
    out = []
    for c in recs:
        if c["type"] != "call" or c["by"] >= today or (c["ticker"], c.get("start_date"), c["by"]) in done:
            continue
        px = close_on(c["ticker"], c["by"])
        if px:
            out.append(outcome(c, px))
    if out:
        with Path(path).open("a") as f:
            for r in out:
                f.write(json.dumps(r, default=float, separators=(",", ":")) + "\n")
    return len(out)


# --- the record -----------------------------------------------------------------------------------------
def _t(x):
    x = np.asarray(x, float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")
    t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    return float(x.mean()), float(t), float(0.5 * math.erfc(t / math.sqrt(2)))


def evaluate(path: Path, test: str, min_calls: int = 5) -> dict:
    """Week by week (by the week's start): the rank correlation of the forecast / priced ratio with the
    actual move / priced; the result of following the sides net of half the spread; always selling (the
    options' usual premium); and how often moves beat the priced move."""
    from scipy.stats import spearmanr
    recs = read(path)
    calls = {(r["ticker"], r.get("start_date"), r["by"]): r for r in recs if r["type"] == "call" and not (test == "V1" and r["event"])}
    weeks = {}
    for j in (r for r in recs if r["type"] == "judged"):
        c = calls.get((j["ticker"], j.get("start_date"), j["by"]))
        if c and c.get("ratio"):
            weeks.setdefault(c.get("start_date") or c["made_at"][:10], []).append((c, j))
    ic, follow, sell_all, beat, n = [], [], [], [], 0
    for wk, rows in sorted(weeks.items()):
        if len(rows) < min_calls:
            continue
        n += len(rows)
        ratio = [c["ratio"] for c, _ in rows]
        real = [abs(j["move_pct"]) / c["priced_pct"] for c, j in rows]
        ic.append(spearmanr(ratio, real)[0])
        b = [j["buyer"] for c, j in rows if c["side"] == "buy"]
        s = [j["seller"] for c, j in rows if c["side"] == "sell"]
        if b or s:
            follow.append(float(np.mean(b + s)))
        sell_all.append(float(np.mean([j["seller"] for _, j in rows])))
        beat.append(float(np.mean([j["beat"] for _, j in rows])))
    m_ic, t_ic, p_ic = _t(ic)
    m_f, t_f, p_f = _t(follow)
    weeks_n = len(ic)
    ready = weeks_n >= MIN_WEEKS
    checks = {"its forecast ranks the moves beyond the options' price (mean weekly rank correlation > 0, p < 0.05)": m_ic > 0 and p_ic < 0.05,
              "following its sides makes money after half the spread (p < 0.05)": m_f > 0 and p_f < 0.05}
    return {"test": test, "weeks": weeks_n, "calls": n, "ready": ready,
            "rank_corr": {"mean": m_ic, "t": t_ic, "p": p_ic}, "follow_sides": {"mean": m_f, "t": t_f, "p": p_f},
            "always_sell": dict(zip(("mean", "t", "p"), _t(sell_all))), "beat_rate": float(np.mean(beat)) if beat else None,
            "checks": checks, "verdict": ("PASS" if all(checks.values()) else "FAIL") if ready else f"too early ({weeks_n} of {MIN_WEEKS} weeks)"}


# --- running it -----------------------------------------------------------------------------------------
def last_friday(today: date) -> date:
    return today - timedelta(days=(today.weekday() - 4) % 7)


class Prices:
    """Daily bars from the repo's price source, one fetch per ticker per run."""

    def __init__(self, start: str):
        self.start, self.cache = start, {}

    def bars(self, t: str):
        if t not in self.cache:
            from brain.sources import prices
            try:
                df = prices.fetch_daily(t, start=self.start)[0]
                df["date"] = df["date"].astype(str)
                self.cache[t] = df
            except Exception:  # noqa: BLE001
                self.cache[t] = None
        return self.cache[t]

    def through(self, t: str, day: str):
        df = self.bars(t)
        if df is None:
            return None
        d = df[df["date"] <= day]
        if not len(d) or d["date"].iloc[-1] != day:      # no bar for that Friday: not tradable then
            return None
        return d["adjclose"].values.astype(float), float(d["close"].iloc[-1])

    def close_on(self, t: str, day: str):
        df = self.bars(t)
        if df is None:
            return None
        d = df[df["date"] == day]
        return float(d["close"].iloc[0]) if len(d) else None


def ensure_index(bands=("broad", "top"), max_age_days: float = 6, log=print) -> None:
    """The earnings radar's index of report dates for the companies with liquid options (the broad list and
    the largest SEC-listed companies), read afresh when missing or a week old: their filings and prices
    first, then the index, and the market series M1 reads."""
    import time
    from brain import earnradar
    from brain.bank import Bank
    from brain.ingest import ingest
    p = earnradar.index_path()
    if p.exists() and time.time() - p.stat().st_mtime < max_age_days * 86400:
        return
    tickers = earnradar.universe(bands)
    log(f"O1: reading filings and prices for the earnings index ({len(tickers)} companies; slow)")
    for t in ["SPY", "^VIX"] + earnradar._missing(tickers):
        try:
            ingest(Bank(), t, ("prices",) if t.startswith("^") or t == "SPY" else ("filings", "prices"), log=lambda m: None)
        except Exception:  # noqa: BLE001 - one company's outage must not stop the rest
            pass
    earnradar.build_index(Bank(), tickers, log=lambda m: None)


def week(today: date | None = None, v1: bool = True, o1: bool = True, dry: bool = False, limit: int | None = None,
         log=print) -> dict:
    """The weekly job, run on a weekend: judge what has expired, then record V1 for the week after the
    last Friday's close and O1 for the coming week's reporters."""
    import time
    from brain.sources import options
    today = today or date.today()
    fri = last_friday(today).isoformat()
    px = Prices((today - timedelta(days=200)).isoformat())
    out = {"friday": fri, "judged": {}, "recorded": {}}
    if not dry:
        for name, path in (("V1", V1_PATH), ("O1", O1_PATH)):
            out["judged"][name] = judge(path, px.close_on, today.isoformat())
    if today.weekday() < 5 and not dry:
        log("V1 and O1 are recorded on a weekend, from Friday's close: nothing recorded on a weekday.")
        return out
    if v1:
        vixd = px.through("^VIX", fri)
        tickers = json.loads(UNIVERSE.read_text())["tickers"][:limit]

        def chain(t):
            time.sleep(0.25)                             # under four requests a second to CBOE
            return options.fetch_chain(t)
        calls = record_v1(fri, lambda t: px.through(t, fri), chain, vixd[1] if vixd else None, tickers,
                          Har.load(), log=lambda m: None) if vixd else []
        out["recorded"]["V1"] = calls if dry else append(V1_PATH, calls)
        log(f"V1: {len(calls)} calls for the week after {fri} ({sum(c['event'] for c in calls)} priced an event; "
            f"{sum(c['side'] == 'buy' for c in calls)} buy, {sum(c['side'] == 'sell' for c in calls)} sell)")
    if o1:
        start = date.fromisoformat(fri) + timedelta(days=3)
        try:
            ensure_index(log=log)
            calls = record_o1(start, log=log)
            for c in calls:
                c["start_date"] = fri
            out["recorded"]["O1"] = calls if dry else append(O1_PATH, calls)
            log(f"O1: {len(calls)} earnings calls for the week of {start}")
        except FileNotFoundError as e:
            log(f"O1: skipped ({e})")
    return out


def _cmd_week(a) -> int:
    today = date.fromisoformat(a.today) if a.today else None
    r = week(today, v1=not a.no_v1, o1=not a.no_o1, dry=a.dry, limit=a.limit)
    if r["judged"]:
        print("judged: " + ", ".join(f"{k} {v}" for k, v in r["judged"].items()))
    if a.dry:
        for k, calls in r["recorded"].items():
            for c in sorted(calls, key=lambda c: -c["ratio"] if c.get("ratio") else 0)[:12]:
                print(f"  {k} {c['ticker']:6} to {c['by']}: options price ±{c['priced_pct']:.1f}%, forecast ±{c['expected_pct']:.1f}% "
                      f"({c['ratio']:.2f}x){' [event]' if c['event'] and k == 'V1' else ''} side {c['side']}")
    return 0


def _cmd_status(a) -> int:
    for name, path in (("V1", V1_PATH), ("O1", O1_PATH)):
        e = evaluate(path, name)
        print(f"{name}: {e['weeks']} weeks judged ({e['calls']} calls); verdict: {e['verdict']}")
        if e["weeks"]:
            r, f, s = e["rank_corr"], e["follow_sides"], e["always_sell"]
            print(f"  forecast vs the options' price, rank correlation with the actual move: {r['mean']:+.3f} (t {r['t']:+.2f})")
            print(f"  following its sides, per week, per dollar of stock after half the spread: {f['mean']:+.4f} (t {f['t']:+.2f})")
            print(f"  always selling (the options' usual premium): {s['mean']:+.4f} (t {s['t']:+.2f}); moves beat the priced move "
                  f"{e['beat_rate']:.0%} of the time")
    print("Paper only. Under 30 judged weeks a record says nothing yet.")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("priced", help="our predicted move sizes against what the options price (docs/PRICED.md)")
    ps = p.add_subparsers(dest="priced_cmd", required=True)
    s = ps.add_parser("week", help="weekend job: judge expired calls, record V1 (any week) and O1 (earnings)")
    s.add_argument("--today", help="YYYY-MM-DD (default today)")
    s.add_argument("--no-v1", action="store_true")
    s.add_argument("--no-o1", action="store_true")
    s.add_argument("--dry", action="store_true", help="compute and show calls without writing them")
    s.add_argument("--limit", type=int, help="only the first N tickers of the list (a dry run)")
    s.set_defaults(fn=_cmd_week)
    s = ps.add_parser("status", help="the V1 and O1 records so far")
    s.set_defaults(fn=_cmd_status)
