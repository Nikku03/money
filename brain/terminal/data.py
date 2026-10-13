"""Terminal functions: one parsed command in, one JSON-ready dict out.

Everything comes from the bank through a View, and every row carries the id of the
fact behind it (``cite``), so a screen never shows a number the bank cannot back.
The computations are shared with the brief (``brain/brief.py``): the same price
summary, the same derived metrics written back as facts, the same signal evidence.
Response shapes are specified in docs/TERMINAL.md section 2.

User-level problems (no data for a ticker, a bad argument) come back as
``{"ok": false, "error", "hint"}`` where the hint is usually the command that fixes
it (``INGEST XYZ``). Anything else is a bug and raises.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from brain import brief, claims, config, markets, metrics, series
from brain.bank import Bank, Fact, View
from brain.metrics import PriceBook
from brain.sources import fred, treasury
from brain.terminal.commands import (FUNCTIONS, KEYS, RANGE_SHAPE, RANGES, TICKER, Command, CommandError, parse,
                                     suggest)
from brain.timeutil import days_between, today

MAX_POINTS = 1500
RANGE_MONTHS = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "2Y": 24, "5Y": 60, "10Y": 120}
SERVER_ONLY = {"JOBS": "JOBS lists the terminal server's background jobs",
               "ASK": "ASK talks to the analyst through the terminal server",
               "INGEST": "INGEST starts a background job on the terminal server"}
PRICE = "series.price_daily"
_HEX = re.compile(r"[0-9a-f]{4,20}")


class UserError(Exception):
    """Something the user can fix: missing data or a bad argument."""

    def __init__(self, error: str, hint: str = ""):
        super().__init__(error)
        self.error, self.hint = error, hint


# ---------- plumbing ----------
def clean(o):
    """JSON-safe copy: NaN/inf -> None, numpy scalars and arrays -> Python, tuples -> lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _now() -> str:
    """The terminal's point in time: the end of the current UTC day. Date-only facts (a daily close,
    a filing date) count as known at the end of their day, so a close collected this evening would
    otherwise stay hidden until midnight UTC. Nothing dated after today becomes visible."""
    return today() + "T23:59:59Z"


def _view(bank: Bank) -> View:
    return bank.view(_now())


def _ok(cmd: Command, title: str, asof: str | None, view: str, data: dict) -> dict:
    return {"ok": True, "fn": cmd.fn, "ticker": cmd.ticker, "args": list(cmd.args), "title": title,
            "asof": asof, "view": view, "data": data}


def _err(cmd: Command, error: str, hint: str = "") -> dict:
    return {"ok": False, "fn": cmd.fn, "ticker": cmd.ticker, "args": list(cmd.args), "error": error, "hint": hint}


def cite(f: Fact | None) -> str | None:
    return f.id[:8] if f is not None else None


def _ingest_hint(t: str) -> str:
    return "INGEST MARKETS" if t in markets.BY_SYMBOL else f"INGEST {t}"


def _per_year(t: str) -> int:
    return 365 if markets.kind(t) == "crypto" else 252


def kind_of(bank: Bank, t: str) -> str:
    """index / volatility / fx / commodity / crypto for watched market symbols; equity when SEC
    identity facts exist; other otherwise (ETFs such as SPY, foreign listings)."""
    k = markets.kind(t)
    if k:
        return k
    return "equity" if any(a.startswith("identity.") for a in bank.attrs(t)) else "other"


def _name(view: View, t: str) -> str | None:
    f = view.latest(t, "identity.name")
    if f is not None:
        return str(f.value)
    if t in markets.BY_SYMBOL:
        return markets.BY_SYMBOL[t]["name"]
    snap = view.latest(t, PRICE)
    return snap.context.get("longName") if snap is not None else None


def _title(t: str, name: str | None, what: str) -> str:
    return f"{t} — {name} · {what}" if name else f"{t} · {what}"


def _require(bank: Bank, t: str) -> None:
    if not bank.attrs(t):
        raise UserError(f"nothing in the bank for {t}", _ingest_hint(t))


def _frame(f: Fact, view: View) -> pd.DataFrame:
    df = series.load(f.value)
    return df[df["date"] <= view.t].reset_index(drop=True)


def _minus_months(date: str, months: int) -> str:
    return (pd.Timestamp(date) - pd.DateOffset(months=months)).strftime("%Y-%m-%d")


def _on_or_before(dates: np.ndarray, date: str) -> int | None:
    i = int(np.searchsorted(dates, date, side="right")) - 1
    return i if i >= 0 else None


def downsample(n: int, max_points: int = MAX_POINTS) -> np.ndarray:
    """Evenly spaced indices into n points, at most max_points, always keeping the first and last."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def _headlines(view: View, t: str, n: int) -> list[dict]:
    hl = sorted(view.facts(t, "headline"), key=lambda f: f.known_at, reverse=True)[:n]
    return [{"time": f.known_at, "title": f.value, "url": f.source.get("url"), "source": f.source.get("name"),
             "cite": cite(f)} for f in hl]


def _table(columns: list[dict], rows: list[dict], note: str = "", groups: list[str] | None = None) -> dict:
    if groups is None:
        groups = list(dict.fromkeys(r["_group"] for r in rows if r.get("_group")))
    return {"columns": columns, "rows": rows, "groups": groups, "note": note}


def _col(key: str, label: str, type_: str = "text", digits: int | None = None) -> dict:
    c = {"key": key, "label": label, "type": type_}
    if digits is not None:
        c["digits"] = digits
    return c


# ---------- HELP ----------
def _help(bank: Bank, cmd: Command) -> dict:
    fns = [{"code": c, "scope": s["scope"], "args": s["args"], "desc": s["desc"]} for c, s in FUNCTIONS.items()]
    return _ok(cmd, "HELP — functions and keys", None, "help", {"functions": fns, "keys": list(KEYS)})


# ---------- DES ----------
def _metric_rows(bank: Bank, view: View, t: str, book: PriceBook) -> list[dict]:
    rows = []
    for nm, m, unit, f in brief.derived(bank, view, t, book):
        rows.append({"key": nm, "label": brief.DERIVED_LABELS.get(nm, nm), "value": m.value, "unit": unit,
                     "as_of": m.as_of, "cite": cite(f), "inputs": [cite(x) for x in m.inputs]})
        if nm == "earnings_yield" and m.value > 0:  # display only: no fact of its own
            rows.append({"key": "pe", "label": "P/E", "value": 1 / m.value, "unit": "x", "as_of": m.as_of,
                         "cite": cite(f), "inputs": [cite(f)], "display": f"{1 / m.value:,.1f}"})
    return rows


def _des(bank: Bank, cmd: Command) -> dict:
    t = cmd.ticker
    _require(bank, t)
    view = _view(bank)
    kind = kind_of(bank, t)
    name = _name(view, t)
    book = PriceBook(bank, [t], asof=_now())
    ps = brief.price_summary(view, book, t, _per_year(t))
    price = {**ps, "cite": cite(book.snapshot[t])} if ps else None
    identity, mets, filings = [], [], []
    if markets.kind(t) is None:
        identity = [{"label": label, "value": brief.identity_value(f), "cite": cite(f)}
                    for a, label in brief.IDENTITY if (f := view.latest(t, a)) is not None]
        mets = _metric_rows(bank, view, t, book)
        fil = sorted(view.facts(t, "filing"), key=lambda f: f.known_at, reverse=True)
        filings = [{"date": f.known_at[:10], "form": f.value, "desc": brief.filing_desc(f), "url": f.source.get("url"),
                    "cite": cite(f)} for f in fil if brief.notable_filing(f.value)][:5]
    headlines = _headlines(view, t, 5)
    dates = [price["date"]] if price else [r["date"] for r in filings] + [h["time"][:10] for h in headlines]
    data = {"ticker": t, "name": name, "kind": kind, "identity": identity, "price": price, "metrics": mets,
            "filings": filings, "headlines": headlines}
    return _ok(cmd, _title(t, name, "description"), max(dates) if dates else None, "des", data)


# ---------- GP ----------
def _range_start(dates: np.ndarray, rng: str) -> int:
    if rng == "MAX":
        return 0
    last = str(dates[-1])
    if rng == "YTD":  # from the last close of the previous year, as DES and WEI measure YTD
        return max(int(np.searchsorted(dates, last[:4] + "-01-01", side="left")) - 1, 0)
    start = _minus_months(last, RANGE_MONTHS[rng])
    return min(int(np.searchsorted(dates, start, side="left")), len(dates) - 1)


def _aligned(df: pd.DataFrame, dates: np.ndarray) -> np.ndarray:
    """A comparison's adjusted close on the main ticker's dates: forward-filled across its own
    holidays, NaN before its first close and after its last."""
    s = pd.Series(df["adjclose"].to_numpy(dtype=float), index=df["date"].to_numpy(dtype=str))
    s = s[~s.index.duplicated(keep="last")]
    idx = pd.Index(dates)
    out = np.array(s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx), dtype=float)  # a copy
    if len(s):
        out[dates > s.index.max()] = np.nan
    return out


def _gp(bank: Bank, cmd: Command) -> dict:
    t = cmd.ticker
    bad_range = [a for a in cmd.args if a not in RANGES and RANGE_SHAPE.fullmatch(a)]
    if bad_range:
        raise UserError(f"unknown range {bad_range[0]}; ranges: {' '.join(RANGES)}", f"{t} GP 1Y")
    ranges = [a for a in cmd.args if a in RANGES]
    cmps = list(dict.fromkeys(a for a in cmd.args if a not in RANGES))
    if len(ranges) > 1:
        raise UserError("one range at a time", f"{t} GP {ranges[0]}")
    rng = ranges[0] if ranges else "1Y"
    if len(cmps) > 3:
        raise UserError("GP compares at most 3 tickers", f"{t} GP {rng} {' '.join(cmps[:3])}")
    bad = [c for c in cmps if not TICKER.fullmatch(c)]
    if bad:
        raise UserError(f"'{bad[0]}' is not a range or a ticker", f"{t} GP {rng}")
    view = _view(bank)
    snap = view.latest(t, PRICE)
    df = _frame(snap, view) if snap is not None else None
    if df is None or df.empty:
        raise UserError(f"no prices for {t}", _ingest_hint(t))
    dates = df["date"].to_numpy(dtype=str)
    w = df.iloc[_range_start(dates, rng):].reset_index(drop=True)
    wd = w["date"].to_numpy(dtype=str)
    close = w["close"].to_numpy(dtype=float)
    adj = w["adjclose"].to_numpy(dtype=float)
    vol = pd.to_numeric(w["volume"], errors="coerce").to_numpy(dtype=float) if "volume" in w else np.full(len(w), np.nan)
    idx = downsample(len(w))
    pts = [{"date": wd[i], "close": close[i], "adj": adj[i],
            "volume": int(vol[i]) if math.isfinite(vol[i]) else None} for i in idx]
    hi, lo = int(np.argmax(close)), int(np.argmin(close))
    stats = {"first": close[0], "last": close[-1], "ret": adj[-1] / adj[0] - 1, "high": close[hi], "low": close[lo],
             "high_date": wd[hi], "low_date": wd[lo]}
    lines, missing = [], []
    for c in cmps:
        cs = view.latest(c, PRICE)
        v = _aligned(_frame(cs, view), wd) if cs is not None else np.array([])
        ok = np.flatnonzero(np.isfinite(v))
        if not len(ok):
            missing.append(c)
            continue
        lines.append((c, v, int(ok[0])))
    # Every line is 100 on one date, so the legend's returns cover one period: the range start, or the
    # first shown date on which every comparison has a close when one starts later (^GSPC from 2015 on MAX).
    b = int(idx[np.searchsorted(idx, max([0, *(first for _, _, first in lines)]))])
    rebased = [{"date": wd[i], "value": adj[i] / adj[b] * 100} for i in idx]
    compare = []
    for c, v, first in lines:
        base = v[b] if math.isfinite(v[b]) else v[first]  # no close on the common date: it ended before
        compare.append({"ticker": c, "start": wd[first], "points": [{"date": wd[i], "value": v[i] / base * 100}
                                                                    for i in idx if math.isfinite(v[i])]})
    name = _name(view, t)
    what = f"price {rng}" + (f" vs {', '.join(c['ticker'] for c in compare)}" if compare else "")
    data = {"ticker": t, "range": rng, "series": pts, "compare": compare, "rebased": rebased, "rebase_date": wd[b],
            "stats": stats, "splits": snap.context.get("splits", []), "cite": cite(snap), "kind": kind_of(bank, t),
            "missing": missing}
    return _ok(cmd, _title(t, name, what), str(wd[-1]), "chart", data)


# ---------- FA ----------
FA_ROWS = [  # (group, label, brief.FUNDAMENTALS label or None when computed, cell type)
    ("Income statement", "Revenue", "Revenue", "money"),
    ("Income statement", "Gross profit", "Gross profit", "money"),
    ("Income statement", "Operating income", "Operating income", "money"),
    ("Income statement", "Net income", "Net income", "money"),
    ("Income statement", "EPS diluted", "EPS, diluted", "number"),
    ("Income statement", "R&D", "R&D", "money"),
    ("Cash flow", "Operating cash flow", "Operating cash flow", "money"),
    ("Cash flow", "Capex", "Capex", "money"),
    ("Cash flow", "Free cash flow", None, "money"),
    ("Cash flow", "Buybacks", "Buybacks", "money"),
    ("Cash flow", "Dividends paid", "Dividends paid", "money"),
    ("Balance sheet", "Total assets", "Total assets", "money"),
    ("Balance sheet", "Total liabilities", "Total liabilities", "money"),
    ("Balance sheet", "Equity", "Shareholders' equity", "money"),
    ("Balance sheet", "Cash", "Cash", "money"),
    ("Balance sheet", "Long-term debt", "Long-term debt", "money"),
]
FA_RATIOS = [("Gross margin", "Gross profit", "Revenue"), ("Operating margin", "Operating income", "Revenue"),
             ("Net margin", "Net income", "Revenue"), ("Revenue growth", None, None), ("ROE", "Net income", "Equity")]
FA_YEARS = 10
PER_SHARE = {"EPS diluted"}


def _pick(by_date: dict[str, Fact], fy: str) -> Fact | None:
    """The fact for fiscal year end ``fy``, allowing a few days' drift between statements."""
    if fy in by_date:
        return by_date[fy]
    near = [d for d in by_date if abs(days_between(d, fy)) <= 7]
    return by_date[min(near, key=lambda d: abs(days_between(d, fy)))] if near else None


def _fa(bank: Bank, cmd: Command) -> dict:
    t = cmd.ticker
    view = _view(bank)
    src = {label: (attrs, flow) for label, attrs, flow in brief.FUNDAMENTALS}
    facts = {label: {f.as_of: f for f in metrics.annual(view, t, src[s][0], flow=src[s][1])}
             for _, label, s, _ in FA_ROWS if s}
    fys = sorted(set(facts["Revenue"]) | set(facts["Net income"]) or set(facts["Total assets"]), reverse=True)
    if not fys:
        hint = f"{t} GP" if markets.kind(t) else _ingest_hint(t)
        raise UserError(f"no SEC facts for {t}", hint)
    fys = fys[:FA_YEARS]
    book = PriceBook(bank, [t], asof=_now())  # its splits put per-share figures on today's share basis
    splits_used: set[tuple[str, float]] = set()
    rows, vals = [], {}
    for group, label, s, type_ in FA_ROWS:
        row = {"item": label, "_group": group, "_type": type_, "_cites": {}}
        for fy in fys:
            if s:
                f = _pick(facts[label], fy)
                if f is not None:
                    v = f.value
                    if label in PER_SHARE and (k := book.basis_after(t, f.known_at)) != 1.0:
                        v = v / k  # filed before a split: 9.21 pre-split is 2.30 in today's shares
                        splits_used |= {s_ for s_ in book.splits[t] if s_[0] > f.known_at[:10]}
                    row[fy], row["_cites"][fy] = v, cite(f)
            else:  # free cash flow = operating cash flow - capex, same as brain.metrics
                ocf, cap = vals.get("Operating cash flow", {}).get(fy), vals.get("Capex", {}).get(fy)
                if ocf is not None and cap is not None:
                    row[fy] = ocf - cap
        vals[label] = {fy: row[fy] for fy in fys if fy in row}
        if vals[label]:
            rows.append(row)
    rev_all = facts["Revenue"]
    for label, num, den in FA_RATIOS:
        row = {"item": label, "_group": "Ratios", "_type": "pct", "_cites": {}}
        for fy in fys:
            if num:
                n, d = vals.get(num, {}).get(fy), vals.get(den, {}).get(fy)
                ok = n is not None and d is not None and (d > 0 if label == "ROE" else d != 0)
                if ok:
                    row[fy] = n / d
            else:  # year-over-year, like brain.metrics.revenue_growth
                cur = _pick(rev_all, fy)
                prev = [d for d in rev_all if cur is not None and 330 <= days_between(d, cur.as_of) <= 400]
                if prev and rev_all[prev[-1]].value > 0:
                    row[fy] = cur.value / rev_all[prev[-1]].value - 1
        if any(fy in row for fy in fys):
            rows.append(row)
    cols = [_col("item", "Item")] + [_col(fy, fy, "money") for fy in fys]
    note = ("Annual figures from 10-K XBRL filings (SEC EDGAR): the latest-known value for each fiscal year, so "
            "restatements win. Free cash flow is operating cash flow minus capex; ratios are computed from the "
            "rows above.")
    if splits_used:
        shown = ", ".join(f"{r:g}:1 on {d}" for d, r in sorted(splits_used))
        note += (f" EPS is on today's share basis: figures filed before a later split ({shown}) are divided by it; "
                 "their citations hold the as-filed numbers.")
    name = _name(view, t)
    return _ok(cmd, _title(t, name, "financial analysis"), fys[0], "table", _table(cols, rows, note))


# ---------- CF ----------
def _form_match(form: str, want: str) -> bool:
    form = str(form).upper()
    return form == want or form.startswith(want + "/")


def _cf(bank: Bank, cmd: Command) -> dict:
    t = cmd.ticker
    view = _view(bank)
    fil = sorted(view.facts(t, "filing"), key=lambda f: f.known_at, reverse=True)
    if not fil:
        raise UserError(f"no SEC filings for {t}", f"{t} GP" if markets.kind(t) else _ingest_hint(t))
    want = " ".join(cmd.args)
    shown = [f for f in fil if _form_match(f.value, want)] if want else fil
    rows = [{"date": f.known_at[:10], "form": f.value, "desc": brief.filing_desc(f), "link": f.source.get("url"),
             "_cite": cite(f), "_url": f.source.get("url")} for f in shown[:200]]
    cols = [_col("date", "Date", "date"), _col("form", "Form"), _col("desc", "Description"), _col("link", "Document", "link")]
    note = (f"{len(rows)} of {len(shown)} {'form ' + want + ' ' if want else ''}filings, newest first; "
            f"{len(fil)} in the bank. "
            "Date is when the filing became public. Form 4 contents are not parsed yet.")
    name = _name(view, t)
    what = f"filings{' ' + want if want else ''}"
    return _ok(cmd, _title(t, name, what), rows[0]["date"] if rows else fil[0].known_at[:10], "table",
               _table(cols, rows, note))


# ---------- CN ----------
def _cn(bank: Bank, cmd: Command) -> dict:
    t = cmd.ticker
    _require(bank, t)
    view = _view(bank)
    headlines = _headlines(view, t, 50)
    cl = sorted(view.facts(t, "claim"), key=lambda f: f.known_at, reverse=True)[:50]
    record = claims.score(bank)["by_source"] if cl else {}
    rows = []
    for f in cl:
        r = record.get(f.source.get("name"))
        rows.append({"time": f.known_at, "text": f.value, "url": f.source.get("url"), "source": f.source.get("name"),
                     "tier": f.tier, "direction": int(f.context.get("direction") or 0), "cite": cite(f),
                     "record": {"n": r["n"], "hit_rate": r["hit_rate"]} if r else None})
    times = [h["time"] for h in headlines] + [c["time"] for c in rows]
    return _ok(cmd, _title(t, _name(view, t), "news"), max(times)[:10] if times else None, "news",
               {"headlines": headlines, "claims": rows})


# ---------- EV ----------
def _ev(bank: Bank, cmd: Command) -> dict:
    t = cmd.ticker
    if markets.kind(t):  # the signals rank stocks; an index or a currency ranked among them means nothing
        raise UserError(f"EV ranks stocks against the research universe; {t} is a market symbol "
                        f"({markets.kind(t)})", f"{t} GP")
    view = _view(bank)
    if view.latest(t, PRICE) is None:
        raise UserError(f"no prices for {t}", _ingest_hint(t))
    try:
        ev = brief.evidence(bank, t)
    except brief.NoEvidence as e:
        raise UserError(str(e), "INGEST UNIVERSE") from None
    return _ok(cmd, _title(t, _name(view, t), "evidence"), ev["rebalance_date"], "evidence", ev)


# ---------- BRIEF ----------
def _brief(bank: Bank, cmd: Command) -> dict:
    t = cmd.ticker
    _require(bank, t)
    view = _view(bank)
    md = brief.build(bank, t, _now())
    snap = view.latest(t, PRICE)
    return _ok(cmd, _title(t, _name(view, t), "brief"), snap.as_of if snap else view.t, "markdown", {"markdown": md})


# ---------- SHOW ----------
def _find(bank: Bank, entity: str, prefix: str) -> list[Fact]:
    return bank.find(entity, prefix)


def _show(bank: Bank, cmd: Command) -> dict:
    arg = cmd.args[0] if cmd.args else ""
    entity, _, prefix = arg.rpartition("/") if "/" in arg else (cmd.ticker or "", "", arg)
    prefix = prefix.lower()
    if not _HEX.fullmatch(prefix):
        raise UserError(f"'{arg}' is not a fact id (4-20 hex characters)", f"{cmd.ticker or 'AAPL'} DES")
    hits = _find(bank, entity, prefix) if entity else []
    if not hits:  # a citation from another screen (CURVE and ECO cite MACRO facts): look everywhere
        for e in ["MACRO", *bank.entities()]:
            if e != entity and (hits := _find(bank, e, prefix)):
                break
    if not hits:
        raise UserError(f"no fact {prefix} in the bank", f"{cmd.ticker} DES" if cmd.ticker else "HELP")
    if len(hits) > 1:
        raise UserError(f"{len(hits)} facts start with {prefix}; give more characters",
                        f"{hits[0].entity} SHOW {hits[0].id[:12]}")
    f = hits[0]
    inputs = []
    for ref in f.inputs:
        g = bank.get(ref)
        inputs.append({"ref": ref, "attr": g.attr if g else None, "value": g.value if g else None,
                       "cite": ref.partition("/")[2][:8]})
    data = {"fact": f.to_dict(), "downgraded": bank.is_downgraded(f.entity, f.id), "inputs": inputs}
    return _ok(cmd, f"{f.entity} · fact {f.id[:8]} · {f.attr}", f.as_of, "fact", data)


# ---------- WEI ----------
def _wei(bank: Bank, cmd: Command) -> dict:
    view = _view(bank)
    book = PriceBook(bank, [m["symbol"] for m in markets.MARKETS], asof=_now())
    rows, missing = [], []
    for m in markets.MARKETS:
        s = m["symbol"]
        ps = brief.price_summary(view, book, s, _per_year(s))
        if ps is None:
            missing.append(s)
            continue
        row = {"symbol": s, "name": m["name"], "last": ps["close"], "chg_1d": ps["chg_1d"], "chg_5d": ps["chg_5d"],
               "ret_1m": ps["ret_1m"], "ret_ytd": ps["ret_ytd"], "date": ps["date"], "_group": m["group"],
               "_ticker": s, "_cite": cite(book.snapshot[s])}
        if m["kind"] == "fx" and abs(ps["close"]) < 10:
            row["_digits"] = 4  # EUR/USD needs four decimals; the column default is two
        rows.append(row)
    if not rows:
        raise UserError("no world-market prices in the bank yet", "INGEST MARKETS")
    cols = [_col("symbol", "Symbol", "ticker"), _col("name", "Name"), _col("last", "Last", "number", 2),
            _col("chg_1d", "1D", "pct"), _col("chg_5d", "5D", "pct"), _col("ret_1m", "1M", "pct"),
            _col("ret_ytd", "YTD", "pct"), _col("date", "Date", "date")]
    note = ("Changes from daily closes over sessions (5D = 5 sessions, 1M = 21; crypto trades daily, so 7 and 30). "
            "YTD from the last close of the previous year.")
    if missing:
        note += f" Not in the bank yet: {', '.join(missing)}. INGEST MARKETS collects them."
    groups = [g for g in markets.GROUPS if any(r["_group"] == g for r in rows)]
    return _ok(cmd, "WEI — world markets", max(r["date"] for r in rows), "table", _table(cols, rows, note, groups))


# ---------- MOV ----------
def _universe() -> list[str]:
    try:
        return list(json.loads(config.UNIVERSE.read_text())["tickers"])
    except (OSError, ValueError, KeyError):
        return []


def _mov(bank: Bank, cmd: Command) -> dict:
    tickers = _universe()
    view = _view(bank)
    book = PriceBook(bank, tickers, asof=_now())
    stats = [(t, ps) for t in tickers if (ps := brief.price_summary(view, book, t)) is not None]
    if not stats:
        raise UserError("no prices for the research universe", "INGEST UNIVERSE")
    # The session most of the universe last closed on (the latest on a tie): one stock ingested after the
    # close must not empty the session lists, which rank only stocks whose last close is that session.
    held = Counter(ps["date"] for _, ps in stats)
    last = max(held, key=lambda d: (held[d], d))
    names = {t: _name(view, t) for t, _ in stats}

    def rows(sel, key, group, reverse, keep=lambda x: True):
        sel = sorted((x for x in sel if math.isfinite(x[1][key]) and keep(x[1][key])), key=lambda x: x[1][key],
                     reverse=reverse)[:8]
        return [{"symbol": t, "name": names[t], "last": ps["close"], "chg_1d": ps["chg_1d"], "ret_1m": ps["ret_1m"],
                 "date": ps["date"], "_group": group, "_ticker": t, "_cite": cite(book.snapshot[t])} for t, ps in sel]
    session = [x for x in stats if x[1]["date"] == last]
    out = (rows(session, "chg_1d", "Top gainers, last session", True, lambda v: v > 0)
           + rows(session, "chg_1d", "Top losers, last session", False, lambda v: v < 0)
           + rows(stats, "ret_1m", "Top 1 month", True) + rows(stats, "ret_1m", "Bottom 1 month", False))
    cols = [_col("symbol", "Symbol", "ticker"), _col("name", "Name"), _col("last", "Last", "number", 2),
            _col("chg_1d", "1D", "pct"), _col("ret_1m", "1M", "pct")]
    off = len(stats) - len(session)
    note = (f"{len(tickers)}-stock research universe (universe/us_large.json); last session {last}. "
            "Total returns (dividend-adjusted); 1M = 21 sessions. Gainers rose that session and losers fell."
            + (f" {off} stocks whose last close is not on {last} are left out of the session lists." if off else "")
            + (f" {len(tickers) - len(stats)} have no prices: INGEST UNIVERSE." if len(stats) < len(tickers) else ""))
    groups = ["Top gainers, last session", "Top losers, last session", "Top 1 month", "Bottom 1 month"]
    return _ok(cmd, "MOV — research-universe movers", max(held), "table", _table(cols, out, note, groups))


# ---------- CURVE and ECO ----------
SPREADS = [("10Y − 2Y", "10 Yr", "2 Yr"), ("10Y − 3M", "10 Yr", "3 Mo"), ("30Y − 5Y", "30 Yr", "5 Yr")]
ECO_TENORS = [("3 Mo", "Treasury 3M par yield, %"), ("2 Yr", "Treasury 2Y par yield, %"),
              ("10 Yr", "Treasury 10Y par yield, %"), ("30 Yr", "Treasury 30Y par yield, %")]


def _treasury(view: View) -> tuple[Fact, pd.DataFrame, dict] | None:
    """The latest curve snapshot and the rows for latest / 1 month ago / 1 year ago."""
    f = view.latest("MACRO", treasury.ATTR)
    df = _frame(f, view) if f is not None else None
    if df is None or df.empty:
        return None
    dates = df["date"].to_numpy(dtype=str)
    last = str(dates[-1])
    at = {"latest": len(df) - 1, "m1": _on_or_before(dates, _minus_months(last, 1)),
          "y1": _on_or_before(dates, _minus_months(last, 12))}
    return f, df, at


def _yield(df: pd.DataFrame, i: int | None, tenor: str):
    if i is None or tenor not in df.columns:
        return None
    v = float(df[tenor].iloc[i])
    return round(v, 4) if math.isfinite(v) else None


def _spread(df: pd.DataFrame, i: int | None, a: str, b: str):
    x, y = _yield(df, i, a), _yield(df, i, b)
    return round(x - y, 4) if x is not None and y is not None else None


def _curve(bank: Bank, cmd: Command) -> dict:
    tr = _treasury(_view(bank))
    if tr is None:
        raise UserError("no Treasury yield curve in the bank", "INGEST MACRO")
    f, df, at = tr
    curves = [{"label": label, "date": str(df["date"].iloc[at[k]]),
               "yields": [_yield(df, at[k], tn) for tn in treasury.TENORS]}
              for k, label in (("latest", "Latest"), ("m1", "1 month ago"), ("y1", "1 year ago")) if at[k] is not None]
    spreads = [{"label": label, **{k: _spread(df, at[k], a, b) for k in ("latest", "m1", "y1")}}
               for label, a, b in SPREADS]
    data = {"tenors": list(treasury.TENORS), "years": list(treasury.YEARS), "curves": curves, "spreads": spreads,
            "cite": cite(f)}
    return _ok(cmd, "US Treasury par yield curve", curves[0]["date"], "curve", data)


def _eco(bank: Bank, cmd: Command) -> dict:
    view = _view(bank)
    rows, dates = [], []
    tr = _treasury(view)
    if tr is not None:
        f, df, at = tr
        d = str(df["date"].iloc[-1])
        dates.append(d)
        for tenor, label in ECO_TENORS:
            rows.append({"series": label, **{k: _yield(df, at[k], tenor) for k in at}, "date": d,
                         "source": "US Treasury", "_group": "Treasury yields", "_cite": cite(f), "_url": f.source["url"]})
        for label, a, b in SPREADS[:2]:
            rows.append({"series": f"Treasury {label}, pp", **{k: _spread(df, at[k], a, b) for k in at}, "date": d,
                         "source": "US Treasury", "_group": "Treasury spreads", "_cite": cite(f), "_url": f.source["url"]})
    n_fred = 0
    for sid, label in fred.DEFAULT.items():
        f = view.latest("MACRO", f"series.fred.{sid}")
        if f is None:
            continue
        s = series.load(f.value)
        s = s[s["date"] <= view.t].reset_index(drop=True)
        if s.empty:
            continue
        sd = s["date"].to_numpy(dtype=str)
        last = str(sd[-1])
        at = {"latest": len(s) - 1, "m1": _on_or_before(sd, _minus_months(last, 1)),
              "y1": _on_or_before(sd, _minus_months(last, 12))}
        rows.append({"series": label, **{k: float(s["value"].iloc[i]) if i is not None else None for k, i in at.items()},
                     "date": last, "source": f"FRED {sid}", "_group": "FRED", "_cite": cite(f), "_url": f.source["url"]})
        dates.append(last)
        n_fred += 1
    if not rows:
        raise UserError("no macro data in the bank", "INGEST MACRO")
    cols = [_col("series", "Series"), _col("latest", "Latest", "number", 2), _col("m1", "1M ago", "number", 2),
            _col("y1", "1Y ago", "number", 2), _col("date", "Date", "date"), _col("source", "Source")]
    note = "Treasury par yields in percent; spreads in percentage points."
    if tr is None:
        note = "No Treasury curve yet: INGEST MACRO."
    note += (" FRED values are the latest vintage; series marked revised are not point-in-time." if n_fred else
             " FRED series are not in the bank: set FRED_API_KEY (free), then INGEST MACRO.")
    return _ok(cmd, "ECO — US rates and macro", max(dates), "table", _table(cols, rows, note))


# ---------- RES ----------
def _res(bank: Bank, cmd: Command) -> dict:
    from brain.research import ledger
    from brain.research import run as research_run
    recs = ledger.read() if config.LEDGER.exists() else []
    declared = {r["id"]: r for r in recs if r["op"] == "declare"}
    results = research_run.verdicts() if recs else []
    rows = [{"id": r["id"], "claim": declared.get(r["id"], {}).get("spec", {}).get("claim", ""),
             "test_net": r["test"]["mean_net_excess"], "perm_p": r["test"]["perm_p"], "bh_q": r["bh_q"],
             "verdict": r["verdict"]} for r in results]
    ran = {r["id"] for r in results}
    rows += [{"id": hid, "claim": d["spec"].get("claim", ""), "test_net": None, "perm_p": None, "bh_q": None,
              "verdict": "not run"} for hid, d in declared.items() if hid not in ran]
    cols = [_col("id", "Hypothesis"), _col("claim", "Claim"), _col("test_net", "TEST net/mo", "pct"),
            _col("perm_p", "Perm p", "number", 4), _col("bh_q", "BH q", "number", 3), _col("verdict", "Verdict", "verdict")]
    if results:
        ctl, ctx = results[0]["controls"], results[0]["context"]
        note = (f"Controls {'passed' if ctl['pass'] else 'FAILED: no verdict is valid'} (oracle TEST p = "
                f"{ctl['oracle_test_perm_p']:.4f}; {ctl['random_false_positive_rate'] * 100:.1f}% of "
                f"{ctl['n_random']} random signals reached p < 0.05). Gate: TEST net excess > 0, permutation p < 0.05, "
                f"BH q < 0.10. Survivorship: over TEST the equal-weight universe returned "
                f"{ctx['universe_ew_ann_return_test'] * 100:+.1f}% a year against SPY's "
                f"{ctx['spy_ann_return_test'] * 100:+.1f}%; members were chosen by today's market value, so any PASS "
                "is an upper bound.")
    else:
        note = "No results in the ledger yet: declare, commit, then `python -m brain research run`."
    asof = max((r["at"][:10] for r in recs if r["op"] == "result"), default=None)
    return _ok(cmd, "RES — research verdicts", asof, "table", _table(cols, rows, note, []))


# ---------- CLM ----------
def _clm(bank: Bank, cmd: Command) -> dict:
    s = claims.score(bank)
    rows = [{"source": src, "n": r["n"], "hits": r["hits"], "hit_rate": r["hit_rate"], "p_vs_coin": r["p_vs_coin"]}
            for src, r in sorted(s["by_source"].items(), key=lambda kv: (-kv[1]["n"], kv[0]))]
    cols = [_col("source", "Source"), _col("n", "Claims", "number", 0), _col("hits", "Hits", "number", 0),
            _col("hit_rate", "Hit rate", "pct"), _col("p_vs_coin", "p vs coin flip", "number", 3)]
    note = (("" if rows else "No directional claims with a completed horizon yet. ")
            + f"{len(s['pending'])} claims pending. A hit means the stock beat (or lagged) SPY in the claimed "
            "direction over the claim's horizon. Assume a coin flip until a source's record says otherwise.")
    return _ok(cmd, "CLM — web claims scored by source", today(), "table", _table(cols, rows, note, []))


HANDLERS = {"HELP": _help, "DES": _des, "GP": _gp, "FA": _fa, "CF": _cf, "CN": _cn, "EV": _ev, "BRIEF": _brief,
            "SHOW": _show, "WEI": _wei, "MOV": _mov, "CURVE": _curve, "ECO": _eco, "RES": _res, "CLM": _clm}


def run(bank: Bank, cmd: Command) -> dict:
    """Run one command. User errors come back as ``ok: false`` with a hint; bugs raise."""
    if cmd.ticker:
        cmd = dataclasses.replace(cmd, ticker=cmd.ticker.upper())
        if not TICKER.fullmatch(cmd.ticker):
            return clean(_err(cmd, f"'{cmd.ticker}' is not a ticker", "HELP"))
    if cmd.fn in SERVER_ONLY:
        return clean(_err(cmd, SERVER_ONLY[cmd.fn], "HELP"))
    handler = HANDLERS.get(cmd.fn)
    if handler is None:
        return clean(_err(cmd, f"unknown function '{cmd.fn}'", suggest(cmd.fn)))
    if FUNCTIONS[cmd.fn]["scope"] == "security" and not cmd.ticker:
        return clean(_err(cmd, f"{cmd.fn} needs a ticker, e.g. AAPL {cmd.fn}", f"AAPL {cmd.fn}"))
    try:
        return clean(handler(bank, cmd))
    except UserError as e:
        return clean(_err(cmd, e.error, e.hint))


def run_line(bank: Bank, line: str) -> dict:
    """Parse and run a command line; a parse error is an ``ok: false`` result too."""
    try:
        cmd = parse(line)
    except CommandError as e:
        return {"ok": False, "fn": None, "ticker": None, "args": [], "error": e.message, "hint": e.hint}
    return run(bank, cmd)
