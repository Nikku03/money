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
import importlib.util
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from brain import brief, claims, config, markets, metrics, series
from brain.bank import Bank, Fact, View
from brain.metrics import PriceBook
from brain.sources import fred, treasury
from brain.terminal.commands import (FUNCTIONS, KEYS, RANGE_SHAPE, RANGES, TICKER, Command, CommandError, dec_line,
                                     parse, suggest)
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


def _col(key: str, label: str, type_: str = "text", digits: int | None = None, nowrap: bool = False) -> dict:
    c = {"key": key, "label": label, "type": type_}
    if digits is not None:
        c["digits"] = digits
    if nowrap:  # long text is cut with an ellipsis (full text in the tooltip) rather than wrapped
        c["nowrap"] = True
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
            "Date is when the filing became public. INS lists the open-market trades in Forms 4 and 5.")
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
    record = claims.score(bank)["by_source"] if cl else {}  # every call, overlapping ones included
    weighed = claims.track_record(bank, _now()) if cl else {}  # the record D1 weighs
    rows = []
    for f in cl:
        r = record.get(f.source.get("name"))
        rows.append({"time": f.known_at, "text": f.value, "url": f.source.get("url"), "source": f.source.get("name"),
                     "tier": f.tier, "direction": int(f.context.get("direction") or 0), "cite": cite(f),
                     "record": {"n": r["n"], "hit_rate": r["hit_rate"]} if r else None,
                     "gate_key": claims.source_key(f), "gate": _gate_row(weighed.get(claims.source_key(f)))})
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


def _scrub(obj, words: set[str]):
    """``obj`` with each whole-word mention of ``words`` (a withheld aircraft's registration and hex) in its
    strings replaced by ``[withheld]``. Case is ignored: a model may write ``n2xc``."""
    pat = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(map(re.escape, sorted(words, key=len, reverse=True)))
                     + r")(?![A-Za-z0-9])", re.I)

    def go(x):
        if isinstance(x, str):
            return pat.sub("[withheld]", x)
        if isinstance(x, dict):
            return {k: go(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [go(v) for v in x]
        return x
    return go(obj)


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
    hidden = _withholder(bank)
    if hidden(f):
        raise UserError(f"fact {prefix} is about an aircraft that is no longer tracked (opted out, or in the "
                        "FAA's LADD/PIA privacy programs), so it is not shown", f"{f.entity} DES")
    # A fact made from such an aircraft fact (a decision citing a landing alert, a judgment citing a flight)
    # is shown without it: the input is left out and counted, as DEC counts it; a judgment's reason that cites
    # it goes with it; and the tail's registration and hex are taken out of the rest of the fact's text, where
    # a model may have named them. Its bare ref stays in the fact's inputs: it names no tail, and SHOW of it
    # is refused above.
    from brain import watch
    inputs, gone, tails = [], set(), set()
    for ref in f.inputs:
        g = bank.get(ref)
        if g is not None and hidden(g):
            gone.add(ref)
            tails.update(str(x) for x in watch.aircraft_of(bank, g) if x)
            continue
        inputs.append({"ref": ref, "attr": g.attr if g else None, "value": g.value if g else None,
                       "cite": ref.partition("/")[2][:8]})
    fact = f.to_dict()
    if gone:
        ctx = dict(fact.get("context") or {})
        if isinstance(ctx.get("reasons"), list):
            ctx["reasons"] = [{**r, "text": "[withheld: it cites an aircraft that is no longer tracked]"}
                              if isinstance(r, dict) and gone & set(map(str, r.get("cites") or [])) else r
                              for r in ctx["reasons"]]
        fact["context"] = ctx
    if tails:
        fact.update(value=_scrub(fact["value"], tails), context=_scrub(fact["context"], tails))
    data = {"fact": fact, "downgraded": bank.is_downgraded(f.entity, f.id), "inputs": inputs, "withheld": len(gone)}
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
def _universe_name(path) -> str:
    """'universe/us_broad.json' -> 'us_broad'."""
    from brain.research.hypotheses import COMMON
    return Path(str(path or COMMON["universe"])).stem


def _res(bank: Bank, cmd: Command) -> dict:
    from brain.research import ledger
    from brain.research import run as research_run
    recs = ledger.read() if config.LEDGER.exists() else []
    declared = {r["id"]: r for r in recs if r["op"] == "declare"}
    results = research_run.verdicts() if recs else []
    rows = [{"id": r["id"], "universe": _universe_name(r.get("universe")),
             "claim": declared.get(r["id"], {}).get("spec", {}).get("claim", ""),
             "test_net": r["test"]["mean_net_excess"], "perm_p": r["test"]["perm_p"], "bh_q": r["bh_q"],
             "verdict": r["verdict"]} for r in results]
    ran = {r["id"] for r in results}
    rows += [{"id": hid, "universe": _universe_name(d["spec"].get("universe")), "claim": d["spec"].get("claim", ""),
              "test_net": None, "perm_p": None, "bh_q": None, "verdict": "not run"}
             for hid, d in declared.items() if hid not in ran]
    cols = [_col("id", "Hypothesis"), _col("universe", "Universe"), _col("claim", "Claim"),
            _col("test_net", "TEST net/mo", "pct"), _col("perm_p", "Perm p", "number", 4),
            _col("bh_q", "BH q", "number", 3), _col("verdict", "Verdict", "verdict")]
    if results:
        # Controls and survivorship belong to a universe's panel: say each one's once.
        by_universe: dict[str, dict] = {}
        for r in results:
            by_universe.setdefault(_universe_name(r.get("universe")), r)
        parts = []
        for name, r in by_universe.items():
            ctl, ctx = r["controls"], r["context"]
            parts.append(f"Controls {'passed' if ctl['pass'] else 'FAILED: no verdict on it is valid'} on {name} "
                         f"(oracle TEST p = {ctl['oracle_test_perm_p']:.4f}; "
                         f"{ctl['random_false_positive_rate'] * 100:.1f}% of {ctl['n_random']} random signals reached "
                         f"p < 0.05"
                         + (f"; held fixed per stock, not gated: {ctl['persistent_random_false_positive_rate'] * 100:.1f}%"
                            if ctl.get("persistent_random_false_positive_rate") is not None else "")
                         + f"); over TEST its equal-weight universe returned "
                         f"{ctx['universe_ew_ann_return_test'] * 100:+.1f}% a year against SPY's "
                         f"{ctx['spy_ann_return_test'] * 100:+.1f}%.")
        note = (" ".join(parts) + " Gate: TEST net excess > 0, permutation p < 0.05, BH q < 0.10 (over every "
                "hypothesis with a result). The random-signal control redraws its signals every month; for a signal "
                "that persists from month to month the null is narrower and its p-value too small (the fixed-per-stock "
                "rate, where a run recorded it, says by how much). Universe members were chosen by today's market "
                "value, so any PASS is an upper bound.")
    else:
        note = "No results in the ledger yet: declare, commit, then `python -m brain research run`."
    asof = max((r["at"][:10] for r in recs if r["op"] == "result"), default=None)
    return _ok(cmd, "RES — research verdicts", asof, "table", _table(cols, rows, note, []))


# ---------- CLM ----------
CLM_WEIGHED = "The record D1 weighs"
CLM_EVERY = "Every call, overlapping"


def _clm(bank: Bank, cmd: Command) -> dict:
    """Two records per source: the one D1 weighs (``claims.track_record``: independent bets, a one-sided p and
    a false-discovery correction; claims and the model's judgments) and every claim counted on its own
    (``claims.score``, overlapping calls included, a two-sided p), which no decision reads."""
    from brain.decide import POLICY
    s = claims.score(bank)
    tr = claims.track_record(bank, _now())
    rows = [{"source": k, "bets": r["bets"], "hits": r["hits"], "hit_rate": r["hit_rate"], "p": r["p"], "q": r["q"],
             "proven": "yes" if r["proven"] else "no", "weight": r["weight"], "why": r["why"], "calls": r["calls"],
             "_group": CLM_WEIGHED}
            for k, r in sorted(tr.items(), key=lambda kv: (-kv[1]["weight"], -kv[1]["bets"], kv[0]))]
    rows += [{"source": src, "calls": r["n"], "hits": r["hits"], "hit_rate": r["hit_rate"], "p": r["p_vs_coin"],
              "_group": CLM_EVERY}
             for src, r in sorted(s["by_source"].items(), key=lambda kv: (-kv[1]["n"], kv[0]))]
    note = (("" if s["by_source"] else "No directional claims with a completed horizon yet. ")
            + f"{len(s['pending'])} claims pending. A hit means the stock beat (or lagged) SPY in the claimed "
            "direction over the claim's horizon. " + _gate_note(POLICY["source_gate"])
            + " Every call, overlapping: each scored claim counted on its own, whatever its horizon, with a "
            "two-sided p (brain score-claims); calls in one window rise and fall together, so this count overstates "
            "the evidence and no decision reads it. Assume a coin flip until a source's record says otherwise.")
    return _ok(cmd, "CLM — web claims scored by source", today(), "table",
               _table(_gate_cols(), rows, note, [CLM_WEIGHED, CLM_EVERY]))


# ---------- shared by the v2 functions ----------
def _money(x: float | None) -> str:
    """$4.95T, $1.79M, $502,855: the note's way of writing a dollar amount."""
    if x is None or not math.isfinite(x):
        return "n/a"
    a = abs(x)
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= div:
            return f"${x / div:,.2f}{unit}"
    return f"${x:,.0f}"


def _research_tickers() -> set[str]:
    """Tickers of every research universe file (the ones `brain insiders` reads by default)."""
    from brain.universe import BROAD
    out: set[str] = set()
    for p in (config.UNIVERSE, BROAD):
        try:
            out |= {str(t).upper() for t in json.loads(p.read_text())["tickers"]}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return out


def _research_line(signal: str) -> str:
    """One sentence on the tested hypothesis behind a signal, quoted from the ledger as it stands: the
    latest current one on the signal with a result (a withdrawn one gives way to its successor,
    hypotheses.current), else the latest declared."""
    from brain.research import ledger
    from brain.research.hypotheses import current
    ids = [h["id"] for h in current(ledger.withdrawn()) if h["signal"] == signal]
    if not ids:
        return ""
    results = {}
    if config.LEDGER.exists():
        from brain.research import run as research_run
        results = {r["id"]: r for r in research_run.verdicts()}
    hid = next((i for i in reversed(ids) if i in results), ids[-1])
    v = results.get(hid)
    if v is None:
        return f" Research: {hid} has no result yet (RES)."
    net, p = v["test"].get("mean_net_excess"), v["test"].get("perm_p")
    nums = []
    if isinstance(net, (int, float)) and math.isfinite(net):
        nums.append(f"TEST net excess {net * 100:+.2f}% a month")
    if isinstance(p, (int, float)) and math.isfinite(p):
        nums.append(f"permutation p {p:.3f}")
    return f" Research: {hid} {v['verdict']}" + (f" ({', '.join(nums)})" if nums else "") + "; see RES."


def _alt_entities(bank: Bank, prefix: str) -> list[str]:
    """Bank entities whose name starts with ``prefix`` (AREA., CAM.), from their shards' first records."""
    out = []
    for p in sorted(bank.root.glob(prefix + "*.jsonl")):
        with p.open(errors="replace") as fh:
            line = fh.readline()
        try:
            rec = json.loads(line)
            name = rec.get("fact", rec).get("entity")
        except (ValueError, AttributeError):
            name = None
        name = name if isinstance(name, str) and name.upper().startswith(prefix) else p.stem
        out.append(name.upper())
    return list(dict.fromkeys(out))


def _samples(view: View, entity: str, attr: str) -> list[Fact]:
    return sorted(view.facts(entity, attr), key=lambda f: (f.known_at, f.id))


def _zscore(facts: list[Fact], same) -> tuple[float | None, int]:
    """The newest sample against the previous comparable ones, the way the alert rules read it
    (``brain.watch``: up to 30 samples, at least 10, a spread above zero). (z, samples compared)."""
    from brain import watch
    if not facts:
        return None, 0
    last = facts[-1]
    prior = [g for g in facts[:-1] if same(g, last)][-watch.Z_WINDOW:]
    if len(prior) < watch.Z_MIN:
        return None, len(prior)
    vals = np.array([float(g.value) for g in prior])
    sd = float(vals.std(ddof=1))
    if not sd > 0:
        return None, len(prior)
    return (float(last.value) - float(vals.mean())) / sd, len(prior)


def _same_ships(g: Fact, f: Fact) -> bool:
    """Comparable samples: the same window, box and basis (what the count counts), as the alert rule reads them."""
    return all(g.context.get(k) == f.context.get(k) for k in ("window_s", "bbox", "basis"))


def _same_camera(g: Fact, f: Fact) -> bool:
    """Comparable samples: the same detector, threshold and decode (brain.vision.DECODE), as the alert rule reads them."""
    return all(g.context.get(k) == f.context.get(k) for k in ("detector", "threshold", "decode"))


def _count_chart(cmd: Command, title: str, entity: str, name: str, facts: list[Fact], unit: str, parts, same) -> dict:
    """View ``chart`` with ``kind: "count"``: one point per sample (vessels in an area, vehicles in a frame)."""
    vals = np.array([float(f.value) for f in facts])
    idx = downsample(len(facts))
    pts = [{"date": facts[i].known_at, "value": facts[i].value, "cite": cite(facts[i]),
            "parts": {k: facts[i].context.get(k) for k in parts}} for i in idx]
    hi, lo = int(np.argmax(vals)), int(np.argmin(vals))
    z, compared = _zscore(facts, same)
    stats = {"first": vals[0], "last": vals[-1], "high": vals[hi], "low": vals[lo], "high_date": facts[hi].known_at,
             "low_date": facts[lo].known_at, "mean": float(vals.mean()), "n": len(facts), "z": z, "z_samples": compared}
    data = {"entity": entity, "name": name, "kind": "count", "unit": unit, "series": pts, "stats": stats,
            "cite": cite(facts[-1]), "source": dict(facts[-1].source)}
    return _ok(cmd, title, facts[-1].known_at[:10], "chart", data)


def _watchlist() -> dict:
    from brain import watch
    try:
        return watch.load()
    except ValueError as e:  # a file that is not valid JSON: say so, never treat it as empty
        raise UserError(str(e), "") from None


def _not_market(t: str, what: str) -> None:
    if markets.kind(t):
        raise UserError(f"{what}; {t} is a market symbol ({markets.kind(t)})", f"{t} GP")


def _company(bank: Bank, t: str, what: str) -> None:
    """INS and JET are about a company that files with the SEC: not an index, not an ETF's prices alone."""
    _not_market(t, what)
    _require(bank, t)
    if kind_of(bank, t) != "equity":
        raise UserError(f"no SEC identity for {t} in the bank; {what}", _ingest_hint(t))


# ---------- INS ----------
INS_ROWS = 200
ROLE_NAMES = {"officer": "officer", "director": "director", "ten_percent_owner": "10% owner", "other": "other"}


def _role(c: dict) -> str:
    rel = ", ".join(ROLE_NAMES.get(r, r) for r in str(c.get("relationship") or "").split(",") if r)
    return rel + (f" ({c['title']})" if c.get("title") else "")


def _ins(bank: Bank, cmd: Command) -> dict:
    from brain.sources import insiders
    t = cmd.ticker
    _company(bank, t, "INS lists insider trades in a company's own shares")
    view = _view(bank)
    fs = sorted(insiders.trades(view, t), key=lambda f: (f.known_at[:10], f.as_of, f.id), reverse=True)
    if not fs and not insiders.dataset_read_for(view, t):  # read, and none filed: an empty table
        hint = "INGEST INSIDERS" if t in _research_tickers() else f"python -m brain insiders {t}"
        raise UserError(f"no insider trades for {t} in the bank", hint)
    rows = [{"filed": f.known_at[:10], "traded": f.as_of, "owner": f.context.get("owner"), "relationship": _role(f.context),
             "code": f.context.get("code"), "shares": f.value, "price": f.context.get("price"),
             "value": f.context.get("value_usd"), "ownership": f.context.get("ownership"),
             "_cite": cite(f), "_url": f.source.get("url")} for f in fs[:INS_ROWS]]
    tot = insiders.buy_sell_totals(view, t, 90)
    cut = insiders.since(view, 90)
    od = sum(float(f.context.get("value_usd") or 0.0) for f in fs
             if f.known_at > cut and f.context.get("code") == "P" and insiders.is_officer_or_director(f))
    note = (f"Filed in the 90 days to {view.t}: insiders bought {_money(tot['buy_usd'])} in {tot['buys']} trade"
            f"{'s' if tot['buys'] != 1 else ''} and sold {_money(tot['sell_usd'])} in {tot['sells']}; officers and "
            f"directors bought {_money(od)} (what the insider_buying signal measures). "
            if fs else f"No open-market purchases or sales by {t}'s insiders are on file. ")
    if len(fs) > INS_ROWS:
        note += f"The newest {INS_ROWS} of {len(fs)} trades are shown. "
    note += ("Open-market purchases (code P) and sales (S) from SEC Forms 4 and 5, newest filing first; awards, "
             "option exercises, gifts, tax withholding and amendments are left out. Insiders appear in their public "
             "role, as the filing states it. D/I: held directly or indirectly. History comes from SEC's quarterly "
             "data sets (INGEST INSIDERS); Form 4s filed this quarter from live reads (INGEST WATCH reads them for "
             "watched tickers).") + _research_line("insider_buying")
    cols = [_col("filed", "Filed", "date"), _col("traded", "Traded", "date"), _col("owner", "Insider"),
            _col("relationship", "Relationship"), _col("code", "Code"), _col("shares", "Shares", "number", 0),
            _col("price", "Price", "number", 2), _col("value", "Value", "money"), _col("ownership", "D/I")]
    what = "insider trades" + (f" · 90 days: bought {_money(tot['buy_usd'])}, sold {_money(tot['sell_usd'])}" if fs else "")
    return _ok(cmd, _title(t, _name(view, t), what), rows[0]["filed"] if rows else None, "table",
               _table(cols, rows, note, []))


# ---------- ERN ----------
def _ern(bank: Bank, cmd: Command) -> dict:
    from brain import events
    t = cmd.ticker
    _not_market(t, "ERN reads a company's quarterly reports")
    view = _view(bank)
    book = PriceBook(bank, [t, "SPY"], asof=_now())
    table = events.earnings_table(view, t, book)
    if not table:
        raise UserError(f"no quarterly EPS for {t} in the bank", _ingest_hint(t))
    evs = {e.ref: e for e in events.earnings_events(view, t)}
    rows = []
    for r in table:
        eps_refs = [x for x in r["cite"] if x not in evs]
        ev = next((evs[x] for x in r["cite"] if x in evs), None)
        cites = {"eps": eps_refs[0].partition("/")[2][:8]} if eps_refs else {}
        if ev is not None:
            cites.update({k: cite(ev) for k in ("release", "car_3d", "drift_20d", "drift_60d")})
        rows.append({"period_end": r["period_end"], "filed": r["filed"], "release": ev.known_at[:10] if ev else None,
                     "eps": r["eps"], "eps_from": "FY − 9M" if len(eps_refs) > 1 else "",
                     "eps_yoy": r["eps_yoy"], "sue": r["sue"], "car_3d": r["car_3d"], "drift_20d": r["drift_20d"],
                     "drift_60d": r["drift_60d"], "_cite": cites.get("eps"), "_cites": cites,
                     "_url": ev.source.get("url") if ev else None, "_refs": list(r["cite"])})
    cols = [_col("period_end", "Quarter end", "date"), _col("filed", "EPS filed", "date"),
            _col("release", "Release", "date"), _col("eps", "EPS", "number", 2), _col("eps_from", "Derived", "flag"),
            _col("eps_yoy", "EPS YoY", "pct", 1), _col("sue", "SUE", "signed", 2), _col("car_3d", "3-day", "pct"),
            _col("drift_20d", "Drift 20d", "pct"), _col("drift_60d", "Drift 60d", "pct")]
    note = ("Quarterly EPS from 10-Q and 10-K XBRL (diluted, else basic), on today's share basis; a fourth quarter "
            "not reported on its own is the fiscal year minus the nine months (flagged FY − 9M). EPS YoY: the "
            "change from the same quarter a year earlier, over that quarter's absolute EPS. SUE: the same change "
            "divided by its standard deviation over the 8 quarters before (at least 6 needed). Release: the last "
            "earnings 8-K (item 2.02) within 90 days after the quarter ended that was public by the day the EPS was "
            "filed (else the first after it). Reactions are the stock's total "
            "return minus SPY's: 3-day from the session before the release to the session after, drifts over "
            "sessions +2 to +21 and +2 to +61; blank until the window has closed.")
    for sym in (t, "SPY"):
        if not book.has(sym):
            note += f" No prices for {sym}, so no reactions: INGEST {sym}."
    note += _research_line("earnings_reaction") + _research_line("sue")
    return _ok(cmd, _title(t, _name(view, t), "earnings"), rows[0]["filed"], "table", _table(cols, rows, note, []))


# ---------- JET ----------
def _aircraft(view: View, t: str) -> tuple[list[Fact], list[Fact], dict | None]:
    """(registered aircraft, recorded flights, the registry link) for a company. Tails in the opt-out file
    and aircraft the poller found in LADD/PIA are left out. A link made against a newer registry download
    than the facts says which tails the company still owns."""
    from brain import watch
    from brain.sources import aviation
    try:
        link = json.loads((config.home() / "aviation_links.json").read_text()).get(t)
    except (OSError, ValueError, AttributeError):
        link = None
    hidden = watch.untracked()
    planes = [f for f in aviation.registered(view, t) if not hidden(f.value, f.context.get("hex"))]
    if planes and isinstance(link, dict) and str(link.get("registry_fetched_at") or "") > max(f.known_at for f in planes):
        owned = {str(a.get("reg")) for a in link.get("aircraft") or []}
        planes = [f for f in planes if f.value in owned]  # sold since: the newer download no longer lists it
    flights = [f for f in aviation.flights(view, t) if not hidden(f.context.get("reg"), f.context.get("hex"))]
    return planes, flights, link if isinstance(link, dict) else None


def _place(code, name) -> str | None:
    if not code and not name:
        return None
    return f"{code or ''} {name or ''}".strip()


JET_FLIGHTS = 50


def _jet(bank: Bank, cmd: Command) -> dict:
    t = cmd.ticker
    _company(bank, t, "JET lists aircraft a company owns")
    view = _view(bank)
    planes, flights, link = _aircraft(view, t)
    if not planes and not flights and link is None:
        raise UserError(f"{t} has not been looked up in the FAA aircraft registry", f"python -m brain jets link {t}")
    model = {f.value: f.context.get("model") for f in planes}
    rows = [{"reg": f.value, "model": f.context.get("model"),
             "year": str(f.context["year"]) if f.context.get("year") is not None else None,
             "_group": "Registered aircraft", "_cite": cite(f), "_url": f.source.get("url")} for f in planes]
    for f in flights[:JET_FLIGHTS]:
        x = f.context
        rows.append({"reg": x.get("reg") or x.get("hex"), "model": model.get(x.get("reg")),
                     "from": _place(x.get("from"), x.get("from_name")), "to": _place(x.get("to") or f.value, x.get("to_name")),
                     "departed": x.get("departed"), "arrived": x.get("arrived") or f.known_at,
                     "_group": "Flights, newest first", "_cite": cite(f), "_url": f.source.get("url")})
    name = _name(view, t) or t
    reg_date = max([f.as_of for f in planes] + [str((link or {}).get("registry_fetched_at") or "")[:10]])
    if planes:
        note = (f"{len(planes)} aircraft registered to {name} itself in the FAA registry"
                + (f" (download of {reg_date})" if reg_date else "") + ": a corporate registrant whose name matches the "
                "company's SEC name word for word, legal form included (an LLC registrant only for a company that is "
                "itself an LLC). ")
    else:
        note = f"No aircraft are registered in {name}'s own name in the FAA registry" + \
               (f" (download of {reg_date}). " if reg_date else ". ")
    try:
        watched = t in _watchlist()["aircraft_of"]
    except UserError:  # a broken watchlist file is WL's error to report; the registry still answers
        watched = False
    if watched:
        note += (f"{len(flights)} flights recorded" + (f", the newest {JET_FLIGHTS} shown" if len(flights) > JET_FLIGHTS else "")
                 + "; INGEST WATCH polls positions. ")
    else:
        note += (f"Flights are recorded only while {t} is on the watchlist's aircraft list "
                 f"(python -m brain watch add aircraft {t}; then INGEST WATCH polls). ")
    note += ("Individuals, trusts and other LLCs are never matched; aircraft in the FAA's privacy programs (LADD, PIA) "
             "and tails in aviation_optout.txt are left out. Flights are airport to airport with times; who was on "
             "board is not known and not inferred. Sources: FAA Aircraft Registry (public domain), adsb.lol (ODbL), "
             "OurAirports.")
    cols = [_col("reg", "Registration"), _col("model", "Model"), _col("year", "Year")]
    if flights:
        cols += [_col("from", "From"), _col("to", "To"), _col("departed", "Departed (UTC)", "date"),
                 _col("arrived", "Arrived (UTC)", "date")]
    dates = [r["arrived"][:10] for r in rows if r.get("arrived")] + ([reg_date] if reg_date else [])
    return _ok(cmd, _title(t, name, "aircraft"), max(dates) if dates else None, "table",
               _table(cols, rows, note, ["Registered aircraft", "Flights, newest first"]))


# ---------- SHIP ----------
AREA_NAME = re.compile(r"[A-Z0-9][A-Z0-9_-]{0,39}")
SHIP_PARTS = ("cargo", "tanker", "other", "untyped", "anchored", "moored", "underway")
AIS_KEY_HINT = ("AIS sampling needs a free aisstream.io key: export AISSTREAM_API_KEY=..., restart the terminal, "
                "then INGEST WATCH")


def _ship(bank: Bank, cmd: Command) -> dict:
    from brain.sources import shipping
    if len(cmd.args) > 1:
        raise UserError("SHIP takes one area at a time", f"SHIP {cmd.args[0]}")
    view = _view(bank)
    w = _watchlist()
    watched = [a["name"] for a in w["areas"]]
    builtin = [a["name"] for a in shipping.AREAS]
    if cmd.args:
        name = cmd.args[0]
        if not AREA_NAME.fullmatch(name):
            raise UserError(f"'{name}' is not an area name", "SHIP")
        facts = _samples(view, f"AREA.{name}", "ships.count")
        if not facts:
            if name in watched:  # INGEST WATCH jobs inherit the server's environment: without the key they skip areas
                hint = "INGEST WATCH" if os.environ.get("AISSTREAM_API_KEY", "").strip() else AIS_KEY_HINT
            elif name in builtin:
                hint = f"python -m brain watch add area {name}"
            else:
                raise UserError(f"unknown area {name}; built-in areas: {', '.join(builtin)}", "SHIP")
            raise UserError(f"no vessel counts for {name} yet", hint)
        return _count_chart(cmd, f"SHIP — {name} · vessels", f"AREA.{name}", name, facts, "vessels", SHIP_PARTS,
                            _same_ships)
    sampled = [e[len("AREA."):] for e in _alt_entities(bank, "AREA.")]
    rows = []
    for name in list(dict.fromkeys(watched + sampled)):
        facts = _samples(view, f"AREA.{name}", "ships.count")
        row = {"area": name, "_group": "Watched areas" if name in watched else "Other sampled areas",
               "_entity": f"AREA.{name}", "_cmds": {"area": f"SHIP {name}"}, "n": len(facts)}
        if facts:
            f = facts[-1]
            z, _ = _zscore(facts, _same_ships)
            row.update({"vessels": f.value, "z": z, **{k: f.context.get(k) for k in SHIP_PARTS},
                        "window_s": f.context.get("window_s"), "at": f.known_at, "_cite": cite(f),
                        "_url": f.source.get("url")})
        rows.append(row)
    cols = [_col("area", "Area"), _col("vessels", "Vessels", "number", 0), _col("z", "z", "signed", 1)] + \
           [_col(k, k.capitalize(), "number", 0) for k in SHIP_PARTS] + \
           [_col("window_s", "Window s", "number", 0), _col("at", "Sampled (UTC)", "date"), _col("n", "Samples", "number", 0)]
    note = ("Vessels: the commercial vessels seen in each area's box during one AIS listening window (aisstream.io), "
            "cargo (AIS types 70-79) plus tanker (80-89); anchored, moored and underway are of those. Other typed "
            "vessels (tugs, passenger ships, fishing) and vessels whose type was not heard (untyped) are tallied "
            "beside the count, not in it; pleasure and sailing craft are never counted, and only the counts are "
            "stored, never a vessel's identity or track. z: the newest count against up to 30 earlier samples with "
            "the same window, box and basis (10 needed). SHIP NAME charts an area; INGEST WATCH samples the watched "
            "ones (a free AISSTREAM_API_KEY is needed); python -m brain watch add area NAME watches one.")
    if not rows:
        note = f"No port areas are watched or sampled yet. Built-in areas: {', '.join(builtin)}. " + note
    times = [r["at"] for r in rows if r.get("at")]
    return _ok(cmd, "SHIP — port areas", max(times)[:10] if times else None, "table",
               _table(cols, rows, note, ["Watched areas", "Other sampled areas"]))


# ---------- CAM ----------
CAMERA_ID = re.compile(r"[A-Z]+\.[A-Z0-9._-]{1,100}")
CAM_PARTS = ("car", "truck", "bus", "motorcycle")
OPENCV_HINT = "vehicle counting needs OpenCV: pip install opencv-python-headless, restart the terminal"


def _cam(bank: Bank, cmd: Command) -> dict:
    if len(cmd.args) > 1:
        raise UserError("CAM takes one camera id at a time", f"CAM {cmd.args[0]}")
    view = _view(bank)
    w = _watchlist()
    if cmd.args:
        cid = cmd.args[0].upper()
        cid = cid[len("CAM."):] if cid.startswith("CAM.") else cid
        if not CAMERA_ID.fullmatch(cid):
            raise UserError(f"'{cmd.args[0]}' is not a camera id (e.g. TFL.00002.00865)", "CAM")
        facts = _samples(view, f"CAM.{cid}", "camera.vehicles")
        if not facts:
            watched = any(c.upper() == cid for s in w["sites"] for c in s["cameras"])
            hint = "INGEST WATCH" if watched else f"python -m brain cams sample {cid}"
            if importlib.util.find_spec("cv2") is None:  # both ways of sampling need it; jobs share our packages
                hint = f"{OPENCV_HINT}, then {hint}"
            raise UserError(f"no vehicle counts for camera {cid} yet", hint)
        label = facts[-1].context.get("name") or cid
        return _count_chart(cmd, f"CAM — {cid} · {label} · vehicles", f"CAM.{cid}", label, facts, "vehicles",
                            CAM_PARTS, _same_camera)
    rows, seen = [], set()

    def row(cid: str, group: str) -> dict:
        facts = _samples(view, f"CAM.{cid}", "camera.vehicles")
        r = {"camera": cid, "_group": group, "_entity": f"CAM.{cid}", "_cmds": {"camera": f"CAM {cid}"},
             "n": len(facts)}
        if facts:
            f = facts[-1]
            z, _ = _zscore(facts, _same_camera)
            r.update({"name": f.context.get("name"), "vehicles": f.value, "z": z,
                      **{k: f.context.get(k) for k in CAM_PARTS}, "at": f.known_at, "_cite": cite(f),
                      "_url": f.source.get("url")})
        return r

    groups = []
    for s in w["sites"]:
        group = s["name"] + (f" · {s['ticker']}" if s.get("ticker") else "")
        groups.append(group)
        for c in s["cameras"]:
            cid = c.upper()
            if cid not in seen:
                seen.add(cid)
                rows.append(row(cid, group))
    for e in _alt_entities(bank, "CAM."):
        cid = e[len("CAM."):]
        if cid not in seen:
            seen.add(cid)
            r = row(cid, "Other sampled cameras")
            if r["n"]:  # an unwatched camera whose only samples were provider placeholders (downgraded): nothing to show
                rows.append(r)
    cols = [_col("camera", "Camera"), _col("name", "Name"), _col("vehicles", "Vehicles", "number", 0),
            _col("z", "z", "signed", 1)] + [_col(k, k.capitalize(), "number", 0) for k in CAM_PARTS] + \
           [_col("at", "Sampled (UTC)", "date"), _col("n", "Samples", "number", 0)]
    note = ("Vehicles counted in one frame of each camera by an open detector (YOLOX, COCO classes car, truck, bus "
            "and motorcycle, score 0.5 or more). Frames are processed in memory and never stored; no people, faces "
            "or number plates are detected. Only official open traffic cameras: Transport for London JamCams "
            "(Powered by TfL Open Data) and Caltrans CCTV. z: the newest count against up to 30 earlier samples by "
            "the same detector, threshold and decode (10 needed). CAM ID charts a camera; INGEST WATCH samples the watched sites "
            "(needs opencv-python-headless); python -m brain watch add site NAME --near LAT LON watches a place.")
    if not rows:
        note = "No camera sites are watched or sampled yet. " + note
    times = [r["at"] for r in rows if r.get("at")]
    return _ok(cmd, "CAM — traffic cameras", max(times)[:10] if times else None, "table",
               _table(cols, rows, note, groups + ["Other sampled cameras"]))


# ---------- WL ----------
def _on_rule(r: dict, k: str) -> bool:
    return r.get(k) is not None and r.get(k) is not False


def _rules_text(r: dict) -> str:
    def on(k):
        return _on_rule(r, k)
    parts = []
    if on("price_move"):
        parts.append(f"a 1-day move of {float(r['price_move']) * 100:g}% or more")
    if on("insider_buy_usd"):
        parts.append(f"insider buying by an officer or director of {_money(float(r['insider_buy_usd']))} or more")
    if on("earnings_car"):
        parts.append(f"each earnings release (high when the 3-day reaction is {float(r['earnings_car']) * 100:g}% "
                     "or more)")
    if r.get("forms"):
        parts.append(f"filings of forms {', '.join(map(str, r['forms']))}")
    if on("zscore"):
        parts.append(f"vessel and vehicle counts {float(r['zscore']):g} sd or more from usual")
    if on("headlines"):
        parts.append("new headlines (info)")
    parts.append("company aircraft landings")
    return "; ".join(parts)


def _wl(bank: Bank, cmd: Command, done: str = "") -> dict:
    from brain import watch
    w = _watchlist()
    view = _view(bank)
    latest: dict[str, list[Fact]] = {}
    for f in watch.alerts(bank, entities=watch.watched_entities(w)):  # newest first
        latest.setdefault(f.entity, []).append(f)

    def alert(entity: str, keep=lambda f: True) -> dict:
        """The newest notice or high alert of the row (headlines are info and would bury them), else the newest."""
        fs = [f for f in latest.get(entity.upper(), []) if keep(f)]
        a = next((f for f in fs if f.context.get("severity") in ("notice", "high")), fs[0] if fs else None)
        return {"alert": a.value if a else None, "alert_at": a.known_at if a else None,
                "_cites": {"alert": cite(a)} if a else {}}

    from brain import decide
    rows = []
    book = PriceBook(bank, w["tickers"], asof=_now())
    for t in w["tickers"]:
        ps = brief.price_summary(view, book, t, _per_year(t)) if book.has(t) else None
        a = alert(t, lambda f: f.context.get("rule") != "aircraft")
        dec = decide.latest(bank, t)  # the latest paper decision; empty when none was recorded
        if dec is not None:
            a["_cites"] = {**a["_cites"], "decision": cite(dec)}
        rows.append({"item": t, "name": _name(view, t), "measure": "close", "last": ps["close"] if ps else None,
                     "chg_1d": ps["chg_1d"] if ps else None, "date": ps["date"] if ps else None,
                     "decision": dec.value if dec is not None else None, **a, "_group": "Tickers", "_ticker": t,
                     "_cmds": {"decision": dec_line(t)} if dec_line(t) else {}, "_cite": cite(book.snapshot.get(t))})
    for t in w["aircraft_of"]:
        planes, flights, _ = _aircraft(view, t)
        last = flights[0] if flights else None
        rows.append({"item": t, "name": _name(view, t), "measure": "aircraft", "last": len(planes), "_digits": 0,
                     "date": (last.context.get("arrived") or last.known_at) if last else None,
                     **alert(t, lambda f: f.context.get("rule") == "aircraft"), "_group": "Company aircraft",
                     "_ticker": t, "_cmds": {"item": f"{t} JET"}, "_cite": cite(last)})
    for a in w["areas"]:
        e = f"AREA.{a['name']}"
        facts = _samples(view, e, "ships.count")
        f = facts[-1] if facts else None
        rows.append({"item": a["name"], "name": "port area", "measure": "vessels", "last": f.value if f else None,
                     "_digits": 0, "date": f.known_at if f else None, **alert(e), "_group": "Port areas",
                     "_entity": e, "_cmds": {"item": f"SHIP {a['name']}"}, "_cite": cite(f)})
    for s in w["sites"]:
        for c in s["cameras"]:
            e = f"CAM.{c.upper()}"
            facts = _samples(view, e, "camera.vehicles")
            f = facts[-1] if facts else None
            label = s["name"] + (f" · {s['ticker']}" if s.get("ticker") else "")  # CAM names the camera itself
            rows.append({"item": c.upper(), "name": label, "measure": "vehicles",
                         "last": f.value if f else None, "_digits": 0, "date": f.known_at if f else None, **alert(e),
                         "_group": "Camera sites", "_entity": e, "_cmds": {"item": f"CAM {c.upper()}"},
                         "_cite": cite(f)})
    try:
        unacked = watch.unacked(bank)
    except ValueError as e:  # the alert state file cannot be read: say which, as ALRT does
        raise UserError(str(e), "") from None
    shown = [f for f in unacked if not watch.is_info(f)]  # what the top bar counts and ALRT lists
    # what a narrow pane shows first: the item, its last value and move, the alert; then the details
    cols = [_col("item", "Item", "ticker"), _col("name", "Name"), _col("last", "Last", "number", 2),
            _col("chg_1d", "1D", "pct"), _col("decision", "Decision"), _col("alert", "Latest alert", nowrap=True),
            _col("alert_at", "Alert time (UTC)", "date"), _col("measure", "Measure"), _col("date", "As of", "date")]
    if rows:
        info = len(unacked) - len(shown)
        note = (f"{len(shown)} unacknowledged alert{'s' if len(shown) != 1 else ''} (ALRT)"
                + (f", and {info} info (ALRT ALL)" if info else "") + f". Rules: {_rules_text(w['rules'])}"
                + ("" if _on_rule(w["rules"], "headlines") else
                   "; headlines are off (python -m brain watch rule headlines on adds each as an info alert)")
                + ". INGEST WATCH collects what is new for every row and checks the rules; python -m brain watch run "
                "repeats that every 15 minutes. Decision: the latest paper decision under policy D1 (T DEC shows it, "
                "T DEC NEW decides afresh); a change raises a notice. ")
    else:
        note = "The watchlist is empty. WL ADD AAPL adds a ticker. "
    note += ("WL ADD T and WL DEL T edit the tickers; company aircraft, port areas and camera sites are added from "
             "the shell: python -m brain watch add aircraft T | area NAME | site NAME --near LAT LON.")
    title = "WL — watchlist" + (f" · {done}" if done else "")
    return _ok(cmd, title, today(), "table",
               _table(cols, rows, note, ["Tickers", "Company aircraft", "Port areas", "Camera sites"]))


def _wl_edit(bank: Bank, cmd: Command) -> dict:
    from brain import watch
    verb, vals = cmd.args[0], list(dict.fromkeys(cmd.args[1:]))
    if not vals:
        raise UserError(f"WL {verb} needs tickers, e.g. WL {verb} AAPL", "WL")
    bad = [v for v in vals if not TICKER.fullmatch(v)]
    if bad:
        raise UserError(f"'{bad[0]}' is not a ticker", "WL")
    before = set(_watchlist()["tickers"])  # a broken file is an error here too, never overwritten
    try:
        watch.acked()  # so is a broken alert state, before anything changes: the answer is WL, which reads it
    except ValueError as e:
        raise UserError(str(e), "") from None
    missing, already = [], []
    for v in vals:
        if verb == "ADD" and v.upper() in before:
            already.append(v)  # watch.add would change nothing
            continue
        try:
            watch.add("ticker", v) if verb == "ADD" else watch.remove("ticker", v)
        except KeyError:
            missing.append(v)
        except ValueError as e:
            raise UserError(str(e), "WL") from None
    changed = [v for v in vals if v not in missing and v not in already]
    parts = ([f"{'added' if verb == 'ADD' else 'removed'} {', '.join(changed)}"] if changed else []) + \
            ([f"already on it: {', '.join(already)}"] if already else []) + \
            ([f"not on it: {', '.join(missing)}"] if missing else [])
    return _wl(bank, cmd, "; ".join(parts))


# ---------- ALRT ----------
ALRT_ROWS = 200
ACK_ID = re.compile(r"[0-9A-Fa-f]{4,20}")


def _alrt(bank: Bank, cmd: Command, done: str = "", everything: bool = False) -> dict:
    """ALRT: notice and high alerts; ALRT ALL (``everything``) adds the info ones (headlines)."""
    from brain import watch
    try:
        acked = watch.acked()
    except ValueError as e:
        raise UserError(str(e), "") from None
    # newest first; within one check (one known_at), high before notice before info, so headlines bury nothing
    every = sorted(watch.alerts(bank), key=lambda f: (f.known_at, watch.SEVERITY.get(f.context.get("severity"), 0), f.id),
                   reverse=True)
    fs = every if everything else [f for f in every if not watch.is_info(f)]
    unacked = [f for f in fs if f.id not in acked]
    info = [f for f in every if watch.is_info(f)]
    info_new = sum(f.id not in acked for f in info)
    rows = []
    for f in fs[:ALRT_ROWS]:
        new = f.id not in acked
        e = f.entity
        row = {"new": "NEW" if new else "", "at": f.known_at, "severity": f.context.get("severity"), "entity": e,
               "rule": f.context.get("rule"), "message": f.value, "ack": "ACK" if new else None,
               "_cite": cite(f), "_entity": e, "_cmds": {}}
        if new:
            row["_cmds"]["ack"] = f"ALRT ACK {f.id[:12]}"
        else:
            row["_dim"] = True
        if e.startswith("AREA."):
            row["_cmds"]["entity"] = f"SHIP {e[len('AREA.'):]}"
        elif e.startswith("CAM."):
            row["_cmds"]["entity"] = f"CAM {e[len('CAM.'):]}"
        else:
            row["_ticker"] = e
        rows.append(row)
    cols = [_col("new", "", "flag"), _col("at", "Time (UTC)", "date"), _col("severity", "Severity", "severity"),
            _col("entity", "Entity", "ticker"), _col("rule", "Rule"), _col("message", "Alert"), _col("ack", "")]
    if fs:
        note = (f"{len(unacked)} of {len(fs)} unacknowledged" + (f"; the newest {ALRT_ROWS} shown" if len(fs) > ALRT_ROWS
                                                                 else "") + ". ACK (ALRT ACK ID) acknowledges one, "
                "ALRT ACK ALL every one, info included. ")
    elif info:
        note = "No notice or high alerts. "
    else:
        note = "No alerts yet. Put tickers on the watchlist (WL ADD AAPL); INGEST WATCH then collects and checks. "
    if info and not everything:
        note += (f"{len(info)} info alert{'s' if len(info) != 1 else ''} (headlines; {info_new} unacknowledged) "
                 f"{'are' if len(info) != 1 else 'is'} not listed or counted: ALRT ALL lists them. ")
    note += ("Alerts come from the watchlist's rules (WL), each citing the facts that set it off. An alert says where "
             "to look; it is not evidence that a price will move.")
    title = "ALRT — alerts" + (" · all, info included" if everything else "") + (f" · {done}" if done else "")
    data = {**_table(cols, rows, note, []), "unacked": len(unacked),
            "high": sum(f.context.get("severity") == "high" for f in unacked), "info": info_new}
    return _ok(cmd, title, fs[0].known_at[:10] if fs else today(), "table", data)


def _alrt_ack(bank: Bank, cmd: Command) -> dict:
    from brain import watch
    rest = cmd.args[1:]
    if len(rest) != 1 or not (rest[0] == "ALL" or ACK_ID.fullmatch(rest[0])):
        raise UserError("ALRT ACK takes one alert id (4 or more hex characters, as shown) or ALL", "ALRT")
    try:
        if rest[0] == "ALL":
            n = watch.ack_all(bank)
            done = f"acknowledged {n} alert{'s' if n != 1 else ''}"
        else:
            f = watch.ack(bank, rest[0].lower())
            done = f"acknowledged {f.id[:8]}"
    except (KeyError, ValueError) as e:
        raise UserError(str(e.args[0]) if e.args else str(e), "ALRT") from None
    return _alrt(bank, cmd, done)


# ---------- DEC ----------
WORD = {1: "up", -1: "down", 0: "none"}
DEC_ATTRS = ("claim", "judgment", "alert")  # what a decision cites besides its price snapshot


def _lean_text(ln: dict | None) -> str:
    """The sources' lean in words, as `brain decide` prints it: always unproven, never evidence."""
    ln = ln or {}
    up, down, n = int(ln.get("up") or 0), int(ln.get("down") or 0), int(ln.get("sources") or 0)
    word = WORD.get(int(ln.get("direction") or 0), "none")
    if not n:
        return "none (no source has a call in the window)"
    if up and down:
        return f"{word}, {up} up and {down} down of {n} sources"
    if up or down:
        return f"{word}, {max(up, down)} of {n} source{'s' if n != 1 else ''}"
    return f"none, no source of {n} points a way"


def _facts_by_ref(bank: Bank, refs: list[str]) -> dict[str, Fact]:
    """The facts behind a decision's refs, read by attribute (claims, judgments, alerts) rather than one
    lookup each: ``bank.get`` reads the whole shard for every ref. Refs not found that way are looked up."""
    out: dict[str, Fact] = {}
    by_entity: dict[str, set[str]] = {}
    for r in refs:
        e, _, fid = str(r).partition("/")
        if e and fid:
            by_entity.setdefault(e.upper(), set()).add(fid)
    for e, ids in by_entity.items():
        have = set(bank.attrs(e))
        for a in DEC_ATTRS:
            if a in have:
                out.update({f.ref: f for f in bank.facts(e, a, include_downgraded=True) if f.id in ids})
    for r in refs:
        if r not in out and (g := bank.get(r)) is not None:
            out[r] = g
    return out


def _ref_cites(refs, facts: dict[str, Fact], hidden) -> tuple[list[dict], int]:
    """[{"entity", "cite"}] for the refs SHOW would show, and how many it withholds (an aircraft no longer
    tracked: opted out, or in LADD/PIA)."""
    shown, withheld = [], 0
    for r in refs or []:
        e, _, fid = str(r).partition("/")
        f = facts.get(r)
        if f is not None and hidden(f):
            withheld += 1
            continue
        shown.append({"entity": e.upper(), "cite": fid[:8]})
    return shown, withheld


def _withholder(bank: Bank):
    """A test ``(fact) -> bool``: a registration, flight or aircraft alert of a tail in aviation_optout.txt or
    found in LADD/PIA, never shown, whether or not it has been withdrawn yet. The opt-out list is read once,
    at the first aircraft fact."""
    from brain import watch
    out = None

    def hidden(f: Fact) -> bool:
        nonlocal out
        if not watch.is_aircraft(f):
            return False
        out = out or watch.untracked()
        return out(*watch.aircraft_of(bank, f))
    return hidden


def _outcome_cols() -> list[dict]:
    return [_col("excess", "Excess vs SPY", "pct"), _col("hit", "Hit"), _col("status", "Scored")]


def _outcome(row: dict | None) -> dict:
    """A decision's outcome from ``decide.score``: its excess over SPY across 21 sessions, hit or miss for a
    BUY or SELL, and whether it counts as its own bet (``scored``), falls inside an earlier one's window
    (``overlapping``), has not closed (``pending``) or closed with no close for the stock (``unpriced``)."""
    if row is None:
        return {"excess": None, "hit": None, "status": "pending"}
    hit = row.get("hit")
    return {"excess": row.get("excess"), "hit": None if hit is None else "hit" if hit else "miss",
            "status": row.get("stream_status") or "pending"}


def _dec_one(bank: Bank, cmd: Command) -> dict:
    from brain import decide
    t = cmd.ticker
    f = decide.latest(bank, t)
    if f is None:
        ok, why = decide.scorable(t)
        if not ok:
            raise UserError(f"no decisions for {t}: {why}", "DEC")
        raise UserError(f"no decision recorded for {t} yet", dec_line(t, new=True) or "DEC")
    d = decide.as_dict(f)
    view = _view(bank)
    refs = [r for i in d.get("items") or [] for r in i.get("refs") or []] + list(d.get("context_refs") or [])
    facts = _facts_by_ref(bank, refs)
    hidden = _withholder(bank)
    items, withheld = [], 0
    for i in d.get("items") or []:
        cites, w = _ref_cites(i.get("refs"), facts, hidden)
        withheld += w
        items.append({"kind": i.get("kind"), "id": i.get("id"), "direction": int(i.get("direction") or 0),
                      "weight": float(i.get("weight") or 0.0), "why": i.get("why") or "", "cites": cites,
                      "record": i.get("record")})
    whys = {c.get("ref"): c.get("why") for c in d.get("context_notes") or [] if isinstance(c, dict)}
    context = []
    for r in d.get("context_refs") or []:
        cites, w = _ref_cites([r], facts, hidden)
        withheld += w
        if cites:
            g = facts.get(r)
            context.append({**cites[0], "attr": g.attr if g else None, "why": whys.get(r) or ""})
    price = d.get("price")
    if isinstance(price, dict) and price.get("ref"):
        price = {"date": price.get("date"), "close": price.get("close"), "cite": str(price["ref"]).partition("/")[2][:8]}
    judgment = d.get("judgment")
    if isinstance(judgment, dict):
        judgment = {"cite": str(judgment["ref"]).partition("/")[2][:8] if judgment.get("ref") else None,
                    "error": judgment.get("error")}
    ln = d.get("lean") or {}
    data = {"ticker": t, "name": _name(view, t), "action": d["action"], "label": decide.LABEL.get(d["action"], d["action"]),
            "direction": int(d.get("direction") or 0), "score": d.get("score"), "threshold": d.get("threshold"),
            "signal_score": d.get("signal_score"), "source_term": d.get("source_term"),
            "policy": d.get("policy"), "policy_sha": d.get("policy_sha"),
            "policy_current": decide.KNOWN_POLICIES.get(d.get("policy")) == d.get("policy_sha"),
            "horizon_days": d.get("horizon_days"), "as_of": d.get("as_of"), "known_at": d.get("known_at"),
            "retrieved_at": f.retrieved_at, "cite": cite(f), "price": price, "items": items, "context": context, "withheld": withheld,
            "lean": {"direction": int(ln.get("direction") or 0), "up": int(ln.get("up") or 0),
                     "down": int(ln.get("down") or 0), "sources": int(ln.get("sources") or 0), "text": _lean_text(ln)},
            "explanation": d.get("explanation") or "", "explained_by": d.get("explained_by") or "template",
            "explain_error": d.get("explain_error"), "unverified_numbers": list(d.get("unverified_numbers") or []),
            "judgment": judgment, "read": d.get("read"), "notes": list(d.get("notes") or []),
            "history": _dec_history(bank, t, decide)}
    return _ok(cmd, _title(t, data["name"], f"decision · {d['action']}"), d.get("as_of"), "decision", data)


def _dec_history(bank: Bank, t: str, decide) -> dict:
    """Every recorded decision on the stock, newest first, with its outcome from ``decide.score``."""
    scored = {r["ref"]: r for r in decide.score(bank, _now())["rows"]}
    rows = []
    for f in decide.history(bank, t):
        c = f.context
        sha = c.get("policy_sha")
        pid = c.get("policy") or f.source.get("policy")
        rows.append({"as_of": f.as_of, "action": f.value, "score": c.get("score"),
                     "lean": _lean_text(c.get("lean")), **_outcome(scored.get(f.ref)),
                     "policy": pid if decide.KNOWN_POLICIES.get(pid) == sha else f"{pid}?{sha}", "_cite": cite(f)})
    cols = [_col("as_of", "Date", "date"), _col("action", "Action"), _col("score", "Score", "signed", 2),
            _col("lean", "Lean (not evidence)"), *_outcome_cols(), _col("policy", "Policy")]
    note = (f"Scored against SPY over {decide.POLICY['horizon_days']} trading days from the first session after each "
            "decision. A decision inside an earlier one's window is overlapping: it is not a bet of its own. "
            "Pending: the window has not closed. Unpriced: it closed, but the bank has no close for the stock "
            "then (ingest it); it is scored once the prices are in.")
    return _table(cols, rows, note, [])


def _dec_all(bank: Bank, cmd: Command) -> dict:
    from brain import decide
    sc = decide.score(bank, _now())
    # every stock with a decision: score reads them all (decide.tickers_with_decisions), and asking twice costs
    # a pass over every shard's index under the server's bank lock
    tickers = sorted({r["ticker"] for r in sc["rows"]})
    rows = []
    for t in tickers:
        f = decide.latest(bank, t)
        if f is None:
            continue
        c = f.context
        rows.append({"ticker": t, "action": f.value, "score": c.get("score"), "lean": _lean_text(c.get("lean")),
                     "as_of": f.as_of, "n": len(decide.history(bank, t)), "_ticker": t,
                     "_cmds": {"ticker": dec_line(t)} if dec_line(t) else {}, "_cite": cite(f)})
    tr = claims.track_record(bank, _now())
    gate = sc.get("gate") or decide.POLICY["source_gate"]
    latest = _table([_col("ticker", "Stock", "ticker"), _col("action", "Action"), _col("score", "Score", "signed", 2),
                     _col("lean", "Lean (not evidence)"), _col("as_of", "Date", "date"),
                     _col("n", "Decisions", "number", 0)], rows,
                    "" if rows else "No decisions recorded yet. T DEC NEW makes one (a background job, paper only).", [])
    card = []
    for key, pol in sc["policies"].items():
        group = f"{key} · sha {pol['sha']}" + ("" if "?" not in key else " · not the code's: the policy was edited")
        for stream, label in (("actions", "BUY/SELL actions"), ("lean", "Lean (not evidence)")):
            r = pol[stream]
            card.append({"stream": label, "bets": r["bets"], "hits": r["hits"], "hit_rate": r["hit_rate"], "p": r["p"],
                         "lower": r["lower"], "mse": r["mean_signed_excess"], "dropped": r["dropped"],
                         "proven": "yes" if r["proven"] else "no", "_group": group})
        ne = pol["no_edge"]
        card.append({"stream": "NO_EDGE (what the stocks did)", "bets": ne["bets"], "mse": ne["mean_excess"],
                     "_group": group})
    pending = "; ".join(f"{k}: {p['pending']} pending"
                        + (f", {p['unpriced']} with no price past their exit" if p.get("unpriced") else "")
                        for k, p in sc["policies"].items())
    scorecard = _table(
        [_col("stream", "Stream"), _col("bets", "Bets", "number", 0), _col("hits", "Hits", "number", 0),
         _col("hit_rate", "Hit rate", "pct", 0), _col("p", "p vs coin", "number", 3),
         _col("lower", "Lower bound", "number", 2), _col("mse", "Mean excess", "pct"),
         _col("dropped", "Overlapping", "number", 0), _col("proven", "Proven")], card,
        f"Each policy's decisions as independent bets against SPY over {decide.POLICY['horizon_days']} trading days "
        f"({pending}). BUY/SELL and the lean are scored as hits (mean excess signed by the call); NO_EDGE shows what "
        "the stocks did after it, its mean excess, never a hit rate. Proven needs "
        f"{gate['min_bets']} bets, p < {gate['p_lt']} and a positive mean signed excess, with at most "
        f"{claims.UNPRICED_MAX_SHARE:.0%} of the decisions past their exit lacking a price.")
    src = [{"source": k, "bets": r["bets"], "hits": r["hits"], "hit_rate": r["hit_rate"], "p": r["p"], "q": r["q"],
            "proven": "yes" if r["proven"] else "no", "weight": r["weight"], "why": r["why"], "calls": r["calls"]}
           for k, r in sorted(tr.items(), key=lambda kv: (-kv[1]["weight"], -kv[1]["bets"], kv[0]))]
    sources = _table(_gate_cols(), src, _gate_note(gate) if src else "No directional calls (claims or judgments) yet.", [])
    note = ("Paper decisions under a declared policy; nothing here places orders or is financial advice. "
            "T DEC shows one decision with its evidence; T DEC NEW decides afresh in a background job.")
    asof = max((r["as_of"] for r in rows), default=None)
    return _ok(cmd, "DEC — decisions, scorecard and source records", asof, "decisions",
               {"latest": latest, "scorecard": scorecard, "sources": sources, "note": note,
                "policy": decide.POLICY["id"], "policy_sha": decide.POLICY_SHA})


def _gate_cols() -> list[dict]:
    """The columns of a track_record row, the record D1 weighs."""
    return [_col("source", "Source"), _col("bets", "Bets", "number", 0), _col("hits", "Hits", "number", 0),
            _col("hit_rate", "Hit rate", "pct", 0), _col("p", "p vs coin", "number", 3), _col("q", "BH q", "number", 3),
            _col("proven", "Proven"), _col("weight", "Weight", "number", 2), _col("why", "Why", nowrap=True),
            _col("calls", "Calls", "number", 0)]


def _gate_note(gate: dict) -> str:
    return (f"The record D1 weighs (claims.track_record): a source's calls on the next 21 sessions, scored from the "
            f"first session after the bank held each one and grouped into independent bets (calls in one window are "
            f"one bet). Proven needs {gate['min_bets']} bets, a one-sided p < {gate['p_lt']} against a coin, "
            f"Benjamini-Hochberg q < {gate['bh_q']} across every source and a positive mean signed excess, with at "
            f"most {claims.UNPRICED_MAX_SHARE:.0%} of its calls past their exit lacking a price for their stock; its "
            f"weight is max(0, 2L - 1) from the {1 - gate['alpha_lower']:.0%} lower bound L of the hit rate.")


def _gate_row(r: dict | None) -> dict | None:
    """The part of a track_record row a screen shows beside a claim."""
    if not r:
        return None
    return {k: r.get(k) for k in ("bets", "hits", "hit_rate", "q", "proven", "weight", "why")}


def _dec(bank: Bank, cmd: Command) -> dict:
    if cmd.args:
        raise UserError("DEC takes no arguments, or NEW", f"{cmd.ticker} DEC" if cmd.ticker else "DEC")
    return _dec_one(bank, cmd) if cmd.ticker else _dec_all(bank, cmd)


# ---------- writes ----------
# function -> first argument that changes state; DEC NEW starts a job (the server runs it, never data.run)
WRITES = {"WL": ("ADD", "DEL"), "ALRT": ("ACK",), "DEC": ("NEW",)}


def is_write(cmd: Command) -> bool:
    """Commands that change local state (the watchlist, acknowledged alerts) or start a decision job (DEC
    NEW). The server runs them only for its own command line, never for the analyst's read-only tools."""
    return bool(cmd.args) and cmd.args[0] in WRITES.get(cmd.fn, ())


def _wl_or_edit(bank: Bank, cmd: Command) -> dict:
    if not cmd.args:
        return _wl(bank, cmd)
    if cmd.args[0] in WRITES["WL"]:
        return _wl_edit(bank, cmd)
    raise UserError("WL takes no arguments, or ADD or DEL and tickers", "WL")


def _alrt_or_ack(bank: Bank, cmd: Command) -> dict:
    if not cmd.args:
        return _alrt(bank, cmd)
    if cmd.args == ["ALL"]:
        return _alrt(bank, cmd, everything=True)
    if cmd.args[0] == "ACK":
        return _alrt_ack(bank, cmd)
    raise UserError("ALRT takes no arguments, or ALL, or ACK ID, or ACK ALL", "ALRT")


HANDLERS = {"HELP": _help, "DES": _des, "GP": _gp, "FA": _fa, "CF": _cf, "CN": _cn, "EV": _ev, "BRIEF": _brief,
            "INS": _ins, "ERN": _ern, "JET": _jet, "SHOW": _show, "WEI": _wei, "MOV": _mov, "CURVE": _curve,
            "ECO": _eco, "RES": _res, "CLM": _clm, "SHIP": _ship, "CAM": _cam, "WL": _wl_or_edit, "ALRT": _alrt_or_ack,
            "DEC": _dec}


def run(bank: Bank, cmd: Command, writes: bool = False) -> dict:
    """Run one command. User errors come back as ``ok: false`` with a hint; bugs raise. Commands that
    change the watchlist or acknowledge alerts (``is_write``) run only with ``writes=True``: the
    server passes it for its own command line, never for the analyst's read-only tools."""
    if cmd.ticker:
        cmd = dataclasses.replace(cmd, ticker=cmd.ticker.upper())
        if not TICKER.fullmatch(cmd.ticker):
            return clean(_err(cmd, f"'{cmd.ticker}' is not a ticker", "HELP"))
    if cmd.fn in SERVER_ONLY:
        return clean(_err(cmd, SERVER_ONLY[cmd.fn], "HELP"))
    if cmd.fn == "DEC" and is_write(cmd):  # a job, with or without writes: only the server starts one
        return clean(_err(cmd, "DEC NEW starts a background decision job; it runs only from the terminal's own "
                               "command line", f"{cmd.ticker} DEC" if cmd.ticker else "DEC"))
    if is_write(cmd) and not writes:
        return clean(_err(cmd, f"{cmd.fn} {cmd.args[0]} changes the {'watchlist' if cmd.fn == 'WL' else 'alerts'}; "
                               "it runs only from the terminal's own command line", cmd.fn))
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
