"""Derived numbers, computed only through a View so they are point-in-time.

The brief and the backtests call the same functions, so a number in a brief is
the number the engine was tested on. Each metric returns the facts it used;
``to_fact`` writes it back to the bank as a derived fact citing them (one hop at
a time, each intermediate result written down and re-readable).

Share basis: Yahoo prices are back-adjusted for splits to today's basis. A share
count reported on date k is moved to that basis by multiplying by every split
after k. The product shares * close then equals the market value at t exactly;
splits after t cancel out, so no future information enters the value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain import series
from brain.bank import Bank, Fact, View
from brain.sources.sec import REVENUE
from brain.timeutil import days_between

STALE_DAYS = 550  # an annual figure older than this is not "current"


@dataclass
class M:
    value: float
    as_of: str
    inputs: list[Fact] = field(default_factory=list)
    note: str = ""


class PriceBook:
    """Latest price snapshot per ticker. Reads go through a View's date."""

    def __init__(self, bank: Bank, tickers, asof: str | None = None):
        v = bank.view(asof)
        self.dates, self.close, self.adj, self.splits, self.snapshot = {}, {}, {}, {}, {}
        for t in tickers:
            t = t.upper()
            f = v.latest(t, "series.price_daily")
            if f is None:
                continue
            df = series.load(f.value)
            self.dates[t] = df["date"].to_numpy(dtype=str)
            self.close[t] = df["close"].to_numpy(dtype=float)
            self.adj[t] = df["adjclose"].to_numpy(dtype=float)
            self.splits[t] = [(d, float(r)) for d, r in f.context.get("splits", [])]
            self.snapshot[t] = f

    def has(self, t: str) -> bool:
        return t.upper() in self.dates

    def _idx(self, view: View, t: str) -> int:
        return int(np.searchsorted(self.dates[t], view.t, side="right")) - 1

    def close_on(self, view: View, t: str):
        t = t.upper()
        if t not in self.dates:
            return None
        i = self._idx(view, t)
        if i < 0:
            return None
        return self.dates[t][i], float(self.close[t][i])

    def adj_upto(self, view: View, t: str) -> tuple[np.ndarray, np.ndarray]:
        t = t.upper()
        i = self._idx(view, t)
        return self.dates[t][: i + 1], self.adj[t][: i + 1]

    def basis_after(self, t: str, k: str) -> float:
        return float(np.prod([r for d, r in self.splits.get(t.upper(), []) if d > k[:10]] or [1.0]))


# ---------- selecting facts ----------
def _dur(f: Fact):
    return f.context.get("duration_days")


def is_annual_flow(f: Fact) -> bool:
    d = _dur(f)
    return str(f.context.get("form", "")).startswith("10-K") and d is not None and 330 <= d <= 400


def is_fy_instant(f: Fact) -> bool:
    return str(f.context.get("form", "")).startswith("10-K") and "start" not in f.context


def annual(view: View, entity: str, attrs, flow: bool = True) -> list[Fact]:
    """Latest-known value for each fiscal year, merged across alternative concepts."""
    attrs = [attrs] if isinstance(attrs, str) else attrs
    pick = is_annual_flow if flow else is_fy_instant
    best: dict[str, Fact] = {}
    for a in attrs:
        for f in view.periods(entity, a, pick):
            if f.as_of not in best or f.known_at > best[f.as_of].known_at:
                best[f.as_of] = f
    return [best[k] for k in sorted(best)]


def latest_shares(view: View, entity: str) -> Fact | None:
    ps = view.periods(entity, "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding")
    if not ps:
        return None
    last = ps[-1].as_of
    same = [f for f in ps if f.as_of == last]
    return min(same, key=lambda f: (_dur(f) or 0, f.known_at))


def _fresh(view: View, f: Fact) -> bool:
    return days_between(f.as_of, view.t) <= STALE_DAYS


# ---------- metrics ----------
def market_cap(view: View, entity: str, book: PriceBook) -> M | None:
    sh = latest_shares(view, entity)
    px = book.close_on(view, entity)
    if sh is None or px is None or not _fresh(view, sh):
        return None
    d, close = px
    mc = sh.value * book.basis_after(entity, sh.known_at) * close
    return M(mc, d, [sh, book.snapshot[entity.upper()]], f"diluted shares x close on {d}")


def earnings_yield(view: View, entity: str, book: PriceBook) -> M | None:
    ni = annual(view, entity, "us-gaap:NetIncomeLoss")
    mc = market_cap(view, entity, book)
    if not ni or mc is None or not _fresh(view, ni[-1]) or mc.value <= 0:
        return None
    ey = ni[-1].value / mc.value
    if not -1.0 <= ey <= 1.0:  # implausible for a listed company: unit or share-class mismatch
        return None
    return M(ey, mc.as_of, [ni[-1], *mc.inputs], "last fiscal-year net income / market value")


def asset_growth(view: View, entity: str) -> M | None:
    a = annual(view, entity, "us-gaap:Assets", flow=False)
    if len(a) < 2 or not _fresh(view, a[-1]):
        return None
    cur, prev = a[-1], a[-2]
    if not 300 <= days_between(prev.as_of, cur.as_of) <= 430 or prev.value <= 0:
        return None
    return M(cur.value / prev.value - 1, cur.as_of, [cur, prev], "total assets, fiscal year over year")


def _growth(series_: list[Fact]) -> M | None:
    if len(series_) < 2:
        return None
    cur, prev = series_[-1], series_[-2]
    if not 330 <= days_between(prev.as_of, cur.as_of) <= 400 or not prev.value:
        return None
    return M(cur.value / abs(prev.value) - 1 if prev.value > 0 else float("nan"), cur.as_of, [cur, prev])


def revenue_growth(view: View, entity: str) -> M | None:
    return _growth(annual(view, entity, REVENUE))


def ratio(view: View, entity: str, num, den, flow_num=True, flow_den=True, note="") -> M | None:
    n = annual(view, entity, num, flow=flow_num)
    d = annual(view, entity, den, flow=flow_den)
    if not n or not d:
        return None
    dd = {f.as_of: f for f in d}
    for f in reversed(n):
        if f.as_of in dd and dd[f.as_of].value:
            return M(f.value / dd[f.as_of].value, f.as_of, [f, dd[f.as_of]], note)
    return None


def free_cash_flow(view: View, entity: str) -> M | None:
    ocf = annual(view, entity, "us-gaap:NetCashProvidedByUsedInOperatingActivities")
    cap = {f.as_of: f for f in annual(view, entity, "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment")}
    for f in reversed(ocf):
        if f.as_of in cap:
            return M(f.value - cap[f.as_of].value, f.as_of, [f, cap[f.as_of]], "operating cash flow - capex")
    return None


def to_fact(entity: str, name: str, m: M, unit: str | None = None) -> Fact:
    known = max(f.known_at for f in m.inputs)
    return Fact(entity=entity, attr=f"derived.{name}", value=float(m.value), unit=unit, as_of=m.as_of,
                known_at=known, tier="derived", source={"name": "brain.metrics", "function": name},
                context={"note": m.note} if m.note else {}, inputs=tuple(f.ref for f in m.inputs))
