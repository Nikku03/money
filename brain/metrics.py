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
from datetime import date, timedelta

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
    """Latest-known value for each fiscal year. Alternative concepts are merged by priority (their
    order in ``attrs``): a year takes the first concept that reports it, so a company that files
    both LongTermDebt and LongTermDebtNoncurrent does not alternate between them year by year."""
    attrs = [attrs] if isinstance(attrs, str) else attrs
    pick = is_annual_flow if flow else is_fy_instant
    best: dict[str, Fact] = {}
    for a in attrs:
        mine: dict[str, Fact] = {}
        for f in view.periods(entity, a, pick):
            if f.as_of not in mine or f.known_at > mine[f.as_of].known_at:
                mine[f.as_of] = f
        for k, f in mine.items():
            best.setdefault(k, f)
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
    s = annual(view, entity, REVENUE)
    return _growth(s) if s and _fresh(view, s[-1]) else None


def ratio(view: View, entity: str, num, den, flow_num=True, flow_den=True, note="") -> M | None:
    """Latest fiscal year both sides report; None when that year is stale (not current)."""
    n = annual(view, entity, num, flow=flow_num)
    d = annual(view, entity, den, flow=flow_den)
    if not n or not d:
        return None
    dd = {f.as_of: f for f in d}
    for f in reversed(n):
        if f.as_of in dd and dd[f.as_of].value:
            return M(f.value / dd[f.as_of].value, f.as_of, [f, dd[f.as_of]], note) if _fresh(view, f) else None
    return None


def free_cash_flow(view: View, entity: str) -> M | None:
    ocf = annual(view, entity, "us-gaap:NetCashProvidedByUsedInOperatingActivities")
    cap = {f.as_of: f for f in annual(view, entity, "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment")}
    for f in reversed(ocf):
        if f.as_of in cap:
            if not _fresh(view, f):
                return None
            return M(f.value - cap[f.as_of].value, f.as_of, [f, cap[f.as_of]], "operating cash flow - capex")
    return None


def to_fact(entity: str, name: str, m: M, unit: str | None = None) -> Fact:
    known = max(f.known_at for f in m.inputs)
    return Fact(entity=entity, attr=f"derived.{name}", value=float(m.value), unit=unit, as_of=m.as_of,
                known_at=known, tier="derived", source={"name": "brain.metrics", "function": name},
                context={"note": m.note} if m.note else {}, inputs=tuple(f.ref for f in m.inputs))


# ---------- quarters and trailing twelve months (new; nothing above reads them) ----------
# A 10-Q reports the quarter; the 10-K reports only the year, so Q4 is the year minus the
# nine-month year-to-date that ends at Q3, known once both are known. The metrics behind
# declared hypotheses (earnings_yield, asset_growth, market_cap) do not use any of this.
QUARTER_DAYS = (80, 100)
YEAR_DAYS = (330, 400)
NINE_MONTH_DAYS = (260, 280)
TTM_STALE_DAYS = 200
EPS = ["us-gaap:EarningsPerShareDiluted", "us-gaap:EarningsPerShareBasic"]
NET_INCOME = ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss", "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic"]
SHARES = ["us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
          "us-gaap:WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
          "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"]


@dataclass
class Quarter:
    """One fiscal quarter of a flow concept."""
    start: str
    end: str
    value: float
    known_at: str           # the later of its inputs' known_at
    inputs: list[Fact]      # 1 fact (reported quarter) or 2 (Q4 = fiscal year minus nine-month YTD)
    derived: bool
    unit: str | None = None


def _by_end(view: View, entity: str, attr: str, lo: int, hi: int, annual_form: bool = False) -> dict[str, Fact]:
    """Latest-known fact per period end among periods lasting lo..hi days."""
    def ok(f: Fact) -> bool:
        d = _dur(f)
        return d is not None and lo <= d <= hi and (not annual_form or str(f.context.get("form", "")).startswith("10-K"))
    out: dict[str, Fact] = {}
    for f in view.periods(entity, attr, ok):
        if f.as_of not in out or f.known_at > out[f.as_of].known_at:
            out[f.as_of] = f
    return out


def _plus_day(d: str) -> str:
    return (date.fromisoformat(d) + timedelta(days=1)).isoformat()


def _quarters_of(view: View, entity: str, attr: str, value) -> dict[str, Quarter]:
    out = {end: Quarter(f.context["start"], end, value(f), f.known_at, [f], False, f.unit)
           for end, f in _by_end(view, entity, attr, *QUARTER_DAYS).items()}
    ytd = list(_by_end(view, entity, attr, *NINE_MONTH_DAYS).values())
    for end, y in _by_end(view, entity, attr, *YEAR_DAYS, annual_form=True).items():
        if end in out:
            continue  # the company reported Q4 itself
        same_year = [n for n in ytd if n.unit == y.unit and abs(days_between(n.context["start"], y.context["start"])) <= 3
                     and QUARTER_DAYS[0] <= days_between(n.as_of, end) <= QUARTER_DAYS[1]]
        if not same_year:
            continue
        n = max(same_year, key=lambda f: (f.as_of, f.known_at))
        out[end] = Quarter(_plus_day(n.as_of), end, value(y) - value(n), max(y.known_at, n.known_at), [y, n], True, y.unit)
    return out


def quarterly(view: View, entity: str, attrs, adjust=None) -> list[Quarter]:
    """Discrete fiscal quarters, oldest first. Concepts merge by priority (their order in ``attrs``),
    as in ``annual``: a quarter takes the first concept that has it. ``adjust(fact) -> float``
    replaces each input's value before the arithmetic (``sue`` moves EPS to one share basis)."""
    attrs = [attrs] if isinstance(attrs, str) else attrs
    value = adjust or (lambda f: float(f.value))
    best: dict[str, Quarter] = {}
    for a in attrs:
        for end, q in _quarters_of(view, entity, a, value).items():
            best.setdefault(end, q)
    return [best[k] for k in sorted(best)]


def _contiguous(qs: list[Quarter]) -> bool:
    return all(0 <= days_between(a.end, b.start) <= 7 for a, b in zip(qs, qs[1:]))


def ttm(view: View, entity: str, attrs) -> M | None:
    """Sum of the last four contiguous quarters; None when the latest ended over 200 days before
    the view, a quarter is missing, or the quarters disagree on unit."""
    qs = quarterly(view, entity, attrs)[-4:]
    if len(qs) < 4 or days_between(qs[-1].end, view.t) > TTM_STALE_DAYS:
        return None
    if not _contiguous(qs) or not 340 <= days_between(qs[0].start, qs[-1].end) <= 380 or len({q.unit for q in qs}) != 1:
        return None
    inputs = list({f.id: f for q in qs for f in q.inputs}.values())
    return M(float(sum(q.value for q in qs)), qs[-1].end, inputs, f"sum of the four quarters ending {qs[-1].end}")


def _split_basis(view: View, entity: str):
    """Per-share values move to the share basis at view.t: a value filed at k is divided by every
    split after k that is known at the view. Only splits up to view.t enter, so nothing later leaks."""
    splits = [(f.as_of, float(f.value)) for f in view.facts(entity, "split") if f.value]

    def adjust(f: Fact) -> float:
        k = f.known_at[:10]
        return float(f.value) / float(np.prod([r for d, r in splits if d > k] or [1.0]))
    return adjust


def _year_before(qs: list[Quarter], i: int) -> int | None:
    for j in range(i - 1, -1, -1):
        gap = days_between(qs[j].end, qs[i].end)
        if 350 <= gap <= 380:
            return j
        if gap > 380:
            return None
    return None


def sue_at(qs: list[Quarter], i: int) -> tuple[float, list[Quarter]] | None:
    """Standardized unexpected earnings of quarter i: (EPS_q - EPS_{q-4}) divided by the standard
    deviation of that same difference over the 8 quarters before q (at least 6 of them)."""
    j = _year_before(qs, i)
    if j is None:
        return None
    diffs, used = [], [qs[i], qs[j]]
    for k in range(max(0, i - 8), i):
        if days_between(qs[k].end, qs[i].end) > 760:  # more than 8 quarters back
            continue
        jk = _year_before(qs, k)
        if jk is not None:
            diffs.append(qs[k].value - qs[jk].value)
            used += [qs[k], qs[jk]]
    if len(diffs) < 6:
        return None
    sd = float(np.std(diffs, ddof=1))
    if not sd > 0:
        return None
    return (qs[i].value - qs[j].value) / sd, used


def eps_quarters(view: View, entity: str) -> list[Quarter]:
    """Quarterly EPS (diluted, else basic), all on the share basis at view.t."""
    return quarterly(view, entity, EPS, adjust=_split_basis(view, entity))


def sue(view: View, entity: str) -> M | None:
    """SUE of the latest quarter known at the view (None if it ended over 200 days before)."""
    qs = eps_quarters(view, entity)
    if not qs or days_between(qs[-1].end, view.t) > TTM_STALE_DAYS:
        return None
    r = sue_at(qs, len(qs) - 1)
    if r is None:
        return None
    s, used = r
    inputs = list({f.id: f for q in used for f in q.inputs}.values())
    return M(float(s), qs[-1].end, inputs, "EPS change vs a year earlier / its sd over the 8 prior quarters")


def latest_shares_any(view: View, entity: str) -> Fact | None:
    """Weighted-average shares: diluted, else basic-and-diluted, else basic; the first concept with a
    current figure. Picks the shortest period at the latest end, as ``latest_shares`` does."""
    for a in SHARES:
        ps = view.periods(entity, a)
        if not ps:
            continue
        last = ps[-1].as_of
        f = min((f for f in ps if f.as_of == last), key=lambda f: (_dur(f) or 0, f.known_at))
        if _fresh(view, f):
            return f
    return None


def market_cap_any(view: View, entity: str, book: PriceBook) -> M | None:
    """Market value as ``market_cap``, with the broader share concepts."""
    sh = latest_shares_any(view, entity)
    px = book.close_on(view, entity)
    if sh is None or px is None:
        return None
    d, close = px
    mc = sh.value * book.basis_after(entity, sh.known_at) * close
    return M(mc, d, [sh, book.snapshot[entity.upper()]], f"{sh.attr.split(':')[-1]} x close on {d}")


def earnings_yield_ttm(view: View, entity: str, book: PriceBook) -> M | None:
    """Trailing-twelve-month net income (NetIncomeLoss, else ProfitLoss, else net income available to
    common holders) / market value. USD only (the price is in USD); the same sanity bound as
    ``earnings_yield``."""
    ni = ttm(view, entity, NET_INCOME)
    mc = market_cap_any(view, entity, book)
    if ni is None or mc is None or mc.value <= 0 or ni.inputs[0].unit != "USD":
        return None
    ey = ni.value / mc.value
    if not -1.0 <= ey <= 1.0:
        return None
    return M(ey, mc.as_of, [*ni.inputs, *mc.inputs], "trailing four quarters' net income / market value")
