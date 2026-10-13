"""Earnings events and how the stock reacted, point-in-time.

An earnings event is an 8-K (or 8-K/A) whose items include 2.02, "Results of Operations".
Its day is the first close at or after the moment it became public: a release after the
4 pm New York close moves the next session. Reactions are excess returns over a benchmark
(SPY): ``car_3d`` spans days -1..+1, the drifts days +2..+21 and +2..+61. A window that ends
after the view's date is None, never a partial number, so a backtest cannot peek.

EDGAR's acceptance times in the submissions JSON run 4-5 hours late (16:30 New York shows
as 00:30Z next day), so known_at is conservative: an after-close release lands on the right
day, a midday one at worst a day late, which the -1..+1 window still covers.
"""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np

from brain.bank import Fact, View
from brain.metrics import PriceBook, Quarter, eps_quarters, sue_at
from brain.timeutil import days_between

NEW_YORK = ZoneInfo("America/New_York")
CLOSE = time(16, 0)
EARNINGS_FORMS = ("8-K", "8-K/A")
REFILE_DAYS = 1        # a 2.02 filing this soon after a release is a re-file of it (the slides, a second copy)
AMEND_DAYS = 20        # an 8-K/A this soon after a release amends it
MATCH_DAYS = 90        # a quarter's release comes within this many days of its end


def is_earnings(f: Fact) -> bool:
    if str(f.value).upper() not in EARNINGS_FORMS:  # first: most filings are not 8-Ks, and this is cheap
        return False
    return "2.02" in {x.strip() for x in str(f.context.get("items") or "").split(",")}


def earnings_events(view: View, ticker: str) -> list[Fact]:
    """Every earnings 8-K known at the view, oldest first."""
    return sorted(view.facts(ticker, "filing", is_earnings), key=lambda f: (f.known_at, f.id))


def distinct(events: list[Fact]) -> list[Fact]:
    """One filing per earnings release, the first. A 2.02 filing within a day of the last release kept
    is a re-file of it, and an 8-K/A within 20 days amends it. Any other 2.02 filing is a release of
    its own, however soon it follows: a pre-announcement and the results two weeks later are two."""
    out: list[Fact] = []
    for e in events:
        if out:
            gap = days_between(out[-1].known_at, e.known_at)
            if gap <= REFILE_DAYS or (str(e.value).upper() == "8-K/A" and gap <= AMEND_DAYS):
                continue
        out.append(e)
    return out


def event_index(dates: np.ndarray, known_at: str) -> int:
    """Index of the first close at or after ``known_at`` (UTC) in a daily calendar."""
    local = datetime.fromisoformat(known_at.replace("Z", "+00:00")).astimezone(NEW_YORK)
    side = "right" if local.time() >= CLOSE else "left"
    return int(np.searchsorted(dates, local.date().isoformat(), side=side))


def _bench_close(dates: np.ndarray, adj: np.ndarray, d: str):
    """The benchmark's close on d, or on the last session before it within 4 days."""
    j = int(np.searchsorted(dates, d, side="right")) - 1
    if j < 0 or days_between(str(dates[j]), d) > 4:
        return None
    return float(adj[j])


def _excess(dates, adj, bdates, badj, i0: int, i1: int) -> float | None:
    if i0 < 0 or i1 >= len(dates):
        return None
    b0, b1 = _bench_close(bdates, badj, str(dates[i0])), _bench_close(bdates, badj, str(dates[i1]))
    if b0 is None or b1 is None or not (adj[i0] > 0 and b0 > 0):
        return None
    return float((adj[i1] / adj[i0] - 1) - (b1 / b0 - 1))


def reaction(book: PriceBook, view: View, ticker: str, event: Fact, bench: str = "SPY") -> dict | None:
    """The stock's excess return around one event, from prices visible at the view only."""
    if event.known_at > view.cutoff or not book.has(ticker) or not book.has(bench):
        return None
    dates, adj = book.adj_upto(view, ticker)
    bdates, badj = book.adj_upto(view, bench)
    e = event_index(dates, event.known_at)
    if e >= len(dates):
        return None  # the event's first session has not closed by the view's date
    ex = lambda a, b: _excess(dates, adj, bdates, badj, e + a, e + b)
    return {"date": str(dates[e]), "car_3d": ex(-2, 1), "drift_20d": ex(1, 21), "drift_60d": ex(1, 61),
            "event": event.ref}


def _first_known(view: View, f: Fact) -> str:
    """When this period's number was first public (a later restatement does not move it)."""
    same = view.facts(f.entity, f.attr, lambda g: g.as_of == f.as_of and g.context.get("start") == f.context.get("start"))
    return min([g.known_at for g in same] or [f.known_at])


def _ny_date(known_at: str) -> str:
    """The New York calendar date of a UTC time (EDGAR dates a filing in Eastern time)."""
    return datetime.fromisoformat(known_at.replace("Z", "+00:00")).astimezone(NEW_YORK).date().isoformat()


def _match(events: list[Fact], q: Quarter, filed: str) -> Fact | None:
    """The quarter's results release among the distinct earnings releases in the 90 days after it ended:
    the last one public by the day its EPS was filed (``filed``, compared as New York dates, so a
    release the evening before the 10-Q counts). A pre-announcement or an interim update earlier in
    the window gives way to the results that followed it, and a guidance 8-K after the filing is not
    taken. When every release in the window came after the EPS filing, the first of them."""
    window = [e for e in events if 0 < days_between(q.end, e.known_at) <= MATCH_DAYS]
    by_filing = [e for e in window if _ny_date(e.known_at) <= filed[:10]]
    return by_filing[-1] if by_filing else (window[0] if window else None)


def earnings_table(view: View, ticker: str, book: PriceBook) -> list[dict]:
    """One row per fiscal quarter with EPS, newest first.

    ``eps`` is diluted EPS (basic if that is all there is) on the share basis at the view;
    ``eps_yoy`` the fractional change from the same quarter a year earlier,
    (EPS_q - EPS_{q-4}) / |EPS_{q-4}|; ``sue`` as ``metrics.sue``; the reactions are those of the
    earnings release matched to the quarter (``_match``: the last one within 90 days after its end
    that was public by the day its EPS was filed), None when there is none or its window is not
    complete at the view. ``filed`` is when the quarter's EPS was first public; ``cite`` lists the
    refs of the facts behind the row, the release last."""
    qs = eps_quarters(view, ticker)
    events = distinct(earnings_events(view, ticker))
    rows = []
    for i, q in enumerate(qs):
        j = next((k for k in range(i - 1, -1, -1) if 350 <= days_between(qs[k].end, q.end) <= 380), None)
        prev = qs[j].value if j is not None else None
        s = sue_at(qs, i)
        filed = max(_first_known(view, f) for f in q.inputs)[:10]
        ev = _match(events, q, filed)
        r = reaction(book, view, ticker, ev) if ev is not None else None
        rows.append({
            "period_end": q.end,
            "filed": filed,
            "eps": q.value,
            "eps_yoy": (q.value - prev) / abs(prev) if prev else None,
            "sue": s[0] if s else None,
            "car_3d": r["car_3d"] if r else None,
            "drift_20d": r["drift_20d"] if r else None,
            "drift_60d": r["drift_60d"] if r else None,
            "cite": [f.ref for f in q.inputs] + ([ev.ref] if ev is not None else []),
        })
    return rows[::-1]
