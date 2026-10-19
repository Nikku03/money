"""What is coming: the scheduled events of the next days, who they ripple to, and what moved beneath the surface
(docs/AHEAD.md).

Scheduled (known in advance; the outcome is not):
- **Earnings releases**, expected from the company's own history: the release (8-K item 2.02) about a year
  earlier plus 52 weeks, shifted by how the company's dates have drifted year over year. Each row says how
  regular the company has been, so a guess is shown as a guess.
- **FOMC meetings**, from the Federal Reserve's published calendar.
- **Option expiries** (the third Friday; March, June, September and December are the quarterly ones) and
  month and quarter ends.

Beneath the surface (already public, little read):
- **Planned insider sales**: Form 144 notices, filed before an affiliate sells, in the last week. Counted per
  company; the sellers are not named here.
- **Quiet changes**: reports filed in the last two weeks that differ from the same report a year earlier more
  than is usual for that company (``brain.changes``).

Ripple: for each company with an event, its customers, suppliers, competitors and partners from the relationship
graph (``brain.graph``), from its own 10-K and from other companies' 10-Ks.

Nothing here is a prediction of what the news will say, and nothing here has been tested as a signal. It is a
map of where to look. Every row cites the facts it comes from.
"""
from __future__ import annotations

import json
import re
import statistics
from datetime import date, timedelta
from typing import Callable

from brain import changes, config, events
from brain.bank import Bank, Fact
from brain.timeutil import days_between, now_iso

FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
YEAR = 364                  # 52 weeks: the same weekday a year on
DONE_DAYS = 45              # a release this close before the expected date means this quarter is already out
QUIET_Z = -1.5              # a report this many of the company's own standard deviations below its usual similarity
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}


def _d(s: str) -> date:
    return date.fromisoformat(s[:10])


def release_day(f: Fact) -> str:
    """The New York date of a filing's release (known_at is UTC; 5 hours back puts an evening release on its day)."""
    from datetime import datetime
    t = datetime.fromisoformat(f.known_at.replace("Z", "+00:00")) - timedelta(hours=5)
    return t.date().isoformat()


def _weekday(d: date) -> date:
    return d - timedelta(days=1) if d.weekday() == 5 else d + timedelta(days=1) if d.weekday() == 6 else d


# ---------- scheduled ----------
def expected_earnings(view, ticker: str, start: str, end: str) -> dict | None:
    """The ticker's next earnings release if it is expected in [start, end], from its own release history.
    None when there is no history, the release for that quarter is already out, or it falls outside the window."""
    evs = events.distinct(events.earnings_events(view, ticker))
    if len(evs) < 2:
        return None
    days = [(_d(release_day(e)), e) for e in evs]
    s, e_ = _d(start), _d(end)
    # Year-over-year drift: for each release, the release nearest 52 weeks later.
    drift = []
    for d, _ in days:
        later = [x for x, _ in days if 340 <= (x - d).days <= 390]
        if later:
            drift.append(min(later, key=lambda x: abs((x - d).days - YEAR)).toordinal() - d.toordinal() - YEAR)
    shift = int(statistics.median(drift[-8:])) if drift else 0
    spread = int(statistics.median([abs(x - shift) for x in drift[-8:]])) if len(drift) >= 2 else None
    best = None
    for d, ev in days:
        exp = _weekday(d + timedelta(days=YEAR + shift))
        if not (s <= exp <= e_):
            continue
        if any(0 < (x - (exp - timedelta(days=DONE_DAYS))).days and x <= _d(view.t) for x, _ in days if x > d):
            return None          # a release since then already covers this quarter
        if best is None or abs((exp - s).days) < abs((best[0] - s).days):
            best = (exp, d, ev)
    if best is None:
        return None
    exp, d, ev = best
    return {"kind": "earnings", "ticker": ticker.upper(), "date": exp.isoformat(), "based_on": d.isoformat(),
            "shift_days": shift, "spread_days": spread, "years_seen": len(drift), "refs": [ev.ref]}


def parse_fomc(page: str) -> list[dict]:
    """Meetings on the Federal Reserve's calendar page: start and end (decision) dates; ``projections`` when the
    meeting publishes a Summary of Economic Projections (marked * on the page)."""
    out = []
    parts = re.split(r">\s*(\d{4}) FOMC Meetings\s*<", page)
    for year, body in zip(parts[1::2], parts[2::2]):
        y = int(year)
        pairs = re.findall(r'fomc-meeting__month[^>]*>\s*(?:<strong>)?\s*([A-Za-z/]+)\s*(?:</strong>)?\s*</div>\s*'
                           r'<div class="fomc-meeting__date[^>]*>\s*([^<]+?)\s*</div>', body)
        for month, days in pairs:
            m = re.fullmatch(r"(\d{1,2})(?:-(\d{1,2}))?(\*?)", days.replace(" ", "").split("(")[0])
            if not m or "notation" in days.lower():
                continue
            names = [x.lower() for x in month.split("/")]
            full = {k[:3]: v for k, v in MONTHS.items()}
            m1 = MONTHS.get(names[0]) or full.get(names[0][:3])
            m2 = MONTHS.get(names[-1]) or full.get(names[-1][:3])
            if not m1 or not m2:
                continue
            d1, d2 = int(m.group(1)), int(m.group(2) or m.group(1))
            y2 = y + 1 if m2 < m1 else y
            out.append({"kind": "fomc", "start": date(y, m1, d1).isoformat(), "date": date(y2, m2, d2).isoformat(),
                        "projections": bool(m.group(3))})
    return out


def fomc_meetings(fetch_page: Callable[[str], str] | None = None) -> list[dict]:
    if fetch_page is None:
        from brain.net import fetch
        page = fetch(FOMC_URL, ttl=86400).body.decode("utf-8", "replace")
    else:
        page = fetch_page(FOMC_URL)
    return parse_fomc(page)


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def market_calendar(start: str, end: str) -> list[dict]:
    """Option expiries and month/quarter ends in [start, end]."""
    s, e = _d(start), _d(end)
    out = []
    y, m = s.year, s.month
    while date(y, m, 1) <= e:
        tf = third_friday(y, m)
        if s <= tf <= e:
            out.append({"kind": "expiry", "date": tf.isoformat(),
                        "what": "quarterly options and futures expiry" if m in (3, 6, 9, 12) else "monthly options expiry"})
        last = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
        while last.weekday() >= 5:
            last -= timedelta(days=1)
        if s <= last <= e:
            out.append({"kind": "period_end", "date": last.isoformat(),
                        "what": "quarter end" if m in (3, 6, 9, 12) else "month end"})
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# ---------- beneath the surface ----------
def insider_sale_notices(view, ticker: str, days: int = 7) -> dict | None:
    """Form 144 notices (an affiliate's planned sale) filed in the last ``days`` days."""
    fs = [f for f in view.facts(ticker, "filing", lambda f: f.value in ("144", "144/A"))
          if days_between(f.known_at, view.t) <= days]
    if not fs:
        return None
    return {"kind": "form144", "ticker": ticker.upper(), "n": len(fs), "last": max(f.known_at for f in fs)[:10],
            "refs": [f.ref for f in fs]}


def quiet_change(view, ticker: str, days: int = 14) -> dict | None:
    """The ticker's latest report, if filed in the last ``days`` days and unusually changed for this company
    (``changes.unusual`` at or below ``QUIET_Z``)."""
    got = changes.unusual(view, ticker, max_age=days)
    if got is None or got[0] > QUIET_Z:
        return None
    z, f = got
    past = [g.value for g in changes.history(view, ticker)
            if g.context.get("form") == f.context.get("form") and g.known_at < f.known_at]
    return {"kind": "quiet_change", "ticker": ticker.upper(), "date": f.known_at[:10], "form": f.context.get("form"),
            "period": f.as_of, "similarity": f.value, "usual": round(statistics.mean(past), 4), "z": round(z, 2),
            "new_share": f.context.get("new_share"), "refs": [f.ref]}


# ---------- ripple ----------
def ripple(edges, ticker: str, limit: int = 8) -> list[tuple[str, str]]:
    """(listed counterparty, its role for ``ticker``), customers and suppliers first."""
    from brain.graph import neighbours
    order = {"customer": 0, "supplier": 1, "partner": 2, "competitor": 3}
    rows = {(o, r) for o, r, _ in neighbours(edges, ticker) if not o.startswith("?")}
    return sorted(rows, key=lambda x: (order[x[1]], x[0]))[:limit]


# ---------- the whole look ahead ----------
def look_ahead(bank: Bank, tickers: list[str], start: str, days: int = 7, *, resolver=None,
               fomc: list[dict] | None = None, at: str | None = None) -> dict:
    """Everything above for [start, start + days], as known at ``at`` (default now)."""
    from brain import graph
    view = bank.view(at)
    end = (_d(start) + timedelta(days=days)).isoformat()
    scheduled = [x for x in (fomc or []) if start <= x["date"] <= end] + market_calendar(start, end)
    surface = []
    for t in tickers:
        for x in (expected_earnings(view, t, start, end),):
            if x:
                scheduled.append(x)
        for x in (insider_sale_notices(view, t), quiet_change(view, t)):
            if x:
                surface.append(x)
    read = [t for t in bank.entities() if graph.latest_scan(view, t) is not None]
    edges = graph.graph_at(bank, read, at, resolver) if resolver is not None else []
    for x in scheduled + surface:
        if x.get("ticker"):
            x["ripple"] = ripple(edges, x["ticker"])
    scheduled.sort(key=lambda x: (x["date"], x["kind"], x.get("ticker", "")))
    surface.sort(key=lambda x: (x["kind"], x.get("z", 0), -x.get("n", 0)))
    return {"made_at": now_iso(), "as_of": view.t, "start": start, "end": end, "tickers": len(tickers),
            "scheduled": scheduled, "surface": surface}


def render(out: dict) -> str:
    lines = [f"Look ahead {out['start']} to {out['end']} ({out['tickers']} companies watched; as known {out['as_of']})", ""]
    by_day: dict[str, list[dict]] = {}
    for x in out["scheduled"]:
        by_day.setdefault(x["date"], []).append(x)
    for d in sorted(by_day):
        lines.append(f"{d} {_d(d).strftime('%a')}")
        for x in by_day[d]:
            if x["kind"] == "earnings":
                how = ("too little history to say how regular" if x["spread_days"] is None
                       else "same weekday every year" if x["spread_days"] == 0
                       else f"in past years usually within {x['spread_days']} days of that")
                lines.append(f"  earnings  {x['ticker']:<6} expected (last year's release {x['based_on']}; {how})")
            elif x["kind"] == "fomc":
                lines.append(f"  FOMC      decision{' with projections' if x['projections'] else ''} "
                             f"(meeting {x['start']} to {x['date']})")
            else:
                lines.append(f"  market    {x['what']}")
            if x.get("ripple"):
                lines.append("            ripple: " + ", ".join(f"{o} ({r})" for o, r in x["ripple"]))
    if not by_day:
        lines.append("  nothing scheduled")
    lines += ["", "Beneath the surface"]
    if not out["surface"]:
        lines.append("  nothing flagged")
    for x in out["surface"]:
        if x["kind"] == "quiet_change":
            lines.append(f"  changed   {x['ticker']:<6} {x['form']} for {x['period']} filed {x['date']}: similarity "
                         f"{x['similarity']:.3f} against its usual {x['usual']:.3f} (z {x['z']:+.1f}), "
                         f"{x['new_share']:.0%} new sentences  -> brain changes show {x['ticker']}")
        elif x["kind"] == "form144":
            lines.append(f"  form 144  {x['ticker']:<6} {x['n']} planned insider sale notice(s) in the last 7 days "
                         f"(latest {x['last']})")
        if x.get("ripple"):
            lines.append("            ripple: " + ", ".join(f"{o} ({r})" for o, r in x["ripple"]))
    lines += ["", "A map of where to look, not a forecast: none of these rows has been tested as a signal."]
    return "\n".join(lines)


def save(out: dict) -> str:
    d = config.home() / "ahead"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{out['start']}.json"
    p.write_text(json.dumps(out, indent=1))
    return str(p)


# ---------- CLI ----------
def _cmd(a) -> int:
    bank = Bank()
    if a.universe:
        from brain.research.run import load_universe
        tickers = load_universe(None if a.universe == "large" else "universe/us_broad.json")["tickers"]
    else:
        tickers = [t.upper() for t in a.tickers] or [t for t in bank.entities() if bank.facts(t, "filing")]
    start = a.start or (a.at or now_iso())[:10]
    try:
        fomc = fomc_meetings()
    except Exception as e:  # noqa: BLE001 - the Fed's page being down must not stop the rest
        print(f"(FOMC calendar unavailable: {e})")
        fomc = []
    try:
        from brain.graph import Resolver
        resolver = Resolver.from_sec()
    except SystemExit:
        resolver = None   # no SEC_USER_AGENT: the look ahead runs without the ripple
    out = look_ahead(bank, tickers, start, a.days, resolver=resolver, fomc=fomc, at=a.at)
    print(render(out))
    print(f"\n(saved to {save(out)})")
    return 0


def add_cli(sub) -> None:
    s = sub.add_parser("ahead", help="scheduled events of the next days, who they ripple to, and what moved beneath "
                                     "the surface (docs/AHEAD.md)")
    s.add_argument("tickers", nargs="*", metavar="TICKER", help="companies to watch (default: every one in the bank)")
    s.add_argument("--universe", choices=["large", "broad"])
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--start", help="first day of the window (default today)")
    s.add_argument("--at", help="as known at this time (default now)")
    s.set_defaults(fn=_cmd)
