"""Earnings radar: every company in the research universe likely to report in the next few days, how far
its stock usually moves on a report, and what the options price (docs/PREDICT.md).

- ``index`` reads the bank once (slow, run it weekly): each company's earnings 8-K dates (item 2.02) and its
  last eight report reactions (``car_3d``, days -1..+1 vs SPY). Saved in the gitignored home.
- ``radar`` estimates each next report as last year's same-season one plus 52 weeks (the same weekday),
  skips a company that already reported this season, and for the likeliest movers reads Cboe's delayed
  quotes: the at-the-money straddle on the first expiry after the report gives the move the market prices,
  set against the stock's own past moves; a strangle at that width shows the cheapest two-sided bet. Each
  run is saved, so ``misses`` can later ask whether the day's big movers were on it.
- ``misses`` lists one session's big movers among the companies that filed an 8-K that day or the evening
  before (SEC's daily form index), with the cause their filing names, and whether a saved radar had them.

Dates are estimates until a company announces its own; a size is not a direction, and nothing here has
been tested as a signal. Paper only.
"""
from __future__ import annotations

import json
import re
import statistics
from datetime import date, datetime, timedelta

import numpy as np

from brain import config, series
from brain.bank import Bank
from brain.events import NEW_YORK, distinct, earnings_events, reaction
from brain.metrics import PriceBook
from brain.timeutil import now_iso

BANDS = {"broad": "universe/us_broad.json", "small": "universe/us_small.json", "micro": "universe/us_micro.json"}
YEAR = 364            # 52 weeks: the same weekday a year later
SEASON_GAP = 50       # a report this recent means the company already reported this season
SLACK = 8             # companies move a report up to a week or so from last year's date: within this many
                      # days of the window, a report is "possible" rather than "likely"
TOP = 1000            # the "top" band: SEC's ticker list in file order (roughly by market value), first TOP
CHUNK = 200           # tickers per PriceBook while indexing, to bound memory
DAILY_INDEX = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/form.{ymd}.idx"
FORM_LINE = re.compile(r"^(8-K(?:/A)?)\s+(.+?)\s+(\d{1,10})\s+(\d{8}|\d{4}-\d{2}-\d{2})\s+(\S+)\s*$")


def index_path():
    return config.home() / "earnings_index.json"


def snap_dir():
    return config.home() / "earnings_radar"


def universe(bands=("broad", "small", "micro", "top")) -> dict[str, str]:
    """The research universe's bands, and "top": the largest SEC-listed companies, which the research lists
    leave out by design but whose reports move markets."""
    out: dict[str, str] = {}
    for b in bands:
        if b == "top":
            from brain.sources import sec
            for row in sec.ticker_table()[:TOP]:
                if re.fullmatch(r"[A-Z]{1,5}", row["ticker"]):
                    out.setdefault(row["ticker"], "top")
            continue
        for t in json.loads((config.ROOT / BANDS[b]).read_text())["tickers"]:
            out.setdefault(t.upper(), b)
    return out


def timing(known_at: str) -> str:
    """Before the open, during the session or after the close, in New York time."""
    local = datetime.fromisoformat(known_at).astimezone(NEW_YORK)
    if local.hour < 9 or (local.hour == 9 and local.minute < 30):
        return "before open"
    return "after close" if local.hour >= 16 else "during session"


def ny_date(known_at: str) -> date:
    return datetime.fromisoformat(known_at).astimezone(NEW_YORK).date()


def build_index(bank: Bank, tickers: dict[str, str], log=print, merge: bool = False) -> dict:
    """Each company's earnings 8-K times (last 12) and report reactions in percent (last 8). ``merge``
    rewrites only these tickers' rows in the saved index."""
    out = {"built_at": now_iso(), "tickers": {}}
    if merge and index_path().exists():
        out = load_index()
    names = sorted(tickers)
    v = bank.view()
    skipped = 0
    for i in range(0, len(names), CHUNK):
        part = names[i:i + CHUNK]
        book = PriceBook(bank, part + ["SPY"])
        for t in part:
            ev = distinct(earnings_events(v, t))
            if not ev:
                skipped += 1
                continue
            moves = []
            for f in ev[-9:]:
                r = reaction(book, v, t, f, "SPY")
                if r and r.get("car_3d") is not None and r["car_3d"] == r["car_3d"]:
                    moves.append(round(r["car_3d"] * 100, 2))
            out["tickers"][t] = {"band": tickers[t], "reports": [f.known_at for f in ev][-12:], "moves": moves[-8:],
                                 "day_vol": day_vol(book, v, t, [f.known_at for f in ev])}
        log(f"  indexed {min(i + CHUNK, len(names))} of {len(names)}")
    out["skipped_no_filings"] = out.get("skipped_no_filings", 0) + skipped if merge else skipped
    index_path().parent.mkdir(parents=True, exist_ok=True)
    index_path().write_text(json.dumps(out))
    return out


def day_vol(book: PriceBook, v, t: str, reports: list[str]) -> float | None:
    """The stock's daily standard deviation in percent over its last 160 sessions, leaving out the sessions
    from two before to three after each report: the ordinary-day part of what an option prices."""
    if not book.has(t):
        return None
    dates, adj = book.adj_upto(v, t)
    if len(adj) < 40:
        return None
    lr = np.diff(np.log(np.asarray(adj, float)))
    near = set()
    for k in reports:
        j = int(np.searchsorted(dates, ny_date(k).isoformat()))
        near.update(range(max(j - 2, 0), j + 4))
    keep = [x for i, x in enumerate(lr[-160:], start=len(lr) - min(160, len(lr)) + 1) if i not in near]
    return round(float(np.std(keep)) * 100, 3) if len(keep) >= 20 else None


def sessions(a: str, b: str) -> int:
    """Weekdays from a to b, both included."""
    d, e, n = date.fromisoformat(a), date.fromisoformat(b), 0
    while d <= e:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


def event_move(implied_pct: float, day_vol_pct: float | None, react: str, expiry: str) -> float:
    """The report's share of an option's priced move, on the same three-session basis as the stock's past
    moves (car_3d): strip the ordinary sessions between the report and expiry, add back two ordinary days.
    An at-the-money straddle prices about 0.8 standard deviations (the mean absolute move of a normal)."""
    if not day_vol_pct:
        return implied_pct
    total = (implied_pct / 0.798) ** 2
    extra = max(sessions(react, expiry) - 1, 0)
    event = max(total - extra * day_vol_pct ** 2, 0)
    return round(0.798 * (event + 2 * day_vol_pct ** 2) ** 0.5, 2)


def load_index() -> dict:
    p = index_path()
    if not p.exists():
        raise FileNotFoundError("no earnings index yet: run `python -m brain earnings index`")
    return json.loads(p.read_text())


def next_report(row: dict, start: date, days: int) -> dict | None:
    """Last year's same-season report plus 52 weeks: "likely" if that lands in [start, start+days), "possible"
    if within SLACK days of it; None if the company reported in the last SEASON_GAP days."""
    reps = row.get("reports") or []
    if not reps or (start - ny_date(reps[-1])).days < SEASON_GAP:
        return None
    end = start + timedelta(days=days)
    best = None
    for r in reps:
        est = ny_date(r) + timedelta(days=YEAR)
        if start <= est < end:
            return {"date": est.isoformat(), "timing": timing(r), "basis": ny_date(r).isoformat(), "confidence": "likely"}
        off = (start - est).days if est < start else (est - end).days + 1
        if off <= SLACK and (best is None or off < best[0]):
            best = (off, {"date": est.isoformat(), "timing": timing(r), "basis": ny_date(r).isoformat(),
                          "confidence": "possible"})
    return best[1] if best else None


def candidates(index: dict, start: date, days: int) -> dict[str, str]:
    """Tickers the index expects to report in the window, with their band."""
    return {t: row["band"] for t, row in index["tickers"].items() if next_report(row, start, days)}


def refresh(bank: Bank, tickers: dict[str, str], log=print, ttl: float = 6 * 3600) -> int:
    """Re-read the candidates' latest filings (SEC's recent page only), so a company that already reported
    this season drops out, then rewrite their index rows."""
    from brain.sources import sec
    done = {}
    for t, band in tickers.items():
        try:
            fil, _ = sec.filings_for(t, all_pages=False, ttl=ttl)
        except Exception as e:  # noqa: BLE001 - one company's outage must not stop the rest
            log(f"  {t}: filings not refreshed ({type(e).__name__})")
            continue
        bank.add_many(fil)
        done[t] = band
    if done:
        build_index(bank, done, log=lambda m: None, merge=True)
    return len(done)


def reaction_day(d: str, when: str) -> str:
    """The first session the report can move: the next weekday after an after-close report."""
    day = date.fromisoformat(d)
    if when == "after close":
        day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.isoformat()


def verdict(beat: int, n: int) -> str:
    """Cheap when the stock moved more than the options price after most of its last reports."""
    if n < 4:
        return "too few reports"
    share = beat / n
    return "cheap" if share >= 0.6 else "expensive" if share <= 0.3 else "fair"


def radar(start: date, days: int = 5, budget: float = 100.0, with_options: bool = True, max_options: int = 60,
          index: dict | None = None, log=print, chain_for=None) -> list[dict]:
    """Likely reporters in the window, the biggest usual movers first, with options where read."""
    idx = index or load_index()
    rows = []
    for t, row in idx["tickers"].items():
        nr = next_report(row, start, days)
        moves = row.get("moves") or []
        if not nr or len(moves) < 4:
            continue
        size = [abs(x) for x in moves]
        rows.append({"ticker": t, "band": row["band"], **nr, "react": reaction_day(nr["date"], nr["timing"]),
                     "mean_move": round(statistics.mean(size), 2), "median_move": round(statistics.median(size), 2),
                     "moves": moves})
    rows.sort(key=lambda r: -r["median_move"])
    if with_options:
        from brain.sources import options
        chain_for = chain_for or options.fetch_chain
        for r in rows[:max_options]:
            try:
                c = chain_for(r["ticker"])
            except Exception as e:  # noqa: BLE001 - one symbol's refusal must not stop the radar
                r["options_error"] = f"{type(e).__name__}: {e}"[:120]
                continue
            exp = options.first_expiry(c["chain"], r["react"])
            s = options.straddle(c, exp) if exp else None
            r["spot"] = c["spot"]
            if not s:
                r["options_error"] = "no usable at-the-money quotes"
                continue
            size = [abs(x) for x in r["moves"]]
            r["event_pct"] = event_move(s["implied_pct"], idx["tickers"][r["ticker"]].get("day_vol"), r["react"], exp)
            beat = sum(x > r["event_pct"] for x in size)
            g = options.strangle(c, exp, s["implied_pct"])
            r.update(straddle=s, strangle=g, beat=beat, verdict=verdict(beat, len(size)),
                     fits_budget=any(x and x["cost"] <= budget for x in (s, g)))
            log(f"  {r['ticker']}: options read")
    snap = {"made_at": now_iso(), "start": start.isoformat(), "days": days,
            "budget": budget, "rows": rows}
    snap_dir().mkdir(parents=True, exist_ok=True)
    (snap_dir() / f"{snap['made_at'][:16].replace(':', '')}.json").write_text(json.dumps(snap))
    return rows


# ---------- misses: the day's big movers, and whether a radar had them ----------
def parse_form_index(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        m = FORM_LINE.match(line)
        if m:
            out.append({"form": m[1], "company": m[2].strip(), "cik": int(m[3]), "file": m[5]})
    return out


def eight_k_filers(day: date) -> list[dict]:
    from brain.net import FetchError, fetch
    url = DAILY_INDEX.format(y=day.year, q=(day.month - 1) // 3 + 1, ymd=day.strftime("%Y%m%d"))
    try:
        r = fetch(url, user_agent=config.sec_user_agent(), ttl=3600)
    except FetchError:
        return []   # a weekend, a holiday, or a day not indexed yet
    return parse_form_index(r.body.decode("latin-1"))


def prev_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def cause(items: set[str]) -> str:
    if "2.02" in items:
        return "earnings"
    if items & {"1.01", "2.01"}:
        return "deal or agreement"
    if "8.01" in items or "7.01" in items:
        return "other news"
    return "filing: " + ",".join(sorted(items)) if items else "no 8-K"


def flagged_by_radar(ticker: str, day: str) -> bool:
    """Was the ticker on a saved radar made before that day whose window covered it (the radar's date is
    an estimate, so being on that week's list counts)."""
    d = snap_dir()
    if not d.exists():
        return False
    for p in d.glob("*.json"):
        snap = json.loads(p.read_text())
        start = snap.get("start")
        if not start or snap.get("made_at", "")[:10] > day:
            continue
        end = (date.fromisoformat(start) + timedelta(days=snap.get("days", 5) + SLACK)).isoformat()
        if start <= day <= end and any(r["ticker"] == ticker for r in snap.get("rows", [])):
            return True
    return False


def misses(bank: Bank, day: date, min_move: float = 10.0, only_universe: bool = True, log=print,
           filers=eight_k_filers, ingest_fn=None) -> list[dict]:
    """Big one-day movers among companies that filed an 8-K on ``day`` or the weekday before it."""
    from brain.ingest import ingest
    from brain.sources import sec
    ingest_fn = ingest_fn or (lambda t: ingest(Bank(bank.root), t, ("filings", "prices"), log=lambda m: None))
    prev = prev_weekday(day)
    ciks = {f["cik"] for d in (prev, day) for f in filers(d)}
    by_cik: dict[int, str] = {}
    for row in sec.ticker_table():
        by_cik.setdefault(row["cik"], row["ticker"])
    uni = universe() if only_universe else None
    tickers = sorted({by_cik[c] for c in ciks if c in by_cik and (uni is None or by_cik[c] in uni)})
    log(f"  {len(ciks)} 8-K filers on {prev} and {day}; {len(tickers)} to check")
    out = []
    for t in tickers:
        try:
            ingest_fn(t)
        except Exception as e:  # noqa: BLE001
            log(f"  {t}: not refreshed ({type(e).__name__})")
            continue
        v = bank.view()
        snap = v.latest(t, "series.price_daily")
        if snap is None:
            continue
        df = series.load(snap.value)
        dates = df["date"].astype(str).tolist()
        if day.isoformat() not in dates:
            continue
        i = dates.index(day.isoformat())
        if i == 0:
            continue
        move = (float(df["close"].iloc[i]) / float(df["close"].iloc[i - 1]) - 1) * 100
        if abs(move) < min_move:
            continue
        items: set[str] = set()
        for f in v.facts(t, "filing"):
            if str(f.value).upper().startswith("8-K") and prev.isoformat() <= ny_date(f.known_at).isoformat() <= day.isoformat():
                items |= {x.strip() for x in str(f.context.get("items") or "").split(",") if x.strip()}
        out.append({"ticker": t, "move_pct": round(move, 2), "cause": cause(items),
                    "flagged": flagged_by_radar(t, day.isoformat())})
    return sorted(out, key=lambda r: -abs(r["move_pct"]))


# ---------- CLI ----------
def _cmd_index(a):
    bands = (*BANDS, "top") if a.band == "all" else (a.band,)
    tickers = universe(bands)
    if a.ingest_missing:
        from brain.events import earnings_events
        from brain.ingest import ingest
        v = Bank().view()
        missing = [t for t in tickers if not earnings_events(v, t)]
        print(f"Reading filings and prices for {len(missing)} companies the bank lacks.")
        for t in missing:
            try:
                ingest(Bank(), t, ("filings", "prices"), log=lambda m: None)
            except Exception as e:  # noqa: BLE001 - one company's outage must not stop the rest
                print(f"  {t}: skipped ({type(e).__name__})")
    out = build_index(Bank(), tickers, log=print)
    print(f"Indexed {len(out['tickers'])} companies; {out['skipped_no_filings']} have no earnings 8-Ks in the bank "
          "(ingest their filings first).")


def _fmt(r: dict) -> str:
    s = r.get("straddle")
    when = r["date"] if r.get("confidence") == "likely" else f"~{r['date']}?"
    head = f"{r['ticker']:6} {when:12} {r['timing']:13} usual move ±{r['median_move']:.1f}% (median of {len(r['moves'])})"
    if not s:
        return head + (f"   options: {r['options_error']}" if r.get("options_error") else "")
    g = r.get("strangle")
    ev = f", ±{r['event_pct']:.1f}% for the report" if abs(r.get("event_pct", s["implied_pct"]) - s["implied_pct"]) >= 0.1 else ""
    cheap = (f"   options price ±{s['implied_pct']:.1f}% to {s['expiry']}{ev}: moved more {r['beat']}/{len(r['moves'])}"
             f": {r['verdict'].upper()}")
    cost = f"   straddle ${s['cost']:.0f}" + (f", strangle {g['call']:g}C/{g['put']:g}P ${g['cost']:.0f}" if g else "")
    return head + cheap + cost + ("   [fits budget]" if r.get("fits_budget") else "")


def _cmd_radar(a):
    from brain.timeutil import today
    start = date.fromisoformat(a.start or today())
    try:
        idx = load_index()
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    if not a.no_refresh:
        n = refresh(Bank(), candidates(idx, start, a.days), log=print)
        print(f"Re-read the latest filings of {n} likely reporters.")
        idx = load_index()
    rows = radar(start, a.days, a.budget, not a.no_options, a.max_options, index=idx, log=lambda m: None)
    print(f"{len(rows)} companies likely to report {start} to {start + timedelta(days=a.days - 1)} (dates are "
          "estimates: last year's report + 52 weeks). Biggest usual movers first. Size, not direction.")
    show = rows if a.all else [r for r in rows if r.get("straddle") or r["median_move"] >= a.min_move]
    for r in show[: a.limit]:
        print("  " + _fmt(r))
    fits = [r for r in rows if r.get("fits_budget") and r.get("verdict") == "cheap"]
    if fits:
        print(f"Cheap and within ${a.budget:.0f}: " + ", ".join(r["ticker"] for r in fits))
    return 0


def _cmd_misses(a):
    from brain.timeutil import today
    day = date.fromisoformat(a.date or today())
    rows = misses(Bank(), day, a.min_move, not a.all_filers)
    if not rows:
        print(f"No 8-K filer moved {a.min_move:g}%+ on {day} (or its closes are not in yet).")
        return 0
    hit = sum(r["flagged"] for r in rows)
    print(f"{len(rows)} big movers on {day} among 8-K filers; the radar had {hit} of them on that day.")
    for r in rows:
        print(f"  {r['ticker']:6} {r['move_pct']:+7.1f}%  {r['cause']:18} {'on the radar' if r['flagged'] else 'MISSED'}")
    print("Movers with no 8-K (trial data, rumours, sector moves) are not in this list.")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("earnings", help="earnings radar: likely reporters, usual moves, what options price (docs/PREDICT.md)")
    es = p.add_subparsers(dest="earnings_cmd", required=True)
    s = es.add_parser("index", help="read every universe company's report dates and past reactions (slow; weekly)")
    s.add_argument("--band", choices=["all", *BANDS, "top"], default="all")
    s.add_argument("--ingest-missing", action="store_true", help="first read filings and prices the bank lacks")
    s.set_defaults(fn=_cmd_index)
    s = es.add_parser("radar", help="who likely reports in the next days, and are the options cheap")
    s.add_argument("--start", help="first day of the window, YYYY-MM-DD (default today)")
    s.add_argument("--days", type=int, default=5)
    s.add_argument("--budget", type=float, default=100.0, help="flag bets costing at most this (one contract)")
    s.add_argument("--no-options", action="store_true", help="skip the option quotes")
    s.add_argument("--no-refresh", action="store_true", help="skip re-reading the likely reporters' latest filings")
    s.add_argument("--max-options", type=int, default=60, help="read quotes for this many of the biggest usual movers")
    s.add_argument("--min-move", type=float, default=8.0, help="without quotes, list usual movers of at least this %%")
    s.add_argument("--limit", type=int, default=60)
    s.add_argument("--all", action="store_true", help="list every likely reporter")
    s.set_defaults(fn=_cmd_radar)
    s = es.add_parser("misses", help="a day's big movers among 8-K filers, and whether the radar had them")
    s.add_argument("--date", help="the session, YYYY-MM-DD (default today)")
    s.add_argument("--min-move", type=float, default=10.0)
    s.add_argument("--all-filers", action="store_true", help="every 8-K filer, not only the research universe")
    s.set_defaults(fn=_cmd_misses)
