"""The company file: every piece the bank holds on one company, side by side (docs/FILE.md).

    python -m brain file NVDA [AMBA ...]

Asked for on 2026-11-25: "if we know the company in and out, how it works and doing internally, the pe ratio ...
who it trades with ... its next projects and what it need, and according to timeline, trace every fabric of the
company, we can tell how much stock will move, and if it can deliver".

The file adds nothing of its own. Each section is one of the engine's cards, printed as that card prints it, with
the verdict of the test behind it as recorded in ``research/RESULTS.md``:

1. What it is: its industry, exchange, fiscal year and market value (the bank's identity facts).
2. Its accounts: the latest fiscal year and quarter as filed at the SEC, and what the price pays for them (P/E,
   price to sales).
3. What it promised and whether it delivered: the promise record (``brain/promises.py``; G1a).
4. What it needs and what is on file: the mosaic's pieces and the chance that a share sale, late accounts or a
   restructuring comes (``brain/mosaic.py``; P1a-P3a).
5. Who it sells to, as its annual report says, and who names it as a customer (``brain/customers.py``); then the
   stocks that moved most with it over a year (``brain/connect.py``), which are not its customers or suppliers.
6. Its dates: its last reports with how the stock took them, and a guess at the next one.
7. How far it can move: the scan's ranges for the next five sessions and the chance of a 20%+ day (W1, S1), the size
   of its next report's move (M1) and the run into that report (X1a).
8. What the file does not hold yet.

A section whose piece is missing says so and the file goes on. A range or a chance is a size, never a direction: no
test of a direction has passed (``research/RESULTS.md``). The file is paper: it places no order.
"""
from __future__ import annotations

from datetime import date, timedelta

from brain import metrics
from brain.bank import Bank
from brain.metrics import PriceBook, annual
from brain.sources.sec import REVENUE
from brain.timeutil import days_between, now_iso, today

RELEASE = "release.earnings"
SAME_YEAR_DAYS = 40          # an account item counts for the fiscal year when its period ends this close to it
NEXT_REPORT_SLACK = 5        # a report due up to this many days ago by last year's date may still be coming
LAST_REPORTS = 4
SETUP_DAYS = 14              # the run into a report is shown when the report is guessed this close
NOT_YET = (
    "its suppliers by name, and the customers its report leaves unnamed (many say only \"one customer\")",
    "its projects and plans, and what they need: spending planned against cash on hand",
    "guidance given as a growth rate or as earnings per share (section 3 reads revenue in dollars only)",
    "its earnings calls",
    "a tested direction: no test of which way a stock goes has passed (research/RESULTS.md)",
)


def money(x: float) -> str:
    a = abs(x)
    for div, suf in ((1e12, " trillion"), (1e9, " billion"), (1e6, " million")):
        if a >= div:
            return f"${x / div:,.2f}{suf}"
    return f"${x:,.0f}"


def pct(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:+.{d}f}%"


def verdict(*ids: str) -> str:
    """"W1a PASS, S1a PASS": each test's verdict as recorded in the ledger ("not run" when absent)."""
    try:
        from brain.research import run
        v = {r["id"]: r["verdict"].split(":")[0] for r in run.verdicts()}
    except Exception:  # noqa: BLE001 - a file without verdicts says so
        v = {}
    return ", ".join(f"{i} {v.get(i, 'not run')}" for i in ids)


# ---------- the sections (each returns its lines) ----------
def what_it_is(view, t: str, book: PriceBook) -> list[str]:
    def val(attr):
        f = view.latest(t, attr)
        if f is None:
            return None
        return ", ".join(map(str, f.value)) if isinstance(f.value, (list, tuple)) else str(f.value)
    out = []
    ind, exch, fye = val("identity.industry"), val("identity.exchanges"), val("identity.fiscal_year_end")
    bits = [x for x in (ind, f"listed on {exch}" if exch else None,
                        f"fiscal year ends {fye[:2]}-{fye[2:]}" if fye and len(fye) == 4 else None) if x]
    out.append("  " + ("; ".join(bits) if bits else "no identity facts in the bank (python -m brain ingest " + t + ")"))
    mc = metrics.market_cap(view, t, book) if book.has(t) else None
    if mc:
        out.append(f"  market value {money(mc.value)} at the close of {mc.as_of}")
    return out


def _latest(view, t: str, attrs, flow: bool):
    s = annual(view, t, attrs, flow=flow)
    if not s:
        return None, None
    cur = s[-1]
    prev = s[-2] if len(s) > 1 and 330 <= days_between(s[-2].as_of, cur.as_of) <= 400 else None
    return cur, prev


def accounts(view, t: str, book: PriceBook) -> list[str]:
    rev, rev0 = _latest(view, t, REVENUE, True)
    if rev is None:
        return ["  no annual accounts in the bank (python -m brain ingest " + t + ")"]
    fy = rev.as_of

    def item(attrs, flow=True):
        cur, prev = _latest(view, t, attrs, flow)
        if cur is None or abs(days_between(cur.as_of, fy)) > SAME_YEAR_DAYS:
            return None, None                   # an item last tagged years ago is not this year's
        return cur, prev

    def grew(cur, prev):
        return f" ({pct(cur.value / abs(prev.value) - 1)} on the year)" if prev and prev.value else ""

    def share(cur):
        return f" ({cur.value / rev.value * 100:.1f}% of revenue)" if rev.value else ""
    out = [f"  fiscal year to {fy} (filed {rev.known_at[:10]}): revenue {money(rev.value)}{grew(rev, rev0)}"]
    line = []
    for label, attrs in (("gross profit", "us-gaap:GrossProfit"), ("operating income", "us-gaap:OperatingIncomeLoss"),
                         ("net income", "us-gaap:NetIncomeLoss")):
        cur, _ = item(attrs)
        if cur:
            line.append(f"{label} {money(cur.value)}{share(cur)}")
    eps, eps0 = item("us-gaap:EarningsPerShareDiluted")
    if eps:
        line.append(f"earnings per share {eps.value:,.2f}{grew(eps, eps0)}")
    if line:
        out.append("  " + "; ".join(line))
    line = []
    for label, attrs in (("cash from operations", "us-gaap:NetCashProvidedByUsedInOperatingActivities"),
                         ("spent on plant and equipment", ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
                                                           "us-gaap:PaymentsToAcquireProductiveAssets"]),
                         ("research and development", "us-gaap:ResearchAndDevelopmentExpense"),
                         ("shares bought back", "us-gaap:PaymentsForRepurchaseOfCommonStock"),
                         ("dividends paid", "us-gaap:PaymentsOfDividends")):
        cur, _ = item(attrs)
        if cur:
            line.append(f"{label} {money(cur.value)}")
    if line:
        out.append("  " + "; ".join(line))
    line = []
    for label, attrs in (("cash", "us-gaap:CashAndCashEquivalentsAtCarryingValue"),
                         ("long-term debt", ["us-gaap:LongTermDebtNoncurrent", "us-gaap:LongTermDebt"]),
                         ("total assets", "us-gaap:Assets"), ("total liabilities", "us-gaap:Liabilities")):
        cur, _ = item(attrs, flow=False)
        if cur:
            line.append(f"{label} {money(cur.value)}")
    if line:
        out.append("  at the year's end: " + "; ".join(line) + " (cash leaves out investments; section 4 counts them)")
    from brain.brief import _quarter
    line, period = [], None
    for label, attrs in (("revenue", REVENUE), ("net income", "us-gaap:NetIncomeLoss"),
                         ("earnings per share", "us-gaap:EarningsPerShareDiluted")):
        cur, prev = _quarter(view, t, attrs)
        if cur and cur.as_of > fy:
            period = period or cur.as_of
            if cur.as_of != period:
                continue
            v = f"{cur.value:,.2f}" if cur.unit == "USD/shares" else money(cur.value)
            yoy = f" ({pct(cur.value / abs(prev.value) - 1)} on the year-ago quarter)" if prev and prev.value else ""
            line.append(f"{label} {v}{yoy}")
    if line:
        out.append(f"  latest quarter on file, to {period}: " + "; ".join(line))
    if book.has(t):
        mc, ey = metrics.market_cap(view, t, book), metrics.earnings_yield(view, t, book)
        ttm = metrics.earnings_yield_ttm(view, t, book)
        line = []
        if ey and ey.value > 0:
            line.append(f"{1 / ey.value:,.1f} times the latest fiscal year's earnings (P/E)")
        elif ey:
            line.append("no P/E: the latest fiscal year was a loss")
        if ttm and ttm.value > 0:
            line.append(f"{1 / ttm.value:,.1f} times the last four quarters' earnings")
        if mc and rev.value > 0:
            line.append(f"{mc.value / rev.value:,.1f} times the year's revenue")
        if line and mc:
            out.append(f"  what the price pays (the close of {mc.as_of}): " + "; ".join(line))
    return out


def promised(view, t: str, asof: str) -> list[str]:
    from brain import promises
    return promises.render(t, promises.changes(promises.company(view, t)), asof)[1:]


def needs(bank: Bank, t: str) -> tuple[list[str], list[str]]:
    """(the mosaic's pieces and pictures, its partners line), from the mosaic's card as printed."""
    from brain import mosaic
    card = mosaic.card(bank, t)
    if len(card) == 1:
        return ["  " + card[0]], []
    partners = [x for x in card if x.startswith("  partners")]
    body = [x for x in card[1:] if not x.startswith("  partners") and x != "Pieces"]
    return ["  " + x if not x.startswith(" ") else x for x in body], partners


def sells_to(t: str) -> list[str]:
    """Who it sells to and who names it as a customer, as the customers card prints them (brain/customers.py)."""
    from brain import customers
    return customers.render(t, customers.load())[1:]


def moves_with(partners: list[str]) -> list[str]:
    if not partners:
        return ["  no partners: fewer than a year of prices, or outside the scanned universes"]
    return ["  moves with: " + partners[0].split("partners ", 1)[1],
            "  Partners are stocks that moved together, whatever their business: not its customers or suppliers. "
            f"What a partner's move says about the days after it: {verdict('R1b', 'R1c', 'R1d')} (docs/RIPPLES.md)."]


def next_report(episodes_: list[dict], asof: str) -> tuple[str, str] | None:
    """(the guessed day of the next report, last year's report it is counted from): 364 days after the first past
    report whose anniversary has not passed, at least 45 days after the latest report."""
    days = sorted({e["date"] for e in episodes_})
    if not days:
        return None
    today_, last = date.fromisoformat(asof), date.fromisoformat(days[-1])
    for d in days:
        due = date.fromisoformat(d) + timedelta(days=364)
        if due >= today_ - timedelta(days=NEXT_REPORT_SLACK) and (due - last).days >= 45:
            return due.isoformat(), d
    return None


def dates(view, t: str, episodes_: list[dict], asof: str) -> list[str]:
    out = []
    past = [e for e in episodes_ if e.get("why")][-LAST_REPORTS:]
    if past:
        out.append("  its last reports, with the report session's move against the market: " + "; ".join(
            f"{e['date']} {e['why']} {e['r1'] * 100:+.1f}%" for e in past))
    nxt = next_report(episodes_, asof)
    if nxt:
        out.append(f"  next report: about {nxt[0]} (a guess: a year after its report of {nxt[1]}; "
                   f"`python -m brain expect radar --confirmed` reads the week's calendar)")
    fye = view.latest(t, "identity.fiscal_year_end")
    if fye and len(str(fye.value)) == 4:
        out.append(f"  its fiscal year ends {str(fye.value)[:2]}-{str(fye.value)[2:]}; the open promises of section 3 "
                   "fall due with the quarter or the year they name")
    return out or ["  no results report in the bank for it"]


def how_far(bank: Bank, t: str, setup: dict | None, due_in: int | None = None) -> list[str]:
    out = []
    from brain import scanwatch
    got = scanwatch.latest_scan()
    row = got[1]["rows"].get(t) if got else None
    if row:
        rec = got[1]
        lo90, lo68, hi68, hi90, shock = row
        newer = t in set(rec.get("newer_listings") or [])
        out.append(f"  the five sessions after the close of {rec['asof']}: 2 times in 3 between {lo68 * 100:+.1f}% and "
                   f"{hi68 * 100:+.1f}%; 9 times in 10 between {lo90 * 100:+.1f}% and {hi90 * 100:+.1f}% "
                   f"({verdict('W1b' if newer else 'W1a')}" + ("; a newer listing" if newer else "") + ")")
        if shock is not None:
            out.append(f"  a day of 20% or more against the market in those sessions, either way: {shock * 100:.1f}% "
                       f"({verdict('S1a')}" + ("; tested on the research universe only" if newer else "") + ")")
    else:
        out.append("  not in the latest recorded scan (thin or broken prices, or outside the scanned universes): "
                   "`python -m brain scan run " + t + "`")
    try:
        from brain import moves
        model = moves.load()
        f = moves.live(bank, t) if model else None
        if f:
            q = moves.card(model, f)["quantiles"]
            out.append(f"  its next report's move, either way: middle {q[0.5] * 100:.1f}%, half the time "
                       f"{q[0.25] * 100:.1f}% to {q[0.75] * 100:.1f}%, 8 times in 10 {q[0.1] * 100:.1f}% to "
                       f"{q[0.9] * 100:.1f}%; its last eight: median {f['med8'] * 100:.1f}% ({verdict('M1a')})")
    except Exception:  # noqa: BLE001 - a file without the earnings move is still a file
        pass
    if setup and due_in is not None and due_in <= SETUP_DAYS:
        from brain import expect
        out.append("  with its report about " + str(max(due_in, 0)) + " days away, "
                   + expect.setup_words(setup).split(";")[0] + f" ({verdict('X1a')}: the further a stock "
                   "runs into its report, the more often it falls that session; a tilt, not a trade). "
                   "`python -m brain expect show " + t + "` has the whole card")
    out.append("  These are sizes. None says which way.")
    return out


# ---------- the file ----------
def build(bank: Bank, ticker: str, asof: str | None = None) -> list[str]:
    t = ticker.upper()
    view = bank.view()
    asof = asof or today()
    book = PriceBook(bank, [t, "SPY"])
    name = view.latest(t, "identity.name")
    L = [f"THE COMPANY FILE: {t}" + (f" ({name.value})" if name else "") + f", as the bank stood on {asof}. Public "
         "filings and prices only. Each section is one of the engine's cards with its test's verdict as recorded."]
    partners: list[str] = []
    setup, eps = None, []

    def section(title: str, fn) -> None:
        L.append(title)
        try:
            L.extend(fn() or ["  nothing in the bank"])
        except (Exception, SystemExit) as e:  # noqa: BLE001 - one missing piece must not stop the file
            L.append(f"  not available: {str(e)[:120] or type(e).__name__}")

    def _needs():
        nonlocal partners
        body, partners = needs(bank, t)
        return body

    def _dates():
        nonlocal setup, eps
        from brain import expect
        eps, setup = expect.own(bank, t)
        return dates(view, t, eps, asof)
    section("1. What it is", lambda: what_it_is(view, t, book))
    section("2. Its accounts (as filed at the SEC)", lambda: accounts(view, t, book))
    section("3. What it promised, and whether it delivered (docs/PROMISES.md)", lambda: promised(view, t, now_iso()))
    section("4. What it needs and what is on file (the mosaic, docs/MOSAIC.md)", _needs)
    section("5. Who it sells to, and who it moves with (docs/CUSTOMERS.md)", lambda: sells_to(t) + moves_with(partners))
    section("6. Its dates", _dates)
    def _how_far():
        nxt = next_report(eps, asof)
        due_in = (date.fromisoformat(nxt[0]) - date.fromisoformat(asof)).days if nxt else None
        return how_far(bank, t, setup, due_in)
    section("7. How far it can move", _how_far)
    L.append("8. Not in the file yet")
    L.extend("  " + x for x in NOT_YET)
    return L


def _cmd(a) -> int:
    bank = Bank()
    for t in a.tickers:
        print("\n".join(build(bank, t)))
        print()
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("file", help="the company file: every piece the bank holds on a company, side by side "
                                    "(docs/FILE.md)")
    p.add_argument("tickers", nargs="+")
    p.set_defaults(fn=_cmd)
