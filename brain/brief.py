"""One-page brief on a target, built only from the bank, every line cited.

Citations look like [ab12cd34]: the first 8 characters of a fact id. ``python -m
brain show <TICKER> <id-prefix>`` prints the full fact with its source URL. Derived
numbers are written back to the bank as tier-"derived" facts that cite their inputs.

The pieces are shared with the terminal (``brain/terminal/data.py``): the price
summary, the derived metrics (computed and written once, here), the filing
descriptions and the signal evidence. A number on a terminal screen is the number
in the brief.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from brain import config, markets, metrics
from brain.bank import Bank, Fact
from brain.metrics import M, PriceBook, annual
from brain.sources.fred import DEFAULT as FRED_SERIES
from brain.sources.sec import REVENUE
from brain.timeutil import days_between, now_iso

ITEMS_8K = {"1.01": "material agreement", "1.02": "agreement ended", "1.03": "bankruptcy", "1.04": "mine safety",
            "1.05": "cybersecurity incident", "2.01": "acquisition/disposal", "2.02": "results", "2.03": "new debt",
            "2.04": "debt acceleration", "2.05": "restructuring", "2.06": "impairment", "3.01": "listing notice",
            "3.02": "unregistered equity sale", "3.03": "holders' rights changed", "4.01": "auditor change",
            "4.02": "non-reliance on past financials", "5.01": "change of control", "5.02": "officer/director change",
            "5.03": "charter/bylaw change", "5.04": "benefit-plan trading halt", "5.05": "code of ethics change",
            "5.06": "shell status change", "5.07": "shareholder vote", "5.08": "director nominations",
            "7.01": "Reg FD", "8.01": "other event", "9.01": "exhibits"}

DERIVED_LABELS = {"market_cap": "Market value", "earnings_yield": "Earnings yield", "revenue_growth": "Revenue growth",
                  "net_margin": "Net margin", "gross_margin": "Gross margin", "free_cash_flow": "Free cash flow",
                  "asset_growth": "Asset growth", "fcf_yield": "FCF yield"}

SIGNAL_TO_HYP = {"mom12_1": "H1-mom12_1", "rev1": "H2-rev1", "earnings_yield": "H3-earnings_yield",
                 "low_asset_growth": "H4-low_asset_growth"}
# The hypotheses on the broad universe (H3c, H6, H7b, H8b) are found in HYPOTHESES by their universe (a withdrawn
# one gives way to its successor, hypotheses.current); their signals
# are per-ticker ones, read one stock at a time and cached per month-end (signals.cross_section).
EVIDENCE_BUDGET_S = 15.0   # seconds a call may spend ranking the broad universe; the rest waits for the next call
BROAD_MIN_SHARE = 0.5      # rank against the broad universe only when the bank holds at least this share of it
EARNINGS_ROWS = 8
INSIDER_DAYS = 90
INSIDER_ROWS = 10
ROLE_NAMES = {"officer": "officer", "director": "director", "ten_percent_owner": "10% owner", "other": "other"}

IDENTITY = (("identity.industry", "Industry"), ("identity.exchanges", "Exchange"),
            ("identity.fiscal_year_end", "Fiscal year ends (MMDD)"), ("identity.cik", "SEC CIK"),
            ("identity.website", "Website"))
INSIDER_FORMS = ("4", "3", "5", "144")
REPORT_FORMS = ("10-K", "10-Q", "8-K", "20-F", "40-F", "6-K", "10-KT", "10-QT", "DEF 14A")


def notable_filing(form) -> bool:
    """The company's own annual, quarterly and current reports and its proxy (amendments included): what a
    summary leads with. Everything else stays in the full list (terminal CF): insider and ownership
    reports, and offering paperwork, of which a bank files several a day (JPM's latest 400 filings
    were all 424B2, 424B3 or FWP)."""
    return str(form).upper().split("/")[0] in REPORT_FORMS


def identity_value(f: Fact):
    """An identity fact's value for display. SEC lists one exchange per listed security, so a bank with
    preferred shares reads "NYSE,NYSE,NYSE,..."; repeats are dropped."""
    if f.attr == "identity.exchanges" and isinstance(f.value, str):
        return ", ".join(dict.fromkeys(x.strip() for x in f.value.split(",") if x.strip()))
    return f.value


_BIDI_CTRL = dict.fromkeys([*range(0x00, 0x20), 0x7F, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A)], " ")
_MD_ACTIVE = str.maketrans({"[": "［", "]": "］", "*": "∗", "`": "ʹ", "|": "¦", "<": "‹", ">": "›", "#": "＃"})
_SAFE_URL = re.compile(r"https?://[^\s()<>\[\]]+")


def safe_inline(text, limit: int = 300) -> str:
    """Third-party text (headlines, web claims, source names) made inert inside markdown: one line,
    no control or direction-override characters, markdown-active characters swapped for look-alikes,
    so a stored claim cannot add headings or tables, fake a citation, or hide a verdict."""
    s = " ".join(str(text).translate(_BIDI_CTRL).split()).translate(_MD_ACTIVE)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def md_link(text, url) -> str:
    """A markdown link only for a plain http(s) URL; otherwise the text alone."""
    t = safe_inline(text)
    return f"[{t}]({url})" if isinstance(url, str) and _SAFE_URL.fullmatch(url) else t


def cite(*facts) -> str:
    return " ".join(f"[{f.id[:8]}]" for f in facts if f is not None)


def money(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    a = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{x / div:,.2f}{suf}"
    return f"{x:,.2f}"


def pct(x, d=1) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:+.{d}f}%"


FUNDAMENTALS = [
    ("Revenue", REVENUE, True),
    ("Gross profit", "us-gaap:GrossProfit", True),
    ("Operating income", "us-gaap:OperatingIncomeLoss", True),
    ("Net income", "us-gaap:NetIncomeLoss", True),
    ("EPS, diluted", "us-gaap:EarningsPerShareDiluted", True),
    ("Operating cash flow", "us-gaap:NetCashProvidedByUsedInOperatingActivities", True),
    ("Capex", "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", True),
    ("R&D", "us-gaap:ResearchAndDevelopmentExpense", True),
    ("Buybacks", "us-gaap:PaymentsForRepurchaseOfCommonStock", True),
    ("Dividends paid", "us-gaap:PaymentsOfDividends", True),
    ("Total assets", "us-gaap:Assets", False),
    ("Total liabilities", "us-gaap:Liabilities", False),
    ("Shareholders' equity", "us-gaap:StockholdersEquity", False),
    ("Cash", "us-gaap:CashAndCashEquivalentsAtCarryingValue", False),
    ("Long-term debt", ["us-gaap:LongTermDebtNoncurrent", "us-gaap:LongTermDebt"], False),
]


def _quarter(view, t, attrs):
    attrs = [attrs] if isinstance(attrs, str) else attrs
    q = []
    for a in attrs:
        q += view.periods(t, a, lambda f: 80 <= (f.context.get("duration_days") or 0) <= 100)
    if not q:
        return None, None
    q.sort(key=lambda f: (f.as_of, f.known_at))
    cur = q[-1]
    prev = [f for f in q if 350 <= days_between(f.as_of, cur.as_of) <= 380]
    return cur, (prev[-1] if prev else None)


def price_stats(d, c, a, per_year: int = 252) -> dict | None:
    """Price summary from dates, closes (split-adjusted) and dividend-adjusted closes, all already cut
    at a view's date. Returns and volatility use ``a`` (total return); levels, the 12-month high and
    low, and the distance from that high use ``c``. ``per_year`` is sessions per year: 252, or 365 for
    assets that trade every day, so "1 month" means a month. A window the history does not cover is
    NaN, never a shorter window under the same label."""
    if len(a) < 1:
        return None
    nan = float("nan")
    n = len(a)

    def ret(k):
        return float(a[-1] / a[-1 - k] - 1) if n > k else nan
    prior = int(np.searchsorted(d, str(d[-1])[:4] + "-01-01")) - 1  # last close of the previous year
    full = n > per_year  # a whole year of returns behind the last close
    hi, lo = (float(c[-per_year:].max()), float(c[-per_year:].min())) if full else (nan, nan)
    lr = np.diff(np.log(a[-(per_year + 1):]))
    return {"close": float(c[-1]), "date": str(d[-1]), "chg_1d": ret(1), "chg_5d": ret(round(5 * per_year / 252)),
            "ret_1m": ret(round(per_year / 12)), "ret_3m": ret(round(per_year / 4)),
            "ret_ytd": float(a[-1] / a[prior] - 1) if prior >= 0 else nan, "ret_12m": ret(per_year),
            "vol_12m": float(lr.std() * np.sqrt(per_year)) if full else nan,
            "from_high_12m": float(c[-1] / hi - 1) if full else nan, "high_12m": hi, "low_12m": lo}


def price_summary(view, book: PriceBook, t: str, per_year: int = 252) -> dict | None:
    """:func:`price_stats` for one ticker in a PriceBook, as of the view's date."""
    t = t.upper()
    if not book.has(t):
        return None
    d, a = book.adj_upto(view, t)
    return price_stats(d, book.close[t][: len(a)], a, per_year)


def derived(bank: Bank, view, t_: str, book: PriceBook) -> list[tuple[str, M, str | None, Fact]]:
    """Derived metrics for a ticker as of the view, each written to the bank as a tier-"derived" fact
    citing its inputs. Returns (name, metric, unit, fact) for the ones that exist."""
    has_px = book.has(t_)
    mc = metrics.market_cap(view, t_, book) if has_px else None
    ey = metrics.earnings_yield(view, t_, book) if has_px else None
    fcf = metrics.free_cash_flow(view, t_)
    cand: list[tuple[str, M | None, str | None]] = [
        ("market_cap", mc, "USD"), ("earnings_yield", ey, "ratio"),
        ("revenue_growth", metrics.revenue_growth(view, t_), "ratio"),
        ("net_margin", metrics.ratio(view, t_, "us-gaap:NetIncomeLoss", REVENUE, note="net income / revenue"), "ratio"),
        ("gross_margin", metrics.ratio(view, t_, "us-gaap:GrossProfit", REVENUE, note="gross profit / revenue"), "ratio"),
        ("free_cash_flow", fcf, "USD"),
        ("asset_growth", metrics.asset_growth(view, t_), "ratio")]
    if fcf and mc and mc.value > 0:
        cand.append(("fcf_yield", M(fcf.value / mc.value, mc.as_of, [*fcf.inputs, *mc.inputs], "FCF / market value"), "ratio"))
    out = []
    for nm, m, unit in cand:
        if m is None or not math.isfinite(m.value):
            continue
        f = metrics.to_fact(t_, nm, m, unit)
        bank.add(f)
        out.append((nm, m, unit, f))
    return out


def filing_desc(f) -> str:
    """8-K item numbers in words ("2.02,9.01" -> "results, exhibits"); empty for other forms."""
    items = f.context.get("items", "")
    return ", ".join(ITEMS_8K.get(i.strip(), i.strip()) for i in items.split(",") if i.strip()) if items else ""


def insider_role(c: dict) -> str:
    """An insider's relationship to the issuer and title, as the filing states them."""
    rel = ", ".join(ROLE_NAMES.get(r, r) for r in str(c.get("relationship") or "").split(",") if r)
    return rel + (f" ({c['title']})" if c.get("title") else "")


def _ref_cite(ref: str) -> str:
    return f"[{ref.partition('/')[2][:8]}]"


def _earnings_section(view, t_: str, book: PriceBook) -> list[str]:
    """The last quarters of ``events.earnings_table``: what the terminal's ERN shows, as of the view."""
    from brain import events
    rows = events.earnings_table(view, t_, book)[:EARNINGS_ROWS]
    if not rows:
        return []
    evs = {e.ref: e for e in events.earnings_events(view, t_)}
    L = [f"## Quarterly earnings, last {len(rows)} quarters (SEC filings)", "",
         "EPS from 10-Q and 10-K XBRL (diluted, else basic), on the share basis at the brief's date; a fourth quarter not reported "
         "on its own is the fiscal year minus the nine months. YoY: the change from the same quarter a year earlier "
         "over that quarter's absolute EPS. SUE: the same change over its standard deviation in the 8 quarters before "
         "(at least 6 needed). Release: the quarter's earnings 8-K (item 2.02), the last one public by the day the EPS "
         "was filed. 3-day and 20-day drift: the stock's total return minus SPY's, sessions -1..+1 and +2..+21 around "
         "the release; blank until the window has closed.", "",
         "| Quarter end | EPS filed | Release | EPS | YoY | SUE | 3-day | 20-day drift | Facts |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        ev = next((evs[x] for x in r["cite"] if x in evs), None)
        sue = "–" if r["sue"] is None else f"{r['sue']:+.2f}"
        L.append(f"| {r['period_end']} | {r['filed']} | {ev.known_at[:10] if ev else '–'} | {r['eps']:,.2f} | "
                 f"{pct(r['eps_yoy'])} | {sue} | {pct(r['car_3d'])} | {pct(r['drift_20d'])} | "
                 f"{' '.join(_ref_cite(x) for x in r['cite'])} |")
    return L + [""]


def _insider_section(view, t_: str) -> list[str]:
    """Open-market trades filed in the 90 days to the view: totals, then the newest few. Insiders appear
    as the filing names them, in the role it states, and nothing else about them."""
    from brain.sources import insiders
    cut = insiders.since(view, INSIDER_DAYS)
    recent = sorted(insiders.trades(view, t_, lambda f: f.known_at > cut),
                    key=lambda f: (f.known_at[:10], f.as_of, f.id), reverse=True)
    L = [f"## Insider trades, last {INSIDER_DAYS} days (SEC Forms 4 and 5)", ""]
    if not recent:
        if insiders.dataset_read_for(view, t_) or view.facts(t_, insiders.ATTR):
            return L + [f"- No open-market purchases or sales filed in the {INSIDER_DAYS} days to {view.t}.", ""]
        return L + [f"- No insider trades in the bank for {t_}: `python -m brain insiders {t_}` reads them.", ""]
    tot = insiders.buy_sell_totals(view, t_, INSIDER_DAYS)
    od = sum(f.context.get("value_usd") or 0.0 for f in recent
             if f.context.get("code") == "P" and insiders.is_officer_or_director(f))
    L += [f"Open-market purchases (P) and sales (S) filed in the {INSIDER_DAYS} days to {view.t}, newest filing "
          "first; awards, option exercises, gifts, tax withholding and amendments are left out. Each insider appears "
          "as the filing names them, in the role it states.", "",
          f"- Bought ${money(tot['buy_usd'])} in {tot['buys']} trade{'s' if tot['buys'] != 1 else ''}; sold "
          f"${money(tot['sell_usd'])} in {tot['sells']}. Officers and directors bought ${money(od)} (what H6's "
          "insider_buying signal measures).", "",
          "| Filed | Traded | Insider | Role, as filed | Code | Shares | Price | Value (USD) | Fact |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in recent[:INSIDER_ROWS]:
        c = f.context
        price = "–" if c.get("price") is None else f"{c['price']:,.2f}"
        L.append(f"| {f.known_at[:10]} | {f.as_of} | {safe_inline(c.get('owner') or '–', 80)} | "
                 f"{safe_inline(insider_role(c) or '–', 80)} | {'P, buy' if c.get('code') == 'P' else 'S, sale'} | "
                 f"{f.value:,.0f} | {price} | {money(c.get('value_usd'))} | {cite(f)} |")
    if len(recent) > INSIDER_ROWS:
        L.append(f"\n{len(recent) - INSIDER_ROWS} more in the window: `{t_} INS` in the terminal lists them all.")
    return L + [""]


def build(bank: Bank, ticker: str, t: str | None = None, evidence_budget_s: float | None = EVIDENCE_BUDGET_S) -> str:
    """The brief, as the bank stood at ``t``. ``evidence_budget_s`` bounds the time spent ranking the broad
    universe (None: no bound; a month's first ranking reads every stock's filings, a minute or more on a full bank)."""
    t_ = ticker.upper()
    view = bank.view(t)
    book = PriceBook(bank, [t_, "SPY"], asof=t)
    L = []
    name = view.latest(t_, "identity.name")
    L += [f"# {t_}{' — ' + safe_inline(name.value) if name else ''}", "",
          f"Brief as of {view.t}, generated {now_iso()}. Built only from the memory bank. "
          f"[id] cites a fact: `python -m brain show {t_} <id>` prints it with its source.", ""]

    # identity
    rows = []
    for a, label in IDENTITY:
        f = view.latest(t_, a)
        if f:
            rows.append(f"- {label}: {safe_inline(identity_value(f))} {cite(f)}")
    if rows:
        L += ["## Identity", "", *rows, ""]

    # price
    ps = price_summary(view, book, t_, 365 if markets.kind(t_) == "crypto" else 252)
    if ps:
        L += ["## Price", "",
              f"- Close {ps['close']:,.2f} on {ps['date']} {cite(book.snapshot[t_])}",
              f"- Total return: 1 month {pct(ps['ret_1m'])}, 3 months {pct(ps['ret_3m'])}, 12 months {pct(ps['ret_12m'])}",
              f"- Volatility, 12 months annualised: {pct(ps['vol_12m']).lstrip('+')}",
              f"- Below 12-month high: {pct(ps['from_high_12m'])}", ""]

    # fundamentals
    rows = []
    for label, attrs, flow in FUNDAMENTALS:
        s = annual(view, t_, attrs, flow=flow)
        if not s:
            continue
        cur = s[-1]
        prev = s[-2] if len(s) > 1 and 330 <= days_between(s[-2].as_of, cur.as_of) <= 400 else None
        yoy = (cur.value / abs(prev.value) - 1) if prev and prev.value else None
        val = f"{cur.value:,.2f}" if cur.unit == "USD/shares" else money(cur.value)
        rows.append(f"| {label} | {cur.as_of} | {val} | {pct(yoy)} | {cur.known_at[:10]} | {cite(cur)} |")
    if rows:
        L += ["## Fundamentals, latest fiscal year (SEC filings)", "",
              "| Item | FY end | Value | YoY | Filed | Fact |", "| --- | --- | --- | --- | --- | --- |", *rows, ""]

    qrows = []
    for label, attrs in (("Revenue", REVENUE), ("Net income", "us-gaap:NetIncomeLoss"),
                         ("EPS, diluted", "us-gaap:EarningsPerShareDiluted")):
        cur, prev = _quarter(view, t_, attrs)
        if cur:
            yoy = (cur.value / abs(prev.value) - 1) if prev and prev.value else None
            val = f"{cur.value:,.2f}" if cur.unit == "USD/shares" else money(cur.value)
            qrows.append(f"| {label} | {cur.context.get('start')}..{cur.as_of} | {val} | {pct(yoy)} | {cite(cur, prev)} |")
    if qrows:
        L += ["### Latest quarter", "", "| Item | Period | Value | vs same quarter last year | Facts |",
              "| --- | --- | --- | --- | --- |", *qrows, ""]

    # derived, written back to the bank
    rows = []
    for nm, m, unit, f in derived(bank, view, t_, book):
        shown = money(m.value) if unit == "USD" else pct(m.value)
        extra = f" (P/E {1 / m.value:,.1f})" if nm == "earnings_yield" and m.value > 0 else ""
        rows.append(f"- {DERIVED_LABELS.get(nm, nm)}: {shown}{extra}, as of {m.as_of} {cite(f)} ← {cite(*m.inputs)}")
    if rows:
        L += ["## Derived (written to the bank, each citing its inputs)", "", *rows, ""]

    stock = not markets.kind(t_)  # an index, currency or coin files no 10-Q and has no insiders
    if stock:
        L += _earnings_section(view, t_, book)

    # filings
    fil = sorted(view.facts(t_, "filing"), key=lambda f: f.known_at, reverse=True)
    if fil:
        recent = [f for f in fil if notable_filing(f.value)][:10]
        n4 = sum(1 for f in fil if f.value == "4" and days_between(f.known_at, view.cutoff) <= 90)
        L += ["## Recent filings (10-K, 10-Q, 8-K, proxy)", ""]
        for f in recent:
            desc = filing_desc(f)
            L.append(f"- {f.known_at[:10]} **{safe_inline(f.value)}**{' — ' + safe_inline(desc) if desc else ''} "
                     f"({md_link('doc', f.source.get('url'))}) {cite(f)}")
        L += [f"- Insider transaction reports (Form 4) filed in the last 90 days: {n4}, awards and option exercises "
              f"included; the open-market trades in them are under Insider trades.", ""]
    if stock:
        L += _insider_section(view, t_)

    # headlines
    hl = sorted(view.facts(t_, "headline"), key=lambda f: f.known_at, reverse=True)[:10]
    if hl:
        L += ["## Headlines (reported, unscored: a headline is not evidence)", ""]
        L += [f"- {f.known_at[:10]} {md_link(f.value, f.source.get('url'))} — {safe_inline(f.source.get('name', ''))} "
              f"{cite(f)}" for f in hl]
        L.append("")

    # web claims
    cl = sorted(view.facts(t_, "claim"), key=lambda f: f.known_at, reverse=True)[:10]
    if cl:
        arrow = {1: "↑", -1: "↓", 0: "·"}
        L += ["## Claims noted from the web (tier opinion/reported; see `score-claims` for each source's record)", ""]
        L += [f"- {f.known_at[:10]} {arrow.get(f.context.get('direction', 0), '·')} {safe_inline(f.value)} "
              f"({md_link(f.source.get('name', ''), f.source.get('url'))}) {cite(f)}" for f in cl]
        L.append("")

    L += _signals_section(bank, t_, evidence_budget_s)
    L += _macro_section(view)
    L += ["## Caveats", "",
          "- Annual figures are the latest fiscal year in a 10-K; quarterly rows are for orientation.",
          "- A signal reading is only as good as its TEST verdict in `research/RESULTS.md`. FAIL means it did not "
          "beat a shuffled signal on unseen months after costs.",
          "- Prices are Yahoo's unofficial feed; filings are SEC EDGAR. Not financial advice.", ""]
    text = "\n".join(L)
    out = config.home() / "briefs"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{t_}.md").write_text(text)
    return text


class NoEvidence(LookupError):
    """The repo or bank lacks what a signal reading needs: the research universe, or prices."""


def evidence(bank: Bank, ticker: str, budget_s: float | None = EVIDENCE_BUDGET_S) -> dict:
    """Where ``ticker`` ranks on each tested signal at the last month-end close, against the research
    universe, with that signal's TEST verdict from the ledger. One computation for the brief's
    Signals section and the terminal's EV. Raises :class:`NoEvidence` when it cannot be computed.

    The first rows are H1-H4 against ``config.UNIVERSE``. Then, when the bank holds the broad universe
    (``universe/us_broad.json``), one row per current hypothesis declared on it (H3c, H6, H7b, H8b), ranked against it
    and marked with ``universe``, ``select`` and ``held`` (what the backtest would hold by that
    hypothesis' rule); ``broad`` says how far that ranking got. A month's first broad ranking reads every
    stock's filings: ``budget_s`` bounds the seconds one call spends on it, and until it is complete those
    rows carry the ticker's own value without a rank."""
    if not config.UNIVERSE.exists():
        raise NoEvidence(f"no research universe ({config.UNIVERSE.name})")
    from brain.research import run as research_run
    from brain.research import signals
    from brain.research.hypotheses import HYPOTHESES
    from brain.research.panel import build as build_panel
    t_ = ticker.upper()
    uni = json.loads(config.UNIVERSE.read_text())
    tickers = list(uni["tickers"]) + ([t_] if t_ not in uni["tickers"] else [])
    try:
        panel = build_panel(bank, tickers)
    except SystemExit as e:  # the panel's way of saying prices are missing
        raise NoEvidence(str(e)) from None
    if len(panel.rebal) == 0:
        raise NoEvidence("not enough price history for a month-end ranking")
    vals = signals.compute_last(bank, panel, list(SIGNAL_TO_HYP))
    verdict = {r["id"]: r for r in research_run.verdicts()} if config.LEDGER.exists() else {}
    claims = {h["id"]: h["claim"] for h in HYPOTHESES}
    j = tickers.index(t_)
    rows = []
    for s, hid in SIGNAL_TO_HYP.items():
        v = vals[s]
        x = float(v[j])
        ok = np.isfinite(v)
        r = verdict.get(hid)
        row = {"signal": s, "hypothesis": hid, "value": None, "percentile": None, "in_top_third": None,
               "verdict": r["verdict"] if r else "not run",
               "test_net_excess": r["test"]["mean_net_excess"] if r else None,
               "perm_p": r["test"]["perm_p"] if r else None, "bh_q": r["bh_q"] if r else None,
               "claim": claims.get(hid, "")}
        if math.isfinite(x):
            pctl = (v[ok] < x).sum() / max(ok.sum() - 1, 1)
            row.update(value=x, percentile=round(float(pctl) * 100), in_top_third=bool(pctl >= 2 / 3))
        rows.append(row)
    ctl = next(iter(verdict.values()))["controls"]["pass"] if verdict else None
    broad = _broad_evidence(bank, t_, verdict, claims, budget_s)
    rows += broad.pop("rows")
    n_pass = sum(r["verdict"] == "PASS" for r in rows)
    summary = f"{n_pass} of {len(rows)} signals passed their TEST gate."
    if n_pass == 0:
        summary += (" A FAIL means the signal did not beat a shuffled signal on unseen months after costs, "
                    "so a high rank on it is not evidence of an edge.")
    if ctl is False:
        summary += " The controls failed: no verdict is valid."
    if broad["status"] == "pending":
        summary += (f" The broad-universe ranking is not complete yet ({broad['computed']} of {broad['total']} "
                    "stocks); its rows show this stock's value without a rank until it is.")
    from brain.research.hypotheses import COMMON  # the H1-H4 rows' universe, named as a hypothesis names it
    return {"ticker": t_, "rebalance_date": str(panel.dates[panel.rebal[-1]]), "universe": COMMON["universe"],
            "universe_size": len(uni["tickers"]), "signals": rows, "controls_pass": ctl, "summary": summary,
            "broad": broad}


def _broad_hypotheses() -> list[dict]:
    from brain.research import ledger, signals
    from brain.research.hypotheses import BROAD, current
    return [h for h in current(ledger.withdrawn()) if h.get("universe") == BROAD and h["signal"] in signals.PER_TICKER]


def _broad_evidence(bank: Bank, t_: str, verdict: dict, claims: dict, budget_s: float | None) -> dict:
    """Rows for the hypotheses on the broad universe, and how far their ranking got: status "complete",
    "pending" (the budget ran out first; the next call goes on from the cache) or "unavailable"."""
    from brain.research import harness, signals
    from brain.research import run as research_run
    from brain.research.hypotheses import BROAD, COMMON
    hyps = _broad_hypotheses()
    out = {"universe": BROAD, "rows": []}
    path = research_run.universe_path(BROAD)
    if not hyps or not path.exists():
        return {**out, "status": "unavailable", "reason": f"no {BROAD}"}
    try:
        tickers = [str(t).upper() for t in json.loads(path.read_text())["tickers"]]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {**out, "status": "unavailable", "reason": f"{BROAD} unreadable ({e})"}
    held_here = sum(bank.shard(t).exists() for t in tickers)
    if held_here < BROAD_MIN_SHARE * len(tickers):
        return {**out, "status": "unavailable", "universe_size": len(tickers),
                "reason": f"the bank holds {held_here} of its {len(tickers)} stocks "
                          f"(`python -m brain ingest-universe --broad` adds them)"}
    names = list(dict.fromkeys(h["signal"] for h in hyps))
    every = tickers + ([t_] if t_ not in tickers else [])
    cache = config.home() / "cache" / "evidence" / f"{research_run.universe_sha(BROAD)}.json"
    try:
        cs = signals.cross_section(bank, every, names, cache_path=cache, budget_s=budget_s, first=[t_])
    except SystemExit as e:  # no SPY prices, or too little history
        return {**out, "status": "unavailable", "universe_size": len(tickers), "reason": str(e)}
    j = every.index(t_)
    done = cs["complete"]
    for h in hyps:
        v = cs["values"][h["signal"]]
        x = float(v[j])
        r = verdict.get(h["id"])
        row = {"signal": h["signal"], "hypothesis": h["id"], "value": None, "percentile": None, "in_top_third": None,
               "verdict": r["verdict"] if r else "not run",
               "test_net_excess": r["test"]["mean_net_excess"] if r else None,
               "perm_p": r["test"]["perm_p"] if r else None, "bh_q": r["bh_q"] if r else None,
               "claim": claims.get(h["id"], ""), "universe": BROAD, "universe_size": len(tickers),
               "select": h.get("select", "top_frac"), "held": None}
        if math.isfinite(x):
            row["value"] = x
            if done:
                ok = cs["listed"] & np.isfinite(v)
                pctl = (v[ok] < x).sum() / max(ok.sum() - 1, 1)
                hold, _ = harness.choose(v, cs["listed"], COMMON["frac"], row["select"])
                row.update(percentile=round(float(pctl) * 100), in_top_third=bool(pctl >= 2 / 3),
                           held=bool(hold is not None and j in set(hold.tolist())))
        out["rows"].append(row)
    return {**out, "status": "complete" if done else "pending", "universe_size": len(tickers),
            "rebalance_date": cs["rebalance_date"], "computed": cs["computed"], "total": cs["total"]}


def _signals_section(bank: Bank, t_: str, budget_s: float | None = EVIDENCE_BUDGET_S) -> list[str]:
    if not config.UNIVERSE.exists() or markets.kind(t_):
        return []  # the signals rank stocks; an index, currency or coin has no place in that ranking
    try:
        ev = evidence(bank, t_, budget_s)
    except (SystemExit, Exception) as e:  # noqa: BLE001 - a brief should still render
        return ["## Signals", "", f"Unavailable: {e}", ""]
    L = ["## Signals vs the research universe", "",
         f"Ranked on the last month-end close ({ev['rebalance_date']}) against {ev['universe_size']} universe stocks. "
         "Top third is what the backtest would hold.", "",
         "| Signal | Value | Percentile | In top third | TEST verdict | TEST net excess / mo |",
         "| --- | --- | --- | --- | --- | --- |"]
    broad = [r for r in ev["signals"] if r.get("universe")]
    for r in ev["signals"]:
        if r.get("universe"):
            continue
        if r["value"] is None:
            L.append(f"| {r['signal']} | – | – | – | {r['verdict']} | – |")
            continue
        L.append(f"| {r['signal']} | {r['value']:+.3f} | {r['percentile']:.0f} | {'yes' if r['in_top_third'] else 'no'} | "
                 f"{r['verdict']} | {pct(r['test_net_excess'], 2) if r['test_net_excess'] is not None else '–'} |")
    L.append("")
    b = ev.get("broad") or {}
    if broad:
        L += [f"Against the {b['universe_size']} stocks of `{b['universe']}` on the same day, for the hypotheses "
              "declared on it. Would hold: what that hypothesis' backtest holds (top third by signal; for "
              "`positive`, every stock above 0).", ""]
        if b.get("status") == "pending":
            L += [f"Ranking not complete: {b['computed']} of {b['total']} stocks computed so far. A month's first "
                  "ranking reads every stock's filings; each brief or EV goes on where the last stopped, and "
                  "until then this stock's own value is shown without a rank.", ""]
        L += ["| Signal | Hypothesis | Select | Value | Percentile | Would hold | TEST verdict | TEST net excess / mo |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in broad:
            val = "–" if r["value"] is None else ("$" + money(r["value"]) if r["signal"] == "insider_buying"
                                                   else f"{r['value']:+.3f}")
            pc = "–" if r["percentile"] is None else f"{r['percentile']:.0f}"
            held = "–" if r["held"] is None else ("yes" if r["held"] else "no")
            tn = pct(r["test_net_excess"], 2) if r["test_net_excess"] is not None else "–"
            L.append(f"| {r['signal']} | {r['hypothesis']} | {r['select']} | {val} | {pc} | {held} | {r['verdict']} | {tn} |")
        L.append("")
    elif b.get("status") == "unavailable" and b.get("universe_size"):
        L += [f"Not ranked against `{b['universe']}`: {b['reason']}.", ""]
    return L


def _macro_section(view) -> list[str]:
    from brain import series
    rows = []
    for sid, label in FRED_SERIES.items():
        f = view.latest("MACRO", f"series.fred.{sid}")
        if f is None:
            continue
        df = series.load(f.value)
        last = df.iloc[-1]
        prior = df[df["date"] <= str(int(last["date"][:4]) - 1) + last["date"][4:]]
        was = f", a year earlier {prior['value'].iloc[-1]:.2f}" if len(prior) else ""
        flag = "" if f.context.get("point_in_time", True) else " (revised series)"
        rows.append(f"- {label}: {last['value']:.2f} on {last['date']}{was}{flag} {cite(f)}")
    return (["## Macro context (FRED)", "", *rows, ""]) if rows else []
