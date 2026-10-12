"""One-page brief on a target, built only from the bank, every line cited.

Citations look like [ab12cd34]: the first 8 characters of a fact id. ``python -m
brain show <TICKER> <id-prefix>`` prints the full fact with its source URL. Derived
numbers are written back to the bank as tier-"derived" facts that cite their inputs.
"""
from __future__ import annotations

import json
import math

import numpy as np

from brain import config, metrics
from brain.bank import Bank
from brain.metrics import M, PriceBook, annual
from brain.sources.fred import DEFAULT as FRED_SERIES
from brain.sources.sec import REVENUE
from brain.timeutil import days_between, now_iso

ITEMS_8K = {"1.01": "material agreement", "1.02": "agreement ended", "2.01": "acquisition/disposal",
            "2.02": "results", "2.03": "new debt", "2.05": "restructuring", "2.06": "impairment",
            "3.01": "listing notice", "4.01": "auditor change", "4.02": "non-reliance on past financials",
            "5.01": "change of control", "5.02": "officer/director change", "5.07": "shareholder vote",
            "7.01": "Reg FD", "8.01": "other event", "9.01": "exhibits"}

DERIVED_LABELS = {"market_cap": "Market value", "earnings_yield": "Earnings yield", "revenue_growth": "Revenue growth",
                  "net_margin": "Net margin", "gross_margin": "Gross margin", "free_cash_flow": "Free cash flow",
                  "asset_growth": "Asset growth", "fcf_yield": "FCF yield"}

SIGNAL_TO_HYP = {"mom12_1": "H1-mom12_1", "rev1": "H2-rev1", "earnings_yield": "H3-earnings_yield",
                 "low_asset_growth": "H4-low_asset_growth"}


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


def build(bank: Bank, ticker: str, t: str | None = None) -> str:
    t_ = ticker.upper()
    view = bank.view(t)
    book = PriceBook(bank, [t_, "SPY"], asof=t)
    L = []
    name = view.latest(t_, "identity.name")
    L += [f"# {t_}{' — ' + name.value if name else ''}", "",
          f"Brief as of {view.t}, generated {now_iso()}. Built only from the memory bank. "
          f"[id] cites a fact: `python -m brain show {t_} <id>` prints it with its source.", ""]

    # identity
    rows = []
    for a, label in (("identity.industry", "Industry"), ("identity.exchanges", "Exchange"),
                     ("identity.fiscal_year_end", "Fiscal year ends (MMDD)"), ("identity.cik", "SEC CIK"),
                     ("identity.website", "Website")):
        f = view.latest(t_, a)
        if f:
            rows.append(f"- {label}: {f.value} {cite(f)}")
    if rows:
        L += ["## Identity", "", *rows, ""]

    # price
    if book.has(t_):
        d, a = book.adj_upto(view, t_)
        snap = book.snapshot[t_]
        if len(a) > 1:
            def ret(n):
                return a[-1] / a[-1 - n] - 1 if len(a) > n else float("nan")
            lr = np.diff(np.log(a[-253:]))
            hi = a[-252:].max()
            px = book.close_on(view, t_)
            L += ["## Price", "",
                  f"- Close {px[1]:,.2f} on {px[0]} {cite(snap)}",
                  f"- Total return: 1 month {pct(ret(21))}, 3 months {pct(ret(63))}, 12 months {pct(ret(252))}",
                  f"- Volatility, 12 months annualised: {pct(lr.std() * np.sqrt(252)).lstrip('+')}",
                  f"- Below 12-month high: {pct(a[-1] / hi - 1)}", ""]

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
    derived: list[tuple[str, M | None, str | None]] = []
    mc = metrics.market_cap(view, t_, book) if book.has(t_) else None
    ey = metrics.earnings_yield(view, t_, book) if book.has(t_) else None
    fcf = metrics.free_cash_flow(view, t_)
    derived += [("market_cap", mc, "USD"), ("earnings_yield", ey, "ratio"),
                ("revenue_growth", metrics.revenue_growth(view, t_), "ratio"),
                ("net_margin", metrics.ratio(view, t_, "us-gaap:NetIncomeLoss", REVENUE, note="net income / revenue"), "ratio"),
                ("gross_margin", metrics.ratio(view, t_, "us-gaap:GrossProfit", REVENUE, note="gross profit / revenue"), "ratio"),
                ("free_cash_flow", fcf, "USD"),
                ("asset_growth", metrics.asset_growth(view, t_), "ratio")]
    if fcf and mc and mc.value > 0:
        derived.append(("fcf_yield", M(fcf.value / mc.value, mc.as_of, [*fcf.inputs, *mc.inputs], "FCF / market value"), "ratio"))
    rows = []
    for nm, m, unit in derived:
        if m is None or not math.isfinite(m.value):
            continue
        f = metrics.to_fact(t_, nm, m, unit)
        bank.add(f)
        shown = money(m.value) if unit == "USD" else pct(m.value)
        extra = f" (P/E {1 / m.value:,.1f})" if nm == "earnings_yield" and m.value > 0 else ""
        rows.append(f"- {DERIVED_LABELS.get(nm, nm)}: {shown}{extra} {cite(f)} ← {cite(*m.inputs)}")
    if rows:
        L += ["## Derived (written to the bank, each citing its inputs)", "", *rows, ""]

    # filings
    fil = sorted(view.facts(t_, "filing"), key=lambda f: f.known_at, reverse=True)
    if fil:
        recent = [f for f in fil if f.value not in ("4", "3", "5", "144")][:10]
        n4 = sum(1 for f in fil if f.value == "4" and days_between(f.known_at, view.cutoff) <= 90)
        L += ["## Recent filings", ""]
        for f in recent:
            items = f.context.get("items", "")
            desc = ", ".join(ITEMS_8K.get(i.strip(), i.strip()) for i in items.split(",") if i.strip()) if items else ""
            L.append(f"- {f.known_at[:10]} **{f.value}**{' — ' + desc if desc else ''} ([doc]({f.source['url']})) {cite(f)}")
        L += [f"- Insider transaction reports (Form 4) in the last 90 days: {n4}. Form 4 contents are not parsed yet.", ""]

    # headlines
    hl = sorted(view.facts(t_, "headline"), key=lambda f: f.known_at, reverse=True)[:10]
    if hl:
        L += ["## Headlines (reported, unscored: a headline is not evidence)", ""]
        L += [f"- {f.known_at[:10]} [{f.value}]({f.source['url']}) — {f.source['name']} {cite(f)}" for f in hl]
        L.append("")

    # web claims
    cl = sorted(view.facts(t_, "claim"), key=lambda f: f.known_at, reverse=True)[:10]
    if cl:
        arrow = {1: "↑", -1: "↓", 0: "·"}
        L += ["## Claims noted from the web (tier opinion/reported; see `score-claims` for each source's record)", ""]
        L += [f"- {f.known_at[:10]} {arrow.get(f.context.get('direction', 0), '·')} {f.value} "
              f"([{f.source['name']}]({f.source['url']})) {cite(f)}" for f in cl]
        L.append("")

    L += _signals_section(bank, t_)
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


def _signals_section(bank: Bank, t_: str) -> list[str]:
    if not config.UNIVERSE.exists():
        return []
    from brain.research import run as research_run
    from brain.research import signals
    from brain.research.panel import build as build_panel
    uni = json.loads(config.UNIVERSE.read_text())
    tickers = list(uni["tickers"]) + ([t_] if t_ not in uni["tickers"] else [])
    try:
        panel = build_panel(bank, tickers)
        vals = signals.compute_last(bank, panel, list(SIGNAL_TO_HYP))
    except (SystemExit, Exception) as e:  # noqa: BLE001 - a brief should still render
        return ["## Signals", "", f"Unavailable: {e}", ""]
    verdict = {r["id"]: r for r in research_run.verdicts()} if config.LEDGER.exists() else {}
    j = tickers.index(t_)
    d = str(panel.dates[panel.rebal[-1]])
    L = ["## Signals vs the research universe", "",
         f"Ranked on the last month-end close ({d}) against {len(uni['tickers'])} universe stocks. "
         "Top third is what the backtest would hold.", "",
         "| Signal | Value | Percentile | In top third | TEST verdict | TEST net excess / mo |",
         "| --- | --- | --- | --- | --- | --- |"]
    for s, hid in SIGNAL_TO_HYP.items():
        v = vals[s]
        x = v[j]
        ok = np.isfinite(v)
        if not math.isfinite(x):
            L.append(f"| {s} | – | – | – | {verdict.get(hid, {}).get('verdict', 'not run')} | – |")
            continue
        pctl = (v[ok] < x).sum() / max(ok.sum() - 1, 1)
        r = verdict.get(hid)
        L.append(f"| {s} | {x:+.3f} | {pctl * 100:.0f} | {'yes' if pctl >= 2 / 3 else 'no'} | "
                 f"{r['verdict'] if r else 'not run'} | {pct(r['test']['mean_net_excess'], 2) if r else '–'} |")
    return L + [""]


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
