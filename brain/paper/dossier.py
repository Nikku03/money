"""Know the company before trading it: a point-in-time dossier of each stock (test P5, docs/PAPER.md).

Built at each week's open from what was public before it: the company's own financial reports (SEC
XBRL, each number counted from the session after its filing date), its filing history, its price
history and how it has reacted to its own results. Nothing later.

    size, valuation   market value (shares outstanding x the last close); price to sales
    growth            last quarter's revenue against the same quarter a year before
    quality           gross margin, operating margin, profit or loss, R&D against revenue
    survival          cash runway: cash over the quarterly operating cash burn (capped at 20 quarters
                      when it does not burn); long-term debt over assets; current assets over current
                      liabilities
    dilution          shares outstanding against a year before
    reports           how old its latest report is; its last four results: the mean size and the mean
                      direction of the move from the close before each to the close after
    missing           a flag when it has filed no revenue at all

The trader learns how much any of this matters, like any feature. The ``card`` is the written dossier
kept in the journal for every trade, before the trade.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

REVENUE = ("us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:SalesRevenueNet",
           "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax", "us-gaap:SalesRevenueGoodsNet")
COGS = ("us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold")
DEBT = ("us-gaap:LongTermDebt", "us-gaap:LongTermDebtNoncurrent")
FEATURES = ("f_size", "f_ps", "f_rev_growth", "f_gross_margin", "f_op_margin", "f_profitable", "f_rnd", "f_runway",
            "f_debt", "f_current", "f_dilution", "f_stale", "f_earn_abs", "f_earn_sign", "f_fin_missing")
PLAIN = {"f_size": "bigger company (market value)", "f_ps": "expensive against its sales",
         "f_rev_growth": "revenue growing year on year", "f_gross_margin": "high gross margin",
         "f_op_margin": "operating margin (profitable operations)", "f_profitable": "made a profit last quarter",
         "f_rnd": "spends heavily on R&D against revenue", "f_runway": "long cash runway",
         "f_debt": "carries long-term debt", "f_current": "short-term assets cover short-term bills",
         "f_dilution": "issued many new shares this year", "f_stale": "its latest report is old",
         "f_earn_abs": "moves a lot on its results", "f_earn_sign": "its results usually lift it",
         "f_fin_missing": "reports no revenue"}


class Dossier:
    def __init__(self, facts: dict, names: dict, P: dict, events: dict, sic: dict):
        self.P, self.events, self.sic, self.names = P, events, sic, names
        self.dates = P["close"].index.values
        self.by = {}
        for t, rows in facts.items():
            d = {}
            for attr, val, start, end, filed, form, dur in rows:
                vis = int(np.searchsorted(self.dates, filed, side="right"))   # from the session after the filing date
                d.setdefault(attr, []).append((vis, end, start, dur or 0, val, form))
            # by when it became public, then its period end and length (a missing start must not be compared)
            self.by[t] = {a: sorted(v, key=lambda r: (r[0], r[1], r[3])) for a, v in d.items()}
        # Speed only, the same answers: what a filing-derived fact says changes only when another of the
        # company's rows becomes public, so it is computed once per such moment.
        self._vis = {t: np.sort([r[0] for a in v.values() for r in a]) for t, v in self.by.items()}
        self._res = {t: [k for k, labels in ev if "sec_results" in labels] for t, ev in events.items()}
        self._cache = {}

    # --- reading the filings as of a session ----------------------------------------------------------
    def _rows(self, t, attr, d):
        return [r for r in self.by.get(t, {}).get(attr, []) if r[0] <= d]

    def _quarter(self, t, attrs, d, back_days=0):
        """The latest ~quarterly value (80-100 days) of the first concept that has one, and its end; with
        ``back_days``, the one ending about that many days before the latest."""
        for a in attrs:
            rows = [r for r in self._rows(t, a, d) if 80 <= r[3] <= 100]
            if not rows:
                continue
            latest = max(r[1] for r in rows)
            if not back_days:
                return max((r for r in rows if r[1] == latest), key=lambda r: r[0])[4], latest
            target = pd.Timestamp(latest) - pd.Timedelta(days=back_days)
            near = [r for r in rows if abs((pd.Timestamp(r[1]) - target).days) <= 20]
            if near:
                return near[-1][4], near[-1][1]
        return None, None

    def _instant(self, t, attrs, d, back_days=0):
        for a in attrs:
            rows = [r for r in self._rows(t, a, d) if not r[2]]
            if not rows:
                continue
            latest = max(r[1] for r in rows)
            if not back_days:
                return max((r for r in rows if r[1] == latest), key=lambda r: r[0])[4], latest
            target = pd.Timestamp(latest) - pd.Timedelta(days=back_days)
            near = [r for r in rows if abs((pd.Timestamp(r[1]) - target).days) <= 45]
            if near:
                return near[-1][4], near[-1][1]
        return None, None

    def _rate(self, t, attr, d):
        """A flow per quarter from its latest report, whatever its period (cash flows are often
        year-to-date)."""
        rows = [r for r in self._rows(t, attr, d) if r[3] >= 80]
        if not rows:
            return None
        latest = max(r[1] for r in rows)
        r = min((x for x in rows if x[1] == latest), key=lambda x: x[3])
        return r[4] / r[3] * 91.0

    def facts(self, t: str, d: int) -> dict:
        """As ``facts_uncached``, from a cache keyed by how many of the company's rows are public by d."""
        key = (t, int(np.searchsorted(self._vis.get(t, np.array([])), d, side="right")))
        base = self._cache.get(key)
        if base is None:
            base = self._cache[key] = self.facts_uncached(t, d)
        f = dict(base)
        raw = self.P["raw"][t].iloc[d - 1] if t in self.P["raw"] else np.nan
        f["price"] = raw
        f["mcap"] = f["shares"] * raw if f["shares"] and np.isfinite(raw) else None
        f["sessions_since_report"] = (d - f["_last_report"]) if f["_last_report"] is not None else None
        c = self.P["close"][t].values
        moves = [math.log(c[k] / c[k - 1]) for k in self._res.get(t, [])
                 if 1 <= k < d and np.isfinite(c[k]) and np.isfinite(c[k - 1])]
        f["results_moves"] = moves[-4:]
        return f

    def facts_uncached(self, t: str, d: int) -> dict:
        raw = self.P["raw"][t].iloc[d - 1] if t in self.P["raw"] else np.nan
        rev, rev_end = self._quarter(t, REVENUE, d)
        rev1, _ = self._quarter(t, REVENUE, d, 365) if rev is not None else (None, None)
        gp, _ = self._quarter(t, ("us-gaap:GrossProfit",), d)
        cogs, _ = self._quarter(t, COGS, d)
        op, _ = self._quarter(t, ("us-gaap:OperatingIncomeLoss",), d)
        ni, _ = self._quarter(t, ("us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"), d)
        rnd, _ = self._quarter(t, ("us-gaap:ResearchAndDevelopmentExpense",), d)
        cash, _ = self._instant(t, ("us-gaap:CashAndCashEquivalentsAtCarryingValue",), d)
        assets, _ = self._instant(t, ("us-gaap:Assets",), d)
        ca, _ = self._instant(t, ("us-gaap:AssetsCurrent",), d)
        cl, _ = self._instant(t, ("us-gaap:LiabilitiesCurrent",), d)
        debt, _ = self._instant(t, DEBT, d)
        sh, sh_end = self._instant(t, ("dei:EntityCommonStockSharesOutstanding",), d)
        sh1, _ = self._instant(t, ("dei:EntityCommonStockSharesOutstanding",), d, 365) if sh else (None, None)
        ocf = self._rate(t, "us-gaap:NetCashProvidedByUsedInOperatingActivities", d)
        if gp is None and rev is not None and cogs is not None:
            gp = rev - cogs
        mcap = sh * raw if sh and np.isfinite(raw) else None
        last_report = max((r[0] for a in self.by.get(t, {}).values() for r in a if r[0] <= d and r[5] in ("10-Q", "10-K")), default=None)
        moves = []
        c = self.P["close"][t].values
        for k, labels in self.events.get(t, []):
            if "sec_results" in labels and 1 <= k < d and np.isfinite(c[k]) and np.isfinite(c[k - 1]):
                moves.append(math.log(c[k] / c[k - 1]))
        moves = moves[-4:]
        return {"name": self.names.get(t), "sic": self.sic.get(t), "price": raw, "mcap": mcap, "revenue_q": rev,
                "revenue_q_end": rev_end, "revenue_q_year_ago": rev1, "gross_profit_q": gp, "operating_income_q": op,
                "net_income_q": ni, "rnd_q": rnd, "cash": cash, "assets": assets, "current_assets": ca,
                "current_liabilities": cl, "long_term_debt": debt, "shares": sh, "shares_year_ago": sh1,
                "operating_cash_flow_q": ocf, "sessions_since_report": (d - last_report) if last_report is not None else None,
                "results_moves": moves, "_last_report": last_report}

    # --- as features --------------------------------------------------------------------------------------
    @staticmethod
    def numbers(f: dict) -> dict:
        def div(a, b):
            return a / b if a is not None and b not in (None, 0) else None
        rev = f["revenue_q"]
        runway = None
        if f["cash"] is not None and f["cash"] >= 0 and f["operating_cash_flow_q"] is not None:   # a negative cash figure is a tagging error
            runway = 20.0 if f["operating_cash_flow_q"] >= 0 else min(20.0, f["cash"] / -f["operating_cash_flow_q"])
        out = {"f_size": math.log(f["mcap"]) if f["mcap"] and f["mcap"] > 0 else None,
               "f_ps": math.log(f["mcap"] / (4 * rev)) if f["mcap"] and rev and rev > 0 else None,
               "f_rev_growth": max(-2, min(2, math.log(rev / f["revenue_q_year_ago"]))) if rev and rev > 0 and f["revenue_q_year_ago"] and f["revenue_q_year_ago"] > 0 else None,
               "f_gross_margin": max(-1, min(1, div(f["gross_profit_q"], rev))) if rev and rev > 0 and f["gross_profit_q"] is not None else None,
               "f_op_margin": max(-3, min(1, div(f["operating_income_q"], rev))) if rev and rev > 0 and f["operating_income_q"] is not None else None,
               "f_profitable": float(f["net_income_q"] > 0) if f["net_income_q"] is not None else None,
               "f_rnd": (None if f["rnd_q"] is not None and f["rnd_q"] < 0 else      # negative R&D: a tagging error
                         math.log1p(min(5, f["rnd_q"] / rev)) if rev and rev > 0 and f["rnd_q"] else (0.0 if f["rnd_q"] is None else math.log1p(5))),
               "f_runway": math.log1p(runway) if runway is not None else None,
               "f_debt": max(0, min(2, div(f["long_term_debt"], f["assets"]))) if f["long_term_debt"] is not None and f["assets"] else 0.0,
               "f_current": max(-3, min(3, math.log(f["current_assets"] / f["current_liabilities"]))) if f["current_assets"] and f["current_liabilities"] and f["current_assets"] > 0 and f["current_liabilities"] > 0 else None,
               "f_dilution": max(-1, min(2, math.log(f["shares"] / f["shares_year_ago"]))) if f["shares"] and f["shares_year_ago"] and f["shares"] > 0 and f["shares_year_ago"] > 0 else None,
               "f_stale": min(4, f["sessions_since_report"] / 63) if f["sessions_since_report"] is not None else 4.0,
               "f_earn_abs": float(np.mean(np.abs(f["results_moves"]))) if f["results_moves"] else None,
               "f_earn_sign": float(np.mean(f["results_moves"])) if f["results_moves"] else None,
               "f_fin_missing": float(rev is None)}
        return out

    def features(self, d: int, stocks: list[str]) -> pd.DataFrame:
        F = pd.DataFrame([self.numbers(self.facts(t, d)) for t in stocks], index=stocks, columns=list(FEATURES)).astype(float)
        return F.fillna(F.median()).fillna(0.0)

    def card(self, t: str, d: int) -> str:
        f = self.facts(t, d)
        n = self.numbers(f)
        m = lambda x: "n/a" if x is None else (f"${x / 1e9:.2f}B" if abs(x) >= 1e9 else f"${x / 1e6:.1f}M")
        parts = [f"{f['name'] or t} ({t}), SEC industry {f['sic'] or '?'}; market value {m(f['mcap'])} at ${f['price']:.2f}"]
        if f["revenue_q"] is not None:
            g = f" ({math.exp(n['f_rev_growth']) - 1:+.0%} on a year ago)" if n["f_rev_growth"] is not None else ""
            parts.append(f"revenue last quarter {m(f['revenue_q'])}{g}")
            if n["f_gross_margin"] is not None:
                parts.append(f"gross margin {n['f_gross_margin']:.0%}")
            if n["f_op_margin"] is not None:
                parts.append(f"operating margin {n['f_op_margin']:+.0%}")
        else:
            parts.append("no revenue reported")
        if f["net_income_q"] is not None:
            parts.append(f"net {'profit' if f['net_income_q'] > 0 else 'loss'} {m(abs(f['net_income_q']))} last quarter")
        if f["cash"] is not None:
            rw = math.expm1(n["f_runway"]) if n["f_runway"] is not None else None
            left = "unknown" if rw is None else ("under 1 quarter" if rw < 1 else f"about {rw:.0f} quarters")
            burn = (f"; burning {m(-f['operating_cash_flow_q'])} a quarter, {left} of runway"
                    if f["operating_cash_flow_q"] is not None and f["operating_cash_flow_q"] < 0 else "; operations bring cash in")
            parts.append(f"cash {m(f['cash'])}{burn}")
        if n["f_debt"]:
            parts.append(f"long-term debt {n['f_debt']:.0%} of assets")
        if n["f_dilution"] is not None:
            parts.append(f"shares {math.exp(n['f_dilution']) - 1:+.0%} on a year ago")
        if f["results_moves"]:
            parts.append("last results moves " + ", ".join(f"{x:+.0%}" for x in f["results_moves"]))
        if f["sessions_since_report"] is not None:
            parts.append(f"latest report {f['sessions_since_report']} sessions old")
        return "; ".join(parts)
