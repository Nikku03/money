"""Historical company news from SEC EDGAR: the 8-K item codes (and activist stakes) each company filed,
with the moment each became public. Point-in-time: a filing counts at a decision close only if it
was public by that close (``brain.events.event_index``: the first close at or after its known_at).

The groups are fixed in advance from the 8-K form's own item list, not tuned:
"""
from __future__ import annotations

import numpy as np

from brain.events import event_index

GROUPS = {
    "sec_results": {"2.02"},                                  # results of operations (earnings)
    "sec_deal": {"1.01", "1.02", "2.01"},                     # material agreement, its end, acquisition or sale
    "sec_exec": {"5.02"},                                     # officers or directors leave or join
    "sec_restructure": {"2.05", "2.06"},                      # exit costs (layoffs, closures), impairments
    "sec_trouble": {"4.01", "4.02", "3.01", "1.03", "2.04"},  # auditor change, restatement, delisting notice,
                                                              # bankruptcy, debt acceleration
    "sec_disclosure": {"7.01", "8.01"},                       # Regulation FD and other events
    "sec_capital": {"2.03", "3.02"},                          # new debt, unregistered sale of shares
}
ACTIVIST = {"SC 13D", "SC 13D/A"}                             # a 5%+ holder declaring intent to influence
FILING_FEATURES = tuple(GROUPS) + ("sec_activist",)
CLIP = 2


def history(tickers: list[str], log=print) -> dict[str, list[tuple[str, set]]]:
    """Each ticker's filings as (known_at, labels): which FILING_FEATURES each one counts toward."""
    from brain.sources import sec
    out = {}
    for t in tickers:
        try:
            facts, _ = sec.filings_for(t, all_pages=True)
        except Exception as e:                                     # a company SEC cannot resolve: no news, said so
            log(f"  {t}: no filings ({type(e).__name__}: {e})"[:160])
            out[t] = []
            continue
        evs = []
        for f in facts:
            labels = set()
            if f.value in ("8-K", "8-K/A"):
                items = {x.strip() for x in str(f.context.get("items", "")).split(",") if x.strip()}
                labels = {g for g, codes in GROUPS.items() if items & codes}
            elif f.value in ACTIVIST:
                labels = {"sec_activist"}
            if labels:
                evs.append((f.known_at, labels))
        out[t] = sorted(evs)
    return out


def index_events(hist: dict, dates) -> dict[str, list[tuple[int, set]]]:
    """Map each filing to the index of the first close at or after it was public."""
    d = np.asarray([str(x) for x in dates])
    return {t: [(event_index(d, k), labels) for k, labels in evs] for t, evs in hist.items()}


def counts(indexed: dict, stocks: list[str], lo: int, hi: int) -> np.ndarray:
    """Per stock, per feature: filings public after close ``lo`` and by close ``hi`` (clipped at CLIP)."""
    out = np.zeros((len(stocks), len(FILING_FEATURES)))
    col = {f: j for j, f in enumerate(FILING_FEATURES)}
    for s, t in enumerate(stocks):
        for k, labels in indexed.get(t, []):
            if lo < k <= hi:
                for g in labels:
                    out[s, col[g]] += 1
    return np.minimum(out, CLIP)
