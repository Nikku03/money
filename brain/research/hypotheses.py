"""Hypotheses, declared before they are run.

Rules (from the cell project, carried over):
- A hypothesis and its gate are written here and declared in the ledger before
  any result for it exists. ``research run`` refuses undeclared specs.
- A declared spec is never edited. A change is a new id (H1b, ...), and every
  id ever run counts in the false-discovery correction.
- TRAIN is for looking. The verdict comes from TEST only.

H3b, H3c and H6-H8b run on the broad universe and name their holdings rule (``select``) in
their own dict: a key added to COMMON would change H1-H5's declared spec hashes.
``universe_sha`` pins the exact universe file; ``research run`` refuses a different one.
"""

SPLIT = {"first_rebalance": "2011-01-31", "train_end": "2018-12-31", "test_start": "2019-01-01"}

GATE = {
    "test_mean_net_excess_gt": 0.0,
    "test_perm_p_lt": 0.05,
    "bh_q": 0.10,
    "n_perm": 5000,
    "controls": ("no verdict unless: oracle signal perm_p < 0.001 on TEST, and random signals "
                 "reach p < 0.05 in 1%-10% of 200 draws"),
}

COMMON = {
    "universe": "universe/us_large.json",
    "portfolio": "top third of listed tickers by signal, equal weight, rebalanced monthly",
    "benchmark": "all listed universe tickers, equal weight, same timing",
    "timing": "signal from data known at the month-end close; fill at the next day's close",
    "frac": 1 / 3,
    "cost_bps": 10,
    "split": SPLIT,
    "gate": GATE,
}

BROAD = "universe/us_broad.json"
BROAD_SHA = "dc4c64e327b34755"  # sha256[:16] of universe/us_broad.json as built (741 tickers, 2026-10-13)

HYPOTHESES = [
    {"id": "H1-mom12_1", "signal": "mom12_1",
     "claim": "Stocks with the highest return from 12 to 1 months ago beat the universe next month."},
    {"id": "H2-rev1", "signal": "rev1",
     "claim": "Last month's biggest losers beat the universe next month (short-term reversal)."},
    {"id": "H3-earnings_yield", "signal": "earnings_yield",
     "claim": "Stocks with the highest earnings yield (last fiscal-year net income / market value, "
              "from SEC filings known at the time) beat the universe next month."},
    {"id": "H4-low_asset_growth", "signal": "low_asset_growth",
     "claim": "Stocks whose total assets grew least last fiscal year beat the universe next month."},
    {"id": "H5-ridge", "signal": "ridge",
     "features": ["earnings_yield", "low_asset_growth", "mom12_1", "rev1"], "lam": 1.0, "min_months": 24,
     "claim": "A ridge learner on the ranks of the four signals above, refit monthly on months whose "
              "outcome was already realised, beats the universe next month."},
    {"id": "H3b-earnings_yield_ttm", "signal": "earnings_yield_ttm", "universe": BROAD, "universe_sha": BROAD_SHA,
     "select": "top_frac",
     "claim": "Stocks with the highest trailing-twelve-month earnings yield (net income over the last four fiscal "
              "quarters, Q4 = fiscal year minus nine-month YTD; NetIncomeLoss, else ProfitLoss, else net income "
              "available to common holders; over market value from diluted, else basic weighted-average shares; "
              "SEC filings known at the time) beat the universe next month."},
    {"id": "H6-insider_buying", "signal": "insider_buying", "universe": BROAD, "universe_sha": BROAD_SHA,
     "select": "positive",
     "portfolio": "every listed ticker with signal > 0 (at least 3), equal weight, rebalanced monthly",
     "null": "as many names drawn at random from all listed tickers, same months",
     "data": "SEC Insider Transactions Data Sets (Forms 4 and 5; amendments skipped) and Form 4 XML; "
             "non-derivative open-market purchases (code P)",
     "claim": "Stocks whose officers or directors bought shares in the open market (USD value of code-P trades "
              "in Form 4/5 filings known in the 90 days to the rebalance day, > 0) beat the universe next month."},
    {"id": "H7-earnings_reaction", "signal": "earnings_reaction", "universe": BROAD, "universe_sha": BROAD_SHA,
     "select": "top_frac_vs_all",
     "portfolio": "top third by signal of listed tickers that have one, equal weight, rebalanced monthly",
     "null": "as many names drawn at random from all listed tickers, same months",
     "claim": "Stocks with the strongest 3-day excess reaction over SPY (days -1..+1) to an earnings release "
              "(8-K item 2.02) known in the 31 days to the rebalance day keep drifting the same way: they beat "
              "the universe next month."},
    {"id": "H8-sue", "signal": "sue", "universe": BROAD, "universe_sha": BROAD_SHA, "select": "top_frac",
     "claim": "Stocks with the highest standardized unexpected earnings (latest quarterly EPS known at the "
              "rebalance day minus the same quarter a year earlier, over the standard deviation of that "
              "difference in the 8 prior quarters, split-adjusted) beat the universe next month."},
    # H3b, H7 and H8 were withdrawn after review found two mistakes in the code behind them (see the
    # ledger): a derived Q4 paired the year as first filed with nine months recast later, and every
    # 2.02 filing within 20 days of another was dropped as a duplicate, losing results that followed a
    # pre-announcement. The corrected versions below are new ids; all of them count in the BH correction.
    {"id": "H3c-earnings_yield_ttm", "signal": "earnings_yield_ttm", "universe": BROAD, "universe_sha": BROAD_SHA,
     "select": "top_frac",
     "claim": "Stocks with the highest trailing-twelve-month earnings yield (net income over the last four fiscal "
              "quarters, Q4 = fiscal year minus the nine-month YTD as known when that year was filed; "
              "NetIncomeLoss, else ProfitLoss, else net income available to common holders; over market value "
              "from diluted, else basic weighted-average shares; SEC filings known at the time) beat the "
              "universe next month."},
    {"id": "H7b-earnings_reaction", "signal": "earnings_reaction", "universe": BROAD, "universe_sha": BROAD_SHA,
     "select": "top_frac_vs_all",
     "portfolio": "top third by signal of listed tickers that have one, equal weight, rebalanced monthly",
     "null": "as many names drawn at random from all listed tickers, same months",
     "claim": "Stocks with the strongest 3-day excess reaction over SPY (days -1..+1) to the latest earnings "
              "release (8-K item 2.02; a re-file within a day or an 8-K/A within 20 days is the same release) "
              "known in the 31 days to the rebalance day keep drifting the same way: they beat the universe "
              "next month."},
    {"id": "H8b-sue", "signal": "sue", "universe": BROAD, "universe_sha": BROAD_SHA, "select": "top_frac",
     "claim": "Stocks with the highest standardized unexpected earnings (latest quarterly EPS known at the "
              "rebalance day minus the same quarter a year earlier, over the standard deviation of that "
              "difference in the 8 prior quarters, split-adjusted; Q4 = fiscal-year EPS minus nine-month EPS "
              "as known when the year was filed) beat the universe next month."},
    # H9 is kept for the relationship graph's propagation test (docs/GRAPH.md), not yet declared.
    {"id": "H10-text_similarity", "signal": "text_similarity", "select": "top_frac",
     "data": "SEC EDGAR 10-K and 10-Q primary documents from 2008 on; each against the same form whose period "
             "ended 300-430 days earlier (brain/changes.py, METHOD 1: cosine of word counts, letters only, common "
             "words left out)",
     "claim": "Stocks whose latest 10-K or 10-Q (known at the rebalance day, filed in the last 400 days) is most "
              "similar to the same report a year earlier beat the universe next month: firms that quietly change "
              "their reports do worse (Cohen, Malloy & Nguyen 2020, 'Lazy Prices')."},
]


def current(withdrawn=()) -> list[dict]:
    """HYPOTHESES as a reader should see them: a withdrawn hypothesis gives way to any later one on the
    same signal and universe (H7 to H7b). One with no successor stays, so its withdrawal is still
    shown. The ledger and the false-discovery correction keep every id regardless."""
    out = []
    for i, h in enumerate(HYPOTHESES):
        same = lambda g: g["signal"] == h["signal"] and g.get("universe") == h.get("universe")  # noqa: E731
        if h["id"] in withdrawn and any(same(g) for g in HYPOTHESES[i + 1:]):
            continue
        out.append(h)
    return out
