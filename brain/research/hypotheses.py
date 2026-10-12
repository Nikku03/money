"""Hypotheses, declared before they are run.

Rules (from the cell project, carried over):
- A hypothesis and its gate are written here and declared in the ledger before
  any result for it exists. ``research run`` refuses undeclared specs.
- A declared spec is never edited. A change is a new id (H1b, ...), and every
  id ever run counts in the false-discovery correction.
- TRAIN is for looking. The verdict comes from TEST only.
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
]
