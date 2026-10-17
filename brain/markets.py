"""World markets the terminal watches (WEI): equity indices, volatility, FX, commodities, crypto.

Context, not research: none of these is in the research universe, and nothing here
feeds a backtest. Prices come from the same Yahoo chart endpoint as equities
(``brain/sources/prices.py``), so the same terms apply: personal research only,
snapshots stay in the gitignored home directory.
"""
from __future__ import annotations

from brain.bank import Bank
from brain.sources import prices

GROUPS = ("Americas", "Europe", "Asia-Pacific", "Volatility", "FX", "Commodities", "Crypto")
KIND = {"Americas": "index", "Europe": "index", "Asia-Pacific": "index", "Volatility": "volatility",
        "FX": "fx", "Commodities": "commodity", "Crypto": "crypto"}
START = "2000-01-01"   # the link tests (brain/chain/linktest.py) read oil and the dollar from 2010

_LIST = {
    "Americas": [("^GSPC", "S&P 500"), ("^DJI", "Dow Jones Industrial Average"), ("^IXIC", "Nasdaq Composite"),
                 ("^RUT", "Russell 2000"), ("^GSPTSE", "S&P/TSX Composite"), ("^BVSP", "Bovespa")],
    "Europe": [("^FTSE", "FTSE 100"), ("^GDAXI", "DAX"), ("^FCHI", "CAC 40"), ("^STOXX50E", "Euro Stoxx 50")],
    "Asia-Pacific": [("^N225", "Nikkei 225"), ("^HSI", "Hang Seng"), ("000001.SS", "Shanghai Composite"),
                     ("^NSEI", "Nifty 50"), ("^BSESN", "BSE Sensex"), ("^AXJO", "S&P/ASX 200"), ("^KS11", "KOSPI")],
    "Volatility": [("^VIX", "CBOE Volatility Index")],
    "FX": [("DX-Y.NYB", "US Dollar Index"), ("EURUSD=X", "EUR/USD"), ("GBPUSD=X", "GBP/USD"), ("JPY=X", "USD/JPY"),
           ("CNY=X", "USD/CNY"), ("INR=X", "USD/INR")],
    "Commodities": [("CL=F", "WTI crude"), ("BZ=F", "Brent crude"), ("GC=F", "Gold"), ("SI=F", "Silver"),
                    ("HG=F", "Copper"), ("NG=F", "Natural gas")],
    "Crypto": [("BTC-USD", "Bitcoin"), ("ETH-USD", "Ether")],
}

MARKETS: list[dict] = [{"symbol": s, "name": n, "group": g, "kind": KIND[g]} for g in GROUPS for s, n in _LIST[g]]
BY_SYMBOL: dict[str, dict] = {m["symbol"]: m for m in MARKETS}


def kind(symbol: str) -> str | None:
    """index / volatility / fx / commodity / crypto for a watched symbol, else None."""
    m = BY_SYMBOL.get(symbol.upper())
    return m["kind"] if m else None


def ingest(bank: Bank, log=print) -> dict:
    """Daily prices for every symbol since START. A symbol that fails is logged and skipped."""
    added: dict[str, int] = {}
    failed: dict[str, str] = {}
    for m in MARKETS:
        s = m["symbol"]
        try:
            added[s] = bank.add_many(prices.ingest(s, start=START))
        except Exception as e:  # noqa: BLE001 - one symbol's outage must not stop the rest
            failed[s] = f"{type(e).__name__}: {e}"
            log(f"  {s}: skipped ({failed[s]})")
            continue
        log(f"  {s}: {added[s]} new facts")
    return {"added": added, "failed": failed}
