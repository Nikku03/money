"""How stocks are connected, beyond their industry label (docs/DESK.md, "Connections").

An industry code says what a company files as. Prices say what it moves with. From the last LOOKBACK sessions of
daily closes in the bank this module works out, for every stock the scan covers:

- **its partners:** the stocks whose daily returns, with the market's part taken out, moved most with its own
  (Corteva's is FMC; Super Micro's are Dell and Hewlett Packard Enterprise; CoreWeave's are the bitcoin miners that
  host AI servers);
- **its drivers:** how much of its daily return went with oil, the 10-year yield, the dollar, gold and bitcoin,
  beside the market (one regression per stock), so a move can be split into what the day's drivers carried and
  what was the stock's own;
- **stories that tie stocks together:** one article carried by two stocks' news feeds, or naming both.

``clusters`` groups the day's movers that any of these link. Everything here describes the past year and today:
which stocks have moved together, not which will. Whether a partner follows a stock's news is a claim to test like
any other (CLAUDE.md), and no such test has passed.
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from urllib.parse import urlparse, urlsplit

import numpy as np

from brain import config, shocks, why
from brain.timeutil import rfc822_to_iso

LOOKBACK = 250          # sessions of daily returns behind every link
MIN_OBS = 120           # sessions two stocks must share before their link counts
PARTNERS = 8            # partners kept for a stock
MIN_CORR = 0.30         # a partner's correlation of market-adjusted daily returns, at least
CLIP_Z = 3.0            # a day counts for at most this many of the stock's usual days, so one jump is not a link
MIN_T = 3.0             # a driver counts for a stock when its coefficient is this many standard errors from zero
WRAP_FEEDS = 5          # a story in this many of the read feeds is a market wrap, not a link between two stocks
SUMMARY = 400           # characters of a story's summary kept
WRAP_NAMES = 3          # a story whose title names this many of the stocks looked at is a round-up
WRAP = re.compile(r"stock market today|stocks? to watch|futures|movers|upgrades and downgrades|wall street|"
                  r"week ahead|stocks making|stocks that|today.s market|pre-?market|in focus|morning minute|"
                  r"market wrap", re.IGNORECASE)
# all-capital words in headlines that are not tickers
ACRONYMS = {"CEO", "CFO", "COO", "CTO", "FDA", "DOJ", "SEC", "FTC", "IPO", "ETF", "USA", "GDP", "CPI", "EPS", "AID",
            "NYSE", "IBD", "API", "ALL", "NOW", "NEW", "TOP", "BIG", "FED", "OPEC", "ESG", "LLC", "USD", "EUR",
            "AML", "RFID", "CAR", "DNA", "RNA", "HIV", "PCE", "PMI", "ECB", "IRS", "EPA", "FAA", "NASA", "ARR"}
# first words too ordinary to name a company alone: the next word is needed too
COMMON = why.GENERIC | {
    "ADVANCED", "APPLIED", "SUPER", "CORE", "TARGET", "BEST", "BLOCK", "MATCH", "UNITY", "SNAP", "GAP", "ALLY", "CHIME",
    "LIFE", "OPEN", "SIMPLY", "REAL", "LIVE", "UNIVERSAL", "LIBERTY", "SUMMIT", "FRONTIER", "ALLIANCE", "PREMIER",
    "PINNACLE", "PROGRESSIVE", "DOMINION", "CARRIER", "VITAL", "DIGITAL", "ENERGY", "EAGLE", "ATLANTIC", "PACIFIC",
    "SOUTHERN", "NORTHERN", "WESTERN", "EASTERN", "CENTRAL", "CAPITAL", "COMMUNITY", "HOME", "HERITAGE",
    "INDEPENDENT", "CENTURY", "DISCOVER", "PRINCIPAL", "RELIANCE", "REPUBLIC", "SERVICE", "STANDARD", "SUN", "STAR",
    "STERLING", "TRIUMPH", "TRINITY", "VALLEY", "VILLAGE", "VISION", "VISTA", "VICTORY", "ARROW", "SELECT", "KEY",
    "MARKET", "MARKETS", "STOCK", "STOCKS", "BANK", "TRADE", "DATA", "POWER", "GREEN", "BLUE", "RED", "BLACK", "WHITE",
    "GOLD", "SILVER", "OIL", "GAS", "SOLAR", "HEALTH", "MEDICAL", "CARE", "SMART", "NEXT", "ONE", "TWO", "FIVE", "TEN",
    "PRIME", "PURE", "CLEAR", "BRIGHT", "FRESH", "GOOD", "GREAT", "GRAND", "ROYAL", "CROWN", "PARK", "CITY", "STATE",
    "WORLD", "EARTH", "OCEAN", "RIVER", "MOUNTAIN", "ROCK", "STONE", "IRON", "STEEL", "COPPER", "URANIUM", "LITHIUM",
    "AIR", "SKY", "SPACE", "TIME", "MONEY", "CASH", "CREDIT", "TRUE", "FAIR", "SAFE", "SURE", "FAST", "QUICK", "PLAY",
    "GAME", "MEDIA", "NEWS", "WEB", "NET", "CLOUD", "SOFTWARE", "CYBER", "NANO", "BIO", "GENE", "CELL", "MODERN",
    "FUTURE", "LEGACY", "CLASSIC", "ORIGIN", "SOURCE", "FOCUS", "FORWARD", "ASCEND", "ELEVATE", "EVOLVE", "INSPIRE",
    "ENVISION", "IMAGINE", "CONSTELLATION", "ESSENTIAL", "INTEGRATED", "INNOVATIVE", "STRATEGIC", "STRATEGY", "SOUND",
    "ENTERPRISE", "PUBLIC", "PRIVATE", "OLD", "LITTLE", "MIDDLE", "UPPER", "LOWER", "EAST", "WEST", "NORTH", "SOUTH"}


# ---------- a year of daily returns ----------
def book(bank, tickers: list[str], asof: str, lookback: int = LOOKBACK) -> dict:
    """Each ticker's daily return, and SPY's, on the ``lookback`` sessions ending at ``asof`` (adjusted closes as
    the bank holds them; a day beyond the scan's data-error bounds is left out)."""
    view = bank.view()
    mk = shocks.market(view)
    if mk is None:
        raise SystemExit(f"no {shocks.BENCH} or {shocks.VIX} prices in the bank: python -m brain scan run --refresh")
    cal = mk[0]
    cal = cal[cal <= asof][-(lookback + 1):]
    A = np.full((len(cal), len(tickers)), np.nan)
    for j, t in enumerate(tickers):
        df = shocks._series(view, t)
        if df is not None and len(df):
            A[:, j] = shocks._aligned(df, "adjclose", cal)
    spy = shocks._aligned(shocks._series(view, shocks.BENCH), "adjclose", cal)
    with np.errstate(all="ignore"):
        R = A[1:] / A[:-1] - 1
        m = spy[1:] / spy[:-1] - 1
    R[(R > shocks.BAD_UP) | (R < shocks.BAD_DOWN)] = np.nan
    return {"asof": asof, "cal": cal, "tickers": list(tickers), "R": R, "m": m}


def cached_book(bank, tickers: list[str], asof: str, lookback: int = LOOKBACK) -> dict:
    """``book``, kept for the day in the local cache: a scan's close does not change between its checks."""
    key = hashlib.sha256(("\n".join(tickers) + f"|{lookback}").encode()).hexdigest()[:12]
    p = config.home() / "cache" / "connect" / f"{asof}-{key}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return {"asof": asof, "cal": z["cal"].astype(str), "tickers": list(tickers), "R": z["R"].astype(float),
                "m": z["m"].astype(float)}
    b = book(bank, tickers, asof, lookback)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, cal=b["cal"].astype("U10"), R=b["R"].astype(np.float32), m=b["m"])
    return b


def residuals(R: np.ndarray, m: np.ndarray, min_obs: int = MIN_OBS) -> tuple[np.ndarray, np.ndarray]:
    """Each stock's daily return with the market's part taken out, by its own beta to SPY over the window, and
    those betas. A stock with fewer than ``min_obs`` sessions gets none."""
    ok = ~np.isnan(R) & ~np.isnan(m)[:, None]
    n = ok.sum(0)
    with np.errstate(all="ignore"):
        mbar = np.where(ok, m[:, None], 0.0).sum(0) / n
        rbar = np.where(ok, R, 0.0).sum(0) / n
        dm = np.where(ok, m[:, None] - mbar, 0.0)
        dr = np.where(ok, R - rbar, 0.0)
        beta = (dm * dr).sum(0) / (dm ** 2).sum(0)
        E = np.where(ok, dr - beta * dm, np.nan)
    E[:, n < min_obs] = np.nan
    beta = np.where(n < min_obs, np.nan, beta)
    return E, beta


def correlations(E: np.ndarray, clip: float = CLIP_Z, min_obs: int = MIN_OBS) -> np.ndarray:
    """The correlation of every pair of stocks' market-adjusted daily returns over the sessions both traded, each
    day clipped at ``clip`` of the stock's usual days. Pairs sharing fewer than ``min_obs`` sessions get none."""
    with np.errstate(all="ignore"):
        Z = np.clip(E / np.nanstd(E, axis=0), -clip, clip)
    ok = ~np.isnan(Z)
    Z0, M = np.where(ok, Z, 0.0).astype(np.float32), ok.astype(np.float32)
    n = M.T @ M
    with np.errstate(all="ignore"):
        q = (Z0 ** 2).T @ M                     # q[i, j]: stock i's squares over the days it shares with j
        C = (Z0.T @ Z0) / np.sqrt(q * q.T)
    C[n < min_obs] = np.nan
    np.fill_diagonal(C, np.nan)
    return C


def partners(C: np.ndarray, tickers: list[str], t: str, k: int = PARTNERS, min_corr: float = MIN_CORR) -> list[tuple]:
    """The ``k`` stocks that moved most with ``t`` (its market-adjusted daily returns), the closest first."""
    if t not in tickers:
        return []
    row = C[tickers.index(t)]
    order = np.argsort(-np.nan_to_num(row, nan=-2.0))[:k]
    return [(tickers[i], float(row[i])) for i in order if row[i] >= min_corr]


# ---------- drivers ----------
def driver_changes(cal, fetch=None) -> dict[str, np.ndarray]:
    """Each driver's change on each session after the first of ``cal``: close to close, as a return, or in
    percentage points for the 10-year yield. A driver that trades on other days (bitcoin at the weekend) is read at
    its last close on or before each session."""
    import pandas as pd

    from brain.sources import prices
    fetch = fetch or (lambda sym, start: prices.fetch_daily(sym, start=start, ttl=6 * 3600)[0])
    start = (date.fromisoformat(str(cal[0])) - timedelta(days=10)).isoformat()
    out = {}
    for sym, _, unit in why.DRIVERS:
        try:
            df = fetch(sym, start)
        except Exception:  # noqa: BLE001, S112 - one driver's outage must not stop the rest
            continue
        s = pd.Series(df["close"].to_numpy(float), index=df["date"].to_numpy(dtype=str))
        s = s[~s.index.duplicated(keep="last")]
        lv = s.reindex(sorted(set(s.index) | {str(d) for d in cal})).ffill().reindex([str(d) for d in cal]).to_numpy()
        with np.errstate(all="ignore"):
            out[sym] = lv[1:] - lv[:-1] if unit == "points" else lv[1:] / lv[:-1] - 1
    return out


def exposures(R: np.ndarray, m: np.ndarray, D: dict[str, np.ndarray], min_obs: int = MIN_OBS,
              clip: float = CLIP_Z) -> dict:
    """One regression per stock: its daily return on the market's and on every driver's change that day. Returns
    each driver's coefficient (return per unit of the driver: per 1.00 for a return, per point for the yield), how
    many standard errors it is from zero, the stock's market beta in the same regression, and each driver's usual
    daily change (the standard deviation)."""
    syms = list(D)
    X = np.column_stack([np.ones(len(m)), m, *[D[s] for s in syms]]) if syms else np.empty((len(m), 0))
    okx = ~np.isnan(X).any(1) if syms else np.zeros(len(m), bool)
    B = np.full((R.shape[1], len(syms)), np.nan)
    T = np.full_like(B, np.nan)
    mb = np.full(R.shape[1], np.nan)
    for j in range(R.shape[1]):
        ok = okx & ~np.isnan(R[:, j])
        if ok.sum() < max(min_obs, X.shape[1] + 2):
            continue
        x, y = X[ok], R[ok, j]
        sd = y.std()
        y = np.clip(y, y.mean() - clip * sd, y.mean() + clip * sd)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        res = y - x @ coef
        se = np.sqrt(np.diag(np.linalg.pinv(x.T @ x)) * (res @ res) / (len(y) - x.shape[1]))
        with np.errstate(all="ignore"):
            B[j], T[j], mb[j] = coef[2:], coef[2:] / se[2:], coef[1]
    usual = {s: float(np.nanstd(D[s])) for s in syms}
    return {"symbols": syms, "beta": B, "t": T, "market": mb, "usual": usual}


def carried(expo: dict, j: int, moves: dict[str, float], min_t: float = MIN_T) -> list[dict]:
    """What the day's driver moves carried of stock ``j``'s move at its past-year rates: one line for each driver
    the stock has clearly moved with (``min_t`` standard errors), the largest first."""
    out = []
    for k, sym in enumerate(expo["symbols"]):
        b, t = expo["beta"][j, k], expo["t"][j, k]
        if sym in moves and np.isfinite(b) and np.isfinite(t) and abs(t) >= min_t:
            out.append({"driver": sym, "beta": float(b), "t": float(t), "implied": float(b * moves[sym])})
    return sorted(out, key=lambda x: -abs(x["implied"]))


def most_exposed(expo: dict, tickers: list[str], sym: str, k: int, min_t: float = MIN_T) -> list[tuple]:
    """The ``k`` stocks that moved most reliably with a driver over the window, either way: (ticker, beta, t)."""
    if sym not in expo["symbols"]:
        return []
    c = expo["symbols"].index(sym)
    t = np.nan_to_num(expo["t"][:, c], nan=0.0)
    order = np.argsort(-np.abs(t))[:k]
    return [(tickers[i], float(expo["beta"][i, c]), float(t[i])) for i in order if abs(t[i]) >= min_t]


# ---------- stories ----------
def feed_items(ticker: str, fetch=None) -> list[dict]:
    """A ticker's Yahoo Finance news feed: each item's title, summary, link, publisher and publication time."""
    from brain.net import fetch as net_fetch
    from brain.sources import news
    r = (fetch or net_fetch)(news.RSS.format(sym=ticker.upper()), ttl=3600)
    out = []
    for item in ET.fromstring(r.body).iter("item"):
        title, link, pub = ((item.findtext(k) or "").strip() for k in ("title", "link", "pubDate"))
        if title and link and pub:
            summary = " ".join(re.sub(r"<[^>]+>", " ", item.findtext("description") or "").split())
            out.append({"title": title, "summary": summary[:SUMMARY], "url": link,
                        "source": urlparse(link).hostname or "unknown", "at": rfc822_to_iso(pub)})
    return out


def name_key(company: str | None) -> str | None:
    """The words a headline would use for a company: its first word, or its first two when the first is too
    ordinary to name it ("Super Micro", "Global Payments"). None when nothing distinctive is left."""
    words = [w for w in re.split(r"[^A-Za-z0-9&']+", company or "") if w]
    while words and words[-1].upper() in why.GENERIC:
        words.pop()
    if not words:
        return None
    if words[0].upper() in COMMON:
        return " ".join(words[:2]) if len(words) > 1 else None
    return words[0] if len(words[0]) >= 3 else None


def named(title: str, names: dict[str, str | None]) -> list[str]:
    """The tickers among ``names`` (ticker -> company name) that a headline names: the ticker in brackets or as a
    word in capitals, or the company's name key."""
    caps = set(re.findall(r"(?<![A-Za-z.&])\$?([A-Z]{3,5})(?![A-Za-z&])", title)) - ACRONYMS
    out = []
    for t, company in names.items():
        if t in caps or re.search(rf"\((?:\s*(?:NYSE|NASDAQ|Nasdaq)\s*:)?\s*{re.escape(t)}\s*\)", title):
            out.append(t)
            continue
        key = name_key(company)
        if not key:
            continue
        if len(key) <= 3:      # a three-letter name ("FMC", "Vir") only as written, not inside a longer word
            hit = re.search(rf"(?<![A-Za-z]){re.escape(key.upper())}(?![A-Za-z])|(?<![A-Za-z]){re.escape(key.title())}"
                            rf"(?![A-Za-z])", title)
        else:
            hit = re.search(rf"(?<![A-Za-z]){re.escape(key)}(?![A-Za-z])", title, re.IGNORECASE)
        if hit:
            out.append(t)
    return out


def story_key(url: str) -> str:
    u = urlsplit(url)
    return f"{(u.hostname or '').lower()}{u.path.rstrip('/')}"


def stories(feeds: dict[str, list[dict]], since: str, names: dict[str, str | None],
            wrap_feeds: int = WRAP_FEEDS) -> list[dict]:
    """The stories published after ``since`` in the read feeds, each once, the newest first: the feeds that carry
    it, the stocks its title names, and whether it is a market wrap (carried by ``wrap_feeds`` feeds or more, naming
    WRAP_NAMES of the stocks or more, or titled like one). A wrap is part of the day's backdrop; it does not link
    the stocks it lists."""
    by: dict[str, dict] = {}
    for t, items in feeds.items():
        for it in items:
            if it["at"] <= since:
                continue
            s = by.setdefault(story_key(it["url"]), {**it, "feeds": []})
            if t not in s["feeds"]:
                s["feeds"].append(t)
    out = sorted(by.values(), key=lambda s: (s["at"], s["url"]), reverse=True)
    for s in out:
        s["names"] = named(s["title"], names)
        s["wrap"] = len(s["feeds"]) >= wrap_feeds or len(s["names"]) >= WRAP_NAMES or bool(WRAP.search(s["title"]))
    return out


def story_links(story: dict, movers: set[str]) -> list[str]:
    """The movers one story ties together: those whose feed carries it or whose name its title carries."""
    return sorted((set(story["feeds"]) | set(story["names"])) & movers)


# ---------- clusters ----------
def clusters(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Groups of nodes joined by any chain of edges, the largest first; a node with no edge is in no group."""
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return sorted((g for g in groups.values() if len(g) > 1), key=lambda g: (-len(g), g[0]))
