"""US equity option chains from Cboe's public delayed quotes (about 15 minutes late).

The JSON at cdn-api.cboe.com is what cboe.com's own quote pages read; the host publishes no robots.txt.
Requests identify the client and are spaced well apart (brain/net.py MIN_INTERVAL): the host rate-limits,
and a refusal is an answer, not something to get around. Personal research only; snapshots stay in the
gitignored home. Nothing here places or routes an order.
"""
from __future__ import annotations

import json
import re

from brain.net import fetch

URL = "https://cdn-api.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
SOURCE_NAME = "Cboe delayed quotes"
TTL = 15 * 60
OCC = re.compile(r"^([A-Z.]+?)(\d{6})([CP])(\d{8})$")
MAX_SPREAD = 0.5   # a leg whose bid-ask spread is over half its mid is not a usable quote


def parse(data: dict) -> dict:
    """The chain as {expiry: {strike: {"C": quote, "P": quote}}} plus the underlying's spot and stamp."""
    d = data.get("data") or {}
    chain: dict[str, dict[float, dict]] = {}
    for o in d.get("options") or []:
        m = OCC.match(str(o.get("option", "")))
        if not m:
            continue
        exp = f"20{m[2][:2]}-{m[2][2:4]}-{m[2][4:]}"
        q = {k: float(o.get(k) or 0) for k in ("bid", "ask", "iv", "open_interest", "volume")}
        chain.setdefault(exp, {}).setdefault(int(m[4]) / 1000, {})[m[3]] = q
    spot = d.get("current_price") or d.get("close")
    return {"symbol": d.get("symbol"), "spot": float(spot) if spot else None, "asof": data.get("timestamp"),
            "iv30": d.get("iv30"), "chain": chain}


def fetch_chain(symbol: str, ttl: float = TTL) -> dict:
    r = fetch(URL.format(symbol=symbol.upper()), ttl=ttl)
    out = parse(json.loads(r.body))
    out["fetched_at"] = r.fetched_at
    return out


def usable(q: dict | None) -> bool:
    if not q or q["bid"] <= 0 or q["ask"] <= 0:
        return False
    return (q["ask"] - q["bid"]) / ((q["ask"] + q["bid"]) / 2) <= MAX_SPREAD


def first_expiry(chain: dict, on_or_after: str) -> str | None:
    later = sorted(e for e in chain if e >= on_or_after)
    return later[0] if later else None


def atm_iv(c: dict, expiry: str) -> float | None:
    """The at-the-money implied volatility on ``expiry`` (the call's and put's average), or None."""
    spot, legs = c["spot"], c["chain"].get(expiry) or {}
    both = [k for k, v in legs.items() if "C" in v and "P" in v and v["C"]["iv"] > 0 and v["P"]["iv"] > 0]
    if not spot or not both:
        return None
    k = min(both, key=lambda x: abs(x - spot))
    return (legs[k]["C"]["iv"] + legs[k]["P"]["iv"]) / 2


def event_bump(c: dict, expiry: str) -> float | None:
    """The expiry's at-the-money volatility over the next expiry's. A report before the first expiry lifts
    it well above the later one; a ratio near 1 means the options see no event in between."""
    later = sorted(e for e in c["chain"] if e > expiry)
    a = atm_iv(c, expiry)
    b = atm_iv(c, later[0]) if later else None
    return round(a / b, 3) if a and b else None


def straddle(c: dict, expiry: str) -> dict | None:
    """The at-the-money straddle on ``expiry``: its mid, its cost at the ask (one contract, x100) and the
    move it prices (mid / spot). None when the chain has no usable pair near the money."""
    spot, legs = c["spot"], c["chain"].get(expiry) or {}
    both = [k for k, v in legs.items() if "C" in v and "P" in v]
    if not spot or not both:
        return None
    k = min(both, key=lambda x: abs(x - spot))
    call, put = legs[k]["C"], legs[k]["P"]
    if not (usable(call) and usable(put)):
        return None
    mid = (call["bid"] + call["ask"]) / 2 + (put["bid"] + put["ask"]) / 2
    return {"expiry": expiry, "strike": k, "mid": round(mid, 4), "cost": round((call["ask"] + put["ask"]) * 100, 2),
            "implied_pct": round(mid / spot * 100, 2)}


def strangle(c: dict, expiry: str, width_pct: float) -> dict | None:
    """A call and a put about ``width_pct`` out of the money each way, at the ask (one contract each)."""
    spot, legs = c["spot"], c["chain"].get(expiry) or {}
    calls = {k: v["C"] for k, v in legs.items() if "C" in v and k > spot}
    puts = {k: v["P"] for k, v in legs.items() if "P" in v and k < spot}
    if not spot or not calls or not puts:
        return None
    kc = min(calls, key=lambda x: abs(x - spot * (1 + width_pct / 100)))
    kp = min(puts, key=lambda x: abs(x - spot * (1 - width_pct / 100)))
    if not (usable(calls[kc]) and usable(puts[kp])):
        return None
    cost = (calls[kc]["ask"] + puts[kp]["ask"]) * 100
    return {"expiry": expiry, "call": kc, "put": kp, "cost": round(cost, 2),
            "breakeven_up_pct": round((kc + cost / 100) / spot * 100 - 100, 2),
            "breakeven_down_pct": round(100 - (kp - cost / 100) / spot * 100, 2)}
