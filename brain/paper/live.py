"""The live paper trader: one step per day, after the close.

    close   if a book is open and HORIZON sessions have passed since its entry close, every stock it
            scored is marked at that close; the book's P&L (after costs) is recorded; the outcomes
            update the learner.
    open    with no open book: score every stock from today's features (prices, cross-market links,
            and today's tagged headlines), and take the long-short book if the predicted spread beats
            the round-trip cost; otherwise record the scores and stay flat (it still learns from them).

Everything is written to research/paper/ledger.jsonl (append-only, one event per line) and the
learner and open book to research/paper/state.json, before any outcome is known. Commit both after
each step: the container this runs in is temporary.

Paper only. Nothing here places an order.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np

from brain import config
from brain.paper import data, features, news
from brain.paper.learner import Learner
from brain.paper.replay import HORIZON, ROUND_TRIP, TOP, stats
from brain.sources import news as feed
from brain.sources import prices
from brain.timeutil import now_iso

DIR = config.ROOT / "research" / "paper"
LEDGER, STATE = DIR / "ledger.jsonl", DIR / "state.json"
NAMES = list(features.PRICE_FEATURES) + list(news.NEWS_FEATURES)
NEWS_HOURS = 24


def _append(event: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"at": now_iso(), **event}, separators=(",", ":")) + "\n")


def events() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()] if LEDGER.exists() else []


def load() -> dict:
    if STATE.exists():
        s = json.loads(STATE.read_text())
        s["learner"] = Learner.load(s["learner"])
        return s
    return {"learner": Learner(NAMES), "book": None}


def save(s: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    out = dict(s, learner=s["learner"].state())
    STATE.write_text(json.dumps(out, indent=1))


def seed_from_replay(weights: list[float], s: dict) -> None:
    """Start the price weights where the replay left them; news weights stay at 0 (no history)."""
    s["learner"].w[: len(weights)] = weights


def headlines(stocks: list[str], now: datetime) -> dict:
    """Tag counts per stock from headlines published in the last NEWS_HOURS that name the company."""
    out, kept = {}, []
    since = now - timedelta(hours=NEWS_HOURS)
    for t in stocks:
        counts = dict.fromkeys(news.NEWS_FEATURES, 0)
        try:
            meta = prices.fetch_daily(t, start=(now - timedelta(days=10)).date().isoformat())[2]
            names = news.aliases(t, meta.get("longName"))
            for f in feed.ingest(t):
                when = datetime.fromisoformat(f.known_at.replace("Z", "+00:00"))
                if when < since or when > now or not news.about(f.value, names):
                    continue
                for tg in news.tag(f.value):
                    counts[tg] += 1
                    kept.append({"ticker": t, "tag": tg, "headline": f.value, "url": f.source.get("url"),
                                 "known_at": f.known_at})
        except Exception as e:                              # a feed that fails counts as no news, and says so
            kept.append({"ticker": t, "error": f"{type(e).__name__}: {e}"[:200]})
        out[t] = counts
    return {"counts": out, "kept": kept}


def step(now: datetime | None = None, use_news: bool = True) -> dict:
    now = now or datetime.now(timezone.utc)
    s = load()
    L: Learner = s["learner"]
    stocks = data.universe()
    px = data.closes(stocks + [m for m in data.MACRO if m != "SPY"], start=(now - timedelta(days=800)).date().isoformat())
    stocks = [t for t in stocks if t in px and px[t].iloc[-features.YEAR:].notna().all()]
    dates = [str(d) for d in px.index]
    report = {"date": dates[-1], "closed": None, "opened": None}
    book = s.get("book")
    if book and book["entry"] in dates and len(dates) - 1 - dates.index(book["entry"]) >= HORIZON:
        i0 = dates.index(book["entry"])
        y = features.forward(px, i0, book["stocks"], HORIZON).values
        idx = {t: k for k, t in enumerate(book["stocks"])}
        gross = (np.mean([y[idx[t]] for t in book["longs"]]) - np.mean([y[idx[t]] for t in book["shorts"]])
                 if book["trade"] else 0.0)
        net = float(gross - ROUND_TRIP) if book["trade"] else 0.0
        L.update(np.array(book["X"]), y)
        report["closed"] = {"entry": book["entry"], "exit": dates[i0 + HORIZON], "trade": book["trade"],
                            "gross": float(gross), "net": net}
        _append({"event": "close", **report["closed"], "longs": book["longs"], "shorts": book["shorts"],
                 "outcomes": dict(zip(book["stocks"], np.round(y, 6).tolist())), "w_after": L.w.round(6).tolist()})
        s["book"] = book = None
    if book is None:
        i = len(px) - 1
        F = features.at(px, i, stocks)
        nw = headlines(stocks, now) if use_news else {"counts": {t: dict.fromkeys(news.NEWS_FEATURES, 0) for t in stocks}, "kept": []}
        for tg in news.NEWS_FEATURES:
            F[tg] = [min(3, nw["counts"][t][tg]) for t in stocks]
        X = F[NAMES].values
        pred = L.predict(X)
        order = np.argsort(pred)
        longs, shorts = [stocks[j] for j in order[-TOP:]], [stocks[j] for j in order[:TOP]]
        spread_hat = float(pred[order[-TOP:]].mean() - pred[order[:TOP]].mean())
        trade = L.updates >= 1 and spread_hat > ROUND_TRIP
        s["book"] = {"entry": dates[i], "stocks": stocks, "X": X.round(6).tolist(), "trade": bool(trade),
                     "longs": longs if trade else [], "shorts": shorts if trade else [], "spread_hat": spread_hat}
        report["opened"] = {"entry": dates[i], "trade": bool(trade), "longs": s["book"]["longs"],
                            "shorts": s["book"]["shorts"], "spread_hat": spread_hat, "headlines": nw["kept"]}
        _append({"event": "open", "entry": dates[i], "trade": bool(trade), "longs": s["book"]["longs"],
                 "shorts": s["book"]["shorts"], "spread_hat": spread_hat, "w": L.w.round(6).tolist(),
                 "names": NAMES, "headlines": nw["kept"]})
    save(s)
    return report


def status() -> dict:
    ev = events()
    closes = [e for e in ev if e["event"] == "close"]
    s = load()
    L = s["learner"]
    return {"closed_weeks": len(closes), "traded_weeks": sum(e["trade"] for e in closes),
            "stats": stats([e["net"] for e in closes]) if closes else None,
            "weights": dict(zip(L.names, np.round(L.w, 6).tolist())), "updates": L.updates,
            "open": s.get("book") and {k: s["book"][k] for k in ("entry", "trade", "longs", "shorts", "spread_hat")}}
