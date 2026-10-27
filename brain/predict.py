"""The agent's own calls, written down before the outcome and scored after it (docs/PREDICT.md).

A call names a ticker, a direction (up, down) or a size (move), the chance it is right, and the date it is
judged on. A paper trade can ride along: what was bought and what it cost, closed later at its exit price.
The record is append-only and committed (research/predictions.jsonl): a call is never edited; a mistake is
withdrawn with a reason. `score` judges each call against the bank's daily closes once its date has closed,
and reports the hit rate, the Brier score against a coin flip, and the paper trades' profit and loss.

None of this is evidence for the engine: a call carries weight only through this record, and fewer than
MIN_CALLS judged calls say nothing either way.
"""
from __future__ import annotations

import hashlib

from brain import config, series
from brain.bank import Bank, canon
from brain.research import ledger
from brain.timeutil import now_iso

CALLS = ("up", "down", "move")
MIN_CALLS = 30  # the same bar the engine sets for a source's record (CLAUDE.md, 'Claims are not evidence')


def _path(path=None):
    return path or config.PREDICTIONS


def read(path=None) -> list[dict]:
    return ledger.read(_path(path))


def _closes(bank: Bank, ticker: str):
    f = bank.view().latest(ticker.upper(), "series.price_daily")
    if f is None:
        return None
    df = series.load(f.value)
    return list(zip(df["date"].astype(str), df["close"].astype(float)))


def add(bank: Bank, ticker: str, call: str, prob: float, by: str, thesis: str, *, threshold: float = 0.0,
        price: float | None = None, sources=(), trade: dict | None = None, path=None, made_at: str | None = None) -> dict:
    """Record a call. ``prob`` is the chance the call is right; ``threshold`` (in percent) is how far the close
    on ``by`` must be from the start price: up/down need at least that move their way, move needs it either way."""
    if call not in CALLS:
        raise ValueError(f"call must be one of {CALLS}")
    if not 0 < prob < 1:
        raise ValueError("prob is a chance strictly between 0 and 1")
    if threshold < 0 or (call == "move" and threshold <= 0):
        raise ValueError("threshold is a positive percent (required for a move call)")
    made_at = made_at or now_iso()
    if by < made_at[:10]:
        raise ValueError(f"the judging date {by} is already past ({made_at[:10]})")
    closes = _closes(bank, ticker)
    if closes and closes[-1][0] >= by:   # a call written once its deciding close is known is not a forecast
        raise ValueError(f"the bank already has {ticker.upper()}'s close for {closes[-1][0]}: {by}'s outcome is known")
    start_date = made_at[:10]
    if price is None:
        if not closes:
            raise ValueError(f"no daily closes for {ticker.upper()} in the bank: ingest it, or give --price")
        start_date, price = closes[-1]
    rec = {"type": "call", "made_at": made_at, "ticker": ticker.upper(), "call": call, "prob": round(float(prob), 4),
           "threshold_pct": float(threshold), "start_price": float(price), "start_date": start_date, "by": by,
           "thesis": thesis.strip(), "sources": list(sources)}
    if trade:
        rec["trade"] = {"desc": trade["desc"], "paid": float(trade["paid"]), "qty": int(trade.get("qty", 1)),
                        "multiplier": int(trade.get("multiplier", 100))}
    rec["id"] = hashlib.sha256(canon(rec).encode()).hexdigest()[:8]
    if any(r.get("id") == rec["id"] for r in read(path)):
        raise ValueError(f"call {rec['id']} is already recorded")
    ledger.append(rec, _path(path))
    return rec


def _find(rid: str, path=None) -> dict:
    hits = [r for r in read(path) if r["type"] == "call" and r["id"].startswith(rid)]
    if len(hits) != 1:
        raise ValueError(f"{'no' if not hits else 'more than one'} call matches {rid!r}")
    return hits[0]


def close(rid: str, exit_price: float, note: str = "", path=None) -> dict:
    """Close a call's paper trade at ``exit_price`` (per unit, the same basis as what was paid)."""
    c = _find(rid, path)
    if "trade" not in c:
        raise ValueError(f"call {c['id']} has no paper trade to close")
    if any(r["type"] == "close" and r["id"] == c["id"] for r in read(path)):
        raise ValueError(f"call {c['id']}'s trade is already closed")
    rec = {"type": "close", "id": c["id"], "closed_at": now_iso(), "exit": float(exit_price), "note": note.strip()}
    ledger.append(rec, _path(path))
    return rec


def withdraw(rid: str, reason: str, path=None) -> dict:
    """A mistaken call (a wrong price, a typo) leaves the score with its reason; the line itself stays."""
    if not reason.strip():
        raise ValueError("a withdrawal needs a reason")
    c = _find(rid, path)
    rec = {"type": "withdraw", "id": c["id"], "at": now_iso(), "reason": reason.strip()}
    ledger.append(rec, _path(path))
    return rec


def state(bank: Bank, path=None) -> list[dict]:
    """Every call with its status: open, judged (hit or miss) or withdrawn, and its trade's result if closed."""
    recs = read(path)
    gone = {r["id"]: r for r in recs if r["type"] == "withdraw"}
    exits = {r["id"]: r for r in recs if r["type"] == "close"}
    cache: dict[str, list | None] = {}
    out = []
    for c in (r for r in recs if r["type"] == "call"):
        row = dict(c)
        if c["id"] in gone:
            row["status"] = "withdrawn"
            row["reason"] = gone[c["id"]]["reason"]
        else:
            closes = cache.setdefault(c["ticker"], _closes(bank, c["ticker"]))
            judged = _judge(c, closes)
            row.update(judged) if judged else row.update(status="open")
        if "trade" in c and c["id"] in exits:
            t, x = c["trade"], exits[c["id"]]["exit"]
            row["trade_pnl"] = round((x - t["paid"]) * t["qty"] * t["multiplier"], 2)
            row["trade_return"] = round(x / t["paid"] - 1, 4) if t["paid"] else None
            row["exit"] = x
        out.append(row)
    return out


def _judge(c: dict, closes) -> dict | None:
    """The call's outcome from the last close on or before its date. None until the bank's daily series
    reaches that date (the price source keeps a session out until it has closed)."""
    if not closes or closes[-1][0] < c["by"]:
        return None
    final = [p for d, p in closes if d <= c["by"]]
    if not final:
        return None
    move = (final[-1] / c["start_price"] - 1) * 100
    need = c["threshold_pct"]
    hit = {"up": move >= need and move > 0, "down": move <= -need and move < 0, "move": abs(move) >= need}[c["call"]]
    return {"status": "hit" if hit else "miss", "move_pct": round(move, 2), "final_price": final[-1],
            "brier": round((c["prob"] - (1.0 if hit else 0.0)) ** 2, 4)}


def score(rows: list[dict]) -> dict:
    """Hit rate, mean stated chance, Brier score (a coin flip scores 0.25; lower is better) and paper P&L."""
    judged = [r for r in rows if r["status"] in ("hit", "miss")]
    trades = [r for r in rows if "trade_pnl" in r and r["status"] != "withdrawn"]
    out = {"calls": len(rows), "open": sum(r["status"] == "open" for r in rows),
           "withdrawn": sum(r["status"] == "withdrawn" for r in rows), "judged": len(judged),
           "enough": len(judged) >= MIN_CALLS, "min_calls": MIN_CALLS}
    if judged:
        hits = sum(r["status"] == "hit" for r in judged)
        out.update(hits=hits, hit_rate=round(hits / len(judged), 3),
                   mean_prob=round(sum(r["prob"] for r in judged) / len(judged), 3),
                   brier=round(sum(r["brier"] for r in judged) / len(judged), 4), coin_flip_brier=0.25)
    if trades:
        cost = sum(r["trade"]["paid"] * r["trade"]["qty"] * r["trade"]["multiplier"] for r in trades)
        pnl = sum(r["trade_pnl"] for r in trades)
        out.update(trades_closed=len(trades), trade_wins=sum(r["trade_pnl"] > 0 for r in trades),
                   trade_cost=round(cost, 2), trade_pnl=round(pnl, 2), trade_return=round(pnl / cost, 4) if cost else None)
    return out


def _fmt_row(r: dict) -> str:
    need = f" by {r['threshold_pct']:g}%+" if r["threshold_pct"] else ""
    head = f"{r['id']}  {r['ticker']:6} {r['call']}{need} by {r['by']} at {r['prob']:.0%} (from {r['start_price']:g} on {r['start_date']})"
    tail = {"open": "open", "withdrawn": f"withdrawn: {r.get('reason', '')}"}.get(
        r["status"], f"{r['status'].upper()} ({r.get('move_pct', 0):+.1f}% to {r.get('final_price', 0):g})")
    trade = ""
    if "trade" in r:
        t = r["trade"]
        trade = f"\n          trade: {t['desc']} x{t['qty']} paid {t['paid']:g}" + (
            f", exit {r['exit']:g}: {r['trade_pnl']:+.2f} ({r['trade_return']:+.0%})" if "exit" in r else ", still open")
    return f"{head}  ->  {tail}{trade}\n          why: {r['thesis']}"


def _cmd_add(a):
    trade = None
    if a.trade:
        if a.paid is None:
            raise SystemExit("--trade needs --paid")
        trade = {"desc": a.trade, "paid": a.paid, "qty": a.qty, "multiplier": a.multiplier}
    try:
        rec = add(Bank(), a.ticker, a.call, a.prob, a.by, a.thesis, threshold=a.threshold, price=a.price,
                  sources=a.source or (), trade=trade)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print(f"Recorded {rec['id']}: {rec['ticker']} {rec['call']} by {rec['by']} at {rec['prob']:.0%} "
          f"(start {rec['start_price']:g} on {rec['start_date']}). Commit research/predictions.jsonl.")


def _cmd_close(a):
    try:
        rec = close(a.id, a.exit, a.note or "")
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print(f"Closed {rec['id']} at {rec['exit']:g}.")


def _cmd_withdraw(a):
    try:
        rec = withdraw(a.id, a.reason)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print(f"Withdrew {rec['id']}: {rec['reason']}")


def _refresh(tickers, log=print):
    from brain.ingest import ingest
    for t in sorted(set(tickers)):
        try:
            ingest(Bank(), t, ("prices",), log=log)
        except Exception as e:  # noqa: BLE001 - one ticker's outage must not stop the rest
            log(f"  {t}: prices not refreshed ({type(e).__name__}: {e})")


def _cmd_list(a):
    if a.refresh:
        _refresh(r["ticker"] for r in read() if r["type"] == "call")
    rows = state(Bank())
    if a.open:
        rows = [r for r in rows if r["status"] == "open" or ("trade" in r and "exit" not in r and r["status"] != "withdrawn")]
    if not rows:
        print("No calls recorded." if not a.open else "No open calls.")
    for r in rows:
        print(_fmt_row(r))


def _cmd_score(a):
    if a.refresh:
        _refresh(r["ticker"] for r in read() if r["type"] == "call")
    s = score(state(Bank()))
    print(f"Calls {s['calls']}: {s['judged']} judged, {s['open']} open, {s['withdrawn']} withdrawn.")
    if s["judged"]:
        print(f"Judged: {s['hits']} right of {s['judged']} ({s['hit_rate']:.0%}); stated chance on average "
              f"{s['mean_prob']:.0%}. Brier {s['brier']:.3f} (a coin flip scores 0.250; lower is better).")
    if s.get("trades_closed"):
        print(f"Paper trades closed {s['trades_closed']}: {s['trade_wins']} won; cost {s['trade_cost']:.2f}, "
              f"P&L {s['trade_pnl']:+.2f} ({s['trade_return']:+.0%}).")
    if not s["enough"]:
        print(f"Fewer than {s['min_calls']} judged calls: this record says nothing yet, either way.")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("predict", help="the agent's own calls, written before the outcome and scored after (docs/PREDICT.md)")
    ps = p.add_subparsers(dest="predict_cmd", required=True)
    s = ps.add_parser("add", help="record a call: TICKER up|down|move --prob P --by DATE --thesis TEXT")
    s.add_argument("ticker")
    s.add_argument("call", choices=CALLS)
    s.add_argument("--prob", type=float, required=True, help="the chance the call is right, 0-1")
    s.add_argument("--by", required=True, help="the date the call is judged on (its close), YYYY-MM-DD")
    s.add_argument("--threshold", type=float, default=0.0, help="percent move the call needs (required for move)")
    s.add_argument("--price", type=float, help="start price (default: the bank's latest close)")
    s.add_argument("--thesis", required=True, help="why, in one line")
    s.add_argument("--source", action="append", help="a URL the call rests on (repeatable)")
    s.add_argument("--trade", help="a paper trade riding on the call, e.g. 'PYPL 2026-10-30 52 put'")
    s.add_argument("--paid", type=float, help="the trade's price per unit (an option's per-share premium)")
    s.add_argument("--qty", type=int, default=1)
    s.add_argument("--multiplier", type=int, default=100, help="units per contract (100 for US equity options)")
    s.set_defaults(fn=_cmd_add)
    s = ps.add_parser("close", help="close a call's paper trade at its exit price")
    s.add_argument("id")
    s.add_argument("--exit", type=float, required=True)
    s.add_argument("--note")
    s.set_defaults(fn=_cmd_close)
    s = ps.add_parser("withdraw", help="take a mistaken call out of the score, with a reason (the line stays)")
    s.add_argument("id")
    s.add_argument("--reason", required=True)
    s.set_defaults(fn=_cmd_withdraw)
    s = ps.add_parser("list", help="every call and its status")
    s.add_argument("--open", action="store_true", help="only calls or trades still open")
    s.add_argument("--refresh", action="store_true", help="re-read the calls' daily prices first")
    s.set_defaults(fn=_cmd_list)
    s = ps.add_parser("score", help="hit rate, Brier score and paper P&L of the judged calls")
    s.add_argument("--refresh", action="store_true", help="re-read the calls' daily prices first")
    s.set_defaults(fn=_cmd_score)
