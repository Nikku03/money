"""Decisions: weigh the bank's tested evidence on one stock under a declared policy, record it, score it later.

docs/DECIDE.md is the contract. One run (``brain decide TICKER``) reads the bank as it stands at
T_RUN, the end of the current UTC day; optionally lets the model read the web (``--read``) and make
its own call (a ``judgment`` fact); weighs every piece of evidence by its tested record under policy
D1; and writes the outcome as a ``decision`` fact citing every input.

D1 weighs only what has been tested: a research hypothesis that passed its TEST gate (and would hold
the stock), or a source (a web site, or the model) whose scored bets beat a coin after a
false-discovery correction (``claims.track_record``). Everything else weighs 0, with the reason
shown. Today every hypothesis has failed and no source has a record, so the answer is NO_EDGE, with
the sources' unproven lean beside it, labelled as not evidence.

Paper only: nothing here places orders or connects to a broker.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import sys
import time
from datetime import date, timedelta

import numpy as np

from brain import brief, config, markets
from brain.bank import Bank, Fact
from brain.claims import (UNPRICED_MAX_SHARE, anchor, bets, scored_at, source_key, tally, too_unpriced,
                          track_record)
from brain.metrics import PriceBook
from brain.research import ledger
from brain.timeutil import days_between, now_iso, today

POLICY = {
    "id": "D1",
    "benchmark": "SPY",
    "horizon_days": 21,            # trading days; decisions, judgments and claims are scored over this
    "max_lag_days": 5,             # a call counts in a decision only if its entry is 0..5 SPY sessions before the decision's
    "price_max_age_days": 7,       # NO_DATA when the stock's latest close is older (calendar days)
    "threshold": 0.5,              # BUY when score >= 0.5, SELL when score <= -0.5, else NO_EDGE
    "signal_weight": 1.0,          # a PASS hypothesis that would hold the stock (section 3.2)
    "signal_rules": {"min_coverage": 0.5, "ranking": "last completed month-end before T_RUN"},
    "source_gate": {"min_bets": 30, "p_lt": 0.05, "bh_q": 0.10, "alpha_lower": 0.05,
                    "mean_signed_excess_gt": 0.0,
                    # claims.GRACE_SESSIONS and claims.UNPRICED_MAX_SHARE, named here so the sha covers them
                    "grace_sessions": 5, "unpriced_max_share": 0.10},
    "source_weight": "max(0, 2*L - 1), L = one-sided 95% Clopper-Pearson lower bound of the bet hit rate",
    "source_combine": "only the proven source item with the largest weight counts; 0 when proven sources disagree",
}
POLICY_SHA = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()[:16]
# D1 is never edited (tests/test_decide.py freezes its sha). A change is D2, with its own sha.
KNOWN_POLICIES = {POLICY["id"]: POLICY_SHA}

DIRECTION = {"BUY": 1, "SELL": -1, "NO_EDGE": 0, "NO_DATA": 0}
SCOUT_SECONDS, JUDGE_SECONDS, EXPLAIN_SECONDS = 15 * 60, 10 * 60, 2 * 60   # wall-clock limits per step
ALERT_CONTEXT_DAYS = 7
NO_WEIGHT_NOTE = ("No evidence carries weight yet: every hypothesis failed its TEST gate or was not run, "
                  "and no source has a proven record.")
LABEL = {"BUY": "BUY: expect it to beat SPY", "SELL": "SELL: expect it to trail SPY",
         "NO_EDGE": "NO_EDGE: do nothing", "NO_DATA": "NO_DATA: cannot decide"}
WORD = {1: "up", -1: "down", 0: "none"}
# A step's error that says the model itself failed (describe_error's texts, a deadline), so the run's
# remaining model steps are skipped rather than each waiting out its own timeout.
_MODEL_DOWN = re.compile(r"model server|context window|tool calling|timed out|time limit|deadline|out of time|"
                         r"cannot reach|could not reach|unreachable", re.I)


def run_time() -> str:
    """T_RUN: the end of the current UTC day, as the terminal's ``data._now()``. A close fetched after the
    session is stamped as known at the end of its day, so a view at ``now`` would miss it."""
    return today() + "T23:59:59Z"


def scorable(ticker: str) -> tuple[bool, str]:
    """False for the benchmark itself and for any market symbol (markets.kind(t) is not None: an index,
    currency, commodity or coin). Their excess over SPY is built in (SPY's is 0, ^GSPC lags SPY's dividends
    98% of the time), so calls on them say nothing about skill. ETFs not in MARKETS (QQQ, VOO) pass; a
    later policy may require an operating company."""
    t = str(ticker).strip().upper()
    if t == POLICY["benchmark"]:
        return False, f"{t} is the benchmark: its excess over itself is 0, so a call on it says nothing about skill"
    k = markets.kind(t)
    if k is not None:
        return False, f"{t} is a market symbol ({k}), not a stock: its excess over SPY is built in"
    return True, ""


def item(kind: str, id_: str, direction: int = 0, weight: float = 0.0, why: str = "", refs=(), record=None) -> dict:
    """One piece of evidence as a decision records it. Every key is always there."""
    return {"kind": kind, "id": id_, "direction": int(direction), "weight": float(weight), "why": why,
            "refs": list(refs), "record": record}


def plain(x):
    """Plain JSON values only: numpy scalars become Python ones (the journal would write them as strings),
    a non-finite float becomes None, a tuple a list."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return str(x)


# ---------- the benchmark's calendar ----------
def _session_index(cal: np.ndarray, day: str) -> int:
    """Position on the benchmark calendar of the first session strictly after ``day``. Past the calendar's
    end (sessions the bank does not hold yet), weekdays stand in for sessions; holidays are not known ahead."""
    i = int(np.searchsorted(cal, day[:10], side="right"))
    if i < len(cal) or not len(cal):
        return i
    nxt = lambda s: np.datetime64(str(s)[:10], "D") + np.timedelta64(1, "D")  # noqa: E731
    return i + int(np.busday_count(nxt(cal[-1]), nxt(day)))


def last_month_end(book: PriceBook, t: str) -> str | None:
    """The last completed month-end before t on the benchmark's calendar: its last session in a month
    before t's. The ranking a PASS signal may act on is the one from that day."""
    b = POLICY["benchmark"]
    if not book.has(b):
        return None
    cal = book.dates[b]
    i = int(np.searchsorted(cal, t[:7] + "-01", side="left")) - 1
    return str(cal[i]) if i >= 0 else None


def data_gate(bank: Bank, ticker: str, t: str) -> tuple[bool, str, dict | None, PriceBook]:
    """(ok, why, price, book): NO_DATA when the bank holds no prices for the stock, its latest close is older
    than ``price_max_age_days``, or SPY has no close that recent."""
    T, b = ticker.upper(), POLICY["benchmark"]
    book = PriceBook(bank, [T, b], asof=t)
    view = bank.view(t)
    px = book.close_on(view, T) if book.has(T) else None
    if px is None:
        return False, f"no prices for {T}: ingest {T}", None, book
    d, close = px
    price = {"ref": book.snapshot[T].ref, "date": str(d), "close": float(close)}
    age = POLICY["price_max_age_days"]
    if days_between(d, t) > age:
        return False, f"latest close {d}: ingest {T}", price, book
    spy = book.close_on(view, b) if book.has(b) else None
    if spy is None or days_between(spy[0], t) > age:
        return False, f"{b} prices are stale: ingest {b}", price, book
    return True, "", price, book


# ---------- signals ----------
def _verdict_text(verdict: str, net, p) -> str:
    if verdict != "FAIL":
        return verdict  # "not run", "NO VERDICT (controls failed)", "WITHDRAWN: ..."
    bits = []
    if isinstance(net, (int, float)) and math.isfinite(net):
        bits.append(f"net excess {net * 100:+.2f}%/mo")
    if isinstance(p, (int, float)) and math.isfinite(p):
        bits.append(f"p {p:.2f}")
    return "FAIL on TEST" + (f" ({', '.join(bits)})" if bits else "")


def _verdicts() -> dict[str, dict]:
    from brain.research import run as research_run
    try:
        return ({r["id"]: r for r in research_run.verdicts() if research_run.is_hypothesis(r)}
                if config.LEDGER.exists() else {})
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _universe_tickers(name: str) -> set[str] | None:
    from brain.research import run as research_run
    try:
        return {str(x).upper() for x in research_run.load_universe(name)["tickers"]}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def signal_items(bank: Bank, ticker: str, t: str, book: PriceBook) -> list[dict]:
    """One item per current hypothesis. Weight 1 (direction up) needs all of: (a) its verdict is PASS;
    (b) its universe file lists the stock; (c) the ranking is from the last completed month-end before t;
    (d) the ranking covers at least ``min_coverage`` of the universe; (e) the backtest would hold the stock.
    Otherwise weight 0, and ``why`` names the first rule that failed."""
    from brain.research.hypotheses import BROAD, COMMON, current
    T = ticker.upper()
    try:
        # The whole ranking (decide never runs in a request loop), with prices read at T_RUN like every
        # other read in the run: a close ingested this evening marks the month-end rule (c) expects.
        ev = brief.evidence(bank, T, budget_s=None, asof=t)
    except brief.NoEvidence as e:
        return [item("signal", "evidence", why=f"no ranking: {e}")]
    except (Exception, SystemExit) as e:  # noqa: BLE001 - a ranking that cannot be computed weighs nothing
        return [item("signal", "evidence", why=f"no ranking ({type(e).__name__}: {e})")]
    rules = POLICY["signal_rules"]
    expected = last_month_end(book, t)
    rows = {r.get("hypothesis"): r for r in ev.get("signals") or []}
    broad = ev.get("broad") or {}
    verdicts: dict | None = None
    members: dict[str, set[str] | None] = {}
    out = []
    for h in current(ledger.withdrawn()):
        hid, uni = h["id"], h.get("universe") or COMMON["universe"]
        row = rows.get(hid)
        if row is None:  # not ranked per stock (H5's learner), or the broad ranking is unavailable
            verdicts = _verdicts() if verdicts is None else verdicts
            r = verdicts.get(hid)
            verdict = r["verdict"] if r else "not run"
            test = (r or {}).get("test") or {}
            if h["signal"] == "ridge":
                reason = "not ranked per stock"
            elif uni == BROAD:
                reason = broad.get("reason") or f"the {BROAD} ranking is {broad.get('status') or 'unavailable'}"
            else:
                reason = "not ranked"
            if verdict != "PASS":
                reason = f"{_verdict_text(verdict, test.get('mean_net_excess'), test.get('perm_p'))}; {reason}"
            out.append(item("signal", hid, why=reason, record={"verdict": verdict, "universe": uni,
                                                                "rebalance_date": None, "coverage": None, "held": None}))
            continue
        is_broad = bool(row.get("universe"))
        d = broad.get("rebalance_date") if is_broad else ev.get("rebalance_date")
        cov = broad.get("coverage") if is_broad else ev.get("coverage")
        held = row.get("held") if is_broad else row.get("in_top_third")
        verdict = row.get("verdict") or "not run"
        record = {"verdict": verdict, "universe": uni, "rebalance_date": d, "coverage": cov, "held": held}
        if uni not in members:
            members[uni] = _universe_tickers(uni)
        if verdict != "PASS":
            why = _verdict_text(verdict, row.get("test_net_excess"), row.get("perm_p"))
        elif members[uni] is None or T not in members[uni]:
            why = f"PASS on {uni}, which does not list this stock"
        elif d is None or d != expected:
            why = f"ranking as of {d or 'no date'}: ingest SPY and the universe"
        elif not (cov or {}).get("universe_size") or cov.get("ranked", 0) / cov["universe_size"] < rules["min_coverage"]:
            why = f"ranking covers {(cov or {}).get('ranked', 0)} of {(cov or {}).get('universe_size', '?')} stocks"
        elif held is not True:
            why = "PASS, but its backtest would not hold this stock"
        else:
            out.append(item("signal", hid, 1, POLICY["signal_weight"], record=record,
                            why=(f"PASS on TEST, and its backtest would hold this stock on the {d} ranking. It was "
                                 f"tested against the equal-weight universe from month-end, so {POLICY['id']}'s "
                                 f"score against {POLICY['benchmark']} over {POLICY['horizon_days']} days tests "
                                 f"{POLICY['id']}, not the hypothesis.")))
            continue
        out.append(item("signal", hid, why=why, record=record))
    return out


# ---------- sources ----------
def source_items(bank: Bank, ticker: str, t: str, cal: np.ndarray) -> tuple[list[dict], list[dict]]:
    """(items, context). A call (a claim or judgment about the stock, known and retrieved by t, with a
    direction) counts when its horizon is D1's and its entry is 0..max_lag_days sessions before this
    decision's; its source key's track record at t gives the item its weight. Other calls still open (their
    horizon not yet over) and non-directional calls in the lag window go to the context, weight 0."""
    T = ticker.upper()
    view = bank.view(t)
    horizon, max_lag = POLICY["horizon_days"], POLICY["max_lag_days"]
    here = _session_index(cal, t)
    calls = [f for a in ("claim", "judgment") for f in view.facts(T, a) if (f.retrieved_at or f.known_at) <= t]
    groups: dict[str, list[Fact]] = {}
    context = []
    for f in sorted(calls, key=lambda f: (anchor(f), f.id)):
        lag = here - _session_index(cal, anchor(f))
        h = int(f.context.get("horizon_days") or horizon)
        if not f.context.get("direction"):
            if 0 <= lag <= max_lag:
                context.append({"ref": f.ref, "why": "no direction"})
        elif h != horizon:
            if lag <= h:
                context.append({"ref": f.ref, "why": f"horizon {h}, not {POLICY['id']}'s {horizon}"})
        elif not 0 <= lag <= max_lag:
            if lag <= h:
                context.append({"ref": f.ref, "why": f"stale: entry {lag} sessions before this decision"})
        else:
            groups.setdefault(source_key(f), []).append(f)
    records = track_record(bank, t) if groups else {}
    out = []
    for key in sorted(groups):
        fs = groups[key]
        row = records.get(key)
        direction = int(np.sign(sum(int(np.sign(f.context["direction"])) for f in fs)))
        why = row["why"] if row else f"0 bets; needs {POLICY['source_gate']['min_bets']} bets"
        out.append(item("source", key, direction, row["weight"] if row else 0.0, why, [f.ref for f in fs], row))
    return out, context


def alert_context(bank: Bank, ticker: str, t: str) -> list[dict]:
    """Notice and high alerts on the stock known in the 7 days to t, read through watch.alerts (which
    withholds opted-out and LADD/PIA aircraft). Weight 0: alert rules are not tested. Info alerts are left
    out, as ALRT and the top bar leave them out: they are the headlines, one alert each, which would bury
    the rest of the context and the decision's inputs."""
    from brain import watch
    lo = (date.fromisoformat(t[:10]) - timedelta(days=ALERT_CONTEXT_DAYS)).isoformat() + t[10:]
    try:
        fs = watch.alerts(bank, entities=[ticker])
    except (Exception, SystemExit):  # noqa: BLE001 - context only
        return []
    return [{"ref": f.ref, "why": f"alert ({f.context.get('rule') or '?'}); alert rules are untested"}
            for f in sorted(fs, key=lambda f: (f.known_at, f.id)) if lo < f.known_at <= t and not watch.is_info(f)]


def combine(items: list[dict]) -> tuple[float, dict, float]:
    """(signal_score, source_term, score). Signals sum as written. Among source items with weight > 0 only
    the largest counts, and only when every such item points the same way: sources read the same public
    text (and the judge reads the claims), so they never add up."""
    signal = float(sum(i["weight"] * i["direction"] for i in items if i["kind"] == "signal"))
    weighted = [i for i in items if i["kind"] == "source" and i["weight"] > 0]
    term = {"id": None, "direction": 0, "weight": 0.0}
    if weighted:
        top = max(weighted, key=lambda i: i["weight"])  # the first of equal weights, in item order
        if len({i["direction"] for i in weighted}) == 1 and top["direction"]:
            term = {"id": top["id"], "direction": int(top["direction"]), "weight": float(top["weight"])}
    return signal, term, signal + term["direction"] * term["weight"]


def action_for(score: float) -> str:
    thr = POLICY["threshold"]
    s = round(score, 9)  # 1.0 - 0.5 is 0.5, not a hair under it
    return "BUY" if s >= thr else "SELL" if s <= -thr else "NO_EDGE"


def lean(items: list[dict]) -> dict:
    """The sources' unweighted, unproven lean: recorded so it can be scored, always labelled not evidence."""
    src = [i for i in items if i["kind"] == "source"]
    up = sum(1 for i in src if i["direction"] == 1)
    down = sum(1 for i in src if i["direction"] == -1)
    return {"direction": int(np.sign(up - down)), "up": up, "down": down, "sources": len(src)}


def gather(bank: Bank, ticker: str, t: str) -> dict:
    """{"items": [...], "context_refs": [...], "price": {...} | None, "data_ok": bool, "data_why": str,
        "source_term": {...}, "signal_score": float, "score": float}, plus "context_notes" (each context
    ref with why it weighs 0) and "lean"."""
    T = ticker.upper()
    ok, why = scorable(T)
    price = None
    if ok:
        ok, why, price, book = data_gate(bank, T, t)
    out = {"items": [], "context_refs": [], "context_notes": [], "price": price, "data_ok": ok, "data_why": why,
           "source_term": {"id": None, "direction": 0, "weight": 0.0}, "signal_score": 0.0, "score": 0.0,
           "lean": lean([])}
    if not ok:
        return out
    items = signal_items(bank, T, t, book)
    src, context = source_items(bank, T, t, book.dates[POLICY["benchmark"]])
    items += src
    context += alert_context(bank, T, t)
    signal, term, score = combine(items)
    return {**out, "items": items, "context_refs": list(dict.fromkeys(c["ref"] for c in context)),
            "context_notes": context, "source_term": term, "signal_score": signal, "score": score, "lean": lean(items)}


# ---------- the run ----------
def _judge_module():
    """brain.judge (the model's steps), imported when first needed so the engine runs without it."""
    from brain import judge as judge_mod
    return judge_mod


def _try_judge():
    try:
        return _judge_module(), None
    except Exception as e:  # noqa: BLE001 - brain.judge missing or broken: the engine still decides
        return None, f"brain.judge unavailable ({type(e).__name__}: {e})"


def _model_down(res) -> str | None:
    """The error text when a step's result says the model failed (not just that its answer was unusable).
    A result that carries ``down`` (the judge's) says so itself; otherwise its error is read."""
    if not isinstance(res, dict):
        return None
    if "down" in res:
        return str(res["down"]) if res["down"] else None
    if res.get("ok") is True:  # a recorded judgment may still list dropped cites
        return None
    err = res.get("error")
    return str(err) if err and _MODEL_DOWN.search(str(err)) else None


def _exc_down(e: BaseException) -> str | None:
    from brain import llm
    if isinstance(e, llm.LLMError):
        return llm.describe_error(e, None)[0] + f" ({e.message[:200]})"
    return None


def _fallback_template(d: dict) -> str:
    """Used only when brain.judge is not there: the action and why, from the decision's own numbers."""
    if d["action"] == "NO_DATA":
        return f"NO_DATA: {d['notes'][0] if d.get('notes') else 'no data'}. Nothing was decided or recorded."
    thr = d["threshold"]
    weighted = [i["id"] for i in d["items"] if i["weight"] > 0]
    return (f"{d['action']} under policy {d['policy']}: the score is {d['score']:+.2f}, and BUY needs +{thr:.2f}, "
            f"SELL -{thr:.2f}. " + (f"Weighted evidence: {', '.join(weighted)}." if weighted else NO_WEIGHT_NOTE)
            + " Paper only.")


def _template(judge_mod, d: dict) -> str:
    if judge_mod is not None:
        try:
            return str(judge_mod.template_explanation(d))
        except Exception:  # noqa: BLE001 - the template must never stop a decision
            pass
    return _fallback_template(d)


def _blank(T: str, t: str, notes: list[str], price=None) -> dict:
    """Section 3.4's shape with nothing decided (NO_DATA)."""
    return {"ok": True, "ticker": T, "action": "NO_DATA", "direction": 0, "score": 0.0,
            "threshold": POLICY["threshold"], "signal_score": 0.0, "source_term": {"id": None, "direction": 0, "weight": 0.0},
            "items": [], "lean": lean([]), "context_refs": [], "context_notes": [], "price": price, "judgment": None,
            "read": None, "explanation": "", "explained_by": "template", "explain_error": None, "unverified_numbers": [],
            "ref": None, "policy": POLICY["id"], "policy_sha": POLICY_SHA, "horizon_days": POLICY["horizon_days"],
            "as_of": t[:10], "notes": notes}


def history(bank: Bank, ticker: str) -> list[Fact]:
    """Every recorded decision on the stock, newest first. The bank lists a day's facts in the order the
    journal took them in, so two decisions made in the same second still come out in order."""
    return list(reversed(bank.facts(ticker.upper(), "decision")))


def latest(bank: Bank, ticker: str) -> Fact | None:
    h = history(bank, ticker)
    return h[0] if h else None


def as_dict(f: Fact) -> dict:
    """A decision fact as section 3.4's return shape (ok True, ref set)."""
    c = dict(f.context)
    base = _blank(f.entity, f.known_at, list(c.get("notes") or []))
    base.update({k: v for k, v in c.items()})
    base.update({"ok": True, "ticker": f.entity, "action": f.value, "ref": f.ref, "as_of": f.as_of,
                 "known_at": f.known_at})
    return base


def tickers_with_decisions(bank: Bank) -> list[str]:
    return sorted(e for e in bank.entities() if "decision" in Bank(bank.root, use_index=bank.use_index).attrs(e))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _noted_after(bank: Bank, T: str, noted, t: str) -> int:
    """How many of the scout's noted claims were retrieved after t (a read that ran past midnight UTC):
    gather leaves them out, as it leaves out any call not retrieved by T_RUN."""
    ids = {str(n.get("id")) for n in noted or [] if isinstance(n, dict) and n.get("id")}
    if not ids:
        return 0
    return sum(1 for f in bank.facts(T, "claim") if f.id[:8] in ids and (f.retrieved_at or f.known_at) > t)


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def decide(bank: Bank, ticker: str, *, do_ingest: bool = False, do_read: bool = False, do_judge: bool = True,
           do_explain: bool = True, client=None, log=print, max_pages: int = 6, llm_state: dict | None = None) -> dict:
    """Returns {"ok", "ticker", "action": "BUY" | "SELL" | "NO_EDGE" | "NO_DATA", "direction", "score",
    "threshold", "signal_score", "source_term", "items", "lean", "context_refs", "price", "judgment", "read",
    "explanation", "explained_by", "explain_error", "unverified_numbers", "ref" (None for NO_DATA),
    "policy", "policy_sha", "notes"}. Never raises on data, model or network problems: they become notes.

    ``llm_state`` is shared across one run's tickers: after a model timeout or error it holds
    {"down": error}, and the run's remaining model steps are skipped. It also remembers that the
    benchmark's prices were ingested, so a watchlist run fetches them once."""
    T = str(ticker).strip().upper()
    t = run_time()
    state = llm_state if llm_state is not None else {}
    notes: list[str] = []
    try:
        return _decide(bank, T, t, notes, state, do_ingest=do_ingest, do_read=do_read, do_judge=do_judge,
                       do_explain=do_explain, client=client, log=log, max_pages=max_pages)
    except Exception as e:  # noqa: BLE001 - a fault here is reported, never raised into a watchlist run
        d = _blank(T, t, notes + [f"decide failed ({type(e).__name__}: {e}); nothing recorded"])
        d["ok"] = False
        d["explanation"] = _fallback_template(d)
        return d


def _decide(bank: Bank, T: str, t: str, notes: list[str], state: dict, *, do_ingest, do_read, do_judge,
            do_explain, client, log, max_pages) -> dict:
    # 1. Only stocks: the benchmark and market symbols have their excess over SPY built in.
    ok, why = scorable(T)
    if not ok:
        d = _blank(T, t, [why] + notes)
        d["explanation"] = _template(_try_judge()[0], d)
        return d

    # 2. Ingest, each part guarded: a failure is a note and the run goes on to the data gate.
    if do_ingest:
        from brain import ingest as ingest_mod
        try:
            ingest_mod.ingest(bank, T, log=log)
        except (Exception, SystemExit) as e:  # SystemExit: SEC_USER_AGENT unset
            notes.append(f"ingest of {T} failed ({type(e).__name__}: {e}); decided on what the bank holds")
        if not state.get("benchmark_ingested"):
            from brain.sources import prices
            try:
                bank.add_many(prices.ingest(POLICY["benchmark"]))
                state["benchmark_ingested"] = True
            except (Exception, SystemExit) as e:
                notes.append(f"ingest of {POLICY['benchmark']} prices failed ({type(e).__name__}: {e})")

    # 3. The data gate, through the view at T_RUN.
    ok, why, price, _ = data_gate(bank, T, t)
    if not ok:
        asked = [s for s, on in (("read", do_read), ("judgment", do_judge)) if on]
        d = _blank(T, t, [why] + notes + ([f"{' and '.join(asked)} skipped: no data to decide on"] if asked else []),
                   price)
        d["explanation"] = _template(_try_judge()[0], d)
        return d

    judge_mod, judge_err = _try_judge()  # its template explains every decision; the model is asked only below
    ready, ready_why = False, ""
    if do_read or do_judge or do_explain:
        if judge_mod is not None:
            try:
                ready, ready_why = judge_mod.llm_ready()
            except Exception as e:  # noqa: BLE001
                ready, ready_why = False, f"{type(e).__name__}: {e}"

    said: set[str] = set()  # a reason is given in full once per decision, then referred to

    def skip(step: str) -> tuple[str, str] | None:
        """(the note, the full reason) when a model step cannot run, else None."""
        if state.get("down"):
            why = f"the model failed earlier in this run ({state['down']})"
        elif judge_mod is None:
            why = judge_err
        elif not ready:
            why = f"no model reachable ({ready_why or 'brain.llm.available() says no'})"
        else:
            return None
        short = why.split(" (")[0] + " (as above)" if why in said and " (" in why else why
        said.add(why)
        return f"{step} skipped: {short}", why

    def mark_down(err: str | None) -> None:
        if err and not state.get("down"):
            state["down"] = err

    # 4. Read: the scout notes claims from public pages.
    read = None
    if do_read:
        s = skip("read")
        if s:
            notes.append(s[0])
            read = {"claims_noted": 0, "pages_read": 0, "error": s[1]}
        else:
            from brain import scout as scout_mod
            kw = {"max_pages": max_pages, "log": log, "client": client}
            if _accepts(scout_mod.scout, "deadline"):
                kw["deadline"] = time.monotonic() + SCOUT_SECONDS
            log(f"{T}: the model reads public pages (at most {max_pages}, {SCOUT_SECONDS // 60} min)")
            try:
                out = scout_mod.scout(bank, T, **kw)
                read = {"claims_noted": len(out.get("notes") or []),
                        "pages_read": sum(1 for p in out.get("pages") or [] if p.get("ok")),
                        "error": None if out.get("ok") else str(out.get("error") or "the scout gave no answer")}
                late = _noted_after(bank, T, out.get("notes"), t)
                if late:
                    notes.append(f"{_plural(late, 'claim')} the scout noted after {t} (the run crossed midnight UTC) "
                                 f"{'is' if late == 1 else 'are'} not weighed here: they count from the next run")
                if read["error"]:
                    notes.append(f"read: {read['error']}")
                    mark_down(_model_down(out))
            except Exception as e:  # noqa: BLE001
                read = {"claims_noted": 0, "pages_read": 0, "error": f"{type(e).__name__}: {e}"}
                notes.append(f"read failed ({read['error']})")
                mark_down(_exc_down(e))

    # 5. Judge: the model's own call, stored as a judgment fact (a source, weighed by its record).
    judgment = None
    if do_judge:
        s = skip("judgment")
        if s:
            notes.append(s[0])
            judgment = {"ref": None, "error": s[1]}
        else:
            log(f"{T}: the model makes its call ({JUDGE_SECONDS // 60} min at most)")
            kw = {"client": client, "log": log, "deadline": time.monotonic() + JUDGE_SECONDS}
            if _accepts(judge_mod.assess, "t_run"):
                kw["t_run"] = t  # the judgment is known at this run's T_RUN, even across midnight
            try:
                res = judge_mod.assess(bank, T, **kw)
                ref = res.get("ref") if res.get("ok") else None
                if ref and isinstance(res.get("fact"), Fact):
                    bank.add(res["fact"])  # assess records it; adding again is a no-op
                judgment = {"ref": ref, "error": res.get("error")}
                if not ref:
                    notes.append(f"judgment not recorded: {res.get('error') or 'no valid call'}")
                    mark_down(_model_down(res))
            except Exception as e:  # noqa: BLE001
                judgment = {"ref": None, "error": f"{type(e).__name__}: {e}"}
                notes.append(f"judgment failed ({judgment['error']})")
                mark_down(_exc_down(e))
    else:
        notes.append("no judgment: the model was not asked")

    # 6. Gather the evidence and decide.
    g = gather(bank, T, t)
    if not g["data_ok"]:
        d = _blank(T, t, [g["data_why"]] + notes, g["price"])
        d["explanation"] = _template(judge_mod, d)
        return d
    action = action_for(g["score"])
    if not any(i["weight"] > 0 for i in g["items"]):
        notes.insert(0, NO_WEIGHT_NOTE)
    jref = (judgment or {}).get("ref")
    if jref:
        # Cited only when gather could weigh it: known and retrieved by T_RUN (section 3.2's rule for a call).
        jf = bank.get(jref)
        if jf is None:
            notes.append(f"judgment {jref} is not in the bank; not cited")
            judgment = {"ref": None, "error": "not in the bank"}
        elif jf.known_at > t or (jf.retrieved_at or jf.known_at) > t:
            notes.append(f"judgment {jref} was written after {t} (the run crossed midnight UTC): it is recorded and "
                         "scored as its source's call, but not weighed or cited in this decision")
            judgment = {"ref": None, "error": f"written after {t}"}
    d = {"ok": True, "ticker": T, "action": action, "direction": DIRECTION[action], "score": g["score"],
         "threshold": POLICY["threshold"], "signal_score": g["signal_score"], "source_term": g["source_term"],
         "items": g["items"], "lean": g["lean"], "context_refs": g["context_refs"], "context_notes": g["context_notes"],
         "price": g["price"], "judgment": judgment, "read": read, "explanation": None, "explained_by": None,
         "explain_error": None, "unverified_numbers": [], "ref": None, "policy": POLICY["id"],
         "policy_sha": POLICY_SHA, "horizon_days": POLICY["horizon_days"], "as_of": t[:10], "notes": notes}

    # 7. Explain: the model may put the decision in plain words; it cannot change it.
    text, by, err, unverified = None, "template", None, []
    if do_explain:
        s = skip("explanation")
        if s:
            notes.append(s[0].replace("explanation skipped", "explanation by template"))
        else:
            try:
                res = judge_mod.explain(plain(d), client=client, deadline=time.monotonic() + EXPLAIN_SECONDS)
                err = res.get("error")
                text = str(res["text"]) if res.get("text") else None  # the model's words, or explain's own template
                if text and str(res.get("by") or "").startswith("llm:"):
                    by, unverified = str(res["by"]), list(res.get("unverified_numbers") or [])
                mark_down(_model_down(res))
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                mark_down(_exc_down(e))
    d.update(explanation=text or _template(judge_mod, d), explained_by=by, explain_error=err,
             unverified_numbers=unverified)

    # 8. Record, then raise the change alert if one is due.
    prev = next((f for f in history(bank, T) if f.context.get("policy") == POLICY["id"]), None)
    alert_due = False
    if prev is not None and prev.value != action:
        from brain import watch
        try:
            alert_due = T in watch.load()["tickers"]
        except ValueError as e:
            notes.append(f"no change alert: {e}")
    fact = record(T, t, d)
    bank.add(fact)
    out = as_dict(fact)
    if alert_due:
        from brain import watch
        try:
            watch.alert(bank, T, "decision", "notice", f"{T} decision {prev.value} -> {action} (policy {POLICY['id']}, paper)",
                        [fact], [f"decision:{fact.ref}"], fact.known_at)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - the decision stands without its alert
            out["notes"] = out["notes"] + [f"change alert not written ({type(e).__name__}: {e})"]
    return out


_CONTEXT_KEYS = ("policy", "policy_sha", "horizon_days", "direction", "score", "threshold", "signal_score",
                 "source_term", "items", "lean", "context_refs", "context_notes", "price", "judgment", "read",
                 "explanation", "explained_by", "explain_error", "unverified_numbers", "notes")


def record(T: str, t: str, d: dict) -> Fact:
    """The decision as a derived fact, citing the price snapshot, every claim and judgment behind an item,
    the context refs and the judgment."""
    ctx = plain({"git": ledger.git_state(), **{k: d.get(k) for k in _CONTEXT_KEYS}})
    inputs = [d["price"]["ref"]] + [r for i in d["items"] for r in i["refs"]] + list(d["context_refs"])
    if (d.get("judgment") or {}).get("ref"):
        inputs.append(d["judgment"]["ref"])
    return Fact(entity=T, attr="decision", value=d["action"], as_of=t[:10], known_at=t, tier="derived",
                source={"name": "brain.decide", "policy": POLICY["id"]}, context=ctx,
                inputs=tuple(dict.fromkeys(inputs)), retrieved_at=now_iso())


# ---------- scoring decisions ----------
def _stream(bs: list[dict], dropped: int, gate: dict, priced: int = 0, unpriced: int = 0) -> dict:
    """One stream's bets and gate. ``priced`` and ``unpriced`` count its decisions whose window has closed,
    with and without an outcome: as for a source, too many unpriced ones (claims.too_unpriced) bar proven."""
    tl = tally(bs, gate["alpha_lower"])
    proven = bool(tl["bets"] >= gate["min_bets"] and tl["p"] is not None and tl["p"] < gate["p_lt"]
                  and tl["mean_signed_excess"] is not None and tl["mean_signed_excess"] > gate["mean_signed_excess_gt"]
                  and not too_unpriced(unpriced, priced + unpriced))
    return {"bets": tl["bets"], "hits": tl["hits"], "hit_rate": tl["hit_rate"], "p": tl["p"], "lower": tl["lower"],
            "mean_signed_excess": tl["mean_signed_excess"], "dropped": dropped, "unpriced": unpriced, "proven": proven}


def score(bank: Bank, t: str | None = None) -> dict:
    """{"policies": {"D1": {"sha": str, "commits": [str, ...],
         "actions": {"bets", "hits", "hit_rate", "p", "lower", "mean_signed_excess", "dropped", "unpriced", "proven"},
         "lean":    {... same keys ...},
         "no_edge": {"bets", "mean_excess"},            # what the stocks did after NO_EDGE (direction 0)
         "pending": int, "unpriced": int}},
        "rows": [{"ref", "ticker", "as_of", "action", "lean", "entry", "exit", "excess", "hit",
                  "stream_status": "scored" | "overlapping" | "pending" | "unpriced"}],
        "gate": POLICY["source_gate"]}
    Three streams per (policy, sha), each grouped into bets across tickers: BUY/SELL actions, non-zero
    leans, and NO_EDGE (direction 1, reported as the mean excess, never as a hit rate). A decision whose sha
    differs from the code's for its id is reported under "<id>?<sha>". Rows are newest first and also carry
    the policy key. A decision with no outcome is pending until SPY's book is a few sessions past its exit,
    then unpriced (claims.scored_at): its stock's closes are missing. It is scored once they arrive, and a
    stream with too many unpriced decisions is not proven, as for a source (each stream's "unpriced")."""
    t = t or run_time()
    gate = POLICY["source_gate"]
    decisions: list[Fact] = []
    for e in tickers_with_decisions(bank):
        b = Bank(bank.root, use_index=bank.use_index)
        decisions += [f for f in b.view(t).facts(e, "decision") if (f.retrieved_at or f.known_at) <= t]
    book = PriceBook(bank, sorted({f.entity for f in decisions} | {POLICY["benchmark"]}), asof=t)

    def empty(sha: str) -> dict:
        return {"sha": sha, "commits": [], "actions": [], "lean": [], "no_edge": [], "pending": 0, "unpriced": 0,
                "unpriced_actions": 0, "unpriced_lean": 0}
    groups: dict[str, dict] = {POLICY["id"]: empty(POLICY_SHA)}
    rows = []
    # In the order they were made: one day's decisions share known_at (T_RUN), so retrieved_at orders them.
    for f in sorted(decisions, key=lambda f: (f.known_at, f.retrieved_at or "", f.entity, f.id)):
        c = f.context
        pid, sha = str(c.get("policy") or f.source.get("policy") or "?"), str(c.get("policy_sha") or "?")
        key = pid if KNOWN_POLICIES.get(pid) == sha else f"{pid}?{sha}"
        g = groups.setdefault(key, empty(sha))
        commit = (c.get("git") or {}).get("commit")
        if commit and commit not in g["commits"]:
            g["commits"].append(commit)
        horizon = int(c.get("horizon_days") or POLICY["horizon_days"])
        status, o = (scored_at(book, f.entity, anchor(f), horizon, t, POLICY["benchmark"]) if scorable(f.entity)[0]
                     else ("pending", None))
        lean_d = int((c.get("lean") or {}).get("direction") or 0)
        act = DIRECTION.get(f.value, 0)
        row = {"ref": f.ref, "policy": key, "ticker": f.entity, "as_of": f.as_of, "action": f.value, "lean": lean_d,
               "entry": None, "exit": None, "excess": None, "hit": None, "stream_status": status}
        if o is None:
            g[status] += 1
            if status == "unpriced":
                g["unpriced_actions"] += bool(act)
                g["unpriced_lean"] += bool(lean_d)
        else:
            call = {"ref": f.ref, "ticker": f.entity, "entry": o["entry"], "exit": o["exit"], "excess": o["excess"]}
            row.update(entry=o["entry"], exit=o["exit"], excess=o["excess"])
            if act:
                g["actions"].append({**call, "direction": act})
                v = act * o["excess"]
                row["hit"] = True if v > 0 else False if v < 0 else None
            elif f.value == "NO_EDGE":
                g["no_edge"].append({**call, "direction": 1})
            if lean_d:
                g["lean"].append({**call, "direction": lean_d})
        rows.append(row)
    policies, kept = {}, set()
    for key, g in groups.items():
        a_bets, a_drop = bets(g["actions"])
        l_bets, l_drop = bets(g["lean"])
        n_bets, _ = bets(g["no_edge"])
        kept.update(r for b in a_bets + n_bets for r in b["refs"])
        policies[key] = {"sha": g["sha"], "commits": g["commits"],
                         "actions": _stream(a_bets, a_drop, gate, len(g["actions"]), g["unpriced_actions"]),
                         "lean": _stream(l_bets, l_drop, gate, len(g["lean"]), g["unpriced_lean"]),
                         "no_edge": {"bets": len(n_bets),
                                     "mean_excess": float(np.mean([b["value"] for b in n_bets])) if n_bets else None},
                         "pending": g["pending"], "unpriced": g["unpriced"]}
    for r in rows:
        if r["entry"] is not None:
            r["stream_status"] = "scored" if r["ref"] in kept else "overlapping"
    rows.reverse()
    return plain({"policies": policies, "rows": rows, "gate": gate})


# ---------- the printed card ----------
def _signed(x: float) -> str:
    return "0.00" if abs(x) < 0.005 else f"{x:+.2f}"


def _pct(x) -> str:
    return "–" if x is None else f"{x * 100:+.2f}%"


def card(d: dict) -> str:
    """A decision (section 3.4's dict, or ``as_dict`` of a recorded one) as the CLI prints it."""
    T, action = d["ticker"], d["action"]
    thr = d.get("threshold", POLICY["threshold"])
    L = [f"{T}  {LABEL.get(action, action)}   (policy {d.get('policy', POLICY['id'])} · paper only · not financial advice)"]
    notes = list(d.get("notes") or [])
    if action == "NO_DATA":
        L.append(f"Nothing decided: {notes[0] if notes else 'no data'}.")
        notes = notes[1:]
    else:
        weighted = [i for i in d.get("items") or [] if i["weight"] > 0]
        L.append(f"Score {_signed(d['score'])}; BUY needs +{thr:.2f}, SELL -{thr:.2f}."
                 + ("" if weighted else " No evidence carries weight yet."))
        L += ["", f"{'Evidence':<38}{'dir':<6}{'weight':<8}why"]
        for i in d.get("items") or []:
            w = "0" if not i["weight"] else f"{i['weight']:.2f}"
            L.append(f"{i['kind']:<8}{i['id']:<29} {WORD.get(i['direction'], '?'):<5} {w:<7} {i['why']}")
        ctx = d.get("context_notes") or [{"ref": r, "why": ""} for r in d.get("context_refs") or []]
        if ctx:
            L.append(f"Context, weight 0 ({len(ctx)}):")
            L += [f"  [{c['ref'].partition('/')[2][:8]}] {c['why']}".rstrip() for c in ctx[:8]]
            if len(ctx) > 8:
                L.append(f"  ... {len(ctx) - 8} more")
        ln = d.get("lean") or lean([])
        if not ln["sources"]:
            lw = "none (no source has a call in the window)"
        elif ln["up"] and ln["down"]:
            lw = f"{WORD[ln['direction']]}, {ln['up']} up and {ln['down']} down of {ln['sources']} sources"
        elif ln["direction"]:
            lw = f"{WORD[ln['direction']]}, {max(ln['up'], ln['down'])} of {ln['sources']} sources"
        else:
            lw = f"none, no source of {ln['sources']} points a way"
        L.append(f"Lean (unproven, not evidence): {lw}")
    by = d.get("explained_by") or "template"
    L.append(f"Why: {d.get('explanation') or ''}")
    if by != "template":
        L.append(f"     (written by {by}; the decision is the table above)")
    elif d.get("explain_error"):
        L.append(f"     (written by the template: the model's explanation was not used, {d['explain_error']})")
    if d.get("unverified_numbers"):
        L.append(f"UNVERIFIED numbers in the explanation: {', '.join(d['unverified_numbers'])}")
    rd = d.get("read")
    if isinstance(rd, dict) and not rd.get("error"):
        pages, noted = int(rd.get("pages_read") or 0), int(rd.get("claims_noted") or 0)
        L.append(f"Read: {pages} page{'s' * (pages != 1)} read, {noted} claim{'s' * (noted != 1)} noted "
                 f"(a claim weighs 0 until its source has a record).")
    if notes:
        L.append("Notes:")
        L += [f"- {n}" for n in notes]
    if d.get("ref"):
        L.append(f"Recorded {T}/{d['ref'].partition('/')[2][:8]}. Scored after {d.get('horizon_days', POLICY['horizon_days'])} "
                 f"trading days: `brain decide score`.")
    elif action == "NO_DATA":
        L.append("Not recorded: NO_DATA has nothing to score against.")
    else:
        L.append("Not recorded.")
    return "\n".join(L)


def _stream_line(s: dict) -> str:
    u = int(s.get("unpriced") or 0)
    unpriced = f"; {u} with no price past {'its' if u == 1 else 'their'} exit (ingest their tickers)" if u else ""
    if not s["bets"]:
        return "0 bets" + (f" ({s['dropped']} overlapping calls dropped)" if s["dropped"] else "") + unpriced
    return (f"{s['bets']} bets, {s['hits']} hits ({s['hit_rate']:.0%}), p {s['p']:.2f}, lower bound {s['lower']:.2f}, "
            f"mean signed excess {_pct(s['mean_signed_excess'])}, {s['dropped']} dropped: "
            + ("proven" if s["proven"] else "not proven") + unpriced)


def scorecard(sc: dict, tr: dict) -> str:
    gate = sc["gate"]
    L = [f"Decisions scored against {POLICY['benchmark']} over {POLICY['horizon_days']} trading days (paper only). "
         f"Proven needs {gate['min_bets']} bets, p < {gate['p_lt']} and a positive mean signed excess, with at most "
         f"{UNPRICED_MAX_SHARE:.0%} of the decisions past their exit lacking a price.", ""]
    for key, p in sc["policies"].items():
        n = sum(1 for r in sc["rows"] if r.get("policy") == key)
        flag = "" if "?" not in key else "  (this sha is not the code's: the policy was edited after these were made)"
        u = int(p.get("unpriced") or 0)
        L.append(f"{key} (sha {p['sha']}; commits {', '.join(c[:8] for c in p['commits']) or '-'}): "
                 f"{n} decisions, {p['pending']} pending" + (f", {u} with no price past their exit" if u else "") + flag)
        L.append(f"  BUY/SELL actions     {_stream_line(p['actions'])}")
        L.append(f"  lean (not evidence)  {_stream_line(p['lean'])}")
        ne = p["no_edge"]
        L.append(f"  NO_EDGE              {ne['bets']} bets" + (f", mean excess {_pct(ne['mean_excess'])} (what the "
                                                                 "stocks did; not a hit rate)" if ne["bets"] else ""))
    L += ["", f"Source track records (the record {POLICY['id']} weighs: {gate['min_bets']} bets, p < {gate['p_lt']}, "
              f"BH q < {gate['bh_q']}, weight max(0, 2L-1) from the {1 - gate['alpha_lower']:.0%} lower bound L):"]
    if not tr:
        L.append("  no directional calls yet")
    for k, r in sorted(tr.items(), key=lambda kv: (-kv[1]["weight"], -kv[1]["bets"], kv[0])):
        L.append(f"  {k:<44} weight {r['weight']:.2f}  {r['why']}")
    recent = sc["rows"][:12]
    if recent:
        L += ["", "Latest decisions:"]
        for r in recent:
            out = (r["stream_status"] if r["entry"] is None else
                   f"excess {_pct(r['excess'])}" + ("" if r["hit"] is None else f", {'hit' if r['hit'] else 'miss'}")
                   + ("" if r["stream_status"] == "scored" else f" ({r['stream_status']})"))
            L.append(f"  {r['as_of']}  {r['ticker']:<8} {r['action']:<8} lean {WORD.get(r['lean'], '?'):<5} {out}")
    return "\n".join(L)


# ---------- CLI ----------
_TICKER = re.compile(r"[A-Z0-9^][A-Z0-9.^=\-]{0,19}")
WORDS = ("score", "show", "policy")


def _usage(msg: str) -> int:
    print(f"brain decide: {msg}", file=sys.stderr)
    return 1


def _cmd(a) -> int:
    words = list(a.words or [])
    if words and words[0] in WORDS:
        return _cmd_word(a, words[0], words[1:])
    if a.all:
        return _usage("--all goes with `brain decide show TICKER`")
    tickers = [w.strip().upper() for w in words]
    bad = [w for w in tickers if not _TICKER.fullmatch(w)]
    if bad:
        return _usage(f"not a ticker: {bad[0]!r}")
    if a.watchlist:
        from brain import watch
        try:
            tickers += watch.load()["tickers"]
        except ValueError as e:
            return _usage(str(e))
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return _usage("name a ticker (brain decide AAPL), or --watchlist")
    if a.max_pages < 1:
        return _usage("--max-pages is at least 1")
    log = (lambda m: print(m, file=sys.stderr, flush=True)) if a.json else (lambda m: print(m, flush=True))
    bank, state, out = Bank(), {}, []
    use_llm = not a.no_llm
    for t in tickers:
        d = decide(bank, t, do_ingest=a.ingest, do_read=a.read and use_llm, do_judge=use_llm, do_explain=use_llm,
                   log=log, max_pages=a.max_pages, llm_state=state)
        out.append(d)
        if not a.json:
            print(card(d) + "\n", flush=True)
    if a.json:
        print(json.dumps(plain(out), indent=1))
    return 0 if all(d["ok"] for d in out) else 1


def _cmd_word(a, word: str, rest: list[str]) -> int:
    if a.ingest or a.read or a.no_llm or a.watchlist:
        return _usage(f"`{word}` takes no --ingest, --read, --no-llm or --watchlist")
    if word == "policy":
        if rest:
            return _usage("policy takes no arguments")
        if a.json:
            print(json.dumps({"policy": POLICY, "sha": POLICY_SHA}, indent=1))
        else:
            print(json.dumps(POLICY, indent=1, sort_keys=True))
            print(f"POLICY_SHA {POLICY_SHA}")
        return 0
    bank = Bank()
    if word == "score":
        if rest or a.all:
            return _usage("score takes no arguments")
        sc, tr = score(bank), plain(track_record(bank))
        print(json.dumps({"score": sc, "track_record": tr}, indent=1) if a.json else scorecard(sc, tr))
        return 0
    if len(rest) != 1 or not _TICKER.fullmatch(rest[0].strip().upper()):
        return _usage("show takes one ticker: brain decide show AAPL [--all]")
    T = rest[0].strip().upper()
    fs = history(bank, T) if a.all else [f for f in [latest(bank, T)] if f is not None]
    if not fs:
        print(f"no decision recorded for {T}: brain decide {T}", file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps([as_dict(f) for f in fs], indent=1))
        return 0
    for f in fs:
        print(f"Decided {f.as_of} (known {f.known_at}, written {f.retrieved_at})")
        print(card(as_dict(f)) + "\n")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("decide", help="paper decision on stocks under policy D1 (BUY / SELL / NO_EDGE), scored "
                                      "later; also: decide score | show TICKER [--all] | policy",
                       description="Weigh the bank's tested evidence on each stock under policy D1 (docs/DECIDE.md) "
                                   "and record a paper decision. Nothing here places orders.")
    p.add_argument("words", nargs="*", metavar="TICKER | score | show TICKER | policy",
                   help="tickers to decide (upper case), or one of the lowercase words score, show, policy")
    p.add_argument("--watchlist", action="store_true", help="also decide every ticker on the watchlist")
    p.add_argument("--ingest", action="store_true", help="refresh SEC filings, prices and headlines (and SPY) first")
    p.add_argument("--read", action="store_true", help="the model reads the web first (scout)")
    p.add_argument("--no-llm", action="store_true", help="no read, no judgment, template explanation (deterministic only)")
    p.add_argument("--max-pages", type=int, default=6, help="pages the scout may read (default 6)")
    p.add_argument("--json", action="store_true", help="print the decisions (or score, show, policy) as JSON")
    p.add_argument("--all", action="store_true", help="show: every recorded decision, newest first")
    p.set_defaults(fn=_cmd)
