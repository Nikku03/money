"""The local model's own forecast: the market primer, a point-in-time snapshot and one probability
(docs/FORECAST.md, section 9).

F1-llm reads two things: ``docs/MARKET_PRIMER.md`` (general knowledge about markets and people, in its
system message) and a snapshot of one stock after the close of a session d (``snapshot``). It answers with the
chance that the stock beats SPY over the next h sessions. It is not shown F1's forecast: shown it, the small
models tried here copied F1's number every time, so their answer said nothing of their own. What shapes that answer and
does not come from the bank is pinned: the prompt by ``prompt_sha``, the primer by ``primer_sha``, the
snapshot's layout by ``SNAPSHOT_VERSION`` and the weights by ``llm_identity``'s digest. A declared test names
exactly what was asked, and a change to any of them is a new test.

Point in time. The snapshot reads the bank through ``view(T(d))`` and the price book only up to d.
- Price figures are ratios (returns, distances from the 52-week high and low, volatility), never price
  levels: the book's closes are on today's split basis, and a level would tell of splits after d.
- Fundamentals come from the month-end view F1 uses (the last session before d's month); the latest earnings
  release and insider trades from the view at T(d).
- The ticker only, no company name and no headlines: the bank's identity facts and its headlines are newer
  than the tested windows, so with them the tested input and the served one would differ.

The model is small and often wrong, so its answer is parsed leniently (the judge's order, then one repair)
and clipped to [0.01, 0.99]. ``llm_forecast`` never raises: a failed answer is reported as such, and the
caller uses F1's probability in its place.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from brain import events, judge, llm, metrics
from brain.bank import canon
from brain.forecast import features
from brain.sources import insiders
from brain.sources.sec import REVENUE
from brain.terminal import analyst
from brain.timeutil import days_between

SNAPSHOT_VERSION = 2          # 2: F1's forecast is no longer shown
HORIZON = 5                  # sessions, when the forecast does not say
TEMPERATURE = 0
SEED = 7
MAX_TOKENS = 300
P_CLIP = (0.01, 0.99)
MAX_RAW = 4000               # characters of each raw reply kept
MIN_SECONDS = 5.0            # a request is not sent with less time than this before the deadline
NO_THINK = "/no_think"       # Qwen3's documented switch that turns its thinking off for one turn
INSIDER_DAYS = 90
EPS_QUARTERS = 4
WINDOWS = (("1 week", 5), ("1 month", 21), ("3 months", 63), ("12 months", 252))
VOL_SESSIONS = 63
YEAR = 252
BENCH = "SPY"
PRIMER_PATH = judge.PRIMER_PATH
GROSS_PROFIT, ASSETS, LIABILITIES = features.GROSS_PROFIT, features.ASSETS, features.LIABILITIES
OPERATING_CASH_FLOW = features.OPERATING_CASH_FLOW

SYSTEM = """{primer}

---

Your task. The user gives you a snapshot of one stock after the close of one trading day: only facts public by then, from a memory bank of public data. Give the probability that this stock beats SPY over the next {horizon} sessions after the snapshot (its total return minus SPY's, counted from the close of the first session after the snapshot date).
Reply with only a JSON object, no other text: {{"p_beat": 0.0-1.0, "reason": "one or two sentences"}}
Never use knowledge of events after the snapshot date."""
USER = ("{snapshot}\n\nGive the probability that {ticker} beats SPY over the next {horizon} sessions after {date}. "
        "Reply with only the JSON object.")
REPAIR = 'Reply with only a JSON object, no other text: {"p_beat": a number from 0.0 to 1.0, "reason": "one or two sentences"}'
QWEN3_RULE = f"For a model in the Qwen3 family, every user message ends with {NO_THINK}, its documented switch."

# Everything the model is shown that does not come from the bank or the primer, and the request's settings: a
# change to any of it is a new prompt (prompt_sha), so a declared test that pinned the old one refuses to run.
PROMPT = {"system": SYSTEM, "user": USER, "repair": REPAIR, "qwen3": QWEN3_RULE, "no_think": NO_THINK,
          "temperature": TEMPERATURE, "seed": SEED, "max_tokens": MAX_TOKENS, "p_clip": list(P_CLIP), "repairs": 1,
          "horizon_default": HORIZON}


def prompt_sha() -> str:
    """sha256[:16] of PROMPT in canonical JSON (the hash brain.research.forecast_tests pins)."""
    return hashlib.sha256(canon(PROMPT).encode()).hexdigest()[:16]


def primer_sha() -> str:
    """sha256[:16] of docs/MARKET_PRIMER.md's bytes; "missing" when the file is not there."""
    return hashlib.sha256(PRIMER_PATH.read_bytes()).hexdigest()[:16] if PRIMER_PATH.exists() else "missing"


def primer() -> str:
    """The primer as the model is shown it (the text the judge carries too)."""
    return judge.PRIMER


# ---------- the snapshot ----------
def _pct(x, d: int = 1) -> str:
    return f"{x * 100:+.{d}f}%" if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def _money(x) -> str:
    if not (isinstance(x, (int, float)) and math.isfinite(x)):
        return "n/a"
    a = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{x / div:,.2f}{suf}"
    return f"{x:,.2f}"


def _num(x, d: int = 2, sign: bool = False) -> str:
    if not (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _n(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")


def _upto(book, view, t: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dates, adjusted closes, split-adjusted closes) of ``t`` dated on or before the view's date."""
    d, a = book.adj_upto(view, t)
    return d, np.asarray(a, dtype=float), np.asarray(book.close[t.upper()][: len(a)], dtype=float)


def _ret(a: np.ndarray, k: int) -> float:
    if len(a) <= k or not (a[-1 - k] > 0 and np.isfinite(a[-1])):
        return math.nan
    return float(a[-1] / a[-1 - k] - 1)


def _price_lines(book, view, t: str, h: int) -> list[str]:
    if not book.has(t):
        return ["Prices: none in the bank up to this date."]
    _, a, c = _upto(book, view, t)
    if len(a) < 2:
        return ["Prices: fewer than two sessions up to this date."]
    spy = _upto(book, view, BENCH)[1] if book.has(BENCH) and t.upper() != BENCH else None
    rets = []
    for name, k in WINDOWS:
        s = f"{name} {_pct(_ret(a, k))}"
        if spy is not None:
            s += f" (SPY {_pct(_ret(spy, k))})"
        rets.append(s)
    out = [f"- Total return: {', '.join(rets)}."]
    if len(c) > YEAR:
        w = c[-YEAR:]
        out.append(f"- Close against its 52-week high {_pct(float(c[-1] / np.nanmax(w) - 1))}, "
                   f"against its 52-week low {_pct(float(c[-1] / np.nanmin(w) - 1))}.")
    else:
        out.append(f"- 52-week high and low: n/a ({len(c)} sessions of history).")
    w = a[-(VOL_SESSIONS + 1):]
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(w[1:] / w[:-1])  # ratios first, as features.py: a split's factor cancels exactly
    lr = lr[np.isfinite(lr)]
    if len(lr) >= int(0.8 * VOL_SESSIONS):
        sd = float(lr.std(ddof=1))
        out.append(f"- Volatility: {sd * 100:.2f}% a day over the last {VOL_SESSIONS} sessions "
                   f"({sd * math.sqrt(YEAR) * 100:.0f}% a year); a typical {h}-session move is about "
                   f"{sd * math.sqrt(h) * 100:.1f}%.")
    return out


def month_end(book, t: str, d: str) -> str | None:
    """The month-end session whose view F1's fundamentals use on d: the last SPY session (the stock's own,
    when the book has no SPY) before the first day of d's month."""
    cal = book.dates.get(BENCH) if book.has(BENCH) else book.dates.get(t.upper())
    if cal is None or not len(cal):
        return None
    i = int(np.searchsorted(cal, str(d)[:7] + "-01", side="left")) - 1
    return str(cal[i]) if i >= 0 else None


def _latest_year(view, t: str, attrs, flow: bool = True):
    ys = metrics.annual(view, t, attrs, flow=flow)
    if not ys or days_between(ys[-1].as_of, view.t) > metrics.STALE_DAYS:
        return None
    return ys[-1]


def _fundamental_lines(bank, book, t: str, d: str) -> list[str]:
    me = month_end(book, t, d)
    if me is None:
        return ["Fundamentals: no month-end before this date."]
    view = bank.view(features.T(me))
    f = features.fundamentals_at(view, t, book)
    out = [f"Fundamentals (filings known at the month-end view, {me}):"]
    mc = math.exp(f["log_mktcap"]) if math.isfinite(f["log_mktcap"]) else math.nan
    out.append(f"- Market value {_money(mc)}; earnings yield, trailing four quarters, {_pct(f['earnings_yield_ttm'])}.")
    rev = _latest_year(view, t, REVENUE)
    ni = _latest_year(view, t, metrics.NET_INCOME)
    anchor = rev or ni
    if anchor is None:
        out.append("- Latest fiscal year: no current annual figures.")
    else:
        parts = [f"revenue {_money(rev.value if rev else None)} ({_pct(f['sales_growth'])} on the year before)",
                 f"net income {_money(ni.value if ni else None)}"]
        ocf = _latest_year(view, t, OPERATING_CASH_FLOW)
        assets = _latest_year(view, t, ASSETS, flow=False)
        parts += [f"operating cash flow {_money(ocf.value if ocf else None)}",
                  f"total assets {_money(assets.value if assets else None)} ({_pct(f['asset_growth'])} on the year before)",
                  f"liabilities over assets {_num(f['leverage'])}",
                  f"gross profit over assets {_num(f['gross_profitability'])}",
                  f"accruals (net income minus operating cash flow, over assets) {_num(f['accruals'], 3, sign=True)}"]
        out.append(f"- Latest fiscal year (ended {anchor.as_of}, filed {anchor.known_at[:10]}): " + "; ".join(parts) + ".")
    rows = events.earnings_table(view, t, book)[:EPS_QUARTERS][::-1]
    if rows:
        qs = [f"{r['period_end']}: {_num(r['eps'])} ({_pct(r['eps_yoy'], 0)} on a year earlier, SUE {_num(r['sue'], 1, sign=True)})"
              for r in rows]
        out.append("- EPS by quarter (diluted, quarter end): " + "; ".join(qs) + ".")
    else:
        out.append("- EPS by quarter: none on file.")
    return out


def _event_lines(bank, book, t: str, d: str) -> list[str]:
    view = bank.view(features.T(d))
    out = []
    evs = events.distinct(events.earnings_events(view, t))
    if not evs:
        out.append("Last earnings release: none on file.")
    else:
        e = evs[-1]
        r = events.reaction(book, view, t, e)
        ago = days_between(e.known_at, d)
        line = f"Last earnings release: an 8-K (item 2.02) filed {e.known_at[:10]}, {_n(ago, 'day')} before; "
        if r is not None and r.get("car_3d") is not None:
            line += f"3-day reaction against SPY {_pct(r['car_3d'])}."
        else:
            line += "its 3-day reaction window had not closed."
        out.append(line)
    tot = insiders.buy_sell_totals(view, t, INSIDER_DAYS)
    def count(k: int, word: str, usd: float) -> str:
        return _n(k, word) + (f" ({_money(usd)} USD)" if k else "")
    out.append(f"Insider trades (Forms 4 and 5) filed in the {INSIDER_DAYS} days to {d}: "
               f"{count(tot['buys'], 'open-market buy', tot['buy_usd'])}, {count(tot['sells'], 'sale', tot['sell_usd'])}.")
    return out


def _horizon(forecast) -> int:
    try:
        h = int((forecast or {}).get("horizon") or HORIZON) if isinstance(forecast, dict) else HORIZON
    except (TypeError, ValueError):
        return HORIZON
    return h if h > 0 else HORIZON


def snapshot(bank, book, ticker: str, d: str, forecast) -> str:
    """The stock as of the close of session ``d``, in plain lines: the ticker (no name), price ratios from the
    book up to d, fundamentals from the month-end view F1 uses, and the latest earnings release and the insider
    trades known at T(d). Neither headlines nor F1's forecast; ``forecast`` gives only the horizon.
    SNAPSHOT_VERSION names this layout."""
    t, d = str(ticker).strip().upper(), str(d)[:10]
    view = bank.view(features.T(d))
    lines = [f"Snapshot of {t} after the close of {d}. Only facts public by then; no company name and no headlines.",
             "", "Prices (ratios only; no price levels):"]
    lines += _price_lines(book, view, t, _horizon(forecast))
    lines += [""] + _fundamental_lines(bank, book, t, d)
    lines += [""] + _event_lines(bank, book, t, d)
    return "\n".join(lines)


# ---------- the request ----------
def is_qwen3(model) -> bool:
    """A Qwen3-family model name: "qwen3:8b", "/models/Qwen3-1.7B-Q8_0.gguf", "Qwen/Qwen3-8B". Not Qwen2.5."""
    return re.match(r"qwen-?3(?![0-9.])", judge.canonical_model(str(model or ""))) is not None


def served_model(client) -> str:
    """The id the server lists for the client's model (``_served``; llama.cpp lists the one GGUF it serves,
    whatever model name a request carries), else the client's own model name when the list cannot be read."""
    want = str(getattr(client, "model", "") or "")
    try:
        ids = [str(i) for i in client.models()]
    except Exception:  # noqa: BLE001 - no list to read: the name asked for is all there is to go on
        return want
    return _served(client, ids) or want


def uses_no_think(client) -> bool:
    """Whether requests to ``client`` end each user turn with /no_think: the model the server serves (the
    one F1d's identity pins) is a Qwen3, whatever BRAIN_LLM_MODEL says."""
    return is_qwen3(served_model(client))


def _turn(text: str, no_think: bool) -> str:
    return f"{text.rstrip()} {NO_THINK}" if no_think else text


def messages(bank, book, ticker: str, d: str, forecast, model="", no_think: bool | None = None) -> list[dict]:
    """The system message (primer and task) and the user message (snapshot and question), as sent. The user
    message ends with /no_think when ``no_think`` says so (default: when ``model`` is a Qwen3)."""
    t, d = str(ticker).strip().upper(), str(d)[:10]
    h = _horizon(forecast)
    nt = is_qwen3(model) if no_think is None else bool(no_think)
    user = USER.format(snapshot=snapshot(bank, book, t, d, forecast), ticker=t, horizon=h, date=d)
    return [{"role": "system", "content": SYSTEM.format(primer=primer(), horizon=h)},
            {"role": "user", "content": _turn(user, nt)}]


def _left(deadline) -> float | None:
    if deadline is None:
        return None
    return deadline - (time.time() if deadline >= 1e9 else time.monotonic())


def _complete(client, msgs: list[dict], deadline) -> tuple[str, str | None, str | None]:
    """(text, finish_reason, server's model) of one request: no tools, temperature 0, seed 7, 300 tokens."""
    kw = {}
    left = _left(deadline)
    if left is not None:
        kw["timeout"] = max(MIN_SECONDS, min(float(getattr(client, "timeout", left) or left), left))
    if isinstance(client, llm.Client):
        kw["seed"] = SEED  # a stand-in client without llm.Client's seed is asked without it
    resp = client.complete(msgs, tools=None, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, **kw)
    choice = resp["choices"][0]
    return analyst._content(choice.get("message") or {}), choice.get("finish_reason"), resp.get("model")


# ---------- parsing the answer ----------
def probability(x) -> float | None:
    """A number in [0, 1]; one in (1, 100] is a percent; "0.55" and "55%" are parsed; anything else None."""
    v = None
    if isinstance(x, str):
        m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*(%?)\s*", x)
        if m:
            v = float(m.group(1)) / (100 if m.group(2) else 1)
    elif isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
        v = float(x)
    if v is not None and 1 < v <= 100:
        v /= 100
    return v if v is not None and 0 <= v <= 1 else None


def parse(text: str) -> tuple[dict | None, str]:
    """({"p_beat", "reason"}, "") or (None, why not), in judge.parse's order: the whole text without its
    thinking and fences, its outermost {...} span, then the first object anywhere in it with a "p_beat" key.
    p_beat is clipped to [0.01, 0.99]; the reason is one line of at most 300 characters."""
    s = judge.clean(text)
    if "{" not in s:
        return None, "the answer holds no JSON object"
    found = None
    for cand in (s, s[s.find("{"): s.rfind("}") + 1]):
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict) and "p_beat" in obj:
            found = obj
            break
    if found is None:
        dec, i = json.JSONDecoder(), 0
        while (i := s.find("{", i)) != -1:
            try:
                obj, _ = dec.raw_decode(s, i)
            except ValueError:
                obj = None
            if isinstance(obj, dict) and "p_beat" in obj:
                found = obj
                break
            i += 1  # nested objects too: {"answer": {"p_beat": ...}}
    if found is None:
        return None, "no JSON object with a p_beat in the answer"
    p = probability(found.get("p_beat"))
    if p is None:
        return None, f"p_beat {str(found.get('p_beat'))[:40]!r} is not a probability"
    reason = found.get("reason")
    return {"p_beat": round(min(P_CLIP[1], max(P_CLIP[0], p)), 4),
            "reason": judge._one_line(reason if isinstance(reason, str) else "")}, ""


# ---------- the forecast ----------
def llm_forecast(bank, book, ticker: str, d: str, forecast, client=None, deadline=None,
                 no_think: bool | None = None) -> dict:
    """The local model's chance that ``ticker`` beats SPY over the forecast's horizon after session ``d``.

    {"ok", "p_beat", "reason", "raw", "model", "seconds", "error", "down"}. One request without tools
    (temperature 0, seed 7, 300 tokens); when its answer is not the JSON object, one repair. ``p_beat`` is
    None unless ``ok``: the caller then uses F1's probability and counts the failure. ``raw`` is the reply as
    the server sent it (both replies after a repair); ``down`` is the server's error when it could not be
    asked at all. ``deadline`` is a time.time() or time.monotonic() value (analyst.expired). ``no_think``:
    whether the user turns end with /no_think; by default, when the served model is a Qwen3
    (``uses_no_think``). F1d passes the decision its declaration pinned. Never raises."""
    t0 = time.monotonic()
    out = {"ok": False, "p_beat": None, "reason": None, "raw": "", "model": None, "seconds": 0.0, "error": None,
           "down": None}
    try:
        _ask(bank, book, str(ticker or "").strip().upper(), str(d)[:10], forecast, client or llm.Client.from_env(),
             deadline, out, no_think)
    except Exception as e:  # noqa: BLE001 - a failed answer is F1's answer, counted; the run goes on
        out.update(ok=False, p_beat=None, reason=None, error=judge._one_line(f"{type(e).__name__}: {e}", 400))
    out["seconds"] = round(time.monotonic() - t0, 2)
    return out


def _ask(bank, book, ticker: str, d: str, forecast, client, deadline, out: dict, no_think: bool | None = None) -> None:
    model = getattr(client, "model", "")
    out["model"] = model or None
    nt = uses_no_think(client) if no_think is None else bool(no_think)
    msgs = messages(bank, book, ticker, d, forecast, model, no_think=nt)
    raws: list[str] = []
    why = ""
    while len(raws) < 2:  # the request, and one repair when its answer is not the JSON object
        if analyst.expired(deadline):
            out["error"] = (f"no valid answer before the time limit ({why})" if raws
                            else "no time left to ask the model")
            return
        try:
            text, finish, served = _complete(client, msgs, deadline)
        except llm.LLMError as e:
            error, hint = llm.describe_error(e, client)
            out["error"] = out["down"] = judge._one_line(f"{error}: {hint}", 400)
            return
        out["model"] = served or out["model"]
        raws.append(text or "")
        out["raw"] = "\n--- repair ---\n".join(r[:MAX_RAW] for r in raws)
        call, why = parse(text)
        if call is not None:
            out.update(ok=True, p_beat=call["p_beat"], reason=call["reason"], error=None)
            return
        if finish == "length":
            why = f"the answer was cut off at the length limit ({why})"
        msgs = msgs + [{"role": "assistant", "content": text or ""}, {"role": "user", "content": _turn(REPAIR, nt)}]
    out["error"] = f"the answer was not the JSON object asked for, even after one repair: {why}"


# ---------- the model's identity ----------
_DIGESTS: dict[tuple, str] = {}      # (path, size, mtime_ns, inode) -> sha256 hex: a GGUF is hashed once a process
_OLLAMA_BLOB = re.compile(r"sha256[-:]([0-9a-f]{64})")


def file_sha256(path) -> str:
    """sha256 hex of a file, read in 8 MB blocks; remembered per (path, size, mtime, inode)."""
    p = Path(path).resolve()
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns, st.st_ino)
    if key not in _DIGESTS:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            while block := fh.read(8 << 20):
                h.update(block)
        _DIGESTS[key] = h.hexdigest()
    return _DIGESTS[key]


def _served(client, ids: list[str]) -> str | None:
    """The model id the server lists for the client's model, or its only model."""
    want = str(getattr(client, "model", "") or "")
    for i in ids:
        if llm._same_model(want, i) or judge.canonical_model(want) == judge.canonical_model(i):
            return i
    return ids[0] if len(ids) == 1 else None


def _ollama_digest(client, name: str) -> str | None:
    """The weights blob's digest from Ollama's /api/show (its modelfile's FROM line names the blob), else
    a "digest" field the server gives. None when the server is not Ollama or shows neither."""
    base = re.sub(r"/v1/?$", "", str(getattr(client, "url", "") or "").rstrip("/"))
    show = llm.Client(url=base, model=name, key=getattr(client, "key", None), timeout=30.0)
    try:
        out = show._request("/api/show", {"model": name, "name": name}, timeout=30.0)
    except llm.LLMError:
        return None
    m = None
    for line in str(out.get("modelfile") or "").splitlines():
        if line.strip().upper().startswith("FROM "):
            m = _OLLAMA_BLOB.search(line)
            if m:
                break
    if m is None:
        m = _OLLAMA_BLOB.search(str(out.get("digest") or "")) or re.fullmatch(r"([0-9a-f]{64})", str(out.get("digest") or ""))
    return f"sha256:{m.group(1)}" if m else None


def llm_identity(client=None) -> dict:
    """{"model", "digest", "via", "why"} of the model the server serves for ``client``.

    ``model`` is its canonical name (judge.canonical_model of the id the server lists). ``digest`` is
    "sha256:<hex>" of the weights: the sha256 of the GGUF file when the server's model id is a path to a file
    on this machine (llama.cpp's server lists the --model path), else the weights blob that Ollama's /api/show
    names. When the server shows neither, ``digest`` is None and ``why`` says so: a declared test then refuses
    (brain.research.forecast_tests). Raises llm.LLMError when the server cannot be reached."""
    client = client or llm.Client.from_env()
    ids = client.models()
    served = _served(client, ids)
    if served is None:
        return {"model": judge.canonical_model(getattr(client, "model", "")), "digest": None, "via": None,
                "why": f"the server does not list {getattr(client, 'model', '')!r} (it lists {', '.join(ids[:8]) or 'none'})"}
    name = judge.canonical_model(served)
    p = Path(os.path.expanduser(served))
    if re.search(r"[/\\]|\.gguf$", served, re.IGNORECASE) and p.is_file():
        return {"model": name, "digest": f"sha256:{file_sha256(p)}", "via": f"sha256 of {p.resolve()}", "why": None}
    dg = _ollama_digest(client, served)
    if dg:
        return {"model": name, "digest": dg, "via": "ollama /api/show", "why": None}
    return {"model": name, "digest": None, "via": None,
            "why": f"the server reports no weights digest for {served!r}: it is not a GGUF file on this machine, "
                   f"and the server does not answer Ollama's /api/show"}
