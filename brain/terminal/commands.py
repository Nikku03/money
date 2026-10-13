"""The terminal's command line: `[TICKER] FUNCTION [ARGS...]`, then GO.

Forgiving on purpose, because people type the way other terminals taught them:
function first (`GP AAPL 5Y`), a bare ticker (`AAPL` means `AAPL DES`), and the
noise words `US EQUITY <GO>` are all accepted. What it will not do is guess a
function it does not know: an unknown code is an error with the closest codes as
the hint. Grammar and shapes: docs/TERMINAL.md section 1.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from brain.markets import BY_SYMBOL


@dataclass
class Command:
    fn: str                 # upper-case function code, e.g. "GP"
    ticker: str | None      # upper-case, e.g. "AAPL", "^GSPC"
    args: list[str] = field(default_factory=list)  # upper-cased, except ASK's question (verbatim)
    raw: str = ""


class CommandError(ValueError):
    """Bad input. ``message`` says what is wrong; ``hint`` is a command or the closest codes."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message, hint)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message


FUNCTIONS: dict[str, dict] = {
    "HELP": {"scope": "global", "args": "", "desc": "list of functions and keys"},
    "DES": {"scope": "security", "args": "",
            "desc": "description: identity, price summary, key derived metrics, latest filings and headlines"},
    "GP": {"scope": "security", "args": "[RANGE] [CMP...]",
           "desc": "price chart; RANGE 1M 3M 6M YTD 1Y 2Y 5Y 10Y MAX (default 1Y); up to 3 comparisons rebased to 100"},
    "FA": {"scope": "security", "args": "", "desc": "financial analysis: up to 10 fiscal years of annual statements and ratios"},
    "CF": {"scope": "security", "args": "[FORM]", "desc": "company filings, optionally one form (10-K, 8-K, 4 ...)"},
    "CN": {"scope": "security", "args": "",
           "desc": "company news: headlines (unscored) and web claims with their source's record"},
    "EV": {"scope": "security", "args": "",
           "desc": "evidence: where the ticker ranks on each tested signal, and that signal's TEST verdict"},
    "BRIEF": {"scope": "security", "args": "", "desc": "the full cited brief"},
    "INS": {"scope": "security", "args": "",
            "desc": "insider trades (SEC Forms 4/5): open-market buys and sales, newest first, with 90-day totals"},
    "ERN": {"scope": "security", "args": "",
            "desc": "earnings by quarter: EPS, EPS YoY, SUE and the stock's reaction and drift vs SPY"},
    "JET": {"scope": "security", "args": "",
            "desc": "aircraft the company itself owns (FAA registry) and their airport-to-airport flights"},
    "SHOW": {"scope": "both", "args": "ID", "desc": "one fact by id prefix: full JSON and source link"},
    "WEI": {"scope": "global", "args": "", "desc": "world markets: equity indices, volatility, FX, commodities, crypto"},
    "MOV": {"scope": "global", "args": "", "desc": "research-universe movers: last session and 1 month"},
    "CURVE": {"scope": "global", "args": "",
              "desc": "US Treasury par yield curve: latest, 1 month ago, 1 year ago, key spreads"},
    "ECO": {"scope": "global", "args": "", "desc": "macro table: Treasury yields and spreads, FRED series if ingested"},
    "RES": {"scope": "global", "args": "", "desc": "research ledger verdicts"},
    "CLM": {"scope": "global", "args": "", "desc": "web-claim scoring per source"},
    "SHIP": {"scope": "global", "args": "[AREA]",
             "desc": "port areas: latest commercial-vessel count per area with its z-score; SHIP AREA charts one"},
    "CAM": {"scope": "global", "args": "[ID]",
            "desc": "official traffic cameras: latest vehicle count per camera with its z-score; CAM ID charts one"},
    "WL": {"scope": "global", "args": "[ADD|DEL TICKER...]",
           "desc": "watchlist: last price, 1D and latest alert per row; WL ADD / WL DEL edit its tickers"},
    "ALRT": {"scope": "global", "args": "[ACK ID | ACK ALL]",
             "desc": "alerts from the watchlist's rules, newest first; ALRT ACK ID acknowledges one"},
    "ASK": {"scope": "global", "args": "QUESTION...",
            "desc": "ask the analyst (an open-weight model on Ollama or another local server); it reads the bank"},
    "INGEST": {"scope": "both", "args": "TICKER | MARKETS | MACRO | UNIVERSE | INSIDERS | WATCH",
               "desc": "start a background collection job"},
    "JOBS": {"scope": "global", "args": "", "desc": "background jobs and their logs"},
}

KEYS = [
    {"key": "Enter", "desc": "run the command (GO)"},
    {"key": "Esc", "desc": "clear the command line"},
    {"key": "Up / Down", "desc": "walk the command history"},
    {"key": "Tab", "desc": "accept the highlighted completion"},
    {"key": "F1", "desc": "HELP"},
    {"key": "Ctrl+1..4", "desc": "focus pane 1-4"},
    {"key": "Ctrl+Enter", "desc": "run the command in the next pane"},
    {"key": "Click a ticker", "desc": "run <TICKER> DES in the same pane (Shift: next pane)"},
    {"key": "Click [abcd1234]", "desc": "show that fact and its source"},
    {"key": "Click ALRT n", "desc": "(top bar) ALRT: the alerts, n of them unacknowledged"},
]

NOISE = {"US", "EQUITY", "EQUITIES", "<GO>", "GO"}
TRAILING_NOISE = {"GO"}  # also a ticker (Grocery Outlet): noise only at the end of a line (`AAPL GO`)
RANGES = ("1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "10Y", "MAX")
RANGE_SHAPE = re.compile(r"\d+[DWMY]")  # fullmatch; a GP argument even when it is not one of RANGES
VERBS = {"INGEST", "SHOW"}  # first on a line they are always the function: `INGEST CF` ingests CF
FREE_ARGS = {"ASK", "HELP"}  # first on a line they are always the function, whatever follows
INGEST_KEYWORDS = ("MARKETS", "MACRO", "UNIVERSE", "INSIDERS", "WATCH")
TICKER = re.compile(r"[A-Za-z0-9.^=\-]{1,15}")  # fullmatch; same rule as ingest targets
_FACT_ID = re.compile(r"[0-9A-Fa-f]{4,20}")
MAX_COMPLETIONS = 12
NOT_TICKERS = {"MACRO"}  # bank entities that hold macro series, not securities
ALT_PREFIXES = ("AREA.", "CAM.")  # bank entities of port areas and traffic cameras, not securities
WL_VERBS = ("ADD", "DEL")


def _is_arg(fn: str, w: str) -> bool:
    """Tokens that are this function's arguments, never its ticker (so `GP 5Y AAPL` finds AAPL)."""
    if fn == "GP":
        return w in RANGES or bool(RANGE_SHAPE.fullmatch(w))  # `GP 3Y AAPL`: 3Y is a bad range, not the ticker
    if fn == "CF":  # forms carry a digit (10-K, 8-K, 4, 144); market symbols with digits carry . ^ or =
        return any(c.isdigit() for c in w) and not any(c in w for c in ".^=")
    if fn == "SHOW":
        return "/" in w or (bool(_FACT_ID.fullmatch(w)) and any(c.isdigit() for c in w))
    if fn == "INGEST":
        return w in INGEST_KEYWORDS
    return False


def suggest(code: str) -> str:
    """Hint for an unknown function code: the closest codes, or HELP."""
    close = difflib.get_close_matches(code.upper(), list(FUNCTIONS), n=3, cutoff=0.5)
    return f"closest: {', '.join(close)}" if close else "HELP"


def _function_position(words: list[str]) -> int | None:
    """Index of the function code: `[TICKER] FUNCTION` wins over function-first when both read,
    so a ticker that is also a code still works: `CF DES`, `MOV DES`, `RES GP 1Y`. Both read only
    when the second code takes a ticker (`GP MOV 5Y` charts MOV). ASK and HELP take free text, and
    the verbs INGEST and SHOW take a target, so first on a line they are always the function."""
    w0 = words[0]
    w1 = words[1] if len(words) > 1 else None
    if w0 in FUNCTIONS:
        if w1 in FUNCTIONS and FUNCTIONS[w1]["scope"] != "global" and w0 not in VERBS | FREE_ARGS:
            return 1
        return 0
    if w1 in FUNCTIONS:
        return 1
    return None


def parse(line: str) -> Command:
    raw = line if isinstance(line, str) else ""
    spans = [(m.group().upper(), m.end()) for m in re.finditer(r"\S+", raw)]
    words = [(w, end) for w, end in spans if w not in NOISE - TRAILING_NOISE]
    dropped = None  # `AAPL GO` is AAPL; `GO DES` and a bare `GO` are the ticker GO
    while len(words) > 1 and words[-1][0] in TRAILING_NOISE:
        dropped = words.pop()[0]
    if not words:
        raise CommandError("empty command", "HELP")
    pos = _function_position([w for w, _ in words])
    if pos is None:
        if len(words) > 1:
            raise CommandError(f"unknown function '{words[1][0]}'", suggest(words[1][0]))
        fn, ticker, after = "DES", words[0][0], []
    else:
        fn = words[pos][0]
        ticker = words[0][0] if pos == 1 else None
        after = words[pos + 1:]

    args: list[str] = []
    if fn == "ASK":
        question = raw[words[pos][1]:].strip()
        args = [question] if question else []
    else:
        scope = FUNCTIONS[fn]["scope"]
        for w, _ in after:
            if ticker is None and scope != "global" and not _is_arg(fn, w):
                ticker = w
            else:
                args.append(w)

    if dropped and ticker is None and (FUNCTIONS[fn]["scope"] == "security" or (fn == "INGEST" and not args)):
        ticker = dropped  # `DES GO`, `INGEST GO`: the trailing GO was the only ticker on the line
    if ticker is not None and not TICKER.fullmatch(ticker):
        raise CommandError(f"'{ticker}' is not a ticker or a function", "HELP")
    if FUNCTIONS[fn]["scope"] == "security" and ticker is None:
        raise CommandError(f"{fn} needs a ticker, e.g. AAPL {fn}", f"AAPL {fn}")
    if args and FUNCTIONS[fn]["scope"] == "global" and not FUNCTIONS[fn]["args"] and fn not in FREE_ARGS:
        raise CommandError(f"{fn} takes no arguments", fn)  # never ignored silently: `MOV XYZ` is not MOV
    if fn == "INGEST" and ticker is None and not args:
        raise CommandError("INGEST needs a target: a ticker, MARKETS, MACRO, UNIVERSE, INSIDERS or WATCH",
                           "INGEST AAPL")
    if fn == "SHOW" and not args:
        raise CommandError("SHOW needs a fact id, e.g. AAPL SHOW 9de710d2", f"{ticker or 'AAPL'} DES")
    return Command(fn=fn, ticker=ticker, args=args, raw=raw)


def is_security(entity: str) -> bool:
    """A bank entity that is a security (or market symbol): not MACRO, a port area or a camera."""
    e = entity.upper()
    return e not in NOT_TICKERS and not e.startswith(ALT_PREFIXES)


def _words(words, partial: str, desc: str) -> list[dict]:
    return [{"text": w, "kind": "keyword", "desc": desc} for w in words if w.startswith(partial)]


def _ticker_items(partial: str, tickers: list[str]) -> list[dict]:
    """Bank securities plus the world-market symbols (which can be ingested on demand), by symbol or name."""
    out = []
    known = sorted({t.upper() for t in tickers if is_security(t)} | set(BY_SYMBOL))
    for t in known:
        name = BY_SYMBOL.get(t, {}).get("name", "")
        if t.startswith(partial) or (len(partial) >= 3 and name.upper().startswith(partial)):
            out.append({"text": t, "kind": "ticker", "desc": name})
    return out


def _area_names(tickers: list[str]) -> list[str]:
    names = {t.upper()[len("AREA."):] for t in tickers if t.upper().startswith("AREA.")}
    try:
        from brain.sources.shipping import AREAS
        names |= {a["name"] for a in AREAS}
    except Exception:  # noqa: BLE001 - completion is a convenience; a broken source must not break typing
        pass
    return sorted(names)


def _arg_items(fn: str, rest: list[str], partial: str, tickers: list[str]) -> list[dict]:
    """Completions for the arguments of the global functions that take some."""
    if fn == "SHIP" and not rest:
        return _words(_area_names(tickers), partial, "port area")
    if fn == "CAM" and not rest:
        cams = sorted({t.upper()[len("CAM."):] for t in tickers if t.upper().startswith("CAM.")})
        return _words(cams, partial, "camera sampled into the bank")
    if fn == "WL":
        if not rest:
            return _words(WL_VERBS, partial, "add tickers to the watchlist, or take them off")
        if rest[0] in WL_VERBS and partial:
            return _ticker_items(partial, tickers)
    if fn == "ALRT":
        if not rest:
            return _words(("ACK",), partial, "acknowledge an alert: ALRT ACK ID, or ALRT ACK ALL")
        if rest == ["ACK"]:
            return _words(("ALL",), partial, "acknowledge every alert")
    return []


def complete(prefix: str, tickers: list[str]) -> list[dict]:
    """Completions for the token being typed: function codes, tickers (bank securities plus the
    world-market symbols, which can be ingested on demand) and the keywords some functions take
    (INGEST targets, SHIP areas, CAM ids, WL ADD/DEL, ALRT ACK)."""
    text = prefix or ""
    if not text.strip():
        return []
    toks = text.split()
    partial = "" if text[-1].isspace() else toks[-1].upper()
    done = [t.upper() for i, t in enumerate(toks if not partial else toks[:-1])
            if t.upper() not in NOISE or (i == 0 and t.upper() in TRAILING_NOISE)]  # a leading GO is a ticker
    fns = [t for t in done if t in FUNCTIONS]
    if "ASK" in fns:
        return []  # free text from here on
    fn = fns[0] if fns else None
    rest = done[done.index(fn) + 1:] if fn else []
    has_ticker = any(t not in FUNCTIONS for t in done)

    out: list[dict] = []
    if fn in ("SHIP", "CAM", "WL", "ALRT"):
        out = _arg_items(fn, rest, partial, tickers)
    else:
        if not fns:
            security_first = sorted(FUNCTIONS, key=lambda c: has_ticker and FUNCTIONS[c]["scope"] == "global")
            out += [{"text": c, "kind": "function", "desc": FUNCTIONS[c]["desc"]}
                    for c in security_first if c.startswith(partial)]
        if fn == "INGEST" and not rest:
            out += _words(INGEST_KEYWORDS, partial, "background collection job")
        if partial and (not has_ticker or "GP" in fns):  # GP takes comparison tickers
            out += _ticker_items(partial, tickers)
    out.sort(key=lambda d: d["text"] != partial)  # an exact match first; otherwise order is kept
    return out[:MAX_COMPLETIONS]
