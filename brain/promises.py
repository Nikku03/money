"""The promise record: what a company said it would deliver, and whether it did (docs/PROMISES.md).

Asked for on 2026-11-25: "if we know the company in and out ... its next projects and what it need, and according to
timeline ... we can tell how much stock will move, and if it can deliver". The first promise a company makes in
public, on a date, with a number, is its revenue guidance: "we expect revenue of $70 million to $75 million next
quarter". The next report says whether it delivered.

**The promise.** Fixed rules read each earnings release's opening and forward-looking passages
(``brain/sources/releases.py``, reader 3) for revenue guidance stated in dollars: a range ("$3.12 billion to $3.38
billion"), a midpoint with a tolerance ("$2.70 billion, plus or minus $100 million", "± 5%"), or a single figure. A
figure counts only when a forward word (expects, guidance, outlook, forecast, ...) governs it in the same sentence or
bullet list, and nothing in between says it already happened (was, were, totaled). Its horizon is the next quarter or
the fiscal year, from the words beside it and from its size against the company's own reported revenue: a promise
must sit within half to twice the revenue of the comparable period (the last quarter or the year-ago quarter; the
last four quarters for a year), which drops a segment's guidance and a misread unit. When a release holds several for
one horizon, the largest is the company's total. Tables without units are not read (version 1): the record is
partial, and says how partial.

**The delivery.** A quarter's promise is held against the revenue the company reported (SEC XBRL, as the bank holds
it) for the first fiscal quarter ending after the release; a year's against the first fiscal year ending after it.
The figure is the one first reported for that period, not a later restatement (a company that splits in two
restates its past without the part it sold, which would turn every old promise into a miss); among the revenue
concepts a company files, the largest is its total. An actual further than half from the promise is a misread or a
bad account figure: the promise is kept in the record as a mismatch and not graded.
``surprise`` is the actual over the promise's midpoint, less one. The promise is ``beat`` above its top, ``missed``
below its bottom, else ``met`` (half a percent of tolerance for rounding). The outcome is known when the actual was
filed, never before.

**The record.** At any date, a company's record is its last eight quarterly promises whose outcomes were known by
then: how many it kept (met or beat), how many it beat, and its usual surprise. ``stretch`` says how much a new
promise asks: its midpoint's growth over the year-ago quarter, less the growth the company last reported.

A record describes the past. Whether it tells anything about the next report is for a declared test.
"""
from __future__ import annotations

import gzip
import json
import math
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from brain import config
from brain.bank import Bank

ATTR = "release.earnings"
TOL = 0.005                  # rounding tolerance on met / beat / missed
Q_BAND = (0.5, 2.0)          # a quarterly promise against the last or the year-ago quarter
Y_BAND = (0.6, 1.8)          # a yearly promise against the last four quarters
Q_AHEAD = 125                # the promised quarter ends within this many days of the release
RECORD = 8                   # promises in a company's record
MIN_RECORD = 4
READER_MIN = 3               # releases read with their forward-looking passages (releases.READER)
HABIT = 4                    # deliveries in a company's habit: the median surprise of its last four
OPEN_DAYS = 130              # an open promise older than this is no longer the one in force
# Where the delivery falls around the promise's midpoint moved by the habit: the surprise less the habit, pooled
# over the promises made in 2011-2023 (research/promises/explore_train.log, section E): the middle 2 in 3 and the
# middle 9 in 10.
BAND68 = (-0.035, 0.030)
BAND90 = (-0.081, 0.081)

_UNIT = {"thousand": 1e3, "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}
_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_AMT = r"\$\s?" + _NUM
_U = r"\s?(thousand|million|mn|billion|bn)\b"
_GAP = r"[^$;•]{0,90}?"
_REV = re.compile(r"\b(?:total\s+|net\s+)?(?:revenues?|sales)\b", re.IGNORECASE)
_FWD = re.compile(r"\b(expects?|expected|expecting|anticipates?|anticipated|forecasts?|forecasting|projects?|"
                  r"projected|guidance|outlook|guiding|targets?|targeting|estimates?|estimated)\b", re.IGNORECASE)
_PAST = re.compile(r"\b(was|were|totaled|totalled|came in|exceeded|increased|decreased|grew|declined|rose|fell)\b",
                   re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?][\"”’)]?\s+(?=[A-Z])")
_PM = re.compile(_GAP + _AMT + _U + r",?\s*(?:plus or minus|\+\s?/\s?-|±)\s*(?:" + _AMT + _U + r"|" + _NUM + r"\s?%)",
                 re.IGNORECASE)
_RANGE = re.compile(_GAP + _AMT + r"(?:" + _U + r")?\s*(?:to|and|-|–|—)\s*" + _AMT + _U, re.IGNORECASE)
_POINT = re.compile(_GAP + _AMT + _U, re.IGNORECASE)
_QTR = re.compile(r"\b(?:first|second|third|fourth|next|current|upcoming)[- ](?:fiscal )?quarter|\bQ[1-4]\b|"
                  r"\bquarter ending\b|\b[FQ]{1,2}[1-4][- ]?(?:FY)?\d{2,4}\b", re.IGNORECASE)
_YEAR = re.compile(r"\bfull[- ]year\b|\bfiscal (?:year )?(?:20)?\d{2}\b|\bFY ?(?:20)?\d{2}\b|\bfor the year\b|"
                   r"\bannual\b|\byear ending\b|\b20\d\d (?:revenue|sales|guidance|outlook)\b", re.IGNORECASE)
_NOT_REVENUE = re.compile(r"(cost of|deferred|percent of|% of|unearned)\s*$", re.IGNORECASE)
_GROWTH = re.compile(r"\b(growth|increase|increases|decline|declines|decrease|improvement|headwind|tailwind|impact)\b",
                     re.IGNORECASE)
_SEGMENT_AFTER = re.compile(r"\b(?:for|from|of|in)\s+(?!Q[1-4]\b|FY|Fiscal\b|The\b|Full\b|First\b|Second\b|Third\b|"
                            r"Fourth\b)[A-Z][A-Za-z]")
_WORD_BEFORE = re.compile(r"([A-Za-z][A-Za-z\-]*|\d{2,4})[ ,]*$")      # none when a bullet or a colon comes first
# The word just before "revenue" or "sales" when the figure is the company's total: anything else ("service
# revenues", "Cloud revenue", "property sales", "product sales") names a part of it.
_TOTAL_BEFORE = frozenset((
    "total net consolidated company our its the of in expects expect expected expecting anticipates anticipate "
    "anticipated estimated estimate forecast forecasts forecasting projects projected quarter year annual full-year "
    "fiscal quarterly for with generate deliver report and gaap that see sees is are be to guidance outlook "
    "targeting target revenue reported group").split())
_BREAK = re.compile(r"[•▪■●◦·–—;:|*]")
MISMATCH = (-0.30, 0.50)     # an actual this far below or above the promise is a misread or a company that
#                              changed shape (a split, a merger closing): kept, not graded
PRELIM_AFTER, PRELIM_BEFORE = 25, 40   # a release this soon after a quarter's end, with another within 40 days, is
#                                        a pre-announcement of that quarter, not a promise
_OTHER = re.compile(r"\band\b|\binclud", re.IGNORECASE)
# A quarter named with its year beside the figure ("4Q19", "Q4 2019", "fourth quarter of fiscal 2019"): when that
# quarter ended before the release by the plain calendar, the figure is a result, not a promise. A fiscal year
# named ahead of the calendar only makes a quarter look later, so a real promise is never dropped by this.
_QNUM = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_QLABEL = re.compile(r"\b([1-4])Q\s?'?(20\d{2}|\d{2})\b|\bQ([1-4])\s?(?:FY)?\s?'?(20\d{2}|\d{2})\b|"
                     r"\b(first|second|third|fourth)[- ](?:fiscal )?quarter(?: of)?(?: fiscal(?: year)?)? (20\d{2})\b",
                     re.IGNORECASE)


def quarter_label(text: str, start: int, end: int) -> tuple[int, int] | None:
    """(quarter, year) of the last quarter named with its year in ``text[start:end]``; None when a quarter is named
    after it without a year ("Reports Fourth Quarter 2020 Results; Expects First Quarter Revenue of ..."): then
    the later one governs the figure."""
    got, at = None, 0
    for m in _QLABEL.finditer(text, start, end):
        if m.group(1):
            k, y = int(m.group(1)), int(m.group(2))
        elif m.group(3):
            k, y = int(m.group(3)), int(m.group(4))
        else:
            k, y = _QNUM[m.group(5).lower()], int(m.group(6))
        got, at = (k, y + (2000 if y < 100 else 0)), m.end()
    if got and any(q.start() >= at for q in _QTR.finditer(text, at, end)):
        return None
    return got


_LABEL_STOP = re.compile(r"[.!?][\"”’)]?(?:\s+\d{1,2})?\s+(?=[A-Z•▪■●◦])")     # a sentence's end, a bullet after it


def _sentence_start(text: str, lo: int, hi: int) -> int:
    """Where the sentence holding ``hi`` starts, looking back no further than ``lo``."""
    ends = [m.end() for m in _LABEL_STOP.finditer(text, lo, hi)]
    return ends[-1] if ends else lo


def ended_before(qlabel: tuple[int, int] | None, d: str) -> bool:
    """Whether the labelled quarter ended, by the plain calendar, before the date ``d``."""
    if not qlabel:
        return False
    k, y = qlabel
    return f"{y}-{3 * k:02d}-{31 if k in (1, 4) else 30}" < d
_LABEL = re.compile(r"\b(?:fiscal(?: year)?|FY|full[- ]year|year)[\s']{0,2}((?:20)?\d{2})\b|"
                    r"\b(20\d\d)\s+(?:total\s+|net\s+)?(?:revenues?|sales|guidance|outlook)", re.IGNORECASE)


def path() -> Path:
    return config.ROOT / "var" / "promises" / "promises.json.gz"


# ---------- reading a release (pure) ----------
def _money(num: str, unit: str | None) -> float | None:
    u = _UNIT.get((unit or "").lower())
    return float(num.replace(",", "")) * u if u else None


def _amount(text: str, start: int) -> tuple[float, float, int, bool] | None:
    """(low, high, end index, is a single point) of the first figure after ``start``, or None."""
    seg = text[start:start + 260]
    if (m := _PM.match(seg)):
        mid = _money(m.group(1), m.group(2))
        if mid is None:
            return None
        tol = _money(m.group(3), m.group(4)) if m.group(3) else mid * float(m.group(5)) / 100
        if tol is None or not (0 <= tol < mid):
            return None
        return mid - tol, mid + tol, start + m.end(), False
    if (m := _RANGE.match(seg)):
        hi = _money(m.group(3), m.group(4))
        lo = _money(m.group(1), m.group(2) or m.group(4))
        if lo is None or hi is None or not (0 < lo <= hi <= 3 * lo):
            return None
        return lo, hi, start + m.end(), False
    if (m := _POINT.match(seg)):
        v = _money(m.group(1), m.group(2))
        return (v, v, start + m.end(), True) if v else None
    return None


def candidates(text: str) -> list[dict]:
    """Every revenue figure of ``text`` that a forward word governs: {low, high, point, words (quarter, year or
    both), at}. Sizes are not checked here."""
    out = []
    for m in _REV.finditer(text):
        if _NOT_REVENUE.search(text[max(0, m.start() - 14):m.start()]):
            continue
        w = _WORD_BEFORE.search(text[max(0, m.start() - 40):m.start()])
        if w and not w.group(1).isdigit() and w.group(1).lower() not in _TOTAL_BEFORE:
            continue
        got = _amount(text, m.end())
        if not got:
            continue
        lo, hi, end, point = got
        gap = text[m.end():end]
        lead = gap.split("$")[0]
        if _GROWTH.search(gap) or _SEGMENT_AFTER.search(lead) or _OTHER.search(lead):
            continue
        between = [f for f in _FWD.finditer(text, m.end(), end)]
        before = [f for f in _FWD.finditer(text, max(0, m.start() - 220), m.start())]
        before = [f for f in before if not text[max(0, f.start() - 5):f.start()].lower().endswith("than ")]
        if between:
            f = between[-1]
            span = text[m.start():end]
        elif before:
            f = before[-1]
            if _SENTENCE_END.search(text[f.end():m.start()]):
                continue
            # a single figure in a bullet under an "Outlook" heading is as often the quarter's reported sales:
            # it counts only when the forward word governs it directly ("expects 2023 revenue to be $42.7 billion")
            if point and (m.start() - f.end() > 60 or _BREAK.search(text[f.end():m.start()])):
                continue
            span = text[f.start():end]
        else:
            continue
        if _PAST.search(span) or "includ" in text[max(0, f.start() - 25):f.start()].lower():
            continue
        ctx = text[max(0, f.start() - 120):min(len(text), end + 60)]
        words = ("quarter" if _QTR.search(ctx) else "") + ("year" if _YEAR.search(ctx) else "")
        labels = [int(x.group(1) or x.group(2)) for x in _LABEL.finditer(text, max(0, f.start() - 120), end)]
        label = (labels[-1] + (2000 if labels[-1] < 100 else 0)) if labels else None
        out.append({"low": lo, "high": hi, "point": point, "words": words, "at": m.start(), "label": label,
                    "qlabel": quarter_label(text, _sentence_start(text, max(0, f.start() - 120),
                                                                  min(f.start(), m.start())), end),
                    "text": " ".join(text[max(0, f.start() - 40):end].split())[:260]})
    return out


def horizon(c: dict, last_q: float | None, year_ago_q: float | None, ttm: float | None) -> str | None:
    """"quarter", "year" or None: by the words beside the figure, and by its size against the company's own revenue
    (the bands of the module docstring). With both or neither word, the closer size decides."""
    mid = (c["low"] + c["high"]) / 2

    def near(base, band):
        return base and base > 0 and band[0] <= mid / base <= band[1]
    q_ok = near(last_q, Q_BAND) or near(year_ago_q, Q_BAND)
    y_ok = near(ttm, Y_BAND)
    if c["words"] == "quarter":
        return "quarter" if q_ok else None
    if c["words"] == "year":
        return "year" if y_ok else None
    if q_ok and y_ok:
        dq = min(abs(math.log(mid / b)) for b in (last_q, year_ago_q) if b and b > 0)
        return "quarter" if dq <= abs(math.log(mid / ttm)) else "year"
    return "quarter" if q_ok else "year" if y_ok else None


def grade(actual: float, low: float, high: float) -> str:
    if actual > high * (1 + TOL):
        return "beat"
    if actual < low * (1 - TOL):
        return "missed"
    return "met"


# ---------- revenue as first reported ----------
def _first(facts) -> dict[str, tuple[float, str]]:
    """Per period end, the earliest-known figure of each concept, then the largest across concepts (the total):
    end -> (value, known_at)."""
    per: dict[tuple[str, str], object] = {}
    for f in facts:
        k = (f.attr, f.as_of)
        if k not in per or f.known_at < per[k].known_at:
            per[k] = f
    out: dict[str, tuple[float, str, str]] = {}
    for (_, end), f in per.items():
        v = float(f.value)
        if end not in out or v > out[end][0]:
            out[end] = (v, f.known_at, f.context.get("start", ""))
    return out


def first_reported(view, ticker: str) -> tuple[list[dict], list[dict]]:
    """(quarters, years) of revenue as first reported, oldest first: {end, value, known_at}. A fourth quarter the
    company did not report by itself is its year less its nine months, both as first reported."""
    from brain.metrics import NINE_MONTH_DAYS, QUARTER_DAYS, YEAR_DAYS, _dur
    from brain.sources.sec import REVENUE

    def span(lo, hi, annual_form=False):
        got = []
        for a in REVENUE:
            got += view.facts(ticker, a, lambda f: (d := _dur(f)) is not None and lo <= d <= hi and (
                not annual_form or str(f.context.get("form", "")).startswith("10-K")))
        return _first(got)
    q, y, n = span(*QUARTER_DAYS), span(*YEAR_DAYS, annual_form=True), span(*NINE_MONTH_DAYS)
    for end, (v, ka, start) in y.items():
        nine = [(e, x) for e, x in n.items() if x[2][:7] == start[:7] and 80 <= (date.fromisoformat(end)
                                                                                   - date.fromisoformat(e)).days <= 100]
        if nine:
            e, (nv, nka, _) = max(nine)
            known = max(ka, nka)
            if end not in q or known < q[end][1]:      # the fourth quarter as first known, not a later restatement
                q[end] = (v - nv, known, "")
    quarters = [{"end": e, "value": v, "known_at": ka} for e, (v, ka, _) in sorted(q.items())]
    years = [{"end": e, "value": v, "known_at": ka} for e, (v, ka, _) in sorted(y.items())]
    return quarters, years


# ---------- a company's promises, from the bank ----------
def _ny(known_at: str) -> str:
    from brain.events import _ny_date
    return _ny_date(known_at)


def company(view, ticker: str) -> list[dict]:
    """Every revenue promise read from ``ticker``'s releases, oldest first, each with its outcome when known."""
    t = ticker.upper()
    rels = sorted((f for f in view.facts(t, ATTR) if (f.context or {}).get("reader", 1) >= READER_MIN),
                  key=lambda f: f.known_at)
    if not rels:
        return []
    qs, years = first_reported(view, t)
    out = []
    days = [date.fromisoformat(_ny(f.known_at)) for f in rels]
    for k, f in enumerate(rels):
        c = f.context or {}
        text = (c.get("opening") or "") + " … " + (c.get("outlook") or "")
        d = _ny(f.known_at)
        ended = [q["end"] for q in qs if q["end"] < d]
        if (ended and (days[k] - date.fromisoformat(ended[-1])).days <= PRELIM_AFTER and k + 1 < len(days)
                and (days[k + 1] - days[k]).days <= PRELIM_BEFORE):
            continue                                   # a pre-announcement of the quarter just ended
        past = [q for q in qs if q["end"] < d and q["value"] > 0]
        last_q = past[-1]["value"] if past else None
        four = past[-4:]
        whole = len(four) == 4 and 250 <= (date.fromisoformat(four[-1]["end"])
                                           - date.fromisoformat(four[0]["end"])).days <= 290
        ttm = sum(q["value"] for q in four) if whole else None
        nxt = next((q for q in qs if q["end"] >= d), None)              # the quarter in progress at the release
        if nxt and (date.fromisoformat(nxt["end"]) - date.fromisoformat(d)).days > Q_AHEAD:
            nxt = None
        year_ago = None
        if nxt:
            ya = [q for q in qs if abs((date.fromisoformat(nxt["end"]) - date.fromisoformat(q["end"])).days - 364) <= 10]
            year_ago = ya[0]["value"] if ya else None
        elif past:
            # the promised quarter is not on file yet (an open promise): it is the first quarter to end after the
            # release, counted from the last quarter on file, which may be one behind the release
            last_end = date.fromisoformat(past[-1]["end"])
            n = max(1, math.ceil((date.fromisoformat(d) - last_end).days / 91.31))
            target = last_end + timedelta(days=round(91.31 * n))
            ya = [q for q in qs if abs((target - date.fromisoformat(q["end"])).days - 364) <= 12]
            year_ago = ya[0]["value"] if ya else None
        best: dict[str, dict] = {}
        for cand in candidates(text):
            h = horizon(cand, last_q, year_ago, ttm)
            if h == "quarter" and ended_before(cand.get("qlabel"), d):
                continue                               # a quarter already over: a result, not a promise
            if h and (h not in best or cand["high"] + cand["low"] > best[h]["high"] + best[h]["low"]):
                best[h] = cand
        for h, cand in best.items():
            mid = (cand["low"] + cand["high"]) / 2
            row = {"t": t, "made": d, "known_at": f.known_at, "horizon": h, "low": cand["low"], "high": cand["high"],
                   "mid": mid, "point": cand["point"], "text": cand["text"], "ref": f.ref,
                   "last_q": last_q, "year_ago_q": year_ago, "ttm": ttm}
            if h == "quarter":
                if nxt:
                    row.update(period_end=nxt["end"], actual=nxt["value"], outcome_known=nxt["known_at"])
                if year_ago and year_ago > 0:
                    row["asks"] = mid / year_ago - 1          # growth over the year-ago quarter the promise asks for
                    lya = [q for q in qs if past and abs((date.fromisoformat(past[-1]["end"])
                                                          - date.fromisoformat(q["end"])).days - 364) <= 10]
                    if lya and lya[0]["value"] > 0:
                        row["did"] = past[-1]["value"] / lya[0]["value"] - 1    # the growth it last reported
                        row["stretch"] = row["asks"] - row["did"]
            else:
                fy = next((y for y in years if y["end"] >= d), None)
                done_fy = [y["end"] for y in years if y["end"] < d]
                end = fy["end"] if fy else ((date.fromisoformat(done_fy[-1]) + timedelta(days=365)).isoformat()
                                            if done_fy else None)
                if cand.get("label") and end:
                    named = int(end[:4]) - (1 if int(end[5:7]) <= 2 else 0)     # a year ending in Jan-Feb may carry
                    if cand["label"] not in (named, int(end[:4])):              # the year before's name
                        continue            # a year already ended (a result, not a promise) or one further out
                if fy:
                    row.update(period_end=fy["end"], actual=fy["value"], outcome_known=fy["known_at"])
                elif end:
                    row["period_end"] = end        # the year in progress: where it should end, so that a later
                    #                                promise for the same year is seen as a raise or a cut
            if row.get("actual") is not None:
                if row["actual"] <= 0 or not (MISMATCH[0] <= row["actual"] / mid - 1 <= MISMATCH[1]):
                    row["mismatch"] = True        # a misread promise or a bad account figure: kept, not graded
                else:
                    row["surprise"] = row["actual"] / mid - 1
                    row["grade"] = grade(row["actual"], cand["low"], cand["high"])
            out.append(row)
    return out


def changes(rows: list[dict]) -> list[dict]:
    """Each yearly promise against the company's previous promise for the same fiscal year: raised, cut or held
    (a midpoint move of 1% or more)."""
    last: dict[tuple[str, str], float] = {}
    for r in rows:
        if r["horizon"] != "year" or not r.get("period_end"):
            continue
        k = (r["t"], r["period_end"])
        if k in last and last[k] > 0:
            ch = r["mid"] / last[k] - 1
            r["change"] = ch
            r["changed"] = "raised" if ch >= 0.01 else "cut" if ch <= -0.01 else "held"
        last[k] = r["mid"]
    return rows


def record(rows: list[dict], asof: str, horizon_: str = "quarter") -> dict | None:
    """The company's record at ``asof`` (UTC time): its last RECORD promises of that horizon whose outcomes were known
    by then. None with fewer than MIN_RECORD."""
    done = [r for r in rows if r["horizon"] == horizon_ and r.get("grade") and r["outcome_known"] <= asof]
    done = done[-RECORD:]
    if len(done) < MIN_RECORD:
        return None
    s = np.array([r["surprise"] for r in done])
    return {"n": len(done), "kept": float(np.mean([r["grade"] != "missed" for r in done])),
            "beat": float(np.mean([r["grade"] == "beat" for r in done])),
            "missed": float(np.mean([r["grade"] == "missed" for r in done])),
            "surprise": float(np.median(s)), "last": done[-1]["grade"], "last_surprise": float(s[-1])}


def habit(rows: list[dict], made: str) -> float | None:
    """The company's habit when a promise is made on the date ``made``: the median surprise of its last HABIT
    quarterly deliveries, the quarter it has just reported included. A delivery counts when its quarter ended
    before ``made`` (the release that makes a promise reports the quarter before; the bank holds that figure from
    the accounts filed days later, which state the same revenue). A quarter promised twice counts once, by its
    last promise. None with fewer than HABIT."""
    last: dict[str, dict] = {}
    for r in rows:
        if (r["horizon"] == "quarter" and r.get("grade") and r.get("period_end") and r["period_end"] < made
                and (r["period_end"] not in last or r["known_at"] > last[r["period_end"]]["known_at"])):
            last[r["period_end"]] = r
    got = [last[k]["surprise"] for k in sorted(last)][-HABIT:]
    return float(np.median(got)) if len(got) >= HABIT else None


def expected(rows: list[dict], asof: str) -> dict | None:
    """Where the company's habit puts the delivery of its latest open quarterly promise: the midpoint moved by the
    habit, with the ranges BAND68 and BAND90 around that. None without a habit, or without an open promise made in
    the last OPEN_DAYS days. In the days between a release and the accounts of the quarter it reports, the habit
    is one delivery behind."""
    open_ = [r for r in rows if r["horizon"] == "quarter" and not r.get("grade") and not r.get("mismatch")
             and r["known_at"] <= asof]
    if not open_:
        return None
    r = open_[-1]
    if (date.fromisoformat(asof[:10]) - date.fromisoformat(r["made"])).days > OPEN_DAYS:
        return None
    u = habit([x for x in rows if x.get("outcome_known", "") <= asof], r["made"])
    if u is None:
        return None
    mid = r["mid"]
    return {"made": r["made"], "mid": mid, "usual": u, "point": mid * (1 + u),
            "lo68": mid * (1 + u + BAND68[0]), "hi68": mid * (1 + u + BAND68[1]),
            "lo90": mid * (1 + u + BAND90[0]), "hi90": mid * (1 + u + BAND90[1])}


def verdicts() -> dict[str, str]:
    """The recorded verdict of the declared test G1a ("not run" when absent)."""
    try:
        from brain.research import run
        v = {r["id"]: r["verdict"].split(":")[0] for r in run.verdicts() if r.get("kind") == "promise"}
    except Exception:  # noqa: BLE001 - a card without verdicts says so
        v = {}
    return {t: v.get(t, "not run") for t in ("G1a",)}


def after_close(known_at: str, asof: str) -> bool:
    """Whether a release came out after the 4 p.m. New York close of the session ``asof``."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ny = datetime.fromisoformat(known_at.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
    d = ny.date().isoformat()
    return d > asof or (d == asof and ny.hour >= 16)


def said(view, ticker: str, asof: str) -> str | None:
    """For a company that reported after the close of ``asof`` (a date): what it had promised for the quarter it
    just reported, its record, and what its new release promises. One line for the desk's case file; None when no
    promise can be read."""
    t = ticker.upper()
    rows = changes(company(view, t))
    if not rows:
        return None
    new = [r for r in rows if after_close(r["known_at"], asof)]
    before = [r for r in rows if not after_close(r["known_at"], asof)]
    old = [r for r in before if r["horizon"] == "quarter"]
    parts = []
    stamp = (before[-1]["known_at"] if before else asof) if not new else min(r["known_at"] for r in new)
    stamp = max(stamp, asof + "T20:00:00Z") if not new else stamp
    if old and (date.fromisoformat(asof) - date.fromisoformat(old[-1]["made"])).days <= OPEN_DAYS:
        r = old[-1]
        rng = _usd(r["low"]) if r["point"] else f"{_usd(r['low'])} to {_usd(r['high'])}"
        line = f"for the quarter it has just reported it had promised revenue of {rng} (on {r['made']})"
        if (rec := record(rows, stamp)):
            line += (f"; its record over its last {rec['n']} quarterly promises: kept {rec['kept'] * 100:.0f}%, beat "
                     f"{rec['beat'] * 100:.0f}%")
        if (e := expected(before, stamp)):
            line += (f"; its habit ({e['usual'] * 100:+.1f}% against the midpoint over its last {HABIT} deliveries "
                     f"on file) put the delivery near {_usd(e['point'])}")
        parts.append(line)
    for r in new:
        rng = _usd(r["low"]) if r["point"] else f"{_usd(r['low'])} to {_usd(r['high'])}"
        if r["horizon"] == "quarter":
            ask = (f", which asks for {r['asks'] * 100:+.0f}% over the year-ago quarter after {r['did'] * 100:+.0f}% "
                   f"in its last quarter on file" if r.get("stretch") is not None else "")
            parts.append(f"its new promise for the next quarter is {rng}{ask}")
        else:
            ch = (f" ({r['changed']} {r['change'] * 100:+.1f}% against its last promise for the same year)"
                  if r.get("changed") else "")
            parts.append(f"its promise for the year is {rng}{ch}")
    if not parts:
        return None
    v = verdicts()
    return ("; ".join(parts) + f". Read by fixed rules from its releases (docs/PROMISES.md): check the figures "
            f"against the release. Test: the habit carries to the next delivery G1a {v['G1a']}; what the price "
            f"does after a raise or a cut has not been tested")


# ---------- the library ----------
def build(bank: Bank, tickers: list[str], log=print) -> list[dict]:
    view = bank.view()
    out: list[dict] = []
    for k, t in enumerate(tickers):
        try:
            out += changes(company(view, t))
        except Exception as e:  # noqa: BLE001 - one company's accounts must not stop the build
            if log:
                log(f"{t}: skipped ({type(e).__name__}: {str(e)[:80]})")
        if log and (k + 1) % 100 == 0:
            log(f"{k + 1}/{len(tickers)} companies, {len(out)} promises")
    return out


def save(rows: list[dict], p: Path | None = None) -> Path:
    p = p or path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return p


def load(p: Path | None = None) -> list[dict]:
    p = p or path()
    if not p.exists():
        return []
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


# ---------- the card ----------
def _usd(x: float) -> str:
    return f"${x / 1e9:.2f} billion" if abs(x) >= 1e9 else f"${x / 1e6:.1f} million"


def render(ticker: str, rows: list[dict], asof: str) -> list[str]:
    t = ticker.upper()
    L = [f"{t}: its promises (revenue guidance read from its earnings releases) and whether it delivered"]
    if not rows:
        return L + ["  no revenue promise could be read from its releases (many give none, or give it in a table "
                    "without units, which version 1 does not read)"]
    rec = record(rows, asof)
    if rec:
        L.append(f"  record, its last {rec['n']} quarterly promises with a known outcome: kept {rec['kept'] * 100:.0f}% "
                 f"(beat {rec['beat'] * 100:.0f}%, missed {rec['missed'] * 100:.0f}%); usually "
                 f"{rec['surprise'] * 100:+.1f}% against its midpoint; the last one {rec['last']} "
                 f"({rec['last_surprise'] * 100:+.1f}%)")
    else:
        L.append("  record: fewer than four quarterly promises with a known outcome")
    open_ = [r for r in rows if not r.get("grade") and not r.get("mismatch")]
    for r in open_[-2:]:
        rng = _usd(r["low"]) if r["point"] else f"{_usd(r['low'])} to {_usd(r['high'])}"
        ask = (f"; that asks for {r['asks'] * 100:+.0f}% over the year-ago quarter, after {r['did'] * 100:+.0f}% in its "
               f"last quarter on file" if r.get("stretch") is not None else "")
        L.append(f"  open promise ({r['horizon']}, made {r['made']}): revenue of {rng}{ask}")
    v = verdicts()
    if (e := expected(rows, asof)):
        L.append(f"  its habit (the median of its last {HABIT} deliveries on file: {e['usual'] * 100:+.1f}% against "
                 f"the midpoint) puts the delivery near {_usd(e['point'])}: 2 in 3 between {_usd(e['lo68'])} and "
                 f"{_usd(e['hi68'])}, 9 in 10 between {_usd(e['lo90'])} and {_usd(e['hi90'])} (ranges from "
                 f"2011-2023's promises; the habit's test G1a {v['G1a']}, research/RESULTS.md)")
    for r in [r for r in rows if r.get("grade")][-6:]:
        rng = _usd(r["low"]) if r["point"] else f"{_usd(r['low'])} to {_usd(r['high'])}"
        ch = f", {r['changed']} {r['change'] * 100:+.1f}%" if r.get("changed") else ""
        L.append(f"  {r['made']} {r['horizon']} to {r['period_end']}: promised {rng}{ch}; delivered {_usd(r['actual'])}: "
                 f"{r['grade']} ({r['surprise'] * 100:+.1f}%)")
    L.append(f"  Test (research/RESULTS.md): the habit carries to the next delivery G1a {v['G1a']}. Not tested yet: "
             f"whether a promise that asks for more falls short, and what the price does after the year's promise "
             f"changes. A delivery is not a direction for the stock (docs/PROMISES.md).")
    return L


def _cmd(a) -> int:
    from brain.timeutil import now_iso
    bank = Bank()
    if a.action == "build":
        from brain import episodes
        tickers = [x.upper() for x in a.tickers] or list(episodes.universe_tickers())
        rows = build(bank, tickers)
        p = save(rows)
        got = len({r["t"] for r in rows})
        print(f"{len(rows)} promises of {got} companies written to {p}")
        return 0
    view = bank.view()
    for t in a.tickers:
        print("\n".join(render(t, changes(company(view, t)), now_iso())))
        print()
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("promises", help="what a company said it would deliver, and whether it did (docs/PROMISES.md)")
    p.add_argument("action", choices=["show", "build"])
    p.add_argument("tickers", nargs="*")
    p.set_defaults(fn=_cmd)
