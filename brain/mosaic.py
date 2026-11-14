"""The mosaic: what a company's public pieces say about things only people inside it know yet (docs/MOSAIC.md).

People inside a company know before the public that it is about to sell shares, or that its accounts will be late
or wrong. Each of those becomes public later, through a dated SEC filing. Until then the public holds pieces, none
of which says it outright: how many quarters of cash are left, a registration to sell securities already on file, a
jump in the price, a report that is later than the company's own habit, a change of auditor, a letter from the
SEC's staff. A mosaic sets the pieces side by side. Every piece here is public and dated; nothing here is, or
needs, inside information.

A *picture* is one hidden event, the filing that will reveal it, and the pieces that bear on it:

- **SALE**: the company sells new shares. Revealed by a prospectus (424B5, 424B4, 424B1) for common shares that
  opens a deal: one with no such prospectus in the ten sessions before it (a deal's preliminary and final
  prospectus are one deal). Most prospectuses of these forms sell bonds, so each one's first page is read
  (``read_offerings``) and filed as shares, debt, preferred or a resale by holders (``offering_kind``); only shares
  sold by the company count. A prospectus not yet read counts as nothing.
- **BOOKS**: its accounts are late or cannot be relied on. Revealed by a notice of late filing (NT 10-K, NT 10-Q)
  or an 8-K item 4.02 (earlier statements should no longer be relied on), again with none in the ten sessions before.
- **CUTS**: it cuts back or writes assets off. Revealed by an 8-K item 2.05 (costs of an exit or a restructuring,
  which is how layoffs and closures are reported) or 2.06 (a material impairment), with none in the ten before.

A row is one stock at one week's last session. Everything in it was public at that close:

- from prices (``brain/shocks.py``): ``lv60``, ``ret5``, ``ret20``, ``ret60``, ``a5``, ``big1``, ``logp``, ``pen``,
  ``ldv``, ``volsurge``, ``hi52``, ``lo52``, ``bio``, ``bio_known``, ``vix``; and ``up5``, the largest one-session
  rise over SPY's in the last five (0 when none rose): companies sell shares into a jump;
- share sales: ``sales3y`` (log of 1 + the deals opened in three years), ``sale_gap`` (log of 1 + sessions since
  the last sale prospectus, three years when none), ``sale_never``, ``sale10`` (one in the last ten sessions),
  ``debt250`` (a prospectus for bonds or preferred shares in a year: a company that can borrow);
- registrations: ``shelf3y`` and ``shelf60`` (an S-3, S-1, F-3 or F-1 filed in three years, in sixty sessions),
  ``effect60`` (the SEC declared one effective in sixty sessions), ``resale250`` (a resale prospectus in a year:
  a 424B3 or 424B7, or one of the sale forms whose first page says the holders are selling);
- the window: ``earn5`` (results reported in the last five sessions, when companies are free to sell again),
  ``k101_5`` (an 8-K item 1.01, a material agreement, in five sessions), ``unreg250`` (8-K item 3.02, shares sold
  privately, in a year);
- accounts (SEC XBRL, each number from the filing that first reported it): ``acct`` (cash and operating cash flow
  known, none older than 200 days), ``burn`` (operations used cash in the latest period), ``lrun`` (log of 1 + the
  quarters its liquid assets cover at that rate, 40 when it is not burning or not known; liquid assets are the
  larger of cash and current assets less current liabilities), ``negeq`` (stockholders' equity below zero),
  ``loss`` (a net loss in the latest period);
- the books: ``nt250`` and ``nt750`` (late-filing notices in one and three years), ``nt_gap`` (log of 1 + sessions
  since the last, three years when none), ``aud250`` (8-K item 4.01, a change of auditor, in a year), ``nonrel750``
  (item 4.02 in three years), ``delist250`` (item 3.01, a listing-rule notice, in a year), ``amend250`` (amended
  annual or quarterly reports in a year), ``off60`` (8-K item 5.02, officers or directors arriving or leaving, in
  sixty sessions), ``sec60`` (letters between the SEC's staff and the company made public in sixty sessions),
  ``late10`` and ``late_res`` (how many sessions a quarterly or annual report, and a results release, is overdue
  against the same one a year earlier, 0 to 60);
- cuts: ``cut750`` (log of 1 + the restructurings and write-offs opened in three years), ``cut_gap`` (log of 1 +
  sessions since the last, three years when none); ``earn_win`` and ``earn_known`` (S1's: a results report is
  expected within eight days, by last year's date), since cuts and share sales often come with or just after results.

The outcomes are ``y_sale``, ``y_books`` and ``y_cuts`` (the event came out in the next H sessions) and ``fwd`` (the
stock's return over SPY's across those sessions). Each picture has a model (a ridge logistic regression on its pieces,
standardized and clipped as in ``shocks.fit``) and a baseline that knows only the company's habit (how often and
how lately it has done this before). Both are fitted on 2011-2023 and live in ``knowledge/mosaic_p1.json``; the
years after belong to the declared tests (``brain/research/mosaictest.py``).

The universe is the companies listed today, so companies that failed or were bought are missing: a picture of a
coming takeover or bankruptcy cannot be tested here and is not offered. The card shows such pieces without a chance.

A chance that a filing comes is not a direction. Whether a flagged stock lags the market afterwards is its own
declared test. Nothing here trades.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from brain import config, events, shocks
from brain.bank import Bank, Fact, canon
from brain.timeutil import now_iso

H = 20                      # sessions ahead: four weeks
GAP = 10                    # a filing this soon after one of its kind belongs to the same event
TRAIN_START, TRAIN_END = shocks.TRAIN_START, shocks.TRAIN_END
UNIVERSES = ("us_broad", "us_small", "us_micro", "us_watch")
SALE_FORMS = ("424B5", "424B4", "424B1")
SHELF_FORMS = ("S-3", "S-1", "F-3", "F-1")
RESALE_FORMS = ("424B3", "424B7")
PERIODIC = ("10-K", "10-Q", "10-KT", "10-QT")
AMENDED = ("10-K/A", "10-Q/A")
LETTERS = ("UPLOAD", "CORRESP")
D13 = ("SC 13D", "SCHEDULE 13D")
ITEMS = {"2.02": "results", "4.01": "auditor", "4.02": "nonrel", "3.01": "delist", "5.02": "officer",
         "1.01": "agreement", "3.02": "unreg", "2.05": "cut", "2.06": "cut"}
YEAR, YEAR3 = 252, 756      # sessions
PAD = YEAR3 + 60            # room before the calendar's first session for filings made before it
STALE = 200                 # days: older accounts are not current
RUN_CAP = 40.0              # quarters of cash, capped
LATE_CAP = 60               # sessions overdue, capped
CASH = "us-gaap:CashAndCashEquivalentsAtCarryingValue"
CUR_A, CUR_L = "us-gaap:AssetsCurrent", "us-gaap:LiabilitiesCurrent"
OP_CF = "us-gaap:NetCashProvidedByUsedInOperatingActivities"
NET = ("us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss")
EQUITY = "us-gaap:StockholdersEquity"
PRICE = ("lv60", "ret5", "ret20", "ret60", "a5", "big1", "logp", "pen", "ldv", "volsurge", "hi52", "lo52", "bio",
         "bio_known", "vix")
SALE_HABIT = ("sales3y", "sale_gap", "sale_never", "sale10", "shelf3y")
SALE_FEATS = (*PRICE, "up5", *SALE_HABIT, "shelf60", "effect60", "resale250", "debt250", "earn5", "earn_win",
              "earn_known", "k101_5", "unreg250", "acct", "burn", "lrun", "negeq", "loss")
BOOKS_HABIT = ("nt250", "nt750", "nt_gap")
BOOKS_FEATS = ("lv60", "ret60", "logp", "pen", "ldv", "hi52", "bio", "bio_known", *BOOKS_HABIT, "aud250",
               "nonrel750", "delist250", "amend250", "off60", "sec60", "late10", "late_res", "acct", "burn", "lrun",
               "negeq", "loss")
CUTS_HABIT = ("cut750", "cut_gap")
CUTS_FEATS = ("lv60", "ret20", "ret60", "logp", "pen", "ldv", "hi52", "lo52", "bio", "bio_known", "vix", *CUTS_HABIT,
              "off60", "earn5", "earn_win", "earn_known", "acct", "burn", "lrun", "negeq", "loss", "nt750",
              "delist250")
COLS = tuple(dict.fromkeys((*SALE_FEATS, *BOOKS_FEATS, *CUTS_FEATS)))
PICTURES = {"sale": {"feats": SALE_FEATS, "habit": SALE_HABIT, "y": "y_sale",
                     "name": "a sale of new shares"},
            "books": {"feats": BOOKS_FEATS, "habit": BOOKS_HABIT, "y": "y_books",
                      "name": "late or unreliable accounts"},
            "cuts": {"feats": CUTS_FEATS, "habit": CUTS_HABIT, "y": "y_cuts",
                     "name": "a restructuring or a write-off"}}
TESTS = {"sale": ("P1a", "P1b"), "books": ("P2a", "P2b"), "cuts": ("P3a", "P3b")}
FIRST = {"sale": "sale_never", "books": "nt_gap", "cuts": "cut_gap"}   # the piece that says: none in three years
OUTCOMES = ("y_sale", "y_books", "y_cuts", "fwd")
LAM = 1.0
DEFAULT_MODEL = config.ROOT / "knowledge" / "mosaic_p1.json"
OFFER = "offering"          # the bank's fact for what a prospectus sells
READER = 2                  # the version of ``offering_kind``; a new rule is a new reader (1 lacked forward sales)
READ_FROM = "2008-01-01"    # three years before the fit's first rows
COVER_BYTES, COVER_CHARS = 40_000, 3000
OFFER_KIND = {"shares": "sale", "debt": "bond", "preferred": "bond", "holders": "resale"}
_DEBT = re.compile(r"\b(notes|debentures|bonds)\b", re.IGNORECASE)
_PREF = re.compile(r"preferred (stock|shares|units|securities)|preference shares|(?<!american )depositary shares",
                   re.IGNORECASE)
_SHARES = re.compile(r"common stock|common shares|ordinary shares|american depositary shares|common units|"
                     r"class [abc] (common )?(stock|shares)|\bunits\b|pre-funded warrants|subordinate voting shares|"
                     r"shares of beneficial interest|limited partner", re.IGNORECASE)
_NO_PROCEEDS = re.compile(r"will not receive any (of the )?(net )?proceeds", re.IGNORECASE)
_BY_US = re.compile(r"\bwe are (offering|selling) (?!any)|\bwe may (offer|sell)\b|forward sale agreement",
                    re.IGNORECASE)


# ---------- what a prospectus sells ----------
def offering_kind(text: str) -> tuple[str, str]:
    """What a prospectus sells, from the text of its first page, and the words that say so. The kind named first on
    the page wins: ``debt`` (notes, debentures, bonds), ``preferred`` (preferred or depositary shares), ``shares``
    (common stock, ordinary shares, American depositary shares, units, pre-funded warrants). Shares whose page says
    the company will not receive the proceeds, and nowhere that the company is offering, are ``holders``: a
    resale. (Under a forward sale agreement the company is paid later; that is still its own sale.) A page that
    names none is ``unknown``."""
    head = " ".join(text.split())[:COVER_CHARS]
    hits = {k: m.start() for k, rx in (("debt", _DEBT), ("preferred", _PREF), ("shares", _SHARES))
            if (m := rx.search(head))}
    if not hits:
        return "unknown", head[:200]
    kind = min(hits, key=hits.get)
    at = hits[kind]
    if kind == "shares" and _NO_PROCEEDS.search(head) and not _BY_US.search(head):
        kind = "holders"
    return kind, head[max(at - 80, 0):at + 160]


def read_offering(f: Fact) -> Fact:
    """The offering fact for one prospectus (one SEC request: the start of the document)."""
    from brain import scout
    from brain.sources import releases
    url = (f.source or {}).get("url", "")
    raw = releases._get(url, COVER_BYTES).decode("utf-8", "replace")
    kind, cover = offering_kind(scout.html_to_text(releases._WRAPPER.sub(" ", raw))["text"])
    return Fact(entity=f.entity, attr=OFFER, value=kind, as_of=f.as_of, known_at=f.known_at, tier="filed",
                source={"name": "SEC EDGAR prospectus", "url": url},
                context={"accn": (f.context or {}).get("accn"), "form": str(f.value), "cover": cover,
                         "reader": READER}, retrieved_at=now_iso())


def unread(view, t: str) -> list[Fact]:
    """The sale-form prospectuses of ``t`` since READ_FROM that this reader has not filed yet."""
    have = {(f.context or {}).get("accn") for f in view.facts(t, OFFER) if (f.context or {}).get("reader") == READER}
    return [f for f in view.facts(t, "filing")
            if str(f.value).upper().startswith(SALE_FORMS) and f.known_at[:10] >= READ_FROM
            and (f.context or {}).get("accn") not in have and (f.source or {}).get("url")]


def read_offerings(bank: Bank, tickers: list[str], log=print, limit: int | None = None) -> dict:
    """Read the first page of every sale-form prospectus not yet read and keep what each sells. Resumes if re-run.
    The SEC's shared rate limit spaces the requests (``brain/net.py``)."""
    from concurrent.futures import ThreadPoolExecutor

    from brain.sources import releases
    done = failed = 0
    with ThreadPoolExecutor(releases.WORKERS) as pool:
        for c, t in enumerate(tickers):
            b = Bank(bank.root)
            todo = unread(b.view(), t)
            if limit is not None:
                todo = todo[:max(limit - done, 0)]
            got = list(pool.map(_try_read, todo))
            new = [g for g in got if g is not None]
            if new:
                b.add_many(new)
            done += len(new)
            failed += len(got) - len(new)
            if log and (c + 1) % 100 == 0:
                log(f"{c + 1}/{len(tickers)} tickers, {done} prospectuses read, {failed} not read")
            if limit is not None and done >= limit:
                break
    if log:
        log(f"{done} prospectuses read, {failed} not read (they are tried again on the next run)")
    return {"read": done, "failed": failed}


def stale_accounts(view, t: str) -> bool:
    """A quarterly or annual report was filed on a later day than the latest account figure the bank holds."""
    filed_ = [f.known_at[:10] for f in view.facts(t, "filing") if str(f.value).upper() in PERIODIC]
    if not filed_:
        return False
    have = [f.known_at[:10] for a in (CASH, "us-gaap:Assets", OP_CF) for f in view.facts(t, a)]
    return not have or max(filed_) > max(have)


def refresh_accounts(bank: Bank, tickers: list[str], log=print) -> dict:
    """Read the SEC accounts again for every company that has reported since the bank last read them, so the cash
    pieces are current. One request a company; a company with nothing new is not asked for."""
    from brain.sources import sec
    done = failed = 0
    for t in tickers:
        b = Bank(bank.root)
        if not stale_accounts(b.view(), t):
            continue
        try:
            b.add_many(sec.facts_for(t))
            done += 1
        except Exception:  # noqa: BLE001 - a company with no XBRL accounts, or an outage: tried again next run
            failed += 1
    if log:
        log(f"accounts read again for {done} companies that reported since; {failed} could not be read")
    return {"read": done, "failed": failed}


def _try_read(f: Fact) -> Fact | None:
    try:
        return read_offering(f)
    except Exception:  # noqa: BLE001 - one document's outage must not stop the sweep; it stays unread
        return None


# ---------- pieces, from arrays (pure) ----------
def kinds(form: str, items: set[str]) -> list[str]:
    """What a filing is, for the mosaic: zero or more of the kinds its pieces are built from."""
    f = form.upper()
    out = []
    if f.startswith(SALE_FORMS):
        out.append("prospectus")              # what it sells is read from its first page (``filed``)
    if f.startswith(SHELF_FORMS):
        out.append("shelf")
    if f == "EFFECT":
        out.append("effect")
    if f.startswith(RESALE_FORMS):
        out.append("resale")
    if f.startswith("NT 10-"):
        out.append("nt")
    if f in PERIODIC:
        out.append("periodic")
    if f in AMENDED:
        out.append("amend")
    if f in LETTERS:
        out.append("letter")
    if f in D13:
        out.append("d13")
    if f in ("8-K", "8-K/A"):
        out += list(dict.fromkeys(name for item, name in ITEMS.items() if item in items))
    return out


def position(cal: np.ndarray, known_at: str) -> int:
    """The session at whose close a filing was first known. A filing made before the calendar starts gets a negative
    position, about as many sessions back as it was made (252 a year), so that a three-year count still sees it."""
    day = events._ny_date(known_at)
    if day < str(cal[0]):
        gap = (np.datetime64(str(cal[0])) - np.datetime64(day)).astype(int)
        return -round(gap * YEAR / 365.25) - 1
    return events.event_index(cal, known_at)


def _marks(n: int, idx) -> np.ndarray:
    m = np.zeros(n + PAD)
    for k in idx:
        if -PAD <= k < n:
            m[k + PAD] += 1
    return m


def count_in(n: int, idx, w: int) -> np.ndarray:
    """At each session, how many of ``idx`` fell in the last ``w`` sessions, that session included."""
    c = np.cumsum(_marks(n, idx))
    out = c.copy()
    out[w:] -= c[:-w]
    return out[PAD:]


def since(n: int, idx, cap: int) -> np.ndarray:
    """At each session, the sessions since the latest of ``idx`` at or before it; ``cap`` when none or longer."""
    m = _marks(n, idx)
    last = np.where(m > 0, np.arange(n + PAD), -1)
    last = np.maximum.accumulate(last)
    s = np.where(last >= 0, np.arange(n + PAD) - last, cap)
    return np.minimum(s, cap)[PAD:].astype(float)


def fresh(idx, gap: int = GAP) -> list[int]:
    """The positions that open an event: no position of the kind in the ``gap`` sessions before."""
    out, prev = [], None
    for k in sorted({int(k) for k in idx}):
        if prev is None or k - prev > gap:
            out.append(k)
        prev = k
    return out


def overdue(n: int, idx, year: int = YEAR, early: int = 30, grace: int = 3, reach: int = LATE_CAP) -> np.ndarray:
    """At each session, how many sessions a filing of this kind is overdue against the company's own habit. Each past
    filing sets an expected one a year (252 sessions) later. From ``grace`` sessions after that, until one arrives
    (any from ``early`` sessions before the expected session counts) or ``reach`` sessions pass, the kind is
    overdue by the sessions since the expected one. Only filings already made are read, so it is point in time."""
    ks = np.array(sorted({int(k) for k in idx}), int)
    out = np.zeros(n)
    for k in ks:
        e = int(k) + year
        j = int(np.searchsorted(ks, e - early))
        end = int(ks[j]) - 1 if j < len(ks) else n
        lo, hi = max(e + grace, 0), min(e + reach, end, n - 1)
        if lo <= hi:
            out[lo:hi + 1] = np.maximum(out[lo:hi + 1], np.arange(lo, hi + 1) - e)
    return out


def up5(adj: np.ndarray, bclose: np.ndarray, close: np.ndarray) -> np.ndarray:
    """The largest one-session rise over SPY's in the last five sessions, 0 when none rose (data errors left out,
    by the rule of ``shocks.frame``)."""
    with np.errstate(all="ignore"):
        r = np.r_[np.nan, adj[1:] / adj[:-1] - 1]
        rb = np.r_[np.nan, bclose[1:] / bclose[:-1] - 1]
    bad = (r > shocks.BAD_UP) | (r < shocks.BAD_DOWN) | (close < shocks.MIN_PRICE)
    return shocks._roll(np.clip(np.where(bad, np.nan, r - rb), 0, None), 5, "max", 4)


def step(n: int, facts) -> tuple[np.ndarray, np.ndarray]:
    """The latest period's value known at each session and that period's end, from ``facts``: (session first known,
    period end as a day number, value[, tiebreak]). A later period replaces an earlier one; a restatement of the
    same period replaces it from the session it was known."""
    val, end = np.full(n, np.nan), np.full(n, np.nan)
    best = -np.inf
    for f in sorted(facts):
        k, a, v = int(f[0]), float(f[1]), float(f[2])
        if k >= n or a < best:
            continue
        best = a
        val[max(k, 0):], end[max(k, 0):] = v, a
    return val, end


def money(n: int, days: np.ndarray, acc: dict) -> dict[str, np.ndarray]:
    """The account pieces at each session. ``days`` is each session's day number; ``acc`` holds, per concept, the
    facts as ``step`` reads them (flows already as a rate per quarter)."""
    cash, cash_end = step(n, acc.get("cash", []))
    ca, ca_end = step(n, acc.get("cur_a", []))
    cl, cl_end = step(n, acc.get("cur_l", []))
    cf, cf_end = step(n, acc.get("op_cf", []))
    ni, ni_end = step(n, acc.get("net", []))
    eq, eq_end = step(n, acc.get("equity", []))
    with np.errstate(all="ignore"):
        wc = np.where((days - ca_end <= STALE) & (days - cl_end <= STALE), ca - cl, np.nan)
        cash = np.where(days - cash_end <= STALE, cash, np.nan)
        liquid = np.fmax(cash, wc)
        known = np.isfinite(liquid) & np.isfinite(cf) & (days - cf_end <= STALE)
        burn = known & (cf < 0)
        run = np.where(burn, np.clip(liquid, 0, None) / np.where(burn, -cf, 1.0), RUN_CAP)
        negeq = np.isfinite(eq) & (days - eq_end <= STALE) & (eq < 0)
        loss = np.isfinite(ni) & (days - ni_end <= STALE) & (ni < 0)
    return {"acct": known.astype(float), "burn": burn.astype(float),
            "lrun": np.log1p(np.clip(run, 0, RUN_CAP)), "negeq": negeq.astype(float), "loss": loss.astype(float),
            "liquid": liquid, "cash": cash, "burn_q": np.where(burn, -cf, np.nan), "runway": np.where(burn, run, np.nan)}


def pieces(n: int, mk: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """The filing pieces at each session, from each kind's positions (``kinds``, ``position``)."""
    g = lambda k: mk.get(k, [])
    sale, nt = g("sale"), g("nt")
    books = [*nt, *g("nonrel")]
    return {
        "sales3y": np.log1p(count_in(n, fresh(sale), YEAR3)),
        "sale_gap": np.log1p(since(n, sale, YEAR3)),
        "sale_never": (since(n, sale, YEAR3) >= YEAR3).astype(float),
        "sale10": (count_in(n, sale, GAP) > 0).astype(float),
        "shelf3y": (count_in(n, g("shelf"), YEAR3) > 0).astype(float),
        "shelf60": (count_in(n, g("shelf"), 60) > 0).astype(float),
        "effect60": (count_in(n, g("effect"), 60) > 0).astype(float),
        "resale250": (count_in(n, g("resale"), YEAR) > 0).astype(float),
        "debt250": (count_in(n, g("bond"), YEAR) > 0).astype(float),
        "earn5": (count_in(n, g("results"), 5) > 0).astype(float),
        "k101_5": (count_in(n, g("agreement"), 5) > 0).astype(float),
        "unreg250": np.log1p(count_in(n, g("unreg"), YEAR)),
        "nt250": np.log1p(count_in(n, fresh(nt), YEAR)),
        "nt750": np.log1p(count_in(n, fresh(nt), YEAR3)),
        "nt_gap": np.log1p(since(n, books, YEAR3)),
        "aud250": (count_in(n, g("auditor"), YEAR) > 0).astype(float),
        "nonrel750": (count_in(n, g("nonrel"), YEAR3) > 0).astype(float),
        "delist250": (count_in(n, g("delist"), YEAR) > 0).astype(float),
        "amend250": np.log1p(count_in(n, g("amend"), YEAR)),
        "off60": np.log1p(count_in(n, g("officer"), 60)),
        "sec60": (count_in(n, g("letter"), 60) > 0).astype(float),
        "late10": overdue(n, g("periodic")),
        "late_res": overdue(n, g("results")),
        "cut750": np.log1p(count_in(n, fresh(g("cut")), YEAR3)),
        "cut_gap": np.log1p(since(n, g("cut"), YEAR3)),
    }


def outcomes(n: int, mk: dict[str, list[int]], adj: np.ndarray, bclose: np.ndarray) -> dict[str, np.ndarray]:
    """What the next H sessions brought: an event of each picture opening, and the return over SPY's."""
    def opened(idx) -> np.ndarray:
        m = np.zeros(n, bool)
        for k in fresh(idx):
            if 0 <= k < n:
                m[k] = True
        return shocks._fwd_any(m, H)

    fwd = np.full(n, np.nan)
    with np.errstate(all="ignore"):
        fwd[:-H] = (adj[H:] / adj[:-H]) / (bclose[H:] / bclose[:-H]) - 1
    return {"y_sale": opened(mk.get("sale", [])), "y_books": opened([*mk.get("nt", []), *mk.get("nonrel", [])]),
            "y_cuts": opened(mk.get("cut", [])), "fwd": fwd}


# ---------- from the bank ----------
def filed(view, t: str) -> list[tuple[Fact, list[str]]]:
    """Every filing of ``t`` that is a piece, with its kinds. A sale-form prospectus takes the kind its first page
    gives it (``OFFER_KIND``): ``sale`` for shares, ``bond`` for debt or preferred, ``resale`` for holders; one not
    read yet, or whose page names nothing, is ``unread`` and counts as nothing."""
    sells = {(f.context or {}).get("accn"): str(f.value) for f in view.facts(t, OFFER)
             if (f.context or {}).get("reader") == READER}
    out = []
    for f in view.facts(t, "filing"):
        form = str(f.value)
        items = {x.strip() for x in str((f.context or {}).get("items") or "").split(",")} \
            if form.upper() in ("8-K", "8-K/A") else set()
        ks = [OFFER_KIND.get(sells.get((f.context or {}).get("accn")), "unread") if k == "prospectus" else k
              for k in kinds(form, items)]
        if ks:
            out.append((f, ks))
    return out


def marks(view, t: str, cal: np.ndarray) -> dict[str, list[int]]:
    """Each kind's positions for ``t``: the session at whose close each such filing was first known."""
    out: dict[str, list[int]] = {}
    for f, ks in filed(view, t):
        k = position(cal, f.known_at)
        for name in ks:
            out.setdefault(name, []).append(k)
    return out


def _day(s: str) -> int:
    return int(np.datetime64(s[:10]).astype(int))


def quarter_rates(raw: list[tuple]) -> list[tuple]:
    """Flows (session first known, period end as a day number, days in the period, value) as ``step`` reads them,
    each as a rate per quarter. Where the filings allow it the rate is the trailing year's: a year-to-date figure
    plus the last full year less the same year-to-date a year earlier, each as known when the figure came out (one
    quarter with a single large receipt then moves the rate by a quarter of it, not all of it). Otherwise it is the
    period's own rate. Of two flows ending the same day the longer period wins."""
    years = [x for x in raw if x[2] >= 350]
    out = []
    for k, e, d, v in raw:
        rate = v * 91.25 / d
        if d < 350:
            fy = [x for x in years if x[0] <= k and abs(x[1] - (e - d)) <= 10]
            prior = [x for x in raw if x[0] <= k and abs(x[1] - (e - 365)) <= 10 and abs(x[2] - d) <= 10]
            if fy and prior:
                rate = (max(fy)[3] + v - max(prior)[3]) / 4
        out.append((k, e + d / 1000.0, rate))
    return out


def accounts(view, t: str, cal: np.ndarray) -> dict[str, list[tuple]]:
    """The account facts of ``t`` as ``step`` reads them. A flow (operating cash flow, net income) becomes a rate
    per quarter (``quarter_rates``); periods under 80 or over 380 days are left out."""
    def instant(attr: str) -> list[tuple]:
        return [(position(cal, f.known_at), _day(f.as_of), float(f.value)) for f in view.facts(t, attr)
                if isinstance(f.value, (int, float))]

    def flow(attrs) -> list[tuple]:
        raw = []
        for a in attrs if isinstance(attrs, tuple) else (attrs,):
            for f in view.facts(t, a):
                d = (f.context or {}).get("duration_days")
                if isinstance(f.value, (int, float)) and d and 80 <= d <= 380:
                    raw.append((position(cal, f.known_at), _day(f.as_of), int(d), float(f.value)))
        return quarter_rates(raw)

    return {"cash": instant(CASH), "cur_a": instant(CUR_A), "cur_l": instant(CUR_L), "equity": instant(EQUITY),
            "op_cf": flow(OP_CF), "net": flow(NET)}


def ticker_frame(view, t: str, mk) -> tuple[np.ndarray, dict] | None:
    """Every piece and outcome for every session of ``t``: S1's price features with the mosaic's added."""
    got = shocks.ticker_frame(view, t, mk)
    if not got:
        return None
    cal, f = got
    n = len(cal)
    df = shocks._series(view, t)
    adj, close = shocks._aligned(df, "adjclose", cal), shocks._aligned(df, "close", cal)
    f["up5"] = up5(adj, mk[1], close)
    m = marks(view, t, cal)
    f.update(pieces(n, m))
    f.update(money(n, cal.astype("datetime64[D]").astype(int), accounts(view, t, cal)))
    f.update(outcomes(n, m, adj, mk[1]))
    return cal, f


def universe_tickers(names=UNIVERSES) -> list[str]:
    return shocks.universe_tickers(names)


def table(bank: Bank, tickers: list[str], first: str, last: str, log=print, outcome: bool = True) -> dict:
    """Weekly rows (each week's last session) in [first, last] whose pieces are all known and, with ``outcome``,
    whose H sessions have closed: ``t`` and ``date`` per row, the pieces as the columns of ``X`` (``COLS``), and
    the outcomes. Arrays, not dicts: this is a few million numbers."""
    T, D, X, Y = [], [], [], {k: [] for k in OUTCOMES}
    for c in range(0, len(tickers), shocks.CHUNK):
        b = Bank(bank.root)
        v = b.view()
        mk = shocks.market(v)
        if mk is None:
            raise SystemExit(f"no {shocks.BENCH} or {shocks.VIX} prices in the bank")
        ends = shocks.week_ends(mk[0])
        ends = ends[(mk[0][ends] >= first) & (mk[0][ends] <= last)]
        for t in tickers[c:c + shocks.CHUNK]:
            got = ticker_frame(v, t, mk)
            if not got:
                continue
            cal, f = got
            M = np.column_stack([f[k][ends] for k in COLS])
            ok = np.isfinite(M).all(1)
            if outcome:
                ok &= np.isfinite(f["y_sale"][ends])
            if ok.any():
                T.append(np.full(int(ok.sum()), t))
                D.append(cal[ends][ok])
                X.append(M[ok].astype(np.float32))
                for k, col in Y.items():
                    col.append(f[k][ends][ok])
        if log:
            log(f"{min(c + shocks.CHUNK, len(tickers))}/{len(tickers)} tickers, {sum(len(x) for x in T)} stock-weeks")
        del b, v
        gc.collect()
    if not T:
        return {"t": np.array([], str), "date": np.array([], str), "X": np.zeros((0, len(COLS)), np.float32),
                **{k: np.array([]) for k in Y}}
    return {"t": np.concatenate(T), "date": np.concatenate(D).astype(str), "X": np.vstack(X),
            **{k: np.concatenate(v) for k, v in Y.items()}}


# ---------- the models ----------
def cols(feats) -> list[int]:
    return [COLS.index(f) for f in feats]


def first_time(pic: str, X: np.ndarray) -> np.ndarray:
    """The rows whose company has had no event of the picture in three years. Its habit is silent there, so what
    the model sees in those rows comes from the other pieces alone."""
    x = np.asarray(X)[:, COLS.index(FIRST[pic])]
    return x > 0 if pic == "sale" else x >= np.log1p(YEAR3) - 1e-4


def fit(Z: np.ndarray, y: np.ndarray, feats, lam: float = LAM) -> dict:
    """``shocks.fit`` on a matrix: clip at the 0.1 and 99.9 percentiles, standardize, ridge logistic regression."""
    Z = np.asarray(Z, float)
    lo, hi = np.percentile(Z, shocks.CLIP[0], axis=0), np.percentile(Z, shocks.CLIP[1], axis=0)
    Z = np.clip(Z, lo, hi)
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    w = shocks._logit(np.column_stack([np.ones(len(Z)), (Z - mu) / sd]), np.asarray(y, float), lam)
    return {"feats": list(feats), "lo": lo.tolist(), "hi": hi.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
            "w": w.tolist(), "lam": lam, "n": len(Z), "rate": float(np.mean(y))}


def chance(part: dict, X: np.ndarray) -> np.ndarray:
    """The fitted chance for each row of ``X`` (columns ``COLS``)."""
    if not len(X):
        return np.zeros(0)
    Z = np.clip(np.asarray(X, float)[:, cols(part["feats"])], part["lo"], part["hi"])
    S = np.column_stack([np.ones(len(Z)), (Z - np.array(part["mu"])) / np.array(part["sd"])])
    return 1 / (1 + np.exp(-np.clip(S @ np.array(part["w"]), -30, 30)))


def model_path() -> Path:
    """The fitted models' file: ``knowledge/mosaic_p1.json``, or BRAIN_MOSAIC_MODEL (tests, trials)."""
    return Path(os.environ.get("BRAIN_MOSAIC_MODEL") or DEFAULT_MODEL)


def load(path: Path | None = None) -> dict | None:
    p = Path(path or model_path())
    return json.loads(p.read_text()) if p.exists() else None


def model_sha(path: Path | None = None) -> str:
    return hashlib.sha256(Path(path or model_path()).read_bytes()).hexdigest()[:16]


def train_last(bank: Bank) -> str:
    """The last session whose H-session outcome closes by TRAIN_END: the fit reads no row after it, so no filing or
    price of the test's years enters the fit, even as an outcome."""
    cal = shocks.market(Bank(bank.root).view())[0]
    i = int(np.searchsorted(cal, TRAIN_END, side="right")) - 1 - H
    return str(cal[max(i, 0)])


def fit_and_save(bank: Bank, path: Path | None = None, log=print, tickers=None) -> dict:
    """Fit each picture's model and its habit-only baseline on TRAIN_START..TRAIN_END and save them. Nothing after
    TRAIN_END is read into the fit, as a piece or as an outcome: those years belong to the declared tests."""
    tab = table(bank, tickers or universe_tickers(), TRAIN_START, train_last(bank), log=log)
    if not len(tab["t"]):
        raise SystemExit("no stock-weeks to fit on")
    out = {"name": "P1", "train": [TRAIN_START, TRAIN_END], "universes": list(UNIVERSES), "h": H, "gap": GAP,
           "cols": list(COLS)}
    for name, pic in PICTURES.items():
        y = tab[pic["y"]]
        out[name] = {"model": fit(tab["X"][:, cols(pic["feats"])], y, pic["feats"]),
                     "baseline": fit(tab["X"][:, cols(pic["habit"])], y, pic["habit"])}
        if log:
            log(f"{name}: fitted on {len(y)} stock-weeks ({TRAIN_START}..{TRAIN_END}); {pic['name']} opened in "
                f"{float(np.mean(y)) * 100:.2f}% of them within {H} sessions")
    p = Path(path or model_path())
    p.write_text(canon(out) + "\n")
    if log:
        log(f"saved {p}")
    return out


# ---------- live ----------
def radar(bank: Bank, tickers: list[str], log=None, asof: str | None = None) -> list[dict]:
    """Each ticker's pieces at the bank's latest close (or at the close of ``asof``, reading nothing filed after
    it) and each picture's chance, as a list of dicts."""
    model = load()
    if model is None:
        raise SystemExit("no fitted model: run python -m brain mosaic fit")
    out = []
    for c in range(0, len(tickers), shocks.CHUNK):
        b = Bank(bank.root)
        v = b.view()
        mk = shocks.market(v)
        if mk is None:
            raise SystemExit(f"no {shocks.BENCH} or {shocks.VIX} prices in the bank")
        for t in tickers[c:c + shocks.CHUNK]:
            got = ticker_frame(v, t.upper(), mk)
            if not got:
                continue
            cal, f = got
            i = len(cal) - 1 if asof is None else int(np.searchsorted(cal, asof, side="right")) - 1
            while i > 0 and not np.isfinite(f["a1"][i]):
                i -= 1
            x = np.array([[f[k][i] for k in COLS]], float)
            if not np.isfinite(x).all():
                continue
            row = {"t": t.upper(), "date": str(cal[i]), **{k: float(f[k][i]) for k in COLS},
                   **{k: (float(f[k][i]) if np.isfinite(f[k][i]) else None) for k in ("liquid", "cash", "burn_q",
                                                                                     "runway")}}
            for name in PICTURES:
                row[name] = float(chance(model[name]["model"], x)[0])
                row[name + "_habit"] = float(chance(model[name]["baseline"], x)[0])
            out.append(row)
        if log:
            log(f"{min(c + shocks.CHUNK, len(tickers))}/{len(tickers)} tickers")
        del b, v
        gc.collect()
    return out


# ---------- live calls, recorded before their outcome ----------
RECORD_TOP = 25
DEFAULT_CALLS = config.ROOT / "research" / "mosaic_calls.jsonl"


def calls_path() -> Path:
    """The radar's recorded calls: ``research/mosaic_calls.jsonl``, or BRAIN_MOSAIC_CALLS (tests, trials)."""
    return Path(os.environ.get("BRAIN_MOSAIC_CALLS") or DEFAULT_CALLS)


def read_calls(path: Path | None = None) -> list[dict]:
    p = Path(path or calls_path())
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def record(rows: list[dict], path: Path | None = None, top: int = RECORD_TOP) -> int:
    """Write each picture's ``top`` stocks at the radar's close, with their chances, before the next H sessions
    trade. Only rows read at the close most rows share are taken, and a close already recorded for a picture is
    not written again."""
    if not rows:
        return 0
    p = Path(path or calls_path())
    dates = [r["date"] for r in rows]
    asof = max(set(dates), key=dates.count)
    have = {(c["asof"], c["picture"]) for c in read_calls(p)}
    new = []
    for pic in PICTURES:
        if (asof, pic) in have:
            continue
        for r in sorted((r for r in rows if r["date"] == asof), key=lambda r: -r[pic])[:top]:
            new.append({"asof": asof, "picture": pic, "t": r["t"], "chance": round(r[pic], 5),
                        "habit": round(r[pic + "_habit"], 5), "h": H, "model_sha": model_sha(), "at": now_iso()})
    if new:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.writelines(canon(c) + "\n" for c in new)
    return len(new)


def score(bank: Bank, path: Path | None = None) -> dict:
    """Each picture's recorded calls whose H sessions have closed: how many, for how many the filing came, and
    the mean chance they were given. A call is open until its H sessions have closed."""
    v = bank.view()
    cal = shocks.market(v)[0]
    seen: dict[str, dict] = {}
    out = {pic: {"judged": 0, "came": 0, "chance": 0.0, "open": 0} for pic in PICTURES}
    for c in read_calls(path):
        o = out[c["picture"]]
        i = int(np.searchsorted(cal, c["asof"]))
        if i + c["h"] > len(cal) - 1:
            o["open"] += 1
            continue
        mk = seen.setdefault(c["t"], marks(v, c["t"], cal))
        idx = {"sale": mk.get("sale", []), "books": [*mk.get("nt", []), *mk.get("nonrel", [])],
               "cuts": mk.get("cut", [])}[c["picture"]]
        o["judged"] += 1
        o["came"] += int(any(i < k <= i + c["h"] for k in fresh(idx)))
        o["chance"] += c["chance"]
    for o in out.values():
        o["chance"] = o["chance"] / o["judged"] if o["judged"] else None
    return out


def score_lines(sc: dict) -> list[str]:
    out = []
    for pic, o in sc.items():
        if o["judged"]:
            out.append(f"{PICTURES[pic]['name'].capitalize()}: {o['judged']} judged calls; the filing came for "
                       f"{o['came']} ({o['came'] / o['judged'] * 100:.1f}%); their chances averaged "
                       f"{o['chance'] * 100:.1f}%. {o['open']} open."
                       + (" Under 30 judged calls this says nothing yet." if o["judged"] < 30 else ""))
        else:
            out.append(f"{PICTURES[pic]['name'].capitalize()}: no judged calls yet; {o['open']} open.")
    return out


# ---------- in words ----------
def _pct(x: float) -> str:
    return f"{x * 100:+.0f}%"


def phrase(name: str, r: dict) -> str | None:
    """One piece in plain words, from a row's values; None for a piece that is not worth a line."""
    n = lambda k: round(float(np.expm1(r[k])))
    ago = lambda k: "this session" if k == 0 else f"{k} session{'s' if k != 1 else ''} ago"
    if name in ("lrun", "burn"):
        return (f"operations are using cash, with about {r['runway']:.1f} quarters of it left"
                if r.get("runway") is not None else None)
    if name in ("ret5", "ret20", "ret60"):
        return f"{_pct(np.expm1(r[name]))} against the market over {name[3:]} sessions"
    if name in ("a5", "big1"):
        return f"a {r['a5'] * 100:.0f}% session against the market in the last five"
    if name == "up5":
        return f"a rise of {r['up5'] * 100:.0f}% against the market in one of the last five sessions"
    if name in ("logp", "pen"):
        return f"a ${np.exp(r['logp']):.2f} stock"
    if name in ("nt250", "nt750"):
        return f"{n(name)} late-filing notice{'s' if n(name) != 1 else ''} in {'a year' if name == 'nt250' else 'three years'}"
    table_ = {
        "shelf60": "a registration to sell securities filed in the last 60 sessions",
        "shelf3y": "a registration to sell securities on file",
        "effect60": "a registration declared effective in the last 60 sessions",
        "sales3y": f"{n('sales3y')} sale{'s' if n('sales3y') != 1 else ''} of new shares opened in three years",
        "sale_gap": f"its last sale prospectus {ago(n('sale_gap'))}" if not r["sale_never"] else None,
        "sale_never": None,
        "sale10": "a sale prospectus in the last ten sessions",
        "earn5": "results reported in the last five sessions",
        "k101_5": "a material agreement (8-K item 1.01) in the last five sessions",
        "unreg250": "shares sold privately in the last year (8-K item 3.02)",
        "resale250": "holders registered shares for resale in the last year",
        "debt250": "sold bonds or preferred shares in the last year",
        "hi52": f"{-r['hi52'] * 100:.0f}% below its 52-week high",
        "lo52": f"{r['lo52'] * 100:.0f}% above its 52-week low",
        "lv60": f"moves about {np.exp(r['lv60']) * 100:.1f}% a day against the market",
        "ldv": (f"about ${np.exp(r['ldv']) / 1e6:.1f} m traded a day" if np.exp(r["ldv"]) >= 5e4 else
                "under $50,000 traded a day"),
        "volsurge": "its trading volume is up this week",
        "bio": "a drug or biotech company",
        "nt_gap": f"its last late-filing notice {ago(n('nt_gap'))}" if n("nt_gap") < YEAR3 else None,
        "aud250": "a change of auditor in the last year (8-K item 4.01)",
        "nonrel750": "a notice not to rely on earlier accounts in three years (8-K item 4.02)",
        "delist250": "a listing-rule notice in the last year (8-K item 3.01)",
        "amend250": f"{n('amend250')} amended report{'s' if n('amend250') != 1 else ''} in a year",
        "off60": f"{n('off60')} officer or director change{'s' if n('off60') != 1 else ''} in sixty sessions",
        "sec60": "letters with the SEC's staff made public in the last sixty sessions",
        "late10": f"a quarterly or annual report {r['late10']:.0f} sessions overdue against its own habit",
        "late_res": f"results {r['late_res']:.0f} sessions overdue against its own habit",
        "negeq": "liabilities exceed assets",
        "loss": "a net loss in the latest period",
        "cut750": f"{n('cut750')} restructuring or write-off notice{'s' if n('cut750') != 1 else ''} in three years",
        "cut_gap": f"its last restructuring or write-off {ago(n('cut_gap'))}" if n("cut_gap") < YEAR3 else None,
        "earn_win": "results due within eight days (by last year's date)" if not r["earn5"] else None,
    }
    return table_.get(name)


def why(part: dict, r: dict, k: int = 3) -> list[str]:
    """The ``k`` pieces that raise this row's chance most (weight times the standardized piece), in words."""
    z = (np.clip([r[f] for f in part["feats"]], part["lo"], part["hi"]) - np.array(part["mu"])) / np.array(part["sd"])
    push = np.array(part["w"][1:]) * z
    out: list[str] = []
    for i in np.argsort(-push):
        if push[i] <= 0 or len(out) >= k:
            break
        name = part["feats"][i]
        if r[name] == 0 and name not in ("lrun", "ret5", "ret20", "ret60", "hi52", "lo52", "logp", "lv60", "ldv"):
            continue                                   # a piece that is absent is not a reason
        s = phrase(name, r)
        if s and s not in out:
            out.append(s)
    return out


def verdict_lines() -> dict[str, str]:
    """Each mosaic test's recorded verdict, read from the ledger; a test with no result says so."""
    from brain.research import mosaictest, run
    got = {r["id"]: r["verdict"].split(":")[0] for r in run.verdicts() if run.is_mosaic(r)}
    return {tid: got.get(tid, "not run yet") for tid in mosaictest.IDS}


def latest_filings(view, t: str) -> dict[str, list[tuple[str, str, str]]]:
    """Per kind, the filings of ``t`` (New York date, form, address), newest first."""
    out: dict[str, list] = {}
    for f, ks in sorted(filed(view, t), key=lambda x: x[0].known_at, reverse=True):
        for name in ks:
            out.setdefault(name, []).append((events._ny_date(f.known_at), str(f.value), (f.source or {}).get("url", "")))
    return out


PARTNER_REACH = 45         # sessions back in which a partner's results report still counts as this season's


def partner_reports(bank: Bank, t: str) -> list[dict]:
    """The stock's partners (``brain/connect.py``: the stocks whose market-adjusted daily returns moved most with
    its own over the past year) and, for each that reported results in the last PARTNER_REACH sessions, the move
    over the three sessions around its report against SPY's."""
    from brain import connect
    v = bank.view()
    mk = shocks.market(v)
    if mk is None:
        return []
    cal, bclose, _ = mk
    tickers = universe_tickers()
    if t not in tickers:
        tickers = [*tickers, t]
    import warnings
    bk = connect.cached_book(bank, tickers, str(cal[-1]))
    with warnings.catch_warnings():            # a stock with no returns in the year has no spread to divide by
        warnings.simplefilter("ignore", RuntimeWarning)
        C = connect.correlations(connect.residuals(bk["R"], bk["m"])[0])
    out, n = [], len(cal)
    for p, c in connect.partners(C, tickers, t):
        row = {"t": p, "corr": float(c), "date": None, "move": None}
        reports, _, _ = shocks.filings(v, p, cal)
        idx = [int(np.searchsorted(cal, d)) for d in reports]
        idx = [e for e in idx if n - 1 - PARTNER_REACH <= e and e + 1 <= n - 1 and e >= 2]
        df = shocks._series(v, p)
        if idx and df is not None:
            e, adj = idx[-1], shocks._aligned(df, "adjclose", cal)
            with np.errstate(all="ignore"):
                mv = (adj[e + 1] / adj[e - 2]) / (bclose[e + 1] / bclose[e - 2]) - 1
            if np.isfinite(mv):
                row.update(date=str(cal[e]), move=float(mv))
        out.append(row)
    return out


def _m(x: float) -> str:
    return f"${x / 1e9:.2f} bn" if abs(x) >= 1e9 else f"${x / 1e6:.1f} m"


def card(bank: Bank, t: str) -> list[str]:
    """The mosaic of one stock, as lines: its pieces with the filing behind each, then each picture's chance."""
    t = t.upper()
    rows = radar(bank, [t])
    if not rows:
        return [f"{t}: not enough price history in the bank for a mosaic (python -m brain ingest {t})"]
    r, model, v = rows[0], load(), bank.view()
    fl = latest_filings(v, t)
    name = v.latest(t, "identity.name")
    cite: list[str] = []

    def ref(kind: str, i: int = 0) -> str:
        day, form, url = fl[kind][i]
        if url and url not in cite:
            cite.append(url)
        return f"{form}, {day}" + (f" [{cite.index(url) + 1}]" if url else "")

    out = [f"MOSAIC for {t}" + (f" ({name.value})" if name else "") + f" at the close of {r['date']}. Public filings "
           "and prices only; every piece is dated. A mosaic says what the pieces point to, not what will happen."]
    out.append("Pieces")
    if r["acct"]:
        cash = f"  cash     {_m(r['liquid'])} of liquid assets"
        out.append(cash + (f"; operations used {_m(r['burn_q'])} a quarter over the latest year, so about "
                           f"{r['runway']:.1f} quarters are left" if r["burn"] else "; operations are not using cash")
                   + ("; a net loss in the latest period" if r["loss"] and not r["burn"] else "")
                   + ("; liabilities exceed assets" if r["negeq"] else "")
                   + (f" ({ref('periodic')})" if fl.get("periodic") else ""))
    else:
        out.append("  cash     no current accounts in the bank (none filed in 200 days, or a company that files none)")
    if fl.get("shelf"):
        out.append(f"  shelf    a registration to sell securities: {ref('shelf')}"
                   + ("" if r["shelf3y"] else " (over three years old)")
                   + (f"; declared effective: {ref('effect')}" if r["effect60"] and fl.get("effect") else ""))
    else:
        out.append("  shelf    no registration to sell securities on file")
    n_s = round(float(np.expm1(r["sales3y"])))
    out.append(f"  sales    {n_s} sale{'s' if n_s != 1 else ''} of new shares opened in three years"
               + (f"; the last prospectus: {ref('sale')}" if fl.get("sale") else "")
               + (f"; holders registered shares for resale: {ref('resale')}" if r["resale250"] and fl.get("resale") else "")
               + (f"; shares sold privately: {ref('unreg')}" if r["unreg250"] and fl.get("unreg") else "")
               + (f"; bonds or preferred shares sold: {ref('bond')}" if r["debt250"] and fl.get("bond") else "")
               + (f"; {len(fl['unread'])} prospectus{'es' if len(fl['unread']) != 1 else ''} not read yet "
                  f"(python -m brain mosaic read)" if fl.get("unread") else ""))
    out.append("  window   " + (f"results reported: {ref('results')}" if fl.get("results") else "no results release on file")
               + (f"; overdue by {r['late_res']:.0f} sessions against its own habit" if r["late_res"] else "")
               + (f"; a material agreement: {ref('agreement')}" if r["k101_5"] and fl.get("agreement") else ""))
    out.append(f"  price    {_pct(np.expm1(r['ret20']))} against the market over 20 sessions and "
               f"{_pct(np.expm1(r['ret60']))} over 60; {-r['hi52'] * 100:.0f}% below its 52-week high; moves about "
               f"{np.exp(r['lv60']) * 100:.1f}% a day against the market; a ${np.exp(r['logp']):.2f} stock"
               + ("; a drug or biotech company" if r["bio"] else ""))
    books = []
    n_nt = round(float(np.expm1(r["nt750"])))
    books.append(f"{n_nt} late-filing notice{'s' if n_nt != 1 else ''} in three years"
                 + (f" (the last: {ref('nt')})" if n_nt and fl.get("nt") else ""))
    if r["late10"]:
        books.append(f"a quarterly or annual report is {r['late10']:.0f} sessions overdue against its own habit")
    if r["aud250"] and fl.get("auditor"):
        books.append(f"a change of auditor: {ref('auditor')}")
    if r["nonrel750"] and fl.get("nonrel"):
        books.append(f"a notice not to rely on earlier accounts: {ref('nonrel')}")
    if r["delist250"] and fl.get("delist"):
        books.append(f"a listing-rule notice: {ref('delist')}")
    if r["amend250"] and fl.get("amend"):
        books.append(f"an amended report: {ref('amend')}")
    if r["sec60"] and fl.get("letter"):
        books.append(f"letters with the SEC's staff made public: {ref('letter')}")
    out.append("  books    " + "; ".join(books))
    n_c = round(float(np.expm1(r["cut750"])))
    out.append(f"  cuts     {n_c} restructuring or write-off notice{'s' if n_c != 1 else ''} (8-K items 2.05, 2.06) in "
               f"three years" + (f"; the last: {ref('cut')}" if n_c and fl.get("cut") else ""))
    n_o = round(float(np.expm1(r["off60"])))
    out.append(f"  people   {n_o} officer or director change{'s' if n_o != 1 else ''} (8-K item 5.02) in sixty sessions"
               + (f"; the last: {ref('officer')}" if n_o and fl.get("officer") else ""))
    if fl.get("d13"):
        out.append(f"  holders  a holder of 5% or more who may seek influence filed: {ref('d13')}")
    try:
        pr = partner_reports(bank, t)
    except Exception:  # noqa: BLE001 - a card without its partners is still a card
        pr = []
    if pr:
        told = [f"{x['t']} {x['move'] * 100:+.1f}% around its report of {x['date']}" for x in pr if x["move"] is not None]
        out.append("  partners the stocks that moved most with it over the past year: "
                   + ", ".join(f"{x['t']} ({x['corr']:.2f})" for x in pr)
                   + (". Their latest results, against the market: " + "; ".join(told) if told else
                      ". None has reported in the last 45 sessions")
                   + ". Looked at on 2011-2023 and not declared as a test: partners' report moves barely told the "
                     "stock's own (research/MOSAIC-2026-11.md).")
    out.append(f"Pictures: the chance that the filing comes in the next {H} sessions (the average stock-week of "
               f"{model['train'][0][:4]}-{model['train'][1][:4]} in brackets)")
    tests = verdict_lines()
    for pic, ids in TESTS.items():
        part = model[pic]
        reasons = why(part["model"], r)
        out.append(f"  {PICTURES[pic]['name']}: {r[pic] * 100:.1f}% ({part['model']['rate'] * 100:.1f}%; its habit alone "
                   f"says {r[pic + '_habit'] * 100:.1f}%)" + (". Raised most by: " + "; ".join(reasons) if reasons else ""))
        out.append(f"      tests: does the chance see it coming, {ids[0]} {tests[ids[0]]}; do flagged stocks then "
                   f"lag, {ids[1]} {tests[ids[1]]} (research/RESULTS.md)")
    out.append("Not offered: a takeover or a failure. Companies that were bought or failed are no longer in the "
               "universe, so such a picture could not be tested here.")
    if cite:
        out.append("Sources: " + "  ".join(f"[{i + 1}] {u}" for i, u in enumerate(cite)))
    return out


def _cmd(a) -> int:
    bank = Bank()
    if a.action == "fit":
        fit_and_save(bank)
        return 0
    if a.action == "read":
        names = [t.upper() for t in a.tickers] or universe_tickers()
        read_offerings(bank, names)
        refresh_accounts(bank, names)
        return 0
    if a.action == "show":
        for t in a.tickers:
            print("\n".join(card(bank, t)), end="\n\n")
        return 0
    if a.action == "score":
        print("The mosaic radar's recorded calls (research/mosaic_calls.jsonl), judged once their 20 sessions closed. "
              "The tested evidence is P1a-P3b in research/RESULTS.md.")
        print("\n".join(score_lines(score(bank))))
        return 0
    model = load()
    if model is None:
        raise SystemExit("no fitted model: run python -m brain mosaic fit")
    rows = radar(bank, [t.upper() for t in a.tickers] or universe_tickers())
    tests = verdict_lines()
    print(f"Mosaic radar at the bank's latest close, {len(rows)} stocks. For each picture: the chance that the filing "
          f"which reveals it comes in the next {H} sessions, from public pieces only. A chance is not a direction.")
    for pic, ids in TESTS.items():
        part = model[pic]
        print(f"\n{PICTURES[pic]['name'].capitalize()} (the average stock-week of {model['train'][0][:4]}-"
              f"{model['train'][1][:4]}: {part['model']['rate'] * 100:.1f}%). Tests: sees it coming, {ids[0]} "
              f"{tests[ids[0]]}; flagged stocks then lag, {ids[1]} {tests[ids[1]]}.")
        for r in sorted(rows, key=lambda r: -r[pic])[:a.top]:
            print(f"  {r['t']:6} {r[pic] * 100:5.1f}%  (habit alone {r[pic + '_habit'] * 100:4.1f}%)  "
                  + "; ".join(why(part["model"], r)))
    if a.record:
        n = record(rows)
        print(f"\nrecorded {n} calls in {calls_path()}" + ("; commit it before the next session opens" if n else
                                                           " (this close was already recorded)"))
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("mosaic", help="what a company's public pieces point to: a share sale, late accounts, "
                                      "cuts (docs/MOSAIC.md)")
    p.add_argument("action", choices=["read", "fit", "radar", "show", "score"])
    p.add_argument("tickers", nargs="*")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--record", action="store_true",
                   help="radar: write each picture's top 25 to research/mosaic_calls.jsonl before their sessions trade")
    p.set_defaults(fn=_cmd)
