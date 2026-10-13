"""Insider trades: open-market purchases (P) and sales (S) reported on Forms 4 and 5.

History comes from SEC's Insider Transactions Data Sets (one zip per quarter, public
domain); the current quarter from each Form 4's XML on EDGAR. Only the issuer's own
listed shares (the non-derivative table) and only codes P and S are kept: awards, option
exercises, gifts and tax withholding say nothing about what an insider thinks the stock
is worth.

Insiders appear in their public role only, as the filing states it: name, SEC CIK,
relationship to the issuer and title. Addresses, signatures and remarks in the filings are
never read into the bank. Amendments (4/A, 5/A) are skipped: they restate trades already
counted from the original filing.

Facts are content-addressed, so two lots in one filing with the same date, size and price
(three 5,000-share sales at $0.33, say) would be one fact. Each repeat carries
``context["lot"]`` = 2, 3, ... so every reported transaction counts once.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from xml.etree import ElementTree as ET

from brain import config
from brain.bank import Bank, Fact
from brain.net import FetchError, fetch
from brain.sources.sec import ARCHIVE
from brain.timeutil import now_iso, norm_known_at

DATASET_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
# From 2026q2 SEC publishes the data sets under a new folder; older ones stay where they were.
DATASET_URLS = (DATASET_URL, "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/"
                             "{q}_form345.zip")
ATTR = "insider.trade"
CODES = ("P", "S")
FORMS = ("4", "5")
SOURCE = "SEC Form 4"
RELATIONSHIPS = (("officer", "Officer"), ("director", "Director"), ("ten_percent_owner", "TenPercentOwner"),
                 ("other", "Other"))
_MONTHS = {m: i for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
                                       "DEC"), 1)}


def index_url(cik: int | str, accn: str) -> str:
    """The filing's index page on EDGAR, the citation for every trade in it."""
    return ARCHIVE.format(cik=int(cik), accn=accn.replace("-", "")) + f"{accn}-index.htm"


def _date(s: str) -> str | None:
    """'15-JAN-2024' (the data sets) or '2024-01-15' (Form 4 XML) -> ISO date; None when unreadable or
    outside 1980-2100 (the data sets hold typos such as year 0015)."""
    s = (s or "").strip()
    m = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})", s)
    try:
        d = date(int(m[3]), _MONTHS[m[2].upper()], int(m[1])) if m else date.fromisoformat(s[:10]) if s else None
    except (KeyError, ValueError):
        return None
    return d.isoformat() if d is not None and 1980 <= d.year <= 2100 else None


def _num(s) -> float | None:
    try:
        x = float(str(s).strip())
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _relationship(flags: set[str]) -> str:
    return ",".join(k for k, _ in RELATIONSHIPS if k in flags) or "other"


def _fact(ticker, accn, url, filed_known, trans_date, code, shares, price, owners, ownership, fetched_at) -> Fact | None:
    if trans_date is None or shares is None or code not in CODES or trans_date > filed_known[:10]:
        return None  # no date, no amount, or a trade "after" its own filing: unusable
    shares = abs(shares)
    flags = {r for o in owners for r in o["relationship"].split(",") if r}
    titles = [o["title"] for o in owners if o.get("title")]
    return Fact(entity=ticker, attr=ATTR, value=shares, unit="shares", as_of=trans_date, known_at=filed_known,
                tier="filed", source={"name": SOURCE, "url": url},
                context={"code": code, "price": price, "value_usd": round(shares * price, 2) if price is not None else None,
                         "owner": "; ".join(o["name"] for o in owners), "owner_cik": ",".join(o["cik"] for o in owners),
                         "relationship": _relationship(flags), "title": "; ".join(titles) or None, "accn": accn,
                         "ownership": "I" if ownership == "I" else "D"},
                retrieved_at=fetched_at)


def _number_lots(facts: list[Fact]) -> list[Fact]:
    """Keep repeated identical transactions distinct: the k-th copy (k >= 2) gets context lot = k."""
    seen: dict[str, int] = {}
    out = []
    for f in facts:
        k = seen[f.id] = seen.get(f.id, 0) + 1
        if k > 1:
            f = Fact(entity=f.entity, attr=f.attr, value=f.value, unit=f.unit, as_of=f.as_of, known_at=f.known_at,
                     tier=f.tier, source=f.source, context={**f.context, "lot": k}, retrieved_at=f.retrieved_at)
        out.append(f)
    return out


# ---------- bulk: the quarterly data sets ----------
def _tsv(z: zipfile.ZipFile, name: str):
    with z.open(name) as raw:
        yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline=""),
                                  delimiter="\t", quoting=csv.QUOTE_NONE)


def parse_dataset(zip_bytes: bytes, cik_to_ticker: dict[int, str], fetched_at: str) -> list[Fact]:
    """Open-market trades in one quarterly data set, for the issuers in ``cik_to_ticker``."""
    subs: dict[str, tuple[str, int]] = {}
    owners: dict[str, list[dict]] = {}
    out = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for r in _tsv(z, "SUBMISSION.tsv"):
            try:
                cik = int(r.get("ISSUERCIK") or 0)
            except ValueError:
                continue
            filed = _date(r.get("FILING_DATE", ""))
            if cik in cik_to_ticker and (r.get("DOCUMENT_TYPE") or "").strip() in FORMS and filed:
                subs[r["ACCESSION_NUMBER"]] = (filed, cik)
        for r in _tsv(z, "REPORTINGOWNER.tsv"):
            if r.get("ACCESSION_NUMBER") in subs:
                rel = {k for k, label in RELATIONSHIPS if label in (r.get("RPTOWNER_RELATIONSHIP") or "").split(",")}
                owners.setdefault(r["ACCESSION_NUMBER"], []).append({
                    "cik": (r.get("RPTOWNERCIK") or "").strip(), "name": (r.get("RPTOWNERNAME") or "").strip(),
                    "relationship": _relationship(rel), "title": (r.get("RPTOWNER_TITLE") or "").strip() or None})
        for r in _tsv(z, "NONDERIV_TRANS.tsv"):
            accn = r.get("ACCESSION_NUMBER")
            if accn not in subs or (r.get("TRANS_CODE") or "").strip() not in CODES:
                continue
            filed, cik = subs[accn]
            try:
                f = _fact(cik_to_ticker[cik], accn, index_url(cik, accn), norm_known_at(filed),
                          _date(r.get("TRANS_DATE", "")), r["TRANS_CODE"].strip(), _num(r.get("TRANS_SHARES")),
                          _num(r.get("TRANS_PRICEPERSHARE")),
                          owners.get(accn) or [{"cik": "", "name": "", "relationship": "other", "title": None}],
                          (r.get("DIRECT_INDIRECT_OWNERSHIP") or "").strip(), fetched_at)
            except ValueError:  # one malformed row must not cost the quarter
                continue
            if f is not None:
                out.append(f)
    return _number_lots(out)


def quarters(start: str, today: str | None = None) -> list[str]:
    """'2011q1' .. the last quarter that has ended."""
    m = re.fullmatch(r"(\d{4})q([1-4])", start.lower())
    if not m:
        raise ValueError(f"quarter like 2011q1, not {start!r}")
    now = datetime.fromisoformat(today) if today else datetime.now(timezone.utc)
    y, q, out = int(m[1]), int(m[2]), []
    while (y, q) < (now.year, (now.month - 1) // 3 + 1):
        out.append(f"{y}q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def _done_path():
    return config.home() / "insiders_done.json"


def _load_done() -> dict[str, list[int]]:
    try:
        return {k: list(v) for k, v in json.loads(_done_path().read_text()).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def _save_done(done: dict) -> None:
    p = _done_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: sorted(set(v)) for k, v in sorted(done.items())}))
    tmp.replace(p)


def _add_new(bank: Bank, facts: list[Fact]) -> int:
    """Add trades, except from filings already read live (same filing, an acceptance-time known_at):
    a transaction must not count twice. A filing read in bulk before dedupes by fact id."""
    by: dict[str, list[Fact]] = {}
    for f in facts:
        by.setdefault(f.entity, []).append(f)
    n = 0
    for t, fs in by.items():
        seen: dict[str, set] = {}
        for g in bank.facts(t, ATTR, include_downgraded=True):
            seen.setdefault(g.context.get("accn"), set()).add(g.known_at)
        n += bank.add_many(f for f in fs if f.known_at in seen.get(f.context["accn"], {f.known_at}))
    return n


def _fetch_dataset(q: str, log=print):
    errors = []
    for url in DATASET_URLS:
        try:
            return fetch(url.format(q=q), user_agent=config.sec_user_agent(), ttl=365 * 86400)  # published once
        except FetchError as e:
            errors.append(str(e))
    log(f"  {q}: not available ({'; '.join(errors)})")
    return None


def ingest_bulk(bank: Bank, cik_to_ticker: dict[int, str], start: str = "2011q1", log=print) -> dict:
    """Every quarterly data set from ``start`` to the last ended quarter, for these issuers. A quarter
    already read for an issuer is remembered in var/insiders_done.json and skipped."""
    done = _load_done()
    out = {"quarters": [], "skipped": [], "unavailable": [], "facts": 0}
    for q in quarters(start):
        have = set(done.get(q, []))
        want = {int(c): t for c, t in cik_to_ticker.items() if int(c) not in have}
        if not want:
            out["skipped"].append(q)
            continue
        r = _fetch_dataset(q, log)
        if r is None:
            out["unavailable"].append(q)
            continue
        facts = parse_dataset(r.body, want, r.fetched_at)
        b = Bank(bank.root, use_index=bank.use_index)  # one quarter's worth in memory at a time
        n = _add_new(b, facts)
        done[q] = sorted(set(done.get(q, [])) | set(want))
        _save_done(done)
        out["quarters"].append(q)
        out["facts"] += n
        log(f"  {q}: {len(facts)} trades for {len({f.entity for f in facts})} issuers, {n} new")
    return out


# ---------- live: Form 4 XML for the latest filings ----------
def _text(el, path: str) -> str:
    x = el.find(path)
    return (x.text or "").strip() if x is not None and x.text else ""


def _flag(el, path: str) -> bool:
    return _text(el, path).lower() in ("1", "true")


def parse_form4(xml: bytes, ticker: str, accn: str, filed: str, url: str) -> list[Fact]:
    """Open-market trades in one Form 4 (or 5) XML document; ``filed`` is its acceptance time."""
    root = ET.fromstring(xml)
    if _text(root, "documentType") not in FORMS:
        return []
    owners = []
    for o in root.findall("reportingOwner"):
        rel = {k for k, tag in (("director", "isDirector"), ("officer", "isOfficer"),
                                ("ten_percent_owner", "isTenPercentOwner"), ("other", "isOther"))
               if _flag(o, f"reportingOwnerRelationship/{tag}")}
        owners.append({"cik": _text(o, "reportingOwnerId/rptOwnerCik"), "name": _text(o, "reportingOwnerId/rptOwnerName"),
                       "relationship": _relationship(rel),
                       "title": _text(o, "reportingOwnerRelationship/officerTitle") or None})
    owners = owners or [{"cik": "", "name": "", "relationship": "other", "title": None}]
    known = norm_known_at(filed)
    out = []
    for tr in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        amounts = "transactionAmounts/"
        f = _fact(ticker, accn, url, known, _date(_text(tr, "transactionDate/value")),
                  _text(tr, "transactionCoding/transactionCode"), _num(_text(tr, amounts + "transactionShares/value")),
                  _num(_text(tr, amounts + "transactionPricePerShare/value")), owners,
                  _text(tr, "ownershipNature/directOrIndirectOwnership/value"), now_iso())
        if f is not None:
            out.append(f)
    return _number_lots(out)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def xml_url(filing: Fact) -> str | None:
    """The raw XML behind a Form 4's primary document (EDGAR lists the rendered xslF345X0n/ view)."""
    url = filing.source.get("url", "")
    if not url.lower().endswith(".xml"):
        return None
    return re.sub(r"/xslF345X\d+/", "/", url)


def ingest_live(bank: Bank, ticker: str, days: int = 120) -> int:
    """Form 4s filed in the last ``days`` whose trades are not in the bank yet."""
    from brain.sources import sec
    t = ticker.upper()
    fil, _ = sec.filings_for(t, all_pages=False, ttl=3600)
    bank.add_many(fil)
    cik = sec.cik_for(t)
    have = {g.context.get("accn") for g in bank.facts(t, ATTR, include_downgraded=True)}
    since = (_now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    n = 0
    for f in sorted(fil, key=lambda f: f.known_at):
        accn = f.context.get("accn")
        url = xml_url(f)
        if f.value != "4" or f.known_at < since or accn in have or url is None:
            continue
        try:
            r = fetch(url, user_agent=config.sec_user_agent(), ttl=365 * 86400)  # a filed document never changes
            n += bank.add_many(parse_form4(r.body, t, accn, f.known_at, index_url(cik, accn)))
        except (FetchError, ET.ParseError) as e:
            print(f"  {t} {accn}: Form 4 unreadable ({e})", file=sys.stderr)
    return n


# ---------- reading ----------
def trades(view, ticker: str, where=None) -> list[Fact]:
    """Trades known at the view, oldest filing first; one copy per transaction even if a filing was
    read both live and in bulk (the earlier-known copy wins)."""
    seen, out = set(), []
    for f in sorted(view.facts(ticker, ATTR, where), key=lambda f: (f.known_at, f.id)):
        c = f.context
        key = (c.get("accn"), c.get("owner_cik"), f.as_of, c.get("code"), f.value, c.get("price"), c.get("ownership"),
               c.get("lot"))
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def is_officer_or_director(f: Fact) -> bool:
    rel = set(str(f.context.get("relationship") or "").split(","))
    return bool(rel & {"officer", "director"})


def buy_sell_totals(view, ticker: str, days: int = 90) -> dict:
    """USD bought and sold by insiders in filings known in the last ``days`` to the view."""
    cut = norm_known_at((datetime.fromisoformat(view.t) - timedelta(days=days)).strftime("%Y-%m-%d"))
    tot = {"buy_usd": 0.0, "sell_usd": 0.0, "buys": 0, "sells": 0}
    for f in trades(view, ticker, lambda f: f.known_at > cut):
        v = f.context.get("value_usd") or 0.0
        if f.context.get("code") == "P":
            tot["buy_usd"] += v
            tot["buys"] += 1
        else:
            tot["sell_usd"] += v
            tot["sells"] += 1
    return tot


# ---------- command line ----------
def _universe_ciks(tickers: list[str]) -> dict[int, str]:
    from brain.sources import sec
    by_ticker = {r["ticker"]: r["cik"] for r in sec.ticker_table()}
    out, missing = {}, []
    for t in tickers:
        t = t.upper()
        if t in by_ticker:
            out.setdefault(by_ticker[t], t)
        else:
            missing.append(t)
    if missing:
        print(f"not in SEC's ticker list, skipped: {', '.join(missing)}", file=sys.stderr)
    return out


def _default_tickers() -> list[str]:
    from brain.universe import BROAD
    out = []
    for p in (BROAD, config.UNIVERSE):
        try:
            out += json.loads(p.read_text())["tickers"]
        except (OSError, ValueError, KeyError):
            pass
    return list(dict.fromkeys(out))


def cmd_insiders(a) -> int:
    bank = Bank()
    tickers = [t.upper() for t in a.tickers] or _default_tickers()
    if not tickers:
        raise SystemExit("no tickers given and no universe file to take them from")
    if not a.live_only:
        res = ingest_bulk(bank, _universe_ciks(tickers), a.start, log=lambda m: print(m, flush=True))
        print(f"bulk: {len(res['quarters'])} quarters read, {len(res['skipped'])} already done, "
              f"{len(res['unavailable'])} not available; {res['facts']} trades added")
    if a.tickers:
        for t in tickers:
            print(f"{t}: {ingest_live(bank, t, a.days)} trades from recent Form 4s")
    return 0


def add_cli(sub) -> None:
    s = sub.add_parser("insiders", help="insider open-market trades (SEC Forms 4/5): bulk history, then recent Form 4s")
    s.add_argument("tickers", nargs="*", help="default: every ticker in the research universes (bulk only)")
    s.add_argument("--start", default="2011q1", help="first quarterly data set to read (default 2011q1)")
    s.add_argument("--days", type=int, default=120, help="recent Form 4s: how far back (default 120 days)")
    s.add_argument("--live-only", action="store_true", help="skip the quarterly data sets")
    s.set_defaults(fn=cmd_insiders)
