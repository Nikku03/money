"""Who a company sells to, as its annual report says (docs/CUSTOMERS.md).

    python -m brain customers read --broad          # read the latest 10-K of each company from SEC EDGAR
    python -m brain customers read NVDA SWKS
    python -m brain customers read --broad --cached # again from the kept passages, without SEC (a new READER)
    python -m brain customers build                 # the library, with who names whom (var/customers/)
    python -m brain customers read --broad --all-years   # every annual report since 2010 (hours; resumes)
    python -m brain customers build --all-years     # the history, for point-in-time links
    python -m brain customers show SWKS AAPL        # who it sells to; who names it as a customer

Asked for on 2026-11-25: "who it trades with". A company must say in its annual report when one customer brings a
tenth or more of its revenue, and many name that customer ("Apple ... accounted for 61% of the Company's net
revenue"). Fixed rules, no model, read those sentences:

- **A customer sentence** has a customer word, a revenue word and a percentage, and is not about receivables, a
  region or a product line.
- **Its largest stated share**: the first percentage of a sentence about one named or unnamed customer (reports
  list the latest year first), or of a table row.
- **How many customers of a tenth or more**: from "one customer", "two customers", "Customer A".
- **The top five or ten**: "our five largest customers accounted for approximately 37%".
- **A named customer**: a name in the sentence that is the name of a company in SEC's ticker list (its first
  TOP_RANK entries, the larger companies), or one of a few well-known short names (ALIASES), and that stands where
  a customer stands: after "sales to" or "revenue from", before its own share ("Apple ... accounted for 61%"), in
  a row of a customer table (under the word, or two or more known names each with its percentage), or in a list
  right after the word. A name with "products", "brands", "segment",
  "acquisition" or "Ratings" after it is not a customer, nor is any name in a sentence about vendors, brands
  sold, credit ratings or total assets. The company's own name is never its customer.

Each company's reading is kept as one ``customers.major`` fact with the filing's URL and the sentences themselves,
so the card quotes the report, not the reader. The reader recognises names; it does not understand sentences: a
customer named but not listed in the United States (Samsung, a distributor) stays in the quoted sentence only, and
a listed name beside a share may be a partner or a competitor. The library's links are for reading, not evidence:
whether a customer's news reaches its supplier's stock is not tested here.
"""
from __future__ import annotations

import gzip
import html as html_lib
import json
import re
from pathlib import Path

from brain import config
from brain.bank import Bank, Fact
from brain.timeutil import now_iso

ATTR = "customers.major"
READER = 4            # 2: a name counts only where a customer stands; 3: homonyms, product and unit shares
#                       out; 4: tables of names, a star is not this year's customer, "no customer brings a tenth"
TOP_RANK = 2500            # a name may match one of the first this many companies of SEC's ticker list
MAX_SENTENCES = 8          # kept per company, the ones with a named customer first, then by share
SENTENCE_CHARS = 420
WORKERS = 3
FORMS = ("10-K", "10-K405", "10-KT")
SINCE = "2010-01-01"       # --all-years reads the annual reports filed since then
MIN_SHARE, MAX_SHARE = 0.03, 0.99
BELOW = -1.0               # pair(): a table's star, below the reporting threshold in the latest year

_SUFFIX = frozenset(("INC INCORPORATED CORP CORPORATION CO COMPANY COMPANIES LTD LIMITED PLC LLC LP NV SA AG SE "
                     "HOLDINGS HOLDING GROUP THE DE NEW ADR TRUST & AND OF").split())
_TAIL = _SUFFIX - {"THE", "AND", "&", "OF", "DE", "NEW"}      # words that may end a written name ("Stryker Corporation")
# Short names a report uses for a listed company whose registered name is longer or different.
ALIASES = {
    "AMAZON": "AMZN", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "META": "META", "FACEBOOK": "META", "WAL-MART": "WMT",
    "WALMART": "WMT", "SAMS CLUB": "WMT", "COSTCO": "COST", "LOWES": "LOW", "HOME DEPOT": "HD", "THE HOME DEPOT": "HD",
    "FORD": "F", "GM": "GM", "TOYOTA": "TM", "HONDA": "HMC", "STELLANTIS": "STLA", "TESLA": "TSLA", "BOEING": "BA",
    "LOCKHEED": "LMT", "LOCKHEED MARTIN": "LMT", "RAYTHEON": "RTX", "NORTHROP": "NOC", "GENERAL DYNAMICS": "GD",
    "CISCO": "CSCO", "DELL": "DELL", "HP": "HPQ", "HEWLETT PACKARD ENTERPRISE": "HPE", "HPE": "HPE", "IBM": "IBM",
    "MICROSOFT": "MSFT", "APPLE": "AAPL", "NVIDIA": "NVDA", "INTEL": "INTC", "AMD": "AMD", "QUALCOMM": "QCOM",
    "BROADCOM": "AVGO", "TSMC": "TSM", "TAIWAN SEMICONDUCTOR": "TSM", "MICRON": "MU", "ORACLE": "ORCL",
    "MCKESSON": "MCK", "AMERISOURCEBERGEN": "COR", "CENCORA": "COR", "CARDINAL HEALTH": "CAH", "CVS": "CVS",
    "CVS HEALTH": "CVS", "CVS CAREMARK": "CVS", "EXPRESS SCRIPTS": "CI", "CIGNA": "CI", "UNITEDHEALTH": "UNH",
    "OPTUM": "UNH", "HUMANA": "HUM", "KROGER": "KR", "TARGET": "TGT", "BEST BUY": "BBY", "SYSCO": "SYY",
    "AT&T": "T", "VERIZON": "VZ", "T-MOBILE": "TMUS", "COMCAST": "CMCSA", "DISNEY": "DIS", "NETFLIX": "NFLX",
    "CATERPILLAR": "CAT", "DEERE": "DE", "JOHN DEERE": "DE", "PACCAR": "PCAR", "CUMMINS": "CMI", "GE": "GE",
    "GENERAL ELECTRIC": "GE", "HONEYWELL": "HON", "EXXONMOBIL": "XOM", "EXXON MOBIL": "XOM", "CHEVRON": "CVX",
    "SHELL": "SHEL", "PFIZER": "PFE", "MERCK": "MRK", "NOVARTIS": "NVS", "SANOFI": "SNY", "ASTRAZENECA": "AZN",
    "GSK": "GSK", "ROCHE": "RHHBY", "MODERNA": "MRNA", "MEDTRONIC": "MDT", "ABBOTT": "ABT", "UBER": "UBER",
    "FEDEX": "FDX", "UPS": "UPS", "STARBUCKS": "SBUX", "MCDONALDS": "MCD", "PEPSICO": "PEP", "COCA-COLA": "KO",
    "DOLLAR GENERAL": "DG", "DOLLAR TREE": "DLTR", "WALGREENS": None, "SAMSUNG": None, "HUAWEI": None,
    "VOLKSWAGEN": None, "AIRBUS": None, "FOXCONN": None, "HON HAI": None, "BMW": None, "HYUNDAI": None,
    "XIAOMI": None, "LENOVO": None, "SONY": None, "BOSCH": None, "NISSAN": None, "MERCEDES-BENZ": None,
}
# Words that start a sentence or head a table and are never a customer here.
_NEVER = frozenset(("COMPANY CUSTOMER CUSTOMERS FISCAL NOTE TABLE ITEM PART FORM ANNUAL REPORT CONSOLIDATED SEGMENT "
                    "NET SALES REVENUE REVENUES TOTAL JANUARY FEBRUARY MARCH APRIL MAY JUNE JULY AUGUST SEPTEMBER "
                    "OCTOBER NOVEMBER DECEMBER ASIA CHINA EUROPE JAPAN AMERICA AMERICAS CANADA MEXICO UNITED STATES "
                    "INTERNATIONAL GLOBAL NORTH SOUTH EAST WEST DIRECT DISTRIBUTOR OTHER PREMIER CORE NATIONAL").split())

_PCT = re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s?(?:%|percent\b)")
# a sentence ends at a full stop, a semicolon or a colon before a capital; not after "Inc." and the like, and not
# inside "U.S. Government"
_SPLIT = re.compile(r"(?<!\bInc\.)(?<!\bCorp\.)(?<!\bCo\.)(?<!\bLtd\.)(?<!\bNo\.)(?<!\bvs\.)(?<=[.;:])\s+"
                    r"(?!(?<=U\.S\.\s)(?:Government|Department|Federal|Navy|Army|Air Force|Treasury|Postal)\b)"
                    r"(?=[A-Z0-9(“\"])")
_CUSTOMER = re.compile(r"\bcustomers?\b|\bdistributors?\b|\bpayors?\b", re.IGNORECASE)
_REVENUE = re.compile(r"\b(?:revenues?|net sales|sales)\b", re.IGNORECASE)
_STRONG = r"(?:accounted for|accounting for|represented|representing|comprised|constituted|contributed|made up)"
# where a customer stands
_OWN_SHARE = re.compile(r"^.{0,220}?\b" + _STRONG + r"\b.{0,60}?\d{1,2}(?:\.\d{1,2})?\s?(?:%|percent\b)",
                        re.IGNORECASE)                                           # NAME ... accounted for 61%
_SOLD_TO = re.compile(r"\b(?:sales|sold|revenues?)\b[^.;:]{0,45}?\b(?:to|from|with)\s+(?:the\s+|our\s+largest\s+"
                      r"customer,\s+)?(?:The\s+)?$", re.IGNORECASE)             # sales to NAME, revenue from NAME
_WERE_TO = re.compile(r"\b(?:were|was|came|derived|sold)\s+(?:to|from)\s+(?:the\s+)?$", re.IGNORECASE)   # 39% were to NAME
_CELL = re.compile(r"^(?:\s(?!Segment|Products|Business)[A-Z][A-Za-z]+){0,2}"
                   r"(?:,?\s?(?:Inc|Corp|Corporation|Ltd|Limited|LLC|Co|plc)\.?)*(?:\s?\([^)]{0,40}\))?\s?"
                   r"(?:\$\s?)?(?:[\d,]+(?:\.\d+)?\s+)?(\d{1,2}(?:\.\d{1,2})?\s?%|\*|—|N/A)")   # a table row
_IN_BRACKETS = re.compile(r"^\s?\(\s?\d{1,2}(?:\.\d{1,2})?\s?%\s?\)")       # T-Mobile (18%)
_LIST_WORD = re.compile(r"\b(?:customers?|distributors?|wholesalers?|tenants?|retailers?|carriers?)\b", re.IGNORECASE)
_NOT_AFTER = re.compile(r"^\W{0,3}(?:products?|brands?|acquisition|(?:reportable |operating )?segments?|Resort|World|"
                        r"business|division|unit|platform|collaboration agreement|"
                        r"Ratings|Global Select|Stock Market|Capital Market|flow|of\s+[A-Z])\b")
_NOT_NAMED = re.compile(r"\bbrands?\b|\bdealerships?\b|\bvendors?\b|\bproducts purchased\b|\bpurchased from\b|"
                        r"manufacturers/publishers|\bcredit ratings?\b|\btotal assets\b|\bcommon stock\b|"
                        r"\bnet asset value\b|\bcompetitors?\b", re.IGNORECASE)
_LIST = re.compile(r".*\b(?:customers?|distributors?|wholesalers?|tenants?|retailers?|carriers?)\b"
                   r"(?:,?\s+(?:are|were|was|is|include[sd]?|including|such as|by total revenue were))?[\s,:(]+"
                   r"(?:[A-Z][^.;:]{0,90}?(?:,|\band\b)\s+)*(?:The\s+)?$", re.DOTALL)
_BESIDE = re.compile(r"^(?:,?\s?(?:[A-Z][A-Za-z]{1,10}\.?|\([^)]{0,30}\)))?[\s,]*(?:(?:and|&)\s+)?(?:The\s+)?$")
_NONE = re.compile(r"\b[Nn]o (?:single |one |individual )?(?:end |external )?customers?\b[^.;]{0,140}?\b10\s?(?:%|percent)|"
                   r"\b[Nn]o sales to an? (?:single |individual )?customer\b[^.;]{0,140}?\b10\s?(?:%|percent)")
_INCLUDING = re.compile(r"\bincluding\s+(?:The\s+)?$", re.IGNORECASE)
# a share is stated by one of these, with the percentage soon after
_STATED = re.compile(r"\b(?:accounted for|accounting for|account for|represented|representing|comprised|comprising|"
                     r"constituted|contributed|generated|made up|were|was)\b.{0,160}?\d{1,2}(?:\.\d{1,2})?\s?"
                     r"(?:%|percent\b)|\d{1,2}(?:\.\d{1,2})?\s?%\s+\d{1,2}(?:\.\d{1,2})?\s?%", re.IGNORECASE)
_SHARE_VERB = (r"(?:accounted for|accounting for|account for|represented|representing|comprised|comprising|"
               r"constituted|contributed|generated|made up|were|was)")
_AFTER_NAME = re.compile(r"^.{0,110}?\b" + _SHARE_VERB + r"\b", re.IGNORECASE)   # the name's share follows
_AFTER_RISKY = re.compile(r"^(?:,?\s?(?:Inc|Corp|Corporation)\.?,?)?(?:\s+and its (?:affiliates|subsidiaries))?"
                          r"(?:\s?\([^)]{0,30}\))?,?\s+(?:accounted for|represented|comprised|contributed)\b",
                          re.IGNORECASE)
_VERB = re.compile(r"\b" + _SHARE_VERB + r"\b", re.IGNORECASE)
_COMPANY_WORD = re.compile(r"^\W{0,2}(?:Inc|Corp|Corporation|Stores|Company|Co|plc|Group)\b")
# One-word names that are also ordinary words: taken only with a company word after them ("Target Corporation"),
# with their share after them, or in a list beside another known name ("Walmart and Target").
RISKY = frozenset(("TARGET BLOCK GAP MATCH UNITY COMPASS BALL VISA SHELL ORACLE SNAP SQUARE DOW FOX NEWS BILL TOAST "
                   "ROOT LIFE GLOBE KEY OPEN ALLY SUN POOL META CORNING PROGRESSIVE PRINCIPAL DISCOVER DOVER "
                   "REPUBLIC GARMIN TRADE WASTE").split())
_NOT = re.compile(r"\blocated\b|outside (?:of )?the|headquartered|geograph|\bcountr|\bregions?\b|\bdomestic\b|"
                  r"\binternational customers\b|\bin (?:Asia|China|Europe|Japan|the United States|North America)\b|"
                  r"\bno (?:other )?(?:single |one |individual )?(?:end )?customer|\bnone of\b|\bdid not\b|"
                  r"\bno customers?\b|not (?:exceed|account|represent)|\bcost of (?:sales|revenues?|goods)\b|"
                  r"\bour purchases\b|market share|\bincrease in\b|\bdecrease in\b|\bgrowth\b|\bexisting customers\b|"
                  r"\bchannel\b|\bindustry\b", re.IGNORECASE)
# a share of a part of the company (a segment's sales) or of units sold, not of the company's revenue
_PART = re.compile(r"\bsales volume\b|\bsegment(?:[’']s)? (?:net |operating )?(?:sales|revenues?)\b|"
                   r"\bof the (?:net )?(?:sales|revenues?) of\b|"
                   r"[a-z]s?[’'] (?:net |operating )?(?:sales|revenues?)\b", re.IGNORECASE)
# a share of one product's sales ("of total VIVITROL gross sales"), or of a subsidiary's ("Transco's largest customer was")
_PRODUCT = re.compile(r"\bof (?:total )?(?:[A-Z]{4,}\b(?:\s+and\s+|\s+)?){1,4}(?:[A-Z]+\s+)?(?:gross |net )?(?:sales|revenues?)\b"
                      r"|\b(?!Company\b)[A-Z][A-Za-z]+[’']s largest customer (?:was|is)\b")
_OWN = re.compile(r"\b(?:the )?Company[’']s\b|\bour\b|\bits\b", re.IGNORECASE)
_RECEIVABLE = re.compile(r"receivable", re.IGNORECASE)
_CLAUSE = re.compile(r",\s+and,?\s+|;\s+|\s+and,?\s+as of\s+")
# "more than 10%", "10% or more": the reporting threshold, not a share
_THRESHOLD_BEFORE = re.compile(r"(?:more than|greater than|at least|over|in excess of|excess of|exceed(?:s|ed|ing)?|"
                               r"less than|equal to or greater than|above)\s*(?:approximately\s*)?$", re.IGNORECASE)
_THRESHOLD_AFTER = re.compile(r"^\s*(?:or more|or greater|or above|or higher)", re.IGNORECASE)
_TOPN = re.compile(r"\b(?:top|largest)\s+(five|ten|5|10)\b|\b(five|ten|5|10)\s+(?:\(\d+\)\s+)?largest\b", re.IGNORECASE)
_NUMBER = r"(?:one|two|three|four|five|six|another)"
_COUNT = re.compile(r"\b(" + _NUMBER + r")\s+(?:of (?:our|its|the Company[’']s)\s+)?"
                    r"(?:direct |indirect |end |major |significant |individual |single |other |automotive )*"
                    r"(?:customers?|distributors?)\b", re.IGNORECASE)
_COUNT_LIST = re.compile(r"\b(" + _NUMBER + r")(?:,\s*" + _NUMBER + r")*,?\s+and\s+" + _NUMBER
                         + r"\s+(?:customers?|distributors?),? respectively", re.IGNORECASE)
_SINGLE = re.compile(r"\b(?:largest|major|significant|principal) (?:end |direct )?customer\b|"
                     r"\bcustomers? (?:representing|accounting for|that (?:represented|accounted)|"
                     r"(?:represented|accounted for) (?:more than )?10)", re.IGNORECASE)
_LETTERED = re.compile(r"\b((?:Customer|Distributor)\s+[A-H])\b(?:\s*\(\d\))?\s*(?:(\d{1,2}(?:\.\d{1,2})?)\s?%)?")
_GOVERNMENT = re.compile(r"\bU\.?S\.? (?:federal )?[Gg]overnment\b|\bDepartment of (?:Defense|War)\b")
_YEAR = re.compile(r"\b(20\d\d)\b")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9&’'\-]*\.?")
_N = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "another": 1, "ten": 10, "5": 5, "10": 10}
_FOLDER = re.compile(r"/Archives/edgar/data/(\d+)/(\d+)/")
_WRAPPER = re.compile(r"(?is)<(script|style|ix:header)\b.*?</\1>")
_TAG = re.compile(r"(?s)<[^>]+>")


def path() -> Path:
    return config.ROOT / "var" / "customers" / "customers.json.gz"


# ---------- names (pure) ----------
def core(name: str) -> tuple[str, ...]:
    """A company name as its distinguishing words, upper case: "Costco Wholesale Corp /NEW" -> (COSTCO, WHOLESALE)."""
    name = name.split("/")[0].replace("’", "'").replace("'s", "s").replace("'S", "S").replace(".", "").replace(",", " ")
    words = [w for w in re.findall(r"[A-Za-z0-9&\-]+", name.upper()) if w != "&"]
    while words and words[-1] in _SUFFIX:
        words.pop()
    while words and words[0] == "THE":
        words.pop(0)
    return tuple(words)


def name_table(tickers: list[dict], top: int = TOP_RANK) -> dict[tuple[str, ...], str | None]:
    """Names a report may use -> ticker: the full distinguishing name of each of the ``top`` first companies of
    SEC's list (the first to claim a name keeps it; a one-word name only when no other company of the whole list
    starts with that word), then ALIASES (None: named, not listed in the United States)."""
    firsts: dict[str, set] = {}                # a first word -> the companies whose name starts with it
    for r in tickers:
        k = core(r["title"])
        if k:
            firsts.setdefault(k[0], set()).add(r.get("cik", r["ticker"]))
    out: dict[tuple[str, ...], str | None] = {}
    for r in tickers[:top]:
        k = core(r["title"])
        if not k or k in out:
            continue
        if len(k) == 1 and (len(k[0]) < 3 or k[0] in _NEVER or len(firsts[k[0]]) > 1):
            continue                           # "Frontier" is the airline and the telephone company: no one's name
        out[k] = r["ticker"].upper()
    for k, v in ALIASES.items():
        out[tuple(k.split())] = v
    return out


_JOIN = frozenset(("of", "and", "&", "de"))


def names_in(sentence: str, table: dict, own: str | None = None) -> list[tuple[str, str | None, int, int]]:
    """(name as written, ticker or None, start, end) of each known company named in ``sentence``, in order. The
    longest name at a place wins. A one-word name is not taken as the sentence's first word (a capital says nothing
    there), nor in capitals of four letters or fewer unless it is in ALIASES ("PTC" is a tax credit), nor when it is
    the start of a longer name ("National Bank of Canada")."""
    toks = [(m.group(0), m.start()) for m in _WORD.finditer(sentence)]
    longest = max((len(k) for k in table), default=1)
    out, i = [], 0
    while i < len(toks):
        w, at = toks[i]
        if not w[0].isupper():
            i += 1
            continue
        hit = None
        for n in range(min(longest, len(toks) - i), 0, -1):
            span = toks[i:i + n]
            if not (span[-1][0][0].isupper() and all(x[0][0].isupper() or x[0] in _JOIN for x in span)):
                continue
            k = core(" ".join(x[0] for x in span))
            if len(k) != n and not (k and len(k) < n and all(x[0].upper().strip(".,") in _TAIL for x in span[len(k):])):
                continue
            if k not in table or (len(k) == 1 and k[0] in _NEVER):
                continue
            end = span[-1][1] + len(span[-1][0])
            if len(k) == 1 and n == 1:
                alias = k[0] in ALIASES
                bare = not _COMPANY_WORD.match(sentence[end:end + 16])
                if len(w) <= 3 and not w.isupper():                         # "DoW" is not Dow, "Ge" is not GE
                    continue
                if w.isupper() and len(w) <= 4 and not alias and bare:      # "PTC", "AES", "ATS": initials
                    continue
                if i == 0 and not alias:
                    continue
                if k[0] in RISKY and bare:
                    beside = bool(out) and 0 <= at - out[-1][3] <= 7
                    if not (beside or _AFTER_RISKY.match(sentence[end:end + 90])):
                        continue
            if re.match(r"\s+of\s+[A-Z]", sentence[end:end + 8]):          # the start of a longer name
                continue
            hit = (" ".join(x[0] for x in span).rstrip("."), table[k], at, end, n)
            break
        if hit and (own is None or hit[1] != own.upper()):
            out.append(hit[:4])
        i += hit[4] if hit else 1
    seen, uniq = set(), []
    for nm, tk, at, end in out:
        key = tk or nm.upper()
        if key not in seen:
            seen.add(key)
            uniq.append((nm, tk, at, end))
    return uniq


def customers_in(sentence: str, table: dict, own: str | None = None) -> list[tuple[str, str | None, int, int]]:
    """The known companies named in ``sentence`` that stand where a customer stands (the module's docstring)."""
    if _NOT_NAMED.search(sentence) or re.search(r"\bsales of [A-Z][\w&.,’' ]{0,40}?products\b", sentence):
        return []
    colon = sentence.rfind(": ")
    under_list = colon > 0 and bool(_LIST_WORD.search(sentence[:colon]))
    out: list[tuple] = []
    found = names_in(sentence, table, own)
    rows = sum(bool(_CELL.match(sentence[end:])) for _, _, _, end in found)
    for nm, tk, at, end in found:
        after, before = sentence[end:], sentence[max(0, at - 70):at]
        if _NOT_AFTER.match(after):
            continue
        row = bool(_CELL.match(after)) or bool(_IN_BRACKETS.match(after))
        beside = bool(out) and bool(_BESIDE.match(sentence[out[-1][3]:at]))     # ", " or " and " after a customer
        listed = under_list and at > colon and not re.search(r"\b(?!and\b)[a-z]{4,}\b", sentence[colon:at])
        near = any(0 <= at - x.end() <= 130 for x in _LIST_WORD.finditer(sentence[:at]))
        known_well = len(core(nm)) > 1 or core(nm)[0] in ALIASES
        ok = (bool(_OWN_SHARE.match(after)) or bool(_SOLD_TO.search(before)) or bool(_WERE_TO.search(before))
              or beside or (row and (under_list or near or bool(out) or rows >= 2)) or (listed and known_well)
              or (near and known_well and bool(_LIST.match(before))))
        if ok:
            out.append((nm, tk, at, end))
    return out


# ---------- reading a report (pure) ----------
def flatten(html: str) -> str:
    return " ".join(html_lib.unescape(_TAG.sub(" ", _WRAPPER.sub(" ", html))).split())


def revenue_part(sentence: str) -> str:
    """The sentence without its clauses about receivables ("... and, as of April 30, these two customers accounted
    for 30% of our receivables")."""
    if not _RECEIVABLE.search(sentence):
        return sentence
    return " ".join(c for c in _CLAUSE.split(sentence) if not _RECEIVABLE.search(c))


def shares_at(sentence: str) -> list[tuple[float, int]]:
    """(share, where) of each share a sentence states, in order: its percentages without the reporting threshold."""
    out = []
    for m in _PCT.finditer(sentence):
        v = round(float(m.group(1)) / 100, 4)
        if not (MIN_SHARE <= v <= MAX_SHARE):
            continue
        if (_THRESHOLD_BEFORE.search(sentence[max(0, m.start() - 40):m.start()])
                or _THRESHOLD_AFTER.match(sentence[m.end():m.end() + 16])):
            continue
        out.append((v, m.start()))
    return out


def shares(sentence: str) -> list[float]:
    return [v for v, _ in shares_at(sentence)]


def candidates(text: str) -> list[str]:
    """The passages of a report's text that could be customer sentences: a percentage, a revenue word and a stated
    share, without their clauses about receivables. A passage that ends in a colon takes the table row after it.
    These are what a reading is made from, and what is cached beside it (``chunks_path``)."""
    chunks = [c.strip() for c in _SPLIT.split(text)]
    out = []
    for i, s in enumerate(chunks):
        if s.endswith(":") and i + 1 < len(chunks):
            s = s + " " + chunks[i + 1][:400]
        s = revenue_part(s)
        if 30 <= len(s) <= 1300 and _PCT.search(s) and _REVENUE.search(s) and _STATED.search(s):
            out.append(s)
    return out


def sentences(text, table: dict | None = None, own: str | None = None) -> list[str]:
    """The customer sentences of a report (its text, or its ``candidates``), each once: a share of the company's
    revenue stated for a customer (the word, or a known company standing where a customer stands)."""
    out, seen = [], set()
    for s in (candidates(text) if isinstance(text, str) else text):
        gov = bool(_GOVERNMENT.search(s))
        if (_NOT.search(s) and not gov) or _PART.search(_OWN.sub("the", s)) or _PRODUCT.search(s):
            continue
        if not (_CUSTOMER.search(s) or gov or (table and customers_in(s, table, own))):
            continue
        k = re.sub(r"\W+", "", s.lower())[-90:]
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def how_many(sentence: str) -> int:
    """How many customers of a tenth or more the sentence speaks of, for its latest year."""
    if (m := _COUNT_LIST.search(sentence)):
        return _N[m.group(1).lower()]
    found = list(_COUNT.finditer(sentence))
    if not found:
        return 1 if _SINGLE.search(sentence) else 0
    stated = set(at for _, at in shares_at(sentence))
    own_share = []                 # counts whose own share follows: "one customer represented 22% and another 14%"
    for m in found:
        after = sentence[m.end():m.end() + 60]
        v = _VERB.search(after)
        p = _PCT.search(after, v.end()) if v else None
        if p and (m.end() + p.start()) in stated:
            own_share.append(_N[m.group(1).lower()])
    if len(own_share) >= 2 and len(set(_YEAR.findall(sentence))) < 2:
        return sum(own_share)
    return _N[found[0].group(1).lower()]


def years_of(sentence: str) -> list[int]:
    return list(dict.fromkeys(int(y) for y in _YEAR.findall(sentence)))


def pick(sentence: str, seg: list[float]) -> float | None:
    """Of the percentages stated for one customer, the latest year's; None when they are a list this reader cannot
    line up ("A and B represented 53% and 29%, respectively", with A unknown)."""
    if not seg:
        return None
    years = years_of(sentence)
    if len(years) >= 2 and len(seg) == len(years):
        return seg[-1] if years == sorted(years) else seg[0]
    if len(seg) == 1:
        return seg[0]
    return None if "respectively" in sentence.lower() else seg[0]


def pair(sentence: str, got: list[tuple], stated: list[tuple[float, int]]) -> list[float | None]:
    """The share of each named customer of a sentence, in the order of ``got``; None where it cannot be told.

    - "39% ... were to Airbus": the percentages before the name are its own.
    - "Apple accounted for 61%, 67% and 69%": the percentages after the name, up to the next name, are its own.
    - "A and B represented 30% and 12%, respectively": names that run together take the percentages after them in
      order, when there are as many (and the sentence says "respectively", or it is a table row).
    - "four customers, including PACCAR, were 31%": a share of a group is no one's own.
    - A table cell that is a star or a dash: BELOW, not a customer of a tenth in the latest year."""
    out: list[float | None] = [None] * len(got)
    if not got or not stated:
        return out
    starts = [g[2] for g in got]
    bounds = starts + [len(sentence)]
    respectively = "respectively" in sentence.lower()
    pending: list[int] = []
    for i, (nm, tk, at, end) in enumerate(got):
        cell = _CELL.match(sentence[end:])
        if cell and not cell.group(1)[0].isdigit():
            out[i] = BELOW                                       # "*": below a tenth in the latest year
            continue
        if _INCLUDING.search(sentence[max(0, at - 14):at]):
            continue
        if _WERE_TO.search(sentence[max(0, at - 30):at]):
            lo = got[i - 1][3] if i else 0
            seg = [v for v, p in stated if lo <= p < at]
            out[i] = pick(sentence, seg) if len(seg) == len(years_of(sentence)) >= 2 else seg[-1] if seg else None
            pending = []
            continue
        seg = [v for v, p in stated if end <= p < bounds[i + 1]]
        if not seg:
            pending.append(i)
            continue
        group = pending + [i]
        pending = []
        if len(group) == 1:
            out[i] = pick(sentence, seg)
            continue
        first = min(p for _, p in stated if p >= end)
        table_row = not _VERB.search(sentence[end:first])
        if len(seg) == len(group) and (respectively or table_row):
            for k, v in zip(group, seg):
                out[k] = v
        elif table_row and len(seg) % len(group) == 0:           # a table: one row a year, the latest first
            for k, v in zip(group, seg):
                out[k] = v
    return out


def read(text, table: dict, own: str | None = None) -> dict:
    """What the report (its text, or its ``candidates``) says of its customers: {top, n_major, top5, top10,
    named: [{name, ticker, share}], government, none, sentences}. A value the report does not state is None.
    ``none`` is the report's own sentence that no customer brings a tenth of its revenue, when it says so and
    nothing else was read."""
    named: dict[str, dict] = {}
    top, n_major, top5, top10, government = None, None, None, None, None
    kept = []
    cands = candidates(text) if isinstance(text, str) else text
    none = next((s[m.start():][:SENTENCE_CHARS] for s in cands if (m := _NONE.search(s))), None)
    for s in sentences(cands, table, own):
        stated = shares_at(s)
        v = [x for x, _ in stated]
        got = customers_in(s, table, own)
        tn = _TOPN.search(s)
        if tn:
            n = _N[(tn.group(1) or tn.group(2)).lower()]
            if v and n == 5 and top5 is None:
                top5 = v[0]
            elif v and n == 10 and top10 is None:
                top10 = v[0]
            if v:
                kept.append((0, v[0], s))
            continue
        gov = bool(_GOVERNMENT.search(s))
        if gov and government is None and v:
            government = v[0]
        each = pair(s, got, stated)
        got = [g for g, sh in zip(got, each) if sh != BELOW]     # a customer of an earlier year is not named
        each = [sh for sh in each if sh != BELOW]
        for (nm, tk, _, _), sh in zip(got, each):
            key = tk or nm.upper()
            if key not in named or (named[key]["share"] is None and sh is not None):
                named[key] = {"name": nm, "ticker": tk, "share": sh}
        shown = {m.group(1): round(float(m.group(2)) / 100, 4) for m in _LETTERED.finditer(s) if m.group(2)}
        count = how_many(s)
        n = max(count, len(shown), len(got), 1 if gov else 0)
        if not n:
            continue
        if n_major is None:
            n_major = n
        if top is None:
            paired = [x for x in each if x is not None]
            top = (max(shown.values()) if shown else max(paired) if paired else None if got
                   else pick(s, v) if count == 1 else v[0] if v else None)
        kept.append((2 if got else 1, v[0] if v else 0.0, s))
    kept.sort(key=lambda x: (-x[0], -x[1]))
    return {"top": top, "n_major": n_major, "top5": top5, "top10": top10, "government": government,
            "named": sorted(named.values(), key=lambda r: -(r["share"] or 0)),
            "none": none if (top is None and n_major is None and not named) else None,
            "sentences": [s[-SENTENCE_CHARS:] for _, _, s in kept[:MAX_SENTENCES]]}


# ---------- from SEC EDGAR to the bank ----------
def reports(view, ticker: str, since: str = SINCE) -> list[Fact]:
    """The annual reports of ``ticker`` filed since ``since``, one a filing, oldest first."""
    by: dict[str, Fact] = {}
    for f in view.facts(ticker.upper(), "filing"):
        if str(f.value) in FORMS and f.known_at >= since:
            by.setdefault(str(f.context.get("accn")), f)
    return sorted(by.values(), key=lambda f: f.known_at)


def latest_report(view, ticker: str) -> Fact | None:
    fs = reports(view, ticker, "")
    return fs[-1] if fs else None


def settled(view, ticker: str, filing: Fact) -> bool:
    return any((f.context or {}).get("accn") == filing.context.get("accn") and (f.context or {}).get("reader") == READER
               for f in view.facts(ticker.upper(), ATTR))


def chunks_path(filing: Fact) -> Path:
    """Where a report's candidate passages are kept, so that a later reader can read it again without SEC."""
    accn = str(filing.context.get("accn") or "none").replace("/", "_")
    return config.ROOT / "var" / "customers" / "chunks" / f"{filing.entity}_{accn}.json.gz"


def read_report(filing: Fact, table: dict, fetched_at: str | None = None, cached: bool = False) -> Fact:
    """The customers fact of one annual report: one SEC request, or none with ``cached`` (its kept passages)."""
    url = filing.source.get("url", "")
    p = chunks_path(filing)
    if cached:
        with gzip.open(p, "rt") as f:
            cands = json.load(f)
    else:
        from brain.sources.releases import _get
        if not _FOLDER.search(url):
            raise ValueError(f"{filing.ref}: no EDGAR document in its URL")
        cands = candidates(flatten(_get(url).decode("utf-8", "replace")))
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt") as f:
            json.dump(cands, f)
    got = read(cands, table, own=filing.entity)
    return Fact(entity=filing.entity, attr=ATTR, value=got["top"] if got["top"] is not None else 0.0, unit="ratio",
                as_of=filing.as_of, known_at=filing.known_at, tier="filed",
                source={"name": f"SEC EDGAR {filing.value}", "url": url},
                context={"accn": filing.context.get("accn"), "filing": filing.ref, "reader": READER, **got},
                retrieved_at=fetched_at or now_iso())


def collect(bank: Bank, tickers, log=print, workers: int = WORKERS, table: dict | None = None,
            cached: bool = False, again: bool = False, every: bool = False) -> dict:
    """Read the latest annual report of each ticker whose reading the bank does not hold yet (at this READER).
    ``cached``: from the passages kept at an earlier reading, without SEC; a report never fetched is left.
    ``again``: also the reports already read at this READER (a reading that says the same is not kept twice).
    ``every``: each ticker's annual reports since SINCE, not only the latest (resumes where it stopped)."""
    from concurrent.futures import ThreadPoolExecutor
    if table is None:
        from brain.sources import sec
        table = name_table(sec.ticker_table())
    view = bank.view()
    todo = []
    for t in [x.upper() for x in tickers]:
        for f in (reports(view, t) if every else [latest_report(view, t)]):
            if f is not None and (again or not settled(view, t, f)) and (not cached or chunks_path(f).exists()):
                todo.append(f)

    def one(f):
        try:
            return read_report(f, table, cached=cached), None
        except Exception as e:  # noqa: BLE001 - one report's trouble is recorded, the rest still run
            return None, {"filing": f.ref, "why": f"{type(e).__name__}: {str(e)[:120]}"}
    kept, failed = 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, (r, why) in enumerate(pool.map(one, todo)):
            if r is None:
                failed.append(why)
                continue
            kept += bank.add(r) or 0
            for f in view.facts(r.entity, ATTR):          # one live reading a report: an older reader's is kept,
                if f.context.get("accn") == r.context.get("accn") and f.id != r.id:            # downgraded
                    bank.downgrade(r.entity, f.id, f"superseded by {r.id}: read again by reader {READER}")
            c = r.context
            if log:
                log(f"{r.entity} {r.known_at[:10]}: " + (", ".join(x["name"] + (f" {x['share'] * 100:.0f}%" if x["share"] else "")
                                                 for x in c["named"]) or "no known customer named")
                    + (f"; largest {c['top'] * 100:.0f}%" if c["top"] else ""))
            if log and (k + 1) % 100 == 0:
                log(f"  {k + 1}/{len(todo)} reports")
    return {"read": len(todo) - len(failed), "kept": kept, "failed": failed}


# ---------- the library ----------
def history_path() -> Path:
    return config.ROOT / "var" / "customers" / "history.json.gz"


def build(bank: Bank, tickers, history: bool = False) -> list[dict]:
    """One row a company with a reading: its latest, with the filing's date and URL. ``history``: one row a
    report read, oldest first (for point-in-time work: a row is known from ``known_at``)."""
    view = bank.view()
    out = []
    for t in sorted({x.upper() for x in tickers}):
        fs = sorted((f for f in view.facts(t, ATTR) if (f.context or {}).get("reader") == READER),
                    key=lambda f: f.known_at)
        for f in (fs if history else fs[-1:]):
            c = f.context
            out.append({"t": t, "filed": f.known_at[:10], "known_at": f.known_at, "year_end": f.as_of,
                        "url": f.source.get("url"),
                        **{k: c.get(k) for k in ("top", "n_major", "top5", "top10", "government", "named", "none",
                                                 "sentences")}})
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


LINK_DAYS = 456            # a link holds until the company's next report, or this long after its last


def links(history: list[dict]) -> list[dict]:
    """Point-in-time links from the history: {s (the supplier), c (its named, listed customer), share, known_at,
    until}: known from the moment the report was filed until the supplier's next report read (or LINK_DAYS)."""
    from datetime import date, timedelta
    by: dict[str, list[dict]] = {}
    for r in history:
        by.setdefault(r["t"], []).append(r)
    out = []
    for t, rows in by.items():
        rows.sort(key=lambda r: r["known_at"])
        for i, r in enumerate(rows):
            cap = (date.fromisoformat(r["filed"]) + timedelta(days=LINK_DAYS)).isoformat()
            until = min(rows[i + 1]["known_at"], cap) if i + 1 < len(rows) else cap
            for x in r.get("named") or []:
                if x.get("ticker") and x["ticker"] != t:
                    out.append({"s": t, "c": x["ticker"], "share": x.get("share"), "known_at": r["known_at"],
                                "until": until})
    return out


def named_by(lib: list[dict], ticker: str) -> list[dict]:
    """The companies whose report names ``ticker`` as a customer, largest share first: {t, share, filed}."""
    t = ticker.upper()
    out = [{"t": r["t"], "share": x.get("share"), "filed": r["filed"]}
           for r in lib for x in (r.get("named") or []) if x.get("ticker") == t and r["t"] != t]
    return sorted(out, key=lambda r: -(r["share"] or 0))


def pct(x: float | None) -> str:
    return "a share it does not state" if x is None else f"{x * 100:.0f}%"


def render(ticker: str, lib: list[dict]) -> list[str]:
    t = ticker.upper()
    row = next((r for r in lib if r["t"] == t), None)
    L = [f"{t}: who it sells to, as its annual report says"]
    if row is None:
        L.append(f"  its annual report has not been read (python -m brain customers read {t}; then customers build)")
    else:
        L[0] += f" (filed {row['filed']}, fiscal year to {row['year_end']})"
        named = row.get("named") or []
        if named:
            L.append("  named: " + "; ".join(
                f"{x['name']}" + (f" ({x['ticker']})" if x.get("ticker") else " (not listed in the United States)")
                + f" {pct(x.get('share'))} of revenue" for x in named))
        bits = []
        if row.get("top") is not None:
            n = row.get("n_major")
            bits.append(f"its largest stated customer share is {pct(row['top'])}"
                        + (f"; {n} customer{'s' if n != 1 else ''} of a tenth or more" if n else ""))
        if row.get("top5") is not None:
            bits.append(f"its five largest customers bring {pct(row['top5'])}")
        if row.get("top10") is not None:
            bits.append(f"its ten largest {pct(row['top10'])}")
        if row.get("government") is not None:
            bits.append(f"the U.S. government {pct(row['government'])}")
        if bits:
            L.append("  " + "; ".join(bits))
        if row.get("none"):
            L.append("  its report says no customer brings a tenth of its revenue:")
            L.append(f"    \"{row['none']}\"")
        elif not (named or bits):
            L.append("  the reader found no customer of a tenth or more of revenue in its report")
        for s in (row.get("sentences") or [])[:3]:
            L.append(f"    \"{s}\"")
        L.append(f"  source: {row['url']}")
    by = named_by(lib, t)
    if by:
        L.append("  named as a customer by: " + "; ".join(f"{r['t']} ({pct(r['share'])} of its revenue)" for r in by[:12])
                 + (f"; and {len(by) - 12} more" if len(by) > 12 else ""))
    L.append("  Read by fixed rules from the report: check a figure against the quoted sentence before using it. A "
             "name beside a share may be a partner, not a customer. Whether a customer's news reaches its supplier's "
             "stock is not tested (docs/CUSTOMERS.md).")
    return L


# ---------- command line ----------
def _cmd(a) -> int:
    bank = Bank()
    if a.action == "show":
        lib = load()
        for t in a.tickers:
            print("\n".join(render(t, lib)))
            print()
        return 0
    from brain import episodes, universe
    if a.tickers:
        tickers = [x.upper() for x in a.tickers]
    elif a.band or a.broad:
        tickers = json.loads((universe.BANDS[a.band][2] if a.band else universe.BROAD).read_text())["tickers"]
    else:
        tickers = list(episodes.universe_tickers())
    if a.action == "read":
        res = collect(bank, tickers, cached=a.cached, again=a.again, every=a.all_years)
        print(f"{res['read']} annual reports read, {len(res['failed'])} failed")
        for w in res["failed"][:10]:
            print(f"  failed: {w['filing']}: {w['why']}")
        return 0
    if a.all_years:
        rows = build(bank, tickers, history=True)
        p = save(rows, history_path())
        print(f"{len(rows)} annual reports of {len({r['t'] for r in rows})} companies written to {p}; "
              f"{len(links(rows))} point-in-time links to listed customers")
        return 0
    rows = build(bank, tickers)
    p = save(rows)
    named = sum(bool(r.get("named")) for r in rows)
    print(f"{len(rows)} companies written to {p}; {named} name a known customer")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("customers", help="who a company sells to, as its annual report says (docs/CUSTOMERS.md)")
    p.add_argument("action", choices=["read", "build", "show"])
    p.add_argument("tickers", nargs="*")
    p.add_argument("--broad", action="store_true", help="the research universe (universe/us_broad.json)")
    p.add_argument("--band", choices=["small", "micro"], help="universe/us_small.json or us_micro.json")
    p.add_argument("--cached", action="store_true",
                   help="read again from the passages kept at an earlier reading, without SEC (after a new READER)")
    p.add_argument("--again", action="store_true", help="also the reports already read by this reader")
    p.add_argument("--all-years", action="store_true",
                   help=f"read: every annual report since {SINCE}; build: the history (var/customers/history.json.gz)")
    p.set_defaults(fn=_cmd)
