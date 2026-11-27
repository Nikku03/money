"""Reddit, read by company and by topic, never by person (docs/REDDIT.md).

    python -m brain reddit check                      # is a Reddit key saved, and does Reddit accept it?
    python -m brain reddit collect [TICKERS] [--universe us_broad] [--t week] [--pages 2]
    python -m brain reddit show TICKER [--days 30]    # a company's posts by day, topic and forum
    python -m brain reddit top [--days 7]             # the companies with the most topic mentions

Asked for on 2026-11-27 ("try tracking linkedin and twitter and reddit of company employees"). The person's own
rules keep this to companies: no account is followed, no username or post text is stored, and LinkedIn and X
(which hide posts behind a login and forbid collection) are not read at all. Reddit is read only through its
official API with the person's app key, which the session's network proxy adds to requests to www.reddit.com: the
key never appears in this code, a file or a log. Each search is Reddit's own site-wide search for a company's
common name, so the forums where a company's staff talk are included with the investing forums.

What is kept, per post that names a company: its id, the forum, its time, its score and comment count, and the
topics fixed rules found in its title and text (layoffs, a hiring freeze, a reorganisation, an outage, a recall,
a delay, a strike, closures, a leader leaving, morale). The text itself is read and dropped. A post can be opened
again from its id while it exists (https://www.reddit.com/comments/<id>); a deleted post is gone.

Nothing here is tested yet. Reddit's search reaches back only a little way, so the record runs forward from the
first collection: after about eight weeks of it, a test is declared (does a jump in a company's layoff talk come
before its restructuring filings, or its stock's moves?) and judged on the weeks after.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain import config
from brain.timeutil import now_iso

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API = "https://oauth.reddit.com"
USER_AGENT = "linux:money-brain-research:v0.1"
GAP = 0.65                 # seconds between requests: about 90 a minute, under Reddit's 100
PAGE = 100                 # posts a page, Reddit's maximum
PAGES = 2                  # pages a company at most
MIN_NAME = 4               # a one-word common name shorter than this is not searched alone

INVESTING = frozenset(s.lower() for s in (
    "wallstreetbets stocks investing StockMarket options pennystocks Daytrading SecurityAnalysis ValueInvesting "
    "dividends thetagang smallstreetbets biotechplays Superstonk wallstreetbetsELITE RobinHood stocksandtrading "
    "investing_discussion Shortsqueeze trading swingtrading FluentInFinance finance Economics "
    "SPACs CanadianInvestor UKInvestor ETFs Bogleheads").split())

TOPICS = {
    "layoffs": r"\blayoffs?\b|\blaid off\b|\blay(?:s|ing)? off\b|\bjob cuts?\b|\bcutting jobs\b|\bRIFs?\b|"
               r"\breduction in force\b|\bseverance\b|\bdownsiz",
    "hiring_freeze": r"\bhiring freeze\b|\bheadcount freeze\b|\brescind(?:ed|s)?\b|\boffer (?:was )?(?:pulled|revoked)",
    "reorg": r"\breorg\b|\breorgani[sz]|\brestructur",
    "outage": r"\boutages?\b|\bservice disruption\b|\bdown for everyone\b|\bservers? (?:are |is |were |was )?down\b|"
              r"\b(?:site|app|website|service|platform|login) (?:is |was )?(?:down|not working)\b",
    "recall": r"\brecall(?:s|ed|ing)? (?:of|on|affecting|over|for)\b|"
              r"\b(?:issues?|issued|announces?|voluntary|safety|product|vehicle|nationwide|massive) recalls?\b|"
              r"\bdefects?\b",
    "delay": r"\b(?:launch|shipment|shipping|production|release|delivery|deliveries|rollout|approval|trial|project|"
             r"plant)s? (?:is |was |are |has been |have been |gets? |got )?(?:delayed|pushed back|postponed)\b|"
             r"\bdelay(?:s|ed|ing)? (?:the |its |their )?(?:launch|release|shipments?|production|deliver(?:y|ies)|"
             r"rollout|approval|decision)\b",
    "strike": r"\bon strike\b|\bstrike (?:vote|action|ends|deal)\b|\bstriking (?:workers|employees|staff)\b|"
              r"\bwalk ?outs?\b|\bunioni[sz]|\bpicket",
    "closures": r"\bclos(?:ing|ures?|es)\b.{0,25}\b(?:stores?|locations?|plants?|offices?|factories|branches)\b|"
                r"\bstore closures?\b",
    "leader_leaving": r"\b(?:ceo|cfo|coo|cto|president|chairman)\b.{0,30}\b(?:resign|steps? down|stepping down|"
                      r"fired|ousted|leaves|leaving|depart|retir)",
    "morale": r"\breturn to office\b|\bRTO\b|\bbonus(?:es)? (?:cut|reduced|slashed)\b|\bpay cuts?\b|\bmorale\b|"
              r"\bPIPs?\b|\bstock refresh",
}
_TOPICS = {k: re.compile(v, re.IGNORECASE) for k, v in TOPICS.items()}
_ACRONYM = frozenset(("RIF", "RTO", "PIP"))       # these count only in capitals

# Tickers that are ordinary words or common abbreviations: searched and matched only with a "$" in front
COMMON = frozenset((
    "A ALL AN ARE AS AT BE BIG BY CAN CAR CASH CEO CFO DD DOW EAT FOR FUN GO HAS HE IT KEY LOW MAN ME NEW NOW ON ONE "
    "OR OUT PM POST RUN SEE SO TEAM TECH TWO UP USA WELL WORK YOU AI EV IPO ETF FDA SEC CAT DOG BOX LIFE LOVE GOOD "
    "OPEN PLAY REAL SAFE TRUE FAST HOME HOPE JOB JOBS MAIN NICE PEAK RARE ROCK SAVE SHOP SITE STAY TALK TRIP TURN "
    "WAY WISH ZONE ANY ARM AIR APP BEST BILL BLUE BOOT CARE COLD COOL CORE DATA DEEP EDGE EYE FIVE FLOW FOLD FORM "
    "FREE GAIN GOLD HEAR HIGH HUGE INFO KIND LAND LAZY LINE LOAN MIND MOVE NEXT NICE NOTE PLUS PURE RENT RIDE ROOT "
    "SELF SHE SIX SKY SOLO SPOT STEP SUN TAN TELL TEN TOP TOUR TREE VIA WAIT WAVE WEB WIRE YEAR").split())


# First words of company names that are ordinary words, places or people's names ("Analog" Devices, "Walt"
# Disney, "Delta" in options talk): the full name is searched instead. Read off the research universe's names on
# 2026-11-27; a word missing here only adds noise to a company's counts.
AMBIGUOUS = frozenset((
    "ARCH AGREE ANALOG AUTOMATIC ARTHUR ALIGN AFFILIATED ALEXANDRIA ARROWHEAD AVERY ARMSTRONG AXIS ACUITY BAXTER "
    "BOOKING BRISTOL POPULAR CHEESECAKE CASEYS COMMERCE CARNIVAL CADENCE CELSIUS CULLEN CHURCH CHOICE CINCINNATI "
    "CALUMET COHERENT COOPER CHARLES CARPENTER CIRRUS CARLISLE COGNIZANT CURTISS DOMINION DELTA DARLING DONALDSON "
    "HORTON DIODES WALT DICKS LEONARDO DOMINOS DEVON EVEREST ENCOMPASS EDISON ENSIGN ERIE ESSEX EDWARDS EXTRA FAIR "
    "COMFORT GLACIER ALPHABET GENUINE HENRY HANCOCK IRIDIUM JACOBS JAZZ HUNT KIRBY KIMBERLY KITE LAMAR LEAR LIGAND "
    "LATTICE LUMEN MICROCHIP MOODYS MOHAWK MERIT MORNINGSTAR MONOLITHIC MARSH STRATEGY VAIL MICRON NASDAQ NEWMARKET "
    "NORFOLK REALTY OWENS OPTION OCCIDENTAL PROSPERITY PIPER PACKAGING PLEXUS PHILIP PORTLAND PILGRIMS PRIMO QUANTA "
    "RYDER REGIONS REINSURANCE RYMAN RAYMOND RALPH ROLLINS ROPER ROSS REGAL SHERWIN SELECTIVE SIRIUS CHAMPION "
    "SIMPSON STEEL STATE STANLEY TENET THERMO TOLL TUTOR TAPESTRY PRICE TRAVELERS TRACTOR TAKE TYLER VALLEY "
    "WESTINGHOUSE WATERS WELLS BERKLEY WATTS WOODWARD ZEBRA GENERAL AMERICAN NATIONAL UNITED FIRST SOUTHERN "
    "WESTERN EASTERN NORTHERN GLOBAL").split())

# The names people actually use, where the filing's name is not it
ALIASES = {"GOOGL": "Google", "GOOG": "Google", "DIS": "Disney", "META": "Meta", "BRK-B": "Berkshire Hathaway",
           "JPM": "JPMorgan", "WFC": "Wells Fargo", "BAC": "Bank of America", "KO": "Coca-Cola", "HD": "Home Depot",
           "UNH": "UnitedHealth", "MMM": "3M", "FDX": "FedEx", "AAPL": "Apple", "TSLA": "Tesla", "BLK": "BlackRock",
           "ORCL": "Oracle", "PGR": "Progressive", "SO": "Southern Company", "LULU": "Lululemon",
           "SHW": "Sherwin-Williams", "TMO": "Thermo Fisher", "TOL": "Toll Brothers", "ROST": "Ross Stores",
           "WAT": "Waters Corporation", "MCO": "Moody's", "MU": "Micron", "NDAQ": "Nasdaq Inc", "DAL": "Delta Air"}

# Names counted only as written with a capital ("Meta", not "meta"; "Intel", not "intel")
CASED = frozenset(("Meta", "Intel", "Micron", "Apple", "Oracle", "Progressive"))


# ---------- names (pure) ----------
def _plain(word: str) -> bool:
    from brain import customers
    return (len(word) >= MIN_NAME and word not in customers.RISKY and word not in customers._NEVER
            and word not in COMMON and word not in AMBIGUOUS)


def common_name(title: str, firsts: dict[str, int]) -> str | None:
    """The name people use for a company: its first distinguishing word when no other listed company's name
    starts with it and it is not an ordinary word, a place or a person's name; a one-word name even when shared
    (Apple Hospitality does not stop "Apple"); else its distinguishing words together. None when nothing safe is
    left."""
    from brain import customers
    k = customers.core(title)
    if not k:
        return None
    one = k[0]
    if _plain(one) and (firsts.get(one, 0) == 1 or len(k) == 1):
        return one.title()
    if len(k) == 1:
        return None
    return " ".join(w.title() for w in k)


def names_for(tickers: list[str], table: list[dict]) -> dict[str, str | None]:
    """Each ticker's common name. A first word counts once per company (share classes share a name)."""
    from brain import customers
    owners: dict[str, set] = defaultdict(set)
    for r in table:
        k = customers.core(r["title"])
        if k:
            owners[k[0]].add(r.get("cik", r["ticker"]))
    firsts = {w: len(c) for w, c in owners.items()}
    by = {r["ticker"].upper(): r["title"] for r in table}
    return {t: ALIASES.get(t) or (common_name(by[t], firsts) if t in by else None) for t in tickers}


def safe_ticker(ticker: str) -> bool:
    return ticker not in COMMON and len(ticker) >= 3


def query_for(ticker: str, name: str | None) -> str:
    """Reddit's search words for a company: its common name in quotes, or its ticker as well when the ticker is
    not an ordinary word ("Advanced Micro Devices" OR AMD); with no safe name, the ticker alone."""
    if name:
        return f'"{name}" OR {ticker}' if safe_ticker(ticker) else f'"{name}"'
    return ticker if safe_ticker(ticker) else f'"${ticker}"'


def names_it(text: str, ticker: str, name: str | None) -> bool:
    """Does a post's title and text name the company: its common name as words, "$TICKER", or the ticker in
    capitals as a word (not for a ticker that is an ordinary word or one or two letters)?"""
    if name:
        edge = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
        if name in CASED:
            hit = re.search(edge[0] + "(?:" + re.escape(name) + "|" + re.escape(name.upper()) + ")" + edge[1], text)
        else:
            hit = re.search(edge[0] + re.escape(name) + edge[1], text, re.IGNORECASE)
        if hit:
            return True
    if re.search(r"\$" + re.escape(ticker) + r"\b", text, re.IGNORECASE):
        return True
    if not safe_ticker(ticker):
        return False
    return re.search(r"(?<![A-Za-z0-9$])" + re.escape(ticker) + r"(?![A-Za-z0-9])", text) is not None


def topics_in(text: str) -> list[str]:
    out = []
    for k, rx in _TOPICS.items():
        for m in rx.finditer(text):
            w = m.group(0).strip().rstrip("sS")
            if w.upper() in _ACRONYM and not w.isupper():
                continue                          # "rif", "pip" in lower case are other words
            out.append(k)
            break
    return out


def record_of(post: dict, ticker: str, name: str | None, fetched_at: str) -> dict | None:
    """What is kept of one search result (a Reddit "t3" listing child's data), or None when it does not name the
    company. No author and no text."""
    title, body = post.get("title") or "", post.get("selftext") or ""
    text = f"{title}\n{body}"
    if not names_it(text, ticker, name):
        return None
    sub = (post.get("subreddit") or "").lower()
    return {"id": post.get("id"), "t": ticker, "sub": sub,
            "created": datetime.fromtimestamp(float(post.get("created_utc") or 0), UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "score": int(post.get("score") or 0), "comments": int(post.get("num_comments") or 0),
            "kind": "investing" if sub in INVESTING else "other", "topics": topics_in(text),
            "fetched_at": fetched_at}


# ---------- the API ----------
class NoKey(RuntimeError):
    pass


def _basic_header() -> dict:
    """Only when the key is in this process's environment (a machine without the session's proxy). On this
    machine the proxy adds the saved key to requests to www.reddit.com, and nothing is sent from here."""
    cid, sec = os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")
    if cid and sec:
        return {"Authorization": "Basic " + base64.b64encode(f"{cid}:{sec}".encode()).decode()}
    return {}


def token(opener=urllib.request.urlopen) -> str:
    """An app-only access token (about an hour). Raises NoKey when Reddit refuses."""
    req = urllib.request.Request(TOKEN_URL, data=b"grant_type=client_credentials", method="POST",
                                 headers={"User-Agent": USER_AGENT, **_basic_header()})
    try:
        with opener(req, timeout=30) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise NoKey("Reddit refused the key (HTTP %d): none is saved for www.reddit.com, or the app is not "
                        "approved for API access" % e.code) from None
        raise
    tok = body.get("access_token")
    if not tok:
        raise NoKey(f"Reddit gave no token: {body.get('error') or body}")
    return tok


class Client:
    def __init__(self, opener=urllib.request.urlopen, sleep=time.sleep):
        self.opener, self.sleep, self._tok, self._last = opener, sleep, None, 0.0

    def get(self, path: str, params: dict) -> dict:
        if self._tok is None:
            self._tok = token(self.opener)
        wait = self._last + GAP - time.time()
        if wait > 0:
            self.sleep(wait)
        url = f"{API}{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Authorization": f"bearer {self._tok}"})
        for attempt in range(4):
            self._last = time.time()
            try:
                with self.opener(req, timeout=30) as r:
                    left = r.headers.get("X-Ratelimit-Remaining") if getattr(r, "headers", None) else None
                    reset = r.headers.get("X-Ratelimit-Reset") if getattr(r, "headers", None) else None
                    body = json.loads(r.read())
                if left is not None and float(left) < 3 and reset:
                    self.sleep(float(reset) + 1)
                return body
            except urllib.error.HTTPError as e:
                if e.code == 401 and attempt == 0:
                    self._tok = token(self.opener)
                    req.add_header("Authorization", f"bearer {self._tok}")
                    continue
                if e.code in (429, 500, 502, 503, 504):
                    self.sleep(2 ** (attempt + 2))
                    continue
                raise
        raise RuntimeError(f"Reddit kept failing on {path}")

    def search(self, q: str, t: str = "week", pages: int = PAGES) -> list[dict]:
        out, after = [], None
        for _ in range(pages):
            p = {"q": q, "sort": "new", "t": t, "limit": PAGE, "type": "link", "raw_json": 1}
            if after:
                p["after"] = after
            body = self.get("/search", p)
            data = body.get("data") or {}
            out += [c.get("data") or {} for c in data.get("children") or [] if c.get("kind") == "t3"]
            after = data.get("after")
            if not after:
                break
        return out


# ---------- the record ----------
def path() -> Path:
    p = config.home() / "reddit" / "posts.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load(p: Path | None = None) -> list[dict]:
    p = p or path()
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def collect(tickers: list[str], names: dict[str, str | None], client: Client, t: str = "week", pages: int = PAGES,
            p: Path | None = None, log=print) -> dict:
    """Search each company, keep the posts that name it, append the ones not kept before. Returns counts."""
    p = p or path()
    have = {(r["id"], r["t"]) for r in load(p)}
    new = Counter()
    seen = 0
    with p.open("a") as fh:
        for i, tk in enumerate(tickers, 1):
            name = names.get(tk)
            got = client.search(query_for(tk, name), t=t, pages=pages)
            seen += len(got)
            now = now_iso()
            for post in got:
                r = record_of(post, tk, name, now)
                if r and r["id"] and (r["id"], tk) not in have:
                    have.add((r["id"], tk))
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
                    new[tk] += 1
            if log and i % 50 == 0:
                log(f"  {i}/{len(tickers)} companies, {sum(new.values())} new posts kept")
    return {"companies": len(tickers), "results": seen, "kept": sum(new.values()), "by": dict(new)}


def window(rows: list[dict], days: int, asof: datetime | None = None) -> list[dict]:
    asof = asof or datetime.now(UTC)
    lo = (asof - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [r for r in rows if r["created"] >= lo]


def summary(rows: list[dict]) -> dict:
    by_t: dict[str, dict] = defaultdict(lambda: {"posts": 0, "topics": Counter(), "investing": 0, "other": 0})
    for r in rows:
        s = by_t[r["t"]]
        s["posts"] += 1
        s[r["kind"]] += 1
        s["topics"].update(r["topics"])
    return by_t


def show(rows: list[dict], ticker: str, days: int = 30, asof: datetime | None = None) -> list[str]:
    w = sorted((r for r in window(rows, days, asof) if r["t"] == ticker), key=lambda r: r["created"])
    if not w:
        return [f"{ticker}: no post kept in the last {days} days (python -m brain reddit collect {ticker})"]
    s = summary(w)[ticker]
    subs = Counter(r["sub"] for r in w).most_common(6)
    L = [f"{ticker} on Reddit, the last {days} days: {s['posts']} posts name it ({s['investing']} in investing "
         f"forums, {s['other']} elsewhere)",
         "  forums: " + ", ".join(f"r/{k} {v}" for k, v in subs),
         "  topics: " + (", ".join(f"{k} {v}" for k, v in s["topics"].most_common()) or "none found")]
    days_c = Counter(r["created"][:10] for r in w)
    L.append("  by day: " + ", ".join(f"{d[5:]} {n}" for d, n in sorted(days_c.items())[-14:]))
    for r in [r for r in w if r["topics"]][-8:]:
        L.append(f"  {r['created'][:16]}Z r/{r['sub']} {'/'.join(r['topics'])} (score {r['score']}, "
                 f"{r['comments']} comments) https://www.reddit.com/comments/{r['id']}")
    L.append("  Counts of public posts by company and topic; no person is followed. Untested: not a forecast "
             "(docs/REDDIT.md).")
    return L


def top(rows: list[dict], days: int = 7, n: int = 20, asof: datetime | None = None) -> list[str]:
    s = summary(window(rows, days, asof))
    if not s:
        return [f"no post kept in the last {days} days (python -m brain reddit collect)"]
    rank = sorted(s.items(), key=lambda kv: (-sum(kv[1]["topics"].values()), -kv[1]["posts"]))[:n]
    L = [f"Reddit, the last {days} days: the companies with the most topic mentions (posts that name them)"]
    for t, v in rank:
        L.append(f"  {t:<6} {v['posts']:>4} posts ({v['other']} outside investing forums); topics: "
                 + (", ".join(f"{k} {c}" for k, c in v["topics"].most_common(4)) or "none"))
    L.append("  Counts, not a forecast; nothing here is tested yet (docs/REDDIT.md).")
    return L


# ---------- the command ----------
def _universe(name: str) -> list[str]:
    return [t.upper() for t in json.loads((config.ROOT / "universe" / f"{name}.json").read_text())["tickers"]]


def _cmd(a) -> int:
    if a.action == "check":
        try:
            token()
        except NoKey as e:
            print(f"not ready: {e}. See docs/REDDIT.md for how to create the app and save its key.")
            return 1
        print("ready: Reddit accepted the key and gave an access token")
        return 0
    if a.action == "collect":
        from brain.sources import sec
        tickers = [x.upper() for x in a.tickers] or _universe(a.universe)
        names = names_for(tickers, sec.ticker_table())
        try:
            res = collect(tickers, names, Client(), t=a.t, pages=a.pages)
        except NoKey as e:
            print(f"not collected: {e}. See docs/REDDIT.md.")
            return 1
        print(f"searched {res['companies']} companies: {res['results']} results, {res['kept']} new posts kept "
              f"({path()})")
        return 0
    rows = load()
    if a.action == "show":
        for t in a.tickers:
            print("\n".join(show(rows, t.upper(), a.days)))
        return 0
    print("\n".join(top(rows, a.days)))
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("reddit", help="public Reddit posts counted by company and topic (no person is followed)")
    p.add_argument("action", choices=["check", "collect", "show", "top"])
    p.add_argument("tickers", nargs="*")
    p.add_argument("--universe", default="us_broad")
    p.add_argument("--t", default="week", choices=["hour", "day", "week", "month", "year", "all"])
    p.add_argument("--pages", type=int, default=PAGES)
    p.add_argument("--days", type=int, default=None)
    p.set_defaults(fn=lambda a: _cmd(_days(a)))


def _days(a):
    if a.days is None:
        a.days = 30 if a.action == "show" else 7
    return a
