"""Attention: what deserves a closer look at a check, and the digging done on it (docs/ATTENTION.md).

The desk's case file treats its movers alike, furthest outside their ranges first. A good analyst does not: they look
hardest where the surprise is largest, where nothing explains the move, where the news is unusual, where it reaches
many other stocks, where a stock went against its neighbours, and where it touches what they were already watching.
This module scores every candidate on those parts, turns the scores into attention weights (a softmax, so the weights
sum to 100% and the top few take most of them, as attention does), and for the top few digs further:

- **their filings since the close**, read deeper than the case file's opening, with the sentences that carry numbers
  or guidance pulled out;
- **their stories**, read in full through the engine's web reader (robots.txt, rate limits; a refused page is left);
- **the brain's memory of them** (``brain.memory.recall``): what was said or found about them before;
- **the pond**: the wave they felt, how they moved against it, and who their move reaches.

The parts and their weights are fixed by reasoning, not fitted: no test here says which items turn out to matter.
Each check's focus is recorded (``research/attention/``) so that can be measured later. Attention says where to look;
it is not a forecast, and what it reads is for the note to reason about and cite.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from brain import config, connect, ripple
from brain.timeutil import now_iso

TOP = 6
REACH_CORR = 0.45               # a link this strong counts for reach (a partner, RING, is 0.30)
THIN_BARS = 2                   # a move resting on this many distinct prices or fewer is thin
THIN_DISCOUNT = 0.5             # a thin move's score is multiplied by this
TEMP = 0.12                     # softmax temperature: lower puts more of the attention on the top items
WEIGHTS = {"surprise": 0.25, "unexplained": 0.20, "news": 0.15, "reach": 0.10, "against": 0.10, "memory": 0.10,
           "fragile": 0.10}
SERIOUS = {"4.02": "accounts no longer to be relied on", "1.03": "bankruptcy", "2.06": "an impairment",
           "3.01": "a listing notice", "4.01": "a change of auditor", "2.05": "a restructuring", "1.01": "a material deal",
           "2.01": "an acquisition or sale", "5.02": "an officer or director change", "2.04": "a debt default trigger"}
FILING_CHARS = 6000
PAGE_CHARS = 4000
KEY_SENTENCES = 6
_KEY = re.compile(r"(\$\s?\d|\d+(\.\d+)?\s?%|guidance|outlook|expects?|forecast|lower|raise|cut|delay|default|"
                  r"impair|restat|resign|terminat|acquir|merger|approval|FDA|investigat|lawsuit|settle)", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_TICK = re.compile(r"\b[A-Z]{1,5}\b")


def focus_dir() -> Path:
    return config.ROOT / "research" / "attention"


# ---------- the parts ----------
def surprise(move: float | None, range90: list | None, z: float | None) -> float:
    """How far past its range the move went: 0 inside a half of the 9-in-10 range, 1 at twice its edge or beyond.
    Without a range, the own move in usual days (3 usual days scores 1)."""
    if move is not None and range90:
        edge = range90[1] if move > 0 else range90[0]
        if edge:
            return float(np.clip((abs(move) / abs(edge) - 0.5) / 1.5, 0, 1))
    if z is not None and np.isfinite(z):
        return float(np.clip(abs(z) / 3.0, 0, 1))
    return 0.0


def unexplained(move: float | None, left: float | None, why: str | None) -> float:
    """The share of the move left after the market and the drivers, more when no news of its own was found."""
    if move is None or left is None or not move:
        return 0.0
    u = float(np.clip(abs(left) / abs(move), 0, 1))
    return float(min(1.0, u + (0.25 if why and "no news found" in why else 0.0)))


def news_score(filed: list[str], named_stories: int) -> tuple[float, list[str]]:
    """Filings since the close (serious 8-K items score most) and stories that name the stock."""
    reasons, s = [], 0.0
    for f in filed:
        items = re.findall(r"\b\d\.\d\d\b", f)
        bad = [SERIOUS[i] for i in items if i in SERIOUS]
        if bad:
            s = max(s, 1.0)
            reasons.append("filed " + ", ".join(bad))
        elif "results" in f or "2.02" in items:
            s = max(s, 0.7)
            reasons.append("filed its results")
        else:
            s = max(s, 0.5)
            reasons.append("filed " + f)
    if named_stories:
        s = min(1.0, s + 0.3)
        reasons.append(f"{named_stories} {'story names' if named_stories == 1 else 'stories name'} it")
    return s, reasons


def reach_score(degree: int) -> float:
    """How many stocks moved closely with it over the year (a link of REACH_CORR or more), on a log scale."""
    return float(min(1.0, np.log1p(degree) / np.log1p(40)))


def fragile_score(m: dict) -> tuple[float, list[str]]:
    """Its mosaic chances well above the usual stock's, or little cash against its losses."""
    s, why = 0.0, []
    for v in ((m.get("mosaic") or {}).get("pictures") or {}).values():
        p, u = v.get("chance"), v.get("usual")
        if p and u and p / u >= 3:
            s = max(s, float(min(1.0, np.log(p / u) / np.log(20))))
            why.append(f"chance of {v.get('name', 'a filing')} {p * 100:.1f}% against {u * 100:.1f}% usually")
    q = (m.get("numbers") or {}).get("cash_covers_quarters")
    if q is not None and q < 4:
        s = max(s, 1.0 if q < 2 else 0.6)
        why.append(f"cash covers about {q:.1f} quarters of losses")
    return float(s), why


def watched(case: dict) -> dict[str, str]:
    """Stocks the brain was already watching: the last note's themes and open questions, open desk calls and open
    calls to the person."""
    out: dict[str, str] = {}
    tickers = set(case.get("moves") or {})
    mem = case.get("memory") or {}
    for th in mem.get("themes") or []:
        for t in th.get("members") or []:
            out.setdefault(t, f"in the last note's theme {th['name']!r}")
    for q in mem.get("threads") or []:
        for t in set(_TICK.findall(q)) & tickers:
            out.setdefault(t, "in the last note's open questions")
    for path, what in ((config.ROOT / "research" / "desk_calls.jsonl", "an open desk call"),
                       (config.ROOT / "research" / "predictions.jsonl", "an open call to the person")):
        if path.exists():
            for ln in path.read_text().splitlines():
                try:
                    c = json.loads(ln)
                except ValueError:
                    continue
                t = c.get("ticker")
                if t and c.get("by", "") >= case["asof"]:
                    out.setdefault(t, what)
    return out


# ---------- scoring ----------
def candidates(case: dict, pond: dict | None) -> list[dict]:
    """Every stock worth scoring: the case file's movers, the stocks that moved against their waves, and the stocks
    the brain was watching, each with its parts and the reasons behind them."""
    movers = {m["t"]: m for m in case["movers"]}
    named = {}
    for s in case.get("stories") or []:
        for t in s.get("names") or []:
            named[t] = named.get(t, 0) + 1
    seen = watched(case)
    z = {} if pond is None else dict(zip(pond["tickers"], pond["z"]))
    Hh = {} if pond is None else dict(zip(pond["tickers"], pond["H"]))
    stone = {} if pond is None else dict(zip(pond["tickers"], pond["stone"]))
    deg = {} if pond is None else dict(zip(pond["tickers"], (np.abs(pond["pond"]["C"]) >= REACH_CORR).sum(1)))
    against = set() if pond is None else {
        t for t in pond["tickers"] if np.isfinite(z.get(t, np.nan)) and
        ripple.how_it_moved(float(Hh[t]), float(z[t]), bool(stone[t])) == "against its wave"}
    moves = case.get("moves") or {}
    out = []
    for t in sorted(set(movers) | against | (set(seen) & set(moves))):
        m = movers.get(t, {})
        zt = float(z[t]) if t in z and np.isfinite(z[t]) else None
        parts, reasons = {}, []
        parts["surprise"] = surprise(m.get("move", moves.get(t)), m.get("range90"), zt)
        if m.get("outside90"):
            reasons.append("outside its 9-in-10 range")
        parts["unexplained"] = unexplained(m.get("move"), m.get("left"), m.get("why"))
        if parts["unexplained"] >= 0.6:
            reasons.append("most of the move is its own" + (", with no news found" if "no news" in (m.get("why") or "") else ""))
        parts["news"], why_news = news_score(m.get("filed") or [], named.get(t, 0))
        reasons += why_news
        parts["reach"] = reach_score(int(deg.get(t, 0)))
        if deg.get(t, 0) >= 10:
            reasons.append(f"{int(deg[t])} stocks moved closely with it over the year")
        a = t in against
        parts["against"] = float(min(1.0, abs((zt or 0) - Hh.get(t, 0)) / 3.0)) if a else 0.0
        if a:
            reasons.append(f"moved against its wave ({zt:+.1f} usual days against a wave of {Hh[t]:+.1f})")
        parts["memory"] = 1.0 if t in seen else 0.0
        if t in seen:
            reasons.append(seen[t])
        parts["fragile"], why_f = fragile_score(m)
        reasons += why_f
        score = sum(WEIGHTS[k] * v for k, v in parts.items())
        out.append({"t": t, "company": m.get("company"), "move": m.get("move", moves.get(t)), "z": zt,
                    "wave": float(Hh[t]) if t in Hh else None, "mover": t in movers, "score": float(score),
                    "parts": {k: round(v, 3) for k, v in parts.items()}, "reasons": reasons})
    return weigh(out)


def trading_behind(t: str, since_utc: str, until_utc: str | None = None, move: float | None = None,
                   fetch=None) -> int | None:
    """How many distinct prices after ``since_utc`` (the scan's close) and up to ``until_utc`` (the check) carry at
    least half of ``move`` from the close, read from Yahoo's five-minute chart with trading outside the session. One
    or two such prices mean the move rests on a stray print. None when it cannot be read."""
    from datetime import datetime
    from brain import net
    try:
        r = (fetch or net.fetch)(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=2d&interval=5m"
                                 "&includePrePost=true", ttl=0)
        res = json.loads(r.body)["chart"]["result"][0]
    except Exception:  # noqa: BLE001 - no reading is not a reason to stop
        return None
    ts, close = res.get("timestamp") or [], res["indicators"]["quote"][0].get("close") or []
    cut = datetime.fromisoformat(since_utc.replace("Z", "+00:00")).timestamp()
    end = datetime.fromisoformat(until_utc.replace("Z", "+00:00")).timestamp() if until_utc else float("inf")
    before = [c for a, c in zip(ts, close) if c is not None and a < cut]
    if not before or not move:
        return None
    ref = before[-1]
    after = [c for a, c in zip(ts, close) if c is not None and cut < a <= end]
    # Yahoo repeats the last price in later bars when nothing traded, so count distinct prices, not bars
    return len({round(c / ref, 3) for c in after if (c / ref - 1) * np.sign(move) >= abs(move) / 2})


def check_trading(cands: list[dict], close_utc: str, k: int, until_utc: str | None = None, fetch=None) -> None:
    """For the top ``k``, discount a move that rests on one or two trades between the close and the check (a stray
    print)."""
    for c in sorted(cands, key=lambda c: -c["score"])[:k]:
        n = trading_behind(c["t"], close_utc, until_utc, c.get("move"), fetch)
        c["bars"] = n
        if n is not None and n <= THIN_BARS:
            c["score"] *= THIN_DISCOUNT
            c["reasons"].append(f"but only {n} trade{'s' if n != 1 else ''} at the new price since the close: "
                                "likely a stray print")


def weigh(cands: list[dict]) -> list[dict]:
    """Attention weights from the scores (a softmax at TEMP), the highest first."""
    if cands:
        s = np.array([c["score"] for c in cands])
        w = np.exp((s - s.max()) / TEMP)
        w /= w.sum()
        for c, a in zip(cands, w):
            c["attention"] = float(a)
    return sorted(cands, key=lambda c: -c["score"])


# ---------- digging ----------
def key_sentences(text: str, names: list[str], k: int = KEY_SENTENCES) -> list[str]:
    """The sentences that carry numbers, guidance or events, those naming the company first."""
    sents = [s.strip() for s in _SENT.split(" ".join(text.split())) if 30 <= len(s.strip()) <= 400]
    low = [n.lower() for n in names if n]
    scored = [(2 * any(n in s.lower() for n in low) + len(_KEY.findall(s)), i, s) for i, s in enumerate(sents)]
    best = sorted([x for x in scored if x[0] > 0], key=lambda x: (-x[0], x[1]))[:k]
    return [s for _, _, s in sorted(best, key=lambda x: x[1])]


def read_filings(t: str, asof: str, names: list[str]) -> list[dict]:
    """The stock's filings since the close, read deeper (the press release of an 8-K, else the filing's own text)."""
    from brain import desk
    from brain.sources import releases
    out = []
    for f in desk.filings_of(t, asof)[-2:]:
        url = (f.source or {}).get("url", "")
        try:
            if str(f.value).upper().startswith("8-K"):
                r = releases.read_release(f)
                url = r.source["url"]
            raw = releases._get(url, releases.HEAD_BYTES).decode("utf-8", "replace")
            title, _ = releases.opening(raw)
            from brain import scout
            text = " ".join(scout.html_to_text(raw)["text"].split())[:FILING_CHARS]
            out.append({"what": desk.scanwatch.describe(f), "at": f.known_at, "title": title, "url": url,
                        "key": key_sentences(text, names), "text": text})
        except Exception as e:  # noqa: BLE001 - one filing that cannot be read is listed without its text
            out.append({"what": desk.scanwatch.describe(f), "at": f.known_at, "url": url, "error": str(e)[:200]})
    return out


def read_pages(urls: list[str], names: list[str], gateway=None) -> list[dict]:
    """Story pages read through the engine's web reader; a refused page is reported, never got around."""
    from brain.web import extract
    from brain.web.gateway import Gateway
    g = gateway or Gateway(mode="http")
    out = []
    try:
        for u in urls:
            p = g.open(u)
            if not p.ok:
                out.append({"url": u, "refused": str(p.error)[:160]})
                continue
            a = extract.article(p.html, p.final_url or u)
            text = " ".join((a.get("text") or "").split())[:PAGE_CHARS]
            out.append({"url": p.final_url or u, "title": a.get("title"), "published": a.get("published"),
                        "key": key_sentences(text, names), "text": text})
    finally:
        if gateway is None:
            g.close()
    return out


def remembered(t: str, company: str | None, k: int = 3) -> list[dict]:
    """What the brain said or found about the stock before (its memory bank), newest passages first among the best."""
    try:
        from brain import memory
        hits = memory.recall(f"{t} {company or ''}".strip(), k=k * 3, kinds={"me", "note", "summary", "you"})
    except Exception:  # noqa: BLE001 - no memory index is not a reason to stop
        return []
    out = []
    for h in hits:
        text = " ".join((h.get("text") or "").split())
        m = re.search(rf"\b{re.escape(t)}\b", text)
        if not m:
            continue
        a = max(0, m.start() - 180)
        out.append({"at": (h.get("at") or "")[:10], "kind": h.get("kind"),
                    "text": ("..." if a else "") + text[a:m.start() + 260] + "..."})
        if len(out) >= k:
            break
    return out


def pond_view(pond: dict | None, t: str, moves: dict, k: int = 5) -> dict | None:
    """The wave the stock felt, the stones that sent it, and who its own move reaches (its strongest links)."""
    if pond is None or t not in pond["tickers"]:
        return None
    j = pond["tickers"].index(t)
    tk = pond["tickers"]
    src = ripple.contributions(pond, pond["z"], pond["S"], j)
    C = pond["pond"]["C"][:, j]
    near = [i for i in np.argsort(-np.abs(C))[:k] if abs(C[i]) >= ripple.RING and i != j]
    z = pond["z"][j]
    return {"wave": float(pond["H"][j]), "own": float(z) if np.isfinite(z) else None,
            "how": ripple.how_it_moved(float(pond["H"][j]), float(z), bool(pond["stone"][j])) if np.isfinite(z) else None,
            "from": [[tk[i], round(a, 2)] for i, a in src],
            "reaches": [[tk[i], round(float(C[i]), 2), moves.get(tk[i])] for i in near]}


def focus(bank, case: dict, *, top: int = TOP, read: bool = True, recall: bool = True, trades: bool = True,
          log=None) -> dict:
    """The check's focus: every candidate scored, the top ``top`` dug into."""
    say = log or (lambda m: None)
    moves = {t: v for t, v in (case.get("moves") or {}).items() if v is not None}
    spy = next((m["move"] for m in case.get("market") or [] if m.get("symbol") == "SPY"), None)
    pond = None
    if spy is not None and moves:
        say("  the pond on the check's moves")
        try:
            pond = ripple.live(bank, case["asof"], moves, float(spy))
        except Exception as e:  # noqa: BLE001 - without a pond, attention still scores the movers
            say(f"  no pond: {e}")
    cands = candidates(case, pond)
    from brain import why
    if trades:
        say("  the trading behind the top moves")
        at = str(case.get("at") or "")
        check_trading(cands, why.close_utc(case["asof"]), top + 4, at if at.endswith("Z") else None)
        cands = weigh(cands)
    missing = [c["t"] for c in cands[:top] if not c.get("company")]
    if missing:
        ind = why.industry_map(bank.view(), missing)
        for c in cands[:top]:
            c["company"] = c.get("company") or (ind.get(c["t"]) or {}).get("company")
    stories = {s["id"]: s for s in case.get("stories") or []}
    movers = {m["t"]: m for m in case["movers"]}
    dug = []
    for c in cands[:top]:
        t = c["t"]
        names = [t, c.get("company") or ""]
        if c.get("company"):
            names.append(str(c["company"]).split()[0])
        item = dict(c)
        if read:
            say(f"  reading {t}")
            item["filings"] = read_filings(t, case["asof"], names)
            urls = [stories[i]["url"] for i in (movers.get(t, {}).get("stories") or [])[:2] if i in stories]
            item["pages"] = read_pages(urls, names) if urls else []
        if recall:
            item["memory"] = remembered(t, c.get("company"))
        item["pond"] = pond_view(pond, t, moves)
        dug.append(item)
    return {"check": case["check"], "asof": case["asof"], "built_at": now_iso(), "weights": WEIGHTS, "temp": TEMP,
            "candidates": len(cands), "focus": dug,
            "rest": [{k: c[k] for k in ("t", "score", "attention", "reasons")} for c in cands[top:top + 20]]}


def save(f: dict) -> Path:
    p = focus_dir() / f["asof"] / f"{f['check']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(f, indent=1))
    return p


def render(f: dict, full: bool = False) -> list[str]:
    L = [f"ATTENTION for the check {f['check']} (against the close of {f['asof']}): {f['candidates']} stocks scored on "
         "surprise, what nothing explains, news, reach, moving against the wave, what was already watched and "
         "fragility; the weights sum to 100%."]
    for k, c in enumerate(f["focus"], 1):
        mv = f"{c['move'] * 100:+.1f}%" if c.get("move") is not None else "-"
        L.append(f"{k}. {c['t']} ({c.get('company') or ''}) {mv}: attention {c['attention'] * 100:.0f}% "
                 f"(score {c['score']:.2f}) -- " + ("; ".join(c["reasons"]) or "no single reason stands out"))
        pv = c.get("pond")
        if pv and pv.get("own") is not None:
            L.append(f"     pond: wave {pv['wave']:+.1f}, own move {pv['own']:+.1f} usual days ({pv['how']})"
                     + (f"; from {', '.join(f'{a} {b:+.1f}' for a, b in pv['from'])}" if pv["from"] else "")
                     + (f"; reaches {', '.join(f'{a} ({r:.2f}) {m * 100:+.1f}%' if m is not None else a for a, r, m in pv['reaches'])}"
                        if pv["reaches"] else ""))
        for fl in c.get("filings") or []:
            if fl.get("error"):
                L.append(f"     filing {fl['what']}: not read ({fl['error'][:80]})")
                continue
            L.append(f"     filing {fl['what']} at {fl['at'][:16]}Z: {fl['title'][:120]}  <{fl['url']}>")
            L += [f"       - {s[:300]}" for s in (fl["key"] if not full else fl["key"] + [fl["text"]])]
        for pg in c.get("pages") or []:
            if pg.get("refused"):
                L.append(f"     page not read: {pg['url']} ({pg['refused'][:80]})")
                continue
            L.append(f"     page {pg.get('published') or ''} {(pg.get('title') or '')[:100]}  <{pg['url']}>")
            L += [f"       - {s[:300]}" for s in pg["key"]]
        for h in c.get("memory") or []:
            L.append(f"     remembered ({(h['at'] + ', ') if h['at'] else ''}{h['kind']}): {h['text'][:440]}")
    if f["rest"]:
        L.append("Next: " + ", ".join(f"{c['t']} {c['attention'] * 100:.1f}%" for c in f["rest"][:12]))
    L.append("Attention says where to look, not which way; what it read is for the note to reason about and cite "
             "(pages by their URL under 'read').")
    return L


def _cmd(a) -> int:
    from brain import desk
    from brain.bank import Bank
    case = desk.load_case(a.check)
    f = focus(Bank(), case, top=a.top, read=not a.no_read, recall=not a.no_recall, log=print)
    p = save(f)
    for line in render(f, a.full):
        print(line)
    print(f"focus recorded at {p}")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("attention", help="what deserves a closer look at the latest check, and digging into it")
    p.add_argument("--check", help="a case file's check stamp (default: the latest)")
    p.add_argument("--top", type=int, default=TOP)
    p.add_argument("--no-read", action="store_true", help="score only: do not read filings and pages")
    p.add_argument("--no-recall", action="store_true", help="do not search the memory bank")
    p.add_argument("--full", action="store_true", help="print the read texts in full")
    p.set_defaults(fn=_cmd)
