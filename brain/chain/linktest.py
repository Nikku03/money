"""Declared tests of the chain engine's cause -> effect links (docs/LINKS.md).

A link is measured where the bank can see both ends: a cause the bank holds a daily series for (WTI crude,
the 10-year Treasury yield, the dollar index) and an effect that is a fixed basket of stocks, measured as the
basket's equal-weight total return minus SPY's. A chain whose middle steps the bank cannot see (fuel costs,
profits, mortgage rates) is tested end to end: oil -> airline stocks covers the three links between them.
Links the bank cannot reach at either end stay untested, and ``needs`` says what would measure them.

Each pair is tested twice, on month-end closes:

- **same month** (descriptive, REPORT): does the effect move with the cause in the same month, and by how much?
- **next month** (gated): does this month's change in the cause predict next month's effect? This is the
  tradeable question: a link the market prices at once shows up in the same month and not in the next.

The rules are the research ledger's. A test is declared (its full spec, the code's sha and the git commit)
before it runs, and a declared spec is frozen: a change is a new id. TRAIN is 2011-2018 and TEST runs from
2019 to the last complete month at declare; the verdict reads TEST only. The p-value is one-sided, in the
direction the links imply, from shuffling the cause's months. A next-month test passes when its slope points
that way, p < 0.05, its controls pass, and its Benjamini-Hochberg q < 0.10 across every gated test in the
ledger (``research/run.py``). Results are written to the ledger and RESULTS.md whatever they say.
"""
from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np

from brain import config
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research.forecast_tests import (  # noqa: F401 - is_trial: the CLI's trial rule
    Refused,
    is_trial,
    paths,
)
from brain.timeutil import now_iso

BENCH = "SPY"
SPLIT = {"train": ["2011-01", "2018-12"], "test_start": "2019-01"}
TRIAL_SPLIT = {"train": ["2011-01", "2014-12"], "test": ["2015-01", "2018-12"]}   # TRAIN years only
N_PERM = 5000
N_BOOT = 2000
SEED = 11
ALPHA = 0.05
CONTROLS = {"n_random": 200, "n_perm": 500, "fpr_max": 0.10, "oracle_p_max": 0.01}
MIN_SHARE = 0.8     # a window needs this share of its months measured, or the test is refused
MIN_MEMBERS = 2     # basket members with both month-end closes, or that month is not measured
STALE_DAYS = 7      # a series' last value may be this many days before the month's last session
CLEAN = ("brain", "knowledge")
LINKS = config.ROOT / "knowledge" / "links.json"
HERE = Path(__file__).resolve()

# What measures each node. ``dir`` maps the node's name onto its series: +1 when the series rises as named
# ("oil_price_up": oil's change), -1 when it falls ("airline_stocks_down": the airlines' excess return).
NODES = {
    "oil_price_up": {"measure": "price", "symbol": "CL=F", "dir": 1, "move": 0.10, "move_label": "+10% oil",
                     "move_words": "oil up 10%",
                     "label": "WTI crude (front-month future), monthly % change"},
    "interest_rates_up": {"measure": "treasury", "tenor": "10 Yr", "dir": 1, "move": 0.25,
                          "move_label": "+0.25 pt 10-year", "move_words": "the 10-year yield up 0.25 point",
                          "label": "10-year Treasury par yield, monthly change in percentage points"},
    "dollar_up": {"measure": "price", "symbol": "DX-Y.NYB", "dir": 1, "move": 0.02, "move_label": "+2% dollar",
                  "move_words": "the dollar up 2%",
                  "label": "US dollar index, monthly % change"},
    "airline_stocks_down": {"measure": "basket", "tickers": ["DAL", "UAL", "LUV", "AAL", "ALK"], "dir": -1,
                            "label": "airline stocks"},
    "oil_producer_stocks_up": {"measure": "basket", "tickers": ["XOM", "CVX", "COP", "EOG", "OXY"], "dir": 1,
                               "label": "oil producer stocks"},
    "bank_stocks_up": {"measure": "basket", "tickers": ["JPM", "BAC", "WFC", "C", "PNC"], "dir": 1, "label": "bank stocks"},
    "tech_stocks_down": {"measure": "basket", "tickers": ["AAPL", "MSFT", "NVDA", "ORCL", "ADBE"], "dir": -1,
                         "label": "tech stocks"},
    "homebuilder_stocks_down": {"measure": "basket", "tickers": ["LEN", "DHI", "PHM", "NVR", "TOL"], "dir": -1,
                                "label": "homebuilder stocks"},
}
SHORT = {"oil_price_up": "oil", "interest_rates_up": "rates", "dollar_up": "dollar", "airline_stocks_down": "airlines",
         "oil_producer_stocks_up": "producers", "bank_stocks_up": "banks", "tech_stocks_down": "tech",
         "homebuilder_stocks_down": "homebuilders"}
NEEDS = {
    "strait_disruption": "a dated list of shipping disruptions",
    "inflation_up": "CPI with its release dates (FRED, with FRED_API_KEY)",
    "mortgage_rates_up": "a mortgage-rate series (FRED MORTGAGE30US)",
    "exporter_revenue_down": "each company's foreign sales",
    "exporter_stocks_down": "a basket of exporters, from each company's share of foreign sales",
}
QUARTERLY = "quarterly figures from filings (a different, quarterly test)"


def needs(node: str) -> str | None:
    """What would measure ``node``; None when the bank already does."""
    if node in NODES:
        return None
    return NEEDS.get(node, QUARTERLY)


# ---------- the tests ----------
def pairs(links: list[dict]) -> list[dict]:
    """Every (measured cause, measured basket) pair the links connect through unmeasured nodes only, with the
    path and the sign the path implies for the slope of the basket's excess return on the cause's change."""
    out_of: dict[str, list[dict]] = {}
    for ln in links:
        out_of.setdefault(ln["cause"], []).append(ln)
    found = []
    for a, m in NODES.items():
        if m["measure"] == "basket":
            continue

        def walk(node, path, a=a, m=m):
            for ln in out_of.get(node, []):
                b = ln["effect"]
                if b == a or any(p["cause"] == b for p in path):
                    continue
                step = path + [{"cause": ln["cause"], "effect": b, "sign": int(ln.get("sign", 1))}]
                if b in NODES:
                    if NODES[b]["measure"] == "basket":
                        sign = int(np.prod([s["sign"] for s in step])) * m["dir"] * NODES[b]["dir"]
                        found.append({"cause": a, "effect": b, "path": step, "sign": sign})
                else:
                    walk(b, step)
        walk(a, [])
    return found


def test_id(pair: dict, lag: str) -> str:
    return f"L-{SHORT[pair['cause']]}-{SHORT[pair['effect']]}-{lag}"


def tests(links: list[dict] | None = None) -> list[dict]:
    """The link tests the links allow, same month then next month for each pair."""
    from brain import chain
    links = chain.load_links() if links is None else links
    return [{"id": test_id(p, lag), "lag": lag, **p} for p in pairs(links) for lag in ("same", "next")]


def code_files() -> list[Path]:
    """What a link test's numbers run through: this file, the verdicts (research/run.py), the price book and
    the series store. A change to one after declare makes ``test`` refuse."""
    from brain import metrics, series
    from brain.research import run
    return [HERE, Path(run.__file__).resolve(), Path(metrics.__file__).resolve(), Path(series.__file__).resolve()]


def code_sha() -> str:
    h = hashlib.sha256()
    root = config.ROOT.resolve()
    for p in code_files():
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


def _code_paths() -> list[str]:
    root = config.ROOT.resolve()
    return [p.relative_to(root).as_posix() for p in code_files()]


def claim(t: dict) -> str:
    c, e = NODES[t["cause"]], NODES[t["effect"]]
    when = "that month" if t["lag"] == "same" else "the next month"
    way = "beats" if t["sign"] > 0 else "trails"
    return (f"When the {c['label'].split(',')[0]} rises in a month, the {e['label']} basket {way} SPY {when}: the "
            f"slope of its excess return on the cause's monthly change is {'>' if t['sign'] > 0 else '<'} 0.")


def spec(t: dict, anchor: str, trial: bool) -> dict:
    split = ({"train": TRIAL_SPLIT["train"], "test": TRIAL_SPLIT["test"]} if trial
             else {"train": SPLIT["train"], "test": [SPLIT["test_start"], anchor]})
    return {"kind": "link", "id": t["id"], "cause": t["cause"], "effect": t["effect"], "path": t["path"],
            "sign": t["sign"], "lag": t["lag"], "cause_measure": NODES[t["cause"]], "effect_measure": NODES[t["effect"]],
            "bench": BENCH, "gate": "slope" if t["lag"] == "next" else "report", **split,
            "n_perm": N_PERM, "n_boot": N_BOOT, "seed": SEED, "alpha": ALPHA, "controls": CONTROLS,
            "min_share": MIN_SHARE, "min_members": MIN_MEMBERS, "stale_days": STALE_DAYS,
            "claim": claim(t), "code_sha": code_sha(), "trial": bool(trial)}


def spec_sha(s: dict) -> str:
    return hashlib.sha256(canon(s).encode()).hexdigest()[:16]


# ---------- the data ----------
def month_ends(cal) -> dict[str, str]:
    """Each month's last session on ``cal`` (YYYY-MM -> YYYY-MM-DD)."""
    out = {}
    for d in np.asarray(cal).astype(str):
        out[d[:7]] = d
    return out


def _days(a: str, b: str) -> int:
    return int((np.datetime64(b) - np.datetime64(a)) / np.timedelta64(1, "D"))


def sample(dates, values, ends: dict[str, str]) -> dict[str, float]:
    """The series' last value on or before each month's last session, if it is at most STALE_DAYS older."""
    dates = np.asarray(dates).astype(str)
    values = np.asarray(values, dtype=float)
    out = {}
    for m, d in ends.items():
        i = int(np.searchsorted(dates, d, side="right")) - 1
        if i >= 0 and math.isfinite(values[i]) and _days(dates[i], d) <= STALE_DAYS:
            out[m] = float(values[i])
    return out


def _change(lv: dict[str, float], months: list[str], kind: str) -> np.ndarray:
    out = np.full(len(months), np.nan)
    for i in range(1, len(months)):
        a, b = lv.get(months[i - 1]), lv.get(months[i])
        if a is None or b is None:
            continue
        if kind == "treasury":
            out[i] = b - a
        elif a > 0 and b > 0:
            out[i] = b / a - 1
    return out


def monthly(bank: Bank, s: dict) -> dict:
    """Month by month on SPY's calendar: the cause's change (``x``), the basket's equal-weight total return
    minus SPY's (``y``) and how many members it averaged (``members``)."""
    from brain import series
    from brain.metrics import PriceBook
    from brain.sources import treasury
    cm, em = s["cause_measure"], s["effect_measure"]
    syms = [*em["tickers"], s["bench"]] + ([cm["symbol"]] if cm["measure"] == "price" else [])
    book = PriceBook(bank, syms)
    if not book.has(s["bench"]):
        raise Refused(f"no {s['bench']} prices in the bank: python -m brain chain measure")
    ends = month_ends(book.dates[s["bench"]])
    months = sorted(ends)
    if cm["measure"] == "price":
        if not book.has(cm["symbol"]):
            raise Refused(f"no {cm['symbol']} prices in the bank: python -m brain chain measure")
        lv = sample(book.dates[cm["symbol"]], book.close[cm["symbol"]], ends)
    else:
        f = bank.view().latest("MACRO", treasury.ATTR)
        if f is None:
            raise Refused("no Treasury yield curve in the bank: python -m brain chain measure")
        df = series.load(f.value)
        lv = sample(df["date"], df[cm["tenor"]], ends)
    x = _change(lv, months, cm["measure"])
    bench = _change(sample(book.dates[s["bench"]], book.adj[s["bench"]], ends), months, "price")
    rets = np.array([_change(sample(book.dates[t], book.adj[t], ends), months, "price") if book.has(t)
                     else np.full(len(months), np.nan) for t in em["tickers"]])
    n = np.isfinite(rets).sum(0)
    total = np.where(np.isfinite(rets), rets, 0.0).sum(0)
    basket = np.where(n >= s["min_members"], total / np.maximum(n, 1), np.nan)
    return {"months": months, "x": x, "y": basket - bench, "members": n}


def window_pairs(mo: dict, lag: str, first: str, last: str) -> dict:
    """(cause month, effect month) pairs whose effect month is in [first, last]: the same month, or the next."""
    months, x, y = mo["months"], mo["x"], mo["y"]
    k = 0 if lag == "same" else 1
    keep = [i for i in range(len(months) - k) if first <= months[i + k] <= last]
    xs = np.array([x[i] for i in keep])
    ys = np.array([y[i + k] for i in keep])
    ok = np.isfinite(xs) & np.isfinite(ys)
    return {"x": xs[ok], "y": ys[ok], "months": [months[i + k] for i, o in zip(keep, ok) if o], "in_window": len(keep),
            "members": float(np.mean([mo["members"][i + k] for i, o in zip(keep, ok) if o])) if ok.any() else 0.0}


# ---------- the statistics ----------
def slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    v = float((xc * xc).sum())
    return float((xc * (y - y.mean())).sum() / v) if v > 0 else float("nan")


def perm_p(x: np.ndarray, y: np.ndarray, sign: int, n: int, rng) -> float:
    """One-sided: the share of shuffles of the cause's months whose slope points ``sign``'s way at least as far."""
    obs = sign * slope(x, y)
    if not math.isfinite(obs):
        return 1.0
    X = np.array([rng.permutation(x) for _ in range(n)])
    Xc = X - X.mean(1, keepdims=True)
    b = (Xc @ (y - y.mean())) / (Xc * Xc).sum(1)
    return float((1 + np.sum(sign * b >= obs - 1e-12)) / (n + 1))


def boot_interval(x: np.ndarray, y: np.ndarray, n: int, rng) -> list[float]:
    """The slope's 90% interval from resampling months."""
    idx = rng.integers(0, len(x), (n, len(x)))
    X, Y = x[idx], y[idx]
    Xc = X - X.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (Xc * (Y - Y.mean(1, keepdims=True))).sum(1) / (Xc * Xc).sum(1)
    b = b[np.isfinite(b)]
    return [float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))] if len(b) else [float("nan")] * 2


def measure(x: np.ndarray, y: np.ndarray, s: dict, rng) -> dict:
    b = slope(x, y)
    move = s["cause_measure"]["move"]
    lo, hi = boot_interval(x, y, s["n_boot"], rng)
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
    return {"n": len(x), "slope": b, "per_move": b * move, "interval90": [lo * move, hi * move],
            "corr": corr, "p": perm_p(x, y, s["sign"], s["n_perm"], rng), "cause_sd": float(x.std()) if len(x) else None}


def controls(x: np.ndarray, y: np.ndarray, sign: int, rng, c: dict = CONTROLS) -> dict:
    """Random causes (white noise) should reach p < 0.05 about 5% of the time; a cause that is the effect
    itself should reach it every time. Either failing voids the verdict."""
    hits = sum(perm_p(rng.standard_normal(len(y)), y, sign, c["n_perm"], rng) < ALPHA for _ in range(c["n_random"]))
    fpr = hits / c["n_random"]
    oracle = perm_p(sign * y, y, sign, c["n_perm"], rng)
    return {"random_fpr": fpr, "oracle_p": oracle, "n_random": c["n_random"],
            "pass": bool(fpr <= c["fpr_max"] and oracle < c["oracle_p_max"])}


def evaluate(mo: dict, s: dict) -> dict:
    """The result record's numbers for spec ``s`` on monthly data ``mo``. Refuses a window too thin to say much."""
    rng = np.random.default_rng(s["seed"])
    out = {}
    for name in ("train", "test"):
        first, last = s[name]
        w = window_pairs(mo, s["lag"], first, last)
        if w["in_window"] == 0 or len(w["x"]) < s["min_share"] * w["in_window"]:
            raise Refused(f"{s['id']}: {len(w['x'])} of {w['in_window']} {name.upper()} months are measured "
                          f"(at least {s['min_share']:.0%} needed): python -m brain chain measure")
        out[name] = {**measure(w["x"], w["y"], s, rng), "months": [w["months"][0], w["months"][-1]],
                     "in_window": w["in_window"], "members_mean": round(w["members"], 2)}
        if name == "test":
            ctl = controls(w["x"], w["y"], s["sign"], rng, s["controls"])
    te = out["test"]
    sign_ok = bool(math.isfinite(te["slope"]) and s["sign"] * te["slope"] > 0)
    gate = {"sign_ok": sign_ok, "p_ok": bool(te["p"] < s["alpha"])}
    gate["pass"] = bool(s["gate"] == "slope" and sign_ok and gate["p_ok"] and ctl["pass"])
    return {"metrics": out, "controls": ctl, "gate": gate, "p": te["p"]}


# ---------- declare and test ----------
def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=False)


def tree_dirty() -> list[str]:
    try:
        r = _git("status", "--porcelain", "--", *CLEAN)
    except OSError as e:
        return [f"git: {e}"]
    return [ln for ln in r.stdout.splitlines() if ln.strip()] if r.returncode == 0 else [f"git: {r.stderr.strip()}"]


def lineage_problem(commit: str | None) -> str | None:
    """None when HEAD is ``commit`` or a descendant that leaves the code files unchanged; else why not."""
    if not commit:
        return "the declaration has no commit (git was not available at declare)"
    if _git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        return f"HEAD is not the declare commit {commit[:12]} or a descendant of it"
    d = _git("diff", "--name-only", commit, "HEAD", "--", *_code_paths())
    changed = [x for x in d.stdout.splitlines() if x.strip()]
    return f"code changed since the declare commit {commit[:12]}: {', '.join(changed)}" if changed else None


def _records(led: Path) -> list[dict]:
    return [r for r in ledger.read(led) if r.get("kind") == "link"]


def anchor_month(bank: Bank) -> str:
    """The last complete calendar month before SPY's latest session in the bank."""
    from brain.metrics import PriceBook
    book = PriceBook(bank, [BENCH])
    if not book.has(BENCH):
        raise Refused(f"no {BENCH} prices in the bank: python -m brain chain measure")
    y, m = map(int, str(book.dates[BENCH][-1])[:7].split("-"))
    y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return f"{y:04d}-{m:02d}"


def declare(bank: Bank, *, trial: bool = False, ids=None, log=print) -> list[dict]:
    """Write each link test's spec to the ledger before it runs. An id already declared keeps its spec."""
    led, _ = paths(trial)
    if not trial and (dirty := tree_dirty()):
        raise Refused("commit brain/ and knowledge/ first; a declaration pins the committed code: " + "; ".join(dirty))
    anchor = TRIAL_SPLIT["test"][1] if trial else anchor_month(bank)
    have = {r["id"] for r in _records(led) if r["op"] == "declare"}
    new = []
    for t in tests():
        if (ids and t["id"] not in ids) or t["id"] in have:
            continue
        s = spec(t, anchor, trial)
        rec = {"op": "declare", "kind": "link", "id": t["id"], "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
               **ledger.git_state()}
        ledger.append(rec, led)
        new.append(rec)
        if log:
            log(f"declared {t['id']} ({rec['spec_sha']}): TEST {s['test'][0]}..{s['test'][1]}, {s['gate']}")
    return new


def test(bank: Bank, ids=None, *, trial: bool = False, log=print) -> list[dict]:
    """Run declared link tests (default: every declared one without a result) and rewrite RESULTS.md."""
    from brain.research import run
    led, res = paths(trial)
    recs = _records(led)
    decls = {r["id"]: r for r in recs if r["op"] == "declare"}
    done = {r["id"] for r in recs if r["op"] == "result"}
    gone = ledger.withdrawn(led)
    want = list(ids) if ids else [i for i in decls if i not in done and i not in gone]
    if not want:
        if log:
            log("nothing to test: every declared link test has a result (python -m brain chain declare adds new ones)")
        return []
    if not trial and (dirty := tree_dirty()):
        raise Refused("the working tree has changes under brain/ or knowledge/: " + "; ".join(dirty))
    for i in want:
        d = decls.get(i)
        if d is None:
            raise Refused(f"{i} is not declared: python -m brain chain declare, then commit the ledger")
        if i in gone:
            raise Refused(f"{i} was withdrawn; a retest needs a new id")
        if i in done and not trial:
            raise Refused(f"{i} already has a result; a rerun needs research withdraw and a new id")
        if not trial and (why := lineage_problem(d.get("commit"))):
            raise Refused(f"{i}: {why}")
        if not trial and d["spec"]["code_sha"] != code_sha():
            raise Refused(f"{i}: the code is not the declared code ({d['spec']['code_sha']} declared, {code_sha()} now)")
    done_now = {i: evaluate(monthly(bank, decls[i]["spec"]), decls[i]["spec"]) for i in want}  # all, or none written
    out = []
    for i, r in done_now.items():
        s = decls[i]["spec"]
        rec = {"op": "result", "kind": "link", "id": i, "spec_sha": decls[i]["spec_sha"],
               "gate_type": s["gate"], **r, "at": now_iso(), **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            log(summary(rec, s))
    run.report(led, res)
    if log:
        log(f"wrote {res}" + ("" if trial else "; commit the ledger and RESULTS.md, whatever they say"))
    return out


def summary(rec: dict, s: dict) -> str:
    te = rec["metrics"]["test"]
    lo, hi = te["interval90"]
    return (f"{rec['id']}: {te['per_move'] * 100:+.2f}% vs SPY per {s['cause_measure']['move_label']} "
            f"(90% {lo * 100:+.2f}..{hi * 100:+.2f}), p {te['p']:.4f}, {te['n']} months"
            + ("" if s["gate"] == "report" else f", gate {'met' if rec['gate']['pass'] else 'not met'} before BH"))


# ---------- collecting what the tests read ----------
def collect(bank: Bank, log=print) -> dict:
    """Daily prices for every basket stock, SPY and the market causes, and the Treasury curve since 2010."""
    from brain.sources import prices, treasury
    syms = sorted({t for m in NODES.values() if m["measure"] == "basket" for t in m["tickers"]}
                  | {m["symbol"] for m in NODES.values() if m["measure"] == "price"} | {BENCH})
    added, failed = {}, []
    for s in syms:
        try:
            added[s] = bank.add_many(prices.ingest(s))
        except Exception as e:  # noqa: BLE001 - one symbol's outage must not stop the rest
            failed.append(s)
            if log:
                log(f"  {s}: {e}")
    try:
        added["treasury"] = bank.add_many(treasury.ingest())
    except Exception as e:  # noqa: BLE001
        failed.append("treasury")
        if log:
            log(f"  treasury: {e}")
    return {"added": added, "failed": failed}


# ---------- what the chain engine shows ----------
def status(path=None) -> dict[str, dict]:
    """Each link test's latest verdict row (research/run.py), by id."""
    from brain.research import run
    return {r["id"]: r for r in run.verdicts(path) if r.get("kind") == "link"}


def covering(chain_links: list[dict], links: list[dict] | None = None) -> list[dict]:
    """The pairs whose whole path lies on ``chain_links`` (a chain from ``brain.chain.paths``)."""
    on = {(ln["cause"], ln["effect"]) for ln in chain_links}
    from brain import chain
    links = chain.load_links() if links is None else links
    return [p for p in pairs(links) if all((s["cause"], s["effect"]) in on for s in p["path"])]

