"""Declared tests of the chain engine's cause -> effect links (docs/LINKS.md).

A link is measured where the bank can see both ends: a cause the bank holds a daily series for (WTI crude,
the 10-year Treasury yield, the dollar index) and an effect that is a fixed basket of stocks, measured as the
basket's equal-weight total return minus SPY's. A chain whose middle steps the bank cannot see (fuel costs,
profits, mortgage rates) is tested end to end: oil -> airline stocks covers the three links between them.
Links the bank cannot reach at either end stay untested, and ``needs`` says what would measure them.

Each pair is tested twice, on SPY's calendar:

- **same month** (descriptive, REPORT): does the effect move with the cause in the same month (month-end close
  to month-end close), and by how much?
- **next month** (gated): does this month's change in the cause predict next month's effect, measured from the
  close of the next month's first session to its last? This is the tradeable question, on the research
  harness's rule (signal at the month-end close, fill at the next session's close): a link the market prices at
  once shows up in the same month and not in the next.

The rules are the research ledger's. A test is declared (its full spec, the code's sha and the git commit)
before it runs, and a declared spec is frozen: a change is a new id. TRAIN is 2011-2018 and TEST runs from
2019 to the last complete month at declare; the verdict reads TEST only. Every series is read as the bank
held it at declare (``data_asof``), and the result names each snapshot it read. The p-value is one-sided, in
the direction the links imply, from shuffling the cause's months; the controls check that null against the
cause's own autocorrelation (circular shifts of the real series). A next-month test passes when its slope
points that way, p < 0.05, its controls pass, and its Benjamini-Hochberg q < 0.10 across every gated test in
the ledger (``research/run.py``). Results are written to the ledger and RESULTS.md whatever they say.
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
from brain.research.forecast_tests import (
    Refused,
    is_trial,
    paths,
)
from brain.timeutil import now_iso

VERSION = 1         # in every id: a change to the design or the code after declare is a new version
BENCH = "SPY"
SPLIT = {"train": ["2011-01", "2018-12"], "test_start": "2019-01"}
TRIAL_SPLIT = {"train": ["2011-01", "2014-12"], "test": ["2015-01", "2018-12"]}   # TRAIN years only
N_PERM = 5000
N_BOOT = 2000
SEED = 11
ALPHA = 0.05
CONTROLS = {"shift_min": 3, "n_perm": 500, "fpr_max": 0.10, "oracle_p_max": 0.01}
BH = {"bh_q": 0.10, "bh_over": "every hypothesis, every gated forecast test and every gated link test with a "
                               "result in the ledger, withdrawn ones included"}
MIN_SHARE = 0.8     # a window needs this share of its months measured, and both its ends, or it is refused
MIN_MEMBERS = 2     # basket members with prices on both sessions, or that month is not measured
STALE_DAYS = {"cause": 7, "stocks": 0}   # how old a value may be at a session: stocks and SPY must trade on it
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
    return f"L{VERSION}-{SHORT[pair['cause']]}-{SHORT[pair['effect']]}-{lag}"


def tests(links: list[dict] | None = None) -> list[dict]:
    """The link tests the links allow, same month then next month for each pair. Two paths between the same
    pair would share an id: that is refused, so the second gets a name of its own first."""
    from brain import chain
    links = chain.load_links() if links is None else links
    out = [{"id": test_id(p, lag), "lag": lag, **p} for p in pairs(links) for lag in ("same", "next")]
    ids = [t["id"] for t in out]
    if len(set(ids)) != len(ids):
        raise Refused(f"two paths give the same link test id: {sorted({i for i in ids if ids.count(i) > 1})}")
    return out


def code_files() -> list[Path]:
    """What a link test's numbers run through: this file, the verdicts and their correction (research/run.py,
    research/stats.py), the trial rules (research/forecast_tests.py), the bank's point-in-time view, the price
    book, the series store and the Treasury reader. A change to one after declare makes ``test`` refuse."""
    from brain import bank, metrics, series
    from brain.research import forecast_tests, run, stats
    from brain.sources import treasury
    return [HERE, *(Path(m.__file__).resolve() for m in (run, stats, forecast_tests, bank, metrics, series, treasury))]


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
    when = "the same month" if t["lag"] == "same" else "the next month"
    way = "higher" if t["sign"] > 0 else "lower"
    return (f"The more the {c['label'].split(',')[0]} rises in a month, the {way} the {e['label']} basket's return "
            f"over SPY in {when}: the slope of that excess return on the cause's monthly change is "
            f"{'>' if t['sign'] > 0 else '<'} 0.")


VARYING = ("test", "code_sha", "data_asof", "trial")   # what a later declare may differ in without a new id


def spec(t: dict, anchor: str, trial: bool, data_asof: str | None = None) -> dict:
    split = ({"train": TRIAL_SPLIT["train"], "test": TRIAL_SPLIT["test"]} if trial
             else {"train": SPLIT["train"], "test": [SPLIT["test_start"], anchor]})
    return {"kind": "link", "version": VERSION, "id": t["id"], "cause": t["cause"], "effect": t["effect"],
            "path": t["path"], "sign": t["sign"], "lag": t["lag"], "cause_measure": NODES[t["cause"]],
            "effect_measure": NODES[t["effect"]], "bench": BENCH, "gate": "slope" if t["lag"] == "next" else "report",
            "timing": ("month-end close to month-end close" if t["lag"] == "same" else
                       "cause: month-end close to month-end close; effect: the next month's first-session close "
                       "to its last-session close"),
            **split, "n_perm": N_PERM, "n_boot": N_BOOT, "seed": SEED, "alpha": ALPHA, **BH, "controls": CONTROLS,
            "min_share": MIN_SHARE, "min_members": MIN_MEMBERS, "stale_days": STALE_DAYS,
            "claim": claim(t), "code_sha": code_sha(), "data_asof": data_asof or now_iso(), "trial": bool(trial)}


def spec_sha(s: dict) -> str:
    return hashlib.sha256(canon(s).encode()).hexdigest()[:16]


# ---------- the data ----------
def month_ends(cal) -> dict[str, str]:
    """Each month's last session on ``cal`` (YYYY-MM -> YYYY-MM-DD)."""
    out = {}
    for d in np.asarray(cal).astype(str):
        out[d[:7]] = d
    return out


def month_starts(cal) -> dict[str, str]:
    """Each month's first session on ``cal``."""
    out = {}
    for d in np.asarray(cal).astype(str):
        out.setdefault(d[:7], d)
    return out


def _days(a: str, b: str) -> int:
    return int((np.datetime64(b) - np.datetime64(a)) / np.timedelta64(1, "D"))


def sample(dates, values, at: dict[str, str], stale: int = STALE_DAYS["cause"]) -> dict[str, float]:
    """The series' last value on or before each month's session in ``at``, if it is at most ``stale`` days older
    (0: it must be that session's own)."""
    dates = np.asarray(dates).astype(str)
    values = np.asarray(values, dtype=float)
    out = {}
    for m, d in at.items():
        i = int(np.searchsorted(dates, d, side="right")) - 1
        if i >= 0 and math.isfinite(values[i]) and _days(dates[i], d) <= stale:
            out[m] = float(values[i])
    return out


def _change(lv: dict[str, float], months: list[str], kind: str) -> np.ndarray:
    """Month i's change from month i-1's value to month i's."""
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


def _within(first: dict[str, float], last: dict[str, float], months: list[str]) -> np.ndarray:
    """Month i's return from its first session's close to its last's."""
    out = np.full(len(months), np.nan)
    for i, m in enumerate(months):
        a, b = first.get(m), last.get(m)
        if a is not None and b is not None and a > 0 and b > 0:
            out[i] = b / a - 1
    return out


def _excess(rets: np.ndarray, bench: np.ndarray, min_members: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.isfinite(rets).sum(0)
    total = np.where(np.isfinite(rets), rets, 0.0).sum(0)
    return np.where(n >= min_members, total / np.maximum(n, 1), np.nan) - bench, n


def monthly(bank: Bank, s: dict) -> dict:
    """Month by month on SPY's calendar, from the series as the bank held them at ``s['data_asof']``: the
    cause's change (``x``), the basket's equal-weight total return minus SPY's over the same month (``y``) and
    from the month's first-session close to its last (``y_next``, what the next-month test reads), how many
    members each averaged, and the snapshots read (``inputs``)."""
    from brain import series
    from brain.metrics import PriceBook, fetched_by
    from brain.sources import treasury
    cm, em, asof = s["cause_measure"], s["effect_measure"], s.get("data_asof")
    syms = [*em["tickers"], s["bench"]] + ([cm["symbol"]] if cm["measure"] == "price" else [])
    book = PriceBook(bank, syms, retrieved_by=asof)
    if not book.has(s["bench"]):
        raise Refused(f"no {s['bench']} prices in the bank: python -m brain chain measure")
    cal = book.dates[s["bench"]]
    ends, starts = month_ends(cal), month_starts(cal)
    months = sorted(ends)
    inputs = {t: {"fact": book.snapshot[t].ref, "series": book.snapshot[t].value, "last": str(book.dates[t][-1])}
              for t in syms if book.has(t)}
    if cm["measure"] == "price":
        if not book.has(cm["symbol"]):
            raise Refused(f"no {cm['symbol']} prices in the bank: python -m brain chain measure")
        lv = sample(book.dates[cm["symbol"]], book.close[cm["symbol"]], ends, s["stale_days"]["cause"])
    else:
        f = bank.view().latest("MACRO", treasury.ATTR, fetched_by(asof) if asof else None)
        if f is None:
            raise Refused("no Treasury yield curve in the bank: python -m brain chain measure")
        df = series.load(f.value)
        inputs["treasury"] = {"fact": f.ref, "series": f.value, "last": str(df["date"].iloc[-1])}
        lv = sample(df["date"], df[cm["tenor"]], ends, s["stale_days"]["cause"])
    x = _change(lv, months, cm["measure"])
    st = s["stale_days"]["stocks"]

    def px(t):
        return book.dates[t], book.adj[t]
    month = {t: _change(sample(*px(t), ends, st), months, "price") for t in syms if book.has(t)}
    inner = {t: _within(sample(*px(t), starts, st), sample(*px(t), ends, st), months) for t in syms if book.has(t)}
    none = np.full(len(months), np.nan)
    y, n = _excess(np.array([month.get(t, none) for t in em["tickers"]]), month[s["bench"]], s["min_members"])
    y_next, n_next = _excess(np.array([inner.get(t, none) for t in em["tickers"]]), inner[s["bench"]], s["min_members"])
    return {"months": months, "x": x, "y": y, "y_next": y_next, "members": n, "members_next": n_next, "inputs": inputs}


def window_pairs(mo: dict, lag: str, first: str, last: str) -> dict:
    """(cause month, effect month) pairs whose effect month is in [first, last]: the same month (``y``), or the
    next (``y_next``, from that month's first-session close)."""
    months, x = mo["months"], mo["x"]
    k = 0 if lag == "same" else 1
    y, members = (mo["y"], mo["members"]) if lag == "same" else (mo["y_next"], mo["members_next"])
    keep = [i for i in range(len(months) - k) if first <= months[i + k] <= last]
    xs = np.array([x[i] for i in keep])
    ys = np.array([y[i + k] for i in keep])
    ok = np.isfinite(xs) & np.isfinite(ys)
    return {"x": xs[ok], "y": ys[ok], "months": [months[i + k] for i, o in zip(keep, ok) if o], "in_window": len(keep),
            "missing": [months[i + k] for i, o in zip(keep, ok) if not o],
            "members": float(np.mean([members[i + k] for i, o in zip(keep, ok) if o])) if ok.any() else 0.0}


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
    """Circular shifts of the real cause by ``shift_min`` months or more keep its autocorrelation, fat tails and
    jumps but break any link to the effect: if more than ``fpr_max`` of them reach p < 0.05, the shuffle null
    is too narrow for these series and the verdict is void. A cause equal to the effect must reach p <
    ``oracle_p_max``, or the test cannot see a real link."""
    k0 = c["shift_min"]
    shifts = list(range(k0, len(x) - k0 + 1))
    hits = sum(perm_p(np.roll(x, k), y, sign, c["n_perm"], rng) < ALPHA for k in shifts)
    fpr = hits / len(shifts) if shifts else float("nan")
    oracle = perm_p(sign * y, y, sign, c["n_perm"], rng)
    return {"shift_fpr": fpr, "n_shifts": len(shifts), "oracle_p": oracle,
            "pass": bool(shifts and fpr <= c["fpr_max"] and oracle < c["oracle_p_max"])}


def evaluate(mo: dict, s: dict) -> dict:
    """The result record's numbers for spec ``s`` on monthly data ``mo``. Refuses a window too thin to say much."""
    rng = np.random.default_rng(s["seed"])
    out = {}
    for name in ("train", "test"):
        first, last = s[name]
        w = window_pairs(mo, s["lag"], first, last)
        if (w["in_window"] == 0 or len(w["x"]) < s["min_share"] * w["in_window"]
                or w["months"][0] != first or w["months"][-1] != last):
            raise Refused(f"{s['id']}: {len(w['x'])} of {w['in_window']} {name.upper()} months are measured "
                          f"(at least {s['min_share']:.0%} needed, and both {first} and {last}): "
                          f"python -m brain chain measure, then declare again under a new version")
        out[name] = {**measure(w["x"], w["y"], s, rng), "months": [w["months"][0], w["months"][-1]],
                     "in_window": w["in_window"], "missing": w["missing"], "members_mean": round(w["members"], 2)}
        if name == "test":
            ctl = controls(w["x"], w["y"], s["sign"], rng, s["controls"])
    te = out["test"]
    sign_ok = bool(math.isfinite(te["slope"]) and s["sign"] * te["slope"] > 0)
    gate = {"sign_ok": sign_ok, "p_ok": bool(te["p"] < s["alpha"])}
    gate["pass"] = bool(s["gate"] == "slope" and sign_ok and gate["p_ok"] and ctl["pass"])
    return {"metrics": out, "controls": ctl, "gate": gate, "p": te["p"], "inputs": mo.get("inputs", {})}


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


def _frozen(s: dict) -> dict:
    return {k: v for k, v in s.items() if k not in VARYING}


def committed_declarations() -> set[tuple[str, str]]:
    """(id, spec_sha) of the link declarations in HEAD's research/ledger.jsonl."""
    import json
    r = _git("show", "HEAD:research/ledger.jsonl")
    out = set()
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("op") == "declare" and rec.get("kind") == "link":
            out.add((rec.get("id"), rec.get("spec_sha")))
    return out


def declare(bank: Bank, *, trial: bool = False, ids=None, log=print) -> list[dict]:
    """Write each link test's spec to the ledger before it runs. An id already declared keeps its spec; one
    whose design has changed since is refused (a change is a new VERSION)."""
    trial = is_trial(trial)
    led, _ = paths(trial)
    if not trial and (dirty := tree_dirty()):
        raise Refused("commit brain/ and knowledge/ first; a declaration pins the committed code: " + "; ".join(dirty))
    anchor = TRIAL_SPLIT["test"][1] if trial else anchor_month(bank)
    asof = now_iso()
    have = {r["id"]: r["spec"] for r in _records(led) if r["op"] == "declare"}
    new = []
    for t in tests():
        if ids and t["id"] not in ids:
            continue
        s = spec(t, anchor, trial, asof)
        if t["id"] in have:
            if _frozen(have[t["id"]]) != _frozen(s):
                raise Refused(f"{t['id']} was declared with a different design; declared specs are frozen: raise "
                              f"VERSION in brain/chain/linktest.py for new ids")
            continue
        rec = {"op": "declare", "kind": "link", "id": t["id"], "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
               **ledger.git_state()}
        ledger.append(rec, led)
        new.append(rec)
        if log:
            log(f"declared {t['id']} ({rec['spec_sha']}): TEST {s['test'][0]}..{s['test'][1]}, {s['gate']}")
    return new


def test(bank: Bank, ids=None, *, trial: bool = False, log=print) -> list[dict]:
    """Run declared link tests (default: every declared one without a result) and rewrite RESULTS.md. A trial
    runs trial declarations only (TRAIN years); a real run, real declarations committed at HEAD only."""
    from brain.research import run
    trial = is_trial(trial)
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
        if spec_sha(d["spec"]) != d["spec_sha"]:
            raise Refused(f"{i}: its declared spec does not match its spec_sha")
        if trial and (not d["spec"].get("trial") or d["spec"]["test"][1] > TRIAL_SPLIT["test"][1]):
            raise Refused(f"{i} is not a trial declaration: a trial reads TRAIN years only")
        if not trial and d["spec"].get("trial"):
            raise Refused(f"{i} is a trial declaration")
        if not trial and d["spec"]["code_sha"] != code_sha():
            raise Refused(f"{i}: the code is not the declared code ({d['spec']['code_sha']} declared, {code_sha()} now)")
        if not trial and (i, d["spec_sha"]) not in committed_declarations():
            raise Refused(f"{i}: commit the ledger with its declaration before running it")
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
    return (f"{rec['id']}: {te['per_move'] * 100:+.2f} points of excess return per {s['cause_measure']['move_label']} "
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

