"""Declared tests of corporate events: does a kind of event move the stock? (docs/EVENTS.md)

The events are earnings press releases (SEC 8-K exhibit 99, ``brain/sources/releases.py``) whose opening text the
fixed event rules tag (``brain/web/events.py``): "guidance raised", "guidance cut". A release tagged both is left
out of both. Each test asks, of one event kind and one window:

- **reaction** (descriptive, REPORT): the stock's excess return over SPY on days -1..+1 around the release
  (``brain/events.py``: day 0 is the first close at or after the filing became public);
- **drift** (gated): the excess return over days +2..+21, which an investor could still have acted on.

The comparison is with the other earnings releases of the same month in the same universe: each month, the
tagged releases' mean minus the untagged releases' mean. That keeps an earnings-season move, which every release
shares, from passing for the event's own. The test statistic is the mean of those monthly differences (calendar
time: a month with many releases counts once), with a one-sided sign-flip p in the direction the event implies
(version 1, the 60 large caps) or a two-sided one (version 2, the small and micro caps). The move against SPY alone
is shown beside it, and for version 2 the cost of a round trip in such stocks.

The rules are the research ledger's (CLAUDE.md): declared before running, frozen (a change is a new VERSION), TRAIN
2011-2018 and TEST from 2019, the verdict on TEST only, controls, and Benjamini-Hochberg q < 0.10 across every gated
test in the ledger (``research/run.py``). Results are written whatever they say.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from brain import config
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

VERSION = 1
UNIVERSE = "universe/us_large.json"
BENCH = "SPY"
ATTR = "release.earnings"
# The event kinds tested, and the direction each implies. guidance_cut (-1) is left out of version 1: the rules find
# 10 cuts among the 60 stocks' releases since 2011, 4 of them in TEST, too few to say anything.
EVENTS = {"guidance_raised": 1}
# Version 2: the small and micro caps (universe/us_small.json, us_micro.json; SEC ranks 1501-3000 and 3001-4500).
# Version 1 found the large caps' drift after a raise pointing the other way from the one declared, so version 2 asks
# either way (sign 0: a two-sided p). A round trip in these stocks costs far more than in the large caps; the cost
# is stated beside each result, since a drift smaller than it could not be acted on.
SUITES = [
    {"version": 1, "tag": None, "universe": None, "events": EVENTS},   # None: UNIVERSE (the 60 stocks)
    {"version": 2, "tag": "small", "universe": "universe/us_small.json", "events": {"guidance_raised": 0, "guidance_cut": 0},
     "cost_round_trip": 0.01},
    {"version": 2, "tag": "micro", "universe": "universe/us_micro.json", "events": {"guidance_raised": 0, "guidance_cut": 0},
     "cost_round_trip": 0.02},
]
EXCLUSIVE = ("guidance_raised", "guidance_cut")       # a release tagged with both is neither
WINDOWS = {"reaction": {"key": "car_3d", "days": "-1..+1", "gate": "report"},
           "drift": {"key": "drift_20d", "days": "+2..+21", "gate": "diff"}}
SPLIT = {"train": ["2011-01", "2018-12"], "test_start": "2019-01"}
TRIAL_SPLIT = {"train": ["2011-01", "2014-12"], "test": ["2015-01", "2018-12"]}
N_FLIP = 20000
N_BOOT = 2000
SEED = 13
ALPHA = 0.05
MIN_OTHERS = 3            # untagged releases a month needs to be compared with
MIN_MONTHS = 12           # TEST months with a tagged release; fewer is recorded as too few events
CONTROLS = {"n_placebo": 200, "n_flip": 2000, "fpr_max": 0.10, "oracle_p_max": 0.01}
BH = {"bh_q": 0.10, "bh_over": "every hypothesis, every gated forecast test, every gated link test and every gated "
                               "event test with a result in the ledger, withdrawn ones included"}
CLEAN = ("brain", "knowledge")
HERE = Path(__file__).resolve()


def test_id(event: str, window: str, version: int = VERSION, tag: str | None = None) -> str:
    return f"E{version}-{tag + '-' if tag else ''}{event}-{window}"


def tests(suites=None) -> list[dict]:
    """Every declared-able test: each suite's events in each window. A suite whose universe file is missing is left
    out (build it first: python -m brain universe --band small|micro)."""
    out = []
    for su in suites or SUITES:
        uni = su["universe"] or UNIVERSE
        if not run_universe_exists(uni):
            continue
        for e, s in su["events"].items():
            for w in WINDOWS:
                out.append({"id": test_id(e, w, su["version"], su["tag"]), "event": e, "window": w, "sign": s,
                            "version": su["version"], "suite": su["tag"], "universe": uni,
                            "cost_round_trip": su.get("cost_round_trip")})
    return out


def run_universe_exists(name: str) -> bool:
    from brain.research import run
    return run.universe_path(name).exists()


def code_files() -> list[Path]:
    """What the numbers run through: this file, the event rules, the release reader, the reaction windows, the
    verdicts and their correction, the trial rules, the bank's view and the price book."""
    from brain import bank, events, metrics
    from brain.research import forecast_tests, run, stats
    from brain.sources import releases
    from brain.web import events as tags
    return [HERE, *(Path(m.__file__).resolve() for m in (tags, releases, events, run, stats, forecast_tests, bank, metrics))]


def code_sha() -> str:
    h = hashlib.sha256()
    root = config.ROOT.resolve()
    for p in code_files():
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


def claim(t: dict) -> str:
    w = WINDOWS[t["window"]]
    if not t["sign"]:
        return (f"Earnings releases whose opening the rules tag {t['event']} differ from the month's other earnings "
                f"releases over days {w['days']}, either way (excess return over SPY; mean of monthly differences "
                f"!= 0, two-sided).")
    way = "beat" if t["sign"] > 0 else "trail"
    return (f"Earnings releases whose opening the rules tag {t['event']} {way} the month's other earnings releases "
            f"over days {w['days']} (excess return over SPY; mean of monthly differences "
            f"{'>' if t['sign'] > 0 else '<'} 0).")


VARYING = ("test", "code_sha", "data_asof", "trial")


def spec(t: dict, anchor: str, trial: bool, data_asof: str) -> dict:
    split = (dict(TRIAL_SPLIT) if trial else {"train": SPLIT["train"], "test": [SPLIT["test_start"], anchor]})
    w = WINDOWS[t["window"]]
    v2 = {"suite": t["suite"], "two_sided": not t["sign"], "cost_round_trip": t["cost_round_trip"]} \
        if t.get("version", VERSION) >= 2 else {}   # version 1's specs stay exactly as declared
    return {"kind": "event", "version": t.get("version", VERSION), "id": t["id"], "event": t["event"],
            "exclusive": list(EXCLUSIVE), "sign": t["sign"], "window": t["window"], "days": w["days"],
            "reaction_key": w["key"], "gate": w["gate"], **v2,
            "universe": t.get("universe") or UNIVERSE, "attr": ATTR, "bench": BENCH, **split, "n_flip": N_FLIP, "n_boot": N_BOOT,
            "seed": SEED, "alpha": ALPHA, "min_others": MIN_OTHERS, "min_months": MIN_MONTHS, "controls": CONTROLS,
            **BH, "claim": claim(t),
            "code_sha": code_sha(), "data_asof": data_asof, "trial": bool(trial)}


def spec_sha(s: dict) -> str:
    return hashlib.sha256(canon(s).encode()).hexdigest()[:16]


# ---------- the data ----------
def releases(bank: Bank, s: dict) -> list[dict]:
    """Every earnings release of the universe, with its tags (the rules applied now to its stored opening), its
    reaction windows (``brain/events.py``) and its month. Prices are read as the bank held them at data_asof."""
    from brain import events
    from brain.metrics import PriceBook
    from brain.research import run
    from brain.web import events as tags
    tickers = [t.upper() for t in run.load_universe(s["universe"])["tickers"]]
    book = PriceBook(bank, [*tickers, s["bench"]], retrieved_by=s.get("data_asof"))
    view = bank.view()
    out = []
    for t in tickers:
        for f in view.facts(t, s["attr"]):
            if s.get("data_asof") and f.retrieved_at and f.retrieved_at > s["data_asof"]:
                continue
            r = events.reaction(book, view, t, f, s["bench"])
            if r is None or r.get(s["reaction_key"]) is None:
                continue
            kinds = set(tags.tag((f.context or {}).get("opening", "")))
            out.append({"ticker": t, "month": r["date"][:7], "date": r["date"], "x": float(r[s["reaction_key"]]),
                        "kinds": kinds, "ref": f.ref})
    return out


def is_tagged(rel: dict, s: dict) -> bool:
    return s["event"] in rel["kinds"] and not (set(s["exclusive"]) - {s["event"]}) & rel["kinds"]


def monthly(rels: list[dict], s: dict, first: str, last: str) -> dict:
    """Per month in [first, last] with a tagged release and MIN_OTHERS others: the tagged mean, the others' mean
    and their difference."""
    by: dict[str, dict] = {}
    for r in rels:
        if first <= r["month"] <= last:
            by.setdefault(r["month"], {"tagged": [], "others": []})["tagged" if is_tagged(r, s) else "others"].append(r["x"])
    months = sorted(m for m, g in by.items() if g["tagged"] and len(g["others"]) >= s["min_others"])
    tag_mean = np.array([np.mean(by[m]["tagged"]) for m in months])
    oth_mean = np.array([np.mean(by[m]["others"]) for m in months])
    return {"months": months, "tagged": tag_mean, "others": oth_mean, "diff": tag_mean - oth_mean,
            "n_tagged": int(sum(len(by[m]["tagged"]) for m in months)),
            "n_others": int(sum(len(by[m]["others"]) for m in months)), "groups": {m: by[m] for m in months}}


# ---------- the statistics ----------
def flip_p(d: np.ndarray, sign: int, n: int, rng) -> float:
    """Sign-flip p of mean(d) (Monte Carlo, with the observed one counted): one-sided in ``sign``'s direction, or
    two-sided when ``sign`` is 0."""
    if len(d) == 0:
        return 1.0
    flips = rng.choice((-1.0, 1.0), size=(n, len(d)))
    null = (flips * d).mean(1)
    if not sign:
        return float((1 + np.sum(np.abs(null) >= abs(float(d.mean())) - 1e-15)) / (n + 1))
    obs = sign * float(d.mean())
    return float((1 + np.sum(sign * null >= obs - 1e-15)) / (n + 1))


def boot_interval(d: np.ndarray, n: int, rng) -> list[float]:
    if len(d) == 0:
        return [float("nan")] * 2
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))]


def measure(mo: dict, s: dict, rng) -> dict:
    d = mo["diff"]
    return {"months": len(d), "events": mo["n_tagged"], "others": mo["n_others"],
            "first": mo["months"][0] if mo["months"] else None, "last": mo["months"][-1] if mo["months"] else None,
            "diff": float(d.mean()) if len(d) else float("nan"), "interval90": boot_interval(d, s["n_boot"], rng),
            "p": flip_p(d, s["sign"], s["n_flip"], rng),
            "vs_spy": float(mo["tagged"].mean()) if len(d) else float("nan"),
            "others_vs_spy": float(mo["others"].mean()) if len(d) else float("nan")}


def controls(mo: dict, s: dict, rng) -> dict:
    """Placebo: each month's tagged releases replaced by as many untagged ones drawn at random; the test must reach
    p < 0.05 for at most ``fpr_max`` of the draws. Oracle: the month's best (or worst) releases as the tagged ones;
    the test must reach p < ``oracle_p_max``."""
    c = s["controls"]
    if not mo["months"]:
        return {"placebo_fpr": float("nan"), "n_placebo": 0, "oracle_p": 1.0, "pass": False}
    hits = 0
    for _ in range(c["n_placebo"]):
        diffs = []
        for m in mo["months"]:
            g = mo["groups"][m]
            k, oth = len(g["tagged"]), np.array(g["others"])
            if len(oth) - k < s["min_others"]:
                continue
            pick = rng.permutation(len(oth))
            diffs.append(oth[pick[:k]].mean() - oth[pick[k:]].mean())
        hits += flip_p(np.array(diffs), s["sign"], c["n_flip"], rng) < ALPHA
    oracle = []
    for m in mo["months"]:
        g = mo["groups"][m]
        allx = np.sort(np.array(g["tagged"] + g["others"]))
        k = len(g["tagged"])
        top = allx[-k:] if s["sign"] >= 0 else allx[:k]
        rest = allx[:-k] if s["sign"] >= 0 else allx[k:]
        oracle.append(top.mean() - rest.mean())
    o = flip_p(np.array(oracle), s["sign"], c["n_flip"], rng)
    fpr = hits / c["n_placebo"]
    return {"placebo_fpr": fpr, "n_placebo": c["n_placebo"], "oracle_p": o,
            "pass": bool(fpr <= c["fpr_max"] and o < c["oracle_p_max"])}


def evaluate(rels: list[dict], s: dict) -> dict:
    if not rels:
        raise Refused(f"{s['id']}: no earnings releases in the bank; collect them first (python -m brain releases)")
    rng = np.random.default_rng(s["seed"])
    out = {}
    for name in ("train", "test"):
        mo = monthly(rels, s, *s[name])
        out[name] = measure(mo, s, rng)
        if name == "test":
            ctl = controls(mo, s, rng)
    te = out["test"]
    thin = te["months"] < s.get("min_months", MIN_MONTHS)   # recorded, not refused: a declared test always gets a result
    # Two-sided (sign 0): either direction counts, and the result says which.
    sign_ok = bool(math.isfinite(te["diff"]) and (s["sign"] * te["diff"] > 0 or not s["sign"]))
    gate = {"sign_ok": sign_ok, "p_ok": bool(te["p"] < s["alpha"]), "enough": not thin}
    gate["pass"] = bool(s["gate"] == "diff" and sign_ok and gate["p_ok"] and ctl["pass"] and not thin)
    return {"metrics": out, "controls": ctl, "gate": gate, "p": te["p"], "thin": thin}


# ---------- declare and test ----------
def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=False)


def tree_dirty() -> list[str]:
    r = _git("status", "--porcelain", "--", *CLEAN)
    return [ln for ln in r.stdout.splitlines() if ln.strip()] if r.returncode == 0 else [f"git: {r.stderr.strip()}"]


def lineage_problem(commit: str | None) -> str | None:
    if not commit:
        return "the declaration has no commit"
    if _git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        return f"HEAD is not the declare commit {commit[:12]} or a descendant of it"
    root = config.ROOT.resolve()
    d = _git("diff", "--name-only", commit, "HEAD", "--", *[p.relative_to(root).as_posix() for p in code_files()])
    changed = [x for x in d.stdout.splitlines() if x.strip()]
    return f"code changed since the declare commit {commit[:12]}: {', '.join(changed)}" if changed else None


def committed_declarations() -> set[tuple[str, str]]:
    r = _git("show", "HEAD:research/ledger.jsonl")
    out = set()
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("op") == "declare" and rec.get("kind") == "event":
            out.add((rec.get("id"), rec.get("spec_sha")))
    return out


def _records(led) -> list[dict]:
    return [r for r in ledger.read(led) if r.get("kind") == "event"]


def anchor_month(bank: Bank) -> str:
    """The last month whose drift windows (21 sessions) have all closed: two months before SPY's latest session."""
    from brain.metrics import PriceBook
    book = PriceBook(bank, [BENCH])
    if not book.has(BENCH):
        raise Refused(f"no {BENCH} prices in the bank")
    y, m = map(int, str(book.dates[BENCH][-1])[:7].split("-"))
    m -= 2
    while m < 1:
        y, m = y - 1, m + 12
    return f"{y:04d}-{m:02d}"


def declare(bank: Bank, *, trial: bool = False, log=print) -> list[dict]:
    trial = is_trial(trial)
    led, _ = paths(trial)
    if not trial and (dirty := tree_dirty()):
        raise Refused("commit brain/ and knowledge/ first; a declaration pins the committed code: " + "; ".join(dirty))
    anchor = TRIAL_SPLIT["test"][1] if trial else anchor_month(bank)
    asof = now_iso()
    have = {r["id"]: r["spec"] for r in _records(led) if r["op"] == "declare"}
    new = []
    for t in tests():
        s = spec(t, anchor, trial, asof)
        if t["id"] in have:
            if {k: v for k, v in have[t["id"]].items() if k not in VARYING} != {k: v for k, v in s.items() if k not in VARYING}:
                raise Refused(f"{t['id']} was declared with a different design; raise VERSION for new ids")
            continue
        rec = {"op": "declare", "kind": "event", "id": t["id"], "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
               **ledger.git_state()}
        ledger.append(rec, led)
        new.append(rec)
        if log:
            log(f"declared {t['id']} ({rec['spec_sha']}): TEST {s['test'][0]}..{s['test'][1]}, {s['gate']}")
    return new


def test(bank: Bank, ids=None, *, trial: bool = False, log=print) -> list[dict]:
    from brain.research import run
    trial = is_trial(trial)
    led, res = paths(trial)
    recs = _records(led)
    decls = {r["id"]: r for r in recs if r["op"] == "declare"}
    done = {r["id"] for r in recs if r["op"] == "result"}
    gone = ledger.withdrawn(led)
    want = list(ids) if ids else [i for i in decls if i not in done and i not in gone]   # E1 already has its results
    if not want:
        if log:
            log("nothing to test")
        return []
    if not trial and (dirty := tree_dirty()):
        raise Refused("the working tree has changes under brain/ or knowledge/: " + "; ".join(dirty))
    for i in want:
        d = decls.get(i)
        if d is None:
            raise Refused(f"{i} is not declared")
        if spec_sha(d["spec"]) != d["spec_sha"]:
            raise Refused(f"{i}: its declared spec does not match its spec_sha")
        if trial and (not d["spec"].get("trial") or d["spec"]["test"][1] > TRIAL_SPLIT["test"][1]):
            raise Refused(f"{i} is not a trial declaration: a trial reads TRAIN years only")
        if not trial:
            if d["spec"].get("trial"):
                raise Refused(f"{i} is a trial declaration")
            if i in done:
                raise Refused(f"{i} already has a result; a rerun needs research withdraw and a new id")
            if why := lineage_problem(d.get("commit")):
                raise Refused(f"{i}: {why}")
            if d["spec"]["code_sha"] != code_sha():
                raise Refused(f"{i}: the code is not the declared code")
            if (i, d["spec_sha"]) not in committed_declarations():
                raise Refused(f"{i}: commit the ledger with its declaration before running it")
    cache: dict[tuple, list] = {}
    results = {}
    for i in want:   # every result computed before any is written
        s = decls[i]["spec"]
        key = (s["universe"], s["reaction_key"], s.get("data_asof"))
        cache.setdefault(key, releases(bank, s))
        results[i] = evaluate(cache[key], s)
    out = []
    for i, r in results.items():
        s = decls[i]["spec"]
        rec = {"op": "result", "kind": "event", "id": i, "spec_sha": decls[i]["spec_sha"], "gate_type": s["gate"],
               **r, "at": now_iso(), **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            te = r["metrics"]["test"]
            log(f"{i}: {te['events']} events in {te['months']} months; vs other releases {te['diff'] * 100:+.2f} points "
                f"(90% {te['interval90'][0] * 100:+.2f}..{te['interval90'][1] * 100:+.2f}), p {te['p']:.4f}; "
                f"vs SPY {te['vs_spy'] * 100:+.2f}")
    run.report(led, res)
    if log:
        log(f"wrote {res}" + ("" if trial else "; commit the ledger and RESULTS.md, whatever they say"))
    return out


def _cmd(a) -> int:
    trial = is_trial(a.trial)
    if a.action == "declare":
        new = declare(Bank(), trial=trial)
        print(f"{len(new)} declared" + ("" if trial or not new else "; commit the ledger before `brain events test`"))
        return 0
    test(Bank(), a.ids or None, trial=trial)
    return 0


def add_cli(sub) -> None:
    s = sub.add_parser("events", help="declared tests of corporate events in earnings releases: declare | test [ID...]")
    s.add_argument("action", choices=["declare", "test"])
    s.add_argument("ids", nargs="*")
    s.add_argument("--trial", action="store_true", help="TRAIN years only, in the trial ledger")
    s.set_defaults(fn=_cmd)
