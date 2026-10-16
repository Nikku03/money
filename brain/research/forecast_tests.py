"""Declared forecast tests: what F1 and F1-llm are tested on, pinned before they run (docs/FORECAST.md,
sections 7 and 8).

``brain forecast declare`` writes each test's full spec into the research ledger: the test as written below,
the universe and its sha, LEARN, RANGE, the feature names, the anchor and ``prices_asof`` (pinned now), the
gate's details and ``code_sha`` (every file under ``brain/forecast/``, the modules the features read, this
file and the market primer). For F1d it also pins the local model: its name, its weights' digest, its stated
training cutoff, the primer's and the prompt's shas. It refuses unless the working tree is clean for
``brain/`` and the primer.

``brain forecast test ID...`` refuses unless the test is declared with this very spec, the code is the
declared code (clean tree, HEAD the declare commit or a descendant that leaves it unchanged), F1d's model
is the declared one, and no result exists yet for that id and spec. A rerun needs ``research withdraw``
and a new id.

**Trials.** With ``--trial``, a ledger other than ``research/ledger.jsonl`` (BRAIN_LEDGER) or a MONEY_HOME
other than ``var/``, everything runs on TRAIN years only: every anchor becomes the last session of 2018, the
gated windows start in 2015, and the trial ledger is written (never the committed one). A trial is plumbing,
not evidence: it may run on an uncommitted tree, and says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from brain import config
from brain.bank import canon
from brain.forecast import evaluate, learn
from brain.research import ledger
from brain.timeutil import now_iso

FORECAST_TESTS = [
    {"id": "F1a-w5-recent", "forecaster": "F1", "horizon": 5, "periods": 8, "gate": "report",
     "retrain": "once, at cutoff = the first forecast date d_8",
     "claim": "Retest on the last 8 non-overlapping weeks (about two months, held out): how F1's chance-to-beat-SPY "
              "forecasts and ranges did. Descriptive: detects mean IC >= 0.15 at 80% power."},
    {"id": "F1b-w5-test", "forecaster": "F1", "horizon": 5, "start": "2019-01-01", "end": "<F1a's d_8>", "gate": "ic",
     "retrain": "every 26 periods (about half a year), cutoff = that forecast date",
     "claim": "From 2019 to the start of F1a's window, F1's weekly forecasts rank stocks by what they then did beyond size: "
              "mean size-neutral IC > 0."},
    {"id": "F1c-m21-test", "forecaster": "F1", "horizon": 21, "start": "2019-01-01", "end": "<anchor>", "gate": "ic",
     "retrain": "every 6 periods",
     "claim": "From 2019 to the anchor, F1's 21-session forecasts have mean size-neutral IC > 0."},
    {"id": "F1d-llm-w5-recent", "forecaster": "F1-llm", "horizon": 5, "periods": 8, "gate": "report",
     "tickers": ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "JPM", "CVX", "JNJ", "PG", "WMT", "KO", "CAT"],
     "retrain": "F1a's model",
     "claim": "Retest on F1a's 8 weeks and these 12 stocks: does the local model's probability, given F1's forecast, a "
              "point-in-time snapshot and the market primer, have a lower Brier score than F1 alone? Descriptive."},
]
BY_ID = {t["id"]: t for t in FORECAST_TESTS}
RECENT = "F1a-w5-recent"          # its first date ends F1b's window, and its model is F1d's
UNIVERSE = "universe/us_broad.json"
BENCH = "SPY"
PRIMER = config.ROOT / "docs" / "MARKET_PRIMER.md"
CODE_DIR = config.ROOT / "brain" / "forecast"
CLEAN = ("brain", "docs/MARKET_PRIMER.md")   # declare and test need these clean in git
DETECTS = {"F1a-w5-recent": 0.15}             # the mean IC each descriptive test can detect at 80% power
LLM_FIXED = {"temperature": 0, "seed": 7}

GATE_DETAILS = {
    "ic": {"metric": "mean over forecast dates of the size-neutral IC (Spearman of p_beat's rank residual on "
                     "log_mktcap's rank, against excess)",
           "pass": "mean > 0 and p < 0.05 and BH q < 0.10 and both controls pass",
           "test": "one-sided date-level sign-flip on IC_1..IC_K: exact for K <= 16, else 20000 draws seeded 7",
           "sign_flip": evaluate.SIGN_FLIP, "p_lt": evaluate.GATE_P, "bh_q_lt": 0.10,
           "bh_over": "every hypothesis and gated forecast test with a result in the ledger, withdrawn ones included",
           "controls": {"oracle": "p_beat = the realised excess rank; must reach the test's minimum attainable p",
                        "placebo": "F1's p_beat with each date's sign flipped by a fair coin; the test must reject "
                                   "in 1%-10% of the draws", "placebo_draws": evaluate.PLACEBO},
           "diagnostic": {"within_date_permutation": evaluate.PERMUTATION,
                          "note": "reported only: it ignores how stocks move together"},
           "trial_start": evaluate.TRIAL_START},
    "report": {"verdict": "REPORT: descriptive, outside BH",
               "interval": "each mean's date-level 90% interval, mean +- 1.645 sd / sqrt(K)",
               "label": "inconclusive when the size-neutral IC's interval spans 0 (F1d: the Brier difference's), "
                        "else positive or negative"},
}


class Refused(SystemExit):
    """A declare or test that the rules do not allow; the message says which rule."""


# ---------- where a run writes ----------
DEFAULT_LEDGER = config.ROOT / "research" / "ledger.jsonl"
DEFAULT_RESULTS = config.ROOT / "research" / "RESULTS.md"
DEFAULT_HOME = config.ROOT / "var"


def _home() -> Path:
    return Path(os.environ.get("MONEY_HOME") or DEFAULT_HOME)


def _same(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


def is_trial(flag: bool = False) -> bool:
    """A trial: asked for, or the ledger or the home is not the committed one."""
    return bool(flag) or not _same(config.LEDGER, DEFAULT_LEDGER) or not _same(_home(), DEFAULT_HOME)


def paths(trial: bool) -> tuple[Path, Path]:
    """(ledger, RESULTS.md) a run reads and writes. A trial never gets the committed files: it takes
    BRAIN_LEDGER / BRAIN_RESULTS when they point elsewhere, else ``<home>/trial/``."""
    if not trial:
        return Path(config.LEDGER), Path(config.RESULTS)
    led = Path(config.LEDGER) if not _same(config.LEDGER, DEFAULT_LEDGER) else _home() / "trial" / "ledger.jsonl"
    res = Path(config.RESULTS) if not _same(config.RESULTS, DEFAULT_RESULTS) else _home() / "trial" / "RESULTS.md"
    if _same(led, DEFAULT_LEDGER) or _same(res, DEFAULT_RESULTS):  # belt and braces
        raise Refused("a trial must not write the committed ledger or RESULTS.md")
    return led, res


# ---------- pinning ----------
def dep_files() -> list[Path]:
    """The modules outside brain/forecast/ that compute a feature or the label (``features._DEPS``: metrics,
    events, claims, sources/insiders, research/signals, series). A change to one changes F1's numbers."""
    from brain.forecast import features  # the real module, even when a test stands in for it elsewhere
    return [Path(m.__file__).resolve() for m in features._DEPS]


def code_files() -> list[Path]:
    """Every file under brain/forecast/ (bytecode caches aside), the modules the features read, this file and
    the primer."""
    out = [p for p in sorted(CODE_DIR.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
           and p.suffix not in (".pyc", ".pyo")]
    return [*out, *dep_files(), Path(__file__).resolve(), PRIMER]


def code_sha() -> str:
    """sha256[:16] over each code file's path (from the repo root) and bytes; a missing primer counts as missing."""
    h = hashlib.sha256()
    for p in code_files():
        rel = p.resolve().relative_to(config.ROOT.resolve()).as_posix() if p.resolve().is_relative_to(config.ROOT.resolve()) else p.name
        h.update(rel.encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest() if p.exists() else b"missing")
    return h.hexdigest()[:16]


def _code_paths() -> list[str]:
    root = config.ROOT.resolve()
    return sorted({p.resolve().relative_to(root).as_posix() for p in code_files() if p.resolve().is_relative_to(root)}
                  | {"brain/forecast"})


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=False)


def tree_dirty() -> list[str]:
    """``git status --porcelain`` lines for brain/ and the primer (untracked files count). ['git: ...'] when
    git cannot say."""
    try:
        r = _git("status", "--porcelain", "--", *CLEAN)
    except OSError as e:
        return [f"git: {e}"]
    if r.returncode != 0:
        return [f"git: {r.stderr.strip() or 'status failed'}"]
    return [line for line in r.stdout.splitlines() if line.strip()]


def lineage_problem(commit: str | None) -> str | None:
    """None when HEAD is ``commit`` or a descendant that leaves every code file unchanged; else why not."""
    if not commit:
        return "the declaration has no commit (git was not available at declare)"
    try:
        if _git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
            return f"HEAD is not the declare commit {commit[:12]} or a descendant of it"
        d = _git("diff", "--name-only", commit, "HEAD", "--", *_code_paths())
    except OSError as e:
        return f"git: {e}"
    if d.returncode != 0:
        return f"git diff failed: {d.stderr.strip()}"
    changed = [x for x in d.stdout.splitlines() if x.strip()]
    return f"code changed since the declare commit {commit[:12]}: {', '.join(changed)}" if changed else None


def _sha(obj) -> str:
    return hashlib.sha256(canon(obj).encode()).hexdigest()[:16]


def _think():
    from brain.forecast import think
    return think


def llm_pins(think=None) -> dict:
    """The primer's and the prompt's shas and the snapshot version, as F1d's spec pins them."""
    think = think or _think()

    def pick(fn, const, fallback):
        f = getattr(think, fn, None)
        if callable(f):
            return str(f())
        v = getattr(think, const, None)
        return str(v) if v else fallback()

    def prompt_fallback():
        return _sha(think.PROMPT)

    def primer_fallback():
        return hashlib.sha256(PRIMER.read_bytes()).hexdigest()[:16] if PRIMER.exists() else "missing"
    return {"primer_sha": pick("primer_sha", "PRIMER_SHA", primer_fallback),
            "prompt_sha": pick("prompt_sha", "PROMPT_SHA", prompt_fallback),
            "snapshot_version": getattr(think, "SNAPSHOT_VERSION", None)}


def _client(client=None):
    if client is not None:
        return client
    from brain import llm
    return llm.Client.from_env()


def llm_identity(client=None, think=None) -> dict:
    """{"model", "digest"} of the live local model; Refused when it cannot be read."""
    think = think or _think()
    try:
        ident = think.llm_identity(_client(client)) or {}
    except Exception as e:  # noqa: BLE001 - an unreachable server is a refusal, not a crash
        raise Refused(f"F1d: cannot read the local model's identity ({type(e).__name__}: {e})") from None
    if not ident.get("model") or not ident.get("digest"):
        raise Refused(f"F1d: the local model's name and weights digest are needed; the server gave {ident!r}")
    return {"model": str(ident["model"]), "digest": str(ident["digest"])}


# ---------- the spec ----------
def universe_tickers() -> list[str]:
    from brain.research import run
    return [t for t in (str(x).upper() for x in run.load_universe(UNIVERSE)["tickers"]) if t != BENCH]


def calendar(bank, prices_asof: str):
    """SPY's sessions in the book at ``prices_asof``."""
    from brain.metrics import PriceBook
    book = PriceBook(bank, [BENCH], prices_asof)
    if not book.has(BENCH):
        raise Refused(f"no {BENCH} price snapshot known at {prices_asof}")
    return book.dates[BENCH]


def _features_pin() -> dict:
    features = evaluate._features()
    return {"version": features.FEATURES_VERSION, "names": [n for n, _, _ in features.FEATURES]}


def fill(test: dict, *, anchor: str, prices_asof: str, code: str, window: dict, trial: bool, llm: dict | None = None) -> dict:
    """The full declared spec of ``test``: what the spec sha covers."""
    from brain.research import run
    gate = dict(GATE_DETAILS[test["gate"]])
    if test["id"] in DETECTS:
        gate["detects_mean_ic"] = DETECTS[test["id"]]
    spec = {**test, "kind": "forecast", "universe": UNIVERSE, "universe_sha": run.universe_sha(UNIVERSE),
            "LEARN": learn.LEARN, "RANGE": learn.RANGE, "FEATURES": _features_pin(), "anchor": anchor,
            "prices_asof": prices_asof, "code_sha": code, "gate_details": gate, "window": window, "trial": bool(trial)}
    if test["forecaster"] == "F1-llm":
        spec["llm"] = llm
    return json.loads(json.dumps(spec))   # tuples become lists: the sha is the stored spec's


def spec_sha(spec: dict) -> str:
    return _sha(spec)


def window(test: dict, cal, anchor: str, trial: bool) -> dict:
    pl = evaluate.plan(test, cal, anchor, trial=trial, recent_periods=int(BY_ID[RECENT]["periods"]))
    return {k: pl[k] for k in ("first", "last", "last_exit", "n_dates", "start")}


def _forecast_records(path) -> list[dict]:
    return [r for r in ledger.read(path) if r.get("kind") == "forecast"]


def declared(test_id: str, path) -> dict | None:
    for r in _forecast_records(path):
        if r["op"] == "declare" and r["id"] == test_id:
            return r
    return None


def _say(log, msg):
    if log:
        log(msg)


# ---------- declare ----------
def declare(bank, *, llm_cutoff: str | None = None, trial: bool = False, ids=None, client=None, think=None,
            log=print) -> list[dict]:
    """Append a declaration for each forecast test not yet declared in the ledger (all, or ``ids``)."""
    trial = is_trial(trial)
    led, _ = paths(trial)
    dirty = tree_dirty()
    if dirty and not trial:
        raise Refused("declare refused: the working tree is not clean for brain/ and docs/MARKET_PRIMER.md "
                      f"({'; '.join(dirty[:5])}). Commit first: the declaration pins the committed code.")
    if dirty:
        _say(log, f"trial: the tree is not clean ({len(dirty)} paths); the declaration pins the code as it is")
    known = {r["id"]: r for r in _forecast_records(led) if r["op"] == "declare"}
    for i in ids or ():
        if i not in BY_ID:
            raise Refused(f"no forecast test {i!r}; declared tests: {', '.join(BY_ID)}")
    todo = [t for t in FORECAST_TESTS if t["id"] not in known and (not ids or t["id"] in ids)]
    if not todo:
        return []
    first = known.get(RECENT) or next(iter(known.values()), None)
    if first:  # one anchor for the family: F1b ends where F1a starts, F1d is F1a's weeks
        anchor, asof = first["spec"]["anchor"], first["spec"]["prices_asof"]
        cal = calendar(bank, asof)
    else:
        asof = now_iso()
        cal = calendar(bank, asof)
        if trial:
            anchor = evaluate.trial_anchor(cal)
        else:
            from brain.metrics import PriceBook
            anchor = evaluate.find_anchor(PriceBook(bank, [*universe_tickers(), BENCH], asof), universe_tickers(), BENCH)
    if trial and anchor > evaluate.TRIAL_LAST:
        raise Refused(f"trial ledger {led} holds a declaration anchored at {anchor}, after {evaluate.TRIAL_LAST}")
    llm = None
    if any(t["forecaster"] == "F1-llm" for t in todo):
        if not llm_cutoff:
            todo = [t for t in todo if t["forecaster"] != "F1-llm"]
            _say(log, "F1d not declared: it needs --llm-cutoff YYYY-MM (the local model's stated training-data cutoff)")
        else:
            if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", llm_cutoff):
                raise Refused(f"--llm-cutoff is YYYY-MM, not {llm_cutoff!r}")
            d8 = window(BY_ID[RECENT], cal, anchor, trial)["first"]
            if not llm_cutoff < d8[:7]:
                msg = (f"F1d: the model's training cutoff {llm_cutoff} is not earlier than F1a's first forecast "
                       f"date {d8}, so it may have read about the weeks it would forecast")
                if not trial:
                    raise Refused(f"declare refused. {msg}.")
                _say(log, f"trial: {msg}; declared anyway (a trial is not evidence)")
            think = think or _think()
            llm = {**llm_identity(client, think), "cutoff": llm_cutoff, **LLM_FIXED, **llm_pins(think)}
    code = code_sha()
    out = []
    for t in todo:
        spec = fill(t, anchor=anchor, prices_asof=asof, code=code, window=window(t, cal, anchor, trial), trial=trial,
                    llm=llm)
        rec = {"op": "declare", "kind": "forecast", "id": t["id"], "spec_sha": spec_sha(spec), "spec": spec,
               "at": now_iso(), **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
    return out


# ---------- test ----------
def check(test_id: str, *, trial: bool, led, bank, client=None, think=None, cal_cache: dict | None = None) -> dict:
    """The declaration of ``test_id`` when every rule of section 8 allows a run now; else Refused."""
    if test_id not in BY_ID:
        raise Refused(f"no forecast test {test_id!r}; declared tests: {', '.join(BY_ID)}")
    d = declared(test_id, led)
    if d is None:
        raise Refused(f"{test_id} is not declared in {led}. Run `brain forecast declare`, commit, then test.")
    sp = d["spec"]
    code = code_sha()
    if code != sp.get("code_sha"):
        raise Refused(f"{test_id}: the code has changed since it was declared (code_sha {code}, declared "
                      f"{sp.get('code_sha')}). A changed forecaster is a new test: declare a new id.")
    cal_cache = {} if cal_cache is None else cal_cache
    if sp["prices_asof"] not in cal_cache:
        cal_cache[sp["prices_asof"]] = calendar(bank, sp["prices_asof"])
    cal = cal_cache[sp["prices_asof"]]
    now = fill(BY_ID[test_id], anchor=sp["anchor"], prices_asof=sp["prices_asof"], code=code,
               window=window(BY_ID[test_id], cal, sp["anchor"], bool(sp.get("trial"))), trial=bool(sp.get("trial")),
               llm=sp.get("llm"))
    if spec_sha(now) != d["spec_sha"] or spec_sha(sp) != d["spec_sha"]:
        changed = sorted(k for k in set(now) | set(sp) if now.get(k) != sp.get(k))
        raise Refused(f"{test_id}: its spec no longer matches the declaration (changed: {', '.join(changed) or 'sha'}). "
                      "Declared specs are frozen: declare a new id.")
    if not trial:
        dirty = tree_dirty()
        if dirty:
            raise Refused(f"{test_id}: the working tree is not clean for brain/ and docs/MARKET_PRIMER.md "
                          f"({'; '.join(dirty[:5])})")
        why = lineage_problem(d.get("commit"))
        if why:
            raise Refused(f"{test_id}: {why}")
    if BY_ID[test_id]["forecaster"] == "F1-llm":
        think = think or _think()
        want = sp.get("llm") or {}
        ident = llm_identity(client, think)
        pins = llm_pins(think)
        for k, v in [*ident.items(), *pins.items()]:
            if str(want.get(k)) != str(v):
                raise Refused(f"{test_id}: the declared {k} was {want.get(k)!r}; now it is {v!r}. "
                              "A different model, primer or prompt is a new test.")
    for r in _forecast_records(led):
        if r["op"] == "result" and r["id"] == test_id and r["spec_sha"] == d["spec_sha"]:
            raise Refused(f"{test_id} already has a result for spec {d['spec_sha']} ({r['at'][:10]}). One result per "
                          "test: withdraw it (`research withdraw`) and declare a new id to run again.")
    return d


def test(bank, ids, *, trial: bool = False, client=None, think=None, log=print) -> list[dict]:
    """Run the declared tests ``ids`` once each, append their results and regenerate RESULTS.md."""
    from brain.research import run as research_run
    trial = is_trial(trial)
    led, res = paths(trial)
    if not ids:
        raise Refused(f"name the tests to run: {', '.join(BY_ID)}")
    cals: dict = {}
    decls = {i: check(i, trial=trial, led=led, bank=bank, client=client, think=think, cal_cache=cals)
             for i in dict.fromkeys(ids)}
    tickers = universe_tickers()
    cache: dict = {}
    out = []
    try:
        _run_all(bank, decls, cals, tickers, cache, out, trial=trial, led=led, client=client, think=think, log=log)
    finally:
        if out:  # whatever was recorded is reported, even if a later test stopped with an error
            research_run.report(ledger_path=led, results_path=res)
            _say(log, f"-> {res}")
    return out


def _run_all(bank, decls, cals, tickers, cache, out, *, trial, led, client, think, log) -> None:
    for test_id, d in decls.items():
        sp, t = d["spec"], BY_ID[test_id]
        cal = cals[sp["prices_asof"]]
        anchor = evaluate.trial_anchor(cal) if trial else sp["anchor"]
        _say(log, f"{test_id}: anchor {anchor}{' (trial: TRAIN years only)' if trial else ''}, prices as of "
                  f"{sp['prices_asof']}")
        kw = {"anchor": anchor, "prices_asof": sp["prices_asof"], "tickers": tickers, "cal": cal, "trial": trial,
              "log": log, "cache": cache}
        if t["forecaster"] == "F1-llm":
            body = evaluate.run_llm(bank, t, client=_client(client), think=think, **kw)
        else:
            body = evaluate.run(bank, t, recent_periods=int(BY_ID[RECENT]["periods"]), detects=DETECTS.get(test_id), **kw)
        rec = {"op": "result", "kind": "forecast", "id": test_id, "spec_sha": d["spec_sha"], "at": now_iso(),
               **ledger.git_state(), "gate_type": t["gate"], "anchor": anchor, "trial": bool(trial), **body}
        rec = _finite(json.loads(json.dumps(rec, default=_jsonable)))
        ledger.append(rec, led)
        out.append(rec)
        _say(log, summary(rec))


def _finite(o):
    """NaN and infinities become null: the ledger is plain JSON."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def _jsonable(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON: {type(o).__name__}")


def _f(x, fmt="+.3f"):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def summary(rec: dict) -> str:
    """One line on a recorded result."""
    m = rec["metrics"]
    if rec["id"].startswith("F1d"):
        c = m["comparison"]
        return (f"{rec['id']}: Brier F1 minus model {_f(c['mean'], '+.4f')} per date (90% {_f(c['lo90'], '+.4f')}.."
                f"{_f(c['hi90'], '+.4f')}), p {_f(c['p'], '.3f')}; {m['llm_failed']} of {m['llm_asked']} answers "
                f"failed; {c['label']}")
    sn, rg = m["ic_size_neutral"], m["range"]
    head = (f"{rec['id']}: {rec['n_dates']} dates, {rec['n_samples']} samples; IC {_f(m['ic']['mean'])}, "
            f"size-neutral IC {_f(sn['mean'])}")
    tail = (f"; 68% band held {_f(rg['cover68']['mean'], '.1%')}, 90% band {_f(rg['cover90']['mean'], '.1%')}")
    if rec.get("gate_type") == "ic":
        c = rec["controls"]
        return (head + f" (p {_f(rec.get('p'), '.4f')}); controls {'pass' if c['pass'] else 'FAIL'} (oracle p "
                f"{_f(c['oracle_p'], '.5f')}, placebo {_f(c['placebo_reject_rate'], '.1%')})" + tail)
    return head + f" (90% {_f(sn['lo90'])}..{_f(sn['hi90'])}), {m.get('label')}" + tail


# ---------- the command line ----------
WORDS = ("declare", "test")


def add_arguments(p: argparse.ArgumentParser) -> None:
    """The options ``brain forecast declare|test`` reads, on the product's ``forecast`` parser."""
    p.add_argument("--llm-cutoff", metavar="YYYY-MM",
                   help="declare: the local model's stated training-data cutoff (F1d is declared only with it)")
    p.add_argument("--trial", action="store_true",
                   help="declare/test on TRAIN years only (anchor: the last session of 2018), in the trial ledger")


def cmd(a) -> int:
    """``brain forecast declare [ID...] [--llm-cutoff YYYY-MM] [--trial]`` or ``brain forecast test ID... [--trial]``."""
    from brain.bank import Bank
    words = list(getattr(a, "words", None) or [])
    if not words or words[0] not in WORDS:
        print("usage: brain forecast declare [--llm-cutoff YYYY-MM] [--trial] | brain forecast test ID... [--trial]",
              file=sys.stderr)
        return 2
    action, ids = words[0], words[1:]
    trial = is_trial(getattr(a, "trial", False))
    if trial:
        led, res = paths(True)
        print(f"trial: TRAIN years only; ledger {led}, results {res}", file=sys.stderr)
    if action == "declare":
        new = declare(Bank(), llm_cutoff=getattr(a, "llm_cutoff", None), trial=trial, ids=ids or None)
        for r in new:
            w = r["spec"]["window"]
            print(f"declared {r['id']} ({r['spec_sha']}): anchor {r['spec']['anchor']}, {w['n_dates']} dates "
                  f"{w['first']}..{w['last']}")
        if not new:
            print("nothing new to declare")
        elif not trial:
            print("Commit the ledger before `brain forecast test`.")
        return 0
    test(Bank(), ids, trial=trial)
    return 0


def add_cli(target) -> object:
    """Register ``brain forecast declare|test``. Given the product's ``forecast`` parser (an ArgumentParser),
    add this module's options to it and return ``cmd`` for its handler to call when the first word is in
    ``WORDS``. Given the top-level subparsers, register a ``forecast`` command of its own (declare and test
    only)."""
    if isinstance(target, argparse.ArgumentParser):
        add_arguments(target)
        return cmd
    p = target.add_parser("forecast", help="declare or run the forecast tests (docs/FORECAST.md)")
    p.add_argument("words", nargs="*", metavar="declare | test ID...")
    add_arguments(p)
    p.set_defaults(fn=cmd)
    return cmd
