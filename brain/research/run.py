"""Run declared hypotheses, with controls, and write research/RESULTS.md.

Each hypothesis names its universe (``universe/us_large.json`` unless it says otherwise).
One panel is built per universe, with its own controls, so adding hypotheses on a new
universe leaves the H1-H5 panel, and every number computed on it, exactly as it was.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from brain import config
from brain.bank import Bank
from brain.research import harness, ledger, signals
from brain.research.hypotheses import COMMON, GATE, HYPOTHESES, SPLIT
from brain.research.panel import build
from brain.research.stats import bh_adjust
from brain.timeutil import now_iso


def universe_path(name: str | None = None):
    """The file behind a hypothesis' ``universe`` (the default one is ``config.UNIVERSE``)."""
    from brain import universe
    if name is None or name == COMMON["universe"]:
        return config.UNIVERSE
    if name == "universe/us_broad.json":
        return universe.BROAD
    return config.ROOT / name


def load_universe(name: str | None = None) -> dict:
    return json.loads(universe_path(name).read_text())


def universe_sha(name: str | None = None) -> str:
    return hashlib.sha256(universe_path(name).read_bytes()).hexdigest()[:16]


def masks(panel) -> tuple[np.ndarray, np.ndarray]:
    mo = panel.months
    start = mo >= SPLIT["first_rebalance"]
    train = np.flatnonzero(start & (mo <= SPLIT["train_end"]))
    test = np.flatnonzero(mo >= SPLIT["test_start"])
    return train, test


def controls(panel, test, n_random: int = 200) -> dict:
    """Positive control (oracle) must win; random signals redrawn every month must pass at about the
    nominal rate. Those two are the declared gate.

    Also reported, outside the gate: as many random scores held fixed per stock in every month, as a
    slow-moving signal (fundamentals) is. The null redraws its holdings every month, so it ignores how
    long real holdings persist; for such a signal it is too narrow and its p-values too small. The
    rate at which these reach p < 0.05 says by how much, on this universe."""
    M = len(panel.entry)
    N = panel.R.shape[1]
    oracle = np.vstack([panel.R, np.full((1, N), np.nan)])
    kw = dict(frac=COMMON["frac"], cost_bps=COMMON["cost_bps"])
    o = harness.evaluate(oracle[:M], panel.R, panel.listed[:M], test, n_perm=2000, seed=1, **kw)
    rng = np.random.default_rng(12345)
    ps = []
    for s in range(n_random):
        S = rng.random((M, N))
        ps.append(harness.evaluate(S, panel.R, panel.listed[:M], test, n_perm=500, seed=100 + s, **kw)["perm_p"])
    fpr = float(np.mean(np.array(ps) < 0.05))
    fixed = np.random.default_rng(54321)  # its own stream: the monthly draws above stay as they were
    pf = []
    for s in range(n_random):
        S = np.tile(fixed.random(N), (M, 1))
        pf.append(harness.evaluate(S, panel.R, panel.listed[:M], test, n_perm=500, seed=5000 + s, **kw)["perm_p"])
    ok = o["perm_p"] < 0.001 and 0.01 <= fpr <= 0.10
    return {"oracle_test_perm_p": o["perm_p"], "oracle_test_mean_gross_excess": o["mean_gross_excess"],
            "random_false_positive_rate": fpr, "n_random": n_random,
            "persistent_random_false_positive_rate": float(np.mean(np.array(pf) < 0.05)), "pass": bool(ok)}


def _check_declared(hyps: list[dict]) -> None:
    for h in hyps:
        d = ledger.declared(h["id"])
        if d is None or d["spec_sha"] != ledger.spec_sha(h):
            raise SystemExit(f"{h['id']} is not declared with this spec. Run `python -m brain research declare` "
                             f"and commit the ledger first.")
        if h.get("universe_sha") and universe_sha(h.get("universe")) != h["universe_sha"]:
            raise SystemExit(f"{h['id']} was declared on {h['universe']} with sha {h['universe_sha']}; the file "
                             f"there now differs. A rebuilt universe is a new experiment: declare a new id.")


def run(bank: Bank, ids: list[str] | None = None, n_random: int = 200, asof: str | None = None) -> list[dict]:
    """Run the declared hypotheses (all, or ``ids``). ``asof`` pins the price snapshots to those known
    at that time, so a past run can be reproduced from a bank that has newer prices since."""
    hyps = [h for h in HYPOTHESES if not ids or h["id"] in ids]
    _check_declared(hyps)
    groups: dict[str, list[dict]] = {}
    for h in hyps:
        groups.setdefault(ledger.spec(h)["universe"], []).append(h)
    out = []
    for name, group in groups.items():
        out += _run_universe(bank, name, group, n_random, asof)
    report()
    return out


def _run_universe(bank: Bank, name: str, hyps: list[dict], n_random: int, asof: str | None) -> list[dict]:
    uni = load_universe(name)
    panel = build(bank, uni["tickers"], asof=asof)
    M = len(panel.entry)
    train, test = masks(panel)
    print(f"panel ({name}): {len(panel.tickers)} tickers, months {panel.months[train[0]]}..{panel.months[test[-1]]} "
          f"(train {len(train)}, test {len(test)})")

    ctl = controls(panel, test, n_random)
    print(f"controls: oracle p={ctl['oracle_test_perm_p']:.4f}, random FPR={ctl['random_false_positive_rate']:.3f} "
          f"-> {'PASS' if ctl['pass'] else 'FAIL'} (fixed per stock, not gated: "
          f"FPR={ctl['persistent_random_false_positive_rate']:.3f})")

    known = {**signals.HAND, **signals.PER_TICKER}
    need = sorted({h["signal"] for h in hyps if h["signal"] in known} | {f for h in hyps for f in h.get("features", [])})
    feats = signals.compute(bank, panel, need)

    spy = pd.Series(panel.book.adj["SPY"], index=panel.book.dates["SPY"]).reindex(panel.dates).to_numpy()
    spy_r = spy[panel.exit] / spy[panel.entry] - 1
    context = {
        "spy_ann_return_test": float((1 + spy_r[test].mean()) ** 12 - 1),
        "universe_ew_ann_return_test": float((1 + np.nanmean(panel.R[test], axis=1).mean()) ** 12 - 1),
    }

    out = []
    for h in hyps:
        if h["signal"] == "ridge":
            S, used = signals.ridge_learner({f: feats[f] for f in h["features"]}, panel,
                                            lam=h["lam"], min_months=h["min_months"])
            for m, ks in used.items():  # audit: no training month's outcome is after month m's rebalance
                assert all(panel.exit[k] <= panel.rebal[m] for k in ks)
        else:
            S = feats[h["signal"]]
        kw = dict(frac=COMMON["frac"], cost_bps=COMMON["cost_bps"], n_perm=GATE["n_perm"])
        if "select" in h:  # only hypotheses that name a mode pass it (H1-H5 run exactly as declared)
            kw["select"] = h["select"]
        tr = harness.evaluate(S[:M], panel.R, panel.listed[:M], train, seed=7, **kw)
        te = harness.evaluate(S[:M], panel.R, panel.listed[:M], test, seed=8, **kw)
        coverage = float(np.isfinite(S[:M][test]).sum() / panel.listed[:M][test].sum())
        rec = {"op": "result", "id": h["id"], "spec_sha": ledger.spec_sha(h), "at": now_iso(),
               **ledger.git_state(), "panel_digest": panel.digest(), "n_tickers": len(panel.tickers),
               "test_signal_coverage": coverage, "train": tr, "test": te, "controls": ctl, "context": context}
        if name != COMMON["universe"]:
            rec["universe"] = name
        if asof:
            rec["prices_asof"] = asof
        ledger.append(rec)
        out.append(rec)
        nan = float("nan")
        print(f"{h['id']:<22} TEST net excess {te.get('mean_net_excess', nan)*100:+.2f}%/mo  p={te['perm_p']:.4f}  "
              f"TRAIN {tr.get('mean_net_excess', nan)*100:+.2f}%/mo p={tr['perm_p']:.4f}")
    return out


FORECAST_BH_Q = 0.10   # docs/FORECAST.md, section 8: the same q as the hypotheses' gate


def is_forecast(r: dict) -> bool:
    return r.get("kind") == "forecast"


def is_link(r: dict) -> bool:
    """A test of the chain engine's links (brain/chain/linktest.py)."""
    return r.get("kind") == "link"


def is_hypothesis(r: dict) -> bool:
    """A hypothesis result (brain/research/hypotheses.py): no ``kind``."""
    return r.get("kind") is None


def is_gated_link(r: dict) -> bool:
    """A next-month link test; the same-month ones are descriptive."""
    return is_link(r) and r.get("gate_type") == "slope"


def is_event(r: dict) -> bool:
    """A test of a corporate event kind (brain/research/eventtest.py)."""
    return r.get("kind") == "event"


def is_gated_event(r: dict) -> bool:
    """A drift event test; the reaction ones are descriptive."""
    return is_event(r) and r.get("gate_type") == "diff"


def is_move(r: dict) -> bool:
    """A test of the earnings move-size model (brain/research/movetest.py)."""
    return r.get("kind") == "move"


def is_gated_move(r: dict) -> bool:
    return is_move(r) and r.get("gate_type") == "pinball"


def is_shock(r: dict) -> bool:
    """A test of the shock radar (brain/research/shocktest.py)."""
    return r.get("kind") == "shock"


def is_gated_shock(r: dict) -> bool:
    return is_shock(r) and r.get("gate_type") in ("logloss", "fade", "logloss_report")


def is_range(r: dict) -> bool:
    """A test of the scan's weekly range (brain/research/rangetest.py)."""
    return r.get("kind") == "range"


def is_gated_range(r: dict) -> bool:
    return is_range(r) and r.get("gate_type") == "pinball"


def is_gated_forecast(r: dict) -> bool:
    """A forecast result with a declared gate (F1b, F1c); descriptive ones (F1a, F1d) are not."""
    return is_forecast(r) and (r.get("gate_type") == "ic" or (r.get("gate_type") is None and "p" in r))


def _bh_p(r: dict) -> float:
    if is_forecast(r) or is_link(r) or is_event(r) or is_move(r) or is_shock(r) or is_range(r):
        p = r.get("p")
        return float(p) if isinstance(p, (int, float)) and math.isfinite(p) else 1.0  # no p counts as 1
    return r["test"]["perm_p"]


def _forecast_verdict(r: dict, q: float) -> str:
    if not is_gated_forecast(r):
        return "REPORT"
    if not (r.get("controls") or {}).get("pass"):
        return "FAIL (controls)"
    g = r.get("gate") or {}
    return "PASS" if g.get("pass") and q < FORECAST_BH_Q else "FAIL"


def _link_verdict(r: dict, q: float) -> str:
    """A link or event test's verdict: REPORT when descriptive, else its gate, its controls and q."""
    if not (is_gated_link(r) or is_gated_event(r)):
        return "REPORT"
    if r.get("thin"):
        return "FAIL (too few events)"
    if not (r.get("controls") or {}).get("pass"):
        return "FAIL (controls)"
    return "PASS" if (r.get("gate") or {}).get("pass") and q < FORECAST_BH_Q else "FAIL"


def _move_verdict(r: dict, q: float) -> str:
    if not (is_gated_move(r) or is_gated_shock(r) or is_gated_range(r)):
        return "REPORT"
    if not (r.get("controls") or {}).get("pass"):
        return "FAIL (controls)"
    return "PASS" if (r.get("gate") or {}).get("pass") and q < FORECAST_BH_Q else "FAIL"


def verdicts(path=None) -> list[dict]:
    """Verdicts for every hypothesis, forecast test and link test with a result. The false-discovery
    correction runs over every hypothesis, every gated forecast test and every gated link test (``p`` for a
    forecast or a link, ``perm_p`` for a hypothesis), withdrawn ones included: a withdrawn test was still tried,
    and dropping it would make every other q-value look better than it is. Descriptive tests (F1a, F1d, the
    same-month link tests) get the verdict REPORT and stay outside it. A forecast row carries ``kind:
    "forecast"``, a link row ``kind: "link"``."""
    res = ledger.latest_results(path)
    gone = ledger.withdrawn(path)
    tested = {k: r for k, r in res.items()
              if is_hypothesis(r) or is_gated_forecast(r) or is_gated_link(r) or is_gated_event(r)
              or is_gated_move(r) or is_gated_shock(r) or is_gated_range(r)}
    q_of = dict(zip(tested, bh_adjust([_bh_p(r) for r in tested.values()])))
    rows = []
    for k, r in res.items():
        if k in gone:
            continue
        q = q_of.get(k, float("nan"))
        if is_forecast(r):
            rows.append({**r, "bh_q": q, "verdict": _forecast_verdict(r, q)})
            continue
        if is_link(r) or is_event(r):
            rows.append({**r, "bh_q": q, "verdict": _link_verdict(r, q)})
            continue
        if is_move(r) or is_shock(r) or is_range(r):
            rows.append({**r, "bh_q": q, "verdict": _move_verdict(r, q)})
            continue
        te = r["test"]
        if not r["controls"]["pass"]:
            v = "NO VERDICT (controls failed)"
        elif te.get("mean_net_excess", math.nan) > GATE["test_mean_net_excess_gt"] and te["perm_p"] < GATE["test_perm_p_lt"] and q < GATE["bh_q"]:
            v = "PASS"
        else:
            v = "FAIL"
        rows.append({**r, "bh_q": q, "verdict": v})
    for k, reason in gone.items():
        if k in res:
            rows.append({**res[k], "bh_q": float("nan"), "verdict": f"WITHDRAWN: {reason}"})
    return rows


def forecast_verdicts(path=None) -> list[dict]:
    """The forecast rows of ``verdicts``."""
    return [r for r in verdicts(path) if is_forecast(r)]


def _pct(x, digits=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x*100:+.{digits}f}%"


def _universe_of(r: dict) -> str:
    return r.get("universe") or COMMON["universe"]


def _universe_line(name: str, rows: list[dict]) -> str:
    try:
        uni = load_universe(name)
        what = f"{len(uni['tickers'])} US stocks ({uni['rule_short']})"
    except (OSError, ValueError, KeyError):
        what = f"{rows[0]['n_tickers']} tickers (universe file missing)"
    ids = ", ".join(r["id"] for r in rows)
    return f"- `{name}`: {what}. Hypotheses: {ids}."


def _persistent_line(ctl: dict) -> str:
    """What the controls say about signals that persist (results recorded before this check lack it)."""
    why = ("The null redraws its holdings every month, so for a signal that persists from month to month "
           "(fundamentals, slow-moving events) it is too narrow and the p-value too small")
    if "persistent_random_false_positive_rate" not in ctl:
        return f"- Random scores held fixed per stock were not tried in this run. {why}, by an amount not measured here."
    return (f"- {ctl['n_random']} random scores held fixed per stock in every month: "
            f"{ctl['persistent_random_false_positive_rate']*100:.1f}% reached p < 0.05. {why}; above about 5%, "
            f"this is by how much. Reported, not part of the declared gate.")


def report(ledger_path=None, results_path=None) -> str:
    """Write RESULTS.md (``results_path``, default config.RESULTS) from the ledger (default config.LEDGER):
    the hypotheses, then the forecasts. Returns the text."""
    every = verdicts(ledger_path)
    if not every:
        return ""
    rows = [r for r in every if is_hypothesis(r)]
    forecasts = [r for r in every if is_forecast(r)]
    links = [r for r in every if is_link(r)]
    evts = [r for r in every if is_event(r)]
    mvs = [r for r in every if is_move(r)]
    shk = [r for r in every if is_shock(r)]
    rng = [r for r in every if is_range(r)]
    led = Path(ledger_path or config.LEDGER)
    try:
        led_name = led.resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        led_name = str(led)
    if not rows:
        text = "\n".join(["# Research results", "",
                          f"Generated {now_iso()} from `{led_name}`. Every number below is copied from the ledger.",
                          "", *_forecast_section(forecasts), *_link_section(links, ledger_path),
                          *_event_section(evts, ledger_path), *_move_section(mvs), *_shock_section(shk),
                          *_range_section(rng)]) + "\n"
        Path(results_path or config.RESULTS).write_text(text)
        return text
    by_uni: dict[str, list[dict]] = {}
    for r in rows:
        by_uni.setdefault(_universe_of(r), []).append(r)
    selects = {h["id"]: h.get("select", "top_frac") for h in HYPOTHESES}
    lines = [
        "# Research results",
        "",
        f"Generated {now_iso()} from `{led_name}`. Every number below is copied from the ledger.",
        "",
        f"TRAIN {SPLIT['first_rebalance']}..{SPLIT['train_end']}, TEST from {SPLIT['test_start']}. Monthly, "
        f"equal weight against the equal-weight universe, {COMMON['cost_bps']} bps per unit turnover. Holdings: "
        "the top third by signal unless the hypothesis names another selection (Select column; see "
        "`brain/research/harness.py`).",
        "",
        "Universes (each has its own panel and controls):",
        "",
        *[_universe_line(name, rs) for name, rs in by_uni.items()],
        "",
        "Gate (declared before running): TEST mean net excess > 0, permutation p < 0.05, and Benjamini–Hochberg "
        "q < 0.10 across every hypothesis in the ledger"
        + (" and every gated forecast test (see Forecasts)" if any(is_gated_forecast(r) for r in forecasts) else "")
        + (" and every gated link test (see Links)" if any(is_gated_link(r) for r in links) else "")
        + (" and every gated event test (see Events)" if any(is_gated_event(r) for r in evts) else "")
        + (" and every gated move-size test (see Move sizes)" if any(is_gated_move(r) for r in mvs) else "")
        + (" and every gated shock test (see Shocks)" if any(is_gated_shock(r) for r in shk) else "")
        + (" and every gated weekly-range test (see Weekly ranges)" if any(is_gated_range(r) for r in rng) else "")
        + ".",
        "",
        "| Hypothesis | Universe | Select | TEST net excess / mo | Annualised | Info ratio | Months won | Perm p | BH q | TRAIN net excess / mo | TRAIN p | TEST coverage | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    nan = float("nan")
    for r in rows:
        te = {"mean_net_excess": nan, "ann_net_excess": nan, "info_ratio": nan, "months_beating_benchmark": 0, **r["test"]}
        tr = {"mean_net_excess": nan, **r["train"]}
        q = f"{r['bh_q']:.3f}" if math.isfinite(r["bh_q"]) else "–"
        lines.append(
            f"| {r['id']} | {_universe_of(r).split('/')[-1].removesuffix('.json')} | {selects.get(r['id'], 'top_frac')} | "
            f"{_pct(te['mean_net_excess'])} | {_pct(te['ann_net_excess'], 1)} | {te['info_ratio']:.2f} | "
            f"{te['months_beating_benchmark']}/{te['n_months']} | {te['perm_p']:.4f} | {q} | "
            f"{_pct(tr['mean_net_excess'])} | {tr['perm_p']:.4f} | {r.get('test_signal_coverage', float('nan')) * 100:.0f}% | "
            f"**{r['verdict'].split(':')[0]}** |")
    lines += [
        "",
        "TEST coverage is the share of listed stock-months with a signal value; the rest are not ranked.",
        "",
    ]
    gone = [r for r in rows if r["verdict"].startswith("WITHDRAWN")]
    if gone:
        lines += ["## Withdrawn", "",
                  "Run, then withdrawn for a mistake. The result stays in the ledger and still counts in the "
                  "Benjamini–Hochberg correction (so its own q is not shown).", ""]
        lines += [f"- {r['id']}: {r['verdict'].split(':', 1)[1].strip()}" for r in gone]
        lines.append("")
    lines += ["## Controls", ""]
    for name, rs in by_uni.items():
        last = max(rs, key=lambda r: r["at"])  # the latest run on this universe
        ctl = last["controls"]
        lines += [
            f"`{name}` (latest run, {last['at'][:10]}):",
            "",
            f"- Oracle (next month's actual return as the signal): TEST p = {ctl['oracle_test_perm_p']:.4f}, "
            f"gross excess {_pct(ctl['oracle_test_mean_gross_excess'])}/mo. The harness can see a real edge.",
            f"- {ctl['n_random']} random signals, redrawn every month: {ctl['random_false_positive_rate']*100:.1f}% "
            f"reached p < 0.05 (expected about 5%). This is what the null is checked against: signals that change "
            f"from month to month.",
            _persistent_line(ctl),
            f"- Controls {'passed' if ctl['pass'] else 'FAILED: no verdicts on this universe are valid'}.",
            "",
        ]
    lines += ["## Survivorship bias, measured", ""]
    for name, rs in by_uni.items():
        ctx = rs[0]["context"]
        lines.append(f"- `{name}`: over TEST the equal-weight universe returned {_pct(ctx['universe_ew_ann_return_test'], 1)} "
                     f"a year; SPY returned {_pct(ctx['spy_ann_return_test'], 1)}.")
    lines += [
        "",
        "The gap is mostly selection: each universe is chosen by today's market value, so every member is a "
        "survivor and a past winner. Comparing against the same universe cancels the level, not the tilt: "
        "survivorship still flatters momentum and growth, and hurts value. A PASS here is an upper bound.",
        "",
        *(["## Ledger notes", ""] + [f"- {n['at'][:10]}: {n['text']}" for n in ledger.read(ledger_path)
                                     if n["op"] == "note"] + [""]
          if any(n["op"] == "note" for n in ledger.read(ledger_path)) else []),
        "## Reading this",
        "",
        "- FAIL means the claim did not beat a shuffled signal on unseen months after costs. It does not mean the "
        "opposite is true.",
        *[f"- `{name}`: {rs[0]['test']['n_months']} TEST months and {rs[0]['n_tickers']} stocks." for name, rs in by_uni.items()],
        "- Few TEST months give little power: an edge of a few tenths of a percent a month can hide in this noise.",
        "- Not financial advice. Nothing here places orders.",
    ]
    if forecasts:
        lines += ["", *_forecast_section(forecasts)]
    if links:
        lines += ["", *_link_section(links, ledger_path)]
    if evts:
        lines += ["", *_event_section(evts, ledger_path)]
    if mvs:
        lines += ["", *_move_section(mvs)]
    if shk:
        lines += ["", *_shock_section(shk)]
    if rng:
        lines += ["", *_range_section(rng)]
    text = "\n".join(lines) + "\n"
    Path(results_path or config.RESULTS).write_text(text)
    return text


# ---------- the Forecasts section (docs/FORECAST.md, section 8) ----------
def _n(x, fmt="+.3f") -> str:
    return "–" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else format(x, fmt)


def _iv(d: dict | None, fmt="+.3f") -> str:
    d = d or {}
    if d.get("lo90") is None or d.get("hi90") is None:
        return "–"
    return f"{_n(d['lo90'], fmt)}..{_n(d['hi90'], fmt)}"


def _share(x):
    """A coverage bound shown as a share: the normal interval can run past 0% or 100%, a share cannot."""
    return None if x is None else min(1.0, max(0.0, x))


def _cover(d: dict | None) -> str:
    d = d or {}
    return f"{_n(d.get('mean'), '.1%')} ({_n(_share(d.get('lo90')), '.1%')}-{_n(_share(d.get('hi90')), '.1%')})"


def _fmetrics(r: dict) -> dict:
    """The metrics a row shows: F1d's are the local model's own, on its 12 stocks."""
    m = r.get("metrics") or {}
    return m.get("llm", m) if r["id"].startswith("F1d") or "llm" in m else m


def _forecast_row(r: dict) -> str:
    m = _fmetrics(r)
    sn = m.get("ic_size_neutral") or {}
    pl = r.get("plan") or {}
    if is_gated_forecast(r):
        sig = f"p {_n(r.get('p'), '.4f')}"
    else:
        sig = f"90% {_iv(sn)}"
    if r["id"].startswith("F1d"):
        c = (r.get("metrics") or {}).get("comparison") or {}
        sig += f"; Brier F1 - model {_n(c.get('mean'), '+.4f')} (p {_n(c.get('p'), '.3f')})"
    q = f"{r['bh_q']:.3f}" if isinstance(r.get("bh_q"), float) and math.isfinite(r["bh_q"]) else "–"
    b = m.get("brier") or {}
    base = m.get("baselines") or {}
    rg = m.get("range") or {}
    h = ((r.get("model") or [{}])[0].get("range") or {}).get("h", "")
    dates = f"{pl.get('first', '?')}..{pl.get('last', '?')} ({r.get('n_dates')})"
    name = r["id"] + (" (trial)" if r.get("trial") else "")
    return (f"| {name} | {h} | {dates} | {r.get('n_samples')} | {_n((m.get('ic') or {}).get('mean'))} | "
            f"{_n(sn.get('mean'))} | {sig} | {q} | {_n(((base.get('size') or {}).get('ic') or {}).get('mean'))} | "
            f"{_n(((base.get('momentum') or {}).get('ic') or {}).get('mean'))} | "
            f"{_n((m.get('spread') or {}).get('mean'), '+.2%')} | {_n(b.get('skill'), '+.2%')} "
            f"({_n(b.get('within_skill'), '+.2%')}) | {_n((m.get('placement') or {}).get('mean'), '.1%')} | "
            f"{_cover(rg.get('cover68'))} | {_cover(rg.get('cover90'))} | **{r['verdict'].split(':')[0]}** |")


def _forecast_words(r: dict) -> str:
    """A plain-words line on what the range and the chance showed."""
    m = _fmetrics(r)
    rg = m.get("range") or {}
    c68, c90 = rg.get("cover68") or {}, rg.get("cover90") or {}
    pb, pn = rg.get("pinball"), rg.get("pinball_naive")
    pl = r.get("plan") or {}
    sn = m.get("ic_size_neutral") or {}
    if pb is not None and pn:
        vs = f"its pinball loss was {abs(1 - pb / pn):.0%} {'below' if pb < pn else 'above'} a naive normal band's"
    else:
        vs = "no pinball comparison"
    rng = (f"the 68% band held {_n(c68.get('mean'), '.1%')} of the time per date (date-level 90% interval "
           f"{_n(_share(c68.get('lo90')), '.1%')} to {_n(_share(c68.get('hi90')), '.1%')}) and the 90% band "
           f"{_n(c90.get('mean'), '.1%')} ({_n(_share(c90.get('lo90')), '.1%')} to {_n(_share(c90.get('hi90')), '.1%')}), "
           f"against 68% and 90% aimed for; {vs}")
    if is_gated_forecast(r):
        ctl = r.get("controls") or {}
        chance = (f"the chance to beat SPY ranked stocks beyond size with mean IC {_n(sn.get('mean'))} "
                  f"(date-level p {_n(r.get('p'), '.4f')}, q {_n(r.get('bh_q'), '.3f')}): {r['verdict']}. "
                  f"Controls {'passed' if ctl.get('pass') else 'FAILED'}: the oracle reached p "
                  f"{_n(ctl.get('oracle_p'), '.5f')} (smallest possible {_n(ctl.get('min_p'), '.5f')}), so the test "
                  f"can reject, and {_n(ctl.get('placebo_reject_rate'), '.1%')} of {ctl.get('placebo_draws')} "
                  "coin-flipped placebos passed (1%-10% expected): a check of the sign-flip test itself, not of the "
                  "forecaster, since coin-flipped signs pass about 5% of the time for any series, a look-ahead leak "
                  "included (the features' leak tests are what check for one). Within-date permutation p "
                  f"{_n(sn.get('within_date_perm_p'), '.4f')} (a diagnostic: too narrow, since stocks move together)")
    elif r["id"].startswith("F1d"):
        full = r.get("metrics") or {}
        c = full.get("comparison") or {}
        chance = (f"the local model's Brier score against F1's on the same stocks: F1 minus model "
                  f"{_n(c.get('mean'), '+.4f')} per date (90% interval {_iv(c, '+.4f')}, exact sign-flip p "
                  f"{_n(c.get('p'), '.3f')}); {full.get('llm_failed')} of {full.get('llm_asked')} answers failed and "
                  f"count as F1's; {c.get('label') or ''}")
    else:
        chance = (f"the chance to beat SPY had mean size-neutral IC {_n(sn.get('mean'))} (90% interval {_iv(sn)}); "
                  f"{m.get('label') or ''}")
    when = f"{pl.get('first', '?')} to {pl.get('last', '?')}, {r.get('n_dates')} dates"
    return f"- {r['id']}{' (trial)' if r.get('trial') else ''} ({when}): {rng}. Then {chance}."


def _forecast_section(rows: list[dict]) -> list[str]:
    live = [r for r in rows if not r["verdict"].startswith("WITHDRAWN")]
    gone = [r for r in rows if r["verdict"].startswith("WITHDRAWN")]
    lines = [
        "## Forecasts",
        "",
        "F1 forecasts each stock's chance to beat SPY over the next h sessions, and the likely range of its price "
        "(docs/FORECAST.md). Every metric is computed per forecast date across the stocks forecast that date, then "
        "over dates. Stocks move together within a week, so every interval and p-value is date-level.",
        "",
        "Gate (F1b, F1c; declared before running): mean size-neutral IC > 0 with a one-sided date-level sign-flip "
        "p < 0.05, Benjamini–Hochberg q < 0.10 across every hypothesis and gated forecast test in the ledger, and "
        "both controls passing. F1a and F1d are descriptive retests (REPORT): their 90% intervals are shown, they "
        "get no verdict and stay out of the correction.",
        "",
        "| Test | h | Dates | Samples | IC | Size-neutral IC | p or 90% interval | BH q | Size IC | Momentum IC | "
        "Spread | Brier skill (within-date) | Placement | 68% band held | 90% band held | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *[_forecast_row(r) for r in rows],
        "",
        "IC: Spearman correlation of the chance to beat SPY with what each stock then did beyond SPY. Size-neutral IC: "
        "the same after the forecast's rank is cleared of its fit on market value, since the universe holds today's "
        "survivors and small-then predicts rose-later. Size IC and Momentum IC: the same samples ranked by size "
        "alone (smaller first) and by 12-1 momentum alone. Spread: top fifth minus bottom fifth by forecast, mean "
        "excess per date (F1d: halves of 12). Brier skill: against the training base rate as a constant; the "
        "within-date part is the only one that speaks to ranking. Placement: share placed on the right side of "
        "the date's median. Bands: share of stocks whose close at the exit fell inside, per date (interval).",
        "",
        "### What the range and the chance showed",
        "",
        *[_forecast_words(r) for r in live],
        "",
        "### Calibration",
        "",
    ]
    for r in live:
        cal = _fmetrics(r).get("calibration") or []
        if not cal:
            continue
        lines += [f"{r['id']}{' (trial)' if r.get('trial') else ''}:", "",
                  "| Forecast range | Mean forecast | Beat SPY | n |", "| --- | --- | --- | --- |",
                  *[f"| {c['lo']:.3f}-{c['hi']:.3f} | {c['mean_p']:.3f} | {c['freq']:.3f} | {c['n']} |" for c in cal], ""]
    if gone:
        lines += ["Withdrawn (still counted in the correction when gated):", "",
                  *[f"- {r['id']}: {r['verdict'].split(':', 1)[1].strip()}" for r in gone], ""]
    lines += ["Forecasts are paper only. Nothing here places orders, and none of it is financial advice."]
    return lines


# ---------- the Links section (docs/LINKS.md) ----------
def _pp(x, digits=2) -> str:
    return f"{x * 100:+.{digits}f}%" if isinstance(x, (int, float)) and math.isfinite(x) else "–"


def _link_section(rows: list[dict], ledger_path=None) -> list[str]:
    """The link tests' table and plain words, from the ledger's results and declared specs."""
    if not rows:
        return []
    specs = {r["id"]: r["spec"] for r in ledger.read(ledger_path) if r["op"] == "declare" and is_link(r)}
    lines = [
        "## Links",
        "",
        "Tests of the chain engine's cause → effect links (`knowledge/links.json`, `docs/LINKS.md`). Each one "
        "measures a chain end to end where the bank can see both ends: the cause's monthly change, and a fixed "
        "basket's equal-weight total return minus SPY's. **Same month** is descriptive (REPORT): how far the "
        "basket's excess return moves with the cause, month-end to month-end. **Next month** is gated, declared "
        "before running: the basket's excess return over the next month, from its first session's close to its "
        "last, against this month's change in the cause. It passes when the slope points the way the links say, "
        "one-sided permutation p < 0.05, the controls pass, and Benjamini–Hochberg q < 0.10 across every gated "
        "test in the ledger. TEST from 2019; TRAIN 2011–2018 is shown beside it. The effect is the slope times a "
        "typical move: percentage points of excess return per move, not the average excess return.",
        "",
        "| Test | Cause → effect | Timing | TEST months | Effect per move, points (TEST) | 90% interval | p | BH q | "
        "TRAIN effect | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    words = []
    for r in rows:
        s = specs.get(r["id"], {})
        te, tr = r["metrics"]["test"], r["metrics"]["train"]
        cm, em = s.get("cause_measure", {}), s.get("effect_measure", {})
        pair = f"{s.get('cause', '?')} → {s.get('effect', '?')}"
        timing = "same month" if s.get("lag") == "same" else "next month"
        lo, hi = te["interval90"]
        q = f"{r['bh_q']:.3f}" if isinstance(r.get("bh_q"), float) and math.isfinite(r["bh_q"]) else "–"
        verdict = r["verdict"].split(":")[0]
        lines.append(f"| {r['id']} | {pair} | {timing} | {te['n']} ({te['months'][0]}..{te['months'][1]}) | "
                     f"{_pp(te['per_move'])} per {cm.get('move_label', 'move')} | {_pp(lo)}..{_pp(hi)} | "
                     f"{te['p']:.4f} | {q} | {_pp(tr['per_move'])} | **{verdict}** |")
        way = "higher" if te["per_move"] > 0 else "lower"
        same = s.get("lag") == "same"
        words.append(f"- **{r['id']}**: for each month with {cm.get('move_words', 'the cause up')}, the "
                     f"{em.get('label', 'basket')}' return over SPY {'that month' if same else 'the next month'} was "
                     f"{abs(te['per_move']) * 100:.2f} points {way} (90% interval {_pp(lo)} to {_pp(hi)}; TRAIN "
                     f"{_pp(tr['per_move'])}). The links say {'higher' if s.get('sign', 0) > 0 else 'lower'}. {verdict}.")
    ctl = [r["controls"] for r in rows]
    lines += ["", *words, "",
              f"Controls on each test's TEST months: the real cause, shifted round by 3 months or more (which keeps "
              f"its autocorrelation and breaks any link), reached p < 0.05 in "
              f"{min(c['shift_fpr'] for c in ctl) * 100:.1f}%–{max(c['shift_fpr'] for c in ctl) * 100:.1f}% of "
              f"{ctl[0]['n_shifts']} shifts (at most 10% allowed); a cause equal to the effect reached p ≤ "
              f"{max(c['oracle_p'] for c in ctl):.4f}. "
              f"{'All passed.' if all(c['pass'] for c in ctl) else 'Some FAILED: those verdicts are void.'}",
              "",
              "Reading this: the baskets are chosen today, so companies that failed or merged are missing "
              "(survivorship). Oil is the front-month future, so its monthly change includes roll effects. A "
              "same-month move says the link is priced as it happens; only a next-month PASS would say it can "
              "be acted on. Links the bank cannot measure stay untested: `python -m brain chain links` lists "
              "what each would need.",
              ""]
    return lines


# ---------- the Events section (docs/EVENTS.md) ----------
def _event_section(rows: list[dict], ledger_path=None) -> list[str]:
    """The event tests' table and plain words, from the ledger's results and declared specs."""
    if not rows:
        return []
    specs = {r["id"]: r["spec"] for r in ledger.read(ledger_path) if r["op"] == "declare" and is_event(r)}
    one = any(specs.get(r["id"], {}).get("sign") for r in rows)
    two = any(not specs.get(r["id"], {}).get("sign") for r in rows)
    pdesc = ("a sign-flip p, one-sided in the direction the event implies (version 1) or two-sided (version 2)"
             if one and two else "a two-sided sign-flip p" if two else "a one-sided sign-flip p")
    way = ("points the way the event implies (either way for a two-sided test)" if two
           else "points the way the event implies")
    lines = [
        "## Events",
        "",
        "Tests of corporate events named in earnings press releases (SEC 8-K exhibit 99; `docs/EVENTS.md`). Fixed "
        "rules tag each release's opening text (`brain/web/events.py`). Each month, the tagged releases' mean excess "
        "return over SPY is compared with the month's other earnings releases; the statistic is the mean of those "
        f"monthly differences, with {pdesc}. **Reaction** (days -1..+1) is descriptive: does the event "
        "move the stock? **Drift** (days +2..+21) is gated, declared before running: it passes when the difference "
        f"{way}, p < 0.05, the controls pass, and Benjamini–Hochberg q < 0.10 across every "
        "gated test in the ledger. TEST from 2019; TRAIN 2011–2018 is shown beside it.",
        "",
        "| Test | Event | Days | TEST events (months) | vs other releases | 90% interval | p | BH q | vs SPY | "
        "TRAIN vs others | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    words = []
    for r in rows:
        s = specs.get(r["id"], {})
        te, tr = r["metrics"]["test"], r["metrics"]["train"]
        lo, hi = te["interval90"]
        q = f"{r['bh_q']:.3f}" if isinstance(r.get("bh_q"), float) and math.isfinite(r["bh_q"]) else "–"
        verdict = r["verdict"].split(":")[0]
        lines.append(f"| {r['id']} | {s.get('event', '?')} | {s.get('days', '?')} | {te['events']} ({te['months']}) | "
                     f"{_pp(te['diff'])} | {_pp(lo)}..{_pp(hi)} | {te['p']:.4f} | {q} | {_pp(te['vs_spy'])} | "
                     f"{_pp(tr['diff'])} | **{verdict}** |")
        implies = ("Tested either way" if not s.get("sign") else
                   f"The event implies {'up' if s.get('sign', 0) > 0 else 'down'}")
        cost = (f" A round trip in these stocks costs about {s['cost_round_trip'] * 100:.0f}%."
                if s.get("cost_round_trip") else "")
        if (cov := r.get("coverage")) and cov.get("unread"):
            cost += (f" {cov['unread']} of the {cov['releases']} releases had no text read (no exhibit 99 found) "
                     f"and are left out of both groups.")
        words.append(f"- **{r['id']}**: over days {s.get('days', '?')}, releases tagged {s.get('event', '?')} did "
                     f"{_pp(te['diff'])} against the month's other releases (90% interval {_pp(lo)} to {_pp(hi)}) and "
                     f"{_pp(te['vs_spy'])} against SPY, over {te['events']} releases in {te['months']} months. "
                     f"{implies}.{cost} {verdict}.")
    ctl = [r["controls"] for r in rows if r["controls"].get("n_placebo")]
    if not ctl:
        ctl = [{"placebo_fpr": float("nan"), "n_placebo": 0, "oracle_p": float("nan"), "pass": False}]
    lines += ["", *words, "",
              f"Controls on each test's TEST months: untagged releases drawn at random in place of the tagged ones reached "
              f"p < 0.05 in {min(c['placebo_fpr'] for c in ctl) * 100:.1f}%–{max(c['placebo_fpr'] for c in ctl) * 100:.1f}% "
              f"of {ctl[0]['n_placebo']} draws (at most 10% allowed); the month's best or worst releases in their place "
              f"reached p ≤ {max(c['oracle_p'] for c in ctl):.4f}. "
              f"{'All passed.' if all(c['pass'] for c in ctl) else 'Some FAILED: those verdicts are void.'}",
              "",
              "Reading this: a tag says only that the release's opening uses the words (\"raises full-year guidance\"); "
              "the rules miss other phrasings and sometimes tag a passing mention. The universe is chosen today "
              "(survivorship). A reaction with no drift means the market prices the event at once.",
              ""]
    return lines


def _move_section(rows: list[dict]) -> list[str]:
    """The move-size tests' table and plain words (brain/research/movetest.py, docs/MOVES.md)."""
    if not rows:
        return []
    lines = [
        "## Move sizes",
        "",
        ("Tests of the earnings move-size model M1 (`brain/moves.py`, `docs/MOVES.md`): the size of a release's move "
         "over days -1..+1 against SPY, whichever way, predicted as a range from what was known two sessions "
         "before. Per release the score is the mean pinball loss over the 10/25/50/75/90% quantiles; each month the "
         "history-only baseline's mean loss minus M1's, and the statistic is the mean of those monthly differences "
         "with a one-sided sign-flip p. The gate, declared before running: positive, p < 0.05, M1's 80% range "
         "holding 75%-85% of TEST releases, the controls passing, and Benjamini–Hochberg q < 0.10 across every "
         "gated test in the ledger."),
        "",
        ("| Test | TEST releases (months) | Loss M1 | Loss baseline | Monthly difference | 90% interval | p | BH q | "
         "80% range held | 50% range held | Verdict |"),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    words = []
    for r in rows:
        m = r["metrics"]
        lo, hi = m["interval90"]
        q = f"{r['bh_q']:.3f}" if isinstance(r.get("bh_q"), float) and math.isfinite(r["bh_q"]) else "–"
        verdict = r["verdict"].split(":")[0]
        lines.append(f"| {r['id']} | {m['releases']} ({m['months']}) | {m['loss_m1'] * 100:.3f} | "
                     f"{m['loss_base'] * 100:.3f} | {m['diff'] * 100:+.3f} | {lo * 100:+.3f}..{hi * 100:+.3f} | "
                     f"{m['p']:.4f} | {q} | {m['cover80'] * 100:.1f}% | {m['cover50'] * 100:.1f}% | **{verdict}** |")
        c = r.get("controls") or {}
        words.append(
            f"- **{r['id']}**: over {m['releases']} releases from {m['first']} to {m['last']}, M1's ranges scored "
            f"{m['loss_m1'] * 100:.3f} points of pinball loss against the baseline's {m['loss_base'] * 100:.3f} "
            f"(better in {m['months_better']} of {m['months']} months). Its 80% range held {m['cover80'] * 100:.1f}% "
            f"of the moves (the baseline's {m['base_cover80'] * 100:.1f}%) and its 50% range {m['cover50'] * 100:.1f}%. "
            f"The chance of moving more than the usual move (median of the last eight) scored a Brier of "
            f"{m['brier_m1']:.4f} against the baseline's {m['brier_base']:.4f} and the base rate's {m['brier_rate']:.4f} "
            f"({m['above_usual_rate'] * 100:.1f}% did). Controls: shuffled ranges reached p < 0.05 in "
            f"{c.get('placebo_fpr', float('nan')) * 100:.1f}% of {c.get('n_placebo', 0)} draws; the actual move as the "
            f"middle reached p {c.get('oracle_p', float('nan')):.4f}. {verdict}.")
    return lines + ["", *words, "",
                    ("Reading this: a size is not a direction. The universe is chosen today (survivorship), and the "
                     "pinball loss is in points of move (0.5 is half a percentage point)."), ""]


def _shock_section(rows: list[dict]) -> list[str]:
    """The shock tests (brain/research/shocktest.py, docs/SHOCKS.md): the radar S1a, then what follows a shock."""
    radar = [r for r in rows if r.get("gate_type") == "logloss"]
    fade = [r for r in rows if r.get("gate_type") == "fade"]
    due = [r for r in rows if r.get("gate_type") == "logloss_report"]
    if not (radar or fade or due):
        return []
    lines, words = ["## Shocks", ""], []
    if radar:
        lines += [
            ("Tests of the shock radar S1 (`brain/shocks.py`, `docs/SHOCKS.md`): the chance that a stock moves 20% "
             "or more against SPY, either way, in one session of the next five, from what was known at a week's last "
             "close. Per stock-week the score is the log loss of the chance; each month the volatility-only "
             "baseline's mean loss minus S1's, and the statistic is the mean of those monthly differences with a "
             "one-sided sign-flip p. The gate, declared before running: positive, p < 0.05, S1's chances calibrated "
             "overall and in each week's top twentieth (actual over predicted within the declared band), the "
             "controls passing, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger."),
            "",
            ("| Test | TEST stock-weeks (months) | Loss S1 | Loss baseline | Monthly difference | 90% interval | p | "
             "BH q | Actual / predicted (all, top 5%) | Top 5% caught | Verdict |"),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    for r in radar:
        m = r["metrics"]
        lo, hi = m["interval90"]
        q = f"{r['bh_q']:.3f}" if isinstance(r.get("bh_q"), float) and math.isfinite(r["bh_q"]) else "–"
        verdict = r["verdict"].split(":")[0]
        cap, capb = m["capture"], m["capture_base"]
        lines.append(f"| {r['id']} | {m['stock_weeks']} ({m['months']}) | {m['loss_s1']:.5f} | {m['loss_base']:.5f} | "
                     f"{m['diff']:+.5f} | {lo:+.5f}..{hi:+.5f} | {m['p']:.4f} | {q} | {m['cal_ratio']:.2f}, "
                     f"{m['top_ratio']:.2f} | {cap['top5'][0] * 100:.0f}% | **{verdict}** |")
        c = r.get("controls") or {}
        words.append(
            f"- **{r['id']}**: over {m['stock_weeks']} stock-weeks of {m['stocks']} stocks from {m['first']} to "
            f"{m['last']}, {m['rate'] * 100:.2f}% of stock-weeks had a 20%+ session ({m['up_share'] * 100:.0f}% of "
            f"those were up). S1's chances averaged {m['mean_s1'] * 100:.2f}% (the baseline's "
            f"{m['mean_base'] * 100:.2f}%) and beat the baseline's log loss in {m['months_better']} of {m['months']} "
            f"months; AUC {m['auc_s1']:.3f} against {m['auc_base']:.3f}. Each week's top 1%, 5% and 10% by S1 held "
            f"{cap['top1'][0] * 100:.0f}%, {cap['top5'][0] * 100:.0f}% and {cap['top10'][0] * 100:.0f}% of all shocks "
            f"(the baseline's {capb['top1'][0] * 100:.0f}%, {capb['top5'][0] * 100:.0f}% and "
            f"{capb['top10'][0] * 100:.0f}%); in S1's top 5% a shock came {m['top_rate'] * 100:.1f}% of the time "
            f"against a predicted {m['top_chance'] * 100:.1f}%. Controls: shuffled chances reached p < 0.05 in "
            f"{c.get('placebo_fpr', float('nan')) * 100:.1f}% of {c.get('n_placebo', 0)} draws; an oracle reached p "
            f"{c.get('oracle_p', float('nan')):.4f}. {verdict}.")
    if due:
        lines += [
            *([""] if radar else []),
            ("Tests of the earnings-aware radar S3 (`brain/catalysts.py`, `docs/SHOCKS.md`, Catalysts): S1 plus a "
             "report expected a quarter after the last one and the stock's earnings habit (the median move against "
             "SPY on its last eight report sessions), judged against S1 on the stock-weeks whose report is expected "
             "1-8 days ahead. Per stock-week the score is the log loss of the chance; each month S1's mean loss minus "
             "S3's, and the statistic is the mean of those monthly differences with a one-sided sign-flip p. The gate, "
             "declared before running: positive, p < 0.05, S3's chances calibrated overall and in each week's top "
             "twentieth, the controls passing, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger."),
            "",
            ("| Test | TEST report-weeks (months) | Loss S3 | Loss S1 | Monthly difference | 90% interval | p | BH q | "
             "Actual / predicted (all, top 5%) | Top 5% caught (S3, S1) | Verdict |"),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    for r in due:
        m = r["metrics"]
        lo, hi = m["interval90"]
        q = f"{r['bh_q']:.3f}" if isinstance(r.get("bh_q"), float) and math.isfinite(r["bh_q"]) else "–"
        verdict = r["verdict"].split(":")[0]
        c3, c1, aw = m["capture_s3"], m["capture_s1"], m["all_weeks"]
        lines.append(f"| {r['id']} | {m['stock_weeks']} ({m['months']}) | {m['loss_s3']:.5f} | {m['loss_s1']:.5f} | "
                     f"{m['diff']:+.5f} | {lo:+.5f}..{hi:+.5f} | {m['p']:.4f} | {q} | {m['cal_ratio']:.2f}, "
                     f"{m['top_ratio']:.2f} | {c3['top5'] * 100:.0f}%, {c1['top5'] * 100:.0f}% | **{verdict}** |")
        c = r.get("controls") or {}
        words.append(
            f"- **{r['id']}**: over {m['stock_weeks']} report-week stock-weeks of {m['stocks']} stocks from {m['first']} "
            f"to {m['last']}, {m['rate'] * 100:.2f}% had a 20%+ session ({m['up_share'] * 100:.0f}% of those were "
            f"up). S3's chances averaged {m['mean_s3'] * 100:.2f}% (S1's {m['mean_s1'] * 100:.2f}%) and beat S1's log "
            f"loss in {m['months_better']} of {m['months']} months; AUC {m['auc_s3']:.3f} against {m['auc_s1']:.3f}. "
            f"Each week's top 5% and 10% of report weeks by S3 held {c3['top5'] * 100:.0f}% and "
            f"{c3['top10'] * 100:.0f}% of their shocks (S1's {c1['top5'] * 100:.0f}% and {c1['top10'] * 100:.0f}%); "
            f"in S3's top 5% a shock came {m['top_rate'] * 100:.1f}% of the time against a predicted "
            f"{m['top_chance'] * 100:.1f}%. On every stock-week ({aw['stock_weeks']}; report weeks were "
            f"{aw['due_share'] * 100:.0f}% of them and held {aw['due_shock_share'] * 100:.0f}% of the shocks), S3's "
            f"loss was {aw['loss_s3']:.5f} against S1's {aw['loss_s1']:.5f}, better in {aw['months_better']} of "
            f"{aw['months']} months (p {aw['p']:.4f}, not gated); each week's top 5% held {aw['top5_s3'] * 100:.0f}% "
            f"of all shocks (S1's {aw['top5_s1'] * 100:.0f}%). Controls: shuffled chances reached p < 0.05 in "
            f"{c.get('placebo_fpr', float('nan')) * 100:.1f}% of {c.get('n_placebo', 0)} draws; an oracle reached p "
            f"{c.get('oracle_p', float('nan')):.4f}. {verdict}.")
    if fade:
        lines += [
            *([""] if radar or due else []),
            ("Tests of what follows a shock: an up-shock episode opens on a session 20%+ above SPY without a results "
             "report and with no up-shock in the 30 days before. Its outcome is its excess return over the next 21 "
             "sessions, against the stocks in its 60-session volatility decile that did not shock that day. Per month, "
             "**percentile** averages the share of peers each episode beat less one half (the typical episode); "
             "**abnormal** averages each episode's outcome less its peers' mean (the average episode, before costs). "
             "The gate, declared before running: the mean of monthly values negative with a one-sided sign-flip "
             "p < 0.05, the controls passing, and Benjamini–Hochberg q < 0.10 across every gated test."),
            "",
            "| Test | Statistic | TEST episodes (months) | Mean of monthly values | 90% interval | p | BH q | Verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    for r in fade:
        m = r["metrics"]
        lo, hi = m["interval90"]
        q = f"{r['bh_q']:.3f}" if isinstance(r.get("bh_q"), float) and math.isfinite(r["bh_q"]) else "–"
        verdict = r["verdict"].split(":")[0]
        stat = "percentile" if r["id"] == "S2a" else "abnormal"
        lines.append(f"| {r['id']} | {stat} | {m['episodes']} ({m['months']}) | {m['stat']:+.4f} | {lo:+.4f}..{hi:+.4f} | "
                     f"{m['p']:.4f} | {q} | **{verdict}** |")
        c = r.get("controls") or {}
        b, bio, u5 = m["big"], m["bio"], m["under5"]
        words.append(
            f"- **{r['id']}** ({stat}): {m['episodes']} episodes of {m['stocks']} stocks from {m['first']} to "
            f"{m['last']}. Over the next 21 sessions their median excess return was {m['median_outcome'] * 100:+.1f}% "
            f"and their mean {m['mean_outcome'] * 100:+.1f}%, against {m['peer_mean'] * 100:+.1f}% for their peers: "
            f"{m['abnormal'] * 100:+.2f} points on average, beating {m['percentile'] * 100:.1f}% of peers "
            f"({m['share_behind'] * 100:.0f}% fell behind the middle one). Negative in {m['months_negative']} of "
            f"{m['months']} months. Jumps of 50%+: {b['n']}, {b['abn'] * 100:+.1f} points; drug and biotech: "
            f"{bio['n']}, {bio['abn'] * 100:+.1f}; under $5: {u5['n']}, {u5['abn'] * 100:+.1f}. Controls: a random "
            f"peer in each episode's place reached p < 0.05 in {c.get('placebo_fpr', float('nan')) * 100:.1f}% of "
            f"{c.get('n_placebo', 0)} draws; an oracle reached p {c.get('oracle_p', float('nan')):.4f}. {verdict}.")
    return lines + ["", *words, "",
                    ("Reading this: a chance of a shock is a size, not a direction, and a fade measured against peers is "
                     "not a trade: shorting a stock that just jumped costs borrow fees and risks a squeeze. The universe "
                     "is chosen today (survivorship: companies bought out or delisted are missing, so shocks are "
                     "undercounted and post-shock returns flattered)."), ""]


def _range_section(rows: list[dict]) -> list[str]:
    """The weekly-range tests' table and plain words (brain/research/rangetest.py, docs/SCAN.md)."""
    if not rows:
        return []
    lines = [
        "## Weekly ranges",
        "",
        ("Tests of the scan's weekly range W1 (`brain/scan.py`, `docs/SCAN.md`): the range of a stock's return over "
         "the next five sessions, from what was known at a close (each week's last for W1a and W1b, every session's for W1c), on tradable stocks. Per stock-week "
         "the score is the mean pinball loss of the log return over the 5/16/50/84/95% quantiles; each month the "
         "baseline's mean loss (the range its 60-session volatility alone gives) minus W1's, and the statistic is the "
         "mean of those monthly differences with a one-sided sign-flip p. The gate, declared before running: "
         "positive, p < 0.05, W1's 68% range holding 63%-73% of TEST stock-weeks and its 90% range 86%-94%, the "
         "controls passing, and Benjamini–Hochberg q < 0.10 across every gated test in the ledger."),
        "",
        ("| Test | TEST stock-weeks (months) | Loss W1 | Loss baseline | Monthly difference | 90% interval | p | BH q | "
         "68% range held | 90% range held | Verdict |"),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    words = []
    for r in rows:
        m = r["metrics"]
        lo, hi = m["interval90"]
        q = f"{r['bh_q']:.3f}" if isinstance(r.get("bh_q"), float) and math.isfinite(r["bh_q"]) else "–"
        verdict = r["verdict"].split(":")[0]
        lines.append(f"| {r['id']} | {m['stock_weeks']} ({m['months']}) | {m['loss_w1'] * 100:.3f} | "
                     f"{m['loss_base'] * 100:.3f} | {m['diff'] * 100:+.3f} | {lo * 100:+.3f}..{hi * 100:+.3f} | "
                     f"{m['p']:.4f} | {q} | {m['cover68'] * 100:.1f}% | {m['cover90'] * 100:.1f}% | **{verdict}** |")
        c = r.get("controls") or {}
        bands = "; ".join(f"{b.replace('us_', '')} {v['cover68'] * 100:.1f}% and {v['cover90'] * 100:.1f}% ({v['n']})"
                          for b, v in (m.get("by_band") or {}).items())
        due, calm = m["report_due"], m["no_report"]
        words.append(
            f"- **{r['id']}**: over {m['stock_weeks']} stock-weeks of {m['stocks']} stocks from {m['first']} to "
            f"{m['last']}, W1's ranges scored {m['loss_w1'] * 100:.3f} points of pinball loss against the baseline's "
            f"{m['loss_base'] * 100:.3f} (better in {m['months_better']} of {m['months']} months). Its 68% range held "
            f"{m['cover68'] * 100:.1f}% of the moves and its 90% range {m['cover90'] * 100:.1f}% (the baseline's "
            f"{m['base_cover68'] * 100:.1f}% and {m['base_cover90'] * 100:.1f}%); {m['below05'] * 100:.1f}% fell below "
            f"the 90% range and {m['above95'] * 100:.1f}% rose above it. By list (68% and 90%): {bands}. With a "
            f"report due: {due['cover68'] * 100:.1f}% and {due['cover90'] * 100:.1f}% ({due['n']}); without: "
            f"{calm['cover68'] * 100:.1f}% and {calm['cover90'] * 100:.1f}%. The typical move was "
            f"{m['median_abs'] * 100:.1f}% either way, against a typical 68% range of ±{m['median_half68'] * 100:.1f}%. "
            f"Within a week, W1's size ranked the stocks' actual moves with a rank correlation of {m['rank_w1']:.3f} "
            f"(the baseline's {m['rank_base']:.3f}). Direction, reported and not tested: {m['up_rate'] * 100:.1f}% of "
            f"stock-weeks rose (the training years' share for stocks of the same size of move: "
            f"{m['up_said'] * 100:.1f}%), and {m['above_middle'] * 100:.1f}% ended above the range's middle. Controls: "
            f"shuffled ranges reached p < 0.05 in {c.get('placebo_fpr', float('nan')) * 100:.1f}% of "
            f"{c.get('n_placebo', 0)} draws; a range that knows the outcome reached p "
            f"{c.get('oracle_p', float('nan')):.4f}. {verdict}.")
    return lines + ["", *words, "",
                    ("Reading this: a range is a size, not a direction; its middle sits near zero for every stock. The "
                     "universe is chosen today (survivorship: companies bought out or delisted are missing, so the "
                     "largest moves are undercounted), and the pinball loss is in points of log return."), ""]
