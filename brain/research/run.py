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


def is_gated_forecast(r: dict) -> bool:
    """A forecast result with a declared gate (F1b, F1c); descriptive ones (F1a, F1d) are not."""
    return is_forecast(r) and (r.get("gate_type") == "ic" or (r.get("gate_type") is None and "p" in r))


def _bh_p(r: dict) -> float:
    if is_forecast(r) or is_link(r):
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
    if not is_gated_link(r):
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
              if is_hypothesis(r) or is_gated_forecast(r) or is_gated_link(r)}
    q_of = dict(zip(tested, bh_adjust([_bh_p(r) for r in tested.values()])))
    rows = []
    for k, r in res.items():
        if k in gone:
            continue
        q = q_of.get(k, float("nan"))
        if is_forecast(r):
            rows.append({**r, "bh_q": q, "verdict": _forecast_verdict(r, q)})
            continue
        if is_link(r):
            rows.append({**r, "bh_q": q, "verdict": _link_verdict(r, q)})
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
    led = Path(ledger_path or config.LEDGER)
    try:
        led_name = led.resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        led_name = str(led)
    if not rows:
        text = "\n".join(["# Research results", "",
                          f"Generated {now_iso()} from `{led_name}`. Every number below is copied from the ledger.",
                          "", *_forecast_section(forecasts), *_link_section(links, ledger_path)]) + "\n"
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
        + (" and every gated link test (see Links)" if any(is_gated_link(r) for r in links) else "") + ".",
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
        "basket's equal-weight total return minus SPY's, on month-end closes. **Same month** is descriptive "
        "(REPORT): how far the basket moves with the cause. **Next month** is gated, declared before running: "
        "the slope points the way the links say, one-sided permutation p < 0.05, controls pass, and "
        "Benjamini–Hochberg q < 0.10 across every gated test in the ledger. TEST from 2019; TRAIN 2011–2018 is "
        "shown beside it.",
        "",
        "| Test | Cause → effect | Timing | TEST months | Effect per move (TEST) | 90% interval | p | BH q | "
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
        way = "beat" if te["per_move"] > 0 else "trailed"
        same = s.get("lag") == "same"
        words.append(f"- **{r['id']}**: {'in' if same else 'after'} a month with {cm.get('move_words', 'the cause up')}, "
                     f"the {em.get('label', 'basket')} {way} SPY by {abs(te['per_move']) * 100:.2f}% "
                     f"{'that month' if same else 'the next month'} on average (90% interval {_pp(lo)} to {_pp(hi)}; "
                     f"TRAIN {_pp(tr['per_move'])}). The links say {'up' if s.get('sign', 0) > 0 else 'down'}. {verdict}.")
    ctl = [r["controls"] for r in rows]
    lines += ["", *words, "",
              f"Controls on each test's TEST months: random causes reached p < 0.05 in "
              f"{min(c['random_fpr'] for c in ctl) * 100:.1f}%–{max(c['random_fpr'] for c in ctl) * 100:.1f}% of "
              f"{ctl[0]['n_random']} draws (about 5% expected); a cause equal to the effect reached p ≤ "
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
