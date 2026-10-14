"""Run declared hypotheses, with controls, and write research/RESULTS.md.

Each hypothesis names its universe (``universe/us_large.json`` unless it says otherwise).
One panel is built per universe, with its own controls, so adding hypotheses on a new
universe leaves the H1-H5 panel, and every number computed on it, exactly as it was.
"""
from __future__ import annotations

import hashlib
import json
import math

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


def verdicts() -> list[dict]:
    """Verdicts for every hypothesis with a result. The false-discovery correction runs over all of
    them, withdrawn ones included: a withdrawn hypothesis was still tried, and dropping it would
    make every other q-value look better than it is."""
    res = ledger.latest_results()
    gone = ledger.withdrawn()
    q_of = dict(zip(res, bh_adjust([r["test"]["perm_p"] for r in res.values()])))
    rows = []
    for k, r in res.items():
        if k in gone:
            continue
        q = q_of[k]
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


def report() -> str:
    rows = verdicts()
    if not rows:
        return ""
    by_uni: dict[str, list[dict]] = {}
    for r in rows:
        by_uni.setdefault(_universe_of(r), []).append(r)
    selects = {h["id"]: h.get("select", "top_frac") for h in HYPOTHESES}
    lines = [
        "# Research results",
        "",
        f"Generated {now_iso()} from `research/ledger.jsonl`. Every number below is copied from the ledger.",
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
        "q < 0.10 across every hypothesis in the ledger.",
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
        *(["## Ledger notes", ""] + [f"- {n['at'][:10]}: {n['text']}" for n in ledger.read() if n["op"] == "note"] + [""]
          if any(n["op"] == "note" for n in ledger.read()) else []),
        "## Reading this",
        "",
        "- FAIL means the claim did not beat a shuffled signal on unseen months after costs. It does not mean the "
        "opposite is true.",
        *[f"- `{name}`: {rs[0]['test']['n_months']} TEST months and {rs[0]['n_tickers']} stocks." for name, rs in by_uni.items()],
        "- Few TEST months give little power: an edge of a few tenths of a percent a month can hide in this noise.",
        "- Not financial advice. Nothing here places orders.",
    ]
    text = "\n".join(lines) + "\n"
    config.RESULTS.write_text(text)
    return text
