"""Run declared hypotheses, with controls, and write research/RESULTS.md."""
from __future__ import annotations

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


def load_universe() -> dict:
    return json.loads(config.UNIVERSE.read_text())


def masks(panel) -> tuple[np.ndarray, np.ndarray]:
    mo = panel.months
    start = mo >= SPLIT["first_rebalance"]
    train = np.flatnonzero(start & (mo <= SPLIT["train_end"]))
    test = np.flatnonzero(mo >= SPLIT["test_start"])
    return train, test


def controls(panel, test, n_random: int = 200) -> dict:
    """Positive control (oracle) must win; random signals must pass at about the nominal rate."""
    M = len(panel.entry)
    oracle = np.vstack([panel.R, np.full((1, panel.R.shape[1]), np.nan)])
    kw = dict(frac=COMMON["frac"], cost_bps=COMMON["cost_bps"])
    o = harness.evaluate(oracle[:M], panel.R, panel.listed[:M], test, n_perm=2000, seed=1, **kw)
    rng = np.random.default_rng(12345)
    ps = []
    for s in range(n_random):
        S = rng.random((M, panel.R.shape[1]))
        ps.append(harness.evaluate(S, panel.R, panel.listed[:M], test, n_perm=500, seed=100 + s, **kw)["perm_p"])
    fpr = float(np.mean(np.array(ps) < 0.05))
    ok = o["perm_p"] < 0.001 and 0.01 <= fpr <= 0.10
    return {"oracle_test_perm_p": o["perm_p"], "oracle_test_mean_gross_excess": o["mean_gross_excess"],
            "random_false_positive_rate": fpr, "n_random": n_random, "pass": bool(ok)}


def run(bank: Bank, ids: list[str] | None = None, n_random: int = 200) -> list[dict]:
    uni = load_universe()
    hyps = [h for h in HYPOTHESES if not ids or h["id"] in ids]
    for h in hyps:
        d = ledger.declared(h["id"])
        if d is None or d["spec_sha"] != ledger.spec_sha(h):
            raise SystemExit(f"{h['id']} is not declared with this spec. Run `python -m brain research declare` "
                             f"and commit the ledger first.")
    panel = build(bank, uni["tickers"])
    M = len(panel.entry)
    train, test = masks(panel)
    print(f"panel: {len(panel.tickers)} tickers, months {panel.months[train[0]]}..{panel.months[test[-1]]} "
          f"(train {len(train)}, test {len(test)})")

    ctl = controls(panel, test, n_random)
    print(f"controls: oracle p={ctl['oracle_test_perm_p']:.4f}, random FPR={ctl['random_false_positive_rate']:.3f} "
          f"-> {'PASS' if ctl['pass'] else 'FAIL'}")

    need = sorted({h["signal"] for h in hyps if h["signal"] in signals.HAND}
                  | {f for h in hyps for f in h.get("features", [])})
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
        tr = harness.evaluate(S[:M], panel.R, panel.listed[:M], train, seed=7, **kw)
        te = harness.evaluate(S[:M], panel.R, panel.listed[:M], test, seed=8, **kw)
        coverage = float(np.isfinite(S[:M][test]).sum() / panel.listed[:M][test].sum())
        rec = {"op": "result", "id": h["id"], "spec_sha": ledger.spec_sha(h), "at": now_iso(),
               **ledger.git_state(), "panel_digest": panel.digest(), "n_tickers": len(panel.tickers),
               "test_signal_coverage": coverage, "train": tr, "test": te, "controls": ctl, "context": context}
        ledger.append(rec)
        out.append(rec)
        print(f"{h['id']:<22} TEST net excess {te['mean_net_excess']*100:+.2f}%/mo  p={te['perm_p']:.4f}  "
              f"TRAIN {tr['mean_net_excess']*100:+.2f}%/mo p={tr['perm_p']:.4f}")
    report()
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
        elif te["mean_net_excess"] > GATE["test_mean_net_excess_gt"] and te["perm_p"] < GATE["test_perm_p_lt"] and q < GATE["bh_q"]:
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


def report() -> str:
    rows = verdicts()
    if not rows:
        return ""
    uni = load_universe()
    r0 = rows[0]
    ctl, ctx = r0["controls"], r0["context"]
    lines = [
        "# Research results",
        "",
        f"Generated {now_iso()} from `research/ledger.jsonl`. Every number below is copied from the ledger.",
        "",
        f"Universe: {len(uni['tickers'])} US large caps ({uni['rule_short']}). "
        f"TRAIN {SPLIT['first_rebalance']}..{SPLIT['train_end']}, TEST from {SPLIT['test_start']}. "
        f"Top third by signal vs the equal-weight universe, monthly, {COMMON['cost_bps']} bps per unit turnover.",
        "",
        "Gate (declared before running): TEST mean net excess > 0, permutation p < 0.05, and Benjamini–Hochberg "
        "q < 0.10 across every hypothesis in the ledger.",
        "",
        "| Hypothesis | TEST net excess / mo | Annualised | Info ratio | Months won | Perm p | BH q | TRAIN net excess / mo | TRAIN p | TEST coverage | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        te, tr = r["test"], r["train"]
        lines.append(
            f"| {r['id']} | {_pct(te['mean_net_excess'])} | {_pct(te['ann_net_excess'], 1)} | {te['info_ratio']:.2f} | "
            f"{te['months_beating_benchmark']}/{te['n_months']} | {te['perm_p']:.4f} | {r['bh_q']:.3f} | "
            f"{_pct(tr['mean_net_excess'])} | {tr['perm_p']:.4f} | {r.get('test_signal_coverage', float('nan')) * 100:.0f}% | "
            f"**{r['verdict']}** |")
    lines += [
        "",
        "TEST coverage is the share of listed stock-months with a signal value; the rest are not ranked.",
        "",
        "## Controls",
        "",
        f"- Oracle (next month's actual return as the signal): TEST p = {ctl['oracle_test_perm_p']:.4f}, "
        f"gross excess {_pct(ctl['oracle_test_mean_gross_excess'])}/mo. The harness can see a real edge.",
        f"- {ctl['n_random']} random signals: {ctl['random_false_positive_rate']*100:.1f}% reached p < 0.05 "
        f"(expected about 5%). The null is calibrated.",
        f"- Controls {'passed' if ctl['pass'] else 'FAILED: no verdicts are valid'}.",
        "",
        "## Survivorship bias, measured",
        "",
        f"Over TEST the equal-weight universe returned {_pct(ctx['universe_ew_ann_return_test'], 1)} a year; SPY returned "
        f"{_pct(ctx['spy_ann_return_test'], 1)}. The gap is mostly selection: the universe is today's largest "
        "companies, so every member is a survivor and a past winner. Comparing against the same universe cancels the "
        "level, not the tilt: survivorship still flatters momentum and growth, and hurts value. A PASS here is an "
        "upper bound.",
        "",
        *(["## Ledger notes", ""] + [f"- {n['at'][:10]}: {n['text']}" for n in ledger.read() if n["op"] == "note"] + [""]
          if any(n["op"] == "note" for n in ledger.read()) else []),
        "## Reading this",
        "",
        "- FAIL means the claim did not beat a shuffled signal on unseen months after costs. It does not mean the "
        "opposite is true.",
        f"- {r0['test']['n_months']} TEST months and {r0['n_tickers']} stocks give little power: an edge of a few "
        "tenths of a percent a month can hide in this noise.",
        "- Not financial advice. Nothing here places orders.",
    ]
    text = "\n".join(lines) + "\n"
    config.RESULTS.write_text(text)
    return text
