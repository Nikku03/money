"""``python -m brain discover earth``: run every experiment on the simulated Earth and report what the
engine found, against the truth it never saw (docs/DISCOVER.md)."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _job(spec):
    from brain.discover import earth, laws
    name, kind, kw = spec
    if kind == "field":
        r = laws.field_law(**kw)
        L = r.pop("law")
        return name, {"kind": kind, "found": L.found, "equation": L.equation(4), "families": L.families,
                      "val_err": L.val_err, "test_err": L.test_err, "note": L.note, **r}
    if kind == "motion":
        r = laws.motion_law(**kw)
        L = r.pop("law")
        return name, {"kind": kind, "found": L.found, "equation": L.equation(5), "val_err": L.val_err,
                      "test_err": L.test_err, "note": L.note, "known": kw.get("known"),
                      "stability": L.stability, **r}
    if kind == "maxwell":
        return name, {"kind": kind, **laws.maxwell(**kw)}
    if kind == "invariants":
        r = laws.invariants(**kw)
        return name, {"kind": kind, "qm": r["qm"], "alone": r["alone"],
                      "conserved_alone": [k for k, v in r["alone"].items() if v < 0.02],
                      "combos": [{"equation": L.equation(4), "test_err": L.test_err} for L in r["combos"]]}
    raise ValueError(kind)


JOBS = [
    ("field, no noise", "field", {"noise": 0.0}),
    ("field, 1% noise", "field", {"noise": 0.01}),
    ("field, 5% noise", "field", {"noise": 0.05}),
    ("gravity from satellites", "motion", {"kind": "satellite"}),
    ("protons, unaided", "motion", {"kind": "proton"}),
    ("protons, wider orbits", "motion", {"kind": "proton", "train_L": (1.5, 4.0), "test_L": (4.0, 5.5)}),
    ("protons, gravity carried over", "motion", {"kind": "proton", "known": {"u/r^2": -1.0}}),
    ("Maxwell, from derivatives", "maxwell", {}),
    ("invariants, q/m 50", "invariants", {"qm": 50.0}),
    ("invariants, q/m 500", "invariants", {"qm": 500.0}),
    ("control: field shuffled", "field", {"shuffle": True}),
    ("control: field without r^-3", "field", {"powers": (1, 2, 4, 5)}),
    ("control: protons shuffled", "motion", {"kind": "proton", "shuffle": True, "known": {"u/r^2": -1.0}}),
    ("control: protons without v x B", "motion", {"kind": "proton", "drop": ("v x B",), "known": {"u/r^2": -1.0}}),
]


def _earth(a):
    from brain.discover import earth
    jobs = [j for j in JOBS if not a.quick or "q/m 500" not in j[0]]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = dict(ex.map(_job, jobs))
    truth = {"dipole tilt (deg)": 11.0, "q/m": earth.QM, "GM": earth.GM}
    print("The simulated Earth's laws, as the engine recovered them from measurements (truth in brackets).\n")
    for name, _, _ in jobs:
        r = results[name]
        if r["kind"] in ("field", "motion"):
            tag = "FOUND    " if r["found"] else "NOT FOUND"
            print(f"{tag} {name}")
            print(f"          {r['equation'][:150]}{'...' if len(r['equation']) > 150 else ''}")
            print(f"          error: fit {r['val_err']:.2%}, held-out (extrapolated) {r['test_err']:.2%}")
            if r["kind"] == "field" and r["found"]:
                print(f"          families: {', '.join(r['families'])}")
                print(f"          dipole tilt {r['tilt_deg']:.2f} deg [11], dipole error {r['m_error']:.4f},"
                      f" quadrupole error {r['Q_error']:.0%}, monopole {r['monopole']:.2g} [0]")
            if r["kind"] == "motion" and r.get("q/m"):
                print(f"          q/m {r['q/m']:.3f} [{earth.QM:g}]")
            if r.get("known"):
                print(f"          carried over: {r['known']}")
            if r["note"]:
                print(f"          {r['note']}")
        elif r["kind"] == "maxwell":
            print(f"FOUND     {name}: {len(r['equations'])} relations among the 9 derivatives")
            for e in r["equations"]:
                print(f"          {e['equation']}   (held-out error {e['test_err']:.3%})")
        else:
            print(f"          {name}: conserved alone (under 2% spread along each path): "
                  f"{', '.join(r['conserved_alone']) or 'nothing'};"
                  f" v_perp^2/|B| spread {r['alone']['v_perp^2/|B|']:.1%}")
            for c in r["combos"]:
                print(f"FOUND     {c['equation']}   (held-out error {c['test_err']:.2%})")
        print()
    print(f"truth the engine never read: {truth}")
    if a.json:
        Path(a.json).write_text(json.dumps(results, default=float, indent=1))
        print(f"results written to {a.json}")


def _compare(a):
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from brain.discover import compare
        import pysindy  # noqa: F401
    except ImportError:
        print("needs PySINDy and scikit-learn: pip install pysindy")
        return 1
    seeds = tuple(range(a.seeds))
    rows = compare.benchmark(seeds=seeds, workers=a.workers)
    print(f"Same data, same library, same held-out tests; {len(seeds)} seed(s) per task (docs/DISCOVER.md).")
    print(compare.table(rows))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=float))
    if a.pysr:
        for r in compare.compare_pysr(seconds=a.pysr):
            print(f"\nPySR, {r['task']}  ({r['seconds']:.0f} s)\n  found: {r['equation']}\n  truth: {r['truth']}"
                  f"\n  held-out error {r['test_err']:.2%}")


def add_cli(sub) -> None:
    p = sub.add_parser("discover", help="equation discovery on a simulated Earth (docs/DISCOVER.md)")
    s = p.add_subparsers(dest="discover_cmd", required=True)
    q = s.add_parser("earth", help="find the field, force and conservation laws from measurements")
    q.add_argument("--workers", type=int, default=4)
    q.add_argument("--quick", action="store_true", help="skip the slow q/m 500 invariant run")
    q.add_argument("--json", help="write every result here")
    q.set_defaults(fn=_earth)
    q = s.add_parser("compare", help="this engine against PySINDy's optimizers and LassoCV (and PySR)")
    q.add_argument("--seeds", type=int, default=3)
    q.add_argument("--workers", type=int, default=4)
    q.add_argument("--pysr", type=int, default=0, help="also run PySR, this many seconds per task (needs pysr)")
    q.add_argument("--json", help="write every row here")
    q.set_defaults(fn=_compare)
