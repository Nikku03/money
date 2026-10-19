"""Command line. Run ``python -m brain -h``."""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from brain import config

# Modules that register their own subcommands with ``add_cli(subparsers)``. A module that is not
# there (or not finished) is skipped, so one part of the brain cannot take the whole CLI down.
PLUGINS = ("brain.sources.insiders", "brain.sources.aviation", "brain.sources.shipping",
           "brain.sources.cameras", "brain.watch", "brain.scout", "brain.decide", "brain.forecast.live", "brain.chain",
           "brain.web")


def _bank():
    from brain.bank import Bank
    return Bank()


def cmd_ingest(a):
    from brain.ingest import PARTS, ingest
    bank = _bank()
    parts = [p for p in PARTS if p not in (a.skip or [])]
    for t in a.tickers:
        print(f"{t.upper()}: {ingest(bank, t, parts, a.all_concepts)}")


def cmd_ingest_universe(a):
    from brain.bank import Bank
    from brain.ingest import ingest
    from brain.sources import prices
    from brain.universe import BROAD
    bank = _bank()
    uni = json.loads((BROAD if a.broad else config.UNIVERSE).read_text())
    for t in uni["tickers"]:
        # One Bank per ticker: a long list must not hold every company's facts in memory at once.
        print(f"{t}: {ingest(Bank(bank.root), t, ('sec', 'filings', 'prices'))}", flush=True)
    print(f"SPY: {bank.add_many(prices.ingest('SPY'))}")


def cmd_macro(a):
    from brain.sources import fred
    out = fred.ingest_all(_bank(), a.series or None)
    return 1 if out["failed"] else 0


def cmd_markets(a):
    from brain import markets
    out = markets.ingest(_bank())
    print(f"{len(out['added'])} of {len(markets.MARKETS)} market symbols updated"
          + (f"; failed: {', '.join(out['failed'])}" if out["failed"] else ""))
    return 1 if not out["added"] else 0


def cmd_terminal(a):
    from brain.terminal.server import serve
    serve(a.host, a.port, not a.no_browser)


def _port(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {text!r}")
    if not 0 <= n <= 65535:
        raise argparse.ArgumentTypeError(f"a port is 0-65535, not {n}")
    return n


def cmd_note(a):
    from brain.claims import note
    try:
        f = note(_bank(), a.ticker, a.text, a.source, tier=a.tier, direction=a.direction,
                 known_at=a.known_at, horizon_days=a.horizon)
    except ValueError as e:  # not an http(s) URL, or a publication time in the future
        print(f"note: {e}", file=sys.stderr)
        return 1
    print(f"stored {f.ref}")


def cmd_ask(a):
    bank = _bank()
    view = bank.view(a.at)
    if not a.attr:
        for attr in bank.attrs(a.ticker):
            print(f"{attr}  ({len(view.facts(a.ticker, attr))} visible)")
        return
    ps = view.periods(a.ticker, a.attr)
    for f in ps[-a.n:]:
        print(f"{f.as_of}  {f.value!s:>22}  {f.unit or '':<10} known {f.known_at}  {f.context.get('form') or ''}"
              f"  [{f.id[:8]}] {f.source.get('url', '')}")


def cmd_show(a):
    bank = _bank()
    hits = [f for f in bank.facts(a.ticker, include_downgraded=True) if f.id.startswith(a.id)]
    for f in hits:
        print(json.dumps(f.to_dict(), indent=1))
    if not hits:
        print("no such fact")
        return 1


def cmd_brief(a):
    from brain.brief import build
    # Offline, the brief completes the broad-universe ranking (about a minute the first time each month)
    # instead of stopping at the terminal's time budget.
    text = build(_bank(), a.ticker, a.at, evidence_budget_s=None)
    print(text)
    print(f"\n(saved to {config.home() / 'briefs' / (a.ticker.upper() + '.md')})", file=sys.stderr)


def cmd_check(a):
    from brain.check import run
    errors, warnings = run(_bank())
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


def cmd_score_claims(a):
    from brain.claims import score
    s = score(_bank())
    if not s["by_source"]:
        print("no directional claims with a completed horizon yet")
    for src, r in sorted(s["by_source"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"{src:<40} n={r['n']:<4} hit rate {r['hit_rate']:.2f}  p vs coin flip {r['p_vs_coin']:.3f}")
    print(f"{len(s['pending'])} claims pending")


def cmd_universe(a):
    from brain.universe import BROAD, build, build_broad
    if a.broad:
        out = build_broad(_bank(), a.max_rank, log=lambda m: print(m, flush=True))
        print(f"{out['n']} tickers kept, {len(out['rejected'])} rejected -> {BROAD}")
        return
    out = build(_bank(), a.n)
    print(f"{out['n']} tickers kept, {len(out['rejected'])} rejected -> {config.UNIVERSE}")


def cmd_research(a):
    from brain.research import ledger
    if a.action == "declare":
        new = ledger.declare_all()
        print("\n".join(f"declared {r['id']} ({r['spec_sha']})" for r in new) or "nothing new to declare")
    elif a.action == "run":
        from brain.research.run import run
        run(_bank(), a.ids or None, n_random=a.n_random, asof=a.prices_asof)
        print(f"-> {config.RESULTS}")
    elif a.action == "report":
        from brain.research.run import report
        print(report())
    elif a.action == "withdraw":
        if not (a.ids and a.reason):
            raise SystemExit("withdraw needs an id and --reason")
        for hid in a.ids:
            ledger.withdraw(hid, a.reason)
        print("withdrawn; the result stays in the ledger")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="brain", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="collect SEC facts, filings, prices and headlines for tickers")
    s.add_argument("tickers", nargs="+")
    s.add_argument("--skip", nargs="*", choices=["sec", "filings", "prices", "news"])
    s.add_argument("--all-concepts", action="store_true", help="keep every XBRL concept, not the curated set")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("ingest-universe", help="refresh SEC facts, filings and prices for the research universe")
    s.add_argument("--broad", action="store_true", help="the broad universe (universe/us_broad.json)")
    s.set_defaults(fn=cmd_ingest_universe)

    s = sub.add_parser("macro", help="collect the Treasury yield curve, and FRED series when FRED_API_KEY is set")
    s.add_argument("series", nargs="*", help="FRED series ids (default: the standard set)")
    s.set_defaults(fn=cmd_macro)

    s = sub.add_parser("markets", help="collect daily prices for world indices, FX, commodities and crypto")
    s.set_defaults(fn=cmd_markets)

    s = sub.add_parser("terminal", help="serve the MONEY TERMINAL on this machine")
    s.add_argument("--host", default="127.0.0.1", help="bind address (default loopback only)")
    s.add_argument("--port", type=_port, default=8765, help="0-65535; 0 picks a free port (default 8765)")
    s.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    s.set_defaults(fn=cmd_terminal)

    s = sub.add_parser("note", help="store a claim found on the web, with its source")
    s.add_argument("ticker")
    s.add_argument("text")
    s.add_argument("--source", required=True, help="URL the claim came from")
    s.add_argument("--tier", default="opinion", choices=["opinion", "reported"])
    s.add_argument("--direction", default="none", choices=["up", "down", "none"])
    s.add_argument("--known-at", help="when it was published (ISO); default now")
    s.add_argument("--horizon", type=int, default=21, help="trading days over which to score it")
    s.set_defaults(fn=cmd_note)

    s = sub.add_parser("ask", help="point-in-time lookup: list attributes, or show one attribute's history")
    s.add_argument("ticker")
    s.add_argument("attr", nargs="?")
    s.add_argument("--at", help="as the bank stood at this date/time")
    s.add_argument("-n", type=int, default=12)
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("show", help="print a fact by id prefix")
    s.add_argument("ticker")
    s.add_argument("id")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("brief", help="cited one-page brief on a ticker")
    s.add_argument("ticker")
    s.add_argument("--at")
    s.set_defaults(fn=cmd_brief)

    s = sub.add_parser("check", help="verify bank invariants and journal replay")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("score-claims", help="every web claim per source vs SPY, overlapping ones included (decide score: the record D1 weighs)")
    s.set_defaults(fn=cmd_score_claims)

    s = sub.add_parser("universe", help="rebuild the research universe by its written rule")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--broad", action="store_true",
                   help="build universe/us_broad.json: every qualifying CIK in SEC's first --max-rank rows")
    s.add_argument("--max-rank", type=int, default=1500)
    s.set_defaults(fn=cmd_universe)

    s = sub.add_parser("research", help="declare / run / report / withdraw hypotheses")
    s.add_argument("action", choices=["declare", "run", "report", "withdraw"])
    s.add_argument("ids", nargs="*")
    s.add_argument("--reason")
    s.add_argument("--n-random", type=int, default=200)
    s.add_argument("--prices-asof", help="run: use the price snapshots known at this time (reproduce a past run)")
    s.set_defaults(fn=cmd_research)

    add_plugins(sub)
    a = p.parse_args(argv)
    return a.fn(a) or 0


def add_plugins(sub, modules=PLUGINS) -> list[str]:
    """Let each module in ``modules`` register its subcommands; returns the ones that did."""
    done = []
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name != name:  # the module exists but needs something missing: say so, carry on
                print(f"warning: {name} not loaded ({e})", file=sys.stderr)
            continue
        except Exception as e:  # a module mid-edit or broken: the rest of the CLI still works
            print(f"warning: {name} not loaded ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        if callable(getattr(mod, "add_cli", None)):
            try:
                mod.add_cli(sub)
            except Exception as e:  # e.g. a subcommand name another module already took
                print(f"warning: {name} subcommands not added ({type(e).__name__}: {e})", file=sys.stderr)
                continue
            done.append(name)
    return done
