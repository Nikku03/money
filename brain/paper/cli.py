"""``python -m brain paper replay|step|status``: the learning paper trader (docs/PAPER.md)."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _run(kw):
    from brain.paper import replay
    return replay.run(**kw)


def _replay(a):
    from brain.paper import data, filings, replay
    px = data.closes(data.universe() + [m for m in data.MACRO if m != "SPY"])
    fl = None
    if a.test == "P2":
        print("reading each company's SEC filing history (EDGAR; SEC_USER_AGENT required)...")
        fl = filings.history(data.universe())
        print(f"  {sum(len(v) for v in fl.values())} tagged filings across {sum(1 for v in fl.values() if v)} companies")
    with ProcessPoolExecutor(max_workers=3) as ex:
        trader, shuffled, mom = ex.map(_run, [{"px": px, "filings": fl}, {"px": px, "shuffle": True, "filings": fl},
                                              {"px": px, "mode": "momentum"}])
    g = replay.grade(trader, shuffled, test=a.test)
    tw = replay.test_weeks(trader)
    k = len(tw)
    spy = replay.stats(trader["spy_weekly"][-k:])
    ew = replay.stats(trader["ew_weekly"][-k:])
    momt = replay.stats([w["net"] for w in mom["weeks"][-k:]])
    print(f"{a.test} test period {g['from']} .. {g['to']}: {g['trader']['weeks']} weeks, traded in {g['traded_weeks']}")
    fmt = lambda s: (f"mean/week {s['mean']:+.3%}  t {s['t']:+.2f}  p {s['p']:.3f}  annual {s['ann']:+.1%}"
                     f"  Sharpe {s['sharpe']:+.2f}  total {s['total']:+.1%}  worst drawdown {s['max_dd']:.1%}")
    print(f"  trader (after costs)  {fmt(g['trader'])}")
    print(f"    first half          {fmt(g['first_half'])}")
    print(f"    second half         {fmt(g['second_half'])}")
    print(f"  shuffled control      {fmt(g['shuffled'])}")
    print(f"  12-1 momentum book    {fmt(momt)}")
    print(f"  SPY held (context)    {fmt(spy)}")
    print(f"  60 stocks held, equal (context; survivorship-biased)  {fmt(ew)}")
    for c, ok in g["checks"].items():
        print(f"  [{'x' if ok else ' '}] {c}")
    print(f"{a.test}: {g['verdict']}")
    print("final weights:", {n: round(w, 5) for n, w in zip(trader["features"], trader["final_w"])})
    if a.json:
        Path(a.json).write_text(json.dumps({"grade": g, "trader": trader, "shuffled": shuffled, "momentum": mom},
                                           default=float))
        print(f"written to {a.json}")


def _step(a):
    from brain.paper import live
    r = live.step(use_news=not a.no_news)
    print(f"session {r['date']}")
    if r["closed"]:
        c = r["closed"]
        print(f"  closed the book opened {c['entry']} at {c['exit']}: "
              + (f"net {c['net']:+.3%} after costs" if c["trade"] else "it was flat (scores only); learned from them"))
    if r["opened"]:
        o = r["opened"]
        if o["trade"]:
            print(f"  opened: long {', '.join(o['longs'])}; short {', '.join(o['shorts'])} (predicted spread {o['spread_hat']:+.3%})")
        else:
            print(f"  flat: predicted spread {o['spread_hat']:+.3%} does not beat the {0.004:.1%} round trip")
        tagged = [h for h in o["headlines"] if "tag" in h]
        print(f"  headlines tagged: {len(tagged)}" + "".join(f"\n    {h['ticker']} {h['tag']}: {h['headline'][:90]}" for h in tagged[:12]))
    print("paper only; commit research/paper/ to keep this step (the container is temporary)")


def _restart(a):
    from brain.paper import live
    rep = json.loads(Path(a.from_replay).read_text())
    s = live.restart_from_replay(rep["trader"]["learner"], a.reason)
    L = s["learner"]
    print(f"live learner replaced by the replay's memory: {L.updates} weekly lessons through {rep['trader']['last_date']};"
          f" {len(L.names)} features ({sum(1 for n in L.names if n.startswith('news_'))} headline features start at 0)")


def _status(a):
    from brain.paper import live
    s = live.status()
    print(f"closed weeks {s['closed_weeks']} (traded {s['traded_weeks']}); learner updates {s['updates']}")
    if s["stats"]:
        st = s["stats"]
        print(f"  net per week {st['mean']:+.3%}, t {st['t']:+.2f}, total {st['total']:+.1%}")
    if s["closed_weeks"] < 30:
        print("  under 30 closed weeks: this record says nothing yet, good or bad (CLAUDE.md)")
    print("  weights:", ", ".join(f"{k} {v:+.5f}" for k, v in s["weights"].items() if v))
    if s["open"]:
        print(f"  open since {s['open']['entry']}: " + ("long " + ", ".join(s["open"]["longs"]) + "; short "
              + ", ".join(s["open"]["shorts"]) if s["open"]["trade"] else "flat"))


def add_cli(sub) -> None:
    p = sub.add_parser("paper", help="a paper trader that learns from its own closed trades (docs/PAPER.md)")
    s = p.add_subparsers(dest="paper_cmd", required=True)
    q = s.add_parser("replay", help="walk-forward replay over past years, graded by a declared test")
    q.add_argument("--test", choices=["P1", "P2"], default="P1", help="P1: prices; P2: prices and SEC filings")
    q.add_argument("--json", help="write the full replay here")
    q.set_defaults(fn=_replay)
    q = s.add_parser("step", help="one live day: close a due book, learn, open the next (paper only)")
    q.add_argument("--no-news", action="store_true")
    q.set_defaults(fn=_step)
    q = s.add_parser("restart", help="give the live trader a replay's whole memory (recorded in the ledger)")
    q.add_argument("--from-replay", required=True, help="a replay's --json output")
    q.add_argument("--reason", required=True)
    q.set_defaults(fn=_restart)
    q = s.add_parser("status", help="the live record so far")
    q.set_defaults(fn=_status)
