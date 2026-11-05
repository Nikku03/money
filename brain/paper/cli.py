"""``python -m brain paper replay|step|status``: the learning paper trader (docs/PAPER.md)."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _run(kw):
    from brain.paper import replay
    return replay.run(**kw)


def _replay(a):
    from brain.paper import data, replay
    px = data.closes(data.universe() + [m for m in data.MACRO if m != "SPY"])
    with ProcessPoolExecutor(max_workers=3) as ex:
        trader, shuffled, mom = ex.map(_run, [{"px": px}, {"px": px, "shuffle": True}, {"px": px, "mode": "momentum"}])
    g = replay.grade(trader, shuffled)
    tw = replay.test_weeks(trader)
    k = len(tw)
    spy = replay.stats(trader["spy_weekly"][-k:])
    ew = replay.stats(trader["ew_weekly"][-k:])
    momt = replay.stats([w["net"] for w in mom["weeks"][-k:]])
    print(f"P1 test period {g['from']} .. {g['to']}: {g['trader']['weeks']} weeks, traded in {g['traded_weeks']}")
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
    print(f"P1: {g['verdict']}")
    print("final weights:", {n: round(w, 5) for n, w in zip(trader["features"], trader["final_w"])})
    if a.json:
        Path(a.json).write_text(json.dumps({"grade": g, "trader": trader, "shuffled": shuffled, "momentum": mom},
                                           default=float))
        print(f"written to {a.json}")


def _step(a):
    from brain.paper import live
    if a.seed_json and not live.STATE.exists():
        s = live.load()
        live.seed_from_replay(json.loads(Path(a.seed_json).read_text())["trader"]["final_w"], s)
        live.save(s)
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
    q = s.add_parser("replay", help="walk-forward replay over past years, graded by the declared test P1")
    q.add_argument("--json", help="write the full replay here")
    q.set_defaults(fn=_replay)
    q = s.add_parser("step", help="one live day: close a due book, learn, open the next (paper only)")
    q.add_argument("--no-news", action="store_true")
    q.add_argument("--seed-json", help="start the price weights from a replay's --json output")
    q.set_defaults(fn=_step)
    q = s.add_parser("status", help="the live record so far")
    q.set_defaults(fn=_status)
