"""Declared tests of the customer links (brain/customers.py, docs/CUSTOMERS.md): when a customer reports its
results, what does the stock of a company that names it as a customer do?

    python -m brain customertest declare      # after committing the code
    python -m brain customertest test         # once; then commit the ledger and research/RESULTS.md

A link is point-in-time: supplier S named listed customer C in an annual report already filed, and S has not filed
its next one (``customers.links``, from every annual report since 2010 of the research universe). An event is a
results report of C while a link holds; C's report-session move against the market is ``r1`` of the episode
library. What S did:
  same   S's move against the market over the same session
  after  S's move against the market over the next AFTER sessions
Events where S itself reports from the session before C's to five sessions after are left out. Industry peers are
up to PEERS stocks of S's industry (the first three digits of its SIC code) that do not name C, read the same way.

Explored on customers' reports of 2011-2023 only (``research/customers/explore_train.py`` and its log): the same
session +0.220 for linked suppliers against +0.075 for their industry peers; the next five sessions +0.040 (t +1.4;
+0.099 on 2011-2018, +0.010 on 2019-2023); the supplier's own later report -0.028 (not declared).

C1a  (the day)   On a customer's report session a supplier that names it moves with it more than its industry peers
                 that do not. Per season (the calendar quarter of the customer's report, at least MIN_N events of
                 each kind): the rank correlation of the customer's move with the linked suppliers' moves, less the
                 same for the peers. The statistic is its mean over the seasons; one-sided sign-flip p. This is
                 about the same session: it says a link is real and explains a move. It is not a forecast.
C1b  (after)     The customer's report-session move tells the supplier's next AFTER sessions. Per season the rank
                 correlation, averaged; one-sided sign-flip p. With eleven seasons of about a hundred events this
                 can tell an effect of the size seen on 2011-2018 (+0.10) from none, not one of +0.04.

Gates: a positive statistic with p < 0.05, the controls (a placebo: the customers' moves shuffled inside a season
reach p < 0.05 in at most 10% of draws; an oracle: the outcome as its own prediction reaches p < 0.01), and
Benjamini-Hochberg q < 0.10 across every gated test in the ledger. The link set is pinned by its digest: a
reading that changes the links is a new experiment.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from brain import config, customers, episodes
from brain.bank import Bank, canon
from brain.research import ledger
from brain.research.earntest import ranks, shuffled
from brain.research.eventtest import boot_interval, flip_p
from brain.research.forecast_tests import Refused, is_trial, paths
from brain.timeutil import now_iso

KIND = "customer"
IDS = ("C1a", "C1b")
GATE_TYPE = {"C1a": "day", "C1b": "after"}
QUESTION = {"C1a": "on a customer's report session, does a supplier that names it move with it more than its "
                   "industry peers that do not",
            "C1b": "does a customer's report-session move tell its supplier's next five sessions"}
VERSION = 1
UNIVERSE = "us_broad"
TRAIN = ("2011-01-01", "2023-12-31")
TEST = ("2024-01-01", "2026-09-30")
MIN_N = 20           # events in a season, of each kind
PEERS = 3            # industry peers a link
AFTER = 5            # sessions after the customer's report session
LONG = 21            # reported beside it
OWN_BEFORE, OWN_AFTER = 4, 8   # days around the customer's report in which the supplier's own report rules it out
BENCH = "SPY"
P_MAX = 0.05
PLACEBO_FPR_MAX, ORACLE_P_MAX = 0.10, 0.01
N_FLIP, N_BOOT, N_PLACEBO, SEED = 10000, 2000, 200, 37
CLEAN = ("brain", "knowledge", "universe")
HERE = Path(__file__).resolve()


def code_files() -> list[Path]:
    from brain import bank, events, metrics, series, timeutil
    from brain.research import earntest, eventtest, stats
    return [HERE, *(Path(m.__file__).resolve()
                    for m in (customers, episodes, events, metrics, bank, series, timeutil, earntest, eventtest, stats))]


def code_sha() -> str:
    h = hashlib.sha256()
    for p in code_files():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def universe() -> list[str]:
    return sorted({t.upper() for t in json.loads((config.ROOT / "universe" / f"{UNIVERSE}.json")
                                                 .read_text())["tickers"]})


def universe_sha() -> str:
    return hashlib.sha256((config.ROOT / "universe" / f"{UNIVERSE}.json").read_bytes()).hexdigest()[:16]


def links_digest(links: list[dict]) -> str:
    """The link set's fingerprint: every (supplier, customer, known from, until), sorted."""
    rows = sorted((x["s"], x["c"], x["known_at"], x["until"]) for x in links)
    return hashlib.sha256(canon(rows).encode()).hexdigest()[:16]


def spec(tid: str, links: list[dict]) -> dict:
    return {"id": tid, "version": VERSION, "train": list(TRAIN), "test": list(TEST), "universe": UNIVERSE,
            "universe_sha": universe_sha(), "min_n": MIN_N, "peers": PEERS, "after": AFTER, "long": LONG,
            "own_window": [OWN_BEFORE, OWN_AFTER], "reader": customers.READER, "link_days": customers.LINK_DAYS,
            "links": len(links), "links_digest": links_digest(links), "code_sha": code_sha(), "n_flip": N_FLIP,
            "n_boot": N_BOOT, "n_placebo": N_PLACEBO, "seed": SEED, "gate_type": GATE_TYPE[tid],
            "gate": {"stat_gt": 0.0, "p_lt": P_MAX, "bh_q_lt": 0.10, "placebo_fpr_max": PLACEBO_FPR_MAX,
                     "oracle_p_max": ORACLE_P_MAX}}


def spec_sha(s: dict) -> str:
    return hashlib.sha256(canon(s).encode()).hexdigest()[:16]


def _git(*args):
    import subprocess
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
    files = [p.relative_to(root).as_posix() for p in code_files()]
    changed = [x for x in _git("diff", "--name-only", commit, "HEAD", "--", *files).stdout.splitlines() if x.strip()]
    return f"code changed since the declare commit {commit[:12]}: {', '.join(changed)}" if changed else None


def committed_declaration(tid: str, sha: str) -> bool:
    r = _git("show", "HEAD:research/ledger.jsonl")
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("op") == "declare" and rec.get("kind") == KIND \
                and rec.get("id") == tid and rec.get("spec_sha") == sha:
            return True
    return False


def _declared(recs: list[dict], tid: str) -> dict | None:
    return next((r for r in recs if r["op"] == "declare" and r.get("kind") == KIND and r["id"] == tid), None)


def history(bank: Bank) -> list[dict]:
    """Every annual report read of the research universe, from the bank (``customers.build`` with its history)."""
    return customers.build(bank, universe(), history=True)


def declare(bank: Bank | None = None, *, trial: bool = False, log=print, hist: list[dict] | None = None) -> list[dict]:
    trial = is_trial(trial)
    led, _ = paths(trial)
    if not trial and (dirty := tree_dirty()):
        raise Refused("commit brain/, knowledge/ and universe/ first; a declaration pins the committed code: "
                      + "; ".join(dirty))
    links = customers.links(hist if hist is not None else history(bank or Bank()))
    out = []
    for tid in IDS:
        s = spec(tid, links)
        have = _declared(ledger.read(led), tid)
        if have:
            if have["spec_sha"] != spec_sha(s):
                raise Refused(f"{tid} was declared with a different spec; a change is a new id")
            if log:
                log(f"{tid} is already declared ({have['spec_sha']})")
            continue
        rec = {"op": "declare", "kind": KIND, "id": tid, "spec_sha": spec_sha(s), "spec": s, "at": now_iso(),
               **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            log(f"declared {tid} ({rec['spec_sha']}): {QUESTION[tid]}; {len(links)} links; commit the ledger before "
                f"running")
    return out


# ---------- the events (pure, given links, episodes and prices) ----------
def season(d: str) -> str:
    """The reporting season of a date: its calendar quarter, "2025Q3"."""
    return f"{d[:4]}Q{(int(d[5:7]) - 1) // 3 + 1}"


class Prices:
    """Each stock's adjusted closes with the market's on the same days, from the bank."""

    def __init__(self, bank: Bank, tickers):
        from brain.metrics import PriceBook
        self.view = bank.view()
        self.book = PriceBook(bank, sorted(set(tickers) | {BENCH}))
        self.bd, self.ba = self.book.adj_upto(self.view, BENCH)
        self.cache: dict[str, tuple | None] = {}

    def get(self, t: str):
        if t not in self.cache:
            if not self.book.has(t):
                self.cache[t] = None
            else:
                d, a = self.book.adj_upto(self.view, t)
                self.cache[t] = (d.astype(str), a, episodes.aligned(d, self.bd, self.ba))
        return self.cache[t]

    def move(self, t: str, start: str, sessions: int | None = None, end: str | None = None) -> float | None:
        """t's move against the market from the close of ``start`` to the close ``sessions`` later (or of ``end``)."""
        got = self.get(t)
        if got is None:
            return None
        d, a, b = got
        i = int(np.searchsorted(d, start))
        if i >= len(d) or d[i] != start:
            return None
        j = i + sessions if sessions is not None else int(np.searchsorted(d, end))
        if j >= len(d) or (end is not None and d[j] != end):
            return None
        return episodes._excess(a, b, i, j)


def peers_of(links: list[dict], sic: dict[str, str]) -> list[dict]:
    """For each link, up to PEERS stocks of the supplier's industry that never name that customer (the first in
    ticker order), as links of their own with ``s`` the peer."""
    by_sic: dict[str, list[str]] = defaultdict(list)
    for t, k in sorted(sic.items()):
        by_sic[k].append(t)
    linked = {(x["s"], x["c"]) for x in links}
    out = []
    for x in links:
        pool = [p for p in by_sic.get(sic.get(x["s"], ""), []) if p not in (x["s"], x["c"]) and (p, x["c"]) not in linked]
        out += [{**x, "s": p} for p in pool[:PEERS]]
    return out


def events(links: list[dict], eps_by: dict[str, list[dict]], move, span: tuple[str, str]) -> list[dict]:
    """One row a (link, results report of the customer inside ``span`` while the link holds): the customer's move
    ``c_r1`` and the supplier's ``same``, ``after`` and ``long`` (None when the supplier itself reports nearby or
    has no price). ``move(ticker, start, sessions=None, end=None)`` is Prices.move."""
    out = []
    for x in links:
        s, c = x["s"], x["c"]
        own = [e["date"] for e in eps_by.get(s, [])]
        for e in eps_by.get(c, []):
            if e.get("r1") is None or not (span[0] <= e["date"] <= span[1]):
                continue
            if not (x["known_at"][:10] < e["pre"] and e["date"] <= x["until"][:10]):
                continue
            row = {"s": s, "c": c, "share": x.get("share"), "date": e["date"], "season": season(e["date"]),
                   "c_r1": e["r1"], "same": None, "after": None, "long": None}
            d1 = date.fromisoformat(e["date"])
            if not any(-OWN_BEFORE <= (date.fromisoformat(d) - d1).days <= OWN_AFTER for d in own):
                row["same"] = move(s, e["pre"], end=e["date"])
                row["after"] = move(s, e["date"], sessions=AFTER)
                row["long"] = move(s, e["date"], sessions=LONG)
            out.append(row)
    return out


# ---------- the statistics (pure) ----------
def season_ic(group: np.ndarray, pred: np.ndarray, y: np.ndarray, min_n: int = MIN_N) -> dict[str, float]:
    """Per season with at least ``min_n`` usable rows: the rank correlation of ``pred`` with ``y``."""
    out = {}
    for g in sorted(set(group.tolist())):
        k = (group == g) & np.isfinite(pred) & np.isfinite(y)
        if k.sum() < min_n:
            continue
        a, b = ranks(pred[k]), ranks(y[k])
        if a.std() == 0 or b.std() == 0:
            continue
        out[g] = float(np.corrcoef(a, b)[0, 1])
    return out


def _cols(rows: list[dict], key: str):
    g = np.array([r["season"] for r in rows])
    x = np.array([r["c_r1"] for r in rows], dtype=float)
    y = np.array([r[key] if r.get(key) is not None else np.nan for r in rows], dtype=float)
    return g, x, y


def _controls(stat_of, group: np.ndarray, pred: np.ndarray, y: np.ndarray, rng, n_placebo: int) -> dict:
    """``stat_of(pred)`` gives the per-season values of the statistic for a prediction column."""
    hits = 0
    for _ in range(n_placebo):
        d = stat_of(shuffled(group, pred, rng))
        hits += bool(len(d)) and flip_p(d, 1, 2000, rng) < P_MAX
    do = stat_of(y)
    oracle_p = flip_p(do, 1, 2000, rng) if len(do) else 1.0
    fpr = hits / n_placebo if n_placebo else 0.0
    return {"placebo_fpr": fpr, "oracle_p": oracle_p, "pass": bool(fpr <= PLACEBO_FPR_MAX and oracle_p < ORACLE_P_MAX)}


def fifths(rows: list[dict]) -> dict:
    """What the suppliers did after the customer's best and worst fifth of reports (rows with a same-session move)."""
    k = [r for r in rows if r.get("same") is not None]
    if len(k) < 10:
        return {}
    q = np.quantile([r["c_r1"] for r in k], [0.2, 0.8])
    out = {}
    for name, rr in (("best", [r for r in k if r["c_r1"] >= q[1]]), ("worst", [r for r in k if r["c_r1"] <= q[0]])):
        a5 = [r["after"] for r in rr if r.get("after") is not None]
        out[name] = {"n": len(rr), "customer": float(np.mean([r["c_r1"] for r in rr])),
                     "same": float(np.mean([r["same"] for r in rr])),
                     "rose": float(np.mean([r["same"] > 0 for r in rr])),
                     "after": float(np.mean(a5)) if a5 else None}
    return out


def evaluate(linked: list[dict], peer: list[dict], rng=None, n_placebo: int = N_PLACEBO) -> dict[str, dict]:
    """Both tests' results from the linked suppliers' and the peers' events (pure)."""
    rng = rng or np.random.default_rng(SEED)
    out = {}
    # C1a: the same session, linked less peers
    g, x, y = _cols(linked, "same")
    pg, px_, py = _cols(peer, "same")
    peer_ic = season_ic(pg, px_, py)

    def day(pred: np.ndarray) -> np.ndarray:
        mine = season_ic(g, pred, y)
        return np.array([mine[s] - peer_ic[s] for s in sorted(mine) if s in peer_ic])
    lic = season_ic(g, x, y)
    seasons = [s for s in sorted(lic) if s in peer_ic]
    d = day(x)
    out["C1a"] = {"seasons": len(d), "events": int((np.isfinite(y)).sum()), "peer_events": int(np.isfinite(py).sum()),
                  "links": len({(r["s"], r["c"]) for r in linked if r.get("same") is not None}),
                  "customers": len({r["c"] for r in linked if r.get("same") is not None}),
                  "first": seasons[0] if seasons else None, "last": seasons[-1] if seasons else None,
                  "stat": float(d.mean()) if len(d) else float("nan"),
                  "linked": float(np.mean([lic[s] for s in seasons])) if seasons else float("nan"),
                  "peers": float(np.mean([peer_ic[s] for s in seasons])) if seasons else float("nan"),
                  "interval90": boot_interval(d, N_BOOT, rng), "seasons_positive": int((d > 0).sum()),
                  "p": flip_p(d, 1, N_FLIP, rng) if len(d) else 1.0, "fifths": fifths(linked),
                  "controls": _controls(day, g, x, y, rng, n_placebo)}
    # C1b: the next AFTER sessions
    g, x, y = _cols(linked, "after")

    def after(pred: np.ndarray) -> np.ndarray:
        got = season_ic(g, pred, y)
        return np.array([got[s] for s in sorted(got)])
    d = after(x)
    _, lx, ly = _cols(linked, "long")
    long_ic = season_ic(g, lx, ly)
    pg, px_, py = _cols(peer, "after")
    pa = season_ic(pg, px_, py)
    got = season_ic(g, x, y)
    out["C1b"] = {"seasons": len(d), "events": int(np.isfinite(y).sum()),
                  "first": min(got) if got else None, "last": max(got) if got else None,
                  "stat": float(d.mean()) if len(d) else float("nan"), "interval90": boot_interval(d, N_BOOT, rng),
                  "seasons_positive": int((d > 0).sum()), "p": flip_p(d, 1, N_FLIP, rng) if len(d) else 1.0,
                  "long": float(np.mean(list(long_ic.values()))) if long_ic else float("nan"),
                  "peers": float(np.mean(list(pa.values()))) if pa else float("nan"),
                  "controls": _controls(after, g, x, y, rng, n_placebo)}
    return out


def words(tid: str, m: dict) -> str:
    c = m["controls"]
    tail = (f" Controls: shuffled customer moves reached p < 0.05 in {c['placebo_fpr'] * 100:.1f}% of draws; the "
            f"oracle p {c['oracle_p']:.4f}.")
    if tid == "C1a":
        f = m.get("fifths") or {}
        fifth = ""
        if f.get("best") and f.get("worst"):
            b, w = f["best"], f["worst"]
            fifth = (f" After the customers' best fifth of reports ({b['customer'] * 100:+.2f}%), their suppliers "
                     f"moved {b['same'] * 100:+.2f}% the same session ({b['rose'] * 100:.0f}% rose); after the worst "
                     f"fifth ({w['customer'] * 100:+.2f}%), {w['same'] * 100:+.2f}% ({w['rose'] * 100:.0f}% rose).")
        return (f"{m['events']} reports of a customer while a link held ({m['links']} links, {m['customers']} "
                f"customers) in {m['seasons']} seasons. Rank correlation of the customer's report-session move with "
                f"its suppliers' move the same session {m['linked']:+.4f}, with their industry peers' "
                f"{m['peers']:+.4f}: the link adds {m['stat']:+.4f} (positive in {m['seasons_positive']} of "
                f"{m['seasons']} seasons).{fifth} The same session: it explains a move, it is not a forecast.{tail}")
    return (f"{m['events']} reports of a customer while a link held, in {m['seasons']} seasons. Rank correlation of "
            f"the customer's report-session move with its suppliers' next {AFTER} sessions {m['stat']:+.4f} (positive "
            f"in {m['seasons_positive']} of {m['seasons']} seasons); the next {LONG} sessions {m['long']:+.4f}; "
            f"industry peers' next {AFTER} sessions {m['peers']:+.4f}.{tail}")


# ---------- from the bank ----------
def sic_of(bank: Bank, tickers) -> dict[str, str]:
    view = bank.view()
    out = {}
    for t in tickers:
        f = view.latest(t, "identity.sic")
        if f is not None:
            out[t] = str(f.value)[:3]
    return out


def rows_for(bank: Bank, links: list[dict], eps: list[dict], span: tuple[str, str], sic: dict | None = None,
             move=None) -> tuple[list[dict], list[dict]]:
    """(linked events, peer events) inside ``span``."""
    eps_by: dict[str, list[dict]] = defaultdict(list)
    for e in sorted(eps, key=lambda e: e["date"]):
        eps_by[e["t"]].append(e)
    peer = peers_of(links, sic if sic is not None else sic_of(bank, universe()))
    if move is None:
        move = Prices(bank, {x["s"] for x in links} | {x["s"] for x in peer}).move
    return events(links, eps_by, move, span), events(peer, eps_by, move, span)


def test(bank: Bank, *, trial: bool = False, log=print, hist: list[dict] | None = None,
         eps: list[dict] | None = None, sic: dict | None = None, move=None) -> list[dict]:
    from brain.research import run
    trial = is_trial(trial)
    led, res = paths(trial)
    recs = ledger.read(led)
    links = customers.links(hist if hist is not None else history(bank))
    decls = {}
    for tid in IDS:
        decl = _declared(recs, tid)
        if decl is None:
            raise Refused(f"{tid} is not declared")
        if spec_sha(decl["spec"]) != decl["spec_sha"]:
            raise Refused(f"{tid}: its declared spec does not match its spec_sha")
        if decl["spec"]["links_digest"] != links_digest(links):
            raise Refused(f"{tid}: the links are not the declared links (a new reading is a new experiment)")
        if not trial:
            if dirty := tree_dirty():
                raise Refused("the working tree has changes under brain/, knowledge/ or universe/: " + "; ".join(dirty))
            if why := lineage_problem(decl.get("commit")):
                raise Refused(f"{tid}: {why}")
            if decl["spec"]["code_sha"] != code_sha():
                raise Refused(f"{tid}: the code is not the declared code")
            if not committed_declaration(tid, decl["spec_sha"]):
                raise Refused(f"{tid}: commit the ledger with its declaration first")
        decls[tid] = decl
    todo = [t for t in IDS if not any(r["op"] == "result" and r.get("kind") == KIND and r["id"] == t for r in recs)]
    if not todo:
        if log:
            log("every customer test already has its result")
        return []
    if eps is None:
        if log:
            log("building the episode library from the bank (the report sessions' moves)")
        eps = episodes.build(bank, log=log)
        episodes.save(eps)
    linked, peer = rows_for(bank, links, eps, TEST, sic, move)
    result = evaluate(linked, peer)
    out = []
    for tid in todo:
        m = result[tid]
        gate = {"stat_positive": m["stat"] > 0, "p_ok": m["p"] < P_MAX, "controls": m["controls"]["pass"]}
        gate["pass"] = bool(all(gate.values()))
        rec = {"op": "result", "kind": KIND, "id": tid, "spec_sha": decls[tid]["spec_sha"],
               "gate_type": GATE_TYPE[tid], "metrics": {**{k: v for k, v in m.items() if k != "controls"},
                                                        "question": QUESTION[tid], "words": words(tid, m)},
               "controls": m["controls"], "gate": gate, "p": m["p"], "at": now_iso(), **ledger.git_state()}
        ledger.append(rec, led)
        out.append(rec)
        if log:
            log(f"{tid}: {rec['metrics']['words']}")
    run.report(led, res)
    if log:
        log(f"wrote {res}" + ("" if trial else "; commit the ledger and RESULTS.md, whatever they say"))
    return out


def _cmd(a) -> int:
    bank = Bank()
    if a.action == "declare":
        declare(bank, trial=a.trial)
    else:
        eps = episodes.load() if (a.cached and a.trial) else None     # a real run builds the episodes afresh
        test(bank, trial=a.trial, eps=eps or None)
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("customertest", help="declare or run C1a, C1b: does a customer's report reach its supplier")
    p.add_argument("action", choices=["declare", "test"])
    p.add_argument("--trial", action="store_true", help="a trial run: a scratch ledger, no git checks")
    p.add_argument("--cached", action="store_true", help="use the episode library already built (var/episodes)")
    p.set_defaults(fn=_cmd)
