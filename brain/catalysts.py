"""Catalyst calls: probabilities written down before a clinical readout, then scored (research/catalysts/).

A call is a yes/no question with a threshold anyone can check against the presentation ("OS hazard ratio at or
below 0.75"), the probability of yes, why, and the sources read. It is appended to
``research/catalysts/calls.jsonl`` and committed before the event; the commit's time is the proof it came
first. After the event, ``resolve`` appends the outcome with its source. Nothing is edited or deleted: a call
found to be badly posed is voided with a reason, and stays in the file.

``score`` reports the Brier score of every resolved call, against the 0.25 of always saying 50%, and a
calibration table. Like every other source's calls, these carry no weight until the record is long enough to
say something (CLAUDE.md: claims are not evidence).
"""
from __future__ import annotations

import json
from pathlib import Path

from brain import config
from brain.timeutil import now_iso

CALLS = config.ROOT / "research" / "catalysts" / "calls.jsonl"


def read(path: Path | None = None) -> list[dict]:
    p = Path(path or CALLS)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append(rec: dict, path: Path | None = None) -> None:
    p = Path(path or CALLS)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def state(path: Path | None = None) -> dict[str, dict]:
    """Each call by id, with ``outcome`` (True/False), ``void`` (a reason) or neither."""
    out: dict[str, dict] = {}
    for r in read(path):
        if r["op"] == "call":
            if r["id"] in out:
                raise ValueError(f"call {r['id']} appears twice")
            out[r["id"]] = dict(r)
        elif r["id"] not in out:
            raise ValueError(f"{r['op']} for unknown call {r['id']}")
        elif "outcome" in out[r["id"]] or "void" in out[r["id"]]:
            raise ValueError(f"call {r['id']} is already settled")
        elif r["op"] == "outcome":
            if r["at"] < out[r["id"]]["made_at"]:
                raise ValueError(f"outcome for {r['id']} dated before the call")
            out[r["id"]]["outcome"] = bool(r["outcome"])
            out[r["id"]]["resolved"] = r
        elif r["op"] == "void":
            out[r["id"]]["void"] = r["reason"]
    return out


def call(rec: dict, path: Path | None = None) -> dict:
    """Append a new call. Required: id, ticker, event, event_date, question, p; optional: why, known, sources."""
    for k in ("id", "ticker", "event", "event_date", "question", "p"):
        if k not in rec:
            raise ValueError(f"a call needs {k!r}")
    if not 0.0 <= float(rec["p"]) <= 1.0:
        raise ValueError("p is a probability, 0 to 1")
    if rec["id"] in state(path):
        raise ValueError(f"call {rec['id']} exists; a new call needs a new id")
    made = now_iso()
    if made[:10] > rec["event_date"]:
        raise ValueError(f"the event ({rec['event_date']}) is already past; a call is made before it")
    full = {"op": "call", "made_at": made, "by": rec.get("by", "agent"), **rec, "p": float(rec["p"])}
    append(full, path)
    return full


def resolve(cid: str, outcome: bool, source: str, note: str = "", path: Path | None = None) -> dict:
    st = state(path)
    if cid not in st:
        raise ValueError(f"no call {cid}")
    if "outcome" in st[cid] or "void" in st[cid]:
        raise ValueError(f"call {cid} is already settled")
    if not source.startswith(("http://", "https://")):
        raise ValueError("an outcome cites the http(s) page that settles it")
    rec = {"op": "outcome", "id": cid, "outcome": bool(outcome), "source": source, "note": note, "at": now_iso()}
    append(rec, path)
    return rec


def void(cid: str, reason: str, path: Path | None = None) -> dict:
    st = state(path)
    if cid not in st:
        raise ValueError(f"no call {cid}")
    if "outcome" in st[cid] or "void" in st[cid]:
        raise ValueError(f"call {cid} is already settled")
    rec = {"op": "void", "id": cid, "reason": reason, "at": now_iso()}
    append(rec, path)
    return rec


def score(path: Path | None = None) -> dict:
    """Brier score over resolved calls (lower is better; always saying 50% scores 0.25), and calibration by bin."""
    done = [c for c in state(path).values() if "outcome" in c]
    if not done:
        return {"n": 0}
    brier = sum((c["p"] - c["outcome"]) ** 2 for c in done) / len(done)
    bins = []
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        b = [c for c in done if lo <= c["p"] < hi]
        if b:
            bins.append({"range": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(b),
                         "mean_p": round(sum(c["p"] for c in b) / len(b), 3),
                         "hit_rate": round(sum(c["outcome"] for c in b) / len(b), 3)})
    return {"n": len(done), "brier": round(brier, 4), "baseline_brier": 0.25, "bins": bins}


def _cmd(a) -> int:
    if a.action == "list":
        for c in state().values():
            s = ("void" if "void" in c else "YES" if c.get("outcome") else "no" if "outcome" in c else "open")
            print(f"{c['event_date']}  {c['ticker']:<9} p={c['p']:.2f}  {s:<4}  {c['id']}: {c['question']}")
        return 0
    if a.action == "score":
        s = score()
        if not s["n"]:
            print("no resolved calls yet")
            return 0
        print(f"{s['n']} resolved calls: Brier {s['brier']} (always 50%: 0.25; lower is better)")
        for b in s["bins"]:
            print(f"  said {b['range']}: n={b['n']}, mean p {b['mean_p']}, came true {b['hit_rate']:.0%}")
        return 0
    if a.action == "resolve":
        if not (a.id and a.outcome and a.source):
            print("catalysts resolve ID yes|no --source URL [--note TEXT]")
            return 2
        resolve(a.id, a.outcome == "yes", a.source, a.note or "")
        print(f"resolved {a.id}: {a.outcome}")
        return 0
    return 2


def add_cli(sub) -> None:
    s = sub.add_parser("catalysts", help="probabilities called before clinical readouts, then scored "
                                         "(research/catalysts/): list | score | resolve ID yes|no --source URL")
    s.add_argument("action", choices=["list", "score", "resolve"])
    s.add_argument("id", nargs="?")
    s.add_argument("outcome", nargs="?", choices=["yes", "no"])
    s.add_argument("--source")
    s.add_argument("--note")
    s.set_defaults(fn=_cmd)
