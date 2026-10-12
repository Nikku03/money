"""Append-only research ledger (committed to git): declarations, results, withdrawals.

Negatives stay. A wrong claim is withdrawn with a reason, never deleted.
"""
from __future__ import annotations

import hashlib
import json
import subprocess

from brain import config
from brain.bank import canon
from brain.research.hypotheses import COMMON, HYPOTHESES
from brain.timeutil import now_iso


def spec(h: dict) -> dict:
    return {**COMMON, **h}


def spec_sha(h: dict) -> str:
    return hashlib.sha256(canon(spec(h)).encode()).hexdigest()[:16]


def read(path=None) -> list[dict]:
    p = path or config.LEDGER
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append(rec: dict, path=None) -> None:
    p = path or config.LEDGER
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(canon(rec) + "\n")


def git_state() -> dict:
    try:
        r = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=config.ROOT, capture_output=True, text=True)
        sha = r.stdout.strip() if r.returncode == 0 else ""
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "brain"], cwd=config.ROOT,
                               capture_output=True, text=True).stdout.strip()
        return {"commit": sha or None, "brain_dirty": bool(dirty)}
    except OSError:
        return {"commit": None, "brain_dirty": None}


def declare_all(path=None) -> list[dict]:
    recs = read(path)
    have = {(r["id"], r["spec_sha"]) for r in recs if r["op"] == "declare"}
    ids = {r["id"]: r["spec_sha"] for r in recs if r["op"] == "declare"}
    new = []
    for h in HYPOTHESES:
        sha = spec_sha(h)
        if (h["id"], sha) in have:
            continue
        if h["id"] in ids:
            raise SystemExit(f"{h['id']} was declared with a different spec. Declared specs are frozen: "
                             f"give the change a new id.")
        rec = {"op": "declare", "id": h["id"], "spec_sha": sha, "spec": spec(h), "at": now_iso(), **git_state()}
        append(rec, path)
        new.append(rec)
    return new


def declared(hid: str, path=None) -> dict | None:
    for r in read(path):
        if r["op"] == "declare" and r["id"] == hid:
            return r
    return None


def withdrawn(path=None) -> dict[str, str]:
    return {r["id"]: r["reason"] for r in read(path) if r["op"] == "withdraw"}


def latest_results(path=None) -> dict[str, dict]:
    out = {}
    for r in read(path):
        if r["op"] == "result":
            out[r["id"]] = r
    return out


def note(text: str, path=None) -> None:
    """A dated amendment that changes no declaration or result."""
    append({"op": "note", "text": text, "at": now_iso(), **git_state()}, path)


def withdraw(hid: str, reason: str, path=None) -> None:
    append({"op": "withdraw", "id": hid, "reason": reason, "at": now_iso()}, path)
