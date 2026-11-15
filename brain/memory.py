"""Memory between sessions (docs/MEMORY.md): what the agent needs to pick up where it left off after a reset.

The machine this runs on is reset now and then. The repository comes back from git; the local home (``var/``) does
not. Most of the home is public data that can be read again: SEC filings and accounts, prices, prospectuses. What
cannot be read again is what the agent added itself: the claims it noted from pages it read. The agent's own notes
on the work are files in ``memory/``: ``MEMORY.md`` (where things stand) and ``JOURNAL.md`` (what happened, by
date). They are committed, and the agent keeps a copy of ``memory/`` on the person's Google Drive.

- ``save``: write the bank's claims to ``memory/claims.jsonl``, and list ``memory/``'s files with their sha256, which
  the agent's Drive upload names each file by.
- ``load``: put ``memory/claims.jsonl`` back into the bank, then say what the home lacks and the commands that
  rebuild it.
- ``status``: what the home holds and lacks, without changing anything.

The agent's working memory is not made larger by any of this. It is a notebook the agent reads at the start of a
session and searches when it needs something it no longer holds.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from brain import config
from brain.bank import Bank, Fact, canon

CLAIMS = "claims.jsonl"
NOTES = ("MEMORY.md", "JOURNAL.md")
DRIVE_FOLDER = "money-brain memory"
REBUILD = (
    ("prices and the latest filings of every scan stock", "python -m brain scan run --refresh", "about 40 minutes"),
    ("company accounts and what each prospectus sells", "python -m brain mosaic read", "about 1-2 hours from empty"),
    ("the mosaic's model is committed; only its data needs reading", "python -m brain mosaic radar", "a few minutes"),
)


def memory_dir() -> Path:
    """``memory/`` in the repository, or BRAIN_MEMORY (tests, trials)."""
    return Path(os.environ.get("BRAIN_MEMORY") or config.ROOT / "memory")


def claims(bank: Bank) -> list[Fact]:
    """Every claim the bank holds (``brain note``, a desk note's pages), oldest first."""
    out = [f for e in bank.entities() for f in bank.facts(e, "claim")]
    return sorted(out, key=lambda f: (f.known_at, f.entity, f.id))


def export_claims(bank: Bank, path: Path | None = None) -> int:
    """Write the bank's claims to ``memory/claims.jsonl``, one fact a line. Returns how many."""
    p = Path(path or memory_dir() / CLAIMS)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = claims(bank)
    p.write_text("".join(canon(f.to_dict()) + "\n" for f in rows))
    return len(rows)


def import_claims(bank: Bank, path: Path | None = None) -> int:
    """Put the claims of ``memory/claims.jsonl`` back into the bank. A claim it already holds is not added twice.
    Returns how many were added."""
    p = Path(path or memory_dir() / CLAIMS)
    if not p.exists():
        return 0
    facts = [Fact.from_dict(json.loads(ln)) for ln in p.read_text().splitlines() if ln.strip()]
    return bank.add_many(facts)


def files(path: Path | None = None) -> list[dict]:
    """The files of ``memory/`` with their size and sha256."""
    d = Path(path or memory_dir())
    if not d.exists():
        return []
    return [{"name": f.name, "path": str(f.resolve()), "bytes": f.stat().st_size,
             "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
            for f in sorted(d.iterdir()) if f.is_file()]


def status(bank: Bank) -> dict:
    """What the home holds, by kind, and whether the market's own prices are there."""
    have = {"stocks with prices": 0, "stocks with filings": 0, "stocks with accounts": 0,
            "stocks with prospectuses read": 0, "claims": 0}
    for e in bank.entities():
        attrs = set(bank.attrs(e))
        have["stocks with prices"] += "series.price_daily" in attrs
        have["stocks with filings"] += "filing" in attrs
        have["stocks with accounts"] += any(a.startswith("us-gaap:") for a in attrs)
        have["stocks with prospectuses read"] += "offering" in attrs
        have["claims"] += len(bank.facts(e, "claim")) if "claim" in attrs else 0
    spy = bank.view().latest("SPY", "series.price_daily")
    return {"have": have, "spy_last": spy.as_of if spy else None, "empty": have["stocks with prices"] < 100}


def _cmd(a) -> int:
    bank = Bank()
    if a.action == "save":
        n = export_claims(bank)
        print(f"wrote {n} claims to {memory_dir() / CLAIMS}")
        print(f"Files to keep on Google Drive (folder '{DRIVE_FOLDER}'), with the sha256 an upload names each by:")
        for f in files():
            print(f"  {f['name']:14} {f['bytes']:>8} bytes  sha256:{f['sha256']}:{f['path']}")
        return 0
    if a.action == "load":
        print(f"put {import_claims(bank)} claims back into the bank from {memory_dir() / CLAIMS}")
    st = status(bank)
    print("The local home holds: " + ", ".join(f"{v} {k}" for k, v in st["have"].items())
          + f"; SPY's prices to {st['spy_last'] or 'none'}.")
    if st["empty"]:
        print("The home looks reset. Rebuild it from public sources, in this order:")
        for what, cmd, took in REBUILD:
            print(f"  {cmd:40} {what} ({took})")
    print(f"Read {memory_dir() / 'MEMORY.md'} first: it says where the work stands.")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("memory", help="memory between sessions: save the agent's own records, load them after a "
                                      "reset (docs/MEMORY.md)")
    p.add_argument("action", choices=["save", "load", "status"])
    p.set_defaults(fn=_cmd)
