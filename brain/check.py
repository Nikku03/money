"""Invariants for the bank. Exit 0 means no errors.

Errors: journal corruption (stored id != content), replay mismatch, a filed or
market fact known before the date it describes, a derived fact without inputs or
known before one of its inputs, a series snapshot whose file is missing or altered.
"""
from __future__ import annotations

from brain import series
from brain.bank import Bank
from brain.timeutil import days_between, today


def run(bank: Bank) -> tuple[list[str], list[str]]:
    # Check the journal itself: replay every shard from scratch, ignoring the sidecar indexes.
    bank = Bank(bank.root, use_index=False)
    errors, warnings = [], []
    for e in bank.entities():
        facts = bank.facts(e, include_downgraded=True)
        errors += bank.journal_errors
        bank.journal_errors = []
        errors += bank.verify(e)
        if bank.digest(e) != bank.replay_digest(e):
            errors.append(f"{e}: state differs from journal replay")
        for f in facts:
            tag = f"{e}/{f.id} {f.attr}"
            if not f.source.get("name"):
                errors.append(f"{tag}: source has no name")
            if f.tier != "derived" and not f.source.get("url"):
                errors.append(f"{tag}: non-derived fact without a source url")
            if f.tier in ("filed", "market") and f.known_at[:10] < f.as_of:
                errors.append(f"{tag}: known_at {f.known_at} before as_of {f.as_of}")
            if f.tier == "derived":
                if not f.inputs:
                    errors.append(f"{tag}: derived fact cites no inputs")
                for ref in f.inputs:
                    g = bank.get(ref)
                    if g is None:
                        errors.append(f"{tag}: input {ref} not in bank")
                    elif g.known_at > f.known_at:
                        errors.append(f"{tag}: known before its input {ref}")
            if f.attr.startswith("series.") and not series.verify(f.value):
                errors.append(f"{tag}: series file {f.value[:12]} missing or altered")
        px = [f for f in facts if f.attr == "series.price_daily"]
        if px:
            last = max(f.as_of for f in px)
            if days_between(last, today()) > 7:
                warnings.append(f"{e}: latest prices {last}, older than 7 days")
        ident = [f for f in facts if f.attr.startswith("identity.")]
        if ident and days_between(max(f.as_of for f in ident), today()) > 90:
            warnings.append(f"{e}: identity last checked over 90 days ago")
    return errors, warnings
