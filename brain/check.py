"""Invariants for the bank. Exit 0 means no errors.

Errors: journal corruption (stored id != content, unreadable lines), a sidecar index
that disagrees with the journal, a filed or market fact known before the date it
describes, a derived fact without inputs or known before one of its inputs, a series
snapshot whose file is missing or altered.

Each entity is replayed from its journal alone, one at a time, so memory stays at the
size of the largest entity however big the bank grows.
"""
from __future__ import annotations

import json

from brain import series
from brain.bank import Bank, _Index
from brain.timeutil import days_between, today


def _index_errors(strict: Bank, entity: str) -> list[str]:
    """The saved index must describe a prefix of the journal exactly as a fresh scan does. An index
    that pointed at the wrong record would be noticed and rebuilt on read; one that silently left a
    record out would hide a fact, so any disagreement is an error."""
    p = strict._idx_path(entity)
    if not p.exists():
        return []
    try:
        saved = _Index.from_json(json.loads(p.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [f"{p.name}: unreadable index ({e}); delete it and it is rebuilt from the journal"]
    fresh = strict._idx[entity]
    bad = saved.size > fresh.size
    for a in set(saved.attrs) | set(fresh.attrs):
        want = [t for i, t in enumerate(zip(*[iter(fresh.attrs.get(a, []))] * 3)) if t[0] < saved.size]
        have = list(zip(*[iter(saved.attrs.get(a, []))] * 3))
        bad = bad or want != have
    return [f"{p.name}: index disagrees with the journal; delete it and it is rebuilt"] if bad else []


def run(bank: Bank) -> tuple[list[str], list[str]]:
    root = bank.root
    errors, warnings = [], []
    lister = Bank(root, use_index=False)
    names = lister.entities()
    errors += lister.journal_errors
    elsewhere: dict[str, list[tuple[str, str, str]]] = {}  # input entity -> [(tag, ref, known_at)]
    for e in names:
        b = Bank(root, use_index=False)  # replay this entity from its journal alone
        facts = b.facts(e, include_downgraded=True)
        errors += b.journal_errors
        errors += b.verify(e)
        errors += _index_errors(b, e)
        known = {f.id: f.known_at for f in facts}
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
                    ent, _, fid = ref.partition("/")
                    if ent.upper() != e.upper():
                        elsewhere.setdefault(ent.upper(), []).append((tag, ref, f.known_at))
                    elif fid not in known:
                        errors.append(f"{tag}: input {ref} not in bank")
                    elif known[fid] > f.known_at:
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
    for ent, items in elsewhere.items():  # inputs cited across entities (rare): one replay per entity
        b = Bank(root, use_index=False)
        for tag, ref, known_at in items:
            g = b.get(ref)
            if g is None:
                errors.append(f"{tag}: input {ref} not in bank")
            elif g.known_at > known_at:
                errors.append(f"{tag}: known before its input {ref}")
    return errors, warnings
