"""Memory bank: atomic, sourced, bitemporal facts in an append-only journal.

Every fact says what it describes (``as_of``), when the public could first know
it (``known_at``), where it came from (``source``) and how far to trust it
(``tier``). A :class:`View` at time t shows only facts known by t, so anything
that reasons through a View cannot see the future. Restatements are new facts
with a later ``known_at``; nothing is overwritten.

Journal: one JSONL shard per entity under ``<home>/bank/``. Ops are ``add`` and
``downgrade``. State is rebuilt by replaying the journal. Stored ids are trusted
on load; ``Bank.verify`` (run by ``brain check``) recomputes them from content.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from brain import config
from brain.timeutil import norm_date, norm_known_at, now_iso

TIERS = {
    "filed": "regulatory filing (SEC EDGAR)",
    "market": "exchange data: prices, splits",
    "reported": "published by a third party: news, data vendors",
    "derived": "computed here from other facts; cites them in inputs",
    "opinion": "forecasts, ratings, sentiment, an agent's reading of a page",
}


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass(frozen=True, slots=True)
class Fact:
    entity: str
    attr: str
    value: object
    as_of: str
    known_at: str
    tier: str
    source: dict
    unit: str | None = None
    context: dict = field(default_factory=dict)
    inputs: tuple = ()  # "ENTITY/id" references, for derived facts
    retrieved_at: str | None = None
    _id: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "entity", self.entity.upper())
        object.__setattr__(self, "as_of", norm_date(self.as_of))
        object.__setattr__(self, "known_at", norm_known_at(self.known_at))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}; use one of {sorted(TIERS)}")

    def content(self) -> dict:
        """Everything that defines the fact. retrieved_at is excluded so a re-fetch of the
        same data yields the same id."""
        return {
            "entity": self.entity, "attr": self.attr, "value": self.value, "unit": self.unit,
            "as_of": self.as_of, "known_at": self.known_at, "tier": self.tier,
            "source": self.source, "context": self.context, "inputs": list(self.inputs),
        }

    def content_id(self) -> str:
        return hashlib.sha256(canon(self.content()).encode()).hexdigest()[:20]

    @property
    def id(self) -> str:
        fid = self._id
        if fid is None:
            fid = self.content_id()
            object.__setattr__(self, "_id", fid)
        return fid

    @property
    def ref(self) -> str:
        return f"{self.entity}/{self.id}"

    def to_dict(self) -> dict:
        d = self.content()
        d["retrieved_at"] = self.retrieved_at
        d["id"] = self.id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(
            entity=d["entity"], attr=d["attr"], value=d["value"], as_of=d["as_of"],
            known_at=d["known_at"], tier=d["tier"], source=d["source"], unit=d.get("unit"),
            context=d.get("context") or {}, inputs=tuple(d.get("inputs") or ()),
            retrieved_at=d.get("retrieved_at"),
        )


def period_key(f: Fact) -> tuple:
    """Two facts with the same key describe the same thing; the later-known one wins."""
    return (f.attr, f.as_of, f.context.get("start"))


class LookAheadError(RuntimeError):
    pass


class Bank:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else config.home() / "bank"
        self.root.mkdir(parents=True, exist_ok=True)
        self._facts: dict[str, dict[str, Fact]] = {}
        self._down: dict[str, dict[str, str]] = {}
        self._index: dict[str, dict[str, list[Fact]]] = {}
        self._keys: dict[str, dict[str, list[str]]] = {}
        self.journal_errors: list[str] = []

    # ---------- storage ----------
    def shard(self, entity: str) -> Path:
        safe = re.sub(r"[^A-Z0-9._-]", "_", entity.upper())
        return self.root / f"{safe}.jsonl"

    def entities(self) -> list[str]:
        """One name per shard, read from its first record. A shard whose first record cannot be
        read (a torn write, manual damage) still counts, under its file name, and the problem goes
        to ``journal_errors`` as in ``_load``: one bad line must not stop every caller."""
        out = []
        for p in sorted(self.root.glob("*.jsonl")):
            name = None
            with p.open(errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        name = rec.get("fact", rec).get("entity", p.stem)
                    except (ValueError, AttributeError) as e:
                        self.journal_errors.append(f"{p.name}:{n}: unreadable first record ({e}); "
                                                   f"entity taken from the file name")
                    if not isinstance(name, str) or not name:
                        name = p.stem
                    break
            if name is not None:
                out.append(name)
        return out

    def _load(self, entity: str) -> None:
        entity = entity.upper()
        if entity in self._facts:
            return
        self._facts[entity], self._down[entity], self._index[entity], self._keys[entity] = {}, {}, {}, {}
        p = self.shard(entity)
        if not p.exists():
            return
        with p.open(errors="replace") as fh:  # a torn multi-byte character must not stop the replay
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    self.journal_errors.append(f"{p.name}:{n}: unparseable ({e})")
                    continue
                try:
                    self._apply(entity, rec, where=f"{p.name}:{n}")
                except (KeyError, TypeError, ValueError, AttributeError) as e:
                    self.journal_errors.append(f"{p.name}:{n}: malformed record ({type(e).__name__}: {e})")

    def _apply(self, entity: str, rec: dict, where: str = "") -> None:
        op = rec.get("op")
        if op == "add":
            f = Fact.from_dict(rec["fact"])
            stored = rec["fact"].get("id")
            if stored:
                # Trusted on load for speed; `verify` (run by `brain check`) recomputes every id.
                object.__setattr__(f, "_id", stored)
            self._insert(entity, f)
        elif op == "downgrade":
            if rec["id"] not in self._facts[entity]:
                self.journal_errors.append(f"{where}: downgrade of unknown fact {rec['id']}")
            self._down[entity][rec["id"]] = rec.get("reason", "")
        else:
            self.journal_errors.append(f"{where}: unknown op {op!r}")

    def _insert(self, entity: str, f: Fact) -> None:
        if f.id in self._facts[entity]:
            return  # the same fact appended twice (e.g. by two processes) is still one fact
        self._facts[entity][f.id] = f
        lst = self._index[entity].setdefault(f.attr, [])
        keys = self._keys[entity].setdefault(f.attr, [])
        k = f.known_at + f.as_of
        i = bisect.bisect_right(keys, k)
        keys.insert(i, k)
        lst.insert(i, f)

    def _write(self, entity: str, recs: Iterable[dict]) -> None:
        with self.shard(entity).open("a") as fh:
            for r in recs:
                fh.write(canon(r) + "\n")

    # ---------- writes ----------
    def add(self, fact: Fact) -> bool:
        return self.add_many([fact]) == 1

    def add_many(self, facts: Iterable[Fact]) -> int:
        by_entity: dict[str, list[dict]] = {}
        n = 0
        for f in facts:
            self._load(f.entity)
            if f.id in self._facts[f.entity]:
                continue
            self._insert(f.entity, f)
            by_entity.setdefault(f.entity, []).append({"op": "add", "fact": f.to_dict()})
            n += 1
        for e, recs in by_entity.items():
            self._write(e, recs)
        return n

    def downgrade(self, entity: str, fact_id: str, reason: str) -> None:
        """Weight a fact to zero (it stays in the journal, with the reason)."""
        entity = entity.upper()
        self._load(entity)
        if fact_id not in self._facts[entity]:
            raise KeyError(f"{entity}/{fact_id} not in bank")
        self._down[entity][fact_id] = reason
        self._write(entity, [{"op": "downgrade", "entity": entity, "id": fact_id, "reason": reason, "at": now_iso()}])

    # ---------- reads ----------
    def get(self, ref: str) -> Fact | None:
        entity, _, fid = ref.partition("/")
        self._load(entity)
        return self._facts[entity.upper()].get(fid)

    def facts(self, entity: str, attr: str | None = None, include_downgraded: bool = False) -> list[Fact]:
        entity = entity.upper()
        self._load(entity)
        if attr is None:
            src = [f for lst in self._index[entity].values() for f in lst]
        else:
            src = list(self._index[entity].get(attr, []))
        if not include_downgraded:
            src = [f for f in src if f.id not in self._down[entity]]
        return src

    def attrs(self, entity: str) -> list[str]:
        self._load(entity)
        return sorted(self._index[entity.upper()])

    def is_downgraded(self, entity: str, fact_id: str) -> bool:
        self._load(entity)
        return fact_id in self._down[entity.upper()]

    def view(self, t: str | None = None) -> "View":
        return View(self, t or now_iso())

    def digest(self, entity: str) -> str:
        entity = entity.upper()
        self._load(entity)
        h = hashlib.sha256()
        for fid in sorted(self._facts[entity]):
            h.update(fid.encode())
            h.update(b"D" if fid in self._down[entity] else b"A")
        return h.hexdigest()

    def verify(self, entity: str) -> list[str]:
        """Recompute every fact's id from its content; a mismatch means the journal was altered."""
        entity = entity.upper()
        self._load(entity)
        return [f"{self.shard(entity).name}: stored id {fid} != content id {f.content_id()}"
                for fid, f in self._facts[entity].items() if f.content_id() != fid]

    def replay_digest(self, entity: str) -> str:
        """Digest of the state rebuilt from the journal file alone."""
        return Bank(self.root).digest(entity)


class View:
    """The bank as it stood at time t. Only facts with known_at <= t are visible."""

    def __init__(self, bank: Bank, t: str):
        self.bank = bank
        self.cutoff = norm_known_at(t)
        self.t = self.cutoff[:10]

    def facts(self, entity: str, attr: str, where: Callable[[Fact], bool] | None = None) -> list[Fact]:
        entity = entity.upper()
        self.bank._load(entity)
        lst = self.bank._index[entity].get(attr, [])
        keys = self.bank._keys[entity].get(attr, [])
        i = bisect.bisect_right(keys, self.cutoff + "￿")
        down = self.bank._down[entity]
        out = []
        for f in lst[:i]:
            if f.known_at > self.cutoff:
                continue
            if f.id in down:
                continue
            if where is None or where(f):
                out.append(f)
        return out

    def periods(self, entity: str, attr: str, where: Callable[[Fact], bool] | None = None) -> list[Fact]:
        """Latest-known version of each period, sorted by as_of."""
        best: dict[tuple, Fact] = {}
        for f in self.facts(entity, attr, where):
            k = period_key(f)
            if k not in best or f.known_at >= best[k].known_at:
                best[k] = f
        return sorted(best.values(), key=lambda f: (f.as_of, f.known_at))

    def latest(self, entity: str, attr: str, where: Callable[[Fact], bool] | None = None) -> Fact | None:
        ps = self.periods(entity, attr, where)
        return ps[-1] if ps else None

    def require_not_after(self, date: str) -> None:
        if date[:10] > self.t:
            raise LookAheadError(f"asked for {date} through a view at {self.t}")
