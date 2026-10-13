"""Memory bank: atomic, sourced, bitemporal facts in an append-only journal.

Every fact says what it describes (``as_of``), when the public could first know
it (``known_at``), where it came from (``source``) and how far to trust it
(``tier``). A :class:`View` at time t shows only facts known by t, so anything
that reasons through a View cannot see the future. Restatements are new facts
with a later ``known_at``; nothing is overwritten.

Journal: one JSONL shard per entity under ``<home>/bank/``. Ops are ``add`` and
``downgrade``. State is rebuilt by replaying the journal, one attribute at a time
as readers ask for it (a sidecar ``.idx`` file per shard says where each
attribute's records are; it is a cache and can always be rebuilt from the
journal). Stored ids are trusted on load; ``Bank.verify`` (run by
``brain check``) recomputes them from content.
"""
from __future__ import annotations

import bisect
import fcntl
import hashlib
import json
import os
import re
from array import array
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


_ADD_HEAD, _ADD_TAIL = b'{"fact":{', b',"op":"add"}'
_ATTR = re.compile(rb'"attr":"((?:[^"\\]|\\.)*)"')
_IDX_VERSION = 1


class _Index:
    """Where each attribute's records sit in a shard: flat (offset, length, line number) triples.
    Covers complete lines only, up to ``size`` bytes; ``tail`` hashes the bytes just before ``size``
    so a rewritten (not appended-to) shard is noticed and the index rebuilt."""

    __slots__ = ("inode", "size", "lines", "tail", "attrs", "ops", "bad", "dirty")

    def __init__(self):
        self.inode, self.size, self.lines, self.tail = None, 0, 0, ""
        self.attrs: dict[str, array] = {}
        self.ops = array("q")
        self.bad: list[str] = []  # journal errors found while indexing, reported by every reader
        self.dirty = False

    def to_json(self) -> dict:
        return {"v": _IDX_VERSION, "inode": self.inode, "size": self.size, "lines": self.lines, "tail": self.tail,
                "attrs": {a: list(v) for a, v in self.attrs.items()}, "ops": list(self.ops), "bad": self.bad}

    @classmethod
    def from_json(cls, d: dict) -> "_Index":
        if d.get("v") != _IDX_VERSION:
            raise ValueError("index version")
        ix = cls()
        ix.inode, ix.size, ix.lines, ix.tail = d["inode"], int(d["size"]), int(d["lines"]), str(d["tail"])
        ix.attrs = {str(a): array("q", v) for a, v in d["attrs"].items()}
        ix.ops = array("q", d["ops"])
        ix.bad = [str(x) for x in d.get("bad", [])]
        if any(len(v) % 3 for v in [*ix.attrs.values(), ix.ops]):
            raise ValueError("index triples")
        return ix


def _tail_hash(fh, size: int) -> str:
    if size <= 0:
        return ""
    fh.seek(max(0, size - 64))
    return hashlib.sha1(fh.read(min(64, size))).hexdigest()


class Bank:
    """Facts are read lazily, one attribute at a time. Each shard has a sidecar index
    (``<shard>.idx``) of where every attribute's records sit, kept current by scanning only what
    was appended since. Writers take an exclusive file lock and first take in anything another
    process appended, so several processes can share one bank."""

    def __init__(self, root: Path | str | None = None, use_index: bool = True):
        self.root = Path(root) if root else config.home() / "bank"
        self.use_index = use_index  # False: ignore and never write sidecar indexes (integrity checks)
        self.root.mkdir(parents=True, exist_ok=True)
        self._facts: dict[str, dict[str, Fact]] = {}
        self._down: dict[str, dict[str, str]] = {}
        self._index: dict[str, dict[str, list[Fact]]] = {}
        self._keys: dict[str, dict[str, list[str]]] = {}
        self._idx: dict[str, _Index] = {}
        self._applied: dict[str, dict[str, int]] = {}  # entity -> attr -> index entries already in memory
        self._ops_applied: dict[str, int] = {}
        self.journal_errors: list[str] = []

    # ---------- storage ----------
    def shard(self, entity: str) -> Path:
        safe = re.sub(r"[^A-Z0-9._-]", "_", entity.upper())
        return self.root / f"{safe}.jsonl"

    def _idx_path(self, entity: str) -> Path:
        return self.shard(entity).with_suffix(".jsonl.idx")

    def entities(self) -> list[str]:
        """One name per shard, read from its first record. A shard whose first record cannot be
        read (a torn write, manual damage) still counts, under its file name, and the problem goes
        to ``journal_errors``: one bad line must not stop every caller."""
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

    def _state(self, entity: str) -> None:
        if entity not in self._facts:
            self._facts[entity], self._down[entity], self._index[entity], self._keys[entity] = {}, {}, {}, {}
            self._applied[entity], self._ops_applied[entity] = {}, 0

    def _reset(self, entity: str) -> None:
        for d in (self._facts, self._down, self._index, self._keys, self._idx, self._applied, self._ops_applied):
            d.pop(entity, None)
        self._state(entity)

    def _read_idx(self, entity: str) -> _Index | None:
        if not self.use_index:
            return None
        try:
            ix = _Index.from_json(json.loads(self._idx_path(entity).read_text()))
        except (OSError, ValueError, KeyError, TypeError):
            return None
        self.journal_errors += ix.bad
        return ix

    def _save_idx(self, entity: str) -> None:
        ix = self._idx.get(entity)
        if ix is None or not ix.dirty or not self.use_index:
            return
        p = self._idx_path(entity)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(ix.to_json(), separators=(",", ":")))
            os.replace(tmp, p)
            ix.dirty = False
        except OSError:
            tmp.unlink(missing_ok=True)  # the index is a cache; the journal is the record

    def _sync(self, entity: str) -> _Index:
        """Bring the index up to date with the shard (scan only what was appended since)."""
        self._state(entity)
        p = self.shard(entity)
        ix = self._idx.get(entity)
        if ix is None:
            ix = self._idx[entity] = self._read_idx(entity) or _Index()
        try:
            st = os.stat(p)
        except FileNotFoundError:
            return ix
        with p.open("rb") as fh:
            stale = (ix.inode is not None and ix.inode != st.st_ino) or st.st_size < ix.size \
                or _tail_hash(fh, ix.size) != ix.tail
            if stale:  # rewritten, not appended to: forget everything about it and start over
                self._reset(entity)
                ix = self._idx[entity] = _Index()
            if st.st_size > ix.size:
                self._scan(entity, ix, fh, st)
        return ix

    def _scan(self, entity: str, ix: _Index, fh, st) -> None:
        fh.seek(ix.size)
        data = fh.read(st.st_size - ix.size)
        name, base, pos = self.shard(entity).name, ix.size, 0
        while True:
            nl = data.find(b"\n", pos)
            if nl < 0:
                break
            line, off = data[pos:nl], base + pos
            ix.lines += 1
            if line.strip():
                self._classify(name, ix, line, off, ix.lines)
            pos = nl + 1
        if pos < len(data) and data[pos:].strip():  # a last line without its newline: torn, or still being written
            self.journal_errors.append(f"{name}:{ix.lines + 1}: unparseable (incomplete last line)")
        ix.size, ix.inode = base + pos, st.st_ino
        ix.tail = _tail_hash(fh, ix.size)
        ix.dirty = True

    def _classify(self, name: str, ix: _Index, line: bytes, off: int, n: int) -> None:
        m = _ATTR.search(line) if line.startswith(_ADD_HEAD) and line.endswith(_ADD_TAIL) else None
        if m:
            attr = json.loads(b'"' + m.group(1) + b'"')
            ix.attrs.setdefault(attr, array("q")).extend((off, len(line), n))
            return
        try:
            rec = json.loads(line)
        except ValueError as e:
            self._bad(ix, f"{name}:{n}: unparseable ({e})")
            return
        if not isinstance(rec, dict):
            self._bad(ix, f"{name}:{n}: malformed record (not an object)")
        elif rec.get("op") == "add" and isinstance(rec.get("fact"), dict) and isinstance(rec["fact"].get("attr"), str):
            ix.attrs.setdefault(rec["fact"]["attr"], array("q")).extend((off, len(line), n))
        else:
            ix.ops.extend((off, len(line), n))

    def _bad(self, ix: _Index, msg: str) -> None:
        self.journal_errors.append(msg)
        ix.bad.append(msg)

    def _read_records(self, entity: str, triples) -> list[tuple[int, bytes]]:
        out = []
        with self.shard(entity).open("rb") as fh:
            for i in range(0, len(triples), 3):
                fh.seek(triples[i])
                out.append((triples[i + 2], fh.read(triples[i + 1])))
        return out

    def _catch_up(self, entity: str, attr: str, retried: bool = False) -> None:
        """Read into memory the records of one attribute that are not there yet."""
        ix = self._idx[entity]
        arr = ix.attrs.get(attr)
        done = self._applied[entity].get(attr, 0)
        if arr is None or done * 3 >= len(arr):
            self._applied[entity][attr] = done
            return
        name = self.shard(entity).name
        recs = []
        for n, raw in self._read_records(entity, arr[done * 3:]):
            try:
                rec = json.loads(raw)
                ok = rec.get("op") == "add" and rec["fact"]["attr"] == attr
            except (ValueError, KeyError, TypeError, AttributeError):
                ok = False
            if not ok:  # the index does not match the file: rebuild it once from the journal
                if retried:
                    self.journal_errors.append(f"{name}:{n}: index points at a record it does not describe")
                    continue
                self._rebuild(entity, also=attr)
                return
            recs.append((n, rec))
        for n, rec in recs:
            self._apply(entity, rec, where=f"{name}:{n}")
        self._applied[entity][attr] = len(arr) // 3

    def _rebuild(self, entity: str, also: str | None = None) -> None:
        loaded = list(self._applied.get(entity, {})) + ([also] if also else [])
        self._reset(entity)
        self._idx_path(entity).unlink(missing_ok=True)
        self._idx[entity] = _Index()
        self._sync(entity)
        self._catch_up_ops(entity)
        for a in dict.fromkeys(loaded):
            self._catch_up(entity, a, retried=True)
        self._idx[entity].dirty = True
        self._save_idx(entity)

    def _catch_up_ops(self, entity: str) -> None:
        ix = self._idx[entity]
        done = self._ops_applied[entity]
        if done * 3 >= len(ix.ops):
            return
        name = self.shard(entity).name
        for n, raw in self._read_records(entity, ix.ops[done * 3:]):
            try:
                self._apply(entity, json.loads(raw), where=f"{name}:{n}", check_known=False)
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                self.journal_errors.append(f"{name}:{n}: malformed record ({type(e).__name__}: {e})")
        self._ops_applied[entity] = len(ix.ops) // 3

    def _ensure(self, entity: str, attr: str) -> None:
        entity = entity.upper()
        self._sync(entity)
        self._catch_up_ops(entity)
        self._catch_up(entity, attr)
        self._save_idx(entity)

    def _load(self, entity: str) -> None:
        """Every attribute of one entity (checks, digests, a full listing)."""
        entity = entity.upper()
        ix = self._sync(entity)
        self._catch_up_ops(entity)
        for a in list(ix.attrs):
            self._catch_up(entity, a)
        self._save_idx(entity)

    def refresh(self) -> None:
        """Take in whatever other processes appended to shards this Bank has read."""
        for entity in list(self._applied):
            self._sync(entity)
            self._catch_up_ops(entity)
            for a in list(self._applied[entity]):
                self._catch_up(entity, a)
            self._save_idx(entity)

    def _apply(self, entity: str, rec: dict, where: str = "", check_known: bool = True) -> None:
        op = rec.get("op")
        if op == "add":
            try:
                f = Fact.from_dict(rec["fact"])
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                self.journal_errors.append(f"{where}: malformed record ({type(e).__name__}: {e})")
                return
            stored = rec["fact"].get("id")
            if stored:
                # Trusted on load for speed; `verify` (run by `brain check`) recomputes every id.
                object.__setattr__(f, "_id", stored)
            self._insert(entity, f)
        elif op == "downgrade":
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

    def _write(self, entity: str, recs: list[dict]) -> None:
        """Append under an exclusive lock, after taking in what other processes appended."""
        p = self.shard(entity)
        lines = [canon(r).encode() + b"\n" for r in recs]
        with p.open("ab") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                ix = self._sync(entity)
                self._catch_up_ops(entity)
                for a in list(self._applied[entity]):
                    self._catch_up(entity, a)
                ix = self._idx[entity]
                end = os.fstat(fh.fileno()).st_size
                if end != ix.size:  # a torn last line from a crashed writer: end it so it stays alone
                    fh.write(b"\n")
                    fh.flush()
                    ix = self._sync(entity)
                    end = ix.size
                for r, b in zip(recs, lines):
                    ix.lines += 1
                    triple = (end, len(b) - 1, ix.lines)
                    if r.get("op") == "add":
                        ix.attrs.setdefault(r["fact"]["attr"], array("q")).extend(triple)
                    else:
                        ix.ops.extend(triple)
                    fh.write(b)
                    end += len(b)
                fh.flush()
                ix.size, ix.inode, ix.dirty = end, os.fstat(fh.fileno()).st_ino, True
                with p.open("rb") as rf:
                    ix.tail = _tail_hash(rf, end)
                for r in recs:  # our own records are in memory already
                    if r.get("op") == "add":
                        a = r["fact"]["attr"]
                        self._applied[entity][a] = len(ix.attrs[a]) // 3
                    else:
                        self._ops_applied[entity] = len(ix.ops) // 3
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        self._save_idx(entity)

    # ---------- writes ----------
    def add(self, fact: Fact) -> bool:
        return self.add_many([fact]) == 1

    def add_many(self, facts: Iterable[Fact]) -> int:
        by_entity: dict[str, list[Fact]] = {}
        for f in facts:
            by_entity.setdefault(f.entity, []).append(f)
        n = 0
        for e, fs in by_entity.items():
            for a in {f.attr for f in fs}:  # dedupe against what the journal already holds
                self._ensure(e, a)
            recs = []
            for f in fs:
                if f.id in self._facts[e]:
                    continue
                self._insert(e, f)
                recs.append({"op": "add", "fact": f.to_dict()})
            if recs:
                self._write(e, recs)
                n += len(recs)
        return n

    def downgrade(self, entity: str, fact_id: str, reason: str) -> None:
        """Weight a fact to zero (it stays in the journal, with the reason)."""
        entity = entity.upper()
        if self.get(f"{entity}/{fact_id}") is None:
            raise KeyError(f"{entity}/{fact_id} not in bank")
        self._down[entity][fact_id] = reason
        self._write(entity, [{"op": "downgrade", "entity": entity, "id": fact_id, "reason": reason, "at": now_iso()}])

    # ---------- reads ----------
    def find(self, entity: str, prefix: str) -> list[Fact]:
        """Facts of one entity whose id starts with ``prefix``, found without reading the whole shard."""
        entity, prefix = entity.upper(), prefix.lower()
        if not re.fullmatch(r"[0-9a-f]{1,20}", prefix):
            return []
        self._sync(entity)
        self._catch_up_ops(entity)
        p = self.shard(entity)
        if not p.exists():
            return []
        pat = re.compile(rb'"id":"' + prefix.encode() + rb'[0-9a-f]*","inputs"')
        data = p.read_bytes()
        attrs = set()
        for m in pat.finditer(data):
            start = data.rfind(b"\n", 0, m.start()) + 1
            end = data.find(b"\n", m.end())
            hit = _ATTR.search(data, start, end if end >= 0 else len(data))
            if hit:
                attrs.add(json.loads(b'"' + hit.group(1) + b'"'))
        for a in attrs:
            self._catch_up(entity, a)
        self._save_idx(entity)
        return [f for a in attrs for f in self._index[entity].get(a, []) if f.id.startswith(prefix)]

    def get(self, ref: str) -> Fact | None:
        entity, _, fid = ref.partition("/")
        hits = [f for f in self.find(entity, fid) if f.id == fid]
        return hits[0] if hits else None

    def facts(self, entity: str, attr: str | None = None, include_downgraded: bool = False) -> list[Fact]:
        entity = entity.upper()
        if attr is None:
            self._load(entity)
            src = [f for lst in self._index[entity].values() for f in lst]
        else:
            self._ensure(entity, attr)
            src = list(self._index[entity].get(attr, []))
        if not include_downgraded:
            src = [f for f in src if f.id not in self._down[entity]]
        return src

    def attrs(self, entity: str) -> list[str]:
        ix = self._sync(entity.upper())
        self._save_idx(entity.upper())
        return sorted(a for a, v in ix.attrs.items() if len(v))

    def is_downgraded(self, entity: str, fact_id: str) -> bool:
        entity = entity.upper()
        self._sync(entity)
        self._catch_up_ops(entity)
        return fact_id in self._down[entity]

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
        """Recompute every fact's id from its content; a mismatch means the journal was altered.
        Also reports downgrades of facts the journal never added."""
        entity = entity.upper()
        self._load(entity)
        out = [f"{self.shard(entity).name}: stored id {fid} != content id {f.content_id()}"
               for fid, f in self._facts[entity].items() if f.content_id() != fid]
        out += [f"{self.shard(entity).name}: downgrade of unknown fact {fid}"
                for fid in self._down[entity] if fid not in self._facts[entity]]
        return out

    def replay_digest(self, entity: str) -> str:
        """Digest of the state rebuilt from the journal file alone (the sidecar index ignored)."""
        fresh = Bank(self.root)
        fresh._idx[entity.upper()] = _Index()
        fresh._state(entity.upper())
        return fresh.digest(entity)


class View:
    """The bank as it stood at time t. Only facts with known_at <= t are visible."""

    def __init__(self, bank: Bank, t: str):
        self.bank = bank
        self.cutoff = norm_known_at(t)
        self.t = self.cutoff[:10]

    def facts(self, entity: str, attr: str, where: Callable[[Fact], bool] | None = None) -> list[Fact]:
        entity = entity.upper()
        self.bank._ensure(entity, attr)
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
