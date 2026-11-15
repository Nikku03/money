"""The agent's memory bank (docs/MEMORY.md): everything said and done, kept outside the model and searched on demand.

A model's context window is fixed. Long conversations are summarized and the details drop out, and the machine is
reset now and then. So nothing here makes the window larger. It works like a library beside a desk: the window is
the desk, this is the library, and the agent fetches the pages it needs.

What the library holds:

- **talk**: every turn of the conversation, from the session transcripts on this machine (``capture``): the person's
  messages, the agent's replies, its reasoning, each tool call and the head and tail of its result, and the
  summaries written when the context was compacted. Kept in the local home (``var/memory/talk.jsonl``), with email
  addresses and keys masked, and on the person's Google Drive (``save`` packs it; ``load --talk`` merges it back).
- **notes**: ``memory/MEMORY.md`` (where the work stands, read first in every session), ``memory/JOURNAL.md``
  (what happened, by date) and ``memory/notes.jsonl`` (facts the agent chose to keep, ``remember``). Committed.
- **docs**: everything the repo writes down: ``CLAUDE.md``, ``README.md``, ``docs/``, ``research/`` notes and
  results, the desk's recorded notes, and each module's opening docstring. Read in place.
- **claims**: the claims the agent noted from pages it read, the one part of the bank that cannot be downloaded
  again (``memory/claims.jsonl``).

``recall`` ranks every passage of the library against a question (BM25, with no model and no network) and prints
the best ones with their date and source. ``show`` prints one passage in full, a tool call with its result.
``start`` prints what a new session needs first: where the work stands, the person's latest messages, and whether
the local home was reset.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from brain import config
from brain.bank import Bank, Fact, canon

CLAIMS = "claims.jsonl"
NOTES_FILE = "notes.jsonl"
NOTES = ("MEMORY.md", "JOURNAL.md")
TALK = "talk.jsonl"
DRIVE_FOLDER = "money-brain memory"
REBUILD = (
    ("prices and the latest filings of every scan stock", "python -m brain scan run --refresh", "about 40 minutes"),
    ("company accounts and what each prospectus sells", "python -m brain mosaic read", "about 1-2 hours from empty"),
    ("the mosaic's model is committed; only its data needs reading", "python -m brain mosaic radar", "a few minutes"),
)
RESULT_HEAD, RESULT_TAIL = 1800, 700     # characters of a tool's result kept: its start and its end
PART_CHARS = 900                          # a long message is kept as pieces of about this size
TOOL_CHARS, THOUGHT_CHARS = 1500, 4000
CHUNK_CHARS = 1200                        # a document is cut into passages of about this size, at paragraph breaks
WEIGHT = {"you": 1.3, "note": 1.3, "summary": 1.2, "me": 1.2, "doc": 1.0, "result": 0.9, "tool": 0.6,
          "thought": 0.8}
K1, B = 1.2, 0.75
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SECRET = re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abp]-[A-Za-z0-9-]{10,}|"
                     r"AKIA[0-9A-Z]{16})\b|((?:api_?key|token|password|secret)=)[^&\s\"']+", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9][a-z0-9_]*")
_SHOWN = re.compile(r"\(show d?[0-9a-f]{15,16}\)")      # a line of recall's own output
STOP = frozenset(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "so", "that", "the", "their", "then", "there", "these", "this", "to", "was", "were", "what", "when", "which", "who", "will", "with", "you", "your", "we", "our", "not", "no", "do", "does", "did", "can", "could", "would", "should"])


# ---------- where things are ----------
def memory_dir() -> Path:
    """``memory/`` in the repository, or BRAIN_MEMORY (tests, trials)."""
    return Path(os.environ.get("BRAIN_MEMORY") or config.ROOT / "memory")


def talk_dir() -> Path:
    """The local home's ``memory/``: the captured conversation and the capture's place in each transcript."""
    p = config.home() / "memory"
    p.mkdir(parents=True, exist_ok=True)
    return p


def transcripts() -> list[Path]:
    """This repository's session transcripts on this machine (BRAIN_TRANSCRIPTS: a glob, for tests)."""
    pattern = os.environ.get("BRAIN_TRANSCRIPTS")
    if not pattern:
        project = "-" + str(config.ROOT.resolve()).strip("/").replace("/", "-")
        pattern = str(Path.home() / ".claude" / "projects" / project / "*.jsonl")
    return [Path(p) for p in sorted(glob.glob(pattern))]


# ---------- talk: the conversation, captured ----------
def mask(text: str) -> str:
    """Email addresses and keys out of a text before it is kept."""
    text = _EMAIL.sub("[email]", text)
    return _SECRET.sub(lambda m: (m.group(2) or "") + "[secret]", text)


def _clip(text: str, head: int, tail: int = 0) -> str:
    if len(text) <= head + tail + 20:
        return text
    return text[:head] + f"\n[... {len(text) - head - tail} characters left out ...]\n" + (text[-tail:] if tail else "")


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def chunks_of(line: dict) -> list[dict]:
    """The passages one transcript line holds. Kept: the person's messages, compaction summaries, the agent's replies,
    reasoning and tool calls, and tool results (clipped). Left out: system reminders, notifications, bookkeeping."""
    kind_of = line.get("type")
    msg = line.get("message") or {}
    content = msg.get("content")
    base = {"at": line.get("timestamp") or "", "session": line.get("sessionId") or ""}
    uid = line.get("uuid") or hashlib.sha1(canon(line).encode()).hexdigest()
    out = []

    def add(i: int, kind: str, text: str, **extra) -> None:
        text = mask(text).strip()
        if not text:
            return
        whole = hashlib.sha1(f"{uid}:{i}".encode()).hexdigest()[:16]
        parts = pieces(text)
        for p, piece in enumerate(parts):
            pid = whole if p == 0 else hashlib.sha1(f"{uid}:{i}:{p}".encode()).hexdigest()[:16]
            out.append({"id": pid, **base, "kind": kind, "text": piece, **extra,
                        **({"of": whole, "part": p} if len(parts) > 1 else {})})

    if kind_of == "user":
        if line.get("isCompactSummary"):
            add(0, "summary", _text_of(content))
        elif (line.get("origin") or {}).get("kind") == "human" and not line.get("isMeta"):
            add(0, "you", _text_of(content))
        elif isinstance(content, list):
            for i, b in enumerate(content):
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    add(i, "result", _clip(_text_of(b.get("content")) or str(b.get("content") or ""),
                                           RESULT_HEAD, RESULT_TAIL), tid=b.get("tool_use_id"))
    elif kind_of == "assistant" and isinstance(content, list):
        for i, b in enumerate(content):
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                add(i, "me", b.get("text", ""))
            elif b.get("type") == "thinking":
                add(i, "thought", _clip(b.get("thinking", ""), THOUGHT_CHARS))
            elif b.get("type") == "tool_use":
                inp = b.get("input") or {}
                what = inp.get("description") or ""
                body = inp.get("command") or inp.get("query") or inp.get("prompt") or canon(inp)
                add(i, "tool", _clip(f"{b.get('name')}: {what}\n{body}", TOOL_CHARS), tid=b.get("id"))
    return out


def pieces(text: str, size: int = PART_CHARS) -> list[str]:
    """A long text cut into pieces of about ``size`` characters, at paragraph or line breaks where it can be, so
    that each piece has one meaning to rank by (a long summary read as one block blurs). A short text is one piece."""
    if len(text) <= size * 1.3:
        return [text]
    out, buf = [], ""
    for para in re.split(r"(\n\s*\n|\n)", text):
        if len(buf) + len(para) > size and buf.strip():
            out.append(buf.strip())
            buf = ""
        buf += para
        while len(buf) > size * 1.5:                     # one long line: cut at a space
            cut = buf.rfind(" ", 0, size) if buf.rfind(" ", 0, size) > size // 2 else size
            out.append(buf[:cut].strip())
            buf = buf[cut:]
    if buf.strip():
        out.append(buf.strip())
    return out


def read_talk(path: Path | None = None) -> list[dict]:
    p = Path(path or talk_dir() / TALK)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def capture(log=None) -> int:
    """Add every transcript line not read yet to the talk. Each transcript is read from where the last capture
    stopped; a passage already kept is not kept twice. Returns how many passages were added."""
    state_p = talk_dir() / "capture.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    have = {c["id"] for c in read_talk()}
    new = []
    for t in transcripts():
        start = state.get(str(t), 0)
        size = t.stat().st_size
        if size < start:                         # a transcript rewritten from the start: read it again
            start = 0
        with t.open("rb") as fh:
            fh.seek(start)
            data = fh.read()
        end = data.rfind(b"\n") + 1               # a line still being written waits for the next capture
        for raw in data[:end].splitlines():
            try:
                line = json.loads(raw)
            except ValueError:
                continue
            for c in chunks_of(line):
                if c["id"] not in have:
                    have.add(c["id"])
                    new.append(c)
        state[str(t)] = start + end
    if new:
        with (talk_dir() / TALK).open("a") as fh:
            fh.writelines(canon(c) + "\n" for c in new)
    state_p.write_text(canon(state))
    if log:
        log(f"captured {len(new)} new passages from {len(transcripts())} transcripts")
    return len(new)


def _unwrap(path: Path) -> bytes:
    """A file's bytes. Drive's download tool may save a file as JSON with its bytes base64-encoded under
    ``content``; that is unwrapped."""
    raw = Path(path).read_bytes()
    if raw[:1] == b"{":
        try:
            import base64
            return base64.b64decode(json.loads(raw)["content"])
        except (ValueError, KeyError):
            pass
    return raw


def merge_vectors(path: Path) -> int:
    """Merge a vector file (``save``'s ``vectors.npz``, downloaded from Drive) into this home's cache, so a reset
    does not mean embedding everything again. Returns how many vectors were new here."""
    import io
    z = np.load(io.BytesIO(_unwrap(path)))
    p = talk_dir() / VECTORS
    have: dict = {}
    if p.exists():
        y = np.load(p)
        have = dict(zip(y["keys"].tolist(), y["vecs"]))
    new = {k: v for k, v in zip(z["keys"].tolist(), z["vecs"]) if k not in have}
    have.update(new)
    if have:
        np.savez(p, keys=np.array(list(have)), vecs=np.stack(list(have.values())))
    return len(new)


def merge_talk(path: Path) -> int:
    """Merge a packed talk (``save``'s ``talk.jsonl.gz``, downloaded from Drive) into this home's talk."""
    raw = _unwrap(path)
    text = gzip.decompress(raw).decode() if raw[:2] == b"\x1f\x8b" else raw.decode()
    have = {c["id"] for c in read_talk()}
    new = [c for c in (json.loads(ln) for ln in text.splitlines() if ln.strip()) if c["id"] not in have]
    if new:
        with (talk_dir() / TALK).open("a") as fh:
            fh.writelines(canon(c) + "\n" for c in sorted(new, key=lambda c: c["at"]))
    return len(new)


NOT_A_PERSON = ("<", "Stop hook feedback", "[Image", "[Request interrupted", "Caveat:")


def event_lines(path: Path) -> list[dict]:
    """Transcript lines from one saved page of another session's events (the remote-session tool's list_events,
    as saved to a file), oldest first, in the shape ``chunks_of`` reads. That session's own reasoning is not shared
    (it comes back empty) and is left out; its messages, replies, tool calls, results and summaries are kept."""
    text = Path(path).read_text()
    body = next((ln for ln in text.split("\n") if ln.lstrip().startswith(("{\"ccr\"", "{\"data\""))), None)
    if body is None:
        return []
    page = json.loads(body)
    out = []
    for e in (page.get("ccr") or page).get("data") or []:
        kind = next((k for k in e if k in ("user", "assistant")), None)
        if kind is None:
            continue
        inner = e[kind]
        ia = inner.get("internal_anthropic_catchall") or {}
        msg = ia.get("message") or inner.get("message") or {}
        line = {"type": kind, "uuid": inner.get("uuid") or ia.get("uuid"), "message": msg,
                "timestamp": ia.get("timestamp") or inner.get("timestamp") or e.get("created_at") or "",
                "sessionId": ia.get("session_id") or inner.get("session_id") or ""}
        content = msg.get("content")
        if kind == "user" and (isinstance(content, str) or (isinstance(content, list) and content
                                                             and all(b.get("type") == "text" for b in content))):
            said = _text_of(content).lstrip()
            if said.startswith("This session is being continued"):
                line["isCompactSummary"] = True
            elif not said.startswith(NOT_A_PERSON):        # notifications, hook messages, image notes
                line["origin"] = {"kind": "human"}
        out.append(line)
    return out


def import_events(paths: list[Path], session: str | None = None) -> int:
    """Add another session's saved event pages to the talk, each passage labelled with ``session`` (its name in
    the session list) when given. A passage already kept is not kept twice. Returns how many were added."""
    have = {c["id"] for c in read_talk()}
    new = []
    for p in paths:
        for line in event_lines(Path(p)):
            if session:
                line["sessionId"] = session
            for c in chunks_of(line):
                if c["id"] not in have:
                    have.add(c["id"])
                    new.append(c)
    if new:
        with (talk_dir() / TALK).open("a") as fh:
            fh.writelines(canon(c) + "\n" for c in sorted(new, key=lambda c: c["at"]))
    return len(new)


def pack_talk() -> Path:
    """The talk, compressed, for the Drive copy."""
    p = talk_dir() / (TALK + ".gz")
    src = talk_dir() / TALK
    p.write_bytes(gzip.compress(src.read_bytes() if src.exists() else b"", mtime=0))
    return p


# ---------- notes and docs ----------
def remember(text: str, topic: str = "") -> dict:
    """Keep one fact in ``memory/notes.jsonl``: something the agent will want later and that no file states."""
    rec = {"at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "topic": topic, "text": mask(text)}
    p = memory_dir() / NOTES_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(canon(rec) + "\n")
    return rec


def split_markdown(text: str, src: str, limit: int = CHUNK_CHARS) -> list[dict]:
    """Passages of a markdown text: by heading, then at paragraph breaks under ``limit`` characters. Each passage
    starts with the headings it sits under."""
    out, heads, buf = [], [], []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            title = " > ".join(heads)
            out.append({"src": src, "head": title, "text": (title + "\n" if title else "") + body})
        buf.clear()

    for ln in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            flush()
            heads[:] = heads[:len(m.group(1)) - 1] + [m.group(2).strip()]
            continue
        if not ln.strip() and sum(len(x) for x in buf) > limit:
            flush()
        buf.append(ln)
        if sum(len(x) for x in buf) > 2 * limit:     # one long paragraph or table: cut it anyway
            flush()
    flush()
    return out


def _desk_passages(p: Path, src: str) -> list[dict]:
    try:
        n = json.loads(p.read_text())
    except ValueError:
        return []
    note = n.get("note") or n
    out = [{"src": src, "head": "backdrop", "text": f"desk note {n.get('check', '')}: {note.get('backdrop', '')}"}]
    for th in note.get("themes") or []:
        out.append({"src": src, "head": th.get("name", ""), "text": " ".join(
            str(th.get(k, "")) for k in ("name", "cause", "mechanism", "beneath", "watch")) + " members: "
            + ", ".join(th.get("members") or [])})
    for s in note.get("singles") or []:
        out.append({"src": src, "head": s.get("t", ""), "text": f"{s.get('t')}: {s.get('cause', '')} {s.get('beneath', '')}"})
    if note.get("threads"):
        out.append({"src": src, "head": "open questions", "text": "open questions: " + " | ".join(note["threads"])})
    return out


def documents() -> list[dict]:
    """Every passage of the repo's written record, with where it is."""
    root, mem = config.ROOT, memory_dir()
    files = [root / "CLAUDE.md", root / "README.md", *sorted((root / "docs").glob("*.md")),
             *sorted((root / "research").glob("*.md")), *sorted(mem.glob("*.md"))]
    out = []
    for f in files:
        if f.exists():
            rel = f.relative_to(root) if f.is_relative_to(root) else f
            out += [{**p, "kind": "note" if f.parent == mem else "doc"} for p in split_markdown(f.read_text(), str(rel))]
    for f in sorted((root / "research" / "desk").glob("*/*.json")):
        out += [{**p, "kind": "doc"} for p in _desk_passages(f, str(f.relative_to(root)))]
    notes = mem / NOTES_FILE
    if notes.exists():
        for ln in notes.read_text().splitlines():
            if ln.strip():
                n = json.loads(ln)
                out.append({"src": str(notes.relative_to(root)) if notes.is_relative_to(root) else str(notes),
                            "head": n.get("topic", ""), "kind": "note", "at": n["at"],
                            "text": (f"[{n['topic']}] " if n.get("topic") else "") + n["text"]})
    for f in sorted((root / "brain").rglob("*.py")):
        m = re.match(r'\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', f.read_text(), re.DOTALL)
        if m and len(m.group(1)) > 80:
            out.append({"src": str(f.relative_to(root)), "head": "module", "kind": "doc",
                        "text": f"{f.relative_to(root)}: {m.group(1).strip()[:3000]}"})
    for i, p in enumerate(out):
        p.setdefault("id", "d" + hashlib.sha1(f"{p['src']}:{p['head']}:{i}".encode()).hexdigest()[:15])
    return out


# ---------- recall ----------
# A few words people and the agent use for the same thing; a question is widened by them before it is ranked.
SAME = [{"prefer", "want", "like", "wish"}, {"explain", "explanation", "simple", "plain", "simply"},
        {"fail", "failed", "failure"}, {"pass", "passed"}, {"result", "verdict", "outcome"},
        {"bug", "flaw", "error", "mistake", "wrong"}, {"fix", "fixed", "repair"},
        {"user", "person", "asked", "asks"}, {"stock", "share", "ticker"}, {"sale", "sell", "offering", "raise"},
        {"plan", "next", "todo", "pending", "open"}, {"decide", "decided", "decision", "chose", "agreed"},
        {"schedule", "scheduled", "trigger", "cron"}, {"memory", "remember", "recall", "context"}]


def widen(query: str) -> str:
    """The question with the other words of each group in SAME that it uses."""
    ws = set(words(query))
    extra = [w for group in SAME if ws & {words(x)[0] for x in group if words(x)} for w in sorted(group)]
    return query + (" " + " ".join(extra) if extra else "")


def words(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if w in STOP or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _about_memory(x: dict) -> bool:
    """A call of this command, or its output: the library reading itself."""
    return x["kind"] in ("tool", "result") and any(m in x["text"] for m in ("brain memory", "brain.memory",
                                                                            "memory.py", "from brain import memory")) \
        or x["kind"] == "result" and (x["text"].startswith("captured ") or _SHOWN.search(x["text"]) is not None)


def library(kinds: set[str] | None = None) -> list[dict]:
    """Everything recall can read: the talk (captured first) and the written record."""
    capture()
    items = [x for x in read_talk() if not _about_memory(x)] + documents()
    return [x for x in items if not kinds or x["kind"] in kinds]


def rank(items: list[dict], query: str, k: int = 8, since: str | None = None) -> list[tuple[float, dict]]:
    """The ``k`` passages that best answer ``query`` (BM25 over words, weighted by kind of passage)."""
    q = Counter(words(query))
    if not q:
        return []
    items = [x for x in items if not since or (x.get("at") or "") >= since]   # with a date asked for, undated docs drop out
    toks = [words(x["text"]) for x in items]
    n = len(items)
    if not n:
        return []
    avg = sum(len(t) for t in toks) / n
    df = Counter(w for t in toks for w in set(t) if w in q)
    idf = {w: math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in q if df[w]}
    scored = []
    for x, t in zip(items, toks):
        if not t:
            continue
        tf = Counter(w for w in t if w in idf)
        if not tf:
            continue
        s = sum(idf[w] * tf[w] * (K1 + 1) / (tf[w] + K1 * (1 - B + B * len(t) / avg)) for w in tf)
        scored.append((s * WEIGHT.get(x["kind"], 1.0), x))
    scored.sort(key=lambda p: -p[0])
    return scored[:k]


def snippet(text: str, query: str, width: int = 500) -> str:
    """The part of a passage where the query's words cluster most."""
    if len(text) <= width:
        return text
    q = set(words(query))
    low = text.lower()
    best, at = -1, 0
    for start in range(0, len(text) - width + 1, max(width // 5, 1)):
        hits = sum(1 for w in q if w in low[start:start + width])
        if hits > best:
            best, at = hits, start
    return ("... " if at else "") + text[at:at + width].strip() + (" ..." if at + width < len(text) else "")


# ---------- meaning: a small embedding model, when it is installed ----------
MODEL = "BAAI/bge-small-en-v1.5"   # 384 numbers per passage; about 65 MB; runs on the CPU (pip install fastembed)
VECTORS = "vectors.npz"
EMBED_CHARS = 2000                 # the model reads about 512 tokens; a passage's start is what it embeds
POOL, RRF_K = 100, 60              # each ranking's top POOL meet; a passage scores 1/(RRF_K + its place) in each
DIM = 384
RECALL_EMBEDS = 60                 # passages a recall embeds before answering; index and save embed the rest
_model = None


def embedder():
    """The embedding model, loaded once; None when fastembed is not installed or the model cannot be had (then
    recall ranks by words alone)."""
    global _model
    if _model is None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        try:
            from fastembed import TextEmbedding
            _model = TextEmbedding(MODEL, cache_dir=str(talk_dir() / "model"))
        except Exception:  # noqa: BLE001 - not installed, or no network for the first download: words alone
            _model = False
    return _model or None


def _vkey(x: dict) -> str:
    return x["id"] + ":" + hashlib.sha1(x["text"][:EMBED_CHARS].encode()).hexdigest()[:10]


def vectors(items: list[dict], log=None, limit: int | None = None) -> np.ndarray | None:
    """A unit vector per passage, from the cache in the local home; passages new or changed since are embedded and
    the cache rewritten with exactly the current ones. With ``limit``, at most that many are embedded now (the
    newest first) and the rest get a zero vector, which ranks them last by meaning until ``index`` or ``save``
    embeds them. None when there is no model."""
    p = talk_dir() / VECTORS
    keys = [_vkey(x) for x in items]
    cache: dict[str, np.ndarray] = {}
    if p.exists():
        z = np.load(p)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if limit is not None and len(todo) > limit:
        todo = sorted(todo, key=lambda i: items[i].get("at") or "", reverse=True)[:limit]
    if todo:
        model = embedder()
        if model is None:
            return None
        if log:
            log(f"embedding {len(todo)} new passages")
        got = model.embed([items[i]["text"][:EMBED_CHARS] for i in todo], batch_size=32)
        for i, v in zip(todo, got):
            v = np.asarray(v, np.float32)
            cache[keys[i]] = (v / (np.linalg.norm(v) or 1.0)).astype(np.float16)
    live = [k for k in dict.fromkeys(keys) if k in cache]
    if todo or len(cache) != len(live):
        np.savez(p, keys=np.array(live), vecs=np.stack([cache[k] for k in live]) if live else
                 np.zeros((0, DIM), np.float16))
    zero = np.zeros(len(next(iter(cache.values()))) if cache else DIM, np.float16)
    return np.stack([cache.get(k, zero) for k in keys]).astype(np.float32) if keys else np.zeros((0, DIM), np.float32)


def by_meaning(items: list[dict], vecs: np.ndarray, query: str, k: int) -> list[tuple[float, dict]]:
    """The ``k`` passages whose meaning is closest to the question's (cosine of their vectors)."""
    model = embedder()
    if model is None or not len(items):
        return []
    q = np.asarray(next(iter(model.query_embed([query]))), np.float32)
    s = vecs @ (q / (np.linalg.norm(q) or 1.0))
    top = np.argsort(-s)[:k]
    return [(float(s[i]), items[i]) for i in top]


def fuse(by_words: list[tuple[float, dict]], by_sense: list[tuple[float, dict]], k: int) -> list[tuple[float, dict]]:
    """One ranking from two (reciprocal rank fusion): a passage near the top of either comes near the top, and one
    near the top of both comes first. The kind of passage weighs as in BM25."""
    score: dict[str, float] = {}
    item: dict[str, dict] = {}
    for ranking in (by_words, by_sense):
        for place, (_, x) in enumerate(ranking):
            score[x["id"]] = score.get(x["id"], 0.0) + WEIGHT.get(x["kind"], 1.0) / (RRF_K + place + 1)
            item[x["id"]] = x
    return sorted(((s, item[i]) for i, s in score.items()), key=lambda p: -p[0])[:k]


def recall(query: str, k: int = 8, kinds: set[str] | None = None, since: str | None = None,
           meaning: bool = True) -> list[dict]:
    """The passages that best answer ``query``: by words (BM25) and, when the model is there, by meaning, fused.
    A question of nothing but common words asks nothing."""
    if not words(query):
        return []
    every = library()
    keep = [x for x in every if (not kinds or x["kind"] in kinds) and (not since or (x.get("at") or "") >= since)]
    words_ = rank(keep, widen(query), POOL)
    vecs = vectors(every, limit=RECALL_EMBEDS) if meaning else None
    if vecs is None:
        return [{**x, "score": s, "by": "words"} for s, x in words_[:k]]
    mask_ = np.array([(not kinds or x["kind"] in kinds) and (not since or (x.get("at") or "") >= since)
                      for x in every], bool)
    sense = by_meaning([x for x, m in zip(every, mask_) if m], vecs[mask_], query, POOL)
    return [{**x, "score": s, "by": "words and meaning"} for s, x in fuse(words_, sense, k)]


def find(pid: str) -> list[dict]:
    """One passage by its id, whole: a piece comes with the rest of its message, a tool call with its result and a
    result with its call."""
    talk = read_talk()
    hit = [x for x in talk if x["id"] == pid] or [x for x in documents() if x["id"] == pid]
    if not hit:
        return []
    x = hit[0]
    if x.get("tid"):
        group = [y for y in talk if y.get("tid") == x["tid"]]
        return sorted(group, key=lambda y: (y["kind"] != "tool", y.get("part", 0)))
    if x.get("of"):
        return sorted([y for y in talk if y.get("of") == x["of"]], key=lambda y: y["part"])
    return hit


def _when(x: dict) -> str:
    return (x.get("at") or "")[:16].replace("T", " ")


# ---------- the bank's claims ----------
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
    return [describe(f) for f in sorted(d.iterdir()) if f.is_file()]


def describe(f: Path) -> dict:
    return {"name": f.name, "path": str(f.resolve()), "bytes": f.stat().st_size,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}


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


# ---------- the command ----------
def _print_status(bank: Bank) -> None:
    st = status(bank)
    print("The local home holds: " + ", ".join(f"{v} {k}" for k, v in st["have"].items())
          + f"; SPY's prices to {st['spy_last'] or 'none'}; {len(read_talk())} passages of talk.")
    if st["empty"]:
        print("The home looks reset. Rebuild it from public sources, in this order:")
        for what, cmd, took in REBUILD:
            print(f"  {cmd:40} {what} ({took})")
    if not read_talk():
        print(f"No talk in this home: download the newest 'talk' file from the Drive folder '{DRIVE_FOLDER}' and "
              "run python -m brain memory load --talk PATH")


def _cmd(a) -> int:
    if a.action == "recall":
        q = " ".join(a.words)
        hits = recall(q, a.k, set(a.kind) if a.kind else None, a.since, meaning=not a.words_only)
        if not hits:
            print("nothing in the memory bank matches")
        elif hits[0].get("by") == "words":
            print("(ranked by words only: pip install fastembed for ranking by meaning too)")
        for i, h in enumerate(hits, 1):
            where = h.get("src") or f"talk {h.get('session', '').removeprefix('session_')[:8]}"
            print(f"[{i}] {_when(h) or '-':16} {h['kind']:7} {where}  (show {h['id']})")
            print("    " + snippet(h["text"], q, a.width).replace("\n", "\n    "))
        return 0
    if a.action == "show":
        for pid in a.words:
            for x in find(pid) or [{"kind": "-", "text": f"no passage {pid}"}]:
                print(f"--- {_when(x)} {x['kind']} {x.get('src') or x.get('session', '')}\n{x['text']}")
        return 0
    if a.action == "remember":
        rec = remember(" ".join(a.words), a.topic or "")
        print(f"kept in {memory_dir() / NOTES_FILE}: {rec['text']}")
        return 0
    if a.action == "capture":
        capture(log=print)
        return 0
    if a.action == "import":
        n = import_events([Path(x) for x in a.words], a.session)
        sessions = sorted({c["session"] for c in read_talk()})
        print(f"imported {n} passages from {len(a.words)} saved pages; the talk now holds {len(sessions)} sessions")
        return 0
    if a.action == "index":
        every = library()
        v = vectors(every, log=print)
        print(f"{len(every)} passages" + (f", each with a vector ({v.nbytes // 1024} KB in memory)" if v is not None
                                          else "; no model, so words only (pip install fastembed)"))
        return 0
    bank = Bank()
    if a.action == "start":
        capture()
        mem = memory_dir() / "MEMORY.md"
        print(mem.read_text() if mem.exists() else f"no {mem}: load it from the Drive folder '{DRIVE_FOLDER}'")
        mine = [x for x in read_talk() if x["kind"] == "you"][-a.k:]
        print("\nThe person's latest messages:")
        for x in mine:
            print(f"  {_when(x)}  {x['text'][:300]}")
        _print_status(bank)
        return 0
    if a.action == "save":
        capture()
        vectors(library(), log=print)
        n = export_claims(bank)
        print(f"wrote {n} claims to {memory_dir() / CLAIMS}; packed {len(read_talk())} passages of talk")
        print(f"Files to keep on Google Drive (folder '{DRIVE_FOLDER}'), with the sha256 an upload names each by:")
        packed = [describe(pack_talk())] + ([describe(talk_dir() / VECTORS)] if (talk_dir() / VECTORS).exists() else [])
        for f in [*files(), *packed]:
            print(f"  {f['name']:16} {f['bytes']:>9} bytes  sha256:{f['sha256']}:{f['path']}")
        return 0
    if a.action == "load":
        if a.talk:
            print(f"merged {merge_talk(Path(a.talk))} passages of talk from {a.talk}")
        if a.vectors:
            print(f"merged {merge_vectors(Path(a.vectors))} vectors from {a.vectors}")
        print(f"put {import_claims(bank)} claims back into the bank from {memory_dir() / CLAIMS}")
    _print_status(bank)
    print(f"Read {memory_dir() / 'MEMORY.md'} first: it says where the work stands.")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("memory", help="the agent's memory bank: recall what was said and done, keep it across "
                                      "resets (docs/MEMORY.md)")
    p.add_argument("action", choices=["recall", "show", "remember", "start", "capture", "import", "index", "save",
                                      "load", "status"])
    p.add_argument("words", nargs="*", help="recall: the question; show: passage ids; remember: the fact; import: "
                                            "saved pages of another session's events")
    p.add_argument("--k", type=int, default=8, help="recall: how many passages; start: how many messages")
    p.add_argument("--kind", action="append", choices=sorted(WEIGHT), help="recall only these kinds of passage")
    p.add_argument("--since", help="recall only passages from this date (YYYY-MM-DD) on")
    p.add_argument("--width", type=int, default=500, help="recall: characters shown of each passage")
    p.add_argument("--topic", help="remember: a word to file the fact under")
    p.add_argument("--talk", help="load: a talk file downloaded from Drive, to merge in")
    p.add_argument("--session", help="import: the session's name, to label its passages")
    p.add_argument("--vectors", help="load: a vectors file downloaded from Drive, to merge in")
    p.add_argument("--words-only", action="store_true", help="recall: rank by words alone (no model)")
    p.set_defaults(fn=_cmd)
