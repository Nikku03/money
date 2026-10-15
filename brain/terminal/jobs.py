"""Background collection jobs for the terminal.

Ingest runs in a child process, never inside the server: a slow SEC or Yahoo
fetch must not hold the bank lock, and a crash in a source must not take the
terminal down. The child writes to the journal the same way the CLI does; when
it exits the server swaps in a fresh Bank so the new facts appear, and only then
does the job read as finished.

A child runs in its own session and writes its output to a log file that the
server follows, not to a pipe: stopping the server (or Ctrl+C in its terminal)
must not kill a child halfway through an ingest, and a child writing to a pipe
nobody reads any more dies at its next print.

What reaches the child is fixed here. A target is a ticker matching a strict
pattern or one of the keywords, and it becomes one element of an argument
list (never a shell string), so nothing typed into the browser can reach a
command interpreter or smuggle in an option.

Two keywords run more than a fetch. INSIDERS reads SEC's quarterly insider data
sets for the research universes (quarters already read are skipped, so after the
first run it only picks up a newly published quarter). WATCH is one round of the
watchlist loop: collect what is new for everything watched, then check the alert
rules (``python -m brain watch run --once``).

A decision job (``T DEC NEW``, target ``DEC T``) runs ``python -m brain decide T
--ingest --read``: it refreshes T's data, lets the model read the web and make its
call when one is reachable, and records a paper decision (docs/DECIDE.md). It has a
slot of its own, one at a time: an INGEST never waits behind a decision that is
reading pages, and two decisions never compete for one model.
"""
from __future__ import annotations

import codecs
import io
import itertools
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from brain import config
from brain.terminal.commands import dec_line
from brain.timeutil import now_iso

TICKER = re.compile(r"[A-Za-z0-9.^=\-]{1,15}")  # matched with fullmatch: no trailing-newline loophole
KEYWORDS = {"MARKETS": ["markets"], "MACRO": ["macro"], "UNIVERSE": ["ingest-universe"],
            "INSIDERS": ["insiders"], "WATCH": ["watch", "run", "--once"]}
TARGETS_TEXT = "a ticker, MARKETS, MACRO, UNIVERSE, INSIDERS or WATCH"
DECIDE = "DEC "  # a decision job's target is "DEC <TICKER>": its own kind, slot and argument list
MAX_RUNNING = 2  # ingest jobs at once
MAX_DECIDE = 1   # decision jobs at once, in their own slot
LOG_LINES = 200
MAX_QUEUED = 50
KEEP_JOBS = 100  # finished jobs beyond this are forgotten, oldest first
POLL = 0.1  # seconds between reads of a running child's log file


class JobError(ValueError):
    def __init__(self, message: str, hint: str = "INGEST AAPL"):
        super().__init__(message)
        self.message, self.hint = message, hint


def _ticker(t: str, what: str) -> str:
    if not TICKER.fullmatch(t):
        raise JobError(f"not a valid {what} target: {t[:40]!r}")
    if t.startswith("-"):
        raise JobError(f"not a valid {what} target: {t!r} (a ticker cannot start with '-')")
    return t.upper()


def validate(target) -> str:
    """The canonical (upper-case) target, or JobError. Keywords are case-insensitive; ``DEC T`` is a
    decision job on the ticker T (exactly one space, so it is never one token typed as a ticker)."""
    if not isinstance(target, str) or not target:
        raise JobError(f"ingest needs a target: {TARGETS_TEXT}")
    if target[:len(DECIDE)].upper() == DECIDE:
        return DECIDE + _ticker(target[len(DECIDE):], "decision")
    if target.upper() in KEYWORDS:
        return target.upper()
    return _ticker(target, "ingest")


def decide_target(ticker: str) -> str:
    """The target of a decision job on ``ticker``: ``DEC AAPL``."""
    return validate(DECIDE + str(ticker))


def kind_of(target: str) -> str:
    """``decide`` for a decision job's target, ``ingest`` for everything else."""
    return "decide" if target.startswith(DECIDE) else "ingest"


def ticker_of(target: str) -> str | None:
    """The ticker a job is about: T for ``T`` and ``DEC T``, None for the keywords."""
    if kind_of(target) == "decide":
        return target[len(DECIDE):]
    return None if target in KEYWORDS else target


def argv_for(target: str) -> list[str]:
    """The child's argument list for a validated target."""
    if kind_of(target) == "decide":
        return [sys.executable, "-m", "brain", "decide", ticker_of(target), "--ingest", "--read"]
    rest = KEYWORDS.get(target, ["ingest", target])
    return [sys.executable, "-m", "brain", *rest]


def is_ticker(target: str) -> bool:
    """An ingest job of one ticker (a decision job's target is ``DEC T``, not a ticker)."""
    return kind_of(target) == "ingest" and target not in KEYWORDS


@dataclass
class Job:
    id: str
    target: str
    argv: list[str]
    kind: str = "ingest"  # ingest | decide
    status: str = "queued"  # queued | running | done | failed
    started: str | None = None
    finished: str | None = None
    returncode: int | None = None
    log: deque = field(default_factory=lambda: deque(maxlen=LOG_LINES))
    log_file: str | None = None  # the child's output while it runs; not part of the job record

    def to_dict(self) -> dict:
        return {"id": self.id, "target": self.target, "kind": self.kind, "status": self.status,
                "started": self.started, "finished": self.finished, "returncode": self.returncode,
                "log": list(self.log)}


class JobManager:
    """At most ``max_running`` ingest children and ``max_decide`` decision children at once, each kind in
    its own slots; the rest wait in a FIFO queue per kind, so an ingest never waits behind a decision.

    ``argv_for`` maps a validated target to an argument list; tests inject a
    harmless one. ``on_finish(job_dict)`` runs after every job, done or failed
    (a failed ingest may still have written facts), outside this manager's lock
    so it may take the server's bank lock without risking a deadlock. The job
    keeps reading as ``running`` (and counts in ``running()``) until on_finish
    returns: a client that sees it done must also see its facts.
    """

    def __init__(self, on_finish: Callable[[dict], None] | None = None, max_running: int = MAX_RUNNING,
                 argv_for: Callable[[str], list[str]] = argv_for, cwd=config.ROOT, env: dict | None = None,
                 max_decide: int = MAX_DECIDE):
        self.on_finish = on_finish
        self.max_running = max_running
        self.max_decide = max_decide
        self.argv_for = argv_for
        self.cwd = str(cwd)
        self.env = env
        self._jobs: dict[str, Job] = {}
        self._queues: dict[str, deque[Job]] = {"ingest": deque(), "decide": deque()}
        self._workers = {"ingest": 0, "decide": 0}  # threads alive per kind, including their on_finish call
        self._finished = 0  # jobs that stopped being queued or running, since this manager started
        self._ids = itertools.count(1)
        self._cv = threading.Condition()
        self._closed = False

    # ---------- public ----------
    def submit(self, target) -> dict:
        """Queue a job for ``target``; a target already queued or running returns that job."""
        return self.submit_many([target])[0]

    def submit_many(self, targets) -> list[dict]:
        """Queue a job for each target: all of them, or (JobError) none. A target already queued or
        running, or named twice, returns that one job."""
        ts = [validate(t) for t in targets]
        with self._cv:
            if self._closed:
                raise JobError("the server is shutting down", hint="")
            # one job per (kind, ticker): a target names both ("AAPL" ingests AAPL, "DEC AAPL" decides on it)
            active = {j.target: j for j in self._jobs.values() if j.status in ("queued", "running")}
            new = [t for t in dict.fromkeys(ts) if t not in active]
            waiting = 0
            for k, q in self._queues.items():
                free = max(0, self._slots(k) - self._workers[k]) if not q else 0
                waiting += max(0, len(q) + sum(kind_of(t) == k for t in new) - free)
            if waiting > MAX_QUEUED:
                raise JobError(f"{MAX_QUEUED} jobs are already queued; wait for some to finish", hint="JOBS")
            for t in new:
                job = Job(id=f"j{next(self._ids)}", target=t, argv=list(self.argv_for(t)), kind=kind_of(t))
                self._jobs[job.id] = active[t] = job
                self._queues[job.kind].append(job)
            self._forget_old()
            self._pump()
            return [active[t].to_dict() for t in ts]

    def list_jobs(self) -> list[dict]:
        """Newest first."""
        with self._cv:
            return [j.to_dict() for j in reversed(self._jobs.values())]

    def get(self, job_id: str) -> dict | None:
        with self._cv:
            j = self._jobs.get(job_id)
            return j.to_dict() if j else None

    def running(self) -> int:
        with self._cv:
            return sum(j.status == "running" for j in self._jobs.values())

    def queued(self) -> int:
        with self._cv:
            return sum(len(q) for q in self._queues.values())

    def summary(self) -> dict:
        """What /api/status reports, read at one moment: ``running`` and ``queued`` counts, ``active`` (the
        targets queued or running, oldest first: ``["DEC AAPL", "MSFT"]``) and ``finished``, the jobs finished
        since this manager started. ``finished`` only grows, so a page that sees it rise knows a job ended even
        when a queued one started in its place and the running count stayed the same."""
        with self._cv:
            act = [j for j in self._jobs.values() if j.status in ("queued", "running")]
            return {"running": sum(j.status == "running" for j in act), "queued": sum(j.status == "queued" for j in act),
                    "active": [j.target for j in act], "finished": self._finished}

    def wait(self, timeout: float | None = None) -> bool:
        """Block until nothing is queued or running (on_finish included). False on timeout."""
        with self._cv:
            return self._cv.wait_for(lambda: not any(self._queues.values()) and not any(self._workers.values()),
                                     timeout)

    def shutdown(self) -> list[tuple[str, str | None]]:
        """Cancel queued jobs. Running children are left to finish (killing one mid-write could leave a
        torn journal line): each runs in its own session, so Ctrl+C in the server's terminal does not
        reach it, and writes to a log file, so it outlives the server. Returns (target, log file) for
        each job still running."""
        with self._cv:
            self._closed = True
            for q in self._queues.values():
                while q:
                    j = q.popleft()
                    j.status, j.finished = "failed", now_iso()
                    j.log.append("cancelled: the server stopped before this job started")
                    self._finished += 1
            self._cv.notify_all()
            return [(j.target, j.log_file) for j in self._jobs.values() if j.status == "running"]

    # ---------- internals (call with self._cv held) ----------
    def _slots(self, kind: str) -> int:
        return self.max_decide if kind == "decide" else self.max_running

    def _pump(self) -> None:
        for kind, q in self._queues.items():
            while q and self._workers[kind] < self._slots(kind) and not self._closed:
                job = q.popleft()
                job.status, job.started = "running", now_iso()
                self._workers[kind] += 1
                threading.Thread(target=self._run, args=(job,), name=f"job-{job.id}", daemon=True).start()

    def _forget_old(self) -> None:
        done = [k for k, j in self._jobs.items() if j.status in ("done", "failed")]
        for k in done[: max(0, len(self._jobs) - KEEP_JOBS)]:
            del self._jobs[k]

    # ---------- worker ----------
    def _run(self, job: Job) -> None:
        rc = None
        try:
            rc = self._follow(job)
        except Exception as e:  # noqa: BLE001 - a job that cannot start is a failed job, not a server crash
            with self._cv:
                job.log.append(f"could not run job: {type(e).__name__}: {e}")
        with self._cv:
            final = {"status": "done" if rc == 0 else "failed", "returncode": rc, "finished": now_iso()}
            snapshot = {**job.to_dict(), **final}
        try:
            if self.on_finish:
                self.on_finish(snapshot)
        except Exception as e:  # noqa: BLE001
            print(f"job {job.id}: on_finish failed: {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            with self._cv:
                # Published only now, after on_finish swapped in the bank: a client polling /api/status
                # refreshes its panes when a job stops counting as running, and must then see the facts.
                job.status, job.returncode, job.finished = final["status"], final["returncode"], final["finished"]
                self._finished += 1
                self._workers[job.kind] -= 1
                self._pump()
                self._cv.notify_all()

    def _follow(self, job: Job) -> int:
        """Start the child and copy its output into the job log until it exits; its return code."""
        env = dict(self.env if self.env is not None else os.environ, PYTHONUNBUFFERED="1")
        fd, path = tempfile.mkstemp(prefix=f"money-job-{job.id}-", suffix=".log")
        try:
            try:
                proc = subprocess.Popen(job.argv, cwd=self.cwd, env=env, shell=False, stdin=subprocess.DEVNULL,
                                        stdout=fd, stderr=subprocess.STDOUT, start_new_session=True)
            finally:
                os.close(fd)  # the child has its own copy
            with self._cv:
                job.log_file = path
            # UTF-8 split across reads and \r\n, \r or \n line ends, as a text pipe would read them
            decoder, rest = io.IncrementalNewlineDecoder(codecs.getincrementaldecoder("utf-8")("replace"), True), ""
            with open(path, "rb") as fh:
                while True:
                    exited = proc.poll() is not None  # checked before reading, so the last read gets everything
                    rest = self._take_lines(job, rest + decoder.decode(fh.read()))
                    if exited:
                        break
                    try:
                        proc.wait(POLL)
                    except subprocess.TimeoutExpired:
                        pass
            rest += decoder.decode(b"", final=True)
            if rest:
                self._take_lines(job, rest + "\n")
            return proc.returncode
        finally:
            with self._cv:
                job.log_file = None
            try:
                os.unlink(path)
            except OSError:
                pass

    def _take_lines(self, job: Job, text: str) -> str:
        """Append the complete lines of ``text`` to the job log; the unfinished last line comes back."""
        *lines, rest = text.split("\n")
        if lines:
            with self._cv:
                job.log.extend(line.rstrip("\r") for line in lines)
        return rest


def table(jobs: list[dict]) -> dict:
    """The JOBS view: one row per job, newest first, in the terminal's table shape."""
    rows = []
    for j in jobs:
        row = {"id": j["id"], "target": j["target"], "status": j["status"], "started": j["started"],
               "finished": j["finished"], "last": j["log"][-1] if j["log"] else ""}
        t = ticker_of(j["target"])
        if t:
            row["_ticker"] = t
        if kind_of(j["target"]) == "decide":
            if dec_line(t):  # the decision it records; clicking never starts another job
                row["_cmds"] = {"target": dec_line(t)}
        rows.append(row)
    running = sum(j["status"] == "running" for j in jobs)
    queued = sum(j["status"] == "queued" for j in jobs)
    if not jobs:
        note = ("No background jobs. Start one with INGEST AAPL, INGEST MARKETS, INGEST MACRO, INGEST UNIVERSE, "
                "INGEST INSIDERS or INGEST WATCH, or a paper decision with AAPL DEC NEW.")
    else:
        note = (f"{running} running, {queued} queued (at most {MAX_RUNNING} ingest jobs and {MAX_DECIDE} decision "
                f"at once, each in its own slots). New facts appear when a job finishes.")
        newest = jobs[0]
        if newest["log"]:
            tail = "\n".join(newest["log"][-10:])
            note += f"\n\nLog of {newest['id']} ({newest['target']}), last lines:\n{tail}"
    return {
        "columns": [{"key": "id", "label": "Job", "type": "text"},
                    {"key": "target", "label": "Target", "type": "text"},
                    {"key": "status", "label": "Status", "type": "text"},
                    {"key": "started", "label": "Started (UTC)", "type": "date"},
                    {"key": "finished", "label": "Finished (UTC)", "type": "date"},
                    {"key": "last", "label": "Last log line", "type": "text"}],
        "rows": rows,
        "groups": [],
        "note": note,
    }
