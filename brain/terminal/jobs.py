"""Background collection jobs for the terminal.

Ingest runs in a child process, never inside the server: a slow SEC or Yahoo
fetch must not hold the bank lock, and a crash in a source must not take the
terminal down. The child writes to the journal the same way the CLI does; when
it exits the server swaps in a fresh Bank so the new facts appear.

What reaches the child is fixed here. A target is a ticker matching a strict
pattern or one of three keywords, and it becomes one element of an argument
list (never a shell string), so nothing typed into the browser can reach a
command interpreter or smuggle in an option.
"""
from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from brain import config
from brain.timeutil import now_iso

TICKER = re.compile(r"[A-Za-z0-9.^=\-]{1,15}")  # matched with fullmatch: no trailing-newline loophole
KEYWORDS = {"MARKETS": ["markets"], "MACRO": ["macro"], "UNIVERSE": ["ingest-universe"]}
MAX_RUNNING = 2
LOG_LINES = 200
MAX_QUEUED = 50
KEEP_JOBS = 100  # finished jobs beyond this are forgotten, oldest first


class JobError(ValueError):
    def __init__(self, message: str, hint: str = "INGEST AAPL"):
        super().__init__(message)
        self.message, self.hint = message, hint


def validate(target) -> str:
    """The canonical (upper-case) target, or JobError. Keywords are case-insensitive."""
    if not isinstance(target, str) or not target:
        raise JobError("ingest needs a target: a ticker, MARKETS, MACRO or UNIVERSE")
    if target.upper() in KEYWORDS:
        return target.upper()
    if not TICKER.fullmatch(target):
        raise JobError(f"not a valid ingest target: {target[:40]!r}")
    if target.startswith("-"):
        raise JobError(f"not a valid ingest target: {target!r} (a ticker cannot start with '-')")
    return target.upper()


def argv_for(target: str) -> list[str]:
    """The child's argument list for a validated target."""
    rest = KEYWORDS.get(target, ["ingest", target])
    return [sys.executable, "-m", "brain", *rest]


def is_ticker(target: str) -> bool:
    return target not in KEYWORDS


@dataclass
class Job:
    id: str
    target: str
    argv: list[str]
    status: str = "queued"  # queued | running | done | failed
    started: str | None = None
    finished: str | None = None
    returncode: int | None = None
    log: deque = field(default_factory=lambda: deque(maxlen=LOG_LINES))

    def to_dict(self) -> dict:
        return {"id": self.id, "target": self.target, "status": self.status, "started": self.started,
                "finished": self.finished, "returncode": self.returncode, "log": list(self.log)}


class JobManager:
    """At most ``max_running`` children at once; the rest wait in a FIFO queue.

    ``argv_for`` maps a validated target to an argument list; tests inject a
    harmless one. ``on_finish(job_dict)`` runs after every job, done or failed
    (a failed ingest may still have written facts), outside this manager's lock
    so it may take the server's bank lock without risking a deadlock.
    """

    def __init__(self, on_finish: Callable[[dict], None] | None = None, max_running: int = MAX_RUNNING,
                 argv_for: Callable[[str], list[str]] = argv_for, cwd=config.ROOT, env: dict | None = None):
        self.on_finish = on_finish
        self.max_running = max_running
        self.argv_for = argv_for
        self.cwd = str(cwd)
        self.env = env
        self._jobs: dict[str, Job] = {}
        self._queue: deque[Job] = deque()
        self._workers = 0  # threads alive, including their on_finish call
        self._ids = itertools.count(1)
        self._cv = threading.Condition()
        self._closed = False

    # ---------- public ----------
    def submit(self, target) -> dict:
        """Queue a job for ``target``; a target already queued or running returns that job."""
        t = validate(target)
        with self._cv:
            if self._closed:
                raise JobError("the server is shutting down", hint="")
            for j in self._jobs.values():
                if j.target == t and j.status in ("queued", "running"):
                    return j.to_dict()
            if len(self._queue) >= MAX_QUEUED:
                raise JobError(f"{MAX_QUEUED} jobs are already queued; wait for some to finish", hint="JOBS")
            job = Job(id=f"j{next(self._ids)}", target=t, argv=list(self.argv_for(t)))
            self._jobs[job.id] = job
            self._queue.append(job)
            self._forget_old()
            self._pump()
            return job.to_dict()

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
            return len(self._queue)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until nothing is queued or running (on_finish included). False on timeout."""
        with self._cv:
            return self._cv.wait_for(lambda: not self._queue and self._workers == 0, timeout)

    def shutdown(self) -> None:
        """Cancel queued jobs. Running children are left to finish: killing one mid-write could
        leave a torn journal line, and Ctrl+C in the terminal already reaches them."""
        with self._cv:
            self._closed = True
            while self._queue:
                j = self._queue.popleft()
                j.status, j.finished = "failed", now_iso()
                j.log.append("cancelled: the server stopped before this job started")
            self._cv.notify_all()

    # ---------- internals (call with self._cv held) ----------
    def _pump(self) -> None:
        while self._queue and self._workers < self.max_running and not self._closed:
            job = self._queue.popleft()
            job.status, job.started = "running", now_iso()
            self._workers += 1
            threading.Thread(target=self._run, args=(job,), name=f"job-{job.id}", daemon=True).start()

    def _forget_old(self) -> None:
        done = [k for k, j in self._jobs.items() if j.status in ("done", "failed")]
        for k in done[: max(0, len(self._jobs) - KEEP_JOBS)]:
            del self._jobs[k]

    # ---------- worker ----------
    def _run(self, job: Job) -> None:
        rc = None
        try:
            env = dict(self.env if self.env is not None else os.environ, PYTHONUNBUFFERED="1")
            with subprocess.Popen(job.argv, cwd=self.cwd, env=env, shell=False, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace", bufsize=1) as proc:
                for line in proc.stdout:
                    with self._cv:
                        job.log.append(line.rstrip("\r\n"))
                rc = proc.wait()
        except Exception as e:  # noqa: BLE001 - a job that cannot start is a failed job, not a server crash
            with self._cv:
                job.log.append(f"could not run job: {type(e).__name__}: {e}")
        with self._cv:
            job.returncode = rc
            job.status = "done" if rc == 0 else "failed"
            job.finished = now_iso()
            snapshot = job.to_dict()
        try:
            if self.on_finish:
                self.on_finish(snapshot)
        except Exception as e:  # noqa: BLE001
            print(f"job {job.id}: on_finish failed: {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            with self._cv:
                self._workers -= 1
                self._pump()
                self._cv.notify_all()


def table(jobs: list[dict]) -> dict:
    """The JOBS view: one row per job, newest first, in the terminal's table shape."""
    rows = []
    for j in jobs:
        row = {"id": j["id"], "target": j["target"], "status": j["status"], "started": j["started"],
               "finished": j["finished"], "last": j["log"][-1] if j["log"] else ""}
        if is_ticker(j["target"]):
            row["_ticker"] = j["target"]
        rows.append(row)
    running = sum(j["status"] == "running" for j in jobs)
    queued = sum(j["status"] == "queued" for j in jobs)
    if not jobs:
        note = "No background jobs. Start one with INGEST AAPL, INGEST MARKETS, INGEST MACRO or INGEST UNIVERSE."
    else:
        note = (f"{running} running, {queued} queued (at most {MAX_RUNNING} run at once). "
                f"New facts appear when a job finishes.")
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
