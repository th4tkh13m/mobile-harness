"""Bounded, durable non-GUI delegation for research/workspace-only subtasks."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Any, Callable


@dataclass(frozen=True)
class DelegatedTask:
    id: str
    goal: str
    tool_allowlist: tuple[str, ...] = ("web_search", "web_read", "search_files", "read_file", "session_search", "memory_search", "skill_search")
    timeout_seconds: float = 120.0
    parent_session_id: str = ""


@dataclass(frozen=True)
class DelegatedResult:
    id: str
    ok: bool
    summary: str
    evidence: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class DelegationRecord:
    task: DelegatedTask
    status: str
    updated_at: str
    result: DelegatedResult | None = None
    error: str | None = None


class JsonSubprocessWorker:
    """Trusted host-configured, non-GUI worker process using a narrow JSON protocol.

    The model never supplies this command. The child receives only the delegated
    task (including its read-only allowlist), not a device, broker, workspace
    writer, approval broker, or parent-session memory. A deadline kills the child
    process rather than relying on cooperative thread cancellation.
    """
    def __init__(self, command: list[str], *, cwd: str | Path | None = None, max_output_bytes: int = 256_000) -> None:
        if not command: raise ValueError("isolated worker command is empty")
        if not 1_024 <= max_output_bytes <= 4_000_000: raise ValueError("worker output limit must be between 1 KiB and 4 MiB")
        self.command, self.cwd, self.max_output_bytes = list(command), str(cwd) if cwd else None, max_output_bytes

    def __call__(self, task: DelegatedTask) -> DelegatedResult:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=stdout_file, stderr=stderr_file,
                                       cwd=self.cwd)
            try:
                process.communicate(json.dumps({"task": asdict(task)}).encode("utf-8") + b"\n", timeout=task.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                process.terminate()
                try: process.communicate(timeout=2)
                except subprocess.TimeoutExpired: process.kill(); process.communicate()
                raise TimeoutError(f"isolated delegated worker exceeded {task.timeout_seconds:g}s") from exc
            stdout, stderr = self._bounded_output(stdout_file), self._bounded_output(stderr_file)
        if process.returncode != 0:
            raise RuntimeError(f"isolated delegated worker exited {process.returncode}: {stderr.decode('utf-8', errors='replace')[:1000]}")
        try:
            payload = json.loads(stdout.decode("utf-8"))
            evidence = tuple(item for item in payload.get("evidence", ()) if isinstance(item, dict))
            if len(evidence) > 32 or len(str(payload["summary"])) > 16_000:
                raise ValueError("delegated result exceeds summary/evidence limits")
            return DelegatedResult(str(payload.get("id", task.id)), bool(payload["ok"]), str(payload["summary"]), evidence)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("isolated delegated worker returned invalid JSON result") from exc

    def _bounded_output(self, handle) -> bytes:
        handle.seek(0)
        data = handle.read(self.max_output_bytes + 1)
        if len(data) > self.max_output_bytes:
            raise RuntimeError(f"isolated delegated worker exceeded {self.max_output_bytes} byte output limit")
        return data


class DelegationManager:
    """No delegated task receives Android, write, command, approval, or recursive authority.

    A ``JsonSubprocessWorker`` gives tasks a hard-killable process boundary;
    ordinary injected Python workers remain cooperative for test/local use.
    """
    def __init__(self, worker: Callable[[DelegatedTask], DelegatedResult], max_workers: int = 3, store_root: str | Path | None = None) -> None:
        self.worker, self.max_workers = worker, max_workers
        self.store_root = Path(store_root) if store_root else None
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[str, Future[DelegatedResult]] = {}
        self._tasks: dict[str, DelegatedTask] = {}
        self._completed: dict[str, DelegatedResult] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, tasks: list[DelegatedTask]) -> list[str]:
        self._validate(tasks)
        with self._lock:
            duplicate = set(task.id for task in tasks) & set(self._futures)
            if duplicate: raise ValueError(f"delegated task already active: {sorted(duplicate)[0]}")
            for task in tasks:
                self._record(DelegationRecord(task, "queued", self._now()))
                self._tasks[task.id] = task
                self._futures[task.id] = self._pool.submit(self._run_one, task)
        return [task.id for task in tasks]

    def run(self, tasks: list[DelegatedTask]) -> list[DelegatedResult]:
        """Compatibility wrapper for callers that need completed results now."""
        task_ids = self.submit(tasks)
        return [self.result(task_id) for task_id in task_ids]

    def result(self, task_id: str) -> DelegatedResult:
        with self._lock:
            future = self._futures.get(task_id)
            completed = self._completed.get(task_id)
        if completed is not None:
            return completed
        if future is None:
            record = self.status(task_id)
            if not record or not record.result: raise KeyError(f"no completed delegation named {task_id}")
            return record.result
        task = self._tasks.get(task_id)
        try:
            return future.result(timeout=task.timeout_seconds if task else None)
        except FutureTimeout as exc:
            self._cancelled.add(task_id)
            if task:
                self._record(DelegationRecord(task, "timed_out", self._now(), error=f"deadline exceeded ({task.timeout_seconds:g}s)"))
            raise TimeoutError(f"delegated task {task_id} exceeded its deadline") from exc

    def status(self, task_id: str) -> DelegationRecord | None:
        live = self._latest().get(task_id)
        return live

    def records(self, *, parent_session_id: str | None = None) -> list[DelegationRecord]:
        """Return latest durable delegation records, optionally for one session."""
        values = self._latest().values()
        return sorted((record for record in values if parent_session_id is None or record.task.parent_session_id == parent_session_id),
                      key=lambda record: (record.updated_at, record.task.id))

    def cancel(self, task_id: str) -> bool:
        """Cancel queued work or request cancellation of a running read-only worker."""
        with self._lock:
            future = self._futures.get(task_id)
            if not future:
                return False
            self._cancelled.add(task_id)
            stopped = future.cancel()
        existing = self.status(task_id)
        if existing:
            self._record(DelegationRecord(existing.task, "cancelled", self._now(), error="cancelled before execution" if stopped else "cancellation requested"))
        return True

    def resume_pending(self) -> list[str]:
        """Retry journaled queued/running tasks after process recovery; never auto-runs GUI work."""
        pending = [record.task for record in self._latest().values() if record.status in {"queued", "running"}]
        return self.submit(pending) if pending else []

    def close(self, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)

    def _run_one(self, task: DelegatedTask) -> DelegatedResult:
        if task.id in self._cancelled:
            raise RuntimeError("delegation cancelled")
        self._record(DelegationRecord(task, "running", self._now()))
        try:
            result = self.worker(task)
            if task.id in self._cancelled:
                self._record(DelegationRecord(task, "cancelled", self._now(), error="cancellation requested"))
                raise RuntimeError("delegation cancelled")
            self._record(DelegationRecord(task, "completed", self._now(), result=result))
            with self._lock:
                self._completed[task.id] = result
            return result
        except Exception as exc:
            if task.id not in self._cancelled:
                self._record(DelegationRecord(task, "failed", self._now(), error=str(exc)))
            raise
        finally:
            with self._lock:
                self._futures.pop(task.id, None)

    @staticmethod
    def _validate(tasks: list[DelegatedTask]) -> None:
        forbidden = {"tap", "swipe", "drag", "type_text", "key", "back", "home", "launch_app", "write_file", "patch", "run_command", "delegate_non_gui"}
        for task in tasks:
            if forbidden & set(task.tool_allowlist):
                raise ValueError("delegated tasks may not receive GUI, write, command, or recursive delegation tools")
            if not 1 <= task.timeout_seconds <= 600:
                raise ValueError("delegated task timeout must be between 1 and 600 seconds")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _latest(self) -> dict[str, DelegationRecord]:
        if not self.store_root or not (self.store_root / "delegations.jsonl").exists(): return {}
        latest: dict[str, DelegationRecord] = {}
        for line in (self.store_root / "delegations.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line); row = raw["record"]
                task = DelegatedTask(**row["task"])
                result = DelegatedResult(**{**row["result"], "evidence": tuple(row["result"].get("evidence", ()))}) if row.get("result") else None
                latest[task.id] = DelegationRecord(task, row["status"], row["updated_at"], result, row.get("error"))
            except (KeyError, TypeError, json.JSONDecodeError):
                # A partial trailing write must not prevent recovery of prior
                # delegation decisions; the next record remains authoritative.
                continue
        return latest

    def _record(self, record: DelegationRecord) -> None:
        if not self.store_root: return
        self.store_root.mkdir(parents=True, exist_ok=True)
        with (self.store_root / "delegations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"record": asdict(record)}, ensure_ascii=False) + "\n")
