"""Durable, tool-oriented mobile agent runtime.

This is intentionally independent from benchmark adapters.  ``Harness`` remains
the compatibility action loop; ``MobileAgentRuntime`` is the long-horizon loop
used by real agents.
"""
from __future__ import annotations

import json
import hashlib
import errno
import os
import time
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Protocol
import re

from .model import Observation
from .prompts import PromptAssembler
from .tools import ToolBroker, ToolCall, ToolResult
from .history import HistoryIndex
from .recovery import RecoveryPolicy
from .ports import DevicePort


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PlanStep:
    id: str
    description: str
    status: str = "pending"  # pending | in_progress | completed | blocked
    evidence: str = ""
    blocker: str = ""
    attempts: int = 0
    budget: int | None = None
    depends_on: tuple[str, ...] = ()
    retry_not_before: str = ""


@dataclass
class RuntimeEvent:
    sequence: int
    kind: str
    payload: dict[str, Any]
    at: str = field(default_factory=_now)


@dataclass
class SessionState:
    id: str
    task: str
    workspace_root: str
    plan: list[PlanStep] = field(default_factory=list)
    events: list[RuntimeEvent] = field(default_factory=list)
    summary: str = ""
    status: str = "running"  # running | awaiting_approval | verified | failed | blocked
    verifier_evidence: list[str] = field(default_factory=list)
    claimed_evidence: list[dict[str, Any]] = field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    observation: dict[str, Any] = field(default_factory=dict)
    observation_fingerprint: str = ""
    stagnant_turns: int = 0


class RuntimeModel(Protocol):
    """Provider-neutral model boundary; adapters return text plus any tool calls."""

    def respond(self, *, system: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> tuple[str, list[ToolCall]]: ...


class RuntimeVerifier(Protocol):
    def verify(self, task: str, observation: Observation, events: tuple[RuntimeEvent, ...]) -> tuple[bool, str]: ...


@dataclass(frozen=True)
class VerificationEvidence:
    passed: bool
    summary: str
    checks: tuple[dict[str, Any], ...] = ()
    source: str = "verifier"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UiObservationVerifier:
    """Composable, deterministic verifier for real Android observations.

    This complements authoritative backend/benchmark verifiers where a task's
    terminal state is visible on device. It never trusts the model's `claim_done`;
    every required predicate is evaluated from the current observation or a
    prior content-addressed evidence-capture result in the durable event trail.
    """
    def __init__(self, *, activity_pattern: str | None = None, required_text: tuple[str, ...] = (),
                 required_content_desc: tuple[str, ...] = (), required_resource_ids: tuple[str, ...] = (),
                 required_evidence_claims: tuple[str, ...] = (), source: str = "android_ui_recipe") -> None:
        self.activity_pattern = activity_pattern
        self.required_text, self.required_content_desc = tuple(required_text), tuple(required_content_desc)
        self.required_resource_ids, self.required_evidence_claims, self.source = tuple(required_resource_ids), tuple(required_evidence_claims), source
        if activity_pattern:
            re.compile(activity_pattern)

    def verify(self, task: str, observation: Observation, events: tuple[RuntimeEvent, ...]) -> VerificationEvidence:
        texts = {element.text.casefold() for element in observation.elements}
        descriptions = {element.content_desc.casefold() for element in observation.elements}
        resource_ids = {element.resource_id.casefold() for element in observation.elements}
        checks: list[dict[str, Any]] = []
        if self.activity_pattern:
            actual = observation.activity or ""
            checks.append({"kind": "activity_pattern", "expected": self.activity_pattern, "actual": actual,
                           "passed": bool(re.search(self.activity_pattern, actual))})
        for expected in self.required_text:
            checks.append({"kind": "ui_text", "expected": expected, "passed": expected.casefold() in texts})
        for expected in self.required_content_desc:
            checks.append({"kind": "content_desc", "expected": expected, "passed": expected.casefold() in descriptions})
        for expected in self.required_resource_ids:
            checks.append({"kind": "resource_id", "expected": expected, "passed": expected.casefold() in resource_ids})
        captured = set()
        for event in events:
            if event.kind != "tool_result":
                continue
            payload = event.payload
            if payload.get("call", {}).get("name") == "capture_evidence" and payload.get("result", {}).get("ok"):
                captured.add(str(payload.get("result", {}).get("content", {}).get("claim", "")))
        for expected in self.required_evidence_claims:
            checks.append({"kind": "captured_evidence_claim", "expected": expected, "passed": expected in captured})
        if not checks:
            return VerificationEvidence(False, "UI verifier recipe has no required predicates", (), self.source)
        passed = all(bool(check["passed"]) for check in checks)
        failed = [f"{check['kind']}={check['expected']}" for check in checks if not check["passed"]]
        summary = "Android UI verifier passed" if passed else "Android UI verifier missing: " + ", ".join(failed)
        return VerificationEvidence(passed, summary, tuple(checks), self.source)


class LegacyVerifierAdapter:
    """Adapts an existing benchmark verifier without transferring its authority.

    The benchmark still evaluates the live device; this adapter only translates
    its decision into the durable runtime's typed evidence record.
    """
    def __init__(self, verifier: Any, device: DevicePort, source: str = "benchmark") -> None:
        self.verifier, self.device, self.source = verifier, device, source

    def verify(self, task: str, observation: Observation, events: tuple[RuntimeEvent, ...]) -> VerificationEvidence:
        # Existing benchmark verifiers commonly ignore history. Provide a compact
        # compatible history shape while retaining the full runtime journal separately.
        passed, summary = self.verifier.verify(task, self.device, observation, ())
        return VerificationEvidence(bool(passed), str(summary), ({"kind": "legacy_benchmark_verifier", "event_count": len(events), "passed": bool(passed)},), self.source)


class SessionStore:
    """Snapshot plus append-only journal: interrupted writes never erase history."""
    snapshot_version = 2

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.history = HistoryIndex(self.root / "events")

    @contextmanager
    def lease(self, session_id: str, *, timeout_seconds: float = 5.0):
        """Take an exclusive cross-process lease for one mutable session.

        Journal hash chains cannot safely merge two independently resumed state
        snapshots. The atomic create is therefore a fail-closed session lease,
        deliberately separate from any operating-system execution sandbox.
        """
        if timeout_seconds < 0:
            raise ValueError("session lease timeout must be non-negative")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{session_id}.lease"
        token = uuid.uuid4().hex
        owner = {"token": token, "pid": os.getpid()}
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(owner, handle, sort_keys=True)
                break
            except FileExistsError:
                if self._lease_owner_is_dead(path):
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"session is already leased by another runtime worker: {session_id}")
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            yield
        finally:
            # Only release the exact lease we acquired; never delete another
            # worker's replacement lease after an exceptional cleanup race.
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("token") == token:
                    path.unlink()
            except (OSError, json.JSONDecodeError):
                pass

    @staticmethod
    def _lease_owner_is_dead(path: Path) -> bool:
        """Reap only a corrupt or definitely dead local-worker lease."""
        try:
            owner = json.loads(path.read_text(encoding="utf-8"))
            pid = int(owner["pid"])
            if pid <= 0:
                return True
            os.kill(pid, 0)
            return False
        except ProcessLookupError:
            return True
        except PermissionError:
            # A process we cannot inspect is not proof of death; preserve the
            # lease rather than risking two concurrent session writers.
            return False
        except OSError as exc:
            return exc.errno == errno.ESRCH
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return True

    def save(self, state: SessionState) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{state.id}.json"
        temporary = target.with_suffix(".json.tmp")
        payload = {"schema_version": self.snapshot_version, "state": asdict(state)}
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        temporary.write_bytes(encoded)
        temporary.replace(target)
        self._write_snapshot_digest(state.id, encoded)

    def load(self, session_id: str) -> SessionState:
        self.verify_evidence(session_id)
        snapshot = self.root / f"{session_id}.json"
        encoded = snapshot.read_bytes()
        self._verify_snapshot_digest(session_id, encoded)
        data = self._migrate_snapshot(json.loads(encoded))
        data["plan"] = [PlanStep(**item) for item in data.get("plan", [])]
        data["events"] = [RuntimeEvent(**item) for item in data.get("events", [])]
        state = SessionState(**data)
        # A crash after journal append but before snapshot replacement loses no events.
        journal = self.history.read(session_id)
        known = {event.sequence for event in state.events}
        for item in journal:
            if item.get("sequence") not in known:
                # Journal provenance/integrity fields are deliberately outside the
                # RuntimeEvent schema.  They are verified by HistoryIndex before
                # replay, then removed when reconstructing the in-memory event.
                replay_item = dict(item)
                for key in ("session_id", "previous_hash", "entry_hash"):
                    replay_item.pop(key, None)
                state.events.append(RuntimeEvent(**replay_item))
        state.events.sort(key=lambda event: event.sequence)
        return state

    def _migrate_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Upgrade known snapshot shapes before constructing runtime dataclasses.

        Version 0 was the original flat state object. Version 1 introduced an
        envelope but predated evidence claims and plan dependency edges. Keep
        these migrations explicit so a future incompatible snapshot never gets
        interpreted as a valid session.
        """
        if "schema_version" not in payload:
            version, data = 0, dict(payload)
        else:
            version, data = int(payload["schema_version"]), dict(payload.get("state", {}))
        if version > self.snapshot_version:
            raise RuntimeError(f"session snapshot schema is newer than this runtime: {version}")
        if version < 0:
            raise RuntimeError(f"invalid session snapshot schema: {version}")
        if version <= 1:
            data.setdefault("claimed_evidence", [])
            for step in data.get("plan", []):
                step.setdefault("depends_on", ())
        required = {"id", "task", "workspace_root"}
        if not required.issubset(data):
            raise RuntimeError("session snapshot is missing required state fields")
        return data

    def append(self, state: SessionState, event: RuntimeEvent) -> None:
        self.history.append(state.id, event)

    def capture_observation(self, session_id: str, observation: Observation) -> dict[str, str]:
        """Store replay evidence outside the JSON snapshot, deduplicated by content.

        The state only carries relative references; screenshots/UI dumps never get
        expanded into prompt context or the event journal.
        """
        directory = self.root / "evidence" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        evidence: dict[str, str] = {}
        if observation.screenshot_png:
            digest = hashlib.sha256(observation.screenshot_png).hexdigest()
            screenshot = directory / f"{digest}.png"
            if not screenshot.exists():
                screenshot.write_bytes(observation.screenshot_png)
            evidence["screenshot"] = str(screenshot.relative_to(self.root))
        if observation.ui_xml:
            digest = hashlib.sha256(observation.ui_xml.encode("utf-8")).hexdigest()
            xml = directory / f"{digest}.xml"
            if not xml.exists():
                xml.write_text(observation.ui_xml, encoding="utf-8")
            evidence["ui_xml"] = str(xml.relative_to(self.root))
        if evidence:
            self._update_evidence_manifest(session_id, evidence)
        return evidence

    def verify_evidence(self, session_id: str) -> None:
        """Reject replay when a referenced evidence artifact no longer matches its digest."""
        manifest = self.root / "evidence" / session_id / "manifest.json"
        if not manifest.is_file(): return
        try:
            rows = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("session evidence manifest is unreadable") from exc
        for relative, digest in rows.items():
            path = self.root / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"session evidence integrity check failed: {relative}")

    def _update_evidence_manifest(self, session_id: str, evidence: dict[str, str]) -> None:
        directory = self.root / "evidence" / session_id
        manifest = directory / "manifest.json"
        rows = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        for relative in evidence.values():
            path = self.root / relative
            rows[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(manifest)

    def _write_snapshot_digest(self, session_id: str, content: bytes) -> None:
        target = self.root / f"{session_id}.sha256"
        temporary = target.with_suffix(".sha256.tmp")
        temporary.write_text(hashlib.sha256(content).hexdigest(), encoding="ascii")
        temporary.replace(target)

    def _verify_snapshot_digest(self, session_id: str, content: bytes) -> None:
        digest = self.root / f"{session_id}.sha256"
        # Sessions written before integrity manifests stay readable, but every
        # newly written snapshot has a digest and fails closed on mismatch.
        if not digest.is_file(): return
        expected = digest.read_text(encoding="ascii").strip()
        if expected != hashlib.sha256(content).hexdigest():
            raise RuntimeError(f"session snapshot integrity check failed: {session_id}")

    def search(self, query: str, *, session_id: str | None = None, limit: int = 12) -> list[dict[str, Any]]:
        return self.history.search(query, session_id=session_id, limit=limit)

    def trace(self, session_id: str, sequence: int, *, before: int = 3, after: int = 3) -> list[dict[str, Any]]:
        return self.history.trace(session_id, sequence, before=before, after=after)


class MobileAgentRuntime:
    """Plan -> model/tool loop -> verify/recover -> durable review boundary."""

    def __init__(self, model: RuntimeModel, broker: ToolBroker, verifier: RuntimeVerifier, store: SessionStore,
                 prompts: PromptAssembler | None = None, max_turns: int = 40, recovery: RecoveryPolicy | None = None,
                 parallel_tool_timeout_seconds: float = 20.0) -> None:
        self.model, self.broker, self.verifier, self.store = model, broker, verifier, store
        self.prompts, self.max_turns = prompts or PromptAssembler(), max_turns
        self.recovery = recovery or RecoveryPolicy()
        if not 0.1 <= parallel_tool_timeout_seconds <= 120:
            raise ValueError("parallel tool timeout must be between 0.1 and 120 seconds")
        self.parallel_tool_timeout_seconds = parallel_tool_timeout_seconds
        if self.broker.session_search is None:
            self.broker.session_search = lambda query, limit: self.store.search(query, limit=limit)
        if self.broker.session_history is None:
            self.broker.session_history = self.store.history
        if self.broker.session_trace is None:
            self.broker.session_trace = lambda session_id, sequence, before, after: self.store.trace(session_id, sequence, before=before, after=after)

    def start(self, task: str, workspace_root: str, session_id: str | None = None) -> SessionState:
        state = SessionState(session_id or uuid.uuid4().hex, task, str(Path(workspace_root).resolve()))
        with self.store.lease(state.id):
            self._event(state, "session_started", {"prompt_version": self.prompts.version, "task": task, "workspace_root": state.workspace_root})
            self.store.save(state)
        return state

    def resume(self, session_id: str, *, approve: bool | None = None) -> SessionState:
        with self.store.lease(session_id):
            state = self.store.load(session_id)
            if state.status == "awaiting_approval":
                if approve is None:
                    return state
                pending = state.pending_approval or {}
                self.broker.authority.resolve_approval(pending.get("approval", pending), approve, state.id)
                self._event(state, "approval_resolved", {"approved": approve})
                state.pending_approval = None
                state.status = "running" if approve else "blocked"
                if approve and pending.get("call"):
                    call = ToolCall(**pending["call"])
                    observation = self.broker.device.observe()
                    result = self.broker.execute(call, state, observation)
                    self._event(state, "tool_result", {"call": asdict(call), "result": result.to_dict(), "approved_replay": True})
                    if not result.ok:
                        self._recover(state, result.error or "approved action failed")
            return self._run_locked(state)

    def run(self, state: SessionState) -> SessionState:
        with self.store.lease(state.id):
            return self._run_locked(state)

    def _run_locked(self, state: SessionState) -> SessionState:
        for _ in range(self.max_turns):
            if state.status != "running":
                break
            observation = self.broker.device.observe()
            state.observation = self.broker.observation_payload(observation)
            state.observation["evidence"] = self.store.capture_observation(state.id, observation)
            fingerprint = hashlib.sha256(json.dumps(state.observation, sort_keys=True).encode("utf-8")).hexdigest()
            state.stagnant_turns = state.stagnant_turns + 1 if fingerprint == state.observation_fingerprint else 0
            state.observation_fingerprint = fingerprint
            # The first observation establishes the baseline; two repeats means
            # three consecutive identical UI states.
            if state.stagnant_turns >= 2:
                self._recover(state, "stale observation: three consecutive identical UI states")
                state.stagnant_turns = 0
            system, context = self.prompts.render(state, self.broker)
            try:
                text, calls = self.model.respond(system=system, context=context, tools=self.broker.schemas())
            except Exception as exc:
                self._event(state, "model_failure", {"error": str(exc), "attempts": self._model_attempts()})
                self._recover(state, f"model provider failure: {exc}")
                continue
            attempts = self._model_attempts()
            if attempts:
                self._event(state, "model_attempts", {"attempts": attempts})
            self._event(state, "model_turn", {"text": text, "tool_count": len(calls)})
            if not calls:
                state.status = "blocked"
                self._event(state, "blocked", {"reason": "model returned no tool call"})
                break
            results = self._execute_calls(calls, state, observation)
            for call, result in results:
                self._event(state, "tool_result", {"call": asdict(call), "result": result.to_dict()})
                if result.approval_required:
                    state.status, state.pending_approval = "awaiting_approval", {"approval": result.approval, "call": asdict(call)}
                    break
                if not result.ok:
                    self._recover(state, result.error or f"{call.name} failed")
                    break
                if call.name == "claim_done" and result.ok:
                    verdict = self._verify(state, observation)
                    verification = verdict.to_dict()
                    verification["claimed_evidence"] = state.claimed_evidence[-8:]
                    self._event(state, "verification", verification)
                    if verdict.passed:
                        state.status = "verified"
                        state.verifier_evidence.append(verdict.summary)
                        if self.broker.memory:
                            promoted = self.broker.memory.curate(state.id, verifier_evidence=verdict.summary)
                            self._event(state, "memory_curated", {"count": len(promoted), "items": [item.to_dict() for item in promoted]})
                    else:
                        self._recover(state, verdict.summary)
                    break
            if self.prompts.compact_if_needed(state):
                self.prompts.invalidate_session(state)
            self.store.save(state)
        if state.status == "running":
            state.status = "failed"
            self._event(state, "failed", {"reason": f"turn limit reached ({self.max_turns})"})
        self.store.save(state)
        return state

    def _execute_calls(self, calls: list[ToolCall], state: SessionState, observation: Observation) -> list[tuple[ToolCall, ToolResult]]:
        """Parallelize only independent read-only operations; preserve action order."""
        readonly = {"observe", "web_search", "web_read", "search_files", "read_file", "memory_search", "skill_search", "skills_list", "skill_view", "session_search", "session_trace"}
        parallel = [call for call in calls if call.name in readonly]
        serial = [call for call in calls if call.name not in readonly]
        output: list[tuple[ToolCall, ToolResult]] = []
        if parallel:
            pool = ThreadPoolExecutor(max_workers=min(4, len(parallel)))
            futures = {pool.submit(self.broker.execute, call, state, observation): call for call in parallel}
            _, pending = wait(futures, timeout=self.parallel_tool_timeout_seconds)
            for future, call in futures.items():
                if future in pending:
                    future.cancel()
                    output.append((call, ToolResult(False, error=f"parallel read-only tool deadline exceeded ({self.parallel_tool_timeout_seconds:g}s)")))
                else:
                    try:
                        output.append((call, future.result()))
                    except Exception as exc:
                        output.append((call, ToolResult(False, error=f"parallel read-only tool failed: {exc}")))
            # Do not wait for an injected read-only provider that ignores its own
            # cancellation contract. Built-in network tools retain I/O timeouts.
            pool.shutdown(wait=False, cancel_futures=True)
        output.extend((call, self.broker.execute(call, state, observation)) for call in serial)
        return output

    def _verify(self, state: SessionState, observation: Observation) -> VerificationEvidence:
        raw = self.verifier.verify(state.task, observation, tuple(state.events))
        if isinstance(raw, VerificationEvidence):
            return raw
        passed, summary = raw
        return VerificationEvidence(bool(passed), str(summary), ({"kind": "independent_verifier", "passed": bool(passed)},))

    def _model_attempts(self) -> list[dict[str, Any]]:
        """Persist fallback diagnostics when a resilient model exposes them."""
        attempts = getattr(self.model, "attempts", ())
        return [asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item) for item in attempts]

    def _recover(self, state: SessionState, evidence: str) -> None:
        directive = self.recovery.classify(evidence=evidence, plan=state.plan)
        self._event(state, "replan", {"reason": evidence, "classification": directive.classification, "guidance": directive.guidance, "retry_allowed": directive.retry_allowed, "next_step_id": directive.next_step_id})
        active = next((step for step in state.plan if step.status == "in_progress"), None)
        if active:
            active.attempts += 1
            active.evidence, active.blocker = evidence, evidence
            if directive.classification == "transient":
                delay_seconds = min(60, 2 ** max(0, active.attempts - 1))
                active.retry_not_before = (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat()
                self._event(state, "retry_scheduled", {"step_id": active.id, "not_before": active.retry_not_before, "delay_seconds": delay_seconds})
            active.status = "blocked" if active.budget is not None and active.attempts >= active.budget else "pending"
        if directive.retry_allowed and directive.next_step_id:
            next_step = next((step for step in state.plan if step.id == directive.next_step_id), None)
            if next_step and next_step is not active and next_step.status in {"pending", "blocked"}:
                next_step.status = "in_progress"

    def _event(self, state: SessionState, kind: str, payload: dict[str, Any]) -> None:
        event = RuntimeEvent(max((item.sequence for item in state.events), default=0) + 1, kind, payload)
        state.events.append(event)
        self.store.append(state, event)
