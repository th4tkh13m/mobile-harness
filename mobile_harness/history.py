"""Append-only durable event history with indexed cross-session search."""
from __future__ import annotations

import json
import re
import hashlib
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{2,}", text.lower()))


class HistoryIndex:
    """JSONL is the source of truth; the index is rebuildable and never authoritative."""
    _disk_locks: dict[str, threading.RLock] = {}
    _disk_locks_guard = threading.Lock()
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "search-index.json"
        self._lock = threading.RLock()

    def append(self, session_id: str, event: Any) -> None:
        with self._lock:
            with self._mutation_lock():
                self.root.mkdir(parents=True, exist_ok=True)
                payload = asdict(event) if hasattr(event, "__dataclass_fields__") else dict(event)
                payload["session_id"] = session_id
                previous_hash = self._last_hash(session_id)
                payload["previous_hash"] = previous_hash
                payload["entry_hash"] = self._hash(payload)
                with (self.root / f"{session_id}.events.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                index = self._load_index()
                key = f"{session_id}:{payload.get('sequence', 0)}"
                index["events"][key] = {"session_id": session_id, "sequence": payload.get("sequence", 0), "kind": payload.get("kind", ""), "at": payload.get("at", ""), "terms": sorted(_terms(json.dumps(payload, ensure_ascii=False)))}
                for term in index["events"][key]["terms"]:
                    if key not in index["terms"].setdefault(term, []):
                        index["terms"][term].append(key)
                self._save_index(index)

    def read(self, session_id: str) -> list[dict[str, Any]]:
        path = self.root / f"{session_id}.events.jsonl"
        if not path.is_file(): return []
        out, previous_hash, chained = [], "", False
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
                has_chain = "entry_hash" in payload or "previous_hash" in payload
                if has_chain:
                    if not isinstance(payload.get("entry_hash"), str) or payload.get("previous_hash") != previous_hash:
                        raise RuntimeError(f"event journal integrity check failed: {session_id}")
                    expected = self._hash({key: value for key, value in payload.items() if key != "entry_hash"})
                    if payload["entry_hash"] != expected:
                        raise RuntimeError(f"event journal integrity check failed: {session_id}")
                    previous_hash, chained = payload["entry_hash"], True
                elif chained:
                    raise RuntimeError(f"event journal integrity check failed: unchained entry after chain in {session_id}")
                out.append(payload)
            except json.JSONDecodeError: continue
        return out

    def search(self, query: str, *, limit: int = 12, session_id: str | None = None) -> list[dict[str, Any]]:
        wanted = _terms(query)
        if not wanted: return []
        index = self._load_index()
        candidates: set[str] = set()
        for term in wanted: candidates.update(index["terms"].get(term, ()))
        if not candidates and any(self.root.glob("*.events.jsonl")):
            self.rebuild(); index = self._load_index()
            for term in wanted: candidates.update(index["terms"].get(term, ()))
        scored = []
        for key in candidates:
            metadata = index["events"].get(key, {})
            if session_id and metadata.get("session_id") != session_id: continue
            event = next((item for item in self.read(str(metadata.get("session_id", ""))) if item.get("sequence") == metadata.get("sequence")), None)
            if not event: continue
            score = len(wanted & set(metadata.get("terms", ())))
            if event.get("kind") in {"verification", "replan", "failed", "blocked", "tool_result"}: score += 2
            scored.append((score, event))
        return [event for _, event in sorted(scored, key=lambda item: (item[0], item[1].get("at", "")), reverse=True)[:limit]]

    def trace(self, session_id: str, sequence: int, *, before: int = 3, after: int = 3) -> list[dict[str, Any]]:
        """Return a bounded, replay-validated causal window around one event."""
        if not 0 <= before <= 24 or not 0 <= after <= 24:
            raise ValueError("trace window must be between 0 and 24 events on each side")
        events = self.read(session_id)  # validates the journal chain before slicing
        position = next((index for index, event in enumerate(events) if event.get("sequence") == sequence), None)
        if position is None:
            raise KeyError(f"event sequence not found in session: {sequence}")
        return events[max(0, position - before): position + after + 1]

    def browse(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Compact recent-session listing for the session_search browse shape."""
        rows = []
        for path in self.root.glob("*.events.jsonl"):
            session_id = path.name.removesuffix(".events.jsonl")
            events = self.read(session_id)
            if events:
                rows.append({"session_id": session_id, "when": events[-1].get("at", ""), "preview": str(events[0].get("payload", {}))[:300]})
        return sorted(rows, key=lambda row: row["when"], reverse=True)[:limit]

    def rebuild(self) -> None:
        """Recover the derived index from append-only journals after crashes or upgrades."""
        with self._mutation_lock():
            index: dict[str, Any] = {"version": 1, "terms": {}, "events": {}}
            for path in self.root.glob("*.events.jsonl"):
                session_id = path.name.removesuffix(".events.jsonl")
                for payload in self.read(session_id):
                    key = f"{session_id}:{payload.get('sequence', 0)}"
                    terms = sorted(_terms(json.dumps(payload, ensure_ascii=False)))
                    index["events"][key] = {"session_id": session_id, "sequence": payload.get("sequence", 0), "kind": payload.get("kind", ""), "at": payload.get("at", ""), "terms": terms}
                    for term in terms: index["terms"].setdefault(term, []).append(key)
            self._save_index(index)

    @contextmanager
    def _mutation_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        key = str(self.root.resolve())
        with self._disk_locks_guard:
            lock = self._disk_locks.setdefault(key, threading.RLock())
        with lock:
            path = self.root / ".search-index.lock"
            with path.open("a+b") as handle:
                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0"); handle.flush()
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    try:
                        yield
                    finally:
                        handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.is_file(): return {"version": 1, "terms": {}, "events": {}}
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
            return value if isinstance(value.get("terms"), dict) and isinstance(value.get("events"), dict) else {"version": 1, "terms": {}, "events": {}}
        except json.JSONDecodeError:
            return {"version": 1, "terms": {}, "events": {}}

    def _save_index(self, index: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_name(f"{self.index_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(index, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(self.index_path)

    def _last_hash(self, session_id: str) -> str:
        path = self.root / f"{session_id}.events.jsonl"
        if not path.is_file(): return ""
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines: return ""
        try:
            payload = json.loads(lines[-1])
            return str(payload.get("entry_hash", ""))
        except json.JSONDecodeError:
            return ""

    @staticmethod
    def _hash(payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
