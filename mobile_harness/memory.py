"""Small, explicit experience store for memory benchmarks and real devices."""
from __future__ import annotations

import json
import re
import hashlib
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal
from datetime import datetime, timezone, timedelta

from .trust import scan_and_sanitize


# Small transparent concept map for the recurring Android surface.  It is not a
# replacement for embeddings: it makes bounded on-device retrieval more useful
# while keeping every score explainable and deterministic.
_CONCEPTS = (
    {"wifi", "wi-fi", "wlan", "wireless", "network"},
    {"settings", "setting", "preferences", "preference", "configuration", "config"},
    {"open", "launch", "start", "navigate"},
    {"tap", "click", "press", "select"},
    {"permission", "permissions", "allow", "deny"},
    {"notification", "notifications", "alert", "alerts"},
)


def _retrieval_terms(text: str) -> set[str]:
    lowered = text.lower()
    # Preserve both forms of common UI compounds: "wi-fi" should answer a
    # "wifi" query rather than degrading into unrelated "wi" and "fi" tokens.
    terms = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    terms.update(match.group(0).replace("-", "") for match in re.finditer(r"[a-z0-9_]{2,}(?:-[a-z0-9_]{2,})+", lowered))
    # Basic suffix normalization captures common action/task variants.
    terms.update(term[:-3] for term in tuple(terms) if term.endswith("ing") and len(term) > 5)
    terms.update(term[:-1] for term in tuple(terms) if term.endswith("s") and len(term) > 3)
    for concept in _CONCEPTS:
        if terms & concept:
            terms.update(concept)
    return terms


def _recall_score(query: str, text: str) -> int:
    wanted, candidate = _retrieval_terms(query), _retrieval_terms(text)
    lexical = len(wanted & candidate)
    phrase = 3 if query.lower().strip() and query.lower().strip() in text.lower() else 0
    return lexical + phrase


@dataclass(frozen=True)
class Experience:
    task: str
    summary: str
    evidence: str
    tags: tuple[str, ...] = ()


class VerifiedExperienceStore:
    """Append-only local memory. Only a passed independent verifier may promote it."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def recall(self, query: str, limit: int = 5) -> tuple[Experience, ...]:
        scored: list[tuple[int, Experience]] = []
        for experience in self._read():
            score = _recall_score(query, experience.task + " " + experience.summary + " " + " ".join(experience.tags))
            if score:
                scored.append((score, experience))
        return tuple(item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit])

    def promote(self, experience: Experience, verifier_passed: bool) -> None:
        if not verifier_passed:
            raise PermissionError("experience promotion requires independent verifier success")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(experience), ensure_ascii=False) + "\n")

    def _read(self) -> tuple[Experience, ...]:
        if not self.path.is_file():
            return ()
        results: list[Experience] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                results.append(Experience(item["task"], item["summary"], item["evidence"], tuple(item.get("tags", ()))))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return tuple(results)


@dataclass(frozen=True)
class CuratedMemory:
    summary: str
    kind: Literal["success", "failure_avoidance", "preference"]
    evidence: str
    session_id: str = ""
    created_at: str = ""
    provenance: str = "runtime"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class CuratedMemoryStore:
    """Staging is separate from promotion; only reviewed evidence becomes recallable."""
    _local_locks: dict[str, threading.RLock] = {}
    _local_locks_guard = threading.Lock()

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.promoted, self.staged = self.root / "memory.jsonl", self.root / "candidates.jsonl"
        self.promotion_ledger = self.root / "promotion-ledger.jsonl"

    def stage(self, summary: str, kind: str, session_id: str) -> None:
        clean = self._sanitize(summary)
        if not clean:
            raise ValueError("memory candidate is empty or unsafe")
        resolved = kind if kind in {"success", "failure_avoidance", "preference"} else "success"
        with self._promotion_lock():
            self._append(self.staged, CuratedMemory(clean, resolved, "", session_id, datetime.now(timezone.utc).isoformat(), "model_candidate"))

    def promote(self, memory: CuratedMemory, *, verified: bool, reviewed: bool) -> None:
        if memory.kind == "success" and not verified: raise PermissionError("success memory requires verifier evidence")
        if memory.kind == "failure_avoidance" and not reviewed: raise PermissionError("failure avoidance requires curator review")
        with self._promotion_lock():
            self._append(self.promoted, memory)

    def recall(self, query: str, limit: int = 5) -> tuple[CuratedMemory, ...]:
        scored = []
        for item in self._read(self.promoted):
            score = _recall_score(query, item.summary + " " + item.evidence)
            if score:
                # Prefer verifier-backed successful procedures at equal relevance;
                # avoid mixing failure warnings into success recipes by default.
                quality = 2 if item.kind == "success" and item.provenance == "verified_curator" else 1 if item.kind == "preference" else 0
                scored.append((score, quality, item.created_at, item))
        return tuple(item for _, _, _, item in sorted(scored, reverse=True, key=lambda pair: pair[:3])[:limit])

    def snapshot(self, limit: int = 12, char_limit: int = 4_000) -> tuple[CuratedMemory, ...]:
        """Bounded prompt-safe memory injection, newest entries first."""
        used, selected = 0, []
        for item in reversed(self._read(self.promoted)):
            size = len(item.summary) + len(item.evidence)
            if selected and (len(selected) >= limit or used + size > char_limit):
                continue
            selected.append(item); used += size
        return tuple(reversed(selected))

    def _append(self, path: Path, item: CuratedMemory) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    def candidates(self, session_id: str | None = None) -> tuple[CuratedMemory, ...]:
        values = self._read(self.staged)
        return tuple(item for item in values if session_id is None or item.session_id == session_id)

    def curate(self, session_id: str, *, verifier_evidence: str = "", reviewer: Callable[[CuratedMemory], bool] | None = None) -> tuple[CuratedMemory, ...]:
        # The ledger check and its matching promotion must be one transaction:
        # separate runtime workers may complete at the same time after a shared
        # verifier success. The file lock covers cooperating processes, while
        # the keyed RLock also protects threads on platforms with process-wide
        # advisory locks.
        with self._promotion_lock():
            promoted = []
            for candidate in self.candidates(session_id):
                candidate_id = self._candidate_id(candidate)
                if candidate_id in self._promoted_candidate_ids():
                    continue
                reviewed = bool(reviewer(candidate)) if reviewer else candidate.kind == "success" and bool(verifier_evidence)
                verified = candidate.kind != "success" or bool(verifier_evidence)
                if (candidate.kind == "success" and verified) or (candidate.kind == "failure_avoidance" and reviewed):
                    item = CuratedMemory(candidate.summary, candidate.kind, verifier_evidence or "curator-reviewed", session_id, datetime.now(timezone.utc).isoformat(), "verified_curator" if verifier_evidence else "reviewed_curator")
                    # We already hold the cross-process promotion transaction
                    # lock. Calling the public promote() would re-open and lock
                    # the same file descriptor path, which can deadlock on
                    # POSIX advisory locks.
                    self._append(self.promoted, item)
                    self._append_ledger(candidate_id)
                    promoted.append(item)
            return tuple(promoted)

    @contextmanager
    def _promotion_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        key = str(self.root.resolve())
        with self._local_locks_guard:
            lock = self._local_locks.setdefault(key, threading.RLock())
        with lock:
            path = self.root / ".promotion.lock"
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

    def prune(self, retention_days: int = 90, keep: int = 500) -> int:
        """Bound durable recall while retaining newest evidence-backed memories."""
        with self._promotion_lock():
            now = datetime.now(timezone.utc); retained = []
            for item in self._read(self.promoted):
                try: age = now - datetime.fromisoformat(item.created_at)
                except ValueError: age = timedelta(0)
                if age.days <= retention_days: retained.append(item)
            retained = retained[-keep:]; self.promoted.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.promoted.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in retained), encoding="utf-8")
            temporary.replace(self.promoted)
            return len(retained)

    @staticmethod
    def _candidate_id(candidate: CuratedMemory) -> str:
        raw = json.dumps({"summary": candidate.summary, "kind": candidate.kind, "session_id": candidate.session_id, "created_at": candidate.created_at}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _promoted_candidate_ids(self) -> set[str]:
        if not self.promotion_ledger.is_file(): return set()
        return {line.strip() for line in self.promotion_ledger.read_text(encoding="utf-8").splitlines() if re.fullmatch(r"[0-9a-f]{64}", line.strip())}

    def _append_ledger(self, candidate_id: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.promotion_ledger.open("a", encoding="utf-8") as handle:
            handle.write(candidate_id + "\n")

    @staticmethod
    def _sanitize(text: str) -> str:
        return scan_and_sanitize(text, max_chars=2000)[0].strip()

    @staticmethod
    def _read(path: Path) -> tuple[CuratedMemory, ...]:
        if not path.is_file(): return ()
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try: values.append(CuratedMemory(**json.loads(line)))
            except (TypeError, json.JSONDecodeError): pass
        return tuple(values)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    provenance: str = "user_or_curator"

    def to_dict(self) -> dict[str, str]: return asdict(self)


@dataclass(frozen=True)
class SkillCandidate:
    skill: Skill
    session_id: str
    created_at: str


class SkillStore:
    """Versioned skills: model proposals stage first; only reviewed skills become reusable."""
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.candidate_path = self.root / "candidates.jsonl"

    def stage(self, skill: Skill, session_id: str) -> None:
        clean = self._clean(skill)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.candidate_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(SkillCandidate(clean, session_id, datetime.now(timezone.utc).isoformat())), ensure_ascii=False) + "\n")

    def candidates(self, session_id: str | None = None) -> tuple[SkillCandidate, ...]:
        if not self.candidate_path.is_file(): return ()
        found = []
        for line in self.candidate_path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line); item = SkillCandidate(Skill(**raw["skill"]), raw["session_id"], raw["created_at"])
                if session_id is None or item.session_id == session_id: found.append(item)
            except (KeyError, TypeError, json.JSONDecodeError): pass
        return tuple(found)

    def promote(self, candidate: SkillCandidate, *, reviewed: bool, verifier_evidence: str = "") -> Path:
        if not reviewed:
            raise PermissionError("skill promotion requires curator review")
        return self.save(Skill(candidate.skill.name, candidate.skill.description, candidate.skill.body, "reviewed_curator"), evidence=verifier_evidence)

    def curate(self, session_id: str, reviewer: Callable[[SkillCandidate], bool] | None = None, verifier_evidence: str = "") -> tuple[Path, ...]:
        # Verifier evidence is necessary but never sufficient for a reusable procedure.
        return tuple(self.promote(item, reviewed=bool(reviewer(item)) if reviewer else False, verifier_evidence=verifier_evidence)
                     for item in self.candidates(session_id) if reviewer and reviewer(item))

    def save(self, skill: Skill, *, evidence: str = "") -> Path:
        skill = self._clean(skill)
        if skill.provenance not in {"user", "reviewed_curator", "verified_curator", "user_or_curator"}:
            raise PermissionError("only user or curator skills may be saved")
        # Reuse the process-safe per-store lock used by curated promotion: it
        # serializes version allocation across independent reviewer workers.
        with CuratedMemoryStore(self.root)._promotion_lock():
            versions = self.root / ".versions" / skill.name
            versions.mkdir(parents=True, exist_ok=True)
            version = len(list(versions.glob("*.md"))) + 1
            rendered = f"---\nprovenance: {skill.provenance}\nversion: {version}\nevidence: {CuratedMemoryStore._sanitize(evidence)}\n---\n# {skill.name}\n\n{skill.description}\n\n{skill.body}\n"
            version_path = versions / f"v{version}.md"
            self._atomic_write(version_path, rendered)
            path = self.root / f"{skill.name}.md"
            self._atomic_write(path, rendered)
            return path

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".mobile-runtime.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _clean(skill: Skill) -> Skill:
        if not re.fullmatch(r"[a-z0-9_-]+", skill.name): raise ValueError("skill name must be lowercase alphanumeric/dash/underscore")
        body = CuratedMemoryStore._sanitize(skill.body)
        if not body: raise ValueError("skill body is empty or unsafe")
        return Skill(skill.name, CuratedMemoryStore._sanitize(skill.description), body, skill.provenance)
    def recall(self, query: str) -> tuple[Skill, ...]:
        found: list[tuple[int, Skill]] = []
        for path in self.root.glob("*.md") if self.root.is_dir() else ():
            text = path.read_text(encoding="utf-8", errors="replace")
            score = _recall_score(query, path.stem + " " + text)
            if score:
                body = text
                provenance = "unknown"
                if text.startswith("---\n"):
                    _, _, remainder = text.partition("\n---\n")
                    header, body = text[4:].partition("\n---\n")[0], remainder
                    provenance = next((line.split(":", 1)[1].strip() for line in header.splitlines() if line.startswith("provenance:")), provenance)
                lines = [line for line in body.splitlines() if line.strip()]
                description = lines[1] if len(lines) > 1 and lines[0].startswith("# ") else (lines[0] if lines else path.stem)
                found.append((score, Skill(path.stem, description, body[:12_000], provenance)))
        return tuple(item for _, item in sorted(found, key=lambda row: (row[0], row[1].name), reverse=True)[:5])

    def list(self) -> tuple[Skill, ...]:
        """Return compact skill-index entries for the native skills_list tool."""
        return tuple(self._load_path(path) for path in sorted(self.root.glob("*.md"))) if self.root.is_dir() else ()

    def view(self, name: str) -> Skill:
        """Return one reviewed skill's complete current body."""
        if not re.fullmatch(r"[a-z0-9_-]+", name):
            raise ValueError("invalid skill name")
        path = self.root / f"{name}.md"
        if not path.is_file():
            raise KeyError(f"skill not found: {name}")
        return self._load_path(path)

    @staticmethod
    def _load_path(path: Path) -> Skill:
        text = path.read_text(encoding="utf-8", errors="replace")
        body, provenance = text, "unknown"
        if text.startswith("---\n"):
            header, _, body = text[4:].partition("\n---\n")
            provenance = next((line.split(":", 1)[1].strip() for line in header.splitlines() if line.startswith("provenance:")), provenance)
        lines = [line for line in body.splitlines() if line.strip()]
        description = lines[1] if len(lines) > 1 and lines[0].startswith("# ") else (lines[0] if lines else path.stem)
        return Skill(path.stem, description, body[:12_000], provenance)
