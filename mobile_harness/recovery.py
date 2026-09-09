"""Deterministic recovery policy for failed tools, stale UI, and failed verification."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class RecoveryDirective:
    classification: str
    next_step_id: str | None
    guidance: str
    retry_allowed: bool


class RecoveryPolicy:
    """Makes failures explicit model context rather than an undifferentiated retry."""
    def classify(self, *, evidence: str, plan: list[Any], observation_changed: bool = False) -> RecoveryDirective:
        lower = evidence.lower()
        if "approval" in lower or "permission" in lower:
            label, retry = "approval_required", False
        elif "locator" in lower or "element" in lower or "stale" in lower or observation_changed:
            label, retry = "stale_or_ambiguous_ui", True
        elif "timeout" in lower or "temporar" in lower or "network" in lower:
            label, retry = "transient", True
        elif "verif" in lower or "missing" in lower:
            label, retry = "verification_gap", True
        else:
            label, retry = "tool_or_strategy_failure", True
        active = next((step for step in plan if step.status == "in_progress"), None)
        completed = {step.id for step in plan if step.status == "completed"}
        # Never activate a downstream action until every named prerequisite has
        # recorded completion evidence. Unknown prerequisites stay ineligible.
        now = datetime.now(timezone.utc)
        def eligible(step: Any) -> bool:
            if step.status != "pending" or not set(getattr(step, "depends_on", ())).issubset(completed):
                return False
            retry_at = getattr(step, "retry_not_before", "")
            if not retry_at:
                return True
            try:
                return datetime.fromisoformat(retry_at) <= now
            except ValueError:
                return False
        pending = next((step for step in plan if eligible(step)), None)
        next_step = pending.id if pending else (active.id if active else None)
        return RecoveryDirective(label, next_step, f"Recovery classification: {label}. Evidence: {evidence[:600]}", retry)
