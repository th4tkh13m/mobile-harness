from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .model import Action, ActionKind, Observation

SENSITIVE_TERMS = ("password", "permission", "allow", "payment", "checkout", "purchase", "send", "submit")


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str = ""
    needs_approval: bool = False


class DevicePolicy:
    """Fail closed on sensitive UI; embedding apps may explicitly approve an action."""

    def __init__(self, approve: Callable[[Action, Observation, str], bool] | None = None) -> None:
        self._approve = approve

    def check(self, action: Action, observation: Observation) -> PolicyDecision:
        visible = " ".join(item for element in observation.elements for item in (element.text, element.content_desc)).lower()
        if action.kind in {ActionKind.TAP, ActionKind.TYPE_TEXT, ActionKind.KEY}:
            term = next((term for term in SENSITIVE_TERMS if term in visible), None)
            if term:
                reason = f"sensitive UI detected: {term}"
                if self._approve and self._approve(action, observation, reason):
                    return PolicyDecision(True, reason, needs_approval=True)
                return PolicyDecision(False, reason, needs_approval=True)
        return PolicyDecision(True)


class AuthorityBroker:
    """Cross-tool consequence policy; approval is resolved explicitly by runtime clients."""

    def __init__(self, approve: Callable[[str, str], bool] | None = None) -> None:
        self.approve = approve
        self._grants: set[tuple[str, str, str]] = set()

    def check(self, category: str, detail: str, session_id: str = "") -> PolicyDecision:
        if (session_id, category, detail) in self._grants:
            return PolicyDecision(True, "approved for session", needs_approval=True)
        requires = category in {"mobile_consequential", "workspace_write", "command", "command_high_risk", "mcp", "curator_review"}
        if not requires:
            return PolicyDecision(True)
        reason = f"approval required for {category}: {detail[:160]}"
        if self.approve and self.approve(category, detail):
            return PolicyDecision(True, reason, needs_approval=True)
        return PolicyDecision(False, reason, needs_approval=True)

    def resolve_approval(self, approval: dict[str, str], accepted: bool, session_id: str = "") -> None:
        # Decision is recorded in the durable session; calls are intentionally not replayed automatically.
        if accepted:
            self._grants.add((session_id, str(approval.get("category", "")), str(approval.get("detail", ""))))
