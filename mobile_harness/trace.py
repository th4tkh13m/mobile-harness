from __future__ import annotations

import json
from pathlib import Path

from .core import Event


class JsonlTrace:
    """Append-only audit trace; observations/screenshots stay with the device owner."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: Event) -> None:
        decision = {"done": event.decision.done, "reason": event.decision.reason}
        if event.decision.action:
            decision["action"] = {key: value for key, value in event.decision.action.__dict__.items() if value is not None}
            decision["action"]["kind"] = event.decision.action.kind.value
        payload = {"step": event.step, "decision": decision, "policy": event.policy_message}
        if event.result:
            payload["result"] = {"ok": event.result.ok, "message": event.result.message, "metadata": event.result.metadata}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
