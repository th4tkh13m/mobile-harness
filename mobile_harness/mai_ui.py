"""Compatibility boundary for MAI-UI / MobileWorld-style normalized actions."""
from __future__ import annotations

from typing import Any

from .model import Action, ActionKind, Decision


def decision_from_mai_action(payload: dict[str, Any]) -> Decision:
    """Translate MAI-UI's documented action JSON without importing MAI-UI.

    MAI-UI emits coordinates normalized to [0, 1] after parsing its native
    0..999 format. Its `terminate` is deliberately only a claim; this harness
    invokes the independent verifier before accepting it.
    """
    action = payload.get("action")
    if action == "terminate":
        return Decision(done=True, reason=str(payload.get("status", "")))
    if action == "click":
        return Decision(action=Action(ActionKind.TAP, x=_point(payload, "coordinate")[0], y=_point(payload, "coordinate")[1]))
    if action == "type":
        return Decision(action=Action(ActionKind.TYPE_TEXT, text=_require_string(payload, "text")))
    if action == "swipe":
        direction = _require_string(payload, "direction").lower()
        x, y = _point(payload, "coordinate", default=(.5, .5))
        endpoints = {
            "up": (x, .8, x, .2), "down": (x, .2, x, .8),
            "left": (.8, y, .2, y), "right": (.2, y, .8, y),
        }
        if direction not in endpoints:
            raise ValueError(f"unsupported MAI-UI swipe direction: {direction}")
        return Decision(action=Action(ActionKind.SWIPE, *endpoints[direction]))
    if action == "drag":
        x, y = _point(payload, "start_coordinate")
        x2, y2 = _point(payload, "end_coordinate")
        return Decision(action=Action(ActionKind.SWIPE, x, y, x2, y2, duration_ms=700))
    if action == "open":
        # Real-device integrations need a package resolver. Do not guess package names.
        raise ValueError("MAI-UI open requires an app-name-to-package resolver supplied by the benchmark/device adapter")
    if action == "system_button":
        button = _require_string(payload, "button").lower()
        if button == "back":
            return Decision(action=Action(ActionKind.BACK))
        if button == "home":
            return Decision(action=Action(ActionKind.HOME))
        if button == "enter":
            return Decision(action=Action(ActionKind.KEY, key="ENTER"))
        raise ValueError(f"unsupported MAI-UI system button: {button}")
    if action == "wait":
        return Decision(action=Action(ActionKind.WAIT))
    raise ValueError(f"unsupported MAI-UI action: {action!r}")


def _point(payload: dict[str, Any], key: str, default: tuple[float, float] | None = None) -> tuple[float, float]:
    value = payload.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"MAI-UI action requires {key}=[x,y]")
    x, y = (float(item) for item in value)
    if not 0 <= x <= 1 or not 0 <= y <= 1:
        raise ValueError("MAI-UI action must be normalized to [0,1] before translation")
    return x, y


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"MAI-UI action requires non-empty {key}")
    return value
