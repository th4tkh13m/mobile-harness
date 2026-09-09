"""Provider-neutral function schemas and strict parsing for the mobile action waist."""
from __future__ import annotations

from typing import Any

from .model import Action, ActionKind, Decision


TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {"name": "tap", "description": "Tap one uniquely identified UI element, or use normalized x/y only when no locator is available.", "parameters": {"type": "object", "properties": {"x": {"type": "number", "minimum": 0, "maximum": 1}, "y": {"type": "number", "minimum": 0, "maximum": 1}, "locator": {"type": "object", "properties": {"text": {"type": "string"}, "content_desc": {"type": "string"}, "resource_id": {"type": "string"}}, "additionalProperties": False}}, "additionalProperties": False}},
    {"name": "swipe", "description": "Swipe using normalized coordinates in [0,1].", "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "x2": {"type": "number"}, "y2": {"type": "number"}, "duration_ms": {"type": "integer", "minimum": 1}}, "required": ["x", "y", "x2", "y2"], "additionalProperties": False}},
    {"name": "type_text", "description": "Type text into the focused field. Do not type credentials or secrets.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "minLength": 1}}, "required": ["text"], "additionalProperties": False}},
    {"name": "key", "description": "Press a named Android key, such as ENTER or DELETE.", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"], "additionalProperties": False}},
    {"name": "back", "description": "Navigate back one screen.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "home", "description": "Navigate to the Android home screen.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "launch_app", "description": "Launch an installed application package.", "parameters": {"type": "object", "properties": {"package": {"type": "string"}}, "required": ["package"], "additionalProperties": False}},
    {"name": "wait", "description": "Wait for a UI transition, then the harness will capture a new observation.", "parameters": {"type": "object", "properties": {"duration_ms": {"type": "integer", "minimum": 1, "maximum": 10000}}, "additionalProperties": False}},
    {"name": "claim_done", "description": "Claim task completion. The harness verifies real task state before returning success.", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "additionalProperties": False}},
)


def openai_tools() -> list[dict[str, Any]]:
    return [{"type": "function", "function": schema} for schema in TOOL_SCHEMAS]


def decision_from_tool_call(name: str, arguments: dict[str, Any]) -> Decision:
    if name == "claim_done":
        return Decision(done=True, reason=str(arguments.get("reason", "")))
    try:
        kind = ActionKind(name)
    except ValueError as exc:
        raise ValueError(f"unknown mobile tool: {name}") from exc
    allowed = {"x", "y", "x2", "y2", "text", "key", "package", "locator", "duration_ms"}
    extra = set(arguments) - allowed
    if extra:
        raise ValueError(f"unsupported action arguments: {sorted(extra)}")
    action = Action(kind=kind, **arguments)
    if kind is ActionKind.TAP and not action.locator and (action.x is None or action.y is None):
        raise ValueError("tap requires locator or x/y")
    if kind is ActionKind.SWIPE and None in (action.x, action.y, action.x2, action.y2):
        raise ValueError("swipe requires x/y/x2/y2")
    if kind is ActionKind.TYPE_TEXT and not action.text:
        raise ValueError("type_text requires text")
    return Decision(action=action)
