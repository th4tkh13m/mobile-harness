"""Provider-neutral function schemas and strict parsing for the mobile action waist."""
from __future__ import annotations

from typing import Any

from .model import Action, ActionKind, Decision, SUPPORTED_KEY_NAMES, validate_key_name, normalize_model_coordinate


TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {"name": "tap", "description": "Tap one uniquely identified UI element, or use integer x/y on the 0..1000 logical screenshot only when no locator is available.", "parameters": {"type": "object", "properties": {"x": {"type": "integer", "minimum": 0, "maximum": 1000}, "y": {"type": "integer", "minimum": 0, "maximum": 1000}, "locator": {"type": "object", "properties": {"text": {"type": "string"}, "content_desc": {"type": "string"}, "resource_id": {"type": "string"}}, "additionalProperties": False}}, "additionalProperties": False}},
    {"name": "swipe", "description": "Swipe using integer coordinates in the 0..1000 logical screenshot space.", "parameters": {"type": "object", "properties": {"x": {"type": "integer", "minimum": 0, "maximum": 1000}, "y": {"type": "integer", "minimum": 0, "maximum": 1000}, "x2": {"type": "integer", "minimum": 0, "maximum": 1000}, "y2": {"type": "integer", "minimum": 0, "maximum": 1000}, "duration_ms": {"type": "integer", "minimum": 1}}, "required": ["x", "y", "x2", "y2"], "additionalProperties": False}},
    {"name": "type_text", "description": "Type text into the focused field. Do not type credentials or secrets.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "minLength": 1}}, "required": ["text"], "additionalProperties": False}},
    {"name": "key", "description": "Press one supported Android key. BACK and HOME are key values, not separate actions.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "enum": list(SUPPORTED_KEY_NAMES), "description": "ENTER/BACK/HOME/TAB/DEL/FORWARD_DEL/ESCAPE/SPACE, D-pad, cursor navigation, paging, or volume keys."}}, "required": ["key"], "additionalProperties": False}},
    {"name": "zoom", "description": "Create an aspect-preserving cropped observation around integer x/y in the 0..1000 logical screenshot space to refine grounding. This does not touch the device; use one subsequent grounded action in the crop.", "parameters": {"type": "object", "properties": {"x": {"type": "integer", "minimum": 0, "maximum": 1000}, "y": {"type": "integer", "minimum": 0, "maximum": 1000}, "ratio": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}}, "required": ["x", "y"], "additionalProperties": False}},
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
    allowed = {"x", "y", "x2", "y2", "text", "key", "package", "locator", "duration_ms", "ratio"}
    extra = set(arguments) - allowed
    if extra:
        raise ValueError(f"unsupported action arguments: {sorted(extra)}")
    internal = dict(arguments)
    for coordinate in ("x", "y", "x2", "y2"):
        if coordinate in internal:
            internal[coordinate] = normalize_model_coordinate(internal[coordinate], coordinate)
    action = Action(kind=kind, **internal)
    if kind is ActionKind.TAP and not action.locator and (action.x is None or action.y is None):
        raise ValueError("tap requires locator or x/y")
    if kind is ActionKind.SWIPE and None in (action.x, action.y, action.x2, action.y2):
        raise ValueError("swipe requires x/y/x2/y2")
    if kind is ActionKind.TYPE_TEXT and not action.text:
        raise ValueError("type_text requires text")
    if kind is ActionKind.KEY:
        validate_key_name(action.key)
    if kind is ActionKind.ZOOM and (action.x is None or action.y is None or (action.ratio is not None and not 0 < action.ratio <= 1)):
        raise ValueError("zoom requires x/y and an optional ratio in (0, 1]")
    return Decision(action=action)
