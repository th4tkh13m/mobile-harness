"""Validated settings for a live ADB harness run; secrets stay in the environment."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .model import ActionKind


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass(frozen=True)
class ModelSettings:
    name: str | None
    base_url: str | None


@dataclass(frozen=True)
class DeviceSettings:
    serial: str | None
    adb_timeout_seconds: float


@dataclass(frozen=True)
class HarnessSettings:
    max_steps: int
    trace_path: str
    zoom_ratio: float


@dataclass(frozen=True)
class RuntimeSettings:
    provider: str
    max_turns: int
    session_root: str
    workspace_root: str
    capture_after_actions: bool
    require_plan_before_actions: bool
    require_plan_update_on_transition: bool
    context_window_tokens: int
    memory_root: str
    skills_root: str
    compaction_summary_timeout_seconds: float
    model_stale_timeout_seconds: float
    model_streaming: bool
    model_stream_progress_interval_seconds: float


@dataclass(frozen=True)
class AuthoritySettings:
    auto_approve_mobile_actions: bool


@dataclass(frozen=True)
class PolicySettings:
    sensitive_terms: tuple[str, ...]
    guarded_action_kinds: tuple[str, ...]


@dataclass(frozen=True)
class HarnessConfig:
    model: ModelSettings
    device: DeviceSettings
    harness: HarnessSettings
    runtime: RuntimeSettings
    authority: AuthoritySettings
    policy: PolicySettings


def load_harness_config(config_path: str | Path | None = None) -> HarnessConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    try:
        with open(path, encoding="utf-8") as config_file:
            data = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load harness config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("harness config must be a JSON object")
    model = _section(data, "model")
    device = _section(data, "device")
    harness = _section(data, "harness")
    runtime = _section(data, "runtime")
    authority = _section(data, "authority")
    policy = _section(data, "policy")
    actions = _string_list(policy, "guarded_action_kinds")
    unknown = set(actions) - {kind.value for kind in ActionKind}
    if unknown:
        raise ValueError(f"harness config has unsupported action kinds: {', '.join(sorted(unknown))}")
    return HarnessConfig(
        model=ModelSettings(_optional_string(model, "name"), _optional_string(model, "base_url")),
        device=DeviceSettings(_optional_string(device, "serial"), _positive_number(device, "adb_timeout_seconds")),
        harness=HarnessSettings(_positive_int(harness, "max_steps"), _required_string(harness, "trace_path"), _unit_ratio(harness, "zoom_ratio")),
        runtime=RuntimeSettings(_enum(runtime, "provider", {"chat_completions", "responses"}), _positive_int(runtime, "max_turns"), _required_string(runtime, "session_root"), _required_string(runtime, "workspace_root"), _bool(runtime, "capture_after_actions"), _bool(runtime, "require_plan_before_actions"), _bool(runtime, "require_plan_update_on_transition"), _positive_int(runtime, "context_window_tokens"), _required_string(runtime, "memory_root"), _required_string(runtime, "skills_root"), _positive_number(runtime, "compaction_summary_timeout_seconds"), _positive_number(runtime, "model_stale_timeout_seconds"), _bool(runtime, "model_streaming"), _positive_number(runtime, "model_stream_progress_interval_seconds")),
        authority=AuthoritySettings(_bool(authority, "auto_approve_mobile_actions")),
        policy=PolicySettings(_string_list(policy, "sensitive_terms", allow_empty=True), actions),
    )


def _section(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"harness config {key} must be an object")
    return value


def _optional_string(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"harness config {key} must be a non-empty string or null")
    return value


def _required_string(data: dict[str, object], key: str) -> str:
    value = _optional_string(data, key)
    if value is None:
        raise ValueError(f"harness config {key} must be a non-empty string")
    return value


def _string_list(data: dict[str, object], key: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    values = data.get(key)
    if not isinstance(values, list) or (not allow_empty and not values) or not all(isinstance(value, str) and value.strip() for value in values):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"harness config {key} must be {qualifier} of strings")
    return tuple(value.lower() for value in values)


def _positive_number(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"harness config {key} must be positive")
    return float(value)


def _positive_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"harness config {key} must be a positive integer")
    return value


def _unit_ratio(data: dict[str, object], key: str) -> float:
    value = _positive_number(data, key)
    if value > 1:
        raise ValueError(f"harness config {key} must be at most 1")
    return value


def _enum(data: dict[str, object], key: str, allowed: set[str]) -> str:
    value = _required_string(data, key)
    if value not in allowed:
        raise ValueError(f"harness config {key} must be one of: {', '.join(sorted(allowed))}")
    return value


def _bool(data: dict[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"harness config {key} must be a boolean")
    return value
