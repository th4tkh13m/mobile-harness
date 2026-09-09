"""Provider failover and bounded retry used by the runtime model boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol
import time

from .tools import ToolCall


class ModelLike(Protocol):
    def respond(self, *, system: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> tuple[str, list[ToolCall]]: ...


@dataclass(frozen=True)
class ModelAttempt:
    provider_index: int
    error: str = ""
    retry_delay_seconds: float = 0.0


class ResilientModel:
    """Retries transient model failures, then advances to a compatible fallback."""
    def __init__(self, models: list[ModelLike], retries_per_model: int = 1, *, initial_backoff_seconds: float = 0.5,
                 max_backoff_seconds: float = 8.0, sleeper: Callable[[float], None] = time.sleep) -> None:
        if not models: raise ValueError("at least one model is required")
        if retries_per_model < 0 or initial_backoff_seconds < 0 or max_backoff_seconds < 0:
            raise ValueError("retry and backoff settings must be non-negative")
        self.models, self.retries_per_model, self.attempts = models, retries_per_model, []
        self.initial_backoff_seconds, self.max_backoff_seconds, self.sleeper = initial_backoff_seconds, max_backoff_seconds, sleeper

    def respond(self, *, system: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> tuple[str, list[ToolCall]]:
        self.attempts = []
        failure: Exception | None = None
        for index, model in enumerate(self.models):
            for attempt_index in range(self.retries_per_model + 1):
                try:
                    result = model.respond(system=system, context=context, tools=tools)
                    self.attempts.append(ModelAttempt(index))
                    return result
                except (OSError, TimeoutError, ValueError, KeyError) as exc:
                    delay = self._retry_delay(exc, attempt_index) if attempt_index < self.retries_per_model else 0.0
                    failure = exc; self.attempts.append(ModelAttempt(index, str(exc), delay))
                    if delay:
                        self.sleeper(delay)
        raise RuntimeError(f"all model providers failed after {len(self.attempts)} attempts: {failure}")

    def _retry_delay(self, error: Exception, attempt_index: int) -> float:
        """Prefer a provider's numeric Retry-After hint, always within policy."""
        hint = getattr(error, "retry_after_seconds", None)
        if hint is None:
            headers = getattr(error, "headers", None)
            try:
                hint = headers.get("Retry-After") if headers else None
            except AttributeError:
                hint = None
        try:
            if hint is not None and float(hint) >= 0:
                return min(self.max_backoff_seconds, float(hint))
        except (TypeError, ValueError):
            pass
        return min(self.max_backoff_seconds, self.initial_backoff_seconds * (2 ** attempt_index))
