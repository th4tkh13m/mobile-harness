from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from .model import ActionKind, ActionResult, Decision, Observation
from .policy import DevicePolicy
from .ports import DevicePort, resolve_tap


class Agent(Protocol):
    def decide(self, task: str, observation: Observation, history: tuple["Event", ...]) -> Decision: ...


class Verifier(Protocol):
    def verify(self, task: str, device: DevicePort, observation: Observation, history: tuple["Event", ...]) -> tuple[bool, str]: ...


@dataclass(frozen=True)
class Event:
    step: int
    decision: Decision
    result: ActionResult | None
    policy_message: str = ""


@dataclass(frozen=True)
class HarnessResult:
    success: bool
    reason: str
    events: tuple[Event, ...]
    final_observation: Observation


class Harness:
    """Observe -> one policy-gated action -> re-observe -> verify terminal claims."""

    def __init__(self, device: DevicePort, verifier: Verifier, policy: DevicePolicy | None = None, max_steps: int = 30, trace: object | None = None, zoom_ratio: float = .5) -> None:
        self.device = device
        self.verifier = verifier
        self.policy = policy or DevicePolicy()
        self.max_steps = max_steps
        self.trace = trace
        if not 0 < zoom_ratio <= 1:
            raise ValueError("zoom_ratio must be in (0, 1]")
        self.zoom_ratio = zoom_ratio

    def _record(self, events: list[Event], event: Event) -> None:
        events.append(event)
        if self.trace is not None:
            self.trace.write(event)

    @staticmethod
    def _resolve_action(action, observation: Observation):
        """Resolve a visible locator before crossing any device/benchmark port.

        This preserves one tool contract for raw ADB and benchmark adapters:
        models can request a semantic target, while each port receives only a
        concrete normalized gesture.
        """
        if action.locator is None:
            return action
        x, y = resolve_tap(action, observation)
        return replace(action, x=x / observation.width, y=y / observation.height)

    def run(self, task: str, agent: Agent) -> HarnessResult:
        observation = self.device.observe()
        events: list[Event] = []
        for step in range(1, self.max_steps + 1):
            decision = agent.decide(task, observation, tuple(events))
            if decision.done:
                passed, message = self.verifier.verify(task, self.device, observation, tuple(events))
                self._record(events, Event(step, decision, None, message))
                if passed:
                    return HarnessResult(True, message, tuple(events), observation)
                observation = self.device.observe()
                continue
            assert decision.action is not None
            action = self._resolve_action(decision.action, observation)
            policy = self.policy.check(action, observation)
            if not policy.allowed:
                self._record(events, Event(step, decision, ActionResult(False, "blocked by policy"), policy.reason))
                observation = self.device.observe()
                continue
            if action.kind is ActionKind.ZOOM:
                try:
                    if action.x is None or action.y is None:
                        raise ValueError("zoom requires x/y")
                    ratio = action.ratio or self.zoom_ratio
                    observation = observation.zoomed(action.x, action.y, ratio)
                    self._record(events, Event(step, decision, ActionResult(True, f"zoomed to {observation.width}x{observation.height}")))
                except ValueError as exc:
                    self._record(events, Event(step, decision, ActionResult(False, str(exc))))
                continue
            result = self.device.act(action, observation)
            self._record(events, Event(step, replace(decision, action=action), result, policy.reason))
            observation = self.device.observe()
        return HarnessResult(False, f"step limit reached ({self.max_steps})", tuple(events), observation)
