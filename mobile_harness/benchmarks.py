from __future__ import annotations

import io
from typing import Any, Callable

from .core import Event, Verifier
from .model import Action, ActionKind, ActionResult, Observation, Rect, UIElement
from .ports import CallbackAdapter, DevicePort


class AndroidWorldAdapter(CallbackAdapter):
    """Bridge an AndroidWorld AsyncEnv without patching its source."""

    def __init__(
        self,
        env: Any,
        execute_json_action: Callable[[Any, list[Any], tuple[int, int], Any], None],
        action_factory: Callable[[Action, int, int], Any],
    ) -> None:
        self.env = env
        self._execute_json_action = execute_json_action
        self._action_factory = action_factory
        super().__init__(self._observe, self._act)

    def _observe(self) -> Observation:
        # Harness already follows every action with a fresh observation.  Do
        # not invoke AndroidWorld's nested multi-sample stability loop here:
        # under its UIAutomator fallback that starts several dump processes per
        # turn and can exhaust a real emulator.
        state = self.env.get_state(wait_to_stabilize=False)
        width, height = self.env.logical_screen_size
        elements = tuple(UIElement(
            Rect(int(item.bbox_pixels.x_min), int(item.bbox_pixels.y_min), int(item.bbox_pixels.x_max), int(item.bbox_pixels.y_max)),
            text=item.text or "", content_desc=item.content_description or "",
        ) for item in state.ui_elements if item.bbox_pixels is not None)
        screenshot = _androidworld_png(state.pixels)
        return Observation(width, height, screenshot_png=screenshot, elements=elements, activity=self.env.foreground_activity_name)

    def _act(self, action: Action, observation: Observation) -> ActionResult:
        try:
            state = self.env.get_state(wait_to_stabilize=False)
            self._execute_json_action(
                self._action_factory(action, observation.width, observation.height),
                state.ui_elements,
                (observation.width, observation.height),
                self.env.controller,
            )
            return ActionResult(True)
        except Exception as exc:
            return ActionResult(False, str(exc))


def _androidworld_png(pixels: Any) -> bytes | None:
    """Convert the actual AndroidWorld RGB observation to vision-model input."""
    if pixels is None:
        return None
    try:
        from PIL import Image
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        return None


def androidworld_action_factory(action: Action, width: int, height: int) -> Any:
    """Convert the portable vocabulary to AndroidWorld's JSONAction at the edge.

    AndroidWorld is imported lazily so ADB users do not need its dependencies.
    This intentionally covers the action types implemented by AndroidWorld's
    `execute_adb_action`, not the benchmark-only answer/terminate pseudo-tools.
    """
    from android_world.agents.new_json_action import JSONAction
    from .ports import pixel

    if action.kind is ActionKind.TAP:
        if action.x is None or action.y is None:
            raise ValueError("AndroidWorld tap requires normalized x/y; resolve locators before conversion")
        return JSONAction(action_type="click", x=pixel(action.x, width), y=pixel(action.y, height))
    if action.kind is ActionKind.TYPE_TEXT:
        return JSONAction(action_type="input_text", text=action.text)
    if action.kind is ActionKind.KEY and action.key in {"BACK", "KEYCODE_BACK"}:
        return JSONAction(action_type="navigate_back")
    if action.kind is ActionKind.KEY and action.key in {"HOME", "KEYCODE_HOME"}:
        return JSONAction(action_type="navigate_home")
    if action.kind is ActionKind.LAUNCH_APP:
        return JSONAction(action_type="open_app", app_name=action.package)
    if action.kind is ActionKind.WAIT:
        return JSONAction(action_type="wait")
    if action.kind is ActionKind.SWIPE:
        if None in (action.x, action.y, action.x2, action.y2):
            raise ValueError("AndroidWorld swipe requires four normalized coordinates")
        converted = JSONAction(action_type="swipe")
        converted.direction = (pixel(action.x, width), pixel(action.y, height), pixel(action.x2, width), pixel(action.y2, height))
        return converted
    raise ValueError(f"AndroidWorld cannot map action {action.kind}")


class AndroidWorldVerifier(Verifier):
    """Calls AndroidWorld's canonical task.is_successful(env) online."""

    def __init__(self, task_eval: Any, env: Any) -> None:
        self.task_eval = task_eval
        self.env = env

    def verify(self, task: str, device: DevicePort, observation: Observation, history: tuple[Event, ...]) -> tuple[bool, str]:
        score = float(self.task_eval.is_successful(self.env))
        return score >= 1.0, f"AndroidWorld score={score}"


class BenchmarkSuccessVerifier(Verifier):
    """Turns a benchmark's authoritative success callback into online verification."""

    def __init__(self, is_successful: Callable[[DevicePort], bool]) -> None:
        self._is_successful = is_successful

    def verify(self, task: str, device: DevicePort, observation: Observation, history: tuple[Event, ...]) -> tuple[bool, str]:
        return (True, "benchmark task success") if self._is_successful(device) else (False, "benchmark has not observed task success")


class MobileWorldAdapter(CallbackAdapter):
    """Concrete port for MobileWorld's `AndroidEnvClient`.

    Create it only after `env.initialize_task(task_name)`: task snapshots and
    task-specific initialization belong to MobileWorld, while this harness owns
    the observe/act/verify loop afterwards.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        super().__init__(self._observe, self._act)

    def _observe(self) -> Observation:
        image = self.env.get_screenshot(wait_to_stabilize=True)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return Observation(image.width, image.height, screenshot_png=buffer.getvalue())

    def _act(self, action: Action, observation: Observation) -> ActionResult:
        try:
            self.env.execute_action(_mobileworld_action(action, observation.width, observation.height))
            return ActionResult(True)
        except Exception as exc:
            return ActionResult(False, str(exc))


class MobileWorldVerifier(Verifier):
    """Uses MobileWorld's task-specific backend/storage/callback evaluator."""

    def __init__(self, env: Any, task_name: str) -> None:
        self.env = env
        self.task_name = task_name

    def verify(self, task: str, device: DevicePort, observation: Observation, history: tuple[Event, ...]) -> tuple[bool, str]:
        score, reason = self.env.get_task_score(task_type=self.task_name)
        return score > 0.0, f"MobileWorld score={score}: {reason}"


class MemGUIAdapter(MobileWorldAdapter):
    """Concrete port for MemGUI-Bench's MobileWorld-compatible runtime client.

    MemGUI-Bench ships a `src/mobile_world` runtime, including the same
    screenshot/action/task-score boundary as MobileWorld. Memory persistence is
    handled by `VerifiedExperienceStore`; the benchmark adapter only mediates
    the live device episode.
    """


class MemGUIVerifier(MobileWorldVerifier):
    """Calls MemGUI-Bench's authoritative task evaluator on terminal claims."""

    def verify(self, task: str, device: DevicePort, observation: Observation, history: tuple[Event, ...]) -> tuple[bool, str]:
        score, reason = self.env.get_task_score(task_type=self.task_name)
        return score > 0.0, f"MemGUI score={score}: {reason}"


def _mobileworld_action(action: Action, width: int, height: int) -> Any:
    """Map portable actions to MobileWorld's real `JSONAction` at the edge."""
    try:
        from mobile_world.runtime.utils.models import JSONAction
    except ImportError as exc:
        raise RuntimeError("MobileWorld is not importable; install its dependencies and add its src directory to PYTHONPATH") from exc
    from .ports import pixel

    if action.kind is ActionKind.TAP:
        if action.x is None or action.y is None:
            raise ValueError("MobileWorld tap requires normalized x/y; resolve locators before conversion")
        return JSONAction(action_type="click", x=pixel(action.x, width), y=pixel(action.y, height))
    if action.kind is ActionKind.SWIPE:
        if None in (action.x, action.y, action.x2, action.y2):
            raise ValueError("MobileWorld swipe requires four normalized coordinates")
        return JSONAction(action_type="drag", start_x=pixel(action.x, width), start_y=pixel(action.y, height), end_x=pixel(action.x2, width), end_y=pixel(action.y2, height))
    if action.kind is ActionKind.TYPE_TEXT:
        return JSONAction(action_type="input_text", text=action.text)
    if action.kind is ActionKind.KEY:
        if action.key in {"BACK", "KEYCODE_BACK"}:
            return JSONAction(action_type="navigate_back")
        if action.key in {"HOME", "KEYCODE_HOME"}:
            return JSONAction(action_type="navigate_home")
        if action.key != "ENTER":
            raise ValueError("MobileWorld adapter currently supports only KEY ENTER, BACK, or HOME")
        return JSONAction(action_type="keyboard_enter")
    if action.kind is ActionKind.LAUNCH_APP:
        return JSONAction(action_type="open_app", app_name=action.package)
    if action.kind is ActionKind.WAIT:
        return JSONAction(action_type="wait")
    raise ValueError(f"MobileWorld cannot map action {action.kind}")
