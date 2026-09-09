from __future__ import annotations

import re
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Callable, Protocol

from .model import Action, ActionKind, ActionResult, Observation, Rect, UIElement, validate_key_name


class DevicePort(Protocol):
    """The entire execution authority visible to the harness."""

    def observe(self) -> Observation: ...
    def act(self, action: Action, observation: Observation) -> ActionResult: ...


def _parse_bounds(value: str) -> Rect | None:
    match = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", value or "")
    return Rect(*(int(part) for part in match.groups())) if match else None


def parse_uiautomator_xml(xml: str | None) -> tuple[UIElement, ...]:
    if not xml:
        return ()
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ()
    elements: list[UIElement] = []
    for node in root.iter("node"):
        bounds = _parse_bounds(node.get("bounds", ""))
        if bounds:
            elements.append(UIElement(
                bounds=bounds, text=node.get("text", ""),
                content_desc=node.get("content-desc", ""),
                resource_id=node.get("resource-id", ""), clickable=node.get("clickable") == "true",
            ))
    return tuple(elements)


@dataclass
class AdbDevice(DevicePort):
    """ADB-only real-device port: no Termux, root, or arbitrary shell tool."""

    serial: str | None = None
    adb_path: str | None = None
    timeout_seconds: float = 15.0

    def _adb(self, *args: str) -> bytes:
        command = [self.adb_path or resolve_adb_path()]
        if self.serial:
            command += ["-s", self.serial]
        return subprocess.check_output(command + list(args), stderr=subprocess.STDOUT, timeout=self.timeout_seconds)

    def _text(self, *args: str) -> str:
        return self._adb(*args).decode("utf-8", errors="replace")

    def observe(self) -> Observation:
        size = self._text("shell", "wm", "size")
        match = re.search(r"(?:Override size|Physical size):\s*(\d+)x(\d+)", size)
        if not match:
            raise RuntimeError(f"Cannot determine Android screen size: {size}")
        width, height = (int(item) for item in match.groups())
        screenshot = self._adb("exec-out", "screencap", "-p")
        self._text("shell", "uiautomator", "dump", "/sdcard/window.xml")
        xml = self._text("exec-out", "cat", "/sdcard/window.xml")
        # Android 16 no longer consistently reports mCurrentFocus in the window
        # dump; activity state retains the resumed component.
        window_state = self._text("shell", "dumpsys", "activity", "activities")
        return Observation(
            width, height, screenshot, xml, parse_uiautomator_xml(xml),
            activity=foreground_activity(window_state),
        )

    def act(self, action: Action, observation: Observation) -> ActionResult:
        try:
            argv = self._action_argv(action, observation)
            if argv is None:
                return ActionResult(True, "wait")
            return ActionResult(True, self._text(*argv).strip())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
            return ActionResult(False, str(exc))

    def _action_argv(self, action: Action, observation: Observation) -> tuple[str, ...] | None:
        if action.kind is ActionKind.WAIT:
            time.sleep(max(action.duration_ms, 0) / 1000)
            return None
        if action.kind is ActionKind.TAP:
            x, y = resolve_tap(action, observation)
            x, y = observation.to_physical_point(x, y)
            return ("shell", "input", "tap", str(x), str(y))
        if action.kind is ActionKind.SWIPE:
            if None in (action.x, action.y, action.x2, action.y2):
                raise ValueError("swipe requires four normalized coordinates")
            x, y = observation.to_physical_point(pixel(action.x, observation.width), pixel(action.y, observation.height))
            x2, y2 = observation.to_physical_point(pixel(action.x2, observation.width), pixel(action.y2, observation.height))
            return ("shell", "input", "swipe", str(x), str(y), str(x2), str(y2), str(action.duration_ms))
        if action.kind is ActionKind.TYPE_TEXT:
            if not action.text:
                raise ValueError("type_text requires text")
            if any(ord(char) > 127 for char in action.text):
                raise ValueError("ADB input text only supports ASCII; configure an IME adapter for Unicode")
            return ("shell", "input", "text", action.text.replace(" ", "%s"))
        if action.kind is ActionKind.KEY:
            return ("shell", "input", "keyevent", validate_key_name(action.key))
        if action.kind is ActionKind.LAUNCH_APP and action.package:
            return ("shell", "monkey", "-p", action.package, "1")
        raise ValueError(f"unsupported or incomplete action: {action}")


def pixel(value: float, size: int) -> int:
    if not 0 <= value <= 1:
        raise ValueError("normalized coordinates must be in [0, 1]")
    return min(round(value * size), size - 1)


def foreground_activity(window_state: str) -> str | None:
    """Extract the focused Android component without feeding dumpsys noise to a model."""
    match = re.search(r"mCurrentFocus=.*?\s([\w.$]+/[\w.$]+)", window_state)
    if not match:
        match = re.search(r"mFocusedApp=.*?\s([\w.$]+/[\w.$]+)", window_state)
    return match.group(1) if match else None


def resolve_adb_path() -> str:
    """Find platform-tools on PATH or standard Android SDK locations."""
    executable = "adb.exe" if os.name == "nt" else "adb"
    if located := shutil.which(executable):
        return located
    roots = [os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT")]
    if os.environ.get("LOCALAPPDATA"):
        roots.append(str(Path(os.environ["LOCALAPPDATA"]) / "Android" / "Sdk"))
    for root in filter(None, roots):
        # Permit a Windows SDK path during cross-platform orchestration/tests;
        # host OS is not the authority for the target SDK path format.
        candidate = (PureWindowsPath(root) / "platform-tools" / "adb.exe") if re.match(r"^[A-Za-z]:[\\/]", root) else (Path(root) / "platform-tools" / executable)
        if Path(str(candidate)).is_file():
            return str(candidate)
    raise FileNotFoundError("ADB was not found; set ANDROID_HOME/ANDROID_SDK_ROOT or add platform-tools to PATH")


def resolve_tap(action: Action, observation: Observation) -> tuple[int, int]:
    if action.locator:
        matches = [element for element in observation.elements if all(
            getattr(element, name.replace("-", "_"), None) == expected
            for name, expected in action.locator.items()
        )]
        if len(matches) != 1:
            raise ValueError(f"locator must resolve exactly one element; found {len(matches)}")
        return matches[0].bounds.center
    if action.x is None or action.y is None:
        raise ValueError("tap requires normalized x/y or a locator")
    return pixel(action.x, observation.width), pixel(action.y, observation.height)


class CallbackAdapter(DevicePort):
    """Benchmark adapter with explicit callbacks, usable for MobileWorld/MEMGUI."""

    def __init__(self, observe: Callable[[], Observation], act: Callable[[Action, Observation], ActionResult]) -> None:
        self._observe = observe
        self._act = act

    def observe(self) -> Observation:
        return self._observe()

    def act(self, action: Action, observation: Observation) -> ActionResult:
        return self._act(action, observation)
