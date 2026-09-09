from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionKind(str, Enum):
    TAP = "tap"
    SWIPE = "swipe"
    TYPE_TEXT = "type_text"
    KEY = "key"
    BACK = "back"
    HOME = "home"
    LAUNCH_APP = "launch_app"
    WAIT = "wait"


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)


@dataclass(frozen=True)
class UIElement:
    bounds: Rect
    text: str = ""
    content_desc: str = ""
    resource_id: str = ""
    clickable: bool = False


@dataclass(frozen=True)
class Observation:
    width: int
    height: int
    screenshot_png: bytes | None = None
    ui_xml: str | None = None
    elements: tuple[UIElement, ...] = ()
    activity: str | None = None


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    text: str | None = None
    key: str | None = None
    package: str | None = None
    locator: dict[str, str] | None = None
    duration_ms: int = 300


@dataclass(frozen=True)
class Decision:
    action: Action | None = None
    done: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.done == (self.action is not None):
            raise ValueError("A decision must contain exactly one of done or action")


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
