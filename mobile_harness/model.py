from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SUPPORTED_KEY_NAMES = (
    "ENTER", "BACK", "HOME", "TAB", "DEL", "FORWARD_DEL", "ESCAPE", "SPACE",
    "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT", "DPAD_CENTER",
    "MOVE_HOME", "MOVE_END", "PAGE_UP", "PAGE_DOWN",
    "VOLUME_UP", "VOLUME_DOWN", "VOLUME_MUTE",
)

# The model-facing action space follows Mobile-Agent-v3.5 / GUI-Owl: every
# screenshot and crop is treated as a 1000 by 1000 logical canvas.  The device
# port still receives normalized internal coordinates, which keeps benchmark
# adapters independent of the model contract.
MODEL_COORDINATE_MAX = 1000


def normalize_model_coordinate(value: object, name: str) -> float:
    """Validate one integer model coordinate and convert it to [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"{name} must be an integer in [0, {MODEL_COORDINATE_MAX}]")
    if not 0 <= value <= MODEL_COORDINATE_MAX:
        raise ValueError(f"{name} must be in [0, {MODEL_COORDINATE_MAX}]")
    return int(value) / MODEL_COORDINATE_MAX


def validate_key_name(key: str | None) -> str:
    if key not in SUPPORTED_KEY_NAMES:
        raise ValueError(f"unsupported key {key!r}; use one of: {', '.join(SUPPORTED_KEY_NAMES)}")
    return key


class ActionKind(str, Enum):
    TAP = "tap"
    SWIPE = "swipe"
    TYPE_TEXT = "type_text"
    KEY = "key"
    ZOOM = "zoom"
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
    viewport_left: int = 0
    viewport_top: int = 0
    physical_width: int | None = None
    physical_height: int | None = None

    def zoomed(self, x: float, y: float, ratio: float) -> "Observation":
        """Return an aspect-preserving, coordinate-preserving visual crop."""
        if not self.screenshot_png:
            raise ValueError("zoom requires a screenshot")
        if not 0 <= x <= 1 or not 0 <= y <= 1 or not 0 < ratio <= 1:
            raise ValueError("zoom requires x/y in [0, 1] and ratio in (0, 1]")
        crop_width, crop_height = max(1, round(self.width * ratio)), max(1, round(self.height * ratio))
        left = min(max(round(x * self.width - crop_width / 2), 0), self.width - crop_width)
        top = min(max(round(y * self.height - crop_height / 2), 0), self.height - crop_height)
        return self.cropped(left, top, crop_width, crop_height)

    def cropped(self, left: int, top: int, crop_width: int, crop_height: int) -> "Observation":
        """Return a viewport crop in this observation's local pixel space."""
        if not self.screenshot_png:
            raise ValueError("crop requires a screenshot")
        if left < 0 or top < 0 or crop_width < 1 or crop_height < 1 or left + crop_width > self.width or top + crop_height > self.height:
            raise ValueError("crop is outside the current observation")
        from PIL import Image
        with Image.open(io.BytesIO(self.screenshot_png)) as image:
            crop = image.crop((left, top, left + crop_width, top + crop_height))
            output = io.BytesIO()
            crop.save(output, format="PNG")
        elements = tuple(
            UIElement(
                Rect(max(item.bounds.left, left) - left, max(item.bounds.top, top) - top,
                     min(item.bounds.right, left + crop_width) - left, min(item.bounds.bottom, top + crop_height) - top),
                item.text, item.content_desc, item.resource_id, item.clickable,
            )
            for item in self.elements
            if item.bounds.right > left and item.bounds.left < left + crop_width and item.bounds.bottom > top and item.bounds.top < top + crop_height
        )
        return Observation(crop_width, crop_height, output.getvalue(), None, elements, self.activity,
                           self.viewport_left + left, self.viewport_top + top,
                           self.physical_width or self.width, self.physical_height or self.height)

    def to_physical_point(self, x: int, y: int) -> tuple[int, int]:
        width, height = self.physical_width or self.width, self.physical_height or self.height
        return min(self.viewport_left + x, width - 1), min(self.viewport_top + y, height - 1)


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
    ratio: float | None = None
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
