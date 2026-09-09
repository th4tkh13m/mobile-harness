"""Model adapters for the durable runtime; no provider SDK is required."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib import request

from .tools import ToolCall


def token_counter_from_encoding(encode: Callable[[str], Any]) -> Callable[[str], int]:
    """Adapt any provider/model tokenizer's encode function for PromptAssembler."""
    return lambda text: len(encode(text))


def _post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    req = request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with request.urlopen(req, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def _calls(items: list[dict[str, Any]]) -> list[ToolCall]:
    answer = []
    for item in items:
        function = item.get("function", item)
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str): arguments = json.loads(arguments)
        answer.append(ToolCall(function.get("name", ""), arguments, item.get("id", "")))
    return answer


class ProviderStalledError(TimeoutError):
    """The provider accepted a streaming request but stopped making progress."""


StreamEvent = Callable[[str, dict[str, Any]], None]


def _emit(callback: StreamEvent | None, kind: str, **payload: Any) -> None:
    if callback:
        callback(kind, payload)


def _stream_chat(url: str, headers: dict[str, str], body: dict[str, Any], *, stale_timeout_seconds: float,
                 progress_interval_seconds: float = 15.0, on_event: StreamEvent | None = None) -> tuple[str, list[ToolCall]]:
    """Consume OpenAI-compatible SSE without retaining every generated token.

    ``urlopen``'s timeout is deliberately an *idle* timeout here: each received
    SSE event resets it, so slow but actively decoding servers remain healthy.
    """
    started = time.monotonic()
    _emit(on_event, "request_started", streaming=True, stale_timeout_seconds=stale_timeout_seconds)
    # Avoid optional extensions (such as ``stream_options``) so vLLM and other
    # OpenAI-compatible servers can use this path unchanged.
    request_body = dict(body, stream=True)
    req = request.Request(url, data=json.dumps(request_body).encode("utf-8"), headers=headers, method="POST")
    text_parts: list[str] = []
    tool_parts: dict[int, dict[str, Any]] = {}
    first_delta_at: float | None = None
    stream_events = content_characters = reasoning_characters = tool_argument_characters = 0
    event_kinds: set[str] = set()
    next_progress_at = started + progress_interval_seconds

    def progress_payload(*, elapsed: float) -> dict[str, Any]:
        return {"elapsed_seconds": round(elapsed, 3), "stream_events": stream_events,
                "content_characters": content_characters, "reasoning_characters": reasoning_characters,
                "tool_argument_characters": tool_argument_characters, "event_kinds": sorted(event_kinds)}
    try:
        with request.urlopen(req, timeout=stale_timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid provider SSE frame: {exc}") from exc
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                stream_events += 1
                for field in ("content", "reasoning", "reasoning_content", "tool_calls"):
                    if delta.get(field):
                        event_kinds.add(field)
                if first_delta_at is None and delta:
                    first_delta_at = time.monotonic()
                    _emit(on_event, "first_delta", elapsed_seconds=round(first_delta_at - started, 3))
                if isinstance(delta.get("content"), str):
                    text_parts.append(delta["content"])
                    content_characters += len(delta["content"])
                for field in ("reasoning", "reasoning_content"):
                    if isinstance(delta.get(field), str):
                        reasoning_characters += len(delta[field])
                for item in delta.get("tool_calls") or []:
                    index = int(item.get("index", 0))
                    assembled = tool_parts.setdefault(index, {"id": "", "function": {"name": "", "arguments": "{}"}})
                    if item.get("id"):
                        assembled["id"] = item["id"]
                    function = item.get("function") or {}
                    if function.get("name"):
                        assembled["function"]["name"] += function["name"]
                    if function.get("arguments"):
                        existing = assembled["function"]["arguments"]
                        assembled["function"]["arguments"] = ("" if existing == "{}" else existing) + function["arguments"]
                        tool_argument_characters += len(function["arguments"])
                now = time.monotonic()
                if now >= next_progress_at:
                    _emit(on_event, "stream_progress", **progress_payload(elapsed=now - started))
                    next_progress_at = now + progress_interval_seconds
    except TimeoutError as exc:
        elapsed = round(time.monotonic() - started, 3)
        _emit(on_event, "stalled", elapsed_seconds=elapsed, first_delta_seconds=(round(first_delta_at - started, 3) if first_delta_at else None))
        raise ProviderStalledError(f"provider stream idle for {stale_timeout_seconds:g}s (elapsed {elapsed:g}s)") from exc
    calls = _calls([tool_parts[index] for index in sorted(tool_parts)])
    elapsed = round(time.monotonic() - started, 3)
    completed_payload = progress_payload(elapsed=elapsed)
    completed_payload.update(first_delta_seconds=(round(first_delta_at - started, 3) if first_delta_at else None),
                             text_characters=sum(map(len, text_parts)), tool_count=len(calls), streaming=True)
    _emit(on_event, "request_completed", **completed_payload)
    return "".join(text_parts), calls


@dataclass
class OpenAIChatRuntimeModel:
    base_url: str
    api_key: str
    model: str
    post: Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]] = _post
    stale_timeout_seconds: float = 150.0
    streaming: bool = True
    stream_progress_interval_seconds: float = 15.0

    @staticmethod
    def token_counter(encode: Callable[[str], Any]) -> Callable[[str], int]:
        return token_counter_from_encoding(encode)

    def respond(self, *, system: str, context: dict[str, Any], tools: list[dict[str, Any]],
                on_event: StreamEvent | None = None) -> tuple[str, list[ToolCall]]:
        context = dict(context)
        image = context.pop("current_screen_image", None)
        content: Any = json.dumps(context, ensure_ascii=False)
        if image:
            content = [{"type": "text", "text": content}, {"type": "image_url", "image_url": {"url": image}}]
        payload = {"model": self.model, "temperature": 0, "tools": tools,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        # Custom ``post`` functions are a synchronous test/integration seam; a
        # real OpenAI-compatible endpoint uses the observable SSE path.
        if self.streaming and self.post is _post:
            return _stream_chat(self.base_url.rstrip("/") + "/chat/completions",
                                {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, payload,
                                stale_timeout_seconds=self.stale_timeout_seconds,
                                progress_interval_seconds=self.stream_progress_interval_seconds, on_event=on_event)
        started = time.monotonic()
        _emit(on_event, "request_started", streaming=False)
        message = self.post(self.base_url.rstrip("/") + "/chat/completions", {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, payload)["choices"][0]["message"]
        text, calls = message.get("content") or "", _calls(message.get("tool_calls") or [])
        _emit(on_event, "request_completed", elapsed_seconds=round(time.monotonic() - started, 3), text_characters=len(text), tool_count=len(calls), streaming=False)
        return text, calls


@dataclass
class OpenAIResponsesRuntimeModel:
    base_url: str
    api_key: str
    model: str
    post: Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]] = _post

    @staticmethod
    def token_counter(encode: Callable[[str], Any]) -> Callable[[str], int]:
        return token_counter_from_encoding(encode)

    def respond(self, *, system: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> tuple[str, list[ToolCall]]:
        context = dict(context)
        image = context.pop("current_screen_image", None)
        content: Any = json.dumps(context, ensure_ascii=False)
        if image:
            content = [{"type": "input_text", "text": content}, {"type": "input_image", "image_url": image}]
        payload = {"model": self.model, "instructions": system, "tools": tools, "input": content}
        response = self.post(self.base_url.rstrip("/") + "/responses", {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, payload)
        output = response.get("output", [])
        text = "".join(part.get("text", "") for item in output for part in item.get("content", []) if part.get("type") == "output_text")
        calls = [ToolCall(item.get("name", ""), json.loads(item.get("arguments", "{}")), item.get("call_id", "")) for item in output if item.get("type") == "function_call"]
        return text, calls
