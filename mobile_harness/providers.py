"""Model adapters for the durable runtime; no provider SDK is required."""
from __future__ import annotations

import json
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


@dataclass
class OpenAIChatRuntimeModel:
    base_url: str
    api_key: str
    model: str
    post: Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]] = _post

    @staticmethod
    def token_counter(encode: Callable[[str], Any]) -> Callable[[str], int]:
        return token_counter_from_encoding(encode)

    def respond(self, *, system: str, context: dict[str, Any], tools: list[dict[str, Any]]) -> tuple[str, list[ToolCall]]:
        payload = {"model": self.model, "temperature": 0, "tools": tools,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]}
        message = self.post(self.base_url.rstrip("/") + "/chat/completions", {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, payload)["choices"][0]["message"]
        return message.get("content") or "", _calls(message.get("tool_calls") or [])


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
        payload = {"model": self.model, "instructions": system, "tools": tools, "input": json.dumps(context, ensure_ascii=False)}
        response = self.post(self.base_url.rstrip("/") + "/responses", {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, payload)
        output = response.get("output", [])
        text = "".join(part.get("text", "") for item in output for part in item.get("content", []) if part.get("type") == "output_text")
        calls = [ToolCall(item.get("name", ""), json.loads(item.get("arguments", "{}")), item.get("call_id", "")) for item in output if item.get("type") == "function_call"]
        return text, calls
