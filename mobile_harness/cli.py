from __future__ import annotations

import argparse
import getpass
import os

from .core import Event, Harness
from .openai_agent import OpenAICompatibleAgent
from .policy import DevicePolicy
from .ports import AdbDevice, DevicePort
from .trace import JsonlTrace


class PromptVerifier:
    """Real-device terminal check: never convert a model claim directly to success."""

    def verify(self, task: str, device: DevicePort, observation, history: tuple[Event, ...]) -> tuple[bool, str]:
        answer = input("The agent claims it completed the task. Verify the real device outcome and type yes to accept: ").strip().lower()
        return (True, "user verified real device outcome") if answer in {"y", "yes"} else (False, "user did not verify completion")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the constrained mobile harness against an ADB device.")
    parser.add_argument("task")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL, without /chat/completions")
    parser.add_argument("--serial")
    parser.add_argument("--trace", default="mobile_harness_trace.jsonl")
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("Model API key: ")
    device = AdbDevice(serial=args.serial)
    agent = OpenAICompatibleAgent(args.base_url, api_key, args.model)
    result = Harness(device, PromptVerifier(), DevicePolicy(), args.max_steps, JsonlTrace(args.trace)).run(args.task, agent)
    print(result.reason)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
