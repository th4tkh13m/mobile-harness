from __future__ import annotations

import argparse
import getpass
import os

from .config import load_harness_config
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
    parser.add_argument("--config", help="Path to central JSON settings; defaults to bundled config.json")
    parser.add_argument("--model", help="Overrides model.name from the config")
    parser.add_argument("--base-url", help="Overrides model.base_url from the config; without /chat/completions")
    parser.add_argument("--serial")
    parser.add_argument("--trace", help="Overrides harness.trace_path from the config")
    parser.add_argument("--max-steps", type=int, help="Overrides harness.max_steps from the config")
    args = parser.parse_args()
    config = load_harness_config(args.config)
    model = args.model or config.model.name
    base_url = args.base_url or config.model.base_url
    if not model or not base_url:
        parser.error("set model.name and model.base_url in --config, or pass --model and --base-url")
    api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("Model API key: ")
    device = AdbDevice(serial=args.serial or config.device.serial, timeout_seconds=config.device.adb_timeout_seconds)
    agent = OpenAICompatibleAgent(base_url, api_key, model)
    result = Harness(device, PromptVerifier(), DevicePolicy(settings=config.policy), args.max_steps or config.harness.max_steps, JsonlTrace(args.trace or config.harness.trace_path), zoom_ratio=config.harness.zoom_ratio).run(args.task, agent)
    print(result.reason)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
