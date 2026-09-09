"""Credential-free MobileWorld transport smoke test.

This is intentionally not an evaluation: it proves the portable harness can
observe, execute a safe HOME gesture, and re-observe a live MobileWorld server.
Task initialization and scoring remain owned by `run_mobileworld_task`.
"""
from __future__ import annotations

import argparse

from mobile_harness.benchmarks import MobileWorldAdapter
from mobile_harness.core import Harness
from mobile_harness.model import Action, ActionKind, Decision


class _HomeThenDone:
    def decide(self, task, observation, history):
        if not history:
            return Decision(action=Action(ActionKind.HOME))
        return Decision(done=True)


class _FreshObservationVerifier:
    def verify(self, task, device, observation, history):
        image = observation.screenshot_png
        return bool(image), f"MobileWorld transport smoke: screenshot_bytes={len(image or b'')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:6800")
    parser.add_argument("--device", default="emulator-5554")
    args = parser.parse_args()

    from mobile_world.runtime.client import AndroidEnvClient

    env = AndroidEnvClient(url=args.url, device=args.device, step_wait_time=0.5)
    result = Harness(MobileWorldAdapter(env), _FreshObservationVerifier(), max_steps=2).run(
        "MobileWorld live transport smoke", _HomeThenDone()
    )
    print(f"HARNESS_SUCCESS={result.success}")
    print(f"HARNESS_REASON={result.reason}")
    print(f"HARNESS_STEPS={len(result.events)}")
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
