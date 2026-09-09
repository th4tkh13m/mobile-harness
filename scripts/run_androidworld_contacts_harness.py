"""End-to-end AndroidWorld smoke run through the mobile harness.

This deliberately scripted agent proves the harness boundary and verifier; it
is not intended as the general-purpose policy. A model-backed agent can issue
the same typed actions through ``OpenAICompatibleAgent``.
"""
from __future__ import annotations

import re

from android_world.env import android_world_controller, interface
from android_world.task_evals.single.contacts import ContactsAddContact

from mobile_harness.core import Agent
from mobile_harness.model import Action, ActionKind, Decision, Observation
from mobile_harness.runners import run_androidworld_task


ADB_PATH = r"C:\Users\Khiem\AppData\Local\Android\Sdk\platform-tools\adb.exe"


class ContactsSmokeAgent(Agent):
    """Replay the visible Contacts flow for the randomized benchmark goal."""

    def __init__(self) -> None:
        self._step = 0
        self._first = ""
        self._last = ""
        self._number = ""

    def decide(self, task: str, observation: Observation, history) -> Decision:
        if not self._first:
            match = re.fullmatch(r"Create a new contact for (.+)\. Their number is (.+)\.", task)
            if not match:
                raise ValueError(f"Unexpected ContactsAddContact goal: {task}")
            self._first, self._last = match.group(1).rsplit(" ", 1)
            self._number = match.group(2)
        actions = (
            Action(ActionKind.LAUNCH_APP, package="contacts"),
            Action(ActionKind.TAP, locator={"content_desc": "Create contact"}),
            Action(ActionKind.TAP, locator={"text": "First name"}),
            Action(ActionKind.TYPE_TEXT, text=self._first),
            Action(ActionKind.TAP, locator={"text": "Last name"}),
            Action(ActionKind.TYPE_TEXT, text=self._last),
            Action(ActionKind.TAP, locator={"text": "Phone"}),
            Action(ActionKind.TYPE_TEXT, text=self._number),
            Action(ActionKind.TAP, locator={"text": "Save"}),
        )
        if self._step >= len(actions):
            print("AGENT claim_done", flush=True)
            return Decision(done=True, reason="contact saved; request benchmark verification")
        action = actions[self._step]
        print(f"AGENT step={self._step + 1} action={action.kind.value} locator={action.locator}", flush=True)
        self._step += 1
        return Decision(action=action)


def main() -> None:
    controller = android_world_controller.get_controller(
        console_port=5554,
        adb_path=ADB_PATH,
        a11y_method=android_world_controller.A11yMethod.UIAUTOMATOR,
        install_a11y_forwarding_app=False,
    )
    env = interface.AsyncAndroidEnv(controller)
    task = ContactsAddContact(ContactsAddContact.generate_random_params())
    try:
        print(f"TASK goal={task.goal}", flush=True)
        result = run_androidworld_task(env, task, ContactsSmokeAgent(), max_steps=12)
        print(f"HARNESS_SUCCESS={result.success}")
        print(f"HARNESS_REASON={result.reason}")
        print(f"HARNESS_STEPS={len(result.events)}")
        if not result.success:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
