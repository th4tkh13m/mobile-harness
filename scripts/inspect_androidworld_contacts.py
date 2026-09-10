"""Initialize one AndroidWorld Contacts task and print its visible UI elements."""
from __future__ import annotations

from android_world.task_evals.single.contacts import ContactsAddContact
from mobile_harness.androidworld import build_environment
from mobile_harness.benchmarks import execute_androidworld_action
from mobile_harness.model import Action, ActionKind


def main() -> None:
    env = build_environment(device_name="10.212.43.61:5555", grpc_port=8554)
    task = ContactsAddContact(ContactsAddContact.generate_random_params())
    task.initialize_task(env)
    env.reset(go_home=True)
    print(f"GOAL={task.goal}")
    execute_androidworld_action(
        Action(kind=ActionKind.LAUNCH_APP, package="com.google.android.contacts"),
        env.get_state(wait_to_stabilize=False),
        env.controller,
    )
    for element in env.get_state(wait_to_stabilize=True).ui_elements:
        print(
            "UI",
            repr(element.text), repr(element.content_description),
            element.bbox_pixels,
        )
    env.close()


if __name__ == "__main__":
    main()
