"""Initialize one AndroidWorld Contacts task and print its visible UI elements."""
from __future__ import annotations

from android_world.env import android_world_controller, interface
from android_world.agents.new_json_action import JSONAction
from android_world.task_evals.single.contacts import ContactsAddContact


ADB_PATH = r"C:\Users\Khiem\AppData\Local\Android\Sdk\platform-tools\adb.exe"


def main() -> None:
    controller = android_world_controller.get_controller(
        console_port=5554,
        adb_path=ADB_PATH,
        a11y_method=android_world_controller.A11yMethod.UIAUTOMATOR,
        install_a11y_forwarding_app=False,
    )
    env = interface.AsyncAndroidEnv(controller)
    task = ContactsAddContact(ContactsAddContact.generate_random_params())
    task.initialize_task(env)
    env.reset(go_home=True)
    print(f"GOAL={task.goal}")
    env.execute_action(JSONAction(action_type="open_app", app_name="contacts"))
    for element in env.get_state(wait_to_stabilize=True).ui_elements:
        print(
            "UI",
            repr(element.text), repr(element.content_description),
            element.bbox_pixels,
        )
    env.close()


if __name__ == "__main__":
    main()
