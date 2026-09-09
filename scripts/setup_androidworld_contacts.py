"""Prepare the smallest AndroidWorld app set needed for Contacts smoke tests.

Run this with AndroidWorld's Python environment after launching AndroidWorldAvd
with ``-grpc 8554``.  It intentionally avoids the full 24-app one-time setup.
"""
from __future__ import annotations

from android_world.env import android_world_controller, interface
from android_world.env.setup_device import apps, setup


ADB_PATH = r"C:\Users\Khiem\AppData\Local\Android\Sdk\platform-tools\adb.exe"


def main() -> None:
    print("ANDROIDWORLD_CONNECTING", flush=True)
    controller = android_world_controller.get_controller(
        console_port=5554,
        adb_path=ADB_PATH,
        a11y_method=android_world_controller.A11yMethod.UIAUTOMATOR,
        install_a11y_forwarding_app=False,
    )
    env = interface.AsyncAndroidEnv(controller)
    try:
        print(f"ANDROIDWORLD_CONNECTED screen={env.logical_screen_size}", flush=True)
        setup.setup_apps(env, app_list=(apps.AndroidWorldApp, apps.ContactsApp))
        print(f"ANDROIDWORLD_CONTACTS_SETUP_OK screen={env.logical_screen_size}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
