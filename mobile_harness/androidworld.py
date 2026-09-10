"""AndroidWorld environment boundary owned by :mod:`mobile_harness`.

AndroidWorld itself remains an optional benchmark dependency.  This module is
the only place where its launcher, direct remote-emulator transport, and
accessibility-forwarder compatibility code are adapted for this harness.  It
does not import or require DGM or any Mobile-Agent checkout.
"""
from __future__ import annotations

import subprocess
from typing import Any


def build_environment(
    *,
    console_port: int = 5554,
    grpc_port: int = 8554,
    adb_path: str = "adb",
    device_name: str | None = None,
    a11y_method: str = "forwarder",
) -> Any:
    """Create an AndroidWorld ``AsyncAndroidEnv`` for a local or remote AVD.

    Supplying ``device_name`` selects an already-running ADB device.  In that
    case AndroidWorld's local emulator launch is disabled and its gRPC channel
    is directed to the ADB host.  The benchmark package is deliberately
    imported here, so normal ``mobile_harness`` use needs no AndroidWorld
    installation.
    """
    from android_env import loader
    from android_env.components import config_classes
    from android_env.components.simulators.emulator import emulator_launcher, emulator_simulator
    from android_env.wrappers import a11y_grpc_wrapper
    from android_world.env import android_world_controller, interface

    if device_name:
        _connect_adb(adb_path, device_name)
        _configure_remote_device(
            device_name=device_name,
            adb_path=adb_path,
            emulator_simulator=emulator_simulator,
            emulator_launcher=emulator_launcher,
            a11y_grpc_wrapper=a11y_grpc_wrapper,
        )

    config = config_classes.AndroidEnvConfig(
        task=config_classes.FilesystemTaskConfig(
            path=android_world_controller._write_default_task_proto(),
        ),
        simulator=config_classes.EmulatorConfig(
            emulator_launcher=config_classes.EmulatorLauncherConfig(
                emulator_console_port=console_port,
                adb_port=5555,
                grpc_port=grpc_port,
            ),
            adb_controller=config_classes.AdbControllerConfig(
                adb_path=adb_path,
                device_name=device_name or "",
            ),
        ),
    )
    environment = loader.load(config)
    method = (
        android_world_controller.A11yMethod.UIAUTOMATOR
        if a11y_method == "uiautomator"
        else android_world_controller.A11yMethod.A11Y_FORWARDER_APP
    )
    return interface.AsyncAndroidEnv(
        android_world_controller.AndroidWorldController(
            environment,
            a11y_method=method,
            install_a11y_forwarding_app=False,
        )
    )


def _connect_adb(adb_path: str, device_name: str) -> None:
    """Ensure the AndroidEnv-owned ADB server sees a remote serial."""
    if ":" not in device_name:
        return
    result = subprocess.run(
        [adb_path, "connect", device_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not connect ADB device {device_name}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    state = subprocess.run(
        [adb_path, "-s", device_name, "get-state"],
        check=False,
        capture_output=True,
        text=True,
    )
    if state.returncode != 0 or state.stdout.strip() != "device":
        raise RuntimeError(
            f"ADB device {device_name} is not ready: "
            f"{(state.stderr or state.stdout).strip()}"
        )


def _configure_remote_device(*, device_name: str, adb_path: str, emulator_simulator: Any,
                             emulator_launcher: Any, a11y_grpc_wrapper: Any) -> None:
    """Adapt AndroidWorld's local-only launcher to one explicitly selected AVD."""
    target_host = device_name.rsplit(":", 1)[0]
    emulator_simulator.EmulatorSimulator.adb_device_name = lambda self: device_name
    emulator_launcher.EmulatorLauncher.launch_emulator_process = lambda self: None
    emulator_launcher.EmulatorLauncher.confirm_shutdown = lambda self: None

    def connect_to_remote(self: Any, grpc_port: int, timeout_sec: int = 100) -> tuple[Any, Any]:
        import grpc
        from android_env.proto import emulator_controller_pb2_grpc, snapshot_service_pb2_grpc

        address = f"{target_host}:{grpc_port}"
        try:
            self._channel = grpc.insecure_channel(address, options=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
            ])
            grpc.channel_ready_future(self._channel).result(timeout=timeout_sec)
        except (grpc.RpcError, grpc.FutureTimeoutError) as exc:
            raise emulator_simulator.EmulatorBootError(
                f"Could not connect to AndroidWorld emulator at {address}"
            ) from exc
        return (
            emulator_controller_pb2_grpc.EmulatorControllerStub(self._channel),
            snapshot_service_pb2_grpc.SnapshotServiceStub(self._channel),
        )

    emulator_simulator.EmulatorSimulator._connect_to_emulator = connect_to_remote
    original_init = a11y_grpc_wrapper.A11yGrpcWrapper.__init__

    def configure_forwarder(wrapper: Any, *args: Any, **kwargs: Any) -> None:
        original_init(wrapper, *args, **kwargs)
        port = getattr(wrapper, "_port", None)
        if port is None:
            return
        subprocess.run([adb_path, "-s", device_name, "reverse", f"tcp:{port}", f"tcp:{port}"], check=False)
        subprocess.run([
            adb_path, "-s", device_name, "shell", "am", "broadcast",
            "-a", "accessibility_forwarder.intent.action.SET_GRPC",
            "-n", "com.google.androidenv.accessibilityforwarder/.FlagsBroadcastReceiver",
            "--es", "host", "127.0.0.1", "--ei", "port", str(port),
        ], check=False)

    a11y_grpc_wrapper.A11yGrpcWrapper.__init__ = configure_forwarder
