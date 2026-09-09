# Benchmark readiness

This is a verification record, not a claim that all benchmarks have run.

| Target | Harness boundary | Current evidence | Remaining requirement |
| --- | --- | --- | --- |
| Real Android device/emulator | `AdbDevice` | Live ADB smoke test passed: screenshot, UI dump, focused activity, and HOME action. | A user-selected real task and model endpoint. |
| AndroidWorld | `AndroidWorldAdapter` + `run_androidworld_task` | **Live verified.** A Pixel 6/API-33 AVD ran `ContactsAddContact` through the harness in 10 events; the canonical validator returned `AndroidWorld score=1.0`. | Expand from the Contacts smoke flow to the full app setup/task suite. |
| MobileWorld | `MobileWorldAdapter` + `run_mobileworld_task` | **Live transport verified.** The official `ghcr.io/tongyi-mai/mobile_world:latest` image launched one KVM-backed Android environment (201 tasks registered); `MobileWorldAdapter` observed it, executed portable `HOME`, and re-observed a 21,708-byte screenshot. | Run a benchmark task with a configured model endpoint and preserve its canonical task score. |
| MemGUI-Bench | `MemGUIAdapter` + `run_memgui_task` | Official source plus 40- and 128-task CSVs are checked out; the adapter follows its published MobileWorld-compatible client/action/score surface and contract tests pass. | **Deferred by user.** Its official image has a 24.7 GB compressed snapshot layer and was not live-run on this workstation. |

## AndroidWorld native validation

```powershell
# Launch the required API-33 AVD, not an arbitrary newer system image.
emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554

# Use the UIAutomator fallback when the emulator's gRPC accessibility-forwarder
# cannot reset reliably, then run the checked-in end-to-end smoke script.
PYTHONPATH=<workspace>/mobile_harness \
  <android-world>/.venv311/Scripts/python.exe \
  <workspace>/mobile_harness/scripts/run_androidworld_contacts_harness.py
```

Observed result: `HARNESS_SUCCESS=True`, `AndroidWorld score=1.0`.

## MobileWorld/MemGUI validation

Run the official preflight from Linux or WSL2 with KVM enabled:

```bash
uv run mw env check
uv run mw env run --dev
```

### Latest verified state (2026-08-13)

Docker Desktop integration for Ubuntu is active, and the Ubuntu user has
`kvm` access. The official MobileWorld preflight passed Docker and KVM checks;
its WSL host NAT-module heuristic still reports unavailable, but the official
image launched successfully with its documented in-container fallback.

Once the environment client is available, invoke `run_mobileworld_task` or
`run_memgui_task` with a deliberately bounded, non-MCP task and preserve the
trace plus the benchmark's authoritative score.

### MobileWorld live transport evidence

The official image was pulled and launched with one development container on
this workstation after Docker Desktop WSL integration and Ubuntu `kvm` access
were enabled. Its own readiness check passed after 346 seconds. The checked-in
`scripts/smoke_mobileworld_live.py` then ran *inside that container* against
the native backend (`127.0.0.1:6800`), deliberately without model or MCP
credentials. It completed:

```text
HARNESS_SUCCESS=True
HARNESS_REASON=MobileWorld transport smoke: screenshot_bytes=21708
HARNESS_STEPS=2
```

This proves the harness-to-live-runtime observation/action boundary; it is not
an evaluation result and must not be represented as a task score.
