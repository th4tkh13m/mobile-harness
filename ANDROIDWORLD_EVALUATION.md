# AndroidWorld evaluation

The AndroidWorld integration is owned by this repository. It uses the official
AndroidWorld package only as the benchmark environment and evaluator; it does
not import a DGM or Mobile-Agent script, prompt, or runner.

## Requirements

Create or select a Python environment that has the official AndroidWorld and
AndroidEnv dependencies installed, then set `ANDROIDWORLD_PYTHON` to that
interpreter. The harness itself remains available with only its normal
dependencies because AndroidWorld imports are lazy.

If AndroidWorld is used directly from a source checkout rather than installed
into that interpreter, set `ANDROIDWORLD_SOURCE` to the official checkout.

For a remote, already-running AVD, set `ANDROIDWORLD_DEVICE` to its ADB serial.
The selected device's host must also expose the emulator gRPC port used by
`--grpc-port` (8554 by default). The shared
[`mobile_harness/androidworld.py`](mobile_harness/androidworld.py) boundary
disables AndroidWorld's local launcher, targets that host for gRPC, and
configures the accessibility forwarder over ADB reverse.

## Run

From the repository root:

```bash
ANDROIDWORLD_PYTHON=/path/to/androidworld-python \
ANDROIDWORLD_SOURCE=/path/to/official/android_world \
ANDROIDWORLD_DEVICE=host.example:5555 \
ANDROIDWORLD_BASE_URL=http://127.0.0.1:8000/v1 \
ANDROIDWORLD_MODEL=your-model \
scripts/run_eval_local.sh --seed 42 --max-steps 15
```

The launcher evaluates the `smoke` suite by default. Use a named suite or
explicit task classes for staged validation:

```bash
ANDROIDWORLD_SUITE=development scripts/run_eval_local.sh --seed 42
scripts/run_eval_local.sh --tasks ContactsAddContact --seed 42
# Bound a diagnostic smoke on a host where a task may hang during setup.
scripts/run_eval_local.sh --tasks ContactsAddContact --seed 42 --task-timeout 25
```

Do not start the full suite until its image has all required app snapshots and
the transport survives a smoke run:

```bash
ANDROIDWORLD_SUITE=full scripts/run_eval_local.sh --seed 42
```

Each run writes a command, seed, summary, and per-task trace under
`runs/androidworld_eval/`. `task.is_successful(env)` is the only success
authority. Model/runtime status and successful action dispatch are diagnostic
evidence only; evaluator exceptions fail the task and are preserved in its
trace.

## Review of the generated evaluator

The generated launcher previously hard-coded both a DGM virtual environment
and a Mobile-Agent `PYTHONPATH`; both have been removed. Its recorded full run
was not a valid full-suite result: the remote image was missing app snapshots,
then the gRPC transport disconnected and reconnects went to local
`127.0.0.1:8554`. Treat that log as a transport/setup failure, not 116 model
failures. Confirm remote reconnect behavior in the selected AndroidWorld
environment before reporting any multi-task result.
