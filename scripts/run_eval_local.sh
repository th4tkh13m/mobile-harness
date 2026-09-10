#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Optional checkout of the official AndroidWorld benchmark package. This is
# intentionally separate from the harness and is never inferred from a DGM or
# Mobile-Agent path.
if [[ -n "${ANDROIDWORLD_SOURCE:-}" ]]; then
  export PYTHONPATH="$ANDROIDWORLD_SOURCE:$PYTHONPATH"
fi
# Point this at a Python environment where the official AndroidWorld package
# is installed.  No DGM or Mobile-Agent checkout is used by this launcher.
PYTHON_BIN="${ANDROIDWORLD_PYTHON:-python3}"

# 1. Connect local ADB to remote emulator
ANDROIDWORLD_DEVICE="${ANDROIDWORLD_DEVICE:-10.212.43.61:5555}"
ANDROIDWORLD_GRPC_PORT="${ANDROIDWORLD_GRPC_PORT:-8554}"
ANDROIDWORLD_BASE_URL="${ANDROIDWORLD_BASE_URL:-http://127.0.0.1:8000/v1}"
ANDROIDWORLD_MODEL="${ANDROIDWORLD_MODEL:-Qwen/Qwen3.8-27B}"
adb connect "$ANDROIDWORLD_DEVICE" || true

# 2. Direct gRPC connection to target IP 10.212.43.61:8554 (no SSH tunnel needed)

# 3. Execute a bounded suite by default. Request `ANDROIDWORLD_SUITE=full`
# only after the image has all required app snapshots and a successful smoke.
ANDROIDWORLD_SUITE="${ANDROIDWORLD_SUITE:-smoke}"
TASK_SELECTION_ARGS=(--suite "$ANDROIDWORLD_SUITE")
if [[ -n "${ANDROIDWORLD_TASK_FILE:-}" ]]; then
  TASK_SELECTION_ARGS=(--task-file "$ANDROIDWORLD_TASK_FILE" --subset-split "${ANDROIDWORLD_SUBSET_SPLIT:-evaluation}")
fi
exec "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_androidworld.py" \
  --runtime core \
  "${TASK_SELECTION_ARGS[@]}" \
  --base-url "$ANDROIDWORLD_BASE_URL" \
  --model "$ANDROIDWORLD_MODEL" \
  --device-name "$ANDROIDWORLD_DEVICE" \
  --grpc-port "$ANDROIDWORLD_GRPC_PORT" "$@"
