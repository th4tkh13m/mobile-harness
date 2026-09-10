"""AndroidWorld benchmark evaluation script for mobile-harness.

Runs locally on this host, connecting to:
- Local model endpoint (http://127.0.0.1:8000/v1 with Qwen/Qwen3.8-27B)
- Remote AndroidWorld emulator on 10.212.43.61 (via ADB device_name and gRPC port forwarding)
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Ensure mobile_harness is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Preset benchmark suites
SUITES: dict[str, list[str]] = {
    "smoke": [
        "ContactsAddContact",
    ],
    "development": [
        "ClockStopWatchRunning",
        "ContactsAddContact",
        "SimpleCalendarAddOneEvent",
        "SystemBluetoothTurnOnVerify",
        "SystemWifiTurnOffVerify",
        "MarkorCreateNote",
        "SimpleSmsSend",
        "OpenAppTaskEval",
    ],
    "heldout": [
        "ClockTimerEntry",
        "ContactsNewContactDraft",
        "SimpleCalendarAddOneEventTomorrow",
        "SystemBluetoothTurnOffVerify",
        "SystemWifiTurnOnVerify",
        "MarkorEditNote",
        "SimpleSmsReply",
        "CameraTakePhoto",
    ],
}


@contextlib.contextmanager
def task_deadline(seconds: float):
    """Bound one task on POSIX hosts without changing the default behavior."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def expired(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"task exceeded {seconds:g}s deadline")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def resolve_task_list(
    suite_name: str | None,
    task_names: list[str] | None,
    all_registry: dict[str, Any],
    task_file: str | None = None,
    subset_split: str = "evaluation",
) -> list[str]:
    """Resolve the list of task class names to execute."""
    if task_file:
        payload = json.loads(Path(task_file).read_text(encoding="utf-8"))
        selected = payload.get(subset_split)
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"Task file {task_file} has no non-empty '{subset_split}' list")
        task_names = [str(name) for name in selected]
    if task_names:
        selected = []
        for name in task_names:
            if name not in all_registry:
                raise ValueError(f"Task '{name}' not found in AndroidWorld registry. Available: {list(all_registry.keys())[:10]}...")
            selected.append(name)
        return selected

    if suite_name:
        if suite_name == "full":
            return sorted(list(all_registry.keys()))
        if suite_name in SUITES:
            return SUITES[suite_name]
        raise ValueError(f"Unknown suite '{suite_name}'. Choose from: smoke, development, heldout, full")

    return SUITES["smoke"]


def run_task_with_core_runtime(
    env: Any,
    task_eval: Any,
    args: argparse.Namespace,
    session_dir: Path,
) -> tuple[bool, float, int, str, list[dict[str, Any]]]:
    """Run one AndroidWorld task using the production MobileAgentRuntime."""
    from mobile_harness.benchmarks import (
        AndroidWorldAdapter,
        AndroidWorldVerifier,
        execute_androidworld_action,
    )
    from mobile_harness.memory import CuratedMemoryStore, SkillStore
    from mobile_harness.policy import AuthorityBroker
    from mobile_harness.prompts import PromptAssembler
    from mobile_harness.providers import OpenAIChatRuntimeModel
    from mobile_harness.runtime import MobileAgentRuntime, SessionStore
    from mobile_harness.tools import ToolBroker

    # 1. Task initialization
    task_eval.initialize_task(env)
    goal = str(getattr(task_eval, "goal", "Complete task"))

    # 2. Build device adapter, verifier, & workspace
    device = AndroidWorldAdapter(env, action_executor=execute_androidworld_action)
    verifier = AndroidWorldVerifier(task_eval, env)
    workspace = session_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    memory_root = workspace / "memory"
    skills_root = workspace / "skills"

    # 3. Build ToolBroker & AuthorityBroker
    authority = AuthorityBroker(approve=lambda category, _: category.startswith("mobile_"))
    store = SessionStore(session_dir / "store")
    broker = ToolBroker(
        device,
        workspace,
        authority=authority,
        zoom_ratio=0.5,
        memory=CuratedMemoryStore(memory_root),
        skills=SkillStore(skills_root),
        session_search=lambda query, limit: store.search(query, limit=limit),
        session_trace=lambda sid, seq, before, after: store.history.trace(sid, seq, before=before, after=after),
        session_history=store.history,
    )

    # 4. Build Model & MobileAgentRuntime
    model = OpenAIChatRuntimeModel(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        streaming=args.streaming,
        stale_timeout_seconds=args.model_timeout,
    )
    runtime = MobileAgentRuntime(
        model,
        broker,
        verifier,
        store,
        prompts=PromptAssembler(max_context_tokens=200000),
        max_turns=args.max_steps,
        capture_after_actions=True,
        require_plan_before_mutation=True,
    )

    # 5. Execute Session
    start_session = runtime.start(goal, workspace)
    session_state = runtime.run(start_session)

    # 6. Evaluation Score
    score = float(task_eval.is_successful(env))
    # AndroidWorld's evaluator is authoritative; a runtime status is only
    # diagnostic evidence and must never turn an evaluator error into a pass.
    is_success = score >= 1.0
    steps = len(session_state.events)
    events_summary = [
        {"sequence": ev.sequence, "kind": ev.kind, "at": ev.at, "payload": ev.payload}
        for ev in session_state.events
    ]
    return is_success, score, steps, session_state.status, events_summary


def run_task_with_legacy_harness(
    env: Any,
    task_eval: Any,
    args: argparse.Namespace,
) -> tuple[bool, float, int, str, list[dict[str, Any]]]:
    """Run one AndroidWorld task using the legacy Harness compatibility loop."""
    from mobile_harness.openai_agent import OpenAICompatibleAgent
    from mobile_harness.runners import run_androidworld_task

    agent = OpenAICompatibleAgent(base_url=args.base_url, api_key=args.api_key, model=args.model)
    result = run_androidworld_task(env, task_eval, agent, max_steps=args.max_steps)

    score = float(task_eval.is_successful(env))

    events_summary = [
        {
            "decision": event.decision.__dict__ if hasattr(event.decision, "__dict__") else str(event.decision),
            "policy_message": event.policy_message,
            "result": event.result.__dict__ if hasattr(event.result, "__dict__") else str(event.result),
        }
        for event in result.events
    ]
    return score >= 1.0, score, len(result.events), result.reason, events_summary


def run_evaluation(args: argparse.Namespace) -> int:
    from android_world import registry
    from mobile_harness.androidworld import build_environment

    random.seed(args.seed)

    # 1. Load Task Registry
    task_reg_instance = registry.TaskRegistry()
    full_registry = task_reg_instance.get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
    target_tasks = resolve_task_list(
        args.suite, args.tasks, full_registry,
        task_file=args.task_file, subset_split=args.subset_split,
    )

    # 2. Setup Output Directory
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / f"eval_{args.runtime}_{args.suite or 'custom'}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)

    print(f"==================================================")
    print(f" AndroidWorld Local Host Evaluation ({args.runtime.upper()} RUNTIME)")
    print(f" Tasks count   : {len(target_tasks)}")
    print(f" Model         : {args.model}")
    print(f" Seed          : {args.seed}")
    print(f" Base URL      : {args.base_url}")
    print(f" Device Serial : {args.device_name}")
    print(f" Output Dir    : {output_dir}")
    print(f"==================================================")

    # 3. Create Environment
    env = build_environment(
        console_port=args.console_port,
        grpc_port=args.grpc_port,
        adb_path=args.adb_path,
        device_name=args.device_name,
        a11y_method=args.a11y_method,
    )

    results: list[dict[str, Any]] = []
    successes = 0

    try:
        for idx, task_name in enumerate(target_tasks, start=1):
            print(f"\n[{idx}/{len(target_tasks)}] Executing task: {task_name}", flush=True)
            task_class = full_registry[task_name]

            # Ensure ADB TCP connection to remote emulator is active before initializing task
            if args.device_name and ":" in args.device_name:
                import subprocess
                subprocess.run([args.adb_path, "connect", args.device_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Generate task instance with random parameters
            try:
                task_params = task_class.generate_random_params()
                task_eval = task_class(task_params)
            except Exception as exc:
                print(f"ERROR generating task params for {task_name}: {exc}", flush=True)
                results.append({
                    "task_name": task_name,
                    "success": False,
                    "score": 0.0,
                    "steps": 0,
                    "reason": f"Params generation error: {exc}",
                    "duration_sec": 0.0,
                })
                continue

            goal = str(getattr(task_eval, "goal", task_name))
            print(f"  Goal: {goal}", flush=True)

            start_time = time.time()
            task_session_dir = traces_dir / f"{idx:03d}_{task_name}"
            events_summary: list[dict[str, Any]] = []

            try:
                with task_deadline(args.task_timeout):
                    if args.runtime == "core":
                        is_success, score, steps, status, events_summary = run_task_with_core_runtime(
                            env, task_eval, args, task_session_dir
                        )
                    else:
                        is_success, score, steps, status, events_summary = run_task_with_legacy_harness(
                            env, task_eval, args
                        )

                duration = time.time() - start_time
                print(f"  Outcome: {'SUCCESS' if is_success else 'FAILED'} (score={score}, steps={steps}, status={status}, duration={duration:.1f}s)", flush=True)

            except Exception as exc:
                duration = time.time() - start_time
                print(f"  ERROR running task {task_name}: {exc}", flush=True)
                is_success = False
                score = 0.0
                steps = 0
                status = f"Execution error: {exc}"

            finally:
                # Always tear down task state
                try:
                    task_eval.tear_down(env)
                except Exception as exc:
                    print(f"  Warning: tear_down failed for {task_name}: {exc}", flush=True)

            # Preserve infrastructure and evaluator failures as per-task
            # evidence; otherwise they look like ordinary 0-score failures.
            trace_file = traces_dir / f"{idx:03d}_{task_name}.json"
            trace_file.write_text(json.dumps({
                "task_name": task_name,
                "goal": goal,
                "runtime": args.runtime,
                "success": is_success,
                "score": score,
                "steps": steps,
                "duration_sec": duration,
                "status": status,
                "events": events_summary,
            }, indent=2, default=str), encoding="utf-8")

            if is_success:
                successes += 1

            results.append({
                "task_name": task_name,
                "goal": goal,
                "success": is_success,
                "score": score,
                "steps": steps,
                "duration_sec": duration,
                "status": status,
            })

    finally:
        try:
            env.close()
        except Exception:
            pass

    # 4. Summary & Metrics Report
    total = len(results)
    success_rate = (successes / total * 100) if total > 0 else 0.0
    avg_steps = (sum(r["steps"] for r in results) / total) if total > 0 else 0.0
    total_duration = sum(r["duration_sec"] for r in results)

    summary = {
        "timestamp": timestamp,
        "runtime": args.runtime,
        "seed": args.seed,
        "suite": args.suite,
        "model": args.model,
        "base_url": args.base_url,
        "device_name": args.device_name,
        "total_tasks": total,
        "successful_tasks": successes,
        "failed_tasks": total - successes,
        "success_rate_pct": round(success_rate, 2),
        "average_steps": round(avg_steps, 2),
        "total_duration_sec": round(total_duration, 2),
        "task_results": results,
        "command": sys.argv,
    }

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n==================================================")
    print(f" EVALUATION SUMMARY ({args.runtime.upper()} RUNTIME)")
    print("==================================================")
    print(f" Total Tasks     : {total}")
    print(f" Successful      : {successes}")
    print(f" Failed          : {total - successes}")
    print(f" Success Rate    : {success_rate:.1f}%")
    print(f" Avg Steps/Task  : {avg_steps:.1f}")
    print(f" Total Duration  : {total_duration:.1f}s")
    print(f" Summary Written : {summary_file}")
    print("==================================================\n")

    return 0 if successes == total else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mobile-harness on AndroidWorld.")
    parser.add_argument("--runtime", choices=["core", "harness"], default="core", help="Runtime mode: 'core' (durable MobileAgentRuntime with 5-tier PromptAssembler) or 'harness' (legacy compatibility loop).")
    parser.add_argument("--suite", choices=["smoke", "development", "heldout", "full"], help="Preset suite of tasks to run.")
    parser.add_argument("--tasks", nargs="+", help="Explicit list of AndroidWorld task names.")
    parser.add_argument("--task-file", help="JSON subset manifest containing named split lists.")
    parser.add_argument("--subset-split", choices=["screen", "selection", "evaluation"], default="evaluation", help="Split to load from --task-file.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible model API base URL.")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B", help="Model name.")
    parser.add_argument("--api-key", default="EMPTY", help="API key.")
    parser.add_argument("--max-steps", type=int, default=15, help="Max turns per task.")
    parser.add_argument("--task-timeout", type=float, default=0, help="Optional per-task wall-clock deadline in seconds; 0 disables it.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for task parameter generation.")
    parser.add_argument("--output-dir", default="runs/androidworld_eval", help="Directory to write output traces and summary.")
    parser.add_argument("--console-port", type=int, default=5554, help="Emulator console port.")
    parser.add_argument("--grpc-port", type=int, default=8554, help="Emulator gRPC port.")
    parser.add_argument("--adb-path", default="adb", help="ADB binary path.")
    parser.add_argument("--device-name", default=None, help="Existing ADB serial; omit to let AndroidWorld launch its local AVD.")
    parser.add_argument("--a11y-method", choices=["forwarder", "uiautomator"], default="forwarder", help="Method to extract UI tree: 'forwarder' (gRPC AccessibilityForwarder app) or 'uiautomator' (ADB UiAutomator dump fallback).")
    parser.add_argument("--model-timeout", type=float, default=300.0, help="Model endpoint response timeout in seconds (default: 300s).")
    parser.add_argument("--streaming", action="store_true", help="Enable SSE streaming mode for model responses.")

    args = parser.parse_args()
    if not args.suite and not args.tasks:
        args.suite = "smoke"

    sys.exit(run_evaluation(args))


if __name__ == "__main__":
    main()
