"""Production entrypoint for the durable, prompt-assembled mobile runtime."""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from .config import load_harness_config
from .memory import CuratedMemoryStore, SkillStore
from .ports import AdbDevice
from .policy import AuthorityBroker
from .prompts import PromptAssembler
from .providers import OpenAIChatRuntimeModel, OpenAIResponsesRuntimeModel
from .runtime import MobileAgentRuntime, SessionStore
from .tools import ToolBroker


class OperatorVerifier:
    def verify(self, task, observation, events):
        answer = input("The runtime claimed completion. Verify the real device outcome and type yes to accept: ").strip().lower()
        return answer in {"y", "yes"}, "operator verified device outcome" if answer in {"y", "yes"} else "operator did not verify completion"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the durable MobileAgentRuntime against an ADB device.")
    parser.add_argument("task", nargs="?")
    parser.add_argument("--config", help="Path to central JSON settings; defaults to bundled config.json")
    parser.add_argument("--model", help="Overrides model.name")
    parser.add_argument("--base-url", help="Overrides model.base_url")
    parser.add_argument("--serial", help="Overrides device.serial")
    parser.add_argument("--workspace-root", help="Overrides runtime.workspace_root")
    parser.add_argument("--session-root", help="Overrides runtime.session_root")
    parser.add_argument("--max-turns", type=int, help="Overrides runtime.max_turns")
    parser.add_argument("--resume", help="Resume an existing durable session ID")
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument("--approve", action="store_true", help="Approve the pending action when used with --resume")
    approval.add_argument("--deny", action="store_true", help="Deny the pending action when used with --resume")
    args = parser.parse_args()
    if bool(args.task) == bool(args.resume):
        parser.error("provide exactly one of task or --resume")
    if (args.approve or args.deny) and not args.resume:
        parser.error("--approve/--deny requires --resume")
    config = load_harness_config(args.config)
    model_name, base_url = args.model or config.model.name, args.base_url or config.model.base_url
    if not model_name or not base_url:
        parser.error("set model.name and model.base_url in --config, or pass --model and --base-url")
    api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("Model API key: ")
    workspace = Path(args.workspace_root or config.runtime.workspace_root).resolve()
    session_root = Path(args.session_root or config.runtime.session_root)
    if not session_root.is_absolute():
        session_root = workspace / session_root
    device = AdbDevice(serial=args.serial or config.device.serial, timeout_seconds=config.device.adb_timeout_seconds)
    model_type = OpenAIChatRuntimeModel if config.runtime.provider == "chat_completions" else OpenAIResponsesRuntimeModel
    authority = AuthorityBroker(approve=lambda category, _: config.authority.auto_approve_mobile_actions and category.startswith("mobile_"))
    store = SessionStore(session_root)
    memory_root, skills_root = Path(config.runtime.memory_root), Path(config.runtime.skills_root)
    if not memory_root.is_absolute(): memory_root = workspace / memory_root
    if not skills_root.is_absolute(): skills_root = workspace / skills_root
    broker = ToolBroker(device, workspace, authority=authority, zoom_ratio=config.harness.zoom_ratio,
                        memory=CuratedMemoryStore(memory_root), skills=SkillStore(skills_root),
                        session_search=lambda query, limit: store.search(query, limit=limit),
                        session_trace=lambda sid, seq, before, after: store.history.trace(sid, seq, before=before, after=after),
                        session_history=store.history)
    model_kwargs = {}
    if model_type is OpenAIChatRuntimeModel:
        model_kwargs = {"stale_timeout_seconds": config.runtime.model_stale_timeout_seconds, "streaming": config.runtime.model_streaming, "stream_progress_interval_seconds": config.runtime.model_stream_progress_interval_seconds}
    runtime = MobileAgentRuntime(model_type(base_url, api_key, model_name, **model_kwargs), broker, OperatorVerifier(), store, prompts=PromptAssembler(max_context_tokens=config.runtime.context_window_tokens, summary_timeout_seconds=config.runtime.compaction_summary_timeout_seconds), max_turns=args.max_turns or config.runtime.max_turns, capture_after_actions=config.runtime.capture_after_actions, require_plan_before_mutation=config.runtime.require_plan_before_actions, require_plan_update_on_transition=config.runtime.require_plan_update_on_transition)
    decision = True if args.approve else False if args.deny else None
    state = runtime.resume(args.resume, approve=decision) if args.resume else runtime.run(runtime.start(args.task, workspace))
    print(f"SESSION_ID={state.id}")
    print(f"STATUS={state.status}")
    return 0 if state.status == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
