# Mobile Agent Runtime

## Goals and implemented feature boundary

This package is evolving from a benchmark action loop into a mobile-first
Mobile agent runtime. Android is its computer-use surface; the
runtime keeps benchmark ports compatible while adding durable, policy-governed
agent execution.

Implemented runtime capabilities:

- durable JSON session records with plans, observations, tool results,
  approvals, verification evidence, compaction summaries, and replay;
- versioned stable/context/volatile prompt assembly;
- provider-neutral multi-tool turns, with OpenAI Chat Completions and Responses
  adapters;
- grounded Android tools plus explicit observation and completion claims;
- web search/read tool hooks and untrusted-content sanitization;
- workspace search/read/write/patch/controlled-command tools, scoped to a root;
- one authority broker for consequential device actions, writes, and commands;
- staged, curated success/failure memory and reusable markdown skills;
- verifier-gated completion and structured replan events.

The durable runtime additionally keeps append-only SHA-256 hash-chained journals
and indexed session search; content-addressed Android evidence and snapshots are
SHA-256 verified before replay; workspace writes are atomic and return unified diffs; successful
memory promotion is idempotent; and read-only tool calls can run in parallel.

First-milestone non-goals remain browser login/posting, desktop control, cron,
social integrations, media generation, and autonomous subagents.

## Runtime quick start

Construct a `ToolBroker` with a `DevicePort` and workspace root, pair it with a
runtime model and verifier, then run a started session. A client must call
`runtime.resume(session_id, approve=True|False)` after an approval pause.

```python
runtime = MobileAgentRuntime(model, broker, verifier, SessionStore("sessions"))
state = runtime.run(runtime.start("Complete the Android task", "."))
```

## Durable runtime command

`python -m mobile_harness` runs `MobileAgentRuntime`, not the legacy compatibility loop. Configure the non-secret model/device/runtime settings in [`mobile_harness/config.json`](mobile_harness/config.json), export `OPENAI_API_KEY`, then run:

```powershell
python -m mobile_harness "Open Settings and show Wi-Fi" --config path/to/harness-config.json
```

It writes durable sessions beneath `runtime.session_root`. Resume one with `--resume <session-id>`; resolve a paused consequential action explicitly with `--resume <session-id> --approve` or `--deny`. The former loop remains available only as `mobile-harness-legacy` for compatibility.

`authority.auto_approve_mobile_actions` controls whether the runtime pauses for approval before consequential mobile UI actions. It is enabled in the current config. It does not authorize workspace writes, commands, or external integrations.

### Isolated non-GUI delegation

Delegation is intentionally restricted to research and read-only workspace
tasks. `JsonSubprocessWorker` is the production-oriented option: configure its
command in trusted host configuration, never from model output. The child gets
one JSON object containing only the `DelegatedTask` contract (goal, read-only
allowlist, deadline), returns one JSON `DelegatedResult`, and is terminated on
deadline. It does not receive a device port, workspace writer, command tool,
approval broker, or parent-session memory.
Its stdout/stderr are captured outside the parent heap and bounded (256 KiB by
default); accepted summaries and evidence are bounded as well.

```python
from mobile_harness import DelegationManager, JsonSubprocessWorker

worker = JsonSubprocessWorker(["python", "trusted_readonly_worker.py"])
delegation = DelegationManager(worker, store_root="sessions/delegations")
```

The trusted worker must emit exactly one JSON object such as
`{"id":"research-1","ok":true,"summary":"...","evidence":[...]}` to stdout.

`mobile_harness` is a model-agnostic control layer for Android agents.

It retains useful harness ideas—stable tool contracts, policy
gates, traces, and outcome verification—without importing its login,
messaging, terminal, browser, cron, or general desktop-agent subsystems.

## Control loop

```text
task + first observation
  -> policy-constrained agent decision
  -> one Android action (ADB or benchmark port)
  -> fresh screenshot + accessibility/UI observation
  -> agent proposes success
  -> independent verifier checks device/task state
  -> success | verifier feedback and repair
```

The agent never receives a shell tool. It receives `tap`, `swipe`,
`type_text`, `key`, `back`, `home`, `launch_app`, and `wait`. A `tap` may use
an accessibility locator; pixel coordinates are a normalized fallback.

## Ports

- `AdbDevice`: real phones and emulators through `adb`.
- `AndroidWorldAdapter`: bridges an AndroidWorld `AsyncEnv` and its executor.
- `MobileWorldAdapter` and `MemGUIAdapter`: bridge their environment-client
  APIs and canonical task scoring.
- `CallbackAdapter`: a thin integration point for other benchmarks.

The AndroidWorld, MobileWorld, and MemGUI-Bench source trees are checked in
locally. MemGUI's 40- and 128-task CSVs are included with its source; its
separate runtime image remains deferred because its snapshot layer is 24.7 GB
compressed.

## Smoke test

```powershell
python -m unittest discover -s mobile_harness/tests -v
```

Run it from the `mobile_harness` directory (or install the package first).

For editable installation:

```powershell
python -m pip install -e .
```

No model SDK is bundled. Connect an LLM by implementing `Agent.decide`; this
keeps credentials and prompt formatting outside the device runtime.

## Legacy compatibility loop

Live-run defaults are centralized in [`mobile_harness/config.json`](mobile_harness/config.json): model name/base URL, ADB serial and timeout, trace path, step limit, and mobile policy settings. Keep API keys out of this file; provide `OPENAI_API_KEY` through the environment. Copy it to a run-specific JSON file, set the non-secret values, and pass it with `--config`. Command-line flags override the corresponding config value for one run.

The agent-facing coordinate space is a fixed logical 0..1000 by 0..1000 canvas: all `tap`, `swipe`, `drag`, and `zoom` coordinates are integers in that space, regardless of device resolution or crop size. The runtime converts them to its normalized internal representation before ADB dispatch. Following Hermes Computer Use, `tap(element=N)` using the current observation's 1-based element index is preferred to raw coordinates; coordinates are only fallback. For dense, ambiguous, or previously missed coordinate targets, `zoom` crops an aspect-preserving view without sending a device gesture; the next action is grounded in that crop and maps back to the physical screen. `harness.zoom_ratio` controls the default crop size (0.5 means half the screen width and height); an action may supply its own ratio in `(0, 1]`.

The durable runtime follows the Hermes Computer Use evidence rule: ADB acceptance is only dispatch success, never proof that the UI changed as intended. With `runtime.capture_after_actions: true`, it immediately captures the post-action Android state, stores before/after evidence, and gives the next model turn a structured `action_outcome`: `unverifiable` requires inspecting the new state, while `suspected_noop` supplies a recovery recommendation such as `zoom` for a missed coordinate gesture.

`runtime.require_plan_before_actions` is enabled by default. It requires an independent `plan` or `todo` turn before Android mutation, assigns each plan update a durable revision, and appends `plan_updated` records to the session journal. This follows Hermes’s task-list model: small ordered steps, one active step, and completion only when its evidence is recorded.

`runtime.context_window_tokens` is set to `200000`, matching the available model context. The runtime does not compact merely because a session has accumulated a fixed number of events; it preserves the complete plan/outcome journal until it approaches this provider context window. `runtime.memory_root` and `runtime.skills_root` select the persistent cross-session fact and procedure stores used by the production entrypoint.

The Chat Completions runtime uses Hermes-style SSE streaming by default. Its
`model_request_started`, `model_first_delta`, `model_request_completed`, and
`model_stalled` journal events distinguish slow prefill/decoding from a server
that has stopped producing output. `runtime.model_stale_timeout_seconds` is the
idle-event watchdog (150 seconds by default), not a model token limit; active
stream events keep a long decode alive. Set `runtime.model_streaming` to false
only for an endpoint that does not support OpenAI-compatible SSE.
`runtime.model_stream_progress_interval_seconds` adds periodic non-content
diagnostics—SSE event count, event kinds, and generated content/reasoning/tool
argument character totals—without persisting raw hidden reasoning.

The former lightweight ADB loop does not use `MobileAgentRuntime` or [`prompts.py`](mobile_harness/prompts.py). It remains only for compatibility with callers that implement `OpenAICompatibleAgent` behavior directly. With Android Platform Tools available on `PATH`, run:

```powershell
$env:OPENAI_API_KEY = "..."
mobile-harness-legacy "Open Settings and show Wi-Fi" --config path/to/harness-config.json
```

The agent has only the typed mobile tools. When it calls `claim_done`, the
runtime asks the operator to verify the actual device outcome; it will not
turn a model assertion into a successful run. Benchmark adapters should use
the benchmark's authoritative success callback instead.

## Benchmark invocation

Use lifecycle helpers so task initialization and teardown remain owned by the
benchmark while terminal claims continue to be independently scored:

```python
from mobile_harness.runners import run_mobileworld_task

result = run_mobileworld_task(env, task_name, agent, max_steps=15)
```

This calls MobileWorld's real task evaluator each time the model claims done,
so a failed completion claim remains repairable within the harness loop.

AndroidWorld follows the same pattern after its `AsyncEnv` and concrete task
are created with the benchmark's launcher and registry:

```python
from mobile_harness.runners import run_androidworld_task

result = run_androidworld_task(env, task, agent)
```

MobileWorld requires its Docker/KVM environment; AndroidWorld requires the
documented API-33 `AndroidWorldAvd` plus its app setup. Those environment
requirements are intentionally not replaced by the generic ADB emulator used
for the smoke test.

See [`BENCHMARK_READINESS.md`](BENCHMARK_READINESS.md) for the current
evidence, host requirements, and exact next validation commands.

### Verified AndroidWorld smoke run

The checked-in `scripts/run_androidworld_contacts_harness.py` initializes a
real randomized `ContactsAddContact` task, drives only the typed portable
actions through `run_androidworld_task`, and accepts completion only when
`task.is_successful(env)` returns 1.0. It is a deterministic integration smoke
agent, not the default policy for production use.

`MemGUIAdapter` and `MemGUIVerifier` use MemGUI-Bench's published
MobileWorld-compatible `src/mobile_world` runtime surface: screenshots,
`JSONAction` execution, and task scoring. Pair them with
`VerifiedExperienceStore` when evaluating persistence across tasks/sessions.

For the harness-owned model evaluation launcher, its dependency boundary, and
staged run commands, see [ANDROIDWORLD_EVALUATION.md](ANDROIDWORLD_EVALUATION.md).
