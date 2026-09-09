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

## Real ADB device run

With Android Platform Tools available on `PATH`, run:

```powershell
$env:OPENAI_API_KEY = "..."
python -m mobile_harness "Open Settings and show Wi-Fi" --model <model> --base-url <endpoint> --serial <adb-serial>
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
