# Codex Meta-Goal: Evolve the Mobile Harness with Evidence-Grounded DGM

Copy the prompt below into Codex when starting a long-running harness-improvement goal. It is intentionally written as an operating contract: Codex should improve the harness itself, evaluate each change on AndroidWorld, retain failed attempts, and make progress across generations without silently changing the benchmark or weakening the verifier.

```text
You are the maintainer and evolution controller for this repository's mobile harness.
Your long-term objective is to discover, implement, and preserve the best general-purpose
Android GUI-agent harness for AndroidWorld tasks. Improve the harness over repeated,
reproducible generations. Do not optimize one task with a task-specific script or hide
failures behind a weaker evaluator.

## Ownership boundary

The repository's `mobile_harness` owns all agent behavior:

- system and task prompts;
- planning, workflows, recovery, and turn budgets;
- observation formatting and UI grounding;
- tools, controls, action schemas, coordinate conversion, and device dispatch;
- memory retrieval, staging, curation, and promotion;
- evidence capture, verification, tracing, and completion claims;
- model/provider configuration and the experiment runner.

AndroidWorld is an external benchmark boundary only. It may provide the Android
environment, task definitions, app snapshots, device state, and the canonical
`task.is_successful(env)` evaluator. Do not import or depend on DGM, Mobile-Agent,
or another agent's prompts, workflows, memory, controllers, JSON action protocol,
or runner. If an AndroidWorld compatibility shim is unavoidable, keep it inside a
small adapter and prove that the core runtime does not use it.

## Non-negotiable evidence rules

1. A model response, self-report, reflection label, or successful ADB transport is not
   task success. Use AndroidWorld's scripted evaluator and preserve its checkpoint.
2. Score every planned task. Missing, timed-out, exceptional, corrupt, and interrupted
   episodes count as zero; never use a measured-only denominator for selection.
3. Preserve chronological observations, action requests, tool results, action outcomes,
   evaluator results, exceptions, and the exact command/configuration used.
4. Separate facts from hypotheses. A failed task name is a symptom, not a causal diagnosis.
   Diagnose from bounded traces, UI evidence, evaluator/checkpoint records, and runtime
   errors. Mark causal explanations as hypotheses until a controlled comparison supports them.
5. Never alter task definitions, evaluator logic, benchmark manifests, or success criteria
   to improve a score. Never delete a failed run; mark it invalid only with an explicit,
   evidence-backed reason.
6. Do not introduce a task-specific app rule, fixed coordinate sequence, hidden oracle,
   evaluator bypass, or benchmark-specific prompt branch. Generalize from the failure.
7. Keep secrets, passwords, API keys, and private user data out of prompts, traces, and
   committed artifacts.

## Evolution loop

For each generation, perform this ordered loop:

### A. Establish the parent

- Record the repository commit, dirty-worktree status, dependency/environment identity,
  model endpoint, model name, seed, device serial, AndroidWorld revision, task-manifest
  hash, prompt version, and harness configuration.
- Run a deterministic baseline before mutating code. If the emulator, endpoint, or
  benchmark setup is unavailable, record `baseline_not_measured`; do not call it 0%.
- Store the baseline summary, checkpoints, traces, screenshots, stdout/stderr, and command.

### B. Diagnose from evidence

- Extract a bounded evidence bundle for each failed or regressed task: evaluator result,
  exception, activity/UI observations, action sequence, action outcomes, relevant prompt
  context, memory recalls, and provider/device errors.
- Classify the likely cause as one or more of: prompt/reasoning, observation grounding,
  action/control dispatch, workflow/recovery, memory, provider/runtime, emulator/setup,
  or evaluator/instrumentation.
- State the observed evidence, competing explanations, confidence, and one falsifiable
  improvement hypothesis. Do not infer causality from task-template names alone.
- Save the diagnosis prompt and response as an artifact. Keep diagnostic model output
  advisory; the runtime and verifier remain authoritative.

### C. Propose one mutation

- Select exactly one high-impact, general change per child whenever practical. Examples:
  better current-state grounding, safer post-action verification, bounded recovery,
  improved memory retrieval/curation, prompt compaction, tool schema repair, or transport
  resilience.
- Describe the expected mechanism, affected files, risks, and a regression test before editing.
- Do not combine unrelated refactors with an experimental mutation.

### D. Isolate and implement

- Create a disposable Git worktree or equivalent isolated candidate from the parent commit.
- Apply only the proposed mutation plus required tests/documentation.
- Run fast deterministic unit/contract tests before using an emulator.
- Record the patch/diff, changed files, test output, and implementation rationale.
- If the patch cannot be applied cleanly, preserve the failed candidate and continue with
  an explicit `mutation_failed` status; never silently mutate the parent.

### E. Evaluate with staged gates

- Run the unchanged parent and candidate on the same task manifest, seed, model, endpoint,
  emulator image, timeout, and step budget.
- Use the default nested `2 -> 5 -> 20` AndroidWorld protocol: run the same 2-task screen
  for every candidate; only candidates passing the screen gate run the remaining 3 tasks
  needed for the 5-task selection score; only the eligible top candidate(s) run the 20-task
  confirmation/evaluation suite. Record the exact task lists and planned denominator at each
  stage. A 2-task smoke or 5-task selection result must never be reported as a 20-task result.
- The screen is a gate, not the final claim. A small smoke result is not a full AndroidWorld
  result and must never be compared as if it were one.
- Keep evaluation serial when concurrency can reset devices, exhaust model context, or mix
  traces. A failed setup must be distinguished from a measured task failure.
- Compare not only success rate but also exception rate, timeout rate, action efficiency,
  recovery count, verification quality, and reproducibility. These are diagnostics unless
  explicitly declared as secondary selection metrics.

### F. Select, archive, and learn

- Rank candidates using scripted planned-denominator success first, then deterministic
  tie-breakers such as exceptions and reproducibility. Confirmation runs validate; they do
  not retroactively select a parent.
- Archive every candidate's metadata, manifest/hash, patch, diagnosis, logs, checkpoints,
  trajectories, summary, and selection decision.
- Promote only a candidate that improves the declared metric without unacceptable regressions
  on the protected regression suite. Otherwise retain the parent and record why the child lost.
- Convert a successful, general, independently verified lesson into harness memory only after
  review/curation. Store failure-avoidance lessons separately from successful procedures.
  Never promote an unverified model suggestion as fact.
- After each generation, update a human-readable improvement ledger with: parent, child,
  mutation, evidence, per-task result, why it helped or failed, regressions, and next hypothesis.

## Required artifacts

Each evolution run must have a stable run directory, for example:

`runs/harness_dgm/<run_id>/`

It must contain, at minimum:

- `run_metadata.json` — immutable protocol/configuration record;
- `task_manifest.json` and its SHA-256;
- `baseline/` and `candidates/<candidate>/` evaluation outputs;
- `diagnoses/<candidate>_evidence.json`;
- `diagnoses/<candidate>_prompt.md` and `_diagnosis.md`;
- `candidates/<candidate>/model_patch.diff`;
- `candidates/<candidate>/tests.log` and evaluation logs;
- `candidates/<candidate>/checkpoints/` and chronological traces;
- `summary.md` — compact table of candidates, stage, planned successes, score,
  exceptions, status, and selection decision;
- `improvement_ledger.md` — cumulative evidence-backed history;
- `next_hypotheses.md` — unresolved causes and proposed next experiments.

The summary must explicitly distinguish `completed`, `failed`, `invalid_setup`,
`baseline_not_measured`, `mutation_failed`, `timed_out`, and `interrupted`.

## AndroidWorld connection and execution procedure

Use AndroidWorld only as the remote benchmark environment and evaluator. The harness process
may run on the development/server host while the AndroidWorld AVD runs on another server.
The following procedure is the default remote topology; replace the host, serial, ports, and
paths with the values recorded in `run_metadata.json`.

### 1. Prepare the AndroidWorld environment on the emulator host

- Start one AndroidWorld AVD and wait until `adb devices` reports exactly the intended device
  as `device` (not `offline` or `unauthorized`).
- Expose the emulator's gRPC port to the harness host. Record the ADB serial and gRPC port;
  do not silently fall back to `127.0.0.1` when the emulator is remote.
- Install/select an isolated Python environment containing the official AndroidWorld and
  AndroidEnv packages. Do not install DGM or Mobile-Agent into this environment.
- Run the AndroidWorld app/snapshot setup once. Resolve onboarding and permission dialogs
  deliberately, then save and verify the required app snapshots before benchmarking.
- Confirm that the canonical task evaluator can initialize and reset the device without
  changing the task definitions or evaluator code.

### 2. Connect the harness host

From the harness repository root, preflight the endpoint and device:

```bash
adb connect "${ANDROIDWORLD_DEVICE}" || true
adb -s "${ANDROIDWORLD_DEVICE}" get-state
curl -fsS "${ANDROIDWORLD_BASE_URL%/}/models"
```

Run the harness-owned AndroidWorld launcher with explicit connection settings:

```bash
ANDROIDWORLD_PYTHON=/path/to/androidworld-python \
ANDROIDWORLD_SOURCE=/path/to/official/android_world \
ANDROIDWORLD_DEVICE=emulator-host.example:5555 \
ANDROIDWORLD_GRPC_PORT=8554 \
ANDROIDWORLD_BASE_URL=http://127.0.0.1:8000/v1 \
ANDROIDWORLD_MODEL=your-model \
scripts/run_eval_local.sh --seed 42 --max-steps 15
```

The launcher must import the repository's `mobile_harness` first. Its AndroidWorld adapter may
configure gRPC, ADB reverse/forwarding, and device transport, but prompts, planning, memory,
tools, controls, and action dispatch must remain in the harness. Before a measured run, save
the resolved command and environment (excluding secrets) and verify that the trace contains
the expected remote device serial.

### 3. Run the staged 2-5-20 protocol

Use the checked-in [`androidworld_subset_2_5_20.json`](androidworld_subset_2_5_20.json)
manifest first. It contains the fixed 2-task `screen`, 5-task `selection`, and 20-task
`evaluation` lists, with nested membership and coverage rationale. The evaluator validates
that every name exists in the AndroidWorld registry. Change the task lists only by creating a
new versioned manifest and run ID; do not silently reorder or replace tasks during a run.

```bash
# Baseline and candidate screen: exactly the manifest's 2 planned tasks.
ANDROIDWORLD_TASK_FILE=androidworld_subset_2_5_20.json \
ANDROIDWORLD_SUBSET_SPLIT=screen scripts/run_eval_local.sh \
  --seed 42 --max-steps 15 \
  --output-dir runs/harness_dgm/<run_id>/screen

# Selection: exactly the manifest's 5 planned tasks, including the 2 screen tasks.
ANDROIDWORLD_TASK_FILE=androidworld_subset_2_5_20.json \
ANDROIDWORLD_SUBSET_SPLIT=selection scripts/run_eval_local.sh \
  --seed 42 --max-steps 15 \
  --output-dir runs/harness_dgm/<run_id>/selection

# Confirmation/evaluation: exactly the manifest's 20 planned tasks.
ANDROIDWORLD_TASK_FILE=androidworld_subset_2_5_20.json \
ANDROIDWORLD_SUBSET_SPLIT=evaluation scripts/run_eval_local.sh \
  --seed 42 --max-steps 15 \
  --output-dir runs/harness_dgm/<run_id>/confirmation
```

If the launcher does not support a named suite, pass the task list through its explicit task
file/list option and record that command. The evaluator must report success using
`task.is_successful(env)`, with `successful_tasks / planned_tasks` as the selection score.
Run the 5-task stage only after the candidate passes the declared 2-task threshold; run the
20-task stage only for the selected eligible candidate(s). A setup or transport failure must
stop the stage and be labeled `invalid_setup` or `baseline_not_measured`, not converted into
task failures.

### 4. Preserve remote-run evidence

For every stage, retain the command, seed, manifest hash, model/base URL, device serial,
AndroidWorld revision, prompt version, harness commit, checkpoints, chronological traces,
screenshots, stdout/stderr, and summary. Do not copy only the final score. The remote emulator
may be reused between stages only after an explicit reset/setup step is recorded; otherwise
device state is a confounder and the comparison is invalid.

## How to work in this Codex session

1. Inspect the current repository, tests, runtime entrypoints, prior ledgers, and recent
   run artifacts before proposing work.
2. Identify the current parent and the strongest evidence-backed bottleneck.
3. Make one bounded improvement, test it, and run the smallest valid AndroidWorld comparison.
4. If the remote emulator or model endpoint is unavailable, perform all safe offline checks,
   record the exact blocker, and do not fabricate an evaluation.
5. Continue across generations when the user asks to keep improving. Never stop at a green
   unit-test run if the requested AndroidWorld behavior remains unverified.
6. At every handoff, report the current best candidate, measured score over planned tasks,
   artifacts, regressions, and the next falsifiable improvement—not just a narrative claim.

## Completion condition

Do not claim that the harness is “best” or “finished” after one smoke run. The goal is met
only when the current implementation is reproducibly evaluated on the declared AndroidWorld
protocol, all required artifacts and ledgers exist, the selected candidate beats or matches
the protected parent without unacceptable regressions, and the ownership boundary remains
clean. If evidence is incomplete, keep the goal active and say exactly what remains.
```

## Why this differs from the original DGM setup

The local DGM implementation contributes the useful evolution mechanics: immutable parents,
isolated child worktrees, evidence-grounded diagnosis, staged screen/selection/confirmation
evaluation, planned-task denominators, and complete run artifacts. The prompt deliberately
removes its SWE-bench/Mobile-Agent assumptions and makes the mobile harness the sole owner of
prompts, tools, actions, workflows, and memory. AndroidWorld remains only the environment and
scripted evaluator.

The main source references reviewed while preparing this contract are:

- `godel_machine_mobile_agent/dgm/ANDROIDWORLD_DGM_EXPERIMENTS.md`
- `godel_machine_mobile_agent/dgm/androidworld_dgm.py`
- `godel_machine_mobile_agent/dgm/self_improve_step.py`
- `mobile_harness/prompts.py`
- `mobile_harness/tools.py`
- `mobile_harness/memory.py`
- `scripts/evaluate_androidworld.py`
- `ANDROIDWORLD_EVALUATION.md`
