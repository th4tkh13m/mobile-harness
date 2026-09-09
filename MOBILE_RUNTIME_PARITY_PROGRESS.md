# Mobile Runtime Parity Progress

Last updated: 2026-08-13

Status legend: **Implemented** = usable native runtime capability with tests;
**Partial** = interface/core behavior exists but needs production hardening or
integration; **Deferred** = intentionally outside the first milestone.

| Capability | Reference agent | Codex | Current mobile runtime | Status | Evidence / remaining work |
|---|---|---|---|---|---|
| Multi-turn tool loop | Tool dispatch, retries, recovery | Durable thread/turn/item loop | Durable `MobileAgentRuntime` repeatedly observes, asks the model, dispatches multiple calls, verifies claims, records events, classifies failed tools, and persists provider retry/fallback diagnostics. `ResilientModel` retries eligible provider failures with configurable capped exponential backoff before failover, honoring a bounded numeric `retry_after_seconds` or HTTP-style `Retry-After` hint when available; each failed attempt records its scheduled delay for replay | **Implemented** | `runtime.py`, `resilience.py`; add provider quota telemetry if deployment requires it. |
| Versioned prompt assembly | Stable/context/volatile tiers, assembled once and rebuilt on compaction; rich tool-local schema guidance | Instruction/environment layers and typed tool contracts | Stable/context/volatile system-prompt assembly, cached by session and invalidated only after compaction/restore. The Android computer-use block is a deliberate mobile adaptation. Tool contracts are colocated in `tools.py` and externally branded reference text is removed before model delivery | **Implemented** | `prompts.py`, `tools.py`, branding tests. |
| Task planning | `todo`, context, orchestration | Plans and progress | `todo` is backed by the same durable `PlanStep` records as the richer mobile `plan` tool. It preserves one current task store rather than allowing parallel todo/plan drift; plans add mobile-only evidence, blockers, dependencies, and retry budgets | **Implemented** | `test_hermes_todo_contract_uses_shared_durable_plan`. |
| UI grounding | Desktop computer/browser | Optional computer use/visual input | Screenshot/UI XML remain available through `DevicePort`; explicit `observe`, element payloads, locators, and normalized gestures | **Implemented** | `tools.py`, existing ADB port; add richer inspection/OCR and stale-screen diffs. |
| Mobile execution | No Android-native equivalent | No Android-native equivalent | ADB/benchmark tap, swipe/drag, text, keys, back/home, launch, wait | **Implemented** | Existing ports plus brokered mobile tools; Unicode IME remains an adapter concern. |
| Web research | Search/extract providers | Web search | Native no-key search fallback plus injectable providers return query, URL, title, snippet, source, and explicit untrusted-content metadata; `web_read` permits bounded textual HTTP(S) content types. A bounded fresh-result cache carries explicit hit/age/TTL metadata, and a host cadence guard fails fast before uncached request bursts; cached repeats do not consume another request | **Implemented** | Add robots policy and provider quality controls for production use. |
| Browser automation | Full browser controls | Browser/computer controls | Not included | **Deferred** | Search/read is the approved first-milestone boundary. |
| Workspace files | Read/write/patch/search | Shell + patch + workspace controls | Portable contracts for same-named `search_files`, `read_file`, `write_file`, and `patch`; the constrained workspace implementation supports paging/line-numbered reads, content or filename search, full writes, and replace-mode patches with approvals | **Implemented** | V4A multi-file patch mode, document conversion, and cross-profile behavior are explicitly runtime limitations. |
| Controlled command execution | Terminal/backends/sandboxing | Shell/unified exec/sandbox | Scoped `run_command` uses argument execution (no shell), rejects shell operators, enforces timeout, sanitizes output, and requires approval. A deadline cancels the spawned process group/tree, bounds post-cancellation output collection, and returns typed cancellation audit data. Deployments may configure an explicit executable-basename allowlist; nonmatching commands fail before process launch, while successful results record executable/allowlist audit metadata | **Implemented** | Add cooperative cancellation for commands that provide a native protocol. OS sandboxing is intentionally out of scope. |
| Durable session history | Persistent sessions | Durable threads/items | Atomic schema-versioned snapshots plus append-only JSONL journals record task, plans, tool calls/results, approvals, evidence, and summaries. Legacy snapshots are migrated explicitly; unknown future schema versions fail closed. New journal entries form a backward-compatible SHA-256 hash chain validated before replay; snapshots and content-addressed Android screenshot/UI-XML evidence carry SHA-256 integrity manifests. A per-session atomic filesystem lease serializes full start/resume/run trajectories across local processes, fails closed on a live competing owner, and reclaims a corrupt or definitively dead local-worker lease | **Implemented** | `SessionStore` + `HistoryIndex`; the chain detects retained-chain edits, but is not a signed tamper-proof ledger. Add retention and signed/encrypted evidence storage. |
| Context compaction | Yes | Yes | Keeps recent events and one cumulative structured checkpoint preserving active plan, failures, approvals, blockers, and verification evidence; full evidence remains in journal. An optional bounded semantic-summary hook receives only compact trajectory data, sanitizes its output, and cannot replace the structured checkpoint. Rendering records exact configured-token-counter or labeled fallback accounting, removes volatile untrusted blocks/UI elements first, then prunes only checkpoint narrative and old outcomes until the budget fits; verifier evidence and unresolved blockers are retained | **Partial** | Wire a provider-specific summary model and automatic provider tokenizer selection in deployment configuration. |
| Persistent memory | Curated/bounded/sanitized | Memory/session facilities | Staged/promoted curated JSONL memory; automatic verified-success curation, reviewed failure gate, provenance, sanitization, idempotent promotion ledger, atomic retention pruning, and explainable concept-expanded retrieval with verifier-backed-success preference. Tool-local contracts distinguish durable preference/correction/environment facts from session history and reusable skills; require declarative non-secret candidate entries; prioritize preference/correction memory; and state the success/failure-avoidance evidence gates. Candidate inspection and curator promotion are first-class approval-gated tools | **Implemented** | Add a separate preference-promotion UI and optional vector retrieval for larger corpora. |
| Session search | Yes | Thread/history operations | Cross-session append-only event-journal search exposed as `session_search`; a rebuildable on-disk term index ranks verification/recovery/failure evidence above ordinary turns. The schema now follows Hermes's source-first boundary: live URL/workspace/app/device state must be inspected directly when available; session history is secondary context. `session_trace` returns bounded causal windows around a chosen event only after replay-validating the journal hash chain | **Implemented** | `HistoryIndex`; add browse/scroll call shapes and richer ranking for multi-user deployments. |
| Learning/skill capture | Memory review + skills | Skills/instructions | Curated trajectory lessons plus versioned markdown skills with provenance/evidence frontmatter. The skill-staging contract now requires a self-contained trigger, prerequisites, grounded steps, pitfalls, and verification cues; it forbids secrets and unavailable procedures. Model skill calls only stage candidates; candidate inspection plus approval-gated curator promotion make the review workflow first-class, and recalled skills exclude candidates | **Implemented** | Add an explicit compact `skills_list`/full `skill_view` split and a graphical curator UI. |
| Verification | Hooks/evidence | Test/command/review feedback | `claim_done` invokes common verifier; typed evidence with source/check records persists. `capture_evidence` binds a bounded agent claim (optionally to a plan step) to the current content-addressed Android screenshot/UI XML; those artifact references are carried into the verifier event. `UiObservationVerifier` supplies reusable real-device recipes for activity regexes, visible text/content descriptions/resource IDs, and required evidence claims; every predicate is recorded as a typed pass/fail check. Legacy benchmark verifiers adapt without taking benchmark lifecycle ownership; failed claims generate recovery events | **Implemented** | Add task-specific backend-state adapters where UI-only evidence is insufficient. |
| Recovery/replanning | Retry/classify/compress/failover | Interrupt/retry/approval handling | Failed tools/verification and stale UI receive a recovery classification and plan-directed next step; plans retain blockers, dependency gates, optional retry budgets, and durable exponential retry-not-before timestamps for transient failures. Approval pause/resume replays saved call | **Implemented** | Add cooperative cancellation tied to external device/tool processes. |
| Approval/authority | Guardrails/permissions | Sandbox/granular approval | Central `AuthorityBroker` governs consequential mobile actions, writes, commands, and extensions; approved grants are scoped to exactly one durable session | **Implemented** | Add approval UI payloads, expiry, and command-diff presentation. |
| Prompt-injection defenses | Context/memory/tool scanning | Tool/environment boundary | Web, workspace, UI, memory, and skill content passes a shared structured scanner, is sanitized and source-delimited, and carries categorized/severity audit findings into tool results or prompt context. Current UI observations themselves now retain this untrusted provenance/findings when inserted directly into prompts; tool authority stays outside model text. The scanner also blocks CSS-hidden HTML containers and bounded base64-like tokens only when decoded text matches an existing hostile-directive rule, recording distinct `hidden_html_content`/`encoded_instruction` findings | **Partial** | Expand detector coverage, add adversarial multimodal/image/OCR scanning, and persist dedicated cross-session security analytics. |
| Tool registry/discovery | Toolsets/plugins/dynamic tools | MCP/plugins/dynamic tools | Native schemas plus explicit `CapabilityRegistry`; every discoverable capability has a manifest with authority, ordered/read-only/external execution class, trust boundary, description, and parameters; per-session enabled-tool selection applies throughout. Extensions may register health probes and versions: unavailable/probe-failed or version-incompatible tools are withheld from schemas, fail closed on invocation, and remain inspectable with diagnostic reasons | **Implemented** | Add periodic availability scheduling. |
| MCP/plugin extensibility | Yes | Yes | Explicit capability registry plus allowlisted stdio JSON-RPC discovery/invocation, refresh/cleanup, persisted server identity records, normalized allowlisted toolset-contract fingerprints, and bounded request deadlines. Operator-declared package provenance (for example package/version/digest) is normalized, bound into the persisted server identity fingerprint, survives trusted restart, and is exposed by tool inspection. Refresh reports a before/after contract fingerprint transition. Timeouts first emit MCP `notifications/cancelled` with the request ID and preserve a cooperative transport through a short grace period; unresponsive transports are hard-terminated. A restart requires fingerprint confirmation; MCP calls require approval | **Partial** | Authenticated transports and signature-verified/bundled plugin packages remain. |
| Parallel independent tools | Yes | Yes | Model adapters accept multiple calls; independent read-only calls execute concurrently while mutations stay ordered. A bounded per-turn read-only deadline returns typed timeout failures for stalled calls so recovery can proceed without blocking the turn | **Implemented** | Add dependency graph scheduling and cooperative provider cancellation where supported. |
| Subagent delegation | Yes | Yes | Bounded `delegate_non_gui` manager delegates only research/workspace-read tasks; no Android/write/command/recursive authority. It persists queued/running/completed/failed/cancelled/timed-out records, links each task explicitly to its parent runtime session, supports session-scoped record review, tolerates partial trailing journal writes on recovery, and can resume unfinished work. A trusted `JsonSubprocessWorker` passes only a JSON task contract to an isolated child, hard-terminates it on deadline, captures child output outside parent memory, and enforces output/summary/evidence limits | **Implemented (restricted)** | Add a first-party model-worker adapter and cross-process durable locking before treating it as production-grade delegation. |
| Scheduling/cron | Yes | Product-level automation | Not included | **Deferred** | Outside selected milestone. |
| Messaging/social integrations | Yes | Apps/MCP connectors | Not included | **Deferred** | Outside selected milestone. |
| Voice/image/video generation | Yes | Image capability where enabled | Not included | **Deferred** | Outside selected milestone. |
| Full desktop control | Yes | Available in some surfaces | Android is the computer-use surface | **Implemented (mobile replacement)** | Do not copy desktop control; continue to improve Android observation/action ports. |

## Current completion snapshot

- **Implemented:** durable runtime loop, multi-call provider adapters, plans,
  mobile/tool broker, atomic scoped workspace mutations with diff evidence,
  integrity-checked, cross-process session-leased snapshots/evidence plus append-only hash-chained history/search,
  verifier-gated idempotent success-memory curation, skills,
  approval pause/resume, basic prompt tiers/compaction, concurrent safe reads,
  non-GUI delegation, health-aware capability registry, and benchmark compatibility.
- **Partial next:** web robots policy and provider-quality controls, command protocol-level cooperative cancellation,
  provider-wired semantic compaction/automatic provider-tokenizer selection, external-tool cancellation,
  adversarial multimodal injection scanning and security analytics, a user-facing skill curator workflow, authenticated MCP
  transports. OS sandboxing is intentionally out of scope.
- **Deferred by design:** browser automation, desktop control, scheduling,
  messaging integrations, media generation, and subagents.

## Prompt parity audit (2026-08-13)

The following are checked as complete function-schema equality against the
local `codes/coding_agents/hermes-agent` source: the static prompt-builder
blocks `DEFAULT_AGENT_IDENTITY`, `HERMES_AGENT_HELP_GUIDANCE`,
`TASK_COMPLETION_GUIDANCE`, `TOOL_USE_ENFORCEMENT_GUIDANCE`,
`PARALLEL_TOOL_CALL_GUIDANCE`, `OPENAI_MODEL_EXECUTION_GUIDANCE`,
`GOOGLE_MODEL_OPERATIONAL_GUIDANCE`, `MEMORY_GUIDANCE`,
`SESSION_SEARCH_GUIDANCE`, `SKILLS_GUIDANCE`, and the complete
`STEER_CHANNEL_NOTE`; and the tool schemas `memory`, `web_search`,
`skills_list`, `skill_view`, `skill_manage`, `session_search`, `todo`,
`read_file`, `write_file`, `patch`, and `search_files`.

Only Android computer use is deliberately adapted: it replaces Hermes's
desktop-computer vocabulary with Android observation, locators, gestures,
keys, activity, and approval semantics. `web_read`, `run_command`,
`plan`, evidence, curator, and tool-registry calls are mobile-runtime-specific
extensions, so they are not represented as copied Hermes contracts. Kanban is
excluded because the mobile runtime does not expose kanban tools; the copied
steering note is inert unless the runtime adds a real out-of-band marker to a
tool-result batch. This is intentional capability differentiation—not a claim
of byte parity.

## Test coverage

`python -m unittest discover -s tests -v` currently exercises legacy benchmark
compatibility plus runtime planning/action/verification, approval replay,
curated-memory promotion and auto-curation, untrusted-content sanitization,
Responses multi-tool parsing, durable indexed event-history search, bounded
compaction, curated/versioned learning, injection boundaries, safe parallel
reads, non-GUI delegation recovery/deadlines, MCP timeout and
fingerprint-confirmed restore, isolated delegation protocol, model fallback
replay, and capability manifests.
