# PAO Agent 迁移说明

## 当前入口

统一对话入口是 **Lead Agent**。它负责理解用户问题、选择 Subagent、汇总证据并提出动作建议。Orchestration Runtime 负责并发、预算、超时、取消和失败隔离；Subagent 只处理受限的专业任务；所有写入、仿真和优化动作必须经过 `ActionGateway`。

`src/agents/` 顶层目录固定为：

- `lead/`：用户会话、Lead Agent 和 LLM 客户端。
- `orchestration/`：Runtime、图、HITL 和闭环协调。
- `subagents/`：边界、过程、动力学、接入、技术顾问及工具。
- `context/`：只读 `ProcessContextBuilder`。
- `evidence/`：证据、来源和用户动作契约。
- `actions/`：proposal、validation、preview、确认、执行和审计网关。

旧的 Coordinator、Conversation、Demo Workflow、Analysis Agent 和 Optimization Agent 框架已删除。不要恢复旧目录，也不要让 Agent 直接访问 Aspen COM。

## 命令迁移

只读统一对话：

```powershell
python -m src.main cases/demo_case_2/pareto_config_epsd_aligned.yaml --conversation
```

也可以将 `.bkp`/`.apw` 作为入口；Lead Agent 会建立只读上下文。没有 LLM key 时，系统使用规则路由和可读的降级结果，不伪造模型结论。

普通确定性优化仍然保留，且不依赖 Lead Agent 或 LLM：

```powershell
python -m src.main cases/demo_case/case_config.yaml
python -m src.main cases/demo_case/case_config.yaml --dry-run
```

需要修改变量、运行仿真或执行优化时，先生成 `ActionProposal`/`ApprovalRequest`，再经过 validation 和 preview，等待人工确认，最后由 `ActionGateway.execute()` 调用已有 Aspen/优化执行层。Agent 不改变工程硬边界、目标函数、物性模型或动力学模型。

## 验证边界

不安装 Aspen 的环境可以运行编译检查、规则模式、SQLite/NodeDB、Runtime、ActionGateway 和大多数 pytest。真实 Aspen COM、Windows 驱动行为和真实工艺结果需要 Windows、Aspen Plus、许可证和可用模型文件。BoTorch、gpytorch 以及各 LLM provider SDK 是可选依赖；缺失时应明确 skip 或规则降级。

## 旧入口对应关系

| 旧方式 | 新方式 |
|---|---|
| Coordinator/Conversation REPL | `LeadAgent` + `ConversationSession` |

所有 Agent 请求现在都必须先进入 LeadAgent。`ProcessDecisionAgent` 保留为
闭环内部组件；YAML/`--conversation` 由 LeadAgent 生成确认请求后调用
`ClosedLoopService`。普通确定性优化 CLI 继续直接执行，不属于 Agent 入口。
| 旧专业 Agent 路径 | `src.agents.subagents.*` |
| 旧 `process_context` | `src.agents.context` |
| 旧 `contracts` | `src.agents.evidence` |
| 旧 `graph`/`hitl_protocol`/`closed_loop` | `src.agents.orchestration.*` |
| Agent 直接执行 Aspen | `ActionProposal` → validation → preview → confirmation → `ActionGateway` |

已有 `src/workflows/agent_entry.py`、`agent_optimize.py` 和普通优化 CLI 继续使用确定性优化器、SimulationDB、NodeDB、checkpoint 和恢复能力；迁移只改变 Agent 入口和编排边界。

## Stage 2: Structured intent protocol

Stage 2 adds `src/agents/lead/intent.py` with `IntentClassifier`. Every LeadAgent message is classified before context loading, Subagent dispatch, or ActionProposal creation. The classifier prefers the configured PAO LLM and requires strict JSON. Missing keys, API errors, empty responses, JSON parse errors, schema errors, and low confidence produce explicit `classification_source=rule_fallback` and a fallback reason.

## Stage 4: LeadAgent structured intent main path

LeadAgent now enables LLM classification by default. Passing `use_llm=False` is an explicit test or configuration choice; a normal missing provider key remains `fallback_reason=llm_key_missing`. Every fallback records `classification_source`, the reason, the structured intent, and an audit event on `SessionState`, and the user-facing answer identifies the rule fallback. Strict intent validation now rejects unknown context requirements, invalid confidence values, mixed `none` context, and inconsistent intent/action pairs. Classification still happens before context loading, task planning, or action proposal creation, including implicit project questions that use the current session context.

Stage 4 verification uses fake LLM responses for valid JSON, missing keys, provider errors, invalid JSON, schema errors, empty responses, low confidence, and implicit project questions. The targeted suite is run with a project-local pytest base directory because the default Windows pytest temporary directory is not writable.

## Stage 5: Controlled PAO subagents

`SubagentRegistry` now registers the nine supported roles individually: `project_researcher`, `process_advisor`, `optimization_advisor`, `boundary_advisor`, `kinetics_advisor`, `risk_reviewer`, `code_engineer`, `report_writer`, and `general_agent`. Each role has a `SubagentDefinition` containing its description, read-only tool allowlist, step and budget limits, resident policy, subtask policy, expected input context, and Evidence sources. The existing PAO Advisor implementations remain available behind compatibility adapters; adapters return structured `TaskResult` objects and explicitly record proposal-only behavior where applicable.

`OrchestrationRuntime` receives role definitions and clamps task steps, budget, tools, and subtask permissions before execution. No role is allowed to create subagents by default, and role handlers do not execute Aspen, simulations, optimization, file writes, database writes, or YAML writes. `tests/test_stage5_subagents.py` uses fake context and registry calls to verify all roles, metadata, structured results, proposal-only behavior, and runtime limit enforcement.

## Stage 6: On-demand read-only context

`ContextLoader` now maps `IntentSchema.context_required` to bounded read-only context packets. `none` performs no project load; process, YAML, NodeDB, SimulationDB/history, and selected files are projected independently. The loader records a compact summary and source references in `SessionState.context_summary` and `SessionState.context_sources`. File loading is limited to explicitly selected paths and a bounded byte count. Aspen COM objects are never placed in session state, task results, or checkpoints. Existing `ProcessContextBuilder` remains the read-only source adapter, while unrequested large fields are removed from the packet before Runtime dispatch.

`tests/test_stage6_context_loader.py` verifies no-load general context, process loading, source projection, selected-file provenance, and LeadAgent session persistence. Tests use `-p no:tmpdir` because the Windows pytest temporary-directory cleanup is permission restricted in this environment.

## Stage 7: ActionProposal and ActionGateway chain

LeadAgent proposals now carry payload, evidence identifiers, risks, and the current intent. `ActionGateway.preview_proposal()` maps conversational action names (`read`, `simulate_once`, `optimize`) to PAO action types, passes payload values through, and returns the validated preview together with payload/evidence/risk metadata. The Gateway still refreshes current executor state before validation and revalidates immediately before execution. Structural modification remains rejected by hard validation. Reads map to `inspect`; writes, simulations, and optimization remain approval-gated. `tests/test_stage7_actions.py` covers fake executor confirmation, preview and validation failures, expiry, hard-bound rejection, audit events, payload propagation, and the absence of direct Aspen access from the model boundary.

## Stage 8: Controlled closed-loop optimization

`ClosedLoopService` is now an internal execution boundary. It refuses to run unless the caller supplies explicit confirmation, while `LeadAgent.confirm_closed_loop()` remains the public approval path. The service can persist JSON-safe orchestration metadata containing the task, intent, context summary and sources, Evidence IDs, ActionProposal, ApprovalRequest, budget and audit information; driver and COM objects are never serialized. Existing `agent_optimize` checkpoint logic remains authoritative for optimizer observations, budget consumption, pending evaluations and unknown simulation outcomes. The ordinary optimization CLI path remains independent. Stage 8 tests use a fake runner to verify no execution before confirmation, confirmed execution, metadata checkpointing and COM exclusion.

## Stage 9: Unified user entry points

`ConversationSession` now owns both ordinary turns and closed-loop proposals. `--conversation` is the only Agent CLI entry; the former legacy flags have been removed. `.bkp/.apw` and YAML Agent requests must use `--conversation`, and all messages enter the same LeadAgent construction. `run_configured_agent()` remains an internal adapter and rejects public bypass calls. The ordinary `python -m src.main <yaml>` optimization entry remains separate and does not route through LeadAgent. `tests/test_stage9_entrypoints.py` covers session routing, closed-loop session use, bypass rejection, and absence of the removed flags.

The public contracts in `src/agents/orchestration/models.py` now include `IntentSchema`, `SubagentRequest`, `Classification`, timeout task status, retry and recursion metadata on `TaskSpec`, and intent, classification, fallback, context summary, and checkpoint fields on `SessionState`. Mutating, simulation, optimization, and structural action types require confirmation at schema validation time.

`tests/test_stage2_intent.py` uses fake LLM responses to cover strict JSON, unknown fields, confirmation consistency, missing keys, API and parse failures, low confidence, implicit project context, and SessionState round trips.

## 阶段 3：受约束编排运行时

阶段 3 扩展 `src/agents/orchestration/runtime.py`：

- `TaskPlanner` 将已校验的 `IntentSchema` 转换为带依赖、预算、工具白名单和并行标记的 `TaskSpec`。
- Runtime 按任务成本扣减总预算，限制最大任务数和并发数。
- 支持任务超时、取消、有限重试和结构化 `TaskStatus.TIMEOUT`。
- 通过递归深度和 `allow_subtasks` 限制 Subagent 继续派单。
- 对重复任务签名进行循环检测并记录 `task_loop_detected`。
- 保存可序列化 resident 状态，下一次任务通过 `resident_state` 接收摘要。
- 将工具白名单、最大工具调用数和 allow-check 回调传给 Subagent。
- 记录启动、完成、重试、取消、超时、拒绝和循环事件。
- `validate_results` 检查失败、超时、缺失证据和冲突证据。

新增 `tests/test_stage3_runtime.py`，使用 fake Subagent 覆盖重试、超时、取消、递归限制、循环检测、resident 状态、工具白名单和冲突证据。

## 阶段 3 补充完成

阶段 3 的剩余运行时能力已补齐：

- `OrchestrationRuntime` 支持独立 SQLite runtime checkpoint；只保存 TaskSpec、TaskResult、resident 摘要、预算和过程审计，不保存 Aspen COM 或完整 ProcessContext。
- `resume=True` 会恢复任务结果、resident 摘要和审计记录；已完成任务不会重复执行。
- `arun()` 提供异步调用适配，实际任务仍由受约束的同步 worker 执行。
- `rank_results()` 和 `summarize_results()` 为 LeadAgent 提供结果排序、失败统计和 Evidence 冲突汇总。
- `SessionState.runtime_state` 保存跨轮 resident 摘要；LeadAgent 每轮恢复并更新该状态。
- TaskPlanner 为不同 Subagent 角色生成只读工具白名单；Runtime 同时传递 `is_tool_allowed` 检查回调和最大工具调用数。
- 过程事件继续写入 `SessionState.audit_events`，并随 Runtime checkpoint 持久化。

阶段 3 专项测试覆盖 checkpoint 往返、resident 恢复、异步执行、结果综合、重试、超时、取消、递归限制、循环检测、工具白名单和 Evidence 冲突。
