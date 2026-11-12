# PAO Agent 迁移说明

> 2026-11 执行边界更新：旧 Agent 闭环和布尔授权入口已禁用，以下闭环执行内容属于历史实现说明。当前入口、限制和迁移以 [execution_authorization.md](execution_authorization.md) 为准。

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

也可以将 `.bkp`/`.apw` 作为入口；Lead Agent 在验证 LLM 分类后建立只读上下文。没有 LLM key 时立即终止请求，不生成回答或审批。

普通确定性优化仍然保留，且不依赖 Lead Agent 或 LLM：

```powershell
python -m src.main cases/demo_case/case_config.yaml
python -m src.main cases/demo_case/case_config.yaml --dry-run
```

需要修改变量、运行仿真或执行优化时，先生成 `ActionProposal`/`ApprovalRequest`，再经过 validation 和 preview，等待人工确认，最后由 `ActionGateway.execute()` 调用已有 Aspen/优化执行层。Agent 不改变工程硬边界、目标函数、物性模型或动力学模型。

## 验证边界

Agent 必须使用已配置且通过严格校验的 LLM。无 Aspen 的自动化验证使用显式 fake provider、fake IMA 和 fake runner；缺失依赖或 provider 不可用时抛出 AgentUnavailableError，不切换规则模式。详见 [fail-closed 架构](agent_fail_closed.md)。

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

## 严格意图与失败协议

每轮先执行 IntentClassifier，成功来源仅为 llm/llm_repaired。JSON/schema 最多修复一次；配置、请求、超时、低置信度或修复失败统一终止。use_llm=False 同样失败，不能通过注入 classifier 绕过。会话保存不可用审计并清空审批；恢复旧规则 checkpoint 必须重新开始会话。
