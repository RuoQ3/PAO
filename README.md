# PAO：Process Aspen Optimization

PAO 是一个面向 Aspen Plus 的化工流程优化工具。它把 Aspen Plus 仿真、确定性优化算法、工程约束、结果数据库和可审计的 Agent 编排连接起来。

核心原则：

- Aspen Plus 负责过程求解，优化器负责候选点生成，数据库负责证据留存。
- Agent 负责理解问题、整理证据、选择受限的 Subagent 和提出动作建议。
- Agent 不直接访问 Aspen COM，也不把未经验证的模型输出当作仿真结果。
- 修改、仿真和优化动作必须经过 `proposal → validation → preview → human confirmation → ActionGateway.execute()`。
- 工程硬边界、目标函数、约束、物性模型和动力学模型不会被 Agent 自动改写。

## 当前能力

### 确定性优化和仿真

- Aspen Plus COM 驱动、输入回读校验、超时和失败恢复。
- 单目标贝叶斯优化：`optimizer.type: bayesian`。
- 多目标 Pareto 优化：`optimizer.type: pareto_bayesian`。
- skopt 代理模型和随机采样降级路径。
- 可选 BoTorch qEHVI/qNEHVI 后端；未安装 `torch/botorch/gpytorch` 时自动降级。
- 可行性判断、工程边界校验、敏感性分析、Pareto 和超体积计算。
- 派生变量、整数变量、变量依赖和输入去重。

### 数据和报告

- `SimulationDB`：保存仿真工况、目标、约束、状态和结果投影。
- `NodeDB`：保存 Aspen 节点目录、manifest 和节点快照。
- checkpoint SQLite：保存 Agent 闭环的状态、预算、证据、动作和恢复信息。
- Pareto 报告、闭环 JSON/Markdown 报告和离线分析工具。

### 统一 Agent

统一用户入口是 **Lead Agent**。它可以：

- 直接回答与 PAO 无关的一般问题；
- 根据问题路由到边界、过程、动力学、技术顾问或分析 Subagent；
- 读取 `ProcessContext`、SimulationDB、NodeDB 和已有证据；
- 并发执行相互独立的只读 Subagent 任务；
- 汇总结构化 `TaskResult`、`Evidence`、不确定性和建议；
- 对修改或执行请求生成 `ActionProposal`/`ApprovalRequest`，等待人工确认。

没有 LLM key 时，系统使用规则路由和确定性降级逻辑，并明确标记证据不足；不会伪造 LLM 结论或仿真结果。

## Agent 架构

```text
用户 → Lead Agent → Orchestration Runtime → Subagents / Context / Evidence
                                      ↓
                              ActionGateway
                                      ↓
                         Aspen Driver / 优化器 / 数据库
```

`src/agents/` 顶层目录只有以下六个部分：

| 目录 | 职责 |
|---|---|
| `lead/` | Lead Agent、会话状态和 LLM 客户端 |
| `orchestration/` | Runtime、LangGraph、HITL 和闭环协调 |
| `subagents/` | 专业顾问、接入 Agent 和只读工具 |
| `context/` | 只读工艺上下文构建 |
| `evidence/` | 证据模型、来源协议和用户动作契约 |
| `actions/` | ActionProposal、校验、预览、执行和审计网关 |

旧的 Coordinator、Conversation、Demo Workflow、Analysis Agent 和 Optimization Agent 框架已经删除。旧路径不再是公共入口。

## 环境要求和安装

- Python 3.10 或更高版本；CI 使用 Python 3.12。
- Windows + Aspen Plus + 可用许可证 + `.bkp`/`.apw` 文件，才能运行真实 Aspen COM 仿真。
- Linux/macOS 可以运行不连接 Aspen 的规则模式、配置检查、数据库分析和大多数测试。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

轻量测试环境至少需要：

```powershell
python -m pip install pytest PyYAML numpy scipy scikit-learn scikit-optimize python-dotenv langgraph langchain-core
```

BoTorch 是可选后端：

```powershell
python -m pip install torch botorch gpytorch
```

### LLM 配置

复制 `.env.example` 为 `.env`，只配置所选 provider 的 key：

```powershell
Copy-Item .env.example .env
```

```text
PAO_LLM_PROVIDER=anthropic   # anthropic / openai / deepseek
PAO_LLM_MODEL=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
PAO_LLM_TEMPERATURE=0.2
PAO_LLM_MAX_TOKENS=2048
```

LLM key 不写入 checkpoint，也不会出现在 Agent 上下文或日志中。没有 key 时使用规则模式。IMA 默认使用本地 mock；只有明确配置 HTTP 地址和鉴权后才访问外部服务。

## CLI

所有命令从仓库根目录执行。

### 配置检查

`--dry-run` 只加载并校验 YAML，不连接 Aspen、不调用 LLM、不写回 YAML：

```powershell
python -m src.main cases/demo_case/pareto_config.yaml --dry-run
python -m src.main cases/demo_case_2/agent_loop_config.yaml --dry-run
```

### 普通单目标和 Pareto 优化

单目标配置使用 `optimizer.type: bayesian`：

```powershell
python -m src.main path/to/case_config.yaml
```

多目标配置使用 `optimizer.type: pareto_bayesian`：

```powershell
python -m src.main cases/demo_case/pareto_config.yaml
```

结果默认写入 YAML 所在目录的 `output/simulation.db`，日志写入同目录的 `run.log`。可覆盖路径：

```powershell
python -m src.main path/to/case_config.yaml --db output/run1.db --log DEBUG --log-file output/run1.log
```

真实运行需要 Aspen Plus；没有 Aspen 时使用 `--dry-run` 或测试中的 fake driver。

### Agent 闭环

闭环请求统一通过 `--conversation` 进入 `ConversationSession`。YAML 中的
`agent_loop.enabled: true` 仍由内部 `ClosedLoopService` 执行；普通优化 CLI 不会进入
Lead Agent。

```powershell
python -m src.main cases/demo_case_2/agent_loop_config.yaml --dry-run
python -m src.main cases/demo_case_2/agent_loop_config.yaml --conversation
```

也可以对已有 Pareto YAML 使用 `--conversation` 发起闭环请求：

```powershell
python -m src.main cases/demo_case_2/pareto_config_epsd_aligned.yaml --conversation
```

恢复中断的闭环：

```powershell
python -m src.main cases/demo_case_2/agent_loop_config.yaml --conversation --resume
```

指定新的 checkpoint：

```powershell
python -m src.main cases/demo_case_2/agent_loop_config.yaml --conversation --checkpoint cases/demo_case_2/output/my_agent_run.db
```

checkpoint 是闭环恢复的权威记录，不能与 `SimulationDB` 使用同一个 SQLite 文件。恢复时会检查配置、模型文件、目标模块和知识库指纹。

### 统一只读对话

`--conversation` 是唯一的 Agent 对话入口。该模式不会直接写 YAML、数据库、Aspen 或执行优化：

```powershell
python -m src.main cases/demo_case/pareto_config.yaml --conversation
python -m src.main cases/demo_case/二级氢氰化工段.bkp --conversation
```

也可以指定上下文来源：

```powershell
python -m src.main cases/demo_case/二级氢氰化工段.bkp --conversation --yaml cases/demo_case/pareto_config.yaml --node-db cases/demo_case/output/node.db
```

### 裸 Aspen 文件

`.bkp`/`.apw` 文件不能直接作为普通优化 YAML。使用 Lead Agent 只读接入：

```powershell
python -m src.main cases/demo_case/二级氢氰化工段.bkp --conversation --intent "帮我接入这个工艺并分析可调变量"
```

需要动力学分析时，可以附带实验 CSV：

```powershell
python -m src.main cases/demo_case/二级氢氰化工段.bkp --conversation --intent "拟合这批实验数据的活化能" --data path/to/kinetics.csv
```

## YAML 配置

典型配置包含 `simulator`、`design_variables`、`objectives`、`constraints` 和 `optimizer`：

```yaml
simulator:
  filepath: cases/demo_case/二级氢氰化工段.bkp
  visible: false
  suppress_dialogs: true
  timeout: 300
  reinit: true
  verify_inputs: true

design_variables:
  - name: T0301_BASIS_RR
    aspen_path: \Data\\Blocks\\T0301\\Input\\BASIS_RR
    type: continuous
    lower_bound: 1.5
    upper_bound: 2.6
    initial_value: 2.0
    unit: '-'

objectives:
  - name: REB_DUTY
    type: aspen_path
    aspen_path: \Data\\Blocks\\T0301\\Output\\REB_DUTY
    minimize: true
    unit: kW

constraints:
  - name: purity_min
    aspen_path: \Data\\Streams\\ADN\\Output\\MASSFRAC\\MIXED\\ADN
    operator: ">="
    threshold: 0.999

optimizer:
  type: pareto_bayesian
  surrogate_model: qEHVI
```

设计变量类型：`continuous` 连续变量，`integer` 整数变量，`fixed` 固定变量，`derived` 按配置公式映射到 Aspen 的派生变量。

Agent 闭环常用配置：

```yaml
agent_loop:
  enabled: true
  max_evaluations: 40
  batch_size: 5
  failure_trigger: 3
  max_decisions: 12
  stagnation_batches: 5
  max_step_fraction: 0.1
  candidate_pool_size: 24
  verification_repeats: 1
  verification_rtol: 0.02
  verification_atol: 1.0e-8
  objective_targets: {}
  checkpoint_path: output/agent_checkpoint.db
  process_facts: {}
```

`design_variables.lower_bound/upper_bound` 是工程硬边界；Agent 只能在硬边界内调整软搜索区域和候选点。`objective_targets: {}` 表示不设置目标阈值，不表示已经达标。

`simulator` 控制 Aspen 连接、重置、超时和输入回读；`extraction` 控制 block/stream 节点提取、manifest、语义规则和 NodeDB 写入。仿真未收敛、目标缺失或约束不可用时，工况保持失败/未知状态，不会被转换成成功样本。

## 安全执行边界

```text
用户请求 → ActionProposal → validation → preview → 人工确认
       → ActionGateway.execute → Aspen Driver / 优化器
       → SimulationDB / checkpoint / audit
```

ActionGateway 会检查动作类型、权限、Aspen 路径、类型、单位、旧值、硬边界、变量依赖、模型身份和确认状态。结构拓扑、物性模型、反应动力学、目标函数和约束定义不属于普通 Agent 写操作。

## 闭环状态、恢复和结果

每次调用 Aspen 前，闭环会先保存 pending action 并扣除预算。仿真中途退出时，该次结果视为未知，不会自动重放。恢复时重建优化器和证据，不序列化 COM 对象。

闭环会生成 `*.report.json` 和 `*.report.md`。常见结果状态：

| 状态 | 含义 |
|---|---|
| `target_verified` | 达到指定目标并通过独立复验 |
| `completed_verified` | 代表工况复验通过，不代表全局最优或目标达标 |
| `verification_failed` | 复验失败、偏差超限或无法收敛 |
| `no_feasible_solution` | 当前预算内没有找到可行解 |
| `unverified` | 驱动不可用或未完成复验 |

## 代码结构

```text
src/
├── agents/              # Lead Agent、Runtime、Subagent、Context、Evidence、Actions
├── aspen_driver/        # Aspen COM、catalog、manifest、topology
├── database/            # SimulationDB、NodeDB
├── economics/           # TAC、排放和单位
├── integrations/ima/    # SDK-neutral IMA adapter
├── models/              # ProcessCase、SimulationResult、Tunable 等
├── optimization/        # surrogate、Pareto、可行性、敏感性
├── reporting/           # 汇总、Pareto 图和目标报告
├── utils/               # YAML、日志和单位工具
└── workflows/           # 普通优化和 Agent 闭环入口
```

## 离线分析工具

不连接 Aspen 也可以读取 `SimulationDB` 做闭环分析：

```python
from src.agents.subagents.tools import analyze_closed_loop_tool

report = analyze_closed_loop_tool.invoke({
    "db_path": "cases/demo_case/output/simulation.db",
    "objective_names": "ADN_FLOW,REB_DUTY",
    "session_id": "optional-session-id",
})
```

该工具只分析已保存的工况和证据，不执行写入、仿真或优化。

## 测试和静态检查

推荐使用项目内临时目录，避免系统临时目录权限问题：

```powershell
python -m compileall -q src tests
python -m pytest --basetemp E:\\PAO-test-tmp\\pytest-base -q
git diff --check
```

测试覆盖 Lead Agent 路由和降级、Runtime 并发/预算/超时/失败隔离、证据不足、ActionGateway 确认链、checkpoint 往返恢复、普通闭环和离线分析。真实 Aspen 集成测试需要 Windows + Aspen 环境。

## 文档和迁移

- [Agent 对话架构](docs/agent_conversation_architecture.md)
- [Agent 闭环](docs/agent_closed_loop.md)
- [Agent 迁移说明](docs/agent_migration.md)
- [IMA adapter 阶段说明](docs/ima_adapter_phase1.md)

从旧入口迁移时，使用 `LeadAgent + ConversationSession`。`--conversation` 是唯一用户可见的 Agent CLI 入口；旧版 Agent CLI 选项已删除。闭环由内部 `ClosedLoopService` 调度，所有修改、仿真和优化动作都经过 `ActionProposal`、校验、预览、人工确认和 `ActionGateway.execute()`。普通确定性优化 CLI 仍使用 `python -m src.main <yaml>`，不依赖 LeadAgent、LLM 或 Agent Runtime。

没有 LLM key 时使用明确标记的规则模式，不伪造仿真结果；真实 Aspen/COM 验证需要 Windows 和 Aspen 环境。旧 Coordinator、Conversation、Demo Workflow 和 Analysis Agent/Optimization Agent 入口已收敛到 LeadAgent 及其 Subagent/内部闭环服务。

## 限制

- 真实 Aspen 结果只能由 Aspen Plus 实际运行产生；规则模式或 LLM 输出不能替代仿真。
- BoTorch、IMA HTTP 和具体 LLM provider 都是可选能力，缺失时使用明确的 skip、mock 或规则降级。
- 合成测试验证软件边界，不证明任何具体化工流程的优化收益。
