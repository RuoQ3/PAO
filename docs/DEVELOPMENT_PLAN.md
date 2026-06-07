# PAO 项目开发计划

> 版本：v1.0  
> 更新日期：2026-06-06  
> 当前目标：**协作调优已有 Aspen 流程** — 给定一个可跑通但未调优的 `.bkp/.apw` 文件，多 agent 协助用户接入现有优化框架，达成多目标优化目标（如 TAC 最小、排放最小、纯度约束）。

---

## 背景与约束

### 已有、不改动的核心资产

| 模块 | 说明 |
|------|------|
| `src/aspen_driver/` | Aspen COM 封装，边界清晰，任何 agent 不得直接操作 |
| `src/workflows/optimize_pareto_case.py` | ParEGO 多目标贝叶斯优化，已成熟 |
| `src/workflows/run_case.py` | 单次仿真生命周期 |
| `src/database/simulation_db.py` | ProcessCase SQLite 持久化 |
| `src/database/node_db.py` | 节点目录 / manifest 持久化 |
| `src/aspen_driver/catalog.py` | Aspen 树扫描器 |
| `src/aspen_driver/manifest.py` | 语义规则 → ReadManifest 生成器 |
| `configs/aspen_semantics/radfrac.yaml` | RADFRAC 精馏塔语义规则 |
| `configs/aspen_semantics/heatx.yaml` | HEATX 换热器语义规则 |
| `src/agents/process_advisor.py` | 只读体检（6 个安全工具） |
| `src/agents/process_advisor_agent.py` | 只读体检 + LLM 分析 |
| `src/agents/tools/` | run_case / optimize_pareto / query_* / diagnose / summarize / validate / load |
| `src/economics/tac.py` / `emissions.py` | TAC 与排放估算 |
| `src/optimization/feasibility*.py` | 可行性分类器与质量门控 |
| `cases/demo_case/pareto_config.yaml` | 配置 schema 参考基准，**不改格式** |

### 绝对不做（本阶段）

- 不让 LLM 直接操作 Aspen COM
- 不让 agent 在未经用户确认的情况下自动跑大批量 Aspen 仿真
- 不修改 Aspen 流程拓扑（只调操作参数）
- 不实现 literature / knowledge / build_process 工艺合成路线
- 不修改现有 pareto_config.yaml schema

---

## 三阶段总览

| 阶段 | 目标 | 关键交付 |
|------|------|---------|
| **A 接入层** | 把陌生 Aspen 文件半自动生成优化配置 | `tunable.py` / 语义规则 / `discover_tunables` tool / `config_builder` |
| **B 协作状态机** | 配置草案→用户确认→优化→分析→迭代的有状态闭环 | `onboarding_agent` / `graph.py` LangGraph 状态机 |
| **C 报告层** | 调优结果的综合分析报告 | `summary_report.py` / 变量重要性分析 |

---

## 阶段 A：接入层

> 目标：给定 `.bkp` 文件 + 用户优化意图，产出通过 `validate_config_tool` 的优化配置草案。  
> 验收基准：对 `cases/demo_case/二级氢氰化工段.bkp` 全链路跑通，结果与人工写的 `pareto_config.yaml` 基本吻合。

---

### A1 — 定义变量发现数据结构

**文件**：`src/models/tunable.py`（新建）

**任务清单**：

- [x] **A1-1** 定义 `TunableVariable` dataclass
  - 字段：`aspen_path: str`、`semantic_role: str`、`suggested_type: Literal["continuous","integer"]`、`current_value: float | None`、`suggested_lower: float | None`、`suggested_upper: float | None`、`unit: str`、`confidence: Literal["high","medium","low"]`、`reason: str`
  - `confidence="high"` 仅当语义规则中有明确经验边界；`"medium"` 表示规则匹配但边界是经验估算；`"low"` 表示靠路径模式推断无边界数据
  - 不依赖任何外部库，纯 Python dataclass

- [x] **A1-2** 定义 `ReadableTarget` dataclass
  - 字段：`aspen_path: str`、`semantic_role: str`、`candidate_use: Literal["objective","constraint","both"]`、`unit: str`、`current_value: float | None`

- [x] **A1-3** 定义 `TunableReport` dataclass
  - 字段：`aspen_file: str`、`aspen_file_hash: str`、`tunable_variables: list[TunableVariable]`、`readable_targets: list[ReadableTarget]`、`scan_warnings: list[str]`、`semantic_coverage: float`
  - 方法：`get_high_confidence_vars() -> list[TunableVariable]`、`get_targets_for_use(use) -> list[ReadableTarget]`

- [x] **A1-4** 定义 `GoalSpec` dataclass
  - 字段：`metric: str`（"TAC"/"emissions"/"purity"/"yield"/"flow"/"custom"）、`direction: Literal["min","max"]`、`target_value: float | None`（约束阈值，方向为"min_above"/"max_below"时有值）、`custom_aspen_path: str | None`

- [x] **A1-5** 定义 `OptimizationIntent` dataclass
  - 字段：`goals: list[GoalSpec]`、`hard_constraints: list[GoalSpec]`、`n_initial: int`（默认 20）、`n_iterations: int`（默认 60）、`notes: str`

- [x] **A1-6** 定义 `ConfigDraft` dataclass
  - 字段：`draft_id: str`、`aspen_file: str`、`design_variables: list[dict]`、`objectives: list[dict]`、`constraints: list[dict]`、`optimizer: dict`、`extraction: dict`、`warnings: list[str]`、`confidence_summary: str`
  - 方法：`to_yaml_dict() -> dict`（输出可被 `load_optimize_config` 解析的字典）

- [x] **A1-7** 编写单元测试 `tests/models/test_tunable.py`
  - 覆盖：dataclass 构建、字段默认值、`get_high_confidence_vars` 过滤、`to_yaml_dict` 序列化

**验收标准**：`python -c "from src.models.tunable import TunableReport, ConfigDraft"` 无报错；所有单测通过。

---

### A2 — 补充 Aspen 语义规则

**文件**：`configs/aspen_semantics/` 下新增 4 个 YAML 文件

规则 YAML 格式与 `radfrac.yaml` 完全一致，每个字段含 `required_for`、`required`、`candidates`（`pattern`/`priority`）、`validators`。

---

#### A2-1 — `configs/aspen_semantics/flash.yaml`

**目标**：覆盖 Flash2 / Flash3 单元操作

- [x] **A2-1a** 定义 `equipment_type: FLASH2`
- [x] **A2-1b** 添加可调输入字段规则
  - `temperature`：`Input\TEMP`（连续，经验范围 250–450 K，confidence medium）
  - `pressure`：`Input\PRES`（连续，经验范围 0.1–10 atm，confidence medium）
  - `vapor_fraction`：`Input\VFRAC`（连续，范围 0–1，confidence high）
- [x] **A2-1c** 添加可读输出字段规则
  - `vapor_flow`：`Output\MOLE_FLOW\VAPOR` / `Output\MASS_FLOW\VAPOR`
  - `liquid_flow`：`Output\MOLE_FLOW\LIQUID` / `Output\MASS_FLOW\LIQUID`
  - `heat_duty`：`Output\DUTY`（`required_for: [TAC, EMISSIONS, ENERGY]`）
- [x] **A2-1d** 添加 `FLASH3` 的额外规则条目（第二液相出口）
- [x] **A2-1e** 编写匹配测试：给定模拟 catalog 条目，验证 pattern 匹配正确

---

#### A2-2 — `configs/aspen_semantics/pump.yaml`

**目标**：覆盖 Pump / Compr / MCompr 单元操作

- [x] **A2-2a** 定义 `equipment_type: PUMP`
- [x] **A2-2b** 添加可调输入字段
  - `outlet_pressure`：`Input\PRES`（连续，经验范围 1–100 bar）
  - `efficiency`：`Input\EFF`（连续，范围 0.5–0.9，confidence medium）
- [x] **A2-2c** 添加可读输出字段
  - `power`：`Output\WNET`（`required_for: [TAC, EMISSIONS]`）
  - `outlet_temperature`：`Output\TOUT`
- [x] **A2-2d** 定义 `equipment_type: COMPR`，字段与 PUMP 类似但优先级不同
- [x] **A2-2e** 编写匹配测试

---

#### A2-3 — `configs/aspen_semantics/rstoic.yaml`

**目标**：覆盖 RStoic / REquil / RGibbs 反应器

- [x] **A2-3a** 定义 `equipment_type: RSTOIC`
- [x] **A2-3b** 添加可调输入字段
  - `temperature`：`Input\TEMP`（反应温度，连续，范围依工艺而定，confidence low）
  - `pressure`：`Input\PRES`（连续，confidence low）
  - `conversion`：`Input\CONV\*`（各反应转化率，连续，范围 0–1）
- [x] **A2-3c** 添加可读输出字段
  - `heat_duty`：`Output\QCALC`（`required_for: [TAC, EMISSIONS, ENERGY]`）
  - `outlet_temperature`：`Output\TOUT`
- [x] **A2-3d** 定义 `equipment_type: REQUIL`，同上结构
- [x] **A2-3e** 编写匹配测试

---

#### A2-4 — `configs/aspen_semantics/splitter.yaml`

**目标**：覆盖 FSplit / SSplit / Mixer

- [x] **A2-4a** 定义 `equipment_type: FSPLIT`
- [x] **A2-4b** 添加可调输入字段
  - `split_fraction`：`Input\FRAC\*`（各出口分流比，连续，范围 0–1，confidence high）
- [x] **A2-4c** 添加可读输出字段
  - `outlet_flow`：`Output\MOLE_FLOW\*` / `Output\MASS_FLOW\*`
- [x] **A2-4d** 定义 `equipment_type: MIXER`（无可调输入，只有输出字段）
- [x] **A2-4e** 编写匹配测试

---

#### A2-5 — 验证语义覆盖率

- [x] **A2-5a** 对 `cases/demo_case/二级氢氰化工段.bkp` 运行 `CatalogScanner`，记录所有 block_type
- [x] **A2-5b** 确认 `ManifestBuilder` 对 demo_case 的语义覆盖率 ≥ 0.70
- [x] **A2-5c** 确认 T0301（RADFRAC）的 `BASIS_RR`、`B:F`、`FEED_STAGE`、`REB_DUTY` 均被正确识别

**验收标准**：A2-5b 和 A2-5c 均通过；新增 4 个规则文件的匹配单测全部通过。

---

### A3 — 实现变量发现工具

**文件**：`src/agents/tools/discover_tunables.py`（新建）

> ⚠️ 这是本阶段**唯一**新增的会打开 Aspen 的代码路径，只扫描不仿真，绝不调用 `Engine.Run2` 或 `run_case`。

**任务清单**：

- [x] **A3-1** 实现 `_scan_aspen_file(aspen_path: str, node_db_path: str, max_depth: int) -> CatalogScan`
  - 封装 `AspenDriver` 打开文件 + `CatalogScanner.scan()`
  - 打开后**不运行仿真**（不调 `driver.run()` / `Engine.Run2`）
  - 关闭 Aspen 前确保 `driver.close()` 被调用（`with` 上下文或 `try/finally`）
  - 扫描失败的节点记入 `scan_warnings`，不抛出异常

- [x] **A3-2** 实现 `_build_tunable_variables(scan: CatalogScan, rules_dir: str) -> list[TunableVariable]`
  - 遍历 catalog 中所有 `Input` 节点
  - 用 `ManifestBuilder` 的规则加载逻辑匹配语义角色
  - 命中规则且规则有经验边界 → `confidence="high"` 或 `"medium"`
  - 无规则但节点为 Input 类型 → `confidence="low"`，边界为 None，写入 `reason="未匹配语义规则"`
  - 扫描失败节点（`CatalogScan.failed_nodes`）排除，不进推荐列表

- [x] **A3-3** 实现 `_build_readable_targets(scan: CatalogScan, rules_dir: str) -> list[ReadableTarget]`
  - 遍历 catalog 中所有 `Output` 节点
  - 匹配语义规则中 `required_for: [TAC]` → `candidate_use="objective"`
  - 匹配 `required_for: [EMISSIONS]` → `candidate_use="objective"`
  - 质量分数/摩尔分数节点 → `candidate_use="constraint"`
  - 两者都能用 → `candidate_use="both"`

- [x] **A3-4** 实现 `_compute_semantic_coverage(tunable_vars, readable_targets, scan) -> float`
  - 计算被规则命中的节点数 / 总节点数

- [x] **A3-5** 实现 `discover_tunables_impl(aspen_file_path: str, node_db_path: str, rules_dir: str, max_depth: int) -> TunableReport`
  - 串联 A3-1 ~ A3-4
  - 异常时返回含 `scan_warnings` 的 `TunableReport`，不向上抛

- [x] **A3-6** 用 `@tool` 包装为 `discover_tunables_tool`
  - 输入：`aspen_file_path: str`、`node_db_path: str`、`max_depth: int = 6`
  - 输出：JSON 序列化的 `TunableReport` 文本（或 "错误：..." 前缀的失败描述）
  - 在 `src/agents/tools/__init__.py` 中注册

- [x] **A3-7** 编写单元测试 `tests/agents/tools/test_discover_tunables.py`
  - 用 `MockCatalogScanner` 注入 mock catalog 数据，不依赖真实 Aspen
  - 覆盖：规则命中 → TunableVariable 正确生成；失败节点被排除；覆盖率计算正确
  - 覆盖：tool 输出格式为合法 JSON 字符串

- [x] **A3-8** 编写导入隔离测试（确认 `discover_tunables.py` 不在导入期触碰 `Engine.Run2`）
  - 测试方法：`import src.agents.tools.discover_tunables` 后不触发 COM 调用

**验收标准**：所有单测通过；对 demo_case `.bkp` 实际运行，识别出 `BASIS_RR`/`B:F`/`FEED_STAGE` 为 `TunableVariable`，`REB_DUTY`/`MASSFRAC`/`MASSFLOW` 为 `ReadableTarget`。

---

### A4 — 实现配置构建器

**文件**：`src/agents/config_builder.py`（新建）

> 纯 Python + 可选 LLM，不打开 Aspen，不调 `driver`。

**任务清单**：

#### A4-1 规则映射器（必做，不依赖 LLM）

- [x] **A4-1a** 实现 `_map_goal_to_objective(goal: GoalSpec, targets: list[ReadableTarget], tac_defaults: dict, emissions_defaults: dict) -> dict | None`
  - `metric="TAC"` → `{type: tac, name: TAC, minimize: true, annualization_factor: ..., operating_hours: ..., utility_cost: {...}}`，参数从 `src/economics/tac.py` 中的默认值读取
  - `metric="emissions"` → `{type: emissions, name: EMISSIONS, minimize: true, ...}`，参数从 `src/economics/emissions.py` 默认值读取
  - `metric="flow"` → 从 `targets` 找 `semantic_role` 包含 `flow` 的节点 → `{type: aspen_path, ...}`
  - `metric="custom"` → 用 `goal.custom_aspen_path` 直接生成 `{type: aspen_path, ...}`
  - 找不到匹配节点时返回 `None` 并记录 warning

- [x] **A4-1b** 实现 `_map_constraint_to_dict(constraint: GoalSpec, targets: list[ReadableTarget]) -> dict | None`
  - `metric="purity"` → 从 `targets` 找 `candidate_use` 含 `constraint` 且 `semantic_role` 含 `mass_frac`/`mole_frac` 的节点
  - 生成 `{name: ..., aspen_path: ..., operator: ">=", threshold: constraint.target_value}`
  - 找不到时返回 `None` + warning

- [x] **A4-1c** 实现 `_map_tunable_to_design_var(var: TunableVariable) -> dict`
  - 生成与 `pareto_config.yaml` `design_variables` 段完全一致的字典
  - `suggested_lower` / `suggested_upper` 为 None 时：`lower_bound` / `upper_bound` 填 `null`，并在 draft `warnings` 中追加"请手动填写 {var.aspen_path} 的变量边界"

- [x] **A4-1d** 实现 `_build_optimizer_section(intent: OptimizationIntent, n_vars: int) -> dict`
  - `type: pareto_bayesian`（2 个以上目标时）或 `type: bayesian`（1 个目标时）
  - `n_initial_points`：取 `intent.n_initial`，若未指定则用 `max(10, 5 * n_vars)` 经验公式
  - `n_iterations`：取 `intent.n_iterations`，若未指定则用 `2 * n_initial_points`
  - 默认开启 `early_stopping`（参数与 demo_case 一致）
  - 默认开启 `feasibility_filter`（`enabled: false`，等用户手动开启）

- [x] **A4-1e** 实现 `_build_extraction_section(aspen_file: str, node_db_path: str, tunable_vars: list[TunableVariable], targets: list[ReadableTarget]) -> dict`
  - 自动填充 `blocks`、`streams` 列表（从 tunable_vars 和 targets 的路径中提取 block/stream 名）
  - `mode: manifest`、`catalog_db` 指向 node_db_path
  - `build_manifest_if_missing: true`

- [x] **A4-1f** 实现主函数 `build_config_draft(report: TunableReport, intent: OptimizationIntent) -> ConfigDraft`
  - 调用 A4-1a ~ A4-1e，组装 `ConfigDraft`
  - 对 `confidence != "high"` 的每个设计变量，在 `warnings` 中追加说明
  - 对 `objectives` / `constraints` 中有 None 的每项，在 `warnings` 中追加说明

#### A4-2 LLM 意图解析器（可选，后做）

- [x] **A4-2a** 实现 `parse_intent_from_text(text: str, llm_config: LLMConfig) -> OptimizationIntent`
  - 调用 `llm_client.chat()`，system prompt 要求 LLM 输出 JSON 格式的意图结构
  - LLM 输出必须经 JSON schema 校验，校验失败时抛 `IntentParseError`
  - **LLM 只解析意图，不直接生成配置字段**

- [x] **A4-2b** 实现 `parse_intent_from_text` 的降级路径
  - LLM 未配置或调用失败时，返回只含 "TAC 最小化 + 排放最小化" 的默认 `OptimizationIntent`，并在 notes 中说明"LLM 未配置，已使用默认意图"

#### A4-3 测试

- [x] **A4-3a** 单元测试 `tests/agents/test_config_builder.py`
  - 覆盖：`_map_goal_to_objective` 对 TAC/emissions/flow/custom 各类型的映射
  - 覆盖：找不到目标节点时返回 None + 写入 warning
  - 覆盖：`build_config_draft` 对边界为 None 的变量写入 warning
  - 覆盖：`to_yaml_dict` 输出符合 pareto_config.yaml schema

- [x] **A4-3b** 集成测试：`ConfigDraft.to_yaml_dict()` 写成临时 YAML 文件后，能被 `_impl_load_config` 解析（不报错）

**验收标准**：所有单测通过；集成测试中 `validate_config_tool` 对生成的 YAML 返回 OK（无 fatal 错误）。

---

### A5 — 端到端链路测试

**文件**：`tests/integration/test_onboarding_pipeline.py`（新建）

- [ ] **A5-1** 实现 mock 扫描测试（不依赖真实 Aspen）
  - 构造一个模拟 demo_case 的 `TunableReport`（含 B:F / FEED_STAGE / BASIS_RR 三个 TunableVariable，REB_DUTY / MASSFRAC / MASSFLOW 三个 ReadableTarget）
  - 构造 `OptimizationIntent`（ADN_FLOW↑ / REB_DUTY↓ / purity≥0.9）
  - 调用 `build_config_draft` → `ConfigDraft.to_yaml_dict()`
  - 写临时 YAML 文件
  - 调用 `validate_config_tool` → 断言无 fatal 错误
  - 调用 `load_optimize_config` → 断言能解析为 `ParetoOptimizeCaseConfig`
  - 断言 `objective_names` 包含预期目标；`param_bounds` 包含三个设计变量路径

- [ ] **A5-2** 与人工配置对比测试
  - 加载 `cases/demo_case/pareto_config.yaml`（人工配置）
  - 用 A5-1 中的 mock TunableReport + OptimizationIntent 生成 ConfigDraft
  - 断言：设计变量路径集合与人工配置的 `param_bounds` 键集合完全一致
  - 断言：目标函数名称集合与人工配置的 `objective_names` 一致
  - 断言：约束的 aspen_path 与人工配置的 constraints 一致

- [ ] **A5-3**（可选，需要真实 Aspen）实际扫描 + 完整链路测试
  - 用 `discover_tunables_tool` 扫描 demo_case `.bkp` → 真实 `TunableReport`
  - 走完 A5-1 的链路，断言结果与 A5-2 中人工配置的对比通过

**验收标准**：A5-1 和 A5-2 在 CI 中（不依赖 Aspen）通过；A5-3 在本地有 Aspen 的环境中通过。

---

## 阶段 B：协作状态机

> 前置条件：阶段 A 全部完成。  
> 目标：实现用户↔多 agent 的协作调优闭环（HITL 人在回路）。

---

### B0 — Agent 目录结构重组

**涉及路径**：`src/agents/`

> 在开始开发任何新 agent 之前，先完成目录结构调整，确保后续所有 agent 都落在统一规范的目录结构下。

**目标结构**：

```
src/agents/
├── __init__.py
├── state.py                    # 不动，全局状态定义
├── llm_client.py               # 不动
├── tool_runner.py              # 不动
├── workflow_helpers.py         # 不动
├── workflow_report.py          # 不动
│
├── tools/                      # 不动，底层公共 tool（run_case / optimize_pareto 等）
│   ├── __init__.py
│   ├── run_case.py
│   ├── optimize_pareto.py
│   └── ...
│
├── process_advisor/            # 从现有文件迁移
│   ├── __init__.py             # re-export，保持对外接口不变
│   ├── agent.py                # 原 process_advisor_agent.py 内容
│   ├── tools.py                # 原 process_advisor.py 中的 ReadOnlyToolRunner
│   └── prompts.py              # 原 _SYSTEM_PROMPT / _USER_TEMPLATE
│
├── onboarding_agent/           # B1 新建
│   ├── __init__.py
│   ├── agent.py                # run_onboarding / apply_user_feedback 入口函数
│   ├── tools.py                # 本 agent 被允许调用的工具列表与适配
│   └── prompts.py              # 意图解析 system prompt / few-shot 示例
│
├── optimization_agent/         # B2 新建
│   ├── __init__.py
│   ├── agent.py                # optimization_node 逻辑
│   ├── tools.py                # 只允许 optimize_pareto_tool / query_simulation_db_tool
│   └── prompts.py
│
├── analysis_agent/             # B2 新建
│   ├── __init__.py
│   ├── agent.py                # analysis_node 逻辑
│   ├── tools.py                # 只读工具列表
│   └── prompts.py
│
└── graph.py                    # 保持在 agents/ 根，各 agent 的编排入口
```

**规则约定**：
- `tools/` 目录中的底层公共 tool 不属于任何 agent，供所有 agent 按需引用
- 每个 `xx_agent/tools.py` 只声明"该 agent 被允许调用哪些工具"，不实现新工具
- 每个 `xx_agent/prompts.py` 存放该 agent 的所有 prompt 常量，逻辑代码与 prompt 文本分离
- `__init__.py` 负责 re-export 对外接口，调用方无需感知内部文件结构

**任务清单**：

- [x] **B0-1** 新建 `src/agents/process_advisor/` 目录，迁移现有文件
  - 将 `process_advisor_agent.py` 内容移入 `process_advisor/agent.py`
  - 将 `process_advisor.py` 中的 `ReadOnlyToolRunner` 和工具适配逻辑移入 `process_advisor/tools.py`
  - 将 `_SYSTEM_PROMPT` / `_USER_TEMPLATE` 移入 `process_advisor/prompts.py`
  - 在 `process_advisor/__init__.py` 中 re-export `run_process_advisor_agent` 等对外接口
  - 删除原 `process_advisor.py` 和 `process_advisor_agent.py`

- [x] **B0-2** 验证迁移后现有调用方不受影响
  - 检查项目中所有 `from src.agents.process_advisor` 和 `from src.agents.process_advisor_agent` 的导入
  - 确认通过 `__init__.py` re-export 后，所有现有导入路径不变或更新为新路径
  - 运行现有 `process_advisor` 相关测试，确认全部通过

- [x] **B0-3** 新建后续 agent 目录占位
  - 创建 `src/agents/onboarding_agent/`、`optimization_agent/`、`analysis_agent/` 三个目录
  - 每个目录只放空的 `__init__.py`，内容在 B1/B2 中填充

**验收标准**：迁移完成后，现有所有测试通过；`from src.agents.process_advisor import run_process_advisor_agent` 可正常导入；新的三个空目录已就绪，等待 B1/B2 填充。

---

### B1 — 接入向导 Agent

**文件**：`src/agents/onboarding_agent.py`（新建）

- [x] **B1-1** 定义 `OnboardingResult` dataclass
  - 字段：`config_draft: ConfigDraft`、`tunable_report: TunableReport`、`questions_for_user: list[str]`、`warnings: list[str]`
  - `questions_for_user` 由以下规则生成：
    - 每个 `confidence != "high"` 的设计变量边界 → 生成"请确认 {var.aspen_path} 的合理范围（建议 [{lo}, {hi}]）"
    - 目标函数映射有 warning 时 → 生成对应的确认问题
    - `n_initial` / `n_iterations` 建议值 → 生成"建议运行 {n_initial} 次初始采样 + {n_iterations} 次优化迭代，是否接受？"

- [x] **B1-2** 实现 `run_onboarding(aspen_file_path: str, intent_text: str, node_db_path: str, llm_config: LLMConfig | None) -> OnboardingResult`
  - 调用 `discover_tunables_tool` 获取 `TunableReport`
  - 若 `intent_text` 非空：调用 `parse_intent_from_text` 解析意图；LLM 不可用时使用默认意图
  - 调用 `build_config_draft` 生成 `ConfigDraft`
  - 生成 `questions_for_user` 列表
  - **不启动优化，不调 `run_case` / `optimize_pareto`**

- [x] **B1-3** 实现 `apply_user_feedback(draft: ConfigDraft, feedback: dict) -> ConfigDraft`
  - `feedback` 格式：`{"bounds": {aspen_path: [lo, hi]}, "objectives": [...], "n_initial": int}`
  - 把用户修改合并进 `ConfigDraft`，重新校验
  - 更新 `warnings`（已被用户确认的条目可从 warnings 移除）

- [x] **B1-4** 单元测试 `tests/agents/test_onboarding_agent.py`
  - 覆盖：`questions_for_user` 对各种 confidence 级别的生成
  - 覆盖：`apply_user_feedback` 正确合并用户修改
  - 使用 mock `TunableReport` 和 mock `parse_intent`，不依赖 LLM 和 Aspen

**验收标准**：单测通过；`run_onboarding` 对 mock 数据能产出非空的 `questions_for_user`，且 `config_draft.warnings` 与 `questions_for_user` 覆盖同一批置信度不足的变量。

---

### B2 — LangGraph 协作状态机

**文件**：`src/agents/graph.py`（当前空文件，实现）

> 参考 CMU `2506.20921v2` 的 GroupChat 架构，适配 PAO 的 HITL 场景。

- [x] **B2-1** 定义 `PAOGraphState` dataclass（仅含基本类型，不含 ProcessCase 等底层对象）
  - 字段：`session_id: str`、`aspen_file: str`、`intent_text: str`、`current_phase: str`（`"onboarding"/"confirming"/"optimizing"/"analyzing"/"done"`）、`config_draft: ConfigDraft | None`、`config_yaml_path: str | None`（草案写成文件后的路径）、`db_path: str | None`、`onboarding_result: OnboardingResult | None`、`analysis_report: str`、`iteration: int`、`max_iterations: int`（默认 5）、`messages: list[str]`、`termination_reason: str | None`

- [x] **B2-2** 实现 `onboarding_node(state: PAOGraphState) -> PAOGraphState`
  - 调用 `run_onboarding`，把结果存入 state
  - 更新 `current_phase = "confirming"`
  - 把 `questions_for_user` 追加到 `messages` 供前端/用户展示

- [x] **B2-3** 实现 `human_confirm_node`（HITL 节点）
  - LangGraph `interrupt_before` 模式暂停，等待客户端注入用户反馈
  - 用户反馈通过 `apply_user_feedback` 合并进 `config_draft`
  - 把 `ConfigDraft.to_yaml_dict()` 写成临时 YAML 文件，路径存入 `state.config_yaml_path`
  - 调用 `validate_config_tool` 校验；校验 fatal 失败时回到 `confirming`，最多重试 3 次
  - 通过后更新 `current_phase = "optimizing"`

- [x] **B2-4** 实现 `optimization_node(state: PAOGraphState) -> PAOGraphState`
  - 调用 `optimize_pareto_tool`，传入 `state.config_yaml_path`
  - 更新 `db_path`（从 config yaml 推断）
  - 更新 `current_phase = "analyzing"`
  - 若工具返回错误前缀：更新 `current_phase = "done"`，`termination_reason = "optimization_failed"`

- [x] **B2-5** 实现 `analysis_node(state: PAOGraphState) -> PAOGraphState`
  - 调用 `process_advisor_agent`（只读，6 个安全工具 + LLM 分析）
  - 把报告存入 `state.analysis_report`
  - 更新 `current_phase = "deciding"`
  - 把分析报告摘要追加到 `messages`

- [x] **B2-6** 实现 `human_decide_node`（HITL 节点）
  - LangGraph `interrupt_before` 模式暂停，等待用户决策
  - 用户决策选项：`"continue"`（收窄边界继续优化）/ `"adjust"`（调整意图重新配置）/ `"done"`（终止）
  - `"continue"` 且 `state.iteration < state.max_iterations` → 转到 `human_confirm_node`（带新边界）
  - `"adjust"` → 转到 `onboarding_node`（重新走意图解析）
  - `"done"` 或超过最大迭代次数 → 转到 `done_node`

- [x] **B2-7** 实现 `done_node(state: PAOGraphState) -> PAOGraphState`
  - 更新 `current_phase = "done"`
  - 生成最终摘要追加到 `messages`（总轮次、终止原因、结果数据库、分析报告状态）

- [x] **B2-8** 用 `StateGraph` 串联所有节点
  - 边：`START → onboarding → human_confirm → optimization → analysis → human_decide → (循环或 done_node) → END`
  - 条件边：`human_decide` 根据用户决策 dispatch

- [x] **B2-9** 单元测试 `tests/agents/test_graph.py`
  - 用 mock patch 替换所有外部调用 + `interrupt_before` 模式下的 HITL 流程
  - 覆盖：完整的 mock 闭环（onboarding → confirm → optimize → analyze → done）
  - 覆盖：迭代超过 `max_iterations` 时自动终止
  - 覆盖：optimization 失败时直接终止
  - 覆盖：用户选择 "adjust" 时回到 onboarding

**验收标准**：所有单测通过；状态机在 mock 模式下能完整走完 2 轮迭代后自然终止；`current_phase` 转换路径与设计一致；校验超过 `max_confirm_retries` 次失败时终止（`termination_reason="confirm_validation_failed"`）且 `optimize_pareto_tool` 未被调用；无 checkpointer 时发出 `UserWarning`。

---

## 阶段 C：报告层

> 目标：调优结束后产出一份完整的综合分析报告，供用户决策和存档。

---

### C1 — 综合分析报告

**文件**：`src/reporting/summary_report.py`（新建）

- [x] **C1-1** 实现 `generate_tac_breakdown(db_path: str, session_id: str | None) -> str`
  - 从 SimulationDB 读取 Pareto 第一前沿的工况
  - 对每个工况调用 `tac.py` 的详细计算，输出设备费用 vs 操作费用的构成表
  - 结果格式：Markdown 表格

- [x] **C1-2** 实现 `generate_emissions_summary(db_path: str, session_id: str | None) -> str`
  - 从第一前沿工况读取排放数据
  - 按蒸汽 / 电力 / 冷却水分项输出，对比最优点和最差点
  - 结果格式：Markdown 表格

- [x] **C1-3** 实现 `generate_variable_importance(db_path: str, objective_name: str, session_id: str | None) -> str`
  - 从 SimulationDB 读取所有成功工况的 `design_vars` 和目标值
  - 计算每个设计变量与目标值的 Spearman 相关系数
  - 输出排序后的重要性表（变量名 / 相关系数 / 方向说明）
  - 结果格式：Markdown 表格 + 简短文字解读

- [x] **C1-4** 实现 `generate_failure_summary(db_path: str, session_id: str | None, limit: int = 5) -> str`
  - 调用 `diagnose_case_tool` 对最近 N 个失败工况
  - 归并诊断结果：按失败类型统计（sim_failed / objective_error / infeasible）
  - 找出失败工况的设计变量聚集区域（哪个变量的哪个范围容易失败）
  - 结果格式：Markdown

- [x] **C1-5** 实现 `generate_summary_report(db_path: str, config_path: str, session_id: str | None) -> str`
  - 串联 C1-1 ~ C1-4 + 现有 `plot_pareto` 的摘要文字
  - 输出完整的 Markdown 报告，包含 5 个章节：
    - 【1. 优化总览】（总迭代次数、成功率、Pareto 前沿大小、超体积）
    - 【2. TAC 分解】
    - 【3. 排放分析】
    - 【4. 设计变量重要性】
    - 【5. 失败诊断摘要】

- [x] **C1-6** 编写单元测试 `tests/reporting/test_summary_report.py`
  - 用 mock SimulationDB 数据测试各生成函数的输出格式
  - 覆盖：空数据库时各函数返回"无数据"说明而非报错

**验收标准**：对 demo_case 的 `simulation.db` 运行 `generate_summary_report`，输出有效的 Markdown 报告，包含全部 5 个章节，无 None 或空字段。

---

## 开发顺序与依赖图

```
A1 (tunable.py)
A2 (语义规则)
    ↓         ↓
A3 (discover_tunables)    A4 (config_builder)
    ↓                          ↓
    └──────────┬───────────────┘
               ↓
            A5 (端到端测试)
               ↓
         B1 (onboarding_agent)
               ↓
         B2 (graph.py 状态机)

C1 (summary_report)  ← 与 B 并行，只依赖现有工具
```

A1 和 A2 无依赖，**可同时开始**。A3 和 A4 依赖 A1/A2 完成后并行开发。C1 在阶段 A 完成后即可独立开发。

---

## 文件新增清单

| 文件路径 | 类型 | 阶段 |
|---------|------|------|
| `src/models/tunable.py` | 新建 | A1 |
| `configs/aspen_semantics/flash.yaml` | 新建 | A2 |
| `configs/aspen_semantics/pump.yaml` | 新建 | A2 |
| `configs/aspen_semantics/rstoic.yaml` | 新建 | A2 |
| `configs/aspen_semantics/splitter.yaml` | 新建 | A2 |
| `src/agents/tools/discover_tunables.py` | 新建 | A3 |
| `src/agents/config_builder.py` | 新建 | A4 |
| `tests/models/test_tunable.py` | 新建 | A1 |
| `tests/agents/tools/test_discover_tunables.py` | 新建 | A3 |
| `tests/agents/test_config_builder.py` | 新建 | A4 |
| `tests/integration/test_onboarding_pipeline.py` | 新建 | A5 |
| `src/agents/onboarding_agent.py` | 新建 | B1 |
| `src/agents/graph.py` | 实现（当前空文件）| B2 |
| `tests/agents/test_onboarding_agent.py` | 新建 | B1 |
| `tests/agents/test_graph.py` | 新建 | B2 |
| `src/reporting/summary_report.py` | 新建 | C1 |
| `tests/reporting/test_summary_report.py` | 新建 | C1 |

---

## 工程约束备忘

1. **Aspen driver 边界**：`discover_tunables.py` 是唯一新增的会打开 Aspen 的路径，只读扫描，不仿真。上层所有模块（`config_builder`、`onboarding_agent`、`graph.py`）导入时不得引入 `src.aspen_driver`。

2. **HITL 硬约束**：`graph.py` 的优化节点（`optimization_node`）执行前，必须经过 `human_confirm_node` 的 `interrupt_before` 暂停，用户必须显式注入反馈（`Command(update={'user_feedback': {...}})`）且配置草案通过 `validate_config_tool` 后才能继续。校验始终失败（超过 `max_confirm_retries`）时流程终止，不允许以未通过验证的草案进入优化。

3. **LLM 只解析意图**：`parse_intent_from_text` 中 LLM 的唯一职责是把自由文本解析成结构化 `OptimizationIntent`，不直接生成配置字段。所有配置字段由规则映射器生成。

4. **配置草案必过 validate**：`config_builder` 产物在进入优化节点前，必须通过 `validate_config_tool`，无 fatal 错误。

5. **置信度不高的边界必须告知用户**：`confidence != "high"` 的 `TunableVariable` 对应边界必须出现在 `questions_for_user` 中，不允许静默使用未经确认的经验估算值跑大批量仿真。

6. **现有稳定模块不改动**：`optimize_pareto_case.py` / `run_case.py` / `SimulationDB` / `process_advisor.py` / `pareto_config.yaml schema` 均不改动。

---

## 参考文献

| 文献 | 与本计划的关联 |
|------|-------------|
| Zeng et al., 2025 `2506.20921v2` — LLM-guided Chemical Process Optimization with a Multi-Agent Approach | ContextAgent → A3/A4 设计参考；5 agent 角色分工 → B2 状态机参考；使用强推理模型的必要性 → `llm_client` 模型选择 |
| Tian et al., 2026 `2601.06776v1` — From Text to Simulation | Task Understanding Agent → A4-2 意图解析参考；E-MCTS 迭代闭环 → B2 迭代终止策略参考 |
| Zhou et al., 2024 `cej.pdf` — ReLU-ANN + MILP | 验证 feasibility 分类器的正确性（已在 PAO 实现）；未来 surrogate 加速路线参考（阶段 D） |
| Zhang et al., 2021 AIChE — SA-PSO 非清晰切割蒸馏 | 远期蒸馏序列合成参考，与本阶段无关 |
| Hou et al., 2026 CES — VRC-HIEDS | 远期工艺合成 TAC 优化参考，与本阶段无关 |
