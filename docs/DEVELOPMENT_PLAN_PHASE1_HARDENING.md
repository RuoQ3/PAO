# PAO 项目开发计划 — 阶段一收尾验收（续章）

> 版本：v1.1
> 更新日期：2026-06-07
> 适用前提：`DEVELOPMENT_PLAN.md` 中 A / B / C 全部任务已"代码完成"（code complete）。
> 本文目标：把"在 demo_case 上可演示、mock 全绿"的系统，推进到"**对真实用户文件可用、有目标达成判定**"，从而真正达成第一阶段目标。

---

## 为什么需要这个续章

`DEVELOPMENT_PLAN.md` 的几乎所有验收标准都锚定在单一文件 `cases/demo_case` 上，且大量为 mock 测试。这证明的是**结构完整性**，不是**第一阶段目标达成**。

第一阶段目标的原始定义：**用户拿自己已有的、能跑通但未调优的 Aspen 文件 + 一句优化意图，系统协助调优、达成其理想目标。**

代码完成与目标达成之间存在五个未被原计划覆盖的缺口：

| 缺口 | 说明 | 风险 |
|------|------|------|
| ① 泛化性 | 语义规则/验收全冲 demo_case 单元类型调优，换文件可能覆盖率骤降 | 高 |
| ② 真实端到端 | 全链路除可选 A5-3 外均为 mock，真实 Aspen+LLM+真人确认从未跑通 | 高 |
| ③ 目标达成判定 | 产物是 Pareto 前沿+报告，但不回答"是否/多大程度达到用户初始目标" | 高 |
| ④ 前端联调 | `graph.py` 的 HITL `interrupt()` 未接入现有 FastAPI/Vue 前端 | 中 |
| ⑤ 鲁棒性 | 错边界/无可行点/不收敛等异常路径只在 mock 下验证 | 中 |

---

## 阶段总览

| 任务包 | 目标 | 关键交付 | 优先级 |
|--------|------|---------|--------|
| **H1 真实端到端单跑** | 证明 demo_case 真实链路可通 | 一次真实运行记录 + 缺陷清单 | P0 |
| **H2 目标达成度评估** | 报告能回答"是否达成、改善多少" | `summary_report` 新增达成度章节 | P0 |
| **H3 泛化性验证** | 量化非 demo_case 文件的覆盖率与失败模式 | 多样本扫描报告 + 语义规则缺口清单 | P1 |
| **H4 Vue 前端联调** | 用户在本地浏览器完成确认→优化→看报告，接入真实数据替换 mock | HITL 契约 + FastAPI 改造 + Vue 接入 | P1 |
| **H5 鲁棒性压测** | 异常路径不崩、有可读提示 | 异常用例集 + 通过记录 | P2 |

依赖关系：H1 → (H2 ∥ H3) → H4 → H5。H1 是其余一切的前置（真实链路不通，后面都无从谈起）。

---

## 边界（沿用原计划，本续章不放松）

- 不碰流程拓扑，只调操作参数。
- 不让 LLM 操作 COM；`discover_tunables.py` 仍是唯一允许开 Aspen 的新路径，只扫不跑。
- 不改 `aspen_driver/`、`optimize_pareto_case.py`、`run_case.py`、`SimulationDB`、`pareto_config.yaml` schema。
- 本续章新增能力优先落在 `summary_report.py`、新增评估模块、测试与脚本层，不动稳定核心。

---

## H1 — 真实端到端单跑（P0）✅

> 目标：用真实 Aspen + 真实 LLM + 真人确认，把 demo_case 完整链路跑通一次，把"mock 通过"升级为"真实通过"。这是验收第一阶段的最低事实门槛。
> 前提：P0-FIX 已修复导入断裂；A/B/C 代码完成。

**文件**：`scripts/run_phase1_e2e.py`（新建，编排脚本，非生产代码）；运行记录落 `reports/phase1/`。

**任务清单**：

- [x] **H1-1** 编写端到端编排脚本，串联真实链路
  - `discover_tunables_tool`（真实开 demo_case `.bkp`，只扫；二次运行走缓存 < 1s）
  - `parse_intent_from_text`（LLM 降级路径，降级意图为 ADN_FLOW 最大 + REB_DUTY 最小）
  - `build_config_draft` → `to_yaml_dict()` → 写 YAML
  - `validate_config_tool`（两次校验，无 fatal 才继续）
  - `optimize_pareto_tool`（真实 Aspen，2 次仿真，2 成功，Pareto 2 个解）
  - `generate_summary_report`（已产出 3388 字符报告）
  - 每步打印耗时与状态，全程不静默吞错

- [x] **H1-2** 人工确认环节先用脚本注入模拟（不阻塞 H1，真正的前端确认留给 H4）
  - 用预设的 user feedback dict 模拟"用户确认边界"，走 `apply_user_feedback`
  - 增加 Step 5.5 收窄设计变量到 `DEMO_CONFIRMED_PATHS`，排除未工艺确认的变量

- [x] **H1-3** 记录真实运行结果到 `reports/phase1/e2e_demo_case.md`
  - 总耗时 16s（含 discover 缓存命中 0.9s + optimize 14.9s）
  - LLM 降级路径（ANTHROPIC_API_KEY 被系统 JWT 覆盖，已修复 dotenv override=True）
  - 无中断点，链路完整

- [x] **H1-4** 产出缺陷清单 `reports/phase1/e2e_defects.md`
  - 真实运行暴露 7 个缺陷（D-001～D-007），全部已记录并修复

**验收结果（2026-06-08 02:19:36）**：
- 链路无中断产出 `reports/phase1/summary_report.md` ✅
- 运行记录与缺陷清单落盘 ✅
- 单次仿真耗时约 4.7 秒；H5 超时参考建议 300 秒 ✅
- discover 缓存命中（二次运行 0.9s vs 首次 ~900s）✅

---

## H2 — 目标达成度评估（P0）

> 目标：让系统回答"我们是否、多大程度达到了你最初说的目标"。这是第一阶段对用户真正有用的核心，原计划完全缺失。
> 输入对照：用户的 `OptimizationIntent`（`goals` + `hard_constraints`，见 `src/models/tunable.py:214`）vs 优化产出（`SimulationDB` 中的 Pareto 前沿）。

**文件**：`src/reporting/goal_attainment.py`（新建）；并入 `src/reporting/summary_report.py` 作为新章节。

**任务清单**：

- [ ] **H2-1** 定义 `GoalAttainment` dataclass
  - 字段：`metric: str`、`direction: str`、`target_value: float | None`、`achieved_value: float | None`、`baseline_value: float | None`（初始文件的值）、`satisfied: bool | None`（仅约束有意义）、`improvement: float | None`（相对 baseline 改善幅度/百分比）、`note: str`

- [ ] **H2-2** 实现 `evaluate_constraints(intent, db_path, session_id) -> list[GoalAttainment]`
  - 对 `intent.hard_constraints` 每条，从 Pareto 前沿最优点读取实际值，判定 `satisfied`
  - 数据缺失时 `satisfied=None` 并写明原因，**不伪装成满足**

- [ ] **H2-3** 实现 `evaluate_objectives(intent, db_path, session_id, baseline) -> list[GoalAttainment]`
  - 对 `intent.goals` 每条，给出最优点 `achieved_value`
  - 若提供 baseline（初始文件单跑结果），计算 `improvement`
  - baseline 缺失时 `improvement=None`，note 说明"未提供基线，无法量化改善"

- [ ] **H2-4** 实现 `establish_baseline(config_path, db_path) -> dict`（可选基线建立）
  - 用初始设计变量值跑一次 `run_case`（或从 DB 取初始 case），作为改善幅度的对照
  - 明确标注：无基线时改善幅度不可计算，不阻断报告

- [ ] **H2-5** 实现 `generate_goal_attainment_section(intent, db_path, config_path, session_id) -> str`
  - 输出 Markdown：约束达成表（逐条 满足/未满足/未知）+ 目标改善表（最优值 vs 基线 vs 改善%）
  - 顶部给一句总判定："X/Y 约束满足；主目标相对初始改善 Z%"

- [ ] **H2-6** 接入 `generate_summary_report`（`src/reporting/summary_report.py:803`）
  - 在现有 5 章节前插入【0. 目标达成总览】章节
  - intent 不可得时该章节降级为"未提供优化意图，无法评估达成度"，不报错

- [ ] **H2-7** 单元测试 `tests/reporting/test_goal_attainment.py`
  - mock DB + mock intent，覆盖：约束满足/未满足/数据缺失三态；有/无 baseline 的改善计算；空数据降级

**验收标准**：对 demo_case 的 DB + 预设 intent，报告新增【0. 目标达成总览】章节，能明确给出"约束是否满足 + 目标改善幅度"；缺数据时降级不报错；单测全绿。

---

## H3 — 泛化性验证（P1）

> 目标：量化系统在**非 demo_case** 文件上的表现，找出语义规则与配置生成的真实失败模式。这是第一阶段能否交给真实用户的命门。
> 不修复、只测量——本任务包产出的是"还差什么"的清单，修复归入后续迭代。

**文件**：`scripts/scan_generalization.py`（新建）；报告落 `reports/phase1/generalization.md`。

**任务清单**：

- [ ] **H3-1** 收集 2–3 个非 demo_case 的 Aspen 文件
  - 优先用真实文件；不足时构造覆盖不同单元类型（如含 Flash/Pump/Compr/RStoic 而非仅 RADFRAC）的文件
  - 记录每个文件的单元类型清单

- [ ] **H3-2** 对每个文件跑 `discover_tunables_tool`，采集指标
  - `semantic_coverage` 数值
  - 落到 `confidence=high/medium/low` 的变量分布
  - `scan_warnings` 内容

- [ ] **H3-3** 对每个文件跑 `build_config_draft`（配预设 intent），采集失败模式
  - 目标节点找不到（`_map_goal_to_objective` 返回 None）的次数与原因
  - 约束映射失败的次数与原因
  - 边界为 None（需用户手填）的变量比例

- [ ] **H3-4** 产出 `reports/phase1/generalization.md`
  - 按文件汇总覆盖率与失败模式
  - **关键交付**：语义规则缺口清单——哪些单元类型/字段还没规则覆盖，按出现频次排序，作为后续补 `configs/aspen_semantics/` 的依据

**验收标准**：至少 2 个非 demo_case 文件完成扫描；产出量化的覆盖率数据与排序后的语义规则缺口清单。允许覆盖率低——本任务的价值是**暴露真相**，不是达标。

---

## H4 — Vue 前端联调（P1）

> 目标：把 `graph.py` 状态机的真实数据接入已有 Vue 3 + FastAPI 前端，替换现有 mock 数据，让用户在本地浏览器界面完成"确认→优化→看报告"闭环。
>
> **架构约束（已确认）**：Aspen Plus 通过 Windows COM 调用，前端、FastAPI 后端、Aspen Plus 三者必须运行在同一台 Windows 机器上，用户通过 `localhost` 访问前端。这是第一阶段的既定约束，不需要解决远程访问问题。
>
> **不做 CLI**：CLI 入口已从计划中移除。用户交互入口唯一为 Vue 前端（`frontend/`）+ FastAPI 后端（`mock_backend/`，联调后不再是 mock）。
>
> 前提：P0-FIX 已修复导入断裂；H1 真实链路已通；H2 达成度章节已并入报告；`graph.py` B2 状态机已实现。

### H4-0 — HITL 交互契约（前置）

> `graph.py` 的两个 HITL 节点（`human_confirm_node`、`human_decide_node`）必须暴露一套与前端无关的交互契约，FastAPI 层只是这套契约的 HTTP/SSE 搬运工。

**文件**：`src/agents/hitl_protocol.py`（新建，承载契约 dataclass）。

- [ ] **H4-0-1** 定义"暂停态"载荷 `HitlPrompt` dataclass
  - 字段：`phase: str`（confirming/deciding）、`questions: list[str]`、`config_draft_summary: dict`、`pareto_summary: dict | None`、`pending_bounds: list[dict]`（待确认变量+建议范围+置信度）、`options: list[str]`（如 continue/adjust/done）

- [ ] **H4-0-2** 定义"恢复态"载荷 `HitlResponse` dataclass
  - 字段：`confirmed_bounds: dict[str, list[float]]`、`decision: str | None`、`edited_intent: str | None`

- [ ] **H4-0-3** 约定状态机暂停/恢复 API
  - `start_session(aspen_file, intent_text) -> session_id`
  - `resume_session(session_id, HitlResponse) -> HitlPrompt | FinalResult`
  - 暂停态持久化方式（LangGraph MemorySaver checkpoint）在此定一次

**验收标准**：`HitlPrompt`/`HitlResponse` dataclass 可独立导入；start/resume API 有 docstring 说明契约边界。

### H4-1 — FastAPI 后端替换 mock

> 现有 `mock_backend/` 的 SSE 流是脚本写死的，需替换为真实 `graph.py` 状态机驱动。

**文件**：`mock_backend/`（改造，不重写）；改造完成后可考虑重命名为 `backend/`。

- [ ] **H4-1-1** 新增 `/api/session/start` 端点
  - 接收 `aspen_file`（本地绝对路径）+ `intent_text`
  - 调 `start_session()`，返回 `session_id`

- [ ] **H4-1-2** 改造 `/api/stream/{session_id}` SSE 端点
  - 替换 mock 脚本，改为从 `graph.py` checkpoint 实时读取状态推送
  - 优化进行中：推送每次仿真完成事件（iteration、objective values）
  - HITL 暂停时：推送 `HitlPrompt` payload，前端切换为确认界面

- [ ] **H4-1-3** 新增 `/api/session/{session_id}/resume` 端点
  - 接收前端提交的 `HitlResponse`，调 `resume_session()` 恢复状态机

- [ ] **H4-1-4** 保留现有 mock 数据路由不删除
  - 加 `?mock=true` 参数切换，便于前端开发调试时不依赖真实 Aspen

**验收标准**：三个新端点可通过 `curl`/Postman 手动验证；SSE 流在真实优化运行时有事件推送；HITL 暂停/恢复状态机状态不丢失。

### H4-2 — Vue 前端接入真实数据

> 现有 `/optimization` 页面的 Pareto 图和 HV 曲线已有组件，需把数据源从 mock 切换到真实 SSE 流。

**文件**：`frontend/src/` 下相关组件（改造，不重写）。

- [ ] **H4-2-1** `/optimization` 页面：新增"发起优化"表单
  - 输入：Aspen 文件本地路径 + 自然语言意图
  - 提交后调 `/api/session/start`，拿到 `session_id`，订阅 SSE 流

- [ ] **H4-2-2** HITL 确认界面：变量边界确认表单
  - 收到 `HitlPrompt`（phase=confirming）时弹出
  - 逐条展示低置信度变量 + 建议范围，用户可编辑后提交
  - 提交后调 `/api/session/{id}/resume`

- [ ] **H4-2-3** HITL 决策界面：continue/adjust/done 按钮
  - 收到 `HitlPrompt`（phase=deciding）时展示分析摘要 + 达成度概览
  - 三个按钮对应三种 `HitlResponse.decision`

- [ ] **H4-2-4** 报告展示
  - 优化完成后从后端拉取 `summary_report`（含 H2 达成度章节）
  - 在 `/optimization` 页面内嵌展示 Markdown 报告

- [ ] **H4-2-5** 保留 mock 模式切换
  - 界面上加一个"演示模式"开关，切到 mock 数据，用于展示/汇报时不依赖 Aspen

**验收标准**：用户在浏览器发起一次完整调优（本地 Aspen 文件 → 意图 → 确认变量 → 优化 → 决策 → 看报告），全程不碰命令行；Pareto 图实时更新；报告含达成度章节；mock 模式可切换。






---

## H5 — 鲁棒性压测（P2）

> 目标：验证异常路径不崩、且给用户可读提示。真实优化是长流程，异常处理只在 mock 下验证过。

**文件**：`tests/integration/test_phase1_robustness.py`（新建）。

**任务清单**：

- [ ] **H5-1** 错误边界用例：给一个下界>上界 / 超出物理合理范围的边界 → 断言 `validate_config_tool` 拦截或优化阶段给可读错误
- [ ] **H5-2** 无可行点用例：约束设到不可能满足 → 断言 feasibility gate 生效、状态机走到 done 且 `termination_reason` 明确，不假装成功
- [ ] **H5-3** Aspen 不收敛用例：构造易不收敛工况 → 断言失败 case 被正确标记（不污染可信数据），报告失败诊断章节如实反映
- [ ] **H5-4** adjust 回路用例：模拟用户在 `human_decide_node` 选 adjust → 断言回到 onboarding 重新解析意图，迭代计数与终止逻辑正确
- [ ] **H5-5** 超时预期：依据 H1-3 记录的真实耗时，为长流程设定合理超时与用户提示

**验收标准**：以上异常用例全部"优雅失败"——不崩溃、不伪装成功、有可读提示；adjust 回路与最大迭代终止逻辑在真实/半真实环境下验证通过。

---

## 第一阶段达标判定（Definition of Done）

满足以下全部，方可宣布第一阶段目标达成：

1. **H1 通过**：demo_case 真实端到端链路跑通，有运行记录。
2. **H2 通过**：报告能明确回答"约束是否满足 + 目标改善幅度"。
3. **H3 完成测量**：至少 2 个非 demo_case 文件的覆盖率与失败模式已量化，缺口清单已产出（不要求覆盖率达标，但要求真相清晰）。
4. **H4 通过**：用户能纯界面完成一次完整调优闭环。
5. **H5 通过**：核心异常路径优雅失败。

> 注：H3 的标准是"测量完成"而非"覆盖率达标"。若 H3 暴露覆盖率过低，则补语义规则成为**第一阶段与阶段二之间的过渡迭代**，而非阻断达标判定——因为第一阶段对"用户文件"的承诺，本就以"已有语义规则覆盖的单元类型"为隐含边界，这个边界需要在交付时如实告知用户。

---

## 之后：迈向阶段二的方向（仅定向，不展开）

第一阶段达标后，下一个台阶**不是**立刻做自主流程合成，而是从"调操作参数"过渡到"有限的结构性改进建议"：

- agent 不只调参数，还能**建议结构改动**（如热集成加换热器、调整分离序列），但仍由用户确认、仍走现有 Aspen driver。
- 这对应 `src/workflows/build_process.py` 第一次被赋予真实职责。
- 前置：阶段一的语义规则库足够稳（H3 缺口基本补齐）+ `src/knowledge/` 开始有结构化的"工艺改进模式"可检索。

阶段二的细化，待第一阶段收尾验收（H1–H5）通过后再行规划，避免重蹈"计划标 [x] 但工作树失真"的覆辙。

---

## H 系列任务编排计划

> 前提：`DEVELOPMENT_PLAN.md` 中 A / B / C 全部任务真正完成（代码可导入、测试全绿、无 collect error）。

### 依赖关系图

```
H1（真实端到端单跑）
    │
    ├──────────────────────┐
    ↓                      ↓
H2（目标达成度评估）    H3（泛化性验证）
    │
    ↓
H4-0（HITL 契约定型）
    │
    ├──────────────────┐
    ↓                  ↓
H4-1（FastAPI 改造）  H4-2（Vue 前端接入）
    └────────┬─────────┘
             ↓
           H5（鲁棒性压测）
```

### 并行规则

**H1 必须最先、单独跑完。** 它是所有后续任务的事实基础——真实链路不通，H2 的达成度数据无法验证、H4 接入没有意义。H1 同时会产出缺陷清单，H2/H3 开发时需要参考。

**H2 ∥ H3 可并行**，在 H1 完成后同时放出：

| 任务 | 并行理由 |
|------|---------|
| H2 目标达成度评估 | 纯新模块（`goal_attainment.py`），只读 SimulationDB，不依赖 H3 |
| H3 泛化性验证 | 测量任务，只跑 `discover_tunables`，只需要 Aspen 文件，不依赖 H2 |

**H4-0 → H4-1 ∥ H4-2**，在 H1 + H2 都完成后放出：
- H4-0（HITL 契约）很小，只定两个 dataclass，完成后 H4-1 与 H4-2 可立即并行。
- H4-1（FastAPI）与 H4-2（Vue）并行时，H4-2 可先用 mock API 开发，H4-1 完成后再联调。
- H4 必须等 H2 完成，因为前端报告展示需要包含达成度章节。

**H5 必须最后**，H4 联调收尾、前端稳定后才有意义跑异常压测。

**H3 不阻塞 H4**。H3 是测量任务，产出语义规则缺口清单；补规则属于后续迭代，不卡 H4 进行。

### 时间线参考

| 阶段 | 任务 | 备注 |
|------|------|------|
| Week 1 | **H1** | 可能需要调试真实链路问题，预留充足时间 |
| Week 2 | **H2 ∥ H3** | H2 写代码；H3 跑扫描，速度取决于手头 Aspen 文件数量 |
| Week 3 | **H4-0 → H4-1 ∥ H4-2** | H4-0 一天内完成；H4-1/H4-2 并行开发 |
| Week 4 | **H4 联调收尾 → H5** | H4-1/H4-2 联调；H5 压测 |

### 不能并行的约束（硬依赖）

- H1 → H2：H2 需要真实链路产出的 DB 数据做验证
- H1 → H4：前端接入的是真实 `graph.py` 流程，H1 不通则 H4 接入无意义
- H2 → H4：前端报告页需要达成度章节，H2 未完成则 H4-2 报告展示不完整
- H4 → H5：压测依赖完整链路与前端都稳定

### 派发 agent 的建议

每个任务包独立派发，不混合：
- **H1**：编排脚本 agent，调用已有工具，不写新模块
- **H2**：新模块 agent，只接触 `src/reporting/`，不碰其他层
- **H3**：测量 agent，只运行脚本、产出报告，不改代码
- **H4-0**：设计 agent，只写 `src/agents/hitl_protocol.py`，不改 `graph.py`
- **H4-1**：后端 agent，只改 `mock_backend/`
- **H4-2**：前端 agent，只改 `frontend/`
- **H5**：测试 agent，只写 `tests/integration/test_phase1_robustness.py`
