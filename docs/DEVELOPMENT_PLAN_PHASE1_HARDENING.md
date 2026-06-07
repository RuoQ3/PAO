# PAO 项目开发计划 — 阶段一收尾验收（续章）

> 版本：v1.0
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
| **H4 前端/API 联调** | 用户在界面上完成确认→优化→看报告 | HITL interrupt 接入现有前端 | P1 |
| **H5 鲁棒性压测** | 异常路径不崩、有可读提示 | 异常用例集 + 通过记录 | P2 |

依赖关系：H1 → (H2 ∥ H3) → H4 → H5。H1 是其余一切的前置（真实链路不通，后面都无从谈起）。

---

## 边界（沿用原计划，本续章不放松）

- 不碰流程拓扑，只调操作参数。
- 不让 LLM 操作 COM；`discover_tunables.py` 仍是唯一允许开 Aspen 的新路径，只扫不跑。
- 不改 `aspen_driver/`、`optimize_pareto_case.py`、`run_case.py`、`SimulationDB`、`pareto_config.yaml` schema。
- 本续章新增能力优先落在 `summary_report.py`、新增评估模块、测试与脚本层，不动稳定核心。

---

## H1 — 真实端到端单跑（P0）

> 目标：用真实 Aspen + 真实 LLM + 真人确认，把 demo_case 完整链路跑通一次，把"mock 通过"升级为"真实通过"。这是验收第一阶段的最低事实门槛。
> 前提：P0-FIX 已修复导入断裂；A/B/C 代码完成。

**文件**：`scripts/run_phase1_e2e.py`（新建，编排脚本，非生产代码）；运行记录落 `reports/phase1/`。

**任务清单**：

- [ ] **H1-1** 编写端到端编排脚本，串联真实链路
  - `discover_tunables_tool`（真实开 demo_case `.bkp`，只扫）
  - `parse_intent_from_text`（真实 LLM，意图取自一条预设自然语言）
  - `build_config_draft` → `to_yaml_dict()` → 写临时 YAML
  - `validate_config_tool`（无 fatal 才继续）
  - `optimize_pareto_tool`（真实 Aspen 跑一轮，迭代次数可临时调小以控时长）
  - `generate_summary_report`
  - 每步打印耗时与状态，全程不静默吞错

- [ ] **H1-2** 人工确认环节先用脚本注入模拟（不阻塞 H1，真正的前端确认留给 H4）
  - 用预设的 user feedback dict 模拟"用户确认边界"，走 `apply_user_feedback`

- [ ] **H1-3** 记录真实运行结果到 `reports/phase1/e2e_demo_case.md`
  - 总耗时、各步耗时、Pareto 前沿大小、成功/失败 case 数
  - 真实 LLM 意图解析是否成功、是否走了降级路径
  - 链路中断点（若有）

- [ ] **H1-4** 产出缺陷清单 `reports/phase1/e2e_defects.md`
  - 真实环境暴露而 mock 未暴露的问题逐条记录（不在本任务包内修复，归入后续）

**验收标准**：链路无中断产出一份 `summary_report`；运行记录与缺陷清单落盘；明确记录真实耗时量级（用于 H5 设定超时预期）。

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

## H4 — 用户交互前端联调（P1）

> 目标：让用户真正能通过一个前端完成"确认→优化→看报告"闭环。前端有两种形态——**Web（Vue/FastAPI）** 与 **CLI 向导**——二者共用 `graph.py` 同一套 HITL 交互契约，互为并列，不重复实现状态机。
> 前提：P0-FIX 已修复导入断裂（否则任何 agent 入口 import 即崩）；H1 真实链路已通；H2 达成度章节已并入报告。

### H4-0 — 统一 HITL 交互契约（前置，Web 与 CLI 共用）

> `graph.py` 的两个 HITL 节点（`human_confirm_node`、`human_decide_node`）必须暴露一套**与前端无关**的交互契约。Web 和 CLI 都只是这套契约的渲染器 + 输入回灌器。

**文件**：`src/agents/graph.py`（定义契约数据结构）；建议新增 `src/agents/hitl_protocol.py` 承载契约 dataclass。

- [ ] **H4-0-1** 定义"暂停态"载荷 `HitlPrompt` dataclass
  - 字段：`phase: str`（confirming/deciding）、`questions: list[str]`、`config_draft_summary: dict`、`pareto_summary: dict | None`、`pending_bounds: list[dict]`（待确认变量+建议范围+置信度）、`options: list[str]`（如 continue/adjust/done）
- [ ] **H4-0-2** 定义"恢复态"载荷 `HitlResponse` dataclass
  - 字段：`confirmed_bounds: dict[str, list[float]]`、`decision: str | None`、`edited_intent: str | None`
- [ ] **H4-0-3** 约定状态机暂停/恢复 API：一个 `start_session()` 与一个 `resume_session(session_id, HitlResponse)`，对 Web/CLI 一致
  - 暂停态持久化方式（LangGraph checkpoint 或会话态）在此定一次，两个前端复用

**验收标准**：契约 dataclass 与 start/resume API 定型；有一份说明文档/docstring 明确"前端只读 HitlPrompt、只回 HitlResponse"。

### H4-API — Web 前端联调

> 接入项目记忆 `project_pao_frontend` 中的 FastAPI/Vue。

- [ ] **H4-API-1** API 层把 `HitlPrompt` 经 SSE/轮询推给前端，`resume_session` 接收 `HitlResponse`
- [ ] **H4-API-2** 前端：确认边界表单 + 决策按钮，复用现有 advisor chat / Pareto 图组件
- [ ] **H4-API-3** 一次界面驱动完整走查：发起→看待确认问题→确认→优化→看报告（含 H2 达成度章节）

**验收标准**：用户纯界面完成"确认→优化→看报告"；HITL 暂停/恢复不丢上下文。

<!-- PLACEHOLDER-H4CLI -->



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



