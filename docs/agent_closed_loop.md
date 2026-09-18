# PAO 自主实验闭环

本版本把 Agent 接到每批仿真的决策位置。一个主决策 Agent 使用工艺事实、知识条目和历史证据生成结构化实验动作；程序校验后调用现有 Pareto 优化器和 Aspen 单次运行接口。每次结果都进入下一批决策，并保存动作效果。

## 快速开始

在 Windows、Aspen Plus、已有 LLM 环境配置下，从仓库根目录运行：

```bash
# 只验证配置；不连接 Aspen，不调用 LLM，不修改源 YAML
python -m src.main cases/demo_case_2/agent_loop_config.yaml --dry-run

# 新实验：总预算 40 次，包含复验
python -m src.main cases/demo_case_2/agent_loop_config.yaml --agent

# 同一配置中断后恢复；已结束的会话仅重建报告，不再追加实验
python -m src.main cases/demo_case_2/agent_loop_config.yaml --agent --resume
```

`--agent` + YAML 现在进入闭环，不再覆盖 YAML 的设计变量上下界。`--agent` + `.bkp` / `.apw` 保留原有接入向导；向导生成并核实 YAML 后再启动闭环。没有配置 LLM key 时使用明确标记的规则决策，完整反馈、约束、预算及持久化机制仍然运行。

也可在已有多目标 YAML 中添加 `agent_loop.enabled: true`。这样 CLI、`optimize_pareto_tool`、Coordinator 的运行优化任务都通过同一个 `optimize_pareto_case()` 入口进入闭环。

默认 checkpoint 位于结果数据库同目录的 `agent_checkpoint.db`。示例指定 `cases/demo_case_2/output/agent_epsd.db`。路径已存在时必须显式 `--resume`；新实验使用 `--checkpoint path/to/new_run.db`。不要将 checkpoint 路径设成 SimulationDB 的同一个文件。

## 边界和目标的含义

| 配置 | 作用 |
|---|---|
| `design_variables.lower_bound/upper_bound` | 不可由 Agent 改写的工程硬边界 |
| derived 的 `lo_frac/hi_frac` | 虚拟变量的硬边界，实际写入按原派生公式计算 |
| 顶层 `search_region` | 本轮软搜索区，键可用变量 name 或 Aspen 路径；可扩张、收缩、移动或固定到单点 |
| `agent_loop.max_step_fraction` | 单次候选距离基准点的最大变化，按硬边界宽度归一化；整数变量至少允许一步 |
| `agent_loop.objective_targets` | 可选的原始目标值阈值；方向沿用 `minimize`；同一个可行工况必须同时达标 |

旧配置已经被写窄的上下界无法凭空恢复。本次示例继承原有硬边界，没有把旧注释中的宽范围自动当作设备允许范围。如需扩大硬边界，应先核实设备、流程和模型适用条件，再由工程师修改配置并开始新会话。

```yaml
search_region:
  T1_RR: [0.7, 1.0]
  T1_NSTAGE: [40, 60]

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
  objective_targets: {}  # 填写实际业务阈值；空字典不表示自动达标
  checkpoint_path: output/agent_checkpoint.db  # 相对当前 YAML 所在目录
  process_facts:
    description: 在这里填写已确认的体系、物性方法、设备和操作规定
```

源配置和目标/约束函数不会被 Agent 修改。改变物性、反应动力学、产品纯度、硬边界或目标定义属于新的优化问题，不能在同一 checkpoint 中偷偷替换。

## 决策流程与职责

1. 运行配置中的基准点，记录实际写入、求解状态、目标和约束。
2. 每批结束或连续失败触发主 Agent。模型输入包含近期证据、上次动作的效果、剩余预算和带来源的工艺知识。
3. Agent 输出 `ActionPlan`，确定性校验器检查字段、证据 ID、变量类型、边界和预算。
4. 在软区域与单步限制内由 `ParetoSearchSession.ask()` 生成候选，可行性分类器按需筛选。`probe` 可指定完整向量；远处目标通过受限中间点接近。
5. 依赖修复、整数处理和派生映射后，按**实际 Aspen 输入**去重；`repeat: true` 只对显式 probe 生效，用于有意重复诊断。
6. 仿真结果通过 `tell()` 更新优化器，成功和失败证据进入 checkpoint。
7. 达标、预算耗尽、决策次数耗尽、停滞或 Agent 请求结束时，对一个代表可行工况从 reset 状态复验。

`continue` 继续采样；`set_region` 修改软区域；`probe` 指定候选；`stop` 请求停止。`expected_effect` 只能是 convergence / feasibility / objective / information。动作不允许包含任意代码、任意 Aspen 写入路径、目标或约束改动。

主 Agent 的文本响应不会直接进入执行层。程序先提取其中唯一的 JSON 对象，再检查 `ActionPlan` 协议：
`set_region` 只能填写 `search_region`，`probe` 只能填写完整 `candidate`，而 `continue`/`stop`
必须将这两个映射留空。空响应、Markdown 围栏、前后缀文本或上述字段冲突会触发一次带纠正提示的重试；
重试仍失败时记录拒绝原因并使用确定性的规则动作。该动作仍经过同一边界、证据和预算校验。
协议拒绝不是实验停滞，因此不会增加 `stagnation_count`；真正执行的实验批次仍会按 `stagnation_batches`
计数。checkpoint 的每个动作效果包含 `stagnation_counted` 和当时的 `stagnation_count`，便于审计。

`initialization: previous` 只在当前进程的 Aspen 确实保有上次收敛状态时生效。首次运行、恢复会话、失败或驱动重建后强制 reset。reset 使用配置中的固定初始化值，不复用上一轮 inherit 的循环流估值；复验总是 reset。

本闭环自己管理软区域、步长、停止和预算。因此旧整轮工作流的 Phase 0 DOE、敏感度探针、ThawScheduler、TrustRegion、boundary_refine 与 early_stopping 不会同时启动。已复用的是目标/约束计算、单次仿真、预检查、变量映射、可行性分类器和代理模型。避免两个控制器同时修改同一个搜索区域。

## 已修复的数据流问题

- 收敛但违反约束、且目标有效的样本，现在传给 BoTorch 的目标与约束模型；Pareto 排名仍只接受可行样本。
- 缺失或非有限约束不伪装成零余量；训练期约束名称变化会拒绝。
- constrained qEHVI 的已有 Pareto 分区只使用可行点，不能把目标优异的不达标点当作可行前沿。
- skopt 模型用全局坐标学习历史，在当前软区域内对候选池计算采集函数。改变局部范围不会再导致旧观测因越界被拒绝。
- 每个观测分别保存 `optimizer_inputs` 与真正的 Aspen 输入，恢复时不会丢掉 derived 的虚拟变量。

## 确定性分析技能

闭环每次构造决策快照时，`src.agents.closed_loop.analysis.build_analysis_report()` 会从当前
`ProcessCase` 历史生成版本化的 `analysis_report`。分析代码不调用 LLM，也不连接 Aspen，
因此恢复 checkpoint 或离线重放时可以得到同样的证据。报告包括：

- `data_quality`、`convergence`：样本数、状态分布、近期收敛率和连续失败数；
- `objectives`、`constraints`：目标趋势、最优工况、约束违反率、最大违反和裕量；
- `pareto`、`metrics`：可行 Pareto 前沿、超体积、近期可行率和收敛率；
- `sensitivity`、`failures`：变量敏感性排序、有效样本数、可靠性标记和失败模式。

每个决策保存 `before_analysis`；动作完成后保存 `after_analysis` 与
`analysis_effect`。后者只报告观察到的指标变化，并明确标记为相关性证据，不把一次动作
自动解释成因果结论。敏感性结果在样本不足时会标记 `reliable: false`，主 Agent 不应将
该分数当作经过验证的物理规律。

通用 Agent 或离线审查还可以调用 `analyze_closed_loop_tool`：它从 SimulationDB 按会话、
迭代范围或 tags 过滤历史工况，返回同一结构化 JSON，不需要 Aspen COM。该工具已经通过
`get_agent_tools()` 和 `RealToolRunner.analyze_closed_loop()` 注册；它是数据分析工具，不是
另一个自主决策 Agent。默认只把可行工况用于正式 Pareto，`include_infeasible=true` 仅用于
约束松弛/可行域诊断。

BoTorch 联合模型目前仍要求目标向量完整才能接收该条约束向量；仅有部分目标/约束的数据保留在证据中，但不会被伪造或补零用于联合训练。Aspen 未收敛的输出不进入目标回归模型。

## 知识和经验

默认条目在 `configs/process_knowledge/distillation.yaml`，每条包含 id、适用条件、建议和来源。可通过 `agent_loop.knowledge_files` 指定其他 YAML 文件。第一版采用小型可审查知识库，不依赖向量数据库或模型微调。

这是工艺知识入口，不是经过完整认证的通用化工专家系统。具体组分、物性方法、关键二元参数和设备限制应放在 `process_facts`，未知信息明确标记为未知。收敛错误关键词匹配仅用于生成假设。

动作效果按批次记录：引用哪些工况、期望改善哪个指标、执行后可行率/收敛率/超体积怎样变化。指标改善仅说明相关性，不自动声称证明了因果。知识条目不会被 LLM 自动覆写。

## 持久化与结果

checkpoint SQLite 是闭环的权威记录；SimulationDB 是现有报告和界面的结果投影。记录参数、初始化、预算、证据、动作、动作效果、固定 HV 参考点和随机数状态。模型文件与自定义目标模块内容参与会话指纹；配置或知识改变会拒绝恢复。

每次调用 Aspen **之前**持久化 pending 并扣预算。进程在仿真中途退出时，该次结果视为未知、保留已用预算，不自动重复运行。恢复会话会重建代理模型，保留已经完成的 case_id 和证据；不持久化 COM 对象。BoTorch/底层数值库不保证跨环境逐位一致。

Windows/POSIX 文件锁保证一个 checkpoint 同时只有一个控制器；进程崩溃后锁由操作系统释放。每次尝试均占用总预算，包括失败、预检查拒绝、有意重试和复验。

会话结束后生成同名 `.report.json` 和 `.report.md`：

| result | 含义 |
|---|---|
| `target_verified` | 同一个代表工况达到指定阈值，并通过独立复验 |
| `completed_verified` | 代表工况复验通过；不表示指定目标已达到或全局最优 |
| `verification_failed` | 重跑不达标、不能收敛或目标偏差超出容差 |
| `no_feasible_solution` | 当前预算内未找到可行解 |
| `unverified` | 例如驱动不可用，未完成复验 |

复验只验证一个代表工况，完整 Pareto 前沿仍作为候选结果。未达到目标和“根本不存在可行解”不能混为一谈。

## LangGraph 接入

给 `PAOGraphState` 设置 `agent_loop={"enabled": True, ...}`，可另传 `agent_checkpoint_path` 和 `agent_resume`。保留最初的草案确认；确认之后内部自主迭代，最终分析后直接结束，不再每个小批次暂停等待人工。`llm_config` 传给主决策 Agent，不写入 checkpoint。

仓库当前没有跟踪 README 所述的 backend/frontend 源码；本次提供的是核心、CLI、工具与图节点集成，没有虚构已完成的 Web 控制面板。

## 验证

Linux 可运行不含 Aspen 的测试：

```bash
python -m pip install pytest PyYAML numpy scikit-optimize python-dotenv langgraph langchain-core
python -m pytest -q
```

安装 torch/botorch 后，额外执行真实联合 GP 与采集函数构建测试。不安装时该项明确跳过。

真实验收建议使用同一个 `.bkp` 副本、同一硬边界、目标/约束和初始点，在相同总调用预算下比较原优化器与本闭环；记录可行率、首次达标用时、最终目标、超时数及复验结果，并用多个随机种子重复。比较中要把原工作流额外的 Phase 0/探针调用也计入预算。本仓库内的合成测试只证明软件行为，不证明真实化工案例的优化增益。
