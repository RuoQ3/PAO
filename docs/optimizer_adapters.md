# 第四批：优化器接入与离线验收

> 后续更新：已接入 GA、对话算法优先和审批配置覆盖，并安装目标 pymoo 发行包完成真实数学测试。
> 当前行为见 [对话优化与算法选择](conversation_optimization.md)。下文的 GA 未实现及 pymoo 缺失描述为本批历史记录。

本批已经落地通用问题模型、候选/观测协议、能力注册、BO 迁移桥、PSO/NSGA-II 适配代码，以及 YAML、项目向导、Gateway、预算账本和证据接线。**PSO/NSGA-II 尚未完成真实库验收**：本机 Python、项目 venv 和 pip 离线缓存均没有 pymoo；本任务禁止网络，因此没有下载、安装或请求任何外部服务。两项真实库测试明确跳过，不能据此称这两个后端已验证可用。

## 已核对的前三批调用链

* `plans.execute_optimization → optimize_case → make_legacy_bo → LegacyBOBridge → SurrogateOptimizer → skopt.Optimizer`。LHS DOE、成功样本门槛、EI/UCB(内部 LCB)/PI、RF/ET/GBRT、失败内部惩罚、去重、可行性候选过滤仍由原 BO 流程拥有。
* `optimize_pareto_case → make_legacy_bo → MultiObjectiveBayesianOptimizer`：GP/RF/ET/GBRT 每次随机权重标量化、重建代理，为 ParEGO 策略；qEHVI/NEHVI 则进入 `BoTorchMOOptimizer`，使用 qLogEHVI/qLogNEHVI。修正了原带约束 NEHVI 错建 qLogEHVI 的身份问题。
* `pareto.fast_non_dominated_sort` 只做结果排序，未注册为 NSGA-II。NSGA-II 适配代码调用 pymoo 的完整 NSGA2，包括选择、SBX、PM、种群生存选择。
* `ActionGateway` 校验审批快照、配置/模型 hash，创建 `_Grant`；`run_case` 的 `evaluated_case` 在写入前调用 `grant.evaluation`，校验授权、Input 路径、硬边界、类型、依赖关系、总预算，再持久预占。失败、重试、探测、复验均不退款。`ExecutionUnit` 继续管理一次性模型副本、超时、清理与结果未知状态。
* 第二批 `ProcessCase` 的仿真有效性、指标可用性、约束状态、`_lifecycle`、`_source` 以及执行证据不被优化层覆盖。第三批向导、角色工具和配置解析接入同一算法注册表，推荐不构成用户选择。

## 调用关系与公共接口

```text
YAML / ProjectOnboarding（用户明确选择）
  → load_optimize_config → OptimizationProblem + AlgorithmSpec
  → build_optimization_plan（完整参数、依赖版本、问题/模型/配置/实现 hash）
  → ActionGateway.preview / approve / execute
  → execute_optimization（preflight，在开模型前拒绝缺依赖）
      BO → 原 DOE/BO 工作流 → LegacyBOBridge → 原代理实现
      PSO / NSGA-II → optimize_population → run_optimizer
          → optimizer.ask(remaining) → 完整候选批次
          → 逐个 run_case → evaluated_case → 授权/边界/预算/运行/证据
          → Observation → optimizer.tell(完整真实结果)
  → SimulationDB / UnifiedResult / execution_evidence / 必要专业 Agent
```

`problem.py` 定义变量稳定 ID/路径/类型/单位/硬界、固定及派生定义、目标方向/单位/有效性、约束原始定义/正尺度/残差口径、授权范围、总预算和问题身份。目标原值留在 `Observation.objectives`，内部最小化向量与无量纲 `G=residual/scale` 单列在 `algorithm_transform`；对外最大化方向不改变。

`Optimizer` 协议含初始化（注册工厂）、`ask` 批次候选、`tell` 完整反馈、`stop_reason`、`stop`、`state_dict/load_state_dict`、`capabilities`。`BOAdapter` 提供新类型接口；`LegacyBOBridge` 保留既有点 API 以迁移原工作流，属于过渡接口，并非所有旧 BO 内部已经重写成新协议。原多目标私有类名保留兼容别名，新调用方使用公开类。`search_session` 不再读取优化器私有 RNG/BoTorch 成员。

## 支持矩阵

| 策略 | 目标 | 搜索变量 | 约束方式 | 精确恢复 | 动态区域 | 后端与本次验证 |
| --- | --- | --- | --- | --- | --- | --- |
| BO，GP/RF/ET/GBRT | 1 | 连续、skopt 原生整数 | 可行观测；旧流程显式内部惩罚/可选分类器 | 禁止 | 原 BO 支持 | skopt 0.10.2，真实 GP 接口测试通过 |
| ParEGO + 上述代理 | ≥2 | 连续、skopt 原生整数 | 同上 | 禁止 | 原 BO 支持 | skopt，真实标量化/拟合/局部 ask 测试通过 |
| qEHVI/NEHVI | ≥2 | 连续；统一注册入口拒绝整数 | 可信目标/约束 margin GP；数据不足仍属初始化策略 | 禁止 | 原 BO 支持 | torch/botorch/gpytorch；已有真实 BoTorch 测试通过 |
| 显式 random | 1 或多个 | 连续、整数均匀抽样 | 记录与最终可行筛选 | 新协议数学状态 JSON 可精确继续 | 支持 | Python 标准库，RNG/预算连续性测试通过 |
| PSO | 1 | 连续 + 显式固定值 | 不支持工程约束 | 禁止 | 不支持 | 目标 pymoo 0.6.1.5；未安装，真实库未验收 |
| NSGA-II | ≥2 | 连续 + 显式固定值 | pymoo 可行性优先；显式正尺度残差 | 禁止 | 不支持 | 同上，真实库未验收 |
| GA / GWO（灰狼）/ NSGA-III | 未接入 | — | — | — | — | `not_connected`，绝不映射成其他算法 |

派生变量可以被问题描述记录；真实执行仍按第一批策略拒绝，直到提供明确物理写入包络。退化搜索区间必须显式改为固定变量，不自动挪动边界。种群不支持整数，未提供取整适配。NSGA-II 首版只接受 `<=`/`>=`，要求 `scale`，严格不等式和等式容差未接入。

`/project algorithms`、会话能力清单及 `optimization_advisor.list_optimizers` 查询同一支持矩阵；`available` 表示依赖元数据条件满足，实际导入/初始化/许可证仍须 preflight，不能当作已经完成现场验收。

## 库选择与核实边界

保留已经实际安装、测试的 scikit-optimize 0.10.2（发行元数据 BSD）和 BoTorch（安装版本 0.18.1，发行许可证 MIT）。没有从零实现标准 PSO 或 NSGA-II。

pymoo 的选择理由是其外部 ask/tell 和统一 Population F/G 接口可以让库只做数学更新。代码目标是 `pymoo.algorithms.soo.nonconvex.pso.PSO`、`pymoo.algorithms.moo.nsga2.NSGA2`、`Problem`、`SBX`、`PM`、`Evaluator`。**当前无本地发行包，不能声称已经核实这些接口、Apache-2.0 许可证或最新维护状态**。这是本批未完成的验收项。生产 preflight 会检查精确版本、安装发行包许可证文件的 Apache/2.0 标记、实际导入、构建和 setup；故障直接停止，不安装、不改算法。

得到本地正式 wheel/源码后，先人工核实来源、发行版本/维护记录、许可证全文及传递依赖，再显式安装 `requirements-optimization.txt` 中所选库并运行真实库测试；这不是生产 Agent 的能力。基础 `requirements.txt` 保留 skopt，BoTorch 与 pymoo 移到可选文件；更新环境时原 qEHVI/NEHVI 用户须显式保留 BoTorch 依赖。

## 参数与选择

完整可解析示例见 `configs/optimizer_bo.example.yaml`、`optimizer_pso.example.yaml`、`optimizer_nsga2.example.yaml`。示例模型路径需换成用户已有模型；示例无权授权运行。

BO 保持 `type: bayesian` 或 `pareto_bayesian`、`surrogate_model` 的现有格式，YAML `n_iterations` 仍表示 DOE **之后** 的次数。PSO 使用 `type: pso`，NSGA-II 使用 `type: nsga2`，两者后端为 `pymoo`、参数置于 `parameters`，不使用 `surrogate_model: PSO`。

种群默认值是明确的 PAO 草稿政策：`pop_size=20, generations=10`；PSO `w=.9,c1=2,c2=2,max_velocity_rate=.2`，固定 `adaptive=false,perturb_best=false`；NSGA-II `crossover_probability=.9,crossover_eta=15,mutation_eta=20,mutation_probability=.1`，PM 每个个体调用概率 1，`.1` 是每变量概率。不是最优调参结论；未核实库发行包，不能称这些数值已经与库默认源代码逐项核对。统一采用库的 `FloatRandomSampling` 初始化；默认与用户覆盖的最终合并值、SBX/PM、原生有界运算策略、去重政策均进入审批身份。

向导中明确选择算法、后端、参数、种子，再确认/保存项目；配置确认不是运行审批。依赖缺失可静态解析 YAML 及预览计划，向导最终 validate 和执行 preflight 会拒绝启动。

## 预算、反馈、停止与资源

初始种群是第一代，`generations` 包括初代。每次开代前至少剩 `pop_size`；不足则 `insufficient_budget_for_population`，不评估半代、不补齐预算、不伪造个体。库返回空/不足种群有限停止；超额种群或非法候选报故障。评估始终串行，批量不改变 Aspen 并发资源数。

真实预算继续由 Gateway 持久账本计数，`pop_size × generations` 仅是所有代完整且每次评估一次的上界估算。重试、探测、复验若另行执行仍走同一个 grant，每次消耗额度；本批种群不自动重试、复验或探测。每点还受 deadline、失败限额和模型资源管理约束。

NSGA-II 的可信但不可行点反馈真实 F 和 `G=residual/scale`；可行解/Pareto 报告只包含可行证据。PSO 不接受约束配置。任一点仿真失败/指标缺失，种群立即停止：剩余个体记为未执行，不向库传假 fitness，也不做生存选择。未知/超时停止分发。库初始化、ask、tell 故障都停止并记原因，不切换 random/BO/其他后端。

重复候选遵从算法策略：NSGA-II 库内部去重；PSO 保留粒子重复，跨代重复也真实评估并计预算。公共层不套用 BO 候选池、分类过滤、重采样、依赖修复和动态缩界。库原生有界算子是明确的算法政策，越过问题硬界的输出仍拒绝。BO 旧惩罚仅存在于内部代理更新，原始失败 case/工程指标不被改写。

## 结果、恢复与报告

统一摘要包含请求/实际算法及后端、参数/种子、版本、问题/实现身份、预算、停止原因、全部状态记录、工况 evidence ID、最优可行解或可行 Pareto 集、限制。无可行解时 `no_feasible_solution=true`，不会把违反最少的点叫作满足要求的最优点。BO 保留旧摘要，新增 `summary.unified`；种群直接返回统一摘要。`execution_evidence.optimization` 复用这些已验证事实供现有 Agent 解释，不在数学层生成工程优劣结论或自动算法竞赛。

PSO/NSGA-II、skopt 和 BoTorch 精确恢复明确禁止，快照仅用于诊断（历史/代数/待反馈候选、身份、停止原因），不包含可承诺恢复的完整后端状态，不称其为可继续检查点。只有显式 random 的新协议提供数学状态恢复：完整 Python RNG、历史、待反馈状态和预算身份 JSON；待执行/未知记录拒绝自动恢复；模型/配置/版本/实现 hash 不符拒绝。**真实 Gateway 运行恢复仍禁止**，原执行账本不会凭新优化器快照被清空/重放。旧 Agent 生产闭环入口继续禁用；BoTorch 也不再假装仅恢复外层 RNG 就可继续。

可复现范围限于相同解释器/库/问题/参数和确定评估器的数学过程；不承诺跨平台或真实 Aspen 的逐位一致。

## 文件和测试

新增 `src/optimization/{problem,registry,bo_adapter,population,configuration,execution}.py`、`src/workflows/optimize_population.py`、`tests/test_optimizer_protocol.py`、可选依赖和三个示例。修改现有 plans、BO/Pareto 工作流、search_session、配置加载、向导、角色工具、能力列表、统一 execution_evidence 和 BoTorch 约束采集分支。原工作区修改保留。

```powershell
python -B -m pytest tests -q -p no:cacheprovider --basetemp .tmp/batch4-full
python -B -m pytest tests/test_optimizer_protocol.py -q -rs -p no:cacheprovider --basetemp .tmp/batch4-acceptance
```

最终全量结果 **516 passed、2 skipped**（38.14 秒），`git diff --check` 通过，三个示例均实际解析通过。跳过项目是两个真实 pymoo 测试，调度 fake 只验证预算/授权/证据契约，绝不作为完整算法可用的证明。真实库测试用解析函数检查多代更新、边界、目标方向、有限预算及禁用 objective 回调，不要求偶然命中理论最优点。新增真实 BoTorch 约束 NEHVI 构建测试确认实际采集对象为 qLogNEHVI。运行中的 BoTorch 纯 Python 数值实现提示属于同一算法的编译扩展可用性信息，未触发替换优化算法。

最终专项复核：`python -B -m pytest tests/test_optimizer_protocol.py -q -rs -p no:cacheprovider --basetemp .tmp/batch4-verified`，**40 passed、2 skipped**。首次因 Windows 临时目录 WinError 5 中断，随后同一离线命令在获准执行环境重跑通过；网络/COM/LLM 测试禁用夹具始终生效。

未执行真实 Aspen、COM、LLM 或网络服务；未修改案例原始 BKP/APW；未读取密钥；未实现 GA、灰狼、NSGA-III。
