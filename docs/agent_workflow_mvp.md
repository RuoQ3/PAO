# Agent Workflow MVP — `run_demo_case_workflow` 开发说明

> 阶段 1 上层 Agent Workflow MVP  
> 对应模块：`src/agents/workflows.py`

---

## 1. 定位

`run_demo_case_workflow` 是 PAO 项目 **agent 控制层的最小可行版本（MVP）**。

它的职责是：把现有的底层 tools（`validate_config_tool`、`run_case_tool`、`optimize_pareto_tool` 等）按照固定的编排顺序依次调用，并将结果汇总为一份结构化的文本报告。

**它不是什么：**

| 它不做 | 原因 |
|--------|------|
| 不直接调用 `AspenDriver` | 所有与 Aspen 的交互由底层 tools 和 `src/workflows` 负责 |
| 不操作 COM 接口 | COM 层封装在 `src/aspen_driver` 中，agent 控制层不应直接触碰 |
| 不直接读写 Aspen 树节点 | 节点读写由 `src/aspen_driver` 的 `get_value`/`set_value` 负责 |
| 不替代 `src/workflows` | `src/workflows` 仍是底层仿真/优化循环的宿主，agent 层通过 tool 接口调用 |
| 不是自主多轮 agent | 当前是单次执行，不会自动根据结果重新触发新的优化轮次 |
| 不执行经济模型 | TAC/排放计算由 `src/economics` 负责，agent 层不内嵌该逻辑 |
| 不绕过用户确认 | full 模式需要显式传入 `--allow-aspen` 才会启动 Aspen，不会静默启动 |

---

## 2. 调用方式

### Python

```python
from src.agents.workflows import run_demo_case_workflow
from src.agents.tool_runner import RealToolRunner

report = run_demo_case_workflow(
    "cases/demo_case/pareto_config.yaml",
    tool_runner=RealToolRunner(),
)
print(report)
```

`RealToolRunner` 把协议方法转发到真实 tools；测试时可注入 `FakeRunner`（见 `tests/test_agent_workflows.py`）。

### CLI / smoke 脚本

```bash
# 仅验证配置，不启动 Aspen
python scripts/smoke_agent_workflow.py \
    --config cases/demo_case/pareto_config.yaml \
    --mode validate

# 验证 + 数据库查询（不启动 Aspen）
python scripts/smoke_agent_workflow.py \
    --config cases/demo_case/pareto_config.yaml \
    --mode db

# 完整运行（会启动 Aspen，消耗仿真时间）
python scripts/smoke_agent_workflow.py \
    --config cases/demo_case/pareto_config.yaml \
    --mode full --allow-aspen
```

---

## 3. 安全限制

在执行 `full --allow-aspen` 之前，请务必阅读以下说明：

- **`validate` 和 `db` 模式不会启动 Aspen**，可安全随时运行。
- **`full --allow-aspen` 会启动 Aspen Plus 并实际执行仿真**，每次运行消耗真实的仿真时间和许可证资源。
- **运行 full 前，应先执行 preflight 检查**，确认环境就绪：
  ```bash
  python scripts/preflight_full_aspen.py \
      --config cases/demo_case/pareto_config.yaml --suggest-copy
  ```
- **不建议直接对 `cases/demo_case/output/` 中的历史数据库运行 full**，否则可能追加或修改历史记录，导致本次结果与历史结果混杂。
- **推荐先用隔离目录和小规模参数做 smoke 验证**，再扩大规模：
  ```bash
  # 生成隔离目录（n_initial=2, n_iterations=1）
  python scripts/prepare_isolated_full_smoke.py \
      --config cases/demo_case/pareto_config.yaml \
      --out-root runs --n-initial 2 --n-iterations 1

  # 对隔离目录的新 config 执行 preflight
  python scripts/preflight_full_aspen.py \
      --config runs/<new_dir>/pareto_config.yaml --suggest-copy

  # 确认 PASS 后再运行
  python scripts/smoke_agent_workflow.py \
      --config runs/<new_dir>/pareto_config.yaml \
      --mode full --allow-aspen
  ```

---

## 4. 报告结构

`run_demo_case_workflow` 返回的文本报告固定包含 7 个章节，顺序如下：

| 章节 | 内容 |
|------|------|
| 【1. 配置摘要】 | 配置路径、optimizer 类型、目标函数名、db 路径等 |
| 【2. 校验结果】 | `load_config` 和 `validate_config` 步骤的状态与详情 |
| 【3. 运行/优化结果】 | `optimize_pareto` 或 `run_case` 步骤的状态与详情 |
| 【4. 数据库查询结果】 | `query_simulation_db` 步骤的状态与详情 |
| 【5. 诊断结论】 | 失败工况的 `diagnose_case` 和 `query_node_db` 结果 |
| 【6. Pareto 总结】 | `summarize_pareto` 步骤的状态与详情 |
| 【7. 下一步建议】 | 由 `determine_next_actions` 规则函数生成的建议列表 |

报告中的每个步骤状态用标签标注：`[成功]` / `[失败]` / `[已跳过]`。

---

## 5. 当前限制

以下功能在当前 MVP 阶段**尚未实现**，不应依赖：

1. **不是自主多轮 agent**：每次调用是单次执行，不会根据报告结果自动发起新一轮优化或调整策略。
2. **不会自动执行经济模型**：TAC、LCA 等经济/排放目标的计算由底层工具负责，agent 层不内嵌。
3. **不会绕过用户确认启动 Aspen full run**：`full` 模式必须显式添加 `--allow-aspen` 标志。
4. **`next_actions` 是规则型，不是 LLM 决策**：`determine_next_actions` 依据各步骤状态字段套用预定义规则，不调用语言模型。
5. **不会自动修改 Aspen case 或 `.bkp` 文件**：agent 层不写回 Aspen 仿真文件，所有仿真参数修改须手动操作或通过专用脚本完成。
6. **不会自动修改 YAML 配置**：报告中的 `next_actions` 只是建议文字，不会自动更新 `pareto_config.yaml` 中的任何字段。
7. **不会自动应用配置规划结果**：即使未来接入 Config Planning Agent，参数调整建议也须经用户审核后再写入配置，agent 控制层不代替用户决策。

---

## 6. 与项目目标的关系

`run_demo_case_workflow` 是 PAO 多 agent 闭环规划中的**第一层控制入口**。

```
未来多 agent 架构（规划中）
│
├── Workflow Operator Agent
│       └── 调用 run_demo_case_workflow（受限入口）
│               └── 通过 tool 接口 → validate / run / optimize / query / diagnose / summarize
│
├── Result Analysis Agent
│       └── 读取统一报告，分析 Pareto 前沿、产品纯度、能耗分布
│
├── Config Planning Agent
│       └── 根据当前报告推荐新的设计变量范围或约束条件
│
└── Economics / Energy Agent
        └── 对筛选出的操作点执行经济性和能耗分析
```

当前 MVP 的价值在于：
- 为后续 agent 提供**统一的报告接口**，避免各 agent 各自直接操作 COM/driver；
- 通过 tool 注入（`DemoWorkflowToolRunner` 协议）实现**测试与生产解耦**；
- 建立**步骤状态（WorkflowStep）+ next_actions 规则**的基础结构，后续 LLM 决策可替换规则函数。
