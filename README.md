# PAO — Process Aspen Optimization

> 基于贝叶斯优化的 Aspen Plus 化工流程多目标优化框架

PAO 通过 Windows COM 自动化驱动 Aspen Plus，将贝叶斯代理模型优化算法、帕累托多目标搜索、参数扫描与经济/排放目标计算集成为一套完整的流程优化工具链。

---

## 功能特性

| 功能 | 说明 |
|------|------|
| **单目标贝叶斯优化** | 支持 GP / RF / ET / GBRT 代理模型，EI / UCB / PI 采集函数 |
| **多目标帕累托优化** | NSGA-II 快速非支配排序 + 拥挤距离，WFG 超体积指标 |
| **自适应区域搜索** | 三阶段策略：空间分区 → DOE 采样 → 贝叶斯精化 |
| **参数扫描 (DOE)** | 网格 / LHS 拉丁超立方 / 随机采样 |
| **经济目标** | TAC 总年费用（Turton 方法，CAPEX + OPEX） |
| **排放目标** | LCA 全生命周期 CO₂ 当量核算 |
| **SQLite 持久化** | 所有运行案例自动存入本地数据库，支持查询与断点续算 |
| **YAML 配置驱动** | 无需修改代码，通过配置文件定义优化问题 |
| **优雅降级** | scikit-optimize / numpy 缺失时自动回退到随机采样 |

---

## 环境要求

- **操作系统**：Windows 10 / 11（Aspen Plus COM 接口仅支持 Windows）
- **Python**：3.9 或以上
- **Aspen Plus**：已安装并激活许可证（支持 V11 及以上版本）
- 依赖库见 [requirements.txt](requirements.txt)

---

## 安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd PAO

# 2. 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 快速开始

### 运行单个案例

```bash
python src/main.py cases/demo_case/case_config.yaml
```

### 单目标贝叶斯优化

```bash
python src/main.py cases/demo_case/case_config.yaml --db results.db
```

### 多目标帕累托优化

```bash
python src/main.py cases/demo_case/pareto_config.yaml --db results.db
```

### 干运行（仅验证配置，不启动 Aspen）

```bash
python src/main.py cases/demo_case/pareto_config.yaml --dry-run
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `config` | YAML 配置文件路径 | 必填 |
| `--db` | SQLite 数据库路径 | `results.db` |
| `--log` | 日志级别 (DEBUG/INFO/WARNING) | `INFO` |
| `--log-file` | 日志文件路径 | 无（仅控制台） |
| `--dry-run` | 仅解析配置，不运行仿真 | `False` |

---

## 配置文件格式

```yaml
# 仿真器设置
simulator:
  filepath: path/to/simulation.bkp   # Aspen 备份文件路径
  visible: false                      # 是否显示 Aspen 界面
  suppress_dialogs: true
  timeout: 300                        # 单次运行超时（秒）

# 设计变量
design_variables:
  - name: feed_flow
    aspen_path: \Data\Streams\FEED\Input\TOTFLOW\MIXED
    type: continuous          # continuous | integer
    lower_bound: 100.0
    upper_bound: 500.0
    initial_value: 300.0

# 需要读取的输出路径（非目标路径）
output_paths:
  - \Data\Blocks\COL1\Output\REB_DUTY

# 目标函数
objectives:
  - name: TAC
    type: tac                 # tac | emissions | aspen_path
    minimize: true
    unit: $/yr
  - name: CO2
    type: emissions
    minimize: true
    unit: kg/yr

# 约束条件（可选）
constraints:
  - name: purity
    aspen_path: \Data\Streams\PROD\Output\MOLEFRAC\MIXED\ETHANOL
    operator: ">="
    threshold: 0.995

# 优化器
optimizer:
  type: pareto_bayesian       # run | bayesian | pareto_bayesian | param_scan | adaptive_region
  surrogate_model: GP         # GP | RF | ET | GBRT | random
  acquisition: EI             # EI | UCB | PI
  n_initial: 10               # 初始 DOE 样本数
  n_iterations: 40            # 贝叶斯迭代次数
  scalarization: Tschebycheff # 多目标标量化方法（Tschebycheff | weighted_sum）
```

---

## Agent Workflow MVP

`src/agents/workflows.py` 提供了 **`run_demo_case_workflow`**，作为 agent 控制层的阶段 1 MVP。它编排现有底层 tools，输出结构化报告，不直接操作 Aspen COM 接口。

```bash
# 验证配置（不启动 Aspen）
python scripts/smoke_agent_workflow.py --config cases/demo_case/pareto_config.yaml --mode validate

# 完整运行（会启动 Aspen，消耗仿真时间）
# ⚠ full 前请先执行 preflight，并推荐使用隔离目录，避免污染原始数据库
python scripts/smoke_agent_workflow.py --config cases/demo_case/pareto_config.yaml --mode full --allow-aspen
```

完整说明（调用方式、安全限制、报告结构、当前局限）见 **[docs/agent_workflow_mvp.md](docs/agent_workflow_mvp.md)**。

---

## 工作流说明

```
src/main.py (CLI 入口)
    │
    ├── run_case          单次确定性运行
    ├── optimize_case     单目标贝叶斯优化循环
    ├── optimize_pareto_case  多目标帕累托 + 贝叶斯
    ├── param_scan        DOE 参数扫描
    └── adaptive_region_search  三阶段自适应区域搜索
            │
            ├── Phase 0：输入空间分区（超立方网格）
            ├── Phase 1：各区域 DOE 采样 → 收敛率 / 灵敏度分析
            └── Phase 2：高优先级区域 → optimize_pareto_case 精化
```

每个工作流均通过 `SimulationDB` 将所有案例实时写入 SQLite，支持：
- 中断后从数据库恢复并续算
- 按目标值排序查询最优案例
- 跨会话历史对比

---

## 目录结构

```
PAO/
├── src/
│   ├── main.py                  # CLI 入口
│   ├── aspen_driver/            # Aspen Plus COM 底层驱动
│   ├── models/                  # 数据模型（ProcessCase、SimulationResult 等）
│   ├── workflows/               # 高层工作流（优化、扫描、自适应搜索）
│   ├── optimization/            # 算法层（代理模型、帕累托、灵敏度）
│   ├── database/                # SQLite 持久化
│   ├── economics/               # TAC / 排放目标计算
│   ├── knowledge/               # 领域知识库（预留）
│   ├── literature/              # 文献参数提取（PDF → 表格 → 方程）
│   ├── agents/                  # LLM 智能体控制层（阶段 1 MVP 已实现）
│   └── utils/                   # 日志、文件 IO、单位换算
├── tests/                       # pytest 测试套件（75+ 测试文件）
├── cases/                       # 示例案例与配置文件
│   └── demo_case/
│       ├── pareto_config.yaml
│       └── 二级氢氰化工段.bkp
├── configs/
│   └── aspen_semantics/         # Aspen 语义元数据映射（RADFRAC、HEATX 等）
├── data/                        # 参考数据
├── cache/                       # 运行缓存
├── requirements.txt
└── README.md
```

---

## 运行测试

```bash
# 运行全部测试
pytest tests/

# 仅运行冒烟测试（不依赖 Aspen 连接）
pytest tests/ -k "not aspen"

# 查看详细输出
pytest tests/ -v
```

> **注意**：集成测试（`smoke_test_*.py`）需要有效的 Aspen Plus 安装与许可证。不含 Aspen 环境时，单元测试（`test_*_logic.py`）可独立运行。

---

## 依赖说明

| 包 | 用途 | 是否必需 |
|----|------|----------|
| `pywin32` | Aspen Plus COM 自动化 | ✅ 必需 |
| `PyYAML` | 配置文件解析 | ✅ 必需 |
| `numpy` | LHS 采样加速 | ⚡ 可选（缺失时回退随机采样） |
| `scikit-optimize` | 贝叶斯代理模型 | ⚡ 可选（缺失时回退随机采样） |
| `scipy` / `scikit-learn` | scikit-optimize 依赖 | ⚡ 随 skopt 安装 |
| `matplotlib` | 帕累托前沿可视化脚本 | 📊 仅脚本 |
| `pytest` | 测试框架 | 🧪 仅开发 |

---

## License

本项目为私有研究代码，未经授权禁止分发。
