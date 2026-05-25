# PAO — Aspen Plus 贝叶斯优化框架

PAO（Process Aspen Optimizer）是一个基于 Python 的 Aspen Plus 贝叶斯优化框架。通过 YAML 配置文件描述仿真工况，自动驱动 Aspen Plus 执行仿真、提取结果、拟合高斯过程代理模型，并通过采集函数推荐下一个候选参数点，实现闭环优化。

## 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置文件详解](#配置文件详解)
- [命令行用法](#命令行用法)
- [Python API 用法](#python-api-用法)
- [查询优化结果](#查询优化结果)
- [运行测试](#运行测试)
- [常见问题](#常见问题)

---

## 环境要求

| 依赖 | 版本要求 | 说明 |
|---|---|---|
| Python | 3.10+ | |
| Aspen Plus | V12 或更高 | 需已安装并授权，COM 接口可用 |
| PyYAML | 6.0+ | `pip install pyyaml` |
| numpy | 1.24+ | LHS 采样（可选，缺失时退化为均匀随机采样） |
| scikit-optimize | 0.9+ | 高斯过程代理模型（可选，缺失时退化为随机采样） |
| pywin32 | 306+ | Aspen Plus COM 接口，Windows 专用 |

> **注意**：PAO 仅支持 Windows，因为 Aspen Plus 的 COM 自动化接口依赖 `pywin32`。

---

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd PAO

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装核心依赖
pip install pyyaml pywin32

# 安装优化依赖（推荐，缺失时退化为随机采样）
pip install numpy scikit-optimize
```

---

## 项目结构

```
PAO/
├── cases/
│   └── demo_case/
│       ├── case_config.yaml        # 优化配置文件（主入口）
│       ├── scan_config.yaml        # 参数扫描配置文件
│       ├── 二级氢氰化工段.bkp      # Aspen Plus 仿真文件
│       └── output/
│           └── simulation.db       # 优化结果数据库（自动生成）
├── src/
│   ├── main.py                     # 命令行入口
│   ├── aspen_driver/               # Aspen Plus COM 底层封装
│   │   ├── driver.py               # 连接、读写、运行控制
│   │   ├── runner.py               # 单次仿真运行
│   │   └── exporter.py             # block/stream 数据提取
│   ├── models/                     # 数据模型
│   │   ├── process_case.py         # ProcessCase（工况聚合快照）
│   │   ├── simulation_result.py    # SimulationResult
│   │   ├── block.py                # BlockResult
│   │   └── stream.py               # StreamResult
│   ├── workflows/                  # 业务流程层
│   │   ├── run_case.py             # 单次工况运行
│   │   ├── param_scan.py           # 参数扫描
│   │   └── optimize_case.py        # 贝叶斯优化循环
│   ├── database/
│   │   └── simulation_db.py        # SQLite 持久化层
│   └── utils/
│       └── file_io.py              # YAML 配置加载
└── tests/
    ├── test_run_case_logic.py      # run_case 纯 Python 单元测试
    ├── test_optimize_case_logic.py # optimize_case 纯 Python 单元测试
    └── smoke_test_*.py             # 需要 Aspen Plus 的端到端冒烟测试
```

---

## 快速开始

### 1. 验证配置（不启动 Aspen）

```bash
python -m src.main cases/demo_case/case_config.yaml --dry-run
```

输出示例：

```
10:00:00 [INFO] __main__: 仿真文件：E:\...\二级氢氰化工段.bkp
10:00:00 [INFO] __main__: 设计变量（2 个）：
10:00:00 [INFO] __main__:   B:F  [0.3, 0.9]
10:00:00 [INFO] __main__:   BASIS_RR  [1, 3]
10:00:00 [INFO] __main__: 优化目标：ADN_FRAC（最大化），初始 DOE=10，总迭代=60，采集函数=EI
10:00:00 [INFO] __main__: 结果数据库：E:\...\output\simulation.db
10:00:00 [INFO] __main__: --dry-run 模式，跳过仿真。
```

### 2. 运行优化

```bash
python -m src.main cases/demo_case/case_config.yaml
```

优化过程中每次仿真完成后实时打印状态，结果自动保存到 `cases/demo_case/output/simulation.db`。

### 3. 指定数据库路径和日志级别

```bash
python -m src.main cases/demo_case/case_config.yaml --db output/run1.db --log DEBUG
```

---

## 配置文件详解

配置文件为 YAML 格式，分为六个节。以 `cases/demo_case/case_config.yaml` 为参考：

### `simulator` — 仿真文件与运行参数

```yaml
simulator:
  filepath: cases/demo_case/二级氢氰化工段.bkp  # 相对于项目根目录，或绝对路径
  visible: false           # Aspen Plus 窗口是否可见（调试时设为 true）
  suppress_dialogs: true   # 是否抑制弹窗
  timeout: 300             # 单次仿真超时（秒）
  reinit: true             # 每次运行前是否 reinit（清除上次结果）
  verify_inputs: true      # 写入后是否读回校验
  input_rtol: 1.0e-6       # 输入读回校验相对容差
```

### `design_variables` — 设计变量

每个变量需指定 `type`：

- `continuous`：连续变量，参与贝叶斯优化搜索
- `integer`：整数变量，当前版本固定为 `initial_value`，不参与搜索

```yaml
design_variables:
  - name: T0301_BF
    aspen_path: \Data\Blocks\T0301\Input\B:F   # Aspen 树路径
    type: continuous
    lower_bound: 0.3
    upper_bound: 0.9
    initial_value: 0.6
    unit: "-"

  - name: T0301_FEED_STAGE
    aspen_path: \Data\Blocks\T0301\Input\FEED_STAGE\0318
    type: integer          # 固定为 initial_value=15，不参与优化
    lower_bound: 10
    upper_bound: 20
    initial_value: 15
```

### `output_paths` — 需要读取的 Aspen 输出节点

```yaml
output_paths:
  - \Data\Streams\ADN\Output\MASSFRAC\MIXED\ADN
  - \Data\Streams\ADN\Output\MASSFLOW\MIXED\ADN
```

这些路径的值会在每次仿真后自动读取，存入 `SimulationResult.outputs`，供目标函数和约束函数使用。

### `objectives` — 目标函数

目标函数从 `output_paths` 中读取对应路径的值，无需编写 Python 代码：

```yaml
objectives:
  - name: ADN_FRAC                                        # 目标函数名称
    aspen_path: \Data\Streams\ADN\Output\MASSFRAC\MIXED\ADN  # 必须在 output_paths 中
    minimize: false   # false = 最大化；true = 最小化
    unit: ""

  - name: ADN_FLOW
    aspen_path: \Data\Streams\ADN\Output\MASSFLOW\MIXED\ADN
    minimize: false
    unit: kg/hr
```

> **当前限制**：贝叶斯优化仅支持单目标，使用第一个 objective。多目标支持在后续版本中加入。

### `constraints` — 约束

约束形式为 `value OP threshold`，标准化为 `value <= 0` 表示满足：

```yaml
constraints:
  - name: purity_min
    aspen_path: \Data\Streams\ADN\Output\MASSFRAC\MIXED\ADN
    operator: ">="      # 支持 <=, <, >=, >, ==
    threshold: 0.95     # 要求纯度 >= 0.95
```

无约束时设为空列表：

```yaml
constraints: []
```

### `extraction` — block/stream 提取配置

控制每次仿真后提取哪些 block 和 stream 的详细数据：

```yaml
extraction:
  check_status_paths:       # 检查收敛状态的节点（null = 检查全部）
    - \Data\Blocks\T0301
    - \Data\Streams\ADN
  blocks:                   # 提取 Output 子树的 block（null = 提取全部）
    - T0301
  streams:                  # 提取的 stream（null = 提取全部）
    - ADN
  block_max_depth: 3
  stream_max_depth: 3
  stream_output_subtree: "Output\\STR_MAIN"
  strict_extraction: false  # false = 节点失败记录到 notes，不阻断目标计算
```

### `optimizer` — 贝叶斯优化参数

```yaml
optimizer:
  type: bayesian
  n_initial_points: 10    # 初始 DOE 采样点数（拉丁超立方采样）
  n_iterations: 50        # 贝叶斯优化循环次数（不含初始 DOE）
                          # 总仿真次数 = n_initial_points + n_iterations = 60
  acquisition_function: EI  # EI（期望改进）/ UCB（置信上界）/ PI（改进概率）
  random_seed: 42
```

---

## 命令行用法

```
python -m src.main <config> [--db PATH] [--log LEVEL] [--dry-run]
```

| 参数 | 说明 | 默认值 |
|---|---|---|
| `config` | `case_config.yaml` 路径 | 必填 |
| `--db PATH` | 结果数据库路径 | `<yaml目录>/output/simulation.db` |
| `--log LEVEL` | 日志级别：`DEBUG` / `INFO` / `WARNING` | `INFO` |
| `--dry-run` | 只加载配置并打印摘要，不启动 Aspen | — |

---

## Python API 用法

### 方式一：从 YAML 加载配置（推荐）

```python
from src.utils.file_io import load_optimize_config
from src.aspen_driver.driver import AspenDriver
from src.workflows.optimize_case import optimize_case

opt_cfg, sim_filepath, driver_kwargs = load_optimize_config("cases/demo_case/case_config.yaml")
opt_cfg.db_path = "output/simulation.db"  # 可选：覆盖数据库路径

with AspenDriver(**driver_kwargs) as driver:
    driver.open(sim_filepath)
    result = optimize_case(driver, opt_cfg)

print(f"最优 {result.objective_name} = {result.best_value:.4g}")
print(f"最优参数：{result.best_case.design_vars}")
print(f"收敛历史：{result.convergence_history}")
```

### 方式二：手动构建配置

```python
from src.aspen_driver.driver import AspenDriver
from src.models.process_case import ObjectiveValue, ConstraintValue
from src.workflows.run_case import RunCaseConfig
from src.workflows.optimize_case import OptimizeCaseConfig, optimize_case

def tac_objective(case):
    val = case.sim_result.outputs.get(r"\Data\Streams\ADN\Output\MASSFRAC\MIXED\ADN")
    return ObjectiveValue(name="ADN_FRAC", value=val.value if val else None, minimize=False)

def purity_constraint(case):
    val = case.sim_result.outputs.get(r"\Data\Streams\ADN\Output\MASSFRAC\MIXED\ADN")
    v = val.value if val else None
    return ConstraintValue(name="purity", value=(0.95 - v) if v is not None else None)

run_cfg = RunCaseConfig(
    output_paths=[r"\Data\Streams\ADN\Output\MASSFRAC\MIXED\ADN"],
    objective_fns=[tac_objective],
    constraint_fns=[purity_constraint],
    timeout=300,
)

opt_cfg = OptimizeCaseConfig(
    param_bounds={
        r"\Data\Blocks\T0301\Input\B:F":       (0.3, 0.9),
        r"\Data\Blocks\T0301\Input\BASIS_RR":  (1.0, 3.0),
    },
    fixed_vars={
        r"\Data\Blocks\T0301\Input\FEED_STAGE\0318": 15,
    },
    run_config=run_cfg,
    n_initial=10,
    n_iterations=60,
    objective_name="ADN_FRAC",
    minimize=False,
    acquisition="EI",
    random_seed=42,
    db_path="output/simulation.db",
)

with AspenDriver(visible=False, suppress_dialogs=True) as driver:
    driver.open("cases/demo_case/二级氢氰化工段.bkp")
    result = optimize_case(driver, opt_cfg)
```

### 参数扫描

```python
from src.workflows.param_scan import ParamScanConfig, param_scan, linspace
from src.workflows.run_case import RunCaseConfig
from src.aspen_driver.driver import AspenDriver

run_cfg = RunCaseConfig(
    output_paths=[r"\Data\Streams\ADN\Output\MASSFRAC\MIXED\ADN"],
    objective_fns=[tac_objective],
)

scan_cfg = ParamScanConfig(
    scan_vars={
        r"\Data\Blocks\T0301\Input\BASIS_RR": linspace(1.0, 3.0, 5),
    },
    fixed_vars={
        r"\Data\Blocks\T0301\Input\B:F": 0.6,
        r"\Data\Blocks\T0301\Input\FEED_STAGE\0318": 15,
    },
    run_config=run_cfg,
    mode="grid",
    tags=["sensitivity"],
)

with AspenDriver() as driver:
    driver.open("cases/demo_case/二级氢氰化工段.bkp")
    result = param_scan(driver, scan_cfg)

print(f"成功率：{result.success_rate:.1%}")
for case in result.successful_cases():
    print(case.summary())
```

---

## 查询优化结果

结果以 SQLite 格式保存，可直接用 Python 查询：

```python
from src.database.simulation_db import SimulationDB

with SimulationDB("cases/demo_case/output/simulation.db") as db:
    # 查询所有成功工况
    rows = db.query_cases(status="success", limit=20)

    # 按目标函数值排序（降序取前 5）
    top5 = db.query_by_objective("ADN_FRAC", order_desc=True, limit=5)
    for row in top5:
        print(row["objective_value"], row["design_vars"])

    # 按标签过滤（只看贝叶斯优化阶段的工况）
    bo_cases = db.query_cases(tags=["bayesian_opt"])

    # 按迭代范围查询
    late_cases = db.query_cases(iteration_min=20)

    # 取完整工况（含 blocks/streams 详情）
    full = db.get_case(rows[0]["case_id"])
```

数据库 Schema：

| 表 | 说明 |
|---|---|
| `cases` | 主表，每行一个 ProcessCase |
| `objectives` | 目标函数值，支持按值过滤/排序 |
| `tags` | 标签，支持多标签 AND 过滤 |

---

## 运行测试

### 纯 Python 单元测试（不需要 Aspen Plus）

```bash
# 运行所有单元测试
python -m pytest tests/test_run_case_logic.py tests/test_optimize_case_logic.py -v

# 只运行优化逻辑测试
python -m pytest tests/test_optimize_case_logic.py -v
```

### 端到端冒烟测试（需要 Aspen Plus）

```bash
# 参数扫描冒烟测试
python tests/smoke_test_scan.py

# 数据库冒烟测试
python tests/smoke_test_db.py
```

---

## 常见问题

**Q：运行时报 `AspenConnectionError: 未连接`**

确认 Aspen Plus 已安装并授权，且 COM 接口可用。可以先在 Python 中测试：

```python
import win32com.client
app = win32com.client.Dispatch("Apwn.Document")
```

**Q：目标函数返回 `available=False`，错误为"路径不在 outputs 中"**

检查 `case_config.yaml` 的 `output_paths` 是否包含了 `objectives` 中所有 `aspen_path`。两者必须一致。

**Q：`scikit-optimize` 未安装时会怎样**

优化器自动退化为随机采样，仍可正常运行，但不会利用历史数据推荐候选点，优化效率下降。安装后自动启用高斯过程：

```bash
pip install scikit-optimize
```

**Q：`integer` 类型的设计变量如何处理**

当前版本将 `type: integer` 的变量固定为 `initial_value`，不参与贝叶斯优化搜索。如需整数变量参与优化，可将其改为 `continuous` 并在目标函数中取整。

**Q：如何在优化过程中实时监控进度**

通过 `on_case_complete` 回调：

```python
def on_complete(case, idx, total):
    print(f"[{idx+1}/{total}] status={case.status.value}, "
          f"best={result_so_far}")

opt_cfg.on_case_complete = on_complete
```

**Q：如何从上次中断的地方继续优化**

当前版本不支持断点续跑。但历史数据已保存在数据库中，可以通过 `start_iteration` 参数指定起始迭代编号，并手动从数据库加载历史样本初始化优化器（后续版本将提供自动恢复功能）。
