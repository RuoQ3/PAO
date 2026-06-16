# PAO 优化结果综合分析报告
> 配置文件：`draft_5fc7ed70.yaml`  数据库：`simulation.db`  会话：`4566c4eb-baf1-4381-b1ff-cc062457dd97`

## 0. 目标达成总览

> **总判定：0/1 约束满足，1 条数据缺失；主目标改善 +22.1%**

### 硬约束达成情况

| 约束名 | 方向 | 阈值 | 实际值 | 状态 |
|--------|------|------|--------|------|
| purity_min | ≥ 下限 | 0.9 | N/A | ❓ 数据缺失 |

### 优化目标达成情况

| 目标名 | 方向 | 最优值 | 基线值 | 改善幅度 |
|--------|------|--------|--------|---------|
| ADN_FLOW | 最大化 ↑ | 2.21e+04 | 1.81e+04 | +22.1% |
| REB_DUTY | 最小化 ↓ | 1.445e+05 | 3.235e+05 | +55.3% |

<details>
<summary>详细匹配说明</summary>

- **purity_min**：metric='purity' 无法在目标函数中找到对应列；数据缺失，无法判定是否满足
- **ADN_FLOW**：精确匹配目标名 'ADN_FLOW'，共 6 个有效点；改善 +22.1%（相对基线 1.81e+04）
- **REB_DUTY**：精确匹配目标名 'REB_DUTY'，共 6 个有效点；改善 +55.3%（相对基线 3.235e+05）

</details>


## 1. 优化总览

| 指标 | 值 |
|------|---|
| 总工况数 | 30 |
| 成功工况数 | 20 |
| 成功率 | 66.7% |
| 目标函数 | ADN_FLOW, REB_DUTY |
| Pareto 前沿大小（第一前沿） | 6 |
| Pareto 层数 | 9 |
| 超体积（HV） | 2.66045e+08 |
| 最大迭代编号 | 29 |

## 2. TAC 分解

> 无数据：前沿工况中未找到可计算的 block 数据（blocks 快照为空）。

## 3. 排放分析

**第一 Pareto 前沿（共 6 个工况）**

| 工况 | 蒸汽/热源排放 (t CO₂/yr) | 电力排放 (t CO₂/yr) | Scope 1 (t CO₂/yr) | 合计 (t CO₂/yr) | 完整性 |
|------|-------------------------:|--------------------:|-------------------:|----------------:|--------|
| 42aa2028 | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |
| 6742ab09 | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |
| 99673eff | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |
| e982fcd0 | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |
| 25e1f320 | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |
| 9a5b0897 | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |

> 所有工况均为 PARTIAL（存在 skipped blocks/streams），拒绝最优/最差排序。

## 4. 设计变量重要性

### 4.1 目标：ADN_FLOW

**目标函数：ADN_FLOW  |  样本量：20  |  方法：spearman**

| 排名 | 变量（路径末尾）| Spearman ρ（带符号） | 方向 | 可靠性 |
|------|----------------|:--------------------:|------|--------|
| 1 | `T0301\Input\B:F` | +0.987 | 正相关 ↑ | ✓ 可靠 |
| 2 | `T0301\Input\BASIS_RR` | +0.007 | 正相关 ↑ | ✓ 可靠 |

> **解读**：`T0301\Input\B:F` 对 `ADN_FLOW` 影响最大（ρ = +0.987）。增大该变量会使目标升高。 |ρ| > 0.3 的变量通常值得优先关注；|ρ| < 0.1 的变量可考虑固定以减少搜索维度。

### 4.2 目标：REB_DUTY

**目标函数：REB_DUTY  |  样本量：20  |  方法：spearman**

| 排名 | 变量（路径末尾）| Spearman ρ（带符号） | 方向 | 可靠性 |
|------|----------------|:--------------------:|------|--------|
| 1 | `T0301\Input\BASIS_RR` | +0.968 | 正相关 ↑ | ✓ 可靠 |
| 2 | `T0301\Input\B:F` | -0.200 | 负相关 ↓ | ✓ 可靠 |

> **解读**：`T0301\Input\BASIS_RR` 对 `REB_DUTY` 影响最大（ρ = +0.968）。增大该变量会使目标升高。 |ρ| > 0.3 的变量通常值得优先关注；|ρ| < 0.1 的变量可考虑固定以减少搜索维度。

## 5. 失败诊断摘要

**失败工况统计：10/30 (33.3%)**

| 失败类型 | 数量 | 占比 |
|----------|-----:|-----:|
| 仿真失败（引擎错误/超时） | 10 | 100% |

**近 5 个失败工况的设计变量分布：**

| 变量 | 最小值 | 最大值 | 均值 | 样本数 |
|------|-------:|-------:|-----:|-------:|
| `T0301\Input\B:F` | 0.3 | 0.8894 | 0.4615 | 5 |
| `T0301\Input\BASIS_RR` | 1.737 | 3 | 2.451 | 5 |

> **提示**：上表显示失败工况中各设计变量的取值范围，集中在极端值（接近上下界）附近的变量可能是高危区域，建议适当收窄相应边界。

**逐工况诊断建议：**

- `de68f0b8` (iter=7, sim_failed):
  - 仿真未收敛（block/stream 存在错误标志（errors））：具体错误：以下 block/stream 有错误：['T0302']
  - Aspen history diagnostics: _2032mts.his: *** SEVERE ERROR WHILE EXECUTING UNIT OPERATIONS BLOCK: "T0302" (MODEL: | _2032mts.his: "RADFRAC")                                               (UDL3ZR.3) | _2032mts.his: MATERIAL AND ENERGY BALANCES FAILED TO CONVERGE: CHECK COL-SPECS | _2032mts.his: OR SUPPLY BETTER TEMPERATURE AND COMPOSITION ESTIMATES.。由于仿真未收敛，block/stream 快照未入库，无法进行单元级详细诊断。建议：（a）查看 Aspen .his 文件获取具体错误；（b）调整初值或收敛参数后重跑；（c）使用 query_node_db_tool(mode='recurring_errors') 排查结构性失败路径。
- `22af2f51` (iter=8, sim_failed):
  - 仿真未收敛（block/stream 存在错误标志（errors））：具体错误：以下 block/stream 有错误：['T0302']。由于仿真未收敛，block/stream 快照未入库，无法进行单元级详细诊断。建议：（a）查看 Aspen .his 文件获取具体错误；（b）调整初值或收敛参数后重跑；（c）使用 query_node_db_tool(mode='recurring_errors') 排查结构性失败路径。
- `6c95e931` (iter=18, sim_failed):
  - 仿真未收敛（block/stream 存在错误标志（errors））：具体错误：以下 block/stream 有错误：['T0302']。由于仿真未收敛，block/stream 快照未入库，无法进行单元级详细诊断。建议：（a）查看 Aspen .his 文件获取具体错误；（b）调整初值或收敛参数后重跑；（c）使用 query_node_db_tool(mode='recurring_errors') 排查结构性失败路径。
- `b90e9d99` (iter=25, sim_failed):
  - 仿真未收敛（block/stream 存在错误标志（errors））：具体错误：以下 block/stream 有错误：['T0302']。由于仿真未收敛，block/stream 快照未入库，无法进行单元级详细诊断。建议：（a）查看 Aspen .his 文件获取具体错误；（b）调整初值或收敛参数后重跑；（c）使用 query_node_db_tool(mode='recurring_errors') 排查结构性失败路径。
- `81722d7d` (iter=26, sim_failed):
  - 仿真未收敛（block/stream 存在错误标志（errors））：具体错误：以下 block/stream 有错误：['T0302']。由于仿真未收敛，block/stream 快照未入库，无法进行单元级详细诊断。建议：（a）查看 Aspen .his 文件获取具体错误；（b）调整初值或收敛参数后重跑；（c）使用 query_node_db_tool(mode='recurring_errors') 排查结构性失败路径。
