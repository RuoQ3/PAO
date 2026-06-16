# PAO 优化结果综合分析报告
> 配置文件：`draft_2096340c.yaml`  数据库：`simulation.db`  会话：`b440f711-c5c8-4670-bf78-5b3e7ab8e55b`

## 0. 目标达成总览

> ℹ 未提供优化意图，无法评估达成度。

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
| 5e3ac3ac | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |
| 9bb98f01 | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |
| c11953c0 | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |
| 86191604 | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |
| f2ca2a13 | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |
| 255ec22b | 0.00 | 0.00 | 0.00 | —（无法计算） | ⚠ PARTIAL（无设备快照） |

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

- `c01a5bc3` (iter=7, sim_failed):
  - 仿真未收敛（block/stream 存在错误标志（errors））：具体错误：以下 block/stream 有错误：['T0302']
  - Aspen history diagnostics: _1019awp.his: *** SEVERE ERROR WHILE EXECUTING UNIT OPERATIONS BLOCK: "T0302" (MODEL: | _1019awp.his: "RADFRAC")                                               (UDL3ZR.3) | _1019awp.his: MATERIAL AND ENERGY BALANCES FAILED TO CONVERGE: CHECK COL-SPECS | _1019awp.his: OR SUPPLY BETTER TEMPERATURE AND COMPOSITION ESTIMATES.。由于仿真未收敛，block/stream 快照未入库，无法进行单元级详细诊断。建议：（a）查看 Aspen .his 文件获取具体错误；（b）调整初值或收敛参数后重跑；（c）使用 query_node_db_tool(mode='recurring_errors') 排查结构性失败路径。
- `febf46cd` (iter=8, sim_failed):
  - 仿真未收敛（block/stream 存在错误标志（errors））：具体错误：以下 block/stream 有错误：['T0302']。由于仿真未收敛，block/stream 快照未入库，无法进行单元级详细诊断。建议：（a）查看 Aspen .his 文件获取具体错误；（b）调整初值或收敛参数后重跑；（c）使用 query_node_db_tool(mode='recurring_errors') 排查结构性失败路径。
- `c7f51f54` (iter=18, sim_failed):
  - 仿真未收敛（block/stream 存在错误标志（errors））：具体错误：以下 block/stream 有错误：['T0302']。由于仿真未收敛，block/stream 快照未入库，无法进行单元级详细诊断。建议：（a）查看 Aspen .his 文件获取具体错误；（b）调整初值或收敛参数后重跑；（c）使用 query_node_db_tool(mode='recurring_errors') 排查结构性失败路径。
- `7a3a9b99` (iter=25, sim_failed):
  - 仿真未收敛（block/stream 存在错误标志（errors））：具体错误：以下 block/stream 有错误：['T0302']。由于仿真未收敛，block/stream 快照未入库，无法进行单元级详细诊断。建议：（a）查看 Aspen .his 文件获取具体错误；（b）调整初值或收敛参数后重跑；（c）使用 query_node_db_tool(mode='recurring_errors') 排查结构性失败路径。
- `36d2d1da` (iter=26, sim_failed):
  - 仿真未收敛（block/stream 存在错误标志（errors））：具体错误：以下 block/stream 有错误：['T0302']。由于仿真未收敛，block/stream 快照未入库，无法进行单元级详细诊断。建议：（a）查看 Aspen .his 文件获取具体错误；（b）调整初值或收敛参数后重跑；（c）使用 query_node_db_tool(mode='recurring_errors') 排查结构性失败路径。
