# H3 泛化性验证报告

> 生成时间：2026-06-08 13:20:22
> 扫描目录：`cases/Aspen示例/`
> 本报告只测量、不修复，目标是暴露语义规则真实缺口。

---

## 0. 总览

> **覆盖率说明**：`semantic_coverage = 被规则命中节点数 / 总叶节点数`。
> Aspen 每个设备块有数百个内部参数节点，但规则只覆盖少数有工程意义的可调字段（如塔的回流比/采出比/进料板），
> 因此即使规则完整，覆盖率数值也天然偏低（通常 < 5%）。
> 真正重要的指标是：**已有规则的设备类型是否都被命中**，以及 **medium/high confidence 变量是否涵盖关键调参字段**。

| 指标 | 值 |
|------|----|
| 扫描文件总数 | 2 |
| discover 成功（含空 catalog） | 2 |
| 空 catalog 文件数（无可信数据） | 0 |
| **有效样本文件数** | **2** |
| 平均语义覆盖率（基于有效样本） | 0.8% |
| 总可调变量数 | 10668 |
| high confidence | 0 |
| medium confidence | 72 |
| low confidence（未匹配规则） | 10596 |

---

## 1. 逐文件扫描结果

### xl1.bkp

| 指标 | 值 |
|------|----|
| discover 耗时 | 1.9s |
| 语义覆盖率 | 0.8% |
| 可调变量总数 | 5334 |
| — high confidence | 0 |
| — medium confidence | 36 |
| — low confidence（无规则） | 5298 |
| 可读输出节点数 | 629 |
| 扫描警告数 | 1 |

**build_config_draft 结果：**

| 指标 | 值 |
|------|----|
| build 耗时 | 0.01s |
| 纳入 design_variables | 18 |
| 目标函数生成数 | 2 |
| 约束生成数 | 0 |
| 目标映射失败数 | 0 |
| 约束映射失败数 | 0 |
| 边界缺失变量数 | 0 |
| build 警告数 | 38 |

**扫描警告（前5条）：**

- [缓存] 复用已有 catalog（id=92824783...，102574 节点），未重新扫描 Aspen。

**未命中规则的块名（low confidence）：**Heater

### xl2.bkp

| 指标 | 值 |
|------|----|
| discover 耗时 | 1.6s |
| 语义覆盖率 | 0.8% |
| 可调变量总数 | 5334 |
| — high confidence | 0 |
| — medium confidence | 36 |
| — low confidence（无规则） | 5298 |
| 可读输出节点数 | 629 |
| 扫描警告数 | 1 |

**build_config_draft 结果：**

| 指标 | 值 |
|------|----|
| build 耗时 | 0.00s |
| 纳入 design_variables | 18 |
| 目标函数生成数 | 2 |
| 约束生成数 | 0 |
| 目标映射失败数 | 0 |
| 约束映射失败数 | 0 |
| 边界缺失变量数 | 0 |
| build 警告数 | 38 |

**扫描警告（前5条）：**

- [缓存] 复用已有 catalog（id=ada21e34...，107100 节点），未重新扫描 Aspen。

**未命中规则的块名（low confidence）：**Heater

---

## 2. 语义规则缺口清单（设备类型维度）

> 列出**无对应语义规则**的设备类型（block_type）及其出现文件数。
> 已有规则的设备类型（如 RadFrac/Pump）不出现在此表，即使其 low-confidence 数量较多
> （low-confidence 在已知设备类型中来自非可调的内部参数节点，不是规则缺口）。

| 排名 | 无规则设备类型（block_type）| 出现文件数 |
|------|--------------------------|-----------|
| 1 | `Heater` | 2 |

---

## 2.5 字段级缺口排名

### 2.5-A 无规则设备类型下的字段排名（真实缺口，去噪后）

> 只统计属于**无语义规则设备类型**（第2节）下的 low-confidence 变量字段。
> 这些字段直接对应需要在 `configs/aspen_semantics/` 中新增的规则条目。

| 排名 | 字段名 | 出现次数 |
|------|--------|---------|
| 1 | `AUTO_COMPS_T` | 4 |
| 2 | `AUTO_PHASE_T` | 4 |
| 3 | `CPEQP` | 4 |
| 4 | `CPMED` | 4 |
| 5 | `DEGSUB` | 4 |
| 6 | `DEGSUP` | 4 |
| 7 | `DELT` | 4 |
| 8 | `DPPARM` | 4 |
| 9 | `DUTY` | 4 |
| 10 | `DUTY_TOL` | 4 |
| 11 | `EMBEDDED` | 4 |
| 12 | `EO_COMP_TOL` | 4 |
| 13 | `EO_PT_VALUE` | 4 |
| 14 | `EO_TEMP_TOL` | 4 |
| 15 | `FLOW_TOL` | 4 |
| 16 | `FVN` | 4 |
| 17 | `INCR` | 4 |
| 18 | `IVLOWER` | 4 |
| 19 | `IVSCALE` | 4 |
| 20 | `IVSTEP` | 4 |

### 2.5-B 全量字段排名（含噪声，供诊断参考）

> 包含所有 low-confidence 变量的字段，**含已有规则设备（如 RadFrac）的非可调内部参数**。
> 前几名通常是 Aspen 内部求解参数（如 CS-1、MIXED），不代表规则缺口，仅供调试参考。

| 排名 | 字段名（rel_path末段）| 总出现次数 |
|------|---------------------|-----------|
| 1 | `CS-1` | 1600 |
| 2 | `MIXED` | 440 |
| 3 | `AUTO_COMPS_T` | 62 |
| 4 | `AUTO_PHASE_T` | 62 |
| 5 | `EMBEDDED` | 62 |
| 6 | `EO_TEMP_TOL` | 62 |
| 7 | `FLOW_TOL` | 62 |
| 8 | `NEG_COMP_CHK` | 62 |
| 9 | `NEG_FLOW_CHK` | 62 |
| 10 | `SFRAC_TOL` | 62 |
| 11 | `SHOWASICON` | 62 |
| 12 | `User Table` | 62 |
| 13 | `User Tree` | 62 |
| 14 | `VFRACX_TOL` | 62 |
| 15 | `VFRAC_TOL` | 62 |
| 16 | `COMP_TOL` | 50 |
| 17 | `ATTOTAL` | 40 |
| 18 | `CUM_SUBSATT` | 40 |
| 19 | `D50` | 40 |
| 20 | `D63` | 40 |

---

## 3. 已覆盖语义规则列表

> 当前 `configs/aspen_semantics/` 中存在规则的设备类型（参考）。

| 规则文件 |
|---------|
| `compr.yaml` |
| `flash.yaml` |
| `flash3.yaml` |
| `heatx.yaml` |
| `mcompr.yaml` |
| `mixer.yaml` |
| `pump.yaml` |
| `radfrac.yaml` |
| `requil.yaml` |
| `rgibbs.yaml` |
| `rstoic.yaml` |
| `splitter.yaml` |
| `ssplit.yaml` |

---

## 4. 结论与后续建议

平均语义覆盖率 **0.8%**。**注意：低覆盖率数值不等于规则缺失**——Aspen 每个设备块有几百个内部参数节点，规则只覆盖有工程意义的少数可调字段，因此覆盖率天然偏低（< 5% 属预期）。

**设备类型缺口**：以下设备类型无对应语义规则，需补充：`Heater`（2文件）。medium/high confidence 变量共 72 个，仅来自已有规则覆盖的设备类型。

**后续建议**：在 `configs/aspen_semantics/` 中新增 `heater.yaml`，补充对应设备类型的可调字段与经验边界，即可提升这些文件的语义命中率。

本报告反映系统现状，不要求覆盖率达标。补规则属后续迭代任务，不阻断 H4/H5 推进。
