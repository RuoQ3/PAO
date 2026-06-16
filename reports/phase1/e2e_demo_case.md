# H1 端到端真实链路运行记录 — demo_case

> **H1 验收状态：[H1 PASS — 链路完整，已产出 summary_report]**
>
> 运行时间：2026-06-08 10:59:45
> 脚本参数：--n-initial 10 --n-iter 20
> Aspen 文件：`cases/demo_case/二级氢氰化工段.bkp`
> 脚本：`scripts/run_phase1_e2e.py`

---

## 总体结果

| 指标 | 值 |
|------|-----|
| H1 验收 | [H1 PASS — 链路完整，已产出 summary_report] |
| 总耗时 | 127s |
| 中断点 | 无 |
| Session ID | bcd34243-68de-4adb-827e-a76876c974a5 |
| DB 路径 | reports\phase1\output\simulation.db |

---

## 各步骤耗时

| 步骤 | 耗时 | 状态 | 备注 |
|------|------|------|------|
| discover | 0.7s | — | |
| llm | 11.4s | — | |
| build | 0.0s | — | |
| validate_pre | 0.6s | — | |
| feedback | 0.0s | — | |
| filter_vars | 0.0s | — | |
| write_yaml | 0.0s | — | |
| validate_post | 0.0s | — | |
| optimize | 114.0s | — | |
| report | 0.0s | — | |

---

## 优化结果（session_id 过滤）

| 指标 | 值 |
|------|-----|
| Session ID | bcd34243-68de-4adb-827e-a76876c974a5 |
| Pareto 前沿大小 | 6 |
| 成功仿真次数 | 20 |
| 失败仿真次数 | 10 |
| 成功率 | 67% |

---

## 运行备注

- LLM 意图解析成功，三要素齐全：goals=['ADN_FLOW', 'REB_DUTY']，constraints=['purity_min']

---

## 后续参考（H5 超时设计依据）

- 单次仿真耗时量级：约 4 秒（估算）
- 建议 H5 超时设置：约 11 秒（单次耗时 × 3 保守裕量）
