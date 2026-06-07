# H1 端到端真实链路运行记录 — demo_case

> **H1 验收状态：[H1 PASS — 链路完整，已产出 summary_report]**
>
> 运行时间：2026-06-08 03:09:38
> 脚本参数：--n-initial 3 --n-iter 3
> Aspen 文件：`cases/demo_case/二级氢氰化工段.bkp`
> 脚本：`scripts/run_phase1_e2e.py`

---

## 总体结果

| 指标 | 值 |
|------|-----|
| H1 验收 | [H1 PASS — 链路完整，已产出 summary_report] |
| 总耗时 | 32s |
| 中断点 | 无 |
| Session ID | 9f66a780-c77b-452c-badb-841c0fa08975 |
| DB 路径 | reports\phase1\output\simulation.db |

---

## 各步骤耗时

| 步骤 | 耗时 | 状态 | 备注 |
|------|------|------|------|
| discover | 0.7s | — | |
| llm | 11.0s | — | |
| build | 0.0s | — | |
| validate_pre | 0.5s | — | |
| feedback | 0.0s | — | |
| filter_vars | 0.0s | — | |
| write_yaml | 0.0s | — | |
| validate_post | 0.0s | — | |
| optimize | 20.2s | — | |
| report | 0.0s | — | |

---

## 优化结果（session_id 过滤）

| 指标 | 值 |
|------|-----|
| Session ID | 9f66a780-c77b-452c-badb-841c0fa08975 |
| Pareto 前沿大小 | 1 |
| 成功仿真次数 | 5 |
| 失败仿真次数 | 1 |
| 成功率 | 83% |

---

## 运行备注

- LLM 意图解析成功，三要素齐全：goals=['ADN_FLOW', 'REB_DUTY']，constraints=['purity_min']

---

## 后续参考（H5 超时设计依据）

- 单次仿真耗时量级：约 3 秒（估算）
- 建议 H5 超时设置：约 10 秒（单次耗时 × 3 保守裕量）
