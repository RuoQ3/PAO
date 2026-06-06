# Smoke 验证结果总结

## 1. 执行时间

2026-06-05

## 2. 执行命令

```bash
# validate 模式
python scripts/smoke_agent_workflow.py \
  --config cases/demo_case/pareto_config.yaml \
  --mode validate \
  --out reports/smoke/validate_demo_case.txt \
  --max-chars 2000

# db 模式
python scripts/smoke_agent_workflow.py \
  --config cases/demo_case/pareto_config.yaml \
  --mode db \
  --out reports/smoke/db_demo_case.txt \
  --max-chars 2000

# full 安全门检查（不传 --allow-aspen）
python scripts/smoke_agent_workflow.py \
  --config cases/demo_case/pareto_config.yaml \
  --mode full \
  --max-chars 1000
```

## 3. 退出码

| 模式 | 退出码 | 预期 | 结论 |
|------|--------|------|------|
| validate | 0 | 0 | ✓ 通过 |
| db | 0 | 0 | ✓ 通过 |
| full（无 --allow-aspen） | 1 | ≠ 0 | ✓ 安全门正常 |

## 4. validate 模式结论

**通过。** Python 解析链完整：

- load_optimize_config 成功，识别为多目标 Pareto 配置
- 搜索维度 3（B:F、FEED_STAGE/0318、BASIS_RR），目标函数 2（ADN_FLOW、REB_DUTY），约束 1（ADN 纯度）
- 初始 DOE 20 点，BO 迭代 60 次，总评估 80 次，代理模型 GP
- Aspen 仿真文件 `二级氢氰化工段.bkp` 存在于磁盘
- 数值合理性检查无警告
- validate_config_tool 综合结论：`[通过]`，可进入 run_case smoke test 或直接启动优化

## 5. db 模式结论

**通过，并获取到真实历史数据。**

- SimulationDB 路径 `cases/demo_case/output/simulation.db` 存在，共 80 条工况
- 当前页 10 条全部 `status=success`，成功率 100%，无仿真失败工况
- 因无失败工况，`get_failed_case_ids` 返回空列表，诊断分支跳过（符合预期）
- `summarize_pareto_tool` 成功完成，返回完整 Pareto 分析报告：
  - 第一前沿 3 个解，HV = 3.34836e+07，共 21 层前沿
  - 敏感性分析：BASIS_RR（0.968）> B:F（0.895）>> FEED_STAGE/0318（0.026）
  - 建议固定低敏感性变量 FEED_STAGE/0318

## 6. full 模式安全门结论

**安全门正常工作。** 未传 `--allow-aspen` 时脚本输出明确拒绝信息并以退出码 1 退出，未触发任何 Aspen 调用。

## 7. 是否启动 Aspen

**否。** 三次 smoke 均未连接 Aspen COM，未触发 `run_case_tool` 或 `optimize_pareto_tool`。

## 8. 是否写数据库

**否。** db 模式仅做只读查询，未写入 `simulation.db` 或 `node.db`。

## 9. 是否发现失败标志

**否。** validate 和 db 两份报告均不包含 `[失败]`、`错误：`、`错误:` 标志，退出码均为 0。

## 10. 下一步建议

1. **进入 full --allow-aspen 前置检查（任务十）**：安全 smoke 全通，可开始准备单次真实仿真（`run_case_tool` smoke）。
2. **固定低敏感性变量**：db 报告显示 FEED_STAGE/0318 灵敏度仅 0.026，可在下次优化前考虑固定该维度，从 3D 降到 2D，减少评估次数。
3. **Pareto 前沿较小**（3 个解）：当前 80 次仿真中第一前沿仅 3 点，说明前沿密度不足；下一轮优化可增加 `n_iterations` 或调整采样策略以扩展前沿覆盖度。

## 附：脚本修改记录

本任务在执行 smoke 过程中发现并修复了一个小问题：

- **问题**：`_exit()` 函数直接 `print(display)` 时，报告中包含 ✓/✗ 等非 ASCII 字符，Windows GBK 终端抛 `UnicodeEncodeError`。
- **修复**：`scripts/smoke_agent_workflow.py` 中对终端输出做 `.encode(encoding, errors="replace").decode(encoding)` 安全编码，文件输出（`--out`）保持 UTF-8 完整内容不变。
- **影响范围**：仅影响终端显示，不影响功能逻辑和报告内容。
