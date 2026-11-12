"""
verify_write_feasibility.py — 设计变量写入可行性验证工具

职责
----
对 ConfigDraft.design_variables 中的每个变量执行：
  打开 Aspen 文件 → 读取原值 → 试写测试值 → 读回验证 → 恢复原值 → 关闭文件

不调用 Engine.Run2 / run_case，纯 COM 树操作（只读 + 写节点值）。
返回 WriteFeasibilityReport，列出哪些变量可写、哪些不可写及原因。

调用方
------
graph.py 的 write_feasibility_node 在 onboarding_node 完成后调用此函数，
将不可写变量从 ConfigDraft 中移除后再进入 human_confirm 让用户确认。
"""
from __future__ import annotations

import logging
import math
from typing import Any

from src.models.tunable import WriteCheckResult, WriteFeasibilityReport

_log = logging.getLogger(__name__)

_RTOL = 1e-4   # 读回校验相对容差（比仿真层宽松，允许 Aspen 轻微舍入）


def _values_close(a: Any, b: Any) -> bool:
    """判断两个数值是否在容差内相等。非数值类型直接用相等比较。"""
    try:
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        if fb == 0.0:
            return abs(fa) < 1e-10
        return abs(fa - fb) / abs(fb) <= _RTOL
    except (TypeError, ValueError):
        return a == b


def verify_write_feasibility_impl(
    aspen_file_path: str,
    design_variables: list[dict],
) -> WriteFeasibilityReport:
    """
    对 design_variables 列表中每个变量执行试写验证。

    Parameters
    ----------
    aspen_file_path:
        Aspen 仿真文件绝对路径（.bkp / .apw）。
    design_variables:
        ConfigDraft.design_variables 列表，每项须含 aspen_path 字段；
        lower_bound / initial_value 用于选取试写测试值。

    Returns
    -------
    WriteFeasibilityReport
    """
    raise PermissionError('Legacy execution disabled: no complete ActionGateway execution envelope; use an approved static simulation/optimization plan')
