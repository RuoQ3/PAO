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
    warnings: list[str] = []
    results: list[WriteCheckResult] = []

    if not design_variables:
        return WriteFeasibilityReport(aspen_file=aspen_file_path)

    # ── 懒加载 COM 模块（避免导入时触碰 aspen_driver）─────────────────────────
    try:
        from src.aspen_driver.driver import AspenDriver
        from src.aspen_driver.errors import AspenNodeError
    except ImportError as exc:
        warnings.append(f"无法导入 aspen_driver 模块，跳过试写验证：{exc}")
        return WriteFeasibilityReport(
            aspen_file=aspen_file_path,
            warnings=warnings,
        )

    driver: AspenDriver | None = None
    try:
        driver = AspenDriver(visible=False, suppress_dialogs=True)
        driver.connect()
        driver.open(aspen_file_path)
        _log.info("verify_write_feasibility: 已打开 %s，开始试写 %d 个变量",
                  aspen_file_path, len(design_variables))

        for dv in design_variables:
            path = dv.get("aspen_path", "")
            if not path:
                continue

            # 选取测试值：优先 lower_bound，其次 initial_value
            test_val = dv.get("lower_bound")
            if test_val is None:
                test_val = dv.get("initial_value")
            if test_val is None:
                results.append(WriteCheckResult(
                    aspen_path=path,
                    original_value=None,
                    test_value=None,
                    writable=False,
                    error="无法确定测试值（lower_bound 和 initial_value 均为 None）",
                ))
                continue

            # 1. 读取原值
            try:
                original = driver.get_value(path)
            except AspenNodeError as exc:
                results.append(WriteCheckResult(
                    aspen_path=path,
                    original_value=None,
                    test_value=test_val,
                    writable=False,
                    error=f"读取原值失败：{exc}",
                ))
                continue

            # 2. 试写测试值
            try:
                driver.set_value(path, test_val)
            except (AspenNodeError, Exception) as exc:
                results.append(WriteCheckResult(
                    aspen_path=path,
                    original_value=original,
                    test_value=test_val,
                    writable=False,
                    error=str(exc),
                ))
                _log.debug("verify_write_feasibility: 试写失败 %s: %s", path, exc)
                continue

            # 3. 读回验证
            try:
                actual = driver.get_value(path)
            except AspenNodeError as exc:
                actual = None

            if actual is None or not _values_close(actual, test_val):
                err = f"写入后读回值 {actual!r} 与期望 {test_val!r} 不一致（节点拒绝写入）"
                results.append(WriteCheckResult(
                    aspen_path=path,
                    original_value=original,
                    test_value=test_val,
                    writable=False,
                    error=err,
                ))
                _log.debug("verify_write_feasibility: 读回不一致 %s: %s", path, err)
                # 尝试恢复原值
                if original is not None:
                    try:
                        driver.set_value(path, original)
                    except Exception:
                        pass
                continue

            # 4. 恢复原值
            if original is not None:
                try:
                    driver.set_value(path, original)
                except Exception as exc:
                    warnings.append(f"恢复 {path} 原值失败（{exc}），可能影响后续仿真起点")

            results.append(WriteCheckResult(
                aspen_path=path,
                original_value=original,
                test_value=test_val,
                writable=True,
                error="",
            ))
            _log.debug("verify_write_feasibility: 通过 %s", path)

    except Exception as exc:
        msg = f"打开 Aspen 文件失败，跳过试写验证：{exc}"
        _log.warning("verify_write_feasibility: %s", msg)
        warnings.append(msg)
        # 所有未处理的变量标记为未知（不阻断流程）
        processed = {r.aspen_path for r in results}
        for dv in design_variables:
            p = dv.get("aspen_path", "")
            if p and p not in processed:
                results.append(WriteCheckResult(
                    aspen_path=p,
                    original_value=None,
                    test_value=None,
                    writable=False,
                    error=f"Aspen 文件打开失败，无法验证：{exc}",
                ))
    finally:
        if driver is not None:
            try:
                driver.disconnect()
            except Exception as exc:
                warnings.append(f"关闭 Aspen driver 时出错：{exc}")

    unwritable = [r.aspen_path for r in results if not r.writable]
    _log.info(
        "verify_write_feasibility: 验证完成，%d 可写 / %d 不可写",
        len(results) - len(unwritable), len(unwritable),
    )
    return WriteFeasibilityReport(
        aspen_file=aspen_file_path,
        results=results,
        unwritable_paths=unwritable,
        warnings=warnings,
    )
