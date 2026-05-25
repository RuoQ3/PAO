"""
runner.py — 单次 Aspen Plus 仿真运行的业务流程封装。

职责：在 driver.py 的低级 COM 操作之上，提供：
  - 批量设置输入变量（含写入后读回校验）
  - reinit → run → 检查 block/stream 结果状态 → 读取输出变量
  - 返回结构化的 SimulationResult

不持有 COM 连接，所有操作通过 AspenDriver 委托执行。
数据模型定义见 src/models/simulation_result.py。
"""
from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

from .errors import AspenNodeError, AspenRunError, AspenRunTimeoutError
from .node import AspenNode, ValueType
from ..models.simulation_result import (
    BlockStatus,
    InputVerification,
    RunStatus,
    SimulationResult,
    StatusCheckResult,
    VariableResult,
)

if TYPE_CHECKING:
    from .driver import AspenDriver

_log = logging.getLogger(__name__)

# 输入读回校验的默认相对容差
_DEFAULT_INPUT_RTOL = 1e-6


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    单次 Aspen Plus 仿真运行的业务流程封装。

    典型用法：
        runner = SimulationRunner(driver)
        result = runner.run_case(
            inputs={
                r"\\Data\\Blocks\\B1\\Input\\TEMP": 350.0,
                r"\\Data\\Streams\\FEED\\Input\\FLOW\\MIXED": 100.0,
            },
            output_paths=[
                r"\\Data\\Blocks\\B1\\Output\\DUTY",
                r"\\Data\\Streams\\PROD\\Output\\MOLEFLOW\\MIXED",
            ],
        )
        if result.success:
            duty = result.outputs[r"\\Data\\Blocks\\B1\\Output\\DUTY"].value
    """

    # Aspen 树中 block 和 stream 集合的标准路径
    _BLOCKS_PATH  = r"\Data\Blocks"
    _STREAMS_PATH = r"\Data\Streams"

    def __init__(self, driver: AspenDriver) -> None:
        self._driver = driver

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #

    def run_case(
        self,
        inputs: dict[str, Any],
        output_paths: list[str],
        timeout: float = 300.0,
        reinit: bool = True,
        verify_inputs: bool = True,
        input_rtol: float = _DEFAULT_INPUT_RTOL,
        check_status_paths: list[str] | None = None,
    ) -> SimulationResult:
        """
        执行一次完整的仿真运行。

        流程：
          1. 写入输入变量
          2. （可选）读回校验输入
          3. （可选）reinit
          4. run
          5. 检查 block/stream 结果状态（HAP_COMPSTATUS）
          6. 读取输出变量

        Parameters
        ----------
        inputs:
            {节点路径: 值} 字典，写入 Aspen 树的输入变量。
        output_paths:
            需要读取的输出节点路径列表。
        timeout:
            仿真运行超时（秒），默认 300s。
        reinit:
            运行前是否 reinit，默认 True。
            设为 False 可复用上次收敛解作为初值（参数扫描场景）。
        verify_inputs:
            写入后是否读回校验，默认 True。
            校验失败不中断运行，但会记录在 input_verifications 中。
        input_rtol:
            输入读回校验的相对容差，默认 1e-6。
        check_status_paths:
            需要检查 HAP_COMPSTATUS 的节点路径列表。
            None（默认）：自动检查 \\Data\\Blocks 和 \\Data\\Streams 下所有子节点。
            []：跳过状态检查。

        Returns
        -------
        SimulationResult
        """
        # source_filepath 在整个 run_case 期间不会变化，提前记录一次即可。
        # mutation_snapshot 每次返回时通过 _make_result() 读取当前值，
        # 确保快照反映的是本次尝试结束后 Aspen 树的实际输入状态。
        _source_filepath = self._driver.filepath

        def _make_result(**kwargs: Any) -> SimulationResult:
            return SimulationResult(
                source_filepath=_source_filepath,
                mutation_snapshot=self._driver.mutation_count,
                **kwargs,
            )

        # 1. 写入输入变量
        write_errors = self._write_inputs(inputs)
        if write_errors:
            return _make_result(
                status=RunStatus.WRITE_FAILED,
                success=False,
                requested_inputs=inputs,
                error="写入输入变量失败：\n" + "\n".join(write_errors),
            )

        # 2. 读回校验输入
        verifications: list[InputVerification] = []
        actual_inputs: dict[str, Any] = dict(inputs)
        input_warnings: list[str] = []
        if verify_inputs:
            verifications, actual_inputs = self._verify_inputs(inputs, input_rtol)
            unmatched = [v for v in verifications if not v.match]
            if unmatched:
                _log.warning(
                    "输入写入校验：%d/%d 个变量读回值与请求值不一致。",
                    len(unmatched), len(verifications),
                )
                for v in unmatched:
                    msg = f"输入不一致 {v.path}：请求={v.requested!r} 实际={v.actual!r} {v.note}"
                    _log.warning("  %s", msg)
                    input_warnings.append(msg)

        # 3. Reinit
        if reinit:
            try:
                self._driver.reinit()
            except AspenRunError as exc:
                return _make_result(
                    status=RunStatus.RUN_FAILED,
                    success=False,
                    requested_inputs=inputs,
                    actual_inputs=actual_inputs,
                    input_verifications=verifications,
                    error=f"Reinit 失败：{exc}",
                )

        # 4. Run
        t0 = time.monotonic()
        try:
            self._driver.run(timeout=timeout)
        except AspenRunTimeoutError as exc:
            return _make_result(
                status=RunStatus.RUN_FAILED,
                success=False,
                requested_inputs=inputs,
                actual_inputs=actual_inputs,
                input_verifications=verifications,
                run_time=time.monotonic() - t0,
                error=str(exc),
            )
        except AspenRunError as exc:
            return _make_result(
                status=RunStatus.RUN_FAILED,
                success=False,
                requested_inputs=inputs,
                actual_inputs=actual_inputs,
                input_verifications=verifications,
                run_time=time.monotonic() - t0,
                error=f"仿真运行失败：{exc}",
            )
        run_time = time.monotonic() - t0

        # 5. 检查 block/stream 结果状态
        check_result = self._check_statuses(check_status_paths)

        # 状态检查不可用：hap_constants 缺失，无法验证结果可信性
        if check_result.unavailable:
            return _make_result(
                status=RunStatus.STATUS_UNAVAILABLE,
                success=False,
                requested_inputs=inputs,
                actual_inputs=actual_inputs,
                input_verifications=verifications,
                block_statuses=check_result.statuses,
                run_time=run_time,
                error=(
                    "hap_constants 未加载，无法验证仿真结果状态。"
                    "请使用 require_type_library=True 连接，或显式传入 check_status_paths=[] 跳过检查。"
                ),
                warnings=input_warnings,
            )

        # 状态读取部分失败：只要有任何节点读取失败，结果不可信
        if check_result.failed:
            failed_names = list(check_result.failed.keys())
            _log.warning("以下节点的 HAP_COMPSTATUS 读取失败：%s", failed_names)
            return _make_result(
                status=RunStatus.STATUS_UNAVAILABLE,
                success=False,
                requested_inputs=inputs,
                actual_inputs=actual_inputs,
                input_verifications=verifications,
                block_statuses=check_result.statuses,
                run_time=run_time,
                error=f"以下节点的 HAP_COMPSTATUS 读取失败，无法验证结果可信性：{failed_names}",
                warnings=input_warnings,
            )

        # 自动枚举为空：未找到任何可检查节点，结果不可信
        if not check_result.statuses and not check_result.explicitly_skipped:
            return _make_result(
                status=RunStatus.STATUS_UNAVAILABLE,
                success=False,
                requested_inputs=inputs,
                actual_inputs=actual_inputs,
                input_verifications=verifications,
                block_statuses=[],
                run_time=run_time,
                error=(
                    r"自动枚举未找到任何 block/stream 节点，无法验证仿真结果。"
                    r"请检查 \Data\Blocks 和 \Data\Streams 路径，或显式传入 check_status_paths=[] 跳过检查。"
                ),
                warnings=input_warnings,
            )

        overall_status, status_error, status_warnings = self._aggregate_status(check_result.statuses)
        all_warnings = input_warnings + status_warnings

        # 输入不一致时，将 SUCCESS 降级为 WARNINGS
        if input_warnings and overall_status == RunStatus.SUCCESS:
            overall_status = RunStatus.WARNINGS

        if not overall_status.is_convergent:
            return _make_result(
                status=overall_status,
                success=False,
                requested_inputs=inputs,
                actual_inputs=actual_inputs,
                input_verifications=verifications,
                block_statuses=check_result.statuses,
                run_time=run_time,
                error=status_error,
                warnings=all_warnings,
            )

        # 6. 读取输出变量
        outputs, failed_outputs = self._read_outputs(output_paths)

        has_output_failures = len(failed_outputs) > 0
        final_status = overall_status if not has_output_failures else RunStatus.ERRORS
        success = not has_output_failures and overall_status.is_convergent

        return _make_result(
            status=final_status,
            success=success,
            requested_inputs=inputs,
            actual_inputs=actual_inputs,
            input_verifications=verifications,
            outputs=outputs,
            failed_outputs=failed_outputs,
            block_statuses=check_result.statuses,
            run_time=run_time,
            error=(
                f"读取 {len(failed_outputs)} 个输出变量失败。"
                if has_output_failures else None
            ),
            warnings=all_warnings,
        )

    # ------------------------------------------------------------------ #
    # 输入写入
    # ------------------------------------------------------------------ #

    def _write_inputs(self, inputs: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for path, value in inputs.items():
            try:
                self._driver.set_value(path, value)
            except AspenNodeError as exc:
                errors.append(f"  {path}: {exc}")
        if errors:
            _log.error("写入输入变量时发生 %d 个错误。", len(errors))
        return errors

    def _verify_inputs(
        self,
        inputs: dict[str, Any],
        rtol: float,
    ) -> tuple[list[InputVerification], dict[str, Any]]:
        """
        写入后读回每个输入节点，校验实际值与请求值是否一致。

        浮点数用相对容差比较；非数值类型用相等比较。
        读回失败时 match=False，actual=None，note 记录错误原因。
        """
        verifications: list[InputVerification] = []
        actual_inputs: dict[str, Any] = {}

        for path, requested in inputs.items():
            try:
                actual = self._driver.get_value(path)
                actual_inputs[path] = actual
                match, note = _values_match(requested, actual, rtol)
            except AspenNodeError as exc:
                actual = None
                actual_inputs[path] = None
                match = False
                note = f"读回失败：{exc}"

            verifications.append(InputVerification(
                path=path,
                requested=requested,
                actual=actual,
                match=match,
                note=note,
            ))

        return verifications, actual_inputs

    # ------------------------------------------------------------------ #
    # 结果状态检查（HAP_COMPSTATUS，手册 38-13）
    # ------------------------------------------------------------------ #

    def _check_statuses(
        self,
        paths: list[str] | None,
    ) -> StatusCheckResult:
        """
        检查 block/stream 的 HAP_COMPSTATUS，返回 StatusCheckResult。

        paths=None：自动枚举 \\Data\\Blocks 和 \\Data\\Streams 的直接子节点。
        paths=[]：显式跳过检查，返回空结果（unavailable=False）。
        """
        # 显式跳过
        if paths is not None and len(paths) == 0:
            return StatusCheckResult(explicitly_skipped=True)

        hap = self._driver.hap_constants
        if hap is None:
            _log.warning(
                "hap_constants 未加载，无法执行 HAP_COMPSTATUS 状态检查。"
                "运行 scripts/verify_hap_constants.py 诊断。"
            )
            return StatusCheckResult(unavailable=True)

        comp_attr = hap.get("HAP_COMPSTATUS")
        if comp_attr is None:
            _log.warning("HAP_COMPSTATUS 不在 hap_constants 中，无法执行状态检查。")
            return StatusCheckResult(unavailable=True)

        record_attr = hap.get("HAP_RECORDTYPE")

        if paths is None:
            paths, enum_errors = self._enumerate_block_stream_paths()
            if enum_errors:
                # 枚举失败的父节点写入 failed，触发上层 STATUS_UNAVAILABLE
                failed = dict(enum_errors)
                return StatusCheckResult(statuses=[], failed=failed)

        statuses: list[BlockStatus] = []
        failed: dict[str, str] = {}

        for path in paths:
            node = AspenNode(self._driver, path)
            try:
                comp_status = int(node.get_attribute(comp_attr))
            except AspenNodeError as exc:
                failed[path] = str(exc)
                _log.warning("读取 '%s' 的 HAP_COMPSTATUS 失败：%s", path, exc)
                continue

            record_type = ""
            if record_attr is not None:
                try:
                    record_type = str(node.get_attribute(record_attr) or "")
                except AspenNodeError:
                    pass

            flags = _parse_comp_status(comp_status, hap)
            # 无法解析任何标志时，标记为 UNKNOWN，不当成功
            if not flags:
                flags = ["UNKNOWN"]
                _log.warning(
                    "节点 '%s' 的 HAP_COMPSTATUS=%d 无法解析为已知标志，标记为 UNKNOWN。",
                    path, comp_status,
                )

            statuses.append(BlockStatus(
                name=node.name,
                record_type=record_type,
                comp_status=comp_status,
                status_flags=flags,
            ))

        return StatusCheckResult(statuses=statuses, failed=failed)

    def _enumerate_block_stream_paths(self) -> tuple[list[str], dict[str, str]]:
        """
        枚举 \\Data\\Blocks 和 \\Data\\Streams 下的直接子节点路径。

        Returns
        -------
        (paths, enum_errors)
            paths      : 成功枚举到的节点路径列表。
            enum_errors: {parent_path: 错误信息}，枚举失败的父节点。
        """
        paths: list[str] = []
        enum_errors: dict[str, str] = {}
        for parent_path in (self._BLOCKS_PATH, self._STREAMS_PATH):
            if not self._driver.node_exists(parent_path):
                continue
            parent = AspenNode(self._driver, parent_path)
            try:
                for name in parent.child_names():
                    paths.append(f"{parent_path}\\{name}")
            except AspenNodeError as exc:
                enum_errors[parent_path] = str(exc)
                _log.warning("枚举 '%s' 子节点失败：%s", parent_path, exc)
        return paths, enum_errors

    @staticmethod
    def _aggregate_status(
        block_statuses: list[BlockStatus],
    ) -> tuple[RunStatus, str | None, list[str]]:
        """
        将所有 block/stream 状态聚合为整体 RunStatus。

        优先级（高到低）：ERRORS > UNKNOWN > INCOMPAT > INACCESS > NO_RESULTS > WARNINGS > SUCCESS

        空列表仅在 check_status_paths=[] 显式跳过时出现，返回 SUCCESS。
        UNKNOWN 标志（位掩码无法解析）视为 ERRORS，不当成功。
        """
        if not block_statuses:
            return RunStatus.SUCCESS, None, []

        warnings: list[str] = []
        has_errors = False
        has_unknown = False
        has_no_results = False
        has_incompat = False
        has_inaccess = False
        has_warnings = False

        for bs in block_statuses:
            flags = set(bs.status_flags)
            if "ERRORS" in flags:
                has_errors = True
            if "UNKNOWN" in flags:
                has_unknown = True
            if "NO_RESULTS" in flags:
                has_no_results = True
            if "INCOMPAT" in flags:
                has_incompat = True
            if "INACCESS" in flags:
                has_inaccess = True
            if "WARNINGS" in flags:
                has_warnings = True
                warnings.append(f"{bs.name}（{bs.record_type}）有警告。")

        if has_errors:
            names = [bs.name for bs in block_statuses if "ERRORS" in set(bs.status_flags)]
            return RunStatus.ERRORS, f"以下 block/stream 有错误：{names}", warnings
        if has_unknown:
            names = [bs.name for bs in block_statuses if "UNKNOWN" in set(bs.status_flags)]
            return RunStatus.ERRORS, f"以下 block/stream 状态无法解析（UNKNOWN）：{names}", warnings
        if has_incompat:
            names = [bs.name for bs in block_statuses if "INCOMPAT" in set(bs.status_flags)]
            return RunStatus.INCOMPAT, f"以下结果与输入不兼容（需重新运行）：{names}", warnings
        if has_inaccess:
            names = [bs.name for bs in block_statuses if "INACCESS" in set(bs.status_flags)]
            return RunStatus.INACCESS, f"以下结果不可访问：{names}", warnings
        if has_no_results:
            names = [bs.name for bs in block_statuses if "NO_RESULTS" in set(bs.status_flags)]
            return RunStatus.NO_RESULTS, f"以下 block/stream 无结果：{names}", warnings
        if has_warnings:
            return RunStatus.WARNINGS, None, warnings
        return RunStatus.SUCCESS, None, []

    # ------------------------------------------------------------------ #
    # 输出读取
    # ------------------------------------------------------------------ #

    def _read_outputs(
        self,
        paths: list[str],
    ) -> tuple[dict[str, VariableResult], dict[str, str]]:
        outputs: dict[str, VariableResult] = {}
        failed: dict[str, str] = {}

        for path in paths:
            node = AspenNode(self._driver, path)
            try:
                value = node.value
                unit = node.get_unit()
                try:
                    vtype = int(node.value_type)
                except AspenNodeError:
                    vtype = int(ValueType.UNDEFINED)
                outputs[path] = VariableResult(
                    path=path, value=value, unit=unit, value_type=vtype,
                )
            except AspenNodeError as exc:
                failed[path] = str(exc)
                _log.warning("读取输出节点 '%s' 失败：%s", path, exc)

        return outputs, failed

    # ------------------------------------------------------------------ #
    # 便捷方法
    # ------------------------------------------------------------------ #

    def set_input(self, path: str, value: Any) -> None:
        """写入单个输入变量，失败时抛出 AspenNodeError。"""
        self._driver.set_value(path, value)

    def get_output(self, path: str) -> VariableResult:
        """读取单个输出变量，返回 VariableResult。失败时抛出 AspenNodeError。"""
        node = AspenNode(self._driver, path)
        value = node.value
        unit = node.get_unit()
        try:
            vtype = int(node.value_type)
        except AspenNodeError:
            vtype = int(ValueType.UNDEFINED)
        return VariableResult(path=path, value=value, unit=unit, value_type=vtype)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _values_match(requested: Any, actual: Any, rtol: float) -> tuple[bool, str]:
    """比较请求值与实际值，返回 (match, note)。"""
    if requested is None and actual is None:
        return True, ""
    if requested is None or actual is None:
        return False, f"一方为 None：requested={requested!r} actual={actual!r}"
    try:
        r = float(requested)
        a = float(actual)
        if math.isnan(r) and math.isnan(a):
            return True, ""
        denom = max(abs(r), abs(a), 1e-300)
        rel_err = abs(r - a) / denom
        if rel_err <= rtol:
            return True, ""
        return False, f"相对误差 {rel_err:.2e} 超过容差 {rtol:.2e}"
    except (TypeError, ValueError):
        match = requested == actual
        return match, ("" if match else f"值不相等：{requested!r} != {actual!r}")


def _parse_comp_status(comp_status: int, hap: dict[str, int]) -> list[str]:
    """
    将 HAP_COMPSTATUS 整数值解析为标志名列表（手册 38-13）。

    使用位掩码：(comp_status & mask) == mask 表示该标志置位。
    """
    flag_map = {
        "SUCCESS":  "HAP_RESULTS_SUCCESS",
        "ERRORS":   "HAP_RESULTS_ERRORS",
        "WARNINGS": "HAP_RESULTS_WARNINGS",
        "NO_RESULTS": "HAP_NORESULTS",
        "INCOMPAT": "HAP_RESULTS_INCOMPAT",
        "INACCESS": "HAP_RESULTS_INACCESS",
    }
    flags: list[str] = []
    for flag_name, hap_name in flag_map.items():
        mask = hap.get(hap_name)
        if mask is not None and (comp_status & mask) == mask:
            flags.append(flag_name)
    return flags
