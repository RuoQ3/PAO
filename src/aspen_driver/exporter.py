"""
exporter.py — Aspen Plus 仿真结果导出工具。

职责：
  1. ResultExporter：将 SimulationResult 序列化为 dict / JSON / CSV。
     默认拒绝导出失败仿真的结果，防止无效数据进入数据库。
  2. TreeExporter：从 Aspen Plus 树中批量提取 stream 表、block 输出、
     任意子树的叶节点值。
     必须传入 success=True 的 SimulationResult 才能导出，
     或显式传入已校验的 block/stream 路径列表。

不持有 COM 连接，TreeExporter 通过 AspenDriver 委托执行树访问。
"""
from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .errors import AspenNodeError
from .node import AspenNode
from ..models.simulation_result import SimulationResult

if TYPE_CHECKING:
    from .driver import AspenDriver

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 树节点值记录（含单位、类型、错误字段）
# ---------------------------------------------------------------------------

@dataclass
class TreeValueRecord:
    """单个树节点的读取结果，含单位和错误信息。"""
    path: str           # 绝对路径
    rel_path: str       # 相对于导出根节点的路径
    value: Any          # 节点值；读取失败时为 None
    unit: str           # IHNode.UnitString；无单位或失败时为 ""
    value_type: int     # IHNode.ValueType (0-5)；失败时为 0
    error: str | None   # 读取失败原因；成功时为 None


# ---------------------------------------------------------------------------
# ResultExporter
# ---------------------------------------------------------------------------

class ResultExporter:
    """
    将 SimulationResult 序列化为多种格式。

    不依赖 COM 连接，纯数据转换。
    默认拒绝导出 success=False 的结果，防止无效数据流入下游。
    """

    def __init__(self, result: SimulationResult) -> None:
        self._result = result

    # ------------------------------------------------------------------ #
    # 输出变量导出
    # ------------------------------------------------------------------ #

    def outputs_flat(self, allow_failed: bool = False) -> dict[str, Any]:
        """
        返回 {path: value} 的扁平字典，仅包含成功读取的输出。

        Parameters
        ----------
        allow_failed:
            False（默认）：result.success=False 时抛出 ValueError。
            True：跳过状态校验，直接导出现有 outputs（调试用）。
        """
        self._require_success(allow_failed)
        return {path: var.value for path, var in self._result.outputs.items()}

    def outputs_records(self, allow_failed: bool = False) -> list[dict[str, Any]]:
        """
        返回输出变量的记录列表，每条记录含 path、value、unit、value_type。

        适合写入数据库或转换为 DataFrame。

        Parameters
        ----------
        allow_failed:
            False（默认）：result.success=False 时抛出 ValueError。
        """
        self._require_success(allow_failed)
        return [
            {
                "path": var.path,
                "value": var.value,
                "unit": var.unit,
                "value_type": var.value_type,
            }
            for var in self._result.outputs.values()
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        将完整 SimulationResult 序列化为可 JSON 化的字典。

        不校验 success，始终可调用（用于日志、调试、失败记录存档）。
        包含 status、success、inputs、outputs、failed_outputs、
        block_statuses、run_time、error、warnings。
        """
        r = self._result
        return {
            "status": r.status.value,
            "success": r.success,
            "run_time": r.run_time,
            "source_filepath": str(r.source_filepath) if r.source_filepath else None,
            "result_time": r.result_time,
            "mutation_snapshot": r.mutation_snapshot,
            "error": r.error,
            "warnings": r.warnings,
            "requested_inputs": r.requested_inputs,
            "actual_inputs": r.actual_inputs,
            "input_verifications": [v.to_dict() for v in r.input_verifications],
            "outputs": {
                path: {"value": var.value, "unit": var.unit, "value_type": var.value_type}
                for path, var in r.outputs.items()
            },
            "failed_outputs": r.failed_outputs,
            "block_statuses": [bs.to_dict() for bs in r.block_statuses],
        }

    def to_json(self, indent: int | None = 2) -> str:
        """序列化为 JSON 字符串。不可 JSON 化的值（如 COM 对象）回退为 str()。"""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, default=str
        )

    def to_csv(self, include_units: bool = True, allow_failed: bool = False) -> str:
        """
        将输出变量导出为 CSV 字符串。

        列：path, value[, unit]

        Parameters
        ----------
        include_units:
            True（默认）：包含 unit 列。
        allow_failed:
            False（默认）：result.success=False 时抛出 ValueError，
            防止失败仿真的部分结果被当作有效数据写入文件。
            True：跳过状态校验（调试用）。
        """
        self._require_success(allow_failed)
        buf = io.StringIO()
        fieldnames = ["path", "value", "unit"] if include_units else ["path", "value"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for var in self._result.outputs.values():
            row: dict[str, Any] = {"path": var.path, "value": var.value}
            if include_units:
                row["unit"] = var.unit
            writer.writerow(row)
        return buf.getvalue()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _require_success(self, allow_failed: bool) -> None:
        if allow_failed:
            return
        if not self._result.success:
            raise ValueError(
                f"拒绝导出失败仿真的结果（status={self._result.status.value}，"
                f"error={self._result.error!r}）。"
                "如需强制导出（调试用），请传入 allow_failed=True。"
            )


# ---------------------------------------------------------------------------
# TreeExporter
# ---------------------------------------------------------------------------

class TreeExporter:
    """
    从 Aspen Plus 树中批量提取结构化数据。

    依赖 AspenDriver，需要在 driver.connect() 之后使用。

    结果可信性保证
    --------------
    所有公开方法均要求传入 SimulationResult，并默认校验 result.success=True。
    这确保导出的树数据来自已收敛、已通过 HAP_COMPSTATUS 校验的仿真。
    如需绕过校验（调试用），显式传入 allow_failed=True。

    filepath 一致性约束（重要）
    --------------------------
    result.success=True 只能证明某次 run_case() 成功，不能证明当前 Aspen 树
    仍对应那次结果。若 run_case() 之后又修改了输入节点但未重新运行，Aspen 树
    处于 "Input Changed" 状态，此时导出的值是旧结果。

    强制约束：必须在 run_case() 成功后立即调用导出方法，中间不得修改任何输入节点。
    runner.py 会在 SimulationResult 中记录 source_filepath，导出时会校验
    driver.filepath 是否与之一致，不一致时抛出 ValueError。
    """

    _STREAMS_PATH = r"\Data\Streams"
    _BLOCKS_PATH  = r"\Data\Blocks"

    def __init__(self, driver: AspenDriver) -> None:
        self._driver = driver

    # ------------------------------------------------------------------ #
    # Stream 表
    # ------------------------------------------------------------------ #

    def export_stream_table(
        self,
        result: SimulationResult,
        stream_names: list[str] | None = None,
        output_subtree: str = "Output",
        max_depth: int = 2,
        strict: bool = True,
        allow_failed: bool = False,
    ) -> dict[str, list[TreeValueRecord]]:
        """
        提取 stream 数据表。

        Parameters
        ----------
        result:
            当前仿真的 SimulationResult，用于校验结果可信性。
            默认要求 result.success=True。
        stream_names:
            需要提取的 stream 名称列表。None 表示自动枚举所有 stream。
        output_subtree:
            在每个 stream 节点下读取的子树名称，默认 "Output"。
            传空字符串则读取 stream 节点本身的直接子节点。
        max_depth:
            递归深度，默认 2（覆盖 MOLEFLOW\\MIXED 等二级嵌套）。
            达到上限时抛出 AspenNodeError（strict=True）或记录 error 字段。
        strict:
            True（默认）：任意节点读取失败、路径不存在、深度截断时抛出 AspenNodeError。
            False：容错，失败节点记录 error 字段，value=None。
        allow_failed:
            False（默认）：result.success=False 时抛出 ValueError。
            True：跳过状态校验（调试用）。

        Returns
        -------
        {stream_name: [TreeValueRecord, ...]}
        """
        _require_result_success(result, allow_failed)
        _check_consistency(result, self._driver, allow_failed)

        if stream_names is None:
            stream_names = self._enumerate_names(self._STREAMS_PATH, strict=strict)

        out: dict[str, list[TreeValueRecord]] = {}
        for name in stream_names:
            base = f"{self._STREAMS_PATH}\\{name}"
            root = f"{base}\\{output_subtree}" if output_subtree else base
            out[name] = self._extract_subtree(root, max_depth, strict)
        return out

    # ------------------------------------------------------------------ #
    # Block 输出
    # ------------------------------------------------------------------ #

    def export_block_outputs(
        self,
        result: SimulationResult,
        block_names: list[str] | None = None,
        max_depth: int = 2,
        strict: bool = True,
        allow_failed: bool = False,
    ) -> dict[str, list[TreeValueRecord]]:
        """
        提取所有（或指定）block 的 Output 子树叶节点值。

        Parameters
        ----------
        result:
            当前仿真的 SimulationResult，用于校验结果可信性。
        block_names:
            需要提取的 block 名称列表。None 表示自动枚举所有 block。
        max_depth:
            递归深度，默认 2。
        strict:
            True（默认）：失败时抛出 AspenNodeError。
        allow_failed:
            False（默认）：result.success=False 时抛出 ValueError。

        Returns
        -------
        {block_name: [TreeValueRecord, ...]}
        """
        _require_result_success(result, allow_failed)
        _check_consistency(result, self._driver, allow_failed)

        if block_names is None:
            block_names = self._enumerate_names(self._BLOCKS_PATH, strict=strict)

        out: dict[str, list[TreeValueRecord]] = {}
        for name in block_names:
            root = f"{self._BLOCKS_PATH}\\{name}\\Output"
            out[name] = self._extract_subtree(root, max_depth, strict)
        return out

    # ------------------------------------------------------------------ #
    # 任意子树
    # ------------------------------------------------------------------ #

    def export_subtree_values(
        self,
        result: SimulationResult,
        root_path: str,
        max_depth: int = 3,
        strict: bool = True,
        allow_failed: bool = False,
    ) -> list[TreeValueRecord]:
        """
        递归提取指定子树下所有叶节点的值。

        Parameters
        ----------
        result:
            当前仿真的 SimulationResult，用于校验结果可信性。
        root_path:
            起始节点的 Aspen 树路径。
        max_depth:
            最大递归深度，默认 3。
            达到上限时 strict=True 抛出 AspenNodeError，
            strict=False 记录 error 字段（不静默丢弃）。
        strict:
            True（默认）：失败时抛出 AspenNodeError。
        allow_failed:
            False（默认）：result.success=False 时抛出 ValueError。

        Returns
        -------
        [TreeValueRecord, ...]，按树遍历顺序排列。
        """
        _require_result_success(result, allow_failed)
        _check_consistency(result, self._driver, allow_failed)
        return self._extract_subtree(root_path, max_depth, strict)

    # ------------------------------------------------------------------ #
    # 便捷视图：扁平 dict（仅值，不含单位）
    # ------------------------------------------------------------------ #

    @staticmethod
    def records_to_flat(
        records: list[TreeValueRecord],
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        将 TreeValueRecord 列表转换为 {rel_path: value} 扁平字典。

        Parameters
        ----------
        strict:
            True（默认）：任意记录含 error 字段时抛出 ValueError。
            False：跳过有错误的记录。
        """
        errors = [r for r in records if r.error is not None]
        if strict and errors:
            msgs = [f"  {r.rel_path}: {r.error}" for r in errors]
            raise ValueError(
                f"records_to_flat：{len(errors)} 条记录含错误，无法转换为扁平字典：\n"
                + "\n".join(msgs)
            )
        return {
            r.rel_path: r.value
            for r in records
            if r.error is None
        }

    @staticmethod
    def records_to_dicts(records: list[TreeValueRecord]) -> list[dict[str, Any]]:
        """
        将 TreeValueRecord 列表转换为字典列表，保留全部字段（含 error）。

        适合写入数据库或序列化为 JSON。每条记录包含：
        path、rel_path、value、unit、value_type、error。
        """
        return [
            {
                "path": r.path,
                "rel_path": r.rel_path,
                "value": r.value,
                "unit": r.unit,
                "value_type": r.value_type,
                "error": r.error,
            }
            for r in records
        ]

    @staticmethod
    def records_to_csv(records: list[TreeValueRecord]) -> str:
        """
        将 TreeValueRecord 列表序列化为 CSV 字符串。

        列：rel_path, value, unit, value_type, error
        error 列为空字符串表示读取成功。
        """
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["rel_path", "value", "unit", "value_type", "error"],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "rel_path": r.rel_path,
                "value": "" if r.value is None else r.value,
                "unit": r.unit,
                "value_type": r.value_type,
                "error": r.error or "",
            })
        return buf.getvalue()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _enumerate_names(self, parent_path: str, strict: bool) -> list[str]:
        """枚举父节点下的直接子节点名称列表。"""
        if not self._driver.node_exists(parent_path):
            if strict:
                raise AspenNodeError(f"父节点不存在：'{parent_path}'")
            _log.warning("父节点不存在，跳过枚举：'%s'", parent_path)
            return []
        parent = AspenNode(self._driver, parent_path)
        try:
            return parent.child_names()
        except AspenNodeError as exc:
            if strict:
                raise
            _log.warning("枚举 '%s' 子节点失败：%s", parent_path, exc)
            return []

    def _extract_subtree(
        self,
        root_path: str,
        max_depth: int,
        strict: bool,
    ) -> list[TreeValueRecord]:
        """递归提取 root_path 下所有叶节点，返回 TreeValueRecord 列表。"""
        if not self._driver.node_exists(root_path):
            if strict:
                raise AspenNodeError(f"导出根节点不存在：'{root_path}'")
            _log.warning("导出根节点不存在：'%s'", root_path)
            return [_error_record(root_path, "", f"导出根节点不存在：'{root_path}'")]
        records: list[TreeValueRecord] = []
        self._recurse(root_path, root_path, max_depth, records, strict)
        return records

    def _recurse(
        self,
        current_path: str,
        root_path: str,
        depth: int,
        records: list[TreeValueRecord],
        strict: bool,
    ) -> None:
        node = AspenNode(self._driver, current_path)
        rel_path = current_path[len(root_path):].lstrip("\\")

        try:
            children = node.child_names()
        except AspenNodeError as exc:
            if strict:
                raise
            records.append(_error_record(current_path, rel_path, f"枚举子节点失败：{exc}"))
            return

        if not children:
            # 叶节点：读取值、单位、类型
            records.append(_read_node_record(node, current_path, rel_path, strict))
            return

        if depth <= 0:
            # 达到深度上限：不静默丢弃，记录截断错误
            msg = (
                f"达到 max_depth 上限，节点 '{current_path}' 仍有子节点，未继续展开。"
                "请增大 max_depth 或显式指定路径集合。"
            )
            if strict:
                raise AspenNodeError(msg)
            records.append(_error_record(current_path, rel_path, msg))
            return

        for child_name in children:
            child_path = f"{current_path}\\{child_name}"
            self._recurse(child_path, root_path, depth - 1, records, strict)


# ---------------------------------------------------------------------------
# 模块级工具函数
# ---------------------------------------------------------------------------

def _require_result_success(result: SimulationResult, allow_failed: bool) -> None:
    """校验 SimulationResult 可信性，不可信时抛出 ValueError。"""
    if allow_failed:
        return
    if not result.success:
        raise ValueError(
            f"拒绝从失败仿真中导出树数据（status={result.status.value}，"
            f"error={result.error!r}）。"
            "如需强制导出（调试用），请传入 allow_failed=True。"
        )


def _check_consistency(
    result: SimulationResult,
    driver: "AspenDriver",
    allow_failed: bool,
) -> None:
    """
    校验 driver 当前状态与产生 result 时是否一致。

    两项检查：
    1. filepath 一致性：driver 当前文件必须与 result.source_filepath 相同，
       防止拿 A case 的 result 去导出 B case 的树数据。
    2. mutation counter 一致性：driver.mutation_count 必须与
       result.mutation_snapshot 相同，防止 run_case() 后又调用 set_value()
       修改了输入节点（Aspen 手册第 11/12 章 Input Changed 状态）。

    result.source_filepath 或 result.mutation_snapshot 为 None 时跳过对应检查
    （旧版 runner 未填充这些字段）。
    allow_failed=True 时跳过全部检查。
    """
    if allow_failed:
        return

    if result.source_filepath is not None:
        driver_fp = driver.filepath
        if driver_fp is None:
            raise ValueError(
                "driver 当前未打开任何文件，无法确认与 SimulationResult 的一致性。"
                "请确保在 run_case() 成功后立即导出，中间不要关闭或切换文件。"
            )
        if str(driver_fp) != str(result.source_filepath):
            raise ValueError(
                f"driver 当前文件（{driver_fp}）与产生 SimulationResult 的文件"
                f"（{result.source_filepath}）不一致。"
                "可能在 run_case() 之后切换了仿真文件，导出的树数据将对应错误的 case。"
                "如需强制导出（调试用），请传入 allow_failed=True。"
            )

    if result.mutation_snapshot is not None:
        current = driver.mutation_count
        if current != result.mutation_snapshot:
            raise ValueError(
                f"run_case() 成功后，driver.set_value() 又被调用了 "
                f"{current - result.mutation_snapshot} 次，"
                "Aspen 树可能处于 Input Changed 状态，导出的值将是旧结果。"
                "请重新运行 run_case() 后再导出，或传入 allow_failed=True 强制导出（调试用）。"
            )


def _read_node_record(
    node: AspenNode,
    abs_path: str,
    rel_path: str,
    strict: bool,
) -> TreeValueRecord:
    """读取叶节点的值、单位、ValueType，返回 TreeValueRecord。"""
    errors: list[str] = []
    value: Any = None
    unit = ""
    value_type = 0

    try:
        value = node.value
    except AspenNodeError as exc:
        errors.append(f"读取 Value 失败：{exc}")

    try:
        unit = node.get_unit()
    except Exception as exc:
        errors.append(f"读取 UnitString 失败：{exc}")

    try:
        value_type = int(node.value_type)
    except AspenNodeError as exc:
        errors.append(f"读取 ValueType 失败：{exc}")

    if errors:
        error_msg = "；".join(errors)
        if strict:
            raise AspenNodeError(f"读取节点 '{abs_path}' 失败：{error_msg}")
        return TreeValueRecord(
            path=abs_path,
            rel_path=rel_path,
            value=None,
            unit="",
            value_type=0,
            error=error_msg,
        )

    return TreeValueRecord(
        path=abs_path,
        rel_path=rel_path,
        value=value,
        unit=unit,
        value_type=value_type,
        error=None,
    )


def _error_record(abs_path: str, rel_path: str, error: str) -> TreeValueRecord:
    return TreeValueRecord(
        path=abs_path,
        rel_path=rel_path,
        value=None,
        unit="",
        value_type=0,
        error=error,
    )
