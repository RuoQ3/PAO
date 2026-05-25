"""
simulation_result.py — 仿真运行结果的数据模型。

包含 SimulationRunner.run_case() 返回的所有结构化数据类，
以及驱动状态检查的辅助类型。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 结果状态枚举
# ---------------------------------------------------------------------------

class RunStatus(str, Enum):
    """仿真运行结果的整体状态，供上层 agent 做失败分类。"""
    SUCCESS            = "success"            # 所有 block/stream 收敛，输出可信
    WARNINGS           = "warnings"           # 收敛但有警告，结果可用，建议降权
    ERRORS             = "errors"             # block/stream 有错误，结果不可信
    NO_RESULTS         = "no_results"         # 未产生结果（未运行或被跳过）
    INCOMPAT           = "incompat"           # 结果与输入不兼容（需重新运行）
    INACCESS           = "inaccess"           # 结果不可访问
    RUN_FAILED         = "run_failed"         # 引擎层面失败（超时、COM 错误）
    WRITE_FAILED       = "write_failed"       # 输入变量写入失败
    STATUS_UNAVAILABLE = "status_unavailable" # hap_constants 未加载，无法验证结果

    @property
    def is_convergent(self) -> bool:
        """True 当且仅当仿真收敛（SUCCESS 或 WARNINGS），结果可信。"""
        return self in (RunStatus.SUCCESS, RunStatus.WARNINGS)


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class VariableResult:
    """单个输出变量的读取结果。"""
    path: str
    value: Any
    unit: str
    value_type: int     # ValueType 整数值

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "value": self.value, "unit": self.unit, "value_type": self.value_type}


@dataclass
class InputVerification:
    """单个输入变量的写入校验结果。"""
    path: str
    requested: Any          # 调用方请求写入的值
    actual: Any             # 写入后从 Aspen 读回的值
    match: bool             # 是否在容差内一致
    note: str = ""          # 不一致时的说明

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "requested": self.requested, "actual": self.actual, "match": self.match, "note": self.note}


@dataclass
class BlockStatus:
    """单个 block 或 stream 的结果状态。"""
    name: str
    record_type: str        # HAP_RECORDTYPE，如 "RADFRAC"/"MATERIAL"
    comp_status: int        # HAP_COMPSTATUS 原始整数值
    status_flags: list[str] = field(default_factory=list)  # 解析后的标志名列表
    # status_flags 为空时表示无法解析（位掩码常量缺失或 comp_status 无匹配位）

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "record_type": self.record_type, "comp_status": self.comp_status, "status_flags": self.status_flags}


@dataclass
class StatusCheckResult:
    """_check_statuses() 的完整返回，含失败记录。"""
    statuses: list[BlockStatus] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # {path: 错误信息}
    unavailable: bool = False  # hap_constants 缺失或 HAP_COMPSTATUS 缺失
    explicitly_skipped: bool = False  # 调用方传 paths=[] 显式跳过检查


@dataclass
class SimulationResult:
    """
    一次仿真运行的完整结果。

    Attributes
    ----------
    status:
        整体运行状态（RunStatus 枚举），供上层 agent 做失败分类。
    success:
        True 当且仅当 status == SUCCESS 或 WARNINGS，且所有输出均已读取。
    requested_inputs:
        调用方请求写入的输入变量 {path: value}。
    actual_inputs:
        写入后从 Aspen 读回的实际值 {path: value}。
        若 verify_inputs=False 则与 requested_inputs 相同。
    input_verifications:
        每个输入变量的写入校验详情，含 match 标志。
    outputs:
        成功读取的输出变量 {path: VariableResult}。
    failed_outputs:
        读取失败的输出变量 {path: 错误信息}。
    block_statuses:
        检查的 block/stream 结果状态列表。
    run_time:
        仿真运行耗时（秒），不含 reinit 和读取时间。
    error:
        导致失败的主要错误信息；success=True 时为 None。
    warnings:
        警告信息列表（block 有 WARNING 状态时填入）。
    """
    status: RunStatus
    success: bool
    requested_inputs: dict[str, Any]
    actual_inputs: dict[str, Any] = field(default_factory=dict)
    input_verifications: list[InputVerification] = field(default_factory=list)
    outputs: dict[str, VariableResult] = field(default_factory=dict)
    failed_outputs: dict[str, str] = field(default_factory=dict)
    block_statuses: list[BlockStatus] = field(default_factory=list)
    run_time: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    # 用于 TreeExporter 一致性校验：记录产生本结果时的仿真文件路径和 driver mutation counter 快照。
    # mutation_snapshot 与 driver.mutation_count 不一致说明 run 后有输入被修改。
    source_filepath: Path | None = None
    result_time: float = field(default_factory=time.monotonic)
    mutation_snapshot: int | None = None

    def summary(self) -> dict[str, Any]:
        """
        返回轻量摘要字典，供 ProcessCase.to_dict() 等上层聚合使用。

        不含 inputs/outputs/block_statuses 等大字段；
        完整序列化请使用 ResultExporter(self).to_dict()。
        """
        return {
            "status": self.status.value,
            "success": self.success,
            "run_time": self.run_time,
            "source_filepath": str(self.source_filepath) if self.source_filepath else None,
            "result_time": self.result_time,
            "mutation_snapshot": self.mutation_snapshot,
            "error": self.error,
            "warnings": self.warnings,
        }
