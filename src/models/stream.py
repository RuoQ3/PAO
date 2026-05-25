"""
stream.py — Aspen Plus 物流（stream）的业务层数据模型。

职责：描述一个 stream 的完整信息，包括：
  - 流股类型与相态
  - 热力学状态（温度、压力、气相分率）
  - 流量（总摩尔流量、总质量流量、总体积流量）
  - 组分流量与组成
  - 来源追溯（来自哪次仿真、哪个 case 文件）

与 aspen_driver.runner.BlockStatus 的区别
-----------------------------------------
BlockStatus 是驱动层的 HAP_COMPSTATUS 原始状态记录，仅用于
runner.py 内部的收敛判断。StreamResult 是业务层的完整结果快照，
供 workflow、database、agent 层持久化和分析使用。

Aspen 树路径约定（手册第 11/12 章）
-------------------------------------
\\Data\\Streams\\{name}\\Output\\TEMP          温度
\\Data\\Streams\\{name}\\Output\\PRES          压力
\\Data\\Streams\\{name}\\Output\\VFRAC         气相分率
\\Data\\Streams\\{name}\\Output\\MOLEFLMX\\MIXED  总摩尔流量
\\Data\\Streams\\{name}\\Output\\MASSFLMX\\MIXED  总质量流量
\\Data\\Streams\\{name}\\Output\\VOLFLMX\\MIXED   总体积流量
\\Data\\Streams\\{name}\\Output\\MOLEFLOW\\MIXED  组分摩尔流量（子节点为组分名）
\\Data\\Streams\\{name}\\Output\\MASSFLOW\\MIXED  组分质量流量
\\Data\\Streams\\{name}\\Output\\MOLEFRAC\\MIXED  摩尔分率
\\Data\\Streams\\{name}\\Output\\MASSFRAC\\MIXED  质量分率
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 流股类型枚举
# ---------------------------------------------------------------------------

class StreamType(str, Enum):
    """
    Aspen Plus 流股类型。

    值与 Aspen 树中 HAP_RECORDTYPE 返回的字符串对应。
    """
    MATERIAL  = "MATERIAL"   # 物料流股（最常见）
    HEAT      = "HEAT"       # 热流股（Q）
    WORK      = "WORK"       # 功流股（W）
    UNKNOWN   = "UNKNOWN"    # HAP_RECORDTYPE 未在枚举中定义；原始值保留在 StreamResult.raw_record_type


# ---------------------------------------------------------------------------
# 相态枚举
# ---------------------------------------------------------------------------

class PhaseState(str, Enum):
    """流股的相态，由气相分率（VFRAC）推断。"""
    VAPOR       = "vapor"       # 纯气相（VFRAC ≈ 1）
    LIQUID      = "liquid"      # 纯液相（VFRAC ≈ 0）
    TWO_PHASE   = "two_phase"   # 气液两相（0 < VFRAC < 1）
    UNKNOWN     = "unknown"     # VFRAC 未读取或不适用（热/功流股）

    @classmethod
    def from_vfrac(cls, vfrac: float | None, tol: float = 1e-6) -> PhaseState:
        """
        由气相分率推断相态。

        Parameters
        ----------
        vfrac:
            气相分率，None 表示未读取。
        tol:
            判断纯相的容差，默认 1e-6。
        """
        if vfrac is None:
            return cls.UNKNOWN
        if vfrac >= 1.0 - tol:
            return cls.VAPOR
        if vfrac <= tol:
            return cls.LIQUID
        return cls.TWO_PHASE


# ---------------------------------------------------------------------------
# 流股收敛状态
# ---------------------------------------------------------------------------

class StreamConvergenceStatus(str, Enum):
    """Stream 的收敛状态，从 HAP_COMPSTATUS 标志映射而来。"""
    SUCCESS    = "success"
    WARNINGS   = "warnings"
    ERRORS     = "errors"
    NO_RESULTS = "no_results"
    INCOMPAT   = "incompat"
    INACCESS   = "inaccess"
    UNKNOWN    = "unknown"

    @classmethod
    def from_status_flags(cls, flags: list[str]) -> StreamConvergenceStatus:
        """
        从 runner.BlockStatus.status_flags 映射到 StreamConvergenceStatus。

        优先级：ERRORS > UNKNOWN > INCOMPAT > INACCESS > NO_RESULTS > WARNINGS > SUCCESS
        """
        flag_set = set(flags)
        if "ERRORS"     in flag_set: return cls.ERRORS
        if "UNKNOWN"    in flag_set: return cls.UNKNOWN
        if "INCOMPAT"   in flag_set: return cls.INCOMPAT
        if "INACCESS"   in flag_set: return cls.INACCESS
        if "NO_RESULTS" in flag_set: return cls.NO_RESULTS
        if "WARNINGS"   in flag_set: return cls.WARNINGS
        if "SUCCESS"    in flag_set: return cls.SUCCESS
        return cls.UNKNOWN


# ---------------------------------------------------------------------------
# 组分流量/组成记录
# ---------------------------------------------------------------------------

@dataclass
class ComponentFlow:
    """单个组分的流量与组成数据。"""
    component: str          # 组分名称（与 Aspen 组分列表一致）
    mole_flow: float | None = None   # 摩尔流量
    mass_flow: float | None = None   # 质量流量
    mole_frac: float | None = None   # 摩尔分率
    mass_frac: float | None = None   # 质量分率
    mole_flow_unit: str = ""
    mass_flow_unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "mole_flow": self.mole_flow,
            "mole_flow_unit": self.mole_flow_unit,
            "mass_flow": self.mass_flow,
            "mass_flow_unit": self.mass_flow_unit,
            "mole_frac": self.mole_frac,
            "mass_frac": self.mass_frac,
        }


# ---------------------------------------------------------------------------
# StreamResult — 单次仿真中一个 stream 的完整结果快照
# ---------------------------------------------------------------------------

@dataclass
class StreamResult:
    """
    单次仿真运行中一个 stream 的完整结果快照。

    由 workflow 层在 run_case() 成功后构建，供数据库持久化和 agent 分析。

    Attributes
    ----------
    name:
        Stream 名称，与 Aspen 树中的节点名一致。
    stream_type:
        流股类型（StreamType 枚举）。
    convergence:
        收敛状态（StreamConvergenceStatus 枚举）。
    phase:
        相态（由 vfrac 推断）。
    temp:
        温度；未读取时为 None。
    temp_unit:
        温度单位字符串。
    pres:
        压力；未读取时为 None。
    pres_unit:
        压力单位字符串。
    vfrac:
        气相分率；热/功流股或未读取时为 None。
    total_mole_flow:
        总摩尔流量；未读取时为 None。
    total_mole_flow_unit:
        总摩尔流量单位字符串。
    total_mass_flow:
        总质量流量；未读取时为 None。
    total_mass_flow_unit:
        总质量流量单位字符串。
    total_vol_flow:
        总体积流量；未读取时为 None。
    total_vol_flow_unit:
        总体积流量单位字符串。
    components:
        各组分的流量与组成列表。
    comp_status:
        HAP_COMPSTATUS 原始整数值，供底层诊断使用。
    raw_record_type:
        HAP_RECORDTYPE 原始字符串，stream_type=UNKNOWN 时保留真实类型。
    source_filepath:
        产生本结果的 Aspen 仿真文件路径。
    run_id:
        关联的仿真运行 ID（由 workflow/database 层赋值）。
    notes:
        可选的人工或 agent 注释。
    """
    name: str
    stream_type: StreamType
    convergence: StreamConvergenceStatus
    phase: PhaseState = PhaseState.UNKNOWN
    # 热力学状态
    temp: float | None = None
    temp_unit: str = ""
    pres: float | None = None
    pres_unit: str = ""
    vfrac: float | None = None
    # 流量
    total_mole_flow: float | None = None
    total_mole_flow_unit: str = ""
    total_mass_flow: float | None = None
    total_mass_flow_unit: str = ""
    total_vol_flow: float | None = None
    total_vol_flow_unit: str = ""
    # 组分
    components: list[ComponentFlow] = field(default_factory=list)
    # 追溯字段
    comp_status: int = 0
    raw_record_type: str = ""
    source_filepath: Path | None = None
    run_id: str | None = None
    notes: str = ""

    # ------------------------------------------------------------------ #
    # 便捷属性
    # ------------------------------------------------------------------ #

    @property
    def converged(self) -> bool:
        """True 当且仅当收敛状态为 SUCCESS 或 WARNINGS。"""
        return self.convergence in (
            StreamConvergenceStatus.SUCCESS,
            StreamConvergenceStatus.WARNINGS,
        )

    def get_component(self, name: str) -> ComponentFlow | None:
        """按组分名查找，不存在时返回 None。"""
        for c in self.components:
            if c.component == name:
                return c
        return None

    def mole_fracs(self, allow_failed: bool = False) -> dict[str, float | None]:
        """
        返回 {组分名: 摩尔分率} 字典。

        Parameters
        ----------
        allow_failed:
            False（默认）：stream 未收敛时抛出 ValueError。
            True：跳过可信性检查（调试用）。
        """
        if not allow_failed and not self.converged:
            raise ValueError(
                f"Stream '{self.name}' 未收敛（convergence={self.convergence.value}），"
                "拒绝导出组成数据。如需强制导出（调试用），请传入 allow_failed=True。"
            )
        return {c.component: c.mole_frac for c in self.components}

    def mass_fracs(self, allow_failed: bool = False) -> dict[str, float | None]:
        """
        返回 {组分名: 质量分率} 字典。

        Parameters
        ----------
        allow_failed:
            False（默认）：stream 未收敛时抛出 ValueError。
        """
        if not allow_failed and not self.converged:
            raise ValueError(
                f"Stream '{self.name}' 未收敛（convergence={self.convergence.value}），"
                "拒绝导出组成数据。如需强制导出（调试用），请传入 allow_failed=True。"
            )
        return {c.component: c.mass_frac for c in self.components}

    def state_flat(self, allow_failed: bool = False) -> dict[str, Any]:
        """
        返回热力学状态的扁平字典（temp、pres、vfrac、total_mole_flow 等）。

        Parameters
        ----------
        allow_failed:
            False（默认）：stream 未收敛时抛出 ValueError。
        """
        if not allow_failed and not self.converged:
            raise ValueError(
                f"Stream '{self.name}' 未收敛（convergence={self.convergence.value}），"
                "拒绝导出状态数据。如需强制导出（调试用），请传入 allow_failed=True。"
            )
        return {
            "temp": self.temp,
            "temp_unit": self.temp_unit,
            "pres": self.pres,
            "pres_unit": self.pres_unit,
            "vfrac": self.vfrac,
            "phase": self.phase.value,
            "total_mole_flow": self.total_mole_flow,
            "total_mole_flow_unit": self.total_mole_flow_unit,
            "total_mass_flow": self.total_mass_flow,
            "total_mass_flow_unit": self.total_mass_flow_unit,
            "total_vol_flow": self.total_vol_flow,
            "total_vol_flow_unit": self.total_vol_flow_unit,
        }

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典，供数据库写入和日志记录。"""
        return {
            "name": self.name,
            "stream_type": self.stream_type.value,
            "raw_record_type": self.raw_record_type,
            "convergence": self.convergence.value,
            "converged": self.converged,
            "phase": self.phase.value,
            "temp": self.temp,
            "temp_unit": self.temp_unit,
            "pres": self.pres,
            "pres_unit": self.pres_unit,
            "vfrac": self.vfrac,
            "total_mole_flow": self.total_mole_flow,
            "total_mole_flow_unit": self.total_mole_flow_unit,
            "total_mass_flow": self.total_mass_flow,
            "total_mass_flow_unit": self.total_mass_flow_unit,
            "total_vol_flow": self.total_vol_flow,
            "total_vol_flow_unit": self.total_vol_flow_unit,
            "components": [c.to_dict() for c in self.components],
            "comp_status": self.comp_status,
            "source_filepath": str(self.source_filepath) if self.source_filepath else None,
            "run_id": self.run_id,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# 工厂函数：从 runner 结果构建 StreamResult
# ---------------------------------------------------------------------------

def stream_result_from_runner(
    stream_name: str,
    record_type: str,
    status_flags: list[str],
    comp_status: int,
    temp: float | None = None,
    temp_unit: str = "",
    pres: float | None = None,
    pres_unit: str = "",
    vfrac: float | None = None,
    total_mole_flow: float | None = None,
    total_mole_flow_unit: str = "",
    total_mass_flow: float | None = None,
    total_mass_flow_unit: str = "",
    total_vol_flow: float | None = None,
    total_vol_flow_unit: str = "",
    components: list[ComponentFlow] | None = None,
    source_filepath: Path | None = None,
    run_id: str | None = None,
) -> StreamResult:
    """
    从 runner.BlockStatus 和导出数据构建 StreamResult。

    Parameters
    ----------
    stream_name:
        Stream 名称。
    record_type:
        HAP_RECORDTYPE 字符串，用于映射 StreamType。
    status_flags:
        runner.BlockStatus.status_flags，用于映射 StreamConvergenceStatus。
    comp_status:
        HAP_COMPSTATUS 原始整数值。
    temp / pres / vfrac:
        热力学状态，由 workflow 层从 TreeExporter 或 VariableResult 填入。
    total_mole_flow / total_mass_flow / total_vol_flow:
        总流量，由 workflow 层填入。
    components:
        组分流量列表，由 workflow 层填入。
    source_filepath:
        来源仿真文件路径，从 SimulationResult.source_filepath 传入。
    run_id:
        关联的运行 ID。
    """
    try:
        stype = StreamType(record_type) if record_type else StreamType.UNKNOWN
    except ValueError:
        stype = StreamType.UNKNOWN

    convergence = StreamConvergenceStatus.from_status_flags(status_flags)
    phase = PhaseState.from_vfrac(vfrac)

    return StreamResult(
        name=stream_name,
        stream_type=stype,
        convergence=convergence,
        phase=phase,
        temp=temp,
        temp_unit=temp_unit,
        pres=pres,
        pres_unit=pres_unit,
        vfrac=vfrac,
        total_mole_flow=total_mole_flow,
        total_mole_flow_unit=total_mole_flow_unit,
        total_mass_flow=total_mass_flow,
        total_mass_flow_unit=total_mass_flow_unit,
        total_vol_flow=total_vol_flow,
        total_vol_flow_unit=total_vol_flow_unit,
        components=components or [],
        comp_status=comp_status,
        raw_record_type=record_type,
        source_filepath=source_filepath,
        run_id=run_id,
    )
