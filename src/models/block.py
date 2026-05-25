"""
block.py — Aspen Plus 单元操作 block 的业务层数据模型。

职责：描述一个 block 的完整信息，包括：
  - 静态配置（名称、类型、输入参数）
  - 运行结果（输出变量、收敛状态）
  - 来源追溯（来自哪次仿真、哪个 case 文件）

与 aspen_driver.runner.BlockStatus 的区别
-----------------------------------------
BlockStatus 是驱动层的 HAP_COMPSTATUS 原始状态记录，仅用于
runner.py 内部的收敛判断。BlockResult 是业务层的完整结果快照，
供 workflow、database、agent 层持久化和分析使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Block 类型枚举
# ---------------------------------------------------------------------------

class BlockType(str, Enum):
    """
    Aspen Plus 单元操作类型。

    值与 Aspen 树中 HAP_RECORDTYPE 返回的字符串对应，
    便于从 runner 结果直接映射，不需要额外转换。
    """
    # 分离操作
    RADFRAC     = "RADFRAC"      # 严格精馏
    DISTL       = "DISTL"        # 简捷精馏
    EXTRACT     = "EXTRACT"      # 液液萃取
    FLASH2      = "FLASH2"       # 两相闪蒸
    FLASH3      = "FLASH3"       # 三相闪蒸
    DECANTER    = "DECANTER"     # 倾析器
    SEP         = "SEP"          # 组分分离器
    SEP2        = "SEP2"         # 两出口分离器
    # 换热操作
    HEATER      = "HEATER"       # 加热器/冷却器
    HEATX       = "HEATX"        # 换热器
    MHEATX      = "MHEATX"       # 多股流换热器
    # 反应操作
    RSTOIC      = "RSTOIC"       # 化学计量反应器
    RYIELD      = "RYIELD"       # 产率反应器
    REQUIL      = "REQUIL"       # 平衡反应器
    RGIBBS      = "RGIBBS"       # Gibbs 自由能最小化反应器
    RCSTR       = "RCSTR"        # 连续搅拌釜反应器
    RPLUG       = "RPLUG"        # 活塞流反应器
    RBATCH      = "RBATCH"       # 间歇反应器
    # 压力变化操作
    PUMP        = "PUMP"         # 泵
    COMPR       = "COMPR"        # 压缩机
    MCOMPR      = "MCOMPR"       # 多级压缩机
    VALVE       = "VALVE"        # 阀门
    PIPE        = "PIPE"         # 管道
    PIPELINE    = "PIPELINE"     # 管线
    # 固体操作
    CRUSHER     = "CRUSHER"      # 破碎机
    SCREEN      = "SCREEN"       # 筛分器
    CYCLONE     = "CYCLONE"      # 旋风分离器
    FABFL       = "FABFL"        # 布袋过滤器
    CRYSTALLIZER = "CRYSTALLIZER" # 结晶器
    # 流股操作
    MIXER       = "MIXER"        # 混合器
    FSPLIT      = "FSPLIT"       # 流股分割器
    SSPLIT      = "SSPLIT"       # 子流股分割器
    # 用户自定义
    USER        = "USER"         # 用户模型
    USER2       = "USER2"        # 用户模型 2
    # 未知/其他
    UNKNOWN     = "UNKNOWN"      # HAP_RECORDTYPE 未在枚举中定义；原始值保留在 BlockResult.raw_record_type


# ---------------------------------------------------------------------------
# Block 输入参数
# ---------------------------------------------------------------------------

@dataclass
class BlockInput:
    """
    单个 block 输入参数的描述与当前值。

    path 是 Aspen 树的绝对路径，name 是参数的业务名称（如 "TEMP"、"PRES"）。
    unit 来自 IHNode.UnitString，为空字符串表示无量纲或未知单位。
    """
    path: str
    name: str
    value: Any
    unit: str = ""
    value_type: int = 0     # ValueType 整数值（0=UNDEFINED, 1=INT, 2=REAL, 3=STR）
    description: str = ""   # 可选的人工描述，供 agent 理解参数含义


# ---------------------------------------------------------------------------
# Block 输出变量
# ---------------------------------------------------------------------------

@dataclass
class BlockOutput:
    """
    单个 block 输出变量的读取结果。

    与 VariableResult 的区别：BlockOutput 是业务层概念，
    包含 name（业务名称）和 description，适合入库和 agent 分析；
    VariableResult 是驱动层的原始读取记录。
    """
    path: str
    name: str
    value: Any
    unit: str = ""
    value_type: int = 0     # ValueType 整数值（0=UNDEFINED, 1=INT, 2=REAL, 3=STR）
    description: str = ""


# ---------------------------------------------------------------------------
# Block 收敛状态
# ---------------------------------------------------------------------------

class BlockConvergenceStatus(str, Enum):
    """Block 的收敛状态，从 HAP_COMPSTATUS 标志映射而来。"""
    SUCCESS    = "success"    # 收敛，结果可信
    WARNINGS   = "warnings"  # 收敛但有警告，结果可用，建议降权
    ERRORS     = "errors"    # 未收敛或有错误，结果不可信
    NO_RESULTS = "no_results" # 未产生结果
    INCOMPAT   = "incompat"  # 结果与输入不兼容（需重新运行）
    INACCESS   = "inaccess"  # 结果不可访问
    UNKNOWN    = "unknown"   # 无法从 HAP_COMPSTATUS 解析

    @classmethod
    def from_status_flags(cls, flags: list[str]) -> BlockConvergenceStatus:
        """
        从 runner.BlockStatus.status_flags 映射到 BlockConvergenceStatus。

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
# BlockResult — 单次仿真中一个 block 的完整结果快照
# ---------------------------------------------------------------------------

@dataclass
class BlockResult:
    """
    单次仿真运行中一个 block 的完整结果快照。

    由 workflow 层在 run_case() 成功后构建，供数据库持久化和 agent 分析。

    Attributes
    ----------
    name:
        Block 名称，与 Aspen 树中的节点名一致。
    block_type:
        单元操作类型（BlockType 枚举）。
    convergence:
        收敛状态（BlockConvergenceStatus 枚举）。
    inputs:
        本次运行写入的输入参数列表。
    outputs:
        本次运行读取的输出变量列表。
    comp_status:
        HAP_COMPSTATUS 原始整数值，供底层诊断使用。
    source_filepath:
        产生本结果的 Aspen 仿真文件路径。
    run_id:
        关联的仿真运行 ID（由 workflow/database 层赋值）。
    notes:
        可选的人工或 agent 注释。
    """
    name: str
    block_type: BlockType
    convergence: BlockConvergenceStatus
    inputs: list[BlockInput] = field(default_factory=list)
    outputs: list[BlockOutput] = field(default_factory=list)
    comp_status: int = 0
    raw_record_type: str = ""   # HAP_RECORDTYPE 原始字符串，block_type=UNKNOWN 时保留真实类型
    source_filepath: Path | None = None
    run_id: str | None = None
    notes: str = ""

    # ------------------------------------------------------------------ #
    # 便捷访问
    # ------------------------------------------------------------------ #

    @property
    def converged(self) -> bool:
        """True 当且仅当收敛状态为 SUCCESS 或 WARNINGS。"""
        return self.convergence in (
            BlockConvergenceStatus.SUCCESS,
            BlockConvergenceStatus.WARNINGS,
        )

    def get_output(self, name: str) -> BlockOutput | None:
        """按名称查找输出变量，不存在时返回 None。"""
        for o in self.outputs:
            if o.name == name:
                return o
        return None

    def get_input(self, name: str) -> BlockInput | None:
        """按名称查找输入参数，不存在时返回 None。"""
        for i in self.inputs:
            if i.name == name:
                return i
        return None

    def outputs_flat(self, allow_failed: bool = False) -> dict[str, Any]:
        """
        返回 {name: value} 的扁平字典，仅包含成功读取的输出。

        Parameters
        ----------
        allow_failed:
            False（默认）：block 未收敛时抛出 ValueError，防止残留/部分输出
            被当作有效数据用于优化或经济分析。
            True：跳过可信性检查（调试用）。
        """
        if not allow_failed and not self.converged:
            raise ValueError(
                f"Block '{self.name}' 未收敛（convergence={self.convergence.value}），"
                "拒绝导出输出值。如需强制导出（调试用），请传入 allow_failed=True。"
            )
        return {o.name: o.value for o in self.outputs}

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典，供数据库写入和日志记录。"""
        return {
            "name": self.name,
            "block_type": self.block_type.value,
            "raw_record_type": self.raw_record_type,
            "convergence": self.convergence.value,
            "converged": self.converged,
            "comp_status": self.comp_status,
            "source_filepath": str(self.source_filepath) if self.source_filepath else None,
            "run_id": self.run_id,
            "notes": self.notes,
            "inputs": [
                {
                    "path": i.path,
                    "name": i.name,
                    "value": i.value,
                    "unit": i.unit,
                    "value_type": i.value_type,
                    "description": i.description,
                }
                for i in self.inputs
            ],
            "outputs": [
                {
                    "path": o.path,
                    "name": o.name,
                    "value": o.value,
                    "unit": o.unit,
                    "value_type": o.value_type,
                    "description": o.description,
                }
                for o in self.outputs
            ],
        }


# ---------------------------------------------------------------------------
# 工厂函数：从 runner 结果构建 BlockResult
# ---------------------------------------------------------------------------

def block_result_from_runner(
    block_name: str,
    record_type: str,
    status_flags: list[str],
    comp_status: int,
    inputs: list[BlockInput] | None = None,
    outputs: list[BlockOutput] | None = None,
    source_filepath: Path | None = None,
    run_id: str | None = None,
) -> BlockResult:
    """
    从 runner.BlockStatus 和导出数据构建 BlockResult。

    Parameters
    ----------
    block_name:
        Block 名称。
    record_type:
        HAP_RECORDTYPE 字符串，用于映射 BlockType。
    status_flags:
        runner.BlockStatus.status_flags，用于映射 BlockConvergenceStatus。
    comp_status:
        HAP_COMPSTATUS 原始整数值。
    inputs:
        输入参数列表，由 workflow 层从 run_case() 的 inputs 字典构建。
    outputs:
        输出变量列表，由 workflow 层从 TreeExporter 或 VariableResult 构建。
    source_filepath:
        来源仿真文件路径，从 SimulationResult.source_filepath 传入。
    run_id:
        关联的运行 ID。
    """
    # Aspen 返回的 HAP_RECORDTYPE 大小写不一致（如 "RadFrac" vs 枚举值 "RADFRAC"），
    # 先 strip().upper() 标准化后再匹配；原始值保留在 raw_record_type 供追溯。
    try:
        btype = BlockType(record_type.strip().upper()) if record_type else BlockType.UNKNOWN
    except ValueError:
        btype = BlockType.UNKNOWN

    convergence = BlockConvergenceStatus.from_status_flags(status_flags)

    return BlockResult(
        name=block_name,
        block_type=btype,
        convergence=convergence,
        inputs=inputs or [],
        outputs=outputs or [],
        comp_status=comp_status,
        raw_record_type=record_type,
        source_filepath=source_filepath,
        run_id=run_id,
    )
