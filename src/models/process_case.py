"""
process_case.py — 一次完整工况运行的业务层聚合模型。

职责：将一次 Aspen Plus 仿真运行的所有信息聚合为单一对象，包括：
  - 优化输入参数点（设计变量）
  - 仿真运行元数据（来源文件、运行时间、状态）
  - 所有 block 和 stream 的结果快照
  - 优化层需要的目标函数值和约束值

层级关系
---------
ProcessCase（本文件）
  └── SimulationResult（aspen_driver 层，驱动层原始结果）
  └── BlockResult（models/block.py，业务层 block 快照）
  └── StreamResult（models/stream.py，业务层 stream 快照）

ProcessCase 是贝叶斯优化循环的数据单元：
  - 每次 run_case() 对应一个 ProcessCase
  - 数据库以 ProcessCase 为粒度存储和检索
  - agent 以 ProcessCase 为单位做失败归因和参数推荐
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .block import BlockResult
from .node_catalog import SemanticBlock
from .simulation_result import RunStatus, SimulationResult
from .stream import StreamResult


# ---------------------------------------------------------------------------
# 工况状态枚举
# ---------------------------------------------------------------------------

class CaseStatus(str, Enum):
    """
    一次工况运行的整体状态，供优化循环和数据库做分类筛选。

    与 RunStatus 的区别：RunStatus 是驱动层的仿真引擎状态；
    CaseStatus 是业务层的工况状态，额外包含优化层的语义
    （如目标函数/约束是否可用、是否被优化器采纳）。
    """
    SUCCESS           = "success"           # 仿真收敛，目标函数和约束均可用且满足
    WARNINGS          = "warnings"          # 仿真收敛但有警告，结果可用，建议降权
    SIM_FAILED        = "sim_failed"        # 仿真失败（引擎错误、超时、写入失败等）
    INFEASIBLE        = "infeasible"        # 仿真收敛但约束违反，不可行点
    OBJECTIVE_ERROR   = "objective_error"   # 仿真收敛但目标函数计算失败
    CONSTRAINT_ERROR  = "constraint_error"  # 仿真收敛但约束计算失败（无法判断可行性）
    PENDING           = "pending"           # 已创建但尚未运行

    @classmethod
    def from_run_status(cls, run_status: RunStatus) -> CaseStatus:
        """从 SimulationResult.status 映射到 CaseStatus（初步映射，不含约束检查）。"""
        if run_status.is_convergent:
            return cls.SUCCESS if run_status == RunStatus.SUCCESS else cls.WARNINGS
        return cls.SIM_FAILED


# ---------------------------------------------------------------------------
# 目标函数值
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveValue:
    """
    单个优化目标函数的计算结果。

    Attributes
    ----------
    name:
        目标函数名称，如 "TAC"（总年化成本）、"energy"（能耗）。
    value:
        目标函数值；计算失败时为 None。
    unit:
        单位字符串。
    minimize:
        True 表示最小化，False 表示最大化。
    error:
        计算失败原因；成功时为 None。
    """
    name: str
    value: float | None
    unit: str = ""
    minimize: bool = True
    error: str | None = None

    @property
    def available(self) -> bool:
        """目标函数值是否可用（非 None 且无错误）。"""
        return self.value is not None and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "minimize": self.minimize,
            "available": self.available,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# 约束值
# ---------------------------------------------------------------------------

@dataclass
class ConstraintValue:
    """
    单个约束的计算结果。

    约束形式：value <= 0（满足约束）或 value > 0（违反约束）。
    调用方负责将实际约束转换为此标准形式。

    Attributes
    ----------
    name:
        约束名称，如 "purity_min"、"temp_max"。
    value:
        约束值（标准化后）；计算失败时为 None。
    satisfied:
        True 表示约束满足（value <= 0）；None 表示无法判断。
    error:
        计算失败原因；成功时为 None。
    """
    name: str
    value: float | None
    satisfied: bool | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        # value 有效且未手动指定 satisfied 时，按 value <= 0 自动推断
        if self.satisfied is None and self.value is not None and self.error is None:
            self.satisfied = self.value <= 0

    @property
    def available(self) -> bool:
        """约束值是否可用。"""
        return self.value is not None and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "satisfied": self.satisfied,
            "available": self.available,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# ProcessCase — 一次完整工况运行的聚合快照
# ---------------------------------------------------------------------------

@dataclass
class ProcessCase:
    """
    一次完整工况运行的聚合快照。

    Attributes
    ----------
    case_id:
        工况唯一标识符，默认自动生成 UUID。
    iteration:
        优化迭代编号（从 0 开始），由优化循环赋值。
    status:
        工况整体状态（CaseStatus 枚举）。
    design_vars:
        本次运行的设计变量 {参数名: 值}，由优化器提供。
    sim_result:
        驱动层仿真结果（SimulationResult），含原始输入/输出和状态。
        None 表示尚未运行（status=PENDING）。
    blocks:
        所有 block 的业务层结果快照 {block_name: BlockResult}。
    streams:
        所有 stream 的业务层结果快照 {stream_name: StreamResult}。
    objectives:
        目标函数值列表，由 workflow 层在仿真后计算填入。
    constraints:
        约束值列表，由 workflow 层在仿真后计算填入。
    source_filepath:
        产生本工况的 Aspen 仿真文件路径。
    run_id:
        关联的底层运行 ID（可与数据库外键对应）。
    tags:
        可选标签，供 agent 分类和检索（如 "initial_doe"、"exploitation"）。
    notes:
        可选的人工或 agent 注释。
    """
    case_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    iteration: int = 0
    status: CaseStatus = CaseStatus.PENDING
    design_vars: dict[str, Any] = field(default_factory=dict)
    sim_result: SimulationResult | None = None
    blocks: dict[str, BlockResult] = field(default_factory=dict)
    streams: dict[str, StreamResult] = field(default_factory=dict)
    objectives: list[ObjectiveValue] = field(default_factory=list)
    constraints: list[ConstraintValue] = field(default_factory=list)
    source_filepath: Path | None = None
    run_id: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    # manifest runtime 模式产出的语义字段集合，{block_name: SemanticBlock}
    # 非 manifest 模式时为空字典
    semantic_blocks: dict[str, SemanticBlock] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # 便捷属性
    # ------------------------------------------------------------------ #

    @property
    def simulation_valid(self) -> bool:
        """
        True 当且仅当仿真本身收敛（不论约束和目标是否可用）。

        对应 status 为 SUCCESS / WARNINGS / INFEASIBLE / OBJECTIVE_ERROR。
        用于判断 block/stream 数据是否可信，以及是否值得入库追溯。
        """
        return self.status in (
            CaseStatus.SUCCESS,
            CaseStatus.WARNINGS,
            CaseStatus.INFEASIBLE,
            CaseStatus.OBJECTIVE_ERROR,
            CaseStatus.CONSTRAINT_ERROR,
        )

    @property
    def objectives_available(self) -> bool:
        """True 当且仅当至少有一个目标函数且全部可用。"""
        return bool(self.objectives) and all(o.available for o in self.objectives)

    @property
    def has_constraints(self) -> bool:
        """True 当且仅当约束列表非空。"""
        return bool(self.constraints)

    @property
    def constraints_available(self) -> bool:
        """True 当且仅当约束列表非空且全部可用（value 非 None 且无 error）。"""
        return bool(self.constraints) and all(c.available for c in self.constraints)

    @property
    def success(self) -> bool:
        """
        True 当且仅当可作为优化有效目标样本：
          - 仿真收敛（status 为 SUCCESS 或 WARNINGS）
          - 至少有一个目标函数且全部可用
          - 无约束，或约束全部可用且全部满足（feasible is True）

        INFEASIBLE（约束违反）、CONSTRAINT_ERROR（约束计算失败）均不算 success，
        但 simulation_valid=True，目标和约束值仍可通过 objectives_flat/constraints_flat 导出。
        """
        if self.status not in (CaseStatus.SUCCESS, CaseStatus.WARNINGS):
            return False
        if not self.objectives_available:
            return False
        # 有约束时：必须全部可用且全部满足
        if self.has_constraints:
            return self.constraints_available and self.feasible is True
        return True

    @property
    def feasible(self) -> bool | None:
        """
        True 表示所有约束满足；False 表示至少一个约束违反；
        None 表示约束列表为空或存在无法计算的约束。
        """
        if not self.constraints:
            return None
        if any(c.satisfied is None for c in self.constraints):
            return None
        return all(c.satisfied for c in self.constraints)

    @property
    def run_time(self) -> float:
        """仿真运行耗时（秒），来自 sim_result；未运行时为 0.0。"""
        return self.sim_result.run_time if self.sim_result else 0.0

    # ------------------------------------------------------------------ #
    # 目标函数与约束访问
    # ------------------------------------------------------------------ #

    def get_objective(self, name: str) -> ObjectiveValue | None:
        """按名称查找目标函数，不存在时返回 None。"""
        for o in self.objectives:
            if o.name == name:
                return o
        return None

    def get_constraint(self, name: str) -> ConstraintValue | None:
        """按名称查找约束，不存在时返回 None。"""
        for c in self.constraints:
            if c.name == name:
                return c
        return None

    def objectives_flat(self, allow_failed: bool = False) -> dict[str, float | None]:
        """
        返回 {目标名: 值} 的扁平字典。

        允许 INFEASIBLE 且目标可用的工况导出（不可行点的目标值对约束优化有价值）。

        Parameters
        ----------
        allow_failed:
            False（默认）：仿真未收敛或目标不可用时抛出 ValueError。
            True：跳过可信性检查（调试用）。
        """
        if not allow_failed:
            if not self.simulation_valid:
                raise ValueError(
                    f"工况 '{self.case_id}' 仿真未收敛（status={self.status.value}），"
                    "拒绝导出目标函数值。如需强制导出（调试用），请传入 allow_failed=True。"
                )
            if not self.objectives_available:
                raise ValueError(
                    f"工况 '{self.case_id}' 目标函数不可用（objectives 为空或存在计算失败），"
                    "拒绝导出。如需强制导出（调试用），请传入 allow_failed=True。"
                )
        return {o.name: o.value for o in self.objectives}

    def constraints_flat(self, allow_failed: bool = False) -> dict[str, float | None]:
        """
        返回 {约束名: 值} 的扁平字典。

        无约束时返回 {}（无约束优化是合理场景，不视为错误）。
        允许 INFEASIBLE 工况导出约束值（不可行边界对约束优化至关重要）。

        Parameters
        ----------
        allow_failed:
            False（默认）：仿真未收敛或约束存在但不可用时抛出 ValueError。
        """
        if not allow_failed:
            if not self.simulation_valid:
                raise ValueError(
                    f"工况 '{self.case_id}' 仿真未收敛（status={self.status.value}），"
                    "拒绝导出约束值。如需强制导出（调试用），请传入 allow_failed=True。"
                )
            if self.has_constraints and not self.constraints_available:
                raise ValueError(
                    f"工况 '{self.case_id}' 存在约束但部分不可用（存在计算失败），"
                    "拒绝导出。如需强制导出（调试用），请传入 allow_failed=True。"
                )
        return {c.name: c.value for c in self.constraints}

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #

    def to_dict(self, include_sim_result: bool = True) -> dict[str, Any]:
        """
        序列化为可 JSON 化的字典，供数据库写入和日志记录。

        Parameters
        ----------
        include_sim_result:
            True（默认）：包含 sim_result 的追溯摘要
            （status、success、run_time、source_filepath、mutation_snapshot、error、warnings）。
            如需完整的 inputs/outputs/block_statuses，请使用
            ResultExporter(self.sim_result).to_dict()。
            False：不包含 sim_result 字段，减少序列化体积。
        """
        sim_dict: dict[str, Any] | None = None
        if self.sim_result is not None and include_sim_result:
            sim_dict = self.sim_result.summary()

        return {
            "case_id": self.case_id,
            "iteration": self.iteration,
            "status": self.status.value,
            "simulation_valid": self.simulation_valid,
            "success": self.success,
            "feasible": self.feasible,
            "has_constraints": self.has_constraints,
            "objectives_available": self.objectives_available,
            "constraints_available": self.constraints_available,
            "run_time": self.run_time,
            "design_vars": self.design_vars,
            "objectives": [o.to_dict() for o in self.objectives],
            "constraints": [c.to_dict() for c in self.constraints],
            "source_filepath": str(self.source_filepath) if self.source_filepath else None,
            "run_id": self.run_id,
            "tags": self.tags,
            "notes": self.notes,
            "sim_result": sim_dict,
            "blocks": {name: b.to_dict() for name, b in self.blocks.items()},
            "streams": {name: s.to_dict() for name, s in self.streams.items()},
        }

    def summary(self) -> dict[str, Any]:
        """
        返回工况摘要（不含 block/stream 详情），供优化循环快速检索。

        包含：case_id、iteration、status、simulation_valid、success、feasible、
        objectives_available、constraints_available、run_time、design_vars、
        objectives、constraints、tags。
        """
        return {
            "case_id": self.case_id,
            "iteration": self.iteration,
            "status": self.status.value,
            "simulation_valid": self.simulation_valid,
            "success": self.success,
            "feasible": self.feasible,
            "has_constraints": self.has_constraints,
            "objectives_available": self.objectives_available,
            "constraints_available": self.constraints_available,
            "run_time": self.run_time,
            "design_vars": self.design_vars,
            "objectives": {o.name: o.value for o in self.objectives},
            "constraints": {c.name: c.value for c in self.constraints},
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# 工厂函数：从 SimulationResult 构建 ProcessCase
# ---------------------------------------------------------------------------

def process_case_from_sim_result(
    sim_result: SimulationResult,
    design_vars: dict[str, Any],
    iteration: int = 0,
    blocks: dict[str, BlockResult] | None = None,
    streams: dict[str, StreamResult] | None = None,
    objectives: list[ObjectiveValue] | None = None,
    constraints: list[ConstraintValue] | None = None,
    tags: list[str] | None = None,
    run_id: str | None = None,
    semantic_blocks: dict[str, SemanticBlock] | None = None,
) -> ProcessCase:
    """
    从 SimulationResult 构建 ProcessCase。

    Parameters
    ----------
    sim_result:
        runner.run_case() 返回的仿真结果。
    design_vars:
        本次运行的设计变量 {参数名: 值}，由优化器提供。
    iteration:
        优化迭代编号。
    blocks:
        block 结果快照字典，由 workflow 层从 TreeExporter 构建。
    streams:
        stream 结果快照字典，由 workflow 层从 TreeExporter 构建。
    objectives:
        目标函数值列表，由 workflow 层计算后传入。
    constraints:
        约束值列表，由 workflow 层计算后传入。
    tags:
        可选标签列表。
    run_id:
        关联的底层运行 ID。
    """
    status = CaseStatus.from_run_status(sim_result.status)

    if status in (CaseStatus.SUCCESS, CaseStatus.WARNINGS) and constraints:
        # 约束计算失败（value=None 或有 error）→ CONSTRAINT_ERROR
        if any(not c.available for c in constraints):
            status = CaseStatus.CONSTRAINT_ERROR
        # 约束全部可用但存在违反 → INFEASIBLE
        elif any(c.satisfied is False for c in constraints):
            status = CaseStatus.INFEASIBLE

    # 仿真收敛但目标函数计算失败 → OBJECTIVE_ERROR（约束错误优先级更高，不覆盖）
    if status in (CaseStatus.SUCCESS, CaseStatus.WARNINGS) and objectives:
        if any(not o.available for o in objectives):
            status = CaseStatus.OBJECTIVE_ERROR

    return ProcessCase(
        iteration=iteration,
        status=status,
        design_vars=design_vars,
        sim_result=sim_result,
        blocks=blocks or {},
        streams=streams or {},
        objectives=objectives or [],
        constraints=constraints or [],
        source_filepath=sim_result.source_filepath,
        run_id=run_id,
        tags=tags or [],
        semantic_blocks=semantic_blocks or {},
    )
