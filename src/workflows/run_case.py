"""
run_case.py — 单次工况运行的 workflow 层封装。

职责：
  1. 调用 SimulationRunner.run_case() 执行仿真
  2. 调用 TreeExporter 提取 block/stream 结构化数据
  3. 构建 BlockResult / StreamResult 快照
  4. 调用目标函数和约束函数计算优化指标
  5. 组装并返回 ProcessCase

层级关系
---------
run_case()（本文件）
  ├── SimulationRunner.run_case()     → SimulationResult
  ├── TreeExporter                    → block/stream 原始记录
  ├── _extract_blocks()               → dict[str, BlockResult]
  ├── _extract_streams()              → dict[str, StreamResult]
  ├── _compute_objectives/constraints → list[ObjectiveValue/ConstraintValue]
  └── process_case_from_sim_result()  → ProcessCase

目标函数与约束函数约定
-----------------------
ObjectiveFn : (ProcessCase) -> ObjectiveValue
ConstraintFn: (ProcessCase) -> ConstraintValue

传入的 ProcessCase 已填充 sim_result / blocks / streams，
但 objectives / constraints 尚为空，status 为初步映射值。
函数内部不应修改 case，只读取数据并返回计算结果。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

from ..aspen_driver.driver import AspenDriver
from ..aspen_driver.exporter import TreeExporter, TreeValueRecord
from ..aspen_driver.runner import SimulationRunner
from ..models.block import BlockInput, BlockOutput, BlockResult, block_result_from_runner
from ..models.process_case import (
    CaseStatus,
    ConstraintValue,
    ObjectiveValue,
    ProcessCase,
    process_case_from_sim_result,
)
from ..models.simulation_result import SimulationResult
from ..models.stream import ComponentFlow, StreamResult, stream_result_from_runner

_log = logging.getLogger(__name__)

ObjectiveFn  = Callable[[ProcessCase], ObjectiveValue]
ConstraintFn = Callable[[ProcessCase], ConstraintValue]


class _ExtractionResult(NamedTuple):
    """block/stream 提取的结果，含失败诊断信息。"""
    data: dict          # dict[str, BlockResult] 或 dict[str, StreamResult]
    failed_nodes: dict[str, str]   # {节点名/路径: 错误信息}，提取部分失败时非空
    fatal_error: str | None        # 整批提取失败时的错误信息；None 表示无致命错误


# ---------------------------------------------------------------------------
# 运行配置
# ---------------------------------------------------------------------------

@dataclass
class RunCaseConfig:
    """
    run_case() 的配置参数。

    将仿真配置与每次调用的可变参数（design_vars、iteration）分离，
    便于在优化循环中复用同一配置对象。

    Attributes
    ----------
    output_paths:
        需要从 Aspen 树读取的输出节点路径列表，传给 SimulationRunner。
    objective_fns:
        目标函数列表，每个函数签名为 (ProcessCase) -> ObjectiveValue。
    constraint_fns:
        约束函数列表，每个函数签名为 (ProcessCase) -> ConstraintValue。
    timeout:
        仿真超时（秒），默认 300s。
    reinit:
        运行前是否 reinit，默认 True。
    verify_inputs:
        写入后是否读回校验，默认 True。
    input_rtol:
        输入读回校验的相对容差，默认 1e-6。
    check_status_paths:
        需要检查 HAP_COMPSTATUS 的节点路径列表。
        None（默认）：自动枚举所有 block/stream。
        []：跳过状态检查。
    extract_blocks:
        需要提取 Output 子树的 block 名称列表。
        None（默认）：自动枚举所有 block。
        []：跳过 block 提取。
    extract_streams:
        需要提取的 stream 名称列表。
        None（默认）：自动枚举所有 stream。
        []：跳过 stream 提取。
    block_max_depth:
        block Output 子树的递归深度，默认 3（覆盖 B_K 等三级嵌套节点）。
    stream_max_depth:
        stream 子树的递归深度，默认 3（配合 stream_output_subtree 使用）。
    stream_output_subtree:
        stream 提取的子树根节点，相对于 \Data\Streams\{name}。
        默认 "Output\\STR_MAIN"：Aspen Plus 将完整流股数据存放在 STR_MAIN 子树，
        路径格式为 TEMP\MIXED、MOLEFLOW\MIXED\{comp} 等，max_depth=3 可完整覆盖。
        若改为 "Output"，则需要 max_depth=4 才能覆盖组分数据，且会遍历大量无关节点。
    strict_extraction:
        True（默认）：任意节点读取失败视为整批提取失败（fatal_error），
        ProcessCase 降级为 OBJECTIVE_ERROR，优化器不采纳该样本。
        False（调试用）：节点级失败记录到 notes，其余节点仍可用。
    """
    output_paths: list[str] = field(default_factory=list)
    objective_fns: list[ObjectiveFn] = field(default_factory=list)
    constraint_fns: list[ConstraintFn] = field(default_factory=list)
    timeout: float = 300.0
    reinit: bool = True
    verify_inputs: bool = True
    input_rtol: float = 1e-6
    check_status_paths: list[str] | None = None
    extract_blocks: list[str] | None = None
    extract_streams: list[str] | None = None
    block_max_depth: int = 3
    stream_max_depth: int = 3
    stream_output_subtree: str = "Output\\STR_MAIN"
    strict_extraction: bool = True


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_case(
    driver: AspenDriver,
    design_vars: dict[str, Any],
    config: RunCaseConfig,
    iteration: int = 0,
    tags: list[str] | None = None,
    run_id: str | None = None,
) -> ProcessCase:
    """
    执行一次完整的工况运行，返回 ProcessCase。

    即使仿真失败也会返回（status=SIM_FAILED），不会向上抛出异常。
    目标函数/约束函数的异常同样被捕获，记录在对应的 error 字段中。

    block/stream 提取失败时不静默降级：
      - 节点级失败：记录到对应 BlockResult/StreamResult.notes，
        并将失败路径汇总到 ProcessCase.notes。
      - 整批提取失败（TreeExporter 抛出异常）：ProcessCase.notes 记录原因，
        同时将所有目标函数标记为 error，使 status 降级为 OBJECTIVE_ERROR，
        防止优化器把缺数据样本当成有效样本。

    Parameters
    ----------
    driver:
        已连接并打开仿真文件的 AspenDriver 实例。
    design_vars:
        本次运行的设计变量 {Aspen 树路径: 值}，由优化器提供。
        同时作为 SimulationRunner 的 inputs 参数。
    config:
        运行配置，见 RunCaseConfig。
    iteration:
        优化迭代编号，由优化循环赋值，从 0 开始。
    tags:
        可选标签列表，供数据库分类检索（如 "initial_doe"、"exploitation"）。
    run_id:
        关联的底层运行 ID，由调用方赋值后传入数据库外键。

    Returns
    -------
    ProcessCase
    """
    runner   = SimulationRunner(driver)
    exporter = TreeExporter(driver)

    # 1. 运行仿真
    sim_result = runner.run_case(
        inputs=design_vars,
        output_paths=config.output_paths,
        timeout=config.timeout,
        reinit=config.reinit,
        verify_inputs=config.verify_inputs,
        input_rtol=config.input_rtol,
        check_status_paths=config.check_status_paths,
    )

    # 2 & 3. 提取 block/stream 数据（仅在仿真收敛时）
    blocks:  dict[str, BlockResult]  = {}
    streams: dict[str, StreamResult] = {}
    notes_parts: list[str] = []
    extraction_fatal = False

    if sim_result.success:
        block_result  = _extract_blocks(exporter, sim_result, design_vars, config, run_id)
        stream_result = _extract_streams(exporter, sim_result, config, run_id)

        blocks  = block_result.data   # type: ignore[assignment]
        streams = stream_result.data  # type: ignore[assignment]

        # 收集节点级失败信息
        if block_result.failed_nodes:
            parts = [f"  {k}: {v}" for k, v in block_result.failed_nodes.items()]
            notes_parts.append("Block 节点提取失败：\n" + "\n".join(parts))
        if stream_result.failed_nodes:
            parts = [f"  {k}: {v}" for k, v in stream_result.failed_nodes.items()]
            notes_parts.append("Stream 节点提取失败：\n" + "\n".join(parts))

        # 整批提取失败（strict_extraction=True 时节点级失败也会触发此路径）
        if block_result.fatal_error:
            notes_parts.append(f"Block 整批提取失败：{block_result.fatal_error}")
            extraction_fatal = True
        if stream_result.fatal_error:
            notes_parts.append(f"Stream 整批提取失败：{stream_result.fatal_error}")
            extraction_fatal = True

    # 4. 计算目标函数和约束
    objectives:  list[ObjectiveValue]  = []
    constraints: list[ConstraintValue] = []

    if sim_result.success and (config.objective_fns or config.constraint_fns):
        if extraction_fatal:
            # 整批提取失败：强制所有目标为 error，使 status 降级为 OBJECTIVE_ERROR
            fatal_msg = "block/stream 整批提取失败，目标函数数据不可信，拒绝计算。"
            objectives = [
                ObjectiveValue(
                    name=getattr(fn, "__name__", repr(fn)),
                    value=None,
                    error=fatal_msg,
                )
                for fn in config.objective_fns
            ]
            constraints = [
                ConstraintValue(
                    name=getattr(fn, "__name__", repr(fn)),
                    value=None,
                    error=fatal_msg,
                )
                for fn in config.constraint_fns
            ]
        else:
            _partial = ProcessCase(
                iteration=iteration,
                status=CaseStatus.from_run_status(sim_result.status),
                design_vars=design_vars,
                sim_result=sim_result,
                blocks=blocks,
                streams=streams,
                source_filepath=sim_result.source_filepath,
                run_id=run_id,
                tags=tags or [],
            )
            objectives  = _compute_objectives(config.objective_fns, _partial)
            constraints = _compute_constraints(config.constraint_fns, _partial)

    # 5. 组装最终 ProcessCase（含状态推断）
    case = process_case_from_sim_result(
        sim_result=sim_result,
        design_vars=design_vars,
        iteration=iteration,
        blocks=blocks,
        streams=streams,
        objectives=objectives,
        constraints=constraints,
        tags=tags,
        run_id=run_id,
    )

    if notes_parts:
        case.notes = "\n\n".join(notes_parts)

    return case


# ---------------------------------------------------------------------------
# Block 提取
# ---------------------------------------------------------------------------

def _extract_blocks(
    exporter: TreeExporter,
    sim_result: SimulationResult,
    design_vars: dict[str, Any],
    config: RunCaseConfig,
    run_id: str | None,
) -> _ExtractionResult:
    """提取所有（或指定）block 的 Output 子树，构建 BlockResult 字典。

    strict_extraction=True：任意节点失败抛异常，整批视为 fatal_error。
    strict_extraction=False：节点级失败记录到 failed_nodes 和 BlockResult.notes。
    """
    if config.extract_blocks is not None and len(config.extract_blocks) == 0:
        return _ExtractionResult(data={}, failed_nodes={}, fatal_error=None)

    try:
        block_records = exporter.export_block_outputs(
            result=sim_result,
            block_names=config.extract_blocks,
            max_depth=config.block_max_depth,
            strict=config.strict_extraction,
        )
    except Exception as exc:
        _log.warning("Block 整批提取失败：%s", exc)
        return _ExtractionResult(data={}, failed_nodes={}, fatal_error=str(exc))

    status_index = {bs.name: bs for bs in sim_result.block_statuses}
    results: dict[str, BlockResult] = {}
    all_failed_nodes: dict[str, str] = {}

    for block_name, records in block_records.items():
        bs           = status_index.get(block_name)
        record_type  = bs.record_type  if bs else ""
        status_flags = bs.status_flags if bs else []
        comp_status  = bs.comp_status  if bs else 0

        # 收集节点级失败，写入 BlockResult.notes
        failed = {r.rel_path: r.error for r in records if r.error is not None}
        if failed:
            for rel_path, err in failed.items():
                all_failed_nodes[f"{block_name}/{rel_path}"] = err

        inputs  = _inputs_for_block(block_name, design_vars)
        outputs = _records_to_block_outputs(records)

        br = block_result_from_runner(
            block_name=block_name,
            record_type=record_type,
            status_flags=status_flags,
            comp_status=comp_status,
            inputs=inputs,
            outputs=outputs,
            source_filepath=sim_result.source_filepath,
            run_id=run_id,
        )
        if failed:
            br.notes = "节点读取失败：" + "; ".join(f"{k}={v}" for k, v in failed.items())
        results[block_name] = br

    return _ExtractionResult(data=results, failed_nodes=all_failed_nodes, fatal_error=None)


def _inputs_for_block(
    block_name: str,
    design_vars: dict[str, Any],
) -> list[BlockInput]:
    """从 design_vars 中筛选属于指定 block 的输入参数。"""
    prefix = f"\\Data\\Blocks\\{block_name}\\".upper()
    return [
        BlockInput(
            path=path,
            name=path.split("\\")[-1],
            value=value,
        )
        for path, value in design_vars.items()
        if path.upper().startswith(prefix)
    ]


def _records_to_block_outputs(records: list[TreeValueRecord]) -> list[BlockOutput]:
    """将 TreeValueRecord 列表转换为 BlockOutput 列表，跳过读取失败的节点。"""
    return [
        BlockOutput(
            path=r.path,
            name=r.rel_path.replace("\\", "/"),
            value=r.value,
            unit=r.unit,
            value_type=r.value_type,
        )
        for r in records
        if r.error is None
    ]


# ---------------------------------------------------------------------------
# Stream 提取
# ---------------------------------------------------------------------------

def _extract_streams(
    exporter: TreeExporter,
    sim_result: SimulationResult,
    config: RunCaseConfig,
    run_id: str | None,
) -> _ExtractionResult:
    """提取所有（或指定）stream 的 Output 子树，构建 StreamResult 字典。

    strict_extraction=True：任意节点失败抛异常，整批视为 fatal_error。
    strict_extraction=False：节点级失败记录到 failed_nodes 和 StreamResult.notes。
    """
    if config.extract_streams is not None and len(config.extract_streams) == 0:
        return _ExtractionResult(data={}, failed_nodes={}, fatal_error=None)

    try:
        stream_records = exporter.export_stream_table(
            result=sim_result,
            stream_names=config.extract_streams,
            output_subtree=config.stream_output_subtree,
            max_depth=config.stream_max_depth,
            strict=config.strict_extraction,
        )
    except Exception as exc:
        _log.warning("Stream 整批提取失败：%s", exc)
        return _ExtractionResult(data={}, failed_nodes={}, fatal_error=str(exc))

    status_index = {bs.name: bs for bs in sim_result.block_statuses}
    results: dict[str, StreamResult] = {}
    all_failed_nodes: dict[str, str] = {}

    for stream_name, records in stream_records.items():
        bs           = status_index.get(stream_name)
        record_type  = bs.record_type  if bs else ""
        status_flags = bs.status_flags if bs else []
        comp_status  = bs.comp_status  if bs else 0

        failed = {r.rel_path: r.error for r in records if r.error is not None}
        if failed:
            for rel_path, err in failed.items():
                all_failed_nodes[f"{stream_name}/{rel_path}"] = err

        parsed = _parse_stream_records(records)

        sr = stream_result_from_runner(
            stream_name=stream_name,
            record_type=record_type,
            status_flags=status_flags,
            comp_status=comp_status,
            source_filepath=sim_result.source_filepath,
            run_id=run_id,
            **parsed,
        )
        if failed:
            sr.notes = "节点读取失败：" + "; ".join(f"{k}={v}" for k, v in failed.items())
        results[stream_name] = sr

    return _ExtractionResult(data=results, failed_nodes=all_failed_nodes, fatal_error=None)


def _parse_stream_records(records: list[TreeValueRecord]) -> dict[str, Any]:
    """
    将 TreeValueRecord 列表解析为 stream_result_from_runner 的关键字参数。

    rel_path 约定（output_subtree="Output\\STR_MAIN", max_depth=3，路径分隔符统一为 /）：
      TEMP/MIXED            → temp / temp_unit
      PRES/MIXED            → pres / pres_unit
      VFRAC/MIXED           → vfrac
      MOLEFLMX/MIXED        → total_mole_flow / total_mole_flow_unit
      MASSFLMX/MIXED        → total_mass_flow / total_mass_flow_unit
      MOLEFLOW/MIXED/{comp} → ComponentFlow.mole_flow
      MASSFLOW/MIXED/{comp} → ComponentFlow.mass_flow
      MOLEFRAC/MIXED/{comp} → ComponentFlow.mole_frac
      MASSFRAC/MIXED/{comp} → ComponentFlow.mass_frac

    注意：STR_MAIN 子树中 TEMP/PRES/VFRAC 均带 /MIXED 后缀，
    与直接从 Output 提取时的路径格式不同。

    失败节点（error is not None）已由调用方收集，此处仅处理成功记录。
    """
    rec_map: dict[str, TreeValueRecord] = {
        r.rel_path.replace("\\", "/"): r
        for r in records
        if r.error is None
    }

    def _val(key: str) -> Any:
        r = rec_map.get(key)
        return r.value if r else None

    def _unit(key: str) -> str:
        r = rec_map.get(key)
        return r.unit if r else ""

    comp_data: dict[str, ComponentFlow] = {}
    for rel_path, rec in rec_map.items():
        parts = rel_path.split("/")
        if len(parts) != 3 or parts[1] != "MIXED":
            continue
        flow_type, _, comp_name = parts
        cf = comp_data.setdefault(comp_name, ComponentFlow(component=comp_name))
        if flow_type == "MOLEFLOW":
            cf.mole_flow      = rec.value
            cf.mole_flow_unit = rec.unit
        elif flow_type == "MASSFLOW":
            cf.mass_flow      = rec.value
            cf.mass_flow_unit = rec.unit
        elif flow_type == "MOLEFRAC":
            cf.mole_frac = rec.value
        elif flow_type == "MASSFRAC":
            cf.mass_frac = rec.value

    return {
        "temp":                  _val("TEMP/MIXED"),
        "temp_unit":             _unit("TEMP/MIXED"),
        "pres":                  _val("PRES/MIXED"),
        "pres_unit":             _unit("PRES/MIXED"),
        "vfrac":                 _val("VFRAC/MIXED"),
        "total_mole_flow":       _val("MOLEFLMX/MIXED"),
        "total_mole_flow_unit":  _unit("MOLEFLMX/MIXED"),
        "total_mass_flow":       _val("MASSFLMX/MIXED"),
        "total_mass_flow_unit":  _unit("MASSFLMX/MIXED"),
        "total_vol_flow":        _val("VOLFLMX/MIXED"),
        "total_vol_flow_unit":   _unit("VOLFLMX/MIXED"),
        "components":            list(comp_data.values()),
    }


# ---------------------------------------------------------------------------
# 目标函数与约束函数计算
# ---------------------------------------------------------------------------

def _compute_objectives(
    fns: list[ObjectiveFn],
    case: ProcessCase,
) -> list[ObjectiveValue]:
    """
    依次调用目标函数，校验返回值类型和数值有效性。

    契约要求：函数必须返回 ObjectiveValue，value 必须是有限 float 或 None。
    - 返回非 ObjectiveValue：记录为 error，不抢救为有效样本（保留 unit/minimize/name 语义）。
    - value 为 NaN 或 Inf：记录为 error，防止污染 surrogate model 和经济分析。
    """
    results: list[ObjectiveValue] = []
    for fn in fns:
        name = getattr(fn, "__name__", repr(fn))
        try:
            obj = fn(case)
        except Exception as exc:
            _log.warning("目标函数 '%s' 计算失败：%s", name, exc)
            results.append(ObjectiveValue(name=name, value=None, error=str(exc)))
            continue

        if not isinstance(obj, ObjectiveValue):
            msg = f"目标函数 '{name}' 返回了 {type(obj).__name__}，期望 ObjectiveValue（含 unit/minimize/name）。"
            _log.warning(msg)
            results.append(ObjectiveValue(name=name, value=None, error=msg))
            continue

        if obj.value is not None:
            if not isinstance(obj.value, (int, float)):
                msg = f"目标函数 '{name}' 的 value 类型为 {type(obj.value).__name__}，期望 float。"
                _log.warning(msg)
                results.append(ObjectiveValue(name=obj.name, value=None, error=msg))
                continue
            if not math.isfinite(float(obj.value)):
                msg = f"目标函数 '{name}' 的 value={obj.value!r} 不是有限数（NaN/Inf），拒绝入库。"
                _log.warning(msg)
                results.append(ObjectiveValue(name=obj.name, value=None, error=msg))
                continue

        results.append(obj)
    return results


def _compute_constraints(
    fns: list[ConstraintFn],
    case: ProcessCase,
) -> list[ConstraintValue]:
    """
    依次调用约束函数，校验返回值类型和数值有效性。

    契约要求：函数必须返回 ConstraintValue，value 必须是有限 float 或 None。
    - 返回非 ConstraintValue：记录为 error。
    - value 为 NaN 或 Inf：记录为 error，防止约束判断逻辑出现不可控行为。
    """
    results: list[ConstraintValue] = []
    for fn in fns:
        name = getattr(fn, "__name__", repr(fn))
        try:
            con = fn(case)
        except Exception as exc:
            _log.warning("约束函数 '%s' 计算失败：%s", name, exc)
            results.append(ConstraintValue(name=name, value=None, error=str(exc)))
            continue

        if not isinstance(con, ConstraintValue):
            msg = f"约束函数 '{name}' 返回了 {type(con).__name__}，期望 ConstraintValue。"
            _log.warning(msg)
            results.append(ConstraintValue(name=name, value=None, error=msg))
            continue

        if con.value is not None:
            if not isinstance(con.value, (int, float)):
                msg = f"约束函数 '{name}' 的 value 类型为 {type(con.value).__name__}，期望 float。"
                _log.warning(msg)
                results.append(ConstraintValue(name=con.name, value=None, error=msg))
                continue
            if not math.isfinite(float(con.value)):
                msg = f"约束函数 '{name}' 的 value={con.value!r} 不是有限数（NaN/Inf），拒绝入库。"
                _log.warning(msg)
                results.append(ConstraintValue(name=con.name, value=None, error=msg))
                continue

        results.append(con)
    return results
