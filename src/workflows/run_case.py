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
from pathlib import Path
from typing import Any, Callable, NamedTuple

from ..aspen_driver.driver import AspenDriver
from ..aspen_driver.exporter import TreeExporter, TreeValueRecord
from ..aspen_driver.runner import SimulationRunner
from ..models.block import BlockInput, BlockOutput, BlockResult, block_result_from_runner
from ..models.node_catalog import SemanticBlock, SemanticField
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
    # ------------------------------------------------------------------ #
    # manifest runtime 模式配置
    # ------------------------------------------------------------------ #
    # extraction_mode: "full"（默认，递归 TreeExporter）/ "manifest"（直接路径读取）
    extraction_mode: str = "full"
    # manifest runtime 模式所需的 NodeDB 路径
    catalog_db_path: str | Path | None = None
    # manifest_id: "auto" 表示自动查找最新 manifest；也可指定具体 ID
    manifest_id: str = "auto"
    # 语义规则目录，默认 configs/aspen_semantics
    semantic_rules_dir: str | Path = "configs/aspen_semantics"
    # 是否在 manifest 不存在时自动执行 catalog scan + build manifest
    build_manifest_if_missing: bool = True
    # 是否将 manifest 读取结果写入 NodeDB node_values
    write_node_values: bool = True
    # manifest invalid 时是否阻止优化（True=阻止，False=降级为 full 模式）
    strict_manifest: bool = True


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

    if not sim_result.success and sim_result.error:
        _log.warning("仿真失败 [iter=%d, status=%s]：%s", iteration, sim_result.status.value, sim_result.error)

    # 2 & 3. 提取 block/stream 数据（仅在仿真收敛时）
    blocks:  dict[str, BlockResult]  = {}
    streams: dict[str, StreamResult] = {}
    semantic_blocks: dict[str, SemanticBlock] = {}
    notes_parts: list[str] = []
    extraction_fatal = False

    if sim_result.success:
        if config.extraction_mode == "manifest":
            # manifest runtime 模式：直接路径读取，不递归 Elements
            semantic_blocks, manifest_notes, extraction_fatal = _extract_by_manifest(
                exporter, sim_result, config, iteration, run_id
            )
            notes_parts.extend(manifest_notes)
        else:
            # full/debug 模式：递归 TreeExporter（原有逻辑）
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
                semantic_blocks=semantic_blocks,
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
        semantic_blocks=semantic_blocks,
    )

    if notes_parts:
        case.notes = "\n\n".join(notes_parts)

    return case


# ---------------------------------------------------------------------------
# Manifest runtime 提取
# ---------------------------------------------------------------------------

def _extract_by_manifest(
    exporter: TreeExporter,
    sim_result: SimulationResult,
    config: RunCaseConfig,
    iteration: int,
    run_id: str | None,
) -> tuple[dict[str, SemanticBlock], list[str], bool]:
    """
    manifest runtime 模式：从 NodeDB 加载 manifest，直接路径读取节点值，
    构建 SemanticBlock 字典。

    Returns
    -------
    (semantic_blocks, notes_parts, extraction_fatal)
    """
    notes: list[str] = []
    fatal = False

    # 延迟导入，避免循环依赖
    try:
        from ..database.node_db import NodeDB
        from ..aspen_driver.catalog import CatalogScanner, _compute_file_hash
        from ..aspen_driver.manifest import ManifestBuilder
        from ..models.read_manifest import ReadManifest
    except ImportError as exc:
        msg = f"manifest runtime 模式依赖导入失败：{exc}"
        _log.error(msg)
        notes.append(msg)
        return {}, notes, True

    db_path = config.catalog_db_path
    if not db_path:
        msg = "extraction_mode='manifest' 但未配置 catalog_db_path，降级为 full 模式"
        _log.warning(msg)
        notes.append(msg)
        return {}, notes, False

    try:
        node_db = NodeDB(db_path)
    except Exception as exc:
        msg = f"打开 NodeDB 失败（{db_path}）：{exc}"
        _log.error(msg)
        notes.append(msg)
        return {}, notes, config.strict_manifest

    try:
        manifest = _resolve_manifest(
            node_db=node_db,
            config=config,
            driver=exporter._driver,
            sim_result=sim_result,
        )
    except Exception as exc:
        msg = f"manifest 解析失败：{exc}"
        _log.error(msg)
        notes.append(msg)
        node_db.close()
        return {}, notes, config.strict_manifest

    if manifest is None:
        msg = "未找到有效 manifest，降级为 full 模式"
        _log.warning(msg)
        notes.append(msg)
        node_db.close()
        return {}, notes, False

    if not manifest.is_valid:
        msg = f"manifest '{manifest.manifest_id}' 无效：{manifest.error}"
        _log.warning(msg)
        notes.append(msg)
        if config.strict_manifest:
            node_db.close()
            return {}, notes, True

    # 按 manifest 直接路径读取
    try:
        raw_by_source = exporter.export_values_by_manifest(
            sim_result,
            manifest,
            strict_required=config.strict_manifest,
            allow_failed=False,
        )
    except Exception as exc:
        msg = f"manifest 直接路径读取失败：{exc}"
        _log.error(msg)
        notes.append(msg)
        node_db.close()
        return {}, notes, config.strict_manifest

    # 将读取结果写入 NodeDB node_values
    if config.write_node_values and run_id:
        try:
            from ..aspen_driver.exporter import TreeValueRecord as _TVR
            for source_name, records in raw_by_source.items():
                node_db.save_node_values(
                    case_id=run_id,
                    source=f"manifest:{source_name}",
                    records=records,
                )
        except Exception as exc:
            _log.warning("写入 node_values 失败：%s", exc)

    # 构建 SemanticBlock 字典
    semantic_blocks = _build_semantic_blocks(manifest, raw_by_source)

    node_db.close()
    return semantic_blocks, notes, False


def _resolve_manifest(
    node_db: Any,
    config: RunCaseConfig,
    driver: Any,
    sim_result: Any,
) -> Any:
    """
    解析 manifest：按 manifest_id 查找，或自动构建。

    返回 ReadManifest 实例，或 None（无法解析时）。
    """
    from ..aspen_driver.catalog import CatalogScanner, _compute_file_hash
    from ..aspen_driver.manifest import ManifestBuilder

    mid = config.manifest_id

    if mid and mid != "auto":
        # 指定 manifest_id：直接从 DB 加载
        meta = node_db.get_manifest(mid)
        if meta is None:
            raise ValueError(f"manifest_id='{mid}' 在 NodeDB 中不存在")
        return _load_manifest_from_db(node_db, mid)

    # auto 模式：查找最新 catalog，再查找最新 manifest
    file_path = str(driver.filepath) if driver.filepath else ""
    file_hash = _compute_file_hash(file_path) if file_path else ""

    catalog_meta = node_db.get_latest_catalog_scan(file_hash) if file_hash else None

    if catalog_meta is None:
        if not config.build_manifest_if_missing:
            _log.warning("未找到 catalog，build_manifest_if_missing=False，跳过 manifest 构建")
            return None
        # 执行 catalog scan
        _log.info("未找到 catalog，开始自动扫描（file_hash=%s）", file_hash[:8] if file_hash else "N/A")
        scanner = CatalogScanner(driver, node_db)
        scan = scanner.scan(aspen_file_path=file_path, max_depth=6, strict=False)
        catalog_id = scan.catalog_id
    else:
        catalog_id = catalog_meta["catalog_id"]

    # 查找最新 manifest
    manifest_meta = node_db.get_latest_manifest(catalog_id)
    if manifest_meta is None:
        if not config.build_manifest_if_missing:
            _log.warning("未找到 manifest，build_manifest_if_missing=False，跳过")
            return None
        # 构建 manifest
        _log.info("未找到 manifest，开始自动构建（catalog_id=%s）", catalog_id)
        builder = ManifestBuilder(node_db, rules_dir=config.semantic_rules_dir)
        # 从 objective_fns 名称推断 objective_names
        obj_names = [getattr(fn, "__name__", "") for fn in (config.objective_fns or [])]
        obj_names = [n for n in obj_names if n]
        manifest = builder.build(
            catalog_id=catalog_id,
            objective_names=obj_names,
            extra_paths=config.output_paths or [],
        )
        return manifest

    return _load_manifest_from_db(node_db, manifest_meta["manifest_id"])


def _load_manifest_from_db(node_db: Any, manifest_id: str) -> Any:
    """从 NodeDB 加载 ReadManifest 对象。"""
    from ..models.read_manifest import ReadManifest, ReadManifestItem

    meta = node_db.get_manifest(manifest_id)
    if meta is None:
        return None
    items_raw = node_db.get_manifest_items(manifest_id)
    items = [
        ReadManifestItem(
            manifest_id=manifest_id,
            source_type=r["source_type"],
            source_name=r["source_name"],
            equipment_type=r["equipment_type"],
            semantic_field=r["semantic_field"],
            abs_path=r["abs_path"],
            rel_path=r["rel_path"],
            unit_string=r["unit_string"],
            value_type=r["value_type"],
            required=bool(r["required"]),
            confidence=r["confidence"],
            rule_id=r["rule_id"],
            error=r["error"],
        )
        for r in items_raw
    ]
    return ReadManifest(
        manifest_id=manifest_id,
        catalog_id=meta["catalog_id"],
        objective_names=meta["objective_names"],
        items=items,
        is_valid=meta["is_valid"],
        error=meta["error"],
        created_at=meta["created_at"],
    )


def _build_semantic_blocks(
    manifest: Any,
    raw_by_source: dict[str, list[Any]],
) -> dict[str, SemanticBlock]:
    """
    将 manifest + 读取结果转换为 {block_name: SemanticBlock} 字典。
    """
    # 构建 manifest item 索引：{(source_name, semantic_field): item}
    item_index: dict[tuple[str, str], Any] = {}
    for item in manifest.items:
        item_index[(item.source_name, item.semantic_field)] = item

    # 按 source_name 分组
    source_names: set[str] = {item.source_name for item in manifest.items if item.source_name}

    result: dict[str, SemanticBlock] = {}
    for source_name in source_names:
        records = raw_by_source.get(source_name, [])
        # {semantic_field: TreeValueRecord}
        rec_index = {r.rel_path: r for r in records}

        fields: dict[str, SemanticField] = {}
        missing_required: list[str] = []

        for item in manifest.items:
            if item.source_name != source_name:
                continue
            rec = rec_index.get(item.semantic_field)
            if rec is None:
                # 未读取到（可能是 error item 无路径）
                available = False
                value = None
                unit = item.unit_string
                vtype = item.value_type
                error = item.error or f"未读取到字段 '{item.semantic_field}'"
            else:
                available = rec.error is None and rec.value is not None
                value = rec.value
                unit = rec.unit
                vtype = rec.value_type
                error = rec.error or ""

            sf = SemanticField(
                field_name=item.semantic_field,
                abs_path=item.abs_path,
                value=value,
                unit=unit,
                value_type=vtype,
                available=available,
                error=error,
                required=item.required,
                rule_id=item.rule_id,
            )
            fields[item.semantic_field] = sf

            if item.required and not available:
                missing_required.append(item.semantic_field)

        # 获取 block_type（从 manifest items 中取）
        block_type = ""
        for item in manifest.items:
            if item.source_name == source_name and item.equipment_type:
                block_type = item.equipment_type
                break

        sb = SemanticBlock(
            block_name=source_name,
            block_type=block_type,
            fields=fields,
            is_complete=len(missing_required) == 0,
            missing_required=missing_required,
            manifest_id=manifest.manifest_id,
        )
        result[source_name] = sb

    return result


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
