"""
diagnose_case.py — diagnose_case_tool 实现。

功能：给定 case_id，从 SimulationDB 读取完整 ProcessCase 记录，
      多维度分析失败原因（仿真引擎状态、输入写入校验、block/stream 收敛状态、
      输出读取失败、目标函数/约束计算结果），返回结构化诊断报告及改进建议。

不依赖 Aspen COM，可在任意环境中安全调用。
与 query_simulation_db_tool(mode='get_case') 的区别：
  get_case 只展示原始数据结构；diagnose_case 做语义层推断并给出诊断建议。

设计约束
--------
- 对任何非收敛状态，至少输出一条明确说明失败原因的建议，绝不返回"未发现明显异常"。
- include_failed_outputs=True 时从 ProcessCase.notes 及各 block/stream.notes
  中解析节点读取失败记录，并在建议中引用，体现参数的真实效果。
- simulation_valid=False 时 blocks/streams 未入库，诊断基于 sim_result 字段。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

_log = logging.getLogger(__name__)

# 非收敛的 RunStatus 值（对应 SimulationResult.status 字符串）
_NON_CONVERGENT_STATUSES = frozenset(
    {"errors", "no_results", "incompat", "inaccess", "status_unavailable"}
)

# 非收敛状态的人类可读描述（用于诊断建议文本）
_NON_CONV_DESC: dict[str, str] = {
    "errors":             "block/stream 存在错误标志（errors）",
    "no_results":         "仿真未产生结果（no_results），可能为输入翻译失败",
    "incompat":           "仿真结果与当前输入不兼容（incompat）",
    "inaccess":           "仿真结果不可访问（inaccess）",
    "status_unavailable": "无法读取仿真状态（status_unavailable），hap_constants 缺失",
}


# ---------------------------------------------------------------------------
# 数据库路径解析（与 query_node_db 风格一致，独立实现）
# ---------------------------------------------------------------------------

def _resolve_db_path(db_path: str) -> Path:
    """
    解析数据库路径，优先级：
    1. 绝对路径直接使用
    2. 相对于当前工作目录
    3. 相对于项目根目录（src/agents/tools/ 往上三层）
    """
    p = Path(db_path)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"数据库文件不存在：{p}")
        return p

    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return from_cwd.resolve()

    project_root = Path(__file__).parent.parent.parent.parent
    from_root = project_root / p
    if from_root.exists():
        return from_root.resolve()

    raise FileNotFoundError(
        f"数据库文件不存在：{db_path!r}\n"
        f"  已尝试：\n"
        f"    {from_cwd}\n"
        f"    {from_root}"
    )


# ---------------------------------------------------------------------------
# 格式化辅助函数
# ---------------------------------------------------------------------------

def _fmt_path_tail(path: str, tail: int = 4) -> str:
    """截取 Aspen 树路径末尾 N 段，用于紧凑显示。"""
    if not path:
        return "<empty>"
    parts = [p for p in path.replace("/", "\\").split("\\") if p]
    if len(parts) > tail:
        return "...\\" + "\\".join(parts[-tail:])
    return "\\".join(parts)


def _fmt_value(value: Any) -> str:
    """将节点值格式化为简短字符串。"""
    if value is None:
        return "None"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_input_verif_row(verif: dict, index: int) -> str:
    """格式化单条输入写入校验记录。"""
    path = _fmt_path_tail(verif.get("path", "?"))
    match = verif.get("match", False)
    requested = _fmt_value(verif.get("requested"))
    actual = _fmt_value(verif.get("actual"))
    note = verif.get("note", "")
    mark = "OK" if match else "NG"
    base = f"  [{index:3d}] [{mark}] {path}  req={requested}  actual={actual}"
    if not match and note:
        return base + f"\n        -> {note}"
    return base


def _fmt_block_status_row(bs: dict, index: int) -> str:
    """格式化单条 block/stream 状态记录。"""
    name = bs.get("name", "?")
    record_type = bs.get("record_type", "")
    status_flags = bs.get("status_flags", [])
    type_str = f" ({record_type})" if record_type else ""
    flags_str = " | ".join(status_flags) if status_flags else f"comp_status={bs.get('comp_status', '?')}"
    has_error = any(f in {"ERRORS", "NO_RESULTS", "INCOMPAT", "INACCESS"} for f in status_flags)
    mark = " [!]" if has_error else (" [W]" if "WARNINGS" in status_flags else "    ")
    return f"  [{index:3d}]{mark} {name}{type_str:<25} : {flags_str}"


def _fmt_failed_output_row(path: str, error: str, index: int) -> str:
    """格式化单条输出读取失败记录。"""
    short = _fmt_path_tail(path, tail=5)
    return f"  [{index:3d}] {short}\n        错误: {error}"


def _fmt_block_snapshot_row(block_name: str, block_data: dict) -> list[str]:
    """格式化单个 block 或 stream 的数值快照（仅展示关键字段）。"""
    lines: list[str] = []
    block_type = block_data.get("block_type", "") or block_data.get("stream_type", "")
    convergence = block_data.get("convergence", "")
    notes = block_data.get("notes", "")
    type_str = f" ({block_type})" if block_type else ""
    conv_str = f"  收敛={convergence}" if convergence else ""
    lines.append(f"  -- {block_name}{type_str}{conv_str} --")
    outputs = block_data.get("outputs", []) or []
    if outputs:
        for out in outputs[:8]:
            name = out.get("name", out.get("path", "?"))
            value = _fmt_value(out.get("value"))
            unit = out.get("unit", "")
            unit_str = f" [{unit}]" if unit else ""
            lines.append(f"    {name:<25} = {value}{unit_str}")
        if len(outputs) > 8:
            lines.append(f"    ... (共 {len(outputs)} 个输出字段)")
    else:
        lines.append("    (无输出记录)")
    if notes:
        for note_line in notes.split("\n")[:3]:
            lines.append(f"    [!] {note_line}")
    return lines


def _parse_failed_outputs_from_notes(
    case_notes: str,
    blocks_data: dict,
    streams_data: dict,
) -> list[tuple[str, str]]:
    """
    从 ProcessCase.notes 及各 block/stream.notes 中解析节点读取失败记录。

    返回 [(来源标签, 失败描述), ...] 列表，供 include_failed_outputs 区段展示
    和 _build_suggestions() 引用。

    识别以下关键词：节点读取失败、提取失败、failed、extraction error。
    空 notes 则跳过，不产生误报。
    """
    failures: list[tuple[str, str]] = []
    _FAILURE_KEYWORDS = ("节点读取失败", "提取失败", "failed", "extraction error")

    for line in (case_notes or "").split("\n"):
        line = line.strip()
        if line and any(kw in line.lower() for kw in _FAILURE_KEYWORDS):
            failures.append(("case.notes", line))

    for name, bdata in blocks_data.items():
        for line in (bdata.get("notes") or "").split("\n"):
            line = line.strip()
            if line and any(kw in line.lower() for kw in _FAILURE_KEYWORDS):
                failures.append((f"block:{name}", line))

    for name, sdata in streams_data.items():
        for line in (sdata.get("notes") or "").split("\n"):
            line = line.strip()
            if line and any(kw in line.lower() for kw in _FAILURE_KEYWORDS):
                failures.append((f"stream:{name}", line))

    return failures


# ---------------------------------------------------------------------------
# 诊断建议生成
# ---------------------------------------------------------------------------

def _build_suggestions(
    sim_result: dict | None,
    input_verifs: list[dict],
    block_statuses: list[dict],
    failed_output_hints: list[tuple[str, str]],
    objectives: list[dict],
    constraints: list[dict],
    case_status: str,
) -> list[str]:
    """
    根据诊断结果推断失败原因，生成 1~8 条具体建议。

    对任何非收敛状态（sr_status 不是 success/warnings），至少输出一条明确说明
    失败原因的建议，绝不返回"未发现明显异常"。
    """
    suggestions: list[str] = []
    _sim_failure_explained = False  # 是否已对仿真级失败给出过明确解释

    sr_status: str = ""
    sr_error: str = ""
    sr_warnings: list = []
    if sim_result:
        sr_status = sim_result.get("status", "") or ""
        sr_error = sim_result.get("error", "") or ""
        sr_warnings = sim_result.get("warnings", []) or []

    # 1. 输入写入失败
    if sr_status == "write_failed":
        suggestions.append(
            "输入写入失败（write_failed）：Aspen 拒绝接受写入，通常原因为"
            "（a）路径不存在或拼写错误；"
            "（b）参数在当前仿真状态下不可写（如锁定列）；"
            "（c）值超出 Aspen 内部范围。建议用 catalog 工具验证路径是否存在。"
        )
        _sim_failure_explained = True

    # 2. 输入校验不匹配
    mismatches = [v for v in input_verifs if not v.get("match", True)]
    if mismatches:
        mismatch_paths = [_fmt_path_tail(v.get("path", "?"), tail=3) for v in mismatches[:3]]
        suggestions.append(
            f"输入写入校验失败（{len(mismatches)} 项不匹配，如 {', '.join(mismatch_paths)}）："
            "写入后读回值与请求值不符，可能原因为单位不一致或 Aspen 做了内部限幅。"
            "建议开启 reinit=True，并检查对应参数的 Aspen 内置单位。"
        )

    # 3. 运行超时 / 一般 run_failed
    _timeout_kws = ("超时", "timeout", "Timeout", "timed out", "Timed out")
    is_timeout = any(kw in sr_error for kw in _timeout_kws)
    if is_timeout:
        suggestions.append(
            "仿真运行超时（run_failed）：Aspen Engine.Run2 在规定时间内未收敛。"
            "建议：（a）增大 timeout 配置值；"
            "（b）检查初值是否合理（极端参数值易导致长时间不收敛）；"
            "（c）对该设计点启用 reinit=True 从干净状态重跑。"
        )
        _sim_failure_explained = True
    elif sr_status == "run_failed":
        suggestions.append(
            f"仿真引擎运行失败（run_failed）：{sr_error or '未知错误'}。"
            "建议检查 Aspen Plus 安装状态和 COM 连接，确认仿真文件路径正确。"
        )
        _sim_failure_explained = True

    # 4. block/stream 有 ERRORS 或 NO_RESULTS
    error_blocks = [
        bs for bs in block_statuses
        if any(f in {"ERRORS", "NO_RESULTS"} for f in (bs.get("status_flags") or []))
    ]
    if error_blocks:
        names = [bs.get("name", "?") for bs in error_blocks[:4]]
        suggestions.append(
            f"以下 block/stream 收敛失败：{', '.join(names)}。"
            "建议：（a）检查各 block 的操作范围和初值设置；"
            "（b）查看 Aspen .his 诊断（见上方警告列表中含 'history diagnostics' 的行）；"
            "（c）对 RADFRAC 类设备，检查回流比/塔板数等关键参数是否超出可行域。"
        )

    # 5. INCOMPAT / INACCESS
    incompat_blocks = [
        bs for bs in block_statuses
        if any(f in {"INCOMPAT", "INACCESS"} for f in (bs.get("status_flags") or []))
    ]
    if incompat_blocks:
        names = [bs.get("name", "?") for bs in incompat_blocks[:3]]
        suggestions.append(
            f"以下 block/stream 结果不兼容或不可访问：{', '.join(names)}。"
            "通常原因为仿真后发生了二次变更（INCOMPAT），或 COM 访问权限问题（INACCESS）。"
            "建议重跑本工况并开启 reinit=True。"
        )

    # 6. 输出读取失败（从 notes 解析）
    if failed_output_hints:
        suggestions.append(
            f"检测到输出/节点读取失败记录（{len(failed_output_hints)} 条，详见上方【输出读取失败】区段）："
            "建议：（a）用 catalog 工具确认路径是否存在；"
            "（b）若为 manifest 模式，考虑重新构建 manifest 规则；"
            "（c）使用 query_node_db_tool(mode='path_search') 确认路径可读性。"
        )

    # 7. 目标函数计算失败
    failed_objs = [o for o in objectives if not o.get("available", True)]
    if failed_objs:
        names = [o.get("name", "?") for o in failed_objs]
        suggestions.append(
            f"目标函数计算失败（{', '.join(names)}）："
            "目标值所依赖的输出节点读取失败或返回 None。"
            "建议检查目标函数路径配置，或用 query_node_db_tool 确认路径历史读取情况。"
        )

    # 8. 约束违反（infeasible）
    violated = [c for c in constraints if c.get("satisfied") is False]
    if violated and case_status == "infeasible":
        names = [c.get("name", "?") for c in violated[:3]]
        vals = [_fmt_value(c.get("value")) for c in violated[:3]]
        details = "  ".join(f"{n}={v}" for n, v in zip(names, vals))
        suggestions.append(
            f"约束违反（infeasible）：{details}（<=0 为满足）。"
            "建议：（a）检查设计变量搜索空间是否与约束兼容；"
            "（b）不可行点仍有代理模型训练价值，可保留；"
            "（c）若违反量较大，考虑收紧搜索空间边界。"
        )

    # 9. 约束计算失败（constraint_error）
    failed_cons = [c for c in constraints if not c.get("available", True)]
    if failed_cons:
        names = [c.get("name", "?") for c in failed_cons]
        errors = [c.get("error") or "未知" for c in failed_cons]
        details = "；".join(f"{n}（{e}）" for n, e in zip(names, errors))
        suggestions.append(
            f"约束计算失败（{', '.join(names)}）：{details}。"
            "约束计算失败意味着纯度、能耗或经济分析等关键指标不可信。"
            "建议：（a）检查约束函数依赖的 Aspen 路径是否已正确读取；"
            "（b）确认单位换算正确且不存在 None/NaN 中间值；"
            "（c）若为 manifest 模式，检查对应语义字段的 abs_path 映射。"
        )

    # 9. 仅有警告
    if case_status == "warnings" and not suggestions:
        has_hist = any("history diagnostics" in str(w) for w in sr_warnings)
        msg = "仿真收敛但存在警告（warnings）：结果可用，建议降权或人工复核。"
        if has_hist:
            msg += "警告列表中含 Aspen history 诊断信息，请查看详情。"
        suggestions.append(msg)

    # 兜底A：sr_status 为非收敛状态但仿真失败分支均未命中
    if sr_status in _NON_CONVERGENT_STATUSES and not _sim_failure_explained:
        desc = _NON_CONV_DESC.get(sr_status, f"非收敛状态（{sr_status}）")
        error_hint = f"具体错误：{sr_error}。" if sr_error else "数据库中无详细错误信息。"
        has_hist = any("history diagnostics" in str(w).lower() for w in sr_warnings)
        hist_hint = "警告列表含 Aspen history 诊断信息，请查看上方警告列表。" if has_hist else ""
        suggestions.append(
            f"仿真未收敛（{desc}）：{error_hint}{hist_hint}"
            "由于仿真未收敛，block/stream 快照未入库，无法进行单元级详细诊断。"
            "建议：（a）查看 Aspen .his 文件获取具体错误；"
            "（b）调整初值或收敛参数后重跑；"
            "（c）使用 query_node_db_tool(mode='recurring_errors') 排查结构性失败路径。"
        )

    # 兜底B：最终 fallback，严格区分成功/失败/未知
    if not suggestions:
        _OK = frozenset({"success", "warnings"})
        if case_status in _OK:
            suggestions.append(
                "未发现明显异常。工况仿真收敛且目标函数/约束均正常，可作为有效优化样本。"
            )
        elif case_status == "pending":
            suggestions.append(
                "工况处于 pending 状态，尚未运行。请先执行 run_case_tool 运行此工况。"
            )
        else:
            suggestions.append(
                f"工况状态为 {case_status!r}，但未能从现有数据提取到具体失败原因。"
                "可能 sim_result 字段未完整入库，或失败发生在仿真启动之前。"
                "建议重跑此工况并观察实时日志。"
            )

    return suggestions


# ---------------------------------------------------------------------------
# 核心报告格式化
# ---------------------------------------------------------------------------

def _fmt_diagnose_report(
    db_path: str,
    row: dict,
    include_input_verification: bool,
    include_block_details: bool,
    include_failed_outputs: bool,
    include_blocks_snapshot: bool,
    max_items: int,
) -> str:
    """将完整 ProcessCase 记录格式化为诊断报告。"""
    lines: list[str] = ["=== diagnose_case 工况诊断报告 ===", ""]

    # ---- 头部元信息 ----
    lines.append(f"数据库   : {db_path}")
    lines.append(f"case_id  : {row.get('case_id', '?')}")
    lines.append(f"iteration: {row.get('iteration', '?')}")
    lines.append(f"状态     : {row.get('status', '?')}")
    run_time = row.get("run_time", 0.0) or 0.0
    lines.append(f"运行耗时 : {float(run_time):.1f} s")
    source_filepath = row.get("source_filepath") or "—"
    lines.append(f"来源文件 : {source_filepath}")
    tags = row.get("tags") or []
    if tags:
        lines.append(f"标签     : {', '.join(tags)}")
    lines.append("")

    # ---- 整体评估 ----
    success = row.get("success", False)
    simulation_valid = row.get("simulation_valid", False)
    feasible = row.get("feasible")
    feasible_str = "是" if feasible is True else ("否" if feasible is False else "无约束")
    lines.append("【整体评估】")
    lines.append(f"  成功（可采纳）: {'是' if success else '否'}")
    lines.append(f"  仿真收敛      : {'是' if simulation_valid else '否'}")
    lines.append(f"  可行（约束）  : {feasible_str}")
    lines.append("")

    # ---- 仿真引擎诊断 ----
    sim_result: dict | None = row.get("sim_result")
    if sim_result:
        lines.append("【仿真引擎诊断】")
        sr_status = sim_result.get("status", "?")
        sr_error = sim_result.get("error") or ""
        sr_warnings: list = sim_result.get("warnings") or []
        lines.append(f"  引擎状态 : {sr_status}")
        if sr_error:
            lines.append(f"  错误信息 : {sr_error}")
        if sr_warnings:
            lines.append(f"  警告列表 : (共 {len(sr_warnings)} 条)")
            for i, w in enumerate(sr_warnings[:max_items], 1):
                w_str = str(w)
                if len(w_str) > 200:
                    w_str = w_str[:200] + "..."
                lines.append(f"    [{i:3d}] {w_str}")
            if len(sr_warnings) > max_items:
                lines.append(f"    ... 共 {len(sr_warnings)} 条，已截断显示")
        lines.append("")

    # ---- 设计变量 ----
    design_vars: dict = row.get("design_vars") or {}
    if design_vars:
        lines.append("【设计变量（输入点）】")
        for path, val in design_vars.items():
            short = _fmt_path_tail(path, tail=3)
            try:
                val_str = f"{float(val):.6g}"
            except (TypeError, ValueError):
                val_str = str(val)
            lines.append(f"  {short:<35} = {val_str}")
        lines.append("")

    # ---- 输入写入校验 ----
    # 注：SimulationDB 仅存 sim_result 摘要，不含 input_verifications 逐条记录
    input_verifs: list[dict] = []
    # blocks_data / streams_data 在此处预先提取，后续多处复用
    blocks_data: dict = row.get("blocks") or {}
    streams_data: dict = row.get("streams") or {}
    if include_input_verification:
        lines.append("【输入写入校验】")
        lines.append(
            "  (注：SimulationDB 仅存储 sim_result 摘要，不含逐条校验记录。"
            "如需校验详情，请在工况运行时查看实时日志。)"
        )
        lines.append("")

    # ---- block/stream 收敛状态 ----
    if include_block_details and (blocks_data or streams_data):
        total = len(blocks_data) + len(streams_data)
        lines.append(f"【block/stream 收敛状态（共 {total} 个）】")
        n_ok = n_warn = n_err = 0
        problem_items: list[tuple[str, str, str]] = []
        for name, bdata in blocks_data.items():
            conv = str(bdata.get("convergence", "")).lower()
            btype = bdata.get("block_type", "")
            if "error" in conv or "fail" in conv or conv in {"errors", "no_results"}:
                n_err += 1; problem_items.append((name, btype, conv))
            elif "warning" in conv:
                n_warn += 1; problem_items.append((name, btype, conv))
            else:
                n_ok += 1
        for name, sdata in streams_data.items():
            conv = str(sdata.get("convergence", "")).lower()
            stype = sdata.get("stream_type", "")
            if "error" in conv or "fail" in conv:
                n_err += 1; problem_items.append((name, stype, conv))
            elif "warning" in conv:
                n_warn += 1; problem_items.append((name, stype, conv))
            else:
                n_ok += 1
        lines.append(f"  收敛正常: {n_ok}  有警告: {n_warn}  有错误: {n_err}")
        lines.append("")
        if problem_items:
            lines.append("  【异常项（错误/警告优先展示）】")
            for name, btype, conv in problem_items[:max_items]:
                type_str = f" ({btype})" if btype else ""
                lines.append(f"    [!] {name}{type_str}: {conv}")
            if len(problem_items) > max_items:
                lines.append(f"    ... 共 {len(problem_items)} 项异常，已截断")
            lines.append("")
        else:
            lines.append("  所有 block/stream 状态正常或未记录收敛标志。")
            lines.append("")
    elif include_block_details:
        lines.append("【block/stream 收敛状态】")
        lines.append(
            "  (仿真未收敛（simulation_valid=False），block/stream 快照未入库，"
            "无法展示单元级收敛状态。)"
        )
        lines.append("")

    # ---- 输出读取失败（从 notes 解析）----
    extracted_failures = _parse_failed_outputs_from_notes(
        case_notes=str(row.get("notes") or ""),
        blocks_data=blocks_data,
        streams_data=streams_data,
    )
    if include_failed_outputs:
        lines.append("【输出读取失败（从运行注记解析）】")
        if extracted_failures:
            lines.append(f"  共检测到 {len(extracted_failures)} 条节点读取失败记录：")
            for i, (src, desc) in enumerate(extracted_failures[:max_items], 1):
                lines.append(f"  [{i:3d}] [{src}] {desc}")
            if len(extracted_failures) > max_items:
                lines.append(f"  ... 共 {len(extracted_failures)} 条，已截断显示")
        else:
            lines.append("  (运行注记中未检测到节点读取失败记录)")
        lines.append("")

    # ---- 目标函数 ----
    objectives: list[dict] = row.get("objectives") or []
    lines.append("【目标函数】")
    if not objectives:
        lines.append("  (无目标函数记录)")
    else:
        for obj in objectives:
            name = obj.get("name", "?")
            available = obj.get("available", False)
            unit = obj.get("unit", "")
            minimize = obj.get("minimize", True)
            direction = "最小化" if minimize else "最大化"
            if available:
                val = _fmt_value(obj.get("value"))
                unit_str = f" {unit}" if unit else ""
                lines.append(f"  {name:<20} = {val}{unit_str}  [{direction}]")
            else:
                err = obj.get("error") or "不可用"
                lines.append(f"  {name:<20} = [不可用]  错误：{err}")
    lines.append("")

    # ---- 约束条件 ----
    constraints: list[dict] = row.get("constraints") or []
    if constraints:
        lines.append("【约束条件】")
        for con in constraints:
            name = con.get("name", "?")
            available = con.get("available", False)
            satisfied = con.get("satisfied")
            if available:
                val = _fmt_value(con.get("value"))
                sat_str = "满足" if satisfied is True else ("违反" if satisfied is False else "未知")
                lines.append(f"  {name:<25} = {val}  [{sat_str}]  (<=0 为满足)")
            else:
                err = con.get("error") or "不可用"
                lines.append(f"  {name:<25} = [不可用]  错误：{err}")
        lines.append("")

    # ---- 运行注记 ----
    notes = row.get("notes", "")
    if notes and notes.strip():
        lines.append("【运行注记（block/stream 提取失败等）】")
        for note_line in notes.strip().split("\n")[:15]:
            lines.append(f"  {note_line}")
        lines.append("")

    # ---- blocks/streams 数值快照（同时遍历 blocks 和 streams）----
    if include_blocks_snapshot and (blocks_data or streams_data):
        lines.append("【blocks/streams 数值快照】")
        cap = max_items
        items_shown = 0
        for name, bdata in list(blocks_data.items()):
            if items_shown >= cap:
                break
            for snapshot_line in _fmt_block_snapshot_row(name, bdata):
                lines.append(snapshot_line)
            items_shown += 1
        for name, sdata in list(streams_data.items()):
            if items_shown >= cap:
                break
            for snapshot_line in _fmt_block_snapshot_row(name, sdata):
                lines.append(snapshot_line)
            items_shown += 1
        total_items = len(blocks_data) + len(streams_data)
        if total_items > cap:
            lines.append(f"  ... 共 {total_items} 个 block/stream，已截断至 {cap} 个")
        lines.append("")

    # ---- 诊断建议 ----
    approx_block_statuses: list[dict] = []
    for name, bdata in blocks_data.items():
        conv = str(bdata.get("convergence", "")).upper()
        approx_block_statuses.append({
            "name": name,
            "record_type": bdata.get("block_type", ""),
            "comp_status": bdata.get("comp_status", 0),
            "status_flags": [conv] if conv else [],
        })
    for name, sdata in streams_data.items():
        conv = str(sdata.get("convergence", "")).upper()
        approx_block_statuses.append({
            "name": name,
            "record_type": sdata.get("stream_type", ""),
            "comp_status": sdata.get("comp_status", 0),
            "status_flags": [conv] if conv else [],
        })

    suggestions = _build_suggestions(
        sim_result=sim_result,
        input_verifs=input_verifs,
        block_statuses=approx_block_statuses,
        # 仅当 include_failed_outputs=True 时将失败列表传入，
        # 否则传空列表——避免建议文本中引用一个不可见的区段
        failed_output_hints=extracted_failures if include_failed_outputs else [],
        objectives=objectives,
        constraints=constraints,
        case_status=str(row.get("status", "")),
    )
    lines.append("【诊断建议】")
    for i, suggestion in enumerate(suggestions, 1):
        lines.append(f"  {i}. {suggestion}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------

def _impl_diagnose_case(
    db_path: str,
    case_id: str,
    include_input_verification: bool,
    include_block_details: bool,
    include_failed_outputs: bool,
    include_blocks_snapshot: bool,
    max_items: int,
) -> str:
    """diagnose_case_tool 的核心实现，出错时返回 '错误：' 字符串。"""
    if not case_id or not case_id.strip():
        return "错误：case_id 不能为空，请提供目标工况的 UUID。"

    if max_items <= 0 or max_items > 200:
        return f"错误：max_items={max_items} 不合法，必须在 1 到 200 之间。"

    try:
        resolved_path = _resolve_db_path(db_path)
    except FileNotFoundError as exc:
        return f"错误：{exc}"

    try:
        from src.database.simulation_db import SimulationDB
    except ImportError as exc:
        return f"错误：无法导入 SimulationDB — {exc}"

    try:
        db = SimulationDB(resolved_path)
    except Exception as exc:
        return f"错误：无法打开数据库 [{type(exc).__name__}] — {exc}"

    db_path_str = str(resolved_path)
    try:
        row = db.get_case(case_id.strip())
        if row is None:
            total = db.count()
            return (
                f"错误：case_id={case_id!r} 不存在于数据库中。\n"
                f"数据库共 {total} 条记录，可用 query_simulation_db_tool 列出已有工况。"
            )
    except Exception as exc:
        return f"错误：get_case 查询失败 [{type(exc).__name__}] — {exc}"
    finally:
        db.close()

    try:
        report = _fmt_diagnose_report(
            db_path=db_path_str,
            row=row,
            include_input_verification=include_input_verification,
            include_block_details=include_block_details,
            include_failed_outputs=include_failed_outputs,
            include_blocks_snapshot=include_blocks_snapshot,
            max_items=max_items,
        )
    except Exception as exc:
        _log.exception("格式化 diagnose_case 报告时出现意外错误")
        return f"错误：格式化报告时出现意外错误 — {exc}"

    _log.info(
        "diagnose_case_tool: 完成诊断 case_id=%s status=%s",
        case_id.strip(), row.get("status", "?"),
    )
    return report


# ---------------------------------------------------------------------------
# LangChain @tool 定义
# ---------------------------------------------------------------------------

@tool
def diagnose_case_tool(
    db_path: str,
    case_id: str,
    include_input_verification: bool = True,
    include_block_details: bool = True,
    include_failed_outputs: bool = True,
    include_blocks_snapshot: bool = False,
    max_items: int = 30,
) -> str:
    """诊断指定工况的失败原因，返回多维度分析报告和改进建议。

    从 SimulationDB 读取完整 ProcessCase 记录，分析：仿真引擎状态、
    block/stream 收敛情况、目标函数/约束计算结果，并自动推断失败原因
    给出具体的改进建议。不依赖 Aspen COM，可在任意环境中安全调用。

    对任何非收敛状态（sim_result.status 不是 success/warnings），报告中
    至少包含一条明确说明失败原因的建议，不会给出"未发现明显异常"的误导结论。

    与 query_simulation_db_tool(mode='get_case') 的区别：
      get_case 只展示原始数据；diagnose_case 做语义推断，自动归因并给出建议。

    Args:
        db_path: SimulationDB SQLite 文件路径（相对于项目根目录或绝对路径）。
            典型路径：``cases/demo_case/output/simulation.db``。
        case_id: 要诊断的工况 UUID（必填）。
            可通过 query_simulation_db_tool 查询获取。
        include_input_verification: 是否展示输入写入校验说明（默认 True）。
            注意：SimulationDB 只存储 sim_result 摘要，不含逐条校验记录；
            此选项控制是否输出相关说明文字。
        include_block_details: 是否展示 block/stream 收敛状态列表（默认 True）。
            True 时展示各 block/stream 的收敛标志，异常项优先显示。
            simulation_valid=False 时 blocks 未入库，会输出说明文字。
        include_failed_outputs: 是否展示输出读取失败列表（默认 True）。
            True 时从 ProcessCase.notes 及各 block/stream.notes 中解析并展示
            节点读取失败记录。False 时跳过此区段。
        include_blocks_snapshot: 是否展示 blocks/streams 数值快照（默认 False）。
            True 时展示各 block 和 stream 的输出字段值；内容较长，建议按需开启。
        max_items: 各列表最多显示条数（默认 30，最大 200）。

    Returns:
        格式化的诊断报告文本，包含整体评估、各维度诊断和改进建议。
        出错时返回以 "错误：" 开头的描述字符串。
    """
    return _impl_diagnose_case(
        db_path=db_path,
        case_id=case_id,
        include_input_verification=include_input_verification,
        include_block_details=include_block_details,
        include_failed_outputs=include_failed_outputs,
        include_blocks_snapshot=include_blocks_snapshot,
        max_items=max_items,
    )