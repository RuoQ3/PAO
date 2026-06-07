"""
discover_tunables.py — 变量发现工具（阶段 A3）。

职责
----
给定一个 Aspen .bkp 文件路径，打开文件扫描 Aspen 树（不运行仿真），
结合语义规则识别可调设计变量（TunableVariable）和可读输出目标节点（ReadableTarget），
产出 TunableReport，供 config_builder 生成优化配置草案。

核心设计约束
------------
- 唯一会打开 Aspen COM 的新增代码路径，只读扫描，绝不调用 Engine.Run2 / run_case
- 上层模块（config_builder、onboarding_agent、graph.py）导入时不得引入 aspen_driver
  本文件在模块级不导入 aspen_driver，所有 COM 相关代码在函数内部懒加载
- 扫描失败的节点记入 scan_warnings，不向上抛出异常
- 整个实现分为 5 个内部函数 + 1 个 @tool 包装

函数调用链
----------
discover_tunables_tool
  └── discover_tunables_impl(aspen_file_path, node_db_path, rules_dir, max_depth)
        ├── _scan_aspen_file(aspen_path, node_db_path, max_depth)  → CatalogScan + entries
        ├── _build_tunable_variables(entries, rules)               → list[TunableVariable]
        ├── _build_readable_targets(entries, rules)                → list[ReadableTarget]
        └── _compute_semantic_coverage(tunable_vars, targets, entries)  → float

语义规则命中逻辑
----------------
- Input 节点 + 规则命中 → TunableVariable（confidence=high/medium）
- Input 节点 + 无规则   → TunableVariable（confidence=low，边界 None）
- Output 节点 + required_for 含 TAC/EMISSIONS → ReadableTarget(candidate_use=objective)
- Output 节点 + 路径含 MASSFRAC/MOLE_FRAC/MOLEFRAC/MASSFRAC → candidate_use=constraint
- 两者都满足 → candidate_use=both
- 扫描失败节点（sample_error 非空）不进入任何推荐列表
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from src.models.tunable import (
    ReadableTarget,
    TunableReport,
    TunableVariable,
)

_log = logging.getLogger(__name__)

# 路径段中标示"Input"或"Output"的正则（大小写不敏感）
_RE_INPUT  = re.compile(r"[\\\/]Input[\\\/]", re.IGNORECASE)
_RE_OUTPUT = re.compile(r"[\\\/]Output[\\\/]", re.IGNORECASE)

# 路径末尾段中含纯度相关关键词的模式
_RE_PURITY = re.compile(r"(MASSFRAC|MOLE_FRAC|MOLEFRAC|MASSFRAC|MOLFRAC)", re.IGNORECASE)

# required_for 中触发"objective"候选的目标名称集合
_OBJECTIVE_FOR = frozenset({"TAC", "EMISSIONS", "ENERGY"})

# 语义规则的"高置信"来源：candidates 里有明确经验边界
_HIGH_CONFIDENCE_PATTERNS = frozenset({
    "B:F", "BASIS_RR", "FEED_STAGE",
    "PRES", "TEMP", "VFRAC", "EFF",
    "FRAC", "CONV",
})


# ---------------------------------------------------------------------------
# A3-1  打开 Aspen 文件并扫描 catalog
# ---------------------------------------------------------------------------

def _scan_aspen_file(
    aspen_path: str,
    node_db_path: str,
    max_depth: int = 6,
) -> tuple[Any, list[dict[str, Any]], list[str]]:
    """
    打开 Aspen 文件、执行 CatalogScanner.scan()，返回 (CatalogScan, entries, warnings)。

    只读扫描，不调用 driver.run() / Engine.Run2 / run_case。
    打开后必须关闭 Aspen：通过 try/finally 确保 driver.close()。

    Parameters
    ----------
    aspen_path:
        Aspen 仿真文件的绝对/相对路径（.bkp / .apw）。
    node_db_path:
        NodeDB SQLite 文件路径（不存在则自动创建）。
    max_depth:
        节点树扫描最大深度，默认 6。

    Returns
    -------
    (CatalogScan, entries, warnings)
        - CatalogScan 元数据对象
        - entries: list[dict]（get_catalog_entries 返回的字典列表）
        - warnings: 扫描过程中产生的警告文本列表
    """
    warnings: list[str] = []
    driver = None

    try:
        # 懒加载 COM 相关模块，确保导入时不触碰 aspen_driver
        from src.aspen_driver.driver import AspenDriver
        from src.aspen_driver.catalog import CatalogScanner
        from src.database.node_db import NodeDB
    except ImportError as exc:
        warnings.append(f"无法导入 aspen_driver 模块：{exc}")
        # 返回一个空的 mock scan 供上层处理
        return _make_empty_scan(aspen_path), [], warnings

    try:
        driver = AspenDriver(visible=False, suppress_dialogs=True)
        driver.open(aspen_path)

        with NodeDB(node_db_path) as node_db:
            scanner = CatalogScanner(driver, node_db)
            scan = scanner.scan(
                aspen_file_path=aspen_path,
                max_depth=max_depth,
                strict=False,          # 节点失败不中断扫描
                include_streams=True,
            )

            # 收集 CatalogScan.notes 中的失败信息
            if scan.notes:
                warnings.append(f"Catalog 扫描注记：{scan.notes}")

            # 读取全部节点条目（不过滤，由上层按 rel_path 判断 Input/Output）
            entries = node_db.get_catalog_entries(scan.catalog_id)

        return scan, entries, warnings

    except Exception as exc:
        warnings.append(f"扫描 Aspen 文件失败（{type(exc).__name__}）：{exc}")
        return _make_empty_scan(aspen_path), [], warnings
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception as exc:
                warnings.append(f"关闭 Aspen driver 时出错：{exc}")


def _make_empty_scan(aspen_path: str) -> Any:
    """
    在 Aspen 扫描失败时返回最小化的 CatalogScan 占位对象，
    避免上层代码因 None 报错。
    """
    from src.models.node_catalog import CatalogScan
    import uuid
    from datetime import datetime, timezone
    return CatalogScan(
        catalog_id=str(uuid.uuid4()),
        aspen_file_path=aspen_path,
        aspen_file_hash="",
        aspen_version="",
        n_blocks=0,
        n_streams=0,
        n_entries=0,
        scan_depth=0,
        created_at=datetime.now(timezone.utc).isoformat(),
        notes="扫描失败（空占位）",
    )


# ---------------------------------------------------------------------------
# A3-2  从 catalog entries + 语义规则构建 TunableVariable 列表
# ---------------------------------------------------------------------------

def _build_tunable_variables(
    entries: list[dict[str, Any]],
    rules: dict[str, dict[str, Any]],
) -> list[TunableVariable]:
    """
    遍历 catalog 中所有 Input 节点，结合语义规则生成 TunableVariable 列表。

    命中规则且规则有经验边界 → confidence="high" 或 "medium"
    无规则但节点为 Input 类型 → confidence="low"，边界为 None
    扫描失败节点（sample_error 非空）不进入推荐列表

    Parameters
    ----------
    entries:
        NodeDB.get_catalog_entries() 返回的字典列表。
    rules:
        load_semantic_rules() 返回的 {equipment_type: rule_dict} 映射。

    Returns
    -------
    list[TunableVariable]
    """
    result: list[TunableVariable] = []
    seen_paths: set[str] = set()
    failed_warnings: list[str] = []  # 收集失败节点，供调用方追加到 scan_warnings

    for entry in entries:
        abs_path: str = entry.get("abs_path", "")
        if not abs_path or abs_path in seen_paths:
            continue

        # 只处理 Input 节点
        rel_path: str = entry.get("rel_path", "")
        if not _is_input_path(rel_path):
            continue

        # P1-2：扫描失败节点记入 warnings，不进入推荐列表
        sample_error = entry.get("sample_error", "")
        if sample_error:
            failed_warnings.append(f"Input 节点读取失败：{abs_path}（{sample_error}）")
            continue

        # P1-3：只允许叶节点且 value_type 为数值（1=INTEGER / 2=REAL）进入候选
        if not entry.get("is_leaf", True):
            continue
        vtype_int = int(entry.get("value_type", 0))
        if vtype_int not in (1, 2):
            continue

        block_type: str = (entry.get("block_type") or "").upper()
        rule_dict = rules.get(block_type, {})
        fields: dict[str, Any] = rule_dict.get("fields", {})

        # 尝试用规则匹配
        matched_field, matched_field_name = _match_field_for_entry(entry, fields, is_input=True)

        # P1-1：tunable: false 表示规则明确声明此字段不可调，跳过
        if matched_field is not None and matched_field.get("tunable") is False:
            continue

        var = _make_tunable_variable(entry, matched_field, matched_field_name)
        if var is not None:
            result.append(var)
            seen_paths.add(abs_path)

    return result, failed_warnings


def _is_input_path(rel_path: str) -> bool:
    """判断相对路径是否属于 Input 子树。"""
    # rel_path 格式如 "Input\BASIS_RR" 或 "Input\FEED_STAGE\0318"
    parts = rel_path.replace("/", "\\").split("\\")
    return len(parts) >= 1 and parts[0].upper() == "INPUT"


def _is_output_path(rel_path: str) -> bool:
    """判断相对路径是否属于 Output 子树。"""
    parts = rel_path.replace("/", "\\").split("\\")
    return len(parts) >= 1 and parts[0].upper() == "OUTPUT"


def _match_field_for_entry(
    entry: dict[str, Any],
    fields: dict[str, Any],
    is_input: bool,
) -> tuple[dict[str, Any] | None, str]:
    """
    尝试将 catalog entry 与语义规则字段匹配。

    遍历规则字段，检查 candidates[].pattern 是否与 entry.rel_path 匹配。
    按 priority 降序取最佳匹配。

    Returns
    -------
    (field_data, field_name)
        - 未命中时返回 (None, "")
    """
    best_field: dict[str, Any] | None = None
    best_field_name: str = ""
    best_priority: int = -1

    rel_path_upper = entry.get("rel_path", "").upper()

    for field_name, field_data in fields.items():
        # 跳过不符合 Input/Output 方向的字段
        candidates = field_data.get("candidates") or []
        for cand in candidates:
            pattern = (cand.get("pattern") or "").upper()
            priority = int(cand.get("priority", 0))

            # 检查候选 pattern 方向与节点方向是否一致
            if is_input and not pattern.startswith("INPUT\\"):
                continue
            if not is_input and not pattern.startswith("OUTPUT\\"):
                continue

            if priority <= best_priority:
                continue

            if _path_matches_pattern(rel_path_upper, pattern):
                best_field = field_data
                best_field_name = field_name
                best_priority = priority

    return best_field, best_field_name


def _path_matches_pattern(rel_path_upper: str, pattern_upper: str) -> bool:
    """
    判断节点 rel_path 是否匹配规则 pattern。

    支持：
    - 精确匹配（无通配符）
    - * 匹配单个路径段（不含 \\）
    - ** 匹配任意数量路径段
    大小写不敏感（统一转大写后比较）。
    """
    rel_upper = rel_path_upper.upper()
    pat_upper = pattern_upper.upper()

    if "*" not in pat_upper:
        # 精确匹配：rel_path 以 pattern 开头或完全相等
        return rel_upper == pat_upper or rel_upper.startswith(pat_upper + "\\")

    # glob 匹配：按路径段比较
    path_segs = rel_upper.split("\\")
    pat_segs  = pat_upper.split("\\")
    return _match_segs(path_segs, 0, pat_segs, 0)


def _match_segs(path: list[str], pi: int, pat: list[str], si: int) -> bool:
    """递归路径段 glob 匹配（与 manifest.py 保持逻辑一致）。"""
    while si < len(pat):
        if pat[si] == "**":
            # ** 匹配零或多段
            for skip in range(pi, len(path) + 1):
                if _match_segs(path, skip, pat, si + 1):
                    return True
            return False
        if pi >= len(path):
            return False
        seg_pat = pat[si]
        seg_val = path[pi]
        if seg_pat == "*" or _glob_single(seg_val, seg_pat):
            pi += 1
            si += 1
        else:
            return False
    return pi >= len(path)


def _glob_single(s: str, pat: str) -> bool:
    """单段的 fnmatch 式匹配（* 可出现在段内，如 OUTPUT\\QREB*）。"""
    import fnmatch
    return fnmatch.fnmatch(s, pat)


def _make_tunable_variable(
    entry: dict[str, Any],
    field_data: dict[str, Any] | None,
    field_name: str,
) -> TunableVariable | None:
    """
    从 catalog entry + 可选语义字段构建 TunableVariable。

    无法构建（路径为空等）时返回 None。
    """
    abs_path = entry.get("abs_path", "")
    if not abs_path:
        return None

    unit = entry.get("unit_string") or "-"
    current_value: float | None = None
    raw_val = entry.get("sample_value")
    if raw_val is not None:
        try:
            current_value = float(raw_val)
        except (TypeError, ValueError):
            pass

    # 判断 integer / continuous（基于节点 value_type：1=INTEGER, 2=REAL）
    vtype_int = int(entry.get("value_type", 0))
    suggested_type: str = "integer" if vtype_int == 1 else "continuous"

    if field_data is None:
        # 无规则命中：confidence=low
        return TunableVariable(
            aspen_path=abs_path,
            semantic_role="",
            suggested_type=suggested_type,
            current_value=current_value,
            suggested_lower=None,
            suggested_upper=None,
            unit=unit,
            confidence="low",
            reason="未匹配语义规则，仅按路径推断为可调输入节点",
        )

    # 有规则命中：从规则 candidates 中读取经验边界（如有）
    lower, upper = _extract_bounds_from_rule(field_data)
    confidence, reason = _determine_confidence(field_data, field_name, lower, upper)

    return TunableVariable(
        aspen_path=abs_path,
        semantic_role=field_name,
        suggested_type=suggested_type,
        current_value=current_value,
        suggested_lower=lower,
        suggested_upper=upper,
        unit=unit,
        confidence=confidence,
        reason=reason,
    )


def _extract_bounds_from_rule(field_data: dict[str, Any]) -> tuple[float | None, float | None]:
    """
    从规则字段 data 中提取经验上下界。

    读取优先级（高 → 低）：
    1. suggested_lower / suggested_upper （真实 YAML schema 使用的字段名）
    2. bounds: [lo, hi]   （向后兼容的旧格式）
    3. lower_bound / upper_bound （备用格式）
    若规则中无边界信息则返回 (None, None)。
    """
    # 优先读真实 YAML schema 字段
    lo = _try_float(field_data.get("suggested_lower"))
    hi = _try_float(field_data.get("suggested_upper"))
    if lo is not None or hi is not None:
        return lo, hi

    # bounds: [lo, hi] 格式（兼容旧测试）
    bounds = field_data.get("bounds")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        try:
            return float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError):
            pass

    lower = field_data.get("lower_bound")
    upper = field_data.get("upper_bound")
    return _try_float(lower), _try_float(upper)


def _determine_confidence(
    field_data: dict[str, Any],
    field_name: str,
    lower: float | None,
    upper: float | None,
) -> tuple[str, str]:
    """
    根据规则数据判断置信度等级。

    优先级：
    1. 规则 YAML 中明确声明 confidence: high/medium/low → 直接采用
    2. 无声明但有完整边界（lower + upper 均非 None）→ high
    3. 无声明且边界不完整 → medium（规则命中但边界不足，需用户确认）
    注：调用方已确认 field_data 不为 None。
    """
    yaml_conf = (field_data.get("confidence") or "").strip().lower()
    if yaml_conf in ("high", "medium", "low"):
        if yaml_conf == "high":
            return "high", f"语义规则 '{field_name}' 命中，YAML 标注 confidence=high，边界 [{lower}, {upper}]"
        if yaml_conf == "medium":
            bounds_note = f"边界 [{lower}, {upper}]" if (lower is not None and upper is not None) else "边界需用户确认"
            return "medium", f"语义规则 '{field_name}' 命中，YAML 标注 confidence=medium，{bounds_note}"
        # low
        return "low", f"语义规则 '{field_name}' 命中，YAML 标注 confidence=low，边界未配置，需用户手动填写"

    # 无 YAML confidence 声明，按边界完整性退化推断
    if lower is not None and upper is not None:
        return "high", f"语义规则 '{field_name}' 命中，含明确经验边界 [{lower}, {upper}]"
    return (
        "medium",
        f"语义规则 '{field_name}' 命中，但规则中未配置经验边界，建议用户确认合理范围",
    )


def _try_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# A3-3  从 catalog entries + 语义规则构建 ReadableTarget 列表
# ---------------------------------------------------------------------------

def _build_readable_targets(
    entries: list[dict[str, Any]],
    rules: dict[str, dict[str, Any]],
) -> list[ReadableTarget]:
    """
    遍历 catalog 中所有 Output 节点，结合语义规则生成 ReadableTarget 列表。

    命中规则且 required_for 含 TAC/EMISSIONS → candidate_use="objective"
    路径含纯度关键词（MASSFRAC/MOLE_FRAC 等）→ candidate_use="constraint"
    两者均满足 → candidate_use="both"
    无规则命中的 Output 叶节点 → 不纳入列表（避免噪声）

    Parameters
    ----------
    entries:
        NodeDB.get_catalog_entries() 返回的字典列表。
    rules:
        load_semantic_rules() 返回的 {equipment_type: rule_dict} 映射。

    Returns
    -------
    (list[ReadableTarget], list[str])
        第二项是失败节点的 warnings 列表，供调用方追加到 scan_warnings。
    """
    result: list[ReadableTarget] = []
    seen_paths: set[str] = set()
    failed_warnings: list[str] = []

    for entry in entries:
        abs_path: str = entry.get("abs_path", "")
        if not abs_path or abs_path in seen_paths:
            continue

        rel_path: str = entry.get("rel_path", "")
        if not _is_output_path(rel_path):
            continue

        # P1-2：扫描失败节点记入 warnings，不进入推荐列表
        sample_error = entry.get("sample_error", "")
        if sample_error:
            failed_warnings.append(f"Output 节点读取失败：{abs_path}（{sample_error}）")
            continue

        # 跳过非叶节点（只对叶节点生成 ReadableTarget，减少冗余）
        if not entry.get("is_leaf", True):
            continue

        block_type: str = (entry.get("block_type") or "").upper()
        # stream 节点 block_type 为 "MATERIAL"，特殊处理
        rule_dict = rules.get(block_type, {})
        fields: dict[str, Any] = rule_dict.get("fields", {})

        matched_field, matched_field_name = _match_field_for_entry(entry, fields, is_input=False)

        if matched_field is None and not _is_purity_path(abs_path):
            # 无规则命中且不是纯度路径，跳过（减少噪声）
            continue

        target = _make_readable_target(entry, matched_field, matched_field_name)
        if target is not None:
            result.append(target)
            seen_paths.add(abs_path)

    return result, failed_warnings


def _is_purity_path(abs_path: str) -> bool:
    """判断节点路径是否与产品纯度/质量分数相关。"""
    return bool(_RE_PURITY.search(abs_path))


def _make_readable_target(
    entry: dict[str, Any],
    field_data: dict[str, Any] | None,
    field_name: str,
) -> ReadableTarget | None:
    """从 catalog entry + 可选语义字段构建 ReadableTarget。"""
    abs_path = entry.get("abs_path", "")
    if not abs_path:
        return None

    unit = entry.get("unit_string") or ""
    current_value: float | None = None
    raw_val = entry.get("sample_value")
    if raw_val is not None:
        try:
            current_value = float(raw_val)
        except (TypeError, ValueError):
            pass

    # 判断候选用途
    is_objective = False
    is_constraint = _is_purity_path(abs_path)

    if field_data is not None:
        required_for = field_data.get("required_for") or []
        if any(r.upper() in _OBJECTIVE_FOR for r in required_for):
            is_objective = True

    if is_objective and is_constraint:
        candidate_use = "both"
    elif is_objective:
        candidate_use = "objective"
    elif is_constraint:
        candidate_use = "constraint"
    else:
        # 无明确用途（规则命中但 required_for 不含目标）
        candidate_use = "objective"

    # P1-4：为纯度/组成约束路径生成稳定的 semantic_role
    # 规则命中时使用规则 field_name；无规则命中时按路径关键词推断
    effective_role = field_name
    if not effective_role and is_constraint:
        path_upper = abs_path.upper()
        if "MASSFRAC" in path_upper or "MASS_FRAC" in path_upper:
            effective_role = "mass_frac_product"
        elif "MOLEFRAC" in path_upper or "MOLE_FRAC" in path_upper:
            effective_role = "mole_frac_product"
        elif "MOLFRAC" in path_upper:
            effective_role = "mole_frac_product"
        else:
            effective_role = "purity_candidate"

    return ReadableTarget(
        aspen_path=abs_path,
        semantic_role=effective_role,
        candidate_use=candidate_use,
        unit=unit,
        current_value=current_value,
    )


# ---------------------------------------------------------------------------
# A3-4  计算语义覆盖率
# ---------------------------------------------------------------------------

def _compute_semantic_coverage(
    tunable_vars: list[TunableVariable],
    readable_targets: list[ReadableTarget],
    entries: list[dict[str, Any]],
) -> float:
    """
    计算语义覆盖率 = 被规则命中的节点数 / 总叶节点数。

    "被规则命中"定义：
    - TunableVariable 中 confidence != "low"（即有语义规则命中）
    - ReadableTarget 中 semantic_role != ""（即有语义规则命中）

    分母为 catalog 中所有叶节点的数量。
    """
    total_leaves = sum(1 for e in entries if e.get("is_leaf", True))
    if total_leaves == 0:
        return 0.0

    matched_paths: set[str] = set()
    for v in tunable_vars:
        if v.confidence != "low":
            matched_paths.add(v.aspen_path)
    for t in readable_targets:
        if t.semantic_role:
            matched_paths.add(t.aspen_path)

    return len(matched_paths) / total_leaves


# ---------------------------------------------------------------------------
# A3-5  主实现函数
# ---------------------------------------------------------------------------

def discover_tunables_impl(
    aspen_file_path: str,
    node_db_path: str,
    rules_dir: str = "configs/aspen_semantics",
    max_depth: int = 6,
) -> TunableReport:
    """
    串联 A3-1 ~ A3-4，产出 TunableReport。

    异常时返回含 scan_warnings 的 TunableReport（空变量列表），不向上抛。

    Parameters
    ----------
    aspen_file_path:
        Aspen 仿真文件路径（.bkp / .apw）。
    node_db_path:
        NodeDB SQLite 文件路径。
    rules_dir:
        语义规则 YAML 目录，默认 "configs/aspen_semantics"。
    max_depth:
        节点树扫描最大深度，默认 6。

    Returns
    -------
    TunableReport
    """
    all_warnings: list[str] = []

    # 加载语义规则（不依赖 Aspen，可在导入时调用）
    try:
        from src.aspen_driver.manifest import load_semantic_rules
        rules = load_semantic_rules(rules_dir)
        if not rules:
            all_warnings.append(f"语义规则目录 '{rules_dir}' 为空或无法读取，将退化为低置信度推断")
    except Exception as exc:
        all_warnings.append(f"加载语义规则失败：{exc}")
        rules = {}

    # 计算 Aspen 文件 hash（用于 TunableReport）
    aspen_file_hash = _compute_file_hash(aspen_file_path)

    # 扫描 Aspen 文件
    scan, entries, scan_warnings = _scan_aspen_file(
        aspen_file_path, node_db_path, max_depth
    )
    all_warnings.extend(scan_warnings)

    if not entries:
        all_warnings.append("未获取到 catalog 节点数据（entries 为空），将返回空报告")
        return TunableReport(
            aspen_file=aspen_file_path,
            aspen_file_hash=aspen_file_hash,
            tunable_variables=[],
            readable_targets=[],
            scan_warnings=all_warnings,
            semantic_coverage=0.0,
        )

    # 构建变量和目标列表；两个函数均返回 (list, warnings)
    tunable_vars, var_warnings = _build_tunable_variables(entries, rules)
    readable_targets, tgt_warnings = _build_readable_targets(entries, rules)
    all_warnings.extend(var_warnings)
    all_warnings.extend(tgt_warnings)
    coverage = _compute_semantic_coverage(tunable_vars, readable_targets, entries)

    _log.info(
        "discover_tunables_impl: 发现 %d 个可调变量，%d 个可读目标，语义覆盖率 %.1f%%",
        len(tunable_vars), len(readable_targets), coverage * 100,
    )

    return TunableReport(
        aspen_file=aspen_file_path,
        aspen_file_hash=aspen_file_hash,
        tunable_variables=tunable_vars,
        readable_targets=readable_targets,
        scan_warnings=all_warnings,
        semantic_coverage=coverage,
    )


def _compute_file_hash(filepath: str) -> str:
    """计算文件 MD5，文件不存在或读取失败时返回空字符串。"""
    try:
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# A3-6  @tool 包装
# ---------------------------------------------------------------------------

@tool
def discover_tunables_tool(
    aspen_file_path: str,
    node_db_path: str,
    max_depth: int = 6,
) -> str:
    """扫描 Aspen 仿真文件，发现可调设计变量和可读输出节点，返回 TunableReport JSON。

    只读扫描，不运行仿真（不调用 Engine.Run2 / run_case）。
    扫描结果包含：
    - tunable_variables：可调设计变量列表（含置信度、建议边界）
    - readable_targets：可读输出节点列表（目标函数/约束候选）
    - semantic_coverage：语义规则覆盖率
    - scan_warnings：扫描过程中的非致命问题

    Args:
        aspen_file_path: Aspen 仿真文件路径（.bkp / .apw），绝对路径或相对于项目根目录。
        node_db_path: NodeDB SQLite 文件路径，用于存储 catalog 数据（不存在则自动创建）。
        max_depth: 节点树扫描最大深度，默认 6，建议 6~8。

    Returns:
        JSON 字符串格式的 TunableReport，或以 "错误：" 开头的失败描述。
    """
    try:
        report = discover_tunables_impl(
            aspen_file_path=aspen_file_path,
            node_db_path=node_db_path,
            max_depth=max_depth,
        )
        return _serialize_report(report)
    except Exception as exc:
        _log.exception("discover_tunables_tool 意外失败")
        return f"错误：discover_tunables_tool 意外失败（{type(exc).__name__}）：{exc}"


def _serialize_report(report: TunableReport) -> str:
    """将 TunableReport 序列化为 JSON 字符串。"""
    def _var_to_dict(v: TunableVariable) -> dict:
        return {
            "aspen_path": v.aspen_path,
            "semantic_role": v.semantic_role,
            "suggested_type": v.suggested_type,
            "current_value": v.current_value,
            "suggested_lower": v.suggested_lower,
            "suggested_upper": v.suggested_upper,
            "unit": v.unit,
            "confidence": v.confidence,
            "reason": v.reason,
        }

    def _target_to_dict(t: ReadableTarget) -> dict:
        return {
            "aspen_path": t.aspen_path,
            "semantic_role": t.semantic_role,
            "candidate_use": t.candidate_use,
            "unit": t.unit,
            "current_value": t.current_value,
        }

    payload = {
        "aspen_file": report.aspen_file,
        "aspen_file_hash": report.aspen_file_hash,
        "semantic_coverage": report.semantic_coverage,
        "scan_warnings": report.scan_warnings,
        "tunable_variables": [_var_to_dict(v) for v in report.tunable_variables],
        "readable_targets": [_target_to_dict(t) for t in report.readable_targets],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
