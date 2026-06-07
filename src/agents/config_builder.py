"""
config_builder.py — A4 配置构建器。

职责
----
接收 TunableReport（变量发现报告）+ OptimizationIntent（用户意图），
用纯规则方式生成 ConfigDraft（可写入 pareto_config.yaml 的配置草案）。

设计原则
--------
- 纯 Python + 可选 LLM，不导入 aspen_driver，不打开 Aspen，不调 driver。
- LLM 只做意图解析（free-text → OptimizationIntent），不直接生成配置字段。
- 所有配置字段由规则映射器生成；边界/路径为 None 时写 warning，不静默填假值。
- 不修改现有 pareto_config.yaml schema（to_yaml_dict 输出与已有格式完全兼容）。

对外接口
--------
  build_config_draft(report, intent) → ConfigDraft        # A4-1f 主函数
  parse_intent_from_text(text, llm_config) → OptimizationIntent  # A4-2a
  IntentParseError                                         # A4-2a 异常类
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.models.tunable import (
    ConfigDraft,
    GoalSpec,
    OptimizationIntent,
    ReadableTarget,
    TunableReport,
    TunableVariable,
)

_log = logging.getLogger(__name__)

# 模块级 import：让 monkeypatch 可在测试中替换这些名字
try:
    from src.agents.llm_client import chat, is_configured, load_llm_config  # noqa: F401
except Exception:  # llm_client 可能依赖可选依赖，降级处理
    chat = None  # type: ignore[assignment]
    is_configured = None  # type: ignore[assignment]
    load_llm_config = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 从经济模块读取默认参数（延迟导入，避免影响导入速度）
# ---------------------------------------------------------------------------

def _get_tac_defaults() -> dict[str, Any]:
    """从 TACConfig / UtilityCost 默认值组装 TAC 目标函数参数字典。"""
    try:
        from src.economics.tac import TACConfig
        cfg = TACConfig()
        uc = cfg.utility_cost
        return {
            "annualization_factor": cfg.annualization_factor,
            "operating_hours":      cfg.operating_hours,
            "utility_cost": {
                "steam_price":         uc.steam_price,
                "cooling_water_price": uc.cooling_water_price,
                "electricity_price":   uc.electricity_price,
            },
        }
    except Exception as exc:
        _log.warning("读取 TACConfig 默认值失败，使用内置兜底值：%s", exc)
        return {
            "annualization_factor": 0.1,
            "operating_hours":      8000.0,
            "utility_cost": {
                "steam_price":         14.19,
                "cooling_water_price": 0.354,
                "electricity_price":   0.0775,
            },
        }


def _get_emissions_defaults() -> dict[str, Any]:
    """从 EmissionsConfig / EmissionFactors 默认值组装排放目标函数参数字典。"""
    try:
        from src.economics.emissions import EmissionsConfig
        cfg = EmissionsConfig()
        ef = cfg.emission_factors
        return {
            "operating_hours": cfg.operating_hours,
            "emission_factors": {
                "steam_factor":       ef.steam_factor,
                "electricity_factor": ef.electricity_factor,
            },
        }
    except Exception as exc:
        _log.warning("读取 EmissionsConfig 默认值失败，使用内置兜底值：%s", exc)
        return {
            "operating_hours": 8000.0,
            "emission_factors": {
                "steam_factor":       66.0,
                "electricity_factor": 0.581,
            },
        }


# ---------------------------------------------------------------------------
# A4-1a  目标函数映射器
# ---------------------------------------------------------------------------

def _map_goal_to_objective(
    goal: GoalSpec,
    targets: list[ReadableTarget],
    tac_defaults: dict[str, Any],
    emissions_defaults: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any] | None:
    """将一个 GoalSpec 映射为 pareto_config.yaml objectives 条目。

    支持的 metric：TAC / emissions / purity / yield / flow / custom。
    找不到匹配节点时返回 None，并向 warnings 追加说明。
    """
    metric = goal.metric.lower()
    minimize = (goal.direction == "min")

    # ── TAC ────────────────────────────────────────────────────────────────
    if metric == "tac":
        obj: dict[str, Any] = {
            "name":                 "TAC",
            "type":                 "tac",
            "minimize":             minimize,
            "annualization_factor": tac_defaults["annualization_factor"],
            "operating_hours":      tac_defaults["operating_hours"],
            "utility_cost":         tac_defaults["utility_cost"],
        }
        return obj

    # ── emissions ──────────────────────────────────────────────────────────
    if metric == "emissions":
        return {
            "name":              "EMISSIONS",
            "type":              "emissions",
            "minimize":          minimize,
            "operating_hours":   emissions_defaults["operating_hours"],
            "emission_factors":  emissions_defaults["emission_factors"],
        }

    # ── flow / yield：从 targets 找 semantic_role 含对应关键词的节点 ────────
    if metric in ("flow", "yield"):
        keyword = "flow" if metric == "flow" else "flow"  # yield 也找 flow 节点
        candidates = [
            t for t in targets
            if t.candidate_use in ("objective", "both")
            and keyword in t.semantic_role.lower()
        ]
        if not candidates:
            warnings.append(
                f"目标 metric='{goal.metric}'：在 ReadableTarget 列表中未找到"
                f" semantic_role 含 '{keyword}' 的节点，该目标将被跳过。"
                "请改用 metric='custom' 并手动指定 custom_aspen_path。"
            )
            return None
        t = candidates[0]
        return {
            "name":       goal.metric.upper(),
            "type":       "aspen_path",
            "aspen_path": t.aspen_path,
            "minimize":   minimize,
            "unit":       t.unit,
        }

    # ── purity：通常作约束，但支持作目标 ────────────────────────────────────
    if metric == "purity":
        cands = [
            t for t in targets
            if t.candidate_use in ("objective", "both")
            and any(k in t.semantic_role.lower() for k in ("frac", "purity"))
        ]
        if not cands:
            warnings.append(
                "目标 metric='purity'：未找到 semantic_role 含 'frac'/'purity' 的"
                " objective 节点，该目标将被跳过。"
            )
            return None
        t = cands[0]
        return {
            "name":       "PURITY",
            "type":       "aspen_path",
            "aspen_path": t.aspen_path,
            "minimize":   minimize,
            "unit":       t.unit,
        }

    # ── custom：直接使用用户指定路径 ──────────────────────────────────────────
    if metric == "custom":
        if not goal.custom_aspen_path:
            warnings.append(
                "目标 metric='custom'：custom_aspen_path 为空，该目标将被跳过。"
            )
            return None
        return {
            "name":       "CUSTOM",
            "type":       "aspen_path",
            "aspen_path": goal.custom_aspen_path,
            "minimize":   minimize,
            "unit":       "",
        }

    # ── 未知 metric ──────────────────────────────────────────────────────────
    warnings.append(
        f"目标 metric='{goal.metric}' 不支持，将被跳过。"
        "支持的值：TAC / emissions / purity / yield / flow / custom。"
    )
    return None


# ---------------------------------------------------------------------------
# A4-1b  约束映射器
# ---------------------------------------------------------------------------

def _map_constraint_to_dict(
    constraint: GoalSpec,
    targets: list[ReadableTarget],
    warnings: list[str],
) -> dict[str, Any] | None:
    """将一个约束型 GoalSpec 映射为 pareto_config.yaml constraints 条目。"""
    metric = constraint.metric.lower()

    # 阈值类约束必须有数值 threshold（手工构造路径的防御层；LLM 路径已由 schema 校验拦截）
    _THRESHOLD_NEEDED = {"purity", "yield", "custom"}
    if metric in _THRESHOLD_NEEDED and constraint.target_value is None:
        warnings.append(
            f"约束 metric='{constraint.metric}'：target_value 为 None，"
            "无法生成有效阈值约束（threshold: null 会导致优化器解析失败）。"
            "请手动补充 target_value 后重新构建草案，该约束将被跳过。"
        )
        return None

    # ── purity：找 candidate_use 含 constraint 且 semantic_role 含分数关键词 ──
    if metric == "purity":
        frac_roles = ("mass_frac", "mole_frac", "massfrac", "molefrac", "frac", "purity")
        cands = [
            t for t in targets
            if t.candidate_use in ("constraint", "both")
            and any(r in t.semantic_role.lower() for r in frac_roles)
        ]
        if not cands:
            warnings.append(
                "约束 metric='purity'：未找到 candidate_use='constraint'/'both' 且"
                " semantic_role 含分数/纯度关键词的节点，该约束将被跳过。"
            )
            return None
        t = cands[0]
        # purity 语义是"产品纯度下限"，operator 固定为 ">="。
        # 若需表达"杂质上限 <= 阈值"，应使用 metric='custom'，不复用 purity。
        return {
            "name":       "purity_min",
            "aspen_path": t.aspen_path,
            "operator":   ">=",
            "threshold":  constraint.target_value,
        }

    # ── yield：类似 purity，找 flow/yield 类约束节点 ─────────────────────────
    if metric == "yield":
        cands = [
            t for t in targets
            if t.candidate_use in ("constraint", "both")
            and "flow" in t.semantic_role.lower()
        ]
        if not cands:
            warnings.append(
                "约束 metric='yield'：未找到合适的约束节点，该约束将被跳过。"
            )
            return None
        t = cands[0]
        # yield 语义是"产率下限"，operator 固定为 ">="。
        return {
            "name":       "yield_min",
            "aspen_path": t.aspen_path,
            "operator":   ">=",
            "threshold":  constraint.target_value,
        }

    # ── custom：直接使用 custom_aspen_path ────────────────────────────────────
    if metric == "custom":
        if not constraint.custom_aspen_path:
            warnings.append(
                "约束 metric='custom'：custom_aspen_path 为空，该约束将被跳过。"
            )
            return None
        op = ">=" if constraint.direction == "max" else "<="
        return {
            "name":       "custom_constraint",
            "aspen_path": constraint.custom_aspen_path,
            "operator":   op,
            "threshold":  constraint.target_value,
        }

    warnings.append(
        f"约束 metric='{constraint.metric}' 暂不支持自动映射，该约束将被跳过。"
        "支持的值：purity / yield / custom。"
    )
    return None


# ---------------------------------------------------------------------------
# A4-1c  设计变量映射器
# ---------------------------------------------------------------------------

def _map_tunable_to_design_var(
    var: TunableVariable,
    warnings: list[str],
) -> dict[str, Any]:
    """将 TunableVariable 映射为 pareto_config.yaml design_variables 条目。"""
    # 路径末段作为变量名（去掉反斜杠）
    name = var.aspen_path.replace("\\", "_").replace("/", "_").lstrip("_")
    # 取最后两段作为简短名称，避免过长
    parts = [p for p in var.aspen_path.replace("/", "\\").split("\\") if p]
    if len(parts) >= 2:
        name = f"{parts[-2]}_{parts[-1]}"
    else:
        name = parts[-1] if parts else "var"

    # 边界为 None 时写 warning
    if var.suggested_lower is None or var.suggested_upper is None:
        warnings.append(
            f"请手动填写 {var.aspen_path} 的变量边界"
            f"（suggested_lower={var.suggested_lower}，"
            f"suggested_upper={var.suggested_upper}）。"
            f"  置信度={var.confidence}，原因：{var.reason}"
        )

    # initial_value：用当前值，若无则取边界中点（若边界均有效），再否则 None
    initial: float | None = var.current_value
    if initial is None and var.suggested_lower is not None and var.suggested_upper is not None:
        initial = (var.suggested_lower + var.suggested_upper) / 2.0

    return {
        "name":          name,
        "description":   f"{var.semantic_role} ({var.aspen_path})",
        "aspen_path":    var.aspen_path,
        "type":          var.suggested_type,
        "lower_bound":   var.suggested_lower,
        "upper_bound":   var.suggested_upper,
        "initial_value": initial,
        "unit":          var.unit,
    }


# ---------------------------------------------------------------------------
# A4-1d  优化器配置构建器
# ---------------------------------------------------------------------------

def _build_optimizer_section(
    intent: OptimizationIntent,
    n_vars: int,
    n_objectives: int,
) -> dict[str, Any]:
    """构建 pareto_config.yaml optimizer 段。"""
    # n_initial：用意图值，否则用经验公式 max(10, 5 * n_vars)
    n_initial = intent.n_initial if intent.n_initial > 0 else max(10, 5 * n_vars)
    # n_iterations：用意图值，否则用 2 * n_initial
    n_iter = intent.n_iterations if intent.n_iterations > 0 else 2 * n_initial
    # 优化类型
    opt_type = "pareto_bayesian" if n_objectives >= 2 else "bayesian"

    return {
        "type":                 opt_type,
        "n_initial_points":     n_initial,
        "n_iterations":         n_iter,
        "scalarization":        "chebyshev",
        "acquisition_function": "EI",
        "random_seed":          42,
        # feasibility_filter：默认关闭，等用户手动开启
        "feasibility_filter": {
            "enabled":           False,
            "model":             "extra_trees",
            "min_samples":       20,
            "threshold":         0.40,
            "candidate_pool_size": 200,
            "random_seed":       42,
        },
        # early_stopping：与 demo_case 配置一致
        "early_stopping": {
            "enabled":                   True,
            "min_iterations":            max(30, n_initial + 10),
            "patience":                  10,
            "min_delta":                 1.0e-6,
            "relative_delta":            None,
            "max_duplicate_suggestions": 3,
            "check_hypervolume":         True,
            "check_first_front":         True,
        },
    }


# ---------------------------------------------------------------------------
# A4-1e  提取配置构建器
# ---------------------------------------------------------------------------

_BLOCK_ROOT  = r"\Data\Blocks"
_STREAM_ROOT = r"\Data\Streams"


def _extract_name_from_path(path: str, root: str) -> str | None:
    """从 Aspen 绝对路径中提取 block 或 stream 名。"""
    upper = path.upper()
    root_upper = root.upper()
    idx = upper.find(root_upper)
    if idx == -1:
        return None
    rest = path[idx + len(root):]
    # 跳过前导分隔符，取第一个路径段
    rest = rest.lstrip("\\/")
    name = rest.split("\\")[0].split("/")[0]
    return name if name else None


def _build_extraction_section(
    aspen_file: str,
    node_db_path: str,
    tunable_vars: list[TunableVariable],
    targets: list[ReadableTarget],
) -> dict[str, Any]:
    """构建 pareto_config.yaml extraction 段。"""
    blocks: list[str] = []
    streams: list[str] = []

    all_paths = (
        [v.aspen_path for v in tunable_vars]
        + [t.aspen_path for t in targets]
    )
    for path in all_paths:
        name = _extract_name_from_path(path, _BLOCK_ROOT)
        if name and name not in blocks:
            blocks.append(name)
            continue
        name = _extract_name_from_path(path, _STREAM_ROOT)
        if name and name not in streams:
            streams.append(name)

    # check_status_paths：blocks + streams 各自的顶层路径
    check_paths = (
        [rf"\Data\Blocks\{b}" for b in blocks]
        + [rf"\Data\Streams\{s}" for s in streams]
    )

    return {
        "check_status_paths":    check_paths,
        "blocks":                blocks,
        "streams":               streams,
        "block_max_depth":       3,
        "stream_max_depth":      3,
        "stream_output_subtree": r"Output\STR_MAIN",
        "strict_extraction":     False,
        "mode":                  "manifest",
        "catalog_db":            node_db_path,
        "manifest_id":           "auto",
        "semantic_rules_dir":    "configs/aspen_semantics",
        "build_manifest_if_missing": True,
        "write_node_values":     True,
        "strict_manifest":       True,
    }


# ---------------------------------------------------------------------------
# A4-1f  主函数
# ---------------------------------------------------------------------------

def build_config_draft(
    report: TunableReport,
    intent: OptimizationIntent,
    node_db_path: str = "",
) -> ConfigDraft:
    """将 TunableReport + OptimizationIntent 组装为 ConfigDraft。

    纯规则映射，不打开 Aspen，不调 driver，不依赖 LLM。

    Parameters
    ----------
    report:
        discover_tunables_tool 产出的变量发现报告。
    intent:
        用户优化意图（可由 parse_intent_from_text 解析，也可手动构造）。
    node_db_path:
        NodeDB 路径；为空时取 report.aspen_file 同目录下 output/node.db。

    Returns
    -------
    ConfigDraft
        包含 design_variables / objectives / constraints / optimizer /
        extraction / warnings 的配置草案。
    """
    import os
    warnings: list[str] = []

    # ── 推断 node_db_path ──────────────────────────────────────────────────
    if not node_db_path and report.aspen_file:
        _dir = os.path.dirname(os.path.abspath(report.aspen_file))
        node_db_path = os.path.join(_dir, "output", "node.db")

    # ── 读取经济模块默认值 ─────────────────────────────────────────────────
    tac_defs   = _get_tac_defaults()
    emiss_defs = _get_emissions_defaults()

    # ── 设计变量 ───────────────────────────────────────────────────────────
    design_vars: list[dict[str, Any]] = []
    for var in report.tunable_variables:
        dv = _map_tunable_to_design_var(var, warnings)
        design_vars.append(dv)
        if var.confidence != "high":
            warnings.append(
                f"变量 {var.aspen_path} 的置信度为 '{var.confidence}'，"
                f"建议边界 [{var.suggested_lower}, {var.suggested_upper}] 需用户确认。"
                f"  原因：{var.reason}"
            )

    # ── 目标函数 ───────────────────────────────────────────────────────────
    all_targets = report.readable_targets
    objectives: list[dict[str, Any]] = []
    for goal in intent.goals:
        obj = _map_goal_to_objective(
            goal, all_targets, tac_defs, emiss_defs, warnings
        )
        if obj is not None:
            objectives.append(obj)

    # ── 约束 ───────────────────────────────────────────────────────────────
    constraints: list[dict[str, Any]] = []
    for con in intent.hard_constraints:
        c = _map_constraint_to_dict(con, all_targets, warnings)
        if c is not None:
            constraints.append(c)

    # ── 优化器 ─────────────────────────────────────────────────────────────
    optimizer = _build_optimizer_section(intent, len(design_vars), len(objectives))

    # ── 提取配置 ───────────────────────────────────────────────────────────
    extraction = _build_extraction_section(
        report.aspen_file,
        node_db_path,
        report.tunable_variables,
        report.readable_targets,
    )

    # ── 置信度摘要 ─────────────────────────────────────────────────────────
    n_high   = sum(1 for v in report.tunable_variables if v.confidence == "high")
    n_med    = sum(1 for v in report.tunable_variables if v.confidence == "medium")
    n_low    = sum(1 for v in report.tunable_variables if v.confidence == "low")
    n_total  = len(report.tunable_variables)
    null_bounds = sum(
        1 for dv in design_vars
        if dv.get("lower_bound") is None or dv.get("upper_bound") is None
    )
    confidence_summary = (
        f"共 {n_total} 个设计变量（high={n_high}, medium={n_med}, low={n_low}），"
        f"{null_bounds} 个变量边界待补充；"
        f"{len(objectives)} 个目标函数，{len(constraints)} 个约束。"
        f"语义覆盖率={report.semantic_coverage:.0%}。"
    )
    if warnings:
        confidence_summary += f" 共 {len(warnings)} 条待确认警告。"

    return ConfigDraft(
        aspen_file=report.aspen_file,
        design_variables=design_vars,
        objectives=objectives,
        constraints=constraints,
        optimizer=optimizer,
        extraction=extraction,
        warnings=warnings,
        confidence_summary=confidence_summary,
    )


# ---------------------------------------------------------------------------
# A4-2  LLM 意图解析器
# ---------------------------------------------------------------------------

class IntentParseError(ValueError):
    """LLM 输出无法解析为合法 OptimizationIntent 时抛出。"""


# JSON Schema：LLM 输出必须满足此结构
_INTENT_SCHEMA = {
    "type": "object",
    "required": ["goals"],
    "properties": {
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["metric", "direction"],
                "properties": {
                    "metric":            {"type": "string"},
                    "direction":         {"enum": ["min", "max"]},
                    "target_value":      {"type": ["number", "null"]},
                    "custom_aspen_path": {"type": ["string", "null"]},
                },
            },
        },
        "hard_constraints": {
            "type": "array",
            "items": {"type": "object"},
        },
        "n_initial":    {"type": "integer", "minimum": 1},
        "n_iterations": {"type": "integer", "minimum": 1},
        "notes":        {"type": "string"},
    },
}

_INTENT_SYSTEM_PROMPT = """\
你是一个化工过程优化意图解析助手。
用户会用自然语言描述优化目标和约束，你需要将其解析为结构化 JSON。

输出 JSON 格式（严格按此结构，不添加任何说明文字）：
{
  "goals": [
    {"metric": "TAC", "direction": "min"},
    {"metric": "emissions", "direction": "min"},
    {"metric": "flow", "direction": "max"},
    {"metric": "custom", "direction": "min", "custom_aspen_path": "\\\\Data\\\\..."}
  ],
  "hard_constraints": [
    {"metric": "purity", "direction": "max", "target_value": 0.9}
  ],
  "n_initial": 20,
  "n_iterations": 60,
  "notes": "原始用户意图"
}

metric 取值规则：
- TAC        → 总年化成本（minimize）
- emissions  → CO₂排放（minimize）
- purity     → 产品纯度约束（通常 maximize 或设阈值）
- yield      → 产率
- flow       → 质量/摩尔流量
- custom     → 自定义 Aspen 节点路径

若用户意图不够明确，使用 TAC 最小化 + 排放最小化作为默认目标。
只输出 JSON，不输出任何解释。"""


def _validate_intent_schema(data: dict) -> None:
    """基本 schema 校验，不依赖 jsonschema 库。

    goals 和 hard_constraints 中的每一项都必须提供 metric 和合法的 direction。
    hard_constraints 中阈值类约束（purity / yield / custom）必须提供数值 target_value。
    缺字段、非法 direction、缺 target_value 均抛 IntentParseError，不静默默认。
    """
    if not isinstance(data, dict):
        raise IntentParseError("LLM 输出不是 JSON 对象")
    if "goals" not in data:
        raise IntentParseError("LLM 输出缺少 'goals' 字段")
    if not isinstance(data["goals"], list):
        raise IntentParseError("'goals' 字段不是列表")

    # 需要 target_value 的约束 metric 类型（阈值约束）
    _THRESHOLD_METRICS = {"purity", "yield", "custom"}

    def _check_items(items: list, section: str, require_threshold: bool = False) -> None:
        for i, g in enumerate(items):
            if not isinstance(g, dict):
                raise IntentParseError(f"{section}[{i}] 不是字典")
            if "metric" not in g:
                raise IntentParseError(f"{section}[{i}] 缺少 'metric' 字段")
            if "direction" not in g:
                raise IntentParseError(
                    f"{section}[{i}] 缺少 'direction' 字段（metric='{g.get('metric')}'）。"
                    "约束/目标的方向必须显式指定，不允许静默默认。"
                )
            if g["direction"] not in ("min", "max"):
                raise IntentParseError(
                    f"{section}[{i}].direction='{g['direction']}' 不合法，仅支持 'min'/'max'"
                )
            # hard_constraints 的阈值类约束必须提供数值 target_value
            if require_threshold and g.get("metric", "").lower() in _THRESHOLD_METRICS:
                tv = g.get("target_value")
                if tv is None:
                    raise IntentParseError(
                        f"{section}[{i}]（metric='{g['metric']}'）缺少 'target_value'。"
                        "阈值约束必须显式提供数值阈值，不允许 null/缺失。"
                    )
                if not isinstance(tv, (int, float)):
                    raise IntentParseError(
                        f"{section}[{i}].target_value='{tv}' 不是数值"
                        f"（metric='{g['metric']}'），阈值必须是数字。"
                    )

    _check_items(data["goals"], "goals", require_threshold=False)

    # hard_constraints 可选字段，但若存在则必须合法，且阈值类约束需要 target_value
    constraints = data.get("hard_constraints", [])
    if not isinstance(constraints, list):
        raise IntentParseError("'hard_constraints' 字段不是列表")
    _check_items(constraints, "hard_constraints", require_threshold=True)


def _dict_to_intent(data: dict) -> OptimizationIntent:
    """将已通过 _validate_intent_schema 校验的 dict 转为 OptimizationIntent。

    所有关键字段（metric / direction）均通过校验，此处不再静默默认。
    """
    def _to_goal(d: dict) -> GoalSpec:
        return GoalSpec(
            metric=str(d["metric"]),          # 校验已保证存在
            direction=d["direction"],          # 校验已保证合法（min/max），不静默默认
            target_value=d.get("target_value"),
            custom_aspen_path=d.get("custom_aspen_path"),
        )

    goals = [_to_goal(g) for g in data.get("goals", [])]
    constraints = [_to_goal(c) for c in data.get("hard_constraints", [])]
    return OptimizationIntent(
        goals=goals,
        hard_constraints=constraints,
        n_initial=int(data.get("n_initial", 20)),
        n_iterations=int(data.get("n_iterations", 60)),
        notes=str(data.get("notes", "")),
    )


def _default_intent(notes: str = "") -> OptimizationIntent:
    """降级返回：TAC 最小化 + 排放最小化默认意图。"""
    return OptimizationIntent(
        goals=[
            GoalSpec(metric="TAC",       direction="min"),
            GoalSpec(metric="emissions", direction="min"),
        ],
        notes=notes,
    )


def parse_intent_from_text(
    text: str,
    llm_config: Any = None,
) -> OptimizationIntent:
    """将自由文本意图解析为 OptimizationIntent。

    LLM 只负责文本→结构化 JSON，不直接生成任何配置字段。

    Parameters
    ----------
    text:
        用户描述优化意图的自然语言文本。
    llm_config:
        LLMConfig 实例；None 时尝试从环境变量读取；
        未配置或调用失败时降级返回默认意图。

    Returns
    -------
    OptimizationIntent
        解析成功时返回 LLM 解析结果；
        降级时返回 TAC+排放 默认意图，notes 中说明原因。

    Raises
    ------
    IntentParseError
        LLM 有配置但输出不符合 schema 时抛出（调用方可捕获后选择降级）。
    """
    # 直接引用模块级名字（monkeypatch 可替换）
    if is_configured is None or load_llm_config is None or chat is None:
        return _default_intent(notes="llm_client 未就绪，已使用默认意图（TAC最小+排放最小）")

    try:
        cfg = llm_config if llm_config is not None else load_llm_config()
    except Exception as exc:
        _log.warning("无法加载 LLMConfig，使用默认意图：%s", exc)
        return _default_intent(notes=f"LLM 配置加载失败：{exc}；已使用默认意图（TAC最小+排放最小）")

    if not is_configured(cfg):
        return _default_intent(
            notes="LLM 未配置，已使用默认意图（TAC最小化 + 排放最小化）"
        )

    try:
        raw = chat(cfg, system=_INTENT_SYSTEM_PROMPT, user=text)
    except Exception as exc:
        _log.warning("LLM 调用失败，使用默认意图：%s", exc)
        return _default_intent(notes=f"LLM 调用失败：{exc}；已使用默认意图（TAC最小+排放最小）")

    # 提取 JSON：LLM 可能包裹在 ```json...``` 中
    json_text = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", json_text)
    if m:
        json_text = m.group(1).strip()

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise IntentParseError(
            f"LLM 输出无法解析为 JSON：{exc}\n原始输出：{raw[:500]}"
        ) from exc

    _validate_intent_schema(data)
    intent = _dict_to_intent(data)
    if not intent.notes:
        intent.notes = text
    return intent
