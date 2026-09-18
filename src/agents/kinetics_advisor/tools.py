"""
tools.py — KineticsAdvisor agent 的确定性工具函数。

职责（不含任何 LLM 调用，纯计算/解析，便于单测）：
  - heuristic_kinetics_bounds：无 LLM 时的规则兜底，从物理常识给出 Ea/k0 的
    保守搜索边界（类比 boundary_advisor.tools.heuristic_k）。
  - build_kinetics_variables_block：把变量元信息渲染成给 LLM 的文本清单。
  - parse_llm_kinetics_json：稳健解析 LLM 返回的边界推荐 JSON。
  - format_fit_result / format_recommendation：报告格式化。

设计原则
--------
  - 所有函数对缺失字段、异常输入都返回安全默认值，绝不抛异常给上层。
  - heuristic_kinetics_bounds 是 LLM 不可用时的退路，保证 KineticsAdvisor
    在无网络/无 key 环境下仍能产出保守但合理的边界。
  - 与 boundary_advisor.tools 的 k 倍数机制保持一致的设计哲学：
    有初值时用倍数锚定初值；无初值时给一个宽泛但物理合理的绝对范围
    （只用于"新增反应、完全没有参考值"的极端冷启动场景）。
"""
from __future__ import annotations

import json
import logging
import re

from src.models.kinetics import KineticsFitResult, KineticsRecommendation, KineticsVarMeta

_log = logging.getLogger(__name__)

# 活化能典型物理范围（kJ/mol），与 fitting.py 的 _EA_TYPICAL_* 保持一致，
# 用于无初值时的绝对兜底范围（覆盖大多数液相/气相反应）。
_EA_ABS_LO_KJ = 40.0
_EA_ABS_HI_KJ = 400.0

# 无初值时 k0 的搜索倍数窗口（以 1.0 为参考锚点在 log 空间展开）不适用，
# k0 跨越的数量级过大，无初值场景直接返回 None 边界，交给用户填写。


# ---------------------------------------------------------------------------
# 规则兜底（无 LLM 时使用）
# ---------------------------------------------------------------------------

def heuristic_kinetics_bounds(v: KineticsVarMeta) -> tuple[float | None, float | None, str]:
    """
    无 LLM 时，从物理常识推断 (lower, upper, reason)。

    活化能（Ea）：
      - 有初值：以初值为中心，[initial/1.5, initial*1.5]（保守倍数，
        活化能对速率呈指数影响，边界不宜过宽）。
      - 无初值：退化为化学反应活化能的典型物理范围 [40, 400] kJ/mol
        （单位需与调用方约定一致，本函数按 v.unit 原样返回数值，
        不做单位换算——假定 v.unit 已是 kJ/mol 或调用方自行换算）。

    频率因子（k0）：
      - 有初值：k0 跨越的数量级极大，用较宽的倍数窗口 [initial/10, initial*10]
        （对数空间锚定，避免线性倍数在小初值时边界过窄）。
      - 无初值：无法给出有意义的绝对范围（k0 依赖反应级数和单位，
        跨越可能达 10 个数量级），返回 (None, None)，必须由用户填写。

    Returns
    -------
    (lower, upper, reason)：lower/upper 为 None 表示无法确定，
    调用方应将该变量标记为 confidence="low" 并要求用户手动填写。
    """
    iv = v.current_value

    if v.param_type == "activation_energy":
        if iv is not None and iv > 0:
            lo, hi = iv / 1.5, iv * 1.5
            return lo, hi, f"以初值 {iv:g} 为中心，活化能对速率呈指数影响，取保守倍数窗口 [/1.5, ×1.5]"
        return (
            _EA_ABS_LO_KJ, _EA_ABS_HI_KJ,
            f"无初值，退化为化学反应活化能典型物理范围 [{_EA_ABS_LO_KJ:.0f}, {_EA_ABS_HI_KJ:.0f}]"
            "（假定单位为 kJ/mol，请核对 unit 字段并按需换算）",
        )

    # param_type == "pre_exponential_factor"
    if iv is not None and iv > 0:
        lo, hi = iv / 10.0, iv * 10.0
        return lo, hi, f"以初值 {iv:g} 为中心，频率因子跨数量级敏感，取较宽倍数窗口 [/10, ×10]"
    return (
        None, None,
        "无初值，频率因子 k0 依赖反应级数和单位，跨越量级过大，无法给出有意义的绝对范围，"
        "请提供初值或实验数据用于拟合",
    )


def heuristic_kinetics_recommendation(v: KineticsVarMeta) -> KineticsRecommendation:
    """把 heuristic_kinetics_bounds 的结果包装为 KineticsRecommendation。"""
    lo, hi, reason = heuristic_kinetics_bounds(v)
    warnings: list[str] = []
    confidence = "medium" if (lo is not None and v.current_value is not None) else "low"
    if lo is None:
        warnings.append(f"变量 {v.name} 无法确定边界，需用户手动填写或提供实验数据")
    return KineticsRecommendation(
        name=v.name, param_type=v.param_type, source="heuristic",
        initial_value=v.current_value, suggested_lower=lo, suggested_upper=hi,
        confidence=confidence, reason=reason, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 渲染给 LLM 的变量清单
# ---------------------------------------------------------------------------

def build_kinetics_variables_block(variables: list[KineticsVarMeta]) -> str:
    """把动力学参数元信息渲染成逐行文本清单，供 LLM prompt 注入。"""
    lines: list[str] = []
    for v in variables:
        iv = "未知" if v.current_value is None else f"{v.current_value:g}"
        unit = v.unit or "-"
        order = "未知" if v.reaction_order is None else f"{v.reaction_order:g}"
        t_range = ""
        if v.temperature_range is not None:
            t_lo, t_hi = v.temperature_range
            t_range = f", 操作温度范围=[{t_lo:g}, {t_hi:g}] K"
        lines.append(
            f"- name={v.name}, param_type={v.param_type}, current_value={iv}, "
            f"unit={unit}, reaction_order={order}{t_range}"
        )
    return "\n".join(lines) if lines else "(无变量)"


# ---------------------------------------------------------------------------
# 解析 LLM JSON（结构与 boundary_advisor 一致：稳健容错）
# ---------------------------------------------------------------------------

def parse_llm_kinetics_json(text: str) -> dict | None:
    """稳健解析 LLM 返回的动力学边界推荐 JSON。

    容忍 markdown 代码围栏包裹、前后散文。解析失败或缺少 'variables'
    字段时返回 None，由上层降级到 heuristic_kinetics_bounds。
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        lo = cleaned.find("{")
        hi = cleaned.rfind("}")
        if lo != -1 and hi != -1 and hi > lo:
            cleaned = cleaned[lo:hi + 1]

    try:
        obj = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        _log.warning("kinetics_advisor：LLM JSON 解析失败：%s", exc)
        return None

    if not isinstance(obj, dict) or "variables" not in obj:
        _log.warning("kinetics_advisor：LLM JSON 缺少 'variables' 字段。")
        return None
    return obj


# ---------------------------------------------------------------------------
# 报告格式化
# ---------------------------------------------------------------------------

def format_fit_result(result: KineticsFitResult) -> str:
    """把 KineticsFitResult 渲染成可读文本，供日志/CLI/HITL 展示。"""
    lines = ["=== 动力学参数拟合结果 ===", ""]
    if result.ea is None or result.k0 is None:
        lines.append("拟合失败。")
        for w in result.warnings:
            lines.append(f"  - {w}")
        return "\n".join(lines)

    lines.append(f"方法：{'非线性精修' if result.method == 'nonlinear' else '仅线性化'}")
    lines.append(f"数据点数：{result.n_points}")
    lines.append(f"拟合质量：{result.fit_quality}（R²={result.r_squared:.4f}）"
                 if result.r_squared is not None else f"拟合质量：{result.fit_quality}")
    ea_kj = result.ea / 1000.0
    lines.append(f"活化能 Ea = {ea_kj:.2f} kJ/mol" +
                 (f" ± {result.ea_stderr / 1000.0:.2f}" if result.ea_stderr is not None else ""))
    lines.append(f"频率因子 k0 = {result.k0:.4g}" +
                 (f" ± {result.k0_stderr:.4g}" if result.k0_stderr is not None else ""))
    if result.warnings:
        lines.append("")
        lines.append("警告：")
        for w in result.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def format_recommendation(rec: KineticsRecommendation) -> str:
    """把 KineticsRecommendation 渲染成可读文本。"""
    lo = "?" if rec.suggested_lower is None else f"{rec.suggested_lower:.4g}"
    hi = "?" if rec.suggested_upper is None else f"{rec.suggested_upper:.4g}"
    lines = [
        f"变量：{rec.name}（{rec.param_type}）",
        f"  来源：{rec.source}　置信度：{rec.confidence}",
        f"  建议边界：[{lo}, {hi}]",
        f"  说明：{rec.reason}",
    ]
    if rec.warnings:
        for w in rec.warnings:
            lines.append(f"  警告：{w}")
    return "\n".join(lines)
