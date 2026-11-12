"""Read-only evidence formatting and explicit numerical tools; no Agent fallback."""
from __future__ import annotations

import json
import logging
import re

from src.models.kinetics import KineticsFitResult, KineticsRecommendation, KineticsVarMeta

_log = logging.getLogger(__name__)

# 活化能典型物理范围（kJ/mol），与 fitting.py 的 _EA_TYPICAL_* 保持一致，
_EA_ABS_LO_KJ = 40.0
_EA_ABS_HI_KJ = 400.0

# 无初值时 k0 的搜索倍数窗口（以 1.0 为参考锚点在 log 空间展开）不适用，
# k0 跨越的数量级过大，无初值场景直接返回 None 边界，交给用户填写。


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------





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
