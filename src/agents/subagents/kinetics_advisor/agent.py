"""Verified kinetics reasoning and explicit numerical fitting tools."""
from __future__ import annotations
from src.agents.errors import unavailable, fail_closed, failure_from
from src.agents.reasoning import configured, redact, reason

import logging
from dataclasses import dataclass

from src.agents.subagents.kinetics_advisor.fitting import fit_arrhenius
from src.agents.subagents.kinetics_advisor.plan import build_fit_plan
from src.agents.subagents.kinetics_advisor.prompts import (
    BOUNDS_SYSTEM_PROMPT,
    BOUNDS_USER_TEMPLATE,
    FIT_REVIEW_SYSTEM_PROMPT,
    FIT_REVIEW_USER_TEMPLATE,
)
from src.agents.subagents.kinetics_advisor.tools import (
    build_kinetics_variables_block,
    format_fit_result,
)
from src.models.kinetics import (
    KineticsDataPoint,
    KineticsFitPlan,
    KineticsFitResult,
    KineticsRecommendation,
    KineticsVarMeta,
)

# 模块级 import：支持测试 monkeypatch（与 boundary_advisor/process_advisor 一致）
from src.agents.lead.llm_client import chat, is_configured, load_llm_config  # noqa: E402

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------

@dataclass
class KineticsBoundsReport:
    """冷启动边界推荐的完整输出（多个变量）。"""
    recommendations: list[KineticsRecommendation]
    used_llm: bool
    global_notes: str
    warnings: list[str]


@dataclass
class KineticsFitOutcome:
    """fit_kinetics_params 的完整输出：拟合计划 + 数值结果 + 可选 LLM 解读。"""
    plan: KineticsFitPlan
    fit_result: KineticsFitResult | None
    recommendation: KineticsRecommendation | None
    llm_review: str          # LLM 对拟合结果的自然语言解读；未调用/失败时为 ""
    used_llm: bool


# ---------------------------------------------------------------------------
# 入口1：冷启动边界推荐 — 纯规则层
# ---------------------------------------------------------------------------

def recommend_kinetics_bounds(variables):
    return recommend_kinetics_bounds_agent(variables)


@fail_closed("kinetics_bounds")
def recommend_kinetics_bounds_agent(variables, context="", model=None, provider=None, llm_config=None):
    import json, math
    cfg = configured(llm_config or load_llm_config(provider=provider, model=model))
    if not variables:
        raise unavailable("agent_output_invalid", stage="kinetics_input")
    data = json.loads(chat(cfg, system=BOUNDS_SYSTEM_PROMPT,
                          user=BOUNDS_USER_TEMPLATE.format(context=context, variables_block=build_kinetics_variables_block(variables))))
    data = redact(data, cfg)
    rows = data["variables"]
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows) or not isinstance(data.get("global_notes"), str):
        raise unavailable("llm_schema_invalid", stage="kinetics_output")
    if len(rows) != len(variables) or {r["name"] for r in rows} != {v.name for v in variables}:
        raise unavailable("agent_output_invalid", stage="kinetics_coverage")
    by_name = {r["name"]: r for r in rows}
    recs = []
    for v in variables:
        row = by_name[v.name]
        lo, hi = row["lower"], row["upper"]
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise unavailable("llm_schema_invalid", stage="kinetics_output")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (lo, hi)) or lo >= hi or row["confidence"] not in {"high", "medium"}:
            raise unavailable("agent_output_invalid", stage="kinetics_output")
        recs.append(KineticsRecommendation(name=v.name, param_type=v.param_type, source="llm",
                    initial_value=v.current_value, suggested_lower=lo, suggested_upper=hi,
                    confidence=row["confidence"], reason=row["reason"]))
    return KineticsBoundsReport(recommendations=recs, used_llm=True,
                               global_notes=data.get("global_notes", ""), warnings=[])


def fit_kinetics_params(
    data_points: list[KineticsDataPoint],
    var_name: str = "kinetics_param",
    context: str = "",
    use_llm_review: bool = True,
    model: str | None = None,
    provider: str | None = None,
    llm_config=None,
) -> KineticsFitOutcome:
    """
    从用户提供的 (温度, 速率常数) 数据拟合 Arrhenius 参数。

    流程（deerflow 式 plan → execute → review）：
      1. build_fit_plan：校验数据量/温度分布，can_fit=False 时直接拒绝，
         不进入数值拟合。
      2. fit_arrhenius：确定性数值拟合（本函数唯一的数值计算来源）。
      3. （可选）LLM 只对已算好的数字做自然语言解读，不修改任何数值。

    Parameters
    ----------
    data_points:
        实验观测点列表。
    var_name:
        变量标识符，用于构建返回的 KineticsRecommendation.name。
    context:
        工艺背景文字，注入 LLM 解读 prompt（可选）。
    use_llm_review:
        是否调用 LLM 解读拟合结果；False 时跳过（节省调用），
        recommendation 仍会正常生成。
    model / provider / llm_config:
        覆盖 LLM 配置，语义与 boundary_advisor 一致。

    Returns
    -------
    KineticsFitOutcome
        plan.can_fit=False 时，fit_result / recommendation 均为 None，
        调用方应展示 plan.rejection_reason 给用户，不应继续处理。
    """
    plan = build_fit_plan(data_points)

    if not plan.can_fit:
        return KineticsFitOutcome(
            plan=plan, fit_result=None, recommendation=None,
            llm_review="", used_llm=False,
        )

    fit_result = fit_arrhenius(data_points)

    recommendation = _build_recommendation_from_fit(var_name, fit_result)

    llm_review = ""
    used_llm = False
    if use_llm_review and fit_result.ea is not None:
        llm_review, used_llm = _review_fit_with_llm(
            fit_result, context=context, model=model, provider=provider, llm_config=llm_config,
        )

    return KineticsFitOutcome(
        plan=plan, fit_result=fit_result, recommendation=recommendation,
        llm_review=llm_review, used_llm=used_llm,
    )


def _build_recommendation_from_fit(
    var_name: str, fit_result: KineticsFitResult,
) -> KineticsRecommendation:
    """把 KineticsFitResult 包装为 KineticsRecommendation（activation_energy 变量）。

    只对 Ea 生成边界建议（以拟合值为中心，按拟合质量决定倍数窗口）；
    调用方若也需要 k0 的推荐，可用 fit_result.k0 自行构造，本函数聚焦 Ea
    因其是决定优化搜索空间的主要敏感参数。
    """
    if fit_result.ea is None:
        return KineticsRecommendation(
            name=var_name, param_type="activation_energy", source="fit",
            initial_value=None, suggested_lower=None, suggested_upper=None,
            confidence="low", reason="拟合失败，无法给出推荐值",
            fit_result=fit_result, warnings=list(fit_result.warnings),
        )

    # 拟合质量越差，边界窗口越宽（给优化器更多容错空间去修正不确定的估计）
    window = {"good": 1.2, "marginal": 1.5, "poor": 2.0}[fit_result.fit_quality]
    confidence = {"good": "high", "marginal": "medium", "poor": "low"}[fit_result.fit_quality]

    return KineticsRecommendation(
        name=var_name, param_type="activation_energy", source="fit",
        initial_value=fit_result.ea,
        suggested_lower=fit_result.ea / window,
        suggested_upper=fit_result.ea * window,
        confidence=confidence,
        reason=f"由 {fit_result.n_points} 个实验数据点拟合（{fit_result.method}），"
               f"拟合质量={fit_result.fit_quality}",
        fit_result=fit_result,
        warnings=list(fit_result.warnings),
    )


def _review_fit_with_llm(
    fit_result: KineticsFitResult,
    context: str,
    model: str | None,
    provider: str | None,
    llm_config,
) -> tuple[str, bool]:
    """调用 LLM 对已算好的拟合结果做自然语言解读；失败时立即终止。"""
    cfg = llm_config if llm_config is not None else load_llm_config(provider=provider, model=model)

    configured(cfg)

    ea_stderr_str = "未知" if fit_result.ea_stderr is None else f"{fit_result.ea_stderr / 1000.0:.3g} kJ/mol"
    k0_stderr_str = "未知" if fit_result.k0_stderr is None else f"{fit_result.k0_stderr:.3g}"
    r2_str = "未知" if fit_result.r_squared is None else f"{fit_result.r_squared:.4f}"
    warnings_block = "\n".join(f"- {w}" for w in fit_result.warnings) if fit_result.warnings else "(无)"

    user = FIT_REVIEW_USER_TEMPLATE.format(
        ea_kj=fit_result.ea / 1000.0, ea_stderr=ea_stderr_str,
        k0=fit_result.k0, k0_stderr=k0_stderr_str,
        r_squared=r2_str, n_points=fit_result.n_points,
        method=fit_result.method, fit_quality=fit_result.fit_quality,
        warnings_block=warnings_block,
    )
    if context:
        user = f"工艺背景：{context}\n\n{user}"

    try:
        data, _ = reason(user, context=fit_result, role="kinetics_fit_review", llm_config=cfg, llm_call=chat)
        return data["text"], True
    except Exception as exc:
        raise failure_from(exc, "kinetics_review") from None
