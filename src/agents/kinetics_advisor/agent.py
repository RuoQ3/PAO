"""
agent.py — KineticsAdvisor agent 核心逻辑。

两条独立入口：
  recommend_kinetics_bounds()       — 冷启动边界推荐，纯规则兜底，不依赖 LLM。
  recommend_kinetics_bounds_agent() — 冷启动边界推荐，LLM 层，失败降级规则兜底。
  fit_kinetics_params()             — 从用户实验数据拟合 Ea/k0（确定性数值计算），
                                       可选调用 LLM 只做结果解读，不做任何计算。

分层原则（与 boundary_advisor / process_advisor 一致的稳健降级风格）：
  - 数值拟合（fitting.py）永远是确定性代码，不因 LLM 是否可用而改变行为。
  - LLM 不可用（缺 key/调用失败/JSON 解析失败）时，边界推荐降级到规则兜底，
    拟合结果解读降级到规则模板文本；两条路径都绝不返回空结果或抛异常给上层。

硬性约束
--------
  - 本模块不驱动 Aspen、不重跑仿真、不写数据库，只做"元信息/数据 → 推荐"的
    纯推理与纯计算。
  - fit_kinetics_params 的输出（KineticsRecommendation）不会自动写入
    ConfigDraft——写入前仍必须经过 write_feasibility_node 的 COM 试写验证
    和 human_confirm_node 的用户确认，即使拟合质量为 "good" 也不允许跳过。
    这是硬约束：数值拟合结果和其他设计变量一样，只是"建议"，不是"决定"。
  - LLM 绝不参与 fit_kinetics_params 的数值计算；FIT_REVIEW 阶段的 LLM 调用
    只产出解读文本，不影响 KineticsFitResult 本身的任何数值字段。

模型：默认走 PAO_LLM_PROVIDER（与项目其他 agent 一致，见 llm_client.py）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.agents.kinetics_advisor.fitting import fit_arrhenius
from src.agents.kinetics_advisor.plan import build_fit_plan
from src.agents.kinetics_advisor.prompts import (
    BOUNDS_SYSTEM_PROMPT,
    BOUNDS_USER_TEMPLATE,
    FIT_REVIEW_SYSTEM_PROMPT,
    FIT_REVIEW_USER_TEMPLATE,
)
from src.agents.kinetics_advisor.tools import (
    build_kinetics_variables_block,
    format_fit_result,
    heuristic_kinetics_recommendation,
    parse_llm_kinetics_json,
)
from src.models.kinetics import (
    KineticsDataPoint,
    KineticsFitPlan,
    KineticsFitResult,
    KineticsRecommendation,
    KineticsVarMeta,
)

# 模块级 import：支持测试 monkeypatch（与 boundary_advisor/process_advisor 一致）
from src.agents.llm_client import chat, is_configured, load_llm_config  # noqa: E402

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

def recommend_kinetics_bounds(variables: list[KineticsVarMeta]) -> KineticsBoundsReport:
    """纯规则兜底：为每个动力学参数推断边界。任何环境都能跑（无网络/无 key）。"""
    recs = [heuristic_kinetics_recommendation(v) for v in variables]
    warnings = [w for r in recs for w in r.warnings]
    return KineticsBoundsReport(
        recommendations=recs, used_llm=False,
        global_notes="(规则兜底模式，未调用大模型)", warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 入口1：冷启动边界推荐 — LLM 层
# ---------------------------------------------------------------------------

def recommend_kinetics_bounds_agent(
    variables: list[KineticsVarMeta],
    context: str = "",
    model: str | None = None,
    provider: str | None = None,
    llm_config=None,
) -> KineticsBoundsReport:
    """调用大模型为每个动力学参数推荐边界；失败逐变量降级到规则兜底。"""
    if not variables:
        return KineticsBoundsReport(recommendations=[], used_llm=False,
                                     global_notes="", warnings=["无变量输入。"])

    cfg = llm_config if llm_config is not None else load_llm_config(provider=provider, model=model)

    if not is_configured(cfg):
        _log.info("kinetics_advisor：未配置 LLM key，使用规则兜底生成边界。")
        rep = recommend_kinetics_bounds(variables)
        rep.warnings.append(f"未配置大模型 API key（请设置 {cfg.api_key_env}），本次为规则兜底结果。")
        return rep

    user = BOUNDS_USER_TEMPLATE.format(
        context=context or "(无)",
        variables_block=build_kinetics_variables_block(variables),
    )

    parsed = None
    llm_warnings: list[str] = []
    try:
        raw = chat(cfg, system=BOUNDS_SYSTEM_PROMPT, user=user)
        parsed = parse_llm_kinetics_json(raw)
        if parsed is None:
            llm_warnings.append("大模型返回无法解析为合法 JSON，已逐变量降级到规则兜底。")
    except Exception as exc:  # noqa: BLE001
        _log.warning("kinetics_advisor：LLM 调用失败，降级规则兜底：%s", exc)
        llm_warnings.append(f"大模型调用失败（{exc}），已降级到规则兜底。")

    if parsed is None:
        rep = recommend_kinetics_bounds(variables)
        rep.warnings.extend(llm_warnings)
        return rep

    item_map: dict[str, dict] = {
        str(it.get("name")): it
        for it in parsed.get("variables", [])
        if isinstance(it, dict) and it.get("name")
    }
    global_notes = str(parsed.get("global_notes", ""))[:300]

    recs: list[KineticsRecommendation] = []
    warnings: list[str] = list(llm_warnings)
    used_llm_any = False
    for v in variables:
        item = item_map.get(v.name)
        if item is None:
            rec = heuristic_kinetics_recommendation(v)
            warnings.append(f"变量 {v.name} 未被 LLM 覆盖，用规则兜底。")
            recs.append(rec)
            continue

        lower = item.get("lower")
        upper = item.get("upper")
        try:
            lower = float(lower) if lower is not None else None
            upper = float(upper) if upper is not None else None
        except (TypeError, ValueError):
            lower = upper = None

        if lower is None or upper is None:
            rec = heuristic_kinetics_recommendation(v)
            warnings.append(f"变量 {v.name} 的 LLM 边界不完整，用规则兜底。")
            recs.append(rec)
            continue

        confidence = item.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        reason = str(item.get("reason", "(LLM 未给 reason)"))

        recs.append(KineticsRecommendation(
            name=v.name, param_type=v.param_type, source="llm",
            initial_value=v.current_value, suggested_lower=lower, suggested_upper=upper,
            confidence=confidence, reason=reason,
        ))
        used_llm_any = True

    return KineticsBoundsReport(
        recommendations=recs, used_llm=used_llm_any,
        global_notes=global_notes, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 入口2：从用户实验数据拟合
# ---------------------------------------------------------------------------

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
    """调用 LLM 对已算好的拟合结果做自然语言解读；失败时降级返回规则模板文本。"""
    cfg = llm_config if llm_config is not None else load_llm_config(provider=provider, model=model)

    if not is_configured(cfg):
        _log.info("kinetics_advisor：未配置 LLM key，跳过拟合结果解读，返回规则格式化文本。")
        return format_fit_result(fit_result), False

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
        review = chat(cfg, system=FIT_REVIEW_SYSTEM_PROMPT, user=user)
        return review.strip(), True
    except Exception as exc:  # noqa: BLE001
        _log.warning("kinetics_advisor：LLM 拟合结果解读失败，降级规则格式化文本：%s", exc)
        return format_fit_result(fit_result), False
