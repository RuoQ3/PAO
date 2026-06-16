"""
agent.py — BoundaryAdvisor agent 核心逻辑。

职责：基于"初始收敛解 + 变量元信息",为每个设计变量推理出搜索倍数 k,
生成自适应搜索边界。这是系统化方案的"冷启动边界判断"层——把原本需要人工
凭工艺经验设定的边界,交给大模型基于物理常识自动判断。

分两层（与 process_advisor 一致的稳健降级风格）：
  recommend_boundaries()        — 纯规则兜底,不依赖 LLM,任何环境都能跑。
  recommend_boundaries_agent()  — LLM 层:调用大模型给 k,失败时降级到规则兜底。

模型：默认走 PAO_LLM_PROVIDER（用户使用 deepseek，模型 deepseek-v4-pro,
通过环境变量 PAO_LLM_PROVIDER=deepseek、PAO_LLM_MODEL=deepseek-v4-pro 指定,
或调用时显式传 provider/model 覆盖）。

安全边界：
  - 不驱动 Aspen、不重跑仿真、不写数据库,只做"元信息 → 边界"的纯推理。
  - LLM 不可用（缺 key/调用失败/JSON 解析失败）时,逐变量降级到 heuristic_k,
    绝不返回空结果或抛异常给上层。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.agents.boundary_advisor.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.agents.boundary_advisor.tools import (
    VarMeta,
    bounds_from_k,
    build_variables_block,
    extract_k_map,
    heuristic_k,
    parse_llm_boundary_json,
)

# 模块级 import：支持测试 monkeypatch
from src.agents.llm_client import chat, is_configured, load_llm_config  # noqa: E402

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------

@dataclass
class BoundaryRecommendation:
    """单个变量的边界推荐结果。"""
    name: str
    lower: float
    upper: float
    k_lo: float
    k_hi: float
    source: str          # "llm" | "heuristic" | "global_fallback"
    reason: str


@dataclass
class BoundaryReport:
    """boundary_advisor 的完整输出。"""
    recommendations: list[BoundaryRecommendation]
    used_llm: bool
    global_notes: str
    warnings: list[str]

    def as_bounds_dict(self) -> dict[str, tuple[float, float]]:
        """返回 {name: (lower, upper)},便于直接覆盖 param_bounds。"""
        return {r.name: (r.lower, r.upper) for r in self.recommendations}


# ---------------------------------------------------------------------------
# 纯规则层（不依赖 LLM）
# ---------------------------------------------------------------------------

def recommend_boundaries(variables: list[VarMeta]) -> BoundaryReport:
    """纯规则兜底：用 heuristic_k 为每个变量推断 k 并换算边界。

    任何环境都能跑（无网络、无 key）。LLM 层失败时也回退到这里。
    """
    recs: list[BoundaryRecommendation] = []
    warnings: list[str] = []
    for v in variables:
        k_lo, k_hi, reason = heuristic_k(v)
        bounds = bounds_from_k(v, k_lo, k_hi)
        if bounds is None:
            # 无法换算（缺初始值且无全局边界）：跳过,保留调用方原边界
            warnings.append(f"变量 {v.name} 无法确定边界（缺初始值且无全局范围），已跳过。")
            continue
        lo, hi = bounds
        source = "heuristic"
        if v.initial_value is None or v.initial_value <= 0:
            source = "global_fallback"
        recs.append(BoundaryRecommendation(
            name=v.name, lower=lo, upper=hi, k_lo=k_lo, k_hi=k_hi,
            source=source, reason=reason,
        ))
    return BoundaryReport(
        recommendations=recs, used_llm=False,
        global_notes="(规则兜底模式,未调用大模型)", warnings=warnings,
    )


# ---------------------------------------------------------------------------
# LLM 层
# ---------------------------------------------------------------------------

def recommend_boundaries_agent(
    variables: list[VarMeta],
    context: str = "",
    model: str | None = None,
    provider: str | None = None,
    llm_config=None,
) -> BoundaryReport:
    """调用大模型为每个变量推荐 k,生成自适应边界;失败逐变量降级到规则兜底。

    Args:
        variables:  设计变量元信息列表。
        context:    工艺/文件背景文字(可选),注入 prompt 帮助 LLM 判断工艺类型。
        model:      覆盖 PAO_LLM_MODEL(如 "deepseek-v4-pro")。
        provider:   覆盖 PAO_LLM_PROVIDER(如 "deepseek")。
        llm_config: 直接注入 LLMConfig(测试用)。

    Returns:
        BoundaryReport。used_llm 标识是否真的用上了大模型。
    """
    if not variables:
        return BoundaryReport(recommendations=[], used_llm=False,
                              global_notes="", warnings=["无变量输入。"])

    cfg = llm_config if llm_config is not None else load_llm_config(provider=provider, model=model)

    # 未配置 key → 直接走纯规则兜底
    if not is_configured(cfg):
        _log.info("boundary_advisor：未配置 LLM key,使用规则兜底生成边界。")
        rep = recommend_boundaries(variables)
        rep.warnings.append(
            f"未配置大模型 API key（请设置 {cfg.api_key_env}）,本次为规则兜底结果。"
        )
        return rep

    # 调用 LLM
    user = USER_TEMPLATE.format(
        context=context or "(无)",
        variables_block=build_variables_block(variables),
    )

    # 按变量个数动态保证足够的输出 token。
    # 每个变量 JSON 约需 300 token（name/k_lo/k_hi/inferred_type/is_vacuum/sensitivity/reason），
    # 加 800 固定开销（global_notes/JSON 大括号/缩进），再乘 1.5 安全系数。
    # 推理模型的 reasoning token 是额外消耗、不占 max_tokens 配额，无需为此加量。
    need_tokens = min(16000, int((800 + 300 * len(variables)) * 1.5))
    if cfg.max_tokens < need_tokens:
        from dataclasses import replace as _dc_replace
        _log.info(
            "boundary_advisor：%d 个变量需要约 %d 输出 token,"
            "本次调用临时从 %d 上调到 %d(不改全局配置)。",
            len(variables), need_tokens, cfg.max_tokens, need_tokens,
        )
        cfg = _dc_replace(cfg, max_tokens=need_tokens)

    parsed = None
    llm_warnings: list[str] = []
    try:
        raw = chat(cfg, system=SYSTEM_PROMPT, user=user)
        parsed = parse_llm_boundary_json(raw)
        if parsed is None:
            llm_warnings.append("大模型返回无法解析为合法 JSON,已逐变量降级到规则兜底。")
    except Exception as exc:  # noqa: BLE001
        _log.warning("boundary_advisor：LLM 调用失败,降级规则兜底：%s", exc)
        llm_warnings.append(f"大模型调用失败({exc}),已降级到规则兜底。")

    if parsed is None:
        rep = recommend_boundaries(variables)
        rep.warnings.extend(llm_warnings)
        return rep

    # 解析成功：逐变量用 LLM 的 k；缺失的变量用 heuristic 补齐
    k_map = extract_k_map(parsed)
    global_notes = str(parsed.get("global_notes", ""))[:300]
    reason_map = {
        str(it.get("name")): str(it.get("reason", ""))
        for it in parsed.get("variables", [])
        if isinstance(it, dict) and it.get("name")
    }

    recs: list[BoundaryRecommendation] = []
    warnings: list[str] = list(llm_warnings)
    used_llm_any = False
    for v in variables:
        if v.name in k_map:
            k_lo, k_hi = k_map[v.name]
            source = "llm"
            reason = reason_map.get(v.name, "(LLM 未给 reason)")
            used_llm_any = True
        else:
            k_lo, k_hi, reason = heuristic_k(v)
            source = "heuristic"
            warnings.append(f"变量 {v.name} 未被 LLM 覆盖,用规则兜底 k。")

        bounds = bounds_from_k(v, k_lo, k_hi)
        if bounds is None:
            warnings.append(f"变量 {v.name} 无法确定边界,已跳过。")
            continue
        lo, hi = bounds
        if (v.initial_value is None or v.initial_value <= 0) and source != "llm":
            source = "global_fallback"
        recs.append(BoundaryRecommendation(
            name=v.name, lower=lo, upper=hi, k_lo=k_lo, k_hi=k_hi,
            source=source, reason=reason,
        ))

    return BoundaryReport(
        recommendations=recs, used_llm=used_llm_any,
        global_notes=global_notes, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 报告格式化（人类可读）
# ---------------------------------------------------------------------------

def format_boundary_report(report: BoundaryReport) -> str:
    """把 BoundaryReport 渲染成可读文本,供日志/CLI 展示。"""
    lines = ["=== BoundaryAdvisor 边界推荐报告 ===", ""]
    lines.append(f"使用大模型：{'是' if report.used_llm else '否（规则兜底）'}")
    if report.global_notes:
        lines.append(f"整体判断：{report.global_notes}")
    lines.append("")
    lines.append(f"{'变量':<28}{'下界':>14}{'上界':>14}  来源     说明")
    for r in report.recommendations:
        short = r.name.replace("\\", "/").split("/")[-1][:26]
        lines.append(
            f"{short:<28}{r.lower:>14.4g}{r.upper:>14.4g}  {r.source:<8} {r.reason}"
        )
    if report.warnings:
        lines.append("")
        lines.append("警告：")
        for w in report.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)
