"""
intent.py — Coordinator agent 的意图分类层。

分两层（与 boundary_advisor / kinetics_advisor 一致的稳健降级风格）：
  classify_intent()       — 纯规则兜底（keyword_classify），不依赖 LLM。
  classify_intent_agent() — LLM 层：调用大模型拆解意图，失败时降级到规则兜底。

安全边界：
  - 本模块只做分类，不执行任何子 agent、不驱动 Aspen、不写数据库。
  - LLM 不可用（缺 key/调用失败/JSON 解析失败）时，降级到 keyword_classify，
    绝不返回空结果或抛异常给上层。
"""
from __future__ import annotations

import logging

from src.agents.coordinator.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.agents.coordinator.tools import (
    build_steps_from_parsed,
    keyword_classify,
    parse_llm_intent_json,
)
from src.models.coordinator import IntentClassification, IntentStep, IntentType

# 模块级 import：支持测试 monkeypatch（与 boundary_advisor/kinetics_advisor 一致）
from src.agents.llm_client import chat, is_configured, load_llm_config  # noqa: E402

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 纯规则层
# ---------------------------------------------------------------------------

def classify_intent(user_text: str) -> IntentClassification:
    """纯规则兜底：按关键词匹配分类。任何环境都能跑（无网络/无 key）。"""
    return keyword_classify(user_text)


# ---------------------------------------------------------------------------
# LLM 层
# ---------------------------------------------------------------------------

def classify_intent_agent(
    user_text: str,
    session_context: str = "",
    model: str | None = None,
    provider: str | None = None,
    llm_config=None,
) -> IntentClassification:
    """调用大模型拆解用户意图；失败时降级到规则兜底。

    Args:
        user_text:       用户的自由文本请求。
        session_context: 连续对话模式下的会话上下文摘要（见
            ChatSessionState.to_context_summary），告知 LLM 当前会话已有
            哪些文件路径，使"这个/这个工艺"等指代词能被正确理解，并让
            LLM 不必因为看不到显式路径就把请求判为信息不足。一次性调用
            （run_coordinator）不传此参数，行为与之前完全一致。
        model:      覆盖 PAO_LLM_MODEL。
        provider:   覆盖 PAO_LLM_PROVIDER。
        llm_config: 直接注入 LLMConfig（测试用）。

    Returns:
        IntentClassification。used_llm 标识是否真的用上了大模型。
        steps 永远非空——分类失败/无法识别时含一个 CLARIFY_NEEDED 步骤。
    """
    if not user_text or not user_text.strip():
        return IntentClassification(
            steps=[IntentStep(
                intent_type=IntentType.CLARIFY_NEEDED,
                reason="用户请求为空，无法分类",
            )],
            notes="", used_llm=False,
        )

    cfg = llm_config if llm_config is not None else load_llm_config(provider=provider, model=model)

    if not is_configured(cfg):
        _log.info("coordinator：未配置 LLM key，使用规则兜底分类意图。")
        result = classify_intent(user_text)
        result.warnings.append(f"未配置大模型 API key（请设置 {cfg.api_key_env}），本次为规则兜底结果。")
        return result

    user = USER_TEMPLATE.format(
        user_text=user_text,
        session_context=session_context or "（无，非连续对话模式或会话刚开始）",
    )

    parsed = None
    llm_warnings: list[str] = []
    try:
        raw = chat(cfg, system=SYSTEM_PROMPT, user=user)
        parsed = parse_llm_intent_json(raw)
        if parsed is None:
            llm_warnings.append("大模型返回无法解析为合法 JSON，已降级到规则兜底。")
    except Exception as exc:  # noqa: BLE001
        _log.warning("coordinator：LLM 调用失败，降级规则兜底：%s", exc)
        llm_warnings.append(f"大模型调用失败（{exc}），已降级到规则兜底。")

    if parsed is None:
        result = classify_intent(user_text)
        result.warnings.extend(llm_warnings)
        return result

    steps, build_warnings = build_steps_from_parsed(parsed)
    warnings = llm_warnings + build_warnings

    if not steps:
        # LLM 返回的 JSON 结构合法但所有步骤都非法/未知，降级到规则兜底
        result = classify_intent(user_text)
        result.warnings.extend(warnings)
        result.warnings.append("LLM 返回的所有步骤均无法识别，已降级到规则兜底。")
        return result

    notes = str(parsed.get("notes", ""))[:300]
    return IntentClassification(steps=steps, notes=notes, used_llm=True, warnings=warnings)
