"""
tools.py — Coordinator agent 的确定性工具函数。

职责（不含任何 LLM 调用，纯计算/解析，便于单测）：
  - keyword_classify：无 LLM 时的规则兜底分类器，基于关键词匹配。
  - parse_llm_intent_json：稳健解析 LLM 返回的意图分类 JSON。
  - format_execution_report：把 ExecutionState 渲染成可读文本。

设计原则
--------
  - 所有函数对缺失字段、异常输入都返回安全默认值，绝不抛异常给上层。
  - keyword_classify 是 LLM 不可用时的退路，保证 Coordinator 在无网络/无 key
    环境下仍能给出一个可用的分类结果（哪怕只是粗略匹配），而不是完全瘫痪。
    但关键词匹配的置信度天然更低，未匹配到任何关键词时必须落到
    CLARIFY_NEEDED，不能强行归类。
"""
from __future__ import annotations

import json
import logging
import re

from src.models.coordinator import (
    ExecutionState,
    IntentClassification,
    IntentStep,
    IntentType,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 规则兜底分类（无 LLM 时使用）
# ---------------------------------------------------------------------------

# 关键词 -> IntentType，按顺序匹配（越靠前优先级越高）。
# "没有数据"类否定信号必须排在 FIT_KINETICS_PARAMS 之前：
# "没有实验数据，帮我估个活化能范围" 会同时命中"活化能"（拟合类泛词）和
# "没有实验"（否定信号），否定信号语义更明确（用户已声明手头没有数据），
# 必须优先匹配，否则会被"活化能"误判为要拟合。
_KEYWORD_RULES: list[tuple[tuple[str, ...], IntentType]] = [
    (("没有数据", "没有实验", "冷启动", "估个", "估算") , IntentType.RECOMMEND_KINETICS_BOUNDS),
    (("拟合", "速率常数", "实验数据"), IntentType.FIT_KINETICS_PARAMS),
    (("动力学",) , IntentType.RECOMMEND_KINETICS_BOUNDS),  # 泛"动力学"但无拟合/数据信号
    (("接入", "扫描", "新工艺", "新的仿真文件", "新的bkp", "新的 bkp"), IntentType.ONBOARD_NEW_CASE),
    (("边界", "搜索范围", "上下界"), IntentType.RECOMMEND_VAR_BOUNDS),
    (("为什么", "诊断", "分析结果", "失败原因", "不收敛"), IntentType.DIAGNOSE_RESULTS),
    (("开始优化", "跑优化", "启动优化", "运行优化"), IntentType.RUN_OPTIMIZATION),
]


def keyword_classify(user_text: str) -> IntentClassification:
    """无 LLM 时的规则兜底：按关键词表顺序匹配第一个命中的类型。

    只产出单一步骤（关键词匹配无法可靠拆解复合请求）。未命中任何规则时
    返回 CLARIFY_NEEDED，不强行归类。
    """
    text = user_text or ""
    for keywords, intent_type in _KEYWORD_RULES:
        if any(kw in text for kw in keywords):
            return IntentClassification(
                steps=[IntentStep(
                    intent_type=intent_type,
                    raw_slots={"context": text} if text else {},
                    confidence="low",
                    reason=f"关键词规则兜底命中：{[kw for kw in keywords if kw in text]}",
                )],
                notes="(规则兜底模式，未调用大模型，仅支持单一意图识别)",
                used_llm=False,
            )
    return IntentClassification(
        steps=[IntentStep(
            intent_type=IntentType.CLARIFY_NEEDED,
            raw_slots={"context": text} if text else {},
            confidence="low",
            reason="规则兜底未命中任何已知关键词，需用户澄清具体需求",
        )],
        notes="(规则兜底模式，未识别到已知任务类型)",
        used_llm=False,
    )


# ---------------------------------------------------------------------------
# 解析 LLM JSON
# ---------------------------------------------------------------------------

_VALID_INTENT_VALUES: set[str] = {t.value for t in IntentType}


def parse_llm_intent_json(text: str) -> dict | None:
    """稳健解析 LLM 返回的意图分类 JSON。

    容忍以下情况：
      - 回复被 ```json ... ``` 代码围栏包裹
      - JSON 前后有少量散文
    解析失败或缺少 'steps' 字段时返回 None，由上层降级到 keyword_classify。
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
        _log.warning("coordinator：LLM JSON 解析失败：%s", exc)
        return None

    if not isinstance(obj, dict) or "steps" not in obj:
        _log.warning("coordinator：LLM JSON 缺少 'steps' 字段。")
        return None
    return obj


def build_steps_from_parsed(parsed: dict) -> tuple[list[IntentStep], list[str]]:
    """把 parse_llm_intent_json 的返回值转成 IntentStep 列表。

    非法/未知的 intent_type 值会被跳过并记入 warnings（不会让整个分类失败）；
    若跳过后 steps 为空，调用方应自行决定是否降级到 CLARIFY_NEEDED。

    Returns
    -------
    (steps, warnings)
    """
    steps: list[IntentStep] = []
    warnings: list[str] = []
    for item in parsed.get("steps", []):
        if not isinstance(item, dict):
            continue
        raw_type = item.get("intent_type")
        if raw_type not in _VALID_INTENT_VALUES:
            warnings.append(f"LLM 返回未知 intent_type={raw_type!r}，已忽略该步骤")
            continue
        raw_slots = item.get("raw_slots")
        if not isinstance(raw_slots, dict):
            raw_slots = {}
        # raw_slots 的值统一转字符串，容忍 LLM 返回非字符串类型
        clean_slots = {str(k): str(v) for k, v in raw_slots.items()}
        confidence = item.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        steps.append(IntentStep(
            intent_type=IntentType(raw_type),
            raw_slots=clean_slots,
            confidence=confidence,
            reason=str(item.get("reason", "")),
        ))
    return steps, warnings


# ---------------------------------------------------------------------------
# 报告格式化
# ---------------------------------------------------------------------------

def format_execution_report(state: ExecutionState) -> str:
    """把 ExecutionState 渲染成可读文本，供 CLI 展示。"""
    lines = ["=== Coordinator 执行报告 ===", ""]
    for step in state.steps:
        status_label = {
            "ok": "[完成]", "error": "[失败]", "skipped": "[跳过]", "pending": "[待执行]",
        }.get(step.status, f"[{step.status}]")
        lines.append(f"{status_label} {step.task_id}（{step.intent_type.value}）")
        if step.status == "skipped" and step.skipped_reason:
            lines.append(f"    原因：{step.skipped_reason}")
        elif step.report:
            preview = step.report if len(step.report) <= 300 else step.report[:300] + "…"
            lines.append(f"    {preview}")
        lines.append("")
    if state.errors:
        lines.append("错误汇总：")
        for e in state.errors:
            lines.append(f"  - {e}")
    return "\n".join(lines)
