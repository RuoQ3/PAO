"""Structured intent classification for the single PAO LeadAgent entry."""
from __future__ import annotations

import json
import re
import logging
from typing import Any, Callable

from src.agents.orchestration.models import (
    Classification,
    IntentSchema,
    SubagentRequest,
)
from . import llm_client
from .intent_response_adapter import IntentResponseAdapter


_PROJECT_HINTS = (
    "aspen", "simulation", "simulate", "optimization", "optimize", "pareto",
    "process", "equipment", "stream", "unit", "model", "case", "experiment",
    "塔", "设备", "装置", "物流", "流程", "工艺", "仿真", "模拟", "优化", "边界",
    "动力学", "参数", "结果", "模型", "实验", "方案", "刚才", "继续", "保存", "fail", "tower", "impact", "credible",
)
_ACTION_HINTS = ("modify", "change", "write", "set ", "run ", "execute", "save", "修改", "改变", "写入", "运行", "执行", "保存")


_CHINESE_PROJECT_HINTS = tuple([chr(39033)+chr(30446),chr(35774)+chr(22791),chr(35013)+chr(32622),chr(29289)+chr(27969),chr(27969)+chr(32929),chr(29289)+chr(26009),chr(24037)+chr(33402),chr(20316)+chr(29992),chr(21151)+chr(33021),chr(21407)+chr(29702),chr(31934)+chr(39311),chr(21453)+chr(24212)+chr(22120),chr(27969)+chr(31243)])
_EQUIPMENT_ID_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,4}\d{3,4}(?:-[A-Za-z0-9]+)?)(?![A-Za-z0-9])")
_FOLLOW_UP_HINTS = tuple([chr(0x6240)+chr(0x4ee5), chr(0x7136)+chr(0x540e), chr(0x8fd9)+chr(0x610f)+chr(0x5473), chr(0x8fd9)+chr(0x8bf4)+chr(0x660e), chr(0x80fd)+chr(0x5206)+chr(0x6790), chr(0x600e)+chr(0x4e48)+chr(0x770b)]) + ("so what", "what does that mean")
_ANALYSIS_HINTS = ("analysis", "analyze", "impact", chr(0x5206)+chr(0x6790), chr(0x4f5c)+chr(0x7528), chr(0x610f)+chr(0x4e49), chr(0x539f)+chr(0x56e0), chr(0x5f71)+chr(0x54cd), chr(0x8bc4)+chr(0x4f30), chr(0x5224)+chr(0x65ad))
_IMPACT_HINTS = ("impact", "sensitivity", "parameter", "parameters", "reflux", "purity", "mole fraction", "影响", "参数", "回流比", "纯度", "摩尔分数", "富集", "权衡", chr(0x5f71)+chr(0x54cd), chr(0x53c2)+chr(0x6570), chr(0x56de)+chr(0x6d41), chr(0x7eaf)+chr(0x5ea6), chr(0x5bcc)+chr(0x96c6), chr(0x6743)+chr(0x8861))
_PARAMETER_SPECIFIC_HINTS = ("sensitivity", "parameter", "parameters", "reflux", "purity", "mole fraction", "参数", "回流比", "纯度", "摩尔分数", "富集", "权衡", chr(0x53c2)+chr(0x6570), chr(0x56de)+chr(0x6d41), chr(0x7eaf)+chr(0x5ea6), chr(0x5bcc)+chr(0x96c6), chr(0x6743)+chr(0x8861))


class IntentClassifier:
    """Classify every turn through strict JSON, with explicit rule fallback."""

    def __init__(self, *, llm_config: Any = None, llm_call: Callable[..., str] | None = None,
                 confidence_threshold: float = 0.55, logger: Any = None, enabled: bool = True):
        self.llm_config = llm_config
        self.llm_call = llm_call or llm_client.chat_json
        self.response_adapter = IntentResponseAdapter()
        self.confidence_threshold = float(confidence_threshold)
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = enabled

    def classify(self, question: str, *, session: Any = None) -> Classification:
        attempts = []

        def finish(result, retries=0):
            result.retry_count = retries
            if session is not None:
                session.intent = result.intent
                session.classification_source = result.classification_source
                session.fallback_reason = result.fallback_reason
                session.classification_retry_count = retries
                session.audit_events.append({
                    "event": "intent_classification_trace", "classification_source": result.classification_source,
                    "fallback_reason": result.fallback_reason, "retry_count": retries, "attempts": attempts,
                })
            self.logger.debug("intent classification trace: %s", {
                "source": result.classification_source, "fallback_reason": result.fallback_reason,
                "retry_count": retries, "attempts": attempts})
            return result

        def fallback(reason, retries=0):
            return finish(self._fallback(question or "", reason, session=session), retries)

        if not isinstance(question, str) or not question.strip():
            return fallback("empty_message")
        if not self.enabled:
            return fallback("llm_disabled")
        try:
            cfg = self.llm_config or llm_client.load_llm_config()
            if not llm_client.is_configured(cfg):
                return fallback("llm_key_missing")
        except Exception as exc:
            self.logger.exception("intent classifier configuration failed")
            return fallback(self._reason_code(exc))

        focus = ((getattr(session, "runtime_state", {}) or {}).get("conversation_focus") or {})
        request = json.dumps({"question": question, "conversation_focus": focus}, ensure_ascii=False)
        for attempt in range(2):
            trace = {"attempt": attempt, "request": request}
            attempts.append(trace)
            try:
                raw = self.llm_call(cfg, system=self._system_prompt(), user=request)
            except Exception as exc:
                trace["error"] = {"code": self._reason_code(exc), "message": str(exc)}
                self.logger.exception("intent provider request failed (attempt=%s)", attempt)
                return fallback(self._reason_code(exc), attempt)
            trace["raw_response"] = raw
            self.logger.debug("intent raw response attempt=%s: %r", attempt, raw)
            try:
                intent = self.response_adapter.parse(raw)
            except (ValueError, TypeError, KeyError) as exc:
                error = {"code": self._reason_code(exc), "message": str(exc)}
                trace["error"] = error
                self.logger.warning("intent validation failed attempt=%s: %s", attempt, error)
                if attempt == 1:
                    return fallback(error["code"], 1)
                request = self.response_adapter.build_repair_request(
                    question=question, focus=focus, raw=raw, error=error)
                self.logger.debug("intent repair request: %s", request)
                continue
            if intent.confidence < self.confidence_threshold:
                trace["error"] = {"code": "low_confidence", "message": "confidence below threshold"}
                return fallback("low_confidence", attempt)
            return finish(Classification(intent=intent,
                classification_source="llm_repaired" if attempt else "llm"), attempt)
        raise AssertionError("unreachable classification state")

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Classify the PAO user message. Return one JSON object only, with exactly these fields: "
            "intent, task_type, project_scope, context_required, subagents, action_type, "
            "requires_confirmation, confidence, missing_information, reason, analysis. "
            "analysis MUST be an object (never string, array or null); ordinary intents return analysis: {}. "
            "context_required and missing_information are arrays of strings; requires_confirmation is boolean; "
            "confidence is a finite number between 0 and 1; reason is a string. "
            "subagents is a list of {name,task,priority,parallel}. "
            "Allowed intent: chat, project_query, analysis, parameter_impact, modify_request, simulation_request, "
            "optimization_request, report_request, code_request, general_question. "
            "Allowed project_scope: none, current_project, current_case, current_session, external. "
            "Allowed action_type: none, read, modify_file, set_parameter, simulate_once, optimize, modify_structure. "
            "Any modify, simulation, optimization, parameter impact or structure action requires_confirmation=true. "
            "For parameter_impact, put slots in analysis: equipment, target_stream, target_component, metric, candidate_parameters. "
            "All five analysis keys are required for parameter_impact; candidate_parameters is an array of strings. "
            "For unresolved scalar slots use null and list their key in missing_information; never invent values. "
            "Input may be a repair_intent_json request containing an invalid response and validation errors. "
            "Return only a corrected COMPLETE JSON object, no explanation and no Markdown. "
            'Example JSON for a greeting: {"intent":"chat","task_type":"chat",'
            '"project_scope":"none","context_required":[],"subagents":[],"action_type":"none",'
            '"requires_confirmation":false,"confidence":0.95,"missing_information":[], '
            '"reason":"The user greets the assistant.","analysis":{}}'
        )

    @staticmethod
    def _reason_code(exc: Exception) -> str:
        if str(exc) == "empty_model_result":
            return "empty_model_result"
        if isinstance(exc, json.JSONDecodeError):
            return "invalid_json"
        if isinstance(exc, (ValueError, TypeError)):
            return "schema_validation_failed"
        if isinstance(exc, TimeoutError):
            return "request_timeout"
        return "llm_api_failure"

    def _fallback(self, question: str, reason: str, *, session: Any = None) -> Classification:
        q = question.casefold()
        project = self._is_project_related(q, session)
        action = self._action_type(q)
        intent = self._intent(q, project, action)
        analysis = self._impact_slots(question, session) if intent == "parameter_impact" else {}
        if intent == "parameter_impact":
            action = "simulate_once"
        confirmation = action in {"modify_file", "set_parameter", "simulate_once", "optimize", "modify_structure"}
        context = ["process_context"] if project else ["none"]
        subagents = self._subagents(intent, q)
        schema = IntentSchema(
            intent=intent,
            task_type=self._task_type(intent),
            project_scope="current_session" if project and session is not None else ("current_project" if project else "none"),
            context_required=context,
            subagents=subagents,
            action_type=action,
            requires_confirmation=confirmation,
            confidence=0.35 if project else 0.2,
            missing_information=self._impact_missing(analysis) if intent == "parameter_impact" else [],
            reason=f"rule fallback: {reason}",
            analysis=analysis,
        )
        result = Classification(intent=schema, classification_source="rule_fallback", fallback_reason=reason)
        return result

    @staticmethod
    def _is_project_related(q: str, session: Any) -> bool:
        if not q.strip(" ?!.,;:???"):
            return False
        if any(h in q for h in _PROJECT_HINTS + _CHINESE_PROJECT_HINTS) or _EQUIPMENT_ID_RE.search(q) or (re.search(r"(?<!\d)\d{4}(?!\d)", q) and any(x in q for x in ("\u6d41\u80a1", "\u7269\u6d41", "\u7269\u6599", "stream"))):
            return True
        # ConversationSession appends the current user turn before invoking
        # the classifier.  Do not let that single entry make an ordinary
        # greeting look like an implicit project follow-up.  Only prior turns
        # containing PAO terms, or explicit context references, carry scope
        # into an otherwise implicit question.
        history = getattr(session, "history", None) if session is not None else None
        prior_history = list(history[:-1]) if isinstance(history, list) and history else []
        prior_text = " ".join(
            str(item.get("content", "")) for item in prior_history
            if isinstance(item, dict)
        ).casefold()
        refs = getattr(session, "context_refs", None) if session is not None else None
        return bool(refs) or any(h in prior_text for h in _PROJECT_HINTS + _CHINESE_PROJECT_HINTS) or bool(_EQUIPMENT_ID_RE.search(prior_text)) or any(h in q for h in _FOLLOW_UP_HINTS)

    @staticmethod
    def _action_type(q: str) -> str:
        if any(x in q for x in ("simulate", "simulation", "仿真", "模拟")):
            return "simulate_once"
        if any(x in q for x in ("optimize", "optimization", "优化", "pareto")) and any(x in q for x in _ACTION_HINTS):
            return "optimize"
        if any(x in q for x in ("boundary", "边界")) and any(x in q for x in ("change", "modify", "set ", "修改", "改变")):
            return "modify_structure"
        if any(x in q for x in ("modify", "change", "write", "set ", "save", "修改", "改变", "写入", "保存")):
            return "modify_file"
        return "read"

    @staticmethod
    def _intent(q: str, project: bool, action: str) -> str:
        # Impact requests have their own slot schema and must be routed before
        # generic analysis, including short follow-ups with implicit context.
        if any(x in q for x in _PARAMETER_SPECIFIC_HINTS + ("影响", "参数", "富集", "纯度", "回收率", "能耗", "再沸器负荷")) and (project or action in {"read", "simulate_once"}) and not any(x in q for x in ("改大", "改小", "调高", "调低")):
            return "parameter_impact"
        if "刚才" in q and "参数" in q and "影响" in q and not any(x in q for x in ("目标组分", "目标物流", "富集")):
            return "analysis"
        if action in {"modify_file", "set_parameter", "modify_structure"}:
            return "modify_request"
        if action == "simulate_once":
            return "simulation_request"
        if action == "optimize":
            return "optimization_request"
        if any(x in q for x in ("report", "报告")):
            return "report_request"
        if any(x in q for x in ("code", "代码", "源码")):
            return "code_request"
        if project:
            return "analysis" if any(x in q for x in _ANALYSIS_HINTS + _FOLLOW_UP_HINTS) else "project_query"
        return "chat" if len(q.split()) < 12 else "general_question"

    @staticmethod
    def _impact_slots(question: str, session: Any = None) -> dict[str, Any]:
        """Extract slots and inherit only structured conversation focus."""
        q = question.strip()
        focus = dict(((getattr(session, "runtime_state", {}) or {}).get("conversation_focus") or {}) if session is not None else {})
        ids = [m.group(1) for m in _EQUIPMENT_ID_RE.finditer(q)]
        streams = re.findall(r"(?<![A-Za-z0-9])\d{4}(?![A-Za-z0-9])", q)
        components = re.findall(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9_-]{1,15})(?![A-Za-z0-9])", q)
        components = [x for x in components if x != "ASPEN" and not re.match(r"^[A-Z]{1,4}\d{3,4}$", x)]
        low = q.casefold()
        if any(x in low for x in ("mole fraction", "molar fraction", "enrichment", "富集", "摩尔分数")): metric = "mole_fraction"
        elif any(x in low for x in ("purity", "纯度")): metric = "purity"
        elif any(x in low for x in ("recovery", "回收率")): metric = "recovery"
        elif any(x in low for x in ("energy", "能耗")): metric = "energy"
        elif any(x in low for x in ("duty", "再沸器负荷", "冷凝器负荷", "热负荷")): metric = "duty"
        else: metric = None
        return {"equipment": ids[-1] if ids else focus.get("equipment"),
                "target_stream": streams[-1] if streams else focus.get("stream"),
                "target_component": components[0] if components else focus.get("component"),
                "metric": metric or focus.get("metric"), "candidate_parameters": []}


    @staticmethod
    def _impact_missing(slots: dict[str, Any]) -> list[str]:
        prompts = {"equipment": "请提供目标设备名称。", "target_stream": "请提供目标物流。", "target_component": "请提供目标组分。", "metric": "请提供评价指标（如 mole_fraction、purity、energy）。"}
        return [prompts[k] for k, v in slots.items() if k in prompts and not v]

    @staticmethod
    def _task_type(intent: str) -> str:
        return {
            "chat": "chat", "project_query": "project_query", "analysis": "analysis",
            "modify_request": "modify_request", "simulation_request": "simulation_request",
            "optimization_request": "optimization_request", "report_request": "report_request",
            "code_request": "code_request", "general_question": "general_question",
            "parameter_impact": "parameter_impact",
        }[intent]

    @staticmethod
    def _subagents(intent: str, q: str) -> list[SubagentRequest]:
        if intent in {"chat", "general_question"}:
            return []
        if intent in {"optimization_request", "parameter_impact"}:
            return [SubagentRequest("optimization_advisor", "Review objectives, constraints, budget and history.")]
        if "边界" in q or "boundary" in q:
            return [SubagentRequest("boundary_advisor", "Review soft search bounds against PAO hard bounds.")]
        if "动力学" in q or "kinetic" in q:
            return [SubagentRequest("kinetics_advisor", "Analyze kinetics data and parameter evidence.")]
        if intent in {"project_query", "analysis"}:
            return [SubagentRequest("process_advisor", "Analyze the read-only PAO process context.")]
        if intent == "code_request":
            return [SubagentRequest("code_engineer", "Inspect the requested code concern without writing files.")]
        if intent == "report_request":
            return [SubagentRequest("report_writer", "Prepare a report from verified evidence.")]
        return [SubagentRequest("general_agent", "Answer the general question.")]

