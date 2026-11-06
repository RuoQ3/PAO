from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.agents.contracts.models import (
    AdviceUncertainty,
    EvidenceItem,
    EvidenceSource,
    ProcessContext,
    TechnicalAdvice,
)
from src.integrations.ima import ImaAnswer, ImaRequest
from src.agents.process_context import ProcessContextBuilder

from .models import Clarification, QuestionIntent, QuestionType
from .prompts import IMA_QUERY_TEMPLATE


class _ImaLike(Protocol):
    def query_with_sources(self, request: ImaRequest) -> ImaAnswer: ...


@dataclass
class TechnicalAdviceResult:
    advice: TechnicalAdvice
    intent: QuestionIntent
    ima_answer: ImaAnswer | None = None
    local_facts: list[EvidenceItem] = field(default_factory=list)
    simulation_evidence: list[EvidenceItem] = field(default_factory=list)
    degraded: bool = False
    query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "advice": self.advice.to_dict(),
            "intent": {
                "question_type": self.intent.question_type.value,
                "equipment": self.intent.equipment,
                "streams": self.intent.streams,
                "variables": self.intent.variables,
                "objectives": self.intent.objectives,
                "constraints": self.intent.constraints,
                "clarifications": [c.__dict__ for c in self.intent.clarifications],
            },
            "ima_answer": None if self.ima_answer is None else {
                "answer": self.ima_answer.answer,
                "source": self.ima_answer.source,
                "confidence": self.ima_answer.confidence,
                "assumptions": list(self.ima_answer.assumptions),
                "request_id": self.ima_answer.request_id,
                "sources": [s.to_dict() for s in self.ima_answer.sources],
            },
            "local_facts": [x.to_dict() for x in self.local_facts],
            "simulation_evidence": [x.to_dict() for x in self.simulation_evidence],
            "degraded": self.degraded,
            "query": self.query,
        }


class TechnicalAdvisor:
    """Read-only technical consultant.

    The advisor never opens Aspen, writes a model, or runs a simulation.  It
    accepts a prebuilt ProcessContext and an injected IMA client.
    """

    def __init__(self, ima_client: _ImaLike | None = None, *, max_history: int = 20):
        self.ima_client = ima_client
        self.max_history = max_history

    @staticmethod
    def _known(context: ProcessContext, key: str) -> list[str]:
        if key == "equipment":
            return [str(x.get("name")) for x in context.equipment if x.get("name")]
        if key == "streams":
            return [str(x.get("name")) for x in context.streams if x.get("name")]
        if key == "variables":
            return [str(x.get("name") or k) for k, x in context.variables.items()]
        if key == "objectives":
            return [str(x.get("name")) for x in context.objectives if x.get("name")]
        if key == "constraints":
            return [str(x.get("name") or x.get("stream") or x.get("variable")) for x in context.constraints if x.get("name") or x.get("stream") or x.get("variable")]
        return []

    @staticmethod
    def _mentioned(question: str, values: list[str]) -> list[str]:
        q = question.lower()
        return [v for v in values if v.lower() in q]

    def classify_question(self, question: str, context: ProcessContext) -> QuestionIntent:
        q = question.strip()
        low = q.lower()
        eq = self._mentioned(q, self._known(context, "equipment"))
        streams = self._mentioned(q, self._known(context, "streams"))
        variables = self._mentioned(q, self._known(context, "variables"))
        objectives = self._mentioned(q, self._known(context, "objectives"))
        constraints = self._mentioned(q, self._known(context, "constraints"))
        aliases = {
            "回流": ("reflux", "reflux_ratio"),
            "温度": ("temperature", "temp"),
            "压力": ("pressure", "press"),
            "流量": ("flow", "flowrate", "rate"),
        }
        for term, needles in aliases.items():
            if term in q:
                variables.extend(v for v in self._known(context, "variables")
                                 if any(n in v.lower() for n in needles))
        # Also recognize common Aspen identifiers when the context is partial.
        for match in re.finditer(r"\\Data\\(Blocks|Streams)\\([^\\\s,，。?？]+)", q, flags=re.I):
            (streams if match.group(1).lower() == "streams" else eq).append(match.group(2))
        if any(x in low for x in ("什么设备", "是什么设备", "设备类型", "什么内容", "是什么", "介绍", "作用", "what equipment", "what is")) and eq:
            typ = QuestionType.EQUIPMENT_EXPLANATION
        elif any(x in low for x in ("从哪里来", "物流追踪", "上游", "下游", "受到哪些设备影响", "stream", "物流")):
            typ = QuestionType.STREAM_TRACE
        elif any(x in low for x in ("提高", "降低", "影响", "参数", "回流比", "purity", "opex")):
            typ = QuestionType.PARAMETER_IMPACT
        elif any(x in low for x in ("约束", "约束诊断", "超限", "constraint")):
            typ = QuestionType.CONSTRAINT_DIAGNOSIS
        elif any(x in low for x in ("不收敛", "收敛", "convergence", "failed")):
            typ = QuestionType.CONVERGENCE_DIAGNOSIS
        elif any(x in low for x in ("capex", "opex", "权衡", "多目标", "tradeoff")):
            typ = QuestionType.MULTI_OBJECTIVE_TRADEOFF
        elif any(x in low for x in ("实验", "试验", "验证", "experiment")):
            typ = QuestionType.EXPERIMENT_SUGGESTION
        else:
            typ = QuestionType.UNKNOWN

        # Resolve stream-related variables for questions such as
        # "哪些变量可能影响 D1 纯度？".  This is an association only and
        # does not assert causality.
        if typ == QuestionType.PARAMETER_IMPACT and not variables and streams:
            stream_set = set(streams)
            variables.extend(
                str(v.get("name") or key)
                for key, v in context.variables.items()
                if stream_set.intersection(set(v.get("related_streams", [])))
            )
        missing: list[Clarification] = []
        if typ == QuestionType.EQUIPMENT_EXPLANATION and not eq:
            missing.append(Clarification("equipment", "请提供要解释的设备名称（例如 T1）或 Aspen 路径。"))
        if typ == QuestionType.STREAM_TRACE and not streams:
            missing.append(Clarification("stream", "请提供要追踪的物流名称（例如 D1）或 Aspen 路径。"))
        if typ == QuestionType.PARAMETER_IMPACT and not variables:
            missing.append(Clarification("variable", "请提供要评估的变量（例如 T1 回流比）。"))
        if typ == QuestionType.MULTI_OBJECTIVE_TRADEOFF and len(objectives) < 2 and len(context.objectives) < 2:
            missing.append(Clarification("objectives", "请提供要比较的目标（例如 CAPEX、OPEX、纯度）。"))
        if typ == QuestionType.UNKNOWN:
            missing.append(Clarification("question_type", "请说明是设备、物流、参数影响、约束/收敛、多目标权衡还是实验建议。"))
        return QuestionIntent(typ, list(dict.fromkeys(eq)), list(dict.fromkeys(streams)), list(dict.fromkeys(variables)), list(dict.fromkeys(objectives)), list(dict.fromkeys(constraints)), missing)

    # Public names used by callers that separate intent recognition from
    # advice generation.
    identify_question = classify_question

    @staticmethod
    def _local_context(context: ProcessContext, intent: QuestionIntent) -> ProcessContext:
        # ProcessContextBuilder.subgraph performs the same read-only filtering.
        builder = ProcessContextBuilder()
        try:
            if intent.equipment:
                return builder.subgraph(context, equipment=intent.equipment[0])
            if intent.streams:
                return builder.subgraph(context, stream=intent.streams[0])
        except Exception:
            pass
        return context

    def _facts(self, context: ProcessContext, intent: QuestionIntent) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
        local = list(context.sources)
        sim = list(context.history_evidence[: self.max_history])
        # Add explicit current values and graph facts so a hand-built context is useful too.
        for path, value in context.current_values.items():
            local.append(EvidenceItem(f"aspen.value.{len(local)}", EvidenceSource.ASPEN_MODEL, f"Aspen 当前值 {path}", path, value=value, metadata={"layer": "aspen_fact"}))
        if intent.equipment or intent.streams:
            selected = set(intent.equipment + intent.streams)
            for item in context.equipment + context.streams:
                if item.get("name") in selected:
                    local.append(EvidenceItem(f"aspen.graph.{len(local)}", EvidenceSource.ASPEN_MODEL, f"流程图事实 {item}", str(item.get("name")), value=item, metadata={"layer": "aspen_fact"}))
        sim = [EvidenceItem(
            id=item.id, source=item.source, claim=item.claim,
            source_ref=item.source_ref, value=item.value, unit=item.unit,
            confidence=item.confidence, uncertainty=item.uncertainty,
            supports=item.supports, observed_at=item.observed_at,
            metadata={**item.metadata, "layer": "simulation_observation"},
        ) for item in sim]
        return local, sim

    def extract_local_context(self, context: ProcessContext, intent: QuestionIntent) -> ProcessContext:
        """Return the read-only equipment/stream subgraph relevant to intent."""
        return self._local_context(context, intent)

    @staticmethod
    def merge_evidence(
        local_facts: list[EvidenceItem],
        simulation_evidence: list[EvidenceItem],
        ima_answer: ImaAnswer | None,
    ) -> list[EvidenceItem]:
        """Merge layers without changing their provenance or precedence."""
        ima_evidence = [] if ima_answer is None else [
            EvidenceItem(
                f"ima.{i}", EvidenceSource.IMA, s.snippet or s.title or s.source,
                s.source, confidence=s.confidence,
                metadata={**s.to_dict(), "layer": "ima_citation"},
            ) for i, s in enumerate(ima_answer.sources)
        ]
        return list(local_facts) + list(simulation_evidence) + ima_evidence

    @staticmethod
    def _context_text(context: ProcessContext, local: list[EvidenceItem], sim: list[EvidenceItem]) -> str:
        payload = {
            "model_id": context.model_id,
            "equipment": context.equipment,
            "streams": context.streams,
            "topology": context.topology,
            "variables": context.variables,
            "objectives": context.objectives,
            "constraints": context.constraints,
            "current_values": context.current_values,
            "local_evidence": [x.to_dict() for x in local],
            "simulation_observations": [x.to_dict() for x in sim],
            "diagnostics": context.diagnostics,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    def build_query(self, question: str, context: ProcessContext, intent: QuestionIntent | None = None) -> str:
        intent = intent or self.classify_question(question, context)
        scoped = self._local_context(context, intent)
        local, sim = self._facts(scoped, intent)
        return IMA_QUERY_TEMPLATE.format(question=question, context=self._context_text(scoped, local, sim))

    @staticmethod
    def _local_fallback(question: str, intent: QuestionIntent, context: ProcessContext, local: list[EvidenceItem], sim: list[EvidenceItem]) -> TechnicalAdvice:
        risks = ["缺少 IMA 外部机理资料；结论只基于本地流程事实和历史观察。"]
        checks = ["核对 Aspen 当前快照、单位和边界条件。"]
        experiments = []
        conflicts: list[str] = []
        if intent.question_type == QuestionType.CONVERGENCE_DIAGNOSIS:
            failed = [e for e in sim if any(x in e.claim.lower() for x in ("fail", "不收敛", "sim_failed", "error"))]
            conclusion = f"历史记录中发现 {len(failed)} 条可能的失败/不收敛观察；无法仅凭统计记录确定根因。"
            mechanism = "可疑根因需按失败日志、初值、物性和约束逐项排查；当前证据未建立因果关系。"
            checks += ["检查失败工况的 Aspen 错误消息、初值、物性方法和约束残差。"]
        elif intent.question_type == QuestionType.STREAM_TRACE:
            conclusion = f"已从本地拓扑提取物流 {', '.join(intent.streams) or '目标'} 的相邻节点；结果受拓扑完整性限制。"
            mechanism = "上游/下游关系来自 Aspen 拓扑快照，属于流程事实。"
        elif intent.question_type == QuestionType.EQUIPMENT_EXPLANATION:
            conclusion = f"本地快照将 {', '.join(intent.equipment) or '目标'} 标记为设备；设备功能和设计机理仍需外部资料确认。"
            mechanism = "设备名称、类型和 Aspen 路径是模型事实，不等于完整工艺机理。"
        elif intent.question_type == QuestionType.PARAMETER_IMPACT:
            conclusion = "历史数据最多支持相关性观察；不能据此断言参数对目标的确定因果影响。"
            mechanism = "参数影响需要机理资料或受控扰动实验验证。"
            experiments.append("在安全边界内做单变量、小步长扰动，并记录纯度、OPEX 和收敛状态。")
        elif intent.question_type == QuestionType.MULTI_OBJECTIVE_TRADEOFF:
            conclusion = "本地目标定义可用于列出 CAPEX、OPEX、纯度的权衡维度，但当前证据不足以给出唯一最优点。"
            mechanism = "多目标结果应以可行 Pareto 观察呈现，并标注数据覆盖范围。"
        elif intent.question_type == QuestionType.CONSTRAINT_DIAGNOSIS:
            conclusion = "已读取本地约束定义；是否导致不可行需结合具体历史工况和残差确认。"
            mechanism = "约束定义是配置事实，约束违例是仿真观察，两者分层处理。"
        else:
            conclusion = "无法从当前上下文形成可靠技术结论。"
            mechanism = "需要补充问题类型或流程证据。"
        if context.diagnostics:
            conflicts.extend(context.diagnostics)
        return TechnicalAdvice(conclusion=conclusion, mechanism=mechanism, evidence=local + sim, risks=risks, suggested_checks=checks, suggested_experiments=experiments, uncertainty=AdviceUncertainty.HIGH, confidence=0.25, assumptions=["未使用外部 IMA 机理资料"], conflicts=conflicts)

    def advise(self, question: str, context: ProcessContext) -> TechnicalAdviceResult:
        intent = self.classify_question(question, context)
        local_context = self._local_context(context, intent)
        local, sim = self._facts(local_context, intent)
        if not intent.complete:
            advice = TechnicalAdvice(
                conclusion="需要补充澄清字段后才能生成技术建议。",
                mechanism=None,
                evidence=local,
                suggested_checks=[c.prompt for c in intent.clarifications],
                uncertainty=AdviceUncertainty.UNKNOWN,
                confidence=0.0,
                assumptions=[],
            )
            return TechnicalAdviceResult(advice, intent, local_facts=local, simulation_evidence=sim, degraded=True)

        query = self.build_query(question, local_context, intent)
        ima: ImaAnswer | None = None
        degraded = False
        try:
            if self.ima_client is None:
                raise RuntimeError("IMA client unavailable")
            ima = self.ima_client.query_with_sources(ImaRequest(query=query, context={"question_type": intent.question_type.value, "model_id": context.model_id}))
            ima_evidence = [EvidenceItem(f"ima.{i}", EvidenceSource.IMA, s.snippet or s.title or s.source, s.source, confidence=s.confidence, metadata={**s.to_dict(), "layer": "ima_citation"}) for i, s in enumerate(ima.sources)]
            all_evidence = local + sim + ima_evidence
            conflicts = list(context.diagnostics)
            if not ima.sources:
                conflicts.append("IMA 回答未提供可保留的文献来源，机理结论需进一步核验。")
            advice = TechnicalAdvice(
                conclusion=ima.answer or "IMA 未返回可用结论。",
                mechanism="IMA 机理回答；具体适用性由本地 Aspen 事实和历史观察共同限定。",
                evidence=all_evidence,
                risks=["IMA 结论需在当前流程边界和物性方法下核验。"],
                suggested_checks=["核对 IMA 引用与当前设备、物流和操作范围是否一致。"],
                suggested_experiments=["对关键因果假设设计受控小步长实验。"] if intent.question_type in (QuestionType.PARAMETER_IMPACT, QuestionType.EXPERIMENT_SUGGESTION) else [],
                uncertainty=AdviceUncertainty.MEDIUM if ima.confidence is None or ima.confidence < .8 else AdviceUncertainty.LOW,
                confidence=ima.confidence,
                assumptions=list(ima.assumptions),
                conflicts=conflicts,
            )
            return TechnicalAdviceResult(advice, intent, ima, local, sim, False, query)
        except Exception as exc:  # IMA is optional; all expected failures degrade locally.
            degraded = True
            advice = self._local_fallback(question, intent, local_context, local, sim)
            advice.conflicts.append(f"IMA 不可用，已降级：{type(exc).__name__}: {exc}")
            return TechnicalAdviceResult(advice, intent, None, local, sim, degraded, query)


def advise_technical(question: str, context: ProcessContext, ima_client: _ImaLike | None = None) -> TechnicalAdviceResult:
    return TechnicalAdvisor(ima_client).advise(question, context)


TechnicalAdvisorAgent = TechnicalAdvisor
