from __future__ import annotations
from src.agents.errors import AgentUnavailableError, unavailable, fail_closed, failure_from
from src.agents.reasoning import configured, redact

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.agents.evidence.models import (
    AdviceUncertainty,
    EvidenceItem,
    EvidenceSource,
    ProcessContext,
    TechnicalAdvice,
)
from src.integrations.ima import ImaAnswer, ImaRequest
from src.agents.context import ProcessContextBuilder

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
                "target_stream": self.intent.target_stream,
                "target_component": self.intent.target_component,
                "metric": self.intent.metric,
                "candidate_parameters": self.intent.candidate_parameters,
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
            "query": self.query,
        }


class TechnicalAdvisor:
    """Read-only technical consultant.

    The advisor never opens Aspen, writes a model, or runs a simulation.  It
    accepts a prebuilt ProcessContext and an injected IMA client.
    """

    def __init__(self, ima_client: _ImaLike | None = None, *, max_history: int = 20, llm_config=None, llm_call=None):
        self.ima_client = ima_client
        # The Agent request fails on the first IMA transport error.
        from src.integrations.ima import ImaServiceClient
        from dataclasses import replace
        if isinstance(ima_client, ImaServiceClient):
            ima_client.config = replace(ima_client.config, retries=0)
        self.max_history = max_history
        self.llm_config, self.llm_call = llm_config, llm_call

    @fail_closed("technical_intent")
    def classify_question(self, question: str, context: ProcessContext) -> QuestionIntent:
        # Classification receives only user input, before reading ProcessContext.
        from src.agents.lead import llm_client
        cfg = configured(self.llm_config)
        fields = {"question_type", "equipment", "streams", "variables", "objectives", "constraints"}
        raw = (self.llm_call or llm_client.chat_json)(cfg,
            system="Classify the technical question. Return JSON with exactly question_type (equipment_explanation, stream_trace, parameter_impact, constraint_diagnosis, convergence_diagnosis, multi_objective_tradeoff, experiment_suggestion, unknown), equipment, streams, variables, objectives, constraints (arrays of strings), confidence (0..1). Do not invent identifiers.",
            user=question)
        logging.getLogger(__name__).debug("technical intent raw: %s", redact(raw, cfg), extra={"pao_file_only": True})
        data = json.loads(redact(raw, cfg))
        if not isinstance(data, dict) or set(data) != fields | {"confidence"}:
            raise ValueError("Invalid technical classification fields")
        if type(data["confidence"]) not in (int, float) or not .55 <= data["confidence"] <= 1:
            raise unavailable("llm_low_confidence", stage="technical_intent")
        for name in fields - {"question_type"}:
            if not isinstance(data[name], list) or not all(isinstance(v, str) and v.strip() for v in data[name]):
                raise ValueError("Invalid technical classification identifiers")
        return QuestionIntent(QuestionType(data["question_type"]), **{k: data[k] for k in fields - {"question_type"}})

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
        except Exception as exc:
            raise failure_from(exc, "technical_context") from None
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

    @fail_closed("technical_advisor")
    def advise(self, question: str, context: ProcessContext) -> TechnicalAdviceResult:
        if self.ima_client is None:
            raise unavailable("subagent_unavailable", stage="technical_advisor_ima")
        intent = self.classify_question(question, context)
        local_context = self._local_context(context, intent)
        local, sim = self._facts(local_context, intent)
        if not intent.complete:
            raise unavailable("agent_output_invalid", stage="technical_advisor_input")

        query = self.build_query(question, local_context, intent)
        cfg = configured(self.llm_config)
        query = redact(query, cfg)
        ima: ImaAnswer | None = None
        try:
            ima = self.ima_client.query_with_sources(ImaRequest(query=query, context={"question_type": intent.question_type.value, "model_id": context.model_id}))
            logging.getLogger(__name__).debug("IMA response: %s", redact(ima.answer, cfg), extra={"pao_file_only": True})
            if not isinstance(ima.answer, str) or not ima.answer.strip() or not ima.sources:
                raise unavailable("agent_output_invalid", stage="technical_advisor_evidence")
            if type(ima.confidence) not in (int, float) or not .55 <= ima.confidence <= 1:
                raise unavailable("llm_low_confidence", stage="technical_advisor_ima")
            from src.agents.reasoning import reason
            data, _ = reason(question, context={"process": local_context.to_dict(),
                "ima_claim": ima.answer, "citations": [x.to_dict() for x in ima.sources],
                "limitation": "IMA claims are literature evidence, not measured Aspen facts."},
                role="technical_advisor", llm_config=self.llm_config, llm_call=self.llm_call)
            advice = TechnicalAdvice(conclusion=data["text"], mechanism="",
                evidence=self.merge_evidence(local, sim, ima), risks=[],
                suggested_checks=[], suggested_experiments=[],
                uncertainty=AdviceUncertainty.MEDIUM, confidence=data["confidence"],
                assumptions=redact(list(ima.assumptions), cfg), conflicts=data["uncertainties"])
            # Citation text is validated provenance; provider prose stays internal.
            for item in advice.evidence:
                item.claim = redact(item.claim, cfg)
                item.source_ref = redact(item.source_ref, cfg)
                item.value = redact(item.value, cfg)
                item.metadata = redact(item.metadata, cfg)
            return TechnicalAdviceResult(advice, intent, None, local, sim, None)
        except Exception as exc:
            raise failure_from(exc, "technical_advisor_ima") from None


def advise_technical(question: str, context: ProcessContext, ima_client: _ImaLike | None = None) -> TechnicalAdviceResult:
    return TechnicalAdvisor(ima_client).advise(question, context)


TechnicalAdvisorAgent = TechnicalAdvisor
