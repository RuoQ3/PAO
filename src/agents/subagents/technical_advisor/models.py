from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    EQUIPMENT_EXPLANATION = "equipment_explanation"
    STREAM_TRACE = "stream_trace"
    PARAMETER_IMPACT = "parameter_impact"
    CONSTRAINT_DIAGNOSIS = "constraint_diagnosis"
    CONVERGENCE_DIAGNOSIS = "convergence_diagnosis"
    MULTI_OBJECTIVE_TRADEOFF = "multi_objective_tradeoff"
    EXPERIMENT_SUGGESTION = "experiment_suggestion"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Clarification:
    field: str
    prompt: str


@dataclass
class QuestionIntent:
    question_type: QuestionType
    equipment: list[str] = field(default_factory=list)
    streams: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    clarifications: list[Clarification] = field(default_factory=list)
    target_stream: str | None = None
    target_component: str | None = None
    metric: str | None = None
    candidate_parameters: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.clarifications
