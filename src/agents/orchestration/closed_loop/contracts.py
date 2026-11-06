"""Typed, strictly validated actions and immutable engineering boundaries."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric parameter")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Parameter must be finite")
    return number


@dataclass
class LoopConfig:
    max_evaluations: int = 40
    batch_size: int = 5
    failure_trigger: int = 3
    max_decisions: int = 12
    stagnation_batches: int = 5
    max_step_fraction: float = 0.1
    candidate_pool_size: int = 24
    verification_repeats: int = 1
    verification_rtol: float = 0.02
    verification_atol: float = 1e-8
    objective_targets: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "LoopConfig":
        raw = dict(raw or {})
        # Configuration-only routing fields are consumed by the integration layer.
        for name in ("enabled", "checkpoint_path", "resume", "knowledge_files", "process_facts"):
            raw.pop(name, None)
        obj = cls(**raw)
        for name in ("max_evaluations", "batch_size", "failure_trigger", "max_decisions",
                     "stagnation_batches", "candidate_pool_size", "verification_repeats"):
            value = getattr(obj, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if obj.max_evaluations <= obj.verification_repeats:
            raise ValueError("Budget must leave room for exploration and verification")
        if not 0 < finite(obj.max_step_fraction) <= 1:
            raise ValueError("max_step_fraction must be in (0, 1]")
        if finite(obj.verification_rtol) < 0 or finite(obj.verification_atol) < 0:
            raise ValueError("Verification tolerances must be nonnegative")
        obj.objective_targets = {k: finite(v) for k, v in obj.objective_targets.items()}
        return obj


@dataclass
class ActionPlan:
    action: str = "continue"
    hypothesis: str = "Continue evidence collection"
    evidence_case_ids: list[str] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)
    search_region: dict[str, list[float]] = field(default_factory=dict)
    candidate: dict[str, float] = field(default_factory=dict)
    batch_size: int = 1
    expected_effect: str = "feasibility"
    initialization: str = "reset"
    repeat: bool = False

    @classmethod
    def parse(cls, raw: dict) -> "ActionPlan":
        if not isinstance(raw, dict):
            raise ValueError("Action must be a JSON object")
        # Unknown fields (including constraints/objectives/hard_bounds) fail closed.
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_plan(plan: ActionPlan, state: dict, cfg: LoopConfig,
                  integer_paths: set[str], knowledge_ids: set[str]) -> ActionPlan:
    if plan.action not in {"continue", "set_region", "probe", "stop"}:
        raise ValueError("Unsupported action")
    if not isinstance(plan.hypothesis, str) or not plan.hypothesis.strip():
        raise ValueError("A hypothesis is required")
    if plan.expected_effect not in {"feasibility", "convergence", "objective", "information"}:
        raise ValueError("Unknown expected_effect")
    if plan.initialization not in {"reset", "previous"} or type(plan.repeat) is not bool:
        raise ValueError("Invalid initialization/repeat")
    if type(plan.batch_size) is not int or not 1 <= plan.batch_size <= cfg.batch_size:
        raise ValueError("Action exceeds per-decision batch budget")
    if not isinstance(plan.evidence_case_ids, list) or not isinstance(plan.knowledge_ids, list):
        raise ValueError("Evidence and knowledge references must be arrays")
    known = {row["case_id"] for row in state["observations"]}
    if not set(plan.evidence_case_ids) <= known:
        raise ValueError("Unknown evidence case")
    if known and not plan.evidence_case_ids:
        raise ValueError("Decision must cite observed evidence")
    if not set(plan.knowledge_ids) <= knowledge_ids:
        raise ValueError("Unknown knowledge reference")
    hard = state["hard_bounds"]
    if not isinstance(plan.search_region, dict) or not isinstance(plan.candidate, dict):
        raise ValueError("Regions and candidates must be mappings")
    if plan.search_region and plan.action != "set_region":
        raise ValueError("Only set_region can change the search region")
    if plan.candidate and plan.action != "probe":
        raise ValueError("Only probe may specify a candidate")
    if plan.action == "set_region" and not plan.search_region:
        raise ValueError("set_region requires search_region")
    if plan.action == "probe" and set(plan.candidate) != set(hard):
        raise ValueError("probe requires every optimizer variable, and no extra keys")
    for key, pair in plan.search_region.items():
        if key not in hard or not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Invalid region variable or interval")
        lo, hi = map(finite, pair)
        if not hard[key][0] <= lo <= hi <= hard[key][1]:
            raise ValueError("Search region exceeds immutable engineering bounds")
        if key in integer_paths and math.ceil(lo) > math.floor(hi):
            raise ValueError("Region contains no integer candidate")
        plan.search_region[key] = [lo, hi]
    for key, value in plan.candidate.items():
        value = finite(value)
        if not hard[key][0] <= value <= hard[key][1]:
            raise ValueError("Candidate exceeds engineering bounds")
        if key in integer_paths and not value.is_integer():
            raise ValueError("Integer variable requires an integer")
        plan.candidate[key] = value
    return plan
