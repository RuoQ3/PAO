"""Structured action protocol and approval gateway.

This package is deliberately Aspen/COM agnostic.  LLMs exchange plain JSON
actions; only a validated, approved action can reach an executor.
"""
from .models import (
    ActionRequest, ActionType, ApprovalStatus, ExecutionMode, ModelIdentity,
    ParameterSpec, ModelSchema, ActionPreview, AuditEvent, AuditEventType,
)
from .validation import ActionValidationError, ValidationResult, validate_action
from .audit import AuditLog
from .executor import MockExecutor, ExecutionError
from .gateway import ActionGateway, ApprovalError
from .parameter_actions import AspenParameterExecutor, BackupToken, parse_parameter_request, compare_results
from src.agents.orchestration.parameter_impact import ParameterImpactPlan, SimulationPlan

__all__ = [
    "ActionRequest", "ActionType", "ApprovalStatus", "ExecutionMode",
    "ModelIdentity", "ParameterSpec", "ModelSchema", "ActionPreview",
    "AuditEvent", "AuditEventType", "ActionValidationError", "ValidationResult",
    "validate_action", "AuditLog", "MockExecutor", "ExecutionError",
    "ActionGateway", "ApprovalError",
    "AspenParameterExecutor", "BackupToken", "parse_parameter_request",
    "compare_results",
    "ParameterImpactPlan", "SimulationPlan",
]
