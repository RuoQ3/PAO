"""Incremental, Aspen-independent conversation session layer."""
from .commands import COMMANDS, HELP_TEXT, CommandResult, ParsedInput, handle_command, parse_input
from .persistence import JsonSessionStore, SessionPersistenceError, load_session, save_session
from .state import ConversationSessionState, ConversationState, PendingAction, SessionState
from .session import ConversationSession

# Migration-friendly aliases for callers that prefer protocol-oriented names.
SessionStore = JsonSessionStore
parse_command = parse_input

__all__ = [
    "ConversationState", "ConversationSessionState", "SessionState", "PendingAction",
    "ParsedInput", "CommandResult", "COMMANDS", "HELP_TEXT", "parse_input", "handle_command",
    "JsonSessionStore", "SessionPersistenceError", "save_session", "load_session",
    "ConversationSession",
    "SessionStore", "parse_command",
]
from .unified import UnifiedConversation, build_unified_session
from .dynamic import (
    ALLOWED_AGENTS, DynamicAnswer, DynamicPlan, DynamicTask, answer_dynamic,
    build_dynamic_plan, plan_conversation_turn,
)
from .agents import (
    AgentResult, CaseOverviewAgent, EquipmentAgent, LocalCaseAgents,
    OptimizationAgent, ParameterImpactAgent, SimulationDiagnosticAgent,
    TechnicalAdvisorAgent, TopologyAgent,
)
from .synthesis import SynthesisResult, synthesize_answer

__all__ += [
    "UnifiedConversation", "build_unified_session",
    "DynamicAnswer", "DynamicPlan", "DynamicTask", "answer_dynamic", "build_dynamic_plan",
    "plan_conversation_turn", "ALLOWED_AGENTS", "AgentResult", "CaseOverviewAgent",
    "EquipmentAgent", "TopologyAgent", "ParameterImpactAgent",
    "SimulationDiagnosticAgent", "OptimizationAgent", "TechnicalAdvisorAgent",
    "LocalCaseAgents", "SynthesisResult", "synthesize_answer",
]
