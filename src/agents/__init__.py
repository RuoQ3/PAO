"""PAO agent layers.

The public conversational entry point is :class:`LeadAgent`.
"""

from .lead import ConversationSession, LeadAgent
from .errors import AgentInternalError, AgentUnavailableError
from .lead import llm_client as llm_client
from .orchestration import (
    ActionProposal, Answer, ApprovalRequest, Classification, Evidence, ExecutionMode,
    IntentSchema, OrchestrationRuntime, RuntimePolicy, SessionState, PendingTask, SubagentRequest, TaskPlanner,
    TaskResult, TaskSpec, TaskStatus,
)
from .orchestration import graph as graph
from .orchestration.parameter_impact import ParameterImpactPlan, SimulationPlan, AspenSandboxRunner, FakeAspenSandboxRunner, SensitivityAnalyzer

__all__ = [
    "ActionProposal", "Answer", "ApprovalRequest", "Classification", "ConversationSession", "Evidence",
    "ExecutionMode", "IntentSchema", "LeadAgent", "OrchestrationRuntime", "RuntimePolicy",
    "SessionState", "PendingTask", "AgentInternalError", "AgentUnavailableError", "SubagentRequest", "TaskPlanner", "TaskResult", "TaskSpec", "TaskStatus",
    # Module aliases keep integrations written during the staged migration
    # importable while the implementation lives under lead/orchestration.
    "llm_client", "graph",
    "ParameterImpactPlan", "SimulationPlan", "AspenSandboxRunner", "FakeAspenSandboxRunner", "SensitivityAnalyzer",
]
