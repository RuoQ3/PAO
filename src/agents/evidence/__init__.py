"""Public, SDK-neutral contracts for the PAO conversation architecture.

The contracts in this package are transport objects only.  They do not import
Aspen, LangGraph, an IMA client, or a database implementation.
"""

from .models import (
    ActionKind,
    AdviceUncertainty,
    EvidenceBundle,
    EvidenceItem,
    EvidenceSource,
    ImaAnswer,
    ProcessContext,
    TechnicalAdvice,
    UserAction,
)
from .protocols import (
    ActionGateway,
    ConversationOrchestrator,
    ImaAdapter,
    ProcessContextBuilder,
    TechnicalAdvisor,
)

__all__ = [
    "ActionGateway",
    "ActionKind",
    "ConversationOrchestrator",
    "AdviceUncertainty",
    "EvidenceBundle",
    "EvidenceItem",
    "EvidenceSource",
    "ImaAdapter",
    "ImaAnswer",
    "ProcessContext",
    "ProcessContextBuilder",
    "TechnicalAdvice",
    "TechnicalAdvisor",
    "UserAction",
]
