"""SDK-neutral IMA integration boundary.

The package intentionally has no dependency on a vendor IMA SDK.  Select a
transport (HTTP, MCP, or Mock) behind :class:`ImaClient`.
"""

from .client import ImaClient, ImaServiceClient, MockImaClient
from .config import ImaConfig, load_ima_config
from .errors import (
    ImaAnswerError,
    ImaAuthError,
    ImaError,
    ImaNotConfiguredError,
    ImaServiceUnavailableError,
    ImaTimeoutError,
)
from .models import ImaAnswer, ImaRequest, ImaSource
from .transport import HttpImaTransport, ImaTransport, McpImaTransport, MockImaTransport

__all__ = [
    "HttpImaTransport", "ImaAnswer", "ImaAnswerError", "ImaAuthError", "ImaClient",
    "ImaConfig", "ImaError", "ImaNotConfiguredError", "ImaRequest", "ImaServiceClient",
    "ImaServiceUnavailableError", "ImaSource", "ImaTimeoutError", "ImaTransport",
    "McpImaTransport", "MockImaClient", "MockImaTransport", "load_ima_config",
]
