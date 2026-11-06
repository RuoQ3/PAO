"""Stable, actionable IMA error categories."""

class ImaError(RuntimeError):
    """Base class for expected IMA integration failures."""

class ImaNotConfiguredError(ImaError):
    """IMA has not been configured (the caller may choose a local fallback)."""

class ImaTimeoutError(ImaError):
    """IMA did not respond within the configured timeout."""

class ImaAuthError(ImaError):
    """IMA rejected credentials or authentication is missing/invalid."""

class ImaServiceUnavailableError(ImaError):
    """IMA endpoint is unavailable or returned a server-side failure."""

class ImaAnswerError(ImaError):
    """IMA returned malformed data or no usable answer."""
