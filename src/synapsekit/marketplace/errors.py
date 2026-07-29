"""Exceptions raised by the agent marketplace package."""

from __future__ import annotations


class AgentMarketplaceError(Exception):
    """Base class for agent bundle and registry errors."""


class InvalidAgentBundleError(AgentMarketplaceError):
    """Raised when a bundle is malformed, incomplete, or tampered with."""


class UntrustedPublisherError(AgentMarketplaceError):
    """Raised when an operation requires an independently trusted signer."""


class SandboxRequiredError(AgentMarketplaceError):
    """Raised when an installed agent has no sandbox runner attached."""
