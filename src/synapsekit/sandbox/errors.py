"""Errors raised by the sandbox lifecycle and apply pipeline."""

from __future__ import annotations


class SandboxError(RuntimeError):
    """Base class for sandbox failures."""


class SandboxStateError(SandboxError):
    """Raised when an operation is invalid for the current lifecycle state."""


class SandboxSecurityError(SandboxError):
    """Raised when a snapshot or diff violates filesystem safety rules."""


class BackendUnavailableError(SandboxError):
    """Raised when a requested isolation backend cannot be used."""


class ApplyConflictError(SandboxError):
    """Raised when the host changed after the sandbox snapshot was created."""
