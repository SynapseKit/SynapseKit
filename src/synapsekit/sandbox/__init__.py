"""Local-first sandboxed PC Twin primitives.

The sandbox package is deliberately dependency-light. Backends are discovered
at runtime and fail closed when their native runtime is unavailable.
"""

from .diff import DiffBundle, DiffPreview
from .errors import (
    ApplyConflictError,
    BackendUnavailableError,
    SandboxError,
    SandboxSecurityError,
    SandboxStateError,
)
from .eval import CallableEvalGate, CommandEvalGate, EvalReceipt, EvalSuiteGate
from .overlay import FsOverlay
from .pc import PCSandbox
from .types import (
    BackendCapabilities,
    CommandResult,
    FileChange,
    FileChangeKind,
    NetworkPolicy,
    SandboxConfig,
    SandboxState,
    SnapshotManifest,
)

__all__ = [
    "ApplyConflictError",
    "BackendCapabilities",
    "BackendUnavailableError",
    "CallableEvalGate",
    "CommandEvalGate",
    "CommandResult",
    "DiffBundle",
    "DiffPreview",
    "EvalReceipt",
    "EvalSuiteGate",
    "FileChange",
    "FileChangeKind",
    "FsOverlay",
    "NetworkPolicy",
    "PCSandbox",
    "SandboxConfig",
    "SandboxError",
    "SandboxSecurityError",
    "SandboxState",
    "SandboxStateError",
    "SnapshotManifest",
]
