"""Agent OS Shell: mixed natural language and safe shell execution."""

from .adapters import (
    BashAdapter,
    FishAdapter,
    PowerShellAdapter,
    ShellAdapter,
    ZshAdapter,
    all_adapters,
    detect_shell,
    get_adapter,
)
from .completion import CompletionEngine
from .context import ContextCollector, JsonAmbientContext, NullAmbientContext
from .executor import DirectShellExecutor, ShellExecutionError, parse_commands
from .history import ShellHistory
from .lexer import lex_input, split_shell_commands
from .planner import (
    CachedPlanner,
    LLMShellPlanner,
    PlanningError,
    RuleBasedPlanner,
    ShellPlanner,
    TranslationCache,
)
from .safety import SafetyAnalyzer, SafetyError, SafetyPolicy
from .security import generate_signing_key, load_signing_policy
from .session import ShellSession, result_to_dict
from .types import (
    CommandResult,
    InputSegment,
    ParsedInput,
    PlannedStep,
    SafetyAssessment,
    SegmentKind,
    ShellCommand,
    ShellContext,
    ShellKind,
    ShellPlan,
    ShellRunResult,
)

__all__ = [
    "BashAdapter",
    "CachedPlanner",
    "CommandResult",
    "CompletionEngine",
    "ContextCollector",
    "DirectShellExecutor",
    "FishAdapter",
    "InputSegment",
    "JsonAmbientContext",
    "LLMShellPlanner",
    "NullAmbientContext",
    "ParsedInput",
    "PlannedStep",
    "PlanningError",
    "PowerShellAdapter",
    "RuleBasedPlanner",
    "SafetyAnalyzer",
    "SafetyAssessment",
    "SafetyError",
    "SafetyPolicy",
    "SegmentKind",
    "ShellAdapter",
    "ShellCommand",
    "ShellContext",
    "ShellExecutionError",
    "ShellHistory",
    "ShellKind",
    "ShellPlan",
    "ShellPlanner",
    "ShellRunResult",
    "ShellSession",
    "TranslationCache",
    "ZshAdapter",
    "all_adapters",
    "detect_shell",
    "generate_signing_key",
    "get_adapter",
    "lex_input",
    "load_signing_policy",
    "parse_commands",
    "result_to_dict",
    "split_shell_commands",
]
