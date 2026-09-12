"""Guardrails & policy middleware -- validated, safe, compliant I/O by default.

A unified policy surface that runs pre-call (input) and post-call (output)
checks on any LLM/agent/tool text, with four enforcement modes
(``block``/``redact``/``flag``/``require_human``). Every decision is recorded to
a signed, replayable audit trail and streamed to SynapseKit Live -- rule names
and counts only, never the inspected text.

Quick start::

    from synapsekit.guardrails import (
        GuardrailPolicy, GuardedLLM, PromptInjectionGuard, PIIRedactionGuard,
    )

    policy = GuardrailPolicy(guards=[PromptInjectionGuard(), PIIRedactionGuard()])
    safe_llm = GuardedLLM(my_llm, policy)
    answer = await safe_llm.generate("...")   # blocked/redacted per policy

Or start from a compliance rulepack::

    from synapsekit.guardrails import GuardrailPolicy, hipaa_rulepack
    policy = GuardrailPolicy(guards=hipaa_rulepack().guards)
"""

from __future__ import annotations

from .base import Guard, GuardContext
from .input_guards import (
    JailbreakGuard,
    MaxCostGuard,
    PromptInjectionGuard,
    TopicGuard,
)
from .llm import GuardedLLM
from .output_guards import (
    CitationRequiredGuard,
    PIIRedactionGuard,
    SchemaGuard,
    ToxicityGuard,
)
from .policy import GuardrailPolicy
from .rulepacks import (
    RULEPACKS,
    Rulepack,
    gdpr_rulepack,
    hipaa_rulepack,
    pci_rulepack,
    rulepack,
)
from .types import (
    GuardFinding,
    GuardrailBlockedError,
    GuardrailReport,
    Mode,
    Stage,
)

__all__ = [
    # core
    "Guard",
    "GuardContext",
    "GuardFinding",
    "GuardrailBlockedError",
    "GuardrailPolicy",
    "GuardrailReport",
    "GuardedLLM",
    "Mode",
    "Stage",
    # input guards
    "JailbreakGuard",
    "MaxCostGuard",
    "PromptInjectionGuard",
    "TopicGuard",
    # output guards
    "CitationRequiredGuard",
    "PIIRedactionGuard",
    "SchemaGuard",
    "ToxicityGuard",
    # rulepacks
    "RULEPACKS",
    "Rulepack",
    "gdpr_rulepack",
    "hipaa_rulepack",
    "pci_rulepack",
    "rulepack",
]
