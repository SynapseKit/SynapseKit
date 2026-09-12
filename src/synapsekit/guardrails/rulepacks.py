"""Compliance rulepacks -- starter guard sets for HIPAA, GDPR, and PCI-DSS.

Each rulepack maps a regime's data-handling intent onto concrete guards. These
are engineering starting points that encode common-sense data-minimisation, not
legal certification -- a compliant deployment still needs review by someone
accountable for the regime. Compose a rulepack straight into a policy::

    from synapsekit.guardrails import GuardrailPolicy, hipaa_rulepack

    pack = hipaa_rulepack()
    policy = GuardrailPolicy(guards=pack.guards)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Guard
from .input_guards import PromptInjectionGuard
from .output_guards import PIIRedactionGuard
from .types import Mode


@dataclass(frozen=True)
class Rulepack:
    """A named bundle of guards for a compliance regime."""

    name: str
    description: str
    guards: list[Guard] = field(default_factory=list)


def hipaa_rulepack() -> Rulepack:
    """HIPAA -- guard Protected Health Information.

    Redacts identifiers (email/phone/SSN/etc.) from model output and blocks
    prompt-injection attempts that might coax PHI out of context."""
    return Rulepack(
        name="HIPAA",
        description="Redact PHI from outputs; block instruction-override attacks.",
        guards=[
            PromptInjectionGuard(mode=Mode.BLOCK, name="hipaa.prompt_injection"),
            PIIRedactionGuard(
                detect=["email", "phone", "ssn", "ip_address"],
                mode=Mode.REDACT,
                name="hipaa.phi_redaction",
            ),
        ],
    )


def gdpr_rulepack() -> Rulepack:
    """GDPR -- minimise personal data in outputs.

    Redacts personal identifiers from output (data minimisation) and blocks
    prompt-injection that could bypass those controls."""
    return Rulepack(
        name="GDPR",
        description="Redact personal data from outputs (data minimisation).",
        guards=[
            PromptInjectionGuard(mode=Mode.BLOCK, name="gdpr.prompt_injection"),
            PIIRedactionGuard(
                detect=["email", "phone", "ip_address"],
                mode=Mode.REDACT,
                name="gdpr.personal_data_redaction",
            ),
        ],
    )


def pci_rulepack() -> Rulepack:
    """PCI-DSS -- never let cardholder data leave in the clear.

    Redacts credit-card numbers from output. Card data is high-severity, so the
    redaction guard is paired with an injection block."""
    return Rulepack(
        name="PCI-DSS",
        description="Redact primary account numbers (PAN) from outputs.",
        guards=[
            PromptInjectionGuard(mode=Mode.BLOCK, name="pci.prompt_injection"),
            PIIRedactionGuard(
                detect=["credit_card"],
                mode=Mode.REDACT,
                name="pci.pan_redaction",
            ),
        ],
    )


#: Rulepacks keyed by short name, for lookup by string.
RULEPACKS = {
    "hipaa": hipaa_rulepack,
    "gdpr": gdpr_rulepack,
    "pci": pci_rulepack,
}


def rulepack(name: str) -> Rulepack:
    """Return a rulepack by short name (``"hipaa"``/``"gdpr"``/``"pci"``)."""
    key = name.lower().replace("-dss", "").replace("-", "")
    if key not in RULEPACKS:
        raise KeyError(f"unknown rulepack {name!r}; known: {sorted(RULEPACKS)}")
    return RULEPACKS[key]()
