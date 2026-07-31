"""Provenance primitives for learning-relevant signals.

This module provides :class:`GroundedSignal` and :class:`SignalSource`: a small,
stateless way to tag any number that influences future behaviour (a reward, a
quality score, a cost observation) with *who vouched for it* — an independent
source, or the same agent/process that is being scored.

The design is deliberately a two-tier split, not a confidence spectrum. Either a
value was supplied by something *other than* the agent under evaluation
(``EXTERNAL_OVERRIDE`` — a human review, a separate judge model, an ``EvalSuite``
computed by code that isn't the agent), or it was supplied by the agent under
evaluation itself, anywhere in the call (``SELF_REPORTED`` — a bid estimate, an
``output["quality"]`` field, a self-graded score). There is no middle tier,
because "extracted from the agent's own output" and "the agent's own bid" are
not meaningfully different in trustworthiness — both are the agent describing
itself, and a middle tier is where "we're mostly sure this is fine" bugs live.

Consumers (reputation stores, routers, eval gates, audit trails) treat an
ungrounded signal as discounted/filterable evidence rather than silently
trusting it as ground truth. The primitive only carries the *label*; each
consumer decides its own policy for what to do with an ungrounded signal.

This is complementary to, and distinct from, ``VerifiableAgent`` (#738): that
answers "is this record authentic and unmodified"; ``GroundedSignal`` answers
"is the *content* of this number trustworthy". A self-reported score can be
faithfully, cryptographically signed and still be wrong — verifiable and
grounded are orthogonal properties.

Note: ``provenance`` is free-form documentation, not proof. The ``source`` enum
is the actual trust boundary — nothing stops a caller from writing a misleading
``provenance`` dict on a self-reported signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["GroundedSignal", "SignalSource"]


class SignalSource(str, Enum):
    """Where a learning-relevant signal came from — a strict two-way split.

    ``EXTERNAL_OVERRIDE`` means something *other than* the agent under
    evaluation supplied the value (an independent judge, a human, an eval
    harness). ``SELF_REPORTED`` means the agent under evaluation supplied it,
    through any code path. There is intentionally no third, "somewhat trusted"
    tier.
    """

    EXTERNAL_OVERRIDE = "external_override"
    SELF_REPORTED = "self_reported"

    @classmethod
    def coerce(cls, value: SignalSource | str) -> SignalSource:
        if isinstance(value, SignalSource):
            return value
        normalised = str(value).replace("-", "_").lower()
        aliases = {
            "external": "external_override",
            "override": "external_override",
            "grounded": "external_override",
            "self": "self_reported",
            "self_report": "self_reported",
            "ungrounded": "self_reported",
        }
        try:
            return cls(aliases.get(normalised, normalised))
        except ValueError as exc:
            raise ValueError(f"Unknown signal source: {value}") from exc


@dataclass(frozen=True, slots=True)
class GroundedSignal:
    """A numeric signal tagged with whether its source is externally grounded.

    ``value`` is the number itself (reward, quality, cost, confidence — the
    primitive is agnostic). ``source`` is the two-tier trust boundary.
    ``provenance`` is a free-form, unvalidated dict for debugging/audit display
    (who/what supplied this, and how) — it is documentation, not proof.

    ``grounded`` is a *derived, read-only* property (``True`` only when
    ``source is EXTERNAL_OVERRIDE``), so a caller can't construct a
    self-reported signal that claims to be grounded.
    """

    value: float
    source: SignalSource
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclass: normalise fields via object.__setattr__.
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "source", SignalSource.coerce(self.source))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def grounded(self) -> bool:
        """``True`` only when the value came from an external source."""
        return self.source is SignalSource.EXTERNAL_OVERRIDE

    @classmethod
    def external(cls, value: float, **provenance: Any) -> GroundedSignal:
        """Build an externally-grounded signal (independent judge/human/eval)."""
        return cls(value=value, source=SignalSource.EXTERNAL_OVERRIDE, provenance=provenance)

    @classmethod
    def self_reported(cls, value: float, **provenance: Any) -> GroundedSignal:
        """Build a self-reported signal (the agent under evaluation supplied it)."""
        return cls(value=value, source=SignalSource.SELF_REPORTED, provenance=provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source.value,
            "grounded": self.grounded,
            "provenance": dict(self.provenance),
        }
