"""EU AI Act record-keeping mode for :class:`~synapsekit.audit.wrapper.VerifiableAgent`.

``VerifiableAgent(agent, compliance="eu_ai_act", retention=RetentionPolicy(role=...))``
builds on the existing hash-chained :class:`~synapsekit.audit.trace.AuditTracer`
— there is no second log. In this mode:

- the first record of the agent's lifetime is a ``SYSTEM_EVENT`` stating the
  mode, the retention policy and the purpose map, so the policy that applied
  is itself inside the hash chain;
- every record emitted through the proxy carries, inside its hashed payload,
  the Article 12(2) purposes it is tagged with (``"12(2)(a)"``, ``"12(2)(b)"``,
  ``"12(2)(c)"`` of Regulation (EU) 2024/1689);
- :func:`end_agent_lifetime` records the end of the agent's lifetime.

References are to Regulation (EU) 2024/1689 as amended by Regulation (EU)
2026/1744. Article 12 asks for a logging *capability*; the retention period
lives in Article 19(1) (providers) and Article 26(6) (deployers): at least six
months, unless provided otherwise in applicable Union or national law.
Hash-chaining makes the log tamper-evident; that is an engineering choice, not
something Article 12 itself requires. See ``docs/eu-ai-act.md`` for the
article-by-article mapping and for what this mode does not cover.

Every input is checked against a closed set: an unknown role, mode, purpose or
event kind is rejected with the reason, never ignored.
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import MAXYEAR, datetime
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

from .types import AuditRecord, EventKind

if TYPE_CHECKING:
    from .wrapper import VerifiableAgent

#: Article 12(2) purposes an event can be tagged with — a closed set.
Purpose = Literal["12(2)(a)", "12(2)(b)", "12(2)(c)"]
PURPOSES: Final[tuple[str, ...]] = get_args(Purpose)

#: Compliance modes ``VerifiableAgent`` accepts — a closed set.
ComplianceMode = Literal["eu_ai_act"]
COMPLIANCE_MODES: Final[tuple[str, ...]] = get_args(ComplianceMode)

#: Who keeps the logs: the role decides which article sets the retention duty.
Role = Literal["provider", "deployer"]

REGULATION: Final = "Regulation (EU) 2024/1689"

#: "at least six months" — Article 19(1) and Article 26(6).
MINIMUM_RETENTION_MONTHS: Final = 6

#: Key under which compliance data is stored in a record's (hashed) payload.
PAYLOAD_KEY: Final = "eu_ai_act"

_LEGAL_BASIS: Final[dict[str, str]] = {
    "provider": "Article 19(1)",
    "deployer": "Article 26(6)",
}


def _require_aware(name: str, value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware (got a naive datetime)")
    return value


@dataclass(frozen=True)
class RetentionPolicy:
    """How long the logs must be kept, and under which article.

    ``role`` is required because it decides the legal basis: ``"provider"``
    → Article 19(1), ``"deployer"`` → Article 26(6). ``months`` defaults to
    six, the minimum both articles set. A shorter period is refused unless
    ``overriding_law`` cites the Union or national law that provides
    otherwise (both articles carve that case out, "in particular" Union
    data-protection law). Longer periods are accepted as given.

    This object declares and records the policy; it does not store or
    delete anything — keeping the logs is up to the sink/storage you use.
    """

    role: Role
    months: int = MINIMUM_RETENTION_MONTHS
    overriding_law: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in _LEGAL_BASIS:
            raise ValueError(
                f"role must be 'provider' (Article 19(1)) or 'deployer' (Article 26(6)), "
                f"got {self.role!r}"
            )
        # bool is a subclass of int; True must not mean "1 month".
        if type(self.months) is not int:
            raise TypeError(f"months must be an int, got {type(self.months).__name__}")
        if self.months < 1:
            raise ValueError(f"months must be at least 1, got {self.months}")
        if self.overriding_law is not None:
            if not isinstance(self.overriding_law, str):
                raise TypeError(
                    f"overriding_law must be a str, got {type(self.overriding_law).__name__}"
                )
            cited = self.overriding_law.strip()
            if not cited:
                raise ValueError(
                    "overriding_law, when given, must cite a law (got an empty string)"
                )
            object.__setattr__(self, "overriding_law", cited)
        if self.months < MINIMUM_RETENTION_MONTHS and self.overriding_law is None:
            raise ValueError(
                f"a retention period of {self.months} month(s) is shorter than the "
                f"{MINIMUM_RETENTION_MONTHS}-month minimum of {self.legal_basis}; pass "
                "overriding_law='...' citing the Union or national law that provides otherwise"
            )

    @classmethod
    def provider(
        cls, months: int = MINIMUM_RETENTION_MONTHS, *, overriding_law: str | None = None
    ) -> RetentionPolicy:
        """Policy for a provider of a high-risk AI system (Article 19(1))."""
        return cls(role="provider", months=months, overriding_law=overriding_law)

    @classmethod
    def deployer(
        cls, months: int = MINIMUM_RETENTION_MONTHS, *, overriding_law: str | None = None
    ) -> RetentionPolicy:
        """Policy for a deployer of a high-risk AI system (Article 26(6))."""
        return cls(role="deployer", months=months, overriding_law=overriding_law)

    @property
    def legal_basis(self) -> str:
        return f"{REGULATION}, {_LEGAL_BASIS[self.role]}"

    def retain_until(self, recorded_at: datetime) -> datetime:
        """``recorded_at`` plus the retention period, in calendar months.

        The day of month is kept, or clamped to the last day of the target
        month (31 August + 6 months → 28 or 29 February). This is SynapseKit's
        arithmetic, not a legal determination of when the period ends.
        """
        ts = _require_aware("recorded_at", recorded_at)
        total = ts.month - 1 + self.months
        year = ts.year + total // 12
        month = total % 12 + 1
        if year > MAXYEAR:
            raise ValueError(
                f"retention of {self.months} months from {ts.isoformat()} overflows the calendar"
            )
        day = min(ts.day, calendar.monthrange(year, month)[1])
        return ts.replace(year=year, month=month, day=day)

    def may_delete(self, record: AuditRecord, *, now: datetime) -> bool:
        """``True`` from :meth:`retain_until` of the record's timestamp onwards."""
        current = _require_aware("now", now)
        return current >= self.retain_until(record.timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "months": self.months,
            "minimum_months": MINIMUM_RETENTION_MONTHS,
            "legal_basis": self.legal_basis,
            "overriding_law": self.overriding_law,
        }


@dataclass(frozen=True)
class RemoteBiometricIdentificationProfile:
    """Optional profile for remote biometric identification systems
    (Annex III, point 1(a)), whose logs must provide at least the
    Article 12(3)(a) to (d) fields. Activate it with
    ``VerifiableAgent(..., profile=RemoteBiometricIdentificationProfile())``
    and record each use with :func:`record_biometric_use`.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "annex_iii_point": "1(a)",
            "fields": ["12(3)(a)", "12(3)(b)", "12(3)(c)", "12(3)(d)"],
        }


def validate_purposes(
    purposes: Mapping[EventKind | str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    """Return the full ``kind -> purposes`` map: every :class:`EventKind`
    tagged with all three Article 12(2) purposes, except the kinds that
    ``purposes`` narrows. Rejects anything outside the closed sets.
    """
    result: dict[str, tuple[str, ...]] = {kind.value: PURPOSES for kind in EventKind}
    if purposes is None:
        return result
    if not isinstance(purposes, Mapping):
        raise TypeError(
            f"purposes must be a mapping of EventKind to purposes, got {type(purposes).__name__}"
        )
    for raw_kind, raw_values in purposes.items():
        kind_value = raw_kind.value if isinstance(raw_kind, EventKind) else raw_kind
        if type(kind_value) is not str or kind_value not in result:
            raise ValueError(f"unknown event kind in purposes: {raw_kind!r}")
        if isinstance(raw_values, str | bytes) or not isinstance(raw_values, Sequence):
            raise TypeError(
                f"purposes[{kind_value!r}] must be a list or tuple of purposes, "
                f"got {type(raw_values).__name__}"
            )
        values = list(raw_values)
        if not values:
            raise ValueError(f"purposes[{kind_value!r}] must name at least one purpose")
        for value in values:
            if not isinstance(value, str) or value not in PURPOSES:
                raise ValueError(
                    f"unknown purpose {value!r} for {kind_value!r}; allowed: {list(PURPOSES)}"
                )
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate purpose in purposes[{kind_value!r}]: {values}")
        result[kind_value] = tuple(p for p in PURPOSES if p in values)
    return result


def _active_compliance(verifiable: VerifiableAgent) -> None:
    from .wrapper import VerifiableAgent

    if not isinstance(verifiable, VerifiableAgent):
        raise TypeError(f"expected a VerifiableAgent, got {type(verifiable).__name__}")
    if verifiable._compliance is None:
        raise ValueError("this VerifiableAgent was not created with compliance='eu_ai_act'")


async def end_agent_lifetime(verifiable: VerifiableAgent) -> AuditRecord | None:
    """Record the end of the agent's lifetime (once; later calls return ``None``).

    Recording does not stop: calls made through the proxy afterwards are
    still recorded and tagged, so nothing is ever dropped silently.
    """
    _active_compliance(verifiable)
    return verifiable._record_lifecycle("agent_lifetime_ended")


async def record_biometric_use(
    verifiable: VerifiableAgent,
    *,
    start: datetime,
    end: datetime,
    reference_database: str,
    matched_input: Any,
    verified_by: Sequence[str],
) -> AuditRecord:
    """Record one use of a remote biometric identification system with the
    Article 12(3) fields: (a) start and end of the use, (b) the reference
    database, (c) the input data for which the search led to a match
    (``None`` when there was no match), (d) the natural persons involved in
    verifying the results (Article 14(5)).

    Requires the agent to be in ``compliance="eu_ai_act"`` mode with a
    :class:`RemoteBiometricIdentificationProfile`. Does not enforce the
    Article 14(5) two-person verification — that is a duty on how the
    system is used, not on what is logged.
    """
    _active_compliance(verifiable)
    if verifiable._compliance_profile is None:
        raise ValueError(
            "record_biometric_use needs profile=RemoteBiometricIdentificationProfile() "
            "on the VerifiableAgent"
        )
    start_ts = _require_aware("start", start)
    end_ts = _require_aware("end", end)
    if end_ts < start_ts:
        raise ValueError("end of use is earlier than its start")
    if not isinstance(reference_database, str) or not reference_database.strip():
        raise ValueError(
            "reference_database must name the database checked against (Article 12(3)(b))"
        )
    if isinstance(verified_by, str | bytes) or not isinstance(verified_by, Sequence):
        raise TypeError(
            f"verified_by must be a list or tuple of names, got {type(verified_by).__name__}"
        )
    persons = list(verified_by)
    for person in persons:
        if not isinstance(person, str) or not person.strip():
            raise ValueError(f"every verified_by entry must be a non-empty name, got {person!r}")
    if len({p.strip() for p in persons}) != len(persons):
        raise ValueError(f"duplicate person in verified_by: {persons}")
    if matched_input is not None and not persons:
        raise ValueError(
            "a match was recorded but verified_by is empty (Article 12(3)(d) asks who verified it)"
        )
    from .wrapper import _safe_value

    payload = {
        "annex_iii_point_1a": {
            "use_start": start_ts.isoformat(),
            "use_end": end_ts.isoformat(),
            "reference_database": reference_database.strip(),
            "matched_input": _safe_value(matched_input),
            "verified_by": [p.strip() for p in persons],
        }
    }
    return verifiable._record_compliance_event(EventKind.SYSTEM_EVENT, payload)


__all__ = [
    "COMPLIANCE_MODES",
    "MINIMUM_RETENTION_MONTHS",
    "PAYLOAD_KEY",
    "PURPOSES",
    "ComplianceMode",
    "Purpose",
    "RemoteBiometricIdentificationProfile",
    "RetentionPolicy",
    "Role",
    "end_agent_lifetime",
    "record_biometric_use",
    "validate_purposes",
]
