"""Provider pricing, spend governance, carbon estimates, and cost simulation."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..observability.budget_guard import BudgetExceededError
from ..observability.tracer import COST_TABLE


@dataclass(frozen=True)
class ModelPricing:
    """Price and service metadata for one provider/model/region combination.

    Prices are USD per one million tokens. ``cached_input_cost_per_million``
    applies to the cached subset of input tokens when supplied; otherwise the
    regular input price is used. ``quality_score`` is an optional externally
    configured routing prior, not a model self-assessment.
    """

    provider: str
    model: str
    input_cost_per_million: float
    output_cost_per_million: float
    cached_input_cost_per_million: float | None = None
    latency_sla_ms: float | None = None
    quality_score: float | None = None
    region: str | None = None
    carbon_grams_per_1k_tokens: float | None = None
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")
        for name, value in (
            ("input_cost_per_million", self.input_cost_per_million),
            ("output_cost_per_million", self.output_cost_per_million),
            ("cached_input_cost_per_million", self.cached_input_cost_per_million),
            ("carbon_grams_per_1k_tokens", self.carbon_grams_per_1k_tokens),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.latency_sla_ms is not None and (
            not math.isfinite(self.latency_sla_ms) or self.latency_sla_ms <= 0
        ):
            raise ValueError("latency_sla_ms must be finite and greater than zero")
        if self.quality_score is not None and not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_input_tokens: int = 0,
    ) -> float:
        """Estimate USD cost for a call's token counts."""
        input_count = max(0, int(input_tokens))
        output_count = max(0, int(output_tokens))
        cached_count = min(input_count, max(0, int(cached_input_tokens)))
        regular_count = input_count - cached_count
        cached_price = (
            self.cached_input_cost_per_million
            if self.cached_input_cost_per_million is not None
            else self.input_cost_per_million
        )
        return (
            regular_count * self.input_cost_per_million
            + cached_count * cached_price
            + output_count * self.output_cost_per_million
        ) / 1_000_000.0


# More descriptive alias for callers that use "entry" terminology.
PricingEntry = ModelPricing


class PricingSource(Protocol):
    """Pluggable source used to refresh a :class:`PricingTable`."""

    def fetch(self) -> Iterable[ModelPricing]: ...


class PricingTable:
    """Thread-safe, overridable pricing table.

    Exact provider/model/region entries win over provider/model entries, and
    wildcard-provider entries provide the built-in model-only defaults. A
    caller can replace the defaults without changing SynapseKit source code.
    """

    def __init__(
        self,
        entries: Iterable[ModelPricing] = (),
        *,
        source: PricingSource | Callable[[], Iterable[ModelPricing]] | None = None,
    ) -> None:
        self._entries: dict[tuple[str, str, str], ModelPricing] = {}
        self._lock = threading.RLock()
        self._source = source
        self.update(entries)

    @classmethod
    def from_cost_table(
        cls,
        cost_table: Mapping[str, Mapping[str, float]] | None = None,
        *,
        quality_scores: Mapping[str, float] | None = None,
    ) -> PricingTable:
        """Build wildcard-provider defaults from the existing cost table."""
        table = cost_table or COST_TABLE
        qualities = quality_scores or {}
        return cls(
            ModelPricing(
                provider="*",
                model=model,
                input_cost_per_million=float(prices.get("input", 0.0)) * 1_000_000,
                output_cost_per_million=float(prices.get("output", 0.0)) * 1_000_000,
                quality_score=qualities.get(model),
            )
            for model, prices in table.items()
        )

    @staticmethod
    def _key(entry: ModelPricing) -> tuple[str, str, str]:
        return (entry.provider, entry.model, entry.region or "*")

    def register(self, entry: ModelPricing, *, overwrite: bool = True) -> None:
        """Register or override one pricing entry."""
        key = self._key(entry)
        with self._lock:
            if not overwrite and key in self._entries:
                raise ValueError(f"pricing entry already exists for {key!r}")
            self._entries[key] = entry

    def update(self, entries: Iterable[ModelPricing]) -> None:
        """Register a collection of entries, replacing matching keys."""
        for entry in entries:
            self.register(entry)

    def override(self, provider: str, model: str, **changes: Any) -> ModelPricing:
        """Copy an existing entry with selected fields replaced."""
        current = self.get(model, provider=provider)
        if current is None:
            raise KeyError(f"no pricing entry for provider={provider!r}, model={model!r}")
        allowed = {
            "input_cost_per_million",
            "output_cost_per_million",
            "cached_input_cost_per_million",
            "latency_sla_ms",
            "quality_score",
            "region",
            "carbon_grams_per_1k_tokens",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise TypeError(f"unknown pricing fields: {sorted(unknown)!r}")
        replacement_provider = provider if current.provider == "*" else current.provider
        replacement = ModelPricing(
            provider=replacement_provider,
            model=current.model,
            input_cost_per_million=changes.get(
                "input_cost_per_million", current.input_cost_per_million
            ),
            output_cost_per_million=changes.get(
                "output_cost_per_million", current.output_cost_per_million
            ),
            cached_input_cost_per_million=changes.get(
                "cached_input_cost_per_million", current.cached_input_cost_per_million
            ),
            latency_sla_ms=changes.get("latency_sla_ms", current.latency_sla_ms),
            quality_score=changes.get("quality_score", current.quality_score),
            region=changes.get("region", current.region),
            carbon_grams_per_1k_tokens=changes.get(
                "carbon_grams_per_1k_tokens", current.carbon_grams_per_1k_tokens
            ),
        )
        self.register(replacement)
        return replacement

    def refresh(
        self,
        source: PricingSource | Callable[[], Iterable[ModelPricing]] | None = None,
    ) -> int:
        """Load fresh entries from the configured or supplied source."""
        resolved = source or self._source
        if resolved is None:
            raise ValueError("no pricing source configured")
        fetch = resolved.fetch if hasattr(resolved, "fetch") else resolved
        entries = list(fetch())
        self.update(entries)
        return len(entries)

    def get(
        self,
        model: str,
        *,
        provider: str | None = None,
        region: str | None = None,
    ) -> ModelPricing | None:
        """Return the most specific matching entry."""
        provider_key = provider or "*"
        region_key = region or "*"
        keys = (
            (provider_key, model, region_key),
            (provider_key, model, "*"),
            ("*", model, region_key),
            ("*", model, "*"),
        )
        with self._lock:
            for key in keys:
                if key in self._entries:
                    return self._entries[key]
        return None

    def snapshot(self) -> list[ModelPricing]:
        """Return a stable copy of all entries."""
        with self._lock:
            return list(self._entries.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


@dataclass(frozen=True)
class RequestClassPolicy:
    """Quality and latency constraints for a request class."""

    quality_floor: float = 0.0
    max_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality_floor <= 1.0:
            raise ValueError("quality_floor must be between 0.0 and 1.0")
        if self.max_latency_ms is not None and self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be greater than zero")


RequestPolicy = RequestClassPolicy


@dataclass(frozen=True)
class BudgetPolicy:
    """A hard USD cap and a soft-alert threshold for one scope."""

    limit_usd: float
    soft_alert_threshold: float = 0.8

    def __post_init__(self) -> None:
        if not math.isfinite(self.limit_usd) or self.limit_usd < 0:
            raise ValueError("limit_usd must be finite and non-negative")
        if not 0.0 <= self.soft_alert_threshold <= 1.0:
            raise ValueError("soft_alert_threshold must be between 0.0 and 1.0")


@dataclass(frozen=True)
class BudgetAlert:
    """A one-time soft budget alert."""

    scope: str
    identifier: str
    spend_usd: float
    limit_usd: float
    remaining_usd: float
    threshold: float


@dataclass(frozen=True)
class SpendAttribution:
    """Immutable record linking spend to tenant, key, and provider/model."""

    cost_usd: float
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    tenant_id: str | None = None
    api_key_id: str | None = None
    request_class: str | None = None
    carbon_grams: float | None = None
    timestamp: float = field(default_factory=time.time)


class BudgetReservation:
    """Opaque reservation returned by :meth:`BudgetLedger.reserve`."""

    def __init__(
        self, ledger: BudgetLedger, amount: float, tenant_id: str | None, api_key_id: str | None
    ):
        self.ledger = ledger
        self.amount = amount
        self.tenant_id = tenant_id
        self.api_key_id = api_key_id
        self.active = True


class BudgetLedger:
    """Thread-safe tenant/key budget ledger with atomic preflight reservations."""

    def __init__(
        self,
        *,
        tenant_budgets: Mapping[str, float | BudgetPolicy] | None = None,
        key_budgets: Mapping[str, float | BudgetPolicy] | None = None,
        on_alert: Callable[[BudgetAlert], None] | None = None,
    ) -> None:
        self._tenant_policies = self._normalize_policies(tenant_budgets)
        self._key_policies = self._normalize_policies(key_budgets)
        self._tenant_spend: dict[str, float] = {}
        self._key_spend: dict[str, float] = {}
        self._tenant_reserved: dict[str, float] = {}
        self._key_reserved: dict[str, float] = {}
        self._attributions: list[SpendAttribution] = []
        self._alerts: list[BudgetAlert] = []
        self._alerted: set[tuple[str, str]] = set()
        self._on_alert = on_alert
        self._lock = threading.RLock()

    @staticmethod
    def _normalize_policies(
        policies: Mapping[str, float | BudgetPolicy] | None,
    ) -> dict[str, BudgetPolicy]:
        return {
            str(identifier): value
            if isinstance(value, BudgetPolicy)
            else BudgetPolicy(float(value))
            for identifier, value in (policies or {}).items()
        }

    def set_tenant_budget(self, tenant_id: str, policy: float | BudgetPolicy) -> None:
        with self._lock:
            self._tenant_policies[str(tenant_id)] = self._coerce_policy(policy)

    def set_key_budget(self, api_key_id: str, policy: float | BudgetPolicy) -> None:
        with self._lock:
            self._key_policies[str(api_key_id)] = self._coerce_policy(policy)

    @staticmethod
    def _coerce_policy(policy: float | BudgetPolicy) -> BudgetPolicy:
        return policy if isinstance(policy, BudgetPolicy) else BudgetPolicy(float(policy))

    @staticmethod
    def _validate_amount(amount: float) -> float:
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("budget amount must be finite and non-negative")
        return float(amount)

    def _scope_rows(self, tenant_id: str | None, api_key_id: str | None):
        if tenant_id is not None and str(tenant_id) in self._tenant_policies:
            yield (
                "tenant",
                str(tenant_id),
                self._tenant_policies[str(tenant_id)],
                self._tenant_spend,
                self._tenant_reserved,
            )
        if api_key_id is not None and str(api_key_id) in self._key_policies:
            yield (
                "api_key",
                str(api_key_id),
                self._key_policies[str(api_key_id)],
                self._key_spend,
                self._key_reserved,
            )

    def _check_locked(self, amount: float, tenant_id: str | None, api_key_id: str | None) -> None:
        for scope, identifier, policy, spend, reserved in self._scope_rows(tenant_id, api_key_id):
            current = spend.get(identifier, 0.0) + reserved.get(identifier, 0.0)
            if current + amount > policy.limit_usd + 1e-12:
                raise BudgetExceededError(
                    f"{scope} '{identifier}' budget would exceed ${policy.limit_usd:.6f}",
                    limit_type=f"{scope}_budget",
                    limit_value=policy.limit_usd,
                    current=current,
                )

    def _check_settlement_locked(self, reservation: BudgetReservation, amount: float) -> None:
        """Validate final spend while excluding the reservation being settled."""
        for scope, identifier, policy, spend, reserved in self._scope_rows(
            reservation.tenant_id, reservation.api_key_id
        ):
            current = (
                spend.get(identifier, 0.0) + reserved.get(identifier, 0.0) - reservation.amount
            )
            if current + amount > policy.limit_usd + 1e-12:
                raise BudgetExceededError(
                    f"{scope} '{identifier}' budget would exceed ${policy.limit_usd:.6f}",
                    limit_type=f"{scope}_budget",
                    limit_value=policy.limit_usd,
                    current=max(0.0, current),
                )

    def check_before(
        self,
        estimated_cost: float,
        *,
        tenant_id: str | None = None,
        api_key_id: str | None = None,
    ) -> None:
        """Check a call without reserving capacity."""
        amount = self._validate_amount(estimated_cost)
        with self._lock:
            self._check_locked(amount, tenant_id, api_key_id)

    def reserve(
        self,
        estimated_cost: float,
        *,
        tenant_id: str | None = None,
        api_key_id: str | None = None,
    ) -> BudgetReservation:
        """Atomically reserve estimated capacity for a call."""
        amount = self._validate_amount(estimated_cost)
        tenant = str(tenant_id) if tenant_id is not None else None
        key = str(api_key_id) if api_key_id is not None else None
        with self._lock:
            self._check_locked(amount, tenant, key)
            for _scope, identifier, _policy, _spend, reserved in self._scope_rows(tenant, key):
                reserved[identifier] = reserved.get(identifier, 0.0) + amount
        return BudgetReservation(self, amount, tenant, key)

    def release(self, reservation: BudgetReservation) -> None:
        """Release an unused reservation after a failed provider call."""
        if reservation.ledger is not self:
            return
        with self._lock:
            if not reservation.active:
                return
            self._release_locked(reservation)
            reservation.active = False

    def _release_locked(self, reservation: BudgetReservation) -> None:
        for _scope, identifier, _policy, _spend, reserved in self._scope_rows(
            reservation.tenant_id, reservation.api_key_id
        ):
            remaining = reserved.get(identifier, 0.0) - reservation.amount
            if remaining > 1e-12:
                reserved[identifier] = remaining
            else:
                reserved.pop(identifier, None)

    def commit(
        self,
        reservation: BudgetReservation,
        actual_cost: float,
        *,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        tenant_id: str | None = None,
        api_key_id: str | None = None,
        request_class: str | None = None,
        carbon_grams: float | None = None,
        force: bool = False,
    ) -> SpendAttribution:
        """Settle a reservation and append actual spend attribution.

        ``force`` skips the settlement cap check. Use it to record real spend
        for a call that already succeeded and cannot be undone (the response
        was already delivered) even though its actual cost exceeded the
        reservation's estimate and would otherwise blow the budget cap.
        """
        if reservation.ledger is not self:
            raise ValueError("reservation is inactive or belongs to another ledger")
        amount = self._validate_amount(actual_cost)
        tenant = str(tenant_id) if tenant_id is not None else reservation.tenant_id
        key = str(api_key_id) if api_key_id is not None else reservation.api_key_id
        if tenant != reservation.tenant_id or key != reservation.api_key_id:
            raise ValueError("settlement tenant and API key must match the reservation")
        callbacks: list[BudgetAlert] = []
        with self._lock:
            if not reservation.active:
                raise ValueError("reservation is inactive or belongs to another ledger")
            if not force:
                self._check_settlement_locked(reservation, amount)
            self._release_locked(reservation)
            reservation.active = False
            attribution = self._record_locked(
                amount,
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tenant_id=tenant,
                api_key_id=key,
                request_class=request_class,
                carbon_grams=carbon_grams,
            )
            callbacks = self._new_alerts_locked(tenant, key)
        self._notify(callbacks)
        return attribution

    def has_budget(self, *, tenant_id: str | None = None, api_key_id: str | None = None) -> bool:
        """Return whether either requested scope has a configured budget policy."""
        with self._lock:
            return any(self._scope_rows(tenant_id, api_key_id))

    def record_spend(
        self,
        actual_cost: float,
        *,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        tenant_id: str | None = None,
        api_key_id: str | None = None,
        request_class: str | None = None,
        carbon_grams: float | None = None,
    ) -> SpendAttribution:
        """Record spend without a prior reservation."""
        amount = self._validate_amount(actual_cost)
        tenant = str(tenant_id) if tenant_id is not None else None
        key = str(api_key_id) if api_key_id is not None else None
        callbacks: list[BudgetAlert] = []
        with self._lock:
            self._check_locked(amount, tenant, key)
            attribution = self._record_locked(
                amount,
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tenant_id=tenant,
                api_key_id=key,
                request_class=request_class,
                carbon_grams=carbon_grams,
            )
            callbacks = self._new_alerts_locked(tenant, key)
        self._notify(callbacks)
        return attribution

    def _record_locked(
        self,
        amount: float,
        *,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        tenant_id: str | None,
        api_key_id: str | None,
        request_class: str | None,
        carbon_grams: float | None,
    ) -> SpendAttribution:
        if tenant_id is not None:
            self._tenant_spend[tenant_id] = self._tenant_spend.get(tenant_id, 0.0) + amount
        if api_key_id is not None:
            self._key_spend[api_key_id] = self._key_spend.get(api_key_id, 0.0) + amount
        attribution = SpendAttribution(
            cost_usd=amount,
            input_tokens=max(0, int(input_tokens)),
            output_tokens=max(0, int(output_tokens)),
            model=model,
            provider=provider,
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            request_class=request_class,
            carbon_grams=carbon_grams,
        )
        self._attributions.append(attribution)
        return attribution

    def _new_alerts_locked(
        self, tenant_id: str | None, api_key_id: str | None
    ) -> list[BudgetAlert]:
        alerts: list[BudgetAlert] = []
        for scope, identifier, policy, spend, _reserved in self._scope_rows(tenant_id, api_key_id):
            current = spend.get(identifier, 0.0)
            marker = (scope, identifier)
            if policy.limit_usd <= 0 or current <= 0:
                continue
            if current / policy.limit_usd >= policy.soft_alert_threshold:
                if marker in self._alerted:
                    continue
                self._alerted.add(marker)
                alerts.append(
                    BudgetAlert(
                        scope=scope,
                        identifier=identifier,
                        spend_usd=current,
                        limit_usd=policy.limit_usd,
                        remaining_usd=max(0.0, policy.limit_usd - current),
                        threshold=policy.soft_alert_threshold,
                    )
                )
        self._alerts.extend(alerts)
        return alerts

    def _notify(self, alerts: Iterable[BudgetAlert]) -> None:
        if self._on_alert is None:
            return
        for alert in alerts:
            try:
                self._on_alert(alert)
            except Exception:
                continue

    def remaining_usd(
        self,
        *,
        tenant_id: str | None = None,
        api_key_id: str | None = None,
    ) -> float | None:
        """Return the most restrictive remaining budget for selected scopes."""
        with self._lock:
            values = [
                max(
                    0.0,
                    policy.limit_usd - spend.get(identifier, 0.0) - reserved.get(identifier, 0.0),
                )
                for _scope, identifier, policy, spend, reserved in self._scope_rows(
                    tenant_id, api_key_id
                )
            ]
        return min(values) if values else None

    def spend_usd(self, *, tenant_id: str | None = None, api_key_id: str | None = None) -> float:
        """Return spend for the most specific selected scopes, or zero."""
        with self._lock:
            values = []
            if tenant_id is not None:
                values.append(self._tenant_spend.get(str(tenant_id), 0.0))
            if api_key_id is not None:
                values.append(self._key_spend.get(str(api_key_id), 0.0))
        return max(values) if values else 0.0

    @property
    def attributions(self) -> list[SpendAttribution]:
        with self._lock:
            return list(self._attributions)

    @property
    def alerts(self) -> list[BudgetAlert]:
        with self._lock:
            return list(self._alerts)

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        with self._lock:
            return {
                "tenants": {
                    identifier: {
                        "spend_usd": self._tenant_spend.get(identifier, 0.0),
                        "remaining_usd": max(
                            0.0,
                            policy.limit_usd - self._tenant_spend.get(identifier, 0.0),
                        ),
                        "limit_usd": policy.limit_usd,
                    }
                    for identifier, policy in self._tenant_policies.items()
                },
                "api_keys": {
                    identifier: {
                        "spend_usd": self._key_spend.get(identifier, 0.0),
                        "remaining_usd": max(
                            0.0, policy.limit_usd - self._key_spend.get(identifier, 0.0)
                        ),
                        "limit_usd": policy.limit_usd,
                    }
                    for identifier, policy in self._key_policies.items()
                },
            }


class CarbonEstimator:
    """Optional provider/region carbon estimator.

    Intensities are grams of CO2e per 1,000 tokens and are intentionally
    caller-provided so an organization can use its own regional accounting.
    """

    def __init__(
        self,
        intensities: Mapping[tuple[str, str], float] | None = None,
        *,
        default_grams_per_1k_tokens: float | None = None,
    ) -> None:
        self._intensities: dict[tuple[str, str], float] = {}
        self._default = default_grams_per_1k_tokens
        if default_grams_per_1k_tokens is not None and default_grams_per_1k_tokens < 0:
            raise ValueError("default carbon intensity must be non-negative")
        for (provider, region), intensity in (intensities or {}).items():
            self.register(provider, region, intensity)

    def register(self, provider: str, region: str, grams_per_1k_tokens: float) -> None:
        if not math.isfinite(grams_per_1k_tokens) or grams_per_1k_tokens < 0:
            raise ValueError("carbon intensity must be finite and non-negative")
        self._intensities[(provider, region)] = float(grams_per_1k_tokens)

    def intensity(self, provider: str, region: str | None = None) -> float | None:
        if region is not None and (provider, region) in self._intensities:
            return self._intensities[(provider, region)]
        if (provider, "*") in self._intensities:
            return self._intensities[(provider, "*")]
        if ("*", region or "*") in self._intensities:
            return self._intensities[("*", region or "*")]
        return self._default

    def estimate(self, provider: str, region: str | None, total_tokens: int) -> float | None:
        """Estimate grams of CO2e for a completed call."""
        rate = self.intensity(provider, region)
        return None if rate is None else max(0, int(total_tokens)) / 1_000.0 * rate


@dataclass(frozen=True)
class SimulationTask:
    """One deterministic request in a cost-arbitrage replay trace."""

    task_id: str
    request_class: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class SimulationModel:
    """A provider/model with deterministic quality and latency by class."""

    provider: str
    model: str
    pricing: ModelPricing
    quality_by_class: Mapping[str, float]
    latency_by_class_ms: Mapping[str, float] = field(default_factory=dict)

    def quality(self, request_class: str) -> float:
        return float(self.quality_by_class.get(request_class, 0.0))

    def latency_ms(self, request_class: str) -> float:
        return float(
            self.latency_by_class_ms.get(request_class, self.pricing.latency_sla_ms or 0.0)
        )


@dataclass(frozen=True)
class SimulationResult:
    """Aggregated result of replaying a trace."""

    tasks: int
    routed_cost_usd: float
    baseline_cost_usd: float
    routed_quality: float
    baseline_quality: float

    @property
    def saved_usd(self) -> float:
        return self.baseline_cost_usd - self.routed_cost_usd

    @property
    def savings_pct(self) -> float:
        return self.saved_usd / self.baseline_cost_usd if self.baseline_cost_usd else 0.0

    @property
    def equal_quality(self) -> bool:
        return self.routed_quality + 1e-12 >= self.baseline_quality

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "tasks": self.tasks,
            "routed_cost_usd": self.routed_cost_usd,
            "baseline_cost_usd": self.baseline_cost_usd,
            "saved_usd": self.saved_usd,
            "savings_pct": self.savings_pct,
            "routed_quality": self.routed_quality,
            "baseline_quality": self.baseline_quality,
            "equal_quality": self.equal_quality,
        }


class CostArbitrageSimulator:
    """Replay provider-free traces against an equal-quality quality floor."""

    def __init__(self, models: Iterable[SimulationModel]) -> None:
        self.models = tuple(models)
        if not self.models:
            raise ValueError("at least one simulation model is required")

    def replay(
        self,
        trace: Iterable[SimulationTask],
        *,
        quality_floor: float = 0.0,
        max_latency_ms: float | None = None,
    ) -> SimulationResult:
        if not 0.0 <= quality_floor <= 1.0:
            raise ValueError("quality_floor must be between 0.0 and 1.0")
        routed_cost = baseline_cost = routed_quality = baseline_quality = 0.0
        task_count = 0
        for task in trace:
            task_count += 1
            eligible = [
                model
                for model in self.models
                if model.quality(task.request_class) >= quality_floor
                and (
                    max_latency_ms is None or model.latency_ms(task.request_class) <= max_latency_ms
                )
            ]
            if not eligible:
                raise ValueError(f"no eligible model for task {task.task_id!r}")
            baseline = max(
                eligible,
                key=lambda model: (
                    model.quality(task.request_class),
                    model.pricing.estimate_cost(task.input_tokens, task.output_tokens),
                ),
            )
            routed = min(
                eligible,
                key=lambda model: model.pricing.estimate_cost(
                    task.input_tokens, task.output_tokens
                ),
            )
            routed_cost += routed.pricing.estimate_cost(task.input_tokens, task.output_tokens)
            baseline_cost += baseline.pricing.estimate_cost(task.input_tokens, task.output_tokens)
            routed_quality += routed.quality(task.request_class)
            baseline_quality += baseline.quality(task.request_class)
        divisor = task_count or 1
        return SimulationResult(
            tasks=task_count,
            routed_cost_usd=routed_cost,
            baseline_cost_usd=baseline_cost,
            routed_quality=routed_quality / divisor,
            baseline_quality=baseline_quality / divisor,
        )


__all__ = [
    "BudgetAlert",
    "BudgetLedger",
    "BudgetPolicy",
    "BudgetReservation",
    "CarbonEstimator",
    "CostArbitrageSimulator",
    "ModelPricing",
    "PricingEntry",
    "PricingSource",
    "PricingTable",
    "RequestClassPolicy",
    "RequestPolicy",
    "SimulationModel",
    "SimulationResult",
    "SimulationTask",
    "SpendAttribution",
]
