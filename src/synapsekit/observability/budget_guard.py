"""Per-request / per-user / daily spending limits with circuit breaker."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded."""

    def __init__(self, message: str, limit_type: str, limit_value: float, current: float) -> None:
        super().__init__(message)
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current = current


@dataclass
class BudgetLimit:
    """Budget limits in USD. All fields are optional."""

    per_request: float | None = None
    per_user: float | None = None
    daily: float | None = None


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class BudgetGuard:
    """Per-request/per-user/daily spending limits with circuit breaker.

    Usage::

        guard = BudgetGuard(BudgetLimit(per_request=0.10, daily=5.00))
        guard.check_before(estimated_cost=0.05)
        # ... make LLM call ...
        guard.record_spend(0.05)
    """

    def __init__(
        self,
        limits: BudgetLimit,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self._limits = limits
        self._cooldown_seconds = cooldown_seconds

        self._daily_spend: float = 0.0
        self._user_spend: dict[str, float] = {}
        self._current_day: int = self._today()
        self._circuit_state = CircuitState.CLOSED
        self._circuit_opened_at: float = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> int:
        """Return current day as ordinal for daily reset detection."""
        import datetime

        return datetime.date.today().toordinal()

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters on calendar day change."""
        today = self._today()
        if today != self._current_day:
            self._daily_spend = 0.0
            self._user_spend.clear()
            self._current_day = today

    def _update_circuit(self) -> None:
        """Transition circuit breaker state."""
        if self._circuit_state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._circuit_opened_at
            if elapsed >= self._cooldown_seconds:
                self._circuit_state = CircuitState.HALF_OPEN

    @property
    def circuit_state(self) -> CircuitState:
        """Current circuit breaker state."""
        with self._lock:
            self._update_circuit()
            return self._circuit_state

    def check_before(self, estimated_cost: float, user_id: str | None = None) -> None:
        """Check if the estimated cost would exceed any limit.

        Raises ``BudgetExceededError`` if any limit would be exceeded.
        """
        with self._lock:
            self._maybe_reset_daily()
            self._update_circuit()

            if self._circuit_state == CircuitState.OPEN:
                raise BudgetExceededError(
                    "Circuit breaker is OPEN — budget exceeded, waiting for cooldown",
                    limit_type="circuit_breaker",
                    limit_value=0,
                    current=self._daily_spend,
                )

            # Per-request check
            if self._limits.per_request is not None and estimated_cost > self._limits.per_request:
                raise BudgetExceededError(
                    f"Estimated cost ${estimated_cost:.4f} exceeds per-request limit "
                    f"${self._limits.per_request:.4f}",
                    limit_type="per_request",
                    limit_value=self._limits.per_request,
                    current=estimated_cost,
                )

            # Daily check
            if (
                self._limits.daily is not None
                and self._daily_spend + estimated_cost > self._limits.daily
            ):
                self._circuit_state = CircuitState.OPEN
                self._circuit_opened_at = time.monotonic()
                raise BudgetExceededError(
                    f"Daily spend ${self._daily_spend + estimated_cost:.4f} would exceed "
                    f"daily limit ${self._limits.daily:.4f}",
                    limit_type="daily",
                    limit_value=self._limits.daily,
                    current=self._daily_spend,
                )

            # Per-user check
            if user_id is not None and self._limits.per_user is not None:
                user_total = self._user_spend.get(user_id, 0.0) + estimated_cost
                if user_total > self._limits.per_user:
                    raise BudgetExceededError(
                        f"User '{user_id}' spend ${user_total:.4f} would exceed "
                        f"per-user limit ${self._limits.per_user:.4f}",
                        limit_type="per_user",
                        limit_value=self._limits.per_user,
                        current=self._user_spend.get(user_id, 0.0),
                    )

    def record_spend(self, cost: float, user_id: str | None = None) -> None:
        """Record actual spend after a call."""
        with self._lock:
            self._maybe_reset_daily()
            self._daily_spend += cost
            if user_id is not None:
                self._user_spend[user_id] = self._user_spend.get(user_id, 0.0) + cost

            # If in HALF_OPEN and spend is under limits, close the circuit
            if self._circuit_state == CircuitState.HALF_OPEN and (
                self._limits.daily is None or self._daily_spend <= self._limits.daily
            ):
                self._circuit_state = CircuitState.CLOSED

    @property
    def daily_spend(self) -> float:
        """Current daily spend."""
        with self._lock:
            self._maybe_reset_daily()
            return self._daily_spend

    def user_spend(self, user_id: str) -> float:
        """Current spend for a specific user."""
        with self._lock:
            self._maybe_reset_daily()
            return self._user_spend.get(user_id, 0.0)

    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._daily_spend = 0.0
            self._user_spend.clear()
            self._circuit_state = CircuitState.CLOSED
