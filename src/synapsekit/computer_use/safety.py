from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from .types import (
    ComputerAction,
    ComputerActionType,
    ComputerObservation,
    SafetyCheck,
    SafetyDecision,
)

_DEFAULT_FORBIDDEN_APPS = (
    "1password",
    "bitwarden",
    "keychain",
    "keychain access",
    "lastpass",
    "keeper",
)
_DANGEROUS_WORDS = (
    "delete",
    "remove",
    "erase",
    "purchase",
    "buy",
    "send",
    "submit",
    "transfer",
    "pay",
    "password",
    "secret",
    "credential",
)
_SENSITIVE_ACTIONS = {
    ComputerActionType.TYPE_TEXT,
    ComputerActionType.KEY,
    ComputerActionType.HOTKEY,
    ComputerActionType.NAVIGATE,
}


@dataclass
class AuditEntry:
    """Audit record emitted for each safety decision."""

    decision: SafetyDecision
    reason: str
    action_type: str
    app: str | None = None
    window_title: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyPolicy:
    """Safety gates for computer-use actions.

    The policy is intentionally conservative. It blocks credential managers
    by default and requires explicit confirmation for sensitive actions that
    match configured risk keywords.
    """

    confirm_before: Sequence[str] = field(default_factory=lambda: ("delete", "send", "purchase"))
    forbidden_apps: Sequence[str] = field(default_factory=lambda: _DEFAULT_FORBIDDEN_APPS)
    allowed_apps: Sequence[str] | None = None
    allowed_domains: Sequence[str] | None = None
    blocked_domains: Sequence[str] = field(default_factory=tuple)
    require_confirmation_for_text: bool = True
    record_session: bool = True
    audit_log: list[AuditEntry] = field(default_factory=list)

    def check(
        self,
        action: ComputerAction,
        observation: ComputerObservation,
        *,
        task: str = "",
    ) -> SafetyCheck:
        """Evaluate a proposed action against the policy."""

        action_type = action.action_type
        app_name = (observation.app or "").lower()
        window_title = (observation.window_title or "").lower()

        if self._matches_any(app_name, self.forbidden_apps) or self._matches_any(
            window_title, self.forbidden_apps
        ):
            return self._record(
                SafetyDecision.BLOCKED,
                "Current app or window is forbidden by SafetyPolicy.",
                action,
                observation,
            )

        if (
            self.allowed_apps is not None
            and app_name
            and not self._matches_any(app_name, self.allowed_apps)
        ):
            return self._record(
                SafetyDecision.BLOCKED,
                "Current app is outside SafetyPolicy.allowed_apps.",
                action,
                observation,
            )

        if action_type == ComputerActionType.NAVIGATE:
            domain_check = self._check_domain(action.url)
            if domain_check is not None:
                return self._record(domain_check[0], domain_check[1], action, observation)

        if self._requires_confirmation(action, observation, task):
            return self._record(
                SafetyDecision.NEEDS_CONFIRMATION,
                "Action requires human confirmation by SafetyPolicy.",
                action,
                observation,
            )

        return self._record(SafetyDecision.ALLOWED, "Action allowed.", action, observation)

    def _requires_confirmation(
        self,
        action: ComputerAction,
        observation: ComputerObservation,
        task: str,
    ) -> bool:
        text = " ".join(
            part
            for part in (
                task,
                action.text or "",
                action.reason or "",
                observation.text,
                observation.window_title or "",
            )
            if part
        ).lower()
        keywords = tuple(self.confirm_before) if self.confirm_before is not None else _DANGEROUS_WORDS
        if any(re.search(rf"\b{re.escape(word.lower())}\b", text) for word in keywords):
            return True
        if self.require_confirmation_for_text and action.action_type in _SENSITIVE_ACTIONS:
            return bool(action.text and self._contains_secret_like_text(action.text))
        return False

    def _check_domain(self, url: str | None) -> tuple[SafetyDecision, str] | None:
        if not url:
            return (SafetyDecision.BLOCKED, "Navigation action has no URL.")
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or not host:
            return (SafetyDecision.BLOCKED, "Navigation URL must be http(s) with a hostname.")
        if self._matches_domain(host, self.blocked_domains):
            return (SafetyDecision.BLOCKED, "Navigation domain is blocked by SafetyPolicy.")
        if self.allowed_domains is not None and not self._matches_domain(
            host, self.allowed_domains
        ):
            return (
                SafetyDecision.NEEDS_CONFIRMATION,
                "Navigation domain is outside SafetyPolicy.allowed_domains.",
            )
        return None

    def _record(
        self,
        decision: SafetyDecision,
        reason: str,
        action: ComputerAction,
        observation: ComputerObservation,
    ) -> SafetyCheck:
        check = SafetyCheck(
            decision=decision, reason=reason, action=action, observation=observation
        )
        self.audit_log.append(
            AuditEntry(
                decision=decision,
                reason=reason,
                action_type=action.action_type.value,
                app=observation.app,
                window_title=observation.window_title,
                url=action.url or observation.url,
            )
        )
        return check

    @staticmethod
    def _matches_any(value: str, patterns: Sequence[str]) -> bool:
        return any(pattern.lower() in value for pattern in patterns)

    @staticmethod
    def _matches_domain(host: str, domains: Sequence[str]) -> bool:
        normalized = [domain.lower() for domain in domains]
        return any(host == domain or host.endswith("." + domain) for domain in normalized)

    @staticmethod
    def _contains_secret_like_text(text: str) -> bool:
        lowered = text.lower()
        if any(word in lowered for word in ("password", "api_key", "token", "secret")):
            return True
        return bool(re.search(r"(sk-|ghp_|xox[baprs]-)[A-Za-z0-9_-]{8,}", text))
