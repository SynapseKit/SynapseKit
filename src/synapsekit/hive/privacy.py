"""On-device pattern mining, pseudonymization, and DP accounting."""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import re
import secrets
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..mesh.privacy import MeshPrivacyFilter
from ..ump import UMPReader
from .types import PatternObservation, PrivacyConfig

_SAFE_TOKEN = re.compile(r"[^a-z0-9_-]+")
_HEADING = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")
_CODE_FENCE = re.compile(r"(?m)^\s*(```|~~~)")

# This vocabulary is intentionally finite.  A bounded vocabulary prevents a
# project name, class name, URL, or arbitrary user text from becoming a Hive
# key.  It can grow through a versioned release rather than user-controlled
# input.
PATTERN_VOCABULARY: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("framework", "fastapi", ("fastapi",)),
    ("framework", "django", ("django",)),
    ("framework", "flask", ("flask",)),
    ("framework", "pydantic", ("pydantic",)),
    ("tooling", "pytest", ("pytest",)),
    ("tooling", "ruff", ("ruff",)),
    ("tooling", "mypy", ("mypy",)),
    ("tooling", "sqlite", ("sqlite",)),
    ("tooling", "postgresql", ("postgres", "postgresql")),
    ("tooling", "redis", ("redis",)),
    ("practice", "async-io", ("asyncio", "async/await", "async def")),
    ("practice", "exponential-backoff", ("exponential backoff", "backoff", "retry")),
    ("practice", "idempotency", ("idempotent", "idempotency")),
    ("practice", "oauth", ("oauth", "oauth2")),
    ("practice", "jwt", ("jwt", "json web token")),
    ("practice", "structured-logging", ("structured logging", "structlog")),
    ("practice", "feature-flags", ("feature flag", "feature-flags")),
    ("practice", "property-testing", ("property-based", "hypothesis")),
)


class HivePrivacyError(ValueError):
    """Raised when the local privacy policy cannot safely proceed."""


@dataclass
class PrivacyBudgetLedger:
    """A small in-memory budget ledger with explicit reservation semantics."""

    config: PrivacyConfig
    spent: float = 0.0
    contribution_count: int = 0
    day: int = field(default_factory=lambda: int(__import__("time").time() // 86400))

    def reserve(self, epsilon: float | None = None) -> None:
        now_day = int(__import__("time").time() // 86400)
        if now_day != self.day:
            self.day = now_day
            self.contribution_count = 0
        amount = self.config.epsilon if epsilon is None else epsilon
        if amount <= 0:
            raise HivePrivacyError("contribution epsilon must be positive")
        if self.spent + amount > self.config.budget_limit + 1e-12:
            raise HivePrivacyError("differential-privacy budget exhausted")
        if self.contribution_count >= self.config.max_contributions_per_day:
            raise HivePrivacyError("daily Hive contribution limit reached")
        self.spent += amount
        self.contribution_count += 1

    @property
    def remaining(self) -> float:
        return max(0.0, self.config.budget_limit - self.spent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spent": self.spent,
            "contribution_count": self.contribution_count,
            "day": self.day,
        }

    @classmethod
    def from_dict(cls, config: PrivacyConfig, value: dict[str, Any]) -> PrivacyBudgetLedger:
        return cls(
            config=config,
            spent=float(value.get("spent", 0.0)),
            contribution_count=int(value.get("contribution_count", 0)),
            day=int(value.get("day", int(__import__("time").time() // 86400))),
        )


class Pseudonymizer:
    """Create non-reversible, scope-specific contributor identifiers."""

    def __init__(self, secret: bytes | None = None) -> None:
        self._secret = secret or secrets.token_bytes(32)

    @property
    def secret(self) -> bytes:
        return self._secret

    def pseudonym(self, identity: str, scope: str) -> str:
        if not identity:
            raise HivePrivacyError("contributor identity must not be empty")
        message = f"hive:v1:{scope}:{identity}".encode()
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()[:32]


class DifferentialPrivacy:
    """Bounded Laplace mechanism with injectable randomness for testing."""

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self.rng = rng or random.SystemRandom()

    def laplace(
        self, value: float, *, epsilon: float, sensitivity: float = 1.0, scale: float = 1.0
    ) -> float:
        if epsilon <= 0 or sensitivity < 0 or scale <= 0:
            raise HivePrivacyError("invalid Laplace mechanism parameters")
        u = self.rng.random() - 0.5
        noise = -sensitivity * scale / epsilon * math.copysign(math.log1p(-2 * abs(u)), u)
        return value + noise

    def privatize(self, counts: Counter[str], config: PrivacyConfig) -> dict[str, float]:
        return {
            key: max(0.0, self.laplace(count, epsilon=config.epsilon, scale=config.noise_scale))
            for key, count in counts.items()
        }


@dataclass(frozen=True)
class MinedPatterns:
    patterns: tuple[PatternObservation, ...]
    selected_keys: tuple[str, ...]
    excluded_keys: tuple[str, ...]


class PatternMiner:
    """Extract bounded convention signals from local UMP markdown files."""

    def __init__(
        self, *, vocabulary: Sequence[tuple[str, str, tuple[str, ...]]] = PATTERN_VOCABULARY
    ) -> None:
        self.vocabulary = tuple(vocabulary)

    def mine(
        self,
        roots: Iterable[str | Path],
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        privacy_filter: MeshPrivacyFilter | None = None,
        max_patterns: int = 64,
    ) -> MinedPatterns:
        filter_ = privacy_filter or MeshPrivacyFilter()
        include_patterns = tuple(include or ("**/*.md", "*.md"))
        exclude_patterns = tuple(exclude or ())
        counts: Counter[str] = Counter()
        categories: dict[str, str] = {}
        excluded: set[str] = set()
        paths: set[Path] = set()
        for root_value in roots:
            root = Path(root_value).expanduser()
            if root.is_file():
                paths.add(root)
                continue
            if not root.exists():
                continue
            for path in root.rglob("*.md"):
                if path.is_file():
                    paths.add(path)

        for path in sorted(paths):
            if not filter_.allows(path):
                excluded.add("privacy-filter")
                continue
            normalized = path.as_posix()
            if include_patterns and not any(
                path.match(pattern) or Path(normalized).match(pattern)
                for pattern in include_patterns
            ):
                excluded.add("include-pattern")
                continue
            if any(path.match(pattern) for pattern in exclude_patterns):
                excluded.add("exclude-pattern")
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                document = UMPReader.parse(raw, source_path="")
            except (OSError, ValueError):
                excluded.add("unreadable")
                continue
            # Only vocabulary matches and coarse structural buckets survive.
            text = (
                f"{document.frontmatter.type} {document.frontmatter.scope} {document.body}".lower()
            )
            for category, key, needles in self.vocabulary:
                if any(needle in text for needle in needles):
                    full_key = f"{category}:{key}"
                    counts[full_key] += 1
                    categories[full_key] = category
            headings = _HEADING.findall(document.body)
            if headings:
                key = "memory:uses-headings"
                counts[key] += 1
                categories[key] = "memory"
            if _CODE_FENCE.search(document.body):
                key = "memory:includes-code-examples"
                counts[key] += 1
                categories[key] = "memory"

        selected = tuple(key for key, _ in counts.most_common(max_patterns))
        observations = tuple(
            PatternObservation(key=key, value=float(counts[key]), category=categories[key])
            for key in selected
        )
        return MinedPatterns(
            patterns=observations,
            selected_keys=selected,
            excluded_keys=tuple(sorted(excluded)),
        )


def stable_scope_id(scope: str, team_id: str | None = None) -> str:
    parsed = scope.strip().lower()
    if parsed == "local":
        return "local"
    if not team_id:
        raise HivePrivacyError("team_id is required for team and community scopes")
    safe = _SAFE_TOKEN.sub("-", team_id.lower()).strip("-")
    if not safe or len(safe) > 96:
        raise HivePrivacyError("team_id must be a bounded, simple identifier")
    if parsed == "community":
        return "community"
    if parsed == "team":
        return f"team:{safe}"
    raise HivePrivacyError("scope must be local, team, or community")
