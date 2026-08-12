"""Privacy-preserving opt-in pooled memory."""

from .aggregator import AllowAllAuthorizer, HiveAggregator, HiveAggregatorError, SQLiteHiveStore
from .client import (
    HiveClient,
    HiveClientError,
    HiveTransport,
    HttpHiveTransport,
    InProcessHiveTransport,
)
from .privacy import (
    PATTERN_VOCABULARY,
    DifferentialPrivacy,
    HivePrivacyError,
    MinedPatterns,
    PatternMiner,
    PrivacyBudgetLedger,
    Pseudonymizer,
    stable_scope_id,
)
from .service import create_hive_app
from .types import (
    HIVE_SCHEMA_VERSION,
    ContributionEnvelope,
    ContributionPayload,
    PatternObservation,
    PrivacyConfig,
    ShareScope,
    Suggestion,
    TransparencyReport,
)

__all__ = [
    "AllowAllAuthorizer",
    "ContributionEnvelope",
    "ContributionPayload",
    "DifferentialPrivacy",
    "HIVE_SCHEMA_VERSION",
    "HiveAggregator",
    "HiveAggregatorError",
    "HiveClient",
    "HiveClientError",
    "HivePrivacyError",
    "HiveTransport",
    "HttpHiveTransport",
    "InProcessHiveTransport",
    "MinedPatterns",
    "PATTERN_VOCABULARY",
    "PatternMiner",
    "PatternObservation",
    "PrivacyBudgetLedger",
    "PrivacyConfig",
    "Pseudonymizer",
    "SQLiteHiveStore",
    "ShareScope",
    "Suggestion",
    "TransparencyReport",
    "create_hive_app",
    "stable_scope_id",
]
