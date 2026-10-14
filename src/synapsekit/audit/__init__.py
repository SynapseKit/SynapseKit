"""Verifiable, cryptographically signed audit trails for SynapseKit.

Not a logger — a tamper-evident provenance system. Every LLM call, tool
invocation, retrieval, memory access, and agent decision can be recorded
into a hash-chained trace, batch-signed with Ed25519 (or a pluggable
KMS/BYOK provider), exported as a portable bundle, and independently
verified without SynapseKit installed.

Typical flow::

    from synapsekit.audit import AuditTracer, EventKind, SigningPolicy, export_audit_bundle

    tracer = AuditTracer()
    tracer.record(EventKind.LLM_CALL, {"model": "gpt-4o-mini", "prompt": "hi"})

    policy = SigningPolicy.ed25519()
    export_audit_bundle(tracer.drain(), policy, "run.audit.zip")

    from synapsekit.audit import verify
    result = verify("run.audit.zip")
    assert result.verdict == "MATCH"
"""

from __future__ import annotations

from .bundle import BUNDLE_SCHEMA_VERSION, export_audit_bundle, export_selective_bundle
from .compliance import (
    RemoteBiometricIdentificationProfile,
    RetentionPolicy,
    end_agent_lifetime,
    record_biometric_use,
)
from .merkle import MerkleHasher, MerkleProof
from .metrics import AuditMetrics
from .otel import OTelAuditExporter, record_to_span_dict
from .redact import (
    DEFAULT_DETECTORS,
    CreditCardDetector,
    EmailDetector,
    IPAddressDetector,
    PhoneDetector,
    PIIRedactor,
    RegexDetector,
    SSNDetector,
)
from .replay import ReplayEngine, ReplayMismatch, ReplayReport
from .serializer import DeterministicSerializer, canonical_json, hash_value
from .signer import (
    AWSKMSSigningProvider,
    AzureKeyVaultSigningProvider,
    BYOKSigningProvider,
    Ed25519SigningProvider,
    GCPKMSSigningProvider,
    KMSSigningProvider,
    SigningPolicy,
    SigningProvider,
    verify_signature,
)
from .sink import (
    AuditSink,
    FileAuditSink,
    KafkaAuditSink,
    MultiSink,
    OTelAuditSink,
    S3AuditSink,
    SignedBatch,
)
from .trace import AuditTracer, ChainIntegrityError, compute_payload_hash, compute_record_hash
from .types import (
    GENESIS_HASH,
    SCHEMA_VERSION,
    AuditRecord,
    AuditVerificationError,
    EventKind,
    RedactionStatus,
    Signature,
    Verdict,
    VerificationResult,
    deep_freeze,
    deep_unfreeze,
)
from .verifier import LoadedBundle, load_bundle, verify
from .wrapper import VerifiableAgent, audited, infer_kind_pair

__all__ = [
    # types
    "AuditRecord",
    "EventKind",
    "RedactionStatus",
    "Signature",
    "Verdict",
    "VerificationResult",
    "AuditVerificationError",
    "SCHEMA_VERSION",
    "GENESIS_HASH",
    "deep_freeze",
    "deep_unfreeze",
    # serializer
    "DeterministicSerializer",
    "canonical_json",
    "hash_value",
    # trace
    "AuditTracer",
    "ChainIntegrityError",
    "compute_record_hash",
    "compute_payload_hash",
    # merkle
    "MerkleHasher",
    "MerkleProof",
    # signer
    "SigningPolicy",
    "SigningProvider",
    "Ed25519SigningProvider",
    "BYOKSigningProvider",
    "KMSSigningProvider",
    "AWSKMSSigningProvider",
    "AzureKeyVaultSigningProvider",
    "GCPKMSSigningProvider",
    "verify_signature",
    # redact
    "PIIRedactor",
    "RegexDetector",
    "DEFAULT_DETECTORS",
    "EmailDetector",
    "PhoneDetector",
    "SSNDetector",
    "CreditCardDetector",
    "IPAddressDetector",
    # bundle
    "export_audit_bundle",
    "export_selective_bundle",
    "BUNDLE_SCHEMA_VERSION",
    # verifier
    "verify",
    "load_bundle",
    "LoadedBundle",
    # replay
    "ReplayEngine",
    "ReplayReport",
    "ReplayMismatch",
    # sinks
    "AuditSink",
    "FileAuditSink",
    "S3AuditSink",
    "KafkaAuditSink",
    "OTelAuditSink",
    "MultiSink",
    "SignedBatch",
    # otel
    "OTelAuditExporter",
    "record_to_span_dict",
    # metrics
    "AuditMetrics",
    # wrapper
    "VerifiableAgent",
    "audited",
    "infer_kind_pair",
    # EU AI Act record-keeping mode
    "RetentionPolicy",
    "RemoteBiometricIdentificationProfile",
    "end_agent_lifetime",
    "record_biometric_use",
]
