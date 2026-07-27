"""File-backed reference registry for signed agent bundles."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import threading
from builtins import list as builtin_list
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any
from uuid import uuid4

from ..audit.serializer import canonical_json
from ..audit.signer import SigningProvider, verify_signature
from .bundle import PublisherIdentity, verify_agent_bundle
from .errors import InvalidAgentBundleError, UntrustedPublisherError

REGISTRY_SCHEMA_VERSION = "1.0"
REVIEW_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """Published agent version recorded in the registry index."""

    name: str
    version: str
    author: str
    description: str
    tags: tuple[str, ...]
    publisher_key_id: str
    bundle_sha256: str
    bundle_path: str
    published_at: str
    eval_score: float | None = None

    def __post_init__(self) -> None:
        _validate_registry_component(self.name, "agent name")
        _validate_registry_component(self.version, "agent version")
        if not re.fullmatch(r"[0-9a-f]{64}", self.bundle_sha256):
            raise ValueError("Registry bundle_sha256 must be a lowercase SHA-256 digest.")
        expected_path = (
            Path("packages") / self.name / self.version / f"{self.name}-{self.version}.agent"
        ).as_posix()
        if self.bundle_path != expected_path:
            raise ValueError("Registry entry bundle_path does not match its name and version.")
        if self.eval_score is not None and not 0.0 <= self.eval_score <= 1.0:
            raise ValueError("Registry eval_score must be between 0.0 and 1.0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "tags": list(self.tags),
            "publisher_key_id": self.publisher_key_id,
            "bundle_sha256": self.bundle_sha256,
            "bundle_path": self.bundle_path,
            "published_at": self.published_at,
            "eval_score": self.eval_score,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RegistryEntry:
        raw_tags = value.get("tags", [])
        if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
            raise ValueError("Registry entry tags must be a list of strings.")
        raw_score = value.get("eval_score")
        return cls(
            name=str(value["name"]),
            version=str(value["version"]),
            author=str(value["author"]),
            description=str(value.get("description", "")),
            tags=tuple(raw_tags),
            publisher_key_id=str(value["publisher_key_id"]),
            bundle_sha256=str(value["bundle_sha256"]),
            bundle_path=str(value["bundle_path"]),
            published_at=str(value["published_at"]),
            eval_score=None if raw_score is None else float(raw_score),
        )


@dataclass(frozen=True, slots=True)
class SignedAgentReview:
    """A cryptographically signed agent rating and eval observation."""

    agent_name: str
    agent_version: str
    reviewer: str
    rating: int
    eval_score: float
    comment: str
    signed_at: str
    reviewer_identity: PublisherIdentity
    signature_b64: str
    schema_version: str = REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REVIEW_SCHEMA_VERSION:
            raise ValueError(f"Unsupported review schema: {self.schema_version!r}.")
        if not self.agent_name or not self.agent_version or not self.reviewer.strip():
            raise ValueError("Review agent name, version, and reviewer are required.")
        _validate_registry_component(self.agent_name, "review agent name")
        _validate_registry_component(self.agent_version, "review agent version")
        if not 1 <= self.rating <= 5:
            raise ValueError("Review rating must be between 1 and 5.")
        if not 0.0 <= self.eval_score <= 1.0:
            raise ValueError("Review eval_score must be between 0.0 and 1.0.")
        try:
            signature = base64.b64decode(self.signature_b64, validate=True)
        except ValueError as exc:
            raise ValueError("Review signature is not valid base64.") from exc
        if len(signature) != 64:
            raise ValueError("Ed25519 review signatures must contain exactly 64 bytes.")

    @classmethod
    def sign(
        cls,
        *,
        agent_name: str,
        agent_version: str,
        reviewer: str,
        rating: int,
        eval_score: float,
        signing_provider: SigningProvider,
        comment: str = "",
        signed_at: str | None = None,
    ) -> SignedAgentReview:
        if signing_provider.algorithm != "ed25519":
            raise ValueError("Review schema 1.0 supports Ed25519 signing only.")
        identity = PublisherIdentity(
            algorithm=signing_provider.algorithm,
            key_id=signing_provider.key_id,
            public_key_b64=base64.b64encode(signing_provider.public_key_bytes()).decode("ascii"),
        )
        timestamp = signed_at or datetime.now(timezone.utc).isoformat()
        unsigned = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "agent_name": agent_name,
            "agent_version": agent_version,
            "reviewer": reviewer,
            "rating": rating,
            "eval_score": eval_score,
            "comment": comment,
            "signed_at": timestamp,
            "reviewer_identity": identity.to_dict(),
        }
        signature = signing_provider.sign(canonical_json(unsigned))
        return cls(
            agent_name=agent_name,
            agent_version=agent_version,
            reviewer=reviewer,
            rating=rating,
            eval_score=eval_score,
            comment=comment,
            signed_at=timestamp,
            reviewer_identity=identity,
            signature_b64=base64.b64encode(signature).decode("ascii"),
        )

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "reviewer": self.reviewer,
            "rating": self.rating,
            "eval_score": self.eval_score,
            "comment": self.comment,
            "signed_at": self.signed_at,
            "reviewer_identity": self.reviewer_identity.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "signature_b64": self.signature_b64}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SignedAgentReview:
        identity = value.get("reviewer_identity")
        if not isinstance(identity, dict):
            raise ValueError("Review reviewer_identity must be an object.")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            agent_name=str(value["agent_name"]),
            agent_version=str(value["agent_version"]),
            reviewer=str(value["reviewer"]),
            rating=int(value["rating"]),
            eval_score=float(value["eval_score"]),
            comment=str(value.get("comment", "")),
            signed_at=str(value["signed_at"]),
            reviewer_identity=PublisherIdentity.from_dict(identity),
            signature_b64=str(value["signature_b64"]),
        )

    def verify(self, trusted_keys: Mapping[str, bytes] | None = None) -> tuple[bool, bool]:
        public_key = self.reviewer_identity.public_key_bytes
        signature = base64.b64decode(self.signature_b64, validate=True)
        integrity_valid = verify_signature(
            algorithm=self.reviewer_identity.algorithm,
            public_key_bytes=public_key,
            data=canonical_json(self.unsigned_dict()),
            signature=signature,
        )
        if not integrity_valid or trusted_keys is None:
            return integrity_valid, False
        pinned_key = trusted_keys.get(self.reviewer_identity.key_id)
        return integrity_valid, pinned_key == public_key


@dataclass(frozen=True, slots=True)
class RankedRegistryEntry:
    """A registry entry plus its deterministic quality ranking."""

    entry: RegistryEntry
    score: float
    review_count: int


class FileAgentRegistry:
    """Self-hostable registry backed by a directory and static JSON index."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self._lock = threading.RLock()

    def publish(
        self,
        bundle: str | Path,
        *,
        trusted_keys: Mapping[str, bytes] | None = None,
        allow_untrusted: bool = False,
    ) -> RegistryEntry:
        """Verify and publish one immutable agent version."""
        verification = verify_agent_bundle(bundle, trusted_keys=trusted_keys)
        manifest = verification.require_valid()
        if not verification.trusted and not allow_untrusted:
            raise UntrustedPublisherError(
                "Registry publishing requires a pinned publisher key. "
                "Use allow_untrusted=True only for an intentionally open registry."
            )
        assert verification.bundle_sha256 is not None
        relative = (
            Path("packages")
            / manifest.name
            / manifest.version
            / (f"{manifest.name}-{manifest.version}.agent")
        )
        destination = self.root / relative
        entry = RegistryEntry(
            name=manifest.name,
            version=manifest.version,
            author=manifest.author,
            description=manifest.description,
            tags=manifest.tags,
            publisher_key_id=manifest.publisher.key_id,
            bundle_sha256=verification.bundle_sha256,
            bundle_path=relative.as_posix(),
            published_at=datetime.now(timezone.utc).isoformat(),
            eval_score=manifest.eval_score,
        )

        with self._lock:
            existing = self.get(manifest.name, manifest.version)
            if existing is not None:
                if existing.bundle_sha256 == entry.bundle_sha256:
                    return existing
                raise FileExistsError(
                    f"Registry already contains {manifest.name} {manifest.version}."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
            try:
                shutil.copyfile(bundle, temporary)
                copied = verify_agent_bundle(temporary, trusted_keys=trusted_keys)
                copied.require_valid()
                if copied.bundle_sha256 != entry.bundle_sha256:
                    raise InvalidAgentBundleError(
                        "Bundle changed between verification and registry publication."
                    )
                if verification.trusted and not copied.trusted:
                    raise InvalidAgentBundleError("Copied bundle lost its publisher trust anchor.")
                temporary.replace(destination)
                entries = self.list()
                entries.append(entry)
                self._write_index(entries)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        return entry

    def get(self, name: str, version: str) -> RegistryEntry | None:
        for entry in self.list():
            if entry.name == name and entry.version == version:
                return entry
        return None

    def list(self) -> builtin_list[RegistryEntry]:
        if not self.index_path.exists():
            return []
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise InvalidAgentBundleError("Registry index has an unsupported schema.")
        raw_entries = data.get("agents")
        if not isinstance(raw_entries, list):
            raise InvalidAgentBundleError("Registry index agents must be a list.")
        return sorted(
            [RegistryEntry.from_dict(_as_mapping(item, "registry entry")) for item in raw_entries],
            key=lambda item: (item.name, item.version),
        )

    def bundle_path(self, name: str, version: str) -> Path:
        entry = self.get(name, version)
        if entry is None:
            raise KeyError(f"Unknown registry agent: {name} {version}")
        path = (self.root / entry.bundle_path).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise InvalidAgentBundleError("Registry index contains an unsafe bundle path.") from exc
        return path

    def add_review(
        self,
        review: SignedAgentReview,
        *,
        trusted_keys: Mapping[str, bytes] | None = None,
        allow_untrusted: bool = False,
    ) -> Path:
        if self.get(review.agent_name, review.agent_version) is None:
            raise KeyError(
                f"Cannot review unpublished agent {review.agent_name} {review.agent_version}."
            )
        integrity_valid, trusted = review.verify(trusted_keys)
        if not integrity_valid:
            raise InvalidAgentBundleError("Review signature is invalid.")
        if not trusted and not allow_untrusted:
            raise UntrustedPublisherError("Registry reviews require a pinned reviewer key.")
        data = _pretty_json(review.to_dict())
        review_id = hashlib.sha256(canonical_json(review.to_dict())).hexdigest()
        path = (
            self.root / "reviews" / review.agent_name / review.agent_version / f"{review_id}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Review id collision at {path}.")
        path.write_bytes(data)
        return path

    def reviews(self, name: str, version: str) -> builtin_list[SignedAgentReview]:
        _validate_registry_component(name, "agent name")
        _validate_registry_component(version, "agent version")
        directory = self.root / "reviews" / name / version
        if not directory.exists():
            return []
        result: builtin_list[SignedAgentReview] = []
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            review = SignedAgentReview.from_dict(_as_mapping(data, "review"))
            integrity_valid, _ = review.verify()
            if not integrity_valid:
                raise InvalidAgentBundleError(f"Stored review signature is invalid: {path.name}")
            result.append(review)
        return result

    def ranked(self) -> builtin_list[RankedRegistryEntry]:
        ranked: builtin_list[RankedRegistryEntry] = []
        for entry in self.list():
            reviews = self.reviews(entry.name, entry.version)
            review_scores = [(review.eval_score + review.rating / 5.0) / 2.0 for review in reviews]
            if entry.eval_score is None:
                score = fmean(review_scores) if review_scores else 0.0
            elif review_scores:
                score = 0.7 * entry.eval_score + 0.3 * fmean(review_scores)
            else:
                score = entry.eval_score
            ranked.append(RankedRegistryEntry(entry, round(score, 12), len(reviews)))
        return sorted(ranked, key=lambda item: (-item.score, item.entry.name, item.entry.version))

    def _write_index(self, entries: builtin_list[RegistryEntry]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "agents": [
                entry.to_dict()
                for entry in sorted(entries, key=lambda item: (item.name, item.version))
            ],
        }
        temporary = self.index_path.with_name(f".{self.index_path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(_pretty_json(data))
            temporary.replace(self.index_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise InvalidAgentBundleError(f"Expected {label} to be a JSON object.")
    return value


def _pretty_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _validate_registry_component(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", value):
        raise ValueError(f"Unsafe {label}: {value!r}.")
    if value.endswith((".", " ")) or value.split(".", 1)[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }:
        raise ValueError(f"Unsafe {label}: {value!r}.")
