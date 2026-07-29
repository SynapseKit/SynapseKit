"""Signed, portable ``.agent`` bundle creation and installation.

The archive is deliberately inert: verification and installation never import
or execute files from the bundle. A caller must supply a sandbox runner before
an installed agent can be used.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from ..audit.serializer import canonical_json
from ..audit.signer import Ed25519SigningProvider, SigningProvider, verify_signature
from .errors import InvalidAgentBundleError, SandboxRequiredError, UntrustedPublisherError

AGENT_BUNDLE_FORMAT = "synapsekit-agent"
AGENT_BUNDLE_SCHEMA_VERSION = "1.0"
MANIFEST_PATH = "manifest.json"
SIGNATURE_PATH = "signature.ed25519"
INSTALL_METADATA_PATH = ".synapsekit-install.json"
MAX_ARCHIVE_FILES = 2048
MAX_FILE_SIZE = 64 * 1024 * 1024
MAX_ARCHIVE_SIZE = 256 * 1024 * 1024

_FIXED_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


@dataclass(frozen=True, slots=True)
class AgentBundleFile:
    """A content file authenticated by an agent manifest."""

    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        _validate_archive_path(self.path)
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError(f"Invalid SHA-256 digest for {self.path!r}.")
        if self.size < 0 or self.size > MAX_FILE_SIZE:
            raise ValueError(f"Invalid file size for {self.path!r}: {self.size}.")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AgentBundleFile:
        return cls(path=str(value["path"]), sha256=str(value["sha256"]), size=int(value["size"]))


@dataclass(frozen=True, slots=True)
class PublisherIdentity:
    """Public identity embedded in a signed agent bundle."""

    algorithm: str
    key_id: str
    public_key_b64: str

    def __post_init__(self) -> None:
        if self.algorithm != "ed25519":
            raise ValueError(f"Unsupported publisher algorithm: {self.algorithm!r}.")
        if not self.key_id:
            raise ValueError("Publisher key_id must not be empty.")
        try:
            public_key = base64.b64decode(self.public_key_b64, validate=True)
        except ValueError as exc:
            raise ValueError("Publisher public key is not valid base64.") from exc
        if len(public_key) != 32:
            raise ValueError("Ed25519 public keys must contain exactly 32 bytes.")

    @property
    def public_key_bytes(self) -> bytes:
        return base64.b64decode(self.public_key_b64, validate=True)

    def to_dict(self) -> dict[str, str]:
        return {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "public_key_b64": self.public_key_b64,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PublisherIdentity:
        return cls(
            algorithm=str(value["algorithm"]),
            key_id=str(value["key_id"]),
            public_key_b64=str(value["public_key_b64"]),
        )


@dataclass(frozen=True, slots=True)
class AgentManifest:
    """Versioned metadata and content inventory for an agent bundle."""

    name: str
    version: str
    author: str
    publisher: PublisherIdentity
    files: tuple[AgentBundleFile, ...]
    description: str = ""
    entrypoint: str | None = None
    tags: tuple[str, ...] = ()
    eval_score: float | None = None
    memory_format: str | None = None
    router_path: str | None = None
    format: str = AGENT_BUNDLE_FORMAT
    schema_version: str = AGENT_BUNDLE_SCHEMA_VERSION
    sandbox_required_for_untrusted: bool = True

    def __post_init__(self) -> None:
        _validate_component(self.name, "Agent name")
        _validate_component(self.version, "Agent version")
        if not self.author.strip():
            raise ValueError("Agent author must not be empty.")
        if self.format != AGENT_BUNDLE_FORMAT:
            raise ValueError(f"Unsupported agent bundle format: {self.format!r}.")
        if self.schema_version != AGENT_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported agent bundle schema: {self.schema_version!r}.")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Agent manifest contains duplicate file paths.")
        if len(paths) != len({path.casefold() for path in paths}):
            raise ValueError("Agent manifest contains case-insensitive path collisions.")
        if MANIFEST_PATH in paths or SIGNATURE_PATH in paths:
            raise ValueError("Control files must not appear in the manifest file inventory.")
        if "README.md" not in paths:
            raise ValueError("Agent bundles must include README.md.")
        if not any(path.startswith("evals/") for path in paths):
            raise ValueError("Agent bundles must include at least one file under evals/.")
        if self.entrypoint is not None:
            entry_file, separator, symbol = self.entrypoint.partition(":")
            _validate_archive_path(entry_file)
            if (
                not separator
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", symbol)
                or entry_file not in paths
            ):
                raise ValueError("Entrypoint must use FILE:SYMBOL and reference a bundled file.")
        if self.eval_score is not None and not 0.0 <= self.eval_score <= 1.0:
            raise ValueError("eval_score must be between 0.0 and 1.0.")
        if self.router_path is not None and not self.router_path.startswith("router/"):
            raise ValueError("router_path must be located under router/.")
        if any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise ValueError("Agent tags must be non-empty strings.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "tags": list(self.tags),
            "evals": {"path": "evals/", "score": self.eval_score},
            "protocols": {"memory": self.memory_format},
            "router": {"path": self.router_path},
            "sandbox": {"required_for_untrusted": self.sandbox_required_for_untrusted},
            "publisher": self.publisher.to_dict(),
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AgentManifest:
        publisher = _mapping(value.get("publisher"), "publisher")
        evals = _mapping(value.get("evals", {}), "evals")
        protocols = _mapping(value.get("protocols", {}), "protocols")
        router = _mapping(value.get("router", {}), "router")
        sandbox = _mapping(value.get("sandbox", {}), "sandbox")
        raw_files = value.get("files")
        if not isinstance(raw_files, list):
            raise ValueError("Manifest files must be a list.")
        files = tuple(AgentBundleFile.from_dict(_mapping(item, "file entry")) for item in raw_files)
        raw_tags = value.get("tags", [])
        if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
            raise ValueError("Manifest tags must be a list of strings.")
        raw_score = evals.get("score")
        raw_entrypoint = value.get("entrypoint")
        raw_memory = protocols.get("memory")
        raw_router = router.get("path")
        sandbox_required = sandbox.get("required_for_untrusted", True)
        if not isinstance(sandbox_required, bool):
            raise ValueError("sandbox.required_for_untrusted must be a boolean.")
        if not sandbox_required:
            raise ValueError("Agent bundle schema 1.0 requires sandboxing for untrusted agents.")
        return cls(
            format=str(value.get("format", "")),
            schema_version=str(value.get("schema_version", "")),
            name=str(value["name"]),
            version=str(value["version"]),
            author=str(value["author"]),
            description=str(value.get("description", "")),
            entrypoint=None if raw_entrypoint is None else str(raw_entrypoint),
            tags=tuple(raw_tags),
            eval_score=None if raw_score is None else float(raw_score),
            memory_format=None if raw_memory is None else str(raw_memory),
            router_path=None if raw_router is None else str(raw_router),
            sandbox_required_for_untrusted=sandbox_required,
            publisher=PublisherIdentity.from_dict(publisher),
            files=files,
        )


@dataclass(frozen=True, slots=True)
class AgentBundleVerification:
    """Integrity and trust result for a bundle verification."""

    integrity_valid: bool
    trusted: bool
    errors: tuple[str, ...]
    manifest: AgentManifest | None = None
    bundle_sha256: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the archive is structurally valid and untampered."""
        return self.integrity_valid

    def require_valid(self) -> AgentManifest:
        if not self.integrity_valid or self.manifest is None:
            details = "; ".join(self.errors) or "unknown verification error"
            raise InvalidAgentBundleError(details)
        return self.manifest

    def require_trusted(self) -> AgentManifest:
        manifest = self.require_valid()
        if not self.trusted:
            raise UntrustedPublisherError(
                f"Publisher key {manifest.publisher.key_id!r} was not independently trusted."
            )
        return manifest


class AgentSandbox(Protocol):
    """Execution boundary required for installed agent bundles."""

    async def run(
        self,
        agent_path: Path,
        manifest: AgentManifest,
        prompt: str,
        **kwargs: Any,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class InstalledAgent:
    """An installed, inert agent bundle that can be bound to a sandbox."""

    path: Path
    manifest: AgentManifest
    trusted: bool
    bundle_sha256: str

    @property
    def sandbox_required(self) -> bool:
        return not self.trusted and self.manifest.sandbox_required_for_untrusted

    async def run(
        self,
        prompt: str,
        *,
        sandbox: AgentSandbox | None = None,
        **kwargs: Any,
    ) -> Any:
        if sandbox is None:
            raise SandboxRequiredError(
                "Installed agent bundles are inert. Supply an AgentSandbox to execute this agent."
            )
        return await sandbox.run(self.path, self.manifest, prompt, **kwargs)


def pack_agent(
    source: str | Path,
    output: str | Path,
    *,
    name: str,
    version: str,
    author: str,
    signing_provider: SigningProvider | bytes,
    description: str = "",
    entrypoint: str | None = None,
    tags: tuple[str, ...] | list[str] = (),
    eval_score: float | None = None,
    key_id: str | None = None,
) -> Path:
    """Pack and sign a source directory as a deterministic ``.agent`` archive."""
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Agent source directory does not exist: {source_path}")
    if output_path.suffix != ".agent":
        raise ValueError("Agent bundle output must use the .agent extension.")
    if _is_relative_to(output_path, source_path):
        raise ValueError("Agent bundle output must be outside the source directory.")

    provider = (
        Ed25519SigningProvider(signing_provider, key_id=key_id)
        if isinstance(signing_provider, bytes)
        else signing_provider
    )
    if provider.algorithm != "ed25519":
        raise ValueError("Agent bundle schema 1.0 supports Ed25519 signing only.")

    payloads = _collect_source_files(source_path)
    inventory = tuple(
        AgentBundleFile(path=path, sha256=_sha256(data), size=len(data))
        for path, data in payloads.items()
    )
    paths = set(payloads)
    manifest = AgentManifest(
        name=name,
        version=version,
        author=author,
        description=description,
        entrypoint=entrypoint,
        tags=tuple(tags),
        eval_score=eval_score,
        memory_format="ump/1.0" if any(path.startswith("memory/") for path in paths) else None,
        router_path="router/" if any(path.startswith("router/") for path in paths) else None,
        publisher=PublisherIdentity(
            algorithm=provider.algorithm,
            key_id=provider.key_id,
            public_key_b64=base64.b64encode(provider.public_key_bytes()).decode("ascii"),
        ),
        files=inventory,
    )
    manifest_bytes = canonical_json(manifest.to_dict())
    signature = provider.sign(manifest_bytes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as archive:
        _write_zip_entry(archive, MANIFEST_PATH, _pretty_json(manifest.to_dict()))
        _write_zip_entry(archive, SIGNATURE_PATH, signature)
        for path, data in payloads.items():
            _write_zip_entry(archive, path, data)
    return output_path


def verify_agent_bundle(
    bundle: str | Path,
    *,
    trusted_keys: Mapping[str, bytes] | None = None,
) -> AgentBundleVerification:
    """Verify archive safety, content hashes, signature, and optional trust pin."""
    try:
        data = _read_bundle_bytes(Path(bundle))
    except (InvalidAgentBundleError, OSError) as exc:
        return AgentBundleVerification(False, False, (str(exc),))
    return _verify_agent_bundle_data(data, trusted_keys=trusted_keys)


def _verify_agent_bundle_data(
    data: bytes,
    *,
    trusted_keys: Mapping[str, bytes] | None = None,
) -> AgentBundleVerification:
    digest = _sha256(data)
    try:
        manifest, signature = _read_verified_archive(data)
        embedded_key = manifest.publisher.public_key_bytes
        if not verify_signature(
            algorithm=manifest.publisher.algorithm,
            public_key_bytes=embedded_key,
            data=canonical_json(manifest.to_dict()),
            signature=signature,
        ):
            raise InvalidAgentBundleError("Manifest signature is invalid.")

        trusted = False
        if trusted_keys is not None and manifest.publisher.key_id in trusted_keys:
            pinned_key = trusted_keys[manifest.publisher.key_id]
            if pinned_key != embedded_key:
                raise InvalidAgentBundleError(
                    f"Pinned key for {manifest.publisher.key_id!r} does not match the bundle."
                )
            trusted = verify_signature(
                algorithm=manifest.publisher.algorithm,
                public_key_bytes=pinned_key,
                data=canonical_json(manifest.to_dict()),
                signature=signature,
            )
        return AgentBundleVerification(True, trusted, (), manifest, digest)
    except (
        InvalidAgentBundleError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        return AgentBundleVerification(False, False, (str(exc),), bundle_sha256=digest)


def unpack_agent(
    bundle: str | Path,
    output: str | Path,
    *,
    trusted_keys: Mapping[str, bytes] | None = None,
    require_trusted: bool = False,
) -> AgentManifest:
    """Verify and atomically unpack a bundle without executing its contents."""
    data = _read_bundle_bytes(Path(bundle))
    verification = _verify_agent_bundle_data(data, trusted_keys=trusted_keys)
    manifest = verification.require_trusted() if require_trusted else verification.require_valid()
    _extract_archive(data, Path(output))
    return manifest


def install_agent(
    bundle: str | Path,
    *,
    install_root: str | Path | None = None,
    trusted_keys: Mapping[str, bytes] | None = None,
    require_trusted: bool = False,
) -> InstalledAgent:
    """Verify and install a bundle under ``NAME/VERSION`` without executing it."""
    data = _read_bundle_bytes(Path(bundle))
    verification = _verify_agent_bundle_data(data, trusted_keys=trusted_keys)
    manifest = verification.require_trusted() if require_trusted else verification.require_valid()
    assert verification.bundle_sha256 is not None
    root = (
        Path(install_root) if install_root is not None else Path.home() / ".synapsekit" / "agents"
    )
    destination = root / manifest.name / manifest.version
    metadata = {
        "bundle_sha256": verification.bundle_sha256,
        "publisher_key_id": manifest.publisher.key_id,
        "trusted": verification.trusted,
        "sandbox_required": not verification.trusted and manifest.sandbox_required_for_untrusted,
    }

    if destination.exists():
        metadata_path = destination / INSTALL_METADATA_PATH
        if metadata_path.is_file():
            installed_metadata = _load_json(metadata_path.read_bytes())
            if installed_metadata.get("bundle_sha256") == verification.bundle_sha256:
                return InstalledAgent(
                    destination,
                    manifest,
                    verification.trusted,
                    verification.bundle_sha256,
                )
        raise FileExistsError(f"A different agent is already installed at {destination}.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{manifest.name}-", dir=destination.parent))
    try:
        _extract_archive_contents(data, staging)
        (staging / INSTALL_METADATA_PATH).write_bytes(_pretty_json(metadata))
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return InstalledAgent(
        destination,
        manifest,
        verification.trusted,
        verification.bundle_sha256,
    )


def _read_verified_archive(data: bytes) -> tuple[AgentManifest, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_FILES + 2:
            raise InvalidAgentBundleError("Agent bundle contains too many files.")
        names: set[str] = set()
        casefolded_names: set[str] = set()
        total_size = 0
        for info in infos:
            name = _validate_archive_path(info.filename)
            if name in names:
                raise InvalidAgentBundleError(f"Duplicate archive entry: {name}")
            names.add(name)
            casefolded_name = name.casefold()
            if casefolded_name in casefolded_names:
                raise InvalidAgentBundleError(f"Case-insensitive archive path collision: {name}")
            casefolded_names.add(casefolded_name)
            if info.is_dir():
                raise InvalidAgentBundleError(f"Explicit directory entries are not allowed: {name}")
            if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                raise InvalidAgentBundleError(f"Symbolic links are not allowed: {name}")
            if info.file_size > MAX_FILE_SIZE:
                raise InvalidAgentBundleError(f"Archive entry exceeds size limit: {name}")
            total_size += info.file_size
            if total_size > MAX_ARCHIVE_SIZE:
                raise InvalidAgentBundleError("Agent bundle exceeds the total size limit.")
        if MANIFEST_PATH not in names or SIGNATURE_PATH not in names:
            raise InvalidAgentBundleError(
                "Agent bundle is missing manifest.json or signature.ed25519."
            )

        manifest = AgentManifest.from_dict(_load_json(archive.read(MANIFEST_PATH)))
        expected = {item.path: item for item in manifest.files}
        actual = names - {MANIFEST_PATH, SIGNATURE_PATH}
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            extra = sorted(actual - set(expected))
            raise InvalidAgentBundleError(
                f"Manifest inventory mismatch (missing={missing}, extra={extra})."
            )
        for name, item in expected.items():
            data = archive.read(name)
            if len(data) != item.size or _sha256(data) != item.sha256:
                raise InvalidAgentBundleError(f"Content hash mismatch for {name!r}.")
        signature = archive.read(SIGNATURE_PATH)
        if len(signature) != 64:
            raise InvalidAgentBundleError("Ed25519 signature must contain exactly 64 bytes.")
        return manifest, signature


def _collect_source_files(source: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    casefolded_paths: set[str] = set()
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise InvalidAgentBundleError(f"Symbolic links are not allowed: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        _validate_archive_path(relative)
        reserved_paths = {MANIFEST_PATH, SIGNATURE_PATH, INSTALL_METADATA_PATH}
        if relative.casefold() in {item.casefold() for item in reserved_paths}:
            raise InvalidAgentBundleError(f"Reserved bundle path in source: {relative}")
        casefolded = relative.casefold()
        if casefolded in casefolded_paths:
            raise InvalidAgentBundleError(f"Case-insensitive source path collision: {relative}")
        casefolded_paths.add(casefolded)
        data = path.read_bytes()
        if len(data) > MAX_FILE_SIZE:
            raise InvalidAgentBundleError(f"Source file exceeds size limit: {relative}")
        payloads[relative] = data
    if len(payloads) > MAX_ARCHIVE_FILES:
        raise InvalidAgentBundleError("Agent source contains too many files.")
    if sum(len(data) for data in payloads.values()) > MAX_ARCHIVE_SIZE:
        raise InvalidAgentBundleError("Agent source exceeds the total size limit.")
    return payloads


def _extract_archive(bundle: bytes, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Unpack destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        _extract_archive_contents(bundle, staging)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _extract_archive_contents(bundle: bytes, output: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(_validate_archive_path(info.filename))
            destination = output.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def _write_zip_entry(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def _load_json(data: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key!r}.")
            result[key] = value
        return result

    value = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Manifest {label} must be an object.")
    return value


def _validate_component(value: str, label: str) -> None:
    reserved_name = value.split(".", 1)[0].upper()
    if (
        not _SAFE_COMPONENT.fullmatch(value)
        or value in {".", ".."}
        or value.endswith((".", " "))
        or reserved_name in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(f"{label} contains unsafe characters: {value!r}.")


def _validate_archive_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise InvalidAgentBundleError(f"Unsafe archive path: {value!r}")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        raise InvalidAgentBundleError(f"Unsafe archive path: {value!r}")
    for part in raw_parts:
        reserved_name = part.split(".", 1)[0].upper()
        if ":" in part or part.endswith((".", " ")) or reserved_name in _WINDOWS_RESERVED_NAMES:
            raise InvalidAgentBundleError(f"Unsafe archive path: {value!r}")
    return path.as_posix()


def _pretty_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bundle_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise InvalidAgentBundleError(f"Agent bundle does not exist: {path}")
    if path.stat().st_size > MAX_ARCHIVE_SIZE + 4 * 1024 * 1024:
        raise InvalidAgentBundleError("Agent bundle exceeds the compressed size limit.")
    return path.read_bytes()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
