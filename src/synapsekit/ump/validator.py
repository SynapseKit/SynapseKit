from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from synapsekit.ump.parser import UMPReader
from synapsekit.ump.types import UMPDocument, UMPFrontmatter, UMPProvenance


@dataclass
class ValidationError:
    field: str
    message: str
    severity: Literal["error", "warning"] = "error"


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)


class UMPValidator:
    """Validate UMP documents against spec constraints."""

    UMP_VERSION: ClassVar[str] = "1.0"
    VALID_TYPES: ClassVar[list[str]] = ["user", "feedback", "project", "reference", "general"]
    VALID_SCOPES: ClassVar[list[str]] = ["global", "project", "session"]
    VALID_VISIBILITIES: ClassVar[list[str]] = ["local", "shared", "team"]

    @classmethod
    def validate(cls, doc: UMPDocument) -> ValidationResult:
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        # Frontmatter validation
        fm_errors, fm_warnings = cls._validate_frontmatter(doc.frontmatter)
        errors.extend(fm_errors)
        warnings.extend(fm_warnings)

        # Body validation
        if not doc.body.strip():
            warnings.append(
                ValidationError(
                    field="body",
                    message="Document body is empty",
                    severity="warning",
                )
            )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    @classmethod
    def validate_file(cls, path: str | Path) -> ValidationResult:
        try:
            doc = UMPReader.read_file(path)
            return cls.validate(doc)
        except Exception as err:
            return ValidationResult(
                is_valid=False,
                errors=[
                    ValidationError(
                        field="file",
                        message=f"Failed to read/parse UMP file: {err}",
                        severity="error",
                    )
                ],
            )

    @classmethod
    def _validate_frontmatter(
        cls, fm: UMPFrontmatter
    ) -> tuple[list[ValidationError], list[ValidationError]]:
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        if fm.ump_version != cls.UMP_VERSION:
            warnings.append(
                ValidationError(
                    field="ump_version",
                    message=f"Expected version '{cls.UMP_VERSION}', got '{fm.ump_version}'",
                    severity="warning",
                )
            )

        if not fm.name.strip():
            warnings.append(
                ValidationError(
                    field="name",
                    message="Document name is empty",
                    severity="warning",
                )
            )

        if fm.type not in cls.VALID_TYPES:
            errors.append(
                ValidationError(
                    field="type",
                    message=f"Invalid type '{fm.type}'. Must be one of {cls.VALID_TYPES}",
                    severity="error",
                )
            )

        if fm.scope not in cls.VALID_SCOPES:
            errors.append(
                ValidationError(
                    field="scope",
                    message=f"Invalid scope '{fm.scope}'. Must be one of {cls.VALID_SCOPES}",
                    severity="error",
                )
            )

        if fm.visibility not in cls.VALID_VISIBILITIES:
            errors.append(
                ValidationError(
                    field="visibility",
                    message=f"Invalid visibility '{fm.visibility}'. Must be one of {cls.VALID_VISIBILITIES}",
                    severity="error",
                )
            )

        # Provenance validation
        prov_errors, prov_warnings = cls._validate_provenance(fm.provenance)
        errors.extend(prov_errors)
        warnings.extend(prov_warnings)

        return errors, warnings

    @classmethod
    def _validate_provenance(
        cls, prov: UMPProvenance
    ) -> tuple[list[ValidationError], list[ValidationError]]:
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        if not prov.authors:
            warnings.append(
                ValidationError(
                    field="provenance.authors",
                    message="No authors specified in provenance",
                    severity="warning",
                )
            )

        return errors, warnings
