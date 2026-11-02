"""Check pinned uv.lock package records against a local Dependabot snapshot.

Run from the repository root with:
    uv run --locked --group dev python scripts/check_dependency_security.py

The check is deterministic and does not access the network. It uses ``packaging``
from the already-resolved development environment to interpret PEP 440 versions.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import tomllib
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO_ROOT / "tests" / "fixtures" / "dependency_security_snapshot.json"
LOCK_PATH = REPO_ROOT / "uv.lock"
EXPECTED_EXCEPTION_IDS = {127, 128, 129}
_SPECIFIER_RE = re.compile(r"^(===|==|!=|~=|<=|>=|<|>|=)\s*(\S+)$")


def normalize_package_name(name: str) -> str:
    """Normalize names as PyPI/uv do, including case and separator aliases."""
    return str(canonicalize_name(name))


def _parse_vulnerable_range(raw_range: str) -> SpecifierSet:
    """Translate GitHub's single-equals syntax and parse every range clause."""
    if not isinstance(raw_range, str) or not raw_range.strip():
        raise ValueError("vulnerable version range must be a non-empty string")

    clauses: list[str] = []
    for raw_clause in raw_range.split(","):
        match = _SPECIFIER_RE.fullmatch(raw_clause.strip())
        if match is None:
            raise ValueError(f"invalid range clause {raw_clause!r}")
        operator, version = match.groups()
        if operator == "=":
            operator = "=="
        clauses.append(f"{operator}{version}")

    try:
        return SpecifierSet(",".join(clauses))
    except InvalidSpecifier as error:
        raise ValueError(f"invalid vulnerable version range {raw_range!r}: {error}") from error


def version_is_vulnerable(version: str, raw_range: str) -> bool:
    """Return whether a version falls in a raw GitHub advisory range."""
    try:
        parsed_version = Version(version)
    except InvalidVersion as error:
        raise ValueError(f"invalid version {version!r}: {error}") from error
    return _parse_vulnerable_range(raw_range).contains(parsed_version, prereleases=True)


def _check_exception_ledger(snapshot: dict[str, Any], failures: list[str]) -> None:
    if "dismissed_exception_ledger" not in snapshot:
        failures.append("dismissed_exception_ledger is required to track known exceptions")
        return

    exceptions = snapshot["dismissed_exception_ledger"]
    if not isinstance(exceptions, list):
        failures.append("dismissed_exception_ledger must be a list")
        return

    ids: list[int] = []
    for exception in exceptions:
        if not isinstance(exception, dict):
            failures.append("dismissed exception ledger contains a non-object entry")
            continue
        exception_id = exception.get("id")
        if not isinstance(exception_id, int) or isinstance(exception_id, bool):
            failures.append(f"dismissed exception ID must be an integer, got {exception_id!r}")
            continue
        ids.append(exception_id)

    if set(ids) != EXPECTED_EXCEPTION_IDS or len(ids) != len(EXPECTED_EXCEPTION_IDS):
        failures.append(
            "dismissed exception ledger must contain exactly IDs 127, 128, and 129 "
            f"(found {sorted(ids)})"
        )

    for exception in exceptions:
        if not isinstance(exception, dict):
            continue
        label = f"dismissed exception #{exception.get('id')}"
        if exception.get("state") != "dismissed":
            failures.append(f"{label} must remain marked dismissed")
        if exception.get("first_patched_version") is not None:
            failures.append(f"{label} must not claim a patched version")
        if exception.get("disposition") != "dismissed_exception_not_fixed":
            failures.append(f"{label} must be labeled as an exception, not fixed")
        if exception.get("current_dependency_cap") != "<1.0.0":
            failures.append(f"{label} must record the current Chroma cap <1.0.0")
        if exception.get("adapter_mode") != "embedded-only":
            failures.append(f"{label} must record the embedded-only adapter rationale")
        if not exception.get("rationale"):
            failures.append(f"{label} must include its rationale")


def check_security_policy(snapshot: dict[str, Any], lock_data: dict[str, Any]) -> list[str]:
    """Return all deterministic snapshot, candidate-floor, and lock violations."""
    failures: list[str] = []
    if not isinstance(snapshot, dict):
        return ["security snapshot must be a JSON object"]
    if not isinstance(lock_data, dict):
        return ["uv.lock must parse to a TOML object"]

    metadata = snapshot.get("metadata")
    alerts = snapshot.get("alerts")
    floors = snapshot.get("candidate_floors")
    records = lock_data.get("package")

    if not isinstance(metadata, dict):
        failures.append("snapshot metadata must be an object")
        metadata = {}
    if not isinstance(alerts, list):
        failures.append("snapshot alerts must be a list")
        alerts = []
    if not isinstance(floors, dict):
        failures.append("snapshot candidate_floors must be an object")
        floors = {}
    if not isinstance(records, list):
        failures.append("uv.lock must contain a package array")
        records = []

    expected_count = metadata.get("open_alert_count")
    expected_open_ids = metadata.get("expected_open_alert_ids")
    expected_open_id_set: set[int] | None = None
    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        failures.append("snapshot metadata must contain an integer open_alert_count")
    elif expected_count != len(alerts):
        failures.append(
            f"snapshot declares {expected_count} open alerts but contains {len(alerts)}"
        )
    if not isinstance(expected_open_ids, list) or any(
        not isinstance(alert_id, int) or isinstance(alert_id, bool)
        for alert_id in expected_open_ids
    ):
        failures.append("snapshot metadata must contain the frozen expected_open_alert_ids list")
    elif len(expected_open_ids) != len(set(expected_open_ids)):
        failures.append("snapshot expected_open_alert_ids contains duplicates")
    else:
        expected_open_id_set = set(expected_open_ids)
        if expected_count != len(expected_open_ids):
            failures.append("snapshot expected_open_alert_ids does not match open_alert_count")

    normalized_floors: dict[str, str] = {}
    for raw_name, floor in floors.items():
        if not isinstance(raw_name, str) or not isinstance(floor, str):
            failures.append(f"invalid candidate floor entry {raw_name!r}: {floor!r}")
            continue
        name = normalize_package_name(raw_name)
        if name in normalized_floors and normalized_floors[name] != floor:
            failures.append(f"conflicting candidate floors normalize to package {name!r}")
        normalized_floors[name] = floor

    seen_ids: set[Any] = set()
    alert_packages: set[str] = set()
    valid_alerts: list[dict[str, Any]] = []
    for alert in alerts:
        if not isinstance(alert, dict):
            failures.append("snapshot alerts contains a non-object entry")
            continue
        alert_id = alert.get("id")
        if not isinstance(alert_id, int) or isinstance(alert_id, bool):
            failures.append(f"snapshot alert ID must be an integer, got {alert_id!r}")
            continue
        label = f"alert #{alert_id}"
        if alert_id in seen_ids:
            failures.append(f"duplicate alert ID {alert_id}")
        seen_ids.add(alert_id)
        if alert.get("state") != "open":
            failures.append(f"{label} must be marked open")
        raw_name = alert.get("package")
        if not isinstance(raw_name, str):
            failures.append(f"{label} has no valid raw package name")
            continue
        normalized_name = normalize_package_name(raw_name)
        alert_packages.add(normalized_name)
        valid_alerts.append(alert)

    if expected_open_id_set is not None and seen_ids != expected_open_id_set:
        missing_ids = sorted(expected_open_id_set - seen_ids)
        unexpected_ids = sorted(seen_ids - expected_open_id_set)
        failures.append(
            "alert IDs do not match the frozen baseline "
            f"(missing={missing_ids}, unexpected={unexpected_ids})"
        )

    for name in sorted(alert_packages - normalized_floors.keys()):
        failures.append(f"snapshot has no candidate floor for alerted package {name!r}")
    for name in sorted(normalized_floors.keys() - alert_packages):
        failures.append(f"candidate floor for {name!r} has no open alert in the snapshot")

    _check_exception_ledger(snapshot, failures)
    exception_ids: set[int] = set()
    exception_ledger = snapshot.get("dismissed_exception_ledger")
    if isinstance(exception_ledger, list):
        exception_ids = {
            item["id"]
            for item in exception_ledger
            if isinstance(item, dict)
            and isinstance(item.get("id"), int)
            and not isinstance(item.get("id"), bool)
        }
    for alert_id in sorted(seen_ids & exception_ids):
        failures.append(
            f"alert ID {alert_id} appears in both the open inventory and exception ledger"
        )

    indexed_records = [
        (index, record)
        for index, record in enumerate(records, start=1)
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    ]
    records_by_name: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, record in indexed_records:
        records_by_name.setdefault(normalize_package_name(record["name"]), []).append(
            (index, record)
        )

    for alert in valid_alerts:
        alert_id = alert.get("id")
        raw_name = alert["package"]
        normalized_name = normalize_package_name(raw_name)
        label = f"alert #{alert_id} ({raw_name})"
        floor = normalized_floors.get(normalized_name)
        floor_version: Version | None = None
        if floor is None:
            failures.append(f"{label}: candidate floor is missing")
        else:
            try:
                floor_version = Version(floor)
            except InvalidVersion as error:
                failures.append(f"{label}: invalid candidate floor {floor!r}: {error}")

        raw_range = alert.get("vulnerable_version_range")
        specifiers: SpecifierSet | None = None
        try:
            specifiers = _parse_vulnerable_range(raw_range)
        except (TypeError, ValueError) as error:
            failures.append(f"{label}: invalid vulnerable range {raw_range!r}: {error}")

        patched_text = alert.get("first_patched_version")
        patched_version: Version | None = None
        if not isinstance(patched_text, str) or not patched_text:
            failures.append(f"{label}: first patched version is missing")
        else:
            try:
                patched_version = Version(patched_text)
            except InvalidVersion as error:
                failures.append(f"{label}: invalid first patched version {patched_text!r}: {error}")

        if floor_version is not None:
            if specifiers is not None and specifiers.contains(floor_version, prereleases=True):
                failures.append(
                    f"{label}: candidate floor {floor} is inside vulnerable range {raw_range!r}"
                )
            if patched_version is not None and floor_version < patched_version:
                failures.append(
                    f"{label}: candidate floor {floor} is below first patched version "
                    f"{patched_text}"
                )

        matching_records = records_by_name.get(normalized_name, [])
        if not matching_records:
            failures.append(f"{label}: no uv.lock package record")
            continue

        for record_index, record in matching_records:
            record_label = f"{label}, uv.lock package record #{record_index}"
            version_text = record.get("version")
            if not isinstance(version_text, str) or not version_text:
                failures.append(f"{record_label}: package version is missing")
                continue
            try:
                locked_version = Version(version_text)
            except InvalidVersion as error:
                failures.append(f"{record_label}: invalid locked version {version_text!r}: {error}")
                continue

            if floor_version is not None and locked_version < floor_version:
                failures.append(
                    f"{record_label}: locked version {version_text} is below candidate floor {floor}"
                )
            if specifiers is not None and specifiers.contains(locked_version, prereleases=True):
                failures.append(
                    f"{record_label}: locked version {version_text} is inside vulnerable "
                    f"range {raw_range!r}"
                )

    return failures


def main() -> int:
    """Run the repository snapshot against every matching uv.lock record."""
    try:
        snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        lock_data = tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        print(f"ERROR: unable to read security snapshot or uv.lock: {error}", file=sys.stderr)
        return 2

    alerts = snapshot.get("alerts", []) if isinstance(snapshot, dict) else []
    floors = snapshot.get("candidate_floors", {}) if isinstance(snapshot, dict) else {}
    exceptions = (
        snapshot.get("dismissed_exception_ledger", []) if isinstance(snapshot, dict) else []
    )
    alert_count = len(alerts) if isinstance(alerts, list) else "invalid"
    floor_count = len(floors) if isinstance(floors, dict) else "invalid"
    print(
        f"Checking {alert_count} open alerts against {floor_count} candidate floors; "
        "all matching uv.lock records are evaluated."
    )
    if exceptions:
        exception_count = len(exceptions) if isinstance(exceptions, list) else "invalid"
        print(
            f"Dismissed exceptions: {exception_count} (recorded separately; not treated as fixed)."
        )

    failures = check_security_policy(snapshot, lock_data)
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        print(f"FAILED: {len(failures)} security policy failure(s).")
        return 1

    print(
        f"PASS: {alert_count} open alerts meet their candidate floors in all matching lock records."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
