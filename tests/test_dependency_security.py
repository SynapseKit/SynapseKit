"""Unit tests for the offline Dependabot lockfile security gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from packaging.version import Version
from scripts.check_dependency_security import (
    check_security_policy,
    main,
    normalize_package_name,
    version_is_vulnerable,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO_ROOT / "tests" / "fixtures" / "dependency_security_snapshot.json"
EXPECTED_FLOORS = {
    "pillow": "13.0.0",
    "pypdf": "6.21.4",
    "authlib": "1.9.1",
    "filelock": "4.2.1",
    "aiohttp": "3.14.4",
    "gitpython": "3.2.1",
    "oauthlib": "4.0.0",
    "pyasn1": "0.6.5",
    "joserfc": "1.7.6",
    "certifi": "2026.10.19",
    "dnspython": "2.9.0",
    "setuptools": "84.1.0",
    "virtualenv": "21.14.13",
    "opentelemetry-api": "1.45.2",
}
EXPECTED_ALERT_IDS = (
    144,
    145,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
    260,
    261,
    262,
    263,
    264,
    265,
    266,
    267,
    268,
    269,
    270,
    271,
    272,
    273,
    274,
    275,
    276,
    277,
)


def _alert(
    *,
    alert_id: int = 1,
    package: str = "demo_pkg",
    vulnerable_range: str = "< 2.0.0",
    first_patched: str = "2.0.0",
) -> dict[str, object]:
    return {
        "id": alert_id,
        "state": "open",
        "package": package,
        "vulnerable_version_range": vulnerable_range,
        "first_patched_version": first_patched,
        "ghsa": f"GHSA-test-{alert_id}",
        "cve": None,
        "summary": "Synthetic test advisory",
    }


def _snapshot(alerts: list[dict[str, object]], floors: dict[str, str]) -> dict[str, object]:
    return {
        "metadata": {
            "open_alert_count": len(alerts),
            "expected_open_alert_ids": [alert["id"] for alert in alerts],
        },
        "candidate_floors": floors,
        "alerts": alerts,
        "dismissed_exception_ledger": [
            {
                "id": alert_id,
                "state": "dismissed",
                "first_patched_version": None,
                "disposition": "dismissed_exception_not_fixed",
                "current_dependency_cap": "<1.0.0",
                "adapter_mode": "embedded-only",
                "rationale": "Synthetic exception fixture.",
            }
            for alert_id in (127, 128, 129)
        ],
    }


def test_normalizes_package_aliases_for_lock_matching() -> None:
    assert normalize_package_name("GitPython") == "gitpython"
    assert normalize_package_name("Gitpython") == "gitpython"
    assert normalize_package_name("My_Pkg.Name") == "my-pkg-name"


@pytest.mark.parametrize(
    ("version", "raw_range", "expected"),
    [
        ("1.2.3", "= 1.2.3", True),
        ("1.2.4", "= 1.2.3", False),
        ("1.0.0", ">= 1.0.0, < 2.0.0", True),
        ("1.9.9", ">= 1.0.0, < 2.0.0", True),
        ("2.0.0", ">= 1.0.0, < 2.0.0", False),
        ("1.0.0", "> 1.0.0, <= 2.0.0", False),
        ("2.0.0", "> 1.0.0, <= 2.0.0", True),
    ],
)
def test_interprets_github_equality_and_range_boundaries(
    version: str, raw_range: str, expected: bool
) -> None:
    assert version_is_vulnerable(version, raw_range) is expected


def test_reports_an_alert_package_missing_from_the_lock() -> None:
    snapshot = _snapshot([_alert()], {"demo-pkg": "2.0.0"})

    failures = check_security_policy(snapshot, {"package": []})

    assert any(
        "alert #1" in failure and "no uv.lock package record" in failure for failure in failures
    )


def test_rejects_a_candidate_floor_inside_the_alert_range() -> None:
    snapshot = _snapshot([_alert()], {"demo-pkg": "1.9.9"})
    lock = {"package": [{"name": "demo-pkg", "version": "2.0.0"}]}

    failures = check_security_policy(snapshot, lock)

    assert any("alert #1" in failure and "candidate floor 1.9.9" in failure for failure in failures)


def test_checks_every_lock_record_for_the_alert_package() -> None:
    snapshot = _snapshot([_alert()], {"demo-pkg": "2.0.0"})
    lock = {
        "package": [
            {"name": "demo-pkg", "version": "2.0.0", "source": {"registry": "one"}},
            {"name": "DEMO_PKG", "version": "1.9.9", "source": {"registry": "two"}},
        ]
    }

    failures = check_security_policy(snapshot, lock)

    assert any("alert #1" in failure and "1.9.9" in failure for failure in failures)


def test_rejects_duplicate_alert_ids() -> None:
    snapshot = _snapshot(
        [_alert(alert_id=7), _alert(alert_id=7, package="DEMO-PKG")],
        {"demo-pkg": "2.0.0"},
    )

    failures = check_security_policy(
        snapshot, {"package": [{"name": "demo-pkg", "version": "2.0.0"}]}
    )

    assert any("duplicate alert ID 7" in failure for failure in failures)


def test_rejects_an_unrecognized_id_replacing_a_baseline_alert() -> None:
    snapshot = _snapshot([_alert(alert_id=7)], {"demo-pkg": "2.0.0"})
    snapshot["alerts"][0]["id"] = 999999
    lock = {"package": [{"name": "demo-pkg", "version": "2.0.0"}]}

    failures = check_security_policy(snapshot, lock)

    assert any("alert IDs do not match the frozen baseline" in failure for failure in failures)


def test_requires_the_dismissed_exception_ledger() -> None:
    snapshot = _snapshot([_alert()], {"demo-pkg": "2.0.0"})
    del snapshot["dismissed_exception_ledger"]
    lock = {"package": [{"name": "demo-pkg", "version": "2.0.0"}]}

    failures = check_security_policy(snapshot, lock)

    assert any("dismissed_exception_ledger is required" in failure for failure in failures)


def test_reports_a_null_exception_ledger_without_raising() -> None:
    snapshot = _snapshot([_alert()], {"demo-pkg": "2.0.0"})
    snapshot["dismissed_exception_ledger"] = None
    lock = {"package": [{"name": "demo-pkg", "version": "2.0.0"}]}

    failures = check_security_policy(snapshot, lock)

    assert any("dismissed_exception_ledger must be a list" in failure for failure in failures)


@pytest.mark.parametrize("invalid_id", [127.0, []])
def test_rejects_noninteger_exception_ids_without_raising(invalid_id: object) -> None:
    snapshot = _snapshot([_alert()], {"demo-pkg": "2.0.0"})
    snapshot["dismissed_exception_ledger"][0]["id"] = invalid_id
    lock = {"package": [{"name": "demo-pkg", "version": "2.0.0"}]}

    failures = check_security_policy(snapshot, lock)

    assert any("exception ID must be an integer" in failure for failure in failures)


def test_main_returns_an_error_for_a_nonlist_exception_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = _snapshot([_alert()], {"demo-pkg": "2.0.0"})
    snapshot["dismissed_exception_ledger"] = 7
    snapshot_path = tmp_path / "snapshot.json"
    lock_path = tmp_path / "uv.lock"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    lock_path.write_text('[[package]]\nname = "demo-pkg"\nversion = "2.0.0"\n', encoding="utf-8")
    monkeypatch.setattr("scripts.check_dependency_security.SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr("scripts.check_dependency_security.LOCK_PATH", lock_path)

    result = main()

    assert result == 1
    assert "dismissed_exception_ledger must be a list" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected"),
    [
        ("alerts", 7, "snapshot alerts must be a list"),
        ("candidate_floors", 7, "snapshot candidate_floors must be an object"),
    ],
)
def test_main_returns_an_error_for_malformed_snapshot_collections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    invalid_value: object,
    expected: str,
) -> None:
    snapshot = _snapshot([_alert()], {"demo-pkg": "2.0.0"})
    snapshot[field] = invalid_value
    snapshot_path = tmp_path / "snapshot.json"
    lock_path = tmp_path / "uv.lock"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    lock_path.write_text('[[package]]\nname = "demo-pkg"\nversion = "2.0.0"\n', encoding="utf-8")
    monkeypatch.setattr("scripts.check_dependency_security.SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr("scripts.check_dependency_security.LOCK_PATH", lock_path)

    result = main()

    assert result == 1
    assert expected in capsys.readouterr().out


def test_repository_snapshot_covers_all_open_alerts_and_separate_exceptions() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    alerts = snapshot["alerts"]
    metadata = snapshot["metadata"]
    alert_ids = [alert["id"] for alert in alerts]
    package_names = {normalize_package_name(alert["package"]) for alert in alerts}

    assert metadata["repository"] == "SynapseKit/SynapseKit"
    assert metadata["immutable_main_sha"] == "b4cea59023ddf329caf6d20f650814fc8efd346a"
    assert metadata["inventory_date"] == "2026-11-01"
    assert metadata["open_alert_count"] == 99
    assert len(alerts) == 99
    assert len(alert_ids) == len(set(alert_ids))
    assert tuple(sorted(alert_ids)) == EXPECTED_ALERT_IDS
    assert metadata["expected_open_alert_ids"] == list(EXPECTED_ALERT_IDS)
    assert set(snapshot["candidate_floors"]) == package_names
    assert snapshot["candidate_floors"] == EXPECTED_FLOORS

    for alert in alerts:
        floor = snapshot["candidate_floors"][normalize_package_name(alert["package"])]
        assert not version_is_vulnerable(floor, alert["vulnerable_version_range"]), alert["id"]
        assert Version(floor) >= Version(alert["first_patched_version"]), alert["id"]

    exceptions = snapshot["dismissed_exception_ledger"]
    assert {exception["id"] for exception in exceptions} == {127, 128, 129}
    assert not (set(alert_ids) & {exception["id"] for exception in exceptions})
    assert all(exception["state"] == "dismissed" for exception in exceptions)
    assert all(exception["first_patched_version"] is None for exception in exceptions)
    assert all(
        exception["disposition"] == "dismissed_exception_not_fixed" for exception in exceptions
    )
    assert {exception["current_dependency_cap"] for exception in exceptions} == {"<1.0.0"}
    assert {exception["adapter_mode"] for exception in exceptions} == {"embedded-only"}
