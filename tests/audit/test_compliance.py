"""Tests for the EU AI Act record-keeping mode (synapsekit.audit.compliance, #954).

Every detection test is paired with its positive control on the same
object: the untampered bundle, verified with pinned keys, must reach MATCH
first, so a verifier that rejected everything could not make these tests
pass.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

import pytest

from synapsekit.audit import (
    AuditTracer,
    EventKind,
    PIIRedactor,
    RegexDetector,
    RemoteBiometricIdentificationProfile,
    RetentionPolicy,
    SigningPolicy,
    Verdict,
    VerifiableAgent,
    end_agent_lifetime,
    export_audit_bundle,
    record_biometric_use,
    verify,
)
from synapsekit.audit.compliance import PURPOSES, validate_purposes
from synapsekit.audit.trace import compute_payload_hash, compute_record_hash
from synapsekit.audit.types import GENESIS_HASH

from .conftest import (
    dump_trace_lines,
    load_trace_lines,
    manifest_keys_as_trusted,
    read_zip_entries,
    write_zip_entries,
)

UTC = timezone.utc
ALL = list(PURPOSES)


class FakeAgent:
    """Hand-written fake covering sync, async, async-generator and failing methods."""

    def __init__(self) -> None:
        self.llm = FakeLLM()

    def run(self, task: str) -> str:
        return f"done:{task}"

    async def search(self, query: str) -> list[str]:
        return [query]

    async def stream_tokens(self, n: int):
        for i in range(n):
            yield f"t{i}"

    async def decide(self, option: str) -> str:
        raise RuntimeError(f"cannot decide {option}")


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return prompt.upper()


def _agent(**kwargs: Any) -> VerifiableAgent:
    kwargs.setdefault("compliance", "eu_ai_act")
    kwargs.setdefault("retention", RetentionPolicy(role="deployer"))
    return VerifiableAgent(FakeAgent(), **kwargs)


async def _exercise(v: VerifiableAgent) -> None:
    v.run("a")
    await v.search("q")
    async for _ in v.stream_tokens(2):
        pass
    with pytest.raises(RuntimeError):
        await v.decide("x")


def _payload(record: Any) -> dict[str, Any]:
    return json.loads(json.dumps(record.to_dict()))["payload"]


async def _bundle(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    v = _agent()
    await _exercise(v)
    await end_agent_lifetime(v)
    path = export_audit_bundle(v.tracer.drain(), SigningPolicy.ed25519(), tmp_path / "b.zip")
    return path, manifest_keys_as_trusted(path)


# --------------------------------------------------------------------------
# A modified log fails verification (positive control: untampered → MATCH)
# --------------------------------------------------------------------------


def _edit_output(lines: list[dict]) -> None:
    lines[1]["payload"]["output"] = "something else"


def _edit_purposes(lines: list[dict]) -> None:
    lines[1]["payload"]["eu_ai_act"]["purposes"] = ["12(2)(a)"]


def _edit_retention(lines: list[dict]) -> None:
    lines[0]["payload"]["eu_ai_act"]["retention"]["months"] = 1


def _edit_and_rehash_chain(lines: list[dict]) -> None:
    """An attacker who rewrites a payload AND recomputes every hash in the chain."""
    lines[1]["payload"]["output"] = "forged"
    prev = GENESIS_HASH
    for rec in lines:
        rec["payload_hash"] = compute_payload_hash(rec["payload"])
        rec["prev_hash"] = prev
        rec["hash"] = compute_record_hash(
            event_id=rec["event_id"],
            parent_id=rec["parent_id"],
            run_id=rec["run_id"],
            kind=rec["kind"],
            actor=rec["actor"],
            payload_hash=rec["payload_hash"],
            timestamp=datetime.fromisoformat(rec["timestamp"]),
            prev_hash=prev,
            schema_version=rec["schema_version"],
            redaction_policy_hash=rec["redaction_policy_hash"],
            redaction_status=rec["redaction_status"],
        )
        prev = rec["hash"]


@pytest.mark.parametrize(
    "tamper", [_edit_output, _edit_purposes, _edit_retention, _edit_and_rehash_chain]
)
async def test_modified_log_fails_verify(tmp_path: Path, tamper) -> None:
    path, keys = await _bundle(tmp_path)
    assert verify(path, trusted_keys=keys).verdict == Verdict.MATCH  # positive control

    entries = read_zip_entries(path)
    lines = load_trace_lines(entries)
    tamper(lines)
    dump_trace_lines(entries, lines)
    write_zip_entries(path, entries)
    assert verify(path, trusted_keys=keys).verdict == Verdict.DRIFT


@pytest.mark.parametrize("drop", ["last", "middle", "first"])
async def test_truncated_log_fails_verify(tmp_path: Path, drop: str) -> None:
    path, keys = await _bundle(tmp_path)
    assert verify(path, trusted_keys=keys).verdict == Verdict.MATCH  # positive control

    entries = read_zip_entries(path)
    lines = load_trace_lines(entries)
    index = {"last": len(lines) - 1, "middle": len(lines) // 2, "first": 0}[drop]
    del lines[index]
    dump_trace_lines(entries, lines)
    write_zip_entries(path, entries)
    result = verify(path, trusted_keys=keys)
    assert result.verdict == Verdict.DRIFT
    if drop == "last":
        # The tail can only be caught by the signed manifest record_count.
        assert any("record_count" in e or "declares" in e for e in result.errors)


# --------------------------------------------------------------------------
# Retention policy
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("role", "article"), [("provider", "19(1)"), ("deployer", "26(6)")])
@pytest.mark.parametrize("months", [1, 2, 3, 4, 5])
def test_retention_shorter_than_six_months_is_refused(role: str, article: str, months: int) -> None:
    with pytest.raises(ValueError, match=rf"Article {re.escape(article)}"):
        RetentionPolicy(role=role, months=months)  # type: ignore[arg-type]


@pytest.mark.parametrize(("role", "article"), [("provider", "19(1)"), ("deployer", "26(6)")])
@pytest.mark.parametrize("months", [6, 7, 12, 120])
def test_retention_of_six_months_or_more_is_accepted(role: str, article: str, months: int) -> None:
    policy = RetentionPolicy(role=role, months=months)  # type: ignore[arg-type]
    assert policy.months == months
    assert policy.legal_basis == f"Regulation (EU) 2024/1689, Article {article}"


def test_retention_period_defaults_to_six_months() -> None:
    assert RetentionPolicy(role="provider").months == 6
    assert RetentionPolicy.deployer().months == 6
    assert RetentionPolicy.provider().role == "provider"


async def test_shorter_retention_with_cited_law_is_accepted_and_recorded() -> None:
    law = "  Member State act No. 1 of 2027, Article 5  "
    policy = RetentionPolicy(role="deployer", months=3, overriding_law=law)
    assert policy.overriding_law == law.strip()
    v = _agent(retention=policy)
    start = _payload(v.tracer.records[0])["eu_ai_act"]
    assert start["event"] == "agent_lifetime_started"
    assert start["retention"]["months"] == 3
    assert start["retention"]["overriding_law"] == law.strip()
    assert start["retention"]["legal_basis"].endswith("Article 26(6)")
    assert start["mode"] == "eu_ai_act"
    assert start["regulation"] == "Regulation (EU) 2024/1689"


@pytest.mark.parametrize("law", ["", "   ", "\n\t"])
def test_empty_overriding_law_is_refused(law: str) -> None:
    with pytest.raises(ValueError, match="cite a law"):
        RetentionPolicy(role="provider", months=3, overriding_law=law)


def test_compliance_without_retention_says_what_is_missing() -> None:
    with pytest.raises(ValueError, match=r"RetentionPolicy\(role='provider'\)"):
        VerifiableAgent(FakeAgent(), compliance="eu_ai_act")


# --------------------------------------------------------------------------
# Closed by construction: anything outside the closed sets is rejected
# --------------------------------------------------------------------------

_BAD_ROLES: list[Any] = [
    None, True, 0, 1.5, "", "Provider", "PROVIDER", " provider", "provider\n",
    "deployer ", "operator", "user", b"provider", ["provider"], ("deployer",), object(),
]  # fmt: skip


@pytest.mark.parametrize("role", _BAD_ROLES)
def test_unknown_role_is_refused(role: Any) -> None:
    with pytest.raises(ValueError, match="role must be"):
        RetentionPolicy(role=role)


@pytest.mark.parametrize("months", [True, False, 6.0, "6", None, b"6", [6]])
def test_months_of_wrong_type_is_refused(months: Any) -> None:
    with pytest.raises(TypeError, match="months must be an int"):
        RetentionPolicy(role="provider", months=months, overriding_law="X law")


@pytest.mark.parametrize("months", [0, -1, -6])
def test_months_below_one_is_refused_even_with_cited_law(months: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RetentionPolicy(role="provider", months=months, overriding_law="X law")


@pytest.mark.parametrize("law", [1, b"law", ["law"], True])
def test_overriding_law_of_wrong_type_is_refused(law: Any) -> None:
    with pytest.raises(TypeError, match="overriding_law must be a str"):
        RetentionPolicy(role="provider", months=3, overriding_law=law)


@pytest.mark.parametrize(
    "mode",
    ["EU_AI_ACT", "eu_ai_act ", "eu-ai-act", "", "gdpr", True, 1, b"eu_ai_act", ["eu_ai_act"]],
)
def test_unknown_compliance_mode_is_refused(mode: Any) -> None:
    with pytest.raises(ValueError, match="compliance must be one of"):
        VerifiableAgent(FakeAgent(), compliance=mode, retention=RetentionPolicy(role="provider"))


@pytest.mark.parametrize(
    "extra",
    [
        {"retention": RetentionPolicy(role="provider")},
        {"purposes": {EventKind.ERROR: ["12(2)(a)"]}},
        {"purposes": {}},
        {"profile": RemoteBiometricIdentificationProfile()},
    ],
)
def test_compliance_options_without_compliance_are_refused(extra: dict[str, Any]) -> None:
    tracer = AuditTracer()
    with pytest.raises(ValueError, match="only apply with compliance"):
        VerifiableAgent(FakeAgent(), tracer=tracer, **extra)
    assert len(tracer) == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"retention": {"role": "provider"}}, "retention must be a RetentionPolicy"),
        ({"profile": {"annex_iii_point": "1(a)"}}, "profile must be"),
        ({"purposes": [("ERROR", ["12(2)(a)"])]}, "purposes must be a mapping"),
    ],
)
def test_compliance_options_of_wrong_type_are_refused(kwargs: dict[str, Any], message: str) -> None:
    kwargs.setdefault("retention", RetentionPolicy(role="provider"))
    tracer = AuditTracer()
    with pytest.raises(TypeError, match=message):
        VerifiableAgent(FakeAgent(), tracer=tracer, compliance="eu_ai_act", **kwargs)
    assert len(tracer) == 0  # nothing is logged by a construction that failed


_KIND_POOL: list[Any] = [
    *list(EventKind), *[k.value for k in EventKind],
    "LLM", "llm_call", "Error", "ERROR ", "NEW_KIND", "", 1, None, b"ERROR", ("ERROR",),
]  # fmt: skip
_VALUE_POOL: list[Any] = [
    ["12(2)(a)"], ["12(2)(b)", "12(2)(c)"], ("12(2)(c)", "12(2)(a)"), ALL,
    [], ["12(2)(d)"], ["12(2)(a)", "12(2)(a)"], [None], ["12(2)(A)"], ["12(2)(a) "],
    "12(2)(a)", b"12(2)(a)", {"12(2)(a)"}, None, 1, [["12(2)(a)"]],
]  # fmt: skip


class _StrSubclass(str):
    pass


@pytest.mark.parametrize(
    ("purposes", "error"),
    [
        ({"ERROR": "12(2)(a)"}, TypeError),
        ({"ERROR": b"12(2)(a)"}, TypeError),
        ({_StrSubclass("ERROR"): ["12(2)(a)"]}, ValueError),
    ],
)
def test_purposes_rejection_type(purposes: dict[Any, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        validate_purposes(purposes)


@pytest.mark.parametrize("override", ["CUSTOM_KIND", "decision", "", 1, None, b"DECISION"])
def test_kind_override_outside_the_taxonomy_is_refused(override: Any) -> None:
    tracer = AuditTracer()
    with pytest.raises(ValueError, match=r"kind_overrides\['run'\]"):
        _agent(tracer=tracer, kind_overrides={"run": override})
    assert len(tracer) == 0


async def test_kind_override_inside_the_taxonomy_is_tagged_with_its_purposes() -> None:
    v = _agent(
        kind_overrides={"run": "DECISION", "search": EventKind.MEMORY_READ},
        purposes={EventKind.DECISION: ["12(2)(c)"]},
    )
    assert v.run("a") == "done:a"
    await v.search("q")
    tags = [(r.kind, _payload(r)["eu_ai_act"]["purposes"]) for r in v.tracer.records[1:]]
    assert tags == [("DECISION", ["12(2)(c)"]), ("MEMORY_READ", ALL)]


def _oracle_ok(mapping: dict[Any, Any]) -> bool:
    kinds = {k.value for k in EventKind}
    for key, values in mapping.items():
        value = key.value if isinstance(key, EventKind) else key
        if type(value) is not str or value not in kinds:
            return False
        if not isinstance(values, list | tuple) or not values:
            return False
        if any(not isinstance(p, str) or p not in PURPOSES for p in values):
            return False
        if len(set(values)) != len(values):
            return False
    return True


def test_purposes_closed_set_generative() -> None:
    """Every mapping built from the pools is accepted iff the oracle accepts it."""
    rng = random.Random(954)
    accepted = rejected = 0
    for _ in range(3000):
        mapping: dict[Any, Any] = {}
        for _ in range(rng.randint(0, 4)):
            key = rng.choice(_KIND_POOL)
            try:
                mapping[key] = rng.choice(_VALUE_POOL)
            except TypeError:  # unhashable key: cannot even be a dict key
                continue
        if _oracle_ok(mapping):
            result = validate_purposes(mapping)
            accepted += 1
            assert set(result) == {k.value for k in EventKind}
            for key, values in mapping.items():
                value = key.value if isinstance(key, EventKind) else key
                assert set(result[value]) == set(values)
        else:
            with pytest.raises((TypeError, ValueError)):
                validate_purposes(mapping)
            rejected += 1
    assert accepted > 100 and rejected > 100  # both branches really exercised


# --------------------------------------------------------------------------
# Lifetime and tagging
# --------------------------------------------------------------------------


async def test_every_event_is_tagged_and_lifetime_is_bracketed() -> None:
    v = _agent(instrument_attrs=["llm"])
    await _exercise(v)
    v.wrapped.llm.generate("hi")  # internal call, captured via instrument_attrs
    first_end = await end_agent_lifetime(v)
    assert first_end is not None
    assert await end_agent_lifetime(v) is None  # recorded once only
    v.run("after")  # recording never stops silently

    records = v.tracer.records
    payloads = [_payload(r) for r in records]
    assert payloads[0]["eu_ai_act"]["event"] == "agent_lifetime_started"
    ended = [p for p in payloads if p["eu_ai_act"].get("event") == "agent_lifetime_ended"]
    assert len(ended) == 1
    kinds = {r.kind for r in records}
    assert {"TOOL_CALL", "TOOL_RESULT", "RETRIEVAL", "ERROR", "LLM_CALL", "LLM_RESPONSE"} <= kinds
    for payload in payloads:
        assert payload["eu_ai_act"]["purposes"] == ALL
    AuditTracer.verify_chain(list(records))


async def test_purposes_can_be_narrowed_per_event_kind() -> None:
    v = _agent(purposes={EventKind.ERROR: ["12(2)(a)"], "TOOL_CALL": ("12(2)(c)", "12(2)(b)")})
    await _exercise(v)
    by_kind: dict[str, list[list[str]]] = {}
    for record in v.tracer.records:
        by_kind.setdefault(record.kind, []).append(_payload(record)["eu_ai_act"]["purposes"])
    assert by_kind["ERROR"] == [["12(2)(a)"]]
    assert by_kind["TOOL_CALL"] == [["12(2)(b)", "12(2)(c)"]]  # canonical order
    assert by_kind["RETRIEVAL"] == [ALL]
    start = _payload(v.tracer.records[0])["eu_ai_act"]
    assert start["purpose_map"]["ERROR"] == ["12(2)(a)"]


async def test_result_records_get_the_purposes_of_their_own_kind() -> None:
    v = _agent(
        purposes={EventKind.TOOL_RESULT: ["12(2)(b)"], EventKind.LLM_RESPONSE: ["12(2)(c)"]},
        instrument_attrs=["llm"],
    )
    v.run("a")
    v.wrapped.llm.generate("x")
    tags = {r.kind: _payload(r)["eu_ai_act"]["purposes"] for r in v.tracer.records[1:]}
    assert tags == {
        "TOOL_CALL": ALL,
        "TOOL_RESULT": ["12(2)(b)"],
        "LLM_CALL": ALL,
        "LLM_RESPONSE": ["12(2)(c)"],
    }


async def test_without_compliance_payloads_are_unchanged() -> None:
    v = VerifiableAgent(FakeAgent())
    v.run("a")
    records = v.tracer.records
    assert len(records) == 2  # no lifecycle event
    assert _payload(records[0]) == {"method": "run", "args": ["a"], "kwargs": {}}
    assert _payload(records[1]) == {
        "method": "run",
        "args": ["a"],
        "kwargs": {},
        "output": "done:a",
        "status": "ok",
    }
    with pytest.raises(ValueError, match="not created with compliance"):
        await end_agent_lifetime(v)


async def test_redactor_cannot_rewrite_the_recorded_legal_basis() -> None:
    redactor = PIIRedactor([RegexDetector("ARTICLE", r"Article \d+\(\d+\)|12\(2\)\([abc]\)")])
    v = _agent(redactor=redactor)
    v.run("see Article 26(6)")
    payloads = [_payload(r) for r in v.tracer.records]
    assert payloads[0]["eu_ai_act"]["retention"]["legal_basis"].endswith("Article 26(6)")
    assert payloads[1]["args"] == ["see [REDACTED:ARTICLE]"]  # call data still redacted
    for payload in payloads:
        assert payload["eu_ai_act"]["purposes"] == ALL


# --------------------------------------------------------------------------
# retain_until / may_delete
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "months", "expected"),
    [
        (datetime(2026, 8, 31, 12, tzinfo=UTC), 6, datetime(2027, 2, 28, 12, tzinfo=UTC)),
        (datetime(2027, 8, 31, 12, tzinfo=UTC), 6, datetime(2028, 2, 29, 12, tzinfo=UTC)),
        (datetime(2026, 3, 31, tzinfo=UTC), 6, datetime(2026, 9, 30, tzinfo=UTC)),
        (datetime(2026, 10, 13, tzinfo=UTC), 6, datetime(2027, 4, 13, tzinfo=UTC)),
        (datetime(2026, 12, 15, tzinfo=UTC), 12, datetime(2027, 12, 15, tzinfo=UTC)),
        (datetime(2026, 7, 1, tzinfo=UTC), 30, datetime(2029, 1, 1, tzinfo=UTC)),
    ],
)
def test_retain_until_uses_calendar_months(
    start: datetime, months: int, expected: datetime
) -> None:
    assert RetentionPolicy(role="provider", months=months).retain_until(start) == expected


class _NoOffset(tzinfo):
    def utcoffset(self, dt: datetime | None) -> None:
        return None

    def dst(self, dt: datetime | None) -> None:
        return None


def test_retain_until_rejects_naive_and_non_datetime() -> None:
    policy = RetentionPolicy(role="provider")
    with pytest.raises(ValueError, match="timezone-aware"):
        policy.retain_until(datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="timezone-aware"):
        policy.retain_until(datetime(2026, 1, 1, tzinfo=_NoOffset()))
    with pytest.raises(ValueError, match="overflows"):
        RetentionPolicy(role="provider", months=1, overriding_law="X law").retain_until(
            datetime(9999, 12, 1, tzinfo=UTC)
        )
    with pytest.raises(TypeError, match="must be a datetime"):
        policy.retain_until("2026-01-01T00:00:00+00:00")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="overflows"):
        RetentionPolicy(role="provider", months=12 * 10_000).retain_until(
            datetime(2026, 1, 1, tzinfo=UTC)
        )


def test_may_delete_only_after_the_whole_period() -> None:
    tracer = AuditTracer()
    record = tracer.record(EventKind.SYSTEM_EVENT, {}, timestamp=datetime(2026, 8, 31, tzinfo=UTC))
    policy = RetentionPolicy(role="deployer")
    boundary = datetime(2027, 2, 28, tzinfo=UTC)
    assert not policy.may_delete(record, now=boundary - timedelta(microseconds=1))
    assert policy.may_delete(record, now=boundary)
    with pytest.raises(ValueError, match="timezone-aware"):
        policy.may_delete(record, now=datetime(2030, 1, 1))


# --------------------------------------------------------------------------
# Annex III point 1(a) profile (Article 12(3) fields)
# --------------------------------------------------------------------------

T0 = datetime(2028, 1, 10, 9, 0, tzinfo=UTC)
T1 = datetime(2028, 1, 10, 9, 5, tzinfo=UTC)


def _rbi_agent() -> VerifiableAgent:
    return _agent(
        retention=RetentionPolicy(role="provider"), profile=RemoteBiometricIdentificationProfile()
    )


async def test_biometric_use_records_article_12_3_fields(tmp_path: Path) -> None:
    v = _rbi_agent()
    record = await record_biometric_use(
        v,
        start=T0,
        end=T1,
        reference_database="watchlist-2028-01",
        matched_input={"frame": 42},
        verified_by=["officer A", "officer B"],
    )
    fields = _payload(record)["annex_iii_point_1a"]
    assert fields == {
        "use_start": T0.isoformat(),
        "use_end": T1.isoformat(),
        "reference_database": "watchlist-2028-01",
        "matched_input": {"frame": 42},
        "verified_by": ["officer A", "officer B"],
    }
    assert _payload(record)["eu_ai_act"]["purposes"] == ALL
    assert _payload(v.tracer.records[0])["eu_ai_act"]["profile"]["annex_iii_point"] == "1(a)"
    path = export_audit_bundle(v.tracer.drain(), SigningPolicy.ed25519(), tmp_path / "rbi.zip")
    assert verify(path, trusted_keys=manifest_keys_as_trusted(path)).verdict == Verdict.MATCH


async def test_biometric_use_without_match_needs_no_verifier() -> None:
    v = _rbi_agent()
    record = await record_biometric_use(
        v, start=T0, end=T0, reference_database="db", matched_input=None, verified_by=[]
    )
    assert _payload(record)["annex_iii_point_1a"]["matched_input"] is None


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"end": T0 - timedelta(seconds=1)}, ValueError, "earlier than its start"),
        ({"start": datetime(2028, 1, 10)}, ValueError, "timezone-aware"),
        ({"end": "2028-01-10"}, TypeError, "must be a datetime"),
        ({"reference_database": ""}, ValueError, "12\\(3\\)\\(b\\)"),
        ({"reference_database": "  "}, ValueError, "12\\(3\\)\\(b\\)"),
        ({"reference_database": None}, ValueError, "12\\(3\\)\\(b\\)"),
        ({"verified_by": []}, ValueError, "12\\(3\\)\\(d\\)"),
        ({"verified_by": "officer A"}, TypeError, "list or tuple"),
        ({"verified_by": ["officer A", ""]}, ValueError, "non-empty name"),
        ({"verified_by": ["officer A", None]}, ValueError, "non-empty name"),
        ({"verified_by": ["officer A", "  "]}, ValueError, "non-empty name"),
        ({"matched_input": 0, "verified_by": []}, ValueError, "12\\(3\\)\\(d\\)"),
        ({"matched_input": "", "verified_by": []}, ValueError, "12\\(3\\)\\(d\\)"),
        ({"verified_by": ["officer A", " officer A "]}, ValueError, "duplicate"),
    ],
)
async def test_biometric_use_rejects_incomplete_fields(
    kwargs: dict[str, Any], error: type[Exception], message: str
) -> None:
    v = _rbi_agent()
    args: dict[str, Any] = {
        "start": T0,
        "end": T1,
        "reference_database": "db",
        "matched_input": "probe",
        "verified_by": ["officer A", "officer B"],
    }
    args.update(kwargs)
    before = len(v.tracer)
    with pytest.raises(error, match=message):
        await record_biometric_use(v, **args)
    assert len(v.tracer) == before  # nothing half-recorded


async def test_biometric_use_is_redacted_and_stored_stripped() -> None:
    redactor = PIIRedactor([RegexDetector("NAME", r"officer [A-Z]")])
    v = _agent(
        retention=RetentionPolicy(role="provider"),
        profile=RemoteBiometricIdentificationProfile(),
        redactor=redactor,
    )
    record = await record_biometric_use(
        v,
        start=T0,
        end=T1,
        reference_database="  db-1  ",
        matched_input="probe",
        verified_by=[" officer A ", "officer B"],
    )
    fields = _payload(record)["annex_iii_point_1a"]
    assert fields["reference_database"] == "db-1"
    assert fields["verified_by"] == ["[REDACTED:NAME]", "[REDACTED:NAME]"]


async def test_biometric_record_made_inside_a_proxied_call_is_its_child() -> None:
    holder: dict[str, VerifiableAgent] = {}

    class Camera:
        async def identify(self, frame: str) -> str:
            await record_biometric_use(
                holder["v"],
                start=T0,
                end=T1,
                reference_database="db",
                matched_input=frame,
                verified_by=["officer A"],
            )
            return "ok"

    v = VerifiableAgent(
        Camera(),
        compliance="eu_ai_act",
        retention=RetentionPolicy(role="provider"),
        profile=RemoteBiometricIdentificationProfile(),
    )
    holder["v"] = v
    await v.identify("f1")
    by_payload = {next(iter(_payload(r))): r for r in v.tracer.records[1:]}
    assert by_payload["annex_iii_point_1a"].parent_id == by_payload["method"].event_id


async def test_biometric_use_requires_the_profile_and_the_mode() -> None:
    args: dict[str, Any] = {
        "start": T0,
        "end": T1,
        "reference_database": "db",
        "matched_input": None,
        "verified_by": [],
    }
    with pytest.raises(ValueError, match="needs profile"):
        await record_biometric_use(_agent(), **args)
    with pytest.raises(ValueError, match="not created with compliance"):
        await record_biometric_use(VerifiableAgent(FakeAgent()), **args)
    with pytest.raises(TypeError, match="expected a VerifiableAgent"):
        await record_biometric_use(FakeAgent(), **args)  # type: ignore[arg-type]
