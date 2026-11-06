"""Build a value-minimizing W3C PROV-JSON document from audit records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from datetime import timezone
from typing import Any

from .._audit_context import source_ids_from_results
from .serializer import hash_value
from .types import AuditRecord, EventKind

_PREFIX = {
    "prov": "http://www.w3.org/ns/prov#",
    "sk": "https://synapsekit.ai/ns/audit-provenance#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}
_CITATION_KEYS = frozenset({"citation_id", "citation_ids"})
_INPUT_KEYS = frozenset({"input_id", "input_ids"})
_OUTPUT_KEYS = frozenset(
    {"output_id", "output_ids", "result_id", "result_ids", "answer_id", "answer_ids"}
)
_PROVIDER_CALL_KEYS = frozenset({"tool_calls", "function_call"})
_CITATION_MARKER = re.compile(r"\[((?:[^\[\]\r\n]|\[[^\[\]\r\n]{1,128}\]){1,256})\]")
_PROMPT_ROLES = frozenset({"system", "developer", "user", "assistant", "tool", "function", "model"})
_EVENT_KIND_VALUES = frozenset(kind.value for kind in EventKind)


def source_id_hash(source_id: str | int | float) -> str:
    """Return a deterministic SHA-256 digest for an explicitly supplied source ID.

    The identifier itself is not copied into the provenance document, but this
    digest is pseudonymous rather than confidential: SHA-256 is not encryption
    and low-entropy identifiers may be guessed by hashing likely values.
    """
    if isinstance(source_id, bool) or not isinstance(source_id, str | int | float):
        raise TypeError("source_id must be a string or number")
    return hashlib.sha256(str(source_id).encode("utf-8")).hexdigest()


def _qualified_type(name: str) -> dict[str, str]:
    return {"$": name, "type": "xsd:QName"}


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reference_values(value: Any):
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            yield from _reference_values(item)
    elif (
        isinstance(value, str | int | float)
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    ):
        text = str(value)
        if text:
            yield text


def _find_references(
    value: Any,
    keys: frozenset[str],
    *,
    excluded_keys: frozenset[str] = frozenset(),
) -> list[str]:
    found: dict[str, None] = {}

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if isinstance(key, str) and key.lower() in excluded_keys:
                    continue
                if isinstance(key, str) and key.lower() in keys:
                    for reference in _reference_values(child):
                        found.setdefault(reference, None)
                visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, str | bytes | bytearray):
            for child in node:
                visit(child)

    visit(value)
    return list(found)


def _scoped_references(
    payload: Mapping[str, Any],
    keys: frozenset[str],
    section_key: str,
    *,
    excluded_keys: frozenset[str] = frozenset(),
) -> list[str]:
    found: dict[str, None] = {}
    for key, value in payload.items():
        if isinstance(key, str) and key.lower() in keys:
            for reference in _reference_values(value):
                found.setdefault(reference, None)
    for reference in _find_references(payload.get(section_key), keys, excluded_keys=excluded_keys):
        found.setdefault(reference, None)
    return list(found)


def _plain_json(value: Any) -> Any:
    """Unwrap immutable record values for deterministic, content-only hashing."""
    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_plain_json(child) for child in value]
    if isinstance(value, set | frozenset):
        return sorted((_plain_json(child) for child in value), key=repr)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _prompt_messages(value: Any) -> list[tuple[str, str, str | None]]:
    """Extract text messages from actual LLM-call arguments, never their values to export."""
    messages: list[tuple[str, str, str | None]] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            role = node.get("role")
            content = node.get("content")
            if isinstance(role, str):
                normalized_role = role.strip().lower()
                if normalized_role not in _PROMPT_ROLES:
                    normalized_role = "other"
                tool_call_id = node.get("tool_call_id")
                safe_tool_call_id = tool_call_id if isinstance(tool_call_id, str) else None
                if isinstance(content, str):
                    messages.append((normalized_role, content, safe_tool_call_id))
                    return
                if isinstance(content, Sequence) and not isinstance(
                    content, str | bytes | bytearray
                ):
                    for block in content:
                        if not isinstance(block, Mapping) or block.get("type") not in {
                            "text",
                            "input_text",
                        }:
                            continue
                        text = block.get("text")
                        if isinstance(text, str):
                            messages.append((normalized_role, text, safe_tool_call_id))
                    return
            for child in node.values():
                visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, str | bytes | bytearray):
            for child in node:
                visit(child)

    visit(value)
    return messages


def _response_citation_ids(record: AuditRecord) -> list[str]:
    if record.kind != "LLM_RESPONSE":
        return []

    output = record.payload.get("output")
    citations: list[str] = []
    if isinstance(output, str):
        response_texts = [output]
    elif isinstance(output, Mapping):
        for key, value in output.items():
            if isinstance(key, str) and key.lower() in _CITATION_KEYS:
                citations.extend(_reference_values(value))
        content = output.get("content")
        response_texts = [content] if isinstance(content, str) else []
    else:
        response_texts = []
    for text in response_texts:
        for match in _CITATION_MARKER.finditer(text):
            citation = match.group(1).strip()
            if citation:
                citations.append(citation)
    return list(dict.fromkeys(citations))


def _tool_calls_from_response(output: Any) -> list[tuple[str, str, Any]]:
    if not isinstance(output, Mapping):
        return []
    tool_calls = output.get("tool_calls")
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, str | bytes | bytearray):
        return []
    calls: list[tuple[str, str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            continue
        call_id = tool_call.get("id")
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
        if isinstance(call_id, str) and isinstance(name, str) and isinstance(arguments, Mapping):
            calls.append((call_id, name, arguments))
    return calls


def _qualified_id(kind: str, *values: str) -> str:
    return f"sk:{kind}-{_digest(values)}"


def _relation_id(kind: str, *values: str) -> str:
    return _qualified_id(kind, *values)


def _build_provenance_records(records: Sequence[AuditRecord]) -> dict[str, Any]:
    """Build a W3C PROV-JSON graph from ordered ``AuditRecord`` instances.

    Audit records become activities and generated record entities. Actors are
    represented as agents. Only explicit source/input/output identifiers are
    translated to entity references; prompts, tool arguments, retrieved text,
    credentials, and model output values are never copied into the graph.
    """
    document: dict[str, Any] = {
        "prefix": dict(_PREFIX),
        "entity": {},
        "activity": {},
        "agent": {},
        "used": {},
        "wasGeneratedBy": {},
        "wasDerivedFrom": {},
        "wasAssociatedWith": {},
    }

    prior_records: dict[tuple[str, str], str] = {}
    seen_records: set[tuple[str, str]] = set()
    source_entities_by_run_and_hash: dict[tuple[str, str], str] = {}
    generated_by_record: list[tuple[list[str], list[str]]] = []

    for record in records:
        record_key = (record.run_id, record.event_id)
        if record_key in seen_records:
            raise ValueError(f"duplicate event_id {record.event_id!r} in run {record.run_id!r}")
        seen_records.add(record_key)

        activity_id = _qualified_id("activity", record.run_id, record.event_id)
        record_entity_id = _qualified_id("record", record.run_id, record.event_id)
        timestamp = record.timestamp.astimezone(timezone.utc).isoformat()

        document["activity"][activity_id] = {
            "prov:type": _qualified_type("sk:AuditActivity"),
            "sk:eventIdHash": source_id_hash(record.event_id),
            "sk:runIdHash": source_id_hash(record.run_id),
            "sk:kind": record.kind if record.kind in _EVENT_KIND_VALUES else "other",
            "sk:recordHash": record.hash,
            "sk:payloadHash": record.payload_hash,
            "sk:redactionStatus": record.redaction_status,
            "prov:startTime": timestamp,
            "prov:endTime": timestamp,
        }
        document["entity"][record_entity_id] = {
            "prov:type": _qualified_type("sk:AuditRecord"),
            "sk:eventIdHash": source_id_hash(record.event_id),
            "sk:runIdHash": source_id_hash(record.run_id),
            "sk:recordHash": record.hash,
            "sk:payloadHash": record.payload_hash,
            "sk:timestamp": timestamp,
        }

        actor_id = _qualified_id("agent", record.actor)
        document["agent"].setdefault(
            actor_id,
            {
                "prov:type": _qualified_type("prov:Agent"),
                "sk:actorHash": source_id_hash(record.actor),
            },
        )
        association_id = _relation_id("association", activity_id, actor_id)
        document["wasAssociatedWith"][association_id] = {
            "prov:activity": activity_id,
            "prov:agent": actor_id,
        }

        used_entities: dict[str, None] = {}
        parent_entity_id = prior_records.get((record.run_id, record.parent_id or ""))
        if parent_entity_id is not None:
            used_entities.setdefault(parent_entity_id, None)

        def reference_entity(reference: str, role: str, run_id: str) -> str:
            reference_hash = source_id_hash(reference)
            run_hash = source_id_hash(run_id)
            entity_id = _qualified_id("reference", run_hash, reference_hash)
            entity = document["entity"].setdefault(
                entity_id,
                {
                    "prov:type": _qualified_type("sk:ReferencedEntity"),
                    "sk:runIdHash": run_hash,
                    "sk:referenceIdHash": reference_hash,
                    "sk:referenceKinds": [],
                },
            )
            entity[f"sk:{role}IdHash"] = reference_hash
            if role not in entity["sk:referenceKinds"]:
                entity["sk:referenceKinds"].append(role)
            return entity_id

        if record.kind in {"RETRIEVAL", "TOOL_RESULT"}:
            for reference in source_ids_from_results(record.payload.get("output")):
                entity_id = reference_entity(reference, "source", record.run_id)
                source_entities_by_run_and_hash.setdefault(
                    (record.run_id, source_id_hash(reference)), entity_id
                )
                used_entities.setdefault(entity_id, None)

        for reference in _scoped_references(
            record.payload,
            _INPUT_KEYS,
            "input",
            excluded_keys=_PROVIDER_CALL_KEYS,
        ):
            used_entities.setdefault(reference_entity(reference, "input", record.run_id), None)

        generated_entities = [record_entity_id]
        for reference in _scoped_references(
            record.payload,
            _OUTPUT_KEYS,
            "output",
            excluded_keys=_PROVIDER_CALL_KEYS,
        ):
            generated_entities.append(reference_entity(reference, "output", record.run_id))

        for entity_id in used_entities:
            relation_id = _relation_id("usage", activity_id, entity_id)
            document["used"][relation_id] = {
                "prov:activity": activity_id,
                "prov:entity": entity_id,
                "prov:time": timestamp,
            }

        for entity_id in generated_entities:
            generation_id = _relation_id("generation", entity_id, activity_id)
            document["wasGeneratedBy"][generation_id] = {
                "prov:entity": entity_id,
                "prov:activity": activity_id,
                "prov:time": timestamp,
            }

        generated_by_record.append((generated_entities, list(used_entities)))
        prior_records[record_key] = record_entity_id

    for generated_entities, used_entity_ids in generated_by_record:
        for generated_entity in generated_entities:
            for used_entity in used_entity_ids:
                if generated_entity == used_entity:
                    continue
                derivation_id = _relation_id("derivation", generated_entity, used_entity)
                document["wasDerivedFrom"].setdefault(
                    derivation_id,
                    {
                        "prov:generatedEntity": generated_entity,
                        "prov:usedEntity": used_entity,
                    },
                )

    return document


def build_provenance(records: Sequence[AuditRecord]) -> dict[str, Any]:
    """Build a value-minimizing PROV graph with runtime evidence links.

    The base graph retains the stable audit-record schema. Runtime LLM call
    records add content-hashed prompt fragments, source links resolve only
    against identifiers from ``RETRIEVAL`` or ``TOOL_RESULT`` records, and tool
    outputs are linked only when actual tool-call and subsequent tool-message
    records correlate.
    """
    document = _build_provenance_records(records)
    source_entities_by_run_and_hash = {
        (entity["sk:runIdHash"], entity["sk:sourceIdHash"]): entity_id
        for entity_id, entity in document["entity"].items()
        if "sk:sourceIdHash" in entity and "sk:runIdHash" in entity
    }

    def source_entity_for(run_id: str, source_id: str) -> str | None:
        return source_entities_by_run_and_hash.get(
            (source_id_hash(run_id), source_id_hash(source_id))
        )

    source_first_seen_index: dict[tuple[str, str], int] = {}
    llm_calls_by_key: dict[tuple[str, str], tuple[int, AuditRecord]] = {}
    for record_index, record in enumerate(records):
        if record.kind in {"RETRIEVAL", "TOOL_RESULT"}:
            for source_id in source_ids_from_results(record.payload.get("output")):
                source_first_seen_index.setdefault(
                    (record.run_id, source_id_hash(source_id)), record_index
                )
        elif record.kind == "LLM_CALL":
            llm_calls_by_key[(record.run_id, record.event_id)] = (record_index, record)

    trusted_source_entities_by_call: dict[tuple[str, str], dict[str, str]] = {}
    trusted_citation_source_hashes_by_call: dict[tuple[str, str], dict[str, str]] = {}
    unresolvable_citation_ids_by_call: dict[tuple[str, str], set[str]] = {}
    for call_key, (call_index, call) in llm_calls_by_key.items():
        unresolvable_citations = call.payload.get("unresolvable_citation_ids")
        if isinstance(unresolvable_citations, Sequence) and not isinstance(
            unresolvable_citations, str | bytes | bytearray
        ):
            unresolvable_citation_ids_by_call[call_key] = {
                citation for citation in unresolvable_citations if isinstance(citation, str)
            }
        trusted_ids = call.payload.get("trusted_source_ids")
        if isinstance(trusted_ids, Sequence) and not isinstance(
            trusted_ids, str | bytes | bytearray
        ):
            for source_id in _reference_values(trusted_ids):
                source_hash = source_id_hash(source_id)
                first_seen_index = source_first_seen_index.get((call.run_id, source_hash))
                if first_seen_index is None or first_seen_index >= call_index:
                    continue
                source_entity_id = source_entity_for(call.run_id, source_id)
                if source_entity_id is not None:
                    trusted_source_entities_by_call.setdefault(call_key, {})[source_hash] = (
                        source_entity_id
                    )
        trusted_sources = trusted_source_entities_by_call.get(call_key, {})
        trusted_citations = call.payload.get("trusted_citation_ids")
        if isinstance(trusted_citations, Mapping):
            for locator, source_id in trusted_citations.items():
                if not isinstance(locator, str) or not isinstance(source_id, str):
                    continue
                source_hash = source_id_hash(source_id)
                if source_hash in trusted_sources:
                    trusted_citation_source_hashes_by_call.setdefault(call_key, {})[locator] = (
                        source_hash
                    )

    activity_by_key = {
        (record.run_id, record.event_id): _qualified_id("activity", record.run_id, record.event_id)
        for record in records
    }
    record_entity_by_key = {
        (record.run_id, record.event_id): _qualified_id("record", record.run_id, record.event_id)
        for record in records
    }

    def add_used(record: AuditRecord, entity_id: str) -> None:
        activity_id = activity_by_key[(record.run_id, record.event_id)]
        relation_id = _relation_id("usage", activity_id, entity_id)
        document["used"].setdefault(
            relation_id,
            {
                "prov:activity": activity_id,
                "prov:entity": entity_id,
                "prov:time": record.timestamp.astimezone(timezone.utc).isoformat(),
            },
        )

    def add_derived(generated_entity: str, used_entity: str) -> None:
        relation_id = _relation_id("derivation", generated_entity, used_entity)
        document["wasDerivedFrom"].setdefault(
            relation_id,
            {
                "prov:generatedEntity": generated_entity,
                "prov:usedEntity": used_entity,
            },
        )

    tool_result_entities: dict[tuple[str, str], str] = {}
    tool_result_values: dict[tuple[str, str], Any] = {}

    for record_index, record in enumerate(records):
        if record.kind == "LLM_CALL":
            last_user_fragment_id: str | None = None
            for fragment_index, (role, content, tool_call_id) in enumerate(
                _prompt_messages(record.payload)
            ):
                fragment_hash = hash_value(content)
                fragment_id = _qualified_id(
                    "prompt-fragment",
                    record.run_id,
                    record.event_id,
                    str(fragment_index),
                    fragment_hash,
                )
                fragment: dict[str, Any] = {
                    "prov:type": _qualified_type("sk:PromptFragment"),
                    "sk:promptFragmentHash": fragment_hash,
                    "sk:promptRole": role,
                    "sk:fragmentIndex": fragment_index,
                }
                if tool_call_id is not None:
                    fragment["sk:toolCallIdHash"] = source_id_hash(tool_call_id)
                document["entity"].setdefault(fragment_id, fragment)
                add_used(record, fragment_id)
                if role == "user":
                    last_user_fragment_id = fragment_id

            trusted_source_entities = trusted_source_entities_by_call.get(
                (record.run_id, record.event_id), {}
            )
            for source_entity_id in trusted_source_entities.values():
                add_used(record, source_entity_id)
                if last_user_fragment_id is not None:
                    add_derived(last_user_fragment_id, source_entity_id)

        if record.kind == "LLM_RESPONSE":
            citations = _response_citation_ids(record)
            if citations:
                unresolved_hashes: list[str] = []
                parent_call_key = (record.run_id, record.parent_id or "")
                parent_call = llm_calls_by_key.get(parent_call_key)
                trusted_sources = trusted_source_entities_by_call.get(parent_call_key, {})
                if parent_call is None or parent_call[0] >= record_index:
                    trusted_sources = {}
                for citation in citations:
                    citation_source_hash = trusted_citation_source_hashes_by_call.get(
                        parent_call_key, {}
                    ).get(citation)
                    if citation_source_hash is None and citation not in (
                        unresolvable_citation_ids_by_call.get(parent_call_key, set())
                    ):
                        citation_source_hash = source_id_hash(citation)
                    source_entity_id = (
                        trusted_sources.get(citation_source_hash)
                        if citation_source_hash is not None
                        else None
                    )
                    if source_entity_id is None:
                        unresolved_hashes.append(source_id_hash(citation))
                        continue
                    add_used(record, source_entity_id)
                    add_derived(
                        record_entity_by_key[(record.run_id, record.event_id)], source_entity_id
                    )
                activity = document["activity"][activity_by_key[(record.run_id, record.event_id)]]
                activity["sk:lineageStatus"] = "unresolved" if unresolved_hashes else "resolved"
                if unresolved_hashes:
                    activity["sk:unresolvedCitationHashes"] = list(dict.fromkeys(unresolved_hashes))

        if record.kind == "TOOL_RESULT":
            result_value = record.payload.get("output")
            result_hash = hash_value(_plain_json(result_value))
            result_entity_id = _qualified_id(
                "tool-result", record.run_id, record.event_id, result_hash
            )
            document["entity"].setdefault(
                result_entity_id,
                {
                    "prov:type": _qualified_type("sk:ToolResult"),
                    "sk:toolResultHash": result_hash,
                    "sk:eventIdHash": source_id_hash(record.event_id),
                    "sk:runIdHash": source_id_hash(record.run_id),
                },
            )
            activity_id = activity_by_key[(record.run_id, record.event_id)]
            generation_id = _relation_id("generation", result_entity_id, activity_id)
            document["wasGeneratedBy"].setdefault(
                generation_id,
                {
                    "prov:entity": result_entity_id,
                    "prov:activity": activity_id,
                    "prov:time": record.timestamp.astimezone(timezone.utc).isoformat(),
                },
            )
            result_key = (record.run_id, record.event_id)
            tool_result_entities[result_key] = result_entity_id
            tool_result_values[result_key] = result_value

    # Associate tool-call IDs from actual LLM responses with one unambiguous
    # tool invocation/result before the next LLM call. Matching requires the
    # component name and exact argument value as well as the message's ID and
    # result text below; chronology alone never creates an edge.
    tool_call_indices: dict[tuple[str, str, str], list[int]] = {}
    tool_calls_by_signature: dict[tuple[str, str, str], list[AuditRecord]] = {}
    tool_results_by_parent: dict[tuple[str, str], list[tuple[int, AuditRecord]]] = {}
    llm_call_indices_by_run: dict[str, list[int]] = {}
    for record_index, candidate in enumerate(records):
        if candidate.kind == "LLM_CALL":
            llm_call_indices_by_run.setdefault(candidate.run_id, []).append(record_index)
        elif (
            candidate.kind == "TOOL_CALL"
            and candidate.payload.get("method") == "run"
            and isinstance(candidate.payload.get("component"), str)
        ):
            signature = (
                candidate.run_id,
                candidate.payload["component"],
                _digest(_plain_json(candidate.payload.get("kwargs", {}))),
            )
            tool_call_indices.setdefault(signature, []).append(record_index)
            tool_calls_by_signature.setdefault(signature, []).append(candidate)
        elif candidate.kind == "TOOL_RESULT" and candidate.parent_id is not None:
            tool_results_by_parent.setdefault((candidate.run_id, candidate.parent_id), []).append(
                (record_index, candidate)
            )

    correlated_results: dict[tuple[str, str], list[tuple[str, Any]]] = {}
    for response_index, response in enumerate(records):
        if response.kind != "LLM_RESPONSE":
            continue
        call_positions = llm_call_indices_by_run.get(response.run_id, [])
        next_call_offset = bisect_right(call_positions, response_index)
        next_llm_call_index = (
            call_positions[next_call_offset]
            if next_call_offset < len(call_positions)
            else len(records)
        )
        for call_id, tool_name, arguments in _tool_calls_from_response(
            response.payload.get("output")
        ):
            signature = (response.run_id, tool_name, _digest(_plain_json(arguments)))
            positions = tool_call_indices.get(signature, [])
            matching_start = bisect_right(positions, response_index)
            matching_end = bisect_left(positions, next_llm_call_index)
            if matching_end - matching_start != 1:
                continue
            tool_call_index = positions[matching_start]
            tool_call = tool_calls_by_signature[signature][matching_start]
            matching_results = [
                candidate
                for tool_result_index, candidate in tool_results_by_parent.get(
                    (response.run_id, tool_call.event_id), []
                )
                if tool_call_index < tool_result_index < next_llm_call_index
                and candidate.payload.get("component") == tool_name
            ]
            if len(matching_results) != 1:
                continue
            tool_result = matching_results[0]
            result_key = (tool_result.run_id, tool_result.event_id)
            correlated_entity_id = tool_result_entities.get(result_key)
            if correlated_entity_id is not None:
                correlated_results.setdefault((response.run_id, call_id), []).append(
                    (correlated_entity_id, tool_result_values[result_key])
                )

    unambiguous_results = {
        key: values[0] for key, values in correlated_results.items() if len(values) == 1
    }
    results_used_by_call: dict[tuple[str, str], dict[str, None]] = {}
    for record in records:
        if record.kind != "LLM_CALL":
            continue
        used_results: dict[str, None] = {}
        for role, content, tool_call_id in _prompt_messages(record.payload):
            if role != "tool" or tool_call_id is None:
                continue
            correlated = unambiguous_results.get((record.run_id, tool_call_id))
            if correlated is None:
                continue
            result_entity_id, result_value = correlated
            if isinstance(result_value, str) and content == result_value:
                used_results.setdefault(result_entity_id, None)
                add_used(record, result_entity_id)
        results_used_by_call[(record.run_id, record.event_id)] = used_results

    for record in records:
        if record.kind != "LLM_RESPONSE" or record.parent_id is None:
            continue
        used_results = results_used_by_call.get((record.run_id, record.parent_id), {})
        for result_entity_id in used_results:
            add_used(record, result_entity_id)
            add_derived(record_entity_by_key[(record.run_id, record.event_id)], result_entity_id)

    return document
