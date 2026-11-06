"""PROV-JSON builder tests for audit-record lineage and privacy."""

from __future__ import annotations

import json

from synapsekit.audit import AuditTracer, EventKind
from synapsekit.audit.provenance import build_provenance, source_id_hash
from synapsekit.audit.serializer import hash_value


def _event(
    tracer: AuditTracer,
    kind: EventKind,
    payload: dict,
    *,
    event_id: str,
    parent_id: str | None = None,
    actor: str = "agent:test",
):
    return tracer.record(
        kind,
        payload,
        event_id=event_id,
        parent_id=parent_id,
        actor=actor,
    )


def _activity_id(provenance: dict, event_id: str) -> str:
    return next(
        identifier
        for identifier, activity in provenance["activity"].items()
        if activity["sk:eventIdHash"] == source_id_hash(event_id)
    )


def _record_entity_id(provenance: dict, event_id: str) -> str:
    return next(
        identifier
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:eventIdHash") == source_id_hash(event_id)
    )


def _source_entity_id(provenance: dict, source_id: str) -> str:
    source_hash = source_id_hash(source_id)
    return next(
        identifier
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:sourceIdHash") == source_hash
    )


def test_unknown_record_kind_is_normalized_in_provenance():
    tracer = AuditTracer(run_id="kind-allowlist")
    custom_kind = "private-custom-kind-token"
    tracer.record(custom_kind, {}, event_id="custom-kind")
    _event(tracer, EventKind.DECISION, {}, event_id="known-kind")

    provenance = build_provenance(tracer.records)
    custom_activity = provenance["activity"][_activity_id(provenance, "custom-kind")]
    known_activity = provenance["activity"][_activity_id(provenance, "known-kind")]
    serialized = json.dumps(provenance, sort_keys=True)

    assert custom_activity["sk:kind"] == "other"
    assert known_activity["sk:kind"] == EventKind.DECISION.value
    assert custom_kind not in serialized


def test_llm_tool_retrieval_and_answer_form_prov_lineage():
    tracer = AuditTracer(run_id="lineage-run")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {
            "output": [
                {
                    "text": "private retrieved text",
                    "metadata": {"source_id": "source-42"},
                }
            ]
        },
        event_id="retrieval",
        actor="retriever",
    )
    llm_call = _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "args": [[{"role": "user", "content": "private prompt"}]],
            "trusted_source_ids": ["source-42"],
            "model": "test-model",
        },
        event_id="llm-call",
        actor="llm:test",
    )
    _event(
        tracer,
        EventKind.TOOL_CALL,
        {"tool_args": {"query": "private tool argument"}},
        event_id="tool-call",
        parent_id=llm_call.event_id,
        actor="tool:search",
    )
    _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {
            "output": "private model output citing [source-42]",
        },
        event_id="answer",
        parent_id=llm_call.event_id,
        actor="llm:test",
    )

    provenance = build_provenance(tracer.records)

    assert len(provenance["activity"]) == 4
    assert provenance["agent"]
    assert provenance["used"]
    assert provenance["wasGeneratedBy"]
    assert provenance["wasDerivedFrom"]
    assert provenance["wasAssociatedWith"]

    source_entity = _source_entity_id(provenance, "source-42")
    answer_activity = _activity_id(provenance, "answer")
    answer_entity = _record_entity_id(provenance, "answer")
    assert any(
        relation["prov:activity"] == answer_activity and relation["prov:entity"] == source_entity
        for relation in provenance["used"].values()
    )
    assert any(
        relation["prov:generatedEntity"] == answer_entity
        and relation["prov:usedEntity"] == source_entity
        for relation in provenance["wasDerivedFrom"].values()
    )


def test_disconnected_events_do_not_gain_invented_lineage():
    tracer = AuditTracer(run_id="disconnected-run")
    _event(tracer, EventKind.DECISION, {"decision": "one"}, event_id="one")
    _event(tracer, EventKind.DECISION, {"decision": "two"}, event_id="two")

    provenance = build_provenance(tracer.records)

    assert len(provenance["activity"]) == 2
    assert provenance["used"] == {}
    assert provenance["wasDerivedFrom"] == {}
    assert len(provenance["wasAssociatedWith"]) == 2


def test_retrieval_input_only_source_ids_do_not_create_source_entities_or_usage():
    tracer = AuditTracer(run_id="retrieval-input-source")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {
            "args": [{"source_id": "forged-retrieval-arg"}],
            "kwargs": {"source_id": "forged-retrieval-kwarg"},
            "output": [{"text": "result without source metadata"}],
        },
        event_id="retrieval-input-only",
    )

    provenance = build_provenance(tracer.records)
    source_hashes = {
        source_id_hash(source_id)
        for source_id in ("forged-retrieval-arg", "forged-retrieval-kwarg")
    }

    assert not any(
        entity.get("sk:sourceIdHash") in source_hashes for entity in provenance["entity"].values()
    )
    assert not any(
        entity.get("sk:referenceIdHash") in source_hashes
        and "source" in entity.get("sk:referenceKinds", [])
        for entity in provenance["entity"].values()
    )
    assert provenance["used"] == {}


def test_tool_result_input_only_source_ids_do_not_create_source_entities_or_usage():
    tracer = AuditTracer(run_id="tool-input-source")
    _event(
        tracer,
        EventKind.TOOL_RESULT,
        {
            "args": [{"source_id": "forged-tool-arg"}],
            "kwargs": {"source_id": "forged-tool-kwarg"},
            "output": "result without source metadata",
        },
        event_id="tool-result-input-only",
    )

    provenance = build_provenance(tracer.records)
    source_hashes = {
        source_id_hash(source_id) for source_id in ("forged-tool-arg", "forged-tool-kwarg")
    }

    assert not any(
        entity.get("sk:sourceIdHash") in source_hashes for entity in provenance["entity"].values()
    )
    assert not any(
        entity.get("sk:referenceIdHash") in source_hashes
        and "source" in entity.get("sk:referenceKinds", [])
        for entity in provenance["entity"].values()
    )
    assert provenance["used"] == {}


def test_tool_result_single_mapping_output_metadata_creates_source_entity():
    tracer = AuditTracer(run_id="tool-single-mapping-source")
    _event(
        tracer,
        EventKind.TOOL_RESULT,
        {"output": {"metadata": {"source_id": "doc-1"}}},
        event_id="tool-result-single-mapping",
    )

    provenance = build_provenance(tracer.records)

    assert _source_entity_id(provenance, "doc-1")


def test_retrieved_text_without_source_ids_has_no_source_entity():
    tracer = AuditTracer(run_id="no-source-run")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {"query": "private query", "output": [{"text": "unidentified text"}]},
        event_id="retrieval-without-metadata",
        actor="retriever",
    )

    provenance = build_provenance(tracer.records)

    assert not any("sk:sourceIdHash" in entity for entity in provenance["entity"].values())
    assert provenance["used"] == {}


def test_rag_prompt_preserves_only_explicit_source_id_metadata():
    from synapsekit.rag.pipeline import RAGPipeline

    with_source_id = RAGPipeline._format_context_result(
        {"text": "retrieved passage", "metadata": {"source_id": "rag-document-3"}}
    )
    without_source_id = RAGPipeline._format_context_result(
        {"text": "unidentified passage", "metadata": {"source": "report.pdf"}}
    )

    assert "source_id: rag-document-3" in with_source_id
    assert "source_id:" not in without_source_id


def test_answer_citation_without_retriever_metadata_does_not_invent_source():
    tracer = AuditTracer(run_id="unsupported-citation")
    _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {"output": "Answer citing [unverified-doc]"},
        event_id="answer-without-source",
    )

    provenance = build_provenance(tracer.records)

    assert not any("sk:sourceIdHash" in entity for entity in provenance["entity"].values())
    assert provenance["used"] == {}


def test_output_reference_is_the_same_entity_when_consumed_as_input():
    tracer = AuditTracer(run_id="reference-handoff")
    producer = _event(
        tracer,
        EventKind.TOOL_RESULT,
        {"output_id": "tool-result-1"},
        event_id="producer",
    )
    consumer = _event(
        tracer,
        EventKind.LLM_CALL,
        {"input_id": "tool-result-1"},
        event_id="consumer",
    )

    provenance = build_provenance(tracer.records)
    output_entity = next(
        relation["prov:entity"]
        for relation in provenance["wasGeneratedBy"].values()
        if relation["prov:activity"] == _activity_id(provenance, producer.event_id)
        and relation["prov:entity"] != _record_entity_id(provenance, producer.event_id)
    )

    assert any(
        relation["prov:entity"] == output_entity
        and relation["prov:activity"] == _activity_id(provenance, consumer.event_id)
        for relation in provenance["used"].values()
    )


def test_reference_that_is_both_input_and_output_does_not_self_derive():
    # Regression: when a single record names the same reference as both its
    # input and its output, the reference resolves to one entity for both
    # roles. The derivation pass must not emit a wasDerivedFrom edge from that
    # entity to itself (a semantically invalid self-cycle in the PROV graph).
    tracer = AuditTracer(run_id="self-derivation")
    _event(
        tracer,
        EventKind.TOOL_RESULT,
        {"input_id": "ref-1", "output_id": "ref-1"},
        event_id="roundtrip",
    )

    provenance = build_provenance(tracer.records)

    assert all(
        relation["prov:generatedEntity"] != relation["prov:usedEntity"]
        for relation in provenance["wasDerivedFrom"].values()
    )


def test_provenance_ids_and_activity_order_are_stable():
    tracer = AuditTracer(run_id="stable-run")
    first = _event(tracer, EventKind.TOOL_CALL, {}, event_id="first")
    _event(tracer, EventKind.TOOL_RESULT, {}, event_id="second", parent_id=first.event_id)

    first_build = build_provenance(tracer.records)
    second_build = build_provenance(tracer.records)

    assert list(first_build["activity"]) == list(second_build["activity"])
    assert list(first_build["activity"]) == [
        _activity_id(first_build, "first"),
        _activity_id(first_build, "second"),
    ]
    assert first_build == second_build


def test_provenance_omits_prompts_retrieved_text_tool_args_outputs_and_source_ids():
    from synapsekit.audit.redact import PIIRedactor

    tracer = AuditTracer(run_id="private-run", redactor=PIIRedactor())
    _event(
        tracer,
        EventKind.LLM_CALL,
        {"prompt": "email alice@example.com about private prompt", "api_key": "api-secret-value"},
        event_id="private-llm-call",
    )
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {
            "output": [
                {
                    "text": "private retrieved document body",
                    "metadata": {"source_id": "private-document-id"},
                }
            ],
            "tool_args": {"token": "tool-secret-value"},
        },
        event_id="private-retrieval",
    )
    _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {"output": "private model output"},
        event_id="private-answer",
    )

    provenance = build_provenance(tracer.records)
    serialized = json.dumps(provenance, sort_keys=True)

    for private_value in (
        "alice@example.com",
        "private prompt",
        "api-secret-value",
        "private retrieved document body",
        "tool-secret-value",
        "private model output",
        "private-document-id",
    ):
        assert private_value not in serialized
    assert all("sk:recordHash" in activity for activity in provenance["activity"].values())
    assert all("sk:payloadHash" in activity for activity in provenance["activity"].values())
    assert any(
        activity["sk:redactionStatus"] == "redacted" for activity in provenance["activity"].values()
    )


def test_llm_message_contents_become_hashed_prompt_fragment_entities():
    tracer = AuditTracer(run_id="prompt-fragments")
    _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "args": [
                [
                    {"role": "system", "content": "private system fragment"},
                    {"role": "user", "content": "private user fragment"},
                ]
            ]
        },
        event_id="fragment-call",
    )

    provenance = build_provenance(tracer.records)
    fragments = [
        entity for entity in provenance["entity"].values() if entity.get("sk:promptFragmentHash")
    ]

    assert {fragment["sk:promptRole"] for fragment in fragments} == {"system", "user"}
    assert len({fragment["sk:promptFragmentHash"] for fragment in fragments}) == 2
    assert all(
        any(
            relation["prov:activity"] == _activity_id(provenance, "fragment-call")
            and relation["prov:entity"] == fragment_id
            for relation in provenance["used"].values()
        )
        for fragment_id in provenance["entity"]
        if provenance["entity"][fragment_id].get("sk:promptFragmentHash")
    )
    serialized = json.dumps(provenance, sort_keys=True)
    assert "private system fragment" not in serialized
    assert "private user fragment" not in serialized


def test_nonstandard_prompt_roles_are_not_exported_verbatim():
    tracer = AuditTracer(run_id="prompt-role-allowlist")
    _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "args": [
                [
                    {"role": "system", "content": "system text"},
                    {"role": "user", "content": "user text"},
                    {
                        "role": "secret-role-token-do-not-export",
                        "content": "sensitive-role text",
                    },
                ]
            ]
        },
        event_id="role-call",
    )

    provenance = build_provenance(tracer.records)
    serialized = json.dumps(provenance, sort_keys=True)
    roles = {
        entity["sk:promptRole"]
        for entity in provenance["entity"].values()
        if "sk:promptRole" in entity
    }

    assert {"system", "user"} <= roles
    assert "secret-role-token-do-not-export" not in serialized


def test_multimodal_messages_hash_only_text_blocks_in_order():
    openai_image_data = "OPENAI_PRIVATE_IMAGE_BASE64"
    anthropic_image_data = "ANTHROPIC_PRIVATE_IMAGE_BASE64"
    text_blocks = [
        "OpenAI text before image",
        "OpenAI text after image",
        "Anthropic text before image",
        "Anthropic text after image",
    ]
    tracer = AuditTracer(run_id="multimodal-prompt-fragments")
    _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "kwargs": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_blocks[0]},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{openai_image_data}"},
                            },
                            {"type": "text", "text": text_blocks[1]},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_blocks[2]},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": anthropic_image_data,
                                },
                            },
                            {"type": "text", "text": text_blocks[3]},
                        ],
                    },
                ]
            }
        },
        event_id="multimodal-call",
    )

    provenance = build_provenance(tracer.records)
    fragments = sorted(
        (entity for entity in provenance["entity"].values() if entity.get("sk:promptFragmentHash")),
        key=lambda entity: entity["sk:fragmentIndex"],
    )
    serialized = json.dumps(provenance, sort_keys=True)

    assert [fragment["sk:promptFragmentHash"] for fragment in fragments] == [
        hash_value(text) for text in text_blocks
    ]
    assert openai_image_data not in serialized
    assert anthropic_image_data not in serialized


def test_text_citations_are_linked_or_marked_unresolved_without_source_invention():
    tracer = AuditTracer(run_id="citation-resolution")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {"output": [{"text": "private retrieved text", "metadata": {"source_id": "known-source"}}]},
        event_id="source-retrieval",
    )
    llm_call = _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "args": [[{"role": "user", "content": "private question"}]],
            "trusted_source_ids": ["known-source"],
        },
        event_id="citation-call",
    )
    _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {"output": "Answer [known-source] and unsupported [unknown-source]."},
        event_id="citation-answer",
        parent_id=llm_call.event_id,
    )

    provenance = build_provenance(tracer.records)
    source_entity = _source_entity_id(provenance, "known-source")
    answer_activity = provenance["activity"][_activity_id(provenance, "citation-answer")]
    answer_entity = _record_entity_id(provenance, "citation-answer")

    assert answer_activity.get("sk:lineageStatus") == "unresolved"
    assert answer_activity.get("sk:unresolvedCitationHashes") == [source_id_hash("unknown-source")]
    assert any(
        relation["prov:generatedEntity"] == answer_entity
        and relation["prov:usedEntity"] == source_entity
        for relation in provenance["wasDerivedFrom"].values()
    )
    assert not any(
        entity.get("sk:sourceIdHash") == source_id_hash("unknown-source")
        for entity in provenance["entity"].values()
    )
    serialized = json.dumps(provenance, sort_keys=True)
    assert "known-source" not in serialized
    assert "unknown-source" not in serialized


def test_tool_result_is_not_linked_by_same_run_order_without_tool_message_evidence():
    tracer = AuditTracer(run_id="uncorrelated-tool-result")
    first_call = _event(
        tracer,
        EventKind.LLM_CALL,
        {"args": [[{"role": "user", "content": "request"}]]},
        event_id="first-llm-call",
    )
    _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {
            "output": {
                "tool_calls": [
                    {
                        "id": "actual-tool-call-id",
                        "name": "lookup",
                        "arguments": {"query": "private argument"},
                    }
                ]
            }
        },
        event_id="first-llm-response",
        parent_id=first_call.event_id,
    )
    tool_call = _event(
        tracer,
        EventKind.TOOL_CALL,
        {"method": "run", "component": "lookup", "kwargs": {"query": "private argument"}},
        event_id="tool-call",
    )
    _event(
        tracer,
        EventKind.TOOL_RESULT,
        {"method": "run", "component": "lookup", "output": "private result"},
        event_id="tool-result",
        parent_id=tool_call.event_id,
    )
    later_call = _event(
        tracer,
        EventKind.LLM_CALL,
        {"args": [[{"role": "user", "content": "unrelated later input"}]]},
        event_id="unrelated-llm-call",
    )

    provenance = build_provenance(tracer.records)
    tool_result_entity = next(
        identifier
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:toolResultHash") == hash_value("private result")
    )
    later_activity = _activity_id(provenance, later_call.event_id)

    assert not any(
        relation["prov:activity"] == later_activity
        and relation["prov:entity"] == tool_result_entity
        for relation in provenance["used"].values()
    )


def test_tool_result_after_consuming_call_is_not_correlated_as_its_input():
    tracer = AuditTracer(run_id="late-tool-result")
    first_call = _event(
        tracer,
        EventKind.LLM_CALL,
        {"args": [[{"role": "user", "content": "request"}]]},
        event_id="initial-llm-call",
    )
    _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {
            "output": {
                "tool_calls": [
                    {
                        "id": "late-result-tool-id",
                        "name": "lookup",
                        "arguments": {"query": "private argument"},
                    }
                ]
            }
        },
        event_id="initial-llm-response",
        parent_id=first_call.event_id,
    )
    tool_call = _event(
        tracer,
        EventKind.TOOL_CALL,
        {"method": "run", "component": "lookup", "kwargs": {"query": "private argument"}},
        event_id="matching-tool-call",
    )
    consuming_call = _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "args": [
                [
                    {
                        "role": "tool",
                        "tool_call_id": "late-result-tool-id",
                        "content": "private result text",
                    }
                ]
            ]
        },
        event_id="consuming-llm-call",
    )
    _event(
        tracer,
        EventKind.TOOL_RESULT,
        {"component": "lookup", "output": "private result text"},
        event_id="late-tool-result-record",
        parent_id=tool_call.event_id,
    )

    provenance = build_provenance(tracer.records)
    tool_result_entity = next(
        identifier
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:toolResultHash") == hash_value("private result text")
    )
    consuming_activity = _activity_id(provenance, consuming_call.event_id)

    assert not any(
        relation["prov:activity"] == consuming_activity
        and relation["prov:entity"] == tool_result_entity
        for relation in provenance["used"].values()
    )


def test_non_source_record_metadata_cannot_authenticate_a_citation():
    tracer = AuditTracer(run_id="forged-source-run")
    _event(
        tracer,
        EventKind.DECISION,
        {"decision": "untrusted", "source_id": "forged-doc"},
        event_id="forged-decision",
    )
    _event(
        tracer,
        EventKind.LLM_CALL,
        {"source_id": "forged-doc", "args": [[{"role": "user", "content": "question"}]]},
        event_id="forged-call",
    )
    _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {"output": "Answer citing [forged-doc]"},
        event_id="forged-answer",
    )

    provenance = build_provenance(tracer.records)
    answer_activity = next(
        activity
        for activity in provenance["activity"].values()
        if activity["sk:kind"] == EventKind.LLM_RESPONSE.value
    )

    assert answer_activity["sk:lineageStatus"] == "unresolved"
    assert answer_activity["sk:unresolvedCitationHashes"] == [source_id_hash("forged-doc")]
    assert not any("sk:sourceIdHash" in entity for entity in provenance["entity"].values())


def test_citation_cannot_resolve_from_a_different_run():
    source_run = AuditTracer(run_id="source-run")
    answer_run = AuditTracer(run_id="answer-run")
    _event(
        source_run,
        EventKind.RETRIEVAL,
        {"output": [{"text": "retrieved", "metadata": {"source_id": "run-local-doc"}}]},
        event_id="source-retrieval",
    )
    _event(
        answer_run,
        EventKind.LLM_RESPONSE,
        {"output": "Answer citing [run-local-doc]"},
        event_id="cross-run-answer",
    )

    provenance = build_provenance([*source_run.records, *answer_run.records])
    answer_activity_id = next(
        identifier
        for identifier, activity in provenance["activity"].items()
        if activity["sk:kind"] == EventKind.LLM_RESPONSE.value
    )
    answer_activity = provenance["activity"][answer_activity_id]
    source_entity_id = _source_entity_id(provenance, "run-local-doc")

    assert answer_activity["sk:lineageStatus"] == "unresolved"
    assert not any(
        relation["prov:activity"] == answer_activity_id
        and relation["prov:entity"] == source_entity_id
        for relation in provenance["used"].values()
    )


def test_same_source_id_gets_distinct_entities_in_distinct_runs():
    first_run = AuditTracer(run_id="first-source-run")
    second_run = AuditTracer(run_id="second-source-run")
    for tracer, event_id in ((first_run, "first-retrieval"), (second_run, "second-retrieval")):
        _event(
            tracer,
            EventKind.RETRIEVAL,
            {"output": [{"text": "retrieved", "metadata": {"source_id": "reused-doc"}}]},
            event_id=event_id,
        )

    provenance = build_provenance([*first_run.records, *second_run.records])
    source_hash = source_id_hash("reused-doc")
    source_entities = [
        (identifier, entity)
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:sourceIdHash") == source_hash
    ]

    assert len(source_entities) == 2
    assert len({identifier for identifier, _entity in source_entities}) == 2
    assert {entity["sk:runIdHash"] for _identifier, entity in source_entities} == {
        source_id_hash("first-source-run"),
        source_id_hash("second-source-run"),
    }


def test_retrieval_citation_id_metadata_authenticates_answer_citation():
    tracer = AuditTracer(run_id="citation-id-source")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {"output": [{"text": "retrieved", "metadata": {"citation_id": "citation-only-doc"}}]},
        event_id="citation-id-retrieval",
    )
    llm_call = _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "args": [[{"role": "user", "content": "private question"}]],
            "trusted_source_ids": ["citation-only-doc"],
        },
        event_id="citation-id-call",
    )
    _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {"output": "Answer citing [citation-only-doc]"},
        event_id="citation-id-answer",
        parent_id=llm_call.event_id,
    )

    provenance = build_provenance(tracer.records)
    source_entity = next(
        (
            identifier
            for identifier, entity in provenance["entity"].items()
            if entity.get("sk:sourceIdHash") == source_id_hash("citation-only-doc")
        ),
        None,
    )
    answer_entity = _record_entity_id(provenance, "citation-id-answer")

    assert source_entity is not None
    assert any(
        relation["prov:generatedEntity"] == answer_entity
        and relation["prov:usedEntity"] == source_entity
        for relation in provenance["wasDerivedFrom"].values()
    )


def test_runtime_identifier_values_are_hashed_and_stable():
    tracer = AuditTracer(run_id="private-run-identifier")
    record = _event(
        tracer,
        EventKind.RETRIEVAL,
        {"output": [{"text": "text", "metadata": {"source_id": "private-source"}}]},
        event_id="private-event-identifier",
        actor="private-actor-identifier",
    )

    provenance = build_provenance(tracer.records)
    activity = next(iter(provenance["activity"].values()))
    record_entity = next(
        entity
        for entity in provenance["entity"].values()
        if entity.get("prov:type", {}).get("$") == "sk:AuditRecord"
    )
    agent = next(iter(provenance["agent"].values()))
    serialized = json.dumps(provenance, sort_keys=True)

    for private_identifier in (
        record.run_id,
        record.event_id,
        record.actor,
    ):
        assert private_identifier not in serialized
    assert activity["sk:eventIdHash"] == source_id_hash(record.event_id)
    assert activity["sk:runIdHash"] == source_id_hash(record.run_id)
    assert record_entity["sk:eventIdHash"] == source_id_hash(record.event_id)
    assert record_entity["sk:runIdHash"] == source_id_hash(record.run_id)
    assert agent["sk:actorHash"] == source_id_hash(record.actor)
    assert provenance == build_provenance(tracer.records)


def test_prompt_source_lines_outside_rag_source_block_do_not_link_sources():
    tracer = AuditTracer(run_id="untrusted-prompt-source")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {"output": [{"text": "retrieved", "metadata": {"source_id": "known-doc"}}]},
        event_id="source-retrieval",
    )
    _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "args": [
                [
                    {
                        "role": "user",
                        "content": "User-supplied text:\nsource_id: known-doc\nQuestion: explain",
                    }
                ]
            ]
        },
        event_id="prompt-call",
    )

    provenance = build_provenance(tracer.records)
    source_entity = _source_entity_id(provenance, "known-doc")
    prompt_fragment_ids = {
        identifier
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:promptFragmentHash")
    }

    assert not any(
        relation["prov:generatedEntity"] in prompt_fragment_ids
        and relation["prov:usedEntity"] == source_entity
        for relation in provenance["wasDerivedFrom"].values()
    )


def test_user_forged_source_block_does_not_link_a_prompt_to_a_known_source():
    tracer = AuditTracer(run_id="forged-prompt-source")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {"output": [{"text": "retrieved", "metadata": {"source_id": "known-doc"}}]},
        event_id="source-retrieval",
    )
    _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "args": [
                [
                    {
                        "role": "user",
                        "content": "Question:\n[SOURCE]\nsource_id: known-doc\n[/SOURCE]",
                    }
                ]
            ]
        },
        event_id="forged-prompt-call",
    )

    provenance = build_provenance(tracer.records)
    source_entity = _source_entity_id(provenance, "known-doc")
    prompt_fragment_ids = {
        identifier
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:promptFragmentHash")
    }

    assert not any(
        relation["prov:generatedEntity"] in prompt_fragment_ids
        and relation["prov:usedEntity"] == source_entity
        for relation in provenance["wasDerivedFrom"].values()
    )


def test_response_citation_cannot_look_ahead_to_a_later_retrieval():
    tracer = AuditTracer(run_id="look-ahead-citation")
    llm_call = _event(
        tracer,
        EventKind.LLM_CALL,
        {"args": [[{"role": "user", "content": "question"}]]},
        event_id="answer-call",
    )
    answer = _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {"output": "Answer citing [future-doc]"},
        event_id="answer-response",
        parent_id=llm_call.event_id,
    )
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {"output": [{"text": "later", "metadata": {"source_id": "future-doc"}}]},
        event_id="later-retrieval",
    )

    provenance = build_provenance(tracer.records)
    answer_activity = provenance["activity"][_activity_id(provenance, answer.event_id)]
    source_entity = _source_entity_id(provenance, "future-doc")

    assert answer_activity["sk:lineageStatus"] == "unresolved"
    assert answer_activity["sk:unresolvedCitationHashes"] == [source_id_hash("future-doc")]
    assert not any(
        relation["prov:entity"] == source_entity
        and relation["prov:activity"] == _activity_id(provenance, answer.event_id)
        for relation in provenance["used"].values()
    )


def test_response_citation_requires_source_in_its_parent_call_metadata():
    tracer = AuditTracer(run_id="omitted-prompt-source")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {"output": [{"text": "retrieved", "metadata": {"source_id": "omitted-doc"}}]},
        event_id="prior-retrieval",
    )
    llm_call = _event(
        tracer,
        EventKind.LLM_CALL,
        {"args": [[{"role": "user", "content": "unrelated question"}]]},
        event_id="omitting-call",
    )
    answer = _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {"output": "Answer citing [omitted-doc]"},
        event_id="omitting-response",
        parent_id=llm_call.event_id,
    )

    provenance = build_provenance(tracer.records)
    answer_activity = provenance["activity"][_activity_id(provenance, answer.event_id)]
    source_entity = _source_entity_id(provenance, "omitted-doc")

    assert answer_activity["sk:lineageStatus"] == "unresolved"
    assert answer_activity["sk:unresolvedCitationHashes"] == [source_id_hash("omitted-doc")]
    assert not any(
        relation["prov:entity"] == source_entity
        and relation["prov:activity"] == _activity_id(provenance, answer.event_id)
        for relation in provenance["used"].values()
    )


def test_llm_call_argument_citation_ids_do_not_create_source_usage():
    tracer = AuditTracer(run_id="input-citation-is-not-usage")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {"output": [{"text": "retrieved", "metadata": {"source_id": "known-source"}}]},
        event_id="retrieval",
    )
    call = _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "args": [{"citation_id": "known-source"}],
            "kwargs": {"citation_ids": ["known-source"]},
        },
        event_id="call-with-input-citation",
    )

    provenance = build_provenance(tracer.records)
    source_entity = _source_entity_id(provenance, "known-source")
    call_activity = _activity_id(provenance, call.event_id)

    assert not any(
        relation["prov:activity"] == call_activity and relation["prov:entity"] == source_entity
        for relation in provenance["used"].values()
    )


def test_tool_call_argument_citation_does_not_resolve_response():
    tracer = AuditTracer(run_id="tool-argument-is-not-citation")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {"output": [{"text": "retrieved", "metadata": {"source_id": "known-source"}}]},
        event_id="retrieval",
    )
    call = _event(
        tracer,
        EventKind.LLM_CALL,
        {"trusted_source_ids": ["known-source"]},
        event_id="call-with-trusted-source",
    )
    answer = _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {
            "output": {
                "tool_calls": [
                    {
                        "id": "lookup-call",
                        "name": "lookup",
                        "arguments": {"citation_id": "known-source"},
                    }
                ]
            }
        },
        event_id="tool-call-response",
        parent_id=call.event_id,
    )

    provenance = build_provenance(tracer.records)
    source_entity = _source_entity_id(provenance, "known-source")
    answer_activity = _activity_id(provenance, answer.event_id)
    answer_entity = _record_entity_id(provenance, answer.event_id)

    assert provenance["activity"][answer_activity].get("sk:lineageStatus") != "resolved"
    assert not any(
        relation["prov:activity"] == answer_activity and relation["prov:entity"] == source_entity
        for relation in provenance["used"].values()
    )
    assert not any(
        relation["prov:generatedEntity"] == answer_entity
        and relation["prov:usedEntity"] == source_entity
        for relation in provenance["wasDerivedFrom"].values()
    )


def test_explicit_output_citation_id_resolves_to_trusted_source():
    tracer = AuditTracer(run_id="explicit-output-citation")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {"output": [{"text": "retrieved", "metadata": {"source_id": "known-source"}}]},
        event_id="retrieval",
    )
    call = _event(
        tracer,
        EventKind.LLM_CALL,
        {"trusted_source_ids": ["known-source"]},
        event_id="call-with-trusted-source",
    )
    answer = _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {"output": {"content": "Supported answer", "citation_ids": ["known-source"]}},
        event_id="explicit-output-answer",
        parent_id=call.event_id,
    )

    provenance = build_provenance(tracer.records)
    source_entity = _source_entity_id(provenance, "known-source")
    answer_activity = _activity_id(provenance, answer.event_id)
    answer_entity = _record_entity_id(provenance, answer.event_id)

    assert provenance["activity"][answer_activity]["sk:lineageStatus"] == "resolved"
    assert any(
        relation["prov:generatedEntity"] == answer_entity
        and relation["prov:usedEntity"] == source_entity
        for relation in provenance["wasDerivedFrom"].values()
    )


def test_response_argument_citation_ids_do_not_resolve_or_link_source():
    tracer = AuditTracer(run_id="response-input-citation-is-not-output")
    _event(
        tracer,
        EventKind.RETRIEVAL,
        {"output": [{"text": "retrieved", "metadata": {"source_id": "known-source"}}]},
        event_id="retrieval",
    )
    call = _event(
        tracer,
        EventKind.LLM_CALL,
        {"trusted_source_ids": ["known-source"]},
        event_id="call-with-trusted-source",
    )
    answer = _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {
            "args": [{"citation_id": "known-source"}],
            "kwargs": {"citation_ids": ["known-source"]},
            "output": "Supported answer without a reference",
        },
        event_id="answer-with-input-citation-only",
        parent_id=call.event_id,
    )

    provenance = build_provenance(tracer.records)
    source_entity = _source_entity_id(provenance, "known-source")
    answer_activity = _activity_id(provenance, answer.event_id)
    answer_entity = _record_entity_id(provenance, answer.event_id)

    assert provenance["activity"][answer_activity].get("sk:lineageStatus") != "resolved"
    assert not any(
        relation["prov:activity"] == answer_activity and relation["prov:entity"] == source_entity
        for relation in provenance["used"].values()
    )
    assert not any(
        relation["prov:generatedEntity"] == answer_entity
        and relation["prov:usedEntity"] == source_entity
        for relation in provenance["wasDerivedFrom"].values()
    )


def test_llm_response_argument_and_nested_direct_input_ids_do_not_create_entities_or_usage():
    tracer = AuditTracer(run_id="response-argument-input-id")
    response = _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {
            "args": [{"input_id": "argument-input-id"}],
            "kwargs": {"input_ids": ["keyword-input-id"]},
            "metadata": {"input_id": "unrelated-branch-input-id"},
            "input_id": {"metadata": {"input_id": "nested-direct-input-id"}},
            "output": "answer",
        },
        event_id="response-with-input-shaped-arguments",
    )

    provenance = build_provenance(tracer.records)
    input_hashes = {
        source_id_hash("argument-input-id"),
        source_id_hash("keyword-input-id"),
        source_id_hash("unrelated-branch-input-id"),
        source_id_hash("nested-direct-input-id"),
    }

    assert not any(
        entity.get("sk:referenceIdHash") in input_hashes
        and "input" in entity.get("sk:referenceKinds", [])
        for entity in provenance["entity"].values()
    )
    assert not any(
        relation["prov:activity"] == _activity_id(provenance, response.event_id)
        for relation in provenance["used"].values()
    )


def test_tool_call_and_nested_direct_output_ids_do_not_create_generated_entities():
    tracer = AuditTracer(run_id="response-tool-argument-output-id")
    response = _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {
            "output": {
                "tool_calls": [
                    {
                        "id": "lookup-call",
                        "name": "lookup",
                        "arguments": {"output_id": "not-produced"},
                    }
                ],
                "function_call": {
                    "name": "lookup",
                    "arguments": {"result_id": "not-produced-by-function"},
                },
            },
            "metadata": {"output_id": "unrelated-branch-output-id"},
            "output_id": {
                "tool_calls": [{"arguments": {"output_id": "nested-direct-tool-output-id"}}],
                "function_call": {"arguments": {"result_id": "nested-direct-function-output-id"}},
            },
        },
        event_id="response-with-output-shaped-tool-arguments",
    )

    provenance = build_provenance(tracer.records)
    response_activity = _activity_id(provenance, response.event_id)

    output_hashes = {
        source_id_hash("not-produced"),
        source_id_hash("not-produced-by-function"),
        source_id_hash("unrelated-branch-output-id"),
        source_id_hash("nested-direct-tool-output-id"),
        source_id_hash("nested-direct-function-output-id"),
    }
    assert not any(
        entity.get("sk:referenceIdHash") in output_hashes
        and "output" in entity.get("sk:referenceKinds", [])
        for entity in provenance["entity"].values()
    )
    assert {
        relation["prov:entity"]
        for relation in provenance["wasGeneratedBy"].values()
        if relation["prov:activity"] == response_activity
    } == {_record_entity_id(provenance, response.event_id)}


def test_input_references_ignore_provider_call_argument_subtrees():
    tracer = AuditTracer(run_id="provider-call-input-references")
    provider_input_ids = {"tool-argument-input-id", "function-argument-input-id"}
    record = _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "input": {
                "metadata": {"input_id": "structured-input-id"},
                "tool_calls": [
                    {"arguments": {"input_id": "tool-argument-input-id"}},
                ],
                "function_call": {
                    "arguments": {"input_id": "function-argument-input-id"},
                },
            }
        },
        event_id="provider-call-inputs",
    )

    provenance = build_provenance(tracer.records)
    provider_input_hashes = {source_id_hash(reference) for reference in provider_input_ids}
    provider_entities = {
        entity_id
        for entity_id, entity in provenance["entity"].items()
        if entity.get("sk:referenceIdHash") in provider_input_hashes
        and "input" in entity.get("sk:referenceKinds", [])
    }
    activity_id = _activity_id(provenance, record.event_id)
    used_reference_hashes = {
        provenance["entity"][relation["prov:entity"]].get("sk:referenceIdHash")
        for relation in provenance["used"].values()
        if relation["prov:activity"] == activity_id
    }

    assert provider_entities == set()
    assert provider_input_hashes.isdisjoint(used_reference_hashes)
    structured_input_entity = next(
        entity_id
        for entity_id, entity in provenance["entity"].items()
        if entity.get("sk:referenceIdHash") == source_id_hash("structured-input-id")
        and "input" in entity.get("sk:referenceKinds", [])
    )
    assert any(
        relation["prov:activity"] == activity_id
        and relation["prov:entity"] == structured_input_entity
        for relation in provenance["used"].values()
    )


def test_structured_input_and_output_sections_create_reference_entities():
    tracer = AuditTracer(run_id="structured-reference-sections")
    record = _event(
        tracer,
        EventKind.DECISION,
        {
            "input": {"metadata": {"input_id": "structured-input-id"}},
            "output": {"result": {"output_id": "structured-output-id"}},
        },
        event_id="structured-references",
    )

    provenance = build_provenance(tracer.records)
    reference_hashes = {
        source_id_hash("structured-input-id"),
        source_id_hash("structured-output-id"),
    }
    reference_entities = {
        entity_id: entity
        for entity_id, entity in provenance["entity"].items()
        if entity.get("sk:referenceIdHash") in reference_hashes
    }
    activity_id = _activity_id(provenance, record.event_id)

    input_entity_id = next(
        entity_id
        for entity_id, entity in reference_entities.items()
        if entity["sk:referenceIdHash"] == source_id_hash("structured-input-id")
    )
    output_entity_id = next(
        entity_id
        for entity_id, entity in reference_entities.items()
        if entity["sk:referenceIdHash"] == source_id_hash("structured-output-id")
    )

    assert {
        entity["sk:referenceIdHash"] for entity in reference_entities.values()
    } == reference_hashes
    assert any(
        relation["prov:activity"] == activity_id and relation["prov:entity"] == input_entity_id
        for relation in provenance["used"].values()
    )
    assert any(
        relation["prov:activity"] == activity_id and relation["prov:entity"] == output_entity_id
        for relation in provenance["wasGeneratedBy"].values()
    )


def test_positional_tool_invocation_is_not_correlated_to_response_arguments():
    tracer = AuditTracer(run_id="positional-tool-correlation")
    llm_call = _event(
        tracer,
        EventKind.LLM_CALL,
        {"args": [[{"role": "user", "content": "request"}]]},
        event_id="initial-call",
    )
    _event(
        tracer,
        EventKind.LLM_RESPONSE,
        {
            "output": {
                "tool_calls": [
                    {
                        "id": "positional-tool-call-id",
                        "name": "lookup",
                        "arguments": {"query": "private argument"},
                    }
                ]
            }
        },
        event_id="tool-call-response",
        parent_id=llm_call.event_id,
    )
    tool_call = _event(
        tracer,
        EventKind.TOOL_CALL,
        {
            "method": "run",
            "component": "lookup",
            "args": ["private argument"],
            "kwargs": {},
        },
        event_id="positional-tool-call",
    )
    _event(
        tracer,
        EventKind.TOOL_RESULT,
        {"component": "lookup", "output": "private result"},
        event_id="positional-tool-result",
        parent_id=tool_call.event_id,
    )
    next_call = _event(
        tracer,
        EventKind.LLM_CALL,
        {
            "args": [
                [
                    {
                        "role": "tool",
                        "tool_call_id": "positional-tool-call-id",
                        "content": "private result",
                    }
                ]
            ]
        },
        event_id="following-call",
    )

    provenance = build_provenance(tracer.records)
    tool_result_entity = next(
        identifier
        for identifier, entity in provenance["entity"].items()
        if entity.get("sk:toolResultHash") == hash_value("private result")
    )
    next_activity = _activity_id(provenance, next_call.event_id)

    assert not any(
        relation["prov:activity"] == next_activity and relation["prov:entity"] == tool_result_entity
        for relation in provenance["used"].values()
    )
