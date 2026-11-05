"""VerifiableAgent — wraps an agent's methods without touching its implementation."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from synapsekit._audit_context import SourceAwareMessages
from synapsekit.audit import AuditTracer, EventKind, VerifiableAgent, build_provenance
from synapsekit.audit.provenance import source_id_hash
from synapsekit.audit.wrapper import audited, infer_kind_pair


class DummyAgent:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        return f"response to {prompt}"

    def retrieve(self, query: str) -> list[str]:
        return [f"doc-for-{query}"]

    async def failing_tool(self, x: int) -> int:
        raise ValueError("boom")


class _KeywordMessagesAgent:
    async def generate_with_messages(self, *, messages):
        return "keyword message received"


def test_wrapper_captures_source_aware_messages_from_keyword_argument():
    async def scenario() -> None:
        tracer = AuditTracer()
        agent = VerifiableAgent(_KeywordMessagesAgent(), tracer=tracer)
        messages = SourceAwareMessages(
            [{"role": "user", "content": "question"}],
            trusted_source_ids=("keyword-source",),
        )

        assert await agent.generate_with_messages(messages=messages) == "keyword message received"
        llm_call = next(
            record for record in tracer.records if record.kind == EventKind.LLM_CALL.value
        )
        assert llm_call.payload.get("trusted_source_ids") == ("keyword-source",)

    asyncio.run(scenario())


class TestInferKindPair:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("generate", (EventKind.LLM_CALL, EventKind.LLM_RESPONSE)),
            ("chat_complete", (EventKind.LLM_CALL, EventKind.LLM_RESPONSE)),
            ("retrieve_docs", (EventKind.RETRIEVAL, None)),
            ("recall_memory", (EventKind.MEMORY_READ, None)),
            ("save_memory", (EventKind.MEMORY_WRITE, None)),
            ("decide_next_step", (EventKind.DECISION, None)),
            ("run_tool", (EventKind.TOOL_CALL, EventKind.TOOL_RESULT)),
            ("frobnicate", (EventKind.SYSTEM_EVENT, None)),
        ],
    )
    def test_keyword_based_inference(self, name, expected):
        assert infer_kind_pair(name) == expected


class TestVerifiableAgent:
    @pytest.mark.asyncio
    async def test_paired_kind_emits_a_call_and_a_response_record(self):
        # "generate" -> (LLM_CALL, LLM_RESPONSE): a paired kind emits two
        # linked records, not one.
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        result = await agent.generate("hello")
        assert result == "response to hello"

        records = tracer.records
        assert len(records) == 2
        call, response = records
        assert call.kind == EventKind.LLM_CALL.value
        assert "output" not in call.payload
        assert response.kind == EventKind.LLM_RESPONSE.value
        assert response.payload["output"] == "response to hello"
        assert response.payload["status"] == "ok"
        assert response.parent_id == call.event_id

    def test_single_kind_emits_one_record(self):
        # "retrieve" -> RETRIEVAL has no paired response kind.
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        result = agent.retrieve("france")
        assert result == ["doc-for-france"]
        records = tracer.records
        assert len(records) == 1
        assert records[0].kind == EventKind.RETRIEVAL.value
        # Frozen payload: lists become tuples (see types.deep_freeze).
        assert records[0].payload["output"] == ("doc-for-france",)

    @pytest.mark.asyncio
    async def test_exceptions_are_recorded_as_error_and_reraised(self):
        # "failing_tool" -> paired (TOOL_CALL, TOOL_RESULT); an exception
        # replaces the result record's kind with ERROR.
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        with pytest.raises(ValueError, match="boom"):
            await agent.failing_tool(1)
        records = tracer.records
        assert len(records) == 2
        call, error = records
        assert call.kind == EventKind.TOOL_CALL.value
        assert error.kind == EventKind.ERROR.value
        assert error.payload["status"] == "error"
        assert error.payload["error_type"] == "ValueError"
        assert error.parent_id == call.event_id

    def test_non_callable_attributes_pass_through(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        assert agent.calls == 0

    def test_uses_the_tracer_instance_passed_in_even_when_empty(self):
        # Regression: an empty AuditTracer defines __len__ == 0, which is
        # falsy — `tracer or AuditTracer()` would silently substitute a
        # fresh tracer instead of using the one the caller passed in.
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        assert agent.tracer is tracer

    def test_wrapped_property_exposes_the_underlying_agent(self):
        inner = DummyAgent()
        agent = VerifiableAgent(inner, tracer=AuditTracer())
        assert agent.wrapped is inner

    def test_pii_is_redacted_before_recording(self):
        from synapsekit.audit.redact import PIIRedactor

        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer, redactor=PIIRedactor())
        agent.retrieve("contact bob@example.com")
        payload = tracer.records[0].payload
        assert "bob@example.com" not in str(payload)

    def test_default_actor_names_the_wrapped_agent_type(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer)
        agent.retrieve("q")
        assert tracer.records[0].actor == "agent:DummyAgent"

    def test_custom_actor_is_used(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(DummyAgent(), tracer=tracer, actor="user:alice")
        agent.retrieve("q")
        assert tracer.records[0].actor == "user:alice"


class _Retriever:
    async def search(self, query: str) -> list[str]:
        return [f"doc-for-{query}"]


class _AgentWithSubComponent:
    def __init__(self) -> None:
        self.retriever = _Retriever()

    async def run(self, query: str) -> str:
        # An internal self-call — NOT made through the outer proxy.
        docs = await self.retriever.search(query)
        return f"answer using {docs}"


class TestInstrumentAttrs:
    """Closes the gap where an agent's own internal calls (not made
    through the outer proxy) were invisible to the tracer."""

    @pytest.mark.asyncio
    async def test_internal_self_call_to_an_instrumented_sub_attr_is_captured(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(
            _AgentWithSubComponent(), tracer=tracer, instrument_attrs=["retriever"]
        )

        result = await agent.run("hello")

        assert result == "answer using ['doc-for-hello']"
        # "run" -> paired (TOOL_CALL, TOOL_RESULT); "search" -> RETRIEVAL (single).
        kinds = sorted(r.kind for r in tracer.records)
        assert kinds == [
            EventKind.RETRIEVAL.value,
            EventKind.TOOL_CALL.value,
            EventKind.TOOL_RESULT.value,
        ]

    @pytest.mark.asyncio
    async def test_captured_internal_call_has_the_outer_call_as_its_parent(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(
            _AgentWithSubComponent(), tracer=tracer, instrument_attrs=["retriever"]
        )
        await agent.run("hello")

        call_record = next(r for r in tracer.records if r.kind == EventKind.TOOL_CALL.value)
        search_record = next(r for r in tracer.records if r.kind == EventKind.RETRIEVAL.value)
        result_record = next(r for r in tracer.records if r.kind == EventKind.TOOL_RESULT.value)
        assert search_record.parent_id == call_record.event_id
        assert result_record.parent_id == call_record.event_id

    @pytest.mark.asyncio
    async def test_without_instrument_attrs_the_internal_call_is_invisible(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(_AgentWithSubComponent(), tracer=tracer)
        await agent.run("hello")

        # Documents the known limitation: only the outermost proxied call
        # is captured (as its call/result pair) unless the sub-attribute
        # is explicitly instrumented — the internal `search` call never
        # appears.
        assert len(tracer.records) == 2
        assert all(r.payload["method"] == "run" for r in tracer.records)
        assert {r.kind for r in tracer.records} == {
            EventKind.TOOL_CALL.value,
            EventKind.TOOL_RESULT.value,
        }

    def test_instrumenting_a_missing_attr_is_a_no_op(self):
        tracer = AuditTracer()
        # Should not raise even though DummyAgent has no 'nonexistent' attr.
        VerifiableAgent(DummyAgent(), tracer=tracer, instrument_attrs=["nonexistent"])


class TestAuditedDecorator:
    @pytest.mark.asyncio
    async def test_decorates_a_standalone_async_tool(self):
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        async def search(query: str) -> str:
            return f"results for {query}"

        result = await search("weather")
        assert result == "results for weather"
        assert len(tracer.records) == 1
        assert tracer.records[0].kind == EventKind.TOOL_CALL.value

    def test_decorates_a_standalone_sync_tool(self):
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5
        assert tracer.records[0].payload["output"] == 5

    @pytest.mark.asyncio
    async def test_nested_audited_calls_get_correct_parent_id(self):
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer, name="inner")
        async def inner() -> str:
            return "done"

        @audited(EventKind.DECISION, tracer=tracer, name="outer")
        async def outer() -> str:
            return await inner()

        await outer()
        records = {r.payload["method"]: r for r in tracer.records}
        assert records["inner"].parent_id == records["outer"].event_id

    @pytest.mark.asyncio
    async def test_exception_is_recorded_with_error_kind_regardless_of_declared_kind(self):
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        async def boom() -> None:
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            await boom()
        assert tracer.records[0].kind == EventKind.ERROR.value

    def test_default_actor_is_system(self):
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        def noop() -> None:
            return None

        noop()
        assert tracer.records[0].actor == "system"

    def test_async_wrapper_stays_a_coroutine_function(self):
        # Regression guard per testing standards: the decorator must
        # preserve coroutine-ness so `await` still works.
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        async def tool() -> str:
            return "ok"

        assert inspect.iscoroutinefunction(tool)


class TestAuditedCancellation:
    """Regression for #809: cancelling an audited coroutine must propagate
    CancelledError (a BaseException) and still write an audit record — the
    old code raised UnboundLocalError from `finally` because `emitted_kind`
    was never assigned on the BaseException path.
    """

    @pytest.mark.asyncio
    async def test_cancelling_an_audited_coroutine_propagates_cancelled_error(self):
        tracer = AuditTracer()
        started = asyncio.Event()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        async def slow() -> str:
            started.set()
            await asyncio.sleep(3600)
            return "never"

        task = asyncio.ensure_future(slow())
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # A record MUST still have been written — not swallowed, and not
        # an UnboundLocalError.
        assert len(tracer.records) == 1
        rec = tracer.records[0]
        assert rec.kind == EventKind.ERROR.value
        assert rec.payload["status"] == "cancelled"
        assert rec.payload["error_type"] == "CancelledError"

    def test_sync_wrapper_records_on_keyboard_interrupt_then_reraises(self):
        # KeyboardInterrupt is also a BaseException, not an Exception —
        # same failure mode as CancelledError on the sync path.
        tracer = AuditTracer()

        @audited(EventKind.TOOL_CALL, tracer=tracer)
        def boom() -> None:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            boom()

        assert len(tracer.records) == 1
        assert tracer.records[0].kind == EventKind.ERROR.value
        assert tracer.records[0].payload["status"] == "cancelled"


class _StreamingAgent:
    """An agent whose streaming methods are async generators."""

    # "stream" infers to SYSTEM_EVENT (single record). "stream_steps"
    # infers to a paired (TOOL_CALL, TOOL_RESULT) kind — both must route
    # through the async-generator wrapper, not the sync one.
    async def stream(self, prompt: str):
        for token in ("Hel", "lo ", "world"):
            yield token

    async def stream_steps(self, prompt: str):
        for token in ("Hel", "lo ", "world"):
            yield token

    async def stream_failing(self, prompt: str):
        yield "partial-"
        raise ValueError("mid-stream boom")


class TestAsyncGeneratorInstrumentation:
    """Regression for #810: async-generator methods must not fall into the
    sync wrapper (which recorded status:ok with a non-deterministic
    "<async_generator object at 0x...>" output before any token, and never
    recorded mid-stream errors).
    """

    @pytest.mark.asyncio
    async def test_single_kind_async_gen_records_deterministic_aggregate(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(_StreamingAgent(), tracer=tracer)

        chunks = [chunk async for chunk in agent.stream("hi")]
        assert chunks == ["Hel", "lo ", "world"]

        # "stream" -> SYSTEM_EVENT (single record after completion).
        assert len(tracer.records) == 1
        result = tracer.records[0]
        assert result.kind == EventKind.SYSTEM_EVENT.value
        assert result.payload["status"] == "ok"
        # Deterministic aggregate — the joined stream, NOT a repr/address.
        assert result.payload["output"] == "Hello world"
        assert result.payload["chunk_count"] == 3
        assert "0x" not in str(result.payload["output"])
        assert "async_generator" not in str(result.payload)

    @pytest.mark.asyncio
    async def test_paired_kind_async_gen_emits_call_then_completed_result(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(_StreamingAgent(), tracer=tracer)

        chunks = [chunk async for chunk in agent.stream_steps("hi")]
        assert chunks == ["Hel", "lo ", "world"]

        # "stream_steps" -> paired (TOOL_CALL, TOOL_RESULT).
        assert len(tracer.records) == 2
        call, result = tracer.records
        assert call.kind == EventKind.TOOL_CALL.value
        assert "output" not in call.payload  # pre-call has no output
        assert result.kind == EventKind.TOOL_RESULT.value
        assert result.parent_id == call.event_id
        assert result.payload["output"] == "Hello world"

    @pytest.mark.asyncio
    async def test_recorded_output_is_stable_across_runs(self):
        # The whole point of the fix: the payload must hash the same every
        # run. A memory address in the output would break the hash chain.
        outputs = []
        for _ in range(2):
            tracer = AuditTracer()
            agent = VerifiableAgent(_StreamingAgent(), tracer=tracer)
            [c async for c in agent.stream("hi")]
            outputs.append(tracer.records[-1].payload["output"])
        assert outputs[0] == outputs[1]

    @pytest.mark.asyncio
    async def test_the_wrapper_is_still_an_async_generator(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(_StreamingAgent(), tracer=tracer)
        assert inspect.isasyncgen(agent.stream("hi"))

    @pytest.mark.asyncio
    async def test_result_is_only_recorded_after_iteration_completes(self):
        # The buggy sync wrapper recorded immediately (before any token)
        # with the generator repr as output; the async-gen wrapper must
        # only finalize the result record after iteration completes.
        tracer = AuditTracer()
        agent = VerifiableAgent(_StreamingAgent(), tracer=tracer)
        gen = agent.stream_steps("hi")
        first = await gen.__anext__()
        assert first == "Hel"
        # The pre-invocation call exists, but NOT the result yet.
        assert [r.kind for r in tracer.records] == [EventKind.TOOL_CALL.value]
        async for _ in gen:  # drain so it finalizes cleanly
            pass
        assert tracer.records[-1].kind == EventKind.TOOL_RESULT.value

    @pytest.mark.asyncio
    async def test_mid_stream_error_is_recorded_and_reraised(self):
        tracer = AuditTracer()
        agent = VerifiableAgent(_StreamingAgent(), tracer=tracer)

        collected = []
        with pytest.raises(ValueError, match="mid-stream boom"):
            async for chunk in agent.stream_failing("hi"):
                collected.append(chunk)

        assert collected == ["partial-"]
        # An ERROR record must be written (the old sync wrapper never saw
        # the mid-stream failure at all).
        error = tracer.records[-1]
        assert error.kind == EventKind.ERROR.value
        assert error.payload["status"] == "error"
        assert error.payload["error_type"] == "ValueError"
        # What streamed before the failure is captured deterministically.
        # (Frozen payload: lists become tuples — see types.deep_freeze.)
        assert error.payload["output"] == ("partial-",)
        assert error.payload["chunk_count"] == 1
        assert "0x" not in str(error.payload)


class _CitationRetriever:
    async def search(self, query: str) -> list[dict]:
        return [
            {
                "text": "private retrieved passage",
                "metadata": {"source_id": "rag-document-1"},
            }
        ]


class _AgentWithCitationRetriever:
    def __init__(self) -> None:
        self.retriever = _CitationRetriever()

    async def run(self, query: str) -> dict[str, object]:
        documents = await self.retriever.search(query)
        return {
            "answer": f"Answer citing [rag-document-1] from {len(documents)} source.",
            "citation_ids": ["rag-document-1"],
        }


class TestRetrievalSourceProvenance:
    @pytest.mark.asyncio
    async def test_nested_retriever_source_is_not_linked_from_tool_result_citation(self):
        tracer = AuditTracer(run_id="rag-tool-run")
        agent = VerifiableAgent(
            _AgentWithCitationRetriever(), tracer=tracer, instrument_attrs=["retriever"]
        )

        answer = await agent.run("question")
        provenance = build_provenance(tracer.records)

        assert "[rag-document-1]" in answer["answer"]
        source_entity = next(
            identifier
            for identifier, entity in provenance["entity"].items()
            if entity.get("sk:sourceIdHash") == source_id_hash("rag-document-1")
        )
        answer_activity = next(
            identifier
            for identifier, activity in provenance["activity"].items()
            if activity["sk:kind"] == EventKind.TOOL_RESULT.value
        )
        answer_entity = next(
            relation["prov:entity"]
            for relation in provenance["wasGeneratedBy"].values()
            if relation["prov:activity"] == answer_activity
        )
        assert not any(
            relation["prov:generatedEntity"] == answer_entity
            and relation["prov:usedEntity"] == source_entity
            for relation in provenance["wasDerivedFrom"].values()
        )
        assert any(
            entity.get("sk:sourceIdHash") == source_id_hash("rag-document-1")
            for entity in provenance["entity"].values()
        )
        assert "private retrieved passage" not in str(provenance)
