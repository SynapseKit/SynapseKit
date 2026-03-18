"""Tests for v0.9.0 features: A2A protocol, guardrails, distributed tracing."""

import time

from synapsekit.a2a import A2AClient, A2AMessage, A2AServer, A2ATask, AgentCard
from synapsekit.agents.guardrails import (
    ContentFilter,
    GuardrailResult,
    Guardrails,
    PIIDetector,
    TopicRestrictor,
)
from synapsekit.observability.distributed import DistributedTracer, TraceSpan

# ---------------------------------------------------------------------------
# Mock executor for A2A tests
# ---------------------------------------------------------------------------


class MockExecutor:
    async def run(self, query):
        return f"Response to: {query}"


class FailingExecutor:
    async def run(self, query):
        raise RuntimeError("executor failed")


# ===========================================================================
# A2A Types
# ===========================================================================


class TestA2AMessage:
    def test_construction(self):
        msg = A2AMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.metadata == {}

    def test_construction_with_metadata(self):
        msg = A2AMessage(role="agent", content="hi", metadata={"key": "val"})
        assert msg.metadata == {"key": "val"}

    def test_role_literal(self):
        msg = A2AMessage(role="agent", content="response")
        assert msg.role == "agent"


class TestA2ATask:
    def test_construction_defaults(self):
        task = A2ATask(id="t1")
        assert task.id == "t1"
        assert task.state == "pending"
        assert task.messages == []
        assert task.artifacts == []
        assert task.metadata == {}

    def test_add_message(self):
        task = A2ATask(id="t1")
        task.add_message("user", "hello")
        assert len(task.messages) == 1
        assert task.messages[0].role == "user"
        assert task.messages[0].content == "hello"

    def test_add_multiple_messages(self):
        task = A2ATask(id="t1")
        task.add_message("user", "q")
        task.add_message("agent", "a")
        assert len(task.messages) == 2

    def test_to_dict(self):
        task = A2ATask(id="t1", state="completed")
        task.add_message("user", "hi")
        d = task.to_dict()
        assert d["id"] == "t1"
        assert d["state"] == "completed"
        assert len(d["messages"]) == 1
        assert d["messages"][0]["role"] == "user"
        assert d["messages"][0]["content"] == "hi"

    def test_to_dict_with_artifacts(self):
        task = A2ATask(id="t2", artifacts=[{"type": "text", "data": "result"}])
        d = task.to_dict()
        assert len(d["artifacts"]) == 1

    def test_state_values(self):
        for state in ("pending", "running", "completed", "failed", "cancelled"):
            task = A2ATask(id="t", state=state)
            assert task.state == state


class TestAgentCard:
    def test_construction(self):
        card = AgentCard(name="test", description="A test agent")
        assert card.name == "test"
        assert card.description == "A test agent"
        assert card.skills == []
        assert card.endpoint == ""
        assert card.version == "1.0.0"

    def test_construction_full(self):
        card = AgentCard(
            name="research",
            description="Research agent",
            skills=["search", "summarize"],
            endpoint="http://localhost:8001",
            version="2.0.0",
            metadata={"author": "test"},
        )
        assert card.skills == ["search", "summarize"]
        assert card.endpoint == "http://localhost:8001"

    def test_to_dict(self):
        card = AgentCard(name="x", description="y", skills=["a"])
        d = card.to_dict()
        assert d["name"] == "x"
        assert d["description"] == "y"
        assert d["skills"] == ["a"]
        assert d["version"] == "1.0.0"


# ===========================================================================
# A2A Client
# ===========================================================================


class TestA2AClient:
    def test_import(self):
        from synapsekit.a2a.client import A2AClient as C

        assert C is not None

    def test_construction(self):
        client = A2AClient(endpoint="http://localhost:8001")
        assert client._endpoint == "http://localhost:8001"

    def test_endpoint_trailing_slash_stripped(self):
        client = A2AClient(endpoint="http://localhost:8001/")
        assert client._endpoint == "http://localhost:8001"


# ===========================================================================
# A2A Server
# ===========================================================================


class TestA2AServer:
    def test_import(self):
        from synapsekit.a2a.server import A2AServer as S

        assert S is not None

    def test_construction(self):
        executor = MockExecutor()
        card = AgentCard(name="test", description="test agent")
        server = A2AServer(executor=executor, card=card)
        assert server._card.name == "test"
        assert server._tasks == {}

    async def test_handle_send(self):
        executor = MockExecutor()
        card = AgentCard(name="test", description="test")
        server = A2AServer(executor=executor, card=card)

        body = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "id": "req-1",
            "params": {
                "id": "task-1",
                "message": {"role": "user", "content": "hello"},
            },
        }
        result = await server.handle_request(body)
        assert result["jsonrpc"] == "2.0"
        assert result["result"]["state"] == "completed"
        assert any(
            m["content"] == "Response to: hello"
            for m in result["result"]["messages"]
            if m["role"] == "agent"
        )

    async def test_handle_send_stores_task(self):
        executor = MockExecutor()
        card = AgentCard(name="test", description="test")
        server = A2AServer(executor=executor, card=card)

        body = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "id": "req-1",
            "params": {"id": "task-1", "message": {"role": "user", "content": "hi"}},
        }
        await server.handle_request(body)
        assert "task-1" in server._tasks

    async def test_handle_get_existing_task(self):
        executor = MockExecutor()
        card = AgentCard(name="test", description="test")
        server = A2AServer(executor=executor, card=card)

        # First send a task
        send_body = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "id": "r1",
            "params": {"id": "t1", "message": {"role": "user", "content": "x"}},
        }
        await server.handle_request(send_body)

        # Then get it
        get_body = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "id": "r2",
            "params": {"id": "t1"},
        }
        result = await server.handle_request(get_body)
        assert result["result"]["id"] == "t1"
        assert result["result"]["state"] == "completed"

    async def test_handle_get_missing_task(self):
        executor = MockExecutor()
        card = AgentCard(name="test", description="test")
        server = A2AServer(executor=executor, card=card)

        body = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "id": "r1",
            "params": {"id": "nonexistent"},
        }
        result = await server.handle_request(body)
        assert "error" in result
        assert result["error"]["code"] == -32602

    async def test_handle_unknown_method(self):
        executor = MockExecutor()
        card = AgentCard(name="test", description="test")
        server = A2AServer(executor=executor, card=card)

        body = {
            "jsonrpc": "2.0",
            "method": "tasks/unknown",
            "id": "r1",
            "params": {},
        }
        result = await server.handle_request(body)
        assert "error" in result
        assert result["error"]["code"] == -32601

    async def test_handle_send_executor_failure(self):
        executor = FailingExecutor()
        card = AgentCard(name="test", description="test")
        server = A2AServer(executor=executor, card=card)

        body = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "id": "r1",
            "params": {"id": "t1", "message": {"role": "user", "content": "fail"}},
        }
        result = await server.handle_request(body)
        assert result["result"]["state"] == "failed"


# ===========================================================================
# Guardrails
# ===========================================================================


class TestGuardrailResult:
    def test_passed(self):
        r = GuardrailResult(passed=True)
        assert r.passed is True
        assert r.violations == []

    def test_failed(self):
        r = GuardrailResult(passed=False, violations=["bad"])
        assert r.passed is False
        assert r.violations == ["bad"]

    def test_repr_passed(self):
        r = GuardrailResult(passed=True)
        assert repr(r) == "GuardrailResult(passed=True)"

    def test_repr_failed(self):
        r = GuardrailResult(passed=False, violations=["v1"])
        assert "passed=False" in repr(r)
        assert "v1" in repr(r)


class TestContentFilter:
    def test_blocked_pattern(self):
        f = ContentFilter(blocked_patterns=[r"password\s*[:=]"])
        result = f.check("My password: abc123")
        assert not result.passed
        assert len(result.violations) == 1

    def test_blocked_words(self):
        f = ContentFilter(blocked_words=["hack", "exploit"])
        result = f.check("How to hack a system")
        assert not result.passed

    def test_max_length(self):
        f = ContentFilter(max_length=10)
        result = f.check("This is way too long")
        assert not result.passed
        assert "max length" in result.violations[0]

    def test_passes_clean_text(self):
        f = ContentFilter(
            blocked_patterns=[r"password\s*[:=]"],
            blocked_words=["hack"],
            max_length=1000,
        )
        result = f.check("Hello world")
        assert result.passed

    def test_multiple_violations(self):
        f = ContentFilter(blocked_words=["hack", "exploit"])
        result = f.check("hack and exploit")
        assert len(result.violations) == 2


class TestPIIDetector:
    def test_email(self):
        d = PIIDetector()
        result = d.check("Email me at john@example.com")
        assert not result.passed
        assert any("email" in v for v in result.violations)

    def test_phone(self):
        d = PIIDetector()
        result = d.check("Call 555-123-4567")
        assert not result.passed

    def test_ssn(self):
        d = PIIDetector()
        result = d.check("SSN: 123-45-6789")
        assert not result.passed

    def test_credit_card(self):
        d = PIIDetector()
        result = d.check("Card: 4111 1111 1111 1111")
        assert not result.passed

    def test_ip_address(self):
        d = PIIDetector()
        result = d.check("Server at 192.168.1.1")
        assert not result.passed

    def test_passes_clean_text(self):
        d = PIIDetector()
        result = d.check("Hello, how are you?")
        assert result.passed

    def test_custom_detect_list(self):
        d = PIIDetector(detect=["email"])
        result = d.check("Call 555-123-4567")
        assert result.passed  # phone not in detect list

        result2 = d.check("Email john@example.com")
        assert not result2.passed


class TestTopicRestrictor:
    def test_blocked_topic(self):
        r = TopicRestrictor(blocked_topics=["politics", "religion"])
        result = r.check("Let's talk about politics today")
        assert not result.passed

    def test_passes_clean_text(self):
        r = TopicRestrictor(blocked_topics=["politics"])
        result = r.check("Let's talk about technology")
        assert result.passed

    def test_multiple_blocked(self):
        r = TopicRestrictor(blocked_topics=["politics", "religion"])
        result = r.check("politics and religion")
        assert len(result.violations) == 2


class TestGuardrails:
    def test_composite_all_pass(self):
        g = Guardrails(
            checks=[
                ContentFilter(blocked_words=["hack"]),
                PIIDetector(detect=["email"]),
            ]
        )
        result = g.check("Hello world")
        assert result.passed

    def test_composite_some_fail(self):
        g = Guardrails(
            checks=[
                ContentFilter(blocked_words=["hack"]),
                PIIDetector(detect=["email"]),
            ]
        )
        result = g.check("hack my email john@test.com")
        assert not result.passed
        assert len(result.violations) == 2

    def test_add_check(self):
        g = Guardrails()
        g.add_check(ContentFilter(blocked_words=["bad"]))
        result = g.check("this is bad")
        assert not result.passed

    def test_empty_guardrails_pass(self):
        g = Guardrails()
        result = g.check("anything")
        assert result.passed


# ===========================================================================
# Distributed Tracing
# ===========================================================================


class TestTraceSpan:
    def test_construction(self):
        span = TraceSpan(trace_id="t1", span_id="s1", name="test")
        assert span.trace_id == "t1"
        assert span.span_id == "s1"
        assert span.name == "test"
        assert span.parent_span_id is None
        assert span.status == "ok"

    def test_end(self):
        span = TraceSpan(trace_id="t1", span_id="s1", name="test")
        assert span.end_time is None
        span.end()
        assert span.end_time is not None

    def test_duration_ms(self):
        span = TraceSpan(trace_id="t1", span_id="s1", name="test", start_time=time.time() - 0.1)
        span.end()
        assert span.duration_ms >= 90  # at least ~100ms

    def test_duration_ms_no_end(self):
        span = TraceSpan(trace_id="t1", span_id="s1", name="test", start_time=time.time() - 0.05)
        # duration_ms uses current time if not ended
        assert span.duration_ms >= 40

    def test_add_event(self):
        span = TraceSpan(trace_id="t1", span_id="s1", name="test")
        span.add_event("checkpoint", {"step": 1})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"
        assert span.events[0]["attributes"] == {"step": 1}

    def test_to_dict(self):
        span = TraceSpan(
            trace_id="t1",
            span_id="s1",
            name="test",
            attributes={"key": "val"},
        )
        d = span.to_dict()
        assert d["trace_id"] == "t1"
        assert d["span_id"] == "s1"
        assert d["name"] == "test"
        assert d["attributes"] == {"key": "val"}
        assert "duration_ms" in d


class TestDistributedTracer:
    def test_construction(self):
        tracer = DistributedTracer()
        assert len(tracer.trace_id) == 16

    def test_custom_trace_id(self):
        tracer = DistributedTracer(trace_id="custom123")
        assert tracer.trace_id == "custom123"

    def test_start_span(self):
        tracer = DistributedTracer()
        span = tracer.start_span("test.op")
        assert span.name == "test.op"
        assert span.trace_id == tracer.trace_id
        assert span.parent_span_id is None

    def test_parent_child(self):
        tracer = DistributedTracer()
        root = tracer.start_span("root")
        child = tracer.start_span("child", parent=root)
        assert child.parent_span_id == root.span_id

    def test_span_attributes(self):
        tracer = DistributedTracer()
        span = tracer.start_span("op", attributes={"model": "gpt-4"})
        assert span.attributes == {"model": "gpt-4"}

    def test_get_trace(self):
        tracer = DistributedTracer()
        tracer.start_span("a")
        tracer.start_span("b")
        trace = tracer.get_trace()
        assert len(trace) == 2
        assert all(isinstance(s, dict) for s in trace)

    def test_get_root_spans(self):
        tracer = DistributedTracer()
        root = tracer.start_span("root")
        tracer.start_span("child", parent=root)
        roots = tracer.get_root_spans()
        assert len(roots) == 1
        assert roots[0].name == "root"

    def test_get_children(self):
        tracer = DistributedTracer()
        root = tracer.start_span("root")
        tracer.start_span("c1", parent=root)
        tracer.start_span("c2", parent=root)
        children = tracer.get_children(root)
        assert len(children) == 2

    def test_total_duration(self):
        tracer = DistributedTracer()
        span = tracer.start_span("root")
        time.sleep(0.01)
        span.end()
        assert tracer.total_duration_ms >= 5

    def test_total_duration_empty(self):
        tracer = DistributedTracer()
        assert tracer.total_duration_ms == 0.0

    def test_span_count(self):
        tracer = DistributedTracer()
        tracer.start_span("a")
        tracer.start_span("b")
        tracer.start_span("c")
        assert tracer.span_count == 3

    def test_clear(self):
        tracer = DistributedTracer()
        tracer.start_span("a")
        tracer.start_span("b")
        tracer.clear()
        assert tracer.span_count == 0
        assert tracer.get_trace() == []
