"""Tests for EventTrigger and StreamTrigger — all external I/O mocked."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.agents.triggers.cron import TriggerResult
from synapsekit.agents.triggers.event import EventTrigger
from synapsekit.agents.triggers.stream import StreamTrigger

# ===========================================================================
# EventTrigger
# ===========================================================================


class TestEventTriggerConstruction:
    def test_defaults(self):
        async def agent_fn(p):
            return "ok"

        t = EventTrigger(agent_fn)
        assert t._host == "127.0.0.1"
        assert t._port == 8765
        assert t._path == "/webhook"
        assert t._secret is None
        assert t._result_sink is None

    def test_custom_params(self):
        async def agent_fn(p):
            return "ok"

        t = EventTrigger(agent_fn, host="0.0.0.0", port=9000, path="/hook", secret="mysecret")
        assert t._host == "0.0.0.0"
        assert t._port == 9000
        assert t._path == "/hook"
        assert t._secret == "mysecret"


class TestEventTriggerSignature:
    def test_verify_signature_no_secret(self):
        async def fn(p):
            return "x"

        t = EventTrigger(fn)
        assert t._verify_signature(b"body", None) is True
        assert t._verify_signature(b"body", "anything") is True

    def test_verify_signature_valid(self):
        secret = "supersecret"

        async def fn(p):
            return "x"

        t = EventTrigger(fn, secret=secret)
        body = b'{"event": "push"}'
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert t._verify_signature(body, expected) is True

    def test_verify_signature_invalid(self):
        async def fn(p):
            return "x"

        t = EventTrigger(fn, secret="secret")
        assert t._verify_signature(b"body", "wrongsig") is False

    def test_verify_signature_missing_header(self):
        async def fn(p):
            return "x"

        t = EventTrigger(fn, secret="secret")
        assert t._verify_signature(b"body", None) is False


def _make_aiohttp_mock():
    """Return a mock aiohttp.web module with Response as a dict-returning callable."""
    mock_web = MagicMock()
    mock_web.Response = MagicMock(side_effect=lambda **kw: kw)
    mock_web.Application = MagicMock(return_value=MagicMock())
    mock_web.AppRunner = MagicMock()
    mock_web.TCPSite = MagicMock()
    mock_aiohttp = MagicMock()
    mock_aiohttp.web = mock_web
    return mock_aiohttp, mock_web


class TestEventTriggerHandle:
    """Test the internal _handle method using a mock aiohttp request."""

    def _make_request(self, body: bytes, headers: dict | None = None):
        req = MagicMock()
        req.read = AsyncMock(return_value=body)
        req.headers = headers or {}
        return req

    def _run_handle(self, t, req, mock_aiohttp, mock_web):
        """Patch aiohttp in sys.modules and run _handle."""
        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp, "aiohttp.web": mock_web}):
            import synapsekit.agents.triggers.event as ev_mod

            # Patch the module-level reference if present
            original = getattr(ev_mod, "web", None)
            ev_mod.web = mock_web  # type: ignore[attr-defined]
            try:
                import asyncio

                return asyncio.get_event_loop().run_until_complete(t._handle(req))
            finally:
                if original is not None:
                    ev_mod.web = original  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_handle_valid_request(self):
        async def agent_fn(payload):
            return f"received: {payload.get('key')}"

        t = EventTrigger(agent_fn)
        body = json.dumps({"key": "value"}).encode()
        req = self._make_request(body)
        mock_aiohttp, mock_web = _make_aiohttp_mock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp, "aiohttp.web": mock_web}):
            response = await t._handle(req)

        assert response["status"] == 200
        assert json.loads(response["text"])["output"] == "received: value"

    @pytest.mark.asyncio
    async def test_handle_unauthorized_bad_sig(self):
        async def agent_fn(payload):
            return "ok"

        t = EventTrigger(agent_fn, secret="secret")
        body = b'{"x": 1}'
        req = self._make_request(body, headers={"X-Webhook-Secret": "badsig"})
        mock_aiohttp, mock_web = _make_aiohttp_mock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp, "aiohttp.web": mock_web}):
            response = await t._handle(req)

        assert response["status"] == 401

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self):
        async def agent_fn(payload):
            return "ok"

        t = EventTrigger(agent_fn)
        req = self._make_request(b"not json at all {")
        mock_aiohttp, mock_web = _make_aiohttp_mock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp, "aiohttp.web": mock_web}):
            response = await t._handle(req)

        assert response["status"] == 400

    @pytest.mark.asyncio
    async def test_handle_agent_fn_exception(self):
        async def agent_fn(payload):
            raise ValueError("agent exploded")

        t = EventTrigger(agent_fn)
        body = json.dumps({"key": "val"}).encode()
        req = self._make_request(body)
        mock_aiohttp, mock_web = _make_aiohttp_mock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp, "aiohttp.web": mock_web}):
            response = await t._handle(req)

        assert response["status"] == 500
        assert "agent exploded" in response["text"]

    @pytest.mark.asyncio
    async def test_handle_calls_result_sink(self):
        results = []

        async def agent_fn(payload):
            return "done"

        async def sink(r: TriggerResult):
            results.append(r)

        t = EventTrigger(agent_fn, result_sink=sink)
        body = json.dumps({"x": 1}).encode()
        req = self._make_request(body)
        mock_aiohttp, mock_web = _make_aiohttp_mock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp, "aiohttp.web": mock_web}):
            await t._handle(req)

        assert len(results) == 1
        assert isinstance(results[0], TriggerResult)
        assert results[0].output == "done"

    @pytest.mark.asyncio
    async def test_handle_sync_result_sink(self):
        results = []

        async def agent_fn(payload):
            return "hello"

        def sink(r: TriggerResult):
            results.append(r)

        t = EventTrigger(agent_fn, result_sink=sink)
        body = json.dumps({}).encode()
        req = self._make_request(body)
        mock_aiohttp, mock_web = _make_aiohttp_mock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp, "aiohttp.web": mock_web}):
            await t._handle(req)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_handle_valid_hmac(self):
        secret = "mysecret"

        async def agent_fn(payload):
            return "ok"

        t = EventTrigger(agent_fn, secret=secret)
        body = json.dumps({"event": "test"}).encode()
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        req = self._make_request(body, headers={"X-Webhook-Secret": sig})
        mock_aiohttp, mock_web = _make_aiohttp_mock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp, "aiohttp.web": mock_web}):
            response = await t._handle(req)

        assert response["status"] == 200

    @pytest.mark.asyncio
    async def test_handle_empty_body(self):
        async def agent_fn(payload):
            assert payload == {}
            return "empty"

        t = EventTrigger(agent_fn)
        req = self._make_request(b"")
        mock_aiohttp, mock_web = _make_aiohttp_mock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp, "aiohttp.web": mock_web}):
            response = await t._handle(req)

        assert response["status"] == 200


class TestEventTriggerStartStop:
    @pytest.mark.asyncio
    async def test_import_error_without_aiohttp(self):
        async def fn(p):
            return "x"

        t = EventTrigger(fn)
        with patch.dict("sys.modules", {"aiohttp": None, "aiohttp.web": None}):
            with pytest.raises(ImportError, match="aiohttp package required"):
                await t.start()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        async def fn(p):
            return "x"

        t = EventTrigger(fn)
        # Should not raise
        await t.stop()

    @pytest.mark.asyncio
    async def test_start_stop(self):
        async def fn(p):
            return "x"

        t = EventTrigger(fn)

        mock_runner = MagicMock()
        mock_runner.setup = AsyncMock()
        mock_runner.cleanup = AsyncMock()
        mock_site = MagicMock()
        mock_site.start = AsyncMock()
        mock_app = MagicMock()
        mock_app.router = MagicMock()
        mock_web = MagicMock()
        mock_web.Application = MagicMock(return_value=mock_app)
        mock_web.AppRunner = MagicMock(return_value=mock_runner)
        mock_web.TCPSite = MagicMock(return_value=mock_site)
        mock_aiohttp = MagicMock()
        mock_aiohttp.web = mock_web

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp, "aiohttp.web": mock_web}):
            await t.start()
            await t.stop()

        mock_runner.setup.assert_called_once()
        mock_site.start.assert_called_once()
        mock_runner.cleanup.assert_called_once()


class TestEventTriggerMethods:
    def test_start_is_coroutine(self):
        async def fn(p):
            return "x"

        t = EventTrigger(fn)
        assert inspect.iscoroutinefunction(t.start)

    def test_stop_is_coroutine(self):
        async def fn(p):
            return "x"

        t = EventTrigger(fn)
        assert inspect.iscoroutinefunction(t.stop)

    def test_run_forever_is_coroutine(self):
        async def fn(p):
            return "x"

        t = EventTrigger(fn)
        assert inspect.iscoroutinefunction(t.run_forever)


# ===========================================================================
# StreamTrigger
# ===========================================================================


class TestStreamTriggerConstruction:
    def test_defaults_redis(self):
        async def fn(m):
            return "ok"

        t = StreamTrigger(fn, topic="my-stream")
        assert t._backend == "redis"
        assert t._topic == "my-stream"
        assert t._group_name == "synapsekit"
        assert t._consumer_name == "agent-1"
        assert t._batch_size == 1
        assert t._poll_interval == 1.0

    def test_kafka_backend(self):
        async def fn(m):
            return "ok"

        t = StreamTrigger(fn, backend="kafka", topic="my-topic", kafka_brokers="b:9092")
        assert t._backend == "kafka"
        assert t._kafka_brokers == "b:9092"

    def test_invalid_backend(self):
        async def fn(m):
            return "ok"

        with pytest.raises(ValueError, match="backend must be"):
            StreamTrigger(fn, backend="sqs", topic="t")

    def test_default_redis_url(self):
        async def fn(m):
            return "ok"

        t = StreamTrigger(fn, topic="t")
        assert t._redis_url == "redis://localhost:6379"

    def test_default_kafka_brokers(self):
        async def fn(m):
            return "ok"

        t = StreamTrigger(fn, backend="kafka", topic="t")
        assert t._kafka_brokers == "localhost:9092"


class TestStreamTriggerImportErrors:
    @pytest.mark.asyncio
    async def test_redis_import_error(self):
        async def fn(m):
            return "ok"

        t = StreamTrigger(fn, backend="redis", topic="t")
        with patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}):
            with pytest.raises(ImportError, match="redis package required"):
                await t._init_redis()

    @pytest.mark.asyncio
    async def test_kafka_import_error(self):
        async def fn(m):
            return "ok"

        t = StreamTrigger(fn, backend="kafka", topic="t")
        with patch.dict("sys.modules", {"aiokafka": None}):
            with pytest.raises(ImportError, match="aiokafka package required"):
                await t._init_kafka()


class TestStreamTriggerDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_calls_agent_fn(self):
        received = []

        async def agent_fn(payload):
            received.append(payload)
            return "dispatched"

        t = StreamTrigger(agent_fn, topic="t")
        payload = {"key": "value"}
        await t._dispatch(payload)

        assert received == [payload]

    @pytest.mark.asyncio
    async def test_dispatch_stores_trigger_result(self):
        results = []

        async def agent_fn(payload):
            return "result"

        async def sink(r: TriggerResult):
            results.append(r)

        t = StreamTrigger(agent_fn, topic="t", result_sink=sink)
        await t._dispatch({"x": 1})

        assert len(results) == 1
        assert results[0].output == "result"
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_dispatch_captures_agent_fn_error(self):
        results = []

        async def agent_fn(payload):
            raise RuntimeError("stream error")

        def sink(r: TriggerResult):
            results.append(r)

        t = StreamTrigger(agent_fn, topic="t", result_sink=sink)
        await t._dispatch({})

        assert results[0].error is not None
        assert "stream error" in results[0].error

    @pytest.mark.asyncio
    async def test_dispatch_acks_redis_message(self):
        async def agent_fn(payload):
            return "ok"

        t = StreamTrigger(agent_fn, topic="my-stream")
        mock_redis = MagicMock()
        mock_redis.xack = AsyncMock()
        t._redis_client = mock_redis

        await t._dispatch({"x": 1}, ack_id="1234-0")

        mock_redis.xack.assert_called_once_with("my-stream", "synapsekit", "1234-0")

    @pytest.mark.asyncio
    async def test_dispatch_acks_kafka_message(self):
        async def agent_fn(payload):
            return "ok"

        committed = []

        async def mock_commit():
            committed.append(True)

        t = StreamTrigger(agent_fn, backend="kafka", topic="t")
        await t._dispatch({}, ack_fn=mock_commit)

        assert committed == [True]

    @pytest.mark.asyncio
    async def test_dispatch_sync_result_sink(self):
        results = []

        async def agent_fn(payload):
            return "x"

        def sync_sink(r: TriggerResult):
            results.append(r)

        t = StreamTrigger(agent_fn, topic="t", result_sink=sync_sink)
        await t._dispatch({})
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dispatch_metadata_includes_backend_topic(self):
        results = []

        async def agent_fn(payload):
            return "y"

        def sink(r: TriggerResult):
            results.append(r)

        t = StreamTrigger(agent_fn, topic="events", result_sink=sink)
        await t._dispatch({})

        meta = results[0].metadata
        assert meta["backend"] == "redis"
        assert meta["topic"] == "events"


class TestStreamTriggerStartStop:
    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        async def fn(m):
            return "ok"

        t = StreamTrigger(fn, topic="t")
        await t.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_start_creates_background_task_redis(self):
        async def fn(m):
            return "ok"

        t = StreamTrigger(fn, topic="t")

        mock_redis = MagicMock()
        mock_redis.aclose = AsyncMock()

        async def fake_init_redis():
            t._redis_client = mock_redis

        async def fake_consume():
            # Never exits naturally; we cancel it
            await asyncio.sleep(10)

        with (
            patch.object(t, "_init_redis", fake_init_redis),
            patch.object(t, "_consume_loop", fake_consume),
        ):
            await t.start()
            assert t._task is not None
            await t.stop()
            assert t._task is None

    @pytest.mark.asyncio
    async def test_start_creates_background_task_kafka(self):
        async def fn(m):
            return "ok"

        t = StreamTrigger(fn, backend="kafka", topic="t")

        mock_consumer = MagicMock()
        mock_consumer.stop = AsyncMock()

        async def fake_init_kafka():
            t._kafka_consumer = mock_consumer

        async def fake_consume():
            await asyncio.sleep(10)

        with (
            patch.object(t, "_init_kafka", fake_init_kafka),
            patch.object(t, "_consume_loop", fake_consume),
        ):
            await t.start()
            assert t._task is not None
            await t.stop()

        mock_consumer.stop.assert_called_once()


class TestStreamTriggerRedisLoop:
    @pytest.mark.asyncio
    async def test_consume_loop_redis_processes_messages(self):
        dispatched = []

        async def agent_fn(payload):
            dispatched.append(payload)
            return "ok"

        t = StreamTrigger(agent_fn, topic="events")

        # Simulate one batch then stop
        call_count = 0

        async def fake_xreadgroup(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [("events", [("1-0", {"field": "value"})])]
            t._stop_event.set()
            return []

        mock_redis = MagicMock()
        mock_redis.xreadgroup = fake_xreadgroup
        mock_redis.xack = AsyncMock()
        t._redis_client = mock_redis

        await t._consume_loop_redis()

        assert len(dispatched) == 1
        assert dispatched[0] == {"field": "value"}


class TestStreamTriggerMethods:
    def test_start_is_coroutine(self):
        async def fn(m):
            return "ok"

        assert inspect.iscoroutinefunction(StreamTrigger(fn, topic="t").start)

    def test_stop_is_coroutine(self):
        async def fn(m):
            return "ok"

        assert inspect.iscoroutinefunction(StreamTrigger(fn, topic="t").stop)


# ===========================================================================
# __init__.py exports
# ===========================================================================


def test_trigger_exports():
    from synapsekit.agents.triggers import CronTrigger, EventTrigger, StreamTrigger, TriggerResult

    assert EventTrigger is not None
    assert StreamTrigger is not None
    assert TriggerResult is not None
    assert CronTrigger is not None


def test_agents_init_exports():
    from synapsekit.agents import EventTrigger, StreamTrigger, TriggerResult

    assert EventTrigger is not None
    assert StreamTrigger is not None
    assert TriggerResult is not None
