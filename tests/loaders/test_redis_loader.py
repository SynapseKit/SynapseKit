from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import Document
from synapsekit.loaders.redis_loader import RedisLoader

# ---------------------------------------------------------------------------
# Initialisation validation
# ---------------------------------------------------------------------------


def test_init_requires_url() -> None:
    with pytest.raises(ValueError, match="url must be provided"):
        RedisLoader(url="")


def test_init_invalid_value_type() -> None:
    with pytest.raises(ValueError, match="value_type must be one of"):
        RedisLoader(url="redis://localhost:6379", value_type="xml")


def test_init_defaults() -> None:
    loader = RedisLoader(url="redis://localhost:6379")
    assert loader._pattern == "*"
    assert loader._value_type == "string"
    assert loader._limit is None


# ---------------------------------------------------------------------------
# Missing dependency
# ---------------------------------------------------------------------------


def test_load_import_error_missing_redis() -> None:
    with patch.dict("sys.modules", {"redis": None}):
        loader = RedisLoader(url="redis://localhost:6379")
        with pytest.raises(ImportError, match="redis required"):
            loader.load()


# ---------------------------------------------------------------------------
# String value type
# ---------------------------------------------------------------------------


@patch.dict("sys.modules", {"redis": MagicMock()})
def test_load_string_values() -> None:
    import sys

    mock_redis_class = sys.modules["redis"].Redis
    mock_client = MagicMock()
    mock_redis_class.from_url.return_value = mock_client

    mock_client.keys.return_value = ["key:1", "key:2"]
    mock_client.get.side_effect = ["hello", "world"]

    loader = RedisLoader(url="redis://localhost:6379", pattern="key:*")
    docs = loader.load()

    assert len(docs) == 2
    assert all(isinstance(doc, Document) for doc in docs)
    assert docs[0].text == "hello"
    assert docs[1].text == "world"
    mock_client.keys.assert_called_once_with("key:*")


@patch.dict("sys.modules", {"redis": MagicMock()})
def test_load_string_metadata_correctness() -> None:
    import sys

    mock_redis_class = sys.modules["redis"].Redis
    mock_client = MagicMock()
    mock_redis_class.from_url.return_value = mock_client

    mock_client.keys.return_value = ["doc:42"]
    mock_client.get.return_value = "some content"

    loader = RedisLoader(url="redis://localhost:6379")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "redis"
    assert docs[0].metadata["key"] == "doc:42"


# ---------------------------------------------------------------------------
# Hash value type
# ---------------------------------------------------------------------------


@patch.dict("sys.modules", {"redis": MagicMock()})
def test_load_hash_values() -> None:
    import sys

    mock_redis_class = sys.modules["redis"].Redis
    mock_client = MagicMock()
    mock_redis_class.from_url.return_value = mock_client

    mock_client.keys.return_value = ["user:1"]
    mock_client.hgetall.return_value = {"name": "Alice", "role": "admin"}

    loader = RedisLoader(url="redis://localhost:6379", value_type="hash")
    docs = loader.load()

    assert len(docs) == 1
    assert "name: Alice" in docs[0].text
    assert "role: admin" in docs[0].text


@patch.dict("sys.modules", {"redis": MagicMock()})
def test_load_hash_empty_hash_is_skipped() -> None:
    import sys

    mock_redis_class = sys.modules["redis"].Redis
    mock_client = MagicMock()
    mock_redis_class.from_url.return_value = mock_client

    mock_client.keys.return_value = ["empty:1"]
    mock_client.hgetall.return_value = {}

    loader = RedisLoader(url="redis://localhost:6379", value_type="hash")
    docs = loader.load()

    assert docs == []


# ---------------------------------------------------------------------------
# JSON value type
# ---------------------------------------------------------------------------


@patch.dict("sys.modules", {"redis": MagicMock()})
def test_load_json_values() -> None:
    import sys

    mock_redis_class = sys.modules["redis"].Redis
    mock_client = MagicMock()
    mock_redis_class.from_url.return_value = mock_client

    payload = {"title": "Redis Rocks", "count": 42}
    mock_client.keys.return_value = ["item:1"]
    mock_client.get.return_value = json.dumps(payload)

    loader = RedisLoader(url="redis://localhost:6379", value_type="json")
    docs = loader.load()

    assert len(docs) == 1
    parsed = json.loads(docs[0].text)
    assert parsed["title"] == "Redis Rocks"
    assert parsed["count"] == 42


@patch.dict("sys.modules", {"redis": MagicMock()})
def test_load_json_invalid_falls_back_to_raw_string() -> None:
    import sys

    mock_redis_class = sys.modules["redis"].Redis
    mock_client = MagicMock()
    mock_redis_class.from_url.return_value = mock_client

    mock_client.keys.return_value = ["bad:1"]
    mock_client.get.return_value = "not-valid-json{{{"

    loader = RedisLoader(url="redis://localhost:6379", value_type="json")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "not-valid-json{{{"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@patch.dict("sys.modules", {"redis": MagicMock()})
def test_load_empty_keys() -> None:
    import sys

    mock_redis_class = sys.modules["redis"].Redis
    mock_client = MagicMock()
    mock_redis_class.from_url.return_value = mock_client

    mock_client.keys.return_value = []

    loader = RedisLoader(url="redis://localhost:6379")
    docs = loader.load()

    assert docs == []


@patch.dict("sys.modules", {"redis": MagicMock()})
def test_load_skips_none_values() -> None:
    import sys

    mock_redis_class = sys.modules["redis"].Redis
    mock_client = MagicMock()
    mock_redis_class.from_url.return_value = mock_client

    mock_client.keys.return_value = ["missing:1", "present:1"]
    mock_client.get.side_effect = [None, "valid text"]

    loader = RedisLoader(url="redis://localhost:6379")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "valid text"
    assert docs[0].metadata["key"] == "present:1"


@patch.dict("sys.modules", {"redis": MagicMock()})
def test_load_respects_limit() -> None:
    import sys

    mock_redis_class = sys.modules["redis"].Redis
    mock_client = MagicMock()
    mock_redis_class.from_url.return_value = mock_client

    mock_client.keys.return_value = ["k:1", "k:2", "k:3", "k:4", "k:5"]
    mock_client.get.return_value = "data"

    loader = RedisLoader(url="redis://localhost:6379", limit=2)
    docs = loader.load()

    assert len(docs) == 2


@patch.dict("sys.modules", {"redis": MagicMock()})
def test_load_uses_decode_responses() -> None:
    """Verify from_url is called with decode_responses=True."""
    import sys

    mock_redis_class = sys.modules["redis"].Redis
    mock_client = MagicMock()
    mock_redis_class.from_url.return_value = mock_client
    mock_client.keys.return_value = []

    loader = RedisLoader(url="redis://localhost:6379")
    loader.load()

    mock_redis_class.from_url.assert_called_once_with(
        "redis://localhost:6379", decode_responses=True
    )


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@patch.dict("sys.modules", {"redis": MagicMock()})
def test_aload() -> None:
    import sys

    mock_redis_class = sys.modules["redis"].Redis
    mock_client = MagicMock()
    mock_redis_class.from_url.return_value = mock_client

    mock_client.keys.return_value = ["async:1"]
    mock_client.get.return_value = "async value"

    loader = RedisLoader(url="redis://localhost:6379")
    docs = asyncio.run(loader.aload())

    assert len(docs) == 1
    assert docs[0].text == "async value"
