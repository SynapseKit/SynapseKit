from __future__ import annotations

from unittest.mock import patch

import pytest

from synapsekit.memory import AgentMemory


def test_postgres_requires_dsn():
    with pytest.raises(ValueError, match="postgres_dsn"):
        AgentMemory(backend="postgres")


def test_redis_backend_import_error_when_package_missing():
    with patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}):
        with pytest.raises(ImportError, match="redis"):
            AgentMemory(backend="redis")


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        AgentMemory(backend="does-not-exist")
