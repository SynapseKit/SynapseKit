import json
from unittest.mock import AsyncMock, patch

import pytest

try:
    from synapsekit.llm._cache_memcached import AIOMCACHE_AVAILABLE, MemcachedCacheBackend
except ImportError:
    AIOMCACHE_AVAILABLE = False


@pytest.fixture
def mock_aiomcache_client():
    with patch("aiomcache.Client") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        yield mock_client


@pytest.mark.skipif(not AIOMCACHE_AVAILABLE, reason="aiomcache not installed")
def test_memcached_cache_init(mock_aiomcache_client):
    cache = MemcachedCacheBackend(host="10.0.0.1", port=11212, ttl_seconds=3600)
    assert cache.host == "10.0.0.1"
    assert cache.port == 11212
    assert cache.ttl_seconds == 3600


@pytest.mark.skipif(not AIOMCACHE_AVAILABLE, reason="aiomcache not installed")
def test_memcached_cache_get_hit(mock_aiomcache_client):
    cache = MemcachedCacheBackend()
    mock_aiomcache_client.get.return_value = json.dumps({"result": "cached_data"}).encode("utf-8")

    result = cache.get("test_key")
    assert result == {"result": "cached_data"}
    assert cache.hits == 1
    assert cache.misses == 0
    mock_aiomcache_client.get.assert_called_once_with(b"test_key")


@pytest.mark.skipif(not AIOMCACHE_AVAILABLE, reason="aiomcache not installed")
def test_memcached_cache_get_miss(mock_aiomcache_client):
    cache = MemcachedCacheBackend()
    mock_aiomcache_client.get.return_value = None

    result = cache.get("test_key")
    assert result is None
    assert cache.hits == 0
    assert cache.misses == 1


@pytest.mark.skipif(not AIOMCACHE_AVAILABLE, reason="aiomcache not installed")
def test_memcached_cache_put(mock_aiomcache_client):
    cache = MemcachedCacheBackend(ttl_seconds=3600)
    cache.put("test_key", {"result": "new_data"})

    mock_aiomcache_client.set.assert_called_once()
    args, kwargs = mock_aiomcache_client.set.call_args
    assert args[0] == b"test_key"
    assert json.loads(args[1].decode("utf-8")) == {"result": "new_data"}
    assert kwargs["exptime"] == 3600


@pytest.mark.skipif(not AIOMCACHE_AVAILABLE, reason="aiomcache not installed")
def test_memcached_cache_clear(mock_aiomcache_client):
    cache = MemcachedCacheBackend()
    cache.clear()

    mock_aiomcache_client.flush_all.assert_called_once()


@pytest.mark.skipif(not AIOMCACHE_AVAILABLE, reason="aiomcache not installed")
def test_memcached_cache_len(mock_aiomcache_client):
    cache = MemcachedCacheBackend()
    assert len(cache) == 0
