from __future__ import annotations

from synapsekit.llm._filesystem_cache import FilesystemLLMCache


class TestFilesystemLLMCache:
    def test_put_and_get(self, tmp_path):
        cache = FilesystemLLMCache(cache_dir=str(tmp_path))
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing(self, tmp_path):
        cache = FilesystemLLMCache(cache_dir=str(tmp_path))
        assert cache.get("missing") is None

    def test_hits_misses(self, tmp_path):
        cache = FilesystemLLMCache(cache_dir=str(tmp_path))
        cache.put("k", "v")
        cache.get("k")
        cache.get("missing")
        assert cache.hits == 1
        assert cache.misses == 1

    def test_len(self, tmp_path):
        cache = FilesystemLLMCache(cache_dir=str(tmp_path))
        assert len(cache) == 0
        cache.put("a", 1)
        cache.put("b", 2)
        assert len(cache) == 2

    def test_clear(self, tmp_path):
        cache = FilesystemLLMCache(cache_dir=str(tmp_path))
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_overwrite(self, tmp_path):
        cache = FilesystemLLMCache(cache_dir=str(tmp_path))
        cache.put("k", "old")
        cache.put("k", "new")
        assert cache.get("k") == "new"
        assert len(cache) == 1

    def test_make_key(self):
        key = FilesystemLLMCache.make_key("model", "prompt", 0.5, 100)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest
