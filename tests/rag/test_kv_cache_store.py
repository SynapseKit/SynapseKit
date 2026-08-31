from __future__ import annotations

import json
import os
import shutil
import tempfile
import pytest

from synapsekit.rag.kv_cache_store import CacheKey, KVCacheStore


@pytest.fixture
def cache_dir() -> str:
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_kv_cache_store_roundtrip(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fingerprint123", "model-a", 2048)
    blob = b"dummy kv cache bytes"
    meta = {"custom_field": "val"}

    store.save(key, blob, meta)
    loaded = store.load(key)

    assert loaded is not None
    loaded_blob, loaded_meta = loaded
    assert loaded_blob == blob
    assert loaded_meta["custom_field"] == "val"
    assert loaded_meta["corpus_fingerprint"] == "fingerprint123"
    assert loaded_meta["model_id"] == "model-a"
    assert loaded_meta["n_ctx"] == 2048
    assert "created_at" in loaded_meta


def test_kv_cache_store_missing_returns_none(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("missing", "model-a", 2048)
    assert store.load(key) is None


def test_kv_cache_store_metadata_mismatch_returns_none(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fingerprint", "model-a", 2048)
    blob = b"dummy bytes"
    meta = {}

    store.save(key, blob, meta)

    # 1. Fingerprint mismatch
    assert store.load(CacheKey("other_fingerprint", "model-a", 2048)) is None
    # 2. Model ID mismatch
    assert store.load(CacheKey("fingerprint", "other-model", 2048)) is None
    # 3. Context size mismatch
    assert store.load(CacheKey("fingerprint", "model-a", 4096)) is None


def test_kv_cache_store_corrupted_json_returns_none(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fingerprint", "model-a", 2048)
    blob = b"dummy bytes"
    meta = {}

    store.save(key, blob, meta)

    # Find and corrupt the json file
    base_path = store._get_filename_base(key)
    json_path = f"{base_path}.json"

    with open(json_path, "w") as f:
        f.write("{invalid json content}")

    assert store.load(key) is None


def test_kv_cache_store_version_mismatch_returns_none(cache_dir: str) -> None:
    store = KVCacheStore(cache_dir=cache_dir)
    key = CacheKey("fingerprint", "model-a", 2048)
    blob = b"dummy bytes"
    meta = {}

    store.save(key, blob, meta)

    # Manipulate JSON file version
    base_path = store._get_filename_base(key)
    json_path = f"{base_path}.json"

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    data["cache_format_version"] = 999  # Incompatible future version

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    assert store.load(key) is None
