from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheKey:
    corpus_fingerprint: str
    model_id: str
    n_ctx: int


class KVCacheStore:
    """Persistent KV Cache Store for Cache-Augmented Generation (CAG).

    Writes cache blobs and metadata atomically using a temporary file and os.replace.
    """

    CACHE_FORMAT_VERSION = 1

    def __init__(self, cache_dir: str = ".synapsekit_cag_cache") -> None:
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_filename_base(self, key: CacheKey) -> str:
        # Use a safe hash of the key tuple for the filenames
        import hashlib

        h = hashlib.sha256(
            f"{key.corpus_fingerprint}:{key.model_id}:{key.n_ctx}".encode()
        ).hexdigest()
        return os.path.join(self._cache_dir, h)

    def save(self, key: CacheKey, blob: bytes, metadata: dict[str, Any]) -> None:
        """Atomically save the cache blob and its associated metadata to disk."""
        base_path = self._get_filename_base(key)
        bin_path = f"{base_path}.bin"
        json_path = f"{base_path}.json"

        # Build full metadata dict
        full_meta = {
            **metadata,
            "corpus_fingerprint": key.corpus_fingerprint,
            "model_id": key.model_id,
            "n_ctx": key.n_ctx,
            "cache_format_version": self.CACHE_FORMAT_VERSION,
            "created_at": time.time(),
        }

        # Write binary blob atomically
        bin_dir = os.path.dirname(bin_path)
        with tempfile.NamedTemporaryFile(dir=bin_dir, delete=False, suffix=".tmp") as f:
            f.write(blob)
            temp_bin = f.name

        try:
            os.replace(temp_bin, bin_path)
        except Exception:
            if os.path.exists(temp_bin):
                os.remove(temp_bin)
            raise

        # Write metadata JSON atomically
        json_content = json.dumps(full_meta, indent=2).encode("utf-8")
        with tempfile.NamedTemporaryFile(dir=bin_dir, delete=False, suffix=".tmp") as f:
            f.write(json_content)
            temp_json = f.name

        try:
            os.replace(temp_json, json_path)
        except Exception:
            if os.path.exists(temp_json):
                os.remove(temp_json)
            raise

    def load(self, key: CacheKey) -> tuple[bytes, dict[str, Any]] | None:
        """Load and validate the cache blob and metadata.

        Returns None on any corruption, mismatch, or missing files.
        """
        base_path = self._get_filename_base(key)
        bin_path = f"{base_path}.bin"
        json_path = f"{base_path}.json"

        if not os.path.exists(bin_path) or not os.path.exists(json_path):
            return None

        try:
            # Load and validate metadata first
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("corpus_fingerprint") != key.corpus_fingerprint:
                return None
            if meta.get("model_id") != key.model_id:
                return None
            if meta.get("n_ctx") != key.n_ctx:
                return None
            if meta.get("cache_format_version") != self.CACHE_FORMAT_VERSION:
                return None

            # Read binary blob
            with open(bin_path, "rb") as f:
                blob = f.read()

            return blob, meta
        except Exception:
            # Treat any file error or JSON corruption as a cache invalidation
            return None
