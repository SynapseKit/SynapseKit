"""ONNX Runtime embeddings backend for edge deployments."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class ONNXEmbeddings:
    """Async embeddings backed by an ONNX Runtime session.

    The class accepts an optional tokenizer callable so deployments can reuse
    an application-specific tokenizer. If omitted, it uses a deterministic
    lightweight byte tokenizer that is sufficient for smoke tests and small
    edge demos, while production users can pass a Hugging Face tokenizer or
    another callable returning ONNX input tensors.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        tokenizer: Callable[[list[str]], dict[str, Any]] | None = None,
        providers: list[str] | None = None,
        normalize: bool = True,
        max_length: int = 256,
    ) -> None:
        self.model_path = str(model_path)
        self._tokenizer = tokenizer
        self._providers = providers
        self._normalize = normalize
        self._max_length = max_length
        self._session: Any = None

    def _get_session(self) -> Any:
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError("onnxruntime required: pip install synapsekit[onnx]") from None

            kwargs: dict[str, Any] = {}
            if self._providers is not None:
                kwargs["providers"] = self._providers
            self._session = ort.InferenceSession(self.model_path, **kwargs)
        return self._session

    def _default_tokenize(self, texts: list[str]) -> dict[str, np.ndarray]:
        input_ids = np.zeros((len(texts), self._max_length), dtype=np.int64)
        attention_mask = np.zeros((len(texts), self._max_length), dtype=np.int64)

        for row, text in enumerate(texts):
            encoded = text.encode("utf-8")[: self._max_length]
            if not encoded:
                continue
            values = np.frombuffer(encoded, dtype=np.uint8).astype(np.int64)
            input_ids[row, : len(values)] = values
            attention_mask[row, : len(values)] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _build_inputs(self, texts: list[str]) -> dict[str, Any]:
        if self._tokenizer is not None:
            return self._tokenizer(texts)
        return self._default_tokenize(texts)

    @staticmethod
    def _first_array(outputs: list[Any] | tuple[Any, ...] | Any) -> np.ndarray:
        if isinstance(outputs, np.ndarray):
            return outputs
        for item in outputs:
            if isinstance(item, np.ndarray):
                return item
        raise ValueError("ONNX model did not return a numpy array")

    @staticmethod
    def _pool(arr: np.ndarray, attention_mask: np.ndarray | None) -> np.ndarray:
        if arr.ndim == 2:
            return arr.astype(np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected ONNX output with 2 or 3 dims, got shape {arr.shape}")

        values = arr.astype(np.float32)
        if attention_mask is None:
            return np.asarray(values.mean(axis=1), dtype=np.float32)

        mask = attention_mask.astype(np.float32)[..., None]
        denom = np.maximum(mask.sum(axis=1), 1.0)
        return np.asarray((values * mask).sum(axis=1) / denom, dtype=np.float32)

    def _normalize_vectors(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if not self._normalize:
            return arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts, returning an ``(N, D)`` float32 array."""

        import asyncio

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        session = self._get_session()
        inputs = self._build_inputs(texts)
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(None, session.run, None, inputs)
        arr = self._first_array(outputs)
        pooled = self._pool(arr, inputs.get("attention_mask"))
        return self._normalize_vectors(pooled)

    async def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string, returning a one-dimensional vector."""

        vectors = await self.embed([text])
        return np.asarray(vectors[0], dtype=np.float32)
