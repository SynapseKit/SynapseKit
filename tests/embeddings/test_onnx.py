from __future__ import annotations

import sys

import numpy as np
import pytest

from synapsekit.embeddings.onnx import ONNXEmbeddings


class FakeONNXSession:
    def __init__(self, output_arrays: list) -> None:
        self._outputs = output_arrays
        self.calls: list[tuple] = []

    def run(self, output_names, input_feeds):
        self.calls.append((output_names, input_feeds))
        return self._outputs


def test_missing_onnxruntime_raises() -> None:
    # If onnxruntime is already installed we cannot simulate its absence without
    # patching — skip so the test only runs in environments where it is absent.
    if "onnxruntime" in sys.modules and sys.modules["onnxruntime"] is not None:
        pytest.skip("onnxruntime is installed; this test only runs when absent")

    embeddings = ONNXEmbeddings("model.onnx")
    # Temporarily remove onnxruntime from sys.modules to simulate absence
    original = sys.modules.pop("onnxruntime", None)
    try:
        with pytest.raises(ImportError) as exc_info:
            embeddings._get_session()
        assert "onnxruntime" in str(exc_info.value)
        assert "pip install" in str(exc_info.value)
    finally:
        if original is not None:
            sys.modules["onnxruntime"] = original


@pytest.mark.asyncio
async def test_embed_with_mocked_session_normalizes_vectors() -> None:
    session = FakeONNXSession(
        [
            np.array(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 2.0], [0.0, 2.0]],
                ],
                dtype=np.float32,
            )
        ]
    )
    embeddings = ONNXEmbeddings(
        "model.onnx",
        tokenizer=lambda texts: {
            "input_ids": np.ones((len(texts), 2), dtype=np.int64),
            "attention_mask": np.ones((len(texts), 2), dtype=np.int64),
        },
    )
    embeddings._session = session

    result = await embeddings.embed(["a", "b"])

    assert result.shape == (2, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), np.array([1.0, 1.0]))
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_embed_one_returns_single_vector() -> None:
    session = FakeONNXSession([np.array([[3.0, 4.0]], dtype=np.float32)])
    embeddings = ONNXEmbeddings("model.onnx")
    embeddings._session = session

    result = await embeddings.embed_one("hello")

    assert result.shape == (2,)
    np.testing.assert_allclose(result, np.array([0.6, 0.8], dtype=np.float32))
    assert len(session.calls) == 1
