from __future__ import annotations

import asyncio
import sys

import pytest

from synapsekit.llm._llamacpp_cag_backend import LlamaCppCAGBackend
from synapsekit.llm.base import LLMConfig
from synapsekit.llm.llamacpp import LlamaCppLLM


def make_config() -> LLMConfig:
    return LLMConfig(model="llama-3.1-8b", api_key="", provider="llamacpp")


class FakeLlamaModel:
    """Hand-written stand-in for a llama-cpp ``Llama`` model.

    Records the calls the CAG backend makes so tests can assert on them
    without a mocking library.
    """

    def __init__(
        self,
        *,
        tokens: list[int] | None = None,
        state: bytes = b"",
        completion_chunks: tuple[dict, ...] = (),
    ) -> None:
        self._tokens = tokens if tokens is not None else []
        self._state = state
        self._completion_chunks = list(completion_chunks)
        self.reset_calls = 0
        self.tokenize_calls: list[tuple[bytes, bool]] = []
        self.eval_calls: list[list[int]] = []
        self.save_state_calls = 0
        self.load_state_calls: list[object] = []
        self.create_completion_calls: list[dict] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def tokenize(self, data: bytes, add_bos: bool = False) -> list[int]:
        self.tokenize_calls.append((data, add_bos))
        return self._tokens

    def eval(self, tokens: list[int]) -> None:
        self.eval_calls.append(tokens)

    def save_state(self) -> bytes:
        self.save_state_calls += 1
        return self._state

    def load_state(self, state: object) -> None:
        self.load_state_calls.append(state)

    def create_completion(self, **kwargs: object) -> list[dict]:
        self.create_completion_calls.append(kwargs)
        return list(self._completion_chunks)


@pytest.fixture
def backend() -> LlamaCppCAGBackend:
    return LlamaCppCAGBackend()


@pytest.fixture
def llm() -> LlamaCppLLM:
    return LlamaCppLLM(make_config(), model_path="/models/test.gguf")


def test_supports(backend: LlamaCppCAGBackend, llm: LlamaCppLLM) -> None:
    assert backend.supports(llm) is True
    # Any non-LlamaCpp object is unsupported.
    assert backend.supports(object()) is False


def test_import_error_without_llama_cpp(monkeypatch: pytest.MonkeyPatch, llm: LlamaCppLLM) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(ImportError, match="llama-cpp-python"):
        llm._get_model()


@pytest.mark.asyncio
async def test_build_cache(backend: LlamaCppCAGBackend, llm: LlamaCppLLM) -> None:
    model = FakeLlamaModel(tokens=[1, 2, 3], state=b"serialized state bytes")
    llm._model = model

    res = await backend.build_cache(llm, "corpus text here")

    assert res["corpus_text"] == "corpus text here"
    assert res["state"] == b"serialized state bytes"
    assert model.reset_calls == 1
    assert model.tokenize_calls == [(b"corpus text here", True)]
    assert model.eval_calls == [[1, 2, 3]]
    assert model.save_state_calls == 1


@pytest.mark.asyncio
async def test_generate_with_cache(backend: LlamaCppCAGBackend, llm: LlamaCppLLM) -> None:
    model = FakeLlamaModel(
        completion_chunks=(
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " world"}]},
        )
    )
    llm._model = model

    cache_handle = {"state": b"serialized state bytes", "corpus_text": "corpus text here"}

    tokens = []
    async for token in backend.generate_with_cache(llm, cache_handle, " query text"):
        tokens.append(token)

    assert tokens == ["Hello", " world"]
    # llama_cpp is not installed in the test environment, so the raw state
    # bytes are passed straight through to load_state.
    assert model.load_state_calls == [b"serialized state bytes"]
    assert model.create_completion_calls == [
        {
            "prompt": "corpus text here query text",
            "stream": True,
            "temperature": llm.config.temperature,
            "max_tokens": llm.config.max_tokens,
            "top_p": llm._top_p,
        }
    ]


@pytest.mark.asyncio
async def test_generate_with_cache_stops_producer_on_early_break(
    backend: LlamaCppCAGBackend, llm: LlamaCppLLM
) -> None:
    produced = 0

    class StreamingModel(FakeLlamaModel):
        def create_completion(self, **kwargs: object) -> object:
            self.create_completion_calls.append(kwargs)

            def gen():
                nonlocal produced
                index = 0
                while True:  # unbounded stream
                    produced += 1
                    yield {"choices": [{"text": f"tok{index}"}]}
                    index += 1

            return gen()

    llm._model = StreamingModel()
    cache_handle = {"state": b"state", "corpus_text": "corpus"}

    gen = backend.generate_with_cache(llm, cache_handle, " query")
    first = await asyncio.wait_for(gen.__anext__(), timeout=5.0)
    assert first == "tok0"

    # Closing the consumer must stop the background producer promptly.
    await asyncio.wait_for(gen.aclose(), timeout=5.0)
    produced_at_close = produced
    await asyncio.sleep(0.05)

    # The producer stopped (count frozen) and never ran unbounded ahead of the
    # consumer (bounded by the queue size, not the infinite stream).
    assert produced == produced_at_close
    assert produced <= 512
