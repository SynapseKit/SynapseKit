"""Contract tests for GeminiEmbeddings against a hand-written async fake.

The fake mirrors the shape of the real ``google-genai`` client: the awaitable
embedding call lives under ``client.aio.models.embed_content`` (the top-level
``client.models.embed_content`` is synchronous). Asserting against a real
``async def`` is what catches an accidental ``await client.models...`` — the
exact bug a mock that makes the sync method awaitable would have hidden.
"""

from __future__ import annotations

import contextlib
import inspect
import sys
from types import ModuleType

import numpy as np
import pytest


class _Embedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _EmbedResponse:
    def __init__(self, embeddings: list[_Embedding]) -> None:
        self.embeddings = embeddings


class _FakeAioModels:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows
        self.calls: list[dict[str, object]] = []

    async def embed_content(self, *, model: str, contents: list[str]) -> _EmbedResponse:
        self.calls.append({"model": model, "contents": contents})
        return _EmbedResponse([_Embedding(list(row)) for row in self._rows])


class _FakeAio:
    def __init__(self, rows: list[list[float]]) -> None:
        self.models = _FakeAioModels(rows)


class _FakeSyncModels:
    """The synchronous surface — awaiting this would raise, as in the real SDK."""

    def embed_content(self, *, model: str, contents: list[str]) -> _EmbedResponse:
        raise AssertionError("embed_content must be awaited via client.aio, not client.models")


class _FakeClient:
    def __init__(self, rows: list[list[float]]) -> None:
        self.models = _FakeSyncModels()
        self.aio = _FakeAio(rows)


class _FakeGenAI(ModuleType):
    """Fake ``google.genai`` module exposing a ``Client`` factory."""

    def __init__(self, rows: list[list[float]]) -> None:
        super().__init__("google.genai")
        self._rows = rows
        self.last_client: _FakeClient | None = None

    def Client(self, *, api_key: str) -> _FakeClient:  # noqa: N802 - SDK spelling
        assert api_key
        self.last_client = _FakeClient(self._rows)
        return self.last_client


@pytest.fixture
def fake_genai():
    """Install a fake ``google.genai`` module for the duration of a test."""

    saved_google = sys.modules.get("google")
    saved_genai_mod = sys.modules.get("google.genai")
    had_attr = saved_google is not None and hasattr(saved_google, "genai")
    saved_attr = getattr(saved_google, "genai", None) if had_attr else None

    def _install(rows: list[list[float]]) -> _FakeGenAI:
        fake = _FakeGenAI(rows)
        google_pkg = sys.modules.get("google") or ModuleType("google")
        sys.modules["google"] = google_pkg
        sys.modules["google.genai"] = fake
        google_pkg.genai = fake  # type: ignore[attr-defined]
        return fake

    try:
        yield _install
    finally:
        for name, saved in (("google", saved_google), ("google.genai", saved_genai_mod)):
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
        if saved_google is not None:
            if had_attr:
                saved_google.genai = saved_attr  # type: ignore[attr-defined]
            else:
                with contextlib.suppress(AttributeError):
                    del saved_google.genai  # type: ignore[attr-defined]


def test_embed_raw_is_a_coroutine() -> None:
    from synapsekit.embeddings.gemini import GeminiEmbeddings

    assert inspect.iscoroutinefunction(GeminiEmbeddings._embed_raw)
    assert inspect.iscoroutinefunction(GeminiEmbeddings.embed)


@pytest.mark.asyncio
async def test_embed_parses_values(fake_genai) -> None:
    fake = fake_genai([[0.1] * 4, [0.2] * 4])

    from synapsekit.embeddings.gemini import GeminiEmbeddings

    emb = GeminiEmbeddings(api_key="test-key")
    vecs = await emb.embed(["a", "b"])

    assert vecs.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert fake.last_client is not None
    call = fake.last_client.aio.models.calls[0]
    assert call["model"] == "text-embedding-004"
    assert call["contents"] == ["a", "b"]


@pytest.mark.asyncio
async def test_embed_one_returns_first_row(fake_genai) -> None:
    fake_genai([[0.3] * 4])

    from synapsekit.embeddings.gemini import GeminiEmbeddings

    emb = GeminiEmbeddings(api_key="test-key")
    vec = await emb.embed_one("a")

    assert vec.shape == (4,)


def test_missing_key_raises(fake_genai, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_genai([[0.0]])  # SDK importable; the key check must still fire

    from synapsekit.embeddings.gemini import GeminiEmbeddings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    emb = GeminiEmbeddings(api_key=None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        emb._get_client()
