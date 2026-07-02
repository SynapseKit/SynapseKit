from __future__ import annotations

import sys

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.mlx import MLXLLM


def make_llm() -> MLXLLM:
    return MLXLLM(LLMConfig(model="mlx-community/test", api_key="", provider="mlx"))


class _FakeTokenItem:
    """Simulates a mlx_lm token stream item with a .text attribute."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeMLXModule:
    """Hand-written fake for the mlx_lm module."""

    def load(self, model_path, **kwargs):
        return object(), object()

    def generate(self, model, tokenizer, *, prompt: str, max_tokens: int, temp: float) -> str:
        return f"hello"

    def stream_generate(
        self, model, tokenizer, *, prompt: str, max_tokens: int, temp: float
    ):
        yield _FakeTokenItem("hello ")
        yield _FakeTokenItem("world")


def _inject_fake_mlx() -> FakeMLXModule:
    """Install FakeMLXModule into sys.modules and return it."""
    fake = FakeMLXModule()
    sys.modules["mlx_lm"] = fake  # type: ignore[assignment]
    return fake


def _remove_fake_mlx(original) -> None:
    if original is None:
        sys.modules.pop("mlx_lm", None)
    else:
        sys.modules["mlx_lm"] = original


def test_missing_mlx_lm_raises() -> None:
    # Remove mlx_lm from sys.modules temporarily to simulate absence.
    original = sys.modules.pop("mlx_lm", None)
    try:
        llm = make_llm()
        with pytest.raises(ImportError) as exc_info:
            llm._load_backend()
        assert "mlx-lm" in str(exc_info.value)
        assert "pip install" in str(exc_info.value)
    finally:
        if original is not None:
            sys.modules["mlx_lm"] = original


@pytest.mark.asyncio
async def test_generate_uses_mlx_backend() -> None:
    original = sys.modules.get("mlx_lm")
    fake = _inject_fake_mlx()
    try:
        result = await make_llm().generate("hi")
    finally:
        _remove_fake_mlx(original)

    assert result == "hello"


@pytest.mark.asyncio
async def test_stream_yields_tokens() -> None:
    original = sys.modules.get("mlx_lm")
    _inject_fake_mlx()
    try:
        tokens = [token async for token in make_llm().stream("hi")]
    finally:
        _remove_fake_mlx(original)

    assert tokens == ["hello ", "world"]
