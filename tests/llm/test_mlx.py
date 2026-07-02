from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.mlx import MLXLLM


def make_llm() -> MLXLLM:
    return MLXLLM(LLMConfig(model="mlx-community/test", api_key="", provider="mlx"))


def test_missing_mlx_lm_raises() -> None:
    llm = make_llm()
    with patch.dict("sys.modules", {"mlx_lm": None}):
        with pytest.raises(ImportError, match="mlx-lm"):
            llm._load_backend()


@pytest.mark.asyncio
async def test_generate_uses_mlx_backend() -> None:
    mock_mlx = MagicMock()
    mock_mlx.load.return_value = ("model", "tokenizer")
    mock_mlx.generate.return_value = "hello"
    mock_mlx.stream_generate.return_value = iter([])

    with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
        result = await make_llm().generate("hi")

    assert result == "hello"
    assert mock_mlx.generate.call_args[1]["prompt"] == "hi"


@pytest.mark.asyncio
async def test_stream_yields_tokens() -> None:
    token = MagicMock()
    token.text = "world"
    mock_mlx = MagicMock()
    mock_mlx.load.return_value = ("model", "tokenizer")
    mock_mlx.generate.return_value = "unused"
    mock_mlx.stream_generate.return_value = iter(["hello ", token])

    with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
        tokens = [token async for token in make_llm().stream("hi")]

    assert tokens == ["hello ", "world"]
