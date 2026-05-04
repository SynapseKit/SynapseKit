from __future__ import annotations

from unittest.mock import patch

import pytest

from synapsekit.retrieval.token_counting import TokenCounter


def test_custom_count_fn_used():
    counter = TokenCounter(count_fn=lambda text: 42)
    assert counter.backend_used == "custom"
    assert counter.count("anything") == 42


def test_tiktoken_backend_counts_tokens():
    pytest.importorskip("tiktoken")
    counter = TokenCounter(model="gpt-4o-mini", backend="tiktoken")
    value = counter.count("hello world")
    assert isinstance(value, int)
    assert value > 0


def test_invalid_backend_raises_value_error():
    with pytest.raises(ValueError):
        TokenCounter(backend="nope")


def test_transformers_backend_missing_raises_import_error():
    with patch.dict("sys.modules", {"transformers": None}):
        with pytest.raises(ImportError, match="transformers backend requested"):
            TokenCounter(model="bert-base-uncased", backend="transformers")
