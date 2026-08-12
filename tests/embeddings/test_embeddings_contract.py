"""Generic contract tests for the hosted embedding providers.

Each provider class is exercised through the uniform async interface
(``embed`` / ``embed_one`` / ``embed_batch`` / ``dimensions``) with the SDK
or HTTP layer mocked — no live API keys or network access.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "cls_name",
    [
        "VoyageEmbeddings",
        "JinaEmbeddings",
        "NomicEmbeddings",
        "MixedbreadEmbeddings",
    ],
)
def test_http_providers_are_importable_from_top_level(cls_name):
    import synapsekit

    cls = getattr(synapsekit, cls_name)
    assert cls.dimensions is not None


@pytest.mark.parametrize(
    "cls_name, module_name",
    [
        ("OpenAIEmbeddings", "openai"),
        ("CohereEmbeddings", "cohere"),
        ("GeminiEmbeddings", "gemini"),
        ("MistralEmbeddings", "mistral"),
    ],
)
def test_lazy_import_error_message(cls_name, module_name):
    """SDK-backed providers raise ImportError with a pip hint when SDK missing."""
    import importlib
    from unittest.mock import patch

    with patch.dict("sys.modules", {module_name: None}):
        mod = importlib.import_module(f"synapsekit.embeddings.{module_name}")
        cls = getattr(mod, cls_name)
        obj = cls(api_key="test-key")
        with pytest.raises(ImportError, match="pip install"):
            obj._get_client()


@pytest.mark.parametrize(
    "cls_name, module_name, env_key, sdk_module",
    [
        ("OpenAIEmbeddings", "openai", "OPENAI_API_KEY", "openai"),
        ("CohereEmbeddings", "cohere", "CO_API_KEY", "cohere"),
        ("GeminiEmbeddings", "gemini", "GEMINI_API_KEY", "google.genai"),
        ("MistralEmbeddings", "mistral", "MISTRAL_API_KEY", "mistralai"),
        ("VoyageEmbeddings", "voyage", "VOYAGE_API_KEY", None),
        ("JinaEmbeddings", "jina", "JINA_API_KEY", None),
        ("NomicEmbeddings", "nomic", "NOMIC_API_KEY", None),
        ("MixedbreadEmbeddings", "mixedbread", "MXBAI_API_KEY", None),
        ("HuggingFaceEmbeddings", "huggingface", "HF_TOKEN", None),
    ],
)
def test_missing_key_raises_value_error(cls_name, module_name, env_key, sdk_module):
    """Missing API key raises a clear ValueError naming the env var."""
    import importlib
    import os
    from unittest.mock import MagicMock, patch

    with patch.dict(os.environ, {}, clear=True):
        if sdk_module:
            fake_sdks = {sdk_module: MagicMock()}
            if sdk_module == "google.genai":
                fake_sdks["google"] = MagicMock()
            with patch.dict("sys.modules", fake_sdks):
                mod = importlib.import_module(f"synapsekit.embeddings.{module_name}")
                cls = getattr(mod, cls_name)
                obj = cls(api_key=None)
                with pytest.raises(ValueError, match=env_key):
                    obj._get_client()
        else:
            mod = importlib.import_module(f"synapsekit.embeddings.{module_name}")
            cls = getattr(mod, cls_name)
            obj = cls(api_key=None)
            with pytest.raises(ValueError, match=env_key):
                obj._get_client()


def test_openai_missing_key_raises():
    import os
    from unittest.mock import patch

    from synapsekit.embeddings.openai import OpenAIEmbeddings

    emb = OpenAIEmbeddings(api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        with patch.dict(os.environ, {}, clear=True):
            emb._get_client()
