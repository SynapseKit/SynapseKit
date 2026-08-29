"""Public factory and lazy-export contracts for issue #887 providers."""

from __future__ import annotations

import importlib

import pytest

from synapsekit import LLMConfig
from synapsekit.llm._factory import make_llm

PROVIDERS = [
    ("nvidia-nim", "NvidiaNIMLLM", "nvidia_nim"),
    ("watsonx", "WatsonxLLM", "watsonx"),
    ("snowflake-cortex", "SnowflakeCortexLLM", "snowflake_cortex"),
    ("deepinfra", "DeepInfraLLM", "deepinfra"),
    ("nebius", "NebiusLLM", "nebius"),
    ("baseten", "BasetenLLM", "baseten"),
    ("upstage", "UpstageLLM", "upstage"),
    ("reka", "RekaLLM", "reka"),
    ("hyperbolic", "HyperbolicLLM", "hyperbolic"),
    ("friendli", "FriendliLLM", "friendli"),
    ("clarifai", "ClarifaiLLM", "clarifai"),
]


def _config(provider: str) -> LLMConfig:
    return LLMConfig(model="test-model", api_key="test-key", provider=provider)


@pytest.mark.parametrize("provider,class_name,module_name", PROVIDERS)
def test_factory_constructs_each_new_provider(
    provider: str, class_name: str, module_name: str
) -> None:
    module = importlib.import_module(f"synapsekit.llm.{module_name}")
    expected = getattr(module, class_name)
    actual = make_llm(
        model="test-model",
        api_key="test-key",
        provider=provider,
        system_prompt="",
        temperature=0.0,
        max_tokens=16,
    )
    assert type(actual) is expected


@pytest.mark.parametrize("provider,class_name,module_name", PROVIDERS)
def test_llm_lazy_exports_resolve(provider: str, class_name: str, module_name: str) -> None:
    del provider
    llm_package = importlib.import_module("synapsekit.llm")
    top_level = importlib.import_module("synapsekit")
    expected = getattr(importlib.import_module(f"synapsekit.llm.{module_name}"), class_name)

    assert getattr(llm_package, class_name) is expected
    assert getattr(top_level, class_name) is expected
    assert class_name in llm_package.__all__
    assert class_name in top_level.__all__


def test_slash_models_remain_openrouter_without_explicit_provider() -> None:
    llm = make_llm(
        model="meta-llama/test-model",
        api_key="test-key",
        provider=None,
        system_prompt="",
        temperature=0.0,
        max_tokens=16,
    )
    assert type(llm).__name__ == "OpenRouterLLM"


def test_issue_887_provider_names_are_in_unknown_provider_message() -> None:
    with pytest.raises(ValueError) as exc_info:
        make_llm("test-model", "test-key", "not-a-provider", "", 0.0, 16)

    message = str(exc_info.value)
    assert all(provider in message for provider, _, _ in PROVIDERS)


@pytest.mark.parametrize("provider,_class_name,_module_name", PROVIDERS)
def test_constructor_does_not_require_optional_provider_dependency(
    provider: str, _class_name: str, _module_name: str
) -> None:
    # Construction is lazy; SDK imports happen only when a request is made.
    llm = make_llm("test-model", "test-key", provider, "", 0.0, 16)
    assert llm.config.provider == provider
    assert llm.config.model == "test-model"
