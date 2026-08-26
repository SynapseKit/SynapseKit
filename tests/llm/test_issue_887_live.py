"""Opt-in live smoke tests for the issue #887 provider expansion.

These tests never contact a provider during the normal suite. Set
SYNAPSEKIT_RUN_LIVE_LLM_TESTS=1 and the provider-specific credentials to opt in.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

import pytest

from synapsekit.llm.base import LLMConfig


@dataclass(frozen=True)
class LiveProvider:
    provider: str
    module: str
    class_name: str
    model: str
    key_env: str
    extra_env: tuple[str, ...] = ()


PROVIDERS = (
    LiveProvider(
        "nvidia-nim",
        "nvidia_nim",
        "NvidiaNIMLLM",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "NVIDIA_API_KEY",
    ),
    LiveProvider(
        "watsonx",
        "watsonx",
        "WatsonxLLM",
        "ibm/granite-3-8b-instruct",
        "WATSONX_API_KEY",
        ("WATSONX_PROJECT_ID",),
    ),
    LiveProvider(
        "snowflake-cortex",
        "snowflake_cortex",
        "SnowflakeCortexLLM",
        "llama3.1-8b",
        "SNOWFLAKE_PAT",
        ("SNOWFLAKE_ACCOUNT_IDENTIFIER",),
    ),
    LiveProvider(
        "deepinfra",
        "deepinfra",
        "DeepInfraLLM",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "DEEPINFRA_API_KEY",
    ),
    LiveProvider(
        "nebius", "nebius", "NebiusLLM", "meta-llama/Meta-Llama-3.1-8B-Instruct", "NEBIUS_API_KEY"
    ),
    LiveProvider("baseten", "baseten", "BasetenLLM", "deepseek-ai/DeepSeek-R1", "BASETEN_API_KEY"),
    LiveProvider("upstage", "upstage", "UpstageLLM", "solar-pro", "UPSTAGE_API_KEY"),
    LiveProvider("reka", "reka", "RekaLLM", "reka-flash-3", "REKA_API_KEY"),
    LiveProvider(
        "hyperbolic",
        "hyperbolic",
        "HyperbolicLLM",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "HYPERBOLIC_API_KEY",
    ),
    LiveProvider(
        "friendli", "friendli", "FriendliLLM", "meta-llama-3.1-8b-instruct", "FRIENDLI_TOKEN"
    ),
    LiveProvider(
        "clarifai",
        "clarifai",
        "ClarifaiLLM",
        "https://clarifai.com/meta/Llama-3/models/llama3-1-8b-instruct",
        "CLARIFAI_PAT",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda item: item.provider)
async def test_live_provider_smoke_is_explicitly_opt_in(provider: LiveProvider) -> None:
    if os.getenv("SYNAPSEKIT_RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip("set SYNAPSEKIT_RUN_LIVE_LLM_TESTS=1 to run live provider checks")

    api_key = os.getenv(provider.key_env)
    if not api_key:
        pytest.skip(f"missing {provider.key_env}")
    missing = [name for name in provider.extra_env if not os.getenv(name)]
    if missing:
        pytest.skip(f"missing {', '.join(missing)}")

    module = importlib.import_module(f"synapsekit.llm.{provider.module}")
    provider_class = getattr(module, provider.class_name)
    config = LLMConfig(
        model=provider.model,
        api_key=api_key,
        provider=provider.provider,
        max_tokens=16,
        max_retries=0,
    )
    kwargs = (
        {
            "project_id": os.environ["WATSONX_PROJECT_ID"],
        }
        if provider.provider == "watsonx"
        else {}
    )
    if provider.provider == "snowflake-cortex":
        kwargs = {"account_identifier": os.environ["SNOWFLAKE_ACCOUNT_IDENTIFIER"]}

    llm = provider_class(config, **kwargs)
    result = await llm.generate("Reply with exactly: OK")
    assert result.strip()
