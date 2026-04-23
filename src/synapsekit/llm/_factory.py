"""Shared LLM factory for high-level facades."""

from __future__ import annotations

from .base import BaseLLM, LLMConfig


def make_llm(
    model: str,
    api_key: str,
    provider: str | None,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> BaseLLM:
    """Auto-detect provider from model name, or use explicit provider."""
    if provider is None:
        if model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith("gemini"):
            provider = "gemini"
        elif model.startswith("command"):
            provider = "cohere"
        elif model.startswith("mistral") or model.startswith("open-mistral"):
            provider = "mistral"
        elif model.startswith("deepseek"):
            provider = "deepseek"
        elif model.startswith("moonshot"):
            provider = "moonshot"
        elif model.startswith("abab") or model.startswith("minimax"):
            provider = "minimax"
        elif model.startswith("glm"):
            provider = "zhipu"
        elif model.startswith("jamba"):
            provider = "ai21"
        elif model.startswith("luminous") or model.startswith("pharia"):
            provider = "aleph-alpha"
        elif model.startswith("@cf/") or model.startswith("@hf/"):
            provider = "cloudflare"
        elif model.startswith("dbrx") or model.startswith("databricks"):
            provider = "databricks"
        elif model.startswith("ernie"):
            provider = "ernie"
        elif model.startswith("sambanova"):
            provider = "sambanova"
        elif model.startswith("llama") or model.startswith("mixtral") or model.startswith("gemma"):
            provider = "groq"
        elif "/" in model:
            # Slash in model name suggests OpenRouter (e.g. "openai/gpt-4o")
            provider = "openrouter"
        else:
            provider = "openai"

    config = LLMConfig(
        model=model,
        api_key=api_key,
        provider=provider,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if provider == "openai":
        from .openai import OpenAILLM

        return OpenAILLM(config)
    elif provider == "anthropic":
        from .anthropic import AnthropicLLM

        return AnthropicLLM(config)
    elif provider == "ollama":
        from .ollama import OllamaLLM

        return OllamaLLM(config)
    elif provider == "ai21":
        from .ai21 import AI21LLM

        return AI21LLM(config)
    elif provider == "cohere":
        from .cohere import CohereLLM

        return CohereLLM(config)
    elif provider == "mistral":
        from .mistral import MistralLLM

        return MistralLLM(config)
    elif provider == "gemini":
        from .gemini import GeminiLLM

        return GeminiLLM(config)
    elif provider == "bedrock":
        from .bedrock import BedrockLLM

        return BedrockLLM(config)
    elif provider == "groq":
        from .groq import GroqLLM

        return GroqLLM(config)
    elif provider == "deepseek":
        from .deepseek import DeepSeekLLM

        return DeepSeekLLM(config)
    elif provider == "openrouter":
        from .openrouter import OpenRouterLLM

        return OpenRouterLLM(config)
    elif provider == "together":
        from .together import TogetherLLM

        return TogetherLLM(config)
    elif provider == "fireworks":
        from .fireworks import FireworksLLM

        return FireworksLLM(config)
    elif provider == "moonshot":
        from .moonshot import MoonshotLLM

        return MoonshotLLM(config)
    elif provider == "minimax":
        from .minimax import MinimaxLLM

        return MinimaxLLM(config)
    elif provider == "zhipu":
        from .zhipu import ZhipuLLM

        return ZhipuLLM(config)
    elif provider == "cloudflare":
        import os

        from .cloudflare import CloudflareLLM

        return CloudflareLLM(config, account_id=os.environ.get("CLOUDFLARE_ACCOUNT_ID"))
    elif provider == "databricks":
        import os

        from .databricks import DatabricksLLM

        return DatabricksLLM(config, workspace_url=os.environ.get("DATABRICKS_HOST"))
    elif provider == "ernie":
        from .ernie import ErnieLLM

        return ErnieLLM(config)
    elif provider == "sambanova":
        from .sambanova import SambaNovaLLM

        return SambaNovaLLM(config)
    elif provider == "aleph-alpha":
        from .aleph_alpha import AlephAlphaLLM

        return AlephAlphaLLM(config)
    elif provider == "llamacpp":
        from .llamacpp import LlamaCppLLM

        return LlamaCppLLM(config, model_path=config.model)
    elif provider == "vllm":
        from .vllm import VLLMLLM

        return VLLMLLM(config)
    elif provider == "gpt4all":
        from .gpt4all import GPT4AllLLM

        return GPT4AllLLM(config)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            "Use 'openai', 'anthropic', 'ollama', 'ai21', 'cohere', 'mistral', 'gemini', "
            "'bedrock', 'groq', 'deepseek', 'openrouter', 'together', 'fireworks', "
            "'moonshot', 'minimax', 'zhipu', 'cloudflare', 'databricks', 'ernie', 'sambanova', "
            "'aleph-alpha', 'llamacpp', 'vllm', or 'gpt4all'."
        )
