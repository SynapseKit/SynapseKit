from .base import BaseLLM, LLMConfig
from .cost_router import QUALITY_TABLE, CostRouter, CostRouterConfig, RouterModelSpec
from .fallback_chain import FallbackChain, FallbackChainConfig
from .structured import generate_structured

__all__ = [
    "AnthropicLLM",
    "AzureOpenAILLM",
    "BaseLLM",
    "BedrockLLM",
    "CerebrasLLM",
    "CloudflareLLM",
    "CohereLLM",
    "CostRouter",
    "CostRouterConfig",
    "DeepSeekLLM",
    "FallbackChain",
    "FallbackChainConfig",
    "FireworksLLM",
    "GeminiLLM",
    "GroqLLM",
    "LLMConfig",
    "LlamaCppLLM",
    "MistralLLM",
    "MoonshotLLM",
    "OllamaLLM",
    "OpenAILLM",
    "OpenRouterLLM",
    "PerplexityLLM",
    "QUALITY_TABLE",
    "RouterModelSpec",
    "TogetherLLM",
    "VertexAILLM",
    "ZhipuLLM",
    "generate_structured",
]

_PROVIDERS = {
    "OpenAILLM": ".openai",
    "AzureOpenAILLM": ".azure_openai",
    "AnthropicLLM": ".anthropic",
    "OllamaLLM": ".ollama",
    "CohereLLM": ".cohere",
    "MistralLLM": ".mistral",
    "GeminiLLM": ".gemini",
    "BedrockLLM": ".bedrock",
    "GroqLLM": ".groq",
    "DeepSeekLLM": ".deepseek",
    "OpenRouterLLM": ".openrouter",
    "TogetherLLM": ".together",
    "FireworksLLM": ".fireworks",
    "PerplexityLLM": ".perplexity",
    "CerebrasLLM": ".cerebras",
    "LlamaCppLLM": ".llamacpp",
    "VertexAILLM": ".vertex_ai",
    "MoonshotLLM": ".moonshot",
    "ZhipuLLM": ".zhipu",
    "CloudflareLLM": ".cloudflare",
}


def __getattr__(name: str):
    if name in _PROVIDERS:
        import importlib

        mod = importlib.import_module(_PROVIDERS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
