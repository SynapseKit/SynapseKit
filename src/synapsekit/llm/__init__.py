from ._cache import AsyncLRUCache
from .base import BaseLLM, LLMConfig
from .cost_router import QUALITY_TABLE, CostRouter, CostRouterConfig, RouterModelSpec
from .fallback_chain import FallbackChain, FallbackChainConfig
from .structured import generate_structured

__all__ = [
    "AI21LLM",
    "AlephAlphaLLM",
    "MemcachedCacheBackend",
    "AnthropicLLM",
    "AsyncLRUCache",
    "AzureOpenAILLM",
    "BaseLLM",
    "BedrockLLM",
    "CerebrasLLM",
    "CloudflareLLM",
    "CohereLLM",
    "CostRouter",
    "CostRouterConfig",
    "DatabricksLLM",
    "DeepSeekLLM",
    "DynamoDBCacheBackend",
    "ErnieLLM",
    "FallbackChain",
    "FallbackChainConfig",
    "FireworksLLM",
    "GeminiLLM",
    "GroqLLM",
    "HuggingFaceLLM",
    "LLMConfig",
    "LlamaCppLLM",
    "MistralLLM",
    "MinimaxLLM",
    "MoonshotLLM",
    "OllamaLLM",
    "OpenAILLM",
    "OpenRouterLLM",
    "PerplexityLLM",
    "QUALITY_TABLE",
    "RouterModelSpec",
    "SambaNovaLLM",
    "TogetherLLM",
    "VertexAILLM",
    "ZhipuLLM",
    "generate_structured",
]

_PROVIDERS = {
    "AI21LLM": ".ai21",
    "AlephAlphaLLM": ".aleph_alpha",
    "MemcachedCacheBackend": "._cache_memcached",
    "OpenAILLM": ".openai",
    "AzureOpenAILLM": ".azure_openai",
    "AnthropicLLM": ".anthropic",
    "OllamaLLM": ".ollama",
    "CohereLLM": ".cohere",
    "MistralLLM": ".mistral",
    "GeminiLLM": ".gemini",
    "BedrockLLM": ".bedrock",
    "GroqLLM": ".groq",
    "HuggingFaceLLM": ".huggingface",
    "DeepSeekLLM": ".deepseek",
    "OpenRouterLLM": ".openrouter",
    "TogetherLLM": ".together",
    "FireworksLLM": ".fireworks",
    "PerplexityLLM": ".perplexity",
    "SambaNovaLLM": ".sambanova",
    "CerebrasLLM": ".cerebras",
    "LlamaCppLLM": ".llamacpp",
    "VertexAILLM": ".vertex_ai",
    "MoonshotLLM": ".moonshot",
    "MinimaxLLM": ".minimax",
    "ZhipuLLM": ".zhipu",
    "CloudflareLLM": ".cloudflare",
    "DatabricksLLM": ".databricks",
    "ErnieLLM": ".ernie",
    "DynamoDBCacheBackend": "._cache_dynamodb",
}


def __getattr__(name: str):
    if name in _PROVIDERS:
        import importlib

        mod = importlib.import_module(_PROVIDERS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
