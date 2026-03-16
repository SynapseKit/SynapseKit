from .base import BaseLLM, LLMConfig
from .structured import generate_structured

__all__ = [
    "AnthropicLLM",
    "AzureOpenAILLM",
    "BaseLLM",
    "BedrockLLM",
    "CerebrasLLM",
    "CohereLLM",
    "DeepSeekLLM",
    "FireworksLLM",
    "GeminiLLM",
    "GroqLLM",
    "LLMConfig",
    "MistralLLM",
    "OllamaLLM",
    "OpenAILLM",
    "OpenRouterLLM",
    "PerplexityLLM",
    "TogetherLLM",
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
}


def __getattr__(name: str):
    if name in _PROVIDERS:
        import importlib

        mod = importlib.import_module(_PROVIDERS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
