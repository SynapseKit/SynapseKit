from .base import BaseLLM, LLMConfig

__all__ = [
    "AnthropicLLM",
    "BaseLLM",
    "BedrockLLM",
    "CohereLLM",
    "GeminiLLM",
    "LLMConfig",
    "MistralLLM",
    "OllamaLLM",
    "OpenAILLM",
]

_PROVIDERS = {
    "OpenAILLM": ".openai",
    "AnthropicLLM": ".anthropic",
    "OllamaLLM": ".ollama",
    "CohereLLM": ".cohere",
    "MistralLLM": ".mistral",
    "GeminiLLM": ".gemini",
    "BedrockLLM": ".bedrock",
}


def __getattr__(name: str):
    if name in _PROVIDERS:
        import importlib

        mod = importlib.import_module(_PROVIDERS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
