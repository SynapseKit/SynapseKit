"""RAG facade — 3-line happy-path entry point."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from .._compat import run_sync
from ..embeddings.backend import SynapsekitEmbeddings
from ..llm.base import BaseLLM, LLMConfig
from ..loaders.base import Document
from ..memory.conversation import ConversationMemory
from ..observability.tracer import TokenTracer
from ..retrieval.retriever import Retriever
from ..retrieval.vectorstore import InMemoryVectorStore
from .pipeline import RAGConfig, RAGPipeline


def _make_llm(
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
        elif model.startswith("@cf/") or model.startswith("@hf/"):
            provider = "cloudflare"
        elif model.startswith("dbrx") or model.startswith("databricks"):
            provider = "databricks"
        elif model.startswith("ernie"):
            provider = "ernie"
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
        from ..llm.openai import OpenAILLM

        return OpenAILLM(config)
    elif provider == "anthropic":
        from ..llm.anthropic import AnthropicLLM

        return AnthropicLLM(config)
    elif provider == "ollama":
        from ..llm.ollama import OllamaLLM

        return OllamaLLM(config)
    elif provider == "ai21":
        from ..llm.ai21 import AI21LLM

        return AI21LLM(config)
    elif provider == "cohere":
        from ..llm.cohere import CohereLLM

        return CohereLLM(config)
    elif provider == "mistral":
        from ..llm.mistral import MistralLLM

        return MistralLLM(config)
    elif provider == "gemini":
        from ..llm.gemini import GeminiLLM

        return GeminiLLM(config)
    elif provider == "bedrock":
        from ..llm.bedrock import BedrockLLM

        return BedrockLLM(config)
    elif provider == "groq":
        from ..llm.groq import GroqLLM

        return GroqLLM(config)
    elif provider == "deepseek":
        from ..llm.deepseek import DeepSeekLLM

        return DeepSeekLLM(config)
    elif provider == "openrouter":
        from ..llm.openrouter import OpenRouterLLM

        return OpenRouterLLM(config)
    elif provider == "together":
        from ..llm.together import TogetherLLM

        return TogetherLLM(config)
    elif provider == "fireworks":
        from ..llm.fireworks import FireworksLLM

        return FireworksLLM(config)
    elif provider == "moonshot":
        from ..llm.moonshot import MoonshotLLM

        return MoonshotLLM(config)
    elif provider == "minimax":
        from ..llm.minimax import MinimaxLLM

        return MinimaxLLM(config)
    elif provider == "zhipu":
        from ..llm.zhipu import ZhipuLLM

        return ZhipuLLM(config)
    elif provider == "cloudflare":
        import os

        from ..llm.cloudflare import CloudflareLLM

        return CloudflareLLM(config, account_id=os.environ.get("CLOUDFLARE_ACCOUNT_ID"))
    elif provider == "databricks":
        import os

        from ..llm.databricks import DatabricksLLM

        return DatabricksLLM(config, workspace_url=os.environ.get("DATABRICKS_HOST"))
    elif provider == "ernie":
        from ..llm.ernie import ErnieLLM

        return ErnieLLM(config)
    elif provider == "llamacpp":
        from ..llm.llamacpp import LlamaCppLLM

        return LlamaCppLLM(config, model_path=config.model)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            "Use 'openai', 'anthropic', 'ollama', 'ai21', 'cohere', 'mistral', 'gemini', "
            "'bedrock', 'groq', 'deepseek', 'openrouter', 'together', 'fireworks', "
            "'moonshot', 'minimax', 'zhipu', 'cloudflare', 'databricks', 'ernie', or 'llamacpp'."
        )


class RAG:
    """
    3-line RAG facade with sane defaults.

    Example::

        rag = RAG(model="gpt-4o-mini", api_key="sk-...")
        rag.add("Your document text here")
        answer = rag.ask_sync("What is the main topic?")
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        provider: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank: bool = False,
        memory_window: int = 10,
        retrieval_top_k: int = 5,
        system_prompt: str = "Answer using only the provided context. If the context does not contain the answer, say so.",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        trace: bool = True,
    ) -> None:
        llm = _make_llm(model, api_key, provider, system_prompt, temperature, max_tokens)
        embeddings = SynapsekitEmbeddings(model=embedding_model)
        vectorstore = InMemoryVectorStore(embeddings)
        retriever = Retriever(vectorstore, rerank=rerank)
        memory = ConversationMemory(window=memory_window)
        tracer = TokenTracer(model=model, enabled=trace)

        self._pipeline = RAGPipeline(
            RAGConfig(
                llm=llm,
                retriever=retriever,
                memory=memory,
                tracer=tracer,
                retrieval_top_k=retrieval_top_k,
                system_prompt=system_prompt,
            )
        )
        self._embeddings = embeddings
        self._vectorstore = vectorstore

    # ------------------------------------------------------------------ #
    # Document ingestion
    # ------------------------------------------------------------------ #

    def add(self, text: str, metadata: dict | None = None) -> None:
        """Sync: chunk and embed text into the vectorstore."""
        run_sync(self._pipeline.add(text, metadata))

    async def add_async(self, text: str, metadata: dict | None = None) -> None:
        """Async: chunk and embed text into the vectorstore."""
        await self._pipeline.add(text, metadata)

    def add_documents(self, docs: list[Document]) -> None:
        """Sync: chunk and embed a list of Documents into the vectorstore."""
        run_sync(self._pipeline.add_documents(docs))

    async def add_documents_async(self, docs: list[Document]) -> None:
        """Async: chunk and embed a list of Documents into the vectorstore."""
        await self._pipeline.add_documents(docs)

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    async def stream(self, query: str, **kw) -> AsyncGenerator[str]:
        """Async generator that yields tokens as they arrive from the LLM."""
        async for token in self._pipeline.stream(query, **kw):
            yield token

    async def ask(self, query: str, **kw) -> str:
        """Async: retrieve and answer, returns full string."""
        return await self._pipeline.ask(query, **kw)

    def ask_sync(self, query: str, **kw) -> str:
        """Sync: retrieve and answer (use in scripts/notebooks)."""
        return run_sync(self._pipeline.ask(query, **kw))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Persist the vectorstore to a .npz file."""
        self._vectorstore.save(path)

    def load(self, path: str) -> None:
        """Load a previously saved vectorstore from a .npz file."""
        self._vectorstore.load(path)

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #

    @property
    def tracer(self) -> TokenTracer | None:
        return self._pipeline.config.tracer

    @property
    def memory(self) -> ConversationMemory:
        return self._pipeline.config.memory
