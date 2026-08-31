"""Example showing how to configure and run the CAGRouter with LlamaCppLLM."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from synapsekit import LLMConfig
from synapsekit.llm.llamacpp import LlamaCppLLM
from synapsekit.rag.cag_router import CAGRouter
from synapsekit.retrieval.retriever import Retriever


class MockVectorStore:
    async def add(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        pass

    async def search(self, query: str, top_k: int = 5, metadata_filter: dict | None = None) -> list[dict]:
        return [{"text": "retrieved via RAG due to fallback", "score": 0.9, "metadata": {}}]


async def main() -> None:
    from synapsekit.retrieval.token_counting import TokenCounter

    # 1. Initialize our retriever and mock local model
    store = MockVectorStore()
    retriever = Retriever(vectorstore=store)  # type: ignore

    llm = LlamaCppLLM(
        config=LLMConfig(model="llama-3.1-8b", api_key="", provider="llamacpp"),
        model_path="/models/test.gguf",
    )
    # Set up mock internal model to simulate save/load/generate functions
    llm._model = MagicMock()
    llm._model.tokenize.return_value = [1, 2, 3]
    llm._model.save_state.return_value = b"serialized state"
    llm._model.create_completion.return_value = [{"choices": [{"text": "CAG Answer"}]}]

    # 2. Instantiate CAGRouter wrapping the retriever and LLM
    router = CAGRouter(
        retriever=retriever,
        llm=llm,
        max_cag_tokens=100,
        stable=True,
        on_cache_miss="rebuild",
        token_counter=TokenCounter(count_fn=len),
    )

    # 3. Ingest documents into the corpus
    print("Ingesting corpus...")
    await router.add(["SynapseKit features high-performance CAG vs RAG routing."])

    # 4. Perform retrieval
    print("Querying via Router...")
    results = await router.retrieve("What features does SynapseKit have?")
    print("Results:", results)
    print("Routing decision:", router.last_route)


if __name__ == "__main__":
    asyncio.run(main())
