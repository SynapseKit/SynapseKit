import asyncio
from collections.abc import AsyncGenerator

import numpy as np

from synapsekit import ExtractionPolicy, WorldModelRAG
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.retrieval.vectorstore import InMemoryVectorStore


class DemoEmbeddings:
    async def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vector(text) for text in texts], dtype=np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        values = np.zeros(8, dtype=np.float32)
        for index, char in enumerate(text.encode("utf-8")):
            values[index % len(values)] += float(char)
        norm = np.linalg.norm(values)
        return values / norm if norm else values


class DemoLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig(model="demo", api_key="", provider="demo"))

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        yield (
            "Alice worked on Search API, and the graph marks Search API as a causal "
            "input to the v1.5 release."
        )


async def main() -> None:
    wm = WorldModelRAG(
        extraction=ExtractionPolicy(
            entities=["person", "product", "event"],
            relations="open_schema",
            temporal=True,
            causal=True,
        ),
        graph_backend="in_memory",
        vector_store=InMemoryVectorStore(DemoEmbeddings()),
        llm=DemoLLM(),
        retrieval_top_k=4,
    )

    await wm.ingest(
        [
            {
                "text": "Alice worked on Search API in 2026. Search API led to v1.5 Release.",
                "metadata": {"source": "release_notes"},
            },
            {
                "text": "Billing Service depends on Deprecated API.",
                "metadata": {"source": "architecture_review"},
            },
        ]
    )

    result = await wm.query(
        "What did Alice work on that led to v1.5?",
        strategy="graph_first",
        as_of="2026-12-01",
    )

    print(result.answer)
    print()
    print(wm.subgraph_to_mermaid("Alice v1.5"))


if __name__ == "__main__":
    asyncio.run(main())
