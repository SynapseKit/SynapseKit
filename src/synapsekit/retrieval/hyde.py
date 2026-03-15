from __future__ import annotations

from ..llm.base import BaseLLM
from .retriever import Retriever

_DEFAULT_PROMPT = (
    "Please write a hypothetical passage that would answer the following question. "
    "Write only the passage, nothing else.\n\nQuestion: {query}"
)


class HyDERetriever:
    """Hypothetical Document Embeddings (HyDE) retriever.

    Generates a hypothetical answer using an LLM and uses it as the
    search query, often improving retrieval for complex questions.

    Usage::

        hyde = HyDERetriever(retriever=retriever, llm=llm)
        results = await hyde.retrieve("What is quantum entanglement?")
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        prompt_template: str | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompt_template = prompt_template or _DEFAULT_PROMPT

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Generate a hypothetical answer, then retrieve using it as the query."""
        prompt = self._prompt_template.format(query=query)
        hypothetical = await self._llm.generate(prompt)
        return await self._retriever.retrieve(
            hypothetical, top_k=top_k, metadata_filter=metadata_filter
        )
