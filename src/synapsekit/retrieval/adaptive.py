"""Adaptive RAG: routes queries to different strategies based on LLM classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import BaseLLM

_CLASSIFY_PROMPT = """\
Classify the following question by complexity. Respond with exactly one word: \
"simple", "moderate", or "complex".

- simple: factual, single-hop, can be answered from a single document
- moderate: requires combining information from a few documents
- complex: multi-hop reasoning, requires synthesis across many sources

Question: {question}"""


class AdaptiveRAGRetriever:
    """Routes queries to different retrieval strategies based on LLM classification.

    Usage::

        adaptive = AdaptiveRAGRetriever(
            llm=llm,
            simple_retriever=basic_retriever,
            moderate_retriever=fusion_retriever,
            complex_retriever=multi_step_retriever,
        )
        results = await adaptive.retrieve("What is X?")
        results, classification = await adaptive.retrieve_with_classification("What is X?")
    """

    def __init__(
        self,
        llm: BaseLLM,
        simple_retriever,
        moderate_retriever=None,
        complex_retriever=None,
        classify_prompt: str | None = None,
    ) -> None:
        self._llm = llm
        self._simple = simple_retriever
        self._moderate = moderate_retriever or simple_retriever
        self._complex = complex_retriever or self._moderate
        self._classify_prompt = classify_prompt or _CLASSIFY_PROMPT

    async def _classify(self, question: str) -> str:
        prompt = self._classify_prompt.format(question=question)
        response = await self._llm.generate(prompt)
        result = response.strip().lower()
        for level in ("simple", "moderate", "complex"):
            if result.startswith(level):
                return level
        return "moderate"

    def _get_retriever(self, classification: str):
        if classification == "simple":
            return self._simple
        if classification == "complex":
            return self._complex
        return self._moderate

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Classify and route to the appropriate retriever."""
        docs, _ = await self.retrieve_with_classification(query, top_k, metadata_filter)
        return docs

    async def retrieve_with_classification(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> tuple[list[str], str]:
        """Classify and route, returning results and classification."""
        classification = await self._classify(query)
        retriever = self._get_retriever(classification)
        results = await retriever.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)
        return results, classification
