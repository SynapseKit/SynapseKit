"""Step-Back Retriever: generate a more abstract question to improve retrieval."""

from __future__ import annotations

import asyncio

from ..llm.base import BaseLLM
from .retriever import Retriever

_DEFAULT_PROMPT = (
    "Given the following question, generate a more abstract, higher-level 'step-back' "
    "question that would help retrieve broader context for answering the original question. "
    "Return only the step-back question, nothing else.\n\n"
    "Original question: {query}"
)


class StepBackRetriever:
    """Step-Back Prompting retriever.

    Generates a more abstract version of the query, retrieves for both the
    original and the step-back question in parallel, then merges and
    deduplicates the results.

    Usage::

        sb = StepBackRetriever(retriever=retriever, llm=llm)
        results = await sb.retrieve("What happens to entropy in a black hole?")
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
        """Retrieve using both original and step-back queries."""
        # Generate the step-back question
        prompt = self._prompt_template.format(query=query)
        step_back_query = await self._llm.generate(prompt)

        # Retrieve in parallel for both queries
        original_task = self._retriever.retrieve(
            query, top_k=top_k, metadata_filter=metadata_filter
        )
        step_back_task = self._retriever.retrieve(
            step_back_query.strip(), top_k=top_k, metadata_filter=metadata_filter
        )
        original_results, step_back_results = await asyncio.gather(original_task, step_back_task)

        # Merge and deduplicate, preserving order
        seen: set[str] = set()
        merged: list[str] = []
        for doc in original_results + step_back_results:
            if doc not in seen:
                seen.add(doc)
                merged.append(doc)

        return merged[:top_k]
