"""FLARE Retriever: Forward-Looking Active REtrieval."""

from __future__ import annotations

import re

from ..llm.base import BaseLLM
from .retriever import Retriever

_DEFAULT_GENERATE_PROMPT = (
    "Answer the following question. If you need to look up additional information, "
    "insert [SEARCH: your search query] markers in your response.\n\n"
    "Question: {query}\n\n"
    "Context:\n{context}"
)

_DEFAULT_REGENERATE_PROMPT = (
    "Continue answering the question using the additional context provided. "
    "If you still need more information, insert [SEARCH: query] markers.\n\n"
    "Question: {query}\n\n"
    "Previous answer so far: {answer}\n\n"
    "Additional context:\n{context}"
)

_SEARCH_MARKER_RE = re.compile(r"\[SEARCH:\s*(.+?)\]")


class FLARERetriever:
    """Forward-Looking Active REtrieval (FLARE).

    Iteratively generates an answer, detects ``[SEARCH: ...]`` markers
    indicating the LLM needs more information, retrieves for those queries,
    and regenerates until no more markers appear or ``max_iterations`` is reached.

    Usage::

        flare = FLARERetriever(retriever=retriever, llm=llm)
        results = await flare.retrieve("Explain quantum computing applications")
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        max_iterations: int = 3,
        generate_prompt: str | None = None,
        regenerate_prompt: str | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_iterations = max_iterations
        self._generate_prompt = generate_prompt or _DEFAULT_GENERATE_PROMPT
        self._regenerate_prompt = regenerate_prompt or _DEFAULT_REGENERATE_PROMPT

    def _parse_search_markers(self, text: str) -> list[str]:
        """Extract search queries from ``[SEARCH: ...]`` markers."""
        return _SEARCH_MARKER_RE.findall(text)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Iterative retrieve-generate loop with active retrieval."""
        # Step 1: initial retrieval
        all_docs: list[str] = await self._retriever.retrieve(
            query, top_k=top_k, metadata_filter=metadata_filter
        )

        context = "\n".join(all_docs)
        answer = ""

        for _iteration in range(self._max_iterations):
            # Step 2: generate answer (or continue)
            if not answer:
                prompt = self._generate_prompt.format(query=query, context=context)
            else:
                prompt = self._regenerate_prompt.format(query=query, answer=answer, context=context)

            answer = await self._llm.generate(prompt)

            # Step 3: check for search markers
            markers = self._parse_search_markers(answer)
            if not markers:
                break

            # Step 4: retrieve for each marker query, deduplicate
            seen = set(all_docs)
            for marker_query in markers:
                new_docs = await self._retriever.retrieve(
                    marker_query.strip(),
                    top_k=top_k,
                    metadata_filter=metadata_filter,
                )
                for doc in new_docs:
                    if doc not in seen:
                        seen.add(doc)
                        all_docs.append(doc)

            context = "\n".join(all_docs)

        return all_docs[:top_k]
