"""Self-Query Retrieval: LLM generates metadata filters from natural language queries."""

from __future__ import annotations

import json

from ..llm.base import BaseLLM
from .retriever import Retriever

_SELF_QUERY_PROMPT = """\
You are a query analyzer. Given a user question, extract:
1. A semantic search query (the core information need)
2. Metadata filters (structured constraints from the question)

Available metadata fields: {fields}

Respond with JSON only:
{{"query": "semantic search text", "filters": {{"field": "value", ...}}}}

If no filters apply, use an empty filters object.
Do not include any text outside the JSON.

User question: {question}"""


class SelfQueryRetriever:
    """Self-Query Retrieval: uses an LLM to decompose a question into
    a semantic query and metadata filters.

    Usage::

        sqr = SelfQueryRetriever(
            retriever=retriever,
            llm=llm,
            metadata_fields=["source", "author", "year", "category"],
        )
        results = await sqr.retrieve("Papers by John about ML from 2024", top_k=5)
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        metadata_fields: list[str],
        prompt: str | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._fields = metadata_fields
        self._prompt = prompt or _SELF_QUERY_PROMPT

    async def _decompose_query(self, question: str) -> tuple[str, dict]:
        """Use LLM to split question into semantic query + metadata filters."""
        prompt = self._prompt.format(
            fields=", ".join(self._fields),
            question=question,
        )
        response = await self._llm.generate(prompt)

        try:
            # Extract JSON from response
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            query = parsed.get("query", question)
            filters = parsed.get("filters", {})
            # Only keep filters for known fields
            filters = {k: v for k, v in filters.items() if k in self._fields and v}
            return query, filters
        except json.JSONDecodeError, KeyError:
            # Fallback: use original question with no filters
            return question, {}

    async def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[str]:
        """Retrieve with LLM-generated metadata filters."""
        query, filters = await self._decompose_query(question)
        metadata_filter = filters if filters else None
        return await self._retriever.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)

    async def retrieve_with_filters(
        self,
        question: str,
        top_k: int = 5,
    ) -> tuple[list[str], dict]:
        """Retrieve and also return the extracted filters for transparency."""
        query, filters = await self._decompose_query(question)
        metadata_filter = filters if filters else None
        results = await self._retriever.retrieve(
            query, top_k=top_k, metadata_filter=metadata_filter
        )
        return results, {"query": query, "filters": filters}
