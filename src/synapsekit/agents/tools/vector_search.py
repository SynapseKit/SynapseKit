"""Vector Search Tool: search a vector store knowledge base."""

from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class VectorSearchTool(BaseTool):
    """Search a vector store knowledge base for relevant documents.

    Wraps a :class:`~synapsekit.retrieval.retriever.Retriever` so agents can
    query an existing knowledge base as a tool action.

    Usage::

        from synapsekit import Retriever, InMemoryVectorStore
        retriever = Retriever(InMemoryVectorStore(dim=384))
        tool = VectorSearchTool(retriever)
        result = await tool.run(query="machine learning basics")
    """

    name = "vector_search"
    description = (
        "Search a vector store knowledge base for relevant documents. "
        "Input: a search query. "
        "Returns: matching documents from the knowledge base."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        retriever: Any,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._retriever = retriever
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description

    async def run(self, query: str = "", top_k: int = 5, **kwargs: Any) -> ToolResult:
        search_query = query or kwargs.get("input", "")
        if not search_query:
            return ToolResult(output="", error="No search query provided.")

        try:
            results = await self._retriever.retrieve(search_query, top_k=top_k)

            if not results:
                return ToolResult(output="No results found.")

            formatted = []
            for i, text in enumerate(results, 1):
                formatted.append(f"{i}. {text}")

            return ToolResult(output="\n\n".join(formatted))
        except Exception as e:
            return ToolResult(output="", error=f"Vector search failed: {e}")
