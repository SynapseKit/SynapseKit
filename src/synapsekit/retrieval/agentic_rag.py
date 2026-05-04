"""AgenticRAGRetriever: tool-using retrieval agent with ReAct loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..llm.base import BaseLLM
    from ..loaders.base import Document

_SYSTEM_PROMPT = """\
You are a retrieval agent. You can call the tool `search_documents(query)` \
to retrieve relevant documents. After searching, decide if you have enough \
information or if you should search again with a different query.

Respond with ONE of:
- SEARCH: <your search query>
- DONE

Current iteration: {iteration}/{max_iterations}
Original question: {question}
Documents found so far: {doc_count}"""


class AgenticRAGRetriever:
    """Tool-using retrieval agent with a lightweight ReAct loop.

    Wraps the retriever as a tool (``search_documents``) and runs a
    ReAct loop to decide when to retrieve, what query to use, and
    whether to retrieve again.

    Usage::

        agent = AgenticRAGRetriever(retriever=retriever, llm=llm)
        results = await agent.retrieve("What is quantum computing?", top_k=5)
    """

    def __init__(
        self,
        retriever: object,
        llm: BaseLLM,
        tools: list[Any] | None = None,
        max_iterations: int = 3,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._tools = tools or []
        self._max_iterations = max_iterations

    def _to_documents(self, results: list) -> list[Document]:
        """Normalise retriever output to Document objects."""
        from ..loaders.base import Document

        docs: list[Document] = []
        for r in results:
            if isinstance(r, str):
                docs.append(Document(text=r))
            elif isinstance(r, dict):
                docs.append(Document(text=r.get("text", str(r)), metadata=r.get("metadata", {})))
            else:
                docs.append(r)
        return docs

    async def _search_documents(self, query: str, top_k: int) -> list[Document]:
        raw = await self._retriever.retrieve(query, top_k=top_k)
        return self._to_documents(raw)

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Run the ReAct loop to retrieve documents."""
        all_docs: list[Document] = []
        seen_texts: set[str] = set()

        for iteration in range(1, self._max_iterations + 1):
            prompt = _SYSTEM_PROMPT.format(
                iteration=iteration,
                max_iterations=self._max_iterations,
                question=query,
                doc_count=len(all_docs),
            )
            response = await self._llm.generate(prompt)
            response = response.strip()

            if response.upper().startswith("DONE") or not response.upper().startswith("SEARCH:"):
                break

            # Parse the search query from "SEARCH: <query>"
            search_query = response[len("SEARCH:") :].strip()
            if not search_query:
                search_query = query

            # Call search_documents tool
            new_docs = await self._search_documents(search_query, top_k)
            for doc in new_docs:
                if doc.text not in seen_texts:
                    all_docs.append(doc)
                    seen_texts.add(doc.text)

        # If the loop produced nothing (LLM replied DONE immediately), do a
        # single direct retrieval to guarantee non-empty results.
        if not all_docs:
            fallback = await self._search_documents(query, top_k)
            for doc in fallback:
                if doc.text not in seen_texts:
                    all_docs.append(doc)
                    seen_texts.add(doc.text)

        return all_docs[:top_k]
