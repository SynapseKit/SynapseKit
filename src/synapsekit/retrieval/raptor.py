"""RAPTORRetriever: Recursive Abstractive Processing for Tree-Organized Retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import BaseLLM
    from ..loaders.base import Document

_SUMMARISE_PROMPT = """\
Summarise the following document passages into a single concise paragraph \
that captures the key information. Return only the summary.

Passages:
{passages}"""


class RAPTORRetriever:
    """Recursive Abstractive Processing for Tree-Organized Retrieval (RAPTOR).

    Algorithm:
    1. Retrieve ``top_k * cluster_size`` docs from base retriever.
    2. Cluster them into groups of ``cluster_size``.
    3. Summarise each cluster with the LLM.
    4. Retrieve again using summaries.
    5. Merge and deduplicate results.

    Usage::

        raptor = RAPTORRetriever(base_retriever=retriever, llm=llm)
        results = await raptor.retrieve("What is quantum computing?", top_k=5)
    """

    def __init__(
        self,
        base_retriever: object,
        llm: BaseLLM,
        levels: int = 2,
        cluster_size: int = 5,
    ) -> None:
        self._base_retriever = base_retriever
        self._llm = llm
        self._levels = levels
        self._cluster_size = cluster_size

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
                # Already a Document-like object
                docs.append(r)
        return docs

    async def _summarise_cluster(self, docs: list[Document]) -> str:
        passages = "\n\n".join(d.text for d in docs)
        prompt = _SUMMARISE_PROMPT.format(passages=passages)
        return await self._llm.generate(prompt)

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve documents using RAPTOR multi-level summarisation."""
        fetch_k = top_k * self._cluster_size
        raw = await self._base_retriever.retrieve(query, top_k=fetch_k)
        docs = self._to_documents(raw)

        # Fall back if not enough docs to cluster
        if len(docs) < self._cluster_size:
            return docs[:top_k]

        for _ in range(self._levels):
            # Cluster into groups of cluster_size
            clusters = [
                docs[i : i + self._cluster_size] for i in range(0, len(docs), self._cluster_size)
            ]

            # Summarise each cluster
            summaries: list[str] = []
            for cluster in clusters:
                summary = await self._summarise_cluster(cluster)
                summaries.append(summary)

            # Retrieve again using each summary
            summary_docs: list[Document] = []
            seen: set[str] = {d.text for d in docs}

            for summary in summaries:
                results = await self._base_retriever.retrieve(summary, top_k=top_k)
                for r in self._to_documents(results):
                    if r.text not in seen:
                        summary_docs.append(r)
                        seen.add(r.text)

            docs = docs + summary_docs

        # Deduplicate final list preserving order
        final: list[Document] = []
        seen_final: set[str] = set()
        for d in docs:
            if d.text not in seen_final:
                final.append(d)
                seen_final.add(d.text)

        return final[:top_k]
