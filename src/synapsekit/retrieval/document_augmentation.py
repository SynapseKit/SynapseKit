"""DocumentAugmentationRetriever: query + document expansion retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import BaseLLM
    from ..loaders.base import Document

_QUERY_EXPANSION_PROMPT = """\
Generate {n} alternative phrasings of the following search query. \
Return one query per line, nothing else.

Original query: {query}"""

_DOC_EXPANSION_PROMPT = """\
Given the following document passage, write a concise hypothetical answer \
that this passage might be responding to. Return only the hypothetical answer.

Passage: {passage}"""


class DocumentAugmentationRetriever:
    """Query and document expansion retrieval.

    Steps:
    1. If ``expand_queries``: use LLM to generate ``n_queries`` alternative
       phrasings, retrieve for each, merge results.
    2. If ``expand_documents``: for each retrieved doc, use LLM to generate a
       hypothetical answer appended to the doc text before re-ranking.

    Usage::

        aug = DocumentAugmentationRetriever(base_retriever=retriever, llm=llm)
        results = await aug.retrieve("What is machine learning?", top_k=5)
    """

    def __init__(
        self,
        base_retriever: object,
        llm: BaseLLM,
        expand_queries: bool = True,
        expand_documents: bool = True,
        n_queries: int = 3,
    ) -> None:
        self._base_retriever = base_retriever
        self._llm = llm
        self._expand_queries = expand_queries
        self._expand_documents = expand_documents
        self._n_queries = n_queries

    def _to_documents(self, results: list) -> list[Document]:
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

    async def _generate_queries(self, query: str) -> list[str]:
        prompt = _QUERY_EXPANSION_PROMPT.format(n=self._n_queries, query=query)
        response = await self._llm.generate(prompt)
        variants = [q.strip() for q in response.strip().splitlines() if q.strip()]
        # Always include original
        return [query, *variants[: self._n_queries]]

    async def _expand_document(self, doc: Document) -> Document:
        from ..loaders.base import Document

        prompt = _DOC_EXPANSION_PROMPT.format(passage=doc.text)
        hypothetical = await self._llm.generate(prompt)
        augmented_text = f"{doc.text}\n\n{hypothetical.strip()}"
        return Document(text=augmented_text, metadata=doc.metadata)

    async def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve with query and/or document expansion."""
        queries = [query]
        if self._expand_queries:
            queries = await self._generate_queries(query)

        # Retrieve for each query, deduplicate by text
        all_docs: list[Document] = []
        seen_texts: set[str] = set()

        for q in queries:
            raw = await self._base_retriever.retrieve(q, top_k=top_k)
            for doc in self._to_documents(raw):
                if doc.text not in seen_texts:
                    all_docs.append(doc)
                    seen_texts.add(doc.text)

        # Document expansion
        if self._expand_documents and all_docs:
            expanded: list[Document] = []
            for doc in all_docs:
                augmented = await self._expand_document(doc)
                expanded.append(augmented)
            all_docs = expanded

        return all_docs[:top_k]
