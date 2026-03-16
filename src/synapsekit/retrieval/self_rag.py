"""Self-RAG: self-reflective retrieval-augmented generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import BaseLLM

from .retriever import Retriever

_RELEVANCE_PROMPT = """\
You are a relevance grader. Given a question and a document, determine if the \
document is relevant to answering the question.

Respond with exactly one word: "relevant" or "irrelevant".

Question: {question}

Document: {document}"""

_GENERATE_PROMPT = """\
Answer the following question using only the provided context. If the context \
does not contain enough information, say so.

Context:
{context}

Question: {question}"""

_SUPPORT_PROMPT = """\
Given a question, an answer, and supporting documents, evaluate how well the \
documents support the answer.

Respond with exactly one word: "fully", "partially", or "not".

Question: {question}

Answer: {answer}

Documents:
{documents}"""

_CRITIQUE_PROMPT = """\
The following answer was not fully supported by the retrieved documents. \
Rewrite the question to retrieve better supporting evidence.

Original question: {question}
Previous answer: {answer}

Return only the rewritten question, nothing else."""


class SelfRAGRetriever:
    """Self-reflective RAG: retrieve, grade relevance, generate, check support, retry.

    Usage::

        self_rag = SelfRAGRetriever(retriever=retriever, llm=llm)
        results = await self_rag.retrieve("What is X?")
        results, meta = await self_rag.retrieve_with_reflection("What is X?")
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        max_iterations: int = 2,
        relevance_threshold: float = 0.5,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_iterations = max_iterations
        self._relevance_threshold = relevance_threshold

    async def _grade_relevance(self, question: str, document: str) -> bool:
        prompt = _RELEVANCE_PROMPT.format(question=question, document=document)
        response = await self._llm.generate(prompt)
        return response.strip().lower().startswith("relevant")

    async def _generate_answer(self, question: str, documents: list[str]) -> str:
        context = "\n\n".join(documents)
        prompt = _GENERATE_PROMPT.format(context=context, question=question)
        return await self._llm.generate(prompt)

    async def _check_support(self, question: str, answer: str, documents: list[str]) -> str:
        docs_str = "\n\n".join(documents)
        prompt = _SUPPORT_PROMPT.format(question=question, answer=answer, documents=docs_str)
        response = await self._llm.generate(prompt)
        result = response.strip().lower()
        for level in ("fully", "partially", "not"):
            if result.startswith(level):
                return level
        return "not"

    async def _rewrite_query(self, question: str, answer: str) -> str:
        prompt = _CRITIQUE_PROMPT.format(question=question, answer=answer)
        return (await self._llm.generate(prompt)).strip()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Retrieve with self-reflection, returning only documents."""
        docs, _ = await self.retrieve_with_reflection(query, top_k, metadata_filter)
        return docs

    async def retrieve_with_reflection(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> tuple[list[str], dict]:
        """Retrieve with self-reflection, returning documents and metadata."""
        current_query = query
        metadata: dict = {"iterations": 0, "support_level": "not"}

        for iteration in range(self._max_iterations):
            metadata["iterations"] = iteration + 1

            # Retrieve
            candidates = await self._retriever.retrieve(
                current_query, top_k=top_k * 2, metadata_filter=metadata_filter
            )

            if not candidates:
                return [], metadata

            # Grade relevance
            relevant = []
            for doc in candidates:
                if await self._grade_relevance(current_query, doc):
                    relevant.append(doc)
                    if len(relevant) >= top_k:
                        break

            if not relevant:
                return [], metadata

            # Generate answer
            answer = await self._generate_answer(current_query, relevant)

            # Check support
            support = await self._check_support(current_query, answer, relevant)
            metadata["support_level"] = support

            if support == "fully":
                return relevant[:top_k], metadata

            # Not fully supported — rewrite and retry (if iterations remain)
            if iteration < self._max_iterations - 1:
                current_query = await self._rewrite_query(current_query, answer)

        return relevant[:top_k], metadata
