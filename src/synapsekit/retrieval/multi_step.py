"""Multi-Step Retriever: iterative retrieval-generation with gap identification."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import BaseLLM

from .retriever import Retriever

_ANSWER_PROMPT = """\
Answer the following question using only the provided context. \
If the context is insufficient, provide the best answer you can.

Context:
{context}

Question: {question}"""

_GAP_PROMPT = """\
Given a question and a partial answer, identify what information is still missing. \
Return a comma-separated list of concise search queries to fill the gaps. \
If the answer is complete, respond with exactly "COMPLETE".

Question: {question}

Current answer: {answer}"""


class MultiStepRetriever:
    """Iterative retrieval-generation: retrieve, generate, identify gaps, repeat.

    Usage::

        ms = MultiStepRetriever(retriever=retriever, llm=llm, max_steps=3)
        results = await ms.retrieve("What is X?")
        results, trace = await ms.retrieve_with_steps("What is X?")
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        max_steps: int = 3,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_steps = max_steps

    async def _generate_answer(self, question: str, context: list[str]) -> str:
        ctx_str = "\n\n".join(context)
        prompt = _ANSWER_PROMPT.format(context=ctx_str, question=question)
        return await self._llm.generate(prompt)

    async def _identify_gaps(self, question: str, answer: str) -> list[str]:
        prompt = _GAP_PROMPT.format(question=question, answer=answer)
        response = await self._llm.generate(prompt)
        text = response.strip()

        if text.upper() == "COMPLETE":
            return []

        return [q.strip() for q in text.split(",") if q.strip()]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]:
        """Iteratively retrieve and fill gaps."""
        docs, _ = await self.retrieve_with_steps(query, top_k, metadata_filter)
        return docs

    async def retrieve_with_steps(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> tuple[list[str], list[dict]]:
        """Iteratively retrieve and fill gaps, returning documents and trace."""
        all_docs: list[str] = []
        seen: set[str] = set()
        trace: list[dict] = []

        # Initial retrieval
        initial = await self._retriever.retrieve(
            query, top_k=top_k, metadata_filter=metadata_filter
        )
        for doc in initial:
            if doc not in seen:
                all_docs.append(doc)
                seen.add(doc)

        trace.append({"step": 0, "query": query, "new_docs": len(initial)})

        for step in range(1, self._max_steps + 1):
            # Generate answer from current docs
            answer = await self._generate_answer(query, all_docs)

            # Identify gaps
            gaps = await self._identify_gaps(query, answer)
            if not gaps:
                trace.append({"step": step, "query": None, "new_docs": 0, "complete": True})
                break

            # Retrieve for each gap query
            step_new = 0
            for gap_query in gaps:
                gap_docs = await self._retriever.retrieve(
                    gap_query, top_k=top_k, metadata_filter=metadata_filter
                )
                for doc in gap_docs:
                    if doc not in seen:
                        all_docs.append(doc)
                        seen.add(doc)
                        step_new += 1

            trace.append({"step": step, "query": gaps, "new_docs": step_new})

        return all_docs, trace
