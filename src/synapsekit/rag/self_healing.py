"""Self-healing RAG: retry with alternate retrieval strategies on low-quality answers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from .._compat import run_sync
from ..evaluation.faithfulness import FaithfulnessMetric
from ..llm.base import BaseLLM
from ..memory.conversation import ConversationMemory
from ..observability.tracer import TokenTracer

logger = logging.getLogger(__name__)


class RetrievalStrategy(Protocol):
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[str]: ...


@dataclass
class SelfHealingReport:
    success: bool
    attempts: int
    retries: int
    strategy: str | None
    scores: list[float] = field(default_factory=list)
    threshold: float = 0.0


class SelfHealingRAG:
    """Retry retrieval strategies when faithfulness is below a threshold.

    Example::

        rag = SelfHealingRAG(
            llm=llm,
            strategies=[
                HybridSearchRetriever(vectorstore=vs),
                CRAGRetriever(retriever=retriever, llm=llm),
            ],
            quality_threshold=0.75,
            max_retries=2,
        )

        answer = await rag.ask("Complex question...")
    """

    def __init__(
        self,
        *,
        llm: BaseLLM,
        strategies: Sequence[RetrievalStrategy],
        quality_threshold: float = 0.75,
        max_retries: int = 2,
        system_prompt: str = "Answer using only the provided context. If the context does not contain the answer, say so.",
        retrieval_top_k: int = 5,
        memory: ConversationMemory | None = None,
        tracer: TokenTracer | None = None,
        metric: FaithfulnessMetric | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("strategies must not be empty")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self._llm = llm
        self._strategies = list(strategies)
        self._quality_threshold = quality_threshold
        self._max_retries = max_retries
        self._system_prompt = system_prompt
        self._retrieval_top_k = retrieval_top_k
        self._memory = memory or ConversationMemory()
        self._tracer = tracer
        self._metric = metric or FaithfulnessMetric(llm)
        self._last_report: SelfHealingReport | None = None

    @property
    def last_report(self) -> SelfHealingReport | None:
        return self._last_report

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    async def ask(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> str:
        k = top_k or self._retrieval_top_k
        max_attempts = min(len(self._strategies), 1 + self._max_retries)
        scores: list[float] = []
        last_answer = ""
        last_strategy: str | None = None

        for attempt in range(max_attempts):
            strategy = self._strategies[attempt]
            last_strategy = type(strategy).__name__

            try:
                chunks = await self._retrieve(strategy, query, k, metadata_filter)
                last_answer = await self._generate_answer(query, chunks)
                metric_result = await self._metric.evaluate(
                    question=query,
                    answer=last_answer,
                    contexts=chunks,
                )
                scores.append(metric_result.score)

                if metric_result.score >= self._quality_threshold:
                    self._commit_answer(query, last_answer)
                    self._last_report = SelfHealingReport(
                        success=True,
                        attempts=attempt + 1,
                        retries=max(0, attempt),
                        strategy=last_strategy,
                        scores=scores,
                        threshold=self._quality_threshold,
                    )
                    logger.info(
                        "SelfHealingRAG success with %s after %s retries (score=%.3f)",
                        last_strategy,
                        max(0, attempt),
                        metric_result.score,
                    )
                    return last_answer

                logger.info(
                    "SelfHealingRAG retry: %s faithfulness %.3f < %.3f",
                    last_strategy,
                    metric_result.score,
                    self._quality_threshold,
                )
            except Exception:
                logger.exception("SelfHealingRAG attempt failed with %s", last_strategy)
                if attempt >= max_attempts - 1:
                    self._last_report = SelfHealingReport(
                        success=False,
                        attempts=max_attempts,
                        retries=max(0, max_attempts - 1),
                        strategy=last_strategy,
                        scores=scores,
                        threshold=self._quality_threshold,
                    )
                    raise

        # No attempt met the threshold; return the last answer produced.
        if last_answer:
            self._commit_answer(query, last_answer)
        self._last_report = SelfHealingReport(
            success=False,
            attempts=max_attempts,
            retries=max(0, max_attempts - 1),
            strategy=last_strategy,
            scores=scores,
            threshold=self._quality_threshold,
        )
        logger.info(
            "SelfHealingRAG exhausted strategies after %s retries; last=%s",
            max(0, max_attempts - 1),
            last_strategy,
        )
        return last_answer

    def ask_sync(self, query: str, **kw) -> str:
        return run_sync(self.ask(query, **kw))

    async def _retrieve(
        self,
        strategy: RetrievalStrategy,
        query: str,
        top_k: int,
        metadata_filter: dict | None,
    ) -> list[str]:
        retrieve = getattr(strategy, "retrieve", None)
        if retrieve is None or not callable(retrieve):
            raise AttributeError(f"Strategy {type(strategy).__name__} has no retrieve method")

        try:
            return cast(
                list[str], await retrieve(query, top_k=top_k, metadata_filter=metadata_filter)
            )
        except TypeError:
            # Some retrievers don't accept metadata_filter
            return cast(list[str], await retrieve(query, top_k=top_k))

    async def _generate_answer(self, query: str, chunks: list[str]) -> str:
        if chunks:
            tagged = [f"<document>\n{chunk}\n</document>" for chunk in chunks]
            context = "\n\n".join(tagged)
        else:
            context = "No context available."

        history = self._memory.format_context()
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
        ]
        if history:
            messages.append({"role": "user", "content": f"Previous conversation:\n{history}"})
            messages.append({"role": "assistant", "content": "Understood."})
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

        tracer = self._tracer
        t0 = tracer.start_timer() if tracer else 0.0
        answer = await self._llm.generate_with_messages(messages)

        if tracer:
            used = self._llm.tokens_used
            tracer.record(
                input_tokens=used["input"],
                output_tokens=used["output"],
                latency_ms=tracer.elapsed_ms(t0),
            )

        return answer

    def _commit_answer(self, query: str, answer: str) -> None:
        self._memory.add("user", query)
        self._memory.add("assistant", answer)
