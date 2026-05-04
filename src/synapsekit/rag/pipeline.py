from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import BaseLLM
from ..loaders.base import Document
from ..memory.conversation import ConversationMemory
from ..observability.tracer import TokenTracer
from ..retrieval.retriever import Retriever
from ..text_splitters.base import BaseSplitter
from ..text_splitters.recursive import RecursiveCharacterTextSplitter

# Backward-compatible alias
TextSplitter = RecursiveCharacterTextSplitter


@dataclass
class RAGConfig:
    llm: BaseLLM
    retriever: Retriever
    memory: ConversationMemory
    tracer: TokenTracer | None = None
    retrieval_top_k: int = 5
    system_prompt: str = "Answer using only the provided context. If the context does not contain the answer, say so."
    chunk_size: int = 512
    chunk_overlap: int = 50
    auto_eval: bool = False
    splitter: BaseSplitter | None = field(default=None)
    context_packer: Any | None = None


class RAGPipeline:
    """
    Full RAG orchestrator.
    Chunks incoming text, embeds + stores it, then retrieves and answers queries.
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._splitter: BaseSplitter = config.splitter or TextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self._auto_eval_tasks: set[asyncio.Task[None]] = set()

    def __repr__(self) -> str:
        model = type(self.config.llm).__name__
        splitter = type(self._splitter).__name__
        return f"RAGPipeline(llm={model!r}, splitter={splitter!r})"

    async def add(self, text: str, metadata: dict | None = None) -> None:
        """Chunk text and add to the vectorstore.

        Silently skips empty or whitespace-only text.
        """
        if not text or not text.strip():
            return

        # Retriever-level long-context strategies can choose full-context
        # ingestion vs chunking themselves.
        add_document = getattr(self.config.retriever, "add_document", None)
        retriever_cls = type(self.config.retriever)
        explicit_add_document = "add_document" in getattr(
            self.config.retriever, "__dict__", {}
        ) or hasattr(retriever_cls, "add_document")
        if explicit_add_document and callable(add_document):
            await add_document(text, metadata=metadata)
            return

        chunks = self._splitter.split(text)
        if not chunks:
            return
        meta = [metadata or {} for _ in chunks]
        await self.config.retriever.add(chunks, meta)

    async def add_documents(self, docs: list[Document]) -> None:
        """Chunk and add a list of Documents to the vectorstore.

        Empty documents are silently skipped.
        """
        for doc in docs:
            await self.add(doc.text, doc.metadata)

    async def _has_documents(self) -> bool:
        """Check if the vectorstore has any documents."""
        probe = await self.config.retriever.retrieve("test", top_k=1)
        return len(probe) > 0

    @staticmethod
    def _format_context_result(result: dict[str, Any]) -> str:
        text = str(result.get("text", "")).strip()
        metadata = result.get("metadata") or {}

        lines: list[str] = []
        for key in (
            "source_type",
            "source",
            "chunk_type",
            "page",
            "timestamp",
            "start_time",
            "end_time",
            "locator",
            "frame_index",
            "media_type",
        ):
            value = metadata.get(key)
            if value is not None:
                lines.append(f"{key}: {value}")

        score = result.get("score")
        if isinstance(score, int | float):
            lines.append(f"score: {score:.4f}")

        source_block = ""
        if lines:
            source_block = "[SOURCE]\n" + "\n".join(lines) + "\n[/SOURCE]\n"

        return f"{source_block}<document>\n{text}\n</document>"

    async def stream(self, query: str, top_k: int | None = None) -> AsyncGenerator[str]:
        """Retrieve context, build prompt, stream LLM response, update memory."""
        from ..observe.runtime import end_span, record_exception, start_span

        k = top_k or self.config.retrieval_top_k
        rag_span = start_span(
            "rag.ask",
            {
                "rag.query": query,
                "rag.top_k": k,
            },
        )
        retrieve_span = start_span(
            "rag.retrieve",
            {
                "rag.query": query,
                "rag.top_k": k,
            },
        )

        results: list[dict] | list[str] = []
        chunks: list[str] = []
        top_score: float | None = None
        try:
            retriever_state = getattr(self.config.retriever, "__dict__", {})
            retrieve_overridden = "retrieve" in retriever_state
            retrieve_with_scores_overridden = "retrieve_with_scores" in retriever_state

            retrieve_with_scores = getattr(self.config.retriever, "retrieve_with_scores", None)
            score_call = None
            if callable(retrieve_with_scores) and (
                not retrieve_overridden or retrieve_with_scores_overridden
            ):
                score_call = retrieve_with_scores(query, top_k=k)

            if score_call is not None and inspect.isawaitable(score_call):
                scored_results = await score_call
                results = scored_results
                chunks = [
                    str(item.get("text", "")) for item in scored_results if isinstance(item, dict)
                ]
                if scored_results and isinstance(scored_results[0], dict):
                    score = scored_results[0].get("score")
                    if score is None:
                        score = scored_results[0].get("relevance_score")
                    if score is None:
                        score = scored_results[0].get("cross_encoder_score")
                    top_score = float(score) if score is not None else None
            else:
                plain_results = await self.config.retriever.retrieve(query, top_k=k)
                results = plain_results
                chunks = [str(item) for item in plain_results]

            if self.config.context_packer is not None and chunks:
                packed = self.config.context_packer.pack(
                    results if results else chunks, query=query
                )
                chunks = [item["text"] if isinstance(item, dict) else str(item) for item in packed]

            end_span(
                retrieve_span,
                attributes={
                    "rag.retrieved_chunks": len(chunks),
                    "rag.top_score": top_score,
                    "rag.retrieval_latency_ms": round(retrieve_span.duration_ms, 3)
                    if retrieve_span is not None
                    else None,
                },
            )
        except Exception as exc:
            record_exception(retrieve_span, exc)
            end_span(retrieve_span, error=exc)
            record_exception(rag_span, exc)
            end_span(rag_span, error=exc)
            raise

        if results:
            if results and isinstance(results[0], dict):
                tagged = [
                    self._format_context_result(result)
                    for result in results
                    if isinstance(result, dict)
                ]
            else:
                tagged = [f"<document>\n{chunk}\n</document>" for chunk in chunks]
            context = "\n\n".join(tagged)
        else:
            context = "No context available."
        history = self.config.memory.format_context()

        messages: list[dict] = [
            {"role": "system", "content": self.config.system_prompt},
        ]
        if history:
            messages.append({"role": "user", "content": f"Previous conversation:\n{history}"})
            messages.append({"role": "assistant", "content": "Understood."})

        messages.append(
            {
                "role": "user",
                "content": (
                    "Context:\n"
                    f"{context}\n\n"
                    "Use the provided source metadata for citations when available.\n\n"
                    f"Question: {query}"
                ),
            }
        )

        tracer = self.config.tracer
        t0 = tracer.start_timer() if tracer else 0.0

        answer_parts: list[str] = []
        try:
            async for token in self.config.llm.stream_with_messages(messages):
                answer_parts.append(token)
                yield token
        except Exception as exc:
            record_exception(rag_span, exc)
            raise
        finally:
            # Commit the turn to memory + tracer only if at least one token
            # was delivered to the consumer. This preserves the user query
            # and partial answer when the consumer disconnects early (client
            # HTTP drop, upstream exception, explicit break) — the finally
            # runs even when GeneratorExit is raised at the yield point.
            # The answer_parts guard prevents recording a "ghost turn" when
            # the LLM call failed before streaming began.
            if answer_parts:
                answer = "".join(answer_parts)
                self.config.memory.add("user", query)
                self.config.memory.add("assistant", answer)

                if tracer:
                    used = self.config.llm.tokens_used
                    call_id = tracer.record(
                        input_tokens=used["input"],
                        output_tokens=used["output"],
                        latency_ms=tracer.elapsed_ms(t0),
                    )

                    if self.config.auto_eval and tracer.enabled:
                        self._schedule_auto_eval(
                            query=query,
                            answer=answer,
                            contexts=chunks,
                            call_id=call_id,
                        )

                response_span = start_span(
                    "rag.response",
                    {
                        "rag.response_length": len(answer),
                    },
                )
                end_span(response_span)

            end_span(
                rag_span,
                attributes={
                    "rag.retrieved_chunks": len(chunks),
                    "rag.response_length": len("".join(answer_parts)) if answer_parts else 0,
                },
            )

    def _schedule_auto_eval(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        call_id: int,
    ) -> None:
        task = asyncio.create_task(
            self._run_auto_eval(
                query=query,
                answer=answer,
                contexts=contexts,
                call_id=call_id,
            )
        )
        self._auto_eval_tasks.add(task)
        task.add_done_callback(self._auto_eval_tasks.discard)

    async def _run_auto_eval(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        call_id: int,
    ) -> None:
        tracer = self.config.tracer
        if tracer is None or not tracer.enabled:
            return

        try:
            from ..evaluation import EvaluationPipeline, FaithfulnessMetric, RelevancyMetric

            metrics = [
                FaithfulnessMetric(self.config.llm),
                RelevancyMetric(self.config.llm),
            ]
            pipeline = EvaluationPipeline(metrics=metrics)
            result = await pipeline.evaluate(
                question=query,
                answer=answer,
                contexts=contexts,
            )
            tracer.record_quality(
                faithfulness=result.scores.get("faithfulness"),
                relevancy=result.scores.get("relevancy"),
                call_id=call_id,
            )
        except Exception:
            # Auto-eval is best-effort. Never break the primary RAG call.
            return

    async def wait_for_auto_eval(self) -> None:
        if not self._auto_eval_tasks:
            return
        await asyncio.gather(*list(self._auto_eval_tasks), return_exceptions=True)

    async def ask(self, query: str, top_k: int | None = None) -> str:
        return "".join([t async for t in self.stream(query, top_k=top_k)])
