from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

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
    splitter: BaseSplitter | None = field(default=None)


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

    async def add(self, text: str, metadata: dict | None = None) -> None:
        """Chunk text and add to the vectorstore."""
        chunks = self._splitter.split(text)
        meta = [metadata or {} for _ in chunks]
        await self.config.retriever.add(chunks, meta)

    async def add_documents(self, docs: list[Document]) -> None:
        """Chunk and add a list of Documents to the vectorstore."""
        for doc in docs:
            await self.add(doc.text, doc.metadata)

    async def stream(self, query: str, top_k: int | None = None) -> AsyncGenerator[str]:
        """Retrieve context, build prompt, stream LLM response, update memory."""
        k = top_k or self.config.retrieval_top_k
        chunks = await self.config.retriever.retrieve(query, top_k=k)

        context = "\n\n".join(chunks) if chunks else "No context available."
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
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            }
        )

        tracer = self.config.tracer
        t0 = tracer.start_timer() if tracer else 0.0

        answer_parts: list[str] = []
        async for token in self.config.llm.stream_with_messages(messages):
            answer_parts.append(token)
            yield token

        answer = "".join(answer_parts)
        self.config.memory.add("user", query)
        self.config.memory.add("assistant", answer)

        if tracer:
            used = self.config.llm.tokens_used
            tracer.record(
                input_tokens=used["input"],
                output_tokens=used["output"],
                latency_ms=tracer.elapsed_ms(t0),
            )

    async def ask(self, query: str, top_k: int | None = None) -> str:
        return "".join([t async for t in self.stream(query, top_k=top_k)])
