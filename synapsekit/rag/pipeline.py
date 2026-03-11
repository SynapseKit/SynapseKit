from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional

from ..llm.base import BaseLLM
from ..memory.conversation import ConversationMemory
from ..observability.tracer import TokenTracer
from ..retrieval.retriever import Retriever


class TextSplitter:
    """
    Simple recursive character text splitter. Zero external dependencies.
    Splits on paragraphs → sentences → words until chunks fit chunk_size.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        # Try splitting by paragraph, then sentence, then words
        for sep in ["\n\n", "\n", ". ", " "]:
            parts = text.split(sep)
            if len(parts) > 1:
                return self._merge(parts, sep)

        # Hard split as last resort
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

    def _merge(self, parts: List[str], sep: str) -> List[str]:
        chunks, current = [], ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single part exceeds chunk_size, split it recursively
                if len(part) > self.chunk_size:
                    chunks.extend(self.split(part))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)

        # Apply overlap: prepend tail of previous chunk
        if self.chunk_overlap <= 0 or len(chunks) < 2:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap:]
            overlapped.append(tail + chunks[i])
        return overlapped


@dataclass
class RAGConfig:
    llm: BaseLLM
    retriever: Retriever
    memory: ConversationMemory
    tracer: Optional[TokenTracer] = None
    retrieval_top_k: int = 5
    system_prompt: str = "Answer using only the provided context. If the context does not contain the answer, say so."
    chunk_size: int = 512
    chunk_overlap: int = 50


class RAGPipeline:
    """
    Full RAG orchestrator.
    Chunks incoming text, embeds + stores it, then retrieves and answers queries.
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._splitter = TextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    async def add(self, text: str, metadata: dict | None = None) -> None:
        """Chunk text and add to the vectorstore."""
        chunks = self._splitter.split(text)
        meta = [metadata or {} for _ in chunks]
        await self.config.retriever._store.add(chunks, meta)

    async def stream(
        self, query: str, top_k: int | None = None
    ) -> AsyncGenerator[str, None]:
        """Retrieve context, build prompt, stream LLM response, update memory."""
        k = top_k or self.config.retrieval_top_k
        chunks = await self.config.retriever.retrieve(query, top_k=k)

        context = "\n\n".join(chunks) if chunks else "No context available."
        history = self.config.memory.format_context()

        messages: List[dict] = [
            {"role": "system", "content": self.config.system_prompt},
        ]
        if history:
            messages.append({"role": "user", "content": f"Previous conversation:\n{history}"})
            messages.append({"role": "assistant", "content": "Understood."})

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        })

        tracer = self.config.tracer
        t0 = tracer.start_timer() if tracer else 0.0

        answer_parts: List[str] = []
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
