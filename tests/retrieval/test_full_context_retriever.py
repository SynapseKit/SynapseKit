from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.retrieval.full_context import FullContextRetriever
from synapsekit.retrieval.token_counting import TokenCounter


def _word_counter() -> TokenCounter:
    return TokenCounter(count_fn=lambda text: len([t for t in text.split() if t]))


@pytest.mark.asyncio
async def test_add_document_uses_full_context_when_doc_fits():
    retriever = MagicMock()
    retriever.add = AsyncMock()

    full = FullContextRetriever(
        retriever=retriever,
        max_doc_tokens=10,
        token_counter=_word_counter(),
    )

    await full.add_document("tiny document", metadata={"source": "unit"})

    retriever.add.assert_awaited_once()
    texts, metadata = retriever.add.await_args.args
    assert texts == ["tiny document"]
    assert metadata[0]["full_context"] is True
    assert metadata[0]["doc_tokens"] == 2


@pytest.mark.asyncio
async def test_add_document_chunks_when_doc_exceeds_limit():
    retriever = MagicMock()
    retriever.add = AsyncMock()

    full = FullContextRetriever(
        retriever=retriever,
        max_doc_tokens=5,
        token_counter=_word_counter(),
        chunk_size=20,
        chunk_overlap=0,
    )

    long_text = "one two three four five six seven eight nine ten"
    await full.add_document(long_text, metadata={"source": "unit"})

    retriever.add.assert_awaited_once()
    chunk_texts, chunk_meta = retriever.add.await_args.args

    assert len(chunk_texts) > 1
    assert all(m["full_context"] is False for m in chunk_meta)
    assert all("chunk_index" in m for m in chunk_meta)
    assert all("chunk_tokens" in m for m in chunk_meta)


@pytest.mark.asyncio
async def test_add_batch_routes_each_document_correctly():
    retriever = MagicMock()
    retriever.add = AsyncMock()

    full = FullContextRetriever(
        retriever=retriever,
        max_doc_tokens=3,
        token_counter=_word_counter(),
        chunk_size=12,
        chunk_overlap=0,
    )

    texts = ["one two", "one two three four five"]
    metadata = [{"id": 1}, {"id": 2}]

    await full.add(texts, metadata)

    assert retriever.add.await_count == 2


@pytest.mark.asyncio
async def test_retrieve_delegates_to_wrapped_retriever():
    retriever = MagicMock()
    retriever.add = AsyncMock()
    retriever.retrieve = AsyncMock(return_value=["chunk A"])
    retriever.retrieve_with_scores = AsyncMock(
        return_value=[{"text": "chunk A", "score": 0.9, "metadata": {}}]
    )

    full = FullContextRetriever(
        retriever=retriever,
        max_doc_tokens=10,
        token_counter=_word_counter(),
    )

    texts = await full.retrieve("query", top_k=1)
    scored = await full.retrieve_with_scores("query", top_k=1)

    assert texts == ["chunk A"]
    assert scored[0]["text"] == "chunk A"


def test_invalid_max_doc_tokens_raises():
    import pytest

    retriever = MagicMock()
    with pytest.raises(ValueError, match="max_doc_tokens must be > 0"):
        FullContextRetriever(retriever=retriever, max_doc_tokens=0)


@pytest.mark.asyncio
async def test_add_document_empty_text_is_noop():
    retriever = MagicMock()
    retriever.add = AsyncMock()
    full = FullContextRetriever(
        retriever=retriever, max_doc_tokens=10, token_counter=_word_counter()
    )
    await full.add_document("   ")
    retriever.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_with_scores_fallback_when_not_available():
    """If wrapped retriever has no retrieve_with_scores, FullContextRetriever synthesises it."""
    retriever = MagicMock(spec=["add", "retrieve"])
    retriever.add = AsyncMock()
    retriever.retrieve = AsyncMock(return_value=["result A"])

    full = FullContextRetriever(
        retriever=retriever, max_doc_tokens=10, token_counter=_word_counter()
    )
    scored = await full.retrieve_with_scores("q")
    assert scored[0]["text"] == "result A"
    assert scored[0]["score"] is None
