from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from synapsekit.retrieval.hyde import HyDERetriever


class TestHyDERetriever:
    @pytest.mark.asyncio
    async def test_retrieve_uses_hypothetical(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "Hypothetical answer about quantum physics"

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = ["chunk1", "chunk2"]

        hyde = HyDERetriever(retriever=mock_retriever, llm=mock_llm)
        results = await hyde.retrieve("What is quantum entanglement?", top_k=3)

        mock_llm.generate.assert_called_once()
        prompt_arg = mock_llm.generate.call_args[0][0]
        assert "quantum entanglement" in prompt_arg

        mock_retriever.retrieve.assert_called_once_with(
            "Hypothetical answer about quantum physics",
            top_k=3,
            metadata_filter=None,
        )
        assert results == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_custom_prompt_template(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "custom answer"

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = []

        template = "Answer this: {query}"
        hyde = HyDERetriever(retriever=mock_retriever, llm=mock_llm, prompt_template=template)
        await hyde.retrieve("test query")

        prompt_arg = mock_llm.generate.call_args[0][0]
        assert prompt_arg == "Answer this: test query"

    @pytest.mark.asyncio
    async def test_metadata_filter_passed(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "hypo"

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = []

        hyde = HyDERetriever(retriever=mock_retriever, llm=mock_llm)
        await hyde.retrieve("q", top_k=2, metadata_filter={"type": "article"})

        mock_retriever.retrieve.assert_called_once_with(
            "hypo", top_k=2, metadata_filter={"type": "article"}
        )
