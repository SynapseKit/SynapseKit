"""Tests for v0.5.1 features: validation, file checks, metadata filtering, exports."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# ------------------------------------------------------------------ #
# Validation: max_iterations
# ------------------------------------------------------------------ #


class TestAgentValidation:
    def test_function_calling_agent_rejects_zero_iterations(self):
        from synapsekit.agents.function_calling import FunctionCallingAgent

        with pytest.raises(ValueError, match="max_iterations"):
            FunctionCallingAgent(llm=MagicMock(), tools=[], max_iterations=0)

    def test_function_calling_agent_rejects_negative_iterations(self):
        from synapsekit.agents.function_calling import FunctionCallingAgent

        with pytest.raises(ValueError, match="max_iterations"):
            FunctionCallingAgent(llm=MagicMock(), tools=[], max_iterations=-1)

    def test_react_agent_rejects_zero_iterations(self):
        from synapsekit.agents.react import ReActAgent

        with pytest.raises(ValueError, match="max_iterations"):
            ReActAgent(llm=MagicMock(), tools=[], max_iterations=0)

    def test_react_agent_rejects_negative_iterations(self):
        from synapsekit.agents.react import ReActAgent

        with pytest.raises(ValueError, match="max_iterations"):
            ReActAgent(llm=MagicMock(), tools=[], max_iterations=-1)

    def test_function_calling_agent_accepts_valid_iterations(self):
        from synapsekit.agents.function_calling import FunctionCallingAgent

        agent = FunctionCallingAgent(llm=MagicMock(), tools=[], max_iterations=1)
        assert agent._max_iterations == 1


# ------------------------------------------------------------------ #
# Validation: ConversationMemory window
# ------------------------------------------------------------------ #


class TestMemoryValidation:
    def test_rejects_zero_window(self):
        from synapsekit.memory.conversation import ConversationMemory

        with pytest.raises(ValueError, match="window"):
            ConversationMemory(window=0)

    def test_rejects_negative_window(self):
        from synapsekit.memory.conversation import ConversationMemory

        with pytest.raises(ValueError, match="window"):
            ConversationMemory(window=-5)

    def test_accepts_valid_window(self):
        from synapsekit.memory.conversation import ConversationMemory

        mem = ConversationMemory(window=1)
        assert mem._window == 1


# ------------------------------------------------------------------ #
# File existence checks
# ------------------------------------------------------------------ #


class TestLoaderFileChecks:
    def test_pdf_loader_missing_file(self):
        from synapsekit.loaders.pdf import PDFLoader

        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            PDFLoader("/nonexistent/file.pdf").load()

    def test_html_loader_missing_file(self):
        from synapsekit.loaders.html import HTMLLoader

        with pytest.raises(FileNotFoundError, match="HTML file not found"):
            HTMLLoader("/nonexistent/file.html").load()

    def test_csv_loader_missing_file(self):
        from synapsekit.loaders.csv import CSVLoader

        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            CSVLoader("/nonexistent/file.csv").load()

    def test_json_loader_missing_file(self):
        from synapsekit.loaders.json_loader import JSONLoader

        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            JSONLoader("/nonexistent/file.json").load()


# ------------------------------------------------------------------ #
# Vector store exports
# ------------------------------------------------------------------ #


class TestVectorStoreExports:
    def test_retrieval_exports_all_backends(self):
        import synapsekit.retrieval as ret

        assert "ChromaVectorStore" in ret.__all__
        assert "FAISSVectorStore" in ret.__all__
        assert "QdrantVectorStore" in ret.__all__
        assert "PineconeVectorStore" in ret.__all__
        assert "MongoDBAtlasVectorStore" in ret.__all__

    def test_top_level_exports(self):
        import synapsekit

        assert "ChromaVectorStore" in synapsekit.__all__
        assert "FAISSVectorStore" in synapsekit.__all__
        assert "MongoDBAtlasVectorStore" in synapsekit.__all__


# ------------------------------------------------------------------ #
# Metadata filtering
# ------------------------------------------------------------------ #


class TestMetadataFiltering:
    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        mock_embeddings = MagicMock()
        # 3 documents: 2 about "python", 1 about "java"
        vecs = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1]], dtype=np.float32)
        mock_embeddings.embed = AsyncMock(return_value=vecs)
        mock_embeddings.embed_one = AsyncMock(return_value=np.array([1, 0, 0], dtype=np.float32))

        store = InMemoryVectorStore(mock_embeddings)
        await store.add(
            ["python basics", "python advanced", "java intro"],
            [{"lang": "python"}, {"lang": "python"}, {"lang": "java"}],
        )

        # Without filter — all 3 candidates
        results = await store.search("programming", top_k=3)
        assert len(results) == 3

        # With filter — only python docs
        results = await store.search("programming", top_k=3, metadata_filter={"lang": "python"})
        assert len(results) == 2
        assert all(r["metadata"]["lang"] == "python" for r in results)

    @pytest.mark.asyncio
    async def test_search_with_no_matching_filter(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        mock_embeddings = MagicMock()
        vecs = np.array([[1, 0, 0]], dtype=np.float32)
        mock_embeddings.embed = AsyncMock(return_value=vecs)
        mock_embeddings.embed_one = AsyncMock(return_value=np.array([1, 0, 0], dtype=np.float32))

        store = InMemoryVectorStore(mock_embeddings)
        await store.add(["doc1"], [{"lang": "python"}])

        results = await store.search("query", top_k=5, metadata_filter={"lang": "rust"})
        assert results == []

    @pytest.mark.asyncio
    async def test_search_without_filter_unchanged(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        mock_embeddings = MagicMock()
        vecs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        mock_embeddings.embed = AsyncMock(return_value=vecs)
        mock_embeddings.embed_one = AsyncMock(return_value=np.array([1, 0, 0], dtype=np.float32))

        store = InMemoryVectorStore(mock_embeddings)
        await store.add(["doc1", "doc2"], [{"a": 1}, {"a": 2}])

        # No filter — same behavior as before
        results = await store.search("query", top_k=2, metadata_filter=None)
        assert len(results) == 2


# ------------------------------------------------------------------ #
# @tool decorator (import from top level)
# ------------------------------------------------------------------ #


class TestToolDecoratorImport:
    def test_importable_from_synapsekit(self):
        from synapsekit import tool

        @tool(name="test_tool", description="A test")
        def my_tool(x: str) -> str:
            return x

        assert my_tool.name == "test_tool"
