"""Tests for vector store backends (Chroma, FAISS, Qdrant, Pinecone) — mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def make_mock_embeddings(dim=4):
    mock = MagicMock()

    async def embed(texts):
        vecs = []
        for i, _t in enumerate(texts):
            v = np.zeros(dim, dtype=np.float32)
            v[i % dim] = 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    async def embed_one(text):
        arr = await embed([text])
        return arr[0]

    mock.embed = embed
    mock.embed_one = embed_one
    return mock


# ------------------------------------------------------------------ #
# VectorStore ABC
# ------------------------------------------------------------------ #


class TestVectorStoreABC:
    def test_abc_cannot_be_instantiated(self):
        from synapsekit.retrieval.base import VectorStore

        with pytest.raises(TypeError):
            VectorStore()

    def test_save_raises_not_implemented(self):
        from synapsekit.retrieval.base import VectorStore

        class Minimal(VectorStore):
            async def add(self, texts, metadata=None):
                pass

            async def search(self, query, top_k=5):
                return []

        vs = Minimal()
        with pytest.raises(NotImplementedError):
            vs.save("path")

    def test_load_raises_not_implemented(self):
        from synapsekit.retrieval.base import VectorStore

        class Minimal(VectorStore):
            async def add(self, texts, metadata=None):
                pass

            async def search(self, query, top_k=5):
                return []

        vs = Minimal()
        with pytest.raises(NotImplementedError):
            vs.load("path")


# ------------------------------------------------------------------ #
# InMemoryVectorStore implements VectorStore
# ------------------------------------------------------------------ #


class TestInMemoryImplementsABC:
    def test_is_subclass_of_vectorstore(self):
        from synapsekit.retrieval.base import VectorStore
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        assert issubclass(InMemoryVectorStore, VectorStore)

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(make_mock_embeddings())
        await store.add(["hello", "world"])
        results = await store.search("hello", top_k=2)
        assert len(results) == 2


# ------------------------------------------------------------------ #
# ChromaVectorStore (mocked chromadb)
# ------------------------------------------------------------------ #


class TestChromaVectorStore:
    def _make_chroma_mocks(self):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"src": "a"}, {"src": "b"}]],
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_chromadb.PersistentClient.return_value = mock_client
        return mock_chromadb, mock_collection

    def test_import_error_without_chromadb(self):
        with patch.dict("sys.modules", {"chromadb": None}):
            import importlib

            import synapsekit.retrieval.chroma as chroma_mod

            importlib.reload(chroma_mod)  # force re-evaluation
            with pytest.raises(ImportError, match="chromadb"):
                chroma_mod.ChromaVectorStore(make_mock_embeddings())

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        mock_chromadb, mock_collection = self._make_chroma_mocks()
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            from synapsekit.retrieval.chroma import ChromaVectorStore

            store = ChromaVectorStore(make_mock_embeddings())
            await store.add(["text1", "text2"])
            mock_collection.add.assert_called_once()

            results = await store.search("query", top_k=2)
            assert len(results) == 2
            assert results[0]["text"] == "doc1"
            assert results[0]["score"] == pytest.approx(0.9, abs=0.01)

    @pytest.mark.asyncio
    async def test_add_empty_does_nothing(self):
        mock_chromadb, mock_collection = self._make_chroma_mocks()
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            from synapsekit.retrieval.chroma import ChromaVectorStore

            store = ChromaVectorStore(make_mock_embeddings())
            await store.add([])
            mock_collection.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_empty_returns_empty(self):
        mock_chromadb, mock_collection = self._make_chroma_mocks()
        mock_collection.count.return_value = 0
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            from synapsekit.retrieval.chroma import ChromaVectorStore

            store = ChromaVectorStore(make_mock_embeddings())
            results = await store.search("query")
            assert results == []


# ------------------------------------------------------------------ #
# FAISSVectorStore (mocked faiss)
# ------------------------------------------------------------------ #


class TestFAISSVectorStore:
    def _make_faiss_mock(self):
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.7]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )
        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        return mock_faiss, mock_index

    def test_import_error_without_faiss(self):
        with patch.dict("sys.modules", {"faiss": None}):
            import importlib

            import synapsekit.retrieval.faiss as faiss_mod

            importlib.reload(faiss_mod)
            with pytest.raises(ImportError, match="faiss-cpu"):
                faiss_mod.FAISSVectorStore(make_mock_embeddings())

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        mock_faiss, _mock_index = self._make_faiss_mock()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from synapsekit.retrieval.faiss import FAISSVectorStore

            store = FAISSVectorStore(make_mock_embeddings())
            await store.add(["alpha", "beta"], metadata=[{"id": 1}, {"id": 2}])
            assert len(store._texts) == 2

            results = await store.search("query", top_k=2)
            assert len(results) == 2
            assert results[0]["text"] == "alpha"

    @pytest.mark.asyncio
    async def test_search_empty_returns_empty(self):
        mock_faiss, _ = self._make_faiss_mock()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from synapsekit.retrieval.faiss import FAISSVectorStore

            store = FAISSVectorStore(make_mock_embeddings())
            results = await store.search("q")
            assert results == []

    @pytest.mark.asyncio
    async def test_add_empty_does_nothing(self):
        mock_faiss, _mock_index = self._make_faiss_mock()
        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from synapsekit.retrieval.faiss import FAISSVectorStore

            store = FAISSVectorStore(make_mock_embeddings())
            await store.add([])
            assert store._index is None


# ------------------------------------------------------------------ #
# QdrantVectorStore (mocked qdrant_client)
# ------------------------------------------------------------------ #


class TestQdrantVectorStore:
    def _make_qdrant_mocks(self):
        mock_result = MagicMock()
        mock_result.payload = {"text": "qdrant doc", "src": "a"}
        mock_result.score = 0.95

        mock_client = MagicMock()
        mock_client.search.return_value = [mock_result]
        mock_client.get_collection.side_effect = Exception("not found")

        mock_qdrant_client = MagicMock()
        mock_qdrant_client.QdrantClient.return_value = mock_client

        mock_models = MagicMock()

        return mock_qdrant_client, mock_models, mock_client

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        mock_qc, mock_models, mock_client = self._make_qdrant_mocks()
        with patch.dict(
            "sys.modules",
            {
                "qdrant_client": mock_qc,
                "qdrant_client.models": mock_models,
            },
        ):
            from synapsekit.retrieval.qdrant import QdrantVectorStore

            store = QdrantVectorStore(make_mock_embeddings())
            await store.add(["test text"])
            mock_client.upsert.assert_called_once()

            results = await store.search("query")
            assert len(results) == 1
            assert results[0]["text"] == "qdrant doc"
            assert results[0]["score"] == pytest.approx(0.95)

    def test_import_error_without_qdrant(self):
        with patch.dict("sys.modules", {"qdrant_client": None}):
            import importlib

            import synapsekit.retrieval.qdrant as qdrant_mod

            importlib.reload(qdrant_mod)
            with pytest.raises(ImportError, match="qdrant-client"):
                qdrant_mod.QdrantVectorStore(make_mock_embeddings())


# ------------------------------------------------------------------ #
# PineconeVectorStore (mocked pinecone)
# ------------------------------------------------------------------ #


class TestPineconeVectorStore:
    def _make_pinecone_mocks(self):
        mock_match = MagicMock()
        mock_match.metadata = {"text": "pine doc", "category": "test"}
        mock_match.score = 0.88

        mock_query_result = MagicMock()
        mock_query_result.matches = [mock_match]

        mock_index = MagicMock()
        mock_index.query.return_value = mock_query_result

        mock_pc_instance = MagicMock()
        mock_pc_instance.Index.return_value = mock_index

        mock_pinecone = MagicMock()
        mock_pinecone.Pinecone.return_value = mock_pc_instance

        return mock_pinecone, mock_index

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        mock_pinecone, mock_index = self._make_pinecone_mocks()
        with patch.dict("sys.modules", {"pinecone": mock_pinecone}):
            from synapsekit.retrieval.pinecone import PineconeVectorStore

            store = PineconeVectorStore(make_mock_embeddings(), index_name="test", api_key="key")
            await store.add(["pine text"], metadata=[{"category": "test"}])
            mock_index.upsert.assert_called_once()

            results = await store.search("query")
            assert len(results) == 1
            assert results[0]["text"] == "pine doc"
            assert results[0]["score"] == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_add_empty_does_nothing(self):
        mock_pinecone, mock_index = self._make_pinecone_mocks()
        with patch.dict("sys.modules", {"pinecone": mock_pinecone}):
            from synapsekit.retrieval.pinecone import PineconeVectorStore

            store = PineconeVectorStore(make_mock_embeddings(), index_name="test", api_key="key")
            await store.add([])
            mock_index.upsert.assert_not_called()

    def test_import_error_without_pinecone(self):
        with patch.dict("sys.modules", {"pinecone": None}):
            import importlib

            import synapsekit.retrieval.pinecone as pine_mod

            importlib.reload(pine_mod)
            with pytest.raises(ImportError, match="pinecone"):
                pine_mod.PineconeVectorStore(
                    make_mock_embeddings(), index_name="idx", api_key="key"
                )
