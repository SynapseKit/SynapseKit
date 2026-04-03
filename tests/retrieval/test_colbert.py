import sys
import types
from unittest.mock import patch

import pytest

from synapsekit.retrieval.strategies.colbert import ColBERTRetriever


@pytest.mark.asyncio
async def test_colbert_retriever_import_error():
    with patch.dict(sys.modules, {"ragatouille": None}):
        retriever = ColBERTRetriever(
            model="colbert-ir/colbertv2.0",
            index_name="test_index",
            index_root=".colbert",
        )
        with pytest.raises(ImportError) as excinfo:
            retriever._get_ragatouille()
        assert "ragatouille required for ColBERTRetriever" in str(excinfo.value)


class DummyModel:
    def __init__(self):
        self.last_index_kwargs = {}
        self.last_search_kwargs = {}

    def index(self, collection, index_name, document_ids, index_root=None):
        self.last_index_kwargs = {
            "collection": collection,
            "index_name": index_name,
            "document_ids": document_ids,
            "index_root": index_root,
        }

    def search(self, query, k=None, top_k=None, index_name=None):
        self.last_search_kwargs = {
            "query": query,
            "k": k,
            "top_k": top_k,
            "index_name": index_name,
        }
        return [
            {"content": "doc 0", "score": 0.9, "document_id": 0},
            {"content": "doc 1", "score": 0.8, "document_id": 1},
        ]


class DummyRAGPretrainedModel:
    @staticmethod
    def from_pretrained(model, index_root=None):
        return DummyModel()


@pytest.mark.asyncio
async def test_colbert_retriever_add_and_retrieve():
    rag_mock = types.ModuleType("ragatouille")
    rag_mock.RAGPretrainedModel = DummyRAGPretrainedModel

    with patch.dict(sys.modules, {"ragatouille": rag_mock}):
        retriever = ColBERTRetriever(
            model="colbert-ir/colbertv2.0",
            index_name="test_index",
            index_root=".colbert",
        )

        texts = ["doc 0", "doc 1"]
        metadata = [{"meta": "data0"}, {"meta": "data1"}]

        await retriever.add(texts, metadata=metadata)

        model_instance = retriever._rag
        assert model_instance.last_index_kwargs["collection"] == texts
        assert model_instance.last_index_kwargs["index_name"] == "test_index"
        assert model_instance.last_index_kwargs["document_ids"] == [0, 1]
        assert model_instance.last_index_kwargs["index_root"] == ".colbert"

        results = await retriever.retrieve_with_scores("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["text"] == "doc 0"
        assert results[0]["score"] == 0.9
        assert results[0]["metadata"] == {"meta": "data0"}

        assert results[1]["text"] == "doc 1"
        assert results[1]["score"] == 0.8
        assert results[1]["metadata"] == {"meta": "data1"}

        assert model_instance.last_search_kwargs["query"] == "test query"
        assert model_instance.last_search_kwargs["index_name"] == "test_index"
        assert (
            model_instance.last_search_kwargs.get("k") == 2
            or model_instance.last_search_kwargs.get("top_k") == 2
        )
