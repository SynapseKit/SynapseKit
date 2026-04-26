import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient

from synapsekit.cli.graph_builder import create_app

client = TestClient(create_app())


def test_get_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>SynapseKit Graph Builder</title>" in response.text


def test_api_schema():
    response = client.get("/api/schema")
    assert response.status_code == 200
    data = response.json()
    assert "node_types" in data
    assert "edge_types" in data
    assert "rag_node" in data["node_types"]


def test_api_mermaid_valid():
    payload = {"nodes": [{"id": "node1", "type": "rag_node"}], "edges": [], "entry_point": "node1"}
    response = client.post("/api/mermaid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "mermaid" in data
    assert "node1" in data["mermaid"]


def test_api_mermaid_invalid():
    payload = "invalid json"
    # testclient will send as json if using `json=`, let's send as string body
    response = client.post("/api/mermaid", content=payload)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_api_codegen_valid():
    payload = {
        "nodes": [{"id": "node1", "type": "rag_node", "config": {"top_k": 5}}],
        "edges": [],
        "entry_point": "node1",
    }
    response = client.post("/api/codegen", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "code" in data
    assert "graph.add_node('node1', rag_node(top_k=5))" in data["code"]


def test_api_codegen_invalid():
    payload = "invalid json"
    response = client.post("/api/codegen", content=payload)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
