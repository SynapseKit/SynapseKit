import json

import pytest

from synapsekit.cli.graph_builder import generate_python_code
from synapsekit.graph import END, StateGraph
from synapsekit.graph.errors import GraphConfigError


def test_json_roundtrip():
    # Build original graph
    graph = StateGraph()

    async def dummy_rag(state):
        return {"output": "rag"}

    async def dummy_llm(state):
        return {"output": "llm"}

    graph.add_node("retrieve", dummy_rag, metadata={"type": "rag_node", "config": {"top_k": 5}})
    graph.add_node(
        "generate", dummy_llm, metadata={"type": "llm_node", "config": {"model": "gpt-4"}}
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.set_entry_point("retrieve")

    json_str = graph.to_json()
    payload = json.loads(json_str)

    assert payload["entry_point"] == "retrieve"
    assert len(payload["nodes"]) == 2
    assert len(payload["edges"]) == 2

    # Import graph
    graph2 = StateGraph.from_json(json_str)

    assert graph2._entry_point == "retrieve"
    assert len(graph2._nodes) == 2
    assert len(graph2._edges) == 2
    assert "retrieve" in graph2._nodes
    assert "generate" in graph2._nodes

    # Metadata should be preserved
    assert graph2._nodes["retrieve"].metadata["type"] == "rag_node"
    assert graph2._nodes["retrieve"].metadata["config"]["top_k"] == 5


def test_invalid_schema_handling():
    invalid_json = '{"nodes": [{"id": "only_node", "type": "custom_node"}], "edges": [{"from": "only_node", "to": "missing_node"}]}'
    graph = StateGraph.from_json(invalid_json)
    graph.set_entry_point("only_node")

    with pytest.raises(GraphConfigError, match="is not a registered node"):
        graph.compile()


def test_python_codegen():
    payload = {
        "nodes": [
            {"id": "retrieve", "type": "rag_node", "config": {"top_k": 5}},
            {"id": "generate", "type": "llm_node", "config": {"model": "gpt-4o-mini"}},
            {"id": "custom", "type": "custom_node", "config": {}},
        ],
        "edges": [{"from": "retrieve", "to": "generate"}, {"from": "generate", "to": "custom"}],
        "conditional_edges": [
            {
                "from": "custom",
                "condition": "check_success",
                "mapping": {"true": "END", "false": "retrieve"},
            }
        ],
        "entry_point": "retrieve",
    }

    code = generate_python_code(payload)

    assert "graph.add_node('retrieve', rag_node(top_k=5))" in code
    assert "graph.add_node('generate', llm_node(model='gpt-4o-mini'))" in code
    assert "# TODO: Define function for custom_node" in code
    assert "graph.add_node('custom', custom_node)" in code
    assert "graph.add_edge('retrieve', 'generate')" in code
    assert (
        "graph.add_conditional_edge('custom', check_success, {'true': 'END', 'false': 'retrieve'})"
        in code
    )
    assert "graph.set_entry_point('retrieve')" in code
