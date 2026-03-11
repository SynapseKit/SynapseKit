"""Tests for get_mermaid() output."""

from synapsekit.graph.graph import StateGraph
from synapsekit.graph.state import END


async def _noop(state):
    return {}


def _compile(g: StateGraph):
    return g.compile()


def test_mermaid_starts_with_flowchart():
    g = StateGraph()
    g.add_node("a", _noop).set_entry_point("a")
    mermaid = _compile(g).get_mermaid()
    assert mermaid.startswith("flowchart TD")


def test_mermaid_includes_entry_arrow():
    g = StateGraph()
    g.add_node("start", _noop).set_entry_point("start")
    mermaid = _compile(g).get_mermaid()
    assert "__start__ --> start" in mermaid


def test_mermaid_static_edge():
    g = StateGraph()
    g.add_node("a", _noop).add_node("b", _noop)
    g.add_edge("a", "b").set_entry_point("a")
    mermaid = _compile(g).get_mermaid()
    assert "a --> b" in mermaid


def test_mermaid_end_renders_as_dunder_end():
    g = StateGraph()
    g.add_node("a", _noop).set_entry_point("a").set_finish_point("a")
    mermaid = _compile(g).get_mermaid()
    assert "__end__" in mermaid
    # END constant itself should not appear as a raw string
    assert END not in mermaid or "__end__" in mermaid


def test_mermaid_conditional_edge_labels():
    g = StateGraph()
    g.add_node("router", _noop)
    g.add_node("yes_node", _noop)
    g.add_node("no_node", _noop)
    g.add_conditional_edge(
        "router",
        lambda s: "yes",
        {"yes": "yes_node", "no": "no_node"},
    )
    g.set_entry_point("router")
    mermaid = _compile(g).get_mermaid()
    assert "router -->|yes| yes_node" in mermaid
    assert "router -->|no| no_node" in mermaid


def test_mermaid_conditional_edge_end_label():
    g = StateGraph()
    g.add_node("a", _noop)
    g.add_conditional_edge("a", lambda s: "stop", {"stop": END})
    g.set_entry_point("a")
    mermaid = _compile(g).get_mermaid()
    assert "a -->|stop| __end__" in mermaid


def test_mermaid_multinode_graph():
    g = StateGraph()
    for n in ["a", "b", "c"]:
        g.add_node(n, _noop)
    g.add_edge("a", "b").add_edge("b", "c").add_edge("c", END)
    g.set_entry_point("a")
    mermaid = _compile(g).get_mermaid()
    assert "a --> b" in mermaid
    assert "b --> c" in mermaid
    assert "__end__" in mermaid
