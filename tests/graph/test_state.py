"""Tests for graph state module — END sentinel and GraphState."""

from synapsekit.graph.state import END, GraphState


def test_end_sentinel_value():
    assert END == "__end__"


def test_end_sentinel_is_string():
    assert isinstance(END, str)


def test_graph_state_is_dict():
    state: GraphState = {"input": "hello", "count": 42}
    assert isinstance(state, dict)
    assert state["input"] == "hello"


def test_state_merge_pattern():
    state: GraphState = {"a": 1, "b": 2}
    partial = {"b": 99, "c": 3}
    state.update(partial)
    assert state == {"a": 1, "b": 99, "c": 3}


def test_end_not_equal_to_valid_node_name():
    assert END != "end"
    assert END != "END"
    assert END != ""


def test_end_usable_as_dict_key():
    mapping = {END: "terminal"}
    assert mapping["__end__"] == "terminal"
