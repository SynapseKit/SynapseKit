from .approval import approval_node
from .checkpointers import (
    BaseCheckpointer,
    InMemoryCheckpointer,
    JSONFileCheckpointer,
    SQLiteCheckpointer,
)
from .compiled import CompiledGraph
from .dynamic_route import dynamic_route_node
from .edge import ConditionalEdge, ConditionFn, Edge
from .errors import GraphConfigError, GraphRuntimeError
from .fan_out import fan_out_node
from .graph import StateGraph
from .interrupt import GraphInterrupt, InterruptState
from .mermaid import get_mermaid_with_trace
from .node import Node, NodeFn, agent_node, llm_node, rag_node
from .state import END, GraphState, StateField, TypedState
from .streaming import EventHooks, GraphEvent, sse_stream, ws_stream
from .subgraph import subgraph_node
from .trace import ExecutionTrace, TraceEntry
from .visualization import GraphVisualizer

__all__ = [
    "END",
    "BaseCheckpointer",
    "CompiledGraph",
    "ConditionFn",
    "ConditionalEdge",
    "Edge",
    "EventHooks",
    "ExecutionTrace",
    "GraphVisualizer",
    "GraphConfigError",
    "GraphEvent",
    "GraphInterrupt",
    "GraphRuntimeError",
    "GraphState",
    "InMemoryCheckpointer",
    "InterruptState",
    "JSONFileCheckpointer",
    "Node",
    "NodeFn",
    "PostgresCheckpointer",
    "RedisCheckpointer",
    "SQLiteCheckpointer",
    "StateField",
    "StateGraph",
    "TraceEntry",
    "TypedState",
    "agent_node",
    "approval_node",
    "dynamic_route_node",
    "fan_out_node",
    "llm_node",
    "rag_node",
    "sse_stream",
    "subgraph_node",
    "get_mermaid_with_trace",
    "ws_stream",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    _lazy = {
        "RedisCheckpointer": "checkpointers.redis",
        "PostgresCheckpointer": "checkpointers.postgres",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(f".{_lazy[name]}", __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
