from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .edge import ConditionalEdge, ConditionFn, Edge
from .errors import GraphConfigError
from .node import Node, NodeFn
from .state import END, TypedState

MigrationResult = dict[str, Any] | tuple[str, dict[str, Any]]
MigrationFn = Callable[[dict[str, Any]], MigrationResult | Awaitable[MigrationResult]]


class StateGraph:
    """
    Fluent builder for DAG-based graph workflows.

    Usage::

        graph = StateGraph()
        graph.add_node("a", fn_a).add_node("b", fn_b)
        graph.add_edge("a", "b")
        graph.set_entry_point("a").set_finish_point("b")
        compiled = graph.compile()
        result = await compiled.run({"input": "hello"})

    With typed state and reducers::

        from synapsekit.graph.state import StateField, TypedState

        schema = TypedState(fields={
            "messages": StateField(default=list, reducer=lambda cur, new: cur + new),
        })
        graph = StateGraph(state_schema=schema)
    """

    def __init__(
        self,
        state_schema: TypedState | None = None,
        *,
        version: str = "1",
        migrations: dict[str, MigrationFn] | None = None,
    ) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge | ConditionalEdge] = []
        self._entry_point: str | None = None
        self._state_schema = state_schema
        self.version = version
        self.migrations = migrations or {}
        self.checkpointer_config: Any | None = None

    def __repr__(self) -> str:
        return f"StateGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"

    # ------------------------------------------------------------------ #
    # Builder API
    # ------------------------------------------------------------------ #

    def add_node(
        self, name: str, fn: NodeFn, *, metadata: dict[str, Any] | None = None
    ) -> StateGraph:
        self._nodes[name] = Node(name=name, fn=fn, metadata=metadata or {})
        return self

    def add_edge(self, src: str, dst: str, *, metadata: dict[str, Any] | None = None) -> StateGraph:
        self._edges.append(Edge(src=src, dst=dst, metadata=metadata or {}))
        return self

    def add_conditional_edge(
        self,
        src: str,
        condition_fn: ConditionFn,
        mapping: dict[str, str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> StateGraph:
        self._edges.append(
            ConditionalEdge(
                src=src, condition_fn=condition_fn, mapping=mapping, metadata=metadata or {}
            )
        )
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        self._entry_point = name
        return self

    def set_finish_point(self, name: str) -> StateGraph:
        """Adds Edge(name, END) — shorthand for the final node."""
        return self.add_edge(name, END)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_json(self) -> str:
        """Export graph definition to a JSON string."""
        import json

        nodes_list = []
        for name, node in self._nodes.items():
            nodes_list.append(
                {
                    "id": name,
                    "type": node.metadata.get("type", "custom_node"),
                    "config": node.metadata.get("config", {}),
                }
            )

        edges_list = []
        conditional_edges_list = []

        for edge in self._edges:
            if isinstance(edge, Edge):
                edges_list.append({"from": edge.src, "to": edge.dst})
            elif isinstance(edge, ConditionalEdge):
                conditional_edges_list.append(
                    {
                        "from": edge.src,
                        "condition": edge.metadata.get("condition_name", "custom_condition"),
                        "mapping": edge.mapping,
                        "config": edge.metadata.get("config", {}),
                    }
                )

        payload = {
            "version": self.version,
            "entry_point": self._entry_point,
            "nodes": nodes_list,
            "edges": edges_list,
            "conditional_edges": conditional_edges_list,
            "checkpointer": getattr(self, "checkpointer_config", None),
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        node_factories: dict[str, Callable[..., NodeFn]] | None = None,
        condition_factories: dict[str, Callable[..., ConditionFn]] | None = None,
    ) -> StateGraph:
        """Import graph definition from a JSON string."""
        import json

        payload = json.loads(json_str)
        graph = cls(version=payload.get("version", "1"))

        graph.checkpointer_config = payload.get("checkpointer")

        node_factories = node_factories or {}
        condition_factories = condition_factories or {}

        for node_data in payload.get("nodes", []):
            node_id = node_data["id"]
            node_type = node_data["type"]
            config = node_data.get("config", {})

            if node_type in node_factories:
                fn = node_factories[node_type](**config)
            else:

                async def _dummy_fn(state: dict[str, Any]) -> dict[str, Any]:
                    return {}

                fn = _dummy_fn

            graph.add_node(node_id, fn, metadata={"type": node_type, "config": config})

        for edge_data in payload.get("edges", []):
            graph.add_edge(edge_data["from"], edge_data["to"])

        for c_edge_data in payload.get("conditional_edges", []):
            src = c_edge_data["from"]
            cond_name = c_edge_data.get("condition", "custom_condition")
            mapping = c_edge_data["mapping"]

            if cond_name in condition_factories:
                cond_fn = condition_factories[cond_name](**c_edge_data.get("config", {}))
            else:

                async def _dummy_cond(state: dict[str, Any]) -> str:
                    return "END"

                cond_fn = _dummy_cond

            graph.add_conditional_edge(
                src,
                cond_fn,
                mapping,
                metadata={"condition_name": cond_name, "config": c_edge_data.get("config", {})},
            )

        entry_point = payload.get("entry_point")
        if entry_point:
            graph.set_entry_point(entry_point)

        return graph

    # ------------------------------------------------------------------ #
    # Compile
    # ------------------------------------------------------------------ #

    def compile(
        self,
        allow_cycles: bool = False,
        max_steps: int | None = None,
    ) -> CompiledGraph:
        self._validate(allow_cycles=allow_cycles)
        from .compiled import CompiledGraph

        return CompiledGraph(self, max_steps=max_steps)

    def _validate(self, allow_cycles: bool = False) -> None:
        if not self._entry_point:
            raise GraphConfigError("Entry point not set. Call set_entry_point() before compile().")

        if self._entry_point not in self._nodes:
            raise GraphConfigError(f"Entry point {self._entry_point!r} is not a registered node.")

        # Validate that all edge endpoints exist (except END)
        for edge in self._edges:
            if edge.src not in self._nodes:
                raise GraphConfigError(f"Edge source {edge.src!r} is not a registered node.")
            if isinstance(edge, Edge):
                if edge.dst != END and edge.dst not in self._nodes:
                    raise GraphConfigError(
                        f"Edge destination {edge.dst!r} is not a registered node."
                    )
            elif isinstance(edge, ConditionalEdge):
                for label, dst in edge.mapping.items():
                    if dst != END and dst not in self._nodes:
                        raise GraphConfigError(
                            f"Conditional edge mapping {label!r} → {dst!r}: "
                            f"{dst!r} is not a registered node."
                        )

        # Cycle detection on static edges only (skip if cycles are allowed)
        if not allow_cycles:
            self._check_cycles()

    def _check_cycles(self) -> None:
        """DFS cycle detection using only static (non-conditional) edges."""
        static_next: dict[str, list[str]] = {n: [] for n in self._nodes}
        for edge in self._edges:
            if isinstance(edge, Edge) and edge.dst != END:
                static_next[edge.src].append(edge.dst)

        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            for nb in static_next.get(node, []):
                if nb not in visited:
                    dfs(nb)
                elif nb in rec_stack:
                    raise GraphConfigError(f"Cycle detected in static edges involving node {nb!r}.")
            rec_stack.discard(node)

        for node in self._nodes:
            if node not in visited:
                dfs(node)


# Avoid circular import — import here so type checkers see it
from .compiled import CompiledGraph  # noqa: E402
