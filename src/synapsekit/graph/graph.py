from __future__ import annotations

from .edge import ConditionalEdge, ConditionFn, Edge
from .errors import GraphConfigError
from .node import Node, NodeFn
from .state import END


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
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge | ConditionalEdge] = []
        self._entry_point: str | None = None

    # ------------------------------------------------------------------ #
    # Builder API
    # ------------------------------------------------------------------ #

    def add_node(self, name: str, fn: NodeFn) -> StateGraph:
        self._nodes[name] = Node(name=name, fn=fn)
        return self

    def add_edge(self, src: str, dst: str) -> StateGraph:
        self._edges.append(Edge(src=src, dst=dst))
        return self

    def add_conditional_edge(
        self,
        src: str,
        condition_fn: ConditionFn,
        mapping: dict[str, str],
    ) -> StateGraph:
        self._edges.append(ConditionalEdge(src=src, condition_fn=condition_fn, mapping=mapping))
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        self._entry_point = name
        return self

    def set_finish_point(self, name: str) -> StateGraph:
        """Adds Edge(name, END) — shorthand for the final node."""
        return self.add_edge(name, END)

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
