"""KnowledgeGraphMemory: knowledge graph conversation memory."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import BaseLLM

_EXTRACT_TRIPLETS_PROMPT = """\
Extract knowledge graph triplets from the following message.
Return a JSON array of objects, each with keys "s" (subject), "p" (predicate), \
"o" (object). Return only the JSON array, nothing else.
If no triplets can be extracted, return an empty array [].

Message: {message}

Example output: [{{"s": "Alice", "p": "works_at", "o": "Acme"}}]"""


class KnowledgeGraphMemory:
    """Knowledge graph conversation memory.

    Extracts entities and relations from messages and stores them as a graph.
    On ``get_context(query)``, returns relevant triplets as context messages.

    Usage::

        mem = KnowledgeGraphMemory(llm=llm)
        await mem.add_message("user", "Alice works at Acme Corp.")
        context = await mem.get_context("Where does Alice work?")
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_triplets: int = 100,
        return_triplets_in_context: bool = True,
    ) -> None:
        self._llm = llm
        self._max_triplets = max_triplets
        self._return_triplets_in_context = return_triplets_in_context
        # graph: entity -> list of (relation, target)
        self._graph: dict[str, list[tuple[str, str]]] = {}
        self._triplets: list[tuple[str, str, str]] = []
        self._messages: list[dict] = []

    async def add_message(self, role: str, content: str) -> None:
        """Add a message, extract triplets and store them in the graph."""
        self._messages.append({"role": role, "content": content})

        prompt = _EXTRACT_TRIPLETS_PROMPT.format(message=content)
        try:
            response = await self._llm.generate(prompt)
            raw = response.strip()
            # Find JSON array in response
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]
            triplets_data = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            # Graceful fallback — skip extraction on parse errors
            return

        for item in triplets_data:
            if not isinstance(item, dict):
                continue
            s = str(item.get("s", "")).strip()
            p = str(item.get("p", "")).strip()
            o = str(item.get("o", "")).strip()
            if not (s and p and o):
                continue

            self._triplets.append((s, p, o))
            self._graph.setdefault(s, []).append((p, o))

        # Evict oldest triplets if over limit
        if len(self._triplets) > self._max_triplets:
            excess = len(self._triplets) - self._max_triplets
            removed = self._triplets[:excess]
            self._triplets = self._triplets[excess:]
            # Rebuild graph to stay consistent
            self._graph.clear()
            for sub, pred, obj in self._triplets:
                self._graph.setdefault(sub, []).append((pred, obj))
            del removed

    async def get_context(self, query: str) -> list[dict]:
        """Return relevant triplets as context messages.

        Finds triplets whose subject or object appears in the query.
        Falls back to all triplets if none match.
        """
        if not self._triplets or not self._return_triplets_in_context:
            return list(self._messages)

        query_lower = query.lower()
        relevant: list[tuple[str, str, str]] = []

        for s, p, o in self._triplets:
            if s.lower() in query_lower or o.lower() in query_lower:
                relevant.append((s, p, o))

        if not relevant:
            relevant = list(self._triplets)

        lines = [f"{s} {p} {o}" for s, p, o in relevant]
        triplet_text = "Known facts:\n" + "\n".join(f"- {line}" for line in lines)
        return [{"role": "system", "content": triplet_text}]

    def clear(self) -> None:
        """Clear all messages, triplets, and graph."""
        self._messages.clear()
        self._triplets.clear()
        self._graph.clear()

    @property
    def graph(self) -> dict[str, list[tuple[str, str]]]:
        """Return a copy of the in-memory graph."""
        return dict(self._graph)

    @property
    def triplets(self) -> list[tuple[str, str, str]]:
        """Return a copy of all stored triplets."""
        return list(self._triplets)

    def __len__(self) -> int:
        return len(self._messages)
