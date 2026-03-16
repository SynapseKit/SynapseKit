"""Entity Memory: LLM-based entity extraction and tracking."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import BaseLLM

_EXTRACT_PROMPT = """\
Extract all named entities (people, places, organizations, concepts) \
from the following message. Return only a comma-separated list of entity names, \
nothing else. If no entities are found, return "NONE".

Message: {message}"""

_SUMMARIZE_ENTITY_PROMPT = """\
You are updating a knowledge base entry for the entity "{entity}".

Previous description: {previous}

New information from conversation:
{message}

Write a concise, updated description for "{entity}" that incorporates \
both the previous knowledge and the new information. Return only the description."""


class EntityMemory:
    """LLM-based entity extraction and tracking memory.

    Extracts entities from each message and maintains running descriptions.

    Usage::

        mem = EntityMemory(llm=llm)
        await mem.add("user", "Alice works at Acme Corp in Paris.")
        print(mem.get_entities())
    """

    def __init__(self, llm: BaseLLM, max_entities: int = 50) -> None:
        self._llm = llm
        self._max_entities = max_entities
        self._messages: list[dict] = []
        self._entities: OrderedDict[str, str] = OrderedDict()

    async def add(self, role: str, content: str) -> None:
        """Add a message and extract/update entities."""
        self._messages.append({"role": role, "content": content})

        # Extract entities
        extract_prompt = _EXTRACT_PROMPT.format(message=content)
        response = await self._llm.generate(extract_prompt)
        raw = response.strip()

        if raw.upper() == "NONE" or not raw:
            return

        entity_names = [e.strip() for e in raw.split(",") if e.strip()]

        # Update each entity's description
        for name in entity_names:
            previous = self._entities.get(name, "No previous information.")
            summarize_prompt = _SUMMARIZE_ENTITY_PROMPT.format(
                entity=name,
                previous=previous,
                message=content,
            )
            description = await self._llm.generate(summarize_prompt)
            # Move to end (most recently updated)
            if name in self._entities:
                self._entities.move_to_end(name)
            self._entities[name] = description.strip()

        # Evict oldest entities if over limit
        while len(self._entities) > self._max_entities:
            self._entities.popitem(last=False)

    def get_messages(self) -> list[dict]:
        """Return a copy of all messages."""
        return list(self._messages)

    def get_entities(self) -> dict[str, str]:
        """Return a copy of the entity store."""
        return dict(self._entities)

    def format_context(self) -> str:
        """Format entities and messages for prompt injection."""
        parts = []

        if self._entities:
            parts.append("Known entities:")
            for name, desc in self._entities.items():
                parts.append(f"  - {name}: {desc}")
            parts.append("")

        for m in self._messages:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")

        return "\n".join(parts)

    def clear(self) -> None:
        self._messages.clear()
        self._entities.clear()

    def __len__(self) -> int:
        return len(self._messages)
