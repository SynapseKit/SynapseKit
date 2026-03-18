"""A2A Agent Card -- describes agent capabilities for discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentCard:
    """Describes an agent's capabilities for A2A discovery.

    Usage::
        card = AgentCard(
            name="research-agent",
            description="Searches the web for information",
            skills=["web_search", "summarization"],
            endpoint="http://localhost:8001",
        )
    """

    name: str
    description: str
    skills: list[str] = field(default_factory=list)
    endpoint: str = ""
    version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "skills": self.skills,
            "endpoint": self.endpoint,
            "version": self.version,
            "metadata": self.metadata,
        }
