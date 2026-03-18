"""A2A Protocol types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TaskState = Literal["pending", "running", "completed", "failed", "cancelled"]


@dataclass
class A2AMessage:
    """A message in an A2A conversation."""

    role: Literal["user", "agent"]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class A2ATask:
    """A task in the A2A protocol."""

    id: str
    state: TaskState = "pending"
    messages: list[A2AMessage] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: Literal["user", "agent"], content: str) -> None:
        self.messages.append(A2AMessage(role=role, content=content))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "messages": [
                {"role": m.role, "content": m.content, "metadata": m.metadata}
                for m in self.messages
            ],
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }
