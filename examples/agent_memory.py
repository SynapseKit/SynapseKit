"""Persistent memory integration for agents.

Run:
    python examples/agent_memory.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from synapsekit import AgentConfig, AgentExecutor, PersistentAgentMemory


class DemoLLM:
    """Tiny demo LLM that inspects injected memory text."""

    async def generate_with_messages(self, messages: list[dict]) -> str:
        system_prompt = str(messages[0].get("content", ""))
        if "prefers tea" in system_prompt.lower():
            return "Thought: I now know the final answer.\nFinal Answer: You prefer tea."
        return "Thought: I now know the final answer.\nFinal Answer: I don't know yet."

    async def stream_with_messages(self, messages: list[dict]) -> AsyncGenerator[str]:
        text = await self.generate_with_messages(messages)
        for token in text.split(" "):
            yield token + " "


async def main() -> None:
    memory = PersistentAgentMemory(backend="memory")
    await memory.store(
        agent_id="demo-user",
        memory_type="semantic",
        content="User prefers tea in the evening.",
    )

    executor = AgentExecutor(
        AgentConfig(
            llm=DemoLLM(),
            tools=[],
            agent_type="react",
            memory=memory,
            agent_id="demo-user",
            memory_top_k=3,
        )
    )

    answer = await executor.run("What drink do I usually prefer?")
    print("Answer:", answer)

    episodic_count = await memory.count(agent_id="demo-user", memory_type="episodic")
    print("Episodic memories stored:", episodic_count)


if __name__ == "__main__":
    asyncio.run(main())
