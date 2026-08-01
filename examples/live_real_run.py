"""Watch REAL SynapseKit subsystems stream into the Live dashboard.

    python examples/live_real_run.py

Opens http://127.0.0.1:7900 and runs a loop of real tool calls + agent-memory
(DB) reads/writes. Nothing here publishes events by hand — these are ordinary
SynapseKit objects; ``synapsekit.live.enable()`` auto-instruments `BaseTool` and
`AgentMemory` so their activity appears in the browser automatically (this is
#898). Add a knowledge-graph or MCP call and it shows up the same way.
"""

from __future__ import annotations

import asyncio

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.live import enable


class LookupOrderTool(BaseTool):
    name = "lookup_order"
    description = "Look up an order by id."
    parameters = {"type": "object", "properties": {"id": {"type": "string"}}}

    async def run(self, **kwargs: object) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult(output=f"order {kwargs.get('id')} — refunded")


async def main() -> None:
    enable(open_browser=True)  # starts the dashboard + instruments the subsystems
    from synapsekit.memory.agent_memory import AgentMemory

    memory = AgentMemory(backend="memory")
    tool = LookupOrderTool()
    print("Running a real tool + memory loop… (Ctrl+C to stop)")

    try:
        i = 0
        while True:
            i += 1
            oid = f"{48000 + i}"
            await tool.run(id=oid, operation="get")  # → tool.call
            await memory.store(agent_id="support", content=f"handled order {oid}")  # → memory.write
            await memory.recall(agent_id="support", query="order")  # → memory.read
            await asyncio.sleep(1.2)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    asyncio.run(main())
