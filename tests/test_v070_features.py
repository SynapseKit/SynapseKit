"""Tests for v0.7.0 features: MCP client/server, multi-agent orchestration."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ------------------------------------------------------------------ #
# Mock LLM helper
# ------------------------------------------------------------------ #


class MockLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def generate(self, prompt: str, **kw):
        r = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return r

    async def stream(self, prompt: str, **kw):
        yield self._responses[self._idx % len(self._responses)]
        self._idx += 1


# ------------------------------------------------------------------ #
# Version
# ------------------------------------------------------------------ #


def test_version():
    import synapsekit

    assert synapsekit.__version__ == "1.1.0"


# ------------------------------------------------------------------ #
# MCP Client
# ------------------------------------------------------------------ #


def test_mcp_client_import():
    from synapsekit.mcp.client import MCPClient, MCPToolAdapter

    assert MCPClient is not None
    assert MCPToolAdapter is not None


def test_mcp_client_top_level_import():
    from synapsekit import MCPClient, MCPToolAdapter

    assert MCPClient is not None
    assert MCPToolAdapter is not None


def test_mcp_tool_adapter_creation():
    from synapsekit.mcp.client import MCPToolAdapter

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.inputSchema = {"type": "object", "properties": {"x": {"type": "string"}}}

    adapter = MCPToolAdapter(mock_tool)
    assert adapter.name == "test_tool"
    assert adapter.description == "A test tool"
    assert adapter.parameters == {"type": "object", "properties": {"x": {"type": "string"}}}


def test_mcp_tool_adapter_no_description():
    from synapsekit.mcp.client import MCPToolAdapter

    mock_tool = MagicMock()
    mock_tool.name = "my_tool"
    mock_tool.description = None
    mock_tool.inputSchema = {}

    adapter = MCPToolAdapter(mock_tool)
    assert adapter.description == "MCP tool: my_tool"


def test_mcp_tool_adapter_no_input_schema():
    from synapsekit.mcp.client import MCPToolAdapter

    mock_tool = MagicMock(spec=["name", "description"])
    mock_tool.name = "bare_tool"
    mock_tool.description = "No schema"

    adapter = MCPToolAdapter(mock_tool)
    assert adapter.parameters == {}


async def test_mcp_tool_adapter_default_run():
    from synapsekit.mcp.client import MCPToolAdapter

    mock_tool = MagicMock()
    mock_tool.name = "t"
    mock_tool.description = "d"
    mock_tool.inputSchema = {}

    adapter = MCPToolAdapter(mock_tool)
    result = await adapter.run()
    assert result.is_error
    assert "not connected" in result.error.lower()


def test_mcp_client_init():
    from synapsekit.mcp.client import MCPClient

    client = MCPClient()
    assert client._session is None
    assert client._tools == []


async def test_mcp_client_connect_stdio_missing_mcp():
    from synapsekit.mcp.client import MCPClient

    # Temporarily remove mcp from sys.modules to force ImportError
    saved = sys.modules.pop("mcp", None)
    saved2 = sys.modules.pop("mcp.client.stdio", None)
    try:
        with patch.dict(sys.modules, {"mcp": None, "mcp.client.stdio": None}):
            client = MCPClient()
            with pytest.raises(ImportError, match="mcp package required"):
                await client.connect_stdio("uvx", ["some-server"])
    finally:
        if saved is not None:
            sys.modules["mcp"] = saved
        if saved2 is not None:
            sys.modules["mcp.client.stdio"] = saved2


async def test_mcp_client_connect_sse_missing_mcp():
    from synapsekit.mcp.client import MCPClient

    saved = sys.modules.pop("mcp", None)
    saved2 = sys.modules.pop("mcp.client.sse", None)
    try:
        with patch.dict(sys.modules, {"mcp": None, "mcp.client.sse": None}):
            client = MCPClient()
            with pytest.raises(ImportError, match="mcp package required"):
                await client.connect_sse("http://localhost:8000")
    finally:
        if saved is not None:
            sys.modules["mcp"] = saved
        if saved2 is not None:
            sys.modules["mcp.client.sse"] = saved2


async def test_mcp_client_load_tools_not_connected():
    from synapsekit.mcp.client import MCPClient

    client = MCPClient()
    with pytest.raises(RuntimeError, match="Not connected"):
        await client._load_tools()


async def test_mcp_client_close_no_session():
    from synapsekit.mcp.client import MCPClient

    client = MCPClient()
    # Should not raise
    await client.close()


async def test_mcp_client_context_manager():
    from synapsekit.mcp.client import MCPClient

    async with MCPClient() as client:
        assert client._session is None


def test_mcp_tool_adapter_schema():
    from synapsekit.mcp.client import MCPToolAdapter

    mock_tool = MagicMock()
    mock_tool.name = "calc"
    mock_tool.description = "Calculator"
    mock_tool.inputSchema = {"type": "object", "properties": {"expr": {"type": "string"}}}

    adapter = MCPToolAdapter(mock_tool)
    schema = adapter.schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calc"


# ------------------------------------------------------------------ #
# MCP Server
# ------------------------------------------------------------------ #


def test_mcp_server_import():
    from synapsekit.mcp.server import MCPServer

    assert MCPServer is not None


def test_mcp_server_top_level_import():
    from synapsekit import MCPServer

    assert MCPServer is not None


def test_mcp_server_init():
    from synapsekit.mcp.server import MCPServer

    server = MCPServer(name="test", version="2.0.0")
    assert server._name == "test"
    assert server._version == "2.0.0"
    assert server._tools == {}


def test_mcp_server_init_with_tools():
    from synapsekit.mcp.server import MCPServer

    tool = MagicMock()
    tool.name = "calc"
    server = MCPServer(tools=[tool])
    assert "calc" in server._tools


def test_mcp_server_add_tool():
    from synapsekit.mcp.server import MCPServer

    server = MCPServer()
    tool = MagicMock()
    tool.name = "my_tool"
    server.add_tool(tool)
    assert "my_tool" in server._tools


def test_mcp_server_build_missing_mcp():
    from synapsekit.mcp.server import MCPServer

    saved = sys.modules.pop("mcp", None)
    saved2 = sys.modules.pop("mcp.server", None)
    saved3 = sys.modules.pop("mcp.types", None)
    try:
        with patch.dict(sys.modules, {"mcp": None, "mcp.server": None, "mcp.types": None}):
            server = MCPServer()
            with pytest.raises(ImportError, match="mcp package required"):
                server._build_server()
    finally:
        if saved is not None:
            sys.modules["mcp"] = saved
        if saved2 is not None:
            sys.modules["mcp.server"] = saved2
        if saved3 is not None:
            sys.modules["mcp.types"] = saved3


def test_mcp_server_run_missing_mcp():
    from synapsekit.mcp.server import MCPServer

    saved = sys.modules.pop("mcp", None)
    saved2 = sys.modules.pop("mcp.server", None)
    saved3 = sys.modules.pop("mcp.server.stdio", None)
    saved4 = sys.modules.pop("mcp.types", None)
    try:
        with patch.dict(
            sys.modules,
            {"mcp": None, "mcp.server": None, "mcp.server.stdio": None, "mcp.types": None},
        ):
            server = MCPServer()
            with pytest.raises(ImportError, match="mcp package required"):
                server.run()
    finally:
        for key, val in [
            ("mcp", saved),
            ("mcp.server", saved2),
            ("mcp.server.stdio", saved3),
            ("mcp.types", saved4),
        ]:
            if val is not None:
                sys.modules[key] = val


# ------------------------------------------------------------------ #
# Supervisor Agent
# ------------------------------------------------------------------ #


def test_supervisor_import():
    from synapsekit.agents.multi.supervisor import SupervisorAgent, WorkerAgent

    assert SupervisorAgent is not None
    assert WorkerAgent is not None


def test_supervisor_top_level_import():
    from synapsekit import SupervisorAgent, WorkerAgent

    assert SupervisorAgent is not None
    assert WorkerAgent is not None


def test_worker_agent_construction():
    from synapsekit.agents.multi.supervisor import WorkerAgent

    executor = MagicMock()
    w = WorkerAgent("researcher", "Does research", executor)
    assert w.name == "researcher"
    assert w.role == "Does research"
    assert w.executor is executor


def test_supervisor_construction():
    from synapsekit.agents.multi.supervisor import SupervisorAgent, WorkerAgent

    llm = MockLLM(["FINAL: done"])
    executor = MagicMock()
    workers = [WorkerAgent("w1", "Worker 1", executor)]
    sup = SupervisorAgent(llm=llm, workers=workers, max_rounds=3)
    assert sup._max_rounds == 3
    assert "w1" in sup._workers


def test_supervisor_default_system_prompt_contains_workers():
    from synapsekit.agents.multi.supervisor import SupervisorAgent, WorkerAgent

    llm = MockLLM(["FINAL: done"])
    executor = MagicMock()
    workers = [
        WorkerAgent("alice", "Does alice stuff", executor),
        WorkerAgent("bob", "Does bob stuff", executor),
    ]
    sup = SupervisorAgent(llm=llm, workers=workers)
    assert "alice" in sup._system_prompt
    assert "bob" in sup._system_prompt
    assert "Does alice stuff" in sup._system_prompt


def test_supervisor_custom_system_prompt():
    from synapsekit.agents.multi.supervisor import SupervisorAgent

    llm = MockLLM(["FINAL: done"])
    sup = SupervisorAgent(llm=llm, workers=[], system_prompt="Custom prompt")
    assert sup._system_prompt == "Custom prompt"


async def test_supervisor_run_final():
    from synapsekit.agents.multi.supervisor import SupervisorAgent

    llm = MockLLM(["FINAL: The answer is 42"])
    sup = SupervisorAgent(llm=llm, workers=[])
    result = await sup.run("What is the answer?")
    assert result == "The answer is 42"


async def test_supervisor_run_delegate_then_final():
    from synapsekit.agents.multi.supervisor import SupervisorAgent, WorkerAgent

    executor = MagicMock()
    executor.run = AsyncMock(return_value="Worker result: 42")

    llm = MockLLM(
        [
            "DELEGATE: calc | compute 6 * 7",
            "FINAL: The answer is 42",
        ]
    )
    workers = [WorkerAgent("calc", "Calculator", executor)]
    sup = SupervisorAgent(llm=llm, workers=workers)
    result = await sup.run("What is 6 * 7?")
    assert result == "The answer is 42"
    executor.run.assert_called_once_with("compute 6 * 7")


async def test_supervisor_run_unknown_worker():
    from synapsekit.agents.multi.supervisor import SupervisorAgent

    llm = MockLLM(
        [
            "DELEGATE: nonexistent | do something",
            "FINAL: fallback answer",
        ]
    )
    sup = SupervisorAgent(llm=llm, workers=[])
    result = await sup.run("test")
    assert result == "fallback answer"


async def test_supervisor_max_rounds():
    from synapsekit.agents.multi.supervisor import SupervisorAgent, WorkerAgent

    executor = MagicMock()
    executor.run = AsyncMock(return_value="still working")

    llm = MockLLM(["DELEGATE: w | keep going"])
    workers = [WorkerAgent("w", "Worker", executor)]
    sup = SupervisorAgent(llm=llm, workers=workers, max_rounds=2)
    result = await sup.run("loop forever")
    assert "Max rounds" in result


async def test_supervisor_run_plain_response():
    from synapsekit.agents.multi.supervisor import SupervisorAgent

    llm = MockLLM(["Just a plain response without prefix"])
    sup = SupervisorAgent(llm=llm, workers=[])
    result = await sup.run("test")
    assert result == "Just a plain response without prefix"


# ------------------------------------------------------------------ #
# Handoff
# ------------------------------------------------------------------ #


def test_handoff_import():
    from synapsekit.agents.multi.handoff import Handoff, HandoffChain, HandoffResult

    assert Handoff is not None
    assert HandoffChain is not None
    assert HandoffResult is not None


def test_handoff_top_level_import():
    from synapsekit import Handoff, HandoffChain, HandoffResult

    assert Handoff is not None
    assert HandoffChain is not None
    assert HandoffResult is not None


def test_handoff_construction():
    from synapsekit.agents.multi.handoff import Handoff

    h = Handoff(target="specialist")
    assert h.target == "specialist"
    assert h.condition("anything") is True
    assert h.transform("x") == "x"


def test_handoff_with_condition():
    from synapsekit.agents.multi.handoff import Handoff

    h = Handoff(target="billing", condition=lambda r: "bill" in r.lower())
    assert h.condition("I have a billing question") is True
    assert h.condition("technical issue") is False


def test_handoff_with_transform():
    from synapsekit.agents.multi.handoff import Handoff

    h = Handoff(target="spec", transform=lambda r: f"Previous: {r}")
    assert h.transform("hello") == "Previous: hello"


def test_handoff_result_construction():
    from synapsekit.agents.multi.handoff import HandoffResult

    r = HandoffResult(final_output="done", history=[{"agent": "a", "output": "done"}])
    assert r.final_output == "done"
    assert len(r.history) == 1


def test_handoff_chain_construction():
    from synapsekit.agents.multi.handoff import HandoffChain

    chain = HandoffChain(max_handoffs=5)
    assert chain._max_handoffs == 5
    assert chain._agents == {}


def test_handoff_chain_add_agent():
    from synapsekit.agents.multi.handoff import Handoff, HandoffChain

    chain = HandoffChain()
    executor = MagicMock()
    chain.add_agent("triage", executor, handoffs=[Handoff("billing")])
    assert "triage" in chain._agents


async def test_handoff_chain_run_with_handoff():
    from synapsekit.agents.multi.handoff import Handoff, HandoffChain

    triage_exec = MagicMock()
    triage_exec.run = AsyncMock(return_value="This is a billing issue")

    billing_exec = MagicMock()
    billing_exec.run = AsyncMock(return_value="Billing resolved")

    chain = HandoffChain()
    chain.add_agent(
        "triage",
        triage_exec,
        handoffs=[Handoff("billing", condition=lambda r: "billing" in r.lower())],
    )
    chain.add_agent("billing", billing_exec)

    result = await chain.run("triage", "I need help with my bill")
    assert result.final_output == "Billing resolved"
    assert len(result.history) == 2
    assert result.history[0]["agent"] == "triage"
    assert result.history[1]["agent"] == "billing"


async def test_handoff_chain_run_no_handoff():
    from synapsekit.agents.multi.handoff import HandoffChain

    executor = MagicMock()
    executor.run = AsyncMock(return_value="Done directly")

    chain = HandoffChain()
    chain.add_agent("agent", executor)

    result = await chain.run("agent", "simple question")
    assert result.final_output == "Done directly"
    assert len(result.history) == 1


async def test_handoff_chain_unknown_agent():
    from synapsekit.agents.multi.handoff import HandoffChain

    chain = HandoffChain()
    with pytest.raises(ValueError, match="Unknown agent"):
        await chain.run("nonexistent", "query")


async def test_handoff_chain_max_handoffs():
    from synapsekit.agents.multi.handoff import Handoff, HandoffChain

    # Two agents that always hand off to each other
    exec_a = MagicMock()
    exec_a.run = AsyncMock(return_value="go to b")
    exec_b = MagicMock()
    exec_b.run = AsyncMock(return_value="go to a")

    chain = HandoffChain(max_handoffs=4)
    chain.add_agent("a", exec_a, handoffs=[Handoff("b")])
    chain.add_agent("b", exec_b, handoffs=[Handoff("a")])

    result = await chain.run("a", "start")
    assert len(result.history) == 4


async def test_handoff_chain_transform():
    from synapsekit.agents.multi.handoff import Handoff, HandoffChain

    triage_exec = MagicMock()
    triage_exec.run = AsyncMock(return_value="needs specialist")

    spec_exec = MagicMock()
    spec_exec.run = AsyncMock(return_value="specialist done")

    chain = HandoffChain()
    chain.add_agent(
        "triage",
        triage_exec,
        handoffs=[
            Handoff(
                "specialist",
                condition=lambda r: "specialist" in r,
                transform=lambda r: f"Context: {r}",
            )
        ],
    )
    chain.add_agent("specialist", spec_exec)

    await chain.run("triage", "help")
    spec_exec.run.assert_called_once_with("Context: needs specialist")


# ------------------------------------------------------------------ #
# Crew
# ------------------------------------------------------------------ #


def test_crew_import():
    from synapsekit.agents.multi.crew import Crew, CrewAgent, CrewResult, Task

    assert Crew is not None
    assert CrewAgent is not None
    assert CrewResult is not None
    assert Task is not None


def test_crew_top_level_import():
    from synapsekit import Crew, CrewAgent, CrewResult, Task

    assert Crew is not None
    assert CrewAgent is not None
    assert CrewResult is not None
    assert Task is not None


def test_crew_agent_construction():
    from synapsekit.agents.multi.crew import CrewAgent

    llm = MockLLM(["resp"])
    agent = CrewAgent(name="researcher", role="Research Analyst", goal="Find info", llm=llm)
    assert agent.name == "researcher"
    assert agent.role == "Research Analyst"
    assert agent.goal == "Find info"
    assert agent.tools == []
    assert agent.backstory == ""


def test_crew_agent_with_backstory():
    from synapsekit.agents.multi.crew import CrewAgent

    llm = MockLLM(["resp"])
    agent = CrewAgent(
        name="writer",
        role="Writer",
        goal="Write well",
        llm=llm,
        backstory="Award-winning journalist",
    )
    assert agent.backstory == "Award-winning journalist"


def test_task_construction():
    from synapsekit.agents.multi.crew import Task

    task = Task(description="Do research", agent="researcher", expected_output="A report")
    assert task.description == "Do research"
    assert task.agent == "researcher"
    assert task.expected_output == "A report"
    assert task.context_from == []


def test_task_with_context():
    from synapsekit.agents.multi.crew import Task

    task = Task(description="Write", agent="writer", context_from=["researcher"])
    assert task.context_from == ["researcher"]


def test_crew_result_construction():
    from synapsekit.agents.multi.crew import CrewResult

    r = CrewResult(output="final", task_results={"a": "result_a"})
    assert r.output == "final"
    assert r.task_results["a"] == "result_a"


def test_crew_result_default():
    from synapsekit.agents.multi.crew import CrewResult

    r = CrewResult(output="done")
    assert r.task_results == {}


async def test_crew_sequential_execution():
    from synapsekit.agents.multi.crew import Crew, CrewAgent, Task

    llm = MockLLM(["Research output", "Written article"])
    researcher = CrewAgent("researcher", "Analyst", "Find info", llm)
    writer = CrewAgent("writer", "Writer", "Write content", llm)

    tasks = [
        Task("Research AI trends", agent="researcher"),
        Task("Write a blog post", agent="writer", context_from=["researcher"]),
    ]

    responses = iter(["Research output", "Written article"])

    crew = Crew(agents=[researcher, writer], tasks=tasks, process="sequential")
    with patch("synapsekit.agents.multi.crew.AgentExecutor") as mock_exec:
        mock_inst = AsyncMock()
        mock_inst.run = AsyncMock(side_effect=lambda q: next(responses))
        mock_exec.return_value = mock_inst
        result = await crew.run()
    assert result.output == "Written article"
    assert "researcher" in result.task_results
    assert "writer" in result.task_results


async def test_crew_parallel_execution():
    from synapsekit.agents.multi.crew import Crew, CrewAgent, Task

    llm = MockLLM(["Result A", "Result B", "Combined result"])
    agent_a = CrewAgent("a", "Agent A", "Do A", llm)
    agent_b = CrewAgent("b", "Agent B", "Do B", llm)
    agent_c = CrewAgent("c", "Agent C", "Combine", llm)

    tasks = [
        Task("Task A", agent="a"),
        Task("Task B", agent="b"),
        Task("Task C", agent="c", context_from=["a", "b"]),
    ]

    responses = iter(["Result A", "Result B", "Combined result"])

    crew = Crew(agents=[agent_a, agent_b, agent_c], tasks=tasks, process="parallel")
    with patch("synapsekit.agents.multi.crew.AgentExecutor") as mock_exec:
        mock_inst = AsyncMock()
        mock_inst.run = AsyncMock(side_effect=lambda q: next(responses))
        mock_exec.return_value = mock_inst
        result = await crew.run()
    assert result.output is not None
    assert len(result.task_results) == 3


async def test_crew_unknown_agent_error():
    from synapsekit.agents.multi.crew import Crew, CrewAgent, Task

    llm = MockLLM(["resp"])
    agent = CrewAgent("a", "Agent", "Goal", llm)
    tasks = [Task("Do something", agent="nonexistent")]
    crew = Crew(agents=[agent], tasks=tasks, process="sequential")

    with pytest.raises(ValueError, match="Unknown agent"):
        await crew.run()


async def test_crew_context_passing():
    from synapsekit.agents.multi.crew import Crew, CrewAgent, Task

    # Track prompts passed to executor.run
    prompts = []

    llm = MockLLM(["resp"])
    researcher = CrewAgent("researcher", "Analyst", "Find info", llm)
    writer = CrewAgent("writer", "Writer", "Write", llm)

    tasks = [
        Task("Research topic", agent="researcher"),
        Task("Write about it", agent="writer", context_from=["researcher"]),
    ]

    crew = Crew(agents=[researcher, writer], tasks=tasks, process="sequential")
    with patch("synapsekit.agents.multi.crew.AgentExecutor") as mock_exec:

        async def track_run(prompt):
            prompts.append(prompt)
            return "tracked response"

        mock_inst = AsyncMock()
        mock_inst.run = AsyncMock(side_effect=track_run)
        mock_exec.return_value = mock_inst
        await crew.run()

    # The writer's prompt should include context from researcher
    assert len(prompts) >= 2
    writer_prompt = prompts[-1]
    assert "Context from researcher" in writer_prompt


async def test_crew_empty_tasks():
    from synapsekit.agents.multi.crew import Crew

    MockLLM(["resp"])
    crew = Crew(agents=[], tasks=[], process="sequential")
    result = await crew.run()
    assert result.output == ""
    assert result.task_results == {}


async def test_crew_expected_output_in_prompt():
    from synapsekit.agents.multi.crew import Crew, CrewAgent, Task

    prompts = []

    llm = MockLLM(["resp"])
    agent = CrewAgent("a", "Agent", "Goal", llm)
    tasks = [Task("Do task", agent="a", expected_output="A JSON object")]

    crew = Crew(agents=[agent], tasks=tasks)
    with patch("synapsekit.agents.multi.crew.AgentExecutor") as mock_exec:

        async def track_run(prompt):
            prompts.append(prompt)
            return "response"

        mock_inst = AsyncMock()
        mock_inst.run = AsyncMock(side_effect=track_run)
        mock_exec.return_value = mock_inst
        await crew.run()

    assert "Expected output format: A JSON object" in prompts[0]


# ------------------------------------------------------------------ #
# Integration: all __all__ exports
# ------------------------------------------------------------------ #


def test_all_new_exports_in_init():
    import synapsekit

    new_exports = [
        "MCPClient",
        "MCPServer",
        "MCPToolAdapter",
        "SupervisorAgent",
        "WorkerAgent",
        "Crew",
        "CrewAgent",
        "CrewResult",
        "Task",
        "Handoff",
        "HandoffChain",
        "HandoffResult",
    ]
    for name in new_exports:
        assert name in synapsekit.__all__, f"{name} not in __all__"
        assert hasattr(synapsekit, name), f"{name} not accessible on module"
