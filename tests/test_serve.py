from unittest.mock import patch

from synapsekit import ReActAgent
from synapsekit.serve import patch_agents, serve_agent


class MockAgent:
    async def run(self, query: str):
        return f"Response to: {query}"

    async def stream(self, query: str):
        yield "Stream "
        yield "Response"


def test_patch_agents():
    # Ensure ReActAgent doesn't have serve patched incorrectly or it gets patched
    patch_agents()
    assert hasattr(ReActAgent, "serve")


@patch("uvicorn.run")
def test_serve_agent_calls_uvicorn(mock_uvicorn_run):
    agent = MockAgent()
    serve_agent(agent, host="127.0.0.1", port=9000, path="/api/chat")

    mock_uvicorn_run.assert_called_once()
    _args, kwargs = mock_uvicorn_run.call_args
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9000
