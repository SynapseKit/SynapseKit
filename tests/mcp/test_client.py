from __future__ import annotations

from typing import Any

import pytest

from synapsekit.mcp.client import MCPClient

_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}


class _Fake:
    """Stand-in for an ``mcp.types`` model."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeSession:
    """Stand-in for ``mcp.ClientSession``."""

    def __init__(self, schema_attr: str, error_attr: str) -> None:
        self.schema_attr = schema_attr
        self.error_attr = error_attr

    async def list_tools(self) -> Any:
        tool = _Fake(name="add", description="Add two integers", **{self.schema_attr: _SCHEMA})
        return _Fake(tools=[tool])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not isinstance(arguments["b"], int):
            text, error = "b must be an integer", True
        else:
            text, error = str(arguments["a"] + arguments["b"]), False
        content = [_Fake(type="text", text=text)]
        return _Fake(content=content, **{self.error_attr: error})


@pytest.mark.parametrize(
    ("schema_attr", "error_attr"),
    [("inputSchema", "isError"), ("input_schema", "is_error")],
    ids=["mcp1", "mcp2"],
)
async def test_mcp_client_reads_schema_and_results(schema_attr, error_attr):
    client = MCPClient()
    client._session = _FakeSession(schema_attr, error_attr)
    (tool,) = await client._load_tools()
    assert tool.parameters == _SCHEMA

    result = await tool.run(a=2, b=3)
    assert not result.is_error, result.error
    assert result.output == "5"

    result = await tool.run(a=2, b="x")
    assert result.is_error
    assert result.error == "b must be an integer"
