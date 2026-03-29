from __future__ import annotations

import pytest

from synapsekit.agents.tools.shell import ShellTool


class TestShellTool:
    @pytest.mark.asyncio
    async def test_echo(self):
        tool = ShellTool()
        result = await tool.run(command="echo hello")
        assert not result.is_error
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_no_command(self):
        tool = ShellTool()
        result = await tool.run()
        assert result.is_error
        assert "No command" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self):
        tool = ShellTool(timeout=1)
        result = await tool.run(command='python -c "import time; time.sleep(10)"')
        assert result.is_error
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_allowed_commands_pass(self):
        tool = ShellTool(allowed_commands=["echo", "ls"])
        result = await tool.run(command="echo allowed")
        assert not result.is_error
        assert "allowed" in result.output

    @pytest.mark.asyncio
    async def test_allowed_commands_block(self):
        tool = ShellTool(allowed_commands=["echo"])
        result = await tool.run(command="ls /tmp")
        assert result.is_error
        assert "not in the allowed list" in result.error

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self):
        tool = ShellTool()
        result = await tool.run(command="false")
        assert result.is_error
        assert "Exit code" in result.error

    def test_schema(self):
        tool = ShellTool()
        schema = tool.schema()
        assert schema["function"]["name"] == "shell"
        assert "command" in schema["function"]["parameters"]["properties"]
