import base64

import pytest

from synapsekit.agents.tools.base64_tool import Base64Tool


@pytest.mark.asyncio
async def test_base64_tool_encode():
    tool = Base64Tool()

    # Test encoding
    result = await tool.run(action="encode", text="Hello SynapseKit")
    assert not result.error
    assert result.output == base64.b64encode(b"Hello SynapseKit").decode("utf-8")

    # Test kwargs
    result_kwargs = await tool.run(input="encode", value="test string")
    assert not result_kwargs.error
    assert result_kwargs.output == base64.b64encode(b"test string").decode("utf-8")


@pytest.mark.asyncio
async def test_base64_tool_decode():
    tool = Base64Tool()

    # Test decoding
    encoded_text = base64.b64encode(b"Hello SynapseKit").decode("utf-8")
    result = await tool.run(action="decode", text=encoded_text)
    assert not result.error
    assert result.output == "Hello SynapseKit"


@pytest.mark.asyncio
async def test_base64_tool_invalid_action():
    tool = Base64Tool()

    result = await tool.run(action="hash", text="test")
    assert result.error
    assert "Unknown action" in result.error


@pytest.mark.asyncio
async def test_base64_tool_invalid_base64():
    tool = Base64Tool()

    # 'test' is not valid base64
    result = await tool.run(action="decode", text="!@#$%")
    assert result.error
    assert "Base64 error" in result.error
