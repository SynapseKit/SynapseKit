from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.agents.tools.api_builder import APIBuilderTool

SAMPLE_SPEC = {
    "openapi": "3.0.0",
    "servers": [{"url": "https://api.example.com"}],
    "paths": {
        "/users": {
            "get": {
                "operationId": "listUsers",
                "summary": "List users",
            },
            "post": {
                "operationId": "createUser",
                "summary": "Create user",
            },
        },
        "/users/{id}": {
            "get": {
                "operationId": "getUser",
                "summary": "Get a user by ID",
            },
            "delete": {
                "operationId": "deleteUser",
                "summary": "Delete a user",
            },
        },
    },
}


class TestAPIBuilderTool:
    # ------------------------------------------------------------------
    # Basic operation selection
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_selects_operation_from_intent(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 200\n[]")
        result = await tool.run(intent="list all users", openapi_spec=SAMPLE_SPEC)

        assert not result.is_error
        assert "Selected operation: listUsers" in result.output
        assert "GET" in result.output and "/users" in result.output
        tool._perform_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_selects_post_operation_for_create_intent(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 201\n{}")
        result = await tool.run(intent="create a new user", openapi_spec=SAMPLE_SPEC)

        assert not result.is_error
        assert "Selected operation: createUser" in result.output
        assert "POST" in result.output

    @pytest.mark.asyncio
    async def test_selects_delete_operation(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 200\n{}")
        result = await tool.run(intent="delete user", openapi_spec=SAMPLE_SPEC)

        assert not result.is_error
        assert "deleteUser" in result.output

    # ------------------------------------------------------------------
    # LLM-assisted selection
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_uses_llm_when_available(self):
        llm = MagicMock()
        llm.generate.return_value = "createUser"
        tool = APIBuilderTool(llm=llm)
        tool._perform_request = AsyncMock(return_value="HTTP 200\n{}")

        result = await tool.run(intent="add a new user", openapi_spec=SAMPLE_SPEC)

        assert not result.is_error
        assert "Selected operation: createUser" in result.output
        llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_fallback_on_invalid_response(self):
        """When LLM returns an invalid operationId, fall back to token scoring."""
        llm = MagicMock()
        llm.generate.return_value = "nonExistentOperation"
        tool = APIBuilderTool(llm=llm)
        tool._perform_request = AsyncMock(return_value="HTTP 200\n[]")

        result = await tool.run(intent="list all users", openapi_spec=SAMPLE_SPEC)

        assert not result.is_error
        assert "Selected operation:" in result.output

    @pytest.mark.asyncio
    async def test_llm_fallback_on_exception(self):
        """When LLM raises an exception, fall back to token scoring."""
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM unavailable")
        tool = APIBuilderTool(llm=llm)
        tool._perform_request = AsyncMock(return_value="HTTP 200\n[]")

        result = await tool.run(intent="list users", openapi_spec=SAMPLE_SPEC)

        assert not result.is_error
        assert "Selected operation:" in result.output

    @pytest.mark.asyncio
    async def test_llm_async_generate(self):
        """When LLM.generate returns an awaitable, it should be awaited."""
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="getUser")
        tool = APIBuilderTool(llm=llm)
        tool._perform_request = AsyncMock(return_value="HTTP 200\n{}")

        result = await tool.run(intent="get user details", openapi_spec=SAMPLE_SPEC)

        assert not result.is_error
        assert "getUser" in result.output

    # ------------------------------------------------------------------
    # Explicit path / method / operation_id
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_builds_request_with_path_and_query_params(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 200\n{}")
        result = await tool.run(
            intent="fetch the user by id",
            path="/users/{id}",
            method="get",
            server_url="https://api.example.com",
            path_params={"id": 42},
            query_params={"expand": "profile"},
            headers={"X-Test": "1"},
        )

        assert not result.is_error
        assert "https://api.example.com/users/42?expand=profile" in result.output
        call = tool._perform_request.await_args
        assert call.args[0] == "https://api.example.com/users/42?expand=profile"
        assert call.args[1] == "GET"
        assert call.args[2]["X-Test"] == "1"

    @pytest.mark.asyncio
    async def test_explicit_operation_id(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 200\n{}")
        result = await tool.run(
            intent="get user",
            openapi_spec=SAMPLE_SPEC,
            operation_id="deleteUser",
        )

        assert not result.is_error
        assert "deleteUser" in result.output
        assert "DELETE" in result.output

    @pytest.mark.asyncio
    async def test_explicit_path_without_spec(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 200\nOK")
        result = await tool.run(
            intent="health check",
            path="/health",
            method="GET",
            server_url="https://api.example.com",
        )

        assert not result.is_error
        assert "https://api.example.com/health" in result.output

    @pytest.mark.asyncio
    async def test_server_url_from_spec(self):
        """When no server_url is given, use the first server from spec."""
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 200\n[]")
        result = await tool.run(intent="list users", openapi_spec=SAMPLE_SPEC)

        assert not result.is_error
        call = tool._perform_request.await_args
        assert call.args[0].startswith("https://api.example.com/")

    # ------------------------------------------------------------------
    # Request body handling
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_post_with_dict_body(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 201\n{}")
        body = {"name": "Alice", "email": "alice@example.com"}
        result = await tool.run(
            intent="create user",
            path="/users",
            method="POST",
            server_url="https://api.example.com",
            body=body,
        )

        assert not result.is_error
        call = tool._perform_request.await_args
        assert call.args[3] == json.dumps(body).encode("utf-8")
        assert call.args[2]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_post_with_json_string_body(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 201\n{}")
        body_str = '{"name": "Bob"}'
        result = await tool.run(
            intent="create user",
            path="/users",
            method="POST",
            server_url="https://api.example.com",
            body=body_str,
        )

        assert not result.is_error
        call = tool._perform_request.await_args
        assert call.args[3] == body_str.encode("utf-8")
        assert call.args[2]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_post_with_plain_text_body(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 200\nOK")
        result = await tool.run(
            intent="send message",
            path="/messages",
            method="POST",
            server_url="https://api.example.com",
            body="Hello world",
        )

        assert not result.is_error
        call = tool._perform_request.await_args
        assert call.args[3] == b"Hello world"
        assert "Content-Type" not in call.args[2]

    @pytest.mark.asyncio
    async def test_get_request_ignores_body(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 200\n[]")
        result = await tool.run(
            intent="list users",
            path="/users",
            method="GET",
            server_url="https://api.example.com",
            body={"ignored": True},
        )

        assert not result.is_error
        call = tool._perform_request.await_args
        assert call.args[3] is None

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_intent_returns_error(self):
        tool = APIBuilderTool()
        result = await tool.run()

        assert result.is_error
        assert "No intent" in result.error

    @pytest.mark.asyncio
    async def test_no_spec_no_path_returns_error(self):
        tool = APIBuilderTool()
        result = await tool.run(intent="do something")

        assert result.is_error
        assert "Could not determine" in result.error

    @pytest.mark.asyncio
    async def test_empty_spec_paths_returns_error(self):
        tool = APIBuilderTool()
        result = await tool.run(
            intent="list items",
            openapi_spec={"openapi": "3.0.0", "paths": {}},
        )

        assert result.is_error
        assert "Could not determine" in result.error

    @pytest.mark.asyncio
    async def test_request_exception_returns_error(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(side_effect=ConnectionError("timeout"))
        result = await tool.run(
            intent="list users",
            path="/users",
            server_url="https://api.example.com",
        )

        assert result.is_error
        assert "API builder failed" in result.error

    # ------------------------------------------------------------------
    # Spec loading
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_loads_spec_from_json_string(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 200\n[]")
        spec_str = json.dumps(SAMPLE_SPEC)
        result = await tool.run(intent="list users", openapi_spec=spec_str)

        assert not result.is_error
        assert "listUsers" in result.output

    @pytest.mark.asyncio
    async def test_loads_spec_from_url(self):
        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 200\n[]")

        spec_bytes = json.dumps(SAMPLE_SPEC).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = spec_bytes
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch(
            "synapsekit.agents.tools.api_builder.urlopen",
            return_value=mock_response,
        ):
            result = await tool.run(
                intent="list users",
                openapi_url="https://example.com/openapi.json",
            )

        assert not result.is_error
        assert "listUsers" in result.output

    # ------------------------------------------------------------------
    # Schema and metadata
    # ------------------------------------------------------------------

    def test_tool_name_and_description(self):
        tool = APIBuilderTool()
        assert tool.name == "api_request_builder"
        assert "API" in tool.description

    def test_schema_returns_valid_openai_format(self):
        tool = APIBuilderTool()
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "api_request_builder"
        assert "intent" in schema["function"]["parameters"]["properties"]

    def test_anthropic_schema_returns_valid_format(self):
        tool = APIBuilderTool()
        schema = tool.anthropic_schema()
        assert schema["name"] == "api_request_builder"
        assert "intent" in schema["input_schema"]["properties"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def test_tokenize(self):
        tool = APIBuilderTool()
        tokens = tool._tokenize("List all Users by ID")
        assert tokens == {"list", "all", "users", "by", "id"}

    def test_iter_operations_skips_non_http_keys(self):
        spec = {
            "paths": {
                "/test": {
                    "get": {"operationId": "getTest"},
                    "parameters": [{"name": "id"}],
                    "x-custom": "ignored",
                }
            }
        }
        tool = APIBuilderTool()
        ops = list(tool._iter_operations(spec))
        assert len(ops) == 1
        assert ops[0]["operationId"] == "getTest"

    def test_score_operation_prefers_matching_method(self):
        tool = APIBuilderTool()
        tokens = tool._tokenize("create new item")
        post_op = {"method": "POST", "operationId": "createItem", "path": "/items"}
        get_op = {"method": "GET", "operationId": "listItems", "path": "/items"}
        assert tool._score_operation(tokens, post_op) > tool._score_operation(tokens, get_op)
