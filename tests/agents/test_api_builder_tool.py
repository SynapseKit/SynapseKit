from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.agents.tools.api_builder import APIBuilderTool


class TestAPIBuilderTool:
    @pytest.mark.asyncio
    async def test_selects_operation_from_intent(self):
        spec = {
            "openapi": "3.0.0",
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
                }
            },
        }

        tool = APIBuilderTool()
        tool._perform_request = AsyncMock(return_value="HTTP 200\n[]")
        result = await tool.run(intent="list all users", openapi_spec=spec)

        assert not result.is_error
        assert "Selected operation: listUsers" in result.output
        assert "Request: GET /users" in result.output
        tool._perform_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_uses_llm_when_available(self):
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/users": {
                    "get": {"operationId": "listUsers", "summary": "List users"},
                    "post": {"operationId": "createUser", "summary": "Create user"},
                }
            },
        }

        llm = MagicMock()
        llm.generate.return_value = "createUser"
        tool = APIBuilderTool(llm=llm)
        tool._perform_request = AsyncMock(return_value="HTTP 200\n{}")

        result = await tool.run(intent="add a new user", openapi_spec=spec)

        assert not result.is_error
        assert "Selected operation: createUser" in result.output
        llm.generate.assert_called_once()

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
