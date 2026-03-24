from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.tools.aws_lambda import AWSLambdaTool


def _mock_boto3(client: MagicMock) -> dict[str, MagicMock]:
    boto3_mod = MagicMock()
    boto3_mod.client.return_value = client
    return {"boto3": boto3_mod}


def _make_lambda_response(
    payload: bytes = b"",
    status: int = 200,
    executed_version: str | None = None,
    function_error: str | None = None,
) -> dict:
    stream = MagicMock()
    stream.read.return_value = payload
    resp: dict = {"StatusCode": status, "Payload": stream}
    if executed_version:
        resp["ExecutedVersion"] = executed_version
    if function_error:
        resp["FunctionError"] = function_error
    return resp


class TestAWSLambdaTool:
    # ------------------------------------------------------------------
    # Successful invocations
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_invoke_lambda_success(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(
            payload=b'{"ok": true}', executed_version="$LATEST"
        )
        boto3_mods = _mock_boto3(client)

        with patch.dict("sys.modules", boto3_mods):
            tool = AWSLambdaTool(region_name="ap-south-1")
            result = await tool.run(
                function_name="demo-fn",
                payload={"hello": "world"},
                invocation_type="RequestResponse",
            )

        assert not result.is_error
        assert "StatusCode: 200" in result.output
        assert "ExecutedVersion: $LATEST" in result.output
        assert '"ok": true' in result.output
        boto3_mods["boto3"].client.assert_called_once_with("lambda", region_name="ap-south-1")
        invoke_kwargs = client.invoke.call_args.kwargs
        assert invoke_kwargs["FunctionName"] == "demo-fn"
        assert invoke_kwargs["InvocationType"] == "RequestResponse"

    @pytest.mark.asyncio
    async def test_invoke_with_qualifier(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(payload=b'"ok"')

        with patch.dict("sys.modules", _mock_boto3(client)):
            tool = AWSLambdaTool(region_name="us-east-1")
            result = await tool.run(function_name="my-fn", qualifier="v2")

        assert not result.is_error
        assert client.invoke.call_args.kwargs["Qualifier"] == "v2"

    @pytest.mark.asyncio
    async def test_invoke_event_type(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(status=202, payload=b"")

        with patch.dict("sys.modules", _mock_boto3(client)):
            tool = AWSLambdaTool(region_name="us-east-1")
            result = await tool.run(function_name="async-fn", invocation_type="Event")

        assert not result.is_error
        assert "StatusCode: 202" in result.output
        assert client.invoke.call_args.kwargs["InvocationType"] == "Event"

    @pytest.mark.asyncio
    async def test_invoke_dry_run(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(status=204, payload=b"")

        with patch.dict("sys.modules", _mock_boto3(client)):
            tool = AWSLambdaTool(region_name="us-east-1")
            result = await tool.run(function_name="my-fn", invocation_type="DryRun")

        assert not result.is_error
        assert "StatusCode: 204" in result.output

    @pytest.mark.asyncio
    async def test_invoke_no_payload(self):
        """Invoking without payload should omit the Payload key."""
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(payload=b'"done"')

        with patch.dict("sys.modules", _mock_boto3(client)):
            tool = AWSLambdaTool(region_name="us-east-1")
            result = await tool.run(function_name="no-payload-fn")

        assert not result.is_error
        assert "Payload" not in client.invoke.call_args.kwargs

    # ------------------------------------------------------------------
    # Region resolution
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_region_from_run_kwarg(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(payload=b'""')
        boto3_mods = _mock_boto3(client)

        with patch.dict("sys.modules", boto3_mods):
            tool = AWSLambdaTool(region_name="us-west-2")
            await tool.run(function_name="fn", region_name="eu-west-1")

        boto3_mods["boto3"].client.assert_called_once_with("lambda", region_name="eu-west-1")

    @pytest.mark.asyncio
    async def test_region_from_constructor(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(payload=b'""')
        boto3_mods = _mock_boto3(client)

        with patch.dict("sys.modules", boto3_mods):
            tool = AWSLambdaTool(region_name="ap-south-1")
            await tool.run(function_name="fn")

        boto3_mods["boto3"].client.assert_called_once_with("lambda", region_name="ap-south-1")

    @pytest.mark.asyncio
    async def test_region_from_env(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(payload=b'""')
        boto3_mods = _mock_boto3(client)

        with (
            patch.dict("sys.modules", boto3_mods),
            patch.dict("os.environ", {"AWS_REGION": "ca-central-1"}, clear=False),
        ):
            tool = AWSLambdaTool()
            await tool.run(function_name="fn")

        boto3_mods["boto3"].client.assert_called_once_with("lambda", region_name="ca-central-1")

    @pytest.mark.asyncio
    async def test_no_region_falls_back_to_default(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(payload=b'""')
        boto3_mods = _mock_boto3(client)

        with (
            patch.dict("sys.modules", boto3_mods),
            patch.dict("os.environ", {"AWS_REGION": "", "AWS_DEFAULT_REGION": ""}, clear=False),
        ):
            tool = AWSLambdaTool()
            await tool.run(function_name="fn")

        boto3_mods["boto3"].client.assert_called_once_with("lambda")

    # ------------------------------------------------------------------
    # Payload serialization
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_payload_dict(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(payload=b'""')

        with patch.dict("sys.modules", _mock_boto3(client)):
            tool = AWSLambdaTool(region_name="us-east-1")
            await tool.run(function_name="fn", payload={"key": "value"})

        raw = client.invoke.call_args.kwargs["Payload"]
        assert json.loads(raw) == {"key": "value"}

    @pytest.mark.asyncio
    async def test_payload_json_string(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(payload=b'""')

        with patch.dict("sys.modules", _mock_boto3(client)):
            tool = AWSLambdaTool(region_name="us-east-1")
            await tool.run(function_name="fn", payload='{"key": "value"}')

        raw = client.invoke.call_args.kwargs["Payload"]
        assert json.loads(raw) == {"key": "value"}

    @pytest.mark.asyncio
    async def test_payload_plain_string(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(payload=b'""')

        with patch.dict("sys.modules", _mock_boto3(client)):
            tool = AWSLambdaTool(region_name="us-east-1")
            await tool.run(function_name="fn", payload="hello")

        raw = client.invoke.call_args.kwargs["Payload"]
        assert raw == b"hello"

    @pytest.mark.asyncio
    async def test_payload_list(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(payload=b'""')

        with patch.dict("sys.modules", _mock_boto3(client)):
            tool = AWSLambdaTool(region_name="us-east-1")
            await tool.run(function_name="fn", payload=[1, 2, 3])

        raw = client.invoke.call_args.kwargs["Payload"]
        assert json.loads(raw) == [1, 2, 3]

    # ------------------------------------------------------------------
    # Response formatting
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_function_error_in_response(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(
            payload=b'{"errorMessage": "boom"}',
            status=200,
            function_error="Unhandled",
        )

        with patch.dict("sys.modules", _mock_boto3(client)):
            tool = AWSLambdaTool(region_name="us-east-1")
            result = await tool.run(function_name="failing-fn")

        assert not result.is_error
        assert "FunctionError: Unhandled" in result.output
        assert "boom" in result.output

    @pytest.mark.asyncio
    async def test_empty_payload_response(self):
        client = MagicMock()
        client.invoke.return_value = _make_lambda_response(payload=b"")

        with patch.dict("sys.modules", _mock_boto3(client)):
            tool = AWSLambdaTool(region_name="us-east-1")
            result = await tool.run(function_name="fn")

        assert "(empty)" in result.output

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_missing_function_name(self):
        tool = AWSLambdaTool()
        result = await tool.run(function_name="")

        assert result.is_error
        assert "function_name" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_function_name_at_all(self):
        tool = AWSLambdaTool()
        result = await tool.run()

        assert result.is_error
        assert "function_name" in result.error.lower()

    @pytest.mark.asyncio
    async def test_boto3_exception_returns_error(self):
        client = MagicMock()
        client.invoke.side_effect = Exception("AccessDenied")

        with patch.dict("sys.modules", _mock_boto3(client)):
            tool = AWSLambdaTool(region_name="us-east-1")
            result = await tool.run(function_name="locked-fn")

        assert result.is_error
        assert "AWS Lambda invocation failed" in result.error
        assert "AccessDenied" in result.error

    # ------------------------------------------------------------------
    # Schema and metadata
    # ------------------------------------------------------------------

    def test_tool_name_and_description(self):
        tool = AWSLambdaTool()
        assert tool.name == "aws_lambda"
        assert "Lambda" in tool.description

    def test_schema_returns_valid_openai_format(self):
        tool = AWSLambdaTool()
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "aws_lambda"
        assert "function_name" in schema["function"]["parameters"]["properties"]
        assert "function_name" in schema["function"]["parameters"]["required"]

    def test_anthropic_schema_returns_valid_format(self):
        tool = AWSLambdaTool()
        schema = tool.anthropic_schema()
        assert schema["name"] == "aws_lambda"
        assert "function_name" in schema["input_schema"]["properties"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def test_serialize_payload_bytes_passthrough(self):
        tool = AWSLambdaTool()
        assert tool._serialize_payload(b"raw") == b"raw"

    def test_serialize_payload_empty_string(self):
        tool = AWSLambdaTool()
        assert tool._serialize_payload("") == b""
