from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.tools.aws_lambda import AWSLambdaTool


class TestAWSLambdaTool:
    @pytest.mark.asyncio
    async def test_invoke_lambda_success(self):
        payload_stream = MagicMock()
        payload_stream.read.return_value = b'{"ok": true}'
        client = MagicMock()
        client.invoke.return_value = {
            "StatusCode": 200,
            "ExecutedVersion": "$LATEST",
            "Payload": payload_stream,
        }
        boto3_mod = MagicMock()
        boto3_mod.client.return_value = client

        with patch.dict("sys.modules", {"boto3": boto3_mod}):
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
        boto3_mod.client.assert_called_once_with("lambda", region_name="ap-south-1")
        client.invoke.assert_called_once()
        invoke_kwargs = client.invoke.call_args.kwargs
        assert invoke_kwargs["FunctionName"] == "demo-fn"
        assert invoke_kwargs["InvocationType"] == "RequestResponse"
        assert b'"hello": "world"' in invoke_kwargs["Payload"]

    @pytest.mark.asyncio
    async def test_missing_function_name(self):
        tool = AWSLambdaTool()
        result = await tool.run(function_name="")
        assert result.is_error
        assert "function_name" in result.error.lower()
