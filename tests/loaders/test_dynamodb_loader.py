from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import Document
from synapsekit.loaders.dynamodb import DynamoDBLoader


def test_init_requires_table_name() -> None:
    with pytest.raises(ValueError, match="table_name must be provided"):
        DynamoDBLoader(table_name="")


def test_init_rejects_invalid_operation() -> None:
    with pytest.raises(ValueError, match="operation must be 'scan' or 'query'"):
        DynamoDBLoader(table_name="t", operation="nope")  # type: ignore[arg-type]


def test_load_missing_boto3_import_error() -> None:
    with patch.dict("sys.modules", {"boto3": None}):
        loader = DynamoDBLoader(table_name="tbl")
        with pytest.raises(ImportError, match=r"synapsekit\[dynamodb\]"):
            loader.load()


@patch.dict("sys.modules", {"boto3": MagicMock()})
def test_scan_loads_items_and_builds_documents() -> None:
    import sys

    mock_boto3 = sys.modules["boto3"]

    mock_table = MagicMock()
    mock_table.scan.return_value = {
        "Items": [
            {"pk": "1", "title": "Hello", "count": Decimal("2")},
            {"pk": "2", "title": "World", "count": Decimal("2.5")},
        ]
    }

    mock_resource = MagicMock()
    mock_resource.Table.return_value = mock_table
    mock_boto3.resource.return_value = mock_resource

    loader = DynamoDBLoader(table_name="my-table", text_attributes=["title"])
    docs = loader.load()

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)

    assert docs[0].text == "Hello"
    assert docs[0].metadata["source"] == "dynamodb"
    assert docs[0].metadata["table"] == "my-table"
    assert docs[0].metadata["operation"] == "scan"
    assert docs[0].metadata["row"] == 0

    # Decimal normalization
    assert docs[0].metadata["count"] == 2
    assert docs[1].metadata["count"] == 2.5

    mock_boto3.resource.assert_called_once_with("dynamodb", region_name="us-east-1")
    mock_resource.Table.assert_called_once_with("my-table")
    mock_table.scan.assert_called_once()


@patch.dict("sys.modules", {"boto3": MagicMock()})
def test_scan_paginates_with_last_evaluated_key() -> None:
    import sys

    mock_boto3 = sys.modules["boto3"]

    mock_table = MagicMock()
    mock_table.scan.side_effect = [
        {"Items": [{"pk": "1", "text": "a"}], "LastEvaluatedKey": {"pk": "1"}},
        {"Items": [{"pk": "2", "text": "b"}]},
    ]

    mock_resource = MagicMock()
    mock_resource.Table.return_value = mock_table
    mock_boto3.resource.return_value = mock_resource

    loader = DynamoDBLoader(table_name="tbl", scan_kwargs={"Limit": 1}, text_attributes=["text"])
    docs = loader.load()

    assert [d.text for d in docs] == ["a", "b"]

    first_call_kwargs = mock_table.scan.call_args_list[0].kwargs
    second_call_kwargs = mock_table.scan.call_args_list[1].kwargs
    assert first_call_kwargs == {"Limit": 1}
    assert second_call_kwargs == {"Limit": 1, "ExclusiveStartKey": {"pk": "1"}}


@patch.dict("sys.modules", {"boto3": MagicMock()})
def test_query_requires_query_kwargs() -> None:
    import sys

    mock_boto3 = sys.modules["boto3"]
    mock_resource = MagicMock()
    mock_resource.Table.return_value = MagicMock()
    mock_boto3.resource.return_value = mock_resource

    loader = DynamoDBLoader(table_name="tbl", operation="query")
    with pytest.raises(ValueError, match="query_kwargs must be provided"):
        loader.load()


@patch.dict("sys.modules", {"boto3": MagicMock()})
def test_query_uses_table_query_and_respects_text_and_metadata_fields() -> None:
    import sys

    mock_boto3 = sys.modules["boto3"]

    mock_table = MagicMock()
    mock_table.query.return_value = {
        "Items": [
            {
                "pk": "u#1",
                "sk": "doc#1",
                "title": "T",
                "body": {"k": [1, 2]},
                "extra": "ignore",
            }
        ]
    }

    mock_resource = MagicMock()
    mock_resource.Table.return_value = mock_table
    mock_boto3.resource.return_value = mock_resource

    loader = DynamoDBLoader(
        table_name="tbl",
        operation="query",
        query_kwargs={"KeyConditionExpression": "pk = :pk"},
        text_attributes=["title", "body", "missing"],
        metadata_attributes=["pk", "sk"],
    )

    docs = loader.load()
    assert len(docs) == 1

    assert docs[0].text == 'T\n{"k": [1, 2]}\n'
    assert docs[0].metadata["pk"] == "u#1"
    assert docs[0].metadata["sk"] == "doc#1"
    assert "extra" not in docs[0].metadata

    mock_table.query.assert_called_once_with(KeyConditionExpression="pk = :pk")


@patch.dict("sys.modules", {"boto3": MagicMock()})
def test_aload_runs() -> None:
    import sys

    mock_boto3 = sys.modules["boto3"]

    mock_table = MagicMock()
    mock_table.scan.return_value = {"Items": [{"pk": "1", "text": "hi"}]}

    mock_resource = MagicMock()
    mock_resource.Table.return_value = mock_table
    mock_boto3.resource.return_value = mock_resource

    loader = DynamoDBLoader(table_name="tbl", text_attributes=["text"])
    docs = asyncio.run(loader.aload())

    assert len(docs) == 1
    assert docs[0].text == "hi"
