import json
from unittest.mock import MagicMock, patch

import pytest

try:
    from synapsekit.llm._cache_dynamodb import BOTO3_AVAILABLE, DynamoDBCacheBackend
except ImportError:
    BOTO3_AVAILABLE = False


@pytest.fixture
def mock_boto3_resource():
    with patch("boto3.resource") as mock_resource:
        mock_table = MagicMock()
        mock_resource.return_value.Table.return_value = mock_table
        yield mock_table


@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
def test_dynamodb_cache_init(mock_boto3_resource):
    cache = DynamoDBCacheBackend(table_name="test-table", region_name="us-east-1", ttl_seconds=3600)
    assert cache.table_name == "test-table"
    assert cache.partition_key == "cache_key"
    assert cache.ttl_seconds == 3600


@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
def test_dynamodb_cache_get_hit(mock_boto3_resource):
    cache = DynamoDBCacheBackend(table_name="test-table")
    mock_boto3_resource.get_item.return_value = {
        "Item": {"cache_key": "test_key", "value": json.dumps({"result": "cached_data"})}
    }

    result = cache.get("test_key")
    assert result == {"result": "cached_data"}
    assert cache.hits == 1
    assert cache.misses == 0
    mock_boto3_resource.get_item.assert_called_once_with(Key={"cache_key": "test_key"})


@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
def test_dynamodb_cache_get_miss(mock_boto3_resource):
    cache = DynamoDBCacheBackend(table_name="test-table")
    mock_boto3_resource.get_item.return_value = {}

    result = cache.get("test_key")
    assert result is None
    assert cache.hits == 0
    assert cache.misses == 1


@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
def test_dynamodb_cache_put(mock_boto3_resource):
    cache = DynamoDBCacheBackend(table_name="test-table")
    cache.put("test_key", {"result": "new_data"})

    mock_boto3_resource.put_item.assert_called_once()
    args = mock_boto3_resource.put_item.call_args[1]
    assert args["Item"]["cache_key"] == "test_key"
    assert json.loads(args["Item"]["value"]) == {"result": "new_data"}
    assert "ttl" not in args["Item"]


@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
@patch("time.time", return_value=100000)
def test_dynamodb_cache_put_with_ttl(mock_time, mock_boto3_resource):
    cache = DynamoDBCacheBackend(table_name="test-table", ttl_seconds=3600)
    cache.put("test_key", {"result": "new_data"})

    args = mock_boto3_resource.put_item.call_args[1]
    assert args["Item"]["ttl"] == 100000 + 3600


@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
def test_dynamodb_cache_len(mock_boto3_resource):
    cache = DynamoDBCacheBackend(table_name="test-table")
    mock_boto3_resource.item_count = 42

    assert len(cache) == 42
    mock_boto3_resource.reload.assert_called_once()
