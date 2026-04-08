from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders.supabase import SupabaseLoader


def test_import_error_missing_supabase():
    with patch.dict("sys.modules", {"supabase": None}):
        loader = SupabaseLoader("users", supabase_url="http://url", supabase_key="key")
        with pytest.raises(ImportError, match="supabase required"):
            loader.load()


def test_init_missing_credentials():
    with patch.dict("os.environ", clear=True):
        with pytest.raises(ValueError, match="supabase_url and supabase_key are required"):
            SupabaseLoader("users")


def test_init_credentials_from_env():
    with patch.dict("os.environ", {"SUPABASE_URL": "env_url", "SUPABASE_KEY": "env_key"}):
        loader = SupabaseLoader("users")
        assert loader._supabase_url == "env_url"
        assert loader._supabase_key == "env_key"


@patch.dict("sys.modules", {"supabase": MagicMock()})
def test_load_all_columns():
    import sys

    mock_create_client = sys.modules["supabase"].create_client

    mock_client = MagicMock()
    mock_create_client.return_value = mock_client
    mock_select = mock_client.table.return_value.select.return_value
    mock_execute = mock_select.execute.return_value
    mock_execute.data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

    loader = SupabaseLoader("users", supabase_url="http://url", supabase_key="key")
    docs = loader.load()

    mock_client.table.assert_called_with("users")
    mock_client.table.return_value.select.assert_called_with("*")
    mock_select.execute.assert_called_once()

    assert len(docs) == 2
    assert docs[0].text == "id: 1\nname: Alice"
    assert docs[0].metadata["table"] == "users"
    assert docs[0].metadata["row"] == 0
    assert docs[1].text == "id: 2\nname: Bob"


@patch.dict("sys.modules", {"supabase": MagicMock()})
def test_load_text_columns_only():
    import sys

    mock_create_client = sys.modules["supabase"].create_client

    mock_client = MagicMock()
    mock_create_client.return_value = mock_client
    mock_execute = mock_client.table.return_value.select.return_value.execute.return_value
    mock_execute.data = [{"id": 1, "title": "Doc1", "content": "Hello"}]

    loader = SupabaseLoader(
        "docs", supabase_url="http://url", supabase_key="key", text_columns=["title", "content"]
    )
    docs = loader.load()

    mock_client.table.return_value.select.assert_called_with("title,content")

    assert len(docs) == 1
    assert docs[0].text == "title: Doc1\ncontent: Hello"
    assert docs[0].metadata["id"] == 1
    assert docs[0].metadata["table"] == "docs"


@patch.dict("sys.modules", {"supabase": MagicMock()})
def test_load_text_and_metadata_columns():
    import sys

    mock_create_client = sys.modules["supabase"].create_client

    mock_client = MagicMock()
    mock_create_client.return_value = mock_client
    mock_execute = mock_client.table.return_value.select.return_value.execute.return_value
    mock_execute.data = [{"id": 1, "title": "Doc1", "content": "Hello", "author": "Alice"}]

    loader = SupabaseLoader(
        "docs",
        supabase_url="http://url",
        supabase_key="key",
        text_columns=["content"],
        metadata_columns=["id", "author"],
    )
    docs = loader.load()

    mock_client.table.return_value.select.assert_called_with("content,id,author")

    assert len(docs) == 1
    assert docs[0].text == "content: Hello"
    assert docs[0].metadata["id"] == 1
    assert docs[0].metadata["author"] == "Alice"
    assert "title" not in docs[0].metadata
