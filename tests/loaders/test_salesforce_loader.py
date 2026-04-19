from __future__ import annotations

import asyncio
import importlib
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import Document, SalesforceLoader


def test_init_requires_soql_query() -> None:
    with pytest.raises(ValueError, match="soql_query must be provided"):
        SalesforceLoader(
            soql_query="",
            username="user@example.com",
            password="secret",
            security_token="token",
        )


def test_init_requires_credentials_without_client() -> None:
    with pytest.raises(ValueError, match="username, password, and security_token are required"):
        SalesforceLoader(soql_query="SELECT Id FROM Account")


def test_load_import_error_missing_simple_salesforce() -> None:
    with patch.dict("sys.modules", {"simple_salesforce": None}):
        loader = SalesforceLoader(
            soql_query="SELECT Id FROM Account",
            username="user@example.com",
            password="secret",
            security_token="token",
        )
        with pytest.raises(ImportError, match="simple-salesforce required"):
            loader.load()


@patch.dict("sys.modules", {"simple_salesforce": MagicMock()})
def test_load_with_credentials_and_default_text_metadata() -> None:
    import sys

    mock_salesforce_cls = sys.modules["simple_salesforce"].Salesforce
    mock_client = MagicMock()
    mock_salesforce_cls.return_value = mock_client

    query = "SELECT Id, Name, Industry FROM Account"
    mock_client.query_all.return_value = {
        "records": [
            {
                "attributes": {
                    "type": "Account",
                    "url": "/services/data/v60.0/sobjects/Account/001",
                },
                "Id": "001",
                "Name": "Acme",
                "Industry": "Technology",
            }
        ]
    }

    loader = SalesforceLoader(
        soql_query=query,
        username="user@example.com",
        password="secret",
        security_token="token",
    )
    docs = loader.load()

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Id: 001" in docs[0].text
    assert "Name: Acme" in docs[0].text
    assert docs[0].metadata["source"] == "salesforce"
    assert docs[0].metadata["query"] == query
    assert docs[0].metadata["row"] == 0
    assert docs[0].metadata["object"] == "Account"
    assert docs[0].metadata["Id"] == "001"

    mock_salesforce_cls.assert_called_once_with(
        username="user@example.com",
        password="secret",
        security_token="token",
        domain="login",
    )
    mock_client.query_all.assert_called_once_with(query)


def test_load_with_injected_client_and_field_selection() -> None:
    mock_client = MagicMock()
    mock_client.query_all.return_value = {
        "records": [
            {
                "attributes": {"type": "Lead"},
                "Id": "00Q1",
                "Name": "Alex",
                "Description": None,
                "OwnerId": "005X",
                "Company": "Acme",
            }
        ]
    }

    loader = SalesforceLoader(
        soql_query="SELECT Id, Name, Description, OwnerId FROM Lead",
        client=mock_client,
        text_fields=["Name", "Description"],
        metadata_fields=["Id", "OwnerId"],
    )

    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "Name: Alex\nDescription: "
    assert docs[0].metadata["source"] == "salesforce"
    assert docs[0].metadata["object"] == "Lead"
    assert docs[0].metadata["Id"] == "00Q1"
    assert docs[0].metadata["OwnerId"] == "005X"
    assert "Company" not in docs[0].metadata


def test_aload() -> None:
    mock_client = MagicMock()
    mock_client.query_all.return_value = {
        "records": [{"attributes": {"type": "Contact"}, "Id": "003A", "Name": "Taylor"}]
    }

    loader = SalesforceLoader(
        soql_query="SELECT Id, Name FROM Contact",
        client=mock_client,
        text_fields=["Name"],
    )

    docs = asyncio.run(loader.aload())
    assert len(docs) == 1
    assert docs[0].text == "Name: Taylor"


def test_load_skips_non_mapping_records() -> None:
    mock_client = MagicMock()
    mock_client.query_all.return_value = {
        "records": [
            "not-a-record",
            {"attributes": {"type": "Contact"}, "Id": "003A", "Name": "Taylor"},
        ]
    }

    loader = SalesforceLoader(
        soql_query="SELECT Id, Name FROM Contact",
        client=mock_client,
        text_fields=["Name"],
    )

    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].metadata["object"] == "Contact"


def test_exports_from_synapsekit_and_loaders_modules() -> None:
    import synapsekit
    from synapsekit.loaders import __all__ as loaders_all

    loaders = importlib.import_module("synapsekit.loaders")

    assert "SalesforceLoader" in synapsekit.__all__
    assert "SalesforceLoader" in loaders_all
    assert "SalesforceLoader" in loaders.__all__
    assert hasattr(synapsekit, "SalesforceLoader")
    assert hasattr(loaders, "SalesforceLoader")
