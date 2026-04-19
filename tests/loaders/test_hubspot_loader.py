from __future__ import annotations

import asyncio
import importlib
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import Document, HubSpotLoader


def test_init_requires_object_type() -> None:
    with pytest.raises(ValueError, match="object_type must be provided"):
        HubSpotLoader(object_type="", access_token="token")


def test_init_requires_object_type_when_none() -> None:
    with pytest.raises(ValueError, match="object_type must be provided"):
        HubSpotLoader(object_type=None, access_token="token")  # type: ignore[arg-type]


def test_init_rejects_unsupported_object_type() -> None:
    with pytest.raises(ValueError, match="object_type must be one of"):
        HubSpotLoader(object_type="companies", access_token="token")


def test_init_requires_access_token_without_client() -> None:
    with pytest.raises(ValueError, match="access_token is required unless client is provided"):
        HubSpotLoader(object_type="contacts")


def test_init_requires_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be greater than 0"):
        HubSpotLoader(object_type="contacts", access_token="token", limit=0)


def test_load_import_error_missing_hubspot_client() -> None:
    with patch.dict("sys.modules", {"hubspot": None}):
        loader = HubSpotLoader(object_type="contacts", access_token="token")
        with pytest.raises(ImportError, match="hubspot-api-client required"):
            loader.load()


@patch.dict("sys.modules", {"hubspot": MagicMock()})
def test_load_contacts_with_credentials_and_default_fields() -> None:
    import sys

    mock_hubspot_cls = sys.modules["hubspot"].HubSpot
    mock_client = MagicMock()
    mock_hubspot_cls.return_value = mock_client

    mock_client.crm.contacts.basic_api.get_page.return_value.results = [
        {
            "id": "1",
            "properties": {
                "firstname": "Ada",
                "lastname": "Lovelace",
                "email": "ada@example.com",
            },
            "archived": False,
        }
    ]

    loader = HubSpotLoader(object_type="contacts", access_token="token")
    docs = loader.load()

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "firstname: Ada" in docs[0].text
    assert "email: ada@example.com" in docs[0].text
    assert docs[0].metadata["source"] == "hubspot"
    assert docs[0].metadata["object_type"] == "contacts"
    assert docs[0].metadata["row"] == 0
    assert docs[0].metadata["id"] == "1"

    mock_hubspot_cls.assert_called_once_with(access_token="token")
    mock_client.crm.contacts.basic_api.get_page.assert_called_once_with(
        limit=100,
        archived=False,
        properties=None,
    )


def test_load_with_injected_client_and_field_selection() -> None:
    mock_client = MagicMock()

    record_obj = MagicMock()
    record_obj.to_dict.return_value = {
        "id": "deal-1",
        "properties": {
            "dealname": "Big Deal",
            "amount": "25000",
            "pipeline": "default",
        },
    }
    mock_client.crm.deals.basic_api.get_page.return_value.results = [record_obj]

    loader = HubSpotLoader(
        object_type="deals",
        client=mock_client,
        text_fields=["dealname", "amount"],
        metadata_fields=["id", "pipeline"],
        limit=25,
    )

    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].text == "dealname: Big Deal\namount: 25000"
    assert docs[0].metadata["source"] == "hubspot"
    assert docs[0].metadata["object_type"] == "deals"
    assert docs[0].metadata["id"] == "deal-1"
    assert docs[0].metadata["pipeline"] == "default"
    assert "amount" not in docs[0].metadata

    mock_client.crm.deals.basic_api.get_page.assert_called_once_with(
        limit=25,
        archived=False,
        properties=["dealname", "amount", "id", "pipeline"],
    )


def test_aload() -> None:
    mock_client = MagicMock()
    mock_client.crm.tickets.basic_api.get_page.return_value.results = [
        {
            "id": "ticket-1",
            "properties": {
                "subject": "Reset password",
                "content": "User cannot sign in",
            },
        }
    ]

    loader = HubSpotLoader(
        object_type="tickets",
        client=mock_client,
        text_fields=["subject", "content"],
    )

    docs = asyncio.run(loader.aload())

    assert len(docs) == 1
    assert docs[0].text == "subject: Reset password\ncontent: User cannot sign in"


def test_load_skips_non_mapping_records() -> None:
    mock_client = MagicMock()
    broken_record = MagicMock()
    broken_record.to_dict.return_value = "not-a-mapping"

    mock_client.crm.contacts.basic_api.get_page.return_value.results = [
        "bad",
        broken_record,
        {"id": "2", "properties": {"firstname": "Grace"}},
    ]

    loader = HubSpotLoader(
        object_type="contacts",
        client=mock_client,
        text_fields=["firstname"],
    )

    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].metadata["id"] == "2"


def test_load_handles_none_results() -> None:
    mock_client = MagicMock()
    mock_client.crm.contacts.basic_api.get_page.return_value.results = None

    loader = HubSpotLoader(object_type="contacts", client=mock_client)
    docs = loader.load()

    assert docs == []


def test_exports_from_synapsekit_and_loaders_modules() -> None:
    import synapsekit
    from synapsekit.loaders import __all__ as loaders_all

    loaders = importlib.import_module("synapsekit.loaders")

    assert "HubSpotLoader" in synapsekit.__all__
    assert "HubSpotLoader" in loaders_all
    assert "HubSpotLoader" in loaders.__all__
    assert hasattr(synapsekit, "HubSpotLoader")
    assert hasattr(loaders, "HubSpotLoader")
