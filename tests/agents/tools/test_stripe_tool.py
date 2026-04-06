from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from synapsekit.agents.tools.stripe import StripeTool


@pytest.mark.asyncio
async def test_requires_api_key() -> None:
    tool = StripeTool()
    with patch.dict("os.environ", {}, clear=True):
        res = await tool.run(action="list_products")
    assert res.is_error
    assert "STRIPE_API_KEY is not set" in res.error


@pytest.mark.asyncio
async def test_get_customer_by_id() -> None:
    tool = StripeTool()
    fake_customer = {
        "id": "cus_abc123",
        "email": "test@example.com",
        "name": "Test User",
        "deleted": False,
    }
    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test_key"}):
        with patch.object(tool, "_request", new=AsyncMock(return_value=fake_customer)):
            res = await tool.get_customer("cus_abc123")

    assert res.error is None
    assert "cus_abc123" in res.output
    assert "test@example.com" in res.output
    assert "Test User" in res.output
    assert "active" in res.output


@pytest.mark.asyncio
async def test_get_customer_by_email() -> None:
    tool = StripeTool()
    fake_search = {
        "data": [
            {
                "id": "cus_xyz789",
                "email": "user@example.com",
                "name": "Email User",
                "deleted": False,
            }
        ]
    }
    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test_key"}):
        with patch.object(tool, "_request", new=AsyncMock(return_value=fake_search)):
            res = await tool.get_customer("user@example.com")

    assert res.error is None
    assert "cus_xyz789" in res.output
    assert "user@example.com" in res.output


@pytest.mark.asyncio
async def test_list_invoices() -> None:
    tool = StripeTool()
    fake_invoices = {
        "data": [
            {
                "id": "in_001",
                "amount_paid": 2000,
                "currency": "usd",
                "status": "paid",
            },
            {
                "id": "in_002",
                "amount_paid": 5000,
                "currency": "usd",
                "status": "open",
            },
        ]
    }
    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test_key"}):
        with patch.object(tool, "_request", new=AsyncMock(return_value=fake_invoices)):
            res = await tool.list_invoices(customer_id="cus_abc123")

    assert res.error is None
    assert "in_001" in res.output
    assert "20.00 USD" in res.output
    assert "paid" in res.output
    assert "in_002" in res.output
    assert "50.00 USD" in res.output


@pytest.mark.asyncio
async def test_get_charge() -> None:
    tool = StripeTool()
    fake_charge = {
        "id": "ch_001",
        "amount": 3500,
        "currency": "usd",
        "paid": True,
        "status": "succeeded",
        "customer": "cus_abc123",
    }
    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test_key"}):
        with patch.object(tool, "_request", new=AsyncMock(return_value=fake_charge)):
            res = await tool.get_charge(charge_id="ch_001")

    assert res.error is None
    assert "ch_001" in res.output
    assert "35.00 USD" in res.output
    assert "True" in res.output
    assert "succeeded" in res.output


@pytest.mark.asyncio
async def test_list_products() -> None:
    tool = StripeTool()
    fake_products = {
        "data": [
            {"id": "prod_001", "name": "Basic Plan"},
            {"id": "prod_002", "name": "Pro Plan"},
        ]
    }
    fake_prices = {
        "data": [
            {
                "id": "price_001",
                "product": "prod_001",
                "unit_amount": 999,
                "currency": "usd",
                "recurring": {"interval": "month"},
            },
            {
                "id": "price_002",
                "product": "prod_002",
                "unit_amount": 2999,
                "currency": "usd",
                "recurring": {"interval": "month"},
            },
        ]
    }

    side_effects = [fake_products, fake_prices]

    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test_key"}):
        with patch.object(tool, "_request", new=AsyncMock(side_effect=side_effects)):
            res = await tool.list_products()

    assert res.error is None
    assert "Basic Plan" in res.output
    assert "Pro Plan" in res.output
    assert "9.99 USD/month" in res.output
    assert "29.99 USD/month" in res.output


@pytest.mark.asyncio
async def test_run_unknown_action() -> None:
    tool = StripeTool()
    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test_key"}):
        res = await tool.run(action="delete_everything")
    assert res.is_error
    assert "Unknown action" in res.error


@pytest.mark.asyncio
async def test_no_action() -> None:
    tool = StripeTool()
    res = await tool.run(action="")
    assert res.is_error
    assert "No action specified" in res.error
