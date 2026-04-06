from __future__ import annotations

from unittest.mock import patch

import pytest

from synapsekit.agents.tools import StripeTool


@pytest.mark.asyncio
async def test_requires_api_key() -> None:
    tool = StripeTool()
    with patch.dict("os.environ", {}, clear=True):
        res = await tool.run(action="list_products")
    assert "STRIPE_API_KEY is not set" in (res.error or "")


@pytest.mark.asyncio
async def test_get_customer_by_id() -> None:
    tool = StripeTool()
    fake = {"id": "cus_123", "email": "a@b.com", "name": "Alice", "deleted": False}
    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test"}):
        with patch.object(tool, "_request", return_value=fake):
            res = await tool.get_customer("cus_123")
    assert res.error is None
    assert "Customer cus_123" in res.output
    assert "Status: active" in res.output


@pytest.mark.asyncio
async def test_get_customer_by_email() -> None:
    tool = StripeTool()
    fake = {"data": [{"id": "cus_999", "email": "alice@example.com", "name": "Alice"}]}
    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test"}):
        with patch.object(tool, "_request", return_value=fake):
            res = await tool.get_customer("alice@example.com")
    assert res.error is None
    assert "cus_999" in res.output


@pytest.mark.asyncio
async def test_list_invoices() -> None:
    tool = StripeTool()
    fake = {
        "data": [
            {"id": "in_1", "amount_paid": 1200, "currency": "usd", "status": "paid"},
            {"id": "in_2", "amount_paid": 500, "currency": "usd", "status": "open"},
        ]
    }
    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test"}):
        with patch.object(tool, "_request", return_value=fake):
            res = await tool.list_invoices("cus_123", limit=2)
    assert res.error is None
    assert "Invoices for cus_123" in res.output
    assert "in_1" in res.output


@pytest.mark.asyncio
async def test_get_charge() -> None:
    tool = StripeTool()
    fake = {
        "id": "ch_123",
        "amount": 2500,
        "currency": "usd",
        "paid": True,
        "status": "succeeded",
        "customer": "cus_123",
    }
    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test"}):
        with patch.object(tool, "_request", return_value=fake):
            res = await tool.get_charge("ch_123")
    assert res.error is None
    assert "Charge ch_123" in res.output
    assert "25.00 USD" in res.output


@pytest.mark.asyncio
async def test_list_products() -> None:
    tool = StripeTool()

    def side_effect(endpoint: str, params=None):
        if endpoint == "products":
            return {"data": [{"id": "prod_1", "name": "Pro Plan"}]}
        if endpoint == "prices":
            return {
                "data": [
                    {
                        "id": "price_1",
                        "product": "prod_1",
                        "unit_amount": 990,
                        "currency": "usd",
                        "recurring": {"interval": "month"},
                    }
                ]
            }
        return {"data": []}

    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test"}):
        with patch.object(tool, "_request", side_effect=side_effect):
            res = await tool.list_products(limit=5)
    assert res.error is None
    assert "Pro Plan" in res.output
    assert "9.90 USD/month" in res.output


@pytest.mark.asyncio
async def test_run_unknown_action() -> None:
    tool = StripeTool()
    with patch.dict("os.environ", {"STRIPE_API_KEY": "sk_test"}):
        res = await tool.run(action="boom")
    assert "Unknown action" in (res.error or "")


@pytest.mark.asyncio
async def test_top_level_export() -> None:
    from synapsekit import StripeTool as TopStripeTool

    assert TopStripeTool is StripeTool
