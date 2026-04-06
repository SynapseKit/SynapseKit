"""StripeTool: read-only Stripe data lookup."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..base import BaseTool, ToolResult


class StripeTool(BaseTool):
    """Read-only Stripe data lookup tool.

    Auth via ``STRIPE_API_KEY`` env var. Uses stdlib ``urllib`` only —
    no extra dependencies. All operations are read-only.
    """

    name = "stripe"
    description = (
        "Query Stripe data using read-only operations. "
        "Actions: get_customer, list_invoices, get_charge, list_products."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get_customer", "list_invoices", "get_charge", "list_products"],
                "description": "Stripe action to perform",
            },
            "id_or_email": {
                "type": "string",
                "description": "Customer ID (cus_...) or email for get_customer",
            },
            "customer_id": {
                "type": "string",
                "description": "Stripe customer ID for list_invoices",
            },
            "charge_id": {
                "type": "string",
                "description": "Stripe charge ID for get_charge",
            },
            "limit": {
                "type": "integer",
                "description": "Max results for list endpoints (default: 10, max: 100)",
                "default": 10,
            },
        },
        "required": ["action"],
    }

    _base_url = "https://api.stripe.com/v1"

    def _get_api_key(self) -> str | None:
        return os.getenv("STRIPE_API_KEY")

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("STRIPE_API_KEY is not set")

        query = f"?{urlencode(params)}" if params else ""
        url = f"{self._base_url}/{endpoint}{query}"
        req = Request(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "SynapseKit/StripeTool",
            },
        )
        loop = asyncio.get_running_loop()

        def _fetch() -> dict[str, Any]:
            with urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Unexpected Stripe API response format")
            return cast(dict[str, Any], payload)

        return await loop.run_in_executor(None, _fetch)

    async def get_customer(self, id_or_email: str) -> ToolResult:
        if not id_or_email:
            return ToolResult(output="", error="id_or_email is required")

        try:
            if id_or_email.startswith("cus_"):
                customer = await self._request(f"customers/{id_or_email}")
                status = "active" if not customer.get("deleted", False) else "deleted"
                return ToolResult(
                    output=(
                        f"Customer {customer.get('id', 'unknown')}\n"
                        f"- Email: {customer.get('email', 'N/A')}\n"
                        f"- Name: {customer.get('name', 'N/A')}\n"
                        f"- Status: {status}"
                    )
                )

            matches = await self._request(
                "customers/search", {"query": f"email:'{id_or_email}'", "limit": 1}
            )
            rows = matches.get("data", [])
            if not rows:
                return ToolResult(output=f"No customer found for email: {id_or_email}")
            customer = rows[0]
            return ToolResult(
                output=(
                    f"Customer {customer.get('id', 'unknown')}\n"
                    f"- Email: {customer.get('email', 'N/A')}\n"
                    f"- Name: {customer.get('name', 'N/A')}\n"
                    f"- Status: {'active' if not customer.get('deleted', False) else 'deleted'}"
                )
            )
        except HTTPError as e:
            return ToolResult(output="", error=f"Stripe API HTTP error: {e.code}")
        except URLError as e:
            return ToolResult(output="", error=f"Stripe API connection error: {e.reason}")
        except Exception as e:
            return ToolResult(output="", error=f"Stripe get_customer failed: {e}")

    async def list_invoices(self, customer_id: str, limit: int = 10) -> ToolResult:
        if not customer_id:
            return ToolResult(output="", error="customer_id is required")

        limit = max(1, min(limit, 100))
        try:
            data = await self._request("invoices", {"customer": customer_id, "limit": limit})
            invoices = data.get("data", [])
            if not invoices:
                return ToolResult(output=f"No invoices found for customer {customer_id}.")

            lines = [f"Invoices for {customer_id} (showing {len(invoices)}):"]
            for inv in invoices:
                amount = inv.get("amount_paid", 0) / 100
                currency = str(inv.get("currency", "")).upper() or "USD"
                status = inv.get("status", "unknown")
                inv_id = inv.get("id", "unknown")
                lines.append(f"- {inv_id}: {amount:.2f} {currency} ({status})")
            return ToolResult(output="\n".join(lines))
        except HTTPError as e:
            return ToolResult(output="", error=f"Stripe API HTTP error: {e.code}")
        except URLError as e:
            return ToolResult(output="", error=f"Stripe API connection error: {e.reason}")
        except Exception as e:
            return ToolResult(output="", error=f"Stripe list_invoices failed: {e}")

    async def get_charge(self, charge_id: str) -> ToolResult:
        if not charge_id:
            return ToolResult(output="", error="charge_id is required")

        try:
            charge = await self._request(f"charges/{charge_id}")
            amount = charge.get("amount", 0) / 100
            currency = str(charge.get("currency", "")).upper() or "USD"
            paid = bool(charge.get("paid", False))
            status = charge.get("status", "unknown")
            customer = charge.get("customer", "N/A")
            return ToolResult(
                output=(
                    f"Charge {charge.get('id', charge_id)}\n"
                    f"- Amount: {amount:.2f} {currency}\n"
                    f"- Paid: {paid}\n"
                    f"- Status: {status}\n"
                    f"- Customer: {customer}"
                )
            )
        except HTTPError as e:
            return ToolResult(output="", error=f"Stripe API HTTP error: {e.code}")
        except URLError as e:
            return ToolResult(output="", error=f"Stripe API connection error: {e.reason}")
        except Exception as e:
            return ToolResult(output="", error=f"Stripe get_charge failed: {e}")

    async def list_products(self, limit: int = 10) -> ToolResult:
        limit = max(1, min(limit, 100))
        try:
            products_data = await self._request("products", {"active": "true", "limit": limit})
            prices_data = await self._request("prices", {"active": "true", "limit": limit})

            products = products_data.get("data", [])
            prices = prices_data.get("data", [])
            if not products:
                return ToolResult(output="No active products found.")

            price_by_product: dict[str, list[dict[str, Any]]] = {}
            for p in prices:
                product = p.get("product")
                if isinstance(product, str):
                    price_by_product.setdefault(product, []).append(p)

            lines = [f"Products (showing {len(products)}):"]
            for prod in products:
                pid = prod.get("id", "unknown")
                pname = prod.get("name", "Unnamed")
                prod_prices = price_by_product.get(pid, [])
                if not prod_prices:
                    lines.append(f"- {pname} ({pid}) — no active prices")
                    continue

                price_parts = []
                for pr in prod_prices:
                    unit = pr.get("unit_amount")
                    currency = str(pr.get("currency", "")).upper() or "USD"
                    recurring = pr.get("recurring", {})
                    interval = recurring.get("interval") if isinstance(recurring, dict) else None
                    amount_text = (
                        f"{unit / 100:.2f} {currency}"
                        if isinstance(unit, int)
                        else f"N/A {currency}"
                    )
                    if interval:
                        amount_text += f"/{interval}"
                    price_parts.append(amount_text)

                lines.append(f"- {pname} ({pid}) — {', '.join(price_parts)}")

            return ToolResult(output="\n".join(lines))
        except HTTPError as e:
            return ToolResult(output="", error=f"Stripe API HTTP error: {e.code}")
        except URLError as e:
            return ToolResult(output="", error=f"Stripe API connection error: {e.reason}")
        except Exception as e:
            return ToolResult(output="", error=f"Stripe list_products failed: {e}")

    async def run(
        self,
        action: str = "",
        id_or_email: str = "",
        customer_id: str = "",
        charge_id: str = "",
        limit: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        if not action:
            return ToolResult(output="", error="No action specified.")

        if action == "get_customer":
            return await self.get_customer(id_or_email or kwargs.get("input", ""))
        if action == "list_invoices":
            return await self.list_invoices(customer_id=customer_id, limit=limit)
        if action == "get_charge":
            return await self.get_charge(charge_id=charge_id)
        if action == "list_products":
            return await self.list_products(limit=limit)

        return ToolResult(
            output="",
            error=f"Unknown action: {action}. Must be one of: get_customer, list_invoices, get_charge, list_products",
        )
